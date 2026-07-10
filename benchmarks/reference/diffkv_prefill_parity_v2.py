#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline tuned-vs-stock PREFILL parity + bench harness for the DiffKV
Triton kernel (A1: VLLM_DIFFKV_PREFILL_TUNE, A2: DIFFKV_TILE64_SAFE).

WHY: A1 re-tiles the 2D prefill launch (BLOCK_M 16->32/64, BLOCK_Q 1->2/4,
num_warps/num_stages explicit, optionally TILE_SIZE 32->64). On sm_121 (GB10)
Triton has a documented 3x history of SILENT GARBAGE from miscompiles, and the
tuned constexpr combos (BLOCK_M=32/64 at HEAD_SIZE_QK_PADDED=256) have NEVER
been compiled on this GPU. Before the env var may be set on a serving boot,
this harness must prove — numerically, on the target GPU, for the EXACT
constexpr set that will ship (incl. the SLIDING_WINDOW=129 SWA specialization,
the fp8 IS_FP8 arm, and mixed prefill+decode batches) — that the tuned launch
equals the stock launch of the SAME kernel on IDENTICAL inputs.

=== TILE64-SAFE v2 additions (A2) ===================================
The kernel's SPLIT_TILE constexpr (mods patch_tile64_safe.py) keeps TILE=64
on the memory side but replays the stock 32-wide compute sequence twice per
tile in stock order -> claimed BIT-IDENTICAL to stock.  Therefore:
  * tune token "safe64" == "32,4,2,64" and any tile=64 tune launches with
    SPLIT_TILE=True (mirrors the prod wrapper, where raw joint-64 compute is
    no longer reachable from the env).
  * PASS bar for split tunes is EXACT: tun-vs-stk abs err MUST be 0.0 (no
    tolerance).  Anything else means the split is not order-preserving ->
    REJECT the combo, do not ship.  A "bit%" diagnostic column reports the
    fraction of bf16 outputs whose BIT PATTERN differs (0.0000 expected;
    a nonzero bit% at abs 0.0 can only be -0.0 vs +0.0 — value-neutral,
    logged for the record).
  * SWA cases under a split tune mirror the prod wrapper downgrade: they run
    tile=32/no-split (annotated "swa>32"); the 0.0 bar still applies (that
    IS the path SWA layers take in prod under safe64).
  * Diagnostic token "BM,W,S,64,raw" launches the KNOWN-BAD raw joint-64
    path (expected FAIL ~3-7e-3; use only with --bench to measure the raw
    ceiling; a raw tune failing parity correctly exits nonzero — never ship).
  * --sinks adds a random per-head sink to BOTH launches and to the torch
    reference (USE_SINKS=True arm; prod MiMo runs with sinks).
  * --sweep safe64 == "32,4,2,64;32,8,2,64" (both split).
=====================================================================

REFERENCE CHAIN:
  * load-bearing: tuned-vs-stock (stock 2D prefill = months-validated prod).
  * backstop:     stock-vs-torch SDPA (catches a harness bug / systematic
                  stock error hiding behind "tuned==stock"); auto-skipped for
                  seq_len > --torch-max (memory).

FAILURE TAXONOMY (what a result means):
  * numeric FAIL / NaN / big-error-fraction > 0  -> the sm_121 silent-garbage
    class. DO NOT SHIP that combo. This is the ONLY gate that catches it.
  * split tune with 0.0 < abs err <= tol -> NOT silent garbage but the split
    is NOT order-preserving: the A2 bit-identity claim is false. REJECT.
  * Python exception from the tuned launch (e.g. triton OutOfResources:
    shared memory) -> LOUD compile failure. Safe (cannot silently corrupt),
    just skip the combo.

RUN (inside a vLLM container on an IDLE GPU — never next to a live boot):
    docker cp diffkv_prefill_parity.py <node-container>:/root/
    docker exec -e CUDA_VISIBLE_DEVICES=0 <node-container> \
        python3 /root/diffkv_prefill_parity.py --vs-torch
    # A2 ship gate (kernel must already carry the tile64-safe body patch):
    docker exec -e CUDA_VISIBLE_DEVICES=0 <node-container> \
        python3 /root/diffkv_prefill_parity.py --tune safe64 --vs-torch
    docker exec -e CUDA_VISIBLE_DEVICES=0 <node-container> \
        python3 /root/diffkv_prefill_parity.py --tune safe64 --sinks
    # sweep mode (parity + kernel-level timing for every combo):
    docker exec -e CUDA_VISIBLE_DEVICES=0 <node-container> \
        python3 /root/diffkv_prefill_parity.py --sweep all --bench
  Run on BOTH nodes: each TP rank compiles its own Triton binaries.

Exit 0 iff every attempted case PASSes (loud compile-fails don't fail the run
unless --strict).
"""
import argparse
import math
import sys

import torch

from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    kernel_unified_attention_diffkv,
)
from vllm.triton_utils import triton

# ---- MiMo-V2.5 geometry per rank under TP=2 (full-attn + SWA layers) ----
HEAD_SIZE_QK = 192
HEAD_SIZE_V = 128
NUM_KV_HEADS = 2            # num_key_value_heads 4 / TP 2
NUM_QUERIES_PER_KV = 16     # 32 q-heads / 2 kv-heads
NUM_QUERY_HEADS = NUM_KV_HEADS * NUM_QUERIES_PER_KV
BLOCK_SIZE = 32             # recipe --block-size 32
MIMO_SWA_WINDOW = 128       # SWA layers -> SLIDING_WINDOW constexpr 129
TOL = 2e-3
BIG_ERR = 0.05              # miscompiles produce O(1) errors on many elements

FP8_DTYPE = torch.float8_e4m3fn

# (name, q_lens, ctx_lens, window) — ctx is the already-cached prefix, so
# seq_len = ctx + q_len (chunked-prefill semantics). Mixed = 1 prefill chunk
# + 2 decode + 1 DFlash-verify row-block sharing ONE launch (the shape class
# prod actually runs at mns 8 / mnbt 8192).
# NOTE for tile=64 tunes: several cases (first512: 512 %64==0 vs deep400k:
# 398336+2048 etc.) intentionally cover max_seq_prefix_len % 64 in both
# halves-populated and trailing-half-masked regimes; first64 covers the
# single-tile case.
CASES = [
    ("first64",   [64],            [0],                      -1),
    ("first512",  [512],           [0],                      -1),
    ("first4096", [4096],          [0],                      -1),
    ("deep50k",   [4096],          [50000 - 4096],           -1),
    ("deep200k",  [4096],          [200000 - 4096],          -1),
    ("deep400k",  [2048],          [400000 - 2048],          -1),
    ("swa512",    [512],           [0],                      MIMO_SWA_WINDOW),
    ("swa50k",    [4096],          [50000 - 4096],           MIMO_SWA_WINDOW),
    ("mixed",     [4096, 1, 1, 8], [50000 - 4096, 120000 - 1,
                                    120000 - 1, 120000 - 8], -1),
]


def _kernel_params():
    fn = kernel_unified_attention_diffkv
    names = getattr(fn, "arg_names", None)
    if names is None:
        names = [p.name for p in fn.params]
    return set(names)


KP = _kernel_params()
HAS_FP8_ARM = "IS_FP8" in KP        # mods/fp8-kv-inline applied
HAS_NVFP4_ARM = "IS_NVFP4" in KP    # host-fork / nvfp4-kv-upstream kernel
HAS_SPLIT_ARM = "SPLIT_TILE" in KP  # mods tile64-safe body patch applied


def build_case(q_lens, ctx_lens, device, kv_fp8: bool, seed: int,
               dtype=torch.bfloat16, sinks_on: bool = False):
    g = torch.Generator(device=device).manual_seed(seed)
    num_seqs = len(q_lens)
    seq_lens_py = [c + q for c, q in zip(ctx_lens, q_lens)]
    n_tokens = sum(q_lens)

    q = torch.randn(n_tokens, NUM_QUERY_HEADS, HEAD_SIZE_QK, device=device,
                    dtype=dtype, generator=g) * 0.5

    bps = [(s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens_py]
    n_blocks = sum(bps) + 8
    k_bf = torch.randn(n_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE_QK,
                       device=device, dtype=dtype, generator=g) * 0.5
    v_bf = torch.randn(n_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE_V,
                       device=device, dtype=dtype, generator=g) * 0.5

    if kv_fp8:
        k_descale = v_descale = 0.5   # non-trivial descale exercises the mul
        kv = torch.empty(n_blocks, BLOCK_SIZE, NUM_KV_HEADS,
                         HEAD_SIZE_QK + HEAD_SIZE_V, device=device,
                         dtype=torch.uint8)
        kv[..., :HEAD_SIZE_QK] = (k_bf.float() / k_descale).to(
            FP8_DTYPE).view(torch.uint8)
        kv[..., HEAD_SIZE_QK:] = (v_bf.float() / v_descale).to(
            FP8_DTYPE).view(torch.uint8)
    else:
        k_descale = v_descale = 1.0
        kv = torch.empty(n_blocks, BLOCK_SIZE, NUM_KV_HEADS,
                         HEAD_SIZE_QK + HEAD_SIZE_V, device=device,
                         dtype=dtype)
        kv[..., :HEAD_SIZE_QK] = k_bf
        kv[..., HEAD_SIZE_QK:] = v_bf
    key_cache = kv[..., :HEAD_SIZE_QK]
    value_cache = kv[..., HEAD_SIZE_QK:HEAD_SIZE_QK + HEAD_SIZE_V]

    # Block table padded by +2 columns (valid block 0): with TILE_SIZE 64 the
    # unmasked block-index gather can read ceil(TILE/BLOCK_SIZE)-1 entries past
    # a row's last block (vLLM's real tables are mml-padded; ours must be too).
    max_bps = max(bps)
    block_table = torch.zeros(num_seqs, max_bps + 2, device=device,
                              dtype=torch.int32)
    perm = torch.randperm(n_blocks, generator=g, device=device)
    off = 0
    for i, nb in enumerate(bps):
        block_table[i, :nb] = perm[off:off + nb].to(torch.int32)
        off += nb

    seq_lens = torch.tensor(seq_lens_py, device=device, dtype=torch.int32)
    cu = torch.zeros(num_seqs + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(q_lens, device=device), 0)

    # Per-head attention sinks (drawn LAST so all other tensors are seed-
    # identical with/without --sinks). Same tensor feeds stock, tuned AND
    # the torch reference.
    sinks = None
    if sinks_on:
        sinks = torch.randn(NUM_QUERY_HEADS, device=device,
                            dtype=torch.float32, generator=g) * 0.5

    return dict(q=q, key_cache=key_cache, value_cache=value_cache,
                block_table=block_table, seq_lens=seq_lens, cu_seqlens_q=cu,
                num_seqs=num_seqs, n_tokens=n_tokens, q_lens=q_lens,
                seq_lens_py=seq_lens_py, k_bf=k_bf, v_bf=v_bf,
                k_descale=k_descale, v_descale=v_descale, kv_fp8=kv_fp8,
                sinks=sinks)


def _launch(case, tune, window, device, out=None):
    """One 2D launch. tune=None -> the stock wrapper params (BLOCK_M=16,
    BLOCK_Q=1, TILE=32, Triton default warps/stages, no split).
    tune=(bm,w,s,t,split) -> exactly what the A1+A2 gates would set.
    A split tune on a SWA case mirrors the prod wrapper downgrade:
    tile 32 / no split (the certified A1 path)."""
    q = case["q"]
    k, v = case["key_cache"], case["value_cache"]
    if out is None:
        out = torch.empty(case["n_tokens"], NUM_QUERY_HEADS, HEAD_SIZE_V,
                          device=device, dtype=q.dtype)
    if tune is None:
        BLOCK_M, tile_size, lk, split = 16, 32, {}, False
    else:
        bm, w, s, t, split = tune
        if split and window >= 0:
            # prod wrapper behavior: SWA layers never take the split-64
            # path (its softmax_step M-guard is order-unsafe under SWA
            # tile pruning) -> certified A1 tile-32 launch instead.
            t, split = 32, False
        BLOCK_M, tile_size = bm, t
        lk = {"num_warps": w, "num_stages": s}
    BLOCK_Q = BLOCK_M // NUM_QUERIES_PER_KV
    total_q_blocks = q.shape[0] // BLOCK_Q + case["num_seqs"]
    grid = (total_q_blocks, NUM_KV_HEADS)
    sliding_window_val = 1 + window if window >= 0 else 0

    kw = dict(
        output_ptr=out, segm_output_ptr=out, segm_max_ptr=out,
        segm_expsum_ptr=out,        # 2D never reads segm ptrs
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v,
        sink_ptr=case["sinks"],
        block_tables_ptr=case["block_table"], seq_lens_ptr=case["seq_lens"],
        alibi_slopes_ptr=None, scale=1.0 / math.sqrt(HEAD_SIZE_QK),
        softcap=0.0, num_query_heads=NUM_QUERY_HEADS,
        num_queries_per_kv=NUM_QUERIES_PER_KV,
        block_table_stride=case["block_table"].stride(0),
        query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        output_stride_0=out.stride(0), output_stride_1=out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE, TILE_SIZE=tile_size,
        HEAD_SIZE_QK=HEAD_SIZE_QK,
        HEAD_SIZE_QK_PADDED=triton.next_power_of_2(HEAD_SIZE_QK),
        HEAD_SIZE_V=HEAD_SIZE_V,
        HEAD_SIZE_V_PADDED=triton.next_power_of_2(HEAD_SIZE_V),
        USE_ALIBI_SLOPES=False, USE_ALIBI_SQRT=False, USE_SOFTCAP=False,
        USE_SINKS=(case["sinks"] is not None),
        SLIDING_WINDOW=sliding_window_val,
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        query_start_len_ptr=case["cu_seqlens_q"], BLOCK_Q=BLOCK_Q,
        num_seqs=case["num_seqs"], BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=1, IS_3D=False,
    )
    if case["kv_fp8"]:
        kw.update(IS_FP8=True, K_DESCALE=case["k_descale"],
                  V_DESCALE=case["v_descale"])
    if split:
        kw.update(SPLIT_TILE=True)
    kernel_unified_attention_diffkv[grid](**kw, **lk)
    torch.cuda.synchronize()
    return out


def torch_reference(case, device, window):
    """Per-sequence bf16->fp32 SDPA over the (dequantized) cache.

    Sinks semantics match the kernel: M starts at the per-head sink and L
    starts at 1.0, i.e. one virtual logit == sink per row that contributes
    to the denominator but not the numerator."""
    outs = []
    scale = 1.0 / math.sqrt(HEAD_SIZE_QK)
    tok0 = 0
    for si, (q_len, seq_len) in enumerate(
            zip(case["q_lens"], case["seq_lens_py"])):
        ctx = seq_len - q_len
        bt = case["block_table"][si]
        kc, vc = case["key_cache"], case["value_cache"]
        nb = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        if case["kv_fp8"]:
            kcat = kc[bt[:nb]].view(FP8_DTYPE).float() * case["k_descale"]
            vcat = vc[bt[:nb]].view(FP8_DTYPE).float() * case["v_descale"]
        else:
            kcat = kc[bt[:nb]].float()
            vcat = vc[bt[:nb]].float()
        kcat = kcat.reshape(-1, NUM_KV_HEADS, HEAD_SIZE_QK)[:seq_len]
        vcat = vcat.reshape(-1, NUM_KV_HEADS, HEAD_SIZE_V)[:seq_len]
        q0 = case["q"][tok0:tok0 + q_len].float()
        o = torch.empty(q_len, NUM_QUERY_HEADS, HEAD_SIZE_V, device=device)
        qpos = ctx + torch.arange(q_len, device=device)[:, None]
        kpos = torch.arange(seq_len, device=device)[None, :]
        mask = kpos <= qpos
        if window >= 0:
            mask = mask & ((qpos - kpos) < (window + 1))
        for h in range(NUM_QUERY_HEADS):
            kvh = h // NUM_QUERIES_PER_KV
            S = (q0[:, h, :] @ kcat[:, kvh, :].T) * scale
            S = S.masked_fill(~mask, float("-inf"))
            if case["sinks"] is not None:
                s_h = case["sinks"][h].float()
                m = torch.maximum(S.max(dim=-1).values, s_h)
                P = torch.exp(S - m[:, None])
                denom = P.sum(dim=-1) + torch.exp(s_h - m)
                o[:, h, :] = (P @ vcat[:, kvh, :]) / denom[:, None]
            else:
                o[:, h, :] = torch.softmax(S, -1) @ vcat[:, kvh, :]
        outs.append(o)
        tok0 += q_len
    return torch.cat(outs, 0)


def _errs(a, b):
    a, b = a.float(), b.float()
    abs_e = (a - b).abs().max().item()
    rel_e = abs_e / (b.abs().max().item() + 1e-9)
    nan = bool(torch.isnan(a).any() or torch.isinf(a).any())
    big = ((a - b).abs() > BIG_ERR).float().mean().item()
    return abs_e, rel_e, nan, big


def _bitdiff(a, b):
    """Fraction of bf16 outputs whose raw bit pattern differs (diagnostic:
    0.0 expected for order-preserving split; -0.0 vs +0.0 is the only
    value-neutral way this can be nonzero at abs err 0.0)."""
    return (a.contiguous().view(torch.int16)
            != b.contiguous().view(torch.int16)).float().mean().item()


def _bench(case, tune, window, device, iters=10):
    out = torch.empty(case["n_tokens"], NUM_QUERY_HEADS, HEAD_SIZE_V,
                      device=device, dtype=case["q"].dtype)
    for _ in range(3):
        _launch(case, tune, window, device, out)
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        ev0.record()
        _launch(case, tune, window, device, out)
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    ts.sort()
    return ts[len(ts) // 2]


def parse_tune(s):
    """Tune spec -> (BM, warps, stages, tile, split).

    "1"       -> 32,8,2,32           (legacy A1 alias)
    "safe64"  -> 32,4,2,64 + split   (the A2 prod token)
    "B,W,S,T" -> split iff T == 64   (mirrors the prod wrapper: raw
                                      joint-64 is not env-reachable)
    "B,W,S,T,raw"   -> force raw (diagnostic; T=64 raw = KNOWN-BAD)
    "B,W,S,T,split" -> force split (requires T == 64)
    """
    if s == "1":
        s = "32,8,2,32"
    if s == "safe64":
        s = "32,4,2,64"
    parts = s.split(",")
    force = None
    if len(parts) == 5:
        if parts[4] not in ("split", "raw"):
            raise ValueError(f"5th tune field must be split|raw, got {parts[4]}")
        force = parts[4] == "split"
        parts = parts[:4]
    bm, w, st, t = (int(x) for x in parts)
    split = (t == 64) if force is None else force
    if split and t != 64:
        raise ValueError("split requires tile == 64")
    return (bm, w, st, t, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="32,8,2,32",
                    help='"BM,warps,stages,tile[,split|raw]" or "safe64" — '
                         "must equal the env value that will ship")
    ap.add_argument("--sweep", default=None,
                    help='"all" = BM{32,64}xW{4,8}xS{2,3}xT{32,64}, '
                         '"safe64" = 32,{4,8},2,64 split, or '
                         '";"-separated tune strings')
    ap.add_argument("--bench", action="store_true",
                    help="also time stock vs tuned per case (median of 10)")
    ap.add_argument("--vs-torch", action="store_true")
    ap.add_argument("--torch-max", type=int, default=65536,
                    help="skip the torch backstop above this seq_len")
    ap.add_argument("--max-seq", type=int, default=400000)
    ap.add_argument("--kv", choices=["auto", "fp8", "bf16"], default="auto")
    ap.add_argument("--sinks", action="store_true",
                    help="attach per-head attention sinks to every case "
                         "(USE_SINKS=True arm; prod MiMo runs with sinks)")
    ap.add_argument("--strict", action="store_true",
                    help="loud compile-failures also fail the run")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA. This harness MUST run on the target GPU "
              "(sm_121 Triton compilation is the thing under test).")
        sys.exit(2)
    device = torch.device("cuda")

    kv_fp8 = (args.kv == "fp8") or (args.kv == "auto" and HAS_FP8_ARM)
    if kv_fp8 and not HAS_FP8_ARM:
        print("FATAL: --kv fp8 but kernel has no IS_FP8 arm "
              "(mods/fp8-kv-inline not applied to this container).")
        sys.exit(2)
    print(f"# A1/A2 prefill parity  kernel-arms: fp8={HAS_FP8_ARM} "
          f"nvfp4={HAS_NVFP4_ARM} split={HAS_SPLIT_ARM} "
          f"| kv-mode={'fp8' if kv_fp8 else 'bf16'} "
          f"| sinks={args.sinks} | tol={TOL} (split tunes: EXACT 0.0) "
          f"big-err-thresh={BIG_ERR}  [TILE64-SAFE v2]")

    if args.sweep == "all":
        tunes = [(bm, w, s, t, t == 64) for bm in (32, 64) for w in (4, 8)
                 for s in (2, 3) for t in (32, 64)]
    elif args.sweep == "safe64":
        tunes = [(32, 4, 2, 64, True), (32, 8, 2, 64, True)]
    elif args.sweep:
        tunes = [parse_tune(x) for x in args.sweep.split(";")]
    else:
        tunes = [parse_tune(args.tune)]

    if any(t[4] for t in tunes) and not HAS_SPLIT_ARM:
        print("FATAL: a split tune was requested but the kernel has no "
              "SPLIT_TILE arm (mods tile64-safe body patch not applied to "
              "this container). Refusing to silently fall back to raw-64.")
        sys.exit(2)

    all_pass = True
    for tune in tunes:
        bm, w, s, t, split = tune
        print(f"\n== tune BLOCK_M={bm} warps={w} stages={s} tile={t} "
              f"split={split} (BLOCK_Q={bm // NUM_QUERIES_PER_KV})"
              f"{' [PASS bar: EXACT 0.0]' if split else ''} ==")
        hdr = (f"{'case':>10} {'q/ctx/win':>18} | {'tun-vs-stk rel':>14} "
               f"{'big%':>6} {'bit%':>7} {'PASS':>5}")
        if args.vs_torch:
            hdr += f" | {'stk-vs-torch rel':>16}"
        if args.bench:
            hdr += f" | {'stock ms':>9} {'tuned ms':>9} {'speedup':>7}"
        print(hdr)
        for name, q_lens, ctx_lens, window in CASES:
            if max(c + q for c, q in zip(ctx_lens, q_lens)) > args.max_seq:
                continue
            swa_downgrade = split and window >= 0
            try:
                case = build_case(q_lens, ctx_lens, device, kv_fp8,
                                  args.seed + hash(name) % 9973,
                                  sinks_on=args.sinks)
                stock = _launch(case, None, window, device)
                tuned = _launch(case, tune, window, device)
                abs_e, rel_e, nan, big = _errs(tuned, stock)
                bitp = _bitdiff(tuned, stock)
                if split:
                    # A2 bit-identity bar: the split must be ORDER-
                    # PRESERVING, not merely accurate.  (Applies to the
                    # SWA-downgraded tile-32 launch too: that is the path
                    # prod SWA layers take under safe64.)
                    ok = (abs_e == 0.0 and not nan and big == 0.0)
                else:
                    ok = (abs_e <= TOL and rel_e <= TOL and not nan
                          and big == 0.0)
                tag = name + ("*" if swa_downgrade else "")
                line = (f"{tag:>10} {str(q_lens[:2])[:9]:>9}/"
                        f"{max(case['seq_lens_py']):>6}/{window:>3} | "
                        f"{rel_e:>14.3e} {big * 100:>5.2f}% "
                        f"{bitp * 100:>6.3f}% "
                        f"{'PASS' if ok else 'FAIL':>5}")
                if nan:
                    line += " NaN/Inf!"
                if args.vs_torch and max(case["seq_lens_py"]) <= args.torch_max:
                    ref = torch_reference(case, device, window)
                    _, rt, _, _ = _errs(stock, ref)
                    ok = ok and rt <= TOL
                    line += f" | {rt:>16.3e}"
                elif args.vs_torch:
                    line += f" | {'(skip>torch-max)':>16}"
                if args.bench:
                    ms0 = _bench(case, None, window, device)
                    ms1 = _bench(case, tune, window, device)
                    line += f" | {ms0:>9.2f} {ms1:>9.2f} {ms0 / ms1:>6.2f}x"
                print(line)
                all_pass = all_pass and ok
                del case, stock, tuned
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"{name:>10} OOM (skip; idle GPU / lower --max-seq)")
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 — triton OutOfResources etc.
                print(f"{name:>10} LOUD-FAIL {type(e).__name__}: "
                      f"{str(e)[:120]}  [compile/launch error = SAFE "
                      f"(cannot silently corrupt), combo unusable]")
                if args.strict:
                    all_pass = False
                torch.cuda.empty_cache()
        if any(w >= 0 for _, _, _, w in CASES) and split:
            print("   (* = swa case ran the prod-wrapper downgrade: "
                  "tile 32, no split — the certified A1 path)")

    print("\nRESULT:", "ALL PASS" if all_pass else
          "FAIL — DO NOT set VLLM_DIFFKV_PREFILL_TUNE for a failing combo")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
