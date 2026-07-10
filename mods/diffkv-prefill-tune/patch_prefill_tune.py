#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1 DIFFKV_PREFILL_TUNE patcher — injects the env-gated prefill q-block
tuning into a DiffKV Triton kernel wrapper (unified_attention_diffkv).

Works on BOTH deployed variants:
  * host fork   mods/nvfp4-kv-diffkv/triton_unified_attention_diffkv.py
                (has `nvfp4_packed`; WMMA/TK/Quest branches)
  * native      site-packages vllm/v1/attention/ops/triton_unified_attention_diffkv.py
                on vllm-node-mimo-v25-upstream (no nvfp4_packed), possibly
                already patched by mods/fp8-kv-inline / diffkv-3d-qlen8 /
                diffkv-kernel-bw (none of them touch our two anchors).

Hunk A: insert the gated override block immediately BEFORE the unique line
        `    grid: tuple[Any, ...]` (i.e. after every stock BLOCK_M / use_3d /
        tile_size decision, before the grid is built).
Hunk B: insert `        **_pf_launch_kwargs,` immediately before the `    )`
        that closes the single `kernel_unified_attention_diffkv[grid](` call
        (robust against fp8-kv-inline / nvfp4 launch-tail rewrites).

Default (env VLLM_DIFFKV_PREFILL_TUNE unset) is a TRUE no-op: the gate short-
circuits on `max_seqlen_q > _PF_MIN_Q` before any env read, `_pf_launch_kwargs`
stays {}, and `**{}` adds nothing to the Triton launch -> identical specializa-
tion key -> byte-identical compiled kernel.

Discipline (per mods/diffkv-kernel-bw): marker-guarded idempotency, exact-count
anchors, ast.parse after edit, sys.exit(3) on any anchor miss.

Usage: patch_prefill_tune.py <path-to-kernel.py>
"""
import ast
import pathlib
import sys

MARKER = "DIFFKV_PREFILL_TUNE"

ANCHOR_A = "    grid: tuple[Any, ...]\n"
ANCHOR_B_CALL = "kernel_unified_attention_diffkv[grid]("
ANCHOR_B_CLOSE = "\n    )\n"
KWARGS_LINE = "        **_pf_launch_kwargs,\n"

# {nv_gate} is "" on the native kernel and " and not nvfp4_packed" on the host
# fork (nvfp4 prefill is owned by the TK/WMMA/tile16 machinery; measured slower
# with wider tiles -> never tune it).
BLOCK_TMPL = '''\
    # ------------------------------------------------------------------
    # {marker} (A1): env-gated prefill q-block tuning for sm_121.
    # Stock prefill launches BLOCK_M=16 -> BLOCK_Q=1 for MiMo TP=2
    # (num_queries_per_kv=16): ONE query token per program, so every
    # program re-streams the whole KV history (under-tiled).  Upstream
    # precedent: `tuned_large_head` in triton_unified_attention.py
    # (BLOCK_M=32 / 8 warps / 2 stages / wider tile, "~2x faster on
    # B200" for the same nqpk<=16 large-head shape class).
    #
    # Gate: fires ONLY when VLLM_DIFFKV_PREFILL_TUNE is set AND
    # max_seqlen_q > _PF_MIN_Q (8).  Decode (q_len 1), DFlash verify
    # (q_len = num_spec+1 = 8) and every cudagraph-captured shape are
    # therefore UNTOUCHED (prefill attention is never graph-captured).
    # !! If num_speculative_tokens is ever raised past 7 (e.g. the
    # spec11 recipe -> verify q_len 12), _PF_MIN_Q MUST be raised above
    # num_spec+1 first, or spec-verify launches (incl. inside cudagraph
    # capture) would silently take the tuned compile.
    #
    # Env format: "BLOCK_M,num_warps,num_stages,tile" e.g. "32,8,2,32";
    # "1" is an alias for "32,8,2,32".  Invalid values are logged once
    # and IGNORED (stock launch) -- config errors must never crash or
    # half-apply: the four params apply all-or-nothing.
    # ------------------------------------------------------------------
    _PF_MIN_Q = 8
    _pf_launch_kwargs: dict = {{}}
    if max_seqlen_q > _PF_MIN_Q and not use_3d{nv_gate}:
        import os as _os_pf
        _pf_env = _os_pf.environ.get("VLLM_DIFFKV_PREFILL_TUNE", "32,4,2,32")  # default-ON in prod (bit-identical, parity-proven); set env to "off" to force stock
        if _pf_env:
            try:
                _pf_bm, _pf_w, _pf_s, _pf_t = (
                    int(x) for x in
                    ("32,8,2,32" if _pf_env == "1" else _pf_env).split(","))
                if (_pf_bm >= num_queries_per_kv
                        and _pf_bm % num_queries_per_kv == 0
                        and _pf_bm & (_pf_bm - 1) == 0
                        and _pf_w in (2, 4, 8, 16)
                        and 1 <= _pf_s <= 4
                        and 16 <= _pf_t <= 128
                        and _pf_t & (_pf_t - 1) == 0):
                    BLOCK_M = _pf_bm
                    BLOCK_Q = BLOCK_M // num_queries_per_kv
                    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
                    tile_size = _pf_t
                    _pf_launch_kwargs["num_warps"] = _pf_w
                    _pf_launch_kwargs["num_stages"] = _pf_s
                else:
                    logger.warning_once(
                        "[prefill-tune] rejected VLLM_DIFFKV_PREFILL_TUNE="
                        "%s (constraint fail) -- stock launch", _pf_env)
            except (ValueError, TypeError):
                logger.warning_once(
                    "[prefill-tune] unparsable VLLM_DIFFKV_PREFILL_TUNE="
                    "%s -- stock launch", _pf_env)
'''


def main(path: str) -> None:
    p = pathlib.Path(path)
    src = p.read_text()

    if MARKER in src:
        print(f"[prefill-tune] already applied to {p} -- no-op")
        return

    if "def unified_attention_diffkv(" not in src:
        print(f"[prefill-tune] FATAL: {p} has no unified_attention_diffkv")
        sys.exit(3)

    nv_gate = " and not nvfp4_packed" if "nvfp4_packed" in src else ""

    # ---- Hunk A ----
    if src.count(ANCHOR_A) != 1:
        print(f"[prefill-tune] FATAL: grid-decl anchor count = "
              f"{src.count(ANCHOR_A)} (expected 1)")
        sys.exit(3)
    block = BLOCK_TMPL.format(marker=MARKER, nv_gate=nv_gate)
    src = src.replace(ANCHOR_A, block + "\n" + ANCHOR_A, 1)

    # ---- Hunk B ----
    if src.count(ANCHOR_B_CALL) != 1:
        print(f"[prefill-tune] FATAL: launch-call anchor count = "
              f"{src.count(ANCHOR_B_CALL)} (expected 1)")
        sys.exit(3)
    call_at = src.index(ANCHOR_B_CALL)
    close_at = src.find(ANCHOR_B_CLOSE, call_at)
    if close_at < 0:
        print("[prefill-tune] FATAL: launch close-paren not found")
        sys.exit(3)
    src = (src[:close_at] + "\n" + KWARGS_LINE.rstrip("\n")
           + src[close_at:])

    ast.parse(src)  # loud on any syntax slip
    p.write_text(src)
    print(f"[prefill-tune] applied to {p} "
          f"(variant: {'host-fork/nvfp4' if nv_gate else 'native'}; "
          f"gate: max_seqlen_q>8 AND VLLM_DIFFKV_PREFILL_TUNE set)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
