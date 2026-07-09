#!/bin/bash
set -uo pipefail
# ============================================================================
# dcp-diffkv — Decode Context Parallelism (DCP) for the MiMo-V2.5 Triton
#   DiffKV / fp8 decode path, gated behind VLLM_DCP (default 0 = byte-for-byte
#   stock, no LSE store, no DCP dispatch).
#
#   Image: vllm-node-mimo-v25-upstream:latest
#   vllm 0.23.1rc1.dev760+g3775d5fca (main @3775d5fc), sm_121a, 2x GB10, TP=2.
#   Target: the 9 FULL-attention DiffKV layers at deep context (200K-500K),
#   where per-rank full-context KV read is the decode bottleneck. SWA(128)
#   layers stay on the normal path (128-token window -> nothing to split).
#
#   Feasibility + file:line evidence: /home/admin/dcp_feasibility.md (GO,
#   ~1.5-1.8x deep @400K, Amdahl-capped to the 9 full-attn layers; comm is
#   latency-bound + tiny, BW saving is large -> win survives the RoCE A2A).
#
# ---------------------------------------------------------------------------
# WHAT DCP DOES (verified from the installed scaffolding, all backend-generic):
#   Partitions the KV-cache SEQUENCE dimension across the DCP ranks (round-robin
#   at cp_kv_cache_interleave_size, default 1 = per-token). Each rank attends
#   only its shard of the context KV; partial (out, LSE) are combined across
#   ranks with an EXACT LSE-weighted reduction (dcp_a2a_lse_reduce, one packed
#   all_to_all_single, or cp_lse_ag_out_rs = AllGather+ReduceScatter). This is
#   the standard FlashDecoding cross-shard merge, done across ranks instead of
#   across intra-rank segments. The split is on seq positions / block_table only
#   -> KV-head- and head_dim-AGNOSTIC -> orthogonal to our TP head-split.
#     ops/dcp_alltoall.py         : the A2A combine + CPU golden _lse_weighted_combine
#     ops/common.py:213           : cp_lse_ag_out_rs (AG+RS combine, same signature)
#     backends/utils.py:885       : get_dcp_local_seq_lens (KV split, head-agnostic)
#     backends/flash_attn.py:1037 : the generic (non-MLA) _forward_with_dcp we copy
#     worker/cp_utils.py:30       : the gate (asserts need_to_return_lse_for_decode)
#     backend.py:757/768/778/822  : can_return_lse_for_decode / lse_base_on_e defaults
#
# THE ONE THING THAT BLOCKS DCP ON OUR TRITON PATH (the report's key finding):
#   cp_utils.py:30 hard-asserts, for dcp_size>1, that every attn impl sets
#   need_to_return_lse_for_decode. backend.py:822 derives that as
#     need_to_return_lse_for_decode = (dcp_world_size>1 AND can_return_lse_for_decode)
#   Base can_return_lse_for_decode=False (backend.py:757); dense FA overrides it
#   True (flash_attn.py:675). Our TritonAttentionDiffKVImpl does NOT -> a naive
#   `--decode-context-parallel-size 2` would abort with
#     "...requires attention implementations to return the softmax LSE during
#      decode, but TritonAttentionDiffKVImpl does not..."
#
#   BUT our 3D split-KV (FlashDecoding) reduce kernel ALREADY computes the LSE
#   internally and then throws it away:
#     ops/triton_unified_attention_diffkv.py :: kernel_reduce_segments_diffkv
#       L387  overall_max    = tl.max(segm_max)
#       L391  overall_expsum = tl.sum(segm_expsum)          # after rescale to overall_max
#       L407  acc = acc_sum / overall_expsum                 # normalize
#       L414  tl.store(output_ptr, acc)                      # <-- discards max+expsum
#   LSE = log(overall_expsum) + overall_max, natural log (base e) -> matches
#   backend.py:768 default lse_base_on_e=True AND the combine kernel's IS_BASE_E
#   arm (dcp_alltoall.py:273 `tl.log(lse_sum)+lse_max`) AND the golden
#   (dcp_alltoall.py:94). So exposure is ONE extra store of an already-correct,
#   already-base-e intermediate -- additive, not a rewrite.
#
# WHAT THIS MOD DOES (all edits marker-guarded / ast-checked / idempotent /
# env-gated so default is a byte-for-byte no-op):
#   A. KERNEL LSE-EXPOSE (ops/triton_unified_attention_diffkv.py), 3 edits:
#      A1. add `out_lse_ptr` + strides + `RETURN_LSE: tl.constexpr` to
#          kernel_reduce_segments_diffkv's signature (additive params).
#      A2. after the normalize (L414 store of `acc`), when RETURN_LSE compute
#          `lse = log(overall_expsum) + overall_max` (base e, guarded for the
#          all-masked overall_expsum==0 row -> -inf, matching the golden's
#          NaN/inf handling) and store it to out_lse_ptr.
#      A3. thread an optional `out_lse` tensor through unified_attention_diffkv:
#          add the param (default None); only the 3D path can return LSE (the 2D
#          path has no per-segment reduce), so pass RETURN_LSE = (out_lse is not
#          None) and the pointer/strides into the reduce launch. When out_lse is
#          None the launch is byte-for-byte the stock call (RETURN_LSE=False,
#          a dummy 1-elem pointer never written).
#   B. BACKEND IMPL (backends/triton_attn_diffkv.py), 3 edits:
#      B1. TritonAttentionDiffKVImpl: class attr can_return_lse_for_decode=True
#          + lse_base_on_e=True (explicit; == default). This is inert until
#          dcp_world_size>1 (backend.py:822 gates need_to_return_lse_for_decode
#          on it), which only happens when the recipe passes
#          --decode-context-parallel-size >1. So at DCP=1 the flag flips nothing.
#      B2. __init__: when dcp_world_size>1, set self.dcp_combine
#          (dcp_a2a_lse_reduce if --dcp-comm-backend a2a else cp_lse_ag_out_rs --
#          identical selection to flash_attn.py:743) and self._dcp_dtype
#          (model dtype, for the context-shard scratch). No-op at DCP=1.
#      B3. forward: BEFORE the stock unified_attention_diffkv call, when
#          self.dcp_world_size>1 AND this is a pure-decode/verify step
#          (max_query_len <= _DIFFKV_3D_Q8_MAXQ so the 3D+LSE path is reachable),
#          route through the new _forward_with_dcp and return. Otherwise fall
#          through to the untouched stock path (prefill, or DCP off).
#      B4. add _forward_with_dcp: a near-copy of flash_attn.py:1037 adapted to
#          the Triton DiffKV kernel + our packed/fp8 cache. STAGE-2 DRAFTED (no
#          longer a bare stub): (1) all-gather Q -> context-shard 3D+LSE attn ->
#          dcp_combine; (2) local query-block self-attn; (3) merge_attn_states
#          with the [B,H]->[H,B] LSE transposes. The full control flow is real,
#          reviewable code. BUT the DiffKV kernel is causal-ONLY + cache-ONLY (the
#          flash ref uses causal=False context + raw-K/V query), so the two
#          seqused_k/block_table constructions (context-non-causal-over-shard,
#          query-only-new-tokens) are un-determinable statically -> the executed
#          body HARD-ASSERTS unless VLLM_DCP_STAGE2_OK=1 acknowledges they were
#          pinned by dcp_parity.py + a shadow A/B. Guarded, not guessed; strictly
#          dead code in prod. Loud logger.info_once TEST-path banner at entry.
#   C. METADATA (backends/triton_attn_diffkv.py builder + parent dataclass):
#      C1. add optional dcp_context_kv_lens / max_dcp_context_kv_len fields +
#          the seven pre-allocated CONTEXT-phase scratch fields
#          (dcp_context_softmax_segm_{output,max,expsum}, dcp_context_{out,lse},
#          dcp_query_{out,lse}) to TritonAttentionMetadata (defaulted None).
#      C2. in the DiffKV builder __init__, acquire dcp_world_size/dcp_rank/
#          cp_kv_cache_interleave_size (the Triton parent does NOT set them, only
#          the FA builder does) and allocate the context-phase scratch ONCE,
#          head-widened to num_heads_q*dcp_world_size (piece i, cudagraph-safe);
#          then wrap build() to compute the DCP local seq-len split
#          (get_dcp_local_seq_lens) + max_dcp_context_kv_len and stash the scratch
#          on the metadata, mirroring flash_attn.py:522-542. No-op at DCP=1.
#
# COMPOSITION (all verified against the installed code):
#   x fp8-KV (VLLM_FP8_INLINE): the fp8 descale happens INSIDE each rank's local
#     shard attention exactly as today (the same key_cache/value_cache slicing +
#     fp8_kwargs are passed into the DCP context/query calls). DCP combines the
#     POST-attention bf16 outputs + fp32 LSE. fp8 NEVER crosses the wire: the A2A
#     payload dtype is _dcp_dtype (= model dtype, bf16) with fp32 LSE bit-packed
#     into 2 bf16 slots (dcp_alltoall.py _dcp_a2a_lse_pack_dim). No numeric
#     interaction with the KV quant.
#   x DFlash q_len=8 verify: DCP requires the 3D+LSE path, which is EXACTLY the
#     path mods/diffkv-3d-qlen8 unlocks for q_len<=8. So DCP is dispatched only
#     when VLLM_DIFFKV_3D_Q8=1 (q_len 2..8) OR q_len==1 (already 3D). The 8-token
#     query self-attn is LOCAL (new tokens, not split); only the cached context
#     KV is sequence-split. merge_attn_states fuses context+query per token.
#   x diffkv-3d-qlen8 3D path: this mod's LSE-expose ADDS an output to the SAME
#     reduce kernel that 3d-qlen8 makes reachable at q_len<=8; the LOCAL segm
#     buffers 3d-qlen8 resized are reused by the query phase. The context phase
#     (Q-all-gathered -> num_heads*dcp_world_size heads) now has its OWN,
#     head-widened segm buffers allocated once in the builder (piece i).
#   x cudagraph: the KNOWN hazard. dcp_alltoall.py:116-129 already solves A2A
#     buffer aliasing under FULL cudagraph by using PRIVATE torch.empty send/recv
#     buffers (NOT the growable workspace) so replay addresses stay valid. The
#     combine we call (dcp_a2a_lse_reduce) uses that discipline internally. This
#     mod's OWN context-phase scratch (the Q-gathered out + LSE + segm buffers)
#     is now allocated ONCE in the builder (replay-stable, sliced per-call) and
#     stashed on the metadata -- the same cudagraph discipline. It STILL must be
#     re-verified live under FULL cudagraph (see NOTES). First bringup MUST be
#     enforce-eager (as with every prior DiffKV kernel change).
#
# ORDERING: run AFTER fix-mimo-v2-upstream, fp8-kv-inline, AND diffkv-3d-qlen8
#   (this mod anchors on the fp8- and 3d-qlen8-patched state of both kernel and
#   backend: the reduce kernel + its launch, the wrapper signature ending in
#   `v_descale`, the impl class, the forward's unified_attention_diffkv call).
#   Verified: none of those three touch the specific anchors this mod uses.
#
# SELECT: env VLLM_DCP=1 arms the DCP dispatch in forward. The recipe ALSO needs
#   `--decode-context-parallel-size 2` (and optionally `--dcp-comm-backend a2a`)
#   for the DCP process group to exist and the gate to require the LSE (which
#   this mod now supplies). VLLM_DCP unset/0 -> forward never calls the DCP path,
#   the LSE store stays off (out_lse=None), and the impl is byte-for-byte stock
#   EXCEPT the inert can_return_lse_for_decode class attr (which only matters
#   when dcp_world_size>1). Belt-and-suspenders: both the env AND the CLI knob
#   must be set for any behavior change.
#
# TEST GATE (sm_121 Triton has a documented history of silent garbage; this mod
#   adds new math to the reduce kernel): DO NOT trust without, IN ORDER:
#     1. /home/admin/dcp_parity.py PASS inside the container (offline, GPU):
#        DiffKV decode DCP=1 (no split) vs DCP=2 (2-way seq split + LSE-combine)
#        vs the CPU golden _lse_weighted_combine, max abs/rel <= 2e-3 on fp8 KV,
#        head_dim 192/128, seq_lens incl. 200K/400K.
#     2. shadow A/B on a SCRATCH port (never :8000/:8001): DCP=1 vs DCP=2 at
#        200K/400K deep decode, enforce-eager first, then cudagraph.
#     3. needle@240K + a coherence/reliability pass.
#   This mod only makes the DCP path REACHABLE + supplies the LSE; it does not by
#   itself prove the sm_121 kernel is bit-correct with the new store.
#
# Discipline: marker DCP_DIFFKV, ast.parse after each edit, sys.exit on
#   anchor-miss, idempotent (no-op if markers already present).
# ============================================================================

SITE="/usr/local/lib/python3.12/dist-packages"
KERNEL="$SITE/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
BACKEND="$SITE/vllm/v1/attention/backends/triton_attn_diffkv.py"
PARENT="$SITE/vllm/v1/attention/backends/triton_attn.py"

echo "[dcp-diffkv] grafting Decode Context Parallelism onto the DiffKV Triton decode (VLLM_DCP gate, default off)"

# ----------------------------------------------------------------------------
# 0. Sanity: all files + load-bearing anchors present, in the fp8/3d-qlen8-
#    patched state this mod expects. Fail loudly (exit 3) on any miss.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import sys, pathlib
K = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
B = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py")
P = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn.py")
for f in (K, B, P):
    if not f.exists():
        print(f"[dcp-diffkv] FATAL: missing file {f}"); sys.exit(3)
ks, bs, ps = K.read_text(), B.read_text(), P.read_text()

# --- kernel anchors ---
if "def kernel_reduce_segments_diffkv(" not in ks:
    print("[dcp-diffkv] FATAL: kernel reduce fn anchor missing"); sys.exit(3)
sig_tail = "    BLOCK_Q: tl.constexpr,\n    NUM_SEGMENTS_PER_SEQ: tl.constexpr,\n):\n"
if ks.count(sig_tail) != 1:
    print(f"[dcp-diffkv] FATAL: reduce signature tail count={ks.count(sig_tail)} (expected 1)"); sys.exit(3)
store_anchor = "    tl.store(output_ptr + output_offset, acc, mask=dim_mask)\n"
if ks.count(store_anchor) != 1:
    print(f"[dcp-diffkv] FATAL: reduce acc-store anchor count={ks.count(store_anchor)} (expected 1)"); sys.exit(3)
if ks.count("    overall_max = tl.max(segm_max)\n") != 1 or \
   ks.count("    overall_expsum = tl.sum(segm_expsum)\n") != 1:
    print("[dcp-diffkv] FATAL: overall_max / overall_expsum anchors missing"); sys.exit(3)
wrap_close = "    v_descale: float = 1.0,\n):\n"
if ks.count(wrap_close) != 1:
    print(f"[dcp-diffkv] FATAL: unified_attention_diffkv sig-close anchor count={ks.count(wrap_close)} (expected 1)"); sys.exit(3)
launch_anchor = "        kernel_reduce_segments_diffkv[(q.shape[0], num_query_heads)](\n"
if ks.count(launch_anchor) != 1:
    print(f"[dcp-diffkv] FATAL: reduce launch anchor count={ks.count(launch_anchor)} (expected 1)"); sys.exit(3)
launch_tail = "            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,\n        )\n"
if ks.count(launch_tail) != 1:
    print(f"[dcp-diffkv] FATAL: reduce launch tail anchor count={ks.count(launch_tail)} (expected 1)"); sys.exit(3)
if "_DIFFKV_3D_Q8_MAXQ" not in ks:
    print("[dcp-diffkv] FATAL: expected diffkv-3d-qlen8 to be applied first (_DIFFKV_3D_Q8_MAXQ missing)"); sys.exit(3)

# --- backend anchors ---
if "class TritonAttentionDiffKVImpl(TritonAttentionImpl):" not in bs:
    print("[dcp-diffkv] FATAL: impl class anchor missing"); sys.exit(3)
impl_ctor = "    def __init__(self, *args, **kwargs) -> None:\n        super().__init__(*args, **kwargs)\n"
if bs.count(impl_ctor) != 1:
    print(f"[dcp-diffkv] FATAL: impl __init__ anchor count={bs.count(impl_ctor)} (expected 1)"); sys.exit(3)
fwd_call = "        unified_attention_diffkv(\n"
if bs.count(fwd_call) != 1:
    print(f"[dcp-diffkv] FATAL: forward unified_attention_diffkv call anchor count={bs.count(fwd_call)} (expected 1)"); sys.exit(3)
fwd_head = ("        num_actual_tokens = attn_metadata.num_actual_tokens\n"
            "        head_size_qk = self.head_size\n"
            "        head_size_v = TritonAttentionDiffKVBackend.head_size_v\n")
if bs.count(fwd_head) != 1:
    print(f"[dcp-diffkv] FATAL: forward head anchor count={bs.count(fwd_head)} (expected 1)"); sys.exit(3)
if "from vllm.v1.attention.ops.triton_unified_attention_diffkv import (" not in bs:
    print("[dcp-diffkv] FATAL: unified_attention_diffkv import anchor missing"); sys.exit(3)

# --- parent dataclass anchor (optional-fields tail of TritonAttentionMetadata) ---
dc_tail = ("    rswa_prefix_lens: torch.Tensor | None = None\n"
           "    rswa_window: int | None = None\n")
if ps.count(dc_tail) != 1:
    print(f"[dcp-diffkv] FATAL: TritonAttentionMetadata optional-tail anchor count={ps.count(dc_tail)} (expected 1)"); sys.exit(3)

print("[dcp-diffkv] sanity OK (kernel reduce+launch+wrapper, backend impl+forward+import, parent dataclass anchors present)")
PY
[ $? -eq 0 ] || { echo "[dcp-diffkv] sanity FAILED — aborting"; exit 1; }


# ============================================================================
# A. KERNEL: expose the softmax LSE from kernel_reduce_segments_diffkv.
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
src = p.read_text()

if "DCP_DIFFKV" in src:
    print("[dcp-diffkv] kernel LSE-expose already patched"); sys.exit(0)

# --- A1: extend the reduce kernel signature (add out_lse_ptr + strides + flag) ---
sig_tail = "    BLOCK_Q: tl.constexpr,\n    NUM_SEGMENTS_PER_SEQ: tl.constexpr,\n):\n"
sig_new = (
    "    BLOCK_Q: tl.constexpr,\n"
    "    NUM_SEGMENTS_PER_SEQ: tl.constexpr,\n"
    "    # DCP_DIFFKV: optional LSE output for Decode Context Parallelism. When\n"
    "    # RETURN_LSE, store LSE = log(overall_expsum) + overall_max (base e) so\n"
    "    # the cross-rank dcp combine (dcp_a2a_lse_reduce / cp_lse_ag_out_rs) can\n"
    "    # do the exact LSE-weighted merge. out_lse_ptr is [num_tokens,\n"
    "    # num_query_heads]; when RETURN_LSE is False it is a dummy 1-elem tensor\n"
    "    # that is never written (byte-for-byte stock behavior).\n"
    "    out_lse_ptr,\n"
    "    out_lse_stride_0: tl.int64,\n"
    "    out_lse_stride_1: tl.int64,\n"
    "    RETURN_LSE: tl.constexpr,\n"
    "):\n"
)
if src.count(sig_tail) != 1:
    print(f"[dcp-diffkv] FATAL: reduce sig-tail count={src.count(sig_tail)}"); sys.exit(3)
src = src.replace(sig_tail, sig_new, 1)

# --- A2: after the acc store, emit the LSE store guarded by RETURN_LSE ---
store_anchor = "    tl.store(output_ptr + output_offset, acc, mask=dim_mask)\n"
store_new = (
    "    tl.store(output_ptr + output_offset, acc, mask=dim_mask)\n"
    "\n"
    "    # DCP_DIFFKV: expose the softmax LSE that this reduce already computed.\n"
    "    # overall_max/overall_expsum are exactly the intra-rank FlashDecoding\n"
    "    # segment reduction; LSE = ln(expsum)+max is the same identity the\n"
    "    # cross-rank combine uses. For an all-masked row (overall_expsum==0,\n"
    "    # e.g. a rank whose shard is empty) emit -inf, which the combine's\n"
    "    # NaN/inf guard maps to a zero weight (dcp_alltoall.py:236/262/294).\n"
    "    if RETURN_LSE:\n"
    "        lse_val = tl.where(\n"
    "            overall_expsum == 0.0,\n"
    "            float(\"-inf\"),\n"
    "            tl.log(overall_expsum) + overall_max,\n"
    "        )\n"
    "        out_lse_offset = (\n"
    "            query_token_idx.to(tl.int64) * out_lse_stride_0\n"
    "            + query_head_idx * out_lse_stride_1\n"
    "        )\n"
    "        tl.store(out_lse_ptr + out_lse_offset, lse_val)\n"
)
if src.count(store_anchor) != 1:
    print(f"[dcp-diffkv] FATAL: acc-store anchor count={src.count(store_anchor)}"); sys.exit(3)
src = src.replace(store_anchor, store_new, 1)

# --- A3a: add the out_lse param to the unified_attention_diffkv wrapper ---
wrap_close = "    v_descale: float = 1.0,\n):\n"
wrap_new = (
    "    v_descale: float = 1.0,\n"
    "    # DCP_DIFFKV: optional LSE output. Only the 3D split-KV path can return\n"
    "    # LSE (the 2D path has no per-segment reduce). When provided (a\n"
    "    # [num_tokens, num_query_heads] fp32 tensor) and the 3D path is taken,\n"
    "    # the reduce kernel also writes LSE = ln(expsum)+max. None -> stock.\n"
    "    out_lse: torch.Tensor | None = None,\n"
    "):\n"
)
if src.count(wrap_close) != 1:
    print(f"[dcp-diffkv] FATAL: wrapper sig-close count={src.count(wrap_close)}"); sys.exit(3)
src = src.replace(wrap_close, wrap_new, 1)

# --- A3b: thread out_lse into the reduce launch (RETURN_LSE + ptr + strides) ---
launch_tail = "            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,\n        )\n"
launch_new = (
    "            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,\n"
    "            # DCP_DIFFKV: emit LSE when the caller (the DCP forward) asked.\n"
    "            # A dummy 1-elem out (stride 0) keeps the pointer non-null and\n"
    "            # never written when RETURN_LSE is False.\n"
    "            out_lse_ptr=(out_lse if out_lse is not None else out),\n"
    "            out_lse_stride_0=(out_lse.stride(0) if out_lse is not None else 0),\n"
    "            out_lse_stride_1=(out_lse.stride(1) if out_lse is not None else 0),\n"
    "            RETURN_LSE=(out_lse is not None),\n"
    "        )\n"
)
if src.count(launch_tail) != 1:
    print(f"[dcp-diffkv] FATAL: reduce launch-tail count={src.count(launch_tail)}"); sys.exit(3)
src = src.replace(launch_tail, launch_new, 1)

ast.parse(src)
p.write_text(src)
print("[dcp-diffkv] kernel LSE-expose applied (reduce sig + LSE store + wrapper out_lse + launch)")
PY
[ $? -eq 0 ] || { echo "[dcp-diffkv] kernel LSE-expose FAILED — aborting"; exit 1; }


# ============================================================================
# C1. PARENT dataclass: add optional DCP metadata fields (additive, default None).
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn.py")
src = p.read_text()

if "DCP_DIFFKV" in src:
    print("[dcp-diffkv] parent dataclass already patched"); sys.exit(0)

dc_tail = ("    rswa_prefix_lens: torch.Tensor | None = None\n"
           "    rswa_window: int | None = None\n")
dc_new = (
    "    rswa_prefix_lens: torch.Tensor | None = None\n"
    "    rswa_window: int | None = None\n"
    "\n"
    "    # DCP_DIFFKV: Decode Context Parallelism metadata. Per-rank LOCAL context\n"
    "    # KV lengths (get_dcp_local_seq_lens) + the max over ranks, used by the\n"
    "    # DiffKV _forward_with_dcp context-shard attention. None when dcp is off.\n"
    "    dcp_context_kv_lens: torch.Tensor | None = None\n"
    "    max_dcp_context_kv_len: int | None = None\n"
    "    # DCP_DIFFKV: pre-allocated, replay-stable CONTEXT-phase scratch (piece i),\n"
    "    # allocated ONCE in TritonAttentionDiffKVMetadataBuilder.__init__ and\n"
    "    # stashed here by its build() override so the impl reads cudagraph-safe\n"
    "    # buffers off the metadata. CONTEXT segm/out are head-widened to\n"
    "    # num_heads_q*dcp_world_size (the Q-all-gather count); the query buffers\n"
    "    # are the LOCAL num_heads_q. All None when dcp is off.\n"
    "    dcp_context_softmax_segm_output: torch.Tensor | None = None\n"
    "    dcp_context_softmax_segm_max: torch.Tensor | None = None\n"
    "    dcp_context_softmax_segm_expsum: torch.Tensor | None = None\n"
    "    dcp_context_out: torch.Tensor | None = None\n"
    "    dcp_context_lse: torch.Tensor | None = None\n"
    "    dcp_query_out: torch.Tensor | None = None\n"
    "    dcp_query_lse: torch.Tensor | None = None\n"
)
if src.count(dc_tail) != 1:
    print(f"[dcp-diffkv] FATAL: dataclass optional-tail count={src.count(dc_tail)}"); sys.exit(3)
src = src.replace(dc_tail, dc_new, 1)

ast.parse(src)
p.write_text(src)
print("[dcp-diffkv] parent TritonAttentionMetadata extended with DCP fields (optional, default None)")
PY
[ $? -eq 0 ] || { echo "[dcp-diffkv] parent dataclass patch FAILED — aborting"; exit 1; }


# ============================================================================
# B + C2. BACKEND: LSE flag, dcp_combine init, DCP dispatch, _forward_with_dcp,
#         and the builder DCP-metadata computation.
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py")
src = p.read_text()

if "DCP_DIFFKV" in src:
    print("[dcp-diffkv] backend DCP wiring already patched"); sys.exit(0)

# --- add imports we need (idempotent: guarded by the DCP_DIFFKV marker above) ---
imp_anchor = ("from vllm.v1.attention.ops.triton_unified_attention_diffkv import (\n"
              "    unified_attention_diffkv,\n"
              ")\n")
if src.count(imp_anchor) != 1:
    print(f"[dcp-diffkv] FATAL: import anchor count={src.count(imp_anchor)}"); sys.exit(3)
imp_new = imp_anchor + (
    "# DCP_DIFFKV: imports for Decode Context Parallelism.\n"
    "import os as _os_dcp\n"
    "from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens as _dcp_local_seq_lens\n"
    "from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce as _dcp_a2a_lse_reduce\n"
    "from vllm.v1.attention.ops.common import cp_lse_ag_out_rs as _cp_lse_ag_out_rs\n"
    "from vllm.v1.attention.ops.merge_attn_states import merge_attn_states as _dcp_merge_attn_states\n"
    "from vllm.config import get_current_vllm_config_or_none as _dcp_get_cfg\n"
    "\n"
    "\n"
    "def _dcp_group_or_none():\n"
    "    # DCP_DIFFKV: fetch the DCP GroupCoordinator, or None if DCP is not\n"
    "    # initialized (dcp=1 / tests). Mirrors flash_attn.py:362-370.\n"
    "    try:\n"
    "        from vllm.distributed.parallel_state import get_dcp_group\n"
    "        return get_dcp_group()\n"
    "    except (AssertionError, Exception):\n"
    "        return None\n"
    "\n"
    "# DCP_DIFFKV master gate. VLLM_DCP=1 arms the DCP dispatch in forward; the\n"
    "# recipe ALSO needs --decode-context-parallel-size >1 for the DCP group to\n"
    "# exist. Both required (belt-and-suspenders): unset/0 => stock, no dispatch.\n"
    "_VLLM_DCP = _os_dcp.environ.get(\"VLLM_DCP\", \"0\") == \"1\"\n"
)
src = src.replace(imp_anchor, imp_new, 1)

# --- B1: class-level LSE-capable flags on the impl (inert until dcp_world_size>1) ---
impl_anchor = ('class TritonAttentionDiffKVImpl(TritonAttentionImpl):\n'
               '    """Triton attention impl for the DiffKV packed KV cache layout."""\n')
impl_new = (
    'class TritonAttentionDiffKVImpl(TritonAttentionImpl):\n'
    '    """Triton attention impl for the DiffKV packed KV cache layout."""\n'
    '\n'
    '    # DCP_DIFFKV: our 3D split-KV reduce now exposes the softmax LSE (base e).\n'
    '    # backend.py:822 derives need_to_return_lse_for_decode = (dcp_world_size>1\n'
    '    # AND can_return_lse_for_decode), so this ONLY has effect when the recipe\n'
    '    # enables DCP; at dcp=1 it flips nothing. lse_base_on_e=True matches the\n'
    '    # reduce kernel (ln(expsum)+max) and the combine IS_BASE_E arm.\n'
    '    can_return_lse_for_decode: bool = True\n'
    '    lse_base_on_e: bool = True\n'
)
if src.count(impl_anchor) != 1:
    print(f"[dcp-diffkv] FATAL: impl class anchor count={src.count(impl_anchor)}"); sys.exit(3)
src = src.replace(impl_anchor, impl_new, 1)

# --- B2: __init__ -> set up dcp_combine + _dcp_dtype when DCP is on ---
ctor_anchor = ("    def __init__(self, *args, **kwargs) -> None:\n"
               "        super().__init__(*args, **kwargs)\n")
ctor_new = ctor_anchor + (
    "        # DCP_DIFFKV: pick the cross-rank combine + the A2A payload dtype.\n"
    "        # Identical selection to flash_attn.py:743 (a2a vs AllGather+RS).\n"
    "        # dcp_world_size/dcp_rank were populated by AttentionImplBase.__new__.\n"
    "        self._dcp_dtype = None\n"
    "        self.dcp_combine = None\n"
    "        if getattr(self, \"dcp_world_size\", 1) > 1:\n"
    "            _cfg = _dcp_get_cfg()\n"
    "            _a2a = (\n"
    "                _cfg is not None\n"
    "                and getattr(_cfg.parallel_config, \"dcp_comm_backend\", \"ag_rs\") == \"a2a\"\n"
    "            )\n"
    "            self.dcp_combine = _dcp_a2a_lse_reduce if _a2a else _cp_lse_ag_out_rs\n"
    "            self._dcp_dtype = _cfg.model_config.dtype if _cfg is not None else None\n"
)
if src.count(ctor_anchor) != 1:
    print(f"[dcp-diffkv] FATAL: impl __init__ anchor count={src.count(ctor_anchor)}"); sys.exit(3)
src = src.replace(ctor_anchor, ctor_new, 1)

# --- B3: forward -> DCP dispatch just after the head-size locals are set ---
fwd_head = ("        num_actual_tokens = attn_metadata.num_actual_tokens\n"
            "        head_size_qk = self.head_size\n"
            "        head_size_v = TritonAttentionDiffKVBackend.head_size_v\n")
fwd_new = fwd_head + (
    "\n"
    "        # DCP_DIFFKV: route decode/verify through the cross-rank DCP path.\n"
    "        # Gated on: the env, an actual DCP group (dcp_world_size>1), the DCP\n"
    "        # metadata being present (built only when dcp is on), AND the step\n"
    "        # being pure-decode/verify (max_query_len <= _DIFFKV_3D_Q8_MAXQ) so\n"
    "        # the 3D+LSE path is reachable. Prefill and dcp-off fall through to\n"
    "        # the untouched stock path below.\n"
    "        if (\n"
    "            _VLLM_DCP\n"
    "            and getattr(self, \"dcp_world_size\", 1) > 1\n"
    "            and self.dcp_combine is not None\n"
    "            and getattr(attn_metadata, \"dcp_context_kv_lens\", None) is not None\n"
    "            and attn_metadata.max_query_len <= _DIFFKV_3D_Q8_MAXQ\n"
    "        ):\n"
    "            return self._forward_with_dcp(\n"
    "                layer, query, key, value, kv_cache, attn_metadata,\n"
    "                output, head_size_qk, head_size_v, num_actual_tokens,\n"
    "            )\n"
)
if src.count(fwd_head) != 1:
    print(f"[dcp-diffkv] FATAL: forward head anchor count={src.count(fwd_head)}"); sys.exit(3)
src = src.replace(fwd_head, fwd_new, 1)

# --- B4: add the _forward_with_dcp method + a small fp8/kv-slice helper, placed
#         immediately before forward() (so `self._forward_with_dcp` resolves) ---
fwd_def_anchor = "    def forward(\n        self,\n        layer: torch.nn.Module,\n"
dcp_method = (
    "    def _dcp_slice_kv(self, layer, kv_cache, head_size_qk, head_size_v):\n"
    "        \"\"\"DCP_DIFFKV: reproduce the forward's fp8/bf16 KV-view + descale\n"
    "        selection so the context DCP attention call reads the cache exactly\n"
    "        like the stock path. Returns (key_cache, value_cache, fp8_kwargs); the\n"
    "        caller uses attn_metadata.block_table (the DCP seq-len split keys off\n"
    "        the metadata's own block_table). fp8 stays IN-KERNEL; nothing fp8\n"
    "        crosses the wire (the A2A payload is the post-attention bf16 out +\n"
    "        fp32 LSE). REQUIREMENT: under DCP the fp8 cache MUST be read INLINE\n"
    "        (VLLM_FP8_INLINE=1, the prod setting) -- the SCRATCH fallback\n"
    "        (VLLM_FP8_INLINE=0) remaps block_table per-call, which is incompatible\n"
    "        with the metadata-owned DCP context split; the forward-dispatch\n"
    "        asserts this so we never silently read a mismatched block_table.\"\"\"\n"
    "        fp8_kwargs = {}\n"
    "        _is_fp8 = _fp8_is_cache(kv_cache)\n"
    "        key_cache = kv_cache[..., :head_size_qk]\n"
    "        value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]\n"
    "        if _is_fp8:\n"
    "            # Guarded in _forward_with_dcp: fp8 under DCP requires INLINE.\n"
    "            fp8_kwargs = dict(\n"
    "                fp8_packed=True,\n"
    "                k_descale=float(getattr(layer, \"_k_scale_float\", 1.0)),\n"
    "                v_descale=float(getattr(layer, \"_v_scale_float\", 1.0)),\n"
    "            )\n"
    "        return key_cache, value_cache, fp8_kwargs\n"
    "\n"
    "    def _forward_with_dcp(\n"
    "        self, layer, query, key, value, kv_cache, attn_metadata,\n"
    "        output, head_size_qk, head_size_v, num_actual_tokens,\n"
    "    ) -> torch.Tensor:\n"
    "        \"\"\"DCP_DIFFKV: cross-rank Decode Context Parallelism for DiffKV.\n"
    "\n"
    "        Near-copy of flash_attn.py:1037 (_forward_with_dcp) adapted to the\n"
    "        Triton DiffKV kernel. Two attention phases per step, then merge:\n"
    "          (1) CONTEXT-SHARD: all-gather Q across DCP ranks (dim=1, heads ->\n"
    "              num_heads*dcp_world_size), attend this rank's LOCAL context-KV\n"
    "              shard (attn_metadata.dcp_context_kv_lens) via the 3D+LSE path,\n"
    "              returning per-token LSE [B, H_all] through the newly-exposed\n"
    "              out_lse; combine across ranks with self.dcp_combine (exact\n"
    "              LSE-weighted) -> head-scattered [B, num_heads, D] + [B,num_heads]\n"
    "              LSE.\n"
    "          (2) QUERY-BLOCK (LOCAL, not split): attend the new q_len<=8 tokens\n"
    "              against the just-written K/V for this step (self-attention over\n"
    "              the local tail) via the LOCAL num_heads segm buffers, out_lse ->\n"
    "              q_out [B, num_heads, D] + q_lse [B, num_heads].\n"
    "          (3) merge_attn_states(output, ctx_out, ctx_lse[H,B], q_out,\n"
    "              q_lse[H,B]) -- LSEs transposed [B,H] -> [H,B] going into merge.\n"
    "\n"
    "        The context KV read is split across ranks (each holds ~1/N of the\n"
    "        positions) -- the win; the query self-attn is tiny (q_len<=8), local.\n"
    "\n"
    "        This path is STRICTLY DEAD CODE in production: it runs only when the\n"
    "        forward dispatch gate fires (VLLM_DCP=1 env AND dcp_world_size>1 AND\n"
    "        the DCP metadata is present AND q_len<=MAXQ), which the PROD recipe\n"
    "        never sets. First live bringup MUST be enforce-eager and gated by\n"
    "        dcp_parity.py; see NOTES.md LIVE-BOOT CHECKLIST.\n"
    "\n"
    "        ---- ONE GENUINELY UN-DETERMINABLE PIECE (guarded, not guessed) ----\n"
    "        The DiffKV Triton kernel (unified_attention_diffkv) ONLY supports\n"
    "        causal=True and derives its causal mask internally from\n"
    "        context_len = seqused_k - query_len (query_abs_pos = context_len +\n"
    "        query_pos). The flash reference instead issues the CONTEXT attention\n"
    "        with causal=False (every query token attends the ENTIRE context shard,\n"
    "        since all context precedes all new tokens) as a SEPARATE flash call\n"
    "        with its own K/V. Our fused kernel cannot express causal=False for the\n"
    "        context shard without either (a) a kernel tweak exposing USE_CAUSAL,\n"
    "        or (b) a seqused_k/cu_seqlens construction that makes every new-token\n"
    "        query see all shard positions. Which construction is bit-correct is\n"
    "        NOT statically determinable on sm_121 (Triton miscompile history) and\n"
    "        MUST be pinned by dcp_parity.py + a shadow A/B at boot. TWO seqused_k/\n"
    "        block_table constructions are therefore un-determinable statically and\n"
    "        are LIVE-BOOT items (NOTES.md), NOT guessed here:\n"
    "          - CONTEXT phase: making every new-token query attend the ENTIRE\n"
    "            local context shard [0, context_len) with NO intra-context causal\n"
    "            exclusion, given the kernel derives context_len = seqused_k -\n"
    "            query_len and is causal-only. seqused_k=dcp_context_kv_lens is the\n"
    "            starting point but the exact query_len/cu_seqlens pairing that\n"
    "            avoids clipping the last query_len context positions must be pinned\n"
    "            live.\n"
    "          - QUERY-BLOCK phase: attending ONLY the new q_len tokens (positions\n"
    "            [context_len, seq_len)) self-causally, WITHOUT re-reading the\n"
    "            context (which would double-count it under the LSE merge). The\n"
    "            fused DiffKV kernel reads K/V from the paged cache via block_table\n"
    "            (it takes no raw key/value for the step, unlike the flash ref), so\n"
    "            the restriction to the new-token positions needs a block_table\n"
    "            view/offset + seqused_k=query_len that the parity harness must pin.\n"
    "        Because BOTH constructions gate correctness and neither is provable on\n"
    "        sm_121 without the harness, the ENTIRE executed body is guarded by an\n"
    "        explicit VLLM_DCP_STAGE2_OK=1 acknowledgment (set ONLY after\n"
    "        dcp_parity.py + a shadow A/B pin the constructions). Absent it, this\n"
    "        HARD-ASSERTS rather than silently mis-attending -- preserving the\n"
    "        stage-1 safety property (DCP can never silently mis-run) while the full\n"
    "        control flow below is real, reviewable, drafted code (not a stub). The\n"
    "        outer forward dispatch already makes this dead code in production\n"
    "        (VLLM_DCP unset); this second gate makes it dead even if someone arms\n"
    "        VLLM_DCP before the harness has validated the decomposition.\n"
    "        --------------------------------------------------------------------\n"
    "        \"\"\"\n"
    "        logger.info_once(\n"
    "            \"[dcp-diffkv] running DCP stage-2 (TEST path) -- dcp_world_size=%d, \"\n"
    "            \"q_len<=%d; this is NEVER the prod recipe. enforce-eager + \"\n"
    "            \"dcp_parity.py + VLLM_DCP_STAGE2_OK must be set.\",\n"
    "            getattr(self, \"dcp_world_size\", 1), _DIFFKV_3D_Q8_MAXQ,\n"
    "        )\n"
    "\n"
    "        dcp_group = _dcp_group_or_none()\n"
    "        assert dcp_group is not None and dcp_group.world_size > 1, (\n"
    "            \"[dcp-diffkv] _forward_with_dcp reached without a live DCP group \"\n"
    "            \"(world_size>1). The forward dispatch gate should have prevented \"\n"
    "            \"this; refusing to run to avoid a silent mis-attention.\"\n"
    "        )\n"
    "        dcp_ws = dcp_group.world_size\n"
    "\n"
    "        # ---- STAGE-2 ACK GATE: the two seqused_k/block_table constructions\n"
    "        # (context non-causal-over-shard, query-only-new-tokens) are the\n"
    "        # documented un-determinable live-boot pieces. Refuse to execute the\n"
    "        # drafted flow until they are pinned by dcp_parity.py + a shadow A/B\n"
    "        # and explicitly acknowledged. This is the stage-1 no-silent-misrun\n"
    "        # property, kept intact. See NOTES.md LIVE-BOOT CHECKLIST.\n"
    "        assert _os_dcp.environ.get(\"VLLM_DCP_STAGE2_OK\", \"0\") == \"1\", (\n"
    "            \"[dcp-diffkv] _forward_with_dcp is DRAFTED but the context/query \"\n"
    "            \"seqused_k+block_table constructions are un-determinable statically \"\n"
    "            \"(the DiffKV kernel is causal-only + cache-only; the flash ref uses \"\n"
    "            \"causal=False context and raw-K/V query). Pin them with \"\n"
    "            \"/home/admin/dcp_parity.py + a scratch-port shadow A/B, then set \"\n"
    "            \"VLLM_DCP_STAGE2_OK=1. Refusing to run to avoid silent \"\n"
    "            \"mis-attention (double-counted context / clipped shard).\"\n"
    "        )\n"
    "\n"
    "        # fp8-under-DCP must be INLINE (see _dcp_slice_kv): the SCRATCH path\n"
    "        # remaps block_table, incompatible with the metadata-owned context\n"
    "        # split. Guard it rather than silently reading a mismatched table.\n"
    "        _is_fp8 = _fp8_is_cache(kv_cache)\n"
    "        assert (not _is_fp8) or _VLLM_FP8_INLINE, (\n"
    "            \"[dcp-diffkv] fp8 KV under DCP requires VLLM_FP8_INLINE=1 \"\n"
    "            \"(in-kernel descale); the fp8 SCRATCH fallback remaps block_table \"\n"
    "            \"per-call and is incompatible with the DCP context split.\"\n"
    "        )\n"
    "\n"
    "        max_q = int(attn_metadata.max_query_len)\n"
    "\n"
    "        # ---- private, replay-stable buffers (allocated ONCE in the builder;\n"
    "        #      here we just SLICE to the live token count). Piece (i).\n"
    "        num_heads = self.num_heads\n"
    "        h_all = num_heads * dcp_ws\n"
    "        # Context-phase segm buffers (head dim = num_heads*dcp_ws). Required\n"
    "        # to exist -- the builder allocates them when VLLM_DCP=1 & dcp_ws>1.\n"
    "        assert getattr(attn_metadata, \"dcp_context_softmax_segm_output\", None) \\\n"
    "            is not None, (\n"
    "            \"[dcp-diffkv] context segm buffers missing on metadata; the \"\n"
    "            \"builder must allocate dcp_context_softmax_segm_* (sized \"\n"
    "            \"capture_cap*dcp_world_size) when DCP is on.\"\n"
    "        )\n"
    "        n = num_actual_tokens\n"
    "        ctx_segm_out = attn_metadata.dcp_context_softmax_segm_output[:n]\n"
    "        ctx_segm_max = attn_metadata.dcp_context_softmax_segm_max[:n]\n"
    "        ctx_segm_exp = attn_metadata.dcp_context_softmax_segm_expsum[:n]\n"
    "\n"
    "        # Per-phase out + LSE scratch (also builder-owned, sliced to n). The\n"
    "        # context out is WIDE (h_all heads); query out is LOCAL (num_heads).\n"
    "        ctx_out = attn_metadata.dcp_context_out[:n]\n"
    "        ctx_lse = attn_metadata.dcp_context_lse[:n]\n"
    "        q_out = attn_metadata.dcp_query_out[:n]\n"
    "        q_lse = attn_metadata.dcp_query_lse[:n]\n"
    "\n"
    "        key_cache, value_cache, fp8_kwargs = self._dcp_slice_kv(\n"
    "            layer, kv_cache, head_size_qk, head_size_v\n"
    "        )\n"
    "        block_table = attn_metadata.block_table\n"
    "\n"
    "        # ========================= (1) CONTEXT PHASE =====================\n"
    "        # All-gather Q along the head dim so this rank computes attention for\n"
    "        # ALL ranks' query heads against ITS local context shard. dim=1 =\n"
    "        # heads (query is [num_tokens, num_heads, head_size_qk]).\n"
    "        q = query[:n].contiguous()\n"
    "        q_gathered = dcp_group.all_gather(q, dim=1)  # [n, h_all, head_size_qk]\n"
    "        # The reduce writes LSE as [num_tokens, num_query_heads] = [n, h_all].\n"
    "        # LIVE-BOOT (NOTES): seqused_k=dcp_context_kv_lens is the starting\n"
    "        # construction; the kernel is causal-only (context_len=seqused_k-\n"
    "        # query_len) so the exact cu_seqlens/query_len pairing that avoids\n"
    "        # clipping the last query_len shard positions must be pinned by the\n"
    "        # parity harness. Gated by VLLM_DCP_STAGE2_OK above.\n"
    "        unified_attention_diffkv(\n"
    "            q=q_gathered,\n"
    "            k=key_cache,\n"
    "            v=value_cache,\n"
    "            out=ctx_out,\n"
    "            cu_seqlens_q=attn_metadata.query_start_loc,\n"
    "            seqused_k=attn_metadata.dcp_context_kv_lens,  # LOCAL shard lengths\n"
    "            softmax_scale=self.scale,\n"
    "            causal=True,\n"
    "            alibi_slopes=self.alibi_slopes,\n"
    "            use_alibi_sqrt=self.use_alibi_sqrt,\n"
    "            window_size=self.sliding_window,\n"
    "            block_table=block_table,\n"
    "            softcap=self.logits_soft_cap,\n"
    "            sinks=self.sinks,\n"
    "            max_seqlen_q=max_q,\n"
    "            seq_threshold_3D=attn_metadata.seq_threshold_3D,\n"
    "            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,\n"
    "            softmax_segm_output=ctx_segm_out,\n"
    "            softmax_segm_max=ctx_segm_max,\n"
    "            softmax_segm_expsum=ctx_segm_exp,\n"
    "            out_lse=ctx_lse,\n"
    "            **fp8_kwargs,\n"
    "        )\n"
    "        # Combine across ranks: exact LSE-weighted merge. Input out=[B,H_all,D],\n"
    "        # lse=[B,H_all]; the DiffKV reduce already emits [B,H] so -- unlike the\n"
    "        # flash reference which transposes FA's [H,B] -> [B,H] here -- we pass\n"
    "        # ctx_lse straight in. Returns head-scattered ctx_out_c=[B,num_heads,D],\n"
    "        # ctx_lse_c=[B,num_heads].\n"
    "        ctx_out_c, ctx_lse_c = self.dcp_combine(\n"
    "            ctx_out,\n"
    "            ctx_lse,\n"
    "            dcp_group,\n"
    "            return_lse=True,\n"
    "            is_lse_base_on_e=self.lse_base_on_e,\n"
    "        )\n"
    "\n"
    "        # ========================= (2) QUERY-BLOCK PHASE =================\n"
    "        # Local self-attention over the new q_len tokens vs the K/V just\n"
    "        # written for THIS step (not split across ranks), mirroring the normal\n"
    "        # local decode attention and reusing the LOCAL (num_heads-wide) builder\n"
    "        # segm buffers via attn_metadata; out_lse=q_lse [B,num_heads].\n"
    "        # LIVE-BOOT (NOTES): this MUST attend ONLY positions\n"
    "        # [context_len, seq_len) (the new tokens) -- NOT the full sequence --\n"
    "        # or the context is double-counted under the merge. seqused_k below is\n"
    "        # the FULL seq_lens as a PLACEHOLDER for the harness to replace with\n"
    "        # the query-only construction (restricted block_table view + seqused_k=\n"
    "        # query_len). It is inert until VLLM_DCP_STAGE2_OK=1 (asserted above),\n"
    "        # so the wrong-until-pinned value can never run in prod or unvalidated.\n"
    "        unified_attention_diffkv(\n"
    "            q=q,\n"
    "            k=key_cache,\n"
    "            v=value_cache,\n"
    "            out=q_out,\n"
    "            cu_seqlens_q=attn_metadata.query_start_loc,\n"
    "            seqused_k=attn_metadata.seq_lens,  # LIVE-BOOT placeholder (see above)\n"
    "            softmax_scale=self.scale,\n"
    "            causal=True,\n"
    "            alibi_slopes=self.alibi_slopes,\n"
    "            use_alibi_sqrt=self.use_alibi_sqrt,\n"
    "            window_size=self.sliding_window,\n"
    "            block_table=block_table,\n"
    "            softcap=self.logits_soft_cap,\n"
    "            sinks=self.sinks,\n"
    "            max_seqlen_q=max_q,\n"
    "            seq_threshold_3D=attn_metadata.seq_threshold_3D,\n"
    "            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,\n"
    "            softmax_segm_output=attn_metadata.softmax_segm_output,\n"
    "            softmax_segm_max=attn_metadata.softmax_segm_max,\n"
    "            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,\n"
    "            out_lse=q_lse,\n"
    "            **fp8_kwargs,\n"
    "        )\n"
    "\n"
    "        # ========================= (3) MERGE (piece ii) ==================\n"
    "        # merge_attn_states wants [B,H,D] outputs and [H,B] LSEs. The combine\n"
    "        # returned ctx_lse_c=[B,num_heads] and our reduce wrote q_lse=[B,\n"
    "        # num_heads]; transpose both to [num_heads,B] + .contiguous() exactly\n"
    "        # as flash does (context_lse.transpose(0,1) at flash_attn.py:1096/1100).\n"
    "        ctx_lse_hb = ctx_lse_c.transpose(0, 1).contiguous()\n"
    "        q_lse_hb = q_lse.transpose(0, 1).contiguous()\n"
    "        assert ctx_out_c.shape == q_out.shape, (\n"
    "            f\"[dcp-diffkv] ctx/query out shape mismatch \"\n"
    "            f\"{tuple(ctx_out_c.shape)} vs {tuple(q_out.shape)}\"\n"
    "        )\n"
    "        assert ctx_lse_hb.shape == q_lse_hb.shape, (\n"
    "            f\"[dcp-diffkv] ctx/query lse shape mismatch \"\n"
    "            f\"{tuple(ctx_lse_hb.shape)} vs {tuple(q_lse_hb.shape)}\"\n"
    "        )\n"
    "        _dcp_merge_attn_states(\n"
    "            output[:n],\n"
    "            ctx_out_c,\n"
    "            ctx_lse_hb,\n"
    "            q_out,\n"
    "            q_lse_hb,\n"
    "        )\n"
    "        return output\n"
    "\n"
)
if src.count(fwd_def_anchor) != 1:
    print(f"[dcp-diffkv] FATAL: forward def anchor count={src.count(fwd_def_anchor)}"); sys.exit(3)
src = src.replace(fwd_def_anchor, dcp_method + fwd_def_anchor, 1)

# --- C2 (+ piece i): builder -> acquire the DCP handles, allocate the
#     CONTEXT-phase segm + out/lse buffers ONCE (cudagraph-safe, private,
#     replay-stable), and wrap build() to attach the local seq-len split
#     (mirrors flash_attn.py:522-542) plus stash the context buffers.
#
#     NOTE: unlike the FlashAttention *builder* (flash_attn.py:362-374), the
#     Triton parent TritonAttentionMetadataBuilder does NOT set dcp_world_size/
#     dcp_rank/cp_kv_cache_interleave_size on itself. So we acquire them here
#     from the DCP group / parallel_config; otherwise the build() split below
#     would silently no-op (getattr default 1) and DCP would never engage. This
#     whole block is gated on VLLM_DCP=1 & a live DCP group, so it is inert in
#     production (no group -> dcp_ws stays 1 -> nothing allocated).
builder_ctor_end = (
    "                logger.info(\n"
    "                    \"[diffkv-3d-qlen8] resized softmax_segm_* first-dim \"\n"
    "                    \"%d -> %d (max_num_seqs=%d * MAXQ=%d) for q_len<=8 3D path\",\n"
    "                    self.seq_threshold_3D, _d3q8_first, _d3q8_mns, _d3q8_maxq,\n"
    "                )\n"
)
builder_new = builder_ctor_end + (
    "\n"
    "        # DCP_DIFFKV (piece i): acquire the DCP handles the parent Triton\n"
    "        # builder does not set, then allocate the CONTEXT-phase scratch.\n"
    "        self.dcp_world_size = getattr(self, \"dcp_world_size\", 1)\n"
    "        self.dcp_rank = getattr(self, \"dcp_rank\", 0)\n"
    "        self.cp_kv_cache_interleave_size = getattr(\n"
    "            self, \"cp_kv_cache_interleave_size\", 1\n"
    "        )\n"
    "        try:\n"
    "            self.cp_kv_cache_interleave_size = (\n"
    "                vllm_config.parallel_config.cp_kv_cache_interleave_size\n"
    "            )\n"
    "        except (AttributeError, Exception):\n"
    "            pass\n"
    "        self.dcp_context_softmax_segm_output = None\n"
    "        self.dcp_context_softmax_segm_max = None\n"
    "        self.dcp_context_softmax_segm_expsum = None\n"
    "        self.dcp_context_out = None\n"
    "        self.dcp_context_lse = None\n"
    "        self.dcp_query_out = None\n"
    "        self.dcp_query_lse = None\n"
    "        if _os_dcp.environ.get(\"VLLM_DCP\", \"0\") == \"1\":\n"
    "            _dcp_grp = _dcp_group_or_none()\n"
    "            if _dcp_grp is not None and _dcp_grp.world_size > 1:\n"
    "                self.dcp_world_size = _dcp_grp.world_size\n"
    "                self.dcp_rank = _dcp_grp.rank_in_group\n"
    "                # First-dim MUST cover the max token count of a decode/verify\n"
    "                # batch (num_seqs * q_len), same reasoning as the diffkv-3d-\n"
    "                # qlen8 resize. Use MAXQ=8 (DFlash verify ceiling); q_len==1\n"
    "                # decode is covered by the max().\n"
    "                _dcp_maxq = 8\n"
    "                _dcp_mns = int(\n"
    "                    getattr(vllm_config.scheduler_config, \"max_num_seqs\", 0)\n"
    "                    or 0\n"
    "                )\n"
    "                _dcp_first = max(self.seq_threshold_3D, _dcp_mns * _dcp_maxq)\n"
    "                _hsv = TritonAttentionDiffKVBackend.head_size_v\n"
    "                _hsv_pad = next_power_of_2(_hsv)\n"
    "                _nseg = self.num_par_softmax_segments\n"
    "                _h_local = self.num_heads_q\n"
    "                _h_all = self.num_heads_q * self.dcp_world_size\n"
    "                # CONTEXT segm buffers: head dim widened to num_heads_q*dcp_ws\n"
    "                # (the Q-all-gather count). fp32, contiguous -> the reduce\n"
    "                # kernel indexes them by (token, head, seg, hsv_padded).\n"
    "                self.dcp_context_softmax_segm_output = torch.empty(\n"
    "                    (_dcp_first, _h_all, _nseg, _hsv_pad),\n"
    "                    dtype=torch.float32, device=device,\n"
    "                )\n"
    "                self.dcp_context_softmax_segm_max = torch.empty(\n"
    "                    (_dcp_first, _h_all, _nseg),\n"
    "                    dtype=torch.float32, device=device,\n"
    "                )\n"
    "                self.dcp_context_softmax_segm_expsum = torch.empty(\n"
    "                    (_dcp_first, _h_all, _nseg),\n"
    "                    dtype=torch.float32, device=device,\n"
    "                )\n"
    "                # Per-phase attention out + LSE scratch (model dtype for out,\n"
    "                # fp32 for LSE). Context out is WIDE (h_all); query is LOCAL.\n"
    "                _mdl_dtype = vllm_config.model_config.dtype\n"
    "                self.dcp_context_out = torch.empty(\n"
    "                    (_dcp_first, _h_all, _hsv),\n"
    "                    dtype=_mdl_dtype, device=device,\n"
    "                )\n"
    "                self.dcp_context_lse = torch.empty(\n"
    "                    (_dcp_first, _h_all),\n"
    "                    dtype=torch.float32, device=device,\n"
    "                )\n"
    "                self.dcp_query_out = torch.empty(\n"
    "                    (_dcp_first, _h_local, _hsv),\n"
    "                    dtype=_mdl_dtype, device=device,\n"
    "                )\n"
    "                self.dcp_query_lse = torch.empty(\n"
    "                    (_dcp_first, _h_local),\n"
    "                    dtype=torch.float32, device=device,\n"
    "                )\n"
    "                logger.info(\n"
    "                    \"[dcp-diffkv] allocated CONTEXT-phase scratch: segm \"\n"
    "                    \"first=%d h_all=%d nseg=%d hsv_pad=%d; ctx_out[%d,%d,%d] \"\n"
    "                    \"(dcp_world_size=%d, num_heads_q=%d)\",\n"
    "                    _dcp_first, _h_all, _nseg, _hsv_pad,\n"
    "                    _dcp_first, _h_all, _hsv,\n"
    "                    self.dcp_world_size, self.num_heads_q,\n"
    "                )\n"
    "\n"
    "    # DCP_DIFFKV: attach the Decode Context Parallelism metadata (local\n"
    "    # context-KV seq-len split + max over ranks) to each built metadata,\n"
    "    # mirroring flash_attn.py:522-542, and stash the context-phase scratch\n"
    "    # buffers so _forward_with_dcp reads them off attn_metadata. No-op off.\n"
    "    def build(self, *args, **kwargs):\n"
    "        md = super().build(*args, **kwargs)\n"
    "        _dcp_ws = getattr(self, \"dcp_world_size\", 1)\n"
    "        if _os_dcp.environ.get(\"VLLM_DCP\", \"0\") == \"1\" and _dcp_ws > 1:\n"
    "            _qsl = md.query_start_loc\n"
    "            _query_lens = _qsl[1:] - _qsl[:-1]\n"
    "            _context_kv_lens = md.seq_lens - _query_lens\n"
    "            _interleave = getattr(self, \"cp_kv_cache_interleave_size\", 1)\n"
    "            _local = _dcp_local_seq_lens(\n"
    "                _context_kv_lens, _dcp_ws,\n"
    "                getattr(self, \"dcp_rank\", 0), _interleave,\n"
    "            )\n"
    "            md.dcp_context_kv_lens = _local\n"
    "            _num_parts = _dcp_ws * _interleave\n"
    "            md.max_dcp_context_kv_len = (\n"
    "                (md.max_seq_len + _num_parts - 1) // _num_parts\n"
    "            ) * _interleave\n"
    "            # Stash the pre-allocated context-phase scratch (piece i) so the\n"
    "            # impl reads replay-stable buffers off the metadata. These are\n"
    "            # the SAME tensors every build -> cudagraph-safe.\n"
    "            md.dcp_context_softmax_segm_output = self.dcp_context_softmax_segm_output\n"
    "            md.dcp_context_softmax_segm_max = self.dcp_context_softmax_segm_max\n"
    "            md.dcp_context_softmax_segm_expsum = self.dcp_context_softmax_segm_expsum\n"
    "            md.dcp_context_out = self.dcp_context_out\n"
    "            md.dcp_context_lse = self.dcp_context_lse\n"
    "            md.dcp_query_out = self.dcp_query_out\n"
    "            md.dcp_query_lse = self.dcp_query_lse\n"
    "        return md\n"
)
if src.count(builder_ctor_end) != 1:
    print(f"[dcp-diffkv] FATAL: builder ctor-end anchor count={src.count(builder_ctor_end)}"); sys.exit(3)
src = src.replace(builder_ctor_end, builder_new, 1)

ast.parse(src)
p.write_text(src)
print("[dcp-diffkv] backend DCP wiring applied (LSE flag + dcp_combine + dispatch + stub forward + builder split)")
PY
[ $? -eq 0 ] || { echo "[dcp-diffkv] backend DCP wiring FAILED — aborting"; exit 1; }


# ----------------------------------------------------------------------------
# clean bytecode + validate: parse+import both modules, assert markers set,
# assert the LSE store is byte-for-byte off by default (RETURN_LSE=(out_lse is
# not None)), and assert VLLM_DCP toggles the module gate.
# ----------------------------------------------------------------------------
find "$SITE/vllm/v1/attention" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import ast, os, importlib
K = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
B = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py"
P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn.py"
ks, bs, ps = open(K).read(), open(B).read(), open(P).read()
ast.parse(ks); ast.parse(bs); ast.parse(ps)
for name, s in (("kernel", ks), ("backend", bs), ("parent", ps)):
    assert "DCP_DIFFKV" in s, f"{name} marker missing"
# kernel: LSE store present + guarded, wrapper param present + default-off launch
assert "RETURN_LSE: tl.constexpr," in ks, "kernel RETURN_LSE param missing"
assert "tl.log(overall_expsum) + overall_max" in ks, "kernel LSE formula missing"
assert "RETURN_LSE=(out_lse is not None)," in ks, "kernel launch RETURN_LSE gate missing"
assert "out_lse: torch.Tensor | None = None," in ks, "wrapper out_lse param missing"
# backend: flag, combine, dispatch, stub, builder split
assert "can_return_lse_for_decode: bool = True" in bs, "impl LSE flag missing"
assert "self.dcp_combine = _dcp_a2a_lse_reduce if _a2a else _cp_lse_ag_out_rs" in bs, "dcp_combine init missing"
assert "return self._forward_with_dcp(" in bs, "DCP dispatch missing"
assert "def _forward_with_dcp(" in bs, "_forward_with_dcp missing"
assert "def build(self, *args, **kwargs):" in bs, "builder DCP build override missing"
assert "md.dcp_context_kv_lens = _local" in bs, "builder DCP metadata attach missing"
# stage-2: real forward body (not a bare NotImplementedError), all 4 pieces.
# Scope the NotImplementedError check to _forward_with_dcp ONLY (the impl's
# __init__/forward keep their STOCK fp8/fused-quant guards, which are fine).
_bt = ast.parse(bs)
_impl_cls = next(n for n in ast.walk(_bt)
                 if isinstance(n, ast.ClassDef) and n.name == "TritonAttentionDiffKVImpl")
_fwd_dcp = next(m for m in _impl_cls.body
                if isinstance(m, ast.FunctionDef) and m.name == "_forward_with_dcp")
_ni = [r for r in ast.walk(_fwd_dcp) if isinstance(r, ast.Raise)
       and getattr(getattr(r.exc, "func", None), "id", None) == "NotImplementedError"]
assert len(_ni) == 0, \
    "_forward_with_dcp must NOT raise NotImplementedError in the executed path"
assert "q_gathered = dcp_group.all_gather(q, dim=1)" in bs, "context-phase Q all-gather missing (piece 1)"
assert "out_lse=ctx_lse," in bs, "context-phase out_lse threading missing (piece 1)"
assert "ctx_out_c, ctx_lse_c = self.dcp_combine(" in bs, "dcp_combine call missing (piece 1)"
assert "out_lse=q_lse," in bs, "query-phase out_lse threading missing (piece 3)"
assert "ctx_lse_hb = ctx_lse_c.transpose(0, 1).contiguous()" in bs, "merge LSE transpose missing (piece ii)"
assert "_dcp_merge_attn_states(" in bs, "merge_attn_states call missing (piece ii)"
assert 'logger.info_once(' in bs and "running DCP stage-2 (TEST path)" in bs, "loud TEST-path log missing"
assert "VLLM_DCP_STAGE2_OK" in bs, "stage-2 acknowledgment guard missing"
# builder context-phase scratch (piece i): allocated once, sized *dcp_world_size
assert "self.dcp_context_softmax_segm_output = torch.empty(" in bs, "builder context segm alloc missing (piece i)"
assert "_h_all = self.num_heads_q * self.dcp_world_size" in bs, "context segm head-widening missing (piece i)"
assert "md.dcp_context_softmax_segm_output = self.dcp_context_softmax_segm_output" in bs, "builder context segm stash missing (piece i)"
# parent dataclass fields
assert "dcp_context_kv_lens: torch.Tensor | None = None" in ps, "parent dcp field missing"
assert "max_dcp_context_kv_len: int | None = None" in ps, "parent dcp max field missing"
assert "dcp_context_softmax_segm_output: torch.Tensor | None = None" in ps, "parent context segm field missing"
assert "dcp_query_lse: torch.Tensor | None = None" in ps, "parent query-lse field missing"

# import both modules (default env: VLLM_DCP unset -> disabled dispatch)
os.environ.pop("VLLM_DCP", None)
import vllm.v1.attention.ops.triton_unified_attention_diffkv as _k
import vllm.v1.attention.backends.triton_attn_diffkv as _b
importlib.reload(_k); importlib.reload(_b)
assert _b._VLLM_DCP is False, f"default (env off) must give _VLLM_DCP False, got {_b._VLLM_DCP}"
os.environ["VLLM_DCP"] = "1"
importlib.reload(_b)
assert _b._VLLM_DCP is True, f"env on must give _VLLM_DCP True, got {_b._VLLM_DCP}"
os.environ.pop("VLLM_DCP", None)
importlib.reload(_b)
print("[dcp-diffkv] validation OK (all 3 files parse+import; LSE store default-off; VLLM_DCP flips gate; stage-2 forward + context scratch present; markers set)")
PY
[ $? -eq 0 ] || { echo "[dcp-diffkv] validation FAILED — aborting"; exit 1; }

echo "[dcp-diffkv] done. Stage-2 (_forward_with_dcp + context-phase scratch) DRAFTED, default OFF."
echo "[dcp-diffkv] To ARM: recipe adds --decode-context-parallel-size 2 [--dcp-comm-backend a2a]"
echo "[dcp-diffkv]   + env VLLM_DCP=1 (and VLLM_DIFFKV_3D_Q8=1 for DFlash q_len=8, VLLM_FP8_INLINE=1)."
echo "[dcp-diffkv] GATE FIRST (single boot): /home/admin/dcp_parity.py (offline GPU) MUST pass;"
echo "[dcp-diffkv]   bringup enforce-eager; the drafted _forward_with_dcp needs VLLM_DCP_STAGE2_OK=1"
echo "[dcp-diffkv]   set ONLY AFTER the context/query seqused_k constructions are pinned by the"
echo "[dcp-diffkv]   parity harness + shadow A/B. See mods/dcp-diffkv/NOTES.md LIVE-BOOT CHECKLIST."