#!/bin/bash
set -uo pipefail
# ============================================================================
# diffkv-3d-qlen8 — let the DiffKV Triton decode take the 3D split-KV
#   (FlashDecoding) path for DFlash verify (q_len up to 8), gated behind
#   VLLM_DIFFKV_3D_Q8 (default 0 = exact current behavior).
#
#   Image: vllm-node-mimo-v25-upstream:latest
#   vllm 0.23.1rc1.dev760+g3775d5fca (main @3775d5fc)
#   Target: MiMo-V2.5 fp8-KV DiffKV decode on 2x GB10 / sm_121a.
#
# THE PROBLEM (verified on this image):
#   unified_attention_diffkv() chooses 2D vs 3D via
#       use_3d = not ( ... or max_seqlen_q > 1 or num_seqs > seq_threshold_3D
#                      or is_batch_invariant )
#   DFlash proposes num_speculative_tokens=7 -> the verify step runs q_len=8,
#   so max_query_len=8 -> `max_seqlen_q > 1` is TRUE -> use_3d = False -> the
#   SLOW 2D path (one program per (q-block, kv-head) walking the whole KV
#   sequence, badly under-fills 48 SMs on deep context). The 3D path splits KV
#   into num_par_softmax_segments and saturates the GPU. Deep-ctx decode ~3x slow.
#
# WHY q_len=8 IS ACTUALLY SAFE FOR THE 3D KERNEL (see FINDINGS):
#   The 3D kernel + kernel_reduce_segments_diffkv already handle BLOCK_Q
#   q-blocks: causal mask uses per-row query_abs_pos, sliding-window pruning
#   uses [qpos_lo,qpos_hi] across the q-block, sinks fold in once via
#   init_softmax_M/reduce. For MiMo num_queries_per_kv=16 -> BLOCK_M=16 ->
#   BLOCK_Q=1, so each program covers ONE query row; q_len=8 just means 8x as
#   many q-block programs, each mapped to its own token by query_offset_0.
#   No kernel-body math change is needed for q_len<=8.
#
#   The ONE real hazard: the per-segment scratch buffers
#     softmax_segm_output : [seq_threshold_3D, num_heads_q, segments, hsv_padded]
#     softmax_segm_max     : [seq_threshold_3D, num_heads_q, segments]
#     softmax_segm_expsum  : [seq_threshold_3D, num_heads_q, segments]
#   are indexed in the epilogue by the ABSOLUTE TOKEN index query_offset_0
#   (0..num_tokens-1) and by the reduce grid dim0 = q.shape[0] (num_tokens),
#   but their first dim is sized to seq_threshold_3D (a count of SEQUENCES).
#   At q_len=1 num_tokens==num_seqs<=seq_threshold_3D so it fits; at q_len=8
#   num_tokens = num_seqs*8 can exceed seq_threshold_3D -> out-of-bounds write.
#   For THIS config it happens to fit exactly (seq_threshold_3D = 128//2 = 64
#   snapped to capture-size 64; mns*qlen = 8*8 = 64), but with ZERO margin and
#   ONLY by coincidence. => We MUST resize the first dim to cover num_seqs*qlen.
#
# WHAT THIS MOD DOES (two edits, both marker-guarded / ast-checked / idempotent,
# both env-gated so default is a byte-for-byte no-op):
#   1. KERNEL GATE (triton_unified_attention_diffkv.py): relax the disqualifier
#        or max_seqlen_q > 1
#      to
#        or max_seqlen_q > _DIFFKV_3D_Q8_MAXQ          # 8 when enabled else 1
#      plus module-level flags  _VLLM_DIFFKV_3D_Q8 = env=="1"  and
#        _DIFFKV_3D_Q8_MAXQ = 8 if _VLLM_DIFFKV_3D_Q8 else 1.
#   2. SEGM BUFFER RESIZE (triton_attn_diffkv.py DiffKV builder __init__): after
#      super().__init__() (which allocated all three at seq_threshold_3D) and the
#      mod's own softmax_segm_output re-alloc, when VLLM_DIFFKV_3D_Q8=1 re-allocate
#      all three with first dim  max(seq_threshold_3D, max_num_seqs * MAXQ)  so
#      the token index never overflows. Cheap (~17-34 MB fp32, once, at init).
#      When the env is off, buffers are left exactly as allocated (no-op).
#
# SEGMENTS: num_par_softmax_segments stays 16 (NUM_PAR_SOFTMAX_SEGMENTS). Raising
#   it (fable's 16->32 long-ctx idea) is a PERF knob that linearly grows all three
#   buffers + the 3D grid + needs the reduce constexpr to match, and it widens the
#   sm_121 Triton-codegen risk surface. NOT changed here (documented). Revisit via
#   a separate env only after the parity harness + shadow A/B prove q_len=8 is
#   bit-safe.
#
# TEST GATE (sm_121 Triton has a 3x history of silent garbage): DO NOT trust this
#   without /home/admin/diffkv_3d_parity.py PASSing inside the container first,
#   then a shadow A/B (VLLM_DIFFKV_3D_Q8=1 on a scratch port), THEN a 240K needle
#   + reliability run. This mod only makes the path REACHABLE; it does not prove
#   the sm_121 kernel is correct at q_len=8.
#
# ORDERING: run AFTER fp8-kv-inline (the kernel file is by then the fp8-patched
#   stock-upstream triton_unified_attention_diffkv.py). The two anchors this mod
#   uses — the `use_3d = not (...)` gate block and the DiffKV builder's
#   softmax_segm_output alloc — are NOT modified by fp8-kv-inline, so the anchors
#   match in both stock and fp8-patched states.
#
# Discipline: marker DIFFKV_3D_QLEN8, ast.parse after each edit, sys.exit on
#   anchor-miss, idempotent (no-op if markers already present).
# ============================================================================

SITE="/usr/local/lib/python3.12/dist-packages"
KERNEL="$SITE/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
BACKEND="$SITE/vllm/v1/attention/backends/triton_attn_diffkv.py"

echo "[diffkv-3d-qlen8] enabling q_len<=8 -> 3D split-KV path (VLLM_DIFFKV_3D_Q8 gate, default off)"

# ----------------------------------------------------------------------------
# 0. Sanity: both files present with the expected load-bearing anchors.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import sys, pathlib
K = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
B = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py")
if not K.exists():
    print(f"[diffkv-3d-qlen8] FATAL: kernel file missing: {K}"); sys.exit(3)
if not B.exists():
    print(f"[diffkv-3d-qlen8] FATAL: backend file missing: {B}"); sys.exit(3)
ks = K.read_text(); bs = B.read_text()
if "\nis_batch_invariant = envs.VLLM_BATCH_INVARIANT\n" not in ks:
    print("[diffkv-3d-qlen8] FATAL: kernel `is_batch_invariant` module-global anchor missing"); sys.exit(3)
if ks.count("        or max_seqlen_q > 1\n") != 1:
    print("[diffkv-3d-qlen8] FATAL: kernel gate anchor `or max_seqlen_q > 1` count != 1"); sys.exit(3)
if "    use_3d = not (\n" not in ks:
    print("[diffkv-3d-qlen8] FATAL: kernel `use_3d = not (` anchor missing"); sys.exit(3)
if "class TritonAttentionDiffKVMetadataBuilder(TritonAttentionMetadataBuilder):" not in bs:
    print("[diffkv-3d-qlen8] FATAL: DiffKV builder class anchor missing"); sys.exit(3)
if bs.count("        self.softmax_segm_output = torch.empty(\n") != 1:
    print("[diffkv-3d-qlen8] FATAL: DiffKV builder softmax_segm_output alloc anchor count != 1"); sys.exit(3)
print("[diffkv-3d-qlen8] sanity OK (kernel gate + module-global + backend builder anchors present)")
PY
[ $? -eq 0 ] || { echo "[diffkv-3d-qlen8] sanity FAILED — aborting"; exit 1; }


# ============================================================================
# 1. KERNEL GATE relax (env-gated).
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
src = p.read_text()

if "DIFFKV_3D_QLEN8" in src:
    print("[diffkv-3d-qlen8] kernel gate already patched"); sys.exit(0)

glob_anchor = "\nis_batch_invariant = envs.VLLM_BATCH_INVARIANT\n"
if src.count(glob_anchor) != 1:
    print(f"[diffkv-3d-qlen8] FATAL: is_batch_invariant global anchor count = {src.count(glob_anchor)} (expected 1)"); sys.exit(3)
glob_block = (
    "\nis_batch_invariant = envs.VLLM_BATCH_INVARIANT\n"
    "\n"
    "# DIFFKV_3D_QLEN8: allow the 3D split-KV (FlashDecoding) path for DFlash\n"
    "# verify (q_len up to 8), not just pure decode (q_len==1). Default OFF ->\n"
    "# _DIFFKV_3D_Q8_MAXQ == 1 -> the gate below is byte-for-byte the stock\n"
    "# `max_seqlen_q > 1`. When VLLM_DIFFKV_3D_Q8=1 the disqualifier becomes\n"
    "# `max_seqlen_q > 8`, so q_len in [2..8] takes 3D. The 3D kernel already\n"
    "# handles q-blocks (see mod FINDINGS); the segm scratch buffers are resized\n"
    "# to cover num_seqs*qlen token indices by mods/diffkv-3d-qlen8 in the\n"
    "# TritonAttentionDiffKVMetadataBuilder. MUST stay in lockstep with that\n"
    "# resize: never enable this env without the buffer resize (same env gates\n"
    "# both) or the epilogue will write out-of-bounds.\n"
    "import os as _os_diffkv3d\n"
    "_VLLM_DIFFKV_3D_Q8 = _os_diffkv3d.environ.get(\"VLLM_DIFFKV_3D_Q8\", \"0\") == \"1\"\n"
    "# Max q_len still eligible for 3D. 1 = stock behavior (only pure decode).\n"
    "_DIFFKV_3D_Q8_MAXQ = 8 if _VLLM_DIFFKV_3D_Q8 else 1\n"
)
src = src.replace(glob_anchor, glob_block, 1)

old_gate = "        or max_seqlen_q > 1\n"
new_gate = "        or max_seqlen_q > _DIFFKV_3D_Q8_MAXQ  # DIFFKV_3D_QLEN8 (8 when enabled, else 1)\n"
if src.count(old_gate) != 1:
    print(f"[diffkv-3d-qlen8] FATAL: gate disqualifier anchor count = {src.count(old_gate)} (expected 1)"); sys.exit(3)
src = src.replace(old_gate, new_gate, 1)

ast.parse(src)
p.write_text(src)
print("[diffkv-3d-qlen8] kernel gate relaxed (env-gated: max_seqlen_q > _DIFFKV_3D_Q8_MAXQ)")
PY
[ $? -eq 0 ] || { echo "[diffkv-3d-qlen8] kernel gate patch FAILED — aborting"; exit 1; }


# ============================================================================
# 2. SEGM BUFFER RESIZE in the DiffKV builder __init__.
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py")
src = p.read_text()

if "DIFFKV_3D_QLEN8" in src:
    print("[diffkv-3d-qlen8] backend segm resize already patched"); sys.exit(0)

anchor = (
    "        self.softmax_segm_output = torch.empty(\n"
    "            (\n"
    "                self.seq_threshold_3D,\n"
    "                self.num_heads_q,\n"
    "                self.num_par_softmax_segments,\n"
    "                head_size_v_padded,\n"
    "            ),\n"
    "            dtype=torch.float32,\n"
    "            device=device,\n"
    "        )\n"
)
if src.count(anchor) != 1:
    print(f"[diffkv-3d-qlen8] FATAL: DiffKV softmax_segm_output alloc block anchor count = {src.count(anchor)} (expected 1)"); sys.exit(3)

resize = (
    "\n"
    "        # DIFFKV_3D_QLEN8: when VLLM_DIFFKV_3D_Q8=1, the kernel gate lets\n"
    "        # q_len<=8 (DFlash verify) take the 3D split-KV path. The 3D epilogue\n"
    "        # and kernel_reduce_segments_diffkv index the segm buffers by the\n"
    "        # ABSOLUTE token index (0..num_tokens-1), but the parent sized their\n"
    "        # first dim to seq_threshold_3D (a SEQUENCE count). At q_len=q, a\n"
    "        # decode+verify batch has up to max_num_seqs*q tokens, which can\n"
    "        # exceed seq_threshold_3D -> out-of-bounds write. Re-allocate all\n"
    "        # three with first dim = max(seq_threshold_3D, max_num_seqs*MAXQ).\n"
    "        # Cheap (~17-34 MB fp32, once at init). When the env is off this is a\n"
    "        # no-op (MAXQ==1 and max_num_seqs*1 <= seq_threshold_3D in every\n"
    "        # decode-eligible batch, so the max() keeps the stock size).\n"
    "        import os as _os_d3q8\n"
    "        if _os_d3q8.environ.get(\"VLLM_DIFFKV_3D_Q8\", \"0\") == \"1\":\n"
    "            _d3q8_maxq = 8\n"
    "            _d3q8_mns = int(\n"
    "                getattr(vllm_config.scheduler_config, \"max_num_seqs\", 0) or 0\n"
    "            )\n"
    "            _d3q8_first = max(self.seq_threshold_3D, _d3q8_mns * _d3q8_maxq)\n"
    "            if _d3q8_first > self.seq_threshold_3D:\n"
    "                self.softmax_segm_output = torch.empty(\n"
    "                    (\n"
    "                        _d3q8_first,\n"
    "                        self.num_heads_q,\n"
    "                        self.num_par_softmax_segments,\n"
    "                        head_size_v_padded,\n"
    "                    ),\n"
    "                    dtype=torch.float32,\n"
    "                    device=device,\n"
    "                )\n"
    "                self.softmax_segm_max = torch.empty(\n"
    "                    (_d3q8_first, self.num_heads_q, self.num_par_softmax_segments),\n"
    "                    dtype=torch.float32,\n"
    "                    device=device,\n"
    "                )\n"
    "                self.softmax_segm_expsum = torch.empty(\n"
    "                    (_d3q8_first, self.num_heads_q, self.num_par_softmax_segments),\n"
    "                    dtype=torch.float32,\n"
    "                    device=device,\n"
    "                )\n"
    "                logger.info(\n"
    "                    \"[diffkv-3d-qlen8] resized softmax_segm_* first-dim \"\n"
    "                    \"%d -> %d (max_num_seqs=%d * MAXQ=%d) for q_len<=8 3D path\",\n"
    "                    self.seq_threshold_3D, _d3q8_first, _d3q8_mns, _d3q8_maxq,\n"
    "                )\n"
)
src = src.replace(anchor, anchor + resize, 1)

ast.parse(src)
p.write_text(src)
print("[diffkv-3d-qlen8] backend segm buffers resized for q_len<=8 (env-gated)")
PY
[ $? -eq 0 ] || { echo "[diffkv-3d-qlen8] backend segm resize patch FAILED — aborting"; exit 1; }


# ----------------------------------------------------------------------------
# clean bytecode + validate.
# ----------------------------------------------------------------------------
find "$SITE/vllm/v1/attention" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import ast, os
K = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
B = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py"
ks = open(K).read(); bs = open(B).read()
ast.parse(ks); ast.parse(bs)
assert "DIFFKV_3D_QLEN8" in ks, "kernel marker missing"
assert "DIFFKV_3D_QLEN8" in bs, "backend marker missing"
assert "_DIFFKV_3D_Q8_MAXQ" in ks, "kernel MAXQ flag missing"
assert "or max_seqlen_q > _DIFFKV_3D_Q8_MAXQ" in ks, "kernel gate not swapped"
import vllm.v1.attention.ops.triton_unified_attention_diffkv as _k  # noqa: F401
import vllm.v1.attention.backends.triton_attn_diffkv as _b  # noqa: F401
import importlib
os.environ.pop("VLLM_DIFFKV_3D_Q8", None)
importlib.reload(_k)
assert _k._DIFFKV_3D_Q8_MAXQ == 1, f"default (env off) must give MAXQ==1, got {_k._DIFFKV_3D_Q8_MAXQ}"
os.environ["VLLM_DIFFKV_3D_Q8"] = "1"
importlib.reload(_k)
assert _k._DIFFKV_3D_Q8_MAXQ == 8, f"env on must give MAXQ==8, got {_k._DIFFKV_3D_Q8_MAXQ}"
os.environ.pop("VLLM_DIFFKV_3D_Q8", None)
importlib.reload(_k)
print("[diffkv-3d-qlen8] validation OK (both files parse+import; MAXQ 1<->8 flips with env; markers set)")
PY
[ $? -eq 0 ] || { echo "[diffkv-3d-qlen8] validation FAILED — aborting"; exit 1; }

echo "[diffkv-3d-qlen8] done (enable with VLLM_DIFFKV_3D_Q8=1; default=stock 2D for q_len>1)"
