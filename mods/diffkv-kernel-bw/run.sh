#!/bin/bash
set -uo pipefail
# ============================================================================
# diffkv-kernel-bw — fix parallelism-starvation in the DiffKV 3D split-KV
#   (FlashDecoding) decode attention kernel, so the deep-context read runs at
#   BANDWIDTH instead of memory-LATENCY. Gated behind VLLM_DIFFKV_BW
#   (default 0 = byte-for-byte the current kernel).
#
#   Image: vllm-node-mimo-v25-upstream:latest
#   vllm 0.23.1rc1.dev760+g3775d5fca (main @3775d5fc)
#   Target: MiMo-V2.5 fp8-KV DiffKV decode on 2x GB10 / sm_121a.
#
# THE DIAGNOSIS (fable, code-grounded, verified against the live container):
#   At deep context (400K), q_len=1, bs=1, the 3D split-KV DiffKV decode kernel
#   reads at only ~21% of GB10's 273 GB/s. Root: NUM_PAR_SOFTMAX_SEGMENTS = 16
#   is a hard constant, so the 3D grid = 1 q-block x 2 kv-heads x 16 segments =
#   32 CTAs on 48 SMs. Each SM holds ONE CTA -> the kernel runs at memory
#   LATENCY, not bandwidth (too few memory transactions in flight). Bytes and
#   coalescing are FINE; the problem is grid occupancy. Raising segments 16->64
#   makes grid = 1 x 2 x 64 = 128 CTAs -> ~2.7 CTAs/SM -> latency is hidden by
#   concurrency -> the attention read approaches bandwidth (~2.5-3.5x on the read).
#
# THE TWO FIXES (both env-gated, default a byte-for-byte no-op):
#   K1 (BIG WIN, cudagraph-safe/low-risk) -- NUM_PAR_SOFTMAX_SEGMENTS 16 -> 64:
#     Change the module constant in triton_attn.py (line 56). MUST stay a power
#     of two: kernel_reduce_segments_diffkv indexes segments with
#     `tl.arange(0, NUM_SEGMENTS_PER_SEQ)` (kernel lines 374/384/397) and the
#     epilogue strides by NUM_SEGMENTS_PER_SEQ (lines 303/304/395/396). A
#     power-of-two count keeps tl.arange masks/strides exact. The env
#     VLLM_DIFFKV_SEGMENTS (default 64) lets us sweep 32/64/128 without a rebuild;
#     it is REJECTED (sys.exit 3) unless it is a power of two in [1,256].
#
#     >>> WHY NO BUFFER-ALLOC EDIT IS NEEDED (the composition, verified): <<<
#     The three per-segment scratch buffers softmax_segm_{output,max,expsum} are
#     ALL allocated symbolically from `self.num_par_softmax_segments`, which is
#     set ONCE to `NUM_PAR_SOFTMAX_SEGMENTS` (triton_attn.py:154). Every alloc
#     site references that member, never a literal:
#       - base builder            triton_attn.py:156-175  (all three)
#       - DiffKV builder re-alloc  triton_attn_diffkv.py:116-125 (output, V-shaped)
#       - diffkv-3d-qlen8 resize   triton_attn_diffkv.py:146-165 (all three, q_len=8)
#     Raising the CONSTANT flows into ALL of them + the 3D grid dim2 (kernel:491)
#     + the kernel's NUM_SEGMENTS_PER_SEQ constexpr (kernel:549/573). So K1 is a
#     ONE-LINE constant bump with ZERO risk of double-count or under-size: the
#     segment dimension is a single symbolic source of truth. The final buffer
#     shape when composed with diffkv-3d-qlen8 is:
#       softmax_segm_output : [max(seq_threshold_3D, mns*8), num_heads_q, 64, hv_pad]
#       softmax_segm_max/eps : [max(seq_threshold_3D, mns*8), num_heads_q, 64]
#     i.e. the qlen8 first-dim (max_num_seqs*8) and the K1 segment dim (64) are
#     ORTHOGONAL axes; each mod owns exactly one. (See NOTES.md for the byte math;
#     ~+150 MB/rank total for the segment dim x4 at the PROD geometry.)
#
#   K2 (low-med risk) -- 3D-path TILE_SIZE 16 -> 32 when the KV cache is fp8:
#     Kernel line 487 picks
#       tile_size = 32 if not use_3d else (16 if q.element_size() >= 2 else 32)
#     keying on the QUERY dtype (bf16 -> element_size 2 -> always 16). But the KV
#     read is what dominates the deep-decode loop, and the fp8 KV cache is uint8
#     (element_size 1). Keying TILE_SIZE on the KV element size lets fp8 use
#     TILE_SIZE=32 -> HALF the KV loop iterations. `k` (the key_cache view) is in
#     scope at this point (num_kv_heads = k.shape[2], line 457). TILE_SIZE is a
#     tl.constexpr threaded IDENTICALLY into BOTH the main kernel (line ~549) and
#     kernel_reduce_segments_diffkv (line 568) from this ONE variable, so the two
#     always agree. Within one deployment the KV dtype is fixed -> TILE_SIZE is a
#     stable per-deployment constant -> cudagraph capture sees fixed shapes.
#     Bf16-KV (drafter / non-fp8) is UNTOUCHED: element_size 2 -> still 16.
#
# GATING:
#   VLLM_DIFFKV_BW unset/0  -> NUM_PAR_SOFTMAX_SEGMENTS stays 16 AND the tile_size
#                              expression is byte-for-byte the stock line -> a
#                              TRUE no-op (safe to ship inert in a recipe).
#   VLLM_DIFFKV_BW=1        -> K1 (segments = VLLM_DIFFKV_SEGMENTS, default 64) +
#                              K2 (fp8 -> TILE_SIZE 32).
#   VLLM_DIFFKV_SEGMENTS=N  -> sweep the segment count (power-of-two, 1..256).
#                              Only consulted when VLLM_DIFFKV_BW=1.
#
# ORDERING: run AFTER fp8-kv-inline AND diffkv-3d-qlen8. The two anchors this mod
#   touches are NOT modified by either of those:
#     - triton_attn.py:56 `NUM_PAR_SOFTMAX_SEGMENTS = 16 ...` is stock (neither
#       fp8-kv-inline nor diffkv-3d-qlen8 edits triton_attn.py; both reference the
#       member symbolically).
#     - kernel:487 tile_size line is stock (fp8-kv-inline wraps the K/V LOAD and
#       the signature, not the tile_size selection; diffkv-3d-qlen8 touches only
#       the use_3d gate + backend buffers).
#   Verified: both anchors present exactly once in the LIVE post-fp8/post-3dq8
#   files (docker cp of vllm_node, markers FP8_KV_INLINE_* + DIFFKV_3D_QLEN8 present).
#
# TEST GATE (sm_121 Triton has a 3x history of silent garbage; DO NOT trust
#   without these, run when the cluster is idle -- NOT beside PROD at util 0.86):
#   (a) offline 2D-vs-3D parity /home/admin/diffkv_3d_parity.py at segments 64
#       (set NUM_PAR_SOFTMAX_SEGMENTS=64 in that harness OR run the container with
#       VLLM_DIFFKV_BW=1 and import; numeric gate, tol 2e-3, low-footprint boot).
#   (b) deep-decode A/B at 200K/400K no-spec (accept & tok/s) vs the shipped
#       baseline. Expected: attn read 40ms -> ~12-16ms (K1) -> ~10-14ms (K1+K2);
#       no-spec @400K 13.2 -> ~19-22 tok/s.
#   (c) needle@240K + short reliability (no quality regression).
#
# Discipline: marker DIFFKV_KERNEL_BW, ast.parse after each .py edit, sys.exit(3)
#   on any load-bearing anchor-miss, idempotent (no-op if markers present),
#   default-off = byte-for-byte the current kernel.
# ============================================================================

SITE="/usr/local/lib/python3.12/dist-packages"
TRITON_ATTN="$SITE/vllm/v1/attention/backends/triton_attn.py"
KERNEL="$SITE/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"

echo "[diffkv-kernel-bw] fixing DiffKV 3D decode parallelism-starvation (VLLM_DIFFKV_BW gate, default off)"

# ----------------------------------------------------------------------------
# 0. Sanity: both files present with the expected load-bearing anchors, in the
#    LIVE post-fp8-kv-inline + post-diffkv-3d-qlen8 state.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import sys, pathlib
T = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn.py")
K = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
if not T.exists():
    print(f"[diffkv-kernel-bw] FATAL: triton_attn.py missing: {T}"); sys.exit(3)
if not K.exists():
    print(f"[diffkv-kernel-bw] FATAL: kernel file missing: {K}"); sys.exit(3)
ts = T.read_text(); ks = K.read_text()

# K1 anchor: the module constant (stock; neither mod touches triton_attn.py).
K1_ANCHOR = "NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments\n"
if ts.count(K1_ANCHOR) != 1:
    print(f"[diffkv-kernel-bw] FATAL: NUM_PAR_SOFTMAX_SEGMENTS constant anchor count = "
          f"{ts.count(K1_ANCHOR)} (expected 1)"); sys.exit(3)
# The single symbolic read the resize composition relies on.
if "        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS\n" not in ts:
    print("[diffkv-kernel-bw] FATAL: `self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS` "
          "member-assign anchor missing (composition invariant broken)"); sys.exit(3)
if "import vllm.envs as envs\n" not in ts:
    print("[diffkv-kernel-bw] FATAL: triton_attn.py missing `import vllm.envs as envs`"); sys.exit(3)

# K2 anchor: the 3D-path tile_size selection (stock; fp8/3dq8 don't touch it).
K2_ANCHOR = "    tile_size = 32 if not use_3d else (16 if q.element_size() >= 2 else 32)\n"
if ks.count(K2_ANCHOR) != 1:
    print(f"[diffkv-kernel-bw] FATAL: tile_size selection anchor count = "
          f"{ks.count(K2_ANCHOR)} (expected 1)"); sys.exit(3)
# `k` (key_cache view) must be defined before the tile_size line so K2 can key
# on the KV element size.
if "    num_kv_heads = k.shape[2]\n" not in ks:
    print("[diffkv-kernel-bw] FATAL: `num_kv_heads = k.shape[2]` anchor missing "
          "(k must be in scope for the KV-element-size tile_size fix)"); sys.exit(3)
# The reduce kernel's power-of-two segment indexing (why segments MUST be 2^n).
if "tl.arange(0, NUM_SEGMENTS_PER_SEQ)" not in ks:
    print("[diffkv-kernel-bw] FATAL: reduce-kernel `tl.arange(0, NUM_SEGMENTS_PER_SEQ)` "
          "anchor missing (power-of-two invariant unverifiable)"); sys.exit(3)

# Confirm the prerequisite mods actually ran (fail loud if ordering is wrong).
if "FP8_KV_INLINE_KVLOAD" not in ks:
    print("[diffkv-kernel-bw] FATAL: kernel lacks FP8_KV_INLINE_KVLOAD "
          "-- run fp8-kv-inline BEFORE this mod"); sys.exit(3)
if "DIFFKV_3D_QLEN8" not in ks:
    print("[diffkv-kernel-bw] FATAL: kernel lacks DIFFKV_3D_QLEN8 "
          "-- run diffkv-3d-qlen8 BEFORE this mod"); sys.exit(3)
print("[diffkv-kernel-bw] sanity OK (K1 constant + symbolic member-read + K2 tile_size + "
      "k-in-scope + power-of-two reduce + fp8/3dq8 prereqs present)")
PY
[ $? -eq 0 ] || { echo "[diffkv-kernel-bw] sanity FAILED — aborting"; exit 1; }


# ============================================================================
# 1. K1 — NUM_PAR_SOFTMAX_SEGMENTS: env-gated segment count (triton_attn.py).
#    Replace the literal `16` constant with an env-derived value. Default off ->
#    16 (byte-for-byte). VLLM_DIFFKV_BW=1 -> VLLM_DIFFKV_SEGMENTS (default 64),
#    power-of-two enforced (else the module import raises -> loud, not silent).
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn.py")
src = p.read_text()

if "DIFFKV_KERNEL_BW" in src:
    print("[diffkv-kernel-bw] K1 segment-count already patched"); sys.exit(0)

anchor = "NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments\n"
if src.count(anchor) != 1:
    print(f"[diffkv-kernel-bw] FATAL: K1 constant anchor count = {src.count(anchor)} "
          f"(expected 1)"); sys.exit(3)

block = (
    "# DIFFKV_KERNEL_BW (K1): the DiffKV 3D split-KV decode grid is\n"
    "#   1 q-block x num_kv_heads x NUM_PAR_SOFTMAX_SEGMENTS CTAs. At deep\n"
    "#   context / q_len=1 / bs=1 that is 1 x 2 x 16 = 32 CTAs on 48 SMs, so each\n"
    "#   SM holds one CTA and the KV read runs at memory LATENCY, not bandwidth\n"
    "#   (~21% of 273 GB/s). Raising segments 16->64 -> 128 CTAs -> ~2.7 CTAs/SM\n"
    "#   -> latency hidden by concurrency -> the read approaches bandwidth.\n"
    "#   The count MUST stay a power of two: kernel_reduce_segments_diffkv indexes\n"
    "#   segments with tl.arange(0, NUM_SEGMENTS_PER_SEQ). Default OFF\n"
    "#   (VLLM_DIFFKV_BW unset/0) keeps 16 -> byte-for-byte the stock constant. The\n"
    "#   three softmax_segm_* scratch buffers are ALL sized symbolically from this\n"
    "#   constant via self.num_par_softmax_segments, so a bump flows into every\n"
    "#   alloc (base + DiffKV + diffkv-3d-qlen8 resize) with no separate edit.\n"
    "import os as _os_diffkv_bw\n"
    "\n"
    "\n"
    "def _diffkv_bw_segments() -> int:\n"
    "    if _os_diffkv_bw.environ.get(\"VLLM_DIFFKV_BW\", \"0\") != \"1\":\n"
    "        return 16  # stock: gate off = byte-for-byte 16.\n"
    "    _raw = _os_diffkv_bw.environ.get(\"VLLM_DIFFKV_SEGMENTS\", \"64\")\n"
    "    try:\n"
    "        _n = int(_raw)\n"
    "    except (TypeError, ValueError):\n"
    "        raise ValueError(\n"
    "            f\"VLLM_DIFFKV_SEGMENTS must be an int, got {_raw!r}\"\n"
    "        )\n"
    "    # power-of-two in [1,256]: the reduce kernel's tl.arange over segments\n"
    "    # and the epilogue segment strides require an exact 2^n count.\n"
    "    if _n < 1 or _n > 256 or (_n & (_n - 1)) != 0:\n"
    "        raise ValueError(\n"
    "            f\"VLLM_DIFFKV_SEGMENTS must be a power of two in [1,256], got {_n}\"\n"
    "        )\n"
    "    return _n\n"
    "\n"
    "\n"
    "NUM_PAR_SOFTMAX_SEGMENTS = _diffkv_bw_segments()  # DIFFKV_KERNEL_BW (16 off / 64 on)\n"
)
src = src.replace(anchor, block, 1)
ast.parse(src)
p.write_text(src)
print("[diffkv-kernel-bw] K1 patched: NUM_PAR_SOFTMAX_SEGMENTS env-gated "
      "(16 off / VLLM_DIFFKV_SEGMENTS default 64 on)")
PY
[ $? -eq 0 ] || { echo "[diffkv-kernel-bw] K1 patch FAILED — aborting"; exit 1; }


# ============================================================================
# 2. K2 — 3D-path TILE_SIZE keys on the KV cache element size, not the query's.
#    Only when VLLM_DIFFKV_BW=1 AND the KV cache is fp8 (uint8, element_size 1)
#    does fp8 get TILE_SIZE=32. Bf16 KV (element_size 2) stays 16. Gate off ->
#    the expression evaluates identically to the stock line for every dtype.
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
src = p.read_text()

if "DIFFKV_KERNEL_BW" in src:
    print("[diffkv-kernel-bw] K2 tile_size already patched"); sys.exit(0)

anchor = "    tile_size = 32 if not use_3d else (16 if q.element_size() >= 2 else 32)\n"
if src.count(anchor) != 1:
    print(f"[diffkv-kernel-bw] FATAL: K2 tile_size anchor count = {src.count(anchor)} "
          f"(expected 1)"); sys.exit(3)

# `k` is the key_cache view (line ~457: num_kv_heads = k.shape[2]). For fp8 KV
# it is uint8 (element_size 1); for bf16 KV it is bfloat16 (element_size 2).
# Replacement semantics:
#   gate OFF  -> _diffkv_bw_tile_fp8 == False -> the ternary is the STOCK
#                `16 if q.element_size() >= 2 else 32` -> byte-for-byte no-op.
#   gate ON   -> when k.element_size() == 1 (fp8) use 32; else fall back to the
#                stock query-keyed choice (bf16 KV -> 16, exotic -> stock).
repl = (
    "    # DIFFKV_KERNEL_BW (K2): the 3D decode loop is dominated by the KV read,\n"
    "    # and the fp8 KV cache is uint8 (element_size 1). Stock keys TILE_SIZE on\n"
    "    # the QUERY dtype (bf16 -> 16 always). When VLLM_DIFFKV_BW=1 and the KV\n"
    "    # cache is 1-byte (fp8/uint8), use TILE_SIZE=32 -> half the KV iterations.\n"
    "    # TILE_SIZE is threaded identically into the main + reduce kernels from\n"
    "    # this one var, and the KV dtype is fixed per deployment -> cudagraph-safe.\n"
    "    # Gate off -> this is exactly the stock `16 if q.element_size() >= 2 else 32`.\n"
    "    import os as _os_diffkv_bw_k2\n"
    "    _diffkv_bw_tile_fp8 = (\n"
    "        _os_diffkv_bw_k2.environ.get(\"VLLM_DIFFKV_BW\", \"0\") == \"1\"\n"
    "        and k.element_size() == 1\n"
    "    )\n"
    "    if not use_3d:\n"
    "        tile_size = 32\n"
    "    elif _diffkv_bw_tile_fp8:\n"
    "        tile_size = 32  # fp8 KV: double the tile, half the KV-loop iterations.\n"
    "    else:\n"
    "        tile_size = 16 if q.element_size() >= 2 else 32  # stock query-keyed choice.\n"
)
src = src.replace(anchor, repl, 1)
ast.parse(src)
p.write_text(src)
print("[diffkv-kernel-bw] K2 patched: 3D TILE_SIZE keys on KV element size "
      "(fp8/uint8 -> 32 when VLLM_DIFFKV_BW=1; bf16 -> stock 16)")
PY
[ $? -eq 0 ] || { echo "[diffkv-kernel-bw] K2 patch FAILED — aborting"; exit 1; }


# ----------------------------------------------------------------------------
# 3. clean bytecode + validate (parse+import both files; assert markers; assert
#    the env gate flips the segment count 16<->64 and rejects non-2^n).
# ----------------------------------------------------------------------------
find "$SITE/vllm/v1/attention" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import ast, importlib, os, sys
T = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn.py"
K = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
ts = open(T).read(); ks = open(K).read()
ast.parse(ts); ast.parse(ks)
assert "DIFFKV_KERNEL_BW" in ts, "K1 marker missing in triton_attn.py"
assert "DIFFKV_KERNEL_BW" in ks, "K2 marker missing in kernel"
assert "_diffkv_bw_segments" in ts, "K1 helper missing"
assert "k.element_size() == 1" in ks, "K2 KV-element-size check missing"

import vllm.v1.attention.backends.triton_attn as A
import vllm.v1.attention.ops.triton_unified_attention_diffkv as _k  # noqa: F401

# default (gate off) MUST be 16 -> byte-for-byte stock.
os.environ.pop("VLLM_DIFFKV_BW", None)
os.environ.pop("VLLM_DIFFKV_SEGMENTS", None)
importlib.reload(A)
assert A.NUM_PAR_SOFTMAX_SEGMENTS == 16, \
    f"gate OFF must give 16, got {A.NUM_PAR_SOFTMAX_SEGMENTS}"

# gate on, default -> 64.
os.environ["VLLM_DIFFKV_BW"] = "1"
importlib.reload(A)
assert A.NUM_PAR_SOFTMAX_SEGMENTS == 64, \
    f"gate ON default must give 64, got {A.NUM_PAR_SOFTMAX_SEGMENTS}"

# sweep override -> 128 (power of two).
os.environ["VLLM_DIFFKV_SEGMENTS"] = "128"
importlib.reload(A)
assert A.NUM_PAR_SOFTMAX_SEGMENTS == 128, \
    f"VLLM_DIFFKV_SEGMENTS=128 must give 128, got {A.NUM_PAR_SOFTMAX_SEGMENTS}"

# non-power-of-two -> must raise (loud, not silent garbage).
os.environ["VLLM_DIFFKV_SEGMENTS"] = "48"
raised = False
try:
    importlib.reload(A)
except ValueError:
    raised = True
assert raised, "VLLM_DIFFKV_SEGMENTS=48 (non-2^n) must raise ValueError, did not"

# restore default-off state for the running interpreter's import cache hygiene.
os.environ.pop("VLLM_DIFFKV_BW", None)
os.environ.pop("VLLM_DIFFKV_SEGMENTS", None)
importlib.reload(A)
assert A.NUM_PAR_SOFTMAX_SEGMENTS == 16
print("[diffkv-kernel-bw] validation OK (both files parse+import; segments "
      "16<->64<->128 flip with env; non-2^n rejected; markers set)")
PY
[ $? -eq 0 ] || { echo "[diffkv-kernel-bw] validation FAILED — aborting"; exit 1; }

echo "[diffkv-kernel-bw] done (enable with VLLM_DIFFKV_BW=1 [+ VLLM_DIFFKV_SEGMENTS=64]; default=stock 16-seg / query-keyed tile)"
