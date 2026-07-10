#!/bin/bash
set -uo pipefail
# ============================================================================
# diffkv-prefill-tune (A1) + tile64-safe (A2) — env-gated prefill q-block
#   tuning for the DiffKV Triton kernel on sm_121 (GB10), now with an
#   ORDER-PRESERVING TILE=64 path.
#
#   A1 (patch_prefill_tune.py, unchanged): BLOCK_M=32/warps4/stages2/tile32
#   launch tune, default-ON (bit-identical, parity-proven), fixes the
#   under-tiled stock prefill launch (BLOCK_M=16 -> BLOCK_Q=1 for MiMo TP=2).
#
#   A2 (patch_tile64_safe.py, NEW): raw tile=64 added 1.45-1.56x on deep
#   prefill but failed strict parity (rel 3-7e-3 = fp32 accumulation
#   reordering: joint softmax rescale / bf16 P-cast after 64-wide max /
#   64-long PV acc). The A2 body patch keeps TILE=64 for the MEMORY side
#   (one K 256x64 + one V 64x128 load + one block-table gather per 64
#   tokens) and splits the loaded tiles into two 32-column halves that
#   replay the stock 32-wide sequence in stock order -> bit-identical by
#   construction. Kernel gains `SPLIT_TILE: tl.constexpr = False` (default
#   -> frontend-dead branch, byte-equivalent compute to today).
#
# GATING (unchanged A1 gate, max_seqlen_q > 8 AND env, all-or-nothing):
#   VLLM_DIFFKV_PREFILL_TUNE unset      -> "32,4,2,32" (A1 prod default,
#                                          bit-identical) — tile64 inert.
#   VLLM_DIFFKV_PREFILL_TUNE="safe64"   -> alias for "32,4,2,64": tile-64
#     loads + SPLIT_TILE=True on full-attn prefill. SWA layers, tables whose
#     rows aren't 64-token aligned, and tile>64 all DOWNGRADE to tile 32.
#     Raw joint-64 compute is NOT reachable from the env anymore.
#   VLLM_DIFFKV_PREFILL_TUNE="off"      -> unparsable -> stock launch.
#
# SHIP GATE: NEVER set "safe64" on a serving boot before
#   diffkv_prefill_parity.py (TILE64-SAFE v2) PASSes ON BOTH NODES with the
#   EXACT-0.0 bar (--tune safe64 --vs-torch, and --tune safe64 --sinks).
#   Any tun-vs-stk != 0.0 means the split is NOT order-preserving -> REJECT.
#   See parity_tile64.md for the full boot-gap protocol.
#
# ORDERING: this mod runs LAST among the kernel mods (after fp8-kv-inline,
#   diffkv-3d-qlen8, diffkv-kernel-bw). Within it, A1 MUST run before A2
#   (A2 anchors on A1's injected tune block and refuses to apply without
#   its marker).
#
# Discipline: marker-guarded idempotency, exact-count anchors, ast.parse
#   after edit, sys.exit(3) on anchor miss (loud boot abort, never partial).
# ============================================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="/usr/local/lib/python3.12/dist-packages"
KERNEL="$SITE/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"

echo "[diffkv-prefill-tune] injecting A1 prefill q-block tune + A2 tile64-safe (gate: VLLM_DIFFKV_PREFILL_TUNE=${VLLM_DIFFKV_PREFILL_TUNE:-<unset -> A1 default 32,4,2,32>})"

# ---- A1: launch tune (unchanged) -------------------------------------------
python3 "$HERE/patch_prefill_tune.py" "$KERNEL"
rc=$?
[ $rc -eq 0 ] || { echo "[diffkv-prefill-tune] FATAL: A1 patcher exited $rc — aborting boot"; exit 1; }

# ---- A2: order-preserving TILE=64 body patch (must follow A1) ---------------
python3 "$HERE/patch_tile64_safe.py" "$KERNEL"
rc=$?
[ $rc -eq 0 ] || { echo "[diffkv-prefill-tune] FATAL: A2 tile64-safe patcher exited $rc — aborting boot (kernel may carry A1 only; tile=64 env values would be REJECTED by A1's launch, not silently raw)"; exit 1; }

# ---- post-patch verify (both hunk sets, single application each) ------------
python3 - <<'PY'
import ast, pathlib
src = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py").read_text()
ast.parse(src)
# A1 markers
assert src.count("DIFFKV_PREFILL_TUNE") >= 1, "A1 marker missing after patch"
assert src.count("**_pf_launch_kwargs,") == 1, "A1 launch kwargs spread missing/duplicated"
# A2 markers — signature param, split branch, wrapper gate, safe64 alias:
# each exactly once (idempotency + no partial application)
assert src.count("DIFFKV_TILE64_SAFE") >= 4, "A2 marker missing after patch"
assert src.count("SPLIT_TILE: tl.constexpr = False,") == 1, "A2 SPLIT_TILE param missing/duplicated"
assert src.count("if SPLIT_TILE:") == 1, "A2 split branch missing/duplicated"
assert src.count('_pf_launch_kwargs["SPLIT_TILE"] = True') == 1, "A2 wrapper gate missing/duplicated"
assert src.count('if _pf_env == "safe64"') == 1, "A2 safe64 alias missing/duplicated"
# A2 safety invariants — the split branch must carry both static asserts
assert 'tl.static_assert(TILE_SIZE == 64' in src, "A2 TILE_SIZE static assert missing"
assert "SPLIT_TILE is order-unsafe under SWA" in src, "A2 SWA static assert missing"
print("[diffkv-prefill-tune] post-patch verify OK (ast + A1 + A2 markers, single application, static asserts present)")
PY
[ $? -eq 0 ] || { echo "[diffkv-prefill-tune] FATAL: post-patch verify failed"; exit 1; }
