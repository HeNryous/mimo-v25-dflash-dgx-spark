#!/bin/bash
set -uo pipefail
# ============================================================================
# diffkv-prefill-tune (A1) — env-gated prefill q-block tuning for the DiffKV
#   Triton kernel on sm_121 (GB10). Cuts cold-prefill TTFT by fixing the
#   under-tiled 2D prefill launch: stock BLOCK_M=16 -> BLOCK_Q=1 for MiMo TP=2
#   (num_queries_per_kv=16) means ONE query token per program, so every program
#   re-streams the whole KV history. Upstream precedent: `tuned_large_head` in
#   triton_unified_attention.py (BLOCK_M=32/8 warps/2 stages/wider tile,
#   "~2x faster on B200" for the same nqpk<=16 large-head shape class).
#
#   Image: vllm-node-mimo-v25-upstream:latest  (also applies cleanly to the
#   nvfp4 host-fork kernel — the patcher auto-detects the variant).
#
# GATING (all four params all-or-nothing; anything else = stock, logged once):
#   VLLM_DIFFKV_PREFILL_TUNE unset/empty -> TRUE no-op: gate short-circuits on
#     max_seqlen_q before any env read; **{} adds nothing to the launch ->
#     identical Triton specialization key -> byte-identical compiled kernel.
#   VLLM_DIFFKV_PREFILL_TUNE="BM,warps,stages,tile" (e.g. "32,8,2,32"; "1" is
#     an alias for "32,8,2,32") -> applied ONLY when max_seqlen_q > 8, so
#     decode (q_len 1), DFlash verify (q_len 8) and every cudagraph-captured
#     shape are untouched (prefill attention is never captured).
#     !! num_speculative_tokens > 7 (spec11-class recipes) needs _PF_MIN_Q
#     raised above num_spec+1 FIRST (see the injected comment in the kernel).
#
# SHIP GATE: NEVER set the env on a serving boot before
#   diffkv_prefill_parity.py PASSes ON BOTH NODES for the exact env value
#   (sm_121 silent-garbage class is only catchable numerically).
#
# ORDERING: run LAST among the kernel mods (after fp8-kv-inline,
#   diffkv-3d-qlen8, diffkv-kernel-bw). Our two anchors — the `grid:` decl and
#   the launch close-paren — are untouched by all of them, but running last
#   guarantees THEIR exact-match anchors still see stock text.
#
# Discipline: marker-guarded idempotency, exact-count anchors, ast.parse after
#   edit, sys.exit(3) on anchor miss (loud boot abort, never a partial patch).
# ============================================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="/usr/local/lib/python3.12/dist-packages"
KERNEL="$SITE/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"

echo "[diffkv-prefill-tune] injecting A1 prefill q-block tune (gate: VLLM_DIFFKV_PREFILL_TUNE=${VLLM_DIFFKV_PREFILL_TUNE:-<unset -> no-op>})"

python3 "$HERE/patch_prefill_tune.py" "$KERNEL"
rc=$?
[ $rc -eq 0 ] || { echo "[diffkv-prefill-tune] FATAL: patcher exited $rc — aborting boot"; exit 1; }

python3 - <<'PY'
import ast, pathlib
src = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py").read_text()
ast.parse(src)
assert src.count("DIFFKV_PREFILL_TUNE") >= 1, "marker missing after patch"
assert src.count("**_pf_launch_kwargs,") == 1, "launch kwargs spread missing/duplicated"
print("[diffkv-prefill-tune] post-patch verify OK (ast + marker + single launch spread)")
PY
[ $? -eq 0 ] || { echo "[diffkv-prefill-tune] FATAL: post-patch verify failed"; exit 1; }
