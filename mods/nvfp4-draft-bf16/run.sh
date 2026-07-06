#!/bin/bash
set -uo pipefail
# ============================================================================
# nvfp4-draft-bf16 — give the DFlash drafter bf16 KV while the MiMo target keeps
#   nvfp4 KV, under a GLOBAL --kv-cache-dtype nvfp4.
#
#   Image: vllm-node-mimo-v25-upstream:latest
#   vllm 0.23.1rc1.dev760+g3775d5fca (main @3775d5fc)
#
# THE PROBLEM
#   --kv-cache-dtype nvfp4 is GLOBAL. Each attention layer's KV dtype is derived
#   from cache_config.cache_dtype in the SHARED Attention.__init__:
#     vllm/model_executor/layers/attention/attention.py:241
#         kv_cache_dtype = cache_config.cache_dtype        # == "nvfp4"
#   which is then handed to get_attn_backend(...) (attention.py:320-323) and, at
#   backend selection, validated against the backend's supported_kv_cache_dtypes:
#     vllm/platforms/cuda.py:404-414  raise ValueError(
#        "Selected backend ... TRITON_ATTN is not valid for this configuration.
#         Reason: ['kv_cache_dtype not supported']")
#   (backend.py:167-171 supports_kv_cache_dtype ->
#    triton_attn.py:277 TritonAttentionBackend.supported_kv_cache_dtypes has NO
#    "nvfp4" entry).
#
#   The DFlash drafter (5-layer Qwen3, class DFlashQwen3ForCausalLM,
#   qwen3_dflash.py:662) is built via LLMBaseProposer._create_draft_vllm_config
#   (llm_base_proposer.py:1284-1301), which does
#     replace(base, attention_config=replace(base.attention_config,
#                                             backend=spec_cfg.attention_backend))
#   i.e. it overrides ONLY attention_config (backend -> TRITON_ATTN from the
#   speculative-config) and SHARES the base cache_config (cache_dtype="nvfp4").
#   Its Qwen3 layers use the STANDARD Attention class (qwen3_dflash.py:20 import,
#   :204 passes cache_config), so they inherit kv_cache_dtype="nvfp4" and hit the
#   TRITON_ATTN validation wall above.
#
# THE FIX (Approach A, per-layer opt-out keyed on the backend enum)
#   Keep the global nvfp4 (target sizing stays 100% native — the working
#   nvfp4-kv-upstream mod is UNCHANGED). Right after attention.py:241, force
#   kv_cache_dtype back to "auto" (== bf16 for this bf16-weighted draft) FOR
#   DRAFTER LAYERS ONLY. This is the same mechanism vLLM ships as
#   --kv-cache-dtype-skip-layers (attention.py:268-284 flips matching layers to
#   "auto"), but with a ROBUST discriminator instead of the shipped
#   sliding-window/layer-index heuristic (which is unreliable here: the Qwen3
#   draft full-attn layers carry NO sliding window, and their layer indices are
#   target_num_layers+i (qwen3_dflash.py:372,675) — model-count dependent, not
#   safe to hardcode).
#
# THE DISCRIMINATOR: get_current_vllm_config().attention_config.backend
#   During drafter layer __init__ the current vllm config IS the draft config
#   (base_loader.load_model -> initialize_model ->
#    set_current_vllm_config(draft_vllm_config); model_loader/utils.py:62,94),
#   whose attention_config.backend == AttentionBackendEnum.TRITON_ATTN.
#   During TARGET (MiMo) layer __init__ the backend is
#   AttentionBackendEnum.TRITON_ATTN_DIFFKV (recipe sets --attention-backend
#   triton_attn_diffkv; and even absent that, MiMo head_size 192 auto-selects
#   DIFFKV, never plain TRITON_ATTN). So `backend == TRITON_ATTN` matches the
#   drafter and ONLY the drafter. AttentionBackendEnum is already imported in
#   attention.py:40 and vllm_config = get_current_vllm_config() is already bound
#   one line above the patch site (attention.py:239) — no new imports/vars.
#
# WHY NOT the alternatives:
#   - Global --kv-cache-dtype auto + force nvfp4 on the diffkv layers: would
#     require re-introducing a per-layer cache_dtype/spec override on the TARGET
#     path (the redundant dev114 get_kv_cache_spec patch the nvfp4-kv-upstream
#     mod deliberately dropped) — more invasive, touches the working target path.
#   - Add nvfp4 to TRITON_ATTN.supported_kv_cache_dtypes: validation would pass
#     but the drafter's self.kv_cache_dtype stays "nvfp4" -> get_kv_cache_spec
#     (attention.py:599) builds an nvfp4 (uint8, quant_mode=nvfp4) spec and
#     TRITON_ATTN.get_kv_cache_shape has no nvfp4 packing -> wrong pool + garbage
#     drafts. Rejected.
#
# EFFECT ON POOL SIZING: none for the target. The drafter now allocates a bf16
#   KV spec (FullAttentionSpec, model dtype) instead of an nvfp4 one. The
#   drafter's KV is tiny (5 Qwen3 layers) so this is a negligible memory delta;
#   the ~3.5x-larger nvfp4 pool comes entirely from the TARGET diffkv layers,
#   which are untouched.
#
# Discipline: marker-guarded idempotency, ast.parse after the edit,
# sys.exit on anchor-miss (load-bearing).
# ============================================================================

SITE="/usr/local/lib/python3.12/dist-packages"
ATTN="$SITE/vllm/model_executor/layers/attention/attention.py"

echo "[nvfp4-draft-bf16] forcing DFlash drafter (TRITON_ATTN) layers to bf16 KV under global nvfp4"

# ----------------------------------------------------------------------------
# 0. Sanity: the shared Attention.__init__ kv-dtype resolution + the enum import
#    must be present in the expected shape.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import sys, pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py")
s = p.read_text()
need = [
    "from vllm.v1.attention.backends.registry import AttentionBackendEnum",
    "        vllm_config = get_current_vllm_config()\n",
    "            kv_cache_dtype = cache_config.cache_dtype\n",
    "        self.kv_cache_dtype = kv_cache_dtype\n",
]
for n in need:
    if n not in s:
        print(f"[nvfp4-draft-bf16] FATAL: attention.py missing anchor {n!r}"); sys.exit(3)
# The enum members we key on must exist.
from vllm.v1.attention.backends.registry import AttentionBackendEnum
assert hasattr(AttentionBackendEnum, "TRITON_ATTN"), "AttentionBackendEnum.TRITON_ATTN missing"
assert hasattr(AttentionBackendEnum, "TRITON_ATTN_DIFFKV"), "AttentionBackendEnum.TRITON_ATTN_DIFFKV missing"
print("[nvfp4-draft-bf16] sanity OK (attention.py kv-dtype resolution + enum present)")
PY
[ $? -eq 0 ] || { echo "[nvfp4-draft-bf16] sanity FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# 1. Insert the drafter opt-out right AFTER the global kv_cache_dtype resolution
#    (the if/else that reads cache_config.cache_dtype). Marker: NVFP4_DRAFT_BF16.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py")
src = p.read_text()
if "NVFP4_DRAFT_BF16" in src:
    print("[nvfp4-draft-bf16] drafter opt-out already present"); sys.exit(0)

anchor = (
    "        vllm_config = get_current_vllm_config()\n"
    "        if cache_config is not None:\n"
    "            kv_cache_dtype = cache_config.cache_dtype\n"
    "            calculate_kv_scales = cache_config.calculate_kv_scales\n"
    "        else:\n"
    "            kv_cache_dtype = \"auto\"\n"
    "            calculate_kv_scales = False\n"
)
if src.count(anchor) != 1:
    print(f"[nvfp4-draft-bf16] FATAL: kv-dtype-resolution anchor count = {src.count(anchor)} "
          f"(expected 1)"); sys.exit(3)

# NB: `vllm_config` is already bound on the first anchor line; reuse it.
# We gate on the CURRENT config's attention backend. For the DFlash drafter this
# is TRITON_ATTN (LLMBaseProposer._create_draft_vllm_config overrides
# attention_config.backend to spec_cfg.attention_backend, and the drafter is
# loaded under set_current_vllm_config(draft_vllm_config)). The MiMo target is
# TRITON_ATTN_DIFFKV -> unaffected. Only quantized global dtypes are forced back
# to "auto" (bf16) so a normally-bf16 run is untouched.
INSERT = (
    "        # NVFP4_DRAFT_BF16: the DFlash drafter (Qwen3, TRITON_ATTN) cannot use\n"
    "        # the target's nvfp4 KV (TRITON_ATTN.supported_kv_cache_dtypes lacks\n"
    "        # nvfp4; and TRITON_ATTN has no nvfp4 dequant path). Under a GLOBAL\n"
    "        # quantized --kv-cache-dtype, force the drafter's layers back to bf16\n"
    "        # (\"auto\"). Discriminator = the current config's attention backend:\n"
    "        # the drafter is loaded under set_current_vllm_config(draft_vllm_config)\n"
    "        # whose attention_config.backend == TRITON_ATTN, while the MiMo target\n"
    "        # is TRITON_ATTN_DIFFKV. Robust vs the shipped skip-layers heuristic\n"
    "        # (draft full-attn layers have no sliding window; their layer indices\n"
    "        # are target_num_layers+i, model-count dependent).\n"
    "        try:\n"
    "            _nvfp4_draft_backend = (\n"
    "                vllm_config.attention_config.backend\n"
    "                if vllm_config is not None\n"
    "                and vllm_config.attention_config is not None\n"
    "                else None\n"
    "            )\n"
    "        except AttributeError:\n"
    "            _nvfp4_draft_backend = None\n"
    "        if (\n"
    "            _nvfp4_draft_backend is AttentionBackendEnum.TRITON_ATTN\n"
    "            and isinstance(kv_cache_dtype, str)\n"
    "            and kv_cache_dtype not in (\"auto\", \"float16\", \"bfloat16\")\n"
    "        ):\n"
    "            logger.info(\n"
    "                \"NVFP4_DRAFT_BF16: forcing draft layer %s kv_cache_dtype \"\n"
    "                \"%s -> auto (bf16); TRITON_ATTN cannot use the target's \"\n"
    "                \"quantized KV.\",\n"
    "                prefix,\n"
    "                kv_cache_dtype,\n"
    "            )\n"
    "            kv_cache_dtype = \"auto\"\n"
    "            calculate_kv_scales = False\n"
)
src = src.replace(anchor, anchor + INSERT)
ast.parse(src)
p.write_text(src)
print("[nvfp4-draft-bf16] inserted drafter bf16 opt-out (backend==TRITON_ATTN -> kv_cache_dtype auto)")
PY
[ $? -eq 0 ] || { echo "[nvfp4-draft-bf16] opt-out patch FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# clean stale bytecode + validate module parses & imports
# ----------------------------------------------------------------------------
find "$SITE/vllm/model_executor/layers/attention" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import ast, sys
f = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py"
src = open(f).read()
ast.parse(src)
assert "NVFP4_DRAFT_BF16" in src, "marker missing after write"
# module imports cleanly
import vllm.model_executor.layers.attention.attention as A  # noqa: F401
# the enum we key on resolves
from vllm.v1.attention.backends.registry import AttentionBackendEnum
assert AttentionBackendEnum.TRITON_ATTN is not AttentionBackendEnum.TRITON_ATTN_DIFFKV
# target backend still rejects nvfp4? no — target uses DIFFKV which the
# nvfp4-kv-upstream mod teaches to accept nvfp4. drafter backend (plain
# TRITON_ATTN) must still NOT list nvfp4 (we route the drafter to bf16 instead
# of widening its support list):
import vllm.v1.attention.backends.triton_attn as T
assert "nvfp4" not in T.TritonAttentionBackend.supported_kv_cache_dtypes, \
    "TRITON_ATTN unexpectedly advertises nvfp4 (this mod routes drafter to bf16, " \
    "it must NOT widen TRITON_ATTN's dtype support)"
print("[nvfp4-draft-bf16] validation OK (attention.py parses+imports, marker set, "
      "enum discriminator resolves, TRITON_ATTN kept nvfp4-free)")
PY
[ $? -eq 0 ] || { echo "[nvfp4-draft-bf16] VALIDATION FAILED — aborting"; exit 1; }

echo "[nvfp4-draft-bf16] done"
