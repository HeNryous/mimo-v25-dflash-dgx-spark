#!/bin/bash
set -uo pipefail
# ============================================================================
# fix-mimo-v2-upstream — SLIM MiMo-V2.5 mod for the UPSTREAM vLLM image
#   vllm-node-mimo-v25-upstream:latest
#   vllm 0.23.1rc1.dev760+g3775d5fca (main @3775d5fc, post PR #46104)
#
# What is NATIVE in this image (DROPPED vs fix-mimo-v2-vllm-0230):
#   - PR #41797  TRITON_ATTN_DIFFKV backend .............. native (file present)
#   - MiMoV2 / MiMoV2Omni / DFlashDraftModel in registry . native
#   - #40727 aux+1 / DFlash SWA (_maybe_symmetrize_window) native
#   - modelopt-mixed / lm_head-nvfp4 / WMMA / TK / Quest . not used (fp8-KV path)
#   - FA3-sinks patch .................................... moot (native DiffKV)
#   - MiMoV2 text-only override / qkv-deinterleave loader  not needed (Omni,
#       fp8 not nvfp4-mixed; recipe forces Omni via hf-overrides)
#
# What upstream STILL LACKS (KEPT here — each anchor verified in THIS image):
#   1. DiffKV accept fp8 KV (2 patches in triton_attn_diffkv.py)
#   2. MimoV2Config HF registration (checkpoint model_type=mimo_v2 unrecognized)
#   3. audio deps (soundfile/librosa/av absent; Omni audio path needs them)
#
# DEFERRED to the daily-driver phase (NOT applied here — not needed for the
# DFlash accept-validation boot; text-only DFlash target, /v1/completions):
#   - PR #251 chat template + mimo reasoning parser + thinking-budget
#   - Qwen3XML tool-parser #42969 fix
#   (Add them back from fix-mimo-v2-vllm-0230 once accept-length is validated.)
#
# Discipline: marker-guarded idempotency, ast.parse syntax check, sys.exit on
# anchor-miss for the load-bearing patches.
# ============================================================================

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
cd "$SITE_PACKAGES"

echo "[fix-mimo-v2-upstream] Applying slim MiMo-V2.5 upstream fixes"

# ----------------------------------------------------------------------------
# 1. DiffKV fp8-KV enable — the critical patch.
#    backend.py:supports_kv_cache_dtype() gates the backend on
#    `kv_cache_dtype in supported_kv_cache_dtypes`; DiffKV declares only
#    ["auto","bfloat16"] so --kv-cache-dtype fp8 is REJECTED at backend
#    selection. Then __init__ raises NotImplementedError on any quantized KV.
#    Verified in this image:
#      - list at triton_attn_diffkv.py:78-81  ["auto","bfloat16"]
#      - guard at triton_attn_diffkv.py:154-158
#      - is_quantized_kv_cache("fp8")==True  -> guard would fire
#    Patch (idempotent, marker EXPERIMENTAL_ALLOW_DIFFKV_FP8_KV):
#      (a) add "fp8","fp8_e4m3" to supported_kv_cache_dtypes
#      (b) gate the NotImplementedError behind `False and`
# ----------------------------------------------------------------------------
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("vllm/v1/attention/backends/triton_attn_diffkv.py")
src = p.read_text()

if "EXPERIMENTAL_ALLOW_DIFFKV_FP8_KV" in src:
    print("[fix-mimo-v2-upstream] DiffKV fp8-KV already patched")
    sys.exit(0)

# (a) supported_kv_cache_dtypes list
old_list = '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]'''
new_list = '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",         # EXPERIMENTAL_ALLOW_DIFFKV_FP8_KV
        "fp8_e4m3",    # EXPERIMENTAL_ALLOW_DIFFKV_FP8_KV
    ]'''

# (b) NotImplementedError guard
old_guard = '''        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not yet support quantized "
                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )'''
new_guard = '''        # EXPERIMENTAL_ALLOW_DIFFKV_FP8_KV: the fp8 store/decode path
        # (triton_reshape_and_cache_flash_diffkv) already accepts dtype +
        # k_scale/v_scale; this guard was defensive, not technical. Gate it off
        # so fp8 KV halves the cache -> larger KV pool. Revert by removing the
        # `False and`.
        if False and is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not yet support quantized "
                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )'''

n_list = src.count(old_list)
n_guard = src.count(old_guard)
if n_list != 1 or n_guard != 1:
    print(f"[fix-mimo-v2-upstream] FATAL DiffKV anchor mismatch: "
          f"list={n_list} guard={n_guard} (expected 1/1)")
    sys.exit(3)

src = src.replace(old_list, new_list).replace(old_guard, new_guard)
ast.parse(src)
p.write_text(src)
print("[fix-mimo-v2-upstream] DiffKV fp8-KV patched (list +fp8/fp8_e4m3, guard neutralized)")
PY
[ $? -eq 0 ] || { echo "[fix-mimo-v2-upstream] DiffKV patch FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# 2. MimoV2Config HF registration.
#    Checkpoint config.json: model_type=mimo_v2, no auto_map, no local
#    configuration_mimo_v2.py. This image's _CONFIG_REGISTRY has NO mimo entry
#    and transformers AutoConfig does not know mimo_v2 ->
#    get_config() raises ValueError "model type `mimo_v2` not recognized".
#    (model_arch_config_convertor.py maps the model_type to a convertor but
#    does NOT register a PretrainedConfig class.) Register a minimal config,
#    same 3-file wiring the dev114 mod used; anchors verified in this image:
#      configs/__init__.py:55  "MiDashengLMConfig": "...configs.midashenglm",
#      configs/__init__.py:129 "MiDashengLMConfig",
#      config.py:105           midashenglm="MiDashengLMConfig",
# ----------------------------------------------------------------------------
cat > "$SITE_PACKAGES/vllm/transformers_utils/configs/mimo_v2.py" <<'PY'
# SPDX-License-Identifier: Apache-2.0
from transformers import PretrainedConfig


class MimoV2Config(PretrainedConfig):
    model_type = "mimo_v2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
PY

python3 - <<'PY'
import sys
from pathlib import Path
site = Path('/usr/local/lib/python3.12/dist-packages')

init = site / 'vllm/transformers_utils/configs/__init__.py'
text = init.read_text()
changed = False
if '"MimoV2Config": "vllm.transformers_utils.configs.mimo_v2"' not in text:
    anchor = '    "MiDashengLMConfig": "vllm.transformers_utils.configs.midashenglm",\n'
    if anchor not in text:
        print("[fix-mimo-v2-upstream] FATAL: configs/__init__.py map anchor missing"); sys.exit(3)
    text = text.replace(
        anchor,
        anchor + '    "MimoV2Config": "vllm.transformers_utils.configs.mimo_v2",\n',
    )
    changed = True
if '    "MimoV2Config",\n' not in text:
    anchor2 = '    "MiDashengLMConfig",\n'
    if anchor2 not in text:
        print("[fix-mimo-v2-upstream] FATAL: configs/__init__.py __all__ anchor missing"); sys.exit(3)
    text = text.replace(anchor2, anchor2 + '    "MimoV2Config",\n')
    changed = True
if changed:
    init.write_text(text)

cfg = site / 'vllm/transformers_utils/config.py'
text = cfg.read_text()
if 'mimo_v2="MimoV2Config"' not in text:
    anchor = '    midashenglm="MiDashengLMConfig",\n'
    if anchor not in text:
        print("[fix-mimo-v2-upstream] FATAL: config.py registry anchor missing"); sys.exit(3)
    text = text.replace(anchor, anchor + '    mimo_v2="MimoV2Config",\n')
    cfg.write_text(text)

print("[fix-mimo-v2-upstream] MimoV2Config registered (configs/__init__.py + config.py)")
PY
[ $? -eq 0 ] || { echo "[fix-mimo-v2-upstream] MimoV2Config registration FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# 3. audio deps. soundfile/librosa/av are absent in this image; vLLM's
#    multimodal audio path (vllm/multimodal/audio.py) imports soundfile, and
#    the Omni model advertises audio -> "Please install vllm[audio]" at
#    audio-input time. mimo_v2_omni imports fine WITHOUT them (lazy), so this is
#    non-fatal for a text-only accept boot, but cheap insurance for the Omni
#    recipe. Idempotent (no-op if already importable).
# ----------------------------------------------------------------------------
python3 - <<'PY' || (uv pip install --quiet soundfile librosa av 2>/dev/null || pip install --quiet soundfile librosa av)
import soundfile  # noqa: F401
import librosa  # noqa: F401
import av  # noqa: F401
PY
echo "[fix-mimo-v2-upstream] audio deps ensured (soundfile/librosa/av)"

# ----------------------------------------------------------------------------
# 4. fp8/MXFP8 fused qkv_proj loader fix — THE BOOT BLOCKER.
#
#    Symptom (TP=2, first boot, this checkpoint):
#      mimo_v2.py:531 _shard_fp8_qkv_proj
#        RuntimeError: The size of tensor a (4096) must match the size of
#        tensor b (16384) at non-singleton dimension 1
#
#    Root cause (verified against the real checkpoint, not assumed):
#      MiMo-lmhead-nvfp4 config.json: hidden=4096, num_heads=64,
#      num_kv_heads=4, head_dim=192 (q/k), v_head_dim=128
#        => fused qkv width = 64*192 + 4*192 + 4*128 = 12288+768+512 = 13568.
#      On-disk qkv_proj.weight = fp8_e4m3fn [13568, 4096];
#               qkv_proj.weight_scale_inv = uint8/E8M0 [13568, 128].
#      hf_quant_config.json: this qkv_proj is quant_algo "MXFP8",
#      group_size 32 (=> 4096/32 = 128 scale cols; scale is PER-OUTPUT-ROW,
#      13568 rows, block only along the input dim). Empirically the fused
#      tensor is in CANONICAL [Q_all | K_all | V_all] order (H3), NOT the
#      per-KV-head-interleaved [Q1 K1 V1 | Q2 K2 V2 | ...] layout upstream's
#      _shard_fp8_qkv_proj assumes.
#
#      Upstream's _shard_fp8_qkv_proj is written for DeepSeek-style fp8
#      128x128 block-quant on an interleaved layout. For our MXFP8 scale it
#      does s_g.repeat_interleave(block=128, dim=1) on the 128-col scale ->
#      16384 cols, mismatching the 4096-col weight-group => the crash. Even if
#      it did not crash, its layout + hard-coded block=128 are wrong for this
#      checkpoint.
#
#    Fix (two idempotent sub-patches):
#      (4a) Register a `weight_scale_inv` ALIAS (same Parameter object) on
#           ModelOptMxFp8LinearMethod, so the checkpoint tensor name
#           `...qkv_proj.weight_scale_inv` resolves in params_dict (the MXFP8
#           method only registers `weight_scale`; upstream has NO
#           weight_scale_inv remap). Without this the scale would be silently
#           skipped after (4b) -> garbage output.
#      (4b) Make MiMoV2FlashForCausalLM._try_load_fp8_qkv_proj DECLINE (return
#           False) so the fused weight + aliased scale fall through to the
#           model's normal path -> QKVParallelLinear.weight_loader, which
#           narrows [Q|K|V] at raw element offsets. Because the MXFP8 scale is
#           a ModelWeightParameter (NOT BlockQuantScaleParameter), the loader
#           correctly SKIPS adjust_block_scale_shard -> dim-0 (per-output-row)
#           is sharded at the plain Q/K/V offsets = correct for H3 + MXFP8.
#           This also sidesteps _shard_fp8_qkv_proj's hard-coded block=128.
#
#    This is the upstream-adapted equivalent of the dev114
#    fix-mimo-v2-vllm(-0230) qkv-deinterleave patch (which routed the fused
#    qkv to QKVParallelLinear.weight_loader on the old code structure) PLUS the
#    fix-modelopt-mixed-mxfp8 weight_scale_inv alias hunk. The rest of
#    fix-modelopt-mixed-mxfp8 (MXFP8 per-layer dispatch) is now NATIVE in this
#    image, so only the alias is ported here.
#
#    Anchors verified present in THIS image:
#      modelopt.py ModelOptMxFp8LinearMethod.create_weights: the MXFP8-unique
#        weight_scale block ("input_size_per_partition // MXFP8_BLOCK_SIZE" +
#        MXFP8_SCALE_DTYPE) ending in register_parameter("weight_scale", ...)
#      mimo_v2.py _try_load_fp8_qkv_proj docstring-end + the
#        `is_weight = (` / `is_scale = ...` lines.
# ----------------------------------------------------------------------------

# (4a) MXFP8 weight_scale_inv alias
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py")
src = p.read_text()

if "EXPERIMENTAL_MXFP8_WEIGHT_SCALE_INV_ALIAS" in src:
    print("[fix-mimo-v2-upstream] MXFP8 weight_scale_inv alias already present")
    sys.exit(0)

# Anchor is the MXFP8-method-UNIQUE weight_scale block: the two lines
# `input_size_per_partition // MXFP8_BLOCK_SIZE,` + `dtype=MXFP8_SCALE_DTYPE,`
# appear ONLY in ModelOptMxFp8LinearMethod.create_weights, so the following
# register_parameter is unambiguously the MXFP8 one (the fp8/nvfp4 methods use
# ChannelQuantScaleParameter / BlockQuantScaleParameter with different text).
old = '''                input_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)'''
new = '''                input_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)
        # EXPERIMENTAL_MXFP8_WEIGHT_SCALE_INV_ALIAS: modelopt exports the MXFP8
        # block scale as `.weight_scale_inv`, but this method registers it as
        # `.weight_scale` and upstream has no name remap. Expose the SAME
        # Parameter object under `weight_scale_inv` too, so checkpoint tensors
        # named `...weight_scale_inv` resolve in params_dict (via
        # named_parameters(remove_duplicate=False)) and load through the normal
        # QKVParallelLinear path. Revert by removing this block.
        if "weight_scale_inv" not in layer._parameters:
            layer.register_parameter("weight_scale_inv", weight_scale)'''

n = src.count(old)
if n != 1:
    print(f"[fix-mimo-v2-upstream] FATAL MXFP8 alias anchor mismatch: found {n} (expected 1)")
    sys.exit(3)
src = src.replace(old, new)
ast.parse(src)
p.write_text(src)
print("[fix-mimo-v2-upstream] MXFP8 weight_scale_inv alias registered")
PY
[ $? -eq 0 ] || { echo "[fix-mimo-v2-upstream] MXFP8 alias patch FAILED — aborting"; exit 1; }

# (4b) _try_load_fp8_qkv_proj decline -> fall through to QKVParallelLinear
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py")
src = p.read_text()

if "EXPERIMENTAL_MIMO_QKV_FALLTHROUGH" in src:
    print("[fix-mimo-v2-upstream] MiMoV2 qkv fall-through already patched")
    sys.exit(0)

# Anchor: the end of the _try_load_fp8_qkv_proj docstring immediately followed
# by the is_weight/is_scale gate. Insert an early `return False` so the fused
# qkv_proj weight + (aliased) weight_scale_inv are NOT consumed here and fall
# through to load_weights' generic tail -> QKVParallelLinear.weight_loader.
old = '''            (caller should skip it); False otherwise, so the caller falls
            through to its normal loading path.
        """
        is_weight = (
            name.endswith("qkv_proj.weight") and tensor.dtype == torch.float8_e4m3fn
        )
        is_scale = name.endswith("qkv_proj.weight_scale_inv")'''
new = '''            (caller should skip it); False otherwise, so the caller falls
            through to its normal loading path.
        """
        # EXPERIMENTAL_MIMO_QKV_FALLTHROUGH: this checkpoint's fused qkv_proj is
        # canonical [Q_all|K_all|V_all] (H3) with a modelopt-MXFP8 group_size=32
        # per-output-row scale, NOT the per-KV-head-interleaved fp8 128-block
        # layout _shard_fp8_qkv_proj expects (which mis-broadcasts the 128-col
        # scale to 16384 -> "4096 vs 16384" crash). Decline here so the fused
        # weight + `weight_scale_inv`-aliased scale fall through to
        # QKVParallelLinear.weight_loader: it narrows [Q|K|V] at raw element
        # offsets and, since the MXFP8 scale is a ModelWeightParameter (not
        # BlockQuantScaleParameter), correctly skips adjust_block_scale_shard.
        # Revert by removing this early return.
        return False
        is_weight = (
            name.endswith("qkv_proj.weight") and tensor.dtype == torch.float8_e4m3fn
        )
        is_scale = name.endswith("qkv_proj.weight_scale_inv")'''

n = src.count(old)
if n != 1:
    print(f"[fix-mimo-v2-upstream] FATAL qkv fall-through anchor mismatch: found {n} (expected 1)")
    sys.exit(3)
src = src.replace(old, new)
ast.parse(src)
p.write_text(src)
print("[fix-mimo-v2-upstream] MiMoV2 _try_load_fp8_qkv_proj now declines (fused qkv -> QKVParallelLinear)")
PY
[ $? -eq 0 ] || { echo "[fix-mimo-v2-upstream] qkv fall-through patch FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# clean stale bytecode + validate
# ----------------------------------------------------------------------------
find "$SITE_PACKAGES/vllm" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import ast
# syntax check the patched files
for f in [
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/__init__.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/config.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/mimo_v2.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py",
]:
    ast.parse(open(f).read())

# config registry now knows mimo_v2
from vllm.transformers_utils.config import _CONFIG_REGISTRY
assert 'mimo_v2' in _CONFIG_REGISTRY, f"mimo_v2 not in _CONFIG_REGISTRY: {list(_CONFIG_REGISTRY)[:5]}..."
from vllm.transformers_utils.configs.mimo_v2 import MimoV2Config
assert MimoV2Config.model_type == 'mimo_v2'

# DiffKV now accepts fp8
from vllm.v1.attention.backends.triton_attn_diffkv import TritonAttentionDiffKVBackend as B
assert 'fp8' in B.supported_kv_cache_dtypes, B.supported_kv_cache_dtypes
assert B.supports_kv_cache_dtype('fp8'), "DiffKV still rejects fp8"

# qkv fix markers present in the two source files (source-level, robust)
mo = open("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py").read()
assert "EXPERIMENTAL_MXFP8_WEIGHT_SCALE_INV_ALIAS" in mo, "MXFP8 weight_scale_inv alias not applied"
mm = open("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py").read()
assert "EXPERIMENTAL_MIMO_QKV_FALLTHROUGH" in mm, "mimo_v2 qkv fall-through not applied"

# both patched modules still import
import vllm.model_executor.layers.quantization.modelopt as _m
assert hasattr(_m, "ModelOptMxFp8LinearMethod"), "ModelOptMxFp8LinearMethod missing"
import vllm.model_executor.models.mimo_v2 as _mm2
# _try_load_fp8_qkv_proj lives on MiMoV2Model (the module that runs load_weights)
assert hasattr(_mm2, "MiMoV2Model"), "MiMoV2Model missing"
import inspect
_src = inspect.getsource(_mm2.MiMoV2Model._try_load_fp8_qkv_proj)
# early 'return False' must precede the is_weight gate (i.e. fused qkv falls through)
_head = _src.split("is_weight = (")[0]
assert "return False" in _head, \
    "early 'return False' not before is_weight gate in _try_load_fp8_qkv_proj"
# and the MXFP8 alias must live inside ModelOptMxFp8LinearMethod.create_weights
assert "EXPERIMENTAL_MXFP8_WEIGHT_SCALE_INV_ALIAS" in inspect.getsource(
    _m.ModelOptMxFp8LinearMethod.create_weights
), "MXFP8 weight_scale_inv alias not inside ModelOptMxFp8LinearMethod.create_weights"

print('[fix-mimo-v2-upstream] validation OK (mimo_v2 config + DiffKV fp8 + MXFP8 scale-inv alias + qkv fall-through)')
PY
[ $? -eq 0 ] || { echo "[fix-mimo-v2-upstream] VALIDATION FAILED — aborting"; exit 1; }

echo "[fix-mimo-v2-upstream] done"
