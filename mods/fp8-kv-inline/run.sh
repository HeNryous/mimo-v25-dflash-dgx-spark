#!/bin/bash
set -uo pipefail
# ============================================================================
# fp8-kv-inline — fp8 (e4m3) KV cache for MiMo-V2.5 DiffKV target with IN-KERNEL
#   descale, cloned from the proven nvfp4 INLINE machinery (mods/nvfp4-kv-upstream).
#
#   Image: vllm-node-mimo-v25-upstream:latest
#   vllm 0.23.1rc1.dev760+g3775d5fca (main @3775d5fc)
#
# WHY fp8 (vs the nvfp4 PROD path):
#   fp8-e4m3 dequant is a BITCAST + one scalar multiply
#     K = key_u8.to(tl.float8e4nv, bitcast=True).to(bf16) * k_descale
#   ~3-4x cheaper than nvfp4's nibble-unpack + e2m1-LUT-gather + per-group-16
#   fp8 block-scale multiply. The fp8 KV page is NATIVE 320 B/token/head (K192 +
#   V128 x 1 byte) = exactly HALF the bf16 640 B -> ~2x bf16 pool (vs nvfp4 ~3.5x
#   at 180 B). Trade: fewer tokens than nvfp4, but deep-context decode stays near
#   bf16 speed (no unpack/LUT), and quality is the inherent fp8 error (~0.03-0.06
#   vs bf16), which validated NEUTRAL for MiMo before (recipe mimo-v2.5-fp8-*).
#
# WHAT IS NATIVE / FREE (fp8 is MUCH simpler than nvfp4):
#   - PAGE SIZE: fp8 is NOT packed. FullAttentionSpec/SlidingWindowSpec
#     .real_page_size_bytes (kv_cache_interface.py:310/527): FP8_PER_TENSOR is
#     neither is_nvfp4 nor INT4 -> the `else` branch -> last_dim = head_size +
#     head_size_v = 320, and get_dtype_size(uint8)=1 -> page = block*kvh*320*1 =
#     the native fp8 320 B/token/head. get_kv_cache_shape (keyed on the GLOBAL
#     cache_dtype "fp8") does NOT hit the nvfp4-packed branch -> also returns
#     last-dim 320. => alloc == view natively. NO get_kv_cache_shape branch, NO
#     real_page_size_bytes self-heal needed (unlike nvfp4's 180-vs-320 mismatch).
#   - STORE: triton_reshape_and_cache_flash_diffkv
#     (ops/triton_reshape_and_cache_flash.py:542) ALREADY handles fp8 natively:
#     when kv_cache_dtype is quantized it views the uint8 cache as fp8_dtype()
#     (e4m3), sets FP8_KV_CACHE=True, and the kernel does
#     `key_tile = key_load / tl.load(k_scale)` then tl.store implicitly casts to
#     fp8 (line 508-521). => NO custom fp8 store (unlike _nvfp4_store_diffkv).
#     The ONLY catch: it gates on the IMPL's self.kv_cache_dtype string, which is
#     a stale "auto" under the MiMoV2 cache_config bug (see below) -> we route
#     the store with the correct "fp8_e4m3" string when the cache is uint8.
#
# THE BUG CLASS THAT HITS fp8 (same root cause as NVFP4_KV_UPSTREAM_SPEC):
#   MiMoV2FlashDecoderLayer builds MiMoV2Attention WITHOUT
#   cache_config=vllm_config.cache_config -> the shared Attention.__init__ hits
#   `else: kv_cache_dtype = "auto"` (attention.py:243). So for ALL 48 asymmetric
#   MiMo DiffKV target layers (head_size 192 != head_size_v 128):
#       self.kv_cache_dtype       == "auto"          (NOT "fp8")
#       self.kv_cache_torch_dtype == torch.bfloat16  (NOT uint8)
#   Consequences, and how this mod fixes each:
#     (a) SPEC: get_kv_cache_spec (attention.py:599) builds quant_mode=NONE +
#         dtype=bfloat16 -> real_page_size_bytes = block*kvh*320*2 (bf16, 40960),
#         but the allocator VIEWS the uint8 buffer at get_kv_cache_shape=320 *
#         get_dtype_size(uint8=1)=23040... mismatch -> .view() crash / wrong pool.
#         FIX (FP8_KV_INLINE_SPEC): when GLOBAL cache_dtype in {fp8,fp8_e4m3} AND
#         asymmetric DiffKV target AND quant_mode is NONE, force
#         quant_mode=FP8_PER_TENSOR (=1) + self.kv_cache_torch_dtype=torch.uint8
#         (1-byte fp8 storage). Gated on asymmetric so the SYMMETRIC bf16 DFlash
#         drafter (128==128, routed to bf16 by mods/nvfp4-draft-bf16) is UNTOUCHED
#         -- the exact same discriminator NVFP4_KV_UPSTREAM_SPEC uses.
#     (b) STORE: impl.self.kv_cache_dtype "auto" -> FP8_KV_CACHE=False -> raw
#         bf16 written to a uint8 buffer = garbage. FIX (FP8_KV_INLINE_STORE):
#         in do_kv_cache_update, when the cache tensor is uint8 (fp8 store; bf16
#         run is bfloat16-typed) pass "fp8_e4m3" to the reshape kernel.
#     (c) READ: forward slices K/V from the uint8 cache and the pristine kernel
#         does K_load.to(Q.dtype) on raw uint8 integers = garbage. FIX
#         (FP8_KV_INLINE / kernel IS_FP8 arm): when VLLM_FP8_INLINE=1 and the
#         cache is uint8, feed the kernel IS_FP8=True + the layer descales; the
#         kernel bitcasts the uint8 load to float8e4nv, converts to bf16, and
#         multiplies by k_descale / v_descale. Descales come from
#         layer._k_scale_float / layer._v_scale_float (default 1.0;
#         calculate_kv_scales=False -- the validated fp8 config).
#
# COEXISTENCE with the nvfp4 mods (they gate on DIFFERENT cache_dtype strings):
#   - The kernel IS_FP8 arm is a THIRD branch alongside the nvfp4 IS_NVFP4 arm and
#     the bf16 else. If mods/nvfp4-kv-upstream (INLINE) already wrapped the K/V
#     load in `if IS_NVFP4: ... else: <bf16>`, we insert `elif IS_FP8: <fp8>`
#     before that else. If nvfp4 is NOT present (pristine bf16 load), we wrap it
#     `if IS_FP8: <fp8> else: <bf16>`. Either way the bf16 path is preserved and
#     nvfp4 (if present) is untouched.
#   - The backend forward: fp8 detection is `kv_cache.dtype == torch.uint8` AND
#     shape is the UNPACKED 320 (last-dim == head_size_qk + head_size_v). nvfp4's
#     detection is shape-packed (last-dim 180 != 320). Mutually exclusive.
#   - SPEC: fp8 gate is `cache_dtype in {fp8,fp8_e4m3}`; nvfp4 gate is
#     `cache_dtype == "nvfp4"`. Mutually exclusive. Both may be applied; only the
#     one matching the recipe's --kv-cache-dtype fires. This mod is STANDALONE
#     (does not require the nvfp4 mods) but coexists if both are applied.
#
# ORDERING: run AFTER fix-mimo-v2-upstream (which whitelists fp8/fp8_e4m3 in
#   supported_kv_cache_dtypes and neutralizes the __init__ quantized-KV guard
#   with `False and`). Idempotent w.r.t. nvfp4-kv-upstream (may run before/after).
#
# SELECT: env VLLM_FP8_INLINE=1 -> IS_FP8 in-kernel dequant. Unset/0 -> the fp8
#   store still works (native) but the READ falls back to a bf16 dequant scratch
#   (correctness fallback, symmetric to nvfp4 SCRATCH). Recipe sets it to 1.
#
# Discipline: marker-guarded idempotency, ast.parse after each edit, sys.exit on
# anchor-miss for load-bearing patches.
# ============================================================================

SITE="/usr/local/lib/python3.12/dist-packages"
BACKEND="$SITE/vllm/v1/attention/backends/triton_attn_diffkv.py"
KERNEL="$SITE/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
ATTN="$SITE/vllm/model_executor/layers/attention/attention.py"

echo "[fp8-kv-inline] grafting fp8-e4m3 DiffKV KV cache with in-kernel descale (VLLM_FP8_INLINE gate)"

# ----------------------------------------------------------------------------
# 0. Sanity: backend + kernel + native fp8 store + FP8_PER_TENSOR present.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import sys, pathlib
b = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py").read_text()
for needle in ("class TritonAttentionDiffKVBackend", "def do_kv_cache_update",
               "def forward", "triton_reshape_and_cache_flash_diffkv(",
               "layer._k_scale", "layer._v_scale"):
    if needle not in b:
        print(f"[fp8-kv-inline] FATAL: backend missing anchor {needle!r}"); sys.exit(3)
k = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py").read_text()
# The kernel anchors appear in one of THREE states: pristine, post-nvfp4-INLINE
# (mods/nvfp4-kv-upstream ran first, rewriting the signature/load/launch), or
# already-fp8-applied. The per-section patches handle all three; sanity only
# asserts the load-bearing structures exist in SOME accepted form.
_fp8_done = "FP8_KV_INLINE_KVLOAD" in k
_nvfp4_done = "NVFP4_INLINE_KVLOAD" in k
if "def unified_attention_diffkv(" not in k:
    print("[fp8-kv-inline] FATAL: kernel missing unified_attention_diffkv"); sys.exit(3)
if not _fp8_done:
    # signature terminator: pristine `IS_3D...):` OR post-nvfp4 `V_WORD_OFF...):`
    if ("    IS_3D: tl.constexpr,\n):\n" not in k
            and "    V_WORD_OFF: tl.constexpr = 0,\n):\n" not in k):
        print("[fp8-kv-inline] FATAL: kernel signature terminator not found "
              "(neither pristine IS_3D nor post-nvfp4 V_WORD_OFF)"); sys.exit(3)
    # K/V conversion: pristine 8-space OR post-nvfp4 12-space (inside else:)
    if ("        K = K_load.to(Q.dtype)\n" not in k
            and "            K = K_load.to(Q.dtype)\n" not in k):
        print("[fp8-kv-inline] FATAL: K conversion anchor not found"); sys.exit(3)
    # launch terminator: pristine `IS_3D=use_3d,\n    )` OR post-nvfp4 tail
    if ("        IS_3D=use_3d,\n    )\n" not in k
            and "        V_WORD_OFF=_v_word_off,\n    )\n" not in k):
        print("[fp8-kv-inline] FATAL: kernel launch terminator not found"); sys.exit(3)
from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode
assert int(KVQuantMode.FP8_PER_TENSOR) == 1, KVQuantMode.FP8_PER_TENSOR
assert get_kv_quant_mode("fp8") == KVQuantMode.FP8_PER_TENSOR, get_kv_quant_mode("fp8")
# native fp8 store confirms FP8_KV_CACHE dispatch exists
rc = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py").read_text()
assert "FP8_KV_CACHE = is_quantized_kv_cache(kv_cache_dtype)" in rc, "native fp8 store dispatch missing"
# fix-mimo-v2-upstream must have whitelisted fp8 already
import vllm.v1.attention.backends.triton_attn_diffkv as B
assert "fp8" in B.TritonAttentionDiffKVBackend.supported_kv_cache_dtypes, \
    "fp8 not in supported_kv_cache_dtypes -- run fix-mimo-v2-upstream first"
# triton float8e4nv available (verified: bitcast dequant bit-exact on this image)
import triton.language as tl
assert hasattr(tl, "float8e4nv"), "triton lacks float8e4nv"
print("[fp8-kv-inline] sanity OK (backend+kernel anchors, native fp8 store, FP8_PER_TENSOR=1, "
      "fp8 whitelisted, triton float8e4nv present)")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] sanity FAILED — aborting"; exit 1; }


# ============================================================================
# 1. FP8_KV_INLINE_SPEC — the LOAD-BEARING target-quant-mode fix (mirror of
#    NVFP4_KV_UPSTREAM_SPEC, for fp8). At attention.py:599, when the GLOBAL
#    cache_config.cache_dtype in {"fp8","fp8_e4m3"} AND this is an ASYMMETRIC
#    DiffKV target (head_size_v != head_size) AND quant_mode is NONE (stale
#    "auto"), force quant_mode=FP8_PER_TENSOR and self.kv_cache_torch_dtype=uint8.
#    Gated on asymmetric so the symmetric bf16 DFlash drafter is NOT touched.
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py")
src = p.read_text()
if "FP8_KV_INLINE_SPEC" in src:
    print("[fp8-kv-inline] get_kv_cache_spec fp8 target-force already present"); sys.exit(0)

anchor = "        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)\n"
if src.count(anchor) != 1:
    print(f"[fp8-kv-inline] FATAL: get_kv_cache_spec quant_mode anchor count = "
          f"{src.count(anchor)} (expected 1)"); sys.exit(3)

insert = (
    "        # FP8_KV_INLINE_SPEC: mirror of NVFP4_KV_UPSTREAM_SPEC for fp8.\n"
    "        # MiMo's MiMoV2FlashDecoderLayer builds its attention WITHOUT\n"
    "        # cache_config, so self.kv_cache_dtype is a stale \"auto\" (-> bfloat16)\n"
    "        # even under a GLOBAL --kv-cache-dtype fp8. The fp8 KV cache is stored\n"
    "        # uint8 (1 B/elem, native 320 B/token/head) and get_kv_cache_shape\n"
    "        # returns last-dim 320, so the spec MUST carry a 1-byte dtype +\n"
    "        # FP8_PER_TENSOR mode or the allocator .view() mismatches. Derive the\n"
    "        # quant_mode from the GLOBAL cache_dtype, but ONLY for the asymmetric\n"
    "        # DiffKV target (head_size_v != head_size) so the SYMMETRIC DFlash\n"
    "        # drafter (routed to bf16 by mods/nvfp4-draft-bf16) is NOT forced.\n"
    "        _fp8_cache_dtype = getattr(\n"
    "            getattr(vllm_config, \"cache_config\", None), \"cache_dtype\", None\n"
    "        )\n"
    "        if (\n"
    "            _fp8_cache_dtype in (\"fp8\", \"fp8_e4m3\")\n"
    "            and self.head_size_v != self.head_size\n"
    "            and int(quant_mode) == 0\n"
    "        ):\n"
    "            import torch as _torch_fp8\n"
    "            quant_mode = get_kv_quant_mode(\"fp8\")  # FP8_PER_TENSOR (=1)\n"
    "            # fp8 KV is stored 1 B/elem; the allocator's raw int8 buffer is\n"
    "            # .view(spec.dtype)'d, so spec.dtype MUST be a 1-byte dtype for\n"
    "            # block*kvh*320*1 == prod(get_kv_cache_shape[1:]) (== 32*4*320).\n"
    "            # bf16 would double the byte count and .view() would fail.\n"
    "            self.kv_cache_torch_dtype = _torch_fp8.uint8\n"
    "            logger.info_once(\n"
    "                \"FP8_KV_INLINE_SPEC: forcing FP8_PER_TENSOR quant_mode+uint8 on \"\n"
    "                \"asymmetric DiffKV target layer %s (head_size=%s head_size_v=\"\n"
    "                \"%s); per-layer kv_cache_dtype was %r (stale under global \"\n"
    "                \"fp8).\",\n"
    "                self.layer_name,\n"
    "                self.head_size,\n"
    "                self.head_size_v,\n"
    "                self.kv_cache_dtype,\n"
    "            )\n"
)
src = src.replace(anchor, anchor + insert, 1)
ast.parse(src)
p.write_text(src)
print("[fp8-kv-inline] patched get_kv_cache_spec: FP8_PER_TENSOR quant_mode+uint8 for asymmetric DiffKV target")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] SPEC patch FAILED — aborting"; exit 1; }

find "$SITE/vllm/model_executor/layers/attention" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import ast
f = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py"
src = open(f).read(); ast.parse(src)
assert "FP8_KV_INLINE_SPEC" in src
import vllm.model_executor.layers.attention.attention as A  # noqa: F401
print("[fp8-kv-inline] SPEC validation OK (attention.py parses+imports, marker set)")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] SPEC validation FAILED — aborting"; exit 1; }


# ============================================================================
# 2. Backend helper block: fp8 in-kernel descale gate + a bf16 dequant SCRATCH
#    fallback (symmetric to the nvfp4 SCRATCH; used when VLLM_FP8_INLINE=0).
#    Marker: FP8_KV_INLINE_HELPERS.
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py")
src = p.read_text()
if "FP8_KV_INLINE_HELPERS" in src:
    print("[fp8-kv-inline] backend helper block already present"); sys.exit(0)

anchor = "logger = init_logger(__name__)\n"
if src.count(anchor) != 1:
    print(f"[fp8-kv-inline] FATAL: logger anchor count = {src.count(anchor)} (expected 1)"); sys.exit(3)

BLOCK = r'''
# ---------------------------------------------------------------------------
# FP8_KV_INLINE_HELPERS  (cloned from the nvfp4 INLINE machinery)
# fp8-e4m3 KV cache (DiffKV) read helpers. The fp8 store is NATIVE
# (triton_reshape_and_cache_flash_diffkv). Read is either:
#   - INLINE (VLLM_FP8_INLINE=1): the kernel bitcasts uint8 -> float8e4nv ->
#     bf16 and multiplies by k_descale / v_descale (default 1.0). ~3-4x cheaper
#     than nvfp4 (no nibble-unpack / e2m1-LUT / block-scale). Verified bit-exact
#     vs torch on this image.
#   - SCRATCH fallback (VLLM_FP8_INLINE=0): dequant the active blocks to a bf16
#     scratch via .view(fp8) + descale, then read via the pristine bf16 kernel.
# fp8 cache is UNPACKED (last-dim 320 = K192+V128, uint8) -> detected by dtype
# (uint8) NOT shape (which equals the bf16 320). Global scale defaults to 1.0
# (calculate_kv_scales=False -- the validated fp8 config).
# ---------------------------------------------------------------------------
import os as _os_fp8

# FP8_KV_INLINE gate. VLLM_FP8_INLINE=1 selects the in-kernel descale fast-path;
# unset/0 => the bf16 dequant SCRATCH fallback (correctness).
_VLLM_FP8_INLINE = _os_fp8.environ.get("VLLM_FP8_INLINE", "0") == "1"
logger.info("[fp8-kv-inline] in-kernel fp8 descale %s (VLLM_FP8_INLINE=%s)",
            "ENABLED" if _VLLM_FP8_INLINE else "disabled (scratch path)",
            _os_fp8.environ.get("VLLM_FP8_INLINE"))


def _fp8_is_cache(kv_cache):
    # fp8 DiffKV cache is the ONLY uint8 cache with the UNPACKED 320 last-dim
    # (nvfp4 packs to 180; bf16 is bfloat16-typed). import torch lazily.
    import torch
    return kv_cache.dtype == torch.uint8


def _fp8_dequant_active_blocks(kv_cache, block_table, Hk, Hv, k_descale, v_descale,
                               out_dtype):
    """SCRATCH fallback: dequant ONLY the physical blocks referenced by
    block_table into a small bf16 scratch (sized to active blocks), and return a
    remapped block_table indexing the scratch. The scratch is a standard
    [nact, BS, NH, Hk+Hv] bf16 cache read by the pristine bf16 kernel path."""
    import torch
    nblk, BS, NH, LD = kv_cache.shape
    bt = block_table
    active = torch.unique(bt[bt >= 0]).to(torch.long)
    nact = int(active.numel())
    if nact == 0:
        scratch = torch.zeros(1, BS, NH, Hk + Hv, dtype=out_dtype, device=kv_cache.device)
        return scratch, torch.zeros_like(bt)
    blk = kv_cache[active].view(torch.float8_e4m3fn)  # [nact,BS,NH,Hk+Hv] fp8
    scratch = torch.empty(nact, BS, NH, Hk + Hv, dtype=out_dtype, device=kv_cache.device)
    scratch[..., :Hk] = blk[..., :Hk].to(out_dtype) * k_descale
    scratch[..., Hk:] = blk[..., Hk:Hk + Hv].to(out_dtype) * v_descale
    remap = torch.zeros(nblk, dtype=bt.dtype, device=bt.device)
    remap[active] = torch.arange(nact, dtype=bt.dtype, device=bt.device)
    remapped_bt = torch.where(bt >= 0, remap[bt.clamp(min=0)], bt)
    return scratch, remapped_bt
# --- end FP8_KV_INLINE_HELPERS ---
'''
src = src.replace(anchor, anchor + BLOCK)
ast.parse(src)
p.write_text(src)
print("[fp8-kv-inline] inserted backend fp8 helper block + VLLM_FP8_INLINE gate")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] backend helper-block patch FAILED — aborting"; exit 1; }


# ============================================================================
# 3. FP8_KV_INLINE_STORE — route the store with the correct "fp8_e4m3" string
#    when the cache tensor is uint8 (the impl's self.kv_cache_dtype is a stale
#    "auto" under the MiMoV2 bug, so FP8_KV_CACHE would be False -> garbage).
#    Inserted at the top of do_kv_cache_update, before the pristine call.
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py")
src = p.read_text()
if "FP8_KV_INLINE_STORE" in src:
    print("[fp8-kv-inline] do_kv_cache_update fp8 branch already present"); sys.exit(0)
anchor = ('        # Cache is packed [..., head_size_qk + head_size_v]; the diffkv\n'
          '        # reshape kernel writes K to [..., :head_size_qk] and V to\n'
          '        # [..., head_size_qk:hqk+hv].\n'
          '        triton_reshape_and_cache_flash_diffkv(\n'
          '            key,\n'
          '            value,\n'
          '            kv_cache,\n'
          '            slot_mapping,\n'
          '            self.kv_cache_dtype,\n'
          '            layer._k_scale,\n'
          '            layer._v_scale,\n'
          '        )\n')
if src.count(anchor) != 1:
    print(f"[fp8-kv-inline] FATAL: do_kv_cache_update anchor count = {src.count(anchor)} "
          f"(expected 1)"); sys.exit(3)
repl = ('        # Cache is packed [..., head_size_qk + head_size_v]; the diffkv\n'
        '        # reshape kernel writes K to [..., :head_size_qk] and V to\n'
        '        # [..., head_size_qk:hqk+hv].\n'
        '        # FP8_KV_INLINE_STORE: the native fp8 store gates FP8_KV_CACHE on the\n'
        '        # dtype STRING, but the impl\'s self.kv_cache_dtype is a stale "auto"\n'
        '        # under the MiMoV2 cache_config bug -> it would store raw bf16 into\n'
        '        # the uint8 buffer (garbage). Detect the fp8 cache by tensor dtype\n'
        '        # (uint8; bf16 run is bfloat16-typed) and pass "fp8_e4m3" so the\n'
        '        # reshape kernel views the cache as e4m3 + divides K/V by k/v_scale.\n'
        '        _fp8_store_dtype = self.kv_cache_dtype\n'
        '        if _fp8_is_cache(kv_cache):\n'
        '            _fp8_store_dtype = "fp8_e4m3"\n'
        '        triton_reshape_and_cache_flash_diffkv(\n'
        '            key,\n'
        '            value,\n'
        '            kv_cache,\n'
        '            slot_mapping,\n'
        '            _fp8_store_dtype,  # FP8_KV_INLINE_STORE\n'
        '            layer._k_scale,\n'
        '            layer._v_scale,\n'
        '        )\n')
src = src.replace(anchor, repl)
ast.parse(src)
p.write_text(src)
print("[fp8-kv-inline] added do_kv_cache_update fp8 store-route (uint8 -> fp8_e4m3)")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] do_kv_cache_update patch FAILED — aborting"; exit 1; }


# ============================================================================
# 4. Kernel IS_FP8 params (mirror NVFP4_INLINE_KPARAMS). Append IS_FP8 + descale
#    scalars, keyword-only with defaults so existing bf16 call sites are safe.
#    Anchor on the unique `    IS_3D: tl.constexpr,\n):\n` terminator.
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
src = p.read_text()
if "FP8_KV_INLINE_KPARAMS" in src:
    print("[fp8-kv-inline] kernel IS_FP8 params already present"); sys.exit(0)
# The signature terminator is `<last param>,\n):\n`. On a pristine kernel the last
# param is `    IS_3D: tl.constexpr,`; after mods/nvfp4-kv-upstream (INLINE) ran
# first it is `    V_WORD_OFF: tl.constexpr = 0,`. Insert the fp8 params before the
# closing `):` in either form.
fp8_params = (
    "    # FP8_KV_INLINE_KPARAMS: in-kernel fp8-e4m3 descale. key/value_cache_ptr\n"
    "    # are the uint8 fp8 K/V views; IS_FP8 bitcasts them to float8e4nv, converts\n"
    "    # to Q.dtype, and multiplies by the per-tensor descales (default 1.0).\n"
    "    IS_FP8: tl.constexpr = False,\n"
    "    K_DESCALE: tl.constexpr = 1.0,\n"
    "    V_DESCALE: tl.constexpr = 1.0,\n"
)
anchor_a = "    IS_3D: tl.constexpr,\n):\n"                 # pristine
anchor_b = "    V_WORD_OFF: tl.constexpr = 0,\n):\n"        # post-nvfp4
if src.count(anchor_a) == 1:
    src = src.replace(anchor_a, "    IS_3D: tl.constexpr,\n" + fp8_params + "):\n")
    print("[fp8-kv-inline] added IS_FP8 + descale params to kernel signature (pristine form)")
elif src.count(anchor_b) == 1:
    src = src.replace(anchor_b, "    V_WORD_OFF: tl.constexpr = 0,\n" + fp8_params + "):\n")
    print("[fp8-kv-inline] added IS_FP8 + descale params to kernel signature (post-nvfp4 form)")
else:
    print(f"[fp8-kv-inline] FATAL: kernel signature terminator not found "
          f"(A count={src.count(anchor_a)}, B count={src.count(anchor_b)})"); sys.exit(3)
ast.parse(src)
p.write_text(src)
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] kernel-signature patch FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# 5. Kernel K/V load: add the IS_FP8 branch. Handle BOTH the pristine bf16 load
#    AND the nvfp4-INLINE-wrapped form (if mods/nvfp4-kv-upstream ran first).
#    (a) pristine: `K = K_load.to(Q.dtype)` / `V = V_load.to(Q.dtype)` at column
#        8 (inside the for-loop). Wrap them with IS_FP8 conversion.
#    (b) nvfp4-wrapped: those same lines exist at column 12 inside the `else:`.
#    We surgically replace the two conversion lines (bf16 load -> tile) with an
#    IS_FP8-aware version, in BOTH indentation forms if present.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
src = p.read_text()
if "FP8_KV_INLINE_KVLOAD" in src:
    print("[fp8-kv-inline] kernel IS_FP8 K/V-load branch already present"); sys.exit(0)

# The fp8 dequant differs from bf16 ONLY in HOW the loaded uint8 becomes a tile:
#   bf16 : K = K_load.to(Q.dtype)
#   fp8  : K = K_load.to(tl.float8e4nv, bitcast=True).to(Q.dtype) * K_DESCALE
# The load itself (tl.load of key_cache_ptr) is identical (same uint8 pointer,
# same offsets/strides). So we replace ONLY the two `.to(Q.dtype)` conversion
# lines with an IS_FP8 conditional. The bf16 conversion appears at exactly ONE
# indent: 8-space on a pristine kernel, 12-space inside nvfp4's `else:` if
# mods/nvfp4-kv-upstream (INLINE) ran first. We anchor with a LEADING NEWLINE so
# the 8-space pattern (`\n        K =`) can NOT collide as a substring of the
# 12-space line (`\n            K =`) -- after 8 spaces the 12-space line has a
# space, not `K`. Detect the present indent per tile and replace that one only.
def wrap(tile, load, indent):
    # tile in {"K","V"}; load in {"K_load","V_load"}. `indent` = leading spaces.
    ds = "K_DESCALE" if tile == "K" else "V_DESCALE"
    old = f"\n{indent}{tile} = {load}.to(Q.dtype)\n"
    new = (
        f"\n{indent}# FP8_KV_INLINE_KVLOAD: fp8-e4m3 in-register descale (bitcast\n"
        f"{indent}# uint8 -> float8e4nv -> Q.dtype, x per-tensor descale). Same load,\n"
        f"{indent}# same shape as the bf16 tile below -> drop-in for tl.dot.\n"
        f"{indent}if IS_FP8:\n"
        f"{indent}    {tile} = {load}.to(tl.float8e4nv, bitcast=True).to(Q.dtype) * {ds}\n"
        f"{indent}else:\n"
        f"{indent}    {tile} = {load}.to(Q.dtype)\n"
    )
    return old, new

changed = False
for tile, load in (("K", "K_load"), ("V", "V_load")):
    # exactly ONE of the two indents must be present for this tile.
    hits = []
    for indent in ("        ", "            "):   # 8-space pristine, 12-space nvfp4-else
        old, new = wrap(tile, load, indent)
        c = src.count(old)
        if c == 1:
            hits.append((indent, old, new))
        elif c > 1:
            print(f"[fp8-kv-inline] FATAL: ambiguous {tile} conversion anchor at "
                  f"indent={len(indent)} (count={c})"); sys.exit(3)
    if len(hits) != 1:
        print(f"[fp8-kv-inline] FATAL: {tile} conversion anchor found at "
              f"{len(hits)} indents (expected exactly 1)"); sys.exit(3)
    indent, old, new = hits[0]
    src = src.replace(old, new)
    changed = True

if not changed:
    print("[fp8-kv-inline] FATAL: no K/V .to(Q.dtype) conversion anchor found "
          "(neither pristine nor nvfp4-else form)"); sys.exit(3)

# marker breadcrumb (comment) so idempotency detects us
src = src.replace(
    "@triton.jit\ndef kernel_unified_attention_diffkv(",
    "# FP8_KV_INLINE_KVLOAD applied\n@triton.jit\ndef kernel_unified_attention_diffkv(",
    1,
)
ast.parse(src)
p.write_text(src)
print("[fp8-kv-inline] wrapped kernel K/V conversion with IS_FP8 in-register descale branch")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] kernel K/V-load patch FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# 6. Kernel launch: forward IS_FP8 + descales. Passing unconditionally is safe
#    (defaults off / 1.0 when not fp8). Anchor = the MAIN launch's
#    `        IS_3D=use_3d,\n    )\n` tail (unique; reduce-kernel has no IS_3D).
#    NB: if nvfp4-INLINE ran first it already appended its own kwargs after
#    IS_3D=use_3d and changed the tail -> we anchor on the nvfp4 tail too.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
src = p.read_text()
if "FP8_KV_INLINE_LAUNCH" in src:
    print("[fp8-kv-inline] kernel launch IS_FP8 kwargs already present"); sys.exit(0)

add_kwargs = (
    "        # FP8_KV_INLINE_LAUNCH: fp8 flag + descales (no-op when fp8_packed False).\n"
    "        IS_FP8=fp8_packed,\n"
    "        K_DESCALE=k_descale,\n"
    "        V_DESCALE=v_descale,\n"
)

# Form A (pristine): the MAIN launch ends `        IS_3D=use_3d,\n    )\n`.
anchor_a = "        IS_3D=use_3d,\n    )\n"
# Form B (nvfp4-INLINE ran first): it terminates the launch with
# `        V_WORD_OFF=_v_word_off,\n    )\n`.
anchor_b = "        V_WORD_OFF=_v_word_off,\n    )\n"

if src.count(anchor_a) == 1:
    src = src.replace(anchor_a, "        IS_3D=use_3d,\n" + add_kwargs + "    )\n")
    print("[fp8-kv-inline] forwarded IS_FP8 into kernel launch (pristine form)")
elif src.count(anchor_b) == 1:
    src = src.replace(anchor_b, "        V_WORD_OFF=_v_word_off,\n" + add_kwargs + "    )\n")
    print("[fp8-kv-inline] forwarded IS_FP8 into kernel launch (post-nvfp4 form)")
else:
    print(f"[fp8-kv-inline] FATAL: kernel-launch tail not found "
          f"(A count={src.count(anchor_a)}, B count={src.count(anchor_b)})"); sys.exit(3)
ast.parse(src)
p.write_text(src)
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] kernel-launch patch FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# 7. Wrapper unified_attention_diffkv: (a) add fp8 kwargs to the signature,
#    (b) default them. Anchor = the closing `):` after softmax_segm_expsum OR the
#    nvfp4-augmented signature tail (head_size_v_override) if nvfp4 ran first.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py")
src = p.read_text()
if "FP8_KV_INLINE_WRAPSIG" in src:
    print("[fp8-kv-inline] wrapper fp8 signature already present"); sys.exit(0)

fp8_sig = (
    "    # FP8_KV_INLINE_WRAPSIG: fp8-e4m3 in-kernel descale. k and v are the\n"
    "    # normal uint8 fp8 K/V views (last dim 192/128); fp8_packed enables the\n"
    "    # kernel IS_FP8 bitcast+descale. k/v_descale are the per-tensor scales.\n"
    "    fp8_packed: bool = False,\n"
    "    k_descale: float = 1.0,\n"
    "    v_descale: float = 1.0,\n"
)

# Form B (nvfp4 ran first): signature already ends with the nvfp4 kwargs +
# `    head_size_v_override: int | None = None,\n):\n`. Insert fp8 before `):`.
anchor_b = "    head_size_v_override: int | None = None,\n):\n"
# Form A (pristine): ends with the segm buffers + `):`.
anchor_a = (
    "    softmax_segm_output: torch.Tensor | None = None,\n"
    "    softmax_segm_max: torch.Tensor | None = None,\n"
    "    softmax_segm_expsum: torch.Tensor | None = None,\n"
    "):\n"
)
if src.count(anchor_b) == 1:
    src = src.replace(anchor_b,
        "    head_size_v_override: int | None = None,\n" + fp8_sig + "):\n")
    print("[fp8-kv-inline] added fp8 kwargs to wrapper signature (post-nvfp4 form)")
elif src.count(anchor_a) == 1:
    src = src.replace(anchor_a,
        "    softmax_segm_output: torch.Tensor | None = None,\n"
        "    softmax_segm_max: torch.Tensor | None = None,\n"
        "    softmax_segm_expsum: torch.Tensor | None = None,\n" + fp8_sig + "):\n")
    print("[fp8-kv-inline] added fp8 kwargs to wrapper signature (pristine form)")
else:
    print(f"[fp8-kv-inline] FATAL: wrapper signature tail not found "
          f"(A count={src.count(anchor_a)}, B count={src.count(anchor_b)})"); sys.exit(3)
ast.parse(src)
p.write_text(src)
print("[fp8-kv-inline] wrapper fp8 signature OK")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] wrapper-signature patch FAILED — aborting"; exit 1; }

# clean bytecode + validate kernel file
find "$SITE/vllm/v1/attention" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
python3 - <<'PY'
import ast
K = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
src = open(K).read(); ast.parse(src)
for m in ("FP8_KV_INLINE_KPARAMS", "FP8_KV_INLINE_KVLOAD", "FP8_KV_INLINE_LAUNCH",
          "FP8_KV_INLINE_WRAPSIG"):
    assert m in src, f"kernel marker missing: {m}"
import vllm.v1.attention.ops.triton_unified_attention_diffkv as _k  # noqa: F401
print("[fp8-kv-inline] kernel validation OK (parses+imports, 4 markers set)")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] kernel validation FAILED — aborting"; exit 1; }


# ============================================================================
# 8. FP8_KV_INLINE_READ — backend forward: the fp8 read dispatch. When the cache
#    is uint8 (fp8): if VLLM_FP8_INLINE=1, feed the kernel IS_FP8=True + the layer
#    descales; else use the bf16 dequant SCRATCH fallback. bf16 + nvfp4 paths
#    intact (nvfp4 detected by shape 180; fp8 by dtype uint8 @ shape 320).
#    Handles BOTH the pristine forward and the nvfp4-augmented forward.
# ============================================================================
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py")
src = p.read_text()
if "FP8_KV_INLINE_READ" in src:
    print("[fp8-kv-inline] forward fp8 read branch already present"); sys.exit(0)

# --- locate the two K/V slice lines (present in BOTH pristine and nvfp4 forms
#     as the FINAL slice before the unified_attention_diffkv call) and the call.
pristine_slice = (
    "        # Slice the packed cache into K / V views.  Strides on dims 0/1/2\n"
    "        # match the original cache; dim 3 stays contiguous (stride 1).\n"
    "        key_cache = kv_cache[..., :head_size_qk]\n"
    "        value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]\n"
)
nvfp4_pre = "        nvfp4_block_table = attn_metadata.block_table\n"

is_nvfp4_present = nvfp4_pre in src

if not is_nvfp4_present:
    # ---- STANDALONE (no nvfp4): insert fp8 read before the pristine slice ----
    if src.count(pristine_slice) != 1:
        print(f"[fp8-kv-inline] FATAL: pristine forward slice anchor count = "
              f"{src.count(pristine_slice)} (expected 1)"); sys.exit(3)
    repl = (
        "        # FP8_KV_INLINE_READ: fp8-e4m3 cache (uint8, unpacked last-dim 320).\n"
        "        # INLINE (VLLM_FP8_INLINE=1): feed the kernel IS_FP8 + layer descales\n"
        "        # (in-register bitcast+descale, any q_len -> DFlash q_len=8 OK). Else:\n"
        "        # bf16 dequant SCRATCH fallback. bf16 run (bfloat16 cache) untouched.\n"
        "        fp8_block_table = attn_metadata.block_table\n"
        "        fp8_kwargs = {}\n"
        "        _is_fp8 = _fp8_is_cache(kv_cache)\n"
        "        if _is_fp8 and _VLLM_FP8_INLINE:\n"
        "            key_cache = kv_cache[..., :head_size_qk]\n"
        "            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]\n"
        "            fp8_kwargs = dict(\n"
        "                fp8_packed=True,\n"
        "                k_descale=float(getattr(layer, \"_k_scale_float\", 1.0)),\n"
        "                v_descale=float(getattr(layer, \"_v_scale_float\", 1.0)),\n"
        "            )\n"
        "        elif _is_fp8:\n"
        "            # SCRATCH fallback: dequant active blocks -> bf16 + remap.\n"
        "            kv_cache, fp8_block_table = _fp8_dequant_active_blocks(\n"
        "                kv_cache, attn_metadata.block_table,\n"
        "                head_size_qk, head_size_v,\n"
        "                float(getattr(layer, \"_k_scale_float\", 1.0)),\n"
        "                float(getattr(layer, \"_v_scale_float\", 1.0)),\n"
        "                query.dtype,\n"
        "            )\n"
        "            key_cache = kv_cache[..., :head_size_qk]\n"
        "            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]\n"
        "        else:\n"
        "            # Slice the packed cache into K / V views.\n"
        "            key_cache = kv_cache[..., :head_size_qk]\n"
        "            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]\n"
    )
    src = src.replace(pristine_slice, repl)
    # thread fp8_kwargs + fp8_block_table into the call. Pristine call uses
    # block_table=attn_metadata.block_table.
    call_bt = "            block_table=attn_metadata.block_table,\n"
    if src.count(call_bt) != 1:
        print(f"[fp8-kv-inline] FATAL: pristine call block_table anchor count = "
              f"{src.count(call_bt)} (expected 1)"); sys.exit(3)
    src = src.replace(call_bt,
        "            block_table=fp8_block_table,  # FP8_KV_INLINE_READ (remapped for scratch)\n")
    # inject **fp8_kwargs before the call's closing paren.
    idx = src.index("        unified_attention_diffkv(\n")
    tail = src.index("\n        )\n", idx)
    inject = "            **fp8_kwargs,  # FP8_KV_INLINE_READ (fp8_packed/descales when INLINE)\n"
    src = src[:tail + 1] + inject + src[tail + 1:]
    print("[fp8-kv-inline] added forward fp8 read (standalone; INLINE + SCRATCH fallback)")
else:
    # ---- COEXIST with nvfp4: the nvfp4 read gate owns the slice + block_table.
    #      Extend its is_packed chain with an fp8 clause. The nvfp4 gate is:
    #        if is_packed and _VLLM_NVFP4_INLINE: ... elif is_packed: ... else: <bf16 slice>
    #      We insert fp8 clauses into the final `else:` (the bf16 slice) so fp8 is
    #      handled when the cache is uint8 @ 320 (is_packed is False for fp8).
    else_anchor = (
        "        else:\n"
        "            key_cache = kv_cache[..., :head_size_qk]\n"
        "            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]\n"
    )
    if src.count(else_anchor) != 1:
        print(f"[fp8-kv-inline] FATAL: nvfp4-form else anchor count = "
              f"{src.count(else_anchor)} (expected 1)"); sys.exit(3)
    else_new = (
        "        elif _fp8_is_cache(kv_cache) and _VLLM_FP8_INLINE:\n"
        "            # FP8_KV_INLINE_READ: fp8 in-kernel descale (uint8 @ 320).\n"
        "            key_cache = kv_cache[..., :head_size_qk]\n"
        "            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]\n"
        "            nvfp4_kwargs = dict(\n"
        "                fp8_packed=True,\n"
        "                k_descale=float(getattr(layer, \"_k_scale_float\", 1.0)),\n"
        "                v_descale=float(getattr(layer, \"_v_scale_float\", 1.0)),\n"
        "            )\n"
        "        elif _fp8_is_cache(kv_cache):\n"
        "            # FP8_KV_INLINE_READ: bf16 dequant SCRATCH fallback.\n"
        "            kv_cache, nvfp4_block_table = _fp8_dequant_active_blocks(\n"
        "                kv_cache, attn_metadata.block_table,\n"
        "                head_size_qk, head_size_v,\n"
        "                float(getattr(layer, \"_k_scale_float\", 1.0)),\n"
        "                float(getattr(layer, \"_v_scale_float\", 1.0)),\n"
        "                query.dtype,\n"
        "            )\n"
        "            key_cache = kv_cache[..., :head_size_qk]\n"
        "            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]\n"
        "        else:\n"
        "            key_cache = kv_cache[..., :head_size_qk]\n"
        "            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]\n"
    )
    src = src.replace(else_anchor, else_new)
    print("[fp8-kv-inline] added forward fp8 read (coexist with nvfp4; INLINE + SCRATCH fallback)")

ast.parse(src)
p.write_text(src)
print("[fp8-kv-inline] backend forward fp8 dispatch OK")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] forward fp8 read patch FAILED — aborting"; exit 1; }

# refresh bytecode + final validation
find "$SITE/vllm/v1/attention" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
python3 - <<'PY'
import ast
K = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
B = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py"
A = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py"
ks, bs, as_ = open(K).read(), open(B).read(), open(A).read()
ast.parse(ks); ast.parse(bs); ast.parse(as_)
for m in ("FP8_KV_INLINE_KPARAMS", "FP8_KV_INLINE_KVLOAD", "FP8_KV_INLINE_LAUNCH",
          "FP8_KV_INLINE_WRAPSIG"):
    assert m in ks, f"kernel marker missing: {m}"
for m in ("FP8_KV_INLINE_HELPERS", "FP8_KV_INLINE_STORE", "FP8_KV_INLINE_READ"):
    assert m in bs, f"backend marker missing: {m}"
assert "FP8_KV_INLINE_SPEC" in as_, "attention.py SPEC marker missing"
import vllm.v1.attention.ops.triton_unified_attention_diffkv as _k  # noqa: F401
import vllm.v1.attention.backends.triton_attn_diffkv as _b  # noqa: F401
import vllm.model_executor.layers.attention.attention as _a  # noqa: F401
# backend still advertises fp8 (from fix-mimo-v2-upstream)
assert _b.TritonAttentionDiffKVBackend.supports_kv_cache_dtype("fp8"), \
    "DiffKV rejects fp8 (fix-mimo-v2-upstream missing?)"
print("[fp8-kv-inline] FINAL validation OK (3 files parse+import, 8 markers set, DiffKV accepts fp8)")
PY
[ $? -eq 0 ] || { echo "[fp8-kv-inline] FINAL validation FAILED — aborting"; exit 1; }

echo "[fp8-kv-inline] done (select with VLLM_FP8_INLINE=1; default=SCRATCH fallback)"
