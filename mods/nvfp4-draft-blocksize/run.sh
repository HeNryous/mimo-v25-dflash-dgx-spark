#!/bin/bash
set -uo pipefail
# ============================================================================
# nvfp4-draft-blocksize — shrink the DFlash drafter's KV block_size (32 -> 16)
#   so its bf16 KV page stops inflating the TARGET's nvfp4 KV pool.
#
#   Image: vllm-node-mimo-v25-upstream:latest
#   vllm 0.23.1rc1.dev760+g3775d5fca (main @3775d5fc)
#
# THE PROBLEM (measured live, boot 2026-07-05 16:19, recipe mimo-nvfp4kv-test):
#   The MiMo nvfp4 DiffKV target + the bf16 DFlash drafter live in ONE uniform-
#   page KV pool. Their per-block page sizes (per TP rank, block_size 32) are:
#       target FULL (9 layers, kvh2, 180 packed, uint8) : 11,520 B/block
#       target SWA  (39 layers, kvh4, 180 packed, uint8): 23,040 B/block
#       drafter     (5 layers, kvh4, 128+128, bf16)     : 65,536 B/block  <-- MAX
#   Every attention backend here returns indexes_kv_by_block_stride()=True
#   (triton_attn / triton_attn_diffkv get_kv_cache_stride_order -> layered[0]!=0),
#   so kv_cache_utils.unify_kv_cache_spec_page_size (kv_cache_utils.py:1085-1088)
#   PADS every spec's page UP to the group max (65,536) via page_size_padded.
#   => num_blocks = available_memory / (65,536 * group_size): the target's real
#   23,040/11,520 nvfp4 pages are billed at 65,536 (2.84x / 5.69x bloat), so the
#   nvfp4 saving never materializes. Live pool = 66,313 tokens @ util0.80/mml32K
#   (== bf16 baseline 70,854; the ~3.5x nvfp4 pool is gone).
#   Reproduced EXACTLY on a throwaway (same code): drafter bf16 bs32 -> 66,239
#   tokens / 2.02x, two padding warnings 11.11% + 2.56% (== live boot).
#
# THE FIX (this mod): build the DFLASH DRAFTER's KVCacheSpec with block_size 16
#   instead of the global 32. The drafter's bf16 KV is symmetric 128/128 kvh4,
#   so its per-token footprint (2,048 B/token/layer) is intrinsically ~2.8x the
#   target SWA nvfp4 (720 B/token/layer); it can NEVER be smaller per-block at
#   equal block_size. Halving the drafter's block_size halves its page to
#   32,768, which becomes the new group max, so the target pool is billed at
#   32,768 instead of 65,536. Measured on the throwaway via the REAL
#   get_kv_cache_configs path: 66,239 -> 124,390 tokens / 3.80x (1.88x pool
#   gain), common page 32,768.
#
#   This is the bf16 OPTIMUM for a single uniform-page pool: TRITON_ATTN
#   (the drafter backend) requires block_size % 16 == 0
#   (triton_attn.py:294 supports_block_size), so 16 is the smallest legal draft
#   block_size; block_size 8 (which would drop the drafter page to 16,384 <
#   23,040 and free the FULL ~254K nvfp4 ceiling) is REJECTED by the backend.
#   Reaching >124K requires either a drafter KV precision drop (fp8, an
#   acceptance risk) or the packed cross-layer allocation path (needs a boot
#   A/B) -- both are follow-ups, deliberately NOT done here.
#
# WHY IT'S SAFE FOR THE DRAFTER:
#   The DFlash speculator reads its OWN group's block size at runtime
#   (speculator.py:138  self.draft_block_size = block_tables.block_sizes[
#   draft_kv_cache_group_id]) and uses it consistently in every context/query
#   slot computation (speculator.py:482-506 ctx_slot = ctx_block_id*block_size
#   + ...; precompute_and_store_context_kv). There is NO assumption anywhere
#   that draft_block_size == target block_size. The runner initializes per-group
#   block sizes natively (gpu_model_runner.py:6991-6998 block_sizes list, one
#   per KV cache group -> InputBatch(block_sizes=...)). block_size 16 is a clean
#   divisor of the target 32. => transparent to correctness; only the drafter's
#   KV tiling granularity changes (more, smaller blocks). The TARGET path
#   (nvfp4-kv-upstream) is UNTOUCHED (target layers keep block_size 32).
#
# THE DISCRIMINATOR (drafter-only, in get_kv_cache_spec):
#   self.attn_backend.get_name() == "TRITON_ATTN". The MiMo target uses
#   TRITON_ATTN_DIFFKV (asymmetric head_size 192 != head_size_v 128, auto-picks
#   DIFFKV; recipe also forces --attention-backend triton_attn_diffkv). The
#   drafter (Qwen3, per mods/nvfp4-draft-bf16 routed to bf16 + plain TRITON_ATTN)
#   is the ONLY TRITON_ATTN layer in this model. Belt-and-suspenders: also gate
#   on symmetry (head_size == head_size_v) so a hypothetical asymmetric
#   TRITON_ATTN layer is never touched. Only shrink when block_size > 16 (never
#   grow, never go below the backend's minimum).
#
# ORDERING: independent of nvfp4-draft-bf16 (that flips the drafter's KV DTYPE
#   in __init__; this changes the drafter's KV BLOCK_SIZE in get_kv_cache_spec).
#   Apply either order. Must run AFTER the image's get_kv_cache_spec exists
#   (it does, upstream) and is compatible with the NVFP4_KV_UPSTREAM_SPEC patch
#   (that patch only touches quant_mode/dtype for the ASYMMETRIC target and
#   returns before the FullAttentionSpec/SlidingWindowSpec build that reads
#   block_size).
#
# Discipline: marker-guarded idempotency, ast.parse after edit, sys.exit on
#   anchor-miss (load-bearing), import + logic self-test.
# ============================================================================

SITE="/usr/local/lib/python3.12/dist-packages"
ATTN="$SITE/vllm/model_executor/layers/attention/attention.py"

echo "[nvfp4-draft-blocksize] shrinking DFlash drafter (TRITON_ATTN) KV block_size 32 -> 16"

# ----------------------------------------------------------------------------
# 0. Sanity: the get_kv_cache_spec block_size anchor + the layer's backend
#    accessor must be present in the expected shape.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import sys, pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py")
s = p.read_text()
need = [
    "    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:\n",
    "        # Block size may get updated after model loading, refresh it\n"
    "        block_size = vllm_config.cache_config.block_size\n",
    "        self.attn_backend = ",       # layer stores its backend
    "        self.head_size_v = ",        # symmetry check field
]
for n in need:
    if n not in s:
        print(f"[nvfp4-draft-blocksize] FATAL: attention.py missing anchor {n!r}"); sys.exit(3)
print("[nvfp4-draft-blocksize] sanity OK (get_kv_cache_spec block_size anchor + backend/head fields present)")
PY
[ $? -eq 0 ] || { echo "[nvfp4-draft-blocksize] sanity FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# 1. Insert the drafter block_size override right AFTER
#    `block_size = vllm_config.cache_config.block_size`. Marker:
#    NVFP4_DRAFT_BLOCKSIZE.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import ast, pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py")
src = p.read_text()
if "NVFP4_DRAFT_BLOCKSIZE" in src:
    print("[nvfp4-draft-blocksize] drafter block_size override already present"); sys.exit(0)

anchor = (
    "        # Block size may get updated after model loading, refresh it\n"
    "        block_size = vllm_config.cache_config.block_size\n"
)
if src.count(anchor) != 1:
    print(f"[nvfp4-draft-blocksize] FATAL: block_size anchor count = {src.count(anchor)} "
          f"(expected 1)"); sys.exit(3)

INSERT = (
    "        # NVFP4_DRAFT_BLOCKSIZE: the DFlash drafter's bf16 KV page (65,536 B\n"
    "        # /block at bs32) is the MAX in the shared uniform-page pool, so\n"
    "        # unify_kv_cache_spec_page_size pads the MiMo nvfp4 target's real\n"
    "        # 23,040/11,520 pages UP to it (page_size_padded) and the nvfp4 pool\n"
    "        # saving is lost (pool stays bf16-level). Halve the drafter's KV\n"
    "        # block_size (32 -> 16) so its page drops to 32,768 and the target\n"
    "        # is billed at 32,768 not 65,536 (~1.9x larger pool). Drafter-only:\n"
    "        # the target uses TRITON_ATTN_DIFFKV; only the Qwen3 drafter is plain\n"
    "        # TRITON_ATTN. Symmetric guard (head_size == head_size_v) so an\n"
    "        # asymmetric layer is never touched. Never grow / never go below the\n"
    "        # backend minimum (TRITON_ATTN requires block_size % 16 == 0). The\n"
    "        # DFlash speculator reads its own group's block_size at runtime\n"
    "        # (speculator.py:138), so a per-group draft block_size of 16 is safe.\n"
    "        try:\n"
    "            _nvfp4_bkname = self.attn_backend.get_name()\n"
    "        except Exception:\n"
    "            _nvfp4_bkname = None\n"
    "        if (\n"
    "            _nvfp4_bkname == \"TRITON_ATTN\"\n"
    "            and getattr(self, \"head_size_v\", self.head_size) == self.head_size\n"
    "            and isinstance(block_size, int)\n"
    "            and block_size > 16\n"
    "        ):\n"
    "            _nvfp4_new_bs = 16\n"
    "            logger.info_once(\n"
    "                \"NVFP4_DRAFT_BLOCKSIZE: DFlash drafter layer %s KV block_size \"\n"
    "                \"%s -> %s (keeps the drafter's bf16 page below the target's, \"\n"
    "                \"so the nvfp4 target pool is not padded up to the drafter).\",\n"
    "                self.layer_name,\n"
    "                block_size,\n"
    "                _nvfp4_new_bs,\n"
    "            )\n"
    "            block_size = _nvfp4_new_bs\n"
)
src = src.replace(anchor, anchor + INSERT)
ast.parse(src)
p.write_text(src)
print("[nvfp4-draft-blocksize] inserted drafter block_size override (TRITON_ATTN symmetric -> 16)")
PY
[ $? -eq 0 ] || { echo "[nvfp4-draft-blocksize] override patch FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# clean stale bytecode + validate module parses & imports
# ----------------------------------------------------------------------------
find "$SITE/vllm/model_executor/layers/attention" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import ast, sys
f = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py"
src = open(f).read()
ast.parse(src)
assert "NVFP4_DRAFT_BLOCKSIZE" in src, "marker missing after write"
# module imports cleanly
import vllm.model_executor.layers.attention.attention as A  # noqa: F401
print("[nvfp4-draft-blocksize] validation OK (attention.py parses+imports, marker set)")
PY
[ $? -eq 0 ] || { echo "[nvfp4-draft-blocksize] VALIDATION FAILED — aborting"; exit 1; }

# ----------------------------------------------------------------------------
# 2. LOGIC self-test: run the ACTUAL kv_cache_utils grouping/config path with
#    faithfully-built MiMo target (nvfp4) + DFlash drafter (bf16) specs and
#    assert (a) drafter bs32 reproduces the broken 65,536 common page and the
#    bf16-level pool, and (b) drafter bs16 drops the common page to 32,768 and
#    grows the pool ~1.9x, target unaffected. This is the throwaway proof.
# ----------------------------------------------------------------------------
python3 - <<'PY'
import sys, types, torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, SlidingWindowSpec, get_kv_quant_mode, KVQuantMode,
)
import vllm.v1.core.kv_cache_utils as ku

nvfp4 = get_kv_quant_mode("nvfp4"); none = KVQuantMode.NONE
GIB = 1024 ** 3

def fake_cfg():
    model_config = types.SimpleNamespace(max_model_len=32768, original_max_model_len=32768)
    parallel = types.SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    cache = types.SimpleNamespace(num_gpu_blocks_override=None)
    sched = types.SimpleNamespace(disable_hybrid_kv_cache_manager=False,
                                  max_num_batched_tokens=8192, enable_chunked_prefill=True)
    return types.SimpleNamespace(scheduler_config=sched, model_config=model_config,
                                 parallel_config=parallel, cache_config=cache,
                                 kv_transfer_config=None)

def specs(drafter_bs):
    d = {}
    for i in range(9):
        d["language_model.model.layers.f%d.attn" % i] = FullAttentionSpec(
            block_size=32, num_kv_heads=2, head_size=192, head_size_v=128,
            dtype=torch.uint8, kv_quant_mode=nvfp4, indexes_kv_by_block_stride=True)
    for i in range(39):
        d["language_model.model.layers.s%d.attn" % i] = SlidingWindowSpec(
            block_size=32, num_kv_heads=4, head_size=192, head_size_v=128,
            dtype=torch.uint8, kv_quant_mode=nvfp4, sliding_window=128,
            indexes_kv_by_block_stride=True)
    for i in range(5):
        d["draft.layers.%d.attn" % i] = SlidingWindowSpec(
            block_size=drafter_bs, num_kv_heads=4, head_size=128, head_size_v=128,
            dtype=torch.bfloat16, kv_quant_mode=none, sliding_window=1024,
            indexes_kv_by_block_stride=True)
    return d

vc = fake_cfg()
avail = [int(4.64 * GIB), int(2.73 * GIB)]  # live spark1 / spark2 KV memory

res = {}
for bs in (32, 16):
    cfgs = ku.get_kv_cache_configs(vc, [specs(bs), specs(bs)], list(avail))
    nt, mc = ku.get_kv_cache_capacity(vc, cfgs[0])
    pages = sorted({g.kv_cache_spec.page_size_bytes for g in cfgs[0].kv_cache_groups})
    # target groups (the 48 MiMo layers) must all carry ONE common page
    tgt_pages = sorted({
        g.kv_cache_spec.page_size_bytes for g in cfgs[0].kv_cache_groups
        if any("language_model" in ln for ln in g.layer_names)
    })
    res[bs] = (nt, mc, pages, tgt_pages)

nt32, mc32, pg32, tp32 = res[32]
nt16, mc16, pg16, tp16 = res[16]

# (a) before: broken 65,536 common page, bf16-level pool
assert pg32 == [65536], ("expected broken common page 65536, got", pg32)
# (b) after: common page halved to 32,768, pool grows
assert pg16 == [32768], ("expected fixed common page 32768, got", pg16)
assert nt16 > nt32 * 1.7, ("expected >1.7x pool growth", nt32, nt16)
# target groups reflect the same common page (billed page), and the fix moved it
assert tp32 == [65536] and tp16 == [32768], ("target billed page", tp32, tp16)

print("[nvfp4-draft-blocksize] SELF-TEST OK: drafter bs32 -> %s tok (common %s) ; "
      "drafter bs16 -> %s tok (common %s) ; pool x%.2f, target no longer padded to 65536."
      % (format(nt32, ","), pg32, format(nt16, ","), pg16, nt16 / nt32))
PY
[ $? -eq 0 ] || { echo "[nvfp4-draft-blocksize] SELF-TEST FAILED — aborting"; exit 1; }

echo "[nvfp4-draft-blocksize] done"
