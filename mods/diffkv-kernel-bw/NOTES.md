# diffkv-kernel-bw — notes

Fixes parallelism-starvation in the MiMo-V2.5 DiffKV **3D split-KV
(FlashDecoding) decode** attention kernel so the deep-context KV read runs at
**bandwidth** instead of **memory-latency**. Two env-gated edits, default a
byte-for-byte no-op. Runs AFTER `fp8-kv-inline` and `diffkv-3d-qlen8`.

Image `vllm-node-mimo-v25-upstream:latest`, vllm
`0.23.1rc1.dev760+g3775d5fca`. Target: 2× GB10 / sm_121a.

## The diagnosis (fable, code-grounded, verified on the live container)

At 400K ctx, q_len=1, bs=1 the 3D grid launches
`1 q-block × num_kv_heads(2) × NUM_PAR_SOFTMAX_SEGMENTS(16) = 32 CTAs` onto 48
SMs. Each SM holds one CTA → the KV read is bound by memory **latency**, not
**bandwidth** (~21 % of 273 GB/s). Bytes and coalescing are fine; there are too
few in-flight memory transactions. Raising segments 16→64 → `1×2×64 = 128 CTAs`
→ ~2.7 CTAs/SM → latency is hidden by concurrency → the read approaches
bandwidth.

## The two fixes

### K1 — `NUM_PAR_SOFTMAX_SEGMENTS` 16 → 64  (the big win, cudagraph-safe/low-risk)

File `vllm/v1/attention/backends/triton_attn.py`, line 56.

Stock:
```python
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments
```
Patched (env-gated helper; default off = 16, byte-for-byte):
```python
# DIFFKV_KERNEL_BW (K1): ...
import os as _os_diffkv_bw
def _diffkv_bw_segments() -> int:
    if _os_diffkv_bw.environ.get("VLLM_DIFFKV_BW", "0") != "1":
        return 16                                   # gate off -> stock 16
    _raw = _os_diffkv_bw.environ.get("VLLM_DIFFKV_SEGMENTS", "64")
    _n = int(_raw)
    if _n < 1 or _n > 256 or (_n & (_n - 1)) != 0:  # power-of-two in [1,256]
        raise ValueError(...)                       # loud, never silent garbage
    return _n
NUM_PAR_SOFTMAX_SEGMENTS = _diffkv_bw_segments()    # 16 off / 64 on
```

**Why it MUST stay a power of two:** `kernel_reduce_segments_diffkv` indexes
segments with `tl.arange(0, NUM_SEGMENTS_PER_SEQ)` (kernel lines 374/384/397)
and the 3D epilogue strides by `NUM_SEGMENTS_PER_SEQ` (kernel lines
303/304/395/396). A non-2ⁿ count breaks the arange mask / stride math. The
`VLLM_DIFFKV_SEGMENTS` sweep var is rejected (module import raises `ValueError`,
so the engine fails loudly at start) unless it is a power of two in `[1,256]`.

### K2 — 3D-path `TILE_SIZE` 16 → 32 when the KV cache is fp8  (low-med risk)

File `vllm/v1/attention/ops/triton_unified_attention_diffkv.py`, line 487.

Stock:
```python
tile_size = 32 if not use_3d else (16 if q.element_size() >= 2 else 32)
```
Patched:
```python
# DIFFKV_KERNEL_BW (K2): key TILE_SIZE on the KV cache element size, not q's.
import os as _os_diffkv_bw_k2
_diffkv_bw_tile_fp8 = (
    _os_diffkv_bw_k2.environ.get("VLLM_DIFFKV_BW", "0") == "1"
    and k.element_size() == 1          # fp8 KV cache is uint8 (1 byte)
)
if not use_3d:
    tile_size = 32
elif _diffkv_bw_tile_fp8:
    tile_size = 32                     # fp8 KV: half the KV-loop iterations
else:
    tile_size = 16 if q.element_size() >= 2 else 32   # stock query-keyed choice
```

The 3D decode loop is dominated by the **KV** read. Stock keys `TILE_SIZE` on
the **query** dtype (bf16 → element_size 2 → always 16), but the fp8 KV cache is
uint8 (element_size 1). `k` (the key_cache view) is already in scope here
(`num_kv_heads = k.shape[2]`, line 457). Keying on `k.element_size()==1` lets fp8
use `TILE_SIZE=32` → half the iterations over the KV dim.

**cudagraph safety:** `tile_size` is a `tl.constexpr` threaded **identically**
into the main kernel (line ~549) and `kernel_reduce_segments_diffkv` (line 568)
from this single variable, so the two never disagree. The KV dtype is fixed for
the whole deployment → `TILE_SIZE` is a stable per-deployment constant → graph
capture sees fixed shapes. Bf16 KV (drafter / non-fp8) is untouched
(element_size 2 → still 16), and gate-off reproduces the stock line exactly for
every dtype.

## Composition with `diffkv-3d-qlen8`'s buffer resize (the load-bearing part)

**No buffer-alloc edit is needed in this mod, and there is zero double-count /
under-size risk — by construction.** The three per-segment scratch buffers are
allocated *symbolically* from `self.num_par_softmax_segments`, which is set once
to the module constant:

```
triton_attn.py:154   self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
```

Every allocation site references that member — never a literal 16/64:

| site | file:lines | shape (segment dim in **bold**) |
|---|---|---|
| base builder (all three) | `triton_attn.py:156-175` | `[seq_threshold_3D, num_heads_q, `**`nseg`**`, headdim_pad]` etc. |
| DiffKV re-alloc (output, V-shaped) | `triton_attn_diffkv.py:116-125` | `[seq_threshold_3D, num_heads_q, `**`nseg`**`, hv_pad]` |
| diffkv-3d-qlen8 resize (all three) | `triton_attn_diffkv.py:146-165` | `[max(seq_threshold_3D, mns*8), num_heads_q, `**`nseg`**`, hv_pad]` |

`nseg == self.num_par_softmax_segments == NUM_PAR_SOFTMAX_SEGMENTS`. Raising the
**constant** flows into all of them **and** the 3D grid dim2 (`kernel:491`) **and**
the kernel's `NUM_SEGMENTS_PER_SEQ` constexpr (`kernel:549/573`). So K1 is a
one-line constant bump; the segment count is a single source of truth.

**The two mods own orthogonal axes:**
- `diffkv-3d-qlen8` owns the buffer **first-dim** (token axis): `max(seq_threshold_3D, max_num_seqs*8)`.
- `diffkv-kernel-bw` (K1) owns the **segment dim**: 16→64.

Final composed shape (both gates on):
```
softmax_segm_output  : [max(seq_threshold_3D, mns*8), num_heads_q, 64, hv_pad]
softmax_segm_max     : [max(seq_threshold_3D, mns*8), num_heads_q, 64]
softmax_segm_expsum  : [max(seq_threshold_3D, mns*8), num_heads_q, 64]
```
There is no interaction between the two edits: the qlen8 resize re-allocates with
whatever `self.num_par_softmax_segments` is at `__init__` time (already 64 after
K1), so it can neither under-size (it always includes the current `nseg`) nor
double-count (it replaces, it doesn't add to, the base alloc).

**Ordering note:** K1 lives in `triton_attn.py`, which neither `fp8-kv-inline`
nor `diffkv-3d-qlen8` edits; both reference the constant symbolically. K2's
`tile_size` line is likewise untouched by those mods (`fp8-kv-inline` wraps the
K/V *load* + signature; `diffkv-3d-qlen8` touches the `use_3d` gate + the
backend buffers). Both anchors were confirmed present **exactly once** in the
LIVE post-fp8/post-3dq8 files (`docker cp vllm_node …`; markers
`FP8_KV_INLINE_*` and `DIFFKV_3D_QLEN8` present in the extracted kernel/backend).

## Memory cost / rank (PROD geometry: num_heads_q=32, hv_pad=128, fp32)

Buffers = `output[first,32,nseg,128]` + `max[first,32,nseg]` + `expsum[first,32,nseg]`.

| first-dim | 16 seg | 64 seg | **Δ (16→64)** |
|---|---|---|---|
| **64**  (`seq_threshold_3D`; PROD mns=8 → qlen8 resize is a no-op since `8*8==64`) | 17.0 MB | 68.2 MB | **+51 MB** |
| 256 (mns=32, qlen8 on) | 68.2 MB | 272.6 MB | +204 MB |

At the **nvfp4 PROD recipe (mns=8)** the K1 bump costs **≈ +51 MB/rank**
(single-buffer, allocated once at init). fable's *~+150 MB/rank* estimate
corresponds to a larger effective first-dim (~188 — i.e. the qlen8 path active at
higher `max_num_seqs`, or a larger `seq_threshold_3D` at a bigger capture size).
Either way it is tens-to-low-hundreds of MB, once, at init — negligible vs the
KV pool and comfortably inside the util-0.86 headroom. **128-seg** doubles the
64-seg cost (`+136 MB` at first-dim 64) — only relevant if a sweep picks it.

## Risks

- **sm_121 Triton correctness (the real one).** This image has a 3× history of
  Triton silently emitting garbage on sm_121. K1 changes the launch geometry and
  the reduce fan-in width; K2 changes the tile iteration count. Both are numeric
  no-ops *in principle*, but MUST be proven by the parity harness + shadow A/B
  **before** trust. Do not ship enabled without step (a) below passing.
- **K2 tile indexing.** `TILE_SIZE=32` doubles the per-iteration KV tile. The
  kernel already parameterizes tile math on `TILE_SIZE`, and stock already uses
  32 for the whole 2D/prefill path, so 32 is a proven value — but the 3D
  epilogue reduction at 32 on fp8 is the specific combination to check in parity.
- **Occupancy is not free above a point.** 64 seg = 128 CTAs ≈ 2.7 CTAs/SM is the
  sweet spot; 128 seg = 256 CTAs may over-subscribe and add reduce overhead. The
  `VLLM_DIFFKV_SEGMENTS` sweep exists exactly to measure 32/64/128 — don't assume
  128 > 64.
- **Reversibility.** Fully reversible: `docker cp` the pristine files back, or
  rebuild the image without this mod. Default-off is byte-for-byte the current
  kernel, so it is safe to add to the recipe **inert** and flip on later.

## Test protocol (cluster, when idle — NOT beside PROD at util 0.86)

1. **Offline 2D-vs-3D parity** — `/home/admin/diffkv_3d_parity.py`, numeric gate,
   low-footprint boot. The harness hard-codes `NUM_PAR_SOFTMAX_SEGMENTS = 16` at
   module top; to exercise K1 either (i) edit that constant to 64 in a copy, or
   (ii) run it inside the container with `VLLM_DIFFKV_BW=1` and have it read
   `triton_attn.NUM_PAR_SOFTMAX_SEGMENTS` instead of the local literal. **Also
   set the harness's 3D `tile_size` to mirror K2** (its `_launch` currently uses
   `16 if q.element_size() >= 2 else 32`; for fp8 KV @ VLLM_DIFFKV_BW=1 it should
   use 32) so parity tests the *shipped* tile choice. PASS iff max abs & rel err
   ≤ 2e-3 across all (seq_len, q_len). Run `--max-seq 200000` first (400K
   allocates a large pool).
2. **Deep-decode A/B** at 200K / 400K, no-spec, on a scratch port:
   `VLLM_DIFFKV_BW=1` vs the shipped baseline. Measure accept & tok/s.
   Expected: attn read 40 ms → **~12-16 ms (K1)** → **~10-14 ms (K1+K2)**;
   no-spec @400K **13.2 → ~19-22 tok/s**.
3. **needle@240K + short reliability** — confirm no quality regression vs the
   fp8/nvfp4 baseline.

## Enable

Add to a recipe (inert until the env is set):
```bash
VLLM_DIFFKV_BW=1            # K1 (segments 64) + K2 (fp8 TILE 32)
# VLLM_DIFFKV_SEGMENTS=64   # optional sweep: 32 / 64 / 128 (power of two)
```
Default unset/0 = the current kernel, unchanged.
