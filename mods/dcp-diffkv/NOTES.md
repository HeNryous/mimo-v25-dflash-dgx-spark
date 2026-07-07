# dcp-diffkv — engineering notes (stage 2 drafted, disk-only)

> 2026-07-07: `_forward_with_dcp` + the builder context-phase scratch are now
> fully DRAFTED on disk and statically validated (`ast.parse` + `py_compile` of
> the mod applied to fresh `docker cp`'d live sources; all anchors `count==1`).
> The live `vllm_node` was NOT booted/restarted/exec'd and PROD was untouched.
> Execution is double-gated (`VLLM_DCP` dispatch + `VLLM_DCP_STAGE2_OK` ack) and
> HARD-ASSERTS until the LIVE-BOOT CHECKLIST items are pinned — see below.

Decode Context Parallelism (DCP) for the MiMo-V2.5 Triton DiffKV / fp8 decode
path. Ports the sequence-split KV attention + cross-rank LSE-weighted combine
that vLLM already ships (backend-generic scaffolding) onto **our** production
Triton path, whose only gap was that it did not expose the softmax LSE at decode.

- Feasibility + file:line evidence: `/home/admin/dcp_feasibility.md` (verdict GO;
  ~1.5–1.8× deep @400K, Amdahl-capped to the 9 full-attn layers).
- Offline correctness gate: `/home/admin/dcp_parity.py` (run inside the container,
  needs a GPU; do **not** run against the live server).
- Image: `vllm-node-mimo-v25-upstream` (vllm 0.23.1rc1.dev760+g3775d5fca, sm_121a,
  2× GB10, TP=2).
- Ordering: run AFTER `fix-mimo-v2-upstream`, `fp8-kv-inline`, `diffkv-3d-qlen8`.

Everything is env-gated `VLLM_DCP` (default 0). At default the two target files
are byte-for-byte stock **except** one inert class attribute
(`can_return_lse_for_decode = True`) that has no effect unless
`dcp_world_size > 1` (i.e. the recipe passes `--decode-context-parallel-size >1`).

---

## Why this is possible at all — the LSE was already computed

DCP's gate (`vllm/v1/worker/cp_utils.py:30`) hard-asserts, for `dcp_size > 1`,
that every attention impl sets `need_to_return_lse_for_decode`. That flag is
derived (`vllm/v1/attention/backend.py:822`) as:

```
need_to_return_lse_for_decode = (dcp_world_size > 1 AND can_return_lse_for_decode)
```

Base `can_return_lse_for_decode = False` (`backend.py:757`); dense FlashAttention
overrides it to `True` (`flash_attn.py:675`). Our `TritonAttentionDiffKVImpl`
did **not** → a naive `--decode-context-parallel-size 2` aborts with
*"…requires attention implementations to return the softmax LSE during decode…"*.

But our 3D split-KV (FlashDecoding) reduce **already materializes the LSE** and
throws it away:

`vllm/v1/attention/ops/triton_unified_attention_diffkv.py :: kernel_reduce_segments_diffkv`
```
L387  overall_max    = tl.max(segm_max)
L391  overall_expsum = tl.sum(segm_expsum)      # after rescaling each segm to overall_max
L407  acc            = acc_sum / overall_expsum  # normalize
L414  tl.store(output_ptr, acc)                  # <-- discards max + expsum
```

`LSE = log(overall_expsum) + overall_max` — **natural log (base e)**, which
matches three independent things and is why no base conversion is needed:
- `backend.py:768` default `lse_base_on_e = True`,
- the combine kernel's `IS_BASE_E` arm (`dcp_alltoall.py:273` `tl.log(lse_sum) + lse_max`),
- the CPU golden (`dcp_alltoall.py:94`).

The intra-rank segment reduction is *the same operation* as DCP's cross-rank
`_lse_weighted_combine`. So exposing the LSE is **one extra store of an
already-correct intermediate** — additive, not a rewrite. That is the whole
reason this is days-not-weeks.

---

## Exact diffs

### A. Kernel — expose LSE (`triton_unified_attention_diffkv.py`), 3 edits

**A1. Reduce-kernel signature** — append 4 params (additive; the stock call
supplies a dummy pointer + `RETURN_LSE=False`):
```python
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
+   out_lse_ptr,
+   out_lse_stride_0: tl.int64,
+   out_lse_stride_1: tl.int64,
+   RETURN_LSE: tl.constexpr,
):
```

**A2. Store the LSE** right after the `acc` store (`L414`):
```python
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)
+
+   if RETURN_LSE:
+       lse_val = tl.where(
+           overall_expsum == 0.0,
+           float("-inf"),                              # empty shard -> -inf -> zero weight
+           tl.log(overall_expsum) + overall_max,       # base e; matches golden & combine
+       )
+       out_lse_offset = (
+           query_token_idx.to(tl.int64) * out_lse_stride_0
+           + query_head_idx * out_lse_stride_1
+       )
+       tl.store(out_lse_ptr + out_lse_offset, lse_val)
```
The `overall_expsum == 0.0` guard (an all-masked row, e.g. a rank whose shard is
empty) emits `-inf`, which the combine's NaN/inf guards
(`dcp_alltoall.py:236/262/294`) map to a zero weight — the correct semantics for
"this shard contributed nothing".

**A3. Wrapper `unified_attention_diffkv`** — add `out_lse=None` param and thread
it into the reduce launch. Only the 3D path can return LSE (the 2D path has no
per-segment reduce). When `out_lse is None`, `RETURN_LSE=False` and a dummy
non-null pointer (`out`, stride 0) is passed but never written ⇒ byte-for-byte
stock.
```python
    v_descale: float = 1.0,
+   out_lse: torch.Tensor | None = None,
):
  ...
        kernel_reduce_segments_diffkv[(q.shape[0], num_query_heads)](
            ...
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
+           out_lse_ptr=(out_lse if out_lse is not None else out),
+           out_lse_stride_0=(out_lse.stride(0) if out_lse is not None else 0),
+           out_lse_stride_1=(out_lse.stride(1) if out_lse is not None else 0),
+           RETURN_LSE=(out_lse is not None),
        )
```
LSE shape written: `[num_tokens, num_query_heads]` = `[B, H]` (the reduce grid is
`(q.shape[0], num_query_heads)`).

### B. Backend impl (`triton_attn_diffkv.py`)

**B1.** Class attrs on `TritonAttentionDiffKVImpl`:
```python
    can_return_lse_for_decode: bool = True   # inert unless dcp_world_size>1 (backend.py:822)
    lse_base_on_e: bool = True               # ln(expsum)+max -> base e
```
`dcp_world_size`/`dcp_rank`/`need_to_return_lse_for_decode` are **auto-populated
by `AttentionImplBase.__new__`** (`backend.py:799-825`) for every subclass, so —
unlike the report's initial estimate — we do **not** hand-copy the dcp init from
`flash_attn.py:363-373`. Setting the class attr is the entire "enable" step.

**B2.** `__init__` — pick the combine + payload dtype when DCP is on (identical
selection to `flash_attn.py:743`; both combine fns share the signature
`(out[B,H,D], lse[B,H], group, ctx, return_lse, is_lse_base_on_e) -> [B,H/N,D]`):
```python
        self._dcp_dtype = None
        self.dcp_combine = None
        if getattr(self, "dcp_world_size", 1) > 1:
            _cfg = _dcp_get_cfg()
            _a2a = _cfg is not None and getattr(_cfg.parallel_config, "dcp_comm_backend", "ag_rs") == "a2a"
            self.dcp_combine = _dcp_a2a_lse_reduce if _a2a else _cp_lse_ag_out_rs
            self._dcp_dtype = _cfg.model_config.dtype if _cfg is not None else None
```

**B3.** `forward` — DCP dispatch, inserted right after the head-size locals.
Requires: the env, a real DCP group, the DCP metadata being present, AND the step
being pure-decode/verify (`max_query_len <= _DIFFKV_3D_Q8_MAXQ`, so the 3D+LSE
path is reachable). Prefill / DCP-off fall through to the untouched stock call.
```python
        if (_VLLM_DCP and getattr(self, "dcp_world_size", 1) > 1
                and self.dcp_combine is not None
                and getattr(attn_metadata, "dcp_context_kv_lens", None) is not None
                and attn_metadata.max_query_len <= _DIFFKV_3D_Q8_MAXQ):
            return self._forward_with_dcp(layer, query, key, value, kv_cache,
                attn_metadata, output, head_size_qk, head_size_v, num_actual_tokens)
```

**B4.** `_forward_with_dcp` + `_dcp_slice_kv` methods (near-copy of
`flash_attn.py:1037`). **STAGE-2 NOW DRAFTED** (2026-07-07): the full control
flow is real, reviewable code — (1) all-gather Q → context-shard 3D+LSE attn →
`dcp_combine`; (2) local query-block self-attn; (3) `merge_attn_states` with the
`[B,H]→[H,B]` LSE transposes. It is **guarded by `VLLM_DCP_STAGE2_OK=1`** (a
second gate beyond the forward-dispatch `VLLM_DCP`) because two `seqused_k`/
`block_table` constructions are un-determinable statically — see the
**LIVE-BOOT CHECKLIST** below. Absent the ack it HARD-ASSERTS (no silent
mis-run). `_dcp_slice_kv` returns `(key_cache, value_cache, fp8_kwargs)` and the
caller uses `attn_metadata.block_table`; fp8-under-DCP is asserted INLINE.

### C. Metadata

**C1.** `TritonAttentionMetadata` (`triton_attn.py`) — two optional fields
(additive, default `None`):
```python
    dcp_context_kv_lens: torch.Tensor | None = None
    max_dcp_context_kv_len: int | None = None
```

**C2 (+ piece i).** `TritonAttentionDiffKVMetadataBuilder` gets **both** an
`__init__` addition and a `build()` override.

`__init__` — **critical fix**: unlike the *FlashAttention* builder
(`flash_attn.py:362-374`), the Triton parent `TritonAttentionMetadataBuilder`
does **NOT** set `dcp_world_size`/`dcp_rank`/`cp_kv_cache_interleave_size` on
itself (verified: the base `AttentionMetadataBuilder` doesn't either). So the
stage-1 `build()` split used `getattr(self,"dcp_world_size",1)` → always `1` →
**would have silently no-op'd DCP**. `__init__` now acquires them from the DCP
group / `parallel_config`, and allocates the **CONTEXT-phase scratch ONCE**
(piece i, cudagraph-safe, private, replay-stable), head-widened to
`num_heads_q*dcp_world_size` — mirroring the local-segm allocation pattern right
above it, sized `max(seq_threshold_3D, max_num_seqs*8)`:
```python
    # (inside __init__, after the diffkv-3d-qlen8 local-segm block)
    self.dcp_world_size = ...  # from _dcp_group_or_none() when VLLM_DCP=1
    if VLLM_DCP and dcp_ws > 1:
        _dcp_first = max(self.seq_threshold_3D, max_num_seqs * 8)
        _h_all = self.num_heads_q * self.dcp_world_size
        self.dcp_context_softmax_segm_output = torch.empty((_dcp_first,_h_all,_nseg,hsv_pad), f32)
        self.dcp_context_softmax_segm_max    = torch.empty((_dcp_first,_h_all,_nseg), f32)
        self.dcp_context_softmax_segm_expsum = torch.empty((_dcp_first,_h_all,_nseg), f32)
        self.dcp_context_out = torch.empty((_dcp_first,_h_all,hsv), model_dtype)  # WIDE
        self.dcp_context_lse = torch.empty((_dcp_first,_h_all), f32)
        self.dcp_query_out   = torch.empty((_dcp_first,num_heads_q,hsv), model_dtype)  # LOCAL
        self.dcp_query_lse   = torch.empty((_dcp_first,num_heads_q), f32)
```

`build()` — override that calls `super().build()` then, when DCP is on, computes
the local KV split (mirrors `flash_attn.py:522-542`) AND stashes the pre-allocated
scratch on the metadata (same tensors every build → replay-stable):
```python
    def build(self, *args, **kwargs):
        md = super().build(*args, **kwargs)
        _dcp_ws = getattr(self, "dcp_world_size", 1)
        if os.environ.get("VLLM_DCP","0")=="1" and _dcp_ws > 1:
            query_lens     = md.query_start_loc[1:] - md.query_start_loc[:-1]
            context_kv_lens = md.seq_lens - query_lens
            interleave      = getattr(self, "cp_kv_cache_interleave_size", 1)
            md.dcp_context_kv_lens   = get_dcp_local_seq_lens(context_kv_lens, _dcp_ws, self.dcp_rank, interleave)
            num_parts       = _dcp_ws * interleave
            md.max_dcp_context_kv_len = ((md.max_seq_len + num_parts - 1) // num_parts) * interleave
            md.dcp_context_softmax_segm_output = self.dcp_context_softmax_segm_output  # + the other 6
        return md
```
The seven scratch buffers are declared as `Optional` fields on
`TritonAttentionMetadata` (C1 extended) so a non-DCP build leaves them `None`.

---

## How LSE is threaded end-to-end (the data path)

1. **Reduce kernel** writes per-token LSE `[B, H]` fp32 into `out_lse` (base e).
2. **Context phase** (`_forward_with_dcp`) passes `out_lse=ctx_lse` into
   `unified_attention_diffkv` for the Q-all-gathered context-shard attention.
3. **Combine** (`self.dcp_combine(ctx_out, ctx_lse, group, return_lse=True,
   is_lse_base_on_e=self.lse_base_on_e)`) does the exact cross-rank LSE-weighted
   merge and returns head-scattered `[B, num_heads, D]` + `[B, num_heads]` LSE.
4. **Query phase** attends the local q_len tail (not split) → `q_out`, `q_lse`.
5. **`merge_attn_states(output, ctx_out, ctx_lse[H,B], q_out, q_lse[H,B])`** fuses
   context+query per token. **Shape note:** `merge_attn_states` wants LSE in
   `[H, B]` (`merge_attn_states.py:33/37`), while the combine returns `[B, H]` —
   so a `.transpose(0,1).contiguous()` sits between them (exactly as
   `flash_attn.py:1100`).

---

## Composition with the rest of the stack (all verified against installed code)

- **× fp8-KV (`VLLM_FP8_INLINE`)** — the fp8 descale happens **inside** each
  rank's local shard attention exactly as today (same `key_cache/value_cache`
  slicing + `fp8_kwargs`, reproduced by `_dcp_slice_kv`). DCP combines the
  **post-attention bf16 outputs + fp32 LSE**. **fp8 never crosses the wire**: the
  A2A payload dtype is `_dcp_dtype` (= model dtype, bf16) with fp32 LSE bit-packed
  into 2 bf16 slots (`dcp_alltoall.py::_dcp_a2a_lse_pack_dim`). No numeric
  interaction with the KV quant. *Caveat:* the fp8 **SCRATCH** fallback
  (`VLLM_FP8_INLINE=0`) remaps `block_table` per-call, which the DCP seq-len split
  keys off — so **under DCP, use `VLLM_FP8_INLINE=1`** (in-kernel descale, the
  prod setting anyway). `_dcp_slice_kv` documents this.
- **× DFlash q_len=8 verify** — DCP needs the 3D+LSE path, which is exactly what
  `diffkv-3d-qlen8` unlocks for `q_len ≤ 8`. So DCP dispatches only when
  `VLLM_DIFFKV_3D_Q8=1` (q_len 2..8) or q_len==1 (already 3D). The 8-token query
  self-attn is **local** (new tokens, not split); only cached context KV is
  sequence-split. `merge_attn_states` fuses the two.
- **× `diffkv-3d-qlen8` 3D path** — this mod adds an *output* to the **same**
  reduce kernel that 3d-qlen8 makes reachable; the local segm buffers 3d-qlen8
  resized are reused by the query phase. The **context** phase (Q-all-gathered →
  `num_heads*dcp_world_size` heads) needs its own, wider segm buffers → the stub.
- **× cudagraph** — the known hazard, already solved for the A2A itself:
  `dcp_alltoall.py:116-129` uses **private `torch.empty` send/recv buffers** (not
  the growable workspace) so replay addresses stay valid; `dcp_a2a_lse_reduce`
  uses that internally. The **remaining** cudagraph item is capturing this mod's
  own context-phase scratch (the Q-gathered out + its segm buffers) against
  private, replay-stable buffers — part of the stub. **First bringup MUST be
  enforce-eager**, as with every prior DiffKV kernel change.

---

## Recipe knob

Add to the recipe (all three needed to actually run DCP):
```yaml
# engine args
decode_context_parallel_size: 2          # --decode-context-parallel-size 2
dcp_comm_backend: a2a                     # --dcp-comm-backend a2a (else ag_rs; a2a = 1 collective/layer)
# cp_kv_cache_interleave_size: 1          # default 1 (per-token split); keep 1 (see below)
```
```bash
# env (container)
VLLM_DCP=1
VLLM_DIFFKV_3D_Q8=1     # so DFlash q_len=8 verify reaches the 3D+LSE path
VLLM_FP8_INLINE=1       # in-kernel fp8 descale (block_table stays the metadata's own)
VLLM_DCP_STAGE2_OK=1    # ONLY after LIVE-BOOT items 1-3 are pinned (parity + A/B).
                        # Without it the drafted _forward_with_dcp HARD-ASSERTS.
```
- `--decode-context-parallel-size 2` is valid at TP=2 (`parallel.py:503` requires
  `tensor_parallel_size % dcp_size == 0`).
- Keep `cp_kv_cache_interleave_size = 1`: at >1 the DCP gate ALSO requires
  `supports_mtp_with_cp_non_trivial_interleave_size` (`cp_utils.py:25-29`), which
  our impl does not set — and we run DFlash (MTP). Interleave=1 sidesteps that
  entirely (and per-token split is the finest granularity).
- `dcp_comm_backend: a2a` sends one packed `all_to_all_single` per full-attn layer
  (vs AllGather+ReduceScatter = 2 collectives). Both are cudagraph-safe here.

---

## Memory cost

DCP is KV-memory **positive**: each rank now stores/reads only ~half the context
KV positions for the full-attn layers, so peak KV pressure drops (the PR reports
~39.77% → ~20.45% at 128K). It does **not** threaten the util ≤0.86 ceiling — if
anything it buys headroom (or lets `mml` grow). The added per-init cost is small:
- The context-phase segm buffers (now allocated in the builder `__init__`): fp32,
  sized `_dcp_first × (num_heads_q*dcp_world_size) × num_par_softmax_segments ×
  head_size_v_padded` for `segm_output` (+ the two rank buffers without the last
  dim). At MiMo-class (num_heads_q per rank — **confirm live**, LIVE-BOOT item 4),
  dcp=2, 16 segments, hsv_padded 128, `_dcp_first`~O(hundreds): order tens of MB,
  once at build. Same class as the ~17–34 MB 3d-qlen8 already allocates. Plus the
  small `dcp_context_out/lse` + `dcp_query_out/lse` (out = model-dtype, lse =
  fp32), also `_dcp_first`-rowed.
- The A2A send/recv buffers: `~[2, B, H/2, D+2]` bf16 per layer, private, tiny
  (~0.5 MB/layer at a single 400K stream). Latency-bound, not a memory concern.

---

## Stage-2 DRAFTED vs LIVE-BOOT remaining — HONEST breakdown

### DONE on disk — verified statically (`ast.parse` + `py_compile` of the mod
applied to fresh `docker cp`'d live sources; anchors all `count==1`; NO boot,
NO parity run, NO container write — the live `vllm_node` was never touched):

- [x] **LSE exposure** from `kernel_reduce_segments_diffkv` — signature, the
  guarded base-e store (`ln(overall_expsum)+overall_max`, `-inf` on empty shard),
  wrapper `out_lse` threading. Default-off proven: `RETURN_LSE=(out_lse is not
  None)`, dummy pointer never written.
- [x] **Gate satisfied** — `can_return_lse_for_decode=True` + `lse_base_on_e=True`
  on the impl; inert at dcp=1 (derives off `dcp_world_size` in `backend.py:822`).
- [x] **Combine selection** — `self.dcp_combine` (a2a vs ag_rs) + `_dcp_dtype`,
  matching `flash_attn.py:738-747`.
- [x] **Metadata split + scratch** — 9 optional dataclass fields (C1) + builder
  `build()` computing `get_dcp_local_seq_lens` + `max_dcp_context_kv_len`
  (`flash_attn.py:522-542`) and stashing the scratch.
- [x] **Dispatch** — the `forward` branch into `_forward_with_dcp`, guarded (env
  + group + metadata present + `max_query_len ≤ _DIFFKV_3D_Q8_MAXQ`).
- [x] **Builder dcp-handle fix (piece i, part A)** — `__init__` now acquires
  `dcp_world_size`/`dcp_rank`/`cp_kv_cache_interleave_size` (the Triton parent
  never set them → the stage-1 split would have silently no-op'd).
- [x] **Context-phase scratch alloc (piece i, part B)** — 7 private,
  replay-stable `torch.empty` buffers allocated **once** in the builder, the
  segm/out/lse ones head-widened to `num_heads_q*dcp_world_size`; sliced `[:n]`
  in the impl (never per-call alloc).
- [x] **`_forward_with_dcp` FULL CONTROL FLOW (pieces 1–4)** — real code:
  (1) `all_gather(q,dim=1)` → context 3D+LSE `unified_attention_diffkv(out_lse=
  ctx_lse)` → `dcp_combine(...,return_lse=True,is_lse_base_on_e=self.lse_base_on_e)`;
  (2) local query-block `unified_attention_diffkv(out_lse=q_lse)` on the LOCAL
  segm buffers; (3) `merge_attn_states(output[:n], ctx_out_c, ctx_lse_c.T, q_out,
  q_lse.T)` with `[B,H]→[H,B]` `.transpose(0,1).contiguous()` (our reduce emits
  `[B,H]`, so — unlike flash which transposes FA's `[H,B]`→`[B,H]` into the
  combine — we pass `[B,H]` straight into `dcp_combine` and only transpose going
  into `merge`).
- [x] **Safety property preserved** — `_forward_with_dcp` has **zero
  `NotImplementedError`** (verified via AST scoping) but its executed body
  HARD-ASSERTS `VLLM_DCP_STAGE2_OK=1`, plus a loud `logger.info_once` TEST-path
  banner, plus fp8-INLINE + live-DCP-group + segm-present + shape-match asserts.
  Strictly dead code in prod (outer `VLLM_DCP` never set by the recipe).
- [x] **Offline parity harness** `/home/admin/dcp_parity.py` present (drives the
  kernel + combine numerics directly; does NOT exercise `_forward_with_dcp`).

### LIVE-BOOT CHECKLIST — what CANNOT be resolved statically (needs one boot)

> The drafted `_forward_with_dcp` will not execute until `VLLM_DCP_STAGE2_OK=1`
> is set, precisely because the items below need a live engine + the parity
> harness. Each names the file:line where the live value/decision comes from.

1. **[BLOCKING] Offline parity gate** — inside the container on a GPU (NOT the
   live server): `VLLM_DCP=1 VLLM_FP8_INLINE=1 python3 /home/admin/dcp_parity.py`.
   Validates the reduce's new LSE store + the cross-rank combine to ≤2e-3 on
   sm_121 (the miscompile gate). Needs `docker exec` (blocked from this Pi task).
   Must PASS before anything else. Source of truth: `dcp_parity.py` asserts.

2. **[BLOCKING, piece ii] CONTEXT-phase `seqused_k`/`cu_seqlens` construction.**
   The DiffKV kernel is **causal-only** and derives `context_len = seqused_k −
   query_len` (`triton_unified_attention_diffkv.py` ~L188 `context_len = seq_len
   - cur_batch_query_len`; mask at helper `compute_kv_seq_mask` ~L314-315
   `seq_offset <= query_abs_pos`). The flash ref instead runs the context with
   `causal=False` (`flash_attn.py:1080`). We currently pass
   `seqused_k=attn_metadata.dcp_context_kv_lens` (impl ~context call) — the
   **starting** construction. Must pin, via `dcp_parity.py` extended to the
   forward decomposition, the exact `cu_seqlens_q`/`query_len` pairing (or a small
   kernel `USE_CAUSAL` exposure) that makes every new-token query see the WHOLE
   shard `[0, context_len)` without clipping its last `query_len` positions.

3. **[BLOCKING, piece ii] QUERY-BLOCK `seqused_k`/`block_table` construction.**
   Must attend ONLY the new tokens `[context_len, seq_len)` self-causally, WITHOUT
   re-reading context (else double-count under `merge_attn_states`). The fused
   DiffKV kernel reads K/V from the paged cache via `block_table` (takes no raw
   `key`/`value`, unlike `flash_attn.py:1105-1108`). The impl currently passes
   `seqused_k=attn_metadata.seq_lens` as a **clearly-marked PLACEHOLDER** (the
   comment says so) — this is WRONG-until-pinned and inert behind
   `VLLM_DCP_STAGE2_OK`. Pin the query-only construction (restricted `block_table`
   view + `seqused_k=query_len`) with the harness. Live values:
   `attn_metadata.query_start_loc`, `attn_metadata.seq_lens`,
   `attn_metadata.block_table` (all built in `triton_attn.py::…Builder.build`).

4. **[piece i] Real `_dcp_first` / head counts for the context segm sizing.**
   `__init__` sizes the context scratch `first = max(seq_threshold_3D,
   max_num_seqs*8)`, head `= num_heads_q * dcp_world_size`. Confirm at boot: the
   live `num_heads_q` per rank (post-TP-split; `triton_attn.py::…Builder.__init__`
   `self.num_heads_q`, ~L117), `seq_threshold_3D` (~L139/149, cudagraph-capture-
   snapped), `max_num_seqs` (`scheduler_config`), and that a decode+verify batch's
   token count never exceeds `_dcp_first` (else OOB — same failure mode
   diffkv-3d-qlen8 guards). Memory: order tens of MB (see "Memory cost").

5. **[piece ii] LSE transpose contiguity vs live layouts.** `ctx_lse_c` comes
   from `dcp_combine` (`dcp_alltoall.py:392` returns `[B,H/N]`), `q_lse` from our
   reduce (`[B,H]`); both `.transpose(0,1).contiguous()` → `[H,B]` for
   `merge_attn_states` (`merge_attn_states.py:33/37` want `[NUM_HEADS,NUM_TOKENS]`).
   Verify the `.contiguous()` + the `head_size_v=128 % 8 == 0` CUDA-kernel
   headdim constraint (`merge_attn_states.py:66-70`) hold on the live bf16
   tensors (else it silently falls to the Triton merge — correct but slower).

6. **cudagraph re-capture** of the context-phase scratch + the A2A. The buffers
   are builder-owned/replay-stable by construction, but **first bringup MUST be
   enforce-eager**; then flip FULL cudagraph and re-verify no illegal-memory /
   replay corruption and that the speedup holds. (`dcp_alltoall.py:116-129`
   already gives the A2A private-buffer discipline.)

7. **NIC micro-measurement** (`dcp_feasibility.md §4`): standalone 2-rank
   `all_to_all_single` of `[2,8,16,130]` bf16 over RoCE, p50/p99 µs; green if
   p50 ≲ 100 µs. Expected-GO (NCCL already cross-node) but measure before
   trusting the ~1.5–1.8× claim.

8. **Confirm the target model/recipe.** `dcp_feasibility.md` flags the live
   `vllm_node` shows `max-model-len 500000`/TP=2, looks DeepSeek-class, not the
   MiMo image the older notes assume. **DeepSeek-V4 is MLA** (its own DCP path
   `flashmla_sparse.py`), NOT this DiffKV/fp8 path — so even though the DiffKV
   kernels are present in the image, confirm which model actually routes through
   `TritonAttentionDiffKVImpl` before scheduling the boot. This mod is a no-op
   for any model that doesn't use the DiffKV backend.

---

## Recommended test sequence (the eventual single boot)

Run strictly in order; do **not** proceed past a failing gate. Never touch
`:8000` / `:8001` (live daily-driver) — use a scratch port.

1. **Offline parity** — inside the container, on a GPU:
   ```bash
   VLLM_DCP=1 VLLM_FP8_INLINE=1 python3 /home/admin/dcp_parity.py
   ```
   Asserts DiffKV decode `DCP=1` (no split) vs `DCP=2` (2-way seq split +
   LSE-combine) vs the CPU golden `_lse_weighted_combine`, max abs/rel ≤ 2e-3, on
   fp8 KV, head_dim 192/128, seq_lens incl. 200K/400K. **This is the load-bearing
   sm_121 miscompile gate — must PASS before anything else.**
2. **Finish the `_forward_with_dcp` stub** using the live capture sizes; re-run
   the parity harness (now exercising the real forward, not just the combine).
3. **Shadow A/B, enforce-eager**, scratch port: `DCP=1` vs `DCP=2` at 200K and
   400K single-stream deep decode. Compare tok/s (expect ~1.5–1.8× on the
   full-attn-dominated deep decode) and assert output parity.
4. **Flip cudagraph** on the scratch port; re-run 3; verify no illegal-memory /
   replay corruption (the private-buffer discipline) and that the speedup holds.
5. **Quality** — needle-in-haystack @240K + a coherence / reliability pass at
   DCP=2, compared to the DCP=1 baseline. Must match (DCP is exact, not lossy).
6. Only then consider promoting to a recipe.
