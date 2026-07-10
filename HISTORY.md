# Engineering history & measurement data

Everything that explains *why* the recipe in [README.md](README.md) looks the way it does:
kernel work, A/B measurements, and the dead ends we measured so you don't have to.
Newest first.

## Final quality gate (2026-07-10, on the exact shipped config)

reliability-bench v1_full **57/79 first-pass** (anchor 56/79 — existence 23/24, refusal 9/9; the
known xref/quant counting weakness unchanged), tool-eval 87.3 ± 2 (11-run band), loop battery 0/12,
needle@126K PASS. Pool 1,667,459 tokens / 3.33× @ mml 500K with all services co-resident.

## 2026-07-10 performance update (all shipped in recipes/mimo-fp8kv-prod.yaml)

Three additive, quality-neutral improvements, each independently A/B-measured on 2× GB10:

1. **Prefill kernel tune — `mods/diffkv-prefill-tune`** (BLOCK_M=32, num_warps=4, num_stages=2,
   tile=32, prefill-only gate `max_seqlen_q > 8`): the stock DiffKV launch under-tiles prefill
   (BLOCK_Q=1 at nqpk=16). The tuned launch is **bit-identical** to stock (parity-verified on both
   nodes incl. SWA/mixed-batch/fp8-arm: rel-err 0.0, zero big-error elements) and cuts cold TTFT by
   **12.6% @50K / 23.3% @200K / 27.8% @400K** (measured: 27.2→23.8s, 234→179s, 804→580s).
   Default-ON when the mod is applied; set `VLLM_DIFFKV_PREFILL_TUNE=off` to force stock.
   Decode/spec/cudagraph paths untouched. Surprising detail: the speedup comes from **num_warps=4**
   — the upstream-precedented warps=8 arms measured 0.9× (slower than stock).
2. **Prefill/decode fairness — `--long-prefill-token-threshold 4096`**: without it, a long prefill
   chunk fills the whole `max_num_batched_tokens` budget every step and concurrent decode streams
   freeze for the entire prefill (measured: a ~150K-token prefill stalled a parallel stream for
   **~120s**). With the cap, decodes co-schedule every step. vLLM's auto-default for this flag
   (4% of max_model_len) is larger than mnbt and therefore inert — set it explicitly.
   Chunk-size A/B (same engine, 150K prefill + parallel decode stream): threshold **4096** beats
   2048 on every axis that matters — prefill under contention 124.3s → **89.4s (−28%)**, decode
   inter-token median during the prefill 1654ms → **148ms (11×)**, worst-case gap 3.2s → 4.2s (the
   one small regression), cold TTFT@200K ~−5%. Larger chunks mean fewer long scheduler steps;
   decodes run nearly freely between them. The MoE side explains it: the fused nvfp4 grouped-GEMM
   runs at 96/83/55% of the memory roofline at chunk 512/2048/8192 — bigger chunks amortize the
   fixed ~8ms weight sweep per layer. (`--max-num-partial-prefills` is hard-blocked in this vLLM —
   "Concurrent Partial Prefill is not supported"; only the threshold works.)
3. **Deep-decode bandwidth — `mods/diffkv-kernel-bw`** (`VLLM_DIFFKV_SEGMENTS` 16→64 in the 3D
   split-KV decode): +18% @200K / +5% @400K no-spec decode, needle-verified.

Operational hardening (see `systemd/`): the pre-start pauses neo4j during vLLM's startup memory
profiling (the 0.86-util guard is razor-thin on GB10; any ~1–2 GiB co-resident service fails the
boot) and kills known guard-margin eaters; an `ExecStartPost` re-triggers the dependent-services
gate after every restart.

## Measured dead ends (2026-07-10) — so you don't repeat them

- **safe-TILE64 prefill kernel**: TILE=64 loads with order-preserving 2×32 `tl.split` sub-tiling is
  provably **bit-identical** (parity rel 0.0 on all cases incl. SWA/mixed) but **0.82–0.87× SLOWER**
  — permute/split register pressure eats the wide-load win. The raw joint-64 path is 1.45–1.56×
  faster but numerically reordered (rel-err 3-7e-3) — rejected. Patcher + parity harness kept under
  `mods/diffkv-prefill-tune/` + `benchmarks/reference/`; the shipped default stays tile=32.
- **num_stages 3/4** on the tuned prefill launch: consistently ≤ stages=2 (measured twice,
  prod-identical kernel).
- **Draft-model fp8** (`speculative_config quantization:"fp8"`): architecturally incompatible with
  DFlash's fused context-KV projection — the drafter reads raw `.weight` tensors at init
  (qwen3_dflash.py:444/516) and fp8-packed layers yield a zero-width fused weight. Fix would
  require rewriting the fusion for quantized weights; parked at +3-4% expected gain.
- **MoE prefill GEMM swap**: at the production chunk size the fused nvfp4 MoE kernel already runs
  at **83.5% of the memory roofline** (96% at chunk 512); autotune on/off is within 2-5%. No
  backend swap can pay here.
- **Decoupled DFlash proposer** (KV-less custom_class): the drafter KV group costs only 3-6% of
  per-request pool in this vLLM's admission accounting — the mechanism's prize doesn't exist here,
  and per-rank bf16 drafter replicas make the literal port net-negative.
- **Async scheduling via the Ray V2 executor**: −4 to −7% single-stream, +3% at 8 streams on
  cross-node TP=2 — net negative for a "one big + few side streams" workload.

## Sampling sweep

<a name="sampling-sweep"></a>
14-run sweep of tool-eval-bench (69 scenarios, same engine/config, sequential) on the prod config:

| arm | runs | mean |
|---|---|---|
| **temperature 0.0 (recommended)** | 88 / 86 / 88 | **87.3** |
| temp 0.3, top_p 0.95 | 87 / 91 | 89.0 |
| temp 0.7, top_p 0.95 | 85 / 91 | 88.0 |
| temp 1.0, top_p 0.95 (model generation_config default) | 86 / 84 | **85.0** |
| temp 0.0 + thinking ON (`enable_thinking: true`) | 89 / 88 | 88.5 |

Takeaways: (1) run-to-run noise is **±2 points even at greedy** (speculative-decode + batching
nondeterminism) — single-run tool-bench comparisons below ~4 points are noise; (2) no temperature
arm beats baseline+noise, and temperature mainly inflates variance; (3) the model-card default
(temp 1.0 / top_p 0.95) is consistently the **worst** arm for tool-calling. Thinking mode gains
nothing on tool-calling either (+1.2, inside noise) while adding per-call latency. We serve greedy
with thinking off for agentic/tool use. Sweep runner: `benchmarks/sampling_sweep.sh`.

## Why this exists

MiMo-V2.5 (a ~309B-total MoE) has no official recipe for the GB10 / sm_121 platform, and the
naive vLLM paths (Flash-DiffKV, Triton-DiffKV) either crash or produce garbage on sm_121. This
repo captures a working, tuned configuration: the base upstream vLLM image plus a small set of
**mods** (runtime patches applied at container start) that make MiMo-V2.5 + DFlash + fp8-KV run
correctly and fast on dual GB10 — including at deep context, which is where the interesting
engineering lives.

## Deep-context decode

Keeping speculative decoding useful *at depth* — not just at short context — took two fixes.
Both ship in the mods and are **enabled by default**; together they take deep single-stream
decode at 400K from ~4 tok/s (drafter effectively off) to **~28–35 tok/s** on structured content.

### 3D split-KV for the DFlash verify shape (`diffkv-3d-qlen8`)

DFlash's verify shape is `q_len = 8`, but the DiffKV Triton launcher gated its 3D
split-KV / FlashDecoding path behind `max_seqlen_q > 1` — so the 8-wide verify fell back to a
**2D kernel running ~16 CTAs on the 48-SM GPU (~6–7 % of memory bandwidth)**. Admitting
`q_len ≤ 8` into the 3D path (env `VLLM_DIFFKV_3D_Q8=1`) launches ~256 CTAs instead:

- **deep-spec decode +85 % @ 400K** (3.9 → 7.2 tok/s), memory-neutral, numerically identical
  (needle-in-haystack @125K verified, no sm_121 miscompile at 64+ softmax segments).

### Speculation past 262 144 tokens (`dflash-cliff-fix` + a drafter config bump)

The DFlash drafter's RoPE `cos_sin_cache` is sized from its `max_position_embeddings` (`262144`),
and vLLM's `_input_fits_in_drafter` guard silently **disables speculation above 262 136 tokens**
to avoid an out-of-bounds RoPE gather — so at 300K–500K the drafter was off (`accept_len = 1.0`)
and deep decode crawled. The fix has three parts:

1. Bump the drafter's `config.json` → `"max_position_embeddings": 524288` (both nodes).
2. `dflash-cliff-fix` (env `VLLM_DFLASH_MAXPOS=1`) extends the RoPE table to the target
   `max_model_len` and adds a **saturating** position clamp as an OOB belt (never EAGLE-style
   clamp-to-zero — that corrupts the relative RoPE offset).
3. **Wipe the torch-compile cache once** as root (`sudo rm -rf ~/.cache/vllm/torch_compile_cache`
   on both nodes) so the drafter recompiles with the extended RoPE. The cache is keyed on config
   and lives on a host bind-mount that **survives container recreation** — otherwise a stale hit
   serves the old 262144-bound Inductor kernel and the fix silently no-ops (verify the fresh
   `…/eagle_head/computation_graph.py` shows `cos_sin_cache: bf16[524288, 64]` before traffic).

Acceptance at 400K then goes **1.0 → ~3.8** on structured content (JSON/code) → **deep-spec decode
~4× (≈7 → ~28–35 tok/s @400K)**, quality-neutral: speculation is lossless — the target verifies
every token, and the config only changes *where* the drafter engages, not the output. The drafter's
sliding-window-1024 attention keeps acceptance healthy at any absolute depth (relative RoPE offsets
stay in-distribution).

### What doesn't help — and why

- **KV-sparsity (Quest)** and **acceptance-adaptive `num_speculative_tokens`** both change the
  decode `q_len`/shape at runtime → the batch misses the captured CUDA graph → attention falls back
  to eager, which *negates* the saving exactly at deep context where it's needed. Only changes that
  **respect the captured shape** or are **orthogonal** to it pay off.
- **MoE-backend autotune** is neutral — deep decode is weight-bandwidth-bound, not MoE-GEMM-bound.

### The ceiling

Deep single-stream @400K is bounded by a ~34–38 ms/step MoE weight-streaming floor plus the KV
read. With speculation on, the verify step runs at **78–100 % of the LPDDR roofline** — the decode
kernel well is dry because the kernels won. Realistic deep decode is ~28–35 tok/s (structured) /
~14–18 (prose).

### Decode context-parallelism (DCP) — architecturally dead here

A tempting lever is DCP — splitting the KV sequence across both nodes for a projected ~1.5–1.8× at
depth. **It cannot work on this configuration.** vLLM gates DCP for non-MLA attention behind
`tensor_parallel_size > total_num_kv_heads` (`vllm/config/model.py`) — and that gate encodes
*correctness*, not preference: non-MLA DCP works by de-duplicating KV-head replicas across ranks
above `num_kv_heads`, and at TP=2 with MiMo's 4 full-attention KV heads there are no replicas to
split — plain TP already stores zero redundant KV. `dcp_size=2` would need **TP≥8**. A drafted
DiffKV DCP implementation lives in `mods/dcp-diffkv` (env-gated, off, **numerically invalid** —
kept only as an engineering-history artifact). MLA models (single latent KV head, e.g.
DeepSeek-class) are the real DCP audience — see upstream PR #44573.

## Results snapshot (2026-07-08, before the 07-10 update)

| Metric | Value |
|---|---|
| KV-cache pool (util 0.86, mml 500K, `limit-mm-per-prompt video:0`) | ~1.5M tokens (3.18×). The single biggest lever was *not* the KV format — it was dropping the multimodal **video worst-case from startup profiling** (`{"image":2,"video":0,"audio":1}`), which was reserving ~10.6 GiB/rank of encoder cache and capping the pool at ~236K. |
| Long context | validated to 240K real tokens with correct needle recall; `max_model_len=500000` |
| Short-context decode | JSON ~48–52 / code ~44 tok/s |
| Multi-stream (code, mns=8, cudagraph) | 1: 33 · 2: 52 · 4: 69 · 8: 111 tok/s aggregate |
| Deep decode without the deep-context fixes | 100K: 12 · 200K: 10 · 300K: 5 · 400K: 4 tok/s |
| Tool-calling (tool-eval-bench) | 90, 0 safety warnings |

**Measure honestly.** Two traps we fell into and fixed: (1) `temperature=0` + repetitive/synthetic
prompts *inflate* speculative-decoding accept-length and overstate tok/s — measure at your real
serving temperature with real, non-repetitive content; (2) speculative decode streams **per step,
not per token** — count tokens via `usage.completion_tokens` (`stream=False`), never SSE chunks.
Speculative decode is single-stream-latency optimized: the per-stream drafter cost does not
amortize across concurrent requests, so aggregate throughput scales sub-linearly.
