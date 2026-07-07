# MiMo-V2.5-DFlash on 2× DGX Spark (GB10 / sm_121) — fp8-KV + Speculative Decoding

An optimized [vLLM](https://github.com/vllm-project/vllm) serving recipe for running
**Xiaomi [MiMo-V2.5](https://huggingface.co/XiaomiMiMo)** across **two NVIDIA DGX Spark
(GB10, sm_121a)** nodes with:

- **DFlash speculative decoding** (5-layer Qwen3 drafter, `num_speculative_tokens=7`) — ~2× single-stream throughput
- **fp8 (e4m3) KV cache** with an in-kernel descale fast-path — **~5.8× larger KV pool** than bf16, at negligible quality cost (KV error ≈ 0.027 vs nvfp4 ≈ 0.095)
- **CUDA graphs** tuned for the DFlash decode batch shape (capture sizes = multiples of 8)
- **Cross-node tensor parallelism (TP=2)** over Ray, with a cluster-join race fix

This is the daily-driver configuration we run on a home 2× GB10 cluster. It is text+omni
capable and exposes an OpenAI-compatible API.

> **Status:** production. Serves an OpenAI-compatible endpoint; wired as the backend for an
> agentic harness (tool-calling + reasoning).

## Why this exists

MiMo-V2.5 (a ~309B-total MoE) has no official recipe for the GB10 / sm_121 platform, and the
naive vLLM paths (Flash-DiffKV, Triton-DiffKV) either crash or produce garbage on sm_121. This
repo captures a working, tuned configuration: the base upstream vLLM image plus a small set of
**mods** (runtime patches applied at container start) that make MiMo-V2.5 + DFlash + fp8-KV run
correctly and fast on dual GB10.

## Results (measured)

| Metric | Value |
|---|---|
| KV-cache pool (util 0.86, mml 500K, **`limit-mm-per-prompt video:0`**) | **~1.5M tokens** (3.18× a 500K request). The single biggest lever was *not* the KV format — it was dropping the multimodal **video worst-case from startup profiling** (`{"image":2,"video":0,"audio":1}`), which was reserving ~10.6 GiB/rank of encoder-cache; that reservation is what capped the pool at ~236K. Measure with your real service load online — don't strip services to inflate it. |
| Long context | validated to **240K real tokens with correct needle recall**; `max_model_len=500000` |
| Short-context decode | JSON ~48–52 / code ~44 tok/s |
| Multi-stream (code, mns=8, cudagraph) | 1: 33 · 2: 52 (28 ea) · 4: 69 (20 ea) · 8: 111 tok/s aggregate (sub-linear — see below) |
| **Deep decode, honest** (real prose, **real temperature**, thinking-off) | *baseline (spec collapsed at depth):* 100K: 12 · 200K: 10 · 300K: 5 · 400K: 4 tok/s. Prefill ≈ 0.8 s / 1K tokens (400K ≈ 5 min cold). **With the deep-context fixes (below): ~28–35 (structured) / ~14–18 (prose) tok/s @400K.** |
| Tool-calling quality (tool-eval-bench) | 90 / ★★★★★ Excellent, 0 safety warnings |
| Reliability (reliability-bench v1_full) | on par with the fp16/fp8 baseline (existence 24/24, refusal 9/9) |

**Measure honestly.** Two traps we fell into and fixed: (1) `temperature=0` + repetitive/synthetic
prompts *inflate* speculative-decoding accept-length and overstate tok/s — measure at your real
serving temperature with real, non-repetitive content; (2) speculative decode streams **per step,
not per token**, so count tokens via `usage.completion_tokens` (`stream=False`), never by counting
SSE chunks. Under real conditions the DFlash speedup largely collapses on unpredictable prose, and
deep-context decode is bounded by full-attention O(context) growth — 500K context is usable
(works, recalls) but slow to decode at depth. Speculative decode is single-stream-latency
optimized: the per-stream drafter cost does not amortize across concurrent requests, so aggregate
throughput scales sub-linearly (good for "one big stream + a few side streams").

## Hardware

- 2× NVIDIA DGX Spark (GB10, sm_121a, 128 GB unified memory each, ~120 GB usable)
- Cross-node interconnect for NCCL (RoCE)
- Head node (rank 0) + worker (rank 1), TP=2 over Ray

## Base image & build environment

The recipe runs inside a locally-built vLLM image; the `mods/` reproduce the exact deltas on
top of upstream vLLM, so the setup is reproducible from an equivalent build.

| | |
|---|---|
| Base image | `vllm-node-mimo-v25-upstream` (local build), 19.4 GB, **linux/arm64** |
| vLLM | `0.23.1rc1.dev760+g3775d5fca` (upstream `main` @ 3775d5fca, ~2026-07-05, post PR #46104) |
| CUDA | 13.2.0 (`NVIDIA_REQUIRE_CUDA cuda>=13.2`, driver ≥ 535) · Python 3.12 |
| GPU arch | NVIDIA GB10 (Grace-Blackwell), **sm_121a**, ARM64 host, 128 GiB unified memory/node |
| Weights on GPU | ~86.9 GiB/rank (MoE nvfp4 + o_proj MXFP8) |

**Already native in this vLLM build** (the mods do NOT re-add these): `TRITON_ATTN_DIFFKV`
attention backend (PR #41797); `MiMoV2` / `MiMoV2Omni` / `DFlashDraftModel` in the model
registry; DFlash aux+1 / SWA window symmetrization (#40727).

**What the mods add on top** (upstream still lacks these for this exact fp8-KV + DFlash path):
- `fp8-kv-inline` — accept fp8 KV in the DiffKV Triton backend + in-kernel bitcast+multiply descale
- `fix-mimo-v2-upstream` — `MimoV2Config` HF registration + Omni audio deps (soundfile/librosa/av)
- `mimo-chat-template` — MiMo chat template (bounded reasoning) + reasoning parser
- `nvfp4-draft-bf16` / `nvfp4-draft-blocksize` — force the DFlash drafter KV to bf16 / block-16 (anti-padding)
- `ray-cvd-fallback` — Ray accelerator ordinal fallback (fixes a CUDA_VISIBLE_DEVICES crash under TP)
- `omni-eagle3` — expose `SupportsEagle3` on `MiMoV2Omni` so the DFlash drafter can attach its EAGLE3 aux-hidden-state interface (upstream #46104 added it only to the non-Omni `MiMoV2Flash` class)
- `diffkv-3d-qlen8` / `dflash-cliff-fix` — deep-context speculative-decode fixes (see **Deep-context speed** below)

**Model artifacts** (not in this repo — pull / quantize separately):
- Target: **MiMo-V2.5** quantized `modelopt_mixed` (NVFP4 MoE + o_proj MXFP8), served from `MiMo-oproj-mxfp8`
- Drafter: **DFlash** (5-layer Qwen3, non-causal, sliding-window 1024), `num_speculative_tokens=7`
- Served under two aliases: `MiMo-V2.5-NVFP4` and `mimo-dflash-test`

> The base image itself is not published (it's a large local CUDA build). To reproduce: build
> upstream vLLM `main` @ ~3775d5fca for CUDA 13.2 / sm_121, then apply the `mods/` at container
> start (each mod is a marker-guarded, idempotent `run.sh` that patches site-packages in place).

## Repository layout

```
recipes/
  mimo-fp8kv-prod.yaml       # the production recipe (cudagraph, mns=8, fp8-KV, DFlash)
mods/                        # runtime patches, applied in recipe order at container start
  drop-caches/               # page-cache drop before launch
  fix-mimo-v2-upstream/      # MiMo-V2.5 support for the upstream vLLM image (config reg, DiffKV fp8, audio deps)
  nvfp4-draft-bf16/          # force the DFlash drafter's KV to bf16
  nvfp4-draft-blocksize/     # drafter block_size 32->16 (reduce uniform-page padding)
  fp8-kv-inline/             # in-kernel fp8 KV descale fast-path (bitcast + multiply)
  diffkv-3d-qlen8/           # 3D split-KV for the DFlash q_len=8 verify shape (+85% deep-spec @400K)
  dflash-cliff-fix/          # unlock speculation past 262K tokens (extend drafter RoPE to max_model_len)
  ray-cvd-fallback/          # Ray accelerator ordinal fallback (fixes a CVD crash under TP)
  omni-eagle3/               # add SupportsEagle3 to MiMoV2Omni (DFlash EAGLE3 aux interface)
  mimo-chat-template/        # writes the MiMo chat template (bounded reasoning) into the container
  dcp-diffkv/                # EXPERIMENTAL (off by default) — decode context-parallelism draft; see "Honest ceiling"
systemd/
  mimo-fp8kv-prod.service    # persistent daily-driver unit (auto-boot, crash-restart, memory cleanup)
  mimo-fp8kv-pre-start.sh    # pre-start cleanup: stop containers + UVM reset + drop_caches (both nodes)
```

## Key configuration knobs

| Knob | Value | Why |
|---|---|---|
| `--kv-cache-dtype fp8` + `VLLM_FP8_INLINE=1` | fp8 e4m3, in-kernel descale | 5.8× KV pool; dequant = bitcast+multiply (3.5–4.8× faster than nvfp4 unpack+LUT) |
| `--speculative-config … method=dflash, num_speculative_tokens=7` | DFlash drafter | ~2× single-stream |
| `--compilation-config cudagraph_capture_sizes=[1,2,4,8,16,24,32,48,64]` | CUDA graphs | multiples of 8 = DFlash decode batch (num_seqs × (num_spec+1)); +5–8% at concurrency |
| `gpu_memory_utilization: 0.86` | hard ceiling on GB10 | 0.87/0.88 freeze the node (unified-memory startup guard) |
| `max_num_seqs: 8` | concurrency cap | "1 big + a few side streams" |
| `--load-format auto` | — | `fastsafetensors` (GDS/cufile) freezes GB10; `instanttensor` costs ~40% pool |
| `--chat-template mimo_chat_template.jinja` | bounded reasoning | prevents the reasoning-boundary over-thinking that truncates long answers |

## GB10 operational notes (hard-won)

- **Unified-memory freeze:** filling memory hangs the whole node. Keep `gpu_memory_utilization ≤ 0.86`.
- **Pool size depends on free memory at boot:** the KV pool is sized at profiling time, so a clean
  boot (full reboot → `drop_caches`) yields a materially larger pool than a dirty one. The pre-start
  script does UVM reset + `drop_caches` on both nodes.
- **`fastsafetensors` freezes GB10** — use `--load-format auto`.
- Never `grep` recursively over `/proc` (kcore) on GB10 — it soft-locks the CPU.

## Quick start

```bash
# Head node (rank 0):
./run-recipe.sh -d recipes/mimo-fp8kv-prod.yaml

# Or as a persistent service:
sudo cp systemd/mimo-fp8kv-prod.service /etc/systemd/system/
sudo cp systemd/mimo-fp8kv-pre-start.sh /usr/local/bin/
sudo systemctl enable --now mimo-fp8kv-prod
```

The recipe brings up both nodes (TP=2 over Ray) and serves an OpenAI-compatible API.

```bash
curl http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mimo-dflash-test","messages":[{"role":"user","content":"Write a binary search in Python."}]}'
```

## Credits

- [Xiaomi MiMo-V2.5](https://huggingface.co/XiaomiMiMo) — the base model
- [vLLM](https://github.com/vllm-project/vllm) — the serving engine
- DFlash speculative decoding

## License

Configuration and mods: MIT. The model weights are governed by their own license (see the
MiMo-V2.5 model card).

## Deep-context speed (update 2026-07)

Two fixes make **speculative decoding work — and stay fast — at deep context**, across the full 500K window.

### 1. 3D split-KV for the DFlash verify shape (`mods/diffkv-3d-qlen8`)

DFlash's verify shape is `q_len = 8`, but the DiffKV Triton launcher gated the 3D
split-KV / FlashDecoding path behind `max_seqlen_q > 1` — so the 8-wide verify fell
back to a **2D kernel running ~16 CTAs on the 48-SM GPU (~6–7 % of memory bandwidth)**.
Relaxing the gate to admit `q_len ≤ 8` into the 3D path (env `VLLM_DIFFKV_3D_Q8=1`)
launches ~256 CTAs instead:

- **deep-spec decode +85 % @ 400K** (3.9 → 7.2 tok/s), memory-neutral, numerically
  identical (needle-in-haystack @125K verified, no sm_121 miscompile at 64/… segments).

### 2. Speculation past 262 144 tokens (`mods/dflash-cliff-fix` + drafter config bump)

The DFlash drafter's RoPE `cos_sin_cache` is sized from its `max_position_embeddings`
(`262144`). vLLM's `_input_fits_in_drafter` guard silently **disables speculation above
262 136 tokens** to avoid an out-of-bounds RoPE gather — so at 300K–500K the drafter was
off (`accept_len = 1.0`) and deep decode crawled. The fix, three parts:

1. Bump the drafter's `config.json` → `"max_position_embeddings": 524288` (both nodes).
2. `mods/dflash-cliff-fix` (env `VLLM_DFLASH_MAXPOS=1`) extends the RoPE table to the
   target `max_model_len` and adds a **saturating** position clamp as an OOB belt
   (never EAGLE-style clamp-to-zero — that corrupts the relative RoPE offset).
3. **Wipe the torch-compile cache once** as root
   (`sudo rm -rf ~/.cache/vllm/torch_compile_cache` on both nodes) so the drafter
   recompiles with the extended RoPE. The cache is keyed on config and lives on a host
   bind-mount that **survives container recreation** — otherwise a stale hit serves the
   old 262144-bound Inductor kernel and the fix silently no-ops (verify the fresh
   `…/eagle_head/computation_graph.py` shows `cos_sin_cache: bf16[524288, 64]` before traffic).

Result: acceptance at 400K goes **1.0 → ~3.8** on structured content (JSON/code) →
**deep-spec decode ~4× (≈7 → ~28–35 tok/s @400K)**, quality-neutral (speculation is
lossless — the target verifies every token; the config only changes *where* the drafter
engages, not the output). The drafter's sliding-window-1024 attention keeps acceptance
healthy at any absolute depth (relative RoPE offsets stay in-distribution).

### What didn't work (for the record)

- **KV-sparsity (Quest)** and **acceptance-adaptive `num_speculative_tokens`**: both change
  the decode `q_len`/shape at runtime → the batch misses the captured CUDA graph → attention
  falls to eager, which *negates* the saving exactly at deep context where it's needed. Only
  changes that **respect the captured shape** (the two above) or are **orthogonal** to it pay off.
- **MoE-backend autotune**: neutral — deep decode is weight-bandwidth-bound, not MoE-GEMM-bound.

### Honest ceiling

Deep single-stream @400K is bounded by a ~34–38 ms/step MoE weight-streaming floor plus the
KV read. Realistic deep decode is **~28–35 tok/s (structured) / ~14–18 (prose)** — not 40+.
The attention kernel itself reads at only ~21 % of the 273 GB/s LPDDR bandwidth (parallelism-
starved at `q_len=1`), but widening it helps only the no-spec path. The next real lever is
**decode context-parallelism** (split the KV sequence across both GB10 nodes for ~1.5–1.8×
deep). A first implementation is **drafted** in `mods/dcp-diffkv` (env-gated behind
`VLLM_DCP` + `VLLM_DCP_STAGE2_OK`, off by default, so it never runs in the prod recipe):
the per-token LSE expose from the DiffKV reduce, the exact LSE-weighted cross-rank combine,
and the metadata seq-len split are in place. The context/query `seqused_k` split and numeric
parity are **marked placeholders pending a live single-boot validation** — it is *not yet
enabled or benchmarked*, so treat the ~1.5–1.8× as a target, not a measured result.
