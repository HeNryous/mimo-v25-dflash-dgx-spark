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
| KV-cache pool (util 0.86, mml 200K) | ~240K–760K tokens (up to 5.8× bf16). The pool is sized from **free memory at KV-profiling time**, so co-resident services and boot-to-boot memory state move it a lot on GB10 unified memory. Measure it with your real service load online — don't strip services to inflate it (that just moves the OOM to later). |
| Single-stream decode | code ~33–44 / JSON ~52 / prose ~24–28 tok/s |
| Deep-context decode (200K) | ~23 tok/s (per-step ~167ms; full-attention O(ctx) growth is physics) |
| Multi-stream (code, mns=8, cudagraph) | 1: 33 · 2: 52 (28 ea) · 4: 69 (20 ea) · 8: 111 tok/s aggregate. Speculative decode is single-stream-latency optimized, so aggregate scales sub-linearly. |
| Tool-calling quality (tool-eval-bench) | 90 / ★★★★★ Excellent, 0 safety warnings |
| Reliability (reliability-bench v1_full) | on par with the fp16/fp8 baseline (existence 24/24, refusal 9/9) |

Speculative decoding is single-stream-latency optimized: the per-stream drafter cost does not
amortize across concurrent requests, so aggregate throughput scales sub-linearly (great for
"one big stream + a few side streams", less so for many-way batching).

## Hardware

- 2× NVIDIA DGX Spark (GB10, sm_121a, 128 GB unified memory each, ~120 GB usable)
- Cross-node interconnect for NCCL (RoCE)
- Head node (rank 0) + worker (rank 1), TP=2 over Ray

## Repository layout

```
recipes/
  mimo-fp8kv-prod.yaml       # the production recipe (cudagraph, mns=8, fp8-KV, DFlash)
mods/
  fix-mimo-v2-upstream/      # MiMo-V2.5 support for the upstream vLLM image (config reg, DiffKV fp8, audio deps)
  fp8-kv-inline/             # in-kernel fp8 KV descale fast-path (bitcast + multiply)
  nvfp4-draft-bf16/          # force the DFlash drafter's KV to bf16
  nvfp4-draft-blocksize/     # drafter block_size 32->16 (reduce uniform-page padding)
  ray-cvd-fallback/          # Ray accelerator ordinal fallback (fixes a CVD crash under TP)
  mimo-chat-template/        # writes the MiMo chat template (bounded reasoning) into the container
  drop-caches/               # page-cache drop before launch
systemd/
  mimo-fp8kv-prod.service    # persistent daily-driver unit (auto-boot, crash-restart, memory cleanup)
  mimo-fp8kv-pre-start.sh    # pre-start cleanup: stop containers + UVM reset + drop_caches (both nodes)
docs/
  DESIGN.md                  # the engineering notes (why each mod exists)
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
