# MiMo-V2.5-DFlash on 2× DGX Spark (GB10) — fp8-KV + Speculative Decoding

A production [vLLM](https://github.com/vllm-project/vllm) recipe for
**Xiaomi [MiMo-V2.5](https://huggingface.co/XiaomiMiMo)** (~309B MoE) on **two NVIDIA DGX Spark
(GB10, sm_121a)** nodes: DFlash speculative decoding, fp8 KV cache, tuned kernels, 500K context.
This is the exact daily-driver configuration we serve — every number below was measured on it.

> Engineering notes, A/B data and the full optimization history live in **[HISTORY.md](HISTORY.md)**.

## Hardware

- 2× NVIDIA DGX Spark (GB10, sm_121a, 128 GB unified memory each, ~120 GB usable)
- Cross-node interconnect for NCCL (RoCE) · head (rank 0) + worker (rank 1), TP=2 over Ray

## Quick start

```bash
# 1) Pull the pre-built vLLM image (linux/arm64, CUDA 13.2, sm_121a, ~19.4 GB):
docker pull ghcr.io/henryous/mimo-v25-dflash-dgx-spark:latest

# 2) Assemble the exact prod checkpoint:
#    a) base ~171 GB:
hf download lukealonso/MiMo-V2.5-NVFP4 --local-dir MiMo-oproj-mxfp8
#    b) our o_proj-MXFP8 overlay (4 files, ~1.7 GB) INTO that same dir
#       (overwrites the 3 JSONs, adds the requant shard):
gh release download prod-overlay-v1 -R HeNryous/mimo-v25-dflash-dgx-spark -D MiMo-oproj-mxfp8
#    c) DFlash drafter (dflash/ subdir of XiaomiMiMo/MiMo-V2.5-DFlash) into mimo-dflash/dflash/,
#       then bump its config.json: "max_position_embeddings": 524288  (both nodes)

# 3) One-time: clear any stale torch-compile cache (both nodes), or the drafter
#    silently keeps the old 262K RoPE table:
sudo rm -rf ~/.cache/vllm/torch_compile_cache

# 3b) IMPORTANT — where weights must live: recipe `model:` paths are paths
#     INSIDE the container. The launcher mounts $HF_HOME (default
#     ~/.cache/huggingface) to /root/.cache/huggingface, so the assembled
#     checkpoint must sit at  ~/.cache/huggingface/MiMo-oproj-mxfp8  and the
#     drafter at  ~/.cache/huggingface/mimo-dflash/dflash  — on BOTH nodes,
#     same path. If you use a custom HF_HOME, export it on both nodes.
#
# 3c) One-time cluster config: node IPs come from .env (CLUSTER_NODES=ip1,ip2)
#     or run  ./launch-cluster.sh --setup  for autodiscovery.
#
# 4) Launch — the recipe applies mods/ at container start (head node, rank 0):
./run-recipe.sh -d recipes/mimo-fp8kv-prod.yaml

# Or as a persistent service:
sudo cp systemd/mimo-fp8kv-prod.service /etc/systemd/system/
sudo cp systemd/mimo-fp8kv-pre-start.sh systemd/mimo-fp8kv-post-start.sh /usr/local/bin/
sudo systemctl enable --now mimo-fp8kv-prod
```

The recipe brings up both nodes (TP=2 over Ray) and serves an OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mimo-dflash-test","messages":[{"role":"user","content":"Write a binary search in Python."}]}'
```

Serve **greedy (temperature 0) with thinking off** for agentic/tool use — a 14-run sampling sweep
found the model-card defaults (temp 1.0 / top_p 0.95) measurably *worse* for tool-calling and no
arm better than greedy ([details](HISTORY.md#sampling-sweep)).

## Expected results

All measured on this exact config (2× GB10, TP=2, all co-resident services online).
Tokens counted via `usage.completion_tokens`, never SSE chunks (DFlash streams per *step*).

**Quality**

| Benchmark | Score |
|---|---|
| reliability-bench `v1_full` (79 tasks) | **57/79** first-pass (existence 23/24, refusal 9/9) |
| tool-eval-bench (69 scenarios) | **87 ± 2** (11-run band; single runs up to 91), 0 safety warnings |
| Loop battery (12 loop-prone prompts) | 0/12 loops |
| Needle-in-haystack | PASS @126K, validated to 240K real tokens |

**Throughput — single stream** (greedy, real content, short context)

| Content | tok/s |
|---|---|
| JSON / structured | ~54 (accept-length ≈ 5.5) |
| Code | ~44 |
| Prose | ~28 |

**Throughput — concurrent streams** (400-token JSON generations, temp 0.7)

| Streams | Aggregate tok/s | Per-stream tok/s |
|---|---|---|
| 1 | 37 | 37 |
| 2 | 39 | 19 |
| 4 | 86 | 21 |
| 6 | 113 | 19 |

**Latency and depth** (cold = unique prompt, no prefix-cache hit)

| Context depth | Cold TTFT | Decode after prefill |
|---|---|---|
| 50K | ~24 s | ~50+ tok/s (structured) |
| 200K | ~170 s | ~22–25 tok/s |
| 400K | ~580 s | ~28–35 structured / ~14–18 prose tok/s |

**Multi-stream under load** — a 150K-token prefill running next to a live decode stream:
prefill completes in **89 s**, the parallel stream keeps flowing at **148 ms median** inter-token
gap (worst ~4 s). Without the shipped fairness flag this freezes the stream for the full ~2 minutes.

**Context**: `max_model_len 500000`, KV pool **1.67M tokens** (3.33× concurrency at full 500K),
measured with neo4j/monitoring/agents co-resident — don't strip services to inflate pool numbers.

## Model & quantization

- **Target — MiMo-V2.5** in NVIDIA ModelOpt `MIXED_PRECISION`, built from
  [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4) + our
  [`prod-overlay-v1`](https://github.com/HeNryous/mimo-v25-dflash-dgx-spark/releases/tag/prod-overlay-v1)
  release (o_proj → MXFP8 requant; the overlay is **required** — the stock checkpoint garbles on
  this image). NVFP4 experts, MXFP8 `o_proj`, bf16 `lm_head`/embeddings. ~86.9 GiB/rank on GPU.
- **Drafter — DFlash** (5-layer Qwen3, non-causal, sliding-window 1024), `num_speculative_tokens=7`.
- **KV cache** quantized at runtime to **fp8-e4m3** (decode-time cache format, independent of weights).
- Base image: upstream vLLM `main` @ 3775d5fca (`0.23.1rc1.dev760`), CUDA 13.2, Python 3.12.
  The `mods/` are marker-guarded idempotent patches applied at container start — see
  [repository layout](#repository-layout) and [HISTORY.md](HISTORY.md) for what each one does and why.

## Key configuration knobs

| Knob | Value | Why |
|---|---|---|
| `--kv-cache-dtype fp8` + `VLLM_FP8_INLINE=1` | fp8 e4m3, in-kernel descale | 5.8× KV pool vs bf16; dequant = bitcast+multiply |
| `--speculative-config … dflash, num_speculative_tokens=7` | DFlash drafter | ~2× single-stream |
| `VLLM_DIFFKV_3D_Q8=1` + `VLLM_DFLASH_MAXPOS=1` | deep-context spec | speculation stays on and fast past 262K |
| `mods/diffkv-prefill-tune` (default-on) | prefill kernel tune | bit-identical, cold TTFT −13/−23/−28 % @50/200/400K |
| `--long-prefill-token-threshold 4096` | prefill/decode fairness | no decode freeze during long prefills (vLLM's auto-default is inert — set explicitly) |
| `VLLM_DIFFKV_BW=1` (`mods/diffkv-kernel-bw`) | 3D decode segments 16→64 | +18 % deep decode @200K |
| `cudagraph_capture_sizes=[1,2,4,8,…,64]` | CUDA graphs | multiples of 8 = DFlash batch shape |
| `gpu_memory_utilization: 0.86` | hard GB10 ceiling | 0.87/0.88 freeze the node |
| `--load-format auto` | — | `fastsafetensors` freezes GB10 |

**Throughput — content-bound reality (measured 2026-07-12, fresh boot)**

DFlash accept-length is set by how predictable the CONTENT is, not by load:

| Content | single stream | 6 streams (per-stream / aggregate) | accept |
|---|---|---|---|
| JSON / structured / code | ~48-59 | 26-30 / ~136 | 4.5-5.5 (stable under load) |
| Free prose (any language, any temp) | ~23 | 11-14 / ~60 | ~2.0-2.6 |

Multi-stream scaling is MoE-coverage physics: 48 concurrent tokens activate
nearly all experts, so a step reads ~2.5x the weight bytes of a single-stream
step over the same fixed bandwidth. Aggregate keeps growing with more streams
(coverage saturates; ~490 tok/s measured at C=32 on an earlier cut).

**Substrate health rule of thumb:** prose at 23 and JSON at 48+ is normal.
If JSON tasks drop clearly below ~40 tok/s single-stream, suspect the
substrate (swap thrashing, see operational notes) — not the model.

## GB10 operational notes

- **Unified-memory freeze:** keep `gpu_memory_utilization ≤ 0.86`. Filling memory hangs the node.
- **Pool size is set at boot profiling:** clean boot (UVM reset + `drop_caches`) = bigger pool.
  The pre-start script does this on both nodes, pauses co-resident services during profiling
  (the 0.86 guard is razor-thin — any ~1–2 GiB neighbor fails the boot), and the post-start
  re-triggers them after vLLM is healthy.
- **Swap thrashing silently destroys throughput (and how to make it harmless):**
  at util 0.86 the weight-load itself displaces ~5-6 GB to swap on every boot, and any
  memory-pressure burst (e.g. many parallel background requests) pushes *hot* pages after
  it. Symptom: everything still "works" but decode collapses (we measured 11 tok/s single /
  accept 2 on JSON that normally does 48+), `vmstat` shows continuous swap-ins, and RAM
  "available" looks fine again while GBs sit in swap. Fix — route swap into compressed RAM:

  ```bash
  # both nodes
  sudo apt-get install -y zram-tools
  printf 'ALGO=zstd\nSIZE=8192\nPRIORITY=100\n' | sudo tee /etc/default/zramswap
  sudo systemctl enable --now zramswap.service && sudo systemctl restart zramswap.service
  swapon --show   # /dev/zram0 prio 100 must sit above the disk swapfile (prio -2)
  ```

  After the next reboot the boot-time displacement lands entirely in zram (~4.3 GB
  compressed to ~1.5 GB physical in our case, disk swap stays at 0 B) and future
  pressure spikes cost microseconds instead of SSD reads. Note for benchmarking:
  numbers taken while a node is thrash-degraded are garbage — reboot first.
- **Protect the engine from the OOM killer:** Ray sets its workers to
  `oom_score_adj 1000`, so under true memory exhaustion the kernel kills your
  100-GB vLLM worker FIRST. `systemd/oomprotect.sh` pins the serving processes to
  `-500`; wire it into your post-start hook (see `systemd/mimo-fp8kv-post-start.sh`)
  so it re-applies on every boot.
- **`fastsafetensors` freezes GB10** — use `--load-format auto`.
- Never `grep` recursively over `/proc` on GB10 (kcore soft-locks the CPU).

## Repository layout

```
recipes/mimo-fp8kv-prod.yaml   # THE production recipe (everything above wired together)
mods/                          # runtime patches, applied in recipe order at container start
  fix-mimo-v2-upstream/        #   MiMo-V2.5 + fp8-DiffKV support on the upstream image
  fp8-kv-inline/               #   in-kernel fp8 KV descale fast-path
  diffkv-3d-qlen8/             #   3D split-KV for the DFlash verify shape (+85 % deep-spec)
  diffkv-kernel-bw/            #   decode segment tuning (+18 % deep)
  diffkv-prefill-tune/         #   prefill kernel tune (bit-identical, −23 % TTFT @200K)
  dflash-cliff-fix/            #   speculation past 262K (drafter RoPE extension)
  vit-sdpa-window/             #   Omni vision fix: ViT windowed-SWA via masked SDPA + sinks
  nvfp4-draft-bf16/ nvfp4-draft-blocksize/ ray-cvd-fallback/ omni-eagle3/ mimo-chat-template/ drop-caches/
  dcp-diffkv/                  #   REFERENCE ONLY — abandoned, see HISTORY.md
systemd/                       # persistent service + hardened pre/post-start
benchmarks/                    # sampling sweep runner + kernel parity harness (reference)
```

## Credits & license

[Xiaomi MiMo-V2.5](https://huggingface.co/XiaomiMiMo) ·
[`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4) ·
[vLLM](https://github.com/vllm-project/vllm) · DFlash speculative decoding.

The launcher (`run-recipe.sh/.py`, `launch-cluster.sh`, `autodiscover.sh`) is vendored
from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (MIT,
(c) Eugene Rakhmatulin), including our local fixes that this recipe was validated with.
Configuration and mods: MIT. Model weights under their own licenses.

**How we got here** — kernel work, A/B data, measured dead ends: **[HISTORY.md](HISTORY.md)**.
