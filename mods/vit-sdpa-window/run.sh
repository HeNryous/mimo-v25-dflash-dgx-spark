#!/bin/bash
set -uo pipefail
# ============================================================================
# vit-sdpa-window — fix the LAST broken ViT attention path on sm_121:
#   the 24 windowed SWA blocks of the MiMo-V2.5 Omni vision encoder hardcode
#   flash_attn_varlen_func (window_size=[64,64], causal=False) which returns
#   corrupted output on GB10/sm_121 — the --mm-encoder-attn-backend flag only
#   reaches the 4 full-attention blocks (MMEncoderAttention), never
#   MiMoVisionAttention._forward_window_attn.
#
#   Image: vllm-node-mimo-v25-upstream (vllm 0.23.1rc1.dev760+g3775d5fca)
#
# SYMPTOM (empirical, 2026-07-11): with TORCH_SDPA full-blocks fixed, photos
#   mostly work but flat-color synthetics fail NON-DETERMINISTICALLY (same
#   image, temp 0 → "brown"/"grey"/"black"); structure survives, low-frequency
#   color dies. Classic sm_121-FA-garbage, confined to the windowed blocks.
#
# THE FIX (env-gated VLLM_VIT_SDPA_WINDOW, default 1 in this mod's recipes;
#   set 0 to A/B back to the FA path byte-for-byte):
#   1. Replace the FA call with chunked masked SDPA-math in fp32: per
#      cu_seqlens segment, banded mask |i-j|<=w, query chunks of 1024 so the
#      logits stay memory-bounded (encoder runs once per image, perf uncritical).
#   2. Apply the learned attention sinks (use_sink=True): the checkpoint's
#      per-head sink logits were loaded but SILENTLY DROPPED in the FA path
#      ("loaded but not used in vLLM flash_attn"). Standard formulation:
#      sink column appended to logits pre-softmax, dropped post-softmax.
#      TP-sharded ranks slice their contiguous head block.
# ============================================================================
TARGET=${1:-/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py}

python3 - "$TARGET" <<'PATCH_EOF'
import re, sys
path = sys.argv[1]
src = open(path).read()
if "VLLM_VIT_SDPA_WINDOW" in src:
    print("[vit-sdpa-window] already applied, skipping")
    sys.exit(0)

NEW = '''    def _forward_window_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Windowed attention via chunked masked SDPA-math (fp32) + sink.

        FA2 sliding-window varlen is corrupted on sm_121, and the FA path
        could not apply the learned sink logits. VLLM_VIT_SDPA_WINDOW=0
        restores the original flash_attn path (sinks dropped).
        """
        import os
        w = self.visual_token_window_size
        if os.environ.get("VLLM_VIT_SDPA_WINDOW", "1") == "0":
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                softmax_scale=self.scale, causal=False, window_size=[w, w])
        H = q.shape[1]
        sinks = None
        if self.use_sink and self.sinks is not None:
            s = self.sinks.to(torch.float32)
            if s.numel() != H:
                from vllm.distributed import get_tensor_model_parallel_rank
                r = get_tensor_model_parallel_rank()
                s = s[r * H:(r + 1) * H]
            sinks = s.view(H, 1, 1).to(q.device)
        rep = q.shape[1] // k.shape[1]
        out = torch.empty(q.shape[0], H, v.shape[2], dtype=q.dtype, device=q.device)
        CHUNK = 1024
        starts = cu_seqlens.tolist()
        for si in range(len(starts) - 1):
            s0, s1 = int(starts[si]), int(starts[si + 1])
            t = s1 - s0
            if t == 0:
                continue
            qs = q[s0:s1].permute(1, 0, 2).float()
            ks = k[s0:s1].permute(1, 0, 2).float()
            vs = v[s0:s1].permute(1, 0, 2).float()
            if rep > 1:
                ks = ks.repeat_interleave(rep, dim=0)
                vs = vs.repeat_interleave(rep, dim=0)
            for c0 in range(0, t, CHUNK):
                c1 = min(c0 + CHUNK, t)
                k0, k1 = max(0, c0 - w), min(t, c1 + w)
                logits = torch.bmm(
                    qs[:, c0:c1], ks[:, k0:k1].transpose(1, 2)) * self.scale
                qi = torch.arange(c0, c1, device=q.device).unsqueeze(1)
                kj = torch.arange(k0, k1, device=q.device).unsqueeze(0)
                logits = logits.masked_fill((qi - kj).abs() > w, float("-inf"))
                if sinks is not None:
                    logits = torch.cat(
                        [logits, sinks.expand(H, c1 - c0, 1)], dim=-1)
                p = torch.softmax(logits, dim=-1)
                if sinks is not None:
                    p = p[..., :-1]
                out[s0 + c0:s0 + c1] = torch.bmm(
                    p, vs[:, k0:k1]).permute(1, 0, 2).to(q.dtype)
        return out
'''

pat = re.compile(
    r"    def _forward_window_attn\(.*?\n        return output\n",
    re.S)
m = pat.search(src)
if not m:
    print("[vit-sdpa-window] ERROR: _forward_window_attn not found/shape changed")
    sys.exit(1)
open(path, "w").write(src[:m.start()] + NEW + src[m.end():])
import py_compile
py_compile.compile(path, doraise=True)
print("[vit-sdpa-window] patched _forward_window_attn -> chunked masked SDPA + sink (gate VLLM_VIT_SDPA_WINDOW)")
PATCH_EOF
