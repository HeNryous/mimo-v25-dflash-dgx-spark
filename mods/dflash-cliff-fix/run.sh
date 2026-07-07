#!/bin/bash
set -uo pipefail
# ============================================================================
# dflash-cliff-fix — lift the DFlash accept-length cliff at ctx > 262144 (2^18)
#
#   Image: vllm-node-mimo-v25-upstream:latest
#   vllm 0.23.1rc1.dev760+g3775d5fca (main @3775d5fc)
#   Drafter: /root/.cache/huggingface/mimo-dflash/dflash  (Qwen3, 5-layer, SWA1024)
#
# THE SYMPTOM (measured live, num_spec=7):
#   accept-length is content-dependent + healthy through 200K (prose 1.86 / code
#   2.57 / json 4.59) then CLIFFS to EXACTLY 1.00 / 0% accept for ALL content at
#   400K. A content-independent hard collapse to exactly zero == a fixed cap, not
#   entropy. The cap is the drafter's max_position_embeddings = 262144 = 2^18.
#
# ROOT CAUSE (this is a DELIBERATE GUARD firing, not garbage output):
#   1. The DFlash drafter checkpoint config.json sets
#        "max_position_embeddings": 262144
#   2. speculative.py:_maybe_override_draft_max_model_len (~L978) caps the drafter
#        draft_model_config.max_model_len = min(draft=262144, target=500000) = 262144
#      (the spec-config "max_model_len" knob is None here, and it can only LOWER —
#       it is validated <= draft and <= target — so it cannot raise this.)
#   3. gpu_model_runner.py:648  effective_drafter_max_model_len = 262144
#   4. gpu_model_runner.py:_input_fits_in_drafter (L4428-4441)
#        fits  iff  max_seq_len + (num_spec_tokens + 1)  <=  262144
#      i.e. spec runs only while  seq_len <= 262136.
#   5. gpu_model_runner.py:sample_tokens (L4542 / L4582-4593)
#        if fits:  run the drafter
#        else:     self._draft_token_ids = zeros(...)   ==> speculation DISABLED
#      => beyond 262136 tokens NO draft is proposed, every step verifies only the
#         base model's own next token -> accept-length is EXACTLY 1.00, 0% accept,
#         identical for prose/code/json. Exact cliff = 262144.
#   The guard exists because the drafter's RoPE cos_sin_cache is only
#   max_position_embeddings (262144) rows (qwen3_dflash.py:187/274 ->
#   rotary_embedding/base.py:97 arange(max_position_embeddings)), and the DFlash
#   context-KV precompute feeds RAW absolute positions with NO clamp
#   (spec_decode/utils.py copy_and_expand_dflash_inputs_kernel writes
#   out_context_positions unclamped; the EAGLE step kernel by contrast clamps
#   position>=max_model_len -> 0). Running the drafter past 262144 would index the
#   RoPE cache out of bounds -> garbage context K -> every draft rejected anyway.
#   So the guard trades "garbage" for "no speculation" (correct, but slow).
#
# THE FIX (one logical lever, applied in two spots so BOTH sub-issues are fixed):
#   Give the drafter a position range that covers the full target context (500K):
#   (A) qwen3_dflash.py — build the drafter RoPE cos_sin_cache with
#         max_position = max(config.max_position_embeddings, target max_model_len)
#       so absolute context positions up to the target's max_model_len are in-range
#       (no OOB when the drafter actually runs at deep context). Reads the target
#       max_model_len from the vllm_config the decoder layer already receives.
#   (B) speculative.py:_maybe_override_draft_max_model_len — for the DFlash path,
#       do NOT clamp the drafter's max_model_len below the target's, so
#       effective_drafter_max_model_len == target (500000) and _input_fits_in_drafter
#       stops firing -> the drafter is invoked at 300K/400K.
#   The drafter attends via SWA-1024 + attention_sink_bias and shares the target's
#   block_table + global KV pool (no separate mml-scaled KV pool), so extending its
#   position range costs only the RoPE cache growth (arange(500000) x rotary_dim/2
#   cos+sin -> a few hundred KB) and NO extra KV memory. Positions <= 262144 are
#   bit-identical to before (same RoPE frequencies; the table is only made longer).
#
# DEFAULT-OFF / A-B CLEAN:
#   Both hunks are gated at RUNTIME on env VLLM_DFLASH_MAXPOS. Unset/0 => the code
#   takes the exact original path (min() cap + max_position_embeddings RoPE) => a
#   clean A/B: same image, flip VLLM_DFLASH_MAXPOS=1 and relaunch.
#   Optional override VLLM_DFLASH_MAXPOS_LEN=<int> forces a specific position range
#   (else it uses the target model's max_model_len).
#
# VERIFY (A/B via Prometheus; metric names spec_decode/metrics.py:229-231):
#   Watch deltas of  vllm:spec_decode_num_drafts  and
#   vllm:spec_decode_num_accepted_tokens  at 200K / 256K / 300K / 400K.
#   accept_length = 1 + d(accepted)/d(drafts).
#   OFF: at 300K/400K num_drafts STOPS incrementing (guard) -> accept 1.00.
#   ON : num_drafts keeps incrementing past 262144 and accepted>0 -> accept > 1.00.
#   The exact cliff is the largest ctx where OFF still drafts (~262136).
# ============================================================================

SITE=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
echo "[dflash-cliff-fix] vllm at $SITE"

# --- (A) qwen3_dflash.py: RoPE max_position covers the target context ----------
python3 - "$SITE" <<'PYEOF'
import sys, ast
site = sys.argv[1]
f = site + "/model_executor/models/qwen3_dflash.py"
src = open(f).read()

old = '''        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,'''
new = '''        # dflash-cliff-fix (A): the drafter RoPE cos_sin_cache is sized to
        # max_position; the stock value (config.max_position_embeddings, e.g.
        # 262144) is SMALLER than the target's max_model_len (e.g. 500000), so at
        # deep context the drafter would index the RoPE cache out of bounds. When
        # VLLM_DFLASH_MAXPOS is set, extend the RoPE range to cover the full target
        # context (positions <= the old cap are unchanged; the table is just made
        # longer). Costs ~a few hundred KB of RoPE cache and no extra KV.
        import os as _os
        _dflash_max_position = config.max_position_embeddings
        if _os.environ.get("VLLM_DFLASH_MAXPOS", "0") not in ("0", "", "false", "False"):
            _forced = _os.environ.get("VLLM_DFLASH_MAXPOS_LEN")
            _tgt = int(_forced) if _forced else int(
                getattr(vllm_config.model_config, "max_model_len", 0) or 0
            )
            _dflash_max_position = max(_dflash_max_position, _tgt)
        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=_dflash_max_position,'''

if "dflash-cliff-fix (A)" in src:
    print("[dflash-cliff-fix][A] already patched")
elif src.count(old) == 1:
    src = src.replace(old, new)
    ast.parse(src)
    open(f, "w").write(src)
    print("[dflash-cliff-fix][A] qwen3_dflash.py RoPE max_position extended (env-gated)")
else:
    print(f"[dflash-cliff-fix][A] ANCHOR MISMATCH count={src.count(old)} — NOT patched")
    sys.exit(2)
PYEOF
A_RC=$?

# --- (B) speculative.py: don't clamp DFlash drafter max_model_len below target --
python3 - "$SITE" <<'PYEOF'
import sys, ast
site = sys.argv[1]
f = site + "/config/speculative.py"
src = open(f).read()

old = '''        result = min(
            draft_max_model_len,
            target_max_model_len,
        )
        if result != draft_max_model_len:'''
new = '''        result = min(
            draft_max_model_len,
            target_max_model_len,
        )
        # dflash-cliff-fix (B): the DFlash drafter's config caps
        # draft_max_model_len (e.g. 262144) below the target (e.g. 500000). That
        # cap becomes effective_drafter_max_model_len, and _input_fits_in_drafter
        # then DISABLES speculation for any request longer than the cap (drafts
        # zeroed -> accept-length collapses to 1.00). With VLLM_DFLASH_MAXPOS set,
        # raise the drafter's max_model_len to the target's so the guard stops
        # firing. Safe only in tandem with hunk (A), which extends the drafter
        # RoPE range to match (else the drafter would read RoPE OOB past the cap).
        import os as _os
        if _os.environ.get("VLLM_DFLASH_MAXPOS", "0") not in ("0", "", "false", "False"):
            _forced = _os.environ.get("VLLM_DFLASH_MAXPOS_LEN")
            _target = int(_forced) if _forced else target_max_model_len
            result = min(_target, target_max_model_len)
        if result != draft_max_model_len:'''

if "dflash-cliff-fix (B)" in src:
    print("[dflash-cliff-fix][B] already patched")
elif src.count(old) == 1:
    src = src.replace(old, new)
    ast.parse(src)
    open(f, "w").write(src)
    print("[dflash-cliff-fix][B] speculative.py drafter max_model_len cap lifted (env-gated)")
else:
    print(f"[dflash-cliff-fix][B] ANCHOR MISMATCH count={src.count(old)} — NOT patched")
    sys.exit(2)
PYEOF
B_RC=$?

# --- (C) qwen3_dflash.py: hard OOB guard — clamp positions to the RoPE cache ----
#   Belt-and-suspenders for (A). (A) EXTENDS the drafter RoPE cos_sin_cache to the
#   target max_model_len so absolute context positions past 262144 are in-range.
#   But ops.rotary_embedding() is an UNCHECKED CUDA gather of cos_sin_cache[pos]:
#   any position that still exceeds the cache length (e.g. if (A) is bypassed at
#   runtime, or the target mml the cache was sized to is itself exceeded) is an
#   illegal memory access -> hard container crash at the first decode step past the
#   cap (exactly the round-1 300K crash). (C) clamps positions to
#   cos_sin_cache.shape[0]-1 at BOTH RoPE-apply sites so no gather can ever OOB.
#
#   IMPORTANT — SATURATE, never zero. The drafter is pure sliding-window-1024 + sink
#   bias; with NeoX RoPE the attention score depends only on the RELATIVE offset
#   pos_q - pos_k (always <= 1024 within the SWA window), so extending the table is
#   accept-neutral. Clamping absolute positions to 0 (EAGLE-style) would corrupt the
#   context-vs-query relative offset and DEGRADE accept for context > cap; saturating
#   with min() to the last row keeps the clamp a NO-OP whenever (A) sized the cache
#   >= the max position (the normal case), acting only as a crash guard otherwise.
python3 - "$SITE" <<'PYEOF'
import sys, ast
site = sys.argv[1]
f = site + "/model_executor/models/qwen3_dflash.py"
src = open(f).read()

# (C1) decode path: DFlashQwen3Attention.forward, before applying rotary_emb.
old1 = '''        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)'''
new1 = '''        # dflash-cliff-fix (C1): saturate positions to the RoPE cache length so the
        # unchecked ops.rotary_embedding gather can never OOB past the (A)-extended
        # cache. No-op when max(pos) < cache rows; pure crash guard otherwise.
        import os as _os
        if _os.environ.get("VLLM_DFLASH_MAXPOS", "0") not in ("0", "", "false", "False"):
            _csc_rows = self.rotary_emb.cos_sin_cache.shape[0]
            positions = positions.clamp_max(_csc_rows - 1)
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)'''

# (C2) context-KV precompute: clamp the repeated context positions before RoPE.
old2 = '''        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache'''
new2 = '''        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        # dflash-cliff-fix (C2): saturate context positions to the RoPE cache length
        # (same rationale as C1) so the precompute RoPE gather can never OOB.
        import os as _os
        if _os.environ.get("VLLM_DFLASH_MAXPOS", "0") not in ("0", "", "false", "False"):
            positions_repeated = positions_repeated.clamp_max(
                self._rope_cos_sin_cache.shape[0] - 1
            )
        cos_sin_cache = self._rope_cos_sin_cache'''

if "dflash-cliff-fix (C1)" in src and "dflash-cliff-fix (C2)" in src:
    print("[dflash-cliff-fix][C] already patched")
    sys.exit(0)

n1, n2 = src.count(old1), src.count(old2)
if n1 == 1 and n2 == 1:
    src = src.replace(old1, new1).replace(old2, new2)
    ast.parse(src)
    open(f, "w").write(src)
    print("[dflash-cliff-fix][C] qwen3_dflash.py RoPE position clamps applied (env-gated)")
else:
    print(f"[dflash-cliff-fix][C] ANCHOR MISMATCH C1={n1} C2={n2} — NOT patched")
    sys.exit(2)
PYEOF
C_RC=$?

# --- syntax gate on both touched files -----------------------------------------
python3 -c "import ast; [ast.parse(open(f).read()) for f in ['$SITE/model_executor/models/qwen3_dflash.py','$SITE/config/speculative.py']]; print('[dflash-cliff-fix] both files syntax OK')" || exit 3

if [ "${A_RC:-0}" -ne 0 ] || [ "${B_RC:-0}" -ne 0 ] || [ "${C_RC:-0}" -ne 0 ]; then
  echo "[dflash-cliff-fix] WARN: one or more hunks did not apply (A_RC=$A_RC B_RC=$B_RC C_RC=$C_RC)"
  exit 2
fi
echo "[dflash-cliff-fix] done. Enable at runtime with VLLM_DFLASH_MAXPOS=1 (default OFF)."
echo "[dflash-cliff-fix] optional: VLLM_DFLASH_MAXPOS_LEN=<int> to force a position range."
