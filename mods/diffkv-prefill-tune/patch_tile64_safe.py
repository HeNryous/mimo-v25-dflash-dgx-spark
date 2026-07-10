#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A2 DIFFKV_TILE64_SAFE patcher — order-preserving TILE=64 prefill subtiling
for the DiffKV Triton kernel (triton_unified_attention_diffkv.py).

WHAT: raw TILE=64 in the A1 launch adds 1.45-1.56x on deep prefill but FAILED
strict parity (rel 3-7e-3, big%=0.00 -> pure fp32-accumulation reordering).
The audit localized the reorder to (i) one softmax rescale per 64 cols vs two
per 32, (ii) the bf16 cast of P after a 64-wide max (dominant), (iii) one
64-long PV accumulation vs two 32-long.  The QK dot is NOT a source.

THE FIX: keep TILE_SIZE=64 for the MEMORY side (one K 256x64 load + one
V 64x128 load + one block-table gather per 64 tokens — the fp8-kv-inline
descale composes untouched, it sits upstream of this patch's region) but
SPLIT the loaded tiles into their two 32-column halves (tl.split on a
reshaped view, Triton 3.6) and run the UNMODIFIED stock 32-wide sequence
(S -> softmax_step -> P-cast -> PV-acc) twice per iteration, low half first.
Iteration j at TILE=64 then performs exactly the columns, exactly the ops,
in exactly the order of stock TILE=32 iterations 2j and 2j+1 -> accumulation
order identical by construction -> bit-identical output (gate: parity harness
tun-vs-stk rel MUST be 0.0).

WHY SWA IS EXCLUDED (wrapper downgrade + tl.static_assert): softmax_step
guards an all-(-inf) row with `m_j = where(m_j > -inf, m_j, 0.0)`, i.e. it
CLOBBERS M from -inf to 0.0.  Under SWA the tile-pruned loop
(tile_start = first_allowed_key // TILE_SIZE) can START on a 64-tile whose
low 32 columns are entirely out-of-window while M is still -inf; stock-32
never visits that half, so the clobber is NOT order-preserving whenever the
row's true running max stays below 0 (rare -> would pass a sampled parity
run and corrupt silently in prod: exactly the failure class we must never
ship).  SWA prefill is O(window=129) per query — the deep-prefill win lives
in the full-attn layers — so SWA layers simply keep the certified A1
tile-32 path.

NON-SWA SAFETY ARGUMENT (why bounds at TILE=64 are exact): tile_start=0 and
max_seq_prefix_len is a TILE-independent token count, so the 64-loop covers
a superset of stock-32's columns whose only extras are a trailing all-masked
half.  By then M is finite for every valid row (causal -> column 0 is always
unmasked in subtile 0), so that half contributes alpha == exp(0) == 1.0
exactly and P == 0 exactly: value-neutral.  Padding rows are store-masked.

BLOCK-TABLE WIDTH GUARD: the kernel's block-index gather is UNMASKED; at
TILE=64 it can read ceil(64/BLOCK_SIZE)-1 entries past a row's last block.
The wrapper only enables tile-64 when block_table.shape[1]*block_size is a
multiple of 64 (prod mml 163840 -> OK), else it downgrades to tile 32.

HUNKS (all-or-nothing; every anchor exact-count-checked before any write):
  S  kernel signature: add `SPLIT_TILE: tl.constexpr = False` (trailing
     default -> every existing call site, incl. the 3D/decode path and the
     parity harness's direct launches, is untouched; the False branch is
     dead-coded at the Triton frontend -> byte-equivalent compute to today).
  B  loop compute block (seq_mask..PV-acc) -> `if SPLIT_TILE:` split
     implementation `else:` the ORIGINAL block re-indented verbatim (the
     else-branch is constructed FROM the matched text, so stock text drift
     is impossible by construction).
  A  env alias: "safe64" -> "32,4,2,64" inside the A1 tune parse line.
  W  wrapper gate after the A1 tune block: tile 64 -> SPLIT_TILE=True via
     _pf_launch_kwargs (raw joint-64 compute becomes UNREACHABLE from the
     env); SWA / misaligned block table / tile>64 -> downgrade to tile 32.

ORDERING: must run AFTER mods/diffkv-prefill-tune (checked via its marker).
Default env (unset -> "32,4,2,32") stays byte-for-byte no-op vs today: tile
!= 64 -> wrapper hunk inert, SPLIT_TILE never passed -> constexpr default
False -> frontend-dead branch.

Discipline (per diffkv-kernel-bw / prefill-tune): marker-guarded idempotency,
exact-count anchors, ast.parse after edit, sys.exit(3) on any anchor miss.

Usage: patch_tile64_safe.py <path-to-kernel.py>
"""
import ast
import pathlib
import sys

MARKER = "DIFFKV_TILE64_SAFE"
PREREQ_MARKER = "DIFFKV_PREFILL_TUNE"

# ---- Hunk S: kernel signature ---------------------------------------------
SIG_ANCHOR = "    V_DESCALE: tl.constexpr = 1.0,\n"
SIG_INSERT = """\
    # DIFFKV_TILE64_SAFE (A2): order-preserving TILE=64 subtiling. False
    # (default) -> the split branch is dead-coded at the Triton frontend:
    # byte-equivalent compute to the unpatched kernel for every existing
    # caller (decode/3D/q<=8 paths never set it).
    SPLIT_TILE: tl.constexpr = False,
"""

# ---- Hunk B: loop compute block --------------------------------------------
# EXACT text of the live kernel's per-tile compute section, from the seq-mask
# build through the PV accumulation (the end of the tile loop body).  The
# K/V load + fp8 descale region sits ABOVE this and is deliberately NOT
# touched (fp8-kv-inline owns it; at TILE_SIZE=64 those loads are already
# 64-wide via offs_t).
BODY_ANCHOR = """\
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            None,  # mm_prefix_range_ptr
            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        acc += tl.dot(P.to(V.dtype), V)
"""

SPLIT_IMPL = """\
        if SPLIT_TILE:
            # DIFFKV_TILE64_SAFE (A2): TILE=64 on the MEMORY side (the K/V
            # loads + block-table gather above are 64 wide), stock TILE=32
            # order on the COMPUTE side.  Raw joint-64 compute failed strict
            # parity (rel 3-7e-3, big%=0.00): one softmax rescale per 64
            # cols instead of two per 32, the bf16 P-cast after a 64-wide
            # max, and one 64-long PV accumulation reorder the fp32 sums.
            # Splitting the loaded tiles into their two 32-column halves and
            # replaying the UNMODIFIED stock sequence (S -> softmax_step ->
            # P-cast -> PV-acc) low half first makes iteration j perform
            # exactly the columns, ops and order of stock TILE=32
            # iterations 2j and 2j+1 -> bit-identical by construction.
            tl.static_assert(TILE_SIZE == 64, "SPLIT_TILE requires TILE_SIZE == 64")
            # SWA is order-UNSAFE here: its pruned loop (tile_start =
            # first_allowed_key // TILE_SIZE) can start on an all-masked low
            # half while M == -inf, and softmax_step's guard clobbers M
            # (-inf -> 0.0) -- not order-preserving when a row's true
            # running max stays below 0.  The launcher downgrades SWA
            # layers to the certified tile-32 path; this assert makes any
            # other caller fail LOUDLY at compile instead of corrupting.
            tl.static_assert(
                SLIDING_WINDOW == 0,
                "SPLIT_TILE is order-unsafe under SWA -- launcher must "
                "downgrade sliding-window layers to TILE_SIZE=32",
            )
            HALF_TILE: tl.constexpr = TILE_SIZE // 2
            # (HQK_P, 64) -> (HQK_P, 2, 32) -> (HQK_P, 32, 2) -> two halves.
            # Pure data movement: values already carry the fp8 descale.
            K_lo, K_hi = tl.split(
                tl.permute(
                    tl.reshape(K, (HEAD_SIZE_QK_PADDED, 2, HALF_TILE)), (0, 2, 1)
                )
            )
            # (64, HV_P) -> (2, 32, HV_P) -> (32, HV_P, 2) -> two halves.
            V_lo, V_hi = tl.split(
                tl.permute(
                    tl.reshape(V, (2, HALF_TILE, HEAD_SIZE_V_PADDED)), (1, 2, 0)
                )
            )
            query_abs_pos = context_len + query_pos[:, None]
            offs_ht = tl.arange(0, HALF_TILE)

            # ---- low half: verbatim stock TILE=32 sequence (columns of
            # stock iteration 2j) ----
            seq_offset_h = j * TILE_SIZE + offs_ht
            seq_mask_h = compute_kv_seq_mask(
                query_abs_pos,
                seq_offset_h,
                seq_idx,
                seq_len,
                None,  # mm_prefix_range_ptr
                SLIDING_WINDOW,
                False,  # USE_MM_PREFIX
                0,  # MAX_MM_RANGES
            )
            S = tl.zeros(shape=(BLOCK_M, HALF_TILE), dtype=tl.float32)
            S += scale * tl.dot(Q, K_lo)
            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)
            S = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None] & seq_mask_h,
                S,
                float("-inf"),
            )
            if USE_ALIBI_SLOPES:
                S = apply_alibi_to_score(
                    S, alibi_slope, seq_offset_h, context_len, query_pos,
                    USE_ALIBI_SQRT,
                )
            M, L, P, alpha = softmax_step(S, M, L)
            acc = acc * alpha[:, None]
            acc += tl.dot(P.to(V_lo.dtype), V_lo)

            # ---- high half: columns of stock iteration 2j+1 ----
            seq_offset_h = seq_offset_h + HALF_TILE
            seq_mask_h = compute_kv_seq_mask(
                query_abs_pos,
                seq_offset_h,
                seq_idx,
                seq_len,
                None,  # mm_prefix_range_ptr
                SLIDING_WINDOW,
                False,  # USE_MM_PREFIX
                0,  # MAX_MM_RANGES
            )
            S = tl.zeros(shape=(BLOCK_M, HALF_TILE), dtype=tl.float32)
            S += scale * tl.dot(Q, K_hi)
            if USE_SOFTCAP:
                S = apply_softcap(S, softcap)
            S = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None] & seq_mask_h,
                S,
                float("-inf"),
            )
            if USE_ALIBI_SLOPES:
                S = apply_alibi_to_score(
                    S, alibi_slope, seq_offset_h, context_len, query_pos,
                    USE_ALIBI_SQRT,
                )
            M, L, P, alpha = softmax_step(S, M, L)
            acc = acc * alpha[:, None]
            acc += tl.dot(P.to(V_hi.dtype), V_hi)
        else:
"""

# ---- Hunk A: env alias "safe64" inside the A1 tune parse ------------------
ALIAS_OLD = (
    '                    ("32,8,2,32" if _pf_env == "1" else _pf_env)'
    '.split(","))\n'
)
ALIAS_NEW = (
    '                    ("32,8,2,32" if _pf_env == "1" else\n'
    '                     # DIFFKV_TILE64_SAFE alias: order-preserving 64\n'
    '                     "32,4,2,64" if _pf_env == "safe64" else\n'
    '                     _pf_env).split(","))\n'
)

# ---- Hunk W: wrapper gate --------------------------------------------------
GRID_ANCHOR = "    grid: tuple[Any, ...]\n"
WRAPPER_BLOCK = """\
    # ------------------------------------------------------------------
    # DIFFKV_TILE64_SAFE (A2): order-preserving TILE=64 prefill subtiling.
    # tile=64 from the A1 tune widens only the MEMORY side (one K 256x64
    # + one V 64x128 load + one block-table gather per 64 tokens); the
    # kernel's SPLIT_TILE branch replays the stock 32-wide compute
    # sequence twice per tile in stock order -> accumulation order is
    # identical by construction.  Raw joint-64 compute failed strict
    # parity (rel 3-7e-3, softmax/bf16-P-cast/PV reordering) and is NOT
    # reachable via env anymore: tile=64 ALWAYS takes the split path.
    #   * SWA layers (sliding_window_val != 0) downgrade to the certified
    #     A1 tile-32 path: SWA tile pruning can start the loop on an
    #     all-masked 32-half while M == -inf, whose guarded softmax_step
    #     (m_j: -inf -> 0.0) is NOT order-preserving.  SWA prefill is
    #     O(window) anyway -- the win lives in the full-attn layers.
    #   * block-table width guard: the kernel's block-index gather is
    #     UNMASKED and at TILE=64 may read ceil(64/block)-1 entries past
    #     a row's last block; require 64-token-aligned table rows
    #     (shape[1]*block_size % 64 == 0; prod mml 163840 -> OK).
    #   * tile > 64 (e.g. env "...,128") downgrades to 32: joint-wide
    #     compute is the known-bad reorder class -- only parity-certified
    #     paths may be reachable from the env.
    # ------------------------------------------------------------------
    if _pf_launch_kwargs:
        if tile_size > 64:
            logger.warning_once(
                "[tile64-safe] tile=%d rejected (only the order-preserving "
                "64-split path is certified above 32) -- using tile 32",
                tile_size)
            tile_size = 32
        elif tile_size == 64:
            if sliding_window_val != 0:
                tile_size = 32  # SWA: certified A1 tile-32 path (see above)
            elif (block_table.shape[1] * block_size) % 64 != 0:
                logger.warning_once(
                    "[tile64-safe] block table row not 64-token aligned "
                    "(%d cols x block %d) -- using tile 32",
                    block_table.shape[1], block_size)
                tile_size = 32
            else:
                _pf_launch_kwargs["SPLIT_TILE"] = True
"""


def _indent(block: str, pad: str = "    ") -> str:
    """Indent every non-blank line; keep blank lines byte-identical."""
    return "\n".join(
        (pad + ln) if ln.strip() else ln for ln in block.split("\n")
    )


def main(path: str) -> None:
    p = pathlib.Path(path)
    src = p.read_text()

    if MARKER in src:
        print(f"[tile64-safe] already applied to {p} -- no-op")
        return

    if PREREQ_MARKER not in src:
        print(f"[tile64-safe] FATAL: {p} lacks the {PREREQ_MARKER} marker -- "
              "mods/diffkv-prefill-tune must be applied FIRST (this patch "
              "extends its launch gate)")
        sys.exit(3)

    if "def unified_attention_diffkv(" not in src:
        print(f"[tile64-safe] FATAL: {p} has no unified_attention_diffkv")
        sys.exit(3)

    # ---- verify EVERY anchor before touching anything (atomicity) ----
    anchors = [
        ("signature V_DESCALE", SIG_ANCHOR, 1),
        ("loop compute block", BODY_ANCHOR, 1),
        ("A1 alias parse line", ALIAS_OLD, 1),
        ("grid decl", GRID_ANCHOR, 1),
    ]
    for name, text, want in anchors:
        got = src.count(text)
        if got != want:
            print(f"[tile64-safe] FATAL: anchor '{name}' count = {got} "
                  f"(expected {want}) -- kernel text drifted, refusing to "
                  "patch")
            sys.exit(3)

    # Hunk S: signature param (after V_DESCALE, before the closing paren).
    src = src.replace(SIG_ANCHOR, SIG_ANCHOR + SIG_INSERT, 1)

    # Hunk B: gate the compute block.  The else-branch is the MATCHED text
    # re-indented by 4 -> stock arithmetic is preserved verbatim by
    # construction (frontend dead-codes the split branch when SPLIT_TILE
    # is False).
    replacement = SPLIT_IMPL + _indent(BODY_ANCHOR)
    src = src.replace(BODY_ANCHOR, replacement, 1)

    # Hunk A: "safe64" env alias.
    src = src.replace(ALIAS_OLD, ALIAS_NEW, 1)

    # Hunk W: wrapper gate between the A1 tune block and the grid decl.
    src = src.replace(GRID_ANCHOR, WRAPPER_BLOCK + "\n" + GRID_ANCHOR, 1)

    ast.parse(src)  # loud on any syntax slip
    p.write_text(src)
    print(f"[tile64-safe] applied to {p} (SPLIT_TILE param + split branch + "
          "safe64 alias + wrapper gate; tile=64 now always order-preserving, "
          "SWA/misaligned-table/tile>64 downgrade to 32)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
