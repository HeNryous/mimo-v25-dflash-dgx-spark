#!/bin/bash
set -uo pipefail
# ============================================================================
# omni-eagle3 — add SupportsEagle3 to MiMoV2OmniForCausalLM (upstream image)
#
# #46104 added SupportsEagle3/EagleModelMixin to MiMoV2FlashForCausalLM only.
# The recipe loads MiMoV2OmniForCausalLM (via --hf-overrides), whose MRO does
# NOT include SupportsEagle3 (verified in this image: MRO lacks it, no
# set_aux_hidden_state_layers). DFlash spec-decode needs the target to expose
# the EAGLE3 aux-hidden-state interface. Same 2-line sed pattern the
# dflash-graft mod used successfully on dev114.
#
# Anchors verified in vllm-node-mimo-v25-upstream:latest:
#   mimo_v2_omni.py:55    "    SupportsQuant,"                       (import)
#   mimo_v2_omni.py:1159  "class MiMoV2OmniForCausalLM(nn.Module,   (bases)
#                          SupportsMultiModal, SupportsPP, SupportsQuant):"
# The Flash class already exposes it, so the Protocol delegates through the
# shared language_model.model (same as dflash-graft relied on).
# ============================================================================
SITE=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
OMNI="$SITE/model_executor/models/mimo_v2_omni.py"

if grep -q "SupportsEagle3" "$OMNI"; then
  echo "[omni-eagle3] SupportsEagle3 already present — nothing to do"
else
  before_import=$(grep -c "SupportsEagle3" "$OMNI" || true)
  # 1) add to the interfaces import (insert before SupportsQuant import line)
  sed -i 's/^    SupportsQuant,$/    SupportsEagle3,\n    SupportsQuant,/' "$OMNI"
  # 2) add to the class bases
  sed -i 's/^class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant):$/class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant, SupportsEagle3):/' "$OMNI"

  # verify BOTH edits landed
  n_import=$(grep -c "^    SupportsEagle3,$" "$OMNI" || true)
  n_class=$(grep -c "^class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant, SupportsEagle3):$" "$OMNI" || true)
  if [ "$n_import" != "1" ] || [ "$n_class" != "1" ]; then
    echo "[omni-eagle3] FATAL: sed did not land both edits (import=$n_import class=$n_class) — anchor drift"
    exit 3
  fi
  echo "[omni-eagle3] SupportsEagle3 added to import + class bases"
fi

find "$SITE/model_executor/models" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PYEOF'
import ast, sys
f = None
import vllm, os
f = os.path.dirname(vllm.__file__) + "/model_executor/models/mimo_v2_omni.py"
ast.parse(open(f).read())
import vllm.model_executor.models.mimo_v2_omni as m
cls = m.MiMoV2OmniForCausalLM
mro = [c.__name__ for c in cls.__mro__]
assert 'SupportsEagle3' in mro, f"SupportsEagle3 NOT in MRO: {mro}"
assert hasattr(cls, 'set_aux_hidden_state_layers'), "set_aux_hidden_state_layers missing after patch"
print("[omni-eagle3] verified: SupportsEagle3 in MRO + set_aux_hidden_state_layers present")
PYEOF
[ $? -eq 0 ] || { echo "[omni-eagle3] VALIDATION FAILED — aborting"; exit 1; }
echo "[omni-eagle3] done"
