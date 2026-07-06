#!/bin/bash
# Path B: make Ray compiled-DAG accelerator_context tolerant of unset CUDA_VISIBLE_DEVICES.
# vLLM sets RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 -> visible list is empty ->
# [].index('0') -> ValueError -> RuntimeError "CUDA_VISIBLE_DEVICES set incorrectly".
# One GPU per node, so the Ray-logical accelerator id maps 1:1 to ordinal index.
set -e
python3 - <<'PYEOF'
import re, ast
p = "/usr/local/lib/python3.12/dist-packages/ray/experimental/channel/accelerator_context.py"
s = open(p).read()
if "VLLM_CVD_FALLBACK" in s:
    print("[ray-cvd-fallback] already patched"); raise SystemExit(0)
lines = s.split("\n"); out = []; i = 0; patched = False
while i < len(lines):
    ln = lines[i]
    m = re.match(r'^(\s*)for accelerator_id in accelerator_ids:\s*$', ln)
    if m and not patched:
        ind = m.group(1)
        out.append(f"{ind}for _vllm_idx, accelerator_id in enumerate(accelerator_ids):  # VLLM_CVD_FALLBACK")
        i += 1
        while i < len(lines):
            l2 = lines[i]
            if 'raise RuntimeError(' in l2:
                rind = re.match(r'^(\s*)', l2).group(1)
                out.append(f"{rind}device_ids.append(_vllm_idx)")
                depth = l2.count('(') - l2.count(')'); i += 1
                while i < len(lines) and depth > 0:
                    depth += lines[i].count('(') - lines[i].count(')'); i += 1
                break
            else:
                out.append(l2); i += 1
        patched = True
        continue
    out.append(ln); i += 1
if not patched:
    print("[ray-cvd-fallback] ANCHOR NOT FOUND"); raise SystemExit(1)
new = "\n".join(out)
ast.parse(new)
open(p, "w").write(new)
print("[ray-cvd-fallback] patched + ast OK")
PYEOF
echo "[ray-cvd-fallback] done"
