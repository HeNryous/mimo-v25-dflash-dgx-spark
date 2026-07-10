#!/bin/bash
# Param-sweep P0+P1 on final prod config (night 2026-07-10). Sequential, ~2-4h.
cd /home/admin/param-sweep
B=/home/admin/.local/bin/tool-eval-bench
URL=http://127.0.0.1:8000
M=mimo-dflash-test
run() {  # run <tag> <extra args...>
  tag=$1; shift
  echo "$(date +%H:%M:%S) START $tag" >> sweep.log
  $B --model $M --backend vllm --base-url $URL --json --json-file $tag.json "$@" >> sweep.log 2>&1
  s=$(python3 -c "import json;print(json.load(open(\"$tag.json\"))[\"final_score\"])" 2>/dev/null)
  echo "$(date +%H:%M:%S) DONE $tag score=$s" >> sweep.log
}
run p0-1 --temperature 0.0
run p0-2 --temperature 0.0
run p0-3 --temperature 0.0
run t03-1 --temperature 0.3 --top-p 0.95
run t03-2 --temperature 0.3 --top-p 0.95
run t07-1 --temperature 0.7 --top-p 0.95
run t07-2 --temperature 0.7 --top-p 0.95
run t10-1 --temperature 1.0 --top-p 0.95
run t10-2 --temperature 1.0 --top-p 0.95
echo "$(date +%H:%M:%S) SWEEP COMPLETE" >> sweep.log
