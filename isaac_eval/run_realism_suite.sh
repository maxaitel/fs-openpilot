#!/usr/bin/env bash
set -euo pipefail
cd /work
root=isaac_eval/runs/realism_v1
mkdir "$root"
cp isaac_eval/protocol_realism.json "$root/protocol.json"
for discipline in trackdrive skidpad; do
  for seed in 91000 91001 91002 91003 91004; do
    variants="plain paddock"
    if (( seed % 2 )); then variants="paddock plain"; fi
    for scene in $variants; do
      name="${scene}_${discipline}_${seed}"
      /isaac-sim/python.sh isaac_eval/evaluate.py --discipline "$discipline" --seed "$seed" --scene "$scene" --protocol "$root/protocol.json" --out "$root/$name" > "$root/$name.log" 2>&1
      test -s "$root/$name/summary.json" || { echo "FAILED missing summary: $name"; exit 1; }
      echo "COMPLETED $name"
    done
  done
done
