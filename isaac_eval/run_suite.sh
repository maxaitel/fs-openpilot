#!/usr/bin/env bash
set -euo pipefail
cd /work
mkdir -p isaac_eval/runs/suite_v2
cp isaac_eval/protocol.json isaac_eval/runs/suite_v2/protocol.json
for discipline in trackdrive skidpad; do
  for seed in 91000 91001 91002 91003 91004; do
    name="${discipline}_${seed}"
    /isaac-sim/python.sh isaac_eval/evaluate.py --discipline "$discipline" --seed "$seed" --out "isaac_eval/runs/suite_v2/$name" > "isaac_eval/runs/suite_v2/$name.log" 2>&1
    echo "COMPLETED $name"
  done
done
