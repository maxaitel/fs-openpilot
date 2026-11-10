#!/bin/sh
set -eu
out=/scratch/fsop/pi_step_size_probe1
control=/scratch/fsop/pi_update_probe2
cd /work
export PYTHONPATH=/work:/work/src:/work/isaac_eval
/isaac-sim/python.sh "$out/probe.py" --out "$out" --control "$control" > "$out/run.log" 2>&1 &
pid=$!
set +e
wait "$pid"
status=$?
set -e
printf '%s\n' "$status" > "$out/exit_code.txt"
[ "$status" -eq 0 ]
/isaac-sim/python.sh "$out/verify.py" --out "$out" --control "$control" --script "$out/probe.py" --pid "$pid"
