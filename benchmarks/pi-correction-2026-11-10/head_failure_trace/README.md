# Why the two head-trial skidpad runs failed

The goal is completed skidpad missions with fewer physical cone contacts. These two exposed runs do not establish a fix. Both accepted only the two right laps. The baseline stopped outside the exit lane after missing the left return gates. The candidate terminated with a reported rollover before its first left return crossing.

[The trajectory plot](trajectory.png) uses the saved car positions, initial cone positions, timing gate, and stop-zone polygon. It shows each first physical contact at the car position before the contact interval. It does not reconstruct cone motion after impact.

## The baseline missed the finite gate

Both saved realizations bind the same canonical track hash. The existing archived `track.json` has that exact hash, so this analysis uses its geometry. Its timing gate lies at x = 0 between y = -1.5 m and +1.5 m.

Linear interpolation of the saved 20 Hz positions reproduces all three recorded accepted crossing times within 1e-10 seconds. The later forward crossings miss the gate.

| Crossing | Time, simulated seconds | World y, metres | Distance beyond the gate end, metres |
|---|---:|---:|---:|
| First left return | 33.123901 | 2.595670 | 1.095670 |
| Second left return | 43.089970 | 2.677646 | 1.177646 |

The baseline drove through the left-circle area twice, but neither return crossed the timing gate. No left lap was accepted. Its path also moved outside the outer cone boundary and then inside the inner boundary. The failure is broader than consistent inward cutting.

Of its 21 physical contacts, three occurred before the second right lap finished and 18 occurred afterward. The legacy cone-displacement score is 19. The physical-contact count is the relevant number here.

## Projected progress reached the end without a valid finish

The baseline first reached the projected mission endpoint at 46.50 simulated seconds. Its position was x = 22.642 m, y = 3.170 m. The saved projected progress ended at 1.001749 of the mission endpoint, while the accepted route remained R,R.

The largest recorded progress increment was 0.457512 m in one 0.05-second interval, with 0.309556 m of physical travel. There is no lap-sized progress jump in this trace. The evidence supports continued nearest-path projection through physically driven loops that failed timing-gate qualification.

The source uses projected progress for terminal braking. At the final state, remaining distance is below 0.5 m and forward speed is below 0.3 m/s. That source rule limits acceleration to at most -1 m/s², matching the reported final acceleration. The compact inputs do not include every controller diagnostic, so this is a source-derived explanation of the final command rather than a replay of each command.

The continuous saved stationary tail begins at 46.85 seconds. The run terminates with standstill DNF at 76.90 seconds. The final position is x = 23.101 m, y = 3.149 m, outside the stop-zone width of ±1.5 m. Route `valid=true` means no explicit route violation was raised. It does not mean the route completed. The scorer correctly reports incomplete and DNF.

Changing the terminal brake condition would not recover the missed gates, remove the earlier contacts, or demonstrate successful skidpad driving. This evidence does not justify calling such a guard a skidpad fix.

## The candidate failed earlier

The candidate terminated with `isaac_rollover` at 32.95 simulated seconds after 13 physical contacts and one off-course event. Four contacts occurred before the second right lap finished. Nine followed it. It did not reach a forward x = 0 crossing on its first left return before termination.

The trace shows contacts on the outer part of the left-circle approach and further contacts on the return. It does not isolate whether perception, predicted motion, tracking, or collision dynamics caused the final rollover. Attitude and per-frame plan diagnostics are outside these compact inputs.

The candidate had fewer contacts and 43.95 fewer simulated seconds of exposure. The count difference is not evidence of improved avoidance. There is one exposed run per model, and neither completed skidpad. The training qualification decision remains rejected.

## Reproduction and scope

Run `python3 benchmarks/pi-correction-2026-11-10/head_failure_trace/analyze.py` from the worktree. The script reads per-episode records and emits crossing, contact, and progress records grouped by model. It also generates the trajectory plot.

`trace.json` records every input hash and the script hash. `output_sha256.json` records the JSON and PNG hashes. The script verifies that the relevant local controller, mission, scorer, and cone-export sources exactly match the source archive from training. It verifies the track binding and reproduces the accepted crossing times from saved positions.

Prove It Works shaped the analysis by reproducing the recorded crossings from actual positions. Fix Root Causes shaped the conclusion by rejecting terminal braking as a demonstrated remedy for earlier route and contact failures. Model the Domain shaped the artifact as per-episode crossing, contact, and progress records.
