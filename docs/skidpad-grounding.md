# Baseline skidpad command and scoring flow

The goal is to make the learned policy complete skidpad by following route commands, then demonstrate that behavior in Spark evaluation and video.

## Overview

The command plumbing exists from simulation through recording, training, and inference. Skidpad is one unrolled path containing entry, two right circles, two left circles, and exit. The simulator uses progress along that path to produce labels and route commands. The learned policy receives camera inputs and command history. It does not receive the centerline as its predicted plan.

## Key concepts

The command has eight channels. Channels 1, 2, and 7 mean left, right, and exit. The policy receives `[25,8]` command history and `[25,512]` visual features. The visual input contains 24 historical features and the current feature. Each expert target contains 33 future samples of 15 values. Referee lap counts and mission progress are separate quantities.

## How the baseline worked

`FSEnv.route_desire` in `src/fsop/sim/env.py` checks distance to the starts of the right, left, and exit segments. It emits one pulse approximately two seconds ahead and mutates `desire_sent`. The minimum speed used for this calculation is 3 m/s. A slow car can therefore receive the command more than five seconds before the turn.

`record_episode` pools pulses across four simulation ticks and stores at 5 Hz. `SimDataset` constructs 25 command slots, 24 feature-history slots, and the expert target. `SupercomboFS.plan` appends the current visual feature and forwards the command tensor to `policy_raw`. Plan loss trains the policy but does not explicitly enforce a counterfactual response.

`TemporalState.push_desire` detects positive edges, stores 100 model ticks, and pools groups of four. Holding a command high does not refresh it. `ModelPolicy` converts the predicted plan into steering and acceleration. A separate mission layer launches the car and brakes near the finish using simulator progress.

## Measured evidence

The original dataset contains 40 skidpad episodes and 7,752 frames. Each episode contains exactly one pulse in each of channels 1, 2, and 7. The baseline checkpoint has 31 training skidpad episodes. On two held-out episodes, holding visual features fixed and injecting a latest-slot left versus right command changes the predicted trajectory. Mean absolute XY differences across the full horizon are 6.00 m and 5.38 m. This establishes sensitivity, not correct obedience. The fresh artificial pulses are a diagnostic intervention.

Twelve original FP32 ONNX versus converted PyTorch policy comparisons cover three command channels at four history ages. Maximum absolute plan difference is 0.00055. The nonzero path agrees numerically at these inputs. The graph reshapes all 25 command slots before its learned layers. Do not infer a nine-slot command window from the nine-slot visual-history comment.

The raw artifacts reside in `runs/spark_skidpad` on the Spark. They are `original-data-audit.json`, `original-command-ablation.json`, and `nonzero-command-parity.json`.

## Hazards confirmed before the repair

Calling `observe` twice can consume a command because it calls the stateful command generator. This has not been established as the cause of the original skidpad failures.

The referee accepts geometric timing-line intersections without direction or route validation. The earlier light failure recorded eleven laps despite only 7.3% mission progress. Its pass flag remains false. A new success claim must validate the physical right, right, left, left route and the final stop independently.

Timeouts terminate the environment while its referee can still report `running`. A final evaluation must report timeout explicitly.

## File map

- `src/fsop/sim/env.py` owns command generation and observations.
- `src/fsop/sim/recorder.py` owns stored command pulses.
- `src/fsop/data/sim_dataset.py` and `src/fsop/model/temporal.py` construct training and live histories.
- `src/fsop/model/build.py` and `src/fsop/model/supercombo.py` forward commands to the policy.
- `src/fsop/eval/closed_loop.py` runs learned control.
- `src/fsop/sim/referee.py` reports race outcomes.
- `src/fsop/track/fixed_layouts.py` defines skidpad geometry.

Two read-only explorers and a synthesis agent checked this flow. The parent ran the numerical probes. The selected repair is described in `skidpad-design.md`. It retains the learned planner and supplies repeated commands at each upcoming junction.
