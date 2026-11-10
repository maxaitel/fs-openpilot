# Skidpad correction experiments

The goal is fewer skidpad cone contacts and valid full missions after native Isaac Lab training, with unchanged evaluation rules and the same recorded vision/debug evidence. Offline improvement is not a driving result.

The branch separates physical desired motion from PI actuator compensation, collects qualified entry and left-transition recoveries, and runs two fixed 500-update recipes on Spark. The first full-policy trial regressed. The second restricts optimization to the existing final plan projection and directly supervises deployed steering. Neither trial produced a checkpoint that passed the frozen qualification rules. All failed checkpoints and setup attempts remain available. No checkpoint is promoted, and fresh test cases remain sealed.

## Completed training

| Trial | Optimization scope | Wall seconds | Median whole-GPU utilization | Validation steering SmoothL1, initial to final | Qualification |
|---|---|---:|---:|---|---|
| PI consistency, full policy | 6,949,412 eligible across 50 tensors; 42 receive loss gradients | 479.69 | 86% | 0.00125601 to 0.00140550 | All scheduled checkpoints rejected |
| Direct steering, final plan head | 254,430 across 2 tensors | 693.98 | 75% | 0.00125601 to 0.00124575 | All scheduled checkpoints rejected |

The complete 31,154,802-parameter model stays loaded. Vision and auxiliary cone heads remain unchanged. The second trial preserves every other state tensor, including the shared policy and final output scale. Its complete-model checkpoint is `dd50806f3ae781e19118a74a2d47d6fd4b09e4e3aa7ba370a72559c86671c5c7`.

The head trial reduced consistency loss by 24.7% and steering loss by 0.82%. Longitudinal validation loss rose by 0.69%. Final mean, p95 and maximum acceleration-secant errors were 1.0162, 4.8054 and 7.1254 m/s², exceeding the fixed 1, 2 and 4 m/s² ceilings. Two per-scene steering ceilings and other coherence limits also failed. Lower aggregate losses do not override these failures.

These timing measurements do not isolate an optimizer-scope performance effect. Several unrelated simpleFoam jobs occupied host CPUs throughout the second run. Its whole-host memory includes those jobs and the RAM-backed output filesystem. PyTorch peak allocation was 1,518,202,880 bytes. GPU utilization measures busy time, not achieved FLOPS.

## Diagnosis and decisions

- [Correction record](../../docs/isaac-lab-training/pi_correction1.md) explains the motion/actuator distinction, recovery collection and first paired failure.
- [Native steering reference](native_lateral_reference/) replays exact recorded controls and privileged native futures. OP plus PI completes the route with one transition contact; this is not neural-model performance.
- [Forty-eight independent optimizer updates](pi_update_probe/report_v2.md) show fitting overshoot and objective conflicts. Invalid CPU FP32 parameter norms and cosines are excluded from interpretation.
- [Learning-rate experiment](pi_step_size_probe/report.md) removes immediate fitting overshoot without removing monitored steering regression.
- [Final-head experiment](pi_head_scope_probe/report.md) reduces, but does not eliminate, that regression. All eight updates retain the exact starting predictions and freeze all other 415 state tensors.
- [Forward arithmetic check](head_forward_parity/report.md) explains the first aborted head probe. Original tolerances were not relaxed.
- [Frozen head-training decision](head_training_decision.md) fixes the loss, scope and learning rate before calibration and training. There are no coefficient retries.
- [Setup record](head_execution_setup.md) distinguishes zero-update launch failures and pre-driving export failures from physical results.

The existing cone heads are debug outputs; they do not directly control avoidance. Current consistency supervision also does not directly constrain steering or uniquely determine target speed. The new steering loss supervises the controller's signed combination of predicted yaw and yaw rate while preserving physical native targets.

## Evidence

The `training/` and `head_training/` folders hold full histories, calibrations, source archives, selection protocols and rejection receipts. `development/` contains the first completed paired evaluation. `vision_pi_development1/` holds the separate failed earlier-initializer diagnostic. `head_pilot/` records the passed complete 120-frame saved-input inference replay and controller/artifact audit for the second trial.

First-trial videos and graphs are linked from `runs/skidpad_correction_report/index.html`. The second paired physical evaluation finished. Both models completed track and failed skidpad. All eight debug and research videos passed producer artifact audits, exact archive copying, and full decoding. The local report is `runs/skidpad_head_correction_report/index.html`. [Training graphs](head_training.png) and [paired outcome graphs](head_paired.png) are also available here. The full 5.96 GB raw evaluation archive and 1.27 GB training archive are verified locally.

Only the two explicitly approved redundant Spark copies of the previous heavy track and skidpad evaluations were removed. Their complete verified local archives and cleanup receipts remain. No other experiment data was deleted.

The complete implementation suite passed 443 tests, with four skips and seven subtests. Independent review passed 79 focused tests and checked actual serialized AdamW state. Attack the Premise shaped the successive experiments; Explain the Number kept fit, validation, physical outcomes and timing separate; Prove It Works required actual checkpoint, inference and video verification.


## Second paired driving comparison

| Case | Model | Result | Simulated seconds | Physical contacts | Off-course events |
|---|---|---|---:|---:|---:|
| Skidpad | Baseline | Standstill DNF after R,R | 76.90 | 21 | 5 |
| Skidpad | Final head diagnostic | Rollover after R,R | 32.95 | 13 | 1 |
| Track | Baseline | Finished | 56.25 | 2 | 1 |
| Track | Final head diagnostic | Finished | 57.05 | 2 | 1 |

These are one run per model per exposed case. Lower candidate skidpad contacts accompany much earlier failure and do not establish improvement. Track contact counts are unchanged. [The failure analysis](head_failure_trace/README.md) and [trajectory plot](head_failure_trace/trajectory.png) show missed timing gates, physical contacts, and terminal states from the saved records.
