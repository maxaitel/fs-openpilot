# First tangent-controller training selection

The user wants native Isaac training that fixes cone contacts and timeouts, with the same auditable debug videos. This is an exploratory 500-update run of the full model. A selected checkpoint earns physical evaluation only. It is not a deployment or task-success claim.

## Decision before updates

Run `native_tangent_trial1.yaml` after the frozen zero-update calibration and an end-to-end pilot pass. Use the full 31,154,802-parameter initializer from vision trial 1, SHA `e59efc8ace2d05a5ecfc29023712e7b5ce1d34a64133dd869a7bc17f849bc35f`. Train only its policy parameters with frozen vision, FP32, 128 draws per update, 8-frame history, plan weight 1, steering weight 3.281158240898283 and longitudinal weight 1. The optimizer, controller, data, validation anchors and source bytes are bound by the existing calibration and trainer checks.

Recovery collection used checkpoint `1ac2e3ba949db39ab445c636f841360973e2add8c86edab689a3a872f040ce55`. It is explicitly off-policy relative to this initializer. Each batch draws 16 recovery and 112 other examples. The 8,000 recovery draws in this run resample 80 distinct anchors; they are not 8,000 independent recoveries.

Select among steps 0, 100, 200, 300, 400 and 500. Step 1 is retained and verified but excluded from selection. Require all global and per-case bounds below, unchanged native target coherence, unchanged non-trainable state, and strictly improved aggregate steering loss against step zero. Among eligible trained checkpoints, take the lowest steering loss and the earliest step on a tie. If none qualifies, retain step zero and state that no trained candidate was selected. The receipt also reports whether that fallback itself meets the bounds.

## Numerical bounds

The calibration covers 256 fixed anchors from eight validation episodes, equally weighted by episode. It is one calibration, with no repeatability estimate. Bounds are engineering screening limits chosen before training; the 10% and 25% budgets are not measured noise estimates or statistical confidence intervals. Fresh test scenes remain unopened.

Aggregate plan L1 may rise by at most 5% from initialization, to 0.9319418359547854. Aggregate steering loss must not exceed 0.0014090503500483464, and selection additionally requires a strict improvement. Aggregate longitudinal loss must not exceed 0.5378836710005999. Each episode's three action metrics may rise by at most 10% from that same episode's initializer value. These per-episode limits prevent an average gain from hiding an unbounded regression in one scene. All are absolute, inclusive ceilings in the JSON protocol.

For each coherence metric and each of mean, p95 and maximum, both globally and per episode, the bound is the smaller of its hard cap below and `max(1e-6, 1.10 * initializer_stat, 1.25 * native_target_stat)`. The tiny floor admits numerical zero. The initializer allowance bounds degradation from the chosen starting model. The native-target allowance avoids rejecting a model merely for approaching the measured discretized native target, whose finite-knot residual is nonzero. Every statistic has its own ceiling; no case or tail is exempted.

| Metric | Hard cap, applied to every statistic |
|---|---:|
| Mean position interval residual per anchor | 0.04 m |
| Worst position interval residual per anchor | 0.10 m |
| Cumulative position residual | 0.20 m |
| Mean velocity interval residual per anchor | 0.30 m/s |
| Worst velocity interval residual per anchor | 0.80 m/s |
| Cumulative velocity residual | 2.25 m/s |
| Mean yaw interval residual per anchor | 0.012 rad |
| Worst yaw interval residual per anchor | 0.025 rad |
| Cumulative yaw residual | 0.12 rad |
| Initial position residual | 0.000001 m |
| Initial yaw residual | 0.000001 rad |
| Initial velocity residual | 1.50 m/s |
| Initial yaw-rate residual | 0.30 rad/s |

The position caps target the observed internal plan notch. The initializer's worst interval and cumulative position residuals are 0.0910571 m and 0.1648455 m. The native targets are 0.0233271 m and 0.0760882 m. The 0.10 m and 0.20 m caps retain the measured initializer but are far below the roughly 0.7 m inconsistency in the rejected action-trained models. They are plan consistency limits, not proven cone-clearance margins.

The initializer already has a cumulative velocity residual as large as 2.0350015 m/s. The 2.25 m/s cap is a retention ceiling for this experiment, not evidence that its acceleration prediction is physically accurate. Native target maximum is 0.2287194 m/s. Initializer/native cumulative yaw maxima are 0.0365943/0.0885628 rad, so the native-target floor materially controls some yaw bounds. These weaknesses remain visible in reports rather than being reclassified as negligible.

## Evidence and next decision

`calibration.json` SHA is `fb844328c2d5509fdc9af72dbde1c93bb4896969fb23e51e4c4ad50e417ca26b`. The numerical protocol is `tangent_selection1.json`, SHA `7183755ecfc559015d4af1e7fed08b93f9b0bcf326d8c0e051575619dc9fb5ff`. The training config SHA is `e33efc430c6539d6adfc733e5f48d6298c18ba23f4f76d4d51a2bfcb64bdeae8`.

The trainer checks calibration identity and exact validation values before any update, then rechecks the protocol and calibration bytes during training. The selector independently checks all 500 updates, scheduled checkpoint provenance, actual data identity, model state, sampling receipts and every bound. Its source and the report source must also be frozen before launch.

The Explain the Number principle changes the decision here. Lower offline steering loss is only an admission gate. Native camera/physics rollouts must establish contact counts, completion, timeout behavior and failure videos. A comparison to the old PP controller measures the model/controller package. A separate same-controller initializer comparison is needed to attribute gains to training. No threshold may be loosened after this run starts to promote a failed candidate.
