# Revised tangent-controller training selection

The user wants native Isaac training that fixes cone contacts and timeouts, with the same auditable debug videos. This is an exploratory 500-update run of the full model. A selected checkpoint earns physical evaluation only. It is not a deployment or task-success claim.

## Decision before updates

Run `native_tangent_trial2.yaml` after the frozen zero-update calibration and an end-to-end pilot pass. Use the full 31,154,802-parameter initializer from vision trial 1, SHA `e59efc8ace2d05a5ecfc29023712e7b5ce1d34a64133dd869a7bc17f849bc35f`. Train only its policy parameters with frozen vision, FP32, 128 draws per update, 8-frame history, plan weight 1, steering weight 3.281158240898283 and longitudinal weight 1. The optimizer, controller, data, validation anchors and source bytes are bound by the existing calibration and trainer checks.

Recovery collection used checkpoint `1ac2e3ba949db39ab445c636f841360973e2add8c86edab689a3a872f040ce55`. It is explicitly off-policy relative to this initializer. Each batch draws 16 recovery and 112 other examples. The 8,000 recovery draws in this run resample 80 distinct anchors; they are not 8,000 independent recoveries.

Select among steps 0, 100, 200, 300, 400 and 500. Step 1 is retained and verified but excluded from selection. Require all global and per-case bounds below, unchanged native target coherence, unchanged non-trainable state, and strictly improved aggregate steering loss against step zero. Among eligible trained checkpoints, take the lowest steering loss and the earliest step on a tie. If none qualifies, retain step zero and state that no trained candidate was selected. The receipt also reports whether that fallback itself meets the bounds.

Draft 1 was rejected before any updates because opposite-sign endpoint accelerations could cancel in the trapezoidal metrics. The separate draft-1 rejection record preserves that finding. This revision adds endpoint/secant checks and binds a new calibration. The first actual optimization run remains named `tangent_trial1`.

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
| Endpoint velocity minus position secant | 1.00 m/s |
| Endpoint acceleration minus velocity secant | Mean 1.00, p95 2.00, maximum 4.00 m/s² |
| Endpoint yaw rate minus yaw secant | 0.25 rad/s |

The position caps target the observed internal plan notch. The initializer's worst interval and cumulative position residuals are 0.0910571 m and 0.1648455 m. The native targets are 0.0233271 m and 0.0760882 m. The 0.10 m and 0.20 m caps retain the measured initializer but are far below the roughly 0.7 m inconsistency in the rejected action-trained models. They are plan consistency limits, not proven cone-clearance margins.

Endpoint/secant checks compare each endpoint derivative separately with the corresponding interval finite difference, then take the worst endpoint and interval for each anchor. They cover the first 0.9765625 seconds, including the controller’s 0.2-second steering and 0.3-second longitudinal interpolation intervals. This removes the identified opposite-endpoint cancellation in the integral checks. It is not a complete proof of plan physicality beyond these measured relationships and horizon.

The new calibration exposes an initializer defect. Its acceleration/secant mismatch has global mean 3.7897568, p95 6.7092464 and maximum 8.5597805 m/s². The native targets are 0.3084191, 0.8901886 and 2.8961862 m/s². The proposed ceilings of 1, 2 and 4 m/s² allow measured native discretization but require a substantial reduction from the initializer. They were not widened to admit step zero. Step zero passes 432 of 459 gates and fails all 27 acceleration/secant gates, the three statistics globally and in each of eight episodes. A fallback to this step is explicitly ineligible and means no promoted model.

The native maximum velocity/secant and yaw-rate/secant mismatches are 0.6353579 m/s and 0.1933466 rad/s. The corresponding initializer maxima are 0.9296807 m/s and 0.1594386 rad/s. The 1.0 m/s and 0.25 rad/s caps constrain their growth while accommodating the measured native targets. The acceleration caps also reject the reviewer’s ±20 m/s² alternating-knot counterexample, which the original 13 metrics reported as perfectly consistent.

The initializer already has a cumulative velocity residual as large as 2.0350015 m/s. The 2.25 m/s cap is a retention ceiling for this experiment, not evidence that its acceleration prediction is physically accurate. Native target maximum is 0.2287194 m/s. Initializer/native cumulative yaw maxima are 0.0365943/0.0885628 rad, so the native-target floor materially controls some yaw bounds. These weaknesses remain visible in reports rather than being reclassified as negligible.

## Rejected-run diagnostic policy

If no trained checkpoint passes all selection gates, no checkpoint is promoted and no fresh test is opened. The final scheduled step, exactly step 500, may be evaluated on the two already exposed heavy track and skidpad cases solely to diagnose the rejected model and record failure videos. It must be labeled rejected in the plan, report and videos. It is not a best-checkpoint choice, and no different failed checkpoint may be substituted after viewing results. The selector receipt remains authoritative and continues to say no trained candidate was selected. A new fix or protocol requires a separately declared experiment.

## Evidence and next decision

`calibration.json` SHA is `94f21a4d3741611578f38f83242ae8b9c7222eb65fff6b31db1188f1f6205300`. The numerical protocol is `tangent_selection2.json`, SHA `f1ce3776281955b0a1b899a201089cea8b099b24c89c9959caa84468f5fb17ba`. The training config SHA is `369a8bd944fda75dfb55463cde5431dece492a1e1eb36e36030176a09f501bac`.

The trainer checks calibration identity and exact validation values before any update, then rechecks the protocol and calibration bytes during training. The selector independently checks all 500 updates, scheduled checkpoint provenance, actual data identity, model state, sampling receipts and every bound. Its source and the report source must also be frozen before launch.

The Explain the Number principle changes the decision here. Lower offline steering loss is only an admission gate. Native camera/physics rollouts must establish contact counts, completion, timeout behavior and failure videos. A comparison to the old PP controller measures the model/controller package. A separate same-controller initializer comparison is needed to attribute gains to training. No threshold may be loosened after this run starts to promote a failed candidate.
