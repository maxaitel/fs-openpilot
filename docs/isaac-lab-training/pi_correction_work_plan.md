# Skidpad correction work plan

The goal is to reduce the 25-contact skidpad failure, explain why training regressed, retrain on the Spark, and deliver paired physical results with the existing debug videos.

- [x] Phase A: Frame.
- [x] Phase B: Design the workflow.
- [x] Phase C: Run the loop. Both bounded trials failed qualification.
- [x] Phase D: Keep the audit trail.
- [ ] Phase E: Verify and hand back.

## Verification sequence

- [x] Replay saved inputs and compare requested motion with native motion.
- [x] Test longitudinal interventions against a frozen native prefix and report the inconclusive primary result.
- [x] Separate desired motion from actuator compensation and verify the PI controller with native reference replay.
- [x] Preserve temporal input history during phase recovery collection and reject contacts at teacher takeover.
- [x] Verify the final mixed recovery dataset and unchanged validation membership.
- [x] Freeze initializer, loss, sampler, calibration, selection bounds, and source identities before optimization.
- [x] Complete 500 consecutive optimizer updates and run the strict checkpoint selector.
- [x] Verify candidate camera inference and controller replay in a 120-frame native pilot.
- [x] Run unchanged and retrained models on the same exposed heavy skidpad and cone track.
- [x] Audit and export every paired run, including failures, with the existing debug compositor.
- [ ] Produce training and hardware graphs, update the report, and open a reviewable stacked PR.

## Throughput checkpoint

The batch contains 128 anchors, including 16 recovery draws. The model consumes eight steps of camera history. The full model stays loaded, with frozen vision and auxiliary heads and 50 trainable policy tensors. FP32 training retains the prior verified execution settings. GPU work is serialized. Large archive transfers pause during training timing. Evaluation records actual camera inputs, inference outputs, PI state, physical contacts, and native motion at 20 Hz.

The physical comparison uses the same PI controller for both checkpoints. The earlier 25-to-12 contact change is a separate controller-only diagnostic. Each exposed case has one run per checkpoint. These cases do not estimate a general completion rate. Fresh cases remain sealed if checkpoint qualification fails.

## Failed trial and next premise test

The 500-update PI trial is rejected. Offline consistency improves while steering validation and paired driving regress. Both skidpad policies fail after R,R. The candidate ends earlier, so seven versus sixteen contacts is not a demonstrated improvement. Track contacts increase from one to three. Fresh cases remain sealed.

A three-arm native calibration now tests exact recorded future steering. Recorded controls reproduce all 1,128 states exactly with zero contacts. OP plus PI completes R,R,L,L with one transition contact. Pure pursuit plus PI records 22 contacts and rolls over. Both fail the predeclared zero-contact gate. The next fix must retain OP and distinguish its transition clearance error from the larger learned plan failure.

The native probe uses a same-image Docker container with a 48 GiB RAM-backed output filesystem and read-only datasets. Its complete outputs were copied locally and all arm inventories verified. Only the two explicitly approved redundant Spark archives were removed.

- [x] Build the rejected-trial report with all eight videos, training plots, and measured hardware throughput.
- [x] Verify the full native reference against recorded controls before steering comparisons.
- [x] Complete full heavy-skid evaluation of the earlier vision-adapted weights under PI.
- [x] Run independent one-update objective comparisons on TRAIN fit and monitoring anchors before choosing another recipe.

## Bounded final-layer retraining

The 48-cell objective probe found production-rate fitting overshoot. Twelve reduced-rate updates removed overshoot but worsened monitored steering. Eight final-affine-only updates reduced that steering regression relative to full-policy updates, without removing it. The first head attempt aborted before updates because changed gradient flags shifted forward arithmetic. A zero-update ABAB test isolated the flag effect. The revised attempt preserved the original flags and passed strict starting parity.

The next experiment is one fixed 500-update run. It keeps the native motion targets, PI controller, sampler, FP32 history, consistency coefficient and absolute qualification ceilings. Only the final affine layer is optimized at policy rate 3e-6. Explicit desired-motion steering supervision uses the pre-existing coefficient 3.281158240898283. Longitudinal supervision stays zero because its speed/feedforward combination has a nullspace that can hide target-speed error. This is a candidate fix, not a demonstrated driving improvement. No validation-driven coefficient retries are planned.

- [x] Add explicit final-plan optimizer scope with unchanged forward flags and complete frozen-state integrity checks.
- [x] Freeze calibration, protocol, source and recipe before 500 updates.
- [x] Execute one bounded retrain, apply unchanged qualification ceilings, and preserve failed checkpoints.
- [x] Verify saved-input inference and complete the exposed skidpad and track comparisons with audited debug videos.
- [x] Update the report with training graphs and the actual physical outcome.
