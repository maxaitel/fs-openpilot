# Native-future action target experiment

The user wants native Isaac training that reduces cone contacts and timeouts, with the same auditable debug videos. This experiment corrects the action-label contract while retaining the previous experiment's controller, optimizer, data and physical screening limits. It does not claim a successful driving model before physical evaluation.

## Reason for the correction

The previous experiment asked one predicted trajectory to match native future states and another controller's immediate commands. The deployed `op_tangent_v1` follower does not produce those teacher commands from the exact native trajectory. The saved-anchor diagnosis measured nonzero action loss even for the native future. A descriptive change to predicted velocity knots improved plan coherence and native-future action error while worsening the teacher-command error. This contradicts the premise that both targets share the same optimum. It does not by itself prove improved closed-loop behavior.

The new `native_future_v1` action targets apply the unchanged deployed follower to the detached native future plan, measured body velocity and the same vehicle limits. Both predicted actions and target actions are compared before mission braking. Recorded teacher-command errors remain separate diagnostics and are excluded from the loss. A perfect native future now has zero action imitation error under this contract. There is no new consistency penalty, runtime smoothing, fitted drag correction or loss-weight search.

## TRAIN-only gradient check before updates

The zero-update diagnostic ran on four fixed production-sampler batches at the unchanged initializer. All model tensors remained unchanged. Its report SHA is `77443145c546967aaa75b19671dba274046714053db109b1f43a115510be6a48`; the exact 512 draws are recorded, including replacement duplicates. CPU video exports overlapped this diagnostic, so its 8.023-second wall time is not an isolated performance benchmark.

The new target does not show universally better parameter-gradient alignment. Cosines between the total gradient and plan gradient fall on all four batches, from 0.816–0.878 to 0.765–0.850. Action-gradient norms rise from 32.36–37.98 to 40.50–45.34 despite unchanged numeric loss weights. This changes the gradient mixture and does not establish easier optimization. No coefficients are fitted from these results.

For the predeclared output direction that interpolates the model's own velocity knots 5 and 6 between knots 4 and 7, all four batch directional derivatives become more negative. Negative per-draw derivatives increase from 443/512 to 492/512. This concerns a fixed output-space intervention and repeated samples, not an Adam update or physical driving. The exact compatibility of native plan and action targets remains the reason to test the correction; measured improvement still requires the frozen selection gates and native rollouts.

## Frozen experiment

Use initializer `e59efc8ace2d05a5ecfc29023712e7b5ce1d34a64133dd869a7bc17f849bc35f`. Retain the full 31,154,802-parameter model and train its 6,949,412 policy parameters with frozen vision. Use 500 updates, FP32, batch size 128, history context 8, seed 0, learning rate 0.0001, policy multiplier 0.3, trunk multiplier 0.1, plan weight 1, steering weight 3.281158240898283 and longitudinal weight 1. No feature cache, BF16 or channels-last path is enabled.

Use the unchanged native v4 manifest and off-policy recovery recipe. Each update draws 16 recovery and 112 other examples with replacement. The 8,000 recovery draws resample 80 distinct recovery anchors. Collection used checkpoint `1ac2e3ba949db39ab445c636f841360973e2add8c86edab689a3a872f040ce55`, which differs from the optimization initializer.

First run zero-update `native_future_calibration1`. The builder verifies its complete calibration identity, exact source inventory and archive, unchanged source-independent identity, unchanged native target coherence, unchanged predicted plan/coherence metrics and unchanged teacher-command diagnostics against calibration2. It then writes schema-2 `native_future_selection1.json` and `native_future_trial1.yaml`. Freeze their bytes, the builder, this rationale and the runtime before the first optimizer update. A binding receipt records the input and output hashes. Calibration is not a training run.

## Selection bounds

Copy all global and per-case coherence ceilings numerically unchanged from `tangent_selection2.json`, SHA `f1ce3776281955b0a1b899a201089cea8b099b24c89c9959caa84468f5fb17ba`. Copy the global plan-L1 ceiling of 0.9319418359547854 and every per-case plan-L1 ceiling unchanged. In particular, acceleration/secant mean, p95 and maximum caps remain 1, 2 and 4 m/s². No failed bound is loosened for this experiment.

Only the steering and longitudinal action bounds change their target meaning. Their global ceilings equal the new zero-update baseline values. Each episode ceiling equals 1.10 times its new baseline value. These are engineering screening limits, not measured confidence intervals. Schema 2 and the explicit action-target contract prevent these numbers from being mistaken for legacy teacher-command error. Teacher-command metrics are finite, auditable diagnostics and are not selection gates.

Select only among steps 0, 100, 200, 300, 400 and 500. Step 1 is retained and verified but cannot be selected. Require all global and per-case bounds, unchanged native target coherence, unchanged non-trainable state, exact source/data/sampler bindings and strict improvement in aggregate new-target steering error over step zero. Among eligible trained checkpoints choose the lowest steering error, then the earliest step on a tie. The trainer's ordinary aggregate-loss checkpoint is not authoritative.

## Failure and physical evaluation policy

If no trained checkpoint qualifies, report no selected trained candidate and no promotion. The fallback step zero may itself be ineligible. Only the final scheduled step, exactly step 500, may then be evaluated on the two already exposed heavy track and skidpad cases for rejected-model diagnosis and failure videos. Label it rejected in plans, reports and videos. Do not substitute another failed checkpoint after inspecting driving results. Fresh test scenes remain unopened.

If a checkpoint qualifies, that earns physical evaluation only. Compare it with both the original PP model/controller package and the unchanged initializer using the same `op_tangent_v1` follower. Preserve camera, scene realization, physics, contacts, route scoring, timeouts and every failure. Broader development and fresh-test coverage still need their existing admission gates. Lower offline loss is not evidence of fewer contacts or completed missions.

The Attack the Premise principle changed the training target instead of making another weight adjustment to the conflicting objective. Model the Domain made the target definition a versioned contract bound to calibration, checkpoints and selection. Prove It Works requires actual source-bound calibration and native physical evaluation before reporting a model improvement.
