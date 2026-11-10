# Skidpad correction and retraining

The goal is fewer cone contacts and completed skidpad missions, with frozen evaluation rules and the same recorded vision/debug evidence. Offline loss is not a driving result.

The original 25-contact checkpoint predicts an inward trajectory that native motion follows closely. At 3.5 seconds its predicted position 0.5 seconds ahead is within 0.079 m of the recorded car position. Same-input comparisons find a large longitudinal change after training and much smaller steering changes. A native hybrid experiment changes early inner contacts to later outer contacts when initializer longitudinal output is restored. No arm exceeds the predeclared 0.1 m radial-improvement threshold, so its primary causal verdict remains mixed/inconclusive.

Native future acceleration and wheel-force command are different quantities. Qualified steady-turn TRAIN examples need about +0.24 m/s² equivalent command while their net tangent acceleration is near zero. The new `op_tangent_pi_v1` execution contract adds bounded speed integral feedback, conditional anti-windup, explicit mission-override resets, and consecutive 20 Hz timestamps. The predicted plan remains physical motion. The runtime and saved-frame auditor execute the same stateful controller sequence.

A frozen native reference probe replays recorded teacher steering by time. Recorded controls reproduce all nine recorded states exactly and reproduce zero contacts. Against the same native future speed reference, PI reduces speed MAE from 0.22957 to 0.03002 m/s and steady-tail signed error from -0.39177 to +0.00339 m/s. Contacts change from eight to four, with zero saturated ticks on either arm. This passes the reference probe, not a learned-policy mission gate.

A 120-frame TRAIN pilot of the latest frozen weights with PI passes full saved-input inference replay and stateful artifact auditing. A separate complete exposed heavy skidpad run finishes DNF with 12 contacts and five off-course events after two right laps. The prior v1 run had 25 contacts and six off-course events. This is a one-run controller intervention with identical weights, not a retraining result or an estimated general completion rate.

## Corrective data

The prior recovery set contains right-entry recoveries but no left-transition corrective examples. Ordinary left-lap teacher examples exist. The new phase state machine observes every control tick and preserves policy history through a teacher prefix. It resets controller integral state after each discarded proposal. Left learner control begins 12 m before the first left-route decision, only after a clean first right lap. A 0.35 m lateral-error trigger selects the first teacher handoff within six seconds. Existing clean-completion, zero-contact, settled-recovery and qualified-future gates remain.

The first PI entry collection in `data/native/lab_v5` was rejected. It never reached the corrective trigger during its six-second learner window. Its later teacher-driven clean finish does not make it a corrective example. The failed attempt is retained.

`phase_recovery_recipe3.json` freezes entry collection with the pre-fix c121/v1 package and left-transition collection with the same c121 weights under PI. Every case declares its checkpoint and controller. The PI wrapper verifies all 417 model tensors are unchanged and records its parent checkpoint. Four old track recoveries retain their original source-bound receipts. Validation cases and membership remain unchanged. This explicitly mixed collection supplies physical corrective targets without claiming all examples are on-policy for the new controller.

## Frozen optimization and selection

The initializer preflight compares short TRAIN integrations of the original, vision-adapted, and latest weights. If the vision-adapted initializer passes integration, retain that appearance adaptation and restart policy optimization from it. Do not restart from the latest regressed action-trained policy.

Use 500 policy updates, FP32, batch 128, 16 recovery draws, history context eight, and the prior learning-rate schedule. Vision and auxiliary heads stay frozen. The full model remains loaded. No BF16, feature cache, or channels-last path is enabled.

Use plan NLL plus native-relative endpoint-acceleration consistency with coefficient 1.0. Steering and longitudinal auxiliary weights are zero. For intervals through the first second, the consistency residual compares each endpoint acceleration with its interval velocity secant, then subtracts the same residual of the detached native target. Matching nonconstant native motion has zero loss. The coefficient was fixed from TRAIN-only gradient diagnostics, before the new calibration or optimization. On the vision-adapted initializer, consistency gradient norms were 36.77 to 51.84 versus plan norms 50.09 to 51.26 across four sampled batches. These are raw gradient diagnostics, not evidence of Adam convergence.

Schema 3 binds the desired-motion target, coefficient, initializer, complete mixed sampler, calibration, source archive and plan-L1 ranking. Preserve all prior absolute plan and coherence ceilings. Action metrics use the explicit unclipped desired-motion contract. Select only eligible scheduled checkpoints that strictly improve aggregate plan L1 over step zero, with the earliest step on a tie. No post-training bound relaxation is permitted. An unqualified final checkpoint may be evaluated only as a rejected diagnostic on exposed development cases. Fresh tests remain sealed until qualification.

The PI training archive includes package, native environment, verifier and selector dependencies. Independent review reproduced and fixed a missing-dependency archive failure and a preflight configuration that would only have been rejected after training. CPU verification passed 430 tests, with four skips and seven subtests. Native rollout, retraining and final videos remain separate completion requirements.
