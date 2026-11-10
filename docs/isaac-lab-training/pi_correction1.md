# Skidpad correction and retraining

The goal is fewer cone contacts and completed skidpad missions, with frozen evaluation rules and the same recorded vision/debug evidence. Offline loss is not a driving result.

The original 25-contact checkpoint predicts an inward trajectory that native motion follows closely. At 3.5 seconds its predicted position 0.5 seconds ahead is within 0.079 m of the recorded car position. Same-input comparisons find a large longitudinal change after training and much smaller steering changes. A native hybrid experiment changes early inner contacts to later outer contacts when initializer longitudinal output is restored. No arm exceeds the predeclared 0.1 m radial-improvement threshold, so its primary causal verdict remains mixed/inconclusive.

Native future acceleration and wheel-force command are different quantities. Qualified steady-turn TRAIN examples need about +0.24 m/s² equivalent command while their net tangent acceleration is near zero. The new `op_tangent_pi_v1` execution contract adds bounded speed integral feedback, conditional anti-windup, explicit mission-override resets, and consecutive 20 Hz timestamps. The predicted plan remains physical motion. The runtime and saved-frame auditor execute the same stateful controller sequence.

A frozen native reference probe replays recorded teacher steering by time. Recorded controls reproduce all nine recorded states exactly and reproduce zero contacts. Against the same native future speed reference, PI reduces speed MAE from 0.22957 to 0.03002 m/s and steady-tail signed error from -0.39177 to +0.00339 m/s. Contacts change from eight to four, with zero saturated ticks on either arm. This passes the reference probe, not a learned-policy mission gate.

A 120-frame TRAIN pilot of the latest frozen weights with PI passes full saved-input inference replay and stateful artifact auditing. A separate complete exposed heavy skidpad run finishes DNF with 12 contacts and five off-course events after two right laps. The prior v1 run had 25 contacts and six off-course events. This is a one-run controller intervention with identical weights, not a retraining result or an estimated general completion rate.

## Corrective data

The prior recovery set contains right-entry recoveries but no left-transition corrective examples. Ordinary left-lap teacher examples exist. The new phase state machine observes every control tick and preserves policy history through a teacher prefix. It resets controller integral state after each discarded proposal. Left learner control begins 12 m before the first left-route decision, only after a clean first right lap. A 0.35 m lateral-error trigger selects the first teacher handoff within six seconds. Existing clean-completion, zero-contact, settled-recovery and qualified-future gates remain.

The first PI entry collection in `data/native/lab_v5` was rejected. It never reached the corrective trigger during its six-second learner window. Its later teacher-driven clean finish does not make it a corrective example. The failed attempt is retained.

`phase_recovery_recipe3.json` initially froze entry collection with the pre-fix c121/v1 package and left-transition collection with the same c121 weights under PI. Every case declares its checkpoint and controller. The PI wrapper verifies all 417 model tensors are unchanged and records its parent checkpoint. Four old track recoveries retain their original source-bound receipts. The fourth phase capture made one contact in the teacher takeover interval and failed qualification. Recipe 4 retains its older clean PP recovery and adopts only the three independently qualified new captures. The finalized v7 dataset has 7,006 TRAIN and 2,067 validation anchors. Validation cases and membership remain unchanged. This explicitly mixed collection supplies physical corrective targets without claiming all examples are on-policy for the new controller.

## Frozen optimization and selection

Short TRAIN integration probes produced two contacts each for the original and vision-adapted weights, versus zero for the latest weights with PI. All three passed integration auditing. The initial proposal favored the vision-adapted initializer; the exploratory initialization choice was revised before optimization to retain the latest c121 weights with PI. Their frozen vision already contains the same appearance adaptation. These six-second probes do not establish a general driving ranking. Both the original proposal and revised decision are retained in the benchmark evidence.

Use 500 policy updates, FP32, batch 128, 16 recovery draws, history context eight, and the prior learning-rate schedule. Vision and auxiliary heads stay frozen. The full model remains loaded. No BF16, feature cache, or channels-last path is enabled.

Use plan NLL plus native-relative endpoint-acceleration consistency with coefficient 1.0. Steering and longitudinal auxiliary weights are zero. For intervals through the first second, the consistency residual compares each endpoint acceleration with its interval velocity secant, then subtracts the same residual of the detached native target. Matching nonconstant native motion has zero loss. The coefficient was fixed from TRAIN-only gradient diagnostics, before the new calibration or optimization. On the vision-adapted initializer, consistency gradient norms were 36.77 to 51.84 versus plan norms 50.09 to 51.26 across four sampled batches. These are raw gradient diagnostics, not evidence of Adam convergence.

Schema 3 binds the desired-motion target, coefficient, initializer, complete mixed sampler, calibration, source archive and plan-L1 ranking. Preserve all prior absolute plan and coherence ceilings. Action metrics use the explicit unclipped desired-motion contract. Select only eligible scheduled checkpoints that strictly improve aggregate plan L1 over step zero, with the earliest step on a tie. No post-training bound relaxation is permitted. An unqualified final checkpoint may be evaluated only as a rejected diagnostic on exposed development cases. Fresh tests remain sealed until qualification.

The PI training archive includes package, native environment, verifier and selector dependencies. Independent review reproduced and fixed a missing-dependency archive failure and a preflight configuration that would only have been rejected after training. CPU verification passed 430 tests, with four skips and seven subtests.

## Executed trial

The 500-update run completed in 479.69 seconds. Median whole-GPU utilization was 86% and median power was 49.1 W. Peak PyTorch allocation was 1.412 GiB. Unrelated CFD CPU jobs overlapped roughly the final 90 seconds, so the 16.164 GiB peak sampled host RAM includes those jobs. This is a measured job duration, not an isolated speedup or a FLOPS-efficiency measurement.

Every scheduled checkpoint failed the frozen qualification rules. The final checkpoint is retained only as a rejected diagnostic. Native-relative consistency fell from 0.098621 to 0.032361, but steering SmoothL1 rose from 0.00125601 to 0.00140550. Weighted plan L1 improved from 0.875179 to 0.869869 while plan NLL worsened from -1.546599 to -1.488385. The total validation objective improved because the consistency reduction outweighed that NLL regression. Neither offline loss nor the lower contact count below establishes a driving improvement.

The two exposed cases were completed with the same PI controller and unchanged camera, scene, route and scoring contracts.

| Case | Model | Result | Simulated seconds | Physical contacts | Off-course events |
|---|---|---|---:|---:|---:|
| Skidpad | Unchanged c121 weights | Invalid route after R,R | 46.30 | 16 | 4 |
| Skidpad | Rejected step 500 | Invalid route after R,R | 32.05 | 7 | 2 |
| Track | Unchanged c121 weights | Finished | 56.55 | 1 | 1 |
| Track | Rejected step 500 | Finished | 57.50 | 3 | 1 |

The candidate skidpad failed earlier at the first left circle. Its seven contacts cannot be called a gain over sixteen because it had less exposure. Both track runs completed, with more candidate contacts. These are one run per model per exposed case, not a general completion-rate estimate. Fresh cases remain sealed. All eight debug and research videos passed producer artifact auditing, hash verification and complete-frame decoding. The earlier incomplete candidate capture stopped for disk reserve and remains a technical failure outside these driving scores.

An earlier c121/PI skidpad run had twelve contacts. Repeating it produced sixteen. Initial physical states, camera matrices and cone positions match, but RGB already differs at frame zero. A separate eight-tick fixed-action probe found no measured physics-time, pose or velocity advance during capture, evidence writing or idle. It distinguishes render/app time from physics time without establishing deterministic rendered inputs or neural rollouts.

## Perfect-reference steering check

Before choosing another training loss, a separate frozen three-arm probe used the original clean TRAIN skidpad 82000 trajectory. Recorded controls reproduced all 1,128 native states exactly, with zero contacts and a valid R,R,L,L finish. Two followers then received the same actual-ego rebased native future after the first right lap. OP steering plus PI completed R,R,L,L and stopped, but made one contact at the right-to-left transition. Pure pursuit plus PI made 22 contacts and rolled over after R,R,L. Both failed the predeclared zero-contact gate. These privileged reference results support retaining OP; they are not learned-model performance.

The consistency objective does not directly supervise position, yaw or yaw rate. It is also invariant to adding a constant velocity offset at every knot. Steering and target speed can therefore worsen while consistency improves. The earlier vision-adapted initializer also failed a complete PI diagnostic. It timed out at 120 simulated seconds with seventeen physical contacts, six off-course events and no accepted laps. Its 2,400 recorded inference frames and both videos were independently copied and verified. This one exposed episode does not support rolling back to that initializer under PI.

The [benchmark evidence](../../benchmarks/pi-correction-2026-11-10/) contains the frozen inputs, rejected selection receipt, completed evaluation seals and video audits. `generate_report.py` rebuilds the linked local report and training graphs from those receipts and the verified media files. The skidpad issue is not yet fixed.

## First-update diagnosis

A frozen TRAIN-only experiment made 48 independent updates across two initializers, four fixed fit batches and six objectives. Every cell reset exact weights and created a fresh production AdamW optimizer. Separate monitoring anchors covered steady right, right-to-left transition, steady left and recovery. Their indices were excluded from fit batches, although episodes and temporal windows could overlap.

Plan NLL alone and plan NLL plus consistency increased their own fit objective in all four batches for both initializers at the production policy learning rate of 3e-5. Steering-only updates reduced fit steering loss but worsened monitored steering in every batch. This does not support simply increasing the steering weight. Independent NumPy checks reproduced the saved output losses and motion metrics. Parameter-space norms and cosines are excluded because their FP32 calculation failed a range check.

A separate twelve-cell experiment changed only the learning rate for the latest initializer and current objective. Rates 3e-6, 3e-7 and 3e-8 all reduced fitting loss. At 3e-6, fit reduction was 0.01694 to 0.03602 across the four batches, well above arithmetic discrepancies. Monitored steering still worsened in all twelve cells. Raw acceleration and target-speed errors decreased in aggregate, with phase tradeoffs. Smaller steps remove the immediate fitting overshoot but do not remove the observed steering conflict. No rate passed the frozen all-batch monitoring criterion.

Eight subsequent updates restricted optimization to the final plan affine projection. They preserved the strict starting-output match and kept all other 415 state tensors unchanged. At 3e-6, fitting loss decreased by 0.003157 to 0.005645, but monitoring steering MAE increased by 0.00000647 to 0.00000827 radians. All eight head-only updates worsened steering less than the corresponding full-policy updates, but none improved over the initializer. Shared-feature changes contribute to the observed regression without explaining all of it.

The initial head-only attempt changed requires_grad flags and failed before any update because forward arithmetic shifted slightly. A zero-update ABAB check reproduced this flag-dependent shift; restoring the original flags restored exact predictions. The revised update test preserved the flags and requested gradients only for the two optimizer-selected leaf parameters. No tolerance changed.

The next bounded recipe adds direct desired-motion steering supervision to the final-layer updates. Native NLL scores yaw and yaw rate separately; the deployed steering request depends on their signed combination. The recipe keeps physical targets and PI, uses the inherited steering coefficient 3.281158240898283, and retains all qualification ceilings. This is one fixed 500-update trial at policy learning rate 3e-6, with no validation-driven coefficient retries. Its rationale and limits are recorded in [the frozen decision](../../benchmarks/pi-correction-2026-11-10/head_training_decision.md).


## Bounded head retraining

The fixed head trial completed all 500 updates in 693.98 seconds. Median sampled GPU utilization was 75% and power was 42.03 W. Peak PyTorch allocation was 1.414 GiB. Several unrelated simpleFoam jobs occupied host CPU cores throughout this run, so its duration cannot isolate a performance effect from restricting the optimizer.

Only the two final affine tensors changed, totaling 254,430 parameters; all other 415 state tensors remained exact. Steering validation SmoothL1 improved from 0.00125601 to 0.00124575 (0.82%). Consistency improved from 0.098621 to 0.074224 (24.7%), while longitudinal loss worsened from 0.184256 to 0.185532 (0.69%). Every scheduled checkpoint failed the unchanged qualification ceilings. The final checkpoint failed 25 checks and remains a rejected diagnostic. Its acceleration-secant p95 and maximum were 4.805 and 7.125 m/s² against fixed limits of 2 and 4.

The predeclared final diagnostic passed a full 120-frame saved-input inference replay and stateful artifact audit with zero pilot contacts. Two prior pilot setup attempts failed before driving because copied-workspace asset symlinks pointed outside the export root. Their logs are retained. Copying the exact material, person and panorama assets into the isolated workspace resolved the export layout without changing the exporter guard or model. The complete paired skidpad and track driving comparison finished with the outcomes below. All eight videos passed the same producer artifact audits, hash checks and full decoding as the first trial. The complete raw evaluation archive and all training checkpoints are verified locally. Training and hardware graphs are linked in the benchmark report.


| Case | Model | Result | Simulated seconds | Physical contacts | Off-course events |
|---|---|---|---:|---:|---:|
| Skidpad | Unchanged c121 weights | Standstill DNF after R,R | 76.90 | 21 | 5 |
| Skidpad | Rejected head step 500 | Rollover after R,R | 32.95 | 13 | 1 |
| Track | Unchanged c121 weights | Finished | 56.25 | 2 | 1 |
| Track | Rejected head step 500 | Finished | 57.05 | 2 | 1 |

Both arms completed one of the two exposed cases. Each completed track and failed skidpad. These counts are descriptive outcomes for these cases, not an estimate of general completion probability. The candidate's shorter failed skidpad exposure prevents interpreting fewer contacts as better avoidance. Track contacts were unchanged and elapsed simulation time increased by 0.80 seconds.

The [recorded trajectory analysis](../../benchmarks/pi-correction-2026-11-10/head_failure_trace/README.md) reproduces the baseline's timing crossings from saved positions. Its two left return crossings missed the finite gate by 1.096 and 1.178 m. Projected route distance reached the endpoint without any lap-sized progress jump, but only R,R had been accepted. The car stopped outside the exit lane. The candidate rolled over before its first left return crossing. Both inward and outward excursions remain, and these records do not isolate the final rollover's causal chain.

Neither training trial qualified or demonstrated successful skidpad driving. No checkpoint is promoted. The next intervention must address the remaining learned motion and transition-clearance errors; changing terminal braking alone does not repair missed gates or prior cone contacts.
