# Candidate B. Learn the executed command

The goal is a vision model that completes cone tracks and a full R,R,L,L skidpad with zero physical cone contacts. This candidate changes the supervised output from a physical trajectory to an actuator request. It retains the complete 31,154,802-parameter SupercomboFS model and freezes every existing tensor.

## Usage

These are prospective call sites for one experimental file. The file does not exist yet. The design sketch contains no training implementation.

```python
policy = NativeActuatorPolicy.load(candidate_package, device="cpu")
policy.reset()
steer_rad, accel_mps2, evidence = policy(obs, env)
```

The native evaluator receives the same policy callable it already accepts. The package declares `direct_native_actuator_v1`. An old plan checkpoint cannot masquerade as this command package. The callable owns image conversion, temporal queues, declared vehicle inputs, command decoding, and the unchanged mission launch and finish intervention.

```sh
python .work/lab/reliability_architecture/actuator_trial.py audit --recipe recipe.json --out run1/preflight
python .work/lab/reliability_architecture/actuator_trial.py train --recipe recipe.json --out run1
python .work/lab/reliability_architecture/actuator_trial.py qualify --recipe recipe.json --package run1/selected.pt --out run1/qualification
```

These commands share one recipe parser. `train` refuses to run without a passed audit bound to the recipe, source, checkpoint, native manifest, and recovery receipts. `qualify` cannot change the frozen recipe or select a second checkpoint after observing driving.

## Grounding

The original assumption was that better native-future fitting would provide sufficient cone clearance through OP plus PI. The two failed 500-update retrains contradict that assumption's sufficiency. The second changed only 254,430 final-plan parameters. Its steering validation loss improved by 0.82%, but it still failed every scheduled checkpoint qualification and rolled over after R,R. The first also failed every scheduled qualification. See `docs/isaac-lab-training/pi_correction1.md`, especially lines 37 through 48 and 77 through 93.

The suffix probe begins from an exact teacher prefix. Position correction, recorded acceleration, and their combination each produced a valid R,R,L,L finish with zero contacts. This identifies multiple ways to repair that particular execution. It does not identify a unique cause or certify standing starts. The unchanged perfect-future OP plus PI standing-start run finished the route with six contacts. See `benchmarks/driving-reliability-2026-11-10/native_transition_correction/run1/comparison.json` and `native_standing_reference/run1/comparison.json`.

`NativeEpisode.sample` already returns the executed `controls[tick,0:2]`. These are the steering request in radians and acceleration request in m/s² sent to the native step. They are not trajectory yaw, desired yaw rate, or measured actuator state. `collect.py` records `action` before `drive.step`. Current `states[tick,3:8]` contains body longitudinal velocity, body lateral velocity, yaw rate, actual actuator steering, and the lagged acceleration actuator state. The last field is actuator telemetry, not measured IMU acceleration. `isaac_lab_fsop/env.py::_sync_state` obtains the velocities from native bodies and the last two values from `drive.act`.

The frozen command probe passed exact saved-output comparisons over 601 frames. Its route intervention changed steering at frame 120 from -0.17329 to +0.05110 radians and changed the transition prediction at frame 430. At frames 170 and 600, those interventions made no change. Frames 430 and 600 followed earlier contacts. This supports preserving the complete route history. It does not establish route compliance. The read artifact is `.work/lab/reliability_command_probe/result.json`, which matches the originally requested probe location before its archive was reorganized.

`ModelPolicy` currently converts RGB to paired packed YUV, advances `TemporalState`, executes the vision and temporal-policy graphs, and sends a 33 by 15 plan to a follower. `fresh_history` recomputes phase-zero 5 Hz pairs from native replay. The old graph reads eight historical 512-wide states plus the current state. The proposed head reads the frozen 256-wide temporal-policy state immediately before `base.policy.p_node_linear_20`. This state receives the same image history and the complete 25 by 8 route pulse tensor. A read-only forward-pre-hook captures the affine input without replacing any graph operation.

## Shape

One experimental file owns the new command contract. It imports the existing model loader, replay verifier, temporal queue, image conversion, mission intervention, and native evaluator. It does not edit those modules during the first experiment. The file contains the private head, immutable input contract, replay adapter, package loader, and policy callable. It also owns the bounded training and qualification commands. This is deliberate locality, not a reusable training framework.

| Item | Shape or count | Semantics |
|---|---:|---|
| Road and wide inputs | Two `[B,12,128,256]` FP32 tensors | Exact current and 0.2-second-old packed frames |
| Visual history | `[B,9,512]` | Eight past states at 5 Hz and current frozen hidden state |
| Route history | `[B,25,8]` | Exact recorded pulse pooling, unchanged channel meanings |
| Traffic convention | `[B,2]` | Existing `[1,0]` convention |
| Frozen temporal-policy state | `[B,256]` | Exact affine input to `base.policy.p_node_linear_20` |
| Vehicle inputs | `[B,3]` | `vx, vy, r` at the same pre-action tick |
| Flat head input | `[B,259]` | 256 temporal values + 3 vehicle values |
| Head | `259 -> 128 -> 2` | One SiLU hidden layer and two raw command outputs |
| Head parameters | 33,538 | 33,280 + 258 |
| Retained base parameters | 31,154,802 | Published existing model inventory, verified again before launch |
| Total loaded parameters | 31,188,340 | Base plus new head |
| Updated parameters | 33,538 | Exactly four head tensors |

The head is a direct policy. It reads the frozen state that precedes the old plan projection, but it does not read the old plan, add a residual to old OP commands, invoke PI, or optimize a plan loss. All old heads remain loaded and available for evidence. The old 417 state tensors remain byte-identical. The full retained model remains part of the saved package. Frozen does not mean changing `requires_grad` flags. Preserve the initializer's exact flags and forward context because earlier probes found flag-dependent arithmetic changes. Exclude base tensors from the optimizer and detach the feature inputs. Validate actual output equality rather than assuming identical weights imply identical arithmetic.

Keep temporal-policy state values in their existing scale. Do not add learned normalization or estimate scaling from validation data. Preserve route pulses and traffic values inside the frozen policy. Divide the three vehicle values by fixed scales `[10, 2, 1]`. The first experiment excludes measured actuator steering and acceleration. Those values are unusually strong persistence shortcuts. These scales are part of the contract. They do not grant permission to feed global x, y, yaw, lateral error, centerline progress, lap index, future state, cones, teacher identity, or episode identifiers into the learned head.

The two raw outputs represent commands in units of 0.1 radians and 2 m/s². Convert to physical units, apply the existing actuator bounds, then apply the unchanged mission intervention once. Record all three values separately. Train against the raw teacher requests, not the bounded or mission-modified prediction, so saturation cannot hide prediction error. Preflight proves all qualified teacher labels lie within the existing actuator bounds. If that is false, stop and resolve the semantics before training. Do not silently clip labels.

The original launch and finish layer still reads the existing progress stand-in. Declare this privileged mission support in every evaluation report. It does not enter the new head. The intervention is deterministic and idempotent for a fixed observation. Verify that reapplying it to recorded teacher commands does not change the action outside the intervention region. Report any changes inside that region separately.

The module's public interface has three operations. Loading establishes the complete contract. Reset clears per-episode queues. Calling the policy advances one observation and returns one request plus evidence. Callers cannot choose between old and new controllers through a boolean or apply a second PI layer. This hides temporal timing, command units, package validation, and mission sequencing behind the existing policy protocol.

The completed pre-contact appearance probe at frame 170 also limits the diagnosis. Removing surfaces changed the plan by 0.03973 m against repeated-control variation of 0.00658 m. Its steering effect was below the predeclared gate. Other tested changes did not exceed their gates. This is one observation and does not establish appearance robustness. The new design retains the known visual model while it tests command execution.

## Targeted experiment

The hypothesis is that the frozen visual representation and route history contain enough information to learn native actuator requests with explicit current velocity and yaw-rate feedback. Bypassing the trajectory-to-controller conversion should remove one demonstrated source of execution mismatch. This is a prediction, not an established result.

Freeze one initializer, one random seed, one architecture, and one recipe before optimization. Use the unchanged c121 PI training package that initialized the two rejected trials. Do not initialize from either failed step-500 checkpoint. Bind its exact SHA-256 and every old tensor hash. Use the verified v7 manifest with 7,006 TRAIN and 2,067 validation anchors and recipe-4 recovery receipts. Keep the existing TRAIN and validation membership. Do not consult final scenes.

Use FP32, contiguous tensors, no feature cache, no channels-last, and no BF16. Recompute historical frozen features through the exact established path. Use the existing 8-frame history context and batch size 128. Draw 112 ordinary TRAIN anchors and 16 qualified corrective-recovery anchors per update. Keep the recovery source mixture from recipe 4 and log realized case counts. Initialization and draw order use seed 0.

Train the new head for one fixed budget of 500 AdamW updates with learning rate 0.001, betas 0.9 and 0.999, epsilon 1e-8, weight decay 0, and gradient norm clip 1.0. Do not search these settings or continue because qualification nearly passed. They are a new-head training recipe rather than another plan-head learning-rate sweep. The roughly nine TRAIN anchor passes are a bounded feasibility check. A lower final training loss does not authorize more updates. Save updates 0, 100, 200, 300, 400, and 500 with the complete base and command package. Record training curves, optimizer settings, hardware telemetry, and attempted runs.

For target `u=(steer_request,accel_request)`, use the mean of `SmoothL1((u_hat_steer-u_steer)/0.02, beta=1)` and `SmoothL1((u_hat_accel-u_accel)/0.5, beta=1)`. The physical scales fix the relative error units. There are no plan, consistency, desired-motion, cone, or residual losses. The 0.02-radian and 0.5-m/s² scales are frozen engineering tolerances, not fitted coefficients.

Before that run, a TRAIN-only wiring test must overfit a fixed 32-anchor set to at most 0.002-radian steering MAE and 0.05-m/s² acceleration MAE within 200 updates. Its head is then discarded. Failure rejects the proposed implementation or representation for this recipe. It does not authorize a sweep. A passing overfit test is only a wiring and capacity check.

## Qualification frozen before training

1. Validate source and data identity. Independently verify native hashes, qualification receipts, anchor counts, no split overlap, exact pre-action alignment, teacher ownership, no contacts in qualified continuations, and declared input availability. Compare each returned target directly to `controls[tick]`. Verify that only the four command tensors enter the optimizer.
2. Validate arithmetic without training. Execute the original path and the wrapped path in ABAB order on actual saved CPU inputs, including the known 601-frame probe and a TRAIN startup, a transition, a recovery, and finish. Require exact old-output equality in the same process and context. Require unchanged old tensor hashes and flags. The capture hook must fire exactly once per policy call, see `[B,256]`, and return nothing. Clear its per-call storage after copying the detached value. A missing or multiply fired hook fails closed. Failure stops before optimization. Cross-device head comparisons use the existing independently frozen inference tolerance, not a newly widened threshold.
3. Validate temporal handling. At 20 Hz, use `TemporalState` unchanged. The replay anchor at 5 Hz must produce the identical `[t-1.6,...,t]` hidden history and route tensor as the corresponding online phase-zero frame. First-frame zero padding and frame-pair duplication must match. Saved replay at off-phase ticks must also reproduce the new command exactly within the existing tolerance. No fabricated 20 Hz image labels are allowed.
4. Validate the learned command. Score all 2,067 validation anchors at every saved update. Require pooled MAE at most 0.01 rad and 0.25 m/s², and p95 absolute error at most 0.03 rad and 0.75 m/s². Require these same ceilings separately for trackdrive, skidpad, low-speed launch, route-pulse transition, and qualified recovery strata wherever validation contains the stratum. Freeze strata using metadata and observations before training. An absent validation stratum stays explicitly unassessed. The absolute maximum must remain finite and within the physical command bounds plus 0.02 rad or 0.5 m/s² before clipping.
5. Reject a persistence shortcut. Compute the parameter-free predictor that requests the current measured steering and actuator acceleration. On predeclared TRAIN dynamic anchors where a teacher request differs from measured state by at least 0.02 rad or 0.5 m/s², require at least a 20% reduction in the corresponding MAE. Require this reduction on the equivalent validation subset when it has at least 32 anchors. Report both counts. Also repeat the command probe with the new head and the complete modified route history. Report action differences relative to unchanged-repeat noise. Do not require a turn response at frames with no effective route change.
6. Check visual dependence. Before selecting a checkpoint, run the frozen held-input intervention using matched original observations and full corresponding temporal history. Blank or permute the two camera histories while retaining route and proprioception. Record the increase in error on the predeclared dynamic TRAIN set. Require the mean normalized command loss to increase by at least 10% and by more than five times the largest repeated-unchanged loss difference on that fixed subset. If it contains fewer than 32 anchors, freeze a larger TRAIN-only subset before training. A smaller effect rejects the claim that this head is a useful vision controller. It may not be promoted on a strong offline command score alone. This intervention diagnoses dependence, not robustness or causal perception quality.
7. Select only among command-qualified checkpoints. Minimize the larger of validation steering-MAE/0.01 and acceleration-MAE/0.25. Break ties by the earlier update. Preserve all failures. If no checkpoint qualifies, record rejection and do not open final cases. Old plan and cone outputs must remain exactly equal to the initializer. Old plan metrics are reported as unchanged diagnostics under this distinct command contract. They are not relabeled as successful command metrics.
8. Pass an actual CPU interface pilot before full driving. Strictly reload the selected package on CPU and execute at least 120 native control ticks from standing in an exposed TRAIN scene. Require valid tensors, finite correctly bounded commands, correct mission order, no contact, and no off-course event. Preserve every attempted setup failure. Audit the whole saved-input sequence, policy state advancement, camera transforms, frame counts, hashes, full video decoding, and both debug and research videos.
9. Run the two exposed development cases from standing, one full R,R,L,L skidpad and one cone track, with three predeclared repeats each. Match camera, physics, scene, initialization, route, referee, and maximum duration. Require every run to finish, zero physical contacts, zero off-course events, no rollover, correct accepted route, and stopped-in-zone where required. Preserve full exposure, lap times, intervention counts, minimum clearance, command rates, and both audited videos. Lower contacts on an earlier failure do not pass. Run the unchanged baseline with the same conditions as a descriptive reference. Its failures do not lower the candidate gate.
10. Select the package before opening final cases. Run the already frozen `acceptance_suite.json` exactly, comprising 16 conditions and four repeats per condition, 64 candidate runs. No retry after an unfavorable drive, no changed membership, no threshold relaxation, and no selection on final outcomes. Keep technical failures separate and resolve them under the predeclared repeat policy. Report all 64 outcomes and uncertainty. Even 64 zero-failure runs yield only a one-sided 95% binomial upper bound near 4.6% under independent-trial assumptions. Repeated scenes weaken those assumptions.

If the command gates pass but standing driving fails, the hypothesis fails as a complete solution. The next experiment must inspect the first divergence before gathering targeted corrective observations or changing representation. Do not respond with another loss-weight sweep. If route or visual dependence fails, that is evidence against sufficiency of the frozen input representation. A geometry-based proposal then earns investigation.

## Throughput and storage checkpoint

Serialize GPU work on DGX Spark. Existing CFD processes are outside this experiment. Preserve the current container and all retained failures. Use the existing source-bound dataset in place. The proposal does not create a feature cache or duplicate camera arrays.

The head adds about 0.128 MiB of FP32 parameters. Its optimizer moments add about 0.256 MiB. Six full 31.19-million-parameter checkpoint payloads have a lower bound around 714 MiB before optimizer data and metadata. Reserve 2 GiB for this run's compact training outputs and telemetry. This is a calculated storage budget, not a measured throughput result.

The initial available capacity is 38 GiB in tmpfs, about 9 GiB on Spark disk, and about 147 GiB locally. Require at least 8 GiB of tmpfs reserve and 5 GiB of Spark disk reserve. Write new temporary outputs to a dedicated run directory. Copy each completed evidence package to local durable storage, verify its manifest and hashes, then release only that task-owned temporary duplicate. Never delete inherited datasets, checkpoints, archives, or evidence.

The 120-tick pilot measures bytes per tick for complete raw evidence and both videos. Before a full episode, reserve its measured worst-case size at the frozen time limit plus 25%. Before the final suite, sum the measured worst-case archive footprint for all 64 candidate runs and required baselines. Require that sum plus 20 GiB of local reserve to fit durable capacity. The 147 GiB starting space may not suffice. Stop for a concrete storage solution before opening final scenes if it does not fit. Compression savings are admissible only after a real complete-frame and saved-input audit of the compressed pilot. Streaming changes storage location, not the requirement to retain evidence.

## Alternatives considered

A direct head on raw visual history and route pulses was the original candidate. With three vehicle signals and no redundant traffic constant, `4811 -> 256 -> 64 -> 2` adds 1,248,450 parameters. It avoids the frozen temporal-policy bottleneck, but it must relearn temporal and route interpretation from 7,006 anchors. The 33,538-parameter latent head is the stronger first test. This is an inductive-bias judgment, not measured superiority. A failed latent head may motivate a raw-history head only if first-divergence and representation checks support that change. The raw-history design is not an automatic second run.

A learned residual over OP plus PI can begin from the old behavior and may need fewer updates. It loses here because residual targets depend on exactly replaying the old PI state at all 20 Hz ticks, while native images are recorded at 5 Hz. It also forces every call site to preserve a plan controller, PI memory, residual units, and command-composition order. It hides less behind a larger contract and retains the demonstrated execution mismatch as its base. It remains a distinct later experiment if exact command replay and residual observability are proved first.

A cone-geometry planner with image detections, temporal cone mapping, route-aware pairing, and a calibrated tracker exposes interpretable clearance constraints. It could address failures that frozen hidden features cannot represent. It loses the first experiment because the current `GeometricPolicy` generates a plan from decoded cones and uses the same follower family. It does not demonstrate zero-contact RRLL competence, and the present cone heads are auxiliary diagnostics. Adopting it requires perception precision, wrong-side association, visibility, route topology, and tracker qualification together. A privileged zero-contact planner would still need a separate vision qualification.

A learned plan-position correction cannot be supervised by the previous native future target at knot zero. Each target starts at the current ego pose, so its position there is zero. Relabeling raw teacher steering as plan yaw would be dimensionally and semantically wrong. This alternative is rejected.

A jointly trained full visual model plus actuator head may be necessary if hidden features omit cone clearance. It loses the first experiment because it changes perception and command mapping simultaneously. This recipe isolates the command mismatch and retains the full architecture. A frozen-head failure is evidence to inform, not permission to silently unfreeze.

## Red-flag screen

| Red flag | Result |
|---|---|
| Shallow module | The single policy operation owns units, timing, package validation, and intervention order. Callers do not coordinate stages. |
| Information leakage | Replay arrays remain private. The learned input excludes ground-truth route geometry and targets. One command contract owns scales and field order. |
| Temporal decomposition | One file owns the command domain across training and deployment. It does not create separate load, transform, or save services. |
| Pass-through methods | No wrapper merely forwards controller calls. The new policy replaces the command-generating behavior. |
| Split ownership | Each policy instance owns its queues. Each run owns a directory. Existing model tensors have no optimizer writer. |
| Two ways to do one task | New packages dispatch only to direct commands. Old packages retain their historical controller. A package cannot switch contracts at runtime. |
| Importable internals | The prototype is a standalone experimental file. Production modules do not import its helpers. Promotion must export only the policy contract. |
| Hand-synced list | One immutable recipe defines proprioception order, scales, output units, and selected tensor inventory. Receipts derive from it. |

## Synthesis decision

This candidate chooses the direct command head on the existing temporal-policy state as its base. It adopts the parent architect's 256-wide state proposal after comparing it with the larger raw-history head. It borrows the existing temporal queue and full-model evidence path. It rejects both a plan correction and a residual controller for the first behavior check. The parent architect still compares this candidate with the independent alternative. No synthesis outcome is claimed here.

## Tradeoffs accepted

- We accept a 256-wide frozen temporal-policy bottleneck and 33,538 new parameters in exchange for a command representation that does not overload the old physical-plan semantics.
- We accept exact but imperfect frozen perception for one experiment in exchange for isolating command execution from visual adaptation.
- We accept a 5 Hz supervised anchor set in exchange for reusing qualified native data, with the unresolved 20 Hz deployment gap tested in real driving.
- We accept declared native vehicle telemetry and the existing mission layer in exchange for a controlled sim experiment. These inputs do not establish hardware deployment readiness.
- We accept one fixed 500-update attempt in exchange for a falsifiable outcome. Failure can indicate optimization, representation, covariate shift, or execution problems, so the first divergence must decide what happens next.

## Open questions and risks

Does the frozen 256-wide temporal-policy state preserve the cone-clearance information needed for corrective commands? Can the head avoid copying measured actuator state? Can 5 Hz teacher anchors support stable 20 Hz commands? Can the current recovery mixture support both entry and right-to-left transition under a new policy? Can complete final evidence fit in available durable storage? These are empirical questions assigned to the predeclared checks. They do not need preference questions or a new learning-rate sweep.

## Next implementation step

After synthesis, write only `actuator_trial.py` against the accompanying sketch and run its CPU contract audit before any training.

## Principle decisions

Attack the Premise changed the supervised output because two plan-fitting retrains and perfect-future execution failed the goal. Model the Domain separates physical plans, actuator requests, and measured actuator state. Boundary Discipline places schema, split, and timing checks in the package and replay boundaries. Laziness Protocol keeps the first behavior experiment in one file and preserves existing production modules.
