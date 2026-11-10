# Direct-command premise diagnostic

The goal remains a vision driver that finishes trackdrive and full R,R,L,L skidpad within 120 seconds with zero physical contacts. The tested premise is that a deterministic function of the deployed image history, route history, and measured vx, vy, and yaw rate can reproduce the recorded teacher commands closely enough to remain safe after its own first errors.

My recommendation is to stop varying the command-fit optimizer. First compare the proposed modular planner with a small learned local-corridor predictor on the same frozen perception outputs. The measurements below support a distribution-shift problem and a material teacher inconsistency. They do not yet prove that acceleration targets are unidentifiable from images.

Throughput checkpoint is n/a, read-only investigation. This report and its small analysis directory are the only new artifacts. No frozen joint source was edited. No GPU, simulator, SSH session, new fit, or final case was opened.

## What is established

| Finding | Evidence and implication |
| --- | --- |
| Command alignment is correct in the inspected builder. | `joint_data.py:319-338` uses `tick = ep.ticks[frame]`, verifies `teacher[tick]`, and pairs `states[tick, 3:6]` with `controls[tick]`. `collect.py:91-132` stores the pre-step state, renders the current image, computes the current action, then steps. There is no demonstrated one-tick target bug. |
| The policy receives 1.6 seconds of visual features and the desire queue. | `lab_train.py:39-67`, `native_replay.py:83-84,150-154`, and `model/temporal.py:3-7,24-54` implement the eight past 5 Hz features plus the current feature. All v7 image anchors are phase zero, enforced at `native_replay.py:119-120`. Training and inference pair alignment is consistent at that phase. Passing numerical replay at the other phases does not establish their action accuracy. |
| The raw teacher reads information outside the head's explicit inputs. | `sim/expert.py:87-103,150-179` constructs a route speed profile and uses unrolled progress and mission end. Steering reads the route point at a speed-dependent lookahead in `sim/expert.py:185-196`. `joint_data.py:333-348` supplies images, desire, traffic, and vx, vy, and yaw rate. It supplies no teacher identity or global route geometry. This is a dependency, not proof of an information-theoretic failure. |
| Three TRAIN teachers differ. | `teacher_parameter_identity.json:1` identifies skidpad 82003, 82007, and 82015 with `k_lookahead=0.15`. All other listed TRAIN and VAL episodes use 0.25. The measured fixed-state intervention below shows this changes steering materially. It has no direct effect on acceleration. |
| Aggregate error hides the states that fail. | Recovery fit `validation_500.json:6-21` reports aggregate steering MAE 0.006224 rad. All four skidpad episode groups pass their offline limits. The actual TRAIN pilot fails its entry. On its 53 strictly precontact states, steering MAE against the same teacher is 0.058514 rad. This is a measured mismatch between two state distributions, not an estimate of independent-scene reliability. |
| Offline validation has no measured recovery stratum here. | `train_joint.py:167-175` adds a recovery group only for episodes tagged as recovery. The archived recovery fit's final validation contains episode, low-speed, and recent-pulse groups, but no recovery group. Its 64 low-speed anchors fail. More TRAIN recovery data does not make this VAL distribution cover learner deviations. |
| The teacher's pure-pursuit request ignores vy, yaw rate, and actuator state. | `sim/expert.py:167-196` depends on vx, pose, route progress, and fixed geometry. Existing fixed-latent head interventions respond to yaw rate. This is a candidate shortcut, not proof that yaw rate is generally an invalid visual-policy input. It can help infer otherwise hidden state. |

Source paths above are relative to this worktree. `joint_data.py` and `train_joint.py` are in `.work/lab/joint_command_trial`. `lab_train.py` is in `src/fsop/train`. `native_replay.py` is in `src/fsop/data`. `collect.py` is in `isaac_lab_fsop`. The quoted validation file is `benchmarks/driving-reliability-2026-11-10/direct_command_recovery_training/validation_500.json`.

The inherited 200-tick horizon filter deserves a precise qualification. `native_replay.py:129-148` filters future teacher continuity. `collect.py:139-157` appends 200 actual terminal-braking ticks. The filter normally excludes that continuation, not the final ten seconds of driving. It can exclude up to ten seconds before a learner interval. That affects sampling, but it is not a proven alignment defect.

## Measured fixed-input teacher intervention

The rerunnable analysis is [analyze.py](premise_analysis_v1/analyze.py). Its [result](premise_analysis_v1/result.json) binds the script, thresholds, metadata, evidence manifests, completion receipts, original diagnostic outputs, and teacher source files. It reconstructs canonical skidpad geometry, checks all serialized geometry fields except noncausal summary statistics, and reproduces the archived teacher queries. Every retained state precedes the first physical-contact interval. The only teacher change is `k_lookahead`, from 0.25 to 0.15. The saved physical state, visual history, route history, and head outputs remain fixed. No new policy inference ran.

| Quantity | Initial direct head | Recovery-data direct head |
| --- | ---: | ---: |
| Precontact ticks | 80 | 53 |
| First contact interval | 80 | 53 |
| Maximum teacher reconstruction residual | 3.60e-13 | 6.94e-18 |
| Repeat noise | 0 | 0 |
| Mean steering gap from lookahead change, rad | 0.011104 | 0.006760 |
| p95 steering gap, rad | 0.057218 | 0.021345 |
| Maximum steering gap, rad | 0.063339 | 0.049475 |
| Ticks with steering gap above 0.02 rad | 15 | 3 |
| Acceleration gap, m/s² | 0 | 0 |
| Actual head precontact steering MAE, rad | 0.053399 | 0.058514 |
| Actual head precontact acceleration MAE, m/s² | 0.634312 | 0.283123 |
| Maximum saved head steering change when yaw rate is zeroed, rad | 0.021944 | 0.059447 |

The original archived diagnostic labels reconstruct within the declared 1e-10 bound. The script reruns deterministically and rejects a different result rather than overwriting evidence. Run it from this worktree with `PYTHONPATH=src:. .venv/bin/python .work/lab/next_strategy/premise_analysis_v1/analyze.py`.

These are correlated ticks from two policies on one TRAIN scene. They are not the three actual episodes with the alternate teacher. The intervention proves that teacher identity can change the correct label while observation inputs are fixed. It does not prove that the original dataset contains exact identical-input conflicts. The preceding trajectory might reveal teacher identity indirectly.

For equally weighted duplicate inputs with labels a and b, the minimum pair-mean absolute error is `abs(a-b)/2`. The measured steering lower bounds would be 0.005552 and 0.003380 rad across these artificial pairs. Both are below the existing 0.01 aggregate limit. Therefore these measurements cannot establish that the existing aggregate gate is impossible. The lookahead mixture also cannot explain acceleration failures or the full recovery steering error.

The root reports that the joint suffix trial also fails all six offline snapshots, with final aggregate MAE 0.006521 rad and 0.176977 m/s² and p95 0.023492 rad and 0.781391 m/s². It reports three pilot contacts with passing CPU replay and rendering. Those are coordinator-provided results. This local investigation has not independently inspected the full joint receipts.

## Bounded next tests

The first two tests use only existing TRAIN arrays and saved inputs. The third requires a separately frozen physical counterfactual if the inexpensive dependency test finds an effect. VAL and final membership remain untouched.

1. **Reconstruct every actual TRAIN label before another fit.** Replay canonical progress from recorded states and query the episode's declared teacher at every currently eligible anchor. Require both command residuals at or below 1e-10, as in the local requery, before interpreting conflicts. Archive counts by teacher setting, ordinary versus recovery data, vx below 2, route transition, and 20 Hz phase where RGB exists. Query the same states again with canonical lookahead 0.25. Report command deltas without relabeling existing arrays. Any reconstruction failure is an alignment or provenance defect and stops a new fit. If the alternate setting causes steering differences above 0.02 rad on actual eligible anchors, canonical TRAIN teacher supervision becomes a distinct experiment worth considering. It is not a reason to change VAL targets or mark the old fit qualified.

2. **Measure identical-input contradictions and off-policy error separately.** Hash the complete normalized policy input, including all historical image pairs, the full desire queue, and float32 proprioception. For exact duplicate groups, compute the componentwise median's unavoidable MAE and the minimum attainable p95. Preserve episode membership and bootstrap by episode only if estimating population uncertainty. Exact group conflicts exceeding the existing per-group limits refute that representation's deterministic-label premise for those groups. No collisions is inconclusive. Near-neighbor retrieval can nominate examples, but cannot prove impossibility. Independently replay the frozen rejected package on the saved learner prefixes and compare with teacher queries at those actual states. Report signed error in fixed 0.5-second windows, maxima, and threshold-exceedance runs in addition to MAE. Existing 0.01/0.25 MAE and 0.03/0.75 p95 limits remain unchanged. Do not choose a checkpoint using these diagnostic outcomes.

3. **Test hidden future geometry without confusing code dependence with visual ambiguity.** Fix all current and historical camera inputs, proprioception, desire queue, teacher parameters, and mission phase for selected precontact TRAIN states. Recompute the teacher speed profile after replacing only a downstream curvature suffix with a legal alternate TRAIN route continuation. Keep near geometry and all behind-car geometry identical. Record `v_t`, `v_ahead`, feedforward, and final raw acceleration. A functional probe can hold stored images fixed while varying this hidden suffix, but that alone is only a dependency test. To establish a physically admissible ambiguity, the alternate scene must preserve every rendered input in the policy's full history at the existing repeat-noise bound, including occlusion and lighting effects, and retain identical route and AS inputs. Reject a pair if those inputs differ. Then acceleration gaps above 0.5 m/s² exceed twice the per-example MAE reference, and gaps above 1.5 exceed twice the p95 reference. The rigorous decision is the unavoidable group MAE and p95 across equally weighted admissible pairs, not one large gap. If those exceed the unchanged gates, that raw acceleration target cannot be reproduced from this observation contract on the paired diagnostic distribution. Report how that distribution differs from the original suite.

Use the full route to perform test 3 only inside the diagnostic teacher. It must not enter a learned or geometric steering controller. Do not merely swap `s`, `s_end`, or `v_ref` and call the result an unobservable scene. Those interventions establish hidden-variable sensitivity but can change the mission or produce an impossible route. The external route-command interface and existing AS `mission_brake` remain provided inputs and behavior.

Reading a global route does not imply arbitrarily distant anticipation. The 8 m/s speed cap and 8 m/s² profile braking limit bound the extra braking distance to at most 4 m before a restriction. The direct acceleration query also previews `0.6*vx`, and curvature smoothing adds support. A restriction far outside that support may have exactly no effect. Therefore the test must locate actual support before spending a simulator run on an invisible-suffix pair. The current evidence leaves this question unresolved.

## Decision and alternative

The structurally distinct candidate is specified in [the corridor sketch](learned_corridor_candidate_v1.md). Compare it with the geometric planner on identical perception and route inputs before choosing an implementation. It predicts the local route from cone observations instead of predicting raw actuators from the policy's 256-value summary.

If only teacher identity is material, a separately versioned canonical TRAIN-label experiment is feasible because every existing VAL teacher already uses 0.25. Keep original raw-label metrics and the existing VAL limits intact. If a real hidden-geometry collision exceeds fixed raw-command limits, neither a larger head nor a different loss can remove that information deficit. A geometry-based controller can still be evaluated as a distinct architecture with explicit geometry criteria and the same physical acceptance suite. It must never be reported as passing the failed raw-command contract. Do not claim that a conservative speed rule necessarily reproduces the globally planned acceleration targets.

Attack the Premise changed the next step from another optimizer adjustment to controlled teacher and input tests. Explain the Number changed the report from aggregate MAE to precontact errors, exact noise checks, and lower-bound qualifications. Build the Lever produced the rerunnable analysis. Model the Domain kept its output as one case record with per-state counterfactual rows.
