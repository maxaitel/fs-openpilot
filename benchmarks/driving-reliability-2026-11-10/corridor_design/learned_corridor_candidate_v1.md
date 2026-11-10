# Learned local corridor as the competing architecture

The goal remains complete trackdrive and R,R,L,L skidpad with zero physical contacts. This candidate learns how observed cones define the local route. It keeps the existing external route commands and AS mission layer. Ground-truth geometry can create TRAIN supervision and score diagnostics, but cannot enter inference steering.

This is a design sketch, not an implemented or qualified model. The parent reports useful correctly associated cone detections at 5 to 10 m and weaker performance nearer and farther away. That result motivates a perception-controlled comparison. I have not independently rerun that detector measurement.

## Smallest distinct candidate

Use the same frozen image detector, calibration, confidence rule, and odometry-compensated cone history as the geometric candidate. Both candidates receive the identical token stream. A token contains estimated ego x and y, color probabilities, confidence, and age. Integrate only measured vx, vy, and yaw rate to transform previous detections into the current ego frame. No true position, simulator heading, centerline, nearest-route index, or teacher setting enters this transform. Bind each transform to recorded sensor inputs so replay can reconstruct it. Preserve the full provided desire queue. Do not replace route commands with a learned lap counter in this experiment.

The new model is a small set encoder with a shared two-layer, width-64 token MLP, masked mean and maximum pooling, and a width-128 output MLP. Append the full flattened desire queue and measured vx, vy, and yaw rate before the output MLP. Train only these new weights. Choose the exact token cap and deterministic overflow rule from the observed TRAIN detector inventory before fitting. Do not silently drop the closest cones when the cap is reached.

Predict eight ego-frame route points at future route arc lengths 0, 1, 2, 3, 5, 8, 12, and 16 m, plus validity or confidence at each station. The zero station is the estimated nearest point on the intended route. It is not forced to the vehicle origin. That makes lateral displacement observable in the prediction and avoids recreating the inherited origin-anchored trajectory problem. Arc-length stations avoid an ambiguous x-to-y curve in a sharp turn. The far stations are hypotheses to test, not trusted extrapolations.

Create the TRAIN point labels using the canonical unrounded route, recorded pose, and unrolled progress. At a skidpad crossing, follow the recorded route branch instead of projecting onto the globally nearest geometric segment. Label data remain train-only. An auxiliary point loss supervises route shape directly. A differentiable teacher-style pure-pursuit calculation can additionally supervise raw steering, using the same fixed lookahead 0.25 and actual vehicle geometry. The initial experiment contains no unconstrained learned steering residual.

Both competing planners feed the same separately specified local-path controller, speed rule, clipping, and unchanged AS mission override. This isolates association and route-shape inference. The speed rule must depend on observed or predicted geometry and measured speed. It cannot query the teacher's global speed profile. The hidden-acceleration diagnostic decides whether preserving the old raw acceleration imitation gate is feasible. A new geometry-contract result must remain distinct from old raw-command qualification.

## Cheap comparison before a new driving run

Freeze the same TRAIN anchor set before seeing candidate errors. Include complete entry prefixes, both right-circle transitions, the right-to-left transition, left exit, ordinary track bends, and the two collected recovery prefixes. Reserve whole TRAIN episodes as an internal design check if fitting is required. This does not replace the existing untouched VAL membership. Use the same detector outputs for both candidates. Run a second diagnostic using correctly associated visible TRAIN cones to separate perception failure from route-association failure. These perfect-perception rows are diagnostic only.

A bounded wiring fit must show that the set encoder can recover route points and steering on a fixed 32-anchor TRAIN subset. Discard those weights. If it cannot achieve steering MAE 0.002 rad or near-route lateral MAE 0.05 m within the inherited 200-update budget, reject the sketch rather than broadening the model or sweeping the optimizer. The full candidate then needs its own frozen training recipe and source identity.

Compare the two methods using this rubric:

| Criterion | Decision rule |
| --- | --- |
| Input contract | Reject any inference dependency on true pose, geometry, teacher identity, or future states. Route history and the existing AS layer remain allowed and disclosed. |
| Recoverable geometry | On the fixed TRAIN check, require lateral error at the controller's selected lookahead to have MAE at most 0.10 m and p95 at most 0.25 m. These are provisional geometry screening limits, not physical-clearance guarantees. Reject if the required lookahead lies outside supported predictions. |
| Branch selection | Require the intended branch at every predeclared entry, crossover, and exit anchor. Report each event separately. An incorrect branch is a failure even when aggregate point error is low. |
| Corrective direction | At precontact recovery anchors, require the sign of the steering correction to agree with the canonical teacher whenever teacher magnitude exceeds 0.03 rad. Report magnitudes and complete errors too. This prevents steady-circle averages from hiding a wrong-way entry request. |
| Existing raw commands | Compute raw steering and acceleration errors at all existing groups using the unchanged 0.01/0.25 MAE and 0.03/0.75 p95 limits. Preserve the pass or fail result. Do not call a geometry-qualified policy raw-command qualified. |
| Visual dependence | Repeat the same frozen scenes with detector observations removed and with scene-matched token swaps while holding proprioception and route fixed. Require a loss increase above 10 percent and five times repeat noise, then inspect branch semantics. Sensitivity alone is insufficient. |
| Four temporal phases | Replay every saved prefix sequentially and score phases 0, 1, 2, and 3 separately. History warp, reset, and missing detections must use the same implementation in training and runtime. A phase-zero pass cannot substitute. |
| Native integration | Preserve full saved-input replay at 1e-5 tolerances, the complete pilot, rendering and contact audits, and all unchanged physical completion gates. No final case opens during selection. |

Choose the deterministic geometric planner if it clears these screens. It requires no route-association fit and is easier to inspect. Choose the learned corridor only if it fixes a repeatable topology or recovery-direction failure on the same frozen tokens while preserving the other screens. If both fail on the perfect-perception diagnostic, resolve the route or controller definition before collecting more images. If both pass with perfect perception and fail on measured detections, improve perception or calibrated history instead of training a larger actuator head.

A missing or uncertain corridor cannot silently become a straight path at speed. Record the unsupported condition and apply a predeclared bounded braking behavior. Any resulting timeout still fails the mission. Existing `src/fsop/eval/geometric.py:54-57` returns a forward plan when no path exists, so that baseline fallback is not sufficient evidence of a safe new controller.

The main tradeoff is explicit. The geometric planner exposes cone pairing and topology rules. The learned corridor can infer associations from imperfect cone sets, but can also memorize the repeated skidpad layout and invent unsupported far geometry. Route-swap, cross-episode, perfect-perception, and recovery-direction checks target those failure modes before any new simulation.
