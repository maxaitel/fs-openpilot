# Constant-offset vision plan prototype fails

The goal is reliable vision driving on full trackdrive and full RRLL skidpad with zero physical cone contacts. This frozen throwaway prototype does not advance. On the 133 authorized TRAIN precontact frames, its constant boundary-offset correction worsens average nearfield geometry and steering direction. Both episode screens and all eight episode-by-control-phase screens fail. No controller parameters changed after the first replay. No physical trial, GPU job, model inference, training, SSH call, or final-case access occurred.

The candidate retains the original NN plan as a route prior. It transports scored 5 to 10 m cone observations using measured body motion, fits a bounded weighted-median lateral offset from nominal boundary width, and uses rear-axle pure pursuit. It shifts the first path point too. The speed request has a fixed 5 m/s cap, a curvature limit, and a declared stopping heuristic. Unknown support brakes. The reference variant uses the identical resampled NN prior, observations, support state, and speed rule without the offset. The third reference is the recorded raw actuator head. [protocol.md](frozen_source/protocol.md) states every assumption and screen. [config.json](frozen_source/config.json) records every constant.

## Measured result

The original model state hash is `14feb933b4399f4e25131acaf60aa4abae1dad7f1421303fc66a4735fc72f652`. Selection includes direct pilot frames 0 through 79 and recovery pilot frames 0 through 52. The modulo-four phase counts are 34, 33, 33, and 33. Canonical route labels contain 113 entry frames and 20 first-right-loop frames. There is no full RRLL or trackdrive coverage here.

| Metric across all 133 frames | Uncorrected PP | Corrected PP |
| --- | ---: | ---: |
| All-candidate lateral target MAE | 0.3054 m | 0.4031 m |
| All-candidate lateral target p95 | 1.5236 m | 1.3657 m |
| Supported-only lateral target MAE | 0.4149 m | 0.5489 m |
| Supported-only lateral target p95 | 1.6156 m | 1.8098 m |
| Geometric PP steering MAE | 0.04884 rad | 0.07633 rad |
| Teacher steering-direction agreement | 35/41 | 27/41 |
| Supported heading-error branch proxies | 4/97 | 3/97 |
| Supported sampled GT footprint conflicts | 43/97 | 65/97 |
| Predicted sampled footprint conflicts | 40/133 | 57/133 |
| Positive acceleration requests | 9/133 | 1/133 |

Support exists on 97 of 133 frames, or 72.9 percent. Both prefixes have no boundary support on frames 3 through 20. This fails the frozen 90 percent support screen. Unknown rows request negative acceleration in every case. The corrected candidate brakes on 132 frames. Its stopping-distance heuristic has a shortfall on 120 frames, so these results do not support launch or route completion.

| Prefix | Supported | Uncorrected lateral MAE / p95 | Corrected lateral MAE / p95 | Corrected direction |
| --- | ---: | ---: | ---: | ---: |
| Direct pilot | 62/80 | 0.3350 / 1.5114 m | 0.4464 / 1.8677 m | 20/26 |
| Recovery pilot | 35/53 | 0.2606 / 1.5826 m | 0.3376 / 0.9075 m | 7/15 |

The raw old actuator head has steering MAE 0.05544 rad, p95 0.20897 rad, and direction agreement 34/41. Corrected requested-steering MAE is 0.07618 rad. Its acceleration error is MAE 4.8103 m/s² and p95 8.0 m/s² because it usually brakes. This acceleration comparison is diagnostic only. The new controller is not evaluated as passing the old learned-actuator qualification.

The geometry screen required lateral MAE at most 0.10 m and p95 at most 0.25 m on all candidate paths and on supported paths. It also required improvement over the identical uncorrected PP baseline, no direction regression, no supported wrong-heading proxy, and no supported sampled footprint conflicts. The aggregate fails every component. A lower aggregate p95 alone cannot justify advancement when average error, supported-tail error, direction, and clearance regress. [result.json](frozen_outputs/result.json) contains every episode and control-phase summary. [rows.json](frozen_outputs/rows.json) contains each proposal, support residual, track, raw command, score, and per-cone margin.

## Why the correction fails

The post-result [diagnose.py](post_result_diagnosis/diagnose.py) reads the same prefixes and uses privileged geometry only for attribution. It does not modify the controller or define a new pass screen. Its output is [attribution.json](post_result_diagnosis/attribution.json). Nearest-cone matching at 0.75 m is a diagnostic heuristic, not a proof of identity or detector recall.

All 97 median-support tracks have a nearest GT cone within 0.400 m. Their nearest-distance MAE is 0.1167 m. Replacing the predicted median cone position with its nearest GT position changes the projected boundary residual by MAE 0.0371 m and p95 0.0956 m. These measurements suggest localization and short-term motion transport are not the dominant error for the selected median observations. They do not evaluate every tracked obstacle or establish uncertainty calibration.

Of those 97 matches, 91 have compatible color classes. Twenty of the 91 differ by more than 0.5 m from the assumed signed 1.5 m boundary on the canonical local route. None of these twenty corrections improves nearfield absolute error. A cone's color and distance from the NN path do not establish that it belongs to the boundary of the current lane.

Concrete scored examples distinguish the mechanisms.

- Direct frame 0 selects an observation classified as orange whose nearest GT cone is yellow. Its position error is 0.1084 m. The clipped +0.75 m offset changes lateral error from -0.00065 m to +0.74935 m. This is a class/side-assignment failure despite a useful position.
- Direct frame 21 selects a correctly colored yellow cone with position error 0.2534 m. On the canonical entry route its signed boundary residual is -1.5465 m. Treating that cone as a nominal 1.5 m boundary shifts lateral error from +0.00610 m to -0.74390 m. The prior and canonical normals differ by only 0.00378 rad. This example implicates boundary membership and the constant-width interpretation, rather than an NN tangent error.
- Direct frame 79 selects a correctly classified orange cone localized within 0.02191 m. Its canonical signed boundary residual is +2.4233 m. The resulting +0.75 m shift increases lateral error from +1.6737 m to +2.3598 m. Accurate cone position does not make the selected route boundary correct.
- Recovery frame 52 selects a correctly classified yellow cone with a canonical boundary residual of -0.2097 m. Its NN and canonical local normals differ by 0.5463 rad. The shift improves lateral error from -2.2783 m to -1.6093 m but cannot repair the prior shape. This supports a distinct NN shape/projection limitation.

Among the remaining 71 class-compatible observations with canonical boundary residual at most 0.5 m, only 35 improve nearfield absolute error. Twenty-nine move the target in the wrong direction and seven overshoot. Across all supported frames, 35 improve, 42 move in the wrong direction, and 20 overshoot. The offset clips on 37 frames. The constant far-boundary offset does not reliably represent the correction needed at the near lookahead point.

This attribution does not isolate a universal width error. Large canonical residuals can indicate a different branch's cone, side mistakes, endpoint projection, or genuine route-shape mismatch. The direct entry example shows that simply changing the nominal width would conflate different nearby boundaries. The recovery example shows that correcting only a scalar offset is insufficient even when the matched boundary is plausible. No width, range, weight, or clipping sweep was performed.

## Next falsifier

Do not run this controller physically or extend its NN-plan inference to all 573 TRAIN frames. First test boundary geometry as an input representation, independent of the NN path normals. Use the same sealed prefixes and a separately frozen, offline-only rule that groups measured-motion cone tracks into a short continuous line or arc. Estimate tangent and lateral offset from those boundary observations. Use the NN route prior only to choose among complete local hypotheses. Do not let a lone correctly colored cone establish membership in the current lane.

The smallest useful check is whether a local boundary chain can reconstruct the near lookahead target when the NN path and boundary geometry disagree. Predeclare the line/arc support and degeneracy rules, transport limits, and any nominal-width assumption before the first output. Keep unknown results in the coverage denominator. Diagnose entry bootstrap separately because valid near entry-lane cones may lie outside the current 5 to 10 m support band. A changed range would require its own detector-quality justification, not a retry chosen from this result.

Reject the representation if it still treats the observed distant circular boundary as the entry-lane boundary, if it cannot establish local tangent without the NN normal, or if it cannot meet the same 0.10 m MAE and 0.25 m p95 geometry screen with declared coverage. Ground-truth associations and lane membership belong only in the scorer. This is a topology and local-geometry test, not another scalar-parameter sweep. Passing would earn a fresh controller design and subsequent frozen physical TRAIN comparison. It would not qualify VAL, full RRLL, or the final case.

## Verification and limits

Five synthetic behavior tests pass. They cover measured SE2 transport, memory expiry/reset, nonzero initial path offset and color sign, persistent unknown class conflicts, and braking with unknown support. The canonical teacher reproduces cached commands to maximum absolute error 3.60e-13 on the direct prefix and 6.94e-18 on recovery. `read_evidence` verifies every selected frame and image hash. The completion receipt binds the complete source manifest and summary.

Source, config, protocol, selection, and scorer dependency hashes were saved before the first replay. A separate process repeated the replay and compared complete serialized outputs. The outputs are byte-identical and all frozen source hashes still match. [repeat_receipt.json](frozen_outputs/repeat_receipt.json) records the checks. Result SHA-256 is `5178985cde389e87bf0c7cd4159ec8a1737427e1ba32fe62bf7c14b5328a653a`. Rows SHA-256 is `38fc1a38d2c522401804d6021f038b126dfa109e4bc02d13d5376f542c9c29ea`.

Run the frozen replay from the worktree with `PYTHONPATH=src:. .venv/bin/python .work/lab/vision_corrected_plan_proto_v1/replay.py`. It refuses changed inputs or sources and preserves different existing outputs rather than overwriting them.

All sampled margins describe proposed reference-path geometry on recorded states. They do not describe a physically driven trajectory or guaranteed swept volume. The saved vehicle states came from different direct-actuator controllers and would change under these requests. The two correlated prefixes are one skidpad scene and only entry plus part of the first right loop. Branch-heading proxies do not qualify RRLL semantics. The runtime uses measured vx/vy/yaw rate and camera-derived predictions. The original model's supplied route commands are external inputs. No on-car lap counter, independent AS completion signal, or safety guarantee is established.

Model the Domain put the runtime inputs behind a whitelist. Explain the Number fixed the selection, thresholds, and paired comparison before replay. Prove It Works required sealed evidence, canonical reconstruction, and identical repeated artifacts. The scratch code remains a throwaway instrument. Existing production and frozen candidate sources are unchanged.
