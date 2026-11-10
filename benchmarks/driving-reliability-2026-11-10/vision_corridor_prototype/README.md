# Streaming cone corridor prototype v1

The goal is a vision driver that completes trackdrive and R,R,L,L skidpad with zero physical contacts. This throwaway CPU prototype fails its feasibility screen and must not be used for a physical launch.

The frozen streaming planner returns `unknown` and a negative acceleration request on all 706 selected frames. It produces no accepted local path. The zero accepted-path collision count is vacuous. It is neither a geometry pass nor a raw-command pass.

## Measured decision

Every saved precontact frame from both rejected pilots was replayed. The new detector records add every stored frame from the full zero-contact TRAIN skidpad and track missions, including terminal braking. The selection and source hashes were frozen before the first saved-data planner replay in [freeze.json](freeze.json). No recipe or inference source changed after the results.

| Recorded trajectory | Frames | Streaming supported | Existing midline available | Midline lookahead lateral MAE / p95, m | Midline padded footprint conflicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial rejected pilot, strictly before contact | 80 | 0 | 77 | 0.584 / 1.616 | 61 |
| Recovery rejected pilot, strictly before contact | 53 | 0 | 53 | 0.608 / 1.196 | 53 |
| Full stored TRAIN skidpad | 286 | 0 | 214 | 0.454 / 1.286 | 129 |
| Full stored TRAIN track | 287 | 0 | 287 | 0.118 / 0.376 | 12 |

The old midline retains its existing 25 m range, origin anchor, greedy pairing, and one-sided offset. Both methods use the same local diagnostic controller. Its path-available count is not an assertion of supported perception. Lateral errors above are conditional on a selected target. Footprint counts measure sampled rectangles along proposed paths at recorded states. They are not physical contacts or a dynamics rollout.

The same-frame-only gate ablation also returns zero supported paths. The streaming failure occurs before route selection. There are 366 frames with no eligible gates. The other 340 have gates but no eligible nearfield seed. Near retained tracks exist in 488 frames, but the first recipe also requires a recent two-sided gate near the ego origin. Tracks lose gate eligibility after 0.3 s without a matching observation even though the memory keeps them for 1.5 s. Color-conflicted tracks appear in 330 frames. These are all-frame descriptive counts from [failure_accounting.json](failure_accounting.json), not a threshold sweep.

This evidence rejects v1's support and bootstrap rules. It does not establish that geometric topology is intrinsically inadequate. No recorded frame reaches a candidate path, so learned association alone has not been shown to fix this failure under the same support contract. A different nearfield or one-boundary proposal needs its own frozen recipe and comparison. Do not add a launch override that accelerates through this planner's `unknown` status.

The full stored skidpad covers entry, right lap one, right lap two, left lap one, left lap two, exit, and terminal braking with 17, 49, 49, 50, 49, 22, and 50 frames respectively. The full stored track has 237 driving frames and 50 terminal frames. All full-mission images are 5 Hz and control phase zero. Only the two short pilots supply all four 20 Hz phases.

## Interface and inference boundary

[planner.py](planner.py) uses only the Python standard library. Its input is an immutable `Observation` containing these values.

- `timestamp` is a monotonic timestamp in seconds.
- `cones` is a tuple of `ConeDetection(x, y, cone_class, score)` values. Coordinates are meters in the vehicle reference frame, with x forward and y left.
- `motion` contains measured body-frame vx, vy, and yaw rate.
- `route_history` contains provided `RouteEvent` values for left, right, or exit. Repeated history is allowed. Future and conflicting simultaneous events return unknown.

The result contains `status`, an offset local `path`, `support_length_m`, a `ControlRequest`, and diagnostics. A path's first point is the projection of the ego origin onto the selected gate segment. The path is not forced through `(0, 0)`. A synthetic offset lane produces a path beginning at `(0, 0.25)` and corrective left steering.

```python
from planner import CorridorPlanner, ConeClass, ConeDetection, Motion, Observation

planner = CorridorPlanner()
result = planner.step(Observation(
    timestamp=0.0,
    cones=(ConeDetection(6.0, 1.5, ConeClass.BLUE, 0.9),
           ConeDetection(6.0, -1.5, ConeClass.YELLOW, 0.9)),
    motion=Motion(vx=0.0, vy=0.0, yaw_rate=0.0),
))
assert result.status == "unknown"
assert result.control.accel_mps2 < 0
```

The planner does not receive world position, heading, progress, centerlines, actual cone maps, teacher outputs, or future data. [inputs.py](inputs.py) whitelists the allowed fields from both saved formats. The inference module does not import either the adapters or scorer.

The route latch records the latest supplied maneuver. It does not infer a lap count or mission phase. Pilot pulses are the saved `commands.applied`. Full-mission desires are the saved 25 by 8 max-pooled histories. Their adapter assigns an event to each provided bin's end and retains its 0.2 s time resolution. These times are not reconstructed raw 20 Hz pulse times. The route pulses originate from simulator progress and are an explicitly provided external input.

The parent runtime owns AS `mission_brake`. This prototype never invokes it. Unknown braking must remain effective when the parent wraps the local request.

## Frozen first recipe

[recipe.json](recipe.json) owns all planner and controller thresholds. Its SHA256 is `e900be8f7b3f76996dd2a4d43b9291607f99d66eabc52067e4c8613aa65d5479`.

A cone needs a score of at least 0.55 and an observation at 5 to 10 m before it can support a boundary. Closer observations can refresh an associated retained track or act as obstacles. They do not create support by themselves. Observations beyond 10 m do not create support. Score is a ranking and filtering value, not calibrated certainty.

The memory warps points with the measured body twist. It uses the average of the previous and current measured velocities and yaw rates over each interval. Large time gaps and nonmonotonic time reset memory and return unknown for that tick. Conflicting nearby colors become unknown obstacles. Small and large orange classes share a lane color.

Gate search accepts blue-left/yellow-right pairs and opposite orange lane endpoints with widths from 2.4 to 4.5 m. Blue-blue and yellow-yellow gates are forbidden. The search limits graph edges by length, heading, width changes, and curvature. Its beam width is 24 and its depth is eight gates. The recipe caps detections, tracks, and gates. Capacity overflow returns unknown. Prior path continuity and the route latch can resolve candidate ambiguity only after a local supported path exists.

The controller uses rear-axle pure pursuit with a 1.55 m wheelbase and 0.75 m rear offset. Lookahead is `1.8 + 0.25 * max(vx, 0)` m. The speed cap is 5 m/s. Curvature and available stopping support further restrict the target speed. Unknown requests zero target speed and braking, with no straight-at-speed path. The footprint uses vehicle width 1.4 m, front extent 1.6 m, and rear extent 1.3 m.

The 0.15 m cone radius, 0.20 m clearance pad, 0.3 s response assumption, and 3 m/s² braking assumption are explicit engineering choices. They are not validated uncertainty, drift, actuation, or stopping bounds. Sampled footprint clearance assumes reference-path tracking and does not certify reachable or continuous swept clearance.

## Scoring and provenance

[scorer.py](scorer.py) is the only prototype module that imports privileged mission geometry. The replay sends only observations to the planner and uses ground truth afterwards for scoring. Input evidence manifests, compact array hashes, and copied canonical-track receipts are verified in [replay.py](replay.py).

Canonical teacher queries reproduce the saved source teacher references within these maximum absolute command residuals.

| Recorded trajectory | Maximum reconstruction residual |
| --- | ---: |
| Initial rejected pilot | 3.60e-13 |
| Recovery rejected pilot | 6.94e-18 |
| Full stored TRAIN skidpad | 4.25e-13 |
| Full stored TRAIN track | 3.89e-16 |

The full-mission check uses original `teacher.npy` flags and every preterminal teacher control tick. Track reconstruction uses the separately copied actual Spark cache, not the different local generated cache. The skidpad factory reconstructs the unrounded geometry. Scoring can fall back to the actual future driven trajectory if canonical reconstruction fails, but that fallback was not required for these results.

Both full stored missions include learner intervals. Stored controls from those intervals are not mislabeled as teacher commands. The scorer queries the canonical teacher at each recorded state, including learner states, and reports terminal frames separately. All-row raw-command errors include unknown braking rows. Every variant fails the unchanged raw-command screen. The heading-based wrong-branch diagnostic is only a proxy and does not qualify semantic branch or lap selection.

[rows.json](rows.json) contains every result, local controller request, track diagnostic, and isolated score. [result.json](result.json) contains counts, phase groups, unknown intervals, correction direction, raw-command errors, and footprint margins. Both replay runs produced byte-identical outputs, as recorded in [repeat_receipt.json](repeat_receipt.json). Repeat spread is zero for these deterministic saved inputs. The samples are correlated and do not estimate independent-scene reliability.

Seven tests passed before the first saved-data replay. [test_planner.py](test_planner.py) covers rigid ego warp, retained offset paths, orange lane gates, same-color rejection, reset and dropout braking, route-event latching, no input mutation, far-outlier rejection, and deterministic replay.

Run from the worktree root.

```sh
python3 -m unittest discover -s .work/lab/vision_corridor_proto_v1 -p 'test_*.py' -v
PYTHONPATH=src:. python3 .work/lab/vision_corridor_proto_v1/replay.py
PYTHONPATH=.work/lab/vision_corridor_proto_v1 python3 .work/lab/vision_corridor_proto_v1/explain_failure.py
```

The replay refuses changed frozen source or changed existing result files. No existing source or dependency was edited. No model inference, GPU, simulator, SSH, training, validation case, or final case ran in this prototype task.

Model the Domain shaped the immutable observation and explicit support result. Test Behavior, Not Implementation shaped the synthetic offset, reset, and latch tests. Explain the Number kept unknown frames in coverage and command denominators and prevented zero accepted-path conflicts from becoming a success claim.
