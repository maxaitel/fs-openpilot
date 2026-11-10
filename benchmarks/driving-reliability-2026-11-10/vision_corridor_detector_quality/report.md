# Full TRAIN detector quality

The recorded detector output is useful evidence for a streaming corridor prototype, but same-frame, same-range blue/yellow support has long gaps on skidpad. This is not a closed-loop result and cannot establish zero-contact performance. Source protocol was fixed before scoring in `protocol.md`. Run `python3 .work/lab/vision_corridor_quality_v1/score.py` from the worktree. `results.json` contains phase, class, range and continuation breakdowns. `per_frame.jsonl` contains the count and confidence bands in every frame. Two reruns produced identical output hashes.

All 286 skidpad and 287 trackdrive stored frames were scored. Driving has 236 and 237 frames, respectively. Each appended terminal continuation has 50 frames. The frame ticks are 4 physics ticks apart and the cadence is 5 Hz. Source metadata places the continuation boundary at tick 943 and 945, respectively. The source recordings had no contact events, but those recordings use the source controller, not the proposed planner.

Driving association below counts exact-color matches within the strict 0.75 m Hungarian gate divided by all predictions of that class in the range. Median and p95 are distances in metres conditional on exact-class matching inside that gate. They cannot describe the error of unmatched predictions or bound error. Range is the predicted radial distance. Matching uses all reference cones, including those outside the camera field of view. It is not a recall metric. Blue and yellow match against their combined pool, and orange and big orange against their combined pool. The color matching and gate follow the earlier scorer.

| Run | Predicted class | Range m | Correct / predicted | Conditional median / p95 m |
|---|---|---:|---:|---:|
| skidpad | blue | 0–5 | 39 / 64 | 0.121 / 0.681 |
| skidpad | yellow | 0–5 | 39 / 65 | 0.122 / 0.655 |
| skidpad | blue | 5–10 | 195 / 200 | 0.047 / 0.183 |
| skidpad | yellow | 5–10 | 194 / 201 | 0.046 / 0.166 |
| skidpad | blue | 10–20 | 241 / 330 | 0.196 / 0.660 |
| skidpad | yellow | 10–20 | 241 / 328 | 0.193 / 0.588 |
| trackdrive | blue | 0–5 | 58 / 67 | 0.196 / 0.680 |
| trackdrive | yellow | 0–5 | 27 / 38 | 0.271 / 0.682 |
| trackdrive | blue | 5–10 | 225 / 225 | 0.045 / 0.143 |
| trackdrive | yellow | 5–10 | 200 / 201 | 0.040 / 0.175 |
| trackdrive | blue | 10–20 | 441 / 498 | 0.097 / 0.569 |
| trackdrive | yellow | 10–20 | 379 / 450 | 0.122 / 0.560 |

Orange predictions are not boundary-color predictions. For skidpad at 0–5, 5–10 and 10–20 m, ordinary orange correct/predicted counts are 18/38, 111/160 and 390/614, and big orange counts are 13/27, 45/49 and 34/49. Trackdrive ordinary orange counts are 0/2, 0/5 and 0/35, and big orange counts are 7/9, 33/33 and 18/66. Class mismatches and conditional medians and p95 for each orange class are in `results.json`.

Both-color support means at least one forward x>0 prediction of each boundary color in the same radial bin. It does not require the predictions to correspond to a real matched pair. During skidpad driving, support was missing in 233/236 frames at 0–5 m, 201/236 at 5–10 m and 163/236 at 10–20 m. The longest missing streaks were 197, 40 and 35 consecutive frames, corresponding to sampled time spans of 39.2, 7.8 and 6.8 seconds. Trackdrive missed support in 222/237, 58/237 and 35/237, with longest streaks of 48, 9 and 10 frames, or 9.4, 1.6 and 1.8 seconds. Those intervals are sampled spans, not interpolated continuous durations.

At 5–10 m on skidpad, the source crossing times partition the driving frames into entry 1/17 missing, right lap 1 44/49, right lap 2 46/49, left lap 1 46/50, left lap 2 45/49, and exit 19/22. These phases are scorer-only labels and cannot be fed to the planner. All 50 skidpad terminal frames lack both-color support in each range. Trackdrive terminal frames lack it in all 50 at 0–5 m but none at 5–10 or 10–20 m. The stopped continuation is separated because repeated views can dominate counts.

Confidence-band counts are descriptive of the already decoded detections, with no additional threshold. For example, skidpad driving 5–10 m includes 59 detections below 0.5, 57 from 0.5 to below 0.75, and 494 at least 0.75 across all four classes. Per-frame distributions and phase summaries are in the results. Confidence does not certify correctness. These findings motivate evaluating temporal persistence and one-sided or otherwise incomplete support in the separate prototype. They do not show that such a planner will succeed.

The scorer validates copy hashes, producer before/after hashes, producer output hashes and metadata hashes for every consumed compact reference. It repeats the checks after calculation. Raw camera arrays remain at their source and were not read. Reference states and cone geometry are used only for scoring. The 0.75 m gate is not a safety margin. The correlated frames and these two TRAIN episodes are not independent reliability samples, and no uncertainty or safety bound is claimed.
