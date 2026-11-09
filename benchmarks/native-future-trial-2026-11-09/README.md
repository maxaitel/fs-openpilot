# Rejected native-future action-target experiment

The goal was native Isaac training that reduces cone contacts and timeouts with the full model and auditable debug videos. This sixth experiment completed, but it did not produce a qualified model.

The 500-update job took 475.7705 seconds. Median sampled whole-GPU utilization was 86%, median power 48.95 W, and peak sampled host unified RAM use was 8.1213 GiB. The full 31,154,802-parameter architecture was retained; 6,949,412 policy parameters were trained. The job used FP32, batch 128, eight-frame history and the unchanged v4 data. These are one-job measurements, not a peak-FLOPS claim or repeated speed benchmark. Collection, physical evaluation and video export are excluded from training time.

All saved checkpoints failed the frozen gates. Step 500 had 31 failed checks, including 25 acceleration/secant, four episode steering, one yaw-interval and one initial-velocity check. These checks are correlated. The final checkpoint was tested only under the predeclared rejected-model diagnostic policy. No checkpoint was promoted and no fresh test was opened.

| Exposed case | Model | Result | Simulated seconds | Physical contacts | Off-course |
|---|---|---|---:|---:|---:|
| Track | Original PP baseline | Timeout | 120.00 | 11 | 10 |
| Track | Unchanged initializer, tangent follower | Finished | 56.55 | 1 | 1 |
| Track | Rejected step 500 | Finished | 56.95 | 2 | 1 |
| Skidpad | Original PP baseline | Finished | 46.35 | 6 | 0 |
| Skidpad | Unchanged initializer, tangent follower | Invalid route | 12.60 | 8 | 0 |
| Skidpad | Rejected step 500 | DNF | 76.25 | 25 | 6 |

Each model completed one of two exposed cases. This is not a population completion-rate estimate. The candidate's skidpad result regressed from the fifth experiment's 17 contacts and five off-course events. Baseline repeats vary materially. A timed-out baseline does not establish a lap-time speedup. CPU exports and saved-record diagnostics overlapped parts of evaluation; the 6.237-second GPU replay ran between the two physical plan jobs.

The 120-frame pilot inference replay passed. It recorded no contacts or off-course events in six simulated seconds. Both pilot videos fully decoded at 20 fps with 160 frames. Pilot checks establish integration, not mission success.

[Saved-anchor analysis](residual/analysis.md) reproduces the validation numbers and localizes the remaining errors. Velocity and acceleration errors compensate in the controller, so low action error can hide an inconsistent trajectory. The fixed output intervention improves consistency but worsens the current training objective on 11 of 14 tail anchors. It leaves steering unchanged and is not a deployed fix.

The [measured-yaw ablation](measured_yaw_validation_ablation.md) rejects another tempting shortcut. Replacing predicted initial yaw rate with raw measured rate worsens steering loss in six of eight validation scenes. The exact body-rate audit distinguishes instantaneous heading rate from a centered finite difference. No new controller mode or seventh training run was introduced from these descriptive results.

The hash manifest preserves exact training, selection, pilot and diagnostic artifacts. Raw replay arrays remain both locally and on Spark, identified by their hashes. Full camera evidence and checkpoints remain on Spark. The source scripts use their original repository-relative working paths; reproducing them requires the bound data and checkpoints, not just this compact archive. The local training report, development report and twelve full-episode videos live under `runs/native_future_trial1_training_report`, `runs/native_future_trial1_development_report` and `runs/development_media/native_future_trial1` in the active worktree.

The `development/` directory retains all six episode summaries, producer artifact/video audits, render bindings, completion receipts, both frozen plans and manifests, aggregate results, strict same-controller comparison proof and contact/evaluation plots. Its `sha256.json` binds the local HTML reports and all 12 MP4s. Those large media files remain outside Git in `runs/development_media/native_future_trial1/`. All report and media hashes were verified before archiving.
