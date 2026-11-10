# Baseline repeat divergence investigation

The goal is to reduce physical cone hits and distinguish training effects from evaluation variation. This read-only investigation located the first observed divergence before the evidence writer receives its first record. It did not run GPU work, modify runtime code, change jobs, or stop processes.

Throughput checkpoint is n/a for a read-only investigation.

## Evidence and method

The two runs are `runs/pi_latest_heavy1` and `runs/development_pi_trial1/baseline/heavy_skid` under `/home/maxaitel/fsop-isaac-lab-training` on Spark. Both use the PI controller. The parent separately verified identical model tensors.

`compare_early.py` is a reproducible CPU-only analysis of complete manifest entries. It verifies each sampled NPZ SHA against its manifest before reading the record. It compares frames 0 through 10, then 20, 50, and 99. The JSON result is `early_comparison.json`. The first 100 manifest input hashes were also compared. No GPU model inference is used.

Run it from this worktree with:

```sh
ssh max-spark 'docker exec -i fsop-isaac-lab-training /isaac-sim/python.sh -' < .work/lab/pi_repeat_divergence/compare_early.py > .work/lab/pi_repeat_divergence/early_comparison.json
```

`scene_source_check.txt` records the full textual scene diff and verifies the archived runtime files against their metadata and currently installed Spark copies. The five runtime files checked were env.py, evaluate.py, runtime.py, evidence.py, and rendering.py. All matched in both runs. The only differing source inventory key is src/fsop/train/lab_train.py. Metadata differs in checkpoint wrapper hash and size, evidence writer type, scene hash, source archive hash, and source inventory.

## First divergence

At frame 0, both runs have exactly equal full vehicle state, all body pose and velocity arrays, actual cone position and orientation arrays, camera matrices, nominal calibration, and progress. Their raw RGB differs before image postprocessing and before the writer is created or receives its first append. The evaluator captures this initial image in drive.reset before export and before constructing the writer.

| Measurement at frame 0 | Road | Wide |
| --- | ---: | ---: |
| Raw RGB mean absolute difference, 0 to 255 scale | 1.665939 | 1.301682 |
| Maximum channel difference | 59 | 40 |
| Fraction of unequal channel values | 0.763280 | 0.554723 |

The first action also differs. Old steering is -0.0007916168 and new steering is -0.0005598402. Old acceleration is 3.8420145 and new acceleration is 3.8947783. Frame 1 has a maximum body position difference of 0.0000038147 m. Frame 99 has a maximum body position difference of 0.3530636 m. None of the first 100 input hashes match.

This establishes raw image divergence as an initiating boundary. It does not establish whether RTX sampling, renderer startup state, or another rendering cause produces the raw difference. It does not prove that later writer scheduling has no influence.

## Physics scheduling

The inspected source does not schedule physics from elapsed wall time.

- LabDrive._advance applies an action and calls _physics_step exactly six times. Each step calls sim.step(render=False) at a configured 1/120 s and scene.update(1/120).
- Isaac Lab SimulationContext.render sets /app/player/playSimulations false during app.update, then restores true.
- capture calls sim.render followed by replicator.orchestrator.step with 16 subframes and delta_time=0.0. The installed Replicator implementation disables timeline auto-update for non-None delta_time and documents zero as no timeline advance. Its source excerpt is saved in replicator_step_source.txt.
- AsyncEvidenceWriter deep-copies each record, then one worker serializes and hashes it. That worker contains no simulator calls. The first image already exists before that worker is constructed.
- At all 14 sampled frames in each run, the cached state XY and yaw exactly match body XY and yaw queried after capture and inference. The maximum discrepancy is 0.0 in both quantities.
- The existing runs/gates/camera_native_retry/verification.json reports pass with capture_sim_dt equal to 0.0 for all six samples. This is supporting gate evidence, not instrumentation of the two full episodes.

No uncontrolled physics advancement was observed. Existing records do not include actual simulator time or its physics step index, so they cannot prove every interval had exactly the expected simulator-clock advancement. Reported state.t is frame/20 and cannot itself verify that property.

## Scene differences and limits

The complete scene diff has five hunks. Three change viewPickingId. One changes Replicator reference-time scheduling values. One changes an unauthored omni:rtx:post:aa:op token to an explicitly authored "none".

The AA token belongs to /Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0. Its camera is /OmniverseKit_Persp and its resolution is 1280 by 720. Runtime launch disables that viewport's updates. It is not either of the two 512 by 256 sensor products. The sensor products differ only in viewPickingId in the textual diff. Therefore this token is not evidence of changed effective sensor AA, and it must not be presented as the cause. The effective settings snapshot does not capture every possible renderer value.

The parent reports the old baseline ended at 79.8 simulated seconds with 12 hits. The new baseline ended at 46.3 seconds with 16 hits and invalid_route. These are unequal exposure durations and single runs. The failed candidate run provides no completed physical comparison. The observed baseline variation alone is enough to reject a causal claim that a one-run historical-versus-new delta measures training improvement.

## Minimal next verification

After disk capacity is resolved and the parent schedules GPU use, run a short two-process fixed-action probe under the same case and writer mode. Log actual simulator time, physics step count, pose, raw input hashes, effective global and per-sensor render settings, and capture frame identity before and after capture, evidence append, and six physics steps. Include one deliberate wall-time delay with no simulator calls. This isolates the disputed wall-clock mechanism without allowing policy feedback to amplify image noise.

The probe should show zero simulated time and pose change across render, writer, and idle boundaries, then exactly 0.05 simulated seconds and six physics steps across the control advance. Compare initial raw images across cold starts. If effective sensor settings match but initial pixels vary, classify it as observed renderer repeat variation. Establishing its exact renderer mechanism may require another controlled probe.

For model quality, retain the same writer mode and effective settings for baseline and candidate. Complete repeated fresh-process evaluations before using a physical difference to accept a checkpoint. Use at least three runs per checkpoint as a small repeatability check, report every result and duration, and do not claim a statistical reliability estimate from three trials. More runs are needed if the result remains smaller than the observed spread. No runtime fix is justified by this investigation alone.

Explain the Number changed the decision by withholding a training-improvement claim when baseline repeats diverged. Fix Root Causes changed the investigation order by locating raw frame-zero divergence before attributing later physical outcomes to evidence scheduling.
