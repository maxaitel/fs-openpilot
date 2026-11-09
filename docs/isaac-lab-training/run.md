# Native Isaac Lab workflow on Spark

`fsop train` now defaults to `configs/train/spark_isaac_lab.yaml`. It trains the full existing image/temporal-plan model from native Isaac Lab episodes. Explicit legacy configs remain available for reproducing previous experiments. The change is the data and simulation backend, not a replacement with a small reinforcement-learning policy.

The migration runs on Spark, but no newly trained native checkpoint has passed every promotion gate. The sixth experiment completed training and saved-input verification, then failed the exposed skidpad with 25 physical contacts. Native results must not be confused with the historical custom-simulator skidpad benchmark. Fresh test scenes remain unopened.

## Runtime boundary

The tested Spark renderer/trainer uses Isaac Lab 2.3, Isaac Sim 5.1, Python 3.11.13, PyTorch 2.9.0+cu130 and NumPy 1.26.0. Its task container is `fsop-isaac-lab-training`; the immutable image is `fsop-isaac-eval:verified` (`sha256:7092a76a990f364ea684b64a48681e9a7c1992a8d4af2edfc758e640b57668bb`). The checkout is mounted at `/work`, with `PYTHONPATH=/work:/work/src:/work/isaac_eval`. Prepared original model graphs and appearance assets are mounted read-only.

The running container has package overlays that are not part of the immutable image. In a fresh container based on that image, install the two runtime dependencies before invoking the CLI:

```sh
/isaac-sim/python.sh -m pip install 'typer==0.27.3' 'omegaconf==2.3.0'
```

The first launch failed before any updates because `typer` was missing. OmegaConf 2.3.0 satisfies the installed Hydra requirement of less than 2.4. The initial 2.4 installation was replaced before the successful launch. These overlays disappear when the container is recreated. Use the image's `/isaac-sim/python.sh`. Do not install this repository's legacy all-in-one `pyproject.toml` dependencies into the renderer: they include NumPy 2 and Rerun, while this tested camera runtime uses NumPy 1.26. Native collection and training use the source checkout directly. The separate CPU exporter image `fsop-isaac-stress:verified-export` has NumPy 2.4.6, Rerun 0.42 and FFmpeg 6.1.1. This separation is intentional and is part of the tested deployment.

The full model's existing FP32 prepared vision/policy graphs and original complete checkpoint are required. The native environment uses real Lab `SimulationContext`, `InteractiveScene` and `RigidObject` instances. PhysX advances at 120 Hz, controls at 20 Hz, and retained training images at phase-zero 5 Hz. Evaluation records every 20 Hz input and prediction. Scene queries and contact processing must stay enabled. The runtime restores the original RTX settings after Lab's renderer preset initialization; receipts preserve those settings.

## Collection and integrity

The trial uses the existing `/work/data/native/lab_v2` dataset on Spark. It contains 37 hard-linked original episodes and three replacement attempts. The original `lab_v1` collection had 40 episodes, of which 37 qualified. The failed training skidpad cases were 82003, 82007, and 82015. A four-case, 16-cell teacher grid selected pure pursuit with lookahead coefficient 0.15 instead of 0.25, at speed cap 8 and lateral acceleration limit 4. The grid tests the teacher on training cases. It is not a learned-model result.

The replacement attempts for 82003 and 82015 qualified. The 82007 replacement still had two physical cone contacts, so it contributes no anchors. The verified dataset has 7,222 training anchors across 31 usable episodes and 2,067 validation anchors across eight episodes. `membership.json` freezes 32 training and eight validation identities. Trackdrive geometry seeds are disjoint across splits. Skidpad retains its defined geometry and varies scene seeds. Fresh test and exposed regression cases are excluded from training labels.

To verify the actual v2 dataset from `/work` inside the renderer or trainer, run:

```sh
sha256sum data/native/lab_v2/manifest.json
/isaac-sim/python.sh scripts/verify_lab_dataset.py \
  --manifest data/native/lab_v2/manifest.json \
  --out /tmp/lab_v2_verification.json
```

The manifest SHA-256 is `df238ccad932d5704f7091cbff1872f580c7d8dce4951468c532e66fb8b85e37`. The verifier receipt must pass and name that same digest. Preserve the existing dataset and its per-episode source archives to reproduce this trial. Running the collector into a new directory creates a new dataset and does not reproduce this manifest byte for byte.

The teacher acts on measured native physical state. Targets use ten seconds of its actually executed future, including an actual braking continuation after scored finish. No bicycle forecast or repeated terminal padding supplies future labels. Only complete, passing, contact-free native teacher episodes contribute anchors; failed episodes remain recorded. Raw `controls.npy` are requested teacher commands before native clipping, lag and rate limits; `states.npy` also records actual filtered actuator states. Direct action losses currently supervise requested PP commands. The longitudinal loss is before the runtime mission launch/finish override, so it is not a complete deployed-action metric.

The reader checks every array digest, horizon, contact mask and split. The independent dataset verifier additionally checks complete membership, metadata seals, producer archives, regenerated track geometry, vehicle, camera and scene realization. A source archive belongs to each episode; never alter source files during an active collector or evaluator.

## Training and selection

Choose an unused output directory before reproducing the trial. For example:

```sh
sed 's|^out: .*|out: runs/lab_v2/vision_reproduction|' \
  configs/train/spark_isaac_lab.yaml > /tmp/spark_isaac_lab_reproduction.yaml
/isaac-sim/python.sh -m fsop.cli train --cfg /tmp/spark_isaac_lab_reproduction.yaml
```

The config points to the v2 manifest, `/frozen/model.pt`, and `runs/lab_v2/vision_trial1`. It specifies 1,200 updates, batch size 128, FP32, and eight historical image encodes. AMP and channels-last are disabled. The context-eight output and gradient parity gate passed for the pinned policy graph. The benchmark and fixed-input numerical checks are in `benchmarks/performance/isaac-lab-2026-11-09.json`.

The first failed launch is recorded in `runs/lab_v2/vision_trial1_telemetry.json`. The retry log is `probes/train_v2_trial1_retry.log`; its completed training telemetry is `runs/lab_v2/vision_trial1_training_telemetry.json`. Do not overwrite the existing run to reproduce it.

Both the vision trunk and temporal policy receive gradients. Auxiliary cone-head tensors remain frozen because native visibility labels are not yet qualified. Their outputs can still change when their shared vision features change, so inspect the debug videos. Historical features are recomputed from the current vision weights, with historical gradients intentionally truncated. There is no persistent stale feature cache. The default recomputes all 24 past slots. Optional `--history-context 8` skips the 16 slots that the pinned policy graph cannot read, retaining the complete input shape and full command history. This option requires both a checked graph/loaded-buffer proof and a measured output/gradient parity gate before use in a recipe.

The run saves the full model, optimizer, exact validation anchors, source archive, initialization digest, per-step losses and timing. FP32 validation uses the same episode-balanced anchors even if training autocast is enabled. The initial model is a valid selection candidate. Open-loop `selected.pt` is provisional: physical development evaluation determines whether it improves driving before fresh test results are opened. An optimizer checkpoint is not an exact resumable training session; RNG restoration/resume is not implemented.

## Evaluation and same-format evidence

Create a frozen plan as documented in `scripts/run_lab_evaluations.py`. It binds case JSON files and each complete checkpoint SHA-256. Then run:

```sh
/isaac-sim/python.sh scripts/run_lab_evaluations.py --plan runs/evaluation_plan.json --out runs/evaluation
```

The coordinator runs one native process per case, checks disk reserve, records technical failures, and resumes only fully sealed episodes with matching identities. A scored timeout is retained as a result. Technical errors are investigated separately and cannot be silently skipped. For bounded archival batches use separate frozen batch plans; resume requires the complete original evidence locally available to the coordinator.

Use `/isaac-sim/python.sh -m isaac_lab_fsop.process --help` inside the CPU exporter for independent state/referee/command auditing and the established debug/research MP4 and Rerun formats. Videos include raw model inputs, predictions, cone-head outputs, controls, commands and failures. The green diagnostic overlay remains the historical bicycle oracle and is labeled accordingly; it is not the native teacher or a model input. Physical cone contacts are reported separately from the legacy referee's cone-displacement penalties.

`scripts/report_lab_training.py` builds standalone training graphs and an HTML results index. Pass the frozen evaluation plan so absent cases remain visible. Fresh tests, development cases and exposed regressions stay separate. Sealed artifacts, audit success and video coverage are distinct checks.

## Appearance diagnosis limits

The fixed-pose appearance experiments `appearance_heavy_10s_v2` and `appearance_heavy_10s_settle2` both failed their predeclared repeatability limits. Their contrasts do not establish which visual change caused the driving failures. Treat the cause as unresolved until a repeatable experiment and closed-loop evaluation support it.

## Native action recovery

The first 1,200-update vision/policy run completed in 1,369.49 seconds. Median training step time was 1.1061 seconds and sampled whole-GPU utilization was 87%. Its provisional step-900 checkpoint passed the 120-frame saved-input inference replay but contacted one cone during that pilot. On the exposed heavy stressed track it finished in 82.1 simulated seconds with 27 physical cone contacts and six off-course events. The same-runtime frozen baseline timed out at 120 seconds with six contacts and 12 off-course events. This single development pair rejects the first candidate against the contact target; it is not the final held-out result.

An offline check on the same 256 validation anchors found that following the exact native teacher future labels gives steering SmoothL1 loss 0.0084458 against the teacher requests. The original model gave 0.0033238 and the new step-900 model gave 0.0086609. Thus fitting the native future path does not reproduce the teacher's instantaneous steering through the current follower. This is a measured target inconsistency, not proof that it caused every collision.

`configs/train/spark_isaac_lab_action.yaml` defines one bounded recovery experiment. It starts from the vision trial's full selected checkpoint, freezes all vision and auxiliary-head tensors, and trains the policy for 500 updates with plan coefficient 0.1. The original direct-action coefficients, FP32 precision, batch 128 and fixed validation anchors remain. Saved tensor bytes, including buffers, must retain the frozen initialization. The full model is still saved.

The optional in-memory feature cache is only valid with frozen vision. Before using `cache_frozen_features=true`, run the same recipe with a new output path and `cache_parity_only=true`. The gate compares the cached and direct plans, losses and policy gradients at fixed tolerances. It never changes the normal full-vision training path, and it never reuses features across runs or checkpoints.

`select_lab_action.py --training ACTION_RUN --baseline-training VISION_RUN --out NEW_RECEIPT` screens the completed recovery checkpoints before physical evaluation. Aggregate steering loss must be at most 1.25 times the original baseline. Per-episode steering must be at most twice that baseline, with a 0.0005 floor. Weighted plan L1 may increase by at most 10% and longitudinal loss by at most 25% relative to recovery step zero. These are explicit experimental screening thresholds, not safety guarantees. Every rejected checkpoint remains in the receipt; no eligible checkpoint produces a rejection result.

The executed action run completed all 500 updates in 490.80 seconds, sampling 64,000 examples with replacement. Median whole-GPU utilization was 87%, median power was 48.471 W, and peak sampled host unified RAM was 8.648 GiB. Its earlier portion overlapped CPU video export, so its wall time is not an isolated speed comparison with the full-vision run. All 31,154,802 model parameters remain in the saved model; 6,949,412 policy parameters were trainable in this stage. The selector chose `step_000500.pt`, SHA-256 `1ac2e3ba949db39ab445c636f841360973e2add8c86edab689a3a872f040ce55`. Its validation steering loss is 0.00122650, weighted plan L1 is 0.892578 and longitudinal loss is 0.208606. All eight episode steering limits passed. The six-second integration pilot had zero physical contacts and its 120-frame saved-input inference replay passed. None of those results establishes a completed race or improved closed-loop driving.

The cache parity probe failed the declared plan and policy-gradient tolerances, so the actual recovery run used direct image encoding. A proposed pre-update diagnostic of separate loss gradients and their pairwise cosines was omitted. The measured target disagreement remains valid, but gradient interference was not demonstrated. The trainer also recorded an additional step-1 validation, which was ineligible and did not affect checkpoint selection. Preserve these protocol deviations with the results.

For final selection, `freeze_lab_selection.py` accepts repeatable `--ancestor-training RUN_DIR CHECKPOINT` pairs in chronological order. Each actual checkpoint must connect the preceding history to the next initialization and retain the complete model signature. Separate completed baseline and candidate plans can contribute paired evidence only when their sealed scene, physics, camera, runtime, source, graph and asset conditions agree. Keep original batch directories intact and use a symlinked model/case view for the final report.

`archive_lab_batch.py` copies a completed, audited, video-covered batch and verifies complete remote/local SHA-256 inventories before writing a sibling receipt. It never removes remote evidence. Container-created copies must be readable by the SSH user before archival; changing file read permissions does not change the sealed contents. Preserve at least 25 GiB free on each machine.

## Physical development and recovery data

The action checkpoint was rejected for promotion. On the exposed heavy track, it finished in 57.65 simulated seconds with 14 physical cone contacts and zero off-course events. Its paired original baseline timed out at 120 seconds with 15 contacts and 18 off-course events. On the exposed heavy skidpad, the action checkpoint recorded 17 contacts and failed at 43.3 seconds with `outside_exit_lane`; all four circles were accepted in the required right, right, left, left order. The paired baseline finished in 46.4 seconds with three contacts. These are two exposed development cases, not a held-out completion rate.

Repeated original-baseline heavy-track runs timed out with six and 15 contacts. Their initial physical states, camera transforms and cone positions match, while their raw RGB and predictions already differ at frame zero. The first observed state difference is at frame one. This establishes substantial run variability, not its exact rendering cause. Preserve both runs and avoid treating a one-contact difference as a demonstrated improvement.

The first learner-prefix collection recipe handed over after eight seconds or 0.6 m lateral drift. Its first pilot finished cleanly, but at handoff the lateral and yaw errors were only 0.25156 m and 0.03263 rad. It failed the predeclared corrective-state gate. The partial `data/native/lab_v3` dataset and receipt remain intact, and no training used it.

`recovery_recipe_event.json` separately freezes an event-triggered follow-up. The learner hands over at 0.4 m lateral drift, with a 120-second cap. The same corrective-state, contact-free completion and return-to-center gates remain. Eight named training episodes are proposed as replacements; 24 original training episodes and all eight validation episodes remain byte-identical. The first four must pass before the remaining four. New collection captures at 20 Hz and stores phase-zero 5 Hz training images. Retained episodes keep their older capture behavior.

Run the two phases inside the Spark training container, without changing the frozen recipe or producer files between phases:

```bash
/isaac-sim/python.sh scripts/collect_lab_recovery.py \
  --recipe docs/isaac-lab-training/recovery_recipe_event.json \
  --source data/native/lab_v2 --out data/native/lab_v4 --phase pilot
/isaac-sim/python.sh scripts/collect_lab_recovery.py \
  --recipe docs/isaac-lab-training/recovery_recipe_event.json \
  --source data/native/lab_v2 --out data/native/lab_v4 --phase remaining
```

The coordinator records exact commands and source identities, stops on the first failed gate, and refuses to silently retry failed attempts. Only a completed and independently verified v4 dataset may feed `spark_isaac_lab_recovery.yaml`. That configuration proposes 500 policy-only updates with 16 of each 128 samples drawn from the first two expert-controlled seconds after qualifying handoffs. The other 112 draws come from the remaining qualified anchors. Recovery pools must exactly match their passed, hash-bound gate receipts and exclude terminal-braking anchors. This is a declared experiment, not an established fix.

Recovery selection adds `--parent-training runs/lab_v2/action_trial1` to the original vision reference. It verifies unchanged validation case identities, array hashes and selected frames across the v2 and v4 datasets, rather than requiring identical training inventories. The durable step-zero model must byte-match the immediate parent. Final ancestry verification uses each stage's own dataset through repeatable `--ancestor-dataset RUN MANIFEST RECEIPT` arguments when recorded paths are unavailable. Physical development and the frozen fresh suite remain required after offline screening.

The event-triggered collection completed all eight recovery gates. Its verified manifest has 7,015 training anchors and the unchanged 2,067 validation anchors. The recovery run completed 500 updates in 476.93 seconds and selected step 500. Its heavy-skidpad test failed with 28 physical contacts and five off-course events, so this candidate was rejected.

The fixed first stored anchor in each recovery case still has 0.07451 rad mean steering error, compared with 0.08038 rad at its parent. Four reconstructed training batches show much larger longitudinal than steering gradients. The total gradient aligns with steering at the parent but opposes steering at the rejected checkpoint. These are local gradient measurements, not measured AdamW updates or a physical causal result.

`steering_recipe.json` freezes one follow-up. `spark_isaac_lab_steering.yaml` restarts from the same action parent with the same data, sampler and 500-update schedule. It changes only the steering weight by a factor of 30. Existing checkpoint selection and retention rules remain. The selected checkpoint must additionally have first-anchor steering MAE at most 0.04 rad across the eight training cases and maximum individual error at most 0.08 rad. These are seen-anchor screening limits, not safety bounds. Only a passing candidate proceeds to saved-input inference checks and physical development.

The steering-weight run completed all 500 updates in 478.12 seconds. Its original validation-only selector chose step 300, which failed the fixed imitation limits at 0.04459 rad mean and 0.09730 rad maximum first-anchor error. That selection and rejection remain intact.

`steering_joint_selection.json` declares a separate selection-only follow-up. It applies the same imitation limits to every checkpoint that already passed the original validation requirements, then ranks joint passes by the original validation steering metric. No physical result from this run or fresh-test result informed that amendment. Steps 0, 100, 200, 300 and 400 failed imitation. Only step 500 passed, with 0.02842 rad mean and 0.06464 rad maximum first-anchor error. This is exploratory post-selection development, not the original preregistered selection result or proof of successful driving.

The selected step-500 checkpoint (`96dfd7d450a1a1a07309ec1d70dca208373dfd92313565dd776feefe46131408`) passed the 120-frame saved-input inference check. Its complete exposed heavy-clutter results were:

| Case | Model | Result | Simulated seconds | Physical contacts | Off-course events |
|---|---|---|---:|---:|---:|
| Track | Original baseline | Timeout | 120.00 | 3 | 25 |
| Track | Steering candidate | Finished | 57.25 | 3 | 2 |
| Skidpad | Original baseline | Finished | 46.35 | 1 | 0 |
| Skidpad | Steering candidate | Finished | 46.45 | 8 | 0 |

The candidate was rejected for the skidpad contact regression. Both supplied development pairs were measured and all eight debug/research videos passed the source-bound audit and local copy hash checks. Required broader development and fresh coverage remains incomplete. Baseline repeats vary materially, so these individual pairs are not reliable improvement estimates. Fresh tests remain unopened.

This fourth training run used 64,000 draws, including 8,000 draws from 80 distinct recovery anchors, in 478.12 seconds. Median sampled whole-GPU utilization was 86%, median power 48.736 W and peak sampled host unified RAM 8.031 GiB. The model retains 31,154,802 parameters; 6,949,412 policy parameters were trainable. Sampled utilization does not measure peak-FLOPS efficiency. Evaluation recorded roughly 3.78–4.12 frames per wall second at 20 Hz simulated control. Rendering averaged 224–243 ms per frame and instrumented policy execution 9–11 ms. CPU exports and short CPU diagnostics overlapped track evaluation, so these are not isolated throughput comparisons.

## Trajectory-to-controller diagnosis

The fourth candidate's first skidpad contacts followed outward predicted motion. Sampled actuator steering closely followed the request. Exact-input comparisons also found an internally inconsistent near-term position knot: predicted position implied about -0.21 then 10.65 m/s while the corresponding predicted velocities were about 5–6 m/s. This local defect was absent in the original and first native vision checkpoints, appeared in the action-trained checkpoint and persisted in later policy continuations. Native teacher targets did not contain it. Smoothing that knot alone would weaken the corrective steering, so it was not deployed as a fix.

Two [native reference bridge experiments](../../benchmarks/native-bridge-2026-11-09/README.md) then tested the mapping independently of model inference. All recorded-control and native-teacher repeats exactly matched the recorded nine-state/contact data before follower comparisons. The original controller used future fixed-frame x velocity and acceleration as forward speed and acceleration, causing false braking in a turn. Computing planar speed and tangential acceleration at each knot before interpolation removed that coordinate error. Together with the existing Openpilot steering mapping, maximum time-aligned error fell from about 1.1 m to about 0.2 m in the tested two-second turn/recovery windows.

These are short selected training windows, not mission completion or learned-model results. The corrected mapping retains a longitudinal residual because net motion acceleration and wheel-torque demand differ. No fitted resistance compensation or physics changes were introduced. Integration and retraining with a mode-bound controller contract must be evaluated separately.

## Versioned tangent-controller experiment

The native evaluator and trainer now share the explicit `op_tangent_v1` controller contract. It preserves Openpilot steering while computing planar speed and tangential acceleration at each plan knot before interpolation. Measured body velocity supplies feedback. Checkpoints, evaluation plans, saved-input replay, artifact audits and both debug video formats retain the controller identity. Legacy checkpoints and PP evidence retain their original interpretation.

The full model is initialized from the first vision-adapted checkpoint, before the action-stage position notch. The next policy stage uses the verified v4 native data with an explicit off-policy recovery contract. Collection-policy identity and optimization initialization are recorded separately. The [frozen selection protocol](tangent_selection2.md) requires action retention, plan consistency and unchanged frozen tensors. The zero-update initializer fails the new acceleration-consistency limits, so retaining it means no eligible candidate.

[Calibration records](../../benchmarks/native-tangent-calibration-2026-11-09/README.md) preserve two zero-update runs, the rejected initial screen, an executable counterexample and independent review. The six-second camera/inference pilot recorded zero physical contacts and zero off-course events. All 120 inference frames replayed, artifact auditing passed, and both 160-frame videos fully decoded. The full local suite passed 364 tests with four skipped and seven subtests. These checks verify integration, not improved full-mission driving.

## Completed tangent and native-future experiments

The fifth tangent-controller run completed 500 policy updates in 474.74 seconds at 87% median sampled GPU utilization. No checkpoint passed its frozen selection gates. The declared final-step diagnostic finished the exposed track in 56.80 simulated seconds with one physical contact and one off-course event, but failed skidpad at 75.65 seconds with 17 contacts and five off-course events. Its same-controller initializer finished track in 56.55 seconds with one contact and one off-course event and violated the skidpad route at 12.55 seconds with nine contacts. [Exact fifth-run evidence](../../benchmarks/native-tangent-trial-2026-11-09/README.md) retains the failure and the target diagnosis.

A trajectory can exactly match the native future while disagreeing with another controller's recorded immediate command. The experimental `native_future_v1` target corrects this mismatch by applying the deployed follower to the detached native future, measured body velocity and vehicle limits before mission braking. Raw teacher-command errors remain separately named diagnostics. This target is versioned through training, calibration, checkpoints and selection. It is available only with `op_tangent_v1`; legacy recipes retain their original target meaning.

The [new calibration and protocol](native_future_selection1.md) exactly reproduced the old predicted-plan metrics, native target consistency and teacher diagnostics before updates. All 441 plan/consistency bound values were retained. Only 18 action ceilings were recalibrated under the explicitly different target definition. The 143-file pre-update freeze was verified on Spark. No threshold was loosened after training.

The sixth run completed 500 updates in 475.77 seconds. Median sampled whole-device GPU utilization was 86%, median power 48.95 W and peak host unified RAM was 8.121 GiB. It retained all 31,154,802 parameters and trained 6,949,412 policy parameters. Its 64,000 draws included 8,000 replacement draws from 80 distinct recovery anchors. Training used FP32 and eight-frame history; BF16 and the feature cache remain disabled after failed parity checks. This isolated training measurement excludes collection, physical evaluation and video export. GPU utilization is busy time, not peak-FLOPS efficiency.

No sixth-run checkpoint passed every frozen gate. Step-500 acceleration/secant mean, p95 and maximum were 1.146, 5.180 and 6.942 m/s² against ceilings 1, 2 and 4. The 120-frame saved-input inference replay passed. The final-step diagnostic then produced:

| Case | Model | Result | Simulated seconds | Physical contacts | Off-course events |
|---|---|---|---:|---:|---:|
| Track | Original PP baseline | Timeout | 120.00 | 11 | 10 |
| Track | Same-controller initializer | Finished | 56.55 | 1 | 1 |
| Track | Rejected step 500 | Finished | 56.95 | 2 | 1 |
| Skidpad | Original PP baseline | Finished | 46.35 | 6 | 0 |
| Skidpad | Same-controller initializer | Invalid route | 12.60 | 8 | 0 |
| Skidpad | Rejected step 500 | DNF | 76.25 | 25 | 6 |

Each model completed one of two exposed cases. These single runs do not estimate reliable completion rates, and the timed-out baseline does not establish a lap-time speedup. The latest model did not fix skidpad or improve overall cone avoidance. No model was promoted. Required broader development and fresh coverage remains incomplete; fresh test scenes remain unopened.

The [sixth-run archive](../../benchmarks/native-future-trial-2026-11-09/README.md) preserves the exact receipts and subsequent no-update analyses. Current saved-anchor replay finds compensating velocity and acceleration errors. A descriptive interpolation reduces consistency error while worsening the current objective on 11 of 14 tail anchors; it leaves steering unchanged. Raw measured-yaw-rate substitution also worsens steering loss in six of eight validation scenes. Neither intervention was deployed or used to justify another optimizer run. The physical skidpad trace cuts inside cones and completes only the two right circles. Its actuator follows the requested steering; the failure cannot be explained away as an unexecuted control command.

The current implementation passed 378 CPU tests with four skips and seven subtests, plus 56 focused target/controller/selector tests and Ruff. These checks establish software contracts and integration, not a successful learned driver. The full camera evidence, checkpoint hashes and unsuccessful videos remain available.

## Producer-version video audit

Strict controller replay in the NumPy 2.4 CPU exporter exposed tiny rounding differences from the NumPy 1.26 producer. Failed audits were preserved. Auditing and rendering with the producer environment reproduced the commands without widening tolerances. The external two-phase exporter then binds source/input/raw-video hashes before H.264 encoding in the CPU exporter and fully decodes both 1280×768, 20 fps videos. The compositor remains unchanged, including camera views, predicted plans, image-head diagnostics, command pulses and referee counters. The green custom-bicycle reference and auxiliary detections remain explicitly diagnostic.

CPU exports and short saved-record diagnostics overlapped parts of physical evaluation. Reported evaluation timing therefore describes the actual instrumented workload, not isolated renderer throughput. Training itself ran without those workloads.
