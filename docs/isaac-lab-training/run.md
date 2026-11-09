# Native Isaac Lab workflow on Spark

`fsop train` now defaults to `configs/train/spark_isaac_lab.yaml`. It trains the full existing image/temporal-plan model from native Isaac Lab episodes. Explicit legacy configs remain available for reproducing previous experiments. The change is the data and simulation backend, not a replacement with a small reinforcement-learning policy.

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

For final selection, `freeze_lab_selection.py` accepts repeatable `--ancestor-training RUN_DIR CHECKPOINT` pairs in chronological order. Each actual checkpoint must connect the preceding history to the next initialization and retain the complete model signature. Separate completed baseline and candidate plans can contribute paired evidence only when their sealed scene, physics, camera, runtime, source, graph and asset conditions agree. Keep original batch directories intact and use a symlinked model/case view for the final report.

`archive_lab_batch.py` copies a completed, audited, video-covered batch and verifies complete remote/local SHA-256 inventories before writing a sibling receipt. It never removes remote evidence. Container-created copies must be readable by the SSH user before archival; changing file read permissions does not change the sealed contents. Preserve at least 25 GiB free on each machine.
