# Native Isaac Lab workflow on Spark

`fsop train` now defaults to `configs/train/spark_isaac_lab.yaml`. It trains the full existing image/temporal-plan model from native Isaac Lab episodes. Explicit legacy configs remain available for reproducing previous experiments. The change is the data and simulation backend, not a replacement with a small reinforcement-learning policy.

## Runtime boundary

The tested Spark renderer/trainer uses Isaac Lab 2.3, Isaac Sim 5.1, Python 3.11.13, PyTorch 2.9.0+cu130 and NumPy 1.26.0. Its task container is `fsop-isaac-lab-training`; the immutable image is `fsop-isaac-eval:verified` (`sha256:7092a76a990f364ea684b64a48681e9a7c1992a8d4af2edfc758e640b57668bb`). The checkout is mounted at `/work`, with `PYTHONPATH=/work:/work/src:/work/isaac_eval`. Prepared original model graphs and appearance assets are mounted read-only.

Use the image's `/isaac-sim/python.sh`. Do not install this repository's legacy all-in-one `pyproject.toml` dependencies into the renderer: they include NumPy 2 and Rerun, while this tested camera runtime uses NumPy 1.26. Native collection and training use the source checkout directly. The separate CPU exporter image `fsop-isaac-stress:verified-export` has NumPy 2.4.6, Rerun 0.42 and FFmpeg 6.1.1. This separation is intentional and is part of the tested deployment.

The full model's existing FP32 prepared vision/policy graphs and original complete checkpoint are required. The native environment uses real Lab `SimulationContext`, `InteractiveScene` and `RigidObject` instances. PhysX advances at 120 Hz, controls at 20 Hz, and retained training images at phase-zero 5 Hz. Evaluation records every 20 Hz input and prediction. Scene queries and contact processing must stay enabled. The runtime restores the original RTX settings after Lab's renderer preset initialization; receipts preserve those settings.

## Collection and integrity

From `/work` inside the renderer/trainer:

```sh
/isaac-sim/python.sh scripts/collect_lab_dataset.py --out data/native/lab_v1
/isaac-sim/python.sh scripts/verify_lab_dataset.py \
  --manifest data/native/lab_v1/manifest.json \
  --out data/native/lab_v1/verification.json
```

Membership is frozen in `membership.json`: 32 training and eight development episodes. Trackdrive geometry seeds are disjoint across splits. Skidpad retains its defined geometry and varies scene seeds. The separate fresh test and exposed regression cases are never collected as training labels.

The teacher acts on measured native physical state. Targets use ten seconds of its actually executed future, including an actual braking continuation after scored finish. No bicycle forecast or repeated terminal padding supplies future labels. Only complete, passing, contact-free native teacher episodes contribute anchors; failed episodes remain recorded. Raw `controls.npy` are requested teacher commands before native clipping, lag and rate limits; `states.npy` also records actual filtered actuator states. Direct action losses currently supervise requested PP commands. The longitudinal loss is before the runtime mission launch/finish override, so it is not a complete deployed-action metric.

The reader checks every array digest, horizon, contact mask and split. The independent dataset verifier additionally checks complete membership, metadata seals, producer archives, regenerated track geometry, vehicle, camera and scene realization. A source archive belongs to each episode; never alter source files during an active collector or evaluator.

## Training and selection

```sh
/isaac-sim/python.sh -m fsop.cli train --cfg configs/train/spark_isaac_lab.yaml
```

Both the vision trunk and temporal policy receive gradients. Auxiliary cone-head tensors remain frozen because native visibility labels are not yet qualified. Their outputs can still change when their shared vision features change, so inspect the debug videos. Historical features are recomputed from the current vision weights, with historical gradients intentionally truncated. There is no persistent stale feature cache. The default recomputes all 24 past slots. Optional `--history-context 8` skips the 16 slots that the pinned policy graph cannot read, retaining the complete input shape and full command history. This option requires both a checked graph/loaded-buffer proof and a measured output/gradient parity gate before use in a recipe.

The run saves the full model, optimizer, exact validation anchors, source archive, initialization digest, per-step losses and timing. FP32 validation uses the same episode-balanced anchors even if training autocast is enabled. The initial model is a valid selection candidate. Open-loop `selected.pt` is provisional: physical development evaluation determines whether it improves driving before fresh test results are opened. An optimizer checkpoint is not an exact resumable training session; RNG restoration/resume is not implemented.

## Evaluation and same-format evidence

Create a frozen plan as documented in `scripts/run_lab_evaluations.py`. It binds case JSON files and each complete checkpoint SHA-256. Then run:

```sh
/isaac-sim/python.sh scripts/run_lab_evaluations.py --plan runs/evaluation_plan.json --out runs/evaluation
```

The coordinator runs one native process per case, checks disk reserve, records technical failures, and resumes only fully sealed episodes with matching identities. A scored timeout is retained as a result. Technical errors are investigated separately and cannot be silently skipped. For bounded archival batches use separate frozen batch plans; resume requires the complete original evidence locally available to the coordinator.

Use `python -m isaac_lab_fsop.process --help` inside the CPU exporter for independent state/referee/command auditing and the established debug/research MP4 and Rerun formats. Videos include raw model inputs, predictions, cone-head outputs, controls, commands and failures. The green diagnostic overlay remains the historical bicycle oracle and is labeled accordingly; it is not the native teacher or a model input. Physical cone contacts are reported separately from the legacy referee's cone-displacement penalties.

`scripts/report_lab_training.py` builds standalone training graphs and an HTML results index. Pass the frozen evaluation plan so absent cases remain visible. Fresh tests, development cases and exposed regressions stay separate. Sealed artifacts, audit success and video coverage are distinct checks.
