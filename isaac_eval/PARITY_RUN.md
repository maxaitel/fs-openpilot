# Run the frozen diagnostic parity suite

Use this guide to record all 41 matched cases and export videos from the saved inference calls. Read the [evidence method](../docs/isaac-parity-method.md) for scoring, diagnostic labels, and comparison limits. The [decision record](../docs/isaac-parity-decisions.md) explains the verification choices.

These commands describe a new run. They do not certify an existing run. Keep the pilot separate from the scored suite, and use new output directories for every experiment.

## Prepare the runtime and assets

Use an isolated ARM64 `nvcr.io/nvidia/isaac-lab:2.3.0` container with Isaac Sim 5.1.0 and Torch 2.9.0+cu130. Provide GPU access, `--shm-size=8g`, a separate writable cache, and these mounts:

- Mount this parity checkout at `/work` with writable `isaac_eval/runs` and `isaac_eval/assets` directories.
- Mount the original prepared artifacts read-only at `/work/artifacts`. Keep `models/prep/vision_fp32.onnx`, `models/prep/policy_fp32.onnx`, and the exact 20 `panoramas/*.jpg` files recorded in `parity_reference/provenance.json`.
- Mount the original `skidpad_clean/final.pt` read-only at `/frozen/model.pt`. Its required SHA-256 is `cf3a8d253dfd5a417e1967d2d6dc8cdd0263e3d16d3f1f02f15495f490a83b7e`.

Keep training and other GPU work stopped during verification and scoring. Keep the original model and environment sources unchanged. The baseline guard rejects changes to the archived source hashes.

Use the project's Python dependencies in this runtime, including `onnx2torch==1.5.15`, `shapely==2.1.2`, `psutil`, and `nvidia-ml-py==13.615.71`. Preserve the image's Torch and CUDA installation. Offline export also needs `ffmpeg` with `libx264` and `rerun-sdk==0.42.0`. On the workstation, use the project's Python environment with NumPy and Matplotlib installed, plus SSH and rsync for collection.

Run the following commands inside the container with `/isaac-sim/python.sh` to use Isaac's Python environment.

```bash
cd /work
export PYTHONPATH=/work:/work/src
/isaac-sim/python.sh -m isaac_eval.realism_assets \
	--out isaac_eval/assets --lock isaac_eval/realism_assets.lock.json
/isaac-sim/python.sh -m isaac_eval.prepare_parity_cases \
	--output isaac_eval/runs/parity_cases \
	--checkpoint /frozen/model.pt --panorama-dir artifacts/panoramas
```

The asset downloader verifies the locked Poly Haven files. It does not supply the original model, ONNX graphs, or panorama inventory. Obtain those original artifacts before continuing. Keep panoramas at `artifacts/panoramas`, because the evaluator reconstructs that path even if preparation receives another `--panorama-dir`.

Check that preparation reports 41 cases and writes `parity_cases/manifest.json` plus one realization per case. The matrix contains 20 skidpad cases, 16 retention cases, four command ablations, and the previous-failure case.

## Verify the vehicle, cameras, and pilot inference

Run each command as a separate process from `/work` in the same container.

```bash
/isaac-sim/python.sh isaac_eval/verify_integration.py \
	--scene paddock --out isaac_eval/runs/parity_vehicle
/isaac-sim/python.sh isaac_eval/verify_parity_camera.py \
	--out isaac_eval/runs/parity_camera
/isaac-sim/python.sh isaac_eval/evaluate_parity.py \
	--case skidpad-72000-light-recorded \
	--manifest isaac_eval/runs/parity_cases/manifest.json \
	--pilot-steps 120 --out isaac_eval/runs/parity_pilot
/isaac-sim/python.sh -m isaac_eval.verify_inference \
	--checkpoint /frozen/model.pt \
	--evidence isaac_eval/runs/parity_pilot/evidence \
	--output isaac_eval/runs/parity_pilot/inference_verification.json
```

Inspect `parity_vehicle/verification.json` and `parity_camera/verification.json` for `pass: true`. Inspect the pilot's `inference_verification.json` for `passed: true` and `frozen_unchanged: true`. Check the receipts rather than relying only on the Isaac launcher's exit status.

The inference gate replays the pilot's saved inputs through the frozen model and compares the saved predictions, image-head outputs, and actions. It also checks that diagnostic capture leaves the policy outputs unchanged. Black-image replay is an input-sensitivity diagnostic, not a scored driving case. This gate verifies the pilot, not a model replay of every scored episode.

## Freeze the protocol and run all cases

Freeze only after all three verification receipts pass. Keep the pilot, receipts, assets, and source files in place for the entire suite.

```bash
/isaac-sim/python.sh -m isaac_eval.freeze_parity_protocol \
	--manifest isaac_eval/runs/parity_cases/manifest.json \
	--pilot isaac_eval/runs/parity_pilot \
	--vehicle isaac_eval/runs/parity_vehicle/verification.json \
	--camera isaac_eval/runs/parity_camera/verification.json \
	--out isaac_eval/runs/parity_protocol.json
/isaac-sim/python.sh -m isaac_eval.run_parity_suite \
	--manifest isaac_eval/runs/parity_cases/manifest.json \
	--protocol isaac_eval/runs/parity_protocol.json \
	--out isaac_eval/runs/parity_v1
```

The protocol binds the manifest, ordered case list, runtime sources, executable graphs, verification receipts, and pilot metadata. Changing a hashed input requires new verification and a new protocol. The protocol output must not already exist. The suite output directory must also be new.

Inspect `parity_v1/status.json` for all 41 cases and `complete: true`. A driving failure is a retained result. A technical failure stops the suite and keeps its log and partial artifacts. The runner has no resume or automatic retry option. Preserve the failed attempt before investigating or starting a separately named experiment.

Each completed case contains `summary.json`, `frames.json`, `telemetry.json`, and lossless inference evidence. `evidence/completion.json` binds the terminal summary to the metadata and frame manifest. Keep the raw `frame_*.npz` files for audit and video export.

## Audit the evidence and export videos

Wait for the measured suite to finish before CPU postprocessing. Run this command in the container, where all raw evidence remains available.

```bash
/isaac-sim/python.sh -m scripts.process_isaac_parity \
	--run isaac_eval/runs/parity_v1 --workers 4
```

If dependencies are missing, the script supports `--install-ffmpeg` and `--install-rerun`. The former uses `apt-get` and requires installation privileges. Add `--wait` only when deliberately starting a process that waits for suite completion.

Check `parity_v1/postprocess_status.json` for `complete: true`. Each case receives an `artifact_audit.json`, `debug.mp4`, `research.mp4`, `debug.rrd`, and `video_audit.json`. The processor independently replays mission progress, route commands, and referee scoring from recorded states. It verifies frame hashes without loading a model. Video generation uses the recorded predictions and labels the offline expert reference separately.

The video audit checks both 1280 by 768 videos at 20 fps, with one frame per recorded inference call and 40 terminal-card frames. It binds the video hashes to the completion receipt and artifact audit. Postprocessing time is separate from measured driving time.

## Collect compact results on the workstation

Run these commands from the same parity checkout on the workstation. Replace the remote location with the host directory mounted at `/work` in your container. This example uses the Spark experiment location.

```bash
export PYTHONPATH="$PWD:$PWD/src"
PARITY_REMOTE=max-spark:/home/maxaitel/fsop-isaac-debug-parity/isaac_eval/runs
python3 scripts/collect_isaac_parity.py \
	--remote "$PARITY_REMOTE/parity_v1" --out isaac_eval/runs/parity_v1
rsync -a "$PARITY_REMOTE/parity_cases" \
	"$PARITY_REMOTE/parity_protocol.json" isaac_eval/runs/
```

The collector waits for processed cases and finishes when all exports are complete. It copies videos, diagnostic tables, summaries, and audit receipts. It excludes raw NPZ frames, Rerun recordings, scene assets, source videos, and logs. The separate rsync command copies the top-level manifest, realizations, and protocol, which the collector does not fetch.

Keep the full evidence and verification directories on the Spark. For a fully local evidence archive, copy those files separately and omit `--remote-evidence-root` in the next command.

## Build the final report and gallery

Use the same remote evidence location supplied to the collector.

```bash
python3 scripts/report_isaac_parity.py \
	--run isaac_eval/runs/parity_v1 \
	--manifest isaac_eval/runs/parity_cases/manifest.json \
	--protocol isaac_eval/runs/parity_protocol.json \
	--remote-evidence-root "$PARITY_REMOTE/parity_v1" \
	--out isaac_eval/runs/parity_v1_report
python3 scripts/build_parity_gallery.py \
	--report isaac_eval/runs/parity_v1_report --run isaac_eval/runs/parity_v1
```

Open `isaac_eval/runs/parity_v1_report/index.html` for the gallery and `report.md` for the written report. Keep the run and report directories together so relative video links resolve. `parity.json` and `cases.csv` retain the per-case results and provenance.

The remote-evidence mode validates copied receipts and videos against the Spark's full-evidence audit. It does not download or independently reread the remote NPZ files. Without that option, the reporter requires a local frame-file inventory. Both report and gallery commands reject incomplete final results. Use their `--partial` options only for monitoring.

Historical custom videos are optional gallery inputs. To show them, restore `video.json` and `skidpad-raw.mp4` under `isaac_eval/runs/custom_video_reference/72000-light` and `71003-heavy`. Those recordings are not generated or collected by this workflow.
