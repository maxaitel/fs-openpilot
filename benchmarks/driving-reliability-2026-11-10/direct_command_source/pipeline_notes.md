# Direct-command native pipeline

The pilot and every driving attempt need distinct output directories. The native evaluator uses `mkdir(..., exist_ok=False)` and the verifier refuses to overwrite its receipt. A completed run seals its frames and summary even if its driving result includes contacts or failure. An exception before sealing leaves its partial evidence in place; it does not produce a valid completion receipt or a verified video. Preserve that directory for diagnosis.

Run these from the repository root in the Isaac Lab environment, substituting the actual exposed-case JSON, package, and unique paths. Do not use final cases for this pilot. The package must be qualified by the separate training gates first.

```sh
PYTHONPATH=src:. /isaac-sim/python.sh .work/lab/direct_command_trial/evaluate.py \
  --case CASE.json --checkpoint PACKAGE.pt --expected-sha PACKAGE_SHA256 \
  --out runs/direct_command/PILOT_ID --pilot-steps 120
PYTHONPATH=src:. /isaac-sim/python.sh .work/lab/direct_command_trial/verify_command_episode.py \
  --episode runs/direct_command/PILOT_ID --checkpoint PACKAGE.pt \
  --out runs/direct_command/PILOT_ID/replay-cpu.json --max-frames 120 --device cpu
```

Native evaluation defaults to the policy's CUDA-first choice and records the actual device. The verifier defaults to CPU. Cross-device disagreement fails at the same absolute and relative `1e-5` thresholds as the original verifier; preserve its receipt and investigate. To diagnose device effects, use a separate receipt path and `--device cuda`. Neither mode widens the tolerance. If the car terminates before 120 frames, the bounded pilot cannot pass. A full sealed episode can be replayed with `--max-frames all`; this replays every frame, including a short failed episode, and reports whether all four temporal phases were present. No full driving run is authorized by the existence of these scripts.

`isaac_lab_fsop/process.py` rejects schema 4. The separate `process.py` copies its physical-state, mission, contact and referee checks, replaces the old plan-to-action audit with direct-command bounds and mission checks, and requires a passed replay receipt covering the entire sealed episode. The evaluator archives and hashes this auditor and the export wrapper. The renderer retains the frozen model's actual plans and cone heads. A display-only banner identifies the plan as diagnostic and the action as the learned direct command.

For each sealed episode, replay all frames first. A 120-frame pilot can use its pilot receipt. Then use the producer NumPy 1.26 environment for rendering, followed by the existing export environment for encoding. Both phases require the same receipt and refuse to overwrite earlier outputs.

```sh
PYTHONPATH=src:. /isaac-sim/python.sh .work/lab/direct_command_trial/verify_command_episode.py \
  --episode runs/direct_command/EPISODE_ID --checkpoint PACKAGE.pt \
  --out runs/direct_command/EPISODE_ID/replay-cpu.json --max-frames all --device cpu
PYTHONPATH=src:. /isaac-sim/python.sh .work/lab/direct_command_trial/export_audited_command.py render \
  --episode runs/direct_command/EPISODE_ID --replay-receipt runs/direct_command/EPISODE_ID/replay-cpu.json
PYTHONPATH=src:. python .work/lab/direct_command_trial/export_audited_command.py encode \
  --episode runs/direct_command/EPISODE_ID --replay-receipt runs/direct_command/EPISODE_ID/replay-cpu.json
```

The producer writes the artifact audit, raw videos and a render receipt. The encoder validates those bindings, encodes H.264, fully decodes each video at 1280x768 and 20 Hz, checks `frame_count + 40` frames, and writes the video audit. The encode command's `python` must be the established export environment with OpenCV and ffmpeg installed. These paths have passed CPU structural checks only. They have not yet been exercised against a sealed schema 4 native episode. A successful replay does not establish zero-contact driving. An incomplete episode cannot pass the audit and must be retained as partial evidence.
