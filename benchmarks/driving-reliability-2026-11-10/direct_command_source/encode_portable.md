# Portable encoding for the direct-command pilot

This command encodes the producer's saved raw debug and research renders on a host with Python, OpenCV, and ffmpeg with libx264. It does not rerun the simulation, inference, artifact auditor, or renderer. It does not qualify the model. The pilot had three contacts and one off-course event in six seconds.

From the repository root, after the episode copy finishes and the complete checkpoint is available:

```sh
.venv/bin/python .work/lab/direct_command_trial/encode_portable.py \
  --episode runs/direct_command_pilot3 \
  --replay-receipt runs/direct_command_pilot3/replay-cuda.json \
  --checkpoint runs/direct_command_trial3_archive/step_500.pt
```

The program requires a passed complete CUDA replay with the original numerical tolerances and binds its exact episode inventory, the completion receipt, the full checkpoint, every archived source byte, the producer artifact audit, the producer render receipt, and both raw videos. Archived renderer and original auditor code are verified against their source bundle. If the derived auditor and wrapper were created after the bundle, their local copies must match the producer's recorded hashes. Graph hashes are matched between metadata and the producer's replay. Graphs are not available locally and are not reexecuted or independently verified by this command.

Each raw and encoded video is fully decoded and checked for 1280 by 768 pixels, 20 fps, and the inference-frame count plus 40 terminal frames. The established ffmpeg settings use libx264, fast preset, CRF 20, yuv420p, one thread, and faststart. Outputs are `debug.mp4`, `research.mp4`, and `lab_video_audit.json`. Existing outputs or temporary files cause refusal. Each temporary video is published by an atomic no-overwrite hard link followed by removal of that successfully published temporary link. On failure, partial files and any already published outputs remain for inspection; move them aside deliberately before a retry. The receipt records the portable encoder and ffmpeg version separately from the producer renderer.
