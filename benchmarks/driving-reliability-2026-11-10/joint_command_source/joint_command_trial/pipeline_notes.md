# Joint command evidence pipeline

The package kind is `direct_native_actuator_joint_suffix_v1`. Schema 5 identifies its native evidence. These scripts reject the old direct-command schema and require the new package contract. The producer and every replay use CPU FP32 and four Torch threads. The full updated base and head are frozen against the saved package for deployment. The 411-tensor original-base complement check belongs to package validation and training, not deployment parity against the original base.

All commands run from the repository root. Use the producer's Isaac Lab Python and environment for evaluation, replay, and rendering. These examples are invocation shapes, not authorization to run a case. The frozen protocol controls case membership, gate ordering, timing preflight, and exposure. All output paths must be new. Keep failed outputs and receipts.

```sh
PYTHONPATH=src:. "$ISAAC_PYTHON" .work/lab/joint_command_trial/evaluate.py --case CASE.json --checkpoint PACKAGE.pt --expected-sha PACKAGE_SHA256 --out EPISODE --device cpu --pilot-steps 120
PYTHONPATH=src:. "$ISAAC_PYTHON" .work/lab/joint_command_trial/verify_command_episode.py --episode EPISODE --checkpoint PACKAGE.pt --out REPLAY.json --max-frames all --device cpu
PYTHONPATH=src:. "$ISAAC_PYTHON" .work/lab/joint_command_trial/export_audited_command.py --episode EPISODE --replay-receipt REPLAY.json
```

Omit `--pilot-steps` for an authorized complete episode. `--max-frames 120` accepts a complete 120-frame pilot only. `--max-frames all` checks every saved frame and is required for all-frame replay. The fixed 1e-5 absolute and relative tolerance applies. Replay checks saved policy raw output, image head, plan, latent, proprioception, command head output, bounds, mission override and applied action, as well as hooked versus plain inference. It does not rerun physical simulation. `process.py` independently reconstructs mission, route, referee, physical-state and contact evidence from saved data. Recorded contacts are not independently simulated.

The producer wrapper runs that one physical auditor, requires NumPy 1.26, renders both views with the original compositor and a joint-policy diagnostic label, and records a source-bound render receipt. It retains raw videos. A missing ffmpeg on the producer is expected. Transport the complete episode, replay receipt, and exact package to a host with ffmpeg and OpenCV. Then run:

```sh
python .work/lab/joint_command_trial/encode_portable.py --episode LOCAL_EPISODE --replay-receipt LOCAL_REPLAY.json --checkpoint LOCAL_PACKAGE.pt
```

The encoder maps producer episode paths into the transported directory. It verifies the full evidence inventory, every archived source, replay and artifact receipts, package hash, raw render hashes, and fixed CPU identity. It verifies graph identities through the producer replay receipt and does not claim to reexecute absent producer graphs. It fully decodes both raw and encoded views at 1280 by 768, 20 Hz, including 40 terminal frames. H.264 uses `fast`, CRF 20, one thread, yuv420p, and faststart. Outputs are exclusive and linked into place atomically. A failed encode leaves its temporary outputs for inspection and must not be overwritten.

Run local boundary tests with `PYTHONPATH=src:. .venv/bin/python .work/lab/joint_command_trial/test_pipeline.py`. Synthetic receipts in those tests exercise transport validation only. They do not represent a real model, parity result, rendering result, physical gate, or qualification. The scripts do not launch training or final cases.
