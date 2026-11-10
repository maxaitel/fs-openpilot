# Cone-set native evidence pipeline

The package kind is `vision_cone_set_actuator_v1`, and native evidence uses schema 6. The producer and replays use CPU FP32 with four Torch threads. The package validator binds the original base weights, all 417 base tensors and eight head tensors. The head has 47,618 parameters. The original plan is diagnostic. The decoded image cones, complete route queue and measured motion drive the new head.

Run commands from the repository root in the producer's Isaac Lab Python environment. These invocation shapes do not authorize a case. The frozen protocol controls case membership and gate ordering. Output paths must be new.

```sh
PYTHONPATH=src:. "$ISAAC_PYTHON" .work/lab/cone_set_actuator_trial/evaluate.py --case CASE.json --checkpoint PACKAGE.pt --expected-sha PACKAGE_SHA256 --out EPISODE --device cpu --pilot-steps 120
PYTHONPATH=src:. "$ISAAC_PYTHON" .work/lab/cone_set_actuator_trial/verify_command_episode.py --episode EPISODE --checkpoint PACKAGE.pt --out REPLAY.json --max-frames all --device cpu
PYTHONPATH=src:. "$ISAAC_PYTHON" .work/lab/cone_set_actuator_trial/export_audited_command.py --episode EPISODE --replay-receipt REPLAY.json
```

Omit `--pilot-steps` only for an authorized complete episode. The 120-frame replay mode accepts a complete pilot only. The all-frame replay rebuilds the full temporal history, compares full image and plan outputs, redecodes the saved image head using nominal calibration, repacks saved and decoded cones, reconstructs the full desire queue, and reruns the head from saved tokens, mask, desire and proprioception. Saved list values are explicitly cast to the declared FP32 or Boolean types before comparison. The same-backend tolerance is 1e-5 absolute and relative, with exact discrete comparisons. Replay does not rerun physics.

The physical auditor reconstructs mission, route, referee, physical-state and contact evidence from saved data. Its passed flag means evidence consistency. It does not mean zero-contact qualification. Qualification additionally requires the full trackdrive or R,R,L,L skidpad mission and stop completion within 120 seconds, zero physical cone contacts and zero off-course events, under the frozen protocol. The legacy referee pass field is retained for reproducibility. Recorded contacts are not independently simulated.

The producer wrapper requires NumPy 1.26 and writes both raw views with an explicit original-plan diagnostic and cone-set action banner. Each inference frame in both views also shows its valid cone count, raw, bounded and applied steering in radians and acceleration in m/s², and the AS override flag. The complete mask remains in the saved numerical evidence. Terminal cards carry no current prediction overlay. Transport the complete episode, replay receipt and exact package to a host with ffmpeg and OpenCV, then run:

```sh
python .work/lab/cone_set_actuator_trial/encode_portable.py --episode LOCAL_EPISODE --replay-receipt LOCAL_REPLAY.json --checkpoint LOCAL_PACKAGE.pt
```

The encoder maps producer paths to the transported episode and checks the full evidence inventory, source archive, producer replay, artifact audit, package and raw render hashes. Graph identities are bound to the producer replay and are not locally reexecuted. It decodes both raw and H.264 videos in full at 1280 by 768 and 20 Hz, including 40 terminal frames. Outputs are exclusive. A failed encode leaves temporary outputs for inspection.

`runtime_preflight.py` provides a saved TRAIN-prefix helper to the training stage. It does not authorize a stage, write receipts, launch a simulator or establish mission success. CPU repeat inference, reconstructed inputs and the independent head are gates. A training-device comparison is diagnostic only. Timing excludes physics, rendering, image file reads, observation reconstruction and evidence writing, so it is not a 20 Hz wall-clock claim.

Run the narrow local boundary tests with `PYTHONPATH=src:. .venv/bin/python .work/lab/cone_set_actuator_trial/test_pipeline.py`. Their synthetic receipts test transport validation and the ffmpeg fixture exercises the actual encoder and full decoder. They do not establish inference parity, physical scoring or qualification.
