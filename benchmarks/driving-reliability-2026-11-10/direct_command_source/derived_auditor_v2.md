# Derived direct-command auditor v2

The original producer sealed `process.py` and its other sources before this auditor existed. The direct-command head converts `vx`, `vy`, and yaw rate to float32 before recording proprioception. The original auditor compared the recorded values with unconverted float64 physical state values at the existing `1e-8` absolute and `1e-9` relative tolerances. V2 applies the producer's float32 conversion to the expected vector. No tolerance or physical, camera, referee, contact, or command bound changes.

`process_v2.py` requires the same complete passed replay and checks all its source, graph, and artifact hashes. It verifies the original producer auditor against metadata, the replay, the live source, and the archived bytes. If metadata contains v2, it verifies its own hash and archived bytes too. Otherwise it records `auditor_origin: derived_after_producer`. Every artifact audit records the original and derived auditor paths and SHA-256 hashes and the reason for derivation. It rechecks v2 and replay-bound inputs before writing its result.

`export_audited_command_v2.py` retains the compositor and banner. Its render and video receipts bind the original producer auditor, derived auditor, derived wrapper, renderer, passed replay, artifact audit, and existing episode inputs. It uses exclusive output creation as before. An earlier failed audit and failed CPU replay remain evidence, not prerequisites to overwrite. Preserve them at distinct names before running the v2 exporter. Verify the preserved failed audit hash against its pre-move hash.

For a sealed pilot with an existing full-frame CUDA replay that already passed the original verifier at `rtol=atol=1e-5`, from the repository root use the same replay receipt in both commands. Replace the placeholders with the producer's actual episode and receipt paths. The rendering interpreter must be the producer's NumPy 1.26 environment. The encoder uses the established OpenCV and ffmpeg export environment.

```sh
PYTHONPATH=src:. /isaac-sim/python.sh .work/lab/direct_command_trial/export_audited_command_v2.py render \
  --episode EPISODE --replay-receipt PASSED_CUDA_REPLAY.json
PYTHONPATH=src:. python .work/lab/direct_command_trial/export_audited_command_v2.py encode \
  --episode EPISODE --replay-receipt PASSED_CUDA_REPLAY.json
```

`evaluate_v2.py` is an experimental copy for future full exposed diagnostic cases only. Its source inventory includes the six original trial files, all v2 files in this directory, and `scripts/audit_isaac_episode.py` in addition to the original source trees. The original verifier accepts the additional archived sources and verifies every listed hash. This copy does not authorize a new driving run, change checkpoint selection, or open fresh test cases. A successful artifact audit does not imply zero contacts or a successful driving result.
