# Run the realistic-scene comparison

For the exact measured runtime, use a checkout of commit `059b831` in the container and for protocol freezing. Use the current branch on the workstation for report generation and artifact/source audits; those utilities were extended after scoring. Copy the completed runs into that workstation checkout. On a later runtime revision, rerun the verification and preview commands below before freezing a new protocol.

Use the isolated Isaac Sim 5.1 ARM64 environment described in `README.md`, with this checkout at `/work`, the prepared original model artifacts mounted read-only at `/work/artifacts`, and the unchanged checkpoint mounted read-only at `/frozen/model.pt`.

Run the standard `python3` commands from the checkout on the host or workstation. Run `/isaac-sim/python.sh` and the suite script inside the container with `/work` as the working directory. Copy completed artifacts to the workstation before report generation. This container does not put a plain `python3` command on its default PATH.

```bash
python3 -m isaac_eval.realism_assets --out isaac_eval/assets --lock isaac_eval/realism_assets.lock.json
/isaac-sim/python.sh isaac_eval/verify_integration.py --scene paddock --out isaac_eval/runs/integration_realistic
# Run each command in a separate process; output directories must be new.
/isaac-sim/python.sh isaac_eval/preview_realism.py --scene plain --discipline trackdrive --out isaac_eval/runs/preview_plain_track
/isaac-sim/python.sh isaac_eval/preview_realism.py --scene paddock --discipline trackdrive --out isaac_eval/runs/preview_paddock_track
/isaac-sim/python.sh isaac_eval/preview_realism.py --scene plain --discipline skidpad --out isaac_eval/runs/preview_plain_skidpad
/isaac-sim/python.sh isaac_eval/preview_realism.py --scene paddock --discipline skidpad --out isaac_eval/runs/preview_paddock_skidpad
# Standard Python on the host can freeze the protocol; this step does not require Isaac.
python3 isaac_eval/freeze_realism_protocol.py
# Execute in the Isaac container. Existing suite output is deliberately rejected.
bash isaac_eval/run_realism_suite.sh
# On the workstation, with ffmpeg, matplotlib and numpy available:
python3 isaac_eval/report_realism.py --root isaac_eval/runs/realism_v1
# Audit downloaded bundles with OpenUSD (usd-core) and ffprobe available.
python3 isaac_eval/verify_realism_artifacts.py --root isaac_eval/runs/realism_v1
# Check the recorded runtime against the measured Git revision.
python3 isaac_eval/verify_frozen_sources.py --protocol isaac_eval/protocol_realism.json --revision 059b831 --out isaac_eval/runs/frozen_source_audit.json
```

The run script uses the suite's immutable protocol copy for every episode. It requires each summary to exist because this Isaac launcher can return success even after a Python exception. The reporter rejects missing cases, changed protocols, changed checkpoint/state hashes, mismatched tracks/start offsets and mismatched physics contracts. Do not rerun into existing output directories or combine episodes from different protocols.

