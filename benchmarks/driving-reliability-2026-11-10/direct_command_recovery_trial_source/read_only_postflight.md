# Read-only postflight for the failed recovery trial

This verifier checks the existing six checkpoint artifacts and failed training evidence in a fresh CUDA process. It does not train, resume optimization, select a checkpoint, or authorize a visual or driving test. Step 500 is the fixed diagnostic checkpoint declared by the frozen trial when none pass. `passed` in its external receipt means only that the artifact and data checks completed. `candidate_qualified` and `optimization_resumed` remain false.

Run from the original repository root on the same CUDA runtime after review and deployment of this file. The original directory, source archive, checkpoint, data, and telemetry receipt must be present in their original paths. No other training process should be writing those files.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /isaac-sim/python.sh .work/lab/direct_command_recovery_trial/read_only_postflight.py
```

The script refuses an existing `runs/direct_command_recovery_trial_v2.postflight.json` and creates that external receipt exclusively only after all checks succeed. It does not write into the original run. Failure prints a traceback and produces no success receipt. Preserve the original training failure and all artifacts. The measurement receipt records the original process exit code 1, not a successful training completion.

The frozen preflight independently re-verifies data and both new recovery receipts and compares their results, the sampler protocol, source inventory, archive hash, frozen base state and flags against the original audit. It loads the datasets once, avoiding the second load that exhausted file descriptors in the original process. Archive entries are also checked against the audit. The verifier records current file descriptor counts and limits for context.

The 500 logged updates must be contiguous. The sampler reconstructs 500 seed-zero draws, verifies the 16 old recovery, 16 new recovery and 96 ordinary labels in each draw, and records reconstructed per-anchor counts. This is deterministic reconstruction from the audited protocol, not an independent observation of actual training draws.

For all six packages, it verifies hashes, contract, recipe, provenance, head shape and digests, and compares every one of the 417 embedded base tensors with both the audited state and the original checkpoint state. Each stored group's pass flag and the overall flag are recalculated using the unchanged original gates. Every recorded validation must be rejected. Only the full 2,067-anchor step 500 validation is recomputed on CUDA. Its complete metrics must exactly match the original record; no numerical tolerance is widened. The other five validations are checked as source-bound records, not independently recomputed. File hashes are checked before and after verification. These checks do not establish physical driving performance.
