# Direct-command recovery trial v1

This trial tests whether two qualified on-policy recovery episodes improve the direct-command head. It is a data-only change to the previous recipe. It does not establish driving reliability or authorize a final-scene run.

Collect and verify the two episodes under `../direct_command_recovery/protocol.md`. The recipe expects the paths `runs/direct_recovery_v1/skidpad` and `runs/direct_recovery_v1/trackdrive`, with external receipts `runs/direct_recovery_v1/skidpad-receipt.json` and `runs/direct_recovery_v1/trackdrive-receipt.json`. It expects the rejected collection checkpoint at `runs/direct_command_trial3/step_500.pt` and pins its SHA-256. The original base checkpoint, manifest, membership, recovery gates, and recipe remain unchanged. Run the commands in the environment containing those artifacts and the pinned graphs. Preserve failed collections and do not start the trial if either receipt fails.

The audit reruns the original manifest verifier and compares it to its existing receipt. It separately invokes the original recovery gate for the old data. It reruns the new verifier and compares its entire result to each external receipt at every stage. It checks the original 7,006 TRAIN and 2,067 validation anchors before adding the two episodes by explicit path. It records the manifest verification, both new receipt hashes and results, expanded membership hash, derived counts, source archive, model state and flags, exact plain and hooked ABAB outputs and latent equality, fixed wiring anchors, and sampling protocol. Each receipt binds the checkpoint, arrays, source archive, metadata, collector, verifier, protocol, membership, and case. The audit fails if current replay or hashes differ.

Each batch has 16 old recovery draws, 16 new recovery draws, and 96 ordinary draws. Both recovery sets are excluded from ordinary draws. Each new recovery draw chooses the skidpad or trackdrive case with equal probability, then chooses a qualified first-two-second anchor within that case. Draws use replacement. The sampler records actual per-anchor counts. The fixed wiring set has 16 old ordinary anchors and eight recovery anchors from each new case.

Run each stage once in a new durable output directory from the repository root. Replace the example output directory with an unused path.

```sh
PYTHONPATH=src:. python .work/lab/direct_command_recovery_trial/trial.py audit --out runs/direct_command_recovery_trial_v1 --device cuda
PYTHONPATH=src:. python .work/lab/direct_command_recovery_trial/trial.py overfit --out runs/direct_command_recovery_trial_v1 --device cuda
PYTHONPATH=src:. python .work/lab/direct_command_recovery_trial/trial.py train --out runs/direct_command_recovery_trial_v1 --device cuda
PYTHONPATH=src:. python .work/lab/direct_command_recovery_trial/trial.py visual --out runs/direct_command_recovery_trial_v1 --device cuda
```

Run the next stage only after the previous one passes. The disposable wiring head must reach 0.002 rad and 0.05 m/s² MAE within 200 updates. Main training starts a fresh seed-zero head, not the collection checkpoint or the wiring head. It retains the 500-update optimizer, frozen FP32 base, history, validation, selection, visual gate, output contract, and telemetry of the previous trial. If no checkpoint passes the unchanged validation gates, do not run the visual gate. A passing visual gate does not replace the separate native pilot or physical qualification.

The CPU tests check sampler composition and counts, wiring membership, membership rejection, and failed or changed receipt rejection. They do not prove CUDA replay, training, selection, visual dependence, or driving performance.
