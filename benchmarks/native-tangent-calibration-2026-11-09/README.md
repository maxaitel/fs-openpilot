# Native tangent-controller calibration

These are actual zero-update Spark calibrations of the full model under `op_tangent_v1`. Neither calibration performed optimization. They precede the first 500-update tangent experiment.

Calibration 1 exposed a gap in the proposed consistency checks. Alternating acceleration knots can cancel in trapezoidal integrals while changing the controller action. The executable synthetic reproduction and actual output are included. This is a constructed counterexample, not an observed trained-model exploit.

Calibration 2 adds endpoint-versus-secant checks. The frozen numerical protocol is [tangent_selection2.json](../../docs/isaac-lab-training/tangent_selection2.json), with [rationale and failure policy](../../docs/isaac-lab-training/tangent_selection2.md). The initializer intentionally fails 27 acceleration gates. No trained checkpoint is promoted merely because it improves average loss.

Each calibration includes its exact trainer source archive, run identity, validation metrics and calibration record. The freeze receipts bind runtime, selector, report, configuration and protocol bytes. The independently replayed camera/inference pilot contains 120 frames; its debug and research videos remain local and on the Spark. The pilot is not a completed mission.

The copied scripts retain their executed `.work/lab` paths. To reproduce, restore them at the paths named by their receipts, use the pinned Spark containers and the verified native v4 dataset, and select unused output directories. The dataset arrays, prepared graphs and original checkpoint remain on the Spark and are not included in this small archive. Do not overwrite historical run outputs.

`sha256.json` inventories the archive bytes. Training results and physical evaluations are separate artifacts.
