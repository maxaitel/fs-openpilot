# Rejected constant-offset vision plan prototype

The goal is reliable vision driving on full trackdrive and full RRLL skidpad with zero physical cone contacts. This archive preserves the rejected CPU prototype and its exact evidence. No physical qualification or final-case result is claimed.

The paired replay covers 133 original-model TRAIN precontact frames from one skidpad scene. Correction worsens lateral MAE from 0.3054 m to 0.4031 m and steering-direction agreement from 35 of 41 to 27 of 41. Support covers 97 of 133 frames. The result does not advance.

The artifact groups are separate.

- [REPORT.md](REPORT.md) is the readable report with links adjusted to this archive. Only its relative link destinations differ from the [exact original report](frozen_outputs/report.md).
- [frozen_source](frozen_source) contains the exact controller, config, protocol, replay, and five behavior checks. Its dependency subtree preserves all other files named by the pre-replay source freeze.
- [frozen_outputs](frozen_outputs) contains every original result, per-frame row, first and repeat stdout, freeze, repeat receipt, and report.
- [post_result_diagnosis](post_result_diagnosis) contains the later attribution source and outputs. It is separate from the frozen controller experiment.
- [evidence_bindings](evidence_bindings) contains exact copies of every file named by the frozen input hash manifest. RGB frame archives remain at the source evidence paths.
- [source_index.json](source_index.json) maps every copied artifact to its original path, hash, category, and pre-replay freeze status.
- [published_copy_receipt.json](published_copy_receipt.json) records copy hashes, evidence verification, frozen identity, and the publication-time behavior-check result.
- [verification/behavior_checks_stdout.txt](verification/behavior_checks_stdout.txt) records all five unchanged behavior checks passing from the published source copy.

Frozen scripts retain their original scratch-relative path assumptions. These exact archival copies are not a relocated runtime package. The source index records restoration paths. The report records the original replay command. Publication did not run another controller variant, inference, training job, or simulator.

The freeze hash is `1ee0f8fca1c0de7e72bed01d1d51f8682476adeca1e0d5dae45fc9acab5d0f6b`. The result hash is `5178985cde389e87bf0c7cd4159ec8a1737427e1ba32fe62bf7c14b5328a653a`. The rows hash is `38fc1a38d2c522401804d6021f038b126dfa109e4bc02d13d5376f542c9c29ea`.
