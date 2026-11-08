# Skidpad work plan

The goal is a learned model that completes the skidpad route, responds to route commands, and produces a recorded run on the Spark.

## Checklist

- [x] Read the Principles section of the poteto-mode skill.
- [x] Phase A: Frame.
- [x] Phase B: Design the workflow.
- [x] Phase C: Run the loop.
- [x] Phase D: Keep the audit trail.
- [x] Phase E: Verify and hand back.

## Verification contract

Measure the existing checkpoint before changing it. Keep image and vision history identical while changing the command history. Inspect input delivery and original ONNX parity with nonzero commands. An input being connected does not establish useful command dependence.

The driving target is the ordered right, right, left, left, exit sequence, followed by a stop in the zone. Require independent physical route validation, no off-course events, and report all cone hits. Evaluate at least ten held-out venue scenes across light and heavy randomization. Preserve the existing eight-track trackdrive comparison. Keep the test scenes out of training and checkpoint selection. A working demonstration alone does not establish robustness.

## Experiment sequence

1. Capture pulse statistics, command ablations, physical route traces, and a baseline.
2. Compare two command-lifetime and route-validation designs before changing shared code.
3. Fix evidenced defects and add behavior tests. Re-run the unchanged checkpoint to separate code effects from training.
4. Record route-conditioned skidpad demonstrations, validate their route and command coverage, and mix replay data from the earlier training split.
5. Adapt the existing full checkpoint on the Spark. Select on validation data. Refresh feature history if needed.
6. Run paired final driving tests and command ablations. Record a predetermined skidpad scene and export training graphs.
7. Independently review the changes and decision trail. Save the report and artifacts, and stop only the dedicated task container.

## Throughput checkpoint

The command tensor is `[B,25,8]`, the feature history is `[B,24,512]`, and the model appends the current 512-element vision feature. Simulation runs at 20 Hz and recording at 5 Hz. Use existing optimized Spark training settings. Profile only if new measurements expose a bottleneck. Avoid a fresh full-dataset run before checking the command path and scoring.

Investigation has two read-only slices, command delivery and skidpad geometry/control/scoring. Each returns evidence to the parent. Design candidates own separate files. Implementation has one writer per source file. All GPU jobs are serialized by the parent.

## Known constraints

The initial workspace contained earlier Spark changes, which were preserved and committed separately. The user's fork is https://github.com/maxaitel/fs-openpilot. The `fork` remote targets it; `origin` remains upstream. Datasets and raw outputs stay on the Spark. Final weights, graphs, and video are copied locally. The configured review models are from the same family.

The decision trail is `docs/skidpad-decisions.tsv`. A claim that skidpad is working requires the measured physical route result, not the existing referee's crossing count alone.
