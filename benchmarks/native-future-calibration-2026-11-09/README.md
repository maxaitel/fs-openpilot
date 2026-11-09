# Native-future target calibration and pre-update freeze

The actual zero-update calibration ran on Spark with the full existing model. The strict builder verified identical native targets, predicted plans, coherence metrics, teacher-command diagnostics, initializer state, sampler and dataset identity against tangent calibration2. Only the versioned action target and implementation source changed. No optimizer history exists in the calibration directory.

The new action target is the deployed trajectory follower applied to detached native future states before mission braking. All 441 plan/coherence bound values remain unchanged. Only 18 aggregate/per-case action ceilings were calibrated under the new metric definition. Aggregate steering and longitudinal ceilings are 0.0012763423064825474 and 0.666768891736865. The global plan-L1 ceiling remains 0.9319418359547854. Acceleration/secant mean, p95 and maximum limits remain 1, 2 and 4 m/s².

The 143-file training freeze was verified on Spark before starting 500 updates. Independent review cleared the design and synthetic rejection checks. The implementation passed 378 CPU tests with four skips and seven subtests. The unfavorable TRAIN-only gradient-alignment result is included in the rationale. No action weights were tuned from it.

This archive establishes experiment identity, not a selected model or physical improvement. Selection requires actual completed training and the frozen gates. No fresh test was opened.
