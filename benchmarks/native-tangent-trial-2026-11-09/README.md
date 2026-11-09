# Rejected tangent training and target diagnosis

The fifth native experiment completed 500 policy updates in 474.736 seconds. All scheduled checkpoints failed the predeclared selection gates. Step zero was retained as an ineligible fallback. Only the declared final step was physically evaluated for diagnosis, with no promotion or fresh tests.

The exposed track run finished in 56.80 simulated seconds with one physical cone contact and one off-course event. The unchanged initializer with the same controller finished in 56.55 seconds with one contact and one off-course event. On skidpad, the trained checkpoint failed at 75.65 seconds with 17 contacts and five off-course events; the initializer violated the route at 12.55 seconds with nine contacts. The original model finished skidpad with one contact and timed out on track with 14. Each is one exposed run, not a reliability estimate. CPU video exports overlapped physical evaluation.

Exact saved-anchor replay reproduced validation and identified conflicting native-plan and recorded-teacher action targets. The TRAIN-only four-batch gradient diagnostic preserved every weight. It did not establish universally improved gradient alignment for the proposed target repair. Its complete numerical result is retained here, including the unfavorable measurements.

The archive preserves exact training, selection, diagnostic and telemetry bytes. The hash manifest identifies external raw NumPy arrays retained both locally and on Spark. Original checkpoints and complete camera evidence remain on Spark. Local reports and audited videos are under `runs/tangent_trial1_training_report` and `runs/development_media/tangent_trial1` in the active worktree. Source and calibration freezes are in the adjacent native-tangent-calibration archive.

The export wrapper audits and renders with the producer's NumPy 1.26, then verifies input and raw-video hashes while encoding in the CPU export environment. This avoids NumPy 2.4 controller-rounding differences without changing tolerances. Failed earlier audits remain in each episode directory. Every final video is fully decoded and checked for cadence, dimensions and frame count.
