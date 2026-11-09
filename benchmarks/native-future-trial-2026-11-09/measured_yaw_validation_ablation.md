# Measured yaw-rate substitution worsens the frozen steering loss

The goal is native Isaac training that reduces cone contacts and timeouts with auditable videos. This local saved-output ablation supplies negative evidence against an unconditional measured-rate controller change.

All 256 existing validation anchors are included. Only the initial yaw-rate term changes. The native target remains exactly invariant. No inference, optimization, GPU, Spark call, fresh scene or physics run occurred.

| Checkpoint | Original loss | Substituted loss | Original MAE, rad | Substituted MAE, rad | Episodes with worse loss |
|---|---:|---:|---:|---:|---:|
| step0 | 0.00127634 | 0.00151726 | 0.00733931 | 0.00665262 | 6 of 8 |
| step500 | 0.00125601 | 0.00152644 | 0.00755987 | 0.00691373 | 6 of 8 |

Loss is the existing SmoothL1 with beta 0.05 against the unchanged native-controller target. Lower MAE does not remove the increased loss. These summaries measure different parts of the error distribution.

| Step-500 episode | Original loss | Substituted loss |
|---|---:|---:|
| val-skidpad-84000-matched | 0.00011289 | 0.00024978 |
| val-skidpad-84001-visual_stress_v1 | 0.00011071 | 0.00027813 |
| val-skidpad-84002-visual_stress_v1 | 0.00012890 | 0.00012748 |
| val-skidpad-84003-visual_stress_v1 | 0.00028220 | 0.00073078 |
| val-trackdrive-84000-matched | 0.00133913 | 0.00253857 |
| val-trackdrive-84001-visual_stress_v1 | 0.00293092 | 0.00301428 |
| val-trackdrive-84002-visual_stress_v1 | 0.00302727 | 0.00441364 |
| val-trackdrive-84003-visual_stress_v1 | 0.00211606 | 0.00085883 |

The body audit corrects the earlier finite-difference interpretation. It reconstructs instantaneous planar heading rate from the saved body-forward axis and world angular velocity. The centered heading slope averages 0.1 seconds and is a different measurement.

| Time, s | Raw world-Z rate | Exact instantaneous heading rate | Centered heading slope | Predicted initial rate |
|---|---:|---:|---:|---:|
| 0.20 | -0.01206559 | -0.01206550 | 0.00166205 | 0.00065248 |
| 2.00 | 0.03493384 | 0.03493398 | -0.00744822 | 0.00091526 |
| 4.00 | -0.62878662 | -0.62878974 | -0.65473947 | -0.58934575 |
| 24.00 | 0.73625755 | 0.73627673 | 0.70786311 | 0.65193492 |
| 26.00 | 0.71376032 | 0.71374552 | 0.62992898 | 0.62322903 |

Across 90 sampled frames, the maximum absolute exact-versus-world-Z difference is 0.00082585 rad/s. The maximum centered-slope-versus-exact difference is 0.09395910 rad/s. The 3D projection does not explain the large discrepancy here. Averaging, phase and dynamics remain unresolved.

The script verifies the replay array and anchor hashes, reproduces saved original actions, checks the exact native-target invariance, and binds the body readout to its source script and six producer input hashes. It recomputes the body formula from retained vectors and the heading slopes from hash-verified local frames. It does not reopen the remote raw frame NPZ files.

These correlated offline anchors do not establish a physical failure cause or a driving improvement. The rejected checkpoint remains ineligible. No runtime change or seventh optimizer run is justified by this result.

Run from the repository root with `python3 .work/lab/measured_yaw_validation_ablation.py`. Use `--check` to recompute and compare both retained reports without writing them.

The [JSON report](measured_yaw_validation_ablation.json) includes input and script hashes, all episode summaries, tail metrics and the five largest loss increases. The [script](measured_yaw_validation_ablation.py) uses only local files and NumPy. The [body producer](native_future_steering_readout2.py) and [body readout](native_future_steering_readout2.json) retain the corrected signal audit.

Build the Lever preserved the executable comparison. Model the Domain kept the intervention on bound replay arrays. Attack the Premise tested the proposed improvement across every existing anchor. Explain the Number kept instantaneous rate, averaged heading change, offline loss and physical outcomes distinct.
