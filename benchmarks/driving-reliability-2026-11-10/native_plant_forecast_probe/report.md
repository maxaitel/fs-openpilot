# Bicycle forecast against physical precontact pilots

The nominal bicycle forecast exceeds the proposed 0.20 m diagnostic pad in 3 of 62 overlapping one-second starts in joint_command_pilot1. Its largest front-effect error is 0.2395 m. None of the other measured pilot/horizon groups exceeds 0.20 m. Known future recorded actions make these optimistic replay diagnostics, not deployable forecasts or safety evidence.

Each cell below is p95 / maximum. Position and front-effect are meters, yaw is radians, and speed is m/s. Counts are overlapping starts in one correlated scene. The front-effect value is position error + 1.6 * absolute yaw error. This proxy is not the exact front-corner displacement and 1.6 m is shorter than the nominal CoG-to-corner radius.

| Pilot | Horizon s | n | Position | Yaw | Speed | Front effect | Over 0.20 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct_command_pilot3 | 0.05 | 79 | 0.0089 / 0.0100 | 0.0010 / 0.0012 | 0.0231 / 0.0741 | 0.0097 / 0.0107 | 0 |
| direct_command_pilot3 | 0.25 | 75 | 0.0499 / 0.0612 | 0.0024 / 0.0028 | 0.0709 / 0.0958 | 0.0529 / 0.0646 | 0 |
| direct_command_pilot3 | 0.50 | 70 | 0.0961 / 0.1111 | 0.0033 / 0.0036 | 0.1467 / 0.1594 | 0.1016 / 0.1161 | 0 |
| direct_command_pilot3 | 1.00 | 60 | 0.1433 / 0.1607 | 0.0026 / 0.0029 | 0.2553 / 0.2646 | 0.1457 / 0.1623 | 0 |
| direct_command_recovery_pilot2 | 0.05 | 52 | 0.0098 / 0.0100 | 0.0011 / 0.0015 | 0.0409 / 0.0743 | 0.0109 / 0.0114 | 0 |
| direct_command_recovery_pilot2 | 0.25 | 48 | 0.0583 / 0.0823 | 0.0047 / 0.0060 | 0.0745 / 0.1318 | 0.0658 / 0.0919 | 0 |
| direct_command_recovery_pilot2 | 0.50 | 43 | 0.0859 / 0.1215 | 0.0056 / 0.0078 | 0.1230 / 0.1258 | 0.0949 / 0.1341 | 0 |
| direct_command_recovery_pilot2 | 1.00 | 33 | 0.1519 / 0.1660 | 0.0075 / 0.0102 | 0.2200 / 0.2459 | 0.1638 / 0.1823 | 0 |
| joint_command_pilot1 | 0.05 | 81 | 0.0108 / 0.0112 | 0.0021 / 0.0023 | 0.0330 / 0.0408 | 0.0141 / 0.0149 | 0 |
| joint_command_pilot1 | 0.25 | 77 | 0.0722 / 0.0797 | 0.0053 / 0.0056 | 0.1029 / 0.1394 | 0.0809 / 0.0856 | 0 |
| joint_command_pilot1 | 0.50 | 72 | 0.1260 / 0.1583 | 0.0071 / 0.0082 | 0.1852 / 0.1992 | 0.1369 / 0.1711 | 0 |
| joint_command_pilot1 | 1.00 | 62 | 0.1830 / 0.2292 | 0.0049 / 0.0065 | 0.2357 / 0.2569 | 0.1850 / 0.2395 | 3 |

The [frozen protocol](protocol.md) and [scorer](probe.py) use every eligible start with the endpoint strictly before the contact interval. All 120, 60, and 120 evidence frames were read and verified. First contacts are 80, 53, and 82. Results and source hashes are in [result.json](result.json). Two complete runs produced the same result SHA-256, `5e3c091d89e74eda15779c847c86a39c6be98aa86837d4ccdc9e4066ab741372`. All metadata vehicle parameters match nominal exactly, so the privileged parameter replay is identical and cannot measure parameter uncertainty.

The recorded action is the command passed to the physical drive, as shown in `isaac_lab_fsop/evaluate.py:282,317`. Physical `LabDrive` applies its actuator update once per 0.05 s and takes six PhysX steps of 1/120 s (`isaac_lab_fsop/env.py:230-233,255-261`, `isaac_eval/vehicle.py:119-122`). The scored bicycle uses five 0.01 s integration steps, the pure model native `FSEnv` convention (`src/fsop/sim/env.py:41-43,203-209`). These are different plants and their actuator lags discretize differently. The initialization takes the recorded actual actuator state (`isaac_lab_fsop/env.py:199-213`); the pure model updates that state within each integration substep (`src/fsop/sim/vehicle.py:72-79`). The bicycle reference is its ground-projected CoG (`src/fsop/sim/vehicle.py:1-2`); physical recorded pose uses the chassis root. This probe applies no inferred reference shift, and exact collider correspondence remains unverified.

The next falsifier is a separately frozen, causal forecast on additional TRAIN trajectories with future candidate controls, measured actuator state and an explicitly verified reference point. Compare swept actual footprint displacement, not only this proxy, across braking and turning regimes. Include latency, perception error, and physical model variation before choosing any combined margin. A pass here does not establish zero contact or full-route completion.

Explain the Number fixed the 0.20 m decision criterion before running and kept correlated starts and optimistic future action access explicit. Model the Domain kept each pilot/horizon replay local to its recorded sequence.
