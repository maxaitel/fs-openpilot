# Isaac Sim zero-shot results

trackdrive: 5/5 passes, 6 cone hits; skidpad: 5/5 passes, 13 cone hits.

[Overview video](runs/suite_v2/overview.mp4) · [Full report and all videos](runs/suite_v2/report.html) · [Outcome graph](runs/suite_v2/outcomes.png) · [Performance graph](runs/suite_v2/performance.png)

| Case | Result | PhysX cone hits | Off-course events | Sim seconds | Wall seconds |
|---|---|---:|---:|---:|---:|
| trackdrive_91000 | PASS | 0 | 0 | 42.85 | 129.8 |
| trackdrive_91001 | PASS | 2 | 0 | 42.85 | 129.2 |
| trackdrive_91002 | PASS | 2 | 0 | 41.65 | 126.2 |
| trackdrive_91003 | PASS | 1 | 0 | 42.00 | 127.2 |
| trackdrive_91004 | PASS | 1 | 0 | 41.80 | 126.1 |
| skidpad_91000 | PASS | 3 | 0 | 47.00 | 137.4 |
| skidpad_91001 | PASS | 2 | 0 | 47.00 | 137.6 |
| skidpad_91002 | PASS | 2 | 0 | 46.55 | 137.0 |
| skidpad_91003 | PASS | 4 | 0 | 46.95 | 138.4 |
| skidpad_91004 | PASS | 2 | 0 | 46.95 | 137.8 |

Aggregate speed: 0.336× real time, 6.72 steps/s. Mean GPU busy: 78.4%; mean power: 31.8 W. All checkpoint and state hashes remained unchanged.

Only 1/10 runs was contact-free. Twelve of the thirteen skidpad contacts occurred from 0.2 seconds before to 1.6 seconds after the right-to-left crossover. Completion success therefore does not imply clean cone avoidance. All scored cases had zero off-course events.

The scored driving loops took 1,326.8 seconds (22.1 minutes), plus 109.2 seconds of initialization/warmup. The combined loop averaged 148.9 ms per step: 99.5 ms rendering, 36.3 ms model/controller under graphics contention, 8.4 ms physics, and 2.9 ms video work. The separate fixed-image model-only benchmark averaged 5.72 ms (5.74 ms p95; 500 calls after 20 warmups).

Mean reported GPU power was 31.8 W; peak GPU temperature was 72°C. Peak unified host RAM usage was 12.17 GiB, process RSS 5.45 GiB, and Torch allocated memory 0.137 GiB. These measurements are not dedicated VRAM usage or whole-system wall power.

The generic FS-sized PhysX Vehicle car passed camera projection/freshness, motion, higher-speed braking and physical-contact checks. Camera marker errors were below 2 pixels and capture did not advance physics. Model parity against the original Torch runtime was within 1e-4 tolerance. The checkpoint and in-memory model state stayed unchanged in every run.

A rejected rigid-wheel prototype produced an unscored setup failure. Its labeled recording is appended to the overview and retained separately. The final vehicle used NVIDIA's tire/suspension solver. This is a small pilot: five lighting/start-pose variants on one fixed cone track and one standard skidpad. The existing ground-truth route commands and mission launch/finish braking remain in use.

All code and output are in the separate `isaac-zero-shot-eval` worktree; the training source, checkpoint and training container were left unchanged. See [integration details](README.md) and the [frozen protocol](protocol.json).
