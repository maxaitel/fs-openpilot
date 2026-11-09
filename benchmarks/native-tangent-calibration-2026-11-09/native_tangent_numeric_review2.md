# Independent numerical review before training

The user wants native Isaac training that reduces realistic cone contacts and timeouts, with auditable debug videos. This review approves the bounded 500-update experiment under protocol 2. It does not establish physical improvement or guarantee an eligible checkpoint.

## Decision

GO for protocol `docs/isaac-lab-training/tangent_selection2.json`, SHA-256 `f1ce3776281955b0a1b899a201089cea8b099b24c89c9959caa84468f5fb17ba`, and config `.work/lab/native_tangent_trial2.yaml`, SHA-256 `369a8bd944fda75dfb55463cde5431dece492a1e1eb36e36030176a09f501bac`.

The config binds calibration 2, SHA-256 `94f21a4d3741611578f38f83242ae8b9c7222eb65fff6b31db1188f1f6205300`. Independent checks matched its controller, manifest, initializer, every frozen training argument, scheduled steps and config protocol hash. All 79 calibrated source files match current `src` bytes.

## Rejected first draft

The original 13 coherence metrics have an exact trapezoidal nullspace. On a constant-speed straight plan, alternating x acceleration of +20 and -20 m/s² over knots 0 through 10 leaves every original metric at zero. The actual tangent controller changes its acceleration command from 0 to 0.4 m/s². The weighted plan L1 difference from that synthetic coherent plan is 0.1854774.

The executable reproduction is `tangent_nullspace_repro1.py`. Its actual output is `tangent_nullspace_repro1.json`. It uses an exact function snapshot and the actual controller implementation. This synthetic diagnostic establishes metric blindness. It does not establish that a trained model exploits the defect or satisfies all action gates.

The revised function measures each derivative endpoint against the secant of its corresponding position, velocity or yaw interval. The norm of the discrepancy is convex under linear interpolation, so its maximum lies at an endpoint. These three checks bound the identified cancellation within each measured interval, including the controller reads at 0.2 and 0.3 seconds. The final measured knot is 0.9765625 seconds. The new acceleration metric returns 20 m/s² for the reproduction.

## Numerical judgment

Calibration 2 gives acceleration/secant mean, p95 and maximum of 3.7897568, 6.7092464 and 8.5597805 m/s² for the initializer. Native targets give 0.3084191, 0.8901886 and 2.8961862 m/s². The revised ceilings of 1, 2 and 4 m/s² require approximately 74%, 70% and 53% reduction in the initializer's global statistics. They do not admit its existing defect by construction.

Independent calculation finds exactly 27 initializer failures among 459 gates. All failures are acceleration/secant mean, p95 and maximum, globally and in each of eight episodes. All 432 coherence bounds admit 1.25 times the corresponding native-target statistic. Thus there is no impossible native-coherence constraint caused by the chosen numerical caps. Compatibility with all action-loss constraints, or convergence within 500 updates, remains an experiment outcome.

The position interval maximum stays at 0.10 m. The cumulative position ceiling is at most 0.20 m, with the actual global bound 0.1813301 m. These reject the previously observed roughly 0.7 m notch. They are consistency limits, not cone-clearance guarantees.

## Scope and fallback

The action and plan retention bounds, off-policy 16/112 recovery mixture, full model retention, frozen vision, FP32 batches of 128, history context 8 and checkpoint schedule remain fixed. The selector requires strict steering improvement and every eligibility gate. The step-zero fallback is ineligible under protocol 2 and cannot be presented as a promoted model.

If all trained candidates fail, no checkpoint is promoted and no fresh test is opened. Exactly step 500 may be run on the two already exposed heavy cases for rejected-model diagnosis and failure videos, as declared before training. The report must preserve that rejected status. Selecting another failed checkpoint after viewing physical results would violate the declared experiment.

Explain the Number shaped this review. It changed the decision from accepting initializer-relative acceleration tolerance to detecting a structural blind spot, recalibrating native finite-interval error and requiring an explicit correction. Offline eligibility remains separate from native mission completion and contact outcomes.
