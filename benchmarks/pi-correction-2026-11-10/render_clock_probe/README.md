# Fixed-action render and physics clock probe

The user goal is to reduce skidpad cone cuts and report honest retraining results. This throwaway probe checks whether capture, evidence writing, or elapsed wall time advances physics. It does not evaluate a model or justify a training-improvement claim.

Two fresh native Isaac Lab processes ran the frozen `skidpad-72000-heavy-recorded` regression case with `visual_stress_v1`. Each ran eight control ticks with steering 0 and acceleration 2. The first used the existing EvidenceWriter. The second used the existing AsyncEvidenceWriter. No runtime source was edited. No model was loaded. Each payload contains actual raw RGB and an explicitly marked zero-plan serialization fixture.

## Predeclared protocol and preserved failure

`protocol.json` was saved and copied before the first run. It specifies 1e-9 seconds clock tolerance, exact integer steps, 1e-7 stationary pose tolerance, and 1e-5 cross-process pose tolerance. It also requires a 25 GiB reserve and less than 100 MB of probe artifacts.

The original `compare.py` result is **false** with 48 failures. Its assumptions about the timeline clock were wrong, and its rational timestep tolerance was too tight for float32. These original criteria and failures remain in `comparison.json`. `interpretation.json` is explicitly a post-observation interpretation, not a replacement passing gate.

## Observed result

| Operation | Samples | Physics clock delta | Physics step delta | PhysX pose and velocity delta | Timeline delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Capture | 16 | 0 | 0 | Exactly 0 | 0.05000000074505806 s |
| Evidence append | 16 | 0 | 0 | Exactly 0 | 0 |
| Idle at least 0.15 s | 16 | 0 | 0 | Exactly 0 | 0 |
| Six-step control advance | 16 | 0.05000000260770321 s | 6 | Vehicle moves | 0 |
| Writer close | 2 | 0 | 0 | Exactly 0 | 0 |

The measured advance is exactly `6 * float(np.float32(1 / 120))`. Its difference from rational 0.05 is 2.6077032061477823e-9 seconds, exceeding the original 1e-9 tolerance. No timestep rounding mismatch remains after comparing with the actual float32 representation. The timeline and physics clocks advance on different operations in this configuration. A timeline increment during capture is not evidence of physics advancement.

All 64 paired operation boundaries have exactly equal physics time, timeline time, physics step index, direct PhysX transforms, and direct PhysX velocities across the two processes. The smallest measured idle interval is 0.15009913500398397 wall seconds.

Initial raw images differ despite equal physical state. Road MAE is 0.7262420654296875 and wide MAE is 0.4871978759765625 on the 0 to 255 scale. Maximum channel differences are 18 and 21. This is observed renderer repeat variation. The initial images are captured before either evidence writer is created.

Captured global settings, camera product paths and resolution, profile, and restored simulation render settings match exactly. The two sensor render products differ only in `viewPickingId`. Several attributes differ on the disabled viewport render product. Those differences are listed in `interpretation.json` and are not attributed as the cause of sensor pixel differences. These snapshots do not expose all possible renderer internal state.

## Artifacts and verification

`probe.py` records each boundary using simulator `current_time`, `current_time_step_index`, timeline time, and direct PhysX view transforms and velocities. Direct PhysX reads avoid relying solely on cached LabDrive state. Each row also records camera frame identity and raw-image SHA-256 digests.

Each run has its source inventory, exact case hash, probe/protocol hashes, render snapshots, raw initial images, trace, eight compressed fixture records, an evidence completion receipt, and a probe completion receipt. Source inventories match across runs and each process verified the inventory remained unchanged. Completion files were written before simulator shutdown. Both child processes exited zero, and `compare.py` independently validated complete receipts, all NPZ hashes, manifest hashes, trace hashes, counts, and script identities after child exit. It still returns one because of the preserved clock criteria failures.

Both runs verified at least 25 GiB available before simulator startup. The combined remote directory was 8,875,125 bytes before this final report, far below 100 MB. The post-run filesystem check showed 30,734,020,608 free bytes. No co-running CPU job was stopped.

Local artifacts are under `.work/lab/render_clock_probe`. Remote artifacts are under `/work/probes/render_clock_probe1` in `fsop-isaac-lab-training`, backed by `/home/maxaitel/fsop-isaac-lab-training/probes/render_clock_probe1` on `max-spark`.

Reproduce on fresh output directories with the existing container:

```sh
docker exec -w /work -e PYTHONPATH=/work:/work/src:/work/isaac_eval fsop-isaac-lab-training /isaac-sim/python.sh /work/probes/render_clock_probe1/probe.py --mode sync --out /work/probes/render_clock_probe1/sync
docker exec -w /work -e PYTHONPATH=/work:/work/src:/work/isaac_eval fsop-isaac-lab-training /isaac-sim/python.sh /work/probes/render_clock_probe1/probe.py --mode async --out /work/probes/render_clock_probe1/async
docker exec -w /work fsop-isaac-lab-training /isaac-sim/python.sh /work/probes/render_clock_probe1/compare.py /work/probes/render_clock_probe1
docker exec -w /work fsop-isaac-lab-training /isaac-sim/python.sh /work/probes/render_clock_probe1/interpret.py /work/probes/render_clock_probe1
```

The output directories intentionally refuse overwrite. The comparison is expected to return one under the unchanged original protocol.

## Decision and limits

This probe supports physics isolation for capture, evidence append/close, and idle boundaries over these short fixed-action traces. It provides no evidence that elapsed wall time advanced vehicle physics. It also confirms that pixel repeat variation can occur before a writer exists. It does not establish learned-driving repeatability, isolate the precise RTX noise mechanism, or cover a full episode. A learned-policy comparison still needs fresh-process repeats and complete physical results under the same writer/settings configuration.

Model the Domain shaped the boundary-row structure and marked serialization fixture. Explain the Number kept the original failures and separated the physics clock from the timeline clock instead of retroactively declaring a passing gate.

Throughput checkpoint was eight ticks per variant. Two native runs completed all requested work with under 10 MB of artifacts. No model training or inference occurred.
