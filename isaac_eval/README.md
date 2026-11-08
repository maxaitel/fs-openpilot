# Frozen-model Isaac Sim evaluation

This directory is an independent evaluation adapter. It does not train a model or change the existing Panda3D training/simulation code. Run artifacts are ignored by Git.

The selected checkpoint is the full 31,154,802-parameter `skidpad_clean/final.pt`, SHA-256 `cf3a8d253dfd5a417e1967d2d6dc8cdd0263e3d16d3f1f02f15495f490a83b7e`. The checkpoint and prepared ONNX graphs are mounted read-only. All parameters have gradients disabled; inference runs in `torch.inference_mode()`. Every scored episode verifies both the checkpoint hash and the in-memory state-dictionary hash before/after execution.

## Vehicle and sensing

No suitable Formula Student physics-ready USD was found in the available Isaac asset collection. The supplied NVIDIA Leatherback was inspected and rejected: its actual asset is an approximately 0.44 m RC car. This evaluation uses a procedural generic Ackermann car: 230 kg, 2.90 m length, 1.40 m overall width, 1.55 m wheelbase, 1.20 m track, and 0.23 m wheel radius.

`vehicle.py` uses NVIDIA's **PhysX Vehicle tire/suspension solver**, with rigid chassis/wheel colliders for cone contacts. It has 80 mm suspension travel, 20 kN/m springs and 2 kN·s/m dampers per wheel. The unchanged controller's acceleration command becomes wheel torque (`mass × acceleration × radius / 4`); steering is mapped to individual Ackermann wheel angles. Nominal actuator lags and limits are preserved. This is a generic dynamics model, not a validated digital twin of a particular FS car.

Both model cameras are RGB, 512×256, mounted nominally at 1.10 m height with x-forward/y-left/z-up axes. Road intrinsics are `fx=fy=910, cx=256, cy=47.6`; wide intrinsics are `fx=fy=455, cx=256, cy=151.8`. They follow the physical chassis. Existing YUV packing, frame pairing, 20 Hz history/desire queues, plan conversion and `PlanPurePursuit` are reused without changes.

The policy receives the existing ground-truth route-command pulses and mission-progress launch/finish braking. It receives no ground-truth steering path. Completion therefore does not test autonomous lap-counting or route-command perception.

RTX snapshots use Replicator with `delta_time=0` and four rendering subframes. Antialiasing and motion blur are disabled; the supported DL denoiser stays enabled. The unused viewport is disabled. Physics runs at 120 Hz and the controller at 20 Hz of simulated time, independently of wall-clock speed.

## Verification and scoring

`verify_integration.py` checks static and moved-camera marker projection, capture without advancing physics, straight/left/right motion at low and higher speed, braking, stability, and a deliberate cone collision with a nearby negative control. Bright marker cores are used for projection measurement to exclude weak RTX reflections/trails; model images are not filtered this way.

`verify_model.py` runs an identical six-frame fixture in the Isaac Torch runtime and the original NGC PyTorch runtime. The recorded maximum plan difference was 0.0000344 and maximum action difference 0.0000072, within `rtol=atol=1e-4`.

`protocol.json` freezes the checkpoint and source hashes before the scored suite. There are five seeds per discipline, one fixed generated cone track (one lap plus stop), and the full standard skidpad (two right laps, two left laps, exit and stop). Seeds vary lighting, starting lateral offset ±0.10 m and yaw ±0.02 rad. This small pilot does not measure broad track generalization.

The time limit is 120 simulated seconds. Early failures are five continuous seconds fully off-course, rollover, invalid skidpad route, standstill or nonfinite model output. **Cone hits are unique cones with a PhysX contact against the car**, not the old geometric footprint approximation. The latter is retained separately as a diagnostic. Existing referee penalties use the legacy approximation and should not be interpreted as PhysX penalties.

A rigid-wheel articulation prototype was rejected after failing higher-speed braking tests. Its one driving recording is retained as an excluded setup attempt, not counted in the scored suite. Renderer shader errors and chase-camera aperture were also corrected before the final suite. No checkpoint or driving-controller tuning was performed using these attempts.

## Reproduction on Spark

The isolated container uses ARM64 `nvcr.io/nvidia/isaac-lab:2.3.0` (Isaac Sim 5.1.0, Torch 2.9.0+cu130). Additional packages inside that container: `onnx2torch==1.5.15`, `shapely==2.1.2`, `nvidia-ml-py==13.615.71`.

Mount this checkout at `/work`, the original prepared artifacts at `/work/artifacts:ro`, and the frozen checkpoint at `/frozen/model.pt:ro`. Set `PYTHONPATH=/work/src`; use a separate cache mount and `--gpus all --shm-size=8g`. Keep the training container stopped.

From `/work`, run verification with `/isaac-sim/python.sh isaac_eval/verify_integration.py`. The scored suite runs with `bash isaac_eval/run_suite.sh`. The runner refuses modified hashed source files or a different checkpoint, and refuses to overwrite episode directories. Recompute a protocol deliberately for a new experiment; do not silently reuse this experiment's name.

Per episode, artifacts include the USD scene, track JSON, full raw video, start/end frames, plans, per-frame timing/state/action records, 1 Hz hardware telemetry and summary. Startup/warmup time is separate from timed driving. GPU utilization/power are whole-device measurements; host RAM is unified-system usage, not a dedicated VRAM allocation. Torch memory is reported separately.

## References

- [Isaac Sim 5.1 requirements, including DGX Spark](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [PhysX Vehicle APIs and tire/suspension simulation](https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/physics/vehicles.html)
- [PhysX vehicle USD representation and direct wheel control](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.physx.vehicle/docs/usd_representation.html)
- [Isaac 5.1 synchronized Replicator capture](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html)

The vehicle setup follows the APIs demonstrated in the installed NVIDIA `omni.physx.vehicle` BasicSetup and WheelController samples. No proprietary vehicle asset is redistributed.
