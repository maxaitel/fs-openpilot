# Paired visual stress method

This exploratory experiment uses four preselected cases from the original frozen 41-case manifest. Each is run once with the matched appearance and once with `visual_stress_v1`. The intervention adds appearance-only pavement, static people, walls and clutter. It does not test pedestrian safety or physical obstacle avoidance. Each arm retains the original checkpoint, model inputs, controller, physics, camera configuration and referee. Later trajectories can diverge, so later images are not matched inputs and differences cannot establish isolated perception causality.

Prepare the full manifest and local assets, then finish all code changes before recording the four case previews and the two 120-frame pilots. Record fresh vehicle and camera verification in the same renderer runtime. Verify each pilot with `isaac_eval.verify_inference` and retain its sealed evidence. Preview each case with `isaac_eval/preview_stress.py`; review the road and wide model cameras at the fixed route poses before scoring. Changes to the runtime or assets require new previews and gates.

Use the verified Isaac renderer container for previews, pilots and episodes. Keep the CPU video export environment separate. Installing Rerun 0.42 in the renderer upgraded NumPy and prevented Isaac camera initialization in this setup. Do not install it in the renderer. The existing `scripts.process_isaac_parity` processor can run separately on each variant folder after the eight episodes complete.

Freeze from the same working directory and runtime, with all input receipts retained in place.

```bash
/isaac-sim/python.sh -m isaac_eval.stress_experiment freeze \
  --manifest isaac_eval/runs/parity_cases/manifest.json \
  --vehicle isaac_eval/runs/stress_vehicle_gate/verification.json \
  --camera isaac_eval/runs/stress_camera_gate/verification.json \
  --matched-pilot isaac_eval/runs/stress_pilots/matched \
  --stress-pilot isaac_eval/runs/stress_pilots/visual_stress_v1 \
  --previews isaac_eval/runs/stress_preview_v1 \
  --out isaac_eval/runs/stress_v1/experiment.json
/isaac-sim/python.sh -m isaac_eval.stress_experiment run --plan isaac_eval/runs/stress_v1/experiment.json
```

The runner follows the frozen case-major order, alternating matched and stress for each case. It prints progress and records each return code and elapsed time. It stops at a technical failure and keeps the partial output. It writes independent `matched` and `visual_stress_v1` folders, each with a `status.json` compatible with the existing CPU processor. Process each folder, then run `python -m scripts.report_isaac_stress --plan isaac_eval/runs/stress_v1/experiment.json` in an environment with access to the frozen plan inputs and the raw evidence. The report verifies the completion seal, audit and processor identities, frame counts, RRD and both saved-frame video hashes before presenting pairs. It checks the paired initial state, case, physics, realization, model state, sources, graphs and calibration. A physically clean episode must pass the referee with zero contacts and zero off-course events. Totals include failures and terminal results. The gallery is portable when copied with its relative video paths; re-auditing requires the original frozen inputs and mounts. The report makes no claim about the full 41-case suite.

A scored episode binds its exact original case specification, plan, visual variant, appearance receipt, flattened scene, copied scene assets and export manifest. The auditor verifies these identities and then independently replays the saved states through the original mission and referee. The model is not replayed for each scored episode, physical contacts are not independently resimulated, and rollover terminal evidence retains the existing audit limitation. The videos use captured inference and the offline expert reference; they do not regenerate model predictions.

The original parity protocol and its 41-case validation remain unchanged. Since runtime source inventories include every top-level `isaac_eval` module, historical 41-case evidence must be reproduced in its preserved original checkout and runtime.

## Scene decisions before scoring

The pavement uses the existing scanned asphalt textures with an opaque vertex-color weather field. It adds branching tar lines, paired tyre marks and broken paint. The static clutter includes three textured adult character assets, concrete walls, blue and yellow barrels, tyre stacks, worktables and wheeled equipment. A deterministic layout checks each prop footprint against the full driving corridor and the cones. The scene builder checks that it has not consumed the environment RNG or changed the physics contract.

Actual RTX camera previews rejected two approaches before any scored episodes. Rectangular repair overlays looked like scattered panels. Feathered opacity produced glass-like patches. An opaque material probe worked, but the first integrated version lost texture because some shader connections crossed material boundaries. Local texture and primvar nodes restored the asphalt detail. The accepted development preview is `stress_preview_dev5`; the separately captured `stress_preview_v1` receipts bind the final sources and assets.

The four cases and alternating run order were selected before scoring. Source review found six experiment/report integrity defects, including acceptance of an edited summary and an incomplete physical-clean definition. These were fixed before the final gates. Tests now reject those changes and require sealed summaries, video provenance, frame counts and paired initial-state invariants. The full CPU suite passed 210 tests with four skips. A separate source review found no remaining blocking defects. Renderer and inference gates remain independent requirements.

Hardware samples cover the episode loop. The report derives median and p95 utilization, memory, temperature and power from the recorded samples and shows missing values explicitly. Telemetry is bound by the artifact audit, not by the original completion seal. Each case runs once per appearance, so hardware timings describe these instrumented episodes and cannot establish a repeatable speedup or regression.
