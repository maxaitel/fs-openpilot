# Realistic-scene frozen evaluation

This branch adds an appearance-only outdoor test paddock to the isolated Isaac evaluation. It does not train the model or modify the training repository. The existing generic PhysX Ackermann car has original Formula Student-style visual bodywork; it is not a validated digital twin of a particular FS car.

The scene uses scanned pavement, roughness and normal maps, a photographed 4K daylight environment, modeled cones with beveled bases and molded tips, restrained scuffs, irregular surface repairs, drainage grills, old parking bays, fencing, barriers, a workshop, tents and equipment. The car has shaped bodywork, an open cockpit, roll hoop, suspension, slick tires and detailed rims. Background scenery is non-colliding and placed beyond the cone envelope. The distant trees/sky are an HDR environment, not nearby 3D vegetation.

Powered by [Poly Haven](https://polyhaven.com). Downloaded assets use its [CC0 asset license](https://polyhaven.com/license). Authors, original URLs, dimensions and hashes are in `realism_assets.lock.json`. Generated venue, cones and car geometry are original procedural assets. The asphalt uses the scanned 2 m scale and an explicitly recorded 4x linear albedo gain; this is an artistic appearance adjustment, not a measured reflectance calibration.

See [Run the realistic-scene comparison](REALISM_RUN.md) for setup and commands. The measured executable source is commit `059b831`; later documentation and comment cleanup do not change the recorded experiment.

## Verification and experimental contract

`scene.py` and `vehicle.py` remain unchanged. `appearance_guard.py` compares physical schemas, attributes, relationships, collider topology/dimensions and ancestor transforms before and after dressing. Render-only meshes are siblings of hidden collider visuals, so cones and wheels follow their actual rigid bodies. Existing ground and tire physics material bindings remain intact. The integration gate exercises camera projection before/after a pose change, zero-time capture, left/right steering, low/high-speed braking, upright stability and positive/negative cone-contact controls.

The road and wide model inputs retain their original 512×256 resolution, intrinsics, principal points, mount, packing and temporal policy. The existing controller, ground-truth route commands and mission progress remain in use. No ground-truth steering is added. Weight files are mounted read-only, inference runs without gradients, and file plus in-memory model hashes are checked per episode.

Four render subframes left old-pose marker ghosts under the HDR lighting. Eight still failed the unchanged 2-pixel projection threshold. An accumulation reset and DLSS Quality mode at four subframes also failed. The selected shared profile uses 16 subframes, DLSS mode 0, the required DL denoiser, no motion blur and the original RTX renderer. It passed the same projection/dynamics gate. Both plain and paddock arms use this profile. The fresh paired results must not be described as a timing comparison with the older four-subframe benchmark. NVIDIA documents increasing subframes for ghosting in its [Isaac 5.1 troubleshooting guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/troubleshooting.html).

Final preview/integration receipts contain executable scene, camera-source and asset-manifest hashes. Protocol freezing rejects stale evidence. Exported scenes use relative texture paths with dependency closure verified by OpenUSD. Each episode retains its scene bundle, protocol, appearance receipt, summary, raw model plans, per-step observations/timing, telemetry, frames and uncut video.

The paired suite has five seeded trials for each of two fixed geometries, in alternating AB/BA order. Seeds vary starting pose and scene-specific details, including plain-scene lighting and paddock surface repairs. Each pair shares the same start and physical layout. It reports route completion, completion plus safe-stop pass, unique physical cone contacts and contact-free completions separately. A completed run with cone hits is not a clean run. Hardware values describe the simulator plus inference workload; host unified RAM, process RSS and Torch allocations are separate quantities. Video follows simulation time, not the slower wall-clock execution.

## Scope

This is a stronger visual simulation test, not evidence of real-world safety or a calibrated sim-to-real result. It covers dry daylight, two fixed layouts and small starting-pose perturbations. Tire friction is unchanged; there is no rain, rolling shutter, measured camera ISP/noise, moving traffic or pedestrian interaction. Most venue props are deliberately outside the driving area and have no collision shapes. Cone bevels differ slightly from their retained collider envelopes. The car retains its original box chassis collider beneath the detailed bodywork. The visual nose tapers inside that box, so a physical contact can occur outside the visible body surface. An HDR background cannot supply close-range parallax. The proper next expansion is separately named, measured lighting/sensor/physics conditions, not training on the held-out evaluation scenes.

## Design and review

Two independent design candidates were compared under poteto-mode. The selected small scene overlay avoids introducing a general scene framework. It incorporates the other candidate's single-HDR lighting and portable export requirements. Both configured reviewers are GPT-family models; no different-family judge was available. The review caught and led to fixes for stale evidence binding, protocol mutation during a suite, and mismatched inspection/scoring capture settings. `REALISM_DECISIONS.tsv` records retained and rejected experiments.
