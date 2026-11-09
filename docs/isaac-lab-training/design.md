# Isaac Lab native training

The active learning task remains imitation learning with the full SupercomboFS vision/temporal-plan model. Isaac Lab owns collection and physical evaluation; PyTorch owns optimization. This does not replace the model with a small reinforcement-learning policy.

## Grounding

The previous evaluator uses standalone Isaac Sim World, while training labels come from the custom bicycle solver. The accepted 31,154,802-parameter checkpoint retained frozen vision during its latest steering adaptation. Cone detections are auxiliary diagnostics, not inputs to the pure-pursuit follower. The stressed track pairs show changed initial predictions and later departures; the light track first leaves the course before physical cone contact. Appearance sensitivity is established, but an individual distractor or perception-versus-tracking cause is not.

The original architecture makes model conversion, deterministic track/mission logic, and the data schema reusable across renderers. Its cached stage was a throughput choice for frozen vision. It cannot provide current-weight history during vision adaptation, and bicycle forecasts cannot certify native PhysX teacher behavior. The migration preserves model/controller and mission semantics while changing simulator ownership and label provenance.

## Candidates and selection

Candidate A is a live Lab environment with delayed teacher labels after learner handoffs. Candidate B is an offline episodic native demonstration pipeline, with immutable physical trajectories and independently trained image sequences. Both require real native teacher competence and current-weight temporal history. Two independent GPT-family design perspectives were used; no different-family reviewer was available.

Select B's immutable offline episodes and A's explicit simulation owner. A LabDrive holds a Lab SimulationContext and InteractiveScene, native PhysX car/cones, calibrated camera capture, action clock, mission synchronization and physical contacts. It never calls FSEnv.step or FSEnv.label; the old FSEnv object supplies only sampled scene, geometry, commands and referee metadata. The initial implementation uses one episode per process, avoiding unverified restoration of wheel/contact/solver state. Lab classes actually advance and update the simulation. A DirectRLEnv reward/autoreset layer is unnecessary for supervised collection and would complicate terminal evidence.

Native expert feedback acts on measured PhysX state. Retained labels are interpolated from the next ten seconds of actually executed uninterrupted expert control. A learner-prefix recovery episode can permanently hand over to the teacher; windows before that handoff are not expert targets. Contacted or incomplete windows are rejected. No speculative chassis-only physics branching or bicycle future labels are permitted.

Replay stores phase-zero 5 Hz packed camera frames, exact 20 Hz states/actions/teacher ownership/contact counts and saved runtime desire histories. Training re-encodes historical pairs with the current trunk, with intentionally truncated historical gradients. No persistent feature cache is reused while vision weights change. The complete checkpoint is saved and strictly reloaded; trainable and retained auxiliary parameters are reported separately. Native occlusion labels are not yet qualified, so image/BEV heads are preserved rather than trained against guessed visibility.

## Interfaces

`LabDrive(env, variant, assets, render)` owns `reset`, `observe`, `step`, `pose`, `summary`, and `close`. It exposes native views for evidence capture, not a second simulation loop. `NativeExpert(drive, params).act()` supplies privileged training controls. `Case` owns split identity, track/scene seed and appearance. NativeReplay owns temporal indexing and executed-future validity; lab_train owns optimizer, losses, selection receipts and complete checkpoint saves.

The old evaluator, protocols and receipts remain reproducible in their historical worktrees. New source hashes and candidate checkpoint identities belong to the Lab protocol. Reusing old source-bound manifests after this migration is forbidden.

## Gates and experiment policy

1. Actual Lab runtime and CUDA probe; native vehicle action/contact, camera capture and teacher qualification.
2. Baseline same-physics comparison, saved-input inference replay and native teacher versus model diagnostics.
3. Native episode schema/temporal/label tests and a measured full-vision training smoke run.
4. Freeze train/development/test membership before collection or checkpoint selection. Existing stress cases are exposed regression cases.
5. Train and select on development only. Evaluate the selected candidate and frozen baseline on the same fresh Lab cases; retain all failures and unchanged debug/research video layouts.

Targets are both formerly timed-out stressed track cases finishing within the existing 120-second limit, lower physical cone contacts without losing matched completions, and a corresponding improvement on fresh cases. They are targets, not promised results. No timeout increase, privileged runtime steering, relaxed contact counting or hidden failed trials can satisfy them.

The final case count, collection size and update budget will be frozen after measured native teacher throughput and storage. Spark started with 109 GiB free; prior eight stress runs occupy 17 GiB. Large debug tensors will require bounded case recording or verified local archival, with explicit coverage. Multi-hour work starts only after a measured capacity checkpoint. Preserve at least 25 GiB remote free space and all existing user artifacts.

## First measured gate

The existing verified arm64 image passed an actual Lab SimulationContext and InteractiveScene probe: 12 physics steps, CUDA available, Torch 2.9.0+cu130, NumPy 1.26.0, 14.34 seconds including startup. This does not yet establish vehicle, camera or teacher compatibility. Lab 2.3 / Sim 5.1 is pinned for comparison; Mimic is not part of this migration.
