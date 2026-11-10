# Candidate A: Direct executed-command head

## Usage first

The following is a proposed interface, not an existing command or implementation. Run one isolated FP32 experiment on DGX Spark after the CPU and data gates pass. It retains the full frozen model and learns a separate actuator-command head. The native evaluator continues to own mission overrides and the vehicle actuator.

```python
# Training, inside one experimental file.
dataset = CommandReplay.open(v7_manifest, split="train", recovery_recipe=recipe4)
experiment = CommandExperiment.from_checkpoint(c121_pi_checkpoint, seed=20261110)
experiment.fit(dataset, validation=CommandReplay.open(v7_manifest, split="val"))

# Simulator: the caller gives only measured observations and receives a command.
policy = experiment.policy()
policy.reset()
command, audit = policy.step(observation)  # steering rad; requested acceleration m/s²
steer, acceleration = command.steer, mission_brake(command.accel, obs, env)

# CPU audit: exact saved input and temporal state reproduce the same pre-override command.
assert replay_saved_frame(policy, saved_frame).matches(atol=1e-5, rtol=1e-5)
```

## Problem and grounding

The goal is reliable camera-based cone-track driving and a full zero-contact RRLL skidpad. The two latest 500-update native-plan retrains both failed their fixed checkpoint gates and failed the exposed skidpad. The full perfect-future OP+PI standing run finished with six contacts. Three privileged transition suffix corrections cleared their zero-contact suffix gate; these do not establish an executable standing-start learner. Recorded controls exactly reproduce the teacher rollout in the referenced native probe. These facts support testing the executed action contract itself, but do not prove that imitation will drive safely from learner-induced states.

`NativeEpisode.sample` constructs each physical plan in the current ego frame and exposes `controls[tick, 0:2]` separately. Since future position starts at zero, the beneficial privileged current-position correction cannot be learned as the plan's zeroth position target. `ModelPolicy` builds camera pairs and a temporal queue, runs frozen vision and the original policy, turns a plan into a controller request, and finally applies the mission layer. `lab_train.py` differentiates plan losses and controller-derived quantities, so the current learned target and executed control are separate contracts. A new head directly predicts the command at the actual actuator boundary. Raw controls supervise only that new output, never the physical-plan target.

The route counterfactual probe passes 601 saved-frame parity checks and shows strong route sensitivity. It is a fixed-input study, not a driving rollout. Preserve route inputs and explicitly test route counterfactuals without claiming that route pulse changes cause a safe maneuver.

## Shape and ownership

One experimental Python file under this candidate's lab directory owns replay adaptation, the command head, experiment, and evaluation adapter. Existing production modules stay unchanged. Reuse `NativeEpisode`, `fresh_history`, `TemporalState`, the native evaluator and the existing qualified v7 manifest without copying their domain rules. Before training, add a lab-local evaluator adapter and saved-frame auditor that accepts command output explicitly. The existing evaluator assumes `info['plan']` and its auditor assumes a plan controller, so adapting both is necessary; never populate a fabricated plan or route the output through OP+PI.

```python
@dataclass(frozen=True)
class CommandObservation:
    road: Tensor          # [B, 12, 128, 256], uint8 packed old/current camera frames
    wide: Tensor          # same shape and packing
    history: Tensor       # [B, 8, 512], frozen encoded features at -1.6..-0.2 s
    desire: Tensor        # [B, 25, 8], existing pulse queue, route included
    proprio: Tensor       # [B, 3], measured vx m/s, vy m/s, yaw rate rad/s

@dataclass(frozen=True)
class Command:
    steer: Tensor         # [B], desired steering angle rad
    accel: Tensor         # [B], requested wheel-force-equivalent acceleration m/s²

class CommandHead(nn.Module):
    # 4811 -> 256 -> 128 -> 2. ReLU after first two layers.
    # 1,265,026 trainable parameters including biases.
    def forward(self, obs: CommandObservation, current_hidden: Tensor) -> Command:
        # Normalize declared proprio using fixed divisors (10 m/s, 5 m/s, 2 rad/s).
        # Flatten the eight history features, current [B,512], and route [B,25,8].
        # Output normalized steering and acceleration with fixed scales 0.5 rad and 4 m/s².
        raise NotImplementedError

class CommandReplay:
    @classmethod
    def open(cls, manifest: Path, split: Literal['train', 'val'],
             recovery_recipe: Path | None = None) -> 'CommandReplay': ...
    def sample(self, index: int) -> tuple[CommandObservation, Command]: ...

class CommandExperiment:
    @classmethod
    def from_checkpoint(cls, checkpoint: Path, seed: int) -> 'CommandExperiment': ...
    def fit(self, train: CommandReplay, validation: CommandReplay) -> None: ...
    def policy(self) -> 'CommandPolicy': ...

class CommandPolicy:
    def reset(self) -> None: ...
    def step(self, observation: MeasuredObservation) -> tuple[Command, CommandAudit]: ...
```

The head input is 9×512 visual features + 25×8 route pulses + 3 measured proprioception = 4,811 scalars. All eight past features and the current feature are produced by the exact frozen `base.encode`, in FP32, with fresh batch-local recomputation. The existing approximately 31M model and all its tensors remain loaded, unchanged, and available for the unchanged image/cone audit, including its policy and auxiliary heads. The head does not consume the predicted plan, privileged pose, cone positions, lap state, oracle corrections, or previous teacher action. It uses only the current measured state and existing route interface. A fixed 20 Hz model call returns requested angle and acceleration. Vehicle limits and the existing mission override retain their current ordering, and both requested and final commands are recorded. Confirm the dataset's `controls` are at exactly this pre-mission/pre-actuator boundary; abort if provenance instead reflects a different boundary.

`TemporalState` represents frames at 20 Hz with a four-tick image-pair gap and features sampled every fourth tick. Native replay saves phase-0 images at 5 Hz, with corresponding 20 Hz `controls[tick]`. At aligned ticks the replay can reproduce its history from those samples. At the other three phases the deployment model sees fresh images absent from the dataset. Before optimization, replay at least one full clean teacher episode through both constructions and compare all nine hidden features, images, desires, timestamps, and state at every available phase-0 tick at 1e-5 tolerance. Prove labels come from the control applied after the sampled state. A 120-frame live audit must also prove correct timing and all four live phases. This design accepts unsupervised intermediate image phases as a measurable distribution-shift risk; it must not pretend 5 Hz recordings supply them. If alignment fails, stop and repair the capture/contract before training. If the learned candidate fails the rollout, consider 20 Hz recollection as a separately frozen next experiment.

## Fixed experiment and gates

Freeze this protocol, source hashes, checkpoint, manifests, sampler, normalization, seeds, and all decision limits before accessing validation results or starting training. Verify disjoint train and validation episode/track identities under the existing rules, verify recovery provenance and qualified suffix labels, and assert the expected 7,006 train and 2,067 validation anchors. No final test scene or candidate-suite result is read for training, selection, or tuning.

The single run uses FP32, contiguous NCHW, frozen full base, no BF16, channels-last, or feature cache. Use AdamW on the new head only, learning rate 3e-4, weight decay 1e-4, batch 128, 500 updates, seed 20261110. Use the established recovery sampler with 16 draws per batch and 112 ordinary qualified draws, fixing its seed and all source membership. Checkpoint at 0, 100, 200, 300, 400 and 500. Each target is `controls[tick]` from a teacher-qualified, contact-free anchor. Minimize the equal average of two SmoothL1 losses with beta 1 in normalized coordinates, steer/0.5 and acceleration/4. No plan loss, action derived from plan, consistency penalty, or additional control residual is present. These are experiment settings, not claims of optimality.

CPU preflight must check raw control sign, radian and acceleration units, sample/action timestamp ordering, bounds, finite tensors, measured-state-only access, all-pad/startup behavior, reset, both route polarities and round-trip checkpoint identity. Assert all frozen model tensor hashes before/after and compute an explicit new-head parameter inventory. The trained head must beat its own step-0 validation normalized command MAE in both components by at least 25%, with steer MAE ≤0.08 rad and acceleration MAE ≤0.8 m/s². In each predeclared validation slice with at least 30 anchors, steering MAE ≤0.12 rad and acceleration MAE ≤1.2 m/s²; report smaller slices without using them to select. Slices are right laps, left laps/transition, track, recovery, and straight/launch as identified by pre-existing episode annotations. If annotations cannot support these disjoint definitions without using privileged labels as inputs, fail the preflight and define labels from dataset metadata before training. Require zero nonfinite outputs, gradient norms and weights, and no frozen state changes.

Among checkpoints satisfying every gate, select the lowest aggregate validation normalized MAE summed equally across outputs; ties go to the earlier update. This is one trial, with no alternate coefficient, seed, LR, iteration or gate relaxation based on observations. Action MAE is only an offline eligibility check. If no checkpoint qualifies, do not promote, do not access the sealed suite, and report failure. The predeclared step-500 model may run only as an explicitly rejected diagnostic on the already exposed development pair, with full artifacts.

Before a candidate can reach frozen acceptance, run the same artifact-backed paired exposed heavy-skidpad and track cases against frozen c121+PI, once each, with complete research and debug video for both arms, hashes, complete decode and independent saved-frame action replay. Both candidate episodes must finish, skidpad must be valid RRLL with zero physical contacts and zero off-course events, and track must have zero physical contacts and zero off-course events. Both arms must receive exactly the same frozen cases and timing rules; report realized RGB differences instead of assuming bitwise deterministic rendering. Count a failure to finish, missing frames, altered scene, bad hashes, unexplained controller override, or an audit failure as a failed gate. A pilot must preserve and check all 120 frames and both videos before any full drive. A zero-contact diagnostic is evidence on those cases, not general reliability.

Only after all gates pass, freeze the selected checkpoint, code, suite hash and acceptance protocol. Evaluate the existing frozen `acceptance_suite.json` as 64 candidate runs, four repeats for each of 16 conditions. Do not inspect its scenes in advance. Require all 64 valid completions with zero physical contacts and zero off-course events, including full RRLL wherever the suite specifies skidpad, plus passing artifacts and replay on every run. Report 64/64 as observed only. With independent trials, its one-sided 95% lower success bound is about 95.4%; shared conditions and correlated repeats make that independence assumption uncertain. Do not use suite outcomes to choose or adjust a checkpoint.

## Storage, evidence, and limits

Probe capacity before each phase. The specified budget is approximately 38 GiB tmpfs, 9 GB free disk on Spark, and 147 GB local storage; measure actual free bytes and reserve at least 2 GB on Spark at all times. Stage only the verified required dataset and model to the location that fits, avoid copying whole episode archives, stream per-run evidence and both complete videos to verified local storage, hash then acknowledge transfer before removing any scratch copy, and stop before reserve exhaustion. Do not delete source datasets, original receipts, failed evidence, or incomplete captures. Check that a single episode's worst-case evidence fits before launch. Record per-run completion, all inputs, actions before/after override, measurements, model/hash, route queue and temporal inputs, simulator timing, and artifact audits.

The main risks are frozen embeddings losing cone geometry or route information, too few transition/recovery examples, teacher-to-learner state distribution shift, the 5 Hz training/20 Hz inference phase gap, hidden evaluator plan assumptions, and command MAE masking rare safety-critical errors. A successful action fit does not imply safe driving. If preflight rejects the labels, the experiment is blocked, not silently relabeled. If it passes offline and fails driving, preserve the failure and investigate on-policy observations before proposing another intervention.

## Alternatives and red-flag screen

**Direct command head, selected.** The public `step` hides preprocessing, history, normalization and command semantics. It has one owner for action generation and one owner for mission overrides. It removes the unproven requirement that a realistic plan and a follower happen to yield the target action. Its extra 1.27M parameters preserve the full base model but cannot guarantee driving. Separate audit inputs and measured proprioception keep privileged data outside inference.

**Residual over OP+PI.** A deep policy might output a correction to the base requested action. It exposes a stateful controller plus residual ordering, saturation and reset semantics, and raw teacher targets become ambiguous unless a baseline action is reconstructed with the exact controller history. It also inherits plan-execution failures and correlates the residual with previous errors. Reject as the first test. It could follow only if the direct head fails in an interpretable way and a new fixed protocol justifies it.

**Cone geometry plus planner and controller.** A single `step` can hide the planner and yield an interpretable route. It introduces detection/association, sparse occlusion, topology, route-specific path selection and speed-planning contracts; the perfect-future OP+PI six-contact standing result already shows a capable reference is not sufficient. Existing BEV/image perception could be useful, but its cone and topology reliability must be proven. Reject for the first targeted intervention, not as a claim it cannot eventually work.

**Red flags.** One lab file avoids temporal module decomposition, pass-through wrappers, shared mutable writers, duplicate declarations and importable production internals. `CommandObservation` makes the measured input boundary explicit; `Command` prevents confusing commands with motion plans. Existing frozen replay/evaluator own dataset integrity and physics. The single adapter must not silently support both plan and command control modes for one run. Record head parameter and contract metadata once in the checkpoint and derive audit checks from it. Independent candidate A is one arena input; the coordinator owns cross-candidate synthesis.

## First implementation step

Implement the CPU-only command boundary and phase-alignment test against real qualified native replay, then the artifact adapter. Do not start training if either contract is unproven. Run the live 120-frame pilot after a checkpoint passes offline eligibility.
