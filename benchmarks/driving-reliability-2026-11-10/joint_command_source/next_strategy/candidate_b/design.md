# Candidate B. Train the last policy suffix and direct command head

The goal is a vision FSdriver that finishes the full RRLL skidpad and track missions with zero contacts within 120 seconds. This is a conditional architecture proposal. The recovery-data-only trial retains priority through its independent postflight. This proposal selects no weights.

At handoff, the parent reported that the data-only run completed 500 updates and all six validation passes, with no checkpoint passing the frozen gates. The reported step-500 MAEs are 0.00622419 rad and 0.19858789 meters per second squared. Its p95 errors are 0.02445922 rad and 0.82706383 meters per second squared. The final dataset postflight then failed with `OSError 24`, too many open files. There is no completed `train.json` or selection. These are parent-reported facts, not an independently verified candidate result. Preserve that failed receipt and finish the independent read-only postflight before starting another training experiment.

The proposed first architecture change trains the last plan-side policy suffix together with the existing direct command head. The image encoders, temporal transformer, earlier policy layers, final plan projection, desire branch, and cone heads remain frozen. There are 296,450 optimized parameters. The experiment changes which parameters may learn, while retaining the verified images, history, teacher-command targets, data membership, sampler, loss, and qualification gates.

## Investigation record

The architect phases are Ground, Sketch, Agree, Implement, and Scrap. Ground and Sketch are complete for this candidate. Agree belongs to the parent arena synthesis. Implement and Scrap are skipped because this assignment is design-only. The parent owns the competing measured-speed-controller candidate and synthesis. This runner did not create another agent or claim a cross-family review.

The investigation checklist is complete. It routed through how, recorded the throughput checkpoint, produced the architecture explanation and comparison, and applied unslop to this document.

throughput checkpoint: n/a, read-only investigation

The static inspection loaded an existing local checkpoint on CPU and reconstructed the pinned policy graph from the public pinned source. The original source SHA-256 and prepared policy SHA-256 matched the existing receipts. The converted policy loaded the checkpoint state strictly. No forward pass, backward pass, fitting, evaluation, GPU operation, or SSH command ran. The inspection wrote only this candidate directory. The graph bytes were processed in memory, not retained as a model artifact.

## Usage, before the interface

These are proposed call sites, not implemented commands. The package owns its contract, trainable scope, serialization, and runtime adapter. Callers do not monkey-patch the old actuator trial.

```python
	prepared = prepare_joint_trial(recipe_path, output_dir, device="cuda")
	receipt = check_joint_conditioning(prepared)
	result = fit_joint_trial(prepared, receipt)

	policy = JointActuatorPolicy.load(result.selected_package, device="cpu")
	policy.reset()
	steer_rad, accel_mps2, evidence = policy(obs, env)

	replay = verify_joint_episode(episode_dir, result.selected_package,
	                            output=receipt_path, device="cpu", frames="all")
```

`fit_joint_trial` cannot start without the exact source, data, scope, conditioning, and disposable wiring receipts. `selected_package` exists only when the unchanged per-group validation rules pass. A successful replay does not qualify a driving result. Native physical qualification remains a separate gate.

## How the existing path works

`actuator_trial.batch_input` loads qualified TRAIN or VAL observations. The target is `ep.controls[tick]`, where `tick = ep.ticks[frame]`, and it requires `ep.teacher[tick]`. The command target is the actual teacher command at that observation tick. It is neither a future plan-derived action nor a teacher command from another state.

The model recomputes eight valid past image features at phase-zero 5 Hz for each batch. The policy receives the 24-slot history plus current hidden feature, the full desire queue, and traffic convention `[1, 0]`. The pinned policy graph reads slots 16 through 24. Measured body `vx`, `vy`, and yaw rate `r` are the only proprioceptive values supplied to the command head.

`Capture` takes the input to `base.policy.p_node_linear_20`. It is the 256-value output of `p_node_relu_7`. The head applies affine-free LayerNorm, concatenates scaled proprioception, applies `Linear(259,128)`, SiLU, and `Linear(128,2)`, then multiplies by `[0.1, 2.0]`. Its outputs mean steering angle in radians and commanded longitudinal acceleration in meters per second squared. The command is bounded, passed through the unchanged mission brake and launch override, and bounded again.

The existing head training places both image encoding and `policy_raw` inside `torch.inference_mode()`. Cloning the captured activation outside that context enables command-head backward, but does not restore a gradient path into the policy. Merely changing `requires_grad` flags therefore cannot implement this candidate.

At runtime, `NativeActuatorPolicy` calls the existing `ModelPolicy` once and captures the same activation. Its old controller actions become diagnostics. The new command head supplies the actual request. The proposed adapter retains that one-call behavior, the temporal queue, mission handling, and evidence fields.

## Actual parameter scope and graph proof

[parameter_inventory.json](parameter_inventory.json) records every one of the 417 base state tensors with shape, dtype, hash, and parameter classification. It also records the actual converted module types, initializer names, exact payload keys, and FX suffix. [graph_inventory.json](graph_inventory.json) records the full static policy graph.

The inspected package is `runs/direct_command_trial3_archive/step_500.pt`, with SHA-256 `9112e138d4856fbbdc383d8f63dfbaeb92fb62449c4b66b5eb6f30b91ef22ba7`. It is evidence about the existing structure, not a selected starting head. The original standalone base is still the recipe-pinned `latest_pi_train.pt`. Main fitting starts its fresh seed-zero command head. The rejected step-500 head must not enter this treatment.

The prepared policy SHA-256 is `a27088e94f581de472f9610183e3979987cf59a1b272816a339bb9757d00a17a`. The verified suffix is:

```text
	p_node_relu_3, width 512
	  p_node_linear_14, 512 to 256
	  p_node_relu_4
	    p_node_linear_16, 256 to 256
	    p_node_relu_6
	    p_node_linear_17, 256 to 256
	  p_node_add_7, relu_4 plus linear_17
	  p_node_relu_7, captured activation
	  p_node_linear_20, 256 to 990
	  p_node_mul_1, fixed plan output scale
```

Only the first three affine modules in that suffix become trainable. The residual addition makes both the direct and residual paths from `linear_14` reach the captured activation. The command loss consumes that activation before `linear_20`, so the final plan projection cannot affect the command loss. Unfreezing only `linear_20` would optimize a disconnected component.

| Scope | Parameter keys | Parameters |
| --- | --- | ---: |
| `base.policy.p_node_linear_14` | `weight [256,512]`, `bias [256]` | 131,328 |
| `base.policy.p_node_linear_16` | `weight [256,256]`, `bias [256]` | 65,792 |
| `base.policy.p_node_linear_17` | `weight [256,256]`, `bias [256]` | 65,792 |
| Command head `linear1` and `linear2` | Four existing keys | 33,538 |
| Total trainable | Ten parameter tensors | 296,450 |
| Frozen base | 411 state tensors, including 388 parameter tensors | 30,891,890 |
| Entire base | 417 state tensors, including 394 parameter tensors | 31,154,802 |

The exact six base keys are the Cartesian product of `base.policy.p_node_linear_14`, `base.policy.p_node_linear_16`, and `base.policy.p_node_linear_17` with `.weight` and `.bias`. The implementation must store the six resulting literal names and shapes in its own scope contract. It must not use substring matching, `set_trunk_trainable("last")`, or all-policy optimization. The existing `set_trunk_trainable` method iterates vision parameters, so it does not express this scope.

This is the last plan-side suffix, not the shared temporal transformer. It tests whether the frozen 256-value activation is a limiting representation under the retained recipe. A failure does not establish that all image or history representations are inadequate. Those earlier components remain fixed.

## Data types and signatures

The data shape is one batch of same-tick qualified teacher commands and one immutable trial contract. The sketch uses opaque receipt types so a caller cannot replace qualification with an unchecked boolean.

```python
	@dataclass(frozen=True)
	class CommandBatch:
	    history: Tensor      # FP32 [B,24,512], eight valid past slots
	    hidden: Tensor       # FP32 [B,512], freshly encoded current frame
	    desire: Tensor       # FP32 full recorded queue
	    traffic: Tensor      # FP32 [B,2]
	    proprio: Tensor      # FP32 [B,3], vx, vy, r
	    target: Tensor       # FP32 [B,2], same-tick teacher steer, accel
	    anchors: tuple[QualifiedAnchor, ...]

	@dataclass(frozen=True)
	class JointRecipe:
	    kind: Literal["direct_native_actuator_joint_suffix_v1"]
	    source_recipe_sha256: Sha256
	    scope: ExactScope
	    data_contract: RecoveryTrialDataContract
	    optimizer_contract: FixedOptimizerContract
	    gates: FrozenGates
	    inference_device: Literal["cpu"]

	def prepare_joint_trial(recipe: Path, out: Path, device: str) -> PreparedTrial: ...
	def check_joint_conditioning(trial: PreparedTrial) -> ConditioningReceipt: ...
	def fit_joint_trial(trial: PreparedTrial, receipt: ConditioningReceipt) -> TrialResult: ...
	def verify_joint_episode(episode: Path, package: Path, *, output: Path,
	                         device: str, frames: Literal["all"]) -> ReplayReceipt: ...

	class JointActuatorPolicy:
	    @classmethod
	    def load(cls, package: Path, *, device: str = "cpu") -> JointActuatorPolicy: ...
	    def reset(self) -> None: ...
	    def __call__(self, obs: Observation, env: DrivingEnv) -> tuple[float, float, dict]: ...
```

There is one contract owner. It validates exact membership, source binding, scope, checkpoint semantics, and the runtime adapter. Its public interface hides the policy hook and tensor ownership. The old data verifier and sampler can remain read-only dependencies, but a new module must not rewrite `base.RECIPE`, `base.datasets`, or `base.preflight` as the current data-only trial does. That pattern would leave old frozen-base checks in control of the joint trial.

## Training forward and serialization

The entire model remains in eval mode even while the six permitted tensors require gradients. Frozen image encoding and fresh history run under `torch.no_grad()`, not a persistent cache. Their returned FP32 tensors are ordinary detached tensors that autograd may save for affine-weight backward. `policy_raw` runs with gradients enabled, and the capture preserves `grad_fn`. The head then consumes the live captured activation. Detach only outputs retained for diagnostics or logging.

All modules begin with `requires_grad_(False)`. Enable exactly the six scoped base tensors and four command-head tensors. Optimizer parameter identities must equal that set exactly once. Global clipping remains 1.0 across the optimized set. AdamW remains at the existing 0.001 rate and zero weight decay. No plan loss, cone loss, regularizer, loss-weight sweep, or learning-rate sweep enters this experiment. The command loss remains mean SmoothL1 on errors divided by `[0.02, 0.5]`.

The new package kind must be `direct_native_actuator_joint_suffix_v1`. Its top-level keys can remain `kind`, `base_payload`, `head_state`, `command_contract`, `recipe`, and `provenance`, but its contract and loader are new. The exact existing `base_payload` keys are `step`, `model`, `config`, `controller_contract`, and `controller_intervention`. There is no `model_state_dict` or `state_dict` payload key. Its observed config is `{"controller_mode": "op_tangent_pi_v1"}`.

Every saved candidate builds a fresh CPU snapshot of the current base `model.state_dict()` and places it in `base_payload["model"]`. Reusing the original `payload` object, as the frozen-base trial does, would silently discard every learned suffix update. The new payload records the joint step and preserves the original controller metadata as inherited diagnostic-controller metadata. The outer direct-command contract owns actuation semantics. Provenance records the source payload metadata and source digest separately, so an inherited `controller_intervention` is never presented as proof of the new training run.

The package records all 417 updated base tensor hashes, all four head tensor hashes, the exact six-key training scope, the original base digest, the saved updated base digest, and the source archive digest. Only the complement of those six base keys is described as unchanged during training. At inference, all 417 base tensors and the head must remain unchanged relative to the saved package. These are different checks.

Frozen final-plan weights do not preserve old plan outputs. The changed activation flows into that frozen projection, so plan and old-controller diagnostics can change. Cone-head outputs remain tied to the unchanged vision path. The plan display must not imply that its old fitting guarantee survived the suffix update.

## Recipe and preflight

Copy the frozen recovery trial's verified data contract into a new recipe with an explicit architecture delta. Do not make the old trial accept the new kind. Preserve the original 7,006 TRAIN anchors and 2,067 VAL anchors. Add only the same two independently verified recovery episodes, with their 219 and 205 eligible anchors, for 7,430 TRAIN anchors. Derive and verify those counts again from the receipts. Retain 16 old recovery, 16 new recovery, and 96 ordinary draws per batch, with both recovery pools excluded from ordinary draws.

The preflight must check the following facts before any optimizer operation.

1. All source, ONNX, checkpoint, manifest, membership, recipe, recovery-receipt, and archived-source hashes match. Re-run the independent original-data verifier and both new recovery verifiers. Compare complete results, not only `passed`.
2. The graph has the inspected suffix and residual addition. All six modules and shapes match the inventory. The ONNX names identify the plan in-layer and plan residual layers. The pinned context proof still holds for the actual loaded gather buffer.
3. Hooked and plain policy outputs are exactly equal in a repeated ABAB check on fixed TRAIN inputs. Captured activations also match. The new grad-enabled forward at initialization matches the old inference forward before any update.
4. Model eval flags, FP32 dtypes, normal memory layout, disabled TF32, disabled AMP, fresh history context eight, and no feature cache match the recipe. Current and historical image encoders have no trainable parameters.
5. The parameter identity set is exactly ten tensors and 296,450 values. The 411 forbidden base tensor hashes match the original checkpoint. No batch normalization buffer, policy scale, or auxiliary state may change.
6. A strict save-load round trip reproduces the current updated base and head digests. It must also reproduce fixed-input commands after the disposable conditioning step, otherwise the package is carrying stale base weights.
7. Source archives include the new package owner, recipe, trainer, evaluator, replay verifier, physical auditor, and renderer wrapper as well as their actual dependencies. The budget and exclusive-output rules remain unchanged. Never overwrite a failed receipt.

One dataset owner keeps one pool of `NativeEpisode` objects and constructs old and expanded index views over those objects. It must not reload the 32 original episodes into duplicate live mmap pools when checking membership or composing the expanded dataset. The source-and-data postflight runs in a fresh process after the training process closes its dataset pool. Bind that receipt to the completed update log and saved package hashes. The observed file-descriptor failure is an infrastructure failure, not grounds to loosen any numerical gate, resume optimization, or invent a selection.

`parameter_inventory.json` makes the scope reviewable now. The future preflight regenerates the inventory from `named_parameters()`, `state_dict()`, `policy_names`, and `policy.graph`; it must not trust this document as an executable whitelist.

## First experiment and stop rules

Proceed only if the current data-only trial fails to qualify and its sources, data, and checkpoint reports have been sealed. A successful current trial makes this architecture intervention unnecessary.

The first operation is one disposable TRAIN-only conditioning update, using the inherited 0.001 rate and loss. The justification is mechanical. Moving the command loss into pretrained layers changes its gradient scale and the clipping set, so the new graph needs a direct update check before a 500-update run. This is not a rate search.

Use the already audited fixed 32 TRAIN wiring anchors, including the two clean recovery cases. Report the loss, both command errors, each module's raw gradient norm, global clipping factor, and per-module update norm. Require finite nonzero aggregate gradients in each of the three suffix modules and the command head, finite commands, unchanged forbidden tensors, changed permitted state, and an exact successful package round trip. Require the fixed batch loss after the update to improve beyond measured repeat noise. If the update fails, stop and diagnose. Do not try another rate or weight inside this trial. Discard both updated base and head after the check.

Next, run the unchanged disposable wiring gate from the pristine base and fresh head. Both output MAEs must reach 0.002 rad and 0.05 meters per second squared within 200 updates. Discard the entire wiring model and optimizer. Start main training again from the pristine base, fresh seed-zero head, and new optimizer.

The smallest experiment that tests the representation premise is one matched 500-update joint-suffix run against the now-sealed data-only run. Use identical data, seed, anchor draws, head initialization, output semantics, loss, and checkpoints at 0, 100, 200, 300, 400, and 500 updates. The trainable scope is the treatment. The one-update check only proves connectivity and conditioning; it cannot settle the representation hypothesis.

Retain every per-episode and per-stratum VAL gate. Both command MAEs must be at most `[0.01, 0.25]` and both p95 errors at most `[0.03, 0.75]`. Select only among passing nonzero checkpoints using the existing aggregate-MAE selection score and earlier-step tie break. If none passes, stop without a visual gate, physical pilot, or final run.

For a passing selected checkpoint, retain the fixed 32 dynamic TRAIN anchors, both-camera black-image ablation for every valid historical and current frame, minimum 10 percent relative loss increase, and increase greater than five times repeat noise. Restore the exact selected package before every comparison. A visual gate demonstrates image dependence, not correct scene understanding.

Only an eligible, visually dependent package proceeds to the native pilot and existing full exposed development cases. Preserve all contact, off-course, RRLL-order, referee, camera, physical, and 120-second predicates. Zero contacts and successful full mission completion remain requirements. Existing final membership and exposure rules remain authoritative. This design neither opens final scenes nor grants qualification from a lower offline error.

## Replay and CPU runtime choice

Use CPU FP32 for candidate deployment inference from its first new pilot onward, with four Torch threads and explicit device selection. Training remains on Spark. This choice avoids treating an old CUDA-produced episode as a CPU parity target after the earlier cross-device replay issue. It makes the producer and independent replay use the same numeric backend. It does not authorize weaker tolerances or a substitute replay of a different episode.

Before a physical pilot, replay fixed TRAIN inputs through the candidate CPU runtime and audit the exact updated-base and head digests. Compare the training-device output as a diagnostic and record CPU latency in a separate timing preflight. No CPU speed claim is made here. The current 20 Hz contract describes simulated control ticks, so this design does not add an unsupported 50 ms wall-clock qualification gate. Simulation time and wall-clock throughput remain separate reported quantities. Do not switch inference devices during an episode.

The new verifier retains all-frame replay of the exact saved images, temporal indices, full desire queue, measured state converted to FP32, captured latent, raw request, bounds, mission override, and applied action. It compares plain and instrumented calls, validates same-tick state and all four 20 Hz phases, and independently reconstructs the head output from the saved latent and proprioception. Keep `rtol=atol=1e-5` and the existing physical auditor tolerances.

The verifier must understand the new kind and hash the updated base. It must not import the old `KIND` or call the old training `frozen()` check. The physical auditor and rendering wrapper bind the new producer and verifier sources explicitly. Reuse their physical predicates unchanged. A new schema must reject old packages as a different contract instead of relabeling old evidence.

## Alternatives and tradeoffs

| Candidate | What it tests | Decision |
| --- | --- | --- |
| Current head-only recovery-data trial | Whether the new clean on-policy examples solve the error with the frozen activation | Finish it first. It is the smallest valid intervention and the matched control. |
| Head-only capacity or training changes | Whether the downstream mapping can fit better | A larger head or another rate would continue assuming that the frozen 256-value activation is adequate. Do not combine it with this scope test. |
| Train only the old final plan projection | Whether old plan outputs improve | Reject for this direct-command objective. It is downstream of the captured activation and receives no command-loss gradient. |
| Measured-speed controller around learned visual steering | Whether explicit longitudinal feedback removes the acceleration failure | A valid competing architecture with a small interface, but it does not directly repair the observed steering error. It requires its own speed-target semantics and qualification. |
| Six-key suffix plus direct command head | Whether task-specific policy compression is limiting clean command learning | Preferred conditional experiment. It retains the small command API and gives the loss access to the last 512-to-256 compression. |
| Full policy or image fine-tuning | Whether earlier features are inadequate | Defer. It changes far more state and cannot localize the first discriminating result. |

The proposal accepts 262,912 additional trainable parameters and backward computation in exchange for testing the frozen-activation premise. It accepts changed plan diagnostics because direct commands own actuation. It keeps early policy and vision features fixed, which limits what a negative result can establish.

The global gradient clip now includes suffix gradients. Record its effect because scope and effective head-step scale are coupled. A successful joint run would support this exact training intervention. It would not prove that information was mathematically absent from the original activation.

The limited recovery coverage and inherited teacher-command distribution remain risks. Passing offline aggregates can still hide a feedback instability. Frozen per-episode and per-stratum gates, image dependence, native replay, and full physical missions remain necessary.

Attack the Premise changed this candidate from another downstream head adjustment to an explicit test of the captured representation. Model the Domain changed loose frozen flags into an exact parameter-scope and package contract. Sequence Work into Verifiable Units keeps conditioning, disposable wiring, training, offline selection, visual dependence, and physical qualification ordered. Prove It Works requires actual graph ancestry, strict updated-base serialization, and replay of the saved artifact instead of a nominal module name or an old base digest.

The next implementation step, if the parent selects this design after the current trial, is to build the isolated scope-and-package owner and its CPU static preflight. No implementation is included here.
