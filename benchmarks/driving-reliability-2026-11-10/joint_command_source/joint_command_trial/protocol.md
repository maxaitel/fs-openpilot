# Joint policy suffix trial

This trial tests whether the final policy compression limits vision driving. The target remains full RRLL skidpad and track completion with zero physical contacts within 120 simulated seconds. No artifact from implementation or CPU fixture tests qualifies a driving result.

## Architecture and ownership

`joint_core.py` owns the exact six trainable base tensors, package validation, command head, and native runtime. The three affine modules are policy linear14, linear16, and linear17. Their weights and biases contain 262912 values. The unchanged four-tensor head contains 33538 values. The optimizer owns exactly ten tensors and 296450 values. The full base still contains 417 states and 31154802 parameters. Its 411-state complement remains equal to the original checkpoint during training. All 417 updated base states and all four head states remain equal to the saved package during deployment.

`base_model_digest` identifies the updated saved base. `original_base_model_digest` identifies the original base. Every package includes every updated base hash, every head hash, the original state hashes, the exact scope, and the inherited payload metadata. The writer creates a CPU clone of the current `model.state_dict()` in `base_payload.model`. The original `config`, `controller_contract`, and `controller_intervention` remain diagnostic provenance. The new outer command contract owns actuation. Updated activations can change the original plan diagnostics despite a frozen final plan projection.

The runtime calls the original model once, captures its 256-value activation, and applies the command head with measured `vx`, `vy`, and yaw rate. It bounds that request, applies the mission override once to the new request, and bounds acceleration again. The original model controller action remains a diagnostic. No label or teacher target enters deployment.

`joint_data.py` owns one original NativeEpisode pool. The two new episodes extend that pool. The original TRAIN and VAL datasets use references to those same episode objects. Explicit closure releases every memory map on success and failure. Independent original and recovery verifiers and the entire original RecoverySampler construction run in child processes before the trainer opens its pool. The returned legacy groups bind to the reopened index, episode metadata, receipts, recipe, and manifest. The child report contains no process or time fields. No module imports `trial_v2.py` or changes an old module's globals.

## Frozen inputs and optimization

The recipe preserves the exact recovery-data trial sources, recipes, membership, receipts, and data. TRAIN has 7430 anchors, composed of the original 7006 and the new 219 skidpad and 205 track anchors. VAL has the unchanged 2067 anchors. Each batch draws 16 old recovery anchors, 16 new recovery anchors, and 96 ordinary anchors. Both recovery pools are excluded from the ordinary pool. Recovery draws remain uniform by case, then uniform by qualified first-two-second anchor, with replacement.

Targets are the actual qualified teacher commands at each observation tick. Every batch recomputes both-camera history and the current feature. The policy reads eight past 5 Hz features and the current feature through its verified gather. It also receives the full desire queue and traffic convention. Frozen encoding runs under `no_grad`. The policy forward runs with gradients enabled and retains the captured activation's gradient path. All modules remain in eval mode. There is no feature cache, AMP, TF32, or channels-last layout.

The loss remains mean SmoothL1 on command errors divided by `[0.02, 0.5]`. AdamW uses learning rate 0.001, zero weight decay, and global clipping at 1.0 over all ten tensors. There is no learning-rate sweep or auxiliary loss.

## Stage receipts

Every stage creates an exclusive attempt marker before starting and preserves any failure receipt. A failed or interrupted attempt cannot resume in its output directory. Every stage rejects any prior failure marker. Training writes six full packages at steps 0, 100, 200, 300, 400, and 500. The minimum free reserve is 25 GiB. The run artifact cap is 1 GiB.

The audit binds all joint training and pipeline sources, tests, documentation, the old pure dependencies, recovery verifiers, recipes, membership, original prepared graphs, and design inventories. The source archive records an explicit source-to-member map, including absolute external paths. Graph bytes are bound by hash and excluded from the source archive. The audit checks graph ancestry, exact scope, original state, fresh history, ABAB plain and hooked outputs, and equality with the original inference forward.

Conditioning starts from the original base and a fresh seed-zero head. It performs exactly one update on the fixed 32 TRAIN wiring anchors. Those anchors contain 16 old ordinary, eight new skidpad recovery, and eight new track recovery anchors. Every optimized tensor must have a finite nonzero gradient and update. The receipt records per-tensor gradient and update norms and the global clip scale. The fixed batch loss must improve by more than five times repeat noise. All 411 forbidden states must remain exact. A saved updated package must reload and reproduce every command exactly on that batch. All conditioning weights are discarded. Failure stops this recipe.

The disposable wiring stage restarts from the original base and a fresh seed-zero head. It uses the same 32 anchors. Both command MAEs must reach 0.002 radians and 0.05 meters per second squared within 200 updates. All weights from this stage are discarded.

The main stage restarts from the original base and a fresh seed-zero head. It runs exactly 500 updates. It saves each validation and package independently. After update 500, it closes its only episode pool and writes `training_updates.json` with postflight incomplete. It starts a fresh process for `postflight`. That process reruns source and complete dataset verification, checks all six saved packages and the 411-state complement, and recomputes the final 2067-anchor validation exactly. Postflight requires the live training process as its parent at entry and again immediately before writing each completion receipt. Only a successful postflight writes `train.json`. A descriptor, resource, or source failure leaves the run incomplete and cannot create a selection.

Validation reports the overall set, every episode, low-speed anchors, recent desire pulses, and recovery strata. Every nonempty group's steering and acceleration MAEs must be at most `[0.01, 0.25]`. Every group's p95 values must be at most `[0.03, 0.75]`. Step zero is ineligible. Selection minimizes `max(overall_steering_MAE / 0.01, overall_acceleration_MAE / 0.25)` and breaks ties by earlier step. If no package passes, fixed step 500 remains diagnostic only.

The visual gate uses the same seed-zero selection of 32 TRAIN anchors with frame at least eight and measured forward speed at least two meters per second. Each comparison reloads the exact selected updated package. It compares ordinary inputs, repeated ordinary inputs, and black RGB in both cameras for the current frame and every valid historical frame. Blanking must increase loss by at least 10 percent and by more than five times repeat noise.

The runtime preflight uses the exact saved RGB prefixes of both verified new TRAIN episodes. It starts at frame zero, keeps full sequential history, uses up to 120 available frames per prefix, and requires all four 20 Hz temporal phases. It records package loading, warmup, ordinary full-model inference, and inference with plan and cone capture separately. The timing excludes Isaac physics, rendering, observation reconstruction, and evidence writes. It checks repeated CPU commands and independent head reconstruction from captured latent values at the original 1e-5 tolerances. The producer and replay stay on CPU FP32 with four Torch threads. A separate policy instance also records training-device output differences as diagnostics. Those differences cannot relax CPU replay tolerances. This stage makes no 20 Hz wall-clock claim.

Native pilot, replay, physical audit, rendering, full exposed development cases, final membership, and exposure controls retain their existing order and gates. Full saved-input replay uses absolute and relative tolerances of 1e-5. Successful full missions, zero physical contacts, no off-course event, correct RRLL order, referee completion, and the 120-second bound remain required. No final scene is opened by these scripts.

## CLI

Commands run from the repository root with the installed training environment. The parent owns launch authorization and GPU serialization.

```sh
PYTHONPATH=src:. python .work/lab/joint_command_trial/train_joint.py audit --out runs/joint_command_trial1 --device cuda
PYTHONPATH=src:. python .work/lab/joint_command_trial/train_joint.py condition --out runs/joint_command_trial1 --device cuda
PYTHONPATH=src:. python .work/lab/joint_command_trial/train_joint.py overfit --out runs/joint_command_trial1 --device cuda
PYTHONPATH=src:. python .work/lab/joint_command_trial/train_joint.py train --out runs/joint_command_trial1 --device cuda
PYTHONPATH=src:. python .work/lab/joint_command_trial/train_joint.py visual --out runs/joint_command_trial1 --device cuda
PYTHONPATH=src:. python .work/lab/joint_command_trial/train_joint.py runtime --out runs/joint_command_trial1 --device cuda
```

The train command launches postflight itself. The device argument controls training and verification of the original CUDA-produced recovery prefixes. Runtime deployment remains CPU. An explicitly authorized diagnostic runtime check can add `--checkpoint runs/joint_command_trial1/step_500.pt` after a completed postflight. That option does not qualify the package.

## Implementation evidence and limits

The local CPU tests exercise exact trainable scope, the live suffix gradient path, frozen-state tamper rejection, updated-payload command roundtrip, same-tick teacher enforcement, sampler exclusion, and idempotent memory-map closure. One test uses the actual local 417-state checkpoint to reject forbidden and unbound tensor changes. The compact forward fixture does not prove the converted full graph or real data pipeline.

The implementation checklist follows the selected candidate-B architecture. Ground and Sketch use the two-candidate synthesis in `next_strategy`. Agree is the parent's selected-B decision. Implement covers this isolated package and stages. Scrap is not triggered. The throughput checkpoint is stage-level wall time and per-update timing in receipts. No performance result is claimed before the real run.

Model the Domain makes the scope a literal six-key map and data ownership one context-managed pool. Boundary Discipline puts package and recipe checks at loading boundaries. Prove It Works makes updated-byte roundtrip and independent postflight required artifacts. No GPU experiment, training launch, or native physical validation ran during local implementation.
