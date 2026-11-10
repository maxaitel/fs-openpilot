# Cone-set core and training review

The goal is a zero-contact vision driver with a reproducible, bounded representation experiment. The implementation is not ready for source freeze until the two stage lifecycle findings below are resolved. This was a read-only source review except for this report. The pipeline had a separate reviewer.

## Findings

1. **P2. A stage can appear complete before cleanup and final verification.** `train_set.py:106-112` treats the existence of `<stage>.json` as proof that an attempt completed. Audit writes `audit.json` at line 517 inside the pool context opened at line 463. Conditioning and overfit similarly write their receipts at lines 576 and 604 inside the contexts opened at lines 545 and 592. The pool closes only on context exit, and `main` calls its final `checked_audit` at line 882 after the stage returns. A kill after a receipt write leaves no failure marker, but the next stage is accepted even though cleanup and final verification never completed. A second process can also start the next stage while the first owns its pool. A temporary-directory reproduction called `begin_stage` for audit, wrote only `audit.json`, then successfully called `begin_stage` for condition. Require a separate exclusive completion marker only after the owning context closes and final verification succeeds. Check that marker for each prerequisite and prohibit concurrent stages.

2. **P2. Postflight can accept a different conditioning receipt from the one used by wiring.** `train_set.py:618-621` checks the overfit receipt's `condition_sha256` at main-fit start. The training record takes fresh hashes after updates at lines 654-655. Postflight calls the two generic gates at lines 678-679 and compares the training record to those later hashes at lines 680-682. It does not recheck that overfit's condition hash matches the current condition receipt. If the condition receipt changes after main fit starts, the stored training hashes and postflight checks can agree with the new file while overfit binds the old file. In a temporary fixture both generic postflight gate calls succeeded with exactly that mismatch. Carry forward the prerequisite hashes captured before the fit, verify the full chain at completion and in postflight, and bind later selected-package gates to the exact prerequisite receipts they consumed.

## Verified and limits

CPU fixture tests passed with `.venv/bin/python`: 8 core tests and 10 training tests. The real archived 417-state fixture was present and its tamper checks passed. They cover frozen model and integer-buffer identity, head inventory, CPU stock-decoder comparison on identical raw tensors, same-tick teacher flag, input recomputation, all eight gradient paths, collision bounds, archive mapping, and the existing attempt guards. They do not establish behavior on the Spark dataset or actual converted graphs. No GPU, simulator, SSH, fitting, or final suite was run. The initial `python` command failed because that executable is not on PATH; all reported tests used the workspace virtual environment.

The independently checked source uses actual frame pairs for each draw, moves detached image-head tensors to the stock CPU decoder, retains the base and original metadata digests, compares training device and CPU as diagnostics, validates the single selected package over all 2067 VAL anchors on singleton CPU calls, and enforces the visual and runtime gate ordering. The saved-step-zero head binding appeared during this review at `train_set.py:706-707`. Recheck all sources and tests after the author finishes concurrent edits.

## Source hashes observed

SHA-256 at the final review snapshot. `cone_core.py` and `train_set.py` changed while the review was in progress. The first hashes read were `a7279bbc063307b5d47f24245ad449bfafce7400d91b05deb8e3bf709519427b` and `fe4158294f223245f5bb131d6d8c144959407f46b122b7d3257e60060fdec9d0`, respectively.

- `.work/lab/cone_set_actuator_trial/cone_core.py` `7e73e45ba42443fc1fb028d886677e2345393ea4b8b8c5f3f79bd8ba7b1251fb`
- `.work/lab/cone_set_actuator_trial/train_set.py` `5a799aac3429c57b6674cb756e4701aa779519fa4d5d0218431e202b5fbbbbdc`
- `.work/lab/cone_set_actuator_trial/recipe.json` `9858b5989ef9298db04334a90a8d6b76a3c3215ffa7613dd4957cb493d6f8df5`
- `.work/lab/cone_set_actuator_trial/protocol.md` `243528cec1daca7f7fb4e6e33cc82be4342de236472ee5aa3927bb0d76c399bb`
- `.work/lab/cone_set_actuator_trial/test_core.py` `153fc4b0e0eea691877b4a62309f355b6e037e9f0a125e9b1f9121968e3830ae`
- `.work/lab/cone_set_actuator_trial/test_training.py` `e6584d2a9464336f249188321be1b6757528ea4a4ebdc4c33a49359fbee14053`
- `.work/lab/next_strategy/cone_set_synthesis_v1.md` `4812acd4f64dcc08c96ebfd9a28afbca396116914e7e8060837fa83df5a3d74e`
- `.work/lab/next_strategy/cone_set_interface_v1.md` `bccb01f1f520d59ad9ff23d055944debfeaeb81d7bae2e99d6f1e7713d7494cd`
- `.work/lab/next_strategy/cone_set_actuator_design_v1.md` `a86c9df7ebd9bbe433779e193500ac05ac5a51fbea7ef87eaafe37789931bb3d`
- `.work/lab/joint_command_trial/joint_data.py` `ff48660fb8bb5f9a36541642bf91f98e3d67afc056b35089945ea2d0b9f96fe6`
- `src/fsop/eval/closed_loop.py` `a14ca4ee943f26921a1bf8ebfad67e12ed4c94b8c3031ddf4cbf4c360d62b98f`
- `src/fsop/eval/img_to_bev.py` `bcf10ece06c575512c45ea773335a1024fbc7ab95d20426a91ff18bd71d2d2d1`
- `src/fsop/data/native_replay.py` `483e089b37f8782ade87cc8f16eaf8a2447efa1e25ee9a6b79f5ac5eac9cb30b`
