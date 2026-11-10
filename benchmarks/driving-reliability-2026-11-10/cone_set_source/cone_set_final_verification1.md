# Cone-set final source verification 1

The goal is a vision driver that completes trackdrive and R,R,L,L skidpad with zero physical contacts. **PASS for the focused source-freeze verification.** No P1 or P2 issue was found in the requested fixes. This verdict does not qualify a model or establish mission performance.

The stage admission table requires each predecessor's completed receipt before creating a new attempt. Each stage attempt uses exclusive creation. A successor cannot enter while its predecessor has only written a provisional report. The allowed postflight child requires the live direct training parent, a matching train attempt, and `training_updates.json` written after the training data context closes. The training parent waits for that child and finalizes its own marker afterward. The child marker binds the provisional train receipt, but `selected_package` also requires the parent's completion marker. Source inspection confirms this order. The CPU tests exercise a real child process at admission and separately exercise parent-marker gating. They do not run the full postflight workload.

The selected package gate rechecks the training record's audit, condition, and overfit receipt and completion-marker hashes. Stage admission captures each predecessor receipt and marker hash. After the stage returns, `main` rechecks the source audit, selected package and applicable downstream gates, then compares the admission snapshot immediately before writing the completion marker. Tests rejected interrupted reports, mutated prerequisite chains, coherent receipt and marker replacements, and changes to consumed CPU and visual gates. The pool-close test held an owner context open while a successor attempted admission. These tests use temporary fixtures and mocked stage bodies where appropriate.

The replay verifier loads the sibling `evaluate.py` with `runpy.run_path`. The new test injected a foreign `sys.modules['evaluate']` and observed the sibling source inventory instead. Calling the actual verifier inventory returned 176 relative paths, all present in this worktree. The external canonical training recipe is absent locally and remains a Spark environment dependency.

The command `PYTHONPATH=src:. .venv/bin/python -m unittest discover -s .work/lab/cone_set_actuator_trial -p 'test_*.py' -v` passed all 39 tests. These comprise 8 core, 21 training, and 10 pipeline tests. The pipeline tests cover the raw, bounded, and applied command display, override and cone count in both views, terminal cards, and encoding with full video decode. Python compilation passed for every trial `.py` file. The `--help` command passed for the training, evaluation, replay, process, export, portable encoding, and runtime preflight CLIs. AST counts confirmed 72 `checks.compare` calls in both the inherited and current process files. The preceding pipeline review recorded the argument comparison and physical-check scope.

No GPU, simulator, SSH, optimizer training, final case, or physical episode ran. There is no local sealed cone-set episode or selected package for end-to-end replay. The CPU results establish the tested behavior and source wiring, not Spark data compatibility, training eligibility, or zero-contact performance. Prove It Works guided the decision to run the current suites and inventory rather than rely on prior reports.

## Source SHA-256

Recorded at 2026-11-10T10:02Z. All paths below are relative to `.work/lab/cone_set_actuator_trial/`.

```text
7e73e45ba42443fc1fb028d886677e2345393ea4b8b8c5f3f79bd8ba7b1251fb  cone_core.py
81ae1c0885db7c493de7dbc70b4b7c6570147c4e4d757656e23a4335cb3e82a8  encode_portable.py
82f9b90d52da591713d44ed30976ccdbf32a5fe3a81b6ee86d9064ecb63070a4  evaluate.py
fdcca66b463494ad08ec251987569c14665d865403875da6a36438fdeeb7a43f  export_audited_command.py
51dd7ac4d1d0d122cf77650043b35718cfe45c21cda99fdfefe47f28c5977f75  process.py
4739bb3d5771331b274bc5623cd8d4e01edf7169eeb6cb79498ace9377d68b36  runtime_preflight.py
153fc4b0e0eea691877b4a62309f355b6e037e9f0a125e9b1f9121968e3830ae  test_core.py
f0b6cc8059ddec0a865103b56afc1fed652e7460747468fda49bec3065255884  test_pipeline.py
fdd59d59282200d419ef66ec1654578fbf66c66a88f3453fb27115db842c0ef0  test_training.py
d319e7858bbead5fe6b4b5adb231cba05d469c9247b09835433322f445c95996  train_set.py
61641da1db7d38fda51da7ce4fefc6d339926568589b1c3ae41185308ff88220  verify_command_episode.py
5378430dda676cd4c4641111df935ad2fcd214501d4a290ca8a0a2bf3b626cac  recipe.json
b06af90eb9f739e0c0fa6261528ef7e80b500f13cfb39a8f7cf5ea24f1d2a107  status.json
56e4a54acdc270a3ea82271a74d49964c1dd80cca3467142e47b6d840e580869  pipeline_notes.md
86ec78729f62a6b66960ee282842c4ae3f6bdbea9106cdfae09dd45b6b8a4eb7  protocol.md
```
