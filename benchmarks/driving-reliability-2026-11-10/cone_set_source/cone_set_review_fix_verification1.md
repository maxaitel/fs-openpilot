# Cone-set review fix verification 1

The goal is a zero-contact vision driver with a reproducible, bounded representation experiment. **ISSUES.** The lifecycle fixes are not ready for source freeze because two P2 gaps remain. This review covers only the accepted core and pipeline findings and their integration. No training, GPU, simulator, SSH, or final mission ran.

## P2 findings

1. **Stage start is not atomic across different stages.** `train_set.py:903-915` scans attempts, then exclusively creates a stage-specific attempt file. Two different stage starts can both finish the scan before either file exists, and both create their own attempt. A temporary-directory repro inserted a thread barrier after the real `reject_interrupted_attempt` and observed both `condition` and `overfit` return successfully from `begin_stage`. For example, starting these after audit can let overfit enter while condition owns its pool. Overfit then fails its missing prerequisite and writes a failure that invalidates the whole attempt. Exclusive creation of individual stage names does not enforce the accepted finding's prohibition on concurrent stages. Use a shared atomic admission mechanism, preserving the live child postflight exception and crash detection.

2. **Selected-package checks do not recheck their recorded prerequisite chain.** `train_set.py:222-243` verifies the train and postflight completion markers and selected package, but does not call `check_prerequisites` against `recorded_prerequisites(train.json)`. The initial `begin_stage` scan catches receipt tampering that already exists. It does not catch mutation after that scan. In a temporary fixture with correctly bound audit, condition, overfit, train and postflight markers, `begin_stage(cpu_validate)` succeeded. After changing `condition.json`, `selected_package` still returned the package, while an explicit `check_prerequisites` rejected the same change. Downstream stage finalization calls `selected_package` again without validating that chain, so it can complete with a changed prerequisite. This leaves the accepted requirement that selected-package gates bind the exact consumed receipts unresolved.

## Checks that passed

- The fit binds audit, condition and overfit receipt and completion-marker hashes before its update loop. It checks the overfit-to-condition receipt hash and compares the bound chain before `training_updates.json`, postflight reports and final train/postflight completion. Source inspection confirms the owning data contexts exit before the enclosing stage returns.
- Completion markers bind the receipt, stage attempt and owner PID. `main` writes the marker after the stage returns and final source checks. Child postflight skips the active train attempt, checks its live direct parent, and cannot finalize the parent marker. `selected_package` requires both markers. A report without a completion marker, and a changed completed report present at stage admission, fail closed. These checks do not repair the two gaps above.
- All 31 CPU tests passed with `.venv/bin/python -m unittest discover -s .work/lab/cone_set_actuator_trial -p 'test_*.py' -v` and `PYTHONPATH=src:.`. The tests comprise 8 core, 14 training and 9 pipeline tests. The pipeline encoder fixture decoded both video views and their 40 terminal frames. The overlay test exercises distinct raw, bounded and applied commands, a nonempty 200-entry mask, the override flag and both views. It checks that terminal cards do not show a current cone count. The actual renderer calls the annotated functions only for inference frames and writes terminal cards directly.
- `train_set.py --help` succeeded. The producer's `evaluate.source_inventory()` returned 176 relative entries, all present. The separate training inventory could not complete locally because the producer-only `/work/docs/isaac-lab-training/phase_recovery_recipe4.json` is absent. This is an environment limit, not a code finding.
- AST comparison found 72 inherited `checks.compare` calls in each process file. The sole argument change casts saved controller proprioception to FP32. Physical comparison calls remain unchanged. Replay retains absolute and relative `1e-5` tolerances.

These checks do not establish real-data training success, model eligibility, or zero-contact mission performance. Prove It Works shaped the decision to run the actual CPU suites and reproduce both gaps, rather than infer correctness from the new markers.

## Reviewed source SHA-256

Recorded at 2026-11-10T09:46:03Z. The hashes identify the reviewed snapshot and are not a source-freeze approval.

```text
7e73e45ba42443fc1fb028d886677e2345393ea4b8b8c5f3f79bd8ba7b1251fb  cone_core.py
ec9d1b86fe696a955a7dcf80e6d2e6cad926509366ae7961f0070f1e9b5b98e2  train_set.py
5378430dda676cd4c4641111df935ad2fcd214501d4a290ca8a0a2bf3b626cac  recipe.json
a464dbe49e902a8430c159dd086729ca690f8619dfab9ef6d4d16f94195700a3  protocol.md
153fc4b0e0eea691877b4a62309f355b6e037e9f0a125e9b1f9121968e3830ae  test_core.py
115574dbe526c46c080b20846b257f0a0dd0b01550294a9f1c50f676e0dea6b0  test_training.py
82f9b90d52da591713d44ed30976ccdbf32a5fe3a81b6ee86d9064ecb63070a4  evaluate.py
74c02f32d9c5c794e39291f968cac7355114e8235d24431dbb43d492c44df494  verify_command_episode.py
51dd7ac4d1d0d122cf77650043b35718cfe45c21cda99fdfefe47f28c5977f75  process.py
fdcca66b463494ad08ec251987569c14665d865403875da6a36438fdeeb7a43f  export_audited_command.py
81ae1c0885db7c493de7dbc70b4b7c6570147c4e4d757656e23a4335cb3e82a8  encode_portable.py
56e4a54acdc270a3ea82271a74d49964c1dd80cca3467142e47b6d840e580869  pipeline_notes.md
6e0741522f1dd19c1b09e85c26dc37ee7d3409ee185c8cf63fdce87e041b0e7f  test_pipeline.py
4739bb3d5771331b274bc5623cd8d4e01edf7169eeb6cb79498ace9377d68b36  runtime_preflight.py
```

The listed files are under `.work/lab/cone_set_actuator_trial/`.
