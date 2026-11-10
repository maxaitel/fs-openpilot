# Cone-set comment audit 1

Read-only audit of all 11 new `.work/lab/cone_set_actuator_trial/*.py` files and the uncommitted `generate_report.py` diff plus the entire new, untracked `test_report.py`. Previous frozen sources and benchmark copied experiment sources were excluded. Snapshot hashes below identify the exact files observed; the author was concurrently editing, so recheck line numbers and hashes before applying.

## Recommendations

Recommended deletion count: **3** (two module docstrings and one suppression; no application source edited here).

- `runtime_preflight.py:1`: delete the module docstring. It narrates the script and is not passed to an argument parser or defining a public API contract.
- `test_pipeline.py:1`: delete the module docstring. It narrates the test module and is not a public API contract.
- `test_pipeline.py:197`: **MUST KILL** `# noqa: F401` on `cv2`. F401 detects unused imports, which can expose errors; this import is intentionally an availability check. Preserve the real import and `ImportError` behavior without the unused binding, e.g. `importlib.import_module('cv2')`, rather than dropping the check. This suggestion needs an author-applied code change; merely removing the suppression leaves the unused-import finding.

## Keeps and skips

- `encode_portable.py:244`: keep as a dependency guarantee relevant to safe publication, not a claim about custom control flow. The local `man 2 link` states that `link()` atomically creates the directory entry and reports `EEXIST` when the destination already exists. This proves the claim for the current host/POSIX interface, not independently for every possible supported host; the broad wording remains a portability qualification.
- The five CLI module docstrings in `encode_portable.py`, `evaluate.py`, `export_audited_command.py`, `process.py`, and `verify_command_episode.py` are used as `argparse.ArgumentParser(description=__doc__)` (five files; `evaluate.py` has a multiline description). Keep as user-facing command contracts. Executable shebangs are not prose comments.
- No comments or suppressions appear in the added `generate_report.py` diff or in the new `test_report.py`. No other scoped suppressions were found.

## SHA-256 snapshot

```text
7e73e45ba42443fc1fb028d886677e2345393ea4b8b8c5f3f79bd8ba7b1251fb  cone_set_actuator_trial/cone_core.py
bbffef7f10e701c75f9df03b14f9886492ce0bcd5b00f142480b500a15c0fdbc  cone_set_actuator_trial/encode_portable.py
82f9b90d52da591713d44ed30976ccdbf32a5fe3a81b6ee86d9064ecb63070a4  cone_set_actuator_trial/evaluate.py
f45d52d073d9d0e3974acbeece147c9c842338ebfc275d8b142125278756b83c  cone_set_actuator_trial/export_audited_command.py
51dd7ac4d1d0d122cf77650043b35718cfe45c21cda99fdfefe47f28c5977f75  cone_set_actuator_trial/process.py
ec99d0bfa9aceb88ab4e3b50849133409233eac8f1c226dc5fef53c7a54612f1  cone_set_actuator_trial/runtime_preflight.py
153fc4b0e0eea691877b4a62309f355b6e037e9f0a125e9b1f9121968e3830ae  cone_set_actuator_trial/test_core.py
9a519fec67ea69ddb9fcef6d865f21737d7d8ff9ec09db110d66dc196e76a79d  cone_set_actuator_trial/test_pipeline.py
115574dbe526c46c080b20846b257f0a0dd0b01550294a9f1c50f676e0dea6b0  cone_set_actuator_trial/test_training.py
ec9d1b86fe696a955a7dcf80e6d2e6cad926509366ae7961f0070f1e9b5b98e2  cone_set_actuator_trial/train_set.py
74c02f32d9c5c794e39291f968cac7355114e8235d24431dbb43d492c44df494  cone_set_actuator_trial/verify_command_episode.py
0f66d05d01260d45ea88b2d5a097ea5ffbceb852607c7d710e3b57fb6c3dd4b4  benchmarks/driving-reliability-2026-11-10/generate_report.py
f546635433b6a01dcf0e01972ca5998e12e8571586aad0c7265a768c75d1b8e8  benchmarks/driving-reliability-2026-11-10/test_report.py
```
