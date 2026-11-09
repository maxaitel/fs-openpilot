"""Freeze a caller-selected checkpoint and the complete predeclared 85-episode coverage.

This verifies training provenance and every supplied completed development episode;
it does not select a winner, execute test cases, or claim that any target was met.
Cases are resolved as COVERAGE_DIR/cases/<case-id>.json. The two generated plan
files are accepted by run_lab_evaluations.py; selection.json binds their hashes
and the exact case-file hashes. Verify those bindings before launching each plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from scripts.lab_comparison import compare_conditions, comparison_conditions
from scripts.lab_training_provenance import MEMBERSHIP, dataset_paths, validation_identity
from scripts.run_lab_evaluations import digest, normalize_case, validate_episode


def read_json(path):
    return json.loads(Path(path).read_text())


def state_digest(state):
    """Use the evaluator's exact ordered key-plus-contiguous-tensor-byte digest."""
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def checkpoint_info(path):
    import torch

    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict) or not data["model"]:
        raise ValueError("Checkpoint does not contain the complete model state dictionary")
    state = data["model"]
    if any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("Model state must contain only named tensors")
    if not isinstance(data.get("step"), int) or isinstance(data["step"], bool) or data["step"] < 0:
        raise ValueError("Checkpoint requires a nonnegative integer step")
    groups = {}
    for group, prefix in (
        ("vision", "base.vision."),
        ("policy", "base.policy."),
        ("bev_head", "bev."),
        ("image_head", "img."),
    ):
        subset = {key: value for key, value in state.items() if key.startswith(prefix)}
        if not subset:
            raise ValueError(f"Checkpoint is missing required {group} tensors")
        groups[group] = {"tensor_count": len(subset), "state_sha256": state_digest(subset)}
    signature = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in state.items()}
    return {
        "checkpoint_sha256": digest(path),
        "checkpoint_size_bytes": Path(path).stat().st_size,
        "step": data["step"],
        "state_sha256": state_digest(state),
        "tensor_count": len(state),
        "groups": groups,
        "signature": signature,
    }


def verify_training(training, dataset, verification_path, baseline_sha, candidate_sha, candidate_step):
    training, dataset, verification_path = Path(training), Path(dataset), Path(verification_path)
    run_path, history_path = training / "run.json", training / "history.jsonl"
    run, manifest, verified = read_json(run_path), read_json(dataset), read_json(verification_path)
    manifest_sha = digest(dataset)
    if run.get("init_sha256") != baseline_sha or run.get("manifest_sha256") != manifest_sha:
        raise ValueError("Training initialization or dataset does not match selected inputs")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("complete") is not True
        or verified.get("passed") is not True
        or verified.get("manifest_sha256") != manifest_sha
        or not manifest.get("membership_sha256")
        or verified.get("membership_sha256") != manifest["membership_sha256"]
    ):
        raise ValueError("Dataset verification is missing, failed, or not bound to the dataset membership")
    training_verification = run.get("dataset_verification", {})
    if (
        training_verification.get("status") != "verified"
        or training_verification.get("sha256") != digest(verification_path)
        or training_verification.get("manifest_sha256") != manifest_sha
        or training_verification.get("membership_sha256") != manifest["membership_sha256"]
    ):
        raise ValueError("Training run is not bound to this dataset verification receipt")
    source = run["source"]
    source_path = training / "source.zip"
    if digest(source_path) != source["archive_sha256"]:
        raise ValueError("Training source archive hash changed")
    with ZipFile(source_path) as archive:
        if sorted(archive.namelist()) != sorted(source["files_sha256"]):
            raise ValueError("Training source archive inventory changed")
        for name, expected in source["files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Training source archive content changed")
    rows = []
    for number, line in enumerate(history_path.read_text().splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid history row {number}")
            rows.append(row)
    baseline_path = training / "baseline_validation.json"
    baseline = read_json(baseline_path)
    matches = []
    for field in ("selected_sha256", "checkpoint_sha256"):
        if baseline.get(field) == candidate_sha:
            matches.append({"file": str(baseline_path.resolve()), "field": field, "step": baseline.get("step")})
    for row in rows:
        for field in ("checkpoint_sha256", "selected_sha256"):
            if row.get(field) == candidate_sha:
                matches.append({"file": str(history_path.resolve()), "field": field, "step": row.get("step")})
    if not matches:
        raise ValueError("Candidate checkpoint identity is absent from training history and step-zero validation")
    if any(row["step"] != candidate_step for row in matches):
        raise ValueError("Candidate checkpoint step differs from its recorded training step")
    return {
        "run_sha256": digest(run_path),
        "history_sha256": digest(history_path),
        "baseline_validation_sha256": digest(baseline_path),
        "source_archive_sha256": digest(source_path),
        "dataset_sha256": manifest_sha,
        "dataset": str(dataset.resolve()),
        "dataset_verification_sha256": digest(verification_path),
        "dataset_verification": str(verification_path.resolve()),
        "membership_sha256": manifest["membership_sha256"],
        "candidate_records": matches,
    }


def verify_training_chain(
    training, dataset, verification, baseline, candidate, ancestor_specs, ancestor_datasets=None, membership=MEMBERSHIP
):
    if not ancestor_specs:
        if ancestor_datasets:
            raise ValueError("Ancestor dataset overrides require ancestor training runs")
        return verify_training(
            training,
            dataset,
            verification,
            baseline["checkpoint_sha256"],
            candidate["checkpoint_sha256"],
            candidate["step"],
        )
    stages = [(Path(run).resolve(), Path(checkpoint).resolve()) for run, checkpoint in ancestor_specs]
    run_paths = [run for run, _ in stages] + [Path(training).resolve()]
    if len(set(run_paths)) != len(run_paths):
        raise ValueError("Duplicate or cyclic training runs")
    overrides = {}
    for run, manifest, receipt in ancestor_datasets or []:
        run = Path(run).resolve()
        if run not in run_paths[:-1] or run in overrides:
            raise ValueError("Unknown or duplicate ancestor dataset override")
        overrides[run] = (manifest, receipt)
    infos = [checkpoint_info(path) for _, path in stages]
    hashes = [
        baseline["checkpoint_sha256"],
        *(info["checkpoint_sha256"] for info in infos),
        candidate["checkpoint_sha256"],
    ]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate or cyclic training checkpoints")
    if any(info["signature"] != baseline["signature"] for info in [*infos, candidate]):
        raise ValueError("Ancestor checkpoint does not retain the complete model signature")
    ancestors = []
    fingerprint = None
    membership_sha = None
    input_sha256 = {}
    for index, ((run, checkpoint), info) in enumerate(zip(stages, infos, strict=True)):
        recorded = read_json(run / "run.json")
        stage_dataset, stage_verification = dataset_paths(recorded, *overrides.get(run, (None, None)))
        proof = verify_training(run, stage_dataset, stage_verification, hashes[index], hashes[index + 1], info["step"])
        identity = validation_identity(recorded, stage_dataset, stage_verification, membership)
        if fingerprint is not None and (
            identity["fingerprint"] != fingerprint or identity["membership_sha256"] != membership_sha
        ):
            raise ValueError("Training chain validation fingerprint or membership differs")
        fingerprint, membership_sha = identity["fingerprint"], identity["membership_sha256"]
        input_sha256.update(identity["input_sha256"])
        proof["validation_fingerprint"] = fingerprint
        ancestors.append(
            {
                "path": str(run),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": info["checkpoint_sha256"],
                "step": info["step"],
                **proof,
            }
        )
    proof = verify_training(training, dataset, verification, hashes[-2], hashes[-1], candidate["step"])
    identity = validation_identity(read_json(Path(training) / "run.json"), dataset, verification, membership)
    if identity["fingerprint"] != fingerprint or identity["membership_sha256"] != membership_sha:
        raise ValueError("Training chain validation fingerprint or membership differs")
    input_sha256.update(identity["input_sha256"])
    proof["validation_fingerprint"] = fingerprint
    proof["dataset_input_sha256"] = input_sha256
    proof["ancestors"] = ancestors
    return proof


def verify_development(lock_paths, candidate_sha, baseline_sha, candidate_state_sha, baseline_state_sha):
    if not lock_paths:
        raise ValueError("At least one completed development plan is required")
    episodes, locks, case_defs = [], [], {}
    pairs = {"candidate": {}, "baseline": {}}
    seen_locks = set()
    lock_conditions = None
    for lock_path in lock_paths:
        lock_path = Path(lock_path).resolve()
        if lock_path in seen_locks:
            raise ValueError("Duplicate development lock")
        seen_locks.add(lock_path)
        lock = read_json(lock_path)
        if lock_path.name != "plan.lock.json":
            raise ValueError("Development plan must be an evaluation plan.lock.json")
        root = lock_path.parent
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("complete") is not True or manifest.get("lock_sha256") != digest(lock_path):
            raise ValueError("Development manifest is incomplete or belongs to a different plan")
        if not any(m["sha256"] in (candidate_sha, baseline_sha) for m in lock["models"]):
            raise ValueError("Development plan does not include a selected model")
        fields = ("source_sha256", "graph_sha256", "coordinator_sha256", "timeout_seconds")
        if any(field not in lock or lock[field] is None for field in fields):
            raise ValueError("Development lock is missing comparison conditions")
        conditions = {field: lock[field] for field in fields}
        if lock_conditions is None:
            lock_conditions = conditions
        elif conditions != lock_conditions:
            raise ValueError("Development lock comparison conditions differ")
        if any(row["case"]["split"] not in ("val", "regression") for row in lock["cases"]):
            raise ValueError("Development plans cannot contain fresh test cases")
        expected_keys = {f"{m['name']}/{c['name']}" for c in lock["cases"] for m in lock["models"]}
        if set(manifest["episodes"]) != expected_keys:
            raise ValueError("Development manifest does not cover the complete frozen plan")
        for case in lock["cases"]:
            identity = case["case"]["id"]
            previous = case_defs.setdefault(identity, case["case"])
            if previous != case["case"]:
                raise ValueError(f"Development case identity has conflicting definitions: {identity}")
            for model in lock["models"]:
                key = f"{model['name']}/{case['name']}"
                row = manifest["episodes"][key]
                if row.get("status") != "complete":
                    raise ValueError(f"Development episode is not complete: {key}")
                episode = root / model["name"] / case["name"]
                receipt = validate_episode(episode, case, model, lock)
                if row.get("receipt") != receipt:
                    raise ValueError(f"Development episode receipt changed: {key}")
                audit_path = episode / "lab_artifact_audit.json"
                audit = read_json(audit_path)
                if (
                    audit.get("passed") is not True
                    or audit.get("completion_sha256") != receipt["completion_sha256"]
                    or audit.get("frames_sha256") != receipt["frames_sha256"]
                    or audit.get("frame_count") != receipt["frame_count"]
                    or audit.get("telemetry_sha256") != digest(episode / "telemetry.json")
                ):
                    raise ValueError(f"Development artifact audit is missing, failed or unbound: {key}")
                role = None
                comparison_artifacts = {}
                if model["sha256"] == candidate_sha:
                    role = "candidate"
                if model["sha256"] == baseline_sha:
                    if candidate_sha == baseline_sha:
                        role = model["name"] if model["name"] in pairs else None
                    else:
                        role = "baseline"
                if role:
                    expected_state = candidate_state_sha if role == "candidate" else baseline_state_sha
                    metadata = read_json(episode / "evidence/metadata.json")
                    recorded_state = metadata.get("state_sha256")
                    if recorded_state != expected_state:
                        raise ValueError(f"Development model state differs from selected checkpoint: {key}")
                    if identity in pairs[role]:
                        raise ValueError(f"Duplicate selected development episode: {role}/{identity}")
                    pairs[role][identity] = comparison_conditions(episode, metadata)
                    comparison_artifacts = {
                        "evidence/metadata.json": digest(episode / "evidence/metadata.json"),
                        "source.zip": metadata["source_archive_sha256"],
                        "realization.json": metadata["realization_sha256"],
                        "appearance.json": metadata["appearance_sha256"],
                        "export.json": metadata["export_sha256"],
                        **{
                            f"assets/{name}": value
                            for name, value in pairs[role][identity]["asset_dependencies"].items()
                        },
                    }
                episodes.append(
                    {
                        "plan": str(lock_path),
                        "key": key,
                        "case": case["case"],
                        "model_sha256": model["sha256"],
                        "role": role,
                        "receipt": receipt,
                        "summary_sha256": digest(episode / "summary.json"),
                        "summary": read_json(episode / "summary.json"),
                        "audit_sha256": digest(audit_path),
                        "terminal_condition_verified": audit.get("terminal_condition_verified"),
                        "comparison_artifacts": comparison_artifacts,
                    }
                )
        locks.append({"path": str(lock_path), "sha256": digest(lock_path), "manifest_sha256": digest(manifest_path)})
    shared = pairs["candidate"].keys() & pairs["baseline"].keys()
    if shared != pairs["candidate"].keys():
        raise ValueError("Every selected-candidate development case requires a matching baseline episode")
    for identity in shared:
        compare_conditions(pairs["baseline"][identity], pairs["candidate"][identity])
    if not any(case_defs[i]["split"] == "val" and not case_defs[i].get("frozen_case") for i in shared):
        raise ValueError("A paired native validation case is required")
    if not any(case_defs[i]["split"] == "regression" and case_defs[i].get("frozen_case") for i in shared):
        raise ValueError("A paired exposed frozen regression case is required")
    return {
        "plans": locks,
        "episodes": episodes,
        "paired_case_ids": sorted(shared),
        "validation_cases": sorted(i for i in shared if case_defs[i]["split"] == "val"),
        "regression_cases": sorted(i for i in shared if case_defs[i]["split"] == "regression"),
    }


def coverage_plans(coverage_path, membership_sha, baseline, candidate, timeout_seconds):
    coverage_path = Path(coverage_path).resolve()
    coverage = read_json(coverage_path)
    paired, legacy = coverage["paired_case_ids"], coverage["candidate_only_legacy_case_ids"]
    if (
        coverage.get("schema_version") != 1
        or coverage.get("frozen_before_training") is not True
        or coverage.get("collection_membership_sha256") != membership_sha
        or not isinstance(paired, list)
        or not isinstance(legacy, list)
        or len(paired) != 24
        or len(legacy) != 37
        or len(set(paired + legacy)) != 61
        or (coverage.get("baseline_episodes"), coverage.get("candidate_episodes"), coverage.get("total_episodes"))
        != (24, 61, 85)
    ):
        raise ValueError("Coverage must bind the dataset membership and the frozen 24/37 case lists")
    inventory, counts = {}, {"test": 0, "regression": 0}
    for case_id in paired + legacy:
        if not isinstance(case_id, str) or not case_id or not all(c.isalnum() or c in "_.-" for c in case_id):
            raise ValueError("Invalid coverage case ID")
        path = (coverage_path.parent / "cases" / f"{case_id}.json").resolve()
        if not path.is_relative_to((coverage_path.parent / "cases").resolve()):
            raise ValueError("Coverage case path escapes the case directory")
        case = normalize_case(read_json(path))
        if case["id"] != case_id or case["split"] not in ("test", "regression"):
            raise ValueError(f"Coverage case identity or split is invalid: {case_id}")
        if case_id in legacy and (case["split"] != "regression" or not case.get("frozen_case")):
            raise ValueError("Legacy cases must be exposed frozen regressions")
        if case_id in paired:
            counts[case["split"]] += 1
        inventory[case_id] = {"path": str(path), "sha256": digest(path), "case": case}
    if counts != {"test": 16, "regression": 8}:
        raise ValueError("Paired coverage must contain 16 fresh tests and 8 exposed regressions")

    def plan(ids, models):
        return {
            "version": 1,
            "timeout_seconds": timeout_seconds,
            "cases": [inventory[case_id]["path"] for case_id in ids],
            "models": models,
        }

    fresh = [case_id for case_id in paired if inventory[case_id]["case"]["split"] == "test"]
    return (
        coverage,
        inventory,
        plan(paired, [baseline, candidate]),
        plan(legacy, [candidate]),
        plan(fresh, [baseline, candidate]),
    )


def recheck_inputs(
    candidate_path,
    baseline_path,
    candidate,
    baseline,
    training,
    dataset,
    verification,
    training_provenance,
    coverage,
    coverage_sha,
    membership_path,
    inventory,
    development,
):
    """Catch input changes during the potentially lengthy episode validation."""
    expected_files = {
        candidate_path: candidate["checkpoint_sha256"],
        baseline_path: baseline["checkpoint_sha256"],
        training / "run.json": training_provenance["run_sha256"],
        training / "history.jsonl": training_provenance["history_sha256"],
        training / "baseline_validation.json": training_provenance["baseline_validation_sha256"],
        training / "source.zip": training_provenance["source_archive_sha256"],
        dataset: training_provenance["dataset_sha256"],
        verification: training_provenance["dataset_verification_sha256"],
        coverage: coverage_sha,
        membership_path: training_provenance["membership_sha256"],
    }
    for row in training_provenance.get("ancestors", []):
        run = Path(row["path"])
        expected_files[Path(row["checkpoint"])] = row["checkpoint_sha256"]
        expected_files[Path(row["dataset"])] = row["dataset_sha256"]
        expected_files[Path(row["dataset_verification"])] = row["dataset_verification_sha256"]
        for filename, key in (
            ("run.json", "run_sha256"),
            ("history.jsonl", "history_sha256"),
            ("baseline_validation.json", "baseline_validation_sha256"),
            ("source.zip", "source_archive_sha256"),
        ):
            expected_files[run / filename] = row[key]
    expected_files.update(
        {Path(path): value for path, value in training_provenance.get("dataset_input_sha256", {}).items()}
    )
    for row in inventory.values():
        expected_files[Path(row["path"])] = row["sha256"]
    for row in development["plans"]:
        path = Path(row["path"])
        expected_files[path] = row["sha256"]
        expected_files[path.parent / "manifest.json"] = row["manifest_sha256"]
    for row in development["episodes"]:
        episode = Path(row["plan"]).parent / row["key"]
        expected_files[episode / "summary.json"] = row["summary_sha256"]
        expected_files[episode / "lab_artifact_audit.json"] = row["audit_sha256"]
        expected_files[episode / "evidence/completion.json"] = row["receipt"]["completion_sha256"]
        expected_files[episode / "frames.json"] = row["receipt"]["frames_sha256"]
        for name, expected in row.get("comparison_artifacts", {}).items():
            expected_files[episode / name] = expected
    for path, expected in expected_files.items():
        if digest(path) != expected:
            raise ValueError(f"Input changed during selection verification: {path}")


def freeze(args):
    candidate_path, baseline_path = Path(args.candidate).resolve(), Path(args.baseline).resolve()
    training, dataset, coverage = (
        Path(args.training).resolve(),
        Path(args.dataset).resolve(),
        Path(args.coverage).resolve(),
    )
    verification = (
        Path(args.dataset_verification).resolve() if args.dataset_verification else dataset.parent / "verification.json"
    )
    if not args.reason.strip():
        raise ValueError("A concrete caller-provided selection reason is required")
    if not 0 < args.timeout_seconds < float("inf"):
        raise ValueError("timeout_seconds must be positive and finite")
    coverage_sha = digest(coverage)
    candidate, baseline = checkpoint_info(candidate_path), checkpoint_info(baseline_path)
    if candidate["signature"] != baseline["signature"]:
        raise ValueError("Candidate does not retain the complete baseline model tensor keys, shapes and dtypes")
    training_provenance = verify_training_chain(
        training,
        dataset,
        verification,
        baseline,
        candidate,
        getattr(args, "ancestor_training", None) or [],
        getattr(args, "ancestor_dataset", None),
        coverage.parent / "membership.json",
    )
    development = verify_development(
        args.development_plan,
        candidate["checkpoint_sha256"],
        baseline["checkpoint_sha256"],
        candidate["state_sha256"],
        baseline["state_sha256"],
    )
    model_base = {"name": "baseline", "checkpoint": str(baseline_path), "sha256": baseline["checkpoint_sha256"]}
    model_candidate = {"name": "candidate", "checkpoint": str(candidate_path), "sha256": candidate["checkpoint_sha256"]}
    cov, inventory, paired_plan, legacy_plan, fresh_plan = coverage_plans(
        coverage, training_provenance["membership_sha256"], model_base, model_candidate, args.timeout_seconds
    )
    membership_path = coverage.parent / "membership.json"
    if digest(membership_path) != training_provenance["membership_sha256"]:
        raise ValueError("Membership file differs from the verified dataset and frozen coverage")
    membership = read_json(membership_path)
    if membership.get("schema_version") != 1 or membership.get("frozen_before_collection") is not True:
        raise ValueError("Membership must be frozen before collection")
    val_rows = [row for row in membership["cases"] if row["split"] == "val"]
    reg_rows = [row for row in membership["cases"] if row["split"] == "regression"]
    expected_val = {row["id"]: normalize_case(row) for row in val_rows}
    expected_reg = {row["id"]: normalize_case(row) for row in reg_rows}
    paired_reg = {i for i in cov["paired_case_ids"] if inventory[i]["case"]["split"] == "regression"}
    if (
        len(val_rows) != 8
        or len(reg_rows) != 8
        or len(expected_val) != 8
        or len(expected_reg) != 8
        or paired_reg != set(expected_reg)
    ):
        raise ValueError("Membership must bind the eight validation and eight paired regression cases")
    for identity, case in {**expected_val, **expected_reg}.items():
        relevant = [e for e in development["episodes"] if e["case"]["id"] == identity and e["role"]]
        if {e["role"] for e in relevant} != {"baseline", "candidate"} or any(e["case"] != case for e in relevant):
            raise ValueError(f"Missing or mismatched paired development episode for {identity}")
        if identity in expected_reg and inventory[identity]["case"] != case:
            raise ValueError(f"Regression case differs between membership and final coverage: {identity}")
    development["reusable_paired_regression_episodes"] = [
        e for e in development["episodes"] if e["case"]["id"] in expected_reg and e["role"]
    ]
    if set(development["paired_case_ids"]) & {
        i for i in cov["paired_case_ids"] if inventory[i]["case"]["split"] == "test"
    }:
        raise ValueError("Development evidence includes a fresh test case")
    recheck_inputs(
        candidate_path,
        baseline_path,
        candidate,
        baseline,
        training,
        dataset,
        verification,
        training_provenance,
        coverage,
        coverage_sha,
        membership_path,
        inventory,
        development,
    )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, payload in (
        ("paired_plan.json", paired_plan),
        ("legacy_plan.json", legacy_plan),
        ("fresh_plan.json", fresh_plan),
    ):
        if getattr(args, "portable_plans", False):
            payload = {
                **payload,
                "cases": [os.path.relpath(path, out) for path in payload["cases"]],
                "models": [
                    {**model, "checkpoint": os.path.relpath(model["checkpoint"], out)} for model in payload["models"]
                ],
            }
        (out / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    # Signatures are checked above; retain a hash rather than a verbose per-tensor table.
    signatures = {
        name: hashlib.sha256(json.dumps(info.pop("signature"), sort_keys=True).encode()).hexdigest()
        for name, info in (("candidate", candidate), ("baseline", baseline))
    }
    selection = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "reason": args.reason,
        "candidate": {"path": str(candidate_path), **candidate},
        "baseline": {"path": str(baseline_path), **baseline},
        "tensor_signature_sha256": signatures,
        "training": {
            "path": str(training),
            "dataset": str(dataset),
            "verification": str(verification),
            **training_provenance,
        },
        "development": development,
        "coverage": {
            "path": str(coverage),
            "sha256": coverage_sha,
            "paired_case_ids": cov["paired_case_ids"],
            "candidate_only_legacy_case_ids": cov["candidate_only_legacy_case_ids"],
            "cases": inventory,
            "membership_path": str(membership_path),
            "membership_sha256": digest(membership_path),
            "total_episodes": 85,
        },
        "plans": {
            "paired_plan.json": digest(out / "paired_plan.json"),
            "legacy_plan.json": digest(out / "legacy_plan.json"),
            "fresh_plan.json": digest(out / "fresh_plan.json"),
        },
        "limitations": [
            "Selection reflects supplied development evidence and the caller's reason; it declares no success.",
            "This checks only supplied development plans; it cannot establish what was viewed elsewhere.",
            "Before running a plan, compare its file and case hashes to this selection receipt.",
            "The full paired plan includes audited exposed regressions; use fresh_plan for remaining test episodes.",
        ],
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument(
        "--ancestor-training",
        nargs=2,
        action="append",
        metavar=("RUN_DIR", "CHECKPOINT"),
        help="Prior run and its actual selected checkpoint, in chronological order; repeat for each stage",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--ancestor-dataset",
        nargs=3,
        action="append",
        metavar=("RUN_DIR", "MANIFEST", "RECEIPT"),
        help="Explicit dataset paths for this ancestor, required when its recorded paths are unavailable locally",
    )
    parser.add_argument("--dataset-verification", type=Path)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--development-plan", type=Path, action="append", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument(
        "--portable-plans",
        action="store_true",
        help="Use paths relative to the selection directory in generated plans",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selection = freeze(args)
    print(
        json.dumps(
            {
                "candidate_sha256": selection["candidate"]["checkpoint_sha256"],
                "step": selection["candidate"]["step"],
                "selection_sha256": digest(args.out / "selection.json"),
                "paired_episodes": 48,
                "legacy_episodes": 37,
            }
        )
    )


if __name__ == "__main__":
    main()
