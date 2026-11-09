"""Build a local CPU-only report from Lab training and sealed evaluation artifacts.

Example: python scripts/report_lab_training.py --training runs/train --evaluation runs/eval --out runs/report
The HTML links to existing videos; it does not copy or change source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from zipfile import BadZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.lab_comparison import compare_conditions, comparison_conditions


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sealed_summary(folder):
    """Verify the completion seal and membership, not every NPZ's contents."""
    evidence = folder / "evidence"
    receipt = read_json(evidence / "completion.json")
    metadata_bytes = (evidence / "metadata.json").read_bytes()
    manifest_bytes = (evidence / "manifest.jsonl").read_bytes()
    metadata = json.loads(metadata_bytes)
    entries = [json.loads(line) for line in manifest_bytes.splitlines()]
    if receipt.get("version") != 1 or receipt.get("frame_count") != len(entries):
        raise ValueError("completion version or frame count mismatch")
    if receipt.get("metadata_sha256") != hashlib.sha256(metadata_bytes).hexdigest():
        raise ValueError("metadata hash mismatch")
    if receipt.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("manifest hash mismatch")
    if (evidence / receipt["summary_path"]).resolve() != (folder / "summary.json").resolve():
        raise ValueError("receipt does not bind this summary")
    if receipt.get("summary_sha256") != digest(folder / "summary.json"):
        raise ValueError("summary hash mismatch")
    names = []
    for i, entry in enumerate(entries):
        expected = f"frame_{i:06d}.npz"
        if (
            entry.get("frame") != i
            or entry.get("file") != expected
            or entry.get("metadata_sha256") != receipt["metadata_sha256"]
        ):
            raise ValueError("invalid frame manifest entry")
        names.append(expected)
    if set(names) != {p.name for p in evidence.glob("frame_*.npz")}:
        raise ValueError("evidence frame membership mismatch")
    summary = read_json(folder / "summary.json")
    if metadata.get("case") != summary.get("case") or metadata.get("pilot_only") != summary.get("pilot_only"):
        raise ValueError("case or pilot identity mismatch")
    if not isinstance(summary.get("pilot_only"), bool):
        raise ValueError("pilot status missing")
    if not summary["pilot_only"] and summary.get("result") in (None, "running"):
        raise ValueError("non-pilot summary is not terminal")
    for key in ("checkpoint_sha256", "state_sha256"):
        if not metadata.get(key) or metadata.get(key) != summary.get(key):
            raise ValueError(f"{key} identity mismatch")
    if summary.get("recorded_frames") != len(entries) or not entries:
        raise ValueError("summary frame count mismatch")
    return summary, receipt, metadata


def audit_status(folder, receipt):
    path = folder / "lab_artifact_audit.json"
    if not path.is_file():
        return "missing", None
    try:
        audit = read_json(path)
        if audit.get("passed") is not True:
            return "failed", audit
        if (
            audit.get("completion_sha256") != digest(folder / "evidence/completion.json")
            or audit.get("frame_count") != receipt.get("frame_count")
            or audit.get("frames_sha256") != digest(folder / "frames.json")
            or audit.get("telemetry_sha256") != digest(folder / "telemetry.json")
        ):
            return "stale or unbound", audit
        return "passed", audit
    except (OSError, ValueError, TypeError):
        return "unreadable", None


def video_status(folder, name, receipt):
    video = folder / f"{name}.mp4"
    if not video.is_file():
        return {"status": "missing", "path": None}
    result = {"status": "present; unverified", "path": str(video.resolve())}
    try:
        audit = read_json(folder / "lab_video_audit.json")
        item = audit["videos"][name]
        if item.get("sha256") != digest(video):
            result["status"] = "hash mismatch"
        elif (
            audit.get("completion_sha256") != digest(folder / "evidence/completion.json")
            or audit.get("artifact_audit_sha256") != digest(folder / "lab_artifact_audit.json")
            or audit.get("inference_frames") != receipt.get("frame_count")
            or audit.get("terminal_frames") != 40
            or item.get("decoded_frames") != receipt.get("frame_count") + 40
            or item.get("fps") != 20
        ):
            result["status"] = "receipt mismatch"
        else:
            result["status"] = "receipt hash/count matched"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return result


def latency(folder, audit, audit_state):
    path = folder / "frames.json"
    if not path.is_file():
        return {"status": "missing"}
    try:
        rows = read_json(path)
        values = [row["policy_with_capture_s"] * 1000 for row in rows if finite(row.get("policy_with_capture_s"))]
        status = (
            "audit-bound" if audit_state == "passed" and audit.get("frames_sha256") == digest(path) else "unverified"
        )
        if not values:
            return {"status": status, "count": 0}
        return {
            "status": status,
            "count": len(values),
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
        }
    except (OSError, ValueError, TypeError, AttributeError):
        return {"status": "unreadable"}


def load_episode(folder, model):
    row = {
        "model": model,
        "case_id": folder.name,
        "path": str(folder.resolve()),
        "seal": "missing or invalid",
        "audit": "not available",
        "videos": {},
        "latency": {"status": "unavailable"},
    }
    try:
        summary, receipt, metadata = sealed_summary(folder)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        row["seal_error"] = str(exc)
        # Missing results and videos remain visible, even for interrupted episodes.
        for name in ("debug", "research"):
            row["videos"][name] = video_status(folder, name, {})
        return row
    case = summary.get("case", {})
    row.update(
        seal="verified",
        case=case,
        case_id=case.get("id", folder.name),
        split=case.get("split", "unknown"),
        discipline=case.get("discipline", summary.get("discipline", "unknown")),
        variant=case.get("variant", "unknown"),
        dr=case.get("dr", "unknown"),
        command_mode=metadata.get("command_mode") if metadata.get("command_mode") in COMMAND_MODES else "unknown",
        pilot=summary.get("pilot_only"),
        result=summary.get("result"),
        passed=summary.get("pass"),
        physical_contacts=summary.get("physical_cone_hits"),
        off_course=summary.get("oc"),
        progress=summary.get("progress"),
        laps=summary.get("laps"),
        wall_seconds=summary.get("wall_seconds"),
        rtf=summary.get("real_time_factor"),
        simulated_seconds=summary.get("seconds_simulated"),
        checkpoint_sha256=summary.get("checkpoint_sha256"),
        frame_count=receipt["frame_count"],
    )
    try:
        conditions = comparison_conditions(folder, metadata)
        original = folder.resolve()
        root = original.parent.parent
        lock_path = root / "plan.lock.json"
        lock, manifest = read_json(lock_path), read_json(root / "manifest.json")
        if manifest.get("complete") is not True or manifest.get("lock_sha256") != digest(lock_path):
            raise ValueError("Original evaluation manifest is incomplete or unbound")
        key = f"{original.parent.name}/{original.name}"
        models, cases = lock["models"], lock["cases"]
        expected = {f"{model['name']}/{case['name']}" for model in models for case in cases}
        if set(manifest["episodes"]) != expected or any(
            item.get("status") != "complete" for item in manifest["episodes"].values()
        ):
            raise ValueError("Original evaluation manifest does not cover its complete plan")
        if not any(
            model["name"] == original.parent.name and model["sha256"] == metadata["checkpoint_sha256"]
            for model in models
        ):
            raise ValueError("Original lock does not bind the episode checkpoint")
        if not any(case["name"] == original.name and case["case"] == metadata["case"] for case in cases):
            raise ValueError("Original lock does not bind the episode case")
        lock_version = lock.get("version", 1)
        if lock_version not in (1, 2):
            raise ValueError("Unsupported original lock version")
        locked_model = next(model for model in models if model["name"] == original.parent.name)
        from fsop.eval.controller import controller_contract
        contract = locked_model.get("controller_contract")
        if lock_version == 2:
            if (not isinstance(contract, dict) or contract.get("mode") not in ("pp", "op_tangent_v1")
                or contract != controller_contract(contract["mode"])
                or metadata.get("controller_contract") != contract or summary.get("controller_contract") != contract):
                raise ValueError("Original lock controller contract differs from episode")
        elif (metadata.get("controller_contract", controller_contract("pp")) != controller_contract("pp")
              or summary.get("controller_contract", controller_contract("pp")) != controller_contract("pp")):
            raise ValueError("Version 1 lock cannot bind a non-legacy controller")
        row["locked_plan_version"] = lock_version
        row["locked_controller_contract"] = contract
        record = manifest["episodes"][key]
        if record.get("status") != "complete" or record.get("receipt", {}).get("completion_sha256") != digest(
            folder / "evidence/completion.json"
        ):
            raise ValueError("Original evaluation manifest does not bind this episode")
        for field in ("source_sha256", "graph_sha256"):
            if field not in lock or lock[field] != conditions[field]:
                raise ValueError(f"Original lock differs from episode {field}")
        for field in ("coordinator_sha256", "timeout_seconds"):
            if field not in lock or lock[field] is None:
                raise ValueError(f"Original lock missing {field}")
            conditions[field] = lock[field]
        row["comparison_conditions"] = conditions
    except (OSError, ValueError, KeyError, TypeError, AttributeError, BadZipFile) as exc:
        row["comparison_error"] = str(exc)
    row["audit"], audit = audit_status(folder, receipt)
    if row.get("locked_plan_version") == 2 and row["audit"] == "passed":
        row["audited_controller_contract"] = audit.get("controller_contract")
        if row["audited_controller_contract"] != row.get("locked_controller_contract"):
            row["audit"] = "controller contract mismatch"
    row["latency"] = latency(folder, audit, row["audit"])
    for name in ("debug", "research"):
        row["videos"][name] = video_status(folder, name, receipt)
    return row


COMMAND_MODES = ("recorded", "zero", "swapped")
GROUP_FIELDS = ("model", "cohort", "split", "discipline", "variant", "dr", "command_mode")


def planned_command_mode(case):
    """Only infer missing planned labels from the frozen case identifier."""
    frozen = case.get("frozen_case")
    if isinstance(frozen, str):
        for mode in COMMAND_MODES:
            if frozen.endswith("-" + mode):
                return mode
    return "unknown"


def plan_matrix(plan_path):
    plan = read_json(plan_path)
    if plan.get("version") not in (1, 2):
        raise ValueError("Unsupported evaluation plan version")
    cases = []
    for spec in plan["cases"]:
        spec = {"path": spec} if isinstance(spec, str) else spec
        path = Path(spec["path"])
        if not path.is_absolute():
            path = plan_path.parent / path
        case = {
            "dr": "heavy",
            "variant": "visual_stress_v1",
            "track_seed": None,
            "frozen_case": None,
            "lateral_start": 0.0,
            "yaw_start": 0.0,
            **(spec.get("case") or read_json(path)),
        }
        name = spec.get("name") or case["id"]
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid case output name")
        cases.append((name, case))
    models = [dict(model, _plan_version=plan["version"]) for model in plan["models"]]
    if plan["version"] == 2:
        from fsop.eval.controller import controller_contract
        for model in models:
            contract = model.get("controller_contract")
            if (not isinstance(contract, dict) or contract.get("mode") not in ("pp", "op_tangent_v1")
                or contract != controller_contract(contract["mode"])):
                raise ValueError("Version 2 plan requires an exact controller contract for each model")
    keys = [(model["name"], name) for model in models for name, _ in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate expected model/case output")
    if any(
        not isinstance(model["name"], str) or Path(model["name"]).name != model["name"] or model["name"] in (".", "..")
        for model in models
    ):
        raise ValueError("Invalid model output name")
    return [(model, name, case) for model in models for name, case in cases]


def merge_plans(paths):
    merged = {}
    model_hashes = {}
    identities = {}
    for path in paths:
        for model, name, case in plan_matrix(path):
            key = (model["name"], name)
            digest_value = (model.get("sha256"), model.get("controller_contract"), model["_plan_version"])
            if model["name"] in model_hashes and model_hashes[model["name"]] != digest_value:
                raise ValueError(f"Conflicting checkpoint identity for model {model['name']}")
            model_hashes[model["name"]] = digest_value
            identity = (model["name"], case["id"])
            if identity in identities and identities[identity] != name:
                raise ValueError(f"Duplicate case identity under different output names: {identity}")
            identities[identity] = name
            if key in merged:
                previous_model, _, previous_case = merged[key]
                previous_identity = (
                    previous_model.get("sha256"),
                    previous_model.get("controller_contract"),
                    previous_model["_plan_version"],
                )
                if previous_case != case or previous_identity != digest_value:
                    raise ValueError(f"Contradictory duplicate planned episode: {key}")
            else:
                merged[key] = (model, name, case)
    return list(merged.values())


def load_coverage(path):
    if path is None:
        return {"paired": set(), "legacy": set()}
    data = read_json(path)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation coverage schema")
    paired = data["paired_case_ids"]
    legacy = data["candidate_only_legacy_case_ids"]
    if (
        not isinstance(paired, list)
        or not isinstance(legacy, list)
        or any(not isinstance(x, str) for x in [*paired, *legacy])
        or len(set(paired)) != len(paired)
        or len(set(legacy)) != len(legacy)
        or set(paired) & set(legacy)
    ):
        raise ValueError("Invalid or overlapping coverage case IDs")
    return {"paired": set(paired), "legacy": set(legacy)}


def cohort_for(row, coverage):
    ident = row["case_id"]
    split = row.get("split")
    if ident in coverage["paired"]:
        return {"test": "paired_fresh", "regression": "paired_exposed"}.get(split, "unknown")
    if ident in coverage["legacy"] and row.get("model") == "candidate" and split == "regression":
        return "candidate_legacy"
    return "unknown"


def paired_comparisons(episodes, coverage, has_plan, scope="same_controller"):
    if not has_plan:
        return {
            "coverage_pairs": len(coverage["paired"]),
            "required_pairs": 0,
            "measured_pairs": 0,
            "unplanned_case_ids": sorted(coverage["paired"]),
            "unmeasured_case_ids": [],
            "comparison_errors": {},
            "cases": [],
        }
    by_key = {(row["model"], row["case_id"]): row for row in episodes if row.get("expected")}
    cases = []
    unplanned = []
    unmeasured = []
    comparison_errors = {}
    required = 0
    for ident in sorted(coverage["paired"]):
        base = by_key.get(("baseline", ident))
        candidate = by_key.get(("candidate", ident))
        if base is None or candidate is None:
            unplanned.append(ident)
            continue
        required += 1
        if not all(
            r["seal"] == "verified" and r.get("pilot") is False and r.get("audit") == "passed"
            for r in (base, candidate)
        ):
            unmeasured.append(ident)
            continue
        if (
            base.get("case") != candidate.get("case")
            or base.get("command_mode") != candidate.get("command_mode")
            or base.get("cohort") not in ("paired_fresh", "paired_exposed")
            or base.get("cohort") != candidate.get("cohort")
        ):
            unmeasured.append(ident)
            continue

        try:
            left, right = base["comparison_conditions"], candidate["comparison_conditions"]
            if scope == "controller_model_package":
                if any(r.get("plan_version") != 2 or r.get("locked_plan_version") != 2 for r in (base, candidate)):
                    raise ValueError("Package comparison requires frozen version 2 plans for both models")
                for r, conditions in ((base, left), (candidate, right)):
                    contract = r.get("planned_controller_contract")
                    if (not contract or r.get("locked_controller_contract") != contract
                        or conditions.get("controller_contract") != contract
                        or r.get("audited_controller_contract") != contract):
                        raise ValueError("Audited controller differs from frozen model contract")
                    expected_schema = {"pp": 1, "op_tangent_v1": 2}.get(contract.get("mode"))
                    if conditions.get("schema_version") != expected_schema:
                        raise ValueError("Evidence schema does not match frozen controller mode")
                # Schema is part of the versioned controller representation; every other
                # shared condition remains under the normal strict comparison.
                left = {**left, "controller_contract": right["controller_contract"],
                        "schema_version": right["schema_version"]}
            compare_conditions(left, right)
            for field in ("coordinator_sha256", "timeout_seconds"):
                if left[field] != right[field]:
                    raise ValueError(f"Noncomparable episode condition: {field}")
        except (KeyError, ValueError) as exc:
            comparison_errors[ident] = base.get("comparison_error") or candidate.get("comparison_error") or str(exc)
            unmeasured.append(ident)
            continue

        def delta(field, base=base, candidate=candidate):
            left, right = base.get(field), candidate.get(field)
            return right - left if finite(left) and finite(right) else None

        cases.append(
            {
                "case_id": ident,
                "scope": scope,
                "baseline_controller_contract": base.get("planned_controller_contract"),
                "candidate_controller_contract": candidate.get("planned_controller_contract"),
                "baseline_evidence_schema_version": base["comparison_conditions"].get("schema_version"),
                "candidate_evidence_schema_version": candidate["comparison_conditions"].get("schema_version"),
                "cohort": base["cohort"],
                "split": base["split"],
                "discipline": base["discipline"],
                "variant": base["variant"],
                "dr": base["dr"],
                "command_mode": base["command_mode"],
                "baseline_result": base["result"],
                "candidate_result": candidate["result"],
                "finished_delta": int(candidate["result"] == "finished") - int(base["result"] == "finished"),
                "physical_contacts_delta": delta("physical_contacts"),
                "off_course_delta": delta("off_course"),
                "progress_delta": delta("progress"),
            }
        )
    return {
        "coverage_pairs": len(coverage["paired"]),
        "required_pairs": required,
        "measured_pairs": len(cases),
        "unplanned_case_ids": unplanned,
        "unmeasured_case_ids": unmeasured,
        "comparison_errors": comparison_errors,
        "cases": cases,
    }


def wilson_interval(successes, count):
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def aggregate(episodes):
    groups = defaultdict(list)
    for row in episodes:
        if row["seal"] != "verified" or row.get("pilot") is not False or row.get("audit") != "passed":
            continue
        key = tuple(row.get(k, "unknown") for k in GROUP_FIELDS)
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        contacts = [r["physical_contacts"] for r in rows if finite(r.get("physical_contacts"))]
        results = [r.get("result") for r in rows]
        item = dict(zip(GROUP_FIELDS, key, strict=True))
        finished = sum(x == "finished" for x in results)
        item.update(
            episodes=len(rows),
            finished=finished,
            finished_fraction=finished / len(rows),
            finished_wilson_95=wilson_interval(finished, len(rows)),
            results_missing=sum(x is None for x in results),
            physical_contacts_sum=sum(contacts) if contacts else None,
            physical_contacts_mean=float(np.mean(contacts)) if contacts else None,
            physical_contacts_measured=len(contacts),
            audit_passed=sum(r["audit"] == "passed" for r in rows),
        )
        result.append(item)
    return result


def final_gate(episodes, expected_keys, has_plan, required):
    if not has_plan:
        return {"required": required, "status": "not evaluated", "planned": None, "failures": []}
    failures = []
    if not expected_keys:
        failures.append("Plan contains no expected episodes")
    for row in episodes:
        ident = f"{row['model']}/{row['case_id']}"
        if not row.get("expected"):
            failures.append(f"{ident}: unexpected episode directory")
            continue
        if row["seal"] != "verified":
            failures.append(f"{ident}: {row['seal']} ({row.get('seal_error', 'no detail')})")
            continue
        if row.get("pilot") is not False:
            failures.append(f"{ident}: pilot evidence")
        if row.get("audit") != "passed":
            failures.append(f"{ident}: artifact audit {row.get('audit', 'missing')}")
        for name in ("debug", "research"):
            status = row.get("videos", {}).get(name, {}).get("status", "missing")
            if status != "receipt hash/count matched":
                failures.append(f"{ident}: {name} video {status}")
    included = sum(
        row.get("expected") and row["seal"] == "verified" and row.get("pilot") is False and row.get("audit") == "passed"
        for row in episodes
    )
    return {
        "required": required,
        "status": "failed" if failures else "passed",
        "planned": len(expected_keys),
        "audited_scored": included,
        "failures": failures,
    }


def read_training(root):
    warnings = []
    try:
        run = read_json(root / "run.json")
    except (OSError, ValueError) as exc:
        run = None
        warnings.append(f"run.json unavailable: {exc}")
    history = []
    try:
        with (root / "history.jsonl").open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not finite(row.get("step")):
                        raise ValueError("missing numeric step")
                    history.append(row)
                except (ValueError, TypeError) as exc:
                    warnings.append(f"history line {number} skipped: {exc}")
    except OSError as exc:
        warnings.append(f"history.jsonl unavailable: {exc}")
    baseline = None
    baseline_path = root / "baseline_validation.json"
    if baseline_path.exists():
        try:
            baseline = read_json(baseline_path)
            if (
                not isinstance(baseline, dict)
                or baseline.get("step") != 0
                or not isinstance(baseline.get("aggregate"), dict)
                or not finite(baseline["aggregate"].get("loss"))
            ):
                raise ValueError("expected finite step-0 aggregate validation loss")
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f"baseline_validation.json unavailable: {exc}")
            baseline = None
    return run, history, baseline, warnings


VALIDATION_COMPONENTS = (
    ("loss", "Total loss"),
    ("plan_nll", "Plan NLL"),
    ("steer", "Steering loss"),
    ("longitudinal", "Longitudinal loss"),
)
TELEMETRY_METRICS = (
    ("gpu_pct", "Whole GPU utilization (%)"),
    ("power_w", "GPU power (W)"),
    ("host_ram_gib", "Host unified RAM used (GiB)"),
    ("cpu_pct", "Host CPU utilization (%)"),
)


def validation_series(history, baseline):
    rows = []
    if baseline is not None:
        values = baseline["aggregate"]
        rows.append(
            {
                "step": 0,
                **{key: values.get(key) if finite(values.get(key)) else None for key, _ in VALIDATION_COMPONENTS},
            }
        )
    for row in history:
        validation = row.get("validation")
        aggregate = validation.get("aggregate") if isinstance(validation, dict) else None
        aggregate = aggregate if isinstance(aggregate, dict) else {}
        values = {key: aggregate.get(key) for key, _ in VALIDATION_COMPONENTS}
        if values["loss"] is None:
            values["loss"] = row.get("val_loss")
        if any(finite(value) for value in values.values()):
            rows.append(
                {"step": row["step"], **{key: value if finite(value) else None for key, value in values.items()}}
            )
    return rows


def read_training_telemetry(path):
    if path is None:
        return {"status": "not supplied", "path": None}
    result = {"status": "unavailable", "path": str(path.resolve())}
    try:
        raw = path.read_bytes()
        result["sha256"] = hashlib.sha256(raw).hexdigest()
        receipt = json.loads(raw)
        if not isinstance(receipt, dict):
            raise ValueError("expected a receipt object")
        command = receipt.get("command")
        code = receipt.get("returncode")
        samples = receipt.get("samples")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(x, str) for x in command)
            or not isinstance(code, int)
            or isinstance(code, bool)
            or not finite(receipt.get("wall_seconds"))
            or receipt["wall_seconds"] < 0
            or not isinstance(samples, list)
            or not samples
        ):
            raise ValueError("missing command, return code, wall time or samples")
        inventories = [receipt.get("compute_processes_before"), receipt.get("compute_processes_after")]
        last_wall = -1
        for sample in samples:
            if (
                not isinstance(sample, dict)
                or not finite(sample.get("wall_s"))
                or not last_wall <= sample["wall_s"] <= receipt["wall_seconds"]
            ):
                raise ValueError("invalid sample wall time")
            last_wall = sample["wall_s"]
            if any(not finite(sample.get(key)) or sample[key] < 0 for key, _ in TELEMETRY_METRICS):
                raise ValueError("invalid or missing telemetry metric")
            inventories.append(sample.get("compute_processes"))
        if any(
            not isinstance(items, list)
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("pid"), int)
                or isinstance(item.get("pid"), bool)
                or not isinstance(item.get("name"), str)
                for item in items
            )
            for items in inventories
        ):
            raise ValueError("invalid compute process inventory")
        processes = sorted({(item["pid"], item["name"]) for items in inventories for item in items})
        result.update(
            status="completed" if code == 0 else "failed",
            command=command,
            returncode=code,
            wall_seconds=receipt["wall_seconds"],
            sample_count=len(samples),
            median_gpu_pct=float(np.median([row["gpu_pct"] for row in samples])),
            median_power_w=float(np.median([row["power_w"] for row in samples])),
            peak_host_ram_gib=max(row["host_ram_gib"] for row in samples),
            compute_processes=[{"pid": pid, "name": name} for pid, name in processes],
            scope="Whole GPU and host; host RAM is unified system memory, not dedicated VRAM",
            samples=[{"wall_s": row["wall_s"], **{key: row[key] for key, _ in TELEMETRY_METRICS}} for row in samples],
        )
    except (OSError, ValueError, TypeError, UnicodeDecodeError) as exc:
        result["status"] = "unavailable" if isinstance(exc, OSError) else "malformed"
        result["error"] = str(exc)
    return result


def validation_graph(rows, out):
    filename = "training-validation.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, (key, label) in zip(axes.flat, VALIDATION_COMPONENTS, strict=True):
        points = [(row["step"], row[key]) for row in rows if finite(row.get(key))]
        if points:
            ax.plot(*zip(*points, strict=True), marker="o", markersize=3, linewidth=1.2)
        else:
            ax.text(0.5, 0.5, "No validation data", transform=ax.transAxes, ha="center", va="center")
        ax.set(title=label, xlabel="Step", ylabel="Validation loss")
        ax.grid(alpha=0.2)
    fig.savefig(out / filename, dpi=130)
    plt.close(fig)
    return filename


def telemetry_graph(telemetry, out):
    filename = "training-hardware.png"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, (key, label) in zip(axes.flat, TELEMETRY_METRICS, strict=True):
        samples = telemetry.get("samples", [])
        if samples:
            ax.plot([row["wall_s"] for row in samples], [row[key] for row in samples], linewidth=1.2)
        else:
            ax.text(0.5, 0.5, f"Telemetry {telemetry['status']}", transform=ax.transAxes, ha="center", va="center")
        ax.set(title=label, xlabel="Wall time (s)")
        ax.grid(alpha=0.2)
    fig.savefig(out / filename, dpi=130)
    plt.close(fig)
    return filename


def training_graphs(history, out):
    graphs = []
    specs = [
        (
            "training-losses.png",
            "Training and validation losses",
            [
                ("loss", "train total"),
                ("val_loss", "validation"),
                ("plan_nll", "train plan NLL"),
                ("steer", "train steer"),
                ("longitudinal", "train longitudinal"),
                ("l2sp", "train L2-SP"),
            ],
            "Loss",
        ),
        (
            "training-gradients.png",
            "Gradient norms before clipping",
            [("grad_norms.vision", "vision"), ("grad_norms.policy", "policy")],
            "Gradient norm",
        ),
        ("training-step-time.png", "Training step time", [("step_seconds", "step")], "Seconds"),
    ]
    for filename, title, series, ylabel in specs:
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        found = False
        for key, label in series:

            def value(r, key=key):
                if "." in key:
                    group, field = key.split(".", 1)
                    return r.get(group, {}).get(field) if isinstance(r.get(group), dict) else None
                return r.get(key)

            points = [(r["step"], value(r)) for r in history if finite(r.get("step")) and finite(value(r))]
            if points:
                found = True
                ax.plot(*zip(*points, strict=True), label=label, linewidth=1.2)
        ax.set(title=title, xlabel="Step", ylabel=ylabel)
        ax.grid(alpha=0.2)
        if found:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No results available", transform=ax.transAxes, ha="center", va="center")
        fig.savefig(out / filename, dpi=130)
        plt.close(fig)
        graphs.append(filename)
    return graphs


def evaluation_graphs(groups, out):
    filename = "evaluation.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, max(5, 2 + 0.45 * len(groups))), layout="constrained")
    if not groups:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No audited sealed non-pilot evaluation results available",
                transform=ax.transAxes,
                ha="center",
            )
            ax.axis("off")
    else:
        labels = [" / ".join(str(g[k]) for k in GROUP_FIELDS) for g in groups]
        x = np.arange(len(groups))
        complete = [g["finished"] / g["episodes"] * 100 for g in groups]
        contacts = [g["physical_contacts_mean"] if g["physical_contacts_mean"] is not None else np.nan for g in groups]
        axes[0].barh(x, complete)
        lower = [g["finished_wilson_95"][0] * 100 for g in groups]
        upper = [g["finished_wilson_95"][1] * 100 for g in groups]
        axes[0].errorbar(
            complete,
            x,
            xerr=[np.array(complete) - lower, upper - np.array(complete)],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        axes[0].set(xlabel="Finished episodes (%)", xlim=(0, 100))
        axes[1].barh(x, contacts)
        axes[1].set(xlabel="Mean unique physical cone contacts / measured episode")
        for ax in axes:
            ax.set_yticks(x, labels=labels, fontsize=8)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2)
    fig.savefig(out / filename, dpi=130)
    plt.close(fig)
    return filename


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}" if math.isfinite(value) else "—"
    if isinstance(value, list):
        return ", ".join(fmt(x) for x in value) if value else "—"
    return str(value)


def link(path, out, text):
    relative = os.path.relpath(path, out.resolve())
    return f'<a href="{quote(relative, safe="/")}">{html.escape(text)}</a>'


def render_html(report, out, graphs, eval_graph):
    def esc(x):
        return html.escape(fmt(x))

    body = [
        '<!doctype html><html><head><meta charset="utf-8"><title>Native Lab training report</title>',
        "<style>body{font:15px system-ui;margin:2em;max-width:1800px} "
        "table{border-collapse:collapse;width:100%} td,th{border:1px solid #ccc;padding:.4em;"
        "text-align:left;vertical-align:top} img{max-width:100%;height:auto} .wrap{overflow-x:auto} "
        "small{color:#555}</style></head><body>",
        "<h1>Native Lab training and evaluation</h1>",
        "<p>Only completion-sealed non-pilot episodes with a passing bound artifact audit are aggregated; "
        "completion percentages use the number of audited episodes, so missing expected cases are shown "
        "in the table and excluded from "
        "the denominator. Cohort, split, model, discipline, appearance, domain randomization and command mode "
        "are separate. "
        "An independent audit is shown separately. A valid seal binds the summary and evidence inventory; "
        "this report does not rehash every NPZ or re-run physics.</p>",
        "<h2>Training</h2>",
    ]
    gate = report["final_gate"]
    if gate["status"] == "not evaluated":
        body.append("<p>Final gate not evaluated: no evaluation plan supplied.</p>")
    else:
        body.append(
            f"<p>Final gate: {esc(gate['status'])}; planned {gate['planned']}, "
            f"audited scored {gate['audited_scored']}, gate problems {len(gate['failures'])}. "
            "A timeout or other valid scored failure is an outcome, not a gate failure.</p>"
        )
        if gate["failures"]:
            body.append("<ul>")
            body.extend(f"<li>{esc(failure)}</li>" for failure in gate["failures"])
            body.append("</ul>")
    if report["training"]["run"] is None and not report["training"]["history_rows"]:
        body.append("<p>No training results available.</p>")
    else:
        metadata_status = "available" if report["training"]["run"] is not None else "missing"
        body.append(f"<p>Recorded steps: {report['training']['history_rows']}; run metadata: {metadata_status}.</p>")
    baseline = report["training"]["baseline_validation"]
    if baseline is not None:
        body.append(
            f"<p>Step-0 baseline validation loss: {esc(baseline['aggregate']['loss'])} "
            f"({esc(baseline.get('aggregation'))}, {esc(baseline.get('precision'))}; "
            f"{esc(baseline.get('count'))} anchors).</p>"
        )
    for warning in report["training"]["warnings"]:
        body.append(f"<p>{esc(warning)}</p>")
    body.extend(f'<p><img src="{esc(name)}" alt="{esc(name)}"></p>' for name in graphs[:-1])
    telemetry = report["training"]["telemetry"]
    body.append(f"<h3>Training hardware telemetry: {esc(telemetry['status'])}</h3>")
    body.append(
        "<p>Whole GPU and host measurements. Host RAM is unified system memory. "
        "GPU utilization is not SM occupancy; process inventory may include the training job.</p>"
    )
    if telemetry.get("error"):
        body.append(f"<p>{esc(telemetry['error'])}</p>")
    if "command" in telemetry:
        body.append(
            f"<p>Job wall seconds: {esc(telemetry['wall_seconds'])}; "
            f"median GPU utilization: {esc(telemetry['median_gpu_pct'])}%; "
            f"median GPU power: {esc(telemetry['median_power_w'])} W; "
            f"peak sampled host RAM: {esc(telemetry['peak_host_ram_gib'])} GiB; "
            f"return code: {esc(telemetry['returncode'])}.</p>"
        )
        body.append(
            f"<p>Observed GPU compute processes, including possible competing processes: "
            f"{esc(json.dumps(telemetry['compute_processes']))}</p>"
        )
        body.append(
            f"<p>Receipt SHA-256: {esc(telemetry['sha256'])}; command: {esc(json.dumps(telemetry['command']))}</p>"
        )
    body.append(f'<p><img src="{esc(graphs[-1])}" alt="{esc(graphs[-1])}"></p>')
    body += [
        "<h2>Evaluation</h2>",
        f'<p><img src="{esc(eval_graph)}" alt="Evaluation results by group"></p>',
        "<p>Completion error bars and intervals are 95% Wilson binomial intervals over the audited episodes "
        "in each group; fixed or selectively chosen episodes do not establish a population success rate.</p>",
        '<div class="wrap"><table><thead><tr><th>Model / cohort / split / discipline / variant / DR / command</th>'
        "<th>Finished / episodes</th><th>95% Wilson interval</th></tr></thead><tbody>",
    ]
    if not report["groups"]:
        body.append('<tr><td colspan="3">No audited sealed non-pilot results available.</td></tr>')
    for group in report["groups"]:
        label = " / ".join(str(group[k]) for k in GROUP_FIELDS)
        lo, hi = group["finished_wilson_95"]
        body.append(
            f"<tr><td>{esc(label)}</td><td>{group['finished']} / {group['episodes']}</td>"
            f"<td>{lo:.1%}–{hi:.1%}</td></tr>"
        )
    body += [
        "</tbody></table></div>",
        "<p>Latency is synchronized policy-call wall time including diagnostic capture, when recorded; "
        "it is not an isolated GPU kernel measurement. Audit-bound means the recorded frame timing file "
        "matches the independent audit hash.</p>",
        '<div class="wrap"><table><thead><tr>'
        + "".join(
            f"<th>{x}</th>"
            for x in (
                "Model",
                "Cohort",
                "Split",
                "Case",
                "Discipline",
                "Variant / DR",
                "Command mode",
                "Seal",
                "Audit",
                "Pilot",
                "Result / pass",
                "Contacts",
                "OC",
                "Progress",
                "Laps (s)",
                "Wall (s) / RTF",
                "Policy latency mean / p50 / p95 ms (n; status)",
                "Videos",
            )
        )
        + "</tr></thead><tbody>",
    ]
    if not report["episodes"]:
        body.append('<tr><td colspan="18">No evaluation results available.</td></tr>')
    for r in report["episodes"]:
        lat = r["latency"]
        values = [
            r.get("model"),
            r.get("cohort", "unknown"),
            r.get("split"),
            r.get("case_id"),
            r.get("discipline"),
            f"{fmt(r.get('variant'))} / {fmt(r.get('dr'))}",
            r.get("command_mode", "unknown"),
            r["seal"] + (": " + r["seal_error"] if "seal_error" in r else ""),
            r["audit"],
            r.get("pilot"),
            f"{fmt(r.get('result'))} / {fmt(r.get('passed'))}",
            r.get("physical_contacts"),
            r.get("off_course"),
            r.get("progress"),
            r.get("laps"),
            f"{fmt(r.get('wall_seconds'))} / {fmt(r.get('rtf'))}",
            f"{fmt(lat.get('mean_ms'))} / {fmt(lat.get('p50_ms'))} / {fmt(lat.get('p95_ms'))} "
            f"({fmt(lat.get('count'))}; {lat['status']})",
        ]
        video_links = []
        for name, v in r["videos"].items():
            video_links.append(
                (link(v["path"], out, name) if v["path"] else html.escape(name)) + " (" + esc(v["status"]) + ")"
            )
        body.append(
            "<tr>"
            + "".join("<td>" + esc(v) + "</td>" for v in values)
            + "<td>"
            + "<br>".join(video_links)
            + "</td></tr>"
        )
    pairs = report["paired_comparisons"]
    body += [
        "</tbody></table></div>",
        "<h2>Paired baseline to candidate</h2>",
        "<p>Frozen controller contracts: "
        f"{esc(json.dumps(report.get('planned_controller_contracts', {}), sort_keys=True))}</p>",
        f"<p>Comparison scope: {esc(report.get('comparison_scope', 'same_controller'))}. "
        + ("Controller/model package comparison; outcomes cannot be attributed to retraining alone."
           if report.get('comparison_scope') == 'controller_model_package' else "") + "</p>",
        f"<p>Measured {pairs['measured_pairs']} of {pairs['required_pairs']} jointly planned paired cases "
        f"({pairs['coverage_pairs']} in coverage); {len(pairs['unplanned_case_ids'])} coverage cases are not "
        f"jointly planned and {len(pairs['unmeasured_case_ids'])} planned pairs could not be compared. "
        "Deltas are candidate minus baseline. Cases must be covered, planned for both models, and audited.</p>",
        '<div class="wrap"><table><thead><tr><th>Case</th><th>Cohort</th><th>Baseline result</th>'
        "<th>Candidate result</th><th>Finished Δ</th><th>Contacts Δ</th><th>OC Δ</th>"
        "<th>Progress Δ</th></tr></thead><tbody>",
    ]
    if not pairs["cases"]:
        body.append('<tr><td colspan="8">No audited paired results available.</td></tr>')
    for pair in pairs["cases"]:
        fields = (
            "case_id",
            "cohort",
            "baseline_result",
            "candidate_result",
            "finished_delta",
            "physical_contacts_delta",
            "off_course_delta",
            "progress_delta",
        )
        body.append("<tr>" + "".join(f"<td>{esc(pair[k])}</td>" for k in fields) + "</tr>")
    body += [
        "</tbody></table></div>",
        '<p>See <a href="aggregates.json">aggregates.json</a> for the group totals '
        "and full episode records.</p></body></html>",
    ]
    (out / "index.html").write_text("\n".join(body) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Coordinator plan or plan.lock.json; includes expected missing cases")
    parser.add_argument(
        "--additional-plan", action="append", type=Path, default=[], help="Additional plan or lock; may be repeated"
    )
    parser.add_argument("--coverage", type=Path, help="Evaluation coverage membership JSON")
    parser.add_argument("--comparison-scope", choices=("same_controller", "controller_model_package"),
                        default="same_controller", help="Explicitly compare frozen version 2 controller/model packages")
    parser.add_argument("--training-telemetry", type=Path, help="Completed measure_native.py receipt")
    parser.add_argument(
        "--require-complete", action="store_true", help="Fail unless every planned episode passes the final gate"
    )
    args = parser.parse_args(argv)
    if args.additional_plan and not args.plan:
        parser.error("--additional-plan requires --plan")
    if args.require_complete and not args.plan:
        parser.error("--require-complete requires --plan")
    args.out.mkdir(parents=True, exist_ok=True)
    run, history, baseline, warnings = read_training(args.training)
    telemetry = read_training_telemetry(args.training_telemetry)
    validation = validation_series(history, baseline)
    coverage = load_coverage(args.coverage)
    plan_paths = [args.plan, *args.additional_plan] if args.plan else []
    if args.comparison_scope == "controller_model_package" and (
        not plan_paths or any(read_json(path).get("version") != 2 for path in plan_paths)
    ):
        raise ValueError("Package comparison requires explicit version 2 plans")
    episodes = []
    expected_keys = set()
    if args.plan:
        for model, name, case in merge_plans(plan_paths):
            key = (model["name"], name)
            expected_keys.add(key)
            row = load_episode(args.evaluation / model["name"] / name, model["name"])
            row["expected"] = True
            row["plan_version"] = model["_plan_version"]
            row["planned_controller_contract"] = model.get("controller_contract")
            if row["seal"] == "verified":
                expected_identity = row["case"] == case and row.get("checkpoint_sha256") == model.get("sha256")
                if model["_plan_version"] == 2 and (
                    row.get("locked_plan_version") != 2
                    or row.get("locked_controller_contract") != model["controller_contract"]
                    or row.get("audited_controller_contract") != model["controller_contract"]
                ):
                    expected_identity = False
                planned_mode = planned_command_mode(case)
                if planned_mode != "unknown" and row["command_mode"] != planned_mode:
                    expected_identity = False
                if not expected_identity:
                    row["seal"] = "plan identity mismatch"
                    row["seal_error"] = "case, checkpoint or command mode differs from plan"
            else:
                row.update(
                    case_id=case.get("id", name),
                    split=case.get("split", "unknown"),
                    discipline=case.get("discipline", "unknown"),
                    variant=case.get("variant", "unknown"),
                    dr=case.get("dr", "unknown"),
                    command_mode=planned_command_mode(case),
                )
            episodes.append(row)
    if args.evaluation.is_dir():
        for model in sorted(p for p in args.evaluation.iterdir() if p.is_dir() and p.name not in {"cases", "logs"}):
            for folder in sorted(p for p in model.iterdir() if p.is_dir()):
                if (model.name, folder.name) not in expected_keys:
                    row = load_episode(folder, model.name)
                    row["expected"] = not bool(args.plan)
                    if args.plan and row["seal"] == "verified":
                        row["seal"] = "not in plan"
                        row["seal_error"] = "unexpected episode excluded from aggregates"
                    episodes.append(row)
    for row in episodes:
        row["cohort"] = cohort_for(row, coverage)
    gate = final_gate(episodes, expected_keys, bool(args.plan), args.require_complete)
    if args.coverage and args.plan:
        planned_ids = {(row["model"], row["case_id"]) for row in episodes if row.get("expected")}
        for model, ids in (("baseline", coverage["paired"]), ("candidate", coverage["paired"] | coverage["legacy"])):
            for ident in sorted(ids):
                if (model, ident) not in planned_ids:
                    gate["failures"].append(f"{model}/{ident}: required coverage case not in supplied plans")
        if gate["failures"]:
            gate["status"] = "failed"
    pairs = paired_comparisons(episodes, coverage, bool(args.plan), args.comparison_scope)
    for ident in pairs["unmeasured_case_ids"]:
        detail = pairs["comparison_errors"].get(ident, "missing or unaudited evidence")
        gate["failures"].append(f"{ident}: paired episodes are incomplete or noncomparable ({detail})")
    if gate["failures"]:
        gate["status"] = "failed"
    report = {
        "training": {
            "path": str(args.training.resolve()),
            "run": run,
            "history_rows": len(history),
            "baseline_validation": baseline,
            "validation_series": validation,
            "telemetry": telemetry,
            "warnings": warnings,
        },
        "final_gate": gate,
        "comparison_scope": args.comparison_scope,
        "planned_controller_contracts": {
            model["name"]: model.get("controller_contract")
            for model, _, _ in merge_plans(plan_paths)
        },
        "evaluation_path": str(args.evaluation.resolve()),
        "plan_path": str(args.plan.resolve()) if args.plan else None,
        "plan_paths": [str(p.resolve()) for p in plan_paths],
        "coverage_path": str(args.coverage.resolve()) if args.coverage else None,
        "episodes": episodes,
        "groups": aggregate(episodes),
        "paired_comparisons": pairs,
        "limitations": [
            "Summary seal and frame membership checked; frame NPZ content hashes not rechecked.",
            "Audit and videos reported separately from summary seal.",
            "Policy latency is synchronized wall time with capture hooks, not GPU kernel time.",
        ],
    }
    plot_history = history
    if baseline is not None:
        plot_history = [{"step": 0, "val_loss": baseline["aggregate"]["loss"]}, *history]
    graphs = training_graphs(plot_history, args.out)
    graphs.append(validation_graph(validation, args.out))
    graphs.append(telemetry_graph(telemetry, args.out))
    eval_graph = evaluation_graphs(report["groups"], args.out)
    (args.out / "aggregates.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    render_html(report, args.out, graphs, eval_graph)
    print(f"Report: {args.out / 'index.html'}; {len(episodes)} episode rows; {len(report['groups'])} groups")
    if args.require_complete and gate["status"] != "passed":
        raise SystemExit(f"Final gate failed: {len(gate['failures'])} problem(s); see {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
