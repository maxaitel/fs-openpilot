import hashlib
import json
from zipfile import ZipFile

import pytest

from scripts.lab_comparison import FIELDS
from scripts.report_lab_training import main, wilson_interval


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(root, model, case, result="finished", command_mode="recorded", pilot=False):
    folder = root / model / case["id"]
    evidence = folder / "evidence"
    evidence.mkdir(parents=True)
    write(folder / "export.json", {"dependencies": [], "sha256": {}})
    for name in ("realization.json", "appearance.json"):
        (folder / name).write_text("{}")
    with ZipFile(folder / "source.zip", "w"):
        pass
    metadata = {key: "same" for key in FIELDS}
    metadata.update(
        {
            "realization_sha256": sha(folder / "realization.json"),
            "appearance_sha256": sha(folder / "appearance.json"),
            "source_archive_sha256": sha(folder / "source.zip"),
        }
    )
    metadata.update(
        {
            "case": case,
            "pilot_only": pilot,
            "checkpoint_sha256": "abc",
            "state_sha256": "def",
            "export_sha256": sha(folder / "export.json"),
            "source_sha256": {},
            "graph_sha256": {},
        }
    )
    if command_mode is not None:
        metadata["command_mode"] = command_mode
    write(evidence / "metadata.json", metadata)
    (evidence / "frame_000000.npz").write_bytes(b"fixture frame")
    (evidence / "manifest.jsonl").write_text(
        json.dumps(
            {
                "frame": 0,
                "file": "frame_000000.npz",
                "sha256": sha(evidence / "frame_000000.npz"),
                "metadata_sha256": sha(evidence / "metadata.json"),
            }
        )
        + "\n"
    )
    write(
        folder / "summary.json",
        {
            "case": case,
            "pilot_only": pilot,
            "recorded_frames": 1,
            "result": result,
            "pass": result == "finished",
            "checkpoint_sha256": "abc",
            "state_sha256": "def",
            "physical_cone_hits": 2,
            "oc": 1,
            "progress": 0.8,
            "laps": [],
            "wall_seconds": 3,
            "real_time_factor": 0.1,
        },
    )
    write(
        evidence / "completion.json",
        {
            "version": 1,
            "frame_count": 1,
            "metadata_sha256": sha(evidence / "metadata.json"),
            "manifest_sha256": sha(evidence / "manifest.jsonl"),
            "summary_path": "../summary.json",
            "summary_sha256": sha(folder / "summary.json"),
        },
    )
    write(folder / "frames.json", [{"policy_with_capture_s": 0.01}, {"policy_with_capture_s": 0.02}])
    write(folder / "telemetry.json", [])
    write(
        folder / "lab_artifact_audit.json",
        {
            "passed": True,
            "frame_count": 1,
            "telemetry_sha256": sha(folder / "telemetry.json"),
            "completion_sha256": sha(evidence / "completion.json"),
            "frames_sha256": sha(folder / "frames.json"),
        },
    )
    (folder / "debug.mp4").write_bytes(b"video")
    (folder / "research.mp4").write_bytes(b"research")
    write(
        folder / "lab_video_audit.json",
        {
            "inference_frames": 1,
            "terminal_frames": 40,
            "completion_sha256": sha(evidence / "completion.json"),
            "artifact_audit_sha256": sha(folder / "lab_artifact_audit.json"),
            "videos": {
                name: {"sha256": sha(folder / f"{name}.mp4"), "fps": 20, "decoded_frames": 41}
                for name in ("debug", "research")
            },
        },
    )
    lock_path = root / "plan.lock.json"
    lock = (
        read_json(lock_path)
        if lock_path.exists()
        else {
            "source_sha256": {},
            "graph_sha256": {},
            "coordinator_sha256": "coord",
            "timeout_seconds": 7200,
            "models": [],
            "cases": [],
        }
    )
    if not any(item["name"] == model for item in lock["models"]):
        lock["models"].append({"name": model, "sha256": "abc"})
    if not any(item["name"] == case["id"] for item in lock["cases"]):
        lock["cases"].append({"name": case["id"], "case": case})
    write(lock_path, lock)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"complete": True, "episodes": {}}
    manifest["lock_sha256"] = sha(lock_path)
    manifest["episodes"][f"{model}/{case['id']}"] = {
        "status": "complete",
        "receipt": {"completion_sha256": sha(evidence / "completion.json")},
    }
    write(manifest_path, manifest)
    return folder


def test_report_separates_splits_and_excludes_unsealed(tmp_path):
    training = tmp_path / "training"
    evaluation = tmp_path / "eval"
    out = tmp_path / "report"
    write(training / "run.json", {"device": "cuda"})
    write(
        training / "baseline_validation.json",
        {
            "step": 0,
            "aggregate": {"loss": 2.5},
            "aggregation": "equal_episode",
            "precision": "float32",
            "count": 6,
        },
    )
    (training / "history.jsonl").write_text(
        json.dumps({"step": 1, "loss": 1, "val_loss": 2, "grad_norms": {"vision": 3, "policy": 4}, "step_seconds": 5})
        + "\n"
    )
    defaults = {
        "discipline": "trackdrive",
        "seed": 0,
        "dr": "heavy",
        "variant": "visual_stress_v1",
        "track_seed": 1,
        "frozen_case": None,
        "lateral_start": 0.0,
        "yaw_start": 0.0,
    }
    cases = [
        dict(defaults, id="regression", split="regression"),
        dict(defaults, id="heldout", split="test"),
        dict(defaults, id="corrupt", split="test"),
        dict(defaults, id="missing", split="test"),
    ]
    for case in cases:
        write(tmp_path / f"{case['id']}.json", case)
    episode(evaluation, "baseline", cases[0])
    episode(evaluation, "baseline", cases[1], "timeout")
    broken = episode(evaluation, "baseline", cases[2], "timeout")
    (broken / "summary.json").write_text("{}")
    write(
        tmp_path / "plan.json",
        {
            "version": 1,
            "cases": [f"{c['id']}.json" for c in cases],
            "models": [{"name": "baseline", "checkpoint": "m.pt", "sha256": "abc"}],
        },
    )
    main(
        [
            "--training",
            str(training),
            "--evaluation",
            str(evaluation),
            "--out",
            str(out),
            "--plan",
            str(tmp_path / "plan.json"),
        ]
    )
    report = json.loads((out / "aggregates.json").read_text())
    assert report["training"]["baseline_validation"]["aggregate"]["loss"] == 2.5
    assert "Step-0 baseline validation loss: 2.500" in (out / "index.html").read_text()
    assert len(report["episodes"]) == 4
    assert [(x["split"], x["episodes"], x["finished"]) for x in report["groups"]] == [
        ("regression", 1, 1),
        ("test", 1, 0),
    ]
    assert report["episodes"][0]["latency"]["p50_ms"] == 15
    assert report["episodes"][0]["videos"]["debug"]["status"] == "receipt hash/count matched"
    assert report["episodes"][1]["result"] == "timeout"
    assert report["episodes"][2]["seal"] == "missing or invalid"
    assert report["episodes"][3]["case_id"] == "missing"
    assert "No results available" not in (out / "index.html").read_text()
    assert all(
        (out / name).is_file()
        for name in ("training-losses.png", "training-gradients.png", "training-step-time.png", "evaluation.png")
    )


def test_empty_report_is_explicit(tmp_path):
    out = tmp_path / "out"
    main(["--training", str(tmp_path / "none"), "--evaluation", str(tmp_path / "none"), "--out", str(out)])
    page = (out / "index.html").read_text()
    assert "No training results available" in page
    assert "No evaluation results available" in page


def test_command_ablation_modes_and_wilson_intervals(tmp_path):
    defaults = {
        "discipline": "skidpad",
        "seed": 0,
        "dr": "heavy",
        "variant": "matched",
        "track_seed": None,
        "lateral_start": 0.0,
        "yaw_start": 0.0,
        "split": "regression",
    }
    cases = [
        dict(defaults, id=mode, frozen_case=f"ablation-skidpad-72000-heavy-{mode}")
        for mode in ("recorded", "zero", "swapped", "unknown")
    ]
    cases[3]["frozen_case"] = None
    missing = dict(defaults, id="missing", frozen_case="ablation-skidpad-72000-heavy-zero")
    evaluation = tmp_path / "eval"
    for case, mode in zip(cases, ("recorded", "zero", "swapped", None), strict=True):
        write(tmp_path / f"{case['id']}.json", case)
        episode(evaluation, "baseline", case, "finished" if mode != "zero" else "timeout", mode)
    write(tmp_path / "missing.json", missing)
    write(
        tmp_path / "plan.json",
        {
            "version": 1,
            "cases": [f"{c['id']}.json" for c in [*cases, missing]],
            "models": [{"name": "baseline", "checkpoint": "m.pt", "sha256": "abc"}],
        },
    )
    out = tmp_path / "report"
    main(
        [
            "--training",
            str(tmp_path / "none"),
            "--evaluation",
            str(evaluation),
            "--out",
            str(out),
            "--plan",
            str(tmp_path / "plan.json"),
        ]
    )
    report = read_json(out / "aggregates.json")
    groups = {g["command_mode"]: g for g in report["groups"]}
    assert set(groups) == {"recorded", "zero", "swapped", "unknown"}
    assert groups["zero"]["finished"] == 0
    assert all(g["episodes"] == 1 for g in groups.values())
    assert 0.206 < groups["recorded"]["finished_wilson_95"][0] < 0.207
    assert 0.793 < groups["zero"]["finished_wilson_95"][1] < 0.794
    assert abs(wilson_interval(5, 10)[0] - 0.236593) < 1e-6
    rows = {row["case_id"]: row for row in report["episodes"]}
    assert rows["unknown"]["command_mode"] == "unknown"
    assert rows["missing"]["command_mode"] == "zero"
    page = (out / "index.html").read_text()
    assert "95% Wilson" in page and "Command mode" in page


def read_json(path):
    return json.loads(path.read_text())


def test_final_gate_requires_audit_video_and_plan_but_accepts_timeout(tmp_path):
    case = {
        "id": "run",
        "discipline": "skidpad",
        "seed": 0,
        "split": "test",
        "dr": "heavy",
        "variant": "matched",
        "track_seed": None,
        "frozen_case": None,
        "lateral_start": 0.0,
        "yaw_start": 0.0,
    }
    evaluation = tmp_path / "eval"
    folder = episode(evaluation, "baseline", case, "timeout")
    write(tmp_path / "case.json", case)
    plan = tmp_path / "plan.json"
    write(
        plan,
        {"version": 1, "cases": ["case.json"], "models": [{"name": "baseline", "checkpoint": "m.pt", "sha256": "abc"}]},
    )
    out = tmp_path / "out"
    args = [
        "--training",
        str(tmp_path / "none"),
        "--evaluation",
        str(evaluation),
        "--plan",
        str(plan),
        "--out",
        str(out),
        "--require-complete",
    ]
    (evaluation / "logs" / "baseline").mkdir(parents=True)
    (evaluation / "logs" / "baseline" / "run.log").write_text("coordinator log")
    (evaluation / "cases").mkdir()
    (evaluation / "cases" / "run.json").write_text("{}")
    main(args)
    report = read_json(out / "aggregates.json")
    assert report["final_gate"]["status"] == "passed"
    assert len(report["episodes"]) == 1
    assert report["groups"][0]["finished"] == 0
    video = read_json(folder / "lab_video_audit.json")
    video["videos"]["research"]["fps"] = 19
    write(folder / "lab_video_audit.json", video)
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(args)
    report = read_json(out / "aggregates.json")
    assert any("research video receipt mismatch" in x for x in report["final_gate"]["failures"])
    video["videos"]["research"]["fps"] = 20
    write(folder / "lab_video_audit.json", video)
    (folder / "telemetry.json").write_text("[1]")
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(args)
    report = read_json(out / "aggregates.json")
    assert report["episodes"][0]["audit"] == "stale or unbound"
    assert report["groups"] == []
    write(folder / "telemetry.json", [])
    (folder / "research.mp4").unlink()
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(args)
    report = read_json(out / "aggregates.json")
    assert any("research video missing" in x for x in report["final_gate"]["failures"])
    assert report["groups"][0]["episodes"] == 1
    write(folder / "lab_artifact_audit.json", {"passed": False})
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(args)
    report = read_json(out / "aggregates.json")
    assert report["groups"] == []
    assert report["episodes"][0]["audit"] == "failed"
    assert any("artifact audit failed" in x for x in report["final_gate"]["failures"])
    with pytest.raises(SystemExit):
        main(
            [
                "--training",
                str(tmp_path / "none"),
                "--evaluation",
                str(evaluation),
                "--out",
                str(out),
                "--require-complete",
            ]
        )


def test_final_gate_rejects_pilot_wrong_identity_and_extras(tmp_path):
    base = {
        "discipline": "skidpad",
        "seed": 0,
        "split": "test",
        "dr": "heavy",
        "variant": "matched",
        "track_seed": None,
        "frozen_case": None,
        "lateral_start": 0.0,
        "yaw_start": 0.0,
    }
    evaluation = tmp_path / "eval"
    for name in ("pilot", "wrong", "extra"):
        episode(evaluation, "baseline", dict(base, id=name), pilot=name == "pilot")
    for name in ("pilot", "wrong"):
        write(tmp_path / f"{name}.json", dict(base, id=name, seed=1 if name == "wrong" else 0))
    plan = tmp_path / "plan.json"
    write(
        plan,
        {
            "version": 1,
            "cases": ["pilot.json", "wrong.json"],
            "models": [{"name": "baseline", "checkpoint": "m.pt", "sha256": "abc"}],
        },
    )
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(
            [
                "--training",
                str(tmp_path / "none"),
                "--evaluation",
                str(evaluation),
                "--plan",
                str(plan),
                "--out",
                str(out),
                "--require-complete",
            ]
        )
    gate = read_json(out / "aggregates.json")["final_gate"]
    assert any("pilot evidence" in x for x in gate["failures"])
    assert any("plan identity mismatch" in x for x in gate["failures"])
    assert any("unexpected episode directory" in x for x in gate["failures"])


def test_unequal_plans_separate_cohorts_and_only_compare_paired_cases(tmp_path):
    base = {
        "discipline": "skidpad",
        "seed": 0,
        "split": "regression",
        "dr": "heavy",
        "variant": "matched",
        "track_seed": None,
        "frozen_case": None,
        "lateral_start": 0.0,
        "yaw_start": 0.0,
    }
    paired = dict(base, id="paired")
    legacy = [dict(base, id="legacy1"), dict(base, id="legacy2")]
    evaluation = tmp_path / "eval"
    paired_root, legacy_root = tmp_path / "paired-origin", tmp_path / "legacy-origin"
    for model in ("baseline", "candidate"):
        (evaluation / model).mkdir(parents=True)
        original = episode(paired_root, model, paired, "finished" if model == "baseline" else "timeout")
        (evaluation / model / paired["id"]).symlink_to(original, target_is_directory=True)
    for case in legacy:
        original = episode(legacy_root, "candidate", case, "finished")
        (evaluation / "candidate" / case["id"]).symlink_to(original, target_is_directory=True)
    models = [
        {"name": name, "checkpoint": "/remote/not-needed.pt", "sha256": "abc"} for name in ("baseline", "candidate")
    ]

    def case_spec(c):
        return {"name": c["id"], "path": "/remote/not-needed.json", "case": c}

    paired_plan = tmp_path / "paired-lock.json"
    legacy_plan = tmp_path / "legacy-lock.json"
    write(paired_plan, {"version": 1, "cases": [case_spec(paired)], "models": models})
    write(legacy_plan, {"version": 1, "cases": [case_spec(c) for c in legacy], "models": models[1:]})
    coverage = tmp_path / "coverage.json"
    write(
        coverage,
        {
            "schema_version": 1,
            "paired_case_ids": ["paired"],
            "candidate_only_legacy_case_ids": [c["id"] for c in legacy],
        },
    )
    out = tmp_path / "out"
    args = [
        "--training",
        str(tmp_path / "none"),
        "--evaluation",
        str(evaluation),
        "--out",
        str(out),
        "--plan",
        str(paired_plan),
        "--additional-plan",
        str(legacy_plan),
        "--coverage",
        str(coverage),
        "--require-complete",
    ]
    main(args)
    report = read_json(out / "aggregates.json")
    assert report["final_gate"]["planned"] == 4
    groups = {(g["model"], g["cohort"]): g for g in report["groups"]}
    assert groups[("candidate", "paired_exposed")]["finished_fraction"] == 0
    assert groups[("candidate", "paired_exposed")]["episodes"] == 1
    assert groups[("candidate", "candidate_legacy")]["episodes"] == 2
    assert groups[("candidate", "candidate_legacy")]["finished_fraction"] == 1
    pairs = report["paired_comparisons"]
    assert pairs["required_pairs"] == pairs["measured_pairs"] == 1
    assert pairs["cases"][0]["case_id"] == "paired" and pairs["cases"][0]["finished_delta"] == -1
    page = (out / "index.html").read_text()
    assert "candidate_legacy" in page and "paired_exposed" in page
    paired_only_args = [
        "--training",
        str(tmp_path / "none"),
        "--evaluation",
        str(evaluation),
        "--out",
        str(out),
        "--plan",
        str(paired_plan),
        "--coverage",
        str(coverage),
        "--require-complete",
    ]
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(paired_only_args)
    partial = read_json(out / "aggregates.json")
    assert any("required coverage case not in supplied plans" in x for x in partial["final_gate"]["failures"])
    conflict = tmp_path / "conflict.json"
    write(conflict, {"version": 1, "cases": [case_spec(dict(paired, seed=9))], "models": models})
    with pytest.raises(ValueError, match="Contradictory duplicate"):
        main([*args, "--additional-plan", str(conflict)])
    (evaluation / "candidate" / "paired" / "lab_artifact_audit.json").unlink()
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(args)
    incomplete = read_json(out / "aggregates.json")["paired_comparisons"]
    assert incomplete["required_pairs"] == 1 and incomplete["measured_pairs"] == 0
    assert incomplete["unmeasured_case_ids"] == ["paired"]


def test_final_gate_rejects_same_length_foreign_video_receipt(tmp_path):
    base = {
        "discipline": "skidpad",
        "seed": 0,
        "split": "test",
        "dr": "heavy",
        "variant": "matched",
        "track_seed": None,
        "frozen_case": None,
        "lateral_start": 0.0,
        "yaw_start": 0.0,
    }
    evaluation = tmp_path / "eval"
    first = episode(evaluation, "baseline", dict(base, id="first"))
    second = episode(evaluation, "baseline", dict(base, id="second"))
    # Identical fake payloads and counts, but the second receipt belongs to a different episode.
    (first / "lab_video_audit.json").write_bytes((second / "lab_video_audit.json").read_bytes())
    for name in ("first", "second"):
        write(tmp_path / f"{name}.json", dict(base, id=name))
    plan = tmp_path / "plan.json"
    write(
        plan,
        {
            "version": 1,
            "cases": ["first.json", "second.json"],
            "models": [{"name": "baseline", "checkpoint": "m.pt", "sha256": "abc"}],
        },
    )
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(
            [
                "--training",
                str(tmp_path / "none"),
                "--evaluation",
                str(evaluation),
                "--plan",
                str(plan),
                "--out",
                str(out),
                "--require-complete",
            ]
        )
    report = read_json(out / "aggregates.json")
    rows = {r["case_id"]: r for r in report["episodes"]}
    assert rows["first"]["videos"]["debug"]["status"] == "receipt mismatch"
    assert rows["second"]["videos"]["debug"]["status"] == "receipt hash/count matched"


def test_validation_components_preserve_baseline_and_independent_changes(tmp_path):
    training = tmp_path / "training"
    out = tmp_path / "out"
    write(
        training / "baseline_validation.json",
        {
            "step": 0,
            "aggregate": {"loss": 2.5, "plan_nll": 2, "steer": 0.1, "longitudinal": 0.2},
        },
    )
    (training / "history.jsonl").write_text(
        json.dumps(
            {
                "step": 10,
                "val_loss": 1.5,
                "validation": {
                    "aggregate": {
                        "loss": 1.5,
                        "plan_nll": 0.8,
                        "steer": 0.3,
                        "longitudinal": 0.15,
                    }
                },
            }
        )
        + "\n"
    )
    main(["--training", str(training), "--evaluation", str(tmp_path / "none"), "--out", str(out)])
    rows = read_json(out / "aggregates.json")["training"]["validation_series"]
    assert rows[0] == {"step": 0, "loss": 2.5, "plan_nll": 2, "steer": 0.1, "longitudinal": 0.2}
    assert rows[1]["plan_nll"] < rows[0]["plan_nll"]
    assert rows[1]["steer"] > rows[0]["steer"]
    assert (out / "training-validation.png").is_file()


def test_training_telemetry_missing_malformed_and_failed_are_visible(tmp_path):
    out = tmp_path / "out"
    receipt = tmp_path / "telemetry.json"
    args = [
        "--training",
        str(tmp_path / "none"),
        "--evaluation",
        str(tmp_path / "none"),
        "--out",
        str(out),
        "--training-telemetry",
        str(receipt),
    ]
    main(args)
    assert read_json(out / "aggregates.json")["training"]["telemetry"]["status"] == "unavailable"
    assert "Training hardware telemetry: unavailable" in (out / "index.html").read_text()
    receipt.write_text("{")
    main(args)
    assert read_json(out / "aggregates.json")["training"]["telemetry"]["status"] == "malformed"
    sample = {
        "wall_s": 1,
        "gpu_pct": 75,
        "power_w": 100,
        "host_ram_gib": 10,
        "cpu_pct": 25,
        "compute_processes": [{"pid": 8, "name": "other"}],
    }
    write(
        receipt,
        {
            "command": ["python", "train.py"],
            "wall_seconds": 2,
            "returncode": 1,
            "samples": [sample],
            "compute_processes_before": [],
            "compute_processes_after": [],
        },
    )
    main(args)
    telemetry = read_json(out / "aggregates.json")["training"]["telemetry"]
    assert telemetry["status"] == "failed" and telemetry["returncode"] == 1
    assert telemetry["sha256"] == sha(receipt) and telemetry["command"] == ["python", "train.py"]
    assert telemetry["median_gpu_pct"] == 75 and telemetry["median_power_w"] == 100
    assert telemetry["peak_host_ram_gib"] == 10 and telemetry["compute_processes"][0]["name"] == "other"
    page = (out / "index.html").read_text()
    assert "Training hardware telemetry: failed" in page and "other" in page
    assert (out / "training-hardware.png").is_file()


def test_final_report_rejects_noncomparable_pairs(tmp_path):
    case = {
        "id": "pair",
        "split": "regression",
        "discipline": "skidpad",
        "seed": 0,
        "dr": "heavy",
        "variant": "matched",
        "track_seed": None,
        "frozen_case": None,
        "lateral_start": 0.0,
        "yaw_start": 0.0,
    }
    evaluation = tmp_path / "eval"
    episode(evaluation, "baseline", case)
    candidate = episode(evaluation, "candidate", case)
    plan = tmp_path / "plan.json"
    write(
        plan,
        {
            "version": 1,
            "cases": [{"path": "unused", "case": case}],
            "models": [{"name": name, "checkpoint": "unused", "sha256": "abc"} for name in ("baseline", "candidate")],
        },
    )
    coverage = tmp_path / "coverage.json"
    write(coverage, {"schema_version": 1, "paired_case_ids": ["pair"], "candidate_only_legacy_case_ids": []})
    out = tmp_path / "report"
    args = [
        "--training",
        str(tmp_path / "none"),
        "--evaluation",
        str(evaluation),
        "--out",
        str(out),
        "--plan",
        str(plan),
        "--coverage",
        str(coverage),
        "--require-complete",
    ]
    main(args)
    assert read_json(out / "aggregates.json")["paired_comparisons"]["measured_pairs"] == 1

    def change_metadata(field, value):
        evidence = candidate / "evidence"
        metadata = read_json(evidence / "metadata.json")
        if value is None:
            metadata.pop(field)
        else:
            metadata[field] = value
        write(evidence / "metadata.json", metadata)
        frame = json.loads((evidence / "manifest.jsonl").read_text())
        frame["metadata_sha256"] = sha(evidence / "metadata.json")
        (evidence / "manifest.jsonl").write_text(json.dumps(frame) + "\n")
        completion = read_json(evidence / "completion.json")
        completion.update(
            metadata_sha256=sha(evidence / "metadata.json"), manifest_sha256=sha(evidence / "manifest.jsonl")
        )
        write(evidence / "completion.json", completion)
        audit = read_json(candidate / "lab_artifact_audit.json")
        audit["completion_sha256"] = sha(evidence / "completion.json")
        write(candidate / "lab_artifact_audit.json", audit)
        video = read_json(candidate / "lab_video_audit.json")
        video.update(
            completion_sha256=sha(evidence / "completion.json"),
            artifact_audit_sha256=sha(candidate / "lab_artifact_audit.json"),
        )
        write(candidate / "lab_video_audit.json", video)
        manifest_path = evaluation / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["episodes"]["candidate/pair"]["receipt"]["completion_sha256"] = sha(evidence / "completion.json")
        write(manifest_path, manifest)

    change_metadata("physics_settings", "different")
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(args)
    assert "physics_settings" in read_json(out / "aggregates.json")["paired_comparisons"]["comparison_errors"]["pair"]
    change_metadata("physics_settings", None)
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(args)
    assert "physics_settings" in read_json(out / "aggregates.json")["paired_comparisons"]["comparison_errors"]["pair"]
    change_metadata("physics_settings", "same")
    (candidate / "export.json").write_text("tampered")
    with pytest.raises(SystemExit, match="Final gate failed"):
        main(args)
    report = read_json(out / "aggregates.json")
    assert report["paired_comparisons"]["cases"] == []
    assert any("noncomparable" in failure for failure in report["final_gate"]["failures"])
