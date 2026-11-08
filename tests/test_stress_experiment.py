import copy

import pytest

from isaac_eval import stress_experiment as stress
from isaac_eval.parity_cases import all_cases, json_sha256, suite_manifest
from isaac_eval.parity_protocol import validate as validate_legacy


def manifest():
    value = suite_manifest()
    value["manifest_sha256"] = json_sha256(value)
    return value


def test_original_manifest_must_retain_all_cases():
    original = manifest()
    stress.validate_manifest(original)
    altered = copy.deepcopy(original)
    altered["cases"] = altered["cases"][:4]
    altered["manifest_sha256"] = json_sha256({k: v for k, v in altered.items() if k != "manifest_sha256"})
    with pytest.raises(ValueError, match="41-case"):
        stress.validate_manifest(altered)


def test_fixed_roster_and_variant_tamper_fail_before_runtime_checks():
    original = manifest()
    rows = [
        {"case_id": case, "variant": variant, "layout_sha256": "a" * 64}
        for case in stress.CASES
        for variant in stress.VARIANTS
    ]
    plan = {
        "schema": "isaac_visual_stress_v1",
        "purpose": "exploratory_paired_subset",
        "manifest_sha256": original["manifest_sha256"],
        "render_profile": stress.PROFILE,
        "episodes": rows,
    }
    plan["plan_sha256"] = json_sha256(plan)
    assert stress.episode_spec(plan, stress.CASES[0], "matched") == rows[0]
    with pytest.raises(ValueError):
        stress.episode_spec(plan, stress.CASES[0], "unknown")
    for change in ("missing", "duplicate", "variant", "digest"):
        damaged = copy.deepcopy(plan)
        if change == "missing":
            damaged["episodes"].pop()
        elif change == "duplicate":
            damaged["episodes"][-1] = damaged["episodes"][0]
        elif change == "variant":
            damaged["episodes"][0]["variant"] = "visual_stress_v1"
        else:
            damaged["purpose"] = "different"
        if change != "digest":
            damaged["plan_sha256"] = json_sha256({k: v for k, v in damaged.items() if k != "plan_sha256"})
        with pytest.raises(ValueError):
            stress.validate_plan(damaged, original)


def test_legacy_protocol_still_requires_all_41():
    m = manifest()
    protocol = {
        "manifest_sha256": m["manifest_sha256"],
        "render_profile": stress.PROFILE,
        "case_ids": [c.case_id for c in all_cases()][:4],
    }
    with pytest.raises(ValueError, match="all 41"):
        validate_legacy(protocol, m)


def test_source_and_asset_changes_reject_plan(tmp_path, monkeypatch):
    original = manifest()
    asset = tmp_path / "person.usd"
    asset.write_bytes(b"original person")
    import hashlib

    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    monkeypatch.setattr(stress, "ASSET_ROOT", tmp_path)
    monkeypatch.setattr(stress, "TOOLS", ())
    monkeypatch.setattr(stress, "GRAPH_FILES", ())
    monkeypatch.setattr(stress, "runtime_sources", lambda: {"runtime.py": "a" * 64})
    monkeypatch.setattr(stress, "verify_baseline", lambda: None)
    real_sha256 = stress.sha256_file
    monkeypatch.setattr(
        stress, "sha256_file", lambda p: stress.CHECKPOINT_SHA256 if str(p) == "/frozen/model.pt" else real_sha256(p)
    )
    plan = {
        "schema": "isaac_visual_stress_v1",
        "purpose": "exploratory_paired_subset",
        "manifest_sha256": original["manifest_sha256"],
        "render_profile": stress.PROFILE,
        "episodes": [{"case_id": c, "variant": v} for c in stress.CASES for v in stress.VARIANTS],
        "source_sha256": {"runtime.py": "a" * 64},
        "tool_sha256": {},
        "graph_sha256": {},
        "checkpoint_sha256": stress.CHECKPOINT_SHA256,
        "asset_sha256": {"person.usd": digest},
        "gates": {},
    }
    plan["plan_sha256"] = json_sha256(plan)
    asset.write_bytes(b"tampered person")
    with pytest.raises(ValueError, match="Asset changed"):
        stress.validate_plan(plan, original)
    asset.write_bytes(b"original person")
    monkeypatch.setattr(stress, "runtime_sources", lambda: {"runtime.py": "b" * 64})
    with pytest.raises(ValueError, match="source changed"):
        stress.validate_plan(plan, original)


def test_runner_executes_frozen_order(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from types import SimpleNamespace

    plan_path = tmp_path / "experiment.json"
    rows = [{"case_id": case, "variant": variant} for case in stress.CASES for variant in stress.VARIANTS]
    plan_path.write_text(json.dumps({"episodes": rows, "manifest": {"path": "manifest.json"}}))
    monkeypatch.setattr(stress, "read_receipt", lambda *_: {})
    monkeypatch.setattr(stress, "validate_plan", lambda *_: None)
    seen = []

    def run(command, **_):
        seen.append((command[command.index("--case") + 1], command[command.index("--variant") + 1]))
        evidence = stress.Path(command[command.index("--out") + 1]) / "evidence"
        evidence.mkdir(parents=True)
        (evidence / "completion.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stress.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["stress_experiment", "run", "--plan", str(plan_path)])
    stress.main()
    assert seen == [(row["case_id"], row["variant"]) for row in rows]
    assert capsys.readouterr().out.count("END ") == 8
    for variant in stress.VARIANTS:
        status = json.loads((tmp_path / variant / "status.json").read_text())
        assert status["complete"] and all(
            row["returncode"] == 0 and row["elapsed_seconds"] >= 0 for row in status["cases"]
        )


def test_gate_roles_require_inventory_and_characteristic_measurements(monkeypatch):
    monkeypatch.setattr(stress, "sha256_file", lambda _: "hash")
    vehicle = {
        "pass": True,
        "render_profile": stress.PROFILE,
        "source_sha256": dict.fromkeys(stress.VEHICLE_SOURCES, "hash"),
        "dofs": ["PhysX Vehicle wheel controllers"],
        "motion": {
            key: dict.fromkeys(("start", "end", "yaw", "speed_before_brake", "speed_after_brake", "max_z", "min_up"))
            for key in ("straight", "left", "right", "fast_straight", "fast_left", "fast_right", "contact")
        },
        "camera": {
            key: dict.fromkeys(("shape", "predicted", "measured", "errors_px", "properties"))
            for key in ("road", "wide")
        },
        "settled_position": [0, 0, 0],
        "capture_sim_dt": 0,
        "capture_position_change_m": 0,
        "one_control_tick_projection_errors_px": {},
        "contact_events": [],
    }
    camera = {
        "pass": True,
        "render_profile": stress.PROFILE,
        "source_sha256": dict.fromkeys(stress.CAMERA_SOURCES, "hash"),
        "samples": [
            {
                "level": level,
                "pose": pose,
                "capture_sim_dt": 0,
                "cameras": {
                    name: dict.fromkeys(("expected_px", "measured_px", "errors_px"), [0, 0, 0])
                    for name in ("road", "wide")
                },
            }
            for level in ("light", "heavy")
            for pose in range(3)
        ],
    }
    stress.validate_gate("vehicle", vehicle)
    stress.validate_gate("camera", camera)
    with pytest.raises(ValueError):
        stress.validate_gate("vehicle", camera)
    camera["source_sha256"] = {}
    with pytest.raises(ValueError):
        stress.validate_gate("camera", camera)
