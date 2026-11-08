"""Freeze and run the fixed exploratory visual-stress pairs."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from isaac_eval.parity_cases import CHECKPOINT_SHA256, all_cases, case_manifest, json_sha256, sha256_file
from isaac_eval.parity_protocol import GRAPH_FILES, runtime_sources, verify_baseline
from isaac_eval.rendering import PROFILE

CASES = (
    "skidpad-72000-light-recorded",
    "skidpad-72000-heavy-recorded",
    "retention-trackdrive_020000-light-recorded",
    "retention-trackdrive_020000-heavy-recorded",
)
VARIANTS = ("matched", "visual_stress_v1")
TOOLS = (
    "scripts/audit_isaac_episode.py",
    "scripts/process_isaac_parity.py",
    "scripts/report_isaac_stress.py",
    "isaac_eval/render_debug.py",
)
ASSET_ROOT = Path("isaac_eval/assets")
CAMERA_SOURCES = {
    "isaac_eval/" + name
    for name in ("verify_parity_camera.py", "parity_geometry.py", "parity_scene.py", "scene.py", "rendering.py")
}
VEHICLE_SOURCES = {
    "isaac_eval/" + name
    for name in (
        "scene.py",
        "rendering.py",
        "vehicle.py",
        "realistic_scene.py",
        "realistic_car.py",
        "appearance_guard.py",
        "export_scene.py",
        "preview_realism.py",
        "verify_integration.py",
        "evaluate.py",
        "assets/manifest.json",
    )
} | {"src/fsop/geometry/camera.py"}


def validate_gate(name, gate):
    expected = VEHICLE_SOURCES if name == "vehicle" else CAMERA_SOURCES
    if (
        gate.get("pass") is not True
        or gate.get("render_profile") != PROFILE
        or set(gate.get("source_sha256", {})) != expected
    ):
        raise ValueError(f"Invalid {name} gate identity")
    if name == "vehicle":
        valid = (
            gate.get("dofs") == ["PhysX Vehicle wheel controllers"]
            and set(gate.get("motion", {}))
            == {"straight", "left", "right", "fast_straight", "fast_left", "fast_right", "contact"}
            and set(gate.get("camera", {})) == {"road", "wide"}
            and isinstance(gate.get("contact_events"), list)
            and isinstance(gate.get("settled_position"), list)
            and {"capture_sim_dt", "capture_position_change_m", "one_control_tick_projection_errors_px"} <= gate.keys()
            and all(
                {"start", "end", "yaw", "speed_before_brake", "speed_after_brake", "max_z", "min_up"} <= motion.keys()
                for motion in gate.get("motion", {}).values()
            )
            and all(
                {"shape", "predicted", "measured", "errors_px", "properties"} <= camera.keys()
                for camera in gate.get("camera", {}).values()
            )
        )
    else:
        samples = gate.get("samples", [])
        valid = (
            len(samples) == 6
            and {(s.get("level"), s.get("pose")) for s in samples}
            == {(level, pose) for level in ("light", "heavy") for pose in range(3)}
            and all(
                set(s.get("cameras", {})) == {"road", "wide"}
                and "capture_sim_dt" in s
                and all(
                    {"expected_px", "measured_px", "errors_px"} <= camera.keys()
                    and all(len(camera[key]) == 3 for key in ("expected_px", "measured_px", "errors_px"))
                    for camera in s["cameras"].values()
                )
                for s in samples
            )
        )
    if not valid:
        raise ValueError(f"Invalid {name} gate measurements")
    for path, expected_hash in gate["source_sha256"].items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Gate source changed: {path}")


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def read_receipt(record):
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Receipt changed: {path}")
    return json.loads(path.read_text())


def validate_manifest(manifest):
    if manifest["manifest_sha256"] != json_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Manifest digest changed")
    if manifest["cases"] != [case_manifest(case) for case in all_cases()]:
        raise ValueError("Original 41-case manifest required")


def validate_plan(plan, manifest):
    validate_manifest(manifest)
    if plan["schema"] != "isaac_visual_stress_v1" or plan["purpose"] != "exploratory_paired_subset":
        raise ValueError("Invalid experiment identity")
    if plan["plan_sha256"] != json_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("Experiment plan digest changed")
    if plan["manifest_sha256"] != manifest["manifest_sha256"] or plan["render_profile"] != PROFILE:
        raise ValueError("Manifest or render profile changed")
    if [(r["case_id"], r["variant"]) for r in plan["episodes"]] != [(c, v) for c in CASES for v in VARIANTS]:
        raise ValueError("Experiment must contain exactly the eight fixed paired episodes")
    if plan["source_sha256"] != runtime_sources() or plan["tool_sha256"] != {p: sha256_file(p) for p in TOOLS}:
        raise ValueError("Runtime or postprocessing source changed")
    verify_baseline()
    if (
        plan["graph_sha256"] != {p: sha256_file(p) for p in GRAPH_FILES}
        or plan["checkpoint_sha256"] != CHECKPOINT_SHA256
    ):
        raise ValueError("Graph or checkpoint identity changed")
    if sha256_file("/frozen/model.pt") != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint changed")
    for name, digest in plan["asset_sha256"].items():
        path = (ASSET_ROOT / name).resolve()
        if not path.is_relative_to(ASSET_ROOT.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"Asset changed: {name}")
    if set(plan["gates"]) != {"vehicle", "camera", *VARIANTS}:
        raise ValueError("Vehicle, camera, and both pilot gates are required")
    for name, entry in plan["gates"].items():
        gate = read_receipt(entry["verification"])
        if not gate.get("passed", gate.get("pass", False)):
            raise ValueError(f"Gate failed: {name}")
        if name in VARIANTS:
            metadata = read_receipt(entry["metadata"])
            summary = read_receipt(entry["summary"])
            from isaac_eval.evidence import verified_summary

            evidence = Path(entry["metadata"]["path"]).parent
            if verified_summary(evidence) != summary:
                raise ValueError(f"Pilot completion seal failed: {name}")
            for filename, digest in gate["evidence_sha256"].items():
                path = (evidence / filename).resolve()
                if not path.is_relative_to(evidence.resolve()) or sha256_file(path) != digest:
                    raise ValueError(f"Pilot evidence changed: {name}/{filename}")
            if (
                metadata["variant"] != name
                or not metadata["pilot_only"]
                or metadata["source_sha256"] != plan["source_sha256"]
                or metadata["graph_sha256"] != plan["graph_sha256"]
                or metadata["asset_sha256"] != plan["asset_sha256"]
                or metadata["checkpoint_sha256"] != CHECKPOINT_SHA256
                or not summary["pilot_only"]
                or summary["variant"] != name
                or summary["recorded_frames"] != 120
                or gate["frames"] != 120
                or not gate["frozen_unchanged"]
                or gate["evidence_sha256"]["metadata.json"] != entry["metadata"]["sha256"]
            ):
                raise ValueError(f"Pilot identity failed: {name}")
        else:
            validate_gate(name, gate)
    if set(plan["previews"]) != set(CASES):
        raise ValueError("Four fixed case previews required")
    for case_id, record in plan["previews"].items():
        preview = read_receipt(record)
        if (
            preview["schema"] != "visual_stress_preview_v1"
            or preview["case_id"] != case_id
            or preview["source_sha256"] != plan["source_sha256"]
            or preview["asset_sha256"] != plan["asset_sha256"]
            or preview["render_profile"] != PROFILE
            or len(preview["poses"]) < 2
            or set(preview["variants"]) != set(VARIANTS)
        ):
            raise ValueError(f"Preview identity failed: {case_id}")
        for variant in VARIANTS:
            row = preview["variants"][variant]
            if not row["images"] or row["physics_sha256"] != preview["variants"]["matched"]["physics_sha256"]:
                raise ValueError("Preview images or physical contract invalid")
            for filename, digest in row["images"].items():
                if sha256_file(Path(record["path"]).parent / variant / filename) != digest:
                    raise ValueError("Preview image changed")
        expected = preview["variants"]["visual_stress_v1"]["stress"]["layout_sha256"]
        for row in plan["episodes"]:
            if row["case_id"] == case_id and row["layout_sha256"] != expected:
                raise ValueError("Episode layout differs from preview")
    return plan


def episode_spec(plan, case_id, variant):
    rows = [r for r in plan["episodes"] if r["case_id"] == case_id and r["variant"] == variant]
    if len(rows) != 1:
        raise ValueError("Episode is not in the fixed experiment")
    return rows[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    for flag in ("manifest", "vehicle", "camera", "matched-pilot", "stress-pilot", "previews", "out"):
        freeze.add_argument("--" + flag, type=Path, required=True)
    for command in ("run", "audit"):
        p = sub.add_parser(command)
        p.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        manifest = json.loads(args.manifest.read_text())
        previews = {c: receipt(args.previews / c / "verification.json") for c in CASES}
        first = read_receipt(previews[CASES[0]])
        gates = {
            name: {"verification": receipt(path)} for name, path in (("vehicle", args.vehicle), ("camera", args.camera))
        }
        for name, path in zip(VARIANTS, (args.matched_pilot, args.stress_pilot), strict=True):
            gates[name] = {
                "verification": receipt(path / "inference_verification.json"),
                "metadata": receipt(path / "evidence/metadata.json"),
                "summary": receipt(path / "summary.json"),
            }
        episodes = [
            {
                "case_id": c,
                "variant": v,
                "layout_sha256": read_receipt(previews[c])["variants"]["visual_stress_v1"]["stress"]["layout_sha256"],
            }
            for c in CASES
            for v in VARIANTS
        ]
        plan = {
            "schema": "isaac_visual_stress_v1",
            "purpose": "exploratory_paired_subset",
            "episodes": episodes,
            "manifest": receipt(args.manifest),
            "manifest_sha256": manifest["manifest_sha256"],
            "source_sha256": runtime_sources(),
            "tool_sha256": {p: sha256_file(p) for p in TOOLS},
            "graph_sha256": {p: sha256_file(p) for p in GRAPH_FILES},
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "asset_sha256": first["asset_sha256"],
            "render_profile": PROFILE,
            "gates": gates,
            "previews": previews,
        }
        plan["plan_sha256"] = json_sha256(plan)
        validate_plan(plan, manifest)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as f:
            json.dump(plan, f, indent=2)
        print(plan["plan_sha256"])
        return
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    manifest = read_receipt(plan["manifest"])
    validate_plan(plan, manifest)
    root = plan_path.parent
    if args.command == "audit":
        from scripts.audit_isaac_episode import audit_episode

        for row in plan["episodes"]:
            if not audit_episode(root / row["variant"] / row["case_id"])["passed"]:
                raise ValueError(f"Episode audit failed: {row}")
        return
    statuses = {}
    for variant in VARIANTS:
        folder = root / variant
        folder.mkdir(exist_ok=False)
        statuses[variant] = {"complete": False, "cases": []}
        (folder / "status.json").write_text(json.dumps(statuses[variant], indent=2))
    for row in plan["episodes"]:
        variant, case_id = row["variant"], row["case_id"]
        folder, status = root / variant, statuses[variant]
        target = folder / case_id
        command = [
            "/isaac-sim/python.sh",
            "isaac_eval/evaluate_parity.py",
            "--case",
            case_id,
            "--manifest",
            plan["manifest"]["path"],
            "--stress-plan",
            str(plan_path),
            "--variant",
            variant,
            "--out",
            str(target),
        ]
        print(f"START {variant}/{case_id}", flush=True)
        started = time.perf_counter()
        with (folder / (case_id + ".log")).open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        success = result.returncode == 0 and (target / "evidence/completion.json").is_file()
        status["cases"].append(
            {
                "case_id": case_id,
                "technical_success": success,
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
            }
        )
        status["complete"] = len(status["cases"]) == len(CASES) and success
        (folder / "status.json").write_text(json.dumps(status, indent=2))
        print(f"END {variant}/{case_id} returncode={result.returncode} elapsed={elapsed:.1f}s", flush=True)
        if not success:
            raise RuntimeError(f"Episode failed: {variant}/{case_id}")


if __name__ == "__main__":
    main()
