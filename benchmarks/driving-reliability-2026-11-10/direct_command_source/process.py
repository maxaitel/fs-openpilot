"""Independently audit and render one sealed native Lab evaluation episode."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fsop.sim.referee import Referee
from fsop.sim.route_commands import RouteCommands
from isaac_eval.evidence import read_evidence, verified_summary
from isaac_eval.parity_cases import realized_environment, sha256_file
from isaac_lab_fsop.cases import Case, environment
from scripts.audit_isaac_episode import Checks, advance


def audit_episode(directory, replay_receipt):
    directory = Path(directory)
    replay_receipt = Path(replay_receipt)
    if (directory / "lab_artifact_audit.json").exists():
        raise FileExistsError("Refusing to overwrite artifact audit")
    checks = Checks()
    report = {
        "passed": False,
        "verification_limits": [
            "Mission, commands and referee are replayed from saved states; physics and inference are not rerun.",
            "Physical contacts are unique recorded native contact events, not independently simulated.",
            "Checkpoint and model hashes bind producer identities but do not independently inspect weights.",
            "Final full body pose is not recorded; a terminal rollover override cannot be independently verified.",
        ],
    }
    try:
        evidence = directory / "evidence"
        summary = verified_summary(evidence)
        if summary is None:
            raise ValueError("A completion receipt is required, including pilots")
        receipt = json.loads((evidence / "completion.json").read_text())
        if (evidence / receipt["summary_path"]).resolve() != (directory / "summary.json").resolve():
            raise ValueError("Completion receipt does not bind the episode summary")
        metadata = json.loads((evidence / "metadata.json").read_text())
        schema = metadata.get("schema_version")
        contract = metadata.get("command_contract")
        if (schema != 4 or metadata.get("controller_name") != "direct_native_actuator_v1"
                or summary.get("controller_name") != metadata["controller_name"]
                or summary.get("command_contract") != contract or not isinstance(contract, dict)
                or metadata.get("controller_contract") != {"mode": "direct_native_actuator_v1", "command_contract": contract}
                or summary.get("controller_contract") != metadata["controller_contract"]):
            raise ValueError("Invalid direct-command evidence contract")
        replay = json.loads(replay_receipt.read_text())
        replay_sha = sha256_file(replay_receipt)
        if (replay.get("passed") is not True or replay.get("frozen_unchanged") is not True
                or replay.get("frames") != receipt["frame_count"]
                or replay.get("expected_frames") != receipt["frame_count"]
                or replay.get("temporal_phases") != [0, 1, 2, 3]
                or replay.get("case") != metadata["case"]
                or replay.get("command_contract") != contract
                or replay.get("checkpoint_sha256") != metadata["checkpoint_sha256"]
                or replay.get("base_state_sha256") != metadata["state_sha256"]
                or replay.get("head_state_sha256") != metadata["head_state_sha256"]
                or replay.get("source_sha256") != metadata["source_sha256"]
                or replay.get("graph_sha256") != metadata["graph_sha256"]
                or replay.get("rtol") != 1e-5 or replay.get("atol") != 1e-5):
            raise ValueError("Replay receipt does not bind the complete direct-command episode")
        for name, digest in {**replay["source_sha256"], **replay["graph_sha256"]}.items():
            if sha256_file(name) != digest:
                raise ValueError(f"Replay source or graph changed: {name}")
        if metadata["source_sha256"].get(".work/lab/direct_command_trial/process.py") != sha256_file(__file__):
            raise ValueError("Auditor differs from sealed source")
        entries = [json.loads(line) for line in (evidence / "manifest.jsonl").read_text().splitlines()]
        paths = [evidence / name for name in ["metadata.json", "manifest.jsonl", "completion.json"] + [e["file"] for e in entries]]
        paths += [directory / name for name in ("summary.json", "source.zip", "case.json", "track.json", "realization.json", "scene.usda", "appearance.json", "export.json", "frames.json", "telemetry.json")]
        export_data = json.loads((directory / "export.json").read_text())
        paths += [directory / "assets" / name for name in export_data["dependencies"]]
        expected_files = {str(path.resolve()): sha256_file(path) for path in paths}
        replay_files = {str(Path(path).resolve()): digest for path, digest in replay["evidence_and_artifacts_sha256"].items()}
        if expected_files != replay_files:
            raise ValueError("Replay receipt artifact inventory or hashes differ")
        if replay_sha != sha256_file(replay_receipt):
            raise ValueError("Replay receipt changed during validation")
        report.update(command_contract=contract, controller_name=metadata["controller_name"],
                      replay_receipt_sha256=replay_sha, auditor_sha256=sha256_file(__file__),
                      prediction_provenance="Base model plan and cone outputs are diagnostics; direct command head actuates the car.")
        case = Case(**metadata["case"])
        env = environment(case)
        expected_realization = realized_environment(env)
        realization = json.loads((directory / "realization.json").read_text())
        checks.compare("realization", expected_realization, realization)
        checks.compare(
            "realization.digest", metadata["realization_sha256"], sha256_file(directory / "realization.json")
        )
        checks.compare("track", env.track.to_dict(), json.loads((directory / "track.json").read_text()))
        checks.compare("case", metadata["case"], json.loads((directory / "case.json").read_text()))
        checks.compare("case.summary", metadata["case"], summary["case"])
        checks.compare("backend", "Isaac Lab PhysX", metadata["backend"])
        checks.compare("fps", 20, metadata["fps"])
        checks.compare("schema_version", schema, metadata["schema_version"])
        for key in ("checkpoint_sha256", "state_sha256", "head_state_sha256", "base_parameter_count", "head_parameter_count", "total_parameter_count", "pilot_only"):
            checks.compare("identity." + key, metadata[key], summary[key])
        for key in ("checkpoint_sha256", "state_sha256"):
            checks.compare(key + ".valid", True, bool(re.fullmatch(r"[0-9a-f]{64}", metadata[key])))
        checks.compare("model_unchanged", True, summary["frozen_unchanged"])
        checks.compare("checkpoint_after", metadata["checkpoint_sha256"], summary["checkpoint_sha256_after"])
        checks.compare("state_after", metadata["state_sha256"], summary["state_sha256_after"])
        checks.compare("head_after", metadata["head_state_sha256"], summary["head_state_sha256_after"])
        for key, name in (
            ("appearance_sha256", "appearance.json"),
            ("scene_sha256", "scene.usda"),
            ("export_sha256", "export.json"),
        ):
            checks.compare(key, metadata[key], sha256_file(directory / name))
        appearance = json.loads((directory / "appearance.json").read_text())
        checks.compare("appearance.physics", metadata["physics_sha256"], appearance["physics_sha256"])
        checks.compare("summary.physics", metadata["physics_sha256"], summary["physics_sha256"])
        export = json.loads((directory / "export.json").read_text())
        checks.compare("asset.inventory", sorted(export["dependencies"]), sorted(export["sha256"]))
        for name, digest in export["sha256"].items():
            asset = (directory / "assets" / name).resolve()
            if not asset.is_relative_to((directory / "assets").resolve()):
                raise ValueError("Export asset escapes bundle")
            checks.compare("asset.hash", digest, sha256_file(asset))
        checks.compare("metadata.track", env.track.to_dict(), metadata["track"])
        checks.compare("metadata.expert", asdict(env.expert_params), metadata["expert_params"])
        checks.compare("metadata.vehicle", asdict(env.vp), metadata["vehicle_params"])
        checks.compare(
            "metadata.cones", np.c_[env.cones_xy, np.full(len(env.cones_xy), 0.02)], metadata["initial_cones_xyz"]
        )
        checks.compare("metadata.cone_colors", [c.value for c in env.cone_colors], metadata["cone_colors"])
        rows = json.loads((directory / "frames.json").read_text())
        contacts = summary["contact_events"]
        env.referee = Referee(env.track, env.cones_xy, env.cone_colors, env.vp)
        env.pending_events = []
        x, y, yaw = metadata["start_pose"]
        initial = metadata["initial_physical_state"]
        env.hint, env.s = env.mission.init_progress(initial["x"], initial["y"])
        env.route_commands = RouteCommands.for_mission(env.track, env.mission, env.cfg.fps)
        count, sum_sq = 0, 0.0
        for _, record, _entry in read_evidence(evidence):
            frame = count
            checks.compare("frame", frame, record["frame"], frame)
            checks.compare("state.time", frame / 20, record["state"]["t"], frame)
            checks.compare("running", False, env.referee.done, frame)
            advance(env, record["state"], frame)
            if frame == 0:
                env.frame_desire = env.route_commands.advance(frame=0, s=env.s, speed=env.state.vx)
                checks.compare("initial.state", metadata["initial_physical_state"], record["state"])
                checks.compare("initial.position", True, bool(np.hypot(env.state.x - x, env.state.y - y) < 1e-4))
                checks.compare("initial.yaw", True, bool(abs(np.angle(np.exp(1j * (env.state.yaw - yaw)))) < 1e-5))
            checks.compare("progress", {"s": env.s, "hint": env.hint, "lat": env.lat}, record["progress"], frame)
            requested = env.frame_desire.copy()
            applied = requested.copy()
            mode = metadata["command_mode"]
            if mode == "zero":
                applied[:] = 0
            elif mode == "swapped":
                applied = applied[[0, 2, 1, 3, 4, 5, 6, 7]]
            elif mode != "recorded":
                raise ValueError(f"Unknown command mode: {mode}")
            checks.compare(
                "commands", {"requested": requested, "applied": applied, "mode": mode}, record["commands"], frame
            )
            checks.compare("events", [asdict(e) for e in env.pending_events], record["events"], frame)
            env.pending_events = []
            checks.compare(
                "counters",
                {
                    "laps": len(env.referee.lap_times()),
                    "total_laps": env.track.laps,
                    "legacy_doo": env.referee.n_doo,
                    "off_course": env.referee.n_oc,
                    "result": env.referee.result,
                    "physical_contacts": sum(c["control_frame"] < frame for c in contacts),
                },
                record["counters"],
                frame,
            )
            checks.compare(
                "temporal",
                {"older_image_frame": max(0, frame - 4), "current_image_frame": frame},
                record["temporal"],
                frame,
            )
            checks.compare("calibration", env.T_ego_cam(), record["nominal_detection_calibration"], frame)
            for camera in ("road", "wide"):
                for key in ("images", "raw_images"):
                    image = np.asarray(record[key][camera])
                    checks.compare(
                        key + ".valid", True, image.shape == (256, 512, 3) and image.dtype == np.uint8, frame
                    )
                outputs = (record.get("image_head") or {}).get(camera, {})
                checks.compare("head.keys", ["heat", "off", "wh"], sorted(outputs), frame)
                for value in outputs.values():
                    a = np.asarray(value)
                    checks.compare(
                        "head.finite", True, a.ndim == 4 and a.size > 0 and bool(np.isfinite(a).all()), frame
                    )
            for key, shape in (
                ("actual_cones_xyz", (len(env.cones_xy), 3)),
                ("actual_cone_quaternions", (len(env.cones_xy), 4)),
                ("T_plan_ego_camera", (4, 4)),
                ("T_world_camera_body", (4, 4)),
            ):
                value = np.asarray(record[key])
                checks.compare(key + ".valid", True, value.shape == shape and bool(np.isfinite(value).all()), frame)
            checks.compare("policy_raw.count", 2, len(record["policy_raw"]), frame)
            for tensor in record["policy_raw"]:
                a = np.asarray(tensor)
                checks.compare("policy_raw.finite", True, a.size > 0 and bool(np.isfinite(a).all()), frame)
            plan = np.asarray(record["prediction"]["plan"])
            checks.compare(
                "plan.valid", True, plan.ndim == 2 and min(plan.shape) >= 2 and bool(np.isfinite(plan).all()), frame
            )
            action = np.asarray(record["action"])
            checks.compare("action.valid", True, action.shape == (2,) and bool(np.isfinite(action).all()), frame)
            from fsop.eval.closed_loop import mission_brake_with_override
            ctrl = record["prediction"]["controller"]
            checks.compare("controller.kind", "direct_native_actuator_v1", ctrl["kind"], frame)
            checks.compare("controller.proprio", [env.state.vx, env.state.vy, env.state.r], ctrl["proprio"], frame)
            raw = np.asarray(ctrl["raw"])
            checks.compare("controller.raw.valid", True, raw.shape == (2,) and bool(np.isfinite(raw).all()), frame)
            bounded = [float(np.clip(raw[0], -env.vp.max_steer, env.vp.max_steer)),
                       float(np.clip(raw[1], -env.vp.max_brake, env.vp.max_accel))]
            accel, override = mission_brake_with_override(bounded[1], SimpleNamespace(state=env.state, s=env.s), env)
            applied = [bounded[0], float(np.clip(accel, -env.vp.max_brake, env.vp.max_accel))]
            checks.compare("controller.bounded", bounded, ctrl["bounded"], frame)
            checks.compare("controller.override", override, ctrl["mission_override"], frame)
            checks.compare("controller.applied", applied, ctrl["applied"], frame)
            checks.compare("controller.action", applied, action, frame)
            body = record["body"]
            rot, velocity = np.asarray(body["rotation"]), np.asarray(body["velocity_world"])
            a = np.arctan2(rot[1, 0], rot[0, 0])
            c, s = np.cos(a), np.sin(a)
            planar = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ velocity
            checks.compare(
                "body.state",
                [*np.asarray(body["position"])[:2], a, *planar[:2], np.asarray(body["angular_velocity_world"])[2]],
                [env.state.x, env.state.y, env.state.yaw, env.state.vx, env.state.vy, env.state.r],
                frame,
            )
            checks.compare("body.velocity", rot.T @ velocity, body["velocity_body"], frame)
            if frame < len(rows):
                expected = {
                    "frame": frame,
                    "t": env.state.t,
                    "x": env.state.x,
                    "y": env.state.y,
                    "yaw": env.state.yaw,
                    "speed": env.state.vx,
                    "s": env.s,
                    "lat": env.lat,
                    "legacy_doo": env.referee.n_doo,
                    "physical_contacts": record["counters"]["physical_contacts"],
                }
                checks.compare("frames", expected, {key: rows[frame].get(key) for key in expected}, frame)
            sum_sq += env.lat**2
            count += 1
        checks.compare("nonempty", True, count > 0)
        checks.compare("bounded", True, count <= 2400)
        checks.compare("frame_count", count, summary["recorded_frames"])
        checks.compare("rows.count", count, len(rows))
        checks.compare("receipt.count", count, receipt["frame_count"])
        checks.compare("terminal.time", count / 20, summary["final_state"]["t"])
        advance(env, summary["final_state"], count)
        if env.state.t >= 120 and not env.referee.done:
            env.referee.finish("timeout")
        terminal_verified = summary["result"] != "isaac_rollover"
        if not terminal_verified and not env.referee.done:
            env.referee.finish("isaac_rollover")
        checks.compare("finished_or_pilot", True, env.referee.done or summary["pilot_only"])
        computed = env.referee.summary()
        checks.compare("referee.summary", computed, {key: summary.get(key) for key in computed})
        checks.compare("referee.events", [asdict(e) for e in env.referee.events], summary["events"])
        checks.compare("pending.events", [asdict(e) for e in env.pending_events], summary["pending_events"])
        checks.compare("summary.progress", env.s / env.mission.s_end, summary["progress"])
        checks.compare("summary.rms_lat", np.sqrt(sum_sq / max(count, 1)), summary["rms_lat"])
        checks.compare("contacts.count", len(contacts), summary["physical_cone_hits"])
        checks.compare("contacts.unique", len(contacts), len({c["cone"] for c in contacts}))
        cone_paths = {f"/World/Cones/c{i:04d}" for i in range(len(env.cones_xy))}
        for contact in contacts:
            f = contact["control_frame"]
            checks.compare("contacts.frame", True, isinstance(f, int) and 0 <= f < count)
            checks.compare("contacts.time", f / 20, contact["t"])
            checks.compare("contacts.path", True, contact["cone"] in cone_paths)
            checks.compare(
                "contacts.convention",
                "start of the control interval containing first contact",
                contact["time_convention"],
            )
        report.update(
            frame_count=count,
            terminal_condition_verified=terminal_verified,
            recomputed_referee_summary=computed,
            completion_sha256=sha256_file(evidence / "completion.json"),
            frames_sha256=sha256_file(directory / "frames.json"),
            telemetry_sha256=sha256_file(directory / "telemetry.json"),
        )
        if (replay_sha != sha256_file(replay_receipt)
                or expected_files != {str(path.resolve()): sha256_file(path) for path in paths}
                or any(sha256_file(name) != digest for name, digest in {**replay["source_sha256"], **replay["graph_sha256"]}.items())):
            raise ValueError("Bound replay inputs changed during artifact audit")
        report["passed"] = not any(row["mismatches"] for row in checks.fields.values())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report.update(comparisons=checks.fields, mismatches=checks.failures)
    with (directory / "lab_artifact_audit.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--replay-receipt", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_episode(args.episode, args.replay_receipt)
    print(json.dumps({"passed": audit["passed"], "frames": audit.get("frame_count")}))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
