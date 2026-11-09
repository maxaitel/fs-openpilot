"""Independently audit and render one sealed native Lab evaluation episode."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fsop.sim.referee import Referee
from fsop.sim.route_commands import RouteCommands
from isaac_eval.evidence import read_evidence, verified_summary
from isaac_eval.parity_cases import realized_environment, sha256_file
from isaac_lab_fsop.cases import Case, environment
from scripts.audit_isaac_episode import Checks, advance


def audit_episode(directory):
    directory = Path(directory)
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
        checks.compare("schema_version", 1, metadata["schema_version"])
        for key in ("checkpoint_sha256", "state_sha256", "pilot_only"):
            checks.compare("identity." + key, metadata[key], summary[key])
        for key in ("checkpoint_sha256", "state_sha256"):
            checks.compare(key + ".valid", True, bool(re.fullmatch(r"[0-9a-f]{64}", metadata[key])))
        checks.compare("model_unchanged", True, summary["frozen_unchanged"])
        checks.compare("checkpoint_after", metadata["checkpoint_sha256"], summary["checkpoint_sha256_after"])
        checks.compare("state_after", metadata["state_sha256"], summary["state_sha256_after"])
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
        paths = {f"/World/Cones/c{i:04d}" for i in range(len(env.cones_xy))}
        for contact in contacts:
            f = contact["control_frame"]
            checks.compare("contacts.frame", True, isinstance(f, int) and 0 <= f < count)
            checks.compare("contacts.time", f / 20, contact["t"])
            checks.compare("contacts.path", True, contact["cone"] in paths)
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
        report["passed"] = not any(row["mismatches"] for row in checks.fields.values())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report.update(comparisons=checks.fields, mismatches=checks.failures)
    (directory / "lab_artifact_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--rrd", action="store_true", help="Also export an offline Rerun recording")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    audit = audit_episode(args.episode)
    if not audit["passed"]:
        raise SystemExit(json.dumps(audit))
    if not args.audit_only:
        import cv2

        from isaac_eval.render_debug import export_rrd, render

        cv2.setNumThreads(1)
        result = render(args.episode / "evidence", args.episode / "debug.mp4", args.episode / "research.mp4")
        if result != {"inference_frames": audit["frame_count"], "terminal_frames": 40}:
            raise ValueError("Video frame count mismatch")
        result["completion_sha256"] = sha256_file(args.episode / "evidence/completion.json")
        result["artifact_audit_sha256"] = sha256_file(args.episode / "lab_artifact_audit.json")
        result["videos"] = {}
        for name in ("debug", "research"):
            path = args.episode / f"{name}.mp4"
            encoded = args.episode / f"{name}-h264.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "1",
                    "-movflags",
                    "+faststart",
                    str(encoded),
                ],
                check=True,
            )
            encoded.replace(path)
            capture = cv2.VideoCapture(str(path))
            fps, decoded = capture.get(cv2.CAP_PROP_FPS), 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape != (768, 1280, 3):
                    raise ValueError(f"Unexpected video dimensions: {path}")
                decoded += 1
            capture.release()
            if fps != 20 or decoded != audit["frame_count"] + 40:
                raise ValueError(f"Video decode or cadence mismatch: {path}")
            result["videos"][name] = {"sha256": sha256_file(path), "fps": fps, "decoded_frames": decoded}
        if args.rrd:
            result["rrd_frames"] = export_rrd(args.episode / "evidence", args.episode / "debug.rrd")
            if result["rrd_frames"] != audit["frame_count"]:
                raise ValueError("Rerun frame count mismatch")
        (args.episode / "lab_video_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "frames": audit["frame_count"]}))


if __name__ == "__main__":
    main()
