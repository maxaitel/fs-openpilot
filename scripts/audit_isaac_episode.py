"""Audit sealed Isaac evidence on CPU without loading a model or checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fsop.geometry.camera import CAMERA_MODELS
from fsop.sim.vehicle import VehicleState
from isaac_eval.evidence import read_evidence, verified_summary
from isaac_eval.parity_cases import (
    CHECKPOINT_SHA256,
    all_cases,
    build_environment,
    case_manifest,
    get_case,
    json_sha256,
    realized_environment,
    sha256_file,
)
from isaac_eval.parity_protocol import GRAPH_FILES

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-8
RTOL = 1e-9


@dataclass
class Checks:
    fields: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def compare(self, name, expected, actual, frame=None):
        if isinstance(expected, dict) and isinstance(actual, dict):
            self.compare(name + ".keys", sorted(expected), sorted(actual), frame)
            for key in expected.keys() & actual.keys():
                self.compare(f"{name}.{key}", expected[key], actual[key], frame)
            return
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            self.compare(name + ".length", len(expected), len(actual), frame)
            for a, b in zip(expected, actual, strict=False):
                self.compare(name + "[]", a, b, frame)
            return
        row = self.fields.setdefault(name, {"comparisons": 0, "mismatches": 0, "max_abs": 0.0})
        row["comparisons"] += 1
        a, b = np.asarray(expected), np.asarray(actual)
        if a.dtype.kind in "biuf" and b.dtype.kind in "biuf" and a.shape == b.shape:
            finite = np.isfinite(a).all() and np.isfinite(b).all()
            if finite:
                error = float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0))
                row["max_abs"] = max(row["max_abs"], error)
            exact = a.dtype.kind in "biu" or b.dtype.kind in "biu"
            matches = finite and (np.array_equal(a, b) if exact else np.allclose(a, b, atol=ATOL, rtol=RTOL))
        else:
            matches = a.shape == b.shape and np.array_equal(a, b)
        if not matches:
            row["mismatches"] += 1
            if len(self.failures) < 50:
                self.failures.append({"field": name, "frame": frame})


def advance(env, state, frame):
    env.state = VehicleState(**state)
    if not np.isfinite(env.state.as_array()).all():
        raise ValueError(f"Nonfinite vehicle state at frame {frame}")
    env.hint, env.s, env.lat = env.mission.update_progress(env.state.x, env.state.y, env.hint, env.s)
    env.frame = frame
    if frame:
        env.pending_events.extend(env.referee.update(env.state, env.mission_done()))
        env.frame_desire = env.route_commands.advance(frame=frame, s=env.s, speed=env.state.vx)


def replay(records, env, summary, rows, checks):
    """Recompute progress and scoring from input states, never saved progress or counters."""
    count, sum_lat_squared = 0, 0.0
    contacts = summary["contact_events"]
    mode = summary["case"]["command_mode"]
    for metadata, record, entry in records:
        frame = count
        checks.compare("frame", frame, record["frame"], frame)
        checks.compare("manifest.frame", frame, entry["frame"], frame)
        checks.compare("state.t", frame / 20, record["state"]["t"], frame)
        checks.compare("referee.running_before_input", False, env.referee.done, frame)
        advance(env, record["state"], frame)
        checks.compare("referee.running_at_input", False, env.referee.done, frame)
        if frame == 0:
            checks.compare("initial_physical_state", metadata["initial_physical_state"], record["state"])
            x, y, yaw = env.track.start_pose
            checks.compare("initial_position", True, bool(np.hypot(env.state.x - x, env.state.y - y) < 1e-4))
            checks.compare("initial_yaw", True, bool(abs(np.angle(np.exp(1j * (env.state.yaw - yaw)))) < 1e-5))
            checks.compare("initial_velocity", [0., 0., 0.], [env.state.vx, env.state.vy, env.state.r])
        checks.compare("progress", {"s": env.s, "lat": env.lat, "hint": env.hint}, record["progress"], frame)
        requested = env.frame_desire.copy()
        applied = requested.copy()
        if mode == "zero":
            applied[:] = 0
        elif mode == "swapped":
            applied = applied[[0, 2, 1, 3, 4, 5, 6, 7]]
        checks.compare("commands", {"requested": requested, "applied": applied, "mode": mode},
                       record["commands"], frame)
        checks.compare("counters", {"laps": len(env.referee.lap_times()), "total_laps": env.track.laps,
                                    "legacy_doo": env.referee.n_doo, "off_course": env.referee.n_oc,
                                    "result": env.referee.result,
                                    "physical_contacts": sum(c["control_frame"] < frame for c in contacts)},
                       record["counters"], frame)
        checks.compare("events", [asdict(event) for event in env.pending_events], record["events"], frame)
        env.pending_events = []
        checks.compare("temporal", {"older_image_frame": max(0, frame - 4), "current_image_frame": frame},
                       record["temporal"], frame)
        checks.compare("nominal_detection_calibration", env.T_ego_cam(),
                       record["nominal_detection_calibration"], frame)
        plan = np.asarray(record["prediction"]["plan"])
        checks.compare("prediction.plan_shape", True, plan.ndim == 2 and min(plan.shape) >= 2, frame)
        checks.compare("prediction.plan_finite", True, bool(np.isfinite(plan).all()), frame)
        action = np.asarray(record["action"])
        checks.compare("action_valid", True, action.shape == (2,) and bool(np.isfinite(action).all()), frame)
        head = record.get("image_head")
        checks.compare("image_head.cameras", ["road", "wide"], sorted(head) if isinstance(head, dict) else [], frame)
        for camera in ("road", "wide"):
            image = np.asarray(record["images"][camera])
            checks.compare("images.valid", True, image.shape == (256, 512, 3) and image.dtype == np.uint8, frame)
            outputs = head.get(camera, {}) if isinstance(head, dict) else {}
            checks.compare("image_head.outputs", ["heat", "off", "wh"], sorted(outputs), frame)
            for value in outputs.values():
                array = np.asarray(value)
                checks.compare("image_head.valid", True,
                               array.ndim == 4 and array.size > 0 and bool(np.isfinite(array).all()), frame)
        body = record["body"]
        rotation = np.asarray(body["rotation"])
        if frame:
            checks.compare("body.no_prior_rollover", True, bool(rotation[2, 2] >= 0.3), frame)
        velocity = np.asarray(body["velocity_world"])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        planar_velocity = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ velocity
        checks.compare("body.state", [*np.asarray(body["position"])[:2], yaw, *planar_velocity[:2],
                                      np.asarray(body["angular_velocity_world"])[2]],
                       [env.state.x, env.state.y, env.state.yaw, env.state.vx, env.state.vy, env.state.r], frame)
        checks.compare("body.velocity_body", rotation.T @ velocity, body["velocity_body"], frame)
        if frame < len(rows):
            expected = {"frame": frame, "t": env.state.t, "x": env.state.x, "y": env.state.y,
                        "yaw": env.state.yaw, "speed": env.state.vx, "s": env.s, "lat": env.lat,
                        "legacy_doo": env.referee.n_doo,
                        "physical_contacts": record["counters"]["physical_contacts"]}
            checks.compare("frames", expected, {key: rows[frame].get(key) for key in expected}, frame)
        sum_lat_squared += env.lat ** 2
        count += 1
    checks.compare("nonempty", True, count > 0)
    checks.compare("within_timeout", True, count <= 2400)
    checks.compare("recorded_frames", count, summary["recorded_frames"])
    checks.compare("frames.length", count, len(rows))
    checks.compare("final_state.t", count / 20, summary["final_state"]["t"])
    advance(env, summary["final_state"], count)
    terminal_verified = True
    if env.state.t >= 120 and not env.referee.done:
        env.referee.finish("timeout")
    if summary["result"] == "isaac_rollover" and not env.referee.done:
        terminal_verified = False
        env.referee.finish("isaac_rollover")
    checks.compare("terminal.finished_or_pilot", True, env.referee.done or summary["pilot_only"])
    computed = env.referee.summary()
    checks.compare("summary.referee", computed, {key: summary.get(key) for key in computed})
    checks.compare("summary.events", [asdict(event) for event in env.referee.events], summary["events"])
    checks.compare("summary.pending_events", [asdict(event) for event in env.pending_events], summary["pending_events"])
    checks.compare("summary.progress", env.s / env.mission.s_end, summary["progress"])
    checks.compare("summary.rms_lat", np.sqrt(sum_lat_squared / max(count, 1)), summary["rms_lat"])
    checks.compare("summary.seconds_simulated", count / 20, summary["seconds_simulated"])
    checks.compare("contacts.count", len(contacts), summary["physical_cone_hits"])
    checks.compare("contacts.unique", len(contacts), len({event["cone"] for event in contacts}))
    for contact in contacts:
        checks.compare("contacts.valid_frame", True,
                       isinstance(contact["control_frame"], int) and 0 <= contact["control_frame"] < count)
        checks.compare("contacts.time", contact["control_frame"] / 20, contact["t"])
        checks.compare("contacts.time_convention", "start of the control interval containing first contact",
                       contact["time_convention"])
        checks.compare("contacts.cone", True, contact["cone"] in
                       {f"/World/Cones/c{i:04d}" for i in range(len(env.cones_xy))})
    return count, terminal_verified, computed


def audit_episode(episode):
    episode = Path(episode).resolve()
    evidence = episode / "evidence"
    checks = Checks()
    report = {"passed": False, "frame_count": 0, "completion_sha256": None, "atol": ATOL, "rtol": RTOL,
              "verification_limits": [
                  "Saved states replay original mission, route commands and referee. Model outputs are not rerun.",
                  "Checkpoint and model-state hashes check metadata consistency, not the actual weights.",
                  "Physical contact counts agree with saved contact events but are not independently simulated.",
                  "Final planar state is receipt-bound. The final physics transition is not reconstructed.",
              ], "source_sha256": {"scripts/audit_isaac_episode.py": sha256_file(__file__)}}
    try:
        report["completion_sha256"] = sha256_file(evidence / "completion.json")
        summary = verified_summary(evidence)
        if summary is None:
            raise ValueError("A sealed completion receipt is required, including for pilots")
        receipt = json.loads((evidence / "completion.json").read_text())
        if (evidence / receipt["summary_path"]).resolve() != episode / "summary.json":
            raise ValueError("Completion receipt must bind this episode's summary.json")
        metadata = json.loads((evidence / "metadata.json").read_text())
        case = get_case(metadata["case"]["case_id"])
        report["case_id"] = case.case_id
        checks.compare("metadata.case", case_manifest(case), metadata["case"])
        for key in ("case", "pilot_only", "checkpoint_sha256", "state_sha256", "manifest_sha256", "protocol_sha256"):
            checks.compare("identity." + key, metadata[key], summary[key])
        checks.compare("metadata.checkpoint_sha256", CHECKPOINT_SHA256, metadata["checkpoint_sha256"])
        for key in ("state_sha256", "manifest_sha256"):
            checks.compare("metadata." + key + ".valid", True, bool(re.fullmatch("[a-f0-9]{64}", metadata[key])))
        checks.compare("metadata.fps", 20, metadata["fps"])
        checks.compare("metadata.backend", "Isaac PhysX", metadata["backend"])
        checks.compare("metadata.schema_version", 1, metadata["schema_version"])
        checks.compare("metadata.graph_inventory", sorted(GRAPH_FILES), sorted(metadata["graph_sha256"]))
        for digest in metadata["graph_sha256"].values():
            checks.compare("metadata.graph_digest_valid", True, bool(re.fullmatch("[a-f0-9]{64}", digest)))
        checks.compare("summary.frozen_unchanged", True, summary["frozen_unchanged"])
        paths = sorted((ROOT / "src/fsop").rglob("*.py")) + sorted((ROOT / "isaac_eval").glob("*.py"))
        sources = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
        report["source_sha256"].update(sources)
        checks.compare("metadata.source_sha256", sources, metadata["source_sha256"])
        if metadata["pilot_only"]:
            checks.compare("metadata.protocol_sha256", None, metadata["protocol_sha256"])
        else:
            protocol_path = episode / "protocol.json"
            protocol = json.loads(protocol_path.read_text())
            checks.compare("protocol.file_sha256", sha256_file(protocol_path), metadata["protocol_sha256"])
            digest = json_sha256({k: v for k, v in protocol.items() if k != "protocol_sha256"})
            checks.compare("protocol.digest", digest,
                           protocol["protocol_sha256"])
            checks.compare("protocol.case_ids", [case.case_id for case in all_cases()], protocol["case_ids"])
            for key in ("manifest_sha256", "source_sha256", "graph_sha256"):
                checks.compare("protocol." + key, protocol[key], metadata[key])
            checks.compare("protocol.render_profile", protocol["render_profile"], summary["render_profile"])
        env = build_environment(case)
        saved_realization = json.loads((episode / "realization.json").read_text())
        saved_track = json.loads((episode / "track.json").read_text())
        checks.compare("realization.digest", json_sha256({k: v for k, v in saved_realization.items()
                                                         if k != "realization_sha256"}),
                       saved_realization["realization_sha256"])
        checks.compare("realization.track_digest", json_sha256(saved_track), saved_realization["track_sha256"])
        digest_keys = {"realization_sha256", "track_sha256"}
        checks.compare("realization", {k: v for k, v in realized_environment(env).items() if k not in digest_keys},
                       {k: v for k, v in saved_realization.items() if k not in digest_keys})
        expected = {"track": env.track.to_dict(), "expert_params": asdict(env.expert_params),
                    "vehicle_params": asdict(env.vp), "label_dt": env.cfg.label_dt,
                    "nominal_detection_calibration": env.T_ego_cam().tolist(),
                    "initial_cones_xyz": np.c_[env.cones_xy, np.full(len(env.cones_xy), 0.02)].tolist(),
                    "cone_colors": [color.value for color in env.cone_colors],
                    "intrinsics": {name: cam.K.tolist() for name, cam in CAMERA_MODELS.items()}}
        checks.compare("metadata.environment", expected, {key: metadata.get(key) for key in expected})
        checks.compare("track", env.track.to_dict(), saved_track)
        rows = json.loads((episode / "frames.json").read_text())
        count, terminal_verified, computed = replay(read_evidence(evidence), env, summary, rows, checks)
        checks.compare("receipt.frame_count", count, receipt["frame_count"])
        checks.compare("npz.frame_count", count, len(list(evidence.glob("frame_*.npz"))))
        checks.compare("completion.unchanged", report["completion_sha256"], sha256_file(evidence / "completion.json"))
        verified_summary(evidence)
        report.update(frame_count=count, terminal_condition_verified=terminal_verified,
                      recomputed_referee_summary=computed,
                      summary_sha256=sha256_file(episode / "summary.json"),
                      metadata_sha256=sha256_file(evidence / "metadata.json"),
                      manifest_sha256=sha256_file(evidence / "manifest.jsonl"))
        if not terminal_verified:
            report["verification_limits"].append(
                "Isaac rollover is an unverified backend override; the final full body transform was not saved.")
        report["passed"] = not any(row["mismatches"] for row in checks.fields.values())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report.update(comparisons=checks.fields, mismatches=checks.failures,
                  max_errors={name: row["max_abs"] for name, row in checks.fields.items()})
    (episode / "artifact_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    args = parser.parse_args()
    report = audit_episode(args.episode)
    print(json.dumps({key: report.get(key) for key in ("passed", "case_id", "frame_count", "error", "mismatches")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
