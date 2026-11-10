import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from fsop.eval.closed_loop import mission_brake_with_override
from fsop.eval.controller import controller_contract, make_controller
from fsop.sim.vehicle import VehicleState
from isaac_eval.evidence import EvidenceWriter, finalize_evidence
from isaac_eval.parity_cases import realized_environment, sha256_file
from isaac_lab_fsop.cases import Case, environment
from isaac_lab_fsop.process import audit_episode


def write_episode(path, *, tamper=None, schema=3):
    case = Case(id="pi-test", split="val", discipline="skidpad", seed=0)
    env = environment(case)
    contract = controller_contract("op_tangent_pi_v1")
    x, y, yaw = env.track.start_pose
    source = {p: sha256_file(p) for p in ("src/fsop/eval/controller.py", "src/fsop/eval/closed_loop.py",
                                             "isaac_lab_fsop/process.py")}
    for name, content in (("realization.json", realized_environment(env)), ("track.json", env.track.to_dict()),
                          ("case.json", asdict(case)), ("appearance.json", {"physics_sha256": "physics"}),
                          ("export.json", {"dependencies": [], "sha256": {}}), ("telemetry.json", [])):
        (path / name).write_text(json.dumps(content))
    (path / "scene.usda").write_text("scene")
    metadata = {
        "schema_version": schema, "controller_contract": contract, "fps": 20, "backend": "Isaac Lab PhysX",
        "case": asdict(case), "checkpoint_sha256": "a" * 64, "state_sha256": "b" * 64,
        "pilot_only": True, "source_sha256": source, "track": env.track.to_dict(),
        "expert_params": asdict(env.expert_params), "vehicle_params": asdict(env.vp),
        "initial_cones_xyz": np.c_[env.cones_xy, np.full(len(env.cones_xy), .02)].tolist(),
        "cone_colors": [c.value for c in env.cone_colors], "start_pose": [x, y, yaw],
        "physics_sha256": "physics", "command_mode": "recorded",
        **{field: sha256_file(path / name) for field, name in (
            ("realization_sha256", "realization.json"), ("appearance_sha256", "appearance.json"),
            ("scene_sha256", "scene.usda"), ("export_sha256", "export.json"))},
    }
    ctrl = make_controller("op_tangent_pi_v1", env.vp)
    plan = np.zeros((33, 15))
    plan[:, 3] = 4.
    records, rows = [], []
    speeds = [0., 3., 3., 3., 1.]
    for frame, speed in enumerate(speeds + [1.]):
        state = VehicleState(x + frame * .15 * np.cos(yaw), y + frame * .15 * np.sin(yaw), yaw, vx=speed,
                             t=frame / 20)
        env.state = state
        env.hint, env.s, env.lat = env.mission.update_progress(state.x, state.y, env.hint, env.s)
        if frame:
            events = env.referee.update(state, env.mission_done())
            env.frame_desire = env.route_commands.advance(frame=frame, s=env.s, speed=speed)
        else:
            events = []
            metadata["initial_physical_state"] = asdict(state)
        if frame == len(speeds):
            break
        obs = SimpleNamespace(state=state, s=env.s)
        steer, accel, diag = ctrl.step(plan, speed, 0., state.t,
                                      lambda command: mission_brake_with_override(command, obs, env))
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        record = {"frame": frame, "state": asdict(state), "progress": {"s": env.s, "lat": env.lat, "hint": env.hint},
                  "commands": {"requested": env.frame_desire.copy(), "applied": env.frame_desire.copy(), "mode": "recorded"},
                  "counters": {"laps": len(env.referee.lap_times()), "total_laps": env.track.laps,
                               "legacy_doo": env.referee.n_doo, "off_course": env.referee.n_oc,
                               "physical_contacts": 0, "result": env.referee.result},
                  "events": [asdict(event) for event in events],
                  "temporal": {"older_image_frame": max(0, frame - 4), "current_image_frame": frame},
                  "nominal_detection_calibration": env.T_ego_cam(), "action": [steer, accel],
                  "prediction": {"plan": plan.copy(), "controller": diag},
                  "images": {cam: np.zeros((256, 512, 3), np.uint8) for cam in ("road", "wide")},
                  "raw_images": {cam: np.zeros((256, 512, 3), np.uint8) for cam in ("road", "wide")},
                  "image_head": {cam: {key: np.zeros((1, 2, 2, 2)) for key in ("heat", "wh", "off")}
                                 for cam in ("road", "wide")},
                  "actual_cones_xyz": np.c_[env.cones_xy, np.full(len(env.cones_xy), .02)],
                  "actual_cone_quaternions": np.ones((len(env.cones_xy), 4)),
                  "T_plan_ego_camera": np.eye(4), "T_world_camera_body": np.eye(4),
                  "policy_raw": [np.zeros(1), np.zeros(1)],
                  "body": {"position": np.array([state.x, state.y, .36]), "rotation": rot,
                           "velocity_world": rot @ np.array([speed, 0., 0.]), "velocity_body": np.array([speed, 0., 0.]),
                           "angular_velocity_world": np.zeros(3)}}
        records.append(record)
        rows.append({"frame": frame, "t": state.t, "x": state.x, "y": state.y, "yaw": state.yaw,
                     "speed": speed, "s": env.s, "lat": env.lat, "legacy_doo": env.referee.n_doo,
                     "physical_contacts": 0})
    assert records[3]["prediction"]["controller"]["integral_before"] > 0
    assert records[4]["prediction"]["controller"]["mission_override"]
    assert records[4]["prediction"]["controller"]["integral_after"] == 0
    if tamper:
        tamper(records)
    summary = {**env.referee.summary(), "case": asdict(case), "controller_contract": contract,
               "pilot_only": True, "checkpoint_sha256": metadata["checkpoint_sha256"],
               "checkpoint_sha256_after": metadata["checkpoint_sha256"], "state_sha256": metadata["state_sha256"],
               "state_sha256_after": metadata["state_sha256"], "physics_sha256": "physics", "frozen_unchanged": True,
               "final_state": asdict(state), "events": [asdict(event) for event in env.referee.events],
               "pending_events": [asdict(event) for event in events], "recorded_frames": len(records),
               "physical_cone_hits": 0, "contact_events": [], "progress": env.s / env.mission.s_end,
               "rms_lat": float(np.sqrt(np.mean([row["lat"] ** 2 for row in rows])))}
    (path / "frames.json").write_text(json.dumps(rows))
    (path / "summary.json").write_text(json.dumps(summary))
    with EvidenceWriter(path / "evidence", metadata) as writer:
        for record in records:
            writer.append(record)
    finalize_evidence(path / "evidence", path / "summary.json")
    return records


def test_pi_audit_replays_state_and_mission_override(tmp_path):
    write_episode(tmp_path)
    result = audit_episode(tmp_path)
    assert result["passed"], result.get("error", result["mismatches"])
    assert result["comparisons"]["controller.diagnostics.integral_before"]["comparisons"] == 5


@pytest.mark.parametrize("tamper", [
    lambda records: records[2]["prediction"]["controller"].update(integral_after=0.),
    lambda records: records[3]["action"].__setitem__(1, 0.),
    lambda records: records[4]["prediction"]["controller"].update(mission_override=False),
    lambda records: records[3]["prediction"].pop("controller"),
])
def test_pi_audit_rejects_resealed_controller_tampering(tmp_path, tamper):
    write_episode(tmp_path, tamper=tamper)
    assert not audit_episode(tmp_path)["passed"]


def test_pi_audit_rejects_schema_mismatch(tmp_path):
    write_episode(tmp_path, schema=2)
    result = audit_episode(tmp_path)
    assert not result["passed"]
    assert "schema does not match" in result["error"]
