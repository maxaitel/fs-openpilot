import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from fsop.geometry.camera import CAMERA_MODELS
from fsop.sim.vehicle import VehicleState
from isaac_eval.evidence import EvidenceWriter, finalize_evidence
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
from isaac_eval.parity_protocol import GRAPH_FILES, runtime_sources
from scripts.audit_isaac_episode import Checks, audit_episode, replay


@pytest.fixture(scope="module")
def episode_data():
    case = get_case("skidpad-72000-light-recorded")
    env = build_environment(case)
    realization = realized_environment(env)
    track = env.track.to_dict()
    metadata = {"schema_version": 1, "fps": 20, "backend": "Isaac PhysX", "case": case_manifest(case),
                "checkpoint_sha256": CHECKPOINT_SHA256, "state_sha256": "a" * 64,
                "manifest_sha256": "b" * 64, "protocol_sha256": None, "pilot_only": True,
                "source_sha256": runtime_sources(), "track": track,
                "graph_sha256": dict.fromkeys(GRAPH_FILES, "d" * 64),
                "nominal_detection_calibration": env.T_ego_cam().tolist(),
                "initial_cones_xyz": np.c_[env.cones_xy, np.full(len(env.cones_xy), 0.02)].tolist(),
                "cone_colors": [color.value for color in env.cone_colors],
                "expert_params": asdict(env.expert_params), "vehicle_params": asdict(env.vp),
                "label_dt": env.cfg.label_dt,
                "intrinsics": {name: cam.K.tolist() for name, cam in CAMERA_MODELS.items()}}
    records, rows = [], []
    start_x, start_y, start_yaw = env.track.start_pose
    for frame in range(5):
        state = VehicleState(start_x + frame * 0.15, start_y, start_yaw, vx=3. if frame else 0., t=frame / 20)
        env.state = state
        env.hint, env.s, env.lat = env.mission.update_progress(state.x, state.y, env.hint, env.s)
        if frame:
            events = env.referee.update(state, env.mission_done())
            env.frame_desire = env.route_commands.advance(frame=frame, s=env.s, speed=state.vx)
        else:
            events = []
            metadata["initial_physical_state"] = asdict(state)
        if frame == 4:
            break
        record = {"frame": frame, "state": asdict(state), "progress": {"s": env.s, "lat": env.lat, "hint": env.hint},
                  "commands": {"requested": env.frame_desire.copy(), "applied": env.frame_desire.copy(),
                               "mode": "recorded"},
                  "counters": {"laps": len(env.referee.lap_times()), "total_laps": env.track.laps,
                               "legacy_doo": env.referee.n_doo, "off_course": env.referee.n_oc,
                               "physical_contacts": 0, "result": env.referee.result},
                  "events": [asdict(event) for event in events],
                  "temporal": {"older_image_frame": max(0, frame - 4), "current_image_frame": frame},
                  "nominal_detection_calibration": env.T_ego_cam(), "action": [0., 0.],
                  "prediction": {"plan": np.zeros((33, 3))},
                  "images": {camera: np.zeros((256, 512, 3), np.uint8) for camera in ("road", "wide")},
                  "image_head": {camera: {key: np.zeros((1, 2, 2, 2)) for key in ("heat", "wh", "off")}
                                 for camera in ("road", "wide")},
                  "body": {"position": np.array([state.x, state.y, 0.36]), "rotation": np.eye(3),
                           "velocity_world": np.array([state.vx, 0., 0.]),
                           "velocity_body": np.array([state.vx, 0., 0.]),
                           "angular_velocity_world": np.zeros(3)}}
        records.append(record)
        rows.append({"frame": frame, "t": state.t, "x": state.x, "y": state.y, "yaw": state.yaw,
                     "speed": state.vx, "s": env.s, "lat": env.lat, "legacy_doo": env.referee.n_doo,
                     "physical_contacts": 0})
    summary = {**env.referee.summary(), **{key: metadata[key] for key in
               ("case", "pilot_only", "checkpoint_sha256", "state_sha256", "manifest_sha256", "protocol_sha256")},
               "final_state": asdict(state), "events": [asdict(event) for event in env.referee.events],
               "pending_events": [asdict(event) for event in events], "recorded_frames": len(records),
               "physical_cone_hits": 0, "contact_events": [], "progress": env.s / env.mission.s_end,
               "rms_lat": float(np.sqrt(np.mean([row["lat"] ** 2 for row in rows]))),
               "seconds_simulated": state.t, "frozen_unchanged": True}
    return metadata, records, rows, summary, realization, track


def write_episode(path, data, mutation=None):
    metadata, records, rows, summary, realization, track = copy.deepcopy(data)
    if mutation:
        mutation(metadata, records, summary)
    with EvidenceWriter(path / "evidence", metadata) as writer:
        for record in records:
            writer.append(record)
    for name, value in (("summary", summary), ("frames", rows), ("realization", realization), ("track", track)):
        (path / f"{name}.json").write_text(json.dumps(value))
    finalize_evidence(path / "evidence", path / "summary.json")


def test_sealed_pilot_reproduces_original_metrics_and_zero_first_referee_update(tmp_path, episode_data):
    write_episode(tmp_path, episode_data)
    report = audit_episode(tmp_path)
    assert report["passed"], report.get("error", report["mismatches"])
    assert report["frame_count"] == 4
    assert report["terminal_condition_verified"]
    assert report["recomputed_referee_summary"]["result"] == "running"
    assert len(report["completion_sha256"]) == 64
    assert json.loads((tmp_path / "artifact_audit.json").read_text())["passed"]


@pytest.mark.parametrize(("mutation", "field"), [
    (lambda m, r, s: r[1]["progress"].update(s=999.), "progress.s"),
    (lambda m, r, s: r[1]["counters"].update(legacy_doo=7), "counters.legacy_doo"),
    (lambda m, r, s: r[1]["commands"]["applied"].fill(1), "commands.applied"),
    (lambda m, r, s: r[1]["temporal"].update(older_image_frame=1), "temporal.older_image_frame"),
    (lambda m, r, s: r[1].update(image_head=None), "image_head.cameras.length"),
    (lambda m, r, s: r[1]["prediction"]["plan"].fill(float("nan")), "prediction.plan_finite"),
    (lambda m, r, s: s.update(doo=9), "summary.referee.doo"),
    (lambda m, r, s: s["final_state"].update(t=1.), "final_state.t"),
    (lambda m, r, s: s.update(state_sha256="c" * 64), "identity.state_sha256"),
])
def test_resealed_incorrect_evidence_fails_semantic_verification(tmp_path, episode_data, mutation, field):
    write_episode(tmp_path, episode_data, mutation)
    report = audit_episode(tmp_path)
    assert not report["passed"]
    assert report["comparisons"][field]["mismatches"] > 0


def test_changed_frame_and_missing_seal_fail_before_replay(tmp_path, episode_data):
    write_episode(tmp_path, episode_data)
    frame = tmp_path / "evidence/frame_000001.npz"
    frame.write_bytes(frame.read_bytes() + b"tampered")
    report = audit_episode(tmp_path)
    assert not report["passed"]
    assert "evidence hash mismatch" in report["error"]
    (tmp_path / "evidence/completion.json").unlink()
    report = audit_episode(tmp_path)
    assert not report["passed"]
    assert "FileNotFoundError" in report["error"]


def test_rollover_override_is_explicitly_unverified(episode_data):
    metadata, records, rows, summary, _, _ = copy.deepcopy(episode_data)
    summary["result"] = "isaac_rollover"
    summary["route"]["termination"] = "isaac_rollover"
    checks = Checks()
    count, terminal_verified, computed = replay(
        ((metadata, record, {"frame": i}) for i, record in enumerate(records)),
        build_environment(get_case(metadata["case"]["case_id"])), summary, rows, checks)
    assert count == 4
    assert not terminal_verified
    assert computed["result"] == "isaac_rollover"
    assert not checks.failures


def test_scored_protocol_is_bound_and_running_scored_episode_fails(tmp_path, episode_data):
    metadata, records, rows, summary, realization, track = copy.deepcopy(episode_data)
    protocol = {"case_ids": [case.case_id for case in all_cases()],
                **{key: metadata[key] for key in ("source_sha256", "manifest_sha256", "graph_sha256")},
                "render_profile": {"renderer": "test"}}
    protocol["protocol_sha256"] = json_sha256(protocol)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    metadata.update(pilot_only=False, protocol_sha256=sha256_file(path))
    summary.update(pilot_only=False, protocol_sha256=sha256_file(path), render_profile=protocol["render_profile"])
    write_episode(tmp_path, (metadata, records, rows, summary, realization, track))
    report = audit_episode(tmp_path)
    assert not report["passed"]
    assert report["mismatches"] == [{"field": "terminal.finished_or_pilot", "frame": None}]
    protocol["manifest_sha256"] = "e" * 64
    path.write_text(json.dumps(protocol))
    report = audit_episode(tmp_path)
    assert report["comparisons"]["protocol.file_sha256"]["mismatches"] == 1
    assert report["comparisons"]["protocol.digest"]["mismatches"] == 1
    assert report["comparisons"]["protocol.manifest_sha256"]["mismatches"] == 1


@pytest.mark.parametrize(("mode", "applied"), [("zero", np.zeros(8)),
                                               ("swapped", np.array([0., 1., 0., 0., 0., 0., 0., 0.]))])
def test_ablation_applies_to_requested_pulse_only(episode_data, mode, applied):
    metadata, records, rows, summary, _, _ = copy.deepcopy(episode_data)
    summary["case"]["command_mode"] = mode
    requested = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
    records[0]["commands"] = {"mode": mode, "requested": requested, "applied": applied}
    for record in records[1:]:
        record["commands"]["mode"] = mode
    env = build_environment(get_case(metadata["case"]["case_id"]))
    env.frame_desire = requested
    checks = Checks()
    replay(((metadata, record, {"frame": i}) for i, record in enumerate(records)), env, summary, rows, checks)
    assert not checks.failures


def test_timeout_is_applied_after_last_physical_state_update(episode_data):
    metadata, records, _, summary, _, _ = copy.deepcopy(episode_data)
    first = records[0]
    final_state = {**first["state"], "vx": 3., "t": 120.}
    summary.update(result="timeout", recorded_frames=2400, final_state=final_state, seconds_simulated=120.,
                   progress=first["progress"]["s"] / build_environment(
                       get_case(metadata["case"]["case_id"])).mission.s_end,
                   rms_lat=abs(first["progress"]["lat"]))
    summary["route"]["termination"] = "timeout"
    rows = []

    def stream():
        for frame in range(2400):
            speed = 3. if frame else 0.
            state = {**first["state"], "vx": speed, "t": frame / 20}
            velocity = np.array([speed, 0., 0.])
            record = {**first, "frame": frame, "state": state,
                      "body": {**first["body"], "velocity_world": velocity, "velocity_body": velocity},
                      "temporal": {"older_image_frame": max(0, frame - 4), "current_image_frame": frame}}
            rows.append({"frame": frame, "t": state["t"], "x": state["x"], "y": state["y"],
                         "yaw": state["yaw"], "speed": speed, "s": first["progress"]["s"],
                         "lat": first["progress"]["lat"], "legacy_doo": 0, "physical_contacts": 0})
            yield metadata, record, {"frame": frame}

    checks = Checks()
    count, verified, computed = replay(stream(), build_environment(get_case(metadata["case"]["case_id"])),
                                       summary, rows, checks)
    assert count == 2400 and verified
    assert computed["result"] == "timeout"
    assert not checks.failures
