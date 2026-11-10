import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fsop.sim.expert import ExpertParams, Mission
from fsop.eval.controller import controller_contract
from fsop.track import centerline
from fsop.track.fixed_layouts import skidpad
from fsop.track.generate import TrackGenCfg, generate_track
from scripts.verify_lab_recovery import route_states, verify

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "docs/isaac-lab-training/recovery_recipe.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_meta(path, meta):
    meta["arrays_sha256"] = {p.name: sha(p) for p in path.glob("*.npy")}
    (path / "meta.json").write_text(json.dumps(meta))


@pytest.fixture
def episode(tmp_path):
    recipe = json.loads(RECIPE.read_text())
    membership = json.loads((RECIPE.parent / recipe["membership"]).read_text())
    case = next(c for c in membership["cases"] if c["id"] == recipe["recovery_case_ids"][0])
    track = generate_track(case["track_seed"], TrackGenCfg())
    track.laps = 1
    mission = Mission(track, ExpertParams(**recipe["teachers"][case["id"]]))
    x, y, _ = track.start_pose
    hint, progress = mission.init_progress(x, y)
    hint, _, _ = mission.update_progress(x, y, hint, progress)
    segment = mission._seg[hint]
    direction = segment / np.linalg.norm(segment)
    point = mission.pts[hint] + 0.5 * segment
    normal = np.array([-direction[1], direction[0]])
    path = tmp_path / "episode"
    path.mkdir()
    states = np.zeros((461, 9))
    states[:, :2] = point
    states[160:180, :2] += 0.4 * normal
    states[:, 2] = np.arctan2(direction[1], direction[0])
    states[:, 8] = np.arange(len(states)) / 20
    ticks = np.arange(0, len(states), 4)
    teacher = np.ones(len(states) - 1, dtype=bool)
    teacher[:160] = False
    arrays = {
        "ticks": ticks,
        "states": states,
        "controls": np.zeros((len(states) - 1, 2)),
        "teacher": teacher,
        "desires": np.zeros((len(ticks), 25, 8)),
        "contacts": np.zeros(len(states), dtype=np.int64),
        "cones": np.empty((0, 3)),
    }
    for name, value in arrays.items():
        np.save(path / (name + ".npy"), value)
    for name in ("road", "wide"):
        image = np.lib.format.open_memmap(
            path / (name + ".npy"), mode="w+", dtype=np.uint8, shape=(len(ticks), 6, 128, 256)
        )
        del image
    (path / "track.json").write_text(json.dumps(track.to_dict()))
    source = b"producer source"
    with zipfile.ZipFile(path / "source.zip", "w") as archive:
        archive.writestr("collector.py", source)
    meta = {
        "schema_version": 1,
        "backend": "IsaacLab",
        "id": case["id"],
        "case": case,
        "membership_sha256": recipe["membership_sha256"],
        "split": "train",
        "discipline": case["discipline"],
        "seed": case["seed"],
        "track_id": track.id,
        "vehicle": {"lr": 0.8, "lf": 0.75, "max_brake": 13.0, "max_accel": 7.0},
        "summary": {"result": "finished", "pass": True, "complete": True, "physical_cone_hits": 0, "oc": 0},
        "terminal_continuation": {"start_tick": 260, "end_tick": 460, "contacts": 0},
        "teacher": recipe["teachers"][case["id"]],
        "checkpoint_sha256": recipe["checkpoint_sha256"],
        "source_sha256": {"collector.py": hashlib.sha256(source).hexdigest()},
        "source_archive_sha256": sha(path / "source.zip"),
        "control_hz": 20,
        "capture_hz": 20,
        "image_hz": 5,
        "collection_policy": {
            "mode": "learner_prefix",
            "requested_takeover_s": 8,
            "lateral_takeover_threshold_m": 0.6,
            "actual_takeover_tick": 160,
            "takeover_reason": "time_limit",
        },
    }
    write_meta(path, meta)
    return path


def test_valid_recovery_receipt_contains_qualified_frames_and_numeric_trace(episode):
    result = verify(episode, RECIPE)
    assert result["passed"], result["reasons"]
    assert result["recovery_ticks"] == list(range(160, 200, 4))
    assert result["takeover"]["lateral_m"] == pytest.approx(0.4, abs=1e-4)
    assert result["settled"]["start_tick"] == 180
    assert result["settled"]["end_tick"] == 190
    assert np.isfinite(result["route_slice"][0]["progress_m"])


@pytest.mark.parametrize("problem", ["handoff", "no_correction", "no_return", "contact", "checkpoint", "source"])
def test_failed_proof_writes_receipt_and_preserves_episode(episode, tmp_path, problem):
    meta = json.loads((episode / "meta.json").read_text())
    if problem == "handoff":
        flags = np.load(episode / "teacher.npy")
        flags[170] = False
        np.save(episode / "teacher.npy", flags)
    elif problem in ("no_correction", "no_return"):
        states = np.load(episode / "states.npy")
        if problem == "no_correction":
            states[160:180, :2] = states[0, :2]
        else:
            states[180:241, :2] = states[160, :2]
        np.save(episode / "states.npy", states)
    elif problem == "contact":
        meta["summary"]["physical_cone_hits"] = 1
    elif problem == "checkpoint":
        meta["checkpoint_sha256"] = "0" * 64
    else:
        meta["source_sha256"]["collector.py"] = "0" * 64
    write_meta(episode, meta)
    before = sha(episode / "meta.json")
    receipt = tmp_path / "failure.json"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_lab_recovery.py"),
            "--episode",
            str(episode),
            "--recipe",
            str(RECIPE),
            "--out",
            str(receipt),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1, run.stderr
    result = json.loads(receipt.read_text())
    assert result["passed"] is False and result["reasons"]
    assert sha(episode / "meta.json") == before


def test_yaw_correction_qualifies_without_lateral_error(episode):
    states = np.load(episode / "states.npy")
    states[160:180, :2] = states[0, :2]
    states[160:180, 2] += 0.2
    np.save(episode / "states.npy", states)
    meta = json.loads((episode / "meta.json").read_text())
    write_meta(episode, meta)
    result = verify(episode, RECIPE)
    assert result["passed"], result["reasons"]
    assert result["takeover"]["yaw_rad"] == pytest.approx(0.2, abs=2e-4)


def test_lateral_trigger_takes_over_at_first_crossing(episode):
    states = np.load(episode / "states.npy")
    delta = states[160, :2] - states[0, :2]
    states[100:160, :2] = states[0, :2] + delta * 1.75
    states[170:180, :2] = states[0, :2]
    np.save(episode / "states.npy", states)
    teacher = np.load(episode / "teacher.npy")
    teacher[100:] = True
    np.save(episode / "teacher.npy", teacher)
    meta = json.loads((episode / "meta.json").read_text())
    meta["collection_policy"].update(actual_takeover_tick=100, takeover_reason="lateral_threshold")
    write_meta(episode, meta)
    result = verify(episode, RECIPE)
    assert result["passed"], result["reasons"]
    assert result["takeover"]["lateral_m"] == pytest.approx(0.7, abs=1e-4)
    assert result["settled"]["end_tick"] == 180


def test_terminal_anchor_and_missing_array_integrity_cannot_qualify(episode):
    meta = json.loads((episode / "meta.json").read_text())
    meta["terminal_continuation"]["start_tick"] = 176
    meta["terminal_continuation"]["end_tick"] = 376
    for name in ("states", "contacts"):
        value = np.load(episode / (name + ".npy"))[:377]
        np.save(episode / (name + ".npy"), value)
    for name in ("teacher", "controls"):
        value = np.load(episode / (name + ".npy"))[:376]
        np.save(episode / (name + ".npy"), value)
    ticks = np.arange(0, 377, 4)
    np.save(episode / "ticks.npy", ticks)
    for name in ("road", "wide", "desires"):
        value = np.load(episode / (name + ".npy"))[: len(ticks)]
        np.save(episode / (name + ".npy"), value)
    write_meta(episode, meta)
    result = verify(episode, RECIPE)
    assert not result["passed"]
    assert all(tick < 176 for tick in result["recovery_ticks"])
    assert any("eight qualified" in reason for reason in result["reasons"])
    with (episode / "controls.npy").open("ab") as stream:
        stream.write(b"changed")
    assert "SHA-256 mismatch" in verify(episode, RECIPE)["reasons"][0]


def test_overlap_projection_follows_progress_not_global_nearest():
    track = skidpad()
    mission = Mission(track, ExpertParams())
    indices = np.arange(0, len(mission.pts) - 1, 5)
    points = mission.pts[indices]
    states = np.zeros((len(indices), 9))
    states[:, :2] = points
    states[:, 2] = np.arctan2(mission._seg[indices, 1], mission._seg[indices, 0])
    route = route_states(mission, states)
    differences = []
    for i, row in zip(indices, route, strict=True):
        nearest, _, _, _ = centerline.project(mission.pts, mission.closed, mission.pts[i])
        if abs(nearest - i) > 120 and abs(row["segment"] - i) <= 1:
            differences.append(row)
    assert differences, "Expected a repeated skidpad position to retain its progressed route segment"
    assert all(abs(row["yaw_rad"]) < 0.05 for row in differences)


def test_event_recipe_uses_first_lateral_crossing(episode):
    event_recipe = ROOT / "docs/isaac-lab-training/recovery_recipe_event.json"
    states = np.load(episode / "states.npy")
    delta = states[160, :2] - states[0, :2]
    states[100:120, :2] = states[0, :2] + delta * 1.5
    states[160:180, :2] = states[0, :2]
    np.save(episode / "states.npy", states)
    teacher = np.load(episode / "teacher.npy")
    teacher[100:] = True
    np.save(episode / "teacher.npy", teacher)
    meta = json.loads((episode / "meta.json").read_text())
    meta["collection_policy"].update(
        requested_takeover_s=120,
        lateral_takeover_threshold_m=0.4,
        actual_takeover_tick=100,
        takeover_reason="lateral_threshold",
    )
    write_meta(episode, meta)
    result = verify(episode, event_recipe)
    assert result["passed"], result["reasons"]
    assert result["first_trigger_tick"] == 100
    assert result["takeover"]["lateral_m"] == pytest.approx(0.6, abs=1e-4)


def test_never_handed_off_even_when_terminal_controls_are_teacher(episode):
    event_recipe = ROOT / "docs/isaac-lab-training/recovery_recipe_event.json"
    teacher = np.load(episode / "teacher.npy")
    teacher[:260] = False
    np.save(episode / "teacher.npy", teacher)
    meta = json.loads((episode / "meta.json").read_text())
    meta["collection_policy"].update(
        requested_takeover_s=120, lateral_takeover_threshold_m=0.4, actual_takeover_tick=None, takeover_reason=None
    )
    write_meta(episode, meta)
    result = verify(episode, event_recipe)
    assert not result["passed"]
    assert any("takeover tick" in reason for reason in result["reasons"])


def test_time_trigger_uses_saved_timestamps(episode):
    states = np.load(episode / "states.npy")
    states[:, 8] += 0.05
    states[159, :2] = states[160, :2]
    np.save(episode / "states.npy", states)
    teacher = np.load(episode / "teacher.npy")
    teacher[159] = True
    np.save(episode / "teacher.npy", teacher)
    meta = json.loads((episode / "meta.json").read_text())
    meta["collection_policy"]["actual_takeover_tick"] = 159
    write_meta(episode, meta)
    result = verify(episode, RECIPE)
    assert result["passed"], result["reasons"]
    assert result["first_trigger_tick"] == 159


@pytest.mark.parametrize("contract, passes", [
    (controller_contract("pp"), True),
    (controller_contract("op_tangent_pi_v1"), False),
    (None, False),
])
def test_legacy_explicit_controller_must_be_pp(episode, contract, passes):
    meta = json.loads((episode / "meta.json").read_text())
    meta["controller_contract"] = contract
    write_meta(episode, meta)
    result = verify(episode, RECIPE)
    assert result["passed"] is passes, result["reasons"]
    if not passes:
        assert any("controller contract" in reason for reason in result["reasons"])
