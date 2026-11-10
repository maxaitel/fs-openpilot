import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fsop.eval.controller import controller_contract
from fsop.sim.expert import ExpertParams, Mission
from fsop.sim.skidpad import SkidpadRoute
from fsop.sim.vehicle import VehicleState
from fsop.track.fixed_layouts import skidpad
from isaac_lab_fsop.recovery_schedule import PhaseRecovery, PhaseSchedule, first_left_decision, phase_control
from scripts.verify_lab_recovery import require_clean_left_prefix, route_states, verify

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "docs/isaac-lab-training/recovery_recipe_event.json"
SCHEDULE = {"mode": "phase_recovery_v1", "phase": "left_transition", "lateral_threshold_m": 0.35,
            "learner_window_seconds": 6, "transition_lead_m": 12}


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def test_phase_schedule_reaches_first_trigger_or_failed_deadline():
    track = skidpad()
    mission = Mission(track, ExpertParams())
    schedule = PhaseSchedule.from_dict(SCHEDULE)
    machine = PhaseRecovery(schedule, first_left_decision(track, mission))
    assert machine.advance(0, 122.0, 0.0) == "teacher_prefix"
    assert machine.advance(1, 123.0, 0.1) == "learner"
    assert machine.advance(2, 124.0, -0.36) == "expert_recovery"
    assert machine.takeover_tick == 2 and machine.takeover_reason == "lateral_threshold"
    assert machine.advance(3, 125.0, 0.0) == "expert_recovery"
    entry = PhaseRecovery(PhaseSchedule.from_dict({**SCHEDULE, "phase": "entry"}), first_left_decision(track, mission))
    assert entry.advance(119, 0, 0) == "learner"
    assert entry.advance(120, 0, 0) == "expert_recovery"
    assert entry.takeover_reason == "no_corrective_trigger"
    with pytest.raises(ValueError):
        PhaseSchedule.from_dict({**SCHEDULE, "learner_window_seconds": 7})


@pytest.fixture(scope="module")
def phase_episode(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase")
    recipe = json.loads(RECIPE.read_text())
    membership = json.loads((RECIPE.parent / recipe["membership"]).read_text())
    case = next(c for c in membership["cases"] if c["id"] == "train-skidpad-82002-visual_stress_v1")
    track = skidpad()
    mission = Mission(track, ExpertParams(**recipe["teachers"][case["id"]]))
    positions = np.arange(0, mission.s_end, 0.35)
    states = np.zeros((len(positions) + 12 + 201, 9))
    for tick, progress in enumerate(positions):
        point = mission.point_at(progress)
        delta = mission.point_at(progress + 0.01) - point
        yaw = np.arctan2(delta[1], delta[0])
        if 365 <= tick < 375:
            point += 0.38 * np.array([-np.sin(yaw), np.cos(yaw)])
        states[tick, :5] = [*point, yaw, 7, 0]
    states[len(positions):, :2] = mission.point_at(mission.s_end)
    states[:, 8] = np.arange(len(states)) / 20
    terminal = len(positions) + 12
    route = route_states(mission, states[:terminal + 1])
    activation = next(r["tick"] for r in route if r["progress_m"] >= first_left_decision(track, mission) - 12)
    assert activation == 351
    referee = SkidpadRoute(track)
    for tick in range(1, activation + 1):
        referee.advance(VehicleState(*states[tick]))
    assert referee.report()["accepted_route"] == ["right"]
    for tick in range(activation + 1, 366):
        referee.advance(VehicleState(*states[tick]))
    takeover_route = referee.report()
    teacher = np.ones(len(states) - 1, dtype=bool)
    teacher[activation:365] = False
    path = root / "episode"
    path.mkdir()
    ticks = np.arange(0, len(states), 4)
    arrays = {"ticks": ticks, "states": states, "controls": np.zeros((len(states) - 1, 2)), "teacher": teacher,
              "desires": np.zeros((len(ticks), 25, 8)), "contacts": np.zeros(len(states), dtype=np.int64),
              "cones": np.empty((0, 3))}
    for name, value in arrays.items():
        np.save(path / f"{name}.npy", value)
    for name in ("road", "wide"):
        image = np.lib.format.open_memmap(path / f"{name}.npy", mode="w+", dtype=np.uint8,
                                         shape=(len(ticks), 6, 128, 256))
        del image
    (path / "track.json").write_text(json.dumps(track.to_dict()))
    source = b"phase producer"
    with zipfile.ZipFile(path / "source.zip", "w") as archive:
        archive.writestr("collector.py", source)
    contract = controller_contract("op_tangent_v1")
    meta = {"schema_version": 1, "backend": "IsaacLab", "id": case["id"], "case": case,
            "membership_sha256": recipe["membership_sha256"], "split": "train", "discipline": "skidpad",
            "seed": case["seed"], "track_id": track.id,
            "vehicle": {"lr": 0.8, "lf": 0.75, "max_brake": 13.0, "max_accel": 7.0},
            "summary": {"result": "finished", "pass": True, "complete": True, "physical_cone_hits": 0, "oc": 0},
            "terminal_continuation": {"start_tick": terminal, "end_tick": len(teacher), "contacts": 0},
            "teacher": recipe["teachers"][case["id"]], "checkpoint_sha256": "a" * 64,
            "source_sha256": {"collector.py": hashlib.sha256(source).hexdigest()},
            "source_archive_sha256": sha(path / "source.zip"), "control_hz": 20, "capture_hz": 20, "image_hz": 5,
            "controller_contract": contract,
            "collection_policy": {"mode": "phase_recovery_v1", "schedule": SCHEDULE,
                "controller_state_during_teacher": "reset_after_each_discarded_proposal",
                "actual_activation_tick": activation, "activation_reason": "first_left_decision_lead",
                "activation_progress_m": first_left_decision(track, mission) - 12,
                "activation": {"progress_m": route[activation]["progress_m"], "lateral_m": route[activation]["lateral_m"],
                               "state": states[activation].tolist(),
                               "route": {**takeover_route, "accepted_route": ["right"]}, "contacts": 0, "off_course": 0},
                "actual_takeover_tick": 365, "takeover_reason": "lateral_threshold",
                "takeover": {"progress_m": route[365]["progress_m"], "lateral_m": route[365]["lateral_m"],
                             "state": states[365].tolist(), "route": takeover_route, "contacts": 0, "off_course": 0},
                "policy_observation_ticks": list(range(len(teacher)))}}
    meta["arrays_sha256"] = {f.name: sha(f) for f in path.glob("*.npy")}
    (path / "meta.json").write_text(json.dumps(meta))
    recipe["schema_version"] = 2
    recipe["membership"] = str((RECIPE.parent / "membership.json").resolve())
    inventory_digest = hashlib.sha256(json.dumps(meta["source_sha256"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    recipe["collection_by_case"] = {key: {"checkpoint_sha256": "a" * 64, "controller_contract": contract,
                                          "source_inventory_sha256": inventory_digest,
                                          "controller_state_during_teacher": "reset_after_each_discarded_proposal",
                                          "collection": SCHEDULE} for key in recipe["recovery_case_ids"]}
    (root / "cases").symlink_to(RECIPE.parent / "cases", target_is_directory=True)
    recipe_path = root / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    return path, recipe_path


def test_phase_receipt_checks_contract_history_and_handoffs(phase_episode):
    path, recipe = phase_episode
    result = verify(path, recipe)
    assert result["passed"], result["reasons"]
    assert result["activation_tick"] == 351 and result["takeover_tick"] == 365
    meta_path = path / "meta.json"
    original = json.loads(meta_path.read_text())
    for change, reason in [
        (lambda m: m.update(controller_contract=controller_contract("pp")), "controller contract"),
        (lambda m: m["collection_policy"].update(controller_state_during_teacher="integrate"), "controller state contract"),
        (lambda m: m["collection_policy"]["policy_observation_ticks"].remove(42), "history"),
        (lambda m: m["collection_policy"].update(actual_activation_tick=350), "phase handoffs"),
        (lambda m: m["collection_policy"].update(actual_takeover_tick=366), "phase handoffs"),
        (lambda m: m["collection_policy"].update(takeover_reason="no_corrective_trigger"), "first qualifying"),
    ]:
        meta = json.loads(json.dumps(original))
        change(meta)
        meta_path.write_text(json.dumps(meta))
        failed = verify(path, recipe)
        assert not failed["passed"] and any(reason in r for r in failed["reasons"]), failed
    meta_path.write_text(json.dumps(original))


def test_policy_history_stays_warm_across_both_handoffs():
    history = []

    def policy(obs, env):
        history.append(obs)
        return (len(history), 0, {})

    class Expert:
        def act(self):
            return (-1, 0)

    actions = [phase_control(policy, Expert(), tick, None, actor)
               for tick, actor in enumerate(["teacher_prefix", "teacher_prefix", "learner", "expert_recovery"])]
    assert history == [0, 1, 2, 3]
    assert actions == [(-1, 0), (-1, 0), (3, 0), (-1, 0)]


def test_rejects_nonclean_prefix_and_absent_trigger(phase_episode):
    path, recipe = phase_episode
    meta_path = path / "meta.json"
    original = json.loads(meta_path.read_text())
    states_path = path / "states.npy"
    original_states = np.load(states_path).copy()
    try:
        for change in ("wrong_route", "no_trigger"):
            states = original_states.copy()
            if change == "wrong_route":
                states[1, :2] = [1.0, 0.0]
            else:
                mission = Mission(skidpad(), ExpertParams(**original["teacher"]))
                for tick in range(365, 375):
                    states[tick, :2] = mission.point_at(tick * 0.35)
            np.save(states_path, states)
            meta = json.loads(json.dumps(original))
            meta["arrays_sha256"]["states.npy"] = sha(states_path)
            if change == "no_trigger":
                row = route_states(mission, states[:366])[365]
                meta["collection_policy"]["takeover"].update(state=states[365].tolist(),
                    progress_m=row["progress_m"], lateral_m=row["lateral_m"])
            meta_path.write_text(json.dumps(meta))
            result = verify(path, recipe)
            assert not result["passed"], change
            if change == "no_trigger":
                assert any("first qualifying" in reason for reason in result["reasons"]), result
    finally:
        np.save(states_path, original_states)
        meta_path.write_text(json.dumps(original))


def test_left_activation_requires_one_clean_right_lap():
    require_clean_left_prefix({"valid": True, "accepted_route": ["right"]}, [0, 0], 0)
    for route, contacts, off_course in [
        ({"valid": True, "accepted_route": []}, [0], 0),
        ({"valid": True, "accepted_route": ["right", "right"]}, [0], 0),
        ({"valid": False, "accepted_route": ["right"]}, [0], 0),
        ({"valid": True, "accepted_route": ["right"]}, [0, 1], 0),
        ({"valid": True, "accepted_route": ["right"]}, [0], 1),
    ]:
        with pytest.raises(ValueError):
            require_clean_left_prefix(route, contacts, off_course)


def test_rejects_consistent_but_early_or_late_handoffs(phase_episode):
    path, recipe = phase_episode
    meta_path = path / "meta.json"
    original = json.loads(meta_path.read_text())
    teacher_path = path / "teacher.npy"
    original_teacher = np.load(teacher_path).copy()
    states = np.load(path / "states.npy")
    try:
        for change in ("early_activation", "late_takeover"):
            meta = json.loads(json.dumps(original))
            flags = original_teacher.copy()
            if change == "early_activation":
                flags[350] = False
                meta["collection_policy"]["actual_activation_tick"] = 350
            else:
                flags[365] = False
                meta["collection_policy"]["actual_takeover_tick"] = 366
                mission = Mission(skidpad(), ExpertParams(**meta["teacher"]))
                row = route_states(mission, states[:367])[366]
                meta["collection_policy"]["takeover"].update(state=states[366].tolist(),
                    progress_m=row["progress_m"], lateral_m=row["lateral_m"])
            np.save(teacher_path, flags)
            meta["arrays_sha256"]["teacher.npy"] = sha(teacher_path)
            meta_path.write_text(json.dumps(meta))
            result = verify(path, recipe)
            expected = "first phase boundary" if change == "early_activation" else "first qualifying"
            assert not result["passed"] and any(expected in reason for reason in result["reasons"]), result
    finally:
        np.save(teacher_path, original_teacher)
        meta_path.write_text(json.dumps(original))


def test_discarded_pi_commands_reset_only_controller_state():
    from fsop.eval.controller import OpenpilotTangentPIFollower
    from fsop.sim.vehicle import VehicleParams

    plan = np.zeros((33, 15))
    plan[:, 3] = 5.0

    class Policy:
        def __init__(self):
            self.ctrl = OpenpilotTangentPIFollower(VehicleParams())
            self.history = []
            self.before = []

        def __call__(self, tick, env):
            self.history.append(tick)
            self.before.append(self.ctrl.integral)
            steer, accel, _ = self.ctrl.step(plan, 4, 0, tick / 20, lambda value: (value, False))
            return steer, accel, {}

    class Expert:
        def act(self):
            return 0, -1

    policy = Policy()
    for tick, state in enumerate(["teacher_prefix", "teacher_prefix", "learner", "learner", "expert_recovery"]):
        phase_control(policy, Expert(), tick, None, state)
        if state != "learner":
            assert policy.ctrl.integral == 0 and policy.ctrl.last_time is None
    assert policy.history == [0, 1, 2, 3, 4]
    assert policy.before[:3] == [0, 0, 0]
    assert policy.before[3] > 0 and policy.before[4] > policy.before[3]


def test_unclean_prefix_never_enables_learner():
    schedule = PhaseSchedule.from_dict(SCHEDULE)
    machine = PhaseRecovery(schedule, 134.0)
    assert machine.advance(20, 122.0, 0.0, prefix_clean=False) == "expert_recovery"
    assert machine.takeover_tick == machine.activation_tick == 20
    assert machine.takeover_reason == "invalid_teacher_prefix"
    assert machine.advance(21, 123.0, 0.0) == "expert_recovery"
