"""Independently reconstruct frozen plans, OP steering and sequential PI state."""

import hashlib
import json
from pathlib import Path

import numpy as np

from fsop.sim.expert import ExpertParams, Mission
from fsop.track.fixed_layouts import skidpad

ROOT = Path(__file__).resolve().parent
TIMES = 10 * (np.arange(33) / 32) ** 2
LIMIT = 1e-10


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


results = {}
for name in ("native_transition_correction", "native_standing_reference"):
    folder = ROOT / name
    protocol = json.loads((folder / "protocol.json").read_text())
    closure = json.loads((folder / "run1/closure_receipt.json").read_text())
    assert digest(folder / "protocol.json") == closure["protocol_sha256"]
    assert digest(folder / "run1/comparison.json") == closure["comparison_sha256"]
    source = folder / "source_snapshot"
    metadata = json.loads((source / "meta.json").read_text())
    for item in source.iterdir():
        expected = protocol["source_files_sha256"].get(str(Path(protocol["source"]) / item.name))
        assert expected is not None and digest(item) == expected
    nominal = np.load(source / "states.npy", allow_pickle=False)
    teacher = np.load(source / "controls.npy", allow_pickle=False)
    track = skidpad()
    generated = json.loads(json.dumps(track.to_dict()))
    archived = json.loads((source / "track.json").read_text())
    assert {key: value for key, value in generated.items() if key != "stats"} == {
        key: value for key, value in archived.items() if key != "stats"
    }
    mission = Mission(track, ExpertParams(**metadata["teacher"]))
    vp = metadata["vehicle"]
    for arm in protocol["arms"]:
        run = folder / "run1" / arm
        arm_receipt = json.loads((run / "closure_receipt.json").read_text())
        for filename, expected in arm_receipt["files_sha256"].items():
            assert digest(run / filename) == expected
        trace = np.load(run / "trace.npz", allow_pickle=False)
        diagnostic = json.loads((run / "diagnostics.json").read_text())
        assert len(diagnostic) == len(trace["plans"]) == len(trace["plan_ticks"])
        integral = 0.0
        residuals = {"plan": 0.0, "steering": 0.0, "acceleration": 0.0, "PI_diagnostics": 0.0}
        branch = protocol["branch_tick"]
        assert np.array_equal(trace["states"][:branch + 1], nominal[:branch + 1])
        assert np.array_equal(trace["actions"][:branch], teacher[:branch])
        for index, (tick, plan, row) in enumerate(zip(trace["plan_ticks"], trace["plans"], diagnostic, strict=True)):
            assert tick == branch + index == row["tick"]
            actual = trace["states"][tick]
            future = nominal[tick:tick + 201]
            clock = future[:, 8] - future[0, 8]
            angle = future[:, 2]
            velocity = np.column_stack((future[:, 3] * np.cos(angle) - future[:, 4] * np.sin(angle),
                                        future[:, 3] * np.sin(angle) + future[:, 4] * np.cos(angle)))
            rotation = np.array([[np.cos(actual[2]), np.sin(actual[2])],
                                 [-np.sin(actual[2]), np.cos(actual[2])]])
            expected_plan = np.zeros((33, 15))
            for columns, value in (([0, 1], (future[:, :2] - actual[:2]) @ rotation.T),
                                   ([3, 4], velocity @ rotation.T),
                                   ([6, 7], np.gradient(velocity, 0.05, axis=0) @ rotation.T)):
                expected_plan[:, columns] = np.stack([np.interp(TIMES, clock, value[:, axis])
                                                      for axis in (0, 1)], axis=1)
            yaw = np.unwrap(np.arctan2(np.sin(angle - actual[2]), np.cos(angle - actual[2])))
            expected_plan[:, 11] = np.interp(TIMES, clock, yaw)
            expected_plan[:, 14] = np.interp(TIMES, clock, future[:, 5])
            residuals["plan"] = max(residuals["plan"], float(np.max(np.abs(expected_plan - plan))))
            velocity = expected_plan[:, 3:5]
            acceleration = expected_plan[:, 6:8]
            speed = np.linalg.norm(velocity, axis=1)
            tangent = np.sum(velocity * acceleration, axis=1) / np.maximum(speed, 0.001)
            target = float(np.interp(0.3, TIMES, speed))
            feedforward = float(np.interp(0.3, TIMES, tangent))
            error = target - np.hypot(actual[3], actual[4])
            before = integral
            current = 0.0 if target <= 0.001 else before
            remaining = mission.s_end - trace["route"][tick, 0]
            launch = actual[3] < 2.0 and remaining > 15.0
            allowed = np.sqrt(2 * 3.5 * max(remaining, 0.0))
            finish = remaining <= 60.0 and (actual[3] > allowed or remaining <= 0.5)
            override = bool(launch or finish)
            if override:
                current = 0.0
            raw = 1.5 * error + 0.5 * feedforward + current
            proposed = float(np.clip(raw, -vp["max_brake"], vp["max_accel"]))
            applied = max(proposed, 2.0) if launch else proposed
            if remaining <= 60.0:
                if actual[3] > allowed:
                    applied = min(applied, -(actual[3] ** 2 - allowed ** 2) / max(2 * max(remaining, 0.5), 1.0) - 0.5)
                if remaining <= 0.5:
                    applied = min(applied, -3.0 if actual[3] > 0.3 else -1.0)
            applied = float(np.clip(applied, -vp["max_brake"], vp["max_accel"]))
            delta = 0.5 * error * 0.05
            integral = (0.0 if override or target <= 0.001 else current if (raw - applied) * delta > 0
                        else float(np.clip(current + delta, -2.0, 2.0)))
            fields = {"target_speed": target, "tangent_accel": feedforward, "speed_error": error,
                      "raw": raw, "proposed": proposed, "applied": applied,
                      "integral_before": before, "integral_after": integral}
            discrepancy = {key: [value, row[key]] for key, value in fields.items() if abs(value - row[key]) > LIMIT}
            if discrepancy:
                print(json.dumps({"first_bad_tick": int(tick), "arm": arm, "mission_end": mission.s_end,
                                  "remaining": float(remaining), "fields": discrepancy}))
                raise RuntimeError("Independent reconstruction discrepancy")
            assert override == row["mission_override"]
            residuals["PI_diagnostics"] = max(residuals["PI_diagnostics"],
                                               max(abs(value - row[key]) for key, value in fields.items()))
            psi = float(np.interp(0.2, TIMES, expected_plan[:, 11]))
            v = max(actual[3], 1.0)
            steer = float(np.arctan((vp["lf"] + vp["lr"]) * (2 * psi / (v * 0.2) - expected_plan[0, 14] / v)))
            if arm in ("op_lateral_pi", "op_lateral_recorded_long"):
                offset = (-np.sin(expected_plan[0, 11]) * expected_plan[0, 0]
                          + np.cos(expected_plan[0, 11]) * expected_plan[0, 1])
                steer += float(np.arctan2(protocol["lateral_gain"] * offset, max(actual[3], 0.0) + 1.0))
            accel = teacher[tick, 1] if arm in ("op_recorded_long", "op_lateral_recorded_long") else applied
            residuals["steering"] = max(residuals["steering"], abs(steer - trace["actions"][tick, 0]))
            residuals["acceleration"] = max(residuals["acceleration"], abs(accel - trace["actions"][tick, 1]))
        assert all(value <= LIMIT for value in residuals.values()), (name, arm, residuals)
        results[f"{name}/{arm}"] = {"follower_controls": len(diagnostic), "max_abs_residuals": residuals,
                                    "trace_sha256": digest(run / "trace.npz")}

report = {"passed": True, "absolute_tolerance": LIMIT, "arms": results,
          "audit_script_sha256": digest(Path(__file__)),
          "scope": "Independent world-coordinate plan and sequential PI, mission and action reconstruction. "
                   "This verifies execution, not vision performance."}
(ROOT / "native_control_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
