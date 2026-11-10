"""Analyze frozen native traces. No simulator, training, or source mutation."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("inputs")
OUTPUT = Path(__file__).with_name("results.json")
T = 10.0 * (np.arange(33) / 32) ** 2
BRANCH = 255


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotate(vector, yaw):
    return np.stack((vector[:, 0] * np.cos(yaw) - vector[:, 1] * np.sin(yaw),
                     vector[:, 0] * np.sin(yaw) + vector[:, 1] * np.cos(yaw)), axis=1)


def metrics(error):
    return {"rms_rad": float(np.sqrt(np.mean(error ** 2))), "mean_rad": float(np.mean(error)),
            "mae_rad": float(np.mean(np.abs(error))), "max_abs_rad": float(np.max(np.abs(error)))}


meta = json.loads((SOURCE / "source_snapshot/meta.json").read_text())
bound = {}
for name in ("states", "controls", "contacts", "teacher", "cones"):
    file = SOURCE / "source_snapshot" / (name + ".npy")
    bound[str(file.relative_to(SOURCE))] = digest(file)
    assert bound[str(file.relative_to(SOURCE))] == meta["arrays_sha256"][file.name]
for name in ("recorded", "op_tangent_pi", "pp_tangent_pi"):
    folder = SOURCE / "run1" / name
    receipt = json.loads((folder / "closure_receipt.json").read_text())
    bound[str((folder / "trace.npz").relative_to(SOURCE))] = digest(folder / "trace.npz")
    assert digest(folder / "trace.npz") == receipt["files_sha256"]["trace.npz"]

states = np.load(SOURCE / "source_snapshot/states.npy", allow_pickle=False)
controls = np.load(SOURCE / "source_snapshot/controls.npy", allow_pickle=False)
cones = np.load(SOURCE / "source_snapshot/cones.npy", allow_pickle=False)
op = np.load(SOURCE / "run1/op_tangent_pi/trace.npz", allow_pickle=False)
recorded = np.load(SOURCE / "run1/recorded/trace.npz", allow_pickle=False)
actual = op["states"]
actions = op["actions"]
diagnostics = {d["tick"]: d for d in json.loads((SOURCE / "run1/op_tangent_pi/diagnostics.json").read_text())}
assert np.array_equal(recorded["states"], states)
assert np.array_equal(recorded["actions"], controls)
assert not recorded["contacts"].any()
assert np.array_equal(actual[:BRANCH + 1], states[:BRANCH + 1])
assert np.array_equal(actions[:BRANCH], controls[:BRANCH])

columns = ("x", "y", "yaw", "vx", "vy", "omega_z", "actuator_steer", "actuator_accel", "t")
first = {}
for index, name in enumerate(columns):
    differences = np.flatnonzero(actual[:, index] != states[:len(actual), index])
    first[name] = int(differences[0]) if len(differences) else None

rows = []
for tick in (255, 256, 270, 300, 350, 400, 430, 435, 440, 445, 448, 449, 450):
    source = states[tick]
    measured = actual[tick]
    delta = measured[:2] - source[:2]
    c, s = np.cos(source[2]), np.sin(source[2])
    source_frame = np.array([[c, s], [-s, c]]) @ delta
    row = {"tick": tick, "source_state": source.tolist(), "op_state": measured.tolist(),
           "source_action": controls[tick].tolist(), "op_action": actions[tick].tolist(),
           "world_position_delta_m": delta.tolist(), "source_heading_position_delta_m": source_frame.tolist(),
           "yaw_delta_rad": float(np.arctan2(np.sin(measured[2] - source[2]), np.cos(measured[2] - source[2]))),
           "source_sideslip_rad": float(np.arctan2(source[4], source[3])),
           "op_sideslip_rad": float(np.arctan2(measured[4], measured[3])),
           "contact_count": int(op["contacts"][tick])}
    lower, upper = max(BRANCH, tick - 20), tick + 21
    points = states[lower:upper, :2]
    segments = np.diff(points, axis=0)
    fraction = np.clip(np.einsum("ij,ij->i", measured[:2] - points[:-1], segments) /
                       np.einsum("ij,ij->i", segments, segments), 0, 1)
    projected = points[:-1] + fraction[:, None] * segments
    nearest = int(np.argmin(np.linalg.norm(projected - measured[:2], axis=1)))
    row["nearby_source_path_projection"] = {
        "distance_m": float(np.linalg.norm(projected[nearest] - measured[:2])),
        "source_time_delta_s": float((lower + nearest + fraction[nearest] - tick) * 0.05),
        "search_source_state_interval": [lower, upper],
    }
    if tick >= 256:
        nominal = states[BRANCH:tick + 1]
        observed = actual[BRANCH:tick + 1]
        forward = rotate(np.c_[observed[:, 3] - nominal[:, 3], np.zeros(len(observed))], nominal[:, 2])
        sideways = rotate(np.c_[np.zeros(len(observed)), observed[:, 4] - nominal[:, 4]], nominal[:, 2])
        heading = rotate(observed[:, 3:5], observed[:, 2]) - rotate(observed[:, 3:5], nominal[:, 2])
        integrated = [np.trapezoid(v, dx=0.05, axis=0) for v in (forward, sideways, heading)]
        row["world_displacement_accounting_m"] = dict(zip(("vx_difference", "vy_difference", "heading_difference"),
                                                         [v.tolist() for v in integrated]))
        row["world_displacement_accounting_m"]["integration_residual"] = (delta - sum(integrated)).tolist()
    rows.append(row)

own = []
for tick in range(BRANCH, 927):
    future = states[tick:tick + 201].copy()
    future[:, 8] -= future[0, 8]
    relative_yaw = np.unwrap(future[:, 2] - future[0, 2])
    yaw_knots = np.interp(T, future[:, 8], relative_yaw)
    original_rates = np.interp(T, future[:, 8], future[:, 5])
    yaw_derivative = np.gradient(np.unwrap(future[:, 2]), 0.05, edge_order=2)
    corrected_rates = np.interp(T, future[:, 8], yaw_derivative)
    psi = np.interp(0.2, T, yaw_knots)
    speed = max(states[tick, 3], 1.0)
    steering = [float(np.arctan(1.55 * (2 * psi / (speed * 0.2) - rate[0] / speed)))
                for rate in (original_rates, corrected_rates)]
    own.append({"tick": tick, "teacher_steer_rad": float(controls[tick, 0]),
                "original_steer_rad": steering[0], "yaw_derivative_steer_rad": steering[1],
                "yaw_at_0p2_rad": float(psi), "omega_z_radps": float(future[0, 5]),
                "forward_second_order_yaw_derivative_radps": float(yaw_derivative[0])})
window_metrics = []
for lower, upper in ((255, 430), (255, 449), (430, 470), (470, 835), (255, 835), (835, 927), (255, 927)):
    chosen = own[lower - BRANCH:upper - BRANCH]
    window_metrics.append({"source_ticks_inclusive_exclusive": [lower, upper], "count": len(chosen),
        "original": metrics(np.array([r["original_steer_rad"] - r["teacher_steer_rad"] for r in chosen])),
        "yaw_derivative": metrics(np.array([r["yaw_derivative_steer_rad"] - r["teacher_steer_rad"] for r in chosen]))})

footprint = []
cone = cones[30, :2]
for tick in (440, 445, 448, 449, 450):
    for name, state in (("teacher", states[tick]), ("op", actual[tick])):
        c, s = np.cos(state[2]), np.sin(state[2])
        local = np.array([[c, s], [-s, c]]) @ (cone - state[:2])
        dx = max(-1.3 - local[0], 0, local[0] - 1.6)
        dy = max(abs(local[1]) - 0.7, 0)
        footprint.append({"tick": tick, "arm": name, "cone_center_ego_m": local.tolist(),
                          "point_to_nominal_body_rectangle_m": float(np.hypot(dx, dy))})

report = {
    "purpose": "Read-only arithmetic on one sealed privileged TRAIN source. No physical or neural experiment.",
    "script_sha256": digest(Path(__file__)), "source_archive": str(SOURCE), "input_sha256": bound,
    "state_columns": columns, "exact_recorded_replay_verified": True,
    "first_action_difference_tick": int(np.flatnonzero(np.any(actions != controls[:len(actions)], axis=1))[0]),
    "first_state_difference_ticks": first,
    "contact_increment_state_ticks": (np.flatnonzero(np.diff(op["contacts"]) > 0) + 1).tolist(),
    "selected_trace_rows": rows, "own_teacher_state_steering_windows": window_metrics,
    "own_teacher_state_steering_rows": own,
    "precontact_PI": {
        "peak_absolute_speed_error_mps": float(max(abs(diagnostics[t]["speed_error"]) for t in range(255, 450))),
        "peak_absolute_speed_error_tick": max(range(255, 450), key=lambda t: abs(diagnostics[t]["speed_error"])),
        "mission_override_ticks": [t for t in range(255, 450) if diagnostics[t]["mission_override"]],
        "saturation_ticks": [t for t in range(255, 450) if abs(diagnostics[t]["raw"] - diagnostics[t]["applied"]) > 1e-10],
    },
    "cone_c0030_world_m": cone.tolist(), "footprint_proxy": footprint,
    "limitations": [
        "Displacement accounting is an identity along the observed paths, not a causal controller intervention.",
        "Projection uses only nearby source segments to avoid rematching repeated skidpad laps.",
        "Rectangle distance excludes cone radius, body roll, wheel geometry and six physics substeps. It is not contact clearance.",
        "No native roll or pitch series is available in the nine-column source states.",
        "Own-state steering residuals do not predict closed-loop success.",
        "Yaw derivative is reconstructed from the complete native future, not model predictions.",
    ],
}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"output": str(OUTPUT), "output_sha256": digest(OUTPUT), "input_hashes_verified": len(bound),
                  "first_action_difference_tick": report["first_action_difference_tick"],
                  "contact_increment_state_ticks": report["contact_increment_state_ticks"]}, indent=2))
