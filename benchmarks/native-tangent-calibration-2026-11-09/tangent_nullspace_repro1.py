from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fsop.eval.controller import OpenpilotTangentFollower
from fsop.sim.expert import T_IDXS
from fsop.sim.vehicle import VehicleParams

ORIGINAL_METRICS = (
    "position_interval_m", "position_interval_max_m", "velocity_interval_mps",
    "velocity_interval_max_mps", "yaw_interval_rad", "yaw_interval_max_rad",
    "position_cumulative_max_m", "velocity_cumulative_max_mps", "yaw_cumulative_max_rad",
    "initial_position_m", "initial_yaw_rad", "initial_velocity_mps", "initial_yaw_rate_radps",
)
snapshot = ROOT / ".work/lab/tangent_nullspace_coherence_snapshot1.py"
namespace = {"np": np, "T_IDXS": T_IDXS}
exec(compile(snapshot.read_text(), str(snapshot), "exec"), namespace)
coherence = namespace["plan_coherence"]

base = np.zeros((33, 15), dtype=np.float64)
base[:, 0] = 5.0 * T_IDXS
base[:, 3] = 5.0
sample = {"speed": 5.0, "lateral_speed": 0.0, "yaw_rate": 0.0}
changed = base.copy()
count = int(np.searchsorted(T_IDXS, 1.0, side="right"))
changed[:count, 6] = 20.0 * (-1.0) ** np.arange(count)
follower = OpenpilotTangentFollower(VehicleParams())
rows = {"base": coherence([base], [sample])[0], "alternating": coherence([changed], [sample])[0]}
actions = {"base": follower(base, 5.0, 0.0), "alternating": follower(changed, 5.0, 0.0)}
for row in rows.values():
    assert all(abs(row[name]) < 1e-12 for name in ORIGINAL_METRICS)
assert abs(actions["alternating"][1] - actions["base"][1] - 0.4) < 1e-12
assert rows["alternating"]["acceleration_secant_max_mps2"] == 20.0

column_weights = np.array([1., 1., .1, .5, .5, .1, .1, .1, 0., 0., 0., .5, 0., 0., .3])
time_weights = np.array([1.0 if t <= 4.0 else max(.2, 1.0 - .8 * (t - 4.0) / 6.0) for t in T_IDXS])
weights = time_weights[:, None] * column_weights[None, :]
plan_l1 = float((np.abs(changed - base) * weights).sum() / weights.sum())
paths = [snapshot, ROOT / "src/fsop/eval/controller.py", ROOT / "src/fsop/sim/expert.py",
         ROOT / "src/fsop/sim/vehicle.py", ROOT / "src/fsop/train/losses.py"]
result = {
    "kind": "synthetic_trapezoidal_nullspace_reproduction",
    "scope": "A synthetic constant-speed straight plan, not native validation or a learned candidate.",
    "execution": "Exact plan_coherence function snapshot plus actual OpenpilotTangentFollower implementation.",
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "changed_channel": "OP x acceleration at knots 0 through 10, alternating +20 and -20 m/s^2",
    "interval_end_s": float(T_IDXS[count - 1]),
    "base_action": {"steer_rad": actions["base"][0], "accel_mps2": actions["base"][1]},
    "changed_action": {"steer_rad": actions["alternating"][0], "accel_mps2": actions["alternating"][1]},
    "original_13_metrics": {kind: {key: row[key] for key in ORIGINAL_METRICS} for kind, row in rows.items()},
    "added_endpoint_metrics": {kind: {key: value for key, value in row.items() if key not in ORIGINAL_METRICS}
                               for kind, row in rows.items()},
    "weighted_plan_l1_vs_synthetic_base": plan_l1,
    "limitation": "This shows metric blindness. It does not establish that an actual trained candidate exploits it or passes all action gates.",
}
output = ROOT / ".work/lab/tangent_nullspace_repro1.json"
output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, indent=2, allow_nan=False))
