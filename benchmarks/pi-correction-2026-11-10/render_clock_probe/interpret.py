import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()
report = json.loads((args.root / "comparison.json").read_text())
expected = 6 * float(np.float32(1 / 120))
zero_operations = [row for run in report["runs"].values() for row in run["operations"] if row["operation"] != "advance"]
advances = [row for run in report["runs"].values() for row in run["operations"] if row["operation"] == "advance"]
settings = [json.loads((args.root / mode / "metadata.json").read_text())["settings"] for mode in ("sync", "async")]
render_differences = []
left, right = [value["render_prims"] for value in settings]
for path in sorted(set(left) | set(right)):
    if path not in left or path not in right:
        render_differences.append({"path": path, "missing_in_one_run": True})
        continue
    for category in ("attributes", "relationships"):
        for key in sorted(set(left[path][category]) | set(right[path][category])):
            a, b = left[path][category].get(key), right[path][category].get(key)
            if a != b:
                render_differences.append({"path": path, "category": category, "key": key, "sync": a, "async": b})
sensor_paths = {camera["render_product"] for value in settings for camera in value["cameras"].values()}
result = {
    "post_observation_interpretation": True,
    "original_protocol_pass": report["pass"],
    "original_failure_count": len(report["failures"]),
    "original_criteria_unchanged": True,
    "physics_zero_advance_exact": all(row["sim_seconds"] == 0 and row["steps"] == 0 and row["pose_max_abs"] == 0 and row["velocity_max_abs"] == 0 for row in zero_operations),
    "zero_advance_operation_count": len(zero_operations),
    "writer_close_physics_unchanged": all(run["close"]["sim_seconds"] == 0 and run["close"]["steps"] == 0 and run["close"]["pose_max_abs"] == 0 and run["close"]["velocity_max_abs"] == 0 for run in report["runs"].values()),
    "six_float32_timesteps_seconds": expected,
    "advance_count": len(advances),
    "all_advances_exactly_six_steps": all(row["steps"] == 6 for row in advances),
    "advance_duration_max_abs_error_vs_float32": max(abs(row["sim_seconds"] - expected) for row in advances),
    "advance_duration_error_vs_rational": expected - 0.05,
    "cross_process_max_abs": {key: max(abs(row[key]) for row in report["cross_process"]) for key in ("sim_seconds", "timeline_seconds", "steps", "pose_max_abs", "velocity_max_abs")},
    "minimum_idle_wall_seconds": min(row["wall_seconds"] for row in zero_operations if row["operation"] == "idle"),
    "capture_timeline_deltas": sorted({row["timeline_seconds"] for row in zero_operations if row["operation"] == "capture"}),
    "advance_timeline_deltas": sorted({row["timeline_seconds"] for row in advances}),
    "sensor_settings_equal_except_picking_id": not any(row["path"] in sensor_paths and row.get("key") != "viewPickingId" for row in render_differences),
    "render_prim_differences": render_differences,
    "initial_pixel_differences": report["initial_pixel_differences"],
    "limitations": "Two 8-tick fixed-action runs, one per writer, do not establish learned-policy repeatability, full-episode timing isolation, or the exact cause of RTX pixel variation. Timeline advancement is separately observed and is not evidence of physics advancement."
}
(args.root / "interpretation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
