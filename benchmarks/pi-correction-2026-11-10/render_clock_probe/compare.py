import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()
protocol = json.loads((args.root / "protocol.json").read_text())
failures = []
results = {}
traces = {}
metadata = {}

def check(condition, message):
    if not condition:
        failures.append(message)

def delta(a, b):
    return {"sim_seconds": b["sim_time"] - a["sim_time"],
            "timeline_seconds": b["timeline_time"] - a["timeline_time"],
            "steps": b["step_index"] - a["step_index"],
            "pose_max_abs": float(np.max(np.abs(np.asarray(b["physx_transform_xyzw"]) - a["physx_transform_xyzw"]))),
            "velocity_max_abs": float(np.max(np.abs(np.asarray(b["physx_velocity"]) - a["physx_velocity"]))),
            "wall_seconds": b["wall_monotonic"] - a["wall_monotonic"]}

for mode in ("sync", "async"):
    root = args.root / mode
    complete = json.loads((root / "complete.json").read_text())
    metadata[mode] = json.loads((root / "metadata.json").read_text())
    trace_bytes = (root / "trace.jsonl").read_bytes()
    traces[mode] = rows = [json.loads(line) for line in trace_bytes.splitlines()]
    check(hashlib.sha256(trace_bytes).hexdigest() == complete["trace_sha256"], mode + " trace integrity")
    check(len(rows) == complete["ticks"] == complete["writer_count"] == protocol["ticks"], mode + " tick count")
    pre = json.loads((root / "preflight.json").read_text())
    check(pre["free_bytes"] >= protocol["reserve_bytes"], mode + " reserve")
    check(complete["source_unchanged"], mode + " source mutation")
    check(pre["protocol_sha256"] == hashlib.sha256((args.root / "protocol.json").read_bytes()).hexdigest(), mode + " protocol identity")
    check(pre["probe_sha256"] == hashlib.sha256((args.root / "probe.py").read_bytes()).hexdigest(), mode + " probe identity")
    manifest = [json.loads(line) for line in (root / "evidence/manifest.jsonl").read_text().splitlines()]
    receipt = json.loads((root / "evidence/completion.json").read_text())
    check(len(manifest) == receipt["frame_count"] == protocol["ticks"], mode + " evidence count")
    check(receipt["manifest_sha256"] == hashlib.sha256((root / "evidence/manifest.jsonl").read_bytes()).hexdigest(), mode + " manifest integrity")
    for entry in manifest:
        check(hashlib.sha256((root / "evidence" / entry["file"]).read_bytes()).hexdigest() == entry["sha256"], mode + " evidence file integrity")
    operations = []
    for row in rows:
        for operation in ("capture", "append", "idle", "advance"):
            change = delta(row["before_" + operation], row["after_" + operation])
            advancing = operation == "advance"
            check(change["steps"] == (6 if advancing else 0), f"{mode} tick {row['tick']} {operation} step delta")
            for clock in ("sim_seconds", "timeline_seconds"):
                check(abs(change[clock] - (0.05 if advancing else 0.0)) <= protocol["clock_tolerance_seconds"], f"{mode} tick {row['tick']} {operation} {clock}")
            if not advancing:
                check(change["pose_max_abs"] <= protocol["stationary_pose_tolerance"], f"{mode} tick {row['tick']} {operation} pose")
            if operation == "idle":
                check(change["wall_seconds"] >= protocol["idle_seconds"], mode + " idle duration")
            operations.append({"tick": row["tick"], "operation": operation, **change})
    summary = json.loads((root / "summary.json").read_text())
    close = delta(summary["writer_close"]["before"], summary["writer_close"]["after"])
    check(close["steps"] == 0 and abs(close["sim_seconds"]) <= protocol["clock_tolerance_seconds"] and abs(close["timeline_seconds"]) <= protocol["clock_tolerance_seconds"] and close["pose_max_abs"] <= protocol["stationary_pose_tolerance"], mode + " writer close isolation")
    results[mode] = {"operations": operations, "close": close, "complete": complete}

cross = []
for left, right in zip(traces["sync"], traces["async"]):
    for boundary in [key for key in left if key.startswith(("before_", "after_"))]:
        change = delta(left[boundary], right[boundary])
        check(change["steps"] == 0 and abs(change["sim_seconds"]) <= protocol["clock_tolerance_seconds"] and abs(change["timeline_seconds"]) <= protocol["clock_tolerance_seconds"] and change["pose_max_abs"] <= protocol["cross_process_pose_tolerance"], f"cross-process {left['tick']} {boundary}")
        cross.append({"tick": left["tick"], "boundary": boundary, **change})

pixel_differences = {}
with np.load(args.root / "sync/initial_images.npz") as left, np.load(args.root / "async/initial_images.npz") as right:
    for name in left.files:
        diff = np.abs(left[name].astype(float) - right[name])
        pixel_differences[name] = {"mae": float(diff.mean()), "max_abs": float(diff.max()), "different_fraction": float(np.mean(diff != 0))}

settings_equal = {key: metadata["sync"]["settings"][key] == metadata["async"]["settings"][key] for key in ("global", "cameras", "profile", "sim_render_settings")}
check(all(settings_equal.values()), "effective global settings or camera identity mismatch")
preflights = [json.loads((args.root / mode / "preflight.json").read_text()) for mode in ("sync", "async")]
check(preflights[0]["source_sha256"] == preflights[1]["source_sha256"], "cross-process source identity mismatch")
artifact_bytes = sum(p.stat().st_size for p in args.root.rglob("*") if p.is_file())
check(artifact_bytes < protocol["artifact_limit_bytes"], "artifact size budget")
report = {"pass": not failures, "failures": failures, "runs": results, "cross_process": cross,
          "initial_pixel_differences": pixel_differences, "settings_equal": settings_equal,
          "same_source_inventory": preflights[0]["source_sha256"] == preflights[1]["source_sha256"],
          "artifact_bytes": artifact_bytes, "limits": protocol["decision"]}
(args.root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key not in ("runs", "cross_process")}, indent=2))
raise SystemExit(bool(failures))
