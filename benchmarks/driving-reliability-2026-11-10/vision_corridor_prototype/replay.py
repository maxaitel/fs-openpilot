"""Read-only full-selection TRAIN replay. Writes only beside this script."""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from itertools import islice
from pathlib import Path
import json
import platform

import numpy as np

from fsop.eval.geometric import midline
from isaac_eval.evidence import read_evidence, verified_summary
from inputs import pooled_record_observation, recorded_pilot_observation
from planner import CorridorPlanner, CorridorResult, Diagnostics, Recipe, local_control, path_length
from scorer import ReferenceScorer

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PROTOCOL = json.loads((OUT / "recipe.json").read_text())
RECIPE = Recipe.load()
SELECTION = PROTOCOL["replay"]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def preserve_json(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"preserving a different existing result at {path}")
    else:
        path.write_text(encoded)


def check_freeze():
    frozen = json.loads((OUT / "freeze.json").read_text())
    for name, expected in frozen["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    return frozen


def baseline(obs):
    xy = np.array([(c.x, c.y) for c in obs.cones], dtype=float).reshape(-1, 2)
    colors = np.array([int(c.cone_class) for c in obs.cones], dtype=int)
    value = midline(xy, colors)
    path = tuple(tuple(float(x) for x in row) for row in value) if value is not None else ()
    control = local_control(path, obs.motion, RECIPE, len(path) >= 2)
    available = control.target_xy is not None
    diagnostics = Diagnostics(None, None, "baseline_ignores_route", 0, 0, 0, 0, 0, 0, (), None,
                              "baseline_path_available" if available else "baseline_no_path", None, ())
    return CorridorResult("supported" if available else "unknown", path, path_length(path), control, diagnostics)


def metric(values):
    a = np.asarray([x for x in values if x is not None], dtype=float)
    return None if not len(a) else {"count": len(a), "mae": float(np.mean(np.abs(a))),
                                   "p95": float(np.percentile(np.abs(a), 95)), "max_abs": float(np.max(np.abs(a)))}


def summarize(rows, dt):
    supported = [r for r in rows if r["result"]["status"] == "supported"]
    moving = [r for r in rows if r["score"]["speed_mps"] > RECIPE.stationary_speed_mps]
    unknown_intervals, pending = [], []
    for row in rows + [None]:
        if row is not None and row["result"]["status"] == "unknown":
            pending.append(row)
        elif pending:
            unknown_intervals.append({"start_frame": pending[0]["frame"], "end_frame": pending[-1]["frame"],
                                      "start_s": pending[0]["timestamp"], "end_s": pending[-1]["timestamp"],
                                      "sample_span_s": pending[-1]["timestamp"] - pending[0]["timestamp"],
                                      "nominal_duration_s": len(pending) * dt, "count": len(pending),
                                      "recovered_at_next_sample": row is not None,
                                      "minimum_available_stop_distance_m": min(x["result"]["control"]["available_stop_distance_m"] for x in pending),
                                      "maximum_required_stop_distance_m": max(x["result"]["control"]["required_stop_distance_m"] for x in pending)})
            pending = []
    commands = [r["score"]["command_error"] for r in rows if r["score"]["command_error"] is not None]
    steering = metric([x[0] for x in commands])
    acceleration = metric([x[1] for x in commands])
    correction = [r["score"]["teacher_correction_direction_agrees"] for r in rows
                  if r["score"]["teacher_correction_direction_agrees"] is not None]
    margins = [r["score"]["gt_sampled_footprint_margin_m"] for r in supported
               if r["score"]["gt_sampled_footprint_margin_m"] is not None]
    geom = metric([r["score"]["lateral_error_m"] for r in supported])
    raw_pass = None if steering is None or acceleration is None else all(
        item["mae"] <= SELECTION["raw_command_mae_limits"][i] and item["p95"] <= SELECTION["raw_command_p95_limits"][i]
        for i, item in enumerate((steering, acceleration)))
    return {
        "frames": len(rows), "moving_frames": len(moving), "supported_frames": len(supported),
        "supported_moving_frames": sum(r["result"]["status"] == "supported" for r in moving),
        "supported_fraction": len(supported) / len(rows) if rows else None,
        "reason_counts": dict(Counter(r["result"]["diagnostics"]["reason"] for r in rows)),
        "unknown_intervals": unknown_intervals,
        "longest_unknown_sample_span_s": max((x["sample_span_s"] for x in unknown_intervals), default=0.),
        "conditional_lateral_error_m": geom,
        "conditional_target_error_m": metric([r["score"]["target_error_m"] for r in supported]),
        "conditional_heading_error_rad": metric([r["score"]["heading_error_rad"] for r in supported]),
        "supported_wrong_branch_proxy_count": sum(r["score"]["wrong_branch_proxy"] is True for r in supported),
        "semantic_branch_qualification": "not established by the heading-error proxy",
        "supported_gt_footprint_conflict_count": sum(x <= 0 for x in margins),
        "supported_gt_footprint_min_margin_m": min(margins, default=None),
        "teacher_correction_direction_agreement_all": {"agree": sum(correction), "eligible": len(correction)},
        "all_row_raw_steering_error_rad": steering,
        "all_row_raw_acceleration_error_mps2": acceleration,
        "raw_command_screen_pass": raw_pass,
        "geometry_conditional_screen_pass": None if geom is None else geom["mae"] <= SELECTION["geometry_lateral_mae_limit_m"] and geom["p95"] <= SELECTION["geometry_lateral_p95_limit_m"],
        "unknown_rows_with_nonnegative_acceleration": sum(r["result"]["status"] == "unknown" and r["result"]["control"]["accel_mps2"] >= 0 for r in rows),
        "rows_with_stop_distance_shortfall": sum(r["result"]["control"]["required_stop_distance_m"] > r["result"]["control"]["available_stop_distance_m"] for r in rows),
    }


def run_episode(name, observations, scorer, records=None, ticks=None, cadence=.2):
    streaming = CorridorPlanner(RECIPE)
    all_rows = {"streaming": [], "stateless_gates": [], "existing_midline": []}
    for i, obs in enumerate(observations):
        results = {"streaming": streaming.step(obs), "stateless_gates": CorridorPlanner(RECIPE).step(obs),
                   "existing_midline": baseline(obs)}
        for variant, result in results.items():
            score = scorer.score(result, i if ticks is None else ticks[i], None if records is None else records[i])
            all_rows[variant].append({"frame": i, "timestamp": obs.timestamp, "tick": i if ticks is None else int(ticks[i]),
                                      "result": asdict(result), "score": score})
    summaries = {}
    for variant, rows in all_rows.items():
        summaries[variant] = summarize(rows, cadence)
        summaries[variant]["by_phase"] = {phase: summarize([r for r in rows if r["score"]["phase"] == phase], cadence)
                                           for phase in sorted({r["score"]["phase"] for r in rows})}
        summaries[variant]["by_20hz_phase"] = {str(phase): summarize([r for r in rows if r["tick"] % 4 == phase], cadence * 4 if cadence == .05 else cadence)
                                              for phase in range(4)}
    return {"episodes_are_independent": False, "cadence_s": cadence,
            "teacher_reconstruction_max_abs": scorer.reconstruction_max_abs,
            "full_episode_teacher_verified": scorer.teacher_verified,
            "variants": summaries}, all_rows


def main():
    frozen = check_freeze()
    input_hashes = {}
    episode_summaries, episode_rows = {}, {}
    for case in SELECTION["cases"]:
        path = ROOT / case["path"]
        summary = verified_summary(path / "evidence")
        first_contact = min(event["control_frame"] for event in summary["contact_events"])
        if first_contact != case["first_contact_frame"]:
            raise ValueError("first contact boundary changed")
        batch = list(islice(read_evidence(path / "evidence"), first_contact))
        metadata = batch[0][0]
        if metadata["case"]["split"] != "train" or metadata["case"]["id"] != SELECTION["case_id"]:
            raise ValueError("unexpected pilot membership")
        records = [record for _, record, _ in batch]
        if [r["frame"] for r in records] != list(range(first_contact)):
            raise ValueError("precontact frame coverage changed")
        for filename in ("metadata.json", "manifest.jsonl", "completion.json"):
            file = path / "evidence" / filename
            input_hashes[str(file.relative_to(ROOT))] = digest(file)
        diagnostic_name = "result.json" if path.name == "direct_command_pilot3" else "result_recovery2.json"
        diagnostic_path = ROOT / "benchmarks/driving-reliability-2026-11-10/direct_command_diagnosis" / diagnostic_name
        diagnostic = json.loads(diagnostic_path.read_text())
        input_hashes[str(diagnostic_path.relative_to(ROOT))] = digest(diagnostic_path)
        for relative, expected in diagnostic["input_sha256"].items():
            if relative in input_hashes and input_hashes[relative] != expected:
                raise ValueError("pilot diagnostic evidence binding changed")
        scorer = ReferenceScorer(metadata, RECIPE, SELECTION, teacher_params=diagnostic["source_teacher_parameters"])
        scorer.bind_pilot_labels(records, diagnostic)
        observations = [recorded_pilot_observation(r, SELECTION["route_activation_threshold"]) for r in records]
        summaries, rows = run_episode(path.name, observations, scorer, records=records, cadence=.05)
        summaries["first_contact_excluded_frame"] = first_contact
        episode_summaries[path.name], episode_rows[path.name] = summaries, rows
    clean = ROOT / Path(SELECTION["clean_records"]).parent
    receipt = json.loads((clean / "copy_receipt.json").read_text())
    if not receipt["all_passed"]:
        raise ValueError("input copy receipt failed")
    for item in receipt["files"]:
        path = ROOT / item["local"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"copied input hash mismatch: {path}")
        input_hashes[str(path.relative_to(ROOT))] = item["sha256"]
    extra_path = clean / "canonical_geometry_copy_receipt.json"
    extra = json.loads(extra_path.read_text())
    input_hashes[str(extra_path.relative_to(ROOT))] = digest(extra_path)
    for item in extra["files"]:
        path = ROOT / item["local"]
        if digest(path) != item["sha256"]:
            raise ValueError("extra canonical reference input changed")
        input_hashes[str(path.relative_to(ROOT))] = item["sha256"]
    source_rows = [json.loads(line) for line in (clean / "frames.jsonl").read_text().splitlines()]
    if set(r["episode"] for r in source_rows) != set(SELECTION["clean_episodes"]):
        raise ValueError("clean episode membership changed")
    for name, count in SELECTION["clean_episodes"].items():
        records = [r for r in source_rows if r["episode"] == name]
        if [r["frame"] for r in records] != list(range(count)):
            raise ValueError("clean stored-frame coverage changed")
        path = clean / "references_data" / name
        metadata = json.loads((path / "meta.json").read_text())
        if metadata["split"] != "train" or metadata["id"] != name:
            raise ValueError("unexpected clean TRAIN membership")
        arrays = {k: np.load(path / f"{k}.npy", allow_pickle=False) for k in ("states", "controls", "ticks", "desires", "cones", "contacts")}
        for key in arrays:
            if digest(path / f"{key}.npy") != metadata["arrays_sha256"][f"{key}.npy"]:
                raise ValueError("reference array changed")
        if np.any(arrays["contacts"]):
            raise ValueError("full recorded episode has physical contact")
        for i, row in enumerate(records):
            tick = int(arrays["ticks"][i])
            if row["tick"] != tick or row["time_s"] != arrays["states"][tick, 8]:
                raise ValueError("recorded timestamp mismatch")
            if not np.array_equal(row["vx_vy_yaw_rate"], arrays["states"][tick, 3:6]):
                raise ValueError("recorded proprioception mismatch")
            if not np.array_equal(row["desire_history"], arrays["desires"][i]):
                raise ValueError("recorded route history mismatch")
        flags_path = path / "teacher.npy"
        flags = np.load(flags_path, allow_pickle=False) if flags_path.exists() else None
        if flags is not None:
            if digest(flags_path) != metadata["arrays_sha256"]["teacher.npy"]:
                raise ValueError("teacher flags hash mismatch")
            input_hashes[str(flags_path.relative_to(ROOT))] = digest(flags_path)
        track_path = clean / "canonical_tracks/trackdrive_081000.json"
        if name.startswith("train-trackdrive") and track_path.exists():
            input_hashes[str(track_path.relative_to(ROOT))] = digest(track_path)
        else:
            track_path = None
        scorer = ReferenceScorer(metadata, RECIPE, SELECTION, states=arrays["states"], controls=arrays["controls"],
                                 flags=flags, cones=arrays["cones"], track_path=track_path)
        observations = [pooled_record_observation(r, SELECTION["desire_history_bin_s"], SELECTION["route_activation_threshold"]) for r in records]
        summaries, rows = run_episode(name, observations, scorer, ticks=arrays["ticks"], cadence=.2)
        summaries["collection_policy_mode"] = metadata.get("collection_policy", {}).get("mode")
        summaries["terminal_start_tick"] = metadata["terminal_continuation"]["start_tick"]
        episode_summaries[name], episode_rows[name] = summaries, rows
    result = {
        "scope": "Geometry feasibility on recorded TRAIN states. No dynamics rollout, physical qualification, or raw-command qualification.",
        "freeze": frozen, "input_sha256": input_hashes, "python": platform.python_version(), "numpy": np.__version__,
        "source_records_repeated_in_one_process": False,
        "total_frame_count": sum(v["variants"]["streaming"]["frames"] for v in episode_summaries.values()),
        "episodes": episode_summaries,
        "limitations": ["All errors are conditional on logged support unless explicitly marked all-row.",
                        "Unknown braking rows remain in coverage and command-error denominators.",
                        "A stopping-distance shortfall uses fixed assumptions and is not a validated physical stopping prediction.",
                        "The full trajectories contain learner intervals and terminal braking. They are not uniformly teacher-driven.",
                        "Clean RGB exists only at 5 Hz and 20 Hz phase zero. Other phases have zero clean coverage.",
                        "Wrong-branch count is a heading-error proxy. It does not certify lap or intersection semantics.",
                        "The old midline uses its unchanged origin anchor and 25 m range. Only its local diagnostic controller is shared.",
                        "External progress-derived route pulses are supplied. No on-car lap counter or finish detector is claimed.",
                        "AS mission braking remains outside this prototype. Unknown braking must not be overridden by launch logic."]
    }
    preserve_json(OUT / "result.json", result)
    preserve_json(OUT / "rows.json", episode_rows)
    print(json.dumps({name: {variant: {k: v for k, v in s.items() if k in ("frames", "supported_frames", "reason_counts", "conditional_lateral_error_m", "supported_gt_footprint_conflict_count", "raw_command_screen_pass")}
                             for variant, s in ep["variants"].items()} for name, ep in episode_summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
