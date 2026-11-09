"""Verify a sealed native learner-prefix episode against the frozen recovery recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from fsop.data.native_replay import NativeEpisode
from fsop.sim.expert import ExpertParams, Mission
from fsop.track.fixed_layouts import skidpad
from fsop.track.generate import TrackGenCfg, generate_track
from fsop.track.io import load_json


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def route_states(mission, states):
    hint, progress = mission.init_progress(float(states[0, 0]), float(states[0, 1]))
    result = []
    for tick, state in enumerate(states):
        hint, progress, lat = mission.update_progress(float(state[0]), float(state[1]), hint, progress)
        segment = mission._seg[hint]
        heading = math.atan2(float(segment[1]), float(segment[0]))
        yaw = math.atan2(math.sin(float(state[2]) - heading), math.cos(float(state[2]) - heading))
        result.append({"tick": tick, "progress_m": progress, "segment": hint, "lateral_m": lat, "yaw_rad": yaw})
    return result


def verify(episode_path, recipe_path):
    episode_path, recipe_path = Path(episode_path).resolve(), Path(recipe_path).resolve()
    report = {
        "schema_version": 1,
        "passed": False,
        "episode": str(episode_path),
        "recipe": str(recipe_path),
        "reasons": [],
        "hashes": {},
    }
    try:
        report["hashes"]["recipe_sha256"] = digest(recipe_path)
        recipe = json.loads(recipe_path.read_text())
        require(recipe.get("schema_version") == 1, "Invalid recipe schema")
        collection = recipe.get("collection")
        allowed_collections = [
            {"takeover_seconds": 8, "lateral_threshold_m": 0.6, "capture_hz": 20, "image_hz": 5},
            {"takeover_seconds": 120, "lateral_threshold_m": 0.4, "capture_hz": 20, "image_hz": 5},
        ]
        gate = {
            "min_lateral_m": 0.3,
            "min_yaw_rad": 0.1,
            "min_anchors": 8,
            "window_seconds": 2,
            "return_within_seconds": 4,
            "settled_lateral_m": 0.2,
            "settled_yaw_rad": 0.1,
            "settled_seconds": 0.5,
        }
        require(
            collection in allowed_collections and recipe.get("gate") == gate,
            "Recipe collection or gate differs from the frozen protocol",
        )
        ids = recipe.get("recovery_case_ids")
        require(
            isinstance(ids, list)
            and len(ids) == 8
            and len(set(ids)) == 8
            and all(isinstance(x, str) and x for x in ids),
            "Invalid recovery case IDs",
        )
        membership_path = (recipe_path.parent / recipe["membership"]).resolve()
        report["hashes"]["membership_sha256"] = digest(membership_path)
        require(
            report["hashes"]["membership_sha256"] == recipe.get("membership_sha256"),
            "Membership digest differs from recipe",
        )
        membership = json.loads(membership_path.read_text())
        ep = NativeEpisode(episode_path)
        m = ep.meta
        report["hashes"]["meta_sha256"] = digest(episode_path / "meta.json")
        report["hashes"]["arrays_sha256"] = m["arrays_sha256"]
        report["case_id"] = m.get("id")
        cases = [c for c in membership["cases"] if c.get("id") == m.get("id")]
        require(
            m.get("id") in ids
            and len(cases) == 1
            and m.get("case") == cases[0]
            and m.get("membership_sha256") == recipe["membership_sha256"]
            and m.get("split") == "train"
            and cases[0].get("split") == "train"
            and m.get("discipline") == cases[0].get("discipline")
            and m.get("seed") == cases[0].get("seed"),
            "Episode is not the exact frozen training case",
        )
        case_path = recipe_path.parent / "cases" / (m["id"] + ".json")
        report["hashes"]["case_sha256"] = digest(case_path)
        require(
            report["hashes"]["case_sha256"] == recipe.get("case_sha256", {}).get(m["id"])
            and json.loads(case_path.read_text()) == cases[0],
            "Case file differs from frozen recipe or membership",
        )
        require(
            m.get("checkpoint_sha256") == recipe.get("checkpoint_sha256")
            and isinstance(m.get("checkpoint_sha256"), str)
            and len(m["checkpoint_sha256"]) == 64,
            "Learner checkpoint digest differs from recipe",
        )
        require(m.get("teacher") == recipe.get("teachers", {}).get(m["id"]), "Teacher parameters differ from recipe")
        archive_path = episode_path / "source.zip"
        report["hashes"]["source_archive_sha256"] = digest(archive_path)
        require(
            report["hashes"]["source_archive_sha256"] == m.get("source_archive_sha256"),
            "Source archive digest differs from metadata",
        )
        with zipfile.ZipFile(archive_path) as archive:
            inventory = m["source_sha256"]
            require(
                isinstance(inventory, dict) and bool(inventory) and sorted(archive.namelist()) == sorted(inventory),
                "Source archive inventory differs",
            )
            for name, expected in inventory.items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == expected, f"Source digest differs for {name}")
        track_path = episode_path / "track.json"
        report["hashes"]["track_sha256"] = digest(track_path)
        track = load_json(track_path)
        case = cases[0]
        expected_track = (
            skidpad() if case["discipline"] == "skidpad" else generate_track(case["track_seed"], TrackGenCfg())
        )
        if case["discipline"] != "skidpad":
            expected_track.laps = 1
        require(
            track.to_dict() == expected_track.to_dict() and m["track_id"] == track.id,
            "Track differs from frozen case realization",
        )
        require(
            m.get("control_hz") == 20 and m.get("capture_hz") == 20 and m.get("image_hz") == 5,
            "Recorded control, capture, or image cadence differs from recipe",
        )
        policy = m.get("collection_policy", {})
        takeover = policy.get("actual_takeover_tick")
        require(
            policy.get("mode") == "learner_prefix"
            and policy.get("requested_takeover_s") == collection["takeover_seconds"]
            and policy.get("lateral_takeover_threshold_m") == collection["lateral_threshold_m"]
            and type(takeover) is int
            and 0 < takeover < len(ep.teacher),
            "Invalid learner-prefix policy or takeover tick",
        )
        report["takeover_tick"] = takeover
        require(
            not np.any(ep.teacher[:takeover]) and np.all(ep.teacher[takeover:]),
            "Teacher flags contradict a single learner-to-teacher handoff",
        )
        summary = m["summary"]
        require(
            summary.get("result") == "finished"
            and summary.get("pass") is True
            and summary.get("complete") is True
            and summary.get("physical_cone_hits") == 0
            and summary.get("oc") == 0
            and not np.any(ep.contacts),
            "Episode is not clean, complete, and finished",
        )
        terminal = m.get("terminal_continuation", {})
        terminal_tick = terminal.get("start_tick")
        require(
            type(terminal_tick) is int
            and takeover < terminal_tick <= len(ep.teacher)
            and terminal.get("end_tick") == len(ep.teacher)
            and terminal.get("end_tick") - terminal_tick == 200
            and terminal.get("contacts") == 0,
            "Invalid terminal continuation boundary or contacts",
        )
        mission = Mission(track, ExpertParams(**m["teacher"]))
        route = route_states(mission, ep.states[: min(terminal_tick, takeover + 81)])
        end = min(terminal_tick, takeover + 81)
        report["route_slice"] = route[takeover:end]
        report["takeover"] = route[takeover]
        first = next(
            (
                r["tick"]
                for r in route[: takeover + 1]
                if ep.states[r["tick"], 8] >= collection["takeover_seconds"]
                or abs(r["lateral_m"]) >= collection["lateral_threshold_m"]
            ),
            None,
        )
        report["first_trigger_tick"] = first
        require(first == takeover, "Takeover does not match the first time or lateral trigger")
        reason = policy.get("takeover_reason")
        expected_reason = (
            "lateral_threshold"
            if abs(route[takeover]["lateral_m"]) >= collection["lateral_threshold_m"]
            else "time_limit"
        )
        require(reason == expected_reason, "Recorded takeover reason differs from observed trigger")
        recovery_frames = [
            int(k)
            for k in ep.anchors
            if takeover <= int(ep.ticks[k]) < takeover + 40 and int(ep.ticks[k]) < terminal_tick
        ]
        report["recovery_frame_indices"] = recovery_frames
        report["recovery_ticks"] = [int(ep.ticks[k]) for k in recovery_frames]
        if len(recovery_frames) < gate["min_anchors"]:
            report["reasons"].append("Fewer than eight qualified expert recovery anchors")
        state = route[takeover]
        if abs(state["lateral_m"]) < 0.3 and abs(state["yaw_rad"]) < 0.1:
            report["reasons"].append("Takeover has no qualifying corrective lateral or yaw error")
        settled = None
        for start in range(takeover, end - 10):
            window = route[start : start + 11]
            if all(abs(r["lateral_m"]) <= 0.2 and abs(r["yaw_rad"]) <= 0.1 for r in window):
                settled = {
                    "start_tick": start,
                    "end_tick": start + 10,
                    "end_seconds_after_takeover": (start + 10 - takeover) / 20,
                }
                break
        report["settled"] = settled
        if settled is None:
            report["reasons"].append("Did not remain within lateral and yaw limits for 0.5 seconds within four seconds")
        report["terminal_start_tick"] = terminal_tick
        report["passed"] = not report["reasons"]
    except (OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as exc:
        report["reasons"].append(f"{type(exc).__name__}: {exc}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("receipt output already exists")
    if args.out.resolve().is_relative_to(args.episode.resolve()):
        parser.error("receipt output must be outside the sealed episode")
    report = verify(args.episode, args.recipe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "reasons": report["reasons"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
