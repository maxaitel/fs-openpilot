"""Run a frozen, sequential development grid for the native skidpad teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (82000, 82001, 82002, 82003)
CANDIDATES = (
    {"id": "baseline", "controller": "pure_pursuit", "speed": 8.0, "alat": 4.0, "lookahead": 0.25},
    {"id": "short_pp", "controller": "pure_pursuit", "speed": 8.0, "alat": 4.0, "lookahead": 0.15},
    {"id": "stanley", "controller": "stanley_ff", "speed": 8.0, "alat": 4.0, "lookahead": 0.25},
    {"id": "lower_alat", "controller": "pure_pursuit", "speed": 8.0, "alat": 3.0, "lookahead": 0.25},
)


def teacher_parameters(candidate):
    return {
        "a_lat_max": candidate["alat"], "a_acc": 5.0, "a_brk": 8.0, "v_cap": candidate["speed"],
        "k_lookahead": candidate["lookahead"], "lookahead_min": 1.8, "lookahead_max": 9.0,
        "kp_speed": 1.5, "stop_decel": 4.0, "controller": candidate["controller"],
        "k_lat": 2.0, "k_head": 0.6, "t_preview": 0.12, "stop_after_line": 12.0,
    }


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_inventory():
    paths = sorted(p for root in ("src/fsop", "isaac_eval", "isaac_lab_fsop") for p in (ROOT / root).rglob("*.py"))
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_cases(membership):
    rows = json.loads(membership.read_text())["cases"]
    cases = []
    for seed in SEEDS:
        found = [case for case in rows if case.get("split") == "train" and case.get("discipline") == "skidpad"
                 and case.get("seed") == seed]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one training skidpad case for seed {seed}")
        case = found[0]
        if (case.get("frozen_case") is not None or case.get("track_seed") is not None
                or case.get("lateral_start") != 0.0 or case.get("yaw_start") != 0.0):
            raise ValueError(f"Seed {seed} is not the prescribed nominal fixed skidpad case")
        if not isinstance(case.get("id"), str) or Path(case["id"]).name != case["id"]:
            raise ValueError(f"Invalid case identity for seed {seed}")
        cases.append(case)
    return cases


def verify_archive(path, source_hashes, expected_digest):
    if digest(path) != expected_digest:
        raise ValueError("Source archive digest mismatch")
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(source_hashes):
            raise ValueError("Source archive inventory mismatch")
        for name, expected in source_hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Source archive content mismatch: {name}")


def read_probe(directory, case, candidate, source_hashes):
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["case_config"] != case:
        raise ValueError("Probe case_config does not match frozen case")
    if summary["source_sha256"] != source_hashes:
        raise ValueError("Probe source inventory differs from frozen inventory")
    verify_archive(directory / "source.zip", source_hashes, summary["source_archive_sha256"])
    expert = summary["expert"]
    if expert != teacher_parameters(candidate):
        raise ValueError("Probe teacher parameters differ from frozen candidate")
    if summary["physics_backend"] != "isaaclab.sim.simulation_context":
        raise ValueError("Unexpected simulator backend")
    rows = np.load(directory / "states.npy", mmap_mode="r", allow_pickle=False)
    if rows.ndim != 2 or rows.shape[1] != 14 or not 0 < len(rows) <= 2400 or not np.isfinite(rows).all():
        raise ValueError("Invalid recorded teacher state rows")
    if not np.allclose(rows[:, 8], np.arange(len(rows)) / 20, rtol=0, atol=1e-5):
        raise ValueError("Teacher state timestamps are not consecutive 20 Hz ticks")
    final_time = float(summary["final_state"]["t"])
    rms_lat, max_speed = float(summary["rms_lat"]), float(summary["max_speed"])
    if (not all(math.isfinite(x) and x >= 0 for x in (final_time, rms_lat, max_speed))
            or not math.isclose(final_time, len(rows) / 20, rel_tol=0, abs_tol=1e-5)):
        raise ValueError("Invalid summary time or metric")
    contacts, events, off_course = summary["physical_cone_hits"], summary["contact_events"], summary["oc"]
    if (type(contacts) is not int or contacts < 0 or type(off_course) is not int or off_course < 0
            or not isinstance(events, list) or len(events) != contacts
            or len({event["cone"] for event in events}) != contacts):
        raise ValueError("Invalid physical contact or off-course summary")
    physics_hash = summary["appearance"]["physics_sha256"]
    if not isinstance(physics_hash, str) or len(physics_hash) != 64:
        raise ValueError("Missing physical scene digest")
    return {
        "summary_sha256": digest(summary_path),
        "states_sha256": digest(directory / "states.npy"),
        "source_archive_sha256": summary["source_archive_sha256"],
        "physics_sha256": physics_hash,
        "result": summary["result"],
        "pass": summary["pass"],
        "complete": summary["complete"],
        "physical_cone_hits": contacts,
        "contact_events": events,
        "oc": off_course,
        "doo": summary["doo"],
        "rms_lat": rms_lat,
        "max_abs_lat": float(np.max(np.abs(rows[:, 10]))),
        "max_speed": max_speed,
        "mission_time": final_time,
        "lap_times": summary["laps"],
        "wall_seconds": summary["wall_seconds"],
        "teacher": expert,
    }


def aggregate(runs):
    results = []
    for candidate in CANDIDATES:
        entries = [run for run in runs if run["candidate"] == candidate["id"]]
        valid = len(entries) == len(SEEDS) and all(run.get("verified", False) for run in entries)
        metrics = [run["metrics"] for run in entries if run.get("verified", False)]
        passes = valid and all(m["result"] == "finished" and m["pass"] is True and m["complete"] is True
                               and m["physical_cone_hits"] == 0 and m["oc"] == 0 for m in metrics)
        results.append({
            "candidate": candidate["id"],
            "runs": len(entries),
            "verified_runs": len(metrics),
            "gate_passed": passes,
            "selection_eligible": passes and candidate["id"] != "baseline",
            "total_contacts": sum(m["physical_cone_hits"] for m in metrics) if metrics else None,
            "total_oc": sum(m["oc"] for m in metrics) if metrics else None,
            "total_doo": sum(m["doo"] for m in metrics) if metrics else None,
            "mean_mission_time": sum(m["mission_time"] for m in metrics) / len(metrics) if valid else None,
            "worst_rms_lat": max((m["rms_lat"] for m in metrics), default=None),
            "max_abs_lat": max((m["max_abs_lat"] for m in metrics), default=None),
            "max_speed": max((m["max_speed"] for m in metrics), default=None),
        })
    eligible = [row for row in results if row["selection_eligible"]]
    selected = min(eligible, key=lambda row: row["mean_mission_time"])["candidate"] if eligible else None
    return results, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--membership", type=Path, default=ROOT / "docs/isaac-lab-training/membership.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    membership = args.membership.resolve()
    cases = load_cases(membership)
    source_hashes = source_inventory()
    script_hash = digest(__file__)
    outroot = args.outroot.resolve()
    outroot.mkdir(parents=True, exist_ok=False)
    (outroot / "cases").mkdir()
    (outroot / "logs").mkdir()
    for case in cases:
        save_json(outroot / "cases" / f'{case["id"]}.json', case)
    protocol = {
        "schema_version": 1,
        "purpose": "Development-only native teacher skidpad grid; not teacher qualification",
        "membership": str(membership), "membership_sha256": digest(membership),
        "script_sha256": script_hash, "source_sha256": source_hashes,
        "cases": cases, "candidates": list(CANDIDATES),
        "teacher_parameters": {candidate["id"]: teacher_parameters(candidate) for candidate in CANDIDATES},
        "interpreter": sys.executable, "control_hz": 20, "max_steps": 2400,
        "render": False, "timeout_seconds": args.timeout,
        "gate": "Every case finished, passed, complete, zero physical contacts and zero off-course transitions",
        "selection": "Among passing alternatives only, minimum mean native mission final time; ties use grid order",
        "baseline_exclusion": "Previously observed physical contacts; retained as a repeatability control only",
    }
    save_json(outroot / "protocol.json", protocol)
    protocol_hash = digest(outroot / "protocol.json")
    runs = []
    interrupted = None
    for candidate in CANDIDATES:
        if interrupted:
            break
        for case in cases:
            if (source_inventory() != source_hashes or digest(__file__) != script_hash
                    or digest(outroot / "protocol.json") != protocol_hash):
                interrupted = "Source or protocol changed after protocol freeze"
                break
            case_file = outroot / "cases" / f'{case["id"]}.json'
            if json.loads(case_file.read_text()) != case or digest(membership) != protocol["membership_sha256"]:
                interrupted = "Case or membership changed after protocol freeze"
                break
            directory = outroot / candidate["id"] / case["id"]
            directory.parent.mkdir(exist_ok=True)
            log_path = outroot / "logs" / f'{candidate["id"]}--{case["id"]}.log'
            command = [sys.executable, "-m", "isaac_lab_fsop.probe_teacher", "--case-file", str(case_file),
                       "--out", str(directory), "--steps", "2400", "--controller", candidate["controller"],
                       "--speed", str(candidate["speed"]), "--alat", str(candidate["alat"]),
                       "--lookahead", str(candidate["lookahead"])]
            run = {"candidate": candidate["id"], "case_id": case["id"], "command": command,
                   "log": str(log_path.relative_to(outroot)), "output": str(directory.relative_to(outroot)),
                   "verified": False}
            try:
                with log_path.open("x") as log:
                    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=args.timeout, check=False)
                run["returncode"] = completed.returncode
                if completed.returncode != 0:
                    run["error"] = f"Probe exited with status {completed.returncode}"
                if (directory / "summary.json").is_file():
                    run["metrics"] = read_probe(directory, case, candidate, source_hashes)
                    run["verified"] = completed.returncode == 0
                elif completed.returncode == 0:
                    run["error"] = "Probe did not produce summary.json"
            except subprocess.TimeoutExpired:
                run["error"] = f"Probe exceeded {args.timeout:g} seconds"
                if (directory / "summary.json").is_file():
                    try:
                        run["metrics"] = read_probe(directory, case, candidate, source_hashes)
                    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as error:
                        run["verification_error"] = f"{type(error).__name__}: {error}"
            except (OSError, ValueError, KeyError, TypeError, BadZipFile) as error:
                run["error"] = f"{type(error).__name__}: {error}"
            runs.append(run)
            aggregates, _ = aggregate(runs)
            save_json(outroot / "progress.json", {"protocol_sha256": protocol_hash, "runs": runs,
                                                   "candidates": aggregates})
    if (source_inventory() != source_hashes or digest(__file__) != script_hash
            or digest(outroot / "protocol.json") != protocol_hash
            or digest(membership) != protocol["membership_sha256"]
            or any(json.loads((outroot / "cases" / f'{case["id"]}.json').read_text()) != case for case in cases)):
        interrupted = "Source, protocol, membership or case changed after protocol freeze"
    for case in cases:
        matching = [run for run in runs if run["case_id"] == case["id"] and run.get("verified")]
        if len({run["metrics"]["physics_sha256"] for run in matching}) > 1:
            for run in matching:
                run["verified"] = False
                run["verification_error"] = "Physical scene digest differs across candidates for this case"
    aggregates, selected = aggregate(runs)
    if interrupted or len(runs) != len(CANDIDATES) * len(SEEDS):
        selected = None
    report = {"protocol_sha256": protocol_hash, "complete": len(runs) == 16 and interrupted is None,
              "interrupted": interrupted, "runs": runs, "candidates": aggregates, "selected": selected,
              "selection_note": "Development probe only; additional qualification and retention gates are required."}
    save_json(outroot / "report.json", report)
    print(json.dumps({"report": str(outroot / "report.json"), "complete": report["complete"],
                      "selected": selected}), flush=True)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
