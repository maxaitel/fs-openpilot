"""Run a frozen native Lab evaluation plan, one fresh process per episode.

Plan JSON (paths are relative to the plan file):
  {"version": 1, "timeout_seconds": 7200,
   "cases": ["cases/development.json", {"path": "cases/test.json", "name": "test"}],
   "models": [{"name": "baseline", "checkpoint": "models/base.pt", "sha256": "..."}]}

Cases are executed in listed order, with models in listed order for each case.
The first invocation freezes case contents, checkpoint identities, runtime source
and graph hashes in OUT/plan.lock.json. Resuming accepts only previously completed,
fully sealed episodes with identical identities. Technical failures or abandoned
runs stop the suite and must be investigated in a new output directory. Scored
failures such as a timeout are valid completed episodes and remain in the suite.

The 25 GiB free-space reserve is checked before and during each subprocess; writes
between checks can cross it. This process never deletes or archives evidence and
does not produce videos. Use --max-new-cases to pause between batches for external
verified archival. Archived evidence must remain available here to validate resume.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[1]
RESERVE_BYTES = 25 * 2**30
GRAPH_FILES = ("artifacts/models/prep/vision_fp32.onnx", "artifacts/models/prep/policy_fp32.onnx")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")
CASE_DEFAULTS = {"dr": "heavy", "variant": "visual_stress_v1", "track_seed": None, "frozen_case": None,
                 "lateral_start": 0.0, "yaw_start": 0.0}
CASE_REQUIRED = {"id", "split", "discipline", "seed"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _name(value, label):
    if not isinstance(value, str) or not NAME.fullmatch(value) or value in (".", ".."):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _resolve(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Plan paths must be nonempty strings")
    return (base / value).resolve()


def source_inventory(repo=REPO):
    paths = sorted(p for root in ("src/fsop", "isaac_eval", "isaac_lab_fsop")
                   for p in (repo / root).rglob("*.py"))
    if not paths:
        raise ValueError("Runtime source inventory is empty")
    return {str(p.relative_to(repo)): digest(p) for p in paths}


def normalize_case(data):
    if not isinstance(data, dict):
        raise ValueError("Case JSON must be an object")
    if not CASE_REQUIRED <= data.keys() or data.keys() - CASE_REQUIRED - CASE_DEFAULTS.keys():
        raise ValueError("Case fields do not match isaac_lab_fsop.cases.Case")
    case = {**CASE_DEFAULTS, **data}
    _name(case["id"], "case ID")
    if case["split"] not in ("regression", "test", "val"):
        raise ValueError("Evaluation cases must use regression, test, or val split")
    return case


def load_plan(path, repo=REPO):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    if set(plan) != {"version", "timeout_seconds", "cases", "models"} or plan["version"] != 1:
        raise ValueError("Plan requires version 1, timeout_seconds, cases, and models")
    timeout = plan["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("timeout_seconds must be positive and finite")
    if not isinstance(plan["cases"], list) or not plan["cases"]:
        raise ValueError("Plan cases must be a nonempty ordered list")
    if not isinstance(plan["models"], list) or not plan["models"]:
        raise ValueError("Plan models must be a nonempty ordered list")
    cases, models = [], []
    for entry in plan["cases"]:
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or not {"path"} <= entry.keys() or entry.keys() - {"path", "name"}:
            raise ValueError("Case entries must be a path or an object with path and optional name")
        case_path = _resolve(path.parent, entry["path"])
        case = normalize_case(json.loads(case_path.read_text()))
        cases.append({"name": _name(entry.get("name", case["id"]), "case output name"),
                      "path": str(case_path), "file_sha256": digest(case_path), "case": case})
    for entry in plan["models"]:
        if not isinstance(entry, dict) or set(entry) != {"name", "checkpoint", "sha256"}:
            raise ValueError("Models require name, checkpoint and sha256")
        name = _name(entry["name"], "model name")
        checkpoint = _resolve(path.parent, entry["checkpoint"])
        expected = entry["sha256"]
        if not isinstance(expected, str) or not SHA.fullmatch(expected) or digest(checkpoint) != expected:
            raise ValueError(f"Checkpoint SHA-256 mismatch for {name}")
        models.append({"name": name, "checkpoint": str(checkpoint), "sha256": expected})
    if any(model["name"] in {"cases", "logs"} for model in models):
        raise ValueError("Model names cannot use reserved output directories")
    if len({c["name"] for c in cases}) != len(cases) or len({m["name"] for m in models}) != len(models):
        raise ValueError("Case and model output names must be unique within their lists")
    return {"version": 1, "plan_path": str(path), "plan_sha256": digest(path), "timeout_seconds": timeout,
            "cases": cases, "models": models, "source_sha256": source_inventory(repo),
            "graph_sha256": {name: digest(repo / name) for name in GRAPH_FILES},
            "coordinator_sha256": digest(__file__)}


def atomic_json(path, value):
    path = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".manifest-", delete=False) as temp:
        name = temp.name
        try:
            temp.write(payload)
            temp.flush()
            os.fsync(temp.fileno())
        except BaseException:
            os.unlink(name)
            raise
    os.replace(name, path)


def prepare(out, lock):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_path, manifest_path = out / "plan.lock.json", out / "manifest.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("Frozen plan, checkpoints, sources, or graphs changed")
        if not manifest_path.is_file():
            raise ValueError("Frozen plan exists without its manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("lock_sha256") != digest(lock_path):
            raise ValueError("Manifest belongs to a different frozen plan")
    else:
        if any(path.name != ".coordinator.lock" for path in out.iterdir()):
            raise ValueError("Output directory is not empty and has no frozen plan")
        with lock_path.open("x") as stream:
            json.dump(lock, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        manifest = {"version": 1, "lock_sha256": digest(lock_path), "episodes": {}, "complete": False}
        atomic_json(manifest_path, manifest)
    return out, manifest


def validate_episode(directory, case, model, lock):
    """Validate every sealed frame byte and binding before accepting a prior run."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Episode output must be a real directory")
    evidence = directory / "evidence"
    metadata_bytes = (evidence / "metadata.json").read_bytes()
    manifest_bytes = (evidence / "manifest.jsonl").read_bytes()
    metadata = json.loads(metadata_bytes)
    receipt = json.loads((evidence / "completion.json").read_text())
    entries = [json.loads(line) for line in manifest_bytes.splitlines()]
    metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
    if (receipt.get("version") != 1 or receipt.get("frame_count") != len(entries)
            or receipt.get("metadata_sha256") != metadata_sha
            or receipt.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()):
        raise ValueError("Evidence completion receipt mismatch")
    if (evidence / receipt["summary_path"]).resolve() != (directory / "summary.json").resolve():
        raise ValueError("Evidence receipt does not bind episode summary")
    summary_path = directory / "summary.json"
    if digest(summary_path) != receipt.get("summary_sha256"):
        raise ValueError("Summary digest mismatch")
    summary = json.loads(summary_path.read_text())
    expected_frames = {f"frame_{i:06d}.npz" for i in range(len(entries))}
    if {p.name for p in evidence.glob("frame_*.npz")} != expected_frames:
        raise ValueError("Missing or extra evidence frames")
    if not 0 < len(entries) <= 2400:
        raise ValueError("Invalid scored frame count")
    # Decode the evidence records as well as checking their file hashes. The
    # reader validates NPZ structure and recomputes each inference-input hash.
    from isaac_eval.evidence import read_evidence

    for index, (_, record, entry) in enumerate(read_evidence(evidence)):
        if (record.get("frame") != index or not isinstance(entry.get("input_sha256"), str)
                or not SHA.fullmatch(entry["input_sha256"])):
            raise ValueError(f"Invalid evidence frame {index}")
    if (metadata.get("case") != case["case"] or summary.get("case") != case["case"]
            or json.loads((directory / "case.json").read_text()) != case["case"]):
        raise ValueError("Episode case differs from the frozen plan")
    if (metadata.get("backend") != "Isaac Lab PhysX" or metadata.get("fps") != 20
            or metadata.get("pilot_only") is not False or summary.get("pilot_only") is not False):
        raise ValueError("Episode is not a scored native Lab run")
    if (metadata.get("checkpoint_sha256") != model["sha256"] or summary.get("checkpoint_sha256") != model["sha256"]
            or summary.get("checkpoint_sha256_after") != model["sha256"]
            or summary.get("state_sha256") != metadata.get("state_sha256")
            or summary.get("state_sha256_after") != metadata.get("state_sha256")
            or summary.get("frozen_unchanged") is not True):
        raise ValueError("Checkpoint or model-state identity mismatch")
    if metadata.get("source_sha256") != lock["source_sha256"] or metadata.get("graph_sha256") != lock["graph_sha256"]:
        raise ValueError("Episode source or graph identity differs from frozen plan")
    state_sha = metadata.get("state_sha256")
    if not isinstance(state_sha, str) or not SHA.fullmatch(state_sha):
        raise ValueError("Episode model-state digest is invalid")
    for key, filename in (("source_archive_sha256", "source.zip"), ("realization_sha256", "realization.json"),
                          ("appearance_sha256", "appearance.json"), ("scene_sha256", "scene.usda"),
                          ("export_sha256", "export.json")):
        if metadata.get(key) != digest(directory / filename):
            raise ValueError(f"Episode artifact mismatch: {filename}")
    with ZipFile(directory / "source.zip") as archive:
        if sorted(archive.namelist()) != sorted(lock["source_sha256"]):
            raise ValueError("Source archive inventory mismatch")
        for name, expected in lock["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Source archive contents differ from frozen plan")
    if json.loads((directory / "track.json").read_text()) != metadata.get("track"):
        raise ValueError("Episode track differs from metadata")
    export = json.loads((directory / "export.json").read_text())
    if sorted(export["dependencies"]) != sorted(export["sha256"]):
        raise ValueError("Scene asset inventory mismatch")
    for name, expected in export["sha256"].items():
        asset = (directory / "assets" / name).resolve()
        if not asset.is_relative_to((directory / "assets").resolve()) or digest(asset) != expected:
            raise ValueError("Scene asset digest mismatch or path escape")
    rows = json.loads((directory / "frames.json").read_text())
    if (not isinstance(rows, list) or len(rows) != len(entries) or summary.get("recorded_frames") != len(entries)
            or summary.get("result") in (None, "running")
            or summary.get("final_state", {}).get("t") != len(entries) / 20):
        raise ValueError("Episode is incomplete or its frame count is inconsistent")
    json.loads((directory / "telemetry.json").read_text())
    return {"frame_count": len(entries), "result": summary["result"],
            "completion_sha256": digest(evidence / "completion.json"), "summary_sha256": digest(summary_path),
            "frames_sha256": digest(directory / "frames.json")}


def check_inputs(lock, repo=REPO):
    if source_inventory(repo) != lock["source_sha256"]:
        raise ValueError("Runtime source inventory changed during the suite")
    if {name: digest(repo / name) for name in GRAPH_FILES} != lock["graph_sha256"]:
        raise ValueError("Model graphs changed during the suite")
    if digest(lock["plan_path"]) != lock["plan_sha256"] or digest(__file__) != lock["coordinator_sha256"]:
        raise ValueError("Plan or coordinator changed during the suite")
    for case in lock["cases"]:
        if digest(case["path"]) != case["file_sha256"]:
            raise ValueError("Case file changed during the suite")
    for model in lock["models"]:
        if digest(model["checkpoint"]) != model["sha256"]:
            raise ValueError("Checkpoint changed during the suite")


def _stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Kill the entire group even if the leader exited after TERM.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_process(command, logfile, timeout, disk_path, reserve_bytes=RESERVE_BYTES):
    with Path(logfile).open("x") as log:
        proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        try:
            while True:
                if shutil.disk_usage(disk_path).free < reserve_bytes:
                    raise RuntimeError("Free space fell below the 25 GiB reserve")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Episode process exceeded its timeout")
                try:
                    status = proc.wait(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if shutil.disk_usage(disk_path).free < reserve_bytes:
                raise RuntimeError("Free space fell below the 25 GiB reserve")
            if status != 0:
                raise RuntimeError(f"Episode process exited with status {status}")
        except BaseException:
            _stop_group(proc)
            raise
    return status


def run(plan_path, out, max_new_cases=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(out / ".coordinator.lock", flags, 0o600)
    with os.fdopen(descriptor, "r+") as suite_lock:
        try:
            fcntl.flock(suite_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another coordinator owns this output directory") from error
        return _run_locked(plan_path, out, max_new_cases)


def _run_locked(plan_path, out, max_new_cases=None):
    if max_new_cases is not None and max_new_cases < 1:
        raise ValueError("--max-new-cases must be positive")
    lock = load_plan(plan_path)
    out, manifest = prepare(out, lock)
    manifest_path = out / "manifest.json"
    ordered = [(case, model) for case in lock["cases"] for model in lock["models"]]
    keys = {f'{model["name"]}/{case["name"]}' for case, model in ordered}
    if set(manifest["episodes"]) - keys:
        raise ValueError("Manifest contains an episode outside the frozen plan")
    # Validate all previous episodes first: never pass over a technical failure or
    # a partially written output, even when the current batch limit is reached.
    for case, model in ordered:
        key = f'{model["name"]}/{case["name"]}'
        directory = out / model["name"] / case["name"]
        row = manifest["episodes"].get(key)
        if row is None:
            if directory.exists() or directory.is_symlink():
                raise ValueError(f"Untracked episode output requires inspection: {key}")
            continue
        if row.get("status") != "complete":
            raise ValueError(f"Episode has an incomplete or failed execution: {key}")
        if validate_episode(directory, case, model, lock) != row.get("receipt"):
            raise ValueError(f"Completed episode receipt changed: {key}")
    check_inputs(lock)
    count = 0
    for case, model in ordered:
        key = f'{model["name"]}/{case["name"]}'
        if key in manifest["episodes"]:
            continue
        if max_new_cases is not None and count >= max_new_cases:
            break
        check_inputs(lock)
        if shutil.disk_usage(out).free < RESERVE_BYTES:
            raise RuntimeError("Insufficient free space to preserve the 25 GiB reserve")
        model_dir = out / model["name"]
        log_dir = out / "logs" / model["name"]
        case_dir = out / "cases"
        for path in (model_dir, log_dir, case_dir):
            path.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise ValueError("Output directories cannot be symlinks")
        snapshot = case_dir / f'{case["name"]}.json'
        if snapshot.exists():
            if json.loads(snapshot.read_text()) != case["case"]:
                raise ValueError("Saved case snapshot differs from the frozen plan")
        else:
            with snapshot.open("x") as stream:
                json.dump(case["case"], stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        directory = model_dir / case["name"]
        logfile = log_dir / f'{case["name"]}.log'
        command = [sys.executable, "-m", "isaac_lab_fsop.evaluate", "--case", str(snapshot),
                   "--checkpoint", model["checkpoint"], "--expected-sha", model["sha256"],
                   "--out", str(directory), "--async-evidence"]
        manifest["episodes"][key] = {"status": "running", "command": command, "log": str(logfile.relative_to(out))}
        atomic_json(manifest_path, manifest)
        try:
            run_process(command, logfile, lock["timeout_seconds"], out)
            check_inputs(lock)
            receipt = validate_episode(directory, case, model, lock)
            manifest["episodes"][key].update(status="complete", receipt=receipt)
            count += 1
        except BaseException as error:
            manifest["episodes"][key].update(status="failed", error=f"{type(error).__name__}: {error}")
            atomic_json(manifest_path, manifest)
            raise
        atomic_json(manifest_path, manifest)
        print("EPISODE", key, json.dumps(receipt), flush=True)
    manifest["complete"] = len(manifest["episodes"]) == len(ordered) and all(
        row.get("status") == "complete" for row in manifest["episodes"].values())
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-cases", type=int)
    args = parser.parse_args()
    manifest = run(args.plan, args.out, args.max_new_cases)
    print(json.dumps({"complete": manifest["complete"], "completed": len(manifest["episodes"])}))


if __name__ == "__main__":
    main()
