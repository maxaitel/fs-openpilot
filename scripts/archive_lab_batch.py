#!/usr/bin/env python3
"""Copy one completed Lab coordinator batch and verify every archived byte."""

import argparse
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

RESERVE = 25 * 1024**3
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Batch root must be a real directory")
    entries = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries[relative] = {"type": "directory"}
            elif stat.S_ISREG(mode):
                with path.open("rb") as stream:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                entries[relative] = {"type": "file", "size": path.stat().st_size, "sha256": digest.hexdigest()}
            else:
                raise ValueError(f"Symlink or special file in batch: {relative}")
    return entries


def remote_inventory(host, root, batch):
    code = (
        """import hashlib, json, os, socket, stat, sys
from pathlib import Path
"""
        + inspect.getsource(inventory)
        + """
root = Path(sys.argv[1]).resolve(strict=True)
path = root
for part in sys.argv[2].split('/'):
    path = path / part
    if path.is_symlink():
        raise ValueError('Symlink in batch path')
path = path.resolve(strict=True)
if not path.is_relative_to(root) or path == root:
    raise ValueError('Batch escapes task root')
result = {'hostname': socket.gethostname(), 'root': str(root), 'batch': str(path), 'entries': inventory(path)}
print(json.dumps(result, sort_keys=True))
"""
    )
    command = " ".join(shlex.quote(arg) for arg in ("python3", "-c", code, root, batch))
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def validate_batch(root, entries):
    def file_hash(relative):
        item = entries.get(relative)
        if not item or item.get("type") != "file":
            raise ValueError(f"Missing regular file: {relative}")
        return item["sha256"]

    def read(relative):
        file_hash(relative)
        return json.loads((root / relative).read_text())

    lock, manifest = read("plan.lock.json"), read("manifest.json")
    if lock.get("version") != 1 or manifest.get("version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Batch plan or manifest is incomplete")
    if manifest.get("lock_sha256") != file_hash("plan.lock.json"):
        raise ValueError("Manifest does not bind plan lock")
    cases, models = lock["cases"], lock["models"]
    if not isinstance(cases, list) or not cases or not isinstance(models, list) or not models:
        raise ValueError("Empty plan")
    for row in cases + models:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not NAME.fullmatch(row["name"])
            or row["name"] in (".", "..")
        ):
            raise ValueError("Unsafe plan name")
    keys = {f"{model['name']}/{case['name']}" for model in models for case in cases}
    if len(keys) != len(models) * len(cases) or set(manifest["episodes"]) != keys:
        raise ValueError("Manifest episode membership differs from plan")
    bindings = {}
    cases_by_name = {case["name"]: case for case in cases}
    models_by_name = {model["name"]: model for model in models}
    for key in sorted(keys):
        model_name, case_name = key.split("/")
        row = manifest["episodes"][key]
        if row.get("status") != "complete":
            raise ValueError(f"Incomplete episode: {key}")

        def h(name, key=key):
            return file_hash(f"{key}/{name}")

        def j(name, key=key):
            return read(f"{key}/{name}")

        completion = j("evidence/completion.json")
        audit = j("lab_artifact_audit.json")
        video = j("lab_video_audit.json")
        count = completion.get("frame_count")
        if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= 2400:
            raise ValueError(f"Invalid frame count: {key}")
        if (
            completion.get("version") != 1
            or completion.get("summary_path") != "../summary.json"
            or completion.get("metadata_sha256") != h("evidence/metadata.json")
            or completion.get("manifest_sha256") != h("evidence/manifest.jsonl")
            or completion.get("summary_sha256") != h("summary.json")
        ):
            raise ValueError(f"Completion seal mismatch: {key}")
        frames = [json.loads(line) for line in (root / key / "evidence/manifest.jsonl").read_text().splitlines()]
        frame_paths = set()
        for index, frame in enumerate(frames):
            name = f"frame_{index:06d}.npz"
            if (
                frame.get("frame") != index
                or frame.get("file") != name
                or frame.get("metadata_sha256") != completion["metadata_sha256"]
                or frame.get("sha256") != h("evidence/" + name)
            ):
                raise ValueError(f"Frame seal mismatch: {key}/{name}")
            frame_paths.add(f"{key}/evidence/{name}")
        actual_frames = {name for name in entries if re.fullmatch(re.escape(key) + r"/evidence/frame_\d+\.npz", name)}
        if len(frames) != count or actual_frames != frame_paths:
            raise ValueError(f"Frame membership mismatch: {key}")
        summary = j("summary.json")
        metadata = j("evidence/metadata.json")
        case = cases_by_name[case_name]["case"]
        checkpoint = models_by_name[model_name]["sha256"]
        if (
            metadata.get("case") != case
            or summary.get("case") != case
            or j("case.json") != case
            or metadata.get("checkpoint_sha256") != checkpoint
            or summary.get("checkpoint_sha256") != checkpoint
            or metadata.get("pilot_only") is not False
            or summary.get("pilot_only") is not False
            or summary.get("recorded_frames") != count
        ):
            raise ValueError(f"Episode identity mismatch: {key}")
        for field, filename in (
            ("source_archive_sha256", "source.zip"),
            ("realization_sha256", "realization.json"),
            ("appearance_sha256", "appearance.json"),
            ("scene_sha256", "scene.usda"),
            ("export_sha256", "export.json"),
        ):
            if metadata.get(field) != h(filename):
                raise ValueError(f"Artifact hash mismatch: {key}/{filename}")
        export = j("export.json")
        assets = export["sha256"]
        if not isinstance(assets, dict) or sorted(export["dependencies"]) != sorted(assets):
            raise ValueError(f"Asset inventory mismatch: {key}")
        for name, expected in assets.items():
            if (
                not isinstance(name, str)
                or any(part in ("", ".", "..") for part in name.split("/"))
                or PurePosixPath(name).is_absolute()
            ):
                raise ValueError(f"Unsafe asset path: {key}/{name}")
            if h("assets/" + name) != expected:
                raise ValueError(f"Asset hash mismatch: {key}/{name}")
        expected_row = {
            "frame_count": count,
            "result": summary.get("result"),
            "completion_sha256": h("evidence/completion.json"),
            "summary_sha256": h("summary.json"),
            "frames_sha256": h("frames.json"),
        }
        if row.get("receipt") != expected_row or summary.get("result") in (None, "running"):
            raise ValueError(f"Coordinator receipt mismatch: {key}")
        if (
            audit.get("passed") is not True
            or audit.get("frame_count") != count
            or audit.get("completion_sha256") != h("evidence/completion.json")
            or audit.get("frames_sha256") != h("frames.json")
            or audit.get("telemetry_sha256") != h("telemetry.json")
        ):
            raise ValueError(f"Artifact audit mismatch: {key}")
        if (
            video.get("completion_sha256") != h("evidence/completion.json")
            or video.get("artifact_audit_sha256") != h("lab_artifact_audit.json")
            or video.get("inference_frames") != count
            or video.get("terminal_frames") != 40
        ):
            raise ValueError(f"Video audit mismatch: {key}")
        for name in ("debug", "research"):
            item = video["videos"][name]
            if (
                item.get("sha256") != h(f"{name}.mp4")
                or item.get("fps") != 20
                or item.get("decoded_frames") != count + 40
            ):
                raise ValueError(f"Video receipt mismatch: {key}/{name}")
        bindings[key] = {
            "completion_sha256": h("evidence/completion.json"),
            "artifact_audit_sha256": h("lab_artifact_audit.json"),
            "video_audit_sha256": h("lab_video_audit.json"),
        }
    return bindings


def archive(host, remote_root, batch, destination):
    if not re.fullmatch(r"[A-Za-z0-9_@.-]+", host) or host.startswith("-"):
        raise ValueError("Invalid SSH host")
    parts = batch.split("/")
    if not batch or any(not p or p in (".", "..") for p in parts) or PurePosixPath(batch).is_absolute():
        raise ValueError("Batch must be a confined relative path")
    if not PurePosixPath(remote_root).is_absolute():
        raise ValueError("Remote task root must be absolute")
    destination = Path(destination).absolute()
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    receipt_path = parent / (destination.name + ".receipt.json")
    if destination.exists() or destination.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("Destination and receipt must not already exist")
    first = remote_inventory(host, remote_root, batch)
    size = sum(e.get("size", 0) for e in first["entries"].values())
    if shutil.disk_usage(parent).free < size + RESERVE:
        raise ValueError("Insufficient space to copy batch and retain 25 GiB reserve")
    destination.mkdir()
    source = host + ":" + shlex.quote(first["batch"] + "/")
    subprocess.run(["rsync", "-r", "--", source, str(destination) + "/"], check=True)
    local = inventory(destination)
    if local != first["entries"]:
        raise ValueError("Local inventory differs from source")
    bindings = validate_batch(destination, local)
    second = remote_inventory(host, remote_root, batch)
    if second != first:
        raise ValueError("Remote batch changed during copy")
    if inventory(destination) != local:
        raise ValueError("Local batch changed during verification")
    if shutil.disk_usage(parent).free < RESERVE:
        raise ValueError("Local free space fell below 25 GiB reserve")
    receipt = {
        "version": 1,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "ssh_host": host,
        "remote_hostname": first["hostname"],
        "remote_root": first["root"],
        "remote_batch": first["batch"],
        "local_destination": str(destination),
        "plan_lock_sha256": local["plan.lock.json"]["sha256"],
        "manifest_sha256": local["manifest.json"]["sha256"],
        "episodes": bindings,
        "inventory": local,
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--remote-root", required=True, help="Absolute remote task root")
    parser.add_argument("--batch", required=True, help="Batch path relative to the task root")
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(archive(args.ssh_host, args.remote_root, args.batch, args.destination))
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Archive failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
