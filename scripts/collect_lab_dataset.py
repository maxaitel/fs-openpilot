"""Collect the frozen native training/development membership, one Lab process at a time."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--membership", type=Path, default=Path("docs/isaac-lab-training/membership.json"))
    p.add_argument("--out", type=Path, default=Path("data/native/lab_v1"))
    p.add_argument("--reserve-gib", type=float, default=25)
    p.add_argument("--timeout", type=float, default=900)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        membership = json.loads(a.membership.read_text())
        membership_sha = digest(a.membership)
        rows = [c for c in membership["cases"] if c["split"] in ("train", "val")]
        logs = a.out / "logs"
        logs.mkdir(exist_ok=True)
        receipts = []
        for case in rows:
            if digest(a.membership) != membership_sha:
                raise RuntimeError("Membership changed during collection")
            destination = a.out / case["id"]
            meta_path = destination / "meta.json"
            started = time.monotonic()
            resumed = meta_path.exists()
            if not resumed:
                if shutil.disk_usage(a.out).free < a.reserve_gib * 2**30:
                    raise RuntimeError("Free disk is below the declared reserve")
                case_file = a.membership.parent / "cases" / (case["id"] + ".json")
                if json.loads(case_file.read_text()) != case:
                    raise ValueError("Case file differs from membership")
                command = [
                    sys.executable,
                    "-m",
                    "isaac_lab_fsop.collect",
                    "--case",
                    str(case_file),
                    "--membership",
                    str(a.membership),
                    "--out",
                    str(destination),
                ]
                with (logs / (case["id"] + ".log")).open("x") as log:
                    result = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=a.timeout,
                        start_new_session=True,
                        check=False,
                    )
                if result.returncode or not meta_path.exists():
                    raise RuntimeError(f"Native collection failed: {case['id']}; inspect {log.name}")
            from fsop.data.native_replay import NativeEpisode

            ep = NativeEpisode(destination)
            if ep.meta["case"] != case or ep.meta["membership_sha256"] != membership_sha:
                raise ValueError("Existing episode has a different collection identity")
            if digest(destination / "source.zip") != ep.meta["source_archive_sha256"]:
                raise ValueError("Producer source archive differs")
            receipts.append(
                {
                    "path": case["id"],
                    "meta_sha256": digest(meta_path),
                    "accepted": len(ep.anchors),
                    "rejections": ep.rejections,
                    "resumed": resumed,
                    "wall_seconds": ep.meta["wall_seconds"],
                    "coordinator_seconds": time.monotonic() - started,
                    "summary": ep.meta["summary"],
                }
            )
            manifest = {
                "schema_version": 1,
                "membership_sha256": membership_sha,
                "episodes": receipts,
                "expected_episodes": len(rows),
                "complete": len(receipts) == len(rows),
            }
            temporary = a.out / "manifest.json.tmp"
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(temporary, a.out / "manifest.json")
            print(
                json.dumps(
                    {
                        "completed": len(receipts),
                        "expected": len(rows),
                        "id": case["id"],
                        "anchors": len(ep.anchors),
                        "result": ep.meta["summary"]["result"],
                        "physical_hits": ep.meta["summary"]["physical_cone_hits"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
