"""Download audited videos and receipts while raw frames remain on the Spark."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="SSH host:path to parity_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    host, remote = args.remote.split(":", 1)
    args.out.mkdir(parents=True, exist_ok=True)
    collected = set()
    while True:
        fetch = subprocess.run(
            ["ssh", host, "cat", f"{remote}/postprocess_status.json"], capture_output=True, text=True
        )
        if fetch.returncode:
            time.sleep(30)
            continue
        status = json.loads(fetch.stdout)
        for row in status["cases"]:
            name = row["case_id"]
            if name in collected:
                continue
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--exclude=assets",
                    "--exclude=scene.usda",
                    "--exclude=frame_*.npz",
                    "--exclude=*.rrd",
                    "--exclude=*-source.mp4",
                    "--exclude=*.log",
                    f"{host}:{remote}/{name}",
                    str(args.out) + "/",
                ],
                check=True,
            )
            collected.add(name)
            print("COLLECTED", name, flush=True)
        subprocess.run(
            [
                "rsync",
                "-a",
                f"{host}:{remote}/status.json",
                f"{host}:{remote}/postprocess_status.json",
                str(args.out) + "/",
            ],
            check=True,
        )
        if status["complete"]:
            print("ALL_CASES_COLLECTED", len(collected), flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
