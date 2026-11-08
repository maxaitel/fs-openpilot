"""Run every frozen case sequentially and retain technical failures without silently retrying."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    records = []
    for case in manifest["cases"]:
        case_id = case["case_id"]
        case_out = args.out / case_id
        command = ["/isaac-sim/python.sh", "isaac_eval/evaluate_parity.py", "--case", case_id,
                   "--manifest", str(args.manifest), "--protocol", str(args.protocol), "--out", str(case_out)]
        print("START", case_id, flush=True)
        with (args.out / f"{case_id}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        summary_path = case_out / "summary.json"
        receipt_path = case_out / "evidence/completion.json"
        record = {"case_id": case_id, "returncode": result.returncode,
                  "technical_success": result.returncode == 0 and summary_path.is_file() and receipt_path.is_file()}
        if record["technical_success"]:
            summary = json.loads(summary_path.read_text())
            record.update({key: summary[key] for key in ("pass", "result", "doo", "oc", "recorded_frames")})
        records.append(record)
        status = {"started_unix": started, "elapsed_wall_seconds": time.time() - started,
                  "expected_cases": len(manifest["cases"]), "cases": records,
                  "complete": len(records) == len(manifest["cases"]) and all(r["technical_success"] for r in records)}
        (args.out / "status.json").write_text(json.dumps(status, indent=2))
        print("END", json.dumps(record), flush=True)
        if not record["technical_success"]:
            raise RuntimeError(f"Technical failure retained at {case_out}; suite stopped for inspection")


if __name__ == "__main__":
    main()
