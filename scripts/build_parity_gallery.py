"""Create a local video and evidence index for the frozen comparison."""

import argparse
import json
import os
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("parity_gallery.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    report = json.loads((args.report / "parity.json").read_text())
    if not report["complete"] and not args.partial:
        raise ValueError("Final gallery requires a complete report")
    rows = []
    for row in report["cases"]:
        if row["status"] != "complete":
            continue
        row = dict(row)
        folder = args.run / row["case_id"]
        for key, name in {
            "debug": "debug.mp4",
            "research": "research.mp4",
            "rrd": "debug.rrd",
            "summary": "summary.json",
            "manifest": "evidence/manifest.jsonl",
            "audit": "artifact_audit.json",
        }.items():
            row[key + "_url"] = os.path.relpath(folder / name, args.report) if (folder / name).is_file() else None
        rows.append(row)
    custom = {}
    for case_id, name in {
        "skidpad-72000-light-recorded": "72000-light",
        "prior-failure-skidpad-71003-heavy-recorded": "71003-heavy",
    }.items():
        folder = args.run.parent / "custom_video_reference" / name
        if (folder / "video.json").is_file():
            original = json.loads((folder / "video.json").read_text())
            custom[case_id] = {
                "url": os.path.relpath(folder / "skidpad-raw.mp4", args.report),
                "result": original["result"],
                "doo": original["doo"],
            }
    data = {
        "complete": report["complete"],
        "cases": rows,
        "groups": report["groups"],
        "custom": custom,
        "checkpoint": report["provenance"]["checkpoint_sha256"],
        "remoteEvidence": report.get("remote_evidence_root"),
    }
    page = TEMPLATE_PATH.read_text().replace("__DATA__", json.dumps(data).replace("<", "\\u003c"))
    (args.report / "index.html").write_text(page)
    print(args.report / "index.html")


if __name__ == "__main__":
    main()
