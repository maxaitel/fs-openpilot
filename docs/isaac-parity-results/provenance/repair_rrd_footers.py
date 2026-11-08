"""Preserved one-off repair for four flushed recordings missing their closing footer."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

CASES = (
    "retention-trackdrive_030061-light-recorded",
    "retention-trackdrive_030061-heavy-recorded",
    "ablation-skidpad-72000-heavy-swapped",
    "prior-failure-skidpad-71003-heavy-recorded",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


run = Path(sys.argv[1])
source = digest(__file__)
results = []
for case_id in CASES:
    folder = run / case_id
    recording, backup = folder / "debug.rrd", folder / "debug.backup.rrd"
    receipt = folder / "rrd_footer_repair.json"
    if receipt.exists():
        row = json.loads(receipt.read_text())
        assert row["repair_source_sha256"] == source
        assert row["before_sha256"] == digest(backup) and row["after_sha256"] == digest(recording)
        results.append(row)
        continue
    audit_path = folder / "video_audit.before_rrd_close.json"
    if not audit_path.exists():
        audit_path = folder / "video_audit.json"
    audit = json.loads(audit_path.read_text())
    original = backup if backup.exists() else recording
    before = digest(original)
    assert before == audit["rrd_sha256"], case_id
    row = {"case_id": case_id, "repair_source_sha256": source, "before_sha256": before,
           "original_audit_sha256": digest(audit_path), "backup_file": backup.name, "commands": [],
           "comparison_ignores": ["log_time", "row_order"]}
    commands = [
        ["verify", "--check-footers=false", str(original)],
        *([] if backup.exists() else [["migrate", str(recording)]]),
        ["verify", "--check-footers=true", str(recording)],
        ["compare", "--unordered", str(backup), str(recording)],
    ]
    for arguments in commands:
        result = subprocess.run([sys.executable, "-m", "rerun", "rrd", *arguments],
                                text=True, capture_output=True, check=True)
        row["commands"].append({"arguments": arguments, "returncode": result.returncode,
                                "stdout": result.stdout, "stderr": result.stderr})
    assert digest(backup) == before
    row["after_sha256"] = digest(recording)
    row["footer_verified"] = True
    row["data_comparison_passed"] = True
    receipt.write_text(json.dumps(row, indent=2) + "\n")
    results.append(row)
    print("REPAIRED_FOOTER", case_id, flush=True)
(run / "rrd_footer_repairs.json").write_text(json.dumps({"complete": True,
    "repair_source_sha256": source, "cases": results}, indent=2) + "\n")
