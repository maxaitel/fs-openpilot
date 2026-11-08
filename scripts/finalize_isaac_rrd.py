"""Verify closed Rerun recordings and preserve the original pre-close audit receipts."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finalize(run, manifest):
    run = Path(run)
    expected = [row["case_id"] for row in json.loads(Path(manifest).read_text())["cases"]]
    status = json.loads((run / "status.json").read_text())
    exported = json.loads((run / "postprocess_status.json").read_text())
    if (
        not status["complete"] or not exported["complete"]
        or status["expected_cases"] != len(expected)
        or [row["case_id"] for row in status["cases"]] != expected
        or sorted(row["case_id"] for row in exported["cases"]) != sorted(expected)
        or len(set(expected)) != len(expected)
        or not all(row["technical_success"] for row in status["cases"])
    ):
        raise ValueError("All expected evaluation and export cases must finish first")
    if not Path("/proc").is_dir():
        raise RuntimeError("Run finalization in the Linux export container")
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            active = path.read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if b"scripts.process_isaac_parity" in active:
            raise RuntimeError("Export workers must exit and close their recordings first")
    source = digest(__file__)
    rows = []
    for case_id in expected:
        folder = run / case_id
        audit_path = folder / "video_audit.json"
        before_path = folder / "video_audit.before_rrd_close.json"
        current = json.loads(audit_path.read_text())
        original_bytes = before_path.read_bytes() if before_path.exists() else audit_path.read_bytes()
        original = json.loads(original_bytes)
        old_digest = hashlib.sha256(original_bytes).hexdigest()
        if before_path.exists():
            unmodified = {key: value for key, value in current.items()
                          if key not in {"rrd_sha256", "rrd_finalization"}}
            if unmodified != {key: value for key, value in original.items() if key != "rrd_sha256"}:
                raise ValueError(f"Unexpected audit mutation: {case_id}")
            prior = current.get("rrd_finalization")
            if prior and (prior["original_audit_sha256"] != old_digest or prior["finalizer_sha256"] != source):
                raise ValueError(f"Different finalization history: {case_id}")
        if (original["case_id"] != case_id
                or original["completion_sha256"] != digest(folder / "evidence/completion.json")
                or original["artifact_audit_sha256"] != digest(folder / "artifact_audit.json")
                or not json.loads((folder / "artifact_audit.json").read_text())["passed"]):
            raise ValueError(f"Unverified evidence binding: {case_id}")
        for name in ("debug", "research"):
            for suffix, key in ((".mp4", "sha256"), ("-source.mp4", "source_sha256")):
                if digest(folder / (name + suffix)) != original["videos"][name][key]:
                    raise ValueError(f"Video changed: {case_id}/{name}{suffix}")
        recording = folder / "debug.rrd"
        verified = subprocess.run(
            [sys.executable, "-m", "rerun", "rrd", "verify", "--check-footers=true", str(recording)],
            text=True, capture_output=True, check=True,
        )
        verification = folder / "rrd_verification.txt"
        verification.write_text(verified.stdout + verified.stderr)
        closed_digest = digest(recording)
        if not before_path.exists():
            with before_path.open("xb") as stream:
                stream.write(original_bytes)
        updated = dict(original)
        updated["rrd_sha256"] = closed_digest
        updated["rrd_finalization"] = {
            "original_audit_file": before_path.name,
            "original_audit_sha256": old_digest,
            "finalizer_sha256": source,
            "footer_verified": True,
            "verification_sha256": digest(verification),
        }
        temporary = audit_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(updated, indent=2) + "\n")
        temporary.replace(audit_path)
        rows.append({"case_id": case_id, "original_audit_sha256": old_digest,
                     "original_rrd_sha256": original["rrd_sha256"], "closed_rrd_sha256": closed_digest,
                     "video_audit_sha256": digest(audit_path), "footer_verified": True})
        print("FINALIZED_RRD", case_id, flush=True)
    result = {"complete": True, "case_count": len(rows), "finalizer_sha256": source,
              "postprocess_status_sha256": digest(run / "postprocess_status.json"), "cases": rows}
    (run / "rrd_finalization.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.run, args.manifest)


if __name__ == "__main__":
    main()
