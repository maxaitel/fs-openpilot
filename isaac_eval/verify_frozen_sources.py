"""Verify a recorded protocol against a Git revision and its local asset manifest."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def verify(protocol_path, revision):
    root = Path(__file__).resolve().parent.parent
    protocol = json.loads(protocol_path.read_text())
    records = []
    for name, expected in protocol["source_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Invalid source path: {name}")
        if name == "isaac_eval/assets/manifest.json":
            data = path.read_bytes()
            origin = "local asset manifest"
        else:
            data = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
            origin = revision
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source mismatch: {name}")
        records.append({"path": name, "sha256": actual, "origin": origin})
    return {
        "pass": True,
        "revision": revision,
        "files": records,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.protocol, args.revision)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(report['files'])} frozen inputs against {args.revision}")
