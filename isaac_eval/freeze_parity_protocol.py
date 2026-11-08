"""Freeze the full case matrix after runtime verification, before scoring."""
import argparse
import json
from pathlib import Path

from isaac_eval.parity_cases import all_cases, json_sha256, sha256_file
from isaac_eval.parity_protocol import GRAPH_FILES, runtime_sources, validate, verify_baseline
from isaac_eval.rendering import PROFILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--vehicle", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    metadata = args.pilot / "evidence/metadata.json"
    gates = {"vehicle": args.vehicle, "camera": args.camera,
             "inference": args.pilot / "inference_verification.json"}
    protocol = {
        "schema_version": 1, "case_ids": [case.case_id for case in all_cases()],
        "manifest_sha256": manifest["manifest_sha256"], "render_profile": PROFILE,
        "source_sha256": runtime_sources(), "graph_sha256": {path: sha256_file(path) for path in GRAPH_FILES},
        "baseline": verify_baseline(),
        "gates": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in gates.items()},
        "pilot_metadata": str(metadata), "pilot_metadata_sha256": sha256_file(metadata),
        "metrics": {
            "primary": "Original referee pass, geometric DOO, off-course, progress and time per matched case",
            "physical_contacts": "Unique PhysX car-cone contacts, separate from original geometric DOO",
            "performance": "Per-frame render, policy including raw-head capture, evidence write and physics time",
            "timing_limits": "No training; no claim of speedup; video/oracle export runs offline and is excluded",
            "case_selection": "All 41 predefined cases, including command ablations and prior failure",
        },
    }
    protocol["protocol_sha256"] = json_sha256(protocol)
    validate(protocol, manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(protocol, stream, indent=2)
    print(json.dumps({"path": str(args.out), "protocol_sha256": protocol["protocol_sha256"]}))


if __name__ == "__main__":
    main()
