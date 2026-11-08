"""Bind the scored suite to its executable inputs and successful integration gates."""
import json
from pathlib import Path

from isaac_eval.parity_cases import all_cases, json_sha256, sha256_file

GRAPH_FILES = ("artifacts/models/prep/vision_fp32.onnx", "artifacts/models/prep/policy_fp32.onnx")


def runtime_sources():
    paths = sorted(Path("src/fsop").rglob("*.py")) + sorted(Path("isaac_eval").glob("*.py"))
    return {str(path): sha256_file(path) for path in paths}


def verify_baseline():
    baseline = json.loads(Path("isaac_eval/parity_reference/baseline_code.json").read_text())
    for path, expected in baseline["sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"Original model/environment code changed: {path}")
    return baseline


def validate(protocol, manifest):
    from isaac_eval.rendering import PROFILE

    if protocol["manifest_sha256"] != manifest["manifest_sha256"] or protocol["render_profile"] != PROFILE:
        raise ValueError("Protocol manifest/render configuration changed")
    if protocol["case_ids"] != [case.case_id for case in all_cases()]:
        raise ValueError("Protocol must contain all 41 frozen cases")
    if protocol["source_sha256"] != runtime_sources():
        raise ValueError("Runtime source inventory changed after verification")
    verify_baseline()
    if set(protocol["graph_sha256"]) != set(GRAPH_FILES):
        raise ValueError("Both executable ONNX graphs must be frozen")
    for path, expected in protocol["graph_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"Executable graph changed: {path}")
    if set(protocol["gates"]) != {"vehicle", "camera", "inference"}:
        raise ValueError("Vehicle, camera and inference gates are required")
    for name, gate in protocol["gates"].items():
        if sha256_file(gate["path"]) != gate["sha256"]:
            raise ValueError(f"Verification receipt changed: {name}")
        receipt = json.loads(Path(gate["path"]).read_text())
        if not receipt.get("passed", receipt.get("pass", False)):
            raise ValueError(f"Verification gate failed: {name}")
        for path, expected in receipt.get("source_sha256", {}).items():
            if sha256_file(path) != expected:
                raise ValueError(f"Verified source changed: {path}")
    metadata_path = Path(protocol["pilot_metadata"])
    if sha256_file(metadata_path) != protocol["pilot_metadata_sha256"]:
        raise ValueError("Verified pilot metadata changed")
    metadata = json.loads(metadata_path.read_text())
    if metadata["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("Pilot did not exercise the frozen runtime")
    if metadata["graph_sha256"] != protocol["graph_sha256"]:
        raise ValueError("Pilot executable graphs differ")
    inference = json.loads(Path(protocol["gates"]["inference"]["path"]).read_text())
    if inference["evidence_sha256"]["metadata.json"] != protocol["pilot_metadata_sha256"]:
        raise ValueError("Inference verification belongs to a different pilot")
    if protocol["protocol_sha256"] != json_sha256({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("Protocol digest changed")
