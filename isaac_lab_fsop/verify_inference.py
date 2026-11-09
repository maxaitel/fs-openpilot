"""Replay a complete 120-frame native Lab pilot from its saved policy inputs.

This does not launch Isaac or re-run physics. FP32 tensor comparisons use
absolute and relative tolerances of 1e-5, with exact integer/class comparisons.
The same checkpoint, ONNX graphs, model code, controller and CPU environment
realization must match the recorded episode. Black RGB is an open-loop positive
control; a meaningful response is required in the plan, not a driving score.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from fsop.eval.closed_loop import ModelPolicy
from isaac_eval.evidence import ImageHeadCapture, read_evidence, verified_summary
from isaac_eval.parity_cases import realized_environment, sha256_file
from isaac_eval.parity_protocol import GRAPH_FILES
from isaac_eval.verify_inference import observation, state_digest
from isaac_lab_fsop.cases import Case, environment

RTOL = 1e-5
ATOL = 1e-5
PILOT_FRAMES = 120


def compare(expected, actual, stats, path):
    """Compare numeric trees, reporting maximum absolute error and mismatches."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            stats.setdefault(path + ".keys", {"max_abs": None, "mismatches": 0})["mismatches"] += 1
        for key in sorted(expected.keys() & actual.keys()):
            compare(expected[key], actual[key], stats, f"{path}.{key}")
        return
    if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        if len(expected) != len(actual):
            stats.setdefault(path + ".length", {"max_abs": None, "mismatches": 0})["mismatches"] += 1
        for i, (a, b) in enumerate(zip(expected, actual, strict=False)):
            compare(a, b, stats, f"{path}[{i}]")
        return
    row = stats.setdefault(path, {"max_abs": 0.0, "mismatches": 0})
    if expected is None or actual is None:
        row["mismatches"] += int(expected is not None or actual is not None)
        return
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape or a.dtype.kind not in "biuf" or b.dtype.kind not in "biuf":
        row["mismatches"] += 1
        return
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        row["mismatches"] += 1
        return
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    row["max_abs"] = max(row["max_abs"], float(diff.max(initial=0)))
    exact = a.dtype.kind in "biu" or b.dtype.kind in "biu"
    row["mismatches"] += int(not (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=RTOL, atol=ATOL)))


@contextmanager
def capture_policy_raw(model):
    """Copy outputs of the existing policy_raw call without altering its return."""
    base = model.base if hasattr(model, "base") else model
    original = base.policy_raw
    captured = {}

    def wrapper(*args, **kwargs):
        output = original(*args, **kwargs)
        captured["output"] = [value.detach().cpu().numpy().copy() for value in output]
        return output

    base.policy_raw = wrapper
    try:
        yield captured
    finally:
        base.policy_raw = original


def _mismatches(stats):
    return sum(row["mismatches"] for row in stats.values())


def verify(episode, checkpoint, output, max_frames=PILOT_FRAMES):
    episode, checkpoint, output = Path(episode), Path(checkpoint), Path(output)
    evidence = episode / "evidence"
    metadata_path = evidence / "metadata.json"
    manifest_path = evidence / "manifest.jsonl"
    metadata = json.loads(metadata_path.read_text())
    summary = verified_summary(evidence)
    if summary is None:
        raise ValueError("Evidence must have a completion receipt")
    receipt = json.loads((evidence / "completion.json").read_text())
    if (evidence / receipt["summary_path"]).resolve() != (episode / "summary.json").resolve():
        raise ValueError("Completion receipt does not bind this episode summary")
    entries = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    if max_frames != PILOT_FRAMES or len(entries) != PILOT_FRAMES or receipt["frame_count"] != PILOT_FRAMES:
        raise ValueError("Verification requires the entire 120-frame pilot and --max-frames 120")
    if not metadata.get("pilot_only") or not summary.get("pilot_only"):
        raise ValueError("This verifier requires a bounded pilot episode")
    if metadata.get("backend") != "Isaac Lab PhysX" or metadata.get("fps") != 20:
        raise ValueError("Evidence is not a 20 Hz native Lab episode")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != metadata["checkpoint_sha256"] or checkpoint_sha != summary["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the recorded checkpoint")
    if checkpoint.stat().st_size != metadata["checkpoint_size_bytes"]:
        raise ValueError("Checkpoint size differs from the recorded checkpoint")
    graph_hashes = metadata["graph_sha256"]
    if set(graph_hashes) != set(GRAPH_FILES):
        raise ValueError("Recorded graph inventory is incomplete")
    for name, expected in graph_hashes.items():
        if sha256_file(name) != expected:
            raise ValueError(f"Executable graph differs from recorded graph: {name}")
    # The actual policy and its preprocessing/controller must be the recorded code.
    source_hashes = metadata["source_sha256"]
    required_sources = {
        name: value
        for name, value in source_hashes.items()
        if name.startswith("src/fsop/")
        or name in ("isaac_lab_fsop/cases.py", "isaac_eval/evidence.py", "isaac_eval/parity_cases.py")
    }
    required_files = {
        "src/fsop/eval/closed_loop.py",
        "src/fsop/eval/controller.py",
        "isaac_lab_fsop/cases.py",
        "isaac_eval/evidence.py",
        "isaac_eval/parity_cases.py",
    }
    if not required_files <= required_sources.keys():
        raise ValueError("Recorded runtime source inventory is incomplete")
    for name, expected in required_sources.items():
        if sha256_file(name) != expected:
            raise ValueError(f"Inference or environment source differs from episode: {name}")
    case = Case(**metadata["case"])
    if summary["case"] != metadata["case"]:
        raise ValueError("Case identity differs between evidence and summary")
    env = environment(case)
    realization_path = episode / "realization.json"
    if sha256_file(realization_path) != metadata["realization_sha256"]:
        raise ValueError("Realization file hash differs from metadata")
    if realized_environment(env) != json.loads(realization_path.read_text()):
        raise ValueError("Reconstructed environment differs from recorded realization")
    if not np.array_equal(metadata["nominal_detection_calibration"], env.T_ego_cam()):
        raise ValueError("Nominal calibration differs from recorded case")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    captured = ModelPolicy.from_checkpoint(checkpoint, controller="pp")
    captured.model.requires_grad_(False)
    plain = ModelPolicy(captured.model, controller="pp", device=captured.device)
    black = ModelPolicy(captured.model, controller="pp", device=captured.device)
    before = state_digest(captured.model)
    if before != metadata["state_sha256"] or before != summary["state_sha256"]:
        raise ValueError("Loaded model state differs from recorded state")
    if captured.model.training or any(p.requires_grad for p in captured.model.parameters()):
        raise ValueError("Model must be frozen and in evaluation mode")
    evidence_names = ["metadata.json", "manifest.jsonl", "completion.json"] + [entry["file"] for entry in entries]
    evidence_before = {name: sha256_file(evidence / name) for name in evidence_names}
    summary_before = sha256_file(episode / "summary.json")
    stats_saved, stats_hook, stats_black = {}, {}, {}
    count = 0
    with torch.inference_mode():
        for _, record, _ in read_evidence(evidence):
            if record["frame"] != count or record["state"]["t"] != count / 20:
                raise ValueError("Evidence frame numbering or timestamp is inconsistent")
            if not np.array_equal(record["nominal_detection_calibration"], env.T_ego_cam()):
                raise ValueError("Recorded nominal calibration changed")
            obs = observation(record, env)
            with capture_policy_raw(captured.model) as raw, ImageHeadCapture(captured.model) as head:
                steer, accel, info = captured(obs, env)
            if head.output is None or "output" not in raw:
                raise ValueError("Diagnostic capture did not observe the policy and image heads")
            got = {"action": [steer, accel], "prediction": info, "image_head": head.output, "policy_raw": raw["output"]}
            expected = {key: record[key] for key in got}
            compare(expected, got, stats_saved, "saved")
            psteer, paccel, pinfo = plain(obs, env)
            compare(
                {"action": [steer, accel], "prediction": info},
                {"action": [psteer, paccel], "prediction": pinfo},
                stats_hook,
                "hook",
            )
            black_obs = observation(record, env, black=True)
            bsteer, baccel, binfo = black(black_obs, env)
            compare(
                {"action": [steer, accel], "prediction": info},
                {"action": [bsteer, baccel], "prediction": binfo},
                stats_black,
                "black",
            )
            count += 1
    after = state_digest(captured.model)
    unchanged = (
        evidence_before == {name: sha256_file(evidence / name) for name in evidence_names}
        and summary_before == sha256_file(episode / "summary.json")
        and checkpoint_sha == sha256_file(checkpoint)
        and before == after
        and all(sha256_file(name) == digest for name, digest in graph_hashes.items())
        and not captured.model.training
        and not any(p.requires_grad for p in captured.model.parameters())
    )
    plan_control = stats_black.get("black.prediction.plan", {}).get("mismatches", 0) > 0
    passed = (
        count == PILOT_FRAMES
        and unchanged
        and not _mismatches(stats_saved)
        and not _mismatches(stats_hook)
        and plan_control
    )
    report = {
        "passed": passed,
        "frames": count,
        "expected_frames": PILOT_FRAMES,
        "case": metadata["case"],
        "checkpoint_sha256": checkpoint_sha,
        "state_sha256": before,
        "state_sha256_after": after,
        "graph_sha256": graph_hashes,
        "source_sha256": required_sources,
        "verifier_sha256": sha256_file(__file__),
        "realization_sha256": sha256_file(realization_path),
        "evidence_sha256": evidence_before,
        "summary_sha256": summary_before,
        "frozen_unchanged": unchanged,
        "rtol": RTOL,
        "atol": ATOL,
        "saved_vs_replay": stats_saved,
        "hook_enabled_vs_disabled": stats_hook,
        "black_image_diagnostic": {
            "description": "Open-loop black RGB input, not a driving score",
            "plan_changed_above_tolerance": plan_control,
            "differences": stats_black,
        },
        "verification_limits": [
            "Only the 120 saved pilot inputs are replayed; no physical simulation or scoring is rerun.",
            "Black RGB is an input-sensitivity control, not evidence of robustness.",
        ],
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=PILOT_FRAMES)
    args = parser.parse_args()
    report = verify(args.episode, args.checkpoint, args.out, args.max_frames)
    print(json.dumps({"passed": report["passed"], "frames": report["frames"], "output": str(args.out)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
