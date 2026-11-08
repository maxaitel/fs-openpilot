"""Replay a parity evidence episode through the frozen model and test capture transparency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fsop.eval.closed_loop import ModelPolicy
from fsop.sim.env import Obs
from fsop.sim.referee import Event
from fsop.sim.vehicle import VehicleState
from isaac_eval.evidence import ImageHeadCapture, read_evidence
from isaac_eval.parity_cases import (
    build_environment,
    case_manifest,
    get_case,
    realized_environment,
    reference_provenance,
    sha256_file,
    verify_checkpoint,
)

# Same-runtime replay allows small floating-point variation, but not changed detections or classes.
RTOL = 1e-4
ATOL = 1e-4


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def observation(record, env, black=False):
    state = VehicleState(**record["state"])
    env.state = state.copy()
    env.frame = record["frame"]
    env.s = record["progress"]["s"]
    env.hint = record["progress"]["hint"]
    env.lat = record["progress"]["lat"]
    images = record["images"]
    if black:
        images = {name: np.zeros_like(image) for name, image in images.items()}
    return Obs(frame=env.frame, t=state.t, state=state, s=env.s, lat=env.lat, images=images,
               T_world_ego=env.T_world_ego(), desire=np.asarray(record["commands"]["applied"]),
               events=[Event(**event) for event in record["events"]], noise=0.0)


def compare(expected, actual, stats, path=""):
    """Accumulate numeric maxima and structure mismatches for a named output tree."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            stats.setdefault(path, {"max_abs": None, "mismatches": 0})["mismatches"] += 1
        for key in expected.keys() & actual.keys():
            compare(expected[key], actual[key], stats, f"{path}.{key}" if path else key)
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


def result(action, prediction, image_head):
    return {"action": action, "prediction": prediction, "image_head": image_head}


def replay(records, env, policies):
    captured, plain, black = policies
    saved_stats, hook_stats, visual_stats = {}, {}, {}
    count = 0
    with torch.inference_mode():
        for _metadata, record, _entry in records:
            if record["frame"] != count:
                raise ValueError("Evidence record frame is not sequential")
            if not np.array_equal(record["nominal_detection_calibration"], env.T_ego_cam()):
                raise ValueError("Recorded nominal calibration differs from frozen case")
            obs = observation(record, env)
            with ImageHeadCapture(captured.model) as hook:
                steer, accel, info = captured(obs, env)
            if hook.output is None:
                raise ValueError("The original image head was not called")
            got = result([steer, accel], info, hook.output)
            compare(result(record["action"], record["prediction"], record["image_head"]), got, saved_stats)
            psteer, paccel, pinfo = plain(obs, env)
            # The hook is removed for this call. Raw head values are checked against saved evidence above.
            compare({"action": [steer, accel], "prediction": info},
                    {"action": [psteer, paccel], "prediction": pinfo}, hook_stats)
            black_obs = observation(record, env, black=True)
            bsteer, baccel, binfo = black(black_obs, env)
            compare({"action": [steer, accel], "prediction": info},
                    {"action": [bsteer, baccel], "prediction": binfo}, visual_stats)
            count += 1
    if not count:
        raise ValueError("Evidence contains no frames")
    return count, saved_stats, hook_stats, visual_stats


def verify(checkpoint, evidence, output):
    checkpoint, evidence, output = Path(checkpoint), Path(evidence), Path(output)
    checkpoint_sha = verify_checkpoint(checkpoint)
    metadata_path = evidence / "metadata.json"
    manifest_path = evidence / "manifest.jsonl"
    metadata = json.loads(metadata_path.read_text())
    case = get_case(metadata["case"]["case_id"])
    if metadata["case"] != case_manifest(case):
        raise ValueError("Evidence case differs from frozen case manifest")
    root = Path(__file__).resolve().parents[1]
    original_sources = {row["source"]: row["sha256"] for row in reference_provenance()["source_files"]}
    policy_sources = {name: sha256_file(root / name) for name in
                      ("src/fsop/eval/closed_loop.py", "src/fsop/eval/controller.py")}
    if any(digest != original_sources[name] for name, digest in policy_sources.items()):
        raise ValueError("Original policy or controller source changed")
    env = build_environment(case)
    realization = realized_environment(env)
    if realization != json.loads((evidence.parent / "realization.json").read_text()):
        raise ValueError("Evidence realization differs from reconstructed frozen case")
    if metadata["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Evidence checkpoint differs from frozen checkpoint")
    if not np.array_equal(metadata["nominal_detection_calibration"], env.T_ego_cam()):
        raise ValueError("Evidence nominal calibration differs from frozen case")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    first = ModelPolicy.from_checkpoint(checkpoint, controller="pp")
    first.model.requires_grad_(False)
    policies = [first, ModelPolicy(first.model, controller="pp", device=first.device),
                ModelPolicy(first.model, controller="pp", device=first.device)]
    before = state_digest(first.model)
    if before != metadata["state_sha256"]:
        raise ValueError("Loaded model state differs from recorded state")
    if first.model.training or any(p.requires_grad for p in first.model.parameters()):
        raise ValueError("Model must be frozen and in evaluation mode")
    hashes_before = {path.name: sha256_file(path) for path in (metadata_path, manifest_path)}
    entries = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    hashes_before.update({entry["file"]: sha256_file(evidence / entry["file"]) for entry in entries})
    count, saved, hook, visual = replay(read_evidence(evidence), env, policies)
    hashes_after = {name: sha256_file(evidence / name) for name in hashes_before}
    unchanged = (hashes_before == hashes_after and verify_checkpoint(checkpoint) == checkpoint_sha
                 and state_digest(first.model) == before and not first.model.training
                 and not any(p.requires_grad for p in first.model.parameters()))
    if not unchanged:
        raise ValueError("Evidence, checkpoint, or frozen model changed during replay")
    passed = all(row["mismatches"] == 0 for stats in (saved, hook) for row in stats.values())
    report = {"passed": passed, "frames": count, "case_id": case.case_id,
              "checkpoint_sha256": checkpoint_sha, "state_sha256": before, "policy_source_sha256": policy_sources,
              "evidence_sha256": hashes_before, "realization_sha256": sha256_file(evidence.parent / "realization.json"),
              "frozen_unchanged": unchanged, "rtol": RTOL, "atol": ATOL,
              "saved_vs_replay": saved, "hook_enabled_vs_disabled": hook,
              "black_image_diagnostic": {"description": "Open-loop black RGB replay; not a driving score",
                                         "differences": visual}}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.checkpoint, args.evidence, args.output)
    print(json.dumps({"passed": report["passed"], "frames": report["frames"], "output": str(args.output)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
