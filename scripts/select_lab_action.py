"""Preselect an action-recovery checkpoint using fixed open-loop validation only.

The thresholds are predeclared screening choices, not safety bounds. A selected
checkpoint still requires native physical development evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

from scripts.freeze_lab_selection import checkpoint_info
from scripts.lab_training_provenance import MEMBERSHIP, dataset_paths, run_checkpoint, validation_identity


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def source_check(root, run):
    path = root / "source.zip"
    source = run["source"]
    if sha(path) != source["archive_sha256"]:
        raise ValueError("source archive hash mismatch")
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sorted(names) != sorted(source["files_sha256"]):
            raise ValueError("source archive inventory mismatch")
        for name in names:
            if hashlib.sha256(archive.read(name)).hexdigest() != source["files_sha256"][name]:
                raise ValueError("source archive contents mismatch")
    return path


def provenance(root, run):
    args = run["args"]
    manifest = Path(args["manifest"]).resolve()
    verification = run["dataset_verification"]
    receipt = Path(verification["path"]).resolve()
    doc, evidence = read(manifest), read(receipt)
    digest = sha(manifest)
    if (
        doc.get("schema_version") != 1
        or doc.get("complete") is not True
        or not doc.get("membership_sha256")
        or evidence.get("passed") is not True
        or evidence.get("manifest_sha256") != digest
        or evidence.get("membership_sha256") != doc["membership_sha256"]
        or verification.get("status") != "verified"
        or verification.get("sha256") != sha(receipt)
        or verification.get("manifest_sha256") != digest
        or verification.get("membership_sha256") != doc["membership_sha256"]
        or run.get("manifest_sha256") != digest
    ):
        raise ValueError("manifest or verification provenance mismatch")
    return manifest, receipt, source_check(root, run)


def frozen_state(checkpoint, step, config=None):
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise ValueError(f"invalid checkpoint {checkpoint}")
    if step is not None and state.get("step") != step:
        raise ValueError(f"checkpoint step mismatch {checkpoint}")
    if config is not None and state.get("config") != config:
        raise ValueError(f"checkpoint config mismatch {checkpoint}")
    model = state["model"]
    result = {}
    for name, tensor in model.items():
        if name.startswith(("base.vision.", "bev.", "img.")):
            v = tensor.detach().cpu().contiguous()
            result[name] = {
                "dtype": str(v.dtype),
                "shape": list(v.shape),
                "sha256": hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
    if any(not any(k.startswith(prefix) for k in result) for prefix in ("base.vision.", "bev.", "img.")):
        raise ValueError(f"missing frozen tensor group in {checkpoint}")
    if not any(k.startswith("base.policy.") for k in model):
        raise ValueError(f"missing policy in {checkpoint}")
    return result


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {label}")
    return value


def validation(doc, anchors, require_plan_l1=True):
    if doc.get("precision") != "float32" or doc.get("aggregation") != "equal_episode":
        raise ValueError("validation metric definition mismatch")
    expected = {a["path"]: len(a["frames"]) for a in anchors}
    rows = doc["episodes"]
    if (
        len(expected) != len(anchors)
        or not expected
        or any(not n for n in expected.values())
        or any(
            not isinstance(a["path"], str)
            or not isinstance(a["frames"], list)
            or any(type(frame) is not int or frame < 0 for frame in a["frames"])
            or len(set(a["frames"])) != len(a["frames"])
            for a in anchors
        )
    ):
        raise ValueError("invalid validation anchors")
    if len(rows) != len(expected) or {e["path"]: e["count"] for e in rows} != expected:
        raise ValueError("validation episodes do not match anchors")
    if doc.get("count") != sum(expected.values()):
        raise ValueError("validation anchor count mismatch")
    metrics = ("steer", "longitudinal", "plan_l1") if require_plan_l1 else ("steer", "longitudinal")
    for group in [doc["aggregate"], *rows]:
        for key in metrics:
            number(group[key], key)
        nll = group["plan_nll"]
        if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll):
            raise ValueError("invalid plan_nll")
    for key in metrics:
        if not math.isclose(doc["aggregate"][key], sum(r[key] for r in rows) / len(rows), rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError(f"inconsistent aggregate {key}")
    return doc


def select(
    training,
    baseline_training,
    expected_steps,
    *,
    parent_training=None,
    dataset=None,
    dataset_verification=None,
    baseline_dataset=None,
    baseline_dataset_verification=None,
    parent_dataset=None,
    parent_dataset_verification=None,
    init_checkpoint=None,
    baseline_checkpoint=None,
    membership=MEMBERSHIP,
):
    training, baseline_training = Path(training).resolve(), Path(baseline_training).resolve()
    if training == baseline_training or expected_steps < 1:
        raise ValueError("training directories must differ and expected steps must be positive")
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha(path)
        return path

    action_run_path, base_run_path = bind(training / "run.json"), bind(baseline_training / "run.json")
    action, base = read(action_run_path), read(base_run_path)
    identities = {}
    parent = None
    if parent_training is not None:
        parent_training = Path(parent_training).resolve()
        if parent_training in (training, baseline_training):
            raise ValueError("Immediate parent must be a separate action training run")
        parent = read(bind(parent_training / "run.json"))
        for root, run, manifest, receipt in (
            (training, action, dataset, dataset_verification),
            (baseline_training, base, baseline_dataset, baseline_dataset_verification),
            (parent_training, parent, parent_dataset, parent_dataset_verification),
        ):
            manifest, receipt = dataset_paths(run, manifest, receipt)
            identity = validation_identity(run, manifest, receipt, membership)
            identities[root] = identity
            inputs.update(identity["input_sha256"])
            bind(source_check(root, run))
        reference_identity = identities[baseline_training]
        if any(
            identity["fingerprint"] != reference_identity["fingerprint"]
            or identity["membership_sha256"] != reference_identity["membership_sha256"]
            for identity in identities.values()
        ):
            raise ValueError("Verified validation identities, bytes or actual validation anchors differ")
    else:
        if any(
            value is not None
            for value in (
                dataset,
                dataset_verification,
                baseline_dataset,
                baseline_dataset_verification,
                parent_dataset,
                parent_dataset_verification,
                init_checkpoint,
                baseline_checkpoint,
            )
        ):
            raise ValueError("Dataset and checkpoint overrides require --parent-training")
        for root, run in ((training, action), (baseline_training, base)):
            for path in provenance(root, run):
                bind(path)
    if parent is None and (
        action["manifest_sha256"] != base["manifest_sha256"]
        or action["dataset_verification"]["sha256"] != base["dataset_verification"]["sha256"]
        or action["validation_selection"] != base["validation_selection"]
        or action["episodes"] != base["episodes"]
    ):
        raise ValueError("verified manifest or actual validation anchors differ")
    args = action["args"]
    if args.get('controller_mode', 'pp') != 'pp' or action.get('controller_contract', {}).get('mode', 'pp') != 'pp':
        raise ValueError('legacy selector does not support this controller contract')
    if args.get("freeze_vision") is not True or args.get("w_plan") != 0.1 or args.get("steps") != expected_steps:
        raise ValueError("action configuration mismatch")
    if args.get("val_every", 0) < 1:
        raise ValueError("invalid validation interval")
    original = validation(
        read(bind(baseline_training / "baseline_validation.json")), base["validation_selection"], False
    )
    zero = validation(read(bind(training / "baseline_validation.json")), action["validation_selection"])
    if original.get("step") != 0 or zero.get("step") != 0:
        raise ValueError("baseline validation must be step zero")
    init_path = bind(init_checkpoint if init_checkpoint is not None else args["init"])
    if inputs[str(init_path)] != action["init_sha256"]:
        raise ValueError("initial checkpoint hash mismatch")
    base_history = [
        json.loads(line) for line in bind(baseline_training / "history.jsonl").read_text().splitlines() if line.strip()
    ]
    init_info = None
    if parent is None:
        if not any(
            row.get("checkpoint_sha256") == action["init_sha256"] or row.get("selected_sha256") == action["init_sha256"]
            for row in base_history
        ):
            raise ValueError("action initialization is absent from original training history")
    else:
        if parent["args"].get("freeze_vision") is not True:
            raise ValueError("Immediate parent must be an action training run")
        init_info = checkpoint_info(init_path)
        vision_path = bind(baseline_checkpoint if baseline_checkpoint is not None else parent["args"]["init"])
        vision_info = checkpoint_info(vision_path)
        parent_history = [
            json.loads(line)
            for line in bind(parent_training / "history.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for run, info, history in ((action, init_info, parent_history), (parent, vision_info, base_history)):
            matches = [
                row
                for row in history
                if info["checkpoint_sha256"] in (row.get("checkpoint_sha256"), row.get("selected_sha256"))
            ]
            if run["init_sha256"] != info["checkpoint_sha256"] or not matches:
                raise ValueError("Initialization is absent from its immediate parent training history")
            if any(row.get("step") != info["step"] for row in matches):
                raise ValueError("Initialization step differs from its parent training history")
        if init_info["signature"] != vision_info["signature"]:
            raise ValueError("Immediate parent does not retain the complete model signature")
        if frozen_state(init_path, init_info["step"], parent["args"]) != frozen_state(
            vision_path, vision_info["step"], base["args"]
        ):
            raise ValueError("Immediate parent frozen state differs from vision ancestor")
    if base["args"].get("freeze_vision", False) is not False:
        raise ValueError("original baseline must be a full-vision training run")
    if (
        args.get("history_context") != base["args"].get("history_context")
        or parent is not None
        and args.get("history_context") != parent["args"].get("history_context")
    ):
        raise ValueError("validation history context differs")
    reference = action["baseline_state_sha256"]
    if frozen_state(init_path, None) != reference or zero.get("saved_state_sha256") != reference:
        raise ValueError("frozen state receipt differs from initialization")
    history_path = bind(training / "history.jsonl")
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if [r.get("step") for r in rows] != list(range(1, expected_steps + 1)):
        raise ValueError("history must contain every consecutive completed update")
    expected_validations = {1, expected_steps} | set(range(args["val_every"], expected_steps + 1, args["val_every"]))
    actual_validations = {r["step"] for r in rows if "validation" in r}
    if actual_validations != expected_validations:
        raise ValueError("validation checkpoint schedule mismatch")
    base_episode = {
        (identities[baseline_training]["path_to_case"][e["path"]] if parent else e["path"]): e["steer"]
        for e in original["episodes"]
    }
    candidates = []
    if parent is not None:
        if not zero.get("checkpoint") or not zero.get("checkpoint_sha256"):
            raise ValueError("Recovery selection requires a durable step-zero checkpoint receipt")
        rows = [
            {
                "step": 0,
                "checkpoint": zero["checkpoint"],
                "checkpoint_sha256": zero["checkpoint_sha256"],
                "state_sha256": zero["saved_state_sha256"],
                "validation": zero,
            },
            *rows,
        ]
    for row in rows:
        if "validation" not in row:
            continue
        step = row["step"]
        checkpoint = bind(
            run_checkpoint(training, action, row["checkpoint"], f"step_{step:06d}.pt") if parent else row["checkpoint"]
        )
        if checkpoint.parent != training or checkpoint.name != f"step_{step:06d}.pt":
            raise ValueError("checkpoint path mismatch")
        if inputs[str(checkpoint)] != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch at step {step}")
        if row.get("state_sha256") != reference or frozen_state(checkpoint, step, args) != reference:
            raise ValueError(f"frozen state mismatch at step {step}")
        if parent is not None:
            info = checkpoint_info(checkpoint)
            if info["signature"] != init_info["signature"]:
                raise ValueError(f"Complete model signature mismatch at step {step}")
            if step == 0 and info["state_sha256"] != init_info["state_sha256"]:
                raise ValueError("Step-zero model state differs from immediate parent")
        v = validation(row["validation"], action["validation_selection"])
        limits = {
            "steer": 1.25 * original["aggregate"]["steer"],
            "plan_l1": 1.1 * zero["aggregate"]["plan_l1"],
            "longitudinal": 1.25 * zero["aggregate"]["longitudinal"],
        }
        reasons = [f"{k} exceeds limit" for k, limit in limits.items() if v["aggregate"][k] > limit]
        episode_limits = {path: max(2 * value, 0.0005) for path, value in base_episode.items()}
        reasons += [
            f"episode steer exceeds limit: {e['path']}"
            for e in v["episodes"]
            if e["steer"] > episode_limits[identities[training]["path_to_case"][e["path"]] if parent else e["path"]]
        ]
        candidates.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": inputs[str(checkpoint)],
                "validation": v,
                "limits": limits,
                "episode_steer_limits": episode_limits,
                "eligible": not reasons,
                "improves_incumbent": v["aggregate"]["steer"] < zero["aggregate"]["steer"],
                "rejection_reasons": reasons,
            }
        )
    eligible = [
        c
        for c in candidates
        if c["eligible"]
        and (parent is None or c["step"] == 0 or c["validation"]["aggregate"]["steer"] < zero["aggregate"]["steer"])
    ]
    chosen = min(eligible, key=lambda c: (c["validation"]["aggregate"]["steer"], c["step"])) if eligible else None
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError(f"input changed during selection: {path}")
    return {
        "schema_version": 1,
        "status": "selected" if chosen else "rejected",
        "screening": (
            "Predeclared open-loop screening choices, not safety or statistical bounds. "
            "Native physical development evaluation is required."
        ),
        "rules": {
            "aggregate_steer": "at most 1.25 times original baseline",
            "episode_steer": "at most max(2 times original baseline, 0.0005)",
            "weighted_plan_l1": "at most 1.10 times action step 0",
            "longitudinal": "at most 1.25 times action step 0",
            "selection": (
                "lowest eligible aggregate steer; updates must improve step zero; ties retain step zero"
                if parent is not None
                else "lowest aggregate steer; ties favor earlier step"
            ),
        },
        "expected_steps": expected_steps,
        "immediate_parent_training": str(parent_training) if parent is not None else None,
        "immediate_parent_checkpoint": ({"path": str(init_path), **init_info} if init_info else None),
        "validation_fingerprint": identities[training]["fingerprint"] if parent is not None else None,
        "input_sha256": inputs,
        "original_baseline_validation": original,
        "action_step_zero_validation": zero,
        "candidates": candidates,
        "selected": ({k: chosen[k] for k in ("step", "checkpoint", "checkpoint_sha256")} if chosen else None),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--baseline-training", type=Path, required=True)
    parser.add_argument(
        "--parent-training", type=Path, help="Immediate action parent; enables verified recovery selection"
    )
    for name in (
        "dataset",
        "dataset-verification",
        "baseline-dataset",
        "baseline-dataset-verification",
        "parent-dataset",
        "parent-dataset-verification",
        "init-checkpoint",
        "baseline-checkpoint",
    ):
        parser.add_argument(f"--{name}", type=Path, help="Explicit local artifact path for a mirrored recovery run")
    parser.add_argument("--membership", type=Path, default=MEMBERSHIP)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=500)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    result = select(
        args.training,
        args.baseline_training,
        args.expected_steps,
        **{
            name: getattr(args, name)
            for name in (
                "parent_training",
                "dataset",
                "dataset_verification",
                "baseline_dataset",
                "baseline_dataset_verification",
                "parent_dataset",
                "parent_dataset_verification",
                "init_checkpoint",
                "baseline_checkpoint",
                "membership",
            )
        },
    )
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    if result["selected"] is None:
        raise SystemExit("No eligible checkpoint; rejection receipt written.")
    print(json.dumps(result["selected"]))


if __name__ == "__main__":
    main()
