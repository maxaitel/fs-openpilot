
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from isaac_lab_fsop.cases import Case, environment
from scripts import freeze_lab_selection as freezer
from scripts import select_lab_action as selector
from scripts.lab_training_provenance import MEMBERSHIP, digest, validation_identity
from scripts.verify_lab_dataset import verify


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def make_dataset(root, membership):
    root.mkdir()
    rows = []
    for case_doc in json.loads(membership.read_text())["cases"]:
        case = Case(**case_doc)
        env = environment(case)
        path = root / case.id
        path.mkdir()
        n = 205
        ticks = np.arange(0, n, 4)
        states = np.zeros((n, 9), np.float64)
        states[:, 8] = np.arange(n) * 0.05
        arrays = {
            "ticks": ticks,
            "states": states,
            "controls": np.zeros((n - 1, 2), np.float32),
            "teacher": np.ones(n - 1, bool),
            "contacts": np.zeros(n, np.int64),
            "desires": np.zeros((len(ticks), 25, 8), np.float32),
            "cones": np.empty((0, 3), np.float32),
        }
        for camera in ("road", "wide"):
            arrays[camera] = np.zeros((len(ticks), 6, 128, 256), np.uint8)
        for name, value in arrays.items():
            np.save(path / f"{name}.npy", value)
        with ZipFile(path / "source.zip", "w") as archive:
            archive.writestr("source.py", b"source")
        write(path / "track.json", env.track.to_dict())
        meta = {
            "schema_version": 1,
            "backend": "IsaacLab",
            "id": case.id,
            "case": case_doc,
            "membership_sha256": digest(membership),
            "split": case.split,
            "discipline": case.discipline,
            "seed": case.seed,
            "track_id": env.track.id,
            "scene": env.scene.to_dict(),
            "vehicle": asdict(env.vp),
            "calibration": {"T_ego_cam": env.T_ego_cam().tolist()},
            "source_archive_sha256": digest(path / "source.zip"),
            "source_sha256": {"source.py": hashlib.sha256(b"source").hexdigest()},
            "arrays_sha256": {f"{name}.npy": digest(path / f"{name}.npy") for name in arrays},
            "summary": {"result": "finished", "pass": True, "complete": True, "physical_cone_hits": 0, "oc": 0},
        }
        write(path / "meta.json", meta)
        rows.append(
            {
                "id": case.id,
                "split": case.split,
                "path": case.id,
                "meta_sha256": digest(path / "meta.json"),
                "accepted": 2,
                "rejections": {"bad_teacher": 0, "horizon": len(ticks) - 2, "control": 0, "contact": 0},
                "summary": meta["summary"],
            }
        )
    manifest = root / "manifest.json"
    write(
        manifest,
        {
            "schema_version": 1,
            "complete": True,
            "expected_episodes": len(rows),
            "membership_sha256": digest(membership),
            "episodes": rows,
        },
    )
    write(root / "verification.json", verify(manifest, membership))
    return manifest


def make_checkpoint(path, step, args, policy=1.0):
    model = {
        f"{prefix}.weight": torch.tensor([policy if prefix == "base.policy" else 1.0])
        for prefix in ("base.vision", "base.policy", "bev", "img")
    }
    torch.save({"step": step, "config": args, "model": model}, path)
    return path


def metrics(path, steer=0.002, plan=1.0, longitudinal=1.0):
    values = {"steer": steer, "plan_l1": plan, "longitudinal": longitudinal, "plan_nll": -1.0}
    return {
        "precision": "float32",
        "aggregation": "equal_episode",
        "count": 1,
        "aggregate": values,
        "episodes": [{"path": path, "count": 1, **values}],
    }


@pytest.fixture
def recovery(tmp_path):
    frozen = json.loads(MEMBERSHIP.read_text())["cases"]
    cases = [next(case for case in frozen if case["split"] == split) for split in ("train", "val")]
    membership = tmp_path / "membership.json"
    write(membership, {"cases": cases})
    v2 = make_dataset(tmp_path / "v2", membership)
    shutil.copytree(v2.parent, tmp_path / "v3", copy_function=os.link)
    v3 = tmp_path / "v3/manifest.json"
    train_meta = v3.parent / cases[0]["id"] / "meta.json"
    meta = json.loads(train_meta.read_text())
    meta["recovery"] = {"takeover_tick": 0}
    train_meta.unlink()
    write(train_meta, meta)
    manifest = json.loads(v3.read_text())
    manifest["episodes"][0]["meta_sha256"] = digest(train_meta)
    v3.unlink()
    write(v3, manifest)
    receipt = v3.parent / "verification.json"
    receipt.unlink()
    write(receipt, verify(v3, membership))
    runs = {}
    for name, dataset in (("vision", v2), ("parent", v2), ("recovery", v3)):
        root = tmp_path / name
        root.mkdir()
        manifest = json.loads(dataset.read_text())
        remote_root = f"/work/data/{dataset.parent.name}"
        args = {
            "manifest": f"{remote_root}/manifest.json",
            "out": f"/work/runs/{name}",
            "init": f"/work/runs/{'parent' if name == 'recovery' else 'vision'}/selected.pt",
            "freeze_vision": name != "vision",
            "w_plan": 0.1,
            "steps": 2,
            "val_every": 1,
            "history_context": 8,
        }
        with ZipFile(root / "source.zip", "w") as archive:
            archive.writestr("a.py", b"abc")
        run = {
            "args": args,
            "manifest_sha256": digest(dataset),
            "dataset_verification": {
                "status": "verified",
                "path": f"{remote_root}/verification.json",
                "sha256": digest(dataset.parent / "verification.json"),
                "manifest_sha256": digest(dataset),
                "membership_sha256": digest(membership),
            },
            "source": {
                "archive_sha256": digest(root / "source.zip"),
                "files_sha256": {"a.py": hashlib.sha256(b"abc").hexdigest()},
            },
            "episodes": [
                {
                    "path": f"{remote_root}/{row['id']}",
                    "split": row["split"],
                    "accepted": row["accepted"],
                    "meta_sha256": row["meta_sha256"],
                }
                for row in manifest["episodes"]
            ],
            "validation_selection": [{"path": f"{remote_root}/{cases[1]['id']}", "frames": [1]}],
        }
        runs[name] = (root, run)
    vision, vision_run = runs["vision"]
    parent, parent_run = runs["parent"]
    action, action_run = runs["recovery"]
    initial = make_checkpoint(tmp_path / "initial.pt", 0, {}, 0.0)
    vision_checkpoint = make_checkpoint(vision / "selected.pt", 9, vision_run["args"])
    parent_checkpoint = make_checkpoint(parent / "step_000500.pt", 500, parent_run["args"], 2.0)
    for name, init in (("vision", initial), ("parent", vision_checkpoint), ("recovery", parent_checkpoint)):
        root, run = runs[name]
        run["init_sha256"] = digest(init)
        run["baseline_state_sha256"] = selector.frozen_state(init, None)
        write(root / "run.json", run)
        zero = metrics(run["validation_selection"][0]["path"])
        zero.update(step=0, saved_state_sha256=run["baseline_state_sha256"])
        if name == "recovery":
            checkpoint = make_checkpoint(root / "step_000000.pt", 0, run["args"], 2.0)
            zero.update(checkpoint=f"{run['args']['out']}/step_000000.pt", checkpoint_sha256=digest(checkpoint))
        write(root / "baseline_validation.json", zero)
    for root, checkpoint, step in ((vision, vision_checkpoint, 9), (parent, parent_checkpoint, 500)):
        (root / "history.jsonl").write_text(json.dumps({"step": step, "checkpoint_sha256": digest(checkpoint)}) + "\n")
    rows = []
    for step in (1, 2):
        checkpoint = make_checkpoint(action / f"step_{step:06d}.pt", step, action_run["args"], 3.0)
        rows.append(
            {
                "step": step,
                "checkpoint": f"/work/runs/recovery/{checkpoint.name}",
                "checkpoint_sha256": digest(checkpoint),
                "state_sha256": action_run["baseline_state_sha256"],
                "validation": metrics(action_run["validation_selection"][0]["path"]),
            }
        )
    (action / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    options = {
        "parent_training": parent,
        "dataset": v3,
        "dataset_verification": v3.parent / "verification.json",
        "baseline_dataset": v2,
        "baseline_dataset_verification": v2.parent / "verification.json",
        "parent_dataset": v2,
        "parent_dataset_verification": v2.parent / "verification.json",
        "init_checkpoint": parent_checkpoint,
        "baseline_checkpoint": vision_checkpoint,
        "membership": membership,
    }
    return {"runs": runs, "options": options, "rows": rows, "initial": initial, "cases": cases}


def select(recovery):
    return selector.select(recovery["runs"]["recovery"][0], recovery["runs"]["vision"][0], 2, **recovery["options"])


def test_recovery_uses_frozen_validation_and_retains_incumbent(recovery):
    action = recovery["runs"]["recovery"][0]
    output = action / "selection.json"
    argv = [
        "--training",
        str(action),
        "--baseline-training",
        str(recovery["runs"]["vision"][0]),
        "--expected-steps",
        "2",
        "--out",
        str(output),
    ]
    for name, value in recovery["options"].items():
        argv += [f"--{name.replace('_', '-')}", str(value)]
    selector.main(argv)
    result = json.loads(output.read_text())
    assert result["selected"]["step"] == 0
    assert all(row["eligible"] for row in result["candidates"])
    assert list(result["validation_fingerprint"]) == [recovery["cases"][1]["id"]]
    recovery["rows"][1]["validation"]["aggregate"]["steer"] = 0.001
    recovery["rows"][1]["validation"]["episodes"][0]["steer"] = 0.001
    (action / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in recovery["rows"]))
    assert select(recovery)["selected"]["step"] == 2


@pytest.mark.parametrize(
    "change,match",
    [
        ("frames", "anchors differ"),
        ("context", "history context"),
        ("history", "step differs"),
        ("signature", "signature"),
        ("zero", "Step-zero model state"),
        ("source", "source archive"),
    ],
)
def test_recovery_rejects_changed_evidence(recovery, change, match):
    action, run = recovery["runs"]["recovery"]
    parent = recovery["runs"]["parent"][0]
    if change == "frames":
        run["validation_selection"][0]["frames"] = [0]
        write(action / "run.json", run)
    elif change == "context":
        run["args"]["history_context"] = 25
        write(action / "run.json", run)
    elif change == "history":
        write(
            parent / "history.jsonl", {"step": 499, "checkpoint_sha256": digest(recovery["options"]["init_checkpoint"])}
        )
    elif change in ("signature", "zero"):
        path = action / ("step_000001.pt" if change == "signature" else "step_000000.pt")
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["model"]["base.policy.weight"] = torch.tensor([4.0, 5.0] if change == "signature" else [4.0])
        torch.save(checkpoint, path)
        if change == "signature":
            recovery["rows"][0]["checkpoint_sha256"] = digest(path)
            (action / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in recovery["rows"]))
        else:
            zero = json.loads((action / "baseline_validation.json").read_text())
            zero["checkpoint_sha256"] = digest(path)
            write(action / "baseline_validation.json", zero)
    else:
        (parent / "source.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match=match):
        select(recovery)


def test_actual_validation_arrays_and_case_identity_are_verified(recovery):
    root, run = recovery["runs"]["recovery"]
    dataset = recovery["options"]["dataset"]
    case = recovery["cases"][1]
    path = dataset.parent / case["id"] / "controls.npy"
    arrays = np.load(path).copy()
    arrays[0, 0] = 1
    path.unlink()
    np.save(path, arrays)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validation_identity(run, dataset, dataset.parent / "verification.json", recovery["options"]["membership"])


def test_verified_validation_metadata_change_is_rejected(recovery):
    action, run = recovery["runs"]["recovery"]
    opts = recovery["options"]
    dataset = opts["dataset"]
    meta_path = dataset.parent / recovery["cases"][1]["id"] / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["capture_hz"] = 20
    meta_path.unlink()
    write(meta_path, meta)
    manifest = json.loads(dataset.read_text())
    manifest["episodes"][1]["meta_sha256"] = digest(meta_path)
    write(dataset, manifest)
    write(opts["dataset_verification"], verify(dataset, opts["membership"]))
    run["manifest_sha256"] = digest(dataset)
    run["dataset_verification"].update(manifest_sha256=digest(dataset), sha256=digest(opts["dataset_verification"]))
    run["episodes"][1]["meta_sha256"] = digest(meta_path)
    write(action / "run.json", run)
    with pytest.raises(ValueError, match="validation identities, bytes"):
        select(recovery)


@pytest.mark.parametrize("metric,value", [("plan_l1", 1.1001), ("longitudinal", 1.2501)])
def test_improved_steering_still_requires_other_bounds(recovery, metric, value):
    action = recovery["runs"]["recovery"][0]
    for row in recovery["rows"]:
        for group in (row["validation"]["aggregate"], row["validation"]["episodes"][0]):
            group["steer"] = 0.001
            group[metric] = value
    (action / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in recovery["rows"]))
    result = select(recovery)
    assert result["selected"]["step"] == 0
    assert all(f"{metric} exceeds limit" in row["rejection_reasons"] for row in result["candidates"][1:])


def test_ancestry_binds_each_dataset_and_portable_overrides(recovery):
    vision = recovery["runs"]["vision"][0]
    parent = recovery["runs"]["parent"][0]
    action = recovery["runs"]["recovery"][0]
    opts = recovery["options"]
    chain = [(vision, opts["baseline_checkpoint"]), (parent, opts["init_checkpoint"])]
    overrides = [(root, opts["baseline_dataset"], opts["baseline_dataset_verification"]) for root in (vision, parent)]
    args = (
        action,
        opts["dataset"],
        opts["dataset_verification"],
        freezer.checkpoint_info(recovery["initial"]),
        freezer.checkpoint_info(action / "step_000001.pt"),
        chain,
    )
    with pytest.raises(ValueError, match="explicit dataset"):
        freezer.verify_training_chain(*args, membership=opts["membership"])
    proof = freezer.verify_training_chain(*args, ancestor_datasets=overrides, membership=opts["membership"])
    assert proof["dataset_sha256"] == digest(opts["dataset"])
    assert all(row["dataset_sha256"] == digest(opts["baseline_dataset"]) for row in proof["ancestors"])
    assert proof["validation_fingerprint"] == proof["ancestors"][0]["validation_fingerprint"]
    bad = [(vision, opts["dataset"], opts["dataset_verification"]), overrides[1]]
    with pytest.raises(ValueError, match="initialization or dataset"):
        freezer.verify_training_chain(*args, ancestor_datasets=bad, membership=opts["membership"])
    with pytest.raises(ValueError, match="duplicate ancestor dataset"):
        freezer.verify_training_chain(*args, ancestor_datasets=overrides + overrides[:1], membership=opts["membership"])
