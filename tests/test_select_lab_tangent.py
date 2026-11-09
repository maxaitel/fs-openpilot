"""Offline tests for the frozen tangent selector and its fail-closed gates."""

import copy
import json
from pathlib import Path

import pytest
import torch

from fsop.eval.controller import controller_contract
from scripts import select_lab_tangent as s


def dump(path, obj):
    path.write_text(json.dumps(obj) + "\n")


def coherence(value=0.1):
    return {k: {stat: value for stat in s.STATS} for k in s.COHERENCE_METRICS}


def fixture(tmp_path, monkeypatch):
    training, calibration = tmp_path / "training", tmp_path / "calibration"
    training.mkdir()
    calibration.mkdir()
    init = tmp_path / "init.pt"
    init.write_bytes(b"initializer")
    monkeypatch.setattr(s, "INITIALIZER", s.sha(init))
    cases = sorted(s.CASE_IDS)
    paths = {f"/recorded/{case}": case for case in cases}
    anchors = [{"path": path, "frames": list(range(32))} for path in paths]

    def metrics(steer):
        rows = [
            {
                "path": path,
                "count": 32,
                "plan_l1": 0.1,
                "steer": steer,
                "longitudinal": 0.1,
                "plan_nll": 0.0,
                "coherence": coherence(),
                "target_coherence": coherence(),
            }
            for path in paths
        ]
        return {
            "precision": "float32",
            "aggregation": "equal_episode",
            "count": 256,
            "aggregate": {"plan_l1": 0.1, "steer": steer, "longitudinal": 0.1, "plan_nll": 0.0},
            "episodes": rows,
            "coherence": coherence(),
            "target_coherence": coherence(),
        }

    contract = controller_contract("op_tangent_v1")
    ref = {str(i): {"sha256": "a"} for i in range(347)}
    recipe = tmp_path / "recipe.json"
    recipe.write_text("recipe")
    gate = tmp_path / "gate.json"
    gate.write_text("gate")
    sampler = {
        "recipe_path": str(recipe),
        "recipe_sha256": s.sha(recipe),
        "recovery_episodes": [{"gate_path": str(gate), "gate_sha256": s.sha(gate)}],
        "provenance_mode": "off_policy_v1",
        "collection_policy_checkpoint_sha": s.COLLECTION,
        "optimization_init_sha256": s.INITIALIZER,
        "recovery_draws_per_batch": 16,
        "other_draws_per_batch": 112,
    }
    args = {
        "init": str(init),
        "out": str(training),
        "controller_mode": "op_tangent_v1",
        "steps": 500,
        "val_every": 100,
        "freeze_vision": True,
        "w_plan": 1.0,
        "w_steer": 3.281158240898283,
        "w_long": 1.0,
        "history_context": 8,
        "l2sp": 0.0,
        "batch_size": 128,
        "history_chunk_size": 64,
        "channels_last": False,
        "cache_frozen_features": False,
        "val_samples": 256,
        "recovery_draws": 16,
        "lr": 0.0001,
        "lr_policy": 0.3,
        "lr_trunk": 0.1,
        "grad_clip": 1.0,
        "threads": 4,
        "amp": False,
        "device": "cuda",
        "seed": 0,
        "recovery_provenance_mode": "off_policy_v1",
        "collection_policy_checkpoint_sha": s.COLLECTION,
        "recovery_cases": str(recipe),
        "expected_calibration": str(calibration / "calibration.json"),
    }
    manifest = "b" * 64
    episodes = [{"path": path} for path in paths]
    names = [f"base.policy.weight{i}" for i in range(42)]
    model = {**{name: torch.tensor([0.0]) for name in names}, **{f"fixed{i}": torch.tensor([0.0]) for i in range(375)}}
    run = {
        "args": args,
        "trainable_tensors": names,
        "calibration_only": False,
        "controller_contract": contract,
        "baseline_state_sha256": ref,
        "init_sha256": s.INITIALIZER,
        "recovery_sampling": sampler,
        "manifest_sha256": manifest,
        "validation_selection": anchors,
        "episodes": episodes,
        "dataset_verification": {"sha256": "c" * 64},
        "source": {"files_sha256": {"source": "d" * 64}},
        "history_context_proof": {"ok": True},
        "validation_precision": "float32",
        "device": "cuda",
        "tf32": {"matmul": False, "cudnn": False},
    }
    ci = {
        "version": 1,
        "controller_contract": contract,
        "init_sha256": s.INITIALIZER,
        "manifest_sha256": manifest,
        "dataset_verification_sha256": "c" * 64,
        "episodes": episodes,
        "validation_selection": anchors,
        "source_files_sha256": run["source"]["files_sha256"],
        "history_context": 8,
        "history_context_proof": run["history_context_proof"],
        "freeze_vision": True,
        "validation_precision": "float32",
        "device": "cuda",
        "validation_batch_size": 128,
        "history_chunk_size": 64,
        "channels_last": False,
        "cache_enabled": False,
        "tf32": run["tf32"],
        "weights": {"plan": 1.0, "steer": args["w_steer"], "longitudinal": 1.0},
        "recovery_sampling": sampler,
    }
    cal = {"version": 1, "identity": ci, "validation": metrics(0.2)}
    for root in (training, calibration):
        dump(root / "calibration.json", cal)
        (root / "source.zip").write_bytes(b"source")
    dump(training / "run.json", run)
    cal_run = copy.deepcopy(run)
    cal_run["calibration_only"] = True
    cal_run["args"]["out"] = str(calibration)
    dump(calibration / "run.json", cal_run)
    cal_ckpt = calibration / "step_000000.pt"
    cal_ckpt.write_text("0")
    cal_zero = {
        **metrics(0.2),
        "step": 0,
        "checkpoint": str(cal_ckpt),
        "checkpoint_sha256": s.sha(cal_ckpt),
        "saved_state_sha256": ref,
        "selected_sha256": "unused",
    }
    dump(calibration / "baseline_validation.json", cal_zero)
    episode_limits = {case: {k: 1.0 for k in s.ACTION_METRICS} for case in cases}

    def limits():
        return {k: {stat: 1.0 for stat in s.STATS} for k in s.COHERENCE_METRICS}

    protocol = {
        "schema_version": 1,
        "kind": "op_tangent_v1_selection",
        "calibration_sha256": s.sha(calibration / "calibration.json"),
        "initializer_sha256": s.INITIALIZER,
        "manifest_sha256": manifest,
        "controller_contract": contract,
        "expected_steps": 500,
        "checkpoint_steps": s.STEPS,
        "training_args": {key: args[key] for key in s.TRAINING_ARGS},
        "bounds": {
            "aggregate": {k: 1.0 for k in s.ACTION_METRICS},
            "episode": episode_limits,
            "coherence": limits(),
            "episode_coherence": {case: limits() for case in cases},
        },
    }
    protocol_path = tmp_path / "protocol.json"
    dump(protocol_path, protocol)
    protocol_sha = s.sha(protocol_path)
    (training / "selection_protocol.json").write_bytes(protocol_path.read_bytes())
    args["selection_protocol"] = str(protocol_path)
    args["selection_protocol_sha"] = protocol_sha
    run["selection_protocol"] = {
        "path": str(protocol_path),
        "sha256": protocol_sha,
        "calibration_path": str(calibration / "calibration.json"),
        "calibration_sha256": protocol["calibration_sha256"],
    }
    dump(training / "run.json", run)
    history = []
    minimal_sampler = {
        k: sampler[k] for k in ("provenance_mode", "collection_policy_checkpoint_sha", "optimization_init_sha256")
    }
    for step in range(501):
        if step in [0, 1, *s.STEPS[1:]]:
            ckpt = training / f"step_{step:06d}.pt"
            ckpt.write_text(str(step))
        if step == 0:
            zero = {
                **metrics(0.2),
                "step": 0,
                "checkpoint": str(ckpt),
                "checkpoint_sha256": s.sha(ckpt),
                "saved_state_sha256": ref,
                "selected_sha256": "unused",
            }
            dump(training / "baseline_validation.json", zero)
            continue
        row = {
            "step": step,
            "controller_contract": contract,
            "recovery_provenance": minimal_sampler,
            "selection_protocol_sha256": protocol_sha,
            "calibration_sha256": protocol["calibration_sha256"],
        }
        if step in [1, *s.STEPS[1:]]:
            counts_path = training / f"sampling_counts_{step:06d}.json"
            dump(
                counts_path,
                {
                    "total_draws": 128 * step,
                    "recovery_draws": 16 * step,
                    "episodes": [{"anchors": [{"draws": 128 * step}]}],
                },
            )
            row["sampling"] = {
                "counts_path": str(counts_path),
                "counts_sha256": s.sha(counts_path),
                "total_draws": 128 * step,
                "recovery_draws": 16 * step,
            }
            row.update(
                checkpoint=str(ckpt),
                checkpoint_sha256=s.sha(ckpt),
                state_sha256=ref,
                validation=metrics(0.1 if step in (1, 100, 200) else 0.3),
            )
        history.append(row)
    (training / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    monkeypatch.setattr(s, "source_check", lambda *a: None)
    monkeypatch.setattr(s, "dataset_paths", lambda *a: (None, None))
    identity = {"fingerprint": {case: {} for case in cases}, "path_to_case": paths, "input_sha256": {}}
    monkeypatch.setattr(s, "validation_identity", lambda *a: identity)
    monkeypatch.setattr(s, "frozen_state", lambda *a: ref)

    def info(path):
        step = 0 if Path(path) == init else int(Path(path).stem.split("_")[-1])
        return {
            "step": step,
            "signature": {"tensor": {}},
            "state_sha256": "init" if step == 0 else str(step),
            "groups": {"policy": {"state_sha256": "init" if step == 0 else str(step)}},
        }

    monkeypatch.setattr(s, "checkpoint_info", info)

    def load(path, **kwargs):
        path = Path(path)
        if path == init:
            return {"model": model}
        step = int(path.stem.split("_")[-1])
        checkpoint_model = dict(model)
        if step:
            checkpoint_model[names[0]] = torch.tensor([float(step)])
        return {
            "model": checkpoint_model,
            "config": cal_run["args"] if path.parent == calibration else args,
            "controller_contract": contract,
            "recovery_provenance": sampler,
            "selection_protocol_sha256": protocol_sha,
            "calibration_sha256": protocol["calibration_sha256"],
        }

    monkeypatch.setattr("torch.load", load)
    return training, calibration, protocol_path, protocol, history


def test_select_and_exclude_step_one(tmp_path, monkeypatch):
    training, cal, protocol, _, _ = fixture(tmp_path, monkeypatch)
    result = s.select(training, cal, protocol)
    assert result["selected"]["step"] == 100
    assert result["excluded_step_one"]["step"] == 1
    assert len(result["candidates"]) == 6
    assert result["trained_candidate"]


def test_missing_history_and_controller_tampering(tmp_path, monkeypatch):
    training, cal, protocol, _, history = fixture(tmp_path, monkeypatch)
    (training / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history[1:]))
    with pytest.raises(ValueError, match="all 500"):
        s.select(training, cal, protocol)
    history[7]["controller_contract"] = {"mode": "pp"}
    (training / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    with pytest.raises(ValueError, match="history controller"):
        s.select(training, cal, protocol)


def test_calibration_and_checkpoint_tampering(tmp_path, monkeypatch):
    training, cal, protocol, _, _ = fixture(tmp_path, monkeypatch)
    cp = cal / "calibration.json"
    cp.write_text(cp.read_text() + " ")
    with pytest.raises(ValueError, match="calibration identity"):
        s.select(training, cal, protocol)
    cp.write_text(cp.read_text().rstrip() + "\n")
    (training / "step_000100.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        s.select(training, cal, protocol)


def test_complete_bounds_and_episode_coherence_gate(tmp_path, monkeypatch):
    training, cal, protocol_path, protocol, _ = fixture(tmp_path, monkeypatch)
    case = sorted(s.CASE_IDS)[0]
    protocol["bounds"]["episode_coherence"][case]["yaw_cumulative_max_rad"]["p95"] = 0.01
    dump(protocol_path, protocol)
    new_sha = s.sha(protocol_path)
    (training / "selection_protocol.json").write_bytes(protocol_path.read_bytes())
    run_path = training / "run.json"
    run = json.loads(run_path.read_text())
    run["selection_protocol"]["sha256"] = new_sha
    run["args"]["selection_protocol_sha"] = new_sha
    dump(run_path, run)
    # The trainer would have saved these hashes before updating; adjust the synthetic receipts.
    history = [json.loads(line) for line in (training / "history.jsonl").read_text().splitlines()]
    for row in history:
        row["selection_protocol_sha256"] = new_sha
    (training / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    original_load = torch.load

    def rebound(path, **kwargs):
        payload = original_load(path, **kwargs)
        if Path(path).parent == training:
            payload["selection_protocol_sha256"] = new_sha
            payload["config"]["selection_protocol_sha"] = new_sha
        return payload

    monkeypatch.setattr("torch.load", rebound)
    result = s.select(training, cal, protocol_path)
    assert result["status"] == "retain_step_zero"
    assert not result["trained_candidate"]
    assert not result["selected_eligible"]
    assert result["selected_rejection_reasons"]
    assert any("yaw_cumulative_max_rad.p95" in x for x in result["candidates"][1]["rejection_reasons"])
    del protocol["bounds"]["episode_coherence"][case]["yaw_cumulative_max_rad"]
    dump(protocol_path, protocol)
    with pytest.raises(ValueError, match="incomplete"):
        s.select(training, cal, protocol_path)


def test_non_trainable_state_tampering(tmp_path, monkeypatch):
    training, cal, protocol, _, _ = fixture(tmp_path, monkeypatch)
    original_load = torch.load

    def altered_load(path, **kwargs):
        payload = original_load(path, **kwargs)
        if Path(path).name == "step_000100.pt":
            payload["model"]["fixed0"] = torch.tensor([1.0])
        return payload

    monkeypatch.setattr("torch.load", altered_load)
    with pytest.raises(ValueError, match="non-trainable tensor changed"):
        s.select(training, cal, protocol)


@pytest.mark.parametrize("scope", ["global", "episode"])
def test_native_target_coherence_cannot_change(tmp_path, monkeypatch, scope):
    training, cal, protocol, _, history = fixture(tmp_path, monkeypatch)
    validation = history[99]["validation"]
    target = validation if scope == "global" else validation["episodes"][0]
    target["target_coherence"]["position_interval_m"]["mean"] = 0.09
    (training / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    with pytest.raises(ValueError, match="native target coherence changed at step 100"):
        s.select(training, cal, protocol)
