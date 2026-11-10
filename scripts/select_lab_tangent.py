"""Select an op_tangent_v1 checkpoint using a frozen, post-calibration protocol.

Protocol JSON (all numeric placeholders must be replaced before training):
{
  "schema_version": 1, "kind": "op_tangent_v1_selection",
  "calibration_sha256": "<sha256>", "initializer_sha256": "<sha256>",
  "manifest_sha256": "<sha256>", "controller_contract": {"<full contract>": "..."},
  "expected_steps": 500, "checkpoint_steps": [0,100,200,300,400,500],
  "training_args": {"<every TRAINING_ARGS key>": "<frozen value>"},
  "bounds": {
    "aggregate": {"plan_l1": null, "steer": null, "longitudinal": null},
    "episode": {"<each validation case id>": {"plan_l1": null, "steer": null, "longitudinal": null}},
    "coherence": {"<each COHERENCE_METRICS key>": {"mean": null, "p95": null, "max": null}},
    "episode_coherence": {"<each validation case id>": {
      "<each COHERENCE_METRICS key>": {"mean": null, "p95": null, "max": null}}}
  }
}
All bounds are absolute, inclusive ceilings. Nulls are deliberately invalid.
The protocol and calibration must be frozen before the first optimization update.
For native-future targets, schema_version 2 additionally requires
action_target_contract and training_args.action_target_mode=native_future_v1.
The action bounds then apply to those targets, not to teacher-command diagnostics.
For PI and desired-motion targets, schema_version 3 uses
kind=op_tangent_pi_v1_selection, the desired-motion contract, a positive frozen
w_consistency, and ranking=PI_RANKING. It selects a strict aggregate plan L1
improvement over step zero, subject to the same absolute coherence gates.
Schema 4 additionally fixes the final_plan optimizer scope and bounded head recipe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from fsop.train.action_target import (
    DESIRED_MOTION_CONTRACT, DESIRED_MOTION_TARGET, LEGACY_ACTION_TARGET,
    NATIVE_FUTURE_ACTION_TARGET, NATIVE_FUTURE_CONTRACT,
)
from scripts.freeze_lab_selection import checkpoint_info
from scripts.lab_training_provenance import MEMBERSHIP, dataset_paths, run_checkpoint, validation_identity
from scripts.select_lab_action import frozen_state, number, read, sha, source_check, validation

INITIALIZER = "e59efc8ace2d05a5ecfc29023712e7b5ce1d34a64133dd869a7bc17f849bc35f"
COLLECTION = "1ac2e3ba949db39ab445c636f841360973e2add8c86edab689a3a872f040ce55"
STEPS = [0, 100, 200, 300, 400, 500]
ACTION_METRICS = ("plan_l1", "steer", "longitudinal")
COHERENCE_METRICS = (
    "position_interval_m",
    "position_interval_max_m",
    "velocity_interval_mps",
    "velocity_interval_max_mps",
    "yaw_interval_rad",
    "yaw_interval_max_rad",
    "initial_position_m",
    "initial_yaw_rad",
    "initial_velocity_mps",
    "initial_yaw_rate_radps",
    "position_cumulative_max_m",
    "velocity_cumulative_max_mps",
    "yaw_cumulative_max_rad",
    "velocity_secant_max_mps",
    "acceleration_secant_max_mps2",
    "yaw_rate_secant_max_radps",
)
TRAINING_ARGS = (
    "steps",
    "val_every",
    "batch_size",
    "lr",
    "lr_policy",
    "lr_trunk",
    "freeze_vision",
    "l2sp",
    "w_plan",
    "w_steer",
    "w_long",
    "grad_clip",
    "val_samples",
    "history_chunk_size",
    "history_context",
    "threads",
    "amp",
    "channels_last",
    "device",
    "seed",
    "cache_frozen_features",
    "recovery_draws",
    "recovery_provenance_mode",
    "collection_policy_checkpoint_sha",
)
PI_RANKING = {"metric": "aggregate.plan_l1", "improvement": "strict_step_zero", "tie_break": "lowest_step"}
STATS = ("mean", "p95", "max")
CASE_IDS = frozenset(c["id"] for c in read(MEMBERSHIP)["cases"] if c["split"] == "val")


def protocol_check(p):
    from fsop.eval.controller import controller_contract

    version = p.get("schema_version")
    if type(version) is not int or version not in (1, 2, 3, 4):
        raise ValueError("invalid protocol version")
    fields = {
        "schema_version",
        "kind",
        "calibration_sha256",
        "initializer_sha256",
        "manifest_sha256",
        "controller_contract",
        "expected_steps",
        "checkpoint_steps",
        "training_args",
        "bounds",
    }
    if version in (2, 3, 4):
        fields.add("action_target_contract")
    if version in (3, 4):
        fields.add("ranking")
    if set(p) != fields:
        raise ValueError("protocol fields mismatch")
    if (
        version not in (1, 2, 3, 4)
        or p["kind"] != ("op_tangent_pi_v1_selection" if version in (3, 4) else "op_tangent_v1_selection")
        or (version in (1, 2) and p["initializer_sha256"] != INITIALIZER)
        or p["expected_steps"] != 500
        or p["checkpoint_steps"] != STEPS
        or p["controller_contract"] != controller_contract("op_tangent_pi_v1" if version in (3, 4) else "op_tangent_v1")
        or (version == 2 and p.get("action_target_contract") != NATIVE_FUTURE_CONTRACT)
        or (version in (3, 4) and (p.get("action_target_contract") != DESIRED_MOTION_CONTRACT
                              or p.get("ranking") != PI_RANKING))
    ):
        raise ValueError("protocol identity mismatch")
    for key in ("calibration_sha256", "manifest_sha256", "initializer_sha256"):
        if not isinstance(p[key], str) or not re.fullmatch("[0-9a-f]{64}", p[key]):
            raise ValueError(f"invalid protocol {key}")
    training_fields = set(TRAINING_ARGS) | ({"action_target_mode"} if version >= 2 else set())
    if version in (3, 4):
        training_fields.add("w_consistency")
    if version == 4:
        training_fields.add("optimizer_scope")
    if not isinstance(p["training_args"], dict) or set(p["training_args"]) != training_fields:
        raise ValueError("incomplete frozen training arguments")
    if version == 2 and p["training_args"]["action_target_mode"] != NATIVE_FUTURE_ACTION_TARGET:
        raise ValueError("action target mode mismatch")
    if version in (3, 4) and (p["training_args"]["action_target_mode"] != DESIRED_MOTION_TARGET
                         or type(p["training_args"]["w_consistency"]) not in (float, int)
                         or not math.isfinite(p["training_args"]["w_consistency"])
                         or p["training_args"]["w_consistency"] <= 0):
        raise ValueError("invalid desired motion target or consistency weight")
    if any(value is None for key, value in p["training_args"].items()
           if not (version in (3, 4) and key == "collection_policy_checkpoint_sha")):
        raise ValueError("missing frozen training argument value")
    if version in (3, 4):
        fixed = {"steps": 500, "val_every": 100, "freeze_vision": True, "w_plan": 1.0,
                 "w_steer": 0.0, "w_long": 0.0, "history_context": 8, "l2sp": 0.0,
                 "batch_size": 128, "val_samples": 256, "recovery_draws": 16,
                 "recovery_provenance_mode": "per_case_v2", "collection_policy_checkpoint_sha": None}
        if version == 4:
            fixed.update(w_steer=3.281158240898283, w_consistency=1.0, optimizer_scope="final_plan", amp=False)
            rate = p["training_args"].get("lr")
            multiplier = p["training_args"].get("lr_policy")
            if (type(rate) not in (int, float) or type(multiplier) not in (int, float)
                    or rate <= 0 or multiplier <= 0
                    or not math.isclose(rate * multiplier, 3e-6, rel_tol=1e-12, abs_tol=0.0)):
                raise ValueError("final plan learning rate mismatch")
        if any(p["training_args"].get(key) != value for key, value in fixed.items()):
            raise ValueError("PI training configuration mismatch")
    b = p["bounds"]
    if not isinstance(b, dict) or set(b) != {"aggregate", "episode", "coherence", "episode_coherence"}:
        raise ValueError("incomplete bound inventory")
    if not isinstance(b["aggregate"], dict) or set(b["aggregate"]) != set(ACTION_METRICS):
        raise ValueError("incomplete aggregate bounds")
    for k, v in b["aggregate"].items():
        number(v, f"aggregate.{k}")
    for scope in ("episode", "episode_coherence"):
        if not isinstance(b[scope], dict) or set(b[scope]) != CASE_IDS:
            raise ValueError(f"incomplete {scope} case inventory")
    for case, limits in b["episode"].items():
        if not isinstance(limits, dict) or set(limits) != set(ACTION_METRICS):
            raise ValueError(f"incomplete episode bounds: {case}")
        for key, value in limits.items():
            number(value, f"episode.{case}.{key}")

    def coherence_bounds(limits, label):
        if not isinstance(limits, dict) or set(limits) != set(COHERENCE_METRICS):
            raise ValueError(f"incomplete {label} bounds")
        for key, stats in limits.items():
            if not isinstance(stats, dict) or set(stats) != set(STATS):
                raise ValueError(f"incomplete {label}.{key} bounds")
            for stat, value in stats.items():
                number(value, f"{label}.{key}.{stat}")

    coherence_bounds(b["coherence"], "coherence")
    for case, limits in b["episode_coherence"].items():
        coherence_bounds(limits, f"episode_coherence.{case}")
    return p


def check_coherence(doc, label):
    if not isinstance(doc, dict) or set(doc) != set(COHERENCE_METRICS):
        raise ValueError(f"incomplete {label} metrics")
    for key, stats in doc.items():
        if not isinstance(stats, dict) or set(stats) != set(STATS):
            raise ValueError(f"incomplete {label}.{key} statistics")
        for stat in STATS:
            number(stats[stat], f"{label}.{key}.{stat}")
        if not stats["mean"] <= stats["max"] or not stats["p95"] <= stats["max"]:
            raise ValueError(f"inconsistent {label}.{key} statistics")


def assess(v, bounds, anchors, path_to_case):
    validation(v, anchors)
    if v["count"] != 256:
        raise ValueError("expected exactly 256 validation anchors")
    check_coherence(v.get("coherence"), "coherence")
    check_coherence(v.get("target_coherence"), "target_coherence")
    for row in v["episodes"]:
        check_coherence(row.get("coherence"), "episode coherence")
        check_coherence(row.get("target_coherence"), "episode target_coherence")
    gates = []

    def gate(path, actual, limit):
        gates.append({"metric": path, "actual": actual, "limit": limit, "passed": actual <= limit})

    for key in ACTION_METRICS:
        gate(f"aggregate.{key}", v["aggregate"][key], bounds["aggregate"][key])
    for row in v["episodes"]:
        case = path_to_case[row["path"]]
        for key in ACTION_METRICS:
            gate(f"episode.{case}.{key}", row[key], bounds["episode"][case][key])
    for key in COHERENCE_METRICS:
        for stat in STATS:
            gate(f"coherence.{key}.{stat}", v["coherence"][key][stat], bounds["coherence"][key][stat])
            for row in v["episodes"]:
                case = path_to_case[row["path"]]
                gate(
                    f"episode_coherence.{case}.{key}.{stat}",
                    row["coherence"][key][stat],
                    bounds["episode_coherence"][case][key][stat],
                )
    return gates


def policy_trainable_names(checkpoint):
    from fsop.model.build import load_fs

    model = load_fs(checkpoint, map_location="cpu")
    return ["base.policy." + name for name, tensor in model.base.policy.named_parameters() if tensor.requires_grad]


def verify_final_plan_optimizer(payload, step, learning_rate):
    expected = {"optimized_parameters": 254430, "gradient_tensors": 0 if step == 0 else 2,
                "optimizer_state_tensors": 0 if step == 0 else 2, "learning_rate": learning_rate}
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict) or payload.get("optimizer_receipt") != expected:
        raise ValueError("final plan optimizer receipt mismatch")
    groups, state = optimizer.get("param_groups"), optimizer.get("state")
    if (not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict)
            or groups[0].get("lr") != learning_rate or groups[0].get("weight_decay") != 1e-4
            or tuple(groups[0].get("betas", ())) != (0.9, 0.999) or groups[0].get("eps") != 1e-8
            or groups[0].get("amsgrad") is not False or not isinstance(groups[0].get("params"), list)
            or len(groups[0]["params"]) != 2 or len(set(groups[0]["params"])) != 2
            or not isinstance(state, dict)
            or set(state) != (set() if step == 0 else set(groups[0]["params"]))):
        raise ValueError("final plan optimizer state mismatch")
    if step:
        import torch

        for param, shape in zip(groups[0]["params"], ((990, 256), (990,)), strict=True):
            slot = state[param]
            if (not isinstance(slot, dict) or set(slot) != {"step", "exp_avg", "exp_avg_sq"}
                    or not isinstance(slot["step"], torch.Tensor) or slot["step"].numel() != 1
                    or slot["step"].item() != step):
                raise ValueError("final plan AdamW step mismatch")
            for key in ("exp_avg", "exp_avg_sq"):
                value = slot[key]
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                        or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                    raise ValueError("final plan AdamW moment mismatch")


def verify_per_case_sampling(sampler, dataset, recipe_path, args, bind):
    from fsop.data.native_replay import NativeReplay, episode_paths, load_episodes
    from fsop.train.lab_train import RecoverySampler
    from scripts.verify_lab_recovery import verify

    recipe = read(recipe_path)
    ids, records = recipe.get("recovery_case_ids"), recipe.get("collection_by_case")
    episodes = sampler["recovery_episodes"]
    if (recipe.get("schema_version") != 2 or not isinstance(ids, list) or not ids
            or len(set(ids)) != len(ids) or not isinstance(records, dict) or set(records) != set(ids)
            or [row.get("case_id") for row in episodes] != ids):
        raise ValueError("per-case recovery inventory mismatch")

    def bind_recipe_inputs(path, case_id, seen=None):
        seen = set() if seen is None else seen
        if path in seen:
            raise ValueError("cyclic recovery recipe")
        seen.add(path)
        doc = read(bind(path))
        bind(path.parent / doc["membership"])
        bind(path.parent / "cases" / (case_id + ".json"))
        legacy = doc.get("collection_by_case", {}).get(case_id, {}).get("legacy_receipt")
        if legacy:
            bind(path.parent / legacy["path"])
            bind_recipe_inputs((path.parent / legacy["recipe"]).resolve(), case_id, seen)

    paths = episode_paths(manifest=dataset)
    for row in episodes:
        case_id = row["case_id"]
        record = records[case_id]
        index = row.get("episode")
        if (not isinstance(record, dict) or type(index) is not int or not 0 <= index < len(paths)
                or row.get("collection_contract") != record
                or row.get("collection_checkpoint_sha256") != record.get("checkpoint_sha256")
                or not isinstance(record.get("checkpoint_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["checkpoint_sha256"])):
            raise ValueError("per-case collection identity mismatch")
        path = Path(paths[index]).resolve()
        meta = read(bind(path / "meta.json"))
        if meta.get("id") != case_id:
            raise ValueError("per-case episode identity mismatch")
        for name in meta["arrays_sha256"]:
            bind(path / name)
        bind(path / "source.zip")
        bind(path / "track.json")
        bind_recipe_inputs(recipe_path, case_id)
        receipt = read(bind(row["gate_path"]))
        if (verify(path, recipe_path) != receipt or receipt.get("passed") is not True
                or receipt.get("case_id") != case_id
                or receipt.get("takeover_tick") != row.get("takeover_tick")
                or receipt.get("terminal_start_tick") != row.get("terminal_start_tick")
                or receipt.get("recovery_frame_indices") != row.get("frames")
                or receipt.get("recovery_ticks") != row.get("ticks")):
            raise ValueError("per-case recovery gate mismatch")
    replay = NativeReplay(load_episodes(paths), "train")
    actual = RecoverySampler(replay, recipe_path, args["recovery_gates"], 128, 16, sampler["optimization_init_sha256"],
                             provenance_mode="per_case_v2", collection_policy_checkpoint_sha=None).protocol()
    if actual != sampler:
        raise ValueError("per-case recovery sampling differs from verified dataset")


def select(training, calibration, protocol):
    training, calibration, protocol = map(lambda x: Path(x).resolve(), (training, calibration, protocol))
    if training == calibration:
        raise ValueError("calibration and training directories must differ")
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha(path)
        return path

    p = protocol_check(read(bind(protocol)))
    cal = read(bind(calibration / "calibration.json"))
    trained_cal_path = bind(training / "calibration.json")
    if sha(calibration / "calibration.json") != p["calibration_sha256"] or read(trained_cal_path) != cal:
        raise ValueError("calibration identity or metrics differ from frozen calibration")
    run, cal_run = read(bind(training / "run.json")), read(bind(calibration / "run.json"))
    version = p["schema_version"]
    initializer = p["initializer_sha256"]
    target_contract = {1: None, 2: NATIVE_FUTURE_CONTRACT, 3: DESIRED_MOTION_CONTRACT, 4: DESIRED_MOTION_CONTRACT}[version]
    target_mode = {1: LEGACY_ACTION_TARGET, 2: NATIVE_FUTURE_ACTION_TARGET, 3: DESIRED_MOTION_TARGET, 4: DESIRED_MOTION_TARGET}[version]
    provenance_mode = "per_case_v2" if version >= 3 else "off_policy_v1"
    collection_sha = None if version >= 3 else COLLECTION
    for record in (run, cal_run):
        if (record.get("action_target_contract") != target_contract or
                record["args"].get("action_target_mode", LEGACY_ACTION_TARGET) != target_mode):
            raise ValueError("run action target contract mismatch")
    if cal.get("identity", {}).get("action_target_contract") != target_contract:
        raise ValueError("calibration action target contract mismatch")
    binding = run.get("selection_protocol", {})
    protocol_sha = inputs[str(protocol)]
    if (
        binding.get("sha256") != protocol_sha
        or binding.get("calibration_sha256") != p["calibration_sha256"]
        or not binding.get("path")
        or not binding.get("calibration_path")
        or sha(bind(binding["path"])) != protocol_sha
        or sha(bind(binding["calibration_path"])) != p["calibration_sha256"]
        or sha(bind(training / "selection_protocol.json")) != protocol_sha
        or run["args"].get("selection_protocol_sha") != protocol_sha
        or not run["args"].get("selection_protocol")
        or sha(bind(run["args"]["selection_protocol"])) != protocol_sha
    ):
        raise ValueError("training selection protocol receipt mismatch")
    if run.get("calibration_only") is not False or cal_run.get("calibration_only") is not True:
        raise ValueError("incorrect calibration or training run type")
    identities = {}
    for root, record in ((training, run), (calibration, cal_run)):
        source_check(root, record)
        if version >= 3:
            repository = Path(__file__).resolve().parents[1]
            for name in ("scripts/select_lab_tangent.py", "scripts/verify_lab_recovery.py",
                         "isaac_lab_fsop/recovery_schedule.py"):
                if record["source"]["files_sha256"].get(name) != sha(repository / name):
                    raise ValueError("PI native training dependency archive mismatch")
        bind(root / "source.zip")
        dataset, receipt = dataset_paths(record)
        identity = validation_identity(record, dataset, receipt, MEMBERSHIP)
        inputs.update(identity["input_sha256"])
        identities[root] = identity
        if sum(len(a["frames"]) for a in record["validation_selection"]) != 256:
            raise ValueError("expected exactly 256 validation anchors")
        if record.get("manifest_sha256") != p["manifest_sha256"]:
            raise ValueError("manifest differs from protocol")
        if record.get("controller_contract") != p["controller_contract"]:
            raise ValueError("controller contract mismatch")
        if record.get("init_sha256") != initializer or sha(bind(record["args"]["init"])) != initializer:
            raise ValueError("initializer mismatch")
        if record["baseline_state_sha256"] != run["baseline_state_sha256"]:
            raise ValueError("frozen initialization state mismatch")
        if len(record["baseline_state_sha256"]) != 347:
            raise ValueError("expected 347 frozen tensors")
        sampler = record.get("recovery_sampling", {})
        if (
            sampler.get("provenance_mode") != provenance_mode
            or sampler.get("collection_policy_checkpoint_sha") != collection_sha
            or sampler.get("optimization_init_sha256") != initializer
            or sampler.get("recovery_draws_per_batch") != 16
            or sampler.get("other_draws_per_batch") != 112
        ):
            raise ValueError("off-policy sampling identity mismatch")
        recipe_path = bind(sampler["recipe_path"])
        if sha(recipe_path) != sampler["recipe_sha256"]:
            raise ValueError("recovery recipe hash mismatch")
        if not sampler.get("recovery_episodes"):
            raise ValueError("missing recovery episode gate inventory")
        for episode in sampler["recovery_episodes"]:
            if sha(bind(episode["gate_path"])) != episode["gate_sha256"]:
                raise ValueError("recovery gate hash mismatch")
        if version >= 3:
            verify_per_case_sampling(sampler, dataset, recipe_path, record["args"], bind)
        if record["args"].get("controller_mode") != ("op_tangent_pi_v1" if version >= 3 else "op_tangent_v1"):
            raise ValueError("training controller mode mismatch")
    if (
        cal_run["source"]["files_sha256"] != run["source"]["files_sha256"]
        or cal_run["episodes"] != run["episodes"]
        or cal_run["validation_selection"] != run["validation_selection"]
        or cal_run["recovery_sampling"] != run["recovery_sampling"]
        or cal_run.get("trainable_tensors") != run.get("trainable_tensors")
    ):
        raise ValueError("calibration run metadata differs from training")
    if (
        identities[training]["fingerprint"] != identities[calibration]["fingerprint"]
        or set(identities[training]["fingerprint"]) != CASE_IDS
    ):
        raise ValueError("calibration validation identity mismatch")
    args = run["args"]
    if {key: args.get(key) for key in p["training_args"]} != p["training_args"]:
        raise ValueError("training arguments differ from frozen protocol")
    if (
        args.get("steps") != 500
        or args.get("val_every") != 100
        or args.get("freeze_vision") is not True
        or args.get("w_plan") != 1.0
        or args.get("w_steer") != (0.0 if version == 3 else 3.281158240898283)
        or args.get("w_long") != (0.0 if version >= 3 else 1.0)
        or args.get("history_context") != 8
        or args.get("l2sp") != 0.0
        or args.get("batch_size") != 128
        or args.get("val_samples") != 256
        or args.get("recovery_draws") != 16
        or args.get("recovery_provenance_mode") != provenance_mode
        or args.get("collection_policy_checkpoint_sha") != collection_sha
        or not args.get("recovery_cases")
        or Path(args["recovery_cases"]).resolve() != Path(run["recovery_sampling"]["recipe_path"]).resolve()
    ):
        raise ValueError("training configuration mismatch")
    if not args.get("expected_calibration") or sha(bind(args["expected_calibration"])) != p["calibration_sha256"]:
        raise ValueError("training did not require the frozen calibration")
    if cal.get("version") != 1 or cal.get("identity", {}).get("controller_contract") != p["controller_contract"]:
        raise ValueError("calibration controller identity mismatch")
    if (
        cal["identity"].get("init_sha256") != initializer
        or cal["identity"].get("manifest_sha256") != p["manifest_sha256"]
        or cal["identity"].get("validation_selection") != run["validation_selection"]
        or cal["identity"].get("episodes") != run["episodes"]
        or cal["identity"].get("source_files_sha256") != run["source"]["files_sha256"]
        or cal["identity"].get("recovery_sampling") != run["recovery_sampling"]
    ):
        raise ValueError("calibration run provenance mismatch")
    ci = cal["identity"]
    expected_identity = {
        "dataset_verification_sha256": run["dataset_verification"]["sha256"],
        "history_context": args.get("history_context"),
        "history_context_proof": run["history_context_proof"],
        "freeze_vision": args.get("freeze_vision"),
        "validation_precision": run["validation_precision"],
        "device": run["device"],
        "validation_batch_size": args.get("batch_size"),
        "history_chunk_size": args.get("history_chunk_size"),
        "channels_last": args.get("channels_last"),
        "cache_enabled": args.get("cache_frozen_features", False),
        "tf32": run["tf32"],
        "weights": {"plan": args.get("w_plan"), "steer": args.get("w_steer"), "longitudinal": args.get("w_long")},
    }
    if version >= 3:
        expected_identity["weights"]["consistency"] = args["w_consistency"]
    if any(ci.get(k) != v for k, v in expected_identity.items()):
        raise ValueError("calibration numerical configuration mismatch")
    zero = read(bind(training / "baseline_validation.json"))
    if {
        k: v
        for k, v in zero.items()
        if k not in ("step", "checkpoint", "checkpoint_sha256", "selected_sha256", "saved_state_sha256")
    } != cal["validation"]:
        raise ValueError("step-zero metrics differ from calibration")
    history = [
        json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        for line in bind(training / "history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if [row.get("step") for row in history] != list(range(1, 501)):
        raise ValueError("history must contain all 500 consecutive updates")
    if {row["step"] for row in history if "validation" in row} != {1, 100, 200, 300, 400, 500}:
        raise ValueError("validation schedule mismatch")
    for row in history:
        if row.get("action_target_contract") != target_contract:
            raise ValueError("history action target contract mismatch")
        if (
            row.get("selection_protocol_sha256") != protocol_sha
            or row.get("calibration_sha256") != p["calibration_sha256"]
        ):
            raise ValueError("history protocol or calibration mismatch")
        if row.get("controller_contract") != p["controller_contract"] or row.get("recovery_provenance") != {
            "provenance_mode": provenance_mode,
            "collection_policy_checkpoint_sha": collection_sha,
            "optimization_init_sha256": initializer,
        }:
            raise ValueError("history controller or sampler mismatch")
    init_info = checkpoint_info(args["init"])
    import torch

    init_model = torch.load(args["init"], map_location="cpu", weights_only=False)["model"]
    names = run.get("trainable_tensors")
    if (
        not isinstance(names, list)
        or names != policy_trainable_names(args["init"])
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n.startswith("base.policy.") or n not in init_model for n in names)
        or len(init_model) != 417
    ):
        raise ValueError("invalid complete model or trainable tensor inventory")
    if version == 4:
        from fsop.train.lab_train import FINAL_PLAN_SHAPES, state_hashes

        optimized_names = list(FINAL_PLAN_SHAPES)
        if (run.get("optimized_tensors") != optimized_names or cal_run.get("optimized_tensors") != optimized_names
                or any(name not in init_model or tuple(init_model[name].shape) != shape
                       for name, shape in FINAL_PLAN_SHAPES.items())
                or run.get("non_head_state_sha256") != state_hashes(init_model, FINAL_PLAN_SHAPES)
                or cal_run.get("non_head_state_sha256") != run["non_head_state_sha256"]
                or run.get("optimized_parameters") != 254430 or cal_run.get("optimized_parameters") != 254430
                or run.get("optimizer_learning_rate") != args["lr"] * args["lr_policy"]
                or cal_run.get("optimizer_learning_rate") != run["optimizer_learning_rate"]):
            raise ValueError("final plan optimizer inventory or frozen state mismatch")
        names = optimized_names
    cal_zero = read(bind(calibration / "baseline_validation.json"))
    cal_ckpt = bind(run_checkpoint(calibration, cal_run, cal_zero["checkpoint"], "step_000000.pt"))
    cal_info = checkpoint_info(cal_ckpt)
    cal_payload = torch.load(cal_ckpt, map_location="cpu", weights_only=False)
    if (
        {
            k: v
            for k, v in cal_zero.items()
            if k not in ("step", "checkpoint", "checkpoint_sha256", "selected_sha256", "saved_state_sha256")
        }
        != cal["validation"]
        or cal_zero.get("step") != 0
        or cal_zero.get("checkpoint_sha256") != sha(cal_ckpt)
        or cal_info["step"] != 0
        or cal_info["state_sha256"] != init_info["state_sha256"]
        or cal_info["signature"] != init_info["signature"]
        or cal_payload.get("config") != cal_run["args"]
        or cal_zero.get("saved_state_sha256") != cal_run["baseline_state_sha256"]
        or frozen_state(cal_ckpt, 0, cal_run["args"]) != cal_run["baseline_state_sha256"]
        or cal_payload.get("controller_contract") != p["controller_contract"]
        or cal_payload.get("recovery_provenance") != cal_run["recovery_sampling"]
        or cal_payload.get("action_target_contract") != target_contract
    ):
        raise ValueError("calibration step-zero checkpoint or metrics mismatch")
    if version == 4:
        verify_final_plan_optimizer(cal_payload, 0, run["optimizer_learning_rate"])
    reference = run["baseline_state_sha256"]
    if frozen_state(args["init"], None) != reference:
        raise ValueError("initializer frozen tensor mismatch")
    rows = [
        {
            "step": 0,
            "checkpoint": zero["checkpoint"],
            "checkpoint_sha256": zero["checkpoint_sha256"],
            "state_sha256": zero["saved_state_sha256"],
            "validation": zero,
        },
        *(r for r in history if r["step"] in [1, *STEPS[1:]]),
    ]
    candidates = []
    excluded_step_one = None
    for row in rows:
        step = row["step"]
        ckpt = bind(run_checkpoint(training, run, row["checkpoint"], f"step_{step:06d}.pt"))
        if step:
            sampling = row.get("sampling", {})
            counts_path = bind(
                run_checkpoint(training, run, sampling["counts_path"], f"sampling_counts_{step:06d}.json")
            )
            counts = read(counts_path)
            if (
                sha(counts_path) != sampling.get("counts_sha256")
                or sampling.get("total_draws") != 128 * step
                or sampling.get("recovery_draws") != 16 * step
                or counts.get("total_draws") != 128 * step
                or counts.get("recovery_draws") != 16 * step
                or sum(anchor["draws"] for ep in counts["episodes"] for anchor in ep["anchors"]) != 128 * step
            ):
                raise ValueError(f"sampling counts mismatch at step {step}")
        if inputs[str(ckpt)] != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch at step {step}")
        info = checkpoint_info(ckpt)
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if (
            info["step"] != step
            or info["signature"] != init_info["signature"]
            or payload.get("config") != args
            or payload.get("controller_contract") != p["controller_contract"]
            or payload.get("action_target_contract") != target_contract
            or payload.get("selection_protocol_sha256") != protocol_sha
            or payload.get("calibration_sha256") != p["calibration_sha256"]
            or payload.get("recovery_provenance") != run["recovery_sampling"]
            or row["state_sha256"] != reference
            or frozen_state(ckpt, step, args) != reference
        ):
            raise ValueError(f"checkpoint state, contract, or signature mismatch at step {step}")
        if version == 4:
            verify_final_plan_optimizer(payload, step, run["optimizer_learning_rate"])
        if version == 4 and state_hashes(payload["model"], FINAL_PLAN_SHAPES) != run["non_head_state_sha256"]:
            raise ValueError(f"non-head state changed at step {step}")
        if any(
            not torch.equal(tensor, payload["model"][name]) for name, tensor in init_model.items() if name not in names
        ):
            raise ValueError(f"non-trainable tensor changed at step {step}")
        if step == 0 and info["state_sha256"] != init_info["state_sha256"]:
            raise ValueError("step zero differs from initializer model")
        changed = info["groups"]["policy"]["state_sha256"] != init_info["groups"]["policy"]["state_sha256"]
        current_validation = row["validation"]
        if current_validation.get("action_target_contract") != target_contract:
            raise ValueError(f"validation action target contract mismatch at step {step}")
        expected_targets = {r["path"]: r["target_coherence"] for r in cal["validation"]["episodes"]}
        actual_targets = {r["path"]: r.get("target_coherence") for r in current_validation["episodes"]}
        if (current_validation.get("target_coherence") != cal["validation"]["target_coherence"]
                or actual_targets != expected_targets):
            raise ValueError(f"native target coherence changed at step {step}")
        gates = assess(
            current_validation, p["bounds"], run["validation_selection"], identities[training]["path_to_case"]
        )
        if version >= 2:
            metrics = ("teacher_steer", "teacher_longitudinal") if version == 2 else ("native_relative_acceleration",)
            for metric in metrics:
                groups = [current_validation["aggregate"], *current_validation["episodes"]]
                for group in groups:
                    number(group.get(metric), metric)
                if not math.isclose(groups[0][metric], sum(g[metric] for g in groups[1:]) / len(groups[1:]),
                                    rel_tol=1e-7, abs_tol=1e-12):
                    raise ValueError(f"inconsistent aggregate {metric}")
        gates.append({"metric": "policy_changed", "actual": changed, "limit": True, "passed": step == 0 or changed})
        candidate = {
            "step": step,
            "checkpoint": str(ckpt),
            "checkpoint_sha256": inputs[str(ckpt)],
            "model_state_sha256": info["state_sha256"],
            "policy_changed": changed,
            "validation": row["validation"],
            "gates": gates,
            "eligible": all(g["passed"] for g in gates),
            "rejection_reasons": [g["metric"] for g in gates if not g["passed"]],
        }
        if step == 1:
            excluded_step_one = candidate
        else:
            candidates.append(candidate)
    ranking_metric = "plan_l1" if version >= 3 else "steer"
    baseline_metric = candidates[0]["validation"]["aggregate"][ranking_metric]
    for c in candidates:
        c["improves_step_zero"] = c["step"] != 0 and c["validation"]["aggregate"][ranking_metric] < baseline_metric
    improved = [c for c in candidates if c["eligible"] and c["improves_step_zero"]]
    chosen = (
        min(improved, key=lambda c: (c["validation"]["aggregate"][ranking_metric], c["step"])) if improved else candidates[0]
    )
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError(f"input changed during selection: {path}")
    return {
        "schema_version": p["schema_version"],
        "kind": p["kind"],
        "status": "trained_candidate_selected" if improved else "retain_step_zero",
        "controller_contract": p["controller_contract"],
        **({"action_target_contract": target_contract} if target_contract is not None else {}),
        "protocol": p,
        "input_sha256": inputs,
        "validation_fingerprint": identities[training]["fingerprint"],
        "initializer": init_info,
        "calibration": cal,
        "candidates": candidates,
        "excluded_step_one": excluded_step_one,
        "selected": {k: chosen[k] for k in ("step", "checkpoint", "checkpoint_sha256")},
        "trained_candidate": bool(improved),
        "selected_eligible": chosen["eligible"],
        "selected_rejection_reasons": chosen["rejection_reasons"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "calibration", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = select(args.training, args.calibration, args.protocol)
    with args.out.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result["selected"]))


if __name__ == "__main__":
    main()
