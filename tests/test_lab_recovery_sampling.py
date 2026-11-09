import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fsop.train.lab_train import RecoverySampler

CHECKPOINT = "a" * 64


def episode(case_id, split="train", takeover=None, anchors=None):
    meta = {"id": case_id, "split": split}
    teacher = np.ones(300, dtype=bool)
    if takeover is not None:
        teacher[:takeover] = False
        meta.update(
            {
                "collection_policy": {"mode": "learner_prefix", "actual_takeover_tick": takeover},
                "checkpoint_sha256": CHECKPOINT,
                "control_hz": 20,
                "terminal_continuation": {"start_tick": 100, "end_tick": 300, "contacts": 0},
            }
        )
    return SimpleNamespace(
        meta=meta, path=Path(case_id), teacher=teacher, ticks=np.arange(40) * 4, anchors=anchors or list(range(40))
    )


class ToyReplay(SimpleNamespace):
    def __len__(self):
        return len(self.index)


def seal(dataset, recipe, gates):
    from fsop.data.native_replay import ARRAY_NAMES

    gates.mkdir(exist_ok=True)
    for ep in dataset.episodes:
        ep.path = gates.parent / ep.meta["id"]
        ep.path.mkdir(exist_ok=True)
        (ep.path / "source.zip").write_bytes(b"source")
        ep.meta["source_archive_sha256"] = hashlib.sha256(b"source").hexdigest()
        ep.meta["arrays_sha256"] = {}
        for name in ARRAY_NAMES:
            content = name.encode()
            (ep.path / f"{name}.npy").write_bytes(content)
            ep.meta["arrays_sha256"][f"{name}.npy"] = hashlib.sha256(content).hexdigest()
        meta = json.dumps(ep.meta).encode()
        (ep.path / "meta.json").write_bytes(meta)
        if ep.meta.get("collection_policy", {}).get("mode") != "learner_prefix":
            continue
        tick = ep.meta["collection_policy"]["actual_takeover_tick"]
        terminal = ep.meta["terminal_continuation"]["start_tick"]
        frames = [frame for ei, frame in dataset.index if dataset.episodes[ei] is ep
                  and tick <= int(ep.ticks[frame]) < min(tick + 40, terminal)]
        receipt = {"schema_version": 1, "passed": True, "reasons": [], "case_id": ep.meta["id"],
                   "episode": str(ep.path.resolve()), "recipe": str(recipe.resolve()),
                   "hashes": {"recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
                              "meta_sha256": hashlib.sha256(meta).hexdigest(),
                              "arrays_sha256": ep.meta["arrays_sha256"],
                              "source_archive_sha256": ep.meta["source_archive_sha256"]},
                   "takeover_tick": tick, "terminal_start_tick": terminal,
                   "recovery_frame_indices": frames, "recovery_ticks": [int(ep.ticks[f]) for f in frames]}
        (gates / f"{ep.meta['id']}.json").write_text(json.dumps(receipt))


def sampler_fixture(tmp_path):
    eps = [
        episode("a", takeover=5),
        episode("b", takeover=9, anchors=list(range(3, 11)) + [20]),
        episode("ordinary"),
        episode("val", split="val", takeover=5),
    ]
    index = [
        (ei, frame)
        for ei, ep in enumerate(eps)
        if ep.meta["split"] == "train"
        for frame in ep.anchors
        if ep.teacher[int(ep.ticks[frame])]
    ]
    dataset = ToyReplay(episodes=eps, index=index)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"recovery_case_ids": ["b", "a"], "checkpoint_sha256": CHECKPOINT}))
    gates = tmp_path / "gates"
    seal(dataset, recipe, gates)
    return dataset, recipe, gates


def test_draws_exact_partition_uniform_episode_and_repeatable(tmp_path):
    dataset, recipe, gates = sampler_fixture(tmp_path)
    first = RecoverySampler(dataset, recipe, gates, 128, 16, CHECKPOINT)
    second = RecoverySampler(dataset, recipe, gates, 128, 16, CHECKPOINT)
    rng, same_rng = np.random.default_rng(0), np.random.default_rng(0)
    for _ in range(1000):
        batch = first.draw(rng)
        assert np.array_equal(batch, second.draw(same_rng))
        assert sum(int(i) in first.recovery_indices for i in batch) == 16
        assert all(dataset.episodes[dataset.index[int(i)][0]].meta["split"] == "train" for i in batch)
    receipt = first.protocol()
    assert [g["case_id"] for g in receipt["recovery_episodes"]] == ["b", "a"]
    assert [len(g["frames"]) for g in receipt["recovery_episodes"]] == [8, 10]
    assert all(
        g["takeover_tick"] <= tick < g["takeover_tick"] + 40
        for g in receipt["recovery_episodes"]
        for tick in g["ticks"]
    )
    draws = first.draw_counts()
    assert draws["total_draws"] == 128000
    assert draws["recovery_draws"] == 16000
    group_draws = [sum(int(first.counts[i]) for i in group["indices"]) for group in first.groups]
    assert abs(group_draws[0] - 8000) < 300
    assert abs(group_draws[1] - 8000) < 300
    assert sum(row["draws"] for ep in draws["episodes"] for row in ep["anchors"]) == 128000


@pytest.mark.parametrize(
    "fault",
    ["missing_case", "extra_case", "wrong_checkpoint", "teacher_gap", "handoff_mismatch", "short_window", "wrong_rate"],
)
def test_invalid_recovery_pool_fails_closed(tmp_path, fault):
    dataset, recipe, gates = sampler_fixture(tmp_path)
    document = json.loads(recipe.read_text())
    if fault == "missing_case":
        document["recovery_case_ids"] = ["a"]
    elif fault == "extra_case":
        document["recovery_case_ids"].append("missing")
    elif fault == "wrong_checkpoint":
        document["checkpoint_sha256"] = "b" * 64
    elif fault == "teacher_gap":
        dataset.episodes[0].teacher[80] = False
    elif fault == "handoff_mismatch":
        dataset.episodes[0].meta["collection_policy"]["actual_takeover_tick"] = 4
    elif fault == "short_window":
        dataset.index = [(ei, k) for ei, k in dataset.index if ei != 1 or k >= 10]
    elif fault == "wrong_rate":
        dataset.episodes[0].meta["control_hz"] = 5
    recipe.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        RecoverySampler(dataset, recipe, gates, 128, 16, CHECKPOINT)


def test_invalid_batch_and_empty_other_pool_fail(tmp_path):
    dataset, recipe, gates = sampler_fixture(tmp_path)
    with pytest.raises(ValueError, match="recovery draws"):
        RecoverySampler(dataset, recipe, gates, 128, 128, CHECKPOINT)
    valid = RecoverySampler(dataset, recipe, gates, 128, 16, CHECKPOINT)
    dataset.index = [row for i, row in enumerate(dataset.index) if i in valid.recovery_indices]
    with pytest.raises(ValueError, match="non-recovery"):
        RecoverySampler(dataset, recipe, gates, 128, 16, CHECKPOINT)


def test_training_keeps_step_zero_and_records_draws(tmp_path, monkeypatch):
    import torch

    from fsop.train import lab_train

    dataset, recipe, gates = sampler_fixture(tmp_path)
    init = tmp_path / "init.pt"
    init.write_bytes(b"initial checkpoint")
    digest = hashlib.sha256(init.read_bytes()).hexdigest()
    document = json.loads(recipe.read_text())
    document["checkpoint_sha256"] = digest
    recipe.write_text(json.dumps(document))
    for ep in dataset.episodes:
        ep.rejections = {}
        if ep.meta.get("checkpoint_sha256"):
            ep.meta["checkpoint_sha256"] = digest
    seal(dataset, recipe, gates)

    class Replay:
        def __init__(self, episodes, split):
            self.episodes = episodes
            self.split = split
            self.index = (
                dataset.index
                if split == "train"
                else [(i, frame) for i, ep in enumerate(episodes) if ep.meta["split"] == "val" for frame in ep.anchors]
            )

        def __len__(self):
            return len(self.index)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Module()
            self.base.vision = torch.nn.Linear(1, 1, bias=False)
            self.base.policy = torch.nn.Linear(1, 1, bias=False)
            self.base.policy.weight.data.zero_()
            self.bev = self.img = None

        def set_trunk_trainable(self, mode):
            self.base.vision.requires_grad_(True)

    def loss(model, dataset, indices, device, *args, **kwargs):
        value = (model.base.policy.weight - 1).square().sum()
        return value, {"plan_nll": value, "plan_l1": value, "steer": value, "longitudinal": value}

    monkeypatch.setattr(lab_train, "episode_paths", lambda **kwargs: [])
    monkeypatch.setattr(lab_train, "load_episodes", lambda paths: dataset.episodes)
    monkeypatch.setattr(lab_train, "NativeReplay", Replay)
    monkeypatch.setattr(lab_train, "load_fs", lambda *args, **kwargs: Model())
    monkeypatch.setattr(lab_train, "batch_loss", loss)
    monkeypatch.setattr(lab_train, "telemetry", lambda: None)
    out = tmp_path / "run"
    args = SimpleNamespace(
        manifest=None,
        data="unused",
        init=str(init),
        out=str(out),
        steps=2,
        batch_size=32,
        recovery_cases=str(recipe),
        recovery_gates=str(gates),
        recovery_draws=16,
        lr=0.1,
        val_every=1,
        val_samples=4,
        history_chunk_size=64,
        w_steer=1.0,
        w_long=1.0,
        lr_policy=1.0,
        lr_trunk=1.0,
        l2sp=0.0,
        grad_clip=1.0,
        device="cpu",
        amp=False,
        channels_last=False,
        threads=1,
        seed=0,
        history_context=24,
        dataset_verification=None,
        w_plan=1.0,
        freeze_vision=True,
    )
    lab_train.run(args)
    baseline = json.loads((out / "baseline_validation.json").read_text())
    initial = torch.load(out / "step_000000.pt", weights_only=False)
    selected = torch.load(out / "selected.pt", weights_only=False)
    assert initial["step"] == 0 and selected["step"] == 2
    assert baseline["checkpoint_sha256"] == lab_train.sha256(out / "step_000000.pt")
    rows = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    for step, row in enumerate(rows, 1):
        sampling = row["sampling"]
        counts = Path(sampling["counts_path"])
        assert sampling["total_draws"] == step * 32
        assert sampling["recovery_draws"] == step * 16
        assert sampling["counts_sha256"] == lab_train.sha256(counts)
        assert json.loads(counts.read_text())["total_draws"] == step * 32


def test_terminal_anchor_is_excluded(tmp_path):
    dataset, recipe, gates = sampler_fixture(tmp_path)
    ep = dataset.episodes[0]
    ep.teacher = np.r_[np.zeros(160, dtype=bool), np.ones(232, dtype=bool)]
    ep.ticks = np.arange(60) * 4
    ep.anchors = list(range(40, 51))
    ep.meta["collection_policy"]["actual_takeover_tick"] = 160
    ep.meta["terminal_continuation"] = {"start_tick": 192, "end_tick": 392, "contacts": 0}
    dataset.index = [(ei, frame) for ei, frame in dataset.index if ei != 0]
    dataset.index.extend((0, frame) for frame in ep.anchors)
    seal(dataset, recipe, gates)
    sampler = RecoverySampler(dataset, recipe, gates, 128, 16, CHECKPOINT)
    group = next(g for g in sampler.protocol()["recovery_episodes"] if g["case_id"] == "a")
    assert group["ticks"] == list(range(160, 192, 4))
    assert group["gate_sha256"] == hashlib.sha256((gates / "a.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("fault", ["missing", "failed", "recipe", "meta", "arrays", "source", "frames", "terminal"])
def test_gate_receipt_and_episode_drift_fail_closed(tmp_path, fault):
    dataset, recipe, gates = sampler_fixture(tmp_path)
    path = gates / "a.json"
    receipt = json.loads(path.read_text())
    if fault == "missing":
        path.unlink()
    elif fault == "failed":
        receipt["passed"] = False
    elif fault == "recipe":
        receipt["hashes"]["recipe_sha256"] = "0" * 64
    elif fault == "meta":
        (dataset.episodes[0].path / "meta.json").write_text("{}")
    elif fault == "arrays":
        (dataset.episodes[0].path / "ticks.npy").write_bytes(b"drift")
    elif fault == "source":
        (dataset.episodes[0].path / "source.zip").write_bytes(b"drift")
    elif fault == "frames":
        receipt["recovery_frame_indices"].pop()
    elif fault == "terminal":
        receipt["terminal_start_tick"] -= 1
    if fault != "missing":
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="invalid recovery gate"):
        RecoverySampler(dataset, recipe, gates, 128, 16, CHECKPOINT)


def test_missing_gates_and_invalid_terminal_fail(tmp_path):
    dataset, recipe, gates = sampler_fixture(tmp_path)
    with pytest.raises(ValueError, match="gates directory"):
        RecoverySampler(dataset, recipe, None, 128, 16, CHECKPOINT)
    dataset.episodes[0].meta["terminal_continuation"]["start_tick"] = 5
    with pytest.raises(ValueError, match="invalid terminal"):
        RecoverySampler(dataset, recipe, gates, 128, 16, CHECKPOINT)
