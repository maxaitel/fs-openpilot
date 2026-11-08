import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fsop.train import trainer


def test_audit_requires_commands_at_both_circles():
    path = Path(__file__).parents[1] / "scripts/prepare_skidpad_training.py"
    spec = importlib.util.spec_from_file_location("prepare", path)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    track = prepare.skidpad()
    schedule = prepare.RouteCommands.for_mission(track, prepare.Mission(track, prepare.ExpertParams()), 20)
    states = np.zeros((5, 12))
    commands = np.zeros((5, 8))
    for i, (at_s, channel) in enumerate(schedule.decisions):
        states[i, 9] = at_s
        commands[i, channel] = 1
    assert prepare.junction_coverage(states, commands) == [1] * 5
    for second_circle in (1, 3):
        missing = commands.copy()
        missing[second_circle] = 0
        assert np.all(missing.sum(0)[[1, 2, 7]] > 0)
        assert prepare.junction_coverage(states, missing)[second_circle] == 0


def test_cached_resume_preserves_unused_image_head(monkeypatch, tmp_path):
    class Model(nn.Module):
        def __init__(self, image_head):
            super().__init__()
            self.base = nn.Module()
            self.base.vision = nn.Linear(2, 2)
            self.base.policy = nn.Linear(2, 2)
            self.bev = nn.Linear(2, 2)
            self.img = nn.Linear(2, 2) if image_head else None

        def set_trunk_trainable(self, mode):
            self.base.vision.requires_grad_(mode != "frozen")

    class Dataset:
        eps = [None]

        def __init__(self, *args):
            pass

        def __len__(self):
            return 1

    original = Model(True).state_dict()
    checkpoint = tmp_path / "full.pt"
    torch.save({"model": original}, checkpoint)
    monkeypatch.setattr(trainer, "SupercomboFS", Model)
    monkeypatch.setattr(trainer, "SimDataset", Dataset)
    run = trainer.Trainer(trainer.TrainCfg(stage=1, init_ckpt=str(checkpoint), out_dir=str(tmp_path), device="cpu"))
    try:
        assert run.model.img is not None
        assert all(not p.requires_grad for p in run.model.img.parameters())
        optimized = {id(p) for group in run.opt.param_groups for p in group["params"]}
        assert not optimized.intersection(id(p) for p in run.model.img.parameters())
        saved = torch.load(run.save("retained"), weights_only=False)["model"]
        assert saved.keys() == original.keys()
        for key in original:
            assert torch.equal(original[key], saved[key]), key
    finally:
        run.tb.close()
