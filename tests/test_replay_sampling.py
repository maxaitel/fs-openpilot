from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler

from fsop.data.sim_dataset import SimDataset
from fsop.train import trainer


class ToyDataset(SimDataset):
    def __init__(self, *args):
        self.eps = [SimpleNamespace(path=Path('recovery_086000_skidpad')), SimpleNamespace(path=Path('original_track'))]
        self.index = [(0, k) for k in range(2)] + [(1, k) for k in range(6)]

    def __getitem__(self, i):
        return {'sample': torch.tensor(i)}


class TinyModel(nn.Module):
    def __init__(self, image_head):
        super().__init__()
        self.base = nn.Module()
        self.base.vision = nn.Linear(2, 2)
        self.base.policy = nn.Linear(2, 2)
        self.bev = nn.Linear(2, 2)
        self.img = None

    def set_trunk_trainable(self, mode):
        self.base.vision.requires_grad_(mode != 'frozen')


@pytest.fixture
def make_trainer(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer, 'SimDataset', ToyDataset)
    monkeypatch.setattr(trainer, 'SupercomboFS', TinyModel)
    runs = []

    def make(weights):
        cfg = trainer.TrainCfg(out_dir=str(tmp_path), name=str(len(runs)), device='cpu', num_workers=0,
                               batch_size=4, episode_prefix_weights=weights)
        run = trainer.Trainer(cfg)
        runs.append(run)
        return run

    yield make
    for run in runs:
        run.tb.close()


def test_default_preserves_shuffled_coverage_and_validation_order(make_trainer):
    run = make_trainer({})
    loader = run._loader(run.train_ds, True)
    assert isinstance(loader.sampler, RandomSampler)
    assert sorted(loader.sampler) == list(range(8))
    assert loader.drop_last
    validation = run._loader(run.val_ds, False)
    assert isinstance(validation.sampler, SequentialSampler)
    assert list(validation.sampler) == list(range(8))


def test_recovery_share_is_weighted_reproducibly_without_affecting_fsoco(make_trainer):
    first = make_trainer({'recovery_': 4.0})
    second = make_trainer({'recovery_': 4.0})
    a = first._loader(first.train_ds, True).sampler
    b = second._loader(second.train_ds, True).sampler
    assert isinstance(a, WeightedRandomSampler)
    a.num_samples = b.num_samples = 20000
    draw = list(a)
    assert draw == list(b)
    assert abs(np.mean(np.array(draw) < 2) - 4 / 7) < 0.02
    assert draw != list(a)
    assert isinstance(first._loader(ToyDataset(), True).sampler, RandomSampler)
    assert isinstance(first._loader(first.val_ds, False).sampler, SequentialSampler)


@pytest.mark.parametrize('weights', [
    {'recovery_': 0}, {'recovery_': -1}, {'recovery_': float('nan')}, {'recovery_': float('inf')},
    {'': 2}, {'missing_': 2}, {'recovery_': 2, 'recovery_086': 3},
])
def test_invalid_or_ambiguous_weights_fail_before_training(weights):
    with pytest.raises(ValueError):
        ToyDataset().sample_weights(weights)


def test_dataset_resume_rejects_mismatched_episode_sources(tmp_path):
    from scripts.prepare_skidpad_recovery_training import link_episode

    first, second, target = (tmp_path / name for name in ('first', 'second', 'episode'))
    first.mkdir()
    second.mkdir()
    link_episode(target, first)
    link_episode(target, first)
    with pytest.raises(ValueError, match='different source'):
        link_episode(target, second)
    assert target.resolve() == first
