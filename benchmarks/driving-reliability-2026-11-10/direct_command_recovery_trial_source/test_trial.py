import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

path = Path(__file__).with_name('trial.py')
spec = importlib.util.spec_from_file_location('recovery_trial', path)
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def sampler_fixture():
    episodes = [SimpleNamespace(meta={'id': x, 'split': 'train'}) for x in ('old', 'skidpad', 'trackdrive')]
    index = [(0, i) for i in range(40)] + [(1, i) for i in range(8)] + [(2, i) for i in range(8)]
    class Dataset:
        def __init__(self): self.index, self.episodes = index, episodes
        def __len__(self): return len(self.index)
    legacy = SimpleNamespace(groups=[{'indices': list(range(8))}, {'indices': list(range(8, 16))}],
                             recovery_indices=set(range(16)), dataset=range(40), protocol=lambda: {})
    groups = [{'case_id': 'skidpad', 'indices': list(range(40, 48))},
              {'case_id': 'trackdrive', 'indices': list(range(48, 56))}]
    return trial.TrialSampler(Dataset(), legacy, groups)


def test_draw_contract_counts_wiring_and_case_sampling():
    sampler = sampler_fixture()
    rng = np.random.default_rng(0)
    case_draws = 0
    for _ in range(2000):
        batch = sampler.draw(rng)
        assert len(batch) == 128
        assert sum(int(i) in sampler.old_indices for i in batch) == 16
        assert sum(int(i) in sampler.new_indices for i in batch) == 16
        case_draws += sum(40 <= int(i) < 48 for i in batch)
    counts = sampler.draw_counts()
    assert (counts['old_recovery_draws'], counts['new_recovery_draws'], counts['ordinary_draws']) == (32000, 32000, 192000)
    assert 15500 < case_draws < 16500
    anchors = trial.wiring_indices(sampler)
    assert len(anchors) == len(set(anchors)) == 32
    assert sum(16 <= i < 40 for i in anchors) == 16
    assert sum(40 <= i < 48 for i in anchors) == 8
    assert sum(48 <= i < 56 for i in anchors) == 8


def test_membership_rejects_changed_original_case(tmp_path):
    recipe = json.loads(trial.RECIPE.read_text())
    result = trial.membership(recipe)
    assert result['train_cases'] == 34
    altered = json.loads(Path(recipe['new_membership']).read_text())
    altered['cases'][0]['seed'] += 1
    path = tmp_path / 'membership.json'
    path.write_text(json.dumps(altered))
    recipe['new_membership'] = str(path)
    with pytest.raises(ValueError): trial.membership(recipe)


def test_receipt_rejects_failed_or_changed_replay(tmp_path):
    episode = tmp_path / 'episode'
    episode.mkdir()
    receipt = tmp_path / 'receipt.json'
    passed = {'passed': True, 'reasons': [], 'episode': str(episode), 'hashes': {'meta_sha256': 'a'}}
    receipt.write_text(json.dumps(passed))
    with patch.object(trial.verify, 'verify', return_value=passed):
        assert trial.verified_receipt(episode, receipt, 'checkpoint', 'cpu')['report'] == passed
    with patch.object(trial.verify, 'verify', return_value={**passed, 'hashes': {'meta_sha256': 'b'}}):
        with pytest.raises(ValueError): trial.verified_receipt(episode, receipt, 'checkpoint', 'cpu')
    receipt.write_text(json.dumps({**passed, 'passed': False}))
    with pytest.raises(ValueError): trial.verified_receipt(episode, receipt, 'checkpoint', 'cpu')
    with pytest.raises(ValueError): trial.verified_receipt(episode, episode / 'receipt.json', 'checkpoint', 'cpu')
