"""Separate-process check of the completed throwaway diagnostic artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def motion(plan, observed, wheelbase):
    times = np.array([(i / 32.) ** 2 * 10. for i in range(33)])
    def interp(values, t):
        hi = int(np.searchsorted(times, t))
        fraction = (t - times[hi - 1]) / (times[hi] - times[hi - 1])
        return values[:, hi - 1] * (1 - fraction) + values[:, hi] * fraction
    speed_for_steer = np.maximum(observed[:, 0], 1.)
    curvature = -2 * interp(plan[:, :, 11], .2) / (speed_for_steer * .2) + plan[:, 0, 14] / speed_for_steer
    steer = np.arctan(wheelbase * curvature)
    speed = np.linalg.norm(plan[:, :, 3:5], axis=-1)
    tangent = (plan[:, :, 3:5] * plan[:, :, 6:8]).sum(-1) / np.maximum(speed, .001)
    target_speed, target_accel = interp(speed, .3), interp(tangent, .3)
    raw = 1.5 * (target_speed - np.linalg.norm(observed[:, :2], axis=-1)) + .5 * target_accel
    return np.stack((steer, raw, target_speed, target_accel), axis=-1)


p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--script', type=Path, required=True)
a = p.parse_args()
protocol = json.loads((a.out / 'protocol.json').read_text())
report = json.loads((a.out / 'report.json').read_text())
rows = [json.loads(line) for line in (a.out / 'rows.jsonl').read_text().splitlines()]
arrays = np.load(a.out / 'arrays.npz', allow_pickle=False)
assert digest(a.script) == report['script_sha256'] == protocol['script_sha256']
assert digest(a.out / 'protocol.json') == report['protocol_sha256']
assert digest(a.out / 'rows.jsonl') == report['rows_sha256']
assert digest(a.out / 'arrays.npz') == report['arrays_sha256']
assert digest(protocol['config_path']) == protocol['config_sha256']
for path, expected in protocol['source_sha256'].items():
    assert digest(Path('/work') / path) == expected
for spec in protocol['models'].values():
    assert digest(spec['path']) == spec['sha256']
assert digest('/work/data/native/lab_v7/manifest.json') == protocol['manifest_sha256']
assert len(rows) == report['updates'] == protocol['expected_updates'] == 48
expected = {(model, batch, arm) for model in protocol['models'] for batch in range(4) for arm in protocol['arms']}
assert {(row['model'], row['batch'], row['arm']) for row in rows} == expected
assert len({row['index'] for row in protocol['monitor']}) == 16
excluded = {row['index'] for row in protocol['monitor']}
recovery = {i for group in protocol['sampler']['recovery_episodes'] for i in group['indices']}
assert all(len(ids) == 128 and not set(ids) & excluded and sum(i in recovery for i in ids) == 16 for ids in protocol['batches'])
for value in (protocol, report, rows):
    finite(value)
for key in arrays.files:
    assert np.isfinite(arrays[key]).all(), key
for row in rows:
    assert row['optimizer_steps'] == [1] and row['parameter_delta_norm'] > 0
    assert row['frozen_state_unchanged'] is True
    assert row['initial_model_digest'] == report['models'][row['model']]['initial_model_digest']
    assert row['after_model_digest'] != row['initial_model_digest']
    assert report['models'][row['model']]['trainable_tensors'] == 50
    assert report['models'][row['model']]['trainable_parameters'] == 6949412
    for split, prefix in [('fit', f"{row['model']}_{row['batch']}_{row['arm']}_fit_after"),
                          ('monitor', f"{row['model']}_{row['batch']}_{row['arm']}_monitor_after")]:
        predicted = motion(arrays[prefix + '_mu'], arrays[prefix + '_observed'], arrays[prefix + '_wheelbase'])
        native = motion(arrays[prefix + '_native_target'], arrays[prefix + '_observed'], arrays[prefix + '_wheelbase'])
        np.testing.assert_allclose(predicted, arrays[prefix + '_desired_motion'], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(native, arrays[prefix + '_native_desired_motion'], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(np.mean(np.abs(predicted - native), axis=0), list(row[split + '_after']['action_mae'].values()), rtol=2e-5, atol=2e-5)
        if split == 'monitor':
            for phase in ('steady_right', 'right_to_left', 'steady_left', 'recovery'):
                indices = [i for i, anchor in enumerate(protocol['monitor']) if anchor['phase'] == phase]
                np.testing.assert_allclose(np.mean(np.abs(predicted[indices] - native[indices]), axis=0),
                                           list(row['monitor_after']['phases'][phase]['action_mae'].values()), rtol=2e-5, atol=2e-5)
assert sum(path.stat().st_size for path in a.out.iterdir() if path.is_file()) < protocol['limits']['artifact_bytes']
receipt = {'passed': True, 'updates': len(rows), 'all_arrays_finite': True, 'model_files_unchanged': True,
           'source_files_unchanged': True, 'independent_numpy_motion_and_phase_metrics_match': True,
           'protocol_sha256': digest(a.out / 'protocol.json'), 'report_sha256': digest(a.out / 'report.json'),
           'verifier_sha256': digest(__file__), 'limitations': 'Frozen tensor equality and exact optimizer calls are asserted by the run process. Only hashes and deltas remain; no updated checkpoints are saved.'}
(a.out / 'independent_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
