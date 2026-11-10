"""Verify the completed first-step probe in a separate process using NumPy."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


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
p.add_argument('--control', type=Path, required=True)
p.add_argument('--script', type=Path, required=True)
p.add_argument('--pid', type=int, required=True)
a = p.parse_args()
try:
    os.kill(a.pid, 0)
except ProcessLookupError:
    pass
else:
    raise AssertionError('run process is still alive or pid was reused')
assert (a.out / 'exit_code.txt').read_text().strip() == '0'
protocol = json.loads((a.out / 'protocol.json').read_text())
report = json.loads((a.out / 'report.json').read_text())
rows = [json.loads(x) for x in (a.out / 'rows.jsonl').read_text().splitlines()]
arrays = np.load(a.out / 'arrays.npz', allow_pickle=False)
old_arrays = np.load(a.control / 'arrays.npz', allow_pickle=False)
assert digest(a.script) == report['script_sha256'] == protocol['script_sha256']
assert digest(__file__) == protocol['verifier_sha256']
for name, expected in protocol['control_sha256'].items():
    assert digest(a.control / name) == expected
for name in ('protocol', 'rows', 'arrays', 'baseline_receipt'):
    suffix = {'protocol': 'json', 'rows': 'jsonl', 'arrays': 'npz', 'baseline_receipt': 'json'}[name]
    assert digest(a.out / f'{name}.{suffix}') == report[name + '_sha256']
assert json.loads((a.out / 'baseline_receipt.json').read_text())['passed']
assert digest(protocol['config_path']) == protocol['config_sha256']
assert digest(protocol['model']['path']) == protocol['model']['sha256']
assert digest('/work/data/native/lab_v7/manifest.json') == protocol['manifest_sha256']
for path, expected in protocol['source_sha256'].items():
    assert digest(Path('/work') / path) == expected
assert len(rows) == report['updates'] == 12
assert {(r['batch'], r['rate']) for r in rows} == {(i, lr) for i in range(4) for lr in protocol['rates']}
assert len({x['index'] for x in protocol['monitor']}) == 16
excluded = {x['index'] for x in protocol['monitor']}
recovery = {i for group in protocol['sampler']['recovery_episodes'] for i in group['indices']}
assert all(len(ids) == 128 and not set(ids) & excluded and sum(i in recovery for i in ids) == 16 for ids in protocol['batches'])
for key in arrays.files:
    assert np.isfinite(arrays[key]).all(), key
for new_prefix, old_prefix in [('monitor_before', 'latest_monitor_before')] + [(f'b{i}_fit_before', f'latest_{i}_fit_before') for i in range(4)]:
    for key in ('mu', 'logstd', 'native_target', 'observed', 'wheelbase', 'desired_motion', 'native_desired_motion'):
        np.testing.assert_allclose(arrays[new_prefix + '_' + key], old_arrays[old_prefix + '_' + key], atol=1e-7, rtol=1e-6)

names = ('steer_rad', 'raw_accel_mps2', 'target_speed_mps', 'tangent_accel_mps2')
phases = ('steady_right', 'right_to_left', 'steady_left', 'recovery')
def verify_motion(prefix, metrics, is_monitor):
    predicted = motion(arrays[prefix + '_mu'], arrays[prefix + '_observed'], arrays[prefix + '_wheelbase'])
    native = motion(arrays[prefix + '_native_target'], arrays[prefix + '_observed'], arrays[prefix + '_wheelbase'])
    np.testing.assert_allclose(predicted, arrays[prefix + '_desired_motion'], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(native, arrays[prefix + '_native_desired_motion'], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.mean(np.abs(predicted-native), axis=0), [metrics['action_mae'][k] for k in names], rtol=2e-5, atol=2e-5)
    if is_monitor:
        for phase in phases:
            idx = [i for i, x in enumerate(protocol['monitor']) if x['phase'] == phase]
            assert len(idx) == 4
            np.testing.assert_allclose(np.mean(np.abs(predicted[idx]-native[idx]), axis=0), [metrics['phases'][phase][k] for k in names], rtol=2e-5, atol=2e-5)

verify_motion('monitor_before', report['monitor_before'], True)
for row in rows:
    assert row['initial_model_digest'] == report['initial_digest'] != row['after_model_digest']
    assert row['frozen_state_unchanged'] and row['optimizer_steps'] == [1]
    assert row['eligible_policy_tensors'] == 50 and row['policy_tensors_with_grad'] == row['optimizer_state_count'] == 42
    assert row['optimizer_group_lrs'] == [row['rate']]
    prefix = f"b{row['batch']}_lr{protocol['rates'].index(row['rate'])}"
    verify_motion(prefix + '_fit_after', row['fit_after'], False)
    verify_motion(prefix + '_monitor_after', row['monitor_after'], True)

before = report['monitor_before']
classification = {}
for rate in protocol['rates']:
    cells = sorted((r for r in rows if r['rate'] == rate), key=lambda x: x['batch'])
    improvements = [r['objective_after'] < r['objective_before'] for r in cells]
    margins = [r['objective_before'] - r['objective_after'] for r in cells]
    threshold = [1e-7 + 1e-6 * abs(r['objective_before']) for r in cells]
    monitor_improves = all(r['monitor_after']['action_mae'][k] < before['action_mae'][k]
                           for r in cells for k in names[:3])
    classification[str(rate)] = {'fit_decreases_all_four': all(improvements), 'fit_decreases_by_batch': improvements,
                                 'fit_decrease_margins': margins, 'reference_tolerances': threshold,
                                 'all_margins_exceed_reference_tolerance': all(d > t for d, t in zip(margins, threshold)),
                                 'all_three_monitor_maes_decrease_in_all_four': monitor_improves,
                                 'monitor_by_batch': [{'batch': r['batch'], 'aggregate': r['monitor_after']['action_mae'],
                                                       'phases': r['monitor_after']['phases']} for r in cells]}
assert sum(path.stat().st_size for path in a.out.iterdir() if path.is_file()) < protocol['limits']['artifact_bytes']
assert shutil.disk_usage(a.out).free >= protocol['limits']['minimum_free_bytes']
receipt = {'passed': True, 'run_pid_closed': a.pid, 'exit_code': 0, 'updates': 12,
           'before_arrays_match_control': True, 'independent_numpy_motion_and_phase_metrics_match': True,
           'source_and_model_hashes_match': True, 'configured_tensors': 50, 'tensors_with_gradient_and_optimizer_state': 42,
           'protocol_sha256': digest(a.out / 'protocol.json'), 'report_sha256': digest(a.out / 'report.json'),
           'verifier_sha256': digest(__file__), 'classification': classification,
           'limitations': 'Optimizer execution and frozen tensor equality are asserted by the run process. Before objective and component losses are matched there against the retained v2 rows. Monitor anchors may share episodes and history or future windows with fit data. No closed-loop result follows.'}
(a.out / 'independent_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps({'passed': True, 'run_pid_closed': a.pid, 'updates': len(rows), 'classification': {k: {x: v[x] for x in ('fit_decreases_all_four', 'all_margins_exceed_reference_tolerance', 'all_three_monitor_maes_decrease_in_all_four')} for k, v in classification.items()}}), flush=True)
