"""Verify closed-process head-scope outputs and retained controls without importing Torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from float64_math import reconstruct


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(error, selections):
    return {group: dict(zip(NAMES, error[ids].mean(0).tolist())) for group, ids in selections.items()}


def differences(after, before):
    return {group: {name: after[group][name] - before[group][name] for name in NAMES} for group in before}


p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--control', type=Path, required=True)
p.add_argument('--rate-control', type=Path, required=True)
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
control_arrays = {3e-5: np.load(a.control / 'arrays.npz', allow_pickle=False),
                  3e-6: np.load(a.rate_control / 'arrays.npz', allow_pickle=False)}
control_rows = {3e-5: {r['batch']: r for r in map(json.loads, (a.control / 'rows.jsonl').read_text().splitlines())
                       if r['model'] == 'latest' and r['arm'] == 'plan_consistency'},
                3e-6: {r['batch']: r for r in map(json.loads, (a.rate_control / 'rows.jsonl').read_text().splitlines()) if r['rate'] == 3e-6}}
for path, field in ((a.out/'probe.py', 'script_sha256'), (a.out/'verify.py', 'verifier_sha256'),
                    (a.out/'float64_math.py', 'math_sha256'), (a.out/'run.sh', 'wrapper_sha256')):
    assert digest(path) == protocol[field]
assert report['script_sha256'] == protocol['script_sha256']
for root, field in ((a.control, 'control_sha256'), (a.rate_control, 'rate_control_sha256')):
    for name, expected in protocol[field].items():
        assert digest(root / name) == expected
for name, suffix in [('protocol', 'json'), ('rows', 'jsonl'), ('arrays', 'npz'),
                     ('baseline_receipt', 'json'), ('frozen_state', 'json')]:
    assert digest(a.out / f'{name}.{suffix}') == report[name + '_sha256']
assert json.loads((a.out / 'baseline_receipt.json').read_text())['passed']
for path, expected in protocol['graph_sha256'].items():
    assert digest(path) == expected
assert digest(protocol['config_path']) == protocol['config_sha256']
assert digest(protocol['model']['path']) == protocol['model']['sha256']
assert digest('/work/data/native/lab_v7/manifest.json') == protocol['manifest_sha256']
for path, expected in protocol['source_sha256'].items():
    assert digest(Path('/work') / path) == expected
frozen = json.loads((a.out / 'frozen_state.json').read_text())
assert len(frozen) == report['frozen_tensor_count']
assert not set(protocol['head']['trainable_keys']) & frozen.keys()
assert frozen[protocol['head']['scale_key']]['sha256'] == protocol['head']['scale_sha256']
assert report['head_scope']['parameters'] == 254430
assert set(report['head_scope']['trainable']) == set(protocol['head']['trainable_keys'])
assert len(rows) == report['updates'] == protocol['expected_updates'] == 8
assert {(r['batch'], r['rate']) for r in rows} == {(i, lr) for i in range(4) for lr in protocol['rates']}
assert len({x['index'] for x in protocol['monitor']}) == 16
excluded = {x['index'] for x in protocol['monitor']}
recovery = {i for group in protocol['sampler']['recovery_episodes'] for i in group['indices']}
assert all(len(ids) == 128 and not set(ids) & excluded and sum(i in recovery for i in ids) == 16 for ids in protocol['batches'])
for key in arrays.files:
    assert np.isfinite(arrays[key]).all(), key
for prefix in ['monitor_before'] + [f'b{i}_fit_before' for i in range(4)]:
    old_prefix = 'latest_monitor_before' if prefix == 'monitor_before' else prefix.replace('b', 'latest_', 1)
    for key in ('mu', 'logstd', 'native_target', 'observed', 'wheelbase', 'desired_motion', 'native_desired_motion'):
        for rate, reference in control_arrays.items():
            reference_prefix = old_prefix if rate == 3e-5 else prefix
            np.testing.assert_allclose(arrays[prefix+'_'+key], reference[reference_prefix+'_'+key], atol=1e-7, rtol=1e-6)

NAMES = ('steer_rad', 'raw_accel_mps2', 'target_speed_mps', 'tangent_accel_mps2')
PHASES = ('steady_right', 'right_to_left', 'steady_left', 'recovery')
selections = {'aggregate': list(range(16)), **{phase: [i for i, row in enumerate(protocol['monitor']) if row['phase'] == phase] for phase in PHASES}}
assert all(len(selections[phase]) == 4 for phase in PHASES)
max_loss_error = max_motion_error = max_mae_error = 0.


def checked(data, prefix, metrics):
    global max_loss_error, max_motion_error, max_mae_error
    result = reconstruct(data, prefix)
    for name, value in result['losses'].items():
        error = abs(value - metrics['losses'][name])
        max_loss_error = max(max_loss_error, error)
        assert error < 2e-6, (prefix, name, error)
    for name, key in [('predicted', 'desired_motion'), ('native', 'native_desired_motion')]:
        error = float(np.max(np.abs(result[name] - data[prefix+'_'+key])))
        max_motion_error = max(max_motion_error, error)
        np.testing.assert_allclose(result[name], data[prefix+'_'+key], atol=2e-5, rtol=2e-5)
    actual = result['absolute_action_error'].mean(0)
    errors = np.abs(actual - [metrics['action_mae'][name] for name in NAMES])
    max_mae_error = max(max_mae_error, float(errors.max()))
    assert errors.max() < 2e-5
    for phase, values in metrics.get('phases', {}).items():
        values = values.get('action_mae', values)
        actual = result['absolute_action_error'][selections[phase]].mean(0)
        errors = np.abs(actual - [values[name] for name in NAMES])
        max_mae_error = max(max_mae_error, float(errors.max()))
        assert errors.max() < 2e-5
    return result


monitor_before = checked(arrays, 'monitor_before', report['monitor_before'])
before_mae = summarize(monitor_before['absolute_action_error'], selections)
results = []
for row in rows:
    assert row['initial_model_digest'] == report['initial_digest'] != row['after_model_digest']
    assert row['frozen_state_unchanged'] and row['optimizer_steps'] == [1]
    assert row['frozen_state_sha256'] == report['frozen_state_sha256'] and row['head_scope'] == report['head_scope']
    assert row['eligible_policy_tensors'] == 50 and row['forward_gradient_flags_unchanged']
    assert row['optimized_policy_tensors'] == row['policy_tensors_with_grad'] == row['optimizer_state_count'] == 2
    assert row['optimizer_group_lrs'] == [row['rate']]
    assert row['optimizer_defaults']['betas'] == [.9, .999] and row['optimizer_defaults']['weight_decay'] == 1e-4
    bi, rate = row['batch'], row['rate']
    prefix = f"b{bi}_lr{protocol['rates'].index(rate)}"
    initial = checked(arrays, f'b{bi}_fit_before', control_rows[3e-5][bi]['fit_before'])
    after = checked(arrays, prefix+'_fit_after', row['fit_after'])
    monitor = checked(arrays, prefix+'_monitor_after', row['monitor_after'])
    for actual, recorded in ((initial['objective'], row['objective_before']), (after['objective'], row['objective_after'])):
        max_loss_error = max(max_loss_error, abs(actual-recorded))
        assert abs(actual-recorded) < 2e-6
    full = control_rows[rate][bi]
    full_prefix = f'latest_{bi}_plan_consistency' if rate == 3e-5 else f'b{bi}_lr0'
    full_fit = checked(control_arrays[rate], full_prefix+'_fit_after', full['fit_after'])
    full_monitor = checked(control_arrays[rate], full_prefix+'_monitor_after', full['monitor_after'])
    after_mae = summarize(monitor['absolute_action_error'], selections)
    full_mae = summarize(full_monitor['absolute_action_error'], selections)
    results.append({'batch': bi, 'rate': rate, 'fit': {'before_objective': initial['objective'],
                    'head_after_objective': after['objective'], 'full_after_objective': full_fit['objective'],
                    'head_decrease': initial['objective']-after['objective'], 'full_decrease': initial['objective']-full_fit['objective'],
                    'head_before_losses': initial['losses'], 'head_after_losses': after['losses'],
                    'head_action_mae_delta': dict(zip(NAMES, (after['absolute_action_error'].mean(0)-initial['absolute_action_error'].mean(0)).tolist()))},
                    'monitor_before': before_mae, 'head_monitor_after': after_mae, 'full_monitor_after': full_mae,
                    'head_minus_initial': differences(after_mae, before_mae),
                    'head_minus_full_scope': differences(after_mae, full_mae)})
classification = {}
for rate in protocol['rates']:
    cells = [row for row in results if row['rate'] == rate]
    classification[str(rate)] = {'fit_decreases_by_batch': [row['fit']['head_decrease'] > 0 for row in cells],
        'monitor_steering_decreases_vs_initial_by_batch': [row['head_minus_initial']['aggregate']['steer_rad'] < 0 for row in cells],
        'monitor_steering_decreases_vs_full_scope_by_batch': [row['head_minus_full_scope']['aggregate']['steer_rad'] < 0 for row in cells]}
receipt = {'passed': True, 'run_pid_closed': a.pid, 'exit_code': 0, 'updates': 8,
           'before_arrays_match_both_controls': True, 'source_model_graph_hashes_match': True,
           'method': 'Independent NumPy float64 reconstruction of all five losses and all four motion channels from saved FP32 predictions after the update process exited. Full-scope controls are retained arrays, never rerun.',
           'limitations': 'Frozen state equality and optimizer execution are asserted in the run process. Monitor indices are disjoint but may share episodes and history or future windows with fit samples. Arithmetic discrepancies are not a repeatability noise floor. Eight local steps do not qualify production training or closed-loop improvement.',
           'max_absolute_loss_discrepancy': max_loss_error, 'max_absolute_motion_discrepancy': max_motion_error,
           'max_absolute_mae_discrepancy': max_mae_error, 'classification': classification, 'cells': results,
           'verifier_repair': 'Read the nested action_mae field in the original full-scope phase records; the frozen bounds, arrays and run protocol are unchanged. Original verifier failure is preserved in verify.log.',
           'original_verifier_sha256': digest(a.out/'verify.py'),
           'protocol_sha256': digest(a.out/'protocol.json'), 'report_sha256': digest(a.out/'report.json'),
           'arrays_sha256': digest(a.out/'arrays.npz'), 'verifier_sha256': digest(__file__), 'math_sha256': digest(a.out/'float64_math.py')}
(a.out/'independent_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
assert sum(path.stat().st_size for path in a.out.iterdir() if path.is_file()) < protocol['limits']['artifact_bytes']
assert shutil.disk_usage(a.out).free >= protocol['limits']['minimum_free_bytes']
print(json.dumps({key: value for key, value in receipt.items() if key != 'cells'}), flush=True)
