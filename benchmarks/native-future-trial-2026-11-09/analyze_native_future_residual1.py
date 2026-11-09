#!/usr/bin/env python3
"""Analyze the sixth native run's saved outputs without inference or training."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {'mean': float(values.mean()), 'p95': float(np.percentile(values, 95)),
            'max': float(values.max())}


def smooth_l1(error):
    error = np.abs(error)
    return np.where(error < .05, .5 * error ** 2 / .05, error - .025)


def bound_document(bindings, path):
    suffix = path.as_posix()
    matches = [value for key, value in bindings.items()
               if key == suffix or key.endswith('/' + suffix)]
    assert matches == [digest(path)], f'Hash binding missing or different for {path}'


@dataclass
class Replay:
    root: Path
    report: dict
    anchors: list
    arrays: dict
    run: dict
    selection: dict

    @classmethod
    def read(cls, root, training):
        report = json.loads((root / 'report.json').read_text())
        assert set(report['output_sha256']) == {'arrays.npz', 'anchors.json'}
        for name, expected in report['output_sha256'].items():
            assert digest(root / name) == expected, f'Changed replay output {root / name}'
        anchors = json.loads((root / 'anchors.json').read_text())
        with np.load(root / 'arrays.npz', allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        run = json.loads((training / 'run.json').read_text())
        selection = json.loads((training / 'selection_receipt.json').read_text())
        for name in ('run.json', 'selection_receipt.json', 'selection_protocol.json'):
            bound_document(report['input_sha256'], training / name)
        archive = training / 'source.zip'
        archive_bindings = [value for key, value in report['input_sha256'].items()
                            if key == archive.as_posix() or key.endswith('/' + archive.as_posix())]
        assert archive_bindings == [run['source']['archive_sha256']]
        if archive.exists():
            bound_document(report['input_sha256'], archive)
        for name in ('run.json', 'history.jsonl', 'selection_protocol.json'):
            bound_document(selection['input_sha256'], training / name)
        assert report['runtime_source_sha256'] == run['source']['files_sha256']
        assert report['manifest_sha256'] == run['manifest_sha256']
        assert report['validation_selection'] == run['validation_selection']
        assert report['history_context_proof'] == run['history_context_proof']
        assert report['dataset_verification'] == run['dataset_verification']
        assert len(anchors) == 256 and arrays['target'].shape == (256, 33, 15)
        assert arrays['native_future_states'].shape == (256, 201, 9)
        for values in arrays.values():
            assert np.isfinite(values).all()
        base = [{key: value for key, value in row.items() if key not in ('step0', 'step500', 'native')}
                for row in anchors]
        assert hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest() == report['anchor_inventory_sha256']
        assert [row['index'] for row in anchors] == list(range(256))
        candidates = {row['step']: row for row in selection['candidates']}
        for name, step in (('step0', 0), ('step500', 500)):
            assert report['checkpoints'][name]['sha256'] == candidates[step]['checkpoint_sha256']
            assert report['models'][name]['replay_matches_recorded_validation']
        return cls(root, report, base, arrays, run, selection)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=Path('.work/lab/native_future_residual1'))
parser.add_argument('--previous', type=Path, default=Path('.work/lab/tangent_residual1'))
parser.add_argument('--training', type=Path, default=Path('runs/lab_v4/native_future_trial1'))
parser.add_argument('--previous-training', type=Path, default=Path('runs/lab_v4/tangent_trial1'))
args = parser.parse_args()
current = Replay.read(args.input, args.training)
previous = Replay.read(args.previous, args.previous_training)
assert current.report['schema_version'] == 1
assert current.selection['schema_version'] == current.selection['protocol']['schema_version'] == 2
bound_document(current.report['input_sha256'], Path('.work/lab/native_future_residual_diagnostic.py'))
assert current.run['action_target_contract']['mode'] == 'native_future_v1'
assert current.report['action_target_contract'] == current.run['action_target_contract']
assert current.selection['action_target_contract'] == current.run['action_target_contract']
assert current.anchors == previous.anchors
assert current.report['anchor_inventory_sha256'] == previous.report['anchor_inventory_sha256']
assert current.report['episodes'] == previous.report['episodes']
assert current.report['manifest_sha256'] == previous.report['manifest_sha256']
assert current.run['init_sha256'] == previous.run['init_sha256']
assert current.report['history_context_proof'] == previous.report['history_context_proof']
for key in ('target', 'times_s', 'interval_dt_s', 'observed_vx_vy_yaw_rate',
            'teacher_actions', 'native_future_states', 'native_actions'):
    np.testing.assert_array_equal(current.arrays[key], previous.arrays[key])
for key in ('step0_mu', 'step0_logstd', 'step0_actions'):
    np.testing.assert_array_equal(current.arrays[key], previous.arrays[key])

z = current.arrays
anchors = current.anchors
times = z['times_s']
n = int(np.searchsorted(times, 1., side='right'))
dt = np.diff(times[:n])
assert n == 11
np.testing.assert_array_equal(dt, z['interval_dt_s'])
target = z['target'].astype(np.float64)
observed = z['observed_vx_vy_yaw_rate'].astype(np.float64)
vehicle = {key: z[key].astype(np.float64) for key in
           ('vehicle_wheelbase', 'vehicle_max_brake', 'vehicle_max_accel')}
assert all(value.shape == (256,) for value in vehicle.values())
assert all((value > 0).all() for value in vehicle.values())
weights = np.where(times <= 4., 1., np.maximum(.2, 1. - .8 * (times - 4.) / 6.))[:, None]
weights = weights * np.array([1, 1, .1, .5, .5, .1, .1, .1, 0, 0, 0, .5, 0, 0, .3])[None, :]
cold = np.array([row['usable_history_frames'] < 8 for row in anchors])
cases = list(dict.fromkeys(row['case'] for row in anchors))
groups = {'all': np.ones(256, bool), 'history_lt8': cold, 'history_8': ~cold}
groups.update({case: np.array([row['case'] == case for row in anchors]) for case in cases})
assert cold.sum() == 13 and all(groups[case].sum() == 32 for case in cases)
rows = np.arange(256)


def residual(plan):
    secant = np.diff(plan[:, :n, 3:5], axis=1) / dt[None, :, None]
    endpoints = np.stack((plan[:, :n - 1, 6:8], plan[:, 1:n, 6:8]), axis=2)
    return endpoints - secant[:, :, None, :]


def interpolate(values, time):
    hi = int(np.searchsorted(times, time))
    alpha = (time - times[hi - 1]) / (times[hi] - times[hi - 1])
    return values[:, hi - 1] * (1. - alpha) + values[:, hi] * alpha


def actions(plan):
    velocity = plan[:, :, 3:5]
    acceleration = plan[:, :, 6:8]
    speed = np.linalg.norm(velocity, axis=-1)
    tangent = (velocity * acceleration).sum(-1) / np.maximum(speed, .001)
    raw = 1.5 * (interpolate(speed, .3) - np.linalg.norm(observed[:, :2], axis=-1))
    raw += .5 * interpolate(tangent, .3)
    curvature = -2. * interpolate(plan[:, :, 11], .2) / (np.maximum(observed[:, 0], 1.) * .2)
    curvature += plan[:, 0, 14] / np.maximum(observed[:, 0], 1.)
    steer = np.arctan(vehicle['vehicle_wheelbase'] * curvature)
    longitudinal = np.clip(raw, -vehicle['vehicle_max_brake'], vehicle['vehicle_max_accel'])
    return np.stack((steer, longitudinal), axis=-1), raw


def coherence(plan):
    pos = np.diff(plan[:, :n, :2], axis=1) - .5 * (plan[:, :n - 1, 3:5] + plan[:, 1:n, 3:5]) * dt[None, :, None]
    vel = np.diff(plan[:, :n, 3:5], axis=1) - .5 * (plan[:, :n - 1, 6:8] + plan[:, 1:n, 6:8]) * dt[None, :, None]
    angle = np.diff(plan[:, :n, 11], axis=1)
    yaw = angle - .5 * (plan[:, :n - 1, 14] + plan[:, 1:n, 14]) * dt[None]
    velocity_secant = np.diff(plan[:, :n, :2], axis=1) / dt[None, :, None]
    velocity_endpoints = np.stack((plan[:, :n - 1, 3:5], plan[:, 1:n, 3:5]), axis=2)
    yaw_endpoints = np.stack((plan[:, :n - 1, 14], plan[:, 1:n, 14]), axis=2)
    return {
        'velocity_secant_max_mps': np.linalg.norm(velocity_endpoints - velocity_secant[:, :, None, :], axis=-1).max((1, 2)),
        'acceleration_secant_max_mps2': np.linalg.norm(residual(plan), axis=-1).max((1, 2)),
        'yaw_rate_secant_max_radps': abs(yaw_endpoints - (angle / dt[None])[:, :, None]).max((1, 2)),
        'position_interval_m': np.linalg.norm(pos, axis=-1).mean(1),
        'position_interval_max_m': np.linalg.norm(pos, axis=-1).max(1),
        'velocity_interval_mps': np.linalg.norm(vel, axis=-1).mean(1),
        'velocity_interval_max_mps': np.linalg.norm(vel, axis=-1).max(1),
        'yaw_interval_rad': abs(yaw).mean(1), 'yaw_interval_max_rad': abs(yaw).max(1),
        'position_cumulative_max_m': np.linalg.norm(np.cumsum(pos, axis=1), axis=-1).max(1),
        'velocity_cumulative_max_mps': np.linalg.norm(np.cumsum(vel, axis=1), axis=-1).max(1),
        'yaw_cumulative_max_rad': abs(np.cumsum(yaw, axis=1)).max(1),
        'initial_position_m': np.linalg.norm(plan[:, 0, :2], axis=-1),
        'initial_yaw_rad': abs(plan[:, 0, 11]),
        'initial_velocity_mps': np.linalg.norm(plan[:, 0, 3:5] - observed[:, :2] * [1, -1], axis=-1),
        'initial_yaw_rate_radps': abs(plan[:, 0, 14] + observed[:, 2]),
    }


native_residual = residual(target)
np.testing.assert_allclose(native_residual, z['native_acceleration_endpoint_residual_mps2'], rtol=1e-6, atol=1e-5)
native_actions, _ = actions(target)
np.testing.assert_allclose(native_actions, z['native_actions'], rtol=1e-5, atol=1e-5)
model_arrays = {'step0': (z, 'step0'), 'fifth_step500': (previous.arrays, 'step500'),
                'step500': (z, 'step500'), 'native': (z, 'native')}
summary = {
    'schema_version': 1,
    'scope': 'Descriptive saved-output analysis on 256 fixed anchors. No training, physical trial, selection, weight fitting or promotion.',
    'inputs_sha256': {str(replay.root / name): digest(replay.root / name)
                     for replay in (current, previous) for name in ('report.json', 'anchors.json', 'arrays.npz')},
    'script_sha256': digest(Path(__file__)),
    'bindings': {'anchor_inventory_sha256': current.report['anchor_inventory_sha256'],
                 'identical_anchor_base_records': True, 'identical_episode_hashes': True,
                 'identical_targets_states_and_native_actions': True, 'identical_step0_outputs': True,
                 'current_source_archive_rehashed_locally': (args.training / 'source.zip').exists(),
                 'previous_source_archive_rehashed_locally': (args.previous_training / 'source.zip').exists(),
                 'current_action_target_contract': current.run['action_target_contract']},
    'models': {}, 'anchor_comparison': [], 'intervention': {},
    'limitations': [
        'One deterministic replay does not estimate repeatability or generalization. Anchors and visual variants are correlated.',
        'The intervention changes saved predictions only. No network can be assumed to realize the same change.',
        'The intervention uses own velocity knots 4 and 7 to interpolate knots 5 and 6. Native labels score the result but do not construct it.',
        'Uncertainty, acceleration, yaw and all other prediction coordinates remain fixed in the intervention.',
        'Finite output changes and uncertainty coefficients do not establish parameter-gradient conflict or the cause of a physical failure.',
        'Existing thresholds are reported unchanged. No intervention or prior checkpoint becomes eligible through this analysis.',
    ],
}
maxima = {}
model_values = {}
for label, (arrays, prefix) in model_arrays.items():
    mu = arrays[prefix + '_mu'].astype(np.float64)
    r = residual(mu)
    np.testing.assert_allclose(r, arrays[prefix + '_acceleration_endpoint_residual_mps2'], rtol=1e-6, atol=1e-5)
    accel_error = np.stack(((mu - target)[:, :n - 1, 6:8], (mu - target)[:, 1:n, 6:8]), axis=2)
    velocity_error = np.diff((mu - target)[:, :n, 3:5], axis=1) / dt[None, :, None]
    np.testing.assert_allclose(r - native_residual, accel_error - velocity_error[:, :, None, :], rtol=1e-6, atol=1e-5)
    norms = np.linalg.norm(r, axis=-1)
    maximum = norms.max(axis=(1, 2))
    maxima[label] = maximum
    location = np.array([np.unravel_index(np.argmax(row), row.shape) for row in norms])
    wr = r[rows, location[:, 0], location[:, 1]]
    wa = accel_error[rows, location[:, 0], location[:, 1]]
    wv = velocity_error[rows, location[:, 0]]
    excess = (r - native_residual)[rows, location[:, 0], location[:, 1]]
    direction = excess / np.maximum(np.linalg.norm(excess, axis=-1)[:, None], 1e-12)
    action, raw = actions(mu)
    np.testing.assert_allclose(action, arrays[prefix + '_actions'], rtol=1e-5, atol=1e-5)
    action_errors = action - native_actions
    action_losses = smooth_l1(action_errors)
    teacher_losses = smooth_l1(action - z['teacher_actions'])
    l1 = (abs(mu - target) * weights[None]).sum((1, 2)) / weights.sum()
    model_values[label] = {'mu': mu, 'residual': r, 'actions': action, 'raw': raw,
                           'plan_l1': l1, 'action_losses': action_losses}
    if prefix != 'native':
        logstd = arrays[prefix + '_logstd'].astype(np.float64)
        logb = np.clip(logstd, -7., 7.)
        coefficient = weights[None] * np.exp(-logb) / weights.sum()
        nll = ((abs(mu - target) * np.exp(-logb) + logb) * weights[None]).sum((1, 2)) / weights.sum()
        np.testing.assert_allclose(coefficient, arrays[prefix + '_nll_mu_gradient_scale'], rtol=1e-5, atol=1e-5)
        model_values[label]['plan_nll'] = nll
    model_groups = {**groups, 'above4': maximum > 4., 'history_8_above4': (~cold) & (maximum > 4.)}
    result = {'groups': {}, 'intervals': [], 'nll_near_knots': {}, 'above4_anchors': []}
    for group, mask in model_groups.items():
        count = int(mask.sum())
        value = {'count': count, 'acceleration_secant': stats(maximum[mask]),
                 'above4': int((maximum[mask] > 4.).sum()),
                 'worst_interval_counts': dict(Counter(location[mask, 0].tolist())),
                 'worst_endpoint_counts': dict(Counter(location[mask, 1].tolist()))}
        if count:
            value.update({
                'dominant_residual_axis_counts': dict(Counter(np.argmax(abs(wr[mask]), axis=-1).tolist())),
                'mean_acceleration_error_projected_on_excess': float((wa[mask] * direction[mask]).sum(-1).mean()),
                'mean_negative_velocity_secant_error_projected_on_excess': float((-wv[mask] * direction[mask]).sum(-1).mean()),
                'mean_excess_norm_at_worst_raw_endpoint': float(np.linalg.norm(excess[mask], axis=-1).mean()),
                'native_relative_residual_max': stats(np.linalg.norm(r[mask] - native_residual[mask], axis=-1).max((1, 2))),
                'forward_fraction_of_absolute_worst_residual': float(abs(wr[mask, 0]).sum() / max(abs(wr[mask]).sum(), 1e-12)),
                'plan_l1': float(l1[mask].mean()),
                'native_action_smooth_l1_steer_longitudinal': action_losses[mask].mean(0).tolist(),
                'teacher_action_smooth_l1_steer_longitudinal': teacher_losses[mask].mean(0).tolist(),
                'native_action_mae_steer_longitudinal': abs(action_errors[mask]).mean(0).tolist(),
                'native_action_mean_signed_error_steer_longitudinal': action_errors[mask].mean(0).tolist(),
                'mean_longitudinal_speed_feedback_error': float((arrays[prefix + '_longitudinal_components'][mask, 2] - z['native_longitudinal_components'][mask, 2]).mean()),
                'mean_longitudinal_acceleration_contribution_error': float(.5 * (arrays[prefix + '_longitudinal_components'][mask, 1] - z['native_longitudinal_components'][mask, 1]).mean()),
                'near_knot_mean_vx_error': (mu - target)[mask, :n, 3].mean(0).tolist(),
                'near_knot_mean_ax_error': (mu - target)[mask, :n, 6].mean(0).tolist(),
                'near_knot_mean_abs_vx_error': abs(mu - target)[mask, :n, 3].mean(0).tolist(),
                'near_knot_mean_abs_ax_error': abs(mu - target)[mask, :n, 6].mean(0).tolist(),
            })
            if prefix != 'native':
                value['plan_nll'] = float(nll[mask].mean())
                result['nll_near_knots'][group] = [
                    {'knot': knot, 'time_s': float(times[knot]), 'column': column,
                     'mean_abs_native_error': float(abs(mu - target)[mask, knot, column].mean()),
                     'mean_logstd': float(logstd[mask, knot, column].mean()),
                     'mean_effective_mu_coefficient': float(coefficient[mask, knot, column].mean())}
                    for knot in range(n) for column in (3, 4, 6, 7)]
        result['groups'][group] = value
    for i in range(n - 1):
        result['intervals'].append({'interval': i, 'lo_s': float(times[i]), 'hi_s': float(times[i + 1]),
                                    'dt_s': float(dt[i]), 'maximum_endpoint_residual': stats(norms[:, i].max(-1))})
    for index in np.flatnonzero(maximum > 4.):
        i, e = location[index]
        result['above4_anchors'].append({'anchor': anchors[index], 'maximum_mps2': float(maximum[index]),
            'interval': int(i), 'endpoint': int(e), 'lo_s': float(times[i]), 'hi_s': float(times[i + 1]),
            'residual_xy': wr[index].tolist(), 'native_residual_xy': native_residual[index, i, e].tolist(),
            'acceleration_error_xy': wa[index].tolist(), 'velocity_secant_error_xy': wv[index].tolist()})
    summary['models'][label] = result

for label, step in (('step0', 0), ('step500', 500)):
    expected = next(row for row in current.selection['candidates'] if row['step'] == step)['validation']
    values = model_values[label]
    for key in ('plan_l1', 'plan_nll'):
        np.testing.assert_allclose(values[key].mean(), expected['aggregate'][key], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(values['action_losses'].mean(0),
                               [expected['aggregate'][key] for key in ('steer', 'longitudinal')], rtol=1e-5, atol=1e-5)
    for key, value in stats(maxima[label]).items():
        np.testing.assert_allclose(value, expected['coherence']['acceleration_secant_max_mps2'][key], rtol=1e-5, atol=1e-5)
    for metric, metric_values in coherence(values['mu']).items():
        for key, value in stats(metric_values).items():
            np.testing.assert_allclose(value, expected['coherence'][metric][key], rtol=1e-5, atol=1e-5)
    for case in cases:
        episode = next(row for row in expected['episodes'] if Path(row['path']).name == case)
        np.testing.assert_allclose(values['action_losses'][groups[case]].mean(0),
                                   [episode[key] for key in ('steer', 'longitudinal')], rtol=1e-5, atol=1e-5)
summary['validation_reproduced_from_saved_arrays'] = True
for index, anchor in enumerate(anchors):
    summary['anchor_comparison'].append({'anchor': anchor,
        'maximum_mps2': {name: float(values[index]) for name, values in maxima.items()},
        'actions_steer_longitudinal': {name: values['actions'][index].tolist() for name, values in model_values.items()},
        'native_action_losses_steer_longitudinal': {name: values['action_losses'][index].tolist() for name, values in model_values.items()},
        'sixth_minus_fifth_maximum_mps2': float(maxima['step500'][index] - maxima['fifth_step500'][index]),
        'sixth_minus_initial_maximum_mps2': float(maxima['step500'][index] - maxima['step0'][index])})
summary['paired_tail_membership'] = {
    'fifth_above4': int((maxima['fifth_step500'] > 4.).sum()),
    'sixth_above4': int((maxima['step500'] > 4.).sum()),
    'both_above4': int(((maxima['fifth_step500'] > 4.) & (maxima['step500'] > 4.)).sum()),
    'newly_above4': [anchors[i] for i in np.flatnonzero((maxima['step500'] > 4.) & (maxima['fifth_step500'] <= 4.))],
}

mu = model_values['step500']['mu']
changed = mu.copy()
for knot in (5, 6):
    alpha = (times[knot] - times[4]) / (times[7] - times[4])
    changed[:, knot, 3:5] = (1. - alpha) * mu[:, 4, 3:5] + alpha * mu[:, 7, 3:5]
logb = np.clip(z['step500_logstd'].astype(np.float64), -7., 7.)
config = current.run['args']
intervention_values = {}
for name, plan in (('actual', mu), ('interpolate_own_velocity_knots5_6', changed)):
    action, raw = actions(plan)
    r = residual(plan)
    nll = ((abs(plan - target) * np.exp(-logb) + logb) * weights[None]).sum((1, 2)) / weights.sum()
    l1 = (abs(plan - target) * weights[None]).sum((1, 2)) / weights.sum()
    loss = smooth_l1(action - native_actions)
    total = config['w_plan'] * nll + config['w_steer'] * loss[:, 0] + config['w_long'] * loss[:, 1]
    intervention_values[name] = {'maximum': np.linalg.norm(r, axis=-1).max((1, 2)),
        'native_relative_maximum': np.linalg.norm(r - native_residual, axis=-1).max((1, 2)),
        'nll': nll, 'l1': l1, 'loss': loss, 'total': total, 'actions': action, 'raw': raw}
    summary['intervention'][name] = {}
    coherence_values = coherence(plan)
    for group, mask in {**groups, 'sixth_above4': maxima['step500'] > 4.}.items():
        value = intervention_values[name]
        summary['intervention'][name][group] = {
            'count': int(mask.sum()), 'acceleration_secant': stats(value['maximum'][mask]),
            'native_relative_maximum': stats(value['native_relative_maximum'][mask]),
            'above4': int((value['maximum'][mask] > 4.).sum()),
            'plan_nll_frozen_logstd': float(nll[mask].mean()), 'plan_l1': float(l1[mask].mean()),
            'native_action_smooth_l1_steer_longitudinal': loss[mask].mean(0).tolist(),
            'weighted_current_objective': float(total[mask].mean()),
            'clipped_longitudinal_count': int((abs(raw[mask] - action[mask, 1]) > 1e-8).sum()),
            'coherence': {metric: stats(values[mask]) for metric, values in coherence_values.items()},
        }
np.testing.assert_array_equal(intervention_values['actual']['actions'][:, 0],
                              intervention_values['interpolate_own_velocity_knots5_6']['actions'][:, 0])
summary['intervention']['paired_deltas'] = {}
before = intervention_values['actual']
after = intervention_values['interpolate_own_velocity_knots5_6']
for group, mask in {**groups, 'sixth_above4': maxima['step500'] > 4.}.items():
    residual_better = after['maximum'] < before['maximum'] - 1e-8
    objective_worse = after['total'] > before['total'] + 1e-8
    summary['intervention']['paired_deltas'][group] = {
        'count': int(mask.sum()), 'mean_objective_delta': float((after['total'] - before['total'])[mask].mean()),
        'mean_plan_nll_delta': float((after['nll'] - before['nll'])[mask].mean()),
        'mean_plan_l1_delta': float((after['l1'] - before['l1'])[mask].mean()),
        'mean_longitudinal_loss_delta': float((after['loss'][:, 1] - before['loss'][:, 1])[mask].mean()),
        'residual_improves_count': int((residual_better & mask).sum()),
        'objective_improves_count': int(((after['total'] < before['total'] - 1e-8) & mask).sum()),
        'residual_improves_objective_worsens_count': int((residual_better & objective_worse & mask).sum()),
    }
summary['intervention']['frozen_coherence_bound_checks'] = []
bounds = current.selection['protocol']['bounds']
for group in ('all', *cases):
    group_bounds = bounds['coherence'] if group == 'all' else bounds['episode_coherence'][group]
    for metric, metric_bounds in group_bounds.items():
        for statistic, limit in metric_bounds.items():
            before_value = summary['intervention']['actual'][group]['coherence'][metric][statistic]
            after_value = summary['intervention']['interpolate_own_velocity_knots5_6'][group]['coherence'][metric][statistic]
            summary['intervention']['frozen_coherence_bound_checks'].append({
                'group': group, 'metric': metric, 'statistic': statistic, 'limit': limit,
                'actual': before_value, 'intervened': after_value,
                'actual_within_bound': bool(before_value <= limit),
                'intervened_within_bound': bool(after_value <= limit),
            })

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25})
colors = {'step0': '#777777', 'fifth_step500': '#d95f02', 'step500': '#2166ac', 'native': '#238b45'}
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for name, values in model_values.items():
    interval = np.linalg.norm(values['residual'], axis=-1).max(-1)
    axes[0, 0].plot(np.arange(10), np.percentile(interval, 95, axis=0), marker='o', label=name, color=colors[name])
axes[0, 0].set(title='Per-interval p95 across the same 256 anchors', xlabel='Interval', ylabel='Endpoint residual (m/s²)')
axes[0, 0].legend()
for ax, mask, title in ((axes[0, 1], cold, '13 anchors with fewer than 8 history frames'),
                         (axes[1, 0], ~cold, '243 anchors with full history')):
    for name in ('step0', 'fifth_step500', 'step500'):
        ax.plot(times[:n], (model_values[name]['mu'] - target)[mask, :n, 3].mean(0), marker='o', label=name, color=colors[name])
    ax.axhline(0, color=colors['native'])
    ax.axvspan(times[5], times[6], color='#dfb23f', alpha=.15)
    ax.set(title=title, xlabel='Plan time (s)', ylabel='Mean forward-velocity error (m/s)')
axes[0, 1].legend()
axes[1, 1].scatter(maxima['fifth_step500'][~cold], maxima['step500'][~cold], s=18, alpha=.6, label='Full history')
axes[1, 1].scatter(maxima['fifth_step500'][cold], maxima['step500'][cold], s=35, marker='x', label='Short history')
upper = max(maxima['fifth_step500'].max(), maxima['step500'].max()) * 1.05
axes[1, 1].plot([0, upper], [0, upper], color='grey', lw=.7)
axes[1, 1].axhline(4, color='#b2182b', ls='--', lw=.8)
axes[1, 1].set(title='Paired fifth and sixth run maxima', xlabel='Fifth run (m/s²)', ylabel='Sixth run (m/s²)')
axes[1, 1].legend()
fig.tight_layout()
fig.savefig(args.input / 'residual_localization.png', dpi=150)
plt.close(fig)

representatives = [int(np.flatnonzero(mask)[np.argmax(maxima['step500'][mask])]) for mask in (cold, ~cold)]
summary['representative_anchor_indices'] = representatives
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for row, index in enumerate(representatives):
    for name, value in model_values.items():
        axes[row, 0].plot(times[:n], value['mu'][index, :n, 3], marker='o', label=name, color=colors[name])
        axes[row, 1].plot(times[:n], value['mu'][index, :n, 6], marker='o', label=name, color=colors[name])
        axes[row, 2].plot(np.arange(10), np.linalg.norm(value['residual'][index], axis=-1).max(-1), marker='o', label=name, color=colors[name])
    axes[row, 0].set(title=anchors[index]['case'] + ' / frame ' + str(anchors[index]['frame']), xlabel='Plan time (s)', ylabel='Forward velocity (m/s)')
    axes[row, 1].set(title='Acceleration head', xlabel='Plan time (s)', ylabel='Forward acceleration (m/s²)')
    axes[row, 2].set(title='Maximum endpoint residual', xlabel='Interval', ylabel='Residual (m/s²)')
axes[0, 0].legend()
fig.tight_layout()
fig.savefig(args.input / 'representative_plans.png', dpi=150)
plt.close(fig)
summary['plot_sha256'] = {name: digest(args.input / name) for name in ('residual_localization.png', 'representative_plans.png')}
(args.input / 'analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps({'output': str(args.input / 'analysis.json'),
    'paired_tail_membership': summary['paired_tail_membership'],
    'sixth_groups': {name: {key: summary['models']['step500']['groups'][name][key]
                           for key in ('count', 'acceleration_secant', 'above4', 'worst_interval_counts')}
                     for name in ('all', 'history_lt8', 'history_8', 'above4')},
    'intervention_deltas': {name: summary['intervention']['paired_deltas'][name] for name in ('all', 'history_lt8', 'history_8', 'sixth_above4')}}, indent=2))
