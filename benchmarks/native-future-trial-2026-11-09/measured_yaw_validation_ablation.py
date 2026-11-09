"""Reproduce a saved-output yaw-rate substitution without inference or physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / '.work/lab'
CHECKPOINT = 'c121faa268d8c98bdac537974a9c21d3523a7b57cdab7f480351cbb2f98f89ac'
BETA = 0.05


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {'mean': float(values.mean()), 'p95': float(np.percentile(values, 95)),
            'max': float(values.max())}


def smooth_l1(error):
    absolute = np.abs(error)
    return np.where(absolute < BETA, 0.5 * absolute ** 2 / BETA, absolute - 0.5 * BETA)


def summarize(original, substituted, target, mask):
    old_error, new_error = (x[mask] - target[mask] for x in (original, substituted))
    old_loss, new_loss = map(smooth_l1, (old_error, new_error))
    return {'anchors': int(mask.sum()),
            'original_smooth_l1': float(old_loss.mean()),
            'substituted_smooth_l1': float(new_loss.mean()),
            'smooth_l1_delta': float((new_loss - old_loss).mean()),
            'original_absolute_error_rad': distribution(np.abs(old_error)),
            'substituted_absolute_error_rad': distribution(np.abs(new_error)),
            'worse_anchors': int(np.sum(new_loss > old_loss)),
            'better_anchors': int(np.sum(new_loss < old_loss)),
            'equal_anchors': int(np.sum(new_loss == old_loss))}


def body_audit(document, local_episode):
    source_episode = Path(document['episode'])
    for source, expected in document['input_sha256'].items():
        path = local_episode / Path(source).relative_to(source_episode)
        require(sha(path) == expected, f'body audit input hash mismatch: {path}')
    frames = read(local_episode / 'frames.json')
    manifest = [json.loads(line) for line in (local_episode / 'evidence/manifest.jsonl').read_text().splitlines()]
    yaw = np.unwrap([row['yaw'] for row in frames])
    rows = document['rows']
    require(len(rows) == 90 and len({row['frame'] for row in rows}) == 90, 'unexpected body audit sample')
    exact_differences, finite_differences = [], []
    for row in rows:
        index = row['frame']
        require(manifest[index]['frame'] == index and manifest[index]['sha256'] == row['frame_sha256'],
                'body audit frame digest differs from bound manifest')
        axis = np.asarray(row['body_forward_axis_world'], dtype=np.float64)
        omega = np.asarray(row['angular_velocity_world'], dtype=np.float64)
        require(axis.shape == omega.shape == (3,) and np.isfinite([axis, omega]).all(), 'invalid body vectors')
        denominator = axis[0] ** 2 + axis[1] ** 2
        require(denominator > 1e-6, 'planar heading is singular')
        exact = omega[2] - axis[2] * (axis[0] * omega[0] + axis[1] * omega[1]) / denominator
        require(abs(exact - row['exact_heading_derivative']) < 1e-12, 'exact heading formula differs')
        require(abs(omega[2] - row['actual_yaw_rate']) < 1e-9, 'world Z differs from state r')
        exact_differences.append(abs(exact - row['actual_yaw_rate']))
        if 0 < index < len(frames) - 1:
            duration = frames[index + 1]['t'] - frames[index - 1]['t']
            require(abs(duration - 0.1) < 1e-10, 'unexpected finite-difference cadence')
            slope = (yaw[index + 1] - yaw[index - 1]) / duration
            require(abs(slope - row['central_heading_slope_01s']) < 1e-10, 'saved heading slope differs')
            finite_differences.append(abs(slope - exact))
        else:
            require(row['central_heading_slope_01s'] is None, 'unexpected endpoint slope')
    fields = ('frame', 't', 'actual_yaw_rate', 'exact_heading_derivative',
              'central_heading_slope_01s', 'predicted_initial_yaw_rate')
    return {'sampled_frames': len(rows), 'finite_difference_frames': len(finite_differences),
            'exact_minus_world_z_absolute_radps': distribution(exact_differences),
            'central_slope_minus_exact_absolute_radps': distribution(finite_differences),
            'examples': [{key: row[key] for key in fields} for row in rows if row['frame'] in (4, 40, 80, 480, 520)],
            'producer_input_sha256': document['input_sha256'],
            'producer_script_sha256': document['script_sha256'],
            'verification': 'Recomputed retained body-vector formula and heading slopes; checked local media and manifest hashes. Raw frame NPZ files were not available locally and were not reopened.',
            'interpretation': 'The 3D projection difference does not explain the large instantaneous-versus-0.1-second-average discrepancy in these samples. Averaging, phase and dynamics remain unresolved. A central finite difference is not an instantaneous-rate measurement.'}


def calculate():
    residual = LAB / 'native_future_residual1'
    report = read(residual / 'report.json')
    for name in ('arrays.npz', 'anchors.json'):
        require(sha(residual / name) == report['output_sha256'][name], f'replay hash mismatch: {name}')
    anchors = read(residual / 'anchors.json')
    require(len(anchors) == 256 and [row['index'] for row in anchors] == list(range(256)), 'unexpected anchor inventory')
    require(report['checkpoints']['step500']['sha256'] == CHECKPOINT and report['step500_eligible'] is False,
            'unexpected diagnostic checkpoint or eligibility')
    readout_path, readout_script = LAB / 'native_future_steering_readout2.json', LAB / 'native_future_steering_readout2.py'
    readout = read(readout_path)
    require(readout['checkpoint_sha256'] == CHECKPOINT, 'body audit checkpoint mismatch')
    require(sha(readout_script) == readout['script_sha256'], 'body audit producer script mismatch')
    source_paths = [ROOT / name for name in ('src/fsop/eval/controller.py', 'src/fsop/train/losses.py',
                                            'src/fsop/sim/expert.py', 'src/fsop/data/native_replay.py')]
    for path in source_paths:
        require(sha(path) == report['runtime_source_sha256'][str(path.relative_to(ROOT / 'src'))],
                f'frozen replay source mismatch: {path}')
    with np.load(residual / 'arrays.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    times = arrays['times_s']
    observed = arrays['observed_vx_vy_yaw_rate']
    require(times.shape == (33,) and np.array_equal(times, 10 * (np.arange(33) / 32) ** 2), 'unexpected time knots')
    require(observed.shape == (256, 3) and arrays['native_mu'].shape == (256, 33, 15), 'unexpected replay shape')
    require(all(np.isfinite(value).all() for value in arrays.values()), 'nonfinite replay array')
    require(np.array_equal(arrays['target'], arrays['native_mu']), 'native plan differs from target')
    require(np.array_equal(-arrays['native_mu'][:, 0, 14], observed[:, 2]), 'native initial rate differs from observed rate')
    require(np.allclose(arrays['native_future_states'][:, 0, 3:6], observed, atol=1e-7, rtol=1e-6),
            'observed state differs from native initial state')
    speed = np.maximum(observed[:, 0], 1.0)
    wheelbase = arrays['vehicle_wheelbase']

    def steering(plan, measured=False):
        yaw = -np.array([np.interp(0.2, times, row[:, 11]) for row in plan])
        rate = observed[:, 2] if measured else -plan[:, 0, 14]
        return np.arctan(wheelbase * (2 * yaw / 0.2 - rate) / speed)

    target = arrays['native_actions'][:, 0]
    native_original, native_substituted = steering(arrays['native_mu']), steering(arrays['native_mu'], True)
    require(np.array_equal(native_original, native_substituted), 'substitution changes exact native target')
    require(np.allclose(native_original, target, atol=1e-7, rtol=1e-6), 'native target reconstruction differs')
    groups = {'all': np.ones(256, dtype=bool)}
    groups.update({name: np.array([row['discipline'] == name for row in anchors]) for name in ('trackdrive', 'skidpad')})
    groups.update({name: np.array([row['case'] == name for row in anchors]) for name in sorted({row['case'] for row in anchors})})
    models = {}
    for name in ('step0', 'step500'):
        original, substituted = steering(arrays[name + '_mu']), steering(arrays[name + '_mu'], True)
        saved = arrays[name + '_actions'][:, 0]
        require(np.allclose(original, saved, atol=1e-7, rtol=1e-6), f'{name} original steering reconstruction differs')
        metrics = {group: summarize(original, substituted, target, mask) for group, mask in groups.items()}
        episodes = {group: value for group, value in metrics.items() if group.startswith('val-')}
        require(len(episodes) == 8 and all(value['anchors'] == 32 for value in episodes.values()), 'episode inventory differs')
        loss_delta = smooth_l1(substituted - target) - smooth_l1(original - target)
        models[name] = {'groups': metrics, 'episodes_worsened': sum(value['smooth_l1_delta'] > 0 for value in episodes.values()),
                        'episodes_total': len(episodes), 'original_reconstruction_max_error_rad': float(np.max(np.abs(original - saved))),
                        'largest_loss_increases': [{'index': int(i), 'case': anchors[i]['case'], 'frame': anchors[i]['frame'],
                                                   'original_steer_rad': float(original[i]), 'substituted_steer_rad': float(substituted[i]),
                                                   'native_target_steer_rad': float(target[i]), 'smooth_l1_delta': float(loss_delta[i])}
                                                  for i in np.argsort(loss_delta)[-5:][::-1]]}
    local_episode = ROOT / 'runs/development_media/native_future_trial1/heavy_skid/candidate' / Path(readout['episode']).name
    audit = body_audit(readout, local_episode)
    paths = [Path(__file__).resolve(), residual / 'report.json', residual / 'arrays.npz', residual / 'anchors.json',
             readout_path, readout_script, *source_paths, ROOT / 'isaac_lab_fsop/env.py', ROOT / 'isaac_lab_fsop/evaluate.py']
    return {'schema_version': 1, 'scope': 'Deterministic local saved-output ablation on all existing 256 validation anchors. No model inference, optimizer, GPU, Spark, new scene, or physics execution.',
            'data_shape': {'plans': '[256,33,15] in OP coordinates', 'observed': '[256,3] measured ego vx, vy, world-Z angular rate'},
            'checkpoint_sha256': CHECKPOINT, 'checkpoint_eligible': False,
            'input_sha256': {str(path.relative_to(ROOT)): sha(path) for path in paths},
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
            'contract': {'action_time_s': 0.2, 'speed_floor_mps': 1.0, 'smooth_l1_beta': BETA,
                         'coordinates': 'Negate OP plan yaw and yaw-rate columns to obtain ego convention.',
                         'intervention': 'Replace only predicted initial yaw rate with recorded observed world-Z angular rate. Keep every learned plan output, velocity input, vehicle parameter and target fixed.',
                         'action_scope': 'Steering before mission layer and physical actuator clipping.',
                         'target': 'Unchanged detached native-future plan through the original op_tangent_v1 mapping.',
                         'reconstruction_atol': 1e-7, 'reconstruction_rtol': 1e-6},
            'native_target_invariance': {'exact': True, 'saved_action_max_error_rad': float(np.max(np.abs(native_original - target)))},
            'models': models, 'physical_signal_audit': audit,
            'decision': 'Negative evidence against an unconditional raw measured-rate fix. Do not change the controller or justify a seventh optimizer run from this ablation.',
            'limitations': ['One deterministic replay has no repeatability or population uncertainty estimate. Anchors and visual variants are correlated.',
                            'This substitution does not reproduce closed-loop feedback, changed images, changed temporal history, contacts or mission completion.',
                            'Lower overall mean absolute error and higher SmoothL1 loss are both reported. No metric, eligibility threshold or loss was relaxed.',
                            'Full model checkpoint bytes and original body NPZ records were not reloaded locally. Their producer receipts and retained artifacts remain the provenance boundary.']}


def markdown(result):
    lines = ['# Measured yaw-rate substitution worsens the frozen steering loss', '',
             'The goal is native Isaac training that reduces cone contacts and timeouts with auditable videos. '
             'This local saved-output ablation supplies negative evidence against an unconditional measured-rate controller change.', '',
             'All 256 existing validation anchors are included. Only the initial yaw-rate term changes. '
             'The native target remains exactly invariant. No inference, optimization, GPU, Spark call, fresh scene or physics run occurred.', '',
             '| Checkpoint | Original loss | Substituted loss | Original MAE, rad | Substituted MAE, rad | Episodes with worse loss |',
             '|---|---:|---:|---:|---:|---:|']
    for name, model in result['models'].items():
        row = model['groups']['all']
        lines.append(f"| {name} | {row['original_smooth_l1']:.8f} | {row['substituted_smooth_l1']:.8f} | "
                     f"{row['original_absolute_error_rad']['mean']:.8f} | {row['substituted_absolute_error_rad']['mean']:.8f} | "
                     f"{model['episodes_worsened']} of {model['episodes_total']} |")
    lines += ['', 'Loss is the existing SmoothL1 with beta 0.05 against the unchanged native-controller target. '
              'Lower MAE does not remove the increased loss. These summaries measure different parts of the error distribution.', '',
              '| Step-500 episode | Original loss | Substituted loss |', '|---|---:|---:|']
    for name, row in result['models']['step500']['groups'].items():
        if name.startswith('val-'):
            lines.append(f"| {name} | {row['original_smooth_l1']:.8f} | {row['substituted_smooth_l1']:.8f} |")
    audit = result['physical_signal_audit']
    lines += ['', 'The body audit corrects the earlier finite-difference interpretation. '
              'It reconstructs instantaneous planar heading rate from the saved body-forward axis and world angular velocity. '
              'The centered heading slope averages 0.1 seconds and is a different measurement.', '',
              '| Time, s | Raw world-Z rate | Exact instantaneous heading rate | Centered heading slope | Predicted initial rate |',
              '|---|---:|---:|---:|---:|']
    for row in audit['examples']:
        lines.append(f"| {row['t']:.2f} | {row['actual_yaw_rate']:.8f} | {row['exact_heading_derivative']:.8f} | "
                     f"{row['central_heading_slope_01s']:.8f} | {row['predicted_initial_yaw_rate']:.8f} |")
    lines += ['', f"Across {audit['sampled_frames']} sampled frames, the maximum absolute exact-versus-world-Z difference is "
              f"{audit['exact_minus_world_z_absolute_radps']['max']:.8f} rad/s. The maximum centered-slope-versus-exact difference is "
              f"{audit['central_slope_minus_exact_absolute_radps']['max']:.8f} rad/s. "
              'The 3D projection does not explain the large discrepancy here. Averaging, phase and dynamics remain unresolved.', '',
              'The script verifies the replay array and anchor hashes, reproduces saved original actions, checks the exact native-target invariance, '
              'and binds the body readout to its source script and six producer input hashes. '
              'It recomputes the body formula from retained vectors and the heading slopes from hash-verified local frames. '
              'It does not reopen the remote raw frame NPZ files.', '',
              'These correlated offline anchors do not establish a physical failure cause or a driving improvement. '
              'The rejected checkpoint remains ineligible. No runtime change or seventh optimizer run is justified by this result.', '',
              'Run from the repository root with `python3 .work/lab/measured_yaw_validation_ablation.py`. '
              'Use `--check` to recompute and compare both retained reports without writing them.', '',
              'The [JSON report](measured_yaw_validation_ablation.json) includes input and script hashes, all episode summaries, '
              'tail metrics and the five largest loss increases. '
              'The [script](measured_yaw_validation_ablation.py) uses only local files and NumPy. '
              'The [body producer](native_future_steering_readout2.py) and [body readout](native_future_steering_readout2.json) retain the corrected signal audit.', '',
              'Build the Lever preserved the executable comparison. Model the Domain kept the intervention on bound replay arrays. '
              'Attack the Premise tested the proposed improvement across every existing anchor. '
              'Explain the Number kept instantaneous rate, averaged heading change, offline loss and physical outcomes distinct.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Recompute and compare reports without writing.')
    args = parser.parse_args()
    result = calculate()
    outputs = {LAB / 'measured_yaw_validation_ablation.json': json.dumps(result, indent=2, allow_nan=False) + '\n',
               LAB / 'measured_yaw_validation_ablation.md': markdown(result)}
    for path, content in outputs.items():
        if args.check:
            require(path.read_text() == content, f'report differs: {path}')
        else:
            path.write_text(content)
    print(json.dumps({'checked' if args.check else 'written': [str(path.relative_to(ROOT)) for path in outputs],
                      'output_sha256': {path.name: sha(path) for path in outputs},
                      'step500_episodes_worsened': result['models']['step500']['episodes_worsened']}))


if __name__ == '__main__':
    main()
