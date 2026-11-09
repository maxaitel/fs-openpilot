#!/usr/bin/env python3
"""Replay the frozen 256 validation anchors after tangent training has finished."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

import fsop
from fsop.data.native_replay import NativeEpisode, NativeReplay, episode_paths
from fsop.model.build import load_fs
from fsop.sim.expert import T_IDXS
from fsop.train.lab_train import (
    batch_loss, configure, dataset_verification,
    plan_coherence, prove_policy_context, sha256, state_hashes, summarize_coherence,
)
from fsop.train.losses import PLAN_COL_W, PLAN_T_W, op_tangent_actions_from_op_plan

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--training', type=Path, required=True)
parser.add_argument('--selection', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
args = parser.parse_args()
signal.alarm(600)
started = time.monotonic()
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device(args.device)
run_path = args.training / 'run.json'
run = json.loads(run_path.read_text())
selection = json.loads(args.selection.read_text())
config = run['args']
protocol_path = args.training / 'selection_protocol.json'
protocol = json.loads(protocol_path.read_text())
history = [json.loads(line) for line in (args.training / 'history.jsonl').read_text().splitlines() if line]
assert [row['step'] for row in history] == list(range(1, 501)), 'Requires the completed 500-update run'
assert selection['kind'] == 'op_tangent_v1_selection' and selection['protocol'] == protocol
assert protocol['expected_steps'] == 500 and config['controller_mode'] == 'op_tangent_v1'
assert config['freeze_vision'] and config['history_context'] == 8 and not config['amp']
assert config['batch_size'] == 128 and config['val_samples'] == 256
assert run['validation_selection'] == selection['calibration']['identity']['validation_selection']
assert selection['calibration']['identity']['source_files_sha256'] == run['source']['files_sha256']
assert selection['calibration']['identity']['manifest_sha256'] == run['manifest_sha256']
for path in (run_path, args.training / 'history.jsonl', protocol_path):
    matching = [digest for name, digest in selection['input_sha256'].items()
                if Path(name).resolve() == path.resolve()]
    assert matching == [sha256(path)], f'Selection does not bind {path}'
source_root = Path(fsop.__file__).resolve().parent.parent
for name, digest in run['source']['files_sha256'].items():
    assert sha256(source_root / name) == digest, f'Runtime source changed: {name}'
assert sha256(args.training / 'source.zip') == run['source']['archive_sha256']
assert sha256(config['manifest']) == run['manifest_sha256']
verification = dataset_verification(config['manifest'], config.get('dataset_verification'))
assert verification == run['dataset_verification']
manifest_paths = set(episode_paths(manifest=config['manifest']))
candidates = {row['step']: row for row in selection['candidates']}
checkpoints = {name: Path(candidates[step]['checkpoint'])
               for name, step in (('step0', 0), ('step500', 500))}
for name, step in (('step0', 0), ('step500', 500)):
    assert sha256(checkpoints[name]) == candidates[step]['checkpoint_sha256']
args.out.mkdir(parents=True, exist_ok=False)
report = {
    'schema_version': 1,
    'scope': 'Paired descriptive replay of existing validation anchors. No fresh scenes, optimization, selection, or physical outcome claim.',
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'selection_status': selection['status'],
    'step500_eligible': candidates[500]['eligible'],
    'decision': 'Localize acceleration endpoint/secant residual before choosing a new experiment. This report does not change frozen thresholds.',
    'input_sha256': {str(path): sha256(path) for path in
                     (run_path, args.selection, protocol_path, args.training / 'source.zip', Path(__file__))},
    'runtime_source_sha256': run['source']['files_sha256'],
    'manifest_sha256': run['manifest_sha256'],
    'dataset_verification': verification,
    'checkpoints': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in checkpoints.items()},
    'execution': {'device': str(device), 'threads': 4, 'precision': 'float32', 'tf32': False,
                  'feature_cache': False, 'episode_batch_size': 32,
                  'deadline_seconds': 600, 'repeats': 1, 'repeatability_estimate': None},
    'models': {},
}
episodes = []
anchors = []
frozen_meta = {row['path']: row['meta_sha256'] for row in run['episodes']}
for ei, group in enumerate(run['validation_selection']):
    path = Path(group['path'])
    assert path.resolve() in manifest_paths
    assert sha256(path / 'meta.json') == frozen_meta[group['path']]
    ep = NativeEpisode(path)
    assert ep.meta['split'] == 'val' and len(group['frames']) == 32
    assert len(set(group['frames'])) == 32 and set(group['frames']).issubset(ep.anchors)
    episodes.append(ep)
    for frame in group['frames']:
        tick = int(ep.ticks[frame])
        anchors.append({'index': len(anchors), 'episode': ei, 'case': ep.meta['id'],
                        'discipline': ep.meta['discipline'], 'path': str(path), 'frame': frame, 'tick': tick,
                        'usable_history_frames': min(frame, 8), 'state': ep.states[tick].tolist(),
                        'teacher_action': ep.controls[tick].tolist()})
assert len(episodes) == 8 and len(anchors) == 256
dataset = NativeReplay(episodes, 'val')
dataset.index = [(row['episode'], row['frame']) for row in anchors]
assert len(dataset) == 256
report['episodes'] = [{'path': str(ep.path), 'meta_sha256': sha256(ep.path / 'meta.json'),
                       'arrays_sha256': ep.meta['arrays_sha256']} for ep in episodes]
report['validation_selection'] = run['validation_selection']
report['anchor_inventory_sha256'] = hashlib.sha256(json.dumps(anchors, sort_keys=True).encode()).hexdigest()
print('Verified completed run, sources, checkpoints, eight episodes and 256 anchors', flush=True)
torch.manual_seed(config['seed'])
model = load_fs(checkpoints['step0']).to(device).eval()
if config['channels_last']:
    model.to(memory_format=torch.channels_last)
configure(model, freeze_vision=True)
report['history_context_proof'] = prove_policy_context(model)
assert report['history_context_proof'] == run['history_context_proof']
frozen = state_hashes(model.state_dict())
assert frozen == run['baseline_state_sha256']
observed = default_collate([dataset[i] for i in range(256)])
targets = observed['plan'].numpy()
arrays = {'target': targets, 'times_s': T_IDXS.copy(),
          'observed_vx_vy_yaw_rate': np.stack([observed[k].numpy() for k in ('speed', 'lateral_speed', 'yaw_rate')], axis=-1),
          'teacher_actions': np.stack([observed[k].numpy() for k in ('steer_target', 'accel_target')], axis=-1),
          'native_future_states': np.stack([episodes[row['episode']].states[row['tick']:row['tick']+201]
                                            for row in anchors])}
observed.pop('img')
observed.pop('big_img')
examples = [dict(speed=row[0], lateral_speed=row[1], yaw_rate=row[2]) for row in arrays['observed_vx_vy_yaw_rate']]
target_coherence = plan_coherence(targets, examples)
assert summarize_coherence(target_coherence) == selection['calibration']['validation']['target_coherence']
report['native_coherence'] = summarize_coherence(target_coherence)
n = int(np.searchsorted(T_IDXS, 1.0, side='right'))
dt = np.diff(T_IDXS[:n])
arrays['interval_dt_s'] = dt
target_secant = np.diff(targets[:, :n, 3:5], axis=1) / dt[None, :, None]
target_endpoints = np.stack((targets[:, :n-1, 6:8], targets[:, 1:n, 6:8]), axis=2)
arrays['native_acceleration_secant_mps2'] = target_secant
arrays['native_acceleration_endpoint_residual_mps2'] = target_endpoints - target_secant[:, :, None, :]
small = {k: observed[k].to(device) for k in ('speed', 'lateral_speed', 'vehicle_wheelbase', 'vehicle_max_brake', 'vehicle_max_accel')}
stats = lambda values: {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)), 'max': float(np.max(values))}
for name, step in (('step0', 0), ('step500', 500), ('native', None)):
    if step is not None:
        if step != 0:
            payload = torch.load(checkpoints[name], map_location='cpu', weights_only=False)
            assert payload['step'] == step and payload['config'] == config
            model.load_state_dict(payload['model'], strict=True)
            del payload
        assert state_hashes(model.state_dict()) == frozen
        predictions, uncertainties, episode_parts = [], [], []
        with torch.no_grad():
            for ei in range(8):
                ids = list(range(ei * 32, (ei + 1) * 32))
                _, parts, output = batch_loss(model, dataset, ids, device, config['w_steer'], config['w_long'],
                    config['history_chunk_size'], amp=False, channels_last=config['channels_last'],
                    history_context=8, w_plan=config['w_plan'],
                    return_outputs=True, controller_mode='op_tangent_v1')
                predictions.append(output['plan_mu'].cpu().numpy())
                uncertainties.append(output['plan_logstd'].cpu().numpy())
                episode_parts.append({key: float(value) for key, value in parts.items()})
        mu, logstd = np.concatenate(predictions), np.concatenate(uncertainties)
        arrays[name + '_logstd'] = logstd
        weights = PLAN_T_W.numpy()[:, None] * PLAN_COL_W.numpy()[None, :]
        arrays[name + '_nll_mu_gradient_scale'] = weights[None] * np.exp(-np.clip(logstd, -7, 7)) / weights.sum()
        expected = candidates[step]['validation']
        reproduced = plan_coherence(mu, examples)
        checks = []
        for ei in range(8):
            actual = summarize_coherence(reproduced[ei*32:(ei+1)*32])
            old = expected['episodes'][ei]
            assert Path(old['path']) == episodes[ei].path
            checks.extend(np.isclose(episode_parts[ei][key], old[key], rtol=1e-5, atol=1e-5)
                          for key in episode_parts[ei])
            checks.extend(np.isclose(value, old['coherence'][metric][stat], rtol=1e-5, atol=1e-5)
                          for metric, values in actual.items() for stat, value in values.items())
        assert all(checks), f'{name} does not reproduce recorded losses/coherence within fixed tolerances'
        report['models'][name] = {'replay_matches_recorded_validation': True,
            'comparison_atol': 1e-5, 'comparison_rtol': 1e-5,
            'coherence': summarize_coherence(reproduced), 'episode_losses': episode_parts}
        report['models'][name]['nll_near_knot_columns'] = [
            {'knot': knot, 'time_s': float(T_IDXS[knot]), 'column': column,
             'mean_abs_native_error': float(np.abs(mu[:,knot,column]-targets[:,knot,column]).mean()),
             'mean_logstd': float(logstd[:,knot,column].mean()),
             'mean_nll_mu_gradient_scale': float(arrays[name+'_nll_mu_gradient_scale'][:,knot,column].mean())}
            for knot in range(n) for column in (3,4,6,7)]
    else:
        mu = targets
        report['models'][name] = {'coherence': report['native_coherence']}
    arrays[name + '_mu'] = mu
    secant = np.diff(mu[:, :n, 3:5], axis=1) / dt[None, :, None]
    endpoints = np.stack((mu[:, :n-1, 6:8], mu[:, 1:n, 6:8]), axis=2)
    residual = endpoints - secant[:, :, None, :]
    accel_error = endpoints - target_endpoints
    velocity_secant_error = secant - target_secant
    np.testing.assert_allclose(residual - arrays['native_acceleration_endpoint_residual_mps2'],
                               accel_error - velocity_secant_error[:, :, None, :], rtol=1e-6, atol=1e-5)
    arrays[name + '_acceleration_endpoint_residual_mps2'] = residual
    arrays[name + '_acceleration_endpoint_error_to_native_mps2'] = accel_error
    arrays[name + '_velocity_secant_error_to_native_mps2'] = velocity_secant_error
    lengths = np.linalg.norm(residual, axis=-1)
    maximum = lengths.max(axis=(1, 2))
    with torch.no_grad():
        action = op_tangent_actions_from_op_plan(torch.from_numpy(mu).to(device), **{
            'vx': small['speed'], 'vy': small['lateral_speed'], 'wheelbase': small['vehicle_wheelbase'],
            'max_brake': small['vehicle_max_brake'], 'max_accel': small['vehicle_max_accel']})
    actions = torch.stack(action, dim=-1).cpu().numpy()
    arrays[name + '_actions'] = actions
    speed = np.linalg.norm(mu[:, :, 3:5], axis=-1)
    tangent = (mu[:, :, 3:5] * mu[:, :, 6:8]).sum(axis=-1) / np.maximum(speed, .001)
    hi = int(np.searchsorted(T_IDXS, .3))
    alpha = (.3 - T_IDXS[hi-1]) / (T_IDXS[hi] - T_IDXS[hi-1])
    target_speed = speed[:, hi-1] * (1-alpha) + speed[:, hi] * alpha
    target_accel = tangent[:, hi-1] * (1-alpha) + tangent[:, hi] * alpha
    speed_feedback = 1.5 * (target_speed - np.linalg.norm(arrays['observed_vx_vy_yaw_rate'][:, :2], axis=-1))
    raw_long = speed_feedback + .5 * target_accel
    arrays[name + '_longitudinal_components'] = np.stack((target_speed, target_accel, speed_feedback, raw_long), axis=-1)
    velocity_direction = targets[:, :n-1, 3:5] + targets[:, 1:n, 3:5]
    direction_norm = np.linalg.norm(velocity_direction, axis=-1)
    direction = velocity_direction / np.maximum(direction_norm[:, :, None], 1e-12)
    arrays[name + '_residual_tangent_component_mps2'] = (residual * direction[:, :, None, :]).sum(axis=-1)
    arrays[name + '_residual_normal_component_mps2'] = residual[..., 1] * direction[:, :, None, 0] - residual[..., 0] * direction[:, :, None, 1]
    arrays['native_direction_defined'] = direction_norm > 1e-12
    groups = {'all': np.arange(256), 'history_lt8': np.array([i for i,r in enumerate(anchors) if r['usable_history_frames'] < 8]),
              'history_8': np.array([i for i,r in enumerate(anchors) if r['usable_history_frames'] == 8])}
    groups.update({ep.meta['id']: np.arange(ei*32, (ei+1)*32) for ei,ep in enumerate(episodes)})
    report['models'][name]['groups'] = {key: {'anchors': len(ids), 'acceleration_secant': stats(maximum[ids])}
                                      for key,ids in groups.items() if len(ids)}
    report['models'][name]['intervals'] = [dict(interval=i, lo_s=float(T_IDXS[i]), hi_s=float(T_IDXS[i+1]),
        dt_s=float(dt[i]), residual=stats(lengths[:, i].max(axis=-1))) for i in range(n-1)]
    error = actions - arrays['teacher_actions']
    report['models'][name]['teacher_action_comparison'] = {
        'steer_mae_rad': float(np.abs(error[:,0]).mean()), 'longitudinal_mae_mps2': float(np.abs(error[:,1]).mean()),
        'steer_smooth_l1': float(F.smooth_l1_loss(torch.from_numpy(actions[:,0]), observed['steer_target'], beta=.05)),
        'longitudinal_smooth_l1': float(F.smooth_l1_loss(torch.from_numpy(actions[:,1]), observed['accel_target'], beta=.05))}
    for index,row in enumerate(anchors):
        interval, endpoint = np.unravel_index(np.argmax(lengths[index]), lengths[index].shape)
        row[name] = {'worst_acceleration_secant_mps2': float(maximum[index]), 'interval': int(interval),
            'endpoint': int(endpoint), 'knot': int(interval+endpoint), 'lo_s': float(T_IDXS[interval]),
            'hi_s': float(T_IDXS[interval+1]), 'residual_xy': residual[index,interval,endpoint].tolist(),
            'native_residual_xy': arrays['native_acceleration_endpoint_residual_mps2'][index,interval,endpoint].tolist(),
            'acceleration_endpoint_error_xy': accel_error[index,interval,endpoint].tolist(),
            'velocity_secant_error_xy': velocity_secant_error[index,interval].tolist(),
            'action': actions[index].tolist(), 'longitudinal_components': arrays[name+'_longitudinal_components'][index].tolist()}
        if step is not None:
            row[name]['endpoint_acceleration_logstd_xy'] = logstd[index,interval+endpoint,6:8].tolist()
            row[name]['interval_velocity_logstd_xy'] = logstd[index,interval:interval+2,3:5].tolist()
    print(name + ' replay and interval decomposition complete', flush=True)
report['array_contract'] = {
    '*_mu and *_logstd': '[anchor,33,15], OP coordinates',
    'native_future_states': '[anchor,201 native ticks,9 state columns], x,y,yaw,vx,vy,yaw_rate,steer,accel,timestamp. Absolute native timestamps are retained.',
    '*_acceleration_endpoint_residual_mps2': '[anchor,10 intervals,2 endpoints,xy]',
    '*_acceleration_endpoint_error_to_native_mps2': '[anchor,10 intervals,2 endpoints,xy]',
    '*_velocity_secant_error_to_native_mps2': '[anchor,10 intervals,xy]',
    'decomposition': 'model_residual - native_residual = acceleration_endpoint_error - velocity_secant_error',
    '*_nll_mu_gradient_scale': 'Absolute NLL derivative coefficient away from zero residual. Includes time/column normalization, excludes batch averaging and global w_plan.',
    '*_actions': '[anchor, steer radians and clipped longitudinal request m/s^2], before mission layer',
    '*_longitudinal_components': '[anchor, target speed, target tangent acceleration, speed feedback, unclipped command]',
    '*_residual_*_component_mps2': 'Projection onto native interval-average velocity direction and its left normal in OP xy. Consult native_direction_defined at zero speed.',
}
report['limitations'] = [
    'Native future and teacher controls are offline labels only. The model receives camera features, desire and traffic inputs.',
    'Teacher requests and native kinematic acceleration have different contracts. Their observed gap is not proof of the cause of model incoherence.',
    'Uncertainty and residual correlation is descriptive. No optimizer gradient conflict or causality is established.',
    'One deterministic replay has no repeatability or population uncertainty estimate. Correlated anchors are not independent physical trials.',
    'All groups and intervals are reported. No new thresholds are fitted and this script cannot promote a checkpoint.',
]
for name,path in checkpoints.items():
    assert sha256(path) == report['checkpoints'][name]['sha256']
for name,digest in run['source']['files_sha256'].items():
    assert sha256(source_root/name) == digest
report['ended_utc'] = datetime.now(timezone.utc).isoformat()
report['elapsed_seconds'] = time.monotonic() - started
np.savez_compressed(args.out/'arrays.npz', **arrays)
(args.out/'anchors.json').write_text(json.dumps(anchors, indent=2, allow_nan=False)+'\n')
report['output_sha256'] = {name: sha256(args.out/name) for name in ('arrays.npz', 'anchors.json')}
(args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps({'out':str(args.out), 'elapsed_seconds':report['elapsed_seconds'],
    'groups': {name: value['groups'] for name,value in report['models'].items()}}, indent=2))
