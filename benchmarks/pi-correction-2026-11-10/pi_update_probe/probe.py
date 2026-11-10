"""Throwaway TRAIN-only experiment with one independent AdamW update per cell."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from fsop.data.native_replay import NativeEpisode, NativeReplay, episode_paths
from fsop.model.build import load_fs
from fsop.sim.expert import T_IDXS
from fsop.train.lab_train import RecoverySampler, batch_loss, configure, dataset_verification, prove_policy_context, sha256, state_hashes
from fsop.train.losses import PLAN_COL_W, PLAN_T_W, native_relative_acceleration_loss, tangent_motion_from_op_plan
from isaac_lab_fsop.evaluate import model_digest, source_inventory


MODELS = {
    'latest': {'path': '/work/probes/pi_models/latest_pi_train.pt', 'sha256': '48db741c8a686b2e635ae7a81b4e4c08d79fbfbd9f4dc4c163260da6f93f0cfb'},
    'vision': {'path': '/work/probes/pi_models/vision_pi.pt', 'sha256': 'ab5d18e2b6984192ad82c2f5d288c64c43b2ae2132e29e1901053f5149b6cc02'},
}
ARMS = {
    'plan_consistency': {'w_plan': 1., 'w_consistency': 1., 'w_steer': 0., 'w_long': 0.},
    'plan': {'w_plan': 1., 'w_consistency': 0., 'w_steer': 0., 'w_long': 0.},
    'consistency': {'w_plan': 0., 'w_consistency': 1., 'w_steer': 0., 'w_long': 0.},
    'action': {'w_plan': 0., 'w_consistency': 0., 'w_steer': 1., 'w_long': 1.},
    'steer': {'w_plan': 0., 'w_consistency': 0., 'w_steer': 1., 'w_long': 0.},
    'longitudinal': {'w_plan': 0., 'w_consistency': 0., 'w_steer': 0., 'w_long': 1.},
}
MANIFEST = Path('/work/data/native/lab_v7/manifest.json')
MANIFEST_SHA = 'db7150b60fdf50e861c6ca003ae844b90ff398fd7b244c9201b7de9d3af5b309'
RECIPE = '/work/docs/isaac-lab-training/phase_recovery_recipe4.json'
GATES = '/work/data/native/lab_v7/recovery_gates'
CONFIG = Path('/work/benchmarks/pi-correction-2026-11-10/pi_trial1.yaml')
PHASES = ['steady_right', 'right_to_left', 'steady_left', 'recovery']
COLS = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'accel_x', 'accel_y', 'accel_z', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_train():
    assert sha256(MANIFEST) == MANIFEST_SHA
    verification = dataset_verification(MANIFEST)
    eps = [NativeEpisode(path) for path in episode_paths(manifest=MANIFEST)
           if json.loads((path / 'meta.json').read_text())['split'] == 'train']
    return NativeReplay(eps, 'train'), verification


def sampler_for(ds, model):
    return RecoverySampler(ds, RECIPE, GATES, 128, 16, model['sha256'], provenance_mode='per_case_v2')


def anchor(ds, index):
    ei, frame = ds.index[int(index)]
    ep = ds.episodes[ei]
    tick = int(ep.ticks[frame])
    return {'index': int(index), 'case': ep.meta['id'], 'frame': frame, 'tick': tick,
            'time_s': float(ep.states[tick, 8]), 'meta_sha256': sha256(ep.path / 'meta.json')}


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    ds, verification = load_train()
    sampler = sampler_for(ds, MODELS['latest'])
    by_episode = {}
    for i, (ei, frame) in enumerate(ds.index):
        by_episode.setdefault(ei, []).append(i)
    monitor = []
    clean = [ei for ei, ep in enumerate(ds.episodes) if ep.meta['discipline'] == 'skidpad'
             and not ep.meta.get('collection_policy') and ei in by_episode]
    clean = sorted(clean, key=lambda ei: ds.episodes[ei].meta['id'])[:4]
    assert len(clean) == 4
    for phase in PHASES[:3]:
        for ei in clean:
            ep = ds.episodes[ei]
            laps = ep.meta['summary']['route']['laps']
            assert [lap['circle'] for lap in laps] == ['right', 'right', 'left', 'left']
            desired = {'steady_right': (laps[1]['start_time'] + laps[1]['finish_time']) / 2,
                       'right_to_left': laps[1]['finish_time'],
                       'steady_left': (laps[2]['start_time'] + laps[2]['finish_time']) / 2}[phase]
            index = min(by_episode[ei], key=lambda i: (abs(float(ep.states[int(ep.ticks[ds.index[i][1]]), 8]) - desired), i))
            monitor.append({**anchor(ds, index), 'phase': phase, 'route_event_time_s': desired,
                            'route_laps': laps, 'rule': 'nearest qualified anchor to recorded lap event or midpoint'})
    recovery_groups = [g for g in sampler.groups if ds.episodes[g['episode']].meta['discipline'] == 'skidpad']
    for group in sorted(recovery_groups, key=lambda g: g['case_id']):
        index = group['indices'][0]
        monitor.append({**anchor(ds, index), 'phase': 'recovery', 'takeover_tick': group['takeover_tick'],
                        'collection_policy': ds.episodes[group['episode']].meta['collection_policy'],
                        'rule': 'first qualified anchor at or after recorded teacher takeover'})
    assert len(monitor) == 16 and len({row['index'] for row in monitor}) == 16
    excluded = {row['index'] for row in monitor}
    sampler.other = np.array([i for i in sampler.other if int(i) not in excluded], dtype=np.int64)
    for group in sampler.groups:
        group['indices'] = [i for i in group['indices'] if i not in excluded]
        group['frames'] = [ds.index[i][1] for i in group['indices']]
        group['ticks'] = [int(ds.episodes[group['episode']].ticks[frame]) for frame in group['frames']]
        assert group['indices']
    sampler.recovery_indices -= excluded
    rng = np.random.default_rng(0)
    batches = [sampler.draw(rng).tolist() for _ in range(4)]
    assert all(not (set(ids) & excluded) and sum(i in sampler.recovery_indices for i in ids) == 16 for ids in batches)
    for model in MODELS.values():
        assert sha256(model['path']) == model['sha256']
    protocol = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256(__file__),
        'source_sha256': source_inventory(), 'manifest_sha256': sha256(MANIFEST), 'verification': verification,
        'config_path': str(CONFIG), 'config_sha256': sha256(CONFIG), 'config_text': CONFIG.read_text(),
        'models': MODELS, 'arms': ARMS, 'batches': batches, 'batch_inventory': [[anchor(ds, i) for i in ids] for ids in batches],
        'monitor': monitor, 'sampler': sampler.protocol(), 'monitor_exclusion': sorted(excluded),
        'sampler_rule': 'Remove monitor indices from recovery groups and other pool before seed-0 draws. Preserve 16 recovery and 112 other draws per batch.',
        'optimizer': {'class': 'torch.optim.AdamW', 'lr': 0.0001 * 0.3, 'weight_decay': 1e-4,
                      'betas': [0.9, 0.999], 'eps': 1e-8, 'amsgrad': False, 'grad_clip': 1.0},
        'expected_updates': 48, 'seed': 0, 'history_context': 8, 'history_chunk_size': 64,
        'precision': 'FP32, TF32 disabled, eval mode, no feature cache',
        'claim': 'Measure local TRAIN action and plan error changes caused by each objective under one actual production-config AdamW update. This does not establish closed-loop improvement.',
        'reset': 'Load exact initial state before every batch-arm cell. Fresh AdamW each cell. Frozen vision and auxiliary tensors must retain their initial hashes.',
        'limits': {'artifact_bytes': 100 * 1024**2, 'minimum_free_bytes': 25 * 1024**3, 'timeout_seconds': 900},
        'scope': 'Only TRAIN native arrays. No validation, fresh cases, checkpoints, promotion, or sequential learning.',
        'monitor_limitations': 'Monitor anchor indices are disjoint from fit batches. Their history and future windows may overlap fit windows, and they may share episodes. Monitor results do not estimate independent generalization.',
    }
    write_json(out / 'protocol.json', protocol)
    print(json.dumps({'prepared': str(out), 'protocol_sha256': sha256(out / 'protocol.json'),
                      'monitor': [{k: row[k] for k in ('phase', 'case', 'frame', 'tick')} for row in monitor]}), flush=True)


def motion(mu, observed, wheelbase):
    steer, raw = tangent_motion_from_op_plan(mu, observed[:, 0], observed[:, 1], wheelbase)
    velocity, acceleration = mu[:, :, 3:5], mu[:, :, 6:8]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    tangent = (velocity * acceleration).sum(-1) / speed.clamp_min(0.001)
    hi = int(np.searchsorted(T_IDXS, 0.3))
    fraction = (0.3 - T_IDXS[hi - 1]) / (T_IDXS[hi] - T_IDXS[hi - 1])
    interp = lambda x: x[:, hi - 1] * (1 - fraction) + x[:, hi] * fraction
    return torch.stack((steer, raw, interp(speed), interp(tangent)), dim=-1)


def measure(parts, output, wheelbase, selection=None):
    mu, logstd, target, observed = [output[k].detach() for k in ('plan_mu', 'plan_logstd', 'coherence_targets', 'coherence_observed')]
    if selection is not None:
        mu, logstd, target, observed, wheelbase = [x[selection] for x in (mu, logstd, target, observed, wheelbase)]
    action = motion(mu, observed, wheelbase)
    native_action = motion(target, observed, wheelbase)
    error = action - native_action
    weights = PLAN_T_W.to(mu)[None, :, None] * PLAN_COL_W.to(mu)[None, None, :]
    effective = weights * torch.exp(-logstd.clamp(-7, 7)) / weights.sum()
    n = int(np.searchsorted(T_IDXS, 1., side='right'))
    metrics = {
        'plan_mae_by_column': dict(zip(COLS, (mu - target).abs().mean((0, 1)).cpu().tolist())),
        'near_plan_mae_by_column': dict(zip(COLS, (mu[:, :n] - target[:, :n]).abs().mean((0, 1)).cpu().tolist())),
        'action_mae': dict(zip(('steer_rad', 'raw_accel_mps2', 'target_speed_mps', 'tangent_accel_mps2'), error.abs().mean(0).cpu().tolist())),
        'action_signed_error': dict(zip(('steer_rad', 'raw_accel_mps2', 'target_speed_mps', 'tangent_accel_mps2'), error.mean(0).cpu().tolist())),
        'effective_nll_mu_weight_by_column': dict(zip(COLS, effective.mean((0, 1)).cpu().tolist())),
        'near_effective_nll_mu_weight_by_column': dict(zip(COLS, effective[:, :n].mean((0, 1)).cpu().tolist())),
        'logstd_mean_by_column': dict(zip(COLS, logstd.mean((0, 1)).cpu().tolist())),
        'consistency': float(native_relative_acceleration_loss(mu, target)),
    }
    arrays = {k: x.cpu().numpy() for k, x in {'mu': mu, 'logstd': logstd, 'native_target': target,
                                            'observed': observed, 'wheelbase': wheelbase,
                                            'desired_motion': action, 'native_desired_motion': native_action}.items()}
    if selection is None:
        metrics['losses'] = {k: float(v.detach()) for k, v in parts.items()}
    return metrics, arrays


def cosine(left, right):
    denominator = float(left.norm() * right.norm())
    return float(torch.dot(left, right)) / denominator if denominator else None


def nullspace(output):
    mu, target = output['plan_mu'].detach().double(), output['coherence_targets'].detach().double()
    base = native_relative_acceleration_loss(mu, target)
    yaw_pos = mu.clone()
    yaw_pos[:, :, [0, 1, 2, 11, 14]] += 0.25
    offset = mu.clone()
    offset[:, :, 3:5] += torch.tensor([0.5, -0.25], dtype=mu.dtype, device=mu.device)
    values = {'base': float(base), 'yaw_position_perturbed': float(native_relative_acceleration_loss(yaw_pos, target)),
              'constant_velocity_offset': float(native_relative_acceleration_loss(offset, target))}
    assert values['base'] == values['yaw_position_perturbed']
    assert abs(values['base'] - values['constant_velocity_offset']) < 1e-10
    values['definition'] = 'Residual uses only planar acceleration and differences of planar velocity. Position and yaw are absent. A constant planar velocity offset cancels between adjacent knots.'
    return values


def run(out):
    protocol_path = out / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['models'] == MODELS and protocol['arms'] == ARMS
    assert not (out / 'report.json').exists() and not (out / 'rows.jsonl').exists()
    assert sha256(__file__) == protocol['script_sha256'] and source_inventory() == protocol['source_sha256']
    assert sha256(CONFIG) == protocol['config_sha256']
    assert shutil.disk_usage(out).free >= protocol['limits']['minimum_free_bytes']
    ds, verification = load_train()
    assert verification == protocol['verification']
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    started = time.monotonic()
    monitor_ids = [row['index'] for row in protocol['monitor']]
    selections = {phase: [i for i, row in enumerate(protocol['monitor']) if row['phase'] == phase] for phase in PHASES}
    rows, arrays, models = [], {}, {}

    def forward(model, ids, weights):
        return batch_loss(model, ds, ids, device, chunk_size=64, history_context=8,
                          return_outputs=True, controller_mode='op_tangent_pi_v1',
                          action_target_mode='desired_motion_v1', **weights)

    def save_measure(prefix, parts, output, ids, monitor=False):
        wheelbase = torch.tensor([ds.episodes[ds.index[int(i)][0]].wheelbase for i in ids], device=device)
        metrics, data = measure(parts, output, wheelbase)
        arrays.update({f'{prefix}_{key}': value for key, value in data.items()})
        if monitor:
            metrics['phases'] = {phase: measure(parts, output, wheelbase, selection)[0] for phase, selection in selections.items()}
        return metrics

    for name, spec in MODELS.items():
        assert sha256(spec['path']) == spec['sha256']
        sampler_for(ds, spec)
        model = load_fs(spec['path']).to(device).eval()
        configure(model, freeze_vision=True)
        context = prove_policy_context(model)
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        frozen = state_hashes(initial)
        initial_digest = model_digest(model)
        params = [p for p in model.parameters() if p.requires_grad]
        policy_params = [p for p in model.base.policy.parameters() if p.requires_grad]
        assert {id(p) for p in params} == {id(p) for p in policy_params}
        assert len(params) == 50 and sum(p.numel() for p in params) == 6949412
        initial_vector = torch.cat([p.detach().cpu().reshape(-1) for p in params])
        models[name] = {'initial_model_digest': initial_digest, 'frozen_state_sha256': frozen,
                        'trainable_tensors': len(params), 'trainable_parameters': sum(p.numel() for p in params),
                        'total_parameters': sum(p.numel() for p in model.parameters()), 'context': context}
        with torch.no_grad():
            _, parts, output = forward(model, monitor_ids, ARMS['plan_consistency'])
            monitor_before = save_measure(f'{name}_monitor_before', parts, output, monitor_ids, True)
            models[name]['nullspace'] = nullspace(output)
        for bi, ids in enumerate(protocol['batches']):
            deltas = {}
            for arm, weights in ARMS.items():
                cell_started = time.monotonic()
                model.load_state_dict(initial)
                assert model_digest(model) == initial_digest
                optimizer = torch.optim.AdamW([{'params': policy_params, 'lr': 0.0001 * 0.3}], weight_decay=1e-4)
                optimizer.zero_grad(set_to_none=True)
                loss, parts, output = forward(model, ids, weights)
                before_key = f'{name}_{bi}_fit_before'
                if arm == next(iter(ARMS)):
                    before = save_measure(before_key, parts, output, ids)
                loss.backward()
                raw_grad = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().cpu().reshape(-1) for p in params])
                assert torch.isfinite(raw_grad).all()
                norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in params)
                assert state_hashes(model.state_dict()) == frozen
                delta = torch.cat([p.detach().cpu().reshape(-1) for p in params]) - initial_vector
                assert torch.isfinite(delta).all() and float(delta.norm()) > 0
                deltas[arm] = delta
                steps = sorted({int(state['step']) for state in optimizer.state.values()})
                assert steps == [1]
                with torch.no_grad():
                    _, after_parts, after_output = forward(model, ids, weights)
                    after = save_measure(f'{name}_{bi}_{arm}_fit_after', after_parts, after_output, ids)
                    _, monitor_parts, monitor_output = forward(model, monitor_ids, weights)
                    monitor_after = save_measure(f'{name}_{bi}_{arm}_monitor_after', monitor_parts, monitor_output, monitor_ids, True)
                row = {'model': name, 'batch': bi, 'arm': arm, 'objective_before': float(loss.detach()),
                       'fit_before_arrays_key': before_key,
                       'fit_before': before, 'fit_after': after, 'monitor_before': monitor_before, 'monitor_after': monitor_after,
                       'gradient_norm_before_clip': norm, 'clip_multiplier': min(1., 1. / (norm + 1e-6)),
                       'parameter_delta_norm': float(delta.norm()), 'parameter_delta_max_abs': float(delta.abs().max()),
                       'delta_cosine_negative_raw_gradient': cosine(delta, -raw_grad),
                       'initial_model_digest': initial_digest, 'after_model_digest': model_digest(model),
                       'frozen_state_unchanged': True, 'optimizer_steps': steps, 'optimizer_state_tensors': len(optimizer.state),
                       'optimizer_defaults': optimizer.defaults, 'seconds': time.monotonic() - cell_started}
                rows.append(row)
                with (out / 'rows.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                print(json.dumps({k: row[k] for k in ('model', 'batch', 'arm', 'gradient_norm_before_clip', 'parameter_delta_norm', 'seconds')}), flush=True)
                assert time.monotonic() - started < protocol['limits']['timeout_seconds']
                del optimizer, loss, parts, output, after_parts, after_output, monitor_parts, monitor_output, raw_grad
            models[name].setdefault('delta_cosines_by_batch', {})[bi] = {left: {right: cosine(x, deltas[right]) for right in ARMS} for left, x in deltas.items()}
        assert sha256(spec['path']) == spec['sha256']
        del model, initial, params, policy_params, initial_vector, deltas
    assert source_inventory() == protocol['source_sha256']
    assert len(rows) == protocol['expected_updates']
    np.savez_compressed(out / 'arrays.npz', **arrays)
    report = {'scope': protocol['scope'], 'claim': protocol['claim'], 'protocol_sha256': sha256(protocol_path),
              'script_sha256': sha256(__file__), 'rows_sha256': sha256(out / 'rows.jsonl'),
              'arrays_sha256': sha256(out / 'arrays.npz'), 'updates': len(rows), 'models': models,
              'all_source_files_unchanged': True, 'elapsed_seconds': time.monotonic() - started}
    write_json(out / 'report.json', report)
    assert sum(path.stat().st_size for path in out.iterdir() if path.is_file()) < protocol['limits']['artifact_bytes']
    print(json.dumps({'completed_updates': len(rows), 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


p = argparse.ArgumentParser()
p.add_argument('mode', choices=('prepare', 'run'))
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
{'prepare': prepare, 'run': run}[a.mode](a.out)
