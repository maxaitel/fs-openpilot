"""Frozen TRAIN-only, independent first-step learning-rate probe."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import scipy
import torch

from fsop.data.native_replay import NativeEpisode, NativeReplay, episode_paths
from fsop.model.build import load_fs
from fsop.sim.expert import T_IDXS
from fsop.train.lab_train import batch_loss, configure, dataset_verification, prove_policy_context, sha256, state_hashes
from fsop.train.losses import tangent_motion_from_op_plan
from isaac_lab_fsop.evaluate import model_digest, source_inventory

MANIFEST = Path('/work/data/native/lab_v7/manifest.json')
WEIGHTS = {'w_plan': 1., 'w_consistency': 1., 'w_steer': 0., 'w_long': 0.}
RATES = [3e-6, 3e-7, 3e-8]
PHASES = ['steady_right', 'right_to_left', 'steady_left', 'recovery']
NAMES = ['steer_rad', 'raw_accel_mps2', 'target_speed_mps', 'tangent_accel_mps2']


def motion(mu, observed, wheelbase):
    steer, raw = tangent_motion_from_op_plan(mu, observed[:, 0], observed[:, 1], wheelbase)
    velocity, acceleration = mu[:, :, 3:5], mu[:, :, 6:8]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    tangent = (velocity * acceleration).sum(-1) / speed.clamp_min(.001)
    hi = int(np.searchsorted(T_IDXS, .3))
    fraction = (.3 - T_IDXS[hi - 1]) / (T_IDXS[hi] - T_IDXS[hi - 1])
    interp = lambda x: x[:, hi - 1] * (1 - fraction) + x[:, hi] * fraction
    return torch.stack((steer, raw, interp(speed), interp(tangent)), dim=-1)


def measure(parts, output, wheelbase, selections):
    mu, logstd, target, observed = [output[k].detach() for k in ('plan_mu', 'plan_logstd', 'coherence_targets', 'coherence_observed')]
    action, native = motion(mu, observed, wheelbase), motion(target, observed, wheelbase)
    error = (action - native).abs()
    metrics = {'losses': {k: float(v.detach()) for k, v in parts.items()},
               'action_mae': dict(zip(NAMES, error.mean(0).cpu().tolist())),
               'phases': {p: dict(zip(NAMES, error[ids].mean(0).cpu().tolist())) for p, ids in selections.items()}}
    arrays = {k: v.cpu().numpy() for k, v in {'mu': mu, 'logstd': logstd, 'native_target': target,
              'observed': observed, 'wheelbase': wheelbase, 'desired_motion': action, 'native_desired_motion': native}.items()}
    return metrics, arrays


def run(out, control):
    protocol_path = out / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['rates'] == RATES and protocol['weights'] == WEIGHTS
    runtime = {'python': f'{sys.version_info.major}.{sys.version_info.minor}', 'numpy': np.__version__,
               'scipy': scipy.__version__, 'torch': torch.__version__}
    assert runtime == protocol['runtime']
    assert sha256(__file__) == protocol['script_sha256']
    for name, expected in protocol['control_sha256'].items():
        assert sha256(control / name) == expected
    assert source_inventory() == protocol['source_sha256']
    assert sha256(MANIFEST) == protocol['manifest_sha256']
    assert sha256(protocol['config_path']) == protocol['config_sha256']
    spec = protocol['model']
    assert sha256(spec['path']) == spec['sha256']
    assert shutil.disk_usage(out).free >= protocol['limits']['minimum_free_bytes']
    assert not any((out / x).exists() for x in ('rows.jsonl', 'report.json', 'arrays.npz'))
    assert dataset_verification(MANIFEST) == protocol['verification']
    eps = [NativeEpisode(path) for path in episode_paths(manifest=MANIFEST)
           if json.loads((path / 'meta.json').read_text())['split'] == 'train']
    ds = NativeReplay(eps, 'train')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    model = load_fs(spec['path']).to(device).eval()
    configure(model, freeze_vision=True)
    context = prove_policy_context(model)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    frozen, initial_digest = state_hashes(initial), model_digest(model)
    params = [p for p in model.parameters() if p.requires_grad]
    policy = [p for p in model.base.policy.parameters() if p.requires_grad]
    assert {id(p) for p in params} == {id(p) for p in policy}
    assert len(params) == 50 and sum(p.numel() for p in params) == 6949412
    old_arrays = np.load(control / 'arrays.npz', allow_pickle=False)
    old_rows = [json.loads(x) for x in (control / 'rows.jsonl').read_text().splitlines()]
    old = {r['batch']: r for r in old_rows if r['model'] == 'latest' and r['arm'] == 'plan_consistency'}
    assert set(old) == set(range(4))
    monitor_ids = [x['index'] for x in protocol['monitor']]
    selections = {p: [i for i, row in enumerate(protocol['monitor']) if row['phase'] == p] for p in PHASES}
    arrays, rows = {}, []
    started = time.monotonic()

    def forward(ids):
        return batch_loss(model, ds, ids, device, chunk_size=64, history_context=8,
                          return_outputs=True, controller_mode='op_tangent_pi_v1',
                          action_target_mode='desired_motion_v1', **WEIGHTS)

    def capture(prefix, parts, output, ids, is_monitor=False):
        wheelbase = torch.tensor([ds.episodes[ds.index[int(i)][0]].wheelbase for i in ids], device=device)
        metrics, data = measure(parts, output, wheelbase, selections if is_monitor else {})
        arrays.update({prefix + '_' + k: v for k, v in data.items()})
        return metrics, data

    def baseline_check(data, prefix):
        for k, value in data.items():
            np.testing.assert_allclose(value, old_arrays[prefix + '_' + k], atol=1e-7, rtol=1e-6, err_msg=prefix + '_' + k)

    with torch.no_grad():
        monitor_loss, monitor_parts, monitor_output = forward(monitor_ids)
        monitor_before, data = capture('monitor_before', monitor_parts, monitor_output, monitor_ids, True)
        baseline_check(data, 'latest_monitor_before')
        for bi, ids in enumerate(protocol['batches']):
            loss, parts, output = forward(ids)
            before, data = capture(f'b{bi}_fit_before', parts, output, ids)
            baseline_check(data, f'latest_{bi}_fit_before')
            np.testing.assert_allclose(float(loss), old[bi]['objective_before'], atol=1e-7, rtol=1e-6)
            for k, v in before['losses'].items():
                np.testing.assert_allclose(v, old[bi]['fit_before']['losses'][k], atol=1e-7, rtol=1e-6)
        for k, v in monitor_before['losses'].items():
            np.testing.assert_allclose(v, old[0]['monitor_before']['losses'][k], atol=1e-7, rtol=1e-6)
    baseline_receipt = {'passed': True, 'fit_batches': 4, 'monitor_anchors': 16,
                        'atol': 1e-7, 'rtol': 1e-6, 'compared': 'all saved before arrays, fit objectives, and component losses'}
    (out / 'baseline_receipt.json').write_text(json.dumps(baseline_receipt, indent=2) + '\n')
    print(json.dumps(baseline_receipt), flush=True)
    for bi, ids in enumerate(protocol['batches']):
        for rate in RATES:
            model.load_state_dict(initial)
            assert model_digest(model) == initial_digest
            opt = torch.optim.AdamW([{'params': policy, 'lr': rate}], weight_decay=1e-4)
            opt.zero_grad(set_to_none=True)
            loss, parts, output = forward(ids)
            np.testing.assert_allclose(float(loss.detach()), old[bi]['objective_before'], atol=1e-7, rtol=1e-6)
            loss.backward()
            with_grad = sum(p.grad is not None for p in params)
            assert with_grad == 42
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            opt.step()
            assert all(torch.isfinite(p).all() for p in params)
            assert state_hashes(model.state_dict()) == frozen
            steps = sorted({int(state['step']) for state in opt.state.values()})
            group_lrs = [group['lr'] for group in opt.param_groups]
            assert steps == [1] and len(opt.state) == with_grad == 42 and group_lrs == [rate]
            after_digest = model_digest(model)
            assert after_digest != initial_digest
            prefix = f'b{bi}_lr{RATES.index(rate)}'
            with torch.no_grad():
                after_loss, after_parts, after_output = forward(ids)
                fit_after, _ = capture(prefix + '_fit_after', after_parts, after_output, ids)
                _, m_parts, m_output = forward(monitor_ids)
                monitor_after, _ = capture(prefix + '_monitor_after', m_parts, m_output, monitor_ids, True)
            row = {'batch': bi, 'rate': rate, 'objective_before': float(loss.detach()),
                   'objective_after': float(after_loss), 'fit_after': fit_after, 'monitor_after': monitor_after,
                   'gradient_norm_before_clip': norm, 'initial_model_digest': initial_digest,
                   'after_model_digest': after_digest, 'frozen_state_unchanged': True, 'optimizer_steps': steps,
                   'eligible_policy_tensors': len(params), 'policy_tensors_with_grad': with_grad,
                   'optimizer_state_count': len(opt.state), 'optimizer_group_lrs': group_lrs,
                   'optimizer_defaults': opt.defaults}
            rows.append(row)
            with (out / 'rows.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps({'batch': bi, 'rate': rate, 'objective_before': row['objective_before'], 'objective_after': row['objective_after']}), flush=True)
            assert time.monotonic() - started < protocol['limits']['timeout_seconds']
            assert shutil.disk_usage(out).free >= protocol['limits']['minimum_free_bytes']
            del opt, loss, parts, output, after_parts, after_output, m_parts, m_output
    assert len(rows) == 12 and source_inventory() == protocol['source_sha256']
    assert sha256(spec['path']) == spec['sha256'] and sha256(MANIFEST) == protocol['manifest_sha256']
    np.savez_compressed(out / 'arrays.npz', **arrays)
    report = {'protocol_sha256': sha256(protocol_path), 'script_sha256': sha256(__file__),
              'rows_sha256': sha256(out / 'rows.jsonl'), 'arrays_sha256': sha256(out / 'arrays.npz'),
              'baseline_receipt_sha256': sha256(out / 'baseline_receipt.json'), 'updates': len(rows),
              'initial_digest': initial_digest, 'context': context, 'runtime': runtime, 'monitor_before': monitor_before,
              'elapsed_seconds': time.monotonic() - started, 'all_source_files_unchanged': True}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < protocol['limits']['artifact_bytes']
    assert shutil.disk_usage(out).free >= protocol['limits']['minimum_free_bytes']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.control)
