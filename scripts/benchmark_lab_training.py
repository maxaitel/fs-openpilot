"""Benchmark native training throughput and fixed-weight numerical parity.

Benchmark checkpoints are timing artifacts, never model candidates. The numeric
worker requires CUDA; this controller does not itself initialize the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ('fp32_contiguous_b8', 8, False, False, 24),
    ('fp32_contiguous_b16', 16, False, False, 24),
    ('fp32_contiguous_b32', 32, False, False, 24),
    ('fp32_channels_last_b32', 32, False, True, 24),
    ('bf16_channels_last_b32', 32, True, True, 24),
)
FAST_CONTEXT_CONFIG = ('fp32_contiguous_b32_context8', 32, False, False, 8)
CONTEXT_TOLERANCE = 1e-5
THRESHOLDS = {
    'fp32_channels_last': {'plan_max_abs': 1e-3, 'steer_max_abs': 1e-4, 'accel_max_abs': 1e-3},
    'bf16_channels_last': {'near_xy_max_abs': 0.1, 'steer_max_abs': 0.01, 'accel_max_abs': 0.1},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def summarize_run(run_dir, receipt_path, batch_size, expected_steps):
    result = {'checkpoint_role': 'benchmark_only_not_candidate'}
    history_path = run_dir / 'history.jsonl'
    if history_path.exists():
        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    else:
        rows = []
    result['completed_steps'] = len(rows)
    result['steps_complete'] = [r.get('step') for r in rows] == list(range(1, expected_steps + 1))
    warm = [float(row['step_seconds']) for row in rows if row['step'] >= 5]
    result['warm_steps'] = len(warm)
    result['warm_step_seconds_median'] = statistics.median(warm) if warm else None
    result['warm_step_seconds_p95'] = quantile(warm, .95)
    result['warm_examples_per_second'] = batch_size * len(warm) / sum(warm) if warm and sum(warm) > 0 else None
    peaks = [r['torch_peak_memory_bytes'] for r in rows if r.get('torch_peak_memory_bytes') is not None]
    result['torch_peak_allocated_bytes'] = max(peaks) if peaks else None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        result['measurement_returncode'] = receipt.get('returncode')
        result['end_to_end_seconds'] = receipt.get('wall_seconds')
        wall = receipt.get('wall_seconds')
        result['end_to_end_examples_per_second'] = batch_size * len(rows) / wall if wall and wall > 0 else None
        samples = receipt.get('samples', [])
        for key in ('gpu_pct', 'power_w', 'host_ram_gib', 'host_available_gib'):
            values = [float(s[key]) for s in samples if s.get(key) is not None]
            result[f'{key}_median'] = statistics.median(values) if values else None
        host = [float(s['host_ram_gib']) for s in samples if s.get('host_ram_gib') is not None]
        result['host_ram_gib_peak_sampled'] = max(host) if host else None
        result['telemetry_scope'] = receipt.get('scope')
        result['telemetry_samples'] = len(samples)
    else:
        result['measurement_returncode'] = None
        result['end_to_end_seconds'] = None
        result['telemetry_scope'] = 'measurement receipt unavailable'
    return result


def run_throughput(args, out, context8_enabled=False):
    results = []
    source = ['--manifest', str(args.manifest)] if args.manifest else ['--data', str(args.data)]
    configs = (*CONFIGS, FAST_CONTEXT_CONFIG) if context8_enabled else CONFIGS
    for name, batch, amp, channels_last, context in configs:
        run_dir = out / name
        receipt = out / f'{name}.measure.json'
        logfile = out / f'{name}.log'
        trainer = [sys.executable, '-m', 'fsop.train.lab_train', *source, '--init', str(args.init),
                   '--out', str(run_dir), '--steps', '20', '--batch-size', str(batch), '--val-every', '20',
                   '--val-samples', '8', '--seed', str(args.seed), '--device', 'cuda', '--threads', str(args.threads),
                   '--history-context', str(context)]
        if amp:
            trainer.append('--amp')
        if channels_last:
            trainer.append('--channels-last')
        command = [sys.executable, str(ROOT / 'scripts/measure_native.py'), '--out', str(receipt), '--', *trainer]
        started = time.monotonic()
        error = None
        with logfile.open('x') as log:
            try:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                returncode = completed.returncode
            except OSError as exc:
                error = f'{type(exc).__name__}: {exc}'
                log.write(error + '\n')
                returncode = None
        try:
            summary = summarize_run(run_dir, receipt, batch, 20)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary = {'checkpoint_role': 'benchmark_only_not_candidate', 'summary_error': str(exc),
                       'steps_complete': False}
        run_info_path = run_dir / 'run.json'
        try:
            run_info = json.loads(run_info_path.read_text())
            summary['init_hash_matches'] = run_info.get('init_sha256') == args.init_hash
            summary['manifest_hash_matches'] = (not args.manifest or
                                                run_info.get('manifest_sha256') == args.manifest_hash ==
                                                sha256(args.manifest))
        except (OSError, ValueError):
            summary['init_hash_matches'] = summary['manifest_hash_matches'] = False
        summary.update(name=name, batch_size=batch, amp=amp, channels_last=channels_last, history_context=context,
                       command=command, returncode=returncode, log=str(logfile),
                       launcher_wall_seconds=time.monotonic() - started, error=error)
        summary['passed'] = (returncode == 0 and summary.get('measurement_returncode') == 0 and
                             summary.get('steps_complete') is True and summary.get('warm_steps', 0) == 16 and
                             summary['init_hash_matches'] and summary['manifest_hash_matches'])
        results.append(summary)
        print(f"{name}: {'completed' if summary['passed'] else 'FAILED'}; log {logfile}", flush=True)
    return results


def compare_modes(reference, current, name, times):
    """Compare complete fixed-weight outputs and enforce predeclared mode limits."""
    import numpy as np

    errors = {}
    for key in reference:
        diff = np.abs(current[key] - reference[key])
        errors[key + '_max_abs'] = float(np.max(diff)) if np.isfinite(diff).all() else None
    near = np.asarray(times) <= 2.0
    diff_xy = np.abs(current['plan_mu'][:, near, :2] - reference['plan_mu'][:, near, :2])
    xy_distance = np.linalg.norm(diff_xy, axis=-1)
    errors['near_xy_max_abs'] = float(np.max(xy_distance)) if np.isfinite(xy_distance).all() else None
    errors['near_xy_max_coordinate_abs'] = float(np.max(diff_xy)) if np.isfinite(diff_xy).all() else None
    plan_errors = (errors['plan_mu_max_abs'], errors['plan_logstd_max_abs'])
    errors['plan_max_abs'] = max(plan_errors) if all(v is not None for v in plan_errors) else None
    all_finite = all(np.isfinite(v).all() for v in reference.values()) and all(
        np.isfinite(v).all() for v in current.values())
    passed = all_finite and all(errors[key] is not None and errors[key] <= limit
                                for key, limit in THRESHOLDS[name].items())
    return errors, bool(passed)


def history_context_gate(model, episodes, episode_ids, frames, batch, device):
    """Compare full, pruned and deliberately perturbed histories with gradients."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from fsop.train.lab_train import configure, fresh_history, pp_accel_from_op_plan
    from fsop.train.losses import plan_nll, pp_steer_from_op_plan

    model.to(memory_format=torch.contiguous_format)
    configure(model)
    full = fresh_history(model, episodes, episode_ids, frames, device, 64, history_context=24)
    short = fresh_history(model, episodes, episode_ids, frames, device, 64, history_context=8)
    perturbed = full.clone()
    perturbed[:, :16] += 137.0
    active_equal = bool(torch.equal(full[:, -8:], short[:, -8:]))
    inactive_zero = bool(torch.count_nonzero(short[:, :16]).item() == 0)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def evaluate(history):
        b = dict(batch, history=history)
        with torch.enable_grad():
            out = model.forward_images(b, heads=('plan',))
            mu, logstd = out['plan_mu'].float(), out['plan_logstd'].float()
            steer = pp_steer_from_op_plan(mu, b['speed'], b['vehicle_lr'], b['vehicle_wheelbase'])
            accel = pp_accel_from_op_plan(mu, b['speed'], b['vehicle_max_brake'], b['vehicle_max_accel'])
            pnll = plan_nll(mu, logstd, b['plan'])
            sloss = F.smooth_l1_loss(steer, b['steer_target'], beta=0.05)
            aloss = F.smooth_l1_loss(accel, b['accel_target'], beta=0.05)
            loss = pnll + 3.281158240898283 * sloss + aloss
            grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
        values = {'hidden': out['hidden'], 'plan_mu': mu, 'plan_logstd': logstd,
                  'desire_state': out['desire_state'], 'steer': steer, 'accel': accel,
                  'plan_nll': pnll, 'steer_loss': sloss, 'longitudinal_loss': aloss, 'loss': loss}
        return ({key: value.detach().float().cpu().numpy() for key, value in values.items()},
                [None if g is None else g.detach().float().cpu().numpy() for g in grads])

    reference, ref_grads = evaluate(full)
    result = {'tolerance': CONTEXT_TOLERANCE, 'active_features_bitwise_equal': active_equal,
              'inactive_features_zero': inactive_zero, 'comparisons': {}}
    for name, history in (('pruned8', short), ('perturbed_early16', perturbed)):
        values, grads = evaluate(history)
        errors = {}
        finite = True
        for key in reference:
            diff = np.abs(values[key] - reference[key])
            valid = np.isfinite(values[key]).all() and np.isfinite(reference[key]).all() and np.isfinite(diff).all()
            finite = finite and valid
            errors[key] = float(np.max(diff)) if valid else None
        grad_error = 0.0
        mismatches = []
        for (param_name, _), a, b in zip(params, ref_grads, grads, strict=True):
            if (a is None) != (b is None):
                mismatches.append(param_name)
            elif a is not None:
                diff = np.abs(a - b)
                if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(diff).all()):
                    finite = False
                else:
                    grad_error = max(grad_error, float(np.max(diff)))
        passed = (finite and not mismatches and all(v is not None and v <= CONTEXT_TOLERANCE
                                                   for v in errors.values()) and grad_error <= CONTEXT_TOLERANCE)
        result['comparisons'][name] = {'max_abs': errors, 'gradient_max_abs': grad_error if finite else None,
                                       'gradient_none_mismatches': mismatches, 'all_finite': bool(finite),
                                       'passed': bool(passed)}
    result['passed'] = (active_equal and inactive_zero and all(c['passed'] for c in result['comparisons'].values()))
    return result


def numeric_worker(args):
    """Run a fixed-baseline comparison; write failure details even for nonfinite outputs."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import default_collate

    from fsop.data.native_replay import NativeReplay, episode_paths, load_episodes
    from fsop.model.build import load_fs
    from fsop.sim.expert import T_IDXS
    from fsop.train.device import autocast_ctx
    from fsop.train.lab_train import fresh_history, pp_accel_from_op_plan, prove_policy_context, validation_samples
    from fsop.train.losses import plan_nll, pp_steer_from_op_plan

    report = {'checkpoint_role': 'fixed_input_comparison_not_candidate', 'thresholds': THRESHOLDS,
              'gate_scope': 'fixed-weight numeric parity only; no closed-loop quality claim', 'modes': {}}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the numeric benchmark')
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                             'gpu': torch.cuda.get_device_name(0),
                             'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                             'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                             'float32_matmul_precision': torch.get_float32_matmul_precision()}
        episodes = load_episodes(episode_paths(manifest=args.manifest, data=args.data))
        val = NativeReplay(episodes, 'val')
        selected = validation_samples(val, 8)
        if len(selected) != 8 or any(len(indices) != 1 for indices in selected.values()):
            raise ValueError('numeric benchmark requires exactly eight valid validation episodes')
        indices = [indices[0] for indices in selected.values()]
        report['samples'] = [{'episode': str(val.episodes[val.index[i][0]].path), 'frame': val.index[i][1]}
                             for i in indices]
        batch = default_collate([val[i] for i in indices])
        episode_ids, frames = batch.pop('episode').tolist(), batch.pop('frame').tolist()
        device = torch.device('cuda')
        model = load_fs(args.init, map_location='cpu').to(device).eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        try:
            proof = prove_policy_context(model)
            report['context8'] = history_context_gate(model, episodes, episode_ids, frames, batch, device)
            report['context8']['proof'] = proof
        except Exception as exc:
            report['context8'] = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        captured = {}
        for name, amp, channels_last in (('fp32_contiguous', False, False),
                                         ('fp32_channels_last', False, True),
                                         ('bf16_channels_last', True, True)):
            try:
                if channels_last:
                    model.to(memory_format=torch.channels_last)
                with torch.no_grad():
                    history = fresh_history(model, episodes, episode_ids, frames, device, 64,
                                            amp=amp, channels_last=channels_last)
                    b = dict(batch, history=history)
                    if channels_last:
                        for key in ('img', 'big_img'):
                            b[key] = b[key].contiguous(memory_format=torch.channels_last)
                    with autocast_ctx(device, enabled=amp):
                        out = model.forward_images(b, heads=('plan',))
                    mu, logstd = out['plan_mu'].float(), out['plan_logstd'].float()
                    steer = pp_steer_from_op_plan(mu, b['speed'], b['vehicle_lr'], b['vehicle_wheelbase'])
                    accel = pp_accel_from_op_plan(mu, b['speed'], b['vehicle_max_brake'], b['vehicle_max_accel'])
                    pnll = plan_nll(mu, logstd, b['plan'])
                    sloss = F.smooth_l1_loss(steer, b['steer_target'], beta=0.05)
                    aloss = F.smooth_l1_loss(accel, b['accel_target'], beta=0.05)
                    values = {'history': history, 'hidden': out['hidden'].float(), 'plan_mu': mu,
                              'plan_logstd': logstd, 'desire_state': out['desire_state'].float(),
                              'steer': steer, 'accel': accel, 'plan_nll': pnll,
                              'steer_loss': sloss, 'longitudinal_loss': aloss,
                              'total_loss': pnll + 3.281158240898283 * sloss + aloss}
                    captured[name] = {k: v.detach().cpu().numpy() for k, v in values.items()}
                finite = {k: bool(np.isfinite(v).all()) for k, v in captured[name].items()}
                report['modes'][name] = {'finite': finite, 'all_finite': all(finite.values()),
                                         'losses': {k: float(captured[name][k]) if finite[k] else None
                                                    for k in ('plan_nll', 'steer_loss', 'longitudinal_loss',
                                                              'total_loss')}}
            except Exception as exc:
                report['modes'][name] = {'all_finite': False, 'error': f'{type(exc).__name__}: {exc}'}
        reference = captured.get('fp32_contiguous')
        for name in ('fp32_channels_last', 'bf16_channels_last'):
            mode = report['modes'][name]
            if reference is None or name not in captured:
                mode['passed'] = False
                continue
            mode['errors'], mode['passed'] = compare_modes(reference, captured[name], name, T_IDXS)
        report['passed'] = (report['context8']['passed'] and
                            all(report['modes'].get(name, {}).get('passed', False)
                                for name in ('fp32_channels_last', 'bf16_channels_last')))
    except Exception as exc:
        report['passed'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
    args.worker_output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path)
    source.add_argument('--data', type=Path)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--numeric-only', action='store_true')
    mode.add_argument('--throughput-only', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--numeric-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    args.init = args.init.expanduser().resolve(strict=True)
    if args.manifest:
        args.manifest = args.manifest.expanduser().resolve(strict=True)
    if args.data:
        args.data = args.data.expanduser().resolve(strict=True)
    args.out = args.out.expanduser().resolve()
    if args.numeric_worker:
        if args.worker_output is None:
            parser.error('numeric worker requires an output file')
        return numeric_worker(args)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'purpose': 'performance and numeric parity only; benchmark checkpoints are not candidates',
              'quality_claim': False, 'thresholds': THRESHOLDS, 'seed': args.seed,
              'benchmark_script_sha256': sha256(Path(__file__).resolve()),
              'init': str(args.init), 'init_sha256': sha256(args.init),
              'manifest': str(args.manifest) if args.manifest else None,
              'manifest_sha256': sha256(args.manifest) if args.manifest else None,
              'data': str(args.data) if args.data else None, 'throughput': [], 'numeric': None}
    args.init_hash = report['init_sha256']
    args.manifest_hash = report['manifest_sha256']
    if not args.throughput_only:
        numeric_path = args.out / 'numeric.json'
        logfile = args.out / 'numeric.log'
        source_args = ['--manifest', str(args.manifest)] if args.manifest else ['--data', str(args.data)]
        command = [sys.executable, str(Path(__file__).resolve()), *source_args, '--init', str(args.init),
                   '--out', str(args.out), '--seed', str(args.seed), '--threads', str(args.threads),
                   '--numeric-worker', '--worker-output', str(numeric_path)]
        with logfile.open('x') as log:
            try:
                worker = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                code = worker.returncode
            except OSError as exc:
                log.write(f'{type(exc).__name__}: {exc}\n')
                code = None
        try:
            numeric = json.loads(numeric_path.read_text())
        except (OSError, ValueError) as exc:
            numeric = {'passed': False, 'error': f'numeric report unavailable: {exc}'}
        numeric.update(returncode=code, command=command, log=str(logfile))
        numeric['init_hash_matches'] = sha256(args.init) == args.init_hash
        numeric['manifest_hash_matches'] = not args.manifest or sha256(args.manifest) == args.manifest_hash
        numeric['passed'] = (numeric.get('passed') is True and code == 0 and
                             numeric['init_hash_matches'] and numeric['manifest_hash_matches'])
        report['numeric'] = numeric
    context_passed = (not args.throughput_only and 'error' not in report['numeric'] and
                      report['numeric'].get('context8', {}).get('passed') is True
                      and report['numeric']['init_hash_matches'] and report['numeric']['manifest_hash_matches'])
    if not args.numeric_only:
        report['throughput'] = run_throughput(args, args.out, context8_enabled=context_passed)
    report['context8_throughput_skipped'] = not context_passed
    expected_configs = len(CONFIGS) + int(context_passed)
    through_ok = args.numeric_only or (len(report['throughput']) == expected_configs and
                                      all(r['passed'] for r in report['throughput']))
    numeric_ok = args.throughput_only or report['numeric']['passed']
    report['passed'] = bool(through_ok and numeric_ok)
    report['numeric_gate_skipped'] = bool(args.throughput_only)
    report['speed_alone_can_select_fast_mode'] = False
    path = args.out / 'benchmark.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Benchmark {'completed' if report['passed'] else 'FAILED'}: {path}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
