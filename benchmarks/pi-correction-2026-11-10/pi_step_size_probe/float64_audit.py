"""Post-run float64 reconstruction from the saved predictions, without running the model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def reconstruct(arrays, prefix):
    def get(k): return arrays[prefix + '_' + k].astype(np.float64)
    mu, logstd, target, observed, wheelbase = [get(k) for k in ('mu', 'logstd', 'native_target', 'observed', 'wheelbase')]
    times = np.array([(i / 32.)**2 * 10 for i in range(33)], dtype=np.float64)
    # The production constants are stored in float32 before the losses use them.
    tw = np.array([1. if t <= 4 else max(.2, 1. - .8 * (t - 4.) / 6.) for t in times], dtype=np.float32).astype(np.float64)
    cw = np.array([1., 1., .1, .5, .5, .1, .1, .1, 0., 0., 0., .5, 0., 0., .3], dtype=np.float32).astype(np.float64)
    w = tw[None, :, None] * cw[None, None, :]
    log_b = np.clip(logstd, -7., 7.)
    nll = np.mean(np.sum((np.abs(target - mu) * np.exp(-log_b) + log_b) * w, axis=(1, 2))) / w.sum()
    n = int(np.searchsorted(times, 1., side='right'))
    dt = np.diff(times[:n])
    def residual(plan):
        secant = (plan[:, 1:n, 3:5] - plan[:, :n-1, 3:5]) / dt[None, :, None]
        return np.stack((plan[:, :n-1, 6:8] - secant, plan[:, 1:n, 6:8] - secant), axis=2)
    err = np.abs(residual(mu) - residual(target))
    consistency = np.mean(np.where(err < 1., .5 * err**2, err - .5))
    def action(plan):
        def interp(x, t):
            hi = int(np.searchsorted(times, t))
            a = (t - times[hi-1]) / (times[hi] - times[hi-1])
            return x[:, hi-1] * (1-a) + x[:, hi] * a
        sp = np.maximum(observed[:, 0], 1.)
        steer = np.arctan(wheelbase * (-2. * interp(plan[:, :, 11], .2) / (sp * .2) + plan[:, 0, 14] / sp))
        speed = np.linalg.norm(plan[:, :, 3:5], axis=-1)
        tangent = (plan[:, :, 3:5] * plan[:, :, 6:8]).sum(-1) / np.maximum(speed, .001)
        target_speed = interp(speed, .3)
        raw = 1.5 * (target_speed - np.linalg.norm(observed[:, :2], axis=-1)) + .5 * interp(tangent, .3)
        return np.stack((steer, raw, target_speed), axis=-1)
    return {'nll': float(nll), 'consistency': float(consistency), 'objective': float(nll+consistency),
            'absolute_action_error': np.abs(action(mu) - action(target))}


p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
out = a.out
report = json.loads((out/'report.json').read_text())
protocol = json.loads((out/'protocol.json').read_text())
receipt = json.loads((out/'independent_receipt.json').read_text())
assert receipt['passed'] and sha(out/'arrays.npz') == report['arrays_sha256'] and sha(out/'rows.jsonl') == report['rows_sha256']
rows = [json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines()]
arrays = np.load(out/'arrays.npz', allow_pickle=False)
mon = reconstruct(arrays, 'monitor_before')
names = ['steer_rad', 'raw_accel_mps2', 'target_speed_mps']
phases = ['steady_right', 'right_to_left', 'steady_left', 'recovery']
phase_ids = {p: [i for i, row in enumerate(protocol['monitor']) if row['phase'] == p] for p in phases}
max_loss_error = 0.
max_monitor_error = 0.
max_monitor_delta_error = 0.
results = []
for row in rows:
    bi, rate = row['batch'], row['rate']
    prefix = f"b{bi}_lr{protocol['rates'].index(rate)}"
    before = reconstruct(arrays, f'b{bi}_fit_before')
    after = reconstruct(arrays, prefix + '_fit_after')
    before_err = before['objective'] - row['objective_before']
    after_err = after['objective'] - row['objective_after']
    max_loss_error = max(max_loss_error, abs(before_err), abs(after_err),
                         abs(after['nll'] - row['fit_after']['losses']['plan_nll']),
                         abs(after['consistency'] - row['fit_after']['losses']['native_relative_acceleration']))
    m_after = reconstruct(arrays, prefix + '_monitor_after')
    aggregate_before = mon['absolute_action_error'].mean(0)
    aggregate_after = m_after['absolute_action_error'].mean(0)
    recorded = row['monitor_after']['action_mae']
    old_recorded = report['monitor_before']['action_mae']
    max_monitor_error = max(max_monitor_error, *(abs(aggregate_after[i] - recorded[k]) for i, k in enumerate(names)),
                            *(abs(aggregate_before[i] - old_recorded[k]) for i, k in enumerate(names)))
    max_monitor_delta_error = max(max_monitor_delta_error, *(abs((aggregate_after[i]-aggregate_before[i])-(recorded[k]-old_recorded[k])) for i,k in enumerate(names)))
    by_phase = {}
    for phase, idx in phase_ids.items():
        phase_before = mon['absolute_action_error'][idx].mean(0)
        phase_after = m_after['absolute_action_error'][idx].mean(0)
        phase_recorded = row['monitor_after']['phases'][phase]
        phase_old = report['monitor_before']['phases'][phase]
        max_monitor_error = max(max_monitor_error, *(abs(phase_after[i] - phase_recorded[k]) for i, k in enumerate(names)),
                                *(abs(phase_before[i] - phase_old[k]) for i,k in enumerate(names)))
        max_monitor_delta_error = max(max_monitor_delta_error, *(abs((phase_after[i]-phase_before[i])-(phase_recorded[k]-phase_old[k])) for i,k in enumerate(names)))
        by_phase[phase] = {k: {'before':float(phase_before[i]), 'after':float(phase_after[i]), 'delta':float(phase_after[i]-phase_before[i])} for i,k in enumerate(names)}
    results.append({'batch':bi, 'rate':rate, 'reported_fit_margin':row['objective_before'] - row['objective_after'],
                    'float64_fit_margin':before['objective'] - after['objective'],
                    'margin_discrepancy':(before['objective']-after['objective'])-(row['objective_before']-row['objective_after']),
                    'before_objective_discrepancy':before_err, 'after_objective_discrepancy':after_err,
                    'monitor':{k:{'before':float(aggregate_before[i]),'after':float(aggregate_after[i]),'delta':float(aggregate_after[i]-aggregate_before[i])} for i,k in enumerate(names)}, 'phases':by_phase})
assert max_loss_error < 2e-6 and max_monitor_error < 2e-5
result = {'passed':True, 'method':'NumPy float64 reconstruction of plan NLL, native-relative consistency and action MAEs from frozen FP32 predictions. Differences from FP32 recomputation include arithmetic precision and do not establish a noise floor.',
          'max_absolute_loss_discrepancy':max_loss_error, 'max_absolute_monitor_mae_discrepancy':max_monitor_error,
          'max_absolute_monitor_delta_discrepancy':max_monitor_delta_error,
          'rows_sha256':sha(out/'rows.jsonl'), 'arrays_sha256':sha(out/'arrays.npz'), 'audit_script_sha256':sha(Path(__file__)),
          'cells':results}
(out/'float64_receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'passed':True,'max_absolute_loss_discrepancy':max_loss_error,'max_absolute_monitor_mae_discrepancy':max_monitor_error,'max_absolute_monitor_delta_discrepancy':max_monitor_delta_error,'cells':[{k:x[k] for k in ('batch','rate','reported_fit_margin','float64_fit_margin','margin_discrepancy')} for x in results]}), flush=True)
