"""Frozen-input command sensitivity probe for the saved PI baseline."""
from __future__ import annotations

import argparse
import copy
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from fsop.eval.closed_loop import ModelPolicy, mission_brake_with_override
from fsop.eval.controller import controller_contract
from fsop.geometry.op_frames import op_to_ego_plan
from fsop.model.temporal import TemporalState
from isaac_eval.evidence import ImageHeadCapture, read_evidence, verified_summary
from isaac_eval.parity_cases import realized_environment, sha256_file
from isaac_eval.parity_protocol import GRAPH_FILES
from isaac_eval.verify_inference import observation, state_digest
from isaac_lab_fsop.cases import Case, environment
from isaac_lab_fsop.verify_inference import compare

ANCHORS = {120: 'early_right', 170: 'steady_right', 430: 'approaching_right_to_left', 600: 'left_area_unverified'}
MODES = ('recorded', 'recorded_repeat', 'swapped', 'zero', 'pulse_right', 'pulse_left')


def transform(name, command, frame):
    result = np.asarray(command, dtype=np.float32).copy()
    if name == 'swapped':
        result[[1, 2]] = result[[2, 1]]
    elif name == 'zero':
        result.fill(0)
    elif name.startswith('pulse_'):
        result.fill(0)
        if frame % 20 == 0:
            result[{'pulse_left': 1, 'pulse_right': 2}[name]] = 1
    return result


def jsonify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    return value


def run(episode, checkpoint, out):
    episode, checkpoint, out = Path(episode), Path(checkpoint), Path(out)
    evidence = episode / 'evidence'
    meta = json.loads((evidence / 'metadata.json').read_text())
    summary = verified_summary(evidence)
    receipt = json.loads((evidence / 'completion.json').read_text())
    if summary is None or (evidence / receipt['summary_path']).resolve() != (episode / 'summary.json').resolve():
        raise ValueError('missing or mismatched completion receipt')
    if meta['case']['id'] != 'regression-skidpad-72000-heavy-recorded-visual_stress_v1':
        raise ValueError('wrong frozen case')
    if meta['fps'] != 20 or meta['command_mode'] != 'recorded':
        raise ValueError('wrong input protocol')
    if meta['controller_contract'] != controller_contract('op_tangent_pi_v1'):
        raise ValueError('wrong PI controller')
    if not summary['contact_events'] or summary['contact_events'][0]['control_frame'] != 181:
        raise ValueError('unexpected first-contact frame')
    if sha256_file(checkpoint) != meta['checkpoint_sha256'] or checkpoint.stat().st_size != meta['checkpoint_size_bytes']:
        raise ValueError('checkpoint mismatch')
    if set(meta['graph_sha256']) != set(GRAPH_FILES):
        raise ValueError('graph inventory mismatch')
    sources = {k: v for k, v in meta['source_sha256'].items() if k.startswith('src/fsop/') or k in
               ('isaac_lab_fsop/cases.py', 'isaac_eval/evidence.py', 'isaac_eval/parity_cases.py')}
    for path, expected in {**sources, **meta['graph_sha256']}.items():
        if sha256_file(path) != expected:
            raise ValueError(f'source or graph mismatch: {path}')
    if summary['case'] != meta['case'] or summary['controller_contract'] != meta['controller_contract']:
        raise ValueError('summary does not match metadata')
    env = environment(Case(**meta['case']))
    realization = episode / 'realization.json'
    if sha256_file(realization) != meta['realization_sha256'] or realized_environment(env) != json.loads(realization.read_text()):
        raise ValueError('realization mismatch')
    if not np.array_equal(meta['nominal_detection_calibration'], env.T_ego_cam()):
        raise ValueError('calibration mismatch')
    fixed = [evidence / 'metadata.json', evidence / 'manifest.jsonl', evidence / 'completion.json',
             episode / 'summary.json', realization, Path(__file__), Path(__file__).with_name('PROTOCOL.md')]
    fixed_before = {str(p): sha256_file(p) for p in fixed}
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = ModelPolicy.from_checkpoint(checkpoint, controller='op_tangent_pi_v1')
    policy.model.requires_grad_(False)
    state_before = state_digest(policy.model)
    if state_before != meta['state_sha256'] or state_before != summary['state_sha256']:
        raise ValueError('model tensor digest mismatch')
    base = policy.model.base if hasattr(policy.model, 'base') else policy.model
    original = base.policy_raw
    captured = {}
    def capture(*args, **kwargs):
        captured['inputs'] = [x.detach().clone() for x in args]
        output = original(*args, **kwargs)
        captured['output'] = [x.detach().cpu().numpy().copy() for x in output]
        return output
    base.policy_raw = capture
    histories = {name: TemporalState() for name in MODES}
    raw_commands = {name: deque([np.zeros(8, np.float32) for _ in range(100)], maxlen=100) for name in MODES}
    anchors, stats, consumed = {}, {}, {}
    try:
        with torch.inference_mode():
            for _, record, entry in read_evidence(evidence):
                frame = record['frame']
                if frame != len(consumed) or record['state']['t'] != frame / 20:
                    raise ValueError('nonsequential evidence')
                if not np.array_equal(record['nominal_detection_calibration'], env.T_ego_cam()):
                    raise ValueError('record calibration mismatch')
                consumed[entry['file']] = entry['sha256']
                command = np.asarray(record['commands']['applied'], dtype=np.float32)
                inputs = {name: transform(name, command, frame) for name in MODES}
                for name in MODES:
                    raw_commands[name].append(inputs[name].copy())
                pooled = {name: ts.push_desire(inputs[name]).copy() for name, ts in histories.items()}
                obs = observation(record, env)
                ctrl = copy.deepcopy(policy.ctrl) if frame in ANCHORS else None
                with ImageHeadCapture(policy.model) as head:
                    steer, accel, info = policy(obs, env)
                if head.output is None or 'inputs' not in captured:
                    raise ValueError('capture missing')
                got = {'action': [steer, accel], 'prediction': info,
                       'image_head': head.output, 'policy_raw': captured['output']}
                compare({k: record[k] for k in got}, got, stats, 'saved')
                if not np.array_equal(pooled['recorded'], captured['inputs'][1].cpu().numpy()[0]):
                    raise ValueError('reconstructed command queue differs from live queue')
                if frame in ANCHORS:
                    anchors[frame] = {'inputs': [x.clone() for x in captured['inputs']], 'controller': ctrl,
                                      'record': record, 'pooled': pooled,
                                      'raw_history': {name: np.stack(q).copy() for name, q in raw_commands.items()},
                                      'pulse_history': {name: np.stack(ts.desire_q).copy() for name, ts in histories.items()},
                                      'expected': got}
                if frame == max(ANCHORS):
                    break
        if len(anchors) != len(ANCHORS) or any(x['mismatches'] for x in stats.values()):
            raise ValueError('saved prediction parity failed: ' + json.dumps(stats))
        results = {}
        with torch.inference_mode():
            for frame, item in anchors.items():
                obs = observation(item['record'], env)
                variants = {}
                for name in MODES:
                    fb, _, traffic = item['inputs']
                    des = torch.from_numpy(item['pooled'][name][None]).to(policy.device)
                    outputs = original(fb, des, traffic)
                    plan = outputs[0].reshape(1, 2, 33, 15)
                    mu = op_to_ego_plan(plan[0, 0].cpu().numpy())
                    std = np.exp(plan[0, 1].clamp(-7, 7).cpu().numpy())
                    ctrl = copy.deepcopy(item['controller'])
                    steer, accel, diagnostics = ctrl.step(mu, obs.state.vx, obs.state.vy, obs.t,
                        lambda c: mission_brake_with_override(c, obs, env))
                    variants[name] = {'command_history_100': item['raw_history'][name],
                                      'desire_pulse_queue_100': item['pulse_history'][name],
                                      'desire_pooled_25': item['pooled'][name],
                                      'policy_raw': [x.cpu().numpy() for x in outputs],
                                      'plan': mu, 'plan_std': std, 'action': [steer, accel],
                                      'controller': diagnostics}
                check = {}
                compare(item['expected']['policy_raw'], variants['recorded']['policy_raw'], check, 'raw')
                compare(item['expected']['action'], variants['recorded']['action'], check, 'action')
                compare(item['expected']['prediction']['controller'], variants['recorded']['controller'], check, 'controller')
                compare(variants['recorded'], variants['recorded_repeat'], check, 'repeat')
                if any(v['mismatches'] for v in check.values()):
                    raise ValueError('branch reconstruction mismatch: ' + json.dumps(check))
                results[str(frame)] = {'phase': ANCHORS[frame], 'state': item['record']['state'],
                                       'progress': item['record']['progress'],
                                       'earlier_contacts': [c for c in summary['contact_events'] if c['control_frame'] < frame],
                                       'features_24_plus_current': item['inputs'][0].cpu().numpy(),
                                       'traffic': item['inputs'][2].cpu().numpy(),
                                       'variants': variants, 'branch_parity': check}
    finally:
        base.policy_raw = original
    after = state_digest(policy.model)
    if (after != state_before or policy.model.training or any(p.requires_grad for p in policy.model.parameters())
            or sha256_file(checkpoint) != meta['checkpoint_sha256']):
        raise ValueError('model changed')
    for name, digest in consumed.items():
        if sha256_file(evidence / name) != digest:
            raise ValueError('evidence changed')
    if fixed_before != {str(p): sha256_file(p) for p in fixed}:
        raise ValueError('protocol or provenance files changed')
    for path, digest in {**sources, **meta['graph_sha256']}.items():
        if sha256_file(path) != digest:
            raise ValueError(f'source or graph changed: {path}')
    report = {'passed': True, 'rtol': 1e-5, 'atol': 1e-5, 'case': meta['case'], 'anchors': results,
              'saved_parity': stats, 'state_sha256_before': state_before, 'state_sha256_after': after,
              'checkpoint_sha256': meta['checkpoint_sha256'], 'source_sha256': sources,
              'graph_sha256': meta['graph_sha256'], 'consumed_evidence_sha256': consumed,
              'file_sha256': fixed_before,
              'limitations': ['Frames 430 and 600 follow earlier physical contacts; the left lap was not accepted.',
                              'Fixed observation and controller memory; no physical trajectory is simulated.']}
    with out.open('x') as f:
        json.dump(jsonify(report), f, indent=2, allow_nan=False)
        f.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.episode, args.checkpoint, args.out)
    print(f'Passed. Results: {args.out}')
