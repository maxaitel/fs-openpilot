#!/usr/bin/env python3
"""Build a PI correction report from completed training and paired development receipts."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'expected finite number, got {value!r}')
    return value


def link(path, out):
    return quote(os.path.relpath(path.resolve(), out.resolve()), safe='/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--development', required=True, type=Path)
    parser.add_argument('--videos', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    t, d, v, out = args.training, args.development, args.videos, args.out
    names = ('run.json', 'history.jsonl', 'calibration.json', 'baseline_validation.json', 'selection_protocol.json', 'source.zip', 'selection_receipt.json', 'telemetry.json')
    context = t.parent
    extra = [context / 'hardware_interference.json', context / 'repeat_divergence/early_comparison.json', context / 'render_clock_probe/interpretation.json']
    inputs = [t / n for n in names] + [d / 'plan.lock.json', d / 'manifest.json'] + extra
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    run, baseline, protocol, selection, telemetry = [read(t / n) for n in ('run.json', 'baseline_validation.json', 'selection_protocol.json', 'selection_receipt.json', 'telemetry.json')]
    rows = [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))) for line in (t / 'history.jsonl').read_text().splitlines() if line.strip()]
    if protocol.get('schema_version') != 3 or selection.get('protocol') != protocol:
        raise ValueError('selection must bind schema 3 protocol')
    if [r.get('step') for r in rows] != list(range(1, 501)):
        raise ValueError('requires 500 consecutive updates')
    if {r['step'] for r in rows if 'validation' in r} != {1, 100, 200, 300, 400, 500}:
        raise ValueError('validation schedule mismatch')
    for r in rows:
        if r.get('selection_protocol_sha256') != sha(t / 'selection_protocol.json'):
            raise ValueError('history protocol hash mismatch')
        for key in ('loss', 'plan_l1', 'native_relative_acceleration'):
            number(r[key])
    data = [(0, baseline)] + [(r['step'], r['validation']) for r in rows if 'validation' in r]
    for _, val in data:
        for key in ('plan_l1', 'native_relative_acceleration', 'steer'):
            number(val['aggregate'][key])
    for filename in ('run.json', 'history.jsonl', 'baseline_validation.json', 'selection_protocol.json'):
        matches = [value for path, value in selection['input_sha256'].items() if path.endswith('/' + filename) and value == sha(t / filename)]
        if not matches:
            raise ValueError(f'selection does not bind {filename}')
    if selection['trained_candidate'] or selection['selected']['step'] != 0 or any(c['eligible'] for c in selection['candidates']):
        raise ValueError('report is specific to the rejected diagnostic trial')
    if telemetry['returncode'] != 0 or not telemetry['samples']:
        raise ValueError('successful measured training required')
    number(telemetry['wall_seconds'])
    for s in telemetry['samples']:
        for key in ('wall_s', 'gpu_pct', 'power_w', 'host_ram_gib'):
            number(s[key])
    plan = read(d / 'plan.lock.json')
    manifest = read(d / 'manifest.json')
    expected_pairs = {f'{arm}/{case}' for arm in ('baseline', 'candidate') for case in ('heavy_skid', 'heavy_track')}
    if manifest.get('complete') is not True or manifest.get('lock_sha256') != sha(d / 'plan.lock.json') or set(manifest['episodes']) != expected_pairs:
        raise ValueError('manifest does not bind the complete locked plan')
    if run['source']['archive_sha256'] != sha(t / 'source.zip'):
        raise ValueError('training source archive mismatch')
    models = {m['name']: m for m in plan['models']}
    cases = {item['name']: item['case'] for item in plan['cases']}
    if set(models) != {'baseline', 'candidate'} or set(cases) != {'heavy_skid', 'heavy_track'} or len(plan['models']) != 2 or len(plan['cases']) != 2:
        raise ValueError('expected exactly two models and two cases in locked plan')
    if models['baseline']['sha256'] != run['init_sha256'] or run['controller_contract'] != models['baseline']['controller_contract']:
        raise ValueError('development baseline differs from training initializer or controller')
    if models['baseline']['controller_contract'] != models['candidate']['controller_contract'] or models['baseline']['controller_contract']['mode'] != 'op_tangent_pi_v1':
        raise ValueError('paired models must use identical PI controller')
    if selection['trained_candidate'] and models['candidate']['sha256'] != selection['selected']['checkpoint_sha256']:
        raise ValueError('candidate differs from selected checkpoint')
    if not selection['trained_candidate']:
        final = next(c for c in selection['candidates'] if c['step'] == 500)
        if models['candidate']['sha256'] != final['checkpoint_sha256']:
            raise ValueError('diagnostic candidate differs from final checkpoint')
    episodes = {}
    for arm in ('baseline', 'candidate'):
        for case_dir in ('heavy_skid', 'heavy_track'):
            folder, media = d / arm / case_dir, v / arm / case_dir
            case_file = d / 'cases' / f'{case_dir}.json'
            paths = [folder / n for n in ('summary.json', 'telemetry.json', 'frames.json', 'lab_artifact_audit.json', 'lab_video_audit.json', 'producer_render_receipt.json', 'evidence/completion.json', 'evidence/manifest.jsonl', 'evidence/metadata.json')] + [case_file, media / 'debug.mp4', media / 'research.mp4']
            inputs.extend(paths)
            summary, artifact, video, completion = [read(folder / n) for n in ('summary.json', 'lab_artifact_audit.json', 'lab_video_audit.json', 'evidence/completion.json')]
            episode = manifest['episodes'][f'{arm}/{case_dir}']
            receipt = episode['receipt']
            command = episode['command']
            expected_args = {'--case': f'/cases/{case_dir}.json', '--checkpoint': models[arm]['checkpoint'], '--expected-sha': models[arm]['sha256'], '--controller': models[arm]['controller_contract']['mode'], '--out': f'/{arm}/{case_dir}'}
            if episode['status'] != 'complete' or any(key not in command or command.index(key) + 1 >= len(command) or not command[command.index(key) + 1].endswith(value) for key, value in expected_args.items()):
                raise ValueError(f'manifest command mismatch for {folder}')
            if read(case_file) != cases[case_dir] or summary['case'] != cases[case_dir] or summary['checkpoint_sha256'] != models[arm]['sha256'] or summary['checkpoint_sha256_after'] != summary['checkpoint_sha256']:
                raise ValueError(f'plan identity mismatch for {folder}')
            if summary['controller_contract'] != models[arm]['controller_contract'] or artifact['controller_contract'] != summary['controller_contract'] or video['controller_contract'] != summary['controller_contract']:
                raise ValueError(f'controller mismatch for {folder}')
            if summary['pilot_only'] or summary['result'] in (None, 'running') or type(summary['complete']) is not bool:
                raise ValueError(f'nonterminal result for {folder}')
            if receipt['summary_sha256'] != sha(folder / 'summary.json') or receipt['completion_sha256'] != sha(folder / 'evidence/completion.json') or receipt['frames_sha256'] != sha(folder / 'frames.json') or receipt['result'] != summary['result'] or receipt['frame_count'] != completion['frame_count']:
                raise ValueError(f'manifest receipt mismatch for {folder}')
            if completion['manifest_sha256'] != sha(folder / 'evidence/manifest.jsonl') or completion['metadata_sha256'] != sha(folder / 'evidence/metadata.json') or artifact['frames_sha256'] != receipt['frames_sha256'] or artifact['telemetry_sha256'] != sha(folder / 'telemetry.json'):
                raise ValueError(f'episode evidence mismatch for {folder}')
            number(summary['seconds_simulated'])
            number(summary['oc'])
            hits = number(summary['physical_cone_hits'])
            if hits < 0 or int(hits) != hits or hits != len(summary['contact_events']):
                raise ValueError(f'invalid contact count for {folder}')
            if artifact['passed'] is not True or completion['summary_sha256'] != sha(folder / 'summary.json') or artifact['completion_sha256'] != sha(folder / 'evidence/completion.json'):
                raise ValueError(f'unbound artifact for {folder}')
            render_audit = folder / 'lab_artifact_audit.json'
            if video['artifact_audit_sha256'] != sha(render_audit):
                render_audit = folder / 'lab_artifact_audit_producer.json'
                if not render_audit.is_file() or read(render_audit) != artifact:
                    raise ValueError(f'render artifact audit differs for {folder}')
                inputs.append(render_audit)
            if video['completion_sha256'] != sha(folder / 'evidence/completion.json') or video['artifact_audit_sha256'] != sha(render_audit) or video['producer_render_receipt_sha256'] != sha(folder / 'producer_render_receipt.json'):
                raise ValueError(f'unbound video audit for {folder}')
            for name in ('debug', 'research'):
                entry = video['videos'][name]
                if entry['sha256'] != sha(media / f'{name}.mp4') or entry['fps'] != 20 or entry['decoded_frames'] != completion['frame_count'] + 40:
                    raise ValueError(f'video hash or frame mismatch for {folder}/{name}')
            episodes[arm, case_dir] = (summary, media)
    for name in ('heavy_skid', 'heavy_track'):
        if episodes['baseline', name][0]['case'] != episodes['candidate', name][0]['case']:
            raise ValueError('pair case mismatch')
    if episodes['baseline', 'heavy_skid'][0]['discipline'] != 'skidpad' or episodes['baseline', 'heavy_track'][0]['discipline'] != 'trackdrive':
        raise ValueError('case directory discipline mismatch')
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, key, title in ((axes[0,0], 'native_relative_acceleration', 'Native-relative consistency'),
                           (axes[0,1], 'plan_nll', 'Plan NLL'), (axes[0,2], 'plan_l1', 'Plan L1'),
                           (axes[1,0], 'steer', 'Steering SmoothL1')):
        ax.plot([r['step'] for r in rows], [r[key] for r in rows], alpha=.45, label='TRAIN batch')
        ax.plot([step for step, _ in data], [val['aggregate'][key] for _, val in data], marker='o', label='Validation aggregate')
        ax.set(title=title, xlabel='Update'); ax.legend(); ax.grid(alpha=.2)
    for ax, key, title in ((axes[1,1], 'gpu_pct', 'Whole GPU utilization (%)'), (axes[1,2], 'host_ram_gib', 'Host RAM used (GiB)')):
        ax.plot([s['wall_s'] for s in telemetry['samples']], [s[key] for s in telemetry['samples']]); ax.set(title=title, xlabel='Job wall seconds'); ax.grid(alpha=.2)
    fig.savefig(out / 'pi_training.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, case, title in zip(axes, ('heavy_skid', 'heavy_track'), ('Heavy skidpad', 'Heavy track')):
        for i, arm in enumerate(('baseline', 'candidate')):
            summary = episodes[arm, case][0]
            hits = summary['physical_cone_hits']
            status = 'finished' if summary['complete'] else 'DNF'
            ax.bar(i, hits, color=('#597da3', '#d88740')[i], width=.55)
            ax.text(i, hits + .35, f"{hits} {'hit' if hits == 1 else 'hits'}\n{summary['seconds_simulated']:.2f} s, {status}", ha='center', va='bottom')
        ax.set_xticks([0, 1], ['Baseline', 'Rejected candidate'])
        ax.set(title=title, ylabel='Physical cone hits', ylim=(0, max(episodes[a, case][0]['physical_cone_hits'] for a in ('baseline', 'candidate')) * 1.35 + 2))
    fig.suptitle('One exposed episode per arm and case. Duration differs.')
    fig.savefig(out / 'pi_paired.png', dpi=140); plt.close(fig)
    existing_plots = ('training-consistency.png', 'training-gradients.png', 'training-hardware.png', 'training-losses.png', 'training-step-time.png', 'training-validation.png')
    inputs.extend(out / name for name in existing_plots)
    esc = html.escape
    role = 'rejected diagnostic checkpoint, selector retained step zero'
    start_metrics, final_metrics = data[0][1]['aggregate'], data[-1][1]['aggregate']
    changes = {key: (final_metrics[key] / start_metrics[key] - 1) * 100 for key in ('plan_l1', 'native_relative_acceleration', 'steer')}
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>PI correction report</title><style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 1em}img{max-width:100%}td,th{border:1px solid #bbb;padding:.4em}table{border-collapse:collapse}</style></head><body>', '<h1>Skidpad correction and retraining</h1>', '<p><strong>Outcome: this trial did not establish a skidpad fix.</strong> All checkpoints failed the strict selector. The final checkpoint is evaluated only as a rejected diagnostic. These are one exposed episode per arm per case, using the same PI controller, and do not estimate a general completion rate.</p>', f'<p>Selector status: {esc(selection["status"])}. Candidate evaluation: {esc(role)}. Selected step: {selection["selected"]["step"]}. Training ran 500 updates.</p>', f'<h2>Training and validation</h2><p>The full model remains loaded. This stage trains {run["trainable_parameters"]:,} policy parameters across {len(run["trainable_tensors"])} tensors. Vision and cone heads are frozen.</p><p>TRAIN values are sampled batch metrics. Validation values are aggregate metrics on the frozen validation set. Native-relative consistency is unweighted. The objective uses coefficient 1.0.</p><img src="pi_training.png" alt="Training, validation and hardware plots">', '<table><tr><th>Step</th><th>Plan L1</th><th>Consistency</th><th>Steering smooth L1</th></tr>']
    for step, val in data:
        parts.append(f'<tr><td>{step}</td><td>{val["aggregate"]["plan_l1"]:.6g}</td><td>{val["aggregate"]["native_relative_acceleration"]:.6g}</td><td>{val["aggregate"]["steer"]:.6g}</td></tr>')
    parts += [f'</table><p>From step 0 to 500, validation consistency changed {changes["native_relative_acceleration"]:+.1f}%, plan L1 {changes["plan_l1"]:+.2f}%, and steering smooth L1 {changes["steer"]:+.1f}%. Lower is better for these metrics. Offline changes do not establish driving improvement.</p>', f'<p>Measured job wall time {telemetry["wall_seconds"]:.1f} seconds. Whole-device median GPU utilization {np.median([s["gpu_pct"] for s in telemetry["samples"]]):.1f}%, median power {np.median([s["power_w"] for s in telemetry["samples"]]):.1f} W, and peak sampled host RAM {max(s["host_ram_gib"] for s in telemetry["samples"]):.2f} GiB. Telemetry includes setup and validation. The last roughly 90 seconds overlapped unrelated CFD CPU jobs, so whole-host RAM includes those processes and the wall time is not an isolated speed comparison. GPU utilization is not achieved FLOPS. Peak training process PyTorch allocation was {max(r["torch_peak_memory_bytes"] for r in rows) / 2**30:.3f} GiB.</p>', '<h2>Paired driving results</h2><p>Both skidpad runs accepted right, right and then failed the route. The candidate ended earlier, so fewer hits came with less exposure and cannot be called a win. Off-course (OC) counts are distinct from physical hits.</p><img src="pi_paired.png" alt="Physical hits, duration, and completion by pair"><table><tr><th>Case</th><th>Arm</th><th>Physical hits</th><th>OC</th><th>Simulated seconds</th><th>Result / route violation</th><th>Complete</th><th>Videos</th></tr>']
    for case in ('heavy_skid', 'heavy_track'):
        for arm in ('baseline', 'candidate'):
            summary, media = episodes[arm, case]
            videos = ' '.join(f'<a href="{esc(link(media / (name + ".mp4"), out), quote=True)}">{name}</a>' for name in ('debug', 'research'))
            parts.append(f'<tr><td>{esc(summary["case"]["id"])}</td><td>{arm}</td><td>{summary["physical_cone_hits"]}</td><td>{summary["oc"]}</td><td>{summary["seconds_simulated"]:.2f}</td><td>{esc(summary["result"])} / {esc(summary.get("route", {}).get("violation") or "none")}</td><td>{summary["complete"]}</td><td>{videos}</td></tr>')
    parts += ['</table><h2>Recorded evaluation throughput</h2><p>Each episode uses 20 Hz simulated control with the same 16-subframe RTX profile. Capture rate is recorded frames divided by measured episode wall time. The instrumented policy call includes capture bookkeeping; it is not a standalone inference benchmark. These runs overlapped unrelated CPU jobs and some CPU video export.</p><table><tr><th>Case</th><th>Arm</th><th>Wall seconds</th><th>Captured frames / wall second</th><th>Real-time ratio</th><th>Median GPU busy</th><th>Mean policy call ms</th></tr>']
    for case in ('heavy_skid', 'heavy_track'):
        for arm in ('baseline', 'candidate'):
            summary, _ = episodes[arm, case]
            frames = read(d / arm / case / 'frames.json')
            samples = read(d / arm / case / 'telemetry.json')
            wall = number(summary['wall_seconds'])
            if wall <= 0 or not frames or not samples:
                raise ValueError('evaluation timing requires positive duration and samples')
            gpu = np.median([number(sample['gpu_pct']) for sample in samples])
            policy_ms = np.mean([number(frame['policy_with_capture_s']) for frame in frames]) * 1000
            parts.append(f'<tr><td>{case}</td><td>{arm}</td><td>{wall:.2f}</td><td>{len(frames) / wall:.2f}</td><td>{summary["seconds_simulated"] / wall:.3f}</td><td>{gpu:.0f}%</td><td>{policy_ms:.2f}</td></tr>')
    parts += ['</table>']
    followup = d.parent / 'vision_pi_development1'
    episode = followup / 'evaluation/vision/heavy_skid'
    media = v.parent / 'vision_pi_development1'
    verification_path = followup / 'verification.json'
    verification = read(verification_path)
    summary = read(episode / 'summary.json')
    completion = read(episode / 'evidence/completion.json')
    if completion['summary_sha256'] != sha(episode / 'summary.json') or verification['completion_sha256'] != sha(episode / 'evidence/completion.json'):
        raise ValueError('unbound vision initializer diagnostic')
    if (verification['diagnostic_only'] is not True
            or summary['checkpoint_sha256'] != 'ab5d18e2b6984192ad82c2f5d288c64c43b2ae2132e29e1901053f5149b6cc02'
            or summary['case'] != episodes['baseline', 'heavy_skid'][0]['case']
            or summary['controller_contract'] != models['baseline']['controller_contract']):
        raise ValueError('vision initializer diagnostic identity mismatch')
    inputs.extend([verification_path, episode / 'summary.json', episode / 'evidence/completion.json'])
    for name in ('debug', 'research'):
        video_path = media / (name + '.mp4')
        entry = verification['videos'][name]
        if entry['sha256'] != sha(video_path) or entry['fps'] != 20 or entry['decoded_frames'] != completion['frame_count'] + 40:
            raise ValueError('vision initializer video mismatch')
        inputs.append(video_path)
    links = ' '.join(f'<a href="{esc(link(media / (name + ".mp4"), out), quote=True)}">{name} video</a>' for name in ('debug', 'research'))
    parts.append(f'<h2>Earlier initializer diagnostic</h2><p>The earlier vision-adapted weights under the same PI controller ended with {esc(summary["result"])} at {summary["seconds_simulated"]:.0f} simulated seconds, {summary["physical_cone_hits"]} physical hits, {summary["oc"]} off-course events and {len(summary["route"]["accepted_route"])} accepted laps. This is a separate one-run diagnostic, not a trained candidate or evidence for a robust ranking. {links}.</p>')
    parts.append('<h2>Training update investigation</h2><p>The production learning rate made a first plan-based AdamW update increase its own fit loss in every sampled batch. Smaller steps removed that fit overshoot but still worsened monitored steering. The monitoring anchors have distinct indices, but share episodes and can overlap temporal windows with fit data. These probes do not measure driving performance.</p>')
    for label, relative in (('Independent objective updates', 'pi_update_probe/report_v2.md'), ('Smaller-step results', 'pi_step_size_probe/report.md')):
        path = d.parent / relative
        inputs.append(path)
        parts.append(f'<p><a href="{esc(link(path, out), quote=True)}">{label}</a></p>')
    parts += ['<p>The candidate track video shows a tilted camera view around 42 seconds. The cause has not been established.</p><p>Earlier training plots: ' + ', '.join(f'<a href="{esc(name, quote=True)}">{esc(name.removesuffix(".png"))}</a>' for name in existing_plots) + '.</p><p>The separate earlier controller-only skidpad intervention and fixed-steering reference probe are not training comparisons. An earlier run of the same baseline had 12 hits versus 16 in this repeat. Physical state at frame zero matched while rendered RGB differed, showing an early rendered-input difference. The cause of the later outcome difference is not established. A short fixed-action probe found no measured physics-time or pose advance during capture, writing, or idle. It does not establish deterministic neural rollouts. Artifact audits replay saved states and commands; they do not rerun physics or inference.</p><h2>Evidence</h2><p>Input paths and SHA-256 hashes are in <a href="pi_report_receipt.json">the report receipt</a>.</p></body></html>']
    (out / 'index.html').write_text('\n'.join(parts))
    receipt = {'kind': 'pi_correction_report', 'input_sha256': {str(p.resolve()): sha(p) for p in inputs + [Path(__file__)]}, 'output_sha256': {name: sha(out / name) for name in ('index.html', 'pi_training.png', 'pi_paired.png')}}
    (out / 'pi_report_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(f'Report written to {out / "index.html"}')


if __name__ == '__main__':
    main()
