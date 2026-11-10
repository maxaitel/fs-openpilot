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
    parser.add_argument('--evaluation', required=True, type=Path)
    args = parser.parse_args()
    t, d, v, out = args.training, args.development, args.videos, args.out
    names = ('run.json', 'history.jsonl', 'calibration.json', 'baseline_validation.json', 'selection_protocol.json', 'source.zip', 'selection_receipt.json', 'telemetry.json')
    context = t.parent
    extra = [context / 'head_hardware_interference.txt']
    inputs = [t / n for n in names] + [d / 'plan.lock.json', d / 'manifest.json'] + extra
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    run, baseline, protocol, selection, telemetry = [read(t / n) for n in ('run.json', 'baseline_validation.json', 'selection_protocol.json', 'selection_receipt.json', 'telemetry.json')]
    rows = [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))) for line in (t / 'history.jsonl').read_text().splitlines() if line.strip()]
    if protocol.get('schema_version') != 4 or selection.get('protocol') != protocol:
        raise ValueError('selection must bind schema 4 protocol')
    if [r.get('step') for r in rows] != list(range(1, 501)):
        raise ValueError('requires 500 consecutive updates')
    if {r['step'] for r in rows if 'validation' in r} != {1, 100, 200, 300, 400, 500}:
        raise ValueError('validation schedule mismatch')
    for r in rows:
        if r.get('selection_protocol_sha256') != sha(t / 'selection_protocol.json'):
            raise ValueError('history protocol hash mismatch')
        if r.get('calibration_sha256') != sha(t / 'calibration.json'):
            raise ValueError('history calibration hash mismatch')
        for key in ('loss', 'plan_nll', 'plan_l1', 'steer', 'longitudinal', 'native_relative_acceleration', 'step_seconds'):
            number(r[key])
    data = [(0, baseline)] + [(r['step'], r['validation']) for r in rows if 'validation' in r]
    for _, val in data:
        for key in ('plan_nll', 'plan_l1', 'native_relative_acceleration', 'steer', 'longitudinal'):
            number(val['aggregate'][key])
    for filename in ('run.json', 'history.jsonl', 'baseline_validation.json', 'selection_protocol.json', 'calibration.json', 'source.zip'):
        matches = [value for path, value in selection['input_sha256'].items() if path.endswith('/' + filename) and value == sha(t / filename)]
        if not matches:
            raise ValueError(f'selection does not bind {filename}')
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
    interpretation_path = args.evaluation / 'interpretation.json'
    interpretation = read(interpretation_path)
    inputs.append(interpretation_path)
    if interpretation['selection_sha256'] != sha(t / 'selection_receipt.json') or interpretation['protocol_sha256'] != sha(t / 'selection_protocol.json'):
        raise ValueError('interpretation selection or protocol hash mismatch')
    if interpretation['models'] != plan['models']:
        raise ValueError('interpretation models differ from locked plan')
    expected_step = selection['selected']['step'] if selection['trained_candidate'] else 500
    if interpretation['evaluated_step'] != expected_step:
        raise ValueError('interpretation checkpoint step mismatch')
    candidate = next(c for c in selection['candidates'] if c['step'] == expected_step)
    history_checkpoint = next(row for row in rows if row['step'] == expected_step)
    if candidate['checkpoint_sha256'] != history_checkpoint['checkpoint_sha256']:
        raise ValueError('selection candidate differs from training history')
    if selection['trained_candidate'] and not (selection['selected_eligible'] and candidate['eligible'] and candidate['improves_step_zero']):
        raise ValueError('selected candidate must qualify and improve step zero')
    pilot_paths = [context / 'head_pilot/receipt.json', context / 'head_pilot/inference.json']
    pilot, inference = [read(path) for path in pilot_paths]
    inputs.extend(pilot_paths)
    if not (pilot['inference_passed'] and pilot['artifact_passed'] and inference['passed'] and pilot['frame_count'] == inference['frames'] == inference['expected_frames'] and pilot['checkpoint_sha256'] == inference['checkpoint_sha256'] == models['candidate']['sha256'] and inference['controller_contract'] == models['candidate']['controller_contract']):
        raise ValueError('pilot verification mismatch')
    pilot_episode = context / 'head_pilot/episode'
    pilot_evidence = [pilot_episode / 'summary.json', pilot_episode / 'lab_artifact_audit.json'] + [pilot_episode / 'evidence' / name for name in ('completion.json', 'metadata.json', 'manifest.jsonl')]
    inputs.extend(pilot_evidence)
    pilot_summary, pilot_audit, pilot_completion = [read(path) for path in pilot_evidence[:3]]
    if not (pilot_summary['pilot_only'] and pilot_summary['checkpoint_sha256'] == pilot['checkpoint_sha256'] == pilot_summary['checkpoint_sha256_after'] and pilot_summary['physical_cone_hits'] == pilot['physical_cone_hits'] and pilot_audit['passed'] and pilot_audit['completion_sha256'] == sha(pilot_evidence[2]) and pilot_completion['summary_sha256'] == sha(pilot_evidence[0]) == inference['summary_sha256'] and pilot_completion['frame_count'] == pilot['frame_count']):
        raise ValueError('pilot evidence mismatch')
    for name in ('completion.json', 'metadata.json', 'manifest.jsonl'):
        path = pilot_episode / 'evidence' / name
        if inference['evidence_sha256'][name] != sha(path):
            raise ValueError(f'pilot inference evidence mismatch: {name}')
    if pilot_completion['metadata_sha256'] != sha(pilot_evidence[3]) or pilot_completion['manifest_sha256'] != sha(pilot_evidence[4]):
        raise ValueError('pilot completion evidence mismatch')
    notes = [context / 'head_training_decision.md', context / 'pi_head_scope_probe/report.md', context / 'pi_step_size_probe/report.md', context / 'pi_update_probe/report_v2.md']
    trace_dir = context / 'head_failure_trace'
    trace_paths = [trace_dir / name for name in ('README.md', 'trace.json', 'trajectory.png', 'output_sha256.json', 'analyze.py')]
    inputs.extend(trace_paths)
    trace = read(trace_paths[1])
    trace_hashes = read(trace_paths[3])
    if any(trace_hashes[name] != sha(trace_dir / name) for name in ('trace.json', 'trajectory.png')) or trace['script_sha256'] != sha(trace_paths[4]):
        raise ValueError('failure trace output or script hash mismatch')
    for path, digest in trace['input_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'failure trace input hash mismatch: {path}')
    for arm in ('baseline', 'candidate'):
        summary = episodes[arm, 'heavy_skid'][0]
        traced = trace['episodes'][arm]
        if traced['seconds_simulated'] != summary['seconds_simulated'] or traced['physical_contacts'] != summary['physical_cone_hits']:
            raise ValueError(f'failure trace episode mismatch: {arm}')
    setup_note = context / 'head_execution_setup.md'
    previous = Path('runs/skidpad_correction_report/index.html')
    for path in notes + [previous, setup_note]:
        if not path.is_file():
            raise FileNotFoundError(path)
    inputs.extend(notes + [previous, setup_note])
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout='constrained')
    metrics = (('plan_nll', 'Plan NLL'), ('native_relative_acceleration', 'Native-relative consistency'), ('steer', 'Steering SmoothL1'), ('longitudinal', 'Longitudinal SmoothL1'), ('plan_l1', 'Plan L1'))
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.plot([r['step'] for r in rows], [r[key] for r in rows], alpha=.45, label='Train batch')
        ax.plot([step for step, _ in data], [val['aggregate'][key] for _, val in data], marker='o', label='Validation aggregate')
        ax.set(title=title, xlabel='Update'); ax.grid(alpha=.2); ax.legend()
    ax = axes.flat[5]
    ax.plot([r['step'] for r in rows], [r['step_seconds'] for r in rows]); ax.set(title='Step duration', xlabel='Update', ylabel='Seconds'); ax.grid(alpha=.2)
    for ax, key, title in zip(axes.flat[6:], ('gpu_pct', 'power_w', 'host_ram_gib'), ('Whole GPU utilization (%)', 'Whole GPU power (W)', 'Whole host RAM (GiB)')):
        ax.plot([s['wall_s'] for s in telemetry['samples']], [s[key] for s in telemetry['samples']]); ax.set(title=title, xlabel='Job wall seconds'); ax.grid(alpha=.2)
    fig.savefig(out / 'training.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    labels = [f"{case.replace('heavy_', '')}\n{arm}" for case in ('heavy_skid', 'heavy_track') for arm in ('baseline', 'candidate')]
    summaries = [episodes[arm, case][0] for case in ('heavy_skid', 'heavy_track') for arm in ('baseline', 'candidate')]
    for ax, (key, title) in zip(axes, (('physical_cone_hits', 'Physical cone contacts'), ('seconds_simulated', 'Exposure (simulated seconds)'), ('complete', 'Completed missions'))):
        vals = [int(s[key]) if key == 'complete' else s[key] for s in summaries]
        bars = ax.bar(range(4), vals, color=['#597da3', '#d88740'] * 2)
        for bar, val, summary in zip(bars, vals, summaries):
            status = 'complete' if summary['complete'] else 'DNF'
            ax.annotate(f'{val:.2f}\n{status}' if key == 'seconds_simulated' else f'{val}\n{status}', (bar.get_x() + bar.get_width()/2, val), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
        ax.set_xticks(range(4), labels); ax.set_title(title); ax.set_ylim(0, max(vals + [1]) * 1.4 + .3)
    fig.suptitle('One episode per arm and case. Completion counts are descriptive, not a general rate.')
    fig.savefig(out / 'paired.png', dpi=140); plt.close(fig)
    esc = html.escape
    def href(path):
        return esc(link(path, out), quote=True)
    role = 'selected eligible checkpoint' if selection['trained_candidate'] else 'final diagnostic checkpoint'
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>Head correction trial</title><style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 1em;line-height:1.5}img{max-width:100%}td,th{border:1px solid #bbb;padding:.4em;text-align:left}table{border-collapse:collapse}video{max-width:100%;width:480px}</style></head><body>', '<h1>Skidpad head correction trial</h1>', f'<p>Selection status: {esc(str(selection["status"]))}. The evaluated candidate is the {esc(role)} at step {expected_step}. Candidate eligible: {esc(str(candidate["eligible"]))}. Strict plan-L1 improvement over step zero: {esc(str(candidate["improves_step_zero"]))}. Selected step: {selection["selected"]["step"]}. These four exposed episodes are one case per arm per discipline. They do not estimate a general completion rate or establish a fix.</p>', f'<p>Evaluation scope: {esc(str(interpretation["scope"]))}. Fresh tests opened: {esc(str(interpretation["fresh_tests_opened"]))}. Candidate checkpoint SHA-256: <code>{esc(models["candidate"]["sha256"])}</code>.</p>', '<h2>Training and validation</h2>', f'<p>The full approximately 31 million parameter model was loaded with vision frozen. The forward-eligible parameter count was {run["trainable_parameters"]:,}; only {run["optimized_parameters"]:,} parameters were optimized. Batch metrics are sampled training values. Validation uses the aggregate frozen validation set. Longitudinal supervision had weight zero.</p>', '<img src="training.png" alt="Training and validation curves and hardware telemetry"><table><tr><th>Step</th><th>Plan NLL</th><th>Consistency</th><th>Steering</th><th>Longitudinal</th><th>Plan L1</th></tr>']
    for step, val in data:
        parts.append('<tr><td>' + str(step) + '</td>' + ''.join(f'<td>{val["aggregate"][key]:.6g}</td>' for key, _ in metrics) + '</tr>')
    nonwarm = rows[1:]
    step_total = sum(number(row['step_seconds']) for row in nonwarm)
    if step_total <= 0:
        raise ValueError('nonwarm training duration must be positive')
    throughput = len(nonwarm) * run['args']['batch_size'] / step_total
    peak_alloc = max(number(row['torch_peak_memory_bytes']) for row in rows) / 2**30
    parts += [f'</table><p>Excluding the first update, median step duration was {np.median([r["step_seconds"] for r in nonwarm]):.3f} s. Training processed {throughput:.1f} examples per summed step-second, calculated as {len(nonwarm)} updates times batch size {run["args"]["batch_size"]}, divided by {step_total:.1f} step seconds. Step timing excludes validation and setup. Peak PyTorch allocated memory was {peak_alloc:.3f} GiB.</p><p>Job wall time {telemetry["wall_seconds"]:.1f} seconds. Sampled median whole-GPU utilization {np.median([s["gpu_pct"] for s in telemetry["samples"]]):.1f}%, median whole-GPU power {np.median([s["power_w"] for s in telemetry["samples"]]):.1f} W, and peak whole-host RAM {max(s["host_ram_gib"] for s in telemetry["samples"]):.2f} GiB. These whole-device and whole-host measurements include setup and validation. Concurrent simpleFoam jobs used host CPUs during this run. Whole-host unified RAM includes other workloads and tmpfs. Contention prevents attribution of timing differences to the training method. GPU utilization is not achieved FLOPS.</p>', '<h2>Paired driving evidence</h2><p>Physical cone contacts and off-course events are different measurements. Exposure can differ when a run ends early, so fewer contacts alone do not establish improvement.</p><img src="paired.png" alt="Contacts, exposure and mission completion with DNF labels"><table><tr><th>Case</th><th>Arm</th><th>Contacts</th><th>Off course</th><th>Simulated seconds</th><th>Result</th><th>Route violation</th><th>Completion</th><th>Videos</th></tr>']
    for case in ('heavy_skid', 'heavy_track'):
        for arm in ('baseline', 'candidate'):
            summary, media = episodes[arm, case]
            videos = ' '.join(f'<a href="{href(media / (name + ".mp4"))}">{name}</a>' for name in ('debug', 'research'))
            case_label = 'Skidpad (heavy)' if case == 'heavy_skid' else 'Track (heavy)'
            case_link = href(d / 'cases' / f'{case}.json')
            parts.append(f'<tr><td><a href="{case_link}">{case_label}</a></td><td>{arm}</td><td>{summary["physical_cone_hits"]}</td><td>{summary["oc"]}</td><td>{summary["seconds_simulated"]:.2f}</td><td>{esc(summary["result"])}</td><td>{esc(summary.get("route", {}).get("violation") or "none")}</td><td>{"Complete" if summary["complete"] else "DNF"}</td><td>{videos}</td></tr>')
    parts.append('</table><h2>Recorded evaluation timing</h2><p>Each episode uses the same 16-subframe RTX render profile. Frame medians exclude frame zero. The policy-with-capture measurement includes capture bookkeeping and is not pure inference. Wall time and timing components reflect the recorded workload and host contention; they do not establish a model speedup.</p><table><tr><th>Case</th><th>Arm</th><th>Wall seconds</th><th>Real-time factor</th><th>Render ms</th><th>Policy with capture ms</th><th>Physics ms</th><th>Step wall ms</th></tr>')
    for case in ('heavy_skid', 'heavy_track'):
        for arm in ('baseline', 'candidate'):
            summary = episodes[arm, case][0]
            frames = read(d / arm / case / 'frames.json')
            if len(frames) < 2 or summary['render_profile']['subframes'] != 16:
                raise ValueError('evaluation timing requires recorded frames and the 16-subframe profile')
            timings = [np.median([number(frame[key]) for frame in frames[1:]]) * 1000 for key in ('render_s', 'policy_with_capture_s', 'physics_s', 'step_wall_s')]
            label = 'Skidpad (heavy)' if case == 'heavy_skid' else 'Track (heavy)'
            parts.append(f'<tr><td>{label}</td><td>{arm}</td><td>{number(summary["wall_seconds"]):.2f}</td><td>{number(summary["real_time_factor"]):.3f}</td>' + ''.join(f'<td>{value:.1f}</td>' for value in timings) + '</tr>')
    misses = [c['gate_miss_m'] for c in trace['episodes']['baseline']['all_x_zero_crossings'] if c['forward'] and not c['inside_gate']]
    if len(misses) != 2:
        raise ValueError('expected two measured baseline gate misses')
    parts.append(f'</table><h2>Skidpad failure trace</h2><p>Saved positions show the baseline missed the two left return gates by {misses[0]:.2f} m and {misses[1]:.2f} m and stopped outside the exit. The candidate rolled over before its first left return. The trace found no lap-sized progress jump. It does not isolate the cause of the rollover.</p><img src="{href(trace_paths[2])}" alt="Recorded skidpad trajectories, timing gate and contacts"><p><a href="{href(trace_paths[0])}">Failure analysis and limits</a></p><h2>Video evidence</h2>')
    for case in ('heavy_skid', 'heavy_track'):
        for arm in ('baseline', 'candidate'):
            media = episodes[arm, case][1]
            for name in ('debug', 'research'):
                parts.append(f'<h3>{esc(case)} / {arm} / {name}</h3><video controls preload="none" src="{href(media / (name + ".mp4"))}"></video>')
    parts.append(f'<h2>Selection and context</h2><p>The {pilot["frame_count"]}-frame integration pilot passed inference verification and artifact audit for the evaluated checkpoint. It was not a completed mission.</p>')
    reasons = candidate.get('rejection_reasons', [])
    reason_text = ', '.join(map(str, reasons)) if reasons else 'none'
    parts.append(f'<p>Candidate rejection reasons: {esc(reason_text)}. Artifact audits replay saved states and commands; they do not rerun physics or inference.</p>')
    for label, path in [('Selection receipt', t / 'selection_receipt.json'), ('Evaluation interpretation', interpretation_path), ('Previous failed trial report', previous), ('Execution setup note', setup_note), ('Pilot receipt', pilot_paths[0]), ('Pilot inference verification', pilot_paths[1]), ('Head hardware snapshot', extra[0]), ('Head training decision', notes[0]), ('Head scope probe', notes[1]), ('Learning rate probe', notes[2]), ('Update probe', notes[3])]:
        parts.append(f'<p><a href="{href(path)}">{esc(label)}</a></p>')
    parts.append('<p>Source and output hashes are recorded in <a href="report_receipt.json">the report receipt</a>.</p></body></html>')
    (out / 'index.html').write_text('\n'.join(parts))
    receipt = {'kind': 'head_correction_report', 'input_sha256': {str(p.resolve()): sha(p) for p in inputs + [Path(__file__)]}, 'output_sha256': {name: sha(out / name) for name in ('index.html', 'training.png', 'paired.png')}}
    (out / 'report_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(f'Report written to {out / "index.html"}')


if __name__ == '__main__':
    main()
