#!/usr/bin/env python3
"""Build a local, evidence-bound driving-reliability report without GPU work."""
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / 'benchmarks/driving-reliability-2026-11-10'
OUT = ROOT / 'runs/driving_reliability_report'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def link(path, label=None):
    return f'<a href="{html.escape(os.path.relpath(path, OUT), quote=True)}">{html.escape(label or path.name)}</a>'


def figure(name):
    path = OUT / name
    plt.savefig(path, dpi=155, bbox_inches='tight')
    plt.close()
    return '<img alt="' + html.escape(name) + '" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">'


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--training-dir', type=Path, default=ROOT / 'runs/direct_command_trial3_archive')
    ap.add_argument('--episodes', type=Path, nargs='*', default=[], help='Local episode directories with saved summaries and audits')
    args = ap.parse_args()
    OUT = args.out.resolve()
    train_dir = args.training_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []
    def bind(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in sources:
            sources.append(path)
        return path
    def read(path):
        return load(bind(path))
    def local(path):
        bind(path)
        return link(path, str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))

    bind(Path(__file__).resolve())
    base = BENCH / 'native_divergence_analysis/inputs'
    cones = np.load(bind(base / 'source_snapshot/cones.npy'))
    specs = [
        ('Recorded teacher', base / 'run1/recorded/trace.npz', 'teacher controls, full exposure'),
        ('Original OP + PI suffix', base / 'run1/op_tangent_pi/trace.npz', 'teacher prefix through tick 254'),
        ('Position feedback + PI suffix', BENCH / 'native_transition_correction/run1/op_lateral_pi/trace.npz', 'teacher prefix through tick 254'),
        ('Recorded longitudinal suffix', BENCH / 'native_transition_correction/run1/op_recorded_long/trace.npz', 'teacher prefix through tick 254'),
        ('Combined suffix', BENCH / 'native_transition_correction/run1/op_lateral_recorded_long/trace.npz', 'teacher prefix through tick 254'),
        ('Standing OP + PI', BENCH / 'native_standing_reference/run1/op_tangent_pi/trace.npz', 'controller from tick 0'),
    ]
    transition = read(BENCH / 'native_transition_correction/run1/comparison.json')
    standing = read(BENCH / 'native_standing_reference/run1/comparison.json')
    divergence = read(BENCH / 'native_divergence_analysis/results.json')
    cmd = read(BENCH / 'reliability_command_probe/summary.json')
    app = read(BENCH / 'reliability_appearance_probe/result/comparison.json')
    audit = read(BENCH / 'native_control_audit.json')
    suite = read(BENCH / 'acceptance_suite.json')
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True, sharex=True, sharey=True)
    for ax, (label, path, exposure) in zip(axes.flat, specs):
        with np.load(bind(path)) as z:
            states, contacts = z['states'], z['contacts']
            hits = np.flatnonzero(np.diff(contacts, prepend=contacts[0]) > 0)
            ax.scatter(cones[:, 0], cones[:, 1], s=6, color='gray', alpha=.45, label='initial cones')
            ax.plot(states[:, 0], states[:, 1], lw=1, color='#1c6b9e', label='vehicle path')
            if len(hits):
                ax.scatter(states[hits, 0], states[hits, 1], marker='x', s=28, c='#c12424', label='vehicle at contact increment')
            ax.set_title(f'{label}\n{exposure}; {int(contacts[-1])} contacts', fontsize=9)
            ax.set_aspect('equal', adjustable='box')
            rows.append({'arm': label, 'exposure': exposure, 'contacts': int(contacts[-1]), 'states': len(states), 'contact_increment_ticks': hits.tolist()})
    fig.suptitle('Privileged native reference paths: one source, no vision policy')
    fig.supxlabel('world x (m)'); fig.supylabel('world y (m)')
    axes.flat[-1].legend(fontsize=7, loc='lower left')
    native_image = figure('native_paths.png')

    anchors = cmd['anchors']
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = [f'{k}\n'+('postcontact' if anchors[k]['earlier_contacts'] else 'precontact') for k in anchors]
    vals = [anchors[k]['variants']['swapped']['action_delta'][0] for k in anchors]
    ax.bar(names, vals, color='#427ba1'); ax.axhline(0, color='black', lw=.6)
    ax.set_ylabel('Steering change (rad)'); ax.set_title('Fixed-state swapped route history')
    command_image = figure('command_probe.png')

    st = app['stability']; contrasts = st['contrasts']; noise = st['repeat_noise_max']['near_xy_m']
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(list(contrasts), [v['magnitude']['near_xy_m'] for v in contrasts.values()], color='#427ba1')
    ax.axhline(noise * st['contrast_multiplier'], color='#b33', linestyle='--', label='5 × largest repeat difference')
    ax.set_ylabel('Near-term XY difference (m)'); ax.set_title('Appearance interventions at precontact frame 170'); ax.legend()
    appearance_image = figure('appearance_probe.png')

    diagnosis_images = []
    for filename, label, output in [('result.json', 'First direct-head trial3', 'first_error.png'),
                                    ('result_recovery2.json', 'Recovery-data trial', 'first_error_recovery.png')]:
        path = BENCH / 'direct_command_diagnosis' / filename
        if not path.is_file():
            continue
        diagnosis = read(path)
        diagnosed = diagnosis['rows']
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        times = [row['time_s'] for row in diagnosed]
        for i, (ax, unit) in enumerate(zip(axes[:2], ['Steering request (rad)', 'Acceleration request (m/s²)'])):
            ax.plot(times, [row['fixed_latent_head_requests']['recorded'][i] for row in diagnosed], label='Learned raw request')
            ax.plot(times, [row['native_teacher_at_actual_state'][i] for row in diagnosed], label='Privileged teacher label at actual state')
            ax.set_ylabel(unit); ax.legend(fontsize=8)
        axes[2].plot(times, [row['progress']['lat'] for row in diagnosed]); axes[2].set_ylabel('Path lateral error (m)')
        axes[2].set_xlabel('Simulated time (s)')
        for ax in axes:
            for event in diagnosis['contact_events']:
                ax.axvline(event['t'], color='#bb3333', alpha=.4, linewidth=.7)
        fig.suptitle(label + '. Red lines mark recorded physical contacts.')
        diagnosis_images.append(figure(output))
    diagnosis_image = ''.join(diagnosis_images)

    recovery_html = ''
    recovery_dirs = [BENCH / 'direct_command_recovery_data' / name for name in ('skidpad', 'trackdrive')]
    if all((p / 'states.npy').is_file() and (p / 'cones.npy').is_file() for p in recovery_dirs):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        details = []
        for ax, directory in zip(axes, recovery_dirs):
            metadata = read(directory / 'meta.json')
            receipt = read(directory / 'receipt_v2.json')
            for name in ('states.npy', 'cones.npy'):
                if digest(bind(directory / name)) != metadata['arrays_sha256'][name]:
                    raise ValueError(f'Recovery graph input changed: {directory / name}')
            states = np.load(directory / 'states.npy')
            initial_cones = np.load(directory / 'cones.npy')
            handoff, terminal = receipt['takeover_tick'], receipt['terminal_start_tick']
            ax.scatter(initial_cones[:, 0], initial_cones[:, 1], s=5, color='gray', alpha=.5)
            ax.plot(states[:handoff+1, 0], states[:handoff+1, 1], label='Learner prefix')
            ax.plot(states[handoff:terminal+1, 0], states[handoff:terminal+1, 1], color='#17845a', label='Privileged teacher suffix')
            ax.scatter(states[handoff, 0], states[handoff, 1], marker='x', color='black', label='Handoff')
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(directory.name + f'; mission ends at {terminal / 20:.2f} s')
            ax.set_xlabel('world x (m)'); ax.set_ylabel('world y (m)'); ax.legend(fontsize=7)
            details.append(f'{directory.name}: handoff tick {handoff}, settling starts at {receipt["settled_start_tick"]}, {receipt["total_qualified_anchors"]} qualified anchors, zero physical contacts and off-course events.')
        fig.suptitle('Corrective TRAIN data. Teacher control is not learned-model success.')
        recovery_html = '<h2>Qualified corrective TRAIN data</h2><p>' + html.escape(' '.join(details)) + '</p>' + figure('recovery_paths.png')

    steps_path = train_dir / 'steps.jsonl'
    train_path = train_dir / 'train.json'
    val_paths = sorted(train_dir.glob('validation_*.json')) if train_dir.is_dir() else []
    steps = [json.loads(line) for line in bind(steps_path).read_text().splitlines() if line.strip()] if steps_path.is_file() else []
    validations = [read(p) for p in val_paths]
    training = read(train_path) if train_path.is_file() else None
    failure_path = train_dir / 'train_failure.json'
    failure = read(failure_path) if failure_path.is_file() else None
    postflight_path = train_dir.with_suffix('.postflight.json')
    postflight = read(postflight_path) if postflight_path.is_file() else None
    measure_path = train_dir.with_suffix('.measure.json')
    measurement = read(measure_path) if measure_path.is_file() else None
    training_html = '<p><strong>PENDING.</strong> No completed train.json is available. A partial log is not a qualification or a driving result.</p>' if training is None else '<p><strong>Completed training receipt.</strong> Qualification passed: ' + html.escape(str(training.get('passed'))) + '. Selection and all validation groups are in ' + local(train_path) + '.</p>'
    if failure is not None:
        training_html = '<p><strong>TRAINING PROCESS FAILED.</strong> ' + html.escape(failure['error']) + '. The recorded optimizer log has ' + str(len(steps)) + ' updates. The original failed receipt remains in ' + local(failure_path) + '. No successful train.json was fabricated.</p>'
        if postflight is not None:
            training_html += '<p>The separate read-only postflight reports artifact/data checks passed = ' + html.escape(str(postflight['passed'])) + ', candidate qualified = ' + html.escape(str(postflight['candidate_qualified'])) + ', optimization resumed = ' + html.escape(str(postflight['optimization_resumed'])) + '. See ' + local(postflight_path) + '. Its exact final validation recomputation does not qualify the rejected model.</p>'
    if training is not None:
        frozen = training.get('frozen_base_state_unchanged', training.get('base_state_unchanged'))
        scope = 'Frozen base complement unchanged' if 'frozen_base_tensors' in training else 'Original base unchanged'
        training_html += '<p>Selected checkpoint: ' + html.escape(str(training.get('selected', {}).get('step') if training.get('selected') else 'none')) + '. Training wall duration: ' + html.escape(str(training.get('seconds'))) + ' s. ' + scope + ': ' + html.escape(str(frozen)) + '.</p>'
        if 'frozen_base_tensors' in training:
            training_html += '<p>The joint trial optimizes 296450 parameters in six original policy tensors and four command-head tensors. Its ' + str(training['frozen_base_tensors']) + '-tensor base complement stays original. Every full checkpoint contains all 417 updated base states and all four head states. No original-base equality is claimed for the six trained tensors.</p>'
    runtime_path = train_dir / 'runtime.json'
    runtime = read(runtime_path) if runtime_path.is_file() else None
    if runtime is not None:
        timing = runtime['inference']
        training_html += '<p>Saved-input runtime preflight: passed = ' + html.escape(str(runtime['passed'])) + ', backend = ' + html.escape(runtime['policy_device']) + ', threads = ' + str(runtime['torch_num_threads']) + ', frames = ' + str(timing['frames']) + ', mean forward seconds = ' + str(timing['mean_seconds']) + ', p95 = ' + str(timing['p95_seconds']) + '. The measurement excludes physics, rendering and evidence writes. It establishes neither driving success nor a real-time 20 Hz rate. See ' + local(runtime_path) + '.</p>'
    if validations:
        training_html += '<ul>' + ''.join('<li>Step ' + str(v['step']) + ': all groups passed = ' + html.escape(str(v['passed'])) + '; failed groups = ' + html.escape(', '.join(k for k, g in v['groups'].items() if not g['passed']) or 'none') + '</li>' for v in validations) + '</ul>'
    if steps:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot([x['step'] for x in steps], [x['loss'] for x in steps], lw=.8)
        ax.set(xlabel='Update', ylabel='Training loss', title='Actual optimization log')
        training_html += figure('training_loss.png')
    if validations:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for i, (ax, units) in enumerate(zip(axes, ['Steering request (rad)', 'Acceleration request (m/s²)'])):
            for metric in ['mae', 'p95']:
                ax.plot([v['step'] for v in validations], [v['groups']['all'][metric][i] for v in validations], marker='o', label=metric)
            ax.axhline([.01, .25][i], color='C0', linestyle='--', linewidth=.8, label='MAE ceiling')
            ax.axhline([.03, .75][i], color='C1', linestyle='--', linewidth=.8, label='p95 ceiling')
            ax.set(xlabel='Update', ylabel=units); ax.legend()
        fig.suptitle('Aggregate command errors. Every scene and stratum must also pass.')
        training_html += figure('validation.png')
    if measurement and measurement.get('samples'):
        samples = measurement['samples']
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for ax, key, label in zip(axes, ['gpu_pct', 'power_w', 'cpu_pct'], ['Whole GPU busy (%)', 'GPU power (W)', 'Whole host CPU (%)']):
            ax.plot([s['wall_s'] for s in samples], [s[key] for s in samples], lw=.8); ax.set_ylabel(label)
        axes[-1].set_xlabel('Job wall time (s)'); fig.suptitle('Actual whole-device and host samples, including any contention')
        training_html += figure('hardware.png')
        training_html += '<p>Measured job duration: ' + html.escape(str(measurement.get('wall_seconds'))) + ' s. Whole-device and host samples do not isolate this process. Duration is not FLOPS efficiency or a speedup.</p>'
    else:
        training_html += '<p>Hardware telemetry is pending; no whole-device measurement file was found at ' + html.escape(str(measure_path)) + '.</p>'

    episodes = []
    outcome_rows = []
    for directory in args.episodes:
        directory = directory.resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted(p for p in directory.rglob('*') if p.is_file() and ((p.suffix.lower() == '.mp4' and not p.name.endswith('-raw.mp4')) or (p.suffix.lower() == '.json' and any(term in p.name.lower() for term in ('summary', 'result', 'completion', 'audit', 'replay', 'verification'))) ))
        episodes.append({'directory': str(directory), 'files': [str(p) for p in files]})
        summary_path = directory / 'summary.json'
        if summary_path.is_file():
            result = read(summary_path)
            route = result.get('route') or {}
            values = [str(directory.relative_to(ROOT)), result.get('result'), result.get('recorded_frames'),
                      result.get('seconds_simulated'), result.get('physical_cone_hits'), result.get('oc'),
                      ','.join(route.get('accepted_route', [])) or 'none', result.get('pass')]
            outcome_rows.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>')
        training_html += '<h3>' + html.escape(str(directory)) + '</h3><p>Available episode artifacts: ' + (', '.join(local(p) for p in files) if files else 'none found') + '. Inspect the receipts for validity and outcome.</p>'
    if outcome_rows:
        training_html += '<h3>Recorded driving outcomes</h3><p>Each row is one exposed diagnostic rollout. These rejected-checkpoint runs are not final acceptance samples. Missing video audits remain unverified.</p><table><thead><tr>' + ''.join('<th>' + label + '</th>' for label in ['Episode', 'Result', 'Frames', 'Simulated seconds', 'Physical contacts', 'Off-course', 'Accepted skidpad route', 'Pass']) + '</tr></thead><tbody>' + ''.join(outcome_rows) + '</tbody></table>'

    videos = sorted((ROOT / 'runs/pi_correction_media/development_head_trial1').rglob('*.mp4'))
    video_links = '<ul>' + ''.join('<li>' + local(p) + '</li>' for p in videos) + '</ul>'
    evidence_paths = [BENCH / 'README.md', BENCH / 'native_transition_correction/README.md', BENCH / 'native_standing_reference/README.md', BENCH / 'reliability_command_probe/PROTOCOL.md', BENCH / 'reliability_appearance_probe/PROTOCOL.md', BENCH / 'native_transition_correction/preparation_failed1.json', ROOT / '.work/lab/direct_command_trial/preparation_failed1/recipe.json', ROOT / '.work/lab/direct_command_trial/preparation_failed2/actuator_trial.py', ROOT / '.work/lab/direct_command_trial/launch_freeze_receipt.json', ROOT / '.work/lab/direct_command_trial/launch_freeze_receipt2.json', ROOT / '.work/lab/direct_command_trial/launch_freeze_receipt3.json', BENCH / 'direct_command_setup/attempt3/audit.json', BENCH / 'direct_command_setup/attempt3/overfit.json', ROOT / 'benchmarks/pi-correction-2026-11-10/README.md', ROOT / 'benchmarks/pi-correction-2026-11-10/head_report_receipt.json', ROOT / 'benchmarks/pi-correction-2026-11-10/head_execution_setup.md', ROOT / 'runs/skidpad_head_correction_report/index.html']
    evidence = '<ul>' + ''.join('<li>' + local(p) + '</li>' for p in evidence_paths if p.is_file()) + '</ul>'
    for p in [BENCH / 'native_transition_correction/protocol.json', BENCH / 'native_standing_reference/protocol.json']:
        bind(p)
    summary = {'status': {'acceptance': 'NOT VERIFIED', 'training': 'FAILED_PROCESS' if failure is not None else 'PENDING' if training is None else 'COMPLETED_RECEIPT', 'episodes': 'NOT PROVIDED' if not episodes else 'ARTIFACTS PROVIDED; INSPECT AUDITS'}, 'native_arms': rows, 'native_audit_passed': audit['passed'], 'first_action_difference_tick': divergence['first_action_difference_tick'], 'command_probe': {'saved_frames': cmd['saved_frames'], 'saved_mismatches': cmd['saved_mismatches'], 'anchors': anchors}, 'appearance_stability': st, 'acceptance': {'conditions': len(suite['cases']), 'candidate_runs': suite['candidate_runs'], 'repeats_per_condition': suite['repeats_per_condition'], 'state': suite['state']}, 'training': {'directory': str(train_dir), 'steps_logged': len(steps), 'validation_steps': [v['step'] for v in validations], 'receipt': training, 'failure': failure, 'read_only_postflight': postflight, 'runtime_preflight': runtime, 'measurement': measurement}, 'episodes': episodes, 'sources': [{'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in sources]}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Driving reliability evidence</title><style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#1d2933}}img{{width:100%;height:auto}}.status{{background:#fff0cc;padding:1em}}code{{overflow-wrap:anywhere}}a{{color:#075b8c}}</style><h1>Driving reliability evidence</h1><p class="status"><strong>NOT VERIFIED.</strong> The final vision-driving acceptance suite remains unopened in the frozen record. No new checkpoint is promoted. These diagnostics do not establish reliable learned driving.</p>
<h2>Privileged native references</h2><p>The source supplies future states or recorded controls. These arms use no vision model to select the executed path. All six finish the full R,R,L,L route; three zero-contact interventions use a teacher-controlled prefix through tick 254. Standing OP + PI has six contacts, the original OP + PI suffix one, and the recorded teacher zero. The three intervened suffixes each have zero. One source cannot establish robustness.</p>{native_image}<p>Gray markers are initial cone positions, not moved-cone trajectories. Red crosses mark vehicle positions at contact-count increments, not contact-point coordinates. Time horizons follow recorded traces. The native control audit passed: {html.escape(str(audit['passed']))}. The first action difference occurs at tick {divergence['first_action_difference_tick']} at identical pose. Position feedback depends on privileged displacement and is not supplied by a neural plan starting at zero.</p>
<h2>Fixed-input model probes</h2><p>Command replay reproduced {cmd['saved_frames']} frames with {cmd['saved_mismatches']} saved-output mismatches. Four fixed-state anchors are shown. Frames 430 and 600 follow physical contacts. Frames 170 and 600 have no active nominal pulse, making the swap a null intervention. These are not closed-loop outcomes.</p>{command_image}<p>At one precontact frame 170, four baseline repeats met noise limits. Removing surface overlays changes near-term XY by {contrasts['no_surface']['magnitude']['near_xy_m']:.5f} m, above five times the largest repeat difference ({noise:.5f} m). Other interventions do not exceed their preset noise thresholds. This is not a general perception robustness test.</p>{appearance_image}
<h2>New direct-command training and episodes</h2><p>The added head has 33,538 parameters. The complete model has 31,188,340 parameters, including the 31,154,802-parameter base. Each trial records its own trainable scope and saved-state checks. Training metrics and observed hardware use are separate from driving qualification.</p>{training_html}<p>Each available first-error plot labels its source trial and contact events. Teacher values are privileged diagnostic labels at actual learner states and were not supplied to the learned controller. Learned raw requests are reconstructed from recorded latents with measured repeat error below 1e-5.</p>{diagnosis_image}
{recovery_html}<h2>Previous learned-driving evidence</h2><p>In the prior exposed paired comparison, both models finished the track with two contacts and one off-course event. The baseline skidpad stopped after R,R with 21 contacts; the diagnostic head rolled over after R,R with 13. Lower contacts came with earlier failure. These are individual runs, not reliability estimates.</p>{video_links}
<h2>Final acceptance</h2><p>The frozen suite specifies {len(suite['cases'])} conditions and {suite['candidate_runs']} candidate runs, four per condition, with zero contacts and zero off-course events required for every run. Repeats share physical realizations. They are not 64 independent scenes, and no universal reliability claim follows. Current acceptance remains NOT VERIFIED.</p>
<h2>Evidence and failed preparations</h2><p>Source file sizes and SHA-256 digests, including all linked videos and consumed traces, are in {link(OUT / 'summary.json', 'summary.json')}. The inherited report documents earlier failed pilot and setup attempts, including retained metadata and archives. The transition preparation and direct-command setup failures are linked below. Missing optional evidence is marked pending rather than inferred.</p>{evidence}</html>'''
    (OUT / 'index.html').write_text(page)
    print(f'Wrote {OUT / "index.html"} and {OUT / "summary.json"}')


if __name__ == '__main__':
    main()
