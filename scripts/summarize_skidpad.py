import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def driving(rows):
    return {'runs': len(rows), 'passed': sum(r['pass'] for r in rows),
            'clean_passes': sum(r['pass'] and r['doo'] == 0 and r['oc'] == 0 for r in rows),
            'total_cones': sum(r['doo'] for r in rows),
            'cones_per_run': sum(r['doo'] for r in rows) / len(rows),
            'off_course_events': sum(r['oc'] for r in rows)}


def command_metrics(path):
    rows = read(path)['episodes']
    count = sum(r['active_frames'] for r in rows)
    return {'episodes': len(rows), 'active_frames': count,
            'lateral_2s_active_m': {
                mode: sum(r['metrics'][mode]['lateral_2s_active'] * r['active_frames'] for r in rows) / count
                for mode in ('recorded', 'zero', 'swapped')}}


def main():
    root = Path('runs/spark_skidpad')
    protocol = read(root / 'final-protocol.json')
    assert hashlib.sha256(Path(protocol['checkpoint']).read_bytes()).hexdigest() == protocol['sha256']
    stages = read(root / 'training-summary.json')
    results = {name: read(root / name / 'driving.json')['runs'] for name in (
        'code_only_validation', 'policy_validation', 'vision_validation', 'recovery_validation',
        'centering_validation', 'final_test', 'retention', 'final_zero', 'final_swapped')}
    baseline = [r for r in read(Path('runs/spark_full/evaluation/closed_loop.json'))['runs']
                if r['discipline'] == 'trackdrive']
    assert len(results['final_test']) == 10 and len(results['retention']) == len(baseline) == 16
    commands = {name: command_metrics(root / folder / 'commands.json') for name, folder in (
        ('baseline', 'baseline_commands_new_data'), ('final', 'final_commands'))}
    total_seconds = sum(s['seconds'] for s in stages)
    result = {'protocol': protocol, 'checkpoint': read(root / 'final-checkpoint-audit.json'),
              'selection': read(root / 'selection.json'),
              'training': {'seconds': total_seconds, 'updates_executed': sum(s['steps'] for s in stages),
                           'gpu_busy_time_weighted_mean_percent': sum(
                               s['seconds'] * s['hardware_mean']['gpu_percent'] for s in stages) / total_seconds,
                           'stages': stages},
              'driving': {name: driving(rows) for name, rows in results.items()},
              'trackdrive_before': driving(baseline), 'command_response': commands,
              'video': read(root / 'video/video.json'), 'runs': results,
              'evaluation_timing': read(root / 'final-evaluation-timing.json')}
    output = Path('benchmarks/performance/skidpad-2026-11-09.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    x = np.arange(3)
    labels = ['Recorded', 'Zeroed', 'Left/right swapped']
    for i, name in enumerate(('baseline', 'final')):
        values = list(commands[name]['lateral_2s_active_m'].values())
        bars = axes[0].bar(x + (i - .5) * .36, values, width=.36, label=name.capitalize())
        axes[0].bar_label(bars, fmt='%.2f', padding=3)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel('2-second lateral prediction error (m)')
    axes[0].set_title('Same images, changed command history\n43 held-out skidpad episodes')
    axes[0].legend()
    rows = results['final_test']
    bars = axes[1].bar(range(10), [r['doo'] for r in rows],
                       color=['#288566' if r['pass'] else '#c94d45' for r in rows])
    axes[1].bar_label(bars)
    axes[1].set_xticks(range(10), [f"{r['seed']}\n{r['dr']}" for r in rows], rotation=45, ha='right')
    axes[1].set_ylabel('Cone hits')
    axes[1].set_title(f"Held-out driving | {sum(r['pass'] for r in rows)}/10 route passes\n"
                      'Green = route pass; red = route failure')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Skidpad evaluation | frozen checkpoint, realistic venue scenes')
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'graphs/results.{ext}', dpi=170)
    plt.close(fig)
    summary = {k: result[k] for k in ('training', 'driving', 'trackdrive_before', 'command_response')}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
