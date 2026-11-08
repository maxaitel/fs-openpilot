import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    root = Path('runs/skidpad_clean')
    output = root / 'graphs'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.2, 'font.size': 10})
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    hardware, hwaxes = plt.subplots(2, 1, figsize=(12, 6), layout='constrained')
    scalars, summaries = {}, []
    offset, elapsed = 0, 0
    for name, label in [('spark_skidpad_control', 'Direct steering supervision'),
                        ('spark_skidpad_clean', 'Recovery data trial (rejected)'),
                        ('spark_skidpad_steering2x', 'Double steering weight (rejected)')]:
        result = json.loads((root / f'{name}.json').read_text())
        events = EventAccumulator(str(Path('runs') / name / 'tb'), size_guidance={'scalars': 0})
        events.Reload()
        scalars[name] = {tag: [{'step': p.step, 'value': p.value} for p in events.Scalars(tag)]
                         for tag in events.Tags()['scalars']}
        for ax, tag, title in zip(axes.flat,
                                  ['train/plan', 'train/steer', 'val/lat_err_1s', 'val/lat_err_2s'],
                                  ['Training plan NLL', 'Weighted steering loss',
                                   'Validation lateral error at 1 s (m)', 'Validation lateral error at 2 s (m)'],
                                  strict=True):
            points = scalars[name].get(tag, [])
            ax.plot([p['step'] + offset for p in points], [p['value'] for p in points], label=label)
            if name == 'spark_skidpad_control':
                selected = next((p for p in points if p['step'] == 500), None)
                if selected is not None:
                    ax.scatter([500], [selected['value']], marker='*', s=110, color='black', zorder=3)
            ax.set_title(title)
            ax.set_xlabel('Optimizer updates executed')
        telemetry = result['telemetry']
        times = [(r['time'] - telemetry[0]['time'] + elapsed) / 60 for r in telemetry]
        for ax, key, title in zip(hwaxes, ['gpu_percent', 'system_used_gib'],
                                  ['GPU busy (%)', 'System memory used (GiB)'], strict=True):
            ax.plot(times, [r.get(key, float('nan')) for r in telemetry], lw=.8, label=label)
            ax.set_ylabel(title)
            ax.set_xlabel('Training minutes, including validation and saves')
        summaries.append({key: result[key] for key in
                          ['label', 'steps', 'fit_seconds', 'peak_allocated_gib', 'hardware_mean']})
        offset += result['steps']
        elapsed += result['fit_seconds']
    axes[0, 0].legend(fontsize=8)
    hwaxes[0].legend(fontsize=8)
    figure.suptitle('DGX Spark skidpad training experiments\n'
                   'Separate trials, including rejected runs; star = selected control checkpoint at step 500')
    hardware.suptitle('Measured hardware use | one-second samples')
    for fig, name in [(figure, 'training'), (hardware, 'hardware')]:
        fig.savefig(output / f'{name}.png', dpi=170)
        fig.savefig(output / f'{name}.pdf')
        plt.close(fig)
    (root / 'training-scalars.json').write_text(json.dumps(scalars, indent=2))
    (root / 'training-summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
