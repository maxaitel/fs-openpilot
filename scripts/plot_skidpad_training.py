import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=['spark_skidpad_policy'])
    args = parser.parse_args()
    output = Path('runs/spark_skidpad')
    graphs = output / 'graphs'
    graphs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.2, 'font.size': 10})
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    hardware, hwaxes = plt.subplots(3, 1, figsize=(12, 8), layout='constrained')
    summaries, scalars = [], {}
    audit_path = output / 'final-checkpoint-audit.json'
    selections = json.loads(audit_path.read_text())['stage_checkpoints'] if audit_path.exists() else {}
    offset, elapsed = 0, 0.0
    for run in args.runs:
        result = json.loads((output / f'{run}.json').read_text())
        events = EventAccumulator(str(Path('runs') / run / 'tb'), size_guidance={'scalars': 0})
        events.Reload()
        scalars[run] = {tag: [{'step': p.step, 'value': p.value} for p in events.Scalars(tag)]
                        for tag in events.Tags()['scalars']}
        label = run.removeprefix('spark_skidpad_').replace('_', ' ').capitalize()
        for ax, tag, title in zip(axes.flat, ['train/plan', 'val/plan', 'val/lat_err_1s', 'val/lat_err_2s'],
                                  ['Training plan loss (Laplace NLL)', 'Validation plan loss (Laplace NLL)',
                                   'Lateral error at 1 s (m)',
                                   'Lateral error at 2 s (m)'], strict=True):
            points = scalars[run].get(tag, [])
            ax.plot([p['step'] + offset for p in points], [p['value'] for p in points], label=label)
            chosen = selections.get(run.removeprefix('spark_skidpad_'), {}).get('selected_step')
            selected = next((p for p in points if p['step'] == chosen), None)
            if selected is not None:
                ax.scatter([selected['step'] + offset], [selected['value']], marker='*', s=90,
                           color=ax.lines[-1].get_color(), edgecolor='black', linewidth=.5, zorder=3)
            ax.set_title(title)
            ax.set_xlabel('Cumulative optimizer updates executed')
        telemetry = result['telemetry']
        start = telemetry[0]['time']
        times = [(row['time'] - start + elapsed) / 60 for row in telemetry]
        for ax, key, title in zip(hwaxes, ['gpu_percent', 'power_w', 'system_used_gib'],
                                  ['GPU busy (%)', 'GPU power (W)', 'System memory used (GiB)'], strict=True):
            ax.plot(times, [row.get(key, np.nan) for row in telemetry], lw=0.8, label=label)
            ax.set_ylabel(title)
            ax.set_xlabel('Training minutes, including validation')
        log_path = output / ('policy.log' if run == 'spark_skidpad_policy' else f'{run}.log')
        points = re.findall(r'step\s+(\d+).*?\|\s+([\d.]+) samples/s', log_path.read_text())
        every, log_every = result['config']['val_every'], result['config']['log_every']
        steady = [float(v) for s, v in points if int(s) > log_every and (int(s) - log_every) % every]
        summaries.append({'run': run, 'steps': result['steps'], 'seconds': result['fit_seconds'],
                          'batch_size': result['config']['batch_size'],
                          'effective_samples_per_second': result['steps'] * result['config']['batch_size']
                          / result['fit_seconds'],
                          'steady_samples_per_second_median': float(np.median(steady)),
                          'peak_allocated_gib': result['peak_allocated_gib'],
                          'hardware_mean': result['hardware_mean']})
        offset += result['steps']
        elapsed += result['fit_seconds']
    axes[0, 0].legend(fontsize=8)
    hwaxes[0].legend(fontsize=8)
    figure.suptitle('DGX Spark skidpad adaptation | held-out validation\n'
                   'Stars = selected checkpoints; each next stage starts from the selected checkpoint\n'
                   'Validation scenes stay fixed; sampled frame counts vary by stage', fontsize=12)
    hardware.suptitle('Measured training hardware use | one-second samples')
    for fig, name in [(figure, 'training'), (hardware, 'hardware')]:
        fig.savefig(graphs / f'{name}.png', dpi=170)
        fig.savefig(graphs / f'{name}.pdf')
        plt.close(fig)
    (output / 'training-summary.json').write_text(json.dumps(summaries, indent=2))
    (output / 'training-scalars.json').write_text(json.dumps(scalars, indent=2))


if __name__ == '__main__':
    main()
