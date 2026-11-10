import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2] / 'cone_set_preflight/overfit.json'
    raw = source.read_bytes()
    record = json.loads(raw)
    errors = np.array([row['absolute_error'] for row in record['per_anchor']])
    groups = ['New skid recovery' if 'skidpad' in row['anchor']['case'] and 'recovery' in row['anchor']['case'] else 'New track recovery' if 'recovery' in row['anchor']['case'] else 'Original TRAIN' for row in record['per_anchor']]
    limits = np.array([.002, .05])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    names = ['Steering (rad)', 'Acceleration (m/s²)']
    for output, axis in enumerate(axes[:2]):
        for group, color in zip(['Original TRAIN', 'New skid recovery', 'New track recovery'], ['#526f9b', '#b37b2e', '#994c72']):
            index = [i for i, item in enumerate(groups) if item == group]
            axis.scatter(index, errors[index, output], label=group, s=24, color=color)
        axis.axhline(limits[output], color='black', linestyle='--', label='Mean-error limit')
        axis.axhline(record['mae'][output], color='#bd3434', label='Measured mean')
        axis.set(xlabel='Fixed TRAIN anchor', ylabel='Absolute error', title=names[output])
    axes[0].legend(fontsize=7)
    ratio = np.asarray(record['mae']) / limits
    axes[2].bar(['Steering', 'Acceleration'], ratio, color='#bd3434')
    axes[2].axhline(1, color='black', linestyle='--')
    axes[2].set(ylabel='Mean error / frozen limit', title='Both wiring gates fail')
    figure.suptitle(f"Cone-set v1: {record['step']} disposable updates, {len(errors)} TRAIN anchors")
    figure.savefig(args.out / 'capacity.png', dpi=160)
    plt.close(figure)
    receipt = {
        'inputs': {str(source): hashlib.sha256(raw).hexdigest()},
        'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256((args.out / 'capacity.png').read_bytes()).hexdigest(),
        'mae': record['mae'], 'capacity_mae_ceilings': limits.tolist(),
        'anchors': len(errors), 'actual_updates': record['step'],
        'whole_stage_seconds': record['seconds'],
        'limits': ['The gate applies to mean errors, not individual dots.',
                   'Only final metrics were retained; there is no per-step training curve.',
                   'Whole-stage time includes data verification, with no isolated speed claim.',
                   'This failed TRAIN capacity screen is not validation or driving.']}
    (args.out / 'plot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
