import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('runs/skidpad_clean')
    before = json.loads((root / 'fresh_before/driving.json').read_text())
    after = json.loads((root / 'fresh_after/driving.json').read_text())
    old = {(r['seed'], r['dr']): r for r in before['runs']}
    new = {(r['seed'], r['dr']): r for r in after['runs']}
    assert old.keys() == new.keys() and len(old) == 20
    cases = sorted(old, key=lambda x: (x[0], x[1] == 'heavy'))
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2, 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 1]}, layout='constrained')
    x = np.arange(len(cases))
    axes[0].bar(x - .2, [old[k]['doo'] for k in cases], .4, label='Before', color='#ae7457')
    axes[0].bar(x + .2, [new[k]['doo'] for k in cases], .4, label='After', color='#258b83')
    axes[0].set_xticks(x, [f'{k[0]} {k[1][0].upper()}' for k in cases], rotation=65, ha='right')
    axes[0].set_ylabel('Cone contacts')
    axes[0].set_title('Same 20 fresh scenes, no expert interventions')
    axes[0].legend()
    for i, key in enumerate(['passed', 'clean']):
        for j, (data, color) in enumerate([(before, '#ae7457'), (after, '#258b83')]):
            value = data['summary'][key]
            axes[1].bar(i + (j - .5) * .35, value, .35, color=color)
            axes[1].text(i + (j - .5) * .35, value + .35, str(value), ha='center')
    axes[1].set_xticks([0, 1], ['Route passes', 'Zero-contact\npasses'])
    axes[1].set_ylim(0, 22)
    axes[1].set_ylabel('Runs out of 20')
    fig.suptitle('Skidpad before and after retraining | checkpoint selected before these scenes')
    output = root / 'graphs'
    output.mkdir(exist_ok=True)
    fig.savefig(output / 'driving.png', dpi=170)
    fig.savefig(output / 'driving.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
