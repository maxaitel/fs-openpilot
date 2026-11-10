import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bench = Path(__file__).resolve().parents[2]
    source = bench / 'vision_corrected_plan_prototype/frozen_outputs/rows.json'
    rows = json.loads(source.read_text())
    episodes = sorted(set(r['episode'] for r in rows))
    fig, axes = plt.subplots(len(episodes), 2, figsize=(12, 7), layout='constrained')
    for i, episode in enumerate(episodes):
        selected = [r for r in rows if r['episode'] == episode]
        time = [r['timestamp'] for r in selected]
        for variant, label, color in [('uncorrected_pp', 'Uncorrected NN path', '#526f9b'), ('corrected_pp', 'Cone-offset path', '#bd3434')]:
            axes[i, 0].plot(time, [r['scores'][variant]['lateral_error_m'] for r in selected], color=color, label=label)
            axes[i, 1].plot(time, [r['scores'][variant]['gt_sampled_footprint_margin_m'] for r in selected], color=color, label=label)
        for j in range(2):
            axes[i, j].axhline(0, color='black', linewidth=.7)
            axes[i, j].set(xlabel='Recorded simulated time (s)', title=episode)
        axes[i, 0].set_ylabel('Signed target lateral error (m)')
        axes[i, 1].set_ylabel('Static sampled footprint margin (m)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Rejected constant-offset prototype: identical precontact TRAIN inputs')
    output = args.out / 'geometry.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    receipt = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'frames': len(rows), 'episodes': episodes,
               'limits': ['Two correlated prefixes of one TRAIN scene.',
                          'Static sampled path margins are not actual contacts or reachable trajectories.',
                          'All selected rows shown, including unsupported proposals.',
                          'Zero is the conflict boundary after the frozen heuristic pad, not a guaranteed safety bound.']}
    (args.out / 'plot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
