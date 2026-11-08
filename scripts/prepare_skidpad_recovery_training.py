import argparse
import json
from pathlib import Path

import numpy as np

from fsop.data.sim_dataset import hash_str, load_index, split_episodes


def link_episode(target, source):
    source = source.resolve()
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise ValueError(f'Existing episode points to a different source: {target}')
    else:
        target.symlink_to(source, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recovery', nargs='+', default=['data/sim/skidpad_recovery_r1'])
    parser.add_argument('--output', default='data/sim/spark_skidpad_recovery')
    parser.add_argument('--full-replay', action='store_true')
    args = parser.parse_args()
    source = Path('data/sim/spark_skidpad_intent')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    original = {row['id']: row for row in load_index(source)}
    rng = np.random.default_rng(85000)
    train = split_episodes(source, 'train')
    selected = list(original) if args.full_replay else []
    if not args.full_replay:
        for discipline, count in [('skidpad', 32), ('trackdrive', 64)]:
            eligible = [eid for eid in train if original[eid]['discipline'] == discipline]
            selected.extend(rng.choice(eligible, count, replace=False).tolist())
        selected += split_episodes(source, 'val') + split_episodes(source, 'test')
    rows = []
    for eid in selected:
        target = output / eid
        link_episode(target, source / eid)
        rows.append(original[eid])
    audit = []
    recovery_rows = [(Path(root), row) for root in args.recovery for row in load_index(root)]
    assert len({row['id'] for _, row in recovery_rows}) == len(recovery_rows)
    for recovery, row in recovery_rows:
        eid = row['id']
        assert hash_str(eid) % 1000 >= 200, eid
        episode = recovery / eid
        meta = json.loads((episode / 'meta.json').read_text())
        assert meta['seed'] >= 85000, eid
        plans = np.load(episode / 'plan.npy')
        commands = np.load(episode / 'desire.npy')
        states = np.load(episode / 'states.npy')
        assert len(plans) == len(commands) == len(states) >= 20, eid
        assert np.isfinite(plans).all() and np.isfinite(commands).all() and np.isfinite(states).all(), eid
        assert meta['route_commands']['schema'] == 'next_junction_pulse_v1'
        audit.append({'id': eid, 'frames': len(plans), 'seed': meta['seed'],
                      'summary': meta['summary'], 'collection': meta['collection'],
                      'command_counts': commands.sum(0).tolist(), 'max_abs_lat': float(abs(states[:, 10]).max())})
        target = output / eid
        link_episode(target, episode)
        rows.append(row)
    (output / 'index.json').write_text(json.dumps({'episodes': rows}, indent=2))
    assert split_episodes(output, 'val') == split_episodes(source, 'val')
    assert split_episodes(output, 'test') == split_episodes(source, 'test')
    assert len({item['seed'] for item in audit}) == len(audit)
    report = {'dataset': str(output), 'episodes': len(rows), 'frames': sum(row['frames'] for row in rows),
              'recovery_episodes': len(audit), 'full_replay': args.full_replay,
              'replay_train_counts': {discipline: sum(eid in train and original[eid]['discipline'] == discipline
                                                    for eid in selected) for discipline in ('skidpad', 'trackdrive')},
              'validation_unchanged': True, 'test_unchanged': True, 'recovery': audit}
    name = 'centering-data.json' if args.full_replay else 'recovery-data.json'
    (Path('runs/spark_skidpad') / name).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'recovery'}, indent=2))


if __name__ == '__main__':
    main()
