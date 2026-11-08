import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fsop.sim.env import FSEnv, SimCfg
from fsop.sim.vehicle import VehicleState
from fsop.track.fixed_layouts import skidpad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob('**/*.npz')):
        if not path.stem.startswith(('light-', 'heavy-')):
            continue
        level, seed = path.stem.split('-')
        cfg = SimCfg(dr=level, venue_prob=1.0, max_time=120, render=False)
        cfg.noise.enabled = False
        env = FSEnv(skidpad(), cfg, seed=int(seed))
        with np.load(path) as data:
            states = data['states']
            physical = np.vstack([data['vehicle_states'], data['final_vehicle_state']])
        hits = []
        for i, state in enumerate(physical):
            for event in env.referee.update(VehicleState(*state)):
                if event.kind != 'doo':
                    continue
                cone = event.data['cone']
                role = env.track.cones[env.cone_src[cone]].role
                delta = env.cones_xy[cone] - state[:2]
                c, s = np.cos(state[2]), np.sin(state[2])
                local = delta @ np.array([[c, -s], [s, c]])
                hits.append({'time': float(state[8]), 'cone': int(cone), 'role': role,
                             'cone_in_body_frame': local.tolist(),
                             'lateral_error': float(states[min(i, len(states) - 1), 6])})
        result = json.loads(path.with_suffix('.json').read_text())
        if len(hits) != result['doo']:
            raise ValueError(f'Contact reconstruction disagrees for {path}')
        rows.append({'seed': int(seed), 'dr': level, 'contacts': hits})
    if not rows:
        raise ValueError('No saved vehicle traces found')
    report = {'runs': rows, 'roles': dict(Counter(h['role'] for r in rows for h in r['contacts']))}
    (root / 'contacts.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
