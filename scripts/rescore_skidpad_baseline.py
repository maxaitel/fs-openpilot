import json
from pathlib import Path

import numpy as np

from fsop.sim.skidpad import SkidpadRoute
from fsop.sim.vehicle import VehicleState
from fsop.track.fixed_layouts import skidpad


def main():
    root = Path('runs/spark_skidpad/baseline_test')
    rows = []
    for path in sorted(root.glob('*.npz')):
        route = SkidpadRoute(skidpad())
        for t, x, y, yaw, vx, _s, _lat in np.load(path)['states']:
            route.advance(VehicleState(t=t, x=x, y=y, yaw=yaw, vx=vx))
        legacy = json.loads(path.with_suffix('.json').read_text())
        route.finish(legacy['result'])
        rows.append({'scene': path.stem, 'legacy': legacy, 'route': route.report()})
    report = {'runs': rows, 'limitation': 'Saved baseline traces omit lateral velocity and the final step. '
              'This audit uses longitudinal speed for the motion plausibility check. '
              'It is an offline geometric route audit, not a new closed-loop run.'}
    (root / 'physical-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps([{'scene': r['scene'], 'route': r['route']['accepted_route'],
                      'violation': r['route']['violation']} for r in rows], indent=2))


if __name__ == '__main__':
    main()
