import hashlib
import json
from pathlib import Path

import numpy as np

from fsop.sim.vehicle import VehicleParams, VehicleState, step
from isaac_eval.evidence import read_evidence, verified_summary

ROOT = Path('.work/lab/native_plant_forecast_probe_v1')
CASES = [('runs/direct_command_pilot3', 80), ('runs/direct_command_recovery_pilot2', 53),
         ('runs/joint_command_pilot1', 82)]
HORIZONS = (1, 5, 10, 20)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stats(values):
    x = np.asarray(values, dtype=float)
    return {'max': float(x.max()), 'p95': float(np.percentile(x, 95))}

def score(records, first, params, ticks):
    errors = []
    for i in range(first - ticks):
        s0 = records[i]['state']
        st = VehicleState(vx=s0['vx'], vy=s0['vy'], r=s0['r'], steer=s0['steer'], accel=s0['accel'])
        for k in range(i, i + ticks):
            steer, accel = records[k]['action']
            for _ in range(5):
                st = step(st, steer, accel, params, 0.01)
        end = records[i + ticks]['state']
        dx, dy = end['x'] - s0['x'], end['y'] - s0['y']
        c, sn = np.cos(s0['yaw']), np.sin(s0['yaw'])
        pos = float(np.hypot(st.x - (c * dx + sn * dy), st.y - (-sn * dx + c * dy)))
        yaw = abs(float(np.arctan2(np.sin(st.yaw - (end['yaw'] - s0['yaw'])),
                                   np.cos(st.yaw - (end['yaw'] - s0['yaw'])))))
        speed = abs(st.speed - float(np.hypot(end['vx'], end['vy'])))
        errors.append((pos, yaw, speed, pos + 1.6 * yaw))
    arr = np.asarray(errors)
    return {'count': len(arr), 'position_m': stats(arr[:, 0]), 'yaw_rad': stats(arr[:, 1]),
            'speed_mps': stats(arr[:, 2]), 'front_effect_m': stats(arr[:, 3]),
            'front_effect_over_0_2_m': int((arr[:, 3] > 0.2).sum())}

out = {'protocol_sha256': sha(ROOT / 'protocol.md'), 'script_sha256': sha(__file__), 'cases': []}
for case, first in CASES:
    path = Path(case) / 'evidence'
    summary = verified_summary(path)
    assert min(e['control_frame'] for e in summary['contact_events']) == first
    records = []
    metadata = None
    for metadata, record, _ in read_evidence(path):
        assert metadata['case']['split'] == 'train'
        assert record['frame'] == len(records)
        assert abs(record['state']['t'] - record['frame'] * 0.05) < 1e-7
        records.append({'state': record['state'], 'action': record['action']})
    assert len(records) >= first
    raw_params = dict(metadata['vehicle_params'])
    raw_params['v_blend'] = tuple(raw_params['v_blend'])
    params = VehicleParams(**raw_params)
    result = {'case': case, 'first_contact': first, 'frames_verified': len(records),
              'source_sha256': {name: sha(path / name) for name in ('metadata.json', 'manifest.jsonl', 'completion.json')},
              'parameter_matches_nominal': params == VehicleParams(), 'results': {}}
    for label, vp in [('nominal', VehicleParams()), ('privileged_metadata', params)]:
        result['results'][label] = {str(t * 0.05): score(records, first, vp, t) for t in HORIZONS}
    out['cases'].append(result)
(ROOT / 'result.json').write_text(json.dumps(out, indent=2) + '\n')
