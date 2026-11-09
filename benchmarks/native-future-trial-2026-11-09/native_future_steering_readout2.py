"""Describe recorded steering and a measured-yaw-rate substitution; no driving claim."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fsop.eval.controller import OpenpilotTangentFollower
from fsop.sim.expert import T_IDXS
from fsop.sim.vehicle import VehicleParams
from isaac_eval.evidence import _unpack


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


p = argparse.ArgumentParser()
p.add_argument('--episode', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert not a.out.exists()
root = a.episode
completion = json.loads((root/'evidence/completion.json').read_text())
audit = json.loads((root/'lab_artifact_audit.json').read_text())
assert audit['passed'] and audit['completion_sha256'] == sha(root/'evidence/completion.json')
assert audit['frames_sha256'] == sha(root/'frames.json')
entries = [json.loads(s) for s in (root/'evidence/manifest.jsonl').read_text().splitlines()]
assert completion['manifest_sha256'] == sha(root/'evidence/manifest.jsonl')
realization = json.loads((root/'realization.json').read_text())
vp = VehicleParams(**realization['vehicle'])
ctrl = OpenpilotTangentFollower(vp)
indices = sorted(set(range(0, 201, 4)) | set(range(400, 541, 4)) | {88, 93, 116, 132, 465, 469})
rows = []
for i in indices:
    entry = entries[i]
    path = root/'evidence'/entry['file']
    assert entry['frame'] == i and sha(path) == entry['sha256']
    with np.load(path, allow_pickle=False) as arrays:
        r = _unpack(json.loads(arrays['__record_json__'].tobytes()), arrays)
    state, plan = r['state'], r['prediction']['plan']
    steer, _ = ctrl(plan, state['vx'], state['vy'])
    assert np.isclose(steer, r['action'][0], atol=1e-8, rtol=1e-9)
    psi = float(np.interp(.2, T_IDXS, plan[:, 11]))
    R = np.asarray(r['body']['rotation'])
    omega = np.asarray(r['body']['angular_velocity_world'])
    axis = R[:, 0]
    yawdot = float(omega[2] - axis[2]*(axis[0]*omega[0]+axis[1]*omega[1])/(axis[0]**2+axis[1]**2))
    assert np.isclose(state['yaw'], np.arctan2(R[1, 0], R[0, 0]), atol=1e-6)
    assert np.isclose(state['r'], omega[2], atol=1e-6)
    measured = float(np.arctan(vp.wheelbase * (2*psi/.2-state['r'])/max(state['vx'], 1.)))
    rows.append(dict(frame=i, t=state['t'], s=r['progress']['s'], lat=r['progress']['lat'],
                     speed=state['vx'], actual_yaw_rate=state['r'], exact_heading_derivative=yawdot,
                     angular_velocity_world=omega.tolist(), body_forward_axis_world=axis.tolist(), predicted_initial_yaw_rate=float(plan[0, 14]),
                     yaw_at_02=psi, requested_steer=steer, actual_steer=state['steer'],
                     measured_yaw_substitution_steer=measured, delta_steer=measured-steer,
                     frame_sha256=entry['sha256']))
frames = json.loads((root/'frames.json').read_text())
s = np.array([r['s'] for r in frames]); xy = np.array([[r['x'], r['y']] for r in frames])
yaw = np.unwrap(np.array([r['yaw'] for r in frames]))
for row in rows:
    i = row['frame']
    row['central_heading_slope_01s'] = float((yaw[i+1]-yaw[i-1])/.1) if i > 0 and i+1 < len(frames) else None
ds = np.diff(s); moved = np.linalg.norm(np.diff(xy, axis=0), axis=1)
jumps = [{'frame': i+1, 't': frames[i+1]['t'], 'progress_change_m': float(ds[i]),
          'physical_displacement_m': float(moved[i])} for i in range(len(ds)) if abs(ds[i]) > moved[i]+2.]
result = dict(scope='Saved-record description and single-input controller substitution, no physics rerun or improvement claim.',
              episode=str(root), checkpoint_sha256=json.loads((root/'summary.json').read_text())['checkpoint_sha256'],
              input_sha256={str(root/n):sha(root/n) for n in ['summary.json','frames.json','realization.json','lab_artifact_audit.json','evidence/completion.json','evidence/manifest.jsonl']},
              script_sha256=sha(Path(__file__)), rows=rows, progress_jumps_over_displacement_plus_2m=jumps,
              route_decisions=realization['route_decisions'], mission=realization['mission'])
a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'sampled_frames': len(rows), 'progress_jumps': len(jumps), 'output': str(a.out)}))
for r in rows:
    if r['frame'] in [64,80,88,93,116,132,440,460,465,469,480,520]:
        print(json.dumps({k:r[k] for k in ['frame','t','lat','actual_yaw_rate','predicted_initial_yaw_rate','exact_heading_derivative','central_heading_slope_01s','requested_steer','measured_yaw_substitution_steer','delta_steer']}))
