"""Diagnose the rejected direct-command pilot with fixed-state head interventions."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path('.work/lab/joint_command_trial').resolve()))
from joint_core import CommandHead, CONTRACT
from fsop.sim.expert import ExpertParams
from isaac_eval.evidence import read_evidence, verified_summary
from isaac_eval.verify_inference import observation
from isaac_lab_fsop.cases import Case, environment
from isaac_lab_fsop.expert import NativeExpert

EPISODE = Path('runs/joint_command_pilot1')
PACKAGE = Path('runs/joint_command_trial1_archive/step_500.pt')
SOURCE = Path('runs/vision_corridor_probe_v1/references_data/train-skidpad-82000-matched')
OUT = Path('.work/lab/direct_command_diagnosis/result_joint1.json')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

torch.set_num_threads(4)
assert not OUT.exists()
evidence = EPISODE / 'evidence'
metadata = json.loads((evidence / 'metadata.json').read_text())
summary = verified_summary(evidence)
assert summary is not None and sha(PACKAGE) == metadata['checkpoint_sha256']
source = json.loads((SOURCE / 'meta.json').read_text())
assert source['case'] == metadata['case']
assert source['vehicle'] == metadata['vehicle_params']
states = np.load(SOURCE / 'states.npy', allow_pickle=False)
assert sha(SOURCE / 'states.npy') == source['arrays_sha256']['states.npy']
env = environment(Case(**metadata['case']))
teacher = NativeExpert(env, params=ExpertParams(**source['teacher']))
package = torch.load(PACKAGE, map_location='cpu', weights_only=False)
assert package['command_contract'] == CONTRACT
head = CommandHead().requires_grad_(False).eval()
head.load_state_dict(package['head_state'], strict=True)
rows, manifests = [], []
maximum_repeat = 0.0
with torch.inference_mode():
    for _, record, entry in read_evidence(evidence):
        obs = observation(record, env)
        expected = teacher.act()
        current = record['prediction']['controller']
        latent = torch.tensor([current['latent']], dtype=torch.float32)
        proprio = np.asarray(current['proprio'], dtype=np.float32)
        tick = int(record['frame'])
        variants = {'recorded': proprio.copy(), 'zero_r': proprio.copy(), 'zero_vy': proprio.copy(),
                    'zero_vy_r': proprio.copy(), 'source_r': proprio.copy(),
                    'source_vy_r': proprio.copy(), 'r_plus_0p1': proprio.copy(), 'r_minus_0p1': proprio.copy()}
        variants['zero_r'][2] = 0
        variants['zero_vy'][1] = 0
        variants['zero_vy_r'][1:] = 0
        variants['source_r'][2] = states[tick, 5]
        variants['source_vy_r'][1:] = states[tick, 4:6]
        variants['r_plus_0p1'][2] += .1
        variants['r_minus_0p1'][2] -= .1
        values = {name: head(latent, torch.from_numpy(p[None])).numpy()[0].tolist() for name, p in variants.items()}
        residual = float(np.max(np.abs(np.asarray(values['recorded']) - current['raw'])))
        maximum_repeat = max(maximum_repeat, residual)
        assert np.allclose(values['recorded'], current['raw'], rtol=1e-5, atol=1e-5)
        before = (env.state.r, env.state.vy)
        env.state.r += .1
        env.state.vy += .1
        teacher_intervention = teacher.act()
        env.state.r, env.state.vy = before
        assert expected == teacher_intervention
        state = record['state']
        rows.append({'frame': tick, 'time_s': state['t'], 'state': state, 'progress': record['progress'],
                     'actual_direct_action': record['action'], 'native_teacher_at_actual_state': list(expected),
                     'raw_request_error': (np.asarray(current['raw']) - expected).tolist(),
                     'source_clock_position_error_m': float(np.linalg.norm(obs.state.as_array()[:2] - states[tick, :2])),
                     'fixed_latent_head_requests': values, 'fixed_latent_proprio_inputs': {k:v.tolist() for k,v in variants.items()},
                     'earlier_physical_contacts': record['counters']['physical_contacts']})
        manifests.append({'frame': tick, 'sha256': entry['sha256']})
fixed = [PACKAGE, SOURCE / 'meta.json', SOURCE / 'states.npy', evidence / 'metadata.json', evidence / 'manifest.jsonl',
         evidence / 'completion.json', EPISODE / 'summary.json', Path(__file__)]
result = {'scope': 'Rejected joint-suffix TRAIN pilot diagnosis. Ground-truth teacher actions are diagnostic labels only. Fixed-state head interventions are not driving results.',
          'input_sha256': {str(p): sha(p) for p in fixed}, 'evidence_frames': manifests,
          'recorded_head_repeat_max_abs': maximum_repeat,
          'teacher_target_invariant_to_r_vy_intervention': True,
          'source_teacher_parameters': source['teacher'], 'contact_events': summary['contact_events'],
          'rows': rows, 'limitations': ['Source-clock displacement includes speed phase differences.',
                     'Substituting source telemetry at a different physical state is only a matched-latent diagnostic.',
                     'The privileged expert uses track geometry and is never supplied to the learned policy.']}
with OUT.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
print(json.dumps({'frames': len(rows), 'contacts': summary['contact_events'], 'maximum_head_repeat': maximum_repeat}, indent=2))
for row in rows:
    if row['frame'] % 10 == 0:
        print(row['frame'], 'steer/pred/teacher', row['actual_direct_action'][0], row['native_teacher_at_actual_state'][0],
              'r', row['state']['r'], 'vy', row['state']['vy'], 'error', row['raw_request_error'],
              'r0steer', row['fixed_latent_head_requests']['zero_r'][0])
