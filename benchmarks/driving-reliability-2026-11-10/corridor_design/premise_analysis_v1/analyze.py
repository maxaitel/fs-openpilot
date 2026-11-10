import hashlib
import json
from pathlib import Path

import numpy as np

from fsop.sim.expert import Expert, ExpertParams, Mission
from fsop.sim.vehicle import VehicleParams, VehicleState
from fsop.track.fixed_layouts import skidpad

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
THRESHOLDS = OUT.parent / 'premise_diagnostic_thresholds_v1.json'
limits = json.loads(THRESHOLDS.read_text())
sources = [
    ('initial_head', 'result.json', 'direct_command_pilot3'),
    ('recovery_head', 'result_recovery2.json', 'direct_command_recovery_pilot2'),
]
inputs = [Path(__file__).resolve(), THRESHOLDS]
reports = {}
for name, diagnostic_name, episode_name in sources:
    diagnostic_path = ROOT / 'benchmarks/driving-reliability-2026-11-10/direct_command_diagnosis' / diagnostic_name
    episode = ROOT / 'runs' / episode_name
    metadata_path = episode / 'evidence/metadata.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    assert metadata['case']['split'] == 'train'
    assert metadata['case']['discipline'] == 'skidpad'
    for path in (metadata_path, episode / 'evidence/manifest.jsonl', episode / 'evidence/completion.json', episode / 'summary.json'):
        relative = str(path.relative_to(ROOT))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == diagnostic['input_sha256'][relative], relative
        inputs.append(path)
    inputs.append(diagnostic_path)
    track = skidpad(seed=metadata['track']['seed'])
    generated = json.loads(json.dumps(track.to_dict()))
    assert {k: v for k, v in generated.items() if k != 'stats'} == {k: v for k, v in metadata['track'].items() if k != 'stats'}
    vp = VehicleParams(**metadata['vehicle_params'])
    teachers = {}
    for value in limits['parameter_values']:
        params = {**diagnostic['source_teacher_parameters'], 'k_lookahead': value}
        teachers[value] = Expert(Mission(track, ExpertParams(**params)), vp)
    first_contact = min(event['control_frame'] for event in diagnostic['contact_events'])
    rows = []
    for saved in diagnostic['rows']:
        if saved['frame'] >= first_contact:
            continue
        state = VehicleState(**saved['state'])
        progress = saved['progress']['s']
        baseline = np.asarray(teachers[0.25].act(state, progress))
        repeated = np.asarray(teachers[0.25].act(state, progress))
        alternate = np.asarray(teachers[0.15].act(state, progress))
        reconstruction = np.abs(baseline - saved['native_teacher_at_actual_state'])
        assert reconstruction.max() <= limits['teacher_reconstruction_max_abs']
        assert np.array_equal(baseline, repeated)
        changed_proprio = state.copy()
        changed_proprio.r += 0.1
        changed_proprio.vy += 0.1
        assert np.array_equal(baseline, teachers[0.25].act(changed_proprio, progress))
        recorded_head = np.asarray(saved['fixed_latent_head_requests']['recorded'])
        zero_r = np.asarray(saved['fixed_latent_head_requests']['zero_r'])
        rows.append({
            'frame': saved['frame'], 'time_s': saved['time_s'],
            'baseline_teacher': baseline.tolist(), 'alternate_teacher': alternate.tolist(),
            'absolute_parameter_gap': np.abs(alternate - baseline).tolist(),
            'reconstruction_max_abs': float(reconstruction.max()),
            'head_error': (recorded_head - baseline).tolist(),
            'head_zero_r_delta': (zero_r - recorded_head).tolist(),
        })
    gaps = np.array([row['absolute_parameter_gap'] for row in rows])
    errors = np.array([row['head_error'] for row in rows])
    yaw_effect = np.array([row['head_zero_r_delta'] for row in rows])
    bins = []
    for start in np.arange(0, first_contact / 20, 0.5):
        ids = [i for i, row in enumerate(rows) if start <= row['time_s'] < start + .5]
        if ids:
            bins.append({'start_s': float(start), 'count': len(ids),
                         'signed_mean_error': errors[ids].mean(0).tolist(),
                         'mae': np.abs(errors[ids]).mean(0).tolist()})
    reports[name] = {
        'case': metadata['case'], 'first_contact_interval': first_contact,
        'precontact_count': len(rows), 'reconstruction_max_abs': max(row['reconstruction_max_abs'] for row in rows),
        'repeat_max_abs': 0.0,
        'teacher_invariant_to_vy_plus_0p1_r_plus_0p1': True,
        'parameter_gap_mean': gaps.mean(0).tolist(), 'parameter_gap_p95': np.percentile(gaps, 95, axis=0).tolist(),
        'parameter_gap_max': gaps.max(0).tolist(),
        'material_pair_count': (gaps > [limits['material_steering_pair_gap_rad'], limits['material_acceleration_pair_gap_mps2']]).sum(0).tolist(),
        'equal_weight_paired_mae_lower_bound': (gaps.mean(0) / 2).tolist(),
        'head_precontact_mae': np.abs(errors).mean(0).tolist(),
        'head_precontact_p95': np.percentile(np.abs(errors), 95, axis=0).tolist(),
        'head_zero_r_delta_max_abs': np.abs(yaw_effect).max(0).tolist(),
        'head_zero_r_material_count': (np.abs(yaw_effect) > [0.02, 0.5]).sum(0).tolist(),
        'half_second_bins': bins, 'rows': rows,
    }
inputs += [ROOT / 'src/fsop' / path for path in ['sim/expert.py', 'sim/vehicle.py', 'track/fixed_layouts.py', 'track/centerline.py', 'track/rules.py', 'track/schema.py']]
report = {
    'scope': limits['scope'], 'interpretation': limits['interpretation'],
    'limits': limits,
    'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    'numpy_version': np.__version__, 'cases': reports,
    'limitations': ['Correlated ticks from two rejected pilots on one TRAIN case are not independent scenes.',
                    'Teacher-parameter interventions change the requested label only, with no new trajectory or image.',
                    'These are retained pilot states, not the three TRAIN episode populations with k_lookahead 0.15.',
                    'A counterfactual pair is not an observed duplicate-input collision in the original data.',
                    'Existing head interventions come from the saved diagnostic receipts. No policy inference was run.'],
}
output = OUT / 'result.json'
encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
if output.exists():
    assert output.read_text() == encoded, 'Existing diagnostic differs; preserve the original and version the analysis.'
else:
    output.write_text(encoded)
print(json.dumps({key: {k: v for k, v in value.items() if k not in ('rows', 'half_second_bins', 'case')} for key, value in reports.items()}, indent=2))
