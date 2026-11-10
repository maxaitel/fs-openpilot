"""Privileged offline scoring of exactly two sealed precontact TRAIN prefixes."""
from collections import Counter
from hashlib import sha256
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import json
import platform
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT.parent / 'vision_corridor_proto_v1'))
from scorer import ReferenceScorer, world_to_ego
from fsop.sim.vehicle import VehicleState
from isaac_eval.evidence import read_evidence, verified_summary
from controller import CorrectedPlan, margins, observed

CONFIG = json.loads((OUT / 'config.json').read_text())
P, SCREEN = CONFIG['planner'], CONFIG['screen']


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def preserve(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f'preserving different prior output {path}')
    else:
        path.write_text(encoded)


def frozen_inputs():
    frozen = json.loads((OUT / 'freeze.json').read_text())
    for section in ('source_sha256', 'input_sha256'):
        for name, expected in frozen[section].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'frozen file changed {name}')
    return frozen


def metric(values):
    x = np.asarray([v for v in values if v is not None], float)
    return None if not len(x) else {'n': len(x), 'mae': float(np.mean(abs(x))),
                                   'p95': float(np.percentile(abs(x), 95)), 'max_abs': float(max(abs(x)))}


def counts(values):
    values = [v for v in values if v is not None]
    return {'eligible': len(values), 'agree': sum(values)}


def summarize(rows):
    out = {'frames': len(rows), 'supported_frames': sum(r['proposal']['status'] == 'supported' for r in rows),
           'phase_modulo_4_counts': dict(sorted(Counter(r['frame'] % 4 for r in rows).items())),
           'route_phase_counts': dict(Counter(r['scores']['corrected_pp']['phase'] for r in rows)),
           'reason_counts': dict(Counter(r['proposal']['reason'] for r in rows)),
           'offset_m': metric([r['proposal']['offset_m'] for r in rows]),
           'variants': {}}
    out['support_fraction'] = out['supported_frames'] / len(rows)
    supported = [r for r in rows if r['proposal']['status'] == 'supported']
    for name in ('uncorrected_pp', 'corrected_pp'):
        commands = [r['proposal']['variants'][name]['control'] for r in rows]
        scored = [r['scores'][name] for r in rows]
        conditional = [r['scores'][name] for r in supported]
        out['variants'][name] = {
            'all_candidate_lateral_error_m': metric([s['lateral_error_m'] for s in scored]),
            'supported_lateral_error_m': metric([s['lateral_error_m'] for s in conditional]),
            'all_candidate_heading_error_rad': metric([s['heading_error_rad'] for s in scored]),
            'all_row_requested_steer_error_rad': metric([s['command_error'][0] for s in scored]),
            'all_row_geometric_pp_steer_error_rad': metric([r['proposal']['variants'][name]['control']['geometric_steer_rad'] - r['scores'][name]['teacher_command'][0] for r in rows]),
            'all_row_acceleration_error_mps2_diagnostic_only': metric([s['command_error'][1] for s in scored]),
            'teacher_correction_direction': counts([s['teacher_correction_direction_agrees'] for s in scored]),
            'supported_wrong_branch_proxy_count': sum(s['wrong_branch_proxy'] is True for s in conditional),
            'all_candidate_wrong_branch_proxy_count': sum(s['wrong_branch_proxy'] is True for s in scored),
            'supported_gt_footprint_conflict_count': sum(s['gt_sampled_footprint_margin_m'] <= 0 for s in conditional),
            'all_candidate_gt_footprint_conflict_count': sum(s['gt_sampled_footprint_margin_m'] <= 0 for s in scored),
            'minimum_gt_margin_m': min(s['gt_sampled_footprint_margin_m'] for s in scored),
            'first_hit_cone_margin_m': metric([s['first_hit_cone_margin_m'] for s in scored]),
            'predicted_conflict_frames': sum(c['predicted_path_conflict'] for c in commands),
            'positive_accel_frames': sum(c['accel_mps2'] > 0 for c in commands),
            'braking_frames': sum(c['accel_mps2'] < 0 for c in commands),
            'unknown_rows_with_nonnegative_accel': sum(r['proposal']['status'] == 'unknown' and r['proposal']['variants'][name]['control']['accel_mps2'] >= 0 for r in rows),
            'stop_distance_shortfall_frames': sum(c['required_stop_distance_m'] > c['available_stop_distance_m'] for c in commands)}
    out['raw_old_actuator_head'] = {'steer_error_rad': metric([r['raw_head']['error'][0] for r in rows]),
                                  'acceleration_error_mps2_diagnostic_only': metric([r['raw_head']['error'][1] for r in rows]),
                                  'teacher_correction_direction': counts([r['raw_head']['correction_agrees'] for r in rows])}
    out['correction_delta_direction_toward_teacher'] = counts([r['correction_delta_agrees'] for r in rows])
    corrected, baseline = (out['variants'][n] for n in ('corrected_pp', 'uncorrected_pp'))
    g, gs = corrected['all_candidate_lateral_error_m'], corrected['supported_lateral_error_m']
    tests = {
        'support': out['support_fraction'] >= SCREEN['minimum_support_fraction'],
        'all_candidate_geometry': bool(g and g['mae'] <= SCREEN['geometry_lateral_mae_limit_m'] and g['p95'] <= SCREEN['geometry_lateral_p95_limit_m']),
        'supported_geometry': bool(gs and gs['mae'] <= SCREEN['geometry_lateral_mae_limit_m'] and gs['p95'] <= SCREEN['geometry_lateral_p95_limit_m']),
        'lateral_improvement': bool(g and g['mae'] < baseline['all_candidate_lateral_error_m']['mae']),
        'direction_no_regression': corrected['teacher_correction_direction']['agree'] >= baseline['teacher_correction_direction']['agree'],
        'no_supported_wrong_branch_proxy': corrected['supported_wrong_branch_proxy_count'] <= SCREEN['maximum_wrong_branch_proxy_count'],
        'no_supported_gt_footprint_conflict': corrected['supported_gt_footprint_conflict_count'] <= SCREEN['maximum_supported_gt_conflict_count']}
    out['frozen_geometry_screen_components'] = tests
    out['frozen_geometry_screen_pass'] = all(tests.values())
    return out


def main():
    frozen = frozen_inputs()
    episodes, all_rows = {}, []
    for case in CONFIG['selection']['cases']:
        path = ROOT / case['path']
        summary = verified_summary(path / 'evidence')
        boundary = min(e['control_frame'] for e in summary['contact_events'])
        if boundary != case['first_contact_frame']:
            raise ValueError('precontact boundary changed')
        first_event = next(e for e in summary['contact_events'] if e['control_frame'] == boundary)
        first_cone = int(first_event['cone'].rsplit('c', 1)[1])
        batch = list(islice(read_evidence(path / 'evidence'), boundary))
        metadata, records = batch[0][0], [r for _, r, _ in batch]
        if metadata['case']['split'] != CONFIG['selection']['split'] or metadata['case']['id'] != CONFIG['selection']['case_id']:
            raise ValueError('unexpected episode selection')
        if metadata['state_sha256'] != CONFIG['selection']['original_state_sha256'] or metadata['fps'] != 20:
            raise ValueError('not frozen original model at 20 Hz')
        if [r['frame'] for r in records] != list(range(boundary)):
            raise ValueError('incomplete precontact prefix')
        diagnostic = json.loads((ROOT / 'benchmarks/driving-reliability-2026-11-10/direct_command_diagnosis' / case['diagnostic']).read_text())
        for relative, expected in diagnostic['input_sha256'].items():
            if relative in frozen['input_sha256'] and frozen['input_sha256'][relative] != expected:
                raise ValueError('canonical diagnostic binding changed')
        scorer = ReferenceScorer(metadata, SimpleNamespace(**P), SCREEN, teacher_params=diagnostic['source_teacher_parameters'])
        scorer.bind_pilot_labels(records, diagnostic)
        planner, rows = CorrectedPlan(P), []
        for record in records:
            obs = observed(record, metadata['case']['discipline'])
            proposal = planner.step(obs)
            scores = {}
            state = VehicleState(**record['state'])
            gt_cones = world_to_ego(record['actual_cones_xyz'][:, :2], state)
            for name, variant in proposal['variants'].items():
                result = SimpleNamespace(path=variant['path'], control=SimpleNamespace(**variant['control']))
                score = scorer.score(result, record['frame'], record)
                per_cone = margins(np.asarray(variant['path']), gt_cones, P)
                score['gt_per_cone_margins_m'] = per_cone.tolist()
                score['first_hit_cone_margin_m'] = float(per_cone[first_cone])
                scores[name] = score
            teacher = scores['corrected_pp']['teacher_command']
            raw = record['prediction']['controller']['raw']
            eligible = abs(teacher[0]) > SCREEN['teacher_correction_min_rad']
            unc = proposal['variants']['uncorrected_pp']['control']['geometric_steer_rad']
            cor = proposal['variants']['corrected_pp']['control']['geometric_steer_rad']
            row = {'episode': path.name, 'frame': record['frame'], 'timestamp': obs.timestamp,
                   'proposal': proposal, 'scores': scores,
                   'raw_head': {'command': raw, 'error': [raw[i] - teacher[i] for i in (0, 1)],
                                'correction_agrees': bool(raw[0] * teacher[0] > 0) if eligible else None},
                   'correction_delta_agrees': bool((cor - unc) * (teacher[0] - unc) > 0) if eligible else None}
            rows.append(row)
        episodes[path.name] = {'summary': summarize(rows), 'teacher_reconstruction_max_abs': scorer.reconstruction_max_abs,
                               'first_contact_excluded_frame': boundary, 'first_hit_cone_index_scoring_only': first_cone,
                               'per_control_phase': {str(phase): summarize([r for r in rows if r['frame'] % 4 == phase]) for phase in range(4)}}
        all_rows.extend(rows)
    result = {'scope': '133 original-model TRAIN precontact frames on one skidpad entry. No counterfactual trajectory or full RRLL claim.',
              'freeze_sha256': digest(OUT / 'freeze.json'), 'python': platform.python_version(), 'numpy': np.__version__,
              'total': summarize(all_rows), 'episodes': episodes,
              'advance': all(e['summary']['frozen_geometry_screen_pass'] for e in episodes.values()),
              'physical_gate': 'Unchanged and untested. Final unopened.'}
    preserve(OUT / 'rows.json', all_rows)
    preserve(OUT / 'result.json', result)
    print(json.dumps({'advance': result['advance'], 'total': result['total'],
                      'episodes': {k: v['summary'] for k, v in episodes.items()}}, indent=2))


if __name__ == '__main__':
    main()
