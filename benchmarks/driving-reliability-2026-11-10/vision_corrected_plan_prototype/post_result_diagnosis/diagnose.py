"""Post-result attribution only. Privileged geometry never enters the controller."""
from pathlib import Path
from types import SimpleNamespace
from itertools import islice
from hashlib import sha256
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT.parent / 'vision_corridor_proto_v1'))
from scorer import ReferenceScorer, world_to_ego
from fsop.sim.vehicle import VehicleState
from isaac_eval.evidence import read_evidence
from controller import project
sys.path.insert(0, str(OUT))
from replay import CONFIG, P, SCREEN, digest, frozen_inputs, metric, preserve


def tangent_at(path, station):
    path = np.asarray(path)
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    ss = np.r_[0., np.cumsum(ds)]
    i = min(max(int(np.searchsorted(ss, station, side='right')) - 1, 0), len(ds) - 1)
    return (path[i + 1] - path[i]) / ds[i]


def run():
    frozen_inputs()
    prior = json.loads((OUT / 'rows.json').read_text())
    rows = []
    for case in CONFIG['selection']['cases']:
        batch = list(islice(read_evidence(ROOT / case['path'] / 'evidence'), case['first_contact_frame']))
        metadata = batch[0][0]
        cached = json.loads((ROOT / 'benchmarks/driving-reliability-2026-11-10/direct_command_diagnosis' / case['diagnostic']).read_text())
        scorer = ReferenceScorer(metadata, SimpleNamespace(**P), SCREEN, teacher_params=cached['source_teacher_parameters'])
        scorer.bind_pilot_labels([r for _, r, _ in batch], cached)
        episode_rows = {r['frame']: r for r in prior if r['episode'] == Path(case['path']).name}
        for _, record, _ in batch:
            row = episode_rows[record['frame']]
            prop = row['proposal']
            if not prop['residuals']:
                continue
            ordered = sorted(prop['residuals'], key=lambda r: (r['residual_m'], r['track_id']))
            weights = np.cumsum([r['weight'] for r in ordered])
            selected = ordered[int(np.searchsorted(weights, weights[-1] / 2))]
            track = next(t for t in prop['tracks'] if t['identifier'] == selected['track_id'])
            state = VehicleState(**record['state'])
            gt = world_to_ego(record['actual_cones_xyz'][:, :2], state)
            distance = np.linalg.norm(gt - track['xy'], axis=1)
            index = int(np.argmin(distance))
            gt_class = {'blue': 0, 'yellow': 1, 'orange': 2, 'big_orange': 3}[metadata['cone_colors'][index]]
            class_match = track['cone_class'] == gt_class or track['cone_class'] in (2, 3) and gt_class in (2, 3)
            s0 = record['progress']['s']
            world = [scorer.mission.point_at(s0 + float(s)) for s in np.arange(-2., 14.001, .1)]
            reference = world_to_ego(world, state)
            reference = reference[np.r_[True, np.linalg.norm(np.diff(reference, axis=0), axis=1) > 1e-8]]
            rs, rd = project(gt[index], reference)
            path = np.asarray(prop['variants']['uncorrected_pp']['path'])
            ps, pd = project(gt[index], path)
            pn, rn = tangent_at(path, ps), tangent_at(reference, rs)
            normal_angle = float(abs(np.arctan2(pn[0] * rn[1] - pn[1] * rn[0], np.dot(pn, rn))))
            gt_prior_residual = pd - selected['side'] * P['nominal_halfwidth_m']['skidpad']
            reference_residual = rd - selected['side'] * P['nominal_halfwidth_m']['skidpad']
            base_error = row['scores']['uncorrected_pp']['lateral_error_m']
            corrected_error = row['scores']['corrected_pp']['lateral_error_m']
            rows.append({'episode': row['episode'], 'frame': row['frame'], 'route_phase': row['scores']['corrected_pp']['phase'],
                         'median_track_id': track['identifier'], 'median_track_class': track['cone_class'],
                         'nearest_gt_cone': index, 'nearest_gt_class': gt_class, 'class_match': class_match,
                         'nearest_gt_distance_m': float(distance[index]),
                         'nearest_within_075m_diagnostic_only': bool(distance[index] <= .75),
                         'median_residual_m': selected['residual_m'], 'offset_m': prop['offset_m'],
                         'offset_clipped': abs(selected['residual_m']) > P['max_offset_m'],
                         'gt_projected_prior_residual_m': gt_prior_residual,
                         'predicted_minus_gt_prior_residual_m': selected['residual_m'] - gt_prior_residual,
                         'gt_reference_boundary_residual_m': reference_residual,
                         'prior_vs_reference_normal_angle_rad': normal_angle,
                         'uncorrected_lateral_error_m': base_error, 'corrected_lateral_error_m': corrected_error,
                         'correction_reduces_absolute_error': abs(corrected_error) < abs(base_error),
                         'correction_change_m': corrected_error - base_error,
                         'correction_opposes_required_direction': (corrected_error - base_error) * (-base_error) < 0,
                         'correct_direction_but_overshoots': ((corrected_error - base_error) * (-base_error) > 0 and abs(corrected_error) > abs(base_error))})
    summaries = {}
    for name in ['total'] + [Path(c['path']).name for c in CONFIG['selection']['cases']]:
        group = rows if name == 'total' else [r for r in rows if r['episode'] == name]
        matched = [r for r in group if r['nearest_within_075m_diagnostic_only']]
        compatible = [r for r in matched if r['class_match']]
        summaries[name] = {'supported_median_rows': len(group), 'matched_gt_within_075m': len(matched),
                           'matched_class_compatible': len(compatible),
                           'matched_nearest_distance_m': metric([r['nearest_gt_distance_m'] for r in matched]),
                           'prediction_transport_residual_error_m': metric([r['predicted_minus_gt_prior_residual_m'] for r in matched]),
                           'class_compatible_gt_boundary_residual_m': metric([r['gt_reference_boundary_residual_m'] for r in compatible]),
                           'class_compatible_prior_reference_normal_angle_rad': metric([r['prior_vs_reference_normal_angle_rad'] for r in compatible]),
                           'class_compatible_boundary_residual_over_05m': sum(abs(r['gt_reference_boundary_residual_m']) > .5 for r in compatible),
                           'improved_rows': sum(r['correction_reduces_absolute_error'] for r in group),
                           'wrong_direction_rows': sum(r['correction_opposes_required_direction'] for r in group),
                           'correct_direction_overshoot_rows': sum(r['correct_direction_but_overshoots'] for r in group),
                           'offset_clipped_rows': sum(r['offset_clipped'] for r in group),
                           'baseline_error_over_025m_rows': sum(abs(r['uncorrected_lateral_error_m']) > .25 for r in group),
                           'maximum_error_frame': max(group, key=lambda r: abs(r['corrected_lateral_error_m']))}
    result = {'status': 'Post-result exploratory attribution. No controller changes or new performance screen.',
              'diagnostic_source_sha256': digest(Path(__file__)), 'rows_sha256': digest(OUT / 'rows.json'),
              'nearest_association_limit': '0.75 m is an inherited diagnostic heuristic; matching is neither identity nor recall proof.',
              'decomposition_limit': 'Canonical route projection uses the selected mission near current progress. Large boundary residual can reflect wrong boundary association, station/end effects, or model shape. It does not isolate nominal width by itself.',
              'summaries': summaries, 'rows': rows}
    preserve(OUT / 'attribution.json', result)
    print(json.dumps(summaries, indent=2))

if __name__ == '__main__':
    run()
