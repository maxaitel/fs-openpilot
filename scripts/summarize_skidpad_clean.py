import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summary(data):
    rows = data['runs']
    times = [r['raw_time'] for r in rows if r['pass']]
    return {'runs': len(rows), 'passed': sum(r['pass'] for r in rows),
            'clean': sum(r['pass'] and r['doo'] == 0 and r['oc'] == 0 for r in rows),
            'cones': sum(r['doo'] for r in rows),
            'cones_per_run': sum(r['doo'] for r in rows) / len(rows),
            'off_course': sum(r['oc'] for r in rows),
            'median_timed_lap_seconds_on_passes': statistics.median(times) if times else None}


def main():
    root = Path('runs/skidpad_clean')
    names = ['baseline_pp', 'control_step500', 'control_step1000', 'control_step1500',
             'recovery_step500', 'recovery_step1000', 'recovery_step1500',
             'steering2x_step500', 'steering2x_step1000', 'steering2x_step1500',
             'control_regression', 'control_retention', 'fresh_before', 'fresh_after',
             'commands_zero', 'commands_swapped']
    evaluations = {name: read(root / name / 'driving.json') for name in names}
    training = []
    for name in ['spark_skidpad_control', 'spark_skidpad_clean', 'spark_skidpad_steering2x']:
        result = read(root / f'{name}.json')
        training.append({key: result[key] for key in ['label', 'steps', 'fit_seconds', 'peak_allocated_gib',
                                                     'hardware_mean', 'config', 'source_sha256']})
    before = {(r['seed'], r['dr']): r for r in evaluations['fresh_before']['runs']}
    after = {(r['seed'], r['dr']): r for r in evaluations['fresh_after']['runs']}
    assert before.keys() == after.keys() and len(before) == 20
    difference = [after[k]['doo'] - before[k]['doo'] for k in before]
    old_cones = sum(r['doo'] for r in before.values())
    report = {'protocol': read(root / 'final-protocol.json'), 'checkpoint': read(root / 'checkpoint-audit.json'),
              'training': training, 'successful_training_seconds': sum(r['fit_seconds'] for r in training),
              'driving': {name: summary(data) for name, data in evaluations.items()},
              'paired_fresh': {'fewer_contacts': sum(x < 0 for x in difference),
                               'same_contacts': sum(x == 0 for x in difference),
                               'more_contacts': sum(x > 0 for x in difference),
                               'contact_reduction_fraction': -sum(difference) / old_cones if old_cones else None},
              'trackdrive_before': read(Path('benchmarks/performance/skidpad-2026-11-09.json'))['driving']['retention'],
              'video': read(root / 'video/video.json'),
              'previous_failure_video': read(root / 'video_previous_failure/video.json'),
              'evaluation_timing': read(root / 'final-timing.json'),
              'runs': {name: data['runs'] for name, data in evaluations.items()}}
    destination = Path('benchmarks/performance/skidpad-clean-2026-11-09.json')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    keys = ['successful_training_seconds', 'driving', 'paired_fresh']
    print(json.dumps({key: report[key] for key in keys}, indent=2))


if __name__ == '__main__':
    main()
