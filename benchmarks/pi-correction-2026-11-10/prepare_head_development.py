import argparse
import hashlib
import json
from pathlib import Path

from fsop.eval.controller import controller_contract


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--selection', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
receipt = json.loads(args.selection.read_text())
protocol_path = Path('benchmarks/pi-correction-2026-11-10/head_selection1.json')
protocol = json.loads(protocol_path.read_text())
assert receipt['protocol'] == protocol
assert receipt['schema_version'] == 4
assert receipt['kind'] == 'op_tangent_pi_v1_selection'
assert receipt['action_target_contract'] == protocol['action_target_contract']
if receipt['trained_candidate']:
    assert receipt['status'] == 'trained_candidate_selected' and receipt['selected_eligible']
    candidate = receipt['selected']
    scope = 'selected_candidate_exposed_development'
else:
    assert receipt['status'] == 'retain_step_zero'
    candidate = next(row for row in receipt['candidates'] if row['step'] == 500)
    scope = 'rejected_step500_diagnosis_only_no_promotion_no_fresh_tests'
initializer = Path('probes/pi_models/latest_pi_train.pt').resolve()
assert digest(initializer) == '48db741c8a686b2e635ae7a81b4e4c08d79fbfbd9f4dc4c163260da6f93f0cfb'
models = [
    dict(name='baseline', checkpoint=str(initializer), sha256=digest(initializer)),
    dict(name='candidate', checkpoint=candidate['checkpoint'], sha256=candidate['checkpoint_sha256']),
]
for model in models:
    assert digest(model['checkpoint']) == model['sha256']
    model['controller_contract'] = controller_contract('op_tangent_pi_v1')
case_ids = {
    'heavy_skid': 'regression-skidpad-72000-heavy-recorded-visual_stress_v1',
    'heavy_track': 'regression-retention-trackdrive_020000-heavy-recorded-visual_stress_v1',
}
cases = []
for name, case_id in case_ids.items():
    path = Path('docs/isaac-lab-training/cases') / f'{case_id}.json'
    assert json.loads(path.read_text())['id'] == case_id
    cases.append(dict(name=name, path=str(path.resolve())))
args.out.mkdir(parents=True, exist_ok=False)
plan = dict(version=2, timeout_seconds=1800, cases=cases, models=models)
interpretation = dict(
    scope=scope, selection_sha256=digest(args.selection), protocol_sha256=digest(protocol_path),
    evaluated_step=candidate['step'], models=models, case_count_per_model=2,
    comparison='unchanged latest weights versus retrained weights under the same PI controller',
    fresh_tests_opened=False,
)
for name, value in [('plan.json', plan), ('interpretation.json', interpretation)]:
    with (args.out / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
print(json.dumps(interpretation))
