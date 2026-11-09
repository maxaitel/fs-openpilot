import argparse
import hashlib
import json
from pathlib import Path

from fsop.eval.controller import controller_contract

p = argparse.ArgumentParser()
p.add_argument('--selection', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
receipt = json.loads(a.selection.read_text())
protocol = Path('docs/isaac-lab-training/native_future_selection1.json')
assert receipt['protocol'] == json.loads(protocol.read_text())
assert receipt['kind'] == 'op_tangent_v1_selection'
assert receipt['schema_version'] == 2
assert receipt['action_target_contract'] == receipt['protocol']['action_target_contract']
if receipt['trained_candidate']:
    assert receipt['status'] == 'trained_candidate_selected' and receipt['selected_eligible']
    candidate = receipt['selected']
    scope = 'selected_candidate_exposed_development'
else:
    assert receipt['status'] == 'retain_step_zero'
    candidate = next(c for c in receipt['candidates'] if c['step'] == 500)
    scope = 'rejected_step500_diagnosis_only_no_promotion_no_fresh_tests'

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

assert digest(candidate['checkpoint']) == candidate['checkpoint_sha256']
initializer = Path('runs/lab_v4/native_future_calibration1/step_000000.pt').resolve()
models = [
    dict(name='baseline', checkpoint='/frozen/model.pt', sha256='cf3a8d253dfd5a417e1967d2d6dc8cdd0263e3d16d3f1f02f15495f490a83b7e', controller_contract=controller_contract('pp')),
    dict(name='initializer', checkpoint=str(initializer), sha256=digest(initializer), controller_contract=controller_contract('op_tangent_v1')),
    dict(name='candidate', checkpoint=candidate['checkpoint'], sha256=candidate['checkpoint_sha256'], controller_contract=controller_contract('op_tangent_v1')),
]
for m in models:
    assert digest(m['checkpoint']) == m['sha256']
a.out.mkdir(parents=True, exist_ok=False)
cases = {
    'heavy_skid': 'regression-skidpad-72000-heavy-recorded-visual_stress_v1',
    'heavy_track': 'regression-retention-trackdrive_020000-heavy-recorded-visual_stress_v1',
}
for name, case in cases.items():
    path = Path('docs/isaac-lab-training/cases') / (case + '.json')
    assert json.loads(path.read_text())['id'] == case
    plan = dict(version=2, timeout_seconds=7200, cases=[str(path.resolve())], models=models)
    (a.out / (name+'.json')).write_text(json.dumps(plan, indent=2)+'\n')
interpretation = dict(scope=scope, selection_sha256=digest(a.selection), protocol_sha256=digest(protocol), diagnostic_step=candidate['step'], models=models,
                     comparisons=['baseline versus candidate measures the controller/model package', 'initializer versus candidate uses the same controller to assess training'],
                     case_count_per_model=2, fresh_tests_opened=False)
(a.out/'interpretation.json').write_text(json.dumps(interpretation, indent=2)+'\n')
print(json.dumps(interpretation))
