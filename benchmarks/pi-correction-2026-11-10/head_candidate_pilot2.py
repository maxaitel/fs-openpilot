import json
import subprocess
import sys
from pathlib import Path


root = Path('runs/head_trial1_pilot2')
root.mkdir(exist_ok=False)
plan = json.loads(Path('benchmarks/pi-correction-2026-11-10/head_evaluation/plan.json').read_text())
candidate = next(model for model in plan['models'] if model['name'] == 'candidate')
episode = root / 'episode'
commands = [
    [sys.executable, '-m', 'isaac_lab_fsop.evaluate',
     '--case', 'docs/isaac-lab-training/cases/train-skidpad-82000-matched.json',
     '--checkpoint', candidate['checkpoint'], '--expected-sha', candidate['sha256'],
     '--controller', 'op_tangent_pi_v1', '--pilot-steps', '120', '--out', str(episode)],
    [sys.executable, '-m', 'isaac_lab_fsop.verify_inference', '--episode', str(episode),
     '--checkpoint', candidate['checkpoint'], '--out', str(root / 'inference.json')],
    [sys.executable, '-m', 'isaac_lab_fsop.process', '--episode', str(episode), '--audit-only'],
]
for index, command in enumerate(commands):
    with (root / f'command_{index}.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if index == 0:
        completion = json.loads((episode / 'evidence/completion.json').read_text())
        assert completion['frame_count'] == 120
inference = json.loads((root / 'inference.json').read_text())
artifact = json.loads((episode / 'lab_artifact_audit.json').read_text())
assert inference['passed'] and artifact['passed']
summary = json.loads((episode / 'summary.json').read_text())
result = dict(checkpoint_sha256=candidate['sha256'], inference_passed=True, artifact_passed=True,
              frame_count=120, physical_cone_hits=summary['physical_cone_hits'],
              scope='integration pilot, not a completed mission')
with (root / 'receipt.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result))
