import json
import os
import shutil
from pathlib import Path

from scripts.run_lab_evaluations import atomic_json, digest, load_plan, prepare, validate_episode


old = Path('runs/development_pi_trial1').resolve()
new = Path('runs/development_pi_trial1_retry1').resolve()
assert not new.exists()
lock = load_plan('benchmarks/pi-correction-2026-11-10/evaluation/plan.json')
prior_lock = json.loads((old / 'plan.lock.json').read_text())
assert prior_lock == lock
prior_manifest = json.loads((old / 'manifest.json').read_text())
assert prior_manifest['lock_sha256'] == digest(old / 'plan.lock.json')
assert prior_manifest['episodes']['candidate/heavy_skid']['status'] == 'failed'
assert prior_manifest['episodes']['candidate/heavy_skid']['error'] == 'RuntimeError: Free space fell below the 25 GiB reserve'
case = next(row for row in lock['cases'] if row['name'] == 'heavy_skid')
model = next(row for row in lock['models'] if row['name'] == 'baseline')
key = 'baseline/heavy_skid'
source = old / key
receipt = validate_episode(source, case, model, lock)
assert prior_manifest['episodes'][key]['status'] == 'complete'
assert prior_manifest['episodes'][key]['receipt'] == receipt
assert not any(path.is_symlink() for path in source.rglob('*'))
new, manifest = prepare(new, lock)
(new / 'baseline').mkdir()
shutil.copytree(source, new / key, copy_function=os.link)
assert validate_episode(new / key, case, model, lock) == receipt
manifest['episodes'][key] = {**prior_manifest['episodes'][key], 'imported_from': str(source)}
atomic_json(new / 'manifest.json', manifest)
provenance = dict(
    reason='Retain the completed baseline and rerun only the disk-interrupted candidate in a new batch.',
    prior_manifest_sha256=digest(old / 'manifest.json'), prior_plan_sha256=digest(old / 'plan.lock.json'),
    imported_episode=str(source), imported_receipt=receipt,
    retained_failed_attempt=str(old / 'candidate/heavy_skid'), shared_files='immutable hardlinks',
)
with (new / 'retry_provenance.json').open('x') as stream:
    json.dump(provenance, stream, indent=2)
print(json.dumps(provenance))
