"""Freeze native-future action ceilings while retaining previous plan/coherence bounds."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import yaml

from fsop.train.action_target import NATIVE_FUTURE_ACTION_TARGET, NATIVE_FUTURE_CONTRACT
from scripts.select_lab_tangent import TRAINING_ARGS, protocol_check

ROOT = Path(__file__).resolve().parents[2]
OLD_CAL = ROOT / 'benchmarks/native-tangent-calibration-2026-11-09/calibration2/calibration.json'
OLD_RUN = OLD_CAL.with_name('run.json')
OLD_PROTOCOL = ROOT / 'docs/isaac-lab-training/tangent_selection2.json'
CAL_CONFIG = ROOT / '.work/lab/native_future_calibration1.yaml'
PROTOCOL = ROOT / 'docs/isaac-lab-training/native_future_selection1.json'
CONFIG = ROOT / '.work/lab/native_future_trial1.yaml'
RECEIPT = ROOT / '.work/lab/native_future_protocol1_receipt.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(calibration):
    cal_path, run_path = calibration / 'calibration.json', calibration / 'run.json'
    cal, run, old, old_run, previous = map(read, (cal_path, run_path, OLD_CAL, OLD_RUN, OLD_PROTOCOL))
    require(sha(OLD_CAL) == '94f21a4d3741611578f38f83242ae8b9c7222eb65fff6b31db1188f1f6205300',
            'old calibration bytes changed')
    require(sha(OLD_PROTOCOL) == 'f1ce3776281955b0a1b899a201089cea8b099b24c89c9959caa84468f5fb17ba',
            'previous bounds protocol bytes changed')
    protocol_check(previous)
    require(run['calibration_only'] is True and run['args']['calibration_only'] is True,
            'requires a zero-update calibration')
    require(not (calibration / 'history.jsonl').exists(), 'calibration contains optimizer history')
    require(run['args']['out'] == 'runs/lab_v4/native_future_calibration1', 'unexpected calibration output')
    require(run['args'].get('action_target_mode') == NATIVE_FUTURE_ACTION_TARGET, 'wrong action target mode')
    require({k: v for k, v in run['args'].items() if k not in ('out', 'action_target_mode')} ==
            {k: v for k, v in old_run['args'].items() if k != 'out'}, 'calibration recipe changed')
    require(cal['version'] == old['version'] == 1, 'calibration schema changed')
    for document in (run, cal['identity'], cal['validation']):
        require(document.get('action_target_contract') == NATIVE_FUTURE_CONTRACT, 'target contract mismatch')
    require({k: v for k, v in cal['identity'].items()
             if k not in ('action_target_contract', 'source_files_sha256')} ==
            {k: v for k, v in old['identity'].items() if k != 'source_files_sha256'},
            'calibration identity changed beyond target contract and source')
    require(run['baseline_state_sha256'] == old_run['baseline_state_sha256'], 'initializer state changed')
    for key in ('init_sha256', 'manifest_sha256', 'dataset_verification', 'episodes', 'validation_selection',
                'train_anchors', 'val_anchors', 'trainable_tensors', 'trainable_parameters', 'recovery_sampling'):
        require(run[key] == old_run[key], f'run identity changed: {key}')
    source = run['source']
    current = {path.relative_to(ROOT / 'src').as_posix(): sha(path)
               for path in sorted((ROOT / 'src/fsop').rglob('*.py'))}
    require(current == source['files_sha256'] == cal['identity']['source_files_sha256'],
            'calibration source differs from current source inventory')
    require(sha(calibration / 'source.zip') == source['archive_sha256'], 'calibration source archive changed')
    with zipfile.ZipFile(calibration / 'source.zip') as archive:
        require(len(archive.namelist()) == len(current) and set(archive.namelist()) == set(current),
                'source archive inventory mismatch')
        require(all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name, digest in current.items()),
                'source archive bytes mismatch')
    validation, old_validation = cal['validation'], old['validation']
    for key in ('count', 'precision', 'aggregation', 'coherence_definition', 'target_coherence', 'coherence'):
        require(validation[key] == old_validation[key], f'zero-update validation changed: {key}')
    for key in ('plan_l1', 'plan_nll'):
        require(validation['aggregate'][key] == old_validation['aggregate'][key], f'baseline {key} changed')
    old_episodes = {row['path']: row for row in old_validation['episodes']}
    require([r['path'] for r in validation['episodes']] == list(old_episodes), 'validation cases reordered or changed')
    for row in validation['episodes']:
        prior = old_episodes[row['path']]
        for key in ('count', 'coherence', 'target_coherence', 'plan_l1', 'plan_nll'):
            require(row[key] == prior[key], f'episode zero-update {key} changed: {row["path"]}')
        for key in ('steer', 'longitudinal'):
            require(row[f'teacher_{key}'] == prior[key], f'legacy teacher diagnostic changed: {row["path"]}')
    for key in ('steer', 'longitudinal'):
        require(validation['aggregate'][f'teacher_{key}'] == old_validation['aggregate'][key],
                f'aggregate legacy teacher diagnostic changed: {key}')
    baseline = read(calibration / 'baseline_validation.json')
    require(baseline['step'] == 0 and all(baseline[key] == value for key, value in validation.items()),
            'saved baseline does not bind calibration validation')
    bounds = deepcopy(previous['bounds'])
    for key in ('steer', 'longitudinal'):
        bounds['aggregate'][key] = validation['aggregate'][key]
        for row in validation['episodes']:
            bounds['episode'][Path(row['path']).name][key] = row[key] * 1.10
    p = {
        'schema_version': 2, 'kind': 'op_tangent_v1_selection', 'calibration_sha256': sha(cal_path),
        'initializer_sha256': cal['identity']['init_sha256'], 'manifest_sha256': cal['identity']['manifest_sha256'],
        'controller_contract': cal['identity']['controller_contract'],
        'action_target_contract': NATIVE_FUTURE_CONTRACT,
        'expected_steps': 500, 'checkpoint_steps': [0, 100, 200, 300, 400, 500],
        'training_args': {**{key: run['args'][key] for key in TRAINING_ARGS},
                          'action_target_mode': NATIVE_FUTURE_ACTION_TARGET},
        'bounds': bounds,
    }
    protocol_check(p)
    config = yaml.safe_load(CAL_CONFIG.read_text())
    require(all(run['args'].get(k) == v for k, v in config.items() if k != 'backend'),
            'calibration config differs from actual run')
    config.update(out='runs/lab_v4/native_future_trial1', calibration_only=False,
                  expected_calibration='runs/lab_v4/native_future_calibration1/calibration.json',
                  selection_protocol='docs/isaac-lab-training/native_future_selection1.json')
    inputs = (cal_path, run_path, calibration / 'baseline_validation.json', calibration / 'source.zip',
              OLD_CAL, OLD_RUN, OLD_PROTOCOL, CAL_CONFIG, Path(__file__))
    return p, config, {str(path.relative_to(ROOT)): sha(path) for path in inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=ROOT / '.work/lab/native_future_calibration1')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    p, config, inputs = build(args.calibration.resolve())
    protocol_bytes = (json.dumps(p, indent=2, allow_nan=False) + '\n').encode()
    config['selection_protocol_sha'] = hashlib.sha256(protocol_bytes).hexdigest()
    config_bytes = yaml.safe_dump(config, sort_keys=False).encode()
    receipt = {'version': 1, 'input_sha256': inputs, 'source_files_sha256':
               read(args.calibration / 'run.json')['source']['files_sha256'],
               'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
               'training_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
               'action_target_contract': NATIVE_FUTURE_CONTRACT,
               'retained_bounds': ['aggregate.plan_l1', 'episode.*.plan_l1', 'coherence', 'episode_coherence'],
               'changed_bounds': ['aggregate.steer', 'aggregate.longitudinal', 'episode.*.steer',
                                  'episode.*.longitudinal']}
    if not args.check_only:
        require(not any(path.exists() for path in (PROTOCOL, CONFIG, RECEIPT)), 'refusing to overwrite frozen output')
        for path, contents in ((PROTOCOL, protocol_bytes), (CONFIG, config_bytes),
                               (RECEIPT, (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode())):
            with path.open('xb') as output:
                output.write(contents)
    print(json.dumps({'check_only': args.check_only, 'protocol_sha256': receipt['protocol_sha256'],
                      'training_config_sha256': receipt['training_config_sha256'],
                      'calibration_sha256': p['calibration_sha256'],
                      'aggregate_bounds': p['bounds']['aggregate']}, indent=2))


if __name__ == '__main__':
    main()
