"""Verify the completed, rejected recovery trial without continuing training."""
from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
import trial_v2 as trial

OUT = Path('runs/direct_command_recovery_trial_v2')
RECEIPT = Path('runs/direct_command_recovery_trial_v2.postflight.json')
MEASURE = Path('runs/direct_command_recovery_trial_v2.measure.json')
STEPS = [0, 100, 200, 300, 400, 500]
TRIAL_SHA = '4ccc4bd5465b8f412182fb96e39357d91141ce110f7e0f545181011856c7031f'
RECIPE_SHA = '958ef9d41c777f9ad8b5b4f993413886bdab3c297f13182a15357d09679bf34f'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return trial.sha256(path)


def check_metrics(metrics):
    require(set(metrics) == {'groups', 'passed', 'selection_score'}, 'unexpected validation shape')
    groups = metrics['groups']
    require(isinstance(groups, dict) and 'all' in groups and groups['all']['count'] == 2067,
            'invalid validation groups')
    for name, group in groups.items():
        require(set(group) == {'count', 'mae', 'p95', 'passed'}, f'invalid group shape: {name}')
        require(type(group['count']) is int and group['count'] > 0, f'invalid count: {name}')
        mae, p95 = group['mae'], group['p95']
        require(len(mae) == len(p95) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x) and x >= 0 for x in mae + p95), f'invalid errors: {name}')
        expected = all(x <= bound for x, bound in zip(mae + p95, [0.01, 0.25, 0.03, 0.75]))
        require(type(group['passed']) is bool and group['passed'] == expected, f'wrong group gate: {name}')
    require(type(metrics['passed']) is bool and metrics['passed'] == all(g['passed'] for g in groups.values()),
            'wrong validation gate')
    require(metrics['selection_score'] == max(groups['all']['mae'][0] / .01, groups['all']['mae'][1] / .25),
            'wrong selection score')
    require(not metrics['passed'], 'a validation passed; rejected-only postflight is inapplicable')


def check_archive(audit):
    root = Path.cwd().resolve()
    expected = {}
    for name, digest in audit['sources'].items():
        path = Path(name)
        if path.suffix == '.onnx':
            continue
        resolved = path.resolve()
        member = str(resolved.relative_to(root)) if resolved.is_relative_to(root) else 'external/' + str(resolved).lstrip('/')
        require(member not in expected, 'duplicate archive member')
        expected[member] = digest
    with zipfile.ZipFile(OUT / 'source.zip') as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(expected), 'source archive inventory differs')
        for name, digest in expected.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, f'archive member changed: {name}')


def main():
    require(not RECEIPT.exists(), 'postflight receipt already exists')
    require(OUT.is_dir() and not (OUT / 'train.json').exists(), 'unexpected training state')
    require(not (OUT / 'visual.json').exists(), 'unexpected visual gate')
    require(sha(trial.__file__) == TRIAL_SHA and sha(trial.RECIPE) == RECIPE_SHA, 'frozen trial or recipe changed')
    recipe = read_json(trial.RECIPE)
    require(recipe['seed'] == 0 and recipe['updates'] == 500 and recipe['batch_size'] == 128
            and recipe['evaluation_steps'] == STEPS and recipe['validation_mae'] == [0.01, 0.25]
            and recipe['validation_p95'] == [0.03, 0.75], 'unexpected frozen recipe')
    measure = read_json(MEASURE)
    cmd = measure['command']
    require(measure['returncode'] == 1 and isinstance(cmd, list)
            and any(str(x).endswith('.work/lab/direct_command_recovery_trial/trial_v2.py') for x in cmd)
            and 'train' in cmd and '--out' in cmd and cmd[cmd.index('--out') + 1] == str(OUT)
            and '--device' in cmd and cmd[cmd.index('--device') + 1] == 'cuda'
            and measure['scope'] == 'Whole GPU and host; host RAM is unified system memory, not dedicated VRAM',
            'measurement command or exit differs')
    failure = read_json(OUT / 'train_failure.json')
    require(failure['stage'] == 'train' and 'Too many open files' in failure['error']
            and 'datasets(recipe)' in failure['traceback'] and 'OSError: [Errno 24]' in failure['traceback'],
            'unexpected original training failure')
    files = [OUT / 'audit.json', OUT / 'source.zip', OUT / 'overfit.json', OUT / 'train_failure.json',
             OUT / 'steps.jsonl', MEASURE, Path('scripts/measure_native.py'), trial.RECIPE, Path(trial.__file__), Path(__file__)]
    files += [OUT / f'{kind}_{step:03d}.{ext}' for step in STEPS
              for kind, ext in [('step', 'pt'), ('validation', 'json')]]
    before = {str(p): sha(p) for p in files}
    require({p.name for p in OUT.glob('step_*.pt')} == {f'step_{s:03d}.pt' for s in STEPS}
            and {p.name for p in OUT.glob('validation_*.json')} == {f'validation_{s:03d}.json' for s in STEPS},
            'checkpoint or validation inventory differs')
    audit, model, train, val, sampler = trial.preflight(OUT, recipe, trial.DEVICE)
    require(len(train) == 7430 and len(val) == 2067 and audit['base_tensors'] == 417
            and audit['base_parameters'] == 31154802 and len(audit['base_state']) == 417,
            'audit inventory differs')
    require(audit['recipe_sha256'] == RECIPE_SHA and audit['source_archive_sha256'] == sha(OUT / 'source.zip'),
            'audit hashes differ')
    check_archive(audit)
    overfit = read_json(OUT / 'overfit.json')
    require(overfit['passed'] is True and overfit['audit_sha256'] == sha(OUT / 'audit.json'), 'overfit gate differs')
    rows = [json.loads(line) for line in (OUT / 'steps.jsonl').read_text().splitlines()]
    require(len(rows) == 500 and [row['step'] for row in rows] == list(range(1, 501)), 'step log is incomplete')
    for row in rows:
        require(all(isinstance(row[k], (int, float)) and not isinstance(row[k], bool)
                    and math.isfinite(row[k]) and row[k] >= 0 for k in ('loss', 'grad_norm', 'seconds')),
                'invalid step log values')
    rng = np.random.default_rng(0)
    ordinary_indices = set(map(int, sampler.other))
    for _ in range(500):
        ids = sampler.draw(rng)
        old = sum(int(i) in sampler.old_indices for i in ids)
        new = sum(int(i) in sampler.new_indices for i in ids)
        ordinary = sum(int(i) in ordinary_indices for i in ids)
        require((len(ids), old, new, ordinary) == (128, 16, 16, 96), 'reconstructed draw differs')
    counts = sampler.draw_counts()
    require([counts[k] for k in ('total_draws', 'old_recovery_draws', 'new_recovery_draws', 'ordinary_draws')]
            == [64000, 8000, 8000, 48000], 'reconstructed draw totals differ')
    base_payload = torch.load(recipe['base_checkpoint'], map_location='cpu', weights_only=False)
    require(sha(recipe['base_checkpoint']) == recipe['base_checkpoint_sha256'], 'base checkpoint hash differs')
    base_state = trial.digest_state(base_payload['model'])
    require(len(base_state) == 417 and base_state == audit['base_state'], 'original base tensors differ from audit')
    head = trial.base.CommandHead().to(trial.DEVICE)
    named = {name: list(param.shape) for name, param in head.named_parameters()}
    require(sum(p.numel() for p in head.parameters()) == 33538, 'head parameter count differs')
    reports = []
    for step in STEPS:
        path = OUT / f'step_{step:03d}.pt'
        package = torch.load(path, map_location='cpu', weights_only=False)
        require(set(package) == {'kind', 'base_payload', 'head_state', 'command_contract', 'recipe', 'provenance'}
                and package['kind'] == trial.KIND and package['command_contract'] == trial.CONTRACT
                and package['recipe'] == recipe, f'package contract differs at {step}')
        require(set(package['base_payload']) == set(base_payload) and
                {k: v for k, v in package['base_payload'].items() if k != 'model'} ==
                {k: v for k, v in base_payload.items() if k != 'model'},
                f'package base metadata differs at {step}')
        require(trial.digest_state(package['base_payload']['model']) == base_state,
                f'package base tensors differ at {step}')
        provenance = package['provenance']
        require(provenance == {'step': step, 'audit_sha256': sha(OUT / 'audit.json'),
                'source_archive_sha256': audit['source_archive_sha256'],
                'base_model_digest': audit['base_model_digest'],
                'head_state_digest': trial.digest_state(package['head_state']),
                'optimizer_parameters': named}, f'package provenance differs at {step}')
        require(set(package['head_state']) == set(named), f'head state keys differ at {step}')
        head.load_state_dict(package['head_state'], strict=True)
        report = read_json(OUT / f'validation_{step:03d}.json')
        require(report['step'] == step and Path(report['package']).resolve() == path.resolve()
                and report['package_sha256'] == sha(path), f'validation binding differs at {step}')
        metrics = {k: v for k, v in report.items() if k not in ('step', 'package', 'package_sha256')}
        check_metrics(metrics)
        reports.append(metrics)
    head.eval()
    recomputed = trial.validate(model, head, val, trial.DEVICE)
    check_metrics(recomputed)
    require(recomputed == reports[-1], 'step 500 recomputed metrics differ from stored metrics')
    trial.frozen(model, audit['base_state'], audit['base_flags'])
    require(trial.model_digest(model) == audit['base_model_digest'], 'base model digest differs after validation')
    require(trial.inventory(recipe) == audit['sources'] and all(sha(p) == before[str(p)] for p in files),
            'source or evidence changed during postflight')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    fd_path = Path('/proc/self/fd')
    result = {'passed': True, 'meaning': 'artifact and data checks only', 'candidate_qualified': False,
              'optimization_resumed': False, 'selected_checkpoint': None, 'diagnostic_only_step': 500,
              'original_training_returncode': measure['returncode'], 'original_measurement_command': cmd,
              'original_measurement_scope': measure['scope'], 'original_failure': failure,
              'evidence_sha256': before, 'base_tensor_count': 417, 'base_tensors_match_all_six': True,
              'optimizer_parameter_shapes': named, 'reconstructed_draw_counts': counts,
              'validation_steps': STEPS, 'all_recorded_validations_rejected': True,
              'independently_recomputed_validation_steps': [500], 'step_500_metrics': recomputed,
              'metric_comparison': 'exact equality', 'open_file_descriptors': len(list(fd_path.iterdir())) if fd_path.exists() else None,
              'open_file_limit_soft': soft, 'open_file_limit_hard': hard}
    with RECEIPT.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trial.DEVICE = torch.device('cuda')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    main()
