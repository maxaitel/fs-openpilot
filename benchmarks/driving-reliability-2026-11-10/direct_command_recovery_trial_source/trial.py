"""Direct-command recovery data trial. Run as an isolated process from the repo root."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'direct_command_trial'
RECOVERY = HERE.parent / 'direct_command_recovery'
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(RECOVERY))
import actuator_trial as base
import collect
import verify
from fsop.data.native_replay import NativeReplay, load_episodes, episode_paths
from scripts.verify_lab_dataset import verify as verify_original_dataset

RECIPE = HERE / 'recipe.json'
DEVICE = None


def membership(recipe):
    original = json.loads(Path(recipe['original_membership']).read_text())
    expanded = json.loads(Path(recipe['new_membership']).read_text())
    extras = [json.loads((RECOVERY / 'cases' / (recipe['new_cases'][d]['id'] + '.json')).read_text())
              for d in ('skidpad', 'trackdrive')]
    cases = original['cases'] + extras
    if (original['train_cases'] != 32 or expanded != {**original, 'cases': cases, 'train_cases': 34}
            or len({c['id'] for c in cases}) != len(cases)):
        raise ValueError('expanded membership must add exactly two unique training cases')
    for case in extras:
        old_id = case['id'].removesuffix('-direct-recovery-v1')
        matches = [c for c in original['cases'] if c['id'] == old_id]
        if len(matches) != 1 or case != {**matches[0], 'id': case['id']} or case['split'] != 'train':
            raise ValueError('new case differs from its matched training case')
    return {'original_sha256': base.sha256(recipe['original_membership']),
            'expanded_sha256': base.sha256(recipe['new_membership']), 'train_cases': 34}


def verified_receipt(episode, receipt_path, checkpoint, device):
    episode, receipt_path = Path(episode).resolve(), Path(receipt_path).resolve()
    if receipt_path.is_relative_to(episode):
        raise ValueError('receipt must be outside episode')
    stored = json.loads(receipt_path.read_text())
    if stored.get('passed') is not True or stored.get('reasons') != []:
        raise ValueError('new recovery receipt did not pass')
    current = verify.verify(episode, Path(checkpoint), str(device))
    if current.get('passed') is not True or stored != current:
        raise ValueError('new recovery receipt does not match independent current replay')
    return {'path': str(receipt_path), 'sha256': base.sha256(receipt_path), 'report': stored}


class TrialSampler:
    def __init__(self, dataset, legacy, new_groups):
        self.dataset, self.legacy, self.new_groups = dataset, legacy, new_groups
        self.old_indices = set(legacy.recovery_indices)
        self.new_indices = {i for g in new_groups for i in g['indices']}
        if (len(new_groups) != 2 or any(len(g['indices']) < 8 for g in new_groups)
                or self.old_indices & self.new_indices):
            raise ValueError('invalid recovery groups')
        self.other = np.array([i for i in range(len(dataset))
                               if i not in self.old_indices and i not in self.new_indices], dtype=np.int64)
        if not len(self.other):
            raise ValueError('ordinary pool is empty')
        self.counts = np.zeros(len(dataset), dtype=np.int64)

    def draw(self, rng):
        def group_draw(groups, n):
            choices = rng.integers(len(groups), size=n)
            return [groups[int(g)]['indices'][int(rng.integers(len(groups[int(g)]['indices'])))] for g in choices]
        result = np.concatenate((np.asarray(group_draw(self.legacy.groups, 16), dtype=np.int64),
                                 np.asarray(group_draw(self.new_groups, 16), dtype=np.int64),
                                 self.other[rng.integers(len(self.other), size=96)]))
        rng.shuffle(result)
        np.add.at(self.counts, result, 1)
        return result

    def protocol(self):
        return {'legacy_gate': self.legacy.protocol(), 'new_groups': self.new_groups,
                'old_recovery_draws': 16, 'new_recovery_draws': 16, 'ordinary_draws': 96,
                'new_case_probability': 0.5, 'with_replacement': True,
                'ordinary_indices': self.other.tolist()}

    def draw_counts(self):
        return {'total_draws': int(self.counts.sum()),
                'old_recovery_draws': int(sum(self.counts[i] for i in self.old_indices)),
                'new_recovery_draws': int(sum(self.counts[i] for i in self.new_indices)),
                'ordinary_draws': int(sum(self.counts[i] for i in self.other)),
                'episodes': [{'case_id': ep.meta['id'], 'anchors': [
                    {'frame': frame, 'draws': int(self.counts[i])}
                    for i, (episode, frame) in enumerate(self.dataset.index) if episode == ei]}
                    for ei, ep in enumerate(self.dataset.episodes) if ep.meta['split'] == 'train']}


def inventory(recipe):
    paths = [RECIPE, Path(__file__), HERE / 'test_trial.py', HERE / 'protocol.md', OLD / 'actuator_trial.py', OLD / 'recipe.json',
             OLD / 'evaluate_v2.py', Path('scripts/verify_lab_dataset.py'), Path(recipe['recovery_recipe']),
             Path('artifacts/models/prep/vision_fp32.onnx'), Path('artifacts/models/prep/policy_fp32.onnx')]
    sources = {**base.source_inventory(), **collect.source_inventory(), **{str(p): base.sha256(p) for p in paths}}
    root = Path.cwd().resolve()
    result = {}
    for name, digest in sources.items():
        path = Path(name).resolve()
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        if key in result and result[key] != digest:
            raise ValueError('source inventory disagrees')
        result[key] = digest
    return result


def datasets(recipe):
    original_recipe = json.loads((OLD / 'recipe.json').read_text())
    for key, value in original_recipe.items():
        if recipe.get(key) != value:
            raise ValueError(f'original recipe changed: {key}')
    m = membership(recipe)
    if base.sha256(recipe['collection_checkpoint']) != recipe['collection_checkpoint_sha256'] or recipe['collection_checkpoint_sha256'] != verify.CHECKPOINT_SHA:
        raise ValueError('collection checkpoint hash differs')
    independent_old = verify_original_dataset(recipe['manifest'], recipe['original_membership'])
    old_train, old_val, legacy, old_verification = original_datasets(recipe)
    stored_old = json.loads(Path(old_verification['path']).read_text())
    if independent_old != stored_old:
        raise ValueError('original manifest verification differs from current independent audit')
    paths = episode_paths(manifest=recipe['manifest'])
    receipts = {}
    for d in ('skidpad', 'trackdrive'):
        item = recipe['new_cases'][d]
        receipts[d] = verified_receipt(item['episode'], item['receipt'], recipe['collection_checkpoint'], DEVICE)
        paths.append(Path(item['episode']))
    episodes = load_episodes(paths)
    train, val = NativeReplay(episodes, 'train'), NativeReplay(episodes, 'val')
    if train.index[:len(old_train)] != old_train.index or val.index != old_val.index or len(old_train) != 7006 or len(val) != 2067:
        raise ValueError('original train or validation anchors changed')
    old_ids = [ep.meta['id'] for ep in old_train.episodes if ep.meta['split'] == 'train']
    expected_ids = [c['id'] for c in json.loads(Path(recipe['original_membership']).read_text())['cases'] if c['split'] == 'train']
    if len(old_ids) != 32 or set(old_ids) != set(expected_ids):
        raise ValueError('original training membership differs')
    groups = []
    for d in ('skidpad', 'trackdrive'):
        item, report = recipe['new_cases'][d], receipts[d]['report']
        matches = [ei for ei, ep in enumerate(episodes) if ep.meta.get('id') == item['id']]
        if len(matches) != 1:
            raise ValueError('new case identity missing or duplicated')
        ei = matches[0]
        ep = episodes[ei]
        indices = [i for i, (e, f) in enumerate(train.index) if e == ei and int(ep.ticks[f]) in report['qualified_anchors_first_2s']]
        ticks = [int(ep.ticks[train.index[i][1]]) for i in indices]
        if ticks != report['qualified_anchors_first_2s'] or ep.meta['split'] != 'train':
            raise ValueError('recovery anchor receipt differs')
        groups.append({'case_id': item['id'], 'episode': ei, 'indices': indices, 'ticks': ticks})
    sampler = TrialSampler(train, legacy, groups)
    added = len(train) - len(old_train)
    verification = {'original': old_verification, 'original_independent': independent_old, 'membership': m, 'new_receipts': receipts,
                    'original_train_count': len(old_train), 'added_train_count': added,
                    'train_count': len(train), 'val_count': len(val)}
    return train, val, sampler, verification


def wiring_indices(sampler):
    rng = np.random.default_rng(0)
    old_other = [i for i in sampler.other if i < len(sampler.legacy.dataset)]
    if len(old_other) < 16:
        raise ValueError('fewer than 16 old ordinary anchors')
    return (rng.choice(old_other, size=16, replace=False).tolist()
            + [i for group in sampler.new_groups for i in rng.choice(group['indices'], size=8, replace=False).tolist()])


original_datasets = base.datasets
original_preflight = base.preflight
base.RECIPE = RECIPE
base.datasets = datasets
base.inventory = inventory


# The frozen trial owns training and validation; only dataset and draw contracts differ.
from actuator_trial import (load_fs, check_base, prove_policy_context, batch_input, digest_state,
                            frozen, model_digest, write_json, sha256, new_head, validate, KIND,
                            CONTRACT, loss, telemetry)
import shutil
import time
import zipfile
from torch import nn
def audit(out, recipe, device):
    train, val, sampler, verification = datasets(recipe)
    model = load_fs(recipe['base_checkpoint']).requires_grad_(False).eval().to(device)
    check_base(model)
    context = prove_policy_context(model)
    ids = wiring_indices(sampler)
    dynamic = [i for i, (ei, frame) in enumerate(train.index) if frame >= 8 and
               train.episodes[ei].states[int(train.episodes[ei].ticks[frame]), 3] >= 2.0]
    if len(dynamic) < 32:
        raise ValueError('fewer than 32 dynamic visual-dependence anchors')
    visual_ids = np.random.default_rng(0).choice(dynamic, size=32, replace=False).tolist()
    a = batch_input(model, train, ids[:4], device, hook=False)[3]
    latent_b, _, _, b = batch_input(model, train, ids[:4], device, hook=True)
    c = batch_input(model, train, ids[:4], device, hook=False)[3]
    latent_d, _, _, d = batch_input(model, train, ids[:4], device, hook=True)
    if not torch.equal(latent_b, latent_d) or not all(torch.equal(x, y) for output in (b, c, d) for x, y in zip(a, output, strict=True)):
        raise ValueError('ABAB plain/hooked policy outputs differ')
    refs = digest_state(model.state_dict())
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    frozen(model, refs, flags)
    sources = inventory(recipe)
    with zipfile.ZipFile(out / 'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            path = Path(name)
            if path.suffix != '.onnx':
                resolved, root = path.resolve(), Path.cwd().resolve()
                member = (str(resolved.relative_to(root)) if resolved.is_relative_to(root)
                          else 'external/' + str(resolved).lstrip('/'))
                archive.write(path, arcname=member)
    result = {'recipe': recipe, 'recipe_sha256': sha256(RECIPE), 'sources': sources,
              'source_archive_sha256': sha256(out / 'source.zip'), 'verification': verification,
              'train_count': len(train), 'val_count': len(val), 'sampler': sampler.protocol(),
              'overfit_indices': ids, 'visual_indices': visual_ids,
              'visual_anchors': [{'case': train.episodes[train.index[i][0]].meta['id'],
              'frame': train.index[i][1], 'tick': int(train.episodes[train.index[i][0]].ticks[train.index[i][1]])} for i in visual_ids],
              'overfit_anchors': [{'case': train.episodes[train.index[i][0]].meta['id'],
              'frame': train.index[i][1], 'tick': int(train.episodes[train.index[i][0]].ticks[train.index[i][1]])} for i in ids],
              'base_state': refs, 'base_flags': flags, 'base_model_digest': model_digest(model),
              'base_parameters': 31154802, 'base_tensors': len(refs), 'context': context, 'abab_exact': True}
    write_json(out / 'audit.json', result)


def train_run(out, recipe, device):
    audit, model, train, val, sampler = preflight(out, recipe, device)
    gate = json.loads((out / 'overfit.json').read_text())
    if not gate['passed'] or gate['audit_sha256'] != sha256(out / 'audit.json'):
        raise ValueError('matching successful overfit gate required')
    head, optimizer = new_head(device)
    named = {n: list(p.shape) for n, p in head.named_parameters()}
    optimized = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(optimized) != {id(p) for p in head.parameters()} or len(optimized) != len(named):
        raise ValueError('optimizer inventory mismatch')
    if shutil.disk_usage(out).free < recipe['min_free_bytes']:
        raise ValueError('free disk below reserve')
    payload = torch.load(recipe['base_checkpoint'], map_location='cpu', weights_only=False)
    rng = np.random.default_rng(0)
    reports = []
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(501):
        if step in recipe['evaluation_steps']:
            frozen(model, audit['base_state'], audit['base_flags'])
            metrics = validate(model, head, val, device)
            provenance = {'step': step, 'audit_sha256': sha256(out / 'audit.json'),
                          'source_archive_sha256': audit['source_archive_sha256'],
                          'base_model_digest': model_digest(model), 'head_state_digest': digest_state(head.state_dict()),
                          'optimizer_parameters': named}
            package = {'kind': KIND, 'base_payload': payload, 'head_state': head.state_dict(),
                       'command_contract': CONTRACT, 'recipe': recipe, 'provenance': provenance}
            if shutil.disk_usage(out).free < recipe['min_free_bytes']:
                raise ValueError('free disk below reserve before checkpoint')
            path = out / f'step_{step:03d}.pt'
            if path.exists():
                raise FileExistsError(path)
            torch.save(package, path)
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > recipe['artifact_budget_bytes']:
                raise ValueError('artifact budget exceeded')
            reports.append({'step': step, 'package': str(path), 'package_sha256': sha256(path), **metrics})
            write_json(out / f'validation_{step:03d}.json', reports[-1])
        if step == 500:
            break
        t0 = time.monotonic()
        ids = sampler.draw(rng)
        if (sum(int(i) in sampler.old_indices for i in ids) != 16 or
                sum(int(i) in sampler.new_indices for i in ids) != 16):
            raise ValueError('sampler violated the recovery draw contract')
        latent, prop, target, _ = batch_input(model, train, ids, device)
        optimizer.zero_grad(set_to_none=True)
        value = loss(head, latent, prop, target)
        if not torch.isfinite(value):
            raise ValueError('nonfinite loss')
        value.backward()
        norm = nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        with (out / 'steps.jsonl').open('a') as f:
            f.write(json.dumps({'step': step + 1, 'loss': float(value.detach()), 'grad_norm': float(norm),
                                'seconds': time.monotonic() - t0, 'gpu': telemetry()}, allow_nan=False) + '\n')
    frozen(model, audit['base_state'], audit['base_flags'])
    eligible = [row for row in reports if row['step'] > 0 and row['passed']]
    selected = min(eligible, key=lambda row: (row['selection_score'], row['step'])) if eligible else None
    _, _, sampler_after, verification_after = datasets(recipe)
    if (verification_after != audit['verification'] or sampler_after.protocol() != audit['sampler']
            or inventory(recipe) != audit['sources'] or sha256(out / 'source.zip') != audit['source_archive_sha256']):
        raise ValueError('source or data changed during training')
    write_json(out / 'train.json', {'passed': bool(selected), 'selected': selected, 'reports': reports,
               'diagnostic_only_step': None if selected else 500, 'seconds': time.monotonic() - started,
               'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               'optimizer_parameters': named, 'base_model_digest_after': model_digest(model),
               'base_state_unchanged': True, 'base_flags_unchanged': True, 'draw_counts': sampler.draw_counts()})



def preflight(out, recipe, device):
    result = original_preflight(out, recipe, device)
    if base.sha256(out / 'source.zip') != result[0]['source_archive_sha256']:
        raise ValueError('source archive changed after audit')
    return result


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'overfit', 'train', 'visual'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.stage == 'audit':
        args.out.mkdir(parents=True, exist_ok=False)
    elif not args.out.is_dir():
        raise ValueError('run audit first')
    receipt = args.out / f'{args.stage}_failure.json'
    if receipt.exists() or (args.out / f'{args.stage}.json').exists():
        raise FileExistsError('stage already attempted; use a new output directory')
    try:
        if args.stage == 'train' and (args.out / 'steps.jsonl').exists():
            raise FileExistsError('training already attempted')
        recipe = json.loads(RECIPE.read_text())
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        DEVICE = torch.device(args.device)
        if DEVICE.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        base.preflight = preflight
        {'audit': audit, 'overfit': base.overfit, 'train': train_run, 'visual': base.visual_gate}[args.stage](args.out, recipe, DEVICE)
    except BaseException as exc:
        base.write_json(receipt, {'stage': args.stage, 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
