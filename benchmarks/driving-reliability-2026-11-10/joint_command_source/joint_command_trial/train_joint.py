"""Frozen-recipe joint suffix trial with disposable conditioning and wiring stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fsop.train.lab_train import prove_policy_context, sha256
from isaac_lab_fsop.evaluate import source_inventory
from joint_core import (KIND, CONTRACT, SCOPE, CommandHead, JointActuatorPolicy, check_base_inventory,
                        check_frozen_complement, digest_state, make_optimizer, make_package,
                        model_digest, model_from_payload, set_training_scope, validate_package)
from joint_data import DATA_TRIAL, HERE, OLD, RECOVERY, JointData, batch_input, read_json, wiring_indices

RECIPE = HERE / 'recipe.json'
DESIGN = HERE.parent / 'next_strategy'
JOINT_FILES = ('joint_core.py', 'joint_data.py', 'train_joint.py', 'recipe.json', 'protocol.md', 'test_joint.py',
               'evaluate.py', 'verify_command_episode.py', 'process.py', 'export_audited_command.py',
               'encode_portable.py', 'pipeline_notes.md', 'test_pipeline.py')


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_recipe(recipe):
    previous = read_json(DATA_TRIAL / 'recipe_v2.json')
    allowed = {'kind', 'trial_version'}
    if any(recipe.get(k) != value for k, value in previous.items() if k not in allowed):
        raise ValueError('recipe changes more than architecture and trial identity')
    if (recipe['kind'] != KIND or recipe['training_scope'] != SCOPE or
            recipe['source_recipe_sha256'] != sha256(DATA_TRIAL / 'recipe_v2.json') or
            recipe['deployment'] != CONTRACT['deployment'] or recipe['conditioning_updates'] != 1 or
            recipe['precision'] != {'amp': False, 'tf32': False, 'channels_last': False, 'feature_cache': False}):
        raise ValueError('joint recipe contract differs')
    if recipe['condition_indices'] != 'same fixed 32 TRAIN anchors as disposable wiring':
        raise ValueError('conditioning anchor rule differs')


def inventory(recipe):
    paths = [HERE / name for name in JOINT_FILES]
    paths += [DATA_TRIAL / name for name in ('trial_v2.py', 'recipe_v2.json', 'protocol_v2.md', 'test_trial_v2.py')]
    paths += [RECOVERY / name for name in ('collect.py', 'verify.py', 'verify_v2.py', 'protocol.md',
                                         'derived_verifier_v2.md', 'membership.json', 'test_bindings.py')]
    paths += [OLD / name for name in ('actuator_trial.py', 'recipe.json', 'evaluate_v2.py')]
    paths += [DESIGN / name for name in ('synthesis.md', 'policy_graph_metadata.json', 'candidate_b/design.md',
                                       'candidate_b/graph_inventory.json', 'candidate_b/parameter_inventory.json')]
    paths += list((RECOVERY / 'cases').glob('*.json'))
    paths += [Path(name) for name in ('scripts/verify_lab_dataset.py', 'scripts/verify_lab_recovery.py',
                                    'scripts/audit_isaac_episode.py', 'docs/isaac-lab-training/phase_recovery_recipe4.json')]
    paths += [Path(recipe[k]) for k in ('recovery_recipe', 'original_membership', 'new_membership')]
    paths += [Path('artifacts/models/prep') / name for name in ('vision_fp32.onnx', 'policy_fp32.onnx')]
    result = {}
    root = Path.cwd().resolve()
    for name in [*source_inventory(), *paths]:
        path = Path(name).resolve()
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        digest = sha256(path)
        if key in result and result[key] != digest:
            raise ValueError('source inventory disagrees')
        result[key] = digest
    return dict(sorted(result.items()))


def archive_sources(out, sources):
    mapping = {}
    with zipfile.ZipFile(out / 'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sources.items():
            path = Path(name)
            if path.suffix == '.onnx':
                continue
            member = 'external/' + str(path).lstrip('/') if path.is_absolute() else name
            if member in mapping.values():
                raise ValueError('source archive member collision')
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('source changed during archival')
            archive.writestr(member, content)
            mapping[name] = member
    return mapping


def check_archive(out, audit):
    if sha256(out / 'source.zip') != audit['source_archive_sha256']:
        raise ValueError('source archive changed')
    expected = {k for k in audit['sources'] if Path(k).suffix != '.onnx'}
    if set(audit['source_archive_members']) != expected:
        raise ValueError('archive source mapping differs')
    with zipfile.ZipFile(out / 'source.zip') as archive:
        mapping = audit['source_archive_members']
        if set(archive.namelist()) != set(mapping.values()) or len(archive.namelist()) != len(mapping):
            raise ValueError('archive member inventory differs')
        for source, member in mapping.items():
            if hashlib.sha256(archive.read(member)).hexdigest() != audit['sources'][source]:
                raise ValueError('archived source differs')


def check_graph(model):
    context = prove_policy_context(model)
    nodes = {node.target: node for node in model.base.policy.graph.nodes if node.op == 'call_module'}
    expected = {'p_node_linear_14': ['p_node_relu_3'], 'p_node_relu_4': ['p_node_linear_14'],
                'p_node_linear_16': ['p_node_relu_4'], 'p_node_relu_6': ['p_node_linear_16'],
                'p_node_linear_17': ['p_node_relu_6'], 'p_node_add_7': ['p_node_relu_4', 'p_node_linear_17'],
                'p_node_relu_7': ['p_node_add_7'], 'p_node_linear_20': ['p_node_relu_7']}
    for name, parents in expected.items():
        if name not in nodes or nodes[name].kwargs or [node.target for node in nodes[name].all_input_nodes] != parents:
            raise ValueError(f'joint policy ancestry differs: {name}')
    for name in ('p_node_linear_14', 'p_node_linear_16', 'p_node_linear_17', 'p_node_linear_20'):
        if not isinstance(model.base.policy.get_submodule(name), nn.Linear):
            raise ValueError(f'joint suffix is not an affine module: {name}')
    return {'context': context, 'suffix_parents': expected}


def original_model(recipe, device):
    payload = torch.load(recipe['base_checkpoint'], map_location='cpu', weights_only=False)
    model = model_from_payload(payload, device)
    set_training_scope(model)
    return payload, model


def new_fit(model, device):
    torch.manual_seed(0)
    head = CommandHead().to(device).eval()
    optimizer, named = make_optimizer(model, head)
    return head, optimizer, named


def command_loss(head, latent, prop, target):
    return F.smooth_l1_loss((head(latent, prop) - target) / target.new_tensor([.02, .5]), torch.zeros_like(target))


def scores(errors):
    values = np.asarray(errors)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('empty or nonfinite metric group')
    mae, p95 = values.mean(0), np.percentile(values, 95, axis=0)
    return {'count': len(values), 'mae': mae.tolist(), 'p95': p95.tolist(),
            'passed': bool(np.all(mae <= [.01, .25]) and np.all(p95 <= [.03, .75]))}


def validate(model, head, val, device):
    groups = {'all': []}
    with torch.no_grad():
        for start in range(0, len(val), 64):
            ids = range(start, min(start + 64, len(val)))
            latent, prop, target, _ = batch_input(model, val, ids, device)
            errors = (head(latent, prop) - target).abs().cpu().numpy()
            for offset, index in enumerate(ids):
                ei, frame = val.index[index]
                ep = val.episodes[ei]
                keys = ['all', 'episode/' + str(ep.meta['id'])]
                if prop[offset, 0].item() < 2:
                    keys.append('stratum/low_vx')
                if np.any(ep.desires[frame, -9:] != 0):
                    keys.append('stratum/recent_pulse')
                if ep.meta.get('collection_policy', {}).get('mode') in ('learner_prefix', 'phase_recovery_v1'):
                    keys.append('stratum/recovery')
                for key in keys:
                    groups.setdefault(key, []).append(errors[offset])
    result = {key: scores(values) for key, values in sorted(groups.items())}
    return {'groups': result, 'passed': all(row['passed'] for row in result.values()),
            'selection_score': max(result['all']['mae'][0] / .01, result['all']['mae'][1] / .25)}


def anchors(dataset, ids):
    return [{'case': dataset.episodes[dataset.index[i][0]].meta['id'], 'frame': dataset.index[i][1],
             'tick': int(dataset.episodes[dataset.index[i][0]].ticks[dataset.index[i][1]])} for i in ids]


def audit_run(out, recipe, device):
    sources = inventory(recipe)
    payload, model = original_model(recipe, device)
    graph = check_graph(model)
    head, _, named = new_fit(model, device)
    with JointData(recipe, str(device)) as data:
        ids = wiring_indices(data.sampler)
        dynamic = [i for i, (ei, frame) in enumerate(data.train.index) if frame >= 8 and
                   data.episodes[ei].states[int(data.episodes[ei].ticks[frame]), 3] >= 2.]
        if len(dynamic) < 32:
            raise ValueError('fewer than 32 dynamic visual anchors')
        visual_ids = np.random.default_rng(0).choice(dynamic, 32, replace=False).tolist()
        outputs, latents = [], []
        for hook in (False, True, False, True):
            latent, _, _, result = batch_input(model, data.train, ids[:4], device, hook=hook)
            outputs.append(result)
            if hook:
                latents.append(latent.detach().clone())
        if (not torch.equal(*latents) or not all(torch.equal(x, y) for result in outputs[1:]
                                               for x, y in zip(outputs[0], result, strict=True))):
            raise ValueError('ABAB plain/hooked outputs differ')
        with torch.no_grad():
            ordinary = batch_input(model, data.train, ids[:4], device)[0]
        with torch.inference_mode():
            sys.path.insert(0, str(OLD))
            from actuator_trial import batch_input as old_batch_input
            old_latent = old_batch_input(model, data.train, ids[:4], device)[0]
        if not torch.equal(ordinary, old_latent):
            raise ValueError('joint initialization differs from old inference forward')
        refs = digest_state(payload['model'])
        if digest_state(model.state_dict()) != refs:
            raise ValueError('audit mutated original base')
        mapping = archive_sources(out, sources)
        record = {'recipe': recipe, 'recipe_sha256': sha256(RECIPE), 'sources': sources,
                  'source_archive_sha256': sha256(out / 'source.zip'), 'source_archive_members': mapping,
                  'verification': data.verification, 'sampler': data.sampler.protocol(),
                  'train_count': len(data.train), 'val_count': len(data.val),
                  'conditioning_indices': ids, 'overfit_indices': ids, 'visual_indices': visual_ids,
                  'overfit_anchors': anchors(data.train, ids), 'visual_anchors': anchors(data.train, visual_ids),
                  'base_state': refs, 'base_model_digest': model_digest(model),
                  'head_initial_state': digest_state(head.state_dict()), 'graph': graph,
                  'optimizer_parameters': {name: list(value.shape) for name, value in named.items()},
                  'trainable_parameters': 296450, 'base_tensors': 417, 'frozen_base_tensors': 411,
                  'abab_exact': True, 'old_inference_forward_exact': True,
                  'device': str(device), 'fresh_history_context': 8, 'feature_cache': False}
        if inventory(recipe) != sources:
            raise ValueError('sources changed during audit')
        write_json(out / 'audit.json', record)


def reject_failed_attempt(out):
    failures = sorted(Path(out).glob('*_failure.json'))
    if failures:
        raise ValueError('failed attempt cannot resume: ' + ', '.join(path.name for path in failures))


def reject_interrupted_attempt(out, stage):
    for attempt in Path(out).glob('*_attempt.json'):
        name = attempt.name.removesuffix('_attempt.json')
        if name == 'train' and stage == 'postflight':
            continue
        if not (Path(out) / (name + '.json')).exists():
            raise FileExistsError(f'interrupted {name} attempt cannot resume')


def require_training_parent(result):
    parent = result['process_id']
    if type(parent) is not int or parent <= 1 or parent == os.getpid() or os.getppid() != parent:
        raise ValueError('postflight requires its live training parent')
    try:
        os.kill(parent, 0)
    except ProcessLookupError as exc:
        raise ValueError('postflight requires its live training parent') from exc


def checked_audit(out, recipe):
    reject_failed_attempt(out)
    audit = read_json(out / 'audit.json')
    if audit['recipe'] != recipe or audit['recipe_sha256'] != sha256(RECIPE) or audit['sources'] != inventory(recipe):
        raise ValueError('recipe or source changed after audit')
    check_archive(out, audit)
    return audit


def check_data(data, audit):
    if data.verification != audit['verification'] or data.sampler.protocol() != audit['sampler']:
        raise ValueError('dataset verification or sampler changed after audit')


def gate(out, name):
    value = read_json(out / (name + '.json'))
    if value.get('passed') is not True or value['audit_sha256'] != sha256(out / 'audit.json'):
        raise ValueError(f'matching successful {name} gate required')
    return value


def provenance(out, audit):
    return {'audit_sha256': sha256(out / 'audit.json'), 'source_archive_sha256': audit['source_archive_sha256'],
            'original_base_checkpoint_sha256': audit['recipe']['base_checkpoint_sha256'],
            'optimizer_parameters': audit['optimizer_parameters']}


def space(out, recipe):
    if shutil.disk_usage(out).free < recipe['min_free_bytes']:
        raise ValueError('free disk below 25 GiB reserve')
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > recipe['artifact_budget_bytes']:
        raise ValueError('artifact budget exceeds 1 GiB')


def save_package(path, package):
    with Path(path).open('xb') as stream:
        torch.save(package, stream)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    validate_package(saved)
    if saved['provenance'] != package['provenance']:
        raise ValueError('serialized package provenance differs')
    return saved


def update(model, head, optimizer, named, dataset, ids, device):
    latent, prop, target, _ = batch_input(model, dataset, ids, device)
    optimizer.zero_grad(set_to_none=True)
    value = command_loss(head, latent, prop, target)
    if not torch.isfinite(value):
        raise ValueError('nonfinite loss')
    value.backward()
    gradients = {}
    for name, parameter in named.items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f'missing or nonfinite gradient: {name}')
        gradients[name] = float(parameter.grad.norm())
    norm = float(nn.utils.clip_grad_norm_(list(named.values()), 1.))
    if not np.isfinite(norm):
        raise ValueError('nonfinite global gradient norm')
    optimizer.step()
    return {'loss': float(value.detach()), 'gradient_norms': gradients, 'global_gradient_norm': norm,
            'global_clip_scale': min(1., 1. / (norm + 1e-6))}


def condition_run(out, recipe, device):
    audit = checked_audit(out, recipe)
    original, model = original_model(recipe, device)
    head, optimizer, named = new_fit(model, device)
    before = {name: parameter.detach().clone() for name, parameter in named.items()}
    with JointData(recipe, str(device)) as data:
        check_data(data, audit)
        ids = audit['conditioning_indices']
        def measured():
            with torch.no_grad():
                latent, prop, target, _ = batch_input(model, data.train, ids, device)
                return float(command_loss(head, latent, prop, target)), head(latent, prop).detach().clone()
        first, _ = measured()
        repeat, _ = measured()
        result = update(model, head, optimizer, named, data.train, ids, device)
        after, commands = measured()
        if any(value <= 0 for value in result['gradient_norms'].values()):
            raise ValueError('conditioning requires nonzero gradients in every optimized tensor')
        changed = check_frozen_complement(model, audit['base_state'], training=True)
        update_norms = {name: float((value.detach() - before[name]).norm()) for name, value in named.items()}
        package = make_package(original, model, head, recipe, provenance(out, audit), 1)
        space(out, recipe)
        with tempfile.TemporaryDirectory(prefix='joint-conditioning-') as directory:
            path = Path(directory) / 'condition.pt'
            saved = save_package(path, package)
            restored = model_from_payload(saved['base_payload'], device)
            restored_head = CommandHead().to(device).eval()
            restored_head.load_state_dict(saved['head_state'], strict=True)
            with torch.no_grad():
                latent, prop, _, _ = batch_input(restored, data.train, ids, device)
                roundtrip = restored_head(latent, prop)
            exact = torch.equal(commands, roundtrip)
            roundtrip_sha = sha256(path)
        noise = abs(first - repeat)
        passed = (first - after > 5 * noise and first - after > 0 and exact and set(changed) == set(SCOPE)
                  and all(value > 0 for value in update_norms.values()))
        write_json(out / 'condition.json', {'passed': passed, 'audit_sha256': sha256(out / 'audit.json'),
                   'step': 1, 'indices': ids, 'normal_loss': first, 'repeat_loss': repeat, 'updated_loss': after,
                   'repeat_noise': noise, 'improvement': first - after, **result, 'update_norms': update_norms,
                   'changed_suffix_keys': changed, 'frozen_base_state_unchanged': True,
                   'frozen_base_tensors': 411, 'roundtrip_commands_exact': exact,
                   'roundtrip_package_sha256': roundtrip_sha, 'discarded_all_weights': True})
        if not passed:
            raise ValueError('fixed-rate conditioning failed; this recipe stops without a rate sweep')


def overfit_run(out, recipe, device):
    audit = checked_audit(out, recipe)
    gate(out, 'condition')
    _, model = original_model(recipe, device)
    head, optimizer, named = new_fit(model, device)
    started = time.monotonic()
    with JointData(recipe, str(device)) as data:
        check_data(data, audit)
        ids = audit['overfit_indices']
        for step in range(201):
            with torch.no_grad():
                latent, prop, target, _ = batch_input(model, data.train, ids, device)
                mae = (head(latent, prop) - target).abs().mean(0).cpu().tolist()
            if np.all(np.asarray(mae) <= [.002, .05]) or step == 200:
                break
            update(model, head, optimizer, named, data.train, ids, device)
        check_frozen_complement(model, audit['base_state'], training=True)
        passed = bool(np.all(np.asarray(mae) <= [.002, .05]))
        write_json(out / 'overfit.json', {'passed': passed, 'step': step, 'mae': mae,
                   'audit_sha256': sha256(out / 'audit.json'), 'condition_sha256': sha256(out / 'condition.json'),
                   'seconds': time.monotonic() - started, 'discarded_all_weights': True,
                   'frozen_base_state_unchanged': True, 'frozen_base_tensors': 411})
        if not passed:
            raise ValueError('fixed 32-anchor wiring gate failed; main run prohibited')


def train_run(out, recipe, device):
    audit = checked_audit(out, recipe)
    gate(out, 'condition')
    wiring = gate(out, 'overfit')
    if wiring['condition_sha256'] != sha256(out / 'condition.json'):
        raise ValueError('conditioning receipt changed after wiring')
    original, model = original_model(recipe, device)
    head, optimizer, named = new_fit(model, device)
    if digest_state(head.state_dict()) != audit['head_initial_state'] or model_digest(model) != audit['base_model_digest']:
        raise ValueError('main fit did not start at pristine base and seed-zero head')
    rng, reports = np.random.default_rng(0), []
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with JointData(recipe, str(device)) as data:
        check_data(data, audit)
        for step in range(501):
            if step in recipe['evaluation_steps']:
                check_frozen_complement(model, audit['base_state'], training=True)
                metrics = validate(model, head, data.val, device)
                package = make_package(original, model, head, recipe, provenance(out, audit), step)
                space(out, recipe)
                path = out / f'step_{step:03d}.pt'
                save_package(path, package)
                space(out, recipe)
                reports.append({'step': step, 'package': str(path), 'package_sha256': sha256(path), **metrics})
                write_json(out / f'validation_{step:03d}.json', reports[-1])
            if step == 500:
                break
            t0 = time.monotonic()
            result = update(model, head, optimizer, named, data.train, data.sampler.draw(rng), device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            with (out / 'steps.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': step + 1, **result, 'seconds': time.monotonic() - t0}, allow_nan=False) + '\n')
        changed = check_frozen_complement(model, audit['base_state'], training=True)
        record = {'updates_complete': True, 'postflight_complete': False, 'reports': reports,
                  'process_id': os.getpid(),
                  'audit_sha256': sha256(out / 'audit.json'), 'condition_sha256': sha256(out / 'condition.json'),
                  'overfit_sha256': sha256(out / 'overfit.json'), 'device': str(device),
                  'seconds': time.monotonic() - started, 'optimizer_parameters': audit['optimizer_parameters'],
                  'base_model_digest_after': model_digest(model), 'changed_suffix_keys': changed,
                  'frozen_base_state_unchanged': True, 'frozen_base_tensors': 411,
                  'draw_counts': data.sampler.draw_counts(),
                  'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    checked_audit(out, recipe)
    write_json(out / 'training_updates.json', record)
    del data, optimizer, named, model, head, original, package
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'postflight', '--out', str(out),
                    '--device', str(device)], check=True)


def postflight_run(out, recipe, device):
    reject_failed_attempt(out)
    result = read_json(out / 'training_updates.json')
    require_training_parent(result)
    audit = checked_audit(out, recipe)
    if (result['audit_sha256'] != sha256(out / 'audit.json') or result['device'] != str(device) or
            result['process_id'] == os.getpid()):
        raise ValueError('training completion receipt differs')
    gate(out, 'condition')
    gate(out, 'overfit')
    if (result['condition_sha256'] != sha256(out / 'condition.json') or
            result['overfit_sha256'] != sha256(out / 'overfit.json')):
        raise ValueError('training stage receipts changed')
    rows = [json.loads(line) for line in (out / 'steps.jsonl').read_text().splitlines()]
    if [row['step'] for row in rows] != list(range(1, 501)):
        raise ValueError('expected exactly 500 updates without a resume')
    if [report['step'] for report in result['reports']] != [0, 100, 200, 300, 400, 500]:
        raise ValueError('expected all six validations')
    packages = []
    for report in result['reports']:
        path = Path(report['package'])
        if sha256(path) != report['package_sha256'] or report != read_json(out / f"validation_{report['step']:03d}.json"):
            raise ValueError('checkpoint or validation receipt changed')
        package = torch.load(path, map_location='cpu', weights_only=False)
        validate_package(package)
        p = package['provenance']
        if (package['recipe'] != recipe or p['original_base_state_digest'] != audit['base_state'] or
                p['audit_sha256'] != sha256(out / 'audit.json') or p['source_archive_sha256'] != audit['source_archive_sha256'] or
                p['step'] != report['step']):
            raise ValueError('checkpoint provenance differs from audited run')
        if p['original_base_model_digest'] != audit['base_model_digest']:
            raise ValueError('original base digest differs from audit')
        if report['step'] == 500 and p['base_model_digest'] != result['base_model_digest_after']:
            raise ValueError('final live base digest differs from saved package')
        if report['step'] == 0 and p['base_state_digest'] != audit['base_state']:
            raise ValueError('step zero base was not pristine')
        packages.append({'step': report['step'], 'sha256': report['package_sha256'],
                         'base_model_digest': p['base_model_digest'], 'all_417_saved_hashes_verified': True,
                         'head_4_saved_hashes_verified': True, 'frozen_complement_411_unchanged': True})
    model = model_from_payload(package['base_payload'], device)
    head = CommandHead().to(device).eval()
    head.load_state_dict(package['head_state'], strict=True)
    with JointData(recipe, str(device)) as data:
        check_data(data, audit)
        rng = np.random.default_rng(0)
        for _ in range(500):
            data.sampler.draw(rng)
        if data.sampler.draw_counts() != result['draw_counts']:
            raise ValueError('fixed sampler draw accounting differs')
        metrics = validate(model, head, data.val, device)
        expected = {key: result['reports'][-1][key] for key in ('groups', 'passed', 'selection_score')}
        if metrics != expected:
            raise ValueError('fresh-process final validation differs')
    checked_audit(out, recipe)
    eligible = [report for report in result['reports'] if report['step'] > 0 and report['passed']]
    selected = min(eligible, key=lambda row: (row['selection_score'], row['step'])) if eligible else None
    reject_failed_attempt(out)
    require_training_parent(result)
    write_json(out / 'postflight.json', {'passed': True, 'audit_sha256': sha256(out / 'audit.json'),
               'fresh_process': True, 'process_id': os.getpid(), 'training_process_id': result['process_id'], 'packages': packages, 'final_validation_exact': True,
               'verification_exact': True, 'source_archive_exact': True})
    reject_failed_attempt(out)
    require_training_parent(result)
    write_json(out / 'train.json', {**result, 'postflight_complete': True, 'passed': bool(selected),
               'selected': selected, 'diagnostic_only_step': None if selected else 500,
               'postflight_sha256': sha256(out / 'postflight.json')})


def selected_package(out, explicit=None):
    reject_failed_attempt(out)
    result = read_json(out / 'train.json')
    postflight = read_json(out / 'postflight.json')
    if result.get('postflight_complete') is not True or postflight.get('passed') is not True or result['postflight_sha256'] != sha256(out / 'postflight.json'):
        raise ValueError('completed independent postflight required')
    if explicit is None:
        if not result['passed'] or result['selected'] is None:
            raise ValueError('eligible selected checkpoint required')
        report = result['selected']
    else:
        matches = [row for row in result['reports'] if Path(row['package']).resolve() == Path(explicit).resolve()]
        if len(matches) != 1:
            raise ValueError('runtime checkpoint must belong to this completed fixed run')
        report = matches[0]
    if sha256(report['package']) != report['package_sha256']:
        raise ValueError('chosen package changed')
    return report


def visual_run(out, recipe, device):
    audit = checked_audit(out, recipe)
    selected = selected_package(out)
    package = torch.load(selected['package'], map_location='cpu', weights_only=False)
    validate_package(package)
    with JointData(recipe, str(device)) as data:
        check_data(data, audit)
        measured = []
        for blank in (False, False, True):
            model = model_from_payload(package['base_payload'], device)
            head = CommandHead().to(device).eval()
            head.load_state_dict(package['head_state'], strict=True)
            with torch.no_grad():
                latent, prop, target, _ = batch_input(model, data.train, audit['visual_indices'], device, blank=blank)
                measured.append(float(command_loss(head, latent, prop, target)))
            if digest_state(model.state_dict()) != package['provenance']['base_state_digest'] or digest_state(head.state_dict()) != package['provenance']['head_state_digest']:
                raise ValueError('selected package mutated during visual gate')
        first, repeat, blanked = measured
        increase, noise = blanked - first, abs(repeat - first)
        passed = increase >= .1 * first and increase > 5 * noise
        write_json(out / 'visual.json', {'passed': passed, 'selected_sha256': selected['package_sha256'],
                   'normal_loss': first, 'repeat_loss': repeat, 'blank_history_loss': blanked,
                   'increase': increase, 'repeat_noise': noise, 'anchor_indices': audit['visual_indices'],
                   'blank': recipe['visual_gate']['blank'], 'all_saved_state_unchanged': True})
        if not passed:
            raise ValueError('full-history visual dependence gate failed')


def runtime_run(out, recipe, device, checkpoint=None):
    from fsop.sim.env import Obs
    from fsop.sim.vehicle import VehicleState
    from isaac_eval.evidence import ImageHeadCapture
    from isaac_lab_fsop.cases import Case, environment
    from isaac_lab_fsop.verify_inference import capture_policy_raw
    from verify_command_episode import compare, mismatches

    audit = checked_audit(out, recipe)
    report = selected_package(out, checkpoint)
    package_path = Path(report['package'])
    load_started = time.perf_counter()
    policy = JointActuatorPolicy.load(package_path, device='cpu')
    load_seconds = time.perf_counter() - load_started
    initial = [model_digest(policy.model), model_digest(policy.head)]
    datasets, stats, timings, diagnostic_timings = [], {}, [], []
    training_device_stats = {}
    diagnostic_policy = None
    if device.type != 'cpu':
        diagnostic_package = torch.load(package_path, map_location='cpu', weights_only=False)
        diagnostic_model = model_from_payload(diagnostic_package['base_payload'], device)
        diagnostic_head = CommandHead().to(device).eval()
        diagnostic_head.load_state_dict(diagnostic_package['head_state'], strict=True)
        diagnostic_policy = JointActuatorPolicy(diagnostic_model, diagnostic_head, CONTRACT, device)
    with JointData(recipe, str(device)) as data:
        check_data(data, audit)
    for discipline in ('skidpad', 'trackdrive'):
        path = Path(recipe['new_cases'][discipline]['episode'])
        metadata = read_json(path / 'meta.json')
        all_rows = read_json(path / 'proposals.json')
        rows = all_rows[:120]
        if metadata['split'] != 'train' or len(rows) < 4:
            raise ValueError('runtime preflight requires actual TRAIN RGB covering all four phases')
        images = {name: np.load(path / f'prefix_{name}_rgb.npy', mmap_mode='r', allow_pickle=False)
                  for name in ('road', 'wide')}
        try:
            env = environment(Case(**metadata['case']))
            actual_track, saved_track = json.loads(json.dumps(env.track.to_dict())), read_json(path / 'track.json')
            actual_track.pop('stats', None)
            saved_track.pop('stats', None)
            if actual_track != saved_track:
                raise ValueError('runtime TRAIN track differs')
            for name, array in images.items():
                if sha256(path / f'prefix_{name}_rgb.npy') != metadata['arrays_sha256'][f'prefix_{name}_rgb.npy']:
                    raise ValueError('runtime saved RGB differs')
                if array.dtype != np.uint8 or array.shape != (len(all_rows), 256, 512, 3):
                    raise ValueError('runtime RGB shape differs')
            def observation(index):
                row = rows[index]
                if row['tick'] != index or not np.isclose(row['state'][8], index / 20, rtol=0, atol=1e-8):
                    raise ValueError('runtime prefix ticks differ')
                for name in images:
                    if hashlib.sha256(np.ascontiguousarray(images[name][index]).tobytes()).hexdigest() != row['image_sha256'][name]:
                        raise ValueError('runtime frame image differs')
                state = VehicleState(*map(float, row['state']))
                env.state, env.frame = state.copy(), index
                env.s, env.lat, env.hint = row['progress_m'], row['lateral_m'], row['hint']
                return Obs(index, state.t, state, env.s, env.lat, {n: images[n][index] for n in images},
                           env.T_world_ego(), np.asarray(row['desire'], dtype=np.float32), [], 0.)
            policy.reset()
            warmup_started = time.perf_counter()
            with torch.inference_mode():
                for i in range(min(8, len(rows))):
                    policy(observation(i), env)
            warmup_seconds = time.perf_counter() - warmup_started
            policy.reset()
            records = []
            with torch.inference_mode():
                for i in range(len(rows)):
                    obs = observation(i)
                    t0 = time.perf_counter()
                    steer, accel, info = policy(obs, env)
                    timings.append({'case': discipline, 'frame': i, 'phase': i % 4,
                                    'seconds': time.perf_counter() - t0})
                    records.append({'action': [steer, accel], 'prediction': info})
            policy.reset()
            with capture_policy_raw(policy.model) as raw, ImageHeadCapture(policy.model) as head, torch.inference_mode():
                for i in range(len(rows)):
                    obs = observation(i)
                    raw.clear()
                    head.clear()
                    t0 = time.perf_counter()
                    steer, accel, info = policy(obs, env)
                    diagnostic_timings.append({'case': discipline, 'frame': i, 'phase': i % 4,
                                               'seconds': time.perf_counter() - t0})
                    if head.output is None or 'output' not in raw:
                        raise ValueError('runtime plan/cone diagnostics missing')
                    compare(records[i], {'action': [steer, accel], 'prediction': info}, stats, discipline + '.repeat')
                    ctrl = info['controller']
                    latent = torch.tensor(ctrl['latent'], dtype=torch.float32).reshape(1, 256)
                    proprio = torch.tensor(ctrl['proprio'], dtype=torch.float32).reshape(1, 3)
                    reconstructed = policy.head(latent, proprio)[0].cpu().numpy()
                    compare(ctrl['raw'], reconstructed, stats, discipline + '.saved_latent_head')
            if diagnostic_policy is not None:
                diagnostic_policy.reset()
                with torch.inference_mode():
                    for i in range(len(rows)):
                        steer, accel, info = diagnostic_policy(observation(i), env)
                        compare(records[i], {'action': [steer, accel], 'prediction': info},
                                training_device_stats, discipline + '.cpu_vs_training_device')
            datasets.append({'case': metadata['id'], 'available_prefix_frames': len(all_rows), 'frames': len(rows), 'phases': sorted({i % 4 for i in range(len(rows))}),
                             'proposals_sha256': sha256(path / 'proposals.json'), 'warmup_frames': min(8, len(rows)),
                             'warmup_seconds': warmup_seconds,
                             'rgb_sha256': {n: sha256(path / f'prefix_{n}_rgb.npy') for n in images}})
        finally:
            for array in images.values():
                array._mmap.close()
    unchanged = initial == [model_digest(policy.model), model_digest(policy.head)]
    passed = unchanged and not mismatches(stats) and all(row['phases'] == [0, 1, 2, 3] for row in datasets)
    def timing_summary(rows):
        return {'frames': len(rows), 'mean_seconds': float(np.mean([row['seconds'] for row in rows])),
                'p95_seconds': float(np.percentile([row['seconds'] for row in rows], 95)),
                'max_seconds': max(row['seconds'] for row in rows),
                'phases': {str(phase): {'frames': len([r for r in rows if r['phase'] == phase]),
                           'mean_seconds': float(np.mean([r['seconds'] for r in rows if r['phase'] == phase]))}
                           for phase in range(4)}}
    checked_audit(out, recipe)
    write_json(out / 'runtime.json', {'passed': passed, 'checkpoint_sha256': report['package_sha256'],
               'policy_device': 'cpu', 'torch_num_threads': 4, 'dtype': 'torch.float32', 'rtol': 1e-5, 'atol': 1e-5,
               'load_seconds': load_seconds, 'datasets': datasets, 'repeat_and_head_checks': stats,
               'all_saved_state_unchanged': unchanged, 'inference': timing_summary(timings),
               'inference_with_plan_cone_capture': timing_summary(diagnostic_timings),
               'inference_per_frame': timings, 'capture_per_frame': diagnostic_timings,
               'timing_excludes': ['Isaac physics', 'rendering', 'RGB file reads', 'observation reconstruction', 'evidence writes'],
               'whole_isaac_render_capture_seconds': None, 'wall_clock_20hz_claim': False,
               'training_device_output_comparison': {'device': str(device), 'measured': diagnostic_policy is not None,
                   'stats': training_device_stats, 'qualification_gate': False,
                   'reason': 'cross-device diagnostics never relax CPU saved-input replay tolerances'},
               'limits': ['Actual saved TRAIN inputs with full sequential history, not a physical replay.',
                          'This gate does not establish mission success or a 20 Hz wall-clock rate.']})
    if not passed:
        raise ValueError('fixed CPU runtime preflight failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit', 'condition', 'overfit', 'train', 'postflight', 'visual', 'runtime'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda', help='training and original recovery receipt replay device')
    parser.add_argument('--checkpoint', type=Path, help='runtime only, an explicit diagnostic package in this completed run')
    args = parser.parse_args()
    if args.checkpoint is not None and args.stage != 'runtime':
        parser.error('--checkpoint is only valid for runtime')
    if args.stage == 'audit':
        args.out.mkdir(parents=True, exist_ok=False)
    elif not args.out.is_dir():
        raise ValueError('run audit first')
    reject_failed_attempt(args.out)
    reject_interrupted_attempt(args.out, args.stage)
    failure = args.out / (args.stage + '_failure.json')
    attempt = args.out / (args.stage + '_attempt.json')
    if attempt.exists() or (args.out / (args.stage + '.json')).exists():
        raise FileExistsError('stage already attempted; use a new output directory')
    if args.stage == 'train' and ((args.out / 'steps.jsonl').exists() or (args.out / 'training_updates.json').exists()):
        raise FileExistsError('main fit already attempted; resuming is prohibited')
    write_json(attempt, {'stage': args.stage})
    try:
        recipe = read_json(RECIPE)
        check_recipe(recipe)
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('selected training device unavailable')
        if args.stage == 'runtime':
            runtime_run(args.out, recipe, device, args.checkpoint)
        else:
            {'audit': audit_run, 'condition': condition_run, 'overfit': overfit_run, 'train': train_run,
             'postflight': postflight_run, 'visual': visual_run}[args.stage](args.out, recipe, device)
    except BaseException as exc:
        write_json(failure, {'stage': args.stage, 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
