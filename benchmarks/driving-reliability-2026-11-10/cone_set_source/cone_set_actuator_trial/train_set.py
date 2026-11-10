from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fsop.eval.det_metrics import decode_img
from fsop.eval.img_to_bev import bev_from_image_head, ground_points
from fsop.geometry.yuv import rgb_to_model
from fsop.train.lab_train import sha256
from isaac_lab_fsop.evaluate import source_inventory
from cone_core import (BASE_DIGEST, HEAD_SCOPE, KIND, ConeSetHead, check_frozen_base,
                       cpu_image_head, decode_batch, digest_state, make_optimizer, make_package,
                       model_digest, model_from_payload, pack_cones, validate_package)

HERE = Path(__file__).resolve().parent
JOINT = HERE.parent / 'joint_command_trial'
OLD = HERE.parent / 'direct_command_trial'
RECOVERY = HERE.parent / 'direct_command_recovery'
DATA_TRIAL = HERE.parent / 'direct_command_recovery_trial'
DESIGN = HERE.parent / 'next_strategy'
RECIPE = HERE / 'recipe.json'
sys.path.append(str(JOINT))
from joint_data import JointData, read_json, wiring_indices

FILES = ('cone_core.py', 'train_set.py', 'recipe.json', 'protocol.md', 'test_core.py', 'test_training.py',
         'evaluate.py', 'verify_command_episode.py', 'process.py', 'export_audited_command.py',
         'encode_portable.py', 'pipeline_notes.md', 'test_pipeline.py', 'runtime_preflight.py')


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


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


def scores(errors):
    values = np.asarray(errors)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('empty or nonfinite metric group')
    mae, p95 = values.mean(0), np.percentile(values, 95, axis=0)
    return {'count': len(values), 'mae': mae.tolist(), 'p95': p95.tolist(),
            'passed': bool(np.all(mae <= [.01, .25]) and np.all(p95 <= [.03, .75]))}


def anchors(dataset, ids):
    return [{'case': dataset.episodes[dataset.index[i][0]].meta['id'], 'frame': dataset.index[i][1],
             'tick': int(dataset.episodes[dataset.index[i][0]].ticks[dataset.index[i][1]])} for i in ids]


def reject_failed_attempt(out):
    failures = sorted(Path(out).glob('*_failure.json'))
    if failures:
        raise ValueError('failed attempt cannot resume: ' + ', '.join(path.name for path in failures))


def reject_interrupted_attempt(out, stage):
    for attempt in Path(out).glob('*_attempt.json'):
        name = attempt.name.removesuffix('_attempt.json')
        if name == 'train' and stage == 'postflight':
            continue
        require_complete(Path(out), name)


def require_complete(out, stage):
    marker = out / (stage + '_complete.json')
    if not marker.is_file():
        raise FileExistsError(f'interrupted {stage} attempt lacks final completion marker')
    completed = read_json(marker)
    attempt = out / (stage + '_attempt.json')
    if (completed['stage'] != stage or completed['receipt_sha256'] != sha256(out / (stage + '.json')) or
            completed['attempt_sha256'] != sha256(attempt) or
            completed['process_id'] != read_json(attempt)['process_id']):
        raise ValueError(f'{stage} completed receipt changed or completion binding differs')
    if stage == 'postflight' and completed['train_sha256'] != sha256(out / 'train.json'):
        raise ValueError('postflight train receipt changed before parent finalization')
    return completed


def mark_complete(out, stage):
    attempt = out / (stage + '_attempt.json')
    if read_json(attempt)['process_id'] != os.getpid():
        raise ValueError('only the stage owner may finalize completion')
    record = {'stage': stage, 'process_id': os.getpid(),
              'attempt_sha256': sha256(attempt), 'receipt_sha256': sha256(out / (stage + '.json'))}
    if stage == 'postflight':
        record['train_sha256'] = sha256(out / 'train.json')
    write_json(out / (stage + '_complete.json'), record)


def require_training_parent(result):
    parent = result['process_id']
    if type(parent) is not int or parent <= 1 or parent == os.getpid() or os.getppid() != parent:
        raise ValueError('postflight requires its live training parent')
    try:
        os.kill(parent, 0)
    except ProcessLookupError as exc:
        raise ValueError('postflight requires its live training parent') from exc


def checked_audit(out, recipe, require_completion=True):
    reject_failed_attempt(out)
    if require_completion:
        require_complete(out, 'audit')
    audit = read_json(out / 'audit.json')
    if audit['recipe'] != recipe or audit['recipe_sha256'] != sha256(RECIPE) or audit['sources'] != inventory(recipe):
        raise ValueError('recipe or source changed after audit')
    check_archive(out, audit)
    if (sha256(out / 'collisions.json') != audit['collision_sha256'] or
            sha256(recipe['inherited_audit']) != audit['inherited_audit_sha256']):
        raise ValueError('collision or inherited fixed-anchor audit changed')
    return audit


def check_data(data, audit):
    if data.verification != audit['verification'] or data.sampler.protocol() != audit['sampler']:
        raise ValueError('dataset verification or sampler changed after audit')


def gate(out, name):
    require_complete(out, name)
    value = read_json(out / (name + '.json'))
    if value.get('passed') is not True or value['audit_sha256'] != sha256(out / 'audit.json'):
        raise ValueError(f'matching successful {name} gate required')
    return value


def training_prerequisites(out):
    require_complete(out, 'audit')
    gate(out, 'condition')
    wiring = gate(out, 'overfit')
    hashes = {name + '_sha256': sha256(out / (name + '.json')) for name in ('audit', 'condition', 'overfit')}
    hashes.update({name + '_complete_sha256': sha256(out / (name + '_complete.json'))
                   for name in ('audit', 'condition', 'overfit')})
    if wiring['condition_sha256'] != hashes['condition_sha256']:
        raise ValueError('conditioning receipt changed after wiring')
    return hashes


def check_prerequisites(out, expected):
    if training_prerequisites(out) != expected:
        raise ValueError('training prerequisite receipts changed after binding')


def recorded_prerequisites(record):
    return {name + suffix: record[name + suffix] for name in ('audit', 'condition', 'overfit')
            for suffix in ('_sha256', '_complete_sha256')}


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


def selected_package(out, explicit=None):
    reject_failed_attempt(out)
    require_complete(out, 'train')
    require_complete(out, 'postflight')
    result = read_json(out / 'train.json')
    check_prerequisites(out, recorded_prerequisites(result))
    postflight = read_json(out / 'postflight.json')
    if result.get('postflight_complete') is not True or postflight.get('passed') is not True or result['postflight_sha256'] != sha256(out / 'postflight.json'):
        raise ValueError('completed independent postflight required')
    if explicit is None:
        if not result['passed'] or result['selected'] is None:
            raise ValueError('eligible selected checkpoint required')
        report = result['selected']
    else:
        raise ValueError('only the single offline-selected checkpoint may continue')
    if sha256(report['package']) != report['package_sha256']:
        raise ValueError('chosen package changed')
    return report


def check_recipe(recipe):
    legacy = read_json(JOINT / 'recipe.json')
    changed = {'kind', 'trial_version', 'training_scope'}
    if any(recipe.get(k) != value for k, value in legacy.items() if k not in changed):
        raise ValueError('recipe changed an inherited fitting or gate field')
    if (recipe['kind'] != KIND or recipe['training_scope'] != HEAD_SCOPE or
            recipe['base_model_digest'] != BASE_DIGEST or
            recipe['command_image_context_seconds'] != .2 or recipe['command_history_features'] != 0 or
            recipe['legacy_recipe_sha256'] != sha256(JOINT / 'recipe.json') or
            recipe['inherited_audit_sha256'] != 'de5c27dd4af484705d27ff55e20e0f9dc56986b3d6cddc47553eef2cdb6f95e6' or
            recipe['probe_cases'] != ['train-skidpad-82000-matched', 'train-trackdrive-82000-matched'] or
            recipe['probe_frames'] != 573 or recipe['cpu_validation']['anchors'] != 2067):
        raise ValueError('cone-set recipe identity differs')


def inventory(recipe):
    paths = [HERE / name for name in FILES]
    paths += [JOINT / name for name in ('joint_core.py', 'joint_data.py', 'recipe.json', 'protocol.md')]
    paths += [DATA_TRIAL / name for name in ('trial_v2.py', 'recipe_v2.json', 'protocol_v2.md', 'test_trial_v2.py')]
    paths += [RECOVERY / name for name in ('collect.py', 'verify.py', 'verify_v2.py', 'protocol.md',
                                         'derived_verifier_v2.md', 'membership.json', 'test_bindings.py')]
    paths += [OLD / name for name in ('actuator_trial.py', 'recipe.json', 'evaluate_v2.py')]
    paths += [DESIGN / name for name in ('cone_set_synthesis_v1.md', 'cone_set_interface_v1.md',
                                       'cone_set_actuator_design_v1.md')]
    paths += list((RECOVERY / 'cases').glob('*.json'))
    paths += [Path(name) for name in ('scripts/verify_lab_dataset.py', 'scripts/verify_lab_recovery.py',
                                    'scripts/audit_isaac_episode.py', 'docs/isaac-lab-training/phase_recovery_recipe4.json')]
    paths += [Path(recipe[k]) for k in ('recovery_recipe', 'original_membership', 'new_membership', 'legacy_recipe')]
    paths += [Path('artifacts/models/prep') / name for name in ('vision_fp32.onnx', 'policy_fp32.onnx')]
    result, root = {}, Path.cwd().resolve()
    for name in [*source_inventory(), *paths]:
        path = Path(name).resolve()
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        value = sha256(path)
        if key in result and result[key] != value:
            raise ValueError('source inventory disagrees')
        result[key] = value
    return dict(sorted(result.items()))


def data_owner(recipe, device):
    if sha256(recipe['legacy_recipe']) != recipe['legacy_recipe_sha256']:
        raise ValueError('canonical data recipe changed')
    return JointData(read_json(recipe['legacy_recipe']), str(device))


def original_model(recipe, device):
    payload = torch.load(recipe['base_checkpoint'], map_location='cpu', weights_only=False)
    return payload, model_from_payload(payload, device)


def new_fit(model, device):
    torch.manual_seed(0)
    head = ConeSetHead().to(device).eval()
    optimizer, named = make_optimizer(model, head)
    return head, optimizer, named


@dataclass(frozen=True)
class Batch:
    tokens: torch.Tensor
    mask: torch.Tensor
    desire: torch.Tensor
    proprio: torch.Tensor
    target: torch.Tensor

    @property
    def inputs(self):
        return self.tokens, self.mask, self.desire, self.proprio


def infer_rows(model, rows, device, blank=False):
    blank_frame = rgb_to_model(np.zeros((256, 512, 3), dtype=np.uint8)) if blank else None
    blank_pair = np.concatenate((blank_frame, blank_frame)) if blank else None
    frames = {camera: torch.from_numpy(np.stack([blank_pair if blank else ep.frame_pair(camera, frame)
                                               for ep, frame in rows])).to(device).float()
              for camera in ('road', 'wide')}
    calibrations = [np.asarray(ep.meta['calibration']['T_ego_cam'], np.float64) for ep, _ in rows]
    with torch.no_grad():
        value = model.base.encode(frames['road'], frames['wide'])
        raw = model.img(value.s4, value.s8, value.s16)
        cpu = cpu_image_head(raw)
        decoded = decode_batch(cpu, calibrations)
    packed = [pack_cones(*row) for row in decoded]
    tokens = torch.from_numpy(np.stack([row[0] for row in packed])).to(device)
    mask = torch.from_numpy(np.stack([row[1] for row in packed])).to(device)
    if torch.is_inference(tokens) or torch.is_inference(mask):
        raise ValueError('training inputs must be ordinary detached tensors')
    return tokens, mask, cpu, decoded


def batch_input(model, dataset, ids, device, blank=False):
    rows = [(dataset.episodes[dataset.index[int(i)][0]], dataset.index[int(i)][1]) for i in ids]
    for ep, frame in rows:
        if not ep.teacher[int(ep.ticks[frame])]:
            raise ValueError('same-tick command is not a teacher control')
    tokens, mask, _, _ = infer_rows(model, rows, device, blank)
    desire = torch.from_numpy(np.stack([np.array(ep.desires[frame], np.float32, copy=True)
                                      for ep, frame in rows])).to(device)
    proprio = torch.from_numpy(np.stack([np.array(ep.states[int(ep.ticks[frame]), 3:6], np.float32, copy=True)
                                       for ep, frame in rows])).to(device)
    target = torch.from_numpy(np.stack([np.array(ep.controls[int(ep.ticks[frame])], np.float32, copy=True)
                                      for ep, frame in rows])).to(device)
    return Batch(tokens, mask, desire, proprio, target)


def command_loss(head, batch):
    error = (head(*batch.inputs) - batch.target) / batch.target.new_tensor([.02, .5])
    return F.smooth_l1_loss(error, torch.zeros_like(error))


def validate(model, head, val, device, batch_size=64):
    groups = {'all': []}
    with torch.no_grad():
        for start in range(0, len(val), batch_size):
            ids = list(range(start, min(start + batch_size, len(val))))
            batch = batch_input(model, val, ids, device)
            errors = (head(*batch.inputs) - batch.target).abs().cpu().numpy()
            for offset, index in enumerate(ids):
                ei, frame = val.index[index]
                ep = val.episodes[ei]
                keys = ['all', 'episode/' + str(ep.meta['id'])]
                if batch.proprio[offset, 0].item() < 2:
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


def input_hash(batch, index):
    digest = hashlib.sha256()
    for value in batch.inputs:
        array = value[index].detach().cpu().contiguous().numpy()
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def minimum_p95(labels, groups):
    values = np.asarray(labels, dtype=np.float64)
    n = len(values)
    h = .95 * (n - 1)
    lower, upper = int(np.floor(h)), int(np.ceil(h))
    weight = h - lower
    best = float('inf')
    for outside_low in itertools.combinations(range(n), n - lower - 1):
        for outside_high in itertools.combinations(outside_low, n - upper - 1):
            a_min = b_min = sum_min = 0.
            for indices in groups:
                low = [values[i] for i in indices if i not in outside_low]
                high = [values[i] for i in indices if i not in outside_high]
                if low:
                    a_min = max(a_min, (max(low) - min(low)) / 2)
                if high:
                    b_min = max(b_min, (max(high) - min(high)) / 2)
                if low and high:
                    sum_min = max(sum_min, max(low) - min(high), max(high) - min(low))
            for a in (a_min, max(a_min, b_min), max(a_min, sum_min / 2), max(a_min, sum_min - b_min)):
                b = max(b_min, a, sum_min - a)
                best = min(best, (1 - weight) * a + weight * b)
    return float(best)


def collision_screen(batch, dataset, ids):
    target = batch.target.detach().cpu().numpy().astype(np.float64)
    groups = {}
    for offset in range(len(ids)):
        groups.setdefault(input_hash(batch, offset), []).append(offset)
    residual = np.zeros_like(target)
    report = []
    for digest, indices in groups.items():
        median = np.median(target[indices], axis=0)
        residual[indices] = np.abs(target[indices] - median)
        entries = []
        for offset in indices:
            ep, frame = dataset.index[ids[offset]]
            episode = dataset.episodes[ep]
            entries.append({'anchor': anchors(dataset, [ids[offset]])[0], 'target': target[offset].tolist(),
                            'teacher_assertion': bool(episode.teacher[int(episode.ticks[frame])]),
                            'teacher_provenance': {key: episode.meta.get(key) for key in
                                                   ('teacher', 'teacher_config', 'collection_policy')},
                            'empty_set': not bool(batch.mask[offset].any())})
        report.append({'input_sha256': digest, 'members': entries, 'median_target': median.tolist(),
                       'median_mae': residual[indices].mean(0).tolist(),
                       'minimum_p95': [minimum_p95(target[indices, c], [list(range(len(indices)))])
                                       for c in range(2)]})
    mae = residual.mean(0)
    p95 = [minimum_p95(target[:, c], list(groups.values())) for c in range(2)]
    arrays = [value.detach().cpu().numpy().reshape(len(ids), -1).astype(np.float64) for value in batch.inputs]
    flat = np.concatenate(arrays, axis=1)
    nearest = []
    for i in range(len(ids)):
        distances = np.max(np.abs(flat - flat[i]), axis=1)
        distances[i] = np.inf
        j = int(np.argmin(distances))
        nearest.append({'anchor_index': ids[i], 'nearest_anchor_index': ids[j],
                        'maximum_input_difference': float(distances[j]),
                        'target_difference': np.abs(target[i] - target[j]).tolist()})
    return {'passed': bool(np.all(mae <= [.002, .05])), 'groups': report,
            'median_mae_lower_bound': mae.tolist(), 'minimum_achievable_p95': p95,
            'p95_method': 'exact interval feasibility for both linearly interpolated percentile order statistics',
            'nearest_inputs_diagnostic': nearest, 'empty_set_indices': [ids[i] for i in range(len(ids))
                                                                       if not bool(batch.mask[i].any())],
            'route_transition_indices': [ids[i] for i in range(len(ids)) if bool(batch.desire[i, -9:].any())],
            'limits': 'Exact equality only; near inputs do not prove contradictory supervision.'}


def decoder_counts(raw, calibrations):
    result = []
    for b, calibration in enumerate(calibrations):
        cameras = {}
        for camera, limit in (('wide', 22.), ('road', 45.)):
            fields = raw[camera]
            boxes, classes, _ = decode_img(*(fields[k][b:b + 1] for k in ('heat', 'wh', 'off')), thresh=.25)[0]
            uv = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
            xy, valid = ground_points(uv, camera, calibration)
            valid &= np.linalg.norm(xy, axis=1) < limit
            cameras[camera] = {'peaks': len(classes), 'valid_projection_and_range': int(valid.sum())}
        merged = bev_from_image_head(raw, calibration, b=b)
        before = sum(row['valid_projection_and_range'] for row in cameras.values())
        if before > 200 or len(merged[0]) > before:
            raise ValueError('decoder count contract violated')
        result.append({'cameras': cameras, 'before_merge': before, 'after_merge': len(merged[0])})
    return result


def decoder_audit(model, data, recipe, device):
    episodes = [ep for ep in data.episodes if ep.meta['id'] in recipe['probe_cases']]
    if len(episodes) != 2 or sum(len(ep.ticks) for ep in episodes) != recipe['probe_frames']:
        raise ValueError('complete 573-frame original TRAIN probe membership differs')
    rows = [(ep, frame) for ep in episodes for frame in range(len(ep.ticks))]
    records = []
    for start in range(0, len(rows), 32):
        group = rows[start:start + 32]
        _, _, raw, decoded = infer_rows(model, group, device)
        calibrations = [np.asarray(ep.meta['calibration']['T_ego_cam'], np.float64) for ep, _ in group]
        counts = decoder_counts(raw, calibrations)
        for i, ((ep, frame), actual) in enumerate(zip(group, decoded, strict=True)):
            singleton = {cam: {name: value[i:i + 1] for name, value in raw[cam].items()} for cam in raw}
            expected = bev_from_image_head(singleton, calibrations[i])
            repeated = decode_batch(singleton, [calibrations[i]])[0]
            if any(not np.array_equal(a, b) for a, b in zip(actual, expected, strict=True)) or any(
                    not np.array_equal(a, b) for a, b in zip(actual, repeated, strict=True)):
                raise ValueError('stock singleton decoder parity failed on identical raw tensors')
            records.append({'case': ep.meta['id'], 'frame': frame, **counts[i]})
    return {'frames': records, 'frame_count': len(records), 'identical_raw_stock_singleton_exact': True,
            'decoder_cpu_repeat_exact': True, 'new_tie_rule': False}


def comparison(left, right):
    return {'exact': all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(left.inputs, right.inputs, strict=True)),
            'mask_exact': torch.equal(left.mask.cpu(), right.mask.cpu()),
            'class_onehot_exact': torch.equal(left.tokens[:, :, 3:7].cpu(), right.tokens[:, :, 3:7].cpu()),
            'maximum_token_difference': float((left.tokens.cpu() - right.tokens.cpu()).abs().max()),
            'left_counts': left.mask.sum(1).cpu().tolist(), 'right_counts': right.mask.sum(1).cpu().tolist()}


def audit_run(out, recipe, device):
    sources = inventory(recipe)
    if sha256(recipe['inherited_audit']) != recipe['inherited_audit_sha256']:
        raise ValueError('inherited fixed-anchor audit differs')
    inherited = read_json(recipe['inherited_audit'])
    inherited_sources = {key: value for key, value in sources.items() if key in inherited['sources']}
    if any(value != inherited['sources'][key] for key, value in inherited_sources.items()):
        raise ValueError('frozen original source or prepared graph changed')
    payload, model = original_model(recipe, device)
    head, _, named = new_fit(model, device)
    with data_owner(recipe, device) as data:
        ids = wiring_indices(data.sampler)
        if ids != inherited['overfit_indices'] or anchors(data.train, ids) != inherited['overfit_anchors']:
            raise ValueError('fixed 32 ordered TRAIN anchor identities differ')
        visual_ids = inherited['visual_indices']
        if anchors(data.train, visual_ids) != inherited['visual_anchors']:
            raise ValueError('fixed visual identities differ')
        batch = batch_input(model, data.train, ids, device)
        collisions = collision_screen(batch, data.train, ids)
        write_json(out / 'collisions.json', collisions)
        if not collisions['passed']:
            raise ValueError('exact input conflicts make inherited wiring MAE unattainable')
        decoder = decoder_audit(model, data, recipe, device)
        cpu_model = model_from_payload(payload, 'cpu')
        def cpu_singletons():
            rows = [batch_input(cpu_model, data.train, [i], 'cpu') for i in ids]
            return Batch(*(torch.cat([getattr(row, field) for row in rows]) for field in
                           ('tokens', 'mask', 'desire', 'proprio', 'target')))
        cpu_batch, cpu_repeat = cpu_singletons(), cpu_singletons()
        if not comparison(cpu_batch, cpu_repeat)['exact']:
            raise ValueError('CPU repeated fresh inputs differ')
        cpu_head = ConeSetHead().eval()
        cpu_head.load_state_dict(head.state_dict(), strict=True)
        with torch.no_grad():
            cpu_command, cpu_repeat_command = cpu_head(*cpu_batch.inputs), cpu_head(*cpu_repeat.inputs)
            if not torch.equal(cpu_command, cpu_repeat_command):
                raise ValueError('CPU repeated command differs')
            training_command = head(*batch.inputs).cpu()
        singleton = [batch_input(model, data.train, [i], device) for i in ids[:4]]
        joined = Batch(*(torch.cat([getattr(row, field) for row in singleton]) for field in
                         ('tokens', 'mask', 'desire', 'proprio', 'target')))
        batched_four = batch_input(model, data.train, ids[:4], device)
        check_frozen_base(model, digest_state(payload['model']))
        mapping = archive_sources(out, sources)
        record = {'recipe': recipe, 'recipe_sha256': sha256(RECIPE), 'sources': sources,
                  'source_archive_sha256': sha256(out / 'source.zip'), 'source_archive_members': mapping,
                  'verification': data.verification, 'sampler': data.sampler.protocol(),
                  'train_count': len(data.train), 'val_count': len(data.val),
                  'conditioning_indices': ids, 'overfit_indices': ids, 'visual_indices': visual_ids,
                  'overfit_anchors': anchors(data.train, ids), 'visual_anchors': anchors(data.train, visual_ids),
                  'inherited_audit_sha256': sha256(recipe['inherited_audit']),
                  'inherited_frozen_source_count': len(inherited_sources),
                  'base_state': digest_state(payload['model']), 'base_model_digest': model_digest(model),
                  'head_initial_state': digest_state(head.state_dict()),
                  'optimizer_parameters': {name: list(value.shape) for name, value in named.items()},
                  'trainable_parameters': 47618, 'base_tensors': 417, 'frozen_base_tensors': 417,
                  'device': str(device), 'feature_cache': False, 'decoder': decoder,
                  'collision_sha256': sha256(out / 'collisions.json'), 'cpu_repeat_exact': True,
                  'batch_singleton_encoder_diagnostic': comparison(batched_four, joined),
                  'training_device_cpu_diagnostic': {**comparison(batch, cpu_batch), 'qualification_gate': False,
                      'cpu_encoder_batch_size': 1,
                      'max_command_difference': (training_command - cpu_command).abs().max(0).values.tolist()}}
        if inventory(recipe) != sources:
            raise ValueError('sources changed during audit')
        write_json(out / 'audit.json', record)


def update(model, head, optimizer, named, dataset, ids, device):
    batch = batch_input(model, dataset, ids, device)
    optimizer.zero_grad(set_to_none=True)
    value = command_loss(head, batch)
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
    with data_owner(recipe, device) as data:
        check_data(data, audit)
        ids = audit['conditioning_indices']
        def measured():
            with torch.no_grad():
                batch = batch_input(model, data.train, ids, device)
                return float(command_loss(head, batch)), head(*batch.inputs).detach().clone()
        first, _ = measured()
        repeat, _ = measured()
        result = update(model, head, optimizer, named, data.train, ids, device)
        after, commands = measured()
        if any(value <= 0 for value in result['gradient_norms'].values()):
            raise ValueError('conditioning requires nonzero gradients in every optimized tensor')
        check_frozen_base(model, audit['base_state'])
        update_norms = {name: float((value.detach() - before[name]).norm()) for name, value in named.items()}
        package = make_package(original, model, head, recipe, provenance(out, audit), 1)
        space(out, recipe)
        with tempfile.TemporaryDirectory(prefix='cone-conditioning-') as directory:
            path = Path(directory) / 'condition.pt'
            saved = save_package(path, package)
            restored = model_from_payload(saved['base_payload'], device)
            restored_head = ConeSetHead().to(device).eval()
            restored_head.load_state_dict(saved['head_state'], strict=True)
            with torch.no_grad():
                restored_batch = batch_input(restored, data.train, ids, device)
                roundtrip = restored_head(*restored_batch.inputs)
            exact = torch.equal(commands, roundtrip) and digest_state(restored_head.state_dict()) == digest_state(head.state_dict())
            roundtrip_sha = sha256(path)
        noise = abs(first - repeat)
        passed = (first - after > 5 * noise and first - after > 0 and exact
                  and all(value > 0 for value in update_norms.values()))
        write_json(out / 'condition.json', {'passed': passed, 'audit_sha256': sha256(out / 'audit.json'),
                   'step': 1, 'indices': ids, 'normal_loss': first, 'repeat_loss': repeat, 'updated_loss': after,
                   'repeat_noise': noise, 'improvement': first - after, **result, 'update_norms': update_norms,
                   'frozen_base_state_unchanged': True,
                   'frozen_base_tensors': 417, 'roundtrip_commands_exact': exact,
                   'roundtrip_package_sha256': roundtrip_sha, 'discarded_all_weights': True})
        if not passed:
            raise ValueError('fixed-rate conditioning failed; this recipe stops without a rate sweep')


def overfit_run(out, recipe, device):
    audit = checked_audit(out, recipe)
    gate(out, 'condition')
    condition_sha = sha256(out / 'condition.json')
    _, model = original_model(recipe, device)
    head, optimizer, named = new_fit(model, device)
    started = time.monotonic()
    with data_owner(recipe, device) as data:
        check_data(data, audit)
        ids = audit['overfit_indices']
        for step in range(201):
            with torch.no_grad():
                batch = batch_input(model, data.train, ids, device)
                mae = (head(*batch.inputs) - batch.target).abs().mean(0).cpu().tolist()
            if np.all(np.asarray(mae) <= [.002, .05]) or step == 200:
                break
            update(model, head, optimizer, named, data.train, ids, device)
        check_frozen_base(model, audit['base_state'])
        gate(out, 'condition')
        if sha256(out / 'condition.json') != condition_sha:
            raise ValueError('condition receipt changed during wiring')
        passed = bool(np.all(np.asarray(mae) <= [.002, .05]))
        write_json(out / 'overfit.json', {'passed': passed, 'step': step, 'mae': mae,
                   'per_anchor': [{'anchor': anchor, 'absolute_error': error, 'empty_set': not bool(mask.any())}
                                  for anchor, error, mask in zip(anchors(data.train, ids),
                                      (head(*batch.inputs) - batch.target).abs().detach().cpu().tolist(),
                                      batch.mask, strict=True)],
                   'audit_sha256': sha256(out / 'audit.json'), 'condition_sha256': condition_sha,
                   'seconds': time.monotonic() - started, 'discarded_all_weights': True,
                   'frozen_base_state_unchanged': True, 'frozen_base_tensors': 417})
        if not passed:
            raise ValueError('fixed 32-anchor wiring gate failed; main run prohibited')


def train_run(out, recipe, device):
    audit = checked_audit(out, recipe)
    prerequisites = training_prerequisites(out)
    original, model = original_model(recipe, device)
    head, optimizer, named = new_fit(model, device)
    if digest_state(head.state_dict()) != audit['head_initial_state'] or model_digest(model) != audit['base_model_digest']:
        raise ValueError('main fit did not start at pristine base and seed-zero head')
    rng, reports = np.random.default_rng(0), []
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with data_owner(recipe, device) as data:
        check_data(data, audit)
        for step in range(501):
            if step in recipe['evaluation_steps']:
                check_frozen_base(model, audit['base_state'])
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
        check_frozen_base(model, audit['base_state'])
        record = {'updates_complete': True, 'postflight_complete': False, 'reports': reports,
                  'process_id': os.getpid(),
                  **prerequisites, 'device': str(device),
                  'seconds': time.monotonic() - started, 'optimizer_parameters': audit['optimizer_parameters'],
                  'base_model_digest_after': model_digest(model),
                  'frozen_base_state_unchanged': True, 'frozen_base_tensors': 417,
                  'draw_counts': data.sampler.draw_counts(),
                  'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    checked_audit(out, recipe)
    check_prerequisites(out, prerequisites)
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
    prerequisites = recorded_prerequisites(result)
    check_prerequisites(out, prerequisites)
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
        if report['step'] == 0 and p['head_state_digest'] != audit['head_initial_state']:
            raise ValueError('step zero head differs from seed-zero initialization')
        packages.append({'step': report['step'], 'sha256': report['package_sha256'],
                         'base_model_digest': p['base_model_digest'], 'all_417_saved_hashes_verified': True,
                         'head_8_saved_hashes_verified': True, 'all_417_base_states_unchanged': True})
    model = model_from_payload(package['base_payload'], device)
    head = ConeSetHead().to(device).eval()
    head.load_state_dict(package['head_state'], strict=True)
    with data_owner(recipe, device) as data:
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
    checked_audit(out, recipe)
    check_prerequisites(out, prerequisites)
    write_json(out / 'postflight.json', {'passed': True, 'audit_sha256': sha256(out / 'audit.json'),
               'fresh_process': True, 'process_id': os.getpid(), 'training_process_id': result['process_id'], 'packages': packages, 'final_validation_exact': True,
               'verification_exact': True, 'source_archive_exact': True})
    reject_failed_attempt(out)
    require_training_parent(result)
    checked_audit(out, recipe)
    check_prerequisites(out, prerequisites)
    write_json(out / 'train.json', {**result, 'postflight_complete': True, 'passed': bool(selected),
               'selected': selected, 'diagnostic_only_step': None if selected else 500,
               'postflight_sha256': sha256(out / 'postflight.json')})


def selected_models(out, device):
    selected = selected_package(out)
    package = torch.load(selected['package'], map_location='cpu', weights_only=False)
    validate_package(package)
    model = model_from_payload(package['base_payload'], device)
    head = ConeSetHead().to(device).eval()
    head.load_state_dict(package['head_state'], strict=True)
    return selected, package, model, head


def cpu_validate_run(out, recipe, device):
    audit = checked_audit(out, recipe)
    selected, package, model, head = selected_models(out, 'cpu')
    started = time.perf_counter()
    with data_owner(recipe, device) as data:
        check_data(data, audit)
        if len(data.val) != 2067:
            raise ValueError('CPU deployment validation membership differs')
        metrics = validate(model, head, data.val, 'cpu', batch_size=1)
    check_frozen_base(model, audit['base_state'])
    if digest_state(head.state_dict()) != package['provenance']['head_state_digest']:
        raise ValueError('CPU validation mutated head')
    checked_audit(out, recipe)
    record = {**metrics, 'audit_sha256': sha256(out / 'audit.json'),
              'selected_sha256': selected['package_sha256'], 'selected_step': selected['step'],
              'fresh_anchors': 2067, 'encoder_batch_size': 1, 'device': 'cpu', 'threads': 4,
              'dtype': 'torch.float32', 'seconds': time.perf_counter() - started,
              'replacement_allowed': False, 'all_saved_state_unchanged': True}
    write_json(out / 'cpu_validate.json', record)
    if not metrics['passed']:
        raise ValueError('selected package failed full singleton CPU validation; no checkpoint replacement')


def selected_gate(out, name, selected):
    receipt = gate(out, name)
    if receipt['selected_sha256'] != selected['package_sha256']:
        raise ValueError(f'{name} receipt belongs to another selected package')
    return receipt


def visual_run(out, recipe, device):
    audit = checked_audit(out, recipe)
    selected = selected_package(out)
    selected_gate(out, 'cpu_validate', selected)
    measured, counts = {}, {}
    with data_owner(recipe, device) as data:
        check_data(data, audit)
        ids = audit['visual_indices']
        pairs = [data.train.index[i] for i in ids]
        swap = list(range(len(ids)))
        for episode in sorted({ei for ei, _ in pairs}):
            positions = [i for i, (ei, _) in enumerate(pairs) if ei == episode]
            for i, j in zip(positions, positions[1:] + positions[:1], strict=True):
                swap[i] = j
        for arm in ('normal', 'repeat', 'black_images', 'zero_tokens', 'scene_matched_swap'):
            _, package, model, head = selected_models(out, 'cpu')
            rows = [batch_input(model, data.train, [i], 'cpu', blank=arm == 'black_images') for i in ids]
            batch = Batch(*(torch.cat([getattr(row, field) for row in rows]) for field in
                            ('tokens', 'mask', 'desire', 'proprio', 'target')))
            if arm == 'zero_tokens':
                batch = Batch(torch.zeros_like(batch.tokens), torch.zeros_like(batch.mask),
                              batch.desire, batch.proprio, batch.target)
            elif arm == 'scene_matched_swap':
                batch = Batch(batch.tokens[swap], batch.mask[swap], batch.desire, batch.proprio, batch.target)
            with torch.no_grad():
                measured[arm] = float(command_loss(head, batch))
            counts[arm] = batch.mask.sum(1).tolist()
            check_frozen_base(model, audit['base_state'])
            if digest_state(head.state_dict()) != package['provenance']['head_state_digest']:
                raise ValueError('visual arm mutated selected package')
        first, repeat, black = (measured[key] for key in ('normal', 'repeat', 'black_images'))
        increase, noise = black - first, abs(repeat - first)
        passed = increase >= .1 * first and increase > 5 * noise
        checked_audit(out, recipe)
        write_json(out / 'visual.json', {'passed': passed, 'audit_sha256': sha256(out / 'audit.json'),
                   'selected_sha256': selected['package_sha256'], 'losses': measured, 'counts': counts,
                   'increase': increase, 'repeat_noise': noise, 'anchor_indices': ids,
                   'scene_matched_swap_positions': swap, 'unchanged_swap_positions': [i for i, j in enumerate(swap) if i == j],
                   'black_inputs': 'both images of the current pair; no diagnostic policy feature is a head input',
                   'history_effect': 'all historical diagnostic images are causally excluded from this head',
                   'minimum_relative_loss_increase': .1, 'minimum_repeat_noise_multiple': 5.,
                   'all_saved_state_unchanged': True, 'device': 'cpu', 'encoder_batch_size': 1})
        if not passed:
            raise ValueError('full-image visual dependence gate failed')


def runtime_run(out, recipe, device):
    from runtime_preflight import run_runtime_preflight

    audit = checked_audit(out, recipe)
    selected = selected_package(out)
    selected_gate(out, 'cpu_validate', selected)
    selected_gate(out, 'visual', selected)
    with data_owner(recipe, device) as data:
        check_data(data, audit)
    report = run_runtime_preflight(selected['package'], recipe, device)
    checked_audit(out, recipe)
    if report['checkpoint_sha256'] != selected['package_sha256']:
        raise ValueError('runtime package differs from selected package')
    write_json(out / 'runtime.json', {**report, 'audit_sha256': sha256(out / 'audit.json'),
                                     'selected_sha256': selected['package_sha256']})
    if report['passed'] is not True:
        raise ValueError('CPU runtime preflight failed')


STAGES = {'audit': audit_run, 'condition': condition_run, 'overfit': overfit_run, 'train': train_run,
          'postflight': postflight_run, 'cpu_validate': cpu_validate_run, 'visual': visual_run, 'runtime': runtime_run}
STAGE_PREREQUISITES = {
    'audit': (),
    'condition': ('audit',),
    'overfit': ('audit', 'condition'),
    'train': ('audit', 'condition', 'overfit'),
    'postflight': ('audit', 'condition', 'overfit'),
    'cpu_validate': ('audit', 'condition', 'overfit', 'train', 'postflight'),
    'visual': ('audit', 'condition', 'overfit', 'train', 'postflight', 'cpu_validate'),
    'runtime': ('audit', 'condition', 'overfit', 'train', 'postflight', 'cpu_validate', 'visual'),
}


def begin_stage(out, stage):
    if stage not in STAGES:
        raise ValueError('unsupported stage')
    if stage == 'audit':
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise ValueError('run audit first')
    reject_failed_attempt(out)
    reject_interrupted_attempt(out, stage)
    for predecessor in STAGE_PREREQUISITES[stage]:
        require_complete(out, predecessor)
    if stage == 'postflight':
        result = read_json(out / 'training_updates.json')
        require_training_parent(result)
        if (result.get('updates_complete') is not True or result.get('postflight_complete') is not False or
                read_json(out / 'train_attempt.json')['process_id'] != result['process_id']):
            raise ValueError('postflight requires its training owner after pool cleanup')
        check_prerequisites(out, recorded_prerequisites(result))
    elif stage == 'train':
        training_prerequisites(out)
    elif stage in ('cpu_validate', 'visual', 'runtime'):
        selected = selected_package(out)
        if stage != 'cpu_validate':
            selected_gate(out, 'cpu_validate', selected)
        if stage == 'runtime':
            selected_gate(out, 'visual', selected)
    attempt = out / (stage + '_attempt.json')
    if attempt.exists() or (out / (stage + '.json')).exists():
        raise FileExistsError('stage already attempted; use a new output directory')
    if stage == 'train' and ((out / 'steps.jsonl').exists() or (out / 'training_updates.json').exists()):
        raise FileExistsError('main fit already attempted; resuming is prohibited')
    prerequisites = {name + suffix: sha256(out / (name + suffix))
                     for name in STAGE_PREREQUISITES[stage] for suffix in ('.json', '_complete.json')}
    if stage == 'postflight':
        prerequisites.update({name: sha256(out / name) for name in ('train_attempt.json', 'training_updates.json')})
    write_json(attempt, {'stage': stage, 'process_id': os.getpid()})
    return prerequisites


def main():
    parser = argparse.ArgumentParser(description='Frozen current cone-set representation trial')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda', help='training and inherited recovery receipt replay device')
    args = parser.parse_args()
    prerequisites = begin_stage(args.out, args.stage)
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
        STAGES[args.stage](args.out, recipe, device)
        checked_audit(args.out, recipe, require_completion=args.stage != 'audit')
        if args.stage in ('train', 'postflight'):
            training_record = read_json(args.out / 'training_updates.json')
            check_prerequisites(args.out, recorded_prerequisites(training_record))
            if args.stage == 'postflight':
                require_training_parent(training_record)
        if args.stage == 'train':
            require_complete(args.out, 'postflight')
        elif args.stage == 'overfit':
            condition_sha = read_json(args.out / 'overfit.json')['condition_sha256']
            gate(args.out, 'condition')
            if sha256(args.out / 'condition.json') != condition_sha:
                raise ValueError('condition receipt changed before overfit completion')
        elif args.stage in ('cpu_validate', 'visual', 'runtime'):
            selected = selected_package(args.out)
            if read_json(args.out / (args.stage + '.json'))['selected_sha256'] != selected['package_sha256']:
                raise ValueError('selected package changed before stage completion')
            if args.stage != 'cpu_validate':
                selected_gate(args.out, 'cpu_validate', selected)
            if args.stage == 'runtime':
                selected_gate(args.out, 'visual', selected)
        if any(sha256(args.out / name) != expected for name, expected in prerequisites.items()):
            raise ValueError('stage prerequisite receipts or completion markers changed after admission')
        mark_complete(args.out, args.stage)
    except BaseException as exc:
        write_json(args.out / (args.stage + '_failure.json'),
                   {'stage': args.stage, 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
