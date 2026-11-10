from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'cone_set_actuator_trial'
sys.path.insert(0, str(OLD))
import cone_core as core
import train_set as frozen


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_freeze(expected):
    if digest(HERE / 'freeze.json') != expected:
        raise ValueError('probe freeze differs from the parent-approved hash')
    freeze = read(HERE / 'freeze.json')
    for name, sha in freeze['files'].items():
        if digest(HERE / name) != sha:
            raise ValueError('probe source or copied receipt changed: ' + name)
    return freeze


def verify_failed(bindings):
    run = Path(bindings['failed_run'])
    for name, sha in bindings['failed_receipts'].items():
        if digest(run / name) != sha:
            raise ValueError('failed receipt changed: ' + name)
    audit = read(run / 'audit.json')
    condition = read(run / 'condition.json')
    overfit = read(run / 'overfit.json')
    failure = read(run / 'overfit_failure.json')
    recipe = read(frozen.RECIPE)
    frozen.check_recipe(recipe)
    if (audit['recipe'] != recipe or audit['recipe_sha256'] != digest(frozen.RECIPE)
            or audit['sources'] != frozen.inventory(recipe)
            or audit['collision_sha256'] != digest(run / 'collisions.json')
            or audit['inherited_audit_sha256'] != digest(recipe['inherited_audit'])
            or audit['source_archive_sha256'] != bindings['audit_source_archive_sha256']):
        raise ValueError('failed audit source, recipe, or archive binding changed')
    frozen.check_archive(run, audit)
    frozen.require_complete(run, 'audit')
    frozen.require_complete(run, 'condition')
    if (condition['passed'] is not True or overfit['passed'] is not False
            or overfit['step'] != 200 or failure['stage'] != 'overfit'
            or condition['audit_sha256'] != digest(run / 'audit.json')
            or overfit['audit_sha256'] != digest(run / 'audit.json')
            or overfit['condition_sha256'] != digest(run / 'condition.json')
            or failure['error'] != 'fixed 32-anchor wiring gate failed; main run prohibited'
            or not overfit['discarded_all_weights'] or not overfit['frozen_base_state_unchanged']):
        raise ValueError('expected failed attempt identity differs')
    if any((run / name).exists() for name in ('train_attempt.json', 'train.json', 'overfit_complete.json')):
        raise ValueError('failed run unexpectedly advanced')
    prelaunch = OLD / 'prelaunch_freeze1.json'
    if digest(prelaunch) != bindings['prelaunch_freeze_sha256']:
        raise ValueError('prelaunch freeze changed')
    source_freeze = read(prelaunch)
    if (source_freeze['sources'] != audit['sources']
            or source_freeze['source_archive_sha256'] != bindings['prelaunch_source_archive_sha256']):
        raise ValueError('prelaunch source archive differs')
    if (audit['overfit_indices'] != bindings['indices']
            or audit['conditioning_indices'] != bindings['indices']
            or audit['overfit_anchors'] != [row['anchor'] for row in bindings['anchors']]):
        raise ValueError('fixed TRAIN anchors changed')
    return audit, recipe


def features(head, tokens, mask, desire, proprio, normalized):
    encoded = F.silu(head.token2(F.silu(head.token1(tokens.masked_fill(~mask[..., None], 0)))))
    count = mask.sum(1, keepdim=True)
    mean = encoded.masked_fill(~mask[..., None], 0).sum(1) / count.clamp_min(1)
    maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(1)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    raw = torch.cat((mean, maximum), 1)
    pooled = F.layer_norm(raw, (128,), weight=None, bias=None, eps=1e-5) if normalized else raw
    combined = torch.cat((pooled, count.to(tokens.dtype) / 200., desire.flatten(1),
                          proprio / proprio.new_tensor([10., 2., 1.])), 1)
    hidden = F.silu(head.linear1(combined))
    output = head.linear2(hidden) * hidden.new_tensor([.1, 2.])
    return raw, pooled, combined, hidden, output


def spectrum(value):
    matrix = value.detach().double().cpu()
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError('invalid diagnostic matrix')
    singular = torch.linalg.svdvals(matrix)
    largest = float(singular[0]) if len(singular) else 0.
    tolerance = max(matrix.shape) * np.finfo(np.float32).eps * largest
    rank = int((singular > tolerance).sum())
    smallest = float(singular[-1]) if len(singular) else 0.
    return {'shape': list(matrix.shape), 'singular_values': singular.tolist(),
            'rank_fp32': rank, 'rank_tolerance': tolerance,
            'condition_full': largest / smallest if smallest > 0 else None,
            'condition_resolved': largest / float(singular[rank - 1]) if rank else None,
            'stable_rank': float(singular.square().sum()) / largest ** 2 if largest else 0.}


def spread(value):
    matrix = value.detach().double().cpu()
    center = matrix.mean(0)
    variation = matrix - center
    offset = float(center.square().mean().sqrt())
    rms = float(variation.square().mean().sqrt())
    return {'common_offset_rms': offset, 'across_anchor_rms': rms,
            'variation_to_common_offset': rms / offset if offset else None,
            'per_feature_mean': center.tolist(),
            'per_feature_std': matrix.std(0, correction=0).tolist(),
            'row_mean': matrix.mean(1).tolist(), 'row_std': matrix.std(1, correction=0).tolist(),
            'uncentered_spectrum': spectrum(matrix), 'centered_spectrum': spectrum(variation)}


def coherent_tokens(tokens, delta):
    lateral, yaw = delta[:, :1], delta[:, 1:2]
    x = tokens[..., 0]
    y = tokens[..., 1] - lateral / 45.
    shifted = torch.stack((x, y), -1)
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    xy = torch.stack((cosine * x + sine * y, -sine * x + cosine * y), -1)
    radius = tokens[..., 2:3] + (torch.linalg.vector_norm(shifted, dim=-1, keepdim=True)
                                - torch.linalg.vector_norm(tokens[..., :2], dim=-1, keepdim=True))
    return torch.cat((xy, radius, tokens[..., 3:]), -1)


def pairwise_report(raw, pooled, combined, hidden, batch, bindings):
    matrices = {'raw_pooled': raw, 'pooled': pooled, 'combined': combined, 'hidden': hidden}
    arrays = {k: v.detach().double().cpu().numpy() for k, v in matrices.items()}
    prop = batch.proprio.detach().double().cpu().numpy() / np.array([10., 2., 1.])
    route = batch.desire.detach().double().cpu().numpy().reshape(32, -1)
    target = batch.target.detach().double().cpu().numpy()
    counts = batch.mask.sum(1).cpu().numpy()
    rows = []
    for i in range(32):
        for j in range(i + 1, 32):
            distances = {k + '_rms': float(np.sqrt(np.mean((v[i] - v[j]) ** 2)))
                         for k, v in arrays.items()}
            rows.append({'offsets': [i, j], 'indices': [bindings['indices'][i], bindings['indices'][j]],
                         **distances, 'route_max': float(np.max(np.abs(route[i] - route[j]))),
                         'scaled_motion_rms': float(np.sqrt(np.mean((prop[i] - prop[j]) ** 2))),
                         'cone_count_difference': int(abs(counts[i] - counts[j])),
                         'target_absolute_difference': np.abs(target[i] - target[j]).tolist()})
    return rows


def variant(head, batch, normalized, bindings):
    tokens = batch.tokens.detach().clone().requires_grad_(True)
    proprio = batch.proprio.detach().clone().requires_grad_(True)
    raw, pooled, combined, hidden, output = features(head, tokens, batch.mask, batch.desire, proprio, normalized)
    error = (output - batch.target) / output.new_tensor([.02, .5])
    loss = F.smooth_l1_loss(error, torch.zeros_like(error))
    named = dict(head.named_parameters())
    gradients = torch.autograd.grad(loss, [tokens, proprio, *named.values()])
    grad_norms = {name: float(g.norm()) for name, g in zip(named, gradients[2:], strict=True)}
    coordinate_grads = gradients[0][..., :3].masked_fill(~batch.mask[..., None], 0)
    bias_norm = grad_norms['linear2.bias']
    loss_report = {'loss': float(loss.detach()), 'parameter_gradient_norms': grad_norms,
                   'token_xyz_gradient_norm': float(coordinate_grads.norm()),
                   'token_xyz_gradient_to_output_bias': float(coordinate_grads.norm()) / bias_norm,
                   'motion_gradient_norm': float(gradients[1].norm()),
                   'loss_motion_gradient_per_anchor': gradients[1].detach().cpu().tolist(),
                   'token1_weight_gradient_to_output_bias': grad_norms['token1.weight'] / bias_norm}
    delta = batch.tokens.new_zeros((32, 2), requires_grad=True)
    motion = batch.proprio.detach().clone().requires_grad_(True)
    physical = coherent_tokens(batch.tokens, delta)
    _, _, _, _, commands = features(head, physical, batch.mask, batch.desire, motion, normalized)
    geometry, motion_rows = [], []
    for channel in range(2):
        geo, kin = torch.autograd.grad(commands[:, channel].sum(), (delta, motion), retain_graph=channel == 0)
        geometry.append(geo)
        motion_rows.append(kin)
    geometry_batch = torch.stack(geometry, 1)
    motion_batch = torch.stack(motion_rows, 1)
    jacobians = []
    for i in range(32):
        token = batch.tokens[i:i + 1]
        mask = batch.mask[i:i + 1]
        route = batch.desire[i:i + 1]
        prop = batch.proprio[i:i + 1]
        geometry_jac = geometry_batch[i]
        motion_jac = motion_batch[i]
        local_delta = token.new_zeros((1, 2), requires_grad=True)
        def hidden_from_xy(change):
            return features(head, coherent_tokens(token, change), mask, route, prop, normalized)[3][0]
        hidden_jac = torch.autograd.functional.jacobian(hidden_from_xy, local_delta, vectorize=True).reshape(128, 2)
        def hidden_from_motion(body_motion):
            return features(head, token, mask, route, body_motion, normalized)[3][0]
        motion_hidden_jac = torch.autograd.functional.jacobian(hidden_from_motion, prop, vectorize=True).reshape(128, 3)
        jacobians.append({'index': bindings['indices'][i], 'cone_count': int(mask.sum()),
                          'geometry_axes': ['ego_lateral_shift_m', 'ego_yaw_rotation_rad'],
                          'geometry_output_rows': geometry_jac.detach().cpu().tolist(),
                          'geometry_output_row_norm': geometry_jac.norm(dim=1).detach().cpu().tolist(),
                          'geometry_to_output_bias_row_norm': (geometry_jac.norm(dim=1) / geometry_jac.new_tensor([.1, 2.])).detach().cpu().tolist(),
                          'motion_output_rows': motion_jac.detach().cpu().tolist(),
                          'motion_output_row_norm': motion_jac.norm(dim=1).detach().cpu().tolist(),
                          'geometry_output_spectrum': spectrum(geometry_jac),
                          'motion_output_spectrum': spectrum(motion_jac),
                          'motion_hidden_spectrum': spectrum(motion_hidden_jac),
                          'geometry_hidden_spectrum': spectrum(hidden_jac)})
    design = torch.cat((hidden.detach(), torch.ones_like(hidden[:, :1])), 1)
    augmented_pooled = torch.cat((pooled.detach(), torch.ones_like(pooled[:, :1])), 1)
    return {'normalized': normalized, 'parameter_count': sum(p.numel() for p in head.parameters()),
            'output': output.detach().cpu().tolist(), 'absolute_error': (output - batch.target).abs().detach().cpu().tolist(),
            'raw_pooled': spread(raw), 'pooled': spread(pooled), 'combined': spread(combined),
            'hidden': spread(hidden), 'readout_design': spectrum(design),
            'pooled_with_bias_design': spectrum(augmented_pooled),
            'loss_gradients': loss_report, 'jacobians': jacobians,
            'pairs': pairwise_report(raw, pooled, combined, hidden, batch, bindings)}


def decision(original, normalized, thresholds):
    pairs = list(zip(original['jacobians'], normalized['jacobians'], strict=True))
    geometry = np.array([np.array(b['geometry_output_row_norm']) / np.maximum(a['geometry_output_row_norm'], 1e-30)
                         for a, b in pairs])
    motion = np.array([np.array(b['motion_output_row_norm']) / np.maximum(a['motion_output_row_norm'], 1e-30)
                       for a, b in pairs])
    geometry_median = np.median(geometry, axis=0)
    motion_median = np.median(motion, axis=0)
    motion_minimum = np.min(motion, axis=0)
    rank_retained = all(b['geometry_hidden_spectrum']['rank_fp32'] >= a['geometry_hidden_spectrum']['rank_fp32']
                        and b['geometry_output_spectrum']['rank_fp32'] >= a['geometry_output_spectrum']['rank_fp32']
                        and b['motion_output_spectrum']['rank_fp32'] >= a['motion_output_spectrum']['rank_fp32']
                        and b['motion_hidden_spectrum']['rank_fp32'] >= a['motion_hidden_spectrum']['rank_fp32']
                        for a, b in pairs)
    before, after = original['readout_design'], normalized['readout_design']
    cond_gain = (before['condition_full'] / after['condition_full']
                 if before['condition_full'] is not None and after['condition_full'] is not None else 0.)
    readout_rank = after['rank_fp32'] >= before['rank_fp32']
    collapsed = []
    for a, b in zip(original['pairs'], normalized['pairs'], strict=True):
        for name in ('pooled', 'combined', 'hidden'):
            rms = max(normalized[name]['across_anchor_rms'], 1e-12)
            old_rms = max(original[name]['across_anchor_rms'], 1e-12)
            if (a[name + '_rms'] > thresholds['pair_collapse_relative'] * old_rms
                    and b[name + '_rms'] <= thresholds['pair_collapse_relative'] * rms):
                collapsed.append({'indices': a['indices'], 'space': name,
                                  'target_absolute_difference': a['target_absolute_difference']})
    material_geometry_gain = bool(np.all(geometry_median >= thresholds['geometry_gain_min']))
    material_condition_gain = cond_gain >= thresholds['readout_condition_gain_min'] and readout_rank
    motion_retained = bool(np.all(motion_median >= thresholds['motion_median_retention_min'])
                           and np.all(motion_minimum >= thresholds['motion_anchor_retention_min']))
    geometry_retained = bool(np.all(np.min(geometry, axis=0) >= thresholds['geometry_anchor_retention_min']))
    eligible = ((material_geometry_gain or material_condition_gain) and rank_retained and readout_rank
                and motion_retained and geometry_retained and not collapsed)
    return {'eligible_for_a_separate_versioned_LN_wiring_trial': bool(eligible),
            'not_evidence_of_learnability_generalization_or_driving': True,
            'geometry_output_row_gain_per_anchor': geometry.tolist(),
            'geometry_output_row_median_gain': geometry_median.tolist(),
            'motion_output_row_gain_per_anchor': motion.tolist(),
            'motion_output_row_median_gain': motion_median.tolist(),
            'motion_output_row_minimum_gain': motion_minimum.tolist(),
            'readout_condition_gain': cond_gain,
            'material_geometry_gain': material_geometry_gain,
            'material_readout_condition_gain': bool(material_condition_gain),
            'all_local_geometry_and_motion_ranks_retained': rank_retained,
            'readout_rank_retained': readout_rank, 'motion_retained': motion_retained,
            'geometry_retained': geometry_retained, 'new_collapsed_pairs': collapsed,
            'limitation': 'LN removes per-anchor feature mean and nearly removes scale. Local ranks and 496 pair checks cannot prove global information preservation.'}


def run(args, freeze, bindings):
    audit, recipe = verify_failed(bindings)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError('the original CUDA audit device is required')
    device = torch.device('cuda')
    if audit['device'] != 'cuda':
        raise ValueError('failed run device differs')
    payload, model = frozen.original_model(recipe, device)
    core.check_frozen_base(model, audit['base_state'])
    torch.manual_seed(0)
    head = core.ConeSetHead().to(device).eval()
    initial = core.digest_state(head.state_dict())
    if initial != audit['head_initial_state'] or sum(p.numel() for p in head.parameters()) != 47618:
        raise ValueError('seed-zero head differs')
    with frozen.data_owner(recipe, device) as data:
        frozen.check_data(data, audit)
        if (frozen.wiring_indices(data.sampler) != bindings['indices']
                or frozen.anchors(data.train, bindings['indices']) != audit['overfit_anchors']):
            raise ValueError('fresh ordered TRAIN anchors differ')
        batch = frozen.batch_input(model, data.train, bindings['indices'], device)
        hashes = [frozen.input_hash(batch, i) for i in range(32)]
        if hashes != [a['input_sha256'] for a in bindings['anchors']]:
            raise ValueError('fresh input hashes differ from failed-run collision receipt')
        expected_targets = np.asarray([a['target'] for a in bindings['anchors']], np.float32)
        if not np.array_equal(batch.target.detach().cpu().numpy(), expected_targets):
            raise ValueError('fresh same-tick teacher targets differ')
        repeat = frozen.batch_input(model, data.train, bindings['indices'], device)
        if [frozen.input_hash(repeat, i) for i in range(32)] != hashes:
            raise ValueError('fresh repeated inputs differ; probe has no noise floor')
        with torch.no_grad():
            original_output = head(*batch.inputs)
            rebuilt_output = features(head, *batch.inputs, False)[-1]
            repeat_output = features(head, *repeat.inputs, False)[-1]
            ln_first = features(head, *batch.inputs, True)[-1]
            ln_repeat = features(head, *repeat.inputs, True)[-1]
        if not torch.equal(original_output, rebuilt_output) or not torch.equal(repeat_output, rebuilt_output):
            raise ValueError('frozen head instrumentation or baseline repeat differs')
        if not torch.equal(ln_first, ln_repeat):
            raise ValueError('LN repeat differs')
        old_loss = float(frozen.command_loss(head, batch).detach())
        if old_loss != read(Path(bindings['failed_run']) / 'condition.json')['normal_loss']:
            raise ValueError('initial loss differs from failed trial')
        write(args.out / 'input_binding.json', {'freeze_sha256': args.freeze_sha256,
              'failed_audit_sha256': digest(Path(bindings['failed_run']) / 'audit.json'),
              'head_initial_state': initial, 'base_state': audit['base_state'],
              'verification': data.verification, 'sampler': data.sampler.protocol(),
              'anchors': bindings['anchors'], 'fresh_input_sha256': hashes,
              'repeat_inputs_exact': True, 'instrumented_baseline_exact': True,
              'both_variant_output_repeats_exact': True, 'initial_loss': old_loss,
              'no_parameter_update': True})
        original = variant(head, batch, False, bindings)
        normalized = variant(head, batch, True, bindings)
        if core.digest_state(head.state_dict()) != initial:
            raise ValueError('probe changed head parameters')
        core.check_frozen_base(model, audit['base_state'])
    del data, model, payload, batch, repeat, head
    torch.cuda.empty_cache()
    with frozen.data_owner(recipe, device) as fresh_data:
        frozen.check_data(fresh_data, audit)
    verify_failed(bindings)
    verify_freeze(args.freeze_sha256)
    report = {'kind': 'cone_set_conditioning_probe_v1', 'freeze_sha256': args.freeze_sha256,
              'input_binding_sha256': digest(args.out / 'input_binding.json'),
              'no_parameter_update': True, 'no_analytic_fit': True, 'weights_exported': False,
              'frozen_base_tensors_verified_unchanged': 417, 'head_state_tensors_verified_unchanged': 8,
              'fresh_end_dataset_verification': True, 'final_suite_opened': False,
              'original': original, 'pooled_layernorm128': normalized,
              'decision': decision(original, normalized, bindings['thresholds'])}
    write(args.out / 'report.json', report)
    write(args.out / 'complete.json', {'report_sha256': digest(args.out / 'report.json'),
          'input_binding_sha256': digest(args.out / 'input_binding.json'),
          'attempt_sha256': digest(args.out / 'attempt.json'), 'freeze_sha256': args.freeze_sha256})
    print(json.dumps(report['decision'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Frozen TRAIN-only zero-update conditioning diagnosis')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze_sha256)
    bindings = read(HERE / 'bindings.json')
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'attempt.json', {'process_id': os.getpid(), 'freeze_sha256': args.freeze_sha256,
          'failed_receipts': bindings['failed_receipts'], 'mode': 'zero-update TRAIN diagnostic'})
    try:
        run(args, freeze, bindings)
    except BaseException as exc:
        write(args.out / 'failure.json', {'error': str(exc), 'traceback': traceback.format_exc(),
              'freeze_sha256': args.freeze_sha256, 'no_resume': True})
        raise


if __name__ == '__main__':
    main()
