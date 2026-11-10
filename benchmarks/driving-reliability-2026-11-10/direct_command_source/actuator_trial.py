"""Frozen native-model direct-command trial. Run from the repository root."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fsop.data.native_replay import NativeReplay, episode_paths, load_episodes
from fsop.eval.closed_loop import ModelPolicy, mission_brake_with_override
from fsop.eval.controller import validate_checkpoint_payload
from fsop.geometry.yuv import rgb_to_model
from fsop.model.build import SupercomboFS, load_fs
from fsop.train.lab_train import RecoverySampler, dataset_verification, fresh_history, prove_policy_context, sha256
from isaac_lab_fsop.evaluate import model_digest, source_inventory

HERE = Path(__file__).resolve().parent
RECIPE = HERE / 'recipe.json'
KIND = 'direct_native_actuator_v1'
CONTRACT = {
    'kind': KIND, 'version': 1, 'latent_module': 'base.policy.p_node_linear_20',
    'latent_shape': ['B', 256], 'latent_normalization': {'kind': 'layer_norm', 'eps': 1e-5, 'affine': False},
    'proprio_order': ['vx_mps', 'vy_mps', 'yaw_rate_radps'], 'proprio_scales': [10.0, 2.0, 1.0],
    'output_order': ['steer_rad', 'accel_mps2'], 'output_scales': [0.1, 2.0],
    'target': 'qualified native teacher controls at the observation tick',
    'history': 'phase-zero 5 Hz, eight past features and current feature, full desire queue',
    'mission': 'bound raw request, apply mission_brake_with_override, bound again',
    'base_parameters': 31154802, 'head_parameters': 33538,
}


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def digest_state(state):
    out = {}
    for name, tensor in state.items():
        t = tensor.detach().cpu().contiguous()
        out[name] = {'dtype': str(t.dtype), 'shape': list(t.shape),
                     'sha256': hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    return out


class CommandHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256, eps=1e-5, elementwise_affine=False)
        self.linear1 = nn.Linear(259, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, latent, proprio):
        if latent.ndim != 2 or latent.shape[1] != 256 or proprio.shape != (latent.shape[0], 3):
            raise ValueError('invalid command head input shapes')
        x = torch.cat((self.norm(latent), proprio / proprio.new_tensor(CONTRACT['proprio_scales'])), dim=1)
        return self.linear2(F.silu(self.linear1(x))) * x.new_tensor(CONTRACT['output_scales'])


class Capture:
    def __init__(self, model):
        self.values = []
        self.handle = model.base.policy.p_node_linear_20.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if len(args) != 1:
            raise ValueError('unexpected policy affine inputs')
        self.values.append(args[0])

    def take(self, batch):
        if len(self.values) != 1 or self.values[0].shape != (batch, 256):
            raise ValueError('policy hook must fire exactly once with shape [B,256]')
        return self.values[0]

    def close(self):
        self.handle.remove()


def model_from_payload(payload, device):
    if not isinstance(payload, dict):
        raise ValueError('base payload must be a mapping')
    config = payload.get('config') or {}
    validate_checkpoint_payload(payload, config.get('controller_mode', 'pp'))
    state = payload['model']
    model = SupercomboFS(image_head=any(k.startswith('img.') for k in state))
    model.load_state_dict(state, strict=True)
    model.controller_contract = payload.get('controller_contract')
    model.requires_grad_(False).eval().to(device)
    check_base(model)
    return model


def check_base(model):
    if sum(p.numel() for p in model.parameters()) != 31154802 or len(model.state_dict()) != 417:
        raise ValueError('unexpected base parameter or tensor inventory')
    if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
        raise ValueError('base must remain fully frozen in eval mode')


class NativeActuatorPolicy:
    controller_name = KIND

    def __init__(self, model, head, contract, device):
        self.model, self.head, self.command_contract, self.device = model, head.requires_grad_(False).eval(), contract, device
        mode = (getattr(model, 'controller_contract', None) or {}).get('mode', 'pp')
        self.base_policy = ModelPolicy(model, controller=mode, device=device)

    @classmethod
    def load(cls, package: Path, device: str | None = None):
        package = torch.load(package, map_location='cpu', weights_only=False)
        if not isinstance(package, dict) or package.get('kind') != KIND or package.get('command_contract') != CONTRACT:
            raise ValueError('not a compatible direct actuator package')
        if set(package) != {'kind', 'base_payload', 'head_state', 'command_contract', 'recipe', 'provenance'}:
            raise ValueError('invalid direct actuator package inventory')
        device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        model = model_from_payload(package['base_payload'], device)
        head = CommandHead().to(device)
        head.load_state_dict(package['head_state'], strict=True)
        if digest_state(head.state_dict()) != package['provenance']['head_state_digest']:
            raise ValueError('command head state digest mismatch')
        if model_digest(model) != package['provenance']['base_model_digest']:
            raise ValueError('base model state digest mismatch')
        if sum(p.numel() for p in head.parameters()) != 33538:
            raise ValueError('unexpected command head parameter inventory')
        return cls(model, head, package['command_contract'], device)

    def reset(self):
        self.base_policy.reset()

    @torch.no_grad()
    def __call__(self, obs, env):
        capture = Capture(self.model)
        try:
            _, _, info = self.base_policy(obs, env)
            latent = capture.take(1)
            proprio = torch.tensor([[obs.state.vx, obs.state.vy, obs.state.r]], device=self.device, dtype=torch.float32)
            raw = self.head(latent, proprio)[0].cpu().numpy()
            if not np.isfinite(raw).all():
                raise ValueError('nonfinite direct command')
            bounded = np.array([np.clip(raw[0], -env.vp.max_steer, env.vp.max_steer),
                                np.clip(raw[1], -env.vp.max_brake, env.vp.max_accel)])
            accel, override = mission_brake_with_override(float(bounded[1]), obs, env)
            applied = [float(bounded[0]), float(np.clip(accel, -env.vp.max_brake, env.vp.max_accel))]
            info['base_controller_diagnostic'] = info.pop('controller', None)
            info['controller'] = {'kind': KIND, 'latent': latent[0].cpu().numpy().tolist(),
                                  'proprio': proprio[0].cpu().numpy().tolist(), 'raw': raw.tolist(),
                                  'bounded': bounded.tolist(), 'applied': applied, 'mission_override': override}
            return applied[0], applied[1], info
        finally:
            capture.close()


def inventory(recipe):
    paths = [RECIPE, Path(__file__), Path(recipe['recovery_recipe']),
             Path('artifacts/models/prep/vision_fp32.onnx'), Path('artifacts/models/prep/policy_fp32.onnx')]
    return {**source_inventory(), **{str(p): sha256(p) for p in paths}}


def datasets(recipe):
    if sha256(recipe['manifest']) != recipe['manifest_sha256'] or sha256(recipe['base_checkpoint']) != recipe['base_checkpoint_sha256']:
        raise ValueError('manifest or original checkpoint hash mismatch')
    verification = dataset_verification(recipe['manifest'])
    episodes = load_episodes(episode_paths(manifest=recipe['manifest']))
    train, val = NativeReplay(episodes, 'train'), NativeReplay(episodes, 'val')
    if (len(train), len(val)) != (7006, 2067):
        raise ValueError('unexpected qualified train/validation anchor counts')
    sampler = RecoverySampler(train, recipe['recovery_recipe'], recipe['recovery_gates'], 128, 16,
                              recipe['base_checkpoint_sha256'], provenance_mode='per_case_v2')
    return train, val, sampler, verification


def batch_input(model, dataset, ids, device, hook=True, blank=False):
    pairs = [dataset.index[int(i)] for i in ids]
    eps, frames = zip(*pairs)
    rows = []
    for ei, frame in pairs:
        ep = dataset.episodes[ei]
        tick = int(ep.ticks[frame])
        if not ep.teacher[tick]:
            raise ValueError('same-tick command is not a teacher control')
        rows.append((ep, frame, tick))
    blank_frame = rgb_to_model(np.zeros((256, 512, 3), dtype=np.uint8)) if blank else None
    blank_pair = np.concatenate((blank_frame, blank_frame)) if blank else None
    class BlankEpisode:
        def frame_pair(self, camera, frame):
            return blank_pair
    history_episodes = [BlankEpisode() for _ in dataset.episodes] if blank else dataset.episodes
    with torch.inference_mode():
        history = fresh_history(model, history_episodes, eps, frames, device, chunk_size=64,
                                amp=False, channels_last=False, history_context=8)
    road = torch.from_numpy(np.stack([blank_pair if blank else ep.frame_pair('road', k) for ep, k, _ in rows])).to(device).float()
    wide = torch.from_numpy(np.stack([blank_pair if blank else ep.frame_pair('wide', k) for ep, k, _ in rows])).to(device).float()
    desire = torch.from_numpy(np.stack([np.array(ep.desires[k], copy=True) for ep, k, _ in rows])).to(device).float()
    traffic = torch.tensor([[1., 0.]], device=device).repeat(len(rows), 1)
    prop = torch.tensor(np.stack([ep.states[tick, 3:6] for ep, _, tick in rows]).copy(), device=device).float()
    target = torch.tensor(np.stack([ep.controls[tick] for ep, _, tick in rows]).copy(), device=device).float()
    capture = Capture(model) if hook else None
    try:
        with torch.inference_mode():
            hidden = model.base.encode(road, wide).hidden
            output = model.base.policy_raw(torch.cat((history, hidden[:, None]), 1), desire, traffic)
            latent = capture.take(len(rows)) if capture else None
        # An inference tensor cannot be saved for the head's backward pass.
        latent = latent.clone() if capture else None
        return latent, prop, target, tuple(x.clone() for x in output)
    finally:
        if capture:
            capture.close()


def frozen(model, reference, flags):
    check_base(model)
    if digest_state(model.state_dict()) != reference or {n: p.requires_grad for n, p in model.named_parameters()} != flags:
        raise ValueError('base tensors or gradient flags changed')


def scores(errors):
    a = np.asarray(errors)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite validation command errors')
    if not len(a):
        raise ValueError('empty metric group')
    mae, p95 = a.mean(0), np.percentile(a, 95, axis=0)
    return {'count': len(a), 'mae': mae.tolist(), 'p95': p95.tolist(),
            'passed': bool(np.all(mae <= [0.01, 0.25]) and np.all(p95 <= [0.03, 0.75]))}


def validate(model, head, val, device):
    groups = {'all': []}
    with torch.no_grad():
        for start in range(0, len(val), 64):
            ids = range(start, min(start + 64, len(val)))
            latent, prop, target, _ = batch_input(model, val, ids, device)
            error = (head(latent, prop) - target).abs().cpu().numpy()
            for j, i in enumerate(ids):
                ei, frame = val.index[i]
                ep = val.episodes[ei]
                keys = ['all', 'episode/' + str(ep.meta['id'])]
                if prop[j, 0].item() < 2:
                    keys.append('stratum/low_vx')
                if np.any(ep.desires[frame, -9:] != 0):
                    keys.append('stratum/recent_pulse')
                if ep.meta.get('collection_policy', {}).get('mode') in ('learner_prefix', 'phase_recovery_v1'):
                    keys.append('stratum/recovery')
                for key in keys:
                    groups.setdefault(key, []).append(error[j])
    result = {key: scores(value) for key, value in sorted(groups.items())}
    return {'groups': result, 'passed': all(x['passed'] for x in result.values()),
            'selection_score': max(result['all']['mae'][0] / .01, result['all']['mae'][1] / .25)}


def telemetry():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,power.draw,memory.used',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def loss(head, latent, prop, target):
    scales = target.new_tensor([.02, .5])
    return F.smooth_l1_loss((head(latent, prop) - target) / scales, torch.zeros_like(target))


def audit(out, recipe, device):
    train, val, sampler, verification = datasets(recipe)
    model = load_fs(recipe['base_checkpoint']).requires_grad_(False).eval().to(device)
    check_base(model)
    context = prove_policy_context(model)
    ids = np.random.default_rng(0).choice(len(train), size=32, replace=False).tolist()
    dynamic = [i for i, (ei, frame) in enumerate(train.index) if frame >= 8 and
               train.episodes[ei].states[int(train.episodes[ei].ticks[frame]), 3] >= 2.0]
    if len(dynamic) < 32:
        raise ValueError('fewer than 32 dynamic visual-dependence anchors')
    visual_ids = np.random.default_rng(0).choice(dynamic, size=32, replace=False).tolist()
    a = batch_input(model, train, ids[:4], device, hook=False)[3]
    b = batch_input(model, train, ids[:4], device, hook=True)[3]
    c = batch_input(model, train, ids[:4], device, hook=False)[3]
    d = batch_input(model, train, ids[:4], device, hook=True)[3]
    if not all(torch.equal(x, y) for output in (b, c, d) for x, y in zip(a, output, strict=True)):
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


def preflight(out, recipe, device):
    audit = json.loads((out / 'audit.json').read_text())
    if audit['recipe'] != recipe or audit['recipe_sha256'] != sha256(RECIPE) or audit['sources'] != inventory(recipe):
        raise ValueError('recipe or source changed after audit')
    train, val, sampler, verification = datasets(recipe)
    if audit['verification'] != verification or audit['sampler'] != sampler.protocol():
        raise ValueError('dataset verification or sampler changed after audit')
    model = load_fs(recipe['base_checkpoint']).requires_grad_(False).eval().to(device)
    frozen(model, audit['base_state'], audit['base_flags'])
    return audit, model, train, val, sampler


def new_head(device):
    torch.manual_seed(0)
    head = CommandHead().to(device)
    if sum(p.numel() for p in head.parameters()) != 33538:
        raise ValueError('head parameter count mismatch')
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0.)
    return head, optimizer


def overfit(out, recipe, device):
    audit, model, train, _, _ = preflight(out, recipe, device)
    head, optimizer = new_head(device)
    ids = audit['overfit_indices']
    started = time.monotonic()
    for step in range(201):
        latent, prop, target, _ = batch_input(model, train, ids, device)
        with torch.no_grad():
            mae = (head(latent, prop) - target).abs().mean(0).cpu().tolist()
        if mae[0] <= .002 and mae[1] <= .05:
            frozen(model, audit['base_state'], audit['base_flags'])
            write_json(out / 'overfit.json', {'passed': True, 'step': step, 'mae': mae,
                       'seconds': time.monotonic() - started, 'audit_sha256': sha256(out / 'audit.json'),
                       'discarded_head': True, 'base_model_digest_after': model_digest(model)})
            return
        if step == 200:
            break
        optimizer.zero_grad(set_to_none=True)
        value = loss(head, latent, prop, target)
        value.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
    torch.save({'head_state': head.state_dict(), 'step': 200, 'audit_sha256': sha256(out / 'audit.json')},
               out / 'overfit_failed_head.pt')
    write_json(out / 'overfit.json', {'passed': False, 'step': 200, 'mae': mae,
               'seconds': time.monotonic() - started, 'audit_sha256': sha256(out / 'audit.json'), 'discarded_head': True})
    raise ValueError('fixed 32-anchor wiring gate failed; main run prohibited')


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
        if sum(int(i) in sampler.recovery_indices for i in ids) != 16:
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
    if verification_after != audit['verification'] or sampler_after.protocol() != audit['sampler'] or inventory(recipe) != audit['sources']:
        raise ValueError('source or data changed during training')
    write_json(out / 'train.json', {'passed': bool(selected), 'selected': selected, 'reports': reports,
               'diagnostic_only_step': None if selected else 500, 'seconds': time.monotonic() - started,
               'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               'optimizer_parameters': named, 'base_model_digest_after': model_digest(model),
               'base_state_unchanged': True, 'base_flags_unchanged': True, 'draw_counts': sampler.draw_counts()})


def visual_gate(out, recipe, device):
    audit_record, model, train, _, _ = preflight(out, recipe, device)
    result = json.loads((out / 'train.json').read_text())
    selected = result['selected']
    if not result['passed'] or selected is None:
        raise ValueError('visual gate requires an eligible selected checkpoint')
    path = Path(selected['package'])
    if sha256(path) != selected['package_sha256']:
        raise ValueError('selected package hash changed')
    package = torch.load(path, map_location='cpu', weights_only=False)
    if package.get('kind') != KIND or package.get('command_contract') != CONTRACT:
        raise ValueError('selected package contract mismatch')
    head = CommandHead().to(device).eval()
    head.load_state_dict(package['head_state'], strict=True)
    ids = audit_record['visual_indices']
    measured = []
    with torch.no_grad():
        for blank in (False, False, True):
            latent, prop, target, _ = batch_input(model, train, ids, device, blank=blank)
            measured.append(float(loss(head, latent, prop, target)))
    first, repeat, blanked = measured
    increase = blanked - first
    noise = abs(repeat - first)
    passed = increase >= .1 * first and increase > 5 * noise
    frozen(model, audit_record['base_state'], audit_record['base_flags'])
    write_json(out / 'visual.json', {'passed': passed, 'selected_sha256': selected['package_sha256'],
               'normal_loss': first, 'repeat_loss': repeat, 'blank_history_loss': blanked,
               'increase': increase, 'repeat_noise': noise, 'anchor_indices': ids,
               'blank': recipe['visual_gate']['blank'], 'base_model_digest_after': model_digest(model)})
    if not passed:
        raise ValueError('full-history visual-dependence gate failed')


def main():
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
    if receipt.exists() or (args.out / {'audit': 'audit.json', 'overfit': 'overfit.json', 'train': 'train.json', 'visual': 'visual.json'}[args.stage]).exists():
        raise FileExistsError('stage already attempted; use a new output directory')
    try:
        if args.stage == 'train' and (args.out / 'steps.jsonl').exists():
            raise FileExistsError('training already attempted')
        recipe = json.loads(RECIPE.read_text())
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        {'audit': audit, 'overfit': overfit, 'train': train_run, 'visual': visual_gate}[args.stage](args.out, recipe, device)
    except BaseException as exc:
        write_json(receipt, {'stage': args.stage, 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
