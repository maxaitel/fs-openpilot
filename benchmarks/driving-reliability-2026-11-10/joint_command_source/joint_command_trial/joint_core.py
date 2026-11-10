from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fsop.eval.closed_loop import ModelPolicy, mission_brake_with_override
from fsop.eval.controller import validate_checkpoint_payload
from fsop.model.build import SupercomboFS
from isaac_lab_fsop.evaluate import model_digest

KIND = 'direct_native_actuator_joint_suffix_v1'
SCOPE = {
    'base.policy.p_node_linear_14.weight': [256, 512],
    'base.policy.p_node_linear_14.bias': [256],
    'base.policy.p_node_linear_16.weight': [256, 256],
    'base.policy.p_node_linear_16.bias': [256],
    'base.policy.p_node_linear_17.weight': [256, 256],
    'base.policy.p_node_linear_17.bias': [256],
}
HEAD_SCOPE = {'linear1.weight': [128, 259], 'linear1.bias': [128],
              'linear2.weight': [2, 128], 'linear2.bias': [2]}
CONTRACT = {
    'kind': KIND, 'version': 1, 'latent_module': 'base.policy.p_node_linear_20',
    'latent_shape': ['B', 256],
    'latent_normalization': {'kind': 'layer_norm', 'eps': 1e-5, 'affine': False},
    'proprio_order': ['vx_mps', 'vy_mps', 'yaw_rate_radps'], 'proprio_scales': [10.0, 2.0, 1.0],
    'output_order': ['steer_rad', 'accel_mps2'], 'output_scales': [0.1, 2.0],
    'target': 'qualified native teacher controls at the observation tick',
    'history': 'phase-zero 5 Hz, eight past features and current feature, full desire queue',
    'mission': 'bound raw request, apply mission_brake_with_override, bound again',
    'base_parameters': 31154802, 'head_parameters': 33538,
    'training_scope': SCOPE, 'suffix_parameters': 262912, 'optimized_parameters': 296450,
    'frozen_base_state_tensors': 411,
    'plan_semantics': 'updated plan and inherited controller are diagnostics only',
    'deployment': {'device': 'cpu', 'dtype': 'torch.float32', 'threads': 4},
}
PAYLOAD_KEYS = {'step', 'model', 'config', 'controller_contract', 'controller_intervention'}
PACKAGE_KEYS = {'kind', 'base_payload', 'head_state', 'command_contract', 'recipe', 'provenance'}


def digest_state(state):
    result = {}
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        result[name] = {'dtype': str(value.dtype), 'shape': list(value.shape),
                        'sha256': hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    return result


def state_model_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def snapshot_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def snapshot_payload(original, model, step):
    if set(original) != PAYLOAD_KEYS:
        raise ValueError('original payload inventory differs')
    return {**copy.deepcopy({k: v for k, v in original.items() if k != 'model'}),
            'step': step, 'model': snapshot_state(model.state_dict())}


class CommandHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256, eps=1e-5, elementwise_affine=False)
        self.linear1 = nn.Linear(259, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, latent, proprio):
        if latent.ndim != 2 or latent.shape[1] != 256 or proprio.shape != (latent.shape[0], 3):
            raise ValueError('invalid command head input shapes')
        value = torch.cat((self.norm(latent), proprio / proprio.new_tensor(CONTRACT['proprio_scales'])), 1)
        return self.linear2(F.silu(self.linear1(value))) * value.new_tensor(CONTRACT['output_scales'])


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
            raise ValueError('policy hook must fire once with [B,256]')
        return self.values[0]

    def close(self):
        self.handle.remove()


def check_scope_shapes(model):
    parameters = dict(model.named_parameters())
    for name, shape in SCOPE.items():
        if name not in parameters or list(parameters[name].shape) != shape:
            raise ValueError(f'suffix tensor differs: {name}')


def check_base_inventory(model):
    check_scope_shapes(model)
    if len(model.state_dict()) != 417 or sum(p.numel() for p in model.parameters()) != 31154802:
        raise ValueError('base must retain 417 states and 31154802 parameters')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('base parameters must be FP32')
    if any(module.training for module in model.modules()):
        raise ValueError('all base modules must remain in eval mode')


def set_training_scope(model):
    check_scope_shapes(model)
    model.requires_grad_(False).eval()
    for name, value in model.named_parameters():
        value.requires_grad_(name in SCOPE)
    return {name: value for name, value in model.named_parameters() if value.requires_grad}


def check_frozen_complement(model, original_state, *, training):
    current = digest_state(model.state_dict())
    if set(current) != set(original_state):
        raise ValueError('base state inventory changed')
    forbidden = [name for name in current if name not in SCOPE and current[name] != original_state[name]]
    if forbidden:
        raise ValueError(f'forbidden base tensors changed: {forbidden}')
    flags = {name for name, value in model.named_parameters() if value.requires_grad}
    if flags != (set(SCOPE) if training else set()):
        raise ValueError('base gradient flags differ from exact scope')
    if any(module.training for module in model.modules()):
        raise ValueError('all base modules must remain in eval mode')
    return [name for name in SCOPE if current[name] != original_state[name]]


def make_optimizer(model, head):
    check_base_inventory(model)
    suffix = set_training_scope(model)
    head.requires_grad_(True).eval()
    if {n: list(v.shape) for n, v in head.named_parameters()} != HEAD_SCOPE:
        raise ValueError('head scope differs')
    named = {**suffix, **{'head.' + n: p for n, p in head.named_parameters()}}
    if len(named) != 10 or sum(p.numel() for p in named.values()) != 296450:
        raise ValueError('joint optimized inventory differs')
    optimizer = torch.optim.AdamW(named.values(), lr=.001, weight_decay=0.)
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != {id(p) for p in named.values()}:
        raise ValueError('optimizer does not own exactly the intended tensors')
    return optimizer, named


def model_from_payload(payload, device='cpu'):
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise ValueError('base payload inventory differs')
    validate_checkpoint_payload(payload, (payload['config'] or {}).get('controller_mode', 'pp'))
    for name in ('vision_fp32.onnx', 'policy_fp32.onnx'):
        if not (Path('artifacts/models/prep') / name).is_file():
            raise FileNotFoundError(f'pinned prepared graph unavailable: {name}')
    model = SupercomboFS(image_head=any(k.startswith('img.') for k in payload['model']))
    model.load_state_dict(payload['model'], strict=True)
    model.controller_contract = copy.deepcopy(payload['controller_contract'])
    model.requires_grad_(False).eval().to(device)
    check_base_inventory(model)
    return model


def make_package(original, model, head, recipe, provenance, step):
    reference = digest_state(original['model'])
    check_base_inventory(model)
    changed = check_frozen_complement(model, reference, training=True)
    payload = snapshot_payload(original, model, step)
    head_state = snapshot_state(head.state_dict())
    record = {**provenance, 'step': step, 'training_scope': copy.deepcopy(SCOPE),
              'base_model_digest': state_model_digest(payload['model']),
              'base_state_digest': digest_state(payload['model']),
              'head_state_digest': digest_state(head_state),
              'original_base_model_digest': state_model_digest(original['model']),
              'original_base_state_digest': reference,
              'original_payload_metadata': copy.deepcopy({k: v for k, v in original.items() if k != 'model'}),
              'inherited_controller_metadata_role': 'diagnostics only; not evidence of joint training',
              'frozen_base_state_unchanged': True, 'frozen_base_state_count': 411,
              'changed_suffix_keys': changed}
    package = {'kind': KIND, 'base_payload': payload, 'head_state': head_state,
               'command_contract': copy.deepcopy(CONTRACT), 'recipe': copy.deepcopy(recipe), 'provenance': record}
    validate_package(package)
    return package


def validate_package(package):
    if (not isinstance(package, dict) or set(package) != PACKAGE_KEYS or
            package['kind'] != KIND or package['command_contract'] != CONTRACT):
        raise ValueError('incompatible joint command package')
    payload, p = package['base_payload'], package['provenance']
    if set(payload) != PAYLOAD_KEYS or len(payload['model']) != 417:
        raise ValueError('joint base payload inventory differs')
    base, head = digest_state(payload['model']), digest_state(package['head_state'])
    if base != p['base_state_digest'] or head != p['head_state_digest']:
        raise ValueError('saved tensor digest mismatch')
    if {n: v['shape'] for n, v in head.items()} != HEAD_SCOPE:
        raise ValueError('saved head inventory differs')
    if (any(base[n]['shape'] != shape for n, shape in SCOPE.items()) or
            any(base[n]['dtype'] != 'torch.float32' for n in SCOPE) or
            any(v['dtype'] != 'torch.float32' for v in head.values())):
        raise ValueError('saved parameter shape or dtype differs')
    if any(not torch.isfinite(payload['model'][name]).all() for name in SCOPE) or any(
            not torch.isfinite(value).all() for value in package['head_state'].values()):
        raise ValueError('nonfinite saved optimized parameters')
    if state_model_digest(payload['model']) != p['base_model_digest']:
        raise ValueError('updated base digest mismatch')
    original = p['original_base_state_digest']
    if set(base) != set(original) or p['training_scope'] != SCOPE:
        raise ValueError('original state or training scope differs')
    if any(base[n] != original[n] for n in base if n not in SCOPE):
        raise ValueError('forbidden saved base tensor changed')
    if (p['frozen_base_state_unchanged'] is not True or p['frozen_base_state_count'] != 411 or
            p['changed_suffix_keys'] != [n for n in SCOPE if base[n] != original[n]]):
        raise ValueError('frozen complement provenance differs')
    original_meta = p['original_payload_metadata']
    if set(original_meta) != PAYLOAD_KEYS - {'model'} or any(
            payload[n] != original_meta[n] for n in original_meta if n != 'step'):
        raise ValueError('inherited controller metadata differs')
    if payload['step'] != p['step']:
        raise ValueError('joint update step differs')


class JointActuatorPolicy:
    controller_name = KIND

    def __init__(self, model, head, contract, device):
        self.model = model.requires_grad_(False).eval()
        self.head = head.requires_grad_(False).eval()
        self.command_contract, self.device = contract, torch.device(device)
        mode = (getattr(model, 'controller_contract', None) or {}).get('mode', 'pp')
        self.base_policy = ModelPolicy(model, controller=mode, device=self.device)

    @classmethod
    def load(cls, package, device='cpu'):
        device = torch.device(device)
        if device.type != 'cpu':
            raise ValueError('joint deployment is fixed to CPU FP32')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        package = torch.load(Path(package), map_location='cpu', weights_only=False)
        validate_package(package)
        model = model_from_payload(package['base_payload'], device)
        head = CommandHead().to(device)
        head.load_state_dict(package['head_state'], strict=True)
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
                raise ValueError('nonfinite joint command')
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
