from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fsop.eval.closed_loop import ModelPolicy, mission_brake_with_override
from fsop.eval.controller import validate_checkpoint_payload
from fsop.eval.img_to_bev import bev_from_image_head
from fsop.model.build import SupercomboFS
from isaac_lab_fsop.evaluate import model_digest as model_digest

KIND = 'vision_cone_set_actuator_v1'
BASE_DIGEST = '14feb933b4399f4e25131acaf60aa4abae1dad7f1421303fc66a4735fc72f652'
BASE_CHECKPOINT_SHA256 = '48db741c8a686b2e635ae7a81b4e4c08d79fbfbd9f4dc4c163260da6f93f0cfb'
BASE_METADATA_SHA256 = '8b18172820e65711b2de2408de6b9afd4e0b834fc9094b9378a4c9816b17540f'
PREPARED_GRAPHS = {
    'vision_fp32.onnx': 'bac14fd2d539b10eabd966b462d0dcba4fcaab6b96bc49cfc4a8703f7450531c',
    'policy_fp32.onnx': 'a27088e94f581de472f9610183e3979987cf59a1b272816a339bb9757d00a17a',
}
HEAD_SCOPE = {'token1.weight': [64, 8], 'token1.bias': [64],
              'token2.weight': [64, 64], 'token2.bias': [64],
              'linear1.weight': [128, 332], 'linear1.bias': [128],
              'linear2.weight': [2, 128], 'linear2.bias': [2]}
CONTRACT = {
    'kind': KIND, 'version': 1, 'tokens_shape': ['B', 200, 8], 'mask_shape': ['B', 200],
    'token_fields': ['x/45', 'y/45', 'range/45', 'blue', 'yellow', 'orange', 'big_orange', 'score'],
    'token_order': 'stock CPU decoder order; wide then road before stock score sort and merge',
    'decoder': {'score_threshold': .25, 'peaks_per_camera': 100, 'merge_dist_m': .6,
                'camera_order': ['wide', 'road'], 'range_limits_m': [22., 45.],
                'ties': 'unchanged stock torch.nonzero/topk and numpy.argsort semantics'},
    'desire_shape': ['B', 25, 8], 'proprio_order': ['vx_mps', 'vy_mps', 'yaw_rate_radps'],
    'proprio_scales': [10., 2., 1.], 'output_order': ['steer_rad', 'accel_mps2'],
    'output_scales': [.1, 2.], 'target': 'qualified native teacher controls at the observation tick',
    'history': 'current image pair only; original diagnostic temporal state retained',
    'empty_set': 'zero mean, maximum and count; no inferred free space',
    'mission': 'bound raw request, apply mission_brake_with_override, bound again',
    'base_parameters': 31154802, 'head_parameters': 47618, 'total_parameters': 31202420,
    'frozen_base_state_tensors': 417, 'head_state_tensors': 8, 'training_scope': HEAD_SCOPE,
    'plan_semantics': 'original plan and inherited controller are diagnostics only',
    'deployment': {'device': 'cpu', 'dtype': 'torch.float32', 'threads': 4},
}
PAYLOAD_KEYS = {'step', 'model', 'config', 'controller_contract', 'controller_intervention'}
PACKAGE_KEYS = {'kind', 'base_payload', 'head_state', 'command_contract', 'recipe', 'provenance'}


def digest_state(state):
    return {name: {'dtype': str(value.dtype), 'shape': list(value.shape),
                   'sha256': hashlib.sha256(value.detach().cpu().contiguous().reshape(-1)
                                            .view(torch.uint8).numpy().tobytes()).hexdigest()}
            for name, value in state.items()}


def state_model_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def snapshot_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def pack_cones(xy, classes, scores):
    xy, classes, scores = np.asarray(xy), np.asarray(classes), np.asarray(scores)
    if xy.shape == (0,):
        xy = xy.reshape(0, 2)
    count = len(xy)
    if xy.shape != (count, 2) or classes.shape != (count,) or scores.shape != (count,) or count > 200:
        raise ValueError('invalid decoded cone shapes or count')
    if not all(np.isfinite(value).all() for value in (xy, classes, scores)):
        raise ValueError('nonfinite decoded cone input')
    if count and (not np.issubdtype(classes.dtype, np.integer) or np.any((classes < 0) | (classes > 3))):
        raise ValueError('invalid predicted cone class')
    tokens, mask = np.zeros((200, 8), np.float32), np.zeros(200, bool)
    tokens[:count, :2] = xy / 45.
    tokens[:count, 2] = np.linalg.norm(xy, axis=1) / 45.
    tokens[np.arange(count), 3 + classes.astype(np.int64)] = 1.
    tokens[:count, 7] = scores
    mask[:count] = True
    if not np.isfinite(tokens).all():
        raise ValueError('cone normalization overflow')
    return tokens, mask


def cpu_image_head(image_head):
    if set(image_head) != {'road', 'wide'}:
        raise ValueError('image head must contain both cameras')
    result, batch = {}, None
    for camera in ('wide', 'road'):
        fields = image_head[camera]
        if set(fields) != {'heat', 'wh', 'off'}:
            raise ValueError('image head fields differ')
        result[camera] = {}
        for name, channels in (('heat', 4), ('wh', 2), ('off', 2)):
            value = fields[name]
            if value.dtype != torch.float32 or value.ndim != 4 or value.shape[1] != channels:
                raise ValueError('image head shape or dtype differs')
            if batch is None:
                batch = value.shape[0]
            if value.shape[0] != batch or not torch.isfinite(value).all():
                raise ValueError('image head batch or finiteness differs')
            result[camera][name] = value.detach().cpu().contiguous()
        if len({tuple(v.shape[2:]) for v in result[camera].values()}) != 1:
            raise ValueError('image head spatial dimensions differ')
    return result


def decode_batch(image_head, calibrations):
    cpu = cpu_image_head(image_head)
    batch = cpu['wide']['heat'].shape[0]
    if len(calibrations) != batch:
        raise ValueError('one calibration required per example')
    result = []
    for index, calibration in enumerate(calibrations):
        calibration = np.asarray(calibration)
        if calibration.shape != (4, 4) or not np.isfinite(calibration).all():
            raise ValueError('invalid runtime camera calibration')
        decoded = bev_from_image_head(cpu, calibration, b=index)
        pack_cones(*decoded)
        result.append(decoded)
    return result


class ConeSetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.token1 = nn.Linear(8, 64)
        self.token2 = nn.Linear(64, 64)
        self.linear1 = nn.Linear(332, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, tokens, mask, desire, proprio):
        if tokens.ndim != 3:
            raise ValueError('cone head input shape differs')
        batch = tokens.shape[0]
        if (tokens.shape != (batch, 200, 8) or mask.shape != (batch, 200) or
                desire.shape != (batch, 25, 8) or proprio.shape != (batch, 3)):
            raise ValueError('cone head input shape differs')
        if mask.dtype != torch.bool or any(x.dtype != torch.float32 for x in (tokens, desire, proprio)):
            raise ValueError('cone head requires FP32 inputs and Boolean mask')
        if len({x.device for x in (tokens, mask, desire, proprio)}) != 1:
            raise ValueError('cone head input devices differ')
        if not all(torch.isfinite(x).all() for x in (tokens, desire, proprio)):
            raise ValueError('nonfinite cone head input')
        encoded = F.silu(self.token2(F.silu(self.token1(tokens.masked_fill(~mask[..., None], 0)))))
        count = mask.sum(1, keepdim=True)
        mean = encoded.masked_fill(~mask[..., None], 0).sum(1) / count.clamp_min(1)
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        pooled = torch.cat((mean, maximum, count.to(tokens.dtype) / 200., desire.flatten(1),
                            proprio / proprio.new_tensor(CONTRACT['proprio_scales'])), 1)
        return self.linear2(F.silu(self.linear1(pooled))) * pooled.new_tensor(CONTRACT['output_scales'])


def check_base_inventory(model):
    state = model.state_dict()
    if len(state) != 417 or sum(p.numel() for p in model.parameters()) != 31154802:
        raise ValueError('base must retain 417 states and 31154802 parameters')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in state.values()) or any(
            p.requires_grad for p in model.parameters()):
        raise ValueError('base must remain frozen FP32')
    if any(module.training for module in model.modules()) or state_model_digest(state) != BASE_DIGEST:
        raise ValueError('original base model bytes or eval state changed')


def check_frozen_base(model, reference):
    check_base_inventory(model)
    if digest_state(model.state_dict()) != reference:
        raise ValueError('original base state changed')


def make_optimizer(model, head):
    check_base_inventory(model)
    head.requires_grad_(True).eval()
    named = dict(head.named_parameters())
    if {name: list(value.shape) for name, value in named.items()} != HEAD_SCOPE:
        raise ValueError('head parameter scope differs')
    optimizer = torch.optim.AdamW(named.values(), lr=.001, weight_decay=0.)
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual) != 8 or set(actual) != {id(p) for p in named.values()}:
        raise ValueError('optimizer must own all and only eight head parameters')
    return optimizer, named


def model_from_payload(payload, device='cpu'):
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise ValueError('base payload inventory differs')
    if state_model_digest(payload['model']) != BASE_DIGEST:
        raise ValueError('base model differs from trusted original digest')
    check_payload_metadata(payload)
    validate_checkpoint_payload(payload, (payload['config'] or {}).get('controller_mode', 'pp'))
    for name, expected in PREPARED_GRAPHS.items():
        path = Path('artifacts/models/prep') / name
        if not path.is_file():
            raise FileNotFoundError(f'pinned prepared graph unavailable: {name}')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'prepared graph differs from trusted original: {name}')
    model = SupercomboFS(image_head=True)
    model.load_state_dict(payload['model'], strict=True)
    model.controller_contract = copy.deepcopy(payload['controller_contract'])
    model.requires_grad_(False).eval().to(device)
    check_base_inventory(model)
    return model


def make_package(original, model, head, recipe, provenance, step):
    check_frozen_base(model, digest_state(original['model']))
    payload = copy.deepcopy(original)
    payload['model'] = snapshot_state(model.state_dict())
    head_state = snapshot_state(head.state_dict())
    record = {**provenance, 'step': step, 'training_scope': copy.deepcopy(HEAD_SCOPE),
              'base_model_digest': BASE_DIGEST, 'original_base_model_digest': BASE_DIGEST,
              'base_state_digest': digest_state(payload['model']),
              'original_base_state_digest': digest_state(original['model']),
              'head_state_digest': digest_state(head_state),
              'original_payload_metadata': copy.deepcopy({k: v for k, v in original.items() if k != 'model'}),
              'frozen_base_state_unchanged': True, 'frozen_base_state_count': 417,
              'inherited_controller_metadata_role': 'original diagnostics only; not evidence of set-head training'}
    package = {'kind': KIND, 'base_payload': payload, 'head_state': head_state,
               'command_contract': copy.deepcopy(CONTRACT), 'recipe': copy.deepcopy(recipe), 'provenance': record}
    validate_package(package)
    return package


def validate_package(package):
    if (not isinstance(package, dict) or set(package) != PACKAGE_KEYS or package['kind'] != KIND or
            package['command_contract'] != CONTRACT):
        raise ValueError('incompatible cone-set package')
    payload, provenance = package['base_payload'], package['provenance']
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS or len(payload['model']) != 417:
        raise ValueError('saved base inventory differs')
    if state_model_digest(payload['model']) != BASE_DIGEST:
        raise ValueError('saved base differs from trusted original digest')
    base, head = digest_state(payload['model']), digest_state(package['head_state'])
    if base != provenance['base_state_digest'] or base != provenance['original_base_state_digest']:
        raise ValueError('saved base digest mismatch')
    if head != provenance['head_state_digest'] or {n: v['shape'] for n, v in head.items()} != HEAD_SCOPE:
        raise ValueError('saved head digest or inventory differs')
    if any(v['dtype'] != 'torch.float32' for v in head.values()) or any(
            v.is_floating_point() and v.dtype != torch.float32 for v in payload['model'].values()):
        raise ValueError('floating saved tensors must be FP32')
    if any(not torch.isfinite(v).all() for v in package['head_state'].values()):
        raise ValueError('nonfinite head parameter')
    if (provenance['training_scope'] != HEAD_SCOPE or provenance['base_model_digest'] != BASE_DIGEST or
            provenance['original_base_model_digest'] != BASE_DIGEST or
            provenance['original_base_checkpoint_sha256'] != BASE_CHECKPOINT_SHA256 or
            provenance['frozen_base_state_unchanged'] is not True or provenance['frozen_base_state_count'] != 417):
        raise ValueError('base or training provenance differs')
    metadata = {k: v for k, v in payload.items() if k != 'model'}
    if metadata != provenance['original_payload_metadata']:
        raise ValueError('original checkpoint metadata changed')
    check_payload_metadata(payload)
    validate_checkpoint_payload(payload, (payload['config'] or {}).get('controller_mode', 'pp'))


def check_payload_metadata(payload):
    metadata = {k: v for k, v in payload.items() if k != 'model'}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    if digest.hexdigest() != BASE_METADATA_SHA256:
        raise ValueError('base metadata differs from trusted original digest')


class ConeSetPolicy:
    controller_name = KIND

    def __init__(self, model, head, contract, device):
        self.device = torch.device(device)
        self.model = model.to(self.device).requires_grad_(False).eval()
        self.head = head.to(self.device).requires_grad_(False).eval()
        self.command_contract = contract
        mode = (getattr(model, 'controller_contract', None) or {}).get('mode', 'pp')
        self.base_policy = ModelPolicy(model, controller=mode, device=self.device)

    @classmethod
    def load(cls, path, device='cpu'):
        if torch.device(device).type != 'cpu':
            raise ValueError('cone-set deployment is fixed to CPU FP32')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        package = torch.load(Path(path), map_location='cpu', weights_only=False)
        validate_package(package)
        model = model_from_payload(package['base_payload'], device)
        head = ConeSetHead()
        head.load_state_dict(package['head_state'], strict=True)
        return cls(model, head, package['command_contract'], device)

    def reset(self):
        self.base_policy.reset()

    @torch.no_grad()
    def __call__(self, obs, env):
        _, _, info = self.base_policy(obs, env)
        tokens, mask = pack_cones(info['cones_xy'], info['cones_cls'], info['cones_score'])
        temporal = self.base_policy.ts
        desire = np.stack(temporal.desire_q).reshape(-1, temporal.fs, 8).max(1)
        proprio = np.array([obs.state.vx, obs.state.vy, obs.state.r], np.float32)
        inputs = [torch.from_numpy(value[None]).to(self.device) for value in (tokens, mask, desire, proprio)]
        raw = self.head(*inputs)[0].cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError('nonfinite cone-set command')
        bounded = np.array([np.clip(raw[0], -env.vp.max_steer, env.vp.max_steer),
                            np.clip(raw[1], -env.vp.max_brake, env.vp.max_accel)])
        accel, override = mission_brake_with_override(float(bounded[1]), obs, env)
        applied = [float(bounded[0]), float(np.clip(accel, -env.vp.max_brake, env.vp.max_accel))]
        info['base_controller_diagnostic'] = info.pop('controller', None)
        info['controller'] = {'kind': KIND, 'tokens': tokens.tolist(), 'mask': mask.tolist(),
                              'desire': desire.tolist(), 'proprio': proprio.tolist(), 'raw': raw.tolist(),
                              'bounded': bounded.tolist(), 'applied': applied, 'mission_override': override}
        return applied[0], applied[1], info
