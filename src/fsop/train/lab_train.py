"""Full-model native Lab replay trainer with optional frozen-vision caching."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate as collate

from fsop.data.native_replay import NativeReplay, episode_paths, history_indices, load_episodes, verify_arrays
from fsop.eval.controller import controller_contract, validate_checkpoint_controller
from fsop.model.build import PREP, load_fs
from fsop.sim.expert import T_IDXS
from fsop.train.action_target import LEGACY_ACTION_TARGET, NATIVE_FUTURE_ACTION_TARGET, action_target_contract
from fsop.train.device import autocast_ctx, pick_device
from fsop.train.losses import op_tangent_actions_from_op_plan, plan_l1, plan_nll, pp_steer_from_op_plan


def pp_accel_from_op_plan(mu, speed, max_brake, max_accel):
  """Differentiable PlanPurePursuit longitudinal command, including clipping."""
  times = torch.as_tensor(T_IDXS, dtype=mu.dtype, device=mu.device)
  t = torch.tensor(0.3, device=mu.device, dtype=mu.dtype)
  hi = int(np.searchsorted(T_IDXS, 0.3))
  a = (t - times[hi - 1]) / (times[hi] - times[hi - 1])
  row = mu[:, hi - 1] * (1 - a) + mu[:, hi] * a
  value = 1.5 * (row[:, 3] - speed) + 0.5 * row[:, 6]
  return torch.maximum(torch.minimum(value, max_accel), -max_brake)


def fresh_history(model, episodes, episode_ids, frame_ids, device, chunk_size=64, amp=False, channels_last=False,
                  history_context=24):
  """Recompute each distinct historical frame pair once per batch with current weights."""
  if history_context not in (8, 24):
    raise ValueError('history context must be 8 or 24')
  destinations = {}
  for b, (ei, frame) in enumerate(zip(episode_ids, frame_ids, strict=True)):
    for slot, k in enumerate(history_indices(int(frame))):
      if k >= 0 and slot >= 24 - history_context:
        destinations.setdefault((int(ei), int(k)), []).append((b, slot))
  pairs = list(destinations)
  result = torch.zeros((len(episode_ids), 24, 512), device=device, dtype=torch.float32)
  with torch.no_grad(), autocast_ctx(device, enabled=amp):
    for start in range(0, len(pairs), chunk_size):
      block = pairs[start:start + chunk_size]
      road = np.stack([episodes[ei].frame_pair('road', k) for ei, k in block])
      wide = np.stack([episodes[ei].frame_pair('wide', k) for ei, k in block])
      road = torch.from_numpy(road).to(device).float()
      wide = torch.from_numpy(wide).to(device).float()
      if channels_last:
        road = road.contiguous(memory_format=torch.channels_last)
        wide = wide.contiguous(memory_format=torch.channels_last)
      hidden = model.base.encode(road, wide).hidden.float()
      if not torch.isfinite(hidden).all():
        raise ValueError('nonfinite historical hidden features')
      for j, key in enumerate(block):
        for batch, slot in destinations[key]:
          result[batch, slot] = hidden[j]
  return result


class FrozenFeatureCache:
  """Episode/frame keyed FP32 features tied to frozen vision and replay arrays."""

  def __init__(self, model, datasets, device, chunk_size, history_context, channels_last=False):
    if (model.training or any(m.training for m in model.base.vision.modules()) or
        any(p.requires_grad for p in model.base.vision.parameters())):
      raise ValueError('feature cache requires eval mode and frozen vision')
    if history_context not in (8, 24):
      raise ValueError('history context must be 8 or 24')
    self.episodes = datasets[0].episodes
    self.history_context = history_context
    if any(ds.episodes is not self.episodes for ds in datasets):
      raise ValueError('feature cache datasets must share episode arrays')
    self.array_ids = [(id(ep.road), id(ep.wide)) for ep in self.episodes]
    self.vision_hash = {k: v for k, v in state_hashes(model.state_dict()).items() if k.startswith('base.vision.')}
    pairs = set()
    for dataset in datasets:
      for ei, frame in dataset.index:
        pairs.add((ei, frame))
        pairs.update((ei, int(k)) for k in history_indices(frame)[-history_context:] if k >= 0)
    pairs = sorted(pairs)
    self.features = {}
    start = time.monotonic()
    with torch.no_grad():
      for offset in range(0, len(pairs), chunk_size):
        block = pairs[offset:offset + chunk_size]
        road = np.stack([self.episodes[ei].frame_pair('road', k) for ei, k in block])
        wide = np.stack([self.episodes[ei].frame_pair('wide', k) for ei, k in block])
        road = torch.from_numpy(road).to(device).float()
        wide = torch.from_numpy(wide).to(device).float()
        if channels_last:
          road = road.contiguous(memory_format=torch.channels_last)
          wide = wide.contiguous(memory_format=torch.channels_last)
        hidden = model.base.encode(road, wide).hidden.float().cpu()
        if hidden.shape != (len(block), 512) or not torch.isfinite(hidden).all():
          raise ValueError('invalid cached hidden features')
        self.features.update(zip(block, hidden.unbind(), strict=True))
    self.build_seconds = time.monotonic() - start
    self.memory_bytes = len(pairs) * 512 * 4
    self.check(model, self.episodes, check_hash=True)

  def check(self, model, episodes, check_hash=False):
    if (model.training or any(m.training for m in model.base.vision.modules()) or
        any(p.requires_grad for p in model.base.vision.parameters())):
      raise ValueError('feature cache requires eval mode and frozen vision')
    if episodes is not self.episodes or [(id(ep.road), id(ep.wide)) for ep in episodes] != self.array_ids:
      raise ValueError('feature cache episode arrays changed')
    if check_hash:
      hashes = {k: v for k, v in state_hashes(model.state_dict()).items() if k.startswith('base.vision.')}
      if hashes != self.vision_hash:
        raise ValueError('feature cache vision state changed')

  def batch(self, model, episodes, episode_ids, frame_ids, device, history_context):
    self.check(model, episodes)
    if history_context != self.history_context:
      raise ValueError('feature cache history context changed')
    history = torch.zeros((len(episode_ids), 24, 512), dtype=torch.float32)
    current = []
    for b, (ei, frame) in enumerate(zip(episode_ids, frame_ids, strict=True)):
      ei, frame = int(ei), int(frame)
      current.append(self.features[(ei, frame)])
      for slot, k in enumerate(history_indices(frame)):
        if k >= 0 and slot >= 24 - history_context:
          history[b, slot] = self.features[(ei, int(k))]
    return history.to(device), torch.stack(current).to(device)

  def receipt(self):
    return {'build_seconds': self.build_seconds, 'feature_bytes': self.memory_bytes,
            'frames': len(self.features), 'vision_state_sha256': self.vision_hash,
            'episodes': [{'path': str(ep.path), 'arrays_sha256': {k: ep.meta['arrays_sha256'][k]
                         for k in ('road.npy', 'wide.npy')}} for ep in self.episodes]}


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
      h.update(chunk)
  return h.hexdigest()


def dataset_verification(manifest, receipt=None):
  """Bind a successful independent verification receipt to the exact native manifest."""
  manifest = Path(manifest).resolve()
  manifest_bytes = manifest.read_bytes()
  document = json.loads(manifest_bytes)
  if not isinstance(document, dict) or document.get('schema_version') != 1:
    raise ValueError('manifest training requires a complete native manifest and verification receipt')
  membership = document.get('membership_sha256')
  manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
  receipt_path = Path(receipt).resolve() if receipt is not None else manifest.parent / 'verification.json'
  try:
    receipt_bytes = receipt_path.read_bytes()
  except FileNotFoundError as exc:
    raise ValueError(f'dataset verification receipt missing: {receipt_path}') from exc
  evidence = json.loads(receipt_bytes)
  if (not isinstance(evidence, dict) or evidence.get('passed') is not True or
      evidence.get('manifest_sha256') != manifest_sha or evidence.get('membership_sha256') != membership):
    raise ValueError('dataset verification receipt does not match manifest and membership')
  return {'status': 'verified', 'path': str(receipt_path),
          'sha256': hashlib.sha256(receipt_bytes).hexdigest(), 'manifest_sha256': manifest_sha,
          'membership_sha256': membership}


PINNED_POLICY_SHA256 = 'a27088e94f581de472f9610183e3979987cf59a1b272816a339bb9757d00a17a'


def prove_policy_context(model, policy_path=PREP / 'policy_fp32.onnx'):
  """Fail closed unless the inspected ONNX and loaded FX gather still discard early slots.

  Inspected ONNX: fb_full [1,25,512] has one consumer, transpose [1,0,2];
  its sole consumer is GatherND(batch_dims=0) with indices [-9,...,-1].
  This selects original history slots 16..24, including the current hidden.
  The loaded checkpoint may replace the integer index buffer, so check it too.
  """
  from onnx2torch.node_converters.gather import OnnxGatherND
  from onnx2torch.node_converters.transpose import OnnxTranspose

  if sha256(policy_path) != PINNED_POLICY_SHA256:
    raise ValueError('history-context 8 requires the inspected pinned policy graph')
  policy = model.base.policy
  graph = getattr(policy, 'graph', None)
  if graph is None:
    raise ValueError('history-context 8 requires an inspectable converted policy graph')
  placeholders = [n for n in graph.nodes if n.op == 'placeholder']
  if len(placeholders) != 3 or len(placeholders[0].users) != 1:
    raise ValueError('policy feature input has unexpected consumers')
  transpose = next(iter(placeholders[0].users))
  if transpose.op != 'call_module' or transpose.kwargs or tuple(transpose.args) != (placeholders[0],):
    raise ValueError('policy feature input is not the inspected transpose')
  module = policy.get_submodule(transpose.target)
  if type(module) is not OnnxTranspose or module.perm != [1, 0, 2] or len(transpose.users) != 1:
    raise ValueError('policy transpose differs from the inspected graph')
  gather = next(iter(transpose.users))
  if gather.op != 'call_module' or gather.kwargs or len(gather.args) != 2 or gather.args[0] is not transpose:
    raise ValueError('policy transpose has unexpected consumers')
  module = policy.get_submodule(gather.target)
  if type(module) is not OnnxGatherND or module._batch_dims != 0:
    raise ValueError('policy gather differs from the inspected graph')
  index_node = gather.args[1]
  if getattr(index_node, 'op', None) != 'get_attr':
    raise ValueError('policy gather indices are not a checkpoint buffer')
  index = policy.get_buffer(index_node.target).detach().cpu()
  expected = torch.arange(-9, 0, dtype=torch.int64).reshape(9, 1)
  if index.dtype != torch.int64 or not torch.equal(index, expected):
    raise ValueError('policy gather indices differ from the inspected graph')
  return {'policy_sha256': PINNED_POLICY_SHA256, 'feature_slots': list(range(16, 25)),
          'transpose': [1, 0, 2], 'gather_indices': list(range(-9, 0)),
          'fx_gather_buffer': index_node.target}


def archive_source(path):
  """Archive package sources and bind each archive entry to its SHA-256."""
  source_root = Path(__file__).resolve().parents[2]
  files = sorted((source_root / 'fsop').rglob('*.py'))
  if not files:
    raise ValueError('fsop package sources are unavailable')
  inventory = {}
  with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for file in files:
      name = file.relative_to(source_root).as_posix()
      contents = file.read_bytes()
      inventory[name] = hashlib.sha256(contents).hexdigest()
      archive.writestr(name, contents)
  return {'archive': str(path), 'archive_sha256': sha256(path), 'files_sha256': inventory}


def telemetry():
  try:
    query = '--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw'
    output = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=5, check=True)
    return output.stdout.strip().splitlines()
  except (OSError, subprocess.SubprocessError):
    return None


def configure(model, freeze_vision=False):
  model.set_trunk_trainable('all')
  if freeze_vision:
    model.base.vision.requires_grad_(False)
  for head in (model.bev, model.img):
    if head is not None:
      head.requires_grad_(False)
  # Frozen ONNX graph constants may contain infinities, but trainable weights
  # must always be finite. Do not silently turn corrupted weights into constants.
  for name, p in model.named_parameters():
    if p.requires_grad and not torch.isfinite(p).all():
      raise ValueError(f'nonfinite trainable parameter: {name}')
  if not any(p.requires_grad for p in model.base.policy.parameters()):
    raise ValueError('policy must have trainable parameters')
  if not freeze_vision and not any(p.requires_grad for p in model.base.vision.parameters()):
    raise ValueError('vision must have trainable parameters')


def state_hashes(state):
  """Hash the exact bytes and metadata of vision and auxiliary state tensors."""
  result = {}
  for name, tensor in state.items():
    if name.startswith(('base.vision.', 'bev.', 'img.')):
      value = tensor.detach().cpu().contiguous()
      result[name] = {'dtype': str(value.dtype), 'shape': list(value.shape),
                      'sha256': hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
  return result


def save_checkpoint(state, path, reference, freeze_vision, trainable_names):
  def check(model_state):
    hashes = state_hashes(model_state)
    if hashes.keys() != reference.keys():
      raise ValueError('vision or auxiliary state keys changed')
    for name, digest in hashes.items():
      if (freeze_vision or not name.startswith('base.vision.')) and digest != reference[name]:
        raise ValueError(f'frozen state changed: {name}')
    for name in trainable_names:
      if not torch.isfinite(model_state[name]).all():
        raise ValueError(f'nonfinite checkpoint parameter: {name}')
    return hashes

  current = check(state['model'])
  torch.save(state, path)
  saved = check(torch.load(path, map_location='cpu', weights_only=False)['model'])
  if current != saved:
    raise ValueError('saved vision or auxiliary state differs from live state')
  return saved


def vision_l2sp(model, reference, weight):
  terms = [(p - reference[name]).square().sum()
           for name, p in model.base.vision.named_parameters() if name in reference]
  return weight * sum(terms, torch.zeros((), device=next(model.parameters()).device))



def batch_loss(model, dataset, indices, device, w_steer, w_long, chunk_size, amp=False, channels_last=False,
               history_context=24, w_plan=1.0, cache=None, return_outputs=False, controller_mode="pp",
               action_target_mode=LEGACY_ACTION_TARGET):
  action_target_contract(action_target_mode, controller_mode)
  if cache is not None and amp:
    raise ValueError('feature cache requires FP32')
  batch = collate([dataset[int(i)] for i in indices])
  episode_ids, frame_ids = batch.pop('episode').tolist(), batch.pop('frame').tolist()
  if cache is None:
    history = fresh_history(model, dataset.episodes, episode_ids, frame_ids, device, chunk_size,
                            amp=amp, channels_last=channels_last, history_context=history_context)
  else:
    history, hidden = cache.batch(model, dataset.episodes, episode_ids, frame_ids, device, history_context)
    batch.pop('img')
    batch.pop('big_img')
  b = {k: v.to(device) for k, v in batch.items()}
  for name, tensor in b.items():
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
      raise ValueError(f'nonfinite input {name}')
  b['history'] = history
  if channels_last and cache is None:
    for key in ('img', 'big_img'):
      b[key] = b[key].contiguous(memory_format=torch.channels_last)
  with autocast_ctx(device, enabled=amp):
    out = (model.forward_images(b, heads=('plan',)) if cache is None else
           model.plan(history, hidden, b['desire'], b['traffic']))
  mu, logstd = out['plan_mu'].float(), out['plan_logstd'].float()
  if not torch.isfinite(mu).all() or not torch.isfinite(logstd).all():
    raise ValueError('nonfinite model plan')
  plan = plan_nll(mu, logstd, b['plan'])
  if controller_mode == 'op_tangent_v1':
    steer, long = op_tangent_actions_from_op_plan(mu, b['speed'], b['lateral_speed'], b['vehicle_wheelbase'],
                                                  b['vehicle_max_brake'], b['vehicle_max_accel'])
  elif controller_mode == 'pp':
    steer = pp_steer_from_op_plan(mu, b['speed'], b['vehicle_lr'], b['vehicle_wheelbase'])
    long = pp_accel_from_op_plan(mu, b['speed'], b['vehicle_max_brake'], b['vehicle_max_accel'])
  else:
    raise ValueError('unsupported native training controller')
  steer_target, long_target = b['steer_target'], b['accel_target']
  if action_target_mode == NATIVE_FUTURE_ACTION_TARGET:
    with torch.no_grad():
      steer_target, long_target = op_tangent_actions_from_op_plan(
        b['plan'], b['speed'], b['lateral_speed'], b['vehicle_wheelbase'],
        b['vehicle_max_brake'], b['vehicle_max_accel'])
  steer_loss = F.smooth_l1_loss(steer, steer_target, beta=0.05)
  long_loss = F.smooth_l1_loss(long, long_target, beta=0.05)
  total = w_plan * plan + w_steer * steer_loss + w_long * long_loss
  components = {'plan_nll': plan, 'plan_l1': plan_l1(mu, b['plan']),
                'steer': steer_loss, 'longitudinal': long_loss}
  if action_target_mode == NATIVE_FUTURE_ACTION_TARGET:
    components.update(teacher_steer=F.smooth_l1_loss(steer, b['steer_target'], beta=0.05),
                      teacher_longitudinal=F.smooth_l1_loss(long, b['accel_target'], beta=0.05))
  if return_outputs and controller_mode == 'op_tangent_v1':
    out['coherence_targets'] = b['plan']
    out['coherence_observed'] = torch.stack((b['speed'], b['lateral_speed'], b['yaw_rate']), dim=-1)
  return (total, components, out) if return_outputs else (total, components)


def frozen_cache_gate(model, dataset, indices, device, args, cache):
  """Compare plan, losses, and every trainable policy gradient at fixed tolerances."""
  cache.check(model, dataset.episodes, check_hash=True)
  parameters = [(name, p) for name, p in model.base.policy.named_parameters() if p.requires_grad]
  if not parameters:
    raise ValueError('cache parity gate requires trainable policy')
  results = []
  for candidate in (None, cache):
    loss, parts, output = batch_loss(model, dataset, indices, device, args.w_steer, args.w_long,
                                     args.history_chunk_size, amp=False, channels_last=args.channels_last,
                                     history_context=args.history_context, w_plan=args.w_plan,
                                     cache=candidate, return_outputs=True,
                                     **({'action_target_mode': args.action_target_mode}
                                        if (getattr(args, 'action_target_mode', LEGACY_ACTION_TARGET)
                                            != LEGACY_ACTION_TARGET)
                                        else {}),
                                     **({'controller_mode': args.controller_mode}
                                        if getattr(args, 'controller_mode', 'pp') != 'pp' else {}))
    grads = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
    results.append(({**{k: v.detach() for k, v in output.items() if k in ('plan_mu', 'plan_logstd', 'desire_state')},
                     'loss': loss.detach(), **{k: v.detach() for k, v in parts.items()}}, grads))
  comparisons = {}
  for name, value in results[0][0].items():
    other = results[1][0][name]
    comparisons[name] = {'passed': bool(torch.allclose(value, other, atol=1e-5, rtol=1e-5)),
                         'max_abs': float((value - other).abs().max())}
  for (name, _), left, right in zip(parameters, results[0][1], results[1][1], strict=True):
    if left is None or right is None:
      comparisons['gradient:' + name] = {'passed': left is None and right is None, 'max_abs': None}
    else:
      comparisons['gradient:' + name] = {'passed': bool(torch.allclose(left, right, atol=1e-5, rtol=1e-4)),
                                         'max_abs': float((left - right).abs().max())}
  cache.check(model, dataset.episodes, check_hash=True)
  return {'passed': all(v['passed'] for v in comparisons.values()),
          'tolerances': {'plan_loss_atol': 1e-5, 'plan_loss_rtol': 1e-5,
                         'gradient_atol': 1e-5, 'gradient_rtol': 1e-4},
          'indices': list(map(int, indices)), 'comparisons': comparisons}


def grad_norms(model):
  groups = {'vision': [], 'policy': []}
  for name, p in model.named_parameters():
    if p.grad is not None:
      group = 'vision' if name.startswith('base.vision.') else 'policy'
      groups[group].append(torch.linalg.vector_norm(p.grad.detach().float()))
  device = next(model.parameters()).device
  norms = []
  for group in ('vision', 'policy'):
    norms.append(torch.linalg.vector_norm(torch.stack(groups[group])) if groups[group]
                 else torch.zeros((), device=device))
  # One device-to-host transfer for the normal case; identify bad tensors only on failure.
  values = torch.stack(norms).cpu().tolist()
  if not all(math.isfinite(value) for value in values):
    for name, p in model.named_parameters():
      if p.grad is not None and not torch.isfinite(p.grad).all():
        raise ValueError(f'nonfinite gradient: {name}')
    raise ValueError('nonfinite aggregate gradient norm')
  return dict(zip(('vision', 'policy'), values, strict=True))


def validation_samples(dataset, limit):
  """Balance a fixed deterministic sample budget across every validation episode."""
  groups = {i: [] for i, ep in enumerate(dataset.episodes) if ep.meta['split'] == 'val'}
  for index, (ei, _) in enumerate(dataset.index):
    groups[ei].append(index)
  if not groups or any(not indices for indices in groups.values()):
    raise ValueError('every validation episode must contain at least one valid anchor')
  if limit < len(groups):
    raise ValueError('val-samples must cover every validation episode')
  target = min(limit, sum(map(len, groups.values())))
  quotas = {ei: 0 for ei in groups}
  remaining = target
  while remaining:
    for ei, indices in groups.items():
      if quotas[ei] < len(indices) and remaining:
        quotas[ei] += 1
        remaining -= 1
  selected = {}
  for ei, indices in groups.items():
    n = quotas[ei]
    positions = ([len(indices) // 2] if n == 1 else np.linspace(0, len(indices) - 1, n, dtype=int))
    selected[ei] = [indices[int(i)] for i in positions]
  return selected



class RecoverySampler:

  def __init__(self, dataset, recipe_path, gates_dir, batch_size, recovery_draws, init_sha256,
               provenance_mode="strict", collection_policy_checkpoint_sha=None):
    if not 0 < recovery_draws < batch_size:
      raise ValueError('recovery draws must be positive and smaller than batch size')
    recipe_path = Path(recipe_path).resolve()
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    if provenance_mode == 'strict':
      if collection_policy_checkpoint_sha is not None:
        raise ValueError('strict recovery provenance does not accept a separate collection checkpoint')
      collection_sha = init_sha256
    elif provenance_mode == 'off_policy_v1':
      if (not isinstance(collection_policy_checkpoint_sha, str) or
          not re.fullmatch(r'[0-9a-f]{64}', collection_policy_checkpoint_sha)):
        raise ValueError('off_policy_v1 requires an explicit collection policy checkpoint SHA-256')
      collection_sha = collection_policy_checkpoint_sha
    else:
      raise ValueError('unknown recovery provenance mode')
    if not isinstance(recipe, dict) or recipe.get('checkpoint_sha256') != collection_sha:
      raise ValueError('recovery recipe checkpoint does not match collection policy')
    self.provenance_mode = provenance_mode
    self.collection_policy_checkpoint_sha = collection_sha
    self.optimization_init_sha256 = init_sha256
    ids = recipe.get('recovery_case_ids') if isinstance(recipe, dict) else None
    if (not isinstance(ids, list) or not ids or
        any(not isinstance(x, str) or not x or Path(x).name != x or x in ('.', '..') for x in ids) or
        len(set(ids)) != len(ids)):
      raise ValueError('recovery recipe must contain unique recovery_case_ids')
    recipe_sha = hashlib.sha256(recipe_bytes).hexdigest()
    if not gates_dir:
      raise ValueError('recovery gates directory is required')
    gates_dir = Path(gates_dir).resolve()
    if not gates_dir.is_dir():
      raise ValueError('recovery gates directory does not exist')
    by_id = {}
    marked = set()
    for ei, ep in enumerate(dataset.episodes):
      if ep.meta['split'] != 'train':
        continue
      case_id = ep.meta.get('id')
      if case_id in by_id:
        raise ValueError('duplicate training case identity')
      by_id[case_id] = ei
      if ep.meta.get('collection_policy', {}).get('mode') == 'learner_prefix':
        marked.add(case_id)
    if set(ids) != marked or not set(ids).issubset(by_id):
      raise ValueError('recovery recipe case IDs must match learner-prefix training episodes')
    index_by_episode = {}
    for index, (ei, frame) in enumerate(dataset.index):
      index_by_episode.setdefault(ei, []).append((index, frame))
    self.groups = []
    self.recovery_indices = set()
    for case_id in ids:
      ei = by_id[case_id]
      ep = dataset.episodes[ei]
      policy = ep.meta['collection_policy']
      tick = policy.get('actual_takeover_tick')
      if (type(tick) is not int or tick <= 0 or tick >= len(ep.teacher) or
          np.any(ep.teacher[:tick]) or not np.all(ep.teacher[tick:]) or
          ep.meta.get('control_hz') != 20 or ep.meta.get('checkpoint_sha256') != collection_sha):
        raise ValueError(f'{case_id}: invalid teacher handoff, control frequency, or learner checkpoint')
      terminal = ep.meta.get('terminal_continuation', {})
      terminal_tick = terminal.get('start_tick')
      if (type(terminal_tick) is not int or not tick < terminal_tick <= len(ep.teacher) or
          terminal.get('end_tick') != len(ep.teacher) or len(ep.teacher) - terminal_tick != 200 or
          terminal.get('contacts') != 0):
        raise ValueError(f'{case_id}: invalid terminal continuation')
      anchors = [(index, frame) for index, frame in index_by_episode.get(ei, [])
                 if tick <= int(ep.ticks[frame]) < min(tick + 40, terminal_tick)]
      if len(anchors) < 8:
        raise ValueError(f'{case_id}: fewer than eight qualified recovery anchors')
      receipt_path = gates_dir / f'{case_id}.json'
      try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        meta_bytes = (ep.path / 'meta.json').read_bytes()
        disk_meta = json.loads(meta_bytes)
        verify_arrays(ep.path, ep.meta.get('arrays_sha256'))
        source_sha = sha256(ep.path / 'source.zip')
        hashes = receipt.get('hashes', {})
        if (type(receipt.get('schema_version')) is not int or receipt['schema_version'] != 1 or
            receipt.get('passed') is not True or
            receipt.get('reasons') != [] or receipt.get('case_id') != case_id or
            Path(receipt.get('episode', '')).resolve() != ep.path.resolve() or
            Path(receipt.get('recipe', '')).resolve() != recipe_path or disk_meta != ep.meta or
            hashes.get('recipe_sha256') != recipe_sha or
            hashes.get('meta_sha256') != hashlib.sha256(meta_bytes).hexdigest() or
            hashes.get('arrays_sha256') != ep.meta.get('arrays_sha256') or
            hashes.get('source_archive_sha256') != source_sha or
            source_sha != ep.meta.get('source_archive_sha256') or
            receipt.get('takeover_tick') != tick or receipt.get('terminal_start_tick') != terminal_tick or
            receipt.get('recovery_frame_indices') != [frame for _, frame in anchors] or
            receipt.get('recovery_ticks') != [int(ep.ticks[frame]) for _, frame in anchors]):
          raise ValueError('receipt does not match episode, recipe, or selected anchors')
      except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f'{case_id}: invalid recovery gate: {exc}') from exc
      self.recovery_indices.update(index for index, _ in anchors)
      self.groups.append({'case_id': case_id, 'episode': ei, 'takeover_tick': tick,
                          'gate_path': str(receipt_path), 'gate_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
                          'terminal_start_tick': terminal_tick, 'indices': [index for index, _ in anchors],
                          'frames': [frame for _, frame in anchors],
                          'ticks': [int(ep.ticks[frame]) for _, frame in anchors]})
    self.other = np.array([i for i in range(len(dataset)) if i not in self.recovery_indices], dtype=np.int64)
    if not len(self.other):
      raise ValueError('non-recovery training pool is empty')
    self.dataset = dataset
    self.batch_size = batch_size
    self.recovery_draws = recovery_draws
    self.counts = np.zeros(len(dataset), dtype=np.int64)
    self.recipe_path = recipe_path
    self.recipe_sha256 = recipe_sha

  def draw(self, rng):
    choices = rng.integers(len(self.groups), size=self.recovery_draws)
    recovery = [self.groups[int(group)]['indices'][int(rng.integers(len(self.groups[int(group)]['indices'])))]
                for group in choices]
    other = self.other[rng.integers(len(self.other), size=self.batch_size - self.recovery_draws)]
    result = np.concatenate((np.asarray(recovery, dtype=np.int64), other))
    rng.shuffle(result)
    np.add.at(self.counts, result, 1)
    return result

  def protocol(self):
    other_groups = {}
    for index in self.other:
      ei, frame = self.dataset.index[int(index)]
      other_groups.setdefault(ei, []).append(frame)
    return {'recipe_path': str(self.recipe_path), 'recipe_sha256': self.recipe_sha256,
            'provenance_mode': self.provenance_mode,
            'collection_policy_checkpoint_sha': self.collection_policy_checkpoint_sha,
            'optimization_init_sha256': self.optimization_init_sha256,
            'window_ticks': 40, 'control_hz': 20, 'with_replacement': True,
            'recovery_draws_per_batch': self.recovery_draws,
            'other_draws_per_batch': self.batch_size - self.recovery_draws,
            'recovery_episode_probability': 1 / len(self.groups),
            'other_anchor_probability': 1 / len(self.other),
            'recovery_anchor_count': len(self.recovery_indices), 'other_anchor_count': len(self.other),
            'recovery_episodes': [{**g, 'anchor_probability': 1 / (len(self.groups) * len(g['indices']))}
                                  for g in self.groups],
            'other_episodes': [{'case_id': self.dataset.episodes[ei].meta['id'],
                                'episode': ei, 'frames': frames}
                               for ei, frames in other_groups.items()]}

  def draw_counts(self):
    return {'total_draws': int(self.counts.sum()),
            'recovery_draws': int(sum(self.counts[i] for i in self.recovery_indices)),
            'episodes': [{'case_id': ep.meta['id'],
                          'anchors': [{'frame': frame, 'draws': int(self.counts[i])}
                                      for i, (episode, frame) in enumerate(self.dataset.index) if episode == ei]}
                         for ei, ep in enumerate(self.dataset.episodes) if ep.meta['split'] == 'train']}


def plan_coherence(plans, examples):
  """Near-horizon consistency in OP coordinates, per anchor; no acceptance threshold."""
  times = np.asarray(T_IDXS)
  n = int(np.searchsorted(times, 1.0, side='right'))
  dt = np.diff(times[:n])
  rows = []
  for plan, sample in zip(plans, examples, strict=True):
    pos = np.diff(plan[:n, :2], axis=0) - .5 * (plan[:n-1, 3:5] + plan[1:n, 3:5]) * dt[:, None]
    vel = np.diff(plan[:n, 3:5], axis=0) - .5 * (plan[:n-1, 6:8] + plan[1:n, 6:8]) * dt[:, None]
    # The follower consumes raw relative yaw; modulo wrapping would hide a 2π shortcut.
    angle_delta = np.diff(plan[:n, 11])
    yaw = angle_delta - .5 * (plan[:n-1, 14] + plan[1:n, 14]) * dt
    velocity_secant = np.diff(plan[:n, :2], axis=0) / dt[:, None]
    acceleration_secant = np.diff(plan[:n, 3:5], axis=0) / dt[:, None]
    yaw_rate_secant = angle_delta / dt
    velocity_endpoint = np.stack((plan[:n-1, 3:5], plan[1:n, 3:5])) - velocity_secant
    acceleration_endpoint = np.stack((plan[:n-1, 6:8], plan[1:n, 6:8])) - acceleration_secant
    yaw_rate_endpoint = np.stack((plan[:n-1, 14], plan[1:n, 14])) - yaw_rate_secant
    rows.append({'velocity_secant_max_mps': float(np.linalg.norm(velocity_endpoint, axis=-1).max()),
                 'acceleration_secant_max_mps2': float(np.linalg.norm(acceleration_endpoint, axis=-1).max()),
                 'yaw_rate_secant_max_radps': float(np.abs(yaw_rate_endpoint).max()),
                 'position_interval_m': float(np.linalg.norm(pos, axis=1).mean()),
                 'position_interval_max_m': float(np.linalg.norm(pos, axis=1).max()),
                 'velocity_interval_mps': float(np.linalg.norm(vel, axis=1).mean()),
                 'velocity_interval_max_mps': float(np.linalg.norm(vel, axis=1).max()),
                 'yaw_interval_rad': float(np.abs(yaw).mean()),
                 'yaw_interval_max_rad': float(np.abs(yaw).max()),
                 'position_cumulative_max_m': float(np.linalg.norm(np.cumsum(pos, axis=0), axis=1).max()),
                 'velocity_cumulative_max_mps': float(np.linalg.norm(np.cumsum(vel, axis=0), axis=1).max()),
                 'yaw_cumulative_max_rad': float(np.abs(np.cumsum(yaw)).max()),
                 'initial_position_m': float(np.linalg.norm(plan[0, :2])),
                 'initial_yaw_rad': float(abs(plan[0, 11])),
                 'initial_velocity_mps': float(np.linalg.norm(
                     plan[0, 3:5] - [sample['speed'], -sample['lateral_speed']])),
                 'initial_yaw_rate_radps': float(abs(plan[0, 14] + sample['yaw_rate']))})
  return rows


def summarize_coherence(rows):
  return {key: {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)),
                'max': float(np.max(values))}
          for key in rows[0] for values in [[row[key] for row in rows]]}


def validate(model, dataset, selected, device, args, cache=None):
  """Evaluate the same episode-balanced anchor set in FP32 regardless of training AMP."""
  episodes = []
  coherence_rows, target_coherence_rows = [], []
  mode = getattr(args, 'controller_mode', 'pp')
  target_mode = getattr(args, 'action_target_mode', LEGACY_ACTION_TARGET)
  action_target_contract(target_mode, mode)
  with torch.no_grad():
    for ei, indices in selected.items():
      episode_coherence, episode_target = [], []
      totals = {'loss': 0., 'plan_nll': 0., 'plan_l1': 0., 'steer': 0., 'longitudinal': 0.}
      if target_mode == NATIVE_FUTURE_ACTION_TARGET:
        totals.update(teacher_steer=0., teacher_longitudinal=0.)
      for start in range(0, len(indices), args.batch_size):
        block = indices[start:start + args.batch_size]
        kwargs = {'cache': cache} if cache is not None else {}
        if mode == 'op_tangent_v1':
          loss, components, output = batch_loss(model, dataset, block, device, args.w_steer, args.w_long,
                                              args.history_chunk_size, amp=False, channels_last=args.channels_last,
                                              history_context=args.history_context, w_plan=args.w_plan,
                                              return_outputs=True, controller_mode=mode,
                                              **({'action_target_mode': target_mode}
                                                 if target_mode != LEGACY_ACTION_TARGET else {}), **kwargs)
          observed = output['coherence_observed'].detach().cpu().numpy()
          examples = [dict(speed=row[0], lateral_speed=row[1], yaw_rate=row[2]) for row in observed]
          episode_coherence.extend(plan_coherence(output['plan_mu'].detach().cpu().numpy(), examples))
          episode_target.extend(plan_coherence(output['coherence_targets'].detach().cpu().numpy(), examples))
        else:
          loss, components = batch_loss(model, dataset, block, device, args.w_steer, args.w_long,
                                        args.history_chunk_size, amp=False, channels_last=args.channels_last,
                                        history_context=args.history_context, w_plan=args.w_plan, **kwargs)
        for name, value in {'loss': loss, **components}.items():
          totals[name] += float(value) * len(block)
      scores = {name: value / len(indices) for name, value in totals.items()}
      if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError('nonfinite validation loss')
      episode = {'path': str(dataset.episodes[ei].path), 'count': len(indices), **scores}
      if episode_coherence:
        episode['coherence'] = summarize_coherence(episode_coherence)
        episode['target_coherence'] = summarize_coherence(episode_target)
        coherence_rows.extend(episode_coherence)
        target_coherence_rows.extend(episode_target)
      episodes.append(episode)
  # Equal episode weighting keeps a short valid episode from being overwhelmed by a long one.
  aggregate = {name: float(np.mean([ep[name] for ep in episodes])) for name in totals}
  result = {'precision': 'float32', 'aggregation': 'equal_episode', 'count': sum(e['count'] for e in episodes),
            'aggregate': aggregate, 'episodes': episodes}
  if target_mode == NATIVE_FUTURE_ACTION_TARGET:
    result['action_target_contract'] = action_target_contract(target_mode, mode)
  if coherence_rows:
    result['coherence'] = summarize_coherence(coherence_rows)
    result['target_coherence'] = summarize_coherence(target_coherence_rows)
    result['coherence_definition'] = (
      'Each anchor has mean and maximum interval residuals for position, velocity and yaw up to 1 s. '
      'Yaw uses raw relative plan angles as consumed by the controller, without modulo wrapping. '
      'Cumulative maxima use successive sums of those signed interval residuals. '
      'Secant maxima compare each endpoint velocity, acceleration and yaw rate separately with the '
      'corresponding interval finite difference, preventing opposite endpoint cancellation. '
      'Initial values compare native measured state. Mean/p95/max aggregate over anchors globally and by episode. '
      'Target uses the same function.')
  return result


SELECTION_TRAINING_FIELDS = (
  'steps', 'val_every', 'batch_size', 'lr', 'lr_policy', 'lr_trunk', 'freeze_vision', 'l2sp',
  'w_plan', 'w_steer', 'w_long', 'grad_clip', 'val_samples', 'history_chunk_size', 'history_context',
  'threads', 'amp', 'channels_last', 'device', 'seed', 'cache_frozen_features', 'recovery_draws',
  'recovery_provenance_mode', 'collection_policy_checkpoint_sha',
)


COHERENCE_METRICS = (
  'position_interval_m', 'position_interval_max_m', 'velocity_interval_mps', 'velocity_interval_max_mps',
  'yaw_interval_rad', 'yaw_interval_max_rad', 'initial_position_m', 'initial_yaw_rad', 'initial_velocity_mps',
  'initial_yaw_rate_radps', 'position_cumulative_max_m', 'velocity_cumulative_max_mps', 'yaw_cumulative_max_rad',
  'velocity_secant_max_mps', 'acceleration_secant_max_mps2', 'yaw_rate_secant_max_radps',
)


def validate_selection_bounds(bounds, validation_cases):
  actions = {'plan_l1', 'steer', 'longitudinal'}
  if (not isinstance(bounds, dict) or set(bounds) != {'aggregate', 'episode', 'coherence', 'episode_coherence'} or
      not isinstance(bounds['aggregate'], dict) or set(bounds['aggregate']) != actions):
    raise ValueError('selection protocol has incomplete bounds')

  def finite_nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
      raise ValueError('selection bounds must be finite nonnegative numbers')

  def coherence(section):
    if not isinstance(section, dict) or set(section) != set(COHERENCE_METRICS):
      raise ValueError('selection protocol has incomplete coherence bounds')
    for stats in section.values():
      if not isinstance(stats, dict) or set(stats) != {'mean', 'p95', 'max'}:
        raise ValueError('selection protocol has incomplete coherence statistics')
      for value in stats.values():
        finite_nonnegative(value)

  for value in bounds['aggregate'].values():
    finite_nonnegative(value)
  for scope in ('episode', 'episode_coherence'):
    if not isinstance(bounds[scope], dict) or set(bounds[scope]) != validation_cases:
      raise ValueError('selection protocol has incomplete validation case bounds')
  for section in bounds['episode'].values():
    if not isinstance(section, dict) or set(section) != actions:
      raise ValueError('selection protocol has incomplete episode action bounds')
    for value in section.values():
      finite_nonnegative(value)
  coherence(bounds['coherence'])
  for section in bounds['episode_coherence'].values():
    coherence(section)


def selection_protocol_binding(args, mode, manifest_sha):
  """Bind predeclared selection identity; selector separately validates its numerical bounds."""
  path = getattr(args, 'selection_protocol', None)
  expected_sha = getattr(args, 'selection_protocol_sha', None)
  calibration_path = getattr(args, 'expected_calibration', None)
  if bool(path) != bool(expected_sha):
    raise ValueError('selection protocol path and SHA-256 must be supplied together')
  if (mode == 'op_tangent_v1' and not getattr(args, 'calibration_only', False) and
      not getattr(args, 'cache_parity_only', False)):
    if not path or not calibration_path:
      raise ValueError('tangent training requires a frozen selection protocol and expected calibration')
  if not path:
    return None
  if mode != 'op_tangent_v1' or getattr(args, 'calibration_only', False):
    raise ValueError('selection protocol is only for tangent training after calibration')
  if not isinstance(expected_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
    raise ValueError('invalid selection protocol SHA-256')
  if not isinstance(manifest_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', manifest_sha):
    raise ValueError('tangent training requires a verified manifest SHA-256')
  raw = Path(path).read_bytes()
  if hashlib.sha256(raw).hexdigest() != expected_sha:
    raise ValueError('selection protocol SHA-256 mismatch')
  doc = json.loads(raw)
  target_mode = getattr(args, 'action_target_mode', LEGACY_ACTION_TARGET)
  contract = action_target_contract(target_mode, mode)
  version = 2 if contract is not None else 1
  if (not isinstance(doc, dict) or doc.get('schema_version') != version or
      doc.get('kind') != 'op_tangent_v1_selection' or
      (version == 2 and doc.get('action_target_contract') != contract) or
      (version == 1 and 'action_target_contract' in doc)):
    raise ValueError('invalid selection protocol schema')
  actual_args = {key: getattr(args, key, False if key == 'cache_frozen_features' else None)
                 for key in SELECTION_TRAINING_FIELDS}
  if version == 2:
    actual_args['action_target_mode'] = target_mode
  eligible = [0] + list(range(args.val_every, args.steps + 1, args.val_every))
  if eligible[-1] != args.steps:
    eligible.append(args.steps)
  if (not calibration_path or doc.get('calibration_sha256') != sha256(calibration_path) or
      doc.get('initializer_sha256') != sha256(args.init) or doc.get('manifest_sha256') != manifest_sha or
      doc.get('controller_contract') != controller_contract(mode) or doc.get('expected_steps') != args.steps or
      doc.get('checkpoint_steps') != eligible or doc.get('training_args') != actual_args):
    raise ValueError('selection protocol identity or training configuration mismatch')
  manifest = json.loads(Path(args.manifest).read_text())
  membership_path = Path('docs/isaac-lab-training/membership.json')
  if sha256(args.manifest) != manifest_sha or sha256(membership_path) != manifest.get('membership_sha256'):
    raise ValueError('manifest or frozen membership changed')
  membership = json.loads(membership_path.read_text())
  validation_cases = {case['id'] for case in membership['cases'] if case['split'] == 'val'}
  if not validation_cases:
    raise ValueError('frozen membership has no validation cases')
  validate_selection_bounds(doc.get('bounds'), validation_cases)
  return {'path': str(Path(path).resolve()), 'sha256': expected_sha,
          'calibration_path': str(Path(calibration_path).resolve()),
          'calibration_sha256': doc['calibration_sha256'], 'bytes': raw}


def recheck_selection_binding(binding, output_path):
  if binding is not None and (sha256(binding['path']) != binding['sha256'] or
                              sha256(output_path) != binding['sha256'] or
                              sha256(binding['calibration_path']) != binding['calibration_sha256']):
    raise ValueError('selection protocol or calibration changed during training')


def run(args):
  mode = getattr(args, 'controller_mode', 'pp')
  if mode not in ('pp', 'op_tangent_v1'):
    raise ValueError('unsupported native training controller')
  target_mode = getattr(args, 'action_target_mode', LEGACY_ACTION_TARGET)
  target_contract = action_target_contract(target_mode, mode)
  target_metadata = {'action_target_contract': target_contract} if target_contract is not None else {}
  cache_enabled = getattr(args, 'cache_frozen_features', False)
  parity_only = getattr(args, 'cache_parity_only', False)
  if parity_only and getattr(args, 'calibration_only', False):
    raise ValueError('calibration-only and cache-parity-only are mutually exclusive')
  if (cache_enabled or parity_only) and (not args.freeze_vision or args.amp):
    raise ValueError('feature cache requires --freeze-vision and FP32 without --amp')
  if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.val_every < 1 or args.history_chunk_size < 1:
    raise ValueError('steps, batch size, learning rate, val interval and history chunk must be positive')
  values = (args.lr, args.w_steer, args.w_long, args.w_plan, args.lr_policy, args.lr_trunk, args.l2sp, args.grad_clip)
  if not all(math.isfinite(v) for v in values) or min(values[1:]) < 0:
    raise ValueError('learning rates, loss weights and gradient clip must be finite and nonnegative')
  if args.lr_policy == 0 or (not args.freeze_vision and args.lr_trunk == 0) or args.grad_clip == 0:
    raise ValueError('learning rate multipliers and gradient clip must be positive')
  if args.threads < 1:
    raise ValueError('threads must be positive')
  torch.set_num_threads(args.threads)
  # Match the deployed evaluator's FP32 path. Autocast is a separate, measured option.
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  if args.data and args.dataset_verification:
    raise ValueError('--dataset-verification requires --manifest')
  verification = (dataset_verification(args.manifest, args.dataset_verification) if args.manifest else
                  {'status': 'unverified_development_data', 'path': None, 'sha256': None})
  manifest_sha = verification.get('manifest_sha256')
  selection_binding = selection_protocol_binding(args, mode, manifest_sha)
  paths = episode_paths(manifest=args.manifest, data=args.data)
  if args.manifest and sha256(args.manifest) != manifest_sha:
    raise ValueError('manifest changed while being read')
  episodes = load_episodes(paths)
  if args.manifest and (sha256(args.manifest) != manifest_sha or
                        episode_paths(manifest=args.manifest) != paths):
    raise ValueError('manifest or episode metadata changed while loading')
  train, val = NativeReplay(episodes, 'train'), NativeReplay(episodes, 'val')
  validation_selection = validation_samples(val, args.val_samples)
  recovery_cases = getattr(args, 'recovery_cases', None)
  recovery_draws = getattr(args, 'recovery_draws', 0)
  recovery_gates = getattr(args, 'recovery_gates', None)
  if len({bool(recovery_cases), bool(recovery_draws), bool(recovery_gates)}) != 1:
    raise ValueError('recovery cases, gates, and positive recovery draws must be supplied together')
  provenance_mode = getattr(args, 'recovery_provenance_mode', 'strict')
  collection_sha = getattr(args, 'collection_policy_checkpoint_sha', None)
  if not recovery_cases and (provenance_mode != 'strict' or collection_sha is not None):
    raise ValueError('recovery provenance options require recovery cases')
  sampler = (RecoverySampler(train, recovery_cases, recovery_gates, args.batch_size, recovery_draws, sha256(args.init),
                             provenance_mode=provenance_mode, collection_policy_checkpoint_sha=collection_sha)
             if recovery_cases else None)
  device = pick_device(args.device)
  if args.amp and device.type != 'cuda':
    raise ValueError('--amp requires a CUDA device')
  torch.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)
  model = load_fs(args.init, map_location='cpu').to(device)
  validate_checkpoint_controller(getattr(model, 'controller_contract', None), mode)
  if args.channels_last:
    model.to(memory_format=torch.channels_last)
  # Eval mode preserves the runtime's behavior; gradients are still enabled.
  model.eval()
  context_proof = prove_policy_context(model) if args.history_context == 8 else None
  configure(model, freeze_vision=args.freeze_vision)
  reference = {n: p.detach().clone() for n, p in model.base.vision.named_parameters() if p.requires_grad}
  groups = [{'params': [p for p in model.base.policy.parameters() if p.requires_grad],
             'lr': args.lr * args.lr_policy}]
  if not args.freeze_vision:
    groups.append({'params': [p for p in model.base.vision.parameters() if p.requires_grad],
                   'lr': args.lr * args.lr_trunk, 'weight_decay': 0.0})
  optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
  frozen_reference = state_hashes(model.state_dict())
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  names = [name for name, p in model.named_parameters() if p.requires_grad]
  info = {'args': vars(args), 'calibration_only': getattr(args, 'calibration_only', False), **target_metadata,
          'controller_contract': controller_contract(mode), 'device': str(device),
          'baseline_state_sha256': frozen_reference,
          'numeric_precision': 'CUDA autocast bfloat16; history outputs and losses float32' if args.amp else 'float32',
          'init_sha256': sha256(args.init), 'train_anchors': len(train),
          'val_anchors': len(val), 'trainable_tensors': names,
          'validation_precision': 'float32', 'validation_aggregation': 'equal_episode',
          'history_context_proof': context_proof,
          'tf32': {'matmul': False, 'cudnn': False},
          'validation_selection': [{'path': str(val.episodes[ei].path),
                                    'frames': [val.index[i][1] for i in indices]}
                                   for ei, indices in validation_selection.items()],
          'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
          'auxiliary_heads': 'frozen: occlusion targets unverified',
          'episodes': [{'path': str(e.path), 'split': e.meta['split'], 'accepted': len(e.anchors),
                        'attempted': len(e.ticks), 'rejected': e.rejections,
                        'meta_sha256': sha256(e.path / 'meta.json')} for e in episodes]}
  if sampler is not None:
    info['recovery_sampling'] = sampler.protocol()
  if selection_binding is not None:
    info['selection_protocol'] = {k: v for k, v in selection_binding.items() if k != 'bytes'}
  history_file = out / 'history.jsonl'
  if (any((out / name).exists() for name in
          ('history.jsonl', 'run.json', 'source.zip', 'baseline_validation.json', 'calibration.json',
           'selection_protocol.json', 'selected.pt')) or
      any(out.glob('step_*.pt')) or any(out.glob('sampling_counts_*.json'))):
    raise FileExistsError(f'{out} contains a run; use a new output directory')
  if selection_binding is not None:
    (out / 'selection_protocol.json').write_bytes(selection_binding['bytes'])
    recheck_selection_binding(selection_binding, out / 'selection_protocol.json')
  cache = (FrozenFeatureCache(model, [train, val], device, args.history_chunk_size,
                              args.history_context, args.channels_last)
           if cache_enabled or parity_only else None)
  if cache is not None:
    info['frozen_feature_cache'] = cache.receipt()
  info['manifest_sha256'] = manifest_sha
  info['dataset_verification'] = verification
  info['source'] = archive_source(out / 'source.zip')
  (out / 'run.json').write_text(json.dumps(info, indent=2) + '\n')
  if parity_only:
    indices = np.linspace(0, len(train) - 1, min(args.batch_size, len(train)), dtype=int)
    result = frozen_cache_gate(model, train, indices, device, args, cache)
    (out / 'cache_parity.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if not result['passed']:
      raise ValueError('frozen feature cache parity gate failed')
    return
  if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
  start = time.monotonic()
  baseline = validate(model, val, validation_selection, device, args, **({'cache': cache} if cache else {}))
  calibration_identity = {
    **target_metadata,
    'version': 1, 'controller_contract': controller_contract(mode), 'init_sha256': sha256(args.init),
    'manifest_sha256': manifest_sha, 'dataset_verification_sha256': verification.get('sha256'),
    'episodes': info['episodes'], 'validation_selection': info['validation_selection'],
    'source_files_sha256': info['source']['files_sha256'], 'history_context': args.history_context,
    'history_context_proof': context_proof, 'freeze_vision': args.freeze_vision,
    'validation_precision': info['validation_precision'], 'device': str(device),
    'validation_batch_size': args.batch_size, 'history_chunk_size': args.history_chunk_size,
    'channels_last': args.channels_last, 'cache_enabled': cache_enabled, 'tf32': info['tf32'],
    'weights': {'plan': args.w_plan, 'steer': args.w_steer, 'longitudinal': args.w_long},
    'recovery_sampling': sampler.protocol() if sampler is not None else None,
  }
  calibration = {'version': 1, 'identity': calibration_identity, 'validation': baseline}
  expected_path = getattr(args, 'expected_calibration', None)
  if expected_path is not None:
    expected = json.loads(Path(expected_path).read_text())
    if calibration != expected:
      raise ValueError('step-zero validation differs from expected calibration identity or metrics')
  (out / 'calibration.json').write_text(json.dumps(calibration, indent=2, allow_nan=False) + '\n')
  best = baseline['aggregate']['loss']
  selected = out / 'selected.pt'
  selection_metadata = ({'selection_protocol_sha256': selection_binding['sha256'],
                         'calibration_sha256': selection_binding['calibration_sha256']}
                        if selection_binding is not None else {})
  baseline_state = {'model': model.state_dict(), 'step': 0, **selection_metadata,
                    **target_metadata,
                    **({'controller_contract': controller_contract(mode)} if mode != 'pp' else {}),
                    'optimizer': optimizer.state_dict(), 'config': vars(args),
                    **({'recovery_provenance': sampler.protocol()} if sampler is not None else {})}
  baseline['saved_state_sha256'] = save_checkpoint(baseline_state, selected, frozen_reference,
                                                   args.freeze_vision, names)
  model.load_state_dict(torch.load(selected, map_location=device, weights_only=False)['model'], strict=True)
  initial_checkpoint = out / 'step_000000.pt'
  save_checkpoint(baseline_state, initial_checkpoint, frozen_reference, args.freeze_vision, names)
  baseline['checkpoint'] = str(initial_checkpoint)
  baseline['checkpoint_sha256'] = sha256(initial_checkpoint)
  baseline['step'] = 0
  baseline['selected_sha256'] = sha256(selected)
  (out / 'baseline_validation.json').write_text(json.dumps(baseline, indent=2, allow_nan=False) + '\n')
  if getattr(args, 'calibration_only', False):
    return
  with history_file.open('x') as log:
    for step in range(1, args.steps + 1):
      recheck_selection_binding(selection_binding, out / 'selection_protocol.json')
      tick = time.monotonic()
      idx = sampler.draw(rng) if sampler is not None else rng.integers(len(train), size=args.batch_size)
      optimizer.zero_grad(set_to_none=True)
      loss, components = batch_loss(model, train, idx, device, args.w_steer, args.w_long, args.history_chunk_size,
                                    amp=args.amp, channels_last=args.channels_last,
                                    history_context=args.history_context, w_plan=args.w_plan,
                                    **({'action_target_mode': target_mode}
                                       if target_mode != LEGACY_ACTION_TARGET else {}),
                                    **({'cache': cache} if cache else {}),
                                    **({'controller_mode': mode} if mode != 'pp' else {}))
      retention = vision_l2sp(model, reference, args.l2sp)
      loss = loss + retention
      if not torch.isfinite(loss):
        raise ValueError('nonfinite loss')
      loss.backward()
      norms = grad_norms(model)
      torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
      optimizer.step()
      row = {'step': step, 'controller_contract': controller_contract(mode), **selection_metadata, **target_metadata,
             **({'recovery_provenance': {'provenance_mode': sampler.provenance_mode,
                                         'collection_policy_checkpoint_sha': sampler.collection_policy_checkpoint_sha,
                                         'optimization_init_sha256': sampler.optimization_init_sha256}}
                if sampler is not None else {}),
             'loss': float(loss.detach()),
             **{k: float(v.detach()) for k, v in components.items()}, 'l2sp': float(retention.detach()),
             'grad_norms': norms,
             'step_seconds': time.monotonic() - tick, 'elapsed_seconds': time.monotonic() - start,
             'torch_peak_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
      if step == 1 or step % args.val_every == 0 or step == args.steps:
        if cache is not None:
          cache.check(model, episodes, check_hash=True)
        validation = validate(model, val, validation_selection, device, args, **({'cache': cache} if cache else {}))
        score = validation['aggregate']['loss']
        row['val_loss'] = score
        if sampler is not None:
          counts_path = out / f'sampling_counts_{step:06d}.json'
          counts_path.write_text(json.dumps(sampler.draw_counts(), separators=(',', ':')) + '\n')
          row['sampling'] = {'total_draws': int(sampler.counts.sum()),
                             'recovery_draws': int(sum(sampler.counts[i] for i in sampler.recovery_indices)),
                             'counts_path': str(counts_path), 'counts_sha256': sha256(counts_path)}
        row['validation'] = validation
        row['nvml'] = telemetry()
        ckpt = out / f'step_{step:06d}.pt'
        recheck_selection_binding(selection_binding, out / 'selection_protocol.json')
        state = {'model': model.state_dict(), 'step': step, 'optimizer': optimizer.state_dict(), 'config': vars(args),
                 **selection_metadata, **target_metadata,
                 **({'controller_contract': controller_contract(mode)} if mode != 'pp' else {}),
                 **({'recovery_provenance': sampler.protocol()} if sampler is not None else {})}
        row['state_sha256'] = save_checkpoint(state, ckpt, frozen_reference, args.freeze_vision, names)
        row['checkpoint'] = str(ckpt)
        row['checkpoint_sha256'] = sha256(ckpt)
        if score < best:
          best = score
          selected = out / 'selected.pt'
          # Same complete state dictionary; strict reload verifies the artifact.
          row['selected_state_sha256'] = save_checkpoint(state, selected, frozen_reference, args.freeze_vision, names)
          model.load_state_dict(torch.load(selected, map_location=device, weights_only=False)['model'], strict=True)
          row['selected_sha256'] = sha256(selected)
      log.write(json.dumps(row, allow_nan=False) + '\n')
      log.flush()
      print(json.dumps(row, allow_nan=False), flush=True)


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  source = p.add_mutually_exclusive_group(required=True)
  source.add_argument('--manifest')
  source.add_argument('--data')
  p.add_argument('--dataset-verification', help='verification receipt (default: manifest directory/verification.json)')
  p.add_argument('--init', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--steps', type=int, default=1000)
  p.add_argument('--batch-size', type=int, default=8)
  p.add_argument('--recovery-cases', help='frozen JSON recipe with ordered recovery_case_ids')
  p.add_argument('--recovery-gates', help='directory of passed per-case recovery gate receipts')
  p.add_argument('--recovery-draws', type=int, default=0)
  p.add_argument('--recovery-provenance-mode', choices=('strict', 'off_policy_v1'), default='strict')
  p.add_argument('--collection-policy-checkpoint-sha')
  p.add_argument('--lr', type=float, default=1e-4)
  p.add_argument('--val-every', type=int, default=100)
  p.add_argument('--val-samples', type=int, default=128)
  p.add_argument('--history-chunk-size', type=int, default=64)
  p.add_argument('--history-context', type=int, choices=(8, 24), default=24)
  p.add_argument('--freeze-vision', action='store_true')
  p.add_argument('--cache-frozen-features', action='store_true')
  p.add_argument('--calibration-only', action='store_true',
                 help='save fixed step-zero validation and checkpoint without optimizer updates')
  p.add_argument('--expected-calibration', help='require exact matching step-zero calibration receipt before updates')
  p.add_argument('--selection-protocol', help='frozen tangent selection protocol')
  p.add_argument('--selection-protocol-sha', help='expected SHA-256 of frozen selection protocol')
  p.add_argument('--cache-parity-only', action='store_true',
                 help='write a fixed-batch numerical parity receipt and exit')
  p.add_argument('--controller-mode', choices=('pp', 'op_tangent_v1'), default='pp')
  p.add_argument('--action-target-mode', choices=(LEGACY_ACTION_TARGET, NATIVE_FUTURE_ACTION_TARGET),
                 default=LEGACY_ACTION_TARGET)
  p.add_argument('--w-plan', type=float, default=1.0)
  p.add_argument('--w-steer', type=float, default=3.281158240898283)
  p.add_argument('--w-long', type=float, default=1.0)
  p.add_argument('--lr-policy', type=float, default=0.3)
  p.add_argument('--lr-trunk', type=float, default=0.1)
  p.add_argument('--l2sp', type=float, default=1e-3)
  p.add_argument('--grad-clip', type=float, default=1.0)
  p.add_argument('--device', default='auto')
  p.add_argument('--amp', action='store_true', help='use CUDA bfloat16 autocast for history and current forward')
  p.add_argument('--channels-last', action='store_true')
  p.add_argument('--threads', type=int, default=4)
  p.add_argument('--seed', type=int, default=0)
  args = p.parse_args(argv)
  if args.val_samples < 1:
    p.error('--val-samples must be positive')
  run(args)


if __name__ == '__main__':
  main()
