"""Full-model native Lab replay trainer; no cached vision features are used."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate as collate

from fsop.data.native_replay import NativeReplay, episode_paths, history_indices, load_episodes
from fsop.model.build import PREP, load_fs
from fsop.sim.expert import T_IDXS
from fsop.train.device import autocast_ctx, pick_device
from fsop.train.losses import plan_nll, pp_steer_from_op_plan


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


def configure(model):
  model.set_trunk_trainable('all')
  for head in (model.bev, model.img):
    if head is not None:
      head.requires_grad_(False)
  # Frozen ONNX graph constants may contain infinities, but trainable weights
  # must always be finite. Do not silently turn corrupted weights into constants.
  for name, p in model.named_parameters():
    if p.requires_grad and not torch.isfinite(p).all():
      raise ValueError(f'nonfinite trainable parameter: {name}')
  if not any(p.requires_grad for p in model.base.vision.parameters()) or not any(
      p.requires_grad for p in model.base.policy.parameters()):
    raise ValueError('vision and policy must both have trainable parameters')


def vision_l2sp(model, reference, weight):
  return weight * sum((p - reference[name]).square().sum()
                      for name, p in model.base.vision.named_parameters() if name in reference)



def batch_loss(model, dataset, indices, device, w_steer, w_long, chunk_size, amp=False, channels_last=False,
               history_context=24):
  batch = collate([dataset[int(i)] for i in indices])
  history = fresh_history(model, dataset.episodes, batch.pop('episode').tolist(), batch.pop('frame').tolist(),
                          device, chunk_size, amp=amp, channels_last=channels_last,
                          history_context=history_context)
  b = {k: v.to(device) for k, v in batch.items()}
  for name, tensor in b.items():
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
      raise ValueError(f'nonfinite input {name}')
  b['history'] = history
  if channels_last:
    for key in ('img', 'big_img'):
      b[key] = b[key].contiguous(memory_format=torch.channels_last)
  with autocast_ctx(device, enabled=amp):
    out = model.forward_images(b, heads=('plan',))
  mu, logstd = out['plan_mu'].float(), out['plan_logstd'].float()
  if not torch.isfinite(mu).all() or not torch.isfinite(logstd).all():
    raise ValueError('nonfinite model plan')
  plan = plan_nll(mu, logstd, b['plan'])
  steer = pp_steer_from_op_plan(mu, b['speed'], b['vehicle_lr'], b['vehicle_wheelbase'])
  steer_loss = F.smooth_l1_loss(steer, b['steer_target'], beta=0.05)
  long = pp_accel_from_op_plan(mu, b['speed'], b['vehicle_max_brake'], b['vehicle_max_accel'])
  long_loss = F.smooth_l1_loss(long, b['accel_target'], beta=0.05)
  total = plan + w_steer * steer_loss + w_long * long_loss
  return total, {'plan_nll': plan, 'steer': steer_loss, 'longitudinal': long_loss}


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


def validate(model, dataset, selected, device, args):
  """Evaluate the same episode-balanced anchor set in FP32 regardless of training AMP."""
  episodes = []
  with torch.no_grad():
    for ei, indices in selected.items():
      totals = {'loss': 0., 'plan_nll': 0., 'steer': 0., 'longitudinal': 0.}
      for start in range(0, len(indices), args.batch_size):
        block = indices[start:start + args.batch_size]
        loss, components = batch_loss(model, dataset, block, device, args.w_steer, args.w_long,
                                      args.history_chunk_size, amp=False, channels_last=args.channels_last,
                                      history_context=args.history_context)
        for name, value in {'loss': loss, **components}.items():
          totals[name] += float(value) * len(block)
      scores = {name: value / len(indices) for name, value in totals.items()}
      if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError('nonfinite validation loss')
      episodes.append({'path': str(dataset.episodes[ei].path), 'count': len(indices), **scores})
  # Equal episode weighting keeps a short valid episode from being overwhelmed by a long one.
  aggregate = {name: float(np.mean([ep[name] for ep in episodes]))
               for name in ('loss', 'plan_nll', 'steer', 'longitudinal')}
  return {'precision': 'float32', 'aggregation': 'equal_episode', 'count': sum(e['count'] for e in episodes),
          'aggregate': aggregate, 'episodes': episodes}


def run(args):
  if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.val_every < 1 or args.history_chunk_size < 1:
    raise ValueError('steps, batch size, learning rate, val interval and history chunk must be positive')
  values = (args.lr, args.w_steer, args.w_long, args.lr_policy, args.lr_trunk, args.l2sp, args.grad_clip)
  if not all(math.isfinite(v) for v in values) or min(values[1:]) < 0:
    raise ValueError('learning rates, loss weights and gradient clip must be finite and nonnegative')
  if args.lr_policy == 0 or args.lr_trunk == 0 or args.grad_clip == 0:
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
  paths = episode_paths(manifest=args.manifest, data=args.data)
  if args.manifest and sha256(args.manifest) != manifest_sha:
    raise ValueError('manifest changed while being read')
  episodes = load_episodes(paths)
  if args.manifest and (sha256(args.manifest) != manifest_sha or
                        episode_paths(manifest=args.manifest) != paths):
    raise ValueError('manifest or episode metadata changed while loading')
  train, val = NativeReplay(episodes, 'train'), NativeReplay(episodes, 'val')
  validation_selection = validation_samples(val, args.val_samples)
  device = pick_device(args.device)
  if args.amp and device.type != 'cuda':
    raise ValueError('--amp requires a CUDA device')
  torch.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)
  model = load_fs(args.init, map_location='cpu').to(device)
  if args.channels_last:
    model.to(memory_format=torch.channels_last)
  # Eval mode preserves the runtime's behavior; gradients are still enabled.
  model.eval()
  context_proof = prove_policy_context(model) if args.history_context == 8 else None
  configure(model)
  reference = {n: p.detach().clone() for n, p in model.base.vision.named_parameters() if p.requires_grad}
  optimizer = torch.optim.AdamW([
    {'params': [p for p in model.base.policy.parameters() if p.requires_grad], 'lr': args.lr * args.lr_policy},
    {'params': [p for p in model.base.vision.parameters() if p.requires_grad],
     'lr': args.lr * args.lr_trunk, 'weight_decay': 0.0},
  ], weight_decay=1e-4)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  names = [name for name, p in model.named_parameters() if p.requires_grad]
  info = {'args': vars(args), 'device': str(device),
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
  history_file = out / 'history.jsonl'
  if any((out / name).exists() for name in
         ('history.jsonl', 'run.json', 'source.zip', 'baseline_validation.json', 'selected.pt')):
    raise FileExistsError(f'{out} contains a run; use a new output directory')
  info['manifest_sha256'] = manifest_sha
  info['dataset_verification'] = verification
  info['source'] = archive_source(out / 'source.zip')
  (out / 'run.json').write_text(json.dumps(info, indent=2) + '\n')
  if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
  start = time.monotonic()
  baseline = validate(model, val, validation_selection, device, args)
  best = baseline['aggregate']['loss']
  selected = out / 'selected.pt'
  baseline_state = {'model': model.state_dict(), 'step': 0,
                    'optimizer': optimizer.state_dict(), 'config': vars(args)}
  torch.save(baseline_state, selected)
  model.load_state_dict(torch.load(selected, map_location=device, weights_only=False)['model'], strict=True)
  baseline['step'] = 0
  baseline['selected_sha256'] = sha256(selected)
  (out / 'baseline_validation.json').write_text(json.dumps(baseline, indent=2, allow_nan=False) + '\n')
  with history_file.open('x') as log:
    for step in range(1, args.steps + 1):
      tick = time.monotonic()
      idx = rng.integers(len(train), size=args.batch_size)
      optimizer.zero_grad(set_to_none=True)
      loss, components = batch_loss(model, train, idx, device, args.w_steer, args.w_long, args.history_chunk_size,
                                    amp=args.amp, channels_last=args.channels_last,
                                    history_context=args.history_context)
      retention = vision_l2sp(model, reference, args.l2sp)
      loss = loss + retention
      if not torch.isfinite(loss):
        raise ValueError('nonfinite loss')
      loss.backward()
      norms = grad_norms(model)
      torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
      optimizer.step()
      row = {'step': step, 'loss': float(loss.detach()),
             **{k: float(v.detach()) for k, v in components.items()}, 'l2sp': float(retention.detach()),
             'grad_norms': norms,
             'step_seconds': time.monotonic() - tick, 'elapsed_seconds': time.monotonic() - start,
             'torch_peak_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
      if step == 1 or step % args.val_every == 0 or step == args.steps:
        validation = validate(model, val, validation_selection, device, args)
        score = validation['aggregate']['loss']
        row['val_loss'] = score
        row['validation'] = validation
        row['nvml'] = telemetry()
        ckpt = out / f'step_{step:06d}.pt'
        state = {'model': model.state_dict(), 'step': step, 'optimizer': optimizer.state_dict(), 'config': vars(args)}
        torch.save(state, ckpt)
        row['checkpoint'] = str(ckpt)
        row['checkpoint_sha256'] = sha256(ckpt)
        if score < best:
          best = score
          selected = out / 'selected.pt'
          # Same complete state dictionary; strict reload verifies the artifact.
          torch.save(state, selected)
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
  p.add_argument('--lr', type=float, default=1e-4)
  p.add_argument('--val-every', type=int, default=100)
  p.add_argument('--val-samples', type=int, default=128)
  p.add_argument('--history-chunk-size', type=int, default=64)
  p.add_argument('--history-context', type=int, choices=(8, 24), default=24)
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
