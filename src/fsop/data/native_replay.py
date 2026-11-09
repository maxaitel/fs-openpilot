"""Memory-mapped Isaac Lab replay with labels from executed native continuations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from fsop.geometry.op_frames import ego_to_op_plan
from fsop.model.temporal import N_FEAT
from fsop.sim.expert import plan_from_rollout

HORIZON = 200
ARRAY_NAMES = ('road', 'wide', 'ticks', 'states', 'controls', 'teacher', 'desires', 'cones', 'contacts')


def verify_arrays(path: Path, hashes: dict) -> None:
  if not isinstance(hashes, dict) or not {f'{name}.npy' for name in ARRAY_NAMES}.issubset(hashes):
    raise ValueError(f'{path}: array hashes must cover all required arrays')
  for name, expected in hashes.items():
    if (not isinstance(name, str) or Path(name).name != name or not name.endswith('.npy') or
        not isinstance(expected, str) or len(expected) != 64 or
        any(c not in '0123456789abcdef' for c in expected)):
      raise ValueError(f'{path}: invalid array hash entry')
    digest = hashlib.sha256()
    with (path / name).open('rb') as f:
      for chunk in iter(lambda: f.read(1024 * 1024), b''):
        digest.update(chunk)
    if digest.hexdigest() != expected:
      raise ValueError(f'{path}: SHA-256 mismatch for {name}')


def episode_paths(*, manifest: str | Path | None = None, data: str | Path | None = None) -> list[Path]:
  if (manifest is None) == (data is None):
    raise ValueError('provide exactly one of manifest or data')
  if data is not None:
    return sorted(p for p in Path(data).iterdir() if (p / 'meta.json').is_file())
  path = Path(manifest).resolve()
  entries = json.loads(path.read_text())
  if isinstance(entries, dict):
    entries = entries['episodes']
  if not isinstance(entries, list):
    raise ValueError('manifest must contain a list of episode paths')
  result = []
  for entry in entries:
    if isinstance(entry, dict):
      entry = entry['path']
    p = Path(entry)
    result.append((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
  return result


def history_indices(k: int) -> np.ndarray:
  return np.arange(k - N_FEAT, k, dtype=np.int64)


class NativeEpisode:
  def __init__(self, path: str | Path):
    self.path = Path(path).resolve()
    self.meta = json.loads((self.path / 'meta.json').read_text())
    m = self.meta
    if m.get('schema_version') != 1 or m.get('backend') != 'IsaacLab' or m.get('split') not in ('train', 'val'):
      raise ValueError(f'{self.path}: invalid native schema, backend, or split')
    if not all(k in m for k in ('track_id', 'discipline', 'seed')):
      raise ValueError(f'{self.path}: track_id, discipline and seed required for split isolation')
    if not isinstance(m.get('summary'), dict):
      raise ValueError(f'{self.path}: native teacher summary is required')
    try:
      self.lr = float(m['vehicle']['lr'])
      self.wheelbase = self.lr + float(m['vehicle']['lf'])
      self.max_brake = float(m['vehicle']['max_brake'])
      self.max_accel = float(m['vehicle']['max_accel'])
    except (KeyError, TypeError, ValueError) as exc:
      raise ValueError(f'{self.path}: invalid vehicle geometry') from exc
    if (not np.isfinite([self.lr, self.wheelbase, self.max_brake, self.max_accel]).all() or
        self.lr <= 0 or self.wheelbase <= self.lr or self.max_brake <= 0 or self.max_accel <= 0):
      raise ValueError(f'{self.path}: invalid vehicle geometry')
    verify_arrays(self.path, m.get('arrays_sha256'))
    for name in ARRAY_NAMES:
      setattr(self, name, np.load(self.path / f'{name}.npy', mmap_mode='r', allow_pickle=False))
    n, t = len(self.states), len(self.ticks)
    if (self.states.shape != (n, 9) or self.controls.shape != (n - 1, 2) or
        self.teacher.shape != (n - 1,) or self.teacher.dtype != np.bool_ or
        self.contacts.shape != (n,) or not np.issubdtype(self.contacts.dtype, np.integer) or
        self.ticks.shape != (t,) or not np.issubdtype(self.ticks.dtype, np.integer) or
        self.desires.shape != (t, 25, 8) or self.cones.ndim != 2 or self.cones.shape[1] != 3 or
        any(a.shape != (t, 6, 128, 256) or a.dtype != np.uint8 for a in (self.road, self.wide))):
      raise ValueError(f'{self.path}: invalid array shapes or dtypes')
    if n < 1 or t < 1 or not np.array_equal(self.ticks, np.arange(t) * 4) or self.ticks[-1] >= n:
      raise ValueError(f'{self.path}: frames must use contiguous phase-0 ticks')
    for name in ('states', 'controls', 'desires', 'cones'):
      if not np.isfinite(getattr(self, name)).all():
        raise ValueError(f'{self.path}: nonfinite {name}')
    if (not np.allclose(np.diff(self.states[:, 8]), 0.05, rtol=0, atol=1e-5) or
        np.any(np.diff(self.contacts) < 0) or np.any(self.contacts < 0)):
      raise ValueError(f'{self.path}: discontinuous timestamps or contacts')
    # Each attempted anchor has one mutually exclusive rejection reason, in
    # priority order: episode qualification, horizon, control, then contact.
    bad = np.r_[0, np.cumsum(~self.teacher, dtype=np.int64)]
    summary = m.get('summary')
    qualified = (summary.get('result') == 'finished' and summary.get('pass') is True and
                 summary.get('complete') is True and summary.get('physical_cone_hits') == 0 and
                 summary.get('oc') == 0)
    self.anchors = []
    self.rejections = {'bad_teacher': 0, 'horizon': 0, 'control': 0, 'contact': 0}
    for k, tick in enumerate(self.ticks):
      tick = int(tick)
      end = tick + HORIZON
      if not qualified:
        self.rejections['bad_teacher'] += 1
      elif end >= n:
        self.rejections['horizon'] += 1
      elif bad[end] != bad[tick]:
        self.rejections['control'] += 1
      elif self.contacts[tick] != 0 or self.contacts[end] != 0:
        self.rejections['contact'] += 1
      else:
        self.anchors.append(k)

  def frame_pair(self, camera: str, k: int) -> np.ndarray:
    if k < 0 or k >= len(self.ticks):
      raise IndexError(k)
    a = getattr(self, camera)
    return np.concatenate((a[max(0, k - 1)], a[k]), axis=0)

  def sample(self, k: int) -> dict:
    tick = int(self.ticks[k])
    future = np.array(self.states[tick:tick + HORIZON + 1], dtype=np.float64, copy=True)
    future[:, 8] -= future[0, 8]
    plan = ego_to_op_plan(plan_from_rollout(future, dt=0.05)).astype(np.float32)
    if not np.isfinite(plan).all():
      raise ValueError(f'{self.path}: nonfinite plan at tick {tick}')
    return {'img': self.frame_pair('road', k), 'big_img': self.frame_pair('wide', k),
            'desire': np.asarray(self.desires[k], dtype=np.float32).copy(),
            'traffic': np.array([1., 0.], dtype=np.float32), 'plan': plan,
            'speed': np.float32(self.states[tick, 3]),
            'steer_target': np.float32(self.controls[tick, 0]),
            'accel_target': np.float32(self.controls[tick, 1]),
            'vehicle_lr': np.float32(self.lr), 'vehicle_wheelbase': np.float32(self.wheelbase),
            'vehicle_max_brake': np.float32(self.max_brake), 'vehicle_max_accel': np.float32(self.max_accel)}


def load_episodes(paths: list[Path]) -> list[NativeEpisode]:
  if not paths or len(set(p.resolve() for p in paths)) != len(paths):
    raise ValueError('episode paths must be nonempty and unique')
  eps = [NativeEpisode(p) for p in paths]
  # Fixed skidpad layout may be shared only when the realization seed differs.
  identities = {}
  for ep in eps:
    m = ep.meta
    track = str(m['track_id'])
    identity = (track, str(m['seed'])) if str(m['discipline']).lower() == 'skidpad' else (track,)
    if identity in identities and identities[identity] != m['split']:
      raise ValueError(f'train/val overlap: {identity}')
    identities[identity] = m['split']
  for split in ('train', 'val'):
    if not any(ep.meta['split'] == split and ep.anchors for ep in eps):
      raise ValueError(f'no valid {split} anchors')
  return eps


class NativeReplay(Dataset):
  def __init__(self, episodes: list[NativeEpisode], split: str):
    self.episodes = episodes
    self.index = [(i, k) for i, ep in enumerate(episodes) if ep.meta['split'] == split for k in ep.anchors]
    if not self.index:
      raise ValueError(f'no valid {split} anchors')

  def __len__(self):
    return len(self.index)

  def __getitem__(self, i):
    ei, k = self.index[i]
    return {**self.episodes[ei].sample(k), 'episode': ei, 'frame': k}
