import hashlib
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fsop.data.native_replay import NativeEpisode, NativeReplay, load_episodes
from fsop.eval.controller import PlanPurePursuit
from fsop.geometry.op_frames import op_to_ego_plan
from fsop.model.temporal import TemporalState
from fsop.sim.vehicle import VehicleParams
from fsop.train.lab_train import fresh_history, pp_accel_from_op_plan
from fsop.train.losses import pp_steer_from_op_plan


def rehash(path):
  meta_path = path / 'meta.json'
  meta = json.loads(meta_path.read_text())
  hashes = {}
  for p in path.glob('*.npy'):
    with p.open('rb') as f:
      hashes[p.name] = hashlib.file_digest(f, 'sha256').hexdigest()
  meta['arrays_sha256'] = hashes
  meta_path.write_text(json.dumps(meta))


def episode(path, *, split='train', track='t1', seed=1, n=213):
  path.mkdir()
  meta = {'schema_version': 1, 'backend': 'IsaacLab', 'split': split, 'track_id': track,
          'discipline': 'trackdrive', 'seed': seed,
          'summary': {'result': 'finished', 'pass': True, 'complete': True, 'physical_cone_hits': 0, 'oc': 0},
          'vehicle': {'lr': 0.8, 'lf': 0.75, 'max_brake': 13., 'max_accel': 7.}}
  (path / 'meta.json').write_text(json.dumps(meta))
  ticks = np.arange(0, n, 4)
  states = np.zeros((n, 9), np.float64)
  states[:, 8] = 50 + np.arange(n) * .05
  states[:, 0] = np.arange(n) * .1
  states[:, 3] = 2
  arrays = {'ticks': ticks, 'states': states, 'controls': np.zeros((n - 1, 2), np.float32),
            'teacher': np.ones(n - 1, bool), 'desires': np.zeros((len(ticks), 25, 8), np.float32),
            'contacts': np.zeros(n, np.int64), 'cones': np.empty((0, 3), np.float32)}
  for name, array in arrays.items():
    np.save(path / (name + '.npy'), array)
  for cam in ('road', 'wide'):
    arr = np.lib.format.open_memmap(path / (cam + '.npy'), mode='w+', dtype=np.uint8,
                                     shape=(len(ticks), 6, 128, 256))
    for k in range(len(ticks)):
      arr[k] = k + 1
    del arr
  rehash(path)
  return path


def test_native_future_relative_time_and_boundaries(tmp_path):
  path = episode(tmp_path / 'ep')
  ep = NativeEpisode(path)
  assert ep.anchors == [0, 1, 2, 3]
  plan = op_to_ego_plan(ep.sample(1)['plan'])
  assert plan[0, 0] == pytest.approx(0)
  assert plan[-1, 0] == pytest.approx(20)
  assert plan[-1, 3] == pytest.approx(2)
  teacher = np.load(path / 'teacher.npy', mmap_mode='r+')
  teacher[2] = False
  teacher.flush()
  contacts = np.load(path / 'contacts.npy', mmap_mode='r+')
  contacts[205:] = 1
  contacts.flush()
  rehash(path)
  ep = NativeEpisode(path)
  assert ep.anchors == [1]


def test_discontinuous_times_and_split_overlap_rejected(tmp_path):
  a = episode(tmp_path / 'a')
  b = episode(tmp_path / 'b', split='val')
  with pytest.raises(ValueError, match='overlap'):
    load_episodes([a, b])
  state = np.load(a / 'states.npy', mmap_mode='r+')
  state[100:, 8] += .1
  state.flush()
  rehash(a)
  with pytest.raises(ValueError, match='discontinuous'):
    NativeEpisode(a)


def test_history_matches_runtime_slots_and_zero_padding(tmp_path):
  ep = NativeEpisode(episode(tmp_path / 'ep'))
  class Encoder:
    def encode(self, road, wide):
      assert torch.equal(road, wide)
      # Output depends on both members of each image pair.
      value = road[:, 0, 0, 0] * 100 + road[:, 6, 0, 0]
      return SimpleNamespace(hidden=value[:, None].expand(-1, 512))
  model = SimpleNamespace(base=Encoder())
  ts = TemporalState()
  expected = None
  for tick in range(13):
    k = tick // 4
    frame = np.full((6, 128, 256), k + 1, np.uint8)
    pair = ts.push_frames({'road': frame, 'wide': frame})
    pulse = np.zeros(8, np.float32)
    if 4 <= tick < 7:
      pulse[2] = 1
    desire = ts.push_desire(pulse)
    if tick == 12:
      expected = ts.features().copy()
    hidden = np.full(512, int(pair['road'][0, 0, 0]) * 100 + int(pair['road'][6, 0, 0]), np.float32)
    ts.push_hidden(hidden)
  actual = fresh_history(model, [ep], [0], [3], torch.device('cpu'), chunk_size=2).numpy()[0]
  np.testing.assert_array_equal(actual, expected)
  assert np.count_nonzero(actual[:-3]) == 0
  recorded = np.load(ep.path / 'desires.npy', mmap_mode='r+')
  recorded[3] = desire
  recorded.flush()
  rehash(ep.path)
  np.testing.assert_array_equal(NativeReplay([ep], 'train')[3]['desire'], desire)


def test_differentiable_follower_matches_runtime():
  rng = np.random.default_rng(2)
  vp = VehicleParams(lr=.8, lf=.75, max_brake=13., max_accel=7.)
  follower = PlanPurePursuit(vp)
  for _ in range(10):
    ego = rng.normal(size=(33, 15)).astype(np.float32)
    speed = float(rng.uniform(0, 20))
    op = torch.from_numpy(op_to_ego_plan(ego).copy())[None]
    steer = pp_steer_from_op_plan(op, torch.tensor([speed]), torch.tensor([vp.lr]), torch.tensor([vp.wheelbase]))
    accel = pp_accel_from_op_plan(op, torch.tensor([speed]), torch.tensor([vp.max_brake]), torch.tensor([vp.max_accel]))
    expected = follower(ego, speed)
    assert steer.item() == pytest.approx(expected[0], abs=3e-6)
    assert accel.item() == pytest.approx(expected[1], abs=3e-6)


def test_training_rejects_bad_weights_and_keeps_frozen_constants():
  from fsop.train.lab_train import configure, vision_l2sp

  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.base = torch.nn.Module()
      self.base.vision = torch.nn.Linear(1, 1)
      self.base.policy = torch.nn.Linear(1, 1)
      self.bev = torch.nn.Linear(1, 1)
      self.img = torch.nn.Linear(1, 1)
      self.constant = torch.nn.Parameter(torch.tensor([float('-inf')]), requires_grad=False)

    def set_trunk_trainable(self, mode):
      assert mode == 'all'
      self.base.vision.requires_grad_(True)

  m = Model()
  configure(m)
  assert not m.constant.requires_grad
  assert not any(p.requires_grad for p in m.bev.parameters())
  reference = {n: p.detach().clone() for n, p in m.base.vision.named_parameters()}
  with torch.no_grad():
    m.base.vision.weight.add_(2)
  assert vision_l2sp(m, reference, 0.1).item() == pytest.approx(.4)
  with torch.no_grad():
    m.base.policy.weight.fill_(float('nan'))
  with pytest.raises(ValueError, match='nonfinite trainable parameter'):
    configure(m)


def test_history_deduplicates_overlapping_pairs(tmp_path):
  ep = NativeEpisode(episode(tmp_path / 'ep'))

  class Encoder:
    def __init__(self):
      self.encoded = 0

    def encode(self, road, wide):
      assert road.is_contiguous(memory_format=torch.channels_last)
      assert wide.is_contiguous(memory_format=torch.channels_last)
      self.encoded += len(road)
      value = road[:, 0, 0, 0] * 100 + road[:, 6, 0, 0]
      return SimpleNamespace(hidden=value[:, None].expand(-1, 512))

  encoder = Encoder()
  frames = [3, 2, 0, 3]
  actual = fresh_history(SimpleNamespace(base=encoder), [ep], [0] * len(frames), frames, torch.device('cpu'),
                         chunk_size=2, channels_last=True).numpy()
  expected = np.zeros((len(frames), 24, 512), np.float32)
  for batch, frame in enumerate(frames):
    for slot, k in enumerate(range(frame - 24, frame)):
      if k >= 0:
        pair = ep.frame_pair('road', k)
        expected[batch, slot] = int(pair[0, 0, 0]) * 100 + int(pair[6, 0, 0])
  np.testing.assert_array_equal(actual, expected)
  assert encoder.encoded == 3  # Eight populated slots reference only three distinct pairs.


def test_rejects_failed_teacher_and_any_post_contact_anchor(tmp_path):
  path = episode(tmp_path / 'ep', n=241)
  contacts = np.load(path / 'contacts.npy', mmap_mode='r+')
  contacts[1:] = 1
  contacts.flush()
  rehash(path)
  ep = NativeEpisode(path)
  assert not ep.anchors
  assert ep.rejections['contact'] == 11
  assert sum(ep.rejections.values()) == len(ep.ticks)
  meta_path = path / 'meta.json'
  meta = json.loads(meta_path.read_text())
  good = {'result': 'finished', 'pass': True, 'complete': True, 'physical_cone_hits': 0, 'oc': 0}
  for change in ({'result': 'timeout'}, {'result': 'unsafe_stop'}, {'pass': False}, {'complete': False},
                 {'physical_cone_hits': 1}, {'oc': 1}, {}):
    meta['summary'] = {**good, **change} if change else {}
    meta_path.write_text(json.dumps(meta))
    ep = NativeEpisode(path)
    assert not ep.anchors
    assert ep.rejections['bad_teacher'] == len(ep.ticks)
  for missing in (None, 'absent'):
    if missing is None:
      meta['summary'] = None
    else:
      del meta['summary']
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='summary is required'):
      NativeEpisode(path)
  meta['summary'] = good
  meta_path.write_text(json.dumps(meta))
  assert NativeEpisode(path).rejections['contact'] == 11


def test_array_hashes_cover_data_and_detect_tampering(tmp_path):
  path = episode(tmp_path / 'ep')
  meta_path = path / 'meta.json'
  meta = json.loads(meta_path.read_text())
  hashes = {}
  for p in path.glob('*.npy'):
    with p.open('rb') as f:
      hashes[p.name] = hashlib.file_digest(f, 'sha256').hexdigest()
  meta['arrays_sha256'] = hashes
  meta_path.write_text(json.dumps(meta))
  assert NativeEpisode(path).anchors
  del meta['arrays_sha256']
  meta_path.write_text(json.dumps(meta))
  with pytest.raises(ValueError, match='cover all required arrays'):
    NativeEpisode(path)
  meta['arrays_sha256'] = hashes
  meta_path.write_text(json.dumps(meta))
  controls = np.load(path / 'controls.npy', mmap_mode='r+')
  controls[0, 0] = 1
  controls.flush()
  with pytest.raises(ValueError, match='SHA-256 mismatch for controls.npy'):
    NativeEpisode(path)
  del meta['arrays_sha256']['states.npy']
  meta_path.write_text(json.dumps(meta))
  with pytest.raises(ValueError, match='cover all required arrays'):
    NativeEpisode(path)


def test_source_archive_contains_matching_hashes(tmp_path):
  from fsop.train.lab_train import archive_source

  path = tmp_path / 'source.zip'
  info = archive_source(path)
  assert info['archive_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
  with zipfile.ZipFile(path) as archive:
    for name in ('fsop/data/native_replay.py', 'fsop/train/lab_train.py',
                 'fsop/model/build.py', 'fsop/eval/controller.py'):
      assert name in info['files_sha256']
      assert hashlib.sha256(archive.read(name)).hexdigest() == info['files_sha256'][name]
