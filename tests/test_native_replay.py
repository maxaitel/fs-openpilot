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


def test_balanced_validation_is_fixed_and_fp32(monkeypatch):
  from fsop.train import lab_train

  episodes = [SimpleNamespace(path=f'ep{i}', meta={'split': 'val'}) for i in range(8)]
  index = [(ei, k) for ei in range(8) for k in range(2 if ei == 0 else 30)]
  dataset = SimpleNamespace(episodes=episodes, index=index)
  selected = lab_train.validation_samples(dataset, 32)
  assert len(selected) == 8
  assert len(selected[0]) == 2
  assert sum(map(len, selected.values())) == 32
  assert selected == lab_train.validation_samples(dataset, 32)
  assert len({i for indices in selected.values() for i in indices}) == 32
  calls = []

  def fake_loss(model, ds, indices, device, w_steer, w_long, chunk_size, amp, channels_last, history_context):
    calls.append((amp, channels_last))
    value = torch.tensor(float(ds.index[indices[0]][0]))
    return value, {'plan_nll': value, 'steer': value * 2, 'longitudinal': value * 3}

  monkeypatch.setattr(lab_train, 'batch_loss', fake_loss)
  args = SimpleNamespace(batch_size=2, w_steer=1., w_long=1., history_chunk_size=64,
                         channels_last=False, amp=True, history_context=24)
  metrics = lab_train.validate(None, dataset, selected, torch.device('cpu'), args)
  assert all(not amp for amp, _ in calls)
  assert len(metrics['episodes']) == 8
  assert metrics['count'] == 32
  assert metrics['aggregate']['loss'] == pytest.approx(3.5)
  assert metrics['aggregate']['steer'] == pytest.approx(7.)
  with pytest.raises(ValueError, match='cover every validation episode'):
    lab_train.validation_samples(dataset, 7)
  empty = SimpleNamespace(episodes=episodes, index=index[:-30])
  with pytest.raises(ValueError, match='every validation episode'):
    lab_train.validation_samples(empty, 32)


def test_gradient_norms_match_and_reject_nonfinite():
  from fsop.train.lab_train import grad_norms

  model = torch.nn.Module()
  model.base = torch.nn.Module()
  model.base.vision = torch.nn.Linear(2, 1, bias=False)
  model.base.policy = torch.nn.Linear(2, 1, bias=False)
  model.base.vision.weight.grad = torch.tensor([[3., 4.]])
  model.base.policy.weight.grad = torch.tensor([[0., 12.]])
  assert grad_norms(model) == pytest.approx({'vision': 5., 'policy': 12.})
  model.base.policy.weight.grad[0, 0] = float('nan')
  with pytest.raises(ValueError, match='nonfinite gradient: base.policy.weight'):
    grad_norms(model)


def test_baseline_checkpoint_can_remain_selected(tmp_path, monkeypatch):
  from fsop.train import lab_train

  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.base = torch.nn.Module()
      self.base.vision = torch.nn.Linear(1, 1, bias=False)
      self.base.policy = torch.nn.Linear(1, 1, bias=False)
      self.bev = torch.nn.Linear(1, 1)
      self.img = torch.nn.Linear(1, 1)
      torch.nn.init.zeros_(self.base.vision.weight)
      torch.nn.init.zeros_(self.base.policy.weight)

    def set_trunk_trainable(self, mode):
      self.base.vision.requires_grad_(True)

  episodes = []
  for i, split in enumerate(('train', 'val', 'val')):
    path = tmp_path / f'ep{i}'
    path.mkdir()
    (path / 'meta.json').write_text('{}')
    episodes.append(SimpleNamespace(path=path, meta={'split': split}, anchors=[0, 1], ticks=[0, 4], rejections={}))

  class Replay:
    def __init__(self, eps, split):
      self.episodes = eps
      self.split = split
      self.index = [(i, k) for i, ep in enumerate(eps) if ep.meta['split'] == split for k in ep.anchors]

    def __len__(self):
      return len(self.index)

  def fake_loss(model, dataset, indices, device, w_steer, w_long, chunk_size, amp=False, channels_last=False,
                history_context=24):
    v, p = model.base.vision.weight, model.base.policy.weight
    loss = ((v - 1).square() + (p - 1).square()).sum() if dataset.split == 'train' else p.square().sum()
    return loss, {'plan_nll': loss, 'steer': loss * 0, 'longitudinal': loss * 0}

  init = tmp_path / 'init.pt'
  init.write_bytes(b'initial checkpoint')
  monkeypatch.setattr(lab_train, 'episode_paths', lambda **kwargs: [])
  monkeypatch.setattr(lab_train, 'load_episodes', lambda paths: episodes)
  monkeypatch.setattr(lab_train, 'NativeReplay', Replay)
  monkeypatch.setattr(lab_train, 'load_fs', lambda *args, **kwargs: Model())
  monkeypatch.setattr(lab_train, 'batch_loss', fake_loss)
  monkeypatch.setattr(lab_train, 'telemetry', lambda: None)
  args = SimpleNamespace(manifest=None, data='unused', init=str(init), out=str(tmp_path / 'run'), steps=1,
                         batch_size=2, lr=.1, val_every=1, val_samples=4, history_chunk_size=64,
                         w_steer=1., w_long=1., lr_policy=1., lr_trunk=1., l2sp=.001,
                         grad_clip=1., device='cpu', amp=False, channels_last=False, threads=1, seed=0,
                         history_context=24, dataset_verification=None)
  lab_train.run(args)
  receipt = json.loads((tmp_path / 'run' / 'run.json').read_text())['dataset_verification']
  assert receipt == {'status': 'unverified_development_data', 'path': None, 'sha256': None}
  selected = torch.load(tmp_path / 'run' / 'selected.pt', weights_only=False)
  assert selected['step'] == 0
  baseline = json.loads((tmp_path / 'run' / 'baseline_validation.json').read_text())
  assert baseline['step'] == 0
  assert baseline['aggregate']['loss'] == 0
  row = json.loads((tmp_path / 'run' / 'history.jsonl').read_text())
  assert row['step'] == 1
  assert row['val_loss'] > baseline['aggregate']['loss']


def test_context_pruning_preserves_active_features_output_and_gradients(tmp_path):
  ep = NativeEpisode(episode(tmp_path / 'ep'))

  class Encoder:
    def encode(self, road, wide):
      value = road[:, 0, 0, 0] * 100 + road[:, 6, 0, 0]
      return SimpleNamespace(hidden=value[:, None].expand(-1, 512))

  model = SimpleNamespace(base=Encoder())
  full = fresh_history(model, [ep], [0], [30], torch.device('cpu'), chunk_size=8)
  short = fresh_history(model, [ep], [0], [30], torch.device('cpu'), chunk_size=8, history_context=8)
  assert torch.equal(full[:, -8:], short[:, -8:])
  assert torch.count_nonzero(short[:, :-8]) == 0
  assert torch.count_nonzero(full[:, :-8]) > 0

  def fake_policy(history):
    weight = torch.tensor(0.125, requires_grad=True)
    current = torch.tensor(4., requires_grad=True)
    complete = torch.cat([history, current.expand(1, 1, 512)], dim=1)
    output = complete[:, -9:].sum() * weight
    gradients = torch.autograd.grad(output, (weight, current))
    return output.detach(), tuple(g.detach() for g in gradients)

  reference, grads = fake_policy(full)
  candidate, short_grads = fake_policy(short)
  assert torch.equal(candidate, reference)
  assert all(torch.equal(a, b) for a, b in zip(grads, short_grads, strict=True))
  perturbed = full.clone()
  perturbed[:, :16] += 99999
  changed, changed_grads = fake_policy(perturbed)
  assert torch.equal(changed, reference)
  assert all(torch.equal(a, b) for a, b in zip(grads, changed_grads, strict=True))


def test_context_proof_rejects_unknown_graph_and_changed_buffer(monkeypatch):
  from onnx2torch.node_converters.gather import OnnxGatherND
  from onnx2torch.node_converters.transpose import OnnxTranspose
  from torch import fx

  from fsop.train import lab_train

  root = torch.nn.Module()
  root.transpose = OnnxTranspose([1, 0, 2])
  root.gather = OnnxGatherND(batch_dims=0)
  root.register_buffer('indices', torch.arange(-9, 0, dtype=torch.int64).reshape(9, 1))
  graph = fx.Graph()
  features = graph.placeholder('features')
  graph.placeholder('desire')
  graph.placeholder('traffic')
  transposed = graph.call_module('transpose', (features,))
  indices = graph.get_attr('indices')
  gathered = graph.call_module('gather', (transposed, indices))
  graph.output(gathered)
  policy = fx.GraphModule(root, graph)
  model = SimpleNamespace(base=SimpleNamespace(policy=policy))
  monkeypatch.setattr(lab_train, 'sha256', lambda _: lab_train.PINNED_POLICY_SHA256)
  proof = lab_train.prove_policy_context(model, 'fake')
  assert proof['feature_slots'] == list(range(16, 25))
  with policy.graph.inserting_before(gathered):
    policy.graph.call_function(torch.neg, (features,))
  with pytest.raises(ValueError, match='unexpected consumers'):
    lab_train.prove_policy_context(model, 'fake')
  unused = next(n for n in policy.graph.nodes if n.op == 'call_function')
  policy.graph.erase_node(unused)
  policy.indices[0] = -10
  with pytest.raises(ValueError, match='indices differ'):
    lab_train.prove_policy_context(model, 'fake')
  monkeypatch.setattr(lab_train, 'sha256', lambda _: 'unrecognized')
  with pytest.raises(ValueError, match='pinned policy'):
    lab_train.prove_policy_context(model, 'fake')


def test_context_numeric_gate_compares_real_gradients(tmp_path):
  from scripts.benchmark_lab_training import history_context_gate

  class Vision(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, road, wide):
      value = road[:, 6, 0, 0].float() * self.scale / 255
      return SimpleNamespace(hidden=value[:, None].expand(-1, 512))

    encode = forward

  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.base = torch.nn.Module()
      self.base.vision = Vision()
      self.base.policy = torch.nn.Linear(1, 1, bias=False)
      self.base.encode = self.base.vision.encode
      self.bev = torch.nn.Linear(1, 1)
      self.img = torch.nn.Linear(1, 1)

    def set_trunk_trainable(self, mode):
      self.base.vision.requires_grad_(True)

    def forward_images(self, batch, heads):
      hidden = self.base.vision(batch['img'], batch['big_img']).hidden
      features = torch.cat([batch['history'], hidden[:, None]], dim=1)
      value = self.base.policy(features[:, -9:].mean(dim=(1, 2))[:, None])
      return {'hidden': hidden, 'plan_mu': value[:, :, None].expand(-1, 33, 15),
              'plan_logstd': value[:, :, None].expand(-1, 33, 15) * 0,
              'desire_state': value.expand(-1, 8)}

  ep = NativeEpisode(episode(tmp_path / 'ep'))
  pair = torch.from_numpy(ep.frame_pair('road', 30))[None]
  batch = {'img': pair, 'big_img': pair, 'speed': torch.tensor([2.]),
           'vehicle_lr': torch.tensor([.75]), 'vehicle_wheelbase': torch.tensor([1.55]),
           'vehicle_max_brake': torch.tensor([13.]), 'vehicle_max_accel': torch.tensor([7.]),
           'plan': torch.zeros((1, 33, 15)), 'steer_target': torch.zeros(1), 'accel_target': torch.zeros(1)}
  result = history_context_gate(Model().eval(), [ep], [0], [30], batch, torch.device('cpu'))
  assert result['passed']
  assert result['active_features_bitwise_equal']
  assert result['comparisons']['pruned8']['gradient_max_abs'] <= 1e-5
  assert result['comparisons']['perturbed_early16']['gradient_max_abs'] <= 1e-5

  class LeakyModel(Model):
    def forward_images(self, batch, heads):
      output = super().forward_images(batch, heads)
      leak = batch['history'][:, :16].mean(dim=(1, 2))[:, None, None]
      output['plan_mu'] = output['plan_mu'] + leak
      return output

  failed = history_context_gate(LeakyModel().eval(), [ep], [0], [30], batch, torch.device('cpu'))
  assert not failed['passed']
  assert not failed['comparisons']['perturbed_early16']['passed']


def test_native_manifest_rejects_incomplete_tampered_and_escaping_entries(tmp_path):
  from fsop.data.native_replay import episode_paths

  root = tmp_path / 'dataset'
  root.mkdir()
  membership = 'a' * 64
  rows = []
  for name, split in (('alpha', 'train'), ('beta', 'val')):
    path = root / name
    path.mkdir()
    meta = {'id': name, 'split': split, 'membership_sha256': membership}
    data = json.dumps(meta).encode()
    (path / 'meta.json').write_bytes(data)
    rows.append({'path': name, 'id': name, 'split': split, 'meta_sha256': hashlib.sha256(data).hexdigest()})
  manifest = {'schema_version': 1, 'complete': True, 'expected_episodes': 2,
              'membership_sha256': membership, 'episodes': rows}
  path = root / 'manifest.json'

  def write():
    path.write_text(json.dumps(manifest))

  write()
  assert episode_paths(manifest=path) == [(root / 'alpha').resolve(), (root / 'beta').resolve()]
  for key, invalid in (('complete', False), ('expected_episodes', 3), ('schema_version', 2),
                       ('membership_sha256', 'bad')):
    old = manifest[key]
    manifest[key] = invalid
    write()
    with pytest.raises(ValueError, match='incomplete or has invalid'):
      episode_paths(manifest=path)
    manifest[key] = old
  rows[0]['split'] = 'val'
  write()
  with pytest.raises(ValueError, match='split mismatch'):
    episode_paths(manifest=path)
  rows[0]['split'] = 'train'
  rows[0]['path'] = '../outside'
  write()
  with pytest.raises(ValueError, match='escapes dataset root'):
    episode_paths(manifest=path)
  rows[0]['path'] = 'alpha'
  (root / 'alpha' / 'meta.json').write_text('{}')
  write()
  with pytest.raises(ValueError, match='metadata SHA-256 mismatch'):
    episode_paths(manifest=path)
  data = json.dumps({'id': 'alpha', 'split': 'train', 'membership_sha256': 'b' * 64}).encode()
  (root / 'alpha' / 'meta.json').write_bytes(data)
  rows[0]['meta_sha256'] = hashlib.sha256(data).hexdigest()
  write()
  with pytest.raises(ValueError, match='membership mismatch'):
    episode_paths(manifest=path)


def test_verification_receipt_binds_manifest_and_membership(tmp_path):
  from fsop.train.lab_train import dataset_verification

  manifest = tmp_path / 'manifest.json'
  membership = 'a' * 64
  manifest.write_text(json.dumps({'schema_version': 1, 'membership_sha256': membership}))
  manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
  receipt_path = tmp_path / 'verification.json'
  with pytest.raises(ValueError, match='receipt missing'):
    dataset_verification(manifest)
  receipt = {'passed': True, 'manifest_sha256': manifest_sha, 'membership_sha256': membership}
  receipt_path.write_text(json.dumps(receipt))
  result = dataset_verification(manifest)
  assert result['status'] == 'verified'
  assert result['sha256'] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
  assert dataset_verification(manifest, receipt_path) == result
  for key, value in (('passed', False), ('manifest_sha256', 'b' * 64), ('membership_sha256', 'c' * 64)):
    changed = {**receipt, key: value}
    receipt_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='does not match'):
      dataset_verification(manifest)
  receipt_path.write_text(json.dumps(receipt))
  manifest.write_text(manifest.read_text() + ' ')
  with pytest.raises(ValueError, match='does not match'):
    dataset_verification(manifest)
