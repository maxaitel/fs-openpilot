"""CPU contract tests. Fixtures do not certify real graph training or physical driving."""
from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import joint_core as core
import joint_data as data_module
import train_joint as train_module
from joint_data import LegacySpec, TrialSampler, batch_input, close_episode, legacy_identity
from train_joint import check_recipe, command_loss, scores, require_training_parent


class FixturePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_node_linear_14 = nn.Linear(512, 256)
        self.p_node_linear_16 = nn.Linear(256, 256)
        self.p_node_linear_17 = nn.Linear(256, 256)
        self.p_node_linear_20 = nn.Linear(256, 990)

    def forward(self, history, desire, traffic):
        x = torch.relu(self.p_node_linear_14(history[:, -1] + history[:, -2]))
        x = torch.relu(x + self.p_node_linear_17(torch.relu(self.p_node_linear_16(x))))
        return self.p_node_linear_20(x), x[:, :8]


class FixtureBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(4, 512)
        self.policy = FixturePolicy()

    def encode(self, road, wide):
        return SimpleNamespace(hidden=self.vision((road + wide) / 2))

    def policy_raw(self, history, desire, traffic):
        return self.policy(history, desire, traffic)


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = FixtureBase()


class FixtureEpisode:
    def __init__(self):
        self.ticks = np.arange(12) * 4
        self.teacher = np.ones(48, dtype=bool)
        self.states = np.ones((48, 9), dtype=np.float32)
        self.controls = np.tile(np.array([[.02, .2]], dtype=np.float32), (48, 1))
        self.desires = np.zeros((12, 25, 8), dtype=np.float32)

    def frame_pair(self, camera, frame):
        return np.array([frame, frame + 1, 2, 3], dtype=np.float32) / 12


class JointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.model = FixtureModel().eval()
        core.set_training_scope(self.model)
        self.head = core.CommandHead().eval()
        self.dataset = SimpleNamespace(index=[(0, 8), (0, 9)], episodes=[FixtureEpisode()])

    def test_only_six_scope_tensors_are_trainable(self):
        self.assertEqual(set(core.set_training_scope(self.model)), set(core.SCOPE))
        self.assertEqual(sum(p.numel() for p in self.model.parameters() if p.requires_grad), 262912)
        self.assertEqual(sum(p.numel() for p in self.head.parameters()), 33538)
        self.assertTrue(all(not m.training for m in self.model.modules()))
        self.model.base.policy.p_node_linear_14 = nn.Linear(512, 255)
        with self.assertRaisesRegex(ValueError, 'suffix tensor differs'):
            core.set_training_scope(self.model)

    def test_fresh_history_gradient_reaches_all_six_and_head(self):
        frozen = core.digest_state(self.model.state_dict())
        latent, prop, target, _ = batch_input(self.model, self.dataset, [0, 1], torch.device('cpu'))
        self.assertIsNotNone(latent.grad_fn)
        self.assertFalse(torch.is_inference(latent))
        command_loss(self.head, latent, prop, target).backward()
        for name, parameter in self.model.named_parameters():
            if name in core.SCOPE:
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.norm()), 0)
            else:
                self.assertIsNone(parameter.grad)
        self.assertTrue(all(p.grad is not None and float(p.grad.norm()) > 0 for p in self.head.parameters()))
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad] + list(self.head.parameters()), lr=.001, weight_decay=0)
        optimizer.step()
        self.assertEqual(set(core.check_frozen_complement(self.model, frozen, training=True)), set(core.SCOPE))
        with torch.no_grad():
            self.model.base.vision.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            core.check_frozen_complement(self.model, frozen, training=True)

    def test_updated_payload_roundtrip_preserves_commands(self):
        original = {'step': 500, 'model': core.snapshot_state(self.model.state_dict()), 'config': {},
                    'controller_contract': {'old': 'diagnostic'}, 'controller_intervention': {'source': 'original'}}
        with torch.no_grad():
            self.model.base.policy.p_node_linear_14.weight.add_(.01)
            latent, prop, _, _ = batch_input(self.model, self.dataset, [0, 1], torch.device('cpu'))
            expected = self.head(latent, prop)
        payload = core.snapshot_payload(original, self.model, 1)
        self.assertNotEqual(core.digest_state(original['model']), core.digest_state(payload['model']))
        buffer = io.BytesIO()
        torch.save({'base_payload': payload, 'head_state': self.head.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location='cpu', weights_only=False)
        restored = FixtureModel().requires_grad_(False).eval()
        restored.load_state_dict(saved['base_payload']['model'], strict=True)
        restored_head = core.CommandHead().eval()
        restored_head.load_state_dict(saved['head_state'], strict=True)
        with torch.no_grad():
            latent, prop, _, _ = batch_input(restored, self.dataset, [0, 1], torch.device('cpu'))
            self.assertTrue(torch.equal(expected, restored_head(latent, prop)))
            self.model.base.policy.p_node_linear_14.weight.add_(1)
        self.assertNotEqual(core.digest_state(self.model.state_dict()), core.digest_state(payload['model']))
        self.assertEqual(payload['controller_intervention'], original['controller_intervention'])
        self.assertEqual(payload['step'], 1)

    def test_inference_context_is_not_a_training_forward(self):
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, 'ordinary frozen encoder tensors'):
            batch_input(self.model, self.dataset, [0, 1], torch.device('cpu'))

    def test_non_teacher_same_tick_rejected(self):
        self.dataset.episodes[0].teacher[32] = False
        with self.assertRaisesRegex(ValueError, 'same-tick'):
            batch_input(self.model, self.dataset, [0, 1], torch.device('cpu'))

    def test_sampler_excludes_both_recovery_pools(self):
        dataset = list(range(100))
        legacy = SimpleNamespace(recovery_indices={0, 1}, groups=[{'indices': [0, 1]}])
        sampler = TrialSampler(dataset, legacy, [{'indices': list(range(2, 12))}, {'indices': list(range(12, 22))}])
        rng = np.random.default_rng(0)
        for _ in range(5):
            ids = sampler.draw(rng)
            self.assertEqual(sum(int(i) in sampler.old_indices for i in ids), 16)
            self.assertEqual(sum(int(i) in sampler.new_indices for i in ids), 16)
            self.assertEqual(sum(int(i) in sampler.other for i in ids), 96)
        self.assertEqual(int(sampler.counts.sum()), 640)

    def test_sampler_rng_sequence_matches_original_algorithm(self):
        dataset = list(range(100))
        legacy = SimpleNamespace(recovery_indices={0, 1}, groups=[{'indices': [0, 1]}])
        new = [{'indices': list(range(2, 12))}, {'indices': list(range(12, 22))}]
        sampler = TrialSampler(dataset, legacy, new)
        actual_rng, expected_rng = np.random.default_rng(7), np.random.default_rng(7)
        for _ in range(5):
            draws = []
            for groups in (legacy.groups, new):
                choices = expected_rng.integers(len(groups), size=16)
                draws.extend(groups[int(g)]['indices'][int(expected_rng.integers(len(groups[int(g)]['indices'])))]
                             for g in choices)
            result = np.concatenate((np.asarray(draws, dtype=np.int64),
                                     np.arange(22, 100)[expected_rng.integers(78, size=96)]))
            expected_rng.shuffle(result)
            np.testing.assert_array_equal(sampler.draw(actual_rng), result)

    def test_episode_mmaps_close_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'road.npy'
            np.save(path, np.arange(4))
            episode = SimpleNamespace(road=np.load(path, mmap_mode='r'))
            close_episode(episode)
            close_episode(episode)
            self.assertTrue(episode.road._mmap.closed)

    def test_legacy_spec_binds_index_metadata_and_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_path = root / 'episode'
            episode_path.mkdir()
            meta = {'id': 'case', 'split': 'train'}
            (episode_path / 'meta.json').write_text(json.dumps(meta))
            (root / 'manifest.json').write_text('{}')
            contract = {'checkpoint_sha256': 'a' * 64}
            (root / 'recovery.json').write_text(json.dumps({'recovery_case_ids': ['case'],
                                                            'collection_by_case': {'case': contract}}))
            gates = root / 'gates'
            gates.mkdir()
            frames = list(range(8))
            receipt = {'recovery_frame_indices': frames, 'recovery_ticks': frames,
                       'takeover_tick': 0, 'terminal_start_tick': 10}
            gate = gates / 'case.json'
            gate.write_text(json.dumps(receipt))
            recipe = {'manifest': str(root / 'manifest.json'), 'recovery_recipe': str(root / 'recovery.json'),
                      'recovery_gates': str(gates), 'base_checkpoint_sha256': 'b' * 64}
            (root / 'recipe.json').write_text(json.dumps(recipe))
            episode = SimpleNamespace(path=episode_path, meta=meta, ticks=np.arange(9))
            dataset = type('FixtureDataset', (SimpleNamespace,), {'__len__': lambda self: len(self.index)})(
                index=[(0, i) for i in range(9)], episodes=[episode])
            val = SimpleNamespace(index=[(0, 8)])
            group = {'case_id': 'case', 'episode': 0, 'takeover_tick': 0, 'gate_path': str(gate.resolve()),
                     'gate_sha256': data_module.sha256(gate), 'terminal_start_tick': 10, 'indices': frames,
                     'frames': frames, 'ticks': frames, 'collection_checkpoint_sha256': 'a' * 64,
                     'collection_contract': contract}
            protocol = {'recipe_path': str((root / 'recovery.json').resolve()), 'recipe_sha256': data_module.sha256(root / 'recovery.json'),
                        'provenance_mode': 'per_case_v2', 'collection_policy_checkpoint_sha': None,
                        'optimization_init_sha256': 'b' * 64, 'window_ticks': 40, 'control_hz': 20,
                        'with_replacement': True, 'recovery_draws_per_batch': 16, 'other_draws_per_batch': 112,
                        'recovery_episode_probability': 1.0, 'other_anchor_probability': 1.0,
                        'recovery_anchor_count': 8, 'other_anchor_count': 1,
                        'recovery_episodes': [dict(group, anchor_probability=.125)],
                        'other_episodes': [{'case_id': 'case', 'episode': 0, 'frames': [8]}]}
            report = {'recipe_file_sha256': data_module.sha256(root / 'recipe.json'),
                      'identity': legacy_identity(dataset, val, recipe), 'groups': [group],
                      'recovery_indices': frames, 'protocol': protocol}
            with patch.object(data_module, 'HERE', root), patch.object(data_module, 'RecoverySampler', side_effect=AssertionError('parent called sampler')):
                spec = LegacySpec(dataset, val, recipe, report)
                self.assertEqual(spec.recovery_indices, set(frames))
                altered = copy.deepcopy(report)
                altered['groups'][0]['indices'][0] = 8
                with self.assertRaisesRegex(ValueError, 'group binding'):
                    LegacySpec(dataset, val, recipe, altered)
                dataset.index[0] = (0, 8)
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    LegacySpec(dataset, val, recipe, report)
                dataset.index[0] = (0, 0)
                episode.meta['id'] = 'tampered'
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    LegacySpec(dataset, val, recipe, report)

    def test_failure_and_live_parent_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'train_failure.json').write_text('{}')
            for function, args in ((train_module.checked_audit, (out, {})),
                                   (train_module.postflight_run, (out, {}, torch.device('cpu'))),
                                   (train_module.selected_package, (out,))):
                with self.assertRaisesRegex(ValueError, 'failed attempt cannot resume'):
                    function(*args)
            with patch('sys.argv', ['train_joint.py', 'postflight', '--out', str(out), '--device', 'cpu']):
                with self.assertRaisesRegex(ValueError, 'failed attempt cannot resume'):
                    train_module.main()
            self.assertFalse((out / 'postflight_attempt.json').exists())
        record = {'process_id': os.getppid()}
        require_training_parent(record)
        with patch.object(train_module.os, 'getppid', return_value=record['process_id'] + 1):
            with self.assertRaisesRegex(ValueError, 'live training parent'):
                require_training_parent(record)
        with patch.object(train_module.os, 'kill', side_effect=ProcessLookupError):
            with self.assertRaisesRegex(ValueError, 'live training parent'):
                require_training_parent(record)

    def test_legacy_child_runs_before_parent_opens_pool(self):
        with patch.object(data_module, 'membership', return_value={}), \
             patch.object(data_module, 'verified_receipts', return_value=({}, {}, {})), \
             patch.object(data_module, 'isolated_report', return_value={}) as child, \
             patch.object(data_module, 'episode_paths', side_effect=RuntimeError('stop after child')), \
             patch.object(data_module, 'RecoverySampler', side_effect=AssertionError('parent called sampler')) as legacy:
            with self.assertRaisesRegex(RuntimeError, 'stop after child'):
                data_module.JointData({'manifest': 'unused'}, 'cpu')
            child.assert_called_once()
            legacy.assert_not_called()

    def test_legacy_child_closes_pool_on_constructor_failure(self):
        class Mapping:
            closed = False
            def close(self):
                self.closed = True
        mapping = Mapping()
        def initialize(episode, path):
            episode.road = SimpleNamespace(_mmap=mapping)
        class Replay:
            def __init__(self, episodes, split):
                self.split = split
            def __len__(self):
                return 7006 if self.split == 'train' else 2067
        recipe = {'manifest': 'unused', 'recovery_recipe': 'unused', 'recovery_gates': 'unused',
                  'base_checkpoint_sha256': 'unused'}
        with patch.object(data_module, 'read_json', return_value=recipe), \
             patch.object(data_module, 'episode_paths', return_value=['unused']), \
             patch.object(data_module.NativeEpisode, '__init__', initialize), \
             patch.object(data_module, 'NativeReplay', Replay), \
             patch.object(data_module, 'RecoverySampler', side_effect=ValueError('legacy failed')):
            with self.assertRaisesRegex(ValueError, 'legacy failed'):
                data_module.legacy_report('unused')
        self.assertTrue(mapping.closed)

    def test_real_child_accepts_its_live_parent(self):
        code = ('import sys; from train_joint import require_training_parent; '
                'require_training_parent({"process_id": int(sys.argv[1])})')
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(Path(__file__).parent.resolve()), env.get('PYTHONPATH', '')))
        subprocess.run([sys.executable, '-c', code, str(os.getpid())], check=True, env=env)

    def test_interrupted_stage_attempt_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'condition_attempt.json').write_text('{}')
            with patch('sys.argv', ['train_joint.py', 'condition', '--out', str(out), '--device', 'cpu']):
                with self.assertRaisesRegex(FileExistsError, 'interrupted condition attempt'):
                    train_module.main()
            with patch('sys.argv', ['train_joint.py', 'overfit', '--out', str(out), '--device', 'cpu']):
                with self.assertRaisesRegex(FileExistsError, 'interrupted condition attempt'):
                    train_module.main()

    def test_missing_prepared_graph_does_not_fetch(self):
        if (Path('artifacts/models/prep') / 'vision_fp32.onnx').exists():
            self.skipTest('prepared graph is present')
        path = Path('runs/direct_command_trial3_archive/step_500.pt')
        if not path.exists():
            self.skipTest('actual local checkpoint unavailable')
        payload = torch.load(path, map_location='cpu', weights_only=False)['base_payload']
        with self.assertRaisesRegex(FileNotFoundError, 'pinned prepared graph unavailable'):
            core.model_from_payload(payload, 'cpu')

    def test_recipe_and_gates_reject_changed_semantics(self):
        recipe = json.loads((Path(__file__).parent / 'recipe.json').read_text())
        check_recipe(recipe)
        recipe['learning_rate'] = .0001
        with self.assertRaisesRegex(ValueError, 'more than architecture'):
            check_recipe(recipe)
        self.assertFalse(scores(np.tile([.001, .8], (20, 1)))['passed'])
        self.assertTrue(scores(np.tile([.005, .1], (20, 1)))['passed'])
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            core.validate_package({'kind': 'direct_native_actuator_v1'})

    def test_actual_saved_state_manifest_rejects_tampering(self):
        path = Path('runs/direct_command_trial3_archive/step_500.pt')
        if not path.exists():
            self.skipTest('actual local 417-state checkpoint unavailable')
        old = torch.load(path, map_location='cpu', weights_only=False)
        payload = old['base_payload']
        state = payload['model']
        original = core.digest_state(state)
        original_digest = core.state_model_digest(state)
        with torch.no_grad():
            state['base.policy.p_node_linear_14.bias'][0] += .01
        p = {'step': 1, 'training_scope': core.SCOPE, 'base_state_digest': core.digest_state(state),
             'head_state_digest': core.digest_state(self.head.state_dict()),
             'base_model_digest': core.state_model_digest(state), 'original_base_model_digest': original_digest,
             'original_base_state_digest': original, 'frozen_base_state_unchanged': True,
             'frozen_base_state_count': 411, 'changed_suffix_keys': ['base.policy.p_node_linear_14.bias'],
             'original_payload_metadata': copy.deepcopy({k: v for k, v in payload.items() if k != 'model'})}
        payload['step'] = 1
        package = {'kind': core.KIND, 'base_payload': payload, 'head_state': self.head.state_dict(),
                   'command_contract': core.CONTRACT, 'recipe': {}, 'provenance': p}
        core.validate_package(package)
        with torch.no_grad():
            state['base.policy.p_node_linear_20.bias'][0] += 1
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            core.validate_package(package)
        p['base_state_digest'] = core.digest_state(state)
        p['base_model_digest'] = core.state_model_digest(state)
        with self.assertRaisesRegex(ValueError, 'forbidden saved'):
            core.validate_package(package)


if __name__ == '__main__':
    unittest.main(verbosity=2)
