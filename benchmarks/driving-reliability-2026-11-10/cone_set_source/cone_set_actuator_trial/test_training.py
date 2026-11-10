from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import cone_core as core
import train_set as train
from test_core import DiagnosticModel, head_inputs


class Episode:
    def __init__(self):
        self.ticks = np.arange(8) * 4
        self.teacher = np.ones(32, bool)
        self.states = np.ones((32, 9), np.float32)
        self.controls = np.tile(np.array([.02, .3], np.float32), (32, 1))
        self.desires = np.zeros((8, 25, 8), np.float32)
        calibration = np.eye(4)
        calibration[2, 3] = 1.
        self.meta = {'id': 'fixture', 'calibration': {'T_ego_cam': calibration.tolist()},
                     'collection_policy': {'mode': 'teacher'}, 'teacher_config': {'name': 'fixture'}}

    def frame_pair(self, camera, frame):
        return np.full((12, 128, 256), frame, np.float32)


class Dataset(SimpleNamespace):
    def __len__(self):
        return len(self.index)


def completed_stage(out, stage, receipt):
    train.write_json(out / (stage + '_attempt.json'), {'stage': stage, 'process_id': os.getpid()})
    train.write_json(out / (stage + '.json'), receipt)
    train.mark_complete(out, stage)


def completed_training(out):
    completed_stage(out, 'audit', {'frozen': True})
    condition = {'passed': True, 'audit_sha256': train.sha256(out / 'audit.json')}
    completed_stage(out, 'condition', condition)
    completed_stage(out, 'overfit', {**condition, 'condition_sha256': train.sha256(out / 'condition.json')})
    package = out / 'selected.pt'
    package.write_bytes(b'fixed selected package')
    selected = {'step': 100, 'package': str(package), 'package_sha256': train.sha256(package)}
    train.write_json(out / 'postflight.json', condition)
    result = {**train.training_prerequisites(out), 'passed': True, 'postflight_complete': True,
              'selected': selected, 'postflight_sha256': train.sha256(out / 'postflight.json')}
    completed_stage(out, 'train', result)
    train.write_json(out / 'postflight_attempt.json', {'stage': 'postflight', 'process_id': os.getpid()})
    train.mark_complete(out, 'postflight')
    return selected


def replace_completed_receipt(out, stage, **changes):
    receipt = {**train.read_json(out / (stage + '.json')), **changes}
    (out / (stage + '.json')).write_text(json.dumps(receipt))
    marker = train.read_json(out / (stage + '_complete.json'))
    marker['receipt_sha256'] = train.sha256(out / (stage + '.json'))
    (out / (stage + '_complete.json')).write_text(json.dumps(marker))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_recipe_keeps_inherited_numerical_contract(self):
        recipe = train.read_json(train.RECIPE)
        train.check_recipe(recipe)
        recipe['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'inherited'):
            train.check_recipe(recipe)

    def test_batch_recomputes_detached_image_path_and_same_tick_targets(self):
        model = DiagnosticModel().requires_grad_(False).eval()
        ep = Episode()
        ep.controls[8] = [.12, .8]
        data = SimpleNamespace(index=[(0, 2)], episodes=[ep])
        first = train.batch_input(model, data, [0], 'cpu')
        second = train.batch_input(model, data, [0], 'cpu')
        self.assertEqual(model.base.calls, 2)
        self.assertFalse(first.tokens.requires_grad or torch.is_inference(first.tokens))
        np.testing.assert_allclose(first.target[0], [.12, .8])
        self.assertTrue(torch.equal(first.tokens, second.tokens))
        self.assertGreater(int(first.mask.sum()), 0)
        ep.teacher[8] = False
        with self.assertRaisesRegex(ValueError, 'same-tick'):
            train.batch_input(model, data, [0], 'cpu')

    def test_disposable_update_reaches_all_eight_parameters_and_improves(self):
        head = core.ConeSetHead()
        inputs = head_inputs(4)
        with torch.no_grad():
            target = head(*inputs) + torch.tensor([.02, .5])
        batch = train.Batch(*inputs, target)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0.)
        named = dict(head.named_parameters())
        before = {name: p.detach().clone() for name, p in named.items()}
        first = float(train.command_loss(head, batch).detach())
        repeat = float(train.command_loss(head, batch).detach())
        with patch.object(train, 'batch_input', return_value=batch) as fresh:
            result = train.update(None, head, optimizer, named, None, [0, 1, 2, 3], 'cpu')
        fresh.assert_called_once()
        after = float(train.command_loss(head, batch).detach())
        self.assertGreater(first - after, 5 * abs(first - repeat))
        for name, parameter in named.items():
            self.assertGreater(result['gradient_norms'][name], 0)
            self.assertFalse(torch.equal(before[name], parameter))
        restored = core.ConeSetHead()
        restored.load_state_dict(head.state_dict())
        self.assertTrue(torch.equal(head(*inputs), restored(*inputs)))

    def test_all_empty_conditioning_cannot_claim_token_connectivity(self):
        head = core.ConeSetHead()
        tokens, mask, desire, proprio = head_inputs()
        loss = head(tokens, torch.zeros_like(mask), desire, proprio).sum()
        loss.backward()
        self.assertEqual(float(head.token1.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(head.token2.weight.grad.abs().sum()), 0.)

    def test_exact_collision_lower_bound_and_quantile(self):
        tokens, mask, desire, proprio = head_inputs(2)
        for value in (tokens, mask, desire, proprio):
            value[1] = value[0]
        batch = train.Batch(tokens, mask, desire, proprio, torch.tensor([[0., 0.], [.1, 1.]]))
        ep = Episode()
        data = SimpleNamespace(index=[(0, 0), (0, 1)], episodes=[ep])
        result = train.collision_screen(batch, data, [0, 1])
        self.assertFalse(result['passed'])
        np.testing.assert_allclose(result['median_mae_lower_bound'], [.05, .5])
        np.testing.assert_allclose(result['minimum_achievable_p95'], [.05, .5])
        self.assertEqual(len(result['groups']), 1)
        self.assertEqual(len(result['groups'][0]['members']), 2)
        labels = [0.] * 30 + [10.] * 2
        self.assertAlmostEqual(train.minimum_p95(labels, [list(range(32))]), 4.5)
        self.assertEqual(train.minimum_p95(labels, [[i] for i in range(32)]), 0.)

    def test_source_archive_has_explicit_external_mapping_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / 'external recipe.json'
            source.write_text('{"canonical": true}\n')
            sources = {str(source): train.sha256(source)}
            mapping = train.archive_sources(out, sources)
            audit = {'sources': sources, 'source_archive_members': mapping,
                     'source_archive_sha256': train.sha256(out / 'source.zip')}
            self.assertEqual(set(mapping), set(sources))
            train.check_archive(out, audit)
            with (out / 'source.zip').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'archive changed'):
                train.check_archive(out, audit)

    def test_attempt_markers_reject_interruption_failure_and_reentry(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            train.begin_stage(out, 'audit')
            with self.assertRaisesRegex(FileExistsError, 'interrupted'):
                train.begin_stage(out, 'condition')
            (out / 'audit.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                train.begin_stage(out, 'audit')
            (out / 'condition_failure.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'cannot resume'):
                train.begin_stage(out, 'condition')

    def test_train_attempt_exception_only_allows_live_postflight(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            completed_stage(out, 'audit', {'frozen': True})
            condition = {'passed': True, 'audit_sha256': train.sha256(out / 'audit.json')}
            completed_stage(out, 'condition', condition)
            completed_stage(out, 'overfit', {**condition, 'condition_sha256': train.sha256(out / 'condition.json')})
            train.begin_stage(out, 'train')
            train.reject_interrupted_attempt(out, 'postflight')
            with self.assertRaises(FileExistsError):
                train.begin_stage(out, 'cpu_validate')
            with self.assertRaises(FileNotFoundError):
                train.begin_stage(out, 'postflight')
            self.assertFalse((out / 'postflight_attempt.json').exists())
            train.write_json(out / 'training_updates.json', {**train.training_prerequisites(out),
                             'process_id': os.getpid(), 'updates_complete': True, 'postflight_complete': False})
            with self.assertRaisesRegex(ValueError, 'live training parent'):
                train.begin_stage(out, 'postflight')
            self.assertFalse((out / 'postflight_attempt.json').exists())
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join((str(train.HERE), str(Path.cwd()), str(Path.cwd() / 'src')))
            code = 'import sys; from pathlib import Path; import train_set; train_set.begin_stage(Path(sys.argv[1]), "postflight")'
            result = subprocess.run([sys.executable, '-c', code, str(out)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / 'postflight_attempt.json').is_file())

    def test_prerequisite_chain_rejects_mutation_after_wiring_and_completion_check(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            train.write_json(out / 'audit.json', {'frozen': True})
            condition = {'passed': True, 'audit_sha256': train.sha256(out / 'audit.json')}
            train.write_json(out / 'condition.json', condition)
            wiring = {**condition, 'condition_sha256': train.sha256(out / 'condition.json')}
            train.write_json(out / 'overfit.json', wiring)
            for stage in ('audit', 'condition', 'overfit'):
                train.write_json(out / (stage + '_attempt.json'), {'stage': stage, 'process_id': os.getpid()})
                train.mark_complete(out, stage)
            bound = train.training_prerequisites(out)
            train.check_prerequisites(out, bound)
            (out / 'condition.json').write_text(json.dumps({**condition, 'modified': True}))
            with self.assertRaisesRegex(ValueError, 'completed receipt changed'):
                train.check_prerequisites(out, bound)
            marker = train.read_json(out / 'condition_complete.json')
            marker['receipt_sha256'] = train.sha256(out / 'condition.json')
            (out / 'condition_complete.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError, 'changed after wiring'):
                train.check_prerequisites(out, bound)
            wiring['condition_sha256'] = train.sha256(out / 'condition.json')
            (out / 'overfit.json').write_text(json.dumps(wiring))
            marker = train.read_json(out / 'overfit_complete.json')
            marker['receipt_sha256'] = train.sha256(out / 'overfit.json')
            (out / 'overfit_complete.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError, 'changed after binding'):
                train.check_prerequisites(out, bound)

    def test_report_without_completion_cannot_start_downstream_and_changed_report_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            train.begin_stage(out, 'audit')
            train.write_json(out / 'audit.json', {'result': 'report written while pool still open'})
            with self.assertRaisesRegex(FileExistsError, 'lacks final completion marker'):
                train.begin_stage(out, 'condition')
            train.mark_complete(out, 'audit')
            train.reject_interrupted_attempt(out, 'condition')
            (out / 'audit.json').write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'completed receipt changed'):
                train.begin_stage(out, 'condition')

    def test_child_postflight_completion_does_not_finalize_parent_train(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for stage in ('train', 'postflight'):
                train.write_json(out / (stage + '_attempt.json'), {'stage': stage, 'process_id': os.getpid()})
                train.write_json(out / (stage + '.json'), {'passed': True})
            train.mark_complete(out, 'postflight')
            with self.assertRaisesRegex(FileExistsError, 'interrupted train'):
                train.selected_package(out)
            (out / 'train.json').write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'train receipt changed'):
                train.require_complete(out, 'postflight')

    def test_stage_admission_requires_predecessor_before_creating_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            completed_stage(out, 'audit', {'passed': True})
            for stage in ('overfit', 'train', 'cpu_validate', 'visual', 'runtime'):
                with self.subTest(stage=stage), self.assertRaises(FileExistsError):
                    train.begin_stage(out, stage)
                self.assertFalse((out / (stage + '_attempt.json')).exists())

    def test_unknown_stage_cannot_write_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(ValueError):
                train.begin_stage(out, 'unsupported')
            self.assertEqual(list(out.iterdir()), [])

    def test_condition_and_overfit_cannot_both_pass_admission_race(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            completed_stage(out, 'audit', {'passed': True})
            barrier = threading.Barrier(2)
            reject = train.reject_interrupted_attempt
            def simultaneous_scan(path, stage):
                reject(path, stage)
                barrier.wait(timeout=5)
            with patch.object(train, 'reject_interrupted_attempt', side_effect=simultaneous_scan), \
                    ThreadPoolExecutor(max_workers=2) as workers:
                condition = workers.submit(train.begin_stage, out, 'condition')
                overfit = workers.submit(train.begin_stage, out, 'overfit')
                condition.result(timeout=5)
                with self.assertRaises(FileExistsError):
                    overfit.result(timeout=5)
            self.assertTrue((out / 'condition_attempt.json').is_file())
            self.assertFalse((out / 'overfit_attempt.json').exists())

    def test_selected_package_rechecks_training_chain_after_stage_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            selected = completed_training(out)
            train.begin_stage(out, 'cpu_validate')
            self.assertEqual(train.selected_package(out), selected)
            condition = train.read_json(out / 'condition.json')
            (out / 'condition.json').write_text(json.dumps({**condition, 'modified': True}))
            with self.assertRaisesRegex(ValueError, 'completed receipt changed'):
                train.selected_package(out)
            replace_completed_receipt(out, 'condition', modified=True)
            with self.assertRaisesRegex(ValueError, 'changed after wiring'):
                train.selected_package(out)
            replace_completed_receipt(out, 'overfit', condition_sha256=train.sha256(out / 'condition.json'))
            with self.assertRaisesRegex(ValueError, 'changed after binding'):
                train.selected_package(out)

    def test_main_rejects_replaced_consumed_downstream_gates(self):
        for stage, consumed in (('visual', 'cpu_validate'), ('runtime', 'cpu_validate'), ('runtime', 'visual')):
            for mutation in ('receipt', 'marker'):
                with self.subTest(stage=stage, consumed=consumed, mutation=mutation), tempfile.TemporaryDirectory() as directory:
                    out = Path(directory)
                    selected = completed_training(out)
                    receipt = {'passed': True, 'audit_sha256': train.sha256(out / 'audit.json'),
                               'selected_sha256': selected['package_sha256']}
                    completed_stage(out, 'cpu_validate', receipt)
                    if stage == 'runtime':
                        completed_stage(out, 'visual', receipt)
                    def run(path, recipe, device):
                        if mutation == 'receipt':
                            replace_completed_receipt(path, consumed, modified=True)
                        else:
                            with (path / (consumed + '_complete.json')).open('a') as stream:
                                stream.write(' ')
                        train.selected_gate(path, consumed, selected)
                        train.write_json(path / (stage + '.json'), receipt)
                    with patch.dict(train.STAGES, {stage: run}), patch.object(train, 'checked_audit'), \
                            patch.object(sys, 'argv', ['train_set', stage, '--out', str(out), '--device', 'cpu']):
                        with self.assertRaisesRegex(ValueError, 'changed after admission'):
                            train.main()
                    self.assertFalse((out / (stage + '_complete.json')).exists())
                    self.assertTrue((out / (stage + '_failure.json')).is_file())

    def test_main_rejects_rebound_training_chain_before_cpu_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            selected = completed_training(out)
            def run(path, recipe, device):
                replace_completed_receipt(path, 'condition', modified=True)
                replace_completed_receipt(path, 'overfit', condition_sha256=train.sha256(path / 'condition.json'))
                train.training_prerequisites(path)
                train.write_json(path / 'cpu_validate.json', {'passed': True, 'selected_sha256': selected['package_sha256']})
            with patch.dict(train.STAGES, {'cpu_validate': run}), patch.object(train, 'checked_audit'), \
                    patch.object(sys, 'argv', ['train_set', 'cpu_validate', '--out', str(out), '--device', 'cpu']):
                with self.assertRaisesRegex(ValueError, 'changed after binding'):
                    train.main()
            self.assertFalse((out / 'cpu_validate_complete.json').exists())
            self.assertTrue((out / 'cpu_validate_failure.json').is_file())

    def test_main_blocks_successor_until_condition_pool_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            completed_stage(out, 'audit', {'passed': True})
            receipt_written, release_pool = threading.Event(), threading.Event()
            state = {'open': False}
            @contextmanager
            def owner():
                state['open'] = True
                try:
                    yield
                finally:
                    self.assertTrue(release_pool.wait(timeout=5))
                    state['open'] = False
            def condition(path, recipe, device):
                with owner():
                    train.write_json(path / 'condition.json', {'passed': True, 'audit_sha256': train.sha256(path / 'audit.json')})
                    receipt_written.set()
            with patch.dict(train.STAGES, {'condition': condition}), patch.object(train, 'checked_audit'), \
                    patch.object(sys, 'argv', ['train_set', 'condition', '--out', str(out), '--device', 'cpu']), \
                    ThreadPoolExecutor(max_workers=1) as workers:
                running = workers.submit(train.main)
                try:
                    self.assertTrue(receipt_written.wait(timeout=5))
                    self.assertTrue(state['open'])
                    with self.assertRaises(FileExistsError):
                        train.begin_stage(out, 'overfit')
                    self.assertFalse((out / 'overfit_attempt.json').exists())
                finally:
                    release_pool.set()
                running.result(timeout=5)
            self.assertFalse(state['open'])
            train.require_complete(out, 'condition')
            train.begin_stage(out, 'overfit')

    def test_main_finalizes_only_after_stage_returns_and_final_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            state = {'closed': False, 'checked': False}
            def stage(path, recipe, device):
                train.write_json(path / 'audit.json', {'passed': True})
                self.assertFalse((path / 'audit_complete.json').exists())
                state['closed'] = True
            def checked(path, recipe, require_completion):
                self.assertTrue(state['closed'])
                self.assertFalse((path / 'audit_complete.json').exists())
                state['checked'] = True
            with patch.dict(train.STAGES, {'audit': stage}, clear=True), patch.object(train, 'checked_audit', side_effect=checked), \
                    patch.object(sys, 'argv', ['train_set', 'audit', '--out', str(out), '--device', 'cpu']):
                train.main()
            self.assertTrue(state['checked'])
            train.require_complete(out, 'audit')

    def test_cpu_validation_uses_every_anchor_as_singleton(self):
        data = Dataset(index=[(0, i) for i in range(4)], episodes=[Episode()])
        head = core.ConeSetHead()
        seen = []
        def fresh(model, dataset, ids, device):
            seen.append(list(ids))
            inputs = head_inputs(len(ids))
            return train.Batch(*inputs, head(*inputs).detach())
        with patch.object(train, 'batch_input', side_effect=fresh):
            report = train.validate(None, head, data, 'cpu', batch_size=1)
        self.assertEqual(seen, [[0], [1], [2], [3]])
        self.assertEqual(report['groups']['all']['count'], 4)
        self.assertTrue(report['passed'])

    def test_model_and_recipe_identity_checks_do_not_mutate_legacy_module(self):
        import joint_data
        before = joint_data.HERE
        recipe = train.read_json(train.RECIPE)
        with patch.object(train, 'JointData', return_value=object()) as owner:
            train.data_owner(recipe, 'cpu')
        self.assertEqual(joint_data.HERE, before)
        self.assertEqual(owner.call_args.args[0], train.read_json(train.JOINT / 'recipe.json'))


if __name__ == '__main__':
    unittest.main()
