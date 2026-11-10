import ast
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import encode_portable as portable


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    episode = root / 'episode'
    evidence = episode / 'evidence'
    evidence.mkdir(parents=True)
    checkpoint = root / 'package.pt'
    checkpoint.write_bytes(b'synthetic package identity')
    source_names = set(portable.SOURCE_NAMES.values()) | {portable.TRIAL + name for name in (
        'cone_core.py', 'train_set.py', 'recipe.json', 'protocol.md', 'evaluate.py',
        'verify_command_episode.py', 'process.py', 'export_audited_command.py', 'encode_portable.py',
        'runtime_preflight.py', 'pipeline_notes.md', 'test_core.py', 'test_training.py', 'test_pipeline.py')} | {
        '.work/lab/joint_command_trial/joint_core.py', '.work/lab/joint_command_trial/joint_data.py',
                             '.work/lab/joint_command_trial/recipe.json', '.work/lab/joint_command_trial/protocol.md',
                             '.work/lab/next_strategy/cone_set_synthesis_v1.md',
                             '.work/lab/next_strategy/cone_set_interface_v1.md',
                             '.work/lab/next_strategy/cone_set_actuator_design_v1.md',
                             '.work/lab/direct_command_trial/actuator_trial.py', 'scripts/audit_isaac_episode.py',
        'src/fsop/eval/closed_loop.py', 'isaac_eval/evidence.py', 'isaac_lab_fsop/cases.py'}
    sources = {name: hashlib.sha256(name.encode()).hexdigest() for name in source_names}
    with ZipFile(episode / 'source.zip', 'w') as archive:
        for name in sources:
            archive.writestr(name, name.encode())
    for name in portable.ARTIFACTS:
        if name not in ('source.zip', 'summary.json', 'export.json'):
            (episode / name).write_text(name)
    write(episode / 'export.json', {'dependencies': [], 'sha256': {}})
    for name in ('debug', 'research'):
        (episode / f'{name}-raw.mp4').write_bytes(name.encode())
    checkpoint_hash = sha(checkpoint)
    metadata = {'schema_version': 6, 'package_kind': portable.KIND, 'controller_name': portable.KIND,
                'policy_device': 'cpu', 'torch_num_threads': 4, 'case': {'id': 'synthetic'},
                'command_contract': {'synthetic': True}, 'checkpoint_sha256': checkpoint_hash,
                'checkpoint_size_bytes': checkpoint.stat().st_size, 'state_sha256': 'base',
                'head_state_sha256': 'head', 'source_sha256': sources, 'graph_sha256': {'producer-only': 'graph'}}
    for key, name in {'source_archive': 'source.zip', 'realization': 'realization.json', 'scene': 'scene.usda',
                      'appearance': 'appearance.json', 'export': 'export.json'}.items():
        metadata[key + '_sha256'] = sha(episode / name)
    summary = {'policy_device': 'cpu', 'torch_num_threads': 4, 'package_kind': portable.KIND,
               'checkpoint_sha256': checkpoint_hash, 'checkpoint_sha256_after': checkpoint_hash,
               'frozen_unchanged': True, 'state_sha256_after': 'base', 'head_state_sha256_after': 'head'}
    write(episode / 'summary.json', summary)
    write(evidence / 'metadata.json', metadata)
    (evidence / 'frame_000000.npz').write_bytes(b'frame')
    entry = {'frame': 0, 'file': 'frame_000000.npz', 'metadata_sha256': sha(evidence / 'metadata.json'),
             'sha256': sha(evidence / 'frame_000000.npz')}
    (evidence / 'manifest.jsonl').write_text(json.dumps(entry) + '\n')
    completion = {'version': 1, 'frame_count': 1, 'summary_path': '../summary.json',
                  'summary_sha256': sha(episode / 'summary.json'), 'metadata_sha256': sha(evidence / 'metadata.json'),
                  'manifest_sha256': sha(evidence / 'manifest.jsonl')}
    write(evidence / 'completion.json', completion)
    names = ['evidence/metadata.json', 'evidence/manifest.jsonl', 'evidence/completion.json',
             'evidence/frame_000000.npz', *portable.ARTIFACTS]
    inventory = {'/producer/episode/' + name: sha(episode / name) for name in names}
    replay = {'passed': True, 'frozen_unchanged': True, 'device': 'cpu', 'torch_num_threads': 4,
              'schema_version': 6, 'package_kind': portable.KIND, 'frames': 1, 'expected_frames': 1,
              'temporal_phases': [0, 1, 2, 3], 'rtol': 1e-5, 'atol': 1e-5,
              'case': metadata['case'], 'command_contract': metadata['command_contract'],
              'checkpoint_sha256': checkpoint_hash, 'base_state_sha256': 'base', 'head_state_sha256': 'head',
              'source_sha256': sources, 'graph_sha256': metadata['graph_sha256'],
              'evidence_and_artifacts_sha256': inventory}
    replay_path = root / 'replay.json'
    write(replay_path, replay)
    inputs = {'completion_sha256': sha(evidence / 'completion.json'), 'frames_sha256': sha(episode / 'frames.json'),
              'telemetry_sha256': sha(episode / 'telemetry.json'), 'replay_receipt_sha256': sha(replay_path),
              **{key: sources[name] for key, name in portable.SOURCE_NAMES.items()}}
    audit = {'passed': True, 'completion_sha256': inputs['completion_sha256'], 'frames_sha256': inputs['frames_sha256'],
             'telemetry_sha256': inputs['telemetry_sha256'], 'frame_count': 1,
             'replay_receipt_sha256': inputs['replay_receipt_sha256'], 'auditor_sha256': inputs['process_code_sha256'],
             'command_contract': metadata['command_contract']}
    write(episode / 'lab_artifact_audit.json', audit)
    inputs['artifact_audit_sha256'] = sha(episode / 'lab_artifact_audit.json')
    receipt = {'version': 2, 'schema_version': 6, 'package_kind': portable.KIND, 'numpy_version': '1.26.4',
               'inputs': inputs, 'inference_frames': 1, 'terminal_frames': 40,
               'command_contract': metadata['command_contract'], 'raw_videos': {
                   name: {'path': f'{name}-raw.mp4', 'sha256': sha(episode / f'{name}-raw.mp4')}
                   for name in ('debug', 'research')}}
    write(episode / 'producer_render_receipt.json', receipt)
    return episode, replay_path, checkpoint


class PortableBoundaryTests(unittest.TestCase):
    def test_replay_source_inventory_ignores_foreign_evaluator_module(self):
        import verify_command_episode as verifier
        self.assertEqual(Path(verifier.__file__).resolve(),
                         Path(__file__).with_name('verify_command_episode.py').resolve())
        foreign = SimpleNamespace(source_inventory=lambda: {'foreign.py': 'wrong'})
        with patch.dict(sys.modules, {'evaluate': foreign}):
            inventory = verifier.required_source_inventory()
        self.assertIn('.work/lab/cone_set_actuator_trial/cone_core.py', inventory)
        self.assertNotIn('foreign.py', inventory)

    def test_transport_detects_tamper_and_wrong_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode, replay, checkpoint = fixture(Path(tmp))
            result = portable.snapshot(episode, replay, checkpoint)
            self.assertEqual(result['producer_prefix'], '/producer/episode')
            self.assertEqual(result['metadata']['graph_sha256'], {'producer-only': 'graph'})
            (episode / 'debug-raw.mp4').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'Raw video differs'):
                portable.snapshot(episode, replay, checkpoint)
            (episode / 'debug-raw.mp4').write_bytes(b'debug')
            data = json.loads(replay.read_text())
            data['device'] = 'cuda'
            write(replay, data)
            with self.assertRaisesRegex(ValueError, 'CPU replay'):
                portable.snapshot(episode, replay, checkpoint)

    def test_schema_and_old_kind_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode, replay, checkpoint = fixture(Path(tmp))
            for field, invalid in (('schema_version', 5), ('package_kind', 'direct_native_actuator_joint_suffix_v1')):
                data = json.loads(replay.read_text())
                data[field] = invalid
                write(replay, data)
                with self.assertRaisesRegex(ValueError, 'CPU replay'):
                    portable.snapshot(episode, replay, checkpoint)
                data[field] = 6 if field == 'schema_version' else portable.KIND
                write(replay, data)

    def test_cone_input_packing_identity(self):
        import numpy as np
        from cone_core import pack_cones
        xy = np.array([[3., 4.], [-9., 12.]], dtype=np.float64)
        classes = np.array([0, 3], dtype=np.int64)
        scores = np.array([0.75, 0.5], dtype=np.float32)
        tokens, mask = pack_cones(xy, classes, scores)
        expected = np.zeros((200, 8), dtype=np.float32)
        expected[0] = [3 / 45, 4 / 45, 5 / 45, 1, 0, 0, 0, .75]
        expected[1] = [-9 / 45, 12 / 45, 15 / 45, 0, 0, 0, 1, .5]
        np.testing.assert_array_equal(tokens, expected)
        np.testing.assert_array_equal(mask, np.arange(200) < 2)
        with self.assertRaises(ValueError):
            pack_cones(np.zeros((201, 2)), np.zeros(201, dtype=int), np.zeros(201))

    def test_saved_head_inputs_replay_after_json_roundtrip(self):
        import numpy as np
        import torch
        from cone_core import ConeSetHead, pack_cones
        torch.manual_seed(7)
        head = ConeSetHead().eval()
        tokens, mask = pack_cones(np.array([[3., 4.]]), np.array([1]), np.array([0.75]))
        desire = np.zeros((25, 8), dtype=np.float32)
        desire[-1, 1] = 1
        motion = np.array([4.25, -0.125, 0.01], dtype=np.float32)
        def forward(values):
            return head(torch.as_tensor(values[0]).reshape(1, 200, 8),
                        torch.as_tensor(values[1]).reshape(1, 200),
                        torch.as_tensor(values[2]).reshape(1, 25, 8),
                        torch.as_tensor(values[3]).reshape(1, 3)).detach().numpy()
        original = [tokens, mask, desire, motion]
        saved = json.loads(json.dumps([value.tolist() for value in original]))
        restored = [np.asarray(value, dtype=dtype) for value, dtype in zip(
            saved, (np.float32, bool, np.float32, np.float32), strict=True)]
        np.testing.assert_array_equal(forward(original), forward(restored))

    def test_physical_check_names_preserved(self):
        root = Path(__file__).resolve().parent.parent
        def names(path):
            tree = ast.parse(path.read_text())
            return [ast.dump(node.args[0], include_attributes=False) for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'compare' and node.args]
        original = names(root / 'joint_command_trial' / 'process.py')
        current = names(root / 'cone_set_actuator_trial' / 'process.py')
        self.assertGreaterEqual(len(original), 70)
        self.assertEqual(original, current)

    def test_path_escape_and_duplicate_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                portable.inside(root, '../escape')
            with self.assertRaises(ValueError):
                portable.map_inventory({'/p/evidence/metadata.json': 'x', '/other/file': 'x'}, root)
            path = root / 'sources.zip'
            with ZipFile(path, 'w') as archive:
                archive.writestr('source.py', 'one')
                archive.writestr('source.py', 'two')
            with ZipFile(path) as archive, self.assertRaisesRegex(ValueError, 'inventory'):
                portable.verify_source_archive(archive, {'source.py': 'irrelevant'})

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg unavailable')
    def test_video_encoding_and_full_decode(self):
        try:
            importlib.import_module('cv2')
        except ImportError:
            self.skipTest('OpenCV unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            episode, replay, checkpoint = fixture(Path(tmp))
            receipt_path = episode / 'producer_render_receipt.json'
            receipt = json.loads(receipt_path.read_text())
            for name in ('debug', 'research'):
                raw = episode / f'{name}-raw.mp4'
                raw.unlink()
                subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
                                '-f', 'lavfi', '-i', 'color=c=black:s=1280x768:r=20',
                                '-frames:v', '41', '-c:v', 'mpeg4', '-threads', '1', str(raw)], check=True)
                receipt['raw_videos'][name]['sha256'] = sha(raw)
            write(receipt_path, receipt)
            portable.encode(episode, replay, checkpoint)
            result = json.loads((episode / 'lab_video_audit.json').read_text())
            for name in ('debug', 'research'):
                self.assertEqual(result['videos'][name]['decoded_frames'], 41)
                self.assertEqual(result['raw_videos'][name]['decoded_frames'], 41)
                self.assertEqual(result['videos'][name]['fps'], 20)
                self.assertTrue((episode / f'{name}.mp4').is_file())
            with self.assertRaises(FileExistsError):
                portable.encode(episode, replay, checkpoint)

    def test_recorded_controller_is_visible_in_both_views_and_not_terminal(self):
        from types import SimpleNamespace
        import export_audited_command as exporter

        mask = [False] * 200
        mask[1] = mask[150] = True
        controller = {'mask': mask, 'raw': [1.25, 2.5], 'bounded': [0.5, 1.5],
                      'applied': [0.5, -3.5], 'mission_override': True}
        prediction = {'controller': controller}
        written = {'debug': [], 'research': []}

        class Writer:
            def __init__(self, name):
                self.name = name
            def write(self, frame):
                written[self.name].append(list(frame))
            def release(self):
                pass

        cv2 = SimpleNamespace(FONT_HERSHEY_SIMPLEX=0, LINE_AA=0,
                              rectangle=lambda *args: None,
                              putText=lambda frame, line, *args: frame.append(line))
        renderer = SimpleNamespace(_writer=lambda path, fps: Writer(path),
                                   compose_frame=lambda *args: [],
                                   research_frame=lambda record, entry, controller: [])
        originals = (renderer._writer, renderer.compose_frame, renderer.research_frame)
        def render(evidence, debug, research):
            main, extra = renderer._writer(debug, 20), renderer._writer(research, 20)
            main.write(renderer.compose_frame(None, None, None, None, None, None, prediction))
            extra.write(renderer.research_frame({'prediction': prediction}, {}, 'mode'))
            for _ in range(40):
                main.write(['TERMINAL STATE - no current model prediction'])
                extra.write(['TERMINAL STATE - no current model prediction'])
            main.release()
            extra.release()
            return {'inference_frames': 1, 'terminal_frames': 40}
        renderer.render = render
        self.assertEqual(exporter.render_annotated(renderer, cv2, 'evidence', 'debug', 'research'),
                         {'inference_frames': 1, 'terminal_frames': 40})
        self.assertEqual((renderer._writer, renderer.compose_frame, renderer.research_frame), originals)
        for frames in written.values():
            self.assertEqual(len(frames), 41)
            self.assertIn('valid cones: 2  steer rad / accel m/s^2  raw (1.25, 2.5)  '
                          'bounded (0.5, 1.5)  applied (0.5, -3.5)  AS override: True', frames[0])
            for frame in frames[1:]:
                self.assertTrue(frame[0].startswith('TERMINAL STATE'))
                self.assertFalse(any('valid cones:' in line for line in frame))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath('lab_video_audit.json').write_text('keep')
            with self.assertRaises(FileExistsError):
                portable.encode(root, root, root)
            self.assertEqual(root.joinpath('lab_video_audit.json').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
