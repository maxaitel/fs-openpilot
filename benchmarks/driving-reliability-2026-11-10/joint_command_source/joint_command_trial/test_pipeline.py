"""CPU-only tests of portable evidence transport boundaries using synthetic receipts."""
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
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
        'joint_core.py', 'joint_data.py', 'train_joint.py', 'recipe.json', 'protocol.md', 'evaluate.py',
        'verify_command_episode.py', 'process.py', 'export_audited_command.py', 'encode_portable.py',
        'pipeline_notes.md', 'test_joint.py', 'test_pipeline.py')} | {
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
    metadata = {'schema_version': 5, 'package_kind': portable.KIND, 'controller_name': portable.KIND,
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
              'schema_version': 5, 'package_kind': portable.KIND, 'frames': 1, 'expected_frames': 1,
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
    receipt = {'version': 2, 'schema_version': 5, 'package_kind': portable.KIND, 'numpy_version': '1.26.4',
               'inputs': inputs, 'inference_frames': 1, 'terminal_frames': 40,
               'command_contract': metadata['command_contract'], 'raw_videos': {
                   name: {'path': f'{name}-raw.mp4', 'sha256': sha(episode / f'{name}-raw.mp4')}
                   for name in ('debug', 'research')}}
    write(episode / 'producer_render_receipt.json', receipt)
    return episode, replay_path, checkpoint


class PortableBoundaryTests(unittest.TestCase):
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
            import cv2  # noqa: F401
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

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath('lab_video_audit.json').write_text('keep')
            with self.assertRaises(FileExistsError):
                portable.encode(root, root, root)
            self.assertEqual(root.joinpath('lab_video_audit.json').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
