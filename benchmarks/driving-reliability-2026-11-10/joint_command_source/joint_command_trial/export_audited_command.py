#!/usr/bin/env python3
"""Render with the producer audit environment and encode with the export environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def inputs(episode, replay_receipt):
    replay = read(replay_receipt)
    if replay.get('passed') is not True:
        raise ValueError('Replay receipt did not pass')
    for name, digest in {**replay['source_sha256'], **replay['graph_sha256'], **replay['evidence_and_artifacts_sha256']}.items():
        if sha(Path(name)) != digest:
            raise ValueError(f'Replay-bound input changed: {name}')
    for name in ('.work/lab/joint_command_trial/process.py',
                 '.work/lab/joint_command_trial/export_audited_command.py'):
        if replay['source_sha256'].get(name) != sha(Path(name)):
            raise ValueError(f'Render source differs from producer bundle: {name}')
    return {
        'completion_sha256': sha(episode/'evidence/completion.json'),
        'artifact_audit_sha256': sha(episode/'lab_artifact_audit.json'),
        'frames_sha256': sha(episode/'frames.json'),
        'telemetry_sha256': sha(episode/'telemetry.json'),
        'render_code_sha256': sha(Path('isaac_eval/render_debug.py')),
        'process_code_sha256': sha(Path('.work/lab/joint_command_trial/process.py')),
        'replay_receipt_sha256': sha(replay_receipt),
        'wrapper_code_sha256': sha(Path(__file__)),
    }


def require_absent(paths):
    existing = [str(path) for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError(f'Refusing to overwrite existing output: {existing}')


def render_phase(episode, replay_receipt):
    import cv2
    import numpy as np
    import isaac_eval.render_debug as render_debug
    from process import audit_episode

    if not np.__version__.startswith('1.26.'):
        raise ValueError(f'Expected producer NumPy 1.26, got {np.__version__}')
    receipt_path = episode/'producer_render_receipt.json'
    raw = {name: episode/f'{name}-raw.mp4' for name in ('debug', 'research')}
    require_absent([receipt_path, episode/'lab_video_audit.json', *raw.values(), *(episode/f'{name}.mp4' for name in raw)])
    audit_path = episode/'lab_artifact_audit.json'
    require_absent([audit_path])
    audit = audit_episode(episode, replay_receipt)
    if not audit.get('passed'):
        raise ValueError('Producer artifact audit failed; see lab_artifact_audit.json')
    before = inputs(episode, replay_receipt)
    completion = read(episode/'evidence/completion.json')
    if (audit.get('completion_sha256') != before['completion_sha256']
            or audit.get('frames_sha256') != before['frames_sha256']
            or audit.get('telemetry_sha256') != before['telemetry_sha256']
            or audit.get('frame_count') != completion['frame_count']
            or audit.get('replay_receipt_sha256') != before['replay_receipt_sha256']
            or audit.get('auditor_sha256') != before['process_code_sha256']):
        raise ValueError('Passed artifact audit does not bind episode inputs')
    cv2.setNumThreads(1)
    original_writer = render_debug._writer
    class AnnotatedWriter:
        def __init__(self, writer):
            self.writer = writer
        def write(self, frame):
            cv2.rectangle(frame, (4, 4), (930, 29), (0, 0, 0), -1)
            cv2.putText(frame, 'Plan: diagnostic after policy training | Action: learned joint command',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            return self.writer.write(frame)
        def release(self):
            return self.writer.release()
    def annotated_writer(*args, **kwargs):
        return AnnotatedWriter(original_writer(*args, **kwargs))
    render_debug._writer = annotated_writer
    try:
        result = render_debug.render(episode/'evidence', raw['debug'], raw['research'])
    finally:
        render_debug._writer = original_writer
    if result != {'inference_frames': completion['frame_count'], 'terminal_frames': 40}:
        raise ValueError(f'Render frame count mismatch: {result}')
    if inputs(episode, replay_receipt) != before:
        raise ValueError('Bound inputs changed during rendering')
    receipt = {
        'version': 2, 'package_kind': 'direct_native_actuator_joint_suffix_v1', 'schema_version': 5, 'numpy_version': np.__version__, 'inputs': before,
        'inference_frames': result['inference_frames'], 'terminal_frames': result['terminal_frames'],
        'command_contract': audit['command_contract'],
        'raw_videos': {name: {'path': path.name, 'sha256': sha(path)} for name, path in raw.items()},
    }
    with receipt_path.open('x') as out:
        json.dump(receipt, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({'phase': 'render', 'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--replay-receipt', type=Path, required=True)
    args = parser.parse_args()
    render_phase(args.episode, args.replay_receipt)


if __name__ == '__main__':
    main()
