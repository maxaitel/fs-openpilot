#!/usr/bin/env python3
"""Render with the producer audit environment and encode with the export environment."""
import argparse
import hashlib
import json
import subprocess
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
    for name in ('.work/lab/direct_command_trial/process_v2.py',
                 '.work/lab/direct_command_trial/export_audited_command_v2.py'):
        if name in replay['source_sha256'] and sha(Path(name)) != replay['source_sha256'][name]:
            raise ValueError(f'Derived source differs from producer bundle: {name}')
    return {
        'completion_sha256': sha(episode/'evidence/completion.json'),
        'artifact_audit_sha256': sha(episode/'lab_artifact_audit.json'),
        'frames_sha256': sha(episode/'frames.json'),
        'telemetry_sha256': sha(episode/'telemetry.json'),
        'render_code_sha256': sha(Path('isaac_eval/render_debug.py')),
        'process_code_sha256': sha(Path('.work/lab/direct_command_trial/process_v2.py')),
        'producer_process_code_sha256': sha(Path('.work/lab/direct_command_trial/process.py')),
        'replay_receipt_sha256': sha(replay_receipt),
        'wrapper_code_sha256': sha(Path(__file__)),
    }


def require_absent(paths):
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite existing output: {existing}')


def render_phase(episode, replay_receipt):
    import cv2
    import numpy as np
    import isaac_eval.render_debug as render_debug
    from process_v2 import audit_episode

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
            cv2.putText(frame, 'Plan: frozen model diagnostic | Action: learned direct command',
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
        'version': 1, 'numpy_version': np.__version__, 'inputs': before,
        'inference_frames': result['inference_frames'], 'terminal_frames': result['terminal_frames'],
        'command_contract': audit['command_contract'],
        'raw_videos': {name: {'path': path.name, 'sha256': sha(path)} for name, path in raw.items()},
    }
    with receipt_path.open('x') as out:
        json.dump(receipt, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({'phase': 'render', 'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path)}))


def encode_phase(episode, replay_receipt):
    import cv2

    receipt_path = episode/'producer_render_receipt.json'
    receipt_sha256 = sha(receipt_path)
    receipt = read(receipt_path)
    raw = {name: episode/f'{name}-raw.mp4' for name in ('debug', 'research')}
    finals = {name: episode/f'{name}.mp4' for name in raw}
    temp = {name: episode/f'{name}-h264.tmp.mp4' for name in raw}
    require_absent([episode/'lab_video_audit.json', *finals.values(), *temp.values()])
    if receipt.get('version') != 1 or not receipt.get('numpy_version', '').startswith('1.26.'):
        raise ValueError('Missing producer-version render receipt')
    before = inputs(episode, replay_receipt)
    if receipt['inputs'] != before:
        raise ValueError('Producer render inputs changed')
    audit = read(episode/'lab_artifact_audit.json')
    completion = read(episode/'evidence/completion.json')
    if (audit.get('passed') is not True or audit.get('completion_sha256') != before['completion_sha256']
            or audit.get('frames_sha256') != before['frames_sha256']
            or audit.get('telemetry_sha256') != before['telemetry_sha256']
            or audit.get('frame_count') != completion['frame_count']
            or audit.get('replay_receipt_sha256') != before['replay_receipt_sha256']
            or audit.get('auditor_sha256') != before['process_code_sha256']
            or receipt['inference_frames'] != completion['frame_count']
            or receipt['terminal_frames'] != 40
            or receipt['command_contract'] != audit['command_contract']):
        raise ValueError('Producer audit or render count does not bind this episode')
    for name, path in raw.items():
        if receipt['raw_videos'][name] != {'path': path.name, 'sha256': sha(path)}:
            raise ValueError(f'Raw {name} video differs from producer receipt')
    cv2.setNumThreads(1)
    videos = {}
    for name, path in raw.items():
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(path),
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                        '-threads', '1', '-movflags', '+faststart', str(temp[name])], check=True)
        capture = cv2.VideoCapture(str(temp[name]))
        fps, decoded = capture.get(cv2.CAP_PROP_FPS), 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape != (768, 1280, 3):
                    raise ValueError(f'Unexpected video dimensions: {temp[name]}')
                decoded += 1
        finally:
            capture.release()
        if fps != 20 or decoded != completion['frame_count'] + 40:
            raise ValueError(f'Video decode or cadence mismatch: {temp[name]}')
        videos[name] = {'sha256': sha(temp[name]), 'fps': fps, 'decoded_frames': decoded}
    if inputs(episode, replay_receipt) != before or sha(receipt_path) != receipt_sha256:
        raise ValueError('Bound inputs changed during encoding')
    for name in raw:
        if sha(raw[name]) != receipt['raw_videos'][name]['sha256']:
            raise ValueError(f'Raw {name} video changed during encoding')
    for name in finals:
        temp[name].replace(finals[name])
    video_audit = {
        'inference_frames': completion['frame_count'], 'terminal_frames': 40,
        'completion_sha256': before['completion_sha256'], 'artifact_audit_sha256': before['artifact_audit_sha256'],
        'command_contract': audit['command_contract'], 'videos': videos,
        'producer_render_receipt_sha256': sha(receipt_path), 'producer_numpy_version': receipt['numpy_version'],
        'render_code_sha256': before['render_code_sha256'], 'process_code_sha256': before['process_code_sha256'],
        'producer_process_code_sha256': before['producer_process_code_sha256'],
        'wrapper_code_sha256': before['wrapper_code_sha256'],
        'replay_receipt_sha256': before['replay_receipt_sha256'],
    }
    with (episode/'lab_video_audit.json').open('x') as out:
        json.dump(video_audit, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({'phase': 'encode', 'video_audit_sha256': sha(episode/'lab_video_audit.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('render', 'encode'))
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--replay-receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'render':
        render_phase(args.episode, args.replay_receipt)
    else:
        encode_phase(args.episode, args.replay_receipt)


if __name__ == '__main__':
    main()
