#!/usr/bin/env python3
"""Verify a transported producer render and encode it without rerendering."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
TRIAL = '.work/lab/cone_set_actuator_trial/'
KIND = 'vision_cone_set_actuator_v1'
SOURCE_NAMES = {
    'render_code_sha256': 'isaac_eval/render_debug.py',
    'process_code_sha256': TRIAL + 'process.py',
    'wrapper_code_sha256': TRIAL + 'export_audited_command.py',
}
ARTIFACTS = ('summary.json', 'source.zip', 'case.json', 'track.json', 'realization.json',
             'scene.usda', 'appearance.json', 'export.json', 'frames.json', 'telemetry.json')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def inside(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(part in ('.', '..') for part in p.parts):
        raise ValueError(f'Unsafe relative path: {name}')
    path = root.joinpath(*p.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes episode: {name}')
    return path


def map_inventory(recorded, episode):
    suffix = '/evidence/metadata.json'
    keys = [key for key in recorded if key.endswith(suffix)]
    if len(keys) != 1:
        raise ValueError('Replay must identify one episode metadata path')
    prefix = keys[0][:-len(suffix)]
    if not prefix or '..' in PurePosixPath(prefix).parts:
        raise ValueError('Unsafe producer episode prefix')
    mapped = {}
    for name, digest in recorded.items():
        if not name.startswith(prefix + '/'):
            raise ValueError(f'Replay artifact outside producer episode: {name}')
        relative = name[len(prefix) + 1:]
        path = inside(episode, relative)
        if relative in mapped:
            raise ValueError(f'Duplicate replay artifact: {relative}')
        mapped[relative] = (path, digest)
    return prefix, mapped



def verify_inventory(recorded, expected):
    if set(recorded) != expected:
        raise ValueError(f'Replay artifact inventory differs: missing={sorted(expected-set(recorded))}, extra={sorted(set(recorded)-expected)}')
    for name, (path, digest) in recorded.items():
        if sha(path) != digest:
            raise ValueError(f'Replay artifact hash differs: {name}')


def verify_source_archive(archive, sources):
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != set(sources):
        raise ValueError('Source archive inventory differs')
    for name, digest in sources.items():
        inside(ROOT, name)
        with archive.open(name) as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise ValueError(f'Archived source hash differs: {name}')

def snapshot(episode, replay_path, checkpoint):
    episode = episode.resolve(strict=True)
    replay = read(replay_path)
    metadata = read(episode / 'evidence/metadata.json')
    completion = read(episode / 'evidence/completion.json')
    summary = read(episode / 'summary.json')
    entries = [json.loads(line) for line in (episode / 'evidence/manifest.jsonl').read_text().splitlines()]
    export = read(episode / 'export.json')
    prefix, recorded = map_inventory(replay['evidence_and_artifacts_sha256'], episode)
    if (not entries or completion.get('version') != 1 or completion.get('frame_count') != len(entries)
            or (episode / 'evidence' / completion['summary_path']).resolve() != episode / 'summary.json'
            or completion.get('summary_sha256') != sha(episode / 'summary.json')
            or completion.get('metadata_sha256') != sha(episode / 'evidence/metadata.json')
            or completion.get('manifest_sha256') != sha(episode / 'evidence/manifest.jsonl')):
        raise ValueError('Completion receipt does not bind the complete episode')
    for i, entry in enumerate(entries):
        if (entry['frame'] != i or entry['file'] != f'frame_{i:06d}.npz'
                or entry['metadata_sha256'] != completion['metadata_sha256']
                or sha(inside(episode / 'evidence', entry['file'])) != entry['sha256']):
            raise ValueError(f'Invalid manifest frame {i}')
    if {p.name for p in (episode / 'evidence').glob('frame_*.npz')} != {e['file'] for e in entries}:
        raise ValueError('Evidence frame inventory differs')
    dependencies = export['dependencies']
    if not isinstance(dependencies, list) or len(set(dependencies)) != len(dependencies):
        raise ValueError('Invalid asset dependencies')
    if set(export['sha256']) != set(dependencies):
        raise ValueError('Asset hash inventory differs')
    expected = {'evidence/metadata.json', 'evidence/manifest.jsonl', 'evidence/completion.json',
                *ARTIFACTS, *(f'evidence/{e["file"]}' for e in entries),
                *(f'assets/{name}' for name in dependencies)}
    verify_inventory(recorded, expected)
    for name, digest in export['sha256'].items():
        if sha(inside(episode / 'assets', name)) != digest:
            raise ValueError(f'Asset hash differs: {name}')
    for key, name in {'source_archive': 'source.zip', 'realization': 'realization.json',
                      'scene': 'scene.usda', 'appearance': 'appearance.json', 'export': 'export.json'}.items():
        if sha(episode / name) != metadata[key + '_sha256']:
            raise ValueError(f'Metadata hash differs: {name}')
    if (replay.get('passed') is not True or replay.get('frozen_unchanged') is not True
            or replay.get('device') != 'cpu' or replay.get('torch_num_threads') != 4
            or replay.get('schema_version') != 6 or replay.get('package_kind') != KIND
            or metadata.get('schema_version') != 6 or metadata.get('package_kind') != KIND
            or metadata.get('controller_name') != KIND or metadata.get('policy_device') != 'cpu'
            or metadata.get('torch_num_threads') != 4 or summary.get('policy_device') != 'cpu'
            or summary.get('torch_num_threads') != 4 or summary.get('package_kind') != KIND or replay.get('frames') != len(entries)
            or replay.get('expected_frames') != len(entries) or replay.get('temporal_phases') != [0, 1, 2, 3]
            or replay.get('rtol') != 1e-5 or replay.get('atol') != 1e-5):
        raise ValueError('Complete passed cone-set CPU replay required')
    for replay_key, metadata_key in {'case': 'case', 'command_contract': 'command_contract',
                                     'checkpoint_sha256': 'checkpoint_sha256', 'base_state_sha256': 'state_sha256',
                                     'head_state_sha256': 'head_state_sha256', 'source_sha256': 'source_sha256',
                                     'graph_sha256': 'graph_sha256'}.items():
        if replay.get(replay_key) != metadata[metadata_key]:
            raise ValueError(f'Replay identity mismatch: {replay_key}')
    required_sources = {TRIAL + name for name in (
        'cone_core.py', 'train_set.py', 'recipe.json', 'protocol.md',
        'evaluate.py', 'verify_command_episode.py', 'process.py', 'export_audited_command.py',
        'encode_portable.py', 'runtime_preflight.py', 'pipeline_notes.md', 'test_core.py', 'test_training.py', 'test_pipeline.py')}
    required_sources.update(SOURCE_NAMES.values())
    required_sources.update({'.work/lab/joint_command_trial/joint_core.py', '.work/lab/joint_command_trial/joint_data.py',
                             '.work/lab/joint_command_trial/recipe.json', '.work/lab/joint_command_trial/protocol.md',
                             '.work/lab/next_strategy/cone_set_synthesis_v1.md',
                             '.work/lab/next_strategy/cone_set_interface_v1.md',
                             '.work/lab/next_strategy/cone_set_actuator_design_v1.md',
                             '.work/lab/direct_command_trial/actuator_trial.py',
                             'scripts/audit_isaac_episode.py', 'src/fsop/eval/closed_loop.py',
                             'isaac_eval/evidence.py', 'isaac_lab_fsop/cases.py'})
    if not metadata['graph_sha256'] or not required_sources <= metadata['source_sha256'].keys():
        raise ValueError('Missing required graph or source inventory')
    checkpoint_hash = sha(checkpoint)
    if checkpoint_hash != metadata['checkpoint_sha256'] or checkpoint.stat().st_size != metadata['checkpoint_size_bytes']:
        raise ValueError('Checkpoint package hash or size differs')
    if (summary['checkpoint_sha256'] != checkpoint_hash or summary.get('frozen_unchanged') is not True
            or summary.get('checkpoint_sha256_after') != checkpoint_hash
            or summary.get('state_sha256_after') != metadata['state_sha256']
            or summary.get('head_state_sha256_after') != metadata['head_state_sha256']):
        raise ValueError('Summary does not bind unchanged package and model state')
    with ZipFile(episode / 'source.zip') as archive:
        verify_source_archive(archive, metadata['source_sha256'])
    receipt_path = episode / 'producer_render_receipt.json'
    audit_path = episode / 'lab_artifact_audit.json'
    receipt, audit = read(receipt_path), read(audit_path)
    inputs = receipt['inputs']
    actual = {'completion_sha256': sha(episode / 'evidence/completion.json'),
              'artifact_audit_sha256': sha(audit_path), 'frames_sha256': sha(episode / 'frames.json'),
              'telemetry_sha256': sha(episode / 'telemetry.json'), 'replay_receipt_sha256': sha(replay_path)}
    if (receipt.get('version') != 2 or receipt.get('schema_version') != 6
            or receipt.get('package_kind') != KIND or not receipt.get('numpy_version', '').startswith('1.26.')):
        raise ValueError('Producer NumPy 1.26 render receipt required')
    for key, digest in actual.items():
        if inputs.get(key) != digest:
            raise ValueError(f'Producer render input differs: {key}')
    code_paths = {}
    for key, name in SOURCE_NAMES.items():
        archived = metadata['source_sha256'].get(name)
        if archived is None or inputs.get(key) != archived:
            raise ValueError(f'Producer code differs from archived source: {name}')
        code_paths[key] = 'producer_source_bundle'
    if (audit.get('passed') is not True or audit.get('completion_sha256') != actual['completion_sha256']
            or audit.get('frames_sha256') != actual['frames_sha256'] or audit.get('telemetry_sha256') != actual['telemetry_sha256']
            or audit.get('frame_count') != len(entries) or audit.get('replay_receipt_sha256') != actual['replay_receipt_sha256']
            or audit.get('auditor_sha256') != inputs['process_code_sha256']
            or receipt.get('inference_frames') != len(entries) or receipt.get('terminal_frames') != 40
            or receipt.get('command_contract') != audit.get('command_contract')
            or receipt.get('command_contract') != metadata['command_contract']):
        raise ValueError('Producer artifact audit or render receipt does not bind episode')
    raw = {}
    for name in ('debug', 'research'):
        path = episode / f'{name}-raw.mp4'
        raw[name] = sha(path)
        if receipt['raw_videos'][name] != {'path': path.name, 'sha256': raw[name]}:
            raise ValueError(f'Raw video differs from receipt: {name}')
    return {'replay': replay, 'metadata': metadata, 'completion': completion, 'summary': summary,
            'receipt': receipt, 'audit': audit, 'inputs': inputs, 'producer_prefix': prefix,
            'inventory': {name: digest for name, (_, digest) in recorded.items()}, 'raw': raw,
            'receipt_sha256': sha(receipt_path), 'checkpoint_sha256': checkpoint_hash,
            'code_origins': code_paths, 'encoder_sha256': sha(Path(__file__).resolve())}


def decode(path, count):
    import cv2
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), 0
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if image.shape != (768, 1280, 3):
                raise ValueError(f'Unexpected frame dimensions: {path}')
            frames += 1
    finally:
        cap.release()
    if fps != 20 or frames != count:
        raise ValueError(f'Video cadence or decode count differs: {path}, {fps}, {frames}')
    return {'sha256': sha(path), 'fps': fps, 'decoded_frames': frames}


def encode(episode, replay_path, checkpoint):
    episode, replay_path, checkpoint = episode.resolve(strict=True), replay_path.resolve(strict=True), checkpoint.resolve(strict=True)
    raw = {name: episode / f'{name}-raw.mp4' for name in ('debug', 'research')}
    temp = {name: episode / f'{name}-h264.tmp.mp4' for name in raw}
    final = {name: episode / f'{name}.mp4' for name in raw}
    audit_path = episode / 'lab_video_audit.json'
    for path in [audit_path, *temp.values(), *final.values()]:
        if os.path.lexists(path):
            raise FileExistsError(f'Refusing to overwrite: {path}')
    before = snapshot(episode, replay_path, checkpoint)
    ffmpeg_version = subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    count = before['completion']['frame_count'] + 40
    raw_decodes = {name: decode(path, count) for name, path in raw.items()}
    videos = {}
    for name, path in raw.items():
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(path),
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                        '-threads', '1', '-movflags', '+faststart', str(temp[name])], check=True)
        videos[name] = decode(temp[name], count)
    if snapshot(episode, replay_path, checkpoint) != before:
        raise ValueError('Bound inputs changed during encoding')
    for name in final:
        # Hard links publish without replacing an existing destination.
        os.link(temp[name], final[name])
        temp[name].unlink()
    inputs = before['inputs']
    audit = {'inference_frames': before['completion']['frame_count'], 'terminal_frames': 40,
             'completion_sha256': inputs['completion_sha256'], 'artifact_audit_sha256': inputs['artifact_audit_sha256'],
             'command_contract': before['audit']['command_contract'], 'videos': videos,
             'producer_render_receipt_sha256': before['receipt_sha256'], 'producer_numpy_version': before['receipt']['numpy_version'],
             **{key: inputs[key] for key in SOURCE_NAMES}, 'replay_receipt_sha256': inputs['replay_receipt_sha256'],
             'producer_renderer_audit': {'artifact_audit_sha256': inputs['artifact_audit_sha256'],
                                         'render_receipt_sha256': before['receipt_sha256'], 'code_origins': before['code_origins']},
             'replay': {'receipt_sha256': inputs['replay_receipt_sha256'], 'producer_episode_prefix': before['producer_prefix'],
                        'graph_sha256': before['metadata']['graph_sha256'],
                        'graph_verification': 'producer_replay_bound_not_locally_reexecuted'},
             'source_sha256': before['metadata']['source_sha256'], 'checkpoint_sha256': before['checkpoint_sha256'],
             'portable_encoder_sha256': before['encoder_sha256'], 'ffmpeg_version': ffmpeg_version,
             'raw_videos': raw_decodes, 'scope': 'Transport encoding only; no physics, inference, or renderer rerun; no model qualification.'}
    with audit_path.open('x') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'phase': 'encode', 'video_audit_sha256': sha(audit_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--replay-receipt', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    encode(args.episode, args.replay_receipt, args.checkpoint)


if __name__ == '__main__':
    main()
