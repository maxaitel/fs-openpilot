#!/usr/bin/env python3
"""Record the original image detector on fixed native TRAIN recordings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import numpy as np

MANIFEST_SHA = 'db7150b60fdf50e861c6ca003ae844b90ff398fd7b244c9201b7de9d3af5b309'
CHECKPOINT_SHA = '48db741c8a686b2e635ae7a81b4e4c08d79fbfbd9f4dc4c163260da6f93f0cfb'
RECOVERY_META_SHA = {
    'skidpad': 'bb4c7b27f21835d81ea37a5fe65ff112c12fe3fb16872c81b4df68b61d52303d',
    'trackdrive': 'eb74ed5dd8e574af86c54764ca07ea0080e5ba15fc4325f8c3f39bf5704e7f91',
}
RECOVERY_VERIFIER_SHA = '5babb4a61c819f829b030f8c8a5321e6b672be1bb422d9a99d5f9ae9c3e3dc73'
IDS = [f'train-{d}-82000-matched' for d in ('skidpad', 'trackdrive')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(json.dumps(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def frame_pair(frames, k):
    if not 0 <= k < len(frames):
        raise IndexError(k)
    return np.concatenate((frames[max(0, k - 1)], frames[k]), axis=0)


def frame_context(ticks, states, desires, k):
    tick = int(ticks[k])
    return {'frame': k, 'tick': tick, 'time_s': float(states[tick, 8]),
            'vx_vy_yaw_rate': [float(x) for x in states[tick, 3:6]],
            'desire_history': np.asarray(desires[k], dtype=np.float32).tolist()}


def self_test():
    frames = np.arange(3, dtype=np.uint8).reshape(3, 1, 1, 1)
    ticks = np.array([0, 4, 8]); states = np.zeros((9, 9)); states[:, 8] = np.arange(9) / 20
    states[:, 3:6] = np.arange(9)[:, None] + [0.1, 0.2, 0.3]
    desires = np.arange(3 * 25 * 8, dtype=np.float32).reshape(3, 25, 8)
    assert frame_pair(frames, 0).flatten().tolist() == [0, 0]
    assert frame_pair(frames, 2).flatten().tolist() == [1, 2]
    context = frame_context(ticks, states, desires, 2)
    assert context['tick'] == 8 and context['time_s'] == 0.4
    assert np.allclose(context['vx_vy_yaw_rate'], [8.1, 8.2, 8.3])
    assert context['desire_history'] == desires[2].tolist()
    print('frame pair and terminal tick alignment passed')


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def paths_from_manifest(manifest):
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError('manifest SHA-256 differs from the frozen TRAIN inventory')
    doc = json.loads(manifest.read_text())
    if doc.get('schema_version') != 1 or doc.get('complete') is not True or doc.get('expected_episodes') != len(doc['episodes']):
        raise ValueError('invalid manifest')
    result = []
    root = manifest.parent.resolve()
    for ident in IDS:
        rows = [row for row in doc['episodes'] if row.get('path') == ident]
        if len(rows) != 1:
            raise ValueError(f'manifest must contain exactly one {ident}')
        row = rows[0]
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or sha(path / 'meta.json') != row['meta_sha256']:
            raise ValueError(f'manifest metadata mismatch for {ident}')
        meta = json.loads((path / 'meta.json').read_text())
        if meta.get('membership_sha256') != doc['membership_sha256']:
            raise ValueError(f'membership mismatch for {ident}')
        result.append((ident, path, None))
    return result


def recovery_paths(root):
    result = []
    for discipline in ('skidpad', 'trackdrive'):
        path = (root / discipline).resolve()
        receipt_path = (root / f'{discipline}-receipt-v2.json').resolve()
        receipt = json.loads(receipt_path.read_text())
        meta_sha = sha(path / 'meta.json')
        meta = json.loads((path / 'meta.json').read_text())
        if (not receipt.get('passed') or receipt.get('reasons') or
                receipt.get('hashes', {}).get('meta_sha256') != meta_sha or
                meta_sha != RECOVERY_META_SHA[discipline] or
                receipt.get('hashes', {}).get('derived_verifier_sha256') != RECOVERY_VERIFIER_SHA or
                receipt.get('arrays_sha256') != meta.get('arrays_sha256')):
            raise ValueError(f'recovery verifier receipt mismatch for {discipline}')
        result.append((f'train-{discipline}-82000-matched-direct-recovery-v1', path, receipt_path))
    return result


def snapshot(paths):
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def head_hash(head):
    result = {}
    for cam in ('road', 'wide'):
        for key in ('heat', 'wh', 'off'):
            value = head[cam][key].detach().cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError(f'nonfinite image head output: {cam}.{key}')
            result[f'{cam}.{key}'] = array_sha(value)
    return result


def run(args, out, receipt):
    manifest, checkpoint, prep = args.manifest.resolve(), args.checkpoint.resolve(), Path('artifacts/models/prep').resolve()
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('checkpoint SHA-256 differs from original pretrained checkpoint')
    for name in ('vision_fp32.onnx', 'policy_fp32.onnx'):
        if not (prep / name).is_file():
            raise ValueError(f'missing prepared graph {prep / name}; refusing fetch')
    episodes = paths_from_manifest(manifest)
    if args.recovery_root:
        episodes.extend(recovery_paths(args.recovery_root.resolve()))
    files = [manifest, checkpoint, Path(__file__).resolve(), Path(__file__).with_name('protocol.md').resolve()]
    files += [p for p in Path('src/fsop').rglob('*.py') if p.is_file()]
    files += [p for p in prep.rglob('*') if p.is_file()]
    for ident, path, recovery_receipt in episodes:
        meta = json.loads((path / 'meta.json').read_text())
        if meta.get('id') != ident or meta.get('split') != 'train' or meta.get('seed') != 82000:
            raise ValueError(f'wrong episode identity: {path}')
        files.append(path / 'meta.json')
        for name in meta['arrays_sha256']:
            if Path(name).name != name or not name.endswith('.npy'):
                raise ValueError(f'invalid array path in metadata: {name}')
            files.append(path / name)
        if recovery_receipt:
            files.append(recovery_receipt)
    before = snapshot(files)
    receipt['input_sha256_before'] = before
    write_json(out / 'receipt.json', receipt)
    import torch
    from fsop.data.native_replay import NativeEpisode
    from fsop.eval.img_to_bev import bev_from_image_head
    from fsop.model.build import load_fs
    import fsop

    if not Path(fsop.__file__).resolve().is_relative_to(Path('src/fsop').resolve()):
        raise ValueError('fsop import does not resolve to the hashed source tree')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    receipt.update(status='running', input_sha256_before=before, episodes=[x[0] for x in episodes],
                   checkpoint_sha256=CHECKPOINT_SHA, manifest_sha256=MANIFEST_SHA,
                   device='cpu', dtype='float32', torch_threads=torch.get_num_threads(),
                   torch_version=torch.__version__, numpy_version=np.__version__, python_version=platform.python_version())
    write_json(out / 'receipt.json', receipt)
    model = load_fs(checkpoint, map_location='cpu').float().eval()
    if model.img is None:
        raise ValueError('checkpoint has no image head')
    if any(p.device.type != 'cpu' or (p.is_floating_point() and p.dtype != torch.float32)
           for p in (*model.parameters(), *model.buffers())):
        raise ValueError('model must be CPU FP32')
    references = []
    counts = {}
    with (out / 'frames.jsonl').open('x') as target, torch.inference_mode():
        for ident, path, _ in episodes:
            ep = NativeEpisode(path)
            summary = ep.meta['summary']
            if not (summary.get('result') == 'finished' and summary.get('pass') is True and
                    summary.get('complete') is True and summary.get('physical_cone_hits') == 0 and summary.get('oc') == 0):
                raise ValueError(f'episode is not a qualified clean continuation: {ident}')
            calibration = np.asarray(ep.meta['calibration']['T_ego_cam'], dtype=np.float64)
            if calibration.shape != (4, 4) or not np.isfinite(calibration).all():
                raise ValueError(f'invalid runtime camera calibration: {ident}')
            references.append({'id': ident, 'episode_path': str(path),
                               'state_path': str(path / 'states.npy'), 'ground_truth_cones_path': str(path / 'cones.npy'),
                               'contacts_path': str(path / 'contacts.npy'), 'metadata_path': str(path / 'meta.json'),
                               'input_arrays': {name: {'path': str(path / f'{name}.npy'),
                                                      'shape': list(getattr(ep, name).shape),
                                                      'dtype': str(getattr(ep, name).dtype)}
                                                for name in ('road', 'wide', 'ticks', 'states', 'desires')},
                               'runtime_T_ego_cam': calibration.tolist()})
            for k in range(len(ep.ticks)):
                pairs = {cam: frame_pair(getattr(ep, cam), k) for cam in ('road', 'wide')}
                def infer():
                    a = torch.from_numpy(pairs['road'][None]).float()
                    b = torch.from_numpy(pairs['wide'][None]).float()
                    v = model.base.encode(a, b)
                    head = model.img(v.s4, v.s8, v.s16)
                    decoded = bev_from_image_head(head, calibration)
                    return head_hash(head), decoded
                raw, (xy, cls, score) = infer()
                if k < 8 or k % 50 == 0 or k == len(ep.ticks) - 1:
                    raw2, decoded2 = infer()
                    if raw != raw2 or any(not np.array_equal(a, b) for a, b in zip((xy, cls, score), decoded2)):
                        raise ValueError(f'nondeterministic output for {ident} frame {k}')
                if not np.isfinite(xy).all() or not np.isfinite(score).all():
                    raise ValueError(f'nonfinite detector output for {ident} frame {k}')
                row = {'episode': ident, **frame_context(ep.ticks, ep.states, ep.desires, k),
                       'pair_sha256': {cam: array_sha(pair) for cam, pair in pairs.items()},
                       'head_sha256': raw, 'cones_xy': xy.tolist(),
                       'cones_cls': cls.tolist(), 'cones_score': score.tolist()}
                target.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
            counts[ident] = len(ep.ticks)
            target.flush()
    write_json(out / 'references.json', references)
    after = snapshot(files)
    receipt['input_sha256_after'] = after
    if before != after:
        raise ValueError('source or input bytes changed during recording')
    receipt.update(status='passed', frame_counts=counts,
                   output_sha256={p.name: sha(p) for p in (out / 'frames.jsonl', out / 'references.json')},
                   deterministic_repeat='frames 0 through 7, every 50th frame, and terminal frame')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/native/lab_v7/manifest.json'))
    parser.add_argument('--checkpoint', type=Path, default=Path('probes/pi_models/latest_pi_train.pt'))
    parser.add_argument('--recovery-root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out is None:
        parser.error('--out is required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'starting'}
    try:
        write_json(out / 'receipt.json', receipt)
        run(args, out, receipt)
    except BaseException as exc:
        if 'input_sha256_before' in receipt and 'input_sha256_after' not in receipt:
            try:
                receipt['input_sha256_after'] = snapshot([Path(p) for p in receipt['input_sha256_before']])
            except Exception as hash_exc:
                receipt['post_hash_error'] = f'{type(hash_exc).__name__}: {hash_exc}'
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        write_json(out / 'receipt.json', receipt)
        raise
    write_json(out / 'receipt.json', receipt)


if __name__ == '__main__':
    main()
