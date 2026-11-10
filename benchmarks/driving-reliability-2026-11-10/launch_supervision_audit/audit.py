import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

source_path = Path('src/fsop/data/native_replay.py')
source = source_path.read_text()
tree = ast.parse(source)
horizon = next(ast.literal_eval(x.value) for x in tree.body if isinstance(x, ast.Assign)
               and any(isinstance(y, ast.Name) and y.id == 'HORIZON' for y in x.targets))
episode_class = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'NativeEpisode')
initializer = next(x for x in episode_class.body if isinstance(x, ast.FunctionDef) and x.name == '__init__')
anchor_start = next(i for i, x in enumerate(initializer.body) if isinstance(x, ast.Assign)
                    and any(isinstance(y, ast.Name) and y.id == 'bad' for y in x.targets))
anchor_code = compile(ast.Module(body=initializer.body[anchor_start:], type_ignores=[]), str(source_path), 'exec')
recipe_path = Path('.work/lab/joint_command_trial/recipe.json')
recipe = json.loads(recipe_path.read_text())
audit = {'kind': 'read_only_launch_supervision_v1', 'horizon': horizon,
         'scope': 'No torch, model, simulation, fit, camera pixel reads, or remote writes.',
         'anchor_implementation': 'Executed only the exact NativeEpisode.__init__ AST suffix beginning at bad=.',
         'sources': {}, 'datasets': [], 'episodes': [], 'aggregate': {}}
for path in (source_path, recipe_path, Path('.work/lab/joint_command_trial/joint_data.py')):
    audit['sources'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
inventory = []
for dataset in ('lab_v7', 'lab_v2'):
    root = Path('/work/data/native') / dataset
    manifest_path = root / 'manifest.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest['complete'] is True and manifest['schema_version'] == 1
    assert manifest['expected_episodes'] == len(manifest['episodes']) == 40
    record = {'name': dataset, 'manifest': str(manifest_path),
              'sha256': hashlib.sha256(manifest_raw).hexdigest(),
              'membership_sha256': manifest['membership_sha256'], 'episode_count': len(manifest['episodes'])}
    if dataset == 'lab_v7':
        assert record['sha256'] == recipe['manifest_sha256']
    audit['datasets'].append(record)
    for entry in manifest['episodes']:
        inventory.append((dataset, root / entry['path'], entry, manifest['membership_sha256']))
for discipline, entry in recipe['new_cases'].items():
    inventory.append(('new_recovery', Path(entry['episode']).resolve(), None, None))

for dataset, path, entry, membership in inventory:
    raw = (path / 'meta.json').read_bytes()
    metadata = json.loads(raw)
    meta_sha = hashlib.sha256(raw).hexdigest()
    if entry is not None:
        assert meta_sha == entry['meta_sha256'], str(path)
        assert metadata['membership_sha256'] == membership, str(path)
        for field in ('id', 'split'):
            assert field not in entry or entry[field] == metadata[field]
    arrays = {}
    hashes = {}
    for name in ('ticks', 'states', 'controls', 'teacher', 'desires', 'cones', 'contacts'):
        file = path / (name + '.npy')
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        assert digest == metadata['arrays_sha256'][file.name], str(file)
        hashes[file.name] = digest
        arrays[name] = np.load(file, mmap_mode='r', allow_pickle=False)
    n = len(arrays['states'])
    ticks = arrays['ticks']
    assert np.array_equal(ticks, np.arange(len(ticks)) * 4)
    assert arrays['states'].shape == (n, 9) and arrays['controls'].shape == (n - 1, 2)
    assert arrays['teacher'].dtype == np.bool_ and arrays['teacher'].shape == (n - 1,)
    assert np.allclose(np.diff(arrays['states'][:, 8]), .05, rtol=0, atol=1e-5)
    cameras = {}
    for name in ('road', 'wide'):
        file = path / (name + '.npy')
        camera = np.load(file, mmap_mode='r', allow_pickle=False)
        assert camera.shape == (len(ticks), 6, 128, 256) and camera.dtype == np.uint8
        cameras[file.name] = {'path': str(file), 'shape': list(camera.shape), 'bytes': file.stat().st_size,
                              'recorded_sha256_unverified': metadata['arrays_sha256'][file.name]}
        camera._mmap.close()
    ep = SimpleNamespace(**arrays)
    context = {'np': np, 'self': ep, 'n': n, 'm': metadata, 'HORIZON': horizon}
    exec(anchor_code, context)
    eligible = np.asarray(ep.anchors, dtype=np.int64)
    if entry is not None:
        assert len(eligible) == entry['accepted'], (path, len(eligible), entry['accepted'])
        assert ep.rejections == entry['rejections'], (path, ep.rejections, entry['rejections'])
    qualified = context['qualified']
    direct = np.array([k for k, t in enumerate(ticks) if qualified and t < n - 1
                       and arrays['teacher'][t] and arrays['contacts'][t] == 0
                       and arrays['contacts'][t + 1] == 0], dtype=np.int64)
    excluded = np.setdiff1d(direct, eligible)
    policy = metadata.get('collection_policy', {})
    policy_small = {key: value for key, value in policy.items()
                    if key not in ('policy_observation_ticks', 'activation', 'takeover')}
    flag_changes = np.r_[0, np.flatnonzero(np.diff(arrays['teacher'].astype(np.int8))) + 1, n - 1]
    teacher_runs = [{'start_tick': int(a), 'end_tick_exclusive': int(b), 'teacher': bool(arrays['teacher'][a])}
                    for a, b in zip(flag_changes[:-1], flag_changes[1:])]
    row = {'dataset': dataset, 'id': metadata['id'], 'split': metadata['split'],
           'discipline': metadata['discipline'], 'path': str(path), 'meta_sha256': meta_sha,
           'small_arrays_verified_sha256': hashes, 'camera_headers_only': cameras,
           'collection_policy': policy_small, 'qualified': qualified, 'summary': metadata['summary'],
           'n_states': n, 'n_frames': len(ticks), 'teacher_runs': teacher_runs,
           'teacher_first72_mask': ''.join('1' if b else '0' for b in arrays['teacher'][:72]),
           'teacher_first72_true': int(arrays['teacher'][:72].sum()), 'rejections': ep.rejections,
           'first_state': arrays['states'][0].tolist(), 'first_control': arrays['controls'][0].tolist(),
           'coverage': {}, 'early_observations': []}
    gate_ticks = []
    gate_path = path.parent / 'recovery_gates' / (metadata['id'] + '.json')
    if dataset == 'lab_v7' and gate_path.is_file():
        gate_raw = gate_path.read_bytes()
        gate_ticks = json.loads(gate_raw)['recovery_ticks']
        row['recovery_gate'] = {'path': str(gate_path), 'sha256': hashlib.sha256(gate_raw).hexdigest()}
    elif dataset == 'new_recovery':
        receipt_path = Path(recipe['new_cases'][metadata['discipline']]['receipt'])
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
        assert receipt['passed'] is True and receipt['reasons'] == []
        assert receipt['hashes']['meta_sha256'] == meta_sha
        assert receipt['total_qualified_anchors'] == len(eligible)
        gate_ticks = receipt['qualified_anchors_first_2s']
        row['recovery_gate'] = {'path': str(receipt_path.resolve()), 'sha256': hashlib.sha256(receipt_raw).hexdigest()}
    group_frames = {'eligible': eligible, 'same_tick_teacher': direct, 'same_tick_excluded': excluded,
                    'ordinary': np.array([k for k in eligible if int(ticks[k]) not in gate_ticks], dtype=np.int64),
                    'recovery': np.array([k for k in eligible if int(ticks[k]) in gate_ticks], dtype=np.int64)}
    for group, frames in group_frames.items():
        selected_ticks = ticks[frames]
        states = arrays['states'][selected_ticks]
        controls = arrays['controls'][selected_ticks] if len(frames) else np.empty((0, 2))
        early = states[:, 8] < 3
        slow = states[:, 3] < 2
        stats = {'count': len(frames), 'earliest_tick': int(selected_ticks[0]) if len(frames) else None,
                 'earliest_time_s': float(states[0, 8]) if len(frames) else None,
                 'earliest_vx': float(states[0, 3]) if len(frames) else None,
                 't_lt_3s': int(early.sum()), 'vx_lt_2': int(slow.sum()),
                 't_lt_3s_and_vx_lt_2': int((early & slow).sum()),
                 't_lt_3s_and_positive_accel': int((early & (controls[:, 1] > 0)).sum()),
                 'vx_lt_2_and_positive_accel': int((slow & (controls[:, 1] > 0)).sum()),
                 'early_ticks': selected_ticks[early].tolist()}
        row['coverage'][group] = stats
    eligible_set = set(eligible.tolist())
    direct_set = set(direct.tolist())
    for k, tick in enumerate(ticks):
        if arrays['states'][tick, 8] >= 5:
            break
        row['early_observations'].append({'frame': k, 'tick': int(tick),
            'teacher': bool(arrays['teacher'][tick]), 'eligible': k in eligible_set,
            'same_tick_teacher': k in direct_set, 'state': arrays['states'][tick].tolist(),
            'control': arrays['controls'][tick].tolist()})
    audit['episodes'].append(row)
    for array in arrays.values():
        array._mmap.close()

for dataset in ('lab_v7', 'lab_v2', 'new_recovery'):
    for split in ('train', 'val'):
        for discipline in ('all', 'skidpad', 'trackdrive'):
            rows = [r for r in audit['episodes'] if r['dataset'] == dataset and r['split'] == split
                    and (discipline == 'all' or r['discipline'] == discipline)]
            if not rows:
                continue
            key = '/'.join((dataset, split, discipline))
            summary = {'episodes': len(rows), 'qualified': sum(r['qualified'] for r in rows),
                       'collection_modes': dict(Counter(r['collection_policy'].get('mode', 'absent') for r in rows)),
                       'initial_teacher_episodes': sum(r['teacher_first72_mask'][0] == '1' for r in rows),
                       'eligible_at_tick0_episodes': sum(r['coverage']['eligible']['earliest_tick'] == 0 for r in rows),
                       'coverage': {}}
            for group in group_frames:
                summary['coverage'][group] = {metric: sum(r['coverage'][group][metric] for r in rows)
                    for metric in ('count', 't_lt_3s', 'vx_lt_2', 't_lt_3s_and_vx_lt_2',
                                   't_lt_3s_and_positive_accel', 'vx_lt_2_and_positive_accel')}
            audit['aggregate'][key] = summary

print(json.dumps(audit, indent=2, allow_nan=False))
