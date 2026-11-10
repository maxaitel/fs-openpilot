"""Collect one frozen direct-command on-policy recovery episode."""
import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
CHECKPOINT_SHA = '9112e138d4856fbbdc383d8f63dfbaeb92fb62449c4b66b5eb6f30b91ef22ba7'
RESERVE = 25 * 2**30
BUDGET = 2**30
HEAD_KEYS = tuple((camera, leaf) for camera in ('road', 'wide') for leaf in ('heat', 'wh', 'off'))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def trigger(lat, yaw):
    return abs(lat) >= 0.35 or abs(yaw) >= 0.1


def source_inventory():
    paths = [p for root in ('src/fsop', 'isaac_eval', 'isaac_lab_fsop') for p in Path(root).rglob('*.py')]
    paths += [ROOT / x for x in ('collect.py', 'verify.py', 'protocol.md', 'membership.json', 'test_bindings.py')]
    paths += [ROOT / 'cases' / f'train-{d}-82000-matched-direct-recovery-v1.json' for d in ('skidpad', 'trackdrive')]
    paths += [ROOT.parent / 'direct_command_trial' / x for x in ('actuator_trial.py', 'recipe.json')]
    paths += [Path(x) for x in ('scripts/verify_lab_recovery.py', 'scripts/audit_isaac_episode.py', 'docs/isaac-lab-training/phase_recovery_recipe4.json', 'docs/isaac-lab-training/membership.json')]
    return {str(p.resolve().relative_to(Path.cwd().resolve())): digest(p) for p in sorted(paths)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    case_data = json.loads(a.case.read_text())
    membership = json.loads((ROOT / 'membership.json').read_text())
    originals = json.loads(Path('docs/isaac-lab-training/membership.json').read_text())
    ids = [f'train-{d}-82000-matched-direct-recovery-v1' for d in ('skidpad', 'trackdrive')]
    extras = [json.loads((ROOT / 'cases' / (i + '.json')).read_text()) for i in ids]
    cases = originals['cases'] + extras
    if membership != {**originals, 'cases': cases, 'train_cases': sum(c['split'] == 'train' for c in cases)} or case_data not in extras:
        raise ValueError('Membership or case differs from the two frozen TRAIN copies')
    for c in extras:
        original = next(x for x in originals['cases'] if x['id'] == c['id'].removesuffix('-direct-recovery-v1'))
        if {**original, 'id': c['id']} != c:
            raise ValueError('Recovery case differs from original geometry or seed')
    if digest(a.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Unexpected checkpoint')
    a.out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(a.out).free < RESERVE + BUDGET:
        (a.out / 'failure.json').write_text(json.dumps({'error': 'insufficient free space', 'case': case_data, 'handoff_tick': None, 'summary': None, 'source_sha256': None, 'partial_array_counts': {}}) + '\n')
        raise RuntimeError('25 GiB reserve plus 1 GiB case budget required')
    try:
        source = source_inventory()
        with ZipFile(a.out / 'source.zip', 'x', compression=ZIP_DEFLATED) as z:
            for name in source:
                z.write(name, name)
        from isaac_lab_fsop.runtime import launch, shutdown
        launcher = launch(render=True)
    except BaseException as exc:
        (a.out / 'failure.json').write_text(json.dumps({'error': f'{type(exc).__name__}: {exc}', 'case': case_data, 'handoff_tick': None, 'summary': None, 'source_sha256': locals().get('source'), 'partial_array_counts': {}}, indent=2) + '\n')
        raise
    drive = None
    arrays = {x: [] for x in ('road', 'wide', 'prefix_road_rgb', 'prefix_wide_rgb', 'ticks', 'states', 'controls', 'teacher', 'desires', 'contacts')}
    head_arrays = {key: [] for key in HEAD_KEYS}
    rows = []
    error = None
    meta = {}
    obs = None
    handoff = None
    summary = None
    cache = None
    try:
        import numpy as np
        import torch
        from fsop.geometry.yuv import rgb_to_model
        from fsop.model.temporal import TemporalState
        from fsop.sim.expert import ExpertParams
        from fsop.track.schema import CONE_CLASS_INDEX
        from isaac_lab_fsop.cases import Case, environment
        from isaac_lab_fsop.env import LabDrive
        from isaac_lab_fsop.expert import NativeExpert
        from isaac_eval.parity_protocol import GRAPH_FILES
        from isaac_eval.evidence import ImageHeadCapture
        sys.path.insert(0, str(ROOT.parent / 'direct_command_trial'))
        from actuator_trial import NativeActuatorPolicy, model_digest

        def jsonable(v):
            if isinstance(v, np.ndarray): return v.tolist()
            if isinstance(v, np.generic): return v.item()
            if isinstance(v, dict): return {k: jsonable(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
            return v

        case = Case(**case_data)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        graphs = {str(p): digest(p) for p in GRAPH_FILES}
        policy = NativeActuatorPolicy.load(a.checkpoint, device=a.device)
        policy.model.requires_grad_(False)
        before = [model_digest(policy.model), model_digest(policy.head)]
        cache_path = (Path('data/native_tracks') / f'trackdrive_{case.track_seed:06d}.json'
                      if case.discipline == 'trackdrive' else None)
        if cache_path is not None and not cache_path.exists():
            environment(case)
        cache = ({'path': str(cache_path), 'sha256': digest(cache_path)} if cache_path else None)
        env = environment(case)
        if cache is not None:
            if digest(cache_path) != cache['sha256']:
                raise RuntimeError('track cache changed while loading')
            with ZipFile(a.out / 'source.zip', 'a', compression=ZIP_DEFLATED) as z:
                z.write(cache_path, str(cache_path))
            if digest(cache_path) != cache['sha256']:
                raise RuntimeError('track cache changed while archiving')
        drive = LabDrive(env, variant=case.variant, render=True)
        x, y, yaw = env.track.start_pose
        obs = drive.reset(start_pose=(x - case.lateral_start * np.sin(yaw), y + case.lateral_start * np.cos(yaw), yaw + case.yaw_start))
        teacher = NativeExpert(drive, ExpertParams(v_cap=8, a_lat_max=4, controller='pure_pursuit', k_lookahead=0.25))
        policy.reset()
        ts = TemporalState()
        handoff = None
        terminal_start = None
        deadline = 120 if case.discipline == 'skidpad' else 400
        started = time.perf_counter()
        proposal_bytes = 0
        (a.out / 'track.json').write_text(json.dumps(env.track.to_dict()) + '\n')
        base = policy.model.base if hasattr(policy.model, 'base') else policy.model
        original_raw = base.policy_raw
        raw = {}
        def capture_raw(*args, **kwargs):
            result = original_raw(*args, **kwargs)
            raw['value'] = [v.detach().cpu().numpy().copy() for v in result]
            return result
        base.policy_raw = capture_raw
        with ImageHeadCapture(policy.model) as head, torch.inference_mode():
            for frame in range(2400):
                estimate = (sum(v.nbytes for name in ('road', 'wide', 'prefix_road_rgb', 'prefix_wide_rgb') for v in arrays[name])
                            + sum(v.nbytes for values in head_arrays.values() for v in values)
                            + proposal_bytes + (a.out/'source.zip').stat().st_size)
                if estimate >= BUDGET - 32 * 2**20 or (frame % 40 == 0 and shutil.disk_usage(a.out).free < RESERVE + BUDGET):
                    raise RuntimeError('case budget or free-space reserve reached')
                route = env.referee.skidpad_route.report() if env.referee.skidpad_route else None
                if handoff is None and (drive.hits or env.referee.n_oc or (route is not None and not route['valid'])):
                    raise RuntimeError('dirty learner prefix')
                segment = env.mission._seg[env.hint]
                error_yaw = math.atan2(math.sin(obs.state.yaw - math.atan2(float(segment[1]), float(segment[0]))), math.cos(obs.state.yaw - math.atan2(float(segment[1]), float(segment[0]))))
                obs = drive.observe(render=True)
                arrays['states'].append(obs.state.as_array())
                arrays['contacts'].append(len(drive.hits))
                desire = ts.push_desire(obs.desire)
                if frame % 4 == 0:
                    arrays['ticks'].append(frame)
                    arrays['desires'].append(desire)
                    for name in ('road', 'wide'):
                        arrays[name].append(rgb_to_model(obs.images[name]))
                if handoff is None:
                    for name in ('road', 'wide'):
                        image = obs.images[name]
                        if image.dtype != np.uint8 or image.shape != (256, 512, 3):
                            raise ValueError('unexpected processed camera image')
                        arrays[f'prefix_{name}_rgb'].append(image.copy())
                    head.clear(); raw.clear()
                    steer, accel, info = policy(obs, env)
                    if not isinstance(head.output, dict) or set(head.output) != {'road', 'wide'}:
                        raise ValueError('missing or unexpected image head cameras')
                    for camera in ('road', 'wide'):
                        if not isinstance(head.output[camera], dict) or set(head.output[camera]) != {'heat', 'wh', 'off'}:
                            raise ValueError('missing or unexpected image head leaves')
                    for key, values in head_arrays.items():
                        value = head.output[key[0]][key[1]]
                        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or not np.isfinite(value).all():
                            raise ValueError(f'invalid image head leaf {key}')
                        if values and value.shape != values[0].shape:
                            raise ValueError(f'image head shape changed for {key}')
                        values.append(value)
                    request = teacher.act()
                    rows.append(jsonable({'tick': frame, 'state': obs.state.as_array(), 'progress_m': obs.s,
                        'hint': env.hint, 'lateral_m': obs.lat, 'yaw_rad': error_yaw, 'desire': obs.desire,
                        'image_sha256': {name: hashlib.sha256(np.ascontiguousarray(obs.images[name]).tobytes()).hexdigest() for name in ('road','wide')},
                        'proposal': [steer, accel], 'prediction': info, 'policy_raw': raw['value'], 'image_head': {'index': frame},
                        'teacher_request': request, 'route': route, 'contacts': len(drive.hits), 'off_course': env.referee.n_oc}))
                    proposal_bytes += len(json.dumps(rows[-1], allow_nan=False).encode())
                    if trigger(obs.lat, error_yaw):
                        handoff = frame
                    elif frame >= deadline:
                        raise RuntimeError('no correction trigger by deadline')
                    action = request if handoff is not None else (steer, accel)
                else:
                    action = teacher.act()
                arrays['controls'].append(action)
                arrays['teacher'].append(handoff is not None)
                obs = drive.step(*action, render=False)
                if frame % 200 == 0:
                    print('COLLECT', case.id, frame, float(obs.s), float(obs.lat), len(drive.hits), flush=True)
                if env.referee.done: break
        base.policy_raw = original_raw
        summary = drive.summary()
        if not (handoff is not None and summary.get('result') == 'finished' and summary.get('pass') is True and summary.get('complete') is True and not drive.hits and env.referee.n_oc == 0 and env.referee.n_doo == 0):
            raise RuntimeError('episode did not finish cleanly with a correction handoff')
        terminal_start = env.frame
        for _ in range(200):
            frame = env.frame
            arrays['states'].append(obs.state.as_array()); arrays['contacts'].append(len(drive.hits))
            desire = ts.push_desire(obs.desire)
            obs = drive.observe(render=True)
            if frame % 4 == 0:
                arrays['ticks'].append(frame); arrays['desires'].append(desire)
                for name in ('road', 'wide'): arrays[name].append(rgb_to_model(obs.images[name]))
            arrays['controls'].append((float(drive.act.angle), -2.0)); arrays['teacher'].append(True)
            obs = drive.step_terminal_brake(render=False)
            if drive.hits: raise RuntimeError('contact during terminal braking')
        arrays['states'].append(obs.state.as_array()); arrays['contacts'].append(len(drive.hits))
        arrays['cones'] = np.c_[env.cones_xy, np.array([CONE_CLASS_INDEX.get(c, -1) for c in env.cone_colors])]
        if cache is not None and digest(cache['path']) != cache['sha256']:
            raise RuntimeError('track cache changed during collection')
        if digest(a.checkpoint) != CHECKPOINT_SHA or before != [model_digest(policy.model), model_digest(policy.head)] or source_inventory() != source or {str(p): digest(p) for p in GRAPH_FILES} != graphs:
            raise RuntimeError('checkpoint, model, or source changed during collection')
        meta = {'schema_version': 1, 'backend': 'IsaacLab', 'id': case.id, 'split': case.split, 'discipline': case.discipline,
            'track_id': env.track.id, 'seed': case.seed, 'case': asdict(case), 'vehicle': asdict(env.vp), 'teacher': asdict(teacher.params),
            'summary': summary, 'membership_sha256': digest(ROOT/'membership.json'), 'case_sha256': digest(a.case),
            'checkpoint_sha256': CHECKPOINT_SHA, 'command_contract': policy.command_contract, 'model_digests': before,
            'collection_policy': {'mode': 'direct_command_recovery_v1', 'actual_takeover_tick': handoff, 'takeover': rows[-1]},
            'terminal_continuation': {'start_tick': terminal_start, 'end_tick': env.frame, 'contacts': len(drive.hits)},
            'source_sha256': source, 'track_cache': cache, 'source_archive_sha256': digest(a.out/'source.zip'), 'graph_sha256': graphs,
            'track_sha256': digest(a.out/'track.json'), 'control_hz': 20, 'capture_hz': 20, 'image_hz': 5,
            'appearance': drive.appearance, 'calibration': {'mount': asdict(env.mount), 'T_ego_cam': env.T_ego_cam().tolist()},
            'scene': env.scene.to_dict(), 'render_settings': drive.sim.fsop_render_settings,
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'physics_dt': drive.sim.cfg.dt, 'device': drive.sim.cfg.device, 'use_fabric': drive.sim.cfg.use_fabric, 'scene_queries': drive.sim.cfg.enable_scene_query_support, 'physx': drive.sim.cfg.physx.to_dict()},
            'wall_seconds': time.perf_counter()-started}
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        try:
            import numpy as np
            if error and obs is not None and len(arrays['states']) == len(arrays['controls']):
                arrays['states'].append(obs.state.as_array()); arrays['contacts'].append(len(drive.hits))
            for name, values in arrays.items():
                dtype = np.uint8 if name in ('road', 'wide', 'prefix_road_rgb', 'prefix_wide_rgb') else bool if name == 'teacher' else None
                np.save(a.out / f'{name}.npy', np.asarray(values, dtype=dtype))
            for (camera, leaf), values in head_arrays.items():
                np.save(a.out / f'head_{camera}_{leaf}.npy', np.asarray(values, dtype=np.float32))
            (a.out/'proposals.json').write_text(json.dumps(rows, allow_nan=False) + '\n')
            if error:
                if drive is not None:
                    try:
                        summary = drive.summary()
                    except Exception:
                        pass
                (a.out/'failure.json').write_text(json.dumps({'error': error, 'steps': len(arrays['controls']), 'case': case_data, 'handoff_tick': handoff, 'summary': summary, 'source_sha256': source, 'track_cache': cache, 'partial_array_counts': {name: len(values) for name, values in arrays.items()}}, indent=2)+'\n')
            else:
                meta['image_head_arrays'] = {f'{camera}.{leaf}': {'file': f'head_{camera}_{leaf}.npy',
                    'shape': list(np.asarray(values).shape)} for (camera, leaf), values in head_arrays.items()}
                meta['arrays_sha256'] = {p.name: digest(p) for p in a.out.glob('*.npy')}
                meta['proposals_sha256'] = digest(a.out/'proposals.json')
                (a.out/'meta.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
            size = sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file())
            if size > BUDGET or shutil.disk_usage(a.out).free < RESERVE:
                (a.out/'budget_failure.json').write_text(json.dumps({'bytes': size, 'free': shutil.disk_usage(a.out).free})+'\n')
                error = error or 'budget exceeded'
        finally:
            shutdown(launcher.app, drive)
    if error: raise RuntimeError(error)


if __name__ == '__main__':
    main()
