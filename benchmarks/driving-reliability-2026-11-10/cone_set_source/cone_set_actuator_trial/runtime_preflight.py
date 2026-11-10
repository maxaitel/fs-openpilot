from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

from cone_core import CONTRACT, ConeSetHead, ConeSetPolicy, model_digest, model_from_payload, pack_cones, decode_batch
from fsop.model.temporal import TemporalState
from fsop.train.lab_train import sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def exact_tree(expected, actual):
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(exact_tree(expected[key], actual[key]) for key in expected)
    if isinstance(expected, str) or isinstance(actual, str):
        return expected == actual
    if expected is None or actual is None:
        return expected is actual
    try:
        return bool(np.array_equal(np.asarray(expected), np.asarray(actual)))
    except (TypeError, ValueError):
        return False


def run_runtime_preflight(checkpoint, recipe, device):
    from fsop.sim.env import Obs
    from fsop.sim.vehicle import VehicleState
    from isaac_eval.evidence import ImageHeadCapture
    from isaac_lab_fsop.cases import Case, environment
    from isaac_lab_fsop.verify_inference import capture_policy_raw
    from verify_command_episode import compare, mismatches

    package_path = Path(checkpoint)
    device = torch.device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_started = time.perf_counter()
    policy = ConeSetPolicy.load(package_path, device='cpu')
    load_seconds = time.perf_counter() - load_started
    initial = [model_digest(policy.model), model_digest(policy.head)]
    datasets, stats, timings, diagnostic_timings = [], {}, [], []
    training_device_stats = {}
    diagnostic_policy = None
    if device.type != 'cpu':
        diagnostic_package = torch.load(package_path, map_location='cpu', weights_only=False)
        diagnostic_model = model_from_payload(diagnostic_package['base_payload'], device)
        diagnostic_head = ConeSetHead().to(device).eval()
        diagnostic_head.load_state_dict(diagnostic_package['head_state'], strict=True)
        diagnostic_policy = ConeSetPolicy(diagnostic_model, diagnostic_head, CONTRACT, device)
    for discipline in ('skidpad', 'trackdrive'):
        path = Path(recipe['new_cases'][discipline]['episode'])
        metadata = read_json(path / 'meta.json')
        all_rows = read_json(path / 'proposals.json')
        rows = all_rows[:120]
        if metadata['split'] != 'train' or len(rows) < 64:
            raise ValueError('runtime preflight requires actual TRAIN RGB covering all four phases')
        images = {name: np.load(path / f'prefix_{name}_rgb.npy', mmap_mode='r', allow_pickle=False)
                  for name in ('road', 'wide')}
        try:
            env = environment(Case(**metadata['case']))
            actual_track, saved_track = json.loads(json.dumps(env.track.to_dict())), read_json(path / 'track.json')
            actual_track.pop('stats', None)
            saved_track.pop('stats', None)
            if actual_track != saved_track:
                raise ValueError('runtime TRAIN track differs')
            for name, array in images.items():
                if sha256(path / f'prefix_{name}_rgb.npy') != metadata['arrays_sha256'][f'prefix_{name}_rgb.npy']:
                    raise ValueError('runtime saved RGB differs')
                if array.dtype != np.uint8 or array.shape != (len(all_rows), 256, 512, 3):
                    raise ValueError('runtime RGB shape differs')
            def observation(index):
                row = rows[index]
                if row['tick'] != index or not np.isclose(row['state'][8], index / 20, rtol=0, atol=1e-8):
                    raise ValueError('runtime prefix ticks differ')
                for name in images:
                    if hashlib.sha256(np.ascontiguousarray(images[name][index]).tobytes()).hexdigest() != row['image_sha256'][name]:
                        raise ValueError('runtime frame image differs')
                state = VehicleState(*map(float, row['state']))
                env.state, env.frame = state.copy(), index
                env.s, env.lat, env.hint = row['progress_m'], row['lateral_m'], row['hint']
                return Obs(index, state.t, state, env.s, env.lat, {n: images[n][index] for n in images},
                           env.T_world_ego(), np.asarray(row['desire'], dtype=np.float32), [], 0.)
            policy.reset()
            warmup_started = time.perf_counter()
            with torch.inference_mode():
                for i in range(min(8, len(rows))):
                    policy(observation(i), env)
            warmup_seconds = time.perf_counter() - warmup_started
            policy.reset()
            records = []
            with torch.inference_mode():
                for i in range(len(rows)):
                    obs = observation(i)
                    t0 = time.perf_counter()
                    steer, accel, info = policy(obs, env)
                    timings.append({'case': discipline, 'frame': i, 'phase': i % 4,
                                    'seconds': time.perf_counter() - t0})
                    records.append({'action': [steer, accel], 'prediction': info})
            policy.reset()
            temporal = TemporalState()
            with capture_policy_raw(policy.model) as raw, ImageHeadCapture(policy.model) as head, torch.inference_mode():
                for i in range(len(rows)):
                    obs = observation(i)
                    raw.clear()
                    head.clear()
                    t0 = time.perf_counter()
                    steer, accel, info = policy(obs, env)
                    diagnostic_timings.append({'case': discipline, 'frame': i, 'phase': i % 4,
                                               'seconds': time.perf_counter() - t0})
                    if head.output is None or 'output' not in raw:
                        raise ValueError('runtime plan/cone diagnostics missing')
                    repeated = {'action': [steer, accel], 'prediction': info}
                    compare(True, exact_tree(records[i], repeated), stats, discipline + '.repeat_exact')
                    ctrl = info['controller']
                    reconstructed = policy.head(
                        torch.tensor(ctrl['tokens'], dtype=torch.float32).reshape(1, 200, 8),
                        torch.tensor(ctrl['mask'], dtype=torch.bool).reshape(1, 200),
                        torch.tensor(ctrl['desire'], dtype=torch.float32).reshape(1, 25, 8),
                        torch.tensor(ctrl['proprio'], dtype=torch.float32).reshape(1, 3),
                    )[0].cpu().numpy()
                    compare(np.asarray(ctrl['raw'], dtype=np.float32), reconstructed, stats, discipline + '.saved_input_head')
                    tokens, mask = pack_cones(info['cones_xy'], info['cones_cls'], info['cones_score'])
                    compare(tokens, np.asarray(ctrl['tokens'], dtype=np.float32), stats, discipline + '.saved_cone_tokens')
                    compare(mask, np.asarray(ctrl['mask'], dtype=bool), stats, discipline + '.saved_cone_mask')
                    decoded = decode_batch({cam: {key: torch.as_tensor(value) for key, value in vals.items()}
                                            for cam, vals in head.output.items()}, [env.T_ego_cam()])[0]
                    decoded_tokens, decoded_mask = pack_cones(*decoded)
                    compare(decoded_tokens, np.asarray(ctrl['tokens'], dtype=np.float32), stats, discipline + '.decoded_tokens')
                    compare(decoded_mask, np.asarray(ctrl['mask'], dtype=bool), stats, discipline + '.decoded_mask')
                    compare(temporal.push_desire(obs.desire), np.asarray(ctrl['desire'], dtype=np.float32), stats, discipline + '.route_history')
                    compare(np.asarray([obs.state.vx, obs.state.vy, obs.state.r], dtype=np.float32),
                            np.asarray(ctrl['proprio'], dtype=np.float32), stats, discipline + '.proprio')
            if diagnostic_policy is not None:
                diagnostic_policy.reset()
                with torch.inference_mode():
                    for i in range(len(rows)):
                        steer, accel, info = diagnostic_policy(observation(i), env)
                        compare(records[i], {'action': [steer, accel], 'prediction': info},
                                training_device_stats, discipline + '.cpu_vs_training_device')
            datasets.append({'case': metadata['id'], 'available_prefix_frames': len(all_rows), 'frames': len(rows), 'phases': sorted({i % 4 for i in range(len(rows))}),
                             'proposals_sha256': sha256(path / 'proposals.json'), 'warmup_frames': min(8, len(rows)),
                             'warmup_seconds': warmup_seconds,
                             'rgb_sha256': {n: sha256(path / f'prefix_{n}_rgb.npy') for n in images}})
        finally:
            for array in images.values():
                array._mmap.close()
    unchanged = initial == [model_digest(policy.model), model_digest(policy.head)]
    passed = unchanged and not mismatches(stats) and all(row['phases'] == [0, 1, 2, 3] for row in datasets)
    def timing_summary(rows):
        return {'frames': len(rows), 'mean_seconds': float(np.mean([row['seconds'] for row in rows])),
                'p95_seconds': float(np.percentile([row['seconds'] for row in rows], 95)),
                'max_seconds': max(row['seconds'] for row in rows),
                'phases': {str(phase): {'frames': len([r for r in rows if r['phase'] == phase]),
                           'mean_seconds': float(np.mean([r['seconds'] for r in rows if r['phase'] == phase]))}
                           for phase in range(4)}}
    return {'passed': passed, 'checkpoint_sha256': sha256(package_path),
               'policy_device': 'cpu', 'torch_num_threads': 4, 'dtype': 'torch.float32', 'rtol': 1e-5, 'atol': 1e-5,
               'load_seconds': load_seconds, 'datasets': datasets, 'repeat_and_head_checks': stats,
               'all_saved_state_unchanged': unchanged, 'inference': timing_summary(timings),
               'inference_with_plan_cone_capture': timing_summary(diagnostic_timings),
               'inference_per_frame': timings, 'capture_per_frame': diagnostic_timings,
               'timing_excludes': ['Isaac physics', 'rendering', 'RGB file reads', 'observation reconstruction', 'evidence writes'],
               'whole_isaac_render_capture_seconds': None, 'wall_clock_20hz_claim': False,
               'training_device_output_comparison': {'device': str(device), 'measured': diagnostic_policy is not None,
                   'stats': training_device_stats, 'qualification_gate': False,
                   'reason': 'cross-device diagnostics never relax CPU saved-input replay tolerances'},
               'limits': ['Actual saved TRAIN inputs with full sequential history, not a physical replay.',
                          'This gate does not establish mission success or a 20 Hz wall-clock rate.']}

