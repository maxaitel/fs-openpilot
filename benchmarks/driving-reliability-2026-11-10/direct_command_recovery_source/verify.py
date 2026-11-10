"""Independently qualify a direct-command recovery episode for data use."""
import argparse
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fsop.data.native_replay import NativeEpisode
from fsop.sim.expert import Expert, ExpertParams, Mission
from fsop.sim.referee import Referee
from fsop.sim.vehicle import VehicleParams, VehicleState
from fsop.sim.env import Obs
from fsop.track.schema import CONE_CLASS_INDEX
from isaac_lab_fsop.cases import Case, environment
from collect import HEAD_KEYS, source_inventory

ROOT = Path(__file__).resolve().parent
CHECKPOINT_SHA = '9112e138d4856fbbdc383d8f63dfbaeb92fb62449c4b66b5eb6f30b91ef22ba7'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def load_head_arrays(path, meta, count):
    manifest = meta['image_head_arrays']
    require(set(manifest) == {f'{camera}.{leaf}' for camera, leaf in HEAD_KEYS}, 'image head keys differ')
    result = {}
    for camera, leaf in HEAD_KEYS:
        entry = manifest[f'{camera}.{leaf}']
        filename = f'head_{camera}_{leaf}.npy'
        require(set(entry) == {'file', 'shape'} and entry['file'] == filename and filename in meta['arrays_sha256'], 'image head receipt differs')
        value = np.load(path / filename, mmap_mode='r', allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == entry['shape'] and value.ndim > 1 and value.shape[0] == count and np.isfinite(value).all(), 'image head array invalid')
        result.setdefault(camera, {})[leaf] = value
    return result


def verify_prefix_inputs(rows, images, ep, env, route):
    from fsop.geometry.yuv import rgb_to_model
    from fsop.model.temporal import TemporalState
    from fsop.sim.route_commands import RouteCommands

    temporal = TemporalState()
    commands = RouteCommands.for_mission(env.track, env.mission, env.cfg.fps)
    for i in range(len(ep.states) - 1):
        reconstructed = commands.advance(frame=i, s=route[i][0], speed=float(ep.states[i, 3]))
        if i < len(rows):
            desire = np.asarray(rows[i]['desire'], dtype=np.float32)
            require(desire.shape == (8,) and np.isfinite(desire).all(), f'invalid desire at {i}')
            require(np.array_equal(desire, reconstructed), f'route desire differs at {i}')
        pooled = temporal.push_desire(reconstructed)
        if i % 4 == 0:
            k = i // 4
            require(np.array_equal(ep.desires[k], pooled), f'packed desire differs at {i}')
            if i < len(rows):
                for name in ('road', 'wide'):
                    require(np.array_equal(getattr(ep, name)[k], rgb_to_model(images[name][i])), f'packed {name} differs at {i}')


def replay_prefix(rows, images, head_arrays, env, checkpoint, device, expected_digests):
    import torch
    from isaac_eval.evidence import ImageHeadCapture
    from isaac_lab_fsop.verify_inference import compare, capture_policy_raw
    from actuator_trial import NativeActuatorPolicy, model_digest

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = NativeActuatorPolicy.load(checkpoint, device=device)
    policy.reset()
    require([model_digest(policy.model), model_digest(policy.head)] == expected_digests, 'model digests differ')
    stats = {}

    def compare_tree(expected, actual, key):
        if isinstance(expected, dict) and isinstance(actual, dict):
            require(set(expected) == set(actual), f'{key} keys differ')
            for name in expected:
                compare_tree(expected[name], actual[name], f'{key}.{name}')
        elif isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            require(len(expected) == len(actual), f'{key} length differs')
            for i, (a, b) in enumerate(zip(expected, actual)):
                compare_tree(a, b, f'{key}[{i}]')
        elif isinstance(expected, str) or isinstance(actual, str):
            require(expected == actual, f'{key} differs')
        else:
            compare(expected, actual, stats, key)

    with capture_policy_raw(policy.model) as raw, ImageHeadCapture(policy.model) as head, torch.inference_mode():
        for i, row in enumerate(rows):
            state = VehicleState(*map(float, row['state']))
            env.state = state.copy(); env.frame = i; env.s = row['progress_m']; env.lat = row['lateral_m']; env.hint = row['hint']
            obs = Obs(i, state.t, state, env.s, env.lat, {n: images[n][i] for n in ('road', 'wide')},
                      env.T_world_ego(), np.asarray(row['desire'], dtype=np.float32), [], 0.0)
            head.clear(); raw.clear()
            steer, accel, info = policy(obs, env)
            for key, actual in (('proposal', [steer, accel]), ('prediction', info),
                                ('policy_raw', raw['output'])):
                compare_tree(row[key], actual, f'{i}.{key}')
            require(row['image_head'] == {'index': i}, f'image head reference differs at {i}')
            expected_head = {camera: {leaf: values[i] for leaf, values in leaves.items()}
                             for camera, leaves in head_arrays.items()}
            require(isinstance(head.output, dict), f'missing image head at {i}')
            compare_tree(expected_head, head.output, f'{i}.image_head')
    require(not any(v['mismatches'] for v in stats.values()), 'full-prefix model replay differs')
    require([model_digest(policy.model), model_digest(policy.head)] == expected_digests, 'model changed during replay')


def verify(path, checkpoint, device="cuda"):
    path = Path(path).resolve()
    report = {'passed': False, 'episode': str(path), 'reasons': [], 'hashes': {}}
    try:
        require(not (path/'failure.json').exists() and not (path/'budget_failure.json').exists(), 'collector reported failure')
        ep = NativeEpisode(path); m = ep.meta
        hashes = report['hashes']
        for name, file in [('meta', path/'meta.json'), ('checkpoint', checkpoint), ('collector', ROOT/'collect.py'),
                           ('verifier', ROOT/'verify.py'), ('protocol', ROOT/'protocol.md'), ('membership', ROOT/'membership.json'),
                           ('source_archive', path/'source.zip'), ('proposals', path/'proposals.json'), ('track', path/'track.json')]:
            hashes[name+'_sha256'] = digest(file)
        require(hashes['checkpoint_sha256'] == m['checkpoint_sha256'] == CHECKPOINT_SHA, 'checkpoint differs')
        require(hashes['membership_sha256'] == m['membership_sha256'], 'membership differs')
        require(hashes['source_archive_sha256'] == m['source_archive_sha256'] and hashes['proposals_sha256'] == m['proposals_sha256'] and hashes['track_sha256'] == m['track_sha256'], 'artifact hash differs')
        membership = json.loads((ROOT/'membership.json').read_text())
        original = json.loads(Path('docs/isaac-lab-training/membership.json').read_text())
        ids = [f'train-{d}-82000-matched-direct-recovery-v1' for d in ('skidpad','trackdrive')]
        extras = [json.loads((ROOT/'cases'/f'{i}.json').read_text()) for i in ids]
        cases = original['cases'] + extras
        require(membership == {**original, 'cases': cases, 'train_cases': sum(c['split'] == 'train' for c in cases)} and m['case'] in extras and m['id'] == m['case']['id'] and m['split'] == 'train', 'case membership differs')
        for c in extras:
            old = next(x for x in original['cases'] if x['id'] == c['id'].removesuffix('-direct-recovery-v1'))
            require(c == {**old, 'id': c['id']}, 'case is not original geometry and seed')
        hashes['case_sha256'] = digest(ROOT/'cases'/f"{m['id']}.json")
        require(hashes['case_sha256'] == m['case_sha256'], 'case hash differs')
        cache = m['track_cache']
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path/'source.zip') as archive:
            inventory = m['source_sha256']
            cache_name = f"data/native_tracks/trackdrive_{m['case']['track_seed']:06d}.json" if m['discipline'] == 'trackdrive' else None
            require((cache is None and cache_name is None) or (cache is not None and cache['path'] == cache_name), 'track cache identity differs')
            names = list(inventory) + ([cache_name] if cache_name else [])
            require(len(archive.namelist()) == len(names) and sorted(archive.namelist()) == sorted(names), 'source inventory differs')
            for name, sha in inventory.items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == sha, f'archive source differs: {name}')
            require(inventory == source_inventory(), 'current source inventory differs from archived producer')
            if cache_name:
                cache_bytes = archive.read(cache_name)
                require(hashlib.sha256(cache_bytes).hexdigest() == cache['sha256'], 'archived track cache differs')
                (Path(tmp)/Path(cache_name).name).write_bytes(cache_bytes)
            env = environment(Case(**m['case']), tracks=Path(tmp))
        archived_track = json.loads((path/'track.json').read_text())
        actual_track = json.loads(json.dumps(env.track.to_dict()))
        archived_track.pop('stats', None); actual_track.pop('stats', None)
        require(archived_track == actual_track and m['track_id'] == env.track.id, 'track geometry differs from case')
        require(m['vehicle'] == asdict(env.vp) and m['calibration']['mount'] == asdict(env.mount)
                and np.array_equal(m['calibration']['T_ego_cam'], env.T_ego_cam())
                and m['scene'] == json.loads(json.dumps(env.scene.to_dict())), 'environment realization differs')
        cones = np.c_[env.cones_xy, np.array([CONE_CLASS_INDEX.get(c, -1) for c in env.cone_colors])]
        require(np.array_equal(ep.cones, cones), 'realized cones differ')
        from isaac_eval.parity_protocol import GRAPH_FILES
        require(m['graph_sha256'] == {str(p): digest(p) for p in GRAPH_FILES}, 'model graph dependencies differ')
        sys.path.insert(0, str(ROOT.parent/'direct_command_trial'))
        from actuator_trial import CONTRACT
        require(m['command_contract'] == CONTRACT, 'command contract differs')
        params = ExpertParams(v_cap=8, a_lat_max=4, controller='pure_pursuit', k_lookahead=0.25)
        require(m['teacher'] == asdict(params), 'teacher parameters differ')
        require((m['control_hz'],m['capture_hz'],m['image_hz']) == (20,20,5), 'cadence differs')
        require(m['collection_policy']['mode'] == 'direct_command_recovery_v1', 'wrong collection mode')
        s = m['summary']; require(s.get('result') == 'finished' and s.get('pass') is True and s.get('complete') is True and s.get('physical_cone_hits') == 0 and s.get('oc') == 0 and s.get('doo') == 0 and not np.any(ep.contacts), 'unclean or incomplete episode')
        mission = env.mission; expert = Expert(Mission(env.track, params), env.vp)
        takeover = m['collection_policy']['actual_takeover_tick']; terminal = m['terminal_continuation']['start_tick']
        require(type(takeover) is int and 0 <= takeover < terminal and m['terminal_continuation']['end_tick'] == len(ep.teacher) and len(ep.teacher)-terminal == 200 and m['terminal_continuation']['contacts'] == 0, 'invalid takeover or terminal continuation')
        require(not np.any(ep.teacher[:takeover]) and np.all(ep.teacher[takeover:]), 'teacher flags differ')
        require(np.array_equal(ep.controls[terminal:, 1], np.full(len(ep.controls)-terminal, -2.0)), 'terminal braking differs')
        require(np.array_equal(ep.controls[terminal:, 0], ep.states[terminal:-1, 6]), 'terminal steering request differs from previous actual angle')
        require(np.all(ep.controls[terminal:, 0] == ep.states[terminal, 6]), 'terminal steering did not remain at the first actual angle')
        require(np.all(np.hypot(ep.states[terminal:-1, 3], ep.states[terminal:-1, 4]) < 0.5), 'terminal braking started above speed limit')
        hint, progress = mission.init_progress(float(ep.states[0,0]), float(ep.states[0,1]))
        route = []
        for tick, values in enumerate(ep.states):
            hint, progress, lat = mission.update_progress(float(values[0]), float(values[1]), hint, progress)
            seg = mission._seg[hint]; heading = math.atan2(float(seg[1]),float(seg[0])); yaw = math.atan2(math.sin(float(values[2])-heading),math.cos(float(values[2])-heading))
            route.append((progress, lat, yaw, hint))
            if takeover <= tick < terminal:
                request = expert.act(VehicleState(*map(float, values)), progress)
                require(np.allclose(request, ep.controls[tick], atol=1e-5, rtol=0), f'teacher control differs at {tick}')
        first = next((i for i, (_, lat, yaw, _) in enumerate(route[:takeover+1]) if abs(lat)>=.35 or abs(yaw)>=.1), None)
        require(first == takeover and takeover <= (120 if m['discipline']=='skidpad' else 400), 'first trigger or deadline differs')
        rows = json.loads((path/'proposals.json').read_text())
        require(len(rows) == takeover+1 and [r['tick'] for r in rows] == list(range(takeover+1)), 'proposal history incomplete')
        images = {}
        for name in ('road', 'wide'):
            key = f'prefix_{name}_rgb.npy'
            require(key in m['arrays_sha256'], 'prefix camera array is not hashed')
            images[name] = np.load(path/key, mmap_mode='r', allow_pickle=False)
            require(images[name].dtype == np.uint8 and images[name].shape == (takeover+1, 256, 512, 3), 'prefix camera shape or dtype differs')
        head_arrays = load_head_arrays(path, m, takeover+1)
        verify_prefix_inputs(rows, images, ep, env, route)
        for i, row in enumerate(rows):
            require(np.allclose(row['state'], ep.states[i], atol=1e-9, rtol=0) and np.allclose([row['progress_m'],row['lateral_m'],row['yaw_rad']], route[i][:3], atol=1e-4, rtol=0) and row['hint'] == route[i][3] and row['contacts'] == row['off_course'] == 0, f'proposal snapshot differs at {i}')
            require(set(row['image_sha256']) == {'road','wide'} and all(len(x)==64 for x in row['image_sha256'].values()) and row['policy_raw'] and row['image_head'] == {'index': i} and 'controller' in row['prediction'], 'proposal evidence incomplete')
            require(np.allclose(row['teacher_request'], expert.act(VehicleState(*map(float, ep.states[i])), route[i][0]), atol=1e-5, rtol=0), f'proposal teacher request differs at {i}')
            for name in ('road', 'wide'):
                require(hashlib.sha256(np.ascontiguousarray(images[name][i]).tobytes()).hexdigest() == row['image_sha256'][name], f'prefix image differs at {i}')
            if i < takeover: require(np.allclose(row['proposal'], ep.controls[i], atol=1e-7, rtol=0), 'executed learner action differs')
        require(np.allclose(rows[-1]['teacher_request'], ep.controls[takeover], atol=1e-7, rtol=0), 'handoff action differs')
        from shapely.geometry import Point
        referee = Referee(env.track, env.cones_xy, env.cone_colors, env.vp)
        for tick, values in enumerate(ep.states[1:terminal+1], start=1):
            referee.update(VehicleState(*map(float,values)), mission_done=route[tick][0] >= mission.s_end - .5)
            if tick <= takeover:
                require(referee.n_oc == 0 and (referee.skidpad_route is None or referee.skidpad_route.report()['valid']), 'invalid independent prefix')
                if tick == takeover and referee.skidpad_route is not None:
                    require(rows[-1]['route'] == referee.skidpad_route.report(), 'handoff route snapshot differs')
        independent = referee.summary()
        require(all(independent[k] == s[k] for k in independent), 'independent referee summary differs')
        require(independent['result'] == 'finished' and independent['pass'] and independent['complete'] and referee.n_oc == 0 and referee.n_doo == 0, 'independent referee did not finish cleanly')
        vp = VehicleParams(**m['vehicle'])
        continuation_referee = Referee(env.track, env.cones_xy, env.cone_colors, env.vp)
        for values in ep.states[terminal+1:]:
            continuation_referee.done = False
            continuation_referee.update(VehicleState(*map(float, values)))
            require(continuation_referee.n_doo == 0 and continuation_referee.n_oc == 0, 'independent terminal contact or off-course')
            c, si = math.cos(float(values[2])), math.sin(float(values[2]))
            wheels = vp.wheels() @ np.array([[c, -si], [si, c]]).T + values[:2]
            require(any(referee.poly.contains(Point(point)) for point in wheels), 'off-course during terminal braking')
        if m['discipline'] == 'skidpad':
            r = referee.skidpad_route.report()
            require(r['valid'] and r['accepted_route'] == ['right','right','left','left'] and r['complete'] and r['exited'] and r['stopped_in_zone'], 'independent skidpad route failed')
        replay_prefix(rows, images, head_arrays, env, checkpoint, device, m['model_digests'])
        anchors = [int(ep.ticks[k]) for k in ep.anchors if takeover <= ep.ticks[k] < takeover+40 and ep.ticks[k] < terminal]
        report['qualified_anchors_first_2s'] = anchors; report['total_qualified_anchors'] = len(ep.anchors)
        require(len(anchors)>=8, 'fewer than eight qualified recovery anchors')
        settled = next((i for i in range(takeover, min(terminal, takeover+80)-9) if all(abs(route[j][1])<=.2 and abs(route[j][2])<=.1 for j in range(i,i+11))), None)
        require(settled is not None, 'did not settle for 0.5 s within four seconds')
        report.update(takeover_tick=takeover, settled_start_tick=settled, terminal_start_tick=terminal, arrays_sha256=m['arrays_sha256'], passed=True)
    except Exception as exc:
        report['reasons'].append(f'{type(exc).__name__}: {exc}')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--episode',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.out.resolve().is_relative_to(a.episode.resolve()): p.error('receipt must be outside the sealed episode')
    result=verify(a.episode,a.checkpoint,a.device)
    with a.out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result,indent=2)); sys.exit(0 if result['passed'] else 1)

if __name__ == '__main__': main()
