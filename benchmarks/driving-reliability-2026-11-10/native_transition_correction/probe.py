"""Privileged TRAIN calibration with a frozen world reference and three predeclared native intervention arms."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path('/work/data/native/lab_v2/train-skidpad-82000-matched')
ARRAYS = ('states', 'controls', 'teacher', 'contacts', 'cones', 'ticks')
ARMS = ('op_lateral_pi', 'op_recorded_long', 'op_lateral_recorded_long')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def inventory():
    return {str(p): digest(p) for root in ('src/fsop', 'isaac_eval', 'isaac_lab_fsop')
            for p in sorted(Path(root).rglob('*.py'))}


def source():
    m = json.loads((SOURCE / 'meta.json').read_text())
    manifest = json.loads((SOURCE.parent / 'manifest.json').read_text())
    entry, = [e for e in manifest['episodes'] if e['path'] == SOURCE.name]
    require(manifest['complete'] is True and entry['meta_sha256'] == digest(SOURCE / 'meta.json'), 'manifest binding')
    require(m['split'] == 'train' and m['discipline'] == 'skidpad' and m['id'] == SOURCE.name, 'TRAIN source identity')
    require(m['checkpoint_sha256'] is None and 'collection_policy' not in m, 'source must be original teacher collection')
    require(m['membership_sha256'] == manifest['membership_sha256'] == digest('docs/isaac-lab-training/membership.json'), 'membership binding')
    require(m['case'] in json.loads(Path('docs/isaac-lab-training/membership.json').read_text())['cases'], 'case membership')
    s = m['summary']
    require(s['result'] == 'finished' and s['pass'] and s['complete'] and s['physical_cone_hits'] == s['oc'] == 0, 'source qualification')
    require(s['route']['valid'] and s['route']['accepted_route'] == ['right', 'right', 'left', 'left'], 'source route')
    a = {}
    for key in ARRAYS:
        file = SOURCE / (key + '.npy')
        require(digest(file) == m['arrays_sha256'][file.name], 'source array hash ' + key)
        a[key] = np.load(file, allow_pickle=False)
    n = len(a['states'])
    require(a['states'].shape == (n, 9) and a['controls'].shape == (n - 1, 2), 'state/control shape')
    require(a['teacher'].shape == (n - 1,) and a['teacher'].dtype == bool and a['teacher'].all(), 'all controls teacher-owned')
    require(a['contacts'].shape == (n,) and not a['contacts'].any(), 'zero source contacts')
    require(np.allclose(np.diff(a['states'][:, 8]), .05, rtol=0, atol=1e-5), 'source clock')
    require(all(np.isfinite(v).all() for v in a.values()), 'finite source')
    require(digest(SOURCE / 'source.zip') == m['source_archive_sha256'], 'producer archive')
    with ZipFile(SOURCE / 'source.zip') as archive:
        require(sorted(archive.namelist()) == sorted(m['source_sha256']), 'producer inventory')
        for key, expected in m['source_sha256'].items():
            require(hashlib.sha256(archive.read(key)).hexdigest() == expected, 'producer member ' + key)
    return m, a


def rebased_plan(states, tick, actual):
    from fsop.sim.expert import T_IDXS, plan_from_rollout

    future = np.array(states[tick:tick + 201], dtype=np.float64, copy=True)
    require(future.shape == (201, 9), 'future must contain 201 recorded native states')
    future[:, 8] -= future[0, 8]
    require(future[-1, 8] >= T_IDXS[-1] - 1e-8, 'no horizon padding')
    plan = plan_from_rollout(future, dt=.05)
    angle = float(np.arctan2(np.sin(future[0, 2] - actual[2]), np.cos(future[0, 2] - actual[2])))
    c, s = np.cos(angle), np.sin(angle)
    source_to_actual = np.array([[c, -s], [s, c]])
    c, s = np.cos(actual[2]), np.sin(actual[2])
    world_to_actual = np.array([[c, s], [-s, c]])
    for columns in ((0, 1), (3, 4), (6, 7)):
        plan[:, columns] = plan[:, columns] @ source_to_actual.T
    plan[:, :2] += world_to_actual @ (future[0, :2] - actual[:2])
    plan[:, 11] += angle
    return plan


def check_rebasing(a):
    from fsop.sim.expert import T_IDXS, plan_from_rollout

    maximum = 0.0
    for tick in (255, 440, 650, 926):
        states = a['states']
        actual = states[tick].copy()
        actual[:3] += [1.7, -2.3, .43]
        p = rebased_plan(states, tick, actual)
        f = states[tick:tick + 201].copy()
        t = f[:, 8] - f[0, 8]
        c, s = np.cos(actual[2]), np.sin(actual[2])
        world_to_actual = np.array([[c, s], [-s, c]])
        position = (f[:, :2] - actual[:2]) @ world_to_actual.T
        yaw = f[:, 2]
        velocity_world = np.c_[f[:, 3] * np.cos(yaw) - f[:, 4] * np.sin(yaw),
                               f[:, 3] * np.sin(yaw) + f[:, 4] * np.cos(yaw)]
        velocity = velocity_world @ world_to_actual.T
        acceleration = np.gradient(velocity, .05, axis=0)
        for columns, vector in (((0, 1), position), ((3, 4), velocity), ((6, 7), acceleration)):
            expected = np.stack([np.interp(T_IDXS, t, vector[:, i]) for i in range(2)], axis=1)
            maximum = max(maximum, float(np.max(np.abs(p[:, columns] - expected))))
        expected_yaw = np.interp(T_IDXS, t, np.unwrap(f[:, 2] - actual[2]))
        maximum = max(maximum, float(np.max(np.abs(np.arctan2(np.sin(p[:, 11] - expected_yaw), np.cos(p[:, 11] - expected_yaw))))))
        f[:, 8] = t
        require(np.allclose(rebased_plan(states, tick, states[tick]), plan_from_rollout(f, dt=.05), rtol=0, atol=1e-12), 'identity rebase')
    require(maximum < 1e-10, 'world-to-actual rebasing residual')
    return {'max_absolute_residual': maximum, 'tested_ticks': [255, 440, 650, 926], 'pose_offset': [1.7, -2.3, .43]}


def resources():
    require(shutil.disk_usage(HERE).free >= 25 * 2**30, '25 GiB output reserve violated')
    size = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    require(size <= 100 * 10**6, 'probe exceeded 100 MB cap')
    return {'output_filesystem': str(HERE), 'free_bytes': shutil.disk_usage(HERE).free, 'probe_bytes': size}


def freeze():
    require(not (HERE / 'protocol.json').exists(), 'refusing to replace frozen protocol')
    resources()
    m, a = source()
    current = inventory()
    old = m['source_sha256']
    diff = {'added': sorted(current.keys() - old.keys()), 'removed': sorted(old.keys() - current.keys()),
            'changed': sorted(k for k in current.keys() & old.keys() if current[k] != old[k])}
    critical = ('isaac_lab_fsop/env.py', 'isaac_lab_fsop/runtime.py', 'isaac_lab_fsop/expert.py', 'isaac_lab_fsop/cases.py')
    require(all(k in old and current[k] == old[k] for k in critical), 'native physics or geometry runtime changed')
    require(not any(k.startswith(('src/fsop/sim/', 'src/fsop/geometry/', 'src/fsop/track/', 'src/fsop/rules/')) for k in diff['changed'] + diff['added'] + diff['removed']), 'simulation geometry or referee source changed')
    write(HERE / 'runtime_inventory.json', current)
    source_files = ['meta.json', 'track.json', 'source.zip'] + [k + '.npy' for k in ARRAYS]
    frozen = {str(SOURCE / k): digest(SOURCE / k) for k in source_files}
    frozen[str(SOURCE.parent / 'manifest.json')] = digest(SOURCE.parent / 'manifest.json')
    frozen['docs/isaac-lab-training/membership.json'] = digest('docs/isaac-lab-training/membership.json')
    input_dir = HERE / 'source_snapshot'
    input_dir.mkdir()
    for name in source_files:
        shutil.copyfile(SOURCE / name, input_dir / name)
    mbranch = int(np.floor(m['summary']['route']['laps'][0]['finish_time'] / .05)) + 1
    require(mbranch == 255 and m['terminal_continuation']['start_tick'] == 927 and len(a['controls']) == 1127, 'predeclared window changed')
    controller = module('lateral_controller', 'controller_snapshot.py')
    p = {
        'schema_version': 1, 'purpose': 'Privileged TRAIN native steering executability calibration. No neural inference or learned driving result.',
        'premise': 'The transition contact comes from accumulated lateral displacement that heading-only OP cannot remove, from PI longitudinal drift, or their interaction.',
        'source': str(SOURCE), 'source_files_sha256': frozen, 'source_selection': 'Original clean teacher-only lab_v2 82000 matched. Every control teacher-owned. No learner prefix.',
        'branch_tick': mbranch, 'follower_end_tick': 927, 'recorded_end_tick': 1127, 'future_state_count': 201,
        'branch_reason': 'First control tick strictly after first right-lap crossing. Exact recorded controls for [0,255), then second right, both left laps, exit and stop through source tick 927.',
        'arms': list(ARMS), 'arm_order': 'All three arms independently reproduce the exact recorded prefix. The inherited full recorded baseline is hash-bound and verified before any new arm.',
        'reference': 'Fixed source tick equals native control tick. No nearest-point rematching, time warping, future padding, oracle refresh, or source-pose anchoring after branch.',
        'rebase': 'World positions minus actual position rotated by negative actual yaw. Source-frame velocity and acceleration rotated by source yaw minus actual yaw. Relative source yaw plus wrapped source yaw minus actual yaw. Yaw rate unchanged.',
        'future_availability': {'last_follower_control_tick': 926, 'last_required_state_tick': 1126, 'source_last_state_tick': 1127, 'terminal_continuation_start_tick': 927, 'policy': 'All 201 future states are real recorded native states. Terminal braking may enter the lookahead and is explicitly included.'},
        'steering': {'op_lateral_pi': 'OP plus atan2(2*path_normal_offset, max(actual_vx,0)+1)', 'op_recorded_long': 'unchanged OP', 'op_lateral_recorded_long': 'same lateral correction'}, 'lateral_gain': 2.0, 'lateral_error': 'Source-minus-actual position projected onto source heading normal at time zero. No path rematching. Gain inherited from existing ExpertParams.k_lat, not tuned to this run.',
        'longitudinal': 'op_lateral_pi uses unchanged PI. Other two arms replace only applied acceleration with recorded teacher control at the same tick. All arms still record the untouched PI proposal and integral evolution. Recorded-longitudinal arms are privileged causal diagnostics.',
        'pi_contract': controller.controller_contract('op_tangent_pi_v1'),
        'repeat_gate': {'max_position_m': 1e-4, 'max_yaw_rad': 1e-4, 'max_velocity_mps': 1e-4, 'max_yaw_rate_radps': 1e-4, 'max_actuator_steer_rad': 1e-4, 'max_actuator_accel_mps2': 1e-4, 'max_time_s': 1e-8},
        'recorded_acceptance': 'Inherited full recorded baseline proved exact 1128-state equality and zero contacts. Verify its hashed closure and source protocol; every new arm must independently reproduce its full 255-tick prefix.',
        'follower_acceptance': 'Primary lateral gate is accepted R,R,L,L, valid route, zero cone hits, zero OC within frozen source-time horizon. Full mission finished/pass/stopped-in-zone is a separate stricter gate. Report every arm even if it fails. No post hoc controller tuning.',
        'termination': 'Followers stop on native referee done or source tick 927. Early native failure remains a failed driving result with full available trace. No padding or extrapolation beyond horizon.',
        'render': 'Cameras and initial reset match producer. Later steps use render=False, qualified by inherited exact full recorded replay and each new independently verified prefix.',
        'limitations': 'One deterministic TRAIN source, one trial per arm. Privileged reference calibration, not neural performance, RGB equivalence, robustness or speed benchmark. Fixed clock includes position phase lag. Recorded repeat gate is the only observed native noise floor.',
        'worker_timeout_seconds': 1800, 'disk_reserve_bytes': 25 * 2**30, 'artifact_cap_bytes': 100 * 10**6,
        'producer_runtime_differences': diff, 'rebase_validation': check_rebasing(a),
    }
    inherited = Path('/scratch/fsop/native_lateral_reference1')
    receipt = json.loads((inherited / 'run1/closure_receipt.json').read_text())
    original = json.loads((inherited / 'run1/recorded/result.json').read_text())
    require(receipt['status'] == 'all_three_children_closed_and_independently_verified', 'inherited closure')
    require(original['full_mission_gate_passed'] and original['physical_cone_hits'] == 0, 'inherited recorded gate')
    require(all(value == 0 for value in original['full_source_repeat_errors'].values()), 'inherited exact replay')
    p['inherited_baseline'] = {str(inherited / name): digest(inherited / name) for name in ('protocol.json', 'runtime_inventory.json', 'run1/closure_receipt.json', 'run1/recorded/result.json', 'run1/op_tangent_pi/result.json')}
    previous_runtime = json.loads((inherited / 'runtime_inventory.json').read_text())
    inherited_diff = sorted(k for k in current.keys() | previous_runtime.keys() if current.get(k) != previous_runtime.get(k))
    require(inherited_diff == ['src/fsop/train/lab_train.py'], 'only inherited training implementation may differ; no native probe imports it')
    p['inherited_runtime_difference'] = {k: {'old': previous_runtime.get(k), 'new': current.get(k), 'relevance': 'Not imported by this privileged native controller probe'} for k in inherited_diff}
    p['files_sha256'] = {name: digest(HERE / name) for name in ('probe.py', 'controller_snapshot.py', 'closed_loop_snapshot.py', 'bridge_snapshot.py', 'runtime_inventory.json')}
    write(HERE / 'protocol.json', p)
    print(json.dumps({'protocol_sha256': digest(HERE / 'protocol.json'), 'resources': resources(), 'rebase_validation': p['rebase_validation']}, indent=2))


def preflight(p):
    resources()
    require(tuple(p['arms']) == ARMS, 'frozen intervention design')
    for name, sha in p['files_sha256'].items():
        require(digest(HERE / name) == sha, 'probe code changed ' + name)
    for name, sha in p['inherited_baseline'].items():
        require(digest(name) == sha, 'inherited baseline changed ' + name)
    for name, sha in p['source_files_sha256'].items():
        require(digest(name) == sha, 'source file changed ' + name)
    require(inventory() == json.loads((HERE / 'runtime_inventory.json').read_text()), 'runtime changed')
    m, a = source()
    require(check_rebasing(a) == p['rebase_validation'], 'rebase check changed')
    return m, a


def gate_errors(actual, expected, p):
    bridge = module('lateral_bridge', 'bridge_snapshot.py')
    errors = bridge.errors(actual, expected)
    return errors, all(v <= p['repeat_gate']['max_' + k] for k, v in errors.items())


def run_arm(p, arm, out):
    m, a = preflight(p)
    from isaac_lab_fsop.runtime import launch, shutdown

    launcher = launch(render=True)
    drive = None
    states, actions, contacts, routes, plans, plan_ticks, diagnostics, route_events = [], [], [], [], [], [], [], []
    result = {'arm': arm, 'worker_exception': None, 'prefix_gate': None, 'mission_summary': None}
    try:
        import torch
        from fsop.track.schema import CONE_CLASS_INDEX
        from isaac_lab_fsop.cases import Case, environment
        from isaac_lab_fsop.env import LabDrive

        controller = module('lateral_controller', 'controller_snapshot.py')
        closed_loop = module('lateral_closed_loop', 'closed_loop_snapshot.py')
        require(controller.controller_contract('op_tangent_pi_v1') == p['pi_contract'], 'PI contract')
        case = Case(**m['case'])
        drive = LabDrive(environment(case), variant=case.variant, render=True)
        require(json.loads(json.dumps(asdict(drive.env.vp))) == m['vehicle'], 'vehicle mismatch')
        require(json.loads(json.dumps(drive.env.scene.to_dict())) == m['scene'], 'scene mismatch')
        require(json.loads(json.dumps(drive.env.track.to_dict())) == json.loads((SOURCE / 'track.json').read_text()), 'track mismatch')
        cones = np.c_[drive.env.cones_xy, [CONE_CLASS_INDEX.get(c, -1) for c in drive.env.cone_colors]]
        require(np.array_equal(cones, a['cones']), 'realized cone mismatch')
        require(json.loads(json.dumps(drive.appearance)) == m['appearance'], 'appearance mismatch')
        runtime = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                   'physics_dt': drive.sim.cfg.dt, 'device': drive.sim.cfg.device, 'use_fabric': drive.sim.cfg.use_fabric,
                   'scene_queries': drive.sim.cfg.enable_scene_query_support, 'physx': drive.sim.cfg.physx.to_dict()}
        require(json.loads(json.dumps(runtime)) == m['runtime'], 'physics/runtime mismatch')
        require(json.loads(json.dumps(drive.sim.fsop_render_settings)) == m['render_settings'], 'render settings mismatch')
        result['runtime'] = runtime
        x, y, yaw = drive.env.track.start_pose
        obs = drive.reset(start_pose=(x - case.lateral_start * np.sin(yaw), y + case.lateral_start * np.cos(yaw), yaw + case.yaw_start))
        pi = controller.OpenpilotTangentPIFollower(drive.env.vp)
        end = p['recorded_end_tick'] if arm == 'recorded' else p['follower_end_tick']
        last_route = None
        for tick in range(end + 1):
            states.append(obs.state.as_array())
            contacts.append(len(drive.hits))
            routes.append((obs.s, obs.lat))
            route = drive.env.referee.summary().get('route')
            compact = None if route is None else {k: route.get(k) for k in ('accepted_route', 'valid', 'violation', 'complete', 'exited', 'stopped_in_zone', 'termination')}
            if compact != last_route:
                route_events.append({'tick': tick, 'route': compact})
                last_route = json.loads(json.dumps(compact))
            if tick == p['branch_tick']:
                errors, passed = gate_errors(np.asarray(states), a['states'][:tick + 1], p)
                passed = passed and bool(np.array_equal(contacts, a['contacts'][:tick + 1]))
                result['prefix_gate'] = {'passed': passed, 'errors': errors, 'tick': tick}
                require(passed, 'prefix replay gate failed')
            if drive.env.referee.done and result['mission_summary'] is None:
                result['mission_summary'] = drive.summary()
                result['mission_done_tick'] = tick
            if tick == end or (drive.env.referee.done and arm != 'recorded'):
                break
            obs = drive.observe(render=False)
            if arm == 'recorded' or tick < p['branch_tick']:
                action = tuple(float(x) for x in a['controls'][tick])
            else:
                plan = rebased_plan(a['states'], tick, np.asarray(obs.state.as_array()))
                steer, accel, diagnostic = pi.step(plan, obs.state.vx, obs.state.vy, obs.t,
                    lambda value: closed_loop.mission_brake_with_override(value, obs, drive.env))
                uncorrected = float(steer)
                normal_error = float(-np.sin(plan[0, 11]) * plan[0, 0] + np.cos(plan[0, 11]) * plan[0, 1])
                correction = (float(np.arctan2(p['lateral_gain'] * normal_error, max(obs.state.vx, 0.0) + 1.0))
                              if arm in ('op_lateral_pi', 'op_lateral_recorded_long') else 0.0)
                steer += correction
                diagnostic.update(op_steer=uncorrected, path_normal_offset=normal_error, lateral_correction=correction, pi_accel=float(accel))
                if arm in ('op_recorded_long', 'op_lateral_recorded_long'):
                    accel = float(a['controls'][tick, 1])
                diagnostic['executed_accel'] = float(accel)
                action = (float(steer), float(accel))
            if drive.env.referee.done:
                require(arm == 'recorded' and tick >= p['follower_end_tick'], 'recorded episode terminated early')
                require(np.allclose(action, (drive.act.angle, -2.0), rtol=0, atol=1e-12), 'terminal controls differ')
                obs = drive.step_terminal_brake(render=False)
            else:
                require(arm != 'recorded' or tick < p['follower_end_tick'], 'recorded mission did not finish on time')
                obs = drive.step(*action, render=False)
            actions.append(action)
            if arm != 'recorded' and tick >= p['branch_tick']:
                plans.append(plan)
                plan_ticks.append(tick)
                diagnostics.append({'tick': tick, **diagnostic, 'steering': steer})
        result['final_summary'] = drive.summary()
        require(inventory() == json.loads((HERE / 'runtime_inventory.json').read_text()), 'runtime changed during arm')
    except Exception:
        result['worker_exception'] = traceback.format_exc()
        if drive is not None:
            result['final_summary'] = drive.summary()
    finally:
        np.savez_compressed(out / 'trace.npz', states=np.asarray(states, dtype=float).reshape(-1, 9),
            actions=np.asarray(actions, dtype=float).reshape(-1, 2), contacts=np.asarray(contacts, dtype=np.int64),
            route=np.asarray(routes, dtype=float).reshape(-1, 2), plans=np.asarray(plans, dtype=float).reshape(-1, 33, 15),
            plan_ticks=np.asarray(plan_ticks, dtype=np.int64))
        write(out / 'diagnostics.json', diagnostics)
        write(out / 'route_events.json', route_events)
        result['trace_sha256'] = digest(out / 'trace.npz')
        write(out / 'worker_result.json', result)
        shutdown(launcher.app, drive)


def verify_arm(p, arm, out):
    _, a = preflight(p)
    r = json.loads((out / 'worker_result.json').read_text())
    require(r['trace_sha256'] == digest(out / 'trace.npz'), 'trace/result binding')
    z = np.load(out / 'trace.npz', allow_pickle=False)
    states, actions, contacts = z['states'], z['actions'], z['contacts']
    require(len(states) == len(actions) + 1 and len(states) == len(contacts), 'state/action length')
    require(all(np.isfinite(z[k]).all() for k in z.files), 'finite trace')
    require(np.allclose(np.diff(states[:, 8]), .05, rtol=0, atol=1e-8), 'actual clock continuity')
    branch = p['branch_tick']
    require(r['prefix_gate'] is not None and r['prefix_gate']['passed'], 'worker prefix failed')
    prefix, prefix_pass = gate_errors(states[:branch + 1], a['states'][:branch + 1], p)
    require(prefix_pass and np.array_equal(contacts[:branch + 1], a['contacts'][:branch + 1]), 'independent prefix gate')
    require(np.array_equal(actions[:branch], a['controls'][:branch]), 'prefix controls changed')
    require(r['worker_exception'] is None, str(r['worker_exception']))
    errors, same = gate_errors(states, a['states'][:len(states)], p)
    exact_contacts = bool(np.array_equal(contacts, a['contacts'][:len(states)]))
    summary = r['mission_summary'] or r['final_summary']
    route = summary.get('route', {})
    lateral_pass = (route.get('accepted_route') == ['right', 'right', 'left', 'left'] and route.get('valid') is True
                    and summary['physical_cone_hits'] == summary['oc'] == 0 and not contacts.any())
    mission_pass = lateral_pass and summary['result'] == 'finished' and summary['pass'] and summary['complete'] and route.get('stopped_in_zone')
    write(out / 'verification.json', {'arm': arm, 'independent_prefix_errors': prefix,
          'full_source_repeat_errors': errors, 'source_contacts_match': exact_contacts,
          'recorded_full_window': len(actions) == p['recorded_end_tick'],
          'lateral_gate_passed': bool(lateral_pass), 'full_mission_gate_passed': bool(mission_pass),
          'mission_done_tick': r.get('mission_done_tick'), 'native_control_ticks': len(actions)})
    if arm == 'recorded':
        require(len(actions) == p['recorded_end_tick'] and np.array_equal(actions, a['controls']), 'recorded full controls')
        require(same and exact_contacts and mission_pass and r.get('mission_done_tick') == p['follower_end_tick'], 'recorded full source repeat gate failed')
    else:
        expected_ticks = np.arange(branch, len(actions))
        require(np.array_equal(z['plan_ticks'], expected_ticks), 'every follower tick has actual rebased plan')
        require(len(z['plans']) == len(expected_ticks), 'plan count')
        for tick, plan in zip(expected_ticks, z['plans']):
            require(np.array_equal(plan, rebased_plan(a['states'], tick, states[tick])), 'saved plan differs from actual-ego world reference')
    controller = module('verify_lateral_controller', 'controller_snapshot.py')
    vp = __import__('fsop.sim.vehicle', fromlist=['VehicleParams']).VehicleParams(**source()[0]['vehicle'])
    diagnostics = json.loads((out / 'diagnostics.json').read_text())
    require(len(diagnostics) == len(z['plans']), 'one diagnostic per executed follower control')
    for tick, plan, row in zip(z['plan_ticks'], z['plans'], diagnostics, strict=True):
        expected, _ = controller.OpenpilotFollower(vp)(plan, states[tick, 3])
        error = float(np.dot(np.array([-np.sin(plan[0, 11]), np.cos(plan[0, 11])]), plan[0, :2]))
        correction = (float(np.arctan2(p['lateral_gain'] * error, max(states[tick, 3], 0.0) + 1.0))
                      if arm in ('op_lateral_pi', 'op_lateral_recorded_long') else 0.0)
        require(np.isclose(actions[tick, 0], expected + correction, rtol=0, atol=1e-12), 'independent steering action')
        require(row['tick'] == int(tick) and np.isclose(row['path_normal_offset'], error, rtol=0, atol=1e-12), 'diagnostic binding')
        if arm in ('op_recorded_long', 'op_lateral_recorded_long'):
            require(actions[tick, 1] == a['controls'][tick, 1], 'recorded longitudinal control')
        else:
            require(actions[tick, 1] == row['pi_accel'] == row['applied'], 'unchanged PI longitudinal control')
    delta = states[branch:, :2] - a['states'][branch:len(states), :2]
    speed = np.hypot(states[branch:, 3], states[branch:, 4])
    source_speed = np.hypot(a['states'][branch:len(states), 3], a['states'][branch:len(states), 4])
    r.update({'independent_prefix_errors': prefix, 'full_source_repeat_errors': errors, 'source_contacts_match': exact_contacts,
              'native_control_ticks': len(actions), 'simulated_seconds': float(states[-1, 8]), 'lateral_gate_passed': bool(lateral_pass),
              'full_mission_gate_passed': bool(mission_pass), 'fixed_clock_position_rmse_m': float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
              'fixed_clock_position_max_m': float(np.max(np.linalg.norm(delta, axis=1))), 'speed_mae_mps': float(np.mean(np.abs(speed - source_speed))),
              'physical_cone_hits': int(contacts[-1]), 'native_route': route, 'saved_transformed_plans': len(z['plans'])})
    write(out / 'result.json', r)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--worker-arm', choices=ARMS)
    parser.add_argument('--expected-protocol-sha')
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    p = json.loads((HERE / 'protocol.json').read_text())
    sha = digest(HERE / 'protocol.json')
    preflight(p)
    if args.preflight:
        print(json.dumps({'protocol_sha256': sha, 'preflight_passed': True, 'resources': resources()}, indent=2))
        return
    require(args.expected_protocol_sha == sha and args.out is not None, 'frozen protocol hash and output required')
    require(args.out.resolve().is_relative_to(HERE), 'output outside isolated probe directory')
    if args.worker_arm:
        require(args.out.is_dir(), 'worker output must exist')
        run_arm(p, args.worker_arm, args.out)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(HERE / 'protocol.json', args.out / 'protocol.json')
    results = {}
    for arm in ARMS:
        out = args.out / arm
        out.mkdir()
        receipt = {'arm': arm, 'protocol_sha256': sha, 'status': 'failed', 'child_exited': False}
        try:
            command = [sys.executable, str(HERE / 'probe.py'), '--worker-arm', arm, '--out', str(out), '--expected-protocol-sha', sha]
            with (out / 'process.log').open('w') as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=p['worker_timeout_seconds'])
            receipt.update({'child_exited': True, 'child_returncode': child.returncode})
            require(child.returncode == 0, 'native worker exit code')
            require(digest(HERE / 'protocol.json') == sha, 'protocol changed during execution')
            results[arm] = verify_arm(p, arm, out)
            receipt['status'] = 'closed_and_independently_verified'
        except Exception:
            receipt['error'] = traceback.format_exc()
        receipt['files_sha256'] = {f.name: digest(f) for f in out.iterdir() if f.is_file()}
        write(out / 'closure_receipt.json', receipt)
        if receipt['status'] != 'closed_and_independently_verified':
            write(args.out / 'FAILED.json', {'arm': arm, 'receipt': receipt, 'completed_arms': list(results), 'conclusion': 'No controller conclusion permitted. Infrastructure or prefix failure stops subsequent arms.'})
            raise RuntimeError('arm closure or calibration gate failed; retained all available evidence')
    write(args.out / 'comparison.json', {'protocol_sha256': sha, 'arms': results,
          'interpretation': 'Single-source privileged native-reference calibration. Lateral gate and full mission gate are distinct. No neural-policy result or general robustness claim.'})
    write(args.out / 'closure_receipt.json', {'status': 'all_three_children_closed_and_independently_verified', 'protocol_sha256': sha,
          'comparison_sha256': digest(args.out / 'comparison.json'), 'resources': resources()})


if __name__ == '__main__':
    main()
