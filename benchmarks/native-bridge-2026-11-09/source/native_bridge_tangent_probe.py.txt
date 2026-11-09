"""Run the frozen experimental tangent-longitudinal native bridge probe. One native process per arm; see protocol JSON.

Run only in the pinned Isaac Lab container, from its repository root. Nothing runs on import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

PROTOCOL = Path('.work/lab/native_bridge_tangent_protocol.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_protocol(path):
    p = json.loads(path.read_text())
    check(p['schema_version'] == 1 and len(p['cases']) == 3 and p['arms'] == ['recorded', 'teacher', 'pp_tangent', 'op_tangent'], 'unexpected protocol')
    baseline = p['baseline_probe']
    for key in ('script', 'protocol'):
        check(digest(baseline[key]) == baseline[key + '_sha256'], f'baseline {key} hash mismatch')
    for key in ('manifest', 'recipe', 'membership'):
        check(digest(p[key]) == p[key + '_sha256'], f'{key} hash mismatch')
    return p


def verify_source(p, spec):
    from fsop.data.native_replay import NativeEpisode

    manifest = json.loads(Path(p['manifest']).read_text())
    check(manifest['complete'] is True and manifest['membership_sha256'] == p['membership_sha256'], 'manifest')
    matches = [r for r in manifest['episodes'] if r['path'] == spec['case_id']]
    check(len(matches) == 1, 'manifest case identity')
    folder = Path(p['manifest']).parent / spec['case_id']
    gate_path = folder.parent / 'recovery_gates' / (spec['case_id'] + '.json')
    gate = json.loads(gate_path.read_text())
    check(gate['passed'] is True and gate['case_id'] == spec['case_id'], 'recovery gate')
    check(gate['hashes']['recipe_sha256'] == p['recipe_sha256'] and gate['hashes']['membership_sha256'] == p['membership_sha256'], 'gate recipe')
    check(digest(folder / 'meta.json') == matches[0]['meta_sha256'] == gate['hashes']['meta_sha256'], 'metadata digest')
    ep = NativeEpisode(folder)  # checks all array hashes, shapes, cadence and qualification
    m = ep.meta
    check(m['split'] == 'train' and m['id'] == spec['case_id'], 'not selected train case')
    check(m['case'] in json.loads(Path(p['membership']).read_text())['cases'], 'case membership')
    check(m['membership_sha256'] == p['membership_sha256'] and m['arrays_sha256'] == gate['hashes']['arrays_sha256'], 'array identity')
    recipe = json.loads(Path(p['recipe']).read_text())
    check(m['teacher'] == recipe['teachers'][spec['case_id']], 'teacher parameters')
    check(m['checkpoint_sha256'] == recipe['checkpoint_sha256'], 'learner prefix checkpoint')
    check(m['collection_policy']['mode'] == 'learner_prefix', 'not learner prefix')
    takeover = gate['takeover_tick']
    check(type(takeover) is int and takeover == m['collection_policy']['actual_takeover_tick'], 'takeover')
    check(not ep.teacher[:takeover].any() and ep.teacher[takeover:].all(), 'teacher flags')
    check(m['summary']['result'] == 'finished' and m['summary']['pass'] is True and m['summary']['complete'] is True and m['summary']['physical_cone_hits'] == 0 and m['summary']['oc'] == 0, 'episode qualification')
    check(digest(folder / 'source.zip') == m['source_archive_sha256'] == gate['hashes']['source_archive_sha256'], 'source archive hash')
    with ZipFile(folder / 'source.zip') as z:
        check(sorted(z.namelist()) == sorted(m['source_sha256']), 'source inventory')
        for name, expected in m['source_sha256'].items():
            check(hashlib.sha256(z.read(name)).hexdigest() == expected, 'archive member ' + name)
    check(digest(folder / 'track.json') == gate['hashes']['track_sha256'], 'track hash')
    check(digest(Path('docs/isaac-lab-training/cases') / (spec['case_id'] + '.json')) == gate['hashes']['case_sha256'], 'case hash')
    terminal = m['terminal_continuation']['start_tick']
    end = p['window_ticks'] + p['future_ticks']
    last = min(terminal - end, len(ep.controls) - end)
    if spec['name'] == 'recovery':
        anchor = takeover
    else:
        anchor = None
        for tick in range(takeover + 80, last + 1):
            segment = np.asarray(ep.states[tick:tick + 21])
            speed = segment[:, 3]
            yawrate = segment[:, 5]
            if spec['name'] == 'straight':
                eligible = np.all(speed >= 4) and np.all(np.abs(yawrate) <= .08)
            else:
                eligible = np.all(speed >= 4) and (np.all(yawrate >= .25) or np.all(yawrate <= -.25))
            if eligible:
                anchor = tick
                break
        check(anchor is not None, 'no qualifying predeclared kinematic window')
    check(anchor == p['baseline_probe']['anchors'][spec['name']], 'window differs from frozen probe 1')
    check(takeover <= anchor <= last, 'insufficient nonterminal future')
    check(ep.teacher[anchor:anchor + end].all() and not ep.contacts[:anchor + end + 1].any(), 'unqualified reference window')
    return ep, gate, anchor


def reference_plan(states, tick, actual):
    """Actual future sampled at source clock, rigidly expressed in current ego."""
    from fsop.sim.expert import T_IDXS, plan_from_rollout

    future = np.array(states[tick:tick + 201], dtype=np.float64, copy=True)
    check(future.shape == (201, 9) and np.allclose(np.diff(future[:, 8]), .05, rtol=0, atol=1e-5), 'reference future missing or discontinuous')
    future[:, 8] -= future[0, 8]
    check(future[-1, 8] + 1e-8 >= T_IDXS[-1], 'reference would require padding')
    plan = plan_from_rollout(future, dt=.05)
    angle = future[0, 2] - actual[2]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    ca, sa = np.cos(actual[2]), np.sin(actual[2])
    translation = np.array([[ca, sa], [-sa, ca]]) @ (future[0, :2] - actual[:2])
    for cols in ((0, 1), (3, 4), (6, 7)):
        plan[:, list(cols)] = plan[:, list(cols)] @ rot.T
    plan[:, :2] += translation
    plan[:, 11] += angle
    return plan


def tangent_longitudinal(plan, actual_vx, actual_vy, vp):
    """Experimental tangent speed/accel at knots; return clipped action and audit values."""
    from fsop.sim.expert import T_IDXS

    check(plan.shape == (len(T_IDXS), 15) and np.isfinite(plan).all(), 'invalid plan')
    check(np.isfinite([actual_vx, actual_vy, vp.max_brake, vp.max_accel]).all(), 'invalid longitudinal inputs')
    check(vp.max_brake > 0 and vp.max_accel > 0, 'invalid acceleration limits')
    velocity = plan[:, 3:5]
    acceleration = plan[:, 6:8]
    speed = np.linalg.norm(velocity, axis=1)
    tangent = np.einsum('ij,ij->i', velocity, acceleration) / np.maximum(speed, 1e-3)
    target_speed = float(np.interp(.3, T_IDXS, speed))
    target_tangent_accel = float(np.interp(.3, T_IDXS, tangent))
    current_speed = float(np.hypot(actual_vx, actual_vy))
    raw = 1.5 * (target_speed - current_speed) + .5 * target_tangent_accel
    clipped = float(np.clip(raw, -vp.max_brake, vp.max_accel))
    return clipped, (target_speed, target_tangent_accel, current_speed, raw, clipped)


def errors(actual, source):
    delta = actual - source
    delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
    return {'position_m': float(np.max(np.linalg.norm(delta[:, :2], axis=1))),
            'yaw_rad': float(np.max(np.abs(delta[:, 2]))),
            'velocity_mps': float(np.max(np.linalg.norm(delta[:, 3:5], axis=1))),
            'yaw_rate_radps': float(np.max(np.abs(delta[:, 5]))),
            'actuator_steer_rad': float(np.max(np.abs(delta[:, 6]))),
            'actuator_accel_mps2': float(np.max(np.abs(delta[:, 7]))),
            'time_s': float(np.max(np.abs(delta[:, 8])))}


def native_arm(p, spec, arm, output):
    ep, gate, anchor = verify_source(p, spec)
    from isaac_lab_fsop.runtime import launch, shutdown
    launcher = launch(render=True)
    app, drive = launcher.app, None
    try:
        import torch
        from fsop.eval.controller import OpenpilotFollower, PlanPurePursuit
        from fsop.eval.closed_loop import mission_brake
        from fsop.sim.expert import ExpertParams
        from fsop.track.schema import CONE_CLASS_INDEX
        from isaac_lab_fsop.cases import Case, environment
        from isaac_lab_fsop.env import LabDrive
        from isaac_lab_fsop.evaluate import source_inventory
        from isaac_lab_fsop.expert import NativeExpert

        m = ep.meta
        sources = source_inventory()
        check(sources == m['source_sha256'], 'native producer source inventory mismatch')
        case = Case(**m['case'])
        drive = LabDrive(environment(case), variant=case.variant, render=True)
        check(json.loads(json.dumps(asdict(drive.env.vp))) == m['vehicle'] and json.loads(json.dumps(drive.env.scene.to_dict())) == m['scene'], 'vehicle/scene mismatch')
        check(json.loads(json.dumps(drive.env.track.to_dict())) == json.loads((ep.path / 'track.json').read_text()), 'track mismatch')
        cones = np.c_[drive.env.cones_xy, [CONE_CLASS_INDEX.get(c, -1) for c in drive.env.cone_colors]]
        check(np.array_equal(cones, ep.cones), 'realized cones mismatch')
        check(json.loads(json.dumps(drive.appearance)) == m['appearance'], 'appearance/physics hash mismatch')
        runtime = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                   'physics_dt': drive.sim.cfg.dt, 'device': drive.sim.cfg.device,
                   'use_fabric': drive.sim.cfg.use_fabric, 'scene_queries': drive.sim.cfg.enable_scene_query_support,
                   'physx': drive.sim.cfg.physx.to_dict()}
        check(json.loads(json.dumps(runtime)) == m['runtime'] and json.loads(json.dumps(drive.sim.fsop_render_settings)) == m['render_settings'], 'runtime/physics/render mismatch')
        x, y, yaw = drive.env.track.start_pose
        pose = (x - case.lateral_start * np.sin(yaw), y + case.lateral_start * np.cos(yaw), yaw + case.yaw_start)
        obs = drive.reset(start_pose=pose)
        expert = NativeExpert(drive, ExpertParams(**m['teacher']))
        follower = OpenpilotFollower(drive.env.vp) if arm == 'op_tangent' else PlanPurePursuit(drive.env.vp)
        states, actions, contacts, route, pre_mission_actions, tangent_audit = [], [], [], [], [], []
        end = anchor + p['window_ticks']
        for tick in range(end):
            states.append(obs.state.as_array())
            contacts.append(len(drive.hits))
            route.append((obs.s, obs.lat))
            obs = drive.observe(render=False)  # no post-reset capture; native repeat gate tests physics parity
            if tick == anchor:
                prefix = errors(np.asarray(states), np.asarray(ep.states[:anchor + 1]))
                prefix_ok = all(prefix[k] <= p['repeat_gate']['max_' + k] for k in prefix)
                prefix_contacts_match = bool(np.array_equal(contacts, ep.contacts[:anchor + 1]))
                np.savez_compressed(output / 'prefix_trace.npz', states=np.asarray(states), actions=np.asarray(actions),
                                    contacts=np.asarray(contacts), route=np.asarray(route))
                (output / 'prefix_gate.json').write_text(json.dumps({
                    'anchor_tick': anchor, 'errors': prefix, 'limits': p['repeat_gate'],
                    'contacts_match_source': prefix_contacts_match,
                    'passed': prefix_ok and prefix_contacts_match,
                    'trace_sha256': digest(output / 'prefix_trace.npz')}, indent=2) + '\n')
                check(prefix_ok and prefix_contacts_match, 'arm prefix repeat gate failed before intervention')
            if tick < anchor or arm == 'recorded':
                action = ep.controls[tick]
            elif arm == 'teacher':
                action = expert.act()
            else:
                plan = reference_plan(ep.states, tick, np.asarray(obs.state.as_array()))
                steer, _ = follower(plan, float(obs.state.vx))
                accel, tangent_row = tangent_longitudinal(plan, float(obs.state.vx), float(obs.state.vy), drive.env.vp)
                tangent_audit.append(tangent_row)
                pre_mission_action = (steer, accel)
                action = (steer, mission_brake(accel, obs, drive.env))
            if tick < anchor or arm in ('recorded', 'teacher'):
                pre_mission_action = action
            pre_mission_actions.append(pre_mission_action)
            actions.append(action)
            obs = drive.step(*action, render=False)
            if drive.env.referee.done:
                break
        states.append(obs.state.as_array())
        contacts.append(len(drive.hits))
        route.append((obs.s, obs.lat))
        states, actions, contacts, route, pre_mission_actions = map(np.asarray, (states, actions, contacts, route, pre_mission_actions))
        np.savez_compressed(output / 'trace.npz', states=states, actions=actions, contacts=contacts, route=route, pre_mission_actions=pre_mission_actions, tangent_audit=np.asarray(tangent_audit, dtype=float).reshape(-1, 5))
        n = len(states)
        metrics = errors(states, np.asarray(ep.states[:n]))
        same_contacts = bool(np.array_equal(contacts, ep.contacts[:n]))
        full = len(actions) == end
        gate_pass = full and same_contacts and all(metrics[k] <= p['repeat_gate']['max_' + k] for k in metrics)
        active = actions[anchor:]
        source_actions = np.asarray(ep.controls[anchor:anchor + len(active)])
        action_mae = np.mean(np.abs(active - source_actions), axis=0).tolist() if len(active) else None
        check(source_inventory() == sources, 'native source changed during run')
        # Reconstruct source progress in recorded order, retaining the route hint at overlaps.
        mission = drive.env.mission
        hint, progress = mission.init_progress(*ep.states[0, :2])
        source_progress = []
        for row in ep.states[:anchor + p['window_ticks'] + 1]:
            hint, progress, _ = mission.update_progress(row[0], row[1], hint, progress)
            source_progress.append(progress)
        source_progress = np.asarray(source_progress)
        nearest = []
        for state, actual_route in zip(states[anchor:], route[anchor:]):
            candidates = np.arange(anchor, anchor + p['window_ticks'] + 1)
            index = int(candidates[np.argmin(np.abs(source_progress[candidates] - actual_route[0]))])
            nearest.append((index, float(source_progress[index] - actual_route[0]),
                            float(np.linalg.norm(state[:2] - ep.states[index, :2]))))
        nearest = np.asarray(nearest, dtype=float).reshape(-1, 3)
        np.save(output / 'nearest_progress.npy', nearest)
        result = {'case': spec['name'], 'case_id': spec['case_id'], 'arm': arm, 'anchor_tick': anchor,
                  'takeover_tick': gate['takeover_tick'], 'end_tick': len(actions), 'full_window': full,
                  'repeat_gate_pass': gate_pass if arm in ('recorded', 'teacher') else None, 'source_errors_from_reset': metrics,
                  'source_errors_active_window': errors(states[anchor:], np.asarray(ep.states[anchor:n])) if n > anchor else None,
                  'contacts_match_source': same_contacts, 'action_mae_active_vs_source': action_mae,
                  'tangent_audit_columns': ['target_speed_mps', 'target_tangent_accel_mps2', 'actual_speed_mps', 'raw_accel_mps2', 'clipped_accel_mps2'],
                  'mission_accel_override_count_active': int(np.count_nonzero(np.abs(actions[anchor:, 1] - pre_mission_actions[anchor:, 1]) > 1e-12)),
                  'nearest_progress_columns': ['source_tick', 'source_minus_actual_progress_m', 'position_error_m'],
                  'nearest_progress_sha256': digest(output / 'nearest_progress.npy'),
                  'referee': drive.summary(), 'runtime': runtime, 'appearance': drive.appearance,
                  'source_meta_sha256': digest(ep.path / 'meta.json'), 'trace_sha256': digest(output / 'trace.npz')}
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        shutdown(app, drive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--expected-protocol-sha')
    parser.add_argument('--expected-script-sha')
    parser.add_argument('--worker-case', choices=['straight', 'steady_turn', 'recovery'])
    parser.add_argument('--worker-arm', choices=['recorded', 'teacher', 'pp_tangent', 'op_tangent'])
    args = parser.parse_args()
    p = read_protocol(args.protocol)
    if args.worker_case or args.worker_arm:
        check(digest(args.protocol) == args.expected_protocol_sha and digest(__file__) == args.expected_script_sha, 'worker protocol or script hash mismatch')
        check(args.worker_case and args.worker_arm and args.out.is_dir(), 'incomplete worker arguments')
        native_arm(p, next(x for x in p['cases'] if x['name'] == args.worker_case), args.worker_arm, args.out)
        check(digest(args.protocol) == args.expected_protocol_sha and digest(__file__) == args.expected_script_sha, 'worker protocol or script changed during run')
        return
    args.out.mkdir(parents=True, exist_ok=False)
    # Validate all source artifacts and freeze selections before starting native simulation.
    selections = {}
    for spec in p['cases']:
        _, _, anchor = verify_source(p, spec)
        selections[spec['name']] = anchor
    (args.out / 'protocol.json').write_bytes(args.protocol.read_bytes())
    protocol_sha = digest(args.out / 'protocol.json')
    script_sha = digest(__file__)
    (args.out / 'selection.json').write_text(json.dumps({'anchors': selections, 'probe_sha256': script_sha, 'protocol_sha256': protocol_sha}, indent=2) + '\n')
    for arm in p['arms']:
        for spec in p['cases']:
            target = args.out / (spec['name'] + '_' + arm)
            target.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), '--out', str(target.resolve()), '--worker-case', spec['name'], '--worker-arm', arm, '--protocol', str((args.out / 'protocol.json').resolve()), '--expected-protocol-sha', protocol_sha, '--expected-script-sha', script_sha]
            with (target / 'process.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=p['worker_timeout_seconds'])
            result = json.loads((target / 'result.json').read_text())
            check(result['anchor_tick'] == selections[spec['name']], 'selection changed')
            if arm in ('recorded', 'teacher'):
                check(result['repeat_gate_pass'], arm + ' native repeat gate failed; no contract claim')
            check(digest(args.out / 'protocol.json') == protocol_sha and digest(__file__) == script_sha, 'protocol or script changed during experiment')
    (args.out / 'COMPLETE').write_text('All recorded and teacher repeat gates passed; all prescribed arms completed. Interpret according to protocol.\n')


if __name__ == '__main__':
    main()
