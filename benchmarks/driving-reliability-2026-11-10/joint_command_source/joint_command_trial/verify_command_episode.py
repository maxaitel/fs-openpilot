"""Replay every saved direct-command input without launching physics or rendering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from joint_core import KIND, CONTRACT, JointActuatorPolicy
from fsop.eval.closed_loop import mission_brake_with_override
from isaac_eval.evidence import ImageHeadCapture, read_evidence, verified_summary
from isaac_eval.parity_cases import realized_environment, sha256_file
from isaac_eval.parity_protocol import GRAPH_FILES
from isaac_eval.verify_inference import observation, state_digest
from isaac_lab_fsop.cases import Case, environment
from isaac_lab_fsop.verify_inference import capture_policy_raw, compare as numeric_compare

RTOL = ATOL = 1e-5
PILOT_FRAMES = 120


def compare(expected, actual, stats, path):
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            stats.setdefault(path + '.keys', {'max_abs': None, 'mismatches': 0})['mismatches'] += 1
        for key in sorted(expected.keys() & actual.keys()):
            compare(expected[key], actual[key], stats, f'{path}.{key}')
    elif isinstance(expected, str) or isinstance(actual, str):
        stats.setdefault(path, {'max_abs': None, 'mismatches': 0})['mismatches'] += int(expected != actual)
    else:
        numeric_compare(expected, actual, stats, path)


def mismatches(stats):
    return sum(row['mismatches'] for row in stats.values())


def verify(episode, checkpoint, output, max_frames='120', device='cpu'):
    if device != 'cpu':
        raise ValueError('Joint replay requires the frozen CPU backend')
    episode, checkpoint, output = Path(episode), Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite verification receipt: {output}')
    evidence = episode / 'evidence'
    metadata = json.loads((evidence / 'metadata.json').read_text())
    summary = verified_summary(evidence)
    if summary is None:
        raise ValueError('Evidence requires a completion receipt')
    receipt = json.loads((evidence / 'completion.json').read_text())
    if (evidence / receipt['summary_path']).resolve() != (episode / 'summary.json').resolve():
        raise ValueError('Completion receipt does not bind this episode summary')
    entries = [json.loads(line) for line in (evidence / 'manifest.jsonl').read_text().splitlines()]
    count_expected = len(entries)
    if receipt['frame_count'] != count_expected or not count_expected:
        raise ValueError('A nonempty sealed episode is required')
    if max_frames != 'all' and (int(max_frames) != PILOT_FRAMES or count_expected != PILOT_FRAMES):
        raise ValueError('Use 120 for a complete pilot or all for a complete sealed episode')
    if max_frames != 'all' and not (metadata.get('pilot_only') and summary.get('pilot_only')):
        raise ValueError('The 120-frame mode requires pilot evidence')
    if metadata.get('schema_version') != 5 or metadata.get('package_kind') != KIND or metadata.get('controller_name') != KIND or metadata.get('command_contract') != CONTRACT:
        raise ValueError('Evidence is not the direct-command schema')
    if metadata.get('policy_device') != 'cpu' or metadata.get('torch_num_threads') != 4 or summary.get('policy_device') != 'cpu' or summary.get('torch_num_threads') != 4 or summary.get('package_kind') != KIND:
        raise ValueError('Producer did not use the fixed CPU runtime')
    if metadata.get('backend') != 'Isaac Lab PhysX' or metadata.get('fps') != 20:
        raise ValueError('Evidence is not a 20 Hz native Lab episode')
    if metadata['case'] != summary['case'] or metadata['command_contract'] != summary['command_contract']:
        raise ValueError('Summary does not match the case or command contract')
    display_contract = {'mode': metadata['controller_name'], 'command_contract': metadata['command_contract']}
    if metadata.get('controller_contract') != display_contract or summary.get('controller_contract') != display_contract:
        raise ValueError('Direct-command display contract differs')
    if metadata['controller_name'] != summary['controller_name']:
        raise ValueError('Summary controller identity differs')
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != metadata['checkpoint_sha256'] or checkpoint_sha != summary['checkpoint_sha256']:
        raise ValueError('Package hash differs from evidence')
    if (summary.get('frozen_unchanged') is not True or summary.get('checkpoint_sha256_after') != checkpoint_sha
            or summary.get('state_sha256_after') != metadata['state_sha256']
            or summary.get('head_state_sha256_after') != metadata['head_state_sha256']):
        raise ValueError('Producer recorded a changed package, base or head')
    if checkpoint.stat().st_size != metadata['checkpoint_size_bytes']:
        raise ValueError('Package size differs from evidence')
    graphs = metadata['graph_sha256']
    if set(graphs) != set(GRAPH_FILES):
        raise ValueError('Graph inventory is incomplete')
    sources = metadata['source_sha256']
    from evaluate import source_inventory
    required = set(source_inventory())
    if set(sources) != required:
        raise ValueError('Recorded source inventory differs from the required inventory')
    for paths in (graphs, sources):
        for name, digest in paths.items():
            if sha256_file(name) != digest:
                raise ValueError(f'Executable source or graph changed: {name}')
    with ZipFile(episode / 'source.zip') as archive:
        if set(archive.namelist()) != set(sources):
            raise ValueError('Archived source inventory differs from metadata')
        import hashlib
        for name, digest in sources.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f'Archived source differs: {name}')
    env = environment(Case(**metadata['case']))
    realization = episode / 'realization.json'
    if sha256_file(realization) != metadata['realization_sha256'] or realized_environment(env) != json.loads(realization.read_text()):
        raise ValueError('Case environment differs from the recorded realization')
    if not np.array_equal(metadata['nominal_detection_calibration'], env.T_ego_cam()):
        raise ValueError('Nominal calibration differs from the reconstructed case')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    captured = JointActuatorPolicy.load(checkpoint, device=device)
    plain = JointActuatorPolicy.load(checkpoint, device=device)
    if captured.command_contract != metadata['command_contract'] or captured.controller_name != metadata['controller_name']:
        raise ValueError('Loaded package contract differs from the evidence')
    base_sha, head_sha = state_digest(captured.model), state_digest(captured.head)
    if any((base_sha != metadata['state_sha256'], base_sha != summary['state_sha256'],
            head_sha != metadata['head_state_sha256'], head_sha != summary['head_state_sha256'])):
        raise ValueError('Loaded base or head state differs from evidence')
    if sum(p.numel() for p in captured.model.parameters()) != metadata['base_parameter_count'] or sum(p.numel() for p in captured.head.parameters()) != metadata['head_parameter_count']:
        raise ValueError('Parameter inventory differs from evidence')
    for module in (captured.model, captured.head, plain.model, plain.head):
        if module.training or any(p.requires_grad for p in module.parameters()):
            raise ValueError('Base and command head must be frozen and in evaluation mode')
    files = [evidence / name for name in ['metadata.json', 'manifest.jsonl', 'completion.json'] + [entry['file'] for entry in entries]]
    files += [episode / name for name in ('summary.json', 'source.zip', 'case.json', 'track.json', 'realization.json', 'scene.usda', 'appearance.json', 'export.json', 'frames.json', 'telemetry.json')]
    exported = json.loads((episode / 'export.json').read_text())
    files += [episode / 'assets' / name for name in exported['dependencies']]
    before = {str(path): sha256_file(path) for path in files}
    for name in ('source_archive', 'realization', 'scene', 'appearance', 'export'):
        path = episode / {'source_archive': 'source.zip', 'realization': 'realization.json', 'scene': 'scene.usda', 'appearance': 'appearance.json', 'export': 'export.json'}[name]
        if before[str(path)] != metadata[name + '_sha256']:
            raise ValueError(f'{name} hash differs from evidence')
    for name, digest in exported['sha256'].items():
        if before[str(episode / 'assets' / name)] != digest:
            raise ValueError(f'Scene asset differs: {name}')
    stats_saved, stats_hook, stats_command = {}, {}, {}
    phases = set()
    count = 0
    with torch.inference_mode():
        for _, record, _ in read_evidence(evidence):
            if record['frame'] != count or record['state']['t'] != count / 20:
                raise ValueError('Evidence frame numbering or timestamp is inconsistent')
            if not np.array_equal(record['nominal_detection_calibration'], env.T_ego_cam()):
                raise ValueError('Recorded nominal calibration changed')
            if record['temporal'] != {'older_image_frame': max(0, count - 4), 'current_image_frame': count}:
                raise ValueError('Temporal history indexes differ')
            phases.add(count % 4)
            obs = observation(record, env)
            with capture_policy_raw(captured.model) as raw, ImageHeadCapture(captured.model) as head:
                steer, accel, info = captured(obs, env)
            if head.output is None or 'output' not in raw:
                raise ValueError('Diagnostic hooks did not fire')
            got = {'action': [steer, accel], 'prediction': info, 'image_head': head.output, 'policy_raw': raw['output']}
            compare({key: record[key] for key in got}, got, stats_saved, 'saved')
            ps, pa, pi = plain(obs, env)
            compare({'action': [steer, accel], 'prediction': info}, {'action': [ps, pa], 'prediction': pi}, stats_hook, 'hook')
            ctrl = record['prediction']['controller']
            compare(KIND, ctrl['kind'], stats_command, 'command.kind')
            compare(np.asarray([obs.state.vx, obs.state.vy, obs.state.r], dtype=np.float32), ctrl['proprio'], stats_command, 'command.proprio')
            latent = torch.as_tensor(np.asarray(ctrl['latent']), dtype=torch.float32, device=device).reshape(1, 256)
            proprio = torch.as_tensor(np.asarray(ctrl['proprio']), dtype=torch.float32, device=device).reshape(1, 3)
            independent_raw = captured.head(latent, proprio).detach().cpu().numpy()[0]
            compare(independent_raw, ctrl['raw'], stats_command, 'command.head_from_saved_latent')
            bounded = [float(np.clip(ctrl['raw'][0], -env.vp.max_steer, env.vp.max_steer)), float(np.clip(ctrl['raw'][1], -env.vp.max_brake, env.vp.max_accel))]
            command_accel, override = mission_brake_with_override(bounded[1], obs, env)
            applied = [bounded[0], float(np.clip(command_accel, -env.vp.max_brake, env.vp.max_accel))]
            compare(bounded, ctrl['bounded'], stats_command, 'command.bounded')
            compare(applied, ctrl['applied'], stats_command, 'command.applied')
            compare(override, ctrl['mission_override'], stats_command, 'command.mission_override')
            compare(ctrl['applied'], record['action'], stats_command, 'command.recorded_action')
            count += 1
    unchanged = (before == {str(path): sha256_file(path) for path in files}
                 and checkpoint_sha == sha256_file(checkpoint)
                 and all(base_sha == state_digest(policy.model) and head_sha == state_digest(policy.head)
                         and not policy.model.training and not policy.head.training
                         and not any(p.requires_grad for p in policy.model.parameters())
                         and not any(p.requires_grad for p in policy.head.parameters())
                         for policy in (captured, plain))
                 and all(sha256_file(name) == digest for name, digest in {**graphs, **sources}.items()))
    passed = count == count_expected and phases == {0, 1, 2, 3} and unchanged and not any(map(mismatches, (stats_saved, stats_hook, stats_command)))
    report = {'passed': passed, 'frames': count, 'expected_frames': count_expected, 'temporal_phases': sorted(phases),
              'case': metadata['case'], 'package_kind': KIND, 'schema_version': 5, 'torch_num_threads': torch.get_num_threads(), 'command_contract': metadata['command_contract'], 'checkpoint_sha256': checkpoint_sha,
              'base_state_sha256': base_sha, 'head_state_sha256': head_sha, 'graph_sha256': graphs, 'source_sha256': sources,
              'evidence_and_artifacts_sha256': before, 'frozen_unchanged': unchanged, 'device': device, 'rtol': RTOL, 'atol': ATOL,
              'saved_vs_replay': stats_saved, 'capture_vs_plain': stats_hook, 'independent_command_checks': stats_command,
              'verification_limits': ['Saved inputs only; physical simulation and scoring are not rerun.', 'Cross-device numerical differences are not exempted from the original 1e-5 tolerance.']}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-frames', default='120', help='120 for a complete pilot, or all for a sealed episode')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    report = verify(args.episode, args.checkpoint, args.out, args.max_frames, args.device)
    print(json.dumps({'passed': report['passed'], 'frames': report['frames'], 'output': str(args.out)}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
