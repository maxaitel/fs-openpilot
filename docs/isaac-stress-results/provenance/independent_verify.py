import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/work')
RUN = ROOT / 'isaac_eval/runs/stress_v1'
CASES = ('skidpad-72000-light-recorded', 'skidpad-72000-heavy-recorded', 'retention-trackdrive_020000-light-recorded', 'retention-trackdrive_020000-heavy-recorded')
VARIANTS = ('matched', 'visual_stress_v1')
PAIR_KEYS = ('case', 'initial_physical_state', 'checkpoint_sha256', 'state_sha256', 'manifest_sha256', 'graph_sha256', 'source_sha256', 'track', 'initial_cones_xyz', 'cone_colors', 'expert_params', 'vehicle_params', 'label_dt', 'nominal_detection_calibration', 'intrinsics')
EXPECTED_PLAN = 'e35dce2437e04f6e67debbdc768e02b07e81c3e6732b60db1cb194ead9c2a8de'

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(p):
    return json.loads(p.read_text())

def check(ok, message):
    if not ok:
        raise AssertionError(message)

plan = read(RUN / 'experiment.json')
check(plan['plan_sha256'] == EXPECTED_PLAN, 'plan identity')
from isaac_eval.stress_experiment import validate_plan, read_receipt
validate_plan(plan, read_receipt(plan['manifest']))
expected = {(v, c) for c in CASES for v in VARIANTS}
actual = {(v, p.name) for v in VARIANTS for p in (RUN / v).iterdir() if p.is_dir()}
check(actual == expected, f'episode membership {actual ^ expected}')
results = {'plan_sha256': EXPECTED_PLAN, 'plan_validation_passed': True, 'exact_episode_membership': True, 'episodes': [], 'pairs': []}
for case in CASES:
    pair = []
    for variant in VARIANTS:
        d = RUN / variant / case
        e = d / 'evidence'
        receipt, summary, metadata = (read(e / 'completion.json'), read(d / 'summary.json'), read(e / 'metadata.json'))
        audit, video = read(d / 'artifact_audit.json'), read(d / 'video_audit.json')
        entries = [json.loads(line) for line in (e / 'manifest.jsonl').read_text().splitlines()]
        check((e / receipt['summary_path']).resolve() == (d / 'summary.json').resolve(), f'{d} summary path')
        for p, key in ((d / 'summary.json', 'summary_sha256'), (e / 'metadata.json', 'metadata_sha256'), (e / 'manifest.jsonl', 'manifest_sha256')):
            check(digest(p) == receipt[key], f'{p} completion hash')
        check(receipt['version'] == 1 and len(entries) == receipt['frame_count'], f'{d} receipt count')
        check({p.name for p in e.glob('frame_*.npz')} == {x['file'] for x in entries}, f'{d} raw membership')
        for i, entry in enumerate(entries):
            check(entry['frame'] == i and entry['file'] == f'frame_{i:06d}.npz', f'{d} sequence {i}')
            check(entry['metadata_sha256'] == receipt['metadata_sha256'], f'{d} raw metadata {i}')
            check(digest(e / entry['file']) == entry['sha256'], f'{d} raw hash {i}')
        expected_sources = {**plan['source_sha256'], 'scripts/audit_isaac_episode.py': plan['tool_sha256']['scripts/audit_isaac_episode.py']}
        check(audit['passed'] is True and audit['case_id'] == case and audit['source_sha256'] == expected_sources, f'{d} audit identity')
        for p, key in ((e/'completion.json','completion_sha256'), (d/'summary.json','summary_sha256'), (e/'metadata.json','metadata_sha256'), (e/'manifest.jsonl','manifest_sha256'), (d/'frames.json','frames_sha256'), (d/'telemetry.json','telemetry_sha256'), (d/'realization.json','realization_sha256')):
            check(digest(p) == audit[key], f'{p} audit hash')
        check(summary['case']['case_id'] == case and summary['variant'] == variant and summary['experiment_plan_sha256'] == EXPECTED_PLAN, f'{d} summary identity')
        check(metadata['source_sha256'] == plan['source_sha256'] and metadata['graph_sha256'] == plan['graph_sha256'] and metadata['checkpoint_sha256'] == plan['checkpoint_sha256'], f'{d} runtime identity')
        count = len(entries)
        check(count == len(read(d/'frames.json')) == summary['recorded_frames'] == audit['frame_count'] == video['inference_frames'], f'{d} frame counts')
        check(video['case_id'] == case and video['terminal_frames'] == 40 and video['completion_sha256'] == digest(e/'completion.json') and video['artifact_audit_sha256'] == digest(d/'artifact_audit.json') and video['producer_sha256'] == plan['tool_sha256']['scripts/process_isaac_parity.py'], f'{d} producer identity')
        video_results = {}
        for name in ('debug', 'research'):
            p = d / f'{name}.mp4'
            recorded = video['videos'][name]
            sha = digest(p)
            check(sha == recorded['sha256'], f'{p} hash')
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=nb_read_frames,avg_frame_rate,width,height', '-of', 'json', str(p)]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            stream = json.loads(proc.stdout)['streams'][0]
            check(not proc.stderr and int(stream['nb_read_frames']) == count + 40 and stream['avg_frame_rate'] == '20/1' and (stream['width'], stream['height']) == (1280,768), f'{p} decoded frames/cadence/dimensions')
            check(recorded['decoded_frames'] == count + 40 and recorded['fps'] == 20, f'{p} receipt')
            video_results[name] = {'sha256': sha, 'decoded_frames': int(stream['nb_read_frames']), 'fps': stream['avg_frame_rate'], 'width': stream['width'], 'height': stream['height']}
        rrd = d / 'debug.rrd'
        rrd_hash = digest(rrd)
        check(rrd_hash == video['rrd_sha256'], f'{rrd} hash')
        proc = subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'verify', '--check-footers=true', str(rrd)], capture_output=True, text=True)
        check(proc.returncode == 0, f'{rrd} verify: {proc.stdout} {proc.stderr}')
        results['episodes'].append({'case': case, 'variant': variant, 'frames': count, 'raw_chunks_verified': len(entries), 'result': summary['result'], 'pass': summary['pass'], 'physical_contacts': summary['physical_cone_hits'], 'oc': summary['oc'], 'videos': video_results, 'rrd_sha256': rrd_hash, 'rrd_verify_stdout': proc.stdout.strip(), 'rrd_verify_stderr': proc.stderr.strip()})
        pair.append((summary, metadata, audit))
        print(f'VERIFIED {variant} {case} frames={count}', file=sys.stderr, flush=True)
    left, right = pair
    check(all(left[1][k] == right[1][k] for k in PAIR_KEYS), f'{case} pair metadata')
    check(left[0]['physics_sha256'] == right[0]['physics_sha256'] and left[2]['realization_sha256'] == right[2]['realization_sha256'], f'{case} pair realization')
    results['pairs'].append({'case': case, 'invariant_keys': list(PAIR_KEYS), 'physics_sha256': left[0]['physics_sha256'], 'realization_sha256': left[2]['realization_sha256']})
print(json.dumps(results, indent=2))
