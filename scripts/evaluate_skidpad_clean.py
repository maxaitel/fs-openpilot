import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def validate_runs(data, protocol):
    if data.get('checkpoint') != protocol['checkpoint']:
        raise ValueError('Evaluation result belongs to a different checkpoint')
    rows = data['runs']
    expected = {(seed, level) for seed in protocol['seeds'] for level in protocol['levels']}
    if len(rows) != len(expected) or {(row['seed'], row['dr']) for row in rows} != expected:
        raise ValueError('Evaluation result does not contain the exact requested cases')
    if any(row.get('controller') != protocol['controller']
           or row.get('command_mode') != protocol['command_mode'] for row in rows):
        raise ValueError('Evaluation result uses a different controller or command mode')
    return rows


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--levels', nargs='+', default=['light', 'heavy'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    identity = {'checkpoint': args.checkpoint, 'sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                'seeds': args.seeds, 'levels': args.levels, 'controller': 'pp', 'command_mode': 'recorded',
                'venue_probability': 1.0, 'max_seconds': 120, 'noise': False}
    manifest = root / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Evaluation directory belongs to a different protocol')
    if not manifest.exists() and ((root / 'driving.json').exists()
                                  or any(root.glob('*/driving.json')) or any(root.glob('*/protocol.json'))):
        raise ValueError('Evaluation results exist without a root protocol')
    if len(args.seeds) != len(set(args.seeds)) or len(args.levels) != len(set(args.levels)):
        raise ValueError('Duplicate cases in evaluation protocol')
    write_json(manifest, identity)

    def run(seed):
        folder = root / str(seed)
        result = folder / 'driving.json'
        completion = folder / 'protocol.json'
        protocol = {**identity, 'seeds': [seed]}
        if result.exists() != completion.exists():
            raise ValueError(f'Incomplete evaluation provenance for seed {seed}')
        if result.exists():
            result_bytes = result.read_bytes()
            expected = {**protocol, 'result_sha256': hashlib.sha256(result_bytes).hexdigest()}
            if json.loads(completion.read_text()) != expected:
                raise ValueError(f'Evaluation completion protocol mismatch for seed {seed}')
            data = json.loads(result_bytes)
        else:
            with (root / f'{seed}.log').open('w') as log:
                subprocess.run([sys.executable, 'scripts/probe_skidpad.py', 'drive', '--checkpoint', args.checkpoint,
                                '--output', str(folder), '--seeds', str(seed), '--levels', *args.levels,
                                '--controller', protocol['controller'], '--command-mode', protocol['command_mode']],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            result_bytes = result.read_bytes()
            data = json.loads(result_bytes)
            validate_runs(data, protocol)
            if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() != identity['sha256']:
                raise ValueError('Checkpoint changed during evaluation')
            write_json(completion, {**protocol, 'result_sha256': hashlib.sha256(result_bytes).hexdigest()})
        return validate_runs(data, protocol)

    with ThreadPoolExecutor(args.workers) as pool:
        rows = [row for group in pool.map(run, args.seeds) for row in group]
    validate_runs({'checkpoint': args.checkpoint, 'runs': rows}, identity)
    if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() != identity['sha256']:
        raise ValueError('Checkpoint changed during evaluation')
    summary = {'runs': len(rows), 'passed': sum(r['pass'] for r in rows),
               'clean': sum(r['pass'] and r['doo'] == 0 and r['oc'] == 0 for r in rows),
               'cones': sum(r['doo'] for r in rows), 'off_course': sum(r['oc'] for r in rows)}
    write_json(root / 'driving.json', {'protocol': identity, 'summary': summary, 'runs': rows})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
