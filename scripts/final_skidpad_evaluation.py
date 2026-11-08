import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def command(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        subprocess.run([sys.executable, *args], stdout=stream, stderr=subprocess.STDOUT, check=True)


def merge(paths, output, checkpoint, count):
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        assert data['checkpoint'] == checkpoint
        rows.extend(data['runs'])
    assert len(rows) == count
    identities = [(row.get('track', 'skidpad'), row['seed'], row['dr']) for row in rows]
    assert len(set(identities)) == count
    output.write_text(json.dumps({'checkpoint': checkpoint, 'runs': rows}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    root = Path('runs/spark_skidpad')
    checkpoint = args.checkpoint
    manifest = {'checkpoint': checkpoint, 'sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                'skidpad_seeds': list(range(71000, 71005)), 'levels': ['light', 'heavy'],
                'command_ablation_seed': 71000, 'video_seed': 71000}
    path = root / 'final-protocol.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Final evaluation is already bound to another checkpoint')
    path.write_text(json.dumps(manifest, indent=2))
    stages = []

    def stage(name, work):
        started = time.time()
        (root / 'status.json').write_text(json.dumps({'phase': name, 'started_at': started}))
        work()
        stages.append({'phase': name, 'seconds': time.time() - started})
        (root / 'final-evaluation-timing.json').write_text(json.dumps(stages, indent=2))

    def skidpad():
        output = root / 'final_test'
        jobs = []
        for seed in manifest['skidpad_seeds']:
            folder = output / str(seed)
            jobs.append((['scripts/probe_skidpad.py', 'drive', '--checkpoint', checkpoint,
                          '--output', str(folder), '--seeds', str(seed)], folder.with_suffix('.log')))
        with ThreadPoolExecutor(5) as pool:
            list(pool.map(lambda job: command(*job), jobs))
        merge([output / str(seed) / 'driving.json' for seed in manifest['skidpad_seeds']],
              output / 'driving.json', checkpoint, 10)

    def retention():
        output = root / 'retention'
        original = json.loads(Path('runs/spark_full/evaluation/protocol.json').read_text())
        tracks = [track for track in original['tracks'] if track.startswith('trackdrive')]
        jobs = []
        for i in range(4):
            folder = output / str(i)
            jobs.append((['scripts/evaluate_skidpad_retention.py', '--checkpoint', checkpoint,
                          '--output', str(folder), '--tracks', *tracks[i::4]], folder.with_suffix('.log')))
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda job: command(*job), jobs))
        merge([output / str(i) / 'driving.json' for i in range(4)], output / 'driving.json', checkpoint, 16)

    def ablations():
        jobs = []
        for mode in ('zero', 'swapped'):
            folder = root / f'final_{mode}'
            jobs.append((['scripts/probe_skidpad.py', 'drive', '--checkpoint', checkpoint, '--output', str(folder),
                          '--command-mode', mode, '--seeds', '71000'], folder.with_suffix('.log')))
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda job: command(*job), jobs))

    stage('final_skidpad_test', skidpad)
    stage('trackdrive_retention', retention)
    stage('live_command_ablations', ablations)
    for name, model in [('baseline_commands_new_data', 'runs/spark_full_refresh/ckpt/best.pt'),
                        ('final_commands', checkpoint)]:
        stage(name, lambda name=name, model=model: command(
            ['scripts/probe_skidpad.py', 'commands', '--checkpoint', model, '--dataset',
             'data/sim/spark_skidpad_intent', '--output', str(root / name)], root / f'{name}.log'))
    stage('final_video', lambda: command(['scripts/render_skidpad.py', '--checkpoint', checkpoint],
                                        root / 'video-render.log'))
    assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == manifest['sha256']
    (root / 'status.json').write_text(json.dumps({'phase': 'final_evaluation_complete'}))


if __name__ == '__main__':
    main()
