import json
import resource
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path('runs/spark_skidpad')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(32768, hard)), hard))
    python = sys.executable
    phases = [
        ('recovery_record', [python, 'scripts/collect_skidpad_recovery.py', '--checkpoint',
         'runs/spark_skidpad_vision/ckpt/best.pt', '--output', 'data/sim/skidpad_recovery_r1', '--count', '32']),
        ('recovery_prepare', [python, 'scripts/prepare_skidpad_recovery_training.py']),
        ('recovery_cache', [python, '-c', "from fsop.data.feature_cache import build_cache; "
         "build_cache('data/sim/spark_skidpad_recovery', ckpt='runs/spark_skidpad_vision/ckpt/best.pt', "
         "batch=64, prefetch=4)"]),
        ('spark_skidpad_recovery', [python, 'scripts/benchmark_training.py', '--cfg',
         'configs/train/spark_skidpad_recovery.yaml', '--label', 'spark_skidpad_recovery', '--fit',
         '--exit-after-save', '--out', str(root)]),
        ('recovery_validation', [python, 'scripts/probe_skidpad.py', 'drive', '--checkpoint',
         'runs/spark_skidpad_recovery/ckpt/best.pt', '--output', str(root / 'recovery_validation'),
         '--seeds', '61000', '61001', '61002']),
    ]
    for name, command in phases:
        marker = root / f'{name}.done.json'
        if marker.exists():
            continue
        started = time.time()
        (root / 'status.json').write_text(json.dumps({'phase': name, 'started_at': started}))
        with (root / f'{name}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        marker.write_text(json.dumps({'seconds': time.time() - started}))
    (root / 'status.json').write_text(json.dumps({'phase': 'recovery_validation_complete'}))


if __name__ == '__main__':
    main()
