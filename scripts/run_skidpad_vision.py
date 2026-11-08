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
    deadline = time.monotonic() + 1800
    while not (root / 'validation.done.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Policy validation did not finish')
        time.sleep(3)
    phases = [
        ('spark_skidpad_vision', [sys.executable, 'scripts/benchmark_training.py', '--cfg',
         'configs/train/spark_skidpad_vision.yaml', '--label', 'spark_skidpad_vision', '--fit',
         '--exit-after-save', '--out', str(root)]),
        ('vision_validation', [sys.executable, 'scripts/probe_skidpad.py', 'drive', '--checkpoint',
         'runs/spark_skidpad_vision/ckpt/best.pt', '--output', str(root / 'vision_validation'),
         '--seeds', '61000', '61001', '61002']),
    ]
    for name, command in phases:
        started = time.time()
        (root / 'status.json').write_text(json.dumps({'phase': name, 'started_at': started}))
        with (root / f'{name}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        (root / f'{name}.done.json').write_text(json.dumps({'seconds': time.time() - started}))
    (root / 'status.json').write_text(json.dumps({'phase': 'vision_validation_complete'}))


if __name__ == '__main__':
    main()
