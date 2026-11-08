"""Generate a deterministic, small simulator dataset for Spark throughput tests."""

import json
import time
from pathlib import Path

from fsop.data.sim_dataset import split_episodes
from fsop.sim.batch_record import plan_dataset, record_dataset
from fsop.track.batch import generate_set
from fsop.track.generate import TrackGenCfg


def main():
  start = time.perf_counter()
  tracks = 'data/tracks/spark_test'
  dataset = 'data/sim/spark_test'
  if Path(dataset, 'index.json').exists():
    print(f'Reusing {dataset}')
    return
  if not Path(tracks, 'index.json').exists():
    generate_set(tracks, 40, TrackGenCfg(), seed0=1000, workers=6)
  specs = plan_dataset(tracks, n_random=1, n_standing=0, n_skidpad=2, n_accel=2, max_seconds=20, seed=42)
  rows = record_dataset(specs, dataset, workers=6)
  result = {'seconds': time.perf_counter() - start, 'episodes': len(rows),
            'frames': sum(r['frames'] for r in rows),
            'splits': {s: split_episodes(dataset, s) for s in ('train', 'val', 'test')}}
  Path('runs/spark_benchmark').mkdir(parents=True, exist_ok=True)
  Path('runs/spark_benchmark/data.json').write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
