"""Check every recorded frame and the split/scene manifest before final evaluation."""
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from fsop.data.sim_dataset import split_episodes

ROOT = Path('data/sim/spark_full_venue')
OUT = Path('runs/spark_full')


def audit(row):
  ep = ROOT / row['id']
  meta = json.loads((ep / 'meta.json').read_text())
  assert meta['frames'] == row['frames']
  frames = zarr.open_group(str(ep / 'frames.zarr'), mode='r')
  for cam in ('road', 'wide'):
    array = frames[cam][:]
    assert array.shape == (meta['frames'], 6, 128, 256), ep
    assert np.all(array.reshape(len(array), -1).any(axis=1)), (ep, cam, 'empty frame')
  assert np.isfinite(np.load(ep / 'plan.npy')).all(), ep
  return {'id': row['id'], 'track': meta['track_id'], 'frames': meta['frames'],
          'venue': meta['scene']['venue'], 'dr': meta['dr']['level'], 'discipline': meta['discipline']}


def main():
  while not (OUT / 'data.exit').exists():
    time.sleep(10)
  assert (OUT / 'data.exit').read_text().strip() == '0'
  start = time.time()
  index = json.loads((ROOT / 'index.json').read_text())['episodes']
  specs = json.loads((OUT / 'episode_specs.json').read_text())
  assert {r['id'] for r in index} == {r['episode_id'] for r in specs}
  with ThreadPoolExecutor(4) as ex:
    rows = list(ex.map(audit, index))
  by_id = {r['id']: r for r in rows}
  splits = {s: [by_id[i] for i in split_episodes(ROOT, s)] for s in ('train', 'val', 'test')}
  tracks = {s: {r['track'] for r in rs if r['discipline'] == 'trackdrive'} for s, rs in splits.items()}
  assert not (tracks['train'] & tracks['val'] or tracks['train'] & tracks['test'] or tracks['val'] & tracks['test'])
  result = {'audit_seconds': time.time() - start, 'episodes': len(rows),
            'frames': sum(r['frames'] for r in rows), 'venue_episodes': sum(r['venue'] for r in rows),
            'venue_frames': sum(r['frames'] for r in rows if r['venue']),
            'dr_episodes': dict(Counter(r['dr'] for r in rows)),
            'splits': {s: {'episodes': len(rs), 'frames': sum(r['frames'] for r in rs),
                           'trackdrive_tracks': len(tracks[s])} for s, rs in splits.items()},
            'all_frames_read': True, 'track_splits_disjoint': True}
  (OUT / 'data-audit.json').write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
