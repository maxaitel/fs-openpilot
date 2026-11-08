import argparse
import hashlib
import json
from pathlib import Path

import torch

from fsop.data.sim_dataset import split_episodes
from fsop.eval.closed_loop import ModelPolicy
from fsop.sim.drive import run_episode
from fsop.sim.env import CAMERAS, FSEnv, SimCfg
from fsop.sim.render.panda3d_backend import Panda3DRenderer
from fsop.track.io import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tracks', nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'checkpoint': args.checkpoint, 'sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()}
    manifest = output / 'checkpoint.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Output directory belongs to a different checkpoint')
    manifest.write_text(json.dumps(identity, indent=2))
    protocol = json.loads(Path('runs/spark_full/evaluation/protocol.json').read_text())
    dataset = Path('data/sim/spark_full_venue')
    tracks = {}
    for eid in split_episodes(dataset, 'test'):
        track = load_json(dataset / eid / 'track.json')
        tracks[track.id] = track
    policy = ModelPolicy.from_checkpoint(args.checkpoint)
    renderer = Panda3DRenderer(CAMERAS)
    rows = []
    for tid in protocol['tracks']:
        if not tid.startswith('trackdrive'):
            continue
        if args.tracks is not None and tid not in args.tracks:
            continue
        track = tracks[tid]
        track.laps = 1
        for level in protocol['dr']:
            path = output / f'{tid}-{level}.json'
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            config = SimCfg(dr=level, max_time=120, venue_prob=1.0)
            config.noise.enabled = False
            env = FSEnv(track, config, seed=0, renderer=renderer)
            policy.reset()
            row = run_episode(env, policy, viz=False, add_noise=False)
            row.update(track=tid, dr=level, seed=0)
            path.write_text(json.dumps(row, indent=2))
            rows.append(row)
            print(json.dumps(row), flush=True)
    (output / 'driving.json').write_text(json.dumps({'checkpoint': args.checkpoint, 'runs': rows}, indent=2))


if __name__ == '__main__':
    main()
