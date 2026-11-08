"""Evaluate a short training run on the smoke dataset's actual held-out tracks.

This is a separate protocol from the upstream v1 leaderboard. Track IDs come from the
same split used by SimDataset, so no training track enters the closed-loop evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fsop.data.sim_dataset import split_episodes
from fsop.eval.closed_loop import ModelPolicy, aggregate
from fsop.eval.open_loop import evaluate
from fsop.sim.drive import run_episode
from fsop.sim.env import CAMERAS, FSEnv, SimCfg
from fsop.sim.render.panda3d_backend import Panda3DRenderer
from fsop.track.fixed_layouts import acceleration, skidpad
from fsop.track.io import load_json


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--dataset', default='data/sim/spark_test', type=Path)
  p.add_argument('--ckpt', default='runs/spark_smoke_stage2/ckpt/best.pt', type=Path)
  p.add_argument('--out', default='runs/spark_benchmark/evaluation', type=Path)
  args = p.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  timings = {}
  for name, ckpt in [('untouched', None), ('trained', args.ckpt)]:
    start = time.perf_counter()
    result = evaluate(args.dataset, ckpt=ckpt)
    timings[f'{name}_open_loop_seconds'] = time.perf_counter() - start
    (args.out / f'{name}_open_loop.json').write_text(json.dumps(result, indent=2))
  test_ids = split_episodes(args.dataset, 'test')
  train_tracks = {json.loads((args.dataset / e / 'meta.json').read_text())['track_id']
                  for e in split_episodes(args.dataset, 'train')}
  tracks = {}
  for eid in test_ids:
    track = load_json(args.dataset / eid / 'track.json')
    assert track.id not in train_tracks
    track.laps = 1
    tracks[track.id] = track
  layouts = [*tracks.values(), skidpad(), acceleration()]
  renderer = Panda3DRenderer(CAMERAS)
  policy = ModelPolicy.from_checkpoint(args.ckpt)
  rows = []
  start = time.perf_counter()
  for track in layouts:
    cfg = SimCfg(dr='light', max_time=90)
    cfg.noise.enabled = False
    env = FSEnv(track, cfg, seed=0, renderer=renderer)
    policy.reset()
    policy_times = []

    def timed_policy(obs, env, policy_times=policy_times):
      if policy.device.type == 'cuda':
        torch.cuda.synchronize()
      t0 = time.perf_counter()
      action = policy(obs, env)
      if policy.device.type == 'cuda':
        torch.cuda.synchronize()
      policy_times.append((time.perf_counter() - t0) * 1000)
      return action

    episode_start = time.perf_counter()
    lat = []
    result = run_episode(env, timed_policy, viz=False, add_noise=False,
                         on_frame=lambda o, e, i, lat=lat: lat.append(abs(o.lat)))
    elapsed = time.perf_counter() - episode_start
    result.update(track=track.id, dr='light', seed=0, progress=float(env.s / env.mission.s_end),
                  rms_lat=float(np.sqrt(np.mean(np.square(lat)))) if lat else None,
                  policy_calls=len(policy_times), wall_seconds=elapsed,
                  simulation_steps_per_second=len(policy_times) / elapsed,
                  policy_median_ms=float(np.median(policy_times)),
                  policy_p95_ms=float(np.percentile(policy_times, 95)))
    rows.append(result)
    (args.out / f'{track.id}.json').write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, default=float), flush=True)
  timings['closed_loop_seconds'] = time.perf_counter() - start
  report = {'protocol': 'spark-smoke-v1: hash-held-out generated tracks, light DR, seed 0, 90s limit',
            'checkpoint': str(args.ckpt), 'test_episodes': test_ids,
            'runs': rows, 'aggregate': aggregate(rows), 'timings': timings}
  (args.out / 'closed_loop.json').write_text(json.dumps(report, indent=2, default=float))
  print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
  main()
