"""Held-out venue evaluation and a predetermined comparison video for the full Spark run."""

import hashlib
import json
import subprocess
import sys
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

OUT = Path("runs/spark_full/evaluation")
DATA = Path("data/sim/spark_full_venue")
CKPT = Path("runs/spark_full_refresh/ckpt/best.pt")


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT.parent / "status.json", {"phase": "evaluation", "started_at": time.time()})
    for name, ckpt in [("untouched", None), ("short", "runs/spark_smoke_stage2/ckpt/best.pt"), ("full", CKPT)]:
        p = OUT / f"{name}_open_loop.json"
        if not p.exists():
            start = time.time()
            result = evaluate(DATA, ckpt=ckpt)
            result["wall_seconds"] = time.time() - start
            dump(p, result)
    train_tracks = {json.loads((DATA / e / "meta.json").read_text())["track_id"] for e in split_episodes(DATA, "train")}
    candidates = {}
    for eid in split_episodes(DATA, "test"):
        track = load_json(DATA / eid / "track.json")
        if track.discipline == "trackdrive":
            assert track.id not in train_tracks
            candidates[track.id] = track
    # Fixed selection before inspecting any outcomes: four regular and four tight tracks.
    regular = [t for t in candidates.values() if int(t.id.rsplit("_", 1)[1]) < 30000][:4]
    tight = [t for t in candidates.values() if int(t.id.rsplit("_", 1)[1]) >= 30000][:4]
    tracks = regular + tight + [skidpad(), acceleration()]
    for track in tracks:
        if track.discipline == "trackdrive":
            track.laps = 1
    dump(
        OUT / "protocol.json",
        {
            "checkpoint": str(CKPT),
            "sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(),
            "tracks": [t.id for t in tracks],
            "dr": ["light", "heavy"],
            "seed": 0,
            "venue_probability": 1.0,
            "laps": 1,
            "max_seconds": 120,
            "selection": "first four regular and first four tight test tracks; fixed layouts",
        },
    )
    renderer = Panda3DRenderer(CAMERAS)
    for model_name, checkpoint in [("short", "runs/spark_smoke_stage2/ckpt/best.pt"), ("full", CKPT)]:
        policy = ModelPolicy.from_checkpoint(checkpoint)
        rows = []
        for track in tracks:
            for dr in ["light", "heavy"]:
                path = OUT / f"{model_name}_{track.id}_{dr}.json"
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                    continue
                cfg = SimCfg(dr=dr, max_time=120, venue_prob=1.0)
                cfg.noise.enabled = False
                env = FSEnv(track, cfg, seed=0, renderer=renderer)
                policy.reset()
                policy_times = []
                lat = []

                def timed_policy(obs, env, policy_times=policy_times, policy=policy):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    action = policy(obs, env)
                    torch.cuda.synchronize()
                    policy_times.append((time.perf_counter() - t) * 1000)
                    return action

                start = time.perf_counter()
                result = run_episode(
                    env,
                    timed_policy,
                    viz=False,
                    add_noise=False,
                    on_frame=lambda o, e, i, lat=lat: lat.append(abs(o.lat)),
                )
                elapsed = time.perf_counter() - start
                result.update(
                    model=model_name,
                    track=track.id,
                    dr=dr,
                    seed=0,
                    venue=True,
                    raw_progress=float(env.s / env.mission.s_end),
                    progress=float(np.clip(env.s / env.mission.s_end, 0.0, 1.0)),
                    rms_lat=float(np.sqrt(np.mean(np.square(lat)))),
                    wall_seconds=elapsed,
                    policy_median_ms=float(np.median(policy_times)),
                    policy_p95_ms=float(np.percentile(policy_times, 95)),
                    simulation_steps_per_second=len(policy_times) / elapsed,
                )
                dump(path, result)
                rows.append(result)
                print(
                    f"{model_name} {track.id} {dr}: {result['result']}, "
                    f"cones={result['doo']}, off-course={result['oc']}",
                    flush=True,
                )
        dump(
            OUT / ("closed_loop.json" if model_name == "full" else "short_closed_loop.json"),
            {"checkpoint": str(checkpoint), "runs": rows, "aggregate": aggregate(rows)},
        )
    # Same track and scene as the prior short-trained venue video, selected before training.
    track = load_json("data/tracks/spark_test/trackdrive_001013.json")
    assert track.id not in train_tracks
    if not (OUT / "video.json").exists():
        # A fresh process prevents the evaluation scene remaining in Panda3D's global render tree.
        subprocess.run([sys.executable, "scripts/render_spark_full.py"], check=True)
    dump(OUT.parent / "status.json", {"phase": "evaluation_complete", "finished_at": time.time()})


if __name__ == "__main__":
    main()
