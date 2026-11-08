"""Reproduce the V2 dataset scale locally, with 85% realistic venue scenes."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from fsop.config import load_config
from fsop.data.sim_dataset import split_episodes
from fsop.sim.batch_record import plan_dataset, record_dataset
from fsop.track.batch import generate_set
from fsop.track.generate import TrackGenCfg


def main():
    started = time.time()
    out = Path("runs/spark_full")
    out.mkdir(parents=True, exist_ok=True)
    dataset = "data/sim/spark_full_venue"
    specs = []
    for i, (name, preset, n, seed) in enumerate([("td", "trackdrive", 480, 20000), ("tight", "tight", 160, 30000)]):
        tracks = f"data/tracks/{name}_spark_full_venue"
        if not Path(tracks, "index.json").exists():
            generate_set(tracks, n, load_config(TrackGenCfg, f"configs/track/{preset}.yaml"), seed0=seed, workers=8)
        specs += plan_dataset(
            tracks,
            n_random=2,
            n_skidpad=40 if i == 0 else 0,
            n_accel=20 if i == 0 else 0,
            max_seconds=40,
            seed=100000 * i,
            offset_max=1.0,
            venue_prob=0.85,
        )
    for i, spec in enumerate(specs):
        spec.episode_id = f"ep{i:05d}_" + spec.episode_id.split("_", 1)[1]
    (out / "episode_specs.json").write_text(json.dumps([asdict(s) for s in specs], indent=2))
    print(f"Recording {len(specs)} episodes", flush=True)
    rows = record_dataset(specs, dataset, workers=16, max_tasks_per_child=20)
    result = {
        "started_at": started,
        "finished_at": time.time(),
        "seconds": time.time() - started,
        "episodes": len(rows),
        "frames": sum(r["frames"] for r in rows),
        "venue_probability": 0.85,
        "splits": {s: split_episodes(dataset, s) for s in ("train", "val", "test")},
    }
    (out / "data.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "splits"}), flush=True)


if __name__ == "__main__":
    main()
