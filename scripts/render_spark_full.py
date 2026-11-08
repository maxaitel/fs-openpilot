"""Render the preselected venue track in a fresh renderer process."""

import hashlib
import json
import time
from pathlib import Path

from fsop.track.io import load_json
from fsop.viz.video import render_run


def main():
    out = Path("runs/spark_full/evaluation")
    ckpt = Path("runs/spark_full_refresh/ckpt/best.pt")
    track = load_json("data/tracks/spark_test/trackdrive_001013.json")
    track.laps = 1
    start = time.perf_counter()
    result = render_run(
        ckpt, track, str(out / "spark-full-venue-raw.mp4"), seconds=120, dr="light", seed=0, venue_prob=1.0
    )
    result.update(
        wall_seconds=time.perf_counter() - start,
        checkpoint=str(ckpt),
        track=track.id,
        checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    )
    (out / "video.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
