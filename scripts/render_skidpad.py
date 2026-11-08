import argparse
import hashlib
import json
from pathlib import Path

import torch

from fsop.track.fixed_layouts import skidpad
from fsop.viz.video import render_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='runs/spark_skidpad/video')
    parser.add_argument('--seed', type=int, default=71000)
    parser.add_argument('--dr', choices=['light', 'heavy'], default='light')
    args = parser.parse_args()
    torch.set_num_threads(8)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = render_run(args.checkpoint, skidpad(), output / 'skidpad-raw.mp4',
                        seconds=120, seed=args.seed, dr=args.dr, venue_prob=1.0)
    result.update(checkpoint=args.checkpoint, seed=args.seed, dr=args.dr, venue_probability=1.0,
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    (output / 'video.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
