"""Model-only latency in the Isaac Torch runtime, without rendering contention."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fsop.eval.closed_loop import ModelPolicy
from fsop.sim.env import FSEnv, SimCfg
from fsop.track.fixed_layouts import skidpad

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
path = Path("/frozen/model.pt")
checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
policy = ModelPolicy.from_checkpoint(path)
policy.model.requires_grad_(False)


def digest():
    h = hashlib.sha256()
    for name, value in policy.model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


before = digest()
env = FSEnv(skidpad(), SimCfg(render=False, dr="off", randomize_expert=False))
obs = env.observe(False)
obs.images = {
    c: np.array(Image.open(f"/work/isaac_eval/runs/integration_vehicle_v3/{c}.png")) for c in ["road", "wide"]
}
latency = []
with torch.inference_mode():
    for i in range(520):
        start = time.perf_counter()
        steer, accel, info = policy(obs, env)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        assert np.isfinite(info["plan"]).all() and np.isfinite([steer, accel]).all()
        if i >= 20:
            latency.append(seconds)
assert before == digest() and checkpoint_sha == hashlib.sha256(path.read_bytes()).hexdigest()
result = {
    "fixture": "fixed lossless camera calibration images, no renderer running",
    "samples": 500,
    "warmup": 20,
    "mean_ms": float(np.mean(latency) * 1000),
    "median_ms": float(np.median(latency) * 1000),
    "p95_ms": float(np.percentile(latency, 95) * 1000),
    "calls_per_second": float(1 / np.mean(latency)),
    "frozen_unchanged": True,
    "checkpoint_sha256": checkpoint_sha,
    "torch": torch.__version__,
    "times_seconds": latency,
}
Path("isaac_eval/runs/model_only_benchmark.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "times_seconds"}), flush=True)
