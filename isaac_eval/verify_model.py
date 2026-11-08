"""Read-only frozen-model smoke/parity fixture; works in both Torch runtimes."""

import argparse
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

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
checkpoint = Path("/frozen/model.pt")


def sha():
    return hashlib.sha256(checkpoint.read_bytes()).hexdigest()


before = sha()
start = time.perf_counter()
policy = ModelPolicy.from_checkpoint(checkpoint)
policy.model.requires_grad_(False)


def digest():
    h = hashlib.sha256()
    for name, value in policy.model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


state_before = digest()
env = FSEnv(skidpad(), SimCfg(render=False, dr="off", randomize_expert=False))
images = {c: np.array(Image.open(f"/work/isaac_eval/runs/integration_v4/{c}.png")) for c in ["road", "wide"]}
plans = []
actions = []
times = []
with torch.inference_mode():
    for i in range(6):
        obs = env.observe(False)
        obs.images = {c: np.roll(img, i, axis=1) for c, img in images.items()}
        obs.desire[:] = 0
        obs.desire[1 if i == 3 else 0] = 1
        t = time.perf_counter()
        steer, accel, info = policy(obs, env)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        plans.append(info["plan"])
        actions.append([steer, accel])
assert np.isfinite(plans).all() and np.isfinite(actions).all()
assert before == sha() and state_before == digest()
assert not any(p.requires_grad for p in policy.model.parameters())
np.savez(a.out, plans=plans, actions=actions)
Path(a.out + ".json").write_text(
    json.dumps(
        {
            "checkpoint_sha256": before,
            "state_sha256": state_before,
            "parameters": sum(p.numel() for p in policy.model.parameters()),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "inference_seconds": times,
            "wall_seconds": time.perf_counter() - start,
            "frozen": True,
            "finite": True,
        },
        indent=2,
    )
)
print("FROZEN_MODEL_OK", a.out, flush=True)
