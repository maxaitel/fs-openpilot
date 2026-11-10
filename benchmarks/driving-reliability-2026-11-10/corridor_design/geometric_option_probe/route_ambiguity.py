"""Compare two saved TRAIN teacher crossing states without using vision or simulation."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2] / "native_lateral_reference_executed" / "source_snapshot"
metadata = json.loads((root / "meta.json").read_text())
if metadata["id"] != "train-skidpad-82000-matched" or metadata["split"] != "train":
    raise ValueError("unexpected episode")
for name in ("states.npy", "controls.npy"):
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != metadata["arrays_sha256"][name]:
        raise ValueError(f"hash mismatch: {name}")
states = np.load(root / "states.npy", allow_pickle=False)
controls = np.load(root / "controls.npy", allow_pickle=False)
indices = (254, 449)
a, b = states[list(indices)]
if not np.allclose([a[8], b[8]], [12.70, 22.45], rtol=0, atol=1e-6):
    raise ValueError("unexpected state times")
result = {
    "episode": metadata["id"],
    "times_s": [float(a[8]), float(b[8])],
    "position_distance_m": float(np.linalg.norm(a[:2] - b[:2])),
    "yaw_difference_rad": float(np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2]))),
    "forward_speed_mps": [float(a[3]), float(b[3])],
    "teacher_steering_rad": controls[list(indices), 0].tolist(),
    "crossing_times_s": metadata["summary"]["route"]["crossings"],
    "limitation": "Teacher-state comparison only. This does not evaluate vision or control performance.",
}
print(json.dumps(result, indent=2))
