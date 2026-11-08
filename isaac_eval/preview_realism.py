# ruff: noqa: E402, I001

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--scene", choices=["plain", "paddock"], required=True)
p.add_argument("--discipline", choices=["trackdrive", "skidpad"], required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
from isaacsim import SimulationApp

app = SimulationApp(
    {"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0, "disable_viewport_updates": True}
)
from rendering import configure_rendering, capture, PROFILE

configure_rendering()
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage
from fsop.track.fixed_layouts import skidpad
from fsop.track.generate import generate_track, TrackGenCfg
from fsop.sim.env import FSEnv, SimCfg
from scene import create_ground, create_car, create_cones, make_cameras, pose_cameras
from appearance_guard import physical_contract, assert_appearance_only, evidence_sources

track = skidpad() if a.discipline == "skidpad" else generate_track(93000, TrackGenCfg(length_min=220, length_max=260))
env = FSEnv(track, SimCfg(render=False, dr="off"), seed=91000)
world = World(physics_dt=1 / 120, rendering_dt=1 / 20, backend="numpy", device="cpu")
stage = get_current_stage()
create_ground(stage, 91000)
robot = create_car(stage, world)
create_cones(stage, env.cones_xy, env.cone_colors)
before = physical_contract(stage)
record = {"scene": a.scene, "discipline": a.discipline, "source_sha256": evidence_sources(), "render_profile": PROFILE}
if a.scene == "paddock":
    from realistic_scene import apply_realism

    record.update(apply_realism(stage, env.cones_xy, env.cone_colors, asset_root=Path("isaac_eval/assets"), seed=91000))
record["physics_sha256"] = assert_appearance_only(before, stage)
world.reset()
cameras, chase = make_cameras()
a.out.mkdir(parents=True, exist_ok=False)
x, y, yaw = track.start_pose
poses = [("start", x, y, yaw)]
xy = np.asarray(track.centerline)
for name, fraction in [("turn", 0.25), ("far", 0.60)]:
    i = int(len(xy) * fraction)
    d = xy[min(i + 1, len(xy) - 1)] - xy[i]
    poses.append((name, float(xy[i, 0]), float(xy[i, 1]), float(np.arctan2(d[1], d[0]))))
for name, x, y, yaw in poses:
    r = Rotation.from_euler("z", yaw).as_matrix()
    q = Rotation.from_euler("z", yaw).as_quat()
    position = np.array([x, y, 0.36])
    robot.set_world_pose(position, np.r_[q[3], q[:3]])
    robot.set_linear_velocity(np.zeros(3))
    robot.set_angular_velocity(np.zeros(3))
    world.step(render=False)
    pose_cameras(cameras, chase, position, r)
    for _ in range(12):
        world.render()
    capture(world)
    for key, cam in {**cameras, "chase": chase}.items():
        rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
        assert rgb.max() - rgb.min() > 20, "Blank camera"
        Image.fromarray(rgb).save(a.out / f"{name}_{key}.png")
record["poses"] = poses
(a.out / "verification.json").write_text(json.dumps(record, indent=2))
from export_scene import export_scene

record["export"] = export_scene(stage, a.out / "scene.usda", Path("isaac_eval/assets"))
(a.out / "verification.json").write_text(json.dumps(record, indent=2))
print("PREVIEW_VERIFIED", record["physics_sha256"], flush=True)
app.close()
