# ruff: noqa: E402, I001
import argparse
import copy
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case",
    required=True,
    choices=[
        "skidpad-72000-light-recorded",
        "skidpad-72000-heavy-recorded",
        "retention-trackdrive_020000-light-recorded",
        "retention-trackdrive_020000-heavy-recorded",
    ],
)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
from isaacsim import SimulationApp

app = SimulationApp(
    {"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0, "disable_viewport_updates": True}
)
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage
from fsop.sim.randomization import postprocess
from fsop.sim.vehicle import VehicleState
from isaac_eval.parity_cases import build_environment, get_case, realized_environment
from isaac_eval.parity_protocol import runtime_sources
from isaac_eval.stress_scene import apply_visual_stress, validate_stress_assets
from isaac_eval.stress_layout import build_layout
from appearance_guard import physical_contract, assert_appearance_only
from parity_geometry import pose_model_cameras
from parity_scene import dress_matched_scene
from rendering import configure_rendering, capture, PROFILE
from scene import create_ground, create_car, create_cones, make_cameras, pose_cameras

configure_rendering()
args.out.mkdir(parents=True, exist_ok=False)
asset_root = Path("isaac_eval/assets")
case = get_case(args.case)
env = build_environment(case)
realization = realized_environment(env)
world = World(physics_dt=1 / 120, rendering_dt=1 / 20, backend="numpy", device="cpu")
stage = get_current_stage()
create_ground(stage, case.seed)
robot = create_car(stage, world)
create_cones(stage, env.cones_xy, env.cone_colors)
before = physical_contract(stage)
appearance = dress_matched_scene(stage, env, asset_root)
physics = assert_appearance_only(before, stage)
_, people_manifest, assets = validate_stress_assets(asset_root)
world.reset()
cameras, overview = make_cameras()
xy = np.asarray(env.track.centerline)
poses = [
    {
        "name": "start",
        "x": float(env.track.start_pose[0]),
        "y": float(env.track.start_pose[1]),
        "yaw": float(env.track.start_pose[2]),
    }
]
for name, fraction in (("route_20", 0.20), ("route_50", 0.50), ("route_80", 0.80)):
    i = min(int(len(xy) * fraction), len(xy) - 2)
    d = xy[i + 1] - xy[i]
    poses.append({"name": name, "x": float(xy[i, 0]), "y": float(xy[i, 1]), "yaw": float(np.arctan2(d[1], d[0]))})
people_xy = np.array(
    [item["position"][:2] for item in build_layout(env)["objects"] if item["kind"].startswith("person_")]
)
candidates = []
for i in range(0, len(xy) - 1, 3):
    d = xy[i + 1] - xy[i]
    heading = float(np.arctan2(d[1], d[0]))
    offsets = people_xy - xy[i]
    distances = np.linalg.norm(offsets, axis=1)
    bearings = np.abs(
        np.arctan2(
            np.sin(np.arctan2(offsets[:, 1], offsets[:, 0]) - heading),
            np.cos(np.arctan2(offsets[:, 1], offsets[:, 0]) - heading),
        )
    )
    for j in np.flatnonzero((distances >= 5) & (distances <= 20) & (bearings < 0.35)):
        candidates.append((float(distances[j] + 10 * bearings[j]), i, heading))
if not candidates:
    raise ValueError("No route-aligned preview pose has a person ahead")
_, i, heading = min(candidates)
poses.append({"name": "people", "x": float(xy[i, 0]), "y": float(xy[i, 1]), "yaw": heading})
initial_rng = copy.deepcopy(env.rng.bit_generator.state)
records = {}
for variant in ("matched", "visual_stress_v1"):
    if variant == "visual_stress_v1":
        overlay_before = physical_contract(stage)
        stress = apply_visual_stress(stage, env, asset_root)
        stress_physics = assert_appearance_only(overlay_before, stage)
        records["matched"]["physics_sha256"] = hashlib.sha256(
            json.dumps(overlay_before, sort_keys=True).encode()
        ).hexdigest()
    else:
        stress = None
        stress_physics = physics
    folder = args.out / variant
    folder.mkdir()
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(initial_rng)
    for pose in poses:
        yaw = pose["yaw"]
        r = Rotation.from_euler("z", yaw).as_matrix()
        q = Rotation.from_euler("z", yaw).as_quat()
        position = np.array([pose["x"], pose["y"], 0.36])
        robot.set_world_pose(position, np.r_[q[3], q[:3]])
        env.state = VehicleState(pose["x"], pose["y"], yaw)
        world_camera, projection = pose_model_cameras(cameras, position, r, env.T_ego_cam(), env.T_world_ego())
        pose_cameras({}, overview, position, r)
        for _ in range(12):
            world.render()
        capture(world)
        for name in env.cfg.cameras:
            rgb = cameras[name].get_rgba()[:, :, :3].astype(np.uint8)
            if rgb.max() - rgb.min() <= 20:
                raise ValueError(f"Blank {name} camera at {pose['name']}")
            Image.fromarray(rgb).save(folder / f"{pose['name']}_{name}_raw.png")
            processed = postprocess(rgb, env.scene, rng, env.state.r)
            Image.fromarray(processed).save(folder / f"{pose['name']}_{name}_processed.png")
        Image.fromarray(overview.get_rgba()[:, :, :3].astype(np.uint8)).save(folder / f"{pose['name']}_overview.png")
        pose["world_camera"] = world_camera.tolist()
        pose["projection"] = projection.tolist()
    records[variant] = {
        "physics_sha256": stress_physics,
        "stress": stress,
        "images": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.png"))},
    }
record = {
    "schema": "visual_stress_preview_v1",
    "case_id": args.case,
    "realization_sha256": realization["realization_sha256"],
    "source_sha256": runtime_sources(),
    "asset_manifest_sha256": people_manifest,
    "asset_sha256": assets,
    "render_profile": PROFILE,
    "initial_physics_sha256": physics,
    "matched_appearance": appearance,
    "poses": poses,
    "variants": records,
    "model_inference": False,
}
(args.out / "verification.json").write_text(json.dumps(record, indent=2, allow_nan=False))
print("PREVIEW_VERIFIED", args.case, physics, flush=True)
app.close()
