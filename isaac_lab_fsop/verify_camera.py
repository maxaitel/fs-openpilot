# ruff: noqa: E402, I001
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
from isaac_lab_fsop.runtime import launch

launcher = launch(render=True)
app = launcher.app
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.sensors.camera")
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from pxr import Gf, UsdGeom
from isaac_lab_fsop.runtime import create_simulation
from isaacsim.core.utils.stage import get_current_stage
from fsop.geometry.camera import CAMERA_MODELS, project_world, se2_to_se3
from isaac_eval.parity_cases import build_environment, get_case, sha256_file
from parity_geometry import pose_model_cameras
from parity_scene import dress_matched_scene
from rendering import capture, configure_rendering, PROFILE
from scene import create_ground, create_car, create_cones, make_cameras, material, bind

configure_rendering()
args.out.mkdir(parents=True, exist_ok=False)
world = create_simulation()
stage = get_current_stage()
env = build_environment(get_case("skidpad-72000-heavy-recorded"))
create_ground(stage, env.seed)
create_car(stage)
create_cones(stage, env.cones_xy, env.cone_colors)
dress_matched_scene(stage, env, Path("isaac_eval/assets"))
points = np.array([[10.0, 0.0, 0.7], [10.0, -0.8, 0.6], [10.0, 0.8, 0.5]])
for index, point in enumerate(points):
    sphere = UsdGeom.Sphere.Define(stage, f"/World/Marker{index}")
    sphere.CreateRadiusAttr(0.045)
    sphere.AddTranslateOp().Set(Gf.Vec3d(*point))
    bind(sphere.GetPrim(), material(stage, f"/World/Materials/Marker{index}", tuple(np.eye(3)[index])))
world.reset()
cameras, _ = make_cameras(with_chase=False)
for _ in range(20):
    world.render()
report = {
    "render_profile": PROFILE,
    "samples": [],
    "source_sha256": {
        name: sha256_file(name)
        for name in [
            "isaac_eval/verify_parity_camera.py",
            "isaac_eval/parity_geometry.py",
            "isaac_eval/parity_scene.py",
            "isaac_eval/scene.py",
            "isaac_eval/rendering.py",
        ]
    },
}
for level in ("light", "heavy"):
    env = build_environment(get_case(f"skidpad-72000-{level}-recorded"))
    for index, (p, angles) in enumerate(
        [
            ([0.0, 0.0, 0.36], [0.0, 0.0, 0.0]),
            ([0.0, 0.3, 0.40], [0.025, -0.02, 0.01]),
            ([0.0, -0.25, 0.34], [-0.03, 0.015, -0.015]),
        ]
    ):
        p = np.array(p)
        rotation = Rotation.from_euler("xyz", angles).as_matrix()
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        planar = se2_to_se3(p[0], p[1], yaw)
        _, projection = pose_model_cameras(cameras, p, rotation, env.T_ego_cam(), planar)
        before_time = world.current_time
        capture(world)
        sample = {"level": level, "pose": index, "capture_sim_dt": world.current_time - before_time, "cameras": {}}
        for name, camera in cameras.items():
            rgb = camera.get_rgba()[:, :, :3].astype(np.uint8)
            Image.fromarray(rgb).save(args.out / f"{level}_{index}_{name}.png")
            uv, _, visible = project_world(points, planar, projection, CAMERA_MODELS[name])
            measured, errors = [], []
            for channel, target in enumerate(uv):
                x, y = np.rint(target).astype(int)
                x0, y0 = max(0, x - 12), max(0, y - 12)
                crop = rgb[y0 : min(256, y + 13), x0 : min(512, x + 13)].astype(float)
                if not visible[channel] or not crop.size:
                    measured.append(None)
                    errors.append(999.0)
                    continue
                score = crop[:, :, channel] - np.max(np.delete(crop, channel, axis=2), axis=2)
                ys, xs = np.where(score > max(30.0, 0.7 * score.max()))
                center = np.array([xs.mean() + x0, ys.mean() + y0]) if len(xs) else None
                measured.append(center.tolist() if center is not None else None)
                errors.append(float(np.linalg.norm(center - target)) if center is not None else 999.0)
            sample["cameras"][name] = {"expected_px": uv.tolist(), "measured_px": measured, "errors_px": errors}
        report["samples"].append(sample)
report["pass"] = all(
    abs(s["capture_sim_dt"]) < 1e-6 and all(e < 2 for camera in s["cameras"].values() for e in camera["errors_px"])
    for s in report["samples"]
)
(args.out / "verification.json").write_text(json.dumps(report, indent=2))
print("VERIFIED", report["pass"], flush=True)
world._disable_app_control_on_stop_handle = True
app.close(wait_for_replicator=False)
