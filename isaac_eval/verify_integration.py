# ruff: noqa: E402, I001
# Isaac extensions must be imported after SimulationApp initialization.
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scene", choices=["plain", "paddock"], default="plain")
parser.add_argument("--out", type=Path, default=Path("/work/isaac_eval/runs/integration_vehicle_v3"))
args = parser.parse_args()
from isaacsim import SimulationApp

app = SimulationApp(
    {"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0, "disable_viewport_updates": True}
)
from rendering import configure_rendering, capture, PROFILE

configure_rendering()
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from pxr import PhysicsSchemaTools, UsdGeom, Gf
import omni.physx
from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage
from fsop.geometry.camera import CAMERA_MODELS, project_world, se3, CameraMount
from fsop.track.schema import ConeColor
from scene import (
    create_car,
    create_ground,
    create_cones,
    make_cameras,
    pose_cameras,
    material,
    bind,
    AckermannActuators,
)

out = args.out
out.mkdir(parents=True, exist_ok=True)
world = World(physics_dt=1 / 120, rendering_dt=1 / 20, backend="numpy", device="cpu")
world.get_physics_context().enable_ccd(True)
stage = get_current_stage()
create_ground(stage)
robot = create_car(stage, world)
create_cones(stage, np.array([[5.0, 200.0], [5.0, 300.0]]), [ConeColor.ORANGE, ConeColor.BLUE])
if args.scene == "paddock":
    from realistic_scene import apply_realism
    from appearance_guard import physical_contract, assert_appearance_only

    before = physical_contract(stage)
    apply_realism(
        stage,
        np.array([[5.0, 200.0], [5.0, 300.0]]),
        [ConeColor.ORANGE, ConeColor.BLUE],
        asset_root=Path("isaac_eval/assets"),
        seed=91000,
    )
    assert_appearance_only(before, stage)
markers = [
    (np.array([10.0, 0.0, 1.1]), (1.0, 0.0, 0.0)),
    (np.array([10.0, -0.8, 0.8]), (0.0, 1.0, 0.0)),
    (np.array([10.0, 0.8, 0.5]), (0.0, 0.0, 1.0)),
]
for i, (pos, color) in enumerate(markers):
    sphere = UsdGeom.Sphere.Define(stage, f"/World/Marker{i}")
    sphere.CreateRadiusAttr(0.045)
    sphere.AddTranslateOp().Set(Gf.Vec3d(*pos))
    bind(sphere.GetPrim(), material(stage, f"/World/Materials/Marker{i}", color))
world.reset()
print("DOFS", ["PhysX Vehicle wheel controllers"], flush=True)
act = AckermannActuators(robot)
for _ in range(120):
    act.apply(0.0, 0.0, 1 / 120)
    world.step(render=False)
cameras, chase = make_cameras()


def state():
    p, q = robot.get_world_pose()
    r = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
    return p, r


from isaacsim.core.prims import SingleRigidPrim

p, r = state()
pose_cameras(cameras, chase, p, r)
for _ in range(20):
    world.render()
from appearance_guard import evidence_sources

report = {
    "source_sha256": evidence_sources(),
    "render_profile": PROFILE,
    "scene": args.scene,
    "dofs": ["PhysX Vehicle wheel controllers"],
    "settled_position": p.tolist(),
    "camera": {},
    "motion": {},
    "contact_events": [],
}
for name, cam in cameras.items():
    rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
    Image.fromarray(rgb).save(out / f"{name}.png")
    uv, _, _ = project_world(
        np.array([x[0] for x in markers]), se3(r, p), CameraMount(z=1.1 - 0.36).T_ego_cam(), CAMERA_MODELS[name]
    )
    measured = []
    for i, (u, v) in enumerate(uv):
        x, y = int(round(u)), int(round(v))
        crop = rgb[max(0, y - 12) : y + 13, max(0, x - 12) : x + 13].astype(float)
        score = crop[:, :, i] - np.max(np.delete(crop, i, axis=2), axis=2)
        mask = score > max(30.0, 0.7 * score.max())
        ys, xs = np.where(mask)
        measured.append(
            [float(xs.mean() + max(0, x - 12)), float(ys.mean() + max(0, y - 12))] if len(xs) else [None, None]
        )
    errors = [
        float(np.linalg.norm(np.array(m) - target)) if m[0] is not None else None
        for m, target in zip(measured, uv, strict=True)
    ]
    report["camera"][name] = {
        "shape": list(rgb.shape),
        "predicted": uv.tolist(),
        "measured": measured,
        "errors_px": errors,
        "properties": list(cam.get_opencv_pinhole_properties()),
    }
print("CAMERA", report["camera"], flush=True)
# Synchronized capture must reflect the current pose without advancing physics.
# Use the bright marker core to exclude weak RTX reflections/temporal trails.
for _ in range(6):
    world.step(render=False)
p, r = state()
p2 = p + np.array([0.0, 0.3, 0.0])
pose_cameras(cameras, chase, p2, r)
capture_time = world.current_time
capture_pose = state()[0].copy()
capture(world)
report["capture_sim_dt"] = float(world.current_time - capture_time)
report["capture_position_change_m"] = float(np.linalg.norm(state()[0] - capture_pose))
freshness = {}
for name, cam in cameras.items():
    rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
    Image.fromarray(rgb).save(out / f"{name}_moved.png")
    uv, _, _ = project_world(
        np.array([x[0] for x in markers]), se3(r, p2), CameraMount(z=0.74).T_ego_cam(), CAMERA_MODELS[name]
    )
    errors = []
    for i, (u, v) in enumerate(uv):
        x, y = int(round(u)), int(round(v))
        crop = rgb[y - 10 : y + 11, x - 10 : x + 11].astype(float)
        score = crop[:, :, i] - np.max(np.delete(crop, i, axis=2), axis=2)
        ys, xs = np.where(score > max(30.0, 0.7 * score.max()))
        errors.append(float(np.linalg.norm([xs.mean() + x - 10 - u, ys.mean() + y - 10 - v])) if len(xs) else 999.0)
    freshness[name] = errors
report["one_control_tick_projection_errors_px"] = freshness

hits = set()


def contact(headers, data):
    for h in headers:
        paths = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.actor0, h.actor1)]
        if any(p.startswith("/World/Car/") for p in paths):
            for path in paths:
                if path.startswith("/World/Cones/") and path not in hits:
                    hits.add(path)
                    report["contact_events"].append({"t": float(world.current_time), "cone": path})


sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
for mode, steer, seconds in [
    ("straight", 0.0, 5.0),
    ("left", 0.15, 5.0),
    ("right", -0.15, 5.0),
    ("fast_straight", 0.0, 7.0),
    ("fast_left", 0.15, 7.0),
    ("fast_right", -0.15, 7.0),
    ("contact", 0.0, 5.0),
]:
    world.reset()
    act = AckermannActuators(robot)
    if mode == "contact":
        for i, xy in enumerate([(5.0, 0.0), (5.0, 3.0)]):
            cone = SingleRigidPrim(f"/World/Cones/c{i:04d}")
            cone.initialize()
            cone.set_world_pose(np.array([*xy, 0.02]))
    for _ in range(60):
        world.step(render=False)
    start = state()[0].copy()
    trajectory = []
    upright = []
    for k in range(int(seconds * 20)):
        act.apply(steer, 2.0 if k < (80 if mode.startswith("fast") else 40) else 0.0)
        for _ in range(6):
            world.step(render=False)
        p, r = state()
        upright.append(float(r[2, 2]))
        v = robot.get_linear_velocity()
        trajectory.append([k * 0.05, *p, float(np.arctan2(r[1, 0], r[0, 0])), *v])
    before = np.linalg.norm(robot.get_linear_velocity()[:2])
    for _ in range(40):
        act.apply(0.0, -5.0)
        for _ in range(6):
            world.step(render=False)
    p, r = state()
    report["motion"][mode] = {
        "start": start.tolist(),
        "end": p.tolist(),
        "yaw": float(np.arctan2(r[1, 0], r[0, 0])),
        "speed_before_brake": float(before),
        "speed_after_brake": float(np.linalg.norm(robot.get_linear_velocity()[:2])),
        "max_z": float(np.max(np.array(trajectory)[:, 3])),
        "min_up": min(upright),
    }
    np.save(out / f"{mode}.npy", trajectory)
    print("MOTION", mode, report["motion"][mode], flush=True)
report["contacts"] = sorted(hits)
report["pass"] = (
    all(e is not None and e < 2 for c in report["camera"].values() for e in c["errors_px"])
    and report["motion"]["straight"]["end"][0] > 5
    and abs(report["motion"]["straight"]["end"][1]) < 0.3
    and report["motion"]["left"]["yaw"] > 0.2
    and report["motion"]["right"]["yaw"] < -0.2
    and all(v["speed_after_brake"] < 0.15 for v in report["motion"].values())
    and hits == {"/World/Cones/c0000"}
    and all(
        0.34 < v["max_z"] < (0.45 if k.startswith("fast") else 0.39) and v["min_up"] > 0.95
        for k, v in report["motion"].items()
        if k != "contact"
    )
)
report["pass"] = (
    report["pass"]
    and abs(report["capture_sim_dt"]) < 1e-6
    and report["capture_position_change_m"] < 1e-5
    and all(e < 2 for errors in freshness.values() for e in errors)
)
(out / "verification.json").write_text(json.dumps(report, indent=2))
print("VERIFIED", report["pass"], flush=True)
app.close()
