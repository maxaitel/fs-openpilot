# ruff: noqa: E402, I001
# Isaac extensions must be imported after SimulationApp initialization.
import json
from pathlib import Path
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})
import numpy as np
from pxr import UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from PIL import Image

world = World(physics_dt=1 / 120, rendering_dt=1 / 20, stage_units_in_meters=1, backend="numpy", device="cpu")
world.scene.add_default_ground_plane()
asset = get_assets_root_path() + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
print("ASSET", asset, flush=True)
add_reference_to_stage(asset, "/World/Car")
robot = world.scene.add(Robot("/World/Car", name="car", position=np.array([0.0, 0.0, 0.1])))
stage = get_current_stage()
light = __import__("pxr", fromlist=["UsdLux"]).UsdLux.DomeLight.Define(stage, "/World/Sky")
light.CreateIntensityAttr(1000.0)
world.reset()
for i in range(120):
    world.step(render=i % 6 == 0)
rows = []
for p in stage.Traverse():
    if not str(p.GetPath()).startswith("/World/Car"):
        continue
    if p.IsA(UsdPhysics.Joint) or p.HasAPI(UsdPhysics.RigidBodyAPI):
        rows.append(
            {
                "path": str(p.GetPath()),
                "type": p.GetTypeName(),
                "attrs": {
                    a.GetName(): str(a.Get())
                    for a in p.GetAttributes()
                    if a.GetName().startswith(("physics:", "physxRigidBody:"))
                },
            }
        )
bbox = (
    UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    .ComputeWorldBound(stage.GetPrimAtPath("/World/Car"))
    .ComputeAlignedBox()
)
report = {
    "asset": asset,
    "position": robot.get_world_pose()[0].tolist(),
    "orientation": robot.get_world_pose()[1].tolist(),
    "dof_names": robot.dof_names,
    "joint_positions": robot.get_joint_positions().tolist(),
    "bbox_min": list(bbox.GetMin()),
    "bbox_max": list(bbox.GetMax()),
    "physics": rows,
}
Path("/work/isaac_eval/runs/asset.json").write_text(json.dumps(report, indent=2))
print("REPORT", json.dumps({k: v for k, v in report.items() if k != "physics"}), flush=True)
camera = Camera("/World/InspectCamera", position=np.array([5.0, -6.0, 4.0]), resolution=(960, 540))
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats

eye = np.array([5.0, -6.0, 4.0])
f = np.array([0.0, 0.0, 0.5]) - eye
f /= np.linalg.norm(f)
r = np.cross(f, [0.0, 0.0, 1.0])
r /= np.linalg.norm(r)
u = np.cross(r, f)
camera.set_world_pose(eye, rot_matrices_to_quats(np.column_stack([r, u, -f])), camera_axes="usd")
camera.initialize()
for _ in range(20):
    world.render()
Image.fromarray(camera.get_rgba()[:, :, :3].astype(np.uint8)).save("/work/isaac_eval/runs/asset.png")
app.close()
