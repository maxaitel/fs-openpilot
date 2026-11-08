# ruff: noqa: E402, I001
# Isaac extensions must be imported after SimulationApp initialization.
"""Frozen checkpoint -> existing controller -> PhysX Vehicle Ackermann car.

No training, no oracle steering, no fsop bicycle dynamics. Each process is one case.
"""

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--discipline", choices=["trackdrive", "skidpad"], required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
process_start = time.perf_counter()
from isaacsim import SimulationApp

app = SimulationApp(
    {"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0, "disable_viewport_updates": True}
)
import cv2
import carb

carb.settings.get_settings().set("/rtx-transient/dldenoiser/enabled", True)
carb.settings.get_settings().set("/rtx/post/motionblur/enabled", False)
import numpy as np
import psutil
import pynvml
import torch
from scipy.spatial.transform import Rotation
from pxr import PhysicsSchemaTools
import omni.physx
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage
from fsop.eval.closed_loop import ModelPolicy
from fsop.sim.env import FSEnv, SimCfg
from fsop.sim.vehicle import VehicleState
from fsop.track.fixed_layouts import skidpad
from fsop.track.generate import generate_track, TrackGenCfg
from scene import create_car, create_ground, create_cones, make_cameras, pose_cameras, AckermannActuators

out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
protocol = json.loads(Path("isaac_eval/protocol.json").read_text())
verification = json.loads(Path(protocol["integration_report"]).read_text())
assert verification["pass"], "Integration gate has not passed"
for filename, sha in protocol["source_sha256"].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == sha, filename
checkpoint = Path("/frozen/model.pt")


def sha():
    return hashlib.sha256(checkpoint.read_bytes()).hexdigest()


checkpoint_before = sha()
assert checkpoint_before == protocol["checkpoint_sha256"]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
policy = ModelPolicy.from_checkpoint(checkpoint)
policy.model.requires_grad_(False)


def model_digest():
    h = hashlib.sha256()
    for name, value in policy.model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


state_before = model_digest()
track = (
    skidpad()
    if args.discipline == "skidpad"
    else generate_track(93000, TrackGenCfg(length_min=220.0, length_max=260.0))
)
track.laps = 1 if args.discipline == "trackdrive" else track.laps
(out / "track.json").write_text(json.dumps(track.to_dict(), indent=2))
env = FSEnv(track, SimCfg(render=False, dr="off", randomize_expert=False, max_time=120.0), seed=args.seed)
world = World(physics_dt=1 / 120, rendering_dt=1 / 20, backend="numpy", device="cpu")
stage = get_current_stage()
create_ground(stage, args.seed)
robot = create_car(stage, world)
create_cones(stage, env.cones_xy, env.cone_colors)
world.reset()
rng = np.random.default_rng(args.seed)
lat = float(rng.uniform(-0.10, 0.10))
yaw_offset = float(rng.uniform(-0.02, 0.02))
x, y, yaw = track.start_pose
q = Rotation.from_euler("z", yaw + yaw_offset).as_quat()
robot.set_world_pose(np.array([x - np.sin(yaw) * lat, y + np.cos(yaw) * lat, 0.36]), np.r_[q[3], q[:3]])
robot.set_linear_velocity(np.zeros(3))
robot.set_angular_velocity(np.zeros(3))
act = AckermannActuators(robot)
for _ in range(120):
    act.apply(0.0, 0.0, 1 / 120)
    world.step(render=False)
cameras, chase = make_cameras()


def pose():
    p, q = robot.get_world_pose()
    r = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
    return p, r


p, r = pose()
pose_cameras(cameras, chase, p, r)
for _ in range(20):
    world.render()
# Warm CUDA kernels using this scene, then discard all temporal/controller state.
obs = env.observe(False)
obs.images = {k: c.get_rgba()[:, :, :3].astype(np.uint8) for k, c in cameras.items()}
with torch.inference_mode():
    for _ in range(4):
        policy(obs, env)
policy.reset()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
assert not any(p.requires_grad for p in policy.model.parameters())
# Preserve the exact procedural scene and starting placement.
stage.GetRootLayer().Export(str(out / "scene.usda"))
hits = set()
contact_events = []


def contact(headers, data):
    for h in headers:
        paths = [str(PhysicsSchemaTools.intToSdfPath(x)) for x in (h.actor0, h.actor1)]
        if any(p.startswith("/World/Car/") for p in paths):
            for path in paths:
                if path.startswith("/World/Cones/") and path not in hits:
                    hits.add(path)
                    contact_events.append({"t": env.frame * 0.05, "cone": path})


subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)


def update_state(frame):
    p, r = pose()
    v = r.T @ robot.get_linear_velocity()
    w = robot.get_angular_velocity()
    env.state = VehicleState(
        float(p[0]),
        float(p[1]),
        float(np.arctan2(r[1, 0], r[0, 0])),
        vx=float(v[0]),
        vy=float(v[1]),
        r=float(w[2]),
        steer=act.angle,
        t=frame * 0.05,
    )
    env.hint, env.s, env.lat = env.mission.update_progress(env.state.x, env.state.y, env.hint, env.s)
    env.frame = frame
    env.frame_desire = env.route_commands.advance(frame=frame, s=env.s, speed=env.state.vx)
    env.referee.update(env.state, env.mission_done())
    return p, r


from fsop.sim.route_commands import RouteCommands

env.route_commands = RouteCommands.for_mission(track, env.mission, 20)
update_state(0)
# Threaded 1 Hz hardware sampling; unified host RAM is reported separately from Torch allocations.
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
telemetry = []
stop = threading.Event()
proc = psutil.Process()


def monitor():
    while not stop.is_set():
        v = psutil.virtual_memory()
        row = {
            "wall_s": time.perf_counter() - run_start,
            "gpu_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "temp_c": pynvml.nvmlDeviceGetTemperature(handle, 0),
            "host_ram_gib": v.used / 2**30,
            "process_rss_gib": proc.memory_info().rss / 2**30,
            "cpu_pct": psutil.cpu_percent(),
        }
        telemetry.append(row)
        stop.wait(1.0)


video = cv2.VideoWriter(str(out / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1280, 640))
assert video.isOpened()
rows = []
plans = []
off_since = None
# Fixed map shows true trajectory for diagnostics; it is never supplied to the policy.
xy = env.cones_xy
lo = xy.min(axis=0) - 5
hi = xy.max(axis=0) + 5
scale = min(290 / (hi - lo)[0], 112 / (hi - lo)[1])
center = (lo + hi) / 2
map_points = np.column_stack([(xy[:, 0] - center[0]) * scale + 160, -(xy[:, 1] - center[1]) * scale + 577]).astype(int)


def composite(steer, accel, label="RUNNING"):
    canvas = np.zeros((640, 1280, 3), np.uint8)
    canvas[:512, :768] = chase.get_rgba()[:, :, :3].astype(np.uint8)
    canvas[:256, 768:] = cameras["road"].get_rgba()[:, :, :3].astype(np.uint8)
    canvas[256:512, 768:] = cameras["wide"].get_rgba()[:, :, :3].astype(np.uint8)
    for point, color in zip(map_points, env.cone_colors, strict=True):
        rgb = (50, 100, 255) if color.value == "blue" else (255, 210, 30)
        cv2.circle(canvas, tuple(point), 1, rgb, -1)
    point = (int((env.state.x - center[0]) * scale + 160), int(-(env.state.y - center[1]) * scale + 577))
    cv2.circle(canvas, point, 4, (255, 60, 60), -1)
    lines = [
        f"ISAAC SIM ZERO-SHOT | {args.discipline} | seed {args.seed} | {label}",
        f"t={env.state.t:.2f}s   speed={env.state.speed:.2f}m/s   progress={env.s:.1f}/{env.mission.s_end:.1f}m",
        f"PhysX cone contacts={len(hits)}   off-course events={env.referee.n_oc}   "
        f"steer={np.rad2deg(steer):.1f}deg   accel={accel:.2f}",
        "Frozen model | original controller | generic Ackermann car | ground-truth route commands",
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (330, 536 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 239, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CHASE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, "ROAD - MODEL INPUT", (780, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "WIDE - MODEL INPUT", (780, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


startup_seconds = time.perf_counter() - process_start
run_start = time.perf_counter()
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
with torch.inference_mode():
    for frame in range(2400):
        tick = time.perf_counter()
        p, r = pose()
        pose_cameras(cameras, chase, p, r)
        t = time.perf_counter()
        world.render()
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        obs = env.observe(False)
        obs.images = {k: c.get_rgba()[:, :, :3].astype(np.uint8) for k, c in cameras.items()}
        render_s = time.perf_counter() - t
        t = time.perf_counter()
        steer, accel, info = policy(obs, env)
        torch.cuda.synchronize()
        inference_s = time.perf_counter() - t
        if not np.isfinite(info["plan"]).all() or not np.isfinite([steer, accel]).all():
            env.referee.finish("nonfinite_model_output")
            break
        plans.append(info["plan"])
        t = time.perf_counter()
        if frame % 2 == 0:
            image = composite(steer, accel)
            video.write(image)
            if frame == 0:
                cv2.imwrite(str(out / "first_frame.jpg"), image)
        video_s = time.perf_counter() - t
        row = {
            "frame": frame,
            "t": frame * 0.05,
            "x": env.state.x,
            "y": env.state.y,
            "yaw": env.state.yaw,
            "speed": env.state.speed,
            "s": env.s,
            "lateral_m": env.lat,
            "steer": steer,
            "accel": accel,
            "contacts": len(hits),
            "off_course": env.referee.off,
            "desire": obs.desire.tolist(),
            "render_s": render_s,
            "inference_s": inference_s,
            "video_s": video_s,
        }
        t = time.perf_counter()
        act.apply(steer, accel)
        for _ in range(6):
            world.step(render=False)
        row["physics_s"] = time.perf_counter() - t
        p, r = update_state(frame + 1)
        if env.referee.off:
            if off_since is None:
                off_since = env.state.t
            if env.state.t - off_since >= 5.0:
                env.referee.finish("off_course_5s")
        else:
            off_since = None
        if r[2, 2] < 0.3:
            env.referee.finish("rollover")
        if env.state.t >= 120.0:
            env.referee.finish("timeout")
        row["step_wall_s"] = time.perf_counter() - tick
        rows.append(row)
        if frame % 100 == 0:
            print(
                "STEP",
                args.discipline,
                args.seed,
                frame,
                round(env.s, 1),
                len(hits),
                round(row["step_wall_s"], 3),
                flush=True,
            )
        if env.referee.done:
            break
elapsed = time.perf_counter() - run_start
stop.set()
thread.join()
p, r = pose()
pose_cameras(cameras, chase, p, r)
world.render()
rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
final_frame = composite(steer, accel, env.referee.result.upper())
cv2.imwrite(str(out / "last_frame.jpg"), final_frame)
for _ in range(30):
    video.write(final_frame)
video.release()
summary = env.referee.summary()
# Legacy geometric footprint contacts are a diagnostic, never the physical cone-hit metric.
summary["legacy_footprint_contacts"] = summary.pop("doo")
summary.update(
    {
        "physical_cone_hits": len(hits),
        "contact_events": contact_events,
        "seed": args.seed,
        "seconds_simulated": env.state.t,
        "wall_seconds": elapsed,
        "real_time_factor": env.state.t / elapsed,
        "control_hz_wall": len(rows) / elapsed,
        "startup_seconds": startup_seconds,
        "start_lateral_offset_m": lat,
        "start_yaw_offset_rad": yaw_offset,
        "checkpoint_sha256": checkpoint_before,
        "state_sha256": state_before,
        "frozen_unchanged": checkpoint_before == sha() and state_before == model_digest(),
        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
)
assert summary["frozen_unchanged"]
for key in ["render_s", "inference_s", "physics_s", "video_s", "step_wall_s"]:
    values = np.array([row[key] for row in rows])
    summary[key] = {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }
(out / "summary.json").write_text(json.dumps(summary, indent=2))
(out / "frames.json").write_text(json.dumps(rows))
(out / "telemetry.json").write_text(json.dumps(telemetry, indent=2))
np.save(out / "plans.npy", plans)
print("RESULT", json.dumps(summary), flush=True)
app.close()
