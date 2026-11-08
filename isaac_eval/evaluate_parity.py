# ruff: noqa: E402, I001
import argparse
import hashlib
import json
import platform
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

parser = argparse.ArgumentParser(description="Run one frozen matched case with lossless inference evidence")
parser.add_argument("--case", required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--protocol", type=Path)
parser.add_argument("--pilot-steps", type=int, default=0)
args = parser.parse_args()
if (args.protocol is None) != (args.pilot_steps > 0):
    parser.error("Use a verified protocol for a scored run, or --pilot-steps without a protocol")
process_start = time.perf_counter()
from isaacsim import SimulationApp

app = SimulationApp(
    {"headless": True, "renderer": "RayTracedLighting", "anti_aliasing": 0, "disable_viewport_updates": True}
)
import numpy as np
import psutil
import pynvml
import torch
from scipy.spatial.transform import Rotation
from pxr import PhysicsSchemaTools
import omni.physx
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import get_current_stage
from fsop.eval.closed_loop import ModelPolicy
from fsop.geometry.camera import CAMERA_MODELS
from fsop.sim.randomization import postprocess
from fsop.sim.vehicle import VehicleState
from isaac_eval.evidence import EvidenceWriter, ImageHeadCapture, finalize_evidence
from isaac_eval.parity_protocol import GRAPH_FILES, runtime_sources, validate, verify_baseline
from isaac_eval.parity_cases import (
    get_case,
    build_environment,
    realized_environment,
    json_sha256,
    sha256_file,
    verify_checkpoint,
)
from appearance_guard import physical_contract, assert_appearance_only
from export_scene import export_scene
from rendering import configure_rendering, capture, PROFILE
from scene import create_ground, create_car, create_cones, make_cameras, AckermannActuators
from parity_geometry import pose_model_cameras
from parity_scene import dress_matched_scene

configure_rendering()
out = args.out
out.mkdir(parents=True, exist_ok=False)
manifest = json.loads(args.manifest.read_text())
assert manifest["manifest_sha256"] == json_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
case = get_case(args.case)
case_spec = next(row for row in manifest["cases"] if row["case_id"] == case.case_id)
realization_spec = next(row for row in manifest["realizations"] if row["case_id"] == case.case_id)
expected_realization = json.loads((args.manifest.parent / realization_spec["path"]).read_text())
protocol = json.loads(args.protocol.read_text()) if args.protocol else None
if protocol:
    validate(protocol, manifest)
    (out / "protocol.json").write_bytes(args.protocol.read_bytes())
verify_baseline()
checkpoint = Path("/frozen/model.pt")
checkpoint_sha = verify_checkpoint(checkpoint)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
policy = ModelPolicy.from_checkpoint(checkpoint, controller="pp")
policy.model.requires_grad_(False)


def model_digest():
    h = hashlib.sha256()
    for key, value in policy.model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


state_sha = model_digest()
env = build_environment(case)
realization = realized_environment(env)
assert realization == expected_realization, "Environment differs from frozen case realization"
(out / "realization.json").write_text(json.dumps(realization, indent=2))
(out / "track.json").write_text(json.dumps(env.track.to_dict(), indent=2))
world = World(physics_dt=1 / 120, rendering_dt=1 / 20, backend="numpy", device="cpu")
stage = get_current_stage()
create_ground(stage, case.seed)
robot = create_car(stage, world)
create_cones(stage, env.cones_xy, env.cone_colors)
before = physical_contract(stage)
appearance = dress_matched_scene(stage, env, Path("isaac_eval/assets"))
appearance["physics_sha256"] = assert_appearance_only(before, stage)
(out / "appearance.json").write_text(json.dumps(appearance, indent=2))
world.reset()
x, y, yaw = env.track.start_pose
q = Rotation.from_euler("z", yaw).as_quat()
robot.set_world_pose(np.array([x, y, 0.36]), np.r_[q[3], q[:3]])
robot.set_linear_velocity(np.zeros(3))
robot.set_angular_velocity(np.zeros(3))
act = AckermannActuators(robot)
for _ in range(120):
    act.apply(0, 0, 1 / 120)
    world.step(render=False)
robot.set_linear_velocity(np.zeros(3))
robot.set_angular_velocity(np.zeros(3))
cones = RigidPrim(
    [f"/World/Cones/c{i:04d}" for i in range(len(env.cones_xy))],
    name="observed_cones",
    reset_xform_properties=False,
    prepare_contact_sensors=False,
    disable_stablization=False,
)
cones.initialize()
assert list(cones.prim_paths) == [f"/World/Cones/c{i:04d}" for i in range(len(env.cones_xy))]
cameras, _ = make_cameras(with_chase=False)


def pose():
    p, q = robot.get_world_pose()
    return p, Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()


def sync_state(frame, update_referee=True):
    p, r = pose()
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    planar_rotation = Rotation.from_euler("z", yaw).as_matrix()
    v = planar_rotation.T @ robot.get_linear_velocity()
    env.state = VehicleState(
        float(p[0]),
        float(p[1]),
        yaw,
        vx=float(v[0]),
        vy=float(v[1]),
        r=float(robot.get_angular_velocity()[2]),
        steer=float(act.angle),
        accel=float(act.accel),
        t=frame / 20,
    )
    env.hint, env.s, env.lat = env.mission.update_progress(env.state.x, env.state.y, env.hint, env.s)
    env.frame = frame
    if update_referee:
        env.pending_events.extend(env.referee.update(env.state, env.mission_done()))
        env.frame_desire = env.route_commands.advance(frame=frame, s=env.s, speed=env.state.vx)
    return p, r


p, r = sync_state(0, False)
assert np.linalg.norm(p[:2] - np.array([x, y])) < 1e-4
assert abs(np.angle(np.exp(1j * (env.state.yaw - yaw)))) < 1e-5
initial_physical_state = asdict(env.state)
pose_model_cameras(cameras, p, r, env.T_ego_cam(), env.T_world_ego())
for _ in range(20):
    world.render()
capture(world)
initial_obs = env.observe(False)
warm = replace(initial_obs, images={name: np.zeros((256, 512, 3), np.uint8) for name in cameras})
with torch.inference_mode():
    for _ in range(4):
        policy(warm, env)
policy.reset()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
assert not any(p.requires_grad for p in policy.model.parameters())
export = export_scene(stage, out / "scene.usda", Path.cwd())
export["sha256"] = {name: sha256_file(out / "assets" / name) for name in export["dependencies"]}
(out / "export.json").write_text(json.dumps(export, indent=2))
hits, contacts = set(), []


def contact(headers, data):
    for header in headers:
        paths = [str(PhysicsSchemaTools.intToSdfPath(i)) for i in (header.actor0, header.actor1)]
        if any(path.startswith("/World/Car/") for path in paths):
            for path in paths:
                if path.startswith("/World/Cones/") and path not in hits:
                    hits.add(path)
                    contacts.append(
                        {
                            "t": env.frame / 20,
                            "control_frame": env.frame,
                            "cone": path,
                            "time_convention": "start of the control interval containing first contact",
                        }
                    )


subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
metadata = {
    "schema_version": 1,
    "fps": 20,
    "backend": "Isaac PhysX",
    "case": case_spec,
    "checkpoint_sha256": checkpoint_sha,
    "state_sha256": state_sha,
    "manifest_sha256": manifest["manifest_sha256"],
    "protocol_sha256": sha256_file(args.protocol) if args.protocol else None,
    "track": env.track.to_dict(),
    "initial_cones_xyz": np.c_[env.cones_xy, np.full(len(env.cones_xy), 0.02)].tolist(),
    "cone_colors": [color.value for color in env.cone_colors],
    "expert_params": asdict(env.expert_params),
    "vehicle_params": asdict(env.vp),
    "label_dt": env.cfg.label_dt,
    "nominal_detection_calibration": env.T_ego_cam().tolist(),
    "intrinsics": {name: cam.K.tolist() for name, cam in CAMERA_MODELS.items()},
    "initial_physical_state": initial_physical_state,
    "source_sha256": runtime_sources(),
    "graph_sha256": {path: sha256_file(path) for path in GRAPH_FILES},
    "velocity_convention": "World XY velocity rotated by planar yaw; full body velocity saved separately",
    "privileged_inputs": [
        "speed",
        "sampled nominal camera calibration",
        "route commands from mission progress",
        "mission progress for launch/finish braking",
    ],
    "prediction_provenance": "One original ModelPolicy call per20Hz tick; image-head hook returns None",
    "uncertainty_semantics": "exp(clamped model second plan output), uncalibrated model scale",
    "image_history": "RGB processed once; image pair max(0,t-4),t; 96-step feature queue;100-step pulse queue",
    "oracle": "Offline custom bicycle expert, never supplied to policy or PhysX",
    "pilot_only": bool(args.pilot_steps),
}
telemetry, rows = [], []
stop = threading.Event()
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
proc = psutil.Process()


def monitor():
    while not stop.is_set():
        telemetry.append(
            {
                "wall_s": time.perf_counter() - run_start,
                "gpu_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                "temp_c": pynvml.nvmlDeviceGetTemperature(handle, 0),
                "host_ram_gib": psutil.virtual_memory().used / 2**30,
                "process_rss_gib": proc.memory_info().rss / 2**30,
                "cpu_pct": psutil.cpu_percent(),
            }
        )
        stop.wait(1)


startup_seconds = time.perf_counter() - process_start
run_start = time.perf_counter()
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
with (
    EvidenceWriter(out / "evidence", metadata) as writer,
    ImageHeadCapture(policy.model) as head,
    torch.inference_mode(),
):
    for frame in range(args.pilot_steps or 2400):
        tick = time.perf_counter()
        p, r = pose()
        world_camera, projection = pose_model_cameras(cameras, p, r, env.T_ego_cam(), env.T_world_ego())
        started = time.perf_counter()
        capture(world)
        obs = env.observe(False)
        obs.images = {
            name: postprocess(cameras[name].get_rgba()[:, :, :3].astype(np.uint8), env.scene, env.rng, env.state.r)
            for name in env.cfg.cameras
        }
        render_s = time.perf_counter() - started
        requested = obs.desire.copy()
        applied = requested.copy()
        if case.command_mode == "zero":
            applied[:] = 0
        elif case.command_mode == "swapped":
            applied = applied[[0, 2, 1, 3, 4, 5, 6, 7]]
        control_obs = replace(obs, desire=applied)
        head.clear()
        started = time.perf_counter()
        steer, accel, info = policy(control_obs, env)
        torch.cuda.synchronize()
        policy_s = time.perf_counter() - started
        if not np.isfinite(info["plan"]).all() or not np.isfinite([steer, accel]).all():
            raise ValueError("Nonfinite model output; failed run requires inspection")
        positions, quaternions = cones.get_world_poses()
        rng_before = json_sha256(env.rng.bit_generator.state)
        record = {
            "frame": frame,
            "images": obs.images,
            "prediction": info,
            "image_head": head.output,
            "state": asdict(obs.state),
            "body": {"position": p, "rotation": r, "velocity_world": robot.get_linear_velocity(),
                     "velocity_body": r.T @ robot.get_linear_velocity(),
                     "angular_velocity_world": robot.get_angular_velocity()},
            "progress": {"s": obs.s, "hint": env.hint, "lat": obs.lat},
            "T_plan_ego_camera": projection,
            "T_world_camera_body": world_camera,
            "nominal_detection_calibration": env.T_ego_cam(),
            "actual_cones_xyz": positions,
            "actual_cone_quaternions": quaternions,
            "action": [float(steer), float(accel)],
            "commands": {"requested": requested, "applied": applied, "mode": case.command_mode},
            "counters": {
                "laps": len(env.referee.lap_times()),
                "total_laps": env.track.laps,
                "legacy_doo": env.referee.n_doo,
                "physical_contacts": len(hits),
                "off_course": env.referee.n_oc,
                "result": env.referee.result,
            },
            "events": [asdict(event) for event in obs.events],
            "diagnostics": {"policy_with_capture_ms": policy_s * 1000, "render_ms": render_s * 1000},
            "temporal": {"older_image_frame": max(0, frame - 4), "current_image_frame": frame},
        }
        started = time.perf_counter()
        writer.append(record)
        evidence_s = time.perf_counter() - started
        assert json_sha256(env.rng.bit_generator.state) == rng_before
        row = {
            "frame": frame,
            "t": obs.t,
            "x": obs.state.x,
            "y": obs.state.y,
            "yaw": obs.state.yaw,
            "speed": obs.state.vx,
            "s": obs.s,
            "lat": obs.lat,
            "render_s": render_s,
            "policy_with_capture_s": policy_s,
            "evidence_s": evidence_s,
            "legacy_doo": env.referee.n_doo,
            "physical_contacts": len(hits),
        }
        started = time.perf_counter()
        act.apply(steer, accel)
        for _ in range(6):
            world.step(render=False)
        row["physics_s"] = time.perf_counter() - started
        p, r = sync_state(frame + 1)
        if env.state.t >= 120 and not env.referee.done:
            env.referee.finish("timeout")
        if r[2, 2] < 0.3 and not env.referee.done:
            env.referee.finish("isaac_rollover")
        row["step_wall_s"] = time.perf_counter() - tick
        rows.append(row)
        if frame % 100 == 0:
            print("STEP", case.case_id, frame, round(env.s, 2), len(hits), round(row["step_wall_s"], 3), flush=True)
        if env.referee.done:
            break
    evidence_seconds = writer.write_seconds
elapsed = time.perf_counter() - run_start
stop.set()
thread.join()
summary = env.referee.summary()
summary.update(
    case=case_spec,
    pilot_only=bool(args.pilot_steps),
    final_state=asdict(env.state),
    events=[asdict(event) for event in env.referee.events],
    pending_events=[asdict(event) for event in env.pending_events],
    recorded_frames=len(rows),
    physical_cone_hits=len(hits),
    contact_events=contacts,
    progress=float(env.s / env.mission.s_end),
    rms_lat=float(np.sqrt(np.mean([row["lat"] ** 2 for row in rows]))),
    wall_seconds=elapsed,
    startup_seconds=startup_seconds,
    evidence_seconds=evidence_seconds,
    seconds_simulated=env.state.t,
    real_time_factor=env.state.t / elapsed,
    checkpoint_sha256=checkpoint_sha,
    state_sha256=state_sha,
    frozen_unchanged=verify_checkpoint(checkpoint) == checkpoint_sha and model_digest() == state_sha,
    manifest_sha256=manifest["manifest_sha256"],
    protocol_sha256=sha256_file(args.protocol) if args.protocol else None,
    render_profile=PROFILE,
    physics_sha256=appearance["physics_sha256"],
    torch_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
    torch_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    runtime={"python": platform.python_version(), "torch": torch.__version__},
)
assert summary["frozen_unchanged"]
(out / "summary.json").write_text(json.dumps(summary, indent=2))
finalize_evidence(out / "evidence", out / "summary.json")
(out / "frames.json").write_text(json.dumps(rows))
(out / "telemetry.json").write_text(json.dumps(telemetry, indent=2))
print("RESULT", json.dumps(summary), flush=True)
app.close()
