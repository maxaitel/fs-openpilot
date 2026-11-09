"""One Isaac Lab simulation owner for the physical FS vehicle."""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from fsop.sim.randomization import postprocess
from fsop.sim.referee import Referee
from fsop.sim.route_commands import RouteCommands
from fsop.sim.vehicle import VehicleState

_EVAL_DIR = str(Path(__file__).resolve().parents[1] / "isaac_eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)


class _RigidObjectView:
    def __init__(self, asset, wheel_controllers=None):
        self.asset = asset
        self.wheel_controllers = wheel_controllers

    @property
    def prim_paths(self):
        return list(self.asset.root_physx_view.prim_paths)

    def get_world_poses(self):
        poses = self.asset.data.root_link_pose_w.detach().cpu().numpy().copy()
        return poses[:, :3], poses[:, 3:7]

    def get_world_pose(self):
        positions, orientations = self.get_world_poses()
        if len(positions) != 1:
            raise RuntimeError("Expected exactly one vehicle rigid body")
        return positions[0], orientations[0]

    def set_world_pose(self, position, orientation):
        import torch

        pose = torch.as_tensor(np.r_[position, orientation], dtype=torch.float32, device=self.asset.device).reshape(
            1, 7
        )
        self.asset.write_root_pose_to_sim(pose)

    def _velocity(self):
        return self.asset.data.root_com_vel_w

    def get_linear_velocity(self):
        return self._velocity()[0, :3].detach().cpu().numpy().copy()

    def get_angular_velocity(self):
        return self._velocity()[0, 3:].detach().cpu().numpy().copy()

    def set_linear_velocity(self, velocity):
        import torch

        state = self._velocity().clone()
        state[0, :3] = torch.as_tensor(velocity, dtype=state.dtype, device=state.device)
        self.asset.write_root_velocity_to_sim(state)

    def set_angular_velocity(self, velocity):
        import torch

        state = self._velocity().clone()
        state[0, 3:] = torch.as_tensor(velocity, dtype=state.dtype, device=state.device)
        self.asset.write_root_velocity_to_sim(state)


class LabDrive:
    def __init__(self, env, variant="matched", assets=Path("isaac_eval/assets"), render=True):
        if variant not in ("matched", "visual_stress_v1"):
            raise ValueError(f"Unknown appearance variant: {variant}")
        if env.cfg.fps != 20:
            raise ValueError("LabDrive requires 20 Hz metadata")
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import get_current_stage

        from .runtime import create_simulation

        enable_extension("isaacsim.sensors.camera")
        from appearance_guard import assert_appearance_only, physical_contract
        from parity_scene import dress_matched_scene
        from rendering import configure_rendering
        from scene import create_car, create_cones, create_ground
        from stress_scene import apply_visual_stress, validate_stress_assets

        self.metadata_env = self.env = env
        self.render_enabled = render
        self.variant = variant
        self.sim = create_simulation()
        self.scene = InteractiveScene(InteractiveSceneCfg(num_envs=1, env_spacing=10.0))
        self.stage = get_current_stage()
        create_ground(self.stage, env.seed)
        authored_car = create_car(self.stage)
        create_cones(self.stage, env.cones_xy, env.cone_colors)
        if len(env.cones_xy) == 0:
            raise ValueError("LabDrive requires at least one physical cone")
        car_asset = RigidObject(RigidObjectCfg(prim_path="/World/Car/Chassis", spawn=None))
        cones_asset = RigidObject(RigidObjectCfg(prim_path="/World/Cones/c.*", spawn=None))
        self.scene.rigid_objects["car"] = car_asset
        self.scene.rigid_objects["cones"] = cones_asset
        self.robot = _RigidObjectView(car_asset, authored_car.wheel_controllers)
        self.cones = _RigidObjectView(cones_asset)
        before = physical_contract(self.stage)
        self.appearance = dress_matched_scene(self.stage, env, Path(assets))
        if variant == "visual_stress_v1":
            _, _, asset_hash = validate_stress_assets(Path(assets))
            self.appearance["stress"] = apply_visual_stress(self.stage, env, Path(assets))
            self.appearance["asset_sha256"] = asset_hash
        self.appearance["physics_sha256"] = assert_appearance_only(before, self.stage)
        if render:
            configure_rendering()
        self.act = None
        self.cameras = {}
        self.world_camera = None
        self.projection = None
        self.hits = set()
        self.contacts = []
        self._subscription = None
        self._reset = False
        self._closed = False
        self._latitudes = []
        self._obs_cache = None
        self._images_captured = False

    def reset(self, start_pose=None, start_speed=0.0):
        if self._reset:
            raise RuntimeError("LabDrive reset is one-shot; create a fresh process for another episode")
        if self._closed:
            raise RuntimeError("LabDrive is closed")
        if not np.isfinite(start_speed) or start_speed < 0:
            raise ValueError("start_speed must be finite and nonnegative")
        import omni.physx
        from pxr import PhysicsSchemaTools
        from scene import AckermannActuators, make_cameras

        self._reset = True
        self.sim.reset()
        self.scene.reset()
        x, y, yaw = self.env.track.start_pose if start_pose is None else start_pose
        if not np.isfinite([x, y, yaw]).all():
            raise ValueError("start_pose must contain finite x, y and yaw")
        q = Rotation.from_euler("z", yaw).as_quat()
        self.robot.set_world_pose(np.array([x, y, 0.36]), np.r_[q[3], q[:3]])
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))
        self.act = AckermannActuators(self.robot)
        for _ in range(120):
            self.act.apply(0, 0, 1 / 120)
            self._physics_step()
        self.robot.set_linear_velocity(np.array([start_speed * np.cos(yaw), start_speed * np.sin(yaw), 0.0]))
        self.robot.set_angular_velocity(np.zeros(3))
        paths = [f"/World/Cones/c{i:04d}" for i in range(len(self.env.cones_xy))]
        if list(self.cones.prim_paths) != paths:
            raise RuntimeError("Native cone view does not match the realized cone paths")
        if self.render_enabled:
            self.cameras, _ = make_cameras(with_chase=False)
        e = self.env
        e.referee = Referee(e.track, e.cones_xy, e.cone_colors, e.vp)
        e.pending_events = []
        e.noise_val = 0.0
        p, r = self.pose()
        e.hint, e.s = e.mission.init_progress(float(p[0]), float(p[1]))
        e.frame = 0
        self._sync_state(0, False)
        e.route_commands = RouteCommands.for_mission(e.track, e.mission, e.cfg.fps)
        e.frame_desire = e.route_commands.advance(frame=0, s=e.s, speed=e.state.vx)

        def contact(headers, data):
            for header in headers:
                actor_paths = [str(PhysicsSchemaTools.intToSdfPath(i)) for i in (header.actor0, header.actor1)]
                if any(path.startswith("/World/Car/") for path in actor_paths):
                    for path in actor_paths:
                        if path.startswith("/World/Cones/") and path not in self.hits:
                            self.hits.add(path)
                            self.contacts.append(
                                {
                                    "t": e.frame / 20,
                                    "control_frame": e.frame,
                                    "cone": path,
                                    "time_convention": "start of the control interval containing first contact",
                                }
                            )

        self._subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
        if self.render_enabled:
            for _ in range(20):
                self.sim.render()
        return self.observe(render=self.render_enabled)

    def pose(self):
        p, q = self.robot.get_world_pose()
        return p, Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()

    def _sync_state(self, frame, update_referee=True):
        e = self.env
        p, r = self.pose()
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
        velocity = Rotation.from_euler("z", yaw).as_matrix().T @ self.robot.get_linear_velocity()
        e.state = VehicleState(
            float(p[0]),
            float(p[1]),
            yaw,
            vx=float(velocity[0]),
            vy=float(velocity[1]),
            r=float(self.robot.get_angular_velocity()[2]),
            steer=float(self.act.angle),
            accel=float(self.act.accel),
            t=frame / 20,
        )
        e.hint, e.s, e.lat = e.mission.update_progress(e.state.x, e.state.y, e.hint, e.s)
        e.frame = frame
        if update_referee:
            e.pending_events.extend(e.referee.update(e.state, e.mission_done()))
            e.frame_desire = e.route_commands.advance(frame=frame, s=e.s, speed=e.state.vx)
            if e.state.t >= e.max_time and not e.referee.done:
                e.referee.finish("timeout")
            if r[2, 2] < 0.3 and not e.referee.done:
                e.referee.finish("isaac_rollover")
        from parity_geometry import pose_model_cameras

        self.world_camera, self.projection = pose_model_cameras(self.cameras, p, r, e.T_ego_cam(), e.T_world_ego())
        self._latitudes.append(float(e.lat))
        return p, r

    def _physics_step(self):
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(1 / 120)

    def observe(self, render=True):
        if not self._reset or self._closed:
            raise RuntimeError("LabDrive must be reset and open before observing")
        e = self.env
        if self._obs_cache is None:
            self._obs_cache = e.observe(False)
        obs = self._obs_cache
        if render and not self._images_captured:
            if not self.render_enabled:
                raise RuntimeError("LabDrive was created without cameras")
            from rendering import capture

            capture(self.sim)
            obs.images = {
                name: postprocess(self.cameras[name].get_rgba()[:, :, :3].astype(np.uint8), e.scene, e.rng, e.state.r)
                for name in e.cfg.cameras
            }
            self._images_captured = True
        return obs

    def _advance(self, steer, accel, render, update_referee):
        if not np.isfinite([steer, accel]).all():
            raise ValueError("Action must be finite")
        self.act.apply(steer, accel)
        for _ in range(6):
            self._physics_step()
        self._sync_state(self.env.frame + 1, update_referee=update_referee)
        if not update_referee:
            e = self.env
            e.frame_desire = e.route_commands.advance(frame=e.frame, s=e.s, speed=e.state.vx)
        self._obs_cache = None
        self._images_captured = False
        return self.observe(render=render)

    def step(self, steer, accel, render=False):
        if not self._reset or self._closed:
            raise RuntimeError("LabDrive must be reset and open before stepping")
        if self.env.referee.done:
            raise RuntimeError("Cannot step a finished episode")
        return self._advance(steer, accel, render, update_referee=True)

    def step_terminal_brake(self, render=False):
        if not self._reset or self._closed:
            raise RuntimeError("LabDrive must be reset and open before stepping")
        e = self.env
        if not e.referee.done or e.referee.result != "finished":
            raise RuntimeError("Terminal braking requires a finished episode")
        speed = float(np.hypot(e.state.vx, e.state.vy))
        if not np.isfinite(speed) or speed >= 0.5:
            raise RuntimeError("Terminal braking requires speed below 0.5 m/s")
        return self._advance(self.act.angle, -2.0, render, update_referee=False)

    def summary(self):
        e = self.env
        result = e.referee.summary()
        result.update(
            final_state=asdict(e.state),
            physical_cone_hits=len(self.hits),
            contact_events=list(self.contacts),
            progress=float(e.s / e.mission.s_end),
            rms_lat=float(np.sqrt(np.mean(np.square(self._latitudes)))),
        )
        return result

    def close(self):
        if not self._closed:
            self._subscription = None
            self.sim._disable_app_control_on_stop_handle = True
            self.sim.stop()
            self.sim.clear_instance()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
