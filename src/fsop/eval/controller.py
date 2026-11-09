"""Plan followers: turn a predicted plan (ego frame) into steering / acceleration commands."""

from __future__ import annotations

import numpy as np

from fsop.sim.expert import T_IDXS
from fsop.sim.vehicle import VehicleParams

MIN_STABLE_DELAY = 0.05  # openpilot drive_helpers


class PlanPurePursuit:
  """Pure pursuit on the predicted path (lookahead point at a time horizon) + speed from the plan."""

  def __init__(self, vp: VehicleParams, t_min: float = 0.35, t_max: float = 1.0, k: float = 0.03, kp_v: float = 1.5,
               action_t: float = 0.3):
    self.vp, self.t_min, self.t_max, self.k, self.kp_v, self.action_t = vp, t_min, t_max, k, kp_v, action_t

  def __call__(self, plan: np.ndarray, v: float) -> tuple[float, float]:
    t_la = float(np.clip(self.t_min + self.k * v, self.t_min, self.t_max))
    x = np.interp(t_la, T_IDXS, plan[:, 0])
    y = np.interp(t_la, T_IDXS, plan[:, 1])
    # path is relative to the CoG; pure pursuit geometry from the rear axle
    xr = x + self.vp.lr
    ld2 = max(xr * xr + y * y, 1.0)
    steer = float(np.arctan(2 * self.vp.wheelbase * y / ld2))
    v_tgt = float(np.interp(self.action_t, T_IDXS, plan[:, 3]))
    a_plan = float(np.interp(self.action_t, T_IDXS, plan[:, 6]))
    accel = self.kp_v * (v_tgt - v) + 0.5 * a_plan
    return steer, float(np.clip(accel, -self.vp.max_brake, self.vp.max_accel))


class OpenpilotFollower:
  """Port of openpilot's get_curvature_from_plan / get_accel_from_plan (MIT), with a kinematic steer map."""

  def __init__(self, vp: VehicleParams, lat_action_t: float = 0.2, long_action_t: float = 0.3):
    self.vp, self.lat_t, self.long_t = vp, lat_action_t, long_action_t

  def __call__(self, plan: np.ndarray, v: float) -> tuple[float, float]:
    yaws, yaw_rates = plan[:, 11], plan[:, 14]
    vego = max(v, 1.0)
    if self.lat_t < MIN_STABLE_DELAY:
      psi = (self.lat_t / MIN_STABLE_DELAY) * np.interp(MIN_STABLE_DELAY, T_IDXS, yaws)
    else:
      psi = np.interp(self.lat_t, T_IDXS, yaws)
    curv = 2 * psi / (vego * self.lat_t) - yaw_rates[0] / vego
    steer = float(np.arctan(self.vp.wheelbase * curv))
    speeds, accels = plan[:, 3], plan[:, 6]
    v_target = np.interp(self.long_t, T_IDXS, speeds)
    a_target = 2 * (v_target - speeds[0]) / self.long_t - accels[0]
    return steer, float(np.clip(a_target, -self.vp.max_brake, self.vp.max_accel))


def make_controller(name: str, vp: VehicleParams):
  controller_contract(name)
  return {"pp": PlanPurePursuit, "op": OpenpilotFollower, "op_tangent_v1": OpenpilotTangentFollower}[name](vp)

# Explicit contract: the old "pp" and "op" modes retain their original semantics.
OP_TANGENT_V1 = {
  "version": 1,
  "mode": "op_tangent_v1",
  "steering": "openpilot_follower_body_vx",
  "steering_action_time_s": 0.2,
  "steering_min_speed_mps": 1.0,
  "steering_delay_floor_s": 0.05,
  "longitudinal": "knotwise_planar_tangent",
  "action_time_s": 0.3,
  "speed_gain": 1.5,
  "accel_feedforward": 0.5,
  "speed_floor_mps": 0.001,
  "current_velocity": "measured_body_vx_vy",
  "mission_layer": "mission_brake_after_follower",
  "mission_brake_decel_mps2": 3.5,
  "mission_launch_accel_mps2": 2.0,
}


def controller_contract(name: str) -> dict:
  if name == "op_tangent_v1":
    return dict(OP_TANGENT_V1)
  if name in ("pp", "op"):
    return {"version": 1, "mode": name, "mission_layer": "mission_brake_after_follower"}
  raise ValueError(f"Unknown controller mode: {name}")


def validate_checkpoint_controller(checkpoint_contract, mode: str) -> None:
  """Legacy weights can be explicitly tested with a new mode; bound weights cannot change modes."""
  controller_contract(mode)
  if checkpoint_contract is not None:
    expected = controller_contract(checkpoint_contract.get("mode")) if isinstance(checkpoint_contract, dict) else None
    if checkpoint_contract != expected:
      raise ValueError("Invalid checkpoint controller contract")
    if checkpoint_contract["mode"] != mode:
      raise ValueError("Checkpoint controller contract does not match requested controller")


def validate_checkpoint_payload(payload, requested_mode: str) -> None:
  """Verify a checkpoint's declared mode before loading or evaluating its weights."""
  if not isinstance(payload, dict):
    raise ValueError("Checkpoint must be a mapping")
  config = payload.get("config") or {}
  if not isinstance(config, dict):
    raise ValueError("Checkpoint config must be a mapping")
  trained_mode = config.get("controller_mode", "pp")
  contract = payload.get("controller_contract")
  if trained_mode != "pp" and contract is None:
    raise ValueError("Checkpoint controller contract is missing")
  if contract is not None:
    validate_checkpoint_controller(contract, trained_mode)
  validate_checkpoint_controller(contract, requested_mode)


class OpenpilotTangentFollower:
  """OP steering with experimental tangent-based speed and acceleration feedback."""

  def __init__(self, vp: VehicleParams):
    self.vp = vp
    self.steering = OpenpilotFollower(vp)

  def __call__(self, plan: np.ndarray, vx: float, vy: float | None = None) -> tuple[float, float]:
    if vy is None or not np.isfinite([vx, vy]).all():
      raise ValueError("op_tangent_v1 requires finite measured body vx and vy")
    plan = np.asarray(plan)
    if plan.shape != (len(T_IDXS), 15) or not np.isfinite(plan).all():
      raise ValueError("op_tangent_v1 requires a finite 33 by 15 plan")
    steer, _ = self.steering(plan, vx)
    velocity, acceleration = plan[:, 3:5], plan[:, 6:8]
    speed = np.linalg.norm(velocity, axis=1)
    tangent = np.einsum("ij,ij->i", velocity, acceleration) / np.maximum(speed, 0.001)
    v_target = np.interp(0.3, T_IDXS, speed)
    a_target = np.interp(0.3, T_IDXS, tangent)
    accel = 1.5 * (v_target - np.hypot(vx, vy)) + 0.5 * a_target
    return steer, float(np.clip(accel, -self.vp.max_brake, self.vp.max_accel))
