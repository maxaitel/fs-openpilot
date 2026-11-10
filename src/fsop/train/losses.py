"""Losses: Laplacian NLL for the plan (openpilot-style), CenterNet focal + L1 for cone heads."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from fsop.sim.expert import T_IDXS

# per-column weights: position, velocity, acceleration, euler, orientation rate (x3 each)
PLAN_COL_W = torch.tensor([1.0, 1.0, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3])
# per-time weights: full weight up to 4 s, fading to 0.2 at 10 s (far beyond the cones' visibility)
PLAN_T_W = torch.tensor([1.0 if t <= 4.0 else max(0.2, 1.0 - 0.8 * (t - 4.0) / 6.0) for t in T_IDXS])


def pp_steer_from_op_plan(mu: torch.Tensor, speed: torch.Tensor, lr: torch.Tensor,
                          wheelbase: torch.Tensor) -> torch.Tensor:
  mu, speed, lr, wheelbase = (x.float() for x in (mu, speed, lr, wheelbase))
  times = torch.as_tensor(T_IDXS, dtype=mu.dtype, device=mu.device)
  t = (0.35 + 0.03 * speed).clamp(0.35, 1.0)
  hi = torch.searchsorted(times, t.contiguous()).clamp(1, len(T_IDXS) - 1)
  lo = hi - 1
  a = (t - times[lo]) / (times[hi] - times[lo])
  rows = torch.arange(mu.shape[0], device=mu.device)
  point = mu[rows, lo, :2] * (1 - a[:, None]) + mu[rows, hi, :2] * a[:, None]
  x, y = point[:, 0] + lr, -point[:, 1]
  return torch.atan(2 * wheelbase * y / (x.square() + y.square()).clamp_min(1.0))


def steering_loss(mu: torch.Tensor, b: dict) -> torch.Tensor:
  pred = pp_steer_from_op_plan(mu, b["speed"], b["vehicle_lr"], b["vehicle_wheelbase"])
  mask = b["steer_valid"].bool()
  error = F.smooth_l1_loss(pred, b["steer_target"].float(), reduction="none", beta=0.05)
  return (error * mask).sum() / mask.sum().clamp_min(1)


def plan_nll(mu: torch.Tensor, log_std: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Laplace NLL |y-mu|/b + log(2b), b = exp(log_std); mu/y [B,33,15]."""
  log_b = log_std.clamp(-7.0, 7.0)
  nll = (y - mu).abs() * torch.exp(-log_b) + log_b
  w = PLAN_T_W.to(mu)[None, :, None] * PLAN_COL_W.to(mu)[None, None, :]
  return (nll * w).sum((1, 2)).mean() / w.sum()


def plan_l1(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  w = PLAN_T_W.to(mu)[None, :, None] * PLAN_COL_W.to(mu)[None, None, :]
  return ((mu - y).abs() * w).sum((1, 2)).mean() / w.sum()


def focal_heatmap(logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None, alpha: float = 2.0,
                  beta: float = 4.0) -> torch.Tensor:
  """CenterNet penalty-reduced focal loss. logits/gt [B,C,H,W]; mask [B,H,W] (1 = supervise)."""
  p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
  pos = gt.eq(1).float()
  neg = 1.0 - pos
  pos_l = torch.log(p) * (1 - p) ** alpha * pos
  neg_l = torch.log(1 - p) * p**alpha * (1 - gt) ** beta * neg
  if mask is not None:
    m = mask[:, None].float()
    pos_l, neg_l = pos_l * m, neg_l * m
  n_pos = pos.sum().clamp(min=1.0)
  return -(pos_l.sum() + neg_l.sum()) / n_pos


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
  """pred/gt [B,K,H,W], pos [B,H,W]."""
  m = pos[:, None].float()
  return ((pred - gt).abs() * m).sum() / (m.sum() * pred.shape[1]).clamp(min=1.0)


def bev_loss(out: dict, b: dict) -> dict[str, torch.Tensor]:
  heat = focal_heatmap(out["heat"], b["bev_heat"], b["bev_mask"])
  off = masked_l1(torch.sigmoid(out["off"]), b["bev_off"], b["bev_pos"])
  return {"bev_heat": heat, "bev_off": off}


def img_loss(out: dict, b: dict, cams=("road", "wide")) -> dict[str, torch.Tensor]:
  res = {}
  for c in cams:
    if f"{c}_heat" not in b:
      continue
    o = out[c]
    valid = b.get(f"{c}_valid")  # per-sample mask for FSOCO samples without that slot
    m = None if valid is None else valid[:, None, None].float().expand_as(b[f"{c}_pos"])
    res[f"img_{c}_heat"] = focal_heatmap(o["heat"], b[f"{c}_heat"], m)
    res[f"img_{c}_wh"] = 0.1 * masked_l1(o["wh"], b[f"{c}_wh"], b[f"{c}_pos"])
    res[f"img_{c}_off"] = masked_l1(torch.sigmoid(o["off"]), b[f"{c}_off"], b[f"{c}_pos"])
  return res


def tangent_motion_from_op_plan(mu, vx, vy, wheelbase):
  """Stateless desired steering and motion acceleration, before integral feedback or actuator limits."""
  mu, vx, vy, wheelbase = (x.float() for x in (mu, vx, vy, wheelbase))
  times = torch.as_tensor(T_IDXS, dtype=mu.dtype, device=mu.device)

  def interpolate(values, t):
    hi = int(np.searchsorted(T_IDXS, t))
    weight = (t - times[hi - 1]) / (times[hi] - times[hi - 1])
    return values[:, hi - 1] * (1 - weight) + values[:, hi] * weight

  # OP coordinates have opposite yaw and yaw-rate signs to the ego frame.
  speed_for_steer = vx.clamp_min(1.0)
  yaw = -interpolate(mu[:, :, 11], 0.2)
  yaw_rate = -mu[:, 0, 14]
  curvature = 2 * yaw / (speed_for_steer * 0.2) - yaw_rate / speed_for_steer
  steer = torch.atan(wheelbase * curvature)
  velocity, acceleration = mu[:, :, 3:5], mu[:, :, 6:8]
  knot_speed = torch.linalg.vector_norm(velocity, dim=-1)
  tangent = (velocity * acceleration).sum(dim=-1) / knot_speed.clamp_min(0.001)
  target_speed = interpolate(knot_speed, 0.3)
  target_accel = interpolate(tangent, 0.3)
  current_speed = torch.linalg.vector_norm(torch.stack((vx, vy), dim=-1), dim=-1)
  raw = 1.5 * (target_speed - current_speed) + 0.5 * target_accel
  return steer, raw


def op_tangent_actions_from_op_plan(mu, vx, vy, wheelbase, max_brake, max_accel):
  """FP32 differentiable equivalent of OpenpilotTangentFollower before mission braking."""
  steer, raw = tangent_motion_from_op_plan(mu, vx, vy, wheelbase)
  return steer, torch.maximum(torch.minimum(raw, max_accel.float()), -max_brake.float())


def native_relative_acceleration_loss(mu, target):
  """Match native endpoint acceleration residuals without assuming constant acceleration."""
  n = int(np.searchsorted(T_IDXS, 1.0, side='right'))
  times = torch.as_tensor(T_IDXS[:n], dtype=mu.dtype, device=mu.device)
  dt = times[1:] - times[:-1]

  def residual(plan):
    secant = (plan[:, 1:n, 3:5] - plan[:, :n-1, 3:5]) / dt[None, :, None]
    return torch.stack((plan[:, :n-1, 6:8] - secant, plan[:, 1:n, 6:8] - secant), dim=2)

  error = residual(mu) - residual(target).detach()
  return F.smooth_l1_loss(error, torch.zeros_like(error), beta=1.0)
