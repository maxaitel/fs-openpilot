"""Render a closed-loop run to an MP4: camera views with projected predictions + top-down track maps.

Layout (1280x768):
  left   : wide camera (top) and road camera (bottom), predicted plan (magenta, model scale),
           expert plan (green), detected cones (rings in class colour)
  right  : full track map with the driven trail (top), ego-centric 40 m map with GT cones (dots),
           detected cones (rings) and both plans (bottom); HUD with time, speed, laps, DOO / OC
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fsop.geometry.camera import CAMERA_MODELS
from fsop.track.schema import CONE_CLASSES, CONE_VIZ_RGB

PRED = (230, 60, 200)
GT = (40, 220, 80)
CLS_RGB = [CONE_VIZ_RGB[c] for c in CONE_CLASSES]


def _bgr(rgb):
  return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _project(pts_ego: np.ndarray, T_ego_cam: np.ndarray, cam: str):
  T = np.linalg.inv(T_ego_cam)
  pc = pts_ego @ T[:3, :3].T + T[:3, 3]
  return CAMERA_MODELS[cam].project_cam(pc)


def _draw_path(img, plan, T_ego_cam, cam, color, scale, std=None, thick=2):
  pts = np.c_[plan[:, 0], plan[:, 1], np.zeros(len(plan))]
  m = plan[:, 0] > 1.0
  if std is not None:
    h = np.arctan2(np.gradient(plan[:, 1]), np.gradient(plan[:, 0]) + 1e-6)
    n = np.stack([-np.sin(h), np.cos(h), 0 * h], 1) * std[:, 1:2]
    for side in (pts - n, pts + n):
      uv, z = _project(side[m], T_ego_cam, cam)
      ok = z > 0.5
      cv2.polylines(img, [(uv[ok] * scale).astype(np.int32)], False, _bgr(color), 1, cv2.LINE_AA)
  uv, z = _project(pts[m], T_ego_cam, cam)
  ok = z > 0.5
  cv2.polylines(img, [(uv[ok] * scale).astype(np.int32)], False, _bgr(color), thick, cv2.LINE_AA)


def _draw_cones_cam(img, xy, cls, T_ego_cam, cam, scale):
  if xy is None or len(xy) == 0:
    return
  uv, z = _project(np.c_[xy, np.zeros(len(xy))], T_ego_cam, cam)
  for (u, v), zz, c in zip(uv, z, cls, strict=True):
    if zz > 0.5 and 0 <= c < 4:
      r = int(np.clip(60 / zz, 3, 14) * scale)
      cv2.circle(img, (int(u * scale), int(v * scale)), r, _bgr(CLS_RGB[c]), 2, cv2.LINE_AA)


class Map:
  def __init__(self, track, cones_xy, cone_colors, size=(512, 384), margin=8):
    self.track, self.size = track, size
    lo, hi = track.centerline.min(0) - 6, track.centerline.max(0) + 6
    self.scale = min((size[0] - 2 * margin) / (hi[0] - lo[0]), (size[1] - 2 * margin) / (hi[1] - lo[1]))
    self.off = np.array([margin, margin]) - lo * self.scale + (np.array(size) - 2 * margin - (hi - lo) * self.scale) / 2
    self.base = np.full((size[1], size[0], 3), 245, np.uint8)
    left, right = track.boundaries()
    for b in (left, right):
      cv2.polylines(self.base, [self.px(b)], track.closed, (190, 190, 190), 1, cv2.LINE_AA)
    for (x, y), c in zip(cones_xy, cone_colors, strict=True):
      cv2.circle(self.base, tuple(self.px(np.array([[x, y]]))[0]), 2, _bgr(CONE_VIZ_RGB[c]), -1, cv2.LINE_AA)
    self.trail = []

  def px(self, xy):
    p = xy * self.scale + self.off
    return np.c_[p[:, 0], self.size[1] - p[:, 1]].astype(np.int32)

  def draw(self, x, y, yaw, plan_world, trail=None):
    img = self.base.copy()
    if trail is None:
      self.trail.append((x, y))
      trail = self.trail
    if len(trail) > 1:
      cv2.polylines(img, [self.px(np.asarray(trail))], False, (60, 60, 200), 2, cv2.LINE_AA)
    if plan_world is not None:
      cv2.polylines(img, [self.px(plan_world)], False, _bgr(PRED), 2, cv2.LINE_AA)
    c = self.px(np.array([[x, y]]))[0]
    tip = self.px(np.array([[x + 3 * np.cos(yaw), y + 3 * np.sin(yaw)]]))[0]
    cv2.arrowedLine(img, tuple(c), tuple(tip), (0, 0, 220), 2, tipLength=0.5)
    return img


def ego_map(gt_xy, gt_cls, det_xy, det_cls, plan, gt_plan, size=(512, 384), rng=40.0):
  img = np.full((size[1], size[0], 3), 30, np.uint8)
  s = size[1] / rng  # px per metre, x forward = up
  def px(p):  # noqa: E306
    return np.c_[size[0] / 2 - p[:, 1] * s, size[1] - 20 - p[:, 0] * s].astype(np.int32)
  for d in range(10, int(rng) + 1, 10):
    cv2.circle(img, tuple(px(np.array([[0.0, 0.0]]))[0]), int(d * s), (60, 60, 60), 1, cv2.LINE_AA)
  for p, c in zip(gt_xy, gt_cls, strict=True):
    if 0 <= c < 4:
      cv2.circle(img, tuple(px(p[None])[0]), 3, _bgr(CLS_RGB[c]), -1, cv2.LINE_AA)
  for p, c in zip(det_xy, det_cls, strict=True):
    if 0 <= c < 4:
      cv2.circle(img, tuple(px(p[None])[0]), 7, _bgr(CLS_RGB[c]), 2, cv2.LINE_AA)
  if gt_plan is not None:
    cv2.polylines(img, [px(gt_plan[:, :2])], False, _bgr(GT), 2, cv2.LINE_AA)
  if plan is not None:
    cv2.polylines(img, [px(plan[:, :2])], False, _bgr(PRED), 2, cv2.LINE_AA)
  car = px(np.array([[1.6, 0.7], [1.6, -0.7], [-1.3, -0.7], [-1.3, 0.7]]))
  cv2.fillPoly(img, [car], (40, 40, 200))
  cv2.putText(img, "ego view 40 m: dots = simulator GT, rings = detected", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
              (220, 220, 220), 1, cv2.LINE_AA)
  return img


def compose_frame(track, cones_xy, cone_colors, images, state, T_ego_cam, prediction, reference, trail, hud):
  """Compose BGR diagnostic frame from arrays; no input or history is mutated."""
  from fsop.data.targets import visible_cones, world_to_ego
  from fsop.track.schema import CONE_CLASS_INDEX

  plan = np.asarray(prediction["plan"])
  if plan.ndim != 2 or plan.shape[0] < 2 or plan.shape[1] < 2:
    raise ValueError("prediction.plan must be [N, >=2]")
  std = prediction.get("plan_std")
  det_xy = prediction.get("cones_xy")
  det_cls = prediction.get("cones_cls")
  if (det_xy is None) != (det_cls is None):
    raise ValueError("cone positions and classes must both be present or absent")
  frame = np.zeros((768, 1280, 3), np.uint8)
  for row, cam in enumerate(("wide", "road")):
    im = cv2.resize(images[cam][..., ::-1], (768, 384), interpolation=cv2.INTER_LINEAR)
    sc = 768 / 512
    if reference is not None:
      _draw_path(im, reference, T_ego_cam, cam, GT, sc)
    _draw_path(im, plan, T_ego_cam, cam, PRED, sc, std, 3)
    _draw_cones_cam(im, det_xy, det_cls, T_ego_cam, cam, sc)
    cv2.putText(im, f"{cam} camera", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    frame[row * 384:(row + 1) * 384, :768] = im
  x, y, yaw = state["x"], state["y"], state["yaw"]
  c, sn = np.cos(yaw), np.sin(yaw)
  pw = np.c_[x + c * plan[:, 0] - sn * plan[:, 1], y + sn * plan[:, 0] + c * plan[:, 1]]
  frame[:384, 768:] = Map(track, cones_xy, cone_colors).draw(x, y, yaw, pw, trail=trail)
  cv2.putText(frame, "track map: simulator GT cone positions", (776, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
              (50, 50, 50), 1, cv2.LINE_AA)
  ge = world_to_ego(np.asarray(cones_xy), x, y, yaw)
  vis = visible_cones(ge, T_ego_cam) & (np.linalg.norm(ge, axis=1) < 45)
  cls = np.array([CONE_CLASS_INDEX.get(c, -1) for c in cone_colors])
  frame[384:, 768:] = ego_map(ge[vis], cls[vis], np.empty((0, 2)) if det_xy is None else det_xy,
                              np.empty(0, int) if det_cls is None else det_cls, plan, reference)
  cv2.rectangle(frame, (768, 0), (1280, 30), (255, 255, 255), -1)
  cv2.putText(frame, hud, (776, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
  status = "unavailable" if det_xy is None else f"{len(det_xy)}"
  scale_label = " +/- model scale" if std is not None else " (scale unavailable)"
  cv2.putText(frame, f"magenta: model{scale_label}  green: oracle reference (custom bicycle)  detections: {status}",
              (10, 758), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
  return frame


def render_run(ckpt, track, out: str | Path, seconds: float = 60.0, dr: str = "light", seed: int = 0,
               fps: int = 20, every: int = 1, profile: str | None = None, venue_prob: float = 0.0) -> dict:
  from fsop.eval.closed_loop import ModelPolicy
  from fsop.sim.env import FSEnv, SimCfg

  cfg = SimCfg(dr=dr, max_time=seconds, venue_prob=venue_prob)
  cfg.noise.enabled = False
  from fsop.profile import load_profile

  env = FSEnv(track, cfg, seed=seed, profile=load_profile(profile))
  policy = ModelPolicy.from_checkpoint(ckpt)
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  W, H = 1280, 768
  vw = None
  for fourcc in ("avc1", "mp4v"):
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), fps / every, (W, H))
    if vw.isOpened():
      break
  trail = []
  T_ec = env.T_ego_cam()
  obs = env.observe()
  n = 0
  while not env.done:
    steer, accel, info = policy(obs, env)
    if obs.frame % every == 0:
      st = obs.state
      trail.append((st.x, st.y))
      r = env.referee
      hud = (f"t {obs.t:5.1f}s  v {st.vx * 3.6:4.0f} km/h  laps {len(r.lap_times())}/{track.laps}  "
             f"DOO {r.n_doo}  OC {r.n_oc}")
      frame = compose_frame(track, env.cones_xy, env.cone_colors, obs.images, vars(st), T_ec, info,
                            env.label(), trail, hud)
      vw.write(frame)
      n += 1
    obs = env.step(steer, accel, add_noise=False)
  vw.release()
  return {"frames": n, "path": str(out), **env.referee.summary()}


def mp4_to_gif(src: str | Path, out: str | Path, start_s: float = 0.0, duration_s: float = 12.0, fps: float = 10.0,
               width: int = 640, colors: int = 128, quality: int = 70) -> Path:
  """Convert a clip of an MP4 to an animation for READMEs: an optimized GIF (shared palette, no dithering), or an
  animated WebP when ``out`` ends in ``.webp`` (``quality``)."""
  from PIL import Image

  cap = cv2.VideoCapture(str(src))
  src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
  step = max(int(round(src_fps / fps)), 1)
  first, last = int(start_s * src_fps), int((start_s + duration_s) * src_fps)
  frames = []
  cap.set(cv2.CAP_PROP_POS_FRAMES, first)
  for i in range(first, last):
    ok, fr = cap.read()
    if not ok:
      break
    if (i - first) % step:
      continue
    h = int(round(fr.shape[0] * width / fr.shape[1]))
    fr = cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA)
    frames.append(Image.fromarray(fr[..., ::-1]))
  cap.release()
  if not frames:
    raise ValueError(f"no frames read from {src}")
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  if out.suffix.lower() == ".webp":  # animated WebP: full colour, ~3x smaller than a GIF at the same size
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(round(1000 * step / src_fps)), loop=0,
                   quality=quality, method=6)
    return out
  # shared palette from a mosaic of frames across the clip (keeps small-but-important colours like cones)
  picks = frames[:: max(len(frames) // 6, 1)][:6]
  mosaic = Image.new("RGB", (picks[0].width, picks[0].height * len(picks)))
  for i, f in enumerate(picks):
    mosaic.paste(f, (0, i * f.height))
  palette = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
  q = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
  q[0].save(out, save_all=True, append_images=q[1:], duration=int(round(1000 * step / src_fps)), loop=0, optimize=True)
  return out
