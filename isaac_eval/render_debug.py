"""Render recorded policy calls offline. No checkpoint or inference is used."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from fsop.sim.expert import Expert, ExpertParams, Mission, plan_from_rollout, rollout
from fsop.sim.vehicle import VehicleParams, VehicleState
from fsop.track.schema import Track
from fsop.viz.video import CLS_RGB, _bgr, compose_frame
from isaac_eval.evidence import read_evidence, verified_summary


class OfflineOracle:
  def __init__(self, metadata):
    self.track = Track.from_dict(metadata["track"])
    self.vehicle = VehicleParams(**metadata["vehicle_params"])
    self.mission = Mission(self.track, ExpertParams(**metadata["expert_params"]))
    self.expert = Expert(self.mission, self.vehicle)
    self.dt = metadata["label_dt"]

  def reference(self, record):
    state = VehicleState(**record["state"])
    progress = record["progress"]
    trajectory = rollout(self.mission, self.expert, self.vehicle, state, progress["s"], progress["hint"], dt=self.dt)
    return plan_from_rollout(trajectory, dt=self.dt)


def command_label(value):
  if value is None:
    return "unavailable"
  values = np.asarray(value).reshape(-1)
  channels = np.flatnonzero(values)
  if not len(channels):
    return "none"
  names = {1: "left", 2: "right", 7: "exit"}
  return ", ".join(f"{names.get(int(i), 'channel')}[{i}]={values[i]:.2g}" for i in channels)


def terminal_frame(summary, count):
  frame = np.full((768, 1280, 3), 26, np.uint8)
  state = summary.get("final_state", {})
  lines = [
    "TERMINAL STATE - no current model prediction",
    f"result: {summary.get('result', 'unavailable')}  pass: {summary.get('pass', 'unavailable')}  "
    f"complete: {summary.get('complete', 'unavailable')}",
    f"laps: {summary.get('n_laps', 'unavailable')}  legacy DOO: {summary.get('doo', 'unavailable')}  "
    f"off-course: {summary.get('oc', 'unavailable')}",
    f"PhysX cone contacts: {summary.get('physical_cone_hits', 'unavailable')}",
    f"final time: {state.get('t', 'unavailable')} s  speed: {state.get('vx', 'unavailable')} m/s",
    f"recorded pre-step inference frames: {count}",
  ]
  for i, line in enumerate(lines):
    cv2.putText(frame, str(line), (48, 120 + i * 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (245, 245, 245), 2, cv2.LINE_AA)
  return frame


def research_frame(record, entry, controller=None):
  """Raw same-call image-head boxes; projection and ground-truth are not used."""
  frame = np.full((768, 1280, 3), 24, np.uint8)
  outputs = record.get("image_head")
  for i, cam in enumerate(("wide", "road")):
    im = cv2.resize(record["images"][cam][..., ::-1], (640, 320))
    if outputs is not None:
      import torch

      from fsop.eval.det_metrics import decode_img
      out = outputs[cam]
      boxes, classes, scores = decode_img(*(torch.from_numpy(out[k]) for k in ("heat", "wh", "off")), thresh=0.25)[0]
      for box, cls, score in zip(boxes, classes, scores, strict=True):
        x1, y1, x2, y2 = (box * 1.25).astype(int)
        color = _bgr(CLS_RGB[cls])
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 1)
        cv2.putText(im, f"{int(cls)} {score:.2f}", (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.putText(im, f"{cam}: raw same-call image head" if outputs is not None else f"{cam}: image head unavailable",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    frame[:320, i * 640:(i + 1) * 640] = im
  counters = record.get("counters", {})
  commands = record.get("commands", {})
  diagnostics = record.get("diagnostics", {})
  lines = [
    f"frame {entry['frame']}  RGB input SHA256 {entry['input_sha256']}",
    f"route mode: {commands.get('mode', 'unavailable')}  "
    f"requested: {command_label(commands.get('requested'))}  applied: {command_label(commands.get('applied'))}",
    f"action steer / accel: {record['action']}",
    f"PhysX contacts: {counters.get('physical_contacts', 'unavailable')}  "
    f"legacy DOO: {counters.get('legacy_doo', 'unavailable')}  "
    f"off-course: {counters.get('off_course', 'unavailable')}",
    f"policy with capture: {diagnostics.get('policy_with_capture_ms', 'unavailable')} ms  "
    f"render: {diagnostics.get('render_ms', 'unavailable')} ms",
    "Boxes are image-head predictions before ground projection; model scale is not calibrated uncertainty.",
  ]
  if controller is not None:
    lines.append(f"controller: {controller}")
  for i, line in enumerate(lines):
    cv2.putText(frame, str(line)[:150], (12, 350 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (230, 230, 230), 1, cv2.LINE_AA)
  return frame


def _writer(path, fps):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  for codec in ("avc1", "mp4v"):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (1280, 768))
    if writer.isOpened():
      return writer
    writer.release()
  raise RuntimeError(f"could not open video writer: {path}")



def export_rrd(directory, output):
  import rerun as rr

  from fsop.track.schema import CONE_VIZ_RGB, ConeColor
  from fsop.viz import rr_log

  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  rr_log.init(app="fsop-recorded-isaac", spawn=False, save=str(output))
  oracle = None
  count = 0
  for metadata, record, entry in read_evidence(directory):
    if oracle is None:
      if metadata["fps"] != 20:
        raise ValueError("evidence must use the recorded 20 Hz cadence")
      oracle = OfflineOracle(metadata)
      rr_log.log_track(oracle.track, np.asarray(metadata["initial_cones_xyz"])[:, :2],
                       metadata["cone_colors"], path="world/track_initial_gt")
    state = record["state"]
    rr.set_time("frame", sequence=entry["frame"])
    rr.set_time("sim_time", duration=state["t"])
    yaw = state["yaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    T_world_ego = np.eye(4)
    T_world_ego[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T_world_ego[:3, 3] = [state["x"], state["y"], 0]
    rr_log.log_ego(T_world_ego, oracle.vehicle)
    rr_log.log_cameras(np.asarray(record["T_plan_ego_camera"]))
    rr_log.log_images(record["images"], jpeg=None)
    prediction = record["prediction"]
    rr_log.log_plan("pred", prediction["plan"], (230, 60, 200), prediction.get("plan_std"))
    rr_log.log_plan("oracle_reference_custom_bicycle", oracle.reference(record), (40, 220, 80))
    if prediction.get("cones_xy") is not None and prediction.get("cones_cls") is not None:
      rr_log.log_cones_ego("cones_pred", prediction["cones_xy"], prediction["cones_cls"],
                           prediction.get("cones_score"))
    else:
      rr.log("world/ego/cones_pred", rr.Clear(recursive=False))
    xyz = np.asarray(record["actual_cones_xyz"])
    for color in ConeColor:
      mask = np.array([value == color for value in metadata["cone_colors"]])
      if mask.any():
        rr.log(f"world/track_actual_gt/cones/{color.value}",
               rr.Points3D(xyz[mask], colors=CONE_VIZ_RGB[color], radii=0.12))
    rr_log.log_scalars("metrics", {"speed_mps": state["vx"], "progress_m": record["progress"]["s"],
                                   "steer_rad": state["steer"], "yaw_rate_rad_s": state["r"],
                                   "lat_m": record["progress"].get("lat"),
                                   **{k: v for k, v in record.get("counters", {}).items()
                                      if isinstance(v, (int, float, np.number))}})
    rr_log.log_events([SimpleNamespace(**event) for event in record.get("events", [])])
    count += 1
  if oracle is None:
    raise ValueError("no evidence frames")
  summary = verified_summary(directory)
  if summary is not None:
    state = summary.get("final_state", {})
    rr.set_time("frame", sequence=count)
    if "t" in state:
      rr.set_time("sim_time", duration=state["t"])
    rr.log("world/ego/plan", rr.Clear(recursive=True))
    rr.log("world/ego/cones_pred", rr.Clear(recursive=True))
    rr.log("world/ego/cam_wide", rr.Clear(recursive=True))
    rr.log("world/ego/cam_road", rr.Clear(recursive=True))
    if all(key in state for key in ("x", "y", "yaw")):
      c, s = np.cos(state["yaw"]), np.sin(state["yaw"])
      transform = np.eye(4)
      transform[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
      transform[:3, 3] = [state["x"], state["y"], 0]
      rr_log.log_ego(transform, oracle.vehicle)
    rr_log.log_events([SimpleNamespace(**event) for event in summary.get("pending_events", [])])
    rr.log("race/terminal", rr.TextLog("Terminal state. No current model prediction."))
  rr.get_global_data_recording().flush()
  return count

def render(directory, output, research=None, fps=20):
  main = _writer(output, fps)
  extra = _writer(research, fps) if research else None
  trail = []
  count = 0
  terminal_count = 0
  oracle = None
  try:
    for metadata, record, entry in read_evidence(directory):
      if oracle is None:
        if metadata["fps"] != 20 or fps != metadata["fps"]:
          raise ValueError("evidence and output must both use the recorded 20 Hz cadence")
        oracle = OfflineOracle(metadata)
      state = record["state"]
      trail.append((state["x"], state["y"]))
      counters = record.get("counters", {})
      hud = (f"t {state['t']:.1f}s v {state['vx'] * 3.6:.0f} km/h "
             f"laps {counters.get('laps', '?')}/{counters.get('total_laps', '?')} "
             f"DOO {counters.get('legacy_doo', '?')} OC {counters.get('off_course', '?')}")
      if "controller_contract" in metadata:
        hud += f" controller {metadata['controller_contract']['mode']}"
      cones = np.asarray(record["actual_cones_xyz"])[:, :2]
      frame = compose_frame(oracle.track, cones, metadata["cone_colors"], record["images"], state,
                            np.asarray(record["T_plan_ego_camera"]), record["prediction"],
                            oracle.reference(record), trail, hud)
      main.write(frame)
      if extra is not None:
        extra.write(research_frame(record, entry, metadata.get("controller_contract", {}).get("mode")))
      count += 1
    if not count:
      raise ValueError("no evidence frames")
    summary = verified_summary(directory)
    if summary is not None:
      card = terminal_frame(summary, count)
      terminal_count = int(round(2 * fps))
      for _ in range(terminal_count):
        main.write(card)
        if extra is not None:
          extra.write(card)
  finally:
    main.release()
    if extra is not None:
      extra.release()
  return {"inference_frames": count, "terminal_frames": terminal_count}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directory")
  parser.add_argument("output")
  parser.add_argument("--research")
  parser.add_argument("--rrd", help="write offline Rerun recording")
  parser.add_argument("--fps", type=float, default=20)
  args = parser.parse_args()
  counts = render(args.directory, args.output, args.research, args.fps)
  if args.rrd:
    counts["rrd_frames"] = export_rrd(args.directory, args.rrd)
  print(json.dumps(counts))


if __name__ == "__main__":
  main()
