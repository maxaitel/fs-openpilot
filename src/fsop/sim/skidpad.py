from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon

from fsop.sim.vehicle import VehicleState
from fsop.track.schema import Track


class SkidpadRoute:
  def __init__(self, track: Track):
    route = track.route
    if (len(route) < 3 or route[0] != "entry" or route[-1] != "exit"
        or any(c not in ("right", "left") for c in route[1:-1])
        or not track.timekeeping_line or not track.stop_zone):
      raise ValueError("skidpad needs an entry, circle sequence, exit, timing gate and stop zone")
    self.expected = list(route[1:-1])
    gate = np.asarray(track.timekeeping_line, dtype=float)
    self.origin = gate.mean(axis=0)
    self.forward = np.array([np.cos(track.start_pose[2]), np.sin(track.start_pose[2])])
    self.left = np.array([-self.forward[1], self.forward[0]])
    self.half_gate = float(np.linalg.norm(gate[1] - gate[0]) / 2)
    if abs(float((gate[1] - gate[0]) @ self.forward)) > 1e-6:
      raise ValueError("skidpad timing gate must be perpendicular to the entry lane")
    dc = float(track.generator["center_distance"])
    self.centers = self.origin + np.array([-1, 1])[:, None] * self.left * dc / 2
    self.r_min = float(track.generator["R_in"]) - 0.05
    self.r_max = float(track.generator["R_out"]) + 0.05
    self.stop_zone = Polygon(track.stop_zone).buffer(0.05)
    self.laps: list[dict] = []
    self.crossings: list[float] = []
    self.reverse_crossings = 0
    self.violation: str | None = None
    self.exited = False
    self.stopped_in_zone = False
    self.stopped_since: float | None = None
    self.termination = "running"
    self.prev: tuple[np.ndarray, float, float] | None = None
    self.side = 0
    self.armed = False
    self.winding = np.zeros(2)
    self.backtrack = np.zeros(2)
    self.quadrants = np.zeros(2, dtype=int)
    self.corridor = np.ones(2, dtype=bool)

  def _reject(self, reason: str, t: float, events: list) -> None:
    if self.violation is None:
      self.violation = reason
      events.append((t, "invalid_route", {"reason": reason}))

  def _integrate(self, a: np.ndarray, b: np.ndarray, t: float, events: list) -> None:
    if not self.crossings or len(self.laps) == len(self.expected):
      return
    va, vb = a - self.centers, b - self.centers
    delta = np.arctan2(va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0], (va * vb).sum(axis=1))
    if np.any(np.abs(delta) > np.pi / 4):
      self._reject("undersampled_turn", t, events)
    self.winding += delta
    directed = delta * [-1, 1]
    self.backtrack += np.maximum(-directed, 0)
    progress = self.winding * [-1, 1]
    self.quadrants = np.maximum(self.quadrants, np.floor((progress + 1e-6) / (np.pi / 2)).astype(int))
    radii = np.linalg.norm(np.stack([a, (a + b) / 2, b])[:, None, :] - self.centers, axis=2)
    self.corridor &= np.all((radii >= self.r_min) & (radii <= self.r_max), axis=0)

  def _cross(self, t: float, events: list) -> None:
    self.crossings.append(t)
    events.append((t, "line_cross", {"n": len(self.crossings)}))
    if len(self.crossings) == 1:
      return
    if len(self.laps) == len(self.expected):
      self._reject("extra_circle", t, events)
      return
    candidates = []
    for i, name in enumerate(("right", "left")):
      if (abs(self.winding[i] - (-1 if i == 0 else 1) * 2 * np.pi) < 0.1
          and abs(self.winding[1 - i]) < 0.1 and self.quadrants[i] >= 4
          and self.backtrack[i] < 0.2 and self.corridor[i]):
        candidates.append(name)
    if len(candidates) != 1:
      self._reject("incomplete_or_outside_circle", t, events)
      return
    circle = candidates[0]
    if circle != self.expected[len(self.laps)]:
      self._reject(f"wrong_circle_expected_{self.expected[len(self.laps)]}_got_{circle}", t, events)
      return
    lap = {"circle": circle, "start_time": self.crossings[-2], "finish_time": t,
           "duration": t - self.crossings[-2], "winding_right": float(self.winding[0]),
           "winding_left": float(self.winding[1])}
    self.laps.append(lap)
    events.append((t, "lap", dict(lap)))
    self.winding[:] = 0
    self.backtrack[:] = 0
    self.quadrants[:] = 0
    self.corridor[:] = True

  def advance(self, st: VehicleState) -> list[tuple[float, str, dict]]:
    events: list[tuple[float, str, dict]] = []
    if self.violation is not None or self.termination != "running":
      return events
    xy = np.array([st.x, st.y], dtype=float)
    speed = st.speed
    if not np.isfinite([*xy, st.t, speed]).all():
      self._reject("nonfinite_state", st.t, events)
      return events
    x, y = (xy - self.origin) @ self.forward, (xy - self.origin) @ self.left
    if self.prev is None:
      if x >= 0 or abs(y) > self.half_gate + 0.05:
        self._reject("invalid_entry", st.t, events)
      self.prev = (xy, st.t, speed)
      self.side = int(np.sign(x))
      self.armed = x < -0.1
      return events
    prev, prev_t, prev_speed = self.prev
    dt = st.t - prev_t
    distance = float(np.linalg.norm(xy - prev))
    if dt == 0 and distance == 0 and speed == prev_speed:
      return events
    if dt <= 0 or dt > 0.25 or distance > min(2.0, max(speed, prev_speed) * dt + 0.25):
      self._reject("unverifiable_motion", st.t, events)
      return events
    prev_x = float((prev - self.origin) @ self.forward)
    side = int(np.sign(x))
    crossing = None
    if side and self.side and side != self.side and abs(x - prev_x) > 1e-12:
      alpha = float(np.clip(-prev_x / (x - prev_x), 0, 1))
      point = prev + alpha * (xy - prev)
      if abs(float((point - self.origin) @ self.left)) <= self.half_gate:
        crossing = (point, prev_t + alpha * dt)
    if crossing is not None:
      point, t = crossing
      self._integrate(prev, point, t, events)
      if side < 0:
        self.reverse_crossings += 1
        self._reject("reverse_gate_crossing", t, events)
      elif not self.armed:
        self._reject("unarmed_gate_crossing", t, events)
      else:
        self._cross(t, events)
      self.armed = False
      self._integrate(point, xy, st.t, events)
    else:
      self._integrate(prev, xy, st.t, events)
    if side:
      self.side = side
    if x < -0.1:
      self.armed = True
    if not self.crossings and (x > 0 or abs(y) > self.half_gate + 0.05):
      self._reject("invalid_entry", st.t, events)
    if len(self.laps) == len(self.expected):
      if x < -0.05 or abs(y) > self.half_gate + 0.05:
        self._reject("outside_exit_lane", st.t, events)
      in_zone = self.stop_zone.covers(Point(xy))
      if in_zone and self.violation is None:
        self.exited = True
      if in_zone and max(speed, distance / dt) < 0.05 and self.violation is None:
        if self.stopped_since is None:
          self.stopped_since = st.t
        if st.t - self.stopped_since >= 0.5 and not self.stopped_in_zone:
          self.stopped_in_zone = True
          events.append((st.t, "stopped", {"in_stop_zone": True}))
      else:
        self.stopped_since = None
    self.prev = (xy, st.t, speed)
    return events

  def finish(self, reason: str) -> None:
    if self.termination == "running":
      self.termination = reason

  def report(self) -> dict:
    return {"expected_route": list(self.expected), "accepted_route": [p["circle"] for p in self.laps],
            "laps": [dict(p) for p in self.laps], "crossings": list(self.crossings),
            "complete": self.violation is None and len(self.laps) == len(self.expected),
            "exited": self.exited, "stopped_in_zone": self.stopped_in_zone,
            "valid": self.violation is None, "violation": self.violation,
            "reverse_crossings": self.reverse_crossings, "termination": self.termination}
