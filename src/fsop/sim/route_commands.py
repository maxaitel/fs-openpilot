from dataclasses import dataclass
from itertools import groupby

import numpy as np

DESIRE_LEN = 8
ROUTE_DESIRE = {"left": 1, "right": 2, "exit": 7}


@dataclass(frozen=True)
class PulseSettings:
  refresh_seconds: float = 1.0
  lead_seconds: float = 2.0
  minimum_lead_metres: float = 2.0
  release_metres: float = 2.0


class RouteCommands:
  def __init__(self, decisions: list[tuple[float, int]], fps: int):
    if fps < 1:
      raise ValueError("fps must be positive")
    self.decisions = decisions
    self.fps = fps
    self.settings = PulseSettings()
    self.index = 0
    self.last_pulse = -max(fps, 2)
    self.last_decision = -1
    self.last_frame = -1
    self.last_input = None
    self.cached = np.zeros(DESIRE_LEN, np.float32)

  @classmethod
  def for_mission(cls, track, mission, fps: int):
    if track.discipline != "skidpad":
      return cls([], fps)
    segments = track.generator["segments"]
    groups = [(name, len(list(items))) for name, items in groupby(track.route)]
    if [name for name, _ in groups] != ["entry", "right", "left", "exit"]:
      raise ValueError("invalid skidpad route")
    decisions = []
    previous_end = 0
    for name, count in groups:
      start, end = segments[name]
      if start != previous_end or end <= start or end > mission.n or (end - start) % count:
        raise ValueError("inconsistent skidpad segment ranges")
      previous_end = end
      if name != "entry":
        for i in range(count):
          index = start + i * ((end - start) // count)
          decisions.append((float(mission.s_arr[index]), ROUTE_DESIRE[name]))
    if previous_end != mission.n:
      raise ValueError("incomplete skidpad segment ranges")
    return cls(decisions, fps)

  def advance(self, *, frame: int, s: float, speed: float) -> np.ndarray:
    inputs = (s, speed)
    if frame == self.last_frame:
      if inputs != self.last_input:
        raise ValueError("different route-command inputs for the same frame")
      return self.cached.copy()
    if frame < self.last_frame:
      raise ValueError("route-command frame went backwards")
    self.last_frame, self.last_input = frame, inputs
    self.cached = np.zeros(DESIRE_LEN, np.float32)
    while self.index < len(self.decisions) and s > self.decisions[self.index][0] + self.settings.release_metres:
      self.index += 1
    if self.index < len(self.decisions):
      at_s, channel = self.decisions[self.index]
      lead = max(self.settings.minimum_lead_metres, self.settings.lead_seconds * max(speed, 0.0))
      due = self.index != self.last_decision or frame - self.last_pulse >= self.settings.refresh_seconds * self.fps
      if s >= at_s - lead and frame - self.last_pulse >= 2 and due:
        self.cached[channel] = 1.0
        self.last_pulse, self.last_decision = frame, self.index
    return self.cached.copy()
