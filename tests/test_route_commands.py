from types import SimpleNamespace

import numpy as np
import pytest

from fsop.model.temporal import TemporalState
from fsop.sim.env import FSEnv, Obs, SimCfg
from fsop.sim.expert import ExpertParams, Mission
from fsop.sim.recorder import record_episode
from fsop.sim.route_commands import RouteCommands
from fsop.track.fixed_layouts import acceleration, skidpad


def make_schedule(fps=20):
  track = skidpad()
  return RouteCommands.for_mission(track, Mission(track, ExpertParams()), fps)


def test_route_choices_follow_segment_metadata():
  track = skidpad()
  mission = Mission(track, ExpertParams())
  schedule = RouteCommands.for_mission(track, mission, 20)
  segments = track.generator["segments"]
  r0, r1 = segments["right"]
  l0, l1 = segments["left"]
  indices = [r0, (r0 + r1) // 2, l0, (l0 + l1) // 2, segments["exit"][0]]
  frame = 0
  for index, channel in zip(indices, [2, 2, 1, 1, 7], strict=True):
    s = float(mission.s_arr[index])
    pulse = schedule.advance(frame=frame, s=s - 1.0, speed=0)
    assert np.flatnonzero(pulse).tolist() == [channel]
    assert not schedule.advance(frame=frame + 1, s=s + 3.0, speed=0).any()
    frame += 3


@pytest.mark.parametrize("fps", [10, 20, 30])
def test_stalled_approach_repeats_isolated_pulses_at_simulator_rate(fps):
  schedule = make_schedule(fps)
  at_s = schedule.decisions[0][0]
  pulses = [schedule.advance(frame=f, s=at_s - 1.0, speed=0) for f in range(7 * fps)]
  assert [i for i, pulse in enumerate(pulses) if pulse[2]] == list(range(0, 7 * fps, fps))
  assert all(not np.any(pulse[[0, 1, 3, 4, 5, 6, 7]]) for pulse in pulses)


def test_observe_does_not_consume_or_share_command():
  env = FSEnv(skidpad(), SimCfg(render=False), seed=0)
  at_s = env.route_commands.decisions[0][0]
  env.s = at_s - 1.0
  env.frame += 1
  env.frame_desire = env.route_commands.advance(frame=env.frame, s=env.s, speed=env.state.vx)
  first = env.observe(render=False)
  assert first.desire[2] == 1
  first.desire[:] = 0
  assert env.observe(render=False).desire[2] == 1
  assert env.observe(render=False).desire[2] == 1


def test_frame_cache_and_non_skidpad():
  schedule = make_schedule()
  s = schedule.decisions[0][0]
  assert schedule.advance(frame=0, s=s, speed=0)[2] == 1
  assert schedule.advance(frame=0, s=s, speed=0)[2] == 1
  with pytest.raises(ValueError):
    schedule.advance(frame=0, s=s + 1, speed=0)
  with pytest.raises(ValueError):
    schedule.advance(frame=-1, s=s, speed=0)
  track = acceleration()
  other = RouteCommands.for_mission(track, Mission(track, ExpertParams()), 20)
  assert all(not other.advance(frame=f, s=f, speed=5).any() for f in range(30))


def test_recorded_and_live_histories_match_without_mutating_observations(monkeypatch, tmp_path):
  import fsop.sim.recorder as recorder

  class Writer:
    def __init__(self, *args, **kwargs):
      self.desires = []
      writers.append(self)

    def add(self, obs, plan):
      self.desires.append(obs.desire.copy())

    def close(self, summary):
      return summary

  class Environment:
    def __init__(self):
      self.cfg = SimpleNamespace(fps=20)
      self.max_time = 8.0
      self.referee = SimpleNamespace(summary=lambda: {})
      self.schedule = make_schedule()
      self.milestones = [s for s, _ in self.schedule.decisions]
      self.frame = 0
      self.live = []
      self.raw = []
      self.ts = TemporalState(cams=())
      self.obs = self.make_obs()

    @property
    def done(self):
      return self.frame >= 160

    def make_obs(self):
      index = min(max((self.frame - 3) // 25, 0), 4)
      s = self.milestones[index] - (3 if self.frame < 3 else 1)
      desire = self.schedule.advance(frame=self.frame, s=s, speed=0)
      history = self.ts.push_desire(desire)
      self.live.append(history.copy())
      self.raw.append(desire.copy())
      return Obs(frame=self.frame, t=self.frame / 20, state=None, s=s, lat=0, images={},
                 T_world_ego=np.eye(4), desire=desire, events=[], noise=0)

    def observe(self):
      return self.obs

    def label(self):
      return None

    def expert_action(self):
      return 0, 0

    def step(self, *args, **kwargs):
      np.testing.assert_array_equal(self.obs.desire, self.raw[-1])
      self.frame += 1
      self.obs = self.make_obs()
      return self.obs

  writers = []
  monkeypatch.setattr(recorder, "EpisodeWriter", Writer)
  env = Environment()
  record_episode(env, tmp_path, "episode")
  recorded = writers[0].desires
  assert set(np.flatnonzero(np.max(recorded, axis=0))) == {1, 2, 7}
  for k in range(len(recorded)):
    expected = np.zeros((25, 8), np.float32)
    slots = np.asarray(recorded[max(0, k - 24):k + 1])
    expected[-len(slots):] = slots
    np.testing.assert_array_equal(expected, env.live[k * 4])


def test_timeout_result_does_not_override_referee_finish(monkeypatch):
  env = FSEnv(acceleration(), SimCfg(render=False, max_time=0.01), seed=0)
  env.step(0, 0, add_noise=False, render=False)
  assert env.done and env.referee.summary()["result"] == "timeout"
  env = FSEnv(acceleration(), SimCfg(render=False, max_time=0.01), seed=0)

  def finish(*args):
    env.referee.done = True
    env.referee.result = "finished"
    return []

  monkeypatch.setattr(env.referee, "update", finish)
  env.step(0, 0, add_noise=False, render=False)
  assert env.referee.summary()["result"] == "finished"


def test_recording_saves_command_settings(tmp_path, monkeypatch):
  env = FSEnv(skidpad(), SimCfg(render=False, cameras=()), seed=0)
  monkeypatch.setattr(env, "label", lambda: np.zeros((33, 15), np.float32))
  meta = record_episode(env, tmp_path, "episode", max_seconds=0.21)
  assert meta["route_commands"] == {
    "schema": "next_junction_pulse_v1", "refresh_seconds": 1.0, "lead_seconds": 2.0,
    "minimum_lead_metres": 2.0, "release_metres": 2.0,
  }
  assert np.load(tmp_path / "episode" / "desire.npy").shape == (2, 8)
