import numpy as np
import pytest

from fsop.sim.env import FSEnv, SimCfg
from fsop.sim.referee import Referee
from fsop.sim.vehicle import VehicleParams, VehicleState
from fsop.track.fixed_layouts import skidpad


def _circle(name, angles=None, outside=0.0):
  angles = np.linspace(0, 2 * np.pi, 241)[1:] if angles is None else np.asarray(angles)
  radius = skidpad().generator["R_drive"]
  r = radius + outside * np.sin(angles / 2) ** 2
  sign = -1 if name == "right" else 1
  return np.c_[r * np.sin(angles), sign * (radius - r * np.cos(angles))]


def _path(order=("right", "right", "left", "left"), stop=True, stop_x=15.0):
  points = [np.c_[np.linspace(-1, 0, 5), np.zeros(5)]]
  points.extend(_circle(name) for name in order)
  points.append(np.c_[np.linspace(0, stop_x, 61)[1:], np.zeros(60)])
  if stop:
    points.append(np.tile([stop_x, 0.0], (14, 1)))
  return np.vstack(points)


def _referee(track=None):
  track = skidpad() if track is None else track
  return Referee(track, np.empty((0, 2)), [], VehicleParams())


def _drive(referee, points, mission_done=False):
  for i, point in enumerate(points):
    velocity = (point - points[i - 1]) / 0.05 if i else np.zeros(2)
    referee.update(VehicleState(*point, vx=float(np.linalg.norm(velocity)), t=i * 0.05), mission_done)
  return referee.summary()


def test_correct_physical_route_finishes_without_mission_progress():
  referee = _referee()
  report = _drive(referee, _path())
  assert report["pass"] and report["complete"]
  assert report["route"]["accepted_route"] == ["right", "right", "left", "left"]
  assert report["route"]["exited"] and report["route"]["stopped_in_zone"]
  assert len(report["route"]["crossings"]) == 5
  np.testing.assert_allclose(report["laps"], [12.0] * 4, atol=1e-8)
  assert report["raw_time"] == pytest.approx(12.0)
  assert report == referee.summary() == referee.summary()
  referee.finish("timeout")
  assert report == referee.summary()


def test_expected_order_comes_from_track_route():
  track = skidpad()
  order = ["left", "right", "left", "right"]
  track.route = ["entry", *order, "exit"]
  report = _drive(_referee(track), _path(order))
  assert report["pass"]
  assert report["route"]["expected_route"] == order


def test_gate_time_is_interpolated_between_observed_positions():
  report = _drive(_referee(), np.array([[-0.4, 0.0], [0.1, 0.0]]))
  assert report["route"]["crossings"] == pytest.approx([0.04])


def test_timing_uses_second_right_and_second_left_physical_laps():
  points = [np.array([[-0.5, 0.0], [0.0, 0.0]])]
  for name, samples in zip(("right", "right", "left", "left"), (180, 240, 120, 300), strict=True):
    points.append(_circle(name, np.linspace(0, 2 * np.pi, samples + 1)[1:]))
  points.append(np.array([[0.25, 0.0]]))
  report = _drive(_referee(), np.vstack(points))
  assert report["complete"]
  assert report["laps"] == pytest.approx([9.0, 12.0, 6.0, 15.0])
  assert report["raw_time"] == pytest.approx(13.5)


@pytest.mark.parametrize("order", [("right",) * 4, ("left", "left", "right", "right"),
                                   ("right", "left", "right", "left")])
def test_wrong_circles_cannot_pass_even_with_mission_done(order):
  report = _drive(_referee(), _path(order), mission_done=True)
  assert not report["pass"] and not report["complete"]
  assert report["result"] == "invalid_route"
  assert report["route"]["violation"].startswith("wrong_circle_")


def test_gate_touches_do_not_count_and_reverse_crossings_invalidate():
  referee = _referee()
  report = _drive(referee, np.array([[-0.5, 0], [0, 0], [-0.5, 0], [0, 0], [-0.5, 0]]))
  assert report["route"]["crossings"] == []
  assert report["n_laps"] == 0
  report = _drive(_referee(), np.array([[-0.5, 0], [0.5, 0], [-0.5, 0], [0.5, 0]]))
  assert report["route"]["reverse_crossings"] == 1
  assert report["route"]["violation"] == "reverse_gate_crossing"
  assert report["n_laps"] == 0 and not report["pass"]


@pytest.mark.parametrize("kind", ["outside", "backtrack", "shortcut", "reverse"])
def test_geometrically_invalid_circle_is_rejected(kind):
  angles = np.linspace(0, 2 * np.pi, 241)[1:]
  if kind == "backtrack":
    angles = np.r_[np.linspace(0, np.pi, 121)[1:], np.linspace(np.pi, np.pi - 0.5, 21)[1:],
                   np.linspace(np.pi - 0.5, 2 * np.pi, 141)[1:]]
  elif kind == "reverse":
    angles = -angles
  elif kind == "shortcut":
    angles = angles[np.r_[np.arange(60), np.arange(120, 240)]]
  points = np.vstack([[[-0.5, 0], [0, 0]], _circle("right", angles, outside=4 if kind == "outside" else 0),
                      [[0.25, 0]]])
  report = _drive(_referee(), points)
  assert not report["pass"] and report["n_laps"] == 0
  assert report["result"] == "invalid_route"


def test_small_gate_loop_cannot_substitute_for_a_circle():
  phi = np.linspace(0, 2 * np.pi, 81)[1:]
  points = np.vstack([[[-0.5, 0], [0, 0]], np.c_[np.sin(phi), np.cos(phi) - 1], [[0.25, 0]]])
  report = _drive(_referee(), points)
  assert not report["complete"] and report["n_laps"] == 0
  assert report["route"]["violation"] == "incomplete_or_outside_circle"


def test_four_circles_without_stop_remain_incomplete_for_pass():
  referee = _referee()
  report = _drive(referee, _path(stop=False))
  assert report["complete"] and not report["pass"]
  assert report["route"]["exited"] and not report["route"]["stopped_in_zone"]
  referee.finish("timeout")
  assert referee.summary()["result"] == "timeout"
  assert referee.summary()["route"]["termination"] == "timeout"


def test_stop_before_zone_does_not_pass():
  report = _drive(_referee(), _path(stop_x=5.0))
  assert report["complete"] and not report["pass"]
  assert not report["route"]["exited"] and not report["route"]["stopped_in_zone"]


def test_extra_circle_invalidates_completed_route():
  report = _drive(_referee(), _path(("right", "right", "left", "left", "left")))
  assert not report["pass"] and not report["complete"]
  assert report["result"] == "invalid_route"


@pytest.mark.parametrize("vx,vy", [(-1.0, 0.0), (0.0, 1.0)])
def test_speed_magnitude_prevents_false_standstill(vx, vy):
  referee = _referee()
  points = _path(stop=False)
  _drive(referee, points)
  for i in range(1, 16):
    referee.update(VehicleState(*points[-1], vx=vx, vy=vy, t=(len(points) - 1 + i) * 0.05), True)
  report = referee.summary()
  assert not report["pass"] and not report["route"]["stopped_in_zone"]


def test_mission_done_cannot_fabricate_a_skidpad_finish():
  report = _drive(_referee(), np.tile([-1.0, 0.0], (30, 1)), mission_done=True)
  assert not report["pass"] and not report["complete"]
  assert report["route"]["accepted_route"] == []


def test_missing_trace_interval_cannot_fabricate_a_circle():
  referee = _referee()
  referee.update(VehicleState(-1.0, 0.0, t=0.0))
  referee.update(VehicleState(0.1, 0.0, vx=8.0, t=1.0))
  assert referee.summary()["route"]["violation"] == "unverifiable_motion"
  assert not referee.summary()["pass"]


def test_expert_physically_completes_skidpad():
  cfg = SimCfg(render=False)
  cfg.noise.enabled = False
  env = FSEnv(skidpad(), cfg, seed=0)
  while not env.done:
    env.step(*env.expert_action())
  report = env.referee.summary()
  assert report["pass"] and report["doo"] == report["oc"] == 0
  assert report["route"]["accepted_route"] == ["right", "right", "left", "left"]
  assert report["route"]["stopped_in_zone"]
