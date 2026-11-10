import copy
from dataclasses import asdict
from math import pi
import unittest

from planner import (ConeClass, ConeDetection, CorridorPlanner, Motion, Observation,
                     RouteCommand, RouteEvent, warp_point)


def lane_observation(t, color=False, offset=0.25):
    cones = tuple(ConeDetection(x - 5 * t, side + offset, cls, .95)
                  for x in (6., 9., 12., 15., 18.)
                  for side, cls in ((1.5, ConeClass.ORANGE if color else ConeClass.BLUE),
                                    (-1.5, ConeClass.BIG_ORANGE if color else ConeClass.YELLOW)))
    return Observation(t, cones, Motion(5., 0., 0.))


class PlannerBehavior(unittest.TestCase):
    def test_rigid_ego_warp(self):
        self.assertEqual(warp_point((5., 2.), Motion(3., 1., 0.), .2), (4.4, 1.8))
        x, y = warp_point((2., 0.), Motion(0., 0., pi / 2), 1.)
        self.assertAlmostEqual(x, 0.)
        self.assertAlmostEqual(y, -2.)
        x, y = warp_point((5., 2.), Motion(2., 0., pi / 2), 1.)
        self.assertAlmostEqual(x, 2 - 4 / pi)
        self.assertAlmostEqual(y, -5 + 4 / pi)

    def test_offset_path_from_retained_observations(self):
        planner = CorridorPlanner()
        for i in range(7):
            result = planner.step(lane_observation(i * .2))
        self.assertEqual(result.status, "supported", result.diagnostics.reason)
        self.assertAlmostEqual(result.path[0][0], 0.)
        self.assertAlmostEqual(result.path[0][1], .25)
        self.assertGreater(result.control.steer_rad, 0.)
        self.assertLessEqual(result.control.target_speed_mps, 5.)
        self.assertGreater(result.support_length_m, result.control.lookahead_m)

    def test_orange_lane_and_same_color_rejection(self):
        planner = CorridorPlanner()
        for i in range(7):
            result = planner.step(lane_observation(i * .2, color=True))
        self.assertEqual(result.status, "supported", result.diagnostics.reason)
        self.assertTrue(all(g.color == "orange" for g in result.diagnostics.selected_gates))
        planner.reset()
        for i in range(7):
            obs = lane_observation(i * .2)
            obs = Observation(obs.timestamp, tuple(ConeDetection(c.x, c.y, ConeClass.BLUE, c.score)
                                                   for c in obs.cones), obs.motion)
            result = planner.step(obs)
        self.assertEqual(result.status, "unknown")
        self.assertTrue(result.control.brake)
        self.assertGreater(result.diagnostics.same_color_pairs_rejected, 0)

    def test_reset_dropout_and_unknown_brake(self):
        planner = CorridorPlanner()
        cold = planner.step(lane_observation(0.))
        self.assertEqual(cold.status, "unknown")
        self.assertEqual(cold.control.target_speed_mps, 0.)
        self.assertLess(cold.control.accel_mps2, 0.)
        for i in range(1, 7):
            result = planner.step(lane_observation(i * .2))
        self.assertEqual(result.status, "supported")
        for t in (1.4, 1.6):
            result = planner.step(Observation(t, (), Motion(5., 0., 0.)))
        self.assertEqual(result.status, "unknown")
        self.assertTrue(result.control.brake)
        self.assertLess(result.control.accel_mps2, 0.)
        planner.reset()
        result = planner.step(lane_observation(1.2))
        self.assertEqual(result.status, "unknown")
        result = planner.step(lane_observation(1.2))
        self.assertEqual(result.diagnostics.reset_reason, "nonmonotonic_time")
        result = planner.step(lane_observation(3.))
        self.assertEqual(result.diagnostics.reset_reason, "motion_gap")
        self.assertEqual(result.status, "unknown")

    def test_route_latch_deduplicates_history_and_rejects_future(self):
        planner = CorridorPlanner()
        right = RouteEvent(1., RouteCommand.RIGHT)
        for t, history in [(1., (right,)), (1.1, (right, right)), (1.2, ())]:
            result = planner.step(Observation(t, (), Motion(0., 0., 0.), history))
            self.assertEqual(result.diagnostics.route, RouteCommand.RIGHT)
            self.assertEqual(result.diagnostics.route_event_timestamp, 1.)
        left = RouteEvent(1.3, RouteCommand.LEFT)
        result = planner.step(Observation(1.3, (), Motion(0., 0., 0.), (right, left)))
        self.assertEqual(result.diagnostics.route, RouteCommand.LEFT)
        result = planner.step(Observation(1.4, (), Motion(0., 0., 0.), (RouteEvent(2., RouteCommand.EXIT),)))
        self.assertEqual(result.diagnostics.reason, "invalid_route_history")
        self.assertTrue(result.control.brake)
        planner.reset()
        result = planner.step(Observation(0., (), Motion(0., 0., 0.)))
        self.assertIsNone(result.diagnostics.route)

    def test_no_input_mutation_and_deterministic_replay(self):
        observations = [lane_observation(i * .2) for i in range(8)]
        frozen = copy.deepcopy(observations)
        a, b = CorridorPlanner(), CorridorPlanner()
        for obs in observations:
            self.assertEqual(asdict(a.step(obs)), asdict(b.step(obs)))
        self.assertEqual(observations, frozen)

    def test_far_outliers_do_not_define_support(self):
        planner = CorridorPlanner()
        far = (ConeDetection(100., 1.5, ConeClass.BLUE, 1.), ConeDetection(100., -1.5, ConeClass.YELLOW, 1.))
        result = planner.step(Observation(0., far, Motion(5., 0., 0.)))
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.diagnostics.gate_count, 0)
        self.assertEqual(result.diagnostics.rejected_detections, 2)
        self.assertEqual(result.control.target_speed_mps, 0.)


if __name__ == "__main__":
    unittest.main()
