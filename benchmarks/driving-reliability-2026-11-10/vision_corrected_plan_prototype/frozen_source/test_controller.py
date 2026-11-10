import json
from pathlib import Path
import unittest
import numpy as np
from controller import CorrectedPlan, Observation, observed, warp

P = json.loads(Path(__file__).with_name('config.json').read_text())['planner']
PATH = tuple((float(x), 0.) for x in np.linspace(0, 12, 49))

def obs(t=0., cones=((6., 2., 0, .9),), motion=(0., 0., 0.)):
    return Observation(t, PATH, cones, motion, 'skidpad')

class Behavior(unittest.TestCase):
    def test_warp_integrates_measured_body_motion(self):
        np.testing.assert_allclose(warp((6, 2), (2, 1, 0), .5), (5, 1.5), atol=1e-12)
        np.testing.assert_allclose(warp((2, 0), (0, 0, np.pi / 2), 1), (0, -2), atol=1e-12)
        np.testing.assert_allclose(warp((2, 0), (1, 0, np.pi / 2), 1), (-2 / np.pi, -2 + 2 / np.pi), atol=1e-12)

    def test_offset_is_not_origin_anchored_and_color_has_correct_sign(self):
        left = CorrectedPlan(P).step(obs())
        right = CorrectedPlan(P).step(obs(cones=((6., -2., 1, .9),)))
        self.assertAlmostEqual(left['variants']['corrected_pp']['path'][0][1], .5)
        self.assertGreater(left['variants']['corrected_pp']['control']['steer_rad'], 0)
        self.assertLess(right['variants']['corrected_pp']['control']['steer_rad'], 0)

    def test_dropout_carries_then_expires_and_gap_resets(self):
        planner = CorrectedPlan(P)
        planner.step(obs(motion=(2., 0., 0.)))
        carried = planner.step(obs(.2, (), (2., 0., 0.)))
        self.assertAlmostEqual(carried['tracks'][0]['xy'][0], 5.6)
        for t in np.arange(.4, 1.61, .2):
            result = planner.step(obs(float(t), (), (2., 0., 0.)))
        self.assertEqual(result['status'], 'unknown')
        planner.step(obs(2.))
        result = planner.step(obs(3., ()))
        self.assertTrue(result['reset'])
        self.assertEqual(result['tracks'], [])

    def test_conflicting_color_becomes_unknown_obstacle(self):
        planner = CorrectedPlan(P)
        planner.step(obs())
        result = planner.step(obs(.05, ((6., 2., 1, .9),)))
        self.assertEqual(result['tracks'][0]['cone_class'], 4)
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(result['obstacles_xy'])

    def test_unknown_brakes_at_zero_and_moving_speed(self):
        for speed in (0., 5.):
            result = CorrectedPlan(P).step(obs(cones=(), motion=(speed, 0., 0.)))
            c = result['variants']['corrected_pp']['control']
            self.assertEqual(result['status'], 'unknown')
            self.assertLess(c['accel_mps2'], 0)
            self.assertEqual(c['target_speed_mps'], 0)

if __name__ == '__main__':
    unittest.main()
