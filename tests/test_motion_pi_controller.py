from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fsop.eval.closed_loop import ModelPolicy, mission_brake, mission_brake_with_override
from fsop.eval.controller import (
    OpenpilotTangentFollower,
    OpenpilotTangentPIFollower,
    controller_contract,
    validate_checkpoint_controller,
    validate_checkpoint_payload,
)
from fsop.sim.vehicle import VehicleParams


def plan_at(speed):
    plan = np.zeros((33, 15))
    plan[:, 3] = speed
    return plan


def no_override(command):
    return command, False


def test_constant_resistance_requires_integral_for_zero_speed_error():
    vp = VehicleParams()
    pi = OpenpilotTangentPIFollower(vp)
    p = OpenpilotTangentFollower(vp)
    plan = plan_at(5.0)
    v_pi = v_p = 5.0
    for tick in range(800):
        _, command, diag = pi.step(plan, v_pi, 0.0, tick * .05, no_override)
        v_pi += .05 * (command - .24)
        _, command_p = p(plan, v_p, 0.0)
        v_p += .05 * (command_p - .24)
    assert v_pi == pytest.approx(5.0, abs=1e-5)
    assert v_p == pytest.approx(5.0 - .24 / 1.5, abs=1e-5)
    assert diag['integral_after'] == pytest.approx(.24, abs=1e-5)


def test_saturation_blocks_windup_and_allows_recovery():
    vp = VehicleParams(max_accel=1.0)
    pi = OpenpilotTangentPIFollower(vp)
    for tick in range(100):
        _, command, diag = pi.step(plan_at(10), 0, 0, tick * .05, no_override)
        assert command == 1
        assert diag['integral_after'] == 0
    for tick in range(100, 200):
        _, _, diag = pi.step(plan_at(1), 1, 0, tick * .05, no_override)
    assert diag['integral_after'] == 0
    assert diag['applied'] == 0

    pi.reset()
    for tick in range(30):
        pi.step(plan_at(1), .5, 0, tick * .05, no_override)
    before = pi.integral
    _, _, diag = pi.step(plan_at(1), 2, 0, 1.5, lambda command: (min(command, -2), False))
    assert diag['integral_after'] < before


def test_integral_limit_and_failed_mission_do_not_commit_state():
    pi = OpenpilotTangentPIFollower(VehicleParams())
    for tick in range(200):
        _, _, diag = pi.step(plan_at(5), 4, 0, tick * .05, no_override)
    assert diag['integral_after'] == 2.
    with pytest.raises(ValueError, match='finite'):
        pi.step(plan_at(5), 4, 0, 10., lambda command: (float('nan'), False))
    _, _, diag = pi.step(plan_at(5), 5, 0, 10., no_override)
    assert diag['integral_before'] == diag['integral_after'] == 2.


def test_mission_override_resets_integral_and_preserves_mission_brake():
    pi = OpenpilotTangentPIFollower(VehicleParams())
    for tick in range(20):
        pi.step(plan_at(5), 4, 0, tick * .05, no_override)
    assert pi.integral > 0
    for tick, speed, remaining in [(20, 1., 100.), (21, 10., 1.), (22, 0., .2)]:
        obs = SimpleNamespace(state=SimpleNamespace(vx=speed), s=0.)
        env = SimpleNamespace(mission=SimpleNamespace(s_end=remaining))
        _, command, diag = pi.step(plan_at(5), speed, 0, tick * .05,
                                   lambda proposed, obs=obs, env=env: mission_brake_with_override(proposed, obs, env))
        assert diag['mission_override']
        assert diag['integral_after'] == 0
        if tick == 20:
            assert diag['integral_before'] > 0
            assert diag['raw'] == pytest.approx(6.)
            assert command == pytest.approx(6.)
        assert command == pytest.approx(np.clip(mission_brake(diag['proposed'], obs, env), -13, 7))
    for speed in [0., 1.9, 2., 5., 10.]:
        for remaining in [0., .5, 1., 15., 16., 60., 61., 100.]:
            obs = SimpleNamespace(state=SimpleNamespace(vx=speed), s=0.)
            env = SimpleNamespace(mission=SimpleNamespace(s_end=remaining))
            for command in [-8., 0., 5.]:
                assert mission_brake_with_override(command, obs, env)[0] == mission_brake(command, obs, env)


def test_reset_timing_validation_stopped_target_and_binding():
    pi = OpenpilotTangentPIFollower(VehicleParams())
    plan = plan_at(4)
    first = pi.step(plan, 3, .4, 1., no_override)
    with pytest.raises(ValueError, match='20 Hz'):
        pi.step(plan, 3, .4, 1., no_override)
    with pytest.raises(ValueError, match='20 Hz'):
        pi.step(plan, 3, .4, 1.1, no_override)
    with pytest.raises(ValueError, match='vx and vy'):
        pi.step(plan, 3, None, 1.05, no_override)
    bad = plan.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        pi.step(bad, 3, 0, 1.05, no_override)
    _, _, stopped = pi.step(plan_at(0), 3, 0, 1.05, no_override)
    assert stopped['integral_after'] == 0
    pi.reset()
    assert pi.step(plan, 3, .4, 1., no_override) == first
    old = controller_contract('op_tangent_v1')
    new = controller_contract('op_tangent_pi_v1')
    validate_checkpoint_payload({'model': {}}, 'op_tangent_pi_v1')
    validate_checkpoint_controller(new, 'op_tangent_pi_v1')
    with pytest.raises(ValueError, match='does not match'):
        validate_checkpoint_controller(old, 'op_tangent_pi_v1')
    with pytest.raises(ValueError, match='missing'):
        validate_checkpoint_payload({'config': {'controller_mode': 'op_tangent_pi_v1'}}, 'op_tangent_pi_v1')


def test_first_step_matches_existing_follower_and_preserves_model_plan(monkeypatch):
    from fsop.eval import closed_loop

    vp = VehicleParams()
    plan = plan_at(5.)
    plan[:, 4] = 1.
    plan[:, 6] = .2
    plan[:, 11] = .1
    pi = OpenpilotTangentPIFollower(vp)
    old = OpenpilotTangentFollower(vp)
    steer, command, _ = pi.step(plan, 4., 1., 0., no_override)
    assert (steer, command) == old(plan, 4., 1.)

    monkeypatch.setattr(closed_loop, 'rgb_to_model', lambda image: np.zeros((6, 128, 256), dtype=np.uint8))
    class Base(torch.nn.Module):
        def encode(self, img, big):
            return SimpleNamespace(hidden=torch.zeros((1, 512)))
        def policy_raw(self, features, desire, traffic):
            return torch.zeros((1, 2 * 33 * 15)), None
    model = torch.nn.Module()
    model.base = Base()
    policy = ModelPolicy(model, controller='op_tangent_pi_v1', device='cpu', bev=False)
    obs = SimpleNamespace(images={'road': None, 'wide': None}, desire=np.zeros(8),
                          state=SimpleNamespace(vx=5., vy=.4), s=0., t=0.)
    env = SimpleNamespace(vp=vp, mission=SimpleNamespace(s_end=100.))
    _, _, info = policy(obs, env)
    np.testing.assert_array_equal(info['plan'], np.zeros((33, 15)))
    assert info['controller']['applied'] == info['controller']['proposed']
