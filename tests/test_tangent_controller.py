from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fsop.eval.controller import (
    OpenpilotFollower,
    OpenpilotTangentFollower,
    controller_contract,
    validate_checkpoint_controller,
    validate_checkpoint_payload,
)
from fsop.geometry.op_frames import ego_to_op_plan
from fsop.sim.expert import T_IDXS
from fsop.sim.vehicle import VehicleParams
from fsop.train.lab_train import plan_coherence
from fsop.train.losses import op_tangent_actions_from_op_plan


def test_runtime_torch_actions_match_and_gradients_finite():
    vp = VehicleParams()
    plans = np.zeros((4, 33, 15), dtype=np.float32)
    theta = T_IDXS * .7
    plans[0, :, 3] = 6
    plans[0, :, 6] = 2
    plans[1, :, 3] = 6 * np.cos(theta)
    plans[1, :, 4] = 6 * np.sin(theta)
    plans[1, :, 6] = -4.2 * np.sin(theta)
    plans[1, :, 7] = 4.2 * np.cos(theta)
    plans[1, :, 11] = theta
    plans[1, :, 14] = .7
    plans[2, :, 6] = 100  # standstill: denominator floor and finite derivatives
    plans[3, :, 3] = 100  # accel clipping
    vx = np.array([3, 6, 0, 1], dtype=np.float32)
    vy = np.array([4, 1, 0, 2], dtype=np.float32)
    mu = torch.tensor(ego_to_op_plan(plans), requires_grad=True)
    actual = op_tangent_actions_from_op_plan(mu, torch.tensor(vx), torch.tensor(vy),
                                             torch.full((4,), vp.wheelbase),
                                             torch.full((4,), vp.max_brake), torch.full((4,), vp.max_accel))
    runtime = np.array([OpenpilotTangentFollower(vp)(plan, float(x), float(y))
                        for plan, x, y in zip(plans, vx, vy, strict=True)])
    np.testing.assert_allclose(torch.stack(actual, dim=1).detach().numpy(), runtime, atol=2e-6, rtol=1e-6)
    for i, plan in enumerate(plans):
        assert runtime[i, 0] == pytest.approx(OpenpilotFollower(vp)(plan, float(vx[i]))[0])
    (actual[0].sum() + actual[1].sum()).backward()
    assert torch.isfinite(mu.grad).all()


def test_rotation_and_nodewise_interpolation_and_missing_velocity():
    vp = VehicleParams()
    plan = np.zeros((33, 15))
    plan[:, 3] = 6
    plan[np.searchsorted(T_IDXS, .3):, 3] = -6
    follower = OpenpilotTangentFollower(vp)
    assert follower(plan, 6, 0)[1] == pytest.approx(0)
    rotation = np.array([[0, -1], [1, 0]])
    turned = plan.copy()
    for columns in ([3, 4], [6, 7]):
        turned[:, columns] = plan[:, columns] @ rotation.T
    assert follower(turned, 6, 0)[1] == pytest.approx(follower(plan, 6, 0)[1])
    with pytest.raises(ValueError, match='vx and vy'):
        follower(plan, 6)
    with pytest.raises(ValueError, match='vx and vy'):
        follower(plan, 6, np.nan)


def test_controller_contract_binding_and_legacy_opt_in():
    contract = controller_contract('op_tangent_v1')
    validate_checkpoint_controller(None, 'op_tangent_v1')
    validate_checkpoint_controller(contract, 'op_tangent_v1')
    with pytest.raises(ValueError, match='does not match'):
        validate_checkpoint_controller(contract, 'pp')
    with pytest.raises(ValueError, match='Invalid checkpoint'):
        validate_checkpoint_controller({**contract, 'speed_gain': 2}, 'op_tangent_v1')
    validate_checkpoint_payload({'model': {}}, 'op_tangent_v1')
    payload = {'model': {}, 'config': {'controller_mode': 'op_tangent_v1'}, 'controller_contract': contract}
    validate_checkpoint_payload(payload, 'op_tangent_v1')
    with pytest.raises(ValueError, match='does not match'):
        validate_checkpoint_payload(payload, 'pp')
    with pytest.raises(ValueError, match='missing'):
        validate_checkpoint_payload({'model': {}, 'config': {'controller_mode': 'op_tangent_v1'}}, 'pp')


def test_coherence_detects_inconsistent_motion_and_initial_state():
    plan = np.zeros((1, 33, 15))
    plan[0, :, 0] = 3 * T_IDXS
    plan[0, :, 3] = 3
    examples = [dict(speed=3, lateral_speed=0, yaw_rate=0)]
    coherent = plan_coherence(plan, examples)[0]
    assert max(coherent.values()) < 1e-12
    plan[0, 7, 0] -= .75
    plan[0, 0, 14] = .2
    broken = plan_coherence(plan, examples)[0]
    assert broken['position_interval_max_m'] > .7
    assert broken['position_cumulative_max_m'] > .7
    assert broken['initial_yaw_rate_radps'] == pytest.approx(.2)
    shifted = np.zeros((1, 33, 15))
    shifted[0, :, 11] = 2 * np.pi
    zero_state = [dict(speed=0, lateral_speed=0, yaw_rate=0)]
    assert plan_coherence(shifted, zero_state)[0]['initial_yaw_rad'] == pytest.approx(2 * np.pi)
    shifted[0, 5:, 11] += 2 * np.pi
    assert plan_coherence(shifted, zero_state)[0]['yaw_interval_max_rad'] == pytest.approx(2 * np.pi)


def test_model_policy_passes_measured_vx_and_vy_to_explicit_mode(monkeypatch):
    from fsop.eval import closed_loop

    calls = []
    monkeypatch.setattr(closed_loop, 'rgb_to_model', lambda image: np.zeros((6, 128, 256), dtype=np.uint8))
    monkeypatch.setattr(closed_loop, 'make_controller',
                        lambda mode, vp: lambda plan, vx, vy: calls.append((mode, vx, vy)) or (0., 0.))
    class Base(torch.nn.Module):
        def encode(self, img, big):
            return SimpleNamespace(hidden=torch.zeros((1, 512)))
        def policy_raw(self, features, desire, traffic):
            return torch.zeros((1, 2 * 33 * 15)), None
    model = torch.nn.Module()
    model.base = Base()
    policy = closed_loop.ModelPolicy(model, controller='op_tangent_v1', device='cpu', bev=False)
    state = SimpleNamespace(vx=5.0, vy=.4)
    obs = SimpleNamespace(images={'road': None, 'wide': None}, desire=np.zeros(8), state=state, s=0)
    env = SimpleNamespace(vp=VehicleParams(), mission=SimpleNamespace(s_end=100.0))
    policy(obs, env)
    assert calls == [('op_tangent_v1', 5.0, .4)]


def test_validation_reports_predicted_and_native_target_coherence(monkeypatch):
    from fsop.train import lab_train

    target = np.zeros((33, 15), dtype=np.float32)
    target[:, 0] = 3 * T_IDXS
    target[:, 3] = 3
    prediction = target.copy()
    prediction[7, 0] -= .75
    sample = {'plan': target, 'speed': np.float32(3), 'lateral_speed': np.float32(0), 'yaw_rate': np.float32(0)}
    class Dataset:
        episodes = [SimpleNamespace(path='train')]
        def __getitem__(self, index):
            return sample
    value = torch.tensor(0.)
    components = dict(plan_nll=value, plan_l1=value, steer=value, longitudinal=value)
    monkeypatch.setattr(lab_train, 'batch_loss', lambda *a, **kw: (value, components, {
        'plan_mu': torch.tensor(prediction[None]), 'coherence_targets': torch.tensor(target[None]),
        'coherence_observed': torch.tensor([[3., 0., 0.]])}))
    args = SimpleNamespace(controller_mode='op_tangent_v1', batch_size=1, w_steer=1, w_long=1,
                           history_chunk_size=1, channels_last=False, history_context=8, w_plan=1)
    result = lab_train.validate(None, Dataset(), {0: [0]}, torch.device('cpu'), args)
    assert result['coherence']['position_interval_max_m']['max'] > .7
    assert result['target_coherence']['position_interval_max_m']['max'] < 1e-6
    assert set(result['coherence']['yaw_interval_max_rad']) == {'mean', 'p95', 'max'}
    assert result['episodes'][0]['coherence'] == result['coherence']
    assert result['episodes'][0]['target_coherence'] == result['target_coherence']
    assert result['coherence']['position_cumulative_max_m']['max'] > .7
    assert result['target_coherence']['position_cumulative_max_m']['max'] < 1e-6


def test_native_batch_loss_uses_measured_lateral_velocity(monkeypatch):
    from fsop.train import lab_train

    class Dataset:
        episodes = []
        def __getitem__(self, index):
            plan = np.zeros((33, 15), dtype=np.float32)
            plan[:, 3] = 6
            return {'episode': 0, 'frame': 0, 'img': np.zeros((1,), np.float32),
                    'big_img': np.zeros((1,), np.float32), 'plan': plan, 'speed': np.float32(3),
                    'lateral_speed': np.float32(4), 'yaw_rate': np.float32(0),
                    'vehicle_wheelbase': np.float32(1.6), 'vehicle_lr': np.float32(.8),
                    'vehicle_max_brake': np.float32(8), 'vehicle_max_accel': np.float32(5),
                    'steer_target': np.float32(0), 'accel_target': np.float32(1.5)}
    class Model:
        def forward_images(self, batch, heads):
            mu = batch['plan'].clone().requires_grad_()
            return {'plan_mu': mu, 'plan_logstd': torch.zeros_like(mu)}
    monkeypatch.setattr(lab_train, 'fresh_history', lambda *args, **kwargs: torch.zeros((1, 24, 512)))
    _, components, output = lab_train.batch_loss(Model(), Dataset(), [0], torch.device('cpu'), 1, 1, 1,
                                                 controller_mode='op_tangent_v1', return_outputs=True)
    assert components['longitudinal'].item() == pytest.approx(0)
    assert output['coherence_observed'].tolist() == [[3, 4, 0]]


def test_frozen_selection_protocol_is_required_and_rechecked(tmp_path):
    import hashlib
    import json

    from fsop.train.lab_train import (
        COHERENCE_METRICS,
        SELECTION_TRAINING_FIELDS,
        recheck_selection_binding,
        selection_protocol_binding,
    )

    checkpoint = tmp_path / 'init.pt'
    checkpoint.write_bytes(b'init')
    calibration = tmp_path / 'calibration.json'
    calibration.write_bytes(b'calibration')
    args = SimpleNamespace(
        steps=500, val_every=100, batch_size=128, lr=.0001, lr_policy=.3, lr_trunk=.1,
        freeze_vision=True, l2sp=0., w_plan=1., w_steer=3.281158240898283, w_long=1., grad_clip=1.,
        val_samples=256, history_chunk_size=64, history_context=8, threads=4, amp=False, channels_last=False,
        device='cuda', seed=0, cache_frozen_features=False, recovery_draws=16,
        recovery_provenance_mode='off_policy_v1', collection_policy_checkpoint_sha='a' * 64,
        init=str(checkpoint), expected_calibration=str(calibration), calibration_only=False,
        selection_protocol=None, selection_protocol_sha=None,
    )
    with pytest.raises(ValueError, match='requires a frozen selection'):
        selection_protocol_binding(args, 'op_tangent_v1', 'b' * 64)
    args.calibration_only = True
    assert selection_protocol_binding(args, 'op_tangent_v1', 'b' * 64) is None
    args.calibration_only = False
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    membership = Path('docs/isaac-lab-training/membership.json')
    cases = [c['id'] for c in json.loads(membership.read_text())['cases'] if c['split'] == 'val']
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'membership_sha256': sha(membership)}))
    args.manifest = str(manifest)
    action_limits = {key: 1. for key in ('plan_l1', 'steer', 'longitudinal')}
    coherence_limits = {key: {stat: 1. for stat in ('mean', 'p95', 'max')} for key in COHERENCE_METRICS}
    bounds = {'aggregate': action_limits, 'episode': {case: action_limits for case in cases},
              'coherence': coherence_limits, 'episode_coherence': {case: coherence_limits for case in cases}}
    protocol = {'schema_version': 1, 'kind': 'op_tangent_v1_selection', 'calibration_sha256': sha(calibration),
                'initializer_sha256': sha(checkpoint), 'manifest_sha256': sha(manifest),
                'controller_contract': controller_contract('op_tangent_v1'), 'expected_steps': 500,
                'checkpoint_steps': [0, 100, 200, 300, 400, 500],
                'training_args': {key: getattr(args, key) for key in SELECTION_TRAINING_FIELDS}, 'bounds': bounds}
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    args.selection_protocol, args.selection_protocol_sha = str(path), sha(path)
    bound = selection_protocol_binding(args, 'op_tangent_v1', sha(manifest))
    copied = tmp_path / 'copied.json'
    copied.write_bytes(bound['bytes'])
    recheck_selection_binding(bound, copied)
    protocol['bounds']['coherence']['initial_yaw_rad']['max'] = None
    path.write_text(json.dumps(protocol))
    args.selection_protocol_sha = sha(path)
    with pytest.raises(ValueError, match='finite nonnegative'):
        selection_protocol_binding(args, 'op_tangent_v1', sha(manifest))
    path.write_bytes(bound['bytes'])
    args.selection_protocol_sha = sha(path)
    args.batch_size += 1
    with pytest.raises(ValueError, match='configuration mismatch'):
        selection_protocol_binding(args, 'op_tangent_v1', sha(manifest))
    calibration.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed during training'):
        recheck_selection_binding(bound, copied)


@pytest.mark.parametrize("channel,metric", [
    (3, "velocity_secant_max_mps"),
    (6, "acceleration_secant_max_mps2"),
    (14, "yaw_rate_secant_max_radps"),
])
def test_endpoint_secants_detect_opposite_knot_cancellation(channel, metric):
    plan = np.zeros((1, 33, 15))
    plan[0, :, channel] = 20 * (-1.) ** np.arange(33)
    sample = dict(speed=plan[0, 0, 3], lateral_speed=0, yaw_rate=-plan[0, 0, 14])
    measured = plan_coherence(plan, [sample])[0]
    assert measured[metric] == pytest.approx(20)
    interval = {3: 'position_interval_max_m', 6: 'velocity_interval_max_mps', 14: 'yaw_interval_max_rad'}
    assert measured[interval[channel]] == 0


def test_acceleration_alternation_is_visible_despite_zero_integral_residuals():
    plan = np.zeros((1, 33, 15))
    plan[0, :, 0] = 3 * T_IDXS
    plan[0, :, 3] = 3
    plan[0, :, 6] = 20 * (-1.) ** np.arange(33)
    measured = plan_coherence(plan, [dict(speed=3, lateral_speed=0, yaw_rate=0)])[0]
    assert measured['acceleration_secant_max_mps2'] == pytest.approx(20)
    assert max(v for k, v in measured.items() if 'secant' not in k) < 1e-12


def test_secant_metrics_preserve_finite_interval_dynamics():
    plan = np.zeros((1, 33, 15))
    plan[0, :, 0] = 3 * T_IDXS + T_IDXS ** 2
    plan[0, :, 3] = 3 + 2 * T_IDXS
    plan[0, :, 6] = 2
    plan[0, :, 11] = .5 * T_IDXS ** 2
    plan[0, :, 14] = T_IDXS
    measured = plan_coherence(plan, [dict(speed=3, lateral_speed=0, yaw_rate=0)])[0]
    n = np.searchsorted(T_IDXS, 1., side='right')
    max_dt = np.diff(T_IDXS[:n]).max()
    assert measured['velocity_secant_max_mps'] == pytest.approx(max_dt)
    assert measured['yaw_rate_secant_max_radps'] == pytest.approx(.5 * max_dt)
    assert measured['acceleration_secant_max_mps2'] < 1e-12
    assert measured['position_interval_max_m'] < 1e-12
