import numpy as np
import pytest
import torch

from fsop.data.sim_dataset import Episode
from fsop.eval.controller import PlanPurePursuit
from fsop.geometry.op_frames import op_to_ego_plan
from fsop.sim.vehicle import VehicleParams
from fsop.train.losses import pp_steer_from_op_plan, steering_loss


def episode(tmp_path, control_times=(0.0, 0.05, 0.10, 0.15, 0.20), state_times=(0.0, 0.20)):
    ep = Episode.__new__(Episode)
    ep.path = tmp_path
    ep.T = len(state_times)
    ep.states = np.zeros((ep.T, 12))
    ep.states[:, 0] = state_times
    ep.meta = {'collection': {'assisted': True, 'noise': False, 'control_fps': 20,
                              'control_columns': ['t', 'expert_steer']},
               'vehicle': {'lf': 0.8, 'lr': 0.75, 'max_steer': 0.4}}
    np.save(tmp_path / 'controls.npy', np.column_stack((control_times, np.arange(len(control_times)) / 10)))
    return ep


def test_labels_match_observation_time_and_clip(tmp_path):
    ep = episode(tmp_path)
    target, lr, wheelbase, limit = ep._steering_labels()
    np.testing.assert_allclose(target, [0.0, 0.4])
    assert (lr, wheelbase, limit) == (0.75, 1.55, 0.4)


@pytest.mark.parametrize('control_times', [(0.0, 0.05, 0.10, 0.15), (0.0, 0.05, 0.10, 0.10, 0.20)])
def test_labels_reject_missing_or_duplicate_times(tmp_path, control_times):
    ep = episode(tmp_path, control_times=control_times)
    with pytest.raises(ValueError):
        ep._steering_labels()


def test_follower_matches_runtime_and_propagates_gradients():
    rng = np.random.default_rng(11)
    op = rng.normal(size=(6, 33, 15)).astype(np.float32)
    speed = np.array([-1, 0, 3, 8, 22, 100], np.float32)
    lrs = np.array([0.75, 0.8, 0.7, 0.9, 0.6, 0.75], np.float32)
    wheelbases = np.array([1.55, 1.6, 1.5, 1.8, 1.4, 1.55], np.float32)
    mu = torch.tensor(op, requires_grad=True)
    got = pp_steer_from_op_plan(mu, torch.tensor(speed), torch.tensor(lrs), torch.tensor(wheelbases))
    expected = []
    for plan, v, lr, wb in zip(op, speed, lrs, wheelbases, strict=True):
        vp = VehicleParams(lf=float(wb - lr), lr=float(lr))
        expected.append(PlanPurePursuit(vp)(op_to_ego_plan(plan), float(v))[0])
    np.testing.assert_allclose(got.detach().numpy(), expected, atol=2e-6, rtol=2e-6)
    got.sum().backward()
    assert torch.isfinite(mu.grad).all()
    assert mu.grad.abs().sum() > 0


def test_mask_excludes_unlabeled_samples_and_empty_batch_is_finite():
    mu = torch.zeros(2, 33, 15, requires_grad=True)
    batch = {'speed': torch.tensor([5., 5.]), 'vehicle_lr': torch.tensor([0.75, 0.75]),
             'vehicle_wheelbase': torch.tensor([1.55, 1.55]), 'steer_target': torch.tensor([0.2, -0.3]),
             'steer_valid': torch.tensor([True, False])}
    loss = steering_loss(mu, batch)
    assert loss.item() == pytest.approx(0.175)
    loss.backward()
    assert mu.grad[0].abs().sum() > 0
    assert mu.grad[1].abs().sum() == 0
    mu.grad.zero_()
    batch['steer_valid'][:] = False
    empty = steering_loss(mu, batch)
    assert empty.item() == 0
    empty.backward()
    assert torch.isfinite(mu.grad).all() and mu.grad.abs().sum() == 0
