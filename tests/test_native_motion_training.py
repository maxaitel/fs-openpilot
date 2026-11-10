import numpy as np
import pytest
import torch

from fsop.eval.controller import tangent_motion_request
from fsop.geometry.op_frames import ego_to_op_plan
from fsop.sim.expert import T_IDXS
from fsop.sim.vehicle import VehicleParams
from fsop.train.action_target import action_target_contract
from fsop.train.losses import native_relative_acceleration_loss, tangent_motion_from_op_plan


def test_desired_motion_matches_runtime_before_saturation():
    vp = VehicleParams()
    plan = np.zeros((33, 15), dtype=np.float32)
    plan[:, 3] = 30
    plan[:, 6] = 4
    plan[:, 11] = T_IDXS * .2
    plan[:, 14] = .2
    request = tangent_motion_request(plan, 4, 3, vp)
    steer, motion = tangent_motion_from_op_plan(torch.tensor(ego_to_op_plan(plan)[None]),
                                               torch.tensor([4.]), torch.tensor([3.]),
                                               torch.tensor([vp.wheelbase]))
    assert steer.item() == pytest.approx(request.steer, abs=1e-6)
    assert motion.item() == pytest.approx(1.5 * (request.target_speed - 5) + .5 * request.tangent_accel)
    assert motion.item() > vp.max_accel
    action_target_contract('desired_motion_v1', 'op_tangent_pi_v1')
    with pytest.raises(ValueError):
        action_target_contract('teacher_command_v1', 'op_tangent_pi_v1')


def test_native_consistency_preserves_nonconstant_native_motion_and_detaches_target():
    times = torch.tensor(T_IDXS, dtype=torch.float32)
    target = torch.zeros((1, 33, 15))
    target[0, :, 3] = 6 * torch.cos(.7 * times)
    target[0, :, 4] = 6 * torch.sin(.7 * times)
    target[0, :, 6] = -4.2 * torch.sin(.7 * times)
    target[0, :, 7] = 4.2 * torch.cos(.7 * times)
    target.requires_grad_()
    predicted = target.detach().clone().requires_grad_()
    assert native_relative_acceleration_loss(predicted, target).item() == 0
    changed = predicted.clone()
    changed[:, 3, 3] += .2
    loss = native_relative_acceleration_loss(changed, target)
    assert loss.item() > 0
    loss.backward()
    assert target.grad is None
    assert torch.isfinite(predicted.grad).all()
    assert torch.count_nonzero(predicted.grad) > 0
    late = target.detach().clone()
    late[:, T_IDXS > 1., 3:8] += 5
    assert native_relative_acceleration_loss(late, target).item() == 0


def test_native_training_archive_includes_runtime_validation_dependencies(tmp_path):
    import hashlib
    from zipfile import ZipFile
    from fsop.train.lab_train import archive_source

    path = tmp_path / 'source.zip'
    record = archive_source(path, include_native=True)
    required = {'fsop/train/lab_train.py', 'scripts/select_lab_tangent.py',
                'scripts/verify_lab_recovery.py', 'isaac_lab_fsop/recovery_schedule.py'}
    assert required <= record['files_sha256'].keys()
    with ZipFile(path) as archive:
        assert set(archive.namelist()) == set(record['files_sha256'])
        for name in required:
            assert hashlib.sha256(archive.read(name)).hexdigest() == record['files_sha256'][name]
