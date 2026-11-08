from dataclasses import asdict

import numpy as np
import pytest
import torch

from fsop.sim.vehicle import VehicleState
from isaac_eval.evidence import ImageHeadCapture
from isaac_eval.verify_inference import compare, observation, replay


class Head(torch.nn.Module):
    def forward(self, value):
        return {"road": {"heat": value * 2}}


class Policy:
    def __init__(self, model):
        self.model = model
        self.history = 0.0

    def __call__(self, obs, env):
        self.history += float(obs.images["road"].mean()) + float(obs.desire[1])
        value = torch.tensor([self.history])
        head = self.model.img(value)
        return self.history, obs.s, {"plan": value.numpy(), "plan_std": value.numpy() + 1,
                                    "cones_xy": head["road"]["heat"].numpy(),
                                    "cones_cls": np.array([1]), "cones_score": np.array([0.5])}


class Environment:
    def T_ego_cam(self):
        return np.eye(4)

    def T_world_ego(self):
        return np.eye(4)


def episode():
    env = Environment()
    model = torch.nn.Module()
    model.img = Head()
    policy = Policy(model)
    records = []
    for frame in range(3):
        record = {"frame": frame, "state": asdict(VehicleState(vx=frame, t=frame / 20)),
                  "progress": {"s": frame * 2., "lat": 0., "hint": frame},
                  "images": {name: np.full((2, 2, 3), frame + 1, np.uint8) for name in ("road", "wide")},
                  "commands": {"applied": np.array([0, 1 if frame == 1 else 0])},
                  "events": [{"t": 0., "kind": "lap", "data": {}}] if frame == 1 else [],
                  "nominal_detection_calibration": np.eye(4)}
        with ImageHeadCapture(model) as hook:
            steer, accel, prediction = policy(observation(record, env), env)
        record.update(action=[steer, accel], prediction=prediction, image_head=hook.output)
        records.append(({}, record, {}))
    return records, env, model


def test_sequential_replay_checks_saved_outputs_and_isolates_black_history():
    records, env, model = episode()
    count, saved, hook, visual = replay(iter(records), env, [Policy(model), Policy(model), Policy(model)])
    assert count == 3
    assert all(row["mismatches"] == 0 for row in saved.values())
    assert all(row["mismatches"] == 0 for row in hook.values())
    assert visual["prediction.plan"]["max_abs"] == 6.0
    assert records[1][1]["images"]["road"].mean() == 2
    assert observation(records[1][1], env).events[0].kind == "lap"
    records[1][1]["commands"]["applied"][1] = 0
    _, saved, _, _ = replay(iter(records), env, [Policy(model), Policy(model), Policy(model)])
    assert saved["prediction.plan"]["mismatches"] == 2


def test_replay_rejects_wrong_calibration_and_skipped_frame():
    records, env, model = episode()
    records[1][1]["nominal_detection_calibration"][0, 0] = 2
    with pytest.raises(ValueError, match="calibration"):
        replay(iter(records), env, [Policy(model), Policy(model), Policy(model)])
    records[1][1]["nominal_detection_calibration"][0, 0] = 1
    with pytest.raises(ValueError, match="not sequential"):
        replay(iter(records[1:]), env, [Policy(model), Policy(model), Policy(model)])


def test_comparison_reports_structure_nonfinite_and_class_changes():
    stats = {}
    compare({"cls": np.array([1]), "heat": np.array([1.]), "missing": np.array([2])},
            {"cls": np.array([2]), "heat": np.array([np.nan]), "extra": 3}, stats)
    assert stats[""]["mismatches"] == 1
    assert stats["cls"]["mismatches"] == 1
    assert stats["heat"]["mismatches"] == 1
