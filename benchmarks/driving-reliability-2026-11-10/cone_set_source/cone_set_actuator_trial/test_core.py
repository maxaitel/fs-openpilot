from __future__ import annotations

import copy
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import cone_core as core
from fsop.model.temporal import TemporalState
from fsop.sim.vehicle import VehicleParams


def head_inputs(batch=4):
    generator = torch.Generator().manual_seed(73)
    tokens = torch.randn(batch, 200, 8, generator=generator)
    mask = torch.arange(200)[None] < torch.arange(1, batch + 1)[:, None] * 3
    desire = torch.randn(batch, 25, 8, generator=generator)
    proprio = torch.randn(batch, 3, generator=generator)
    return tokens, mask, desire, proprio


def raw_head(batch=2):
    result = {}
    for camera in ('wide', 'road'):
        heat = torch.full((batch, 4, 64, 128), -40.)
        heat[:, 0, 50, 60] = 1.
        heat[:, 1, 50, 70] = 1.
        result[camera] = {'heat': heat, 'wh': torch.ones(batch, 2, 64, 128),
                          'off': torch.zeros(batch, 2, 64, 128)}
    return result


class DiagnosticBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.desires = []

    def encode(self, road, wide):
        self.calls += 1
        hidden = torch.full((1, 512), float(self.calls))
        return SimpleNamespace(hidden=hidden, s4=road, s8=wide, s16=road)

    def policy_raw(self, history, desire, traffic):
        self.desires.append(desire.clone())
        return torch.zeros(1, 990), torch.zeros(1, 8)


class DiagnosticImage(nn.Module):
    def forward(self, *unused):
        return raw_head(1)


class DiagnosticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = DiagnosticBase()
        self.img = DiagnosticImage()
        self.bev = nn.Identity()


class ConeCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_exact_parameter_inventory(self):
        head = core.ConeSetHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 47618)
        self.assertEqual({n: list(p.shape) for n, p in head.state_dict().items()}, core.HEAD_SCOPE)
        self.assertEqual(len(head.state_dict()), 8)

    def test_pack_normalization_padding_empty_and_nonmutation(self):
        xy = np.array([[3., -4.], [45., 0.]])
        classes, scores = np.array([1, 3]), np.array([.7, 1.2])
        before = [a.copy() for a in (xy, classes, scores)]
        tokens, mask = core.pack_cones(xy, classes, scores)
        np.testing.assert_allclose(tokens[0], [3/45, -4/45, 5/45, 0, 1, 0, 0, .7])
        self.assertEqual(tokens.dtype, np.float32)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[:2].all())
        self.assertFalse(mask[2:].any())
        self.assertFalse(tokens[2:].any())
        for current, original in zip((xy, classes, scores), before, strict=True):
            np.testing.assert_array_equal(current, original)
        empty, empty_mask = core.pack_cones(*json.loads('[[],[],[]]'))
        self.assertEqual(empty.shape, (200, 8))
        self.assertFalse(empty.any() or empty_mask.any())

    def test_invalid_decoder_boundaries(self):
        for xy, classes, scores in (([[1., 2.]], [4], [.5]), ([[np.nan, 2]], [0], [.5]),
                                    ([[1., 2.]], [1.5], [.5]), ([1., 2.], [0], [.5]),
                                    (np.zeros((201, 2)), np.zeros(201, int), np.zeros(201))):
            with self.assertRaises(ValueError):
                core.pack_cones(xy, classes, scores)

    def test_order_padding_empty_and_head_nonmutation(self):
        head = core.ConeSetHead().eval()
        inputs = head_inputs()
        before = [x.clone() for x in inputs]
        expected = head(*inputs)
        order = torch.randperm(200)
        actual = head(inputs[0][:, order], inputs[1][:, order], *inputs[2:])
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
        padded = inputs[0].clone()
        padded[~inputs[1]] = 1e20
        self.assertTrue(torch.equal(expected, head(padded, *inputs[1:])))
        empty = torch.zeros_like(inputs[1])
        self.assertTrue(torch.equal(head(inputs[0], empty, *inputs[2:]), head(torch.zeros_like(inputs[0]), empty, *inputs[2:])))
        for a, b in zip(inputs, before, strict=True):
            self.assertTrue(torch.equal(a, b))
        changed_desire = inputs[2].clone()
        changed_desire[:, 0, 0] += 1
        self.assertFalse(torch.equal(expected, head(inputs[0], inputs[1], changed_desire, inputs[3])))

    def test_masked_padding_has_no_gradient(self):
        head = core.ConeSetHead()
        tokens, mask, desire, proprio = head_inputs()
        tokens.requires_grad_(True)
        head(tokens, mask, desire, proprio).sum().backward()
        self.assertTrue(torch.equal(tokens.grad[~mask], torch.zeros_like(tokens.grad[~mask])))
        self.assertGreater(float(tokens.grad[mask].abs().sum()), 0)

    def test_same_raw_decoder_stock_singleton_exact(self):
        raw = raw_head()
        calibration = np.eye(4)
        calibration[2, 3] = 1.
        actual = core.decode_batch(raw, [calibration, calibration])
        for index, decoded in enumerate(actual):
            singleton = {c: {k: v[index:index+1] for k, v in fields.items()} for c, fields in raw.items()}
            expected = core.bev_from_image_head(singleton, calibration)
            for a, b in zip(decoded, expected, strict=True):
                np.testing.assert_array_equal(a, b)
        for camera in raw:
            raw[camera]['heat'].fill_(-40.)
        self.assertEqual(len(core.decode_batch(raw, [calibration, calibration])[0][0]), 0)

    def test_policy_calls_original_once_and_preserves_diagnostic_queue(self):
        model = DiagnosticModel()
        policy = core.ConeSetPolicy(model, core.ConeSetHead(), core.CONTRACT, 'cpu')
        env = SimpleNamespace(vp=VehicleParams(), mission=SimpleNamespace(s_end=100.), T_ego_cam=lambda: np.eye(4))
        expected = TemporalState()
        obs = SimpleNamespace(state=SimpleNamespace(vx=3., vy=.1, r=.02), s=0., t=0.,
                              images={c: np.zeros((256, 512, 3), np.uint8) for c in ('road', 'wide')})
        with patch('fsop.eval.img_to_bev.bev_from_image_head', return_value=(np.empty((0, 2)), np.empty(0, int), np.empty(0))):
            for tick in range(7):
                obs.desire = np.eye(8, dtype=np.float32)[tick % 3]
                obs.t = tick / 20
                want = expected.push_desire(obs.desire)
                _, _, info = policy(obs, env)
                np.testing.assert_array_equal(info['controller']['desire'], want)
                np.testing.assert_array_equal(model.base.desires[-1][0], want)
                self.assertEqual(model.base.calls, tick + 1)
                self.assertFalse(any(info['controller']['mask']))
                self.assertIn('plan', info)
                self.assertIn('base_controller_diagnostic', info)
        policy.reset()
        self.assertTrue(all(not row.any() for row in policy.base_policy.ts.desire_q))

    def test_real_417_base_package_tamper_rejection(self):
        path = Path('runs/joint_command_trial1_archive/step_000.pt')
        if not path.exists():
            path = Path('runs/joint_command_trial1/step_000.pt')
        if not path.exists():
            self.skipTest('original frozen 417-state fixture is unavailable')
        prior = torch.load(path, map_location='cpu', weights_only=False)
        payload = {**prior['provenance']['original_payload_metadata'], 'model': prior['base_payload']['model']}
        state = core.ConeSetHead().state_dict()
        base = core.digest_state(payload['model'])
        self.assertEqual(len(base), 417)
        self.assertTrue(any(v.dtype == torch.int64 for v in payload['model'].values()))
        provenance = {'step': 1, 'training_scope': core.HEAD_SCOPE, 'base_model_digest': core.BASE_DIGEST,
                      'original_base_model_digest': core.BASE_DIGEST, 'base_state_digest': base,
                      'original_base_state_digest': base, 'head_state_digest': core.digest_state(state),
                      'original_base_checkpoint_sha256': core.BASE_CHECKPOINT_SHA256,
                      'original_payload_metadata': prior['provenance']['original_payload_metadata'],
                      'frozen_base_state_unchanged': True, 'frozen_base_state_count': 417}
        package = {'kind': core.KIND, 'base_payload': payload, 'head_state': state,
                   'command_contract': copy.deepcopy(core.CONTRACT), 'recipe': {}, 'provenance': provenance}
        core.validate_package(package)
        buffer = io.BytesIO()
        torch.save(package, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        core.validate_package(restored)
        with torch.no_grad():
            restored['head_state']['token1.bias'][0] += .01
        with self.assertRaisesRegex(ValueError, 'head digest'):
            core.validate_package(restored)
        metadata_tamper = {**package, 'base_payload': {**payload, 'step': 1},
                           'provenance': {**provenance, 'original_payload_metadata': {**provenance['original_payload_metadata'], 'step': 1}}}
        with self.assertRaisesRegex(ValueError, 'trusted original'):
            core.validate_package(metadata_tamper)
        first = next(iter(payload['model']))
        forged_state = dict(payload['model'])
        forged_state[first] = forged_state[first] + 1
        forged_digest = core.digest_state(forged_state)
        tampered = {**package, 'base_payload': {**payload, 'model': forged_state},
                    'provenance': {**provenance, 'base_state_digest': forged_digest,
                                   'original_base_state_digest': forged_digest,
                                   'base_model_digest': core.state_model_digest(forged_state)}}
        with self.assertRaisesRegex(ValueError, 'trusted original'):
            core.validate_package(tampered)


if __name__ == '__main__':
    unittest.main()
