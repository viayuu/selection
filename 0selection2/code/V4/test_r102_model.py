"""Small synthetic model checks, not native-source or efficacy evidence."""

import copy
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from code.unified_selector.registry import PROBLEMS
from .pairwise_objective import LABEL_KEYS
from .r88_model import make_model as original_model
from .r94_objective import RDropObjective as PreviousObjective
from .r102_model import (
    ARM_VARIANTS, DONOR_FIELDS, MODEL_ROLE_ARM, PROJECTION_NAMES,
    WIDTH_BY_PROBLEM, make_model, problem_name,
)
from .r102_objective import RDropObjective
from .test_r94_model import fixture_params
from .test_r76 import fixture_batch


class FakeDonor(nn.Module):
    SUPPORTED_PROBLEMS = PROBLEMS

    def __init__(self, arm):
        super().__init__()
        self.variant = ARM_VARIANTS[arm]
        self.provenance = dict(source='synthetic_test_only', variant=self.variant)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.projection = nn.Linear(8, 512, bias=False)
        self.dropout = nn.Dropout(.9)
        self.calls = []

    def forward(self, batch):
        if set(batch) & set(LABEL_KEYS):
            raise AssertionError('Labels reached donor')
        self.calls.append((self.training, torch.is_grad_enabled()))
        if batch['kind'] == 'matrix':
            x = batch['matrix'].mean(-1, keepdim=True).expand(-1, -1, 8)
        else:
            x = F.pad(batch['node'], (0, 8 - batch['node'].shape[-1]))
        return self.dropout(self.projection(x))[..., :WIDTH_BY_PROBLEM[problem_name(batch)]]


def params():
    return dict(fixture_params(), embedding_dim=128)


def batch(problem):
    result, _ = fixture_batch(problem, 2)
    if problem == 'TSP':
        result['node'] = result['node'][..., :2].clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(102)
        result['nco_nodes'] = torch.randn(*result['node_mask'].shape, WIDTH_BY_PROBLEM[problem])
    result['nco_node_mask'] = result['node_mask'].clone()
    return result


def model(arm='A'):
    torch.manual_seed(2)
    return make_model(params(), arm, donor=FakeDonor(arm))


class R102ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_common_parameters_and_rng_exactly_retained(self):
        candidates = []
        for arm in ('A', 'B'):
            torch.manual_seed(2)
            original = original_model(params(), MODEL_ROLE_ARM)
            expected_rng = torch.get_rng_state()
            got = model(arm)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            current = dict(got.named_parameters())
            for name, value in original.named_parameters():
                self.assertTrue(torch.equal(value, current[name]), name)
            added = {name: value for name, value in current.items()
                     if value.requires_grad and name not in dict(original.named_parameters())}
            self.assertEqual(set(added), set(PROJECTION_NAMES))
            self.assertEqual(sum(value.numel() for value in added.values()), 98304)
            candidates.append({name: value.detach().clone() for name, value in current.items() if value.requires_grad})
        self.assertEqual(set(candidates[0]), set(candidates[1]))
        self.assertTrue(all(torch.equal(value, candidates[1][name]) for name, value in candidates[0].items()))

    def test_zero_projections_recover_original_all_input_types(self):
        got = model().eval()
        torch.manual_seed(2)
        original = original_model(params(), MODEL_ROLE_ARM).eval()
        with torch.no_grad():
            for projection in got.instance_encoder.nco_projections.values():
                projection.weight.zero_()
            for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
                x = batch(problem)
                torch.testing.assert_close(got(x)['logits'], original(x)['logits'], atol=2e-6, rtol=2e-6)

    def test_labels_do_not_change_forward_and_padding_is_ignored(self):
        got = model().eval()
        with torch.no_grad():
            for problem in ('TSP', 'CVRP', 'ATSP'):
                x = batch(problem)
                expected = got(x)['logits']
                changed = dict(x, costs=x['costs'] * 7 + 3, ind=x['ind'].roll(1))
                changed['nco_nodes'] = x['nco_nodes'].clone()
                changed['nco_nodes'][~x['node_mask']] = float('nan')
                torch.testing.assert_close(got(changed)['logits'], expected, atol=0, rtol=0)

    def test_source_freeze_and_active_projection_gradients(self):
        got = model().train()
        self.assertFalse(got.donor.training)
        for problem, group in (('TSP', 'TSP'), ('ATSP', 'ATSP'), ('CVRP', 'VRP')):
            got.zero_grad(set_to_none=True)
            x = batch(problem)
            x['nco_nodes'].requires_grad_(True)
            got(x)['logits'].square().mean().backward()
            grad = got.instance_encoder.nco_projections[group].weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)
            self.assertIsNone(x['nco_nodes'].grad)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in got.donor.parameters()))

    def test_raw_eval_calls_donor_once_and_training_requires_cache(self):
        got = model()
        raw = {k: v for k, v in batch('ATSP').items() if k not in DONOR_FIELDS}
        with self.assertRaises(ValueError):
            got(raw)
        got.eval()
        with torch.no_grad():
            prepared = got.prepare_inputs(raw)
            first = got(prepared)['logits']
            second = got(prepared)['logits']
        self.assertEqual(got.donor.calls, [(False, False)])
        self.assertEqual(prepared['nco_nodes'].dtype, torch.float16)
        self.assertFalse(set(prepared) & set(LABEL_KEYS))
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_physical_node_relabeling_with_roles_and_cached_features(self):
        got = model().eval()
        with torch.no_grad():
            for problem in ('TSP', 'ATSP'):
                x = got.prepare_inputs(batch(problem))
                order = torch.arange(x['node_mask'].shape[1] - 1, -1, -1)
                changed = dict(x)
                for key in ('node', 'node_geom', 'node_mask', *DONOR_FIELDS):
                    if key in changed:
                        changed[key] = changed[key][:, order]
                if 'matrix' in changed:
                    changed['matrix'] = changed['matrix'][:, order][:, :, order]
                torch.testing.assert_close(got(changed)['logits'], got(x)['logits'], atol=2e-5, rtol=2e-5)

    def test_restore_skips_external_weight_initialization(self):
        got = model('B').eval()
        state = copy.deepcopy(got.state_dict())
        calls = []

        def factory(arm, **kwargs):
            calls.append(kwargs)
            return FakeDonor(arm)

        module = types.ModuleType('code.V4.r102_donor')
        module.FullTypeDonor = factory
        with patch.dict(sys.modules, {'code.V4.r102_donor': module}):
            restored = make_model(got.params, 'B', state_dict=state).eval()
        self.assertFalse(calls[0]['initialize_from_checkpoint'])
        self.assertEqual(calls[0]['expected_provenance'], got.frozen_donor['provenance'])
        with torch.no_grad():
            x = batch('ATSP')
            torch.testing.assert_close(restored(x)['logits'], got(x)['logits'], atol=0, rtol=0)

    def test_loss_is_original_two_view_node_fusion_objective(self):
        got = model().train()
        x = batch('TSP')
        before = torch.get_rng_state()
        expected, old_parts = PreviousObjective(got)(x, .17)
        torch.set_rng_state(before)
        actual, new_parts = RDropObjective(got)(x, .17)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(new_parts, old_parts, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
