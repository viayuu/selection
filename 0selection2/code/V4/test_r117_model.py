"""Small CPU contracts, with a fake donor rather than native execution."""

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .r117_model import (ARMS, DONOR_FIELDS, PROJECTION_SHAPES, WIDTH_BY_PROBLEM,
                        inputs_with_donor, make_model)
from .test_r94_model import cached_batch, fixture_params


class FakeNative(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.register_buffer('marker', torch.tensor([1. if arm == 'A' else 2.]))
        self.provenance = dict(arm=arm, fake=True)

    def forward(self, batch):
        raise AssertionError('Cached CPU model must not execute native modules')


def fixture(problem):
    from code.unified_selector.registry import PROBLEMS
    batch = cached_batch(problem)
    width = WIDTH_BY_PROBLEM[problem]
    batch['native_nodes'] = torch.arange(2 * 5 * width).float().reshape(2, 5, width).sin()
    batch['native_node_mask'] = batch['node_mask'].clone()
    batch['problem_id'] = PROBLEMS.index(problem)
    return batch


class NativeFusionTests(unittest.TestCase):
    def test_shared_initialization_and_zero_label_input(self):
        torch.set_num_threads(2)
        states = []
        for arm in ARMS:
            torch.manual_seed(2)
            model = make_model(fixture_params(), arm, donor=FakeNative(arm))
            states.append({key: value.clone() for key, value in model.named_parameters() if value.requires_grad})
            self.assertTrue(set(PROJECTION_SHAPES) <= set(states[-1]))
            model.eval()
            batch = fixture('TSP')
            first = model(batch)['logits']
            altered = dict(batch, costs=torch.full_like(batch['costs'], 123456.),
                ind=1 - batch['ind'], costs_fp64=torch.randn(2, 1), winner=torch.tensor([1, 1]))
            torch.testing.assert_close(first, model(altered)['logits'], rtol=0, atol=0)
            self.assertFalse({'ind', 'costs', 'costs_fp64', 'winner'} & set(inputs_with_donor(altered)))
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_all_native_types_have_selector_gradients_but_donor_is_frozen(self):
        torch.set_num_threads(2)
        model = make_model(fixture_params(), 'B', donor=FakeNative('B'))
        model.train()
        for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPBLTW'):
            model.zero_grad(set_to_none=True)
            batch = fixture(problem)
            batch['native_nodes'].requires_grad_()
            output = model(batch)
            F.cross_entropy(output['logits'], batch['ind']).backward()
            group = problem if problem in ('TSP', 'CVRP', 'ATSP') else 'VRP'
            grad = model.instance_encoder.native_projections[group].weight.grad
            self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)
            self.assertIsNone(batch['native_nodes'].grad)
            self.assertFalse(model.donor.training)
        self.assertFalse(any(p.requires_grad for p in model.donor.parameters()))


if __name__ == '__main__':
    unittest.main()
