"""Focused synthetic CPU checks for R93's one-forward native CE."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from . import r93_experiment as runner
from .pairwise_objective import LABEL_KEYS
from .pairwise_selector import maximin_logits
from .r76_experiment import PART_KEYS, Runtime
from .r93_objective import PlainClassificationObjective


class ToySelector(nn.Module):
    def __init__(self, dropout=.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Dropout(dropout))
        self.score_head = nn.Linear(6, 4)
        self.outputs = []

    def forward(self, batch):
        if set(batch) & set(LABEL_KEYS):
            raise AssertionError('Supervision reached model inputs')
        utility = self.score_head(self.encoder(batch['x']))
        mask = batch['solver_mask']
        margin = utility[:, :, None] - utility[:, None, :]
        output = dict(utility=utility, solver_mask=mask, logits=maximin_logits(margin, mask))
        self.outputs.append(output)
        return output


def fixture(rows=5):
    mask = torch.ones(rows, 4, dtype=torch.bool)
    mask[-1, -1] = False
    return dict(x=torch.randn(rows, 3), solver_mask=mask, pool_ids=torch.tensor([4, 1, 9, 6]),
        ind=torch.arange(rows) % 3, costs=torch.full((rows, 4), float('nan')),
        performance_target=torch.full((rows, 4), float('nan')),
        cmp_target=torch.full((rows, 6), float('nan')),
        cmp_valid=torch.zeros(rows, 6, dtype=torch.bool),
        cmp_true3=torch.zeros(rows, 6, dtype=torch.bool),
        cost_regret=torch.full((rows, 4), float('nan')))


class R93ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2)
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_one_forward_exact_unweighted_raw_utility_ce_and_only_native_target(self):
        batch = fixture()
        # FP32 costs would select method 0, but native ind is deliberately 1.
        batch['costs'] = torch.ones(5, 4)
        batch['ind'][0] = 1
        objective = PlainClassificationObjective(ToySelector()).train()
        loss, parts = objective(batch, float('nan'))
        self.assertEqual(len(objective.selector.outputs), 1)
        output = objective.selector.outputs[0]
        expected = F.cross_entropy(output['utility'].masked_fill(~output['solver_mask'], -torch.inf), batch['ind'])
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        self.assertEqual(parts.shape, (len(PART_KEYS) - 1,))
        torch.testing.assert_close(parts[:2], expected.detach().expand(2), rtol=0, atol=0)
        self.assertTrue((parts[2:] == 0).all())
        self.assertFalse(parts.requires_grad)
        self.assertNotEqual(loss.item(), F.cross_entropy(output['logits'], batch['ind']).item())
        loss.backward()
        for parameter in objective.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(sum(p.grad.abs().sum().item() for p in objective.parameters()), 0.)

    def test_legacy_consistency_weight_does_not_change_loss_gradient_or_rng(self):
        batch = fixture()
        left = PlainClassificationObjective(ToySelector()).train()
        right = copy.deepcopy(left)
        rng = torch.get_rng_state()
        loss, parts = left(batch, 0.)
        after = torch.get_rng_state()
        loss.backward()
        torch.set_rng_state(rng)
        other, other_parts = right(batch, 1000.)
        other.backward()
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        torch.testing.assert_close(parts, other_parts, rtol=0, atol=0)
        for a, b in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_illegal_nonfinite_utilities_and_singleton_pool_have_zero_gradient(self):
        class FixedSelector(nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = nn.Parameter(torch.tensor([[2., -1., float('nan')],
                                                        [3., float('inf'), float('nan')]]))

            def forward(self, batch):
                return dict(utility=self.scores, solver_mask=batch['solver_mask'])

        objective = PlainClassificationObjective(FixedSelector())
        mask = torch.tensor([[True, True, False], [True, False, False]])
        loss, parts = objective(dict(ind=torch.tensor([0, 0]), solver_mask=mask))
        self.assertTrue(torch.isfinite(parts).all())
        loss.backward()
        self.assertTrue(torch.isfinite(objective.selector.scores.grad).all())
        self.assertTrue((objective.selector.scores.grad[~mask] == 0).all())
        self.assertTrue((objective.selector.scores.grad[1] == 0).all())

    def test_row_weighted_1024_update_matches_full_ce_with_unequal_tail(self):
        batch = fixture(1024)
        full = PlainClassificationObjective(ToySelector(dropout=0.)).train()
        micro = copy.deepcopy(full)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
                      for m in (full, micro)]
        loss, expected_parts = full(batch, 0.)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(full.parameters(), 1.)
        optimizers[0].step()
        parts, retries, metrics = runner.update_batch(micro, optimizers[1],
            torch.amp.GradScaler('cuda', enabled=False), batch, .35,
            Runtime(0, 1, torch.device('cpu')), 352, problem='TSP')
        self.assertEqual([len(o['utility']) for o in micro.selector.outputs], [352, 352, 320])
        self.assertEqual(retries, 0)
        torch.testing.assert_close(parts[:-1], expected_parts.double(), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(parts[-1], norm.double(), rtol=2e-6, atol=2e-7)
        self.assertEqual(set(metrics), {'train_utility_ce'})
        for a, b in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)

    def test_idle_and_incomplete_logical_batch_guards(self):
        objective = PlainClassificationObjective(ToySelector())
        loss, parts = objective(None)
        self.assertEqual(loss.item(), 0.)
        self.assertTrue((parts == 0).all())
        self.assertEqual(objective.selector.outputs, [])
        loss.backward()
        with self.assertRaises(ValueError):
            runner.update_batch(objective, None, None, fixture(1023), 0.,
                Runtime(0, 1, torch.device('cpu')), 352, problem='TSP')


if __name__ == '__main__':
    unittest.main()
