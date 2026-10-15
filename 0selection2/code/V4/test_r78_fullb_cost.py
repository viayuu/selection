"""Five bounded synthetic CPU witnesses for the authorized round003 path."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from code.unified_selector.registry import PROBLEMS
from . import r78_fullb_cost_experiment as r78
from .pairwise_objective import LABEL_KEYS, comparison_targets
from .r76_experiment import RDropObjective, Runtime, update_batch


class TinySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.2, -.3, .1]))
        self.calls = 0

    def forward(self, batch):
        if set(batch).intersection((*LABEL_KEYS, r78.MONO_LABEL)):
            raise AssertionError('A supervision tensor leaked into a model forward')
        self.calls += 1
        scores = self.weight[None, :] + batch['x']
        return dict(logits=scores, pair_margin=scores[:, :, None] - scores[:, None, :],
                    solver_mask=batch['solver_mask'])


def fixture(rows=7):
    costs = np.tile(np.asarray([[1., 1.2, 1.1]], dtype=np.float64), (rows, 1))
    weights, _ = r78.train_weights(costs)
    return dict(x=torch.zeros(rows, 3), ind=torch.zeros(rows, dtype=torch.long),
        costs=torch.from_numpy(costs.astype(np.float32)), pool_ids=torch.tensor([0, 1, 2]),
        solver_mask=torch.ones(rows, 3, dtype=torch.bool), **comparison_targets(costs),
        **{r78.MONO_LABEL: torch.from_numpy(weights)})


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters):
        super().__init__(parameters, lr=1e-4, weight_decay=1e-4)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


class FullBCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_fp64_gap_and_task_scale_not_instance_scale(self):
        costs = np.asarray([[1., 1. + 1e-10, np.nan], [2., 4., 6.]], dtype=np.float64)
        mask = np.asarray([[True, True, False], [True, True, True]])
        regret, _ = r78.relative_regrets(costs, mask)
        self.assertGreater(regret[0, 1], 0)
        weights, scale = r78.train_weights(costs, mask)
        self.assertAlmostEqual(scale, ((1e-10 / 2) + 1.) / 2, places=12)
        self.assertEqual(weights[0, 2], 0)
        self.assertGreater(weights[1].mean(), weights[0, :2].mean())
        with self.assertRaises(ValueError):
            r78.train_weights(costs.astype(np.float32), mask)

    def test_centered_softplus_mask_shift_and_exact_zero_block(self):
        scores = torch.tensor([[2., -3., 999.]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        weights = torch.tensor([[0., 2., 888.]])
        risk = r78.monotone_risk(scores, weights, mask)
        shifted = r78.monotone_risk(scores + 8., weights, mask)
        torch.testing.assert_close(risk, shifted)
        risk.backward()
        self.assertEqual(float(scores.grad[0, 2]), 0.)
        self.assertAlmostEqual(float(scores.grad.sum()), 0., places=7)
        zero, scale = r78.train_weights(np.ones((2, 3), dtype=np.float64))
        self.assertEqual(scale, 0.)
        self.assertEqual(float(r78.monotone_risk(torch.randn(2, 3), torch.from_numpy(zero),
                                               torch.ones(2, 3, dtype=torch.bool))), 0.)

    def test_control_recipe_and_two_label_free_dropout_branches(self):
        batch = fixture()
        base = RDropObjective(TinySelector())
        control = r78.FullBCostObjective(TinySelector(), 'control')
        treatment = r78.FullBCostObjective(TinySelector(), 'treatment')
        original_batch = {key: value for key, value in batch.items() if key != r78.MONO_LABEL}
        base_loss, base_parts = base(original_batch, .35)
        old_loss, old_parts = control(batch, .35)
        new_loss, new_parts = treatment(batch, .35)
        torch.testing.assert_close(base_loss, old_loss)
        torch.testing.assert_close(base_parts, old_parts)
        keys = r78.PART_KEYS[:-1]
        for key in ('ce', 'cmp', 'kl', 'true3', 'pred3', 'all_pairs'):
            torch.testing.assert_close(old_parts[keys.index(key)], new_parts[keys.index(key)])
        torch.testing.assert_close(new_loss - old_loss,
            .02 * (new_parts[keys.index('risk')] - old_parts[keys.index('risk')]), atol=1e-7, rtol=1e-5)
        self.assertEqual((control.selector.calls, treatment.selector.calls), (2, 2))

    def test_exact1024_micro_accumulation_one_fresh_optimizer_and_clip(self):
        batch = fixture(r78.BATCH_SIZE)
        models = []
        for cap in (1024, 128):
            selector = TinySelector()
            models.append(selector)
            objective = r78.FullBCostObjective(selector, 'treatment')
            optimizer = CountingAdamW(selector.parameters())
            self.assertFalse(optimizer.state)
            scaler = torch.amp.GradScaler('cpu', enabled=False)
            with patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
                _, retries = update_batch(objective, optimizer, scaler, batch, .35, 1024, 1024,
                    Runtime(0, 1, torch.device('cpu')), microbatch_cap=cap)
            self.assertEqual((optimizer.steps, clip.call_count, retries), (1, 1, 0))
            self.assertEqual(selector.calls, 2 * (1024 // cap))
        torch.testing.assert_close(models[0].weight, models[1].weight)

    def test_matched_late_schedule_and_whole18_unrounded_gate(self):
        self.assertEqual(r78.SOURCE_UPDATE + r78.CONTINUATION_UPDATES, 49680)
        self.assertEqual(r78.ENDPOINTS, (4968, 9936, 14904))
        self.assertEqual([r78.SOURCE_UPDATE + u for u in r78.ENDPOINTS], [39744, 44712, 49680])
        rates = [r78.learning_rate(r78.SOURCE_UPDATE + u) for u in range(1, 14905)]
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        self.assertEqual(rates[-1], 1e-6)
        order = r78.task_order(14904, seed=2)
        self.assertEqual(set(r78.Counter(order).values()), {828})
        metrics = dict(macro_top1=.7089430894308943, macro_mean_cost=r78.GATE_COST,
                       macro_actual_regret_pct=r78.GATE_REGRET)
        baseline = dict(continuation_update=0, val=dict(macro=metrics, per_problem={p: {} for p in PROBLEMS}))
        good = copy.deepcopy(baseline)
        good['continuation_update'] = 4968
        good['val']['macro']['macro_top1'] += .001
        bad = copy.deepcopy(good)
        bad['continuation_update'] = 9936
        bad['val']['macro']['macro_mean_cost'] += 1e-9
        self.assertIs(r78.select_checkpoint(baseline, [bad, good]), good)
        self.assertIs(r78.select_checkpoint(baseline, [bad]), baseline)
        incomplete = copy.deepcopy(good)
        incomplete['val']['per_problem'].pop('TSP')
        with self.assertRaises(ValueError):
            r78.select_checkpoint(baseline, [incomplete])


if __name__ == '__main__':
    unittest.main()
