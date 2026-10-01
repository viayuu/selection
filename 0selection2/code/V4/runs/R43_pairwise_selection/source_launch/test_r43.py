import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .dual_stream import DualStreamSelector
from .pairwise_selector import make_r43_model, maximin_logits, rank_by_solver_ids
from .pairwise_objective import comparison_targets, inputs_only, predicted_top3, comparison_loss, single_objective
from .r43_experiment import policy_metrics, update_batch
from .test_dual_stream import params
from .test_solver_features import sample_batch


class R43Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        torch.set_num_threads(2)
        self.params = dict(params(), pair_mode='explicit')

    def fixture(self, problem='TSP'):
        batch = sample_batch(problem)
        k = len(batch['pool_ids'])
        costs = np.stack((np.arange(k) + 1., np.arange(k)[::-1] + 1.)).astype(np.float64)
        batch.update(comparison_targets(costs))
        batch['ind'] = torch.tensor([0, k - 1])
        return batch

    def test_shared_initialization_and_old_interface(self):
        torch.manual_seed(2)
        a = make_r43_model(dict(self.params, pair_mode='score_difference')).eval()
        torch.manual_seed(2)
        b = make_r43_model(self.params).eval()
        for key, value in a.state_dict().items():
            if key.startswith(('instance_encoder.', 'solver_encoder.', 'joint_layers.', 'pool.')):
                torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
        old = DualStreamSelector(**self.params).eval()
        old.load_state_dict(a.state_dict(), strict=True)
        batch = self.fixture()
        self.assertTrue(torch.equal(old(batch)['logits'].argmax(-1), a(batch)['logits'].argmax(-1)))
        self.assertEqual(len(old.encode_state(batch)), 5)
        self.assertFalse(hasattr(b, 'decoder'))
        self.assertFalse(hasattr(b, 'score_head'))
        self.assertIsNone(b.delta_proj.bias)

    def test_full_pools_antisymmetry_and_no_label_input(self):
        for mode in ('score_difference', 'explicit'):
            model = make_r43_model(dict(self.params, pair_mode=mode)).eval()
            for problem in PROBLEMS:
                batch = self.fixture(problem)
                output = model(inputs_only(batch))
                torch.testing.assert_close(output['pair_margin'], -output['pair_margin'].transpose(-1, -2), atol=0, rtol=0)
                self.assertTrue((output['pair_margin'].diagonal(dim1=-2, dim2=-1) == 0).all())
                changed = dict(batch, costs=torch.full_like(batch['costs'], float('nan')), ind=torch.tensor([1, 0]))
                torch.testing.assert_close(model(changed)['logits'], output['logits'], atol=0, rtol=0)
                self.assertTrue(torch.isfinite(output['logits']).all())

    def test_permutation_and_padding(self):
        for mode in ('score_difference', 'explicit'):
            model = make_r43_model(dict(self.params, pair_mode=mode)).eval()
            for p in ('CVRP', 'ATSP', 'OVRPTW'):
                batch = self.fixture(p)
                result = model(batch)
                order = torch.randperm(len(batch['pool_ids']))
                permuted = model(dict(batch, pool_ids=batch['pool_ids'][order]))
                torch.testing.assert_close(permuted['logits'], result['logits'][:, order], atol=4e-6, rtol=4e-5)
                torch.testing.assert_close(permuted['pair_margin'], result['pair_margin'][:, order][:, :, order], atol=4e-6, rtol=4e-5)
                padded = copy.deepcopy(batch)
                padded['node'] = F.pad(batch['node'], (0, 0, 0, 3), value=999)
                padded['matrix'] = F.pad(batch['matrix'], (0, 3, 0, 3), value=999)
                padded['node_mask'] = F.pad(batch['node_mask'], (0, 3), value=False)
                torch.testing.assert_close(model(padded)['logits'], result['logits'], atol=4e-6, rtol=4e-5)
                mask = torch.ones(2, len(batch['pool_ids']) + 2, dtype=torch.bool)
                mask[:, -2:] = False
                padded = dict(batch, pool_ids=torch.cat((batch['pool_ids'], torch.tensor([-1, -1]))), solver_mask=mask)
                scores = model(padded)['logits']
                torch.testing.assert_close(scores[:, :-2], result['logits'], atol=4e-6, rtol=4e-5)
                self.assertTrue(torch.isneginf(scores[:, -2:]).all())
                one = model(dict(batch, pool_ids=batch['pool_ids'][:1]))['logits']
                torch.testing.assert_close(one, torch.zeros_like(one))

    def test_condorcet_rule_and_identity_ties(self):
        mask = torch.ones(2, 4, dtype=torch.bool)
        matrix = torch.zeros(2, 4, 4, requires_grad=True)
        values = matrix.clone()
        values[:, 2, :] = 1
        values[:, :, 2] = -1
        values[:, 2, 2] = 0
        scores = maximin_logits(values, mask)
        self.assertTrue((scores.argmax(-1) == 2).all())
        scores.sum().backward()
        self.assertIsNotNone(matrix.grad)
        ids = torch.tensor([9, 3, 5, 1])
        ranks = rank_by_solver_ids(torch.zeros(2, 4), ids)
        self.assertTrue((ids[ranks[:, 0]] == 1).all())

    def test_original_precision_direction_ties_and_nonwinner_supervision(self):
        costs = np.array([[1., 1. + 1e-10, 2., 2., 4.], [1., 1., 1., 1., 1.]], dtype=np.float64)
        targets = comparison_targets(costs)
        self.assertTrue(targets['cmp_valid'][0, 0])
        self.assertTrue(targets['cmp_target'][0, 0])
        i, j = np.triu_indices(5, 1)
        self.assertTrue(targets['cmp_true3'][0, (i == 1) & (j == 3)])
        self.assertFalse(targets['cmp_valid'][0, (i == 2) & (j == 3)])
        margins = torch.zeros(2, 5, 5, requires_grad=True)
        focus = torch.ones(2, 5, dtype=torch.bool)
        loss, _ = comparison_loss(margins, targets, focus)
        torch.testing.assert_close(loss, torch.tensor(np.log(2) / 2, dtype=torch.float32))
        loss.backward()
        self.assertLess(margins.grad[0, 0, 1], 0)
        self.assertEqual(margins.grad[0, 2, 3], 0)
        self.assertTrue((margins.grad[1] == 0).all())
        with self.assertRaises(ValueError):
            comparison_targets(costs.astype(np.float32))

    def test_rdrop_shared_focus_gradients_and_one_update(self):
        for mode in ('score_difference', 'explicit'):
            model = make_r43_model(dict(self.params, pair_mode=mode, dropout=.1)).train()
            batch = self.fixture()
            first, second = model(inputs_only(batch)), model(inputs_only(batch))
            self.assertFalse(torch.equal(first['logits'], second['logits']))
            focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
            self.assertFalse(focus.requires_grad)
            first_loss, parts = single_objective(first, batch, focus)
            torch.testing.assert_close(first_loss, .35*parts['ce'] + .35*parts['cmp'] + .02*parts['risk'])
            first_loss.backward()
            modules = [model.instance_encoder.coord_proj, model.solver_encoder.feature_mlp, model.pool, *model.joint_layers]
            modules += [model.pair_mlp, model.delta_proj, model.pair_out, *model.pair_layers] if mode == 'explicit' else [model.score_head]
            for module in modules:
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                values = update_batch(model, optimizer, scaler, batch, .35)
                self.assertEqual(step.call_count, 1)
                self.assertGreater(values['kl'], 0)

    def test_predictions_replay_same_policy(self):
        scores = np.array([[0., 0., 0.], [2., -1., .1]], dtype=np.float32)
        raw = dict(costs=np.array([[3., 2., 1.], [1., 4., 2.]], dtype=np.float64), winner=np.array([2, 0]),
                   pool=['a', 'b', 'c'], pool_ids=[9, 3, 1])
        metrics, pred = policy_metrics(scores, raw)
        np.testing.assert_array_equal(pred, [2, 0])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'predictions.npz'
            np.savez_compressed(path, logits=scores, **raw)
            with np.load(path) as saved:
                replay, new_pred = policy_metrics(saved['logits'], {k: saved[k] for k in raw})
            np.testing.assert_array_equal(new_pred, pred)
            for key in ('top1', 'top2', 'top3', 'mean_cost', 'actual_regret_pct'):
                self.assertEqual(metrics[key], replay[key])


if __name__ == '__main__':
    unittest.main()
