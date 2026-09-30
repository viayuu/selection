import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import M_GLOBAL, PROBLEMS
from .direct_selector import DirectSelector, make_r40_model
from .multitask_probe import state_hash
from .r40_experiment import classification_metrics, initialize, learning_rate, subdivide_schedule
from .test_dual_stream import params
from .test_solver_features import sample_batch
from .train import selector_loss
from .performance_experiment import fit_args


class R40Tests(unittest.TestCase):
    def setUp(self):
        self.params = dict(params(), local_geometry=True, ignore_coord_dist=True)
        self.geometry = dict(mean=[0.] * 12, std=[1.] * 12)

    def test_shared_encoder_initialization_and_no_unused_performance_head(self):
        a, _ = initialize('A', 2, self.geometry, 'cpu', dict(self.params, architecture='dual_stream'))
        b, _ = initialize('B', 2, self.geometry, 'cpu', dict(self.params, architecture='direct'))
        self.assertEqual(state_hash(a.instance_encoder.state_dict()), state_hash(b.instance_encoder.state_dict()))
        for model in (a, b):
            self.assertFalse(hasattr(model, 'performance_head'))
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(hasattr(b, 'solver_encoder'))
        self.assertEqual(b.classifier[0].in_features, 3 * self.params['embedding_dim'] + 1)
        self.assertEqual(b.classifier[-1].out_features, M_GLOBAL)

    def test_dual_stream_classification_matches_r39_effective_path(self):
        from .dual_stream import PerformanceSelector
        torch.manual_seed(2)
        old = PerformanceSelector(**dict(self.params, architecture='performance', decision_head='classification')).eval()
        torch.manual_seed(2)
        new = make_r40_model(dict(self.params, architecture='dual_stream')).eval()
        for key, value in new.state_dict().items():
            torch.testing.assert_close(value, old.state_dict()[key], atol=0, rtol=0)
        for p in ('TSP', 'CVRP', 'ATSP'):
            batch = sample_batch(p)
            torch.testing.assert_close(old(batch)['logits'], new(batch)['logits'], atol=0, rtol=0)

    def test_all_tasks_are_label_free_and_pool_ordered(self):
        model = DirectSelector(**self.params).eval()
        for p in PROBLEMS:
            batch = sample_batch(p)
            expected = model(batch)['logits']
            self.assertEqual(expected.shape, batch['costs'].shape)
            clean = {k: v for k, v in batch.items() if k != 'costs'}
            torch.testing.assert_close(model(clean)['logits'], expected)
            perm = torch.randperm(len(batch['pool_ids']))
            changed = model(dict(batch, pool_ids=batch['pool_ids'][perm]))['logits']
            torch.testing.assert_close(changed, expected[:, perm])

    def test_padding_batch_isolation_and_candidate_masks(self):
        model = DirectSelector(**self.params).eval()
        for p in ('CVRP', 'ATSP'):
            batch = sample_batch(p)
            original = model(batch)['logits']
            padded = copy.deepcopy(batch)
            padded['node'] = F.pad(batch['node'], (0, 0, 0, 3), value=500.)
            padded['matrix'] = F.pad(batch['matrix'], (0, 3, 0, 3), value=500.)
            padded['node_mask'] = F.pad(batch['node_mask'], (0, 3), value=False)
            torch.testing.assert_close(model(padded)['logits'], original, atol=2e-6, rtol=2e-5)
            changed = copy.deepcopy(batch)
            changed['node'][1] *= 20
            changed['matrix'][1] *= 20
            torch.testing.assert_close(model(changed)['logits'][0], original[0])
            ids = torch.cat([batch['pool_ids'], torch.tensor([-1])])
            mask = torch.ones(2, len(ids), dtype=torch.bool)
            mask[:, -1] = False
            out = model(dict(batch, pool_ids=ids, solver_mask=mask))['logits']
            torch.testing.assert_close(out[:, :-1], original)
            self.assertTrue(torch.isneginf(out[:, -1]).all())
            with self.assertRaises(ValueError):
                model(dict(batch, solver_mask=torch.zeros(2, len(batch['pool_ids']), dtype=torch.bool)))

    def test_same_loss_and_one_optimizer_update_per_real_batch(self):
        from .multitask_probe import update_batch
        for group in ('A', 'B'):
            model, _ = initialize(group, 2, self.geometry, 'cpu', dict(self.params, architecture='dual_stream' if group == 'A' else 'direct'))
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            batch = sample_batch('TSP')
            batch['costs'] = torch.rand_like(batch['costs']) + 1
            batch['ind'] = torch.tensor([0, 1])
            with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                values = update_batch(model, optimizer, scaler, batch, fit_args('A'))
                self.assertEqual(step.call_count, 1)
                self.assertEqual(values['skipped'], 0)
            self.assertGreater(sum(p.grad.abs().sum() for p in model.instance_encoder.coord_proj.parameters()), 0)
            module = model.classifier if group == 'B' else model.score_head
            self.assertGreater(sum(p.grad.abs().sum() for p in module.parameters()), 0)
            out = model.eval()(batch)
            loss, parts = selector_loss(out, batch['costs'], None, fit_args('A'), winner=batch['ind'])
            torch.testing.assert_close(loss, .35 * parts['ce'] + .1 * parts['pair'] + .02 * parts['risk'])

    def test_subdivision_preserves_samples_and_balanced_task_order(self):
        batches = {p: torch.arange(32).reshape(2, 2, 8) for p in PROBLEMS}
        original = dict(batches=batches, orders=[list(PROBLEMS) * 2] * 2)
        divided = subdivide_schedule(original, 2)
        for p in PROBLEMS:
            self.assertEqual(divided['batches'][p].shape, (2, 8, 2))
            torch.testing.assert_close(divided['batches'][p].flatten(), batches[p].flatten())
        for order in divided['orders']:
            for start in range(0, len(order), len(PROBLEMS)):
                self.assertEqual(set(order[start:start + len(PROBLEMS)]), set(PROBLEMS))
        with self.assertRaises(ValueError):
            subdivide_schedule(original, 3)

    def test_successful_update_lr_schedule(self):
        self.assertAlmostEqual(learning_rate(0, 162000), 2e-4 / 8100)
        self.assertAlmostEqual(learning_rate(8099, 162000), 2e-4)
        self.assertAlmostEqual(learning_rate(161999, 162000), 2e-6)
        self.assertGreater(learning_rate(90000, 162000), learning_rate(120000, 162000))

    def test_metrics_keep_native_winners_original_costs_and_topk(self):
        raw = dict(costs=np.array([[1e8, 1e8 + .01], [2., 1.]], dtype=np.float64),
                   winner=np.array([0, 1]), pool=['first', 'second'])
        result, pred = classification_metrics(np.array([[0., 1.], [1., 0.]], dtype=np.float32), raw)
        self.assertEqual(result['top1'], 0.)
        self.assertEqual(result['top2'], 1.)
        self.assertEqual(result['top3'], 1.)
        self.assertEqual(result['mean_cost'], (1e8 + .01 + 2.) / 2)
        np.testing.assert_array_equal(pred, [1, 0])


if __name__ == '__main__':
    unittest.main()
