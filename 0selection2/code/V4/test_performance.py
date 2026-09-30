import copy
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .V4Model import make_selector, score_components
from .performance_evaluation import decision_metrics, metrics_from_outputs
from .performance_experiment import epoch_lr, paired_schedule, snapshot
from .performance_targets import center_valid, fit_scale, fit_training_scales, make_targets, performance_mse
from .test_dual_stream import params
from .test_solver_features import sample_batch
from .train import selector_loss


class TargetTests(unittest.TestCase):
    def test_fp64_subtraction_before_conversion(self):
        costs = np.array([[1e8, 1e8+.01, 1e8+.03]], dtype=np.float64)
        self.assertEqual(np.unique(costs.astype(np.float32)).size, 1)
        target = make_targets(costs, fit_scale(costs))
        self.assertGreater(float(target.abs().max()), 1)
        self.assertAlmostEqual(float(target.square().mean()), 1., places=6)
        with self.assertRaises(ValueError):
            fit_scale(costs.astype(np.float32))

    def test_zero_scale_and_common_shift_invariance(self):
        zero = np.full((2, 3), 8., dtype=np.float64)
        self.assertEqual(fit_scale(zero), 1.)
        torch.testing.assert_close(make_targets(zero, 1), torch.zeros(2, 3))
        costs = np.array([[2., 3., 8.], [4., 7., 5.]], dtype=np.float64)
        shifted = costs + np.array([[23.], [17.]])
        self.assertAlmostEqual(fit_scale(costs), fit_scale(shifted))
        torch.testing.assert_close(make_targets(costs, fit_scale(costs)), make_targets(shifted, fit_scale(costs)))

    def test_masked_centering_and_per_instance_mse(self):
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        pred = torch.tensor([[1., -1., float('inf'), float('inf')], [2., -2., 2., -2.]], requires_grad=True)
        target = torch.zeros_like(pred)
        loss = performance_mse(pred, target, mask)
        self.assertEqual(float(loss), 2.5)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue((pred.grad[~mask] == 0).all())
        costs = np.array([[2., 4., np.inf, np.inf], [3., 4., 5., 6.]], dtype=np.float64)
        self.assertAlmostEqual(fit_scale(costs, mask.numpy())**2, 1.125)
        self.assertTrue(torch.isfinite(make_targets(costs, 1., mask.numpy())).all())

    def test_scales_are_fitted_on_train_only(self):
        entry = dict(costs=np.array([[1., 3.]], dtype=np.float64), winner=np.array([0]),
                     pool=['x', 'y'], pool_ids=[0, 1], label_hash='a', data_hash='b')
        with patch('code.V4.performance_targets.read_raw_costs', return_value=entry) as reader:
            _, scales = fit_training_scales(['TSP', 'CVRP'])
        self.assertEqual(reader.call_args_list[0].args, ('TSP', 'train'))
        self.assertEqual(reader.call_args_list[1].args, ('CVRP', 'train'))
        self.assertEqual(scales['TSP']['scale'], 1.)
        self.assertEqual(scales['TSP']['fitted_on'], 'train')


class PerformanceModelTests(unittest.TestCase):
    def setUp(self):
        self.params = dict(params(), architecture='performance', decision_head='performance', local_geometry=True)
        torch.manual_seed(2)
        self.model = make_selector(self.params).eval()

    def test_heads_are_independent_and_cost_free(self):
        batch = sample_batch('TSP')
        batch.pop('costs')
        out = self.model(batch)
        self.assertEqual(out['logits'].shape, out['pred_performance'].shape)
        self.assertFalse({id(p) for p in self.model.score_head.parameters()} & {id(p) for p in self.model.performance_head.parameters()})
        torch.testing.assert_close(out['pred_performance'].mean(1), torch.zeros(2), atol=1e-6, rtol=0)
        for key in ('costs', 'performance_target', 'ind'):
            batch[key] = torch.tensor(float('nan'))
        altered = self.model(batch)
        torch.testing.assert_close(altered['logits'], out['logits'])
        torch.testing.assert_close(altered['pred_performance'], out['pred_performance'])

    def test_selection_and_classification_are_distinct(self):
        out = self.model(sample_batch('CVRP'))
        scores = score_components(out)
        torch.testing.assert_close(scores['final'], -out['pred_performance'])
        torch.testing.assert_close(scores['classification'], out['logits'])
        self.model.decision_head = 'classification'
        out = self.model(sample_batch('CVRP'))
        torch.testing.assert_close(out['selection_scores'], out['logits'])

    def test_candidate_padding_and_permutation(self):
        batch = sample_batch('CVRP')
        original = self.model(batch)
        order = torch.randperm(len(batch['pool_ids']))
        permuted = self.model(dict(batch, pool_ids=batch['pool_ids'][order]))
        for key in ('logits', 'pred_performance'):
            torch.testing.assert_close(permuted[key], original[key][:, order], atol=2e-6, rtol=2e-5)
        ids = torch.cat([batch['pool_ids'], torch.tensor([-1, -1])])
        mask = torch.ones(2, len(ids), dtype=torch.bool)
        mask[:, -2:] = False
        padded = self.model(dict(batch, pool_ids=ids, solver_mask=mask))
        for key in ('logits', 'pred_performance'):
            torch.testing.assert_close(padded[key][:, :-2], original[key], atol=2e-6, rtol=2e-5)
        self.assertTrue(torch.isposinf(padded['pred_performance'][:, -2:]).all())
        self.assertTrue(torch.isneginf(padded['selection_scores'][:, -2:]).all())

    def test_gradients_reach_shared_path_and_both_heads(self):
        batch = sample_batch('TSP')
        out = self.model(batch)
        winner = torch.tensor([0, 1])
        target = torch.randn_like(out['logits'])
        args = Namespace(loss_mode='performance', ce_weight=.35, performance_weight=.35)
        loss, parts = selector_loss(out, batch['costs'], None, args, winner=winner, performance_target=target)
        loss.backward()
        self.assertEqual(set(parts), {'ce', 'performance'})
        modules = [self.model.instance_encoder.coord_proj, self.model.instance_encoder.geometry_residual.projection[-1],
                   self.model.solver_encoder.feature_mlp, self.model.joint_layers, self.model.pool,
                   self.model.decoder, self.model.score_head, self.model.performance_head]
        for module in modules:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)
        self.assertFalse(self.model.solver_encoder.solver_features.requires_grad)
        with self.assertRaises(ValueError):
            selector_loss(out, batch['costs'], None, args, winner=winner)

    def test_common_initialization_and_rng_consumption(self):
        models = []
        for head in ('classification', 'performance'):
            torch.manual_seed(19)
            models.append(make_selector(dict(self.params, decision_head=head, dropout=.1)).train())
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name])
        states = []
        batch = sample_batch('TSP')
        for model in models:
            torch.manual_seed(27)
            model(batch)
            states.append(torch.get_rng_state())
        torch.testing.assert_close(states[0], states[1])

    def test_legacy_model_still_loads_strictly(self):
        old = make_selector(params()).eval()
        clone = make_selector(params()).eval()
        clone.load_state_dict(copy.deepcopy(old.state_dict()), strict=True)
        batch = sample_batch('TSP')
        torch.testing.assert_close(clone(batch)['logits'], old(batch)['logits'])
        self.assertNotIn('performance_head.0.weight', old.state_dict())


class ProtocolTests(unittest.TestCase):
    def test_padding_infinities_are_excluded_from_metrics(self):
        raw = dict(costs=np.array([[1., 2., np.inf], [3., 1., np.inf]], dtype=np.float64),
                   winner=np.array([0, 1]), pool=['a', 'b', 'padding'])
        mask = np.array([[True, True, False], [True, True, False]])
        logits = np.array([[3., 0., -np.inf], [0., 3., -np.inf]], dtype=np.float32)
        perf = np.array([[-1., 1., np.inf], [1., -1., np.inf]], dtype=np.float32)
        result, _, _ = metrics_from_outputs(logits, perf, raw, 1., 'performance', mask)
        self.assertEqual(result['top1'], 1.)
        self.assertEqual(result['mean_cost'], 1.)
        self.assertTrue(np.isfinite(result['performance_mse']))

    def test_standalone_report_has_problem_identity(self):
        from .evaluate import evaluate_problem, macro_row, write_report
        from .tensor_loader import TensorBatchLoader
        p = dict(params(), architecture='performance', decision_head='performance')
        model = make_selector(p).eval()
        batch = sample_batch('TSP')
        batch['ind'] = torch.tensor([0, 1])
        raw = dict(costs=batch['costs'].double().numpy(), winner=batch['ind'].numpy(),
                   pool=[str(i) for i in range(batch['costs'].size(1))], pool_ids=batch['pool_ids'].tolist())
        loader = TensorBatchLoader(batch, 2, False, False)
        with patch('code.V4.train.make_loader', return_value=loader), patch('code.V4.performance_evaluation.read_raw_costs', return_value=raw):
            result = evaluate_problem(model, 'TSP', 'val', 2, 0, 'cpu')
        self.assertEqual(result['problem'], 'TSP')
        with tempfile.TemporaryDirectory() as d:
            payload = dict(ckpt='fixture', split='val', per_problem={'TSP': result}, all=macro_row({'TSP': result}))
            write_report(payload, Path(d))
            self.assertTrue((Path(d)/'test_summary.md').exists())

    def test_source_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            snapshot(root)
            path = root/'source_launch'/'hashes.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                snapshot(root)
            self.assertEqual(path.read_text(), '{}')

    def test_metrics_do_not_mix_classification_and_performance(self):
        raw = dict(costs=np.array([[1., 2., 3.], [3., 1., 2.]], dtype=np.float64),
                   winner=np.array([0, 1]), pool=['a', 'b', 'c'])
        logits = np.array([[9., 0., 0.], [0., 9., 0.]], dtype=np.float32)
        perf = np.array([[1., 0., -1.], [-1., 1., 0.]], dtype=np.float32)
        result, cp, pp = metrics_from_outputs(logits, perf, raw, 1., 'performance')
        self.assertEqual(result['top1'], 0.)
        self.assertEqual(result['classification']['top1'], 1.)
        self.assertEqual(result['mean_cost'], 3.)
        self.assertEqual(result['classification']['mean_cost'], 1.)
        self.assertAlmostEqual(result['ce'], float(F.cross_entropy(torch.from_numpy(logits), torch.tensor([0, 1]))))
        np.testing.assert_array_equal(cp, [0, 1])
        np.testing.assert_array_equal(pp, [2, 0])

    def test_strict_native_winner_is_not_replaced_with_argmin(self):
        raw = dict(costs=np.array([[1., 1., 2.]], dtype=np.float64), winner=np.array([1]), pool=['a', 'b', 'c'])
        result, _ = decision_metrics(np.array([[3., 1., 0.]]), raw)
        self.assertEqual(result['top1'], 0.)
        self.assertEqual(result['top1_tie_aware'], 1.)

    def test_schedule_pairs_batches_and_balances_tasks(self):
        from code.unified_selector.registry import PROBLEMS
        loaders = {p: Namespace(size=100) for p in PROBLEMS}
        args = Namespace(batch_size=32, epochs=4)
        a, b = paired_schedule(2, args, loaders), paired_schedule(2, args, loaders)
        self.assertEqual(a['orders'], b['orders'])
        for p in PROBLEMS:
            torch.testing.assert_close(a['batches'][p], b['batches'][p])
            self.assertEqual(a['batches'][p].shape, (4, 3, 32))
            self.assertEqual(torch.unique(a['batches'][p][0]).numel(), 96)
            for order in a['orders']:
                self.assertEqual(order.count(p), 3)

    def test_lr_warmup_and_final_floor(self):
        self.assertAlmostEqual(epoch_lr(0, 60, 2e-4), 2e-4/3)
        self.assertAlmostEqual(epoch_lr(2, 60, 2e-4), 2e-4)
        self.assertAlmostEqual(epoch_lr(3, 60, 2e-4), 2e-4)
        self.assertAlmostEqual(epoch_lr(59, 60, 2e-4), 2e-6)


if __name__ == '__main__':
    unittest.main()
