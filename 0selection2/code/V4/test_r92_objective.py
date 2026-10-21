"""Two-thread, synthetic CPU integration checks for the R92 objective."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .pairwise_objective import LABEL_KEYS, comparison_targets, predicted_top3, single_objective
from .r42_experiment import symmetric_kl
from .r76_experiment import PART_KEYS, RDropObjective, Runtime, slice_batch
from .r92_model import make_model
from .r92_objective import HierarchyObjective, update_batch
from .test_r91_objective import ToySelector, fixture as tiny_batch
from .test_r92_model import graph_batch, gradients, model_params


class ZeroAuxiliarySelector(ToySelector):
    """Original tiny score/maximin stub with the A auxiliary-output contract."""

    def forward(self, batch):
        output = super().forward(batch)
        zero = output['utility'].new_zeros(())
        output.update(auxiliary_loss=zero, auxiliary_parts=dict(link=zero, entropy=zero))
        return output


def supervised_graph(problem='ATSP', rows=3, nodes=9):
    batch = graph_batch(problem, rows=rows, nodes=nodes)
    k = batch['pool_ids'].numel()
    costs = np.stack([1. + .025 * np.roll(np.arange(k), row % k)
                      for row in range(rows)]).astype(np.float64)
    costs[0, :2] = [1. + 5e-10, 1.]
    batch.update(comparison_targets(costs), costs=torch.from_numpy(costs.astype(np.float32)),
                 ind=torch.from_numpy(costs.argmin(-1)))
    return batch


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.


class R92ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2)
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_a_zero_aux_matches_original_rdrop_loss_parts_rng_and_every_gradient(self):
        for weight in (0., .175, .35):
            with self.subTest(consistency_weight=weight):
                batch = tiny_batch()
                original = RDropObjective(ZeroAuxiliarySelector()).train()
                hierarchy = HierarchyObjective(copy.deepcopy(original.selector), 'A').train()
                rng = torch.get_rng_state().clone()
                expected_loss, expected_parts = original(batch, weight)
                expected_rng = torch.get_rng_state().clone()
                expected_loss.backward()
                torch.set_rng_state(rng)
                actual_loss, actual_parts = hierarchy(batch, weight)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                actual_loss.backward()
                torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
                torch.testing.assert_close(actual_parts, expected_parts, rtol=0, atol=0)
                self.assertEqual(len(hierarchy.selector.outputs), 2)
                for name, parameter in original.named_parameters():
                    actual = dict(hierarchy.named_parameters())[name]
                    self.assertIsNotNone(parameter.grad)
                    torch.testing.assert_close(actual.grad, parameter.grad, rtol=0, atol=0)
                for left, right in zip(original.selector.outputs, hierarchy.selector.outputs):
                    for key in ('logits', 'utility', 'pair_margin', 'solver_mask'):
                        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                self.assertEqual(set(hierarchy.metrics),
                                 {'hierarchy_auxiliary', 'hierarchy_link', 'hierarchy_entropy'})
                for value in hierarchy.metrics.values():
                    self.assertEqual(value.item(), 0.)
                    self.assertFalse(value.requires_grad)
                self.assertEqual(batch['ind'][0].item(), 1)  # Preserve the supplied native winner.

    def test_b_real_model_two_forward_formula_and_auxiliary_gradients(self):
        for problem in ('TSP', 'ATSP'):
            with self.subTest(problem=problem):
                model = make_model(model_params(), 'B').train()
                objective = HierarchyObjective(model, 'B').train()
                outputs = []

                def input_guard(module, arguments):
                    self.assertFalse(set(LABEL_KEYS) & arguments[0].keys())

                before = model.register_forward_pre_hook(input_guard)
                after = model.register_forward_hook(lambda module, args, output: outputs.append(output))
                try:
                    batch = supervised_graph(problem)
                    loss, parts = objective(batch, .175)
                finally:
                    before.remove()
                    after.remove()
                self.assertEqual(len(outputs), 2)
                first, second = outputs
                focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
                values = [single_objective(output, batch, focus) for output in outputs]
                auxiliary = .5 * (first['auxiliary_loss'] + second['auxiliary_loss'])
                expected = .5 * (values[0][0] + values[1][0]) + .175 * symmetric_kl(
                    first['logits'], second['logits'], first['solver_mask']) + auxiliary
                torch.testing.assert_close(loss, expected, rtol=0, atol=0)
                torch.testing.assert_close(parts[PART_KEYS.index('loss')], loss.detach(), rtol=0, atol=0)
                torch.testing.assert_close(objective.metrics['hierarchy_auxiliary'], auxiliary.detach())
                self.assertGreater(auxiliary.item(), 0.)
                for name in ('link', 'entropy'):
                    expected_part = .5 * sum(output['auxiliary_parts'][name] for output in outputs)
                    torch.testing.assert_close(objective.metrics['hierarchy_' + name], expected_part.detach())
                for network in model.instance_encoder.assignment_nets:
                    auxiliary_grads = torch.autograd.grad(auxiliary, tuple(network.parameters()), retain_graph=True)
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in auxiliary_grads))
                    self.assertGreater(sum(gradient.abs().sum().item() for gradient in auxiliary_grads), 0.)
                loss.backward()
                for module in (*model.instance_encoder.assignment_nets, *model.instance_encoder.blocks,
                               model.score_head):
                    self.assertTrue(gradients(module))
                self.assertFalse(parts.requires_grad)
                self.assertTrue(torch.isfinite(parts).all())
                self.assertTrue(all(not value.requires_grad for value in objective.metrics.values()))

    def test_a_row_weighted_unequal_microbatches_match_full_loss_and_gradients(self):
        batch = tiny_batch()
        full = HierarchyObjective(ZeroAuxiliarySelector(dropout=0.), 'A').train()
        micro = copy.deepcopy(full)
        loss, parts = full(batch, .35)
        loss.backward()
        parts_sum = torch.zeros_like(parts)
        for start, stop in ((0, 2), (2, 4), (4, 5)):
            value, metrics = micro(slice_batch(batch, start, stop), .35)
            fraction = (stop - start) / len(batch['ind'])
            (value * fraction).backward()
            parts_sum += fraction * metrics
        torch.testing.assert_close(parts_sum, parts, rtol=3e-6, atol=3e-7)
        for left, right in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=3e-7)

    def test_b_update_batch_weights_auxiliary_metrics_and_performs_one_step(self):
        batch = supervised_graph(rows=3)
        model = make_model(model_params(), 'B').train()
        disable_dropout(model)  # Compare reductions, not different dropout draws.
        full = HierarchyObjective(model, 'B').train()
        micro = HierarchyObjective(copy.deepcopy(model), 'B').train()
        full_optimizer = torch.optim.SGD(full.parameters(), lr=1e-3)
        micro_optimizer = torch.optim.SGD(micro.parameters(), lr=1e-3)
        full_loss, full_parts = full(batch, .35)
        full_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(full.parameters(), 1.)
        full_optimizer.step()
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        with patch.object(micro_optimizer, 'step', wraps=micro_optimizer.step) as step_call:
            with patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip_call:
                parts, retries, metrics = update_batch(micro, micro_optimizer, scaler, batch, .35,
                    Runtime(0, 1, torch.device('cpu')), cap=2, max_retries=1, problem='ATSP')
        self.assertEqual((retries, step_call.call_count, clip_call.call_count), (0, 1, 1))
        expected_parts = torch.cat([full_parts.double(), norm.detach().double().reshape(1)])
        torch.testing.assert_close(parts, expected_parts, rtol=3e-5, atol=3e-6)
        for name, expected in full.metrics.items():
            torch.testing.assert_close(metrics[name], expected.double(), rtol=3e-5, atol=3e-6)
            self.assertFalse(metrics[name].requires_grad)
        for left, right in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-7)


if __name__ == '__main__':
    unittest.main()
