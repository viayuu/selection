"""Brief CPU checks for adopted projection, complete paired coverage, and integer ties."""

import random
import unittest

import torch

from code.unified_selector.registry import PROBLEMS
from .local_geometry import FIELDS
from .r64_experiment import batch_protocol, paired_triples, prefer_checkpoint, triple_indices
from .r64_gradients import GradientLayout, SHARED_PREFIXES, project_shared, reduce_gradients
from .r64_model import initialize


class R64Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_projection_uses_original_opponents_and_zero_denominator(self):
        original = torch.tensor([[1., 0., 1.], [-2., 1., 0.], [1., -2., 0.]])
        saved, expected, rng = original.clone(), [], random.Random(6402)
        expected_orders = []
        for i in range(3):
            v = original[i].clone()
            order = [j for j in range(3) if j != i]
            rng.shuffle(order)
            expected_orders.append(order)
            for j in order:
                dot = float(v @ saved[j])
                denominator = float(saved[j] @ saved[j])
                if denominator and dot < 0:
                    v -= dot / denominator * saved[j]
            expected.append(v)
        actual, diagnostics = project_shared(original, random.Random(6402))
        torch.testing.assert_close(actual, torch.stack(expected).mean(0))
        torch.testing.assert_close(original, saved, rtol=0, atol=0)
        self.assertEqual(diagnostics['projection_orders'], expected_orders)
        self.assertGreater(diagnostics['conflict_pairs'], 0)
        zero = torch.tensor([[0., 0.], [1., 0.], [-1., 0.]])
        actual, diagnostics = project_shared(zero, random.Random(6402))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(diagnostics['comparable_pairs'], 1)

    def test_shared_prefixes_exist_and_other_parameters_use_mean(self):
        geometry = dict(mean=[0.]*len(FIELDS), std=[1.]*len(FIELDS))
        model, _ = initialize(geometry)
        layout = GradientLayout(model)
        self.assertEqual(set(layout.receipt['matched_parameters']), set(SHARED_PREFIXES))
        self.assertTrue(all(layout.receipt['matched_parameters'].values()))
        self.assertIn('solver_encoder.solver_emb.weight', layout.receipt['ordinary_mean_parameters'])
        original = torch.tensor([[1., 2., 8., -1.], [-1., 2., 4., 3.], [1., -2., 6., 2.]])
        mask = torch.tensor([True, True, False, False])
        mean, _ = reduce_gradients(original, mask, random.Random(6402), False)
        pcgrad, _ = reduce_gradients(original, mask, random.Random(6402), True)
        torch.testing.assert_close(mean, original.mean(0), rtol=0, atol=0)
        torch.testing.assert_close(pcgrad[~mask], original.mean(0)[~mask], rtol=0, atol=0)
        self.assertGreater(float((pcgrad[mask]-mean[mask]).norm()), 0.)

    def test_complete_paired_cycle_coverage_and_larger_real_batch_tails(self):
        sizes = {p: 10000 for p in PROBLEMS}
        for batch_size in (128, 192, 256, 384, 512, 1024):
            with self.subTest(batch_size=batch_size):
                protocol = batch_protocol(batch_size)
                epochs = 40 if batch_size == 256 else 1
                schedule = paired_triples(sizes, epochs=epochs, batch_size=batch_size)
                paired = paired_triples(sizes, epochs=1, batch_size=batch_size)
                total_presentations = 0
                for epoch in range(epochs):
                    self.assertEqual(len(schedule['cycles'][epoch]), protocol['cycles_per_epoch'])
                    self.assertEqual(len(schedule['triples'][epoch]), protocol['updates_per_epoch'])
                    for cycle in schedule['cycles'][epoch]:
                        self.assertEqual(len(cycle), 18)
                        self.assertEqual(set(cycle), set(PROBLEMS))
                    consumed = {p: [] for p in PROBLEMS}
                    rows = 0
                    for step in range(protocol['updates_per_epoch']):
                        triple = triple_indices(schedule, epoch, step)
                        self.assertEqual(len({p for p, _ in triple}), 3)
                        expected = batch_size if step < protocol['updates_per_epoch']-6 else protocol['tail_per_task']
                        self.assertEqual([len(indices) for _, indices in triple], [expected]*3)
                        rows += sum(len(indices) for _, indices in triple)
                        for task, indices in triple:
                            consumed[task].append(indices)
                    for task, parts in consumed.items():
                        actual = torch.cat(parts)
                        torch.testing.assert_close(actual, schedule['permutations'][epoch][task], rtol=0, atol=0)
                        torch.testing.assert_close(actual.sort().values, torch.arange(10000), rtol=0, atol=0)
                    self.assertEqual(rows, 180000)
                    total_presentations += rows
                self.assertEqual(schedule['triples'][0], paired['triples'][0])
                self.assertEqual(total_presentations, epochs*180000)
                self.assertEqual(protocol['warmup_updates'], 3*protocol['updates_per_epoch'])
                self.assertEqual(protocol['maximum_updates_per_arm'], 40*protocol['updates_per_epoch'])
        self.assertEqual((batch_protocol(256)['updates_per_epoch'], batch_protocol(256)['tail_per_task']), (240, 16))
        self.assertEqual((batch_protocol(512)['updates_per_epoch'], batch_protocol(512)['tail_per_task']), (120, 272))
        self.assertEqual((batch_protocol(1024)['updates_per_epoch'], batch_protocol(1024)['tail_per_task']), (60, 784))

    def test_exact_integer_ties_retain_earlier_epoch_never_cost(self):
        earlier = dict(native_correct=8729, native_total=18000, macro_top1=.4849444444444444, mean_cost=12.)
        later = dict(native_correct=8729, native_total=18000, macro_top1=.4849444444444445, mean_cost=10.)
        self.assertTrue(prefer_checkpoint(earlier, None))
        self.assertFalse(prefer_checkpoint(later, earlier))
        self.assertTrue(prefer_checkpoint(dict(later, native_correct=8730, mean_cost=20.), earlier))
        with self.assertRaises(ValueError):
            prefer_checkpoint(dict(later, native_correct=8729.), earlier)


if __name__ == '__main__':
    unittest.main()
