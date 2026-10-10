"""Focused checks for macro accounting, exact costs and start-subset interpretation."""

import csv
import json
import unittest

import numpy as np

from .r52_analysis import start_diagnostics, winner_of
from .r52_error_budget import (CATEGORIES, ROOT, correction_bounds, error_summary,
                               group_errors, instance_records, ordered_predictions)


class ErrorBudgetTests(unittest.TestCase):
    def test_global_id_tie_break(self):
        order = ordered_predictions(np.array([[1., 1., 0.], [0., 2., 2.]]), [8, 2, 5])
        np.testing.assert_array_equal(order, [[1, 0, 2], [1, 2, 0]])

    def test_rank_partition_is_exhaustive(self):
        costs = np.array([[1., 2., 3., 4.]] * 4)
        order = np.array([[0, 1, 2, 3], [1, 0, 2, 3], [1, 2, 0, 3], [1, 2, 3, 0]])
        rows = instance_records('P', costs, np.zeros(4, dtype=int), order, list('ABCD'), 1)
        self.assertEqual([r['category'] for r in rows], ['correct', *CATEGORIES])
        self.assertEqual(sum(r['category'] != 'correct' for r in rows), 3)
        self.assertAlmostEqual(sum(error_summary([r for r in rows if r['category'] == c], rows,
            'ALL')['share_of_total_regret_pct'] for c in CATEGORIES), 100.)

    def test_equal_problem_weight_with_unequal_sizes(self):
        one = instance_records('small', np.array([[1., 2.]]), np.array([0]),
                               np.array([[1, 0]]), ['A', 'B'], 2)
        many = instance_records('large', np.array([[1., 2.]] * 10), np.zeros(10, dtype=int),
                                np.array([[0, 1]] * 10), ['A', 'B'], 2)
        summary = error_summary(one, one + many, 'ALL')
        self.assertAlmostEqual(summary['macro_top1_recovery_pp'], 50.)
        self.assertAlmostEqual(summary['macro_regret_contribution_pp'], 50.)

    def test_loss_ranking_is_not_error_frequency(self):
        rows = instance_records('P', np.array([[1., 2., 1.001]] * 3), np.zeros(3, dtype=int),
            np.array([[1, 0, 2], [2, 0, 1], [2, 0, 1]]), ['A', 'B', 'C'], 1)
        ranked = group_errors(rows, ('problem', 'selected', 'winner'))
        self.assertEqual(ranked[0]['selected'], 'B')
        self.assertEqual(ranked[0]['count'], 1)

    def test_correction_retains_native_winner_residual(self):
        rows = instance_records('P', np.array([[1., 1.1, 2.]]), np.array([1]),
                                np.array([[2, 1, 0]]), ['A', 'B', 'C'], 1)
        ranked = group_errors(rows, ('problem', 'selected', 'winner'))
        bound = correction_bounds(rows, ranked, ('problem', 'selected', 'winner'))[0]
        self.assertAlmostEqual(bound['diagnostic_macro_top1_pct'], 100.)
        self.assertAlmostEqual(bound['diagnostic_macro_actual_regret_pct'], 10.)

    def test_original_fp64_difference_not_float32_tie(self):
        costs = np.array([[1., 1. + 1e-10]])
        self.assertEqual(int(winner_of(costs)[0]), 0)
        self.assertEqual(int(winner_of(costs.astype(np.float32))[0]), -1)


class StartMechanismTests(unittest.TestCase):
    def test_single_exceptional_start_and_opposite_median(self):
        d = start_diagnostics(np.array([[1., 5., 5., 5., 5.], [2., 2., 2., 2., 2.]]), 1)
        self.assertEqual(d['min_winner'], 0)
        self.assertEqual(d['support_count'], 1)
        self.assertEqual(d['quantile_winners'][2], 1)

    def test_broad_advantage_does_not_flip(self):
        d = start_diagnostics(np.array([[1.] * 20, [2.] * 20]), 2)
        self.assertEqual(d['support_count'], 20)
        self.assertEqual(d['strict_flip_count'], 0)

    def test_exact_ties_are_not_strict_reversals(self):
        d = start_diagnostics(np.array([[1.] * 5, [1.] * 5]), 4)
        self.assertEqual(d['min_winner'], -1)
        self.assertEqual(d['strict_flip_count'], 0)
        self.assertEqual(d['subset_tie_count'], 20)

    def test_shared_subsets_match_recorded_minima(self):
        costs = np.array([[1., 5., 5., 5., 5.], [2., 2., 2., 2., 2.]])
        d = start_diagnostics(costs, 23)
        self.assertTrue((d['retained_masks'].sum(1) == 4).all())
        expected = [winner_of(costs[:, mask].min(1)) for mask in d['retained_masks']]
        np.testing.assert_array_equal(d['subset_winners'], expected)
        self.assertGreater(d['strict_flip_count'], 0)

    def test_replay_is_per_original_index_deterministic(self):
        costs = np.array([[1., 3., 4., 2., 3.], [2., 3., 2., 4., 5.]])
        first, second = start_diagnostics(costs, 99), start_diagnostics(costs, 99)
        np.testing.assert_array_equal(first['retained_masks'], second['retained_masks'])
        other = start_diagnostics(costs[::-1], 99)
        np.testing.assert_array_equal(first['retained_masks'], other['retained_masks'])
        self.assertEqual(first['strict_flip_count'], other['strict_flip_count'])
        np.testing.assert_array_equal(other['subset_winners'],
            np.where(first['subset_winners'] < 0, -1, 1 - first['subset_winners']))

    def test_subset_tie_is_separate_from_flip(self):
        d = start_diagnostics(np.array([[1., 2., 2., 2., 2.], [2., 2., 2., 2., 2.]]), 8)
        self.assertEqual(d['strict_flip_count'], 0)
        self.assertGreater(d['subset_tie_count'], 0)


class SavedArtifactTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'winner_mechanism.csv').exists(), 'Run the actual replay/analysis first')
    def test_all_saved_start_metrics_are_recomputable(self):
        with (ROOT / 'winner_mechanism.csv').open() as stream:
            rows = {int(r['index']): r for r in csv.DictReader(stream)}
        with np.load(ROOT / 'start_costs.npz', allow_pickle=False) as costs, \
                np.load(ROOT / 'start_subset_diagnostics.npz', allow_pickle=False) as subsets:
            costs = {key: costs[key] for key in costs.files}
            subsets = {key: subsets[key] for key in subsets.files}
            np.testing.assert_array_equal(costs['indices'], subsets['indices'])
            self.assertEqual(len(rows), 1000)
            for i, index in enumerate(costs['indices']):
                mask = costs['valid_start_mask'][i]
                values = costs['costs'][i][:, mask]
                d = start_diagnostics(values, index)
                saved = rows[int(index)]
                self.assertEqual(d['support_count'], int(saved['winner_support_count']))
                self.assertEqual(d['strict_flip_count'], int(saved['subset_strict_flip_count']))
                np.testing.assert_array_equal(d['subset_winners'], subsets['subset_winners'][i])
                np.testing.assert_array_equal(d['retained_masks'], subsets['retained_start_mask'][i][:, mask])
                np.testing.assert_allclose(values.min(1), costs['historical_costs'][i], atol=5e-5, rtol=2e-6)
                self.assertFalse(subsets['retained_start_mask'][i][:, ~mask].any())

    @unittest.skipUnless((ROOT / 'topk_error_decomposition.csv').exists(), 'Run error budget first')
    def test_saved_18_task_partition_and_macro_budget(self):
        with (ROOT / 'topk_error_decomposition.csv').open() as stream:
            rows = [r for r in csv.DictReader(stream) if r['scope'] == 'ALL']
        baseline = json.loads((ROOT / 'frozen_baseline.json').read_text())
        self.assertEqual(sum(int(r['count']) for r in rows), 9159)
        self.assertAlmostEqual(sum(float(r['share_of_errors_pct']) for r in rows), 100., places=9)
        self.assertAlmostEqual(sum(float(r['macro_regret_contribution_pp']) for r in rows),
                               baseline['metrics']['macro_actual_regret_pct'], places=11)
        self.assertFalse(baseline['test_read'])


if __name__ == '__main__':
    unittest.main()
