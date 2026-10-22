import unittest

import numpy as np

from . import r103_conditional_signal as m


class ConditionalSignalTests(unittest.TestCase):
    def test_group_identity_singletons_and_reproducibility(self):
        n = np.array([10, 20, 10, 30, 20, 20])
        p, info = m.grouped_permutations(n, 20, 2)
        np.testing.assert_array_equal(p, m.grouped_permutations(n, 20, 2)[0])
        for row in p:
            np.testing.assert_array_equal(np.sort(row), np.arange(len(n)))
            np.testing.assert_array_equal(n[row], n)
            self.assertEqual(row[3], 3)
        self.assertEqual(info['eligible_rows'], 5)
        self.assertEqual(info['singleton_groups'], 1)

    def test_saved_argmax_shuffle_equals_full_vector_shuffle(self):
        logits = np.array([[2., 2., 0.], [0., 3., 2.], [3., 0., 4.], [1., 1., 1.]])
        ids = np.array([9, 4, 11])
        p = np.array([3, 2, 0, 1])
        pred = m.stable_argmax(logits, ids)
        self.assertEqual(pred[0], 1)
        np.testing.assert_array_equal(m.stable_argmax(logits[p], ids), pred[p])
        np.testing.assert_array_equal(np.bincount(pred, minlength=3), np.bincount(pred[p], minlength=3))

    def test_native_winner_not_choice_frequency_and_regret(self):
        costs = np.array([[1., 2.], [2., 1.], [3., 3.]], dtype=np.float64)
        winner = np.array([0, 1, 1])
        row = m.row_metrics(costs, winner, np.array([0, 0, 0]), 0, [1])
        self.assertAlmostEqual(row['top1_pct'], 100 / 3)
        self.assertAlmostEqual(row['actual_regret_pct'], 100 / 3)
        self.assertEqual(row['target_winners'], 2)
        self.assertEqual(row['target_selected'], 0)

    def test_full18_macro_keeps_cost_ratios_per_problem(self):
        rows = [dict.fromkeys(m.FIELDS, float(i)) for i in range(18)]
        for row in rows:
            row.update(n=3, target_winners=1, target_correct=0, target_selected=2,
                       target_false_picks=2, target_regret_contribution_pp=.01)
        result = m.macro(rows)
        self.assertEqual(result['top1_pct'], 8.5)
        self.assertEqual(result['n'], 54)
        self.assertEqual(result['target_winners'], 18)
        with self.assertRaises(ValueError):
            m.macro(rows[:-1])

    def test_matched_lift_difference_is_not_observed_gain(self):
        rows = []
        for model, top1, shuffled, regret, shuffled_regret in (
                ('R89B', 70., 65., .5, 1.), ('R94A', 70., 65., .5, 1.),
                ('R94B', 72., 64., .4, 1.2)):
            rows.append(dict(model=model, problem='ALL', observed_top1_pct=top1,
                permuted_mean_top1_pct=shuffled, observed_actual_regret_pct=regret,
                permuted_mean_actual_regret_pct=shuffled_regret,
                instance_regret_benefit_pp=shuffled_regret-regret))
        row = next(x for x in m.matched_comparisons(rows) if x['reference'] == 'R94A')
        self.assertEqual(row['observed_top1_delta_pp'], 2.)
        self.assertEqual(row['permuted_mean_top1_delta_pp'], -1.)
        self.assertEqual(row['additional_instance_top1_lift_pp'], 3.)
        self.assertAlmostEqual(row['additional_instance_regret_benefit_pp'], .3)


if __name__ == '__main__':
    unittest.main()
