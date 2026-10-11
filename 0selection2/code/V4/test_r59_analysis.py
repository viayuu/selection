"""Focused CPU checks for validation replay and equal-task reporting."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from .r59_analysis import advancement_screen, aggregate, changes, replay_prediction


class R59AnalysisTests(unittest.TestCase):
    def prediction(self, pred, costs, winner):
        costs = np.asarray(costs, dtype=np.float64)
        scores = np.zeros_like(costs, dtype=np.float32)
        scores[np.arange(len(costs)), pred] = 1.
        return dict(costs=costs, winner=np.asarray(winner), pred=np.asarray(pred),
                    logits=scores, indices=np.arange(len(costs)),
                    pool=np.asarray(['solver_' + str(i) for i in range(costs.shape[1])]),
                    pool_ids=np.arange(costs.shape[1]),
                    nodes=np.full(len(costs), 50))

    def test_saved_predictions_replay_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'val.npz'
            saved = self.prediction([0, 1], [[1., 2.], [2., 1.]], [0, 1])
            np.savez_compressed(path, **saved)
            metrics, _ = replay_prediction(path)
            self.assertEqual(metrics['top1'], 1.)
            self.assertEqual(metrics['actual_regret_pct'], 0.)
            saved['pred'] = np.asarray([1, 1])
            np.savez_compressed(path, **saved)
            with self.assertRaises(AssertionError):
                replay_prediction(path)

    def test_fp32_cost_storage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = self.prediction([0], [[1., 2.]], [0])
            saved['costs'] = saved['costs'].astype(np.float32)
            path = Path(temporary) / 'val.npz'
            np.savez_compressed(path, **saved)
            with self.assertRaises(ValueError):
                replay_prediction(path)

    def test_screen_requires_both_accuracy_gains_and_cost_nonregression(self):
        reference = dict(macro_top1=.4911666666666667, macro_actual_regret_pct=.926048)
        random = dict(macro_top1=.50, macro_actual_regret_pct=.90)
        pretrained = dict(macro_top1=.53, macro_actual_regret_pct=.89)
        self.assertTrue(advancement_screen(reference, random, pretrained)['passed'])
        self.assertFalse(advancement_screen(reference, dict(random, macro_top1=.52), pretrained)['passed'])
        self.assertFalse(advancement_screen(reference, random, dict(pretrained, macro_actual_regret_pct=.91))['passed'])
        self.assertFalse(advancement_screen(reference, random, dict(pretrained, macro_top1=.52))['passed'])

    def test_equal_task_macro_not_instance_micro(self):
        from .r59_analysis import FIELDS
        per = {'one': {key: 1. for key in FIELDS}, 'many': {key: 0. for key in FIELDS}}
        per['one']['n'], per['many']['n'] = 1, 100
        self.assertEqual(aggregate(per)['macro_top1'], .5)

    def test_changes_use_per_problem_weights(self):
        before = dict(first=self.prediction([1], [[1., 2.]], [0]),
                      second=self.prediction([0, 0, 0], [[10., 1.]] * 3, [1, 1, 1]))
        after = dict(first=self.prediction([0], [[1., 2.]], [0]),
                     second=self.prediction([1, 1, 1], [[10., 1.]] * 3, [1, 1, 1]))
        rows = changes(before, after, 'before', 'after')
        total = next(row for row in rows if row['problem'] == 'ALL' and row['category'] == 'all')
        corrected = next(row for row in rows if row['problem'] == 'ALL' and row['category'] == 'corrected')
        self.assertEqual(corrected['n'], 4)
        self.assertEqual(total['mean_cost_delta'], -5.)
        self.assertEqual(total['actual_regret_delta_pct'], -500.)

    def test_changes_include_cost_changes_when_both_predictions_are_wrong(self):
        costs = [[1., 2., 3.], [1., 2., 3.], [1., 2., 2.]]
        before = dict(task=self.prediction([2, 1, 1], costs, [0, 0, 0]))
        after = dict(task=self.prediction([1, 2, 2], costs, [0, 0, 0]))
        rows = changes(before, after, 'before', 'after')
        totals = {row['category']: row for row in rows if row['problem'] == 'ALL'}
        for category in ('both_wrong_cheaper', 'both_wrong_dearer', 'both_wrong_equal'):
            self.assertEqual(totals[category]['n'], 1)
        self.assertEqual(totals['corrected']['n'], 0)
        self.assertEqual(totals['harmed']['n'], 0)
        self.assertAlmostEqual(totals['both_wrong_cheaper']['mean_cost_delta'], -1. / 3)
        self.assertAlmostEqual(totals['both_wrong_dearer']['actual_regret_delta_pct'], 100. / 3)
        self.assertAlmostEqual(totals['all']['actual_regret_delta_pct'], 0.)


if __name__ == '__main__':
    unittest.main()
