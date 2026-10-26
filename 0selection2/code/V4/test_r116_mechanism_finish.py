"""Two bounded synthetic CPU checks. No current R116 VAL/TEST or GPU reads."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import r116_finish as frozen
from . import r116_mechanism_finish as supplement


def synthetic():
    old, new, targets = {}, {}, []
    winner = np.array([0, 0, 1, 2, 2, 1], dtype=np.int64)
    costs = np.tile(np.array([3., 5., 7.]), (6, 1))
    costs[np.arange(6), winner] = 1.
    for problem in supplement.common.PROBLEMS:
        data = dict(pool=['target0', 'target1', 'outside'], pool_ids=np.arange(3, dtype=np.int64),
            costs=costs.copy(), winner=winner.copy(), indices=np.arange(6),
            source_indices=np.arange(6), release_source_indices=np.arange(6))
        old[problem] = dict(data, pred=np.array([2, 0, 2, 2, 0, 1], dtype=np.int64))
        new[problem] = dict(data, pred=np.array([0, 2, 1, 1, 1, 1], dtype=np.int64))
        targets.extend((problem, method) for method in ('target0', 'target1'))
    return old, new, targets


class R116MechanismTests(unittest.TestCase):
    def test_native_target_metrics_and_additive_full18_partition(self):
        old, new, targets = synthetic()
        details, summary = supplement.recovery_tables(old, new, targets, 'B vs A', 'TEST')
        self.assertEqual(len(details), 36)
        target = next(r for r in summary if r['problem'] == 'TSP' and r['solver'] == 'FIXED36')
        self.assertEqual((target['baseline_true_tp'], target['current_true_tp']), (2, 3))
        self.assertEqual((target['baseline_selected_count'], target['selected_count']), (3, 5))
        self.assertEqual((target['true_rescues'], target['lost_true_picks'], target['new_false_pick_count']), (2, 1, 2))
        self.assertEqual(target['baseline_recall_pct'], 50.)
        self.assertEqual(target['current_recall_pct'], 75.)
        self.assertAlmostEqual(target['baseline_selection_precision_pct'], 200. / 3)
        self.assertEqual(target['current_selection_precision_pct'], 60.)
        for problem in (*supplement.common.PROBLEMS, 'ALL'):
            rows = {r['solver']: r for r in summary if r['problem'] == problem}
            for key in supplement.CONTRIBUTIONS:
                self.assertAlmostEqual(rows['FIXED36'][key] + rows['OUTSIDE36'][key], rows['FULL18'][key], places=12)
        all_fixed = next(r for r in summary if r['problem'] == 'ALL' and r['solver'] == 'FIXED36')
        self.assertEqual(all_fixed['current_true_tp'], 3 * 18)
        self.assertEqual(all_fixed['current_selection_precision_pct'], 60.)
        altered = dict(new, TSP=dict(new['TSP'], costs=new['TSP']['costs'].copy()))
        altered['TSP']['costs'][0, 0] += .01
        with self.assertRaises(AssertionError):
            supplement.recovery_tables(old, altered, targets, 'B vs A', 'VAL')

    def test_private_namespace_and_hard_receipts_precede_any_archive_read(self):
        original_callbacks = sys.modules['code.V4._r116_finish_callbacks']
        snapshot = (frozen.OUTPUTS, frozen.CODE_NAMES, frozen._callbacks, frozen._callbacks.render,
                    frozen._callbacks.completed, supplement.common.load_run, supplement.recovery.target_recovery)
        private = supplement.private_finisher()
        self.assertIsNot(private, frozen)
        self.assertIsNot(private._callbacks, original_callbacks)
        self.assertIs(sys.modules['code.V4._r116_finish_callbacks'], original_callbacks)
        self.assertEqual(private.OUTPUTS, (*frozen.OUTPUTS, *supplement.EXTRA_OUTPUTS))
        self.assertEqual(private.CODE_NAMES, (*frozen.CODE_NAMES, *supplement.EXTRA_CODE))
        self.assertEqual(private._callbacks.__file__, str(supplement.FROZEN_FINISHER))
        self.assertIs(private.publish.__globals__, private._callbacks.__dict__)
        self.assertEqual(snapshot, (frozen.OUTPUTS, frozen.CODE_NAMES, frozen._callbacks, frozen._callbacks.render,
            frozen._callbacks.completed, supplement.common.load_run, supplement.recovery.target_recovery))
        with tempfile.TemporaryDirectory() as directory, patch.object(np, 'load', side_effect=AssertionError('No TEST reads')):
            with self.assertRaises(supplement.common.NotReady):
                private.analyze(Path(directory))
            self.assertFalse((Path(directory) / 'finish_complete.json').exists())


if __name__ == '__main__':
    unittest.main()
