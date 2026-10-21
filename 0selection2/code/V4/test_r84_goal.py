"""Synthetic CPU checks for target freezing, paired arithmetic and gating."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import r84_goal as goal


def data(pred, winner=(0, 1, 1, 2)):
    return goal.Prediction(['a', 'b', 'c'], np.array([1, 3, 7]),
        np.array([[1., 2., 3.], [2., 1., 4.], [3., 1., 2.], [3., 2., 1.]], np.float64),
        np.array(winner, np.int64), np.array(pred, np.int64), np.arange(4), np.arange(100, 104))


class GoalTests(unittest.TestCase):
    def test_status_and_watch_wait_without_numpy_torch_or_prediction_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = ("import sys; from pathlib import Path; from code.V4 import r84_goal as g; "
                "p=Path(sys.argv[1]); s=g.status(p,p/'launch'); "
                "assert s['phase']=='waiting_for_arms' and not s['goal_complete']; "
                "assert g.watch(p,p/'launch',max_polls=1)==0; "
                "assert 'torch' not in sys.modules and 'numpy' not in sys.modules")
            subprocess.run([sys.executable, '-B', '-c', code, str(root)], check=True)
            self.assertFalse((root / 'comparison_artifacts').exists())

    def test_paired_errors_and_costs_are_not_just_selection_frequency(self):
        baseline, current = data([0, 0, 2, 1]), data([1, 1, 0, 0])
        result = goal.paired(baseline, current)
        self.assertEqual((result['corrected_n'], result['harmed_n'], result['both_wrong_n']), (1, 1, 2))
        self.assertEqual(result['both_wrong_dearer_n'], 2)
        self.assertEqual(result['corrected_raw_cost_delta_sum'], -1.)
        self.assertEqual(result['harmed_raw_cost_delta_sum'], 1.)
        self.assertAlmostEqual(result['net_mean_cost_delta'], .5)
        self.assertAlmostEqual(result['net_regret_delta_pp'], 50.)
        row = goal.method_metrics(baseline, current, 1)
        self.assertEqual((row['winner_n'], row['selected_n'], row['true_positive_n']), (2, 2, 1))
        self.assertEqual((row['rescued_winner_n'], row['new_false_picks_n']), (1, 1))
        self.assertEqual((row['recall_pct'], row['precision_pct']), (50., 50.))
        self.assertAlmostEqual(row['new_false_pick_regret_contribution_pp'], 25.)

    def test_matched_inputs_and_saved_global_id_tie_rule(self):
        reference = data([0, 0, 2, 1])
        other = data([0, 0, 2, 1])
        other.costs = other.costs.copy()
        other.costs[0, 1] += .01
        with self.assertRaises(AssertionError):
            goal.paired(reference, other)
        archive = dict(TSP__pool=np.array(['a', 'b', 'c']), TSP__pool_ids=np.array([7, 1, 3]),
            TSP__costs=reference.costs, TSP__winner=reference.winner,
            TSP__logits=np.zeros((4, 3)), TSP__indices=np.arange(4))
        config = dict(candidate_pools={'TSP': ['a', 'b', 'c']}, candidate_pool_ids={'TSP': [7, 1, 3]})
        with patch.object(goal, 'ROWS', dict(train=81000, val=4, test=4)):
            result = goal.load_prediction(archive, 'TSP', config, 'VAL')
            np.testing.assert_array_equal(result.pred, np.ones(4))
            archive['TSP__pred'] = np.zeros(4, np.int64)
            with self.assertRaises(AssertionError):
                goal.load_prediction(archive, 'TSP', config, 'VAL')

    def test_gate_uses_unrounded_val_and_cannot_trade_cost_for_top1(self):
        baseline = dict(top1_pct=50., mean_cost=1., actual_regret_pct=.2)
        better = dict(top1_pct=50.0000000001, mean_cost=1., actual_regret_pct=.2)
        self.assertTrue(goal.val_acceptance(better, baseline))
        self.assertFalse(goal.val_acceptance(dict(better, mean_cost=1.0000000001), baseline))
        self.assertFalse(goal.val_acceptance(dict(better, actual_regret_pct=.2000000001), baseline))
        self.assertFalse(goal.val_acceptance(baseline, baseline))

    def test_failure_is_actionable_and_does_not_retry_or_read_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            goal.save(root / 'C_focal/status.json', dict(phase='failed', error='finite gradient check failed'))
            with patch.object(goal, 'report') as report, patch.object(goal, 'freeze_targets') as freeze:
                self.assertEqual(goal.watch(root, root / 'launch', max_polls=2), 1)
            report.assert_not_called()
            freeze.assert_not_called()
            status = goal.read(root / 'goal_progress.json')
            self.assertEqual(status['next_action'], 'manual_failure_review')
            self.assertFalse(status['automatic_retry'])
            self.assertIn('Inspect', status['errors'][0]['action'])


class CompletedReportTests(unittest.TestCase):
    def fixture(self, root):
        costs = data([0, 0, 2, 1]).costs
        for arm in goal.ARMS:
            run = root / arm
            config = dict(experiment='R84', arm=arm, seed=2, logical_global_batch=1024,
                max_successful_updates=goal.BUDGET, rows_per_problem=goal.ROWS, data_root='synthetic-new90k',
                candidate_pools={p: ['a', 'b', 'c'] for p in goal.PROBLEMS},
                candidate_pool_ids={p: [1, 3, 7] for p in goal.PROBLEMS},
                baselines={p: dict(selection_split='train', sbs_solver='b') for p in goal.PROBLEMS})
            lock = dict(experiment='R84', arm=arm, selected_without_test=True, per_problem_assembly=False,
                selected_update=1440, checkpoint_sha256='f' * 64, state_hash=goal.digest(config))
            goal.save(run / 'protocol.json', config)
            goal.save(run / 'final_test_lock.json', lock)
            goal.save(run / 'training_complete.json', dict(successful_updates=goal.BUDGET))
            goal.save(run / 'test_results.json', dict(complete=True))
            goal.save(run / 'lib_stage_complete.json', dict(complete=True))
            goal.save(run / 'postprocess_status.json', dict(phase='complete', complete=True))
            goal.save(run / 'full_model_statistics/protocol.json', dict(complete=True))
            for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf'):
                (run / 'full_model_statistics' / name).write_bytes(b'%PDF synthetic upstream receipt')
            for split in ('val', 'test'):
                directory = run / f'evaluation/{split}/selected_best'
                goal.save(directory / 'identity.json', dict(experiment='R84', arm=arm, split=split,
                    state_hash=lock['state_hash'], successful_updates=1440, model_state_sha256='e' * 64,
                    checkpoint_sha256=lock['checkpoint_sha256'] if split == 'test' else None))
                # A never chooses b or c. B improves both splits. C is bad on VAL
                # but perfect on TEST, which must not make its acceptance pass.
                pred = [0, 0, 0, 0]
                if arm == 'B_attention_direct' or arm == 'C_focal' and split == 'test':
                    pred = [0, 1, 1, 2]
                arrays = {}
                for problem in goal.PROBLEMS:
                    logits = np.zeros((4, 3), np.float32)
                    logits[np.arange(4), pred] = 1.
                    fields = dict(pool=np.array(['a', 'b', 'c']), pool_ids=np.array([1, 3, 7]),
                        costs=costs, winner=costs.argmin(1), logits=logits, indices=np.arange(4),
                        pred=np.array(pred), release_source_indices=np.arange(100, 104))
                    arrays.update({problem + '__' + k: v for k, v in fields.items()})
                np.savez_compressed(directory / 'rank000.npz', **arrays)

    def test_complete_report_freezes_targets_renders_pdf_and_does_not_use_test_gate(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(goal, 'ROWS', dict(train=81000, val=4, test=4)):
            root = Path(directory)
            self.fixture(root)
            result = goal.report(root, root / 'launch')
            self.assertTrue(result['complete'])
            self.assertFalse(result['goal_complete'])
            self.assertEqual(result['accepted_on_VAL'], ['B_attention_direct'])
            self.assertFalse(result['test_used_for_acceptance'])
            self.assertEqual(result['next_action'], 'review_validation_gain')
            frozen = goal.read(root / 'comparison_artifacts/target_list.json')
            self.assertEqual(len(frozen['targets']), 36)
            self.assertEqual({t['method'] for t in frozen['targets']}, {'b', 'c'})
            self.assertGreater((root / 'comparison_artifacts/COMPARISON.pdf').stat().st_size, 1000)
            self.assertTrue(goal.status(root, root / 'launch')['comparison_complete'])
            with patch.object(goal, 'make_pdf') as render:
                goal.report(root, root / 'launch')
            render.assert_not_called()
            # Targets cannot silently be redefined by modified baseline inputs.
            frozen['identity']['threshold_chosen_pct'] = .2
            goal.save(root / 'comparison_artifacts/target_list.json', frozen)
            with self.assertRaises(ValueError):
                goal.freeze_targets(root)

    def test_report_waits_for_all_arms_and_negative_result_stays_unresolved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(goal, 'ROWS', dict(train=81000, val=4, test=4)):
            root = Path(directory)
            self.fixture(root)
            missing = root / 'D_attention_focal/lib_stage_complete.json'
            missing.unlink()
            with patch.object(goal, 'derive') as derive, self.assertRaises(RuntimeError):
                goal.report(root, root / 'launch')
            derive.assert_not_called()
            goal.save(missing, dict(complete=True))
            original = goal.derive

            def negative(*args):
                result = original(*args)
                for row in result['gates']:
                    row['accepted_on_VAL'] = False
                return result

            with patch.object(goal, 'derive', side_effect=negative):
                result = goal.report(root, root / 'launch')
            self.assertEqual(result['next_action'], 'research_handoff_needed')
            self.assertFalse(result['goal_complete'])


if __name__ == '__main__':
    unittest.main()
