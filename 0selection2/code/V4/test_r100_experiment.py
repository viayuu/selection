"""Narrow CPU CLI, fresh-root and saved native metric checks."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import r100_experiment as experiment
from . import r100_reporting as reporting


class ExperimentTests(unittest.TestCase):
    def test_cli_paths_and_unique_fixed_rule(self):
        with tempfile.TemporaryDirectory() as temp:
            args = experiment.normalize_args(experiment.parser().parse_args(['run', '--root', temp,
                '--context-root', '/tmp/shiys_R97_tabular_context', '--device', 'cuda:0', '--wandb-mode', 'offline']))
            self.assertEqual(args.protocol_lock, Path(temp) / 'protocol_lock.json')
            self.assertEqual(args.context_root, Path('/tmp/shiys_R97_tabular_context'))
            self.assertEqual(args.relation_root, Path(temp) / 'prep/relations')
            self.assertEqual(experiment.INFERENCE_RULE, reporting.INFERENCE_RULE)
            self.assertEqual(experiment.SELECTION_RULE, 'frozen_single_predeclared_bank_no_VAL_search')

    def test_fresh_run_and_immutable_write_no_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            experiment.ensure_fresh_run(root)
            experiment.write_once(root / 'protocol_lock.json', dict(a=1))
            experiment.write_once(root / 'protocol_lock.json', dict(a=1))
            with self.assertRaises(ValueError):
                experiment.write_once(root / 'protocol_lock.json', dict(a=2))
            experiment.write_once(root / 'attempt.json', dict(started=True))
            with self.assertRaisesRegex(RuntimeError, 'no implicit retry'):
                experiment.ensure_fresh_run(root)

    def test_saved_pair_scores_native_winner_metrics_and_tamper(self):
        p = np.array([[[.5, .8], [1.-.8, .5]], [[.5, .5], [.5, .5]]])
        raw = dict(costs=np.array([[1.+2.**-40, 1.], [3., 4.]]), winner=np.array([1, 0], dtype=np.int64),
                   pool=['A', 'B'])
        config = dict(candidate_pool_ids=dict(TSP=[2, 7]), baselines=dict(TSP=dict(
            selection_split='train', sbs_solver='A', sbs_slot=0)))
        metrics, arrays = experiment.saved_prediction('TSP', p, raw, [1000, 1001], [50, 100], config)
        self.assertNotIn('probabilities', arrays)
        np.testing.assert_array_equal(arrays['pred'], [0, 0])
        self.assertEqual(metrics['top1'], .5)
        self.assertGreater(metrics['actual_regret_pct'], 0.)
        view = reporting.load_predictions(arrays, '<fixture>')
        row = reporting.metric_row(view, config['baselines']['TSP'], 'TSP', 'TEST')
        self.assertEqual(row['model'], 'R100')
        self.assertEqual(row['top1_fp32_argmin_pct'], 100.)
        self.assertEqual(row['top1_pct'], 50.)
        self.assertGreater(row['formal_ce'], 0.)
        arrays['mean_opponent_win'] = np.zeros((2, 2))
        with self.assertRaises(AssertionError):
            reporting.load_predictions(arrays, '<tampered>')

    def test_distribution_labels_and_top1_are_model_accuracy_not_frequency(self):
        p = np.array([[[.5, .8], [1.-.8, .5]], [[.5, .5], [.5, .5]]])
        raw = dict(costs=np.array([[2., 1.], [3., 4.]]), winner=np.array([1, 0], dtype=np.int64), pool=['A', 'B'])
        config = dict(candidate_pool_ids=dict(TSP=[2, 7]), baselines=dict(TSP=dict(
            selection_split='train', sbs_solver='A', sbs_slot=0)))
        _, arrays = experiment.saved_prediction('TSP', p, raw, [1000, 1001], [50, 100], config)
        view = reporting.load_predictions(arrays, '<fixture>')
        figure = reporting.plot_distribution('TSP', [('TEST', view), ('TSPLIB', view)], ['A', 'B'], 'fixture',
            dict(checkpoint_sha256='fixture-bank'))
        self.assertIn('R100', figure._suptitle.get_text())
        self.assertIn('Top1=50.00%', figure.axes[0].get_title())
        self.assertEqual([t.get_text() for t in figure.axes[0].get_legend().get_texts()],
                         ['R100 model selection', 'True native winner'])
        self.assertIn('TEST', figure.axes[0].get_title())
        self.assertIn('TSPLIB', figure.axes[1].get_title())
        figure.clear()


if __name__ == '__main__':
    unittest.main()
