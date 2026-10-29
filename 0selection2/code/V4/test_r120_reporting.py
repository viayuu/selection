"""Small native metric tests: directions, true ties, zeros and missing labels."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from . import r120_reporting as reporting


class NativeMetricsTests(unittest.TestCase):
    def test_min_and_max_have_same_directed_loss(self):
        for direction, values in (('min', [[10., 12.], [12., 8.]]),
                                  ('max', [[12., 10.], [8., 12.]])):
            values = np.asarray(values, np.float64)
            observed = np.ones_like(values, dtype=bool)
            reference = reporting.fit_train_reference(values, observed, direction, ['a', 'b'])
            metric, pred = reporting.native_task_metrics(np.asarray([[0., 1.], [0., 1.]]),
                values, observed, direction, reference=reference)
            np.testing.assert_array_equal(pred, [1, 1])
            self.assertEqual(metric['any_best_top1_pct'], 50.)
            self.assertEqual(metric['absolute_excess'], 1.)
            self.assertGreaterEqual(metric['train_scaled_excess'], 0.)
            self.assertEqual(metric['mean_cost'] is None, direction == 'max')

    def test_exact_ties_and_native_winner_are_separate(self):
        values = np.asarray([[1., 1., 3.], [1., 3., 1.]], np.float64)
        scores = np.asarray([[0., 1., 0.], [0., 0., 1.]])
        observed = np.ones_like(values, dtype=bool)
        ref = reporting.fit_train_reference(values, observed, 'min', ['a', 'b', 'c'])
        metric, _ = reporting.native_task_metrics(scores, values, observed, 'min',
                                                native_winner=np.asarray([0, 0]), reference=ref)
        self.assertEqual(metric['any_best_top1_pct'], 100.)
        self.assertEqual(metric['canonical_top1_pct'], 0.)
        self.assertEqual(metric['native_top1_pct'], 0.)
        self.assertEqual(metric['tied_instances'], 2)
        self.assertEqual(metric['actual_regret_pct'], 0.)

    def test_zero_oracle_is_not_epsilon_relative_regret(self):
        values = np.asarray([[0., 8.], [0., 0.], [4., 6.]], np.float64)
        observed = np.ones_like(values, dtype=bool)
        reference = reporting.fit_train_reference(values, observed, 'min', ['a', 'b'])
        metric, _ = reporting.native_task_metrics(np.asarray([[0., 1.]] * 3), values,
                                                observed, 'min', reference=reference)
        self.assertEqual(metric['zero_oracle_count'], 2)
        self.assertEqual(metric['positive_oracle_count'], 1)
        self.assertEqual(metric['actual_regret_pct'], 50.)
        self.assertEqual(metric['zero_oracle_absolute_excess'], 4.)
        self.assertEqual(metric['zero_oracle_positive_loss_pct'], 50.)
        self.assertAlmostEqual(metric['train_scaled_excess'], 2.5)

    def test_all_zero_train_sbs_requires_declared_input_scale(self):
        values = np.asarray([[0., 0.], [0., 1.]], np.float64)
        mask = np.ones_like(values, dtype=bool)
        with self.assertRaisesRegex(ValueError, 'input scale'):
            reporting.fit_train_reference(values, mask, 'min', ['a', 'b'])
        reference = reporting.fit_train_reference(values, mask, 'min', ['a', 'b'], zero_scale=200.)
        self.assertEqual(reference['objective_scale'], 200.)
        self.assertIn('processing', reference['scale_source'])

    def test_missing_candidate_is_not_changed_to_a_legal_mask(self):
        values = np.asarray([[5., np.nan], [3., 4.]], np.float64)
        observed = np.asarray([[True, False], [True, True]])
        reference = reporting.fit_train_reference(values, observed, 'min', ['a', 'b'])
        self.assertEqual(reference['sbs_slot'], 0)
        metric, pred = reporting.native_task_metrics(np.asarray([[0., 1.], [1., 0.]]),
            values, observed, 'min', reference=reference)
        self.assertEqual(pred[0], 1)
        self.assertEqual(metric['selected_objective_missing'], 1)
        self.assertEqual(metric['any_best_top1_pct'], 50.)

    def test_original_fp64_ties_not_float32_rounding(self):
        values = np.asarray([[10., 10. + 1e-8]], np.float64)
        metric, _ = reporting.native_task_metrics(np.asarray([[0., 1.]]), values,
                                                np.ones_like(values, dtype=bool), 'min')
        self.assertEqual(metric['any_best_top1_pct'], 0.)
        self.assertEqual(metric['tied_instances'], 0)
        with self.assertRaisesRegex(ValueError, 'FP64'):
            reporting.native_task_metrics(np.asarray([[0., 1.]]), values.astype(np.float32),
                                          np.ones_like(values, dtype=bool), 'min')

    def test_stable_ce_native_winner_and_best_set_have_distinct_semantics(self):
        values = np.asarray([[1., 1., 2.], [1., 2., 3.]], np.float64)
        scores = np.asarray([[10000., 10000., 9999.], [10000., 0., -10000.]])
        mask = np.ones_like(values, dtype=bool)
        tied, _ = reporting.native_task_metrics(scores, values, mask, 'min')
        native, _ = reporting.native_task_metrics(scores, values, mask, 'min', native_winner=np.asarray([0, 0]))
        self.assertAlmostEqual(tied['ce'], (np.log(2 + np.exp(-1)) - np.log(2)) / 2)
        self.assertAlmostEqual(native['ce'], np.log(2 + np.exp(-1)) / 2)
        hard, _ = reporting.native_task_metrics(np.asarray([[10000., -10000.]]),
            np.asarray([[2., 1.]], np.float64), np.ones((1, 2), dtype=bool), 'min')
        self.assertEqual(hard['ce'], 20000.)

    def test_task_macro_and_sample_micro_not_mixed_units(self):
        table = {}
        for problem, n, right, value in (('TSP', 9, 9, 10.), ('MIS', 1, 0, 100000.)):
            values = np.tile(np.asarray([[value, value + 1]], np.float64), (n, 1))
            scores = np.tile(np.asarray([[1., 0.]] if right else [[0., 1.]]), (n, 1))
            reference = reporting.fit_train_reference(values, np.ones_like(values, dtype=bool), 'min', ['a', 'b'])
            table[problem] = reporting.native_task_metrics(scores, values, np.ones_like(values, dtype=bool),
                                                         'min', reference=reference)[0]
        result = reporting.aggregate_metrics(table)['ALL24']
        self.assertEqual(result['task_macro_any_best_top1_pct'], 50.)
        self.assertEqual(result['sample_micro_any_best_top1_pct'], 90.)
        self.assertAlmostEqual(result['task_macro_ce'], np.mean([row['ce'] for row in table.values()]))
        self.assertAlmostEqual(result['sample_micro_ce'], sum(row['ce'] * row['n'] for row in table.values()) / 10)
        self.assertIsNone(result['mean_cost'])
        self.assertFalse(result['complete_task_scope'])

    def test_routing_legacy_cost_fields_are_task_macro_only(self):
        table = {}
        cases = (
            ('TSP', 'min', [[10., 20.]], [[20., 10.]] * 2),
            ('CVRP', 'min', [[200., 100.]], [[100., 300.], [10., 50.]] * 2),
            ('JSSP', 'min', [[1e6, 2e6]], [[1e6, 2e6]]),
            ('MKP', 'max', [[1e9, 2e9]], [[1e9, 2e9]]),
        )
        for problem, direction, train, evaluation in cases:
            train, values = np.asarray(train, np.float64), np.asarray(evaluation, np.float64)
            reference = reporting.fit_train_reference(train, np.ones_like(train, dtype=bool),
                                                     direction, ['a', 'b'])
            scores = np.tile([[0., 1.]], (len(values), 1))
            table[problem] = reporting.native_task_metrics(scores, values,
                np.ones_like(values, dtype=bool), direction, reference=reference)[0]
        groups = reporting.aggregate_metrics(table)
        routing = groups['ROUTING18']
        expected = dict(mean_cost=92.5, sbs_cost=97.5, oracle_cost=32.5,
                        vs_sbs_pct=-25., vs_oracle_pct=100. * (175. - 55.) / 55. / 2.,
                        actual_regret_pct=150.)
        for key, value in expected.items():
            self.assertAlmostEqual(routing[key], value, msg=key)
        self.assertAlmostEqual(routing['sample_micro_positive_oracle_regret_pct'], 200.)
        self.assertIn('equal-weight per-task', routing['raw_objective_aggregation'])
        for group in ('NONROUTING6', 'ALL24'):
            for key in ('mean_objective', 'mean_cost', 'sbs_cost', 'oracle_cost'):
                self.assertIsNone(groups[group].get(key), msg=f'{group}/{key}')
        table['CVRP']['sbs_cost'] = None
        self.assertIsNone(reporting.aggregate_metrics(table)['ROUTING18']['sbs_cost'])

    def test_method_table_retains_zero_choices_and_tie_support(self):
        values = np.asarray([[1., 1.], [2., 1.]], np.float64)
        scores, observed = np.asarray([[1., 0.], [1., 0.]]), np.ones_like(values, dtype=bool)
        reference = reporting.fit_train_reference(values, observed, 'min', ['a', 'b'])
        metrics, pred = reporting.native_task_metrics(scores, values, observed, 'min', reference=reference)
        data = dict(native_objective=values, observed_mask=observed, all_best_mask=values == values.min(1)[:, None],
                    logits=scores, pool=np.asarray(['a', 'b']))
        rows = reporting.method_rows('BPP', 'test', data, metrics, pred)
        self.assertEqual(rows[1]['selected_count'], 0)
        self.assertEqual(rows[1]['all_best_support_count'], 2)
        self.assertEqual(rows[1]['unique_best_count'], 1)
        self.assertEqual(rows[1]['unique_best_true_positive'], 0)
        self.assertEqual(rows[1]['unique_best_missed'], 1)
        self.assertEqual(rows[1]['unique_best_recall'], 0.)
        self.assertEqual(rows[1]['tied_best_replacement_directed_excess'], 0.)
        self.assertEqual(rows[1]['missed_best_replacement_directed_excess'], 1.)
        self.assertEqual(rows[1]['missed_unique_winner_train_scaled_excess'], 1.)
        self.assertEqual(rows[1]['false_negative'], 2)
        self.assertEqual(rows[1]['recall'], 0.)
        self.assertIsNone(rows[1]['precision'])

    def test_tie_replacements_and_unique_misses_use_native_direction(self):
        original = np.asarray([[2., 2., 5.], [2., 2., 5.], [2., 5., 6.],
                               [2., 5., 6.], [5., 2., 6.]], np.float64)
        chosen = np.asarray([0, 2, 0, 1, 1])
        scores = np.zeros_like(original)
        scores[np.arange(len(scores)), chosen] = 1.
        for direction, values in (('min', original), ('max', 10. - original)):
            observed = np.ones_like(values, dtype=bool)
            reference = reporting.fit_train_reference(values, observed, direction, ['a', 'b', 'c'])
            best = values == (values.min(1) if direction == 'min' else values.max(1))[:, None]
            metrics, pred = reporting.native_task_metrics(scores, values, observed, direction, reference=reference)
            data = dict(native_objective=values, observed_mask=observed, all_best_mask=best,
                        logits=scores, pool=np.asarray(['a', 'b', 'c']))
            rows = reporting.method_rows('BPP', 'test', data, metrics, pred)
            self.assertEqual(rows[0]['unique_best_true_positive'], 1)
            self.assertEqual(rows[0]['unique_best_missed'], 1)
            self.assertEqual(rows[0]['unique_best_recall'], .5)
            self.assertAlmostEqual(rows[0]['missed_unique_winner_train_scaled_excess'],
                                   3. / reference['objective_scale'])
            self.assertEqual(rows[1]['tied_best_replacement_directed_excess'], 3.)
            self.assertEqual(rows[1]['missed_best_replacement_directed_excess'], 3.)
            self.assertEqual(rows[1]['unique_best_recall'], 1.)
            self.assertEqual(rows[1]['positive_cost_missed_best_count'], 1)

    def test_nss_mismatched_population_is_explicitly_missing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'TSP').mkdir()
            values = np.asarray([[1., 2.]], np.float64)
            data = dict(native_objective=values, native_winner_index=np.asarray([0]), pool=np.asarray(['a', 'b']))
            metrics = dict(native_top1_pct=100., mean_cost=1., n=1)
            np.savez(root / 'TSP/test_predictions.npz', scores=np.asarray([[1., 0.]]),
                     costs=values + 1, winner=np.asarray([0]), pool=np.asarray(['a', 'b']))
            row = reporting.nss_rows({('test', 'TSP'): (data, metrics)}, root, 'R120')[0]
            self.assertIsNone(row['nss_top1_pct'])
            self.assertIn('mismatch', row['status'])

    def test_nss_reuse_preserves_its_distinct_train_population(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'TSP').mkdir()
            description = 'R79 historical VALID-selected NSS weights; not retrained on R84/new90k'
            reporting.write_json(root / 'protocol.json', dict(weights_description=description,
                historical_training_data_root='/fixture/old_70114_release', retrained_on_new90k=False))
            values = np.asarray([[1., 2.]], np.float64)
            data = dict(native_objective=values, native_winner_index=np.asarray([0]), pool=np.asarray(['a', 'b']))
            metric = dict(native_top1_pct=100., mean_cost=1., n=1)
            np.savez(root / 'TSP/test_predictions.npz', scores=np.asarray([[1., 0.]]),
                     costs=values, winner=np.asarray([0]), pool=np.asarray(['a', 'b']))
            row = reporting.nss_rows({('test', 'TSP'): (data, metric)}, root, 'R120')[0]
            self.assertEqual(row['nss_top1_pct'], 100.)
            self.assertEqual(row['nss_weights_description'], description)
            self.assertEqual(row['nss_historical_train_rows_per_problem'], 70114)
            self.assertFalse(row['training_protocol_match'])

    def test_learning_curves_distinguish_online_train_from_full_validation(self):
        from unittest.mock import patch
        with TemporaryDirectory() as directory:
            root = Path(directory)
            history = []
            for step, train_accuracy in ((24, .5), (48, .75)):
                history.append(dict(successful_updates=step,
                    train_parts=dict(ce=.8, train_any_best_accuracy=train_accuracy,
                                     train_mixed_accuracy=.4, train_hard_scaled_excess=.15),
                    validation={'aggregates': {'ALL24': dict(task_macro_ce=.9,
                        task_macro_any_best_top1_pct=45., task_macro_train_scaled_excess=.2)}}))
            reporting.write_json(root / 'history.json', history)
            plt = reporting._plotting()
            with patch.object(plt, 'close'):
                reporting.learning_curves(root, root / 'curves.png')
                figure = plt.gcf()
                top1_lines = {line.get_label(): line.get_ydata() for line in figure.axes[1].lines}
                np.testing.assert_array_equal(top1_lines['online TRAIN any-best'], [50., 75.])
                np.testing.assert_array_equal(top1_lines['VAL ALL24'], [45., 45.])
                self.assertIn('not a full-TRAIN evaluation', figure.texts[0].get_text())
            plt.close(figure)
            self.assertTrue((root / 'curves.pdf').is_file())

    def test_portable_pdf_and_csv_from_one_locked_saved_state(self):
        import hashlib
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'best.pt').write_bytes(b'CPU reporting fixture; not a model checkpoint')
            digest = hashlib.sha256((root / 'best.pt').read_bytes()).hexdigest()
            reporting.write_json(root / 'protocol.json', dict(model_name='R120_CPU_TEST',
                model={'problem_names': ['TSP']}, data_version='temporary unit-test fixture'))
            reporting.write_json(root / 'final_test_lock.json', dict(checkpoint_sha256=digest,
                selected_without_test=True, selected_update=1))
            values, scores = np.asarray([[1., 2.], [2., 3.]], np.float64), np.asarray([[1., 0.], [1., 0.]])
            mask = np.ones_like(values, dtype=bool)
            ref = reporting.fit_train_reference(values, mask, 'min', ['a', 'b'])
            metric, pred = reporting.native_task_metrics(scores, values, mask, 'min',
                native_winner=np.asarray([0, 0]), reference=ref)
            for split in ('val', 'test'):
                output = root / 'evaluation' / split / 'selected_best'
                (output / 'by_problem').mkdir(parents=True)
                np.savez(output / 'by_problem/TSP.npz', native_objective=values, logits=scores,
                    observed_mask=mask, all_best_mask=values == values.min(1)[:, None],
                    native_winner_index=np.asarray([0, 0]), pool=np.asarray(['a', 'b']), pred=pred)
                reporting.write_json(output / 'metrics.json', dict(identity={'checkpoint_sha256': digest},
                    per_problem={'TSP': metric}, aggregates=reporting.aggregate_metrics({'TSP': metric})))
            output = reporting.report(root)
            for name in ('RESULTS.pdf', 'selection_distributions.pdf', 'metrics.csv', 'nss_comparison.csv'):
                self.assertTrue((output / name).is_file())
                self.assertGreater((output / name).stat().st_size, 30)
            self.assertFalse(reporting.read_json(output / 'report_complete.json')['complete24'])


if __name__ == '__main__':
    unittest.main()
