"""Small CPU finish contracts; never invoke a real selector, GPU, or solver."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import r122_finish as finish


def toy(pred=(1, 0)):
    costs = np.array([[1., 2., 3.], [10., 11., 12.]], dtype=np.float64)
    scores = np.zeros_like(costs, dtype=np.float32)
    scores[np.arange(2), np.asarray(pred)] = 2.
    return dict(costs=costs, winner=np.array([0, 0], dtype=np.int64), pool=['a', 'b', 'c'],
        pool_ids=np.array([0, 1, 2], dtype=np.int64), indices=np.arange(2, dtype=np.int64),
        source_indices=np.arange(2, dtype=np.int64), release_source_indices=np.arange(2, dtype=np.int64),
        pred=np.array(pred, dtype=np.int64), scores=scores, path='synthetic_cpu_only.npz')


def reference():
    return dict(sbs_solver='a', sbs_slot=0, selection_split='train', mean_cost_by_solver=dict(a=5.5, b=6.5, c=7.5))


def val_result(risk=.02, miss=.1, top1=.75):
    return dict(macro=dict(macro_top1=top1), hard_decision=dict(macro=dict(
        hard_regret=risk, hard_fixed_miss=miss, native_top1=top1)))


class FinishCPUContracts(unittest.TestCase):
    def setUp(self):
        self.directory = finish.DEFAULT_ROOT / 'preparation'
        self.directory.mkdir(parents=True, exist_ok=True)

    def test_epoch_adapter_is_derived_only_and_does_not_mutate_checkpoint_state(self):
        state = dict(successful_updates=1440, state_hash='same-protocol', model={'untouched': 3})
        lock = dict(selected_update=1440, epoch=2, state_hash='same-protocol')
        adapted = finish.legacy_epoch_state(state, lock)
        self.assertEqual(adapted['epoch'], 2)
        self.assertNotIn('epoch', state)
        self.assertIs(adapted['model'], state['model'])
        with self.assertRaises(ValueError):
            finish.legacy_epoch_state(state, dict(lock, epoch=3))
        with self.assertRaises(ValueError):
            finish.legacy_epoch_state(dict(state, successful_updates=720), lock)

    def test_native_cost_precision_and_stable_inference_archive_validation(self):
        costs = np.array([[1. + 1e-10, 1.], [1., 1. + 1e-10]], dtype=np.float64)
        archive = dict(costs=costs, winner=np.array([1, 0], dtype=np.int64),
            logits=np.array([[0., 2.], [2., 0.]], dtype=np.float32), pool=np.array(['a', 'b']),
            pool_ids=np.array([0, 1], dtype=np.int64), indices=np.arange(2, dtype=np.int64))
        config = dict(candidate_pools=dict(x=['a', 'b']), candidate_pool_ids=dict(x=[0, 1]))
        data = finish.read_prediction(archive, '', config, 'x', n=2)
        self.assertEqual(data['pred'].tolist(), [1, 0])
        with self.assertRaises(ValueError):
            finish.read_prediction(dict(archive, costs=costs.astype(np.float32)), '', config, 'x')
        with self.assertRaises(AssertionError):
            finish.read_prediction(dict(archive, pred=np.array([0, 0], dtype=np.int64)), '', config, 'x')
        tied = dict(archive, costs=np.ones((2, 2), dtype=np.float64), winner=np.array([1, 0], dtype=np.int64))
        self.assertEqual(finish.read_prediction(tied, '', config, 'x')['winner'].tolist(), [1, 0])

    def test_original_metrics_distinguish_ratio_of_means_and_actual_regret(self):
        row, _ = finish.metric_row(toy(), reference(), model='SYNTHETIC', split='TEST', problem='x',
            checkpoint='CPU_ONLY', version='synthetic')
        self.assertEqual(row['top1_pct'], 50.)
        self.assertEqual(row['mean_cost'], 6.)
        self.assertEqual(row['oracle_cost'], 5.5)
        self.assertEqual(row['actual_regret_pct'], 50.)
        self.assertAlmostEqual(row['vs_oracle_pct'], 100. * (6. / 5.5 - 1.))
        self.assertNotEqual(row['actual_regret_pct'], row['vs_oracle_pct'])
        self.assertEqual(row['sbs_selection_split'], 'frozen81000 TRAIN')

    def test_macro_is_equal_task_and_never_silently_partial(self):
        rows = [dict(problem=p, n=2 if i else 2000, native_correct=1,
                     **{k: float(i) for k in finish.CORE_METRICS + finish.EXTRA_METRICS})
                for i, p in enumerate(finish.PROBLEMS)]
        total = finish.macro_rows(rows)
        self.assertEqual(total['top1_pct'], 8.5)
        self.assertEqual(total['native_correct'], 18)
        with self.assertRaises(ValueError):
            finish.macro_rows(rows[:-1])

    def test_error_budget_native_partitions_and_train_val_roles(self):
        data = toy()
        natural = {p: copy.deepcopy(data) for p in finish.PROBLEMS}
        predictions = {split: natural for split in ('TRAIN', 'VAL', 'TEST')}
        targets = [(p, name) for p in finish.PROBLEMS for name in ('a', 'b')]
        budgets = finish.error_budgets(predictions, targets)
        full = next(r for r in budgets if (r['split'], r['problem'], r['group']) == ('TRAIN', 'ALL', 'FULL18'))
        self.assertEqual(full['hard_regret_contribution'], .5)
        self.assertEqual(full['hard_error_frequency'], .5)
        config = dict(constraint_reference=dict(train=dict(macro=dict(hard_regret=.6, hard_fixed_miss=.6)),
            val_selection=dict(macro=dict(hard_regret=.4, hard_fixed_miss=.4))))
        rows = finish.constraint_rows(budgets, config, dict(feasible=False))
        self.assertTrue(all(r['satisfied'] for r in rows if r['split'] == 'TRAIN'))
        self.assertFalse(any(r['satisfied'] for r in rows if r['split'] == 'VAL'))
        self.assertTrue(all(r['threshold_source_split'] == 'VAL' and 'NOT a selection gate' in r['role']
                            for r in rows if r['split'] == 'TEST'))

    def test_no_feasible_val_fallback_is_not_hidden_or_reselected(self):
        budget = dict(hard_regret=.01, hard_fixed_miss=.1)
        config = dict(constraint_reference=dict(val_selection=dict(macro=budget)))
        history = []
        for point, update in enumerate(range(720, 14401, 720)):
            val = val_result(risk=.03 - point * .0001)
            key = finish.runner().selection_key(val, budget, update)
            history.append(dict(successful_updates=update, val=val, selection_key=list(key), feasible=False))
        lock = dict(selected_update=14400, selection_key=history[-1]['selection_key'], feasible=False)
        audit = finish.selection_audit(config, lock, history)
        self.assertTrue(audit['fallback_used'])
        self.assertFalse(audit['selected_feasible'])
        self.assertFalse(audit['checkpoint_reselected'])
        self.assertIn('NOT satisfied', audit['failure_disclosure'])
        with self.assertRaises(ValueError):
            finish.selection_audit(config, dict(lock, feasible=True), history)

    def test_fixed36_true_rescue_is_not_a_false_pick_or_selection_count(self):
        data = dict(costs=np.array([[1., 2., 3.], [2., 1., 3.], [2., 3., 1.], [3., 2., 1.]], dtype=np.float64),
            winner=np.array([0, 1, 2, 2], dtype=np.int64), pool=['a', 'b', 'c'], pool_ids=np.arange(3, dtype=np.int64),
            indices=np.arange(4, dtype=np.int64), source_indices=np.arange(4, dtype=np.int64))
        old = {p: dict(data, pred=np.array([2, 1, 2, 2], dtype=np.int64)) for p in finish.PROBLEMS}
        new = {p: dict(data, pred=np.array([0, 2, 0, 2], dtype=np.int64)) for p in finish.PROBLEMS}
        targets = [(p, solver) for p in finish.PROBLEMS for solver in ('a', 'b')]
        details, totals = finish.recovery_tables(old, new, targets, 'synthetic comparison', 'TEST')
        first = next(r for r in details if r['problem'] == finish.PROBLEMS[0] and r['solver'] == 'a')
        self.assertEqual((first['selected_count'], first['true_rescues'], first['false_pick_count']), (2, 1, 1))
        fixed = next(r for r in totals if r['problem'] == 'ALL' and r['solver'] == 'FIXED36')
        self.assertEqual(fixed['net_correct'], 0)
        self.assertEqual(fixed['new_false_pick_count'], 18)
        full = next(r for r in totals if r['problem'] == 'ALL' and r['solver'] == 'FULL18')
        self.assertEqual(full['net_correct'], -18)

    def test_missing_nss_retains_all_four_explicit_rows(self):
        data = toy()
        model = dict(config=dict(data_root='/synthetic/new90k'), lock=dict(checkpoint_sha256='CPU_ONLY'))
        with tempfile.TemporaryDirectory(prefix='finish_nss_', dir=self.directory) as temporary:
            rows, packed = finish.nss_comparisons(dict(TEST=dict(TSP=data, CVRP=data)),
                dict(TSPLIB=data, CVRPLIB=data), model, Path(temporary), {})
        self.assertEqual(len(rows), 4)
        self.assertFalse(packed)
        self.assertTrue(all(r['status'] == 'missing_saved_NSS' and r['nss_top1_native_pct'] is None for r in rows))
        self.assertTrue(all(r['current_top1_native_pct'] == 50. for r in rows))

    def test_distribution_pairs_native_winners_test_then_lib_and_true_top1(self):
        finish.set_style()
        from matplotlib import pyplot as plt
        data = toy()
        lock = dict(checkpoint_sha256='SYNTHETIC_CPU_ONLY', selected_update=720)
        fig = finish.distribution_figure('TSP', [('TEST', data), ('TSPLIB', data)], data['pool'], lock,
            label='SYNTHETIC CPU layout, not R122 results', data_version='synthetic')
        try:
            self.assertTrue(fig.axes[0].get_title().startswith('TEST'))
            self.assertTrue(fig.axes[1].get_title().startswith('TSPLIB'))
            self.assertIn('\u6a21\u578bTop1=50.00%', fig.axes[0].get_title())
            self.assertEqual(len(fig.axes[0].patches), 6)
            with tempfile.TemporaryDirectory(prefix='finish_pdf_', dir=self.directory) as temporary:
                path = Path(temporary) / 'SYNTHETIC_layout.pdf'
                fig.savefig(path)
                self.assertTrue(path.read_bytes().startswith(b'%PDF-'))
            rows = finish.distribution_rows('TSP', 'TEST', data, lock)
            self.assertEqual(sum(r['selected_count'] for r in rows), 2)
            self.assertEqual(sum(r['winner_count'] for r in rows), 2)
            self.assertTrue(all(r['model_top1_native_pct'] == 50. for r in rows))
        finally:
            plt.close(fig)

    def test_missing_fulltest_gate_precedes_lib_inference_or_model_creation(self):
        with tempfile.TemporaryDirectory(prefix='finish_gate_', dir=self.directory) as temporary:
            with patch.object(finish.common, 'r80_lib', side_effect=AssertionError('No GPU/inference allowed')) as infer:
                with self.assertRaises(finish.NotReady):
                    finish.main(['finish', '--root', temporary, '--device', 'cuda:0'])
                infer.assert_not_called()
            self.assertFalse((Path(temporary) / 'model').exists())

    def test_bounded_saved_only_report_spine_keeps_failed_constraints_and_missing_references(self):
        """Tiny synthetic data, mocked gates/rendering; never real R122 results."""
        from matplotlib import pyplot as plt
        data = toy()
        measured = finish.metric_row(data, reference(), model='SYNTHETIC', split='TEST', problem='TSP',
            checkpoint='SYNTHETIC', version='SYNTHETIC')[1]
        with tempfile.TemporaryDirectory(prefix='finish_SYNTHETIC_spine_', dir=self.directory) as temporary:
            root, run = Path(temporary), Path(temporary) / 'model'
            run.mkdir()
            config = dict(data_root=str(root / 'SYNTHETIC_NOT_R122_DATA'),
                rows_per_problem=dict(train=2, val=2, test=2), model_params={}, loss={}, controller={},
                checkpoint_rule='SYNTHETIC', inference_rule='SYNTHETIC',
                baselines={p: reference() for p in finish.PROBLEMS},
                candidate_pools={p: data['pool'] for p in finish.PROBLEMS},
                candidate_pool_ids={p: [0, 1, 2] for p in finish.PROBLEMS},
                constraint_reference=dict(train=dict(macro=dict(hard_regret=.6, hard_fixed_miss=.6)),
                    val_selection=dict(macro=dict(hard_regret=.4, hard_fixed_miss=.4))))
            lock = dict(checkpoint_sha256='SYNTHETIC_NOT_A_CHECKPOINT', selected_update=720,
                epoch=1, state_hash='SYNTHETIC', feasible=False)
            model = dict(run=run, config=config, lock=lock, history=[],
                selection_audit=dict(fallback_used=True, failure_disclosure='SYNTHETIC: no feasible checkpoint'),
                results={split: dict(per_problem={p: measured for p in finish.PROBLEMS}) for split in ('TRAIN', 'VAL', 'TEST')})
            natural = {p: data for p in finish.PROBLEMS}
            libs = dict(TSPLIB=data, CVRPLIB=data)
            lib_results = {name: dict(train_sbs_reference=measured) for name in libs}
            targets = [(p, s) for p in finish.PROBLEMS for s in ('a', 'b')]

            def render(output, *unused):
                fig = plt.figure(figsize=(4, 2))
                fig.text(.05, .5, 'SYNTHETIC CPU fixture; NOT R122 results')
                for name in finish.REQUIRED_PDFS:
                    fig.savefig(output / name)
                plt.close(fig)

            args = finish.parser().parse_args(['report', '--root', temporary, '--device', 'cpu',
                '--nss-root', str(root / 'missing_nss')])
            with patch.object(finish, 'locked_inputs', return_value=model), \
                    patch.object(finish, 'natural_split', return_value=natural), \
                    patch.object(finish, 'load_libs', return_value=(libs, lib_results)), \
                    patch.object(finish, 'load_targets', return_value=targets), \
                    patch.object(finish, 'historical_model', side_effect=FileNotFoundError('SYNTHETIC missing reference')), \
                    patch.object(finish, 'render_reports', side_effect=render), \
                    patch.object(finish.common, 'r80_lib', side_effect=AssertionError('No model inference')):
                complete = finish.report(args)
            self.assertTrue(complete['complete'])
            self.assertFalse(complete['selected_val_feasible'])
            self.assertTrue(complete['scientific_claims']['train_constraints_satisfied'])
            self.assertFalse(complete['scientific_claims']['val_constraints_satisfied'])
            self.assertEqual(len(complete['scientific_claims']['failed_train_val_constraints']), 2)
            output = run / 'full_model_statistics'
            protocol = json.loads((output / 'protocol.json').read_text())
            self.assertTrue(all(r['status'] == 'missing_saved_NSS' for r in protocol['nss_status']))
            self.assertTrue(all(r['status'] == 'missing' for r in protocol['historical_references']))
            self.assertFalse(protocol['inference'])
            self.assertFalse(protocol['checkpoint_reselection'])
            self.assertTrue((output / 'constraint_status.csv').is_file())
            self.assertTrue((output / 'metrics_by_problem.csv').is_file())


if __name__ == '__main__':
    unittest.main()
