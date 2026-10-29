"""Small CPU-only synthetic finish checks; no real TEST/LIB feature access."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import r124_finish as finish


def fixture():
    difference = np.array([-2., -2., 2., 2.], dtype=np.float32)
    stage1 = np.tile(np.array([3., 2., 1.], dtype=np.float32), (4, 1))
    logits = stage1.copy()
    logits[:, 0] = 2. + 1. / (1. + np.exp(-difference))
    logits[:, 1] = 2. + 1. / (1. + np.exp(difference))
    return dict(costs=np.array([[1.1, 1., 1.3], [1., 1.2, 1.4], [1.3, 1.2, 1.], [1.1, 1., 1.2]], dtype=np.float64),
        winner=np.array([1, 0, 2, 1], dtype=np.int64), pool=np.array(['a', 'b', 'c']),
        pool_ids=np.array([3, 5, 9], dtype=np.int64), indices=np.arange(4), release_source_indices=np.arange(100, 104),
        stage1_scores=stage1, logits=logits, pair_indices=np.tile([0, 1], (4, 1)),
        difference=difference, pair_quality=np.zeros(4, dtype=np.float32), nodes=np.full(4, 3),
        forward_seconds=np.array(.1), forward_batch_cap=np.array(2), full_pipeline_timed=np.array(True))


class FinishTests(unittest.TestCase):
    def test_correction_damage_coverage_and_two_terms(self):
        data = fixture()
        summary, candidates = finish.reranking_diagnostics(data)
        self.assertEqual((summary['corrected_count'], summary['damaged_count']), (1, 1))
        self.assertEqual(summary['coverage_native_pct'], 75.)
        self.assertEqual(summary['final_top1_pct'], 25.)
        self.assertAlmostEqual(summary['conditional_native_top1_pct'], 100. / 3.)
        self.assertAlmostEqual(summary['recall_floor_regret_pct'], 5.)
        self.assertAlmostEqual(summary['reranking_excess_regret_pct'], 10.)
        self.assertAlmostEqual(summary['actual_regret_pct'], 15.)
        self.assertEqual(candidates[2]['winner_outside_top2'], 1)
        self.assertEqual(candidates[2]['top2_occurrences'], 0)
        self.assertEqual(candidates[2]['native_candidate_coverage_pct'], 0.)
        old_loss = summary['native_bce_all_rows']
        self.assertAlmostEqual(old_loss, np.logaddexp(0., [-2., 2., 2.]).sum() / 4.)
        data['pair_quality'][:] = 999.
        changed, _ = finish.reranking_diagnostics(data)
        self.assertEqual(changed['native_bce_all_rows'], old_loss)
        self.assertEqual(changed['final_top1_pct'], summary['final_top1_pct'])
        self.assertNotEqual(changed['pair_quality_huber_all_rows'], summary['pair_quality_huber_all_rows'])
        data['logits'][:, 2] = 100.
        with self.assertRaises(ValueError):
            finish.reranking_diagnostics(data)

    def test_native_ties_and_fixed_global_id_scores(self):
        data = fixture()
        data['costs'][0] = [1., 1., 1.3]
        data['winner'][0] = 0
        pred = finish.validate_data(data)
        self.assertNotEqual(pred[0], data['winner'][0])
        self.assertEqual(data['costs'][0, pred[0]], 1.)
        data['difference'][0] = 0.
        data['logits'][0, :2] = 2.5
        data['pool_ids'] = np.array([5, 3, 9])
        self.assertEqual(finish.validate_data(data)[0], 1)
        finish.reranking_diagnostics(data)
        with self.assertRaisesRegex(ValueError, 'NEW4500'):
            finish.validate_data(data, expected_rows=4500)

    def test_macro_is_equal_task_and_conditional_is_not_average(self):
        rows = [dict(problem=p, n=i + 1, coverage_native_pct=50. if i % 2 else 100.,
                     final_top1_pct=50., corrected_count=i, actual_regret_pct=float(i))
                for i, p in enumerate(finish.PROBLEMS)]
        result = finish.macro_rows(rows, 'TEST')
        self.assertEqual(result['n'], sum(r['n'] for r in rows))
        self.assertEqual(result['actual_regret_pct'], 8.5)
        self.assertAlmostEqual(result['conditional_native_top1_pct'], 100. * 50. / 75.)
        self.assertNotEqual(result['conditional_native_top1_pct'], 75.)
        with self.assertRaises(ValueError):
            finish.macro_rows(rows[:-1], 'TEST')

    def test_reference_all_rows_costs_native_identity_and_release(self):
        current = fixture()
        row, col = np.array([2, 0, 3, 1]), np.array([1, 2, 0])
        inverse = np.argsort(col)
        desired = np.array([1, 1, 0, 0])
        other = dict(pool=current['pool'][col], costs=current['costs'][row][:, col],
            winner=inverse[current['winner'][row]], indices=current['indices'][row],
            source_indices=current['release_source_indices'][row], pred=inverse[desired[row]])
        np.testing.assert_array_equal(finish.align_reference(current, other, require_source=True), desired)
        other['costs'][1, 1] += 1e-10
        with self.assertRaisesRegex(ValueError, 'ALL rows'):
            finish.align_reference(current, other, require_source=True)
        other['costs'][1, 1] -= 1e-10
        other['source_indices'][0] += 10
        with self.assertRaisesRegex(ValueError, 'source IDs'):
            finish.align_reference(current, other, require_source=True)

    def test_full_forward_label_isolation_and_variable_microbatch(self):
        import torch
        from .r124_model import replace_top2
        torch.set_num_threads(1)
        raw = fixture()
        labels = torch.from_numpy(raw['costs']).float()
        batch = dict(kind='matrix', problem_id=torch.zeros(4, dtype=torch.long),
            matrix=torch.zeros(4, 3, 3), node_mask=torch.ones(4, 3, dtype=torch.bool),
            n=torch.full((4,), 3), costs=labels, ind=torch.tensor([1, 0, 2, 1]),
            pool_ids=torch.tensor([3, 5, 9]), poisoned_query_label=torch.ones(4))
        loader = SimpleNamespace(batch=batch, lengths=batch['n'])
        seen = []

        def composite(inputs):
            self.assertTrue({'costs', 'ind', 'poisoned_query_label'}.isdisjoint(inputs))
            size = len(inputs['n'])
            seen.append(size)
            scores = torch.tensor([3., 2., 1.]).expand(size, -1)
            pair = torch.tensor([0, 1]).expand(size, -1)
            d = torch.full((size,), -2.)
            return dict(logits=replace_top2(scores, pair, d, inputs['pool_ids'][None].expand(size, -1), inputs['solver_mask']),
                stage1_scores=scores, pair_indices=pair, difference=d, pair_quality=torch.zeros(size))

        first = finish.full_forward(composite, loader, raw, 3, torch.device('cpu'))
        self.assertEqual(seen, [3, 1])
        self.assertTrue(bool(first['full_pipeline_timed']))
        self.assertGreater(float(first['forward_seconds']), 0.)
        seen.clear()
        batch['costs'].fill_(999.)
        batch['ind'].zero_()
        second = finish.full_forward(composite, loader, raw, 2, torch.device('cpu'))
        np.testing.assert_array_equal(first['logits'], second['logits'])
        self.assertEqual(seen, [2, 2])

    def test_tables_and_pdf_distributions(self):
        from matplotlib.backends.backend_pdf import PdfPages
        predictions = {(split, p): fixture() for split in ('VAL', 'TEST') for p in finish.PROBLEMS}
        for p in finish.PROBLEMS:
            predictions['VAL', p]['full_pipeline_timed'] = np.array(False)
        predictions['TSPLIB', 'TSP'] = fixture()
        predictions['CVRPLIB', 'CVRP'] = fixture()
        config = dict(data_root='/fixture/new90k', scenario_solver_aliases={},
                      baselines={p: dict(selection_split='train', sbs_solver='a') for p in finish.PROBLEMS})
        tables = finish.build_tables(predictions, config)
        self.assertEqual(len(tables['candidate_coverage']), 38 * 3)
        self.assertEqual(len(tables['metrics']), 40)
        self.assertTrue(next(r for r in tables['inference_cost'] if r['problem'] == 'ALL' and r['split'] == 'TEST')['full_stage0_plus_reranker'])
        plt = finish.plotting()
        fig = finish.distribution_figure(plt, 'TSP', predictions, config, dict(selected_update=720))
        self.assertEqual(len(fig.axes), 2)
        self.assertIn('25.00%', fig.axes[0].get_title())
        self.assertEqual(len(fig.axes[0].patches), 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'distribution.pdf'
            with PdfPages(path) as pdf:
                pdf.savefig(fig)
            self.assertTrue(path.read_bytes().startswith(b'%PDF-'))
        plt.close(fig)
        predictions['TEST', 'TSP']['full_pipeline_timed'] = np.array(False)
        with self.assertRaisesRegex(ValueError, 'COMPLETE'):
            finish.build_tables(predictions, config)

    def test_four_nss_datasets_and_complete_offline_report(self):
        predictions = {(split, p): fixture() for split in ('VAL', 'TEST') for p in finish.PROBLEMS}
        for p in finish.PROBLEMS:
            predictions['VAL', p]['full_pipeline_timed'] = np.array(False)
        predictions['TSPLIB', 'TSP'] = fixture()
        predictions['CVRPLIB', 'CVRP'] = fixture()
        config = dict(data_root='/fixture/new90k', scenario_solver_aliases={},
                      baselines={p: dict(selection_split='train', sbs_solver='a') for p in finish.PROBLEMS})
        lock = dict(checkpoint_sha256='synthetic-checkpoint', selected_update=720)
        history = [dict(successful_updates=720, optimization=dict(native_ce=.8, pair_quality_huber=2.),
            val=dict(macro=dict(macro_top1=.25, macro_actual_regret_pct=15.)))]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(data_root=config['data_root'], expected_test_rows_per_task=4500, retrained_on_new90k=False,
                        checkpoints={p: dict(sha256='nss-' + p) for p in ('TSP', 'CVRP')})
            finish.write_json(root / 'protocol.json', plan)
            for split, p in (('TEST', 'TSP'), ('TEST', 'CVRP'), ('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')):
                stem = 'test' if split == 'TEST' else split
                reference = fixture()
                reference['pred'] = np.array([0, 1, 0, 0])
                reference['source_indices'] = reference['release_source_indices']
                finish.save_npz(root / p / (stem + '_predictions.npz'), reference)
                finish.write_json(root / p / (stem + '_metrics.json'), dict(checkpoint=dict(sha256='nss-' + p)))
            args = SimpleNamespace(nss_root=root, allow_missing_nss=False)
            nss = finish.nss_comparisons(args, config, predictions, lock)
            self.assertEqual(len(nss), 4)
            self.assertTrue(all(r['nss_status'] == 'complete' for r in nss))
            self.assertTrue(all(r['nss_top1_native_pct'] == 0. for r in nss))
            tables = finish.build_tables(predictions, config)
            output = root / 'reports'
            paths = finish.render_reports(output, predictions, tables, nss, config, lock, history)
            self.assertTrue(all(Path(path).read_bytes().startswith(b'%PDF-') for path in paths.values()))
            self.assertTrue((output / 'candidate_coverage.csv').exists())
            self.assertEqual(len(list(output.glob('*.csv'))), 7)

    def test_prelock_gate_and_arbitrary_v2_describe(self):
        from . import r124_experiment as runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'launch_v2'
            finish.write_json(root / 'model/protocol.json', dict(data_root='/fixture/new90k'))
            with patch.object(runner, 'authorize_test', side_effect=PermissionError('not locked')), patch.object(finish, 'archive') as opened:
                with self.assertRaises(PermissionError):
                    finish.lib_stage(SimpleNamespace(root=root))
                with self.assertRaises(PermissionError):
                    finish.report(SimpleNamespace(root=root))
                opened.assert_not_called()
            with redirect_stdout(StringIO()):
                value = finish.main(['describe', '--root', str(root)])
            self.assertEqual(value['checkpoint'], str(root / 'model/best.pt'))
            self.assertFalse(value['inference_started'])
            self.assertFalse((root / 'model/finish').exists())


if __name__ == '__main__':
    unittest.main()
