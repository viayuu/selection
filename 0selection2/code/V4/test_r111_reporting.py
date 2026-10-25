"""Three focused CPU contracts, using only synthetic temporary evidence."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from . import r111_reporting as report
from .r76_experiment import read_json, write_json


def config():
    return dict(experiment='R111', arm=report.ARMS[0], model_factory=report.MODEL_FACTORY,
        rows_per_problem=dict(train=81000, val=4500, test=4500), local_batch_size=512,
        warmup_updates=4320, eval_batch_size=256, optimizer='AdamW', rdrop_weight=0.,
        selection_head='normalized_family_times_conditional',
        microbatch_caps={p: 256 if p in ('TSP', 'CVRP') else 128 if p == 'ATSP' else 512
                        for p in report.PROBLEMS}, lr=dict(peak=1e-4, end=1e-6),
        loss=dict(nll=1., comparison=0., risk=0., rdrop=0.),
        data_root='/synthetic/frozen_NEW90k', source_sha256={'synthetic_only': 'not-real-evidence'})


def lock():
    return dict(experiment='R111', arm=report.ARMS[0], selected_update=1440, epoch=1,
        checkpoint_sha256='synthetic-checkpoint', state_hash='synthetic-protocol',
        rule='strict first full18 VAL macro_vs_sbs minimum')


class R111ReportingTests(unittest.TestCase):
    def test_cli_private_globals_and_lib_factory_without_inference(self):
        for private, original in ((report.common, report.r80_reporting),
                (report.shared, report.r84_reporting), (report.previous, report.r85_reporting),
                (report.helpers, report.r87_reporting)):
            self.assertIsNot(private, original)
            self.assertEqual(private.FACTORY, report.FACTORY)
            self.assertNotEqual(getattr(original, 'FACTORY', None), report.FACTORY)
        self.assertIs(report.shared.build_reports.__globals__, report.shared.__dict__)
        self.assertIs(report.previous.selected_train, report.selected_train)
        self.assertEqual(report.parser().parse_args(['report']).arm, report.ARMS[0])
        with tempfile.TemporaryDirectory() as temporary:
            args = report.parser().parse_args(['lib', '--root', temporary, '--device', 'cuda:0'])
            run = args.root / args.arm
            run.mkdir()
            with patch.object(report, 'locked_inputs', return_value=(run, {}, lock(),
                    dict(identity=dict(model_state_sha256='synthetic-model')), {})), \
                    patch.object(report, 'nss_reference', return_value=(Path('/synthetic/NSS'), {})), \
                    patch.object(report.common, 'r80_lib', return_value=dict(TSPLIB={}, CVRPLIB={})) as lib:
                report.lib_stage(args)
            self.assertEqual(lib.call_args.kwargs, dict(factory_module='code.V4.r111_experiment'))
            self.assertEqual(lib.call_args.args[4], 'cuda:0')
            self.assertEqual(read_json(run / 'lib_stage_complete.json')['model_factory'], report.MODEL_FACTORY)
            with patch.object(report, 'report', return_value={}) as cpu:
                report.main(['report', '--root', temporary])
                self.assertEqual(cpu.call_args.args[0].device, 'cpu')
            with patch.object(report, '_runner', side_effect=AssertionError('no model/factory')), patch('builtins.print'):
                self.assertFalse(report.main(['status', '--root', temporary])['inference'])

    def test_file_bound_selected_train_val_test_and_unique_final_metrics(self):
        cfg, selected = config(), lock()
        runner = SimpleNamespace(validate_checkpoint_config=Mock(), load_checkpoint=Mock())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / report.ARMS[0]
            results = {}
            for split in ('train', 'val', 'test'):
                path = run / 'evaluation' / split / 'selected_best/rank000.npz'
                path.parent.mkdir(parents=True)
                np.savez(path, synthetic_only=np.array(True))
                identity = dict(experiment='R111', arm=report.ARMS[0], split=split, tag='selected_best',
                    checkpoint_sha256=selected['checkpoint_sha256'], state_hash=selected['state_hash'],
                    successful_updates=1440, model_state_sha256='synthetic-model')
                write_json(path.parent / 'identity.json', identity)
                results[split] = dict(identity=identity, prediction_archive=str(path),
                    per_problem={p: {} for p in report.PROBLEMS})
            final = dict(complete=True, **results)
            write_json(run / 'final_metrics.json', final)
            record = (run, cfg, selected, results['test'], results['val'])
            with patch.object(report.shared, 'locked_inputs', return_value=record), \
                    patch.object(report, '_runner', return_value=runner):
                self.assertEqual(report.locked_inputs(root, report.ARMS[0]), record)
                self.assertEqual(report.selected_train(run, cfg, selected)[0], results['train'])
                runner.load_checkpoint.assert_not_called()
                for split in ('train', 'val', 'test'):
                    bad = copy.deepcopy(final)
                    bad[split]['identity']['checkpoint_sha256'] = None
                    write_json(run / 'final_metrics.json', bad)
                    with self.assertRaises(ValueError):
                        report.locked_inputs(root, report.ARMS[0])
                write_json(run / 'final_metrics.json', final)
                bad = copy.deepcopy(results['val'])
                bad['identity']['model_state_sha256'] = 'another-model'
                with self.assertRaises(ValueError):
                    report._bound_split(run, cfg, selected, bad, 'val', 'synthetic-model')

    def test_saved_full18_render_nss_schema_and_fixed36_no_model_or_test_inference(self):
        helper = report.common.reporting_helpers()
        cfg, selected = config(), lock()
        costs = np.array([[1., 2., 3.], [2., 1., 3.], [3., 2., 1.], [1., 2., 3.]])
        winner, pred = np.array([0, 1, 2, 0]), np.array([0, 0, 0, 1])
        baseline = dict(selection_split='train', sbs_solver='m0')
        cfg.update(baselines={p: baseline for p in report.PROBLEMS}, inference_rule='all legal joint-logP argmax',
            candidate_pools={p: ['m0', 'm1', 'm2'] for p in report.PROBLEMS})
        with tempfile.TemporaryDirectory() as temporary:
            args = report.parser().parse_args(['report', '--root', temporary, '--device', 'cpu'])
            run = args.root / args.arm
            run.mkdir()
            packed, predictions, methods, metrics, results, paths = {}, {}, [], [], {}, {}
            sbs, chosen, oracle = costs[:, 0].mean(), costs[np.arange(4), pred].mean(), costs.min(1).mean()
            saved = dict(n=4, top1=.25, top2=.75, top3=1., mean_cost=chosen, sbs_cost=sbs,
                oracle_cost=oracle, vs_sbs_pct=100 * (chosen / sbs - 1),
                vs_oracle_pct=100 * (chosen / oracle - 1), actual_regret_pct=100 * (chosen / oracle - 1),
                sbs_name='m0', sbs_selection_split='train')
            macro = {'macro_' + k: saved[k] for k in ('top1', 'top2', 'top3', 'mean_cost', 'vs_sbs_pct',
                'vs_oracle_pct', 'actual_regret_pct', 'sbs_cost', 'oracle_cost')}
            targets = [(p, m) for p in report.PROBLEMS for m in ('m0', 'm1')]
            target_identity = dict(identity=dict(rule='synthetic fixed36'), target_count=36,
                categories={p + '/' + m: 'synthetic' for p, m in targets})
            for split in ('TRAIN', 'VAL', 'TEST'):
                packed[split], rows = {}, []
                paths[split] = run / split / 'synthetic.npz'
                results[split] = dict(per_problem={p: saved for p in report.PROBLEMS}, macro=macro)
                for p in report.PROBLEMS:
                    data = helper.Predictions(paths[split], ['m0', 'm1', 'm2'], pred, winner, costs,
                        np.arange(4), np.arange(4), np.arange(3), 0)
                    packed[split][p] = data
                    predictions[split, p] = data
                    rows.append(report.common.metric_row(p, split, data, saved, baseline, cfg, selected, data.path))
                    methods.extend(helper.method_rows(p, split, saved, data, 'synthetic'))
                metrics.extend([report.common.macro_row(split, rows, results[split]), *rows])
            for dataset, p in report.LIBS.items():
                count = helper.LIBS[p][1]
                index = np.arange(count) % 4
                data = helper.Predictions(run / (dataset + '.npz'), ['m0', 'm1', 'm2'], pred[index],
                    winner[index], costs[index], np.arange(count), np.arange(count), np.arange(3), 0)
                predictions[dataset, p] = data
                lib_sbs = data.costs[:, 0].mean()
                lib_saved = dict(saved, n=count, top1=data.top1_pct / 100, mean_cost=data.mean_cost,
                    sbs_cost=lib_sbs, vs_sbs_pct=100 * (data.mean_cost / lib_sbs - 1),
                    vs_oracle_pct=100 * (data.mean_cost - 1), actual_regret_pct=100 * (data.mean_cost - 1))
                metrics.append(report.common.metric_row(p, dataset, data, lib_saved, baseline, cfg, selected, run))
                methods.extend(helper.method_rows(p, dataset, lib_saved, data, 'synthetic native LIB'))
            tables = dict(predictions=packed, methods=methods, summaries=[], results=results, paths=paths,
                historical={}, historical_status={}, pairs={}, recovery={}, recovery_summary={}, target_identity=target_identity)
            old = {split: {p: helper.Predictions(d.path, d.pool, np.ones(4, dtype=int), d.winner,
                d.costs, d.indices, d.source_indices, d.pool_ids, 0) for p, d in values.items()}
                for split, values in packed.items()}
            for ref in ('R84A', 'R94B'):
                paired, recovered, totals = report.r109_reporting.comparison_tables(old, packed, targets,
                    target_identity, 'R111/' + args.arm, ref)
                self.assertEqual(len(recovered), 36 * 3)
                tables['pairs'][ref], tables['recovery'][ref], tables['recovery_summary'][ref] = paired, recovered, totals
                tables['historical'][ref] = dict(synthetic_only=True)
                tables['historical_status'][ref] = dict(status='included')
            comparisons = []
            for dataset, p, split in report.r109_reporting.NSS_DATASETS:
                data = predictions[split, p]
                comparisons.append(dict(dataset=dataset, n=data.n, current_checkpoint_sha256=selected['checkpoint_sha256'],
                    nss_checkpoint_sha256='synthetic-NSS', status='complete', nss_top1_native_pct=50.,
                    current_top1_native_pct=data.top1_pct, nss_mean_cost=1.5, current_mean_cost=data.mean_cost,
                    **report.r109_reporting._nss_metadata(cfg, predictions, dataset, p, split)))
            write_json(run / 'history.json', [dict(successful_updates=1440,
                optimization=dict(loss=.5, ce=.5), train=results['TRAIN'], val=results['VAL'])])
            titles = []
            distribution = report.common._distribution
            def capture(*values, **kwargs):
                fig = distribution(*values, **kwargs)
                if values[1] == 'TSP':
                    titles.extend(ax.get_title() for ax in fig.axes)
                    self.assertEqual([len(ax.patches) for ax in fig.axes], [6, 6])
                return fig
            with patch.object(report, '_runner', side_effect=AssertionError('no models')), \
                    patch.object(report, 'locked_inputs', return_value=(run, cfg, selected, results['TEST'], results['VAL'])), \
                    patch.object(report.shared, 'load_saved_splits', return_value=(predictions,
                        [r for r in metrics if r['split'] != 'TRAIN'], methods)), \
                    patch.object(report.shared, 'saved_libs'), \
                    patch.object(report, 'nss_reference', return_value=(Path('/synthetic/NSS'), {})), \
                    patch.object(report, 'nss_comparisons', return_value=(comparisons, {})), \
                    patch.object(report, 'diagnostic_tables', return_value=tables), \
                    patch.object(report, 'r110_summary', return_value=([], {})), \
                    patch.object(report.common, '_distribution', side_effect=capture):
                value = report.report(args)
            output = run / 'full_model_statistics'
            for name in report.REQUIRED_PDFS:
                self.assertTrue((output / name).read_bytes().startswith(b'%PDF-'))
            self.assertTrue(titles[0].startswith('TSP TEST'))
            self.assertTrue(titles[1].startswith('TSPLIB'))
            self.assertIn('\u6a21\u578bTop1=25.00%', titles[0])
            self.assertIn('\u6a21\u578bTop1=26.53%', titles[1])
            self.assertEqual(len(report.read_csv(output / 'metrics_by_problem.csv')), 59)
            rows = report.read_csv(output / 'selection_distribution.csv')
            chosen_rows = [r for r in rows if r['problem'] == 'TSP' and r['split'] == 'TEST' and r['panel'] == 'model_selection']
            self.assertEqual(max(float(r['percentage']) for r in chosen_rows), 75.)
            nss_path = output / 'nss_comparison.csv'
            self.assertFalse(nss_path.read_bytes().startswith(b'\xef\xbb\xbf'))
            nss_path.write_bytes(b'\xef\xbb\xbf' + nss_path.read_bytes())
            self.assertEqual(report.r109_reporting.validate_nss_csv(nss_path, cfg, selected, predictions)['rows'], 4)
            self.assertEqual(value['inference_timing_status'], 'missing_not_benchmarked')
            self.assertEqual(value['scientific_status'], 'provisional_single_arm_historical_controls')
            self.assertEqual(read_json(run / 'postprocess_status.json')['phase'], 'complete')


if __name__ == '__main__':
    unittest.main()
