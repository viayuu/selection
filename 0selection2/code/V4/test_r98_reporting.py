"""Tiny CPU-only saved-data witnesses; never load real contexts or a model."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from . import r98_reporting as report


def arrays():
    gaps = np.array([[0., 0.], [.02, 0.], [0., .2], [0., .03]], np.float64)
    logits, probabilities = report.descriptive_gap_view(gaps)
    return dict(pool=np.array(['alpha', 'beta']), pool_ids=np.array([2, 7], np.int64),
        costs=np.array([[1.+1e-9, 1.], [2., 1.], [1., 3.], [2., 1.]], np.float64),
        winner=np.array([1, 1, 0, 1], np.int64), pred=np.array([0, 1, 0, 0], np.int64),
        expected_gaps=gaps, predG=gaps, logits=logits, scores=logits, probabilities=probabilities,
        indices=np.arange(4, dtype=np.int64), source_indices=np.arange(4, dtype=np.int64),
        release_source_indices=np.arange(100, 104, dtype=np.int64), nodes=np.full(4, 20, np.int64))


def baseline():
    return dict(selection_split='train', train_rows=81000, sbs_solver='alpha', sbs_slot=0)


def saved_metric(data):
    row = report.metric_row(data, baseline(), 'TSP', 'TEST')
    saved = {name: row[key]/100. if key.startswith('top') else row[key]
             for key, name in report.SAVED_KEYS.items()}
    return dict(saved, n=data.n, native_correct=row['native_correct'],
                sbs_name='alpha', sbs_selection_split='train')


def fixture(root):
    root.mkdir()
    common = report.common
    data = report.load_predictions(arrays(), root / 'synthetic.npz')
    contexts = {p: dict(rows=81000, columns=2, metadata_sha256='a'*64) for p in report.PROBLEMS}
    targets = {p: dict(rows=81000, methods=2, metadata_sha256='e'*64) for p in report.PROBLEMS}
    config = dict(experiment='R98', arm='A', seed=2, data_root=str(root / 'synthetic_new90k'),
        rows_per_problem=dict(train=81000, val=4, test=4), contexts=contexts, targets=targets,
        candidate_pools={p: data.pool for p in report.PROBLEMS},
        candidate_pool_ids={p: data.pool_ids.tolist() for p in report.PROBLEMS},
        baselines={p: baseline() for p in report.PROBLEMS},
        regressor_parameters=dict(n_estimators=1, batch_size=1, norm_methods=['none']),
        public_checkpoint_sha256='b'*64, package_version='tabicl==2.0.2', source_revision='fixture-only',
        inference_rule=report.INFERENCE_RULE, prediction_statistic='official TabICLRegressor mean',
        constant_shortcut='exact constant only', optimizer_updates=0, fine_tuning=False,
        historical_TRAIN_labels_in_context=True, current_query_labels_in_forward=False,
        kv_cache=False, allow_cpu_offload=False, flash_attention_only=True,
        descriptive_probabilities=dict(temperature=.01, calibrated=False, used_for_decisions=False))
    common.write_json(root / 'protocol.json', config)
    bank = dict(protocol_sha256=common.json_hash(config), contexts=contexts, targets=targets,
        public_checkpoint_sha256=config['public_checkpoint_sha256'], inference_rule=report.INFERENCE_RULE,
        weights_updated=False, selection_rule=report.SELECTION_RULE)
    common.write_json(root / 'checkpoint.json', bank)
    lock = dict(checkpoint=str(root / 'checkpoint.json'), checkpoint_sha256=common.file_hash(root / 'checkpoint.json'),
        state_hash=common.json_hash(config), selected_without_test=True, per_problem_assembly=False,
        selection_rule=report.SELECTION_RULE)
    common.write_json(root / 'final_test_lock.json', lock)
    common.write_json(root / 'test_input_lock.json', dict(checkpoint_sha256=lock['checkpoint_sha256'],
        inputs={p: {} for p in report.PROBLEMS}, labels_loaded=False, created_after_unique_VAL_lock=True))
    common.write_json(root / 'wandb_receipt.json', dict(enabled=True, mode='offline', run_id='fixture'))
    saved = saved_metric(data)
    macro = {'macro_'+key: value for key, value in saved.items()
             if key not in ('n', 'native_correct', 'sbs_name', 'sbs_selection_split')}
    summary = dict(complete=True, state_hash=lock['state_hash'],
        per_problem={p: saved for p in report.PROBLEMS}, macro=macro)
    packed = {p+'__'+key: value for p in report.PROBLEMS for key, value in arrays().items()}
    for split in ('val', 'test'):
        path = root / 'evaluation' / split / 'selected_best'
        path.mkdir(parents=True)
        np.savez(path / 'rank000.npz', **packed)
        common.write_json(root / (split+'_results.json'), summary)
    library = root / 'lib_evaluation/predictions'
    library.mkdir(parents=True)
    libs = {}
    for dataset, problem in report.LIBS.items():
        values = dict(arrays(), release_source_indices=-1-np.arange(4, dtype=np.int64))
        np.savez(library / (dataset+'.npz'), **values)
        libs[dataset] = dict(saved, complete=True, state_hash=lock['state_hash'],
                            problem=problem, source_directory='synthetic native '+dataset)
    common.write_json(root / 'lib_results.json', libs)
    records = []
    for problem in report.PROBLEMS:
        splits = ['val', 'test', *[d for d, p in report.LIBS.items() if p == problem]]
        for split in splits:
            base = dict(problem=problem, split=split, train_rows=81000)
            records.append(dict(base, stage='load_context', seconds=1.,
                context_read_and_verify_seconds=.4, target_read_and_verify_seconds=.3,
                exact_constant_method_slots=[]))
            for slot in range(2):
                records.append(dict(base, stage='fit_method', seconds=2., method_slot=slot))
                records.append(dict(base, stage='predict_method', seconds=3., method_slot=slot))
            records.append(dict(base, stage='predict', seconds=11., query_rows=4, methods=2,
                sequential_fullcontext_fits=2, sequential_mean_calls=2, exact_constant_method_slots=[],
                kv_cache=False, context_reencoded=True, excludes_graph_to_statistics_time=True,
                includes_all_sequential_method_fits=True))
            if split in report.LIBS:
                records.append(dict(base, stage='native_input_to_statistics', seconds=4.,
                    includes_native_input_loading=True, labels_loaded=False))
            else:
                records.append(dict(base, stage='cached_query_inputs', seconds=.5,
                    fresh_graph_to_statistics_measured=False))
    common.write_json(root / 'timing.json', dict(records=records))
    common.write_json(root / 'resource_witness.json', dict(complete=True, labels_loaded=False,
        accuracy_computed=False, context_rows=81000, query_rows=4,
        checkpoint_sha256=config['public_checkpoint_sha256'], extra_subset_call=False))
    common.write_json(root / 'run_complete.json', dict(complete=True, checkpoint_sha256=lock['checkpoint_sha256']))
    references = root / 'synthetic_references'
    nss = references / 'nss_reference'
    checkpoints = {p: dict(sha256='c'*64) for p in ('TSP', 'CVRP')}
    common.write_json(nss / 'protocol.json', dict(data_root=config['data_root'], retrained_on_new90k=False,
                                                 checkpoints=checkpoints))
    for problem in ('TSP', 'CVRP'):
        for dataset in ('test', problem+'LIB'):
            path = nss / problem
            path.mkdir(exist_ok=True)
            np.savez(path / (dataset+'_predictions.npz'), **arrays())
            common.write_json(path / (dataset+'_metrics.json'), dict(checkpoint=checkpoints[problem],
                top1_native_ind=.5, mean_cost=float(data.selected_costs.mean())))
    for name, experiment, arm in (('r84', 'R84', 'A_baseline'), ('r89', 'R89', 'B_muon_hybrid')):
        path = references / name
        old_config = dict(config, experiment=experiment, arm=arm)
        old_lock = dict(lock, state_hash=common.json_hash(old_config), checkpoint_sha256='d'*64)
        common.write_json(path / 'protocol.json', old_config)
        common.write_json(path / 'final_test_lock.json', old_lock)
        common.write_json(path / 'test_results.json', dict(complete=True, checkpoint_sha256='d'*64, result=summary))
        archive = path / 'evaluation/test/selected_best'
        archive.mkdir(parents=True)
        np.savez(archive / 'rank000.npz', **packed)
    target_path = references / 'target_list.json'
    common.write_json(target_path, dict(identity=dict(data_root=config['data_root'], baseline_checkpoint_sha256='d'*64),
        targets=[dict(problem=p, method=name, category='fixture') for p in report.PROBLEMS for name in data.pool]))
    return SimpleNamespace(root=root, device='cpu', nss_root=nss, r84_root=references / 'r84',
                           r89_root=references / 'r89', target_list=target_path)


class ReportingTests(unittest.TestCase):
    def test_import_has_no_models_or_mutation(self):
        script = '''
import importlib.abc, os, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'tabicl'):
            raise AssertionError('model import: ' + fullname)
sys.meta_path.insert(0, NoModels())
before = dict(os.environ)
from code.V4 import r97_reporting as old
old_notes = old.NOTES
from code.V4 import r98_reporting as r
assert dict(os.environ) == before
assert old.NOTES is old_notes
assert 'code.V4.r98_experiment' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
assert r.parser().parse_args(['report', '--device', 'cpu']).device == 'cpu'
'''
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=Path(__file__).resolve().parents[2],
            env=dict(os.environ), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_fp64_metrics_not_fp32_or_ratio_of_means(self):
        data = report.load_predictions(arrays(), '/tmp/synthetic-r98.npz')
        row = report.metric_row(data, baseline(), 'TSP', 'TEST')
        self.assertEqual(row['model'], 'R98')
        self.assertEqual(row['top1_pct'], 50.)
        self.assertEqual(row['top1_fp32_argmin_pct'], 75.)
        self.assertAlmostEqual(row['actual_regret_pct'], 25.000000025)
        scaled = replace(data, costs=data.costs*np.array([1., 10., 1., 10.])[:, None])
        other = report.metric_row(scaled, baseline(), 'TSP', 'TEST')
        self.assertAlmostEqual(other['actual_regret_pct'], row['actual_regret_pct'])
        self.assertNotAlmostEqual(other['actual_regret_pct'], other['vs_oracle_pct'])
        macro = report.macro_row([row, dict(row, n=40, top1_pct=100.)], 'TEST')
        self.assertEqual(macro['top1_pct'], 75.)
        self.assertEqual(macro['n'], 44)

    def test_raw_gap_rank_survives_descriptive_rounding_and_negative_means(self):
        gaps = np.array([[-1e-300, -2e-300], [-.02, -.03], [1., 1.]], np.float64)
        rank = report.stable_gap_rank(gaps, np.array([7, 2]))
        np.testing.assert_array_equal(rank[:, 0], [1, 1, 1])
        logits, probabilities = report.descriptive_gap_view(gaps)
        np.testing.assert_array_equal(probabilities[0], [.5, .5])
        np.testing.assert_allclose(np.exp(logits).sum(1), 1.)
        self.assertTrue((gaps < 0).any())
        with self.assertRaises(ValueError):
            report.stable_gap_rank(np.array([[np.nan, 0.]]), np.array([2, 7]))

    def test_loader_rejects_probability_decision_and_mismatched_gap_alias(self):
        values = arrays()
        values['expected_gaps'] = values['expected_gaps'].copy()
        values['expected_gaps'][0] = [1e-300, 0.]
        values['predG'] = values['expected_gaps']
        values['logits'], values['probabilities'] = report.descriptive_gap_view(values['expected_gaps'])
        values['scores'] = values['logits']
        with self.assertRaisesRegex(AssertionError, 'raw mean gap'):
            report.load_predictions(values, '/tmp/synthetic-r98.npz')
        values['pred'] = values['pred'].copy()
        values['pred'][0] = 1
        data = report.load_predictions(values, '/tmp/synthetic-r98.npz')
        self.assertEqual(data.pred[0], 1)
        with self.assertRaises(AssertionError):
            report.load_predictions(dict(values, predG=np.zeros((4, 2))), '/tmp/mismatch.npz')

    def test_paired_plot_is_r98_and_same_axis_test_then_lib(self):
        data = report.load_predictions(arrays(), '/tmp/synthetic-r98.npz')
        figure = report.plot_distribution('TSP', [('TEST', data), ('TSPLIB', data)], data.pool,
            'synthetic-new90k', dict(checkpoint_sha256='f'*64))
        try:
            self.assertIn('R98', figure._suptitle.get_text())
            self.assertNotIn('R97', figure._suptitle.get_text())
            self.assertIn('TSP TEST', figure.axes[0].get_title())
            self.assertIn('TSPLIB', figure.axes[1].get_title())
            for ax in figure.axes:
                self.assertIn('\u6a21\u578bTop1=50.00%', ax.get_title())
                self.assertEqual(len(ax.containers), 2)
                np.testing.assert_allclose([b.get_width() for b in ax.containers[0]], [75., 25.])
                np.testing.assert_allclose([b.get_width() for b in ax.containers[1]], [25., 75.])
        finally:
            figure.clear()

    def test_report_full18_cpu_fixture_and_nested_timing(self):
        with tempfile.TemporaryDirectory(prefix='r98-report-test-') as directory:
            args = fixture(Path(directory) / 'run')
            receipt = args.root / 'run_complete.json'
            complete = report.common.read_json(receipt)
            report.common.write_json(receipt, dict(complete, complete=False))
            with self.assertRaisesRegex(ValueError, 'not complete'):
                report.report(args)
            self.assertFalse((args.root / 'full_model_statistics').exists())
            report.common.write_json(receipt, complete)
            result = report.report(args)
            self.assertEqual(result, report.common.read_json(args.root / 'postprocess_status.json'))
            self.assertTrue(result['complete'])
            for key in ('results_pdf', 'model_statistics_pdf', 'selection_distributions_pdf', 'target_recovery_pdf'):
                self.assertTrue(Path(result['artifacts'][key]).read_bytes().startswith(b'%PDF'))
            self.assertEqual((args.root / 'RESULTS.pdf').read_bytes(),
                Path(result['artifacts']['model_statistics_pdf']).read_bytes())
            config = report.common.read_json(args.root / 'protocol.json')
            resources, _, _, _ = report.context_resources(args.root, config)
            tsp = resources[0]
            self.assertEqual(tsp['sequential_fullcontext_fits'], 6)
            self.assertEqual(tsp['fit_method_seconds'], 12.)
            self.assertEqual(tsp['predict_method_seconds'], 18.)
            self.assertEqual(tsp['predict_including_fits_seconds'], 33.)
            self.assertEqual(tsp['recorded_call_seconds'], 41.)
            self.assertIsNone(tsp['val_test_fresh_graph_to_statistics_seconds'])
            text = (args.root / 'RESULTS.md').read_text()
            self.assertNotIn('R97/A', text)
            self.assertIn('NOT calibrated', text)
            self.assertIn('N/A', text)
            provenance = report.common.read_json(result['artifacts']['provenance_json'])
            self.assertFalse(provenance['inference_performed'])
            self.assertIsNone(provenance['train_self_accuracy'])
            self.assertTrue(provenance['ranking_uses_raw_expected_gaps'])
            rows = (args.root / 'full_model_statistics/paired_comparison.csv').read_text()
            self.assertIn('R89B', rows)
            self.assertNotIn('R97/A', rows)

    def test_lock_and_missing_sequential_calls_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory) / 'run')
            config = report.common.read_json(args.root / 'protocol.json')
            timing_path = args.root / 'timing.json'
            timing = report.common.read_json(timing_path)
            timing['records'].pop(1)
            report.common.write_json(timing_path, timing)
            with self.assertRaisesRegex(ValueError, 'per-method'):
                report.context_resources(args.root, config)
            bank_path = args.root / 'checkpoint.json'
            bank = report.common.read_json(bank_path)
            report.common.write_json(bank_path, dict(bank, weights_updated=True))
            with self.assertRaisesRegex(ValueError, 'unchanged unique'):
                report.locked_inputs(args.root)


if __name__ == '__main__':
    unittest.main()
