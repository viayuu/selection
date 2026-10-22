"""Small CPU fixtures only: no runner, real reports, context labels or GPU."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from . import r97_reporting as report


def arrays():
    probabilities = np.array([[.5, .5], [.2, .8], [.9, .1], [.8, .2]])
    costs = np.array([[1. + 1e-9, 1.], [2., 1.], [1., 3.], [2., 1.]], np.float64)
    return dict(pool=np.array(['alpha', 'beta']), pool_ids=np.array([2, 7], np.int64),
        costs=costs, winner=np.array([1, 1, 0, 1], np.int64),
        pred=probabilities.argmax(1).astype(np.int64), logits=np.log(probabilities),
        scores=np.log(probabilities), probabilities=probabilities,
        indices=np.arange(4, dtype=np.int64), source_indices=np.arange(4, dtype=np.int64),
        release_source_indices=np.arange(100, 104, dtype=np.int64), nodes=np.full(4, 20, np.int64))


def baseline():
    return dict(selection_split='train', train_rows=81000, sbs_solver='alpha', sbs_slot=0)


def saved_metric(data):
    row = report.metric_row(data, baseline(), 'TSP', 'TEST')
    saved = {name: row[key] / 100. if key.startswith('top') else row[key]
             for key, name in report.SAVED_KEYS.items()}
    saved.update(n=data.n, native_correct=row['native_correct'], sbs_name='alpha', sbs_selection_split='train')
    return saved


def fixture(root):
    """A four-row full18 *synthetic* run, with no model or context arrays."""
    root.mkdir()
    data = report.load_predictions(arrays(), root / 'synthetic.npz')
    contexts = {p: dict(rows=81000, columns=2, metadata_sha256='a' * 64) for p in report.PROBLEMS}
    config = dict(experiment='R97', arm='A', data_root=str(root / 'synthetic_new90k'),
        rows_per_problem=dict(train=81000, val=4, test=4),
        candidate_pools={p: data.pool for p in report.PROBLEMS},
        candidate_pool_ids={p: data.pool_ids.tolist() for p in report.PROBLEMS},
        baselines={p: baseline() for p in report.PROBLEMS}, contexts=contexts,
        classifier_parameters=dict(n_estimators=1, norm_methods=['none']),
        public_checkpoint_sha256='b' * 64, package_version='tabicl==2.0.2', source_revision='fixture-only',
        inference_rule='legal-class probability argmax; ascending global-ID exact ties',
        optimizer_updates=0, fine_tuning=False, historical_TRAIN_labels_in_context=True,
        current_query_labels_in_forward=False)
    common = report.common
    common.write_json(root / 'protocol.json', config)
    bank = dict(protocol_sha256=common.json_hash(config), contexts=contexts,
        public_checkpoint_sha256=config['public_checkpoint_sha256'], inference_rule=config['inference_rule'],
        weights_updated=False, selection_rule=report.SELECTION_RULE)
    common.write_json(root / 'checkpoint.json', bank)
    lock = dict(checkpoint=str(root / 'checkpoint.json'), checkpoint_sha256=common.file_hash(root / 'checkpoint.json'),
        state_hash=common.json_hash(config), selected_without_test=True, per_problem_assembly=False,
        selection_rule=report.SELECTION_RULE)
    common.write_json(root / 'final_test_lock.json', lock)
    per_problem = {p: saved_metric(data) for p in report.PROBLEMS}
    macro = {'macro_' + key: value for key, value in saved_metric(data).items()
             if key not in ('n', 'native_correct', 'sbs_name', 'sbs_selection_split')}
    summary = dict(complete=True, state_hash=lock['state_hash'], per_problem=per_problem, macro=macro)
    packed = {p + '__' + key: value for p in report.PROBLEMS for key, value in arrays().items()}
    for split in ('val', 'test'):
        path = root / 'evaluation' / split / 'selected_best'
        path.mkdir(parents=True)
        np.savez(path / 'rank000.npz', **packed)
        common.write_json(root / (split + '_results.json'), summary)
    library = root / 'lib_evaluation/predictions'
    library.mkdir(parents=True)
    libs = {}
    for dataset, problem in report.LIBS.items():
        values = arrays()
        values['release_source_indices'] = -1 - np.arange(4, dtype=np.int64)
        np.savez(library / (dataset + '.npz'), **values)
        libs[dataset] = dict(saved_metric(data), complete=True, state_hash=lock['state_hash'],
                             problem=problem, source_directory='synthetic native ' + dataset)
    common.write_json(root / 'lib_results.json', libs)
    records = []
    for problem in report.PROBLEMS:
        splits = ['val', 'test', *[name for name, p in report.LIBS.items() if p == problem]]
        for split in splits:
            for stage, seconds in (('fit_context', 2.), ('predict', 3.)):
                record = dict(problem=problem, split=split, stage=stage, seconds=seconds, train_rows=81000,
                              peak_sampled_rss_bytes=1024)
                if stage == 'fit_context':
                    record.update(total_seconds=2.75, context_read_and_verify_seconds=.25)
                else:
                    record.update(kv_cache=False, context_reencoded=True, excludes_graph_to_statistics_time=True)
                records.append(record)
            if split in report.LIBS:
                records.append(dict(problem=problem, split=split, stage='native_input_to_statistics',
                                    seconds=4., includes_native_input_loading=True))
    common.write_json(root / 'timing.json', dict(records=records))
    common.write_json(root / 'resource_witness.json', dict(complete=True, labels_loaded=False, accuracy_computed=False,
        checkpoint_sha256=config['public_checkpoint_sha256'], context_rows=81000,
        fit=dict(seconds=2., total_seconds=2.75, context_read_and_verify_seconds=.25), predict=dict(seconds=.5)))
    common.write_json(root / 'run_complete.json', dict(complete=True, checkpoint_sha256=lock['checkpoint_sha256']))

    references = root / 'synthetic_references'
    nss = references / 'nss_reference'
    checkpoints = {p: dict(sha256='c' * 64) for p in ('TSP', 'CVRP')}
    common.write_json(nss / 'protocol.json', dict(data_root=config['data_root'], retrained_on_new90k=False,
                                                 checkpoints=checkpoints))
    for problem in ('TSP', 'CVRP'):
        for dataset in ('test', problem + 'LIB'):
            path = nss / problem
            path.mkdir(exist_ok=True)
            np.savez(path / (dataset + '_predictions.npz'), **arrays())
            common.write_json(path / (dataset + '_metrics.json'), dict(checkpoint=checkpoints[problem],
                top1_native_ind=.5, mean_cost=float(data.selected_costs.mean())))
    for name, experiment, arm in (('r84', 'R84', 'A_baseline'), ('r89', 'R89', 'B_muon_hybrid')):
        path = references / name
        old_config = dict(config, experiment=experiment, arm=arm)
        old_lock = dict(lock, state_hash=common.json_hash(old_config), checkpoint_sha256='d' * 64)
        common.write_json(path / 'protocol.json', old_config)
        common.write_json(path / 'final_test_lock.json', old_lock)
        common.write_json(path / 'test_results.json', dict(complete=True, checkpoint_sha256='d' * 64, result=summary))
        archive = path / 'evaluation/test/selected_best'
        archive.mkdir(parents=True)
        np.savez(archive / 'rank000.npz', **packed)
    targets = references / 'target_list.json'
    common.write_json(targets, dict(identity=dict(data_root=config['data_root'], baseline_checkpoint_sha256='d' * 64),
        targets=[dict(problem=p, method=name, category='fixture') for p in report.PROBLEMS for name in data.pool]))
    return SimpleNamespace(root=root, device='cpu', nss_root=nss, r84_root=references / 'r84',
                           r89_root=references / 'r89', target_list=targets)


class R97ReportingTests(unittest.TestCase):
    def test_import_and_cli_are_cpu_only_without_environment_mutation(self):
        script = """
import importlib.abc, os, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'tabicl'):
            raise AssertionError('model import: ' + fullname)
sys.meta_path.insert(0, NoModels())
before = dict(os.environ)
from code.V4 import r97_reporting as r
assert dict(os.environ) == before
assert r.parser().parse_args(['report', '--root', '/tmp/fixture', '--device', 'cpu']).device == 'cpu'
assert 'code.V4.r97_experiment' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
"""
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=Path(__file__).resolve().parents[2],
                                env=dict(os.environ), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_near_tie_stable_ids_and_equal_task_macro(self):
        data = report.load_predictions(arrays(), '/tmp/synthetic-r97.npz')
        row = report.metric_row(data, baseline(), 'TSP', 'TEST')
        self.assertEqual(row['top1_pct'], 50.)
        self.assertEqual(row['top1_fp32_argmin_pct'], 75.)
        self.assertEqual(data.pred[0], 0)
        self.assertEqual(data.winner[0], 1)
        self.assertAlmostEqual(row['actual_regret_pct'], 25.000000025)
        scaled = replace(data, costs=data.costs * np.array([1., 10., 1., 10.])[:, None])
        varied = report.metric_row(scaled, baseline(), 'TSP', 'TEST')
        self.assertAlmostEqual(varied['actual_regret_pct'], row['actual_regret_pct'])
        self.assertNotAlmostEqual(varied['actual_regret_pct'], varied['vs_oracle_pct'])
        more = dict(row, problem='CVRP', n=40, top1_pct=100., mean_cost=8.)
        macro = report.macro_row([row, more], 'TEST')
        self.assertEqual(macro['top1_pct'], 75.)
        self.assertAlmostEqual(macro['mean_cost'], (row['mean_cost'] + 8.) / 2.)
        self.assertEqual(macro['n'], 44)
        with self.assertRaises(ValueError):
            report.load_predictions(dict(arrays(), costs=data.costs.astype(np.float32)), '/tmp/not-fp64.npz')

    def test_paired_figure_uses_native_top1_and_test_then_lib(self):
        data = report.load_predictions(arrays(), '/tmp/synthetic-r97.npz')
        lib = replace(data, pool=list(reversed(data.pool)), costs=data.costs[:, ::-1],
                      winner=1-data.winner, pred=1-data.pred)
        figure = report.plot_distribution('TSP', [('TEST', data), ('TSPLIB', lib)], data.pool,
                                          'fixture-new90k', dict(checkpoint_sha256='f' * 64))
        try:
            self.assertEqual(len(figure.axes), 2)
            self.assertIn('TSP TEST', figure.axes[0].get_title())
            self.assertIn('TSPLIB', figure.axes[1].get_title())
            for ax in figure.axes:
                self.assertIn('\u6a21\u578bTop1=50.00%', ax.get_title())
                self.assertEqual(len(ax.containers), 2)
                np.testing.assert_allclose([bar.get_width() for bar in ax.containers[0]], [75., 25.])
                np.testing.assert_allclose([bar.get_width() for bar in ax.containers[1]], [25., 75.])
        finally:
            figure.clear()

    def test_reference_remaps_local_native_labels_and_counts_real_recovery(self):
        data = report.load_predictions(arrays(), '/tmp/synthetic-r97.npz')
        before = replace(data, pred=np.array([1, 0, 0, 1], np.int64))
        row_order = np.array([3, 1, 0, 2])
        reference = replace(before, pool=data.pool[::-1], pool_ids=data.pool_ids[::-1],
            costs=before.costs[row_order, ::-1], pred=1-before.pred[row_order], winner=1-before.winner[row_order],
            indices=before.indices[row_order], nodes=before.nodes[row_order],
            source_indices=before.source_indices[row_order], release_source_indices=before.release_source_indices[row_order])
        aligned = report.align_reference(data, reference, require_source=True)
        np.testing.assert_array_equal(aligned.pred, before.pred)
        rows = report.target_recovery({'TSP': data}, {'TSP': reference},
            [dict(problem='TSP', method=name) for name in data.pool], 'R89B')
        alpha, beta = rows
        self.assertEqual((alpha['new_false_pick_count'], beta['rescued'], beta['lost']), (2, 1, 2))
        self.assertEqual(beta['net_recovery'], -1)
        pair = report.paired_comparison({'TSP': data}, {'TSP': reference}, 'R89B')[0]
        self.assertEqual((pair['corrected'], pair['harmed'], pair['net_correct']), (1, 2, -1))
        self.assertEqual(pair['top1_delta_pp'], -25.)
        changed = reference.costs.copy()
        changed[-1, 0] += .01
        with self.assertRaises(AssertionError):
            report.align_reference(data, replace(reference, costs=changed))

    def test_tiny_full18_report_completion_paths_and_context_fit_cost(self):
        with tempfile.TemporaryDirectory(prefix='r97-report-fixture-') as directory:
            args = fixture(Path(directory) / 'run')
            receipt = args.root / 'run_complete.json'
            complete = report.common.read_json(receipt)
            report.common.write_json(receipt, dict(complete, complete=False))
            with self.assertRaisesRegex(ValueError, 'not complete'):
                report.report(args)
            self.assertFalse((args.root / 'full_model_statistics').exists())
            report.common.write_json(receipt, complete)
            result = report.report(args)
            status = report.common.read_json(args.root / 'postprocess_status.json')
            self.assertEqual(result, status)
            self.assertTrue(status['complete'])
            self.assertEqual(status['phase'], 'complete')
            for key in ('model_statistics_pdf', 'selection_distributions_pdf', 'target_recovery_pdf',
                        'results_pdf', 'results_md', 'context_resource_summary_csv', 'provenance_json'):
                self.assertTrue(Path(status['artifacts'][key]).is_file(), key)
            self.assertEqual(Path(status['artifacts']['model_statistics_pdf']),
                             args.root / 'full_model_statistics/MODEL_STATISTICS.pdf')
            self.assertEqual((args.root / 'RESULTS.pdf').read_bytes(),
                             Path(status['artifacts']['model_statistics_pdf']).read_bytes())
            resources, _, _, _ = report.context_resources(args.root, report.common.read_json(args.root / 'protocol.json'))
            tsp = resources[0]
            self.assertEqual(tsp['fit_context_seconds'], 6.)
            self.assertEqual(tsp['fit_context_total_seconds'], 8.25)
            self.assertEqual(tsp['context_read_and_verify_seconds'], .75)
            self.assertEqual(tsp['predict_seconds'], 9.)
            self.assertEqual(tsp['lib_native_input_to_statistics_seconds'], 4.)
            self.assertEqual(tsp['recorded_call_seconds'], 21.75)
            self.assertIsNone(tsp['val_test_fresh_graph_to_statistics_seconds'])
            text = (args.root / 'RESULTS.md').read_text()
            self.assertIn('N/A', text)
            self.assertIn('NOT matched compute', text)
            self.assertIn('Fresh raw-graph-to-statistics/cache-build latency is excluded', text)
            self.assertFalse(list(args.root.rglob('*learning_curve*')))
            provenance = report.common.read_json(status['artifacts']['provenance_json'])
            self.assertFalse(provenance['inference_performed'])
            self.assertIsNone(provenance['train_self_accuracy'])


if __name__ == '__main__':
    unittest.main()
