"""Small CPU-only reporting checks, including two-candidate B tasks."""

import copy
import io
import json
import re
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from code.unified_selector.registry import PROBLEMS, S2I
from . import r125_reporting, r126_reporting
from . import r127_reporting as reporter


def config_fixture():
    pools = {problem: reporter.expected_pool(problem) for problem in PROBLEMS}
    aliases = {'reld_mt': 'RELD_MOEL'}
    return dict(experiment='R127', arm=reporter.ARM, model_factory=reporter.FACTORY + '.load_checkpoint',
        seed=2, rows_per_problem=dict(reporter.SPLIT_ROWS), logical_global_batch=1024,
        max_successful_updates=57600, warmup_updates=4320, eval_every_updates=1440,
        candidate_pools=pools, candidate_pool_ids={problem: [S2I[aliases.get(name, name)] for name in pool]
            for problem, pool in pools.items()}, scenario_solver_aliases=aliases,
        data_root='/tmp/r127-frozen-no-mvmoe', source_dataset_root='/tmp/r84-frozen-full',
        checkpoint_rule='minimum full18 VAL macro_vs_sbs_pct; first wins ties',
        inference_rule='original maximin; ascending global-ID tie break',
        removed_candidates=['MVMOE'], retained_candidates=['BQ', 'LEHD', 'MoSES_RF'],
        source_scope=reporter.SOURCE_SCOPE, role_policy=dict(reporter.ROLE_POLICY),
        model_only_scope={'no_neural_model_solvers': ['GLOP']}, model_params={'solver_representation': 'code'},
        baselines={problem: dict(selection_split='train', sbs_solver=pool[0], sbs_slot=0,
            mean_cost_by_solver=dict.fromkeys(pool, 1.)) for problem, pool in pools.items()})


def lock_fixture():
    return dict(experiment='R127', arm=reporter.ARM, epoch=1, selected_update=1440,
        checkpoint_sha256='a' * 64, state_hash='b' * 64, rule='minimum VAL',
        selected_without_test=True, per_problem_assembly=False)


def prediction_fixture(problem='TSP', n=4):
    helper, config = reporter.common.reporting_helpers(), config_fixture()
    pool = config['candidate_pools'][problem]
    costs = np.full((n, len(pool)), 4., np.float64)
    costs[np.arange(n), np.arange(n) % len(pool)] = 1.
    return helper.Predictions(path=Path('/tmp/synthetic-r127-' + problem + '.npz'), pool=pool,
        pred=np.zeros(n, np.int64), winner=costs.argmin(1), costs=costs, indices=np.arange(n),
        source_indices=np.arange(n), pool_ids=np.asarray(config['candidate_pool_ids'][problem]), score_ties=0)


def metric_fixture(problem, split, data):
    from .r76_experiment import fixed_sbs_metrics
    helper, config = reporter.common.reporting_helpers(), config_fixture()
    scores = np.zeros_like(data.costs, np.float32)
    scores[:, 0] = 1.
    saved, _ = fixed_sbs_metrics(scores,
        dict(costs=data.costs, winner=data.winner, pool=data.pool, pool_ids=data.pool_ids.tolist()),
        config['baselines'][problem])
    return (reporter.common.metric_row(problem, split, data, saved,
        config['baselines'][problem], config, lock_fixture(), data.path),
        helper.method_rows(problem, split, saved, data, 'synthetic'))


def nss_archive(current, problem):
    pool = list(reporter.ORIGINAL_POOLS[problem])
    costs = np.full((current.n, len(pool)), .5, np.float64)
    scores = np.zeros_like(costs, dtype=np.float32)
    for slot, name in enumerate(current.pool):
        costs[:, pool.index(name)] = current.costs[:, slot]
    scores[:, pool.index('BQ')] = 20.
    if 'MoSES_RF' in pool:
        scores[:, pool.index('MoSES_RF')] = 90.
    if 'MVMOE' in pool:
        scores[:, pool.index('MVMOE')] = 100.
    return dict(pool=np.asarray(pool), costs=costs, winner=costs.argmin(1), scores=scores,
        pred=scores.argmax(1), indices=current.indices.copy(), source_indices=np.arange(current.n) + 10000)


class R127ReportingTests(unittest.TestCase):
    def test_runner_cli_factory_and_budget_contract(self):
        from . import r127_experiment as runner
        self.assertEqual((reporter.ARMS, reporter.MAX_UPDATES, reporter.DEFAULT_ROOT),
            (runner.ARMS, runner.MAX_UPDATES, runner.DEFAULT_ROOT))
        self.assertEqual(reporter.FACTORY + '.load_checkpoint', runner.MODEL_FACTORY)
        self.assertEqual(reporter._ENGINE.FACTORY, 'code.V4.r127_experiment')
        for command in ('lib', 'report', 'all', 'status'):
            args = reporter.parser().parse_args([command])
            self.assertEqual((args.arm, args.device, args.lib_batch_size), ('model_only_seed2', 'cpu', 2))
        with patch.object(reporter, 'lib_stage') as lib, patch.object(reporter, 'report', return_value={}) as report:
            with redirect_stdout(io.StringIO()):
                reporter.main(['all'])
            lib.assert_called_once()
            report.assert_called_once()

    def test_r125_r126_sources_and_namespaces_are_unchanged(self):
        digests = {module: reporter.file_hash(Path(module.__file__)) for module in (r125_reporting, r126_reporting)}
        reporter.validate_config(config_fixture())
        self.assertEqual(r125_reporting.REMOVED, frozenset(('BQ', 'LEHD')))
        self.assertEqual(r126_reporting.REMOVED, frozenset(('MoSES_RF',)))
        self.assertIsNot(reporter.common, r126_reporting.common)
        self.assertIsNot(reporter.load_saved_splits.__globals__, r126_reporting.load_saved_splits.__globals__)
        for module, digest in digests.items():
            self.assertEqual(reporter.file_hash(Path(module.__file__)), digest)
        self.assertEqual(digests[r126_reporting], reporter.TEMPLATE_SHA256)

    def test_all_pools_preserve_legal_rf_and_all_b_tasks_have_two_candidates(self):
        config = config_fixture()
        reporter.validate_config(config)
        b_problems = []
        for problem in PROBLEMS:
            pool = config['candidate_pools'][problem]
            k = {'TSP': 8, 'CVRP': 9, 'ATSP': 5}.get(problem, 2 if 'B' in problem else 5)
            self.assertEqual(len(pool), k)
            self.assertNotIn('MVMOE', pool)
            if problem in ('TSP', 'CVRP'):
                self.assertTrue({'BQ', 'LEHD'} <= set(pool))
            if problem == 'CVRP' or problem in PROBLEMS[3:] and 'B' not in problem:
                self.assertIn('MoSES_RF', pool)
            if 'B' in problem:
                b_problems.append(problem)
                self.assertEqual(pool, ['MTPOMO', 'reld_mt'])
        self.assertEqual(len(b_problems), 8)
        rows = reporter.pool_statistics(config)
        self.assertEqual(len(rows), 18)
        for row in rows:
            if row['problem'] in b_problems:
                self.assertEqual(row['structural_unordered_pairs'], 1)
                self.assertEqual((row['top2_effective_k'], row['top3_effective_k']), (2, 2))

    def test_rejects_inherited_rf_deletion_or_non_model_only_roles(self):
        for mutate in (
                lambda value: value['candidate_pools']['CVRP'].remove('MoSES_RF'),
                lambda value: value['candidate_pools']['VRPB'].append('MVMOE'),
                lambda value: value['role_policy'].update(config=True),
                lambda value: value['baselines']['VRPB'].update(selection_split='test')):
            config = config_fixture()
            mutate(config)
            with self.assertRaises(ValueError):
                reporter.validate_config(config)

    def test_selected_val_capture_none_hash_is_bound_in_copy(self):
        lock = lock_fixture()
        identity = dict(experiment='R127', arm=reporter.ARM, successful_updates=1440,
            state_hash=lock['state_hash'], model_state_sha256='c' * 64)
        val = dict(identity=dict(identity, split='val', checkpoint_sha256=None))
        before = copy.deepcopy(val)
        bound = reporter.bind_selected_val(val, lock,
            dict(identity, split='test', checkpoint_sha256=lock['checkpoint_sha256']))
        self.assertEqual(val, before)
        self.assertEqual(bound['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
        self.assertIsNone(bound['archive_capture_identity']['checkpoint_sha256'])

    def test_b2_saved_metrics_top2_top3_are_saturated_not_top1(self):
        data = prediction_fixture('VRPB')
        row, methods = metric_fixture('VRPB', 'TEST', data)
        self.assertEqual(data.pool, ['MTPOMO', 'reld_mt'])
        self.assertEqual((row['top1_pct'], row['top2_pct'], row['top3_pct']), (50., 100., 100.))
        self.assertEqual(len(methods), 2)
        reporter.validate_prediction(data, 'VRPB', config_fixture(), expected_rows=4)

    def test_native_lib_tsp_unchanged_cvrp_removes_mvmoe_not_rf(self):
        for problem in ('TSP', 'CVRP'):
            pool = list(reporter.ORIGINAL_POOLS[problem])
            costs = np.full((2, len(pool)), 10., np.float64)
            costs[:, pool.index('BQ')] = 1. + 2e-8
            winner_name = 'LEHD' if problem == 'TSP' else 'MoSES_RF'
            costs[:, pool.index(winner_name)] = 1. + 1e-8
            if 'MVMOE' in pool:
                costs[:, pool.index('MVMOE')] = .5
            raw = dict(pool=pool, costs=costs, winner=costs.argmin(1))
            retained = reporter.reduced_raw(raw, problem, config_fixture())
            self.assertEqual(len(retained['pool']), 8 if problem == 'TSP' else 9)
            self.assertNotIn('MVMOE', retained['pool'])
            np.testing.assert_array_equal(retained['winner'], [retained['pool'].index(winner_name)] * 2)
            self.assertNotEqual(retained['winner'][0], retained['costs'].astype(np.float32).argmin(1)[0])
            if problem == 'TSP':
                np.testing.assert_array_equal(retained['costs'], costs)
            else:
                self.assertIn('MoSES_RF', retained['pool'])

    def test_nss_retains_rf_reselects_mvmoe_and_rejects_r126_scores(self):
        for problem in ('TSP', 'CVRP'):
            current = prediction_fixture(problem)
            archive = nss_archive(current, problem)
            selected = 'BQ' if problem == 'TSP' else 'MoSES_RF'
            data = reporter.reselect_nss(archive, Path('/tmp/nss.npz'), current)
            np.testing.assert_array_equal(data.pred, [data.pool.index(selected)] * 4)
            np.testing.assert_array_equal(data.winner, current.winner)
            if problem == 'CVRP':
                keep = [i for i, name in enumerate(archive['pool']) if name != 'MoSES_RF']
                archive['pool'], archive['costs'], archive['scores'] = (
                    archive['pool'][keep], archive['costs'][:, keep], archive['scores'][:, keep])
                with self.assertRaises(ValueError):
                    reporter.reselect_nss(archive, Path('/tmp/r126-not-reusable.npz'), current)

    def test_same4500_nss_four_dataset_diagnostic_and_missing_entries(self):
        config, lock = config_fixture(), lock_fixture()
        predictions = {(split, problem): prediction_fixture(problem, n)
            for split, problem, n in (('TEST', 'TSP', 4500), ('TEST', 'CVRP', 4500),
                                     ('TSPLIB', 'TSP', 49), ('CVRPLIB', 'CVRP', 100))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, _, _ = reporter.nss_comparisons(config, lock, predictions, root)
            self.assertTrue(all(row['nss_top1_native_pct'] is None for row in rows))
            reporter.write_json(root / 'protocol.json', dict(data_root=config['source_dataset_root'],
                checkpoints={problem: {'sha256': 'd' * 64} for problem in ('TSP', 'CVRP')}))
            for (split, problem), data in predictions.items():
                name = 'test' if split == 'TEST' else split
                path = root / problem / (name + '_predictions.npz')
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, **nss_archive(data, problem))
                reporter.write_json(path.with_name(name + '_metrics.json'), dict(checkpoint={'sha256': 'd' * 64}))
            rows, alignment, _ = reporter.nss_comparisons(config, lock, predictions, root)
            self.assertTrue(all(row['status'] == 'fixed_NSS_same_pool_diagnostic' for row in rows))
            self.assertTrue(all(not row['nss_training_matched'] and not row['nss_new_inference'] for row in rows))
            self.assertEqual(alignment['TSP TEST']['removed_candidates'], [])
            self.assertEqual(alignment['CVRP TEST']['removed_candidates'], ['MVMOE'])

    def test_b2_method_diagnostic_pdf_has_no_three_candidate_gate(self):
        helper = reporter.common.reporting_helpers()
        helper.setup_style()
        predictions, methods = {}, []
        for split in ('VAL', 'TEST'):
            data = prediction_fixture('VRPB')
            predictions[split, 'VRPB'] = data
            methods.extend(metric_fixture('VRPB', split, data)[1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('code.unified_selector.registry.PROBLEMS', ['VRPB']):
                result = reporter.common.diagnostic_pdf(root, root, predictions, methods,
                    'R127/model_only_seed2', lock_fixture())
            self.assertEqual(len(re.findall(rb'/Type /Page\b', Path(result['path']).read_bytes())), 2)

    def test_synthetic_main_pdf_b2_chart_captions_fonts_and_manifest(self):
        from matplotlib.backends.backend_pdf import PdfPages
        helper, config, lock = reporter.common.reporting_helpers(), config_fixture(), lock_fixture()
        problems = ['TSP', 'CVRP', 'VRPB']
        predictions, metrics, methods = {}, [], []
        for split in ('VAL', 'TEST'):
            rows = []
            for problem in problems:
                data = prediction_fixture(problem)
                predictions[split, problem] = data
                row, method_rows = metric_fixture(problem, split, data)
                rows.append(row)
                methods.extend(method_rows)
            macro = dict(rows[0], problem='ALL', n=sum(row['n'] for row in rows), sbs_name='per-problem')
            for key in reporter.common.METRIC_KEYS:
                macro[key] = float(np.mean([row[key] for row in rows]))
            metrics.extend([macro, *rows])
        for dataset, problem in reporter.LIBS.items():
            data = prediction_fixture(problem, reporter.LIB_ROWS[dataset])
            predictions[dataset, problem] = data
            metrics.append(dict(next(row for row in metrics if row['problem'] == problem), split=dataset))
        captions, save = [], PdfPages.savefig

        def capture(pages, figure, **kwargs):
            captions.extend(text.get_text() for text in figure.texts)
            return save(pages, figure, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            reporter.write_json(run / 'history.json', [dict(successful_updates=1440, optimization=dict(loss=1., ce=.5),
                val=dict(macro=dict(macro_top1=.4, macro_actual_regret_pct=2., macro_vs_sbs_pct=-1.)), train=None)])
            comparisons, alignment, nss = reporter.nss_comparisons(config, lock, predictions, run / 'missing-NSS')
            with patch.object(reporter._ENGINE, 'PROBLEMS', problems), patch.object(reporter.common,
                    'diagnostic_pdf', return_value={}), patch.object(PdfPages, 'savefig', capture), warnings.catch_warnings(record=True) as caught:
                result = reporter.build_reports(run, config, lock, {}, {}, predictions, metrics, methods, comparisons, alignment, nss)
            count = lambda path: len(re.findall(rb'/Type /Page\b', Path(path).read_bytes()))
            self.assertEqual(count(result['model_statistics_pdf']), 8)
            self.assertEqual(count(result['selection_distributions_pdf']), 3)
            text = '\n'.join(captions)
            for forbidden in ('R125', 'R126', 'MoSES_RF removed', 'MoSES_RF deleted', 'non-B5 / B3'):
                self.assertNotIn(forbidden, text)
            self.assertIn('model-only', text)
            self.assertIn('TSP8 / CVRP9 / ATSP5 / non-B5 / B2', text)
            self.assertIn('NOT a matched', text)
            self.assertIn(reporter.TOPK_NOTE, text)
            self.assertFalse(any('Glyph' in str(warning.message) and 'missing' in str(warning.message) for warning in caught))
            protocol = reporter.read_json(run / 'full_model_statistics/protocol.json')
            self.assertEqual(protocol['experiment'], 'R127')
            self.assertEqual(protocol['model_factory'], 'code.V4.r127_experiment.load_checkpoint')
            self.assertEqual(protocol['removed_candidates'], ['MVMOE'])
            self.assertEqual(protocol['retained_candidates'], ['BQ', 'LEHD', 'MoSES_RF'])
            self.assertEqual(len(protocol['changed_task_pools']), 16)
            self.assertEqual(len(protocol['two_candidate_problems']), 8)
            self.assertEqual(protocol['candidate_counts']['VRPB'], 2)
            self.assertEqual(protocol['report_template']['sha256'], reporter.TEMPLATE_SHA256)
            self.assertEqual(len(protocol['candidate_pool_statistics']), 18)


if __name__ == '__main__':
    unittest.main()
