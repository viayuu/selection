"""Synthetic CPU reporting tests; no solver, GPU, or trained model inference."""

import copy
import io
import json
import re
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from code.unified_selector.registry import PROBLEMS, S2I
from . import r125_reporting as original
from . import r126_reporting as reporter


def config_fixture():
    pools = {problem: reporter.expected_pool(problem) for problem in PROBLEMS}
    aliases = {'reld_mt': 'RELD_MOEL'}
    return dict(experiment='R126', arm=reporter.ARM, model_factory=reporter.FACTORY + '.load_checkpoint',
        seed=2, rows_per_problem=dict(reporter.SPLIT_ROWS), logical_global_batch=1024,
        max_successful_updates=57600, warmup_updates=4320, eval_every_updates=1440,
        candidate_pools=pools, candidate_pool_ids={problem: [S2I[aliases.get(name, name)] for name in pool]
            for problem, pool in pools.items()}, scenario_solver_aliases=aliases,
        data_root='/tmp/r126-frozen-no-moses-rf', source_dataset_root='/tmp/r84-frozen-full',
        checkpoint_rule='minimum full18 VAL macro_vs_sbs_pct; first wins ties',
        inference_rule='original maximin; ascending global-ID tie break',
        removed_candidates=['MoSES_RF'], retained_candidates=['BQ', 'LEHD'],
        source_scope=reporter.SOURCE_SCOPE, role_policy=dict(reporter.ROLE_POLICY),
        model_only_scope={'no_neural_model_solvers': ['GLOP']}, model_params={'solver_representation': 'code'},
        baselines={problem: dict(selection_split='train', sbs_solver=pool[0], sbs_slot=0,
            mean_cost_by_solver=dict.fromkeys(pool, 1.)) for problem, pool in pools.items()})


def lock_fixture():
    return dict(experiment='R126', arm=reporter.ARM, epoch=1, selected_update=1440,
        checkpoint_sha256='a' * 64, state_hash='b' * 64, rule='minimum VAL',
        selected_without_test=True, per_problem_assembly=False)


def identity_fixture(split, lock):
    return dict(experiment='R126', arm=reporter.ARM, split=split, successful_updates=lock['selected_update'],
        state_hash=lock['state_hash'], model_state_sha256='c' * 64,
        checkpoint_sha256=lock['checkpoint_sha256'] if split == 'test' else None)


def prediction_fixture(problem='TSP', n=4):
    helper, config = reporter.common.reporting_helpers(), config_fixture()
    pool = config['candidate_pools'][problem]
    costs = np.full((n, len(pool)), 4., np.float64)
    costs[np.arange(n), np.arange(n) % len(pool)] = 1.
    return helper.Predictions(path=Path('/tmp/synthetic-r126-' + problem + '.npz'), pool=pool,
        pred=np.zeros(n, np.int64), winner=costs.argmin(1), costs=costs, indices=np.arange(n),
        source_indices=np.arange(n), pool_ids=np.asarray(config['candidate_pool_ids'][problem]), score_ties=0)


def nss_archive(current, problem):
    pool = list(reporter.ORIGINAL_POOLS[problem])
    costs = np.full((current.n, len(pool)), .5, np.float64)
    scores = np.zeros_like(costs, dtype=np.float32)
    for slot, name in enumerate(current.pool):
        costs[:, pool.index(name)] = current.costs[:, slot]
    scores[:, pool.index('BQ')] = 20.
    scores[:, pool.index('LEHD')] = 10.
    if 'MoSES_RF' in pool:
        scores[:, pool.index('MoSES_RF')] = 100.
    return dict(pool=np.asarray(pool), costs=costs, winner=costs.argmin(1), scores=scores,
        pred=scores.argmax(1), indices=current.indices.copy(), source_indices=np.arange(current.n) + 10000)


class R126ReportingTests(unittest.TestCase):
    def test_runner_and_cli_contract_without_model_loading(self):
        from . import r126_experiment as runner
        self.assertEqual((reporter.ARMS, reporter.MAX_UPDATES, reporter.DEFAULT_ROOT),
            (runner.ARMS, runner.MAX_UPDATES, runner.DEFAULT_ROOT))
        self.assertEqual(reporter.FACTORY + '.load_checkpoint', runner.MODEL_FACTORY)
        self.assertEqual(reporter.SOURCE_SCOPE, runner.SOURCE_SCOPE)
        for command in ('lib', 'report', 'all', 'status'):
            args = reporter.parser().parse_args([command])
            self.assertEqual((args.arm, args.device, args.lib_batch_size), ('model_only_seed2', 'cpu', 2))
            self.assertNotIn('R125', reporter.parser().format_help())
        with patch.object(reporter, 'lib_stage') as lib, patch.object(reporter, 'report', return_value={}) as report:
            with redirect_stdout(io.StringIO()):
                reporter.main(['all'])
            lib.assert_called_once()
            report.assert_called_once()

    def test_private_template_does_not_modify_r125_globals_or_source(self):
        digest = reporter.file_hash(Path(original.__file__))
        reporter.validate_config(config_fixture())
        self.assertEqual(original.ARM, 'code_seed2')
        self.assertEqual(original.REMOVED, frozenset(('BQ', 'LEHD')))
        self.assertEqual(original.FACTORY, 'code.V4.r125_experiment')
        self.assertEqual(len(original.expected_pool('TSP')), 6)
        self.assertEqual(len(reporter.expected_pool('TSP')), 8)
        self.assertIsNot(original.load_saved_splits.__globals__, reporter.load_saved_splits.__globals__)
        self.assertEqual(digest, reporter.TEMPLATE_SHA256)
        self.assertEqual(digest, reporter.file_hash(Path(original.__file__)))

    def test_full18_pools_change_only_cvrp_and_seven_non_b_tasks(self):
        config = config_fixture()
        reporter.validate_config(config)
        changes = []
        for problem in PROBLEMS:
            pool = config['candidate_pools'][problem]
            expected_n = {'TSP': 8, 'CVRP': 9, 'ATSP': 5}.get(problem, 3 if 'B' in problem else 5)
            self.assertEqual(len(pool), expected_n)
            self.assertNotIn('MoSES_RF', pool)
            if problem in ('TSP', 'CVRP'):
                self.assertTrue({'BQ', 'LEHD'} <= set(pool))
            parent = [name for name in original.expected_pool(problem)]
            if problem in ('TSP', 'CVRP'):
                parent = [name for name in reporter.ORIGINAL_POOLS[problem]]
            if 'MoSES_RF' in parent:
                changes.append(problem)
        self.assertEqual(changes, ['CVRP'] + [problem for problem in PROBLEMS[3:] if 'B' not in problem])
        self.assertEqual(len(changes), 8)

    def test_rejects_four_role_or_wrong_candidate_protocols(self):
        for mutate in (
                lambda value: value['role_policy'].update(inference=True),
                lambda value: value['model_only_scope'].update(no_neural_model_solvers=[]),
                lambda value: value['candidate_pools']['TSP'].remove('BQ'),
                lambda value: value['candidate_pools']['CVRP'].append('MoSES_RF'),
                lambda value: value['baselines']['TSP'].update(selection_split='test')):
            config = config_fixture()
            mutate(config)
            with self.assertRaises(ValueError):
                reporter.validate_config(config)

    def test_selected_val_none_checkpoint_is_bound_only_in_copy(self):
        lock = lock_fixture()
        val = dict(identity=identity_fixture('val', lock))
        before = copy.deepcopy(val)
        official = reporter.bind_selected_val(val, lock, identity_fixture('test', lock))
        self.assertEqual(val, before)
        self.assertIsNone(official['archive_capture_identity']['checkpoint_sha256'])
        self.assertEqual(official['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
        self.assertEqual(official['identity']['experiment'], 'R126')
        invalid = identity_fixture('test', lock)
        invalid['model_state_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            reporter.bind_selected_val(val, lock, invalid)

    def test_native_lib_keeps_bq_lehd_and_recomputes_fp64_winner(self):
        config = config_fixture()
        for problem in ('TSP', 'CVRP'):
            pool = list(reporter.ORIGINAL_POOLS[problem])
            costs = np.full((2, len(pool)), 10., np.float64)
            costs[:, pool.index('BQ')] = 1. + 2e-8
            costs[:, pool.index('LEHD')] = 1. + 1e-8
            if 'MoSES_RF' in pool:
                costs[:, pool.index('MoSES_RF')] = .5
            raw = dict(pool=pool, costs=costs, winner=costs.argmin(1))
            reduced = reporter.reduced_raw(raw, problem, config)
            self.assertTrue({'BQ', 'LEHD'} <= set(reduced['pool']))
            self.assertNotIn('MoSES_RF', reduced['pool'])
            np.testing.assert_array_equal(reduced['winner'], [reduced['pool'].index('LEHD')] * 2)
            self.assertNotEqual(reduced['winner'][0], reduced['costs'].astype(np.float32).argmin(1)[0])
            self.assertEqual(raw['pool'], pool)

    def test_nss_accepts_unchanged_tsp8_and_cvrp10_to9(self):
        for problem in ('TSP', 'CVRP'):
            current = prediction_fixture(problem)
            archive = nss_archive(current, problem)
            current.release_source_indices = archive['source_indices'].copy()
            order = np.array([3, 0, 2, 1])
            for key in ('costs', 'scores', 'indices', 'winner', 'pred', 'source_indices'):
                archive[key] = archive[key][order]
            data = reporter.reselect_nss(archive, Path('/tmp/synthetic-nss.npz'), current)
            np.testing.assert_array_equal(data.pred, [data.pool.index('BQ')] * 4)
            np.testing.assert_array_equal(data.winner, current.winner)
            np.testing.assert_array_equal(data.costs, current.costs)
            archive['costs'][0, archive['pool'].tolist().index('BQ')] += .1
            with self.assertRaises(AssertionError):
                reporter.reselect_nss(archive, Path('/tmp/synthetic-nss.npz'), current)

    def test_nss_rejects_r125_candidate_scores_and_uses_global_id_ties(self):
        current = prediction_fixture()
        archive = nss_archive(current, 'TSP')
        archive['scores'].fill(0.)
        data = reporter.reselect_nss(archive, Path('/tmp/synthetic-nss.npz'), current)
        np.testing.assert_array_equal(data.pred, np.zeros(4, np.int64))
        keep = [i for i, name in enumerate(archive['pool']) if name not in ('BQ', 'LEHD')]
        archive['pool'], archive['costs'], archive['scores'] = (
            archive['pool'][keep], archive['costs'][:, keep], archive['scores'][:, keep])
        with self.assertRaises(ValueError):
            reporter.reselect_nss(archive, Path('/tmp/r125-not-reusable.npz'), current)

    def test_saved_nss_four_datasets_and_missing_population_are_explicit(self):
        config, lock = config_fixture(), lock_fixture()
        predictions = {(split, problem): prediction_fixture(problem, n)
            for split, problem, n in (('TEST', 'TSP', 4500), ('TEST', 'CVRP', 4500),
                                     ('TSPLIB', 'TSP', 49), ('CVRPLIB', 'CVRP', 100))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, _, diagnostics = reporter.nss_comparisons(config, lock, predictions, root)
            self.assertTrue(all(row['nss_top1_native_pct'] is None for row in rows))
            self.assertEqual(diagnostics, {})
            reporter.write_json(root / 'protocol.json', dict(data_root=config['source_dataset_root'],
                checkpoints={problem: {'sha256': 'd' * 64} for problem in ('TSP', 'CVRP')}))
            for (split, problem), data in predictions.items():
                name = 'test' if split == 'TEST' else split
                path = root / problem / (name + '_predictions.npz')
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, **nss_archive(data, problem))
                reporter.write_json(path.with_name(name + '_metrics.json'), dict(checkpoint={'sha256': 'd' * 64}))
            rows, alignment, diagnostics = reporter.nss_comparisons(config, lock, predictions, root)
            self.assertEqual(len(diagnostics), 4)
            self.assertTrue(all(row['status'] == 'fixed_NSS_same_pool_diagnostic' for row in rows))
            self.assertTrue(all(not row['nss_training_matched'] and not row['nss_new_inference'] for row in rows))
            self.assertFalse(rows[0]['nss_pool_changed'])
            self.assertTrue(rows[1]['nss_pool_changed'])
            self.assertEqual(alignment['TSP TEST']['removed_candidates'], [])
            self.assertEqual(alignment['CVRP TEST']['removed_candidates'], ['MoSES_RF'])
            np.savez_compressed(root / 'TSP/test_predictions.npz', **nss_archive(prediction_fixture(n=3895), 'TSP'))
            rows, _, _ = reporter.nss_comparisons(config, lock, predictions, root)
            self.assertIsNone(rows[0]['nss_top1_native_pct'])
            self.assertIn('3895', rows[0]['missing_reason'])

    def test_lib_loads_only_r126_factory_and_reuses_saved_predictions(self):
        import torch
        from .r77_data import parameter_hash
        config, lock, data = config_fixture(), lock_fixture(), prediction_fixture('CVRP')
        raw = dict(directory='/tmp/synthetic-native-LIB', pool=data.pool,
            pool_ids=data.pool_ids.tolist(), costs=data.costs, winner=data.winner)
        model = torch.nn.Linear(1, 1)
        test = dict(identity=dict(model_state_sha256=parameter_hash(model.state_dict())))
        state = dict(args=config, epoch=lock['epoch'], successful_updates=lock['selected_update'])
        factory = SimpleNamespace(load_checkpoint=lambda path, device: (model, state))

        def infer_stub(model, loader, batch_size, raw, path):
            scores = np.zeros(raw['costs'].shape, np.float32)
            scores[:, 0] = 1.
            np.savez_compressed(path, scores=scores, logits=scores, pred=scores.argmax(1),
                costs=raw['costs'], winner=raw['winner'], pool=np.asarray(raw['pool']),
                pool_ids=np.asarray(raw['pool_ids']), nodes=np.ones(data.n, np.int64), indices=np.arange(data.n))

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'best.pt').write_bytes(b'synthetic')
            lock['checkpoint_sha256'] = reporter.file_hash(run / 'best.pt')
            with patch.object(reporter._ENGINE, 'LIBS', {'CVRPLIB': 'CVRP'}), patch.object(
                    reporter._ENGINE, 'native_lib', return_value=raw), patch.object(
                    reporter._ENGINE, 'file_hash', return_value=lock['checkpoint_sha256']), patch.object(
                    reporter._ENGINE.importlib, 'import_module', return_value=factory) as imported, patch.object(
                    reporter._ENGINE, 'lib_loader', return_value=object()), patch(
                    'code.V4.nss_comparison.infer', side_effect=infer_stub) as inferred:
                result = reporter.evaluate_libs(run, config, lock, test)
                imported.assert_called_once_with('code.V4.r126_experiment')
                self.assertFalse(result['CVRPLIB']['inference_reused'])
                protocol = result['CVRPLIB']['protocol']
                self.assertEqual(protocol['experiment'], 'R126')
                self.assertEqual(protocol['removed_candidates'], ['MoSES_RF'])
                self.assertTrue({'BQ', 'LEHD'} <= set(protocol['candidate_pool']))
                imported.reset_mock()
                inferred.reset_mock()
                result = reporter.evaluate_libs(run, config, lock, test)
                self.assertTrue(result['CVRPLIB']['inference_reused'])
                imported.assert_not_called()
                inferred.assert_not_called()

    def test_same_axis_distributions_keep_bq_lehd_and_real_model_top1(self):
        helper = reporter.common.reporting_helpers()
        helper.setup_style()
        for problem, lib_n in (('TSP', 49), ('CVRP', 100), ('ATSP', None)):
            test = prediction_fixture(problem)
            lib = prediction_fixture(problem, lib_n) if lib_n else None
            panels = reporter.make_panels(problem, test, lib)
            fig = reporter.distribution_figure(problem, test.pool, panels, 'synthetic-new90k',
                'R126/model_only_seed2', lock_fixture())
            try:
                self.assertIn('TEST', fig.axes[0].get_title())
                self.assertIn('\u6a21\u578bTop1=', fig.axes[0].get_title())
                self.assertTrue(all(len(ax.containers) == 2 for ax in fig.axes))
                labels = [label.get_text() for label in fig.axes[0].get_yticklabels()]
                self.assertEqual(labels, test.pool)
                self.assertNotIn('MoSES_RF', labels)
                text = '\n'.join(label.get_text() for label in fig.texts)
                self.assertNotIn('R125', text)
                if problem == 'ATSP':
                    self.assertIn('ID only', text)
                else:
                    self.assertTrue({'BQ', 'LEHD'} <= set(labels))
            finally:
                import matplotlib.pyplot as plt
                plt.close(fig)

    def test_synthetic_pdf_captions_manifest_metrics_and_fonts(self):
        from matplotlib.backends.backend_pdf import PdfPages
        from .r76_experiment import fixed_sbs_metrics
        helper, config, lock = reporter.common.reporting_helpers(), config_fixture(), lock_fixture()
        problems = ['TSP', 'CVRP', 'ATSP']
        predictions, metrics, methods = {}, [], []
        for split in ('VAL', 'TEST'):
            split_rows = []
            for problem in problems:
                data = prediction_fixture(problem)
                predictions[split, problem] = data
                scores = np.zeros_like(data.costs, np.float32)
                scores[:, 0] = 1.
                saved, _ = fixed_sbs_metrics(scores,
                    dict(costs=data.costs, winner=data.winner, pool=data.pool, pool_ids=data.pool_ids.tolist()),
                    config['baselines'][problem])
                split_rows.append(reporter.common.metric_row(problem, split, data, saved,
                    config['baselines'][problem], config, lock, data.path))
                methods.extend(helper.method_rows(problem, split, saved, data, 'synthetic'))
            macro = dict(split_rows[0], problem='ALL', n=sum(row['n'] for row in split_rows), sbs_name='per-problem')
            for key in reporter.common.METRIC_KEYS:
                macro[key] = float(np.mean([row[key] for row in split_rows]))
            metrics.extend([macro, *split_rows])
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
            reporter.write_json(run / 'history.json', [dict(successful_updates=1440,
                optimization=dict(loss=1., ce=.5), val=dict(macro=dict(macro_top1=.4,
                    macro_actual_regret_pct=2., macro_vs_sbs_pct=-1.)), train=None)])
            comparisons, alignment, nss = reporter.nss_comparisons(config, lock, predictions, run / 'missing-NSS')
            with patch.object(reporter._ENGINE, 'PROBLEMS', problems), patch.object(
                    reporter.common, 'diagnostic_pdf', return_value={}), patch.object(PdfPages, 'savefig', capture), warnings.catch_warnings(record=True) as caught:
                result = reporter.build_reports(run, config, lock, {}, {}, predictions, metrics, methods,
                    comparisons, alignment, nss)
            count = lambda path: len(re.findall(rb'/Type /Page\b', Path(path).read_bytes()))
            self.assertEqual(count(result['model_statistics_pdf']), 8)
            self.assertEqual(count(result['selection_distributions_pdf']), 3)
            self.assertEqual(count(result['learning_curves_pdf']), 1)
            text = '\n'.join(captions)
            for forbidden in ('R125', 'R80D', 'TSP6', 'BQ/LEHD deleted', 'BQ/LEHD removed'):
                self.assertNotIn(forbidden, text)
            self.assertIn('model-only', text)
            self.assertIn('NOT a matched', text)
            self.assertIn('TSP8 / CVRP9', text)
            self.assertIn('encoder+decision active; inference/config absent; GLOP ID-only', text)
            self.assertFalse(any('Glyph' in str(warning.message) and 'missing' in str(warning.message) for warning in caught))
            protocol = reporter.read_json(run / 'full_model_statistics/protocol.json')
            self.assertEqual(protocol['experiment'], 'R126')
            self.assertEqual(protocol['model_factory'], 'code.V4.r126_experiment.load_checkpoint')
            self.assertEqual(protocol['removed_candidates'], ['MoSES_RF'])
            self.assertEqual(protocol['retained_candidates'], ['BQ', 'LEHD'])
            self.assertEqual(protocol['role_policy'], reporter.ROLE_POLICY)
            self.assertEqual(protocol['candidate_counts']['TSP'], 8)
            self.assertEqual(protocol['candidate_counts']['CVRP'], 9)
            self.assertEqual(len(protocol['changed_task_pools']), 8)
            self.assertEqual(protocol['report_template']['sha256'], reporter.TEMPLATE_SHA256)
            self.assertFalse(protocol['training'])
            self.assertFalse(protocol['nss_training_matched'])


if __name__ == '__main__':
    unittest.main()
