"""Focused CPU/synthetic checks for R125 reporting; never run a solver/model."""

import copy
import io
import json
import pickle
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from code.unified_selector.registry import PROBLEMS, S2I
from . import r125_reporting as reporter


def config_fixture():
    pools = {problem: reporter.expected_pool(problem) for problem in PROBLEMS}
    aliases = {'reld_mt': 'RELD_MOEL'}
    return dict(experiment='R125', arm=reporter.ARM, model_factory=reporter.FACTORY + '.load_checkpoint',
        seed=2, rows_per_problem=dict(reporter.SPLIT_ROWS), logical_global_batch=1024,
        max_successful_updates=57600, warmup_updates=4320, eval_every_updates=1440,
        candidate_pools=pools, candidate_pool_ids={problem: [S2I[aliases.get(name, name)] for name in pool]
            for problem, pool in pools.items()}, scenario_solver_aliases=aliases,
        data_root='/tmp/r125-frozen-reduced', source_dataset_root='/tmp/r84-frozen-full',
        checkpoint_rule='minimum full18 VAL macro_vs_sbs_pct; first wins ties',
        inference_rule='original maximin; ascending global-ID tie break',
        baselines={problem: dict(selection_split='train', sbs_solver=pool[0], sbs_slot=0,
            mean_cost_by_solver=dict.fromkeys(pool, 1.)) for problem, pool in pools.items()})


def lock_fixture():
    return dict(experiment='R125', arm=reporter.ARM, epoch=1, selected_update=1440,
        checkpoint_sha256='a' * 64, state_hash='b' * 64, rule='minimum VAL',
        selected_without_test=True, per_problem_assembly=False)


def identity_fixture(split, lock):
    return dict(experiment='R125', arm=reporter.ARM, split=split, successful_updates=lock['selected_update'],
        state_hash=lock['state_hash'], model_state_sha256='c' * 64,
        checkpoint_sha256=lock['checkpoint_sha256'] if split == 'test' else None)


def prediction_fixture(problem='TSP', n=4):
    helper = reporter.common.reporting_helpers()
    config = config_fixture()
    pool = config['candidate_pools'][problem]
    costs = np.full((n, len(pool)), 4., np.float64)
    costs[np.arange(n), np.arange(n) % len(pool)] = 1.
    return helper.Predictions(path=Path('/tmp/synthetic-r125-' + problem + '.npz'), pool=pool,
        pred=np.zeros(n, np.int64), winner=costs.argmin(1), costs=costs, indices=np.arange(n),
        source_indices=np.arange(n), pool_ids=np.asarray(config['candidate_pool_ids'][problem]), score_ties=0)


def nss_archive(current):
    problem = 'TSP' if 'DIFUSCO' in current.pool else 'CVRP'
    pool = list(reporter.ORIGINAL_POOLS[problem])
    costs = np.full((current.n, len(pool)), .5, np.float64)
    scores = np.zeros_like(costs, dtype=np.float32)
    for slot, name in enumerate(current.pool):
        costs[:, pool.index(name)] = current.costs[:, slot]
        scores[:, pool.index(name)] = 2. if slot == 1 else 0.
    scores[:, pool.index('BQ')] = 100.
    scores[:, pool.index('LEHD')] = 99.
    return dict(pool=np.asarray(pool), costs=costs, winner=costs.argmin(1), scores=scores,
        pred=np.full(current.n, pool.index('BQ'), np.int64), indices=current.indices.copy(),
        source_indices=np.arange(current.n) + 10000)


class R125ReportingTests(unittest.TestCase):
    def test_runner_and_cli_contract(self):
        from . import r125_experiment as runner
        self.assertEqual(reporter.ARMS, runner.ARMS)
        self.assertEqual(reporter.MAX_UPDATES, runner.MAX_UPDATES)
        self.assertEqual(reporter.DEFAULT_ROOT, runner.DEFAULT_ROOT)
        self.assertEqual(reporter.FACTORY + '.load_checkpoint', runner.MODEL_FACTORY)
        for command in ('lib', 'report', 'all', 'status'):
            args = reporter.parser().parse_args([command])
            self.assertEqual((args.arm, args.device, args.lib_batch_size), ('code_seed2', 'cpu', 2))
        with patch.object(reporter, 'lib_stage') as lib, patch.object(reporter, 'report', return_value={}) as report:
            with redirect_stdout(io.StringIO()):
                reporter.main(['all'])
            lib.assert_called_once()
            report.assert_called_once()

    def test_full18_pool_and_train_sbs_contract(self):
        config = config_fixture()
        reporter.validate_config(config)
        self.assertEqual(len(config['candidate_pools']['TSP']), 6)
        self.assertEqual(len(config['candidate_pools']['CVRP']), 8)
        self.assertEqual(sum('B' in problem for problem in PROBLEMS), 8)
        for problem in PROBLEMS:
            self.assertFalse(set(config['candidate_pools'][problem]) & reporter.REMOVED)
            self.assertNotIn('RELD_MTL', config['candidate_pools'][problem])
        for mutate in (
                lambda value: value['candidate_pools']['TSP'].append('BQ'),
                lambda value: value['candidate_pools']['OVRP'].remove('reld_mt'),
                lambda value: value['baselines']['TSP'].update(selection_split='test'),
                lambda value: value.update(logical_global_batch=512)):
            invalid = copy.deepcopy(config)
            mutate(invalid)
            with self.assertRaises(ValueError):
                reporter.validate_config(invalid)

    def test_val_none_checkpoint_binding_is_copy_only(self):
        lock = lock_fixture()
        capture = dict(identity=identity_fixture('val', lock), per_problem={'TSP': {'n': 4500}})
        original = copy.deepcopy(capture)
        official = reporter.bind_selected_val(capture, lock, identity_fixture('test', lock))
        self.assertEqual(capture, original)
        self.assertIsNone(official['archive_capture_identity']['checkpoint_sha256'])
        self.assertEqual(official['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
        self.assertTrue(official['checkpoint_binding']['archive_identity_unchanged'])
        for key in ('model_state_sha256', 'state_hash', 'successful_updates', 'checkpoint_sha256'):
            invalid = copy.deepcopy(capture)
            invalid['identity'][key] = 'wrong'
            with self.assertRaises(ValueError):
                reporter.bind_selected_val(invalid, lock, identity_fixture('test', lock))

    def test_completed_runner_lock_accepts_original_val_capture(self):
        config, lock = config_fixture(), lock_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / reporter.ARM
            run.mkdir()
            checkpoint = run / 'best.pt'
            checkpoint.write_bytes(b'synthetic checkpoint; no torch deserialization')
            lock.update(checkpoint=str(checkpoint), checkpoint_sha256=reporter.file_hash(checkpoint),
                state_hash=reporter.common.json_hash(config), rule=config['checkpoint_rule'],
                inference_rule=config['inference_rule'])
            val = dict(identity=identity_fixture('val', lock), per_problem=dict.fromkeys(PROBLEMS, {}))
            test = dict(identity=identity_fixture('test', lock), per_problem=dict.fromkeys(PROBLEMS, {}))
            records = {'protocol.json': config, 'final_test_lock.json': lock,
                'training_complete.json': dict(successful_updates=57600, selected_best=lock),
                'test_results.json': dict(lock, complete=True, result=test),
                'final_metrics.json': dict(lock, val=val)}
            for name, record in records.items():
                reporter.write_json(run / name, record)
            _, _, _, _, official = reporter.locked_inputs(root)
            self.assertEqual(official['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
            self.assertIsNone(reporter.read_json(run / 'final_metrics.json')['val']['identity']['checkpoint_sha256'])

    def test_native_lib_removal_recomputes_fp64_not_fp32_winner(self):
        config = config_fixture()
        pool = list(reporter.ORIGINAL_POOLS['TSP'])
        costs = np.full((2, len(pool)), 10., np.float64)
        costs[:, pool.index('BQ')] = .5
        costs[:, pool.index('LEHD')] = .6
        costs[0, pool.index('DIFUSCO')] = 1. + 2e-8
        costs[0, pool.index('DIFUSCO500')] = 1. + 1e-8
        raw = dict(pool=pool, costs=costs, winner=costs.argmin(1))
        reduced = reporter.reduced_raw(raw, 'TSP', config)
        self.assertEqual(reduced['winner'][0], reduced['pool'].index('DIFUSCO500'))
        self.assertNotEqual(reduced['winner'][0], reduced['costs'].astype(np.float32).argmin(1)[0])
        self.assertIn('BQ', raw['pool'])
        np.testing.assert_array_equal(raw['costs'], costs)

    def test_native_lib_reads_existing_labels_not_solver_results(self):
        pool = list(reporter.ORIGINAL_POOLS['TSP'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'TSPLIB').mkdir()
            labels = {str(i): dict(cost=np.arange(len(pool), dtype=np.float64) + 1., ind=0) for i in range(49)}
            with (root / 'TSPLIB/raw_label.pkl').open('wb') as stream:
                pickle.dump(labels, stream)
            with patch.object(reporter, 'DATA_ROOT', root):
                raw = reporter.native_lib('TSP', 'TSPLIB', config_fixture())
            self.assertEqual(raw['costs'].shape, (49, 6))
            self.assertNotIn('BQ', raw['pool'])

    def test_lib_loader_replaces_costs_winner_ids_and_mask_without_global_mutation(self):
        import torch
        from code.unified_selector import registry
        config = config_fixture()
        original = copy.deepcopy(registry.POOLS)
        pool = list(reporter.ORIGINAL_POOLS['TSP'])
        costs = np.arange(1, 2 * len(pool) + 1, dtype=np.float64).reshape(2, -1)
        raw = reporter.reduced_raw(dict(pool=pool, costs=costs, winner=costs.argmin(1)), 'TSP', config)

        class DummyDataset:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return dict(problem_id=0, kind='coord', node=torch.zeros(3, 2),
                    cbits=torch.zeros(5), problem_desc=torch.zeros(17), coord_dist=4,
                    pool_global_ids=torch.tensor(self.pool_order), mask=torch.tensor(self.mask),
                    costs=torch.tensor(costs[index], dtype=torch.float32), ind=0, n=3)

        with patch('code.unified_selector.data.UnifiedProblemDataset', DummyDataset), patch(
                'code.V4.r41_frozen_evaluation.attach_geometry'):
            loader = reporter.lib_loader('TSP', 'TSPLIB', raw, 1, torch.device('cpu'))
        np.testing.assert_array_equal(loader.batch['costs'], raw['costs'].astype(np.float32))
        np.testing.assert_array_equal(loader.batch['ind'], raw['winner'])
        np.testing.assert_array_equal(loader.batch['pool_ids'], config['candidate_pool_ids']['TSP'])
        self.assertEqual(int(loader.batch['mask'][:, [S2I['BQ'], S2I['LEHD']]].sum()), 0)
        self.assertEqual(registry.POOLS, original)

    def test_lib_uses_only_r125_factory_and_reuses_checked_predictions(self):
        import torch
        from .r77_data import parameter_hash
        config, lock = config_fixture(), lock_fixture()
        data = prediction_fixture()
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
            with patch.object(reporter, 'LIBS', {'TSPLIB': 'TSP'}), patch.object(
                    reporter, 'native_lib', return_value=raw), patch.object(
                    reporter, 'file_hash', return_value=lock['checkpoint_sha256']), patch.object(
                    reporter.importlib, 'import_module', return_value=factory) as imported, patch.object(
                    reporter, 'lib_loader', return_value=object()), patch(
                    'code.V4.nss_comparison.infer', side_effect=infer_stub) as inferred:
                result = reporter.evaluate_libs(run, config, lock, test)
                self.assertFalse(result['TSPLIB']['inference_reused'])
                imported.assert_called_once_with('code.V4.r125_experiment')
                inferred.assert_called_once()
                imported.reset_mock()
                inferred.reset_mock()
                result = reporter.evaluate_libs(run, config, lock, test)
                self.assertTrue(result['TSPLIB']['inference_reused'])
                imported.assert_not_called()
                inferred.assert_not_called()
                self.assertNotIn('BQ', result['TSPLIB']['protocol']['candidate_pool'])

    def test_prediction_rejects_4500_population_or_fp32_label_mismatch(self):
        data, config = prediction_fixture(), config_fixture()
        reporter.validate_prediction(data, 'TSP', config, expected_rows=4)
        with self.assertRaises(ValueError):
            reporter.validate_prediction(data, 'TSP', config, expected_rows=4500)
        data.winner = np.full(4, len(data.pool) - 1, np.int64)
        with self.assertRaises(AssertionError):
            reporter.validate_prediction(data, 'TSP', config)

    def test_nss_reselects_saved_logits_and_realigns_all_rows(self):
        current = prediction_fixture()
        archive = nss_archive(current)
        current.release_source_indices = archive['source_indices'].copy()
        order = np.array([3, 1, 0, 2])
        for key in ('costs', 'scores', 'indices', 'winner', 'pred', 'source_indices'):
            archive[key] = archive[key][order]
        data = reporter.reselect_nss(archive, Path('/tmp/saved-nss.npz'), current)
        np.testing.assert_array_equal(data.pred, np.ones(4, np.int64))
        np.testing.assert_array_equal(data.winner, current.winner)
        self.assertEqual(data.pool, current.pool)
        np.testing.assert_array_equal(data.costs, current.costs)
        archive['costs'][0, archive['pool'].tolist().index(current.pool[0])] += .01
        with self.assertRaises(AssertionError):
            reporter.reselect_nss(archive, Path('/tmp/saved-nss.npz'), current)

    def test_nss_exact_score_tie_uses_global_id(self):
        current = prediction_fixture()
        archive = nss_archive(current)
        archive['scores'].fill(0.)
        data = reporter.reselect_nss(archive, Path('/tmp/saved-nss.npz'), current)
        np.testing.assert_array_equal(data.pred, np.zeros(4, np.int64))
        self.assertEqual(data.score_ties, 4)

    def test_missing_nss_is_four_explicit_missing_rows(self):
        predictions = {(split, problem): prediction_fixture(problem, n)
            for split, problem, n in (('TEST', 'TSP', 4500), ('TEST', 'CVRP', 4500),
                                     ('TSPLIB', 'TSP', 49), ('CVRPLIB', 'CVRP', 100))}
        with tempfile.TemporaryDirectory() as directory:
            rows, checks, diagnostics = reporter.nss_comparisons(config_fixture(), lock_fixture(),
                predictions, Path(directory))
        self.assertEqual(len(rows), 4)
        self.assertEqual(checks, {})
        self.assertEqual(diagnostics, {})
        self.assertTrue(all(row['nss_top1_native_pct'] is None and row['status'] == 'missing_saved_NSS' for row in rows))
        self.assertTrue(all(not row['nss_retrained'] and not row['nss_new_inference'] for row in rows))

    def test_existing_4500_nss_is_diagnostic_not_reduced_pool_training(self):
        config = config_fixture()
        predictions = {(split, problem): prediction_fixture(problem, n)
            for split, problem, n in (('TEST', 'TSP', 4500), ('TEST', 'CVRP', 4500),
                                     ('TSPLIB', 'TSP', 49), ('CVRPLIB', 'CVRP', 100))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter.write_json(root / 'protocol.json', dict(data_root=config['source_dataset_root'],
                checkpoints={'TSP': {'sha256': 'd' * 64}}))
            (root / 'TSP').mkdir()
            path = root / 'TSP/test_predictions.npz'
            np.savez_compressed(path, **nss_archive(predictions['TEST', 'TSP']))
            reporter.write_json(root / 'TSP/test_metrics.json', dict(checkpoint={'sha256': 'd' * 64}))
            rows, checks, diagnostics = reporter.nss_comparisons(config, lock_fixture(), predictions, root)
            self.assertEqual(rows[0]['status'], 'fixed_NSS_reduced_pool_diagnostic')
            self.assertEqual(checks['TSP TEST']['checked_rows'], 4500)
            self.assertFalse(rows[0]['nss_reduced_pool_training'])
            self.assertIn(('TEST', 'TSP'), diagnostics)
            old = nss_archive(prediction_fixture(n=3895))
            np.savez_compressed(path, **old)
            rows, _, _ = reporter.nss_comparisons(config, lock_fixture(), predictions, root)
            self.assertIsNone(rows[0]['nss_top1_native_pct'])
            self.assertIn('3895', rows[0]['missing_reason'])

    def test_distribution_has_same_axis_pairs_test_first_and_top1(self):
        helper = reporter.common.reporting_helpers()
        helper.setup_style()
        test, lib = prediction_fixture(), prediction_fixture(n=49)
        panels = reporter.make_panels('TSP', test, lib)
        self.assertEqual([panel['split'] for panel in panels], ['TEST', 'TEST', 'TSPLIB', 'TSPLIB'])
        fig = reporter.distribution_figure('TSP', test.pool, panels, 'synthetic frozen90k',
            'R125/code_seed2', lock_fixture())
        try:
            self.assertEqual(len(fig.axes), 2)
            for ax in fig.axes:
                self.assertEqual(len(ax.containers), 2)
                self.assertIn('\u6a21\u578bTop1=', ax.get_title())
                self.assertEqual(len(ax.get_yticklabels()), 6)
                self.assertNotIn('BQ', [text.get_text() for text in ax.get_yticklabels()])
            self.assertIn('TEST', fig.axes[0].get_title())
            self.assertIn('TSPLIB', fig.axes[1].get_title())
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_missing_nss_test_panel_is_not_filled_with_current_model_predictions(self):
        panels = reporter.make_panels('TSP', None, prediction_fixture(n=49))
        self.assertEqual(panels[0]['split'], 'TEST')
        self.assertIsNone(panels[0]['counts'])
        self.assertIsNone(panels[0]['top1'])
        self.assertIsNotNone(panels[2]['counts'])

    def test_metric_ratio_of_means_is_not_actual_regret(self):
        from .r76_experiment import fixed_sbs_metrics
        data, config = prediction_fixture(n=2), config_fixture()
        data.costs[:, :2] = [[2., 1.], [100., 200.]]
        data.winner = data.costs.argmin(1)
        # Keep other candidates expensive to separate ratio-of-means regret.
        data.costs[:, 2:] = 1000.
        data.winner = data.costs.argmin(1)
        scores = np.zeros_like(data.costs, np.float32)
        scores[:, 0] = 1.
        raw = dict(pool=data.pool, pool_ids=data.pool_ids.tolist(), costs=data.costs, winner=data.winner)
        saved, _ = fixed_sbs_metrics(scores, raw, config['baselines']['TSP'])
        row = reporter.common.metric_row('TSP', 'TEST', data, saved, config['baselines']['TSP'],
            config, lock_fixture(), data.path)
        self.assertAlmostEqual(row['actual_regret_pct'], 50.)
        self.assertAlmostEqual(row['vs_oracle_pct'], (102. / 101. - 1.) * 100.)
        self.assertNotAlmostEqual(row['actual_regret_pct'], row['vs_oracle_pct'])

    def test_synthetic_pdf_contains_metrics_learning_and_distribution_pages(self):
        from .r76_experiment import fixed_sbs_metrics
        helper = reporter.common.reporting_helpers()
        config, lock = config_fixture(), lock_fixture()
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
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            reporter.write_json(run / 'history.json', [dict(successful_updates=1440,
                optimization=dict(loss=1., ce=.5), val=dict(macro=dict(macro_top1=.4,
                    macro_actual_regret_pct=2., macro_vs_sbs_pct=-1.)), train=None)])
            comparisons = [dict(dataset=name, n=data.n, nss_top1_native_pct=None, nss_mean_cost=None,
                current_top1_native_pct=data.top1_pct, current_mean_cost=data.mean_cost)
                for name, data in (('TSP TEST', predictions['TEST', 'TSP']), ('CVRP TEST', predictions['TEST', 'CVRP']),
                                   ('TSPLIB', predictions['TSPLIB', 'TSP']), ('CVRPLIB', predictions['CVRPLIB', 'CVRP']))]
            with patch.object(reporter, 'PROBLEMS', problems), patch.object(reporter.common, 'diagnostic_pdf', return_value={}):
                result = reporter.build_reports(run, config, lock, {}, {}, predictions, metrics, methods, comparisons, {})
            count = lambda path: len(re.findall(rb'/Type /Page\b', Path(path).read_bytes()))
            self.assertEqual(count(result['model_statistics_pdf']), 8)
            self.assertEqual(count(result['selection_distributions_pdf']), 3)
            self.assertEqual(count(result['learning_curves_pdf']), 1)
            protocol = reporter.read_json(run / 'full_model_statistics/protocol.json')
            self.assertFalse(protocol['training'])
            self.assertFalse(protocol['inference'])
            self.assertEqual(protocol['candidate_pools']['TSP'], config['candidate_pools']['TSP'])


if __name__ == '__main__':
    unittest.main()
