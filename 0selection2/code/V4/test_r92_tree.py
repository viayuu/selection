"""R92 tree checks: synthetic CPU only; optional two TRAIN input rows/task.

Set R92_TREE_INPUT_CACHE to an existing R84 cache to verify its input schema.
No test opens original costs/winners/TEST or starts any GPU workload.
"""

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, P2I, PROBLEMS, constraint_bits, problem_descriptor
from . import r92_tree as tree
from .r77_data import parameter_hash
from .r92_tree_features import COLUMNS, FEATURE_DEFINITION, features


def fixture(problem='CVRP', rows=2, nodes=6):
    generator = torch.Generator().manual_seed(92)
    kind = 'matrix' if problem == 'ATSP' else 'coord'
    width = 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
    mask = torch.ones(rows, nodes, dtype=torch.bool)
    mask[-1, -2:] = False
    batch = dict(kind=kind, problem_id=P2I[problem], n=mask.sum(1), node_mask=mask,
        cbits=torch.tensor(constraint_bits(problem)).float()[None].repeat(rows, 1),
        problem_desc=torch.tensor(problem_descriptor(problem)).float()[None].repeat(rows, 1),
        coord_dist=torch.arange(rows), pool_ids=torch.tensor([0, 3, 7]),
        mask=torch.ones(rows, len(GLOBAL_SOLVERS)))
    if kind == 'coord':
        node = torch.rand(rows, nodes, width, generator=generator)
        if width >= 4:
            node[..., 3] = 0
            node[:, 0, 3] = 1
            node[:, 0, 2] = 0
        batch.update(node=node, d_in=width,
                     node_geom=torch.rand(rows, nodes, 12, generator=generator))
    else:
        batch['matrix'] = torch.rand(rows, nodes, nodes, generator=generator)
    return batch


def constant_bank(arm='C', iteration=100):
    pools = {p: ['a', 'b', 'c'] for p in PROBLEMS}
    ids = {p: [0, 3, 7] for p in PROBLEMS}
    config = dict(experiment='R92', arm=arm, seed=2, solver_names=GLOBAL_SOLVERS,
        candidate_pools=pools, candidate_pool_ids=ids, max_successful_updates=3000,
        rows_per_problem=dict(train=81000, val=4500, test=4500), feature_definition=FEATURE_DEFINITION)
    entries = {p: dict(kind='constant', constant_class=1, arm=arm, pool=pools[p], pool_ids=ids[p],
                       actual_tree_count=0) for p in PROBLEMS}
    return tree.TreeBank(Path('/synthetic/unused'), config, entries, iteration), config, entries


class TreeFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_every_problem_same_finite_feature_schema(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                x, names = features(fixture(problem))
                self.assertEqual(x.shape, (2, len(COLUMNS)))
                self.assertEqual(x.dtype, np.float32)
                self.assertEqual(names, COLUMNS)
                self.assertTrue(np.isfinite(x).all())
        self.assertEqual(len(COLUMNS), len(set(COLUMNS)))

    def test_cost_label_index_route_embedding_isolation(self):
        batch = fixture()
        first, _ = features(batch)
        poisoned = dict(batch)
        for key in ('costs', 'winner', 'ind', 'index', 'source_indices', 'performance_target',
                    'cost_regret', 'cmp_target', 'routes', 'neural_embeddings', 'pool_ids', 'mask'):
            poisoned[key] = object()
        poisoned['problem_id'] = object()
        actual, _ = features(poisoned)
        np.testing.assert_array_equal(first, actual)

    def test_padding_nan_inf_never_changes_coordinate_statistics(self):
        batch = fixture('VRPTW')
        expected, _ = features(batch)
        changed = copy.deepcopy(batch)
        changed['node'][~changed['node_mask']] = float('nan')
        changed['node_geom'][~changed['node_mask']] = float('inf')
        actual, _ = features(changed)
        np.testing.assert_array_equal(expected, actual)
        one = {k: v[-1:] if torch.is_tensor(v) and k != 'pool_ids' else v for k, v in changed.items()}
        count = int(one['n'][0])
        for key in ('node', 'node_geom', 'node_mask'):
            one[key] = one[key][:, :count]
        trimmed, _ = features(one)
        np.testing.assert_allclose(actual[-1:], trimmed, rtol=2e-6, atol=2e-7)

    def test_valid_moments_and_quantiles_only(self):
        batch = fixture('TSP', rows=1)
        batch['node'][0, :4, 0] = torch.tensor([1., 2., 3., 4.])
        batch['node'][0, 4:, 0] = 1000
        x, names = features(batch)
        expected = dict(mean=2.5, std=np.sqrt(1.25), min=1., max=4., q25=1.75, q50=2.5, q75=3.25)
        for stat, value in expected.items():
            self.assertAlmostEqual(float(x[0, names.index('raw_x_'+stat)]), value, places=6)

    def test_atsp_excludes_diagonal_and_padding_in_all_edge_statistics(self):
        batch = fixture('ATSP', rows=1, nodes=5)
        batch['matrix'][0, :3, :3] = torch.tensor([[float('nan'), 2., 4.],
            [6., -1e30, 8.], [10., 12., float('inf')]])
        batch['matrix'][0, 3:] = float('nan')
        batch['matrix'][0, :, 3:] = float('inf')
        x, names = features(batch)
        for field in ('out_mean', 'in_mean', 'out_nearest4', 'out_nearest8'):
            self.assertAlmostEqual(float(x[0, names.index('matrix_'+field+'_mean')]), 7., places=6)
        self.assertAlmostEqual(float(x[0, names.index('matrix_asym_abs_mean_mean')]), 14/3, places=5)
        self.assertAlmostEqual(float(x[0, names.index('matrix_out_min_mean')]), 6., places=6)

    def test_nondepot_tsp_has_no_first_node_anchor(self):
        batch = fixture('TSP', rows=1, nodes=7)
        changed = copy.deepcopy(batch)
        permutation = torch.tensor([2, 0, 4, 1, 3, 5, 6])
        for key in ('node', 'node_geom', 'node_mask'):
            changed[key] = changed[key][:, permutation]
        expected, _ = features(batch)
        actual, _ = features(changed)
        np.testing.assert_allclose(expected, actual, rtol=2e-6, atol=2e-7)

    def test_rejects_nonfinite_real_input_and_wrong_lengths(self):
        batch = fixture()
        batch['node'][0, 1, 0] = float('nan')
        with self.assertRaises(ValueError):
            features(batch)
        batch = fixture()
        batch['n'][0] -= 1
        with self.assertRaises(ValueError):
            features(batch)

    @unittest.skipUnless(os.environ.get('R92_TREE_INPUT_CACHE'), 'optional existing-cache input-only smoke')
    def test_real_cache_first_two_train_input_rows(self):
        root = Path(os.environ['R92_TREE_INPUT_CACHE'])
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                directory = root/'train'/problem
                record = json.loads((directory/'metadata.json').read_text())
                self.assertEqual(record['rows'], 81000)
                arrays = {}
                try:
                    keys = ('n', 'coord_dist', 'node', 'node_geom') if record['kind'] == 'coord' else ('n', 'coord_dist', 'matrix')
                    for key in keys:
                        spec = record['arrays'][key]
                        arrays[key] = np.load(directory/spec['file'], mmap_mode='r', allow_pickle=False)
                    n = torch.from_numpy(arrays['n'][:2].copy())
                    maximum = int(n.max())
                    batch = dict(problem_id=record['problem_id'], kind=record['kind'], n=n,
                        node_mask=torch.arange(maximum)[None] < n[:, None],
                        cbits=torch.tensor(record['cbits']).float()[None].repeat(2, 1),
                        problem_desc=torch.tensor(record['problem_desc']).float()[None].repeat(2, 1),
                        coord_dist=torch.from_numpy(arrays['coord_dist'][:2].copy()))
                    for key in keys[2:]:
                        values = arrays[key][:2, :maximum]
                        if key == 'matrix':
                            values = values[:, :, :maximum]
                        batch[key] = torch.from_numpy(values.copy())
                    x, names = features(batch)
                    self.assertEqual(x.shape, (2, len(names)))
                    self.assertTrue(np.isfinite(x).all())
                    print(f'R92 input-only smoke {problem}: n={n.tolist()} cbits={tuple(batch["cbits"].shape)} desc={tuple(batch["problem_desc"].shape)} features={x.shape}', flush=True)
                finally:
                    for array in arrays.values():
                        array._mmap.close()


class TreeProtocolTests(unittest.TestCase):
    def test_targets_preserve_fp64_before_fp32_conversion_and_all_methods(self):
        costs = np.array([[1.00000000001, 1., 4.], [9., 3., 6.]], dtype=np.float64)
        target = tree.relative_regret_targets(costs)
        self.assertEqual(target.shape, costs.shape)
        self.assertEqual(target.dtype, np.float32)
        self.assertGreater(target[0, 0], 0.)
        np.testing.assert_allclose(target[1], [2., 0., 1.])
        with self.assertRaises(ValueError):
            tree.relative_regret_targets(costs.astype(np.float32))
        for bad in (0., float('nan'), float('inf')):
            changed = costs.copy()
            changed[0, 0] = bad
            with self.assertRaises(ValueError):
                tree.relative_regret_targets(changed)

    def test_one_global_tree_count_not_per_problem_and_first_exact_tie(self):
        grid = range(100, 3001, 100)
        stages = {p: {t: dict(vs_sbs_pct=10.) for t in grid} for p in PROBLEMS}
        for i, problem in enumerate(PROBLEMS):
            stages[problem][100]['vs_sbs_pct'] = 0. if i == 0 else 2.
            stages[problem][200]['vs_sbs_pct'] = 1.
            stages[problem][300]['vs_sbs_pct'] = 1.
        selected, means = tree.choose_global_iteration(stages)
        self.assertEqual(selected, 200)
        self.assertEqual(means[200], means[300])
        del stages[PROBLEMS[-1]]
        with self.assertRaises(ValueError):
            tree.choose_global_iteration(stages)

    def test_unseen_classes_retained_and_regret_sign(self):
        entry = dict(kind='catboost', arm='C', pool=['a', 'b', 'c', 'd'])
        scores = tree.prediction_scores(np.array([[0., 2., -1.]]), entry, classes=[0, 1, 3])
        self.assertEqual(scores.shape, (1, 4))
        self.assertEqual(int(scores.argmax(1)[0]), 1)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(np.exp(scores).sum(1), 1., atol=1e-6)
        entry.update(arm='D')
        scores = tree.prediction_scores([[.2, -.1, 1., .3]], entry)
        self.assertEqual(int(scores.argmax(1)[0]), 1)

    def test_bank_local_forward_global_predict_and_label_isolation(self):
        for arm in tree.ARMS:
            bank, _, _ = constant_bank(arm)
            batch = fixture()
            batch.update(costs=object(), winner=object(), ind=object(), index=object())
            before = parameter_hash(bank.state_dict())
            out = bank(batch)
            self.assertEqual(out['logits'].shape, (2, 3))
            self.assertEqual(out['pair_margin'].shape, (2, 3, 3))
            expanded = bank.predict_batch(batch)
            self.assertEqual(expanded['logits'].shape, (2, len(GLOBAL_SOLVERS)))
            torch.testing.assert_close(expanded['logits'][:, batch['pool_ids']], out['logits'])
            self.assertTrue(torch.equal(expanded['solver_mask'].sum(1), torch.tensor([3, 3])))
            self.assertEqual(before, parameter_hash(bank.state_dict()))
            self.assertEqual(bank.selected_iteration, 100)

    def test_bank_fingerprints_include_iteration_and_every_task(self):
        first, config, entries = constant_bank()
        second = tree.TreeBank('/synthetic/unused', config, entries, 200)
        self.assertNotEqual(parameter_hash(first.state_dict()), parameter_hash(second.state_dict()))
        changed = copy.deepcopy(entries)
        changed[PROBLEMS[-1]]['constant_class'] = 2
        third = tree.TreeBank('/synthetic/unused', config, changed, 100)
        self.assertNotEqual(parameter_hash(first.state_dict()), parameter_hash(third.state_dict()))

    def test_checkpoint_roundtrip_and_unauthorized_test_reads_nothing(self):
        bank, config, entries = constant_bank()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            state = dict(args=config, model=bank.state_dict(), tree_bank=dict(entries=entries, ntree_end=100),
                optimizer=None, state_hash=tree.json_hash(config), successful_updates=100, epoch=1)
            tree.save_checkpoint(run/'best.pt', state)
            restored, saved = tree.load_checkpoint(run/'best.pt')
            self.assertEqual(saved['successful_updates'], 100)
            self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(bank.state_dict()))
            with patch.object(tree, 'open_cache', side_effect=AssertionError('Do not open original data')):
                with self.assertRaises(PermissionError):
                    tree._validate_test_lock(run)

    def test_fixed_parameters_no_early_stop_or_class_rebalancing(self):
        self.assertEqual(tree.TREE_PARAMETERS['iterations'], 3000)
        self.assertEqual(tree.TREE_PARAMETERS['boosting_type'], 'Plain')
        self.assertEqual(tree.TREE_PARAMETERS['task_type'], 'GPU')
        self.assertEqual(tree.TREE_PARAMETERS['border_count'], 128)
        self.assertNotIn('class_weights', tree.TREE_PARAMETERS)
        self.assertNotIn('early_stopping_rounds', tree.TREE_PARAMETERS)

    def test_wandb_initializes_and_finishes_on_worker_failure_without_gpu_access(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(root=Path(directory), arm='C')
            tracker = Mock()
            with patch.object(tree, '_config', return_value={}), \
                    patch.object(tree, 'wandb_run', return_value=tracker) as initialize, \
                    patch.object(tree, '_train_bank', side_effect=RuntimeError('synthetic worker failure')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic worker failure'):
                    tree.train(args, 'cuda:0')
                initialize.assert_called_once_with(args, {})
                tracker.finish.assert_called_once()


class CatBoostCPUAPITests(unittest.TestCase):
    """Two synthetic CPU trees only, explicitly authorized API probes."""

    def test_classifier_and_full_multitarget_regressor_predict_apis(self):
        catboost = tree._catboost()
        x = np.random.default_rng(92).normal(size=(24, len(COLUMNS))).astype(np.float32)
        classes = np.array([0, 1, 3]*8, dtype=np.int64)
        classifier = catboost.CatBoostClassifier(iterations=2, depth=2, loss_function='MultiClass',
            classes_count=4, task_type='CPU', thread_count=2, allow_writing_files=False, random_seed=2)
        classifier.fit(x, classes, verbose=False)
        raw = classifier.predict(x, prediction_type='RawFormulaVal', ntree_end=2, thread_count=2, task_type='CPU')
        entry = dict(kind='catboost', arm='C', pool=['a', 'b', 'c', 'd'])
        scores = tree.prediction_scores(raw, entry, classifier.classes_)
        self.assertEqual(scores.shape, (24, 4))
        prefixes = list(tree.validation_predictions(classifier, x, entry, iterations=2, period=1))
        self.assertEqual(len(prefixes), 2)
        np.testing.assert_array_equal(prefixes[-1], scores)
        target = np.random.default_rng(93).uniform(size=(24, 4)).astype(np.float32)
        regressor = catboost.CatBoostRegressor(iterations=2, depth=2, loss_function='MultiRMSE',
            task_type='CPU', thread_count=2, allow_writing_files=False, random_seed=2)
        regressor.fit(x, target, verbose=False)
        raw = regressor.predict(x, prediction_type='RawFormulaVal', ntree_end=2, thread_count=2, task_type='CPU')
        entry.update(arm='D')
        self.assertEqual(tree.prediction_scores(raw, entry).shape, (24, 4))
        prefixes = list(tree.validation_predictions(regressor, x, entry, iterations=2, period=1))
        self.assertEqual(len(prefixes), 2)
        np.testing.assert_array_equal(prefixes[-1], tree.prediction_scores(raw, entry))


if __name__ == '__main__':
    unittest.main()
