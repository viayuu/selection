"""Focused synthetic CPU checks; never open the released TEST split."""

import json
import os
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, POOLS, PROBLEMS, S2I
from . import r76_cache as cache_module
from .local_geometry import FIELDS, local_geometry
from .new_split_data import build_batch, load_dataset
from .pairwise_objective import comparison_targets, inputs_only
from .r76_cache import ScenarioCache, finalize_cache, prepare_worker


def _json(path, value):
    path.write_text(json.dumps(value) + '\n')


def _fixture(data_root, aliases=True):
    data_root.mkdir(parents=True)
    alias = {'reld_mt': 'RELD_MOEL'} if aliases else {}
    vocabulary = [('reld_mt' if name == 'RELD_MOEL' else name)
                  for name in GLOBAL_SOLVERS if name != 'RELD_MTL'] if aliases else list(GLOBAL_SOLVERS)
    identities = {name: S2I[alias.get(name, name)] for name in vocabulary}
    manifest = dict(status='complete', format_version=1, scenario_id='tiny_r76_' + str(aliases),
                    global_solvers=vocabulary, model_global_solvers=list(GLOBAL_SOLVERS),
                    global_solver_ids=identities, solver_aliases=alias, problems={})
    for problem in PROBLEMS:
        if problem in ('TSP', 'CVRP', 'ATSP'):
            pool = list(POOLS[problem])
        else:
            reld = 'reld_mt' if aliases else 'RELD_MOEL'
            pool = ['MTPOMO', 'MVMOE', reld] if 'B' in problem else [
                'MTPOMO', 'MVMOE', 'MoSES_CaDA', 'MoSES_RF', reld, 'RouteFinder']
        pool_ids = [identities[name] for name in pool]
        manifest['problems'][problem] = dict(methods=pool, global_solver_ids=pool_ids)
        for split in ('train', 'val'):
            directory = data_root / (problem + split)
            (directory / 'results').mkdir(parents=True)
            _json(directory / 'metadata.json', dict(methods=pool, global_solver_ids=pool_ids, instances=2))
            instances = []
            for n in (3, 5):
                xy = np.asarray([[i / n, (i * i % n) / n] for i in range(n)], dtype=np.float32)
                if split == 'val':
                    xy[:, 1] = np.linspace(0., 1., n)
                if problem == 'TSP':
                    instance = xy[None]
                elif problem == 'ATSP':
                    instance = (np.arange(n * n, dtype=np.float32).reshape(1, n, n) + 1) / n
                elif problem == 'CVRP':
                    instance = dict(loc=xy[None, 1:], depot=xy[None, :1],
                                    demand=np.ones((1, n - 1), dtype=np.float32) / 10)
                else:
                    instance = dict(depot_xy=xy[:1], node_xy=xy[1:],
                                    node_demand=np.ones(n - 1) / 10, route_limit=3.,
                                    service_time=np.ones(n - 1) / 20,
                                    tw_start=np.zeros(n - 1), tw_end=np.ones(n - 1))
                instances.append(instance)
            costs = np.asarray([[1.00000002, 1.00000001, 1.00000001] +
                                [2. + i for i in range(len(pool) - 3)],
                                [5., 2., 2.] + [3.] * (len(pool) - 3)], dtype=np.float64)
            labels = {str(i): dict(cost=row.tolist(), ind=int(row.argmin()))
                      for i, row in enumerate(costs)}
            with (directory / 'dataset.pkl').open('wb') as stream:
                pickle.dump(instances, stream)
            with (directory / 'raw_label.pkl').open('wb') as stream:
                pickle.dump(labels, stream)
            np.savez(directory / 'input_metadata.npz', source_index=np.asarray([40, 11]),
                     coord_dist=np.asarray([3, 1] if problem not in ('TSP', 'CVRP', 'ATSP') else [4, 4]))
            for name in pool:
                (directory / 'results' / ('result_' + name + '.txt')).write_text('synthetic\n')
    _json(data_root / 'scenario_manifest.json', manifest)


class R76CacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root, self.data_root = self.base / 'run', self.base / 'data'
        self.cache_root = self.root / 'cache'
        _fixture(self.data_root)

    def tearDown(self):
        self.temporary.cleanup()

    def worker(self, rank=0, workers=1, splits=('train', 'val')):
        return prepare_worker(self.root, self.data_root, self.cache_root,
                              worker_rank=rank, workers=workers, device='cpu', splits=splits)

    def test_four_workers_finalize_train_only_and_reuse(self):
        for rank in range(4):
            manifest = self.worker(rank, 4)
            assigned = set(PROBLEMS[rank::4])
            self.assertTrue(manifest['complete'])
            for split in ('train', 'val'):
                self.assertEqual(set(manifest['splits'][split]), assigned)
            self.assertTrue((self.root / f'cache_manifest_worker_{rank}.json').is_file())
            progress = json.loads((self.root / f'cache_progress_worker_{rank}.json').read_text())
            self.assertEqual(progress['worker_rank'], rank)
            self.assertEqual(progress['phase'], 'complete')
        self.assertFalse((self.root / 'cache_manifest.json').exists())
        self.assertFalse((self.root / 'geometry_stats.json').exists())
        manifest = finalize_cache(self.root, self.data_root, self.cache_root)
        self.assertTrue(manifest['complete'])
        self.assertEqual(manifest['rows_by_split'], {'train': 36, 'val': 36})
        stats = manifest['geometry_stats']
        self.assertEqual(stats['valid_train_nodes'], 17 * 8)
        self.assertEqual(len(stats['valid_nodes_by_problem']), 17)
        expected_sum = np.zeros(len(FIELDS))
        for problem, record in manifest['splits']['train'].items():
            if record['kind'] == 'coord':
                expected_sum += np.asarray(record['geometry_moments']['sum'])
        np.testing.assert_allclose(stats['sum'], expected_sum, rtol=0, atol=0)
        for record in manifest['splits']['val'].values():
            self.assertEqual(record['geometry_moments']['count'], 0)
            self.assertEqual(record['geometry_moments']['sum'], [0.] * len(FIELDS))
        with mock.patch.object(cache_module, 'load_dataset', side_effect=AssertionError('reuse decoded source')):
            self.worker(0, 4)
        self.assertEqual(json.loads((self.root / 'geometry_stats.json').read_text()), stats)

    def test_batch_precision_alias_masks_geometry_and_no_hotpath_calculation(self):
        self.worker(4, 18)  # VRPB: filtered three-candidate pool.
        directory = self.cache_root / 'train' / 'VRPB'
        dataset = load_dataset('VRPB', data_root=self.data_root, root=self.root)
        expected = build_batch(dataset, [1, 0, 1], include_geometry=False)
        with mock.patch.object(cache_module, 'comparison_targets', wraps=comparison_targets) as targets:
            cache = ScenarioCache(directory)
            try:
                self.assertEqual(targets.call_count, 1)
                self.assertEqual(cache.pool, ['MTPOMO', 'MVMOE', 'reld_mt'])
                self.assertEqual(cache.pool_ids, [10, 11, 16])
                self.assertIsInstance(cache.lengths, np.memmap)
                self.assertEqual(cache.raw['costs'].dtype, np.float64)
                self.assertEqual(cache.raw['winner'].dtype, np.int64)
                self.assertFalse(cache.raw['costs'].flags.writeable)
                with mock.patch.object(cache_module, 'local_geometry', side_effect=AssertionError('hotpath geometry')):
                    batch = cache.batch(['1', '0', '1'])
                    small = cache.batch([0])
                self.assertEqual(targets.call_count, 1)
                for key, value in expected.items():
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(batch[key], value), key)
                    else:
                        self.assertEqual(batch[key], value)
                self.assertEqual(small['node'].shape, (1, 3, 8))
                self.assertEqual(batch['coord_dist'].tolist(), [1, 3, 1])
                self.assertEqual(batch['mask'][:, 17].tolist(), [0., 0., 0.])
                self.assertTrue(batch['solver_mask'].all())
                self.assertEqual(int(small['ind'][0]), 1)
                self.assertEqual(int(small['costs'][0].argmin()), 0)
                direct = comparison_targets(dataset.raw_costs[[1, 0, 1]])
                for key, value in direct.items():
                    self.assertTrue(torch.equal(batch[key], value), key)
                geometry = local_geometry(expected['node'][..., :2].double(), expected['node_mask']).float()
                self.assertTrue(torch.equal(batch['node_geom'], geometry))
                clean = inputs_only(batch)
                self.assertFalse(set(('costs', 'ind', *direct)) & set(clean))
                self.assertTrue(all(value.is_contiguous() for value in batch.values() if torch.is_tensor(value)))
                for indices in ([], [-1], [2], ['01']):
                    with self.assertRaises((ValueError, IndexError, KeyError)):
                        cache.batch(indices)
            finally:
                cache.close()
            cache.close()
            with self.assertRaises(RuntimeError):
                cache.batch([0])

    def test_matrix_and_six_candidate_pool_relocate_without_sources(self):
        for problem in ('ATSP', 'OVRP'):
            self.worker(PROBLEMS.index(problem), 18, splits=('val',))
            directory = self.cache_root / 'val' / problem
            local = self.base / 'local' / problem
            shutil.copytree(directory, local)
            cache = ScenarioCache(local)
            try:
                batch = cache.batch([0])
                key = 'matrix' if problem == 'ATSP' else 'node'
                self.assertEqual(batch[key].shape[1], 3)
                self.assertEqual(batch['cmp_target'].shape[1], len(cache.pool) * (len(cache.pool) - 1) // 2)
                if problem == 'OVRP':
                    self.assertEqual(len(cache.pool), 6)
                    self.assertIn(16, cache.pool_ids)
                    self.assertNotIn(17, cache.pool_ids)
                else:
                    self.assertEqual(batch['matrix'].shape, (1, 3, 3))
                    self.assertNotIn('node_geom', batch)
            finally:
                cache.close()

    def test_reject_stale_source_and_old_seven_candidate_metadata(self):
        self.worker(4, 18, splits=('train',))
        path = self.cache_root / 'train' / 'VRPB' / 'metadata.json'
        original = path.read_text()
        source = self.data_root / 'VRPBtrain' / 'input_metadata.npz'
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        with self.assertRaisesRegex(ValueError, 'incompatible scenario/source'):
            self.worker(4, 18, splits=('train',))
        self.assertEqual(path.read_text(), original)
        record = json.loads(original)
        record.pop('cache_format')
        record['pool'] = list(POOLS['VRPB'])
        record['pool_ids'] = [S2I[name] for name in record['pool']]
        _json(path, record)
        with self.assertRaisesRegex(ValueError, 'incompatible R76 cache'):
            ScenarioCache(path.parent)

    def test_partial_resume_does_not_double_count_uncheckpointed_geometry(self):
        original = cache_module.build_batch
        calls = 0

        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('synthetic interruption')
            return original(*args, **kwargs)

        with mock.patch.object(cache_module, 'COLLATE_CHUNK', 1), \
                mock.patch.object(cache_module, 'build_batch', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                self.worker(0, 18, splits=('train',))
        path = self.cache_root / 'train' / 'TSP' / 'metadata.json'
        partial = json.loads(path.read_text())
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['rows_written'], 0)
        self.assertEqual(partial['geometry_moments']['count'], 0)
        with mock.patch.object(cache_module, 'COLLATE_CHUNK', 1), \
                mock.patch.object(cache_module, '_checkpoint', wraps=cache_module._checkpoint) as checkpoint:
            self.worker(0, 18, splits=('train',))
            self.assertEqual(checkpoint.call_count, 1)  # Final flush, not every collate chunk.
        record = json.loads(path.read_text())
        self.assertEqual(record['geometry_moments']['count'], 8)
        cache = ScenarioCache(path.parent)
        try:
            self.assertEqual(cache.batch([1, 0])['n'].tolist(), [5, 3])
        finally:
            cache.close()

    def test_test_gate_precedes_any_source_or_array_access(self):
        missing = self.base / 'missing_data'
        for function in (prepare_worker, finalize_cache):
            with self.assertRaises(PermissionError):
                function(self.root, missing, self.cache_root, splits=('train', 'test'))
        directory = self.base / 'unlocked_test_cache'
        directory.mkdir()
        _json(directory / 'metadata.json', dict(split='test', test_lock_root=str(self.root)))
        with self.assertRaises(PermissionError):
            ScenarioCache(directory)
        self.assertFalse(self.root.exists())

    def test_original_non_alias_scenario_retains_global_ids(self):
        original_data = self.base / 'original_data'
        _fixture(original_data, aliases=False)
        prepare_worker(self.root, original_data, self.cache_root,
                       worker_rank=4, workers=18, device='cpu', splits=('train',))
        cache = ScenarioCache(self.cache_root / 'train' / 'VRPB')
        try:
            self.assertEqual(cache.pool, ['MTPOMO', 'MVMOE', 'RELD_MOEL'])
            self.assertEqual(cache.pool_ids, [10, 11, 16])
            self.assertEqual(cache.metadata['solver_aliases'], {})
        finally:
            cache.close()


if __name__ == '__main__':
    unittest.main()
