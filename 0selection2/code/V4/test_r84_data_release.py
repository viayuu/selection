import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .export_new_split_data import derive_targets
from .export_new_split_data import train_baselines
from .r84_data_release import export_split, extend_splits, release_pool, source_inputs


class ReleaseTest(unittest.TestCase):
    def test_extensions_never_move_previous_train(self):
        inputs = dict(scale=np.repeat([50, 100, 200], 10),
                      distribution=np.array(['uniform'] * 30),
                      capacity_type=np.array(['none'] * 30))
        old = dict(train=np.array([2, 4, 6, 8, 12]), val=np.array([3, 13]), test=np.array([5, 15]))
        sizes = dict(train=20, val=4, test=4)
        splits, extra, unused = extend_splits(inputs, old, 2, np.random.default_rng(2), sizes)
        for key in old:
            np.testing.assert_array_equal(splits[key][:len(old[key])], old[key])
        self.assertEqual(len(unused), 19)
        self.assertFalse(np.intersect1d(splits['val'], old['train']).size)
        self.assertFalse(np.intersect1d(splits['test'], old['train']).size)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(2, 30))
        again, _, _ = extend_splits(inputs, old, 2, np.random.default_rng(2), sizes)
        for key in splits:
            np.testing.assert_array_equal(splits[key], again[key])
        self.assertEqual(len(extra['val']), 2)

    def test_rejects_old_split_overlap(self):
        inputs = dict(scale=np.ones(10), distribution=np.array(['uniform'] * 10), capacity_type=np.array(['none'] * 10))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            extend_splits(inputs, dict(train=np.array([2]), val=np.array([2]), test=np.array([3])),
                          1, np.random.default_rng(2), dict(train=5, val=2, test=2))

    def test_candidate_rules_and_fp64_recomputed_winner(self):
        source = ['MTPOMO', 'MVMOE', 'MoSES_CaDA', 'MoSES_RF', 'RELD_MOEL', 'RELD_MTL', 'RouteFinder']
        names, kept, ids = release_pool('VRPB', source)
        self.assertEqual(names, ['MTPOMO', 'MVMOE', 'reld_mt'])
        self.assertEqual(kept, [0, 1, 4])
        self.assertEqual(ids, [10, 11, 16])
        costs = np.array([[3., 2., 1., 1., 2.00000000001, .5, 1.]])[:, kept]
        targets = derive_targets(costs)
        self.assertEqual(int(targets['winner'][0]), 1)
        self.assertEqual(targets['costs'].dtype, np.float64)
        np.testing.assert_array_equal(targets['ranking'][0], [1, 2, 0])
        np.testing.assert_array_equal(targets['rank'][0], [2, 0, 1])

    def test_exclusion_comes_from_authoritative_metadata(self):
        generation = dict(status='complete', target_instances=100000, original_instances=10000,
                          generated_instances=100000, shards=[dict(start_index=10000, count=90000,
                          scale=50, distribution='uniform', capacity_type='none')])
        inputs, prefix = source_inputs(generation)
        self.assertEqual(prefix, 10000)
        self.assertEqual(int(inputs['scale'][prefix]), 50)
        generation['original_instances'] = 9999
        with self.assertRaises(ValueError):
            source_inputs(generation)

    def test_export_keeps_moel_costs_and_full_source_column_provenance(self):
        source = ['MTPOMO', 'MVMOE', 'MoSES_CaDA', 'MoSES_RF', 'RELD_MOEL', 'RELD_MTL', 'RouteFinder']
        methods, kept, ids = release_pool('VRPB', source)
        costs = np.array([[4., 3., 1., 1., 2., .5, 1.], [5., 2., 1., 1., 3., .4, 1.]])
        inputs = dict(scale=np.array([50, 51]), coord_dist=np.array([0, 0]),
                      distribution=np.array(['uniform', 'uniform']),
                      capacity_type=np.array(['mvrp_default', 'mvrp_default']))
        baseline = train_baselines(costs[:, kept], methods)
        self.assertEqual(baseline['sbs_solver'], 'MVMOE')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'VRPBtrain'
            export_split(path, [None, None], costs, np.zeros_like(costs), costs.argmin(1),
                         inputs, np.array([1, 0], dtype=np.int32), methods, ids, kept,
                         baseline, 'VRPB', 'train', source)
            metadata = json.loads((path / 'metadata.json').read_text())
            self.assertEqual(metadata['source_methods'], source)
            self.assertEqual(metadata['retained_source_slots'], kept)
            self.assertEqual(metadata['global_solver_ids'], [10, 11, 16])
            with np.load(path / 'targets.npz') as targets:
                np.testing.assert_array_equal(targets['costs'], costs[[1, 0]][:, kept])
                np.testing.assert_array_equal(targets['winner'], [1, 2])
                np.testing.assert_array_equal(targets['sbs_train_cost'], [2., 3.])
            with (path / 'raw_label.pkl').open('rb') as stream:
                raw = pickle.load(stream)
            self.assertEqual(raw['0']['cost'], [5., 2., 3.])
            self.assertEqual(raw['1']['ind'], 2)
            actual = np.loadtxt(path / 'results/result_reld_mt.txt', delimiter=',')
            np.testing.assert_array_equal(actual[:, 1], [3., 2.])


if __name__ == '__main__':
    unittest.main()
