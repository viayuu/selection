"""Few focused CPU checks for saved-route conversion, row identities and masks."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .r54_routes import route_successors
from .r60_data import GENERATION_RECORD, PAIR_SOLVERS, _node_input
from .r61_data import behavior_schedule, prepare_behavior
from code.unified_selector.registry import POOLS, constraint_bits


class R61DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.problem = 'OVRPBLTW'
        self.kwargs = dict(expanded_root=self.root / 'expanded', original_root=self.root / 'original',
            route_root=self.root / 'r54', source_root=self.root / 'source', producer_root=self.root / 'producer',
            problems=(self.problem,), original_count=2, expanded_count=4)
        self.instances = [dict(depot_xy=[[.1, .2]], node_xy=[[.3 + i * .1, .4] for i in range(n)],
            node_demand=[.2, -.1, .1][:n], route_limit=3., service_time=[.1] * n,
            tw_start=[.2] * n, tw_end=[1.9] * n) for n in (2, 3, 2, 3, 2, 3)]
        original = self.kwargs['original_root'] / (self.problem + 'train')
        original.mkdir(parents=True)
        with (original / 'dataset.pkl').open('wb') as stream:
            pickle.dump(self.instances[:2], stream)
        self.kwargs['route_root'].mkdir()
        targets = np.full((2, 2, 3), -100, dtype=np.int16)
        for row, n in enumerate((2, 3)):
            for role in range(2):
                targets[row, role, :n] = route_successors(self.route(n, role), n)
        np.savez_compressed(self.kwargs['route_root'] / (self.problem + '_train.npz'), successor=targets,
            customers=np.array([2, 3]), selected_indices=np.arange(2), complete=np.ones((2, 2), bool),
            pool=np.array(PAIR_SOLVERS), data_hash=np.asarray('recorded-original-input-hash'))
        features = self.kwargs['expanded_root'] / self.problem
        features.mkdir(parents=True)
        nodes = np.array([3, 4, 3, 4], dtype=np.int32)
        np.save(features / 'nodes.npy', nodes)
        np.save(features / 'offsets.npy', np.concatenate(([0], np.cumsum(nodes))))
        np.save(features / 'source_indices.npy', np.arange(2, 6))
        np.save(features / 'node_values.npy', np.concatenate([_node_input(x, constraint_bits(self.problem))
                for x in self.instances[2:]]))
        (features / 'manifest.json').write_text(json.dumps(dict(problem=self.problem, count=4, prefix_count=2,
                pair_solvers=list(PAIR_SOLVERS), pool=list(POOLS[self.problem]), recorded_input_repairs=[])))
        generated = self.kwargs['source_root'] / (self.problem + 'train')
        generated.mkdir(parents=True)
        shards = [dict(id='s' + str(i), start_index=2 + i * 2, count=2, sha256='input-' + str(i),
                       path='recorded-input-shard-' + str(i)) for i in range(2)]
        (generated / GENERATION_RECORD).write_text(json.dumps(dict(shards=shards)))
        self.first_meta = None
        for role, solver in enumerate(PAIR_SOLVERS):
            folder = self.kwargs['producer_root'] / 'labels' / self.problem / solver / 'shards'
            folder.mkdir(parents=True)
            for shard in shards:
                path = folder / (shard['id'] + '.json')
                body = dict(problem=self.problem, method=solver, start_index=shard['start_index'],
                    count=2, input_sha256=shard['sha256'], shard_id=shard['id'], profile_sha256='profile',
                    costs=[4., 5.], tours=[self.route(len(self.instances[i]['node_xy']), role)
                        for i in range(shard['start_index'], shard['start_index'] + 2)])
                path.write_text(json.dumps(body))
                meta = {key: body[key] for key in ('problem', 'method', 'start_index', 'count', 'input_sha256', 'profile_sha256')}
                meta.update(result_path=str(path), result_sha256='recorded-result-hash')
                meta_path = folder / (shard['id'] + '.meta.json')
                meta_path.write_text(json.dumps(meta))
                if self.first_meta is None:
                    self.first_meta = meta_path

    @staticmethod
    def route(n, role):
        nodes = list(range(1, n + 1))
        return [0] + (nodes if role == 0 else nodes[::-1]) + [0, 0]

    def test_exact_r54_conversion_including_open_end_and_trailing_depot(self):
        np.testing.assert_array_equal(route_successors([0, 1, 2, 0, 3], 3), [2, 0, 0])
        np.testing.assert_array_equal(route_successors([0, 1, 2, 0, 3, 0, 0], 3), [2, 0, 0])
        with self.assertRaisesRegex(ValueError, 'every customer exactly once'):
            route_successors([0, 1, 2, 2], 3)

    def test_real_source_identity_role_order_padding_and_clean_features(self):
        with patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU only')):
            data = prepare_behavior(self.root / 'cache', **self.kwargs)
            old, new = data['old'][self.problem], data['new'][self.problem]
            np.testing.assert_array_equal(old.source_indices, [0, 1])
            np.testing.assert_array_equal(new.source_indices, [2, 3, 4, 5])
            for dataset in (old, new):
                self.assertIsInstance(dataset.successor, np.memmap)
                self.assertFalse(dataset.successor.flags.writeable)
                inputs, targets = dataset.gather([1, 0])
                self.assertEqual(targets.shape, (2, 2, 3))
                self.assertEqual(targets.dtype, torch.int64)
                self.assertTrue((targets[1, :, 2] == -100).all())
                self.assertEqual(inputs['node_mask'][:, 1:].sum(1).tolist(), [3, 2])
                self.assertEqual(len(inputs['pool_ids']), 7)
                self.assertFalse(set(inputs) & {'costs', 'ind', 'winner', 'successor', 'routes', 'source_indices'})
                for role in range(2):
                    np.testing.assert_array_equal(targets[0, role], route_successors(self.route(3, role), 3))
            with patch('pickle.load', side_effect=AssertionError('Complete feature cache must be reused')):
                prepare_behavior(self.root / 'cache', **self.kwargs)

    def test_two_complete_size_matched_passes_and_metadata_mismatch_rejected(self):
        data = prepare_behavior(self.root / 'cache', **self.kwargs)
        plans = behavior_schedule(data)
        for i in range(2):
            a, b = plans['A'][self.problem][i], plans['B'][self.problem][i]
            np.testing.assert_array_equal(np.sort(b), np.arange(4))
            np.testing.assert_array_equal(data['old'][self.problem].nodes[a], data['new'][self.problem].nodes[b])
        meta = json.loads(self.first_meta.read_text())
        meta['start_index'] += 1
        self.first_meta.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'metadata/input identity mismatch'):
            prepare_behavior(self.root / 'mismatch', **self.kwargs)


if __name__ == '__main__':
    unittest.main()
