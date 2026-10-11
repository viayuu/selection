"""Three focused CPU checks for native fields, frozen partial labels and replay."""

import csv
import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import POOLS, constraint_bits
from .r60_data import FIELDS, GENERATION_RECORD, PROGRESS_RECORD, RECIPE_RECORD, _node_input
from .r62_data import _native_input, portfolio_schedule, prepare_portfolio


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class R62DataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.producer, self.cache = (self.root / n for n in ('source', 'producer', 'cache'))
        self.problem = 'TSP'
        source = self.source / 'TSPtrain'
        (source / 'results').mkdir(parents=True)
        self.instances = [np.arange(n * 2, dtype=np.float32).reshape(1, n, 2) / 10
                          for n in (3, 4, 3, 4, 3, 4)]
        with (source / 'dataset.pkl').open('wb') as stream:
            pickle.dump(self.instances, stream)
        shards = []
        for number, start in enumerate((2, 4)):
            path = self.root / ('input' + str(number) + '.pkl')
            with path.open('wb') as stream:
                pickle.dump(self.instances[start:start + 2], stream)
            shards.append(dict(id='shard' + str(number), path=str(path), start_index=start,
                               count=2, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        write_json(source / GENERATION_RECORD, dict(problem='TSP', target_instances=6,
            original_instances=2, methods=POOLS['TSP'], shards=shards))
        for filename in (RECIPE_RECORD, PROGRESS_RECORD):
            with (self.source / filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[FIELDS['problem'], FIELDS['solver']])
                writer.writeheader()
                for solver in POOLS['TSP']:
                    writer.writerow({FIELDS['problem']: 'TSP', FIELDS['solver']: solver})
        for column, solver in enumerate(POOLS['TSP'][:2]):
            deployment = self.producer / 'labels/TSP' / solver
            profile = dict(problem='TSP', method=solver, budget='auxiliary-not-legacy')
            digest = hashlib.sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            write_json(deployment / 'profile.json', profile)
            if column == 0:
                table = source / 'results' / ('result_' + solver + '.txt')
                table.write_text(''.join(str(i) + ',1.0\n' for i in range(6)))
                write_json(deployment / 'complete.json', dict(problem='TSP', method=solver, instances=6,
                    profile_sha256=digest, result_sha256=hashlib.sha256(table.read_bytes()).hexdigest()))
            else:
                self.partial = deployment / 'shards/part.json'
                identity = dict(problem='TSP', method=solver, start_index=2, count=2,
                                input_sha256=shards[0]['sha256'], profile_sha256=digest)
                write_json(self.partial, dict(identity, costs=[1. + 1e-12, 1.], audit=dict(feasible=True)))
                write_json(self.partial.with_name('part.meta.json'), dict(identity, result_path=str(self.partial)))

    def prepare(self):
        return prepare_portfolio(self.cache, self.source, producer_root=self.producer,
            problems=('TSP',), source_count=6, prefix_count=2, subset_count=2)

    def test_native_fields_without_solver_or_label_inputs(self):
        np.testing.assert_array_equal(_native_input('TSP', self.instances[2]), self.instances[2][0])
        cvrp = dict(loc=np.array([[[.2, .3], [.5, .6]]]), demand=np.array([[.1, .2]]),
                    depot=np.array([[[.8, .9]]]))
        expected = np.array([[.8, .9, 0., 1.], [.2, .3, .1, 0.], [.5, .6, .2, 0.]], dtype=np.float32)
        np.testing.assert_array_equal(_native_input('CVRP', cvrp), expected)
        matrix = np.array([[[0., .2], [.7, 0.]]], dtype=np.float64)
        np.testing.assert_array_equal(_native_input('ATSP', matrix), matrix[0].astype(np.float32))
        mvrp = dict(depot_xy=[[.1, .2]], node_xy=[[.3, .4]], node_demand=[-.2],
                    service_time=[.1], tw_start=[.4], tw_end=[1.8], route_limit=3.)
        np.testing.assert_array_equal(_native_input('OVRPBLTW', mvrp), _node_input(mvrp, constraint_bits('OVRPBLTW')))

    def test_frozen_fp64_partial_identity_padding_and_full_pool(self):
        with patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU data only')):
            data = self.prepare()
            full = data['B']['TSP']
            inputs, costs, observed = full.gather([0, 1, 2, 3])
            self.assertEqual(costs.dtype, torch.float64)
            self.assertEqual(observed.dtype, torch.bool)
            np.testing.assert_array_equal(full.source_indices, [2, 3, 4, 5])
            self.assertEqual(costs.shape, (4, 8))
            self.assertEqual(len(inputs['pool_ids']), 8)
            self.assertTrue((inputs['mask'].sum(1) == 8).all())
            self.assertEqual(observed[:, 1].tolist(), [True, True, False, False])
            self.assertTrue(torch.isnan(costs[~observed]).all())
            self.assertLess(costs[0, 0], costs[0, 1])
            self.assertEqual(costs[0, 0].float(), costs[0, 1].float())
            self.assertEqual(costs[1, 0], costs[1, 1])
            self.assertFalse(set(inputs) & {'costs', 'winner', 'ind', 'source_indices', 'observed', 'rank', 'oracle'})
            for row, source_id in enumerate(full.source_indices):
                n = len(self.instances[source_id][0])
                np.testing.assert_array_equal(inputs['node'][row, :n], self.instances[source_id][0])
                self.assertEqual(int(inputs['node_mask'][row].sum()), n)
                self.assertTrue((inputs['node'][row, n:] == 0).all())
            self.partial.write_text('{}')
            reloaded = self.prepare()['B']['TSP']
            np.testing.assert_array_equal(reloaded.observed, full.observed)
            self.assertEqual(reloaded.manifest['full_coverage']['comparison_bearing_rows'], 1)

    def test_nested_actual_size_schedule_two_complete_passes(self):
        data = self.prepare()
        subset, full = data['A']['TSP'], data['B']['TSP']
        self.assertEqual(len(subset), 2)
        self.assertTrue(set(subset.source_indices) <= set(full.source_indices))
        self.assertEqual(set(subset.nodes), set(full.nodes))
        schedule = portfolio_schedule(data, passes=2, seed=2)
        again = portfolio_schedule(data, passes=2, seed=2)
        for number in range(2):
            a, b = schedule['A']['TSP'][number], schedule['B']['TSP'][number]
            np.testing.assert_array_equal(np.sort(b), np.arange(4))
            np.testing.assert_array_equal(subset.nodes[a], full.nodes[b])
            np.testing.assert_array_equal(a, again['A']['TSP'][number])
        changed_root = self.root / 'changed'
        meta = self.partial.with_name('part.meta.json')
        payload = json.loads(meta.read_text())
        payload['start_index'] = 3
        write_json(meta, payload)
        changed = prepare_portfolio(changed_root, self.source, producer_root=self.producer,
            problems=('TSP',), source_count=6, prefix_count=2, subset_count=2)['B']['TSP']
        self.assertFalse(changed.observed[:, 1].any())
        self.assertEqual(changed.manifest['full_coverage']['observed_cells'], 4)


if __name__ == '__main__':
    unittest.main()
