"""Focused CPU checks for expansion identities, native fields and FP64 costs."""

import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.data import UnifiedProblemDataset
from code.unified_selector.registry import P2I, constraint_bits, problem_descriptor, problem_to_pool_mask
from .r60_data import (CALIBRATED, COMPLETE, FIELDS, GENERATION_RECORD, PAIR_SOLVERS,
                       PROGRESS_RECORD, RECIPE_RECORD, paired_auxiliary_schedule, prepare_expanded)


def instance(size):
    return dict(depot_xy=[[.1, .2]], node_xy=[[i / 10., .7] for i in range(size)],
                node_demand=[-.2 if i % 2 else .3 for i in range(size)], route_limit=3.4,
                service_time=[.1] * size, tw_start=[.2] * size, tw_end=[1.8] * size)


class R60DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.cache = self.root / 'source', self.root / 'cache'
        self.source.mkdir()
        self.problem = 'OVRPBLTW'
        directory = self.source / (self.problem + 'train')
        (directory / 'results').mkdir(parents=True)
        self.instances = [instance(size) for size in (2, 3, 2, 3, 2, 3)]
        with (directory / 'dataset.pkl').open('wb') as stream:
            pickle.dump(self.instances, stream)
        self.costs = np.array([[1., 2.], [2., 1.], [1., 1. + 1e-12],
                               [1. + 1e-12, 1.], [2., 2.], [3., 2.]], dtype=np.float64)
        for col, solver in enumerate(PAIR_SOLVERS):
            with (directory / 'results' / ('result_' + solver + '.txt')).open('w', newline='') as stream:
                csv.writer(stream).writerows(enumerate(self.costs[:, col]))
        fields = list(FIELDS.values())
        for filename in (RECIPE_RECORD, PROGRESS_RECORD):
            with (self.source / filename).open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for solver in PAIR_SOLVERS:
                    writer.writerow({FIELDS['problem']: self.problem, FIELDS['solver']: solver,
                        FIELDS['added']: 4, FIELDS['calibration']: CALIBRATED, FIELDS['calibrated']: 4,
                        FIELDS['source_count']: 6, FIELDS['prefix']: 2, FIELDS['new_count']: 4,
                        FIELDS['status']: COMPLETE})
        (directory / GENERATION_RECORD).write_text(json.dumps(dict(data_repairs=[dict(path='recorded-repair')])) )

    def prepare(self):
        return prepare_expanded(self.cache, self.source, problems=(self.problem,), source_count=6, prefix_count=2)[self.problem]

    def test_fp64_direction_ties_prefix_identity_and_readonly_cache(self):
        with patch('torch.cuda._lazy_init', side_effect=AssertionError('No CUDA in data preparation')):
            dataset = self.prepare()
            self.assertEqual(len(dataset), 4)
            np.testing.assert_array_equal(dataset.source_indices, [2, 3, 4, 5])
            self.assertEqual(dataset.pair_solvers, PAIR_SOLVERS)
            self.assertTrue(all(isinstance(getattr(dataset, key), np.memmap) for key in
                                ('nodes', 'node_values', 'offsets', 'pair_costs', 'source_indices')))
            self.assertFalse(dataset.node_values.flags.writeable)
            inputs, costs = dataset.gather([0, 1, 2, 3], 'cpu')
            self.assertEqual(costs.dtype, torch.float64)
            np.testing.assert_array_equal(costs, self.costs[2:])
            self.assertEqual((costs[:, 0] < costs[:, 1]).tolist(), [True, False, False, False])
            self.assertEqual((costs[:, 0] != costs[:, 1]).tolist(), [True, True, False, True])
            self.assertEqual(costs.float()[0, 0].item(), costs.float()[0, 1].item())
            self.assertEqual(len(inputs['pool_ids']), 7)
            self.assertEqual(inputs['mask'].sum(1).tolist(), [7.] * 4)
            self.assertTrue((inputs['coord_dist'] == 4).all())
            self.assertFalse(set(inputs) & {'costs', 'ind', 'winner', 'pair_costs', 'source_indices', 'oracle'})
            self.assertEqual(dataset.manifest['recorded_input_repairs'], [dict(path='recorded-repair')])
            with patch('pickle.load', side_effect=AssertionError('Complete cache must not reread source pickle')):
                reloaded = self.prepare()
            np.testing.assert_array_equal(reloaded.nodes, dataset.nodes)

    def test_input_fields_match_current_unified_dataset_and_padding(self):
        dataset = self.prepare()
        inputs, _ = dataset.gather([1, 0, 3])
        reference = UnifiedProblemDataset.__new__(UnifiedProblemDataset)
        reference.problem, reference.pid = self.problem, P2I[self.problem]
        reference.instances, reference.base_N = self.instances, 6
        reference.coord_augment, reference.split = 0, 'train'
        reference.pool_order, reference.mask = problem_to_pool_mask(self.problem)
        reference.cbits = constraint_bits(self.problem)
        reference.problem_desc = problem_descriptor(self.problem)
        reference.K_p = len(reference.pool_order)
        reference.labels = {str(i): dict(cost=[1.] * 7, ind=0) for i in range(6)}
        for row, local in enumerate((1, 0, 3)):
            source_index = int(dataset.source_indices[local])
            original = reference[source_index]
            self.assertEqual(inputs['n'][row], original['n'])
            torch.testing.assert_close(inputs['node'][row, :original['n']], original['node'], rtol=0, atol=0)
            self.assertTrue((inputs['node'][row, original['n']:] == 0).all())
            self.assertEqual(inputs['node_mask'][row].sum(), original['n'])
        self.assertTrue((inputs['node'][..., 2] < 0).any())
        for problem in ('VRPB', 'VRPL', 'OVRPTW'):
            from .r60_data import _node_input
            reference.problem, reference.cbits = problem, constraint_bits(problem)
            expected = reference[2]['node']
            torch.testing.assert_close(torch.from_numpy(_node_input(self.instances[2], reference.cbits)), expected, rtol=0, atol=0)

    def test_true_size_paired_schedule_traversal_and_column_order_rejection(self):
        dataset = self.prepare()
        old_nodes = {self.problem: torch.tensor([3, 4])}
        schedule = paired_auxiliary_schedule({self.problem: dataset}, old_nodes, epochs=4, epoch_size=2)
        again = paired_auxiliary_schedule({self.problem: dataset}, old_nodes, epochs=4, epoch_size=2)
        for epoch in range(4):
            a, b = schedule['A'][self.problem][epoch], schedule['B'][self.problem][epoch]
            np.testing.assert_array_equal(old_nodes[self.problem][a], dataset.nodes[b])
            np.testing.assert_array_equal(b, again['B'][self.problem][epoch])
        for start in (0, 2):
            np.testing.assert_array_equal(np.sort(np.concatenate(schedule['B'][self.problem][start:start + 2])), np.arange(4))
        bad = self.source / (self.problem + 'train/results/result_RELD_MTL.txt')
        lines = bad.read_text().splitlines()
        lines[2], lines[3] = lines[3], lines[2]
        bad.write_text('\n'.join(lines) + '\n')
        with self.assertRaisesRegex(ValueError, 'cache inputs/protocol changed'):
            self.prepare()
        with self.assertRaisesRegex(ValueError, 'count/order/finite-cost'):
            prepare_expanded(self.root / 'badcache', self.source, problems=(self.problem,), source_count=6, prefix_count=2)


if __name__ == '__main__':
    unittest.main()
