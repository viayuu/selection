"""Focused checks of R57's independent semantics and label-free controls."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .r48_common import write_csv
from .r57_audit import runtime_evidence
from .r57_probe import permutation, permute_instance, physical_starts, verify_locked_inputs
from .r57_routes import check_route
from .r57_signal import deranged_mapping, fit_prior, prior_scores, true_size


def instance(demand=(.4, .3)):
    n = len(demand)
    return dict(depot_xy=[[0., 0.]], node_xy=[[.1 * (i + 1), 0.] for i in range(n)],
                node_demand=list(demand), capacity=1.)


class IndependentRoutes(unittest.TestCase):
    def test_closed_single_customer_counts_return(self):
        x = instance((.4,))
        self.assertAlmostEqual(check_route(x, [0, 1, 0], 'VRPB')['cost'], .2, places=6)
        self.assertAlmostEqual(check_route(x, [0, 1, 0], 'OVRPB')['cost'], .1, places=6)

    def test_open_excludes_every_return(self):
        x = instance()
        self.assertAlmostEqual(check_route(x, [0, 1, 0, 2, 0], 'OVRP')['cost'], .3, places=6)
        self.assertAlmostEqual(check_route(x, [0, 1, 0, 2, 0], 'VRPL')['cost'], .6, places=6)

    def test_classical_and_signed_load_are_not_interchangeable(self):
        result = check_route(instance((.6, -.3, .6)), [0, 1, 2, 3, 0], 'VRPB')
        self.assertTrue(result['signed_load_ok'])
        self.assertFalse(result['backhaul_precedence_ok'])
        self.assertFalse(result['delivery_capacity_ok'])
        self.assertFalse(result['explicit_classical_feasible'])

    def test_valid_classical_backhaul(self):
        result = check_route(instance((.6, -.3)), [0, 1, 2, 0], 'VRPB')
        self.assertTrue(result['explicit_classical_feasible'])

    def test_separate_pickup_capacity(self):
        result = check_route(instance((.1, -.7, -.7)), [0, 1, 2, 3, 0], 'VRPB')
        self.assertFalse(result['pickup_capacity_ok'])

    def test_waiting_and_service_propagate(self):
        x = instance()
        x.update(service_time=[.2, 0.], tw_start=[1., 0.], tw_end=[2., 1.2])
        self.assertFalse(check_route(x, [0, 1, 2, 0], 'OVRPTW')['customer_tw_ok'])

    def test_open_closed_length(self):
        x = instance((.4,))
        x['route_limit'] = .15
        self.assertTrue(check_route(x, [0, 1, 0], 'OVRPL')['route_limit_ok'])
        self.assertFalse(check_route(x, [0, 1, 0], 'VRPL')['route_limit_ok'])

    def test_missing_depot_deadline_is_not_invented(self):
        x = instance((.4,))
        x.update(service_time=[.2], tw_start=[4.], tw_end=[5.])
        result = check_route(x, [0, 1, 0], 'VRPTW')
        self.assertTrue(result['explicit_classical_feasible'])
        self.assertFalse(result['closed_depot_horizon3_ok'])
        self.assertFalse(result['depot_deadline_in_raw'])

    def test_duplicate_or_missing_customer_rejected(self):
        self.assertFalse(check_route(instance(), [0, 1, 1, 0], 'OVRP')['visit_once'])


class SignalControls(unittest.TestCase):
    def test_locked_data_and_label_hashes(self):
        source = dict(data_hash='data', label_hash='labels')
        verify_locked_inputs(source, dict(source))
        for field in source:
            with self.assertRaisesRegex(ValueError, 'Locked runtime sample'):
                verify_locked_inputs(source, dict(source, **{field: 'changed'}))

    def test_actual_failed_preflight_is_retained(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(runtime_evidence(root, 'GLOP')['preflight_status'], 'not_attempted')
            write_csv(root / 'runtime_preflight.csv', [dict(solver='GLOP',
                status='blocked_runtime_dependency_or_recipe', log='preflight.log')])
            result = runtime_evidence(root, 'GLOP')
            self.assertEqual(result['preflight_status'], 'blocked_runtime_dependency_or_recipe')
            self.assertEqual(result['runtime_log'], 'preflight.log')

    def test_exact_size_derangement(self):
        sizes = np.array([50, 50, 50, 80, 80, 99])
        rows, eligible = deranged_mapping(sizes, np.random.default_rng(2))
        np.testing.assert_array_equal(sizes[rows], sizes)
        np.testing.assert_array_equal(np.sort(rows), np.arange(len(rows)))
        self.assertTrue(np.all(rows[eligible] != np.flatnonzero(eligible)))
        self.assertEqual(rows[-1], 5)
        self.assertFalse(eligible[-1])

    def test_repeatability(self):
        a = deranged_mapping(np.array([2, 2, 2, 2]), np.random.default_rng(2))[0]
        b = deranged_mapping(np.array([2, 2, 2, 2]), np.random.default_rng(2))[0]
        np.testing.assert_array_equal(a, b)

    def test_native_majority_and_mean_gap_are_different_policies(self):
        costs = np.array([[1., 1.01], [1., 1.01], [2., 1.]])
        model = fit_prior(costs, np.array([0, 0, 1]), np.array([50] * 3), [()] * 3)
        majority, cost, _ = prior_scores(model, np.array([50]), [()])
        self.assertEqual(majority.argmax(), 0)
        self.assertEqual(cost.argmax(), 1)

    def test_train_only_fallback(self):
        model = fit_prior(np.array([[1., 2.]]), np.array([0]), np.array([50]), [(1.,)])
        _, _, levels = prior_scores(model, [50, 60], [(2.,), (2.,)])
        self.assertEqual(levels, ['size_only', 'problem_only'])

    def test_true_size_excludes_vrp_depot(self):
        self.assertEqual(true_size('OVRP', instance()), 2)
        self.assertEqual(true_size('TSP', torch.zeros(1, 50, 2)), 50)

    def test_synchronized_permutation(self):
        x = instance((.1, -.2, .3))
        x.update(service_time=[1., 2., 3.], tw_start=[4., 5., 6.], tw_end=[7., 8., 9.])
        order = np.array([2, 0, 1])
        y = permute_instance('OVRPBTW', x, order)
        np.testing.assert_array_equal(y['depot_xy'], x['depot_xy'])
        for field in ('node_xy', 'node_demand', 'service_time', 'tw_start', 'tw_end'):
            np.testing.assert_array_equal(y[field], np.asarray(x[field])[order])

    def test_preserve_truncated_backhaul_starts(self):
        x = instance((.1, .1, -.1, .1, .1, .1))
        order = permutation('VRPB', x, 'preserve_roles', 2, 'RELD_MOEL')
        before = physical_starts('VRPB', x, np.arange(6), 'RELD_MOEL')
        after = physical_starts('VRPB', x, order, 'RELD_MOEL')
        self.assertEqual(before, after)
        self.assertEqual(len(before), 4)

    def test_single_reld_does_not_invent_index_start_subset(self):
        self.assertIsNone(physical_starts('CVRP', {}, np.arange(150), 'RELD_CVRP'))

    def test_tsp_preserves_physical_endpoints(self):
        x = torch.rand(1, 20, 2)
        order = permutation('TSP', x, 'preserve_roles', 3)
        self.assertEqual(order[0], 0)
        self.assertEqual(order[-1], 19)


if __name__ == '__main__':
    unittest.main()
