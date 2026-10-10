"""Scenario semantics, route accounting, and release isolation regression tests."""

import tempfile
import unittest
import pickle
from pathlib import Path

import numpy as np
import torch

from .r58_environments import RoutingState
from .r58_scenario import CONTRACT, canonical_hash, check_route, fields, write_contract


def instance(demand=(.6, -.5, .6)):
    return dict(depot_xy=[[0., 0.]], node_xy=[[.2, 0.], [.3, 0.], [.4, 0.]][:len(demand)],
                node_demand=list(demand), capacity=1.)


class ScenarioTests(unittest.TestCase):
    def test_signed_load_is_not_classical_backhaul(self):
        result = check_route('VRPB', instance(), [0, 1, 2, 3, 0])
        self.assertFalse(result['feasible'])
        self.assertIn('backhaul_precedence', result['errors'])
        self.assertIn('delivery_capacity', result['errors'])

    def test_separate_capacities_and_pure_pickup_allowed(self):
        data = instance((.7, -.8))
        self.assertTrue(check_route('VRPB', data, [0, 1, 2, 0])['feasible'])
        self.assertTrue(check_route('VRPB', data, [0, 2, 0, 1, 0])['feasible'])

    def test_pickup_total_limited(self):
        self.assertIn('pickup_capacity', check_route('VRPB', instance((-.6, -.6)), [0, 1, 2, 0])['errors'])

    def test_open_return_is_not_billed(self):
        data = instance((.5,))
        closed, opened = check_route('VRPB', data, [0, 1, 0]), check_route('OVRPB', data, [0, 1, 0])
        self.assertAlmostEqual(closed['cost'], .4, places=6)
        self.assertAlmostEqual(opened['cost'], .2, places=6)

    def test_closed_single_customer_counts_depot_edge_twice(self):
        data = instance((.5,))
        self.assertAlmostEqual(check_route('VRPB', data, [0, 1])['cost'], .4, places=6)

    def test_length_is_distance_not_waiting_or_service(self):
        data = instance((.5,))
        data.update(route_limit=.5, service_time=[5.], tw_start=[5.], tw_end=[6.])
        self.assertTrue(check_route('VRPBLTW', data, [0, 1, 0])['feasible'])
        data['route_limit'] = .3
        self.assertFalse(check_route('VRPBLTW', data, [0, 1, 0])['feasible'])
        self.assertTrue(check_route('OVRPBLTW', data, [0, 1, 0])['feasible'])

    def test_depot_deadline_is_explicit(self):
        data = instance((.5,))
        data.update(service_time=[5.], tw_start=[5.], tw_end=[6.])
        self.assertTrue(check_route('VRPBTW', data, [0, 1, 0])['feasible'])
        data['depot_tw_end'] = 3.
        self.assertIn('depot_time_window', check_route('VRPBTW', data, [0, 1, 0])['errors'])
        self.assertTrue(check_route('OVRPBTW', data, [0, 1, 0])['feasible'])

    def test_customer_service_start_not_finish(self):
        data = instance((.5,))
        data.update(service_time=[2.], tw_start=[.2], tw_end=[.2])
        self.assertTrue(check_route('VRPBTW', data, [0, 1, 0])['feasible'])
        data['tw_start'] = [0.]
        data['tw_end'] = [.1]
        self.assertIn('customer_time_window', check_route('VRPBTW', data, [0, 1, 0])['errors'])

    def test_visit_validation(self):
        self.assertFalse(check_route('VRPB', instance(), [0, 1, 2, 0])['feasible'])
        self.assertFalse(check_route('VRPB', instance(), [0, 1, 1, 2, 3, 0])['feasible'])

    def test_non_backhaul_rejects_negative_demand(self):
        with self.assertRaises(ValueError):
            fields('OVRP', instance())

    def test_tsp_and_directed_atsp(self):
        tsp = torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]])
        self.assertTrue(check_route('TSP', tsp, [0, 1, 2, 0])['feasible'])
        matrix = torch.tensor([[[0., 1., 4.], [3., 0., 1.], [1., 3., 0.]]])
        self.assertEqual(check_route('ATSP', matrix, [0, 1, 2])['cost'], 3.)
        self.assertEqual(check_route('ATSP', matrix, [0, 2, 1])['cost'], 10.)

    def test_contract_is_versioned_and_old_outputs_not_touched(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = write_contract(Path(directory))
            self.assertEqual(payload, write_contract(Path(directory)))
            self.assertEqual(payload['contract_sha256'], canonical_hash(CONTRACT))

    def test_decoder_masks_backhaul_order_and_capacity(self):
        state = RoutingState('VRPB', [instance()], 'cpu')
        state.step(torch.zeros(1, 3, dtype=torch.long))
        state.step(torch.ones(1, 3, dtype=torch.long))
        self.assertFalse(torch.isfinite(state.mask[0, 0, 3]))
        state.step(torch.full((1, 3), 2))
        self.assertFalse(torch.isfinite(state.mask[0, 0, 3]))
        state.step(torch.zeros(1, 3, dtype=torch.long))
        self.assertTrue(torch.isfinite(state.mask[0, 0, 3]))

    def test_decoder_all_customer_starts_no_index_prefix_assumption(self):
        state = RoutingState('VRPB', [instance((-.2, .3, -.4))], 'cpu')
        self.assertEqual(state.starts.tolist(), [[1, 2, 3]])
        state.step(torch.zeros(1, 3, dtype=torch.long))
        self.assertTrue(torch.isfinite(state.mask[0, 0, 1]))
        self.assertTrue(torch.isfinite(state.mask[0, 0, 2]))

    def test_labels_do_not_enter_physical_state(self):
        a, b = instance(), instance()
        b.update(cost=[-10., 100.], ind=19, winner='fake')
        x, y = RoutingState('VRPB', [a], 'cpu'), RoutingState('VRPB', [b], 'cpu')
        torch.testing.assert_close(x.xy, y.xy)
        torch.testing.assert_close(x.mask, y.mask)

    def test_decoder_cannot_execute_forbidden_action(self):
        state = RoutingState('VRPB', [instance()], 'cpu')
        state.step(torch.zeros(1, 3, dtype=torch.long))
        state.step(torch.ones(1, 3, dtype=torch.long))
        state.step(torch.full((1, 3), 2))
        with self.assertRaises(RuntimeError):
            state.step(torch.full((1, 3), 3))

    def test_nonfinite_physical_fields_fail_closed(self):
        for key, value in (('node_demand', [float('nan')]), ('capacity', float('nan')),
                           ('speed', float('inf')), ('depot_tw_start', float('nan'))):
            data = instance((.5,))
            data[key] = value
            with self.assertRaises(ValueError):
                check_route('VRPB', data, [0, 1, 0])

    def test_normalized_capacity_tolerance_preserves_raw_units(self):
        data = instance((100.001,))
        data['capacity'] = 100.
        self.assertFalse(check_route('VRPB', data, [0, 1, 0])['feasible'])
        with self.assertRaises(RuntimeError):
            RoutingState('VRPB', [data], 'cpu')

    def test_column_provenance_cannot_be_relabelled_by_publication(self):
        from .r58_labels import check_column_metadata
        from .r58_scenario import save_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'solver.jsonl'
            with self.assertRaises(ValueError):
                check_column_metadata(path, {'lock': 'new'})
            save_json(path.with_suffix('.metadata.json'), {'lock': 'old'})
            with self.assertRaises(ValueError):
                check_column_metadata(path, {'lock': 'new'})
            check_column_metadata(path, {'lock': 'old'})

    def test_batch_policy_preserves_singleton_routefinder_and_tail(self):
        from .r58_labels import inference_batch_size, inference_plan
        items = [instance((.5,)) for _ in range(17)]
        self.assertEqual(inference_batch_size('VRPTW', 'RouteFinder'), 1)
        self.assertEqual(inference_batch_size('VRPB', 'MoSES_CaDA'), 1)
        self.assertEqual(inference_batch_size('CVRP', 'MTPOMO'), 1)
        self.assertEqual([len(b) for b in inference_plan('VRPB', items, 'RELD_MTL')], [16, 1])
        self.assertEqual(inference_plan('VRPB', items, 'RouteFinder'), [[i] for i in range(17)])

    def test_locked_registry_order_cannot_silently_change(self):
        from .r58_labels import read_lock
        from .r58_scenario import save_json
        from ..unified_selector.registry import GLOBAL_SOLVERS, POOLS, PROBLEMS
        with tempfile.TemporaryDirectory() as directory:
            payload = dict(global_solver_order=list(reversed(GLOBAL_SOLVERS)),
                           pools={p: list(POOLS[p]) for p in PROBLEMS})
            payload['lock_sha256'] = canonical_hash(payload)
            save_json(Path(directory) / 'deployments.lock.json', payload)
            with self.assertRaisesRegex(ValueError, 'registry names/order changed'):
                read_lock(Path(directory))

    def test_failed_forced_preflight_cannot_reuse_old_passing_record(self):
        import json
        from argparse import Namespace
        from unittest.mock import patch
        from . import r58_pipeline
        from .r58_labels import implementation_hashes
        from .r58_scenario import save_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / 'preflight/VRPB__RELD_MTL.json'
            save_json(record, dict(passed=True, implementation_sha256=implementation_hashes()))
            args = Namespace(root=root, stage='all', force_preflight=True)
            with patch.object(r58_pipeline, 'PROBLEMS', ['VRPB']), \
                 patch.object(r58_pipeline, 'POOLS', {'VRPB': ['RELD_MTL']}), \
                 patch.object(r58_pipeline, 'runtime_environment', return_value={'fixture': True}), \
                 patch.object(r58_pipeline.subprocess, 'run', return_value=Namespace(returncode=9)), \
                 patch.object(r58_pipeline, 'progress', return_value={'complete_preflight': True}), \
                 patch.object(r58_pipeline, 'lock') as lock:
                self.assertEqual(r58_pipeline.run(args), 2)
                lock.assert_not_called()
            self.assertFalse(json.loads(record.read_text())['passed'])
            self.assertEqual(json.loads((root / 'pipeline_state.json').read_text())['status'], 'blocked_preflight')

    def test_full_synthetic_release_regenerates_labels_and_blocks_missing_rows(self):
        import json
        from unittest.mock import patch
        from .r58_data import ScenarioDataset, release_check
        from .r58_labels import column_metadata, publish
        from .r58_scenario import file_hash, save_json
        from ..unified_selector.registry import POOLS, PROBLEMS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original_labels.pkl'
            original.write_bytes(b'legacy labels must stay unchanged')
            locked = dict(lock_sha256='synthetic-lock', pools=POOLS,
                deployments={p: {m: {'method': m} for m in POOLS[p]} for p in PROBLEMS})
            sources = {}
            for problem in PROBLEMS:
                if problem == 'TSP':
                    data = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
                    cheap, expensive = [0, 1, 2, 3], [0, 2, 1, 3]
                elif problem == 'ATSP':
                    data = torch.tensor([[[0., 1., 4.], [3., 0., 1.], [1., 3., 0.]]])
                    cheap, expensive = [0, 1, 2], [0, 2, 1]
                elif problem == 'CVRP':
                    data = dict(loc=torch.tensor([[[.2, 0.], [.3, 0.]]]),
                                depot=torch.zeros(1, 1, 2), demand=torch.tensor([[.3, .4]]))
                    cheap, expensive = [0, 1, 2, 0], [0, 1, 0, 2, 0]
                else:
                    data = instance((.3, -.4) if 'B' in problem else (.3, .4))
                    data.update(route_limit=2., service_time=[.1, .1], tw_start=[0., 0.], tw_end=[5., 5.])
                    cheap, expensive = [0, 1, 2, 0], [0, 1, 0, 2, 0]
                for split in ('train', 'val', 'test'):
                    source = root / 'source' / f'{problem}{split}.pkl'
                    source.parent.mkdir(exist_ok=True)
                    with source.open('wb') as stream:
                        pickle.dump([data], stream)
                    sources[problem, split] = ([data], source)
                    for slot, method in enumerate(POOLS[problem]):
                        route = cheap if slot == 1 else expensive
                        cost = check_route(problem, data, route)['cost']
                        column = root / 'scenario_v2' / f'{problem}{split}' / 'columns' / f'{method}.jsonl'
                        column.parent.mkdir(parents=True, exist_ok=True)
                        column.write_text(json.dumps(dict(index=0, route=route, cost=cost)) + '\n')
                        save_json(column.with_suffix('.metadata.json'),
                            column_metadata(locked, problem, method, split, 1, file_hash(source)))
            with patch('code.V4.r58_labels.read_lock', return_value=locked), \
                 patch('code.V4.r58_labels.load_instances', side_effect=lambda p, s: sources[p, s]), \
                 patch('code.V4.r58_data.read_lock', return_value=locked):
                released = publish(root)
                self.assertTrue(released['ready'])
                release_check(root)
                dataset = ScenarioDataset(root, 'VRPB', 'train', POOLS['VRPB'])
                self.assertEqual(dataset.raw['winner'].tolist(), [1])
                self.assertEqual(dataset.labels['0']['rank'][0], 1)
                bad = root / 'scenario_v2/VRPBtrain/columns' / f'{POOLS["VRPB"][0]}.jsonl'
                bad.write_text('')
                with self.assertRaisesRegex(RuntimeError, 'training prohibited'):
                    publish(root)
                with self.assertRaisesRegex(ValueError, 'complete validated'):
                    release_check(root)
            self.assertEqual(original.read_bytes(), b'legacy labels must stay unchanged')

    def test_best_checkpoint_survives_interrupted_last_save(self):
        from .r58_experiment import resume_checkpoint, save_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = dict(epoch=1, history=[{'epoch': 2}], optimizer={'step': 2})
            best = dict(epoch=2, history=[{'epoch': 2}, {'epoch': 3}], optimizer={'step': 3})
            save_checkpoint(root / 'last.pt', last)
            save_checkpoint(root / 'best.pt', best)
            recovered = resume_checkpoint(root, 'cpu')
            self.assertEqual(recovered['epoch'], 2)
            self.assertEqual(recovered['optimizer']['step'], 3)
            self.assertTrue((root / 'best_eval.json').exists())

    def test_test_gate_rejects_checkpoint_from_another_scenario(self):
        from .r58_experiment import check_test_checkpoint
        metric = {'macro_actual_regret_pct': .5}
        checkpoint = dict(args=dict(deployment_lock_sha256='old', release_sha256='release'),
            macro=metric, epoch=0, history=[dict(val=dict(macro=metric))])
        result = dict(best=dict(epoch=1, val=dict(macro=metric)), best_actual_regret_pct=.5)
        with self.assertRaises(ValueError):
            check_test_checkpoint(checkpoint, result, {'lock_sha256': 'new'}, 'release')
        checkpoint['args']['deployment_lock_sha256'] = 'new'
        check_test_checkpoint(checkpoint, result, {'lock_sha256': 'new'}, 'release')
        with self.assertRaises(ValueError):
            check_test_checkpoint(checkpoint, result, {'lock_sha256': 'new'}, 'another-release')

    def test_test_gate_rejects_nonminimum_or_inconsistent_selection(self):
        from .r58_experiment import check_test_checkpoint
        metric = {'macro_actual_regret_pct': .5}
        checkpoint = dict(args=dict(deployment_lock_sha256='lock', release_sha256='release'),
            macro=metric, epoch=1, history=[dict(val=dict(macro={'macro_actual_regret_pct': .4})),
                                           dict(val=dict(macro=metric))])
        result = dict(best=dict(epoch=2, val=dict(macro=metric)), best_actual_regret_pct=.5)
        with self.assertRaisesRegex(ValueError, 'not the validation minimum'):
            check_test_checkpoint(checkpoint, result, {'lock_sha256': 'lock'}, 'release')
        checkpoint['history'][0]['val']['macro']['macro_actual_regret_pct'] = .6
        check_test_checkpoint(checkpoint, result, {'lock_sha256': 'lock'}, 'release')
        result['best']['epoch'] = 1
        with self.assertRaisesRegex(ValueError, 'selection disagree'):
            check_test_checkpoint(checkpoint, result, {'lock_sha256': 'lock'}, 'release')

    def test_already_stopped_resume_performs_no_more_updates(self):
        from unittest.mock import Mock
        from .r42_experiment import CostController
        from .r58_experiment import remaining_epochs
        controller = CostController()
        controller.bad_epochs = controller.patience
        update = Mock()
        for _ in remaining_epochs(controller, controller.min_epochs):
            update()
        update.assert_not_called()
        self.assertEqual(len(remaining_epochs(controller, controller.min_epochs - 1)),
                         40 - controller.min_epochs + 1)

    def test_routefinder_preserves_explicit_depot_departure_time(self):
        from .r58_environments import routefinder_depot_clock
        td = dict(locs=torch.tensor([[[0., 0.], [.2, 0.]]]), current_node=torch.tensor([0]),
            current_time=torch.tensor([[0.]]), current_route_length=torch.tensor([[0.]]),
            time_windows=torch.tensor([[[2., 10.], [0., 1.]]]),
            service_time=torch.zeros(1, 2), speed=torch.ones(1, 1),
            open_route=torch.zeros(1, 1, dtype=torch.bool),
            demand_linehaul=torch.tensor([[0., .5]]), demand_backhaul=torch.zeros(1, 2),
            used_capacity_linehaul=torch.zeros(1, 1), used_capacity_backhaul=torch.zeros(1, 1),
            vehicle_capacity=torch.ones(1, 1), capacity_original=torch.ones(1, 1),
            visited=torch.zeros(1, 2, dtype=torch.bool), distance_limit=torch.full((1, 1), float('inf')))
        with self.assertRaisesRegex(RuntimeError, 'No feasible RouteFinder action'):
            routefinder_depot_clock(td)
        td['time_windows'][0, 1, 1] = 3.
        result = routefinder_depot_clock(td)
        self.assertEqual(float(result['current_time'][0, 0]), 2.)
        self.assertTrue(bool(result['action_mask'][0, 1]))

    def test_scenario_loader_uses_new_labels_and_explicit_pool(self):
        from .r58_data import ScenarioDataset
        from .r58_scenario import file_hash, save_json
        from ..unified_selector.registry import S2I
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / 'scenario_v2/OVRPBtrain'
            dataset_dir.mkdir(parents=True)
            with (dataset_dir / 'dataset.pkl').open('wb') as stream:
                pickle.dump([instance((.5, -.4))], stream)
            with (dataset_dir / 'raw_label.pkl').open('wb') as stream:
                pickle.dump({'0': dict(cost=[2., 1.], ind=1)}, stream)
            pool = ['RELD_MOEL', 'RELD_MTL']
            save_json(dataset_dir / 'manifest.json', dict(pool=pool,
                input_sha256=file_hash(dataset_dir / 'dataset.pkl'),
                label_sha256=file_hash(dataset_dir / 'raw_label.pkl')))
            dataset = ScenarioDataset(root, 'OVRPB', 'train', pool)
            sample = dataset[0]
            self.assertEqual(sample['ind'], 1)
            self.assertEqual(sample['pool_global_ids'].tolist(), [S2I[name] for name in pool])
            self.assertEqual(int(sample['mask'].sum()), 2)
            self.assertEqual(sample['coord_dist'], 4)


if __name__ == '__main__':
    unittest.main()
