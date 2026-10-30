"""Small CPU-only recovery witnesses, never solver execution or training labels."""

from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import r121_numeric_collection as api
from . import r114_group_worker


def simple_item(problem):
    if problem == 'TSP':
        item = [[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]]
        return item, [0, 1, 2, 3], 4.
    if problem == 'ATSP':
        item = [(np.ones((4, 4)) - np.eye(4)).tolist()]
        return item, [0, 1, 2, 3], 4.
    return dict(depot_xy=[[0., 0.]], node_xy=[[.1, 0.], [.2, 0.]], node_demand=[.3, -.3],
        capacity=1., route_limit=3.), [0, 1, 2, 0], .4


class NumericTests(unittest.TestCase):
    def test_real_saved_diagnostic_strict_first_bounded_accept_no_cost_change(self):
        diagnostic = api.base.read(api.DIAGNOSTIC)
        item = api.base.read(diagnostic['input_path'])['item']
        with self.assertRaisesRegex(ValueError, '^Route-length limit violation$'):
            api.STRICT_ROUTE('VRPBL', item, diagnostic['raw_tour'], diagnostic['raw_cost'])
        audit = api.numeric_check_route('VRPBL', item, diagnostic['raw_tour'], diagnostic['raw_cost'])
        self.assertEqual(api._environment_pin()['epsilon'], 1e-5)
        self.assertEqual(audit['native_cost'], diagnostic['raw_cost'])
        proof = audit['independent_numeric_validation']['boundaries'][0]
        self.assertEqual(proof['charged_edges'], 23)
        self.assertEqual(proof['independent_fp32_sequential_distance'], proof['threshold_fp32'])
        self.assertLessEqual(proof['raw_fp64_distance'], proof['upper_fp64_one_ulp'])
        self.assertAlmostEqual(proof['excess_over_original_strict'], 9.124937167982157e-8, places=13)
        route = proof['route']
        with self.assertRaisesRegex(ValueError, 'Route-length'):
            api._boundary_evidence('VRPBL', item, route, proof['upper_fp64_one_ulp'] + 1e-9, 0, 23)
        moved = copy.deepcopy(item)
        corners = np.asarray([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        depot = np.asarray(item['depot_xy']).reshape(2)
        next_node = np.asarray(item['node_xy'][route[2] - 1])
        moved['node_xy'][route[1] - 1] = corners[np.argmax(
            np.linalg.norm(corners - depot, axis=1) + np.linalg.norm(corners - next_node, axis=1))].tolist()
        # Independent FP32 must also pass, even if the supplied FP64 bound passes.
        with self.assertRaisesRegex(ValueError, 'Route-length'):
            api._boundary_evidence('VRPBL', moved, route, proof['threshold_fp32'], 0, 23)
        with self.assertRaisesRegex(ValueError, 'native cost disagrees'):
            api.numeric_check_route('VRPBL', item, diagnostic['raw_tour'], diagnostic['raw_cost'] + 1.)
        later_capacity = copy.deepcopy(item)
        later_capacity['node_demand'][75 - 1] = 1.
        later_capacity['node_demand'][91 - 1] = 1.
        with self.assertRaisesRegex(ValueError, 'capacity violation'):
            api.numeric_check_route('VRPBL', later_capacity, diagnostic['raw_tour'], diagnostic['raw_cost'])
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            api.numeric_check_route('VRPBL', item, diagnostic['raw_tour'][:1] + diagnostic['raw_tour'][2:], diagnostic['raw_cost'])

    def test_real_create_is_cpu_only_and_locks_exact_source_policy_without_writing(self):
        with patch.object(api.base, 'write') as write:
            result = api.create()
        self.assertEqual(result['deployments'], 89)
        self.assertEqual(result['planned_labels'], 113920)
        self.assertFalse(result['solver_execution'])
        self.assertFalse(result['prefixes_snapshotted'])
        self.assertEqual(write.call_count, 2)
        recovery = write.call_args_list[1].args[1]
        protocol = write.call_args_list[0].args[1]
        api.base.data.verify_seal(recovery, 'numeric_sha256')
        for index, key in enumerate(protocol['group_order']):
            expected = api.OVERLAY if key in recovery['input_boundary_keys'] else api.UNIVERSAL if index % 4 == 1 else api.PARENT
            self.assertEqual(Path(result['source_policy'][key]), expected)
        self.assertEqual(recovery['diagnosed_failure']['profile_sha256'],
            '68db943b4a8bc778714b41ff676c05918592e99b938a0f1318f363600b80f2b1')

    def test_actual_profile_binding_failure_proof_and_mismatch_stop(self):
        diagnostic = api.base.read(api.DIAGNOSTIC)
        task = diagnostic['task']
        protocol = api.base.read(api.PARENT / 'protocol.json')
        plan = api.base.read(protocol['plan_path'])
        inventory = api.base.read(plan['inventory_path'])
        manifest = dict(tasks=[task], endpoints=[dict(endpoint_id=task['endpoint_id'], cell='original',
            input_path=diagnostic['input_path'], input_sha256=task['input_sha256'])])
        proof = api._diagnostic_proof(api.DIAGNOSTIC, api.PARENT, protocol, manifest, inventory)
        self.assertEqual(proof['profile_sha256'], diagnostic['profile_sha256'])
        self.assertEqual(proof['binding_sha256'], diagnostic['binding_sha256'])
        bad = dict(inventory, deployments=dict(inventory['deployments']))
        bad['deployments']['VRPBL/reld_mt'] = dict(profile_sha256='NOT the diagnosed profile')
        with self.assertRaisesRegex(ValueError, 'differs; stop'):
            api._diagnostic_proof(api.DIAGNOSTIC, api.PARENT, protocol, manifest, bad)

    def test_all13_explicit_cpu_stays_cpu_after_lazy_default_device_change(self):
        self.addCleanup(torch.set_default_device, torch.get_default_device())
        keys = [key for key in api.base.read(api.PARENT / 'protocol.json')['group_order']
            if key.split('/')[0] in ('TSP', 'ATSP')]
        self.assertEqual(len(keys), 13)
        for key in keys:
            problem, method = key.split('/')
            item, tour, cost = simple_item(problem)
            seen = []
            def run(items):
                seen.append(str(items[0].device))
                items[0].numpy()
                torch.set_default_device('meta')
                return SimpleNamespace(costs=[cost], tours=[tour], audit={'feasible': True})
            session = SimpleNamespace(torch=torch, device='cpu', backend=SimpleNamespace(run=run), profile_sha256='fixture')
            task = dict(problem=problem, method=method, endpoint_id='fixture', task_id='fixture',
                input_sha256='fixture', solver_seed=42)
            with patch.object(api.base, 'sha', return_value='fixture'), \
                    patch.object(api.base, 'read', return_value=dict(item=item)), \
                    patch.object(r114_group_worker, 'reset_rng'):
                torch.set_default_device('cpu')
                solve = api.numeric_solver({'numeric_sha256': 'CPU_fixture'})
                for _ in range(2):
                    row = solve(session, dict(endpoints=[dict(endpoint_id='fixture', input_path='fixture')]), task)
                    self.assertEqual(row['tour'], tour)
                    self.assertEqual(row['cost'], cost)
                    self.assertEqual(row['numeric_execution'], 'new_explicit_cpu')
                self.assertEqual(seen, ['cpu', 'cpu'])
                self.assertIs(api.base.data.r114.check_route, api.STRICT_ROUTE)

    def test_unknown_new_original_controller_is_not_quiet(self):
        original_path = Path
        root = original_path('/tmp/explicit_CPU_fixture_original')
        recovery = dict(source_root_by_deployment={'VRPBL/reld_mt': str(root)})
        argv = ['python', '-m', 'new_original_controller_not_a_known_module', '--root', '/tmp/resource',
            '--collection-root', str(root), '--lane', '0']
        process = SimpleNamespace(stat=lambda: SimpleNamespace(st_uid=api.os.getuid()),
            parent=SimpleNamespace(name='987654321'), read_bytes=lambda: '\0'.join(argv).encode())
        proc = SimpleNamespace(glob=lambda pattern: [process])
        with patch.object(api, 'Path', side_effect=lambda p: proc if str(p) == '/proc' else original_path(p)):
            with self.assertRaisesRegex(api.base.NotReady, 'producer still active'):
                api.assert_sources_quiet(recovery, ['VRPBL/reld_mt'], 0)

    def test_bootstrap_fixed_three_sources_worker_resume_export_and_first_error(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            home = Path(directory)
            root, parent, overlay, v2 = (home / name for name in ('v3', 'original', 'inputV1', 'v2'))
            stack.enter_context(patch.object(api.base.data, 'owned', side_effect=lambda p: Path(p).resolve()))
            stack.enter_context(patch.object(api.base.data, 'require_release', return_value={}))
            stack.enter_context(patch.object(api.base.data, 'require_binding', return_value='binding'))
            stack.enter_context(patch.object(api.universal, 'require_mutex', return_value=dict(fd=1)))
            quiet = stack.enter_context(patch.object(api, 'assert_sources_quiet'))
            keys = ['ATSP/ICAM_ATSP', 'TSP/OMNI', 'VRPBL/reld_mt']
            protocol = dict(group_order=keys, lanes=1, mode='cached_native_v1', collection_sha256='native_fixture',
                plan_sha256='plan_fixture', input_manifest='fixture_manifest')
            recovery = dict(numeric_sha256='v3_fixture', parent_root=str(parent), input_boundary_root=str(overlay),
                universal_root=str(v2), native_collection_sha256='native_fixture', input_boundary_keys=['TSP/OMNI'],
                input_boundary_recovery_sha256='input_fixture', source_universal_sha256='v2_fixture',
                explicit_cpu_deployments=keys[:2], source_root_by_deployment=dict(zip(keys, map(str, (v2, overlay, parent)))))
            manifest = dict(endpoints=[], tasks=[], pools={}, manifest_sha256='manifest_fixture')
            inventory = dict(deployments={})
            initial, task_records = {}, {}
            for key, gid in zip(keys, (6, 14, 16)):
                p, method = key.split('/')
                source = api.source_root(recovery, key)
                item, tour, cost = simple_item(p)
                manifest['pools'][p] = [dict(method=method, global_id=gid)]
                inventory['deployments'][key] = dict(profile_sha256='profile_' + key, status='recipe_present_runtime_pending')
                tasks = []
                for index, cell in enumerate(('original', 'z00')):
                    eid = p + '_' + cell
                    path = home / 'inputs' / (eid + '.json')
                    api.base.write(path, dict(item=item))
                    task = dict(task_id=eid + '__' + method, endpoint_id=eid, base_id=p + '_source1',
                        problem=p, method=method, global_id=gid, input_sha256=api.base.sha(path),
                        solver_seed=42, deployment_id=key + '@fixture')
                    tasks.append(task)
                    manifest['tasks'].append(task)
                    manifest['endpoints'].append(dict(endpoint_id=eid, input_path=str(path),
                        input_sha256=task['input_sha256'], cell=cell, problem=p))
                    row = dict(task, stage='label', version=api.base.data.VERSION, status='ok', source='fresh_solver',
                        cost=cost, tour=tour, route_audit=api.base.data.check_tour(p, item, tour, cost),
                        collection_sha256='native_fixture', binding_sha256='binding', device='cuda:0',
                        profile_sha256='profile_' + key, execution_mode='cached_native_v1',
                        model_seed=api.base.data.r114.seed_for(2, p, method, 'model'))
                    if key == keys[1]:
                        row.update(input_boundary_version=api.boundary.VERSION,
                            input_boundary_recovery_sha256='input_fixture', observed_native_input={'device': 'cpu'})
                    if key == keys[0]:
                        row.update(universal_version=api.universal.VERSION, universal_sha256='v2_fixture',
                            universal_execution='new_explicit_cpu')
                    task_records[task['task_id']] = row
                api.base.write(source / 'deployment_bindings' / (api.base.slug(key) + '.json'),
                    dict(deployment=key, binding_sha256='binding', collection_sha256='native_fixture'))
                q = source / 'preflight' / (api.base.slug(key) + '.json')
                api.base.write(q, dict(task_records[tasks[0]['task_id']], stage='preflight'))
                path = source / 'labels' / (tasks[0]['task_id'] + '.json')
                api.base.write(path, task_records[tasks[0]['task_id']])
                initial[path] = path.read_bytes()
                api.base.atomic(source / 'progress' / (api.base.slug(key) + '.json'),
                    dict(next_index=1, completed=1, status='running', collection_sha256='native_fixture',
                        last_receipt=str(path), last_receipt_sha256=api.base.sha(path)))
                if key != keys[1]:
                    fail = source / 'unavailable' / (api.base.slug(key) + '.json')
                    api.base.write(fail, dict(tasks[1], stage='label', status='unavailable',
                        collection_sha256='native_fixture', binding_sha256='binding',
                        error_type='TypeError' if key == keys[0] else 'ValueError',
                        first_error="can't convert cuda:0 tensor to numpy" if key == keys[0] else 'Route-length limit violation'))
                    initial[fail] = fail.read_bytes()
                    if key == keys[2]:
                        recovery['diagnosed_failure'] = dict(deployment=key, task_id=tasks[1]['task_id'],
                            failure_path=str(fail), failure_sha256=api.base.sha(fail),
                            binding_sha256='binding', diagnostic_sha256='diagnostic_fixture')
            stack.enter_context(patch.object(api, 'load', return_value=(recovery, protocol, manifest, inventory)))
            stack.enter_context(patch.object(api.base, 'load', return_value=(protocol, manifest, inventory)))
            boot = api.bootstrap_lane(root, 0, 'release', 'binding')
            quiet.assert_called_once()
            self.assertEqual(boot['adopted_labels'], 3)
            self.assertFalse((root / 'unavailable').exists())
            for key in keys:
                self.assertEqual(boot['deployments'][key]['source_root'], recovery['source_root_by_deployment'][key])
                self.assertEqual(api.base.progress(root, key)['next_index'], 1)
            self.assertEqual(api.bootstrap_lane(root, 0, 'release', 'binding')['status'], 'already_bootstrapped')
            with self.assertRaises(api.base.NotReady):
                api.export(root)
            def make_solver(layer):
                def solve(session, unused_manifest, task):
                    row = copy.deepcopy(task_records[task['task_id']])
                    row.update(numeric_version=api.VERSION, numeric_sha256='v3_fixture',
                        numeric_execution='new_explicit_cpu' if task['problem'] in ('TSP', 'ATSP') else 'new_unchanged_session_bounded_independent_audit',
                        independent_numeric_validation=dict(version=api.CHECK_VERSION, mode='original_strict',
                            boundaries=[], original_strict_first=True, other_constraints_and_cost_unchanged=True))
                    return row
                return solve
            with patch.object(api.base, '_session', return_value=object()), \
                    patch.object(api, 'numeric_solver', side_effect=make_solver):
                for key in keys:
                    with patch('sys.stdout', new_callable=io.StringIO):
                        done = api.worker(root, key, 'label', 1, 'cuda:0', 'release', 'binding')
                    self.assertEqual(done['completed'], 1)
            self.assertTrue(api.status(root)['full89_complete'])
            result = api.export(root)
            self.assertEqual(result['labels'], 6)  # Explicit small CPU fixture, NOT the production population.
            self.assertEqual(result['independent_numeric_version'], api.CHECK_VERSION)
            self.assertEqual(result['label_provenance_counts']['adopted_v2'], 1)
            self.assertEqual(result['label_provenance_counts']['adopted_inputV1'], 1)
            self.assertEqual(result['label_provenance_counts']['adopted_original'], 1)
            for path, content in initial.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertIs(api.base.solve_native, api.ORIGINAL_SOLVE)
            # A different old error cannot become a retry even with identical task identity.
            target = manifest['tasks'][1]
            bad = home / 'unknown_source'
            api.base.write(bad / 'unavailable' / (api.base.slug(keys[0]) + '.json'),
                dict(target, stage='label', binding_sha256='binding', collection_sha256='native_fixture',
                    error_type='RuntimeError', first_error='NOT the repaired failure'))
            with self.assertRaisesRegex(api.base.NotReady, 'NOT an automatic retry'):
                api._covered_failure(recovery, bad, keys[0], target, 'label', 'binding', protocol)

    def test_literal_root_and_bootstrap_signature(self):
        with patch.object(api, 'bootstrap_lane', return_value=dict(status='bootstrap_complete')) as bootstrap, \
                patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(api.main(['bootstrap-lane', '--root', str(api.ROOT), '--lane', '0',
                '--release', 'release.json', '--binding', 'binding.json']), 0)
        bootstrap.assert_called_once_with(api.ROOT, 0, Path('release.json'), Path('binding.json'))

    def test_wrong_old_root_never_gets_an_error_result_written(self):
        with patch.object(api, 'worker', side_effect=api.base.NotReady('wrong root')), \
                patch.object(api.base, 'write') as write, patch('sys.stdout', new_callable=io.StringIO):
            code = api.main(['worker', '--root', str(api.PARENT), '--deployment', 'VRPBL/reld_mt',
                '--phase', 'label', '--max-jobs', '1', '--release', 'release', '--binding', 'binding',
                '--result', str(api.PARENT / 'DO_NOT_CREATE_fixture.json')])
        self.assertEqual(code, 3)
        write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
