"""CPU-only default-device/prefix witnesses; no native model or GPU execution."""

import ast
from contextlib import ExitStack
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from . import r121_universal_collection as api
from . import r114_group_worker
from .test_r121_structural_data import fixture


def native_audit():
    path = Path('/public/home/shiys/easynco_v3_bridge/scripts/train_100k_backends.py')
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_tours')
    namespace = dict(np=np, sample_scale=lambda p, item: item.shape[1])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['audit_tours']


def tour_cost(problem, item):
    values = np.asarray(item)[0]
    tour = list(range(len(values)))
    cost = float(values[tour, np.roll(tour, -1)].sum()) if problem == 'ATSP' else float(
        np.linalg.norm(values - np.roll(values, -1, axis=0), axis=1).sum())
    return tour, cost


class UniversalTests(unittest.TestCase):
    def test_all13_second_call_stays_CPU_after_lazy_default_device_change(self):
        self.addCleanup(torch.set_default_device, torch.get_default_device())
        keys = [k for k in api.base.read(api.PARENT / 'protocol.json')['group_order']
                if k.split('/')[0] in ('TSP', 'ATSP')]
        self.assertEqual(len(keys), 13)
        audit = native_audit()
        for key in keys:
            problem, method = key.split('/')
            item = fixture(problem, 1)
            tour, cost = tour_cost(problem, item)
            observed = []

            def lazy_run(items):
                observed.append(str(items[0].device))
                certificate = audit(problem, items, [tour], [cost])
                torch.set_default_device('meta')
                return SimpleNamespace(costs=[cost], tours=[tour], audit=certificate)

            session = SimpleNamespace(torch=torch, device='cpu', backend=SimpleNamespace(run=lazy_run),
                profile_sha256='unchanged')
            manifest = dict(endpoints=[dict(endpoint_id='fixture', input_path='fixture')])
            task = dict(task_id='fixture', endpoint_id='fixture', problem=problem, method=method,
                input_sha256='fixture', solver_seed=123)
            with patch.object(api.base, 'sha', return_value='fixture'), \
                    patch.object(api.base, 'read', return_value=dict(item=item)), \
                    patch.object(r114_group_worker, 'reset_rng'):
                torch.set_default_device('cpu')
                api.ORIGINAL_SOLVE(session, manifest, task)
                with self.assertRaises(TypeError):
                    api.ORIGINAL_SOLVE(session, manifest, task)
                self.assertEqual(observed, ['cpu', 'meta'])
                observed.clear()
                torch.set_default_device('cpu')
                solve = api.explicit_solver({'universal_sha256': 'CPU_fixture'})
                for _ in range(2):
                    result = solve(session, manifest, task)
                    self.assertEqual(result['cost'], cost)
                    self.assertEqual(result['tour'], tour)
                    self.assertTrue(result['route_audit']['verified'])
                    self.assertEqual(result['universal_execution'], 'new_explicit_cpu')
                self.assertEqual(observed, ['cpu', 'cpu'])
                self.assertEqual(torch.get_default_device().type, 'meta')

    def test_other_task_keeps_original_session_solve(self):
        item = fixture('CVRP', 1)
        task = dict(problem='CVRP', task_id='fixture', endpoint_id='fixture', input_sha256='fixture')
        session = SimpleNamespace(solve=Mock(return_value=dict(cost=1., tour=[0])))
        with patch.object(api.base, 'sha', return_value='fixture'), \
                patch.object(api.base, 'read', return_value=dict(item=item)):
            result = api.explicit_solver({'universal_sha256': 'CPU_fixture'})(session,
                dict(endpoints=[dict(endpoint_id='fixture', input_path='fixture')]), task)
        session.solve.assert_called_once_with('fixture')
        self.assertEqual(result['universal_execution'], 'new_unchanged_session_solve')

    def test_bootstrap_adopts_actual_prefix_and_reuses_bytes_without_failure_flags(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            home = Path(directory)
            root, parent, overlay = (home / name for name in ('new', 'original', 'inputV1'))
            stack.enter_context(patch.object(api.base.data, 'owned', side_effect=lambda p: Path(p).resolve()))
            stack.enter_context(patch.object(api.base.data, 'require_release', return_value={}))
            stack.enter_context(patch.object(api.base.data, 'require_binding', return_value='binding'))
            stack.enter_context(patch.object(api, 'require_mutex', return_value=dict(fd=1, device=2, inode=3)))
            stack.enter_context(patch.object(api, 'assert_sources_quiet'))
            keys = ['ATSP/ICAM_ATSP', 'TSP/OMNI']
            protocol = dict(group_order=keys, lanes=1, mode='cached_native_v1',
                collection_sha256='same_native_protocol', plan_sha256='fixture')
            universal = dict(universal_sha256='v2_fixture', parent_root=str(parent),
                input_boundary_root=str(overlay), input_boundary_keys=['TSP/OMNI'],
                input_boundary_recovery_sha256='v1_fixture', native_collection_sha256=protocol['collection_sha256'])
            manifest, inventory = dict(endpoints=[], tasks=[]), dict(deployments={})
            source_contents = {}
            for key in keys:
                problem, method = key.split('/')
                source = parent if key == keys[0] else overlay
                inventory['deployments'][key] = dict(profile_sha256='profile_' + key, status='recipe_present_runtime_pending')
                item = fixture(problem, 1)
                tour, cost = tour_cost(problem, item)
                tasks = []
                for index, cell in enumerate(('original', 'z00', 'z01')):
                    endpoint_id = problem + '_' + cell
                    input_path = home / 'inputs' / (endpoint_id + '.json')
                    api.base.write(input_path, dict(item=item))
                    task = dict(task_id=endpoint_id + '__' + method, endpoint_id=endpoint_id,
                        base_id=problem + '_source1', problem=problem, method=method, global_id=6 if problem == 'ATSP' else 14,
                        input_sha256=api.base.sha(input_path), solver_seed=42, deployment_id=key + '@fixture')
                    tasks.append(task)
                    manifest['tasks'].append(task)
                    manifest['endpoints'].append(dict(endpoint_id=endpoint_id, input_path=str(input_path),
                        input_sha256=task['input_sha256'], cell=cell))
                api.base.write(source / 'deployment_bindings' / (api.base.slug(key) + '.json'),
                    dict(deployment=key, binding_sha256='binding', collection_sha256=protocol['collection_sha256']))

                def receipt(task, stage):
                    row = dict(task, stage=stage, version=api.base.data.VERSION, status='ok', source='fresh_solver',
                        collection_sha256=protocol['collection_sha256'], route_audit={'verified': True},
                        cost=cost, tour=tour, profile_sha256=inventory['deployments'][key]['profile_sha256'],
                        binding_sha256='binding', device='cuda:0', execution_mode='cached_native_v1',
                        model_seed=api.base.data.r114.seed_for(2, problem, method, 'model'))
                    if key == keys[1]:
                        row.update(input_boundary_version=api.boundary.VERSION,
                            input_boundary_recovery_sha256='v1_fixture', observed_native_input={'device': 'cpu'})
                    return row

                api.base.write(source / 'preflight' / (api.base.slug(key) + '.json'), receipt(tasks[0], 'preflight'))
                prefix = 1 if key == keys[0] else 2
                for task in tasks[:prefix]:
                    path = source / 'labels' / (task['task_id'] + '.json')
                    api.base.write(path, receipt(task, 'label'))
                    source_contents[task['task_id']] = path.read_bytes()
                last = source / 'labels' / (tasks[prefix - 1]['task_id'] + '.json')
                api.base.atomic(source / 'progress' / (api.base.slug(key) + '.json'),
                    dict(next_index=prefix, completed=prefix, status='unavailable' if prefix == 1 else 'running',
                        collection_sha256=protocol['collection_sha256'], last_receipt=str(last),
                        last_receipt_sha256=api.base.sha(last), first_error_path='preserved_old_failure'))
            api.base.write(parent / 'unavailable' / 'ATSP__ICAM_ATSP.json',
                dict(task_id='ATSP_z00__ICAM_ATSP', error_type='TypeError',
                    first_error="CPU fixture: can't convert cuda:0 device type tensor to numpy"))
            original_failure = (parent / 'unavailable' / 'ATSP__ICAM_ATSP.json').read_bytes()
            stack.enter_context(patch.object(api, 'load', return_value=(universal, protocol, manifest, inventory)))
            stack.enter_context(patch.object(api.base, 'load', return_value=(protocol, manifest, inventory)))
            adopted = api.bootstrap_lane(root, 0, 'release', 'binding')
            self.assertEqual(adopted['adopted_labels'], 3)
            self.assertEqual(adopted['deployments'][keys[0]]['next_index'], 1)
            self.assertEqual(adopted['deployments'][keys[1]]['source_root'], str(overlay))
            self.assertFalse((root / 'unavailable').exists())
            self.assertEqual(api.base.progress(root, keys[0])['status'], 'running')
            self.assertNotIn('first_error_path', api.base.progress(root, keys[0]))
            for task_id, content in source_contents.items():
                self.assertEqual((root / 'labels' / (task_id + '.json')).read_bytes(), content)
                provenance = api.base.read(root / 'adoption' / 'labels' / (task_id + '.json'))
                self.assertFalse(provenance['new_solver_execution'])
                self.assertEqual(provenance['source_receipt_sha256'], api.base.sha(root / 'labels' / (task_id + '.json')))
            self.assertEqual(api.bootstrap_lane(root, 0, 'release', 'binding')['status'], 'already_bootstrapped')
            # Even model-initialization errors gain the actual complete traceback.
            with patch.object(api.base, '_session', side_effect=RuntimeError('CPU fixture native init failure')):
                failed = api.worker(root, keys[0], 'label', 1, 'cuda:0', 'release', 'binding')
            self.assertEqual(failed['status'], 'unavailable')
            failure = api.base.read(root / 'unavailable' / 'ATSP__ICAM_ATSP.json')
            self.assertEqual(failure['task_id'], 'ATSP_z00__ICAM_ATSP')
            self.assertIn('RuntimeError: CPU fixture native init failure', failure['traceback'])
            self.assertEqual((parent / 'unavailable' / 'ATSP__ICAM_ATSP.json').read_bytes(), original_failure)
            self.assertIs(api.base.solve_native, api.ORIGINAL_SOLVE)

    def test_literal_bootstrap_root_cli_and_signature(self):
        with patch.object(api, 'bootstrap_lane', return_value=dict(status='bootstrap_complete')) as bootstrap, \
                patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(api.main(['bootstrap-lane', '--root', str(api.ROOT), '--lane', '1',
                '--release', 'release.json', '--binding', 'binding.json']), 0)
        bootstrap.assert_called_once_with(api.ROOT, 1, Path('release.json'), Path('binding.json'))


if __name__ == '__main__':
    unittest.main()
