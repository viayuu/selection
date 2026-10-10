"""CPU-only configuration, persistent workers, merge/resume/failure fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from .r58_backends import AdaptedDiffusion, AdaptedRouteFinder, REVISED_RECIPE
from .r58_execution import (ExecutionSession, GPU_UUID, PrivateMPS, WORKER_THREAD_ENV, _write_json,
    execution_profile, group_batches, make_execution_plan, merge_shards,
    read_execution_plan, require_budget_approval, validate_execution_plan)
from .r58_scenario import canonical_hash, check_route


def item(n=2, fail=False):
    return dict(depot_xy=[[0., 0.]], node_xy=[[.1 * (i + 1), 0.] for i in range(n)],
                node_demand=[.1] * n, fail=fail)


class FixtureBackend:
    def __init__(self, problem, method):
        self.problem, self.method, self.calls = problem, method, 0

    def profile(self):
        return dict(fixture=True, problem=self.problem, method=self.method)

    def run(self, items):
        self.calls += 1
        if any(data.get('fail') for data in items):
            raise RuntimeError('Synthetic solver failure')
        tours = [[0, *range(1, len(data['node_xy']) + 1), 0] for data in items]
        costs = [check_route(self.problem, data, route)['cost'] for data, route in zip(items, tours)]
        return SimpleNamespace(tours=tours, costs=costs, audit={'execution_witness': {'calls': self.calls}})


def fixture_factory(problem, method, device):
    if device != 'cpu':
        raise AssertionError('CPU fixture attempted GPU execution')
    return FixtureBackend(problem, method)


class PlanTests(unittest.TestCase):
    def test_global_rf_mode_and_unchanged_other_batch_widths(self):
        for mode, workers in (('serial1', 1), ('normal4', 4), ('mps4', 4)):
            plan = make_execution_plan(mode)
            self.assertEqual(execution_profile('VRPB', 'RouteFinder', plan)['workers'], workers)
            self.assertEqual(execution_profile('CVRP', 'MoSES_RF', plan)['inference_batch_size'], 1)
            self.assertEqual(execution_profile('VRPB', 'RELD_MTL', plan)['workers'], 1)
            self.assertEqual(execution_profile('VRPB', 'RELD_MTL', plan)['inference_batch_size'], 16)
            self.assertEqual(execution_profile('CVRP', 'BQ', plan)['inference_batch_size'], 1)
            self.assertEqual(execution_profile('CVRP', 'BQ', plan)['thread_environment'], WORKER_THREAD_ENV)

    def test_missing_plan_or_changed_identity_does_not_get_default(self):
        for change in ({'seed': 3}, {'seed': True}, {'rf_mode': 'auto'}, {'device_uuid': 'GPU-other'},
                       {'revision': 0}, {'recipe': 'old'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_execution_plan(dict(make_execution_plan('normal4'), **change))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                read_execution_plan(directory)

    def test_whole_100h_approval_bound_to_exact_plan_sources_and_128_deployments(self):
        plan, hashes = make_execution_plan('normal4'), {'frozen.py': 'abc'}
        approval = dict(approved=True, execution_plan_sha256=canonical_hash(plan),
            implementation_sha256=hashes, deployments=128, whole_projected_gpu_hours=99.9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'execution_approval.json'
            with self.assertRaises(FileNotFoundError):
                require_budget_approval(directory, plan, hashes)
            _write_json(path, approval)
            self.assertEqual(require_budget_approval(directory, plan, hashes), approval)
            for change in ({'whole_projected_gpu_hours': 100.01}, {'whole_projected_gpu_hours': True},
                           {'approved': False}, {'deployments': 127}, {'execution_plan_sha256': 'other'},
                           {'implementation_sha256': {'frozen.py': 'changed'}}):
                _write_json(path, dict(approval, **change))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    require_budget_approval(directory, plan, hashes)

    def test_grouping_retains_original_batches_and_size_boundaries(self):
        items = [item(2) for _ in range(33)] + [item(3) for _ in range(17)]
        batches = [list(range(0, 16)), list(range(16, 32)), [32], list(range(33, 49)), [49]]
        groups = group_batches('VRPB', items, batches, max_instances=20)
        self.assertEqual([g['indices'] for g in groups], [batches[0], batches[1] + [32], batches[3] + [49]])

    def test_execution_module_is_part_of_frozen_source_set(self):
        from .r58_labels import implementation_hashes
        self.assertIn('r58_execution.py', implementation_hashes())

    def test_cpu_construction_never_starts_mps_or_workers(self):
        with patch.object(PrivateMPS, '__enter__') as enter, \
             patch('code.V4.r58_execution.mp.get_context') as spawn:
            ExecutionSession('VRPB', 'RouteFinder', make_execution_plan('mps4'), 'unused', {}, instances=[item()])
            enter.assert_not_called()
            spawn.assert_not_called()

    def test_merge_rejects_missing_duplicate_or_foreign_worker_rows(self):
        items = [item() for _ in range(2)]
        metadata = {'execution_profile': {'inference_batch_size': 1}}
        rows = [dict(index=i, route=[0, 1, 2, 0], reported_cost=check_route('VRPB', data, [0, 1, 2, 0])['cost'],
                     cost=check_route('VRPB', data, [0, 1, 2, 0])['cost'], worker=i, inference_batch=[i])
                for i, data in enumerate(items)]
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'worker{i}.json' for i in range(2)]
            valid = [dict(metadata=metadata, worker=i, batches=[[i]], rows=[rows[i]]) for i in range(2)]
            for path, payload in zip(paths, valid):
                _write_json(path, payload)
            merged = Path(directory) / 'merged.json'
            self.assertEqual(len(merge_shards('VRPB', items, paths, [0, 1], metadata, merged)), 2)
            for invalid in (dict(valid[1], rows=[]), dict(valid[1], rows=[rows[0]]),
                            dict(valid[1], metadata={'wrong': 'source'})):
                _write_json(paths[1], invalid)
                merged.unlink(missing_ok=True)
                with self.assertRaises(ValueError):
                    merge_shards('VRPB', items, paths, [0, 1], metadata, merged)
                self.assertFalse(merged.exists())


class SessionTests(unittest.TestCase):
    def session(self, directory, mode='serial1', items=None, metadata=None, expected_profile=None):
        return ExecutionSession('VRPB', 'RouteFinder', make_execution_plan(mode), directory,
            metadata or {'input_sha256': 'synthetic-only'}, instances=items or [item() for _ in range(8)],
            expected_profile=expected_profile, timeout_seconds=60, _test_backend_factory=fixture_factory)

    def test_persistent_serial_and_four_workers_two_anchor_passes(self):
        groups = [dict(id='anchor2a', indices=[0, 1, 2, 3]), dict(id='anchor2b', indices=[4, 5, 6, 7])]
        for mode in ('serial1', 'normal4'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with self.session(directory, mode) as session:
                    session.prepare([groups[0]])
                    first = session.run_groups(groups, pass_id=1)
                    second = session.run_groups(groups, pass_id=2)
                    self.assertEqual(first['worker_pids'], second['worker_pids'])
                    self.assertEqual(len(first['worker_pids']), 1 if mode == 'serial1' else 4)
                    self.assertEqual([row['index'] for row in first['rows']], list(range(8)))
                    self.assertTrue(all(group['solved'] == 4 for group in first['groups']))
                    self.assertTrue(all(group['resumed'] == 0 for group in second['groups']))
                    self.assertGreater(second['rows'][0]['execution_witness']['calls'],
                                       first['rows'][0]['execution_witness']['calls'])
                    self.assertGreaterEqual(first['completion_seconds'], sum(g['seconds'] for g in first['groups']))
                self.assertTrue(all(not process.is_alive() for process in session.processes))

    def test_spawn_locks_native_thread_env_and_restores_parent_shell_values(self):
        import os
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {key: '4' for key in WORKER_THREAD_ENV}):
            with self.session(directory) as session:
                session.prepare([[0]])
                result = session.run_groups([[0, 1]], pass_id='threads')
                self.assertEqual(result['execution_profile']['thread_environment'], WORKER_THREAD_ENV)
                self.assertEqual({key: os.environ[key] for key in WORKER_THREAD_ENV},
                                 {key: '4' for key in WORKER_THREAD_ENV})

    def test_resume_reads_only_matching_checked_worker_shards(self):
        group = [dict(id='stable', indices=[0, 1, 2, 3])]
        with tempfile.TemporaryDirectory() as directory:
            with self.session(directory) as session:
                session.prepare(group)
                first = session.run_groups(group, pass_id='production')
            with self.session(directory) as session:
                session.prepare(group)
                replay = session.run_groups(group, pass_id='production')
            self.assertEqual(replay['rows'], first['rows'])
            self.assertEqual(replay['groups'][0]['resumed'], 4)
            self.assertEqual(replay['groups'][0]['solved'], 0)
            with self.session(directory, metadata={'input_sha256': 'different'}) as session:
                session.prepare(group)
                with self.assertRaisesRegex(RuntimeError, 'another execution/group'):
                    session.run_groups(group, pass_id='production')

    def test_partial_checkpoint_replays_full_width16_batch_and_retains_tail(self):
        items = [item() for _ in range(17)]
        groups = [dict(id='full_and_tail', indices=list(range(17)))]
        with tempfile.TemporaryDirectory() as directory:
            def session():
                return ExecutionSession('VRPB', 'RELD_MTL', make_execution_plan('normal4'), directory,
                    {'input': 'synthetic-only'}, instances=items, _test_backend_factory=fixture_factory,
                    timeout_seconds=60)
            with session() as first:
                first.prepare(groups)
                result = first.run_groups(groups, pass_id='production')
            path = Path(result['groups'][0]['worker_paths'][0])
            checkpoint = json.loads(path.read_text())
            checkpoint['rows'] = [row for row in checkpoint['rows'] if row['index'] not in (1, 16)]
            _write_json(path, checkpoint)
            with session() as resumed:
                resumed.prepare(groups)
                result = resumed.run_groups(groups, pass_id='production')
            self.assertEqual(result['groups'][0]['resumed'], 15)
            self.assertEqual(result['groups'][0]['solved'], 2)
            self.assertEqual(result['rows'][1]['inference_batch'], list(range(16)))
            self.assertEqual(result['rows'][16]['inference_batch'], [16])

    def test_failure_stops_workers_and_does_not_publish_incomplete_group(self):
        items = [item(), item(), item(fail=True), item()]
        with tempfile.TemporaryDirectory() as directory:
            with self.session(directory, items=items) as session:
                session.prepare([[0]])
                with self.assertRaisesRegex(RuntimeError, 'Synthetic solver failure'):
                    session.run_groups([[0, 1, 2, 3]], pass_id='production')
                self.assertFalse(session.connections)
            self.assertFalse((Path(directory) / 'pass_production/group_0/merged.json').exists())
            self.assertTrue(all(not process.is_alive() for process in session.processes))

    def test_actual_profile_must_match_expected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Actual worker deployment differs'):
                with self.session(directory, expected_profile={'not': 'actual'}):
                    self.fail('Mismatched deployment entered session')

    def test_invalid_groups_rejected_before_solver_work(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.session(directory, items=[item(2), item(3)]) as session:
                session.prepare([[0]])
                for groups in ([[0, 0]], [[2]], [[0, 1]],
                               [dict(id='../escape', indices=[0])], [[0], [0]]):
                    with self.subTest(groups=groups), self.assertRaises(ValueError):
                        session.run_groups(groups, pass_id='bad')


class MPSSafetyTests(unittest.TestCase):
    def machine(self, processes='', clients='', mode='Default'):
        return patch('code.V4.r58_execution.subprocess.run', side_effect=[
            SimpleNamespace(stdout=processes), SimpleNamespace(stdout=f'{GPU_UUID}, {mode}\n'),
            SimpleNamespace(stdout=clients)])

    def test_all_user_daemon_or_target_context_blocks_start(self):
        for processes, clients, mode in (
                ('9999 8888 nvidia-cuda-mps /usr/bin/nvidia-cuda-mps-control -d\n', '', 'Default'),
                ('1000 7777 nvidia-cuda-mps /usr/bin/nvidia-cuda-mps-server\n', '', 'Default'),
                ('', f'5555, {GPU_UUID}\n', 'Default'), ('', '', 'Exclusive_Process')):
            with self.subTest(processes=processes, clients=clients, mode=mode), \
                 self.machine(processes, clients, mode), self.assertRaisesRegex(RuntimeError, 'MPS unavailable'):
                PrivateMPS().check_machine()
        with self.machine('1000 2222 python python unrelated_job.py\n', '3333, GPU-busy-other\n'):
            evidence = PrivateMPS().check_machine()
        self.assertEqual(evidence['existing_mps_processes'], [])
        self.assertEqual(evidence['existing_target_context_pids'], [])
        self.assertEqual(evidence['process_scope'], 'all users, ps -eo uid,pid,comm,args')

    def test_guard_rejection_precedes_daemon_and_private_directory_creation(self):
        with patch('code.V4.r58_execution.gpu_guard'), \
             patch('code.V4.r58_execution.os.access', return_value=True), \
             patch('code.V4.r58_execution.os.getuid', return_value=1000), \
             patch.dict('code.V4.r58_execution.os.environ', {}, clear=True), \
             patch.object(PrivateMPS, 'check_machine', side_effect=RuntimeError('foreign daemon')), \
             patch('code.V4.r58_execution.tempfile.mkdtemp') as directory, \
             patch('code.V4.r58_execution.subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'foreign daemon'):
                PrivateMPS().__enter__()
            directory.assert_not_called()
            run.assert_not_called()

    def test_context_shutdown_requires_actual_owned_client_and_cuda_success(self):
        mps = PrivateMPS()
        with patch.object(mps, 'control', side_effect=['90', '11\n12', '0', '0']) as control:
            result = mps.terminate_owned_clients([11, 12, 13, 14])
        self.assertEqual([row['client_pid'] for row in result], [11, 12])
        self.assertEqual([call.args[0] for call in control.call_args_list],
                         ['get_server_list', 'get_client_list 90', 'terminate_client 90 11', 'terminate_client 90 12'])
        with patch.object(mps, 'control', side_effect=['90', '99999']) as control:
            with self.assertRaisesRegex(RuntimeError, 'unowned client'):
                mps.terminate_owned_clients([11, 12, 13, 14])
            self.assertFalse(any(call.args[0].startswith('terminate_client') for call in control.call_args_list))
        with patch.object(mps, 'control', side_effect=['90', '11', '999']):
            with self.assertRaisesRegex(RuntimeError, 'not CUDA_SUCCESS'):
                mps.terminate_owned_clients([11])

    def test_attachment_proof_requires_four_actual_clients_on_exact_uuid(self):
        mps = PrivateMPS()
        mps.env = {'CUDA_MPS_PIPE_DIRECTORY': '/fixture/pipe', 'CUDA_MPS_LOG_DIRECTORY': '/fixture/log'}
        mps.daemon_pid, mps.guard_evidence = 90, {'fixture': True}
        with patch.object(mps, 'control', side_effect=['100', '11\n12\n13\n14']), \
             patch('code.V4.r58_execution.subprocess.run', return_value=SimpleNamespace(stdout=f'100, {GPU_UUID}\n')):
            self.assertTrue(mps.proof([11, 12, 13, 14])['verified'])
        with patch.object(mps, 'control', side_effect=['100', '11\n12\n13']):
            with self.assertRaisesRegex(RuntimeError, 'actual attachment'):
                mps.proof([11, 12, 13, 14])
        with patch.object(mps, 'control', side_effect=['100', '11\n12\n13\n14']), \
             patch('code.V4.r58_execution.subprocess.run', return_value=SimpleNamespace(stdout='100, GPU-other\n')):
            with self.assertRaisesRegex(RuntimeError, 'GPU UUID'):
                mps.proof([11, 12, 13, 14])

    def fake_session(self, directory, terminate):
        session = ExecutionSession('VRPB', 'RouteFinder', make_execution_plan('mps4'), directory,
                                   {}, instances=[item()])
        process = MagicMock()
        process.pid = 11
        process.is_alive.return_value = True
        session.processes = [process]
        session.connections = [MagicMock()]
        session.mps = SimpleNamespace(directory=None, env={'CUDA_MPS_PIPE_DIRECTORY': '/private/fixture'},
                                      terminate_owned_clients=terminate, close=MagicMock())
        return session, process

    def test_terminate_client_success_precedes_process_signal(self):
        order = []
        def terminate(pids):
            order.append('terminate_client=0')
            return [dict(client_pid=11, server_pid=90, result='0')]
        with tempfile.TemporaryDirectory() as directory:
            session, process = self.fake_session(directory, terminate)
            def signal():
                order.append('process_signal')
                process.is_alive.return_value = False
            process.terminate.side_effect = signal
            session.close()
        self.assertEqual(order, ['terminate_client=0', 'process_signal'])
        process.kill.assert_not_called()

    def test_failed_context_shutdown_never_signals_or_quits_daemon(self):
        def terminate(pids):
            raise RuntimeError('MPS shutdown not confirmed')
        with tempfile.TemporaryDirectory() as directory:
            session, process = self.fake_session(directory, terminate)
            mps = session.mps
            with self.assertRaisesRegex(RuntimeError, 'not confirmed'):
                session.close()
            process.terminate.assert_not_called()
            process.kill.assert_not_called()
            mps.close.assert_not_called()
            self.assertTrue((Path(directory) / 'mps_cleanup_blocked.json').exists())


class BackendTests(unittest.TestCase):
    def test_routefinder_passes_unaugmented_singleton_and_all_customer_starts(self):
        class Batch(dict):
            def to(self, device):
                return self
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.call = None
            def forward(self, batch, env, **kwargs):
                self.call = (batch, kwargs)
                return {'reward': torch.tensor([-.4, -.6]),
                        'actions': torch.tensor([[1, 2, 0, 0], [2, 0, 1, 0]])}
        from .r58_scenario import fields
        data = item()
        batch = Batch(locs=torch.tensor(fields('VRPB', data)['xy'], dtype=torch.float32)[None])
        policy = Policy()
        # reset/clock/mask construction is covered by the independent environment
        # suite; isolate the revised policy call and best-start selection here.
        runner = SimpleNamespace(policy=policy, env=SimpleNamespace(reset=lambda value: value))
        backend = AdaptedRouteFinder.__new__(AdaptedRouteFinder)
        backend.problem, backend.method, backend.device = 'VRPB', 'RouteFinder', torch.device('cpu')
        backend.original = SimpleNamespace(pool=SimpleNamespace(get=lambda *args: runner), key='routefinder',
            validated_buckets={50}, profile=lambda: {'budget': {}})
        backend.tensor_dict = lambda items: batch
        backend.base = SimpleNamespace(InferenceResult=lambda costs, tours, audit: SimpleNamespace(
            costs=costs, tours=tours, audit=audit))
        with patch('code.V4.r58_backends.routefinder_depot_clock', side_effect=lambda value: value):
            output = backend.run([data])
        self.assertIs(policy.call[0], batch)
        self.assertEqual(policy.call[1]['num_starts'], 2)
        self.assertEqual(policy.call[1]['decode_type'], 'multistart_greedy')
        self.assertEqual(output.tours, [[0, 1, 2, 0, 0]])
        self.assertEqual(backend.profile()['budget']['augmentation'], 1)
        with self.assertRaisesRegex(ValueError, 'singleton inference'):
            backend.run([data, data])

    def diffusion(self, is_t2t=True, broken=False):
        model = SimpleNamespace(args=SimpleNamespace(rewrite_ratio=.4),
            categorical_denoise_step=lambda: None, guided_categorical_denoise_step=lambda: None)
        module = SimpleNamespace(batched_two_opt_torch=lambda **kwargs: None)
        original = SimpleNamespace(is_t2t=is_t2t, parameters={'rewrite_ratio': .4}, model=model, module=module)
        original.profile = lambda: dict(parameters=original.parameters, budget={'denoising_steps': 50},
                                       budget_source='original NSS recipe')
        def run(items):
            for _ in range(20):
                model.categorical_denoise_step()
            module.batched_two_opt_torch(max_iterations=100)
            if is_t2t and not broken:
                for _ in range(10):
                    model.guided_categorical_denoise_step()
                module.batched_two_opt_torch(max_iterations=100)
            return SimpleNamespace(audit={}, costs=[1.], tours=[[0, 1, 2]])
        original.run = run
        bridge = SimpleNamespace(make_backend=lambda *args: original)
        with patch('code.V4.r58_backends.load_bridge', return_value=bridge):
            return AdaptedDiffusion('TSP', 'T2T' if is_t2t else 'DIFUSCO', 'cpu'), original

    def test_diffusion_sets_both_argument_copies_and_checks_genuine_guided_rewrite(self):
        backend, original = self.diffusion()
        profile = backend.profile()
        self.assertEqual(profile['recipe_revision'], REVISED_RECIPE)
        self.assertIn('Not equivalent', profile['budget_source'])
        self.assertEqual(original.parameters['inference_diffusion_steps'], 20)
        self.assertEqual(original.model.args.rewrite_steps, 1)
        result = backend.run([None])
        self.assertEqual(result.audit['execution_witness'],
                         dict(denoising_calls=20, guided_calls=10, two_opt_limits=[100, 100]))
        backend, _ = self.diffusion(broken=True)
        with self.assertRaisesRegex(ValueError, 'Actual diffusion execution budget differs'):
            backend.run([None])

    def test_diffusion_tampering_fails_and_hooks_restore_on_failure(self):
        backend, original = self.diffusion(broken=True)
        denoise, refine = original.model.categorical_denoise_step, original.module.batched_two_opt_torch
        with self.assertRaises(ValueError):
            backend.run([None])
        self.assertIs(original.model.categorical_denoise_step, denoise)
        self.assertIs(original.module.batched_two_opt_torch, refine)
        original.model.args.inference_diffusion_steps = 50
        with self.assertRaisesRegex(ValueError, 'arguments differ'):
            backend.profile()

    def test_difusco_has_no_guided_rewrite(self):
        backend, _ = self.diffusion(is_t2t=False)
        self.assertEqual(backend.run([None]).audit['execution_witness'],
                         dict(denoising_calls=20, guided_calls=0, two_opt_limits=[100]))


if __name__ == '__main__':
    unittest.main()
