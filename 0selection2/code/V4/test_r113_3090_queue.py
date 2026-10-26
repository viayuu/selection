"""Bounded CPU-only migration safety tests. No GPU, srun, Docker or signals."""

import copy
from contextlib import redirect_stderr
import fcntl
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import r113_3090_queue as queue


def adapter_module():
    spec = importlib.util.spec_from_file_location('_r113_3090_adapter_test', queue.ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QueueTests(unittest.TestCase):
    def plan(self, folder):
        plan = queue.default_plan()
        plan.update(directory=str(folder), root=str(folder), protocol_lock=str(folder / 'protocol_lock.json'),
                    plan_path=str(folder / 'plan.json'))
        queue.save(folder / 'protocol_lock.json', dict(prepared_root='/prepared', data_root='/data',
            primary_cache='/tmp/shiys_R84_new90k_cache', pretrain_root=str(folder / 'pretrain'),
            logical_global_batch=512, microbatch_caps=dict(TSP=384, CVRP=384, ATSP=512, OVRPBLTW=512)))
        return plan

    def test_only_two_original_3090s_and_balanced_priority(self):
        plan = queue.default_plan()
        self.assertEqual(list(queue.SLOTS), ['3090_0', '3090_1'])
        self.assertEqual({v['job'] for v in queue.SLOTS.values()}, {1465})
        self.assertEqual({v['node'] for v in queue.SLOTS.values()}, {'gpu03'})
        self.assertEqual({v['gpu_type'] for v in queue.SLOTS.values()}, {'3090'})
        self.assertEqual([j['job_id'] for j in plan['jobs']], ['A_scratch', 'C_transition', 'B_snapshot', 'R114_labels'])
        self.assertEqual(plan['r112_gpu_jobs'], [])
        self.assertEqual(plan['r112_watcher_owner'], 'main')
        for slot in queue.SLOTS:
            self.assertEqual(queue.SLOTS[slot], queue.original.SLOTS[slot])
        with self.assertRaises(ValueError):
            queue.require_slot('4090_0')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            queue.parser().parse_args(['worker', '--job', 'B_snapshot', '--slot', '4090_0', '--attempt', '1'])

    def test_r114_metadata_comes_from_old_ready_not_old_plan(self):
        path = queue.ROOT / 'preparation/wave_dispatch/ready.json'
        with patch.object(queue, 'read', return_value=dict(r114=dict(plan='/owner/plan.json', timeout_seconds=3600))) as read:
            plan = queue.default_plan()
        read.assert_called_once_with(path)
        self.assertEqual(plan['r114'], dict(plan='/owner/plan.json', timeout_seconds=3600))

    def test_exact_transport_never_other_allocation_or_gpu(self):
        for slot, item in queue.SLOTS.items():
            argv = queue.srun_command(queue.PLAN, 'B_snapshot', slot, 1)
            for flag in ('--jobid=1465', '--nodelist=gpu03', '--gres=none', '--overlap',
                         '--exact', '--cpus-per-task=4', '--mem=9G'):
                self.assertIn(flag, argv)
            self.assertNotIn('1475', ' '.join(argv))
            self.assertNotIn('ssh', argv)
            self.assertIn('code.V4.r113_3090_queue', argv[-1])
            with patch.dict(os.environ, {queue.TOKEN_ENV: 'new', 'SLURM_JOB_ID': '1465', 'SLURM_STEP_ID': '2'}):
                command, env = queue.container_command(queue.PLAN, 'B_snapshot', slot, 1, [1, 2, 3, 4])
            self.assertEqual(command[:6], ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', '2022'])
            self.assertIn(queue.CONTAINER, command)
            self.assertIn(str(queue.WRAPPER), command)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], item['uuid'])
            self.assertIn('SLURM_JOB_ID=1465', command)

    def test_explicit_resume_no_pretrain_or_cap_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.plan(Path(temporary))
            for arm in queue.ARMS:
                rows = queue.commands(plan, arm, '3090_0', {})
                self.assertEqual([r['name'] for r in rows], ['train', 'finalize'])
                for row in rows:
                    argv = row['argv']
                    self.assertEqual(argv[argv.index('--gpu-type') + 1], '3090' if arm == 'A_scratch' else '4090')
                    self.assertEqual('--resume' in argv, arm != 'A_scratch' and row['name'] == 'train')
                    for forbidden in ('pretrain', '--large-micro-cap', '--eval-batch-size', 'torchrun'):
                        self.assertNotIn(forbidden, argv)

    def test_capacity_is_selector_only_once_same_locked_caps(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            plan = self.plan(folder)
            value = dict(plan=plan, protocol_sha256=queue.digest(plan['protocol_lock']))
            command = queue.capacity_command(plan)
            self.assertIn('code.V4.r113_witness', command['argv'])
            self.assertEqual(command['argv'][command['argv'].index('--arm') + 1], 'A_scratch')
            trial = dict(stage='selector', finite=True, rows=512)
            proof = dict(complete=True, assigned_arm='A_scratch', device='NVIDIA GeForce RTX 3090',
                scratch_only=True, formal_updates=0, saved_weights=False,
                main_protocol_sha256=value['protocol_sha256'], trials=[dict(trial, problem=p, cap=cap)
                    for p, cap in [('TSP', 384), ('CVRP', 384), ('ATSP', 512), ('OVRPBLTW', 512)]])
            queue.save(folder / 'capacity_3090.json', proof)
            self.assertTrue(queue.validate_capacity(value)['same_locked_caps'])
            for change in (dict(device='NVIDIA GeForce RTX 4090'), dict(complete=False), dict(formal_updates=1)):
                queue.save(folder / 'capacity_3090.json', dict(proof, **change))
                with self.assertRaises(RuntimeError):
                    queue.validate_capacity(value)
            queue.save(folder / 'capacity_3090.json', proof)
            with patch.object(queue.original, 'run_stage') as run:
                queue.ensure_capacity(value, {'job': 'B_snapshot'}, folder / 'worker.json', {}, 7)
                run.assert_not_called()
            (folder / 'capacity_3090.json').unlink()
            queue.save(folder / 'capacity_attempt.json', {'initialized': True})
            with patch.object(queue.original, 'run_stage') as run, self.assertRaisesRegex(RuntimeError, 'never automatically retry'):
                queue.ensure_capacity(value, {'job': 'B_snapshot'}, folder / 'worker.json', {}, 7)
                run.assert_not_called()

    def test_original_uuid_mutex_inode_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'r81_original.json'
            queue.save(path, {'original': True})
            content, info = path.read_bytes(), path.stat()
            expected = {str(path): dict(device=info.st_dev, inode=info.st_ino)}
            with patch.object(queue, 'mutex_path', return_value=path):
                with path.open('r+') as owned:
                    fcntl.flock(owned, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(queue.ResourceBusy):
                        with queue.original_mutex('3090_0', expected):
                            self.fail('R109 mutex stolen')
                with queue.original_mutex('3090_0', expected):
                    with path.open('r+') as other, self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                remote_mount = {str(path): dict(device=info.st_dev + 23, inode=info.st_ino)}
                self.assertTrue(queue.same_mutex_files(expected, remote_mount))
                with queue.original_mutex('3090_0', remote_mount):
                    pass
                with self.assertRaisesRegex(RuntimeError, 'inode changed'):
                    with queue.original_mutex('3090_0', {str(path): dict(device=info.st_dev, inode=-1)}):
                        self.fail('changed inode accepted')
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_ino, info.st_ino)

    def test_wait_without_new_ready_never_probes_or_launches(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.plan(Path(temporary))
            state = dict(jobs={j: dict(state='waiting_main', attempts=0) for j in queue.JOBS})
            with patch.object(queue, 'load_plan', return_value=plan), \
                    patch.object(queue, 'original_mutex') as lock, \
                    patch.object(queue, 'validated_ready') as approval, \
                    patch.object(queue.subprocess, 'Popen') as popen:
                self.assertFalse(queue.controller_tick(queue.PLAN, state, {}))
            self.assertEqual(state['state'], 'waiting_main')
            lock.assert_not_called()
            approval.assert_not_called()
            popen.assert_not_called()

    def test_only_proved_preinit_contention_defers(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertFalse(queue.safe_deferral(folder, 75, ''))
            log = 'srun: error: Unable to create step for job 1465: Requested nodes are busy'
            self.assertTrue(queue.safe_deferral(folder, 1, log))
            self.assertFalse(queue.safe_deferral(folder, 1, log.replace('1465', '1475')))
            queue.save(folder / 'entry.json', {'initialized': False})
            self.assertFalse(queue.safe_deferral(folder, 1, log))
            queue.save(folder / 'worker.json', dict(state='resource_busy', initialized=False, stages=[]))
            self.assertTrue(queue.safe_deferral(folder, 75, ''))
            for update in (dict(initialized=True), dict(stages=[dict(name='capacity')]), dict(state='failed')):
                queue.save(folder / 'worker.json', dict(dict(state='resource_busy', initialized=False, stages=[]), **update))
                self.assertFalse(queue.safe_deferral(folder, 75, ''))

    def test_missing_or_initialized_failed_receipt_is_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            state = {'state': 'running'}
            queue.finish_attempt(state, folder, 0, '')
            self.assertEqual(state['state'], 'failed')
            queue.save(folder / 'worker.json', dict(state='failed', initialized=True, exit_status=1,
                error='OOM: retain caps', stages=[dict(name='train')]))
            queue.finish_attempt(state, folder, 75, '')
            self.assertEqual(state['state'], 'failed')
            self.assertIn('retain caps', state['reason'])

    def test_balanced_two_slot_dispatch_and_no_retry_failed_peer(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            plan = self.plan(folder)
            queue.save(folder / 'ready.json', {'unit_test': True})
            value = dict(plan=plan, token='test', mutex_inodes={})
            state = dict(jobs={j: dict(state='waiting_main', attempts=0) for j in queue.JOBS})
            active = {}
            def child(*args, **kwargs):
                return Mock(pid=100, poll=Mock(return_value=None))
            with patch.object(queue, 'load_plan', return_value=plan), \
                    patch.object(queue, 'validated_ready', return_value=value), \
                    patch.object(queue, 'original_mutex'), patch.object(queue.subprocess, 'Popen', side_effect=child) as popen:
                try:
                    self.assertFalse(queue.controller_tick(queue.PLAN, state, active))
                    self.assertEqual(set(active), {'A_scratch', 'C_transition'})
                    self.assertEqual(popen.call_count, 2)
                    # Finish C, keep long A running. B gets the freed physical slot.
                    c = active['C_transition']
                    queue.save(c[2] / 'worker.json', dict(state='complete', exit_status=0))
                    c[0].poll.return_value = 0
                    self.assertFalse(queue.controller_tick(queue.PLAN, state, active))
                    self.assertEqual(set(active), {'A_scratch', 'B_snapshot'})
                    self.assertEqual(state['jobs']['B_snapshot']['slot'], state['jobs']['C_transition']['slot'])
                    # A real B failure never retries; remaining independent labels may proceed.
                    b = active['B_snapshot']
                    queue.save(b[2] / 'worker.json', dict(state='failed', exit_status=1, error='fatal'))
                    b[0].poll.return_value = 1
                    self.assertFalse(queue.controller_tick(queue.PLAN, state, active))
                    self.assertEqual(state['jobs']['B_snapshot']['state'], 'failed')
                    self.assertEqual(state['jobs']['B_snapshot']['attempts'], 1)
                    self.assertIn('R114_labels', active)
                    self.assertLessEqual(len(active), 2)
                finally:
                    for _, stream, _ in active.values():
                        stream.close()

    def test_cpu_r112_owner_env_prefix_and_new_adapter_watcher(self):
        prefix = ['/usr/bin/env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=2', sys.executable, '-B', '-m']
        verify = [*prefix, 'code.V4.r112_postprocess_recovery', 'verify', '--device', 'cpu']
        watcher = [*prefix, 'code.V4.r112_postprocess_recovery', 'wait', '--publish']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'readiness.json'
            data = dict(state='ready_for_single_publication', finalized_source_api=True,
                        gpu_compute=False, needs_3090_queue=False,
                        gpu_stages=[], verify_argv=verify, watcher_argv=watcher)
            queue.save(path, data)
            result = queue.check_r112(path)
            self.assertEqual(result['verify_argv'], verify)
            self.assertEqual(result['watcher_argv'], watcher)
            self.assertEqual(result['gpu_jobs'], [])
            queue.save(path, dict(data, gpu_stages=['timing']))
            with self.assertRaises(ValueError):
                queue.check_r112(path)
            for bad in (verify.replace('verify', 'recover') if isinstance(verify, str) else
                        [x.replace('verify', 'recover') for x in verify],
                        ['/bin/bash', '-lc', 'anything']):
                with self.assertRaises(ValueError):
                    queue.owner_cpu_argv(bad, {'code.V4.r112_postprocess_recovery'}, 'verify')

    def test_no_implicit_checkpoint_or_scratch_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(queue, 'ROOT', root):
                queue.check_start({'links': {}}, 'A_scratch')
                queue.save(root / 'A_scratch/old.json', {'old': True})
                with self.assertRaises(RuntimeError):
                    queue.check_start({'links': {}}, 'A_scratch')
                queue.save(root / 'last.pt', {'checkpoint': 1})
                value = dict(links=dict(B_snapshot=dict(checkpoint=str(root / 'last.pt'), checkpoint_sha256='bad')))
                with self.assertRaisesRegex(RuntimeError, 'approved last.pt changed'):
                    queue.check_start(value, 'B_snapshot')

    def test_worker_only_preinitialization_busy_can_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            plan = self.plan(folder)
            value = dict(plan=plan, links={}, mutex_inodes={})
            with patch.object(queue, 'authorized', return_value=value), \
                    patch.object(queue, 'original_mutex'), \
                    patch.object(queue, 'worker_checks', side_effect=queue.ResourceBusy('R109 still live')), \
                    patch.object(queue.original, 'run_stage') as run:
                self.assertEqual(queue.worker(queue.PLAN, 'A_scratch', '3090_0', 1), 75)
                run.assert_not_called()
            path = queue.attempt_path(value, 'A_scratch', 1) / 'worker.json'
            self.assertEqual(queue.read(path)['state'], 'resource_busy')
            self.assertFalse(queue.read(path)['initialized'])
            def initialized(value, receipt, *args):
                receipt['initialized'] = True
                raise queue.ResourceBusy('too late to defer')
            with patch.object(queue, 'authorized', return_value=value), \
                    patch.object(queue, 'original_mutex'), patch.object(queue, 'worker_checks', return_value={}), \
                    patch.object(queue, 'check_start'), patch.object(queue.original, 'check_cache'), \
                    patch.object(queue, 'ensure_capacity', side_effect=initialized):
                self.assertEqual(queue.worker(queue.PLAN, 'A_scratch', '3090_0', 2), 1)
            self.assertEqual(queue.read(queue.attempt_path(value, 'A_scratch', 2) / 'worker.json')['state'], 'failed')

    def test_ready_command_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            plan = self.plan(folder)
            sources = {str(Path(queue.__file__).resolve()): queue.LOADED_SHA256}
            checked = dict(plan=plan, source_sha256=sources, source_set_sha256=queue.canonical(sources),
                mutex_inodes={}, commands={'test': 'original'})
            value = dict(checked, check_sha256=queue.canonical(checked), main_approved=True,
                runner_tests_passed=True, evidence='main test', token='t', confirmed_utc='utc')
            queue.save(folder / 'ready.json', value)
            with patch.object(queue, 'load_plan', return_value=plan), patch.object(queue, 'mutex_inodes', return_value={}):
                self.assertEqual(queue.validated_ready(queue.PLAN)['token'], 't')
                value['commands']['test'] = 'tampered'
                queue.save(folder / 'ready.json', value)
                with self.assertRaises(RuntimeError):
                    queue.validated_ready(queue.PLAN)

    def test_no_original_dispatch_ready_validator(self):
        self.assertNotIn('original.validated_ready(', Path(queue.__file__).read_text())
        self.assertNotIn('os.kill(', Path(queue.__file__).read_text())
        self.assertNotIn('SIGCONT', Path(queue.__file__).read_text())

    def test_environment_keeps_cuda_hidden_for_cpu_watchers(self):
        with patch.dict(os.environ, RANK='1', WORLD_SIZE='2', SLURM_NTASKS='2'):
            env = queue.environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertNotIn('RANK', env)
        self.assertNotIn('SLURM_NTASKS', env)
        value = dict(r112={'watcher_argv': ['already-owned-by-main']})
        watchers = queue.watcher_commands(value)
        self.assertEqual([r['name'] for r in watchers], ['R113', 'R114'])
        self.assertIn(str(queue.ADAPTER), watchers[0]['argv'])

    def test_private_finisher_does_not_mutate_original_module(self):
        from . import r113_finish as original_finish
        completion, sources = original_finish.completion_evidence, original_finish.frozen_sources
        adapter = adapter_module()
        private = adapter.hooks()
        self.assertIsNot(private, original_finish)
        self.assertIs(original_finish.completion_evidence, completion)
        self.assertIs(original_finish.frozen_sources, sources)
        self.assertIsNot(private.completion_evidence, completion)

    def test_standalone_adapter_cli_needs_no_external_pythonpath(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None)
        result = subprocess.run([sys.executable, '-B', str(queue.ADAPTER), '--help'], cwd='/',
                                env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('status,analyze,wait,publish', result.stdout)

    def test_adapter_not_ready_never_uses_failed_old_controller(self):
        adapter = adapter_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue.save(root / 'preparation/wave_dispatch/controller.json', dict(state='failed'))
            with patch.object(adapter, 'ROOT', root), patch.object(adapter, 'HERE', root / 'migration'), \
                    patch.object(queue.subprocess, 'Popen') as popen:
                finish = adapter.hooks()
                with self.assertRaises(finish.NotReady):
                    finish.completion_evidence(root)
                popen.assert_not_called()

    def test_adapter_accepts_linked_failed_history_but_requires_new_rc0(self):
        adapter = adapter_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            here = root / 'migration'
            plan = self.plan(here)
            plan.update(root=str(root))
            queue.save(here / 'ready.json', {'main_approved': True})
            queue.save(here / 'plan.json', {'fixture': 'new_two_slots'})
            queue.save(here / 'stop.json', {'complete': True})
            queue.save(here / 'capacity_3090.json', {'complete': True})
            capacity = dict(path=str(here / 'capacity_3090.json'), sha256=queue.digest(here / 'capacity_3090.json'))
            plan['operations_receipt'] = str(here / 'stop.json')
            links, jobs, commands = {}, {}, {}
            value = dict(plan=plan, main_approved=True, links=links, commands=commands)
            for index, arm in enumerate(queue.ARMS):
                slot = list(queue.SLOTS)[index % 2]
                uuid = queue.SLOTS[slot]['uuid']
                config = dict(gpu_type='3090' if arm == 'A_scratch' else '4090')
                queue.save(root / arm / 'protocol.json', config)
                if arm != 'A_scratch':
                    old = root / 'old' / arm / 'worker.json'
                    queue.save(old, dict(state='failed', exit_status=1, stages=[dict(name='pretrain', exit_status=0)]))
                    queue.save(root / 'pretrain' / arm / 'completion.json', dict(successful_updates=8640))
                    links[arm] = dict(source_worker=str(old), old_worker_sha256=queue.digest(old),
                        state_hash=queue.canonical(config), checkpoint_update=20160)
                commands[arm] = {slot: [dict(name='train', argv=['python', '--resume'] if arm != 'A_scratch' else ['python']),
                                        dict(name='finalize', argv=['python', 'finalize'])]}
                folder = queue.attempt_path(value, arm, 1)
                worker = dict(state='complete', exit_status=0, job=arm, slot=slot, gpu_uuid=uuid,
                    gpu_type='3090', actual_gpu_type='3090', capacity_evidence=capacity,
                    checks=dict(gpu=dict(uuid=uuid, name='NVIDIA GeForce RTX 3090'), allocation=1465,
                                container_id=queue.CONTAINER),
                    stages=[dict(row, state='complete', exit_status=0) for row in commands[arm][slot]])
                if arm in links:
                    worker['resume_link'] = links[arm]
                queue.save(folder / 'worker.json', worker)
                queue.save(folder / 'request.json', dict(job=arm, slot=slot, ready_sha256=queue.digest(here / 'ready.json')))
                queue.save(folder / 'entry.json', dict(state='complete', exit_status=0,
                    allocation=dict(job=1465), container=dict(id=queue.CONTAINER)))
                jobs[arm] = dict(state='complete', exit_status=0, attempts=1, slot=slot, gpu_uuid=uuid,
                    allocation=1465, attempt_dir=str(folder))
            queue.save(here / 'controller.json', dict(jobs=jobs))
            queue.save(root / 'preparation/wave_dispatch/controller.json', dict(state='failed'))
            with patch.object(adapter, 'ROOT', root), patch.object(adapter, 'HERE', here), \
                    patch.object(queue, 'PLAN', here / 'plan.json'), \
                    patch.object(queue, 'validated_ready', return_value=value), \
                    patch.object(queue, 'operations', return_value=links), \
                    patch.object(queue, 'validate_capacity', return_value=capacity):
                finish = adapter.hooks()
                inputs, projection = finish.completion_evidence(root)
                self.assertEqual(set(projection), set(queue.ARMS))
                self.assertEqual(projection['B_snapshot']['original_interruption']['checkpoint_update'], 20160)
                self.assertNotIn(str(root / 'preparation/wave_dispatch/controller.json'), inputs)
                original = queue.read(Path(links['B_snapshot']['source_worker']))
                self.assertEqual(original['state'], 'failed')
                jobs['B_snapshot']['state'] = 'failed'
                queue.save(here / 'controller.json', dict(jobs=jobs))
                with self.assertRaisesRegex(RuntimeError, 'never retry'):
                    finish.completion_evidence(root)


if __name__ == '__main__':
    unittest.main()
