"""Small CPU-only safety checks. Never srun, Docker, initialize CUDA or bind a GPU."""

import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import r113_wave as wave


class WaveTests(unittest.TestCase):
    def plan(self, root):
        plan = copy.deepcopy(wave.load_plan(wave.PLAN))
        plan.update(root=str(root), protocol_lock=str(root / 'protocol_lock.json'),
                    plan_path=str(root / 'wave_plan.json'))
        wave.save(root / 'wave_plan.json', plan)
        wave.save(root / 'protocol_lock.json', dict(prepared_root='/prepared', data_root='/data',
            primary_cache='/tmp/shiys_R84_new90k_cache', pretrain_root=str(root / 'pretrain')))
        return plan

    def test_plan_is_four_slots_and_prioritized_independent_jobs(self):
        plan = wave.load_plan(wave.PLAN)
        self.assertEqual([row['job_id'] for row in plan['jobs']], list(wave.JOB_IDS))
        self.assertEqual(len(plan['slots']), 4)
        self.assertEqual({row['job'] for row in plan['slots'].values()}, {1465, 1475})
        self.assertTrue(all('cap' not in row for row in plan['slots'].values()))
        self.assertNotIn('r113_finish.py', wave.R113_RUNTIME_FILES)
        self.assertIn('r113_diagnostics.py', wave.R113_RUNTIME_FILES)
        self.assertTrue(all(row['allowed_slots'] == ['4090_0', '4090_1'] for row in plan['jobs'][:2]))
        with patch.object(wave.subprocess, 'Popen') as popen, patch.object(wave.subprocess, 'run') as run:
            result = wave.status(wave.PLAN)
            self.assertFalse(result['launches_work'])
            popen.assert_not_called()
            run.assert_not_called()

    def test_exact_allocation_transport_and_container(self):
        for slot, item in wave.SLOTS.items():
            command = wave.srun_command(wave.PLAN, 'A_scratch', slot, 1)
            for flag in ('--jobid=' + str(item['job']), '--overlap', '--exact', '--gres=none',
                         '--cpus-per-task=4', '--mem=9G', '--nodelist=' + item['node']):
                self.assertIn(flag, command)
            self.assertNotIn('ssh', command)
            self.assertNotIn('witness', command[-1])
            with patch.dict(os.environ, {wave.TOKEN_ENV: 'secret', 'SLURM_JOB_ID': str(item['job']),
                                         'SLURM_STEP_ID': '3'}):
                argv, env = wave.container_command(wave.PLAN, 'A_scratch', slot, 1, [0, 1, 2, 3])
            self.assertEqual(argv[:6], ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', '2022'])
            self.assertIn(wave.CONTAINERS[item['node']][1], argv)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], item['uuid'])
            self.assertIn('CUDA_VISIBLE_DEVICES=' + item['uuid'], argv)
            self.assertNotIn('r112_launch', ' '.join(argv))

    def test_environment_strips_ddp(self):
        with patch.dict(os.environ, RANK='1', WORLD_SIZE='2', LOCAL_RANK='1',
                        PMI_RANK='1', PMIX_RANK='1', SLURM_NTASKS='2', MASTER_ADDR='old'):
            env = wave.environment('GPU-test')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-test')
        self.assertEqual(env['OMP_NUM_THREADS'], '4')
        self.assertEqual(env['WANDB_MODE'], 'offline')
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'PMI_RANK', 'PMIX_RANK', 'SLURM_NTASKS', 'MASTER_ADDR'):
            self.assertNotIn(key, env)

    def test_original_mutex_busy_inode_and_content_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'original.json'
            path.write_text('{"original":true}\n')
            content, inode = path.read_bytes(), path.stat().st_ino
            with patch.object(wave, 'mutex_path', return_value=path):
                with path.open('r+') as occupied:
                    fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(wave.ResourceBusy):
                        with wave.original_mutex('3090_0', {str(path): inode}):
                            self.fail('busy lock acquired')
                with wave.original_mutex('3090_0', {str(path): inode}):
                    with path.open('r+') as other:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, 'inode changed'):
                    with wave.original_mutex('3090_0', {str(path): -1}):
                        self.fail('wrong inode acquired')
            self.assertEqual((path.read_bytes(), path.stat().st_ino), (content, inode))

    def test_capacity_then_exact_pipeline_without_caps_or_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            for arm in wave.ARM_IDS:
                rows = wave.commands(plan, arm, '4090_0', {})
                expected = ['capacity', 'train', 'finalize'] if arm == 'A_scratch' else [
                    'capacity', 'pretrain', 'train', 'finalize']
                self.assertEqual([row['name'] for row in rows], expected)
                self.assertIn('code.V4.r113_witness', rows[0]['argv'])
                self.assertIn(str(wave.witness_output(plan, arm)), rows[0]['argv'])
                for row in rows:
                    self.assertIn(arm, row['argv'])
                    for forbidden in ('--resume', '--large-micro-cap', '--eval-batch-size', 'torchrun'):
                        self.assertNotIn(forbidden, row['argv'])

    def test_waiting_for_main_never_probes_or_dispatches(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            state = dict(jobs={job: dict(state='waiting_main', attempts=0) for job in wave.JOB_IDS})
            with patch.object(wave, 'load_plan', return_value=plan), \
                    patch.object(wave, 'original_mutex') as mutex, \
                    patch.object(wave, 'validated_ready') as ready, \
                    patch.object(wave.subprocess, 'Popen') as popen:
                self.assertFalse(wave.controller_tick(Path(plan['plan_path']), state, {}))
            self.assertEqual(state['state'], 'waiting_main')
            mutex.assert_not_called()
            ready.assert_not_called()
            popen.assert_not_called()

    def test_only_proved_before_init_contention_can_defer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(wave.safe_deferral(root, 75, ''))
            rejected = 'srun: error: Unable to create step for job 1475: Requested nodes are busy'
            self.assertTrue(wave.safe_deferral(root, 1, rejected))
            wave.save(root / 'entry.json', {'started': True})
            self.assertFalse(wave.safe_deferral(root, 1, rejected))
            wave.save(root / 'worker.json', dict(state='resource_busy', initialized=False, stages=[]))
            self.assertTrue(wave.safe_deferral(root, 75, ''))
            self.assertFalse(wave.safe_deferral(root, 1, ''))
            for change in ({'initialized': True}, {'stages': [{'name': 'capacity'}]}, {'state': 'failed'}):
                value = dict(state='resource_busy', initialized=False, stages=[])
                value.update(change)
                wave.save(root / 'worker.json', value)
                self.assertFalse(wave.safe_deferral(root, 75, ''))

    def test_missing_receipt_is_terminal_and_capacity_failure_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {'state': 'running'}
            wave.finish_attempt(state, root, 0, '')
            self.assertEqual(state['state'], 'failed')
            wave.save(root / 'worker.json', dict(state='failed', initialized=True,
                stages=[dict(name='capacity', state='failed')], error='OOM; keep caps'))
            wave.finish_attempt(state, root, 75, '')
            self.assertEqual(state['state'], 'failed')
            self.assertIn('keep caps', state['reason'])

    def test_priority_four_jobs_and_failed_peer_does_not_stop_others(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            wave.save(wave.dispatch(plan) / 'ready.json', {'unit_test': True})
            value = dict(plan=plan, token='test', mutex_inodes={})
            state = dict(jobs={job: dict(state='waiting_main', attempts=0) for job in wave.JOB_IDS})
            children, active = [], {}

            def popen(argv, **kwargs):
                child = Mock(pid=100 + len(children))
                child.poll.return_value = None
                children.append(child)
                return child

            with patch.object(wave, 'load_plan', return_value=plan), \
                    patch.object(wave, 'validated_ready', return_value=value), \
                    patch.object(wave, 'original_mutex'), \
                    patch.object(wave.subprocess, 'Popen', side_effect=popen) as execute:
                self.assertFalse(wave.controller_tick(Path(plan['plan_path']), state, active))
                self.assertEqual([state['jobs'][job]['slot'] for job in wave.JOB_IDS],
                                 ['4090_0', '4090_1', '3090_0', '3090_1'])
                children[0].poll.return_value = 9
                self.assertFalse(wave.controller_tick(Path(plan['plan_path']), state, active))
                self.assertEqual(state['jobs']['C_transition']['state'], 'failed')
                self.assertEqual(state['jobs']['B_snapshot']['state'], 'running')
                self.assertEqual(execute.call_count, 4)
                for child in children:
                    child.kill.assert_not_called()
                    child.terminate.assert_not_called()
            for _, stream, _ in active.values():
                stream.close()

    def test_worker_holds_original_mutex_until_all_children_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            path = root / 'original.json'
            path.write_text('{}\n')
            value = dict(plan=plan, token='test', mutex_inodes={str(path): path.stat().st_ino},
                commands={'A_scratch': {'3090_0': [dict(name='capacity', argv=['test']),
                                                  dict(name='train', argv=['test']),
                                                  dict(name='finalize', argv=['test'])]}})
            called = []

            def stage(row, receipt, output, env, fd):
                with path.open('r+') as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(os.fstat(fd).st_ino, path.stat().st_ino)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], wave.SLOTS['3090_0']['uuid'])
                called.append(row['name'])

            with patch.object(wave, 'authorized', return_value=value), \
                    patch.object(wave, 'mutex_path', return_value=path), \
                    patch.object(wave, 'worker_checks', return_value={}), patch.object(wave, 'fresh'), \
                    patch.object(wave, 'check_cache'), patch.object(wave, 'validate_capacity'), \
                    patch.object(wave, 'validate_diagnostics'), \
                    patch.object(wave, 'run_stage', side_effect=stage):
                self.assertEqual(wave.worker(Path(plan['plan_path']), 'A_scratch', '3090_0', 1), 0)
            self.assertEqual(called, ['capacity', 'train', 'finalize'])
            self.assertEqual(json.loads(path.read_text()), {})

    def test_busy_worker_does_not_start_capacity_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            value = dict(plan=plan, mutex_inodes={})
            with patch.object(wave, 'authorized', return_value=value), \
                    patch.object(wave, 'original_mutex', side_effect=wave.ResourceBusy('busy')), \
                    patch.object(wave, 'run_stage') as stage:
                self.assertEqual(wave.worker(Path(plan['plan_path']), 'A_scratch', '3090_0', 1), 75)
            stage.assert_not_called()
            receipt = wave.read(wave.attempt_path(value, 'A_scratch', 1) / 'worker.json')
            self.assertEqual((receipt['state'], receipt['initialized'], receipt['stages']),
                             ('resource_busy', False, []))

    def test_existing_capacity_receipt_blocks_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            wave.save(wave.witness_output(plan, 'A_scratch'), {'complete': False})
            with self.assertRaisesRegex(RuntimeError, 'never repeat'):
                wave.fresh(plan, 'A_scratch')

    def test_r114_direct_scheduler_finalizes_without_a_duplicate_drain_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            config = dict(plan='/queue/plan.json', manifest_path='/queue/manifest.json',
                          deployment_lock_path='/queue/deployments.lock.json', timeout_seconds=3600)
            rows = wave.commands(plan, 'R114_labels', '3090_1', config)
            self.assertEqual(len(rows), 1)
            argv = rows[0]['argv']
            self.assertIn('code.V4.r114_run_queue', argv)
            self.assertIn('--execute', argv)
            self.assertIn('--finalize', argv)
            self.assertEqual(argv[argv.index('--gpu-uuid') + 1], wave.SLOTS['3090_1']['uuid'])
            self.assertEqual(argv[argv.index('--timeout-seconds') + 1], '3600')
            for forbidden in ('--retry-failed', '--max-jobs', 'r114-drain'):
                self.assertNotIn(forbidden, argv)

    def test_r114_blocked_is_finalized_business_status_not_a_resource_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / 'queue'
            run = queue / 'scheduler_runs/run_0001'
            labels = run / 'labels.json'
            wave.save(labels, {'partial_masks': True})
            plan = dict(queue_root=str(queue), plan_sha256='plan', manifest_sha256='manifest', lock_sha256='lock')
            wave.save(queue / 'plan.json', plan)
            result = dict(status='blocked', complete=False, execution=True, automatic_retry=False,
                plan_sha256='plan', manifest_sha256='manifest', lock_sha256='lock', run_directory=str(run),
                labels={'path': str(labels)}, analysis={'status': 'complete'})
            wave.save(queue / 'scheduler_progress.json', result)
            wave.save(run / 'result.json', result)
            value = dict(r114=dict(plan=str(queue / 'plan.json'), plan_sha256='plan'))
            self.assertEqual(wave.r114_completion(value, 2), result)
            wave.save(root / 'worker.json', dict(state='blocked', exit_status=2, initialized=True,
                stages=[dict(name='paired_solver_labels')], reason='partial masks preserved'))
            state = {'state': 'running'}
            wave.finish_attempt(state, root, 2, '')
            self.assertEqual(state['state'], 'blocked')
            self.assertFalse(wave.safe_deferral(root, 2, ''))
            labels.unlink()
            with self.assertRaisesRegex(RuntimeError, 'partial labels'):
                wave.r114_completion(value, 2)

    def test_source_drift_cannot_rearm_or_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            source = root / 'frozen.py'
            source.write_text('original = True\n')
            hashes = {str(Path(wave.__file__).resolve()): wave.LOADED_SHA256,
                      str(source): wave.digest(source)}
            value = dict(main_approved=True, runner_tests_passed=True, evidence='CPU test freeze',
                plan=plan, source_sha256=hashes, source_set_sha256=wave.canonical(hashes), mutex_inodes={})
            wave.save(wave.dispatch(plan) / 'ready.json', value)
            with patch.object(wave, 'load_plan', return_value=plan), \
                    patch.object(wave, 'mutex_inodes', return_value={}):
                self.assertEqual(wave.validated_ready(Path(plan['plan_path'])), value)
                source.write_text('changed = True\n')
                with self.assertRaisesRegex(RuntimeError, 'no dispatch or retry'):
                    wave.validated_ready(Path(plan['plan_path']))

    def test_public_ready_needs_main_evidence_before_any_check(self):
        args = Mock(runner_tests_passed=False, evidence='')
        with patch.object(wave, 'require_controller'), patch.object(wave, 'check_cli') as check:
            with self.assertRaisesRegex(ValueError, 'evidence'):
                wave.ready(wave.PLAN, args)
        check.assert_not_called()

    def test_r114_worker_pins_two_cpus_single_threads_and_keeps_mutex_for_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            original = root / 'original.json'
            original.write_text('{}\n')
            queue_plan = root / 'queue/plan.json'
            wave.save(queue_plan, dict(queue_root=str(queue_plan.parent)))
            value = dict(plan=plan, mutex_inodes={str(original): original.stat().st_ino},
                r114={'plan': str(queue_plan)}, commands={'R114_labels': {'3090_1': [dict(
                    name='paired_solver_labels', argv=['python', '--cpu-ids', '{cpu_ids}'])]}})

            def stage(row, receipt, output, env, fd):
                self.assertEqual(row['argv'][:3], ['taskset', '-c', '4,5'])
                self.assertEqual(env['R114_CPU_IDS'], '4,5')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], wave.SLOTS['3090_1']['uuid'])
                for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    self.assertEqual(env[key], '1')
                with original.open('r+') as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                receipt['stages'].append(dict(name=row['name'], exit_status=2))

            with patch.object(wave, 'authorized', return_value=value), \
                    patch.object(wave, 'mutex_path', return_value=original), \
                    patch.object(wave, 'worker_checks', return_value={'cpus': [4, 5, 6, 7]}), \
                    patch.object(wave, 'fresh'), patch.object(wave, 'run_stage', side_effect=stage), \
                    patch.object(wave, 'r114_completion', return_value={'status': 'blocked'}):
                self.assertEqual(wave.worker(Path(plan['plan_path']), 'R114_labels', '3090_1', 1), 2)
            receipt = wave.read(wave.attempt_path(value, 'R114_labels', 1) / 'worker.json')
            self.assertEqual(receipt['state'], 'blocked')


if __name__ == '__main__':
    unittest.main()
