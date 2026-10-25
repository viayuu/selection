"""CPU-only R112 transport/pipeline checks; no actual srun, Docker or GPU locks."""

import fcntl
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import r112_launch as launch


class LauncherTests(unittest.TestCase):
    def test_independent_commands_and_total_allocation_resources(self):
        for arm, gpu in launch.GPU_BY_ARM.items():
            rows = launch.commands(launch.ROOT, arm)
            self.assertEqual([row['name'] for row in rows], list(launch.STAGES))
            self.assertEqual([row['cuda_visible_devices'] for row in rows], [gpu] * 4 + [''])
            for row in rows:
                self.assertEqual(row['argv'][:4], [launch.sys.executable, '-B', '-u', '-m'])
                self.assertIn(arm, row['argv'])
                self.assertIn(str(launch.ROOT), row['argv'])
                self.assertNotIn('--resume', row['argv'])
                self.assertNotIn('--large-micro-cap', row['argv'])
                self.assertNotIn('--eval-batch-size', row['argv'])
                self.assertNotIn('torchrun', ' '.join(row['argv']))
        self.assertEqual(len(set(launch.CPUS)), 8)
        self.assertFalse(set(launch.CPUS_BY_ARM[launch.ARMS[0]]) & set(launch.CPUS_BY_ARM[launch.ARMS[1]]))
        command = launch.srun_command(launch.ROOT)
        for flag in ('--jobid=1475', '--overlap', '--exact', '--gres=none', '--immediate=5',
                     '-N1', '-n1', '--nodelist=gpu05', '--cpus-per-task=8', '--mem=18G'):
            self.assertIn(flag, command)
        self.assertNotIn('ssh', command)
        with patch.object(launch.subprocess, 'Popen') as popen, patch.object(launch.subprocess, 'run') as run:
            self.assertFalse(launch.plan(launch.ROOT)['launches_work'])
            popen.assert_not_called()
            run.assert_not_called()

    def test_environment_removes_ddp_and_keeps_offline_reason(self):
        with patch.dict(os.environ, dict(RANK='1', WORLD_SIZE='2', LOCAL_RANK='1',
                MASTER_ADDR='old', PMI_RANK='1', PMIX_RANK='1', TORCHELASTIC_RUN_ID='old',
                SLURM_PROCID='1', SLURM_NTASKS='2', SLURM_JOB_ID='1475', WANDB_API_KEY='SECRET')):
            env = launch.environment(launch.UUIDS[0])
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'PMI_RANK',
                    'PMIX_RANK', 'TORCHELASTIC_RUN_ID', 'SLURM_PROCID', 'SLURM_NTASKS'):
            self.assertNotIn(key, env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], launch.UUIDS[0])
        self.assertEqual(env['OMP_NUM_THREADS'], '4')
        self.assertEqual(env['WANDB_MODE'], 'offline')
        self.assertEqual(env['SLURM_JOB_ID'], '1475')
        self.assertTrue(env['PYTHONPATH'].startswith(launch.COMPAT + ':'))
        self.assertNotIn('SECRET', json.dumps(launch.plan(launch.ROOT)))
        self.assertEqual(launch.safe_argv([launch.TOKEN_ENV + '=SECRET']),
            [launch.TOKEN_ENV + '=<REDACTED>'])

    def test_cpu_cli_against_current_runner_with_disposable_config(self):
        from . import r112_experiment as runner
        with tempfile.TemporaryDirectory(prefix='r112_cpu_cli_') as directory:
            root = Path(directory)
            protocol = dict(runner.required_lock_fields(),
                microbatch_caps=dict.fromkeys(runner.PROBLEMS, 1), eval_batch_size=1,
                data_root=str(launch.DATA), prepared_root=str(launch.PREPARED), primary_cache=str(launch.CACHE))
            launch.save(root / 'protocol_lock.json', protocol)
            result = launch.check_cli(root)
            self.assertTrue(result['cpu_only'])
            self.assertEqual(result['protocol_sha256'], launch.digest(root / 'protocol_lock.json'))
            self.assertIn('ten CPU-only stage parsers', result['cli'])
            self.assertFalse(launch.dispatch(root).exists())

    def test_witness_single_gpu_transport_never_writes_formal_metadata(self):
        payload = ['python', '-B', '-u', '-m', 'code.V4.r112_witness', '--caps', '64', '128',
                   '--output', '/tmp/main_owned_witness.json']
        command = launch.srun_command(launch.ROOT, 'witness', launch.UUIDS[0], payload)
        self.assertIn('--action witness', command[-1])
        self.assertIn(launch.UUIDS[0], command[-1])
        self.assertIn(shlex.join(payload), command[-1])
        with patch.dict(os.environ, SLURM_JOB_ID='1475', SLURM_STEP_ID='3'):
            argv, env = launch.container_command(launch.ROOT, 'witness', launch.UUIDS[0], payload)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], launch.UUIDS[0])
            self.assertIn('2022', argv)
            self.assertIn(launch.CONTAINER, argv)
            self.assertNotIn(launch.TOKEN_ENV, ' '.join(argv))
            with patch.object(launch, 'require_entry'), patch.object(launch, 'save') as save, \
                    patch.object(launch, 'gpu_mutexes') as mutexes, \
                    patch.object(launch.subprocess, 'run', return_value=Mock(returncode=0)) as run:
                launch.entry(launch.ROOT, 'witness', launch.UUIDS[0], payload)
                self.assertEqual(run.call_args.args[0], argv)
                save.assert_not_called()
                mutexes.assert_not_called()
        with tempfile.TemporaryDirectory(prefix='r112_cpu_witness_locks_') as directory:
            paths = [Path(directory) / ('original_' + str(i) + '.json') for i in range(2)]
            for path in paths:
                launch.save(path, {'original': True})
            inodes = {str(path): path.stat().st_ino for path in paths}

            def wait():
                for path in paths:
                    with path.open('r+') as handle:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return 0

            child = Mock(pid=301)
            child.wait.side_effect = wait
            with patch.object(launch, 'worker_checks', return_value={'cpu_only_test_double': True}) as checks, \
                    patch.object(launch, 'mutex_paths', return_value=paths), \
                    patch.object(launch, 'save') as save, patch.object(launch, 'authorized') as authorized, \
                    patch.object(launch.subprocess, 'Popen', return_value=child) as execute, \
                    patch.object(launch.sys, 'stderr', io.StringIO()):
                launch.worker(launch.ROOT, 'witness', launch.UUIDS[0], payload)
                checks.assert_called_once_with((launch.UUIDS[0],), check_cache=False)
                self.assertEqual(execute.call_args.args[0], payload)
                self.assertEqual(execute.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], launch.UUIDS[0])
                child.wait.assert_called_once_with()
                save.assert_not_called()
                authorized.assert_not_called()
            self.assertEqual(inodes, {str(path): path.stat().st_ino for path in paths})
            for path in paths:
                with path.open('r+') as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(launch.read(path), {'original': True})

    def test_original_mutex_inodes_and_content_survive_busy_second_lock(self):
        with tempfile.TemporaryDirectory(prefix='r112_cpu_mutex_') as directory:
            paths = [Path(directory) / ('original_' + str(i) + '.json') for i in range(2)]
            for path in paths:
                launch.save(path, {'original': True})
            inodes = {str(path): path.stat().st_ino for path in paths}
            with patch.object(launch, 'mutex_paths', return_value=paths):
                with paths[1].open('r+') as occupied:
                    fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        with launch.gpu_mutexes(inodes):
                            self.fail('second original mutex is busy')
                    with paths[0].open('r+') as released:
                        fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with launch.gpu_mutexes(inodes):
                    self.assertEqual(inodes, {str(path): path.stat().st_ino for path in paths})
                with self.assertRaisesRegex(RuntimeError, 'inode changed'):
                    with launch.gpu_mutexes({str(path): -1 for path in paths}):
                        self.fail('wrong inode must fail')
            self.assertTrue(all(launch.read(path) == {'original': True} for path in paths))

    def test_nvml_refuses_any_compute_process_even_with_low_memory(self):
        devices = '\n'.join(gpu + ', NVIDIA GeForce RTX 4090, 1' for gpu in launch.UUIDS)
        with patch.object(launch.subprocess, 'run', side_effect=[Mock(stdout=devices), Mock(stdout='')]) as run:
            self.assertEqual(launch.nvml_idle(launch.UUIDS)['compute_processes'], [])
            self.assertTrue(all('--id=' + ','.join(launch.UUIDS) in call.args[0] for call in run.call_args_list))
        apps = launch.UUIDS[0] + ', 123, python, 1'
        with patch.object(launch.subprocess, 'run', side_effect=[Mock(stdout=devices), Mock(stdout=apps)]):
            with self.assertRaisesRegex(RuntimeError, 'GPU busy'):
                launch.nvml_idle(launch.UUIDS)
        with patch.object(launch.subprocess, 'run', return_value=Mock(stdout='unexpected wrapper text')):
            with self.assertRaisesRegex(RuntimeError, 'Malformed NVML'):
                launch.nvml_idle(launch.UUIDS)

    def test_cache_ignores_creation_receipts_not_input_identity(self):
        expected = dict(complete=True, rows=4500, pool=['BQ'], arrays={'node': {'shape': [4500, 5, 2]}},
            created_utc='nodeA', completed_utc='endA', seconds=3., test_lock_root='old')
        actual = dict(expected, created_utc='nodeB', completed_utc='endB', seconds=2., test_lock_root='new')
        self.assertTrue(launch.same_cache_record(actual, expected, 4500))
        for change in ({'pool': ['LEHD']}, {'rows': 4499}, {'arrays': {}}, {'complete': False}):
            self.assertFalse(launch.same_cache_record(dict(actual, **change), expected, 4500))

    def test_two_arms_start_before_poll_and_failure_does_not_retry(self):
        started, processes = [], []

        def popen(argv, **kwargs):
            process = Mock(pid=100 + len(started))
            arm = argv[argv.index('--arm') + 1]
            stage = argv[argv.index('-m') + 2]
            started.append((arm, stage, kwargs['env']['CUDA_VISIBLE_DEVICES'], argv[:3]))
            process.poll.side_effect = lambda: 7 if arm == launch.ARMS[0] and stage == 'train' else 0
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory(prefix='r112_cpu_parallel_') as directory:
            root = Path(directory)
            with patch.object(launch, 'authorized'), patch.object(launch.subprocess, 'Popen', side_effect=popen), \
                    patch.object(launch.time, 'sleep'):
                with self.assertRaises(launch.StageFailure):
                    launch.run_arms(root)
            self.assertEqual([row[:2] for row in started[:2]], [(arm, 'train') for arm in launch.ARMS])
            self.assertEqual([row[1] for row in started if row[0] == launch.ARMS[0]], ['train'])
            self.assertEqual([row[1] for row in started if row[0] == launch.ARMS[1]], list(launch.STAGES))
            self.assertEqual(len(processes), 6)
            for arm, stage, cuda, prefix in started:
                self.assertEqual(cuda, '' if stage == 'report' else launch.GPU_BY_ARM[arm])
                self.assertEqual(prefix, ['taskset', '--cpu-list', ','.join(map(str, launch.CPUS_BY_ARM[arm]))])
            a, b = [launch.read(launch.dispatch(root) / (arm + '.json')) for arm in launch.ARMS]
            self.assertEqual((a['state'], a['exit_status'], b['state'], b['exit_status']), ('failed', 7, 'complete', 0))
            self.assertTrue(all(row['pid'] and row['started_utc'] and row['finished_utc'] for row in b['stages']))

    def test_existing_arm_receipt_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='r112_cpu_existing_') as directory:
            root = Path(directory)
            path = launch.dispatch(root) / (launch.ARMS[1] + '.json')
            launch.save(path, {'original_failure': 19})
            original = path.read_bytes()
            with patch.object(launch.subprocess, 'Popen') as popen:
                with self.assertRaises(FileExistsError):
                    launch.run_arms(root)
                popen.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_interruption_waits_for_both_owned_children_before_returning(self):
        children = [Mock(pid=201), Mock(pid=202)]
        children[0].poll.side_effect = KeyboardInterrupt('CPU-only test interruption')
        for child in children:
            child.wait.return_value = 0
        with tempfile.TemporaryDirectory(prefix='r112_cpu_interrupt_') as directory:
            root = Path(directory)
            with patch.object(launch, 'authorized'), \
                    patch.object(launch.subprocess, 'Popen', side_effect=children) as popen:
                with self.assertRaises(KeyboardInterrupt):
                    launch.run_arms(root)
                self.assertEqual(popen.call_count, 2)
            for arm, child in zip(launch.ARMS, children):
                child.wait.assert_called_once_with()
                receipt = launch.read(launch.dispatch(root) / (arm + '.json'))
                self.assertEqual(receipt['state'], 'failed')
                self.assertEqual([row['name'] for row in receipt['stages']], ['train'])
                self.assertEqual(receipt['stages'][0]['child_exit_status'], 0)

    def test_internal_token_and_frozen_sources_are_required(self):
        with tempfile.TemporaryDirectory(prefix='r112_cpu_token_') as directory:
            root = Path(directory)
            frozen = {str(Path(launch.__file__).resolve()): launch.LOADED_LAUNCH_SHA256}
            claim = dict(root=str(root), token='original-token', source_sha256=frozen,
                mutex_inodes={'original': 1})
            launch.save(launch.dispatch(root) / 'claim.json', claim)
            with patch.object(launch, 'source_hashes', return_value=frozen), \
                    patch.object(launch, 'mutex_inodes', return_value={'original': 1}), \
                    patch.dict(os.environ, {launch.TOKEN_ENV: 'original-token'}):
                self.assertEqual(launch.authorized(root), claim)
                with patch.dict(os.environ, {launch.TOKEN_ENV: 'other-token'}):
                    with self.assertRaisesRegex(RuntimeError, 'one-shot token'):
                        launch.authorized(root)
                with patch.object(launch, 'source_hashes', return_value={'changed': 'source'}):
                    with self.assertRaisesRegex(RuntimeError, 'unchanged source'):
                        launch.authorized(root)

    def test_worker_holds_both_original_locks_until_both_arms_exit(self):
        with tempfile.TemporaryDirectory(prefix='r112_cpu_worker_') as directory:
            root = Path(directory)
            paths = [root / ('original_' + str(i) + '.json') for i in range(2)]
            for path in paths:
                launch.save(path, {'original': True})
            expected = {str(path): path.stat().st_ino for path in paths}

            def run_arms(_):
                for path in paths:
                    with path.open('r+') as handle:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return {arm: {'state': 'complete', 'exit_status': 0} for arm in launch.ARMS}

            with patch.object(launch, 'authorized', return_value={'mutex_inodes': expected}), \
                    patch.object(launch, 'mutex_paths', return_value=paths), \
                    patch.object(launch, 'worker_checks', return_value={'cpu_test': True}), \
                    patch.object(launch, 'run_arms', side_effect=run_arms):
                launch.worker(root)
            self.assertEqual(launch.read(launch.dispatch(root) / 'worker.json')['state'], 'complete')
            for path in paths:
                with path.open('r+') as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(launch.read(path), {'original': True})

    def test_unconfirmed_launch_preserves_one_shot_failure(self):
        with tempfile.TemporaryDirectory(prefix='r112_cpu_launch_') as directory:
            root = Path(directory)
            with patch.object(launch, 'require_controller'), \
                    patch.object(launch, 'check_cli', return_value={'protocol_sha256': 'final'}), \
                    patch.object(launch.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'Wait for main'):
                    launch.launch(root, None)
                path = launch.dispatch(root) / 'launch.json'
                original = path.read_bytes()
                receipt = launch.read(path)
                self.assertEqual((receipt['state'], receipt['exit_status']), ('failed', 1))
                self.assertFalse(receipt['automatic_retry'])
                with self.assertRaises(FileExistsError):
                    launch.launch(root, 'final')
                self.assertEqual(path.read_bytes(), original)
                popen.assert_not_called()

    def test_raw_stage_failure_retains_original_rc_and_log(self):
        with tempfile.TemporaryDirectory(prefix='r112_cpu_stage_') as directory:
            path = Path(directory) / 'entry.json'
            receipt = launch.record()
            process = Mock(pid=123)
            process.wait.return_value = 9
            with patch.object(launch.subprocess, 'Popen', return_value=process) as popen:
                with self.assertRaises(launch.StageFailure):
                    launch.run_stage(dict(name='container_exec', argv=['never-run']), receipt, path, {})
                self.assertEqual(popen.call_count, 1)
            row = launch.read(path)['stages'][0]
            self.assertEqual((row['state'], row['pid'], row['exit_status']), ('failed', 123, 9))
            self.assertTrue(row['started_utc'] and row['finished_utc'])
            self.assertTrue(Path(row['log']).is_file())


if __name__ == '__main__':
    unittest.main()
