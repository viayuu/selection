"""Three CPU-only launcher checks; never touch actual locks or allocations."""

from copy import deepcopy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import r111_launch as launch


class LauncherTests(unittest.TestCase):
    def test_cache_creation_receipts_are_not_input_identity(self):
        expected = dict(complete=True, rows=4500, pool=['BQ'], arrays={'node': {'shape': [4500, 5, 2]}},
            created_utc='nodeA', completed_utc='nodeA_end', seconds=3., test_lock_root='R84C')
        actual = dict(expected, created_utc='nodeB', completed_utc='nodeB_end', seconds=2., test_lock_root='R84D')
        self.assertTrue(launch.same_cache_record(actual, expected, 4500))
        self.assertFalse(launch.same_cache_record(dict(actual, pool=['LEHD']), expected, 4500))
        self.assertFalse(launch.same_cache_record(dict(actual, rows=4499), expected, 4500))
        self.assertFalse(launch.same_cache_record(dict(actual, arrays={}), expected, 4500))

    def test_exact_commands_and_no_implicit_resume(self):
        stages = launch.commands(launch.ROOT)
        self.assertEqual([s['name'] for s in stages], list(launch.STAGES))
        train = stages[0]['argv']
        for flag in ('--standalone', '--nnodes=1', '--nproc-per-node=2', '--max-restarts=0'):
            self.assertIn(flag, train)
        self.assertEqual(stages[0]['cuda_visible_devices'], ','.join(launch.UUIDS))
        for stage in stages:
            self.assertNotIn('--resume', stage['argv'])
            self.assertIn(launch.ARM, stage['argv'])
        self.assertEqual([s['cuda_visible_devices'] for s in stages[1:]], [launch.UUIDS[0]] * 3 + [''])
        command = launch.srun_command(launch.ROOT, 'token')
        for flag in ('--jobid=1475', '--overlap', '--exact', '--gres=none', '--cpus-per-task=8', '--mem=18G'):
            self.assertIn(flag, command)

    def test_original_pipeline_requires_both_rc0_and_pdf(self):
        receipts = {}
        for arm, gpu in zip(('A_tied_ce', 'B_private_ce'), launch.UUIDS):
            output = launch.PRIOR / 'preparation/dispatch' / ('R110_' + arm)
            for filename, names in (('worker.json', ('train', 'TEST', 'timing', 'LIB')),
                                    ('controller.json', ('allocation_GPU_pipeline', 'PDF'))):
                receipts[output / filename] = dict(state='complete', exit_status=0,
                    container_id=launch.CONTAINER, gpu_uuid=gpu,
                    stages=[dict(name=n, state='complete', exit_status=0) for n in names])
        with patch.object(Path, 'exists', lambda p: p in receipts), patch.object(launch, 'read', lambda p: deepcopy(receipts[p])):
            self.assertTrue(launch.predecessor_gate()['complete'])
            target = next(p for p in receipts if p.name == 'controller.json')
            receipts[target]['state'] = 'running'
            receipts[target]['exit_status'] = None
            receipts[target]['stages'][-1].update(state='running', exit_status=None)
            self.assertFalse(launch.predecessor_gate()['complete'])
            receipts[target]['stages'][-1].update(state='failed', exit_status=7)
            with self.assertRaisesRegex(RuntimeError, 'terminal failure'):
                launch.predecessor_gate()

    def test_ready_failure_and_one_shot_cpu_receipts(self):
        with tempfile.TemporaryDirectory(prefix='r111_cpu_') as directory:
            root = Path(directory)
            frozen = {str(Path(launch.__file__).resolve()): launch.LOADED_LAUNCH_SHA256}
            ready = dict(main_thread_ready=True, cpu_tests_passed=True, sol_review_passed=True,
                evidence='main CPU evidence', confirmed_utc='2026-10-25T00:00:00Z',
                source_sha256=frozen, mutex_inodes={'original': 1})
            with patch.object(launch, 'source_hashes', return_value=frozen), patch.object(launch, 'mutex_inodes', return_value={'original': 1}):
                with self.assertRaises(FileNotFoundError):
                    launch.ready(root)
                launch.save(root / 'launch_ready.json', ready)
                self.assertEqual(launch.ready(root), ready)
                ready['sol_review_passed'] = False
                launch.save(root / 'launch_ready.json', ready)
                with self.assertRaisesRegex(RuntimeError, 'main-ready'):
                    launch.ready(root)
            with patch.object(launch, 'predecessor_gate', side_effect=RuntimeError('original predecessor failed')), patch.object(launch.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'predecessor failed'):
                    launch.queue(root)
                popen.assert_not_called()
                receipt = json.loads((launch.dispatch(root) / 'queue.json').read_text())
                self.assertEqual(receipt['state'], 'failed')
                self.assertEqual(receipt['exit_status'], 1)
                with self.assertRaises(FileExistsError):
                    launch.queue(root)
            process = unittest.mock.Mock(pid=123)
            process.wait.return_value = 9
            path, record = root / 'fake_worker.json', launch.record()
            with patch.object(launch.subprocess, 'Popen', return_value=process) as popen:
                with self.assertRaises(launch.StageFailure):
                    launch.run_stage(dict(name='train', argv=['never-executed']), record, path, {})
                self.assertEqual(popen.call_count, 1)
            step = launch.read(path)['stages'][0]
            self.assertEqual((step['pid'], step['exit_status'], step['state']), (123, 9, 'failed'))
            self.assertTrue(step['started_utc'] and step['finished_utc'])
            paths = [root / 'original_0.json', root / 'original_1.json']
            for lock in paths:
                launch.save(lock, {'preserve': True})
            inodes = {str(p): p.stat().st_ino for p in paths}
            with patch.object(launch, 'mutex_paths', return_value=paths):
                with paths[1].open('r+') as occupied:
                    fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        with launch.gpu_mutexes(inodes):
                            self.fail('busy original second mutex must block dispatch')
                    with paths[0].open('r+') as released:
                        fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with launch.gpu_mutexes(inodes):
                    self.assertEqual(inodes, {str(p): p.stat().st_ino for p in paths})
            self.assertTrue(all(launch.read(p) == {'preserve': True} for p in paths))


if __name__ == '__main__':
    unittest.main()
