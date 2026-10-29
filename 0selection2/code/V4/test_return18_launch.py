"""CPU-only Return18 contract checks; never contact Slurm, containers or GPUs."""

from contextlib import ExitStack
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import return18_launch as launch


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.runs = self.repo / 'code/V4/runs'
        self.dispatch = self.runs / 'Return18_post_R120/dispatch'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key, value in dict(REPO=self.repo, RUNS=self.runs, DISPATCH=self.dispatch,
                SOURCE=self.runs / 'R84_new90k_attention/input_cache',
                R120=self.runs / 'R120_multidomain_selector',
                CACHE=self.repo / 'local_cache', WRAPPER=self.repo / 'code/V4/run_v4_return18.sh').items():
            self.stack.enter_context(mock.patch.object(launch, key, value))
        self.stack.enter_context(mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        for name in launch.TRANSPORT_SOURCES:
            self.fixture(self.repo / name, 'frozen source\n')
        self.stack.enter_context(mock.patch.object(launch, 'LOADED_SHA256', launch.sha256(self.repo / launch.TRANSPORT_SOURCES[0])))
        self.script = 'code/V4/cpu_fixture.py'
        self.fixture(self.repo / self.script,
            'import os, sys\nprint("CPU fixture", flush=True)\n'
            'assert os.fstat(int(os.environ["RETURN18_GPU_MUTEX_FD"])).st_ino > 0\n'
            'sys.exit(int(sys.argv[sys.argv.index("--exit-code")+1]))\n')
        self.protocol = self.runs / 'Return18_post_R120/protocol.json'
        self.fixture(self.protocol, '{}\n')
        self.plan = dict(schema=launch.SCHEMA, name='cpu_contract', main_approved=True,
            approved_by='main CPU test only', approved_utc=datetime.now(timezone.utc).isoformat(),
            protocol=dict(path=str(self.protocol), sha256=launch.sha256(self.protocol)),
            source_sha256={name: launch.sha256(self.repo / name)
                           for name in (*launch.TRANSPORT_SOURCES, self.script)}, jobs=[])
        for index, slot in enumerate(sorted(launch.SLOTS)):
            arm = f'R{121 + index}'
            root = f'code/V4/runs/{arm}_fixture'
            argv = [sys.executable, '-B', '-u', self.script, '--root', root, '--device', 'cpu', '--exit-code', '0']
            self.plan['jobs'].append(dict(id=arm, arm=arm, slot=slot, root=root,
                stages=[dict(id='train', kind='train', argv=argv)]))
        self.path = self.dispatch / 'cpu_plan.json'
        self.save()

    @staticmethod
    def fixture(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def save(self):
        self.fixture(self.path, json.dumps(self.plan))
        return launch.validate_plan(self.path)

    def test_four_literal_slots_and_cpu_steps(self):
        plan, digest = self.save()
        self.assertEqual(len({job['slot'] for job in plan['jobs']}), 4)
        original = copy.deepcopy(launch.AUTHORIZED_NODES)
        for job in plan['jobs']:
            argv = launch.srun_command(self.path, job, 'token', digest)
            for value in ('--overlap', '--exact', '--gres=none', '--cpus-per-task=4', '--mem=9G'):
                self.assertIn(value, argv)
            self.assertNotIn('--gres=gpu:1', argv)
            self.assertIn(digest, argv)
        self.assertEqual(original, launch.AUTHORIZED_NODES)

    def test_lane_cpu_intersections_are_disjoint(self):
        for node in ('gpu03', 'gpu05'):
            first = launch.lane_cpus(launch.SLOTS[node + '_0'], range(20, 30), range(19, 31))
            second = launch.lane_cpus(launch.SLOTS[node + '_1'], range(20, 30), range(19, 31))
            self.assertEqual(first, [20, 21, 22, 23])
            self.assertEqual(second, [24, 25, 26, 27])
            self.assertFalse(set(first) & set(second))
            with self.assertRaises(ValueError):
                launch.lane_cpus(launch.SLOTS[node + '_1'], range(10), range(7))

    def test_identity_and_exact_cuda_uuid(self):
        slot = launch.SLOTS['gpu03_0']
        env = dict(SLURM_JOB_ID=str(slot['job']), SLURM_STEP_ID='5',
                   CUDA_VISIBLE_DEVICES=slot['gpu_uuid'], RETURN18_CPUS='1,2,3,4')
        launch.entry_identity(slot, 'gpu03', env, '0::/slurm/uid_123/job_1465/step_5\n')
        cpus = launch.worker_identity(slot, env, slot['identity'][:12], '0::/docker/' + slot['identity'], range(8))
        self.assertEqual(cpus, [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            launch.entry_identity(slot, 'gpu03', env, '0::/slurm/job_14650/step_5\n')
        for change in (dict(CUDA_VISIBLE_DEVICES='0'), dict(SLURM_JOB_ID='999'), dict(RETURN18_CPUS='1,2,3')):
            with self.assertRaises(ValueError):
                launch.worker_identity(slot, dict(env, **change), slot['identity'][:12], slot['identity'], range(8))
        with self.assertRaises(ValueError):
            launch.worker_identity(slot, env, 'wrong', slot['identity'], range(8))

    def test_approval_protocol_source_and_plan_tamper(self):
        _, digest = self.save()
        self.fixture(self.path, self.path.read_text() + '\n')
        with self.assertRaises(ValueError):
            launch.validate_plan(self.path, digest)
        self.save()
        self.fixture(self.repo / self.script, '# changed\n')
        with self.assertRaises(ValueError):
            launch.validate_plan(self.path)
        self.plan['source_sha256'][self.script] = launch.sha256(self.repo / self.script)
        self.save()
        self.fixture(self.protocol, '{"changed": true}')
        with self.assertRaises(ValueError):
            launch.validate_plan(self.path)
        self.plan['protocol']['sha256'] = launch.sha256(self.protocol)
        self.plan['main_approved'] = False
        with self.assertRaises(ValueError):
            self.save()

    def test_historic_outputs_shell_and_cuda_indices_are_rejected(self):
        for value in ('code/V4/runs/R120_multidomain_selector', 'code/V4/runs/R121_x/../../R84_x', '../escape'):
            with self.assertRaises(ValueError):
                launch.owned_root(value, 'R121')
        for argv in (['bash', '-c', 'true'], ['python', '-c', 'print(1)']):
            with self.assertRaises(ValueError):
                launch.runner_source(argv)
        self.plan['jobs'][0]['stages'][0]['argv'][-3] = 'cuda:1'
        with self.assertRaises(ValueError):
            self.save()

    def test_mutually_exclusive_original_inode_and_child_inheritance(self):
        path = self.repo / 'original_mutex'
        self.fixture(path, 'unchanged original\n')
        inode = path.stat().st_ino
        with launch.gpu_mutex(path, inode) as descriptor:
            with self.assertRaises(launch.Waiting):
                with launch.gpu_mutex(path, inode):
                    self.fail('duplicate lock')
            child = subprocess.run([sys.executable, '-B', '-c',
                'import os,sys,fcntl; fd=int(sys.argv[1]); assert os.fstat(fd).st_ino==int(sys.argv[2]); '
                'print("inherited")', str(descriptor), str(inode)], pass_fds=(descriptor,),
                capture_output=True, text=True, check=True)
            self.assertEqual(child.stdout.strip(), 'inherited')
            actual_stat = Path.stat
            def different_mount(target, *args, **kwargs):
                value = actual_stat(target, *args, **kwargs)
                return SimpleNamespace(st_ino=value.st_ino, st_dev=value.st_dev + 99)
            with mock.patch.object(Path, 'stat', different_mount):
                with self.assertRaises(launch.Waiting):
                    with launch.gpu_mutex(path, inode):
                        self.fail('same inode should still be locked despite mount device difference')
        self.assertEqual(path.read_text(), 'unchanged original\n')
        with self.assertRaises(ValueError):
            with launch.gpu_mutex(path, inode + 1):
                self.fail('wrong inode')

    def test_failure_logs_exit_code_and_consumes_attempt(self):
        job = self.plan['jobs'][0]
        first = job['stages'][0]
        first['argv'][-1] = '7'
        job['stages'].append(dict(id='never', kind='test', argv=list(first['argv'])))
        plan, digest = self.save()
        lock = self.repo / 'mutex'
        self.fixture(lock, '{}')
        receipt = dict(state='running', stages=[])
        with launch.gpu_mutex(lock, lock.stat().st_ino) as descriptor:
            code = launch.run_stages(self.path, digest, plan, job, descriptor, receipt)
            self.assertEqual(code, 7)
            with self.assertRaises(FileExistsError):
                launch.run_stages(self.path, digest, plan, job, descriptor, dict(state='running', stages=[]))
        directory = launch.job_dir(plan, job)
        attempt = launch.read(directory / 'attempt.json')
        self.assertEqual(attempt['state'], 'failed')
        self.assertEqual(attempt['exit_status'], 7)
        self.assertEqual(len(attempt['stages']), 1)
        self.assertIn('CPU fixture', (directory / 'train.log').read_text())
        self.assertFalse((directory / 'never.log').exists())
        another = dict(plan, name='another_plan_name')
        with launch.gpu_mutex(lock, lock.stat().st_ino) as descriptor:
            with self.assertRaises(FileExistsError):
                launch.run_stages(self.path, digest, another, job, descriptor, dict(state='running', stages=[]))
        self.assertFalse((launch.job_dir(another, job) / 'train.log').exists())

    def test_r120_release_matches_real_lib_format_without_publication(self):
        root = self.runs / 'R120_multidomain_selector'
        with self.assertRaises(launch.Waiting):
            launch.r120_ready(root)
        self.fixture(root / 'best.pt', 'frozen checkpoint')
        config = dict(inference_rule='one rule')
        self.fixture(root / 'protocol.json', json.dumps(config))
        digest = launch.sha256(root / 'best.pt')
        lock = dict(experiment='R120', checkpoint=str(root / 'best.pt'), checkpoint_sha256=digest,
            selected_update=44640, selected_without_test=True, per_problem_assembly=False,
            state_hash=hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            inference_rule='one rule')
        self.fixture(root / 'final_test_lock.json', json.dumps(lock))
        common = dict(complete=True, checkpoint_sha256=digest, selected_update=44640)
        self.fixture(root / 'test_complete.json', json.dumps(common))
        self.fixture(root / 'reports/report_complete.json', json.dumps(dict(common, lib_complete=True, complete24=True)))
        for split in ('val', 'test', 'TSPLIB', 'CVRPLIB'):
            path = root / 'evaluation' / split / 'selected_best/metrics.json'
            self.fixture(path, json.dumps(dict(identity=dict(checkpoint_sha256=digest, successful_updates=44640))))
        self.assertFalse(launch.r120_ready(root)['publication_required'])
        self.fixture(root / 'evaluation/TSPLIB/selected_best/metrics.json', json.dumps(dict(identity=dict(checkpoint_sha256='other'))))
        with self.assertRaises(ValueError):
            launch.r120_ready(root)

    def test_gpu_probe_is_one_snapshot_and_rejects_even_small_process(self):
        slot = launch.SLOTS['gpu05_0']
        xml = ('<nvidia_smi_log><gpu><uuid>' + slot['gpu_uuid'] + '</uuid>'
            '<product_name>NVIDIA RTX 4090</product_name><fb_memory_usage><used>4 MiB</used></fb_memory_usage>'
            '<utilization><gpu_util>0 %</gpu_util></utilization><processes>{}</processes></gpu></nvidia_smi_log>')
        with mock.patch('code.V4.r81_queue.capture', return_value=xml.format('')) as capture:
            self.assertEqual(launch.gpu_probe(slot)['memory_mib'], 4)
            capture.assert_called_once()
        with mock.patch('code.V4.r81_queue.capture', return_value=xml.format('<process_info><pid>1</pid><process_name>other</process_name></process_info>')):
            with self.assertRaises(launch.Waiting):
                launch.gpu_probe(slot)

    def test_complete_local_cache_has_no_copy_or_recovery_dependency(self):
        with mock.patch.object(launch, 'cache_inventory', return_value=dict(complete=True, missing=[])), \
                mock.patch.object(launch.subprocess, 'run') as run:
            receipt = launch.ensure_cache(self.plan, launch.SLOTS['gpu03_0'])
            self.assertFalse(receipt['copied'])
            run.assert_not_called()
        with mock.patch.object(launch, 'cache_inventory', return_value=dict(complete=False, missing=['test/TSP/metadata.json'])), \
                mock.patch.object(launch.subprocess, 'run') as run:
            with self.assertRaises(launch.Waiting):
                launch.ensure_cache(self.plan, launch.SLOTS['gpu03_0'])
            run.assert_not_called()

    def test_parent_test_publication_marker_is_required_and_reused(self):
        slot = launch.SLOTS['gpu03_0']
        with self.assertRaises(launch.Waiting):
            launch.frozen_test_source(self.plan, slot)
        published = launch.R120 / 'existing_test_cache'
        proof = dict(complete=True, preprocessing=False, solver_calls=0, source_container=slot['identity'][:12])
        self.fixture(published / 'complete.json', json.dumps(proof))
        self.assertEqual(launch.frozen_test_source(self.plan, slot), published / 'test')
        self.fixture(launch.SOURCE / 'test/TSP/metadata.json', '{}')
        self.assertEqual(launch.frozen_test_source(self.plan, slot), launch.SOURCE / 'test')

    def test_dispatch_repeats_only_proven_pre_attempt_wait_never_stage75(self):
        plan, digest = self.save()
        counts = {}
        def spawn(argv, **_kwargs):
            name = argv[argv.index('--job') + 1]
            token = argv[argv.index('--token') + 1]
            job = launch.find_job(plan, name)
            directory = launch.job_dir(plan, job)
            counts[name] = counts.get(name, 0) + 1
            waiting = name == 'R121' and counts[name] == 1
            code = 75 if name == 'R121' else 0
            if not waiting:
                launch.write(directory / 'attempt.json', dict(state='failed' if code else 'complete',
                    plan_sha256=digest, exit_status=code), exclusive=True)
            launch.write(directory / 'transports' / (token + '.result.json'),
                dict(state='waiting' if waiting else ('failed' if code else 'complete'),
                     token=token, plan_sha256=digest, exit_status=code), exclusive=True)
            return SimpleNamespace(pid=123, poll=lambda: code, wait=lambda: code)
        args = SimpleNamespace(plan=self.path, wait=True, poll_seconds=1)
        with mock.patch.object(launch, 'r120_ready', return_value={}), \
                mock.patch.object(launch.subprocess, 'Popen', side_effect=spawn), \
                mock.patch.object(launch.time, 'sleep'):
            self.assertEqual(launch.dispatch(args, plan, digest), 1)
            self.assertEqual(counts, dict(R121=2, R122=1, R123=1, R124=1))
            with self.assertRaises(FileExistsError):
                launch.dispatch(args, plan, digest)

    def test_cache_inventory_checks_test_and_array_truncation(self):
        import numpy as np
        tasks = ['TSP'] + [f'TASK{i}' for i in range(17)]
        for split, rows in (('train', 81000), ('val', 4500), ('test', 4500)):
            for task in tasks:
                directory = launch.CACHE / split / task
                metadata = dict(complete=True, problem=task, split=split, rows=rows,
                    scenario_id='new90k_seed2_B_filtered_MOEL_as_reld_mt_v1',
                    arrays=dict(n=dict(file='n.npy', shape=[rows], dtype='uint8')))
                self.fixture(directory / 'metadata.json', json.dumps(metadata))
                np.save(directory / 'n.npy', np.zeros(rows, dtype=np.uint8))
                if split != 'test':
                    self.fixture(launch.SOURCE / split / task / 'metadata.json', json.dumps(metadata))
        self.assertTrue(launch.cache_inventory(launch.CACHE)['complete'])
        path = launch.CACHE / 'test/TSP/n.npy'
        with path.open('r+b') as stream:
            stream.truncate(path.stat().st_size - 1)
        with self.assertRaises(ValueError):
            launch.cache_inventory(launch.CACHE)

    def test_describe_never_touches_resources(self):
        with mock.patch.object(launch, 'r120_ready', side_effect=AssertionError('resource access')), \
                mock.patch.object(launch.subprocess, 'Popen', side_effect=AssertionError('launch')), \
                mock.patch('builtins.print'):
            self.assertEqual(launch.main(['describe', '--plan', str(self.path)]), 0)

    def portable_fixture(self):
        import numpy as np
        left, right = self.repo / 'local_record', self.repo / 'reference_record'
        arrays = dict(costs=np.arange(30, dtype=np.float64).reshape(10, 3),
                      winner=np.zeros(10, dtype=np.int64),
                      matrix=np.arange(120, dtype=np.float32).reshape(10, 3, 4))
        metadata = dict(rows=10, split='test', pool=['a', 'b', 'c'], data_root='unchanged',
            source_files=[dict(path='/frozen/source', size_bytes=12, mtime_ns=123)],
            arrays={key: dict(file=key + '.npy', shape=list(value.shape), dtype=str(value.dtype))
                    for key, value in arrays.items()})
        for directory, arm in ((left, 'D_attention_focal'), (right, 'C_focal')):
            root = self.runs / 'R84_new90k_attention' / arm
            self.fixture(root / 'final_test_lock.json', '{}')
            self.fixture(directory / 'metadata.json', json.dumps(dict(metadata,
                directory=str(directory), cache_root=str(directory.parent), test_lock_root=str(root),
                created_utc=arm, completed_utc=arm, seconds=1 if directory == left else 2)))
            for key, value in arrays.items():
                np.save(directory / (key + '.npy'), value)
        return left / 'metadata.json', right / 'metadata.json'

    def test_portable_metadata_allows_only_six_fields_with_label_hash_and_input_samples(self):
        local, reference = self.portable_fixture()
        proof = launch.portable_cache_identity(local, reference)
        self.assertTrue(set(proof['changed_fields']) <= launch.PORTABLE_CACHE_FIELDS)
        self.assertEqual(set(proof['label_sha256']), {'costs', 'winner'})
        self.assertEqual(proof['sampled_input_arrays'], ['matrix'])
        metadata = launch.read(local)
        metadata['pool'] = ['b', 'a', 'c']
        local.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'semantic metadata'):
            launch.portable_cache_identity(local, reference)

    def test_portable_metadata_rejects_changed_native_cost_or_winner(self):
        import numpy as np
        for key in ('costs', 'winner'):
            local, reference = self.portable_fixture()
            path = local.parent / (key + '.npy')
            array = np.load(path)
            array.flat[-1] += 1
            np.save(path, array)
            with self.assertRaisesRegex(ValueError, 'full SHA256'):
                launch.portable_cache_identity(local, reference)

    def test_portable_metadata_rejects_changed_input_sample_and_missing_test_lock(self):
        import numpy as np
        local, reference = self.portable_fixture()
        path = local.parent / 'matrix.npy'
        array = np.load(path)
        array[5, 0, 0] += 1
        np.save(path, array)
        with self.assertRaisesRegex(ValueError, 'fixed-row'):
            launch.portable_cache_identity(local, reference)
        local, reference = self.portable_fixture()
        metadata = launch.read(local)
        metadata['test_lock_root'] = str(self.repo / 'no_lock')
        local.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'historical TEST access lock'):
            launch.portable_cache_identity(local, reference)


if __name__ == '__main__':
    unittest.main()
