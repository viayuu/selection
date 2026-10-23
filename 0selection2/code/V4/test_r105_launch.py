"""Bounded CPU administrative tests. No hardware probes or data preparation."""

from contextlib import redirect_stdout
import copy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import r105_launch as launch
from . import r105_wave as wave


def fixture(repo, arm=launch.ARMS[0]):
    root = repo / 'R105_conditional_representation_experts'
    science = dict(experiment='R105', locked=True, arms=list(launch.ARMS), seed=2, task_scope='full18',
        data_root=str(launch.DATA_ROOT), prepared_root=str(launch.PREPARED_ROOT),
        primary_cache=str(launch.CACHE_ROOT), rows_per_problem=dict(train=81000, val=4500, test=4500),
        batch_size=1024, logical_global_batch=1024, max_updates=57600, warmup_updates=4320,
        eval_every=1440, train_eval_every=7200, large_micro_cap=128, other_micro_cap=512,
        eval_batch_size=128, current_query_labels_in_forward=False, external_pretraining=False)
    launch.base.save(root / 'protocol_lock.json', science)
    with patch.object(launch, 'MODEL_ROOT', root):
        spec = wave.make_spec(arm)
    path = repo / (launch.run_id(arm) + '.json')
    launch.base.save(path, spec)
    return root, path, spec, science


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='r105-launch-cpu-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)

    def test_private_adapters_preserve_historical_and_r102_resources(self):
        from . import r98_launch, r104_launch, r84_launch, r81_queue, r102_launch
        self.assertIsNot(launch.adapter, r98_launch)
        self.assertIsNot(launch.base, r84_launch)
        self.assertIsNot(launch.base.transport, r81_queue)
        self.assertIsNot(launch.predecessor_goal, importlib.import_module('code.V4.r104_goal'))
        self.assertEqual(launch.base.transport.ROOT, r81_queue.ROOT)
        self.assertEqual(launch.ROOT, r98_launch.ROOT)
        self.assertEqual(r98_launch.TOKEN_ENV, 'R98_DISPATCH_TOKEN')
        self.assertEqual(r104_launch.TOKEN_ENV, 'R104_DISPATCH_TOKEN')
        self.assertEqual(r104_launch.ARMS, ('A_width256', 'B_depth8'))
        self.assertEqual(r102_launch.ARM_SLOTS, dict(A='4090_0', B='4090_1'))
        self.assertEqual(set(launch.SLOTS), {'3090_0', '3090_1'})
        uuids = ('GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50',
                 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d')
        with patch.object(launch.base.subprocess, 'run', side_effect=AssertionError('No processes')):
            for slot, uuid in zip(launch.ARM_SLOTS.values(), uuids):
                item = launch.SLOTS[slot]
                self.assertEqual((item['job'], item['node'], item['uuid'], item['cap']), (1465, 'gpu03', uuid, 128))
                command = launch.container_command('worker', slot, '/tmp/spec', 'fixture-only')
                self.assertIn('b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c', command)
                self.assertIn(str(launch.WRAPPER), command)
                self.assertNotIn('ssh', command)
                step = launch.base.srun_command('entry', slot, '/tmp/spec', cpus=4, memory='9G')
                for flag in ('--jobid=1465', '--nodelist=gpu03', '--gres=none', '--cpus-per-task=4', '--mem=9G'):
                    self.assertIn(flag, step)
        self.assertFalse(set(launch.SLOT_CPUS['3090_0']) & set(launch.SLOT_CPUS['3090_1']))
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=list(range(72)))):
            self.assertEqual(launch.node_resources('3090_1')['allowed_cpus'], [16, 17, 64, 65])
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=[12, 13])):
            with self.assertRaisesRegex(RuntimeError, 'affinity'):
                launch.node_resources('3090_0')

    def test_exact_commands_and_main_lock_without_invented_architecture_keys(self):
        root, path, spec, science = fixture(self.repo)
        with patch.object(launch, 'MODEL_ROOT', root):
            self.assertEqual(launch.load_spec(path), spec)
            commands = launch.base.commands(spec, '3090_0')
            self.assertEqual(tuple(s['name'] for s in commands), launch.STAGES)
            for stage, positional in zip(commands[:3], ('train', 'test', 'timing')):
                args = stage['argv']
                self.assertEqual(args[args.index('-m') + 1:args.index('-m') + 3], ['code.V4.r105_experiment', positional])
                self.assertEqual(args[args.index('--arm') + 1], 'A_static_gate')
                self.assertEqual(args[args.index('--large-micro-cap') + 1], '128')
                self.assertEqual(args[args.index('--eval-batch-size') + 1], '128')
                self.assertEqual(args[args.index('--cache-root') + 1], str(launch.CACHE_ROOT))
                self.assertNotIn('--feature-root', args)
            self.assertEqual(commands[-2]['argv'][-2:], ['--lib-batch-size', '2'])
            self.assertEqual(commands[-1]['argv'][-2:], ['--device', 'cpu'])
            for changes in (dict(slots=['4090_0']), dict(cpus=[0, 1, 2, 3]), dict(memory_gib=10),
                            dict(teacher_root='/tmp/teacher'), dict(stages=spec['stages'][1:])):
                launch.base.save(path, dict(spec, **changes))
                with self.assertRaises(ValueError):
                    launch.load_spec(path)
            launch.base.save(path, spec)
            for changes in (dict(other_micro_cap=1024), dict(primary_cache='/tmp/new'), dict(logical_global_batch=128)):
                launch.base.save(root / 'protocol_lock.json', dict(science, **changes))
                with self.assertRaises(ValueError):
                    launch.load_spec(path)

    def test_actual_main_lock_and_real_stage_cli_parsers_cpu_only(self):
        files = [launch.REPO / 'code/V4' / name for name in ('r105_model.py', 'r105_experiment.py', 'r105_reporting.py')]
        if not all(path.is_file() for path in files):
            self.skipTest('Model/runner/report sources pending; not launch ready')
        experiment = importlib.import_module('code.V4.r105_experiment')
        reporting = importlib.import_module('code.V4.r105_reporting')
        science = launch.base.read(launch.MODEL_ROOT / 'protocol_lock.json')
        self.assertTrue(all(science.get(k) == v for k, v in experiment.required_lock_fields().items()))
        with patch.object(launch.base.subprocess, 'run', side_effect=AssertionError('Parser only')):
            for arm in launch.ARMS:
                spec = wave.make_spec(arm)
                launch.locked_protocol(spec['science_protocol'], spec)
                for stage in launch.base.commands(spec, launch.ARM_SLOTS[arm]):
                    argv = stage['argv']
                    module, args = argv[argv.index('-m') + 1], argv[argv.index('-m') + 2:]
                    parsed = (reporting if module.endswith('_reporting') else experiment).parser().parse_args(args)
                    self.assertEqual(parsed.arm, arm)
                    self.assertEqual(str(parsed.root), str(launch.MODEL_ROOT))
                    self.assertFalse(getattr(parsed, 'resume', False))
            reporting.parser().parse_args(['compare', '--root', str(launch.MODEL_ROOT), '--device', 'cpu'])

    def test_future_spec_allows_pending_sources_but_prepare_does_not(self):
        root, path, spec, _ = fixture(self.repo)
        pending = self.repo / 'pending_model.py'
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch, 'declared_sources', return_value=[pending]), \
                patch.object(launch.adapter, 'prepare') as prepare:
            future = wave.make_spec(launch.ARMS[0])
            launch.base.save(path, future)
            self.assertIn(str(pending), future['source_files'])
            with self.assertRaisesRegex(RuntimeError, 'pending; not launch ready'):
                launch.prepare(path, 'main proof', True)
        prepare.assert_not_called()
        self.assertFalse((self.repo / 'ready.json').exists())

    def test_explicit_main_authorization_and_one_shot_spec(self):
        for evidence, tested in ((None, True), (' ', True), ('main', False)):
            for action in (launch.prepare, launch.start, launch.queue):
                with self.assertRaises(ValueError):
                    action('/absent', evidence, tested)
        root, _, _, _ = fixture(self.repo)
        dispatch = self.repo / 'dispatch'
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch, 'ROOT', dispatch):
            path = wave.write_spec(launch.ARMS[0])
            saved = path.read_bytes()
            self.assertEqual(wave.write_spec(launch.ARMS[0]), path)
            self.assertEqual(path.read_bytes(), saved)
            launch.base.save(path, dict(unrelated=True))
            with self.assertRaisesRegex(RuntimeError, 'Preserve existing'):
                wave.write_spec(launch.ARMS[0])

    def test_exact_source_set_freeze_and_static_import_closure(self):
        root, path, spec, _ = fixture(self.repo)
        dispatch = self.repo / 'dispatch'
        output = dispatch / spec['run_id']
        source = self.repo / 'tiny.py'
        source.write_text('VALUE = 1\n')
        plan = dict(spec=spec, main_thread_ready=True, evidence='main CPU proof',
            primary_cache_reuse_only=True, cache_copy=False, cache_preprocessing=False,
            automatic_retry=False, source_sha256={str(source): launch.base.sha(source)})
        launch.base.save(output / 'plan.json', plan)
        launch.base.save(output / 'ready.json', dict(main_thread_ready=True, runner_tests_passed=True,
            source_freeze_ready=True, evidence='main CPU proof', plan_sha256=launch.base.sha(output / 'plan.json')))
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch.base, 'ROOT', dispatch), \
                patch.object(launch, 'source_files', return_value=[str(source)]):
            launch.check_ready(path)
            (self.repo / 'r105_unrelated_future.py').write_text('UNRELATED = 1\n')
            launch.check_ready(path)
            source.write_text('VALUE = 2\n')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                launch.check_ready(path)
        code = self.repo / 'code/V4'
        code.mkdir(parents=True)
        entry, dep, test = (code / n for n in ('r105_entry.py', 'dependency.py', 'test_exact.py'))
        entry.write_text('from .dependency import VALUE\n')
        dep.write_text('VALUE = 1\n')
        test.write_text('from .unrelated import VALUE\n')
        (code / 'unrelated.py').write_text('VALUE = 2\n')
        with patch.object(launch, 'declared_sources', return_value=[entry, test]), \
                patch.object(launch.adapter, 'REPO', self.repo), \
                patch.object(Path, 'glob', side_effect=AssertionError('No source globs')):
            sources = launch.source_files(dict(source_files=[]))
            self.assertEqual(set(sources), {str(p) for p in (entry, dep, test)})
            with self.assertRaisesRegex(ValueError, 'exact named'):
                launch.source_files(dict(source_files=[str(code / 'unrelated.py')]))

    def test_predecessor_original_r104_only_independent_no_recovery_or_null_val_bypass(self):
        from .test_r104_goal import fixture as prior_fixture
        root, dispatch, _ = prior_fixture(self.repo)
        with patch.object(launch, 'REPO', self.repo), patch.object(launch, 'ROOT', dispatch):
            self.assertTrue(launch.predecessor_gate(launch.ARMS[0])['complete'])
            failed = dispatch / 'R104_B_depth8/worker.json'
            launch.base.save(failed, dict(state='failed', exit_status=1, errors='timing mismatch'))
            self.assertTrue(launch.predecessor_gate(launch.ARMS[0])['complete'])
            self.assertFalse(launch.predecessor_gate(launch.ARMS[1])['complete'])
            path = root / 'A_width256/final_metrics.json'
            original = launch.base.read(path)
            record = copy.deepcopy(original)
            record['val']['identity']['checkpoint_sha256'] = None
            launch.base.save(path, record)
            saved = path.read_bytes()
            self.assertFalse(launch.predecessor_gate(launch.ARMS[0])['complete'])
            self.assertEqual(path.read_bytes(), saved)
            launch.base.save(path, original)
            controller = dispatch / 'R104_A_width256/controller.json'
            launch.base.save(controller, dict(state='failed', exit_status=1, error='original failure'))
            launch.base.save(root / 'preparation/postprocess_recovery/A_width256/recovery_complete.json',
                             dict(schema='r99_postprocess_recovery_v1', complete=True))
            self.assertFalse(launch.predecessor_gate(launch.ARMS[0])['complete'])

    def test_terminal_attempts_source_change_and_predecessor_block(self):
        _, _, spec, _ = fixture(self.repo)
        output = self.repo / 'queue'
        output.mkdir()
        with patch.object(launch, 'check_ready', return_value=(spec, dict(evidence='ready'))), \
                patch.object(launch.base, 'receipts', return_value=output), \
                patch.object(launch, 'predecessor_gate', return_value=dict(complete=False, errors=[])), \
                patch.object(launch, 'original_start') as actual:
            with self.assertRaisesRegex(RuntimeError, 'original R104'):
                launch.start('/spec', 'ready', True)
            actual.assert_not_called()
        with patch.object(launch, 'check_ready', side_effect=[(spec, dict(evidence='ready')), RuntimeError('source changed')]), \
                patch.object(launch.base, 'receipts', return_value=output), patch.object(launch, 'start') as start:
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                launch.queue('/spec', 'ready', True, 1)
            saved = (output / 'queue_receipt.json').read_bytes()
        with patch.object(launch, 'check_ready', return_value=(spec, dict(evidence='ready'))), \
                patch.object(launch.base, 'receipts', return_value=output):
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                launch.queue('/spec', 'ready', True)
        self.assertEqual(saved, (output / 'queue_receipt.json').read_bytes())
        start.assert_not_called()

    def test_cache_validation_has_no_repair_copy_or_rebuild(self):
        root = self.repo / 'cache'
        root.mkdir()
        stamp = root / '.r84_cache_identity.json'
        manifest = dict(identity_sha256='fixed', files=[dict(path='small.bin', bytes=3)])
        launch.base.save(stamp, dict(identity_sha256='fixed'))
        (root / 'small.bin').write_bytes(b'abc')
        self.assertFalse(launch.stage_cache(dict(cache_root=str(root)), manifest)['copied'])
        (root / 'small.bin').unlink()
        with self.assertRaisesRegex(RuntimeError, 'no repair or copy'):
            launch.stage_cache(dict(cache_root=str(root)), manifest)
        self.assertFalse((root / 'small.bin').exists())

    def test_admin_imports_cpu_only_no_subprocess_or_cuda_probe(self):
        program = """import subprocess, sys
import code.V4.r105_launch
def forbidden(*a, **k): raise AssertionError('No subprocess during imports')
subprocess.run = subprocess.check_output = subprocess.Popen = forbidden
import code.V4.r105_wave, code.V4.r105_goal
import code.V4.test_r105_launch, code.V4.test_r105_goal
assert 'torch' not in sys.modules
print('CPU administrative imports only')
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-B', '-c', program], cwd=launch.REPO,
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CPU administrative imports only', result.stdout)

    def test_admin_cli_dispatch_and_wrapper_only_r105(self):
        with patch.object(launch, 'predecessor_gate', return_value=dict(complete=False)) as gate, redirect_stdout(io.StringIO()):
            self.assertEqual(launch.main(['predecessor', '--arm', launch.ARMS[0]]), 0)
        gate.assert_called_once_with(launch.ARMS[0])
        text = launch.WRAPPER.read_text()
        self.assertLess(text.index('source '), text.index('conda activate easynco'))
        self.assertIn('exec python -B -u -m code.V4.r105_launch', text)
        self.assertNotIn('r104_launch', text)


if __name__ == '__main__':
    unittest.main()
