"""Bounded fake CPU dispatch tests; no hardware, real specs, cache or launches."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r108_launch as launch
from . import r108_wave as wave


def fixture(repo, arm=launch.ARMS[0], **caps):
    root = repo / 'R108_conditional_density_ratio'
    science = dict(experiment='R108', locked=True, arms=list(launch.ARMS), seed=2, task_scope='full18',
        data_root=str(launch.DATA_ROOT), prepared_root=str(launch.PREPARED_ROOT),
        primary_cache=str(launch.CACHE_ROOT), rows_per_problem=dict(train=81000, val=4500, test=4500),
        batch_size=1024, logical_global_batch=1024, max_updates=57600, warmup_updates=4320,
        eval_every=1440, train_eval_every=7200, optimizer='AdamW', peak_lr=1e-4,
        model_role_arm='B_execution_anchor', role_feature_slot=3,
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        external_pretraining=False, end_lr=1e-6, dropout=.1,
        loss=dict(ce_or_density_ratio=1., comparison=0., risk=0., kl=0., dropout_passes=1),
        trainable_parameters=dict(original=launch.TRAINABLE_PARAMETERS,
            **dict.fromkeys(launch.ARMS, launch.TRAINABLE_PARAMETERS)),
        identical_complete_fresh_parameters=True, original_fresh_trainable_sha256=launch.ORIGINAL_FRESH_SHA256,
        priors_path=str(root / 'prepared/priors.json'), **dict(launch.DRAFT_CAPS, **caps))
    launch.base.save(root / 'prepared/priors.json', dict(synthetic=True, split='train'))
    science['priors_sha256'] = launch.base.sha(root / 'prepared/priors.json')
    launch.base.save(root / 'protocol_lock.json', science)
    with patch.object(launch, 'MODEL_ROOT', root):
        spec = wave.make_spec(arm, final=True)
    path = repo / (launch.run_id(arm) + '.json')
    launch.base.save(path, spec)
    return root, path, spec, science


def fake_parser(reporter=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('lib', 'report', 'compare') if reporter else ('train', 'test', 'timing'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--arm', choices=launch.ARMS)
    parser.add_argument('--device')
    if reporter:
        parser.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    else:
        for key in ('data-root', 'prepared-root', 'cache-root', 'protocol-lock'):
            parser.add_argument('--' + key, type=Path)
        parser.add_argument('--gpu-type', choices=('4090',))
        parser.add_argument('--large-micro-cap', type=int)
        parser.add_argument('--eval-batch-size', type=int)
        parser.add_argument('--wandb-mode', choices=('offline',))
        parser.add_argument('--resume', action='store_true')
    return parser


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='r108-launch-fake-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)

    def test_private_resources_leave_previous_versions_and_r106_untouched(self):
        from . import r98_launch, r107_launch, r106_launch, r84_launch, r81_queue
        self.assertIsNot(launch.adapter, r98_launch)
        self.assertIsNot(launch.base, r84_launch)
        self.assertIsNot(launch.base.transport, r81_queue)
        self.assertIsNot(launch.predecessor_goal, importlib.import_module('code.V4.r106_goal'))
        self.assertEqual(launch.base.transport.ROOT, r81_queue.ROOT)
        self.assertEqual(launch.ROOT, launch.MODEL_ROOT / 'preparation/dispatch')
        self.assertEqual(r98_launch.TOKEN_ENV, 'R98_DISPATCH_TOKEN')
        self.assertEqual(r107_launch.TOKEN_ENV, 'R107_DISPATCH_TOKEN')
        self.assertEqual(r106_launch.ARM_SLOTS, dict(A_frozen='4090_0', B_finetune='4090_1'))
        self.assertEqual(set(launch.SLOTS), {'4090_0', '4090_1'})
        uuids = ('GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be',
                 'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d')
        with patch.object(launch.base.subprocess, 'run', side_effect=AssertionError('No processes')):
            for slot, uuid in zip(launch.ARM_SLOTS.values(), uuids):
                item = launch.SLOTS[slot]
                self.assertEqual((item['job'], item['node'], item['uuid']), (1475, 'gpu05', uuid))
                self.assertEqual(item['cap'], 512)
                command = launch.container_command('worker', slot, '/tmp/spec', 'fake-only')
                self.assertIn('76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6', command)
                self.assertIn(str(launch.WRAPPER), command)
                self.assertNotIn('ssh', command)
                step = launch.base.srun_command('entry', slot, '/tmp/spec', cpus=4, memory='9G')
                for flag in ('--jobid=1475', '--nodelist=gpu05', '--gres=none', '--cpus-per-task=4', '--mem=9G'):
                    self.assertIn(flag, step)
        self.assertFalse(set(launch.SLOT_CPUS['4090_0']) & set(launch.SLOT_CPUS['4090_1']))
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=list(range(96)))):
            self.assertEqual(launch.node_resources('4090_1')['allowed_cpus'], [4, 5, 72, 73])
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=[0, 1])):
            with self.assertRaisesRegex(RuntimeError, 'affinity'):
                launch.node_resources('4090_0')

    def test_draft_api_and_default_cli_do_not_read_lock_or_write(self):
        root, dispatch = self.repo / 'absent-model', self.repo / 'absent-dispatch'
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch, 'ROOT', dispatch), \
                patch.object(launch.base, 'save', side_effect=AssertionError('No writes')), \
                patch.object(launch.base, 'read', side_effect=AssertionError('No lock reads')):
            spec = wave.make_spec(launch.ARMS[0])
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(wave.main([launch.ARMS[1]]), 0)
            self.assertTrue(json.loads(output.getvalue())['spec_draft'])
        self.assertTrue(spec['spec_draft'])
        self.assertFalse(spec['main_thread_ready'])
        self.assertFalse(root.exists())
        self.assertFalse(dispatch.exists())
        self.assertIn('r108_memory_witness.py', launch.SOURCE_NAMES)
        self.assertIn('run_v4_r108_witness.sh', launch.SOURCE_NAMES)
        self.assertIn(launch.MODEL_ROOT / 'prepared/priors.json', launch.declared_sources())
        self.assertFalse(any(p.suffix in ('.npy', '.npz') for p in launch.declared_sources()))
        self.assertNotIn('run_v4_r108_memory_witness.sh', launch.SOURCE_NAMES)
        self.assertNotIn('routing_diagnostic', json.dumps(spec))

    def test_formal_spec_requires_main_lock_and_never_overwrites_or_marks_ready(self):
        with patch.object(launch, 'MODEL_ROOT', self.repo / 'missing'):
            with self.assertRaisesRegex(ValueError, 'draft is not ready'):
                wave.make_spec(launch.ARMS[0], final=True)
        root, path, spec, _ = fixture(self.repo)
        dispatch = self.repo / 'dispatch'
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch, 'ROOT', dispatch):
            final = wave.write_spec(launch.ARMS[0])
            saved = final.read_bytes()
            self.assertEqual(wave.write_spec(launch.ARMS[0]), final)
            self.assertEqual(final.read_bytes(), saved)
            self.assertFalse(launch.base.read(final)['main_thread_ready'])
            launch.base.save(final, dict(unrelated=True))
            with self.assertRaisesRegex(RuntimeError, 'Preserve existing'):
                wave.write_spec(launch.ARMS[0])
            launch.base.save(path, dict(spec, spec_draft=True))
            with self.assertRaisesRegex(ValueError, 'Only a final spec'):
                launch.load_spec(path)
        self.assertFalse((dispatch / spec['run_id'] / 'ready.json').exists())

    def test_common_lock_caps_and_exact_stage_argv(self):
        root, path, spec, science = fixture(self.repo)
        with patch.object(launch, 'MODEL_ROOT', root):
            self.assertEqual(launch.load_spec(path), spec)
            for arm in launch.ARMS:
                stages = launch.base.commands(wave.make_spec(arm, final=True), launch.ARM_SLOTS[arm])
                self.assertEqual(tuple(s['name'] for s in stages), launch.STAGES)
                for stage, command in zip(stages[:3], ('train', 'test', 'timing')):
                    args = stage['argv']
                    self.assertEqual(args[args.index('-m') + 1:args.index('-m') + 3], ['code.V4.r108_experiment', command])
                    self.assertEqual(args[args.index('--large-micro-cap') + 1], '512')
                    self.assertEqual(args[args.index('--eval-batch-size') + 1], '256')
                    self.assertNotIn('--priors-path', args)
                    self.assertEqual(args[args.index('--cache-root') + 1], str(launch.CACHE_ROOT))
                self.assertEqual(stages[-2]['argv'][-2:], ['--lib-batch-size', '2'])
                self.assertEqual(stages[-1]['argv'][-2:], ['--device', 'cpu'])
            for changes in (dict(slots=['3090_0']), dict(cpus=[12, 13, 14, 15]), dict(memory_gib=10),
                            dict(feature_root='/tmp/features'), dict(stages=spec['stages'][1:])):
                launch.base.save(path, dict(spec, **changes))
                with self.assertRaises(ValueError):
                    launch.load_spec(path)
            launch.base.save(path, spec)
            for changes in (dict(primary_cache='/tmp/new'), dict(logical_global_batch=128),
                            dict(peak_lr=3e-4), dict(end_lr=0.), dict(dropout=0.),
                            dict(loss=dict(ce=1., comparison=1.)), dict(priors_sha256='wrong'),
                            dict(large_micro_cap=True), dict(other_micro_cap=2048)):
                launch.base.save(root / 'protocol_lock.json', dict(science, **changes))
                with self.assertRaises(ValueError):
                    launch.load_spec(path)
            launch.base.save(root / 'protocol_lock.json', dict(science, large_micro_cap=64))
            for arm in launch.ARMS:
                with self.assertRaisesRegex(ValueError, 'confirmed shared witness caps'):
                    wave.make_spec(arm, final=True)
            with self.assertRaises(ValueError):
                launch.load_spec(path)

    def test_fake_stage_parsers_and_main_lock_compare_check_no_runner_import(self):
        _, _, spec, _ = fixture(self.repo)
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            if argv[-2] in ('code.V4.r108_experiment', 'code.V4.r108_reporting'):
                parsed = fake_parser(argv[-2].endswith('_reporting')).parse_args(json.loads(argv[-1]))
                return SimpleNamespace(returncode=0, stdout=json.dumps(vars(parsed), default=str), stderr='')
            self.assertIn('e.protocol_lock(a)', argv[-2])
            self.assertIn('"compare"', argv[-2])
            self.assertFalse(fake_parser().parse_args(json.loads(argv[-1])).resume)
            return SimpleNamespace(returncode=0, stdout='fake main-lock check', stderr='')

        with patch.object(launch.base.subprocess, 'run', side_effect=fake_run):
            parsed = launch.parse_commands(spec)
        self.assertEqual([row['stage'] for row in parsed], list(launch.STAGES))
        self.assertEqual(len(calls), 6)
        with patch.object(launch.base.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='pending model')):
            with self.assertRaisesRegex(RuntimeError, 'CLI is not ready'):
                launch.parse_commands(spec)

    def test_pending_sources_cannot_prepare_and_exact_freeze_detects_edits(self):
        root, path, spec, _ = fixture(self.repo)
        pending = self.repo / 'pending_model.py'
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch, 'declared_sources', return_value=[pending]), \
                patch.object(launch.adapter, 'prepare') as prepare:
            launch.base.save(path, wave.make_spec(launch.ARMS[0], final=True))
            with self.assertRaisesRegex(RuntimeError, 'pending; not main-ready'):
                launch.prepare(path, 'main proof', True)
        prepare.assert_not_called()
        source, dispatch = self.repo / 'tiny.py', self.repo / 'dispatch'
        source.write_text('VALUE = 1\n')
        launch.base.save(path, spec)
        output = dispatch / spec['run_id']
        plan = dict(spec=spec, main_thread_ready=True, evidence='main proof', primary_cache_reuse_only=True,
            cache_copy=False, cache_preprocessing=False, automatic_retry=False,
            source_sha256={str(source): launch.base.sha(source)})
        launch.base.save(output / 'plan.json', plan)
        launch.base.save(output / 'ready.json', dict(main_thread_ready=True, runner_tests_passed=True,
            source_freeze_ready=True, evidence='main proof', plan_sha256=launch.base.sha(output / 'plan.json')))
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch.base, 'ROOT', dispatch), \
                patch.object(launch, 'source_files', return_value=[str(source)]):
            launch.check_ready(path)
            (self.repo / 'unrelated.py').write_text('UNRELATED = 1\n')
            launch.check_ready(path)
            source.write_text('VALUE = 2\n')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                launch.check_ready(path)

    def test_named_source_closure_no_test_expansion_or_future_globs(self):
        code = self.repo / 'code/V4'
        code.mkdir(parents=True)
        entry, dep, test = (code / n for n in ('r108_entry.py', 'dependency.py', 'test_exact.py'))
        entry.write_text('from .dependency import VALUE\n')
        dep.write_text('VALUE = 1\n')
        test.write_text('from .unrelated import VALUE\n')
        (code / 'unrelated.py').write_text('VALUE = 2\n')
        with patch.object(launch, 'declared_sources', return_value=[entry, test]), \
                patch.object(launch.adapter, 'REPO', self.repo), \
                patch.object(Path, 'glob', side_effect=AssertionError('No source globs')):
            self.assertEqual(set(launch.source_files(dict(source_files=[]))), {str(p) for p in (entry, dep, test)})
            with self.assertRaisesRegex(ValueError, 'exact named'):
                launch.source_files(dict(source_files=[str(code / 'unrelated.py')]))

    def test_predecessor_assigned_r106_pipeline_and_report_not_github_or_other_arm(self):
        root = self.repo / 'R106_native_encoder_adaptation'
        dispatch = root / 'preparation/dispatch'
        current = dict(errors=[dict(path=str(root / 'github_publication.json'), error='Git only')],
            invalid_receipts=[], arms=[dict(arm=a, checks=dict(full_TRAIN=True, PDF=True))
                for a in launch.PREDECESSORS.values()])
        for arm in launch.ARMS:
            prior = launch.PREDECESSORS[arm]
            identifier = 'R106_' + prior
            launch.base.save(dispatch / 'specs' / (identifier + '.json'),
                dict(run_id=identifier, run_dir=str(root / prior), slots=[launch.ARM_SLOTS[arm]],
                    stages=[dict(name=n, device='cpu' if n == 'PDF' else 'gpu') for n in launch.STAGES]))
            for name, stages in (('worker.json', ['train', 'TEST', 'timing', 'LIB']),
                                 ('controller.json', ['allocation_GPU_pipeline', 'PDF'])):
                launch.base.save(dispatch / identifier / name, dict(state='complete', exit_status=0,
                    stages=[dict(name=n, state='complete', exit_status=0) for n in stages]))
        with patch.object(launch, 'PREDECESSOR_ROOT', root), \
                patch.object(launch.predecessor_goal, 'status', return_value=current) as status:
            self.assertTrue(launch.predecessor_gate(launch.ARMS[0])['complete'])
            status.assert_called_with('R106', root, dispatch, persist=False)
            self.assertTrue(launch.predecessor_gate(launch.ARMS[1])['complete'])
            launch.base.save(dispatch / 'R106_B_finetune/worker.json', dict(state='failed', exit_status=1, error='other arm'))
            self.assertTrue(launch.predecessor_gate(launch.ARMS[0])['complete'])
            self.assertFalse(launch.predecessor_gate(launch.ARMS[1])['complete'])
            current['arms'][0]['checks']['PDF'] = False
            self.assertFalse(launch.predecessor_gate(launch.ARMS[0])['complete'])
            current['arms'][0]['checks']['PDF'] = True
            current['invalid_receipts'].append(dict(path=str(root / 'A_frozen/protocol.json'), error='invalid'))
            self.assertFalse(launch.predecessor_gate(launch.ARMS[0])['complete'])

    def test_missing_predecessor_or_authorization_blocks_all_launch_paths(self):
        for evidence, tested in ((None, True), (' ', True), ('main', False)):
            for action in (launch.prepare, launch.start, launch.queue):
                with self.assertRaises(ValueError):
                    action('/absent', evidence, tested)
        _, _, spec, _ = fixture(self.repo)
        with patch.object(launch, 'check_ready', return_value=(spec, dict(evidence='main'))), \
                patch.object(launch, 'predecessor_gate', return_value=dict(complete=False, errors=[])), \
                patch.object(launch, 'original_start') as start, patch.object(launch.base, 'controller') as controller:
            with self.assertRaisesRegex(RuntimeError, 'original R106'):
                launch.start('/spec', 'main', True)
            with self.assertRaisesRegex(RuntimeError, 'original R106'):
                launch.controller('/spec')
        start.assert_not_called()
        controller.assert_not_called()

    def test_queue_failure_is_terminal_without_retry_or_refreeze(self):
        _, _, spec, _ = fixture(self.repo)
        output = self.repo / 'queue'
        output.mkdir()
        with patch.object(launch, 'check_ready', side_effect=[(spec, dict(evidence='main')), RuntimeError('source changed')]), \
                patch.object(launch.base, 'receipts', return_value=output), patch.object(launch, 'start') as start:
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                launch.queue('/spec', 'main', True, 1)
        saved = (output / 'queue_receipt.json').read_bytes()
        with patch.object(launch, 'check_ready', return_value=(spec, dict(evidence='main'))), \
                patch.object(launch.base, 'receipts', return_value=output):
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                launch.queue('/spec', 'main', True)
        self.assertEqual(saved, (output / 'queue_receipt.json').read_bytes())
        start.assert_not_called()

    def test_warm_cache_identity_only_no_copy_or_repair(self):
        root = self.repo / 'cache'
        root.mkdir()
        manifest = dict(identity_sha256='fixed', files=[dict(path='small.bin', bytes=3)])
        launch.base.save(root / '.r84_cache_identity.json', dict(identity_sha256='fixed'))
        (root / 'small.bin').write_bytes(b'abc')
        self.assertFalse(launch.stage_cache(dict(cache_root=str(root)), manifest)['copied'])
        (root / 'small.bin').unlink()
        with self.assertRaisesRegex(RuntimeError, 'no repair or copy'):
            launch.stage_cache(dict(cache_root=str(root)), manifest)
        self.assertFalse((root / 'small.bin').exists())

    def test_fake_controller_reuses_physical_mutex_and_requires_free_gpu(self):
        _, _, spec, _ = fixture(self.repo)
        output, mutexes = self.repo / 'controller', self.repo / 'mutexes'
        output.mkdir()
        mutexes.mkdir()
        slot = launch.ARM_SLOTS[launch.ARMS[0]]
        mutex = mutexes / ('queue_gpu_' + launch.SLOTS[slot]['uuid'] + '.json')
        mutex.write_text('{}\n')
        inode = mutex.stat().st_ino
        font = self.repo / 'font.otf'
        font.write_bytes(b'fake')
        probe = dict(gpus={slot: dict(compute_processes=[], memory_used_mib=2, utilization_pct=0)},
            processes=[], allowed_cpus=launch.SLOT_CPUS[slot],
            memory=dict(headroom_excluding_reclaimable_cache_bytes=10 * 2**30))
        with patch.object(launch.base, 'check_ready', return_value=(spec, {})), \
                patch.object(launch.base, 'receipts', return_value=output), \
                patch.object(launch.base, 'ROOT', self.repo), patch.object(launch.base.transport, 'ROOT', mutexes), \
                patch.object(launch.base, 'FONT', font), \
                patch.object(launch.base.socket, 'gethostname', return_value='admin01'), \
                patch.object(launch.base.transport, 'capture', side_effect=lambda *a, **k: json.dumps(probe)), \
                patch.object(launch.base, 'reserved_cpus', return_value=set()), \
                patch.object(launch.base, 'pipeline', return_value=0) as pipeline, \
                patch.object(launch.base.time, 'sleep', side_effect=RuntimeError('stop fake polling')):
            probe['gpus'][slot]['compute_processes'] = ['busy']
            with self.assertRaisesRegex(RuntimeError, 'stop fake polling'):
                launch.base.controller('/spec')
            pipeline.assert_not_called()
            self.assertFalse((output / 'claim.json').exists())
            probe['gpus'][slot]['compute_processes'] = []
            self.assertEqual(launch.base.controller('/spec'), 0)
            self.assertEqual(pipeline.call_args.args[2], slot)
        self.assertEqual(mutex.stat().st_ino, inode)
        self.assertEqual(launch.base.read(output / 'claim.json')['cpus'], launch.SLOT_CPUS[slot])

    def test_fake_gpu_failure_stops_later_stages_and_cpu_report(self):
        _, _, spec, _ = fixture(self.repo)
        output = self.repo / 'worker'
        output.mkdir()
        stages = launch.base.commands(spec, spec['slots'][0])
        seen = []

        def fail_test(stage, *args):
            seen.append(stage['name'])
            if stage['name'] == 'TEST':
                raise RuntimeError('TEST exit=7; no later stages')

        launch.base.save(output / 'claim.json', dict(slot='4090_0', token='fake', cpus=[0, 1, 2, 3]))
        with patch.object(launch.base, 'check_ready', return_value=(spec, dict(cache={}))), \
                patch.object(launch.base, 'assert_container', return_value='fake-container'), \
                patch.object(launch.base, 'receipts', return_value=output), \
                patch.object(launch.base, 'commands', return_value=stages), \
                patch.object(launch.base, 'stage_cache', return_value=dict(reused=True)), \
                patch.object(launch.base.os, 'sched_getaffinity', return_value=set(range(72))), \
                patch.object(launch.base.os, 'sched_setaffinity'), \
                patch.dict(os.environ, {launch.TOKEN_ENV: 'fake', 'CUDA_VISIBLE_DEVICES': launch.SLOTS['4090_0']['uuid']}), \
                patch.object(launch.base, 'run_stage', side_effect=fail_test):
            with self.assertRaisesRegex(RuntimeError, 'TEST exit=7'):
                launch.base.worker('/spec', '4090_0', 'fake')
        self.assertEqual(seen, ['train', 'TEST'])
        self.assertEqual(launch.base.read(output / 'worker.json')['state'], 'failed')
        seen.clear()
        with patch.object(launch.base, 'receipts', return_value=output), \
                patch.object(launch.base, 'run_stage', side_effect=RuntimeError('GPU pipeline exit=7')), \
                patch.object(launch.base, 'commands', side_effect=AssertionError('No CPU report')):
            with self.assertRaisesRegex(RuntimeError, 'GPU pipeline exit=7'):
                launch.base.pipeline('/spec', spec, '4090_0', dict(token='fake'))

    def test_admin_imports_and_cli_are_cpu_only(self):
        program = """import subprocess, sys
def forbidden(*a, **k): raise AssertionError('No process/hardware during imports')
subprocess.run = subprocess.check_output = subprocess.Popen = forbidden
import code.V4.r108_launch, code.V4.r108_wave, code.V4.r108_goal
import code.V4.test_r108_launch, code.V4.test_r108_goal
assert 'torch' not in sys.modules
print('CPU administrative imports only')
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-B', '-c', program], cwd=launch.REPO,
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        for argv in (['prepare'], ['predecessor'], ['worker', '--spec', '/fake'], ['node-resources']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.main(argv)
        with patch.object(launch, 'prepare', return_value=dict(fake=True)) as prepare, redirect_stdout(io.StringIO()):
            self.assertEqual(launch.main(['prepare', '--spec', '/fake', '--evidence', 'main', '--runner-tests-passed']), 0)
        prepare.assert_called_once_with(Path('/fake'), 'main', True)


if __name__ == '__main__':
    unittest.main()
