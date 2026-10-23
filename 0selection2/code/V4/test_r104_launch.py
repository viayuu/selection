"""Focused CPU dispatch fixtures; no probes, allocations, queues or Git run."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import r104_launch as launch
from . import r104_wave as wave


def fixture(repo, arm=launch.ARMS[0]):
    root = repo / 'R104_capacity_scaling'
    science = dict(experiment='R104', locked=True, arms=list(launch.ARMS), seed=2, task_scope='full18',
        data_root=str(launch.DATA_ROOT), prepared_root=str(launch.PREPARED_ROOT),
        rows_per_problem=dict(train=81000, val=4500, test=4500), logical_global_batch=1024,
        max_updates=57600, warmup_updates=4320, eval_every=1440, train_eval_every=7200,
        large_micro_cap=128, other_micro_cap=512, eval_batch_size=128,
        current_query_labels_in_forward=False, external_pretraining=False)
    launch.base.save(root / 'protocol_lock.json', science)
    with patch.object(launch, 'MODEL_ROOT', root):
        spec = dict(schema_version=1, run_id=launch.run_id(arm), run_dir=str(root / arm), common_root=str(root),
            science_protocol=str(root / 'protocol_lock.json'), data_root=str(launch.DATA_ROOT),
            prepared_root=str(launch.PREPARED_ROOT), cache_source=str(launch.CACHE_SOURCE),
            cache_root=str(launch.CACHE_ROOT), cache_kind='r84_inputs', wandb_mode='offline',
            slots=[launch.ARM_SLOTS[arm]], cpus=launch.SLOT_CPUS[launch.ARM_SLOTS[arm]],
            cpu_count=4, memory_gib=9, source_files=[], stages=launch.expected_stages(arm))
    path = repo / (launch.run_id(arm) + '.json')
    launch.base.save(path, spec)
    return root, path, spec, science


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='r104-launch-cpu-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)

    def test_private_adapters_original_mutexes_slots_container_and_cpus(self):
        from . import r98_launch, r84_launch, r81_queue, r102_launch
        self.assertIsNot(launch.adapter, r98_launch)
        self.assertIsNot(launch.base, r84_launch)
        self.assertIsNot(launch.base.transport, r81_queue)
        self.assertEqual(launch.base.transport.ROOT, r81_queue.ROOT)
        self.assertEqual(launch.ROOT, r98_launch.ROOT)
        self.assertEqual(r98_launch.TOKEN_ENV, 'R98_DISPATCH_TOKEN')
        self.assertEqual(r102_launch.ARM_SLOTS, dict(A='4090_0', B='4090_1'))
        expected = ('GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50',
                    'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d')
        for slot, uuid in zip(launch.ARM_SLOTS.values(), expected):
            item = launch.SLOTS[slot]
            self.assertEqual((item['job'], item['node'], item['uuid'], item['cap']), (1465, 'gpu03', uuid, 128))
            command = launch.container_command('worker', slot, '/tmp/spec', 'not-a-secret-fixture')
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

    def test_exact_five_stages_and_fixed_protocol(self):
        root, path, spec, science = fixture(self.repo)
        with patch.object(launch, 'MODEL_ROOT', root):
            self.assertEqual(launch.load_spec(path), spec)
            commands = launch.base.commands(spec, '3090_0')
            self.assertEqual(tuple(s['name'] for s in commands), launch.STAGES)
            for stage in commands[:3]:
                args = stage['argv']
                self.assertEqual(args[args.index('--arm') + 1], 'A_width256')
                self.assertEqual(args[args.index('--large-micro-cap') + 1], '128')
                self.assertEqual(args[args.index('--eval-batch-size') + 1], '128')
                self.assertNotIn('--feature-root', args)
            self.assertEqual(commands[-2]['argv'][-2:], ['--lib-batch-size', '2'])
            self.assertEqual(commands[-1]['argv'][-2:], ['--device', 'cpu'])
            for changed in (dict(slots=['4090_0']), dict(cpus=[0, 1, 2, 3]),
                            dict(feature_root='/tmp/new'), dict(stages=spec['stages'][1:])):
                launch.base.save(path, dict(spec, **changed))
                with self.assertRaises(ValueError):
                    launch.load_spec(path)
            launch.base.save(path, spec)
            launch.base.save(root / 'protocol_lock.json', dict(science, other_micro_cap=1024))
            with self.assertRaises(ValueError):
                launch.load_spec(path)

    def test_explicit_main_ready_and_exact_source_freeze(self):
        for evidence, tested in ((None, True), ('', True), ('main', False)):
            for action in (launch.prepare, launch.start, launch.queue):
                with self.assertRaises(ValueError):
                    action('/absent', evidence, tested)
        root, path, spec, _ = fixture(self.repo)
        dispatch = self.repo / 'dispatch'
        output = dispatch / spec['run_id']
        source = self.repo / 'tiny.py'
        source.write_text('VALUE = 1\n')
        plan = dict(spec=spec, main_thread_ready=True, evidence='main CPU proof',
                    primary_cache_reuse_only=True, cache_copy=False, cache_preprocessing=False, automatic_retry=False,
                    source_sha256={str(source): launch.base.sha(source)})
        launch.base.save(output / 'plan.json', plan)
        launch.base.save(output / 'ready.json', dict(main_thread_ready=True, runner_tests_passed=True, source_freeze_ready=True,
            evidence='main CPU proof', plan_sha256=launch.base.sha(output / 'plan.json')))
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch.base, 'ROOT', dispatch), \
                patch.object(launch, 'source_files', return_value=[str(source)]):
            launch.check_ready(path)
            (self.repo / 'r104_unrelated_future.py').write_text('UNRELATED = 1\n')
            launch.check_ready(path)
            source.write_text('VALUE = 2\n')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                launch.check_ready(path)

    def test_sources_have_no_version_glob(self):
        source = self.repo / 'exact.py'
        source.write_text('VALUE = 1\n')
        with patch.object(launch, 'declared_sources', return_value=[source]), \
                patch.object(Path, 'glob', side_effect=AssertionError('No broad glob')):
            before = launch.source_files(dict(source_files=[]))
            (self.repo / 'test_r104_future.py').write_text('VALUE = 2\n')
            self.assertEqual(before, launch.source_files(dict(source_files=[])))

    def test_parent_gate_keeps_other_arm_independent_and_rejects_null_val_sha(self):
        from .test_r98_goal import fixture as prior_fixture
        root, dispatch, locks = prior_fixture(self.repo, 'R99')
        for arm, slot in (('A', '3090_0'), ('B', '3090_1')):
            stages = [dict(name=n, device='cpu' if n == 'PDF' else 'gpu') for n in
                      ('input_features', 'train', 'TEST', 'timing', 'LIB', 'PDF')]
            launch.base.save(dispatch / 'specs' / ('R99_' + arm + '.json'), dict(run_id='R99_' + arm,
                run_dir=str(root / arm), slots=[slot], stages=stages))
            launch.base.save(dispatch / ('R99_' + arm) / 'worker.json', dict(state='complete', exit_status=0,
                stages=[dict(name=s['name'], state='complete', exit_status=0) for s in stages if s['device'] == 'gpu']))
        with patch.object(launch, 'REPO', self.repo), patch.object(launch, 'ROOT', dispatch):
            self.assertTrue(launch.predecessor_gate('A_width256')['complete'])
            failed = dispatch / 'R99_B/worker.json'
            launch.base.save(failed, dict(state='failed', exit_status=1, error=['timing mismatch']))
            self.assertTrue(launch.predecessor_gate('A_width256')['complete'])
            self.assertFalse(launch.predecessor_gate('B_depth8')['complete'])
            path = root / 'A/final_metrics.json'
            record = launch.base.read(path)
            record['val']['checkpoint_sha256'] = None
            launch.base.save(path, record)
            before = path.read_bytes()
            self.assertFalse(launch.predecessor_gate('A_width256')['complete'])
            self.assertEqual(path.read_bytes(), before)

    def test_queue_failure_one_attempt_and_source_change(self):
        _, _, spec, _ = fixture(self.repo)
        output = self.repo / 'queue'
        output.mkdir()
        with patch.object(launch, 'check_ready', return_value=(spec, dict(evidence='ready'))), \
                patch.object(launch.base, 'receipts', return_value=output), \
                patch.object(launch, 'predecessor_gate', side_effect=[dict(complete=False, errors=[]), dict(complete=True, errors=[])]), \
                patch.object(launch.time, 'sleep') as sleep, \
                patch.object(launch, 'start', side_effect=RuntimeError('terminal failure')) as start:
            with self.assertRaisesRegex(RuntimeError, 'terminal failure'):
                launch.queue('/spec', 'ready', True, 1)
            saved = (output / 'queue_receipt.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                launch.queue('/spec', 'ready', True)
            self.assertEqual((output / 'queue_receipt.json').read_bytes(), saved)
        self.assertEqual(start.call_count, 1)
        self.assertEqual(sleep.call_count, 1)

    def test_explicit_main_recovery_keeps_failures_and_requires_all_evidence(self):
        from .test_r98_goal import fixture as prior_fixture
        model, dispatch, locks = prior_fixture(self.repo, 'R99')
        lock, arm = locks['A'], 'A'
        output = dispatch / 'R99_A'
        stages = [dict(name=n, device='cpu' if n == 'PDF' else 'gpu') for n in
                  ('input_features', 'train', 'TEST', 'timing', 'LIB', 'PDF')]
        launch.base.save(dispatch / 'specs/R99_A.json', dict(run_id='R99_A', run_dir=str(model / arm),
            slots=['3090_0'], stages=stages))
        launch.base.save(output / 'worker.json', dict(state='failed', exit_status=1, error=['original timing mismatch'],
            stages=[dict(name=n, state='complete', exit_status=0) for n in ('input_features', 'train', 'TEST')]
            + [dict(name='timing', state='failed', exit_status=1)]))
        launch.base.save(output / 'controller.json', dict(state='failed', exit_status=1,
            stages=[dict(name='allocation_GPU_pipeline', state='failed', exit_status=1)]))
        for name in ('timing.log', 'allocation_GPU_pipeline.log'):
            (output / name).write_text('Preserved original timing failure\n')

        def ref(path):
            return dict(path=str(path), sha256=launch.base.sha(path))

        failures = {n: ref(output / n) for n in
            ('worker.json', 'controller.json', 'timing.log', 'allocation_GPU_pipeline.log')}
        failure_sha = launch.predecessor_goal.json_hash(failures)
        root = model / 'preparation/postprocess_recovery'
        protocol_path = root / 'report_view/A/full_model_statistics/protocol.json'
        launch.base.save(protocol_path, dict(experiment='R99', complete=True, checkpoint_lock=lock))
        final_path = model / 'A/final_metrics.json'
        final = launch.base.read(final_path)
        capture = dict(final['val'].get('identity', final['val']), checkpoint_sha256=None,
            model_state_sha256='d'*64, successful_updates=lock['selected_update'])
        final['val']['identity'] = dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
        launch.base.save(final_path, final)
        backup = copy.deepcopy(final)
        backup['val']['identity'] = capture
        backup_path = root / 'A/final_metrics.before_val_sha_binding.json'
        archive_path = model / 'A/evaluation/val/selected_best/identity.json'
        launch.base.save(backup_path, backup)
        launch.base.save(archive_path, capture)
        pdfs = {}
        for name in launch.predecessor_goal.R99_PDFS:
            path = protocol_path.parent / name
            path.write_bytes(b'%PDF-synthetic')
            pdfs[name] = ref(path)
        report_path = root / 'A/report_stage.json'
        launch.base.save(report_path, dict(schema='r99_postprocess_recovery_v1', experiment='R99', arm=arm,
            stage='report', complete=True, exit_status=0, lock=lock, original_failure_sha256=failure_sha,
            pdfs=pdfs, report_protocol=ref(protocol_path)))
        binding_path = root / 'A/bind_val_stage.json'
        launch.base.save(binding_path, dict(schema='r99_postprocess_recovery_v1', experiment='R99', arm=arm,
            stage='bind_val', complete=True, exit_status=0, original_failure_sha256=failure_sha,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'], selected_update=lock['selected_update'],
            metadata_only=True, after_sha256=launch.base.sha(model / 'A/final_metrics.json'),
            changed_field='val.identity.checkpoint_sha256', old_value=None, new_value=lock['checkpoint_sha256'],
            backup=ref(backup_path), original_archived_val_identity=ref(archive_path),
            full_report_receipt=ref(report_path), model_state_sha256='d'*64,
            numeric_values_changed=False, prediction_arrays_changed=False, weights_changed=False,
            original_failure_receipts_changed=False))
        timing_path = root / 'A/inference_timing.json'
        timing_rows = [dict(problem=p, instances=128) for p in launch.predecessor_goal.PROBLEMS]
        launch.base.save(timing_path, dict(complete=True, exit_status=0,
            recovery_schema='r99_postprocess_recovery_v1', repetitions=3, warmup_batches=1,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            inference_rule=lock['inference_rule'], original_inference_equivalence=False, per_problem=timing_rows,
            efficacy_reused_as_exact_online=False, timing_path='mixed', requested_tf32_policy='default'))
        launch.base.save(protocol_path, dict(launch.base.read(protocol_path), inference_timing=ref(timing_path)))
        report = launch.base.read(report_path)
        report['report_protocol'] = ref(protocol_path)
        launch.base.save(report_path, report)
        binding = launch.base.read(binding_path)
        binding['full_report_receipt'] = ref(report_path)
        launch.base.save(binding_path, binding)
        lib_path = model / 'A/lib_stage_complete.json'
        manual_path = model / 'preparation/manual_stages/A_lib.json'
        launch.base.save(manual_path, dict(stage='lib', arm=arm, state='complete', exit_status=0,
            checkpoint_lock=lock, original_worker_sha256=failures['worker.json']['sha256'],
            original_controller_sha256=failures['controller.json']['sha256']))
        record = dict(schema='r99_postprocess_recovery_v1', experiment='R99', arm=arm, complete=True,
            exit_status=0, state='complete', checkpoint_lock=lock, checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=lock['state_hash'], original_failures=failures, original_failure_sha256=failure_sha,
            original_failure_receipts_preserved=True, original_worker_state='failed', original_controller_state='failed',
            training_rerun=False, full_test_rerun=False, feature_copy=False, feature_rebuild=False, decision_rule_changed=False,
            efficacy_reused_as_exact_online=False, original_inference_equivalence=False,
            stages_completed=['original_locked_full_TEST', 'recovery_timing', 'original_reporter_LIB',
                'private_CPU_full_report', 'metadata_only_VAL_checkpoint_binding'],
            report_receipt=ref(report_path), val_binding=ref(binding_path), pdfs=pdfs,
            timing=dict(**ref(timing_path), status='complete_private_numerical_revision',
                recovery_schema='r99_postprocess_recovery_v1', efficacy_reused_as_exact_online=False,
                original_inference_equivalence=False, numerical_disclosure='Explicit nonexact online revision',
                timing_path='mixed', requested_tf32_policy='default',
                transform_plus_selector_measured=True, pdf_panel=True, measurements=timing_rows),
            lib=dict(manual_receipt=ref(manual_path), completion=ref(lib_path)))
        complete_path = root / 'A/recovery_complete.json'
        with patch.object(launch, 'REPO', self.repo), patch.object(launch, 'ROOT', dispatch):
            self.assertFalse(launch.predecessor_gate('A_width256')['complete'])
            launch.base.save(complete_path, record)
            accepted = launch.predecessor_gate('A_width256')
            self.assertTrue(accepted['complete'], accepted)
            self.assertTrue(accepted['preserved_original_errors'])
            self.assertFalse(accepted['recovery']['original_inference_equivalence'])
            original_final, original_binding = launch.base.read(final_path), launch.base.read(binding_path)
            altered_final = copy.deepcopy(original_final)
            altered_final['val']['macro']['macro_top1'] = 99.0
            launch.base.save(final_path, altered_final)
            launch.base.save(binding_path, dict(original_binding, after_sha256=launch.base.sha(final_path)))
            launch.base.save(complete_path, dict(record, val_binding=ref(binding_path)))
            self.assertFalse(launch.predecessor_gate('A_width256')['complete'])
            launch.base.save(final_path, original_final)
            launch.base.save(binding_path, original_binding)
            for change in (dict(schema='unknown'), dict(checkpoint_sha256='wrong'), dict(feature_copy=True),
                           dict(efficacy_reused_as_exact_online=True), dict(pdfs={})):
                launch.base.save(complete_path, dict(record, **change))
                self.assertFalse(launch.predecessor_gate('A_width256')['complete'], change)
            launch.base.save(complete_path, record)
            path = model / 'A/final_metrics.json'
            final = launch.base.read(path)
            final['val']['identity']['checkpoint_sha256'] = None
            launch.base.save(path, final)
            self.assertFalse(launch.predecessor_gate('A_width256')['complete'])
        for name, item in failures.items():
            self.assertEqual(launch.base.sha(output / name), item['sha256'])

    def test_cache_has_no_fallback_and_wave_never_overwrites(self):
        with patch.object(launch.adapter, 'original_stage_cache') as copy_cache:
            with self.assertRaisesRegex(RuntimeError, 'no preprocessing'):
                launch.stage_cache(dict(cache_root=str(self.repo / 'missing')), dict(identity_sha256='a'*64))
        copy_cache.assert_not_called()
        _, _, spec, _ = fixture(self.repo)
        with patch.object(wave, 'make_spec', return_value=spec), patch.object(launch, 'ROOT', self.repo / 'dispatch'), \
                patch.object(launch, 'start') as start, patch.object(launch, 'queue') as queue:
            path = wave.write_spec(launch.ARMS[0])
            self.assertEqual(wave.write_spec(launch.ARMS[0]), path)
            changed = copy.deepcopy(spec)
            changed['memory_gib'] = 10
            with patch.object(wave, 'make_spec', return_value=changed):
                with self.assertRaisesRegex(RuntimeError, 'Preserve'):
                    wave.write_spec(launch.ARMS[0])
        start.assert_not_called()
        queue.assert_not_called()

    def test_imports_have_no_models_processes_or_environment_mutation(self):
        script = '''
import importlib.abc, os, subprocess, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'numpy', 'wandb'):
            raise AssertionError(name)
sys.meta_path.insert(0, NoModels())
subprocess.Popen = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('No process'))
before = dict(os.environ)
from code.V4 import r104_launch, r104_wave, r104_goal
assert dict(os.environ) == before
'''
        process = subprocess.run([sys.executable, '-B', '-c', script], cwd=launch.REPO,
                                 text=True, capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == '__main__':
    unittest.main()
