"""Small synthetic CPU tests; no real checkpoint, GPU, inference or Git."""

from contextlib import ExitStack
import base64
import fcntl
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r112_finish as finish
from . import r112_postprocess_recovery as recovery


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.v4 = directory / '0selection2/code/V4'
        self.root = self.v4 / 'runs/R112_region_interfaces'
        self.root.mkdir(parents=True)
        self.font = directory / 'synthetic-cjk-font.ttc'
        self.font.write_bytes(b'SYNTHETIC FONT; NO RENDERING')
        self.stack.enter_context(patch.object(recovery, 'V4', self.v4))
        self.stack.enter_context(patch.object(recovery, 'FONT', self.font))
        self.stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}))
        self.stack.enter_context(patch.object(finish, 'git', side_effect=AssertionError('No Git')))
        sources = []
        for name in recovery.FROZEN:
            path = self.v4 / name
            path.write_text('# synthetic frozen source\n')
            sources.append(path)
        for name in ('r112_postprocess_recovery.py', 'test_r112_postprocess_recovery.py', 'r112_finish.py'):
            (self.v4 / name).write_text('# synthetic recovery source\n')
        (self.v4 / 'r112_finish.py').write_bytes(Path(finish.__file__).read_bytes())
        put(self.root / 'protocol_lock.json', dict(experiment='R112'))
        sources.append(self.root / 'protocol_lock.json')
        self.dispatch = self.root / 'preparation/dispatch'
        put(self.dispatch / 'claim.json', dict(source_sha256=recovery.snapshot(sources)))
        put(self.dispatch / 'launch.json', dict(state='failed', exit_status=1, error='original failed'))
        for arm in recovery.ARMS:
            put(self.dispatch / (arm + '.json'), dict(arm=arm, state='failed', exit_status=1,
                phase='report', gpu_uuid='SYNTHETIC_OLD_GPU', stages=[dict(name=s,
                state='failed' if s == 'report' else 'complete', exit_status=int(s == 'report'))
                for s in recovery.STAGES]))
            (self.dispatch / (arm + '_report.log')).write_text('FileNotFoundError: ' + str(self.font))
            run = self.root / arm
            run.mkdir()
            (run / 'best.pt').write_bytes(b'SYNTHETIC MODEL; NOT DESERIALIZED')
            config = dict(experiment='R112', arm=arm, max_successful_updates=57600,
                          source_sha256={'r112_experiment.py': recovery.digest(self.v4 / 'r112_experiment.py')})
            identity = dict(checkpoint_sha256=recovery.digest(run / 'best.pt'),
                selected_update=43200, state_hash=finish.json_hash(config))
            lock = dict(experiment='R112', arm=arm, selected_without_test=True,
                        per_problem_assembly=False, checkpoint=str(run / 'best.pt'), **identity)
            summaries = {}
            for split in ('train', 'val', 'test'):
                folder = run / 'evaluation' / split
                capture = dict(split=split, successful_updates=43200, state_hash=identity['state_hash'],
                               model_state_sha256='SYNTHETIC_MODEL')
                put(folder / 'identity.json', capture)
                path = folder / 'rank000.npz'
                path.write_bytes(b'SYNTHETIC PREDICTIONS; NOT INFERENCE')
                summaries[split] = dict(identity=capture, prediction_archive=str(path))
            for name, value in (
                    ('protocol.json', config), ('final_test_lock.json', lock),
                    ('training_complete.json', dict(successful_updates=57600)),
                    ('test_results.json', dict(complete=True, result=summaries['test'], **identity)),
                    ('final_metrics.json', dict(complete=True, **summaries, **identity)),
                    ('inference_timing.json', dict(complete=True, device='cuda:0',
                         device_name='NVIDIA GeForce RTX 4090', **identity)),
                    ('lib_stage_complete.json', dict(complete=True, datasets=['TSPLIB', 'CVRPLIB'], **identity)),
                    ('history.json', []), ('initialization.json', {})):
                put(run / name, value)
            for dataset in ('TSPLIB', 'CVRPLIB'):
                put(run / 'lib_evaluation' / (dataset + '_results.json'), {})
                path = run / 'lib_evaluation/predictions' / (dataset + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'SYNTHETIC LIB')

    def report(self, argv, **kwargs):
        self.assertEqual(argv[4:6], ['code.V4.r112_reporting', 'report'])
        self.assertEqual(argv[-2:], ['--device', 'cpu'])
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
        self.assertFalse(any(s in argv for s in ('train', 'test', 'timing', 'lib', 'finalize')))
        arm = argv[argv.index('--arm') + 1]
        run = self.root / arm
        lock = recovery.read(run / 'final_test_lock.json')
        put(run / 'report_complete.json', dict(experiment='R112', arm=arm, complete=True,
            **{k: lock[k] for k in ('checkpoint_sha256', 'state_hash', 'selected_update')}))
        output = run / 'full_model_statistics'
        put(output / 'protocol.json', dict(complete=True, checkpoint_lock=lock))
        for name in recovery.REPORT_FILES:
            if name != 'protocol.json':
                (output / name).write_bytes(b'%PDF-1.4\nSYNTHETIC\n%%EOF\n' if name.endswith('.pdf') else b'synthetic\n')
        return SimpleNamespace(returncode=0)

    def recover(self):
        with patch.object(recovery.subprocess, 'run', side_effect=self.report) as call:
            value = recovery.recover(self.root)
        self.assertEqual(call.call_count, 2)
        return value

    def test_recovery_preserves_failed_receipts_and_finisher_accepts_only_proof(self):
        originals = recovery.snapshot(self.dispatch.iterdir())
        with self.assertRaisesRegex(RuntimeError, 'no restart'):
            finish.completed(self.root)
        value = self.recover()
        self.assertEqual(value['state'], 'complete')
        self.assertFalse(value['gpu_compute'])
        readiness = recovery.read(self.root / recovery.RECOVERY / 'readiness.json')
        self.assertEqual(readiness['state'], 'cpu_reports_complete')
        self.assertEqual(readiness['gpu_stages'], [])
        self.assertFalse(readiness['needs_3090_queue'])
        self.assertEqual(originals, recovery.snapshot(self.dispatch.iterdir()))
        module = recovery.private_finisher(self.root)
        accepted = module.completed(self.root)
        self.assertIn(str(recovery.receipt_path(self.root)), accepted)
        with self.assertRaisesRegex(RuntimeError, 'no restart'):
            finish.completed(self.root)
        with patch.object(recovery.subprocess, 'run', side_effect=AssertionError('No repeated report')):
            self.assertEqual(recovery.recover(self.root), value)

    def test_all_input_output_source_and_receipt_tampering_is_rejected(self):
        self.recover()
        module = recovery.private_finisher(self.root)
        paths = [self.dispatch / 'launch.json', self.v4 / 'r112_experiment.py',
            self.root / recovery.ARMS[0] / 'best.pt',
            self.root / recovery.ARMS[0] / 'inference_timing.json',
            self.root / recovery.ARMS[0] / 'evaluation/test/rank000.npz',
            self.root / recovery.ARMS[1] / 'full_model_statistics/selection_distributions.pdf',
            self.root / recovery.RECOVERY / 'A_collapsed_report.log']
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b'\nTAMPERED')
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    module.completed(self.root)
                path.write_bytes(original)
        receipt = recovery.read(recovery.receipt_path(self.root))
        receipt['arms'][recovery.ARMS[0]]['argv'][-1] = 'cuda:0'
        put(recovery.receipt_path(self.root), receipt)
        with self.assertRaises(ValueError):
            module.completed(self.root)

    def test_nonreport_original_failure_never_runs_recovery(self):
        path = self.dispatch / (recovery.ARMS[0] + '.json')
        row = recovery.read(path)
        row['stages'][2]['exit_status'] = 1
        put(path, row)
        with patch.object(recovery.subprocess, 'run', side_effect=AssertionError('No launch')):
            with self.assertRaisesRegex(ValueError, 'four rc0'):
                recovery.recover(self.root)

    def test_missing_font_and_visible_gpu_fail_before_subprocess(self):
        with patch.object(recovery.subprocess, 'run', side_effect=AssertionError('No launch')):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': recovery.UUIDS_3090[0]}):
                with self.assertRaisesRegex(ValueError, 'explicit CUDA_VISIBLE_DEVICES'):
                    recovery.recover(self.root)
            self.font.unlink()
            with self.assertRaises(FileNotFoundError):
                recovery.recover(self.root)

    def test_failed_cpu_report_keeps_failure_and_disallows_automatic_retry(self):
        with patch.object(recovery.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as call:
            with self.assertRaisesRegex(RuntimeError, 'CPU report failed'):
                recovery.recover(self.root)
            self.assertEqual(call.call_count, 1)
            self.assertFalse(recovery.receipt_path(self.root).exists())
            self.assertEqual(recovery.read(self.root / recovery.RECOVERY / 'attempt.json')['state'], 'failed')
            with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                recovery.recover(self.root)
            self.assertEqual(call.call_count, 1)
        with self.assertRaisesRegex(RuntimeError, 'no restart'):
            finish.completed(self.root)

    def test_recovery_lock_blocks_second_writer(self):
        folder = self.root / recovery.RECOVERY
        folder.mkdir(parents=True)
        with (folder / 'recovery.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                recovery.recover(self.root)

    def test_only_cpu_cli_and_no_gpu_commands_in_plan(self):
        details = recovery.prepare(self.root)
        self.assertFalse(details['gpu_compute'])
        self.assertEqual(details['ownership']['authorized_gpu_job'], 1465)
        self.assertEqual(details['ownership']['authorized_gpu_node'], 'gpu03')
        self.assertEqual(details['ownership']['authorized_gpu_uuids'], recovery.UUIDS_3090)
        self.assertFalse(details['ownership']['gpu_4090_compute_allowed'])
        for command in details['cpu_commands'].values():
            self.assertIn('--device cpu', command)
            self.assertNotIn('cuda:0', command)
        readiness = recovery.read(self.root / recovery.RECOVERY / 'readiness.json')
        self.assertEqual(readiness['state'], 'ready_for_cpu_reports')
        self.assertFalse(readiness['gpu_dispatch_ready'])
        self.assertEqual(readiness['per_arm'][recovery.ARMS[0]]['remaining_stages'], ['report'])
        self.assertIn('CUDA_VISIBLE_DEVICES=', readiness['verify_argv'])
        self.assertIn('--publish', readiness['watcher_argv'])
        self.assertIn('code.V4.r112_postprocess_recovery', readiness['watcher_argv'])
        self.assertEqual(readiness['watcher_argv'], readiness['finish_argv'])
        self.assertTrue(readiness['source_sha256'])

    def test_original_success_receipt_can_use_its_exact_saved_generator_bytes(self):
        self.recover()
        original_source = b'# prior successful CPU report generator\n'
        old_hash = recovery.hashlib.sha256(original_source).hexdigest()
        path = recovery.receipt_path(self.root)
        value = recovery.read(path)
        value['generator_sha256'] = old_hash
        put(path, value)
        source = self.root / recovery.RECOVERY / 'report_generator_v1.json'
        put(source, dict(sha256=old_hash, source_base64=base64.b64encode(original_source).decode()))
        self.assertIn(str(source), recovery.verify_receipt(self.root))
        put(source, dict(sha256=old_hash, source_base64=base64.b64encode(b'TAMPERED').decode()))
        with self.assertRaisesRegex(ValueError, 'source snapshot'):
            recovery.verify_receipt(self.root)

    def test_private_finisher_preserves_source_and_appends_only_exact_recovery_paths(self):
        self.recover()
        source = self.v4 / 'r112_finish.py'
        expected = source.read_bytes()
        module = recovery.private_finisher(self.root)
        module.GIT_ROOT = self.v4.parents[2]
        for name in module.CODE_NAMES:
            path = self.v4 / name
            if not path.exists():
                path.write_text('# synthetic source\n')
        for name in (*module.OUTPUTS, 'finish_complete.json'):
            (self.root / name).write_text('synthetic delivery\n')
        research = module.GIT_ROOT / '0selection2/docs/research'
        for group, names in module.RESEARCH_FILES.items():
            for name in names:
                path = research / group / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic research\n')
        layout = research / module.REPORT_LAYOUT
        layout.parent.mkdir(parents=True, exist_ok=True)
        layout.write_text('# synthetic shared layout\n')
        (self.root / recovery.RECOVERY / 'README.md').write_text('synthetic recovery\n')
        included, _ = module.publication_paths(self.root)
        self.assertIn(self.v4 / 'r112_postprocess_recovery.py', included)
        self.assertIn(self.v4 / 'test_r112_postprocess_recovery.py', included)
        self.assertIn(self.root / recovery.RECOVERY / 'recovery_complete.json', included)
        self.assertIn(self.root / recovery.RECOVERY / 'finish_adapter.json', included)
        self.assertFalse(any('/R113' in str(p) or '/R114' in str(p) for p in included))
        self.assertEqual(source.read_bytes(), expected)
        self.assertEqual(recovery.read(self.root / recovery.RECOVERY / 'finish_adapter.json')['original_sha256'],
                         recovery.digest(source))


if __name__ == '__main__':
    unittest.main()
