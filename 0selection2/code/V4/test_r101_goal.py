"""Small CPU receipts, no models/arrays. External compare and Git are mocked."""

from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r101_goal as goal


def fixture(repo):
    root, launch = repo / 'code/V4/runs/R101_hint_pretraining', repo / 'code/V4/runs/R98_parallel_dispatch'
    locks = {}
    for arm in goal.ARMS:
        run = root / arm
        spec = dict(updates=12800, logical_batch=1024, label_inputs=False,
                    teacher_split='train', teacher_scope=list(goal.TEACHER_TASKS))
        pretrain = dict(experiment='R101', arm=arm, complete=True, successful_updates=12800,
            selected_without_validation=True, checkpoint_sha256=arm.lower()*64, encoder_sha256=arm.lower()*64,
            initial_trainable_sha256='f'*64, unchanged_non_encoder_sha256='e'*64, protocol_sha256='d'*64,
            pretrain_spec=spec, updates_by_problem=dict.fromkeys(goal.TEACHER_TASKS, 800),
            teacher_identity=dict(variant={'A': 'random', 'B': 'pretrained'}[arm], split='train', rows_per_task=81000,
                no_teacher_in_final_model=True, metadata_sha256=dict.fromkeys(goal.TEACHER_TASKS, 'c'*64)))
        goal.save(run / 'pretrain_complete.json', pretrain)
        (run / 'pretrain_updates.jsonl').write_text('{"successful_updates":12800}\n')
        init = dict(initial_trainable_sha256='f'*64, r77_fresh_trainable_sha256='f'*64,
            unchanged_non_encoder_sha256='e'*64, pretrained_encoder_sha256=pretrain['encoder_sha256'],
            historical_selector_checkpoint_loaded=False, transferred_parameter_prefix='instance_encoder.',
            optimizer_reset=True, scaler_reset=True, rng_reset=True, task_sample_streams_reset=True,
            trainable_parameter_count=1842953)
        goal.save(run / 'initialization.json', init)
        config = dict(experiment='R101', arm=arm, seed=2, winner_rule='native FP64 ind',
            rows_per_problem=dict(train=81000, val=4500, test=4500), data_root='/frozen-new90k', prepared_root='/frozen-r84',
            candidate_pools={p: ['alpha', 'beta'] for p in goal.PROBLEMS},
            candidate_pool_ids={p: [2, 7] for p in goal.PROBLEMS},
            baselines={p: dict(selection_split='train', train_rows=81000) for p in goal.PROBLEMS},
            max_successful_updates=57600, checkpoint_rule='minimum full18 VAL macro_vs_sbs_pct',
            inference_rule='original maximin; original inputs only', hint_pretraining=spec,
            pretraining_lineage={k: v for k, v in pretrain.items() if k not in ('updates_by_problem', 'selected_without_validation')},
            main_protocol_lock=dict(sha256='d'*64), pretrained_encoder_sha256=pretrain['encoder_sha256'],
            unchanged_non_encoder_sha256='e'*64, trainable_parameter_count=1842953, new_trainable_parameters=0,
            deployment_teacher=False, downstream_teacher_inputs=False, downstream_hint_loss=False)
        goal.save(run / 'protocol.json', config)
        state = goal.json_hash(config)
        lock = dict(experiment='R101', arm=arm, state_hash=state, checkpoint_sha256=arm.lower()*64,
            selected_without_test=True, per_problem_assembly=False, selected_update=1440,
            inference_rule=config['inference_rule'], rule=config['checkpoint_rule'])
        locks[arm] = lock
        goal.save(run / 'final_test_lock.json', lock)
        result = dict(state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], macro=dict(macro_top1=.5),
            per_problem={p: dict(n=4500) for p in goal.PROBLEMS})
        for name, record in {
            'training_complete.json': dict(successful_updates=57600, state_hash=state,
                updates_by_problem=dict.fromkeys(goal.PROBLEMS, 3200)),
            'final_metrics.json': dict(state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], selected_update=1440, val=result),
            'test_results.json': dict(complete=True, state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], result=result),
            'lib_stage_complete.json': dict(complete=True, exit_status=0, state_hash=state,
                checkpoint_sha256=lock['checkpoint_sha256'], datasets=['TSPLIB', 'CVRPLIB']),
            'postprocess_status.json': dict(complete=True, phase='complete', exit_status=0),
            'inference_timing.json': dict(complete=True, state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'],
                student_only=True, teacher_arrays_opened=False, online_teacher=False),
        }.items():
            goal.save(run / name, record)
        goal.save(run / 'full_model_statistics/protocol.json', dict(experiment='R101', complete=True, checkpoint_lock=lock,
            inference_timing=dict(sha256=goal.file_hash(run / 'inference_timing.json')),
            pretraining_evidence=dict(receipt_sha256=goal.file_hash(run / 'pretrain_complete.json'),
                                     trace_sha256=goal.file_hash(run / 'pretrain_updates.jsonl'))))
        for name in goal.adapter.FROZEN_PDFS:
            (run / 'full_model_statistics' / name).write_bytes(b'%PDF-synthetic')
        stages = [dict(name=name, device='cpu' if name == 'PDF' else 'gpu')
                  for name in ('pretrain', 'train', 'TEST', 'timing', 'LIB', 'PDF')]
        goal.save(launch / 'specs' / ('R101_'+arm+'.json'), dict(run_id='R101_'+arm, run_dir=str(run), stages=stages))
        goal.save(launch / ('R101_'+arm) / 'worker.json', dict(state='complete', exit_status=0,
            stages=[dict(name=s['name'], state='complete', exit_status=0) for s in stages if s['device'] == 'gpu']))
        goal.save(launch / ('R101_'+arm) / 'controller.json', dict(state='complete', exit_status=0,
            stages=[dict(name=name, state='complete', exit_status=0) for name in ('allocation_GPU_pipeline', 'PDF')]))
    return root, launch, locks


def comparison(root, locks):
    goal.save(root / 'comparison_artifacts/protocol.json', dict(experiment='R101', complete=True,
        exit_status=0, checkpoints=locks))
    (root / 'comparison_artifacts/COMPARISON.pdf').write_bytes(b'%PDF-synthetic')
    (root / 'RESULTS.pdf').write_bytes(b'%PDF-synthetic')


class GoalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r101-goal-cpu-')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / 'repo'

    def test_private_adapter_and_cli_do_not_mutate_live_r98_watcher(self):
        from . import r98_goal
        self.assertIsNot(goal.adapter, r98_goal)
        self.assertNotIn('R101', r98_goal.VERSIONS)
        self.assertEqual(r98_goal.FROZEN_PDFS[-1], 'target_recovery.pdf')
        self.assertTrue(goal.parser().parse_args(['--watch', '--publish-github']).publish_github)
        script = '''
import importlib.abc, os, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'numpy', 'tabicl', 'wandb'):
            raise AssertionError(name)
sys.meta_path.insert(0, NoModels())
before = dict(os.environ)
from code.V4 import r101_goal
assert before == dict(os.environ)
'''
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=goal.REPO,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stage_budgets_remain_separate_and_no_weight_or_teacher_files_needed(self):
        root, launch, locks = fixture(self.repo)
        current = goal.status(root=root, launch_root=launch, persist=False)
        self.assertTrue(current['all_arms_ready'], current)
        self.assertFalse(current['complete'])
        self.assertEqual((current['stage_I_updates_per_arm'], current['stage_II_updates_per_arm']), (12800, 57600))
        comparison(root, locks)
        self.assertTrue(goal.status(root=root, launch_root=launch, persist=False)['complete'])
        self.assertFalse((root / 'A/pretrain_end.pt').exists())
        self.assertFalse((root / 'A/best.pt').exists())
        for name, updates in (('pretrain_complete.json', 12799), ('training_complete.json', 57599)):
            path = root / 'B' / name
            original = goal.optional(path)
            goal.save(path, dict(original, successful_updates=updates))
            self.assertFalse(goal.status(root=root, launch_root=launch, persist=False)['complete'])
            goal.save(path, original)

    def test_parent_lineage_nonencoder_fresh_and_student_only_are_required(self):
        root, launch, _ = fixture(self.repo)
        path = root / 'A/initialization.json'
        original = goal.optional(path)
        for change in (dict(unchanged_non_encoder_sha256='wrong'), dict(optimizer_reset=False),
                       dict(pretrained_encoder_sha256='wrong')):
            goal.save(path, dict(original, **change))
            self.assertFalse(goal.status(root=root, launch_root=launch, persist=False)['all_arms_ready'])
        goal.save(path, original)
        timing = root / 'A/inference_timing.json'
        value = goal.optional(timing)
        goal.save(timing, dict(value, teacher_arrays_opened=True))
        self.assertFalse(goal.status(root=root, launch_root=launch, persist=False)['all_arms_ready'])

    def test_compare_cpu_only_once_after_both_arms_with_same_lock(self):
        root, launch, locks = fixture(self.repo)

        def command(argv, **kwargs):
            self.assertEqual(argv[4:6], ['code.V4.r101_reporting', 'compare'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            comparison(root, locks)
            return SimpleNamespace(returncode=0)

        with patch.object(goal.subprocess, 'run', side_effect=command) as run:
            goal.compare_once(root, launch)
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, launch)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(goal.status(root=root, launch_root=launch)['goal_complete'])

    def test_failure_receipt_preserved_no_compare_retry_or_gpu_action(self):
        root, launch, _ = fixture(self.repo)
        with patch.object(goal.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, launch)
            original = (root / 'goal_compare_receipt.json').read_bytes()
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, launch)
        self.assertEqual(run.call_count, 1)
        self.assertEqual((root / 'goal_compare_receipt.json').read_bytes(), original)
        with patch.object(goal.subprocess, 'run') as run, patch.object(goal.time, 'sleep') as sleep, redirect_stdout(io.StringIO()):
            self.assertEqual(goal.watch(root, launch), 1)
            receipt = (root / 'goal_watch_receipt.json').read_bytes()
            with self.assertRaises(RuntimeError):
                goal.watch(root, launch)
            self.assertEqual((root / 'goal_watch_receipt.json').read_bytes(), receipt)
        run.assert_not_called()
        sleep.assert_not_called()

    def test_curated_version_sources_reports_and_lineage_only(self):
        root, launch, _ = fixture(self.repo)
        good = [self.repo / 'code/V4/r101_launch.py', self.repo / 'code/V4/test_r101_goal.py',
                self.repo / 'code/V4/r101_launch_README.md']
        bad = [self.repo / 'code/V4/r102_model.py', root / 'A/pretrain_end.pt', root / 'A/best.pt',
               root / 'teacher_cache/A/train/CVRP/nco_nodes.npy', root / 'A/evaluation/test/pred.npz',
               root / 'A/wandb/config.json']
        for path in [*good, *bad]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}' if path.suffix == '.json' else 'synthetic')
        with patch.object(goal, 'ROOT', root), patch.object(goal, 'LAUNCH', launch), patch.object(goal, 'REPO', self.repo):
            paths = goal.publication_paths('R101', root, launch, self.base)
        self.assertTrue({str(p.relative_to(self.base)) for p in good}.issubset(set(paths)))
        self.assertFalse(set(paths) & {str(p.relative_to(self.base)) for p in bad})
        self.assertIn(str((root / 'A/pretrain_complete.json').relative_to(self.base)), paths)

    def test_quiet_unchanged_and_shared_git_publisher_gate(self):
        root, launch, locks = fixture(self.repo)
        comparison(root, locks)
        first = goal.status(root=root, launch_root=launch)
        path = root / 'goal_progress.json'
        modified = path.stat().st_mtime_ns
        self.assertEqual(goal.status(root=root, launch_root=launch), first)
        self.assertEqual(path.stat().st_mtime_ns, modified)
        with patch.object(goal.adapter, 'publish', return_value=dict(complete=True)) as publish:
            self.assertTrue(goal.publish(root, launch)['complete'])
        publish.assert_called_once_with('R101', root, launch)
        self.assertEqual(goal.adapter.publish.__module__, goal.adapter.__name__)
        self.assertIn('codex-publication.lock', goal.adapter.publish.__code__.co_consts)

    def test_native_test_and_pretraining_report_cannot_be_partial(self):
        root, launch, _ = fixture(self.repo)
        test = root / 'A/test_results.json'
        record = goal.optional(test)
        original = copy.deepcopy(record)
        record['result']['per_problem'].pop('ATSP')
        goal.save(test, record)
        self.assertFalse(goal.status(root=root, launch_root=launch, persist=False)['all_arms_ready'])
        goal.save(test, original)
        (root / 'A/full_model_statistics/pretraining_curves.pdf').unlink()
        self.assertFalse(goal.status(root=root, launch_root=launch, persist=False)['all_arms_ready'])


if __name__ == '__main__':
    unittest.main()
