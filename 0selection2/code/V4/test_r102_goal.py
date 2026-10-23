"""Small saved-receipt CPU checks; comparison/Git/worker activity is mocked."""

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

from . import r102_goal as goal


def fixture(repo):
    root = repo / 'code/V4/runs/R102_full_type_pretrained_fusion'
    launch = repo / 'code/V4/runs/R98_parallel_dispatch'
    locks = {}
    for arm in goal.ARMS:
        run = root / arm
        config = dict(experiment='R102', arm=arm, seed=2, r102_schema_version=1,
            winner_rule='native FP64 ind', initialization='fresh_r84_plus_full_type_fusion',
            data_root='/frozen-new90k', prepared_root='/frozen-r84', rows_per_problem=dict(train=81000, val=4500, test=4500),
            candidate_pools={p: ['alpha', 'beta'] for p in goal.PROBLEMS},
            baselines={p: dict(selection_split='train', train_rows=81000) for p in goal.PROBLEMS},
            max_successful_updates=57600, checkpoint_rule='strict minimum full18 VAL macro_vs_sbs_pct; first ties',
            inference_rule='original maximin', model_spec=dict(three_projections=True),
            retained_parameter_sha256='e'*64, new_parameter_sha256='f'*64,
            trainable_parameter_count=1941257, new_parameter_count=98304,
            projection_shapes=copy.deepcopy(goal.PROJECTION_SHAPES), main_protocol_lock=dict(sha256='d'*64),
            external_solver_pretraining=True, extra_type_pretraining=arm == 'B',
            donor_variant={'A': 'extra_random', 'B': 'full_pretrained'}[arm],
            donor_provenance=dict(components={name: dict(variant=(
                'pretrained' if name == 'VRP' or arm == 'B' else 'random')) for name in ('TSP', 'ATSP', 'VRP')}))
        goal.save(run / 'protocol.json', config)
        state = goal.json_hash(config)
        lock = dict(experiment='R102', arm=arm, state_hash=state, checkpoint_sha256=arm.lower()*64,
            selection_split='val', selected_without_test=True, per_problem_assembly=False, selected_update=1440,
            inference_rule=config['inference_rule'], rule=config['checkpoint_rule'])
        locks[arm] = lock
        goal.save(run / 'final_test_lock.json', lock)
        result = dict(identity=dict(state_hash=state, checkpoint_sha256=lock['checkpoint_sha256']),
            macro=dict(top1=.5, mean_cost=10., actual_regret=.1), per_problem={p: dict(n=4500) for p in goal.PROBLEMS})
        train_result = dict(result, per_problem={p: dict(n=81000) for p in goal.PROBLEMS})
        rows = [dict(problem=p, instances=128, online_donor_verified=True,
            online_fields_removed=['nco_nodes', 'nco_node_mask'],
            donor_plus_selector=dict(batch_seconds=[.128]*3, mean_seconds_per_instance=.001, donor_forward_calls=4, warmup_batches=1),
            cache_hit_selector=dict(batch_seconds=[.064]*3, mean_seconds_per_instance=.0005, donor_forward_calls=0, warmup_batches=1))
            for p in goal.PROBLEMS]
        records = {
            'initialization.json': dict(historical_selector_checkpoint_loaded=False,
                external_solver_pretraining=True, extra_type_pretraining=arm == 'B',
                trainable_parameter_count=1941257, new_parameter_count=98304, removed_parameter_names=[],
                common_parameter_sha256='e'*64, new_parameter_sha256='f'*64, new_parameter_shapes=goal.PROJECTION_SHAPES),
            'training_complete.json': dict(successful_updates=57600, state_hash=state,
                updates_by_problem=dict.fromkeys(goal.PROBLEMS, 3200)),
            'final_metrics.json': dict(state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'],
                selected_update=1440, train=train_result, val=result),
            'test_results.json': dict(complete=True, state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], result=result),
            'lib_stage_complete.json': dict(complete=True, exit_status=0, state_hash=state,
                checkpoint_sha256=lock['checkpoint_sha256'], datasets=['TSPLIB', 'CVRPLIB'], native_counts=dict(TSPLIB=49, CVRPLIB=100)),
            'postprocess_status.json': dict(complete=True, phase='complete', exit_status=0),
            'inference_timing.json': dict(complete=True, state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'],
                inference_rule=lock['inference_rule'], includes_native_encoder=True, online_donor_verified=True,
                all18_donor_coverage=True, cache_hit_is_total=False, repetitions=3,
                scope='GPU-resident batch128, amortized', excludes='I/O, decoder', per_problem=rows),
        }
        for name, value in records.items():
            goal.save(run / name, value)
        goal.save(run / 'full_model_statistics/protocol.json', dict(experiment='R102', complete=True, checkpoint_lock=lock,
            external_solver_pretraining=True, extra_type_pretraining=arm == 'B',
            inference_timing=dict(sha256=goal.file_hash(run / 'inference_timing.json')),
            target_identity=dict(target_count=36, sha256='c'*64)))
        for name in goal.adapter.R99_PDFS:
            (run / 'full_model_statistics' / name).write_bytes(b'%PDF-synthetic')
        stages = [dict(name=name, device='cpu' if name == 'PDF' else 'gpu')
                  for name in ('input_features', 'train', 'TEST', 'timing', 'LIB', 'PDF')]
        goal.save(launch / 'specs' / ('R102_' + arm + '.json'), dict(run_id='R102_' + arm, run_dir=str(run),
            slots=[{'A': '4090_0', 'B': '4090_1'}[arm]], routing_feature_root='/tmp/shiys_R94_B_node_features',
            routing_mode='reuse_original_R94B', stages=stages))
        goal.save(launch / ('R102_' + arm) / 'worker.json', dict(state='complete', exit_status=0,
            stages=[dict(name=s['name'], state='complete', exit_status=0) for s in stages if s['device'] == 'gpu']))
        goal.save(launch / ('R102_' + arm) / 'controller.json', dict(state='complete', exit_status=0,
            stages=[dict(name=name, state='complete', exit_status=0) for name in ('allocation_GPU_pipeline', 'PDF')]))
    return root, launch, locks


def comparison(root, locks):
    goal.save(root / 'comparison_artifacts/protocol.json', dict(experiment='R102', complete=True, exit_status=0,
        checkpoints=locks, target_identity=dict(target_count=36, sha256='c'*64),
        inference_timing={arm: dict(sha256=goal.file_hash(root / arm / 'inference_timing.json')) for arm in goal.ARMS}))
    (root / 'comparison_artifacts/COMPARISON.pdf').write_bytes(b'%PDF-synthetic')
    (root / 'RESULTS.pdf').write_bytes(b'%PDF-synthetic')


class GoalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r102-goal-cpu-')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / 'repo'

    def test_private_imports_are_cpu_only_and_leave_historical_watchers_unchanged(self):
        from . import r98_goal, r101_goal
        self.assertIsNot(goal.adapter, r98_goal)
        self.assertNotIn('R102', r98_goal.VERSIONS)
        self.assertNotIn('R102', r101_goal.adapter.VERSIONS)
        script = '''
import importlib.abc, os, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'numpy', 'wandb', 'tabicl'):
            raise AssertionError(name)
sys.meta_path.insert(0, NoModels())
before = dict(os.environ)
from code.V4 import r102_goal, r102_launch, r102_wave
assert dict(os.environ) == before
'''
        process = subprocess.run([sys.executable, '-B', '-c', script], cwd=goal.REPO,
                                 text=True, capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_both_complete_arms_same_lock_reports_and_no_weights_required(self):
        root, launch, locks = fixture(self.repo)
        before = goal.status(root=root, launch_root=launch)
        self.assertTrue(before['all_arms_ready'], before)
        self.assertFalse(before['complete'])
        comparison(root, locks)
        current = goal.status(root=root, launch_root=launch)
        self.assertTrue(current['complete'], current)
        self.assertFalse(current['goal_complete'])
        self.assertFalse((root / 'goal_progress.json').exists())
        self.assertFalse((root / 'A/best.pt').exists())
        for relative in ('B/full_model_statistics/learning_curves.pdf', 'B/inference_timing.json'):
            path = root / relative
            original = path.read_bytes()
            path.unlink()
            self.assertFalse(goal.status(root=root, launch_root=launch)['all_arms_ready'])
            path.write_bytes(original)

    def test_partial_training_TEST_and_cached_only_timing_fail_closed(self):
        root, launch, _ = fixture(self.repo)
        changes = [
            ('A/final_test_lock.json', lambda r: r.update(selection_split='test')),
            ('A/training_complete.json', lambda r: r.update(successful_updates=57599)),
            ('A/test_results.json', lambda r: r['result']['per_problem'].pop('ATSP')),
            ('A/inference_timing.json', lambda r: r['per_problem'][0]['donor_plus_selector'].update(donor_forward_calls=0)),
            ('A/lib_stage_complete.json', lambda r: r.update(native_counts=dict(TSPLIB=48, CVRPLIB=100))),
            ('A/initialization.json', lambda r: r.update(external_solver_pretraining=False)),
            ('A/full_model_statistics/protocol.json', lambda r: r.update(extra_type_pretraining=True)),
        ]
        for relative, mutate in changes:
            path = root / relative
            original = goal.optional(path)
            changed = copy.deepcopy(original)
            mutate(changed)
            goal.save(path, changed)
            current = goal.status(root=root, launch_root=launch)
            self.assertFalse(current['all_arms_ready'], relative)
            if relative in ('A/initialization.json', 'A/full_model_statistics/protocol.json'):
                self.assertFalse(current['arms'][0]['checks']['external_pretraining_scope'])
            goal.save(path, original)

    def test_main_authorization_before_watch_compare_publish_has_no_side_effects(self):
        root, launch, _ = fixture(self.repo)
        for function in (goal.watch, goal.compare_once, goal.publish):
            with self.assertRaises(ValueError):
                function(root, launch)
        self.assertFalse((root / 'goal_watch.lock').exists())
        with self.assertRaisesRegex(RuntimeError, 'source-frozen'):
            goal.require_ready(root, launch, 'team still WIP')

    def test_source_ready_gate_detects_edits_without_opening_checkpoint(self):
        root, launch, _ = fixture(self.repo)
        source = self.repo / 'code/V4/tiny.py'
        source.write_text('value = 1\n')
        for arm in goal.ARMS:
            output = launch / ('R102_' + arm)
            spec = goal.optional(launch / 'specs' / ('R102_' + arm + '.json'))
            goal.save(output / 'plan.json', dict(spec=spec, evidence='main-ready',
                source_sha256={str(source): goal.file_hash(source), str(root / 'never-open.pt'): 'a'*64}))
            goal.save(output / 'ready.json', dict(main_thread_ready=True, runner_tests_passed=True,
                evidence='main-ready', plan_sha256=goal.file_hash(output / 'plan.json')))
        goal.require_ready(root, launch, 'main-ready')
        self.assertFalse((root / 'never-open.pt').exists())
        source.write_text('value = 2\n')
        with self.assertRaisesRegex(RuntimeError, 'Source changed'):
            goal.require_ready(root, launch, 'main-ready')

    def test_CPU_compare_once_including_r94_reference_and_failure_no_retry(self):
        root, launch, locks = fixture(self.repo)

        def run(argv, **kwargs):
            self.assertEqual(argv[4:6], ['code.V4.r102_reporting', 'compare'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertIn('--r94-root', argv)
            comparison(root, locks)
            return SimpleNamespace(returncode=0)

        with patch.object(goal, 'require_ready'), patch.object(goal.subprocess, 'run', side_effect=run) as called:
            goal.compare_once(root, launch, 'main-ready', r94_root=self.repo / 'R94B')
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, launch, 'main-ready')
        self.assertEqual(called.call_count, 1)
        self.assertEqual(goal.optional(root / 'goal_compare_receipt.json')['exit_status'], 0)
        (root / 'comparison_artifacts/COMPARISON.pdf').unlink()
        with patch.object(goal, 'require_ready'), patch.object(goal.subprocess, 'run') as no_retry:
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                goal.compare_once(root, launch, 'main-ready')
        no_retry.assert_not_called()

    def test_watcher_stays_quiet_for_nonactionable_changes_and_never_replaces(self):
        root, launch, _ = fixture(self.repo)
        waiting = dict(errors=[], all_arms_ready=False, comparison_complete=False, complete=False, phase='waiting_for_arms')
        failed = dict(waiting, errors=[dict(error='synthetic stage failure')], phase='failed')
        stream = io.StringIO()
        with patch.object(goal, 'require_ready'), patch.object(goal, 'status', side_effect=[waiting, dict(waiting, extra='not actionable'), failed]), \
                patch.object(goal.time, 'sleep') as sleep, patch.object(goal, 'compare_once') as compare, \
                patch.object(goal, 'publish') as publish, redirect_stdout(stream):
            self.assertEqual(goal.watch(root, launch, evidence='main-ready'), 1)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(len(stream.getvalue().strip().splitlines()), 1)
        self.assertIn('failed', stream.getvalue())
        compare.assert_not_called()
        publish.assert_not_called()
        original = (root / 'goal_watch_receipt.json').read_bytes()
        with patch.object(goal, 'require_ready'), self.assertRaisesRegex(RuntimeError, 'already attempted'):
            goal.watch(root, launch, evidence='main-ready')
        self.assertEqual((root / 'goal_watch_receipt.json').read_bytes(), original)

    def test_curated_source_dependencies_notes_reports_no_unrelated_WIP_or_arrays(self):
        root, launch, _ = fixture(self.repo)
        good = [self.repo / 'code/V4/r102_launch.py', self.repo / 'code/V4/test_r102_goal.py',
                self.repo / 'code/V4/r98_launch.py', self.repo / 'code/V4/r84_launch.py',
                self.repo / 'docs/research/pretrained_representation_coverage_tsp_atsp.md',
                self.repo / 'docs/research/post_r100_broad_direction.md', root / 'PLAN.md']
        bad = [self.repo / 'code/V4/r103_model.py', self.repo / 'docs/research/unrelated.md',
               root / 'A/best.pt', root / 'teacher_cache/train/CVRP/nco_nodes.npy',
               root / 'A/evaluation/test/pred.npz', root / 'preparation/exports/payload.json']
        for path in [*good, *bad]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}' if path.suffix == '.json' else '# synthetic\n')
        secret = root / 'preparation/credentials.json'
        goal.save(secret, dict(api_key='never-publish-this'))
        with patch.object(goal, 'ROOT', root), patch.object(goal, 'LAUNCH', launch), patch.object(goal, 'REPO', self.repo):
            paths = set(goal.publication_paths('R102', root, launch, self.base))
        self.assertTrue({str(p.relative_to(self.base)) for p in good} <= paths)
        self.assertFalse(paths & {str(p.relative_to(self.base)) for p in [*bad, secret]})

    def test_shared_git_mutex_publisher_is_only_delegated_after_authorization(self):
        root, launch, _ = fixture(self.repo)
        with patch.object(goal, 'require_ready') as ready, \
                patch.object(goal.adapter, 'publish', return_value=dict(complete=True)) as publish:
            self.assertTrue(goal.publish(root, launch, 'main-ready')['complete'])
        ready.assert_called_once_with(root, launch, 'main-ready')
        publish.assert_called_once_with('R102', root, launch)
        self.assertIn('codex-publication.lock', goal.adapter.publish.__code__.co_consts)


if __name__ == '__main__':
    unittest.main()
