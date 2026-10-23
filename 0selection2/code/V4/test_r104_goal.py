"""Saved CPU evidence fixtures; comparison and all Git subprocesses are mocked."""

from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r104_goal as goal


def fixture(repo):
    root = repo / 'code/V4/runs/R104_capacity_scaling'
    dispatch = repo / 'code/V4/runs/R98_parallel_dispatch'
    goal.save(root / 'protocol_lock.json', dict(experiment='R104', locked=True))
    locks = {}
    for index, arm in enumerate(goal.ARMS):
        run = root / arm
        config = dict(experiment='R104', arm=arm, seed=2, winner_rule='native FP64 ind',
            data_root='/frozen-new90k', prepared_root='/frozen-r84', rows_per_problem=dict(train=81000, val=4500, test=4500),
            candidate_pools={p: ['a', 'b'] for p in goal.PROBLEMS},
            baselines={p: dict(selection_split='train', train_rows=81000) for p in goal.PROBLEMS},
            max_successful_updates=57600, checkpoint_rule='strict minimum full18 VAL macro_vs_sbs_pct; first ties',
            selected_val_binding_rule='explicit selected-state final-checkpoint binding',
            inference_rule='original score MLP maximin', main_protocol_lock=dict(sha256=goal.file_hash(root / 'protocol_lock.json')),
            model_spec=dict(capacity=arm), initialization_sha256=str(index)*64,
            trainable_parameter_count=[7288969, 3426057][index])
        goal.save(run / 'protocol.json', config)
        state = goal.json_hash(config)
        lock = dict(experiment='R104', arm=arm, state_hash=state, checkpoint_sha256=str(index+1)*64,
            selection_split='val', selected_without_test=True, per_problem_assembly=False,
            selected_update=1440, inference_rule=config['inference_rule'], rule=config['checkpoint_rule'])
        locks[arm] = lock
        result = dict(identity=dict(state_hash=state, checkpoint_sha256=lock['checkpoint_sha256']),
            macro=dict(top1=.5, mean_cost=10., actual_regret=.1), per_problem={p: dict(n=4500) for p in goal.PROBLEMS})
        final = dict(state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], selected_update=1440,
            train=dict(result, per_problem={p: dict(n=81000) for p in goal.PROBLEMS}), val=result)
        records = {
            'final_test_lock.json': lock,
            'training_complete.json': dict(successful_updates=57600, state_hash=state, updates_by_problem=dict.fromkeys(goal.PROBLEMS, 3200)),
            'final_metrics.json': final,
            'test_results.json': dict(complete=True, state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], result=result),
            'lib_stage_complete.json': dict(experiment='R104', complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
                datasets=['TSPLIB', 'CVRPLIB']),
            'postprocess_status.json': dict(complete=True, phase='complete', exit_status=0),
            'inference_timing.json': dict(experiment='R104', arm=arm, complete=True, state_hash=state,
                checkpoint_sha256=lock['checkpoint_sha256'], parameter_count=config['trainable_parameter_count'],
                inference_rule=lock['inference_rule'], repetitions=3, warmup_batches=1,
                all_current_model_included=True, donor_present=False, cached_feature_latency_used=False,
                end_to_end_measured=False, device='cuda:0', scope='complete_standalone_model_original_GPU_resident_inputs',
                excludes='I/O, H2D, solver', per_problem=[dict(problem=p, instances=128,
                    standalone_model=dict(batch_seconds=[.128]*3, mean_seconds_per_instance=.001)) for p in goal.PROBLEMS]),
        }
        for name, record in records.items():
            goal.save(run / name, record)
        goal.save(run / 'full_model_statistics/protocol.json', dict(experiment='R104', complete=True, checkpoint_lock=lock,
            inference_timing=records['inference_timing.json'],
            target_identity=dict(target_count=36, sha256='c'*64)))
        for name in goal.adapter.R99_PDFS:
            (run / 'full_model_statistics' / name).write_bytes(b'%PDF-synthetic')
        goal.save(dispatch / 'specs' / ('R104_' + arm + '.json'), dict(run_id='R104_' + arm, run_dir=str(run),
            slots=[['3090_0', '3090_1'][index]], stages=[dict(name=n, device='cpu' if n == 'PDF' else 'gpu')
                for n in ('train', 'TEST', 'timing', 'LIB', 'PDF')]))
        goal.save(dispatch / ('R104_' + arm) / 'worker.json', dict(state='complete', exit_status=0,
            stages=[dict(name=n, state='complete', exit_status=0) for n in ('train', 'TEST', 'timing', 'LIB')]))
        goal.save(dispatch / ('R104_' + arm) / 'controller.json', dict(state='complete', exit_status=0,
            stages=[dict(name=n, state='complete', exit_status=0) for n in ('allocation_GPU_pipeline', 'PDF')]))
    return root, dispatch, locks


def comparison(root, locks):
    goal.save(root / 'comparison_artifacts/protocol.json', dict(experiment='R104', complete=True,
        checkpoints=locks, matched=False, parameter_matched=False, equal_flops=False,
        identical_full_initialization=False, training=False, inference=False))
    for name in ('paired_changes_B_vs_A.csv', 'target_recovery_B_vs_A.csv', 'target_recovery_summary_B_vs_A.csv'):
        (root / 'comparison_artifacts' / name).write_text('problem,n\nTSP,4500\n')


class GoalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='r104-goal-cpu-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)

    def test_complete_whole_version_does_not_match_capacity_params_or_init(self):
        root, dispatch, locks = fixture(self.repo)
        current = goal.status(root=root, launch_root=dispatch)
        self.assertTrue(current['all_arms_ready'], current)
        self.assertFalse(current['complete'])
        comparison(root, locks)
        current = goal.status(root=root, launch_root=dispatch)
        self.assertTrue(current['complete'], current)
        self.assertFalse(current['goal_complete'])
        self.assertFalse((root / 'goal_progress.json').exists())
        self.assertFalse((root / 'A_width256/best.pt').exists())

    def test_all_saved_stages_and_same_lock_required(self):
        root, dispatch, locks = fixture(self.repo)
        comparison(root, locks)
        mutations = [
            ('A_width256/final_test_lock.json', lambda r: r.update(selection_split='test')),
            ('A_width256/training_complete.json', lambda r: r.update(successful_updates=57599)),
            ('A_width256/final_metrics.json', lambda r: r['train']['per_problem'].pop('ATSP')),
            ('A_width256/final_metrics.json', lambda r: r['val']['identity'].update(checkpoint_sha256=None)),
            ('A_width256/test_results.json', lambda r: r['result']['identity'].update(checkpoint_sha256='wrong')),
            ('A_width256/lib_stage_complete.json', lambda r: r.update(datasets=['TSPLIB'])),
            ('A_width256/inference_timing.json', lambda r: r['per_problem'][0]['standalone_model'].update(batch_seconds=[.1])),
            ('A_width256/full_model_statistics/protocol.json', lambda r: r['checkpoint_lock'].update(checkpoint_sha256='wrong')),
        ]
        for relative, mutate in mutations:
            path = root / relative
            original = goal.optional(path)
            changed = copy.deepcopy(original)
            mutate(changed)
            goal.save(path, changed)
            self.assertFalse(goal.status(root=root, launch_root=dispatch)['complete'], relative)
            goal.save(path, original)
        (root / 'B_depth8/full_model_statistics/learning_curves.pdf').unlink()
        self.assertFalse(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_error_string_list_and_malformed_receipts_report_honestly(self):
        root, dispatch, _ = fixture(self.repo)
        path = root / 'A_width256/status.json'
        for value in ('timing mismatch', ['timing mismatch', 'no retry']):
            goal.save(path, dict(phase='failed', errors=value))
            result = goal.status(root=root, launch_root=dispatch)
            self.assertEqual(result['phase'], 'failed')
            self.assertIn('timing mismatch', str(result['errors']))
        goal.save(path, ['not a receipt'])
        self.assertIn('Expected a JSON receipt object', str(goal.status(root=root, launch_root=dispatch)['errors']))

    def test_train_capture_null_requires_exact_official_selected_model_binding(self):
        root, dispatch, locks = fixture(self.repo)
        arm = goal.ARMS[0]
        path = root / arm / 'final_metrics.json'
        final = goal.optional(path)
        lock = locks[arm]
        final['train']['identity'].update(checkpoint_sha256=None, model_state_sha256='d'*64,
                                         successful_updates=lock['selected_update'])
        goal.save(path, final)
        self.assertFalse(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])
        final['val']['identity']['model_state_sha256'] = 'd'*64
        final['val']['checkpoint_binding'] = dict(
            rule='explicit selected-state final-checkpoint binding', checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=lock['state_hash'], model_state_sha256='d'*64, selected_update=lock['selected_update'],
            archive_identity_unchanged=True)
        goal.save(path, final)
        self.assertTrue(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])
        final['train']['identity']['model_state_sha256'] = 'wrong'
        goal.save(path, final)
        self.assertFalse(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_authorization_before_watch_compare_or_git(self):
        root, dispatch, _ = fixture(self.repo)
        with patch.object(goal.subprocess, 'run') as run:
            for action in (goal.watch, goal.compare_once, goal.publish):
                with self.assertRaises(ValueError):
                    action(root, dispatch)
        run.assert_not_called()
        self.assertFalse((root / 'goal_watch.lock').exists())

    def test_comparison_is_cpu_only_once_no_retry(self):
        root, dispatch, locks = fixture(self.repo)
        with patch.object(goal, 'require_ready'), patch.object(goal.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run:
            with self.assertRaisesRegex(RuntimeError, 'no retry'):
                goal.compare_once(root, dispatch, 'ready')
            receipt = (root / 'goal_compare_receipt.json').read_bytes()
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, dispatch, 'ready')
            self.assertEqual(receipt, (root / 'goal_compare_receipt.json').read_bytes())
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(run.call_args.args[0][-2:], ['--device', 'cpu'])

    def test_quiet_watch_compare_then_one_publish_and_terminal_source_failure(self):
        root, dispatch, locks = fixture(self.repo)
        with patch.object(goal, 'require_ready'), \
                patch.object(goal, 'compare_once', side_effect=lambda *a, **k: comparison(root, locks)) as compare, \
                patch.object(goal, 'publish') as publish, redirect_stdout(io.StringIO()):
            self.assertEqual(goal.watch(root, dispatch, 1, True, 'main'), 0)
            self.assertEqual(goal.watch(root, dispatch, 1, True, 'main'), 0)
        self.assertEqual(compare.call_count, 1)
        self.assertEqual(publish.call_count, 1)
        (root / 'goal_watch_receipt.json').unlink()
        with patch.object(goal, 'require_ready', side_effect=[None, RuntimeError('source changed')]), \
                patch.object(goal, 'publish') as publish, redirect_stdout(io.StringIO()):
            self.assertEqual(goal.watch(root, dispatch, 1, True, 'main'), 1)
        publish.assert_not_called()
        self.assertIn('source changed', goal.optional(root / 'goal_watch_receipt.json')['error'])

    def test_curated_whole_version_publication_uses_shared_mutex(self):
        root, dispatch, locks = fixture(self.repo)
        comparison(root, locks)
        git_dir = self.repo / '.git'
        git_dir.mkdir()
        commands = []

        def git_output(args, cwd):
            if args == ['rev-parse', '--show-toplevel']:
                return str(self.repo)
            if args == ['rev-parse', '--absolute-git-dir']:
                return str(git_dir)
            if args == ['branch', '--show-current']:
                return 'main'
            if args == ['rev-parse', 'HEAD']:
                return 'head'
            if args == ['diff', '--cached', '--name-only', '-z']:
                return 'one.pdf\0' if commands else ''
            raise AssertionError(args)

        def run(argv, **kwargs):
            commands.append(argv)
            return SimpleNamespace(returncode=0)

        with patch.object(goal, 'require_ready'), patch.object(goal, 'ROOT', root), \
                patch.object(goal.adapter, 'LAUNCH', dispatch), patch.object(goal.adapter, 'git_output', side_effect=git_output), \
                patch.object(goal.adapter, 'publication_paths', return_value=['one.pdf']), \
                patch.object(goal.adapter.subprocess, 'run', side_effect=run):
            first = goal.publish(root, dispatch, 'main')
            self.assertEqual(first, goal.publish(root, dispatch, 'main'))
        self.assertTrue(first['complete'])
        self.assertEqual(sum(c[:2] == ['git', 'commit'] for c in commands), 1)
        self.assertEqual(sum(c[:2] == ['git', 'push'] for c in commands), 1)
        self.assertTrue((git_dir / 'codex-publication.lock').is_file())

    def test_real_publication_allowlist_excludes_payloads_and_other_versions(self):
        root, dispatch, _ = fixture(self.repo)
        code = self.repo / 'code/V4'
        own, other, future = (code / n for n in ('r104_goal.py', 'r102_goal.py', 'r104_unrelated_future.py'))
        for path in (own, other, future):
            path.write_text('VALUE = 1\n')
        from .r104_launch import PUBLICATION_EXTRAS
        extras = [self.repo / name for name in PUBLICATION_EXTRAS]
        for path in extras:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Approved main-owned publication fixture\n')
        for arm in goal.ARMS:
            output = dispatch / ('R104_' + arm)
            goal.save(output / 'ready.json', dict(evidence='main'))
            goal.save(output / 'plan.json', dict(source_sha256={str(own): 'a', str(other): 'b'}))
        safe = root / 'A_width256/full_model_statistics/metrics.csv'
        safe.write_text('problem,n\nTSP,4500\n')
        excluded = [root / 'best.pt', root / 'prediction.json', root / 'A_width256/full_model_statistics/raw_cost.csv',
                    root / 'preparation/teacher/private.csv', root / 'preparation/sources/old.json']
        for path in excluded:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('do not publish\n')
        secret = root / 'preparation/sensitive.json'
        goal.save(secret, dict(token='fixture-not-a-real-secret'))
        linked = root / 'A_width256/full_model_statistics/linked.pdf'
        linked.symlink_to(own)
        with patch.object(goal, 'ROOT', root), patch.object(goal, 'LAUNCH', dispatch), \
                patch.object(goal, 'REPO', self.repo), patch.object(goal, 'require_ready'):
            paths = goal.publication_paths('R104', root, dispatch, self.repo)
        self.assertIn(str(own.relative_to(self.repo)), paths)
        self.assertIn(str(safe.relative_to(self.repo)), paths)
        for path in extras:
            self.assertIn(str(path.relative_to(self.repo)), paths)
        for path in [other, future, linked, secret, *excluded]:
            self.assertNotIn(str(path.relative_to(self.repo)), paths)


if __name__ == '__main__':
    unittest.main()
