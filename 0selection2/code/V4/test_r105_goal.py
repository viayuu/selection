"""Small saved-evidence CPU fixtures; all compare/Git operations are mocked."""

from contextlib import redirect_stdout
import copy
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r105_goal as goal


def routing(lock, model_sha):
    layer = dict(instances=128, valid_physical_nodes=384, mean_expert_weights=[.25]*4,
        mean_graph_mixture_entropy_nats=math.log(4),
        top1_instance_counts=[128, 0, 0, 0], positive_weight_expert_count=4,
        instance_weight_std=[0.]*4, instance_weight_range=[0.]*4, variation_estimable=True)
    return dict(experiment='R105', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], inference_rule=lock['inference_rule'],
        model_state_sha256=model_sha, split='val', first_instances_per_problem=128,
        input_only=True, selection_use=False, grouping='same problem and exact real n',
        labels_read=False, targets_read=False, model_reselection=False, routing_calibration=False,
        auxiliary_loss=False, solver_quotas=False,
        aggregation='valid physical-node mean; graph-mean mixture entropy, not mean node entropy',
        per_problem=[dict(problem=p, instances=128, sample_indices=list(range(128)),
            exact_n_counts={'3': 128}, layers=[dict(layer, layer=i) for i in range(4)],
            within_exact_n=[dict(layer, layer=i, exact_n=3) for i in range(4)]) for p in goal.PROBLEMS])


def fixture(repo):
    root = repo / 'code/V4/runs/R105_conditional_representation_experts'
    dispatch = repo / 'code/V4/runs/R98_parallel_dispatch'
    goal.save(root / 'protocol_lock.json', dict(experiment='R105', locked=True,
                                              original_fresh_trainable_sha256='a'*64))
    locks = {}
    for index, arm in enumerate(goal.ARMS):
        run, model_sha = root / arm, str(index + 5)*64
        config = dict(experiment='R105', arm=arm, seed=2, winner_rule='native FP64 ind',
            data_root='/frozen-new90k', prepared_root='/frozen-r84', rows_per_problem=dict(train=81000, val=4500, test=4500),
            candidate_pools={p: ['a', 'b'] for p in goal.PROBLEMS},
            baselines={p: dict(selection_split='train', train_rows=81000) for p in goal.PROBLEMS},
            max_successful_updates=57600, checkpoint_rule='strict minimum whole18 VAL macro_vs_sbs_pct; first exact tie',
            selected_val_binding_rule='exact_selected_checkpoint_model_state_protocol_state_and_successful_update',
            inference_rule='original score MLP utility differences and maximin; no calibration or solver quotas',
            main_protocol_lock=dict(sha256=goal.file_hash(root / 'protocol_lock.json')),
            trainable_parameter_count=3425561)
        goal.save(run / 'protocol.json', config)
        state = goal.json_hash(config)
        lock = dict(experiment='R105', arm=arm, state_hash=state, checkpoint_sha256=str(index+1)*64,
            selection_split='val', selected_without_test=True, per_problem_assembly=False,
            selected_update=1440, inference_rule=config['inference_rule'], rule=config['checkpoint_rule'])
        locks[arm] = lock
        identity = dict(state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'],
                        model_state_sha256=model_sha, successful_updates=1440)
        capture = dict(identity, checkpoint_sha256=None)
        binding = dict(rule=config['selected_val_binding_rule'], checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=state, model_state_sha256=model_sha, selected_update=1440, archive_identity_unchanged=True)
        result = dict(identity=identity, macro=dict(top1=.5, mean_cost=10., actual_regret=.1),
                      per_problem={p: dict(n=4500) for p in goal.PROBLEMS})
        final = dict(state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], selected_update=1440,
            train=dict(result, identity=dict(capture), per_problem={p: dict(n=81000) for p in goal.PROBLEMS}),
            val=dict(result, checkpoint_binding=binding, archive_capture_identity=capture))
        records = {
            'initialization.json': dict(seed=2, expert_init_seed=105002, fresh_initialization=True,
                historical_checkpoint_loaded=False, state_dict_restored=False, trainable_parameter_count=3425561,
                r84_fresh_trainable_sha256='a'*64, fresh_trainable_sha256='b'*64, trainable_sha256='b'*64,
                state_sha256='c'*64, fresh_state_sha256='c'*64, common_fresh_model_state_sha256='c'*64,
                cross_arm_complete_state_verified=True, identical_complete_fresh_parameters=True,
                inactive_router_feature_parameters=2048 if index == 0 else 0, active_functional_freedom_matched=False),
            'final_test_lock.json': lock,
            'training_complete.json': dict(successful_updates=57600, state_hash=state,
                                          updates_by_problem=dict.fromkeys(goal.PROBLEMS, 3200)),
            'final_metrics.json': final,
            'evaluation/val/selected_best/identity.json': capture,
            'test_results.json': dict(complete=True, state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], result=result),
            'routing_diagnostic.json': routing(lock, model_sha),
            'lib_stage_complete.json': dict(experiment='R105', complete=True,
                checkpoint_sha256=lock['checkpoint_sha256'], datasets=['TSPLIB', 'CVRPLIB']),
            'postprocess_status.json': dict(complete=True, phase='complete', exit_status=0),
            'inference_timing.json': dict(experiment='R105', arm=arm, complete=True, state_hash=state,
                checkpoint_sha256=lock['checkpoint_sha256'], parameter_count=3425561,
                model_state_sha256=model_sha,
                inference_rule=lock['inference_rule'], repetitions=3, warmup_batches=1,
                all_current_model_included=True, donor_present=False, cached_feature_latency_used=False,
                end_to_end_measured=False, device='cuda:0', scope='complete_standalone_model_original_GPU_resident_inputs',
                excludes='I/O, H2D, solver', per_problem=[dict(problem=p, instances=128,
                    standalone_model=dict(batch_seconds=[.128]*3, mean_seconds_per_instance=.001)) for p in goal.PROBLEMS]),
        }
        for name, record in records.items():
            goal.save(run / name, record)
        goal.save(run / 'full_model_statistics/protocol.json', dict(experiment='R105', complete=True, checkpoint_lock=lock,
            inference_timing=records['inference_timing.json'], target_identity=dict(target_count=36, sha256='d'*64)))
        for name in goal.adapter.R99_PDFS:
            (run / 'full_model_statistics' / name).write_bytes(b'%PDF-synthetic')
        goal.save(dispatch / 'specs' / ('R105_' + arm + '.json'), dict(run_id='R105_' + arm, run_dir=str(run),
            slots=[['3090_0', '3090_1'][index]], stages=[dict(name=n, device='cpu' if n == 'PDF' else 'gpu')
                for n in ('train', 'TEST', 'timing', 'LIB', 'PDF')]))
        goal.save(dispatch / ('R105_' + arm) / 'worker.json', dict(state='complete', exit_status=0,
            stages=[dict(name=n, state='complete', exit_status=0) for n in ('train', 'TEST', 'timing', 'LIB')]))
        goal.save(dispatch / ('R105_' + arm) / 'controller.json', dict(state='complete', exit_status=0,
            stages=[dict(name=n, state='complete', exit_status=0) for n in ('allocation_GPU_pipeline', 'PDF')]))
    return root, dispatch, locks


def comparison(root, locks):
    goal.save(root / 'comparison_artifacts/protocol.json', dict(experiment='R105', complete=True,
        checkpoints=locks, total_parameter_matched=True, active_parameter_matched=False,
        equal_flops_claim=False, identical_complete_fresh_parameters=True, training=False, inference=False))
    for name in ('paired_changes_B_vs_A.csv', 'target_recovery_B_vs_A.csv', 'target_recovery_summary_B_vs_A.csv'):
        (root / 'comparison_artifacts' / name).write_text('problem,n\nTSP,4500\n')


class GoalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='r105-goal-cpu-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)

    def test_complete_version_requires_both_pipelines_equal_fresh_state_not_equal_freedom(self):
        root, dispatch, locks = fixture(self.repo)
        current = goal.status(root=root, launch_root=dispatch)
        self.assertTrue(current['all_arms_ready'], current)
        self.assertFalse(current['complete'])
        comparison(root, locks)
        current = goal.status(root=root, launch_root=dispatch)
        self.assertTrue(current['complete'], current)
        self.assertTrue(current['identical_full_initialization'])
        self.assertFalse(current['goal_complete'])
        self.assertIn('A learns gate biases', current['comparison_boundary'])
        self.assertFalse((root / 'goal_progress.json').exists())
        self.assertFalse((root / goal.ARMS[0] / 'best.pt').exists())
        path = root / goal.ARMS[1] / 'initialization.json'
        init = goal.optional(path)
        init['state_sha256'] = 'wrong'
        goal.save(path, init)
        self.assertFalse(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_saved_stages_unique_val_lock_and_selected_archive_cannot_be_bypassed(self):
        root, dispatch, locks = fixture(self.repo)
        comparison(root, locks)
        arm = goal.ARMS[0]
        mutations = [
            ('final_test_lock.json', lambda r: r.update(selection_split='test')),
            ('training_complete.json', lambda r: r.update(successful_updates=57599)),
            ('final_metrics.json', lambda r: r['train']['per_problem'].pop('ATSP')),
            ('final_metrics.json', lambda r: r['val']['identity'].update(checkpoint_sha256=None)),
            ('final_metrics.json', lambda r: r['train']['identity'].update(model_state_sha256='wrong')),
            ('final_metrics.json', lambda r: r['val']['checkpoint_binding'].update(archive_identity_unchanged=False)),
            ('evaluation/val/selected_best/identity.json', lambda r: r.update(checkpoint_sha256='rewritten')),
            ('test_results.json', lambda r: r['result']['identity'].update(checkpoint_sha256='wrong')),
            ('lib_stage_complete.json', lambda r: r.update(datasets=['TSPLIB'])),
            ('inference_timing.json', lambda r: r['per_problem'][0]['standalone_model'].update(batch_seconds=[.1])),
            ('full_model_statistics/protocol.json', lambda r: r['checkpoint_lock'].update(checkpoint_sha256='wrong')),
        ]
        for relative, mutate in mutations:
            with self.subTest(receipt=relative):
                path = root / arm / relative
                original = goal.optional(path)
                changed = copy.deepcopy(original)
                mutate(changed)
                goal.save(path, changed)
                self.assertFalse(goal.status(root=root, launch_root=dispatch)['complete'], relative)
                goal.save(path, original)
        (root / goal.ARMS[1] / 'full_model_statistics/learning_curves.pdf').unlink()
        self.assertFalse(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_fixed_routing_real_node_masks_n_groups_and_no_labels_same_lock(self):
        root, dispatch, locks = fixture(self.repo)
        comparison(root, locks)
        path = root / goal.ARMS[0] / 'routing_diagnostic.json'
        original = goal.optional(path)
        mutations = [lambda r: r.update(split='test'), lambda r: r.update(labels_read=True),
            lambda r: r.update(selected_update=2880), lambda r: r.update(model_state_sha256='wrong'),
            lambda r: r['per_problem'][0].update(sample_indices=list(range(1, 129))),
            lambda r: r['per_problem'][0]['layers'][0].update(valid_physical_nodes=512),
            lambda r: r['per_problem'][0]['within_exact_n'].pop(),
            lambda r: r['per_problem'][0]['layers'][0].update(mean_graph_mixture_entropy_nats=2.)]
        for mutate in mutations:
            record = copy.deepcopy(original)
            mutate(record)
            goal.save(path, record)
            before = path.read_bytes()
            self.assertFalse(goal.status(root=root, launch_root=dispatch)['complete'])
            self.assertEqual(path.read_bytes(), before)
        goal.save(path, original)
        self.assertTrue(goal.status(root=root, launch_root=dispatch)['complete'])
        original['per_problem'][0]['layers'][0]['mean_graph_mixture_entropy_nats'] = float('nan')
        self.assertFalse(goal.route_ready(original, locks[goal.ARMS[0]]))

    def test_route_checker_accepts_actual_runner_masked_statistics_cpu_only(self):
        if not (goal.REPO / 'code/V4/r105_experiment.py').is_file():
            self.skipTest('Runner source pending')
        import torch
        from . import r105_experiment as runner
        root, _, locks = fixture(self.repo)
        lock = locks[goal.ARMS[0]]
        record = goal.optional(root / goal.ARMS[0] / 'routing_diagnostic.json')
        n = torch.tensor([1, 3] * 64)
        batch = dict(n=n, node_mask=torch.arange(3)[None] < n[:, None])
        weights = torch.full((128, 4, 4), .25)
        record['per_problem'] = [runner.routing_statistics(weights, batch, p) for p in goal.PROBLEMS]
        self.assertTrue(goal.route_ready(record, lock))

    def test_train_capture_null_only_through_exact_model_update_val_binding(self):
        root, dispatch, _ = fixture(self.repo)
        path = root / goal.ARMS[0] / 'final_metrics.json'
        original = goal.optional(path)
        self.assertIsNone(original['train']['identity']['checkpoint_sha256'])
        for changes in (dict(model_state_sha256='wrong'), dict(successful_updates=7200), dict(state_hash='wrong')):
            record = copy.deepcopy(original)
            record['train']['identity'].update(changes)
            goal.save(path, record)
            self.assertFalse(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])
        goal.save(path, original)
        self.assertTrue(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_errors_strings_lists_nested_and_malformed_schema_fail_closed(self):
        root, dispatch, _ = fixture(self.repo)
        path = root / goal.ARMS[0] / 'status.json'
        for value in ('timing mismatch', ['timing mismatch', 'no retry']):
            goal.save(path, dict(phase='failed', errors=value))
            current = goal.status(root=root, launch_root=dispatch)
            self.assertEqual(current['phase'], 'failed')
            self.assertIn('timing mismatch', str(current['errors']))
        for record in (['not a receipt'], dict(stages='invalid'), dict(stages=[dict(errors='nested failure')])):
            goal.save(path, record)
            self.assertEqual(goal.status(root=root, launch_root=dispatch)['phase'], 'failed')
        path.write_text('{not json')
        self.assertEqual(goal.status(root=root, launch_root=dispatch)['phase'], 'failed')
        self.assertFalse(goal.stages_ready(dict(state='complete', exit_status=0,
            stages=[dict(name=n, state='complete', exit_status=0) for n in ('TEST', 'train')]), ['train', 'TEST']))

    def test_authorization_precedes_watch_compare_and_git(self):
        root, dispatch, _ = fixture(self.repo)
        with patch.object(goal.subprocess, 'run') as run:
            for action in (goal.watch, goal.compare_once, goal.publish):
                with self.assertRaises(ValueError):
                    action(root, dispatch)
        run.assert_not_called()
        self.assertFalse((root / 'goal_watch.lock').exists())
        self.assertEqual(goal.parser().parse_args([]).interval, 120)
        self.assertTrue(goal.parser().parse_args(['--publish']).publish_github)

    def test_saved_cpu_comparison_only_once_after_both_complete(self):
        root, dispatch, _ = fixture(self.repo)
        with patch.object(goal, 'require_ready'), patch.object(goal.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run:
            with self.assertRaisesRegex(RuntimeError, 'no retry'):
                goal.compare_once(root, dispatch, 'main')
            receipt = (root / 'goal_compare_receipt.json').read_bytes()
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, dispatch, 'main')
            self.assertEqual(receipt, (root / 'goal_compare_receipt.json').read_bytes())
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(run.call_args.args[0][-2:], ['--device', 'cpu'])

    def test_quiet_watch_unchanged_then_compare_without_implicit_publication(self):
        root, dispatch, locks = fixture(self.repo)
        missing = root / goal.ARMS[1] / 'postprocess_status.json'
        post = goal.optional(missing)
        missing.unlink()
        output = io.StringIO()
        with patch.object(goal, 'require_ready'), \
                patch.object(goal.time, 'sleep', side_effect=lambda n: goal.save(missing, post)) as sleep, \
                patch.object(goal, 'compare_once', side_effect=lambda *a, **k: comparison(root, locks)) as compare, \
                patch.object(goal, 'publish') as publish, redirect_stdout(output):
            self.assertEqual(goal.watch(root, dispatch, evidence='main'), 0)
            self.assertEqual(goal.watch(root, dispatch, evidence='main'), 0)
        sleep.assert_called_once_with(120)
        self.assertEqual(compare.call_count, 1)
        publish.assert_not_called()
        self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_watch_explicit_publish_once_and_terminal_source_failure(self):
        root, dispatch, locks = fixture(self.repo)
        comparison(root, locks)
        with patch.object(goal, 'require_ready'), patch.object(goal, 'publish') as publish, redirect_stdout(io.StringIO()):
            self.assertEqual(goal.watch(root, dispatch, 120, True, 'main'), 0)
            self.assertEqual(goal.watch(root, dispatch, 120, True, 'main'), 0)
        self.assertEqual(publish.call_count, 1)
        (root / 'goal_watch_receipt.json').unlink()
        with patch.object(goal, 'require_ready', side_effect=[None, RuntimeError('source changed')]), \
                patch.object(goal, 'publish') as publish, redirect_stdout(io.StringIO()):
            self.assertEqual(goal.watch(root, dispatch, 120, True, 'main'), 1)
        publish.assert_not_called()
        self.assertIn('source changed', goal.optional(root / 'goal_watch_receipt.json')['error'])

    def test_curated_whole_version_publication_one_commit_shared_mutex(self):
        root, dispatch, locks = fixture(self.repo)
        comparison(root, locks)
        git_dir = self.repo / '.git'
        git_dir.mkdir()
        commands = []

        def git_output(args, cwd):
            values = {('rev-parse', '--show-toplevel'): str(self.repo),
                ('rev-parse', '--absolute-git-dir'): str(git_dir), ('branch', '--show-current'): 'main',
                ('rev-parse', 'HEAD'): 'head', ('diff', '--cached', '--name-only', '-z'): 'one.pdf\0' if commands else ''}
            return values[tuple(args)]

        with patch.object(goal, 'require_ready'), patch.object(goal, 'ROOT', root), \
                patch.object(goal.adapter, 'LAUNCH', dispatch), patch.object(goal.adapter, 'git_output', side_effect=git_output), \
                patch.object(goal.adapter, 'publication_paths', return_value=['one.pdf']), \
                patch.object(goal.adapter.subprocess, 'run', side_effect=lambda argv, **k: commands.append(argv)):
            first = goal.publish(root, dispatch, 'main')
            self.assertEqual(first, goal.publish(root, dispatch, 'main'))
        self.assertTrue(first['complete'])
        self.assertEqual(sum(c[:2] == ['git', 'commit'] for c in commands), 1)
        self.assertEqual(sum(c[:2] == ['git', 'push'] for c in commands), 1)
        self.assertTrue((git_dir / 'codex-publication.lock').is_file())

    def test_publication_named_sources_paper_not_weights_labels_logs_or_other_dirty(self):
        root, dispatch, _ = fixture(self.repo)
        code = self.repo / 'code/V4'
        own, other, future = (code / n for n in ('r105_goal.py', 'r102_goal.py', 'r105_unrelated_future.py'))
        for path in (own, other, future):
            path.write_text('VALUE = 1\n')
        from .r105_launch import PUBLICATION_EXTRAS
        extras = [self.repo / name for name in PUBLICATION_EXTRAS]
        for path in extras:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Approved papers-only rationale\n')
        for arm in goal.ARMS:
            output = dispatch / ('R105_' + arm)
            goal.save(output / 'ready.json', dict(evidence='main'))
            goal.save(output / 'plan.json', dict(source_sha256={str(own): 'a', str(other): 'b', str(future): 'c'}))
        safe = root / goal.ARMS[0] / 'full_model_statistics/metrics.csv'
        safe.write_text('problem,n\nTSP,4500\n')
        excluded = [root / 'best.pt', root / 'prediction.json', root / 'raw_labels.csv', root / 'tokens.csv',
            root / 'goal_compare.log', safe.parent / 'raw_cost.csv', root / 'preparation/teacher/private.csv',
            root / 'preparation/sources/old.json', root / 'preparation/cache/input.csv']
        for path in excluded:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('do not publish\n')
        secret = root / 'preparation/sensitive.json'
        goal.save(secret, dict(token='fixture-not-a-real-secret'))
        linked = safe.parent / 'linked.pdf'
        linked.symlink_to(own)
        with patch.object(goal, 'ROOT', root), patch.object(goal, 'LAUNCH', dispatch), \
                patch.object(goal, 'REPO', self.repo), patch.object(goal, 'require_ready'):
            paths = goal.publication_paths('R105', root, dispatch, self.repo)
        for path in [own, safe, *extras]:
            self.assertIn(str(path.relative_to(self.repo)), paths)
        for path in [other, future, linked, secret, *excluded]:
            self.assertNotIn(str(path.relative_to(self.repo)), paths)


if __name__ == '__main__':
    unittest.main()
