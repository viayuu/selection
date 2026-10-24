"""Saved-evidence fake CPU fixtures. All compare, launch and Git work is mocked."""

from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r108_goal as goal


def fixture(repo):
    root = repo / 'code/V4/runs/R108_conditional_density_ratio'
    dispatch = root / 'preparation/dispatch'
    count = goal.TRAINABLE_PARAMETERS
    goal.save(root / 'protocol_lock.json', dict(experiment='R108', locked=True, arms=list(goal.ARMS), seed=2,
        trainable_parameters=dict.fromkeys(goal.ARMS, count),
        original_fresh_trainable_sha256='a'*64))
    locks = {}
    for index, arm in enumerate(goal.ARMS):
        run, model_sha = root / arm, str(index + 5)*64
        config = dict(experiment='R108', arm=arm, seed=2, winner_rule='native FP64 ind',
            data_root='/frozen-new90k', prepared_root='/frozen-r84', rows_per_problem=dict(train=81000, val=4500, test=4500),
            candidate_pools={p: ['a', 'b'] for p in goal.PROBLEMS},
            baselines={p: dict(selection_split='train', train_rows=81000) for p in goal.PROBLEMS},
            max_successful_updates=57600, checkpoint_rule='strict minimum whole18 VAL macro_vs_sbs_pct; first exact tie',
            selected_val_binding_rule='exact_selected_checkpoint_model_state_protocol_state_and_successful_update',
            inference_rule='original score MLP utility differences and maximin; no calibration or solver quotas',
            main_protocol_lock=dict(sha256=goal.file_hash(root / 'protocol_lock.json')),
            trainable_parameter_count=count, eval_batch_size=256,
            prior_spec=dict(condition=['problem', 'exact_true_n']), priors_sha256='e'*64,
            priors_path=str(root / 'prepared/priors.json'),
            loss=dict(ce_or_density_ratio=1., comparison=0., risk=0., kl=0., dropout_passes=1),
            microbatch_caps={p: 512 if p in ('TSP', 'CVRP') else 1024 for p in goal.PROBLEMS})
        goal.save(run / 'protocol.json', config)
        state = goal.json_hash(config)
        lock = dict(experiment='R108', arm=arm, state_hash=state, checkpoint_sha256=str(index+1)*64,
            selection_split='val', selected_without_test=True, per_problem_assembly=False,
            selected_update=1440, inference_rule=config['inference_rule'], rule=config['checkpoint_rule'])
        locks[arm] = lock
        identity = dict(experiment='R108', arm=arm, split='val', state_hash=state,
            checkpoint_sha256=lock['checkpoint_sha256'], model_state_sha256=model_sha, successful_updates=1440)
        capture = dict(identity, checkpoint_sha256=None)
        binding = dict(rule=config['selected_val_binding_rule'], checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=state, model_state_sha256=model_sha, selected_update=1440,
            archive_capture_checkpoint_sha256=None, archive_identity_unchanged=True)
        result = dict(identity=identity, macro=dict(top1=.5, mean_cost=10., actual_regret=.1),
                      per_problem={p: dict(n=4500) for p in goal.PROBLEMS})
        final = dict(state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], selected_update=1440,
            train=dict(result, identity=dict(capture, split='train'), per_problem={p: dict(n=81000) for p in goal.PROBLEMS}),
            val=dict(result, checkpoint_binding=binding, archive_capture_identity=capture))
        records = {
            'initialization.json': dict(seed=2, fresh_initialization=True,
                historical_checkpoint_loaded=False, state_dict_restored=False, trainable_parameter_count=count,
                fresh_trainable_sha256='a'*64, trainable_sha256='a'*64,
                state_sha256='c'*64, fresh_state_sha256='c'*64, common_fresh_model_state_sha256='c'*64,
                cross_arm_complete_state_verified=True, identical_complete_fresh_parameters=True,
                active_functional_freedom_matched=False, new_trainable_parameter_count=0,
                prior_fit_split='train', priors_sha256='e'*64),
            'final_test_lock.json': lock,
            'training_complete.json': dict(successful_updates=57600, state_hash=state,
                                          updates_by_problem=dict.fromkeys(goal.PROBLEMS, 3200)),
            'final_metrics.json': final,
            'evaluation/val/selected_best/identity.json': capture,
            'test_results.json': dict(complete=True, state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], result=result),
            'lib_stage_complete.json': dict(experiment='R108', complete=True,
                checkpoint_sha256=lock['checkpoint_sha256'], datasets=['TSPLIB', 'CVRPLIB']),
            'postprocess_status.json': dict(complete=True, phase='complete', exit_status=0),
            'inference_timing.json': dict(experiment='R108', arm=arm, complete=True, state_hash=state,
                checkpoint_sha256=lock['checkpoint_sha256'], parameter_count=count, selected_update=1440,
                model_state_sha256=model_sha, input_labels_read=False,
                inference_rule=lock['inference_rule'], repetitions=3, warmup_batches=1,
                all_current_model_included=True, prior_lookup_and_residual_computation_included=True,
                donor_present=False, cached_feature_latency_used=False,
                end_to_end_measured=False, device='cuda:0', scope='complete_standalone_model_original_GPU_resident_inputs',
                excludes='I/O, H2D, solver', per_problem=[dict(problem=p, instances=128,
                    standalone_model=dict(batch_seconds=[.128]*3, mean_seconds_per_instance=.001)) for p in goal.PROBLEMS]),
        }
        for name, record in records.items():
            goal.save(run / name, record)
        goal.save(run / 'full_model_statistics/protocol.json', dict(experiment='R108', complete=True, checkpoint_lock=lock,
            prior_spec=config['prior_spec'], prior_provenance=dict(sha256='e'*64, fit_split='train'),
            inference_timing=records['inference_timing.json'], target_identity=dict(target_count=36, sha256='d'*64)))
        (run / 'full_model_statistics/nss_comparison.csv').write_text(
            'dataset,current_checkpoint_sha256,current_top1_native_pct,current_mean_cost,nss_top1_native_pct,nss_mean_cost,status,data_version,candidate_pool,winner_rule\n'
            + ''.join(f'{d},{lock["checkpoint_sha256"]},50,10,50,10,complete,NEW90k,synthetic,native\n'
                for d in ('TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB')))
        for name in goal.adapter.R99_PDFS:
            (run / 'full_model_statistics' / name).write_bytes(b'%PDF-synthetic')
        goal.save(dispatch / 'specs' / ('R108_' + arm + '.json'), dict(run_id='R108_' + arm, run_dir=str(run),
            spec_draft=False, slots=[['4090_0', '4090_1'][index]],
            stages=[dict(name=n, device='cpu' if n == 'PDF' else 'gpu') for n in ('train', 'TEST', 'timing', 'LIB', 'PDF')]))
        for filename, names in (('worker.json', ('train', 'TEST', 'timing', 'LIB')),
                                ('controller.json', ('allocation_GPU_pipeline', 'PDF'))):
            goal.save(dispatch / ('R108_' + arm) / filename, dict(state='complete', exit_status=0,
                stages=[dict(name=n, state='complete', exit_status=0) for n in names]))
    return root, dispatch, locks


def comparison(root, locks):
    goal.save(root / 'comparison_artifacts/protocol.json', dict(experiment='R108', complete=True,
        checkpoints=locks, total_parameter_matched=True, active_parameter_matched=False,
        equal_gradient_scale_claim=False, priors_sha256='e'*64,
        equal_flops_claim=False, identical_complete_fresh_parameters=True, training=False, inference=False))
    for name in ('paired_changes_B_vs_A.csv', 'target_recovery_B_vs_A.csv', 'target_recovery_summary_B_vs_A.csv'):
        (root / 'comparison_artifacts' / name).write_text('problem,n\nTSP,4500\n')


class GoalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='r108-goal-fake-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)

    def test_whole_version_completion_no_routing_or_efficacy_claim(self):
        root, dispatch, locks = fixture(self.repo)
        current = goal.status(root=root, launch_root=dispatch)
        self.assertTrue(current['all_arms_ready'], current)
        self.assertFalse(current['complete'])
        comparison(root, locks)
        current = goal.status(root=root, launch_root=dispatch)
        self.assertTrue(current['complete'], current)
        self.assertTrue(current['identical_full_initialization'])
        self.assertFalse(current['goal_complete'])
        self.assertIn('not evidence of selection improvement', current['comparison_boundary'])
        self.assertNotIn('routing_diagnostic.json', goal.RECORD_NAMES)
        self.assertFalse((root / 'goal_progress.json').exists())
        self.assertEqual(goal.adapter.LAUNCH, goal.LAUNCH)
        self.assertFalse((root / goal.ARMS[0] / 'best.pt').exists())
        with self.assertRaises(ValueError):
            goal.status('R105', root, dispatch)

    def test_locks_complete_budget_bound_train_val_test_lib_timing_and_pdf_fail_closed(self):
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
            ('final_metrics.json', lambda r: r['val']['checkpoint_binding'].update(archive_capture_checkpoint_sha256='wrong')),
            ('evaluation/val/selected_best/identity.json', lambda r: r.update(checkpoint_sha256='rewritten')),
            ('test_results.json', lambda r: r['result']['identity'].update(checkpoint_sha256='wrong')),
            ('lib_stage_complete.json', lambda r: r.update(datasets=['TSPLIB'])),
            ('inference_timing.json', lambda r: r.update(prior_lookup_and_residual_computation_included=False)),
            ('inference_timing.json', lambda r: r.update(parameter_count=1)),
            ('inference_timing.json', lambda r: r.update(model_state_sha256='wrong')),
            ('inference_timing.json', lambda r: r['per_problem'][0]['standalone_model'].update(batch_seconds=[.1])),
            ('full_model_statistics/protocol.json', lambda r: r['checkpoint_lock'].update(checkpoint_sha256='wrong')),
            ('initialization.json', lambda r: r.update(historical_checkpoint_loaded=True)),
            ('initialization.json', lambda r: r.update(prior_fit_split='val')),
            ('initialization.json', lambda r: r.update(priors_sha256='wrong')),
            ('initialization.json', lambda r: r.update(active_functional_freedom_matched=True)),
            ('full_model_statistics/protocol.json', lambda r: r['prior_provenance'].update(sha256='wrong')),
            ('initialization.json', lambda r: r.update(state_sha256='wrong')),
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

    def test_val_capture_allows_exact_checkpoint_sha_but_not_wrong_binding(self):
        root, dispatch, locks = fixture(self.repo)
        arm, lock = goal.ARMS[0], locks[goal.ARMS[0]]
        path = root / arm / 'final_metrics.json'
        final = goal.optional(path)
        final['val']['archive_capture_identity']['checkpoint_sha256'] = lock['checkpoint_sha256']
        final['val']['checkpoint_binding']['archive_capture_checkpoint_sha256'] = lock['checkpoint_sha256']
        goal.save(root / arm / 'evaluation/val/selected_best/identity.json', final['val']['archive_capture_identity'])
        goal.save(path, final)
        self.assertTrue(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])
        final['val']['archive_capture_identity']['checkpoint_sha256'] = 'wrong'
        goal.save(path, final)
        self.assertFalse(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_errors_nested_scalar_invalid_json_and_rc0_stage_order_are_preserved(self):
        root, dispatch, _ = fixture(self.repo)
        path = root / goal.ARMS[0] / 'status.json'
        for record in (dict(phase='failed', errors='timing mismatch'), ['invalid'],
                       dict(stages='invalid'), dict(stages=[dict(error='nested failure')])):
            goal.save(path, record)
            self.assertEqual(goal.status(root=root, launch_root=dispatch)['phase'], 'failed')
        path.write_text('{not json')
        self.assertEqual(goal.status(root=root, launch_root=dispatch)['phase'], 'failed')
        self.assertFalse(goal.stages_ready(dict(state='complete', exit_status=0,
            stages=[dict(name=n, state='complete', exit_status=0) for n in ('TEST', 'train')]), ['train', 'TEST']))
        for name in ('worker.json', 'controller.json'):
            goal.save(dispatch / ('R108_' + goal.ARMS[0]) / name, dict(state='complete', exit_status=False))
            self.assertFalse(goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_no_authorization_means_no_watch_compare_git_or_lock_write(self):
        root, dispatch, _ = fixture(self.repo)
        with patch.object(goal.subprocess, 'run') as run:
            for action in (goal.watch, goal.compare_once, goal.publish):
                with self.assertRaises(ValueError):
                    action(root, dispatch)
        run.assert_not_called()
        self.assertFalse((root / 'goal_watch.lock').exists())
        self.assertEqual(goal.parser().parse_args([]).interval, 120)
        self.assertTrue(goal.parser().parse_args(['--publish']).publish_github)

    def test_real_require_ready_reads_both_frozen_plans_and_exact_roots(self):
        from . import r108_launch as launch
        root, dispatch, _ = fixture(self.repo)

        def ready(path):
            arm = path.stem.removeprefix('R108_')
            return dict(run_dir=str(root / arm)), dict(evidence='main')

        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch, 'ROOT', dispatch), \
                patch.object(launch, 'check_ready', side_effect=ready) as check:
            goal.require_ready(root, dispatch, 'main')
            self.assertEqual(check.call_count, 2)
            with self.assertRaisesRegex(RuntimeError, 'match main-ready'):
                goal.require_ready(root, dispatch, 'wrong')
            with self.assertRaisesRegex(RuntimeError, 'canonical'):
                goal.require_ready(self.repo, dispatch, 'main')

    def test_compare_requires_both_complete_is_cpu_only_and_failure_is_one_shot(self):
        root, dispatch, _ = fixture(self.repo)
        with patch.object(goal, 'require_ready'), \
                patch.object(goal.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run:
            with self.assertRaisesRegex(RuntimeError, 'no retry'):
                goal.compare_once(root, dispatch, 'main')
            receipt = (root / 'goal_compare_receipt.json').read_bytes()
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, dispatch, 'main')
            self.assertEqual(receipt, (root / 'goal_compare_receipt.json').read_bytes())
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(run.call_args.args[0][-2:], ['--device', 'cpu'])

    def test_watch_waits_then_compares_and_only_explicitly_publishes_once(self):
        root, dispatch, locks = fixture(self.repo)
        missing = root / goal.ARMS[1] / 'postprocess_status.json'
        post = goal.optional(missing)
        missing.unlink()
        with patch.object(goal, 'require_ready'), \
                patch.object(goal.time, 'sleep', side_effect=lambda n: goal.save(missing, post)) as sleep, \
                patch.object(goal, 'compare_once', side_effect=lambda *a, **k: comparison(root, locks)) as compare, \
                patch.object(goal, 'publish') as publish, redirect_stdout(io.StringIO()):
            self.assertEqual(goal.watch(root, dispatch, evidence='main'), 0)
            self.assertEqual(goal.watch(root, dispatch, evidence='main'), 0)
        sleep.assert_called_once_with(120)
        self.assertEqual(compare.call_count, 1)
        publish.assert_not_called()
        (root / 'goal_watch_receipt.json').unlink()
        with patch.object(goal, 'require_ready'), patch.object(goal, 'publish') as publish, redirect_stdout(io.StringIO()):
            self.assertEqual(goal.watch(root, dispatch, 120, True, 'main'), 0)
            self.assertEqual(goal.watch(root, dispatch, 120, True, 'main'), 0)
        self.assertEqual(publish.call_count, 1)

    def test_watch_source_failure_stops_before_compare_or_publish(self):
        root, dispatch, locks = fixture(self.repo)
        comparison(root, locks)
        with patch.object(goal, 'require_ready', side_effect=[None, RuntimeError('source changed')]), \
                patch.object(goal, 'publish') as publish, redirect_stdout(io.StringIO()):
            self.assertEqual(goal.watch(root, dispatch, 120, True, 'main'), 1)
        publish.assert_not_called()
        self.assertIn('source changed', goal.optional(root / 'goal_watch_receipt.json')['error'])

    def test_curated_version_publication_one_commit_shared_mutex_all_git_mocked(self):
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

    def test_publication_exact_r108_sources_no_previous_dirty_data_weights_or_credentials(self):
        root, dispatch, _ = fixture(self.repo)
        code = self.repo / 'code/V4'
        own, witness, wrapper, other, future = (code / n for n in
            ('r108_goal.py', 'r108_memory_witness.py', 'run_v4_r108_witness.sh',
             'r106_goal.py', 'r108_unrelated_future.py'))
        for path in (own, witness, wrapper, other, future):
            path.write_text('VALUE = 1\n')
        for arm in goal.ARMS:
            output = dispatch / ('R108_' + arm)
            goal.save(output / 'ready.json', dict(evidence='main'))
            goal.save(output / 'plan.json', dict(source_sha256={str(p): 'a' for p in (own, witness, wrapper, other, future)}))
        safe = root / goal.ARMS[0] / 'full_model_statistics/metrics.csv'
        safe.write_text('problem,n\nTSP,4500\n')
        excluded = [root / 'best.pt', root / 'prediction.json', root / 'raw_labels.csv', root / 'tokens.csv',
            root / 'goal_compare.log', safe.parent / 'raw_cost.csv', root / 'preparation/teacher/private.csv',
            root / 'preparation/sources/old.json', root / 'preparation/cache/input.csv']
        for path in excluded:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('do not publish\n')
        secret = root / 'preparation/sensitive.json'
        goal.save(secret, dict(token='fake-not-a-real-secret'))
        linked = safe.parent / 'linked.pdf'
        linked.symlink_to(own)
        with patch.object(goal, 'ROOT', root), patch.object(goal, 'LAUNCH', dispatch), \
                patch.object(goal, 'REPO', self.repo), patch.object(goal, 'require_ready'):
            paths = goal.publication_paths('R108', root, dispatch, self.repo)
        for path in (own, witness, wrapper, safe):
            self.assertIn(str(path.relative_to(self.repo)), paths)
        for path in [other, future, linked, secret, *excluded]:
            self.assertNotIn(str(path.relative_to(self.repo)), paths)


if __name__ == '__main__':
    unittest.main()
