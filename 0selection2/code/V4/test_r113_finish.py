"""CPU-only R113 closeout contracts. Git is mocked, never run by this suite."""

from contextlib import contextmanager
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import r113_finish as finish
from .test_r112_finish import model_fixture


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def predictions(n=6, new=False):
    return dict(costs=np.tile(np.array([[1., 2., 3.]], dtype=np.float64), (n, 1)),
        pred=np.array(([0, 1, 2, 1, 0, 2] if new else [1, 0, 1, 2, 0, 2])*(n//6)),
        winner=np.zeros(n, dtype=np.int64), pool=['s0', 's1', 's2'], pool_ids=np.arange(3), indices=np.arange(n))


def wave(root):
    jobs = []
    controller = {}
    for arm in finish.ARMS:
        directory = root / 'preparation/wave_dispatch' / arm / 'attempt_0001'
        jobs.append(dict(job_id=arm))
        controller[arm] = dict(state='complete', exit_status=0, attempts=1,
            attempt_dir=str(directory), slot='synthetic', gpu_uuid='synthetic')
        stages = ['capacity']+(['pretrain'] if arm != finish.ARMS[0] else [])+['train', 'finalize']
        put(directory / 'worker.json', dict(job=arm, slot='synthetic', gpu_uuid='synthetic', state='complete', exit_status=0,
            stages=[dict(name=s, state='complete', exit_status=0) for s in stages]))
    jobs.append(dict(job_id='R114_labels'))
    controller['R114_labels'] = dict(state='failed', exit_status=1)
    put(finish.wave_plan(root), dict(root=str(root), jobs=jobs))
    put(root / 'preparation/wave_dispatch/controller.json', dict(state='failed', jobs=controller))


def full_fixture(root, v4):
    root.mkdir(parents=True, exist_ok=True)
    for name in finish.CODE_NAMES:
        path = v4 / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source\n', encoding='utf-8')
    source = {n: finish.digest(v4 / n) for n in finish.CODE_NAMES if n.endswith('.py') and not n.startswith('test_')}
    process = root / 'process_train_v1'
    put(process / 'schema.json', {'synthetic': True})
    put(process / 'source_manifest.json', dict(split='train', seed=2))
    put(process / 'run_config.json', dict(split='train', seed=2))
    put(process / 'manifest.json', dict(status='complete', split='train', seed=2, rows=18*81000,
        source_manifest_sha256=finish.digest(process / 'source_manifest.json')))
    put(process / 'completion_receipt.json', dict(status='complete',
        manifest_sha256=finish.digest(process / 'manifest.json'), per_problem_rows=dict.fromkeys(finish.PROBLEMS, 81000),
        source_manifest_sha256=finish.digest(process / 'source_manifest.json'), schema_sha256=finish.digest(process / 'schema.json')))
    pretrain_fields = dict(arms=list(finish.ARMS[1:]), logical_global_batch=512, max_successful_updates=8640,
        teacher_forcing=False, processor_only=True)
    main = dict(experiment='R113', locked=True, arms=list(finish.ARMS), seed=2, logical_global_batch=512,
        max_successful_updates=57600, pretraining=pretrain_fields, pretrain_root=str(root / 'pretrain'),
        process_manifest_sha256=finish.digest(process / 'manifest.json'),
        checkpoint_rule='synthetic VAL-only locked rule', inference_rule='stable argmax',
        microbatch_caps=dict.fromkeys(finish.PROBLEMS, 384), pretrain_microbatch_caps=dict.fromkeys(finish.PROBLEMS, 512),
        eval_batch_size=512)
    main_path = root / 'protocol_lock.json'
    put(main_path, main)
    binding = dict(path=str(main_path), record=main, sha256=finish.digest(main_path))
    dataset = dict(manifest_sha256=main['process_manifest_sha256'], split='train', rows=18*81000,
        source_manifest_sha256=finish.digest(process / 'source_manifest.json'),
        tensor_schema_sha256=finish.digest(process / 'schema.json'))
    models = {}
    for arm in finish.ARMS:
        run = root / arm
        config = model_fixture(run, main_path, 'A' if arm == finish.ARMS[0] else 'B', 'R113', arm)
        initial = dict(fresh_initialization=True, historical_checkpoint_loaded=False,
            fresh_state_sha256='same_fresh_state', fresh_trainable_sha256='same_fresh_parameters',
            fresh_nonprocessor_sha256='same_nonprocessor', initial_processor_sha256='same_processor',
            parameter_layout_sha256='same_layout', fresh_reference_sha256='same_reference',
            trainable_parameter_count=3, trainable_sha256='same_fresh_parameters' if arm == finish.ARMS[0] else arm+'_initial',
            processor_pretraining_applied=arm != finish.ARMS[0], processor_pretraining=None)
        config.update(source_sha256=source, main_protocol_lock=binding, deployment_mode='transition',
            initialization='same fresh full model, new own processor only',
            fresh_state_sha256=initial['fresh_state_sha256'], fresh_trainable_sha256=initial['fresh_trainable_sha256'],
            fresh_nonprocessor_sha256=initial['fresh_nonprocessor_sha256'],
            initial_parameter_layout_sha256=initial['parameter_layout_sha256'],
            original_fresh_reference_sha256=initial['fresh_reference_sha256'],
            initial_trainable_sha256=initial['trainable_sha256'], current_query_labels_in_forward=False,
            model_spec=dict(deployment='self_initialized_predicted_feedback'), grad_clip=1., processor_pretraining=None)
        if arm != finish.ARMS[0]:
            mode = 'snapshot' if arm == 'B_snapshot' else 'transition'
            pre = dict(experiment='R113_pretrain', arm=arm, seed=2, **pretrain_fields, mode=mode,
                main_protocol_lock=binding, dataset=dataset, initial_full_state_sha256=initial['fresh_state_sha256'],
                initial_processor_sha256=initial['initial_processor_sha256'],
                processor_parameter_schema={'synthetic': True}, task_order_sha256='same_order', source_sha256=source,
                microbatch_caps=main['pretrain_microbatch_caps'], query_labels_in_forward=False, native_solver_calls=0)
            put(root / 'pretrain' / arm / 'protocol.json', pre)
            done = dict(experiment='R113_pretrain', arm=arm, complete=True, successful_updates=8640,
                state_hash=finish.json_hash(pre), processor_sha256=arm+'_processor', processor_file_sha256=arm+'_file',
                dataset=dataset, source_sha256=source, updates_by_problem=dict.fromkeys(finish.PROBLEMS, 480),
                initial_full_state_sha256=initial['fresh_state_sha256'], initial_processor_sha256=initial['initial_processor_sha256'])
            put(root / 'pretrain' / arm / 'completion.json', done)
            applied = dict(file_sha256=done['processor_file_sha256'], state_hash=done['state_hash'],
                processor_sha256=done['processor_sha256'], successful_updates=8640, mode=mode, dataset=dataset, source_sha256=source)
            config['processor_pretraining'] = applied
            initial['processor_pretraining'] = applied
        put(run / 'initialization.json', initial)
        put(run / 'protocol.json', config)
        state_hash = finish.json_hash(config)
        lock = finish.read(run / 'final_test_lock.json')
        lock['state_hash'] = state_hash
        put(run / 'final_test_lock.json', lock)
        final = finish.read(run / 'final_metrics.json')
        test = finish.read(run / 'test_results.json')
        for document in (final, test):
            document['state_hash'] = state_hash
        for split, item in (('VAL', final['val']), ('TEST', test['result'])):
            item['identity']['state_hash'] = state_hash
            put(Path(item['prediction_archive']).parent / 'identity.json', item['identity'])
        final['test'] = test['result']
        put(run / 'final_metrics.json', final)
        put(run / 'test_results.json', test)
        stats = finish.read(run / 'full_model_statistics/protocol.json')
        stats['checkpoint_lock'] = lock
        put(run / 'full_model_statistics/protocol.json', stats)
        for name in ('RESULTS.md', 'metrics_by_problem.csv', 'nss_comparison.csv', 'selection_distribution.csv'):
            (run / 'full_model_statistics' / name).write_text('synthetic report\n', encoding='utf-8')
        identity = dict(experiment='R113', arm=arm, complete=True,
            **{k: lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')})
        for name in ('report_complete.json', 'lib_stage_complete.json', 'inference_timing.json'):
            put(run / name, identity)
        for lib in ('TSPLIB', 'CVRPLIB'):
            path = run / 'lib_evaluation' / (lib+'_results.json')
            row = finish.read(path)
            row['protocol']['state_hash'] = state_hash
            put(path, row)
        models[arm] = config
    frozen = dict(experiment='R113', complete=True, main_protocol_lock_sha256=binding['sha256'],
        policy=dict(split='val', rows_per_problem=64), sources={p: {} for p in finish.PROBLEMS})
    frozen['subset_sha256'] = finish.json_hash(frozen)
    put(root / 'process_diagnostic_inputs.json', frozen)
    for arm in finish.ARMS:
        run = root / arm
        lock = finish.read(run / 'final_test_lock.json')
        identity = dict(experiment='R113', arm=arm, complete=True, version='synthetic_postlock',
            diagnostic_source_sha256=finish.digest(v4 / 'r113_diagnostics.py'),
            **{k: lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')})
        rows = [dict(problem=p, component=c, field='ALL', valid_count=10,
                     accuracy=.5, normalized_mse=.1, precision=None, recall=.5, all_zero_accuracy=.8)
                for p in finish.PROBLEMS for c in ('scalar', 'binary', 'pointer', 'compatible', 'reached', 'frontier')]
        data = dict(identity, inputs=frozen, policy=frozen['policy'], metrics=rows,
            model_reselection=False, teacher_in_model_input=False, teacher_used_for_training=False,
            cost_or_winner_arrays_opened=False, native_solver_calls=0,
            model_state_sha256='synthetic_state_'+arm)
        files = {}
        for suffix in ('json', 'csv', 'pdf'):
            path = run / 'process_diagnostics' / ('process_diagnostics.'+suffix)
            if suffix == 'json':
                put(path, data)
            else:
                path.write_bytes(b'%PDF-1.4\nsynthetic\n%%EOF\n' if suffix == 'pdf' else b'synthetic,csv\n')
            files[suffix] = dict(path=str(path), sha256=finish.digest(path))
        put(run / 'process_diagnostics_complete.json', dict(identity, subset_sha256=frozen['subset_sha256'], files=files))
    wave(root)
    plan = finish.read(finish.wave_plan(root))
    sources = {str(v4 / n): h for n, h in source.items()}
    sources[str(v4 / 'run_v4_r113_wave.sh')] = finish.digest(v4 / 'run_v4_r113_wave.sh')
    sources[str(main_path)] = finish.digest(main_path)
    sources[str(finish.wave_plan(root))] = finish.digest(finish.wave_plan(root))
    put(root / 'preparation/wave_dispatch/ready.json', dict(main_approved=True, runner_tests_passed=True,
        plan=dict(plan, plan_path=str(finish.wave_plan(root))), source_sha256=sources,
        source_set_sha256=finish.json_hash(sources), protocol_sha256=finish.digest(main_path)))
    for arm in finish.ARMS:
        directory = root / 'preparation/wave_dispatch' / arm / 'attempt_0001'
        for n in ('entry.json', 'request.json'):
            put(directory / n, dict(job=arm))
        put(root / 'preparation' / ('capacity_'+arm+'.json'), dict(complete=True))
    target_path = root / 'synthetic_targets.json'
    put(target_path, dict(identity=dict(data_root=models[finish.ARMS[0]]['data_root'], threshold_chosen_pct=.1, rule='fixed VAL'),
        targets=[dict(problem=p, method=s) for p in finish.PROBLEMS for s in ('s0', 's1')]))
    return models, target_path


class FinishTests(unittest.TestCase):
    def test_single_publication_parent_advance_and_no_retry(self):
        for failure in (None, 'commit', 'push', 'index'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                root = repo / '0selection2/code/V4/runs/R113_executable_process'
                root.mkdir(parents=True)
                artifact = root / 'RESULTS.md'
                artifact.write_text('synthetic complete R113\n')
                unrelated = repo / 'dirty_R77.py'
                unrelated.write_text('must remain untouched\n')
                name = str(artifact.relative_to(repo))
                calls, state = [], dict(head='old_parent', staged=[])

                @contextmanager
                def mutex():
                    state['head'] = 'r112_published_while_r113_waited'
                    yield repo / '.git/codex-publication.lock'

                def preflight(_):
                    return dict(paths=[name], excluded=[], parent=state['head'], remote='origin',
                        merge='refs/heads/main', branch='main', sha256={name: finish.digest(artifact)})

                def fake_git(*args, **kwargs):
                    calls.append(args)
                    if args[:2] == ('rev-parse', 'HEAD'):
                        return state['head']
                    if args[:2] == ('rev-parse', 'HEAD^'):
                        return 'r112_published_while_r113_waited'
                    if args[0] == 'rev-parse':
                        return 'synthetic_blob'
                    if args[0] == 'diff':
                        return 'dirty_R77.py' if failure == 'index' else '\0'.join(state['staged'])
                    if args[0] == 'add':
                        state['staged'] = list(args[args.index('--')+1:])
                    if args[0] == 'commit':
                        if failure == 'commit':
                            raise RuntimeError('synthetic failed commit')
                        self.assertIn('--only', args)
                        state['head'] = 'new_r113_commit'
                    if args[0] == 'diff-tree':
                        return name+'\n'+str((root / 'publish_manifest.json').relative_to(repo))
                    if args[0] == 'push' and failure == 'push':
                        raise RuntimeError('synthetic uncertain push')
                    return ''

                with patch.object(finish, 'GIT_ROOT', repo), patch.object(finish, 'publication_mutex', mutex), \
                        patch.object(finish, 'publication_preflight', side_effect=preflight), patch.object(finish, 'git', side_effect=fake_git):
                    if failure:
                        with self.assertRaises((RuntimeError, ValueError)):
                            finish.publish(root)
                        count = len(calls)
                        with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
                            finish.publish(root)
                        self.assertEqual(len(calls), count)
                    else:
                        receipt = finish.publish(root)
                        self.assertEqual(receipt['parent'], 'r112_published_while_r113_waited')
                        self.assertEqual((receipt['commit_attempts'], receipt['push_attempts']), (1, 1))
                        self.assertEqual(finish.publish(root)['commit'], 'new_r113_commit')
                    self.assertEqual(sum(c[0] == 'commit' for c in calls), 0 if failure == 'index' else 1)
                    self.assertEqual(sum(c[0] == 'push' for c in calls), 1 if failure in (None, 'push') else 0)
                    self.assertEqual(unrelated.read_text(), 'must remain untouched\n')
                    self.assertFalse(any(c[0] in ('reset', 'checkout') or '-A' in c for c in calls))

    def test_preflight_index_guard_never_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            root = repo / '0selection2/code/V4/runs/R113_executable_process'
            root.mkdir(parents=True)

            def fake_git(*args, **kwargs):
                if args == ('rev-parse', '--show-toplevel'):
                    return str(repo)
                if args == ('diff', '--cached', '--name-only', '-z'):
                    return 'dirty_R77.py\0'
                raise AssertionError('Index guard must stop before '+str(args))

            with patch.object(finish, 'GIT_ROOT', repo), patch.object(finish, 'verify_delivery', return_value={}), \
                    patch.object(finish, 'publication_paths', return_value=([], [])), patch.object(finish, 'git', side_effect=fake_git):
                with self.assertRaisesRegex(ValueError, 'index untouched'):
                    finish.publication_preflight(root)

    def test_saved_only_full18_pdf_history_and_mutable_r114(self):
        with tempfile.TemporaryDirectory() as directory:
            v4 = Path(directory) / 'repo/0selection2/code/V4'
            root = v4 / 'runs/R113_executable_process'
            configs, targets = full_fixture(root, v4)
            refs = {'R94B': root.parent / 'reference94', 'R89B': root.parent / 'reference89'}
            for name, experiment, arm in (('R94B', 'R94', 'B'), ('R89B', 'R89', 'B_muon_hybrid')):
                model_fixture(refs[name], root / 'protocol_lock.json', 'A', experiment, arm)
            with patch.object(finish, 'V4', v4), patch.object(finish, 'GIT_ROOT', Path(directory) / 'repo'), \
                    patch.object(finish, 'git', side_effect=AssertionError('No Git during analyze')):
                receipt = finish.analyze(root, refs, target_path=targets)
                self.assertTrue(receipt['complete'])
                self.assertEqual([r['status'] for r in receipt['historical']], ['complete', 'complete'])
                self.assertEqual(receipt['comparison_rows'], 2*19*9+6)
                self.assertTrue((root / 'RESULTS.pdf').read_bytes().startswith(b'%PDF-'))
                with (root / 'comparison.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual({r['comparison'] for r in rows}, {
                    'B_snapshot vs A_scratch', 'C_transition vs A_scratch', 'C_transition vs B_snapshot',
                    *[a+' vs '+r for a in finish.ARMS for r in ('R94B', 'R89B')]})
                controller_path = root / 'preparation/wave_dispatch/controller.json'
                controller = finish.read(controller_path)
                controller['jobs']['R114_labels'] = dict(state='running')
                controller['state'] = 'running'
                put(controller_path, controller)
                self.assertEqual(finish.verify_delivery(root)['outputs'], receipt['outputs'])
                paths, excluded = finish.publication_paths(root)
                self.assertFalse(excluded)
                self.assertTrue(all(p.suffix not in ('.pt', '.npz', '.npy') for p in paths))
                self.assertNotIn(controller_path, paths)
                self.assertFalse(any('R114' in str(p) for p in paths))
                self.assertIn(root / 'process_diagnostic_inputs.json', paths)
                self.assertIn(root / 'C_transition/process_diagnostics/process_diagnostics.pdf', paths)
                (root / 'metrics.csv').write_text('tampered')
                with self.assertRaises(ValueError):
                    finish.verify_delivery(root)

    def test_boundary_fresh_not_postpretrained_and_freeze_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            v4 = Path(directory) / 'repo/0selection2/code/V4'
            root = v4 / 'runs/R113_executable_process'
            configs, _ = full_fixture(root, v4)
            with patch.object(finish, 'V4', v4), patch.object(finish, 'GIT_ROOT', Path(directory) / 'repo'):
                models = {a: finish.load_run(root / a, {}, current=False, expected=('R113', a)) for a in finish.ARMS}
                boundary = finish.science_boundary(root, models, {})
                self.assertEqual(boundary['logical_batch'], 512)
                self.assertNotEqual(configs['A_scratch']['initial_trainable_sha256'], configs['B_snapshot']['initial_trainable_sha256'])
                path = root / 'C_transition/initialization.json'
                initial = finish.read(path)
                initial['fresh_state_sha256'] = 'not_same_fresh_model'
                put(path, initial)
                with self.assertRaises(ValueError):
                    finish.science_boundary(root, models, {})
                initial['fresh_state_sha256'] = 'same_fresh_state'
                put(path, initial)
                (v4 / 'r113_model.py').write_text('# changed after launch\n')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    finish.science_boundary(root, models, {})

    def test_postlock_wrong_checkpoint_or_missing_component(self):
        with tempfile.TemporaryDirectory() as directory:
            v4 = Path(directory) / 'repo/0selection2/code/V4'
            root = v4 / 'runs/R113_executable_process'
            full_fixture(root, v4)
            with patch.object(finish, 'V4', v4):
                model = finish.load_run(root / 'A_scratch', {}, current=False, expected=('R113', 'A_scratch'))
                self.assertEqual(len(finish.postlock(model, {})), 108)
                path = model['run'] / 'process_diagnostics_complete.json'
                value = finish.read(path)
                value['selected_update'] += 1
                put(path, value)
                with self.assertRaisesRegex(ValueError, 'unique R113 checkpoint'):
                    finish.postlock(model, {})

    def test_wait_requires_all_three_rc0_pipelines_not_r114(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wave(root)
            self.assertEqual(len(finish.completed(root)), 4)
            controller_path = root / 'preparation/wave_dispatch/controller.json'
            controller = finish.read(controller_path)
            controller['jobs']['C_transition']['state'] = 'running'
            put(controller_path, controller)
            with self.assertRaises(finish.NotReady):
                finish.completed(root)
            controller['jobs']['C_transition']['state'] = 'complete'
            put(controller_path, controller)
            path = root / 'preparation/wave_dispatch/C_transition/attempt_0001/worker.json'
            row = finish.read(path)
            row['state'] = 'running'
            put(path, row)
            with self.assertRaises(finish.NotReady):
                finish.completed(root)
            row.update(state='failed', exit_status=7)
            put(path, row)
            with self.assertRaisesRegex(RuntimeError, 'no restart'):
                finish.completed(root)
            row.update(state='complete', exit_status=0)
            row['stages'][-1]['exit_status'] = 2
            put(path, row)
            with self.assertRaises(ValueError):
                finish.completed(root)

    def test_no_analysis_or_git_until_complete_and_no_retry_on_failure(self):
        with patch.object(finish, 'analyze', side_effect=AssertionError), patch.object(finish, 'git', side_effect=AssertionError):
            with patch.object(finish, 'completed', side_effect=finish.NotReady('pending')), \
                    patch.object(finish.time, 'monotonic', side_effect=[0., 3601.]):
                with self.assertRaises(TimeoutError):
                    finish.wait('/synthetic', timeout_hours=1.)
            with patch.object(finish, 'completed', side_effect=RuntimeError('fatal')):
                with self.assertRaisesRegex(RuntimeError, 'fatal'):
                    finish.wait('/synthetic', publish_after=True)

    def test_pair_and_target_outside_partition(self):
        old, new = predictions(), predictions(new=True)
        row, events = finish.paired(old, new, 'B vs A', 'TEST', 'TSP')
        self.assertEqual([row[k] for k in ('corrected', 'harmed', 'both_wrong_cheaper', 'both_wrong_dearer')], [1, 1, 1, 1])
        self.assertEqual(len(events), row['changed'])
        self.assertAlmostEqual(row['gross_harm_cost']-row['gross_gain_cost'], row['mean_cost_delta'])
        left, right = ({p: copy.deepcopy(d) for p in finish.PROBLEMS} for d in (old, new))
        rows, totals = finish.target_recovery(left, right, [('TSP', 's0')], 'B vs A', 'TEST')
        self.assertEqual(sum(r['winner_count'] for r in totals), 6*18)
        self.assertAlmostEqual(sum(r['mean_cost_delta_contribution'] for r in totals), row['mean_cost_delta'])
        self.assertAlmostEqual(sum(r['actual_regret_delta_contribution_pp'] for r in totals), row['actual_regret_delta_pp'])
        method = next(r for r in rows if r['scope'] == 'fixed36_method')
        self.assertEqual((method['corrected'], method['harmed']), (1, 1))

    def test_cli_wait_publish_forwards_exact_plan(self):
        with patch.object(finish, 'wait', return_value={'complete': True}) as waiting:
            finish.main(['wait', '--publish', '--wave-plan', '/synthetic/plan.json'])
            self.assertTrue(waiting.call_args.kwargs['publish_after'])
            self.assertEqual(waiting.call_args.kwargs['plan_path'], Path('/synthetic/plan.json'))


if __name__ == '__main__':
    unittest.main()
