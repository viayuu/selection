"""Bounded CPU fixtures; no actual GPU, allocation, launch, ready or Git action."""

from contextlib import ExitStack, redirect_stderr
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r102_goal_recovery as goal
from . import r102_gpu_recovery_entry as entry
from .test_r102_goal import fixture as scientific_fixture


def artifact(path):
    return dict(path=str(path), sha256=goal.file_hash(path))


def write(path, value):
    goal.resource.save(path, value)


def locked(value, lock):
    if isinstance(value, dict):
        result = {key: locked(child, lock) for key, child in value.items()}
        for key in ('checkpoint_sha256', 'selected_update'):
            if key in result:
                result[key] = lock[key]
        return result
    if isinstance(value, list):
        return [locked(child, lock) for child in value]
    return value


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r102-recovery-cpu-')
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.repo = self.parent / 'repo'
        self.root, self.launch, locks = scientific_fixture(self.repo)
        self.base = self.root / 'preparation/postprocess_recovery'
        self.view = self.base / 'report_view'
        self.mutex_root = self.repo / 'R81'
        stack = ExitStack()
        self.addCleanup(stack.close)
        for module in (goal, entry):
            stack.enter_context(patch.object(module, 'ROOT', self.root))
        stack.enter_context(patch.object(entry, 'LAUNCH', self.launch))
        for name, value in (('REPO', self.repo), ('BASE', self.base), ('VIEW', self.view), ('LAUNCH', self.launch)):
            stack.enter_context(patch.object(goal, name, value))
        stack.enter_context(patch.object(goal.resource, 'REPO', self.repo))
        stack.enter_context(patch.object(goal.resource.transport, 'ROOT', self.mutex_root))
        self.locks = {}
        error = dict(type='ValueError', message='Full frozen R84 header identity differs', traceback='original first failure')
        for arm in goal.ARMS:
            run, view = self.root / arm, self.view / arm
            view.mkdir(parents=True)
            lock = dict(locks[arm], selected_update=goal.LOCKS[arm][0], checkpoint_sha256=goal.LOCKS[arm][1])
            self.locks[arm] = lock
            write(run / 'final_test_lock.json', lock)
            train = goal.optional(run / 'training_complete.json')
            write(run / 'training_complete.json', dict(train, selected_best=lock, test_read=False))
            for name in ('final_metrics.json', 'test_results.json', 'inference_timing.json', 'lib_stage_complete.json'):
                write(run / name, locked(goal.optional(run / name), lock))
            final = goal.optional(run / 'final_metrics.json')
            final['val']['identity']['model_state_sha256'] = arm.lower() * 64
            write(run / 'final_metrics.json', dict(final, test_read=False))
            for name in ('test_results.json', 'inference_timing.json', 'lib_stage_complete.json', 'postprocess_status.json',
                         'full_model_statistics'):
                shutil.move(str(run / name), str(view / name))
            report = goal.optional(view / 'full_model_statistics/protocol.json')
            write(view / 'full_model_statistics/protocol.json', dict(report, checkpoint_lock=lock,
                inference_timing=dict(sha256=goal.file_hash(view / 'inference_timing.json'))))
            test = goal.optional(view / 'test_results.json')
            write(view / 'test_results.json', dict(test, selected_update=lock['selected_update']))
            write(view / 'test_complete.json', dict(full_tests=1, checkpoint_sha256=lock['checkpoint_sha256']))
            write(view / 'evaluation/test/selected_best/identity.json', dict(state_hash=lock['state_hash'],
                checkpoint_sha256=lock['checkpoint_sha256'], successful_updates=lock['selected_update'],
                model_state_sha256=arm.lower() * 64))
            (view / 'evaluation/test/selected_best/rank000.npz').write_bytes(b'not opened by administration')
            write(run / 'status.json', dict(phase='failed', successful_updates=57600, error=error))
            write(run / 'error.json', error)
            write(run / 'test_started.json', dict(checkpoint_sha256=lock['checkpoint_sha256']))
            slot = goal.resource.SLOTS[goal.SLOTS[arm]]
            identity = goal.resource.transport.CONTAINERS[slot['node']][1]
            write(self.launch / ('R102_' + arm) / 'worker.json', dict(state='failed', phase='train', exit_status=1,
                slot=goal.SLOTS[arm], gpu_uuid=slot['uuid'], cpus=goal.CPUS[arm], container_id=identity,
                stages=[dict(name='input_features', state='complete', exit_status=0),
                        dict(name='train', state='failed', exit_status=1)]))
            write(self.launch / ('R102_' + arm) / 'controller.json', dict(state='failed', exit_status=1))
        write(self.root / 'goal_watch_receipt.json', dict(state='stopped_on_error', exit_status=1))
        proof = dict(schema='r102_cross_node_test_array_identity_v1', complete=True, all_arrays_equal=True,
            problem_count=18, array_count=107, gpu03_header_matches_frozen_reference=True,
            other_gpu05_header_fields_equal=True, arrays_or_headers_modified=False, no_solver_or_label_generation=True,
            allowed_build_receipt_fields=sorted(goal.BUILD_FIELDS),
            per_problem={p: dict(arrays_equal=True, gpu05_different_header_fields=sorted(goal.BUILD_FIELDS)) for p in goal.PROBLEMS}, nodes={})
        for node in ('gpu03', 'gpu05'):
            path = self.base / (node + '_hashes.json')
            write(path, dict(node=node, problems={p: dict(arrays_equal=True) for p in goal.PROBLEMS}))
            proof['nodes'][node] = artifact(path)
        write(self.base / 'array_equality_proof.json', proof)
        for path in goal.required_sources():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# small fixture\n')
        mutexes = {}
        for arm in goal.ARMS:
            path = self.mutex_root / ('queue_gpu_' + goal.resource.SLOTS[goal.SLOTS[arm]]['uuid'] + '.json')
            write(path, {})
            stat = path.stat()
            mutexes[arm] = dict(path=str(path), device=stat.st_dev, inode=stat.st_ino)
        self.ready = dict(schema='r102_postprocess_recovery_ready_v1', main_thread_ready=True, cpu_tests_passed=True,
            evidence='main CPU fixture', source_sha256={str(path): goal.file_hash(path) for path in goal.required_sources()},
            gpu_mutexes=mutexes)

    def completions(self, arm):
        lock, origin = self.locks[arm], goal.original(arm)
        origin_paths = {'worker.json': self.launch / ('R102_' + arm) / 'worker.json',
            'controller.json': self.launch / ('R102_' + arm) / 'controller.json',
            'goal_watch_receipt.json': self.root / 'goal_watch_receipt.json'}
        path = self.base / arm / 'origin.json'
        write(path, dict(original_root=str(self.root), arm=arm, checkpoint_sha256=lock['checkpoint_sha256'],
            selected_update=lock['selected_update'], model_state_sha256=arm.lower() * 64,
            original_files={str(origin_paths.get(name, self.root / arm / name)): digest for name, digest in origin['sha256'].items()}))
        common = dict(schema='r102_postprocess_recovery_v1', experiment='R102', arm=arm, complete=True, exit_status=0,
            origin=artifact(path), report_view=str(self.view), training=False, model_reselection=False,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'], selected_update=lock['selected_update'],
            model_state_sha256=arm.lower() * 64)
        outputs = ('test_results.json', 'test_complete.json', 'inference_timing.json', 'lib_stage_complete.json',
                   'evaluation/test/selected_best/identity.json', 'evaluation/test/selected_best/rank000.npz')
        write(self.base / arm / 'gpu_complete.json', dict(common,
            outputs={name: artifact(self.view / arm / name) for name in outputs}))
        report_files = ('postprocess_status.json', 'full_model_statistics/protocol.json')
        outputs = {name: artifact(self.view / arm / name) for name in report_files}
        outputs.update({name: artifact(self.view / arm / 'full_model_statistics' / name) for name in goal.admin.R99_PDFS})
        write(self.base / arm / 'report_complete.json', dict(common, outputs=outputs,
            gpu_completion=artifact(self.base / arm / 'gpu_complete.json')))
        for stage in ('gpu', 'report'):
            write(self.base / arm / (stage + '_entry.json'), dict(state='complete', exit_status=0, original=origin,
                main_ready_evidence=self.ready['evidence'], source_sha256=self.ready['source_sha256']))
        goal.source_provenance(self.ready['evidence'])

    def comparison(self):
        protocol = dict(experiment='R102', complete=True, exit_status=0, checkpoints=self.locks,
            target_identity=dict(target_count=36, sha256='c'*64),
            inference_timing={arm: dict(sha256=goal.file_hash(self.view / arm / 'inference_timing.json')) for arm in goal.ARMS})
        write(self.view / 'comparison_artifacts/protocol.json', protocol)
        (self.view / 'comparison_artifacts/COMPARISON.pdf').write_bytes(b'%PDF-small CPU fixture')
        (self.view / 'RESULTS.pdf').write_bytes(b'%PDF-small CPU fixture')
        write(self.base / 'compare_complete.json', dict(schema='r102_postprocess_recovery_v1', complete=True,
            exit_status=0, report_view=str(self.view), training=False, inference=False, result=protocol,
            reports={arm: artifact(self.base / arm / 'report_complete.json') for arm in goal.ARMS}))

    def uuid_observation(self):
        arms = {}
        for arm in goal.ARMS:
            path = self.base / arm / 'gpu_entry.json'
            write(path, dict(state='failed', exit_status=1, arm=arm, stage='gpu', original=goal.original(arm)))
            log = self.base / (arm + '_gpu.log')
            log.write_text('ValueError: ' + entry.UUID_ERROR + '\n')
            arms[arm] = dict(entry=artifact(path), log=artifact(log), error=entry.UUID_ERROR,
                origin_absent=True, shadow_absent=True, test_outputs_absent=True)
        write(self.base / 'uuid_preflight_observation.json', dict(schema='r102_uuid_preflight_observation_v1',
            main_authorizes_one_explicit_format_correction=True, arms=arms))
        return {path: path.read_bytes() for path in
                [*(self.base / arm / 'gpu_entry.json' for arm in goal.ARMS),
                 *(self.base / (arm + '_gpu.log') for arm in goal.ARMS), self.base / 'uuid_preflight_observation.json']}

    def timing_failure_chain(self):
        for arm in goal.ARMS:
            self.completions(arm)
        completed = {arm: goal.optional(self.base / arm / 'gpu_entry.json') for arm in goal.ARMS}
        self.uuid_observation()
        for arm in goal.ARMS:
            write(self.base / arm / entry.UUID_CORRECTION_ENTRY, dict(completed[arm], arm=arm, stage='gpu',
                state='failed', exit_status=1, uuid_format_correction=True, **entry.uuid_preflight_observation(arm)))
            write(self.base / arm / 'first_error.json', dict(schema='r102_postprocess_recovery_v1',
                stage='timing', type='ValueError', message=entry.TIMING_DRIFT_ERROR, exit_status=1))
        return completed

    def numerical_completion(self, arm):
        boundary = ('Preserved cached-feature first TEST; NOT certified bitwise online efficacy. '
                    'Original all18 online/cache timings; diagnostic first128 rows/task only.')
        directory = self.base / arm
        check = directory / 'timing_checks/TSP.json'
        write(check, dict(nco_fp16_bitwise_exact=False, valid_scores_bitwise_exact=False, finite=True,
            node_masks_exact=True, solver_masks_exact=True, prediction_changed_count=0, max_abs_score_difference=.0001))
        observations = {}
        for problem in goal.PROBLEMS:
            path = directory / 'timing_observations' / (problem + '.json')
            write(path, dict(problem=problem, instances=128, scope='first128 TEST rows/task only',
                numerical_boundary=boundary, finite=True, inference_changed=False, grouping_changed=False,
                nco_fp16_bitwise_exact=False, valid_scores_bitwise_exact=False,
                prediction_changed_count=int(problem == 'VRPTW'), max_abs_score_difference=.0001))
            observations[problem] = artifact(path)
        current = entry.post_test_evidence(arm)
        fields = dict(completion_kind='post_test_original_timing_observation', test_rerun=False,
            stage_exit_statuses=dict(timing_observation=0, lib_completion=0),
            online_efficacy_bitwise_certified=False, numerical_boundary=boundary, timing_observations=observations,
            prior_timing_failure=dict(failure=current['preserved_first_error'], observation=artifact(check),
                preserved_entries={'gpu_entry.json': current['preserved_gpu_entry'],
                                   entry.UUID_CORRECTION_ENTRY: current['preserved_uuid_correction_entry']}))
        path = directory / 'gpu_complete.json'
        write(path, dict(goal.optional(path), **fields))
        output = self.view / arm / 'full_model_statistics'
        (output / 'numerical_boundary.md').write_text(boundary)
        (output / 'numerical_boundary.pdf').write_bytes(b'%PDF-numerical boundary')
        artifacts = dict(note=artifact(output / 'numerical_boundary.md'), pdf=artifact(output / 'numerical_boundary.pdf'))
        path = output / 'protocol.json'
        write(path, dict(goal.optional(path), recovery_numerical_boundary=boundary,
            online_efficacy_bitwise_certified=False, timing_observations=observations, numerical_boundary_artifacts=artifacts))
        path = directory / 'report_complete.json'
        report = goal.optional(path)
        report['outputs'].update({'full_model_statistics/protocol.json': artifact(output / 'protocol.json'),
                                 **{'numerical_boundary_' + k: v for k, v in artifacts.items()}})
        write(path, dict(report, **fields, gpu_completion=artifact(directory / 'gpu_complete.json')))

    def test_post_test_entry_runs_finish_gpu_once_and_preserves_test_and_prior_failures(self):
        self.timing_failure_chain()
        (self.base / 'A/gpu_complete.json').unlink()
        paths = [self.base / 'A' / name for name in ('gpu_entry.json', entry.UUID_CORRECTION_ENTRY, 'first_error.json')]
        paths.append(self.view / 'A/test_results.json')
        before = {path: path.read_bytes() for path in paths}
        args = entry.parser().parse_args(['--arm', 'A', '--stage', 'post-test', '--evidence', 'main'])
        self.assertEqual(args.stage, 'post-test')
        def finish(argv, **kwargs):
            lock = self.locks['A']
            write(self.base / 'A/gpu_complete.json', dict(schema='r102_postprocess_recovery_v1',
                complete=True, exit_status=0, arm='A', **{key: lock[key] for key in
                    ('checkpoint_sha256', 'state_hash', 'selected_update')}))
            return SimpleNamespace(returncode=0)
        with patch.object(entry.subprocess, 'run', side_effect=finish) as run:
            receipt = entry._execute('A', 'post-test', 'main timing continuation', {})
            self.assertEqual(receipt['state'], 'complete')
            self.assertTrue(receipt['post_test_continuation'])
            self.assertFalse(receipt['test_rerun'])
            self.assertFalse(receipt['first_full_TEST'])
            argv = run.call_args.args[0]
            self.assertIn('finish-gpu', argv)
            self.assertIn('cuda:0', argv)
            self.assertNotIn('train', argv)
            self.assertNotIn('test', argv)
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                entry._execute('A', 'post-test', 'main', {})
            self.assertEqual(run.call_count, 1)
        self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_post_test_rejects_different_failure_lock_and_implicit_retry(self):
        self.timing_failure_chain()
        (self.base / 'A/gpu_complete.json').unlink()
        failure = self.base / 'A/first_error.json'
        first = goal.optional(failure)
        with patch.object(entry.subprocess, 'run') as run:
            write(failure, dict(first, message='unrelated CUDA failure'))
            with self.assertRaisesRegex(RuntimeError, 'exact timing drift'):
                entry._execute('A', 'post-test', 'main', {})
            write(failure, first)
            path = self.view / 'A/test_results.json'
            write(path, dict(goal.optional(path), selected_update=57600))
            with self.assertRaisesRegex(RuntimeError, 'same-lock first TEST'):
                entry._execute('A', 'post-test', 'main', {})
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                entry._execute('A', 'gpu', 'main', {})
            run.assert_not_called()
        self.assertFalse((self.base / 'A' / entry.POST_TEST_ENTRY).exists())

    def test_post_test_gate_requires_complete_declared_chain_not_false_original_success(self):
        completed = self.timing_failure_chain()
        for arm in goal.ARMS:
            self.numerical_completion(arm)
            write(self.base / arm / entry.POST_TEST_ENTRY, dict(completed[arm], stage='post-test',
                post_test_continuation=True, **entry.post_test_evidence(arm)))
        current = goal.status()
        self.assertTrue(current['all_arms_ready'], current)
        self.assertFalse(current['errors'], current)
        self.assertEqual(len(current['preserved_prior_recovery_failures']), 6)
        self.assertFalse(current['original_failed_route_complete'])
        path = self.base / 'A' / entry.POST_TEST_ENTRY
        write(path, dict(goal.optional(path), state='failed', exit_status=1))
        self.assertFalse(goal.stage_complete('A', 'gpu'))
        self.assertTrue(goal.status()['errors'])

    def test_post_test_gate_rejects_fake_bitwise_certificate_and_missing_all18_observations(self):
        completed = self.timing_failure_chain()
        self.numerical_completion('A')
        write(self.base / 'A' / entry.POST_TEST_ENTRY, dict(completed['A'], stage='post-test',
            post_test_continuation=True, **entry.post_test_evidence('A')))
        self.assertTrue(goal.stage_complete('A', 'gpu'))
        path = self.base / 'A/gpu_complete.json'
        record = goal.optional(path)
        write(path, dict(record, online_efficacy_bitwise_certified=True))
        with self.assertRaisesRegex(RuntimeError, 'not bitwise online certification'):
            goal.stage_complete('A', 'gpu')
        reduced = dict(record['timing_observations'])
        reduced.pop('ATSP')
        write(path, dict(record, timing_observations=reduced))
        with self.assertRaisesRegex(RuntimeError, 'all18'):
            goal.stage_complete('A', 'gpu')

    def test_uuid_correction_explicit_one_shot_preserves_first_failure_and_logs(self):
        shutil.rmtree(self.view)
        before = self.uuid_observation()
        args = entry.parser().parse_args(['--arm', 'A', '--stage', 'gpu', '--evidence', 'main', '--uuid-format-correction'])
        self.assertTrue(args.uuid_format_correction)
        with patch.object(entry.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                entry._execute('A', 'gpu', 'main', {})
            def complete(argv, **kwargs):
                lock = self.locks['A']
                write(self.base / 'A/gpu_complete.json', dict(schema='r102_postprocess_recovery_v1',
                    complete=True, exit_status=0, arm='A', **{key: lock[key] for key in
                        ('checkpoint_sha256', 'state_hash', 'selected_update')}))
                return SimpleNamespace(returncode=0)
            run.side_effect = complete
            receipt = entry._execute('A', 'gpu', 'main UUID correction', {}, uuid_format_correction=True)
            self.assertTrue(receipt['uuid_format_correction'])
            self.assertEqual(receipt['state'], 'complete')
            self.assertEqual(run.call_args.args[0], entry.command('A', 'gpu'))
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                entry._execute('A', 'gpu', 'main', {}, uuid_format_correction=True)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(goal.optional(self.base / 'A/gpu_entry.json')['state'], 'failed')
        # A's continuation must not make B fail its independent preflight gate.
        write(self.base / 'A/origin.json', {})
        entry.uuid_preflight_observation('B', require_absent=True)

    def test_uuid_correction_missing_proof_changed_log_and_wrong_stage_fail_closed(self):
        with patch.object(entry.subprocess, 'run') as run:
            with self.assertRaises(FileNotFoundError):
                entry._execute('A', 'gpu', 'main', {}, uuid_format_correction=True)
            with self.assertRaisesRegex(ValueError, 'stage gpu'):
                entry.execute('A', 'report', 'main', uuid_format_correction=True)
            self.uuid_observation()
            with self.assertRaisesRegex(RuntimeError, 'before origin'):
                entry.uuid_preflight_observation('A', require_absent=True)
            (self.base / 'B_gpu.log').write_text('changed\n')
            with self.assertRaisesRegex(RuntimeError, 'receipt/log changed'):
                entry.uuid_preflight_observation('A')
            run.assert_not_called()

    def test_delivery_accepts_only_proven_completed_correction_and_discloses_old_failures(self):
        for arm in goal.ARMS:
            self.completions(arm)
        completed = {arm: goal.optional(self.base / arm / 'gpu_entry.json') for arm in goal.ARMS}
        before = self.uuid_observation()
        for arm in goal.ARMS:
            write(self.base / arm / entry.UUID_CORRECTION_ENTRY, dict(completed[arm], uuid_format_correction=True,
                **entry.uuid_preflight_observation(arm)))
        current = goal.status()
        self.assertTrue(current['all_arms_ready'], current)
        self.assertFalse(current['errors'], current)
        self.assertEqual(len(current['preserved_uuid_preflight_failures']), 2)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        path = self.base / 'A' / entry.UUID_CORRECTION_ENTRY
        write(path, dict(goal.optional(path), state='failed', exit_status=1))
        self.assertFalse(goal.stage_complete('A', 'gpu'))
        self.assertTrue(goal.status()['errors'])
        write(path, dict(completed['A'], uuid_format_correction=True, **entry.uuid_preflight_observation('A')))
        (self.base / 'A_gpu.log').write_text('changed\n')
        self.assertTrue(goal.status()['errors'])

    def test_original_failure_and57600_lock_are_required_not_r99_timing(self):
        baseline = goal.original('A')
        self.assertEqual(baseline['checkpoint_lock'], self.locks['A'])
        path = self.launch / 'R102_A/worker.json'
        worker = goal.optional(path)
        write(path, dict(worker, phase='timing'))
        with self.assertRaisesRegex(RuntimeError, 'TRAIN-stage'):
            goal.original('A')
        write(path, worker)
        path = self.root / 'A/training_complete.json'
        write(path, dict(goal.optional(path), successful_updates=57599))
        with self.assertRaisesRegex(RuntimeError, '57600'):
            goal.original('A')

    def test_parser_and_exact_fixed_commands_no_train_or_retry(self):
        args = entry.parser().parse_args(['--arm', 'A', '--stage', 'gpu', '--evidence', 'main'])
        self.assertEqual(args.stage, 'gpu')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.parser().parse_args(['--arm', 'A', '--stage', 'train', '--evidence', 'main'])
        argv = entry.command('B', 'gpu')
        self.assertIn('/tmp/shiys_R102_B_extra_nodes', argv)
        self.assertIn('/tmp/shiys_R94_B_node_features', argv)
        self.assertNotIn('train', argv)
        self.assertEqual(goal.parser().parse_args([]).interval, 120)
        self.assertFalse(goal.parser().parse_args([]).publish)

    def test_named_science_stages_do_not_become_fake_nested_receipt_errors(self):
        record = dict(schema='r102_postprocess_recovery_v1', complete=True, exit_status=0,
            stages=['timing_observation', 'lib_completion'],
            stage_exit_statuses=dict(timing_observation=0, lib_completion=0))
        self.assertEqual(goal.receipt_errors('gpu_complete.json', record), [])
        self.assertEqual(goal.receipt_errors('report_complete.json', dict(record, stages=['CPU_report'])), [])
        self.assertTrue(goal.receipt_errors('gpu_complete.json', dict(record,
            stage_exit_statuses=dict(timing_observation=1, lib_completion=0))))
        self.assertTrue(goal.receipt_errors('gpu_complete.json', dict(record, stages=['unexpected'])))

    def test_pre_serialization_val_identity_is_bound_only_to_exact_parent_lock(self):
        for arm in goal.ARMS:
            path = self.root / arm / 'final_metrics.json'
            final = goal.optional(path)
            final['val']['identity'].update(checkpoint_sha256=None,
                successful_updates=self.locks[arm]['selected_update'])
            write(path, final)
        for arm in goal.ARMS:
            self.completions(arm)
        current = goal.status()
        self.assertTrue(current['all_arms_ready'], current)
        self.assertFalse(current['errors'], current)
        identity = goal.optional(self.root / 'A/final_metrics.json')['val']['identity']
        self.assertIsNone(identity['checkpoint_sha256'])

    def test_explicit_delivery_continuation_preserves_failed_watch_and_provenance(self):
        for arm in goal.ARMS:
            self.completions(arm)
        path = self.base / 'watch_receipt.json'
        write(path, dict(state='stopped_on_error', exit_status=1,
            error='Recovery failed; preserve every original/new receipt, no automatic retry'))
        paths = (path, self.base / 'source_provenance.json')
        before = {p: p.read_bytes() for p in paths}
        self.assertTrue(goal.status()['errors'])
        with patch.object(goal, 'DELIVERY_CONTINUATION', True):
            current = goal.status()
            self.assertTrue(current['all_arms_ready'], current)
            self.assertFalse(current['errors'], current)
            goal.source_provenance('explicit administrative continuation')
            self.assertTrue((self.base / 'delivery_source_provenance.json').is_file())
        self.assertEqual(before, {p: p.read_bytes() for p in paths})

    def test_ready_source_proof_and_manifest_fail_closed(self):
        self.assertEqual(goal.require_ready(self.ready['evidence']), dict(evidence=self.ready['evidence']))
        with self.assertRaises(ValueError):
            goal.require_ready(None)
        self.assertEqual(goal.require_ready('other main authorization'), dict(evidence='other main authorization'))
        path = self.base / 'gpu05_hashes.json'
        write(path, dict(changed=True))
        with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
            goal.proof_ready()

    def test_source_changes_never_refreeze(self):
        goal.source_provenance(self.ready['evidence'])
        path = next(iter(self.ready['source_sha256']))
        Path(path).write_text('# changed\n')
        before = (self.base / 'source_provenance.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'provenance changed'):
            goal.source_provenance(self.ready['evidence'])
        self.assertEqual(before, (self.base / 'source_provenance.json').read_bytes())

    def test_whole_gate_requires_actual_test_two_reports_and_all_five_pdfs(self):
        before = {arm: goal.original(arm) for arm in goal.ARMS}
        self.assertFalse(goal.status()['complete'])
        for arm in goal.ARMS:
            self.completions(arm)
        current = goal.status()
        self.assertTrue(current['all_arms_ready'], current)
        self.assertFalse(current['complete'])
        self.comparison()
        self.assertFalse(goal.status()['complete'])
        write(self.base / 'compare_entry.json', dict(state='complete', exit_status=0))
        current = goal.status()
        self.assertTrue(current['complete'], current)
        self.assertFalse(current['original_failed_route_complete'])
        self.assertFalse(current['goal_complete'])
        self.assertEqual(before, {arm: goal.original(arm) for arm in goal.ARMS})
        (self.view / 'B/full_model_statistics/learning_curves.pdf').unlink()
        self.assertFalse(goal.status()['complete'])

    def test_actual_test_and_selected_update_cannot_be_faked(self):
        self.completions('A')
        path = self.view / 'A/test_results.json'
        write(path, dict(goal.optional(path), selected_update=57600))
        current = goal.status()
        self.assertTrue(current['errors'])
        self.assertFalse(current['complete'])

    def test_original_test_marker_and_failure_are_never_rewritten(self):
        for arm in goal.ARMS:
            self.completions(arm)
        marker = self.root / 'A/test_started.json'
        write(marker, dict(checkpoint_sha256='e'*64))
        self.assertTrue(goal.status()['errors'])

    def test_entry_failure_is_terminal_even_rc0_without_receipt(self):
        with patch.object(entry.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            with self.assertRaisesRegex(RuntimeError, 'rc0'):
                entry._execute('A', 'gpu', self.ready['evidence'], {})
            receipt = goal.optional(self.base / 'A/gpu_entry.json')
            self.assertEqual(receipt['state'], 'failed')
            self.assertEqual(receipt['exit_status'], 1)
            self.assertEqual(receipt['original_worker_sha256'], goal.original('A')['sha256']['worker.json'])
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                entry._execute('A', 'gpu', self.ready['evidence'], {})
            self.assertEqual(run.call_count, 1)

    def test_report_waits_for_new_gpu_and_is_cpu_only(self):
        with patch.object(entry.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'new complete'):
                entry.execute('A', 'report', self.ready['evidence'])
            run.assert_not_called()

    def test_exact_resources_and_existing_mutex_inode_no_gpu_probe(self):
        gpu = goal.resource.SLOTS['4090_0']['uuid']
        mutex = Path(self.ready['gpu_mutexes']['A']['path'])
        before = mutex.stat().st_ino
        with patch.object(entry.resource, 'assert_container', return_value='original-container'), \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), \
                patch.object(entry.os, 'sched_getaffinity', return_value=set(range(8))), \
                patch.object(entry.os, 'sched_setaffinity') as affinity:
            with entry.reservation('A') as value:
                self.assertEqual(value['cpus'], [0, 1, 2, 3])
                self.assertEqual(value['gpu_mutex']['inode'], before)
            self.assertEqual(affinity.call_args_list[0].args[1], {0, 1, 2, 3})
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='wrong'):
                with self.assertRaisesRegex(RuntimeError, 'UUID'):
                    with entry.reservation('A'):
                        self.fail('foreign UUID accepted')
            stat = mutex.stat()
            changed = SimpleNamespace(st_dev=stat.st_dev, st_ino=stat.st_ino + 1)
            with patch.object(entry.os, 'fstat', return_value=changed):
                with self.assertRaisesRegex(RuntimeError, 'inode'):
                    with entry.reservation('A'):
                        self.fail('replaced mutex accepted')
        self.assertEqual(mutex.stat().st_ino, before)

    def test_comparison_once_saved_cpu_and_failure_never_retries(self):
        for arm in goal.ARMS:
            self.completions(arm)
        with patch.object(goal.subprocess, 'run', return_value=SimpleNamespace(returncode=7)) as run:
            with self.assertRaisesRegex(RuntimeError, 'no retry'):
                goal.compare_once(self.ready['evidence'])
            with self.assertRaises(RuntimeError):
                goal.compare_once(self.ready['evidence'])
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertIn('compare', run.call_args.args[0])
        self.assertEqual(goal.optional(self.base / 'compare_entry.json')['state'], 'failed')

    def test_successful_cpu_comparison_and_whole_publication_gate(self):
        with patch.object(goal.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'No WIP'):
                goal.publish(self.ready['evidence'])
            run.assert_not_called()
        for arm in goal.ARMS:
            self.completions(arm)
        def compare(*args, **kwargs):
            self.comparison()
            return SimpleNamespace(returncode=0)
        with patch.object(goal.subprocess, 'run', side_effect=compare) as run:
            goal.compare_once(self.ready['evidence'])
            self.assertTrue(goal.status()['complete'])
            self.assertEqual(run.call_count, 1)
        base = SimpleNamespace(publish=lambda *args: dict(args=list(map(str, args))))
        with patch.object(goal.previous, 'private', return_value=base):
            value = goal.publish(self.ready['evidence'])
            self.assertEqual(value['args'], ['R102', str(self.base), str(self.launch)])

    def test_quiet120s_watcher_separate_one_shot_and_explicit_publication(self):
        original = (self.root / 'goal_watch_receipt.json').read_bytes()
        waiting = dict(errors=[], all_arms_ready=False, comparison_complete=False, complete=False)
        complete = dict(errors=[], all_arms_ready=True, comparison_complete=True, complete=True)
        with patch.object(goal, 'status', side_effect=[waiting, complete]), \
                patch.object(goal.time, 'sleep') as sleep, patch.object(goal, 'publish') as publish:
            self.assertEqual(goal.watch(self.ready['evidence']), 0)
            sleep.assert_called_once_with(120)
            publish.assert_not_called()
        self.assertEqual(original, (self.root / 'goal_watch_receipt.json').read_bytes())
        with self.assertRaises(FileExistsError):
            goal.watch(self.ready['evidence'])
        with self.assertRaises(ValueError):
            goal.watch(self.ready['evidence'], interval=1)

    def test_curation_includes_whole_science_failure_proof_reports_not_payloads_or_dirty(self):
        for arm in goal.ARMS:
            self.completions(arm)
        self.comparison()
        paths = {'A/test_features/metadata.json': {}, 'A/labels.json': dict(labels=[1]),
                 'A/predictions.json': dict(pred=[0]), 'token.json': dict(token='secret')}
        for relative, value in paths.items():
            write(self.base / relative, value)
        write(self.base / 'unsafe.json', dict(argv=['--token', 'secret']))
        write(self.repo / 'unrelated_dirty.json', dict(keep=True))
        (self.base / 'A/weights.pt').write_bytes(b'excluded')
        (self.base / 'foreign.json').symlink_to(self.repo / 'unrelated_dirty.json')
        from . import r98_launch
        with patch.object(r98_launch, 'source_closure', side_effect=lambda paths: [str(p) for p in paths]):
            selected = goal.publication_paths('R102', self.base, self.launch, self.parent)
        names = '\n'.join(selected)
        self.assertIn('array_equality_proof.json', names)
        self.assertIn('gpu03_hashes.json', names)
        self.assertIn('source_provenance.json', names)
        self.assertIn('A/error.json', names)
        self.assertIn('B/error.json', names)
        self.assertIn('B/full_model_statistics/selection_distributions.pdf', names)
        self.assertIn('r102_gpu_recovery_entry.py', names)
        self.assertIn('r102_cache_identity_probe.py', names)
        self.assertIn('run_v4_r102_cache_probe.sh', names)
        for forbidden in ('test_features', 'labels.json', 'predictions.json', 'unsafe.json', 'token.json',
                          'weights.pt', 'rank000.npz', 'unrelated_dirty', 'foreign.json', 'controller.json'):
            self.assertNotIn(forbidden, names)

    def test_imports_do_not_load_torch_numpy_or_modify_environment(self):
        script = '''
import importlib.abc, os, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'numpy', 'wandb'):
            raise AssertionError(name)
sys.meta_path.insert(0, NoModels())
before = dict(os.environ)
from code.V4 import r102_gpu_recovery_entry
assert 'code.V4.r102_goal_recovery' not in sys.modules
from code.V4 import r102_goal_recovery
assert before == dict(os.environ)
'''
        repo = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=repo,
                                capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
