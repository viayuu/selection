"""Bounded CPU-only recovery contracts; no real caches, checkpoints or CUDA."""

import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import r102_postprocess_recovery as recovery
from . import r102_transfer as transfer


def write(path, record):
    transfer.write_new_json(path, record)


def header(problem):
    rows, maximum = 4500, 3
    matrix = problem == 'ATSP'
    width = None if matrix else 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
    arrays = {k: dict(file=k + '.npy', shape=[rows], dtype='int64')
              for k in ('n', 'winner', 'coord_dist')}
    arrays['costs'] = dict(file='costs.npy', shape=[rows, 1], dtype='float64')
    key = 'matrix' if matrix else 'node'
    arrays[key] = dict(file=key + '.npy', shape=[rows, maximum, maximum if matrix else width], dtype='float32')
    if not matrix:
        arrays['node_geom'] = dict(file='node_geom.npy', shape=[rows, maximum, 12], dtype='float32')
    result = dict.fromkeys(transfer.FROZEN_FIELDS)
    result.update(problem=problem, problem_id=recovery.PROBLEMS.index(problem), split='test',
        rows=rows, true_length_min=2, true_length_max=maximum, kind='matrix' if matrix else 'coord',
        d_in=width, raw_cost_dtype='float64', winner_source='native FP64 ind',
        cache_format='r76_scenario_v1', cache_version=1, complete=True, rows_written=rows,
        geometry_calculation_dtype='float64', geometry_cache_dtype='float32',
        geometry_scales=[4, 8, 16], geometry_fields=[f'k{k}_{n}' for k in (4, 8, 16)
            for n in ('radius', 'mean_distance', 'distance_cv', 'anisotropy')],
        scenario_id='synthetic', source_files=['never_open'], feature_sources=['never_open'],
        cbits=[0], problem_desc=[0.], pool=['reld_mt'], pool_ids=[0],
        scenario_global_solver_ids={'reld_mt': 0}, model_global_solvers=['reld_mt'], mask=[1],
        arrays=arrays, data_root='/never_open', directory='/raw/test/' + problem, cache_root='/raw',
        created_utc='reference_created', completed_utc='reference_completed', seconds=1.,
        test_lock_root='/reference')
    return result


class Fixture:
    def __init__(self, root):
        self.root, self.headers = root, {p: header(p) for p in recovery.PROBLEMS}
        self.manifest_path = root / 'manifest.json'
        write(self.manifest_path, dict(splits=dict(test=self.headers)))
        differences = {k: dict(reference=self.headers['TSP'][k], actual=v) for k, v in dict(
            created_utc='actual_created', completed_utc='actual_completed', seconds=2., test_lock_root='/actual').items()}
        self.raw = {}
        for node in ('gpu03', 'gpu05'):
            problems = {}
            for p, record in self.headers.items():
                arrays = {k: dict(file=s['file'], bytes=128 + math.prod(s['shape']) * (
                    4 if s['dtype'] == 'float32' else 8), sha256='a' * 64) for k, s in record['arrays'].items()}
                problems[p] = dict(arrays=arrays, header_differences=copy.deepcopy(differences) if node == 'gpu05' else {})
            self.raw[node] = dict(node=node, mode='hashes', arrays_hashed=True, CPU_only=True,
                no_GPU_context=True, cache_modified=False, raw_values_interpreted=False, problems=problems)
            write(root / (node + '.json'), self.raw[node])
        per = {p: dict(arrays_equal=True, array_count=len(r['arrays']),
            bytes=sum(a['bytes'] for a in r['arrays'].values()),
            gpu05_different_header_fields=sorted(recovery.BUILD_FIELDS)) for p, r in self.raw['gpu05']['problems'].items()}
        self.proof = dict(schema=recovery.PROOF_SCHEMA, complete=True, all_arrays_equal=True,
            problem_count=18, array_count=sum(r['array_count'] for r in per.values()),
            total_bytes_per_node=sum(r['bytes'] for r in per.values()),
            allowed_build_receipt_fields=sorted(recovery.BUILD_FIELDS),
            gpu03_header_matches_frozen_reference=True, other_gpu05_header_fields_equal=True,
            nodes={n: recovery.artifact(root / (n + '.json')) for n in self.raw}, per_problem=per)
        self.proof_path = root / 'proof.json'
        write(self.proof_path, self.proof)

    def validate(self):
        return recovery.validate_proof(self.proof_path, self.manifest_path)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture = Fixture(self.root)
        no_cuda = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests may not initialize CUDA'))
        no_cuda.start()
        self.addCleanup(no_cuda.stop)

    def replace_json(self, path, value):
        # Test-only fixture mutation, never production evidence.
        path.unlink()
        write(path, value)

    def test_concrete_proof_covers_all107_arrays(self):
        evidence = self.fixture.validate()
        self.assertEqual(evidence['array_count'], 107)
        self.assertEqual(set(evidence['pairs']), set(recovery.PROBLEMS))

    def test_missing_problem_array_nativewinner_or_cost_hash_cannot_pass(self):
        for mutation in ('problem', 'winner', 'costs', 'total', 'raw_hash'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp))
                if mutation == 'problem':
                    del f.proof['per_problem']['TSP']
                elif mutation == 'total':
                    f.proof['array_count'] -= 1
                elif mutation == 'raw_hash':
                    f.proof['nodes']['gpu03']['sha256'] = '0' * 64
                else:
                    raw = f.raw['gpu05']
                    if mutation == 'winner':
                        del raw['problems']['TSP']['arrays']['winner']
                    else:
                        raw['problems']['TSP']['arrays']['costs']['sha256'] = 'b' * 64
                    self.replace_json(f.root / 'gpu05.json', raw)
                    f.proof['nodes']['gpu05'] = recovery.artifact(f.root / 'gpu05.json')
                self.replace_json(f.proof_path, f.proof)
                with self.assertRaises(ValueError):
                    f.validate()

    def test_non_build_header_change_is_never_compatible(self):
        adapter = recovery.HeaderCompatibility(self.fixture.validate())
        pair = adapter.pairs['TSP']
        self.assertEqual(adapter.validate(pair['actual'], pair['reference']), pair['actual'])
        for key, value in (('winner_source', 'FP32'), ('rows', 4501), ('source_files', ['other']),
                           ('created_utc', 'unproven'), ('unknown_future_field', True)):
            bad = copy.deepcopy(pair['actual'])
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.validate(bad, pair['reference'])
        bad = copy.deepcopy(pair['actual'])
        bad['split'] = 'val'
        with self.assertRaises(ValueError):
            adapter.validate(bad, pair['reference'])

    def test_proof_disallows_extra_header_field_even_with_claimed_equal_arrays(self):
        raw = self.fixture.raw['gpu05']
        raw['problems']['TSP']['header_differences']['winner_source'] = dict(actual='FP32', reference='native FP64 ind')
        self.replace_json(self.root / 'gpu05.json', raw)
        self.fixture.proof['nodes']['gpu05'] = recovery.artifact(self.root / 'gpu05.json')
        self.replace_json(self.fixture.proof_path, self.fixture.proof)
        with self.assertRaises(ValueError):
            self.fixture.validate()

    def test_r94_digest_must_bind_exactly_one_proven_header(self):
        adapter = recovery.HeaderCompatibility(self.fixture.validate())
        pair = adapter.pairs['CVRP']
        for variant in ('actual', 'reference'):
            metadata = dict(source_cache=dict(metadata_sha256=recovery.digest(pair[variant])))
            with patch.object(recovery, 'FROZEN_R94_VALIDATE') as original:
                adapter.validate_r94(metadata, pair['reference'], {'donor': 'unchanged'})
                self.assertEqual(original.call_args.args[1], pair[variant])
        with self.assertRaises(ValueError):
            adapter.validate_r94(dict(source_cache=dict(metadata_sha256='0' * 64)), pair['reference'], {})

    def test_shadow_copies_only_metadata_and_never_old_first_test_marker(self):
        original = self.root / 'A'
        original.mkdir()
        for name in ('protocol.json', 'training_complete.json', 'final_test_lock.json', 'final_metrics.json',
                     'best_eval.json', 'history.json', 'input_features_ready.json', 'initialization.json',
                     'test_started.json', 'error.json', 'status.json', 'best.pt'):
            write(original / name, dict(marker=name, checkpoint_sha256=None))
        for split in ('train', 'val'):
            write(original / 'evaluation' / split / 'selected_best/identity.json', dict(checkpoint_sha256=None))
        before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
        shadow = recovery.shadow(self.root, 'A')
        self.assertTrue((shadow / 'best.pt').is_symlink())
        self.assertTrue((shadow / 'evaluation/val').is_symlink())
        self.assertFalse((shadow / 'evaluation').is_symlink())
        self.assertFalse((shadow / 'test_started.json').exists())
        self.assertFalse((shadow / 'error.json').exists())
        self.assertFalse((shadow / 'final_metrics.json').is_symlink())
        self.assertEqual(json.loads((shadow / 'final_metrics.json').read_text())['checkpoint_sha256'], None)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with self.assertRaises(FileExistsError):
            recovery.shadow(self.root, 'A')

    def test_private_guard_rejects_original_paths_and_symlink_escape(self):
        destination = recovery.base(self.root) / 'A/output.json'
        self.assertEqual(recovery.private(destination, self.root), destination)
        with self.assertRaises(ValueError):
            recovery.private(self.root / 'A/final_metrics.json', self.root)
        destination.parent.mkdir(parents=True)
        destination.symlink_to(self.root / 'outside')
        with self.assertRaises(ValueError):
            recovery.save(destination, {}, self.root)

    def test_first_error_is_verbatim_one_shot_and_blocks_later_stages(self):
        origin = dict(original_files={}, original_absent=[], checkpoint_sha256='a' * 64)
        def fail():
            raise RuntimeError('first actual timing drift')
        with self.assertRaisesRegex(RuntimeError, 'first actual'):
            recovery.stage(self.root, 'A', 'timing', fail, origin)
        path = recovery.base(self.root) / 'A/first_error.json'
        before = path.read_bytes()
        recovery.first_error(self.root, 'A', 'later', ValueError('later error'))
        self.assertEqual(before, path.read_bytes())
        with self.assertRaises(PermissionError):
            recovery.stage(self.root, 'A', 'lib', lambda: self.fail('LIB must not run'), origin)

    def test_successful_first_test_stage_cannot_implicitly_rerun(self):
        origin = dict(original_files={}, original_absent=[], checkpoint_sha256='a' * 64)
        self.assertEqual(recovery.stage(self.root, 'A', 'test', lambda: 1, origin), 1)
        with self.assertRaises(PermissionError):
            recovery.stage(self.root, 'A', 'test', lambda: self.fail('no repeat'), origin)

    def test_cpu_report_origin_verification_does_not_require_node_local_tmp(self):
        origin = dict(original_files={}, original_absent=[], array_signatures={'/no_cache_on_login': [1]})
        recovery.verify_origin(origin)
        with self.assertRaises(FileNotFoundError):
            recovery.verify_origin(origin, check_cache=True)

    def test_cpu_or_wrong_uuid_fails_before_cuda_or_model_reads(self):
        args = recovery.parser().parse_args(['gpu', '--arm', 'A', '--device', 'cpu'])
        with patch.object(recovery, 'original_records') as load, self.assertRaises(ValueError):
            recovery.gpu(args)
        load.assert_not_called()
        args.device = 'cuda:0'
        assignment = dict(gpu_uuid='GPU-original', cpus=[])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-wrong'}), self.assertRaises(ValueError):
            recovery.require_gpu(args, assignment)

    def test_actual_uuid_normalizes_only_gpu_prefix_and_keeps_identity_verification(self):
        expected = 'GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be'
        assignment = dict(gpu_uuid=expected, cpus=[0, 1, 2, 3], container_id='original-container')
        args = SimpleNamespace(device='cuda:0', expected_uuid=expected)
        props = SimpleNamespace(uuid=expected, name='mock original GPU')
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=expected), \
                patch.object(os, 'sched_getaffinity', return_value=set(assignment['cpus'])), \
                patch.object(Path, 'read_text', return_value='original-container'), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=1), \
                patch.object(torch.cuda, 'get_device_properties', return_value=props):
            for raw in (expected, expected.removeprefix('GPU-'), expected.removeprefix('GPU-').upper()):
                with self.subTest(raw=raw):
                    props.uuid = raw
                    device, hardware = recovery.require_gpu(args, assignment)
                    self.assertEqual(str(device), 'cuda:0')
                    self.assertEqual(hardware['uuid'], expected)
                    self.assertEqual(hardware['torch_uuid_raw'], raw)
            props.uuid = '597c682a-4b7f-7609-80a6-8ced791f5c5d'
            with self.assertRaisesRegex(ValueError, 'Actual CUDA UUID differs'):
                recovery.require_gpu(args, assignment)
            for malformed in (expected[:20], 'MIG-' + expected[4:], 'GPU-' + expected):
                with self.subTest(raw=malformed), self.assertRaises(ValueError):
                    props.uuid = malformed
                    recovery.require_gpu(args, assignment)

    def test_timing_near_equality_or_unchanged_predictions_never_hides_drift(self):
        record = dict(nco_fp16_bitwise_exact=True, node_masks_exact=True, solver_masks_exact=True,
            valid_scores_bitwise_exact=True, finite=True, prediction_changed_count=0)
        recovery.require_alignment(record)
        for key in ('nco_fp16_bitwise_exact', 'node_masks_exact', 'solver_masks_exact', 'valid_scores_bitwise_exact', 'finite'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'do not change inference'):
                recovery.require_alignment(dict(record, **{key: False}))
        self.assertFalse(recovery.bitwise(torch.tensor([0.]), torch.tensor([-0.])))

    def test_runtime_retains_original_lock_and_only_redirects_test_features(self):
        config = self.config()
        args = recovery.parser().parse_args(['gpu', '--root', str(self.root), '--arm', 'A', '--device', 'cuda:0'])
        runtime = recovery.runtime_args(args, config, shadow_paths=True)
        self.assertEqual(runtime.root, recovery.view(self.root))
        self.assertEqual(runtime.protocol_lock, self.root / 'protocol_lock.json')
        self.assertEqual(runtime.feature_root, recovery.base(self.root) / 'A/test_features')
        args.feature_root = Path('/new_inference_recipe')
        with self.assertRaises(ValueError):
            recovery.runtime_args(args, config)

    def config(self):
        return dict(cache_root='/raw', feature_cache_root='/original_features', routing_feature_cache_root='/routing',
            data_root='/frozen_data', prepared_root='/frozen_prepared', gpu_type='4090', eval_batch_size=256,
            microbatch_caps={'TSP': 352})

    def test_context_restores_functions_and_checks_full_model_state_even_with_null_val_sha(self):
        from . import r102_experiment as experiment, r102_features as features
        from .r77_data import parameter_hash
        args = recovery.parser().parse_args(['gpu', '--root', str(self.root), '--arm', 'A', '--device', 'cuda:0'])
        config, model = self.config(), torch.nn.Linear(2, 2)
        lock = dict(checkpoint=str(self.root / 'A/best.pt'), selected_update=33120)
        state = dict(args=config, successful_updates=33120)
        origin = dict(model_state_sha256=parameter_hash(model.state_dict()), val_capture_checkpoint_sha256=None)
        original_validator, fields = features.validate_frozen_header, transfer.RUNTIME_FIELDS
        def authorize(runtime):
            self.assertEqual(runtime.feature_root, Path('/original_features'))
            self.assertEqual(runtime.protocol_lock, self.root / 'protocol_lock.json')
            return lock, config
        with patch.object(experiment, 'authorize_test', side_effect=authorize), patch.object(
                experiment, 'load_checkpoint', return_value=(model, state)):
            with recovery.compatibility(args, config, lock, origin, self.fixture.validate()) as (runtime, adapter):
                self.assertEqual(experiment.authorize_test(runtime), (lock, config))
                loaded, _ = experiment.load_checkpoint(lock['checkpoint'], 'cpu')
                self.assertFalse(loaded.training)
                self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                state['successful_updates'] = 57600
                with self.assertRaises(PermissionError):
                    experiment.load_checkpoint(lock['checkpoint'], 'cpu')
                self.assertEqual(transfer.RUNTIME_FIELDS, fields)
        self.assertIs(features.validate_frozen_header, original_validator)
        self.assertIs(transfer.validate_frozen_header, recovery.FROZEN_VALIDATE)

    def test_mocked_gpu_pipeline_orders_first_test_timing_lib_and_emits_real_receipt(self):
        from . import r102_experiment as experiment, r102_reporting as reporting
        args = recovery.parser().parse_args(['gpu', '--root', str(self.root), '--arm', 'A', '--device', 'cuda:0'])
        config = self.config()
        lock = dict(checkpoint_sha256='b' * 64, state_hash='c' * 64, selected_update=33120)
        origin = dict(original_files={}, original_absent=[], assignments={'A': {}},
            checkpoint_sha256=lock['checkpoint_sha256'], model_state_sha256='d' * 64)
        run = recovery.view(self.root) / 'A'
        calls = []
        def make_shadow(*unused):
            run.mkdir(parents=True)
            return run
        def test(*unused):
            calls.append('test')
            for name in ('test_results.json', 'test_complete.json', 'evaluation/test/selected_best/identity.json',
                         'evaluation/test/selected_best/rank000.npz'):
                write(run / name, {})
        def timing(*unused):
            calls.append('timing')
            write(run / 'inference_timing.json', {})
        def lib(*unused):
            calls.append('lib')
            write(run / 'lib_stage_complete.json', {})
        @contextmanager
        def compatible(*unused):
            yield SimpleNamespace(), SimpleNamespace(uses=[])
        @contextmanager
        def quiet(*unused):
            yield
        with patch.object(recovery, 'quiescent', quiet), patch.object(recovery, 'original_records', return_value=(config, lock, origin)), \
                patch.object(recovery, 'require_gpu', return_value=(torch.device('cpu'), {})), \
                patch.object(recovery, 'validate_proof', return_value=self.fixture.validate()), \
                patch.object(recovery, 'validate_local_cache', return_value={}), \
                patch.object(recovery, 'shadow', side_effect=make_shadow), patch.object(recovery, 'compatibility', compatible), \
                patch.object(torch, 'set_num_threads'), patch.object(torch, 'set_num_interop_threads'), \
                patch.object(experiment, 'test', side_effect=test), patch.object(experiment, 'timing', side_effect=timing), \
                patch.object(experiment, 'train', side_effect=AssertionError('Never train')), \
                patch.object(reporting, 'lib_stage', side_effect=lib), patch.object(reporting, 'inference_timing'), \
                patch.object(reporting, '_completed_lib'):
            receipt = recovery.gpu(args)
        self.assertEqual(calls, ['test', 'timing', 'lib'])
        self.assertTrue(receipt['complete'])
        path = recovery.base(self.root) / 'A/gpu_complete.json'
        self.assertEqual(recovery.read_json(path)['stages'], ['first_completed_TEST', 'timing', 'LIB'])
        for value in receipt['outputs'].values():
            recovery.checked_artifact(value)

    def known_failure_fixture(self):
        directory = recovery.base(self.root) / 'A'
        write(directory / 'first_error.json', dict(stage='timing', type='ValueError', exit_status=1, complete=False,
            message='R102 cached-vs-online drift; preserve first error, do not change inference/grouping'))
        write(directory / 'timing_checks/TSP.json', dict(finite=True, node_masks_exact=True, solver_masks_exact=True,
            prediction_changed_count=0, nco_fp16_bitwise_exact=False, valid_scores_bitwise_exact=False))
        for name in ('gpu_entry.json', 'gpu_uuid_format_correction_entry.json'):
            write(directory / name, dict(state='failed', exit_status=1))
        pins = (recovery.file_hash(directory / 'first_error.json'),
                recovery.file_hash(directory / 'timing_checks/TSP.json'), 'unused_result', 'unused_archive')
        return directory, pins

    def test_observational_timing_records_drift_and_calls_unchanged_original_measure(self):
        from . import r102_experiment as experiment
        args = recovery.parser().parse_args(['finish-gpu', '--root', str(self.root), '--arm', 'A', '--device', 'cuda:0'])
        config, lock = self.config(), {}
        raw, cached, selector = {'problem_id': 0}, {'original_cached': True}, object()
        comparison = dict(nco_fp16_bitwise_exact=False, valid_scores_bitwise_exact=False, prediction_changed_count=3)
        with patch.object(recovery, 'timing_comparison', return_value=comparison), \
                patch.object(recovery, 'require_alignment', side_effect=AssertionError('No extra hard gate')), \
                patch.object(experiment, 'measure_inputs', return_value={'original_measurement': True}) as original:
            with recovery.compatibility(args, config, lock, {}, self.fixture.validate(), observational_timing=True):
                self.assertEqual(experiment.measure_inputs(selector, raw, cached, torch.device('cpu')),
                                 {'original_measurement': True})
            original.assert_called_once_with(selector, raw, cached, torch.device('cpu'), 3)
        observation = recovery.read_json(recovery.base(self.root) / 'A/timing_observations/TSP.json')
        self.assertEqual(observation['prediction_changed_count'], 3)
        self.assertFalse(observation['nco_fp16_bitwise_exact'])
        self.assertIn('not whole-TEST', observation['numerical_boundary'])

    def test_unknown_prior_failure_sha_cannot_authorize_post_test(self):
        self.known_failure_fixture()
        with self.assertRaisesRegex(PermissionError, 'Unknown/changed prior failure'):
            recovery.known_timing_failure(self.root, 'A')

    def test_new_post_test_failure_preserves_known_first_error_and_blocks_later_work(self):
        directory, pins = self.known_failure_fixture()
        origin = dict(original_files={}, original_absent=[], checkpoint_sha256='a' * 64)
        before = (directory / 'first_error.json').read_bytes()
        with patch.dict(recovery.KNOWN_TIMING_FAILURES, A=pins):
            prior = recovery.known_timing_failure(self.root, 'A')
            with self.assertRaisesRegex(RuntimeError, 'new original timing error'):
                recovery.stage(self.root, 'A', 'timing_observation',
                    lambda: (_ for _ in ()).throw(RuntimeError('new original timing error')),
                    origin, allowed_failure=prior)
            with self.assertRaises(PermissionError):
                recovery.stage(self.root, 'A', 'lib_completion', lambda: self.fail('no LIB'), origin, allowed_failure=prior)
        self.assertEqual((directory / 'first_error.json').read_bytes(), before)
        self.assertEqual(recovery.read_json(directory / 'post_test_error.json')['message'], 'new original timing error')

    def test_finish_gpu_never_calls_train_or_test_and_completion_binds_known_failure(self):
        from . import r102_experiment as experiment, r102_reporting as reporting
        args = recovery.parser().parse_args(['finish-gpu', '--root', str(self.root), '--arm', 'A', '--device', 'cuda:0'])
        directory, pins = self.known_failure_fixture()
        write(directory / 'timing_started.json', dict(old_attempt='preserved'))
        run = recovery.view(self.root) / 'A'
        lock = dict(checkpoint_sha256=recovery.LOCKED['A'][1], state_hash='c' * 64, selected_update=33120)
        write(self.root / 'A/final_test_lock.json', lock)
        for name in ('test_results.json', 'test_complete.json', 'evaluation/test/selected_best/identity.json',
                     'evaluation/test/selected_best/rank000.npz'):
            write(run / name, dict(first_TEST='preserved'))
        before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
        evidence, config, calls = self.fixture.validate(), self.config(), []
        origin = dict(original_files={str(p): recovery.file_hash(p) for p in before}, original_absent=[],
            assignments={'A': {}}, checkpoint_sha256=lock['checkpoint_sha256'], model_state_sha256='d' * 64,
            array_proof={k: v for k, v in evidence.items() if k != 'pairs'})
        def timing(*unused):
            calls.append('timing')
            write(run / 'inference_timing.json', {})
            for p in recovery.PROBLEMS:
                write(directory / 'timing_observations' / (p + '.json'), dict(problem=p, instances=128))
        def lib(*unused):
            calls.append('lib')
            write(run / 'lib_stage_complete.json', {})
        @contextmanager
        def compatible(*unused, **options):
            self.assertTrue(options['observational_timing'])
            yield SimpleNamespace(), SimpleNamespace(uses=[])
        @contextmanager
        def quiet(*unused):
            yield
        with patch.dict(recovery.KNOWN_TIMING_FAILURES, A=pins):
            prior = recovery.known_timing_failure(self.root, 'A')
            origin['prior_timing_failure'] = prior
            with patch.object(recovery, 'quiescent', quiet), \
                    patch.object(recovery, 'post_test_state', return_value=(config, lock, origin)), \
                    patch.object(recovery, 'require_gpu', return_value=(torch.device('cpu'), {})), \
                    patch.object(recovery, 'compatibility', compatible), \
                    patch.object(torch, 'set_num_threads'), patch.object(torch, 'set_num_interop_threads'), \
                    patch.object(experiment, 'test', side_effect=AssertionError('Never TEST')), \
                    patch.object(experiment, 'train', side_effect=AssertionError('Never TRAIN')), \
                    patch.object(experiment, 'prepare_input_features', side_effect=AssertionError('Never prepare TEST')), \
                    patch.object(experiment, 'timing', side_effect=timing), \
                    patch.object(reporting, 'lib_stage', side_effect=lib), patch.object(reporting, 'inference_timing'), \
                    patch.object(reporting, '_completed_lib'):
                receipt = recovery.finish_gpu(args)
            self.assertEqual(calls, ['timing', 'lib'])
            self.assertEqual(receipt['stage_exit_statuses'], dict(timing_observation=0, lib_completion=0))
            self.assertEqual(receipt['prior_timing_failure'], prior)
            self.assertEqual(recovery.checked_completion(self.root, 'A', 'gpu')[0], receipt)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertEqual(recovery.read_json(directory / 'timing_started.json'), dict(old_attempt='preserved'))
            self.assertFalse(receipt['test_rerun'])
            self.assertFalse(receipt['online_efficacy_bitwise_certified'])
            bad = copy.deepcopy(receipt)
            bad['prior_timing_failure']['failure']['sha256'] = '0' * 64
            self.replace_json(directory / 'gpu_complete.json', bad)
            with self.assertRaises(PermissionError):
                recovery.checked_completion(self.root, 'A', 'gpu')


if __name__ == '__main__':
    unittest.main()
