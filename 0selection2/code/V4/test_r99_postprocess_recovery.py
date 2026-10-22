"""Focused CPU-only contracts for the disjoint postprocessing recovery."""

from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from . import r99_postprocess_recovery as recovery


def timing_fixture(exact=False):
    lock = dict(arm='A', checkpoint_sha256='a'*64, state_hash='state', inference_rule='locked')
    comparison = dict(messages=dict(bitwise_exact=exact, unequal_bytes=0 if exact else 1,
        nonfinite_count=0, max_abs=0. if exact else .001), scores=dict(nonfinite_count=0),
        masks_exact=True, online_predictions_global_ids=[2]*128, cached_predictions_global_ids=[2]*128,
        prediction_changed_count=0)
    measurement = dict(batch_seconds=[.1, .2, .3], mean_seconds_per_instance=.6/(3*128))
    prep = dict(experiment='R99', arm='A', variant='row', complete=True, seconds=1.)
    value = dict(recovery_schema=recovery.SCHEMA, experiment='R99', complete=True, exit_status=0,
        arm='A', variant='row', repetitions=3, warmup_batches=1, includes_input_transform=True,
        raw_input_source='original_base_cache_without_relation_fields', online_transform_verified=True,
        full_system_measured=False, end_to_end_measured=False, model_reselection=False,
        efficacy_reused_as_exact_online=False, labels_opened=False,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        inference_rule=lock['inference_rule'], timing_path='mixed', requested_tf32_policy='default',
        scope='fixed first128 original input rows', excludes='I/O and transfer',
        numerical_disclosure='Nonexact online messages are not original cached TEST efficacy.',
        hardware=dict(device_name='synthetic CPU test'), online_fp16_matches_cache=exact,
        original_inference_equivalence=exact,
        preprocessing=dict(prep, splits=['train', 'val']),
        test_preprocessing=dict(prep, splits=['test'], checkpoint_sha256=lock['checkpoint_sha256'], state_hash='state'),
        per_problem=[dict(problem=p, instances=128, max_marginal_residual=.1,
            comparison=copy.deepcopy(comparison), original_mixed_comparison=copy.deepcopy(comparison),
            **{scope: copy.deepcopy(measurement) for scope in recovery.experiment.TIMING_SCOPES})
            for p in recovery.experiment.PROBLEMS])
    return lock, value


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU in CPU tests'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_approximate_is_never_exact_and_negative_zero_is_not_bitwise(self):
        original = torch.tensor([1., 0.], dtype=torch.float16)
        close = torch.tensor([1.0009765625, -0.], dtype=torch.float16)
        value = recovery.difference(close, original)
        self.assertFalse(value['value_exact'])
        self.assertFalse(value['bitwise_exact'])
        self.assertEqual(value['tolerance_bad_count'], 0)
        zero = recovery.difference(close[1:], original[1:])
        self.assertTrue(zero['value_exact'])
        self.assertFalse(zero['bitwise_exact'])

    def test_nonfinite_values_cannot_look_exact_and_json_numbers_are_finite(self):
        value = recovery.difference(torch.tensor([float('nan'), 2.]), torch.tensor([1., 2.]))
        self.assertFalse(value['bitwise_exact'])
        self.assertEqual(value['nonfinite_count'], 1)

    def test_exact_n_crops_without_changing_row_order_or_pool_ids(self):
        raw = dict(n=torch.tensor([2, 3, 2]), node_mask=torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.bool),
            node=torch.arange(18).reshape(3, 3, 2).float(), pool_ids=torch.tensor([0, 2, 7]))
        observed = []

        def transform(batch, variant):
            observed.append((len(batch['n']), batch['node'].shape[1], batch['pool_ids'].tolist()))
            return batch['node'][..., :1].expand(-1, -1, 32).half(), torch.tensor(.1)

        with patch.object(recovery, 'transport', side_effect=transform):
            messages, _ = recovery.transform_path(raw, 'row', 'exact_n')
        self.assertEqual(observed, [(2, 2, [0, 2, 7]), (1, 3, [0, 2, 7])])
        self.assertEqual(messages[2, 0, 0], raw['node'][2, 0, 0])
        self.assertEqual(messages[0, 2].count_nonzero(), 0)

    def test_cpu_gpu_command_rejected_before_data_reads(self):
        args = recovery.parser().parse_args(['diagnose', '--arm', 'A', '--device', 'cpu'])
        with patch.object(recovery, 'provenance') as load, self.assertRaises(ValueError):
            recovery.diagnose(args)
        load.assert_not_called()

    def test_writes_are_private_and_original_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / recovery.RECOVERY_RELATIVE / 'A/diagnosis.json'
            self.assertEqual(recovery.assert_private(allowed, root), allowed)
            with self.assertRaises(ValueError):
                recovery.assert_private(root / 'A/inference_timing.json', root)
            recovery.save_new(allowed, dict(complete=False))
            self.assertTrue(allowed.is_file())

    def test_tf32_policy_restored_without_initializing_gpu(self):
        previous = recovery.numerical_policy()
        with recovery.tf32_policy('on') as policy:
            self.assertTrue(policy['matmul_allow_tf32'])
        self.assertEqual(recovery.numerical_policy(), previous)

    def test_ties_keep_lowest_global_solver_id(self):
        self.assertEqual(recovery.prediction(torch.tensor([[1., 1., 0.]]), torch.tensor([8, 3, 9])), [3])

    def test_private_validator_accepts_honest_mismatch_but_never_forged_exact(self):
        lock, value = timing_fixture()
        result = recovery.validate_timing_record(value, lock)
        self.assertFalse(result['original_inference_equivalence'])
        self.assertFalse(result['online_fp16_matches_cache'])
        self.assertEqual(len(result['measurements']), 18)
        for key in ('original_inference_equivalence', 'online_fp16_matches_cache'):
            bad = copy.deepcopy(value)
            bad[key] = True
            with self.assertRaises(ValueError):
                recovery.validate_timing_record(bad, lock)
        bad = copy.deepcopy(value)
        bad['per_problem'][0]['comparison']['messages']['bitwise_exact'] = True
        with self.assertRaises(ValueError):
            recovery.validate_timing_record(bad, lock)

    def test_private_validator_requires_all18_and_actual_three_samples(self):
        lock, value = timing_fixture(True)
        self.assertTrue(recovery.validate_timing_record(value, lock)['online_fp16_matches_cache'])
        for mutation in ('task', 'samples', 'mean', 'prep', 'masks', 'prediction'):
            bad = copy.deepcopy(value)
            if mutation == 'task':
                bad['per_problem'].pop()
            elif mutation == 'samples':
                bad['per_problem'][0]['transform_only']['batch_seconds'] = [.1]
            elif mutation == 'mean':
                bad['per_problem'][0]['transform_only']['mean_seconds_per_instance'] = float('inf')
            elif mutation == 'prep':
                bad['test_preprocessing']['checkpoint_sha256'] = 'another'
            elif mutation == 'masks':
                bad['per_problem'][0]['comparison']['masks_exact'] = False
            else:
                bad['per_problem'][0]['comparison']['prediction_changed_count'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recovery.validate_timing_record(bad, lock)

    def test_report_adapter_never_rebinds_original_validator(self):
        from . import r99_reporting as original
        validator = original.inference_timing
        module = recovery.private_reporter()
        self.assertIs(original.inference_timing, validator)
        self.assertIsNot(module.inference_timing, validator)
        lock, record = timing_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            recovery.save_new(Path(tmp)/'inference_timing.json', record)
            self.assertFalse(module.inference_timing(Path(tmp), lock)['original_inference_equivalence'])
            with self.assertRaises(ValueError):
                original.inference_timing(Path(tmp), lock)

    def test_val_binding_changes_only_one_original_byte_span(self):
        raw = b'{ "other": {"checkpoint_sha256":null},\n "val" : { "identity": {"checkpoint_sha256": null, "x": 1.00000} }, "n": 3 }\n'
        bound = recovery.bound_val_bytes(raw, 'a'*64)
        self.assertEqual(bound, raw.replace(b'"checkpoint_sha256": null', b'"checkpoint_sha256": "' + b'a'*64 + b'"'))
        left, right = json.loads(raw), json.loads(bound)
        left['val']['identity']['checkpoint_sha256'] = 'a'*64
        self.assertEqual(left, right)
        with self.assertRaises(ValueError):
            recovery.bound_val_bytes(bound, 'a'*64)

    def test_input_reader_never_opens_label_arrays_and_preserves_fp16_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'cache/test/TSP'
            features = root/'features/test/TSP'
            source.mkdir(parents=True)
            features.mkdir(parents=True)
            arrays = dict(n=np.full(4500, 2, dtype=np.int64), coord_dist=np.ones(4500, dtype=np.int64),
                node=np.ones((4500, 2, 2), dtype=np.float32), node_geom=np.ones((4500, 2, 4), dtype=np.float32))
            specs = {}
            for key, array in arrays.items():
                np.save(source/(key+'.npy'), array)
                specs[key] = dict(file=key+'.npy', shape=list(array.shape), dtype=str(array.dtype))
            specs['winner'] = dict(file='winner.npy', dtype='int64', shape=[4500])
            specs['costs'] = dict(file='costs.npy', dtype='float64', shape=[4500, 2])
            record = dict(complete=True, rows=4500, rows_written=4500, problem='TSP', split='test',
                data_root=str(root/'data'), arrays=specs, kind='coord', problem_id=0, d_in=2,
                cbits=[0], problem_desc=[0], pool=['x', 'y'], pool_ids=[0, 1], mask=[1, 1])
            recovery.save_new(source/'metadata.json', record)
            recovery.save_new(root/'prepared/source_cache_manifest.json', dict(splits=dict(test=dict(TSP=record))))
            lock = dict(checkpoint_sha256='a'*64)
            identity = dict(format=recovery.FORMAT, problem='TSP', split='test', variant='row',
                transform=recovery.SPEC, source_identity=recovery.source_identity(record), rows=4500,
                input_only=True, labels_opened=False, test_lock_sha256=recovery.digest(lock))
            recovery.save_new(features/'metadata.json', dict(complete=True, identity=identity, dtype='float16', width=32))
            np.save(features/'offsets.npy', np.arange(4501, dtype=np.int64)*2)
            np.save(features/'messages.npy', np.ones((9000, 32), dtype=np.float16))
            before = recovery.file_hash(features/'messages.npy')
            config = dict(cache_root=str(root/'cache'), feature_cache_root=str(root/'features'), arm='A',
                prepared_root=str(root/'prepared'), data_root=str(root/'data'))
            load = np.load
            opened = []

            def input_load(path, **kwargs):
                opened.append(Path(path).name)
                self.assertNotIn(Path(path).name, ('winner.npy', 'costs.npy', 'raw_label.pkl'))
                return load(path, **kwargs)

            with patch.object(recovery.np, 'load', side_effect=input_load):
                reader = recovery.InputReader(config, 'TSP', lock)
                try:
                    batch = reader.batch(np.arange(128), 'cpu', cached=True)
                    self.assertEqual(batch['relation_nodes'].shape, (128, 2, 32))
                    self.assertNotIn('ind', batch)
                    self.assertNotIn('costs', batch)
                finally:
                    reader.close()
            self.assertEqual(before, recovery.file_hash(features/'messages.npy'))
            self.assertIn('messages.npy', opened)

    def test_existing_private_input_link_is_safe_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root/'A/protocol.json'
            recovery.save_new(original, dict(original=True))
            link = recovery.recovery_root(root)/'report_view/A/protocol.json'
            recovery.ensure_link(link, original, root)
            recovery.ensure_link(link, original, root)
            self.assertEqual(link.resolve(), original.resolve())

    def test_val_binding_requires_completed_report_and_changes_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'A'
            lock = dict(checkpoint_sha256='a'*64, state_hash='state', selected_update=56160)
            recovery.save_new(run/'final_test_lock.json', lock)
            identity = dict(checkpoint_sha256=None, state_hash='state', successful_updates=56160,
                model_state_sha256='model-state')
            target = run/'final_metrics.json'
            recovery.save_new(target, dict(val=dict(identity=identity, macro=dict(macro_top1=.75))))
            archived = run/'evaluation/val/selected_best/identity.json'
            recovery.save_new(archived, identity)
            recovery.save_new(run/'test_results.json', dict(result=dict(identity=dict(
                identity, checkpoint_sha256=lock['checkpoint_sha256']))))
            report = dict(lock=lock, original_frozen={'final_metrics.json': dict(sha256=recovery.file_hash(target))},
                original_failure_sha256='failure')
            report_path = recovery.stage_receipt_path(root, 'A', 'report')
            recovery.save_new(report_path, report)
            original, archived_bytes = target.read_bytes(), archived.read_bytes()
            args = recovery.parser().parse_args(['bind-val', '--root', str(root), '--arm', 'A', '--device', 'cpu'])
            with patch.object(recovery, 'check_report', side_effect=ValueError('report incomplete')):
                with self.assertRaises(ValueError):
                    recovery.bind_val(args)
            self.assertEqual(target.read_bytes(), original)
            with patch.object(recovery, 'check_report', return_value=report), \
                    patch.object(recovery, 'reporting_view'), patch.object(recovery, 'verify_origin'):
                receipt = recovery.bind_val(args)
                self.assertEqual(recovery.bind_val(args), receipt)
            self.assertEqual(target.read_bytes(), recovery.bound_val_bytes(original, lock['checkpoint_sha256']))
            self.assertEqual(Path(receipt['backup']['path']).read_bytes(), original)
            self.assertEqual(archived.read_bytes(), archived_bytes)
            self.assertTrue(recovery.authorized_val_binding(root, 'A', receipt['backup']['sha256']))
            self.assertFalse(receipt['numeric_values_changed'])
            self.assertFalse(receipt['prediction_arrays_changed'])
            self.assertFalse(receipt['weights_changed'])

    def test_val_binding_rejects_different_model_state_before_backup_or_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'A'
            identity = dict(checkpoint_sha256=None, state_hash='state', successful_updates=56160,
                model_state_sha256='VAL-model')
            target = run/'final_metrics.json'
            recovery.save_new(target, dict(val=dict(identity=identity)))
            recovery.save_new(run/'evaluation/val/selected_best/identity.json', identity)
            recovery.save_new(run/'test_results.json', dict(result=dict(identity=dict(
                identity, checkpoint_sha256='a'*64, model_state_sha256='other-model'))))
            report = dict(lock=dict(checkpoint_sha256='a'*64, state_hash='state', selected_update=56160),
                original_frozen={'final_metrics.json': dict(sha256=recovery.file_hash(target))})
            args = recovery.parser().parse_args(['bind-val', '--root', str(root), '--arm', 'A', '--device', 'cpu'])
            original = target.read_bytes()
            with patch.object(recovery, 'check_report', return_value=report), self.assertRaises(ValueError):
                recovery.bind_val(args)
            self.assertEqual(target.read_bytes(), original)
            self.assertFalse((recovery.recovery_root(root)/'A/final_metrics.before_val_sha_binding.json').exists())

    def test_global_gate_requires_same_lock_comparison_and_untampered_child_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = recovery.recovery_root(root)
            children, arms = {}, {}
            for arm in ('A', 'B'):
                children[arm] = dict(checkpoint_lock=dict(arm=arm, checkpoint_sha256=arm*64),
                    original_failure_sha256='failure-'+arm, original_inference_equivalence=False)
                path = base/arm/'recovery_complete.json'
                recovery.save_new(path, children[arm])
                arms[arm] = dict(path=str(path), sha256=recovery.file_hash(path),
                    checkpoint_lock=children[arm]['checkpoint_lock'],
                    original_failure_sha256=children[arm]['original_failure_sha256'])
            locks = {arm: r['checkpoint_lock'] for arm, r in children.items()}
            failures = {arm: r['original_failure_sha256'] for arm, r in children.items()}
            pdf, protocol = base/'report_view/RESULTS.pdf', base/'report_view/comparison_artifacts/protocol.json'
            recovery.save_new(pdf, dict(synthetic_pdf_fixture=True))
            recovery.save_new(protocol, dict(complete=True, checkpoints=locks))
            comparison = dict(schema=recovery.SCHEMA, complete=True, exit_status=0,
                checkpoint_locks=locks, original_failure_sha256=failures,
                comparison_pdf=dict(path=str(pdf), sha256=recovery.file_hash(pdf)),
                protocol=dict(path=str(protocol), sha256=recovery.file_hash(protocol)))
            comparison_path = base/'comparison_stage.json'
            recovery.save_new(comparison_path, comparison)
            value = dict(schema=recovery.SCHEMA, experiment='R99', complete=True, exit_status=0, state='complete',
                same_checkpoint_verified=True, cli_module='code.V4.r99_postprocess_recovery',
                original_failure_receipts_preserved=True, training_rerun=False, full_test_rerun=False,
                feature_copy=False, feature_rebuild=False, decision_rule_changed=False,
                efficacy_reused_as_exact_online=False, original_inference_equivalence=False,
                arms=arms, checkpoint_locks=locks, original_failure_sha256=failures,
                stages_completed=['A_timing_LIB_PDF_VAL_binding', 'B_timing_LIB_PDF_VAL_binding', 'CPU_AB_comparison'],
                comparison=dict(path=str(comparison_path), sha256=recovery.file_hash(comparison_path)))
            output = base/'recovery_complete.json'
            recovery.save_new(output, value)
            with patch.object(recovery, 'validate_completed_arm', side_effect=lambda root, arm: children[arm]):
                self.assertTrue(recovery.validate_recovery_complete(root)['complete'])
                recovery.save_new(base/'A/recovery_complete.json', dict(tampered=True))
                with self.assertRaises(ValueError):
                    recovery.validate_recovery_complete(root)
                recovery.save_new(base/'A/recovery_complete.json', children['A'])
                comparison['checkpoint_locks'] = dict(A=dict(checkpoint_sha256='foreign'), B=locks['B'])
                recovery.save_new(comparison_path, comparison)
                value['comparison']['sha256'] = recovery.file_hash(comparison_path)
                recovery.save_new(output, value)
                with self.assertRaises(ValueError):
                    recovery.validate_recovery_complete(root)


if __name__ == '__main__':
    unittest.main()
