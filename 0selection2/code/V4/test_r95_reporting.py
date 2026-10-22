"""Bounded R95 CPU reporting tests using saved scalar/identity fixtures."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r95_reporting as reporting


def config(arm='A'):
    return dict(experiment='R95', r95_schema_version=1, arm=arm, model_factory=reporting.MODEL_FACTORY,
        data_root='/frozen/new90k', rows_per_problem=dict(train=81000, val=4500, test=4500),
        baselines={'TSP': {'selection_split': 'train'}}, candidate_pools={'TSP': ['BQ', 'T2T']},
        candidate_pool_ids={'TSP': [0, 1]}, winner_rule='native FP64 ind',
        common_trainable_sha256='fresh-R84', preparation_sha256='TRAIN-only', task_order_sha256='same',
        initialization='scratch_shared_r77', seed=2, global_batch_size=1024, logical_global_batch=1024,
        max_successful_updates=57600, warmup_updates=4320, eval_every_updates=1440, eval_batch_size=256,
        weight_decay=1e-4, optimizer='AdamW', grad_clip=1., dropout=.1, rdrop_weight=.35,
        rdrop_ramp_updates=4320, gpu_type='3090' if arm == 'A' else '4090',
        selection_head='original_score_mlp_maximin', microbatch_caps={'TSP': 352, 'CVRP': 352},
        checkpoint_rule='minimum VAL macro_vs_sbs', inference_rule='original maximin',
        main_protocol_lock={'sha256': 'locked'}, loss=dict(ce=.35, comparison=.35, risk=.02),
        sampling=copy.deepcopy(reporting.SAMPLING), objective=dict(forward_passes=2, main='original R77'),
        input_roles=dict(arm='B_execution_anchor', node_column=3),
        model_params=dict(dropout=.1, r95_arm=arm, role_arm='B_execution_anchor',
            r95_spec={'base': 'unchanged original full18'}))


def split_result(arm, split, lock):
    return dict(identity=dict(experiment='R95', arm=arm, split=split,
        successful_updates=lock['selected_update'], state_hash=lock['state_hash'],
        model_state_sha256='same-model', checkpoint_sha256=lock['checkpoint_sha256'] if split == 'test' else None),
        per_problem={p: dict(n=81000 if split == 'train' else 4500) for p in reporting.PROBLEMS})


def finalized_fixture(run, arm='A'):
    lock = dict(checkpoint_sha256='checkpoint', state_hash='state', selected_update=1440)
    test = split_result(arm, 'test', lock)
    reporting.write_json(run / 'final_metrics.json', dict(complete=True, test_read=True, test=test,
        train=split_result(arm, 'train', lock), val=split_result(arm, 'val', lock),
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'], selected_update=1440))
    reporting.write_json(run / 'test_complete.json', dict(full_tests=1,
        checkpoint_sha256=lock['checkpoint_sha256'], saved_predictions_reusable=True))
    return lock, test


def proposal_fixture(run, arm='A'):
    for step in range(7200, 57601, 7200):
        source = run / 'evaluation/train' / f'update{step:07d}' / 'identity.json'
        identity = dict(experiment='R95', arm=arm, split='train', state_hash='state',
            model_state_sha256=f'own-{arm}-{step}', successful_updates=step)
        reporting.write_json(source, identity)
        metrics = dict(n=81000, q_min=1 / 81000, q_max=1 / 81000, max_inverse_weight=1.,
            proposal_effective_population=81000., gradient_norm_mean=2., gradient_norm_max=2.,
            deterministic_last_layer_second_moment_ratio=1.)
        reporting.write_json(run / 'proposals' / f'update{step:07d}.json', dict(
            experiment='R95', arm=arm, complete=True, split='train', successful_updates=step,
            rows_per_problem=81000, state_hash='state', model_state_sha256=identity['model_state_sha256'],
            sampling=reporting.SAMPLING, source_identity_sha256=reporting.file_hash(source),
            source_prediction_sha256='recorded-not-read', proposal_sha256='recorded-not-read',
            per_problem={p: metrics.copy() for p in reporting.PROBLEMS}))
    reporting.write_json(run / 'history.json', [dict(successful_updates=step, optimization=dict(
        sampling_mean_weight=1., sampling_min_weight=1., sampling_max_weight=1., sampling_unique_fraction=.99))
        for step in range(1440, 57601, 1440)])


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_public_parser_factory_and_original_loss_labels(self):
        self.assertEqual(reporting.MODEL_FACTORY, 'code.V4.r95_experiment.load_checkpoint')
        self.assertEqual(reporting.DROPOUT, dict(A=.1, B=.1))
        for command in ('lib', 'report', 'compare', 'status'):
            args = reporting.parser().parse_args([command, '--arm', 'B', '--device', 'cpu'])
            self.assertEqual(args.arm, 'B')
            self.assertEqual(args.root.name, 'R95_adaptive_example_sampling')
        self.assertIn('original R77 loss/R-Drop', reporting.BOUNDARY)
        self.assertIn('not bitwise hardware-matched', reporting.BOUNDARY)
        self.assertIn('No guaranteed variance reduction, causal claim', reporting.BOUNDARY)
        self.assertIn('not an unbiased optimizer-update claim', reporting.PROXY_BOUNDARY)
        source = Path(reporting.__file__).read_text()
        for stale in ('raw utility CE only', 'one-forward utility', 'dropout B=0', 'released3090_1',
                'B:4090_1', 'rdrop=False', "train_ce_probability_link='utility'"):
            self.assertNotIn(stale, source)

    def test_strict_matched_configs_keep_loss_dropout_role_and_all_new_controls(self):
        result = reporting.validate_matched_configs(config('A'), config('B'))
        self.assertTrue(result['matched'])
        self.assertFalse(result['causal_claim'])
        for key, value in (('dropout', 0.), ('rdrop_weight', 0.), ('common_trainable_sha256', 'trained'),
                ('objective', dict(forward_passes=1)), ('new_loss_control', True),
                ('loss', dict(ce=1., comparison=0., risk=0.))):
            other = config('B')
            other[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.same_data(config('A'), other, matched=True)
        other = config('B')
        other['model_params']['new_head'] = True
        with self.assertRaises(ValueError):
            reporting.same_data(config('A'), other, matched=True)

    def test_locked_factory_is_late_bound_and_final_test_is_required(self):
        lock, test = finalized_fixture(self.root)
        result = (self.root, config(), lock, test, split_result('A', 'val', lock))
        with patch.object(reporting.shared, 'locked_inputs', return_value=result), \
                patch.object(reporting.importlib, 'import_module') as module:
            self.assertEqual(reporting.locked_inputs(self.root.parent, 'A'), result)
            module.assert_called_once_with('code.V4.r95_experiment')
            module.return_value.validate_checkpoint_config.assert_called_once_with(result[1])
            final = reporting.read_json(self.root / 'final_metrics.json')
            final['test_read'] = False
            reporting.write_json(self.root / 'final_metrics.json', final)
            with self.assertRaisesRegex(ValueError, 'finalized'):
                reporting.locked_inputs(self.root.parent, 'A')

    def test_finalized_test_rejects_missing_marker_partial_rows_or_changed_state(self):
        lock, test = finalized_fixture(self.root)
        reporting.finalized_test(self.root, config(), lock, test)
        for split, field, value in (('test', 'n', 4499), ('train', 'n', 8100)):
            lock, test = finalized_fixture(self.root)
            final = reporting.read_json(self.root / 'final_metrics.json')
            final[split]['per_problem']['TSP'][field] = value
            reporting.write_json(self.root / 'final_metrics.json', final)
            if split == 'test':
                test = final['test']
            with self.assertRaises(ValueError):
                reporting.finalized_test(self.root, config(), lock, test)
        lock, test = finalized_fixture(self.root)
        (self.root / 'test_complete.json').unlink()
        with self.assertRaises(FileNotFoundError):
            reporting.finalized_test(self.root, config(), lock, test)

    def test_sampling_reads_actual_receipt_schema_without_arrays_or_model_calls(self):
        proposal_fixture(self.root, 'B')
        with patch.object(reporting.importlib, 'import_module', side_effect=AssertionError('No model import')):
            result = reporting.sampling_diagnostics(self.root, config('B'), dict(state_hash='state'))
        self.assertEqual(len(result['proposal_rows']), 8 * 18)
        self.assertEqual(len(result['optimization_rows']), 40)
        self.assertEqual(len(result['source_sha256']), 8)
        self.assertFalse(result['raw_arrays_read'])
        self.assertEqual(sum(row['applied_to_draws'] for row in result['proposal_rows']), 7 * 18)
        proposal_fixture(self.root, 'A')
        result = reporting.sampling_diagnostics(self.root, config(), dict(state_hash='state'))
        self.assertFalse(any(row['applied_to_draws'] for row in result['proposal_rows']))

    def test_sampling_rejects_foreign_arm_heldout_incomplete_and_nonfinite_summaries(self):
        proposal_fixture(self.root)
        path = self.root / 'proposals/update0007200.json'
        original = reporting.read_json(path)
        for key, bad in (('arm', 'B'), ('split', 'val'), ('complete', False), ('state_hash', 'other')):
            reporting.write_json(path, dict(original, **{key: bad}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.sampling_diagnostics(self.root, config(), dict(state_hash='state'))
        changed = copy.deepcopy(original)
        changed['per_problem']['TSP']['q_min'] = .1 / 81000
        reporting.write_json(path, changed)
        with self.assertRaises(ValueError):
            reporting.sampling_diagnostics(self.root, config(), dict(state_hash='state'))
        reporting.write_json(path, original)
        source = self.root / 'evaluation/train/update0007200/identity.json'
        reporting.write_json(source, dict(changed=True))
        with self.assertRaises(ValueError):
            reporting.sampling_diagnostics(self.root, config(), dict(state_hash='state'))

    def test_cpu_admin_rejects_gpu_before_asset_reads(self):
        with patch.object(reporting, 'locked_inputs') as load:
            for action in (reporting.report, reporting.compare):
                with self.assertRaises(ValueError):
                    action(SimpleNamespace(device='cuda:0'))
            load.assert_not_called()

    def test_lib_only_uses_r95_factory_not_pdf_or_fitting(self):
        args = reporting.parser().parse_args(['lib', '--arm', 'A', '--device', 'cuda:0'])
        lock, test = finalized_fixture(self.root)
        with patch.object(reporting, 'locked_inputs', return_value=(self.root, config(), lock, test, {})), \
                patch.object(reporting.previous, 'validate_nss_reference', return_value={}), \
                patch.object(reporting.common, 'r80_lib', return_value={'TSPLIB': {}, 'CVRPLIB': {}}) as lib, \
                patch.object(reporting.shared, 'build_reports') as pdf:
            reporting.lib_stage(args)
        self.assertEqual(lib.call_args.kwargs['factory_module'], 'code.V4.r95_experiment')
        pdf.assert_not_called()
        record = reporting.read_json(self.root / 'lib_stage_complete.json')
        self.assertEqual(record['model_state_sha256'], 'same-model')

    def test_historical_target36_never_reselected(self):
        path = self.root / 'targets.json'
        reporting.write_json(path, {})
        with patch.object(reporting.helpers, 'load_targets', return_value=(list(range(36)), dict(path=str(path)))):
            _, identity = reporting.load_targets({}, {}, {})
            self.assertEqual(identity['target_count'], 36)
        with patch.object(reporting.helpers, 'load_targets', return_value=([1], dict(path=str(path)))):
            with self.assertRaises(ValueError):
                reporting.load_targets({}, {}, {})


if __name__ == '__main__':
    unittest.main()
