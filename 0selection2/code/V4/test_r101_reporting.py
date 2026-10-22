"""Small saved-evidence CPU tests; never run model/teacher/LIB inference."""

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import r101_experiment as runner
from . import r101_pretrain as pretrainer
from . import r101_reporting as reporting
from . import r99_reporting, r93_reporting
from .r76_experiment import read_json, write_json
from .test_r101_experiment import SYNTHETIC_SPEC, synthetic_config


def lock(arm='A'):
    return dict(arm=arm, checkpoint_sha256=arm.lower() * 64, state_hash='state-' + arm,
        inference_rule=runner.INFERENCE_RULE, epoch=1, selected_update=1440)


def timing_record(arm='A'):
    value = dict(student=dict(batch_seconds=[.1, .2, .3], mean_seconds_per_instance=.6 / (3 * 128)))
    return dict(experiment='R101', arm=arm, complete=True,
        checkpoint_sha256=lock(arm)['checkpoint_sha256'], state_hash='state-' + arm,
        inference_rule=runner.INFERENCE_RULE, device='cuda:0', repetitions=3, warmup_batches=1,
        scope='GPU-resident original student only', excludes='I/O and host transfer',
        raw_input_source='original_base_cache', student_only=True, teacher_arrays_opened=False,
        online_teacher=False, r94_cached_latency_used=False, full_system_measured=False,
        end_to_end_measured=False, model_reselection=False,
        per_problem=[dict(problem=p, instances=128, **copy.deepcopy(value)) for p in reporting.PROBLEMS])


def install_pretraining_trace(run, config):
    write_json(run / 'pretrain_complete.json', dict(config['pretraining_lineage'],
        selected_without_validation=True, updates_by_problem=dict.fromkeys(pretrainer.COVERED, 800), seconds=12.))
    records = [dict(experiment='R101', arm=config['arm'], successful_updates=step, max_updates=12800,
        problem=pretrainer.COVERED[(step - 1) % 16], sample_exposures=step * 1024,
        loss=.2, grad_norm=.4, lr=pretrainer.learning_rate(step), seconds_per_update=.1, amp_retries=0)
        for step in (1, *range(16, 12801, 16))]
    (run / 'pretrain_updates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    return records


class R101ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.configs = {arm: synthetic_config(cls.temporary.name, arm)[1] for arm in reporting.ARMS}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        for name, value in (('model_spec', copy.deepcopy(SYNTHETIC_SPEC)),
                            ('PARAMETER_COUNT', self.configs['A']['trainable_parameter_count'])):
            guard = patch.object(runner, name, return_value=value) if name == 'model_spec' else patch.object(runner, name, value)
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU report must not initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def config(self, arm='A'):
        return copy.deepcopy(self.configs[arm])

    def test_private_helpers_no_rebinding_of_historical_modules(self):
        self.assertIsNot(reporting.reuse, r99_reporting)
        self.assertIsNot(reporting.r93, r93_reporting)
        self.assertIsNot(reporting.common, r99_reporting.common)
        self.assertIs(reporting.common._distribution.__globals__, vars(reporting.common))
        self.assertIs(reporting.reuse.r93, reporting.r93)
        self.assertEqual(reporting.MODEL_FACTORY, 'code.V4.r101_experiment.load_checkpoint')
        for phrase in ('12800', '57600', 'not total-budget-matched', 'no speed claim from R94 cached latency',
                       'Development-exposed TEST', 'target36', 'No added head/adapter'):
            self.assertIn(phrase, reporting.REFERENCE_BOUNDARY)

    def test_dictionary_validation_and_matched_teacher_only_contrast_no_weights(self):
        a, b = self.config(), self.config('B')
        saved = copy.deepcopy((a, b))
        with patch.object(torch, 'load', side_effect=AssertionError('no model load')):
            receipt = reporting.validate_matched_configs(a, b)
        self.assertTrue(receipt['matched'])
        self.assertFalse(receipt['historical_references_total_budget_matched'])
        self.assertEqual((a, b), saved)
        b['gpu_type'] = '3090'
        reporting.same_data(a, b, matched=True)
        for key, value in (('max_successful_updates', 57599), ('common_trainable_sha256', 'b' * 64),
                ('unchanged_non_encoder_sha256', 'b' * 64), ('downstream_hint_loss', True),
                ('task_order_sha256', 'b' * 64), ('warmup_updates', 0), ('new_control', True)):
            bad = self.config('B')
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.validate_matched_configs(a, bad)

    def test_locked_inputs_late_validation_never_constructs_a_model(self):
        result = Path('/synthetic/A'), self.config(), lock(), {}, {}
        with patch.object(reporting.shared, 'locked_inputs', return_value=result) as load, \
                patch.object(torch, 'load', side_effect=AssertionError('no weights')):
            self.assertEqual(reporting.locked_inputs(Path('/synthetic'), 'A'), result)
            load.assert_called_once_with(Path('/synthetic'), 'A', experiment='R101', arms=reporting.BUDGETS)

    def test_pretraining_curves_from_trace_not_fabricated_or_teacher_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            records = install_pretraining_trace(run, self.config())
            with patch.object(torch, 'load', side_effect=AssertionError('no endpoint/teacher')):
                evidence = reporting.pretraining_evidence(run, self.config())
            self.assertEqual(len(evidence['rows']), 801)
            self.assertFalse(evidence['teacher_validation_used'])
            self.assertIn('not interval mean', evidence['rows'][0]['loss_scope'])
            output = run / 'full_model_statistics'
            output.mkdir()
            reporting.pretraining_curves(output, self.config(), lock(), evidence)
            self.assertGreater((output / 'pretraining_curves.pdf').stat().st_size, 100)
            self.assertEqual(len((output / 'pretraining_curves.csv').read_text(encoding='utf-8-sig').splitlines()), 802)
            (run / 'pretrain_updates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records[:-1]))
            with self.assertRaises(ValueError):
                reporting.pretraining_evidence(run, self.config())

    def test_timing_requires_original_student_full18_and_exact_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, record = Path(tmp), timing_record()
            write_json(run / 'inference_timing.json', record)
            result = reporting.inference_timing(run, lock())
            self.assertEqual(len(result['measurements']), 18)
            self.assertFalse(result['r94_cached_latency_used'])
            for key, value in (('teacher_arrays_opened', True), ('online_teacher', True),
                    ('r94_cached_latency_used', True), ('raw_input_source', 'R94_cache'),
                    ('per_problem', record['per_problem'][:-1]), ('checkpoint_sha256', 'different')):
                write_json(run / 'inference_timing.json', dict(record, **{key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    reporting.inference_timing(run, lock())
            bad = copy.deepcopy(record)
            bad['per_problem'][0]['student']['mean_seconds_per_instance'] *= 2
            write_json(run / 'inference_timing.json', bad)
            with self.assertRaises(ValueError):
                reporting.inference_timing(run, lock())

    def test_real_paired_rescues_losses_and_false_pick_original_cost(self):
        def predictions(pred):
            return {p: SimpleNamespace(pool=['m0', 'm1'], pred=np.array(pred), winner=np.array([0, 1, 0]),
                costs=np.array([[1., 3.], [3., 1.], [1., 4.]]), n=3, indices=np.arange(3),
                source_indices=np.arange(3), pool_ids=np.arange(2)) for p in reporting.PROBLEMS}
        old = {s: predictions([1, 0, 0]) for s in ('TRAIN', 'VAL', 'TEST')}
        new = {s: predictions([0, 1, 1]) for s in ('TRAIN', 'VAL', 'TEST')}
        targets = [(p, method) for p in reporting.PROBLEMS for method in ('m0', 'm1')]
        identity = dict(identity=dict(rule='fixed R84 target36'),
            categories={p + '/' + m: 'zero' for p, m in targets})
        pairs, rows, totals = reporting.comparison_tables(old, new, targets, identity, 'R101/B', 'R101/A')
        row = next(r for r in rows if r['split'] == 'TEST' and r['problem'] == 'TSP' and r['solver'] == 'm1')
        self.assertEqual(row['rescued'], 1)
        self.assertEqual(row['new_false_pick_count'], 1)
        self.assertEqual(row['new_false_pick_mean_cost'], 4.)
        self.assertAlmostEqual(row['new_false_pick_cost_delta_contribution'], 1.)
        self.assertEqual(totals[-1]['rescued'], 36)
        self.assertEqual(totals[-1]['lost'], 18)
        self.assertAlmostEqual(totals[-1]['new_false_pick_cost_contribution'], 4. / 3)
        self.assertEqual(next(r for r in pairs if r['split'] == 'TEST' and r['problem'] == 'ALL')['net_correct'], 18)

    def test_same_axis_native_distribution_top1_test_then_lib(self):
        import matplotlib.pyplot as plt
        helper = reporting.common.reporting_helpers()
        helper.setup_style()
        test = helper.Predictions(Path('/synthetic'), ['m0', 'm1'], np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]), np.ones((4, 2)), np.arange(4), None, np.arange(2), None)
        for problem, dataset, n in (('TSP', 'TSPLIB', 49), ('CVRP', 'CVRPLIB', 100)):
            lib = helper.Predictions(Path('/synthetic/lib'), test.pool, np.zeros(n, dtype=int),
                np.arange(n) % 2, np.ones((n, 2)), np.arange(n), None, np.arange(2), None)
            panels = helper.make_panels(problem, test, lib, test.pool)
            fig = reporting.common._distribution(helper, problem, test.pool, panels, 'NEW90k', 'R101/A', lock())
            try:
                self.assertIn(problem + ' TEST', fig.axes[0].get_title())
                self.assertIn(dataset, fig.axes[1].get_title())
                for axis, data in zip(fig.axes, (test, lib)):
                    self.assertIn('\u6a21\u578bTop1=' + f'{data.top1_pct:.2f}%', axis.get_title())
                    self.assertEqual(len(axis.patches), 4)
                self.assertTrue(all(r['top1'] is None for r in panels if r['kind'] == 'true_winner'))
            finally:
                plt.close(fig)

    def test_gpu_report_rejected_and_missing_evidence_not_marked_complete(self):
        with patch.object(reporting, 'locked_inputs') as load:
            for function in (reporting.report, reporting.compare):
                with self.assertRaises(ValueError):
                    function(SimpleNamespace(device='cuda:0'))
            load.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = reporting.parser().parse_args(['report', '--arm', 'A'])
            with patch.object(reporting, 'locked_inputs', return_value=(run, self.config(), lock(), {}, {})), \
                    patch.object(reporting, '_completed_lib'), patch.object(reporting.shared, 'build_reports') as render:
                with self.assertRaises(FileNotFoundError):
                    reporting.report(args)
            render.assert_not_called()
            self.assertEqual(read_json(run / 'postprocess_status.json')['exit_status'], 1)

    def test_lib_factory_original_student_and_both_complete_publication_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = reporting.parser().parse_args(['lib', '--arm', 'A', '--device', 'cuda:0'])
            result = run, self.config(), lock(), dict(identity=dict(model_state_sha256='same')), {}
            with patch.object(reporting, 'locked_inputs', return_value=result), \
                    patch.object(reporting.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(reporting.common, 'r80_lib', return_value={'TSPLIB': {}, 'CVRPLIB': {}}) as lib:
                reporting.lib_stage(args)
                self.assertEqual(lib.call_args.kwargs['factory_module'], 'code.V4.r101_experiment')
            self.assertFalse(read_json(run / 'lib_stage_complete.json')['online_teacher'])
            def inputs(root, arm):
                return run / arm, self.config(arm), lock(arm), {}, {}
            with patch.object(reporting, 'locked_inputs', side_effect=inputs), \
                    patch.object(reporting, 'completed_arm_report', side_effect=ValueError('incomplete B')), \
                    patch.object(reporting, 'selected_predictions') as selected:
                with self.assertRaises(ValueError):
                    reporting.compare(SimpleNamespace(device='cpu', root=run))
                selected.assert_not_called()
            self.assertFalse((run / 'RESULTS.pdf').exists())

    def test_single_pdf_merge_can_prepend_disclosure_to_existing_pdf(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / 'provenance.pdf', root / 'model.pdf']
            for path, text in zip(paths, ('SYNTHETIC Stage-I disclosure', 'SYNTHETIC Stage-II results')):
                fig = plt.figure(figsize=(4, 2))
                fig.text(.1, .5, text)
                try:
                    fig.savefig(path)
                finally:
                    plt.close(fig)
            reporting.merge_pdfs(paths[1], paths)
            extracted = subprocess.run(['pdftotext', str(paths[1]), '-'],
                check=True, capture_output=True, text=True).stdout
            self.assertIn('Stage-I disclosure', extracted)
            self.assertIn('Stage-II results', extracted)


if __name__ == '__main__':
    unittest.main()
