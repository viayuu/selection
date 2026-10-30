"""Bounded CPU fixtures: no real learner, collector, features or GPU execution."""

from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from . import r121_finish as finish


def natural_fixture():
    return dict(costs=np.array([[1.1, 1., 1.3], [1., 1.2, 1.4], [1.3, 1.2, 1.], [1.1, 1., 1.2]], dtype=np.float64),
        winner=np.array([1, 0, 2, 1], dtype=np.int64), pool=['a', 'b', 'c'], pool_ids=np.array([0, 1, 2], dtype=np.int64),
        scores=np.array([[1., 3., 0.], [1., 3., 0.], [3., 1., 0.], [3., 1., 0.]], dtype=np.float32),
        pred=np.array([1, 1, 0, 0]), indices=np.arange(4), source_indices=np.arange(4),
        release_source_indices=np.arange(100, 104), path='/SYNTHETIC_CPU_ONLY.npz')


def family_fixture():
    costs = np.array([[1. + i * .01, 1.2 + i * .03, 1.5 - i * .01] for i in range(10)], dtype=np.float64)
    target = np.log(costs)
    target = (target - target.mean(1, keepdims=True)) / .01
    return dict(costs=costs, winner=costs.argmin(1).astype(np.int64), advantage_units=target.astype(np.float32),
        indices=np.arange(10), endpoint_ids=np.array(['endpoint_' + str(i) for i in range(10)]),
        partition=np.array(['family_train'] * 5 + ['family_val'] * 5),
        family_id=np.array(['family_train_1'] * 5 + ['family_val_1'] * 5),
        group_id=np.array(['geometry_train'] * 5 + ['geometry_val'] * 5),
        cell=np.array(['original', 'z00', 'z01', 'z10', 'z11'] * 2),
        edges_train=np.array([[0, i] for i in range(1, 5)]), edges_val=np.array([[5, i] for i in range(6, 10)]))


def config_fixture():
    return dict(data_root='/SYNTHETIC_CPU_ONLY/new90k', experiment='R121', arm='principal',
        baselines={p: dict(sbs_solver='a', sbs_slot=0, selection_split='train',
            mean_cost_by_solver=dict(a=1.1, b=1.2, c=1.3)) for p in finish.PROBLEMS},
        candidate_pools={p: ['a', 'b', 'c'] for p in finish.PROBLEMS},
        candidate_pool_ids={p: [0, 1, 2] for p in finish.PROBLEMS},
        natural_ce_isolation=dict(scope='SYNTHETIC_FIXTURE_ONLY'),
        family_population=dict(per_problem={p: {s: dict(endpoints=5) for s in ('family_train', 'family_val')}
                                           for p in finish.PROBLEMS}),
        natural_ce_eligible_rows_per_problem=dict.fromkeys(finish.PROBLEMS, 3))


class FinishTests(unittest.TestCase):
    def test_common_cli_compatibility_and_cpu_describe(self):
        args = finish.parser().parse_args(['capture-train', '--root', str(finish.DEFAULT_ROOT),
            '--device', 'cuda:0', '--gpu-type', '4090', '--cache-root', '/tmp/shiys_R84_new90k_cache',
            '--test-cache-root', '/tmp/shiys_R84_new90k_cache', '--family-labels', '/fixture/family_labels.jsonl',
            '--family-cache', '/fixture/family_cache', '--release', '/fixture/main_r120_release.json',
            '--binding', '/fixture/main_binding_lane0.json', '--micro-cap', '352', '--eval-batch-size', '256',
            '--lib-batch-size', '1', '--wandb-mode', 'offline'])
        self.assertEqual((args.micro_cap, args.eval_batch_size, args.wandb_mode), (352, 256, 'offline'))
        with redirect_stdout(StringIO()):
            value = finish.main(['describe', '--root', str(finish.DEFAULT_ROOT)])
        self.assertFalse(value['inference_started'])
        self.assertIn('principal/learner_finish/reports/receipt.json', value['receipt'])

    def test_gate_precedes_model_cache_and_authorization(self):
        base = SimpleNamespace(locked_training=Mock(side_effect=PermissionError('not locked')),
            authorized=Mock(), load_checkpoint=Mock())
        with patch.object(finish, 'learner', return_value=base):
            with self.assertRaises(PermissionError):
                finish.capture_train(SimpleNamespace(root=Path('/fixture')))
        base.authorized.assert_not_called()
        base.load_checkpoint.assert_not_called()
        with self.assertRaises(PermissionError):
            finish.finish(SimpleNamespace(device='cuda:0'))

    def test_family_mean_is_train_only_and_endpoint_weights_are_explicit(self):
        data = family_fixture()
        rows = finish.family_statistics(data)
        model = [r for r in rows if r['model'] == 'locked_model']
        self.assertTrue(all(r['endpoint_huber_unique'] < 1e-9 and r['response_huber'] < 1e-9 for r in model))
        for part in ('family_train', 'family_val'):
            zero = next(r for r in rows if r['partition'] == part and r['model'] == 'zero_potential')
            mean = next(r for r in rows if r['partition'] == part and r['model'] == 'family_train_mean_potential')
            self.assertEqual(zero['response_huber'], mean['response_huber'])
            self.assertGreater(zero['endpoint_huber_unique'], 0.)
        old = next(r['train_mean_potential_units'] for r in rows if r['model'] == 'family_train_mean_potential')
        data['costs'][5:, 1] *= 3.
        changed = finish.family_statistics(data)
        self.assertTrue(all(r['train_mean_potential_units'] == old for r in changed if r['model'] == 'family_train_mean_potential'))
        data = family_fixture()
        data['advantage_units'][0] += [2., -2., 0.]
        row = next(r for r in finish.family_statistics(data) if r['partition'] == 'family_train' and r['model'] == 'locked_model')
        self.assertAlmostEqual(row['endpoint_huber_edge_slots'], row['endpoint_huber_unique'] * 2.5, places=6)
        data['group_id'][5:] = data['group_id'][0]
        with self.assertRaisesRegex(ValueError, 'crosses'):
            finish.family_statistics(data)

    def test_family_forward_allowlist_and_variable_microbatch(self):
        import torch
        torch.set_num_threads(1)
        sizes = []
        cache = SimpleNamespace(size=5)
        def batch(rows, device):
            return dict(n=torch.ones(len(rows), dtype=torch.long), kind='coord',
                costs=torch.ones(len(rows), 3, dtype=torch.float64), ind=torch.zeros(len(rows), dtype=torch.long),
                endpoint_id=['hidden'] * len(rows), family_partition=['family_val'] * len(rows),
                unknown_supervision=torch.ones(len(rows)))
        cache.batch = batch
        def model(query):
            self.assertTrue({'costs', 'ind', 'endpoint_id', 'family_partition', 'unknown_supervision'}.isdisjoint(query))
            sizes.append(len(query['n']))
            return dict(logits=torch.zeros(len(query['n']), 3), advantage_units=torch.ones(len(query['n']), 3))
        with torch.inference_mode():
            values = finish.family_forward(model, cache, 3, torch.device('cpu'))
        self.assertEqual(sizes, [3, 2])
        self.assertEqual(values['advantage_units'].shape, (5, 3))
        self.assertGreater(float(values['forward_seconds']), 0.)

    def test_full_train_and_eligible_metrics_are_not_merged(self):
        data = natural_fixture()
        predictions = dict(TRAIN={p: data for p in finish.PROBLEMS})
        eligible = {p: np.array([0, 2, 3]) for p in finish.PROBLEMS}
        rows, results = finish.natural_metrics(predictions, config_fixture(), dict(checkpoint_sha256='fixture'), eligible)
        full = next(r for r in rows if r['problem'] == 'ALL' and r['split'] == 'TRAIN')
        kept = next(r for r in rows if r['problem'] == 'ALL' and r['split'] == 'TRAIN_ELIGIBLE')
        self.assertEqual((full['n'], kept['n']), (72, 54))
        self.assertEqual(full['top1_pct'], 25.)
        self.assertAlmostEqual(kept['top1_pct'], 100. / 3.)
        self.assertNotEqual(results['TRAIN']['macro'], results['TRAIN_ELIGIBLE']['macro'])
        np.testing.assert_array_equal(finish.subset(data, eligible['TSP'])['costs'], data['costs'][[0, 2, 3]])

    def test_eligible_subset_uses_saved_source_mapping_and_hashes(self):
        base, config = finish.learner(), config_fixture()
        data = {p: natural_fixture() for p in finish.PROBLEMS}
        keep = np.array([0, 2, 3])
        isolation = base.structural.seal(dict(per_problem={p: dict(excluded_local_rows=[1],
            eligible_local_rows_sha256=base.index_hash(keep),
            source_index_array_sha256=base.index_hash(data[p]['release_source_indices'])) for p in finish.PROBLEMS}), 'isolation_sha256')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'isolation.json'
            finish.write_json(path, isolation)
            config['natural_ce_isolation'] = dict(path=str(path), sha256=finish.sha(path),
                                                  isolation_sha256=isolation['isolation_sha256'])
            result = finish.eligible_indices(config, data, {})
            np.testing.assert_array_equal(result['TSP'], keep)
            data['TSP']['release_source_indices'][0] += 1
            with self.assertRaisesRegex(ValueError, 'Eligible-gradient'):
                finish.eligible_indices(config, data, {})

    def test_fixed36_true_recovery_and_false_picks_not_selection_breadth(self):
        helper = finish.helpers()
        data = natural_fixture()
        before = dict(data, pred=np.array([0, 0, 1, 0]))
        targets = [(p, s) for p in finish.PROBLEMS for s in ('a', 'b')]
        details, summaries = helper.recovery_tables(dict.fromkeys(finish.PROBLEMS, before),
            dict.fromkeys(finish.PROBLEMS, data), targets, 'R121/principal vs R84A', 'TEST')
        self.assertEqual(len(details), 36)
        full = next(r for r in summaries if r['problem'] == 'ALL' and r['solver'] == 'FULL18')
        self.assertEqual((full['corrected'], full['harmed'], full['net_correct']), (18, 18, 0))
        self.assertGreater(full['new_false_pick_count'], 0)
        for p in finish.PROBLEMS:
            group = {r['solver']: r for r in summaries if r['problem'] == p}
            self.assertAlmostEqual(group['FIXED36']['mean_cost_delta_contribution'] + group['OUTSIDE36']['mean_cost_delta_contribution'],
                                   group['FULL18']['mean_cost_delta_contribution'])

    def test_saved_only_finish_receipt_and_pdf(self):
        actual_base, helper = finish.learner(), finish.helpers()
        config = config_fixture()
        lock = dict(checkpoint_sha256='SYNTHETIC_CHECKPOINT', state_hash='SYNTHETIC_STATE', selected_update=720)
        identity = finish.identity_for(lock, 'SYNTHETIC_MODEL_STATE')
        predictions = {s: {p: natural_fixture() for p in finish.PROBLEMS} for s in ('TRAIN', 'VAL', 'TEST')}
        eligible = {p: np.array([0, 2, 3]) for p in finish.PROBLEMS}
        _, results = finish.natural_metrics(predictions, config, lock, eligible)
        for result in results.values():
            result['identity'] = dict(model_state_sha256=identity['model_state_sha256'])
        targets = [(p, s) for p in finish.PROBLEMS for s in ('a', 'b')]
        details, summary, paired = [], [], []
        for label, _, _, _ in finish.HISTORICAL:
            for split in ('VAL', 'TEST'):
                old = {p: dict(data, pred=np.array([0, 0, 1, 0])) for p, data in predictions[split].items()}
                comparison = 'R121/principal vs ' + label
                d, s = helper.recovery_tables(old, predictions[split], targets, comparison, split)
                details.extend(d)
                summary.extend(s)
                paired.extend(helper.paired_row(old[p], predictions[split][p], comparison=comparison,
                                                split=split, problem=p) for p in finish.PROBLEMS)
        tables = dict(target_recovery=details, target_recovery_summary=summary, paired_changes=paired,
            historical_references=[dict(model=label, status='complete') for label, _, _, _ in finish.HISTORICAL],
            natural_fixed36_error_budgets=helper.error_budgets(predictions, targets))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'principal'
            output = run / 'learner_finish'
            train = dict(identity, complete=True, all18=True, train_rows_per_problem=81000,
                train_eligible_rows_per_problem=config['natural_ce_eligible_rows_per_problem'], result=results['TRAIN'],
                natural_evaluation_seconds=1., family_forward_seconds=.1, input_files_sha256={}, artifacts={})
            finish.write_json(output / 'capture_train_receipt.json', train)
            finish.write_json(output / 'family_capture_receipt.json', dict(identity, complete=True, input_files_sha256={}, artifacts={}))
            def base_finish(args):
                finish.write_json(run / 'delivery_manifest.json', dict(complete=True, main_report_preserved=True))
                return dict(complete=True)
            base = SimpleNamespace(natural=actual_base.natural, finish=Mock(side_effect=base_finish),
                load_checkpoint=Mock(side_effect=AssertionError('finish must not restore weights')),
                evaluate=Mock(side_effect=AssertionError('finish must not forward')))
            with ExitStack() as stack:
                stack.enter_context(patch.object(finish, 'learner', return_value=base))
                stack.enter_context(patch.object(finish, 'locked', return_value=(run, config, lock, [], results['TEST'], results['VAL'])))
                stack.enter_context(patch.object(finish, 'load_natural', side_effect=lambda run, config, lock, result, state, split, inputs: predictions[split]))
                stack.enter_context(patch.object(finish, 'eligible_indices', return_value=eligible))
                stack.enter_context(patch.object(finish, 'load_family', return_value={p: family_fixture() for p in finish.PROBLEMS}))
                stack.enter_context(patch.object(finish, 'mechanism_tables', return_value=(tables, dict(path='/fixture/target_list.json', sha256=finish.TARGET_SHA))))
                forward = stack.enter_context(patch.object(finish, 'family_forward', side_effect=AssertionError('finish must not forward')))
                receipt = finish.finish(SimpleNamespace(root=Path(tmp), device='cpu'))
            base.finish.assert_called_once()
            base.load_checkpoint.assert_not_called()
            base.evaluate.assert_not_called()
            forward.assert_not_called()
            self.assertEqual(receipt['phase'], 'complete_post_TEST_learner_finish')
            self.assertFalse(receipt['model_inference'])
            self.assertFalse(receipt['GPU_execution'])
            self.assertEqual(set(receipt['artifacts']), set(finish.REPORT_FILES))
            pdf = output / 'reports/R121_MECHANISM.pdf'
            self.assertTrue(pdf.read_bytes().startswith(b'%PDF-'))
            self.assertEqual(finish.sha(pdf), receipt['artifacts'][pdf.name]['sha256'])


if __name__ == '__main__':
    unittest.main()
