"""CPU-only regression checks for the additive R94 timing correction."""

import os
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from torch import nn

from code.V4.diagnostics import r94_online_timing_recovery as recovery
from code.V4.r94_model import EarlyNodeFusionSelector
from code.unified_selector.data import UnifiedProblemDataset
from code.unified_selector.registry import P2I, constraint_bits


class TinyDonor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)

    def forward(self, batch):
        for _ in batch['n'].unique():
            self.encoder(torch.ones(1, 1))
        return batch['node'].new_ones((*batch['node'].shape[:2], 128))


class TinySelector(nn.Module):
    def __init__(self, *, broken=False):
        super().__init__()
        self.donor = TinyDonor()
        self.broken = broken

    def forward(self, batch):
        if batch['problem_id'] not in (P2I['TSP'], P2I['ATSP']) and 'nco_nodes' not in batch:
            if not self.broken:
                self.donor(batch)
        return dict(logits=torch.ones(len(batch['n']), 2),
                    solver_mask=torch.ones(len(batch['n']), 2, dtype=torch.bool))


def tiny_batch(problem='CVRP'):
    return dict(problem_id=P2I[problem], kind='coord', n=torch.tensor([2, 3]),
                node=torch.zeros(2, 3, 4), node_mask=torch.tensor([[True, True, False], [True] * 3]))


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_filter_leaks_and_raw_allowlist_removes_cache_and_labels(self):
        from code.V4.pairwise_objective import inputs_only
        batch = dict(tiny_batch(), costs=object(), ind=object(), winner=object(),
                     nco_nodes=object(), nco_node_mask=object(), nco_other=object())
        self.assertIn('nco_nodes', inputs_only(batch))
        clean = recovery.raw_inputs(batch)
        self.assertNotIn('costs', clean)
        self.assertNotIn('ind', clean)
        self.assertNotIn('winner', clean)
        self.assertFalse(any(key.startswith('nco_') for key in clean))
        self.assertIs(clean['node'], batch['node'])
        self.assertIn('nco_nodes', batch)

    def test_three_scopes_execute_expected_native_calls_for_every_rep(self):
        model = TinySelector().eval()
        batch = dict(tiny_batch(), nco_nodes=torch.zeros(2, 3, 128),
                     nco_node_mask=torch.ones(2, 3, dtype=torch.bool))
        original = batch['nco_nodes'].clone()
        result = recovery.benchmark_problem(model, batch, 'CVRP', torch.device('cpu'))
        for scope in recovery.SCOPES:
            measurement = result['measurements'][scope]
            expected = dict(donor=0, encoder=0) if scope == 'selector_only' else dict(donor=1, encoder=2)
            self.assertEqual(measurement['native_calls'], [expected] * 3)
            self.assertEqual(measurement['warmup_native_calls'], expected)
            self.assertEqual(len(measurement['batch_seconds']), 3)
        self.assertTrue(torch.equal(batch['nco_nodes'], original))
        self.assertFalse(model.donor._forward_hooks)
        self.assertFalse(model.donor.encoder._forward_hooks)

    def test_silent_online_bypass_is_a_hard_failure(self):
        model = TinySelector(broken=True).eval()
        with self.assertRaisesRegex(AssertionError, 'Native execution mismatch'):
            recovery.benchmark_problem(model, tiny_batch(), 'CVRP', torch.device('cpu'))
        self.assertFalse(model.donor._forward_hooks)
        self.assertFalse(model.donor.encoder._forward_hooks)

    def test_raw_tasks_execute_no_donor(self):
        for problem in ('TSP', 'ATSP'):
            result = recovery.benchmark_problem(TinySelector().eval(), tiny_batch(problem),
                                                problem, torch.device('cpu'))
            self.assertFalse(result['donor_applicable'])
            self.assertEqual(result['measurements']['native_donor']['status'], 'not_applicable_raw_only')
            self.assertEqual(result['measurements']['native_donor']['mean_ms_per_instance'], 0.)

    def test_timing_synchronizes_and_excludes_validation(self):
        events = []
        times = iter(range(8))
        model = TinySelector()
        fn = lambda: events.append('forward')
        check = lambda value: events.append('check')
        clock = lambda: (events.append('clock'), next(times))[1]
        with mock.patch.object(torch.cuda, 'synchronize', side_effect=lambda device: events.append('sync')):
            _, result = recovery.timed_call(fn, torch.device('cuda:0'), model,
                                            dict(donor=0, encoder=0), check, clock=clock)
        self.assertEqual(events, ['sync', 'clock', 'forward', 'sync', 'clock', 'check'] * 4)
        self.assertEqual(result['batch_seconds'], [1, 1, 1])

    def test_authorization_is_checked_before_any_cuda_discovery(self):
        args = type('Args', (), dict(gpu_authorized=False, device='cuda:0', arms=['A', 'B']))()
        with mock.patch.object(torch.cuda, 'device_count', side_effect=AssertionError('GPU touched')):
            with self.assertRaises(PermissionError):
                recovery.measure(args)

    def test_input_parser_matches_unmodified_loader_without_real_labels(self):
        examples = {
            'TSP': torch.tensor([[[.1, .2], [.3, .4]]]),
            'ATSP': torch.tensor([[[0., .2], [.3, 0.]]]),
            'CVRP': dict(loc=torch.tensor([[[.2, .3], [.4, .5]]]), demand=torch.tensor([[.1, .2]]),
                         depot=torch.tensor([[[.5, .5]]])),
            'OVRPBLTW': dict(node_xy=[[.2, .3], [.4, .5]], node_demand=[-.1, .2],
                             depot_xy=[[.5, .5]], route_limit=3., service_time=[.2, .2],
                             tw_start=[0., .1], tw_end=[.8, .9])}
        for problem, instance in examples.items():
            dataset = UnifiedProblemDataset.__new__(UnifiedProblemDataset)
            dataset.__dict__.update(problem=problem, split='test', instances=[instance],
                labels={'0': dict(cost=[0.], ind=0)}, K_p=1, pid=P2I[problem],
                cbits=constraint_bits(problem), problem_desc=[], pool_order=[0], mask=[],
                coord_augment=0, base_N=1)
            reference = dataset[0]
            key = 'matrix' if problem == 'ATSP' else 'node'
            actual = recovery.input_tensor(instance, problem, dataset.cbits)
            self.assertTrue(torch.equal(actual, reference[key]), problem)

    def test_frozen_prepare_inputs_calls_real_native_donor_after_sanitizing(self):
        from code.V4.r94_donor import StaticReLDDonor
        donor = StaticReLDDonor('random')
        host = type('Host', (), {})()
        host.training, host.donor = False, donor
        batch = tiny_batch()
        batch['node'][0, 0, 3] = batch['node'][1, 0, 3] = 1.
        with mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA initialized')):
            with recovery.native_calls(host) as calls:
                prepared = EarlyNodeFusionSelector.prepare_inputs(host, recovery.raw_inputs(batch))
        self.assertEqual(calls, dict(donor=1, encoder=2))
        self.assertEqual(prepared['nco_nodes'].dtype, torch.float16)
        self.assertFalse(torch.cuda.is_initialized())

    def test_report_rejects_missing_native_execution_and_wrong_rows(self):
        import copy
        rows = []
        for problem in recovery.PROBLEMS:
            covered = problem not in ('TSP', 'ATSP')
            expected = dict(donor=int(covered), encoder=int(covered))
            measurements = {}
            for scope in recovery.SCOPES:
                seconds = 0. if scope == 'native_donor' and not covered else 1.
                calls = dict(donor=0, encoder=0) if scope == 'selector_only' else expected
                measurements[scope] = dict(batch_seconds=[seconds] * 3, mean_batch_seconds=seconds,
                    mean_ms_per_instance=seconds * 1000 / 128, native_calls=[calls] * 3,
                    warmup_native_calls=calls)
            rows.append(dict(problem=problem, instances=128, measurements=measurements,
                raw_has_nco_fields=False, donor_cache_read=False, online_matches_selector_only=True,
                input_identity=dict(problem=problem, instances=128, row_indices=list(range(128)),
                                    unique_lengths=1)))
        receipt = dict(schema='r94_online_timing_recovery_v1', complete=True, arm='A',
            checkpoint_sha256=recovery.PINS['A'], state_hash='locked', inference_rule='fixed',
            rows_per_task=128, repetitions=3, warmup_batches=1, per_problem=rows)
        lock = dict(state_hash='locked', inference_rule='fixed')
        def read(path):
            return lock if path.name == 'final_test_lock.json' else receipt
        with mock.patch.object(recovery.Path, 'exists', return_value=True), \
                mock.patch.object(recovery, 'read_json', side_effect=read):
            self.assertEqual(len(recovery.corrected_rows('A')), 18)
            original = copy.deepcopy(receipt)
            receipt['per_problem'][1]['measurements']['donor_plus_selector']['native_calls'] = [dict(donor=0, encoder=0)] * 3
            with self.assertRaisesRegex(ValueError, 'missing native execution'):
                recovery.corrected_rows('A')
            receipt.clear()
            receipt.update(original)
            receipt['per_problem'][1]['input_identity']['row_indices'][0] = 128
            with self.assertRaisesRegex(ValueError, 'fixed raw inputs'):
                recovery.corrected_rows('A')


if __name__ == '__main__':
    unittest.main(verbosity=2)
