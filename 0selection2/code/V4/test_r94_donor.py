"""Bounded CPU donor checks; no datasets, solver calls or GPU discovery."""

import copy
import hashlib
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from . import r94_donor as module
from .r94_donor import StaticReLDDonor, build_inputs


class StubEncoder(nn.Module):
    """Attention-like context exposes any accidental padded-node influence."""

    def __init__(self):
        super().__init__()
        self.depot = nn.Linear(2, 128)
        self.customer = nn.Linear(5, 128)
        self.noise = nn.Dropout(.8)
        self.calls = []

    def forward(self, depot_xy, node_xy_demand_tw):
        self.calls.append((depot_xy.detach().clone(), node_xy_demand_tw.detach().clone()))
        hidden = torch.cat((self.depot(depot_xy), self.customer(node_xy_demand_tw)), 1)
        hidden = self.noise(hidden + hidden.mean(1, keepdim=True))
        return hidden, hidden.sum() * 0


def synthetic_batch(problem='OVRPBLTW', lengths=(4, 6, 4), padded=8):
    width = 4 if problem == 'CVRP' else 8
    nodes = torch.zeros(len(lengths), padded, width)
    for row, n in enumerate(lengths):
        nodes[row, :n, 0] = torch.arange(n) / 7 + row / 19
        nodes[row, :n, 1] = torch.arange(n).square() / 11
        nodes[row, 0, 3] = 1
        nodes[row, 1:n, 2] = .125
        if 'B' in problem:
            nodes[row, n - 1, 2] = -.25
        if width == 8:
            if 'L' in problem:
                nodes[row, 1:n, 4] = 7.75
            if 'TW' in problem:
                nodes[row, 1:n, 5] = .35
                nodes[row, 1:n, 6] = 2.5
                nodes[row, 1:n, 7] = 9.25
    lengths = torch.tensor(lengths, dtype=torch.long)
    return dict(problem_id=module.PROBLEM_NAMES.index(problem), kind='coord',
                d_in=width, node=nodes, n=lengths,
                node_mask=torch.arange(padded)[None] < lengths[:, None])


class DonorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.builder = mock.patch.object(module, '_build_encoder', side_effect=StubEncoder).start()
        mock.patch.object(module, '_native_sources', return_value=('a' * 64, {'stub.py': 'b' * 64})).start()
        self.addCleanup(mock.patch.stopall)

    def donor(self):
        return StaticReLDDonor('random')

    def test_explicit_native_fields_preserve_units_and_signed_demand(self):
        batch = synthetic_batch(lengths=(4, 4))
        inputs = build_inputs(batch)
        self.assertEqual(set(inputs), {'depot_xy', 'node_xy_demand_tw'})
        torch.testing.assert_close(inputs['depot_xy'], batch['node'][:, :1, :2], rtol=0, atol=0)
        torch.testing.assert_close(inputs['node_xy_demand_tw'], batch['node'][:, 1:4, [0, 1, 2, 6, 7]], rtol=0, atol=0)
        self.assertEqual(float(inputs['node_xy_demand_tw'][0, -1, 2]), -.25)
        changed = copy.deepcopy(batch)
        changed['node'][:, 1:4, 4:6] *= 3
        torch.testing.assert_close(inputs['node_xy_demand_tw'], build_inputs(changed)['node_xy_demand_tw'], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'exact valid-N'):
            build_inputs(synthetic_batch())

    def test_all_16_tasks_and_disabled_zero_tw(self):
        donor = self.donor()
        self.assertEqual(len(module.SUPPORTED_PROBLEMS), 16)
        for problem in module.SUPPORTED_PROBLEMS:
            with self.subTest(problem=problem):
                batch = synthetic_batch(problem, lengths=(4,))
                args = build_inputs(batch)
                if 'TW' not in problem:
                    self.assertEqual(int(args['node_xy_demand_tw'][..., 3:].count_nonzero()), 0)
                self.assertEqual(donor(batch).shape, (1, 8, 128))

    def test_group_by_true_n_not_padding_and_preserve_row_order(self):
        donor, batch = self.donor(), synthetic_batch()
        expected = donor(batch)
        self.assertEqual([call[1].shape[:2] for call in donor.encoder.calls], [(2, 3), (1, 5)])
        self.assertEqual(int(expected[~batch['node_mask']].count_nonzero()), 0)
        changed = copy.deepcopy(batch)
        changed['node'][~batch['node_mask']] = float('nan')
        torch.testing.assert_close(expected, donor(changed), rtol=0, atol=0)
        for row in range(3):
            one = dict(batch, node=batch['node'][row:row + 1], n=batch['n'][row:row + 1],
                       node_mask=batch['node_mask'][row:row + 1])
            torch.testing.assert_close(expected[row:row + 1], donor(one), rtol=1e-6, atol=1e-6)

    def test_nonprefix_mask_and_customer_permutation_keep_identities(self):
        donor, batch = self.donor(), synthetic_batch(lengths=(4,))
        baseline = donor(batch)
        order = torch.tensor([0, 7, 3, 6, 1, 5, 2, 4])
        shuffled = dict(batch, node=batch['node'][:, order], node_mask=batch['node_mask'][:, order])
        torch.testing.assert_close(donor(shuffled), baseline[:, order], rtol=1e-6, atol=1e-6)

    def test_frozen_parent_training_and_no_forward_hashing(self):
        donor, batch = self.donor(), synthetic_batch()
        parent = nn.ModuleDict({'donor': donor})
        parent.train()
        self.assertTrue(parent.training)
        self.assertTrue(all(not child.training for child in donor.modules()))
        self.assertTrue(all(not p.requires_grad for p in donor.parameters()))
        batch['node'].requires_grad_(True)
        with mock.patch.object(module, '_sha256', side_effect=AssertionError('forward file hash')):
            first, second = donor(batch), donor(batch)
        self.assertFalse(first.requires_grad)
        self.assertTrue(all(p.grad is None for p in donor.parameters()))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_local_tf32_policy_restored_on_success_and_exception_without_cuda_init(self):
        donor, batch = self.donor(), synthetic_batch()
        forward = donor.encoder.forward
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                  torch.get_float32_matmul_precision())

        def checked(*args, **kwargs):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
            return forward(*args, **kwargs)

        try:
            torch.set_float32_matmul_precision('medium')
            torch.backends.cudnn.allow_tf32 = True
            with mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA init')), mock.patch.object(donor.encoder, 'forward', side_effect=checked):
                donor(batch)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), 'medium')
            with mock.patch.object(donor.encoder, 'forward', side_effect=RuntimeError('native failure')):
                with self.assertRaisesRegex(RuntimeError, 'native failure'):
                    donor(batch)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), 'medium')
        finally:
            torch.backends.cuda.matmul.allow_tf32 = before[0]
            torch.set_float32_matmul_precision(before[2])
            torch.backends.cudnn.allow_tf32 = before[1]
        self.assertEqual(donor.SPEC['native_precision']['dtype'], 'float32')
        self.assertFalse(donor.SPEC['native_precision']['matmul_tf32'])

    def test_rng_isolation_and_fixed_seed(self):
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with mock.patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('CUDA RNG')):
            first = self.donor()
        self.assertEqual(random.getstate(), py_state)
        for value, expected in zip(np.random.get_state(), np_state):
            np.testing.assert_equal(value, expected)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        torch.rand(7)
        second = self.donor()
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(first.provenance()['random_seed'], 940002)

    def test_supervision_never_read(self):
        class Poison(dict):
            def __getitem__(self, key):
                if key in ('labels', 'winner', 'costs', 'ind', 'routes'):
                    raise AssertionError('supervision/state read')
                return super().__getitem__(key)

        donor = self.donor()
        batch = Poison(synthetic_batch())
        for key in ('labels', 'winner', 'costs', 'ind', 'routes'):
            batch[key] = object()
        torch.testing.assert_close(donor(batch), donor(module.input_only(batch)), rtol=0, atol=0)

    def test_fail_closed_on_unsupported_or_ambiguous_fields(self):
        donor = self.donor()
        for problem in ('TSP', 'ATSP', 'UNKNOWN'):
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                donor(dict(synthetic_batch(), problem_name=problem))
        cases = []
        bad = synthetic_batch('OVRP', lengths=(4,)); bad['node'][0, 1, 7] = 1; cases.append(bad)
        bad = synthetic_batch(); bad['n'][0] = 3; cases.append(bad)
        bad = synthetic_batch(); bad['node'][0, 1, 3] = 1; cases.append(bad)
        bad = synthetic_batch(); bad['node'][0, 0, 6] = 3; cases.append(bad)
        bad = synthetic_batch(); bad['node'][0, 1, 7] = 1; cases.append(bad)
        bad = synthetic_batch(); bad['node'][0, 1, 0] = float('inf'); cases.append(bad)
        for batch in cases:
            with self.assertRaises(ValueError):
                donor(batch)

    def test_provenance_defensive_copy_and_input_contract(self):
        donor = self.donor()
        provenance = donor.provenance()
        self.assertEqual(donor.variant, 'random')
        self.assertEqual(donor.covered_problems, module.SUPPORTED_PROBLEMS)
        self.assertIsInstance(donor.covered_problems, tuple)
        for key in ('variant', 'checkpoint_sha256', 'native_source_sha256', 'output_dim',
                    'supported_problems', 'known_unknown_pretraining_source'):
            self.assertIn(key, provenance)
        provenance['supported_problems'].clear()
        provenance['spec']['model_params']['norm'] = 'batch'
        self.assertEqual(len(donor.provenance()['supported_problems']), 16)
        self.assertEqual(donor.provenance()['spec']['model_params']['norm'], 'none')
        json.dumps(donor.provenance())
        self.assertEqual(set(module.input_only(dict(synthetic_batch(), costs=object()))), set(synthetic_batch()))

    def test_pretrained_exact_pin_and_encoder_only_strict_loading(self):
        reference = self.donor()
        state = {'encoder.' + key: value.clone() + 1 for key, value in reference.encoder.state_dict().items()}
        state['decoder.must_not_instantiate'] = torch.ones(1)
        checkpoint = dict(epoch=5000, problem='Train_ALL', model_state_dict=state)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pinned.pt'
            path.write_bytes(b'synthetic pinned checkpoint')
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(module, 'DEFAULT_CHECKPOINT', path), mock.patch.object(module, 'CHECKPOINT_SHA256', sha), mock.patch.object(torch, 'load', return_value=checkpoint) as load:
                donor = StaticReLDDonor('pretrained')
            self.assertEqual(load.call_count, 1)
            self.assertEqual(load.call_args.kwargs['map_location'], 'cpu')
            self.assertEqual(donor.provenance()['checkpoint_sha256'], sha)
            self.assertFalse(hasattr(donor, 'decoder'))
            for key, value in donor.encoder.state_dict().items():
                torch.testing.assert_close(value, state['encoder.' + key], rtol=0, atol=0)
            with mock.patch.object(torch, 'load', side_effect=AssertionError('unverified load')):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    StaticReLDDonor('pretrained', path)

    def test_bad_variant_random_checkpoint_and_encoder_shape(self):
        with self.assertRaises(ValueError):
            StaticReLDDonor('fine-tuned')
        with self.assertRaises(ValueError):
            StaticReLDDonor('random', '/not/allowed')
        donor = self.donor()
        with mock.patch.object(donor.encoder, 'forward', return_value=torch.zeros(2, 4, 127)):
            with self.assertRaisesRegex(ValueError, 'per-node'):
                donor(synthetic_batch())


@unittest.skipUnless(os.environ.get('R94_ACTUAL_CPU_CHECK') == '1', 'optional one-checkpoint tiny CPU witness')
class ActualNativeTest(unittest.TestCase):
    def test_pinned_native_batch_independence_and_raw_signature(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            donor = StaticReLDDonor('pretrained')
            batch = synthetic_batch(lengths=(4, 4), padded=6)
            hidden = donor(batch)
            with torch.no_grad():
                direct = donor.encoder(**build_inputs(batch))[0]
            torch.testing.assert_close(hidden[:, :4], direct, rtol=0, atol=0)
            one = dict(batch, node=batch['node'][:1], n=batch['n'][:1], node_mask=batch['node_mask'][:1])
            # CPU GEMM reduction order changes with B; observed max abs 1.84e-5
            # on this tiny pinned-weight witness, not cross-instance routing.
            torch.testing.assert_close(hidden[:1], donor(one), rtol=2e-5, atol=3e-5)
            donor.train()
            torch.testing.assert_close(hidden, donor(batch), rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad for p in donor.parameters()))
            self.assertEqual(donor.provenance()['checkpoint_sha256'], module.CHECKPOINT_SHA256)
        finally:
            torch.set_num_threads(threads)


@unittest.skipUnless(os.environ.get('R94_ACTUAL_RANDOM_CPU_CHECK') == '1', 'optional tiny native random CPU witness')
class ActualNativeRandomTest(unittest.TestCase):
    def test_native_routing_without_another_weight_read(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with mock.patch.object(torch, 'load', side_effect=AssertionError('random weight read')):
                donor = StaticReLDDonor('random')
            batch = synthetic_batch(lengths=(4, 4), padded=6)
            hidden = donor(batch)
            one = dict(batch, node=batch['node'][:1], n=batch['n'][:1], node_mask=batch['node_mask'][:1])
            torch.testing.assert_close(hidden[:1], donor(one), rtol=2e-5, atol=3e-5)
            donor.train()
            torch.testing.assert_close(hidden, donor(batch), rtol=0, atol=0)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
