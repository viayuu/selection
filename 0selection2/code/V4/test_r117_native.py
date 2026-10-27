"""Three bounded CPU-only native checks, using synthetic N<=5, not datasets."""

import json
import random
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from . import r117_native as module


def native_batch(problem):
    generator = torch.Generator(device='cpu').manual_seed(117)
    lengths = torch.tensor([5, 3, 5])
    mask = torch.arange(7)[None] < lengths[:, None]
    mask[0, 1], mask[0, 6] = False, True
    batch = dict(problem_name=problem, problem_id=module.PROBLEM_NAMES.index(problem),
                 n=lengths, node_mask=mask)
    if problem == 'ATSP':
        matrix = torch.rand(3, 7, 7, generator=generator)
        matrix.diagonal(dim1=1, dim2=2).zero_()
        matrix[~(mask[:, :, None] & mask[:, None, :])] = float('nan')
        batch['matrix'] = matrix.requires_grad_(True)
    else:
        width = 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
        node = torch.zeros(3, 7, width)
        node[..., :2] = torch.rand(3, 7, 2, generator=generator)
        if width > 2:
            node[:, 1:, 2] = .1
            if 'B' in problem:
                node[:, 2, 2] = -.1
            node[:, 0, 3] = 1.
        if width == 8:
            if 'L' in problem:
                node[:, 1:, 4] = 2.5
            if 'TW' in problem:
                node[:, 1:, 5:8] = torch.tensor([.05, .1, 1.])
        node[~mask] = float('nan')
        batch['node'] = node.requires_grad_(True)
    batch.update(cost=object(), costs=object(), ind=object(), labels=object(), winner=object())
    return batch


class GuardedBatch(dict):
    def __contains__(self, key):
        if key not in module.INPUT_ALLOWLIST:
            raise AssertionError('Queried forbidden input: ' + key)
        return super().__contains__(key)

    def __getitem__(self, key):
        if key not in module.INPUT_ALLOWLIST:
            raise AssertionError('Read forbidden input: ' + key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in module.INPUT_ALLOWLIST:
            raise AssertionError('Read forbidden input: ' + key)
        return super().get(key, default)

    def items(self):
        raise AssertionError('Do not enumerate supervision')

    def __iter__(self):
        raise AssertionError('Do not enumerate supervision')


class NativeComputationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.no_cuda = mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('No GPU'))
        cls.no_cuda.start()
        cls.addClassCleanup(cls.no_cuda.stop)
        cls.addClassCleanup(torch.set_num_threads, cls.old_threads)
        py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        cls.a, cls.b = module.NativeComputation('A'), module.NativeComputation('B')
        cls.rng_preserved = (py == random.getstate() and np.array_equal(numpy[1], np.random.get_state()[1])
                             and numpy[2:] == np.random.get_state()[2:]
                             and torch.equal(cpu, torch.get_rng_state()))

    def test_finite_physical_alignment_no_padding_execution_or_label_access(self):
        for donor in (self.a, self.b):
            problems = module.PROBLEM_NAMES if donor is self.b else ('TSP', 'CVRP', 'OVRPLTW', 'ATSP')
            for problem in problems:
                batch = native_batch(problem)
                sizes = []
                encoder = (donor.tsp.encoder if problem == 'TSP' else donor.cvrp.encoder
                           if problem == 'CVRP' else donor.atsp_encoder
                           if problem == 'ATSP' else donor.base.encoder)
                hook = encoder.register_forward_pre_hook(lambda _, args: sizes.append(args[0].shape[1]))
                # ReLD has a separate [B,1,2] depot argument; customers carry N-1.
                if problem not in ('TSP', 'CVRP', 'ATSP'):
                    hook.remove()
                    hook = encoder.register_forward_pre_hook(
                        lambda _, args, kwargs: sizes.append(kwargs['node_xy_demand_tw'].shape[1] + 1),
                        with_kwargs=True)
                try:
                    with mock.patch.object(donor.tsp.decoder.Linear_final, 'forward',
                                           side_effect=AssertionError('No final score head')), \
                            mock.patch.object(donor.cvrp.decoder.Linear_final, 'forward',
                                              side_effect=AssertionError('No final score head')):
                        output = donor(GuardedBatch(batch))
                finally:
                    hook.remove()
                self.assertEqual(sizes, [3, 5])
                self.assertEqual(set(output), {'base_nodes', 'native_nodes', 'node_mask'})
                torch.testing.assert_close(output['node_mask'], batch['node_mask'])
                for key, width in (('base_nodes', 128), ('native_nodes', module.WIDTH_BY_PROBLEM[problem])):
                    value = output[key]
                    self.assertEqual(tuple(value.shape), (3, 7, width))
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertFalse(value.requires_grad)
                    self.assertTrue(torch.isfinite(value).all())
                    self.assertEqual(int(value[~batch['node_mask']].count_nonzero()), 0)
                if problem in ('TSP', 'ATSP'):
                    self.assertEqual(int(output['base_nodes'].count_nonzero()), 0)

        # Compare LEHD physical scatter against a hook on the unchanged native
        # decoder forward, including its real gather/endpoint assembly.
        for problem, model in (('TSP', self.b.tsp), ('CVRP', self.b.cvrp)):
            batch = native_batch(problem)
            output = self.b(batch)['native_nodes']
            rows = torch.tensor([0, 2])
            positions = torch.nonzero(batch['node_mask'][rows])[:, 1].reshape(2, 5)
            raw = batch['node'][rows[:, None], positions]
            captured = []
            hook = model.decoder.layers[-1].register_forward_hook(
                lambda _, args, out: captured.append(out.detach()))
            try:
                with torch.no_grad(), module.reld._native_precision('cpu'):
                    if problem == 'TSP':
                        model.decoder(model.encoder(raw), torch.zeros(2, 1, dtype=torch.long))
                    else:
                        model.decoder(model.encoder(raw, capacity=1.), torch.empty(2, 0, dtype=torch.long),
                                      capacity=1., remaining_capacity=torch.ones(2))
            finally:
                hook.remove()
            expected = torch.cat(((captured[0][:, :1] + captured[0][:, -1:]) * .5,
                                  captured[0][:, 1:-1]), 1)
            torch.testing.assert_close(output[rows[:, None], positions], expected, atol=0, rtol=0)

    def test_matched_base_frozen_reference_and_quantized_cache_identity(self):
        self.assertTrue(self.rng_preserved)
        self.assertEqual(self.a.base.provenance(), self.b.base.provenance())
        for key, value in self.a.base.state_dict().items():
            self.assertTrue(torch.equal(value, self.b.base.state_dict()[key]))
        for name in ('tsp', 'cvrp', 'mvrp_decoder', 'atsp_encoder', 'atsp_decoder'):
            self.assertFalse(torch.equal(next(getattr(self.a, name).parameters()),
                                         next(getattr(self.b, name).parameters())))
        for donor in (self.a, self.b):
            donor.train(True)
            self.assertFalse(any(child.training for child in donor.modules()))
            self.assertFalse(any(p.requires_grad for p in donor.parameters()))
            before = {key: value.clone() for key, value in donor.named_buffers()}
            batch = native_batch('OVRPBLTW')
            captured = []

            def capture(_, args, kwargs):
                base, attrs, mask = args[0], args[1], kwargs['ninf_mask']
                self.assertEqual(base.dtype, torch.float32)
                torch.testing.assert_close(base, base.half().float(), atol=0, rtol=0)
                self.assertTrue((attrs[..., 0] == 1).all())
                self.assertTrue((attrs[..., 1] == 0).all())
                self.assertTrue((attrs[..., 3] == 1).all())
                self.assertTrue((attrs[:, 0, 2] == 0).all())
                self.assertTrue((attrs[:, 1:, 2] == 2.5).all())
                diagonal = torch.eye(base.shape[1], dtype=torch.bool)[None].expand(len(base), -1, -1)
                self.assertTrue(torch.equal(torch.isneginf(mask), diagonal))
                captured.append(base.clone())

            hook = donor.mvrp_decoder.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                with mock.patch.object(donor.base, 'forward', wraps=donor.base.forward) as base_forward:
                    online = donor(GuardedBatch(batch))
                    self.assertEqual(base_forward.call_count, 1)
                with mock.patch.object(donor.base, 'forward', side_effect=AssertionError('Recomputed cached base')):
                    cached = donor.forward_extra(GuardedBatch(batch), online['base_nodes'].half().float())
                torch.testing.assert_close(cached, online['native_nodes'], atol=0, rtol=0)
                with mock.patch.object(donor.base, 'forward', wraps=donor.base.forward) as base_forward:
                    fallback = donor.forward_extra(GuardedBatch(batch))
                    self.assertEqual(base_forward.call_count, 1)
                torch.testing.assert_close(fallback, cached, atol=0, rtol=0)
            finally:
                hook.remove()
            self.assertEqual(len(captured), 6)
            for key, value in donor.named_buffers():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
            self.assertIsNone(donor.mvrp_decoder.k)
            provenance = json.loads(json.dumps(donor.provenance(), allow_nan=False))
            self.assertEqual(provenance, donor.provenance())
            self.assertIn('float16 -> float32 BEFORE', provenance['spec']['base_quantization'])
            self.assertFalse(provenance['spec']['historical_cost_recipe_certified'])
            provenance['components']['base']['variant'] = 'tampered'
            self.assertEqual(donor.provenance()['components']['base']['variant'], 'pretrained')

    def test_strict_parent_restore_without_external_weights(self):
        forbidden = {str(record[0]) for record in module.CHECKPOINTS.values()}
        original_open = Path.open

        def guarded(path, *args, **kwargs):
            if str(path) in forbidden:
                raise AssertionError('Restore opened external weights: ' + str(path))
            return original_open(path, *args, **kwargs)

        for donor in (self.a, self.b):
            parent = nn.Module()
            parent.add_module('donor', donor)
            with mock.patch.object(torch, 'load', side_effect=AssertionError('No external weight load')), \
                    mock.patch.object(Path, 'open', guarded):
                restored = nn.Module()
                restored.add_module('donor', module.NativeComputation(
                    donor.arm, restore=True, expected_provenance=donor.provenance()))
            restored.load_state_dict(parent.state_dict(), strict=True)
            self.assertEqual(restored.donor.provenance(), donor.provenance())
            for problem in ('TSP', 'CVRP', 'OVRPLTW', 'ATSP'):
                batch = native_batch(problem)
                expected, actual = donor(batch), restored.donor(batch)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            bad = donor.provenance()
            bad['spec']['vrp_reference']['feasible_execution_claim'] = True
            with self.assertRaisesRegex(ValueError, 'provenance'):
                module.NativeComputation(donor.arm, restore=True, expected_provenance=bad)


if __name__ == '__main__':
    unittest.main()
