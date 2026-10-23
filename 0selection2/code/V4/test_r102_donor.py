"""Few actual native CPU checks on synthetic tiny inputs, not datasets."""

import copy
import random
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from . import r102_donor as module


def native_batch(problem, lengths=(7, 5, 7), padded=9):
    generator = torch.Generator(device='cpu').manual_seed(19)
    mask = torch.arange(padded)[None] < torch.tensor(lengths)[:, None]
    # A non-prefix row catches gather/scatter mistakes; node0 has no depot role.
    if padded > max(lengths):
        mask[0, 1], mask[0, padded - 1] = False, True
    batch = dict(problem_name=problem, problem_id=module.PROBLEM_NAMES.index(problem),
                 n=torch.tensor(lengths), node_mask=mask)
    if problem == 'TSP':
        values = torch.rand(len(lengths), padded, 2, generator=generator)
        values[~mask] = float('nan')
        batch.update(kind='coord', d_in=2, node=values)
    else:
        values = torch.rand(len(lengths), padded, padded, generator=generator)
        valid = mask[:, :, None] & mask[:, None, :]
        values[~valid] = float('nan')
        batch.update(kind='matrix', matrix=values)
    return batch


class GuardedBatch(dict):
    def __getitem__(self, key):
        if key not in module.INPUT_ALLOWLIST:
            raise AssertionError('Supervision was accessed: ' + key)
        return super().__getitem__(key)

    def items(self):
        raise AssertionError('Do not enumerate supervision')


class NativeDonorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('No CUDA context')):
            cls.a, cls.b = module.FullTypeDonor('A'), module.FullTypeDonor('B')
        cls.rng_preserved = (python == random.getstate() and np.array_equal(numpy[1], np.random.get_state()[1])
                             and numpy[2:] == np.random.get_state()[2:] and torch.equal(cpu, torch.get_rng_state()))

    @classmethod
    def tearDownClass(cls):
        del cls.a, cls.b
        torch.set_num_threads(cls.old_threads)

    def test_native_weights_seeds_rng_and_identical_routing(self):
        self.assertTrue(self.rng_preserved)
        self.assertEqual(self.a.variant, 'extra_random')
        self.assertEqual(self.b.variant, 'full_pretrained')
        self.assertEqual(self.a.provenance()['components']['VRP'], self.b.provenance()['components']['VRP'])
        for key, value in self.a.routing.state_dict().items():
            self.assertTrue(torch.equal(value, self.b.routing.state_dict()[key]))
        for component, seed in module.RANDOM_SEEDS.items():
            self.assertEqual(self.a.provenance()['components'][component]['random_seed'], seed)
        self.assertEqual(self.a.provenance()['external_pretraining_by_component'], dict(TSP=False, ATSP=False, VRP=True))
        self.assertEqual(self.b.provenance()['external_pretraining_by_component'], dict(TSP=True, ATSP=True, VRP=True))
        state = self.b.provenance()
        state['components']['ATSP']['output_dim'] = 3
        self.assertEqual(self.b.provenance()['components']['ATSP']['output_dim'], 512)

    def test_actual_native_outputs_exact_n_roles_and_zero_padding(self):
        for problem in ('TSP', 'ATSP'):
            batch = native_batch(problem)
            batch['costs'], batch['winner'], batch['ind'] = object(), object(), object()
            hidden = self.b(GuardedBatch(batch))
            self.assertEqual(tuple(hidden.shape), (3, 9, module.WIDTH_BY_PROBLEM[problem]))
            self.assertFalse(hidden.requires_grad)
            self.assertEqual(hidden.dtype, torch.float32)
            self.assertEqual(hidden[~batch['node_mask']].count_nonzero(), 0)
            self.assertTrue(torch.isfinite(hidden).all())
            torch.testing.assert_close(self.b(batch), hidden, atol=0, rtol=0)
            rows = torch.tensor([0, 2])
            positions = torch.nonzero(batch['node_mask'][rows])[:, 1].reshape(2, 7)
            with torch.no_grad(), module.reld._native_precision('cpu'):
                if problem == 'TSP':
                    raw = batch['node'][rows[:, None], positions]
                    expected = self.b.tsp(raw)
                else:
                    raw = batch['matrix'][rows[:, None, None], positions[:, :, None], positions[:, None, :]]
                    identity = torch.eye(256)[:7].unsqueeze(0).expand(2, -1, -1).contiguous()
                    roles = self.b.atsp(torch.zeros(2, 7, 256), identity, raw)
                    expected = torch.cat(roles, -1)
                    self.assertEqual(tuple(expected.shape), (2, 7, 512))
            torch.testing.assert_close(hidden[rows[:, None], positions], expected, atol=0, rtol=0)

    def test_checkpoint_restore_never_opens_external_weights(self):
        forbidden = {str(spec[0]) for spec in module.CHECKPOINTS.values()} | {str(module.reld.DEFAULT_CHECKPOINT)}
        original_open = Path.open

        def guarded(path, *args, **kwargs):
            if str(path) in forbidden:
                raise AssertionError('Restore opened an external pretrained weight')
            return original_open(path, *args, **kwargs)

        for donor in (self.a, self.b):
            with mock.patch.object(torch, 'load', side_effect=AssertionError('External load on restore')), \
                    mock.patch.object(Path, 'open', guarded):
                restored = module.FullTypeDonor(donor.arm, initialize_from_checkpoint=False,
                                                expected_provenance=donor.provenance())
            restored.load_state_dict(donor.state_dict(), strict=True)
            self.assertEqual(restored.provenance(), donor.provenance())
            for problem in ('TSP', 'ATSP'):
                torch.testing.assert_close(restored(native_batch(problem)), donor(native_batch(problem)), rtol=0, atol=0)
        changed = copy.deepcopy(self.b.provenance())
        changed['components']['ATSP']['roles']['output'] = 'average roles'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            module.FullTypeDonor('B', initialize_from_checkpoint=False, expected_provenance=changed)

    def test_frozen_native_eval_and_invalid_identity_inputs(self):
        self.b.train(True)
        self.assertFalse(any(child.training for child in self.b.modules()))
        self.assertFalse(any(p.requires_grad for p in self.b.parameters()))
        for problem in module.PROBLEM_NAMES:
            self.assertIn(self.b.component_for_problem(problem), ('TSP', 'ATSP', 'VRP'))
        batch = native_batch('ATSP', lengths=(257,), padded=257)
        with self.assertRaisesRegex(ValueError, '256'):
            self.b(batch)
        batch = native_batch('TSP')
        batch['problem_id'] = 2
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.b(batch)

    def test_matnet_partition_error_is_observed_not_promised_zero(self):
        batch = native_batch('ATSP', lengths=(7, 7), padded=7)
        together = self.b(batch)
        single = dict(batch)
        for key in ('matrix', 'n', 'node_mask'):
            single[key] = batch[key][:1]
        alone = self.b(single)
        error = float((together[:1] - alone).abs().max())
        self.assertTrue(np.isfinite(error))
        self.assertIn('no bitwise partition invariance', self.b.provenance()['spec']['numerical_warning'])
        print(f'MatNet synthetic N7 CPU FP32 batch2-vs1 maxabs={error:.8g}; observation only')

    def test_matnet_namespace_is_restored_even_on_load_failure(self):
        namespace = 'EasyNCO.neural_solvers.backbones'
        sentinel = types.ModuleType(namespace)
        with mock.patch.dict(sys.modules, {namespace: sentinel}):
            with module._seeded_cpu(3):
                native = module._build_atsp()
            self.assertIs(sys.modules[namespace], sentinel)
            self.assertEqual(native.layers[0].row_encoding_block.normalization_1.forward.__code__.co_filename,
                             str(module.MATNET_BACKBONE))
            with mock.patch.object(module, '_load_file', side_effect=RuntimeError('synthetic failure')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                    module._build_atsp()
            self.assertIs(sys.modules[namespace], sentinel)


if __name__ == '__main__':
    unittest.main()
