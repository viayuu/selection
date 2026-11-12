"""Small CPU checks for the new candidate-set experiment."""

import unittest

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS
from . import r125_experiment as runner
from .pairwise_objective import comparison_targets, inputs_only
from .r76_experiment import RDropObjective
from .r77_data import parameter_hash
from .test_r76 import fixture_batch
from .test_r80 import fresh_fixture


class R125ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_group_cli_and_unchanged_budget(self):
        args = runner.parser().parse_args(['train'])
        self.assertEqual((args.experiment, args.arm), ('R125', 'code_seed2'))
        self.assertEqual((runner.BATCH_SIZE, runner.MAX_UPDATES, runner.WARMUP_UPDATES), (1024, 57600, 4320))
        self.assertEqual(runner.micro_caps()['TSP'], 256)
        self.assertEqual(runner.micro_caps()['ATSP'], 1024)

    def test_real_source_branch_is_fresh_and_receives_loss_gradient(self):
        initial, geometry = fresh_fixture()
        before = parameter_hash(initial['trainable'])
        model, params, receipt = runner.initialize_shared(initial, geometry, runner.ARM)
        self.assertEqual(params['solver_representation'], 'code')
        self.assertFalse(receipt['historical_checkpoint_loaded'])
        self.assertEqual(before, parameter_hash(initial['trainable']))
        self.assertFalse(model.solver_encoder.code_vectors.requires_grad)
        batch, raw_costs = fixture_batch('TSP', 3)
        keep = [i for i, solver in enumerate(batch['pool_ids'].tolist())
                if GLOBAL_SOLVERS[solver] not in ('BQ', 'LEHD')]
        batch['pool_ids'] = batch['pool_ids'][keep]
        batch['solver_mask'] = torch.ones(3, len(keep), dtype=torch.bool)
        if 'mask' in batch:
            batch['mask'][:, [GLOBAL_SOLVERS.index('BQ'), GLOBAL_SOLVERS.index('LEHD')]] = 0
        raw_costs = raw_costs[:, keep]
        batch.update(comparison_targets(raw_costs))
        batch['costs'] = torch.from_numpy(raw_costs.astype(np.float32))
        batch['ind'] = torch.from_numpy(raw_costs.argmin(1))
        model.train()
        loss, parts = RDropObjective(model)(batch, .1)
        loss.backward()
        self.assertTrue(torch.isfinite(parts).all())
        gradient = model.solver_encoder.role_proj[0][1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)
        model.eval()
        with torch.no_grad():
            output = model(inputs_only(batch))['logits']
            batch['costs'].fill_(999.)
            batch['ind'].zero_()
            changed = model(inputs_only(batch))['logits']
        self.assertEqual(output.shape, (3, len(keep)))
        torch.testing.assert_close(output, changed, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
