"""Small CPU witnesses, including the newly two-candidate B tasks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r127_experiment as runner
from .pairwise_objective import comparison_targets, inputs_only
from .r76_experiment import read_json
from .r77_data import parameter_hash
from .r83_experiment import initialize_shared as initialize_r83
from .r84_prepare import DEFAULT_DATA_ROOT
from .test_r76 import fixture_batch
from .test_r80 import fresh_fixture


class R127ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_fresh_group_cli_and_unchanged_budget(self):
        args = runner.parser().parse_args(['train'])
        self.assertEqual((args.experiment, args.arm, args.gpu_type), ('R127', 'model_only_seed2', '3090'))
        self.assertEqual((runner.BATCH_SIZE, runner.MAX_UPDATES, runner.WARMUP_UPDATES), (1024, 57600, 4320))
        self.assertEqual(runner.micro_caps()['TSP'], 256)

    def test_exact_r83_fresh_state_and_finite_original_two_candidate_loss(self):
        initial, geometry = fresh_fixture()
        model, params, receipt = runner.initialize_shared(initial, geometry, runner.ARM)
        reference, _, _ = initialize_r83(initial, geometry, bundle=runner.load_bundle(runner.DEFAULT_EMBEDDING))
        self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(reference.state_dict()))
        self.assertFalse(receipt['historical_checkpoint_loaded'])
        self.assertEqual(receipt['removed_candidates'], ['MVMOE'])
        self.assertFalse(model.solver_encoder.role_mask[:, 2:].any())
        batch, costs = fixture_batch('VRPB', 3)
        keep = [i for i, solver in enumerate(batch['pool_ids'].tolist()) if GLOBAL_SOLVERS[solver] != 'MVMOE']
        batch['pool_ids'] = batch['pool_ids'][keep]
        batch['solver_mask'] = torch.ones(3, len(keep), dtype=torch.bool)
        if 'mask' in batch:
            batch['mask'][:, GLOBAL_SOLVERS.index('MVMOE')] = 0
        costs = costs[:, keep]
        batch.update(comparison_targets(costs))
        batch['costs'] = torch.from_numpy(costs.astype(np.float32))
        batch['ind'] = torch.from_numpy(costs.argmin(1))
        self.assertEqual(costs.shape, (3, 2))
        loss, parts = runner.make_objective(model, runner.ARM)(batch, .1)
        loss.backward()
        self.assertTrue(torch.isfinite(parts).all())
        gradient = model.solver_encoder.role_proj[0][1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertEqual(float(model.solver_encoder.solver_emb.weight.grad[GLOBAL_SOLVERS.index('MVMOE')].abs().sum()), 0.)
        model.eval()
        with torch.no_grad():
            first = model(inputs_only(batch))['logits']
            batch['costs'].fill_(999.)
            batch['ind'].zero_()
            second = model(inputs_only(batch))['logits']
        self.assertEqual(first.shape, (3, 2))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_checkpoint_roundtrip_does_not_inherit_other_deleted_candidates(self):
        initial, geometry = fresh_fixture()
        model, params, _ = runner.initialize_shared(initial, geometry, runner.ARM)
        source = read_json(DEFAULT_DATA_ROOT / 'scenario_manifest.json')
        pools = {p: [m for m in source['problems'][p]['methods'] if m != 'MVMOE'] for p in PROBLEMS}
        aliases = source['solver_aliases']
        ids = {p: [GLOBAL_SOLVERS.index(aliases.get(m, m)) for m in pools[p]] for p in PROBLEMS}
        config = dict(experiment='R127', arm=runner.ARM, seed=2, model_factory=runner.MODEL_FACTORY,
            solver_names=GLOBAL_SOLVERS, removed_candidates=['MVMOE'], retained_candidates=['BQ', 'LEHD', 'MoSES_RF'],
            global_batch_size=1024, max_successful_updates=57600, checkpoint_rule=runner.CHECKPOINT_RULE,
            inference_rule=runner.INFERENCE_RULE, source_scope=runner.SOURCE_SCOPE,
            role_policy=runner.ROLE_POLICY, model_params=params, loss=runner.engine.LOSS_RECIPE,
            candidate_pools=pools, candidate_pool_ids=ids,
            embedding_provenance=runner.bundle_provenance(runner.load_bundle(runner.DEFAULT_EMBEDDING)))
        state = dict(args=config, state_hash=runner.engine.json_hash(config), model=model.state_dict())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic.pt'
            torch.save(state, path)
            restored, _ = runner.load_checkpoint(path)
            self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
        config['candidate_pools']['CVRP'].remove('MoSES_RF')
        with self.assertRaises(ValueError):
            runner.validate_config(config)


if __name__ == '__main__':
    unittest.main()
