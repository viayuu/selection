"""Minimal CPU checks for the new R83 candidate-removal experiment."""

import tempfile
import unittest
from pathlib import Path

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r126_experiment as runner
from .pairwise_objective import inputs_only
from .r77_data import parameter_hash
from .r83_experiment import initialize_shared as initialize_r83
from .r84_prepare import DEFAULT_DATA_ROOT
from .r76_experiment import read_json
from .test_r76 import fixture_batch
from .test_r80 import fresh_fixture


class R126ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_keeps_full_budget_and_single_fresh_group(self):
        args = runner.parser().parse_args(['train'])
        self.assertEqual((args.experiment, args.arm), ('R126', 'model_only_seed2'))
        self.assertEqual((runner.BATCH_SIZE, runner.MAX_UPDATES, runner.WARMUP_UPDATES), (1024, 57600, 4320))
        self.assertEqual(runner.micro_caps()['TSP'], 256)

    def test_exact_actual_r83_initialization_and_active_model_only_views(self):
        initial, geometry = fresh_fixture()
        bundle = runner.load_bundle(runner.DEFAULT_EMBEDDING)
        model, params, receipt = runner.initialize_shared(initial, geometry, runner.ARM)
        reference, _, _ = initialize_r83(initial, geometry, bundle=bundle)
        self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(reference.state_dict()))
        self.assertEqual(params['solver_representation'], 'code')
        self.assertFalse(receipt['historical_checkpoint_loaded'])
        mask = model.solver_encoder.role_mask
        self.assertFalse(mask[:, 2:].any())
        self.assertTrue(mask[GLOBAL_SOLVERS.index('BQ'), :2].all())
        self.assertTrue(mask[GLOBAL_SOLVERS.index('LEHD'), :2].all())
        self.assertFalse(mask[GLOBAL_SOLVERS.index('GLOP')].any())
        batch, _ = fixture_batch('TSP', 3)
        model.train()
        loss, parts = runner.make_objective(model, runner.ARM)(batch, .1)
        loss.backward()
        gradient = model.solver_encoder.role_proj[0][1].weight.grad
        self.assertTrue(torch.isfinite(parts).all())
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)
        model.eval()
        with torch.no_grad():
            first = model(inputs_only(batch))['logits']
            batch['costs'].fill_(999.)
            batch['ind'].zero_()
            second = model(inputs_only(batch))['logits']
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_checkpoint_roundtrip_retains_bq_lehd_and_rejects_removed_method(self):
        initial, geometry = fresh_fixture()
        model, params, _ = runner.initialize_shared(initial, geometry, runner.ARM)
        source = read_json(DEFAULT_DATA_ROOT / 'scenario_manifest.json')
        pools = {p: [m for m in source['problems'][p]['methods'] if m != 'MoSES_RF'] for p in PROBLEMS}
        aliases = source['solver_aliases']
        ids = {p: [GLOBAL_SOLVERS.index(aliases.get(m, m)) for m in pools[p]] for p in PROBLEMS}
        config = dict(experiment='R126', arm=runner.ARM, seed=2, model_factory=runner.MODEL_FACTORY,
            solver_names=GLOBAL_SOLVERS, removed_candidates=['MoSES_RF'], retained_candidates=['BQ', 'LEHD'],
            global_batch_size=1024, max_successful_updates=57600, checkpoint_rule=runner.CHECKPOINT_RULE,
            inference_rule=runner.INFERENCE_RULE, source_scope=runner.SOURCE_SCOPE,
            role_policy=dict(encoder=True, decision=True, inference=False, config=False), model_params=params,
            loss=runner.engine.LOSS_RECIPE, candidate_pools=pools, candidate_pool_ids=ids,
            embedding_provenance=runner.bundle_provenance(runner.load_bundle(runner.DEFAULT_EMBEDDING)))
        state = dict(args=config, state_hash=runner.engine.json_hash(config), model=model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.pt'
            torch.save(state, path)
            restored, _ = runner.load_checkpoint(path)
            self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
        config['candidate_pools']['CVRP'].append('MoSES_RF')
        with self.assertRaises(ValueError):
            runner.validate_config(config)


if __name__ == '__main__':
    unittest.main()
