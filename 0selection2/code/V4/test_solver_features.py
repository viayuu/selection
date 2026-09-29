import copy
import tempfile
import unittest
from pathlib import Path

import torch

from code.unified_selector.registry import (
    GLOBAL_SOLVERS, POOLS, PROBLEMS, S2I, constraint_bits, problem_descriptor,
)

from .V4Model import ProblemToSolverSelector, SolverMemoryEncoder, get_default_model_params
from .evaluate import load_model
from .solver_features import build_solver_features, get_solver_feature_spec


def synthetic_spec():
    # Test fixtures are not factual solver classifications.
    active = {name: [i % 34] for i, name in enumerate(GLOBAL_SOLVERS)}
    active["DIFUSCO500"] = active["DIFUSCO"][:]
    return {
        "feature_dim": 34, "index_base": 0,
        "feature_order": [f"fixture_{i}" for i in range(34)],
        "solver_names": list(GLOBAL_SOLVERS),
        "solver_active_indices": dict(reversed(list(active.items()))),
    }


def small_params():
    params = get_default_model_params()
    params.update(
        embedding_dim=16, head_num=2, qkv_dim=8, ff_hidden_dim=32,
        head_hidden_dim=32, encoder_layer_num=1, solver_set_layer_num=1,
        query_num=2, dropout=0.0, solver_feature_spec=synthetic_spec(),
        support_branch=True, support_generators=4, support_g0_as_main=True,
        support_feature_to_token=False,
    )
    return params


def sample_batch(problem):
    pool = torch.tensor([S2I[name] for name in POOLS[problem]])
    return {
        "problem_id": PROBLEMS.index(problem), "pool_ids": pool,
        "kind": "matrix" if problem == "ATSP" else "coord",
        "node": torch.rand(2, 5, 8), "matrix": torch.rand(2, 5, 5),
        "node_mask": torch.ones(2, 5, dtype=torch.bool),
        "n": torch.full((2,), 5), "coord_dist": torch.full((2,), 4),
        "cbits": torch.tensor([constraint_bits(problem)] * 2, dtype=torch.float32),
        "problem_desc": torch.tensor([problem_descriptor(problem)] * 2),
        "costs": torch.ones(2, len(pool)),
    }


class SolverFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_table_alignment_and_unmarked_row(self):
        spec = synthetic_spec()
        spec["solver_active_indices"]["BQ"] = []
        features = build_solver_features(spec)
        self.assertEqual(tuple(features.shape), (len(GLOBAL_SOLVERS), 34))
        for i, name in enumerate(GLOBAL_SOLVERS):
            self.assertEqual(features[i].nonzero().flatten().tolist(), spec["solver_active_indices"][name])
        torch.testing.assert_close(features[S2I["DIFUSCO"]], features[S2I["DIFUSCO500"]])

    def test_actual_table_covers_current_solvers(self):
        spec = get_solver_feature_spec()
        features = build_solver_features(spec)
        self.assertEqual(set(spec["solver_active_indices"]), set(GLOBAL_SOLVERS))
        self.assertTrue(torch.all((features == 0) | (features == 1)))
        for first, second in (("DIFUSCO", "DIFUSCO500"), ("T2T", "T2T500")):
            self.assertNotEqual(S2I[first], S2I[second])
            torch.testing.assert_close(features[S2I[first]], features[S2I[second]])
        self.assertEqual(features[S2I["MoSES_RF"], 18:21].sum().item(), 0)
        self.assertEqual(features[S2I["MoSES_CaDA"], 18:21].sum().item(), 0)
        self.assertEqual(features[S2I["GLOP"], 21:24].sum().item(), 0)

    def test_rejects_invalid_schema_and_missing_solvers(self):
        bad_specs = []
        spec = synthetic_spec()
        spec["feature_order"].pop()
        bad_specs.append(spec)
        spec = synthetic_spec()
        spec["feature_order"][1] = spec["feature_order"][0]
        bad_specs.append(spec)
        spec = synthetic_spec()
        spec["solver_names"].reverse()
        bad_specs.append(spec)
        spec = synthetic_spec()
        del spec["solver_active_indices"]["BQ"]
        bad_specs.append(spec)
        for indices in ([34], [-1], [1, 1], [True]):
            spec = synthetic_spec()
            spec["solver_active_indices"]["BQ"] = indices
            bad_specs.append(spec)
        for spec in bad_specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                build_solver_features(spec)

    def test_fused_solver_used_in_concat_and_residual(self):
        model = SolverMemoryEncoder(**small_params())
        ids = torch.tensor([S2I["DIFUSCO500"], S2I["DIFUSCO"], S2I["BQ"]])
        pid = torch.tensor([0, 3])
        cond = torch.randn(2, 16)
        fused = model.solver_emb(ids) + 0.3 * model.feature_mlp(model.solver_features[ids])
        solver = fused[None].expand(2, -1, -1)
        problem = model.problem_emb(pid)[:, None].expand_as(solver)
        arm = model.arm_emb(pid[:, None] * len(GLOBAL_SOLVERS) + ids[None])
        z = torch.cat([solver, problem, arm, cond[:, None].expand_as(solver)], dim=-1)
        out = model(ids, pid, cond)
        torch.testing.assert_close(out, model.norm(solver + arm + model.ffn(z)))
        self.assertFalse(torch.allclose(fused[0], fused[1]))
        self.assertIn("solver_features", dict(model.named_buffers()))
        self.assertNotIn("solver_features", dict(model.named_parameters()))
        (out * torch.randn_like(out)).sum().backward()
        self.assertIsNone(model.solver_features.grad)
        self.assertGreater(model.feature_mlp[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.solver_emb.weight.grad.abs().sum().item(), 0)

    def test_zero_weight_preserves_id_only_outputs(self):
        params = small_params()
        params["solver_feature_spec"] = None
        legacy = SolverMemoryEncoder(**params)
        params.update(solver_feature_spec=synthetic_spec(), solver_feature_weight=0.0)
        fused = SolverMemoryEncoder(**params)
        state = fused.state_dict()
        state.update(legacy.state_dict())
        fused.load_state_dict(state, strict=True)
        ids, pid, cond = torch.tensor([2, 0]), torch.tensor([0, 3]), torch.randn(2, 16)
        torch.testing.assert_close(fused(ids, pid, cond), legacy(ids, pid, cond), rtol=0, atol=0)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            params = small_params()
            params["solver_feature_spec"] = get_solver_feature_spec()
            model = ProblemToSolverSelector(**params).eval()
            checkpoint = Path(directory) / "model.pt"
            torch.save({"model": model.state_dict(), "args": {"model_params": params}}, checkpoint)
            restored, _ = load_model(checkpoint, "cpu")
            batch = sample_batch("CVRP")
            with torch.no_grad():
                torch.testing.assert_close(model(batch)["logits"], restored(batch)["logits"])
            self.assertIn("solver_encoder.solver_features", restored.state_dict())
            self.assertEqual(restored.params["solver_feature_spec"], params["solver_feature_spec"])

    def test_legacy_parameters_load_strictly(self):
        params = small_params()
        for key in ("solver_feature_spec", "solver_feature_weight", "solver_feature_hidden"):
            del params[key]
        legacy = ProblemToSolverSelector(**params)
        restored = ProblemToSolverSelector(**copy.deepcopy(params))
        restored.load_state_dict(legacy.state_dict(), strict=True)
        self.assertIsNone(restored.solver_encoder.feature_mlp)
        self.assertFalse(any("solver_features" in key for key in restored.state_dict()))

    def test_full_forward_for_all_problem_pools(self):
        params = small_params()
        params["solver_feature_spec"] = get_solver_feature_spec()
        model = ProblemToSolverSelector(**params).eval()
        for problem in PROBLEMS:
            with self.subTest(problem=problem), torch.no_grad():
                out = model(sample_batch(problem))
                self.assertEqual(tuple(out["logits"].shape), (2, len(POOLS[problem])))
                self.assertTrue(torch.isfinite(out["logits"]).all().item())
                torch.testing.assert_close(out["logits"], out["support_logits"][:, 0])


if __name__ == "__main__":
    unittest.main()
