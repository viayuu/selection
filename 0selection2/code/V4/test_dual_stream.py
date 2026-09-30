import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS, POOLS
from .V4Model import make_selector
from .dual_stream import JointEncoderLayer
from .evaluate import load_model, evaluate_problem, macro_row, write_report
from .test_solver_features import sample_batch, small_params
from .train import selector_loss, evaluate, print_eval


def params():
    p = small_params()
    p.update(architecture="dual_stream", joint_layer_num=2, solver_feature_weight=1.0)
    return p


class DualStreamTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = make_selector(params()).eval()

    def test_all_problems_without_labels(self):
        for problem in PROBLEMS:
            batch = sample_batch(problem)
            del batch["costs"]
            out = self.model(batch)
            self.assertEqual(set(out), {"logits"})
            self.assertEqual(out["logits"].shape, (2, len(POOLS[problem])))
            self.assertTrue(torch.isfinite(out["logits"]).all())
            batch["costs"] = torch.full((2, len(POOLS[problem])), float("nan"))
            torch.testing.assert_close(self.model(batch)["logits"], out["logits"])

    def test_synchronous_updates_and_independent_parameters(self):
        layer = JointEncoderLayer(**params()).eval()
        x, s = torch.randn(2, 5, 16), torch.randn(2, 3, 16)
        xm, sm = torch.ones(2, 5, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool)
        a, b = layer(x, s, xm, sm)
        torch.testing.assert_close(a, layer.node_update(x, s, xm, sm))
        torch.testing.assert_close(b, layer.solver_update(s, x, sm, xm))
        self.assertFalse(torch.allclose(b, layer.solver_update(s, a, sm, xm)))
        a_ids = {id(p) for p in layer.node_update.parameters()}
        self.assertFalse(a_ids & {id(p) for p in layer.solver_update.parameters()})
        layers = self.model.joint_layers
        self.assertFalse({id(p) for p in layers[0].parameters()} & {id(p) for p in layers[1].parameters()})
        self.assertFalse({id(p) for p in self.model.decoder.parameters()} & {id(p) for p in layers.parameters()})

    def test_candidate_permutation_and_padding(self):
        batch = sample_batch("CVRP")
        logits = self.model(batch)["logits"]
        order = torch.randperm(len(batch["pool_ids"]))
        perm = dict(batch, pool_ids=batch["pool_ids"][order])
        torch.testing.assert_close(self.model(perm)["logits"], logits[:, order], atol=2e-6, rtol=2e-5)
        ids = torch.cat([batch["pool_ids"], torch.tensor([-1, -1])])
        mask = torch.ones(2, len(ids), dtype=torch.bool)
        mask[:, -2:] = False
        padded = self.model(dict(batch, pool_ids=ids, solver_mask=mask))["logits"]
        torch.testing.assert_close(padded[:, :-2], logits, atol=2e-6, rtol=2e-5)
        self.assertTrue((padded.softmax(-1)[:, -2:] == 0).all())
        for size in (1, 3):
            out = self.model(dict(batch, pool_ids=batch["pool_ids"][:size]))
            self.assertEqual(out["logits"].shape, (2, size))
        with self.assertRaises(ValueError):
            self.model(dict(batch, solver_mask=torch.zeros_like(logits, dtype=torch.bool)))

    def test_node_padding_and_no_cross_batch_attention(self):
        for problem in ("CVRP", "ATSP"):
            batch = sample_batch(problem)
            logits = self.model(batch)["logits"]
            padded = copy.deepcopy(batch)
            padded["node"] = F.pad(batch["node"], (0, 0, 0, 3), value=123.0)
            padded["matrix"] = F.pad(batch["matrix"], (0, 3, 0, 3), value=123.0)
            padded["node_mask"] = F.pad(batch["node_mask"], (0, 3), value=False)
            torch.testing.assert_close(self.model(padded)["logits"], logits, atol=2e-6, rtol=2e-5)
            changed = copy.deepcopy(batch)
            changed["node"][1] = torch.rand_like(changed["node"][1]) * 20
            changed["matrix"][1] = torch.rand_like(changed["matrix"][1]) * 20
            torch.testing.assert_close(self.model(changed)["logits"][0], logits[0], atol=1e-6, rtol=1e-5)

    def test_all_stages_receive_gradients(self):
        for problem in ("TSP", "ATSP"):
            self.model.zero_grad(set_to_none=True)
            out = self.model(sample_batch(problem))
            F.cross_entropy(out["logits"], torch.tensor([0, 1])).backward()
            projection = self.model.instance_encoder.coord_proj if problem == "TSP" else self.model.instance_encoder.matrix_proj
            modules = [projection, self.model.solver_encoder.feature_mlp, self.model.solver_encoder.solver_emb,
                       *self.model.joint_layers, self.model.pool, self.model.decoder, self.model.score_head]
            for module in modules:
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
            self.assertIsNone(self.model.solver_encoder.solver_features.grad)
            self.assertFalse(hasattr(self.model.instance_encoder, "summary_proj"))

    def test_final_nodes_and_instance_conditioned_solvers_are_used(self):
        batch = sample_batch("CVRP")
        values = []
        handle = self.model.joint_layers[-1].register_forward_hook(lambda m, inp, out: values.append(out))
        original = self.model(batch)["logits"]
        handle.remove()
        nodes, solvers = values[0]
        self.assertFalse(torch.allclose(solvers[0], solvers[1]))
        for index in (0, 1):
            def modify(module, inp, out):
                changed = list(out)
                changed[index] = changed[index] + torch.randn_like(changed[index])
                return tuple(changed)
            handle = self.model.joint_layers[-1].register_forward_hook(modify)
            self.assertFalse(torch.allclose(original, self.model(batch)["logits"]))
            handle.remove()

    def test_checkpoint_and_training_evaluation_interfaces(self):
        batch = sample_batch("ATSP")
        batch["costs"] = torch.rand(2, len(batch["pool_ids"])) + 1
        args = Namespace(architecture="dual_stream", top_focused_pair=True, topk_ce_rank_weights="1,0.4,0.2",
                         topk_ce_mode="sequential", ce_weight=.35, pair_weight=.30, topk_ce_weight=.08, risk_weight=.02)
        loss, parts = selector_loss(self.model(batch), batch["costs"], None, args, sbs_idx=0)
        self.assertEqual(set(parts), {"ce", "pair", "topk_ce", "risk"})
        self.assertTrue(torch.isfinite(loss))
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "best.pt"
            torch.save({"model": self.model.state_dict(), "args": {"model_params": params()}}, ckpt)
            loaded, _ = load_model(ckpt, "cpu")
            torch.testing.assert_close(loaded(batch)["logits"], self.model(batch)["logits"])
            with patch("code.V4.train.make_loader", return_value=[batch]), patch("code.V4.train.get_split_sbs_vbs", return_value=(1.5, 1.0)):
                per, macro = evaluate(loaded, ["ATSP"], "val", 2, 0, "cpu")
            self.assertNotIn("macro_top1_gap", macro)
            print_eval("unit test", per, macro)
            loaded.eval()
            with patch("code.V4.evaluate.make_loader", return_value=(None, [batch])):
                r = evaluate_problem(loaded, "ATSP", "test", 2, 0, "cpu")
            rows = {"ATSP": r}
            payload = dict(ckpt=str(ckpt), split="test", per_problem=rows, all=macro_row(rows))
            write_report(payload, Path(d))
            self.assertNotIn("Component Diagnostics", (Path(d) / "test_summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
