import copy
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .multitask_probe import (TaskRandomStreams, Tee, batch_schedule, evaluate_tsp,
                             gather_batch, restore_continuation, state_hash,
                             task_seed, update_batch)
from .tensor_loader import TensorBatchLoader
from .training_monitor import capture_rng, restore_rng


class DropoutClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)
        self.drop = nn.Dropout(.5)
        self.outputs = []

    def forward(self, batch):
        logits = self.layer(self.drop(batch["x"]))
        self.outputs.append(logits.detach().clone())
        return {"logits": logits}


class RetryScaler:
    def __init__(self):
        self.value = 2.
        self.attempts = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        if self.attempts:
            optimizer.step()

    def update(self):
        if not self.attempts:
            self.value = 1.
        self.attempts += 1


class ProbeTests(unittest.TestCase):
    def test_log_proxy_supports_terminal_inspection(self):
        stream, log = io.StringIO(), io.StringIO()
        tee = Tee(stream, log)
        self.assertFalse(tee.isatty())
        tee.write("progress\n")
        tee.flush()
        self.assertEqual(stream.getvalue(), log.getvalue())

    def test_sampling_is_paired_and_independent(self):
        a = batch_schedule(23, 5, 12, task_seed(2, "TSP"))
        torch.rand(50)
        batch_schedule(23, 5, 12, task_seed(2, "CVRP"))
        b = batch_schedule(23, 5, 12, task_seed(2, "TSP"))
        torch.testing.assert_close(a, b)
        self.assertEqual(len(a[:4].flatten().unique()), 20)
        self.assertFalse(torch.equal(a, batch_schedule(23, 5, 12, task_seed(3, "TSP"))))

    def test_dropout_stream_is_independent_of_other_tasks(self):
        a = TaskRandomStreams(2, ["TSP", "CVRP"])
        b = TaskRandomStreams(2, ["TSP"])
        for _ in range(3):
            with a.activate("TSP"):
                left = nn.functional.dropout(torch.ones(8, 4), .5)
            with a.activate("CVRP"):
                torch.rand(100)
            with b.activate("TSP"):
                right = nn.functional.dropout(torch.ones(8, 4), .5)
            torch.testing.assert_close(left, right)
            self.assertEqual(state_hash(a.states["TSP"]), state_hash(b.states["TSP"]))

    def test_retry_reuses_dropout_and_counts_one_update(self):
        model = DropoutClassifier().train()
        initial_model = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        initial_rng = capture_rng()
        batch = dict(x=torch.ones(6, 4), costs=torch.tensor([[1., 2.]]).repeat(6, 1), ind=torch.zeros(6, dtype=torch.long))
        args = SimpleNamespace(loss_mode="winner_cost", ce_weight=.35, pair_weight=.10, risk_weight=.02, cost_scale=.01)
        stats = update_batch(model, optimizer, RetryScaler(), batch, args)
        final_rng = capture_rng()
        self.assertEqual(stats["skipped"], 1)
        torch.testing.assert_close(model.outputs[0], model.outputs[1])
        self.assertTrue(all(int(s["step"]) == 1 for s in optimizer.state.values()))
        restore_rng(initial_rng)
        reference = DropoutClassifier().train()
        reference.load_state_dict(initial_model)
        restore_rng(initial_rng)
        reference(batch)
        self.assertEqual(state_hash(capture_rng()), state_hash(final_rng))

    def test_restore_preserves_moments_and_does_not_alias_source(self):
        source = nn.Linear(4, 2)
        original = torch.optim.AdamW(source.parameters(), lr=1e-4, weight_decay=1e-4)
        source(torch.ones(2, 4)).sum().backward()
        original.step()
        checkpoint = dict(model=copy.deepcopy(source.state_dict()), optimizer=copy.deepcopy(original.state_dict()),
                          args={"model_params": {}}, scaler={})
        before = state_hash(checkpoint["optimizer"])
        with patch("code.V4.multitask_probe.make_selector", side_effect=lambda p: nn.Linear(4, 2)):
            a, opt_a, _ = restore_continuation(checkpoint, "cpu", 2e-5)
            b, opt_b, _ = restore_continuation(checkpoint, "cpu", 2e-5)
        self.assertEqual(state_hash(opt_a.state_dict()), state_hash(opt_b.state_dict()))
        self.assertEqual(opt_a.param_groups[0]["lr"], 2e-5)
        for left, right in zip(opt_a.state.values(), original.state.values()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(left[key], right[key])
        a(torch.ones(2, 4)).sum().backward()
        opt_a.step()
        self.assertEqual(before, state_hash(checkpoint["optimizer"]))
        self.assertTrue(all(int(s["step"]) == 1 for s in opt_b.state.values()))
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))

    def test_gather_retains_pool_and_crops_padding(self):
        data = dict(node=torch.rand(4, 7, 2), node_mask=torch.ones(4, 7, dtype=torch.bool),
                    matrix=torch.rand(4, 7, 7), costs=torch.rand(4, 2),
                    n=torch.tensor([3, 7, 4, 6]), pool_ids=torch.tensor([5, 9]), kind="coord")
        loader = TensorBatchLoader(data, 2, False, False)
        batch = gather_batch(loader, torch.tensor([2, 0]))
        self.assertEqual(batch["node"].shape, (2, 4, 2))
        self.assertEqual(batch["matrix"].shape, (2, 4, 4))
        torch.testing.assert_close(batch["costs"], data["costs"][[2, 0]])
        torch.testing.assert_close(batch["pool_ids"], data["pool_ids"])

    def test_complete_evaluation_uses_native_winner_and_eval_mode(self):
        data = dict(x=torch.ones(5, 4), costs=torch.tensor([[1., 1.]]).repeat(5, 1),
                    ind=torch.ones(5, dtype=torch.long), n=torch.ones(5, dtype=torch.long), pool_ids=torch.tensor([0, 1]))
        model = DropoutClassifier().train()
        with torch.no_grad():
            model.layer.weight.zero_()
            model.layer.bias.copy_(torch.tensor([0., 1.]))
        loader = TensorBatchLoader(data, 2, False, True)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.npz"
            result = evaluate_tsp(model, loader, path)
            self.assertEqual(result["n"], 5)
            self.assertEqual(result["top1"], 1.)
            self.assertFalse(model.training)
            with np.load(path) as prediction:
                np.testing.assert_array_equal(prediction["indices"], np.arange(5))
                self.assertEqual(prediction["logits"].shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
