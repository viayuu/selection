import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from .V4Model import ConditionEncoder
from .test_solver_features import sample_batch, small_params
from .tensor_loader import TensorBatchLoader
from .train import build_winner_weights, make_loader, selector_loss, winner_first_rank, winner_pair_loss
from .training_monitor import capture_rng, restore_rng
from . import train


class TrainingControlTests(unittest.TestCase):
    def test_checkpoint_trackers_and_rewind_guard(self):
        class TinySelector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.score = torch.nn.Parameter(torch.zeros(2))

            def forward(self, batch):
                return {"logits": self.score.expand(len(batch["costs"]), -1)}

        batch = {"costs": torch.tensor([[1., 2.], [1., 2.], [2., 1.]]), "ind": torch.tensor([0, 0, 1])}
        macro = {"macro_top1": .6, "macro_top2": .9, "macro_top3": 1.,
                 "macro_vs_sbs_pct": -.2, "macro_vbs_gap_closed_pct": 20.}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(train, "build_model", side_effect=lambda args: (TinySelector(), {})), \
                patch.object(train, "make_loader", return_value=[batch]), \
                patch.object(train, "build_sbs_indices", return_value={"TSP": 0}), \
                patch.object(train, "build_winner_weights", return_value={"TSP": torch.ones(2)}) as weights, \
                patch.object(train, "evaluate", return_value=({}, macro)), \
                patch.object(train, "configure_torch"), patch.object(train, "plot_history"), \
                patch("torch.cuda.is_available", return_value=False):
            argv = ["train", "--save-dir", directory, "--architecture", "dual_stream",
                    "--loss-mode", "ce", "--epochs", "2", "--device", "cpu",
                    "--problems", "TSP", "--skip-test", "--native-winner", "--winner-balance"]
            with patch("sys.argv", argv):
                train.main()
            best = torch.load(Path(directory) / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(best["best_top1"], .6)
            self.assertEqual(best["best_top3_safe"], 1.)
            self.assertEqual(best["best_cost"], -.2)
            self.assertEqual(best["winner_weight_policy"], "native")
            weights.assert_called_with(["TSP"], native_winner=True)
            historical = torch.load(Path(directory) / "last.pt", map_location="cpu", weights_only=False)
            historical.pop("winner_weight_policy")
            torch.save(historical, Path(directory) / "historical_last.pt")
            with patch("sys.argv", argv + ["--epochs", "3", "--resume", str(Path(directory) / "historical_last.pt")]):
                train.main()
            weights.assert_called_with(["TSP"], native_winner=False)
            with patch("sys.argv", argv + ["--resume", str(Path(directory) / "best.pt")]), \
                    self.assertRaisesRegex(ValueError, "History is newer"):
                train.main()

    def test_native_class_frequencies_follow_native_labels(self):
        dataset = SimpleNamespace(K_p=2, base_N=3, labels={
            "0": {"cost": [1.00000001, 1.0], "ind": 1},
            "1": {"cost": [1.1, 1.0], "ind": 1},
            "2": {"cost": [.9, 1.0], "ind": 0},
        })
        with patch("code.V4.train.UnifiedProblemDataset", return_value=dataset):
            old = build_winner_weights(["TSP"])["TSP"]
            native = build_winner_weights(["TSP"], native_winner=True)["TSP"]
        torch.testing.assert_close(native, torch.sqrt(torch.tensor([1.5, .75])))
        torch.testing.assert_close(old, native.flip(0))

    def test_disable_unverified_distribution(self):
        params = small_params()
        a = ConditionEncoder(**params).eval()
        b = ConditionEncoder(**dict(params, ignore_coord_dist=True)).eval()
        b.load_state_dict(a.state_dict())
        batch = sample_batch("VRPTW")
        changed = dict(batch, coord_dist=torch.zeros(2, dtype=torch.long))
        stats = torch.randn(2, 12)
        torch.testing.assert_close(b(batch, stats, 0)[0], b(changed, stats, 0)[0])
        self.assertFalse(torch.allclose(a(batch, stats, 0)[0], a(changed, stats, 0)[0]))

    def test_native_winner_and_sequential_rank(self):
        costs = torch.tensor([[1., 1., 2.], [4., 2., 2.]])
        winner = torch.tensor([1, 2])
        rank = winner_first_rank(costs, winner)
        torch.testing.assert_close(rank[:, 0], winner)
        torch.testing.assert_close(rank.sort(1).values, torch.arange(3).expand(2, -1))
        score = torch.tensor([[1., 2., 0.], [1., 0., 2.]], requires_grad=True)
        args = SimpleNamespace(loss_mode="ce", ce_weight=1.)
        loss, parts = selector_loss({"logits": score}, costs, None, args, winner=winner)
        torch.testing.assert_close(loss, F.cross_entropy(score, winner))
        self.assertEqual(set(parts), {"ce"})

    def test_pair_covers_runnerup_and_uses_margin(self):
        score = torch.zeros(1, 4, requires_grad=True)
        costs = torch.tensor([[10., 10.01, 10.1, 11.]])
        loss = winner_pair_loss(score, costs, torch.tensor([0]), .01)
        loss.backward()
        self.assertLess(score.grad[0, 0], 0)
        self.assertTrue((score.grad[0, 1:] > 0).all())
        self.assertLess(score.grad[0, 1], score.grad[0, 2])
        tied = winner_pair_loss(score, torch.ones_like(costs), torch.tensor([1]))
        self.assertEqual(tied.item(), 0)

    def test_train_evaluation_does_not_drop_tail(self):
        batch = dict(n=torch.ones(5, dtype=torch.long), costs=torch.rand(5, 2), pool_ids=torch.arange(2))
        loader = TensorBatchLoader(batch, 2, False, True)
        with patch("code.V4.train.make_tensor_loader", return_value=loader):
            actual = make_loader("TSP", "train", 2, 0, shuffle=False, cache_device="cpu")
        self.assertEqual(sum(len(b["costs"]) for b in actual), 5)

    def test_rng_roundtrip(self):
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(state)
        self.assertEqual(expected[0], random.random())
        self.assertEqual(expected[1], np.random.rand())
        torch.testing.assert_close(expected[2], torch.rand(3))

    def test_scheduler_resume_matches(self):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=2e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=1, factor=.5)
        for value in [.4, .3]:
            sched.step(value)
        opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.)
        sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="max", patience=1, factor=.5)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
        sched2.load_state_dict(copy.deepcopy(sched.state_dict()))
        sched.step(.3); sched2.step(.3)
        self.assertEqual(opt.param_groups[0]["lr"], opt2.param_groups[0]["lr"])
        self.assertEqual(opt.param_groups[0]["lr"], 1e-4)


if __name__ == "__main__":
    unittest.main()
