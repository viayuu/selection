"""Small synthetic parity checks; no TEST/LIB access and no GPU allocation."""

import copy
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from . import r79_nss_experiment as r


class OriginalNSSParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_requested_recipe_overrides(self):
        for task in r.GPUS:
            original, effective = r.recipe(task)
            expected = copy.deepcopy(original)
            expected["seed"] = 2
            expected["train_params"]["train_batch_size"] = 1024
            self.assertEqual(effective, expected)
            self.assertEqual(original["seed"], 2024)
            self.assertEqual(original["train_params"]["train_batch_size"], 64)

    def test_lazy_augmentation_matches_original(self):
        torch.manual_seed(2)
        for dimensions in (2, 3):
            x = torch.rand(1, 7, dimensions)
            labels = [[0, [1., 2.], [0., 0.], [0., 100.]]]
            original = r.ORIGINAL_DATASET.SelectionDataset([x], labels, data_aug=True)
            cache = dict(points=x.squeeze(0), offsets=torch.tensor([0, 7]),
                         costs=torch.tensor([[1., 2.]], dtype=torch.float64),
                         times=torch.tensor([[0., 0.]], dtype=torch.float64),
                         gaps=torch.tensor([[0., 100.]], dtype=torch.float64), winner=torch.tensor([0]))
            lazy = r.NSSDataset(cache, augment=True)
            self.assertEqual(len(lazy), len(original))
            for fold in range(8):
                torch.testing.assert_close(lazy[fold][0], original[fold][0], rtol=0, atol=0)
            a = r.ORIGINAL_DATASET.collate_fn([lazy[i] for i in range(8)])
            b = r.ORIGINAL_DATASET.collate_fn([original[i] for i in range(8)])
            for i in range(7):
                torch.testing.assert_close(a[i], b[i], rtol=0, atol=0)

    def test_padding_and_micro_accumulation_match_original_model(self):
        for task in r.GPUS:
            torch.manual_seed(2)
            _, config = r.recipe(task)
            p = dict(config["model_params"], problem_type=task,
                     output_dim=config["train_params"]["num_classes"], ns_feature=False)
            original = r.ORIGINAL_MODEL.Selection_model(**p)
            adapted = r.make_model(config)
            adapted.load_state_dict(original.state_dict())
            dims, k = (2 if task == "TSP" else 3), p["output_dim"]
            samples = [[torch.rand(7 + [3, 0, 4, 1, 2][i], dims),
                        [0, list(range(1, k + 1)), [0.] * k, [0.] * k], i]
                       for i in range(5)]
            batch = r.ORIGINAL_DATASET.collate_fn(samples)
            x, costs, scales, mask = r.transfer(batch, "cpu")
            full = original(x, scales, None, mask)
            torch.testing.assert_close(full, adapted(x, scales, None, mask), rtol=0, atol=0)
            criterion = r.ORIGINAL_LOSS.RankingLoss(k, k)
            criterion(full, costs).backward()
            r.accumulate(adapted, criterion, batch, "cpu", 2)
            self.assertEqual(list(original.state_dict()), list(adapted.state_dict()))
            for a, b in zip(original.parameters(), adapted.parameters()):
                self.assertEqual(a.grad is None, b.grad is None)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=2e-6)
            grouped = r.make_model(config)
            grouped.load_state_dict(original.state_dict())
            restored_outputs = torch.empty_like(full)
            group_loss = r.accumulate_grouped(grouped, criterion, batch, "cpu", 2, restored_outputs)
            torch.testing.assert_close(restored_outputs, full.detach(), rtol=3e-5, atol=2e-6)
            torch.testing.assert_close(group_loss, criterion(full.detach(), costs), rtol=3e-5, atol=2e-6)
            for a, b in zip(original.parameters(), grouped.parameters()):
                self.assertEqual(a.grad is None, b.grad is None)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=2e-6)

    def test_validation_gap_rule_and_zero_gap_fallback(self):
        cache = dict(costs=torch.tensor([[10., 12.], [100., 105.]], dtype=torch.float64),
                     gaps=torch.tensor([[0., 20.], [0., 5.]], dtype=torch.float64),
                     winner=torch.tensor([0, 0]), pool=["a", "b"])
        scores = torch.tensor([[0., 1.], [1., 0.]])
        result, _ = r.summarize(scores, cache, "a")
        self.assertEqual(result["metric_name"], "gap")
        self.assertEqual(result["top_1"], 10.)
        cache["gaps"].zero_()
        result, _ = r.summarize(scores, cache, "a")
        self.assertEqual(result["metric_name"], "cost")
        self.assertEqual(result["top_1"], 56.)

    def test_smoke_input_pinning_matches_the_input_path(self):
        gpu = mock.Mock(device=torch.device("cuda:0"))
        gpu.is_pinned.side_effect = AssertionError("Resident CUDA input must not require host pinning")
        self.assertTrue(r.training_input_ready(gpu, "cuda:0", "gpu_resident"))
        self.assertFalse(r.training_input_ready(gpu, "cuda:1", "gpu_resident"))
        self.assertFalse(r.training_input_ready(gpu, "cuda:0", "cpu_pinned"))
        gpu.is_pinned.assert_not_called()
        host = mock.Mock(device=torch.device("cpu"))
        host.is_pinned.return_value = True
        self.assertTrue(r.training_input_ready(host, "cuda:0", "cpu_pinned"))
        self.assertFalse(r.training_input_ready(host, "cuda:0", "gpu_resident"))
        host.is_pinned.return_value = False
        self.assertFalse(r.training_input_ready(host, "cuda:0", "cpu_pinned"))
        self.assertFalse(r.training_input_ready(host, "cuda:0", "unknown"))

    def test_freeze_reuses_existing_val_predictions_for_pure_handoff(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            r.write_json(root / "protocol.json", dict(updates_per_model=27400, original_sources={},
                         tasks={task: dict(train_sbs="a") for task in r.GPUS}))
            for task in r.GPUS:
                run = root / task
                run.mkdir()
                (run / "best.pt").write_bytes(b"selected original checkpoint")
                (run / "best_val_predictions.npz").write_bytes(b"existing VALID predictions")
                r.write_json(run / "training_complete.json", dict(updates=27400,
                             best_sha256=r.sha256(run / "best.pt")))
                r.write_json(run / "best_validation.json", dict(epoch=23, updates=13152, n=3895,
                             metric_name="gap", top_1=0.4, top1=0.43, top1_native_ind=0.44, pool=["a", "b"]))
            with mock.patch.object(r, "infer", side_effect=AssertionError("Do not repeat VALID inference")):
                r.freeze(root)
                handoff = r.read_json(root / "locked_train_val_handoff.json")
                self.assertEqual(handoff["allowed_splits"], ["train", "val"])
                self.assertFalse(handoff["test_lib_results_included"])
                self.assertFalse(handoff["inference_repeated"])
                lock = r.read_json(root / "final_test_lock.json")
                for task in r.GPUS:
                    selected = handoff["records"][task]["checkpoint"]
                    self.assertEqual(selected, lock["checkpoints"][task])
                    self.assertEqual(selected["validation_predictions"]["sha256"],
                                     r.sha256(root / task / "best_val_predictions.npz"))
                before = r.sha256(root / "locked_train_val_handoff.json")
                with mock.patch.object(r, "check_boundary"):
                    r.freeze(root)
                self.assertEqual(before, r.sha256(root / "locked_train_val_handoff.json"))

    def test_resident_batch_augmentation_padding_and_sampler_parity(self):
        for dims in (2, 3):
            torch.manual_seed(2)
            cache = dict(points=torch.rand(17, dims), offsets=torch.tensor([0, 7, 17]),
                         lengths=torch.tensor([7, 10]), winner=torch.tensor([0, 1]),
                         costs=torch.tensor([[1., 2.], [3., 1.]], dtype=torch.float64),
                         gaps=torch.tensor([[0., 100.], [200., 0.]], dtype=torch.float64),
                         times=torch.zeros(2, 2, dtype=torch.float64))
            cpu_dataset = r.NSSDataset(cache, augment=True)
            cpu_loader, cpu_rng = r.make_loader(cpu_dataset, 5, 0, True)
            resident = r.ResidentNSSLoader(cache, "cpu", batch_size=5)
            for _ in range(2):
                for cpu, gpu in zip(cpu_loader, resident):
                    self.assertEqual(cpu[-1], gpu[-1])
                    for i in range(7):
                        torch.testing.assert_close(cpu[i], gpu[i], rtol=0, atol=0)
                torch.testing.assert_close(cpu_rng.get_state(), resident.generator.get_state(), rtol=0, atol=0)

    def test_dual_runtime_requires_release_and_migration_license(self):
        for environment in ({}, {"R79_DP_APPROVED": "1"}, {"R79_R77_RELEASED": "1"}):
            with mock.patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(RuntimeError, "explicit migration license"):
                    r.require_dual_approval()
        with mock.patch.dict(os.environ, {"R79_DP_APPROVED": "1", "R79_R77_RELEASED": "1"}):
            r.require_dual_approval()

    def test_resident_mid_epoch_plan_cursor_resume(self):
        cache = dict(points=torch.rand(17, 2), offsets=torch.tensor([0, 7, 17]),
                     lengths=torch.tensor([7, 10]), winner=torch.tensor([0, 1]),
                     costs=torch.tensor([[1., 2.], [3., 1.]], dtype=torch.float64),
                     gaps=torch.tensor([[0., 100.], [200., 0.]], dtype=torch.float64),
                     times=torch.zeros(2, 2, dtype=torch.float64))
        complete = r.ResidentNSSLoader(cache, "cpu", batch_size=5)
        expected = [batch[-1] for batch in complete]
        partial = r.ResidentNSSLoader(cache, "cpu", batch_size=5)
        iterator = iter(partial)
        prefix = [next(iterator)[-1], next(iterator)[-1]]
        state = copy.deepcopy(partial.state_dict())
        resumed = r.ResidentNSSLoader(cache, "cpu", batch_size=5)
        resumed.generator.set_state(partial.generator.get_state())
        resumed.load_state_dict(state)
        self.assertEqual(prefix + [batch[-1] for batch in resumed], expected)
        torch.testing.assert_close(resumed.generator.get_state(), complete.generator.get_state(), rtol=0, atol=0)
        self.assertEqual([batch[-1] for batch in resumed], [batch[-1] for batch in complete])

    def test_original_rezero_parameter_dispatch_rejects_dp_tensor_replica(self):
        original = r.ORIGINAL_MODEL.Add_And_Normalization_Module(embedding_dim=128, norm="rezero")
        replica = original._replicate_for_data_parallel()
        replica.norm = original.norm + 0
        self.assertIsInstance(original.norm, torch.nn.Parameter)
        self.assertNotIsInstance(replica.norm, torch.nn.Parameter)
        x, residual = torch.zeros(1, 2, 128), torch.ones(1, 2, 128)
        torch.testing.assert_close(original(x, residual), x, rtol=0, atol=0)
        torch.testing.assert_close(replica(x, residual), residual, rtol=0, atol=0)

    def test_cpu_wrapper_pair_mean_tail_and_unwrapped_state(self):
        # CPU DataParallel is an identity wrapper, not a claim of two-GPU parity.
        for task in r.GPUS:
            torch.manual_seed(2)
            _, config = r.recipe(task)
            reference, candidate = r.make_model(config), r.make_model(config)
            candidate.load_state_dict(reference.state_dict(), strict=True)
            wrapped = torch.nn.DataParallel(candidate)
            k = config["train_params"]["num_classes"]
            dims = 2 if task == "TSP" else 3
            samples = [[torch.rand(7 + [3, 0, 4, 1, 2][i], dims),
                        [0, torch.arange(1, k + 1).roll(i).tolist(), [0.] * k, [0.] * k], i]
                       for i in range(5)]
            batch = r.ORIGINAL_DATASET.collate_fn(samples)
            criterion = r.ORIGINAL_LOSS.RankingLoss(k, k)
            a = torch.optim.Adam(reference.parameters(), lr=1e-4, weight_decay=1e-6)
            b = torch.optim.Adam(wrapped.parameters(), lr=1e-4, weight_decay=1e-6)
            ref_loss, _, _, _ = r.train_update(reference, a, criterion, batch, "cpu", 2, True)
            pair_loss, _, _, _ = r.train_update(wrapped, b, criterion, batch, "cpu", 4, True)
            self.assertAlmostEqual(ref_loss, pair_loss, places=5)
            for p, q in zip(reference.parameters(), candidate.parameters()):
                self.assertEqual(p.grad is None, q.grad is None)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=2e-6)
            self.assertTrue(all(state["step"].item() == 1 for state in b.state.values()))
            state = r.unwrap_model(wrapped).state_dict()
            self.assertEqual(list(state), list(reference.state_dict()))
            restored = r.make_model(config)
            restored.load_state_dict(state, strict=True)
            with mock.patch.object(wrapped, "device_ids", [0, 1]):
                self.assertEqual(r.forward_micro_batch(wrapped, 64), 128)


if __name__ == "__main__":
    unittest.main()
