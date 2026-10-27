"""Focused synthetic CPU checks; never import or execute a native solver."""

from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn

from .r118_td_trace import capture_td


@dataclass
class DummyResult:
    costs: list
    tours: list
    audit: dict


class DummyDecoder(nn.Module):
    def __init__(self, tsp, wrap=False):
        super().__init__()
        self.tsp, self.wrap = tsp, wrap

    def forward(self, env):
        b, p, n = env.ninf_mask.shape
        if self.tsp and env.t == 0:
            return torch.arange(p)[None].expand(b, p), torch.ones(b, p)
        probs = torch.softmax(torch.arange(n, dtype=torch.float32) + env.ninf_mask, -1)
        if self.tsp:
            probs = probs.masked_fill(~torch.isfinite(env.ninf_mask), -2.)
            return None, probs
        return ([probs.unsqueeze(0).unsqueeze(-1)],) if self.wrap else probs


class DummyEncoder(nn.Module):
    def forward(self, row, col, matrix):
        return row, col


class DummyEnv:
    def reset(self, b, p, n):
        self.env_batch_size, self.pomo_size, self.problem_size = b, p, n
        self.t = 0
        self.current_node = None
        self.selected_count = torch.zeros(b, p, dtype=torch.long)
        self.selected_node_list = torch.empty(b, p, 0, dtype=torch.long)
        self.ninf_mask = torch.zeros(b, p, n)

    def state(self):
        b, p, n = self.ninf_mask.shape
        return TensorDict(dict(
            ninf_mask=self.ninf_mask,
            next=TensorDict(dict(ninf_mask=self.ninf_mask), batch_size=[b, p]),
            done=self.selected_count == n,
            reward=(torch.arange(b * p).reshape(b, p).float() - 10
                    if self.t == n else -torch.ones(b, p)),
        ), batch_size=[b, p])

    def step(self, td):
        action = td["action"]
        self.actions.append(action.clone())
        self.received.append(td)
        self.selected_count += 1
        self.t += 1
        self.current_node = action
        self.selected_node_list = torch.cat((self.selected_node_list, action[..., None]), -1)
        self.ninf_mask.scatter_(-1, action[..., None], -float("inf"))
        output = self.state()
        self.outputs.append(output)
        return output


class DummyBackend:
    def __init__(self, tsp=True, aug=None, wrap=False, roles=True, fault=None):
        self.problem, self.method = ("TSP", "LEHD") if tsp else ("ATSP", "MATNET")
        self.env = DummyEnv()
        self.env.aug_factor = aug if aug is not None else (8 if tsp else 128)
        self.model = SimpleNamespace(decoder=DummyDecoder(tsp, wrap), first_mode="random")
        if roles:
            self.model.encoder = DummyEncoder()
        self.fault, self.calls = fault, 0

    def run(self, items):
        self.calls += 1
        n, b = items[0].shape[1], self.env.aug_factor
        p = 1 if self.problem == "TSP" else n
        env = self.env
        env.reset(b, p, n)
        env.actions, env.received, env.outputs = [], [], []
        if self.problem == "ATSP" and hasattr(self.model, "encoder"):
            row = torch.zeros(b, n, n + 1)
            role_ids = (torch.arange(n)[None] + torch.arange(b)[:, None] + 1) % n
            col = torch.zeros_like(row).scatter_(-1, role_ids[..., None], 1.)
            self.roles = (row.clone(), col.clone())
            self.model.encoder(row, col, items[0].expand(b, -1, -1))
        while env.t < n:
            if env.t == 1 and self.fault == "run":
                raise RuntimeError("dummy failure")
            td = env.state()
            if self.problem == "ATSP" and env.t == 0:
                action = torch.arange(p)[None].expand(b, p)
            elif self.fault == "missing" and env.t == 1:
                action = torch.isfinite(env.ninf_mask).long().argmax(-1)
            else:
                output = self.model.decoder(env)
                if self.fault == "double" and env.t == 1:
                    self.model.decoder(env)
                if self.problem == "TSP":
                    action, probs = output
                    action = probs.argmax(-1) if action is None else action
                else:
                    probs = output[0][0][0, ..., 0] if self.model.decoder.wrap else output
                    action = probs.argmax(-1)
            td["action"] = action
            returned = env.step(td)
            assert returned is env.outputs[-1]
            assert env.received[-1] is td
        rewards = env.outputs[-1]["reward"][0]
        start = rewards.argmax().item()
        self.result = DummyResult([-rewards[start].item()],
                                  [env.selected_node_list[0, start].tolist()], {"dummy": True})
        return self.result


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old_threads)
        guard = mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No GPU"))
        guard.start()
        cls.addClassCleanup(guard.stop)

    def item(self, tsp):
        return torch.zeros(1, 3, 2 if tsp else 3)

    def assert_restored(self, backend, old_step, hooks=0, encoder_hooks=0):
        self.assertEqual(backend.env.step, old_step)
        self.assertEqual(len(backend.model.decoder._forward_hooks), hooks)
        encoder = getattr(backend.model, "encoder", None)
        if encoder is not None:
            self.assertEqual(len(encoder._forward_pre_hooks), encoder_hooks)

    def test_original_actions_outputs_aug0_and_forced_alignment(self):
        for tsp in (True, False):
            with self.subTest(tsp=tsp):
                reference, backend = DummyBackend(tsp), DummyBackend(tsp, wrap=not tsp)
                query = [self.item(tsp)]
                expected = reference.run(query)
                old_step = backend.env.step
                result, a, info = capture_td(backend, query)
                self.assertIs(result, backend.result)
                self.assertEqual(result, expected)
                self.assertEqual(backend.calls, 1)
                self.assertNotIn("step", vars(backend.env))
                self.assert_restored(backend, old_step)
                self.assertEqual(info["executed_augmentations"], 8 if tsp else 128)
                self.assertEqual(info["serialized_augmentation"], 0)
                p = 1 if tsp else 3
                self.assertEqual(a["decoder_probs"].shape, (3, p, 3))
                self.assertEqual(a["decoder_probs"].dtype, np.float32)
                self.assertFalse(a["decoder_valid"][0].any())
                self.assertTrue(a["decoder_forced"][0].all())
                self.assertEqual(np.count_nonzero(a["decoder_probs"][0]), 0)
                np.testing.assert_array_equal(a["decoder_called"], [tsp, True, True])
                np.testing.assert_array_equal(a["decoder_env_count"],
                                              [info["executed_augmentations"] if tsp else 0,
                                               info["executed_augmentations"], info["executed_augmentations"]])
                for actual, original in zip(backend.env.actions, reference.env.actions):
                    torch.testing.assert_close(actual, original)
                for actual, original in zip(backend.env.outputs, reference.env.outputs):
                    torch.testing.assert_close(actual["reward"], original["reward"])
                    torch.testing.assert_close(actual["done"], original["done"])
                np.testing.assert_array_equal(a["action"], torch.stack(backend.env.actions)[:, 0])
                np.testing.assert_array_equal(a["final_tours"], backend.env.selected_node_list[0])
                np.testing.assert_array_equal(a["final_rewards"], np.arange(p) - 10)
                np.testing.assert_array_equal(a["final_costs"], 10 - np.arange(p))
                np.testing.assert_array_equal(a["first_node"], np.arange(p))
                np.testing.assert_array_equal(a["pre_current"][0], -np.ones(p))
                np.testing.assert_array_equal(a["post_current"], a["action"])
                np.testing.assert_array_equal(a["pre_current"][1:], a["action"][:-1])
                np.testing.assert_array_equal(a["pre_visited"][1:], a["post_visited"][:-1])
                np.testing.assert_array_equal(a["pre_feasible"], ~a["pre_visited"])
                np.testing.assert_array_equal(a["post_feasible"], ~a["post_visited"])
                for t in (1, 2):
                    visited = torch.from_numpy(a["pre_visited"][t])
                    logits = torch.arange(3).float().expand(p, 3).masked_fill(visited, -float("inf"))
                    expected_probs = torch.softmax(logits, -1)
                    if tsp:
                        expected_probs.masked_fill_(visited, -2.)
                    np.testing.assert_array_equal(a["decoder_probs"][t], expected_probs)
                    self.assertTrue(a["decoder_valid"][t].all())
                self.assertTrue(a["post_finished"][-1].all())
                self.assertFalse(a["pre_finished"].any())
                self.assertEqual(info["memory_bytes"], sum(x.nbytes for x in a.values()))
                self.assertLessEqual(info["memory_reserved_peak_bytes"], info["max_bytes"])
                if tsp:
                    self.assertEqual(a["decoder_probs"][1, 0, 0], -2.)
                    self.assertIn("deterministic physical node0", info["first_node_rule"])
                else:
                    np.testing.assert_array_equal(a["initial_row_roles"], backend.roles[0][0])
                    np.testing.assert_array_equal(a["initial_col_roles"], backend.roles[1][0])
                    self.assertFalse(np.array_equal(a["initial_col_roles"][:, :3], np.eye(3)))

    def test_failure_cap_and_alignment_restore_existing_hooks_and_instance_step(self):
        for fault in ("run", "missing", "double", "cap"):
            with self.subTest(fault=fault):
                backend = DummyBackend(fault=fault)
                backend.env.step = backend.env.step
                old_step = backend.env.step
                observer = backend.model.decoder.register_forward_hook(lambda *_: None)
                encoder_observer = backend.model.encoder.register_forward_pre_hook(lambda *_: None)
                with self.assertRaises(MemoryError if fault == "cap" else (RuntimeError, ValueError)):
                    capture_td(backend, [self.item(True)], max_bytes=1 if fault == "cap" else 100_000)
                self.assertIs(backend.env.step, old_step)
                self.assert_restored(backend, old_step, 1, 1)
                observer.remove()
                encoder_observer.remove()

    def test_singleton_guard_unrecorded_roles_and_repeat_capture(self):
        backend = DummyBackend(False, roles=False)
        with self.assertRaises(ValueError):
            capture_td(backend, [self.item(False), self.item(False)])
        with self.assertRaises(ValueError):
            capture_td(backend, [self.item(False).expand(2, -1, -1)])
        self.assertEqual(backend.calls, 0)
        for _ in range(2):
            _, arrays, info = capture_td(backend, [self.item(False)])
            self.assertNotIn("initial_col_roles", arrays)
            self.assertIn("unrecorded", info["initial_roles"])
        self.assertEqual(backend.calls, 2)

    def test_encoder_role_allocation_cap_restores_hooks(self):
        backend = DummyBackend(False)
        _, _, info = capture_td(backend, [self.item(False)])
        backend = DummyBackend(False)
        old_step = backend.env.step
        with self.assertRaises(MemoryError):
            capture_td(backend, [self.item(False)], max_bytes=info["memory_reserved_peak_bytes"] - 1)
        self.assertEqual(backend.env.t, 0)
        self.assert_restored(backend, old_step)


if __name__ == "__main__":
    unittest.main()
