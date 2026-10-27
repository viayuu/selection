"""CPU-only fixtures; never import or execute the native solver."""

import importlib
import random
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .r118_reld_trace import capture_reld


@dataclass
class InferenceResult:
    costs: list
    tours: list
    audit: dict


class DummyEnv:
    def __init__(self, *, resources=True):
        self.problem_size = 4
        self.POMO_IDX = torch.tensor([[0, 1]])
        self.selected_count, self.current_node = 0, None
        self.load = torch.ones(1, 2)
        self.finished = torch.zeros(1, 2, dtype=torch.bool)
        self.visited_ninf_flag = torch.zeros(1, 2, 5)
        self.ninf_mask = self.visited_ninf_flag.clone()
        self.selected_node_list = torch.empty(1, 2, 0, dtype=torch.int64)
        self.step_state = types.SimpleNamespace(START_NODE=torch.tensor([[4, 2]]))
        if resources:
            self.current_time, self.length = torch.zeros(1, 2), torch.zeros(1, 2)

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[..., None]), -1)
        self.load.sub_(selected.float() / 10)
        self.visited_ninf_flag.scatter_(2, selected[..., None], -float("inf"))
        self.ninf_mask.copy_(self.visited_ninf_flag)
        self.ninf_mask[..., 1] = -float("inf")  # Masked does not necessarily mean visited.
        self.ninf_mask[..., 0] = 0
        if hasattr(self, "current_time"):
            self.current_time.add_(0.25)
            self.length.add_(0.5)
        self.finished[0, 0] = self.selected_count >= 3
        self.finished[0, 1] = self.selected_count >= 5
        done = bool(self.finished.all())
        return self, torch.tensor([[-7.0, -9.0]]) if done else None, done


class DummyDecoder(nn.Module):
    def __init__(self, tuple_output):
        super().__init__()
        self.tuple_output = tuple_output
        self.bad_shape = self.repeat = False

    def forward(self, state):
        probabilities = torch.zeros(1, 2, 5, dtype=torch.float64)
        probabilities[0, 0, 0] = 1
        probabilities[0, 1, 3 if state.selected_count == 2 else 0] = 1
        if self.bad_shape:
            probabilities = probabilities[..., :-1]
        return (probabilities, torch.tensor(0.0)) if self.tuple_output else probabilities


class DummyModel(nn.Module):
    def __init__(self, tuple_output):
        super().__init__()
        self.decoder = DummyDecoder(tuple_output)

    def forward(self, state):
        count = state.selected_count
        if count == 0 or count == 3:  # A forced row AFTER a decoder event tests stale clearing.
            return torch.zeros(1, 2, dtype=torch.int64)
        if count == 1:
            return torch.tensor([[4, 2]])
        output = self.decoder(state)
        if self.decoder.repeat:
            self.decoder(state)
        return (output[0] if isinstance(output, tuple) else output).argmax(-1)


class DummyBackend:
    def __init__(self, method, bridge=None):
        self.method, self.problem = method, "VRPB" if bridge is None else "CVRP"
        self.env_class, self.module = DummyEnv, bridge
        self.model = DummyModel(method == "RELD_MOEL")
        self.calls, self.native_steps = 0, 0
        self.multiple = self.raise_in_step = False

    def run(self, items):
        self.calls += 1
        self.seen_items = items
        self.inference_enabled = torch.is_inference_mode_enabled()
        captured = []
        module = importlib.import_module("CVRPEnv") if self.module else None
        original = module.CVRPEnv if module else self.env_class
        backend = self

        class WitnessEnv(original):
            def __init__(environment, **kwargs):
                super().__init__(**kwargs)
                captured.append(environment)

            def step(environment, selected):
                backend.native_steps += 1
                if backend.raise_in_step and environment.selected_count == 2:
                    raise RuntimeError("native step failed")
                result = super().step(selected)
                if result[-1]:
                    environment.witness_reward = result[1].detach()
                return result

        def rollout():
            env = (module.CVRPEnv if module else WitnessEnv)(resources=module is None)
            self.environment = env
            if self.multiple:
                (module.CVRPEnv if module else WitnessEnv)()
            # Exercise RNG use without reseeding inside the observer or changing query data.
            random.random()
            torch.rand(1)
            done = False
            while not done:
                selected = self.model(env)
                _, reward, done = env.step(selected)
            return -reward.max(1).values

        if module:
            module.CVRPEnv = WitnessEnv
        try:
            scores = rollout()
        finally:
            if module:
                module.CVRPEnv = original
        env = captured[0]
        best = env.witness_reward.argmax(1)
        self.result = InferenceResult(scores.tolist(), env.selected_node_list[torch.arange(1), best].tolist(),
                                      {"dummy": True})
        return self.result


class TraceTests(unittest.TestCase):
    def test_native_output_rng_and_event_alignment(self):
        for method in ("RELD_MOEL", "RELD_MTL", "RELD_CVRP"):
            with self.subTest(method=method):
                module = types.ModuleType("CVRPEnv")
                module.CVRPEnv = DummyEnv
                bridge = types.SimpleNamespace(CVRPEnv=DummyEnv) if method == "RELD_CVRP" else None
                backend, baseline = DummyBackend(method, bridge), DummyBackend(method, bridge)
                items = [{"native_input": torch.tensor([1.0])}]
                with patch.dict(sys.modules, {"CVRPEnv": module}):
                    python_rng, torch_rng = random.getstate(), torch.get_rng_state()
                    with torch.inference_mode():
                        expected = baseline.run(items)
                    expected_rng, expected_torch_rng = random.getstate(), torch.get_rng_state()
                    random.setstate(python_rng)
                    torch.set_rng_state(torch_rng)
                    result, arrays, info = capture_reld(backend, items)
                    self.assertEqual(result, expected)
                    self.assertIs(result, backend.result)
                    self.assertIs(backend.seen_items, items)
                    self.assertEqual(random.getstate(), expected_rng)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_rng))
                    self.assertEqual(backend.calls, 1)
                    self.assertTrue(backend.inference_enabled)
                    self.assertIs(backend.env_class, DummyEnv)
                    self.assertIs(module.CVRPEnv, DummyEnv)
                    if bridge:
                        self.assertIs(bridge.CVRPEnv, DummyEnv)
                    self.assertFalse(backend.model.decoder._forward_hooks)
                np.testing.assert_array_equal(arrays["step"], np.arange(5))
                np.testing.assert_array_equal(arrays["trajectory_id"], [[0, 1]])
                np.testing.assert_array_equal(arrays["pomo_idx"], [[0, 1]])
                np.testing.assert_array_equal(arrays["node_id"], np.arange(5))
                np.testing.assert_array_equal(arrays["start_node"], [[4, 2]])
                np.testing.assert_array_equal(arrays["action"], baseline.environment.selected_node_list.numpy().transpose(2, 0, 1))
                np.testing.assert_array_equal(arrays["pre_current_node"][0], [[-1, -1]])
                np.testing.assert_array_equal(arrays["post_current_node"], arrays["action"])
                np.testing.assert_array_equal(arrays["pre_load"][1:], arrays["post_load"][:-1])
                np.testing.assert_array_equal(arrays["pre_load"][0], [[1, 1]])
                np.testing.assert_array_equal(arrays["pre_finished"][1:], arrays["post_finished"][:-1])
                np.testing.assert_array_equal(arrays["decoder_valid"][:, 0, 0], [False, False, True, False, True])
                np.testing.assert_array_equal(arrays["forced"], ~arrays["decoder_valid"])
                self.assertEqual(arrays["decoder_probabilities"].shape, (5, 1, 2, 5))
                self.assertEqual(arrays["decoder_probabilities"].dtype, np.float32)
                self.assertTrue(np.isnan(arrays["decoder_probabilities"][[0, 1, 3]]).all())
                self.assertEqual(arrays["decoder_probabilities"][2, 0, 1, 3], 1)
                self.assertEqual(arrays["decoder_probabilities"][4, 0, 1, 0], 1)
                self.assertTrue(arrays["pre_finished"][3, 0, 0])
                self.assertFalse(arrays["pre_finished"][3, 0, 1])
                self.assertFalse(arrays["visited_node"][1, 0, 0, 1])
                self.assertFalse(arrays["feasible"][1, 0, 0, 1])
                self.assertEqual(arrays["visited_node"].dtype, np.bool_)
                np.testing.assert_array_equal(arrays["all_start_reward"], [[-7, -9]])
                np.testing.assert_array_equal(arrays["all_start_cost"], [[7, 9]])
                np.testing.assert_array_equal(arrays["all_start_tours"], baseline.environment.selected_node_list.numpy())
                for name in ("time", "length"):
                    for prefix in ("pre_", "post_"):
                        self.assertTrue((arrays[prefix + name + "_present"] == (bridge is None)).all())
                        if bridge is not None:
                            self.assertTrue(np.isnan(arrays[prefix + name]).all())
                        else:
                            increment = 0.25 if name == "time" else 0.5
                            counts = np.arange(5) + (prefix == "post_")
                            np.testing.assert_array_equal(arrays[prefix + name][:, 0, 0], counts * increment)
                self.assertLessEqual(info["peak_array_bytes_bound"], info["max_bytes"])
                self.assertEqual(info["array_bytes"], sum(value.nbytes for value in arrays.values()))
                self.assertFalse(info["historical_bitwise_replay"])
                self.assertFalse(info["native_mask_is_feasibility_certificate"])

    def test_fail_closed_and_restore(self):
        cases = (("limit", MemoryError), ("multiple", RuntimeError),
                 ("bad_shape", ValueError), ("repeat", RuntimeError), ("step_error", RuntimeError))
        for method in ("RELD_MOEL", "RELD_CVRP"):
            for case, exception in cases:
                with self.subTest(method=method, case=case):
                    module = types.ModuleType("CVRPEnv")
                    module.CVRPEnv = DummyEnv
                    bridge = types.SimpleNamespace(CVRPEnv=DummyEnv) if method == "RELD_CVRP" else None
                    backend = DummyBackend(method, bridge)
                    backend.multiple = case == "multiple"
                    backend.raise_in_step = case == "step_error"
                    backend.model.decoder.bad_shape = case == "bad_shape"
                    backend.model.decoder.repeat = case == "repeat"
                    with patch.dict(sys.modules, {"CVRPEnv": module}):
                        with self.assertRaises(exception):
                            capture_reld(backend, [{}], max_bytes=1 if case == "limit" else 100_000)
                        self.assertIs(module.CVRPEnv, DummyEnv)
                        self.assertIs(backend.env_class, DummyEnv)
                        if bridge:
                            self.assertIs(bridge.CVRPEnv, DummyEnv)
                        self.assertFalse(backend.model.decoder._forward_hooks)
                    self.assertEqual(backend.calls, 1)
                    if case == "limit":
                        self.assertEqual(backend.native_steps, 1)  # Outer WitnessEnv entered, native step not run.
                        self.assertEqual(backend.environment.selected_count, 0)

    def test_memory_guard_precedes_tensor_copy_and_stack_is_bounded(self):
        backend = DummyBackend("RELD_MOEL")
        with patch.object(torch.Tensor, "to", side_effect=AssertionError("copy before byte guard")):
            with self.assertRaises(MemoryError):
                capture_reld(backend, [{}], max_bytes=1)
        _, _, info = capture_reld(backend, [{}])
        bound = info["peak_array_bytes_bound"]
        _, arrays, exact = capture_reld(backend, [{}], max_bytes=bound)
        self.assertEqual(exact["peak_array_bytes_bound"], bound)
        self.assertEqual(sum(array.nbytes for array in arrays.values()) * 2, bound)
        with self.assertRaises(MemoryError):
            capture_reld(backend, [{}], max_bytes=bound - 1)
        self.assertIs(backend.env_class, DummyEnv)
        self.assertFalse(backend.model.decoder._forward_hooks)

    def test_singleton_guard_does_not_run(self):
        backend = DummyBackend("RELD_MOEL")
        for items in ([], [{}, {}]):
            with self.assertRaisesRegex(ValueError, "singleton"):
                capture_reld(backend, items)
        self.assertEqual(backend.calls, 0)


if __name__ == "__main__":
    unittest.main()
