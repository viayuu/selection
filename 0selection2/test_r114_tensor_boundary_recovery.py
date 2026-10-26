"""Three CPU-only tests; guarded CUDA values are doubles, never device probes."""

import ast
import copy
import unittest
from pathlib import Path

import numpy as np
import torch

from code.V4 import r114_tensor_boundary_recovery as recovery


SCRIPTS = Path("/public/home/shiys/easynco_v3_bridge/scripts")


def audit_function(source):
    common = ast.parse((SCRIPTS / "train_100k_common.py").read_text())
    backend = ast.parse(source)
    functions = [next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == name)
                 for tree, name in ((common, "sample_scale"), (backend, "audit_tours"))]
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<audit-only-cpu-test>", "exec"), namespace)
    return namespace["audit_tours"]


class GuardedCudaValue:
    """Emulate only the forbidden NumPy boundary, using host arrays throughout."""

    def __init__(self, value, events):
        self.value = np.asarray(value, dtype=np.float32)
        self.shape = self.value.shape
        self.events = events

    def __getitem__(self, index):
        return GuardedCudaValue(self.value[index], self.events)

    def __array__(self, dtype=None, copy=None):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")

    def detach(self):
        self.events.append("detach")
        return self

    def cpu(self):
        self.events.append("cpu")
        owner = self

        class HostView:
            def numpy(self):
                owner.events.append("numpy")
                return owner.value

        return HostView()


class TensorBoundaryTests(unittest.TestCase):
    def test_real_cpu_tensors_preserve_values_dtype_shape_and_rng(self):
        tensor = torch.tensor([[0.25, 0.75]], dtype=torch.float32,
                              device="cpu", requires_grad=True)
        rng = torch.get_rng_state().clone()
        saved = tensor.detach().clone()
        array = np.array([7], dtype=np.int64)
        value = {"nested": [tensor, (array, 2.0)]}
        observed = recovery.cpu_numpy_view(value)
        self.assertIsInstance(observed["nested"][0], np.ndarray)
        self.assertEqual(observed["nested"][0].dtype, np.float32)
        self.assertEqual(observed["nested"][0].shape, (1, 2))
        np.testing.assert_array_equal(observed["nested"][0], saved.numpy())
        self.assertIs(observed["nested"][1][0], array)
        self.assertIsInstance(observed["nested"][1], tuple)
        self.assertIs(value["nested"][0], tensor)
        self.assertTrue(tensor.requires_grad)
        self.assertEqual(tensor.device.type, "cpu")
        self.assertTrue(torch.equal(tensor, saved))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_actual_frozen_audit_fails_then_private_revision_passes_on_cpu(self):
        source = (SCRIPTS / "train_100k_backends.py").read_text()
        original = audit_function(source)
        revised = audit_function(recovery.revised_source(source))
        events = []
        raw = dict(depot=[[[0.0, 0.0]]], loc=[[[0.5, 0.0], [1.0, 0.0]]],
                   demand=[[0.25, 0.5]])
        item = {key: GuardedCudaValue(value, events) for key, value in raw.items()}
        numpy_asarray = np.asarray
        with self.assertRaisesRegex(TypeError, "can't convert cuda:0"):
            original("CVRP", [item], [[0, 1, 2, 0]], [2.0])
        observed = revised("CVRP", [item], [[0, 1, 2, 0]], [2.0])
        self.assertTrue(observed["feasible"])
        self.assertEqual(observed["maximum_cost_error"], 0.0)
        self.assertEqual(events, ["detach", "cpu", "numpy"] * 3)
        self.assertIs(np.asarray, numpy_asarray)
        with self.assertRaisesRegex(TypeError, "can't convert cuda:0"):
            original("CVRP", [item], [[0, 1, 2, 0]], [2.0])
        with self.assertRaisesRegex(ValueError, "objective mismatch"):
            revised("CVRP", [item], [[0, 1, 2, 0]], [3.0])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            revised("CVRP", [item], [[0, 1, 1, 2, 0]], [2.0])
        overloaded = copy.copy(item)
        overloaded["demand"] = GuardedCudaValue([[0.75, 0.75]], [])
        with self.assertRaisesRegex(ValueError, "capacity"):
            revised("CVRP", [overloaded], [[0, 1, 2, 0]], [2.0])
        for key in raw:
            np.testing.assert_array_equal(item[key].value, np.asarray(raw[key], dtype=np.float32))

    def test_exact_patch_guard_and_unchanged_non_cvrp_audit(self):
        source = (SCRIPTS / "train_100k_backends.py").read_text()
        revised = recovery.revised_source(source)
        self.assertEqual(revised.replace(recovery.AUDIT_INSERTION, "", 1), source)
        with self.assertRaisesRegex(ValueError, "Frozen native adapter changed"):
            recovery.revised_source(source + "\n")
        # Non-CVRP audit behavior stays byte-for-byte identical after the guard.
        original, private = audit_function(source), audit_function(revised)
        tsp = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        cost = 2.0 + np.sqrt(2.0)
        self.assertEqual(original("TSP", [tsp], [[0, 1, 2]], [cost]),
                         private("TSP", [tsp], [[0, 1, 2]], [cost]))


if __name__ == "__main__":
    unittest.main()
