"""Small CPU-only native-cost and immutable row-binding fixtures."""

import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import r97_data as old
from . import r98_data as data


def make_fixture(root):
    release, contexts, output = (root / name for name in ("release", "contexts", "targets"))
    native = release / "TSPtrain"
    native.mkdir(parents=True)
    ids = np.array([10, 11, 10, 12, 13, 12], dtype=np.int64)
    costs = np.array([[1., 1. + 2.**-40, 2.], [2., 2., 4.], [4., 5., 4.],
                      [8., 8. + 2.**-35, 12.], [16., 17., 20.], [32., 33., 32.]])
    np.savez(native / "input_metadata.npz", source_index=ids)
    # Deliberately reversed insertion order; row keys, not pickle order, bind X/G.
    with (native / "raw_label.pkl").open("wb") as stream:
        pickle.dump({str(i): dict(cost=costs[i].tolist(), ind=999) for i in reversed(range(6))}, stream)
    methods, pool_ids = ["C", "A", "reld_mt"], [2, 0, 3]
    native_metadata = dict(problem="TSP", split="train", instances=6, cost_dtype="float64",
        methods=methods, global_solver_ids=pool_ids,
        files={name: dict(sha256=old._file_identity(native / name)["sha256"])
               for name in ("raw_label.pkl", "input_metadata.npz")})
    old._atomic_json(native / "metadata.json", native_metadata)
    directory = contexts / "TSP"
    directory.mkdir(parents=True)
    x = np.column_stack((np.arange(6), np.arange(6)[::-1])).astype(np.float32)
    transform = old.FeatureTransform(("a", "constant", "b"), (0, 2))
    identity = dict(schema=old.SCHEMA, problem="TSP", data_root=str(release), rows=6,
        transform=transform.to_dict(), pool=methods, pool_ids=pool_ids,
        label_splits=["train"], winner_rule=old.WINNER_RULE,
        row_order=data.ROW_ORDER, context_subsampling=False, class_quotas=False,
        source_files=dict(train_input_metadata=old._file_identity(native / "input_metadata.npz")))
    artifacts = {key: old._write_array(directory / name, values.shape, values.dtype, [(0, values)])
                 for key, name, values in (("x", "X.npy", x),
                    ("y", "y.npy", np.zeros(6, dtype=np.int64)),
                    ("source_ids", "source_ids.npy", ids))}
    old._atomic_json(directory / "identity.json", dict(complete=True, identity=identity,
                    identity_sha256=old._json_sha(identity), artifacts=artifacts))
    for split in ("val", "test", "LIB"):
        forbidden = release / f"TSP{split}"
        forbidden.mkdir()
        (forbidden / "raw_label.pkl").write_bytes(b"FORBIDDEN QUERY OUTCOMES")
    return release, contexts, output, costs


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.release, self.contexts, self.output, self.costs = make_fixture(Path(self.temp.name))

    def prepare(self):
        return data.prepare_targets(self.release, self.contexts, self.output)

    def test_train_only_fp64_natural_order_full_pool_and_provenance(self):
        opened = []
        original = Path.open

        def guarded(path, *args, **kwargs):
            opened.append(path)
            if any(part in ("TSPval", "TSPtest", "TSPLIB") for part in path.parts):
                raise AssertionError("Current/held-out query outcomes were accessed")
            if path.name in ("dataset.pkl", "costs.npy"):
                raise AssertionError("Only native TRAIN costs are allowed")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            paths = self.prepare()
        self.assertEqual(paths, {"TSP": self.output / "TSP/identity.json"})
        with data.load_targets(self.output, "TSP") as targets, old.load_context(self.contexts, "TSP") as context:
            self.assertIsInstance(targets.g, np.memmap)
            self.assertEqual(targets.g.dtype, np.float64)
            self.assertFalse(targets.g.flags.writeable)
            np.testing.assert_array_equal(targets.g, data.relative_gaps(self.costs))
            self.assertGreater(targets.g[0, 1], 0.)
            self.assertEqual(np.float32(self.costs[0, 1]), np.float32(self.costs[0, 0]))
            identity = data.validate_target_binding(context, targets.metadata)
            self.assertEqual(identity["rows"], 6)
            self.assertEqual(identity["pool"], ["C", "A", "reld_mt"])
            self.assertEqual(identity["pool_ids"], [2, 0, 3])
            self.assertEqual(identity["context_metadata_sha256"], old._json_sha(context.metadata))
            self.assertEqual(identity["raw_label_sha256"], old._file_identity(
                self.release / "TSPtrain/raw_label.pkl")["sha256"])
            self.assertEqual(identity["label_splits"], ["train"])
        self.assertIn(self.release / "TSPtrain/raw_label.pkl", opened)

    def test_readonly_load_does_not_reopen_any_source(self):
        self.prepare()
        original = Path.open

        def guarded(path, *args, **kwargs):
            if path.is_relative_to(self.release) or path.is_relative_to(self.contexts):
                raise AssertionError("Saved-target load reopened outcome/context source")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded), data.load_targets(self.output, "TSP") as targets:
            self.assertEqual(targets.g.shape, (6, 3))

    def test_existing_output_reused_but_never_overwritten(self):
        self.prepare()
        path = self.output / "TSP/G.npy"
        before = old._file_identity(path)
        self.prepare()
        self.assertEqual(before, old._file_identity(path))
        native_path = self.release / "TSPtrain/metadata.json"
        metadata = json.loads(native_path.read_text())
        metadata["methods"] = list(reversed(metadata["methods"]))
        old._atomic_json(native_path, metadata)
        with self.assertRaisesRegex(ValueError, "pool/source"):
            self.prepare()
        self.assertEqual(before, old._file_identity(path))

    def test_exact_source_id_and_kept_column_binding(self):
        self.prepare()
        with data.load_targets(self.output, "TSP") as targets, old.load_context(self.contexts, "TSP") as context:
            modified = copy.copy(context)
            object.__setattr__(modified, "source_ids", context.source_ids[::-1])
            with self.assertRaisesRegex(ValueError, "row binding"):
                data.validate_target_binding(modified, targets.metadata)
            modified = copy.copy(context)
            object.__setattr__(modified, "kept_columns", np.array([2, 0]))
            with self.assertRaisesRegex(ValueError, "row binding"):
                data.validate_target_binding(modified, targets.metadata)

    def test_corrupt_g_or_identity_fails_closed(self):
        self.prepare()
        path = self.output / "TSP/G.npy"
        g = np.load(path).copy()
        g[0, 1] = .5
        np.save(path, g)
        with self.assertRaisesRegex(ValueError, "SHA"):
            data.load_targets(self.output, "TSP")
        receipt = self.output / "TSP/identity.json"
        metadata = json.loads(receipt.read_text())
        metadata["identity"]["pool"].reverse()
        old._atomic_json(receipt, metadata)
        with self.assertRaisesRegex(ValueError, "identity"):
            data.load_targets(self.output, "TSP")

    def test_invalid_costs_rejected_without_fp32_reconstruction(self):
        for costs in (self.costs.astype(np.float32), np.array([[0., 1.]]),
                      np.array([[-1., 1.]]), np.array([[1., np.nan]]),
                      np.array([[1., np.inf]]), np.empty((0, 3))):
            with self.subTest(costs=repr(costs)):
                with self.assertRaises(ValueError):
                    data.relative_gaps(costs)

    def test_wrong_row_count_and_fp32_native_rows_rejected(self):
        path = self.release / "TSPtrain/raw_label.pkl"
        for raw in ({"0": dict(cost=self.costs[0].tolist())},
                    {str(i): dict(cost=self.costs[i].astype(np.float32)) for i in range(6)}):
            with path.open("wb") as stream:
                pickle.dump(raw, stream)
            metadata_path = path.with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text())
            metadata["files"]["raw_label.pkl"]["sha256"] = old._file_identity(path)["sha256"]
            old._atomic_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "row|FP64"):
                self.prepare()
            self.assertFalse((self.output / "TSP").exists())


if __name__ == "__main__":
    unittest.main()
