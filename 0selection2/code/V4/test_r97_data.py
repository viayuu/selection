"""Small CPU-only isolation fixtures; no real cache preparation or model imports.

Run from the repository in easynco:
    python -B -m unittest code.V4.test_r97_data -v
"""

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import r97_data as data


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data._atomic_json(path, value)


class ContextIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.release = self.root / "release"
        self.features = self.root / "features" / "TRAIN_VAL"
        self.output = self.root / "contexts"
        self.cache = self.root / "cache"
        self.columns = ("vary_a", "train_constant", "vary_b", "always_constant")
        self.X = np.array([[0, 5, 9, 7], [1, 5, 8, 7], [2, 5, 7, 7],
                           [3, 5, 6, 7], [4, 5, 5, 7], [5, 5, 4, 7],
                           [100, 900, 1, 7], [200, 800, 2, 7]], dtype=np.float32)
        # Class 1 is absent on purpose: never densify the observed labels.
        self.y = np.array([2, 0, 2, 2, 0, 2], dtype=np.int64)
        self.ids = np.array([10, 11, 10, 12, 13, 12], dtype=np.int64)
        (self.release / "TSPtrain").mkdir(parents=True)
        np.savez(self.release / "TSPtrain/input_metadata.npz", source_index=self.ids,
                 ignored_non_input=np.array(["never access this member"]))
        self.features.mkdir(parents=True)
        np.save(self.features / "TSP.npy", self.X)
        self.train = self._record("train", 6)
        self.val = self._record("val", 2)
        train_cache = self.cache / "train/TSP"
        np.save(train_cache / "winner.npy", self.y)
        prepared = dict(seed=2, data_root=str(self.release), cache_source=str(self.cache),
                        rows_per_problem=dict(train=6, val=2, test=2))
        _json(self.prepared / "prepared.json", prepared)
        manifest = dict(complete=True, data_root=str(self.release),
                        cache_root=str(self.root / "obsolete_DO_NOT_USE"),
                        splits=dict(train=dict(TSP=self.train), val=dict(TSP=self.val),
                                    test=dict(TSP={"never_read_labels": True})))
        _json(self.prepared / "source_cache_manifest.json", manifest)
        identity = dict(problem="TSP", data_root=str(self.release), splits=["train", "val"],
                        rows=dict(train=6, val=2), sources=dict(train=self.train, val=self.val),
                        preparation_sha256=data._file_identity(self.prepared / "prepared.json")["sha256"],
                        feature_source_sha256="a" * 64,
                        feature_definition=dict(columns=list(self.columns), version="fixture_v1",
                            output_dtype="float32", train_fitted_feature_statistics=False))
        _json(self.features / "TSP.json", dict(complete=True, identity=identity,
                                              shape=[8, 4], dtype="float32"))

    def _record(self, split, rows):
        directory = self.cache / split / "TSP"
        directory.mkdir(parents=True)
        source_files = []
        if split == "train":
            path = self.release / "TSPtrain/input_metadata.npz"
            stat = path.stat()
            source_files.append(dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        # These exist only as tripwires. Preparation must never open/hash them.
        for name in ("dataset.pkl", "raw_label.pkl", "costs.npy"):
            path = directory / name
            path.write_bytes(b"FORBIDDEN: not an input to context preparation")
            source_files.append(dict(path=str(path), size_bytes=path.stat().st_size))
        if split != "train":
            (directory / "winner.npy").write_bytes(b"FORBIDDEN held-out supervision")
        record = dict(problem="TSP", split=split, rows=rows, rows_written=rows, complete=True,
                      directory=str(directory), data_root=str(self.release),
                      pool=["C", "A", "reld_mt"], pool_ids=[2, 0, 3],
                      model_global_solvers=["A", "B", "C", "RELD_MOEL"],
                      scenario_solver_aliases=dict(reld_mt="RELD_MOEL"),
                      raw_cost_dtype="float64", winner_source="native FP64 ind in filtered pool",
                      arrays=dict(winner=dict(file="winner.npy", shape=[rows], dtype="int64")),
                      source_files=source_files)
        _json(directory / "metadata.json", record)
        return record

    def _prepare(self, output=None, feature_root=None):
        return data.prepare_contexts(self.prepared, self.release, feature_root or self.features,
                                     output or self.output, chunk_rows=2)

    def _table(self, output=None):
        table = data.load_context(output or self.output, "TSP")
        self.addCleanup(table.close)
        return table

    def test_train_only_preparation_readonly_contract_and_heldout_invariance(self):
        original_open = Path.open
        opened = []

        def guarded_open(path, *args, **kwargs):
            opened.append(path)
            if path.suffix == ".pkl" or path.name == "costs.npy" or (
                    path.name == "winner.npy" and "val" in path.parts):
                raise AssertionError(f"Forbidden source access: {path}")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            paths = self._prepare(feature_root=self.features.parent)
        self.assertEqual(paths, {"TSP": self.output / "TSP/identity.json"})
        table = self._table()
        self.assertIsInstance(table, data.ContextTable)
        for array in (table.x, table.y, table.source_ids):
            self.assertIsInstance(array, np.memmap)
            self.assertFalse(array.flags.writeable)
        self.assertEqual(table.x.dtype, np.float32)
        self.assertEqual(table.y.dtype, np.int64)
        self.assertEqual(table.source_ids.dtype, np.int64)
        np.testing.assert_array_equal(table.kept_columns, [0, 2])
        np.testing.assert_array_equal(table.x, self.X[:6, [0, 2]])
        np.testing.assert_array_equal(table.y, self.y)
        np.testing.assert_array_equal(table.source_ids, self.ids)
        self.assertEqual(table.pool, ("C", "A", "reld_mt"))
        self.assertEqual(table.pool_ids, (2, 0, 3))
        self.assertEqual(table.x.shape[0], 6)
        self.assertIn(self.cache / "train/TSP/winner.npy", opened)
        sources = table.metadata["identity"]["source_files"]
        self.assertTrue(all(s["size_bytes"] > 0 and len(s["sha256"]) == 64 for s in sources.values()))
        self.assertEqual(table.metadata["identity"]["label_splits"], ["train"])

        changed = self.X.copy()
        changed[6:] = [[-99, -99, -99, -99], [np.nan, np.inf, -1, -2]]
        np.save(self.features / "TSP.npy", changed)
        (self.cache / "val/TSP/winner.npy").write_bytes(b"changed heldout winner/cost")
        other_output = self.root / "other_contexts"
        self._prepare(other_output)
        other = self._table(other_output)
        np.testing.assert_array_equal(other.x, table.x)
        np.testing.assert_array_equal(other.y, table.y)
        np.testing.assert_array_equal(other.kept_columns, table.kept_columns)
        self.assertEqual(other.metadata["identity"]["transform"], table.metadata["identity"]["transform"])
        # Provenance correctly changes when held-out INPUT bytes change, not fit.
        self.assertNotEqual(other.metadata["identity"]["source_files"]["canonical_features"]["sha256"],
                            sources["canonical_features"]["sha256"])

    def test_base_id_isolation_all_related_copies_and_explicit_shuffle(self):
        self._prepare()
        table = self._table()
        query = self.X[:2]
        with self.assertRaisesRegex(ValueError, "overlap"):
            table.for_queries(query, source_ids=np.array([10, 99]))
        batch = table.for_queries(query, source_ids=np.array([10, 12]), train_diagnostic=True)
        np.testing.assert_array_equal(batch.context_rows, [1, 4])
        self.assertEqual(batch.excluded_context_rows, 4)
        self.assertIsNone(batch.shuffle_seed)
        np.testing.assert_array_equal(data.exclude_context_rows(self.ids, np.array([10, 12, 10])), [1, 4])
        with self.assertRaisesRegex(ValueError, "No context"):
            table.for_queries(self.X[:4], source_ids=np.array([10, 11, 12, 13]), train_diagnostic=True)
        natural = table.for_queries(query, source_ids=np.array([90, 91]))
        np.testing.assert_array_equal(natural.context_rows, np.arange(6))
        shuffled = table.for_queries(query, source_ids=np.array([90, 91]), shuffle=True)
        repeated = table.for_queries(query, source_ids=np.array([90, 91]), shuffle=True, seed=2)
        np.testing.assert_array_equal(shuffled.context_rows, repeated.context_rows)
        np.testing.assert_array_equal(np.sort(shuffled.context_rows), natural.context_rows)
        self.assertEqual(shuffled.shuffle_seed, 2)

    def test_full_class_order_remapping_and_invalid_winners(self):
        np.testing.assert_array_equal(data.remap_local_labels(self.y, ("C", "A", "reld_mt"),
                                                              ("A", "reld_mt", "C")), [1, 2, 1, 1, 2, 1])
        for target in (("A", "C"), ("A", "C", "C"), ("A", "C", "B")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                data.remap_local_labels(self.y, ("C", "A", "reld_mt"), target)
        for labels in (self.y.astype(np.float32), np.array([3] * 6, dtype=np.int64),
                       np.array([-1] * 6, dtype=np.int64), self.y[:-1]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                np.save(self.cache / "train/TSP/winner.npy", labels)
                self._prepare()
        np.save(self.cache / "train/TSP/winner.npy", self.y)
        bad = dict(self.train, pool_ids=[0, 2, 3])
        with self.assertRaisesRegex(ValueError, "disagree"):
            data._pool(bad)

    def test_query_api_is_input_only_named_schema_and_nonfinite_rejection(self):
        self._prepare()
        table = self._table()
        self.assertEqual(tuple(inspect.signature(table.transform_queries).parameters), ("X",))
        named = {name: self.X[:2, i] for i, name in reversed(list(enumerate(self.columns)))}
        np.testing.assert_array_equal(table.transform_queries(named), self.X[:2, [0, 2]])
        for key in ("y", "labels", "winner", "costs", "ind"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    table.transform_queries(dict(named, **{key: self.y[:2]}))
                with self.assertRaises(TypeError):
                    table.transform_queries(self.X[:2], **{key: self.y[:2]})
                with self.assertRaises(TypeError):
                    table.for_queries(self.X[:2], source_ids=np.array([99, 98]), **{key: self.y[:2]})
        for X in (self.X[:, :2], self.X[0], self.X.astype(complex),
                  np.full_like(self.X, np.nan), np.full_like(self.X, np.inf),
                  np.full(self.X.shape, 1e300)):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                table.transform_queries(X)
        with self.assertRaisesRegex(ValueError, "align"):
            table.for_queries(self.X[:2], source_ids=np.array([99]))
        with self.assertRaises(ValueError):
            data.exclude_context_rows(self.ids, np.array([10.0]))
        for ids in (np.array([-1]), np.array([[10]]), np.array([2**63], dtype=np.uint64)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                data.exclude_context_rows(self.ids, ids)
        np.testing.assert_array_equal(data._source_ids(np.array([2**63 - 1], dtype=np.uint64)),
                                      np.array([2**63 - 1], dtype=np.int64))

    def test_output_reuse_identity_change_and_artifact_corruption_preserve_files(self):
        self._prepare()
        files = list((self.output / "TSP").iterdir())
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
        self._prepare()
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files})
        changed = self.y.copy()
        changed[0] = 0
        np.save(self.cache / "train/TSP/winner.npy", changed)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self._prepare()
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files})
        y_path = self.output / "TSP/y.npy"
        np.save(y_path, changed)
        corrupted = y_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            data.load_context(self.output, "TSP")
        self.assertEqual(y_path.read_bytes(), corrupted)

    def test_invalid_fit_feature_identity_and_source_alignment_fail_closed(self):
        with patch.object(data, "DEFAULT_DATA_ROOT", self.release):
            with self.assertRaisesRegex(ValueError, "TRAIN81000"):
                self._prepare()
        for X in (np.ones((3, 4), dtype=np.float32), self.X[:0],
                  np.full((3, 4), np.nan, dtype=np.float32)):
            with self.subTest(shape=X.shape), self.assertRaises(ValueError):
                data.FeatureTransform.fit(X, self.columns, chunk_rows=2)
        bad = self.X.copy()
        bad[0, 0] = np.inf
        np.save(self.features / "TSP.npy", bad)
        with self.assertRaisesRegex(ValueError, "Nonfinite TRAIN"):
            self._prepare()
        np.save(self.features / "TSP.npy", self.X[:, :3])
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self._prepare()
        np.save(self.features / "TSP.npy", self.X)
        path = self.features / "TSP.json"
        receipt = json.loads(path.read_text())
        receipt["identity"]["splits"] = ["val", "train"]
        _json(path, receipt)
        with self.assertRaisesRegex(ValueError, "identity/schema"):
            self._prepare()
        self.assertFalse((self.output / "TSP").exists())


if __name__ == "__main__":
    unittest.main()
