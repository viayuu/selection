"""Dummy-regressor isolation, scalar mean and sequential lifetime tests. No GPU."""

import copy
from enum import Enum
import gc
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from . import r98_model as model
from . import r98_data as data
from .r97_data import _json_sha


def tables(g=None, rows=6):
    if g is None:
        g = np.array([[0., .1, .3], [.2, 0., .4], [0., .2, .5],
                      [.1, 0., .6], [0., .3, .7], [.4, 0., .8]])
    rows = len(g)
    # No context.y attribute: this adapter must not need classification labels.
    context = SimpleNamespace(x=np.arange(rows*2, dtype=np.float32).reshape(rows, 2),
        source_ids=np.arange(rows, dtype=np.int64) // 2 + 100,
        kept_columns=np.array([0, 4]), pool=("C", "A", "reld_mt"), pool_ids=(2, 0, 3))
    identity = dict(problem="TSP", data_root="/fixture")
    context.metadata = dict(identity=identity, identity_sha256=_json_sha(identity))
    target_identity = dict(schema=data.SCHEMA, problem="TSP", data_root="/fixture", rows=rows,
        pool=list(context.pool), pool_ids=list(context.pool_ids), label_splits=["train"],
        row_order=data.ROW_ORDER, target_definition=data.TARGET_DEFINITION,
        target_transform="none", context_subsampling=False, class_filter=False,
        context_metadata_sha256=_json_sha(context.metadata),
        context_identity_sha256=context.metadata["identity_sha256"],
        source_ids_values_sha256=data._source_ids_sha256(context.source_ids),
        kept_columns_sha256=_json_sha(context.kept_columns.tolist()))
    metadata = dict(complete=True, identity=target_identity, identity_sha256=_json_sha(target_identity))
    return context, SimpleNamespace(g=np.asarray(g, dtype=np.float64), metadata=metadata)


class PerformanceContextTests(unittest.TestCase):
    def test_all_rows_raw_targets_mean_and_one_live_regressor(self):
        context, targets = tables()
        refs, fits, queries = [], [], []

        class Regressor:
            def __init__(self):
                self.column = len(refs)

            def fit(self, x, y, *, kv_cache):
                self_test.assertIs(kv_cache, False)
                fits.append((x.copy(), y.copy()))
                # Simulate a library mutating its private TRAIN inputs.
                x[:] = -900
                y[:] = -800
                return self

            def predict(self, x, *, output_type):
                self_test.assertEqual(output_type, "mean")
                queries.append(x.copy())
                return np.full(len(x), [-.02, -.01, .3][self.column])

        self_test = self

        def factory():
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
            regressor = Regressor()
            refs.append(weakref.ref(regressor))
            return regressor

        instance = model.FrozenPerformanceContext(context, targets, factory)
        query = np.array([[8., 9.], [10., 11.]], dtype=np.float32)
        means = instance.predict_features(query, [900, 901])
        np.testing.assert_array_equal(means, [[-.02, -.01, .3]] * 2)
        self.assertEqual(means.dtype, np.float64)
        np.testing.assert_array_equal(means.argmin(1), [0, 0])
        self.assertEqual(len(fits), 3)
        self.assertTrue(all(ref() is None for ref in refs))
        for column, (x, y) in enumerate(fits):
            np.testing.assert_array_equal(x, context.x)
            np.testing.assert_array_equal(y, targets.g[:, column])
            self.assertEqual(y.dtype, np.float64)
        for x in queries:
            np.testing.assert_array_equal(x, query)

    def test_any_related_source_overlap_rejected_before_fit(self):
        context, targets = tables()
        factory = Mock(side_effect=AssertionError("Must not construct a regressor"))
        instance = model.FrozenPerformanceContext(context, targets, factory)
        with self.assertRaisesRegex(ValueError, "own context"):
            instance.predict_features(np.ones((2, 2)), [900, 100])
        factory.assert_not_called()

    def test_current_query_outcomes_not_accepted(self):
        context, targets = tables()
        instance = model.FrozenPerformanceContext(context, targets, Mock())
        self.assertEqual(list(inspect.signature(instance.predict_features).parameters),
                         ["query_features", "query_source_ids"])
        for name in ("labels", "winner", "costs", "targets"):
            with self.assertRaises(TypeError):
                instance.predict_features(np.ones((1, 2)), [900], **{name: [0]})
        with self.assertRaises(ValueError):
            instance.predict_features(dict(features=[[1., 2.]], winner=[0]), [900])

    def test_exact_constant_shortcut_zero_and_nonzero_never_calls_factory(self):
        context, targets = tables(np.tile([0., 0., .5], (6, 1)))
        factory = Mock(side_effect=AssertionError("Exact constants do not need a model"))
        instance = model.FrozenPerformanceContext(context, targets, factory)
        self.assertIs(instance.fit(), instance)
        np.testing.assert_array_equal(instance.predict_features(np.ones((1, 2)), [900]), [[0., 0., .5]])
        self.assertEqual(instance.predict(np.ones((1, 2)), [900]).item(), 0)
        factory.assert_not_called()

    def test_nearly_constant_and_never_winning_methods_keep_all81000(self):
        g = np.zeros((81000, 3), dtype=np.float64)
        g[:, 2] = .5
        g[-1, 1] = 2.**-50
        context, targets = tables(g)
        regressor = Mock()
        regressor.predict.return_value = np.array([-.1])
        instance = model.FrozenPerformanceContext(context, targets, lambda: regressor)
        means = instance.predict_features(np.ones((1, 2)), [999999])
        self.assertEqual(means.tolist(), [[0., -.1, .5]])
        self.assertEqual(regressor.fit.call_args.args[0].shape, (81000, 2))
        self.assertEqual(regressor.fit.call_args.args[1][-1], 2.**-50)
        regressor.fit.assert_called_once()

    def test_mean_shape_and_nonfinite_fail_closed(self):
        for output in (np.ones((2, 1)), np.ones(1), np.array([np.nan, 0.]),
                       np.array([np.inf, 0.]), dict(mean=[0., 0.]), np.array(["x", "y"])):
            with self.subTest(output=repr(output)):
                context, targets = tables()
                regressor = Mock()
                regressor.predict.return_value = output
                instance = model.FrozenPerformanceContext(context, targets, lambda: regressor)
                with self.assertRaisesRegex(ValueError, "mean output"):
                    instance.predict_features(np.ones((2, 2)), [900, 901])

    @unittest.skipUnless(np.finfo(np.longdouble).max > np.finfo(np.float64).max,
                         "Platform has no wider-than-FP64 real dtype")
    def test_finite_extended_precision_output_cannot_overflow_returned_means(self):
        context, targets = tables()
        regressor = Mock()
        regressor.predict.return_value = np.array([np.finfo(np.longdouble).max])
        instance = model.FrozenPerformanceContext(context, targets, lambda: regressor)
        with self.assertRaisesRegex(ValueError, "overflowed"):
            instance.predict_features(np.ones((1, 2)), [900])

    def test_exact_pool_context_and_g_schema_checks(self):
        context, targets = tables()
        for attribute, value in (("pool", ("A", "C", "reld_mt")),
                                 ("pool_ids", (0, 2, 3)),
                                 ("source_ids", context.source_ids[::-1]),
                                 ("kept_columns", np.array([4, 0]))):
            modified = copy.copy(context)
            setattr(modified, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, "binding"):
                model.FrozenPerformanceContext(modified, targets, Mock())
        for g in (targets.g.astype(np.float32), targets.g[:, :2],
                  targets.g + 1., np.full_like(targets.g, np.nan), -np.ones_like(targets.g)):
            with self.subTest(g=repr(g)), self.assertRaisesRegex(ValueError, "X/G"):
                model.FrozenPerformanceContext(context, SimpleNamespace(g=g, metadata=targets.metadata), Mock())

    def test_invalid_query_features_and_ids_fail_before_factory(self):
        context, targets = tables()
        factory = Mock(side_effect=AssertionError("Must not fit"))
        instance = model.FrozenPerformanceContext(context, targets, factory)
        for x, ids in ((np.ones((1, 3)), [900]), (np.ones((1, 2)), [900, 901]),
                       (np.full((1, 2), np.nan), [900]), (np.ones((1, 2)), [900.]),
                       (np.ones((1, 2)), [-1]), (np.ones((0, 2)), [])):
            with self.subTest(x=x, ids=ids), self.assertRaises(ValueError):
                instance.predict_features(x, ids)
        factory.assert_not_called()

    def test_gpu_guards_are_instance_only_and_propagate_allocation_failure(self):
        mode = Enum("Mode", "GPU CPU DISK")

        class Manager:
            exe_device = SimpleNamespace(type="cuda")

            def _resolve_offload_mode(self):
                return mode.CPU, "fallback"

            def _allocate_output_buffer(self, *args):
                raise AssertionError("Original allocator must not run")

        original = Manager._allocate_output_buffer
        managers = [Manager() for _ in range(3)]
        other = Manager()
        inference = SimpleNamespace(OffloadMode=mode,
            torch=SimpleNamespace(empty=Mock(side_effect=RuntimeError("CUDA allocation failed"))))
        regressor = SimpleNamespace(model_=SimpleNamespace(modules=lambda: [
            SimpleNamespace(inference_mgr=m) for m in managers]))
        model._guard_gpu_managers(regressor, inference)
        self.assertIs(Manager._allocate_output_buffer, original)
        self.assertEqual(other._resolve_offload_mode()[0], mode.CPU)
        with self.assertRaisesRegex(RuntimeError, "fallback rejected"):
            managers[0]._resolve_offload_mode()
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            managers[0]._allocate_output_buffer(mode.GPU, (1,), "float32")
        with self.assertRaisesRegex(RuntimeError, "assigned GPU"):
            managers[0]._allocate_output_buffer(mode.CPU, (1,), "float32")

    def test_public_factory_enforces_pinned_configuration_before_import(self):
        with self.assertRaisesRegex(ValueError, "library_root"):
            model.public_regressor(dict(library_root="/tmp", regressor_parameters={}), "cpu")
        parameters = copy.deepcopy(model.REQUIRED_PARAMETERS)
        parameters["n_estimators"] = 2
        with patch.object(model.importlib, "import_module", side_effect=AssertionError("No import")):
            with self.assertRaisesRegex(ValueError, "n_estimators"):
                model.public_regressor(dict(library_root=str(model.R97_LIBRARY_ROOT),
                    regressor_parameters=parameters), "cuda:0")


if __name__ == "__main__":
    unittest.main()
