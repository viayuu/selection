"""Frozen scalar TabICL performance contexts, not a learned selector.

Only historical TRAIN X and raw relative GAP targets reach ``fit``. Query
prediction accepts native input statistics and base IDs only. Each legal
method is fitted/predicted/released sequentially, keeping one regressor live.
There is no SGD, nonlinear target transform, clipping or post-hoc calibrator.
"""

import gc
import importlib
from pathlib import Path
import sys
from types import MethodType

import numpy as np

from .r97_data import _file_identity, _source_ids
from .r98_data import validate_target_binding


HF_REPO_ID = "jingang/TabICL"
HF_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
WEIGHT_NAME = "tabicl-regressor-v2-20260212.ckpt"
WEIGHT_SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
R97_LIBRARY_ROOT = (Path(__file__).resolve().parent / "runs" /
                    "R97_tabular_in_context" / "runtime" / "python")
REQUIRED_PARAMETERS = dict(n_estimators=1, norm_methods=["none"], batch_size=1,
                           use_amp=True, use_fa3=False, offload_mode="gpu", random_state=2)


def _guard_gpu_managers(regressor, inference):
    """Guard only this model's managers, without mutating shared library classes."""
    count = 0
    for component in regressor.model_.modules():
        manager = getattr(component, "inference_mgr", None)
        if manager is None or getattr(manager, "_r98_gpu_only", False):
            continue
        original = manager._resolve_offload_mode

        def resolve(self, *args, _original=original, **kwargs):
            mode, reason = _original(*args, **kwargs)
            if mode != inference.OffloadMode.GPU:
                raise RuntimeError("R98 CPU/disk fallback rejected before allocation: " + str(reason))
            return mode, reason

        def allocate(self, mode, shape, dtype):
            if mode != inference.OffloadMode.GPU or self.exe_device.type != "cuda":
                raise RuntimeError("R98 output allocation must stay on the assigned GPU")
            # Upstream catches failed GPU allocation and falls back to CPU.
            output = inference.torch.empty(shape, dtype=dtype, device=self.exe_device)
            return output, dict(mode=mode.name, shape=shape, dtype=str(dtype))

        manager._resolve_offload_mode = MethodType(resolve, manager)
        manager._allocate_output_buffer = MethodType(allocate, manager)
        manager._r98_gpu_only = True
        count += 1
    if count != 3:
        raise RuntimeError("Pinned TabICL must expose exactly three inference managers")


def public_regressor(config, device):
    """Late import from R97's isolated runtime; no pip or automatic download.

    ``config`` keys: ``library_root`` and ``regressor_parameters`` (constructor
    kwargs, including model_path/checkpoint_version/allow_auto_download=False).
    KV cache is a fit argument, enforced by FrozenPerformanceContext instead.
    """
    directory = Path(config["library_root"]).resolve()
    if directory != R97_LIBRARY_ROOT or not directory.is_dir():
        raise ValueError("Regressor must reuse the immutable isolated R97 library_root")
    parameters = dict(config["regressor_parameters"])
    for key, expected in REQUIRED_PARAMETERS.items():
        if key not in parameters or parameters[key] != expected:
            raise ValueError(f"Frozen regressor parameter mismatch: {key}")
    if (parameters.get("allow_auto_download") is not False
            or parameters.get("checkpoint_version") != WEIGHT_NAME
            or config.get("kv_cache", False) is not False):
        raise ValueError("Use the pinned offline regressor checkpoint and kv_cache=False")
    path = Path(parameters["model_path"]).resolve(strict=True)
    expected_path = (Path(__file__).resolve().parent / "runs" /
                     "R98_performance_context" / "runtime" / "weights" / WEIGHT_NAME)
    if path != expected_path or _file_identity(path)["sha256"] != WEIGHT_SHA256:
        raise ValueError("Public R98 regressor checkpoint path/SHA256 mismatch")
    if parameters.get("device") is not None and str(parameters["device"]) != str(device):
        raise ValueError("Regressor must use the launcher-assigned device")
    stages = parameters.get("inference_config", {})
    if (set(stages) != {"COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"}
            or any(stage.get("offload") != "gpu"
                   or stage.get("use_amp", True) is not True
                   or stage.get("use_fa3", False) is not False
                   or (stage.get("device") is not None and str(stage["device"]) != str(device))
                   for stage in stages.values())):
        raise ValueError("R98 requires GPU-only stage configs with fixed AMP/FA3/device")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    module = importlib.import_module("tabicl")
    if not Path(module.__file__).resolve().is_relative_to(directory):
        raise RuntimeError("TabICL was imported outside the locked R97 runtime")
    inference = importlib.import_module("tabicl.model.inference")

    class GpuOnlyRegressor(module.TabICLRegressor):
        def fit(self, X, y, kv_cache=False):
            if kv_cache is not False:
                raise ValueError("R98 requires kv_cache=False")
            super().fit(X, y, kv_cache=False)
            _guard_gpu_managers(self, inference)
            return self

    parameters["device"] = device
    return GpuOnlyRegressor(**parameters)


class FrozenPerformanceContext:
    def __init__(self, context, targets, regressor_factory):
        self.context = context
        self.targets = targets
        self.regressor_factory = regressor_factory
        identity = validate_target_binding(context, targets.metadata)
        x, g = np.asarray(context.x), np.asarray(targets.g)
        if (x.ndim != 2 or x.dtype != np.float32 or not len(x)
                or x.shape[1] != len(context.kept_columns)
                or not np.isfinite(x).all()
                or len(context.source_ids) != len(x)
                or g.dtype != np.float64 or g.shape != (len(x), len(context.pool))
                or not np.isfinite(g).all() or np.any(g < 0)
                or np.any(g.min(axis=1) != 0)
                or len(x) != identity["rows"] or not callable(regressor_factory)):
            raise ValueError("Historical X/G must preserve every TRAIN row and legal candidate")

    def fit(self):
        """Compatibility no-op: real scalar context fitting is lazy in predict."""
        return self

    def _predict_column(self, features, column):
        target = np.asarray(self.targets.g[:, column])
        # Equality is exact: rare/nonwinning or nearly constant methods stay intact.
        if np.all(target == target[0]):
            return np.full(len(features), target[0], dtype=np.float64)
        regressor = self.regressor_factory()
        try:
            regressor.fit(np.array(self.context.x, copy=True),
                          np.array(target, dtype=np.float64, copy=True), kv_cache=False)
            prediction = np.asarray(regressor.predict(features, output_type="mean"))
            if (prediction.shape != (len(features),) or prediction.dtype.kind not in "fiu"
                    or not np.isfinite(prediction).all()):
                raise ValueError("Regressor mean output must be finite with shape (query_rows,)")
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.array(prediction, dtype=np.float64, copy=True)
            if not np.isfinite(result).all():
                raise ValueError("Regressor mean output overflowed finite FP64")
            return result
        finally:
            del regressor
            gc.collect()

    def predict_features(self, query_features, query_source_ids):
        features = np.asarray(query_features)
        source_ids = _source_ids(query_source_ids)
        if (features.ndim != 2 or features.dtype.kind not in "fiu" or not len(features)
                or features.shape[1] != len(self.context.kept_columns)
                or len(source_ids) != len(features) or not np.isfinite(features).all()):
            raise ValueError("Query inputs must match the R97 TRAIN-fitted feature transform")
        if np.intersect1d(source_ids, self.context.source_ids).size:
            raise ValueError("Query/base-instance outcomes would be present in its own context")
        means = np.empty((len(features), len(self.context.pool)), dtype=np.float64)
        for column in range(len(self.context.pool)):
            means[:, column] = self._predict_column(features, column)
        return means

    def predict(self, query_features, query_source_ids):
        """Minimize raw expected GAP; exact ties use the first frozen pool slot."""
        return self.predict_features(query_features, query_source_ids).argmin(axis=1)
