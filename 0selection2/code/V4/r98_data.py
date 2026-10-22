"""Immutable FP64 historical-TRAIN performance targets for frozen R98 ICL.

``prepare_targets(data_root, context_root, output_root, problems=None)`` writes
PROBLEM/G.npy and identity.json, returning problem -> identity.json Path.
``load_targets(output_root, problem)`` returns a read-only table exposing
``.g``, ``.metadata`` and ``.close()``. No held-out labels or graphs are opened.
R97 supplies all TRAIN X/source IDs/kept columns; no rows/classes are sampled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
from io import BytesIO
import os
from pathlib import Path
import pickle
import shutil
import tempfile

import numpy as np

from .r97_data import (
    DEFAULT_DATA_ROOT, _atomic_json, _check_source_stamps, _close,
    _file_identity, _json_sha, _names, _positive_int, _problem_name,
    _read_json, _source_ids, _write_array, load_context,
)


SCHEMA = "r98_train_relative_gap_v1"
TARGET_DEFINITION = "FP64 g[i,m]=(c[i,m]-min_legal(c[i,:]))/min_legal(c[i,:])"
ROW_ORDER = "natural frozen TRAIN row order"


def _source_ids_sha256(source_ids):
    ids = _source_ids(source_ids)
    return hashlib.sha256(ids.astype("<i8", copy=False).tobytes()).hexdigest()


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            return lambda data: torch.load(BytesIO(data), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def relative_gaps(costs):
    """Use native FP64 subtraction/division, never reconstructed FP32 costs."""
    costs = np.asarray(costs)
    if (costs.dtype != np.float64 or costs.ndim != 2 or 0 in costs.shape
            or not np.isfinite(costs).all()):
        raise ValueError("Native costs must be a nonempty finite FP64 matrix")
    best = costs.min(axis=1, keepdims=True)
    if np.any(best <= 0):
        raise ValueError("Relative gap requires a strictly positive legal Oracle cost")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gaps = (costs - best) / best
    if not np.isfinite(gaps).all():
        raise ValueError("Nonfinite FP64 relative gaps")
    return gaps


def _validate_metadata(metadata, problem=None):
    identity = metadata["identity"]
    pool = _names(identity["pool"])
    ids = identity["pool_ids"]
    rows = _positive_int(identity["rows"], "TRAIN rows")
    if (metadata.get("complete") is not True or identity.get("schema") != SCHEMA
            or metadata["identity_sha256"] != _json_sha(identity)
            or (problem is not None and identity["problem"] != problem)
            or identity["label_splits"] != ["train"]
            or identity["row_order"] != ROW_ORDER
            or identity["target_definition"] != TARGET_DEFINITION
            or identity["target_transform"] != "none"
            or identity["context_subsampling"] is not False
            or identity["class_filter"] is not False
            or not isinstance(ids, list) or len(ids) != len(pool)
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Incomplete or changed R98 target identity/pool")
    if identity["data_root"] == str(DEFAULT_DATA_ROOT) and rows != 81000:
        raise ValueError("NEW90k targets must retain all81000 TRAIN rows")
    return identity


def validate_target_binding(context, metadata):
    """Bind target rows and every legal method to this exact R97 context."""
    identity = _validate_metadata(metadata)
    if (identity["context_metadata_sha256"] != _json_sha(context.metadata)
            or identity["context_identity_sha256"] != context.metadata["identity_sha256"]
            or identity["problem"] != context.metadata["identity"]["problem"]
            or identity["data_root"] != context.metadata["identity"]["data_root"]
            or identity["rows"] != len(context.x)
            or identity["pool"] != list(context.pool)
            or identity["pool_ids"] != list(context.pool_ids)
            or identity["source_ids_values_sha256"] != _source_ids_sha256(context.source_ids)
            or identity["kept_columns_sha256"] != _json_sha(context.kept_columns.tolist())):
        raise ValueError("Targets disagree with exact R97 context/row binding/legal pool")
    return identity


def _prepare_problem(data_root, context_root, output_root, problem):
    with load_context(context_root, problem) as context:
        rows, candidates = len(context.x), len(context.pool)
        old = context.metadata["identity"]
        if (old["data_root"] != str(data_root) or old["row_order"] != ROW_ORDER
                or old["context_subsampling"] is not False or old["class_quotas"] is not False):
            raise ValueError("R98 requires the full natural-order R97 TRAIN context")
        directory = data_root / f"{problem}train"
        paths = dict(context_metadata=context_root / problem / "identity.json",
                     native_metadata=directory / "metadata.json",
                     raw_label=directory / "raw_label.pkl",
                     train_input_metadata=directory / "input_metadata.npz")
        sources = {key: _file_identity(path) for key, path in paths.items()}
        native = _read_json(paths["native_metadata"])
        if (native["problem"] != problem or native["split"] != "train"
                or native["instances"] != rows or native["cost_dtype"] != "float64"
                or native["methods"] != list(context.pool)
                or native["global_solver_ids"] != list(context.pool_ids)
                or native["files"]["raw_label.pkl"]["sha256"] != sources["raw_label"]["sha256"]
                or native["files"]["input_metadata.npz"]["sha256"]
                != sources["train_input_metadata"]["sha256"]
                or sources["train_input_metadata"]["sha256"]
                != old["source_files"]["train_input_metadata"]["sha256"]):
            raise ValueError("Native TRAIN costs/pool/source hash differs from frozen context")
        with np.load(paths["train_input_metadata"], allow_pickle=False) as inputs:
            source_ids = _source_ids(inputs["source_index"])
        if not np.array_equal(source_ids, context.source_ids):
            raise ValueError("Native TRAIN source IDs do not match R97 row binding")
        identity = dict(
            schema=SCHEMA, problem=problem, data_root=str(data_root), rows=rows,
            pool=list(context.pool), pool_ids=list(context.pool_ids),
            label_splits=["train"], row_order=ROW_ORDER, target_definition=TARGET_DEFINITION,
            target_transform="none", context_subsampling=False, class_filter=False,
            context_metadata_sha256=_json_sha(context.metadata),
            context_identity_sha256=context.metadata["identity_sha256"],
            source_ids_values_sha256=_source_ids_sha256(source_ids),
            kept_columns_sha256=_json_sha(context.kept_columns.tolist()),
            raw_label_sha256=sources["raw_label"]["sha256"], source_files=sources)
        destination = output_root / problem
        if os.path.lexists(destination):
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("Preserve existing output; use a new target root")
            with load_targets(output_root, problem) as table:
                if table.metadata["identity"] != identity:
                    raise ValueError("Target identity changed; preserve old output and use a new root")
                validate_target_binding(context, table.metadata)
            _check_source_stamps(sources)
            return destination / "identity.json"
        with paths["raw_label"].open("rb") as stream:
            raw = _CPUUnpickler(stream).load()
        if (not isinstance(raw, Mapping) or len(raw) != rows
                or set(raw) != {str(i) for i in range(rows)}):
            raise ValueError("Native TRAIN labels must cover exactly every natural-order row")
        costs = np.empty((rows, candidates), dtype=np.float64)
        for i in range(rows):
            values = np.asarray(raw[str(i)]["cost"])
            if values.shape != (candidates,) or values.dtype != np.float64:
                raise ValueError("Native TRAIN cost row must contain the exact full FP64 legal pool")
            costs[i] = values
        del raw
        gaps = relative_gaps(costs)
        del costs
        stage = Path(tempfile.mkdtemp(prefix=f".{problem}.", dir=output_root))
        try:
            artifact = _write_array(stage / "G.npy", gaps.shape, np.float64, [(0, gaps)])
            _check_source_stamps(sources)
            _atomic_json(stage / "identity.json", dict(complete=True, identity=identity,
                         identity_sha256=_json_sha(identity), artifacts=dict(g=artifact)))
            if os.path.lexists(destination):
                raise ValueError("Output appeared during preparation; preserve it")
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return destination / "identity.json"


def prepare_targets(data_root, context_root, output_root, problems=None):
    """Prepare all requested TRAIN targets sequentially; never overwrite old data."""
    data_root, context_root, output_root = (Path(p).resolve() for p in
                                           (data_root, context_root, output_root))
    available = tuple(sorted(p.name for p in context_root.iterdir()
                            if p.is_dir() and not p.name.startswith(".")
                            and (p / "identity.json").is_file()))
    if isinstance(problems, (str, bytes)):
        raise ValueError("Problems must be a sequence of distinct TRAIN task names")
    problems = available if problems is None else tuple(problems)
    if (not problems or len(set(problems)) != len(problems)
            or any(_problem_name(p) not in available for p in problems)):
        raise ValueError("Problems must be distinct available frozen TRAIN tasks")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".r98_prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return {p: _prepare_problem(data_root, context_root, output_root, p) for p in problems}


@dataclass(frozen=True)
class TargetTable:
    g: np.ndarray
    metadata: dict

    @property
    def pool(self):
        return tuple(self.metadata["identity"]["pool"])

    def close(self):
        _close(self.g)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_targets(output_root, problem):
    """Validate and mmap saved targets without reopening any outcome source."""
    directory = Path(output_root) / _problem_name(problem)
    metadata = _read_json(directory / "identity.json")
    identity = _validate_metadata(metadata, problem)
    shape = (identity["rows"], len(identity["pool"]))
    spec = metadata["artifacts"]["g"]
    path = directory / "G.npy"
    if (spec["file"] != "G.npy" or spec["shape"] != list(shape)
            or spec["dtype"] != "float64" or path.stat().st_size != spec["size_bytes"]
            or _file_identity(path)["sha256"] != spec["sha256"]):
        raise ValueError("Target G schema/SHA mismatch; preserve existing output")
    g = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if (g.shape != shape or g.dtype != np.float64 or not np.isfinite(g).all()
                or np.any(g < 0) or np.any(g.min(axis=1) != 0)):
            raise ValueError("G must contain finite FP64 relative gaps for every legal candidate")
        return TargetTable(g=g, metadata=metadata)
    except BaseException:
        _close(g)
        raise
