"""CPU-only R97 historical-TRAIN context isolation. Importing performs no I/O.

Public API
----------
prepare_contexts(prepared_root, data_root, feature_root, output_root, *,
                 problems=None, chunk_rows=4096) -> dict[str, Path]
    Sequentially prepare ALL declared TRAIN rows for each requested problem.
    feature_root is R92's ``features`` or ``features/TRAIN_VAL`` directory.
    The R84 prepared.json cache_source locates TRAIN winners, NOT the possibly
    stale manifest-level cache_root. No graph, cost, held-out label, or TEST
    array is opened. Hashing the canonical TRAIN_VAL .npy includes VAL INPUT
    bytes, but fitting and output use only its leading TRAIN rows.
    Returns problem -> output_root/PROBLEM/identity.json. Existing outputs
    must match identity and artifact hashes; they are never rebuilt in place.
load_context(output_root, problem, *, verify_hashes=True) -> ContextTable
    Read-only mmap x (float32 N,C), y (int64 native local class IDs), and
    source_ids (int64 N). Also exposes kept_columns (int64 original indices),
    pool/pool_ids (tuples), and metadata (the JSON receipt).
FeatureTransform.fit(X_train, columns, *, chunk_rows=4096)
    Drop EXACTLY constant TRAIN columns, retaining original column order.
    No normalization, PCA, labels, held-out statistics, or feature selection.
table.transform_queries(X) -> float32 ndarray
    Raw-width matrix or dict of ALL original feature names -> 1-D columns.
    Extra keys, including supervision, are errors. This only transforms X;
    callers must also isolate IDs, preferably through for_queries below.
table.for_queries(X, *, source_ids, train_diagnostic=False, shuffle=False,
                  seed=2) -> QueryBatch
    Returns .x and .context_rows indexing this table. Default: reject ANY
    query/context base-ID overlap. Explicit TRAIN diagnostic: exclude ALL
    copies of ALL query base IDs (union exclusion for a batch). No sampling,
    class quotas, or default shuffle. Only shuffle=True uses a local seed.
exclude_context_rows(context_source_ids, query_source_ids) -> int64 ndarray
    Pure natural-order row indices after excluding every related base ID.
remap_local_labels(y, source_pool, target_pool) -> int64 ndarray
    Explicit full-pool permutation by name; never densify observed classes.

Output per problem: X.npy, y.npy, source_ids.npy, identity.json. Native FP64
``ind`` winners are COPIED, never re-argmin'ed from FP32 costs. For NEW90k,
N=81000 and raw width=357, checked against frozen prepared/feature receipts;
small alternate declared shapes are supported for tests. No full preparation
or runner is invoked here. Close tables, or use them as context managers.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping

import numpy as np


SCHEMA = "r97_train_context_v1"
DEFAULT_DATA_ROOT = Path(
    "/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018"
)
WINNER_RULE = "native FP64 ind; local index in frozen pool; no FP32 re-argmin"


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _json_sha(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed while hashing: {path}")
    return dict(path=str(path), size_bytes=after.st_size,
                mtime_ns=after.st_mtime_ns, sha256=digest.hexdigest())


def _atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _problem_name(problem):
    if not isinstance(problem, str) or not problem or not problem.isascii() or not (
            problem.replace("_", "").isalnum()):
        raise ValueError("Problem must be a simple ASCII name")
    return problem


def _names(values):
    if isinstance(values, (str, bytes)):
        raise ValueError("Expected a sequence of distinct names")
    values = tuple(values)
    if (not values or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("Names must be nonempty and distinct")
    return values


def _matrix(X, columns):
    if isinstance(X, Mapping):
        if set(X) != set(columns):
            raise ValueError("Named query input must contain exactly the feature columns")
        arrays = [np.asarray(X[name]) for name in columns]
        if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError("Named feature columns must be aligned 1-D arrays")
        X = np.column_stack(arrays)
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(columns) or X.dtype.kind not in "fiu":
        raise ValueError("Expected a real numeric matrix of the original feature width")
    return X


def _source_ids(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("Base source IDs must be a 1-D integer array")
    if values.size and (int(values.min()) < 0 or int(values.max()) > np.iinfo(np.int64).max):
        raise ValueError("Base source IDs must fit nonnegative int64")
    return values.astype(np.int64, copy=False)


def _labels(y, classes, rows=None):
    y = np.asarray(y)
    if (y.dtype != np.int64 or y.ndim != 1 or (rows is not None and len(y) != rows)
            or (y.size and (y.min() < 0 or y.max() >= classes))):
        raise ValueError("Winners must be aligned int64 LOCAL indices in the full pool")
    return y


def _close(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


@dataclass(frozen=True)
class FeatureTransform:
    columns: tuple
    kept_columns: tuple

    def __post_init__(self):
        columns = _names(self.columns)
        kept = tuple(self.kept_columns)
        if (not kept or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                            or i < 0 or i >= len(columns) for i in kept)
                or list(kept) != sorted(set(kept))):
            raise ValueError("Kept columns must be nonempty, unique, and in original order")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "kept_columns", tuple(int(i) for i in kept))

    @classmethod
    def fit(cls, X_train, columns, *, chunk_rows=4096):
        columns = _names(columns)
        X_train = _matrix(X_train, columns)
        chunk_rows = _positive_int(chunk_rows, "chunk_rows")
        if not len(X_train):
            raise ValueError("Cannot fit on empty TRAIN")
        varying = np.zeros(len(columns), dtype=bool)
        first = X_train[0]
        for start in range(0, len(X_train), chunk_rows):
            block = X_train[start:start + chunk_rows]
            if not np.isfinite(block).all():
                raise ValueError("Nonfinite TRAIN features")
            varying |= np.any(block != first, axis=0)
        return cls(columns, tuple(np.flatnonzero(varying)))

    def transform(self, X):
        """Input-only transform; no query labels, costs, winners, or fit update."""
        X = _matrix(X, self.columns)
        if not np.isfinite(X).all():
            raise ValueError("Nonfinite query features")
        with np.errstate(over="ignore", invalid="ignore"):
            reduced = np.asarray(X[:, self.kept_columns], dtype=np.float32, order="C")
        if not np.isfinite(reduced).all():
            raise ValueError("Features overflowed float32")
        return reduced

    def to_dict(self):
        kept = list(self.kept_columns)
        return dict(columns=list(self.columns), kept_columns=kept,
                    kept_columns_sha256=_json_sha(kept), fitted_on="TRAIN only",
                    operation="drop exactly constant columns; no other fitting")


def remap_local_labels(y, source_pool, target_pool):
    source_pool, target_pool = _names(source_pool), _names(target_pool)
    if set(source_pool) != set(target_pool):
        raise ValueError("Remapping requires identical full candidate sets")
    y = _labels(y, len(source_pool))
    target = {name: i for i, name in enumerate(target_pool)}
    return np.asarray([target[name] for name in source_pool], dtype=np.int64)[y]


def exclude_context_rows(context_source_ids, query_source_ids):
    """Return ALL non-related rows, in natural order, without changing inputs."""
    context_source_ids = _source_ids(context_source_ids)
    query_source_ids = _source_ids(query_source_ids)
    return np.flatnonzero(~np.isin(context_source_ids, query_source_ids))


@dataclass(frozen=True)
class QueryBatch:
    x: np.ndarray
    context_rows: np.ndarray
    excluded_context_rows: int
    train_diagnostic: bool
    shuffle_seed: int | None


@dataclass(frozen=True)
class ContextTable:
    x: np.ndarray
    y: np.ndarray
    source_ids: np.ndarray
    kept_columns: np.ndarray
    pool: tuple
    pool_ids: tuple
    metadata: dict
    transform: FeatureTransform

    def transform_queries(self, X):
        """Raw original-width X -> reduced float32 X, without supervision."""
        return self.transform.transform(X)

    def for_queries(self, X, *, source_ids, train_diagnostic=False, shuffle=False, seed=2):
        """Pair input transformation with mandatory base-ID isolation."""
        source_ids = _source_ids(source_ids)
        if not len(source_ids):
            raise ValueError("Queries must be nonempty")
        if not isinstance(train_diagnostic, bool) or not isinstance(shuffle, bool):
            raise ValueError("Diagnostic and shuffle switches must be explicit booleans")
        x = self.transform_queries(X)
        if len(x) != len(source_ids):
            raise ValueError("Query features and source IDs must align")
        rows = exclude_context_rows(self.source_ids, source_ids)
        excluded = len(self.y) - len(rows)
        if excluded and not train_diagnostic:
            raise ValueError("Query base IDs overlap TRAIN context; use explicit TRAIN diagnostic")
        if not len(rows):
            raise ValueError("No context remains after excluding all related query rows")
        if shuffle:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError("Shuffle seed must be a nonnegative integer")
            rows = np.random.default_rng(int(seed)).permutation(rows)
        rows.flags.writeable = False
        return QueryBatch(x, rows, excluded, train_diagnostic, int(seed) if shuffle else None)

    def close(self):
        for array in (self.x, self.y, self.source_ids):
            _close(array)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _pool(record):
    pool = _names(record["pool"])
    ids = record["pool_ids"]
    if (not isinstance(ids, list) or len(ids) != len(pool)
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Invalid pool/global ID mapping")
    global_names = _names(record["model_global_solvers"])
    aliases = record.get("scenario_solver_aliases", {})
    if any(i >= len(global_names) or global_names[i] != aliases.get(name, name)
           for name, i in zip(pool, ids)):
        raise ValueError("Pool order/global IDs/aliases disagree")
    return pool, tuple(ids)


def _check_source_stamps(sources):
    for item in sources.values():
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size_bytes"], item["mtime_ns"]):
            raise ValueError(f"Source changed during preparation: {item['path']}")


def _write_array(path, shape, dtype, blocks):
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    try:
        for start, block in blocks:
            array[start:start + len(block)] = block
        array.flush()
    finally:
        _close(array)
    result = _file_identity(path)
    return dict(file=path.name, shape=list(shape), dtype=np.dtype(dtype).name,
                size_bytes=result["size_bytes"], sha256=result["sha256"])


def _prepare_problem(data_root, feature_root, output_root, problem,
                     prepared, manifest, shared_sources, chunk_rows):
    train = manifest["splits"]["train"][problem]
    val = manifest["splits"]["val"][problem]
    ntrain = _positive_int(prepared["rows_per_problem"]["train"], "TRAIN rows")
    nval = _positive_int(prepared["rows_per_problem"]["val"], "VAL rows")
    if data_root == DEFAULT_DATA_ROOT and (ntrain, nval) != (81000, 4500):
        raise ValueError("NEW90k requires exactly TRAIN81000 followed by VAL4500")
    for split, record, rows in (("train", train, ntrain), ("val", val, nval)):
        if (record["problem"] != problem or record["split"] != split
                or record["data_root"] != str(data_root) or record["rows"] != rows
                or record.get("complete") is not True or record["rows_written"] != rows):
            raise ValueError(f"Incomplete or mismatched frozen {problem}/{split} schema")
    pool, pool_ids = _pool(train)
    if _pool(val) != (pool, pool_ids):
        raise ValueError("TRAIN/VAL candidate order differs")
    if train.get("raw_cost_dtype") != "float64" or "FP64" not in train.get("winner_source", ""):
        raise ValueError("TRAIN winner provenance must explicitly be native FP64")

    cache = Path(prepared["cache_source"]).resolve() / "train" / problem
    if Path(train["directory"]).resolve() != cache:
        raise ValueError("prepared cache_source and TRAIN directory disagree")
    cache_metadata = cache / "metadata.json"
    if _read_json(cache_metadata) != train:
        raise ValueError("Actual TRAIN cache metadata differs from frozen manifest")
    winner_spec = train["arrays"]["winner"]
    if winner_spec != dict(file="winner.npy", shape=[ntrain], dtype="int64"):
        raise ValueError("Unexpected TRAIN native-winner array schema")

    receipt_path = feature_root / f"{problem}.json"
    feature_receipt = _read_json(receipt_path)
    feature = feature_receipt["identity"]
    definition = feature["feature_definition"]
    columns = _names(definition["columns"])
    if data_root == DEFAULT_DATA_ROOT and len(columns) != 357:
        raise ValueError("NEW90k canonical feature width must be exactly 357")
    if (feature_receipt.get("complete") is not True
            or feature_receipt["dtype"] != "float32"
            or feature_receipt["shape"] != [ntrain + nval, len(columns)]
            or feature["splits"] != ["train", "val"]
            or feature["rows"] != dict(train=ntrain, val=nval)
            or feature["data_root"] != str(data_root) or feature["problem"] != problem
            or feature["preparation_sha256"] != shared_sources["prepared"]["sha256"]
            or feature["sources"] != dict(train=train, val=val)
            or definition.get("output_dtype") != "float32"
            or definition.get("train_fitted_feature_statistics") is not False):
        raise ValueError("Canonical TRAIN_VAL feature identity/schema mismatch")

    input_metadata = data_root / f"{problem}train" / "input_metadata.npz"
    recorded_input = [s for s in train["source_files"]
                      if s["path"] == str(input_metadata)]
    if len(recorded_input) != 1:
        raise ValueError("TRAIN source_index metadata is missing or ambiguous")
    paths = dict(feature_receipt=receipt_path, canonical_features=feature_root / f"{problem}.npy",
                 train_cache_metadata=cache_metadata, train_winners=cache / "winner.npy",
                 train_input_metadata=input_metadata)
    sources = dict(shared_sources, **{name: _file_identity(path) for name, path in paths.items()})
    if any(sources["train_input_metadata"][key] != recorded_input[0][key]
           for key in ("size_bytes", "mtime_ns")):
        raise ValueError("TRAIN input metadata changed since frozen preparation")

    arrays = []
    try:
        X = np.load(paths["canonical_features"], mmap_mode="r", allow_pickle=False)
        arrays.append(X)
        if X.shape != (ntrain + nval, len(columns)) or X.dtype != np.float32:
            raise ValueError("Canonical feature array shape/dtype mismatch")
        transform = FeatureTransform.fit(X[:ntrain], columns, chunk_rows=chunk_rows)
        y = np.load(paths["train_winners"], mmap_mode="r", allow_pickle=False)
        arrays.append(y)
        _labels(y, len(pool), ntrain)
        with np.load(input_metadata, allow_pickle=False) as inputs:
            source_ids = _source_ids(inputs["source_index"])
        if len(source_ids) != ntrain:
            raise ValueError("TRAIN source IDs do not cover exactly all TRAIN rows")
        identity = dict(schema=SCHEMA, problem=problem, data_root=str(data_root),
                        rows=ntrain, seed=prepared["seed"], source_files=sources,
                        feature_definition=definition,
                        feature_source_sha256=feature["feature_source_sha256"],
                        transform=transform.to_dict(), pool=list(pool), pool_ids=list(pool_ids),
                        winner_rule=WINNER_RULE, winner_source=train["winner_source"],
                        label_splits=["train"], row_order="natural frozen TRAIN row order",
                        context_policy="reject overlapping base IDs; explicit TRAIN diagnostic excludes ALL related rows",
                        context_subsampling=False, class_quotas=False)
        destination = output_root / problem
        if os.path.lexists(destination):
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError("Existing output is not a plain context directory; preserve it")
            existing = _read_json(destination / "identity.json")
            if existing.get("identity") != identity:
                raise ValueError("Context identity changed; preserve existing output, use a new root")
            with load_context(output_root, problem):
                pass
            _check_source_stamps(sources)
            return destination / "identity.json"

        stage = Path(tempfile.mkdtemp(prefix=f".{problem}.", dir=output_root))
        try:
            def blocks(values):
                return ((i, values[i:i + chunk_rows]) for i in range(0, ntrain, chunk_rows))

            artifacts = dict(
                x=_write_array(stage / "X.npy", (ntrain, len(transform.kept_columns)), np.float32,
                               ((i, transform.transform(block)) for i, block in blocks(X[:ntrain]))),
                y=_write_array(stage / "y.npy", (ntrain,), np.int64, blocks(y)),
                source_ids=_write_array(stage / "source_ids.npy", (ntrain,), np.int64, blocks(source_ids)))
            _check_source_stamps(sources)
            _atomic_json(stage / "identity.json", dict(complete=True, identity=identity,
                         identity_sha256=_json_sha(identity), artifacts=artifacts))
            if os.path.lexists(destination):
                raise ValueError("Output appeared during preparation; preserve it")
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return destination / "identity.json"
    finally:
        for array in arrays:
            _close(array)


def prepare_contexts(prepared_root, data_root, feature_root, output_root, *,
                     problems=None, chunk_rows=4096):
    """Prepare sequentially; see module docstring for paths and safety contract."""
    prepared_root, data_root, feature_root, output_root = (
        Path(p).resolve() for p in (prepared_root, data_root, feature_root, output_root))
    if feature_root.name != "TRAIN_VAL":
        feature_root = feature_root / "TRAIN_VAL"
    chunk_rows = _positive_int(chunk_rows, "chunk_rows")
    prepared_path = prepared_root / "prepared.json"
    manifest_path = prepared_root / "source_cache_manifest.json"
    shared_sources = dict(prepared=_file_identity(prepared_path),
                          source_manifest=_file_identity(manifest_path),
                          helper=_file_identity(Path(__file__)))
    prepared, manifest = _read_json(prepared_path), _read_json(manifest_path)
    if (prepared["data_root"] != str(data_root) or manifest["data_root"] != str(data_root)
            or prepared["seed"] != 2 or manifest.get("complete") is not True):
        raise ValueError("Preparation/release identity must match frozen seed2 TRAIN")
    available = tuple(manifest["splits"]["train"])
    if not available or set(available) != set(manifest["splits"]["val"]):
        raise ValueError("TRAIN and VAL problem schemas must match")
    problems = available if problems is None else tuple(problems)
    if (not problems or len(set(problems)) != len(problems)
            or any(_problem_name(p) not in available for p in problems)):
        raise ValueError("Requested problems must be distinct declared TRAIN problems")
    # A scoped cooperative lock also prevents two R97 preparers replacing outputs.
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".r97_prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return {problem: _prepare_problem(data_root, feature_root, output_root,
                                          problem, prepared, manifest, shared_sources, chunk_rows)
                for problem in problems}


def load_context(output_root, problem, *, verify_hashes=True):
    """Load a complete context as validated read-only mmaps, without source I/O."""
    directory = Path(output_root) / _problem_name(problem)
    metadata = _read_json(directory / "identity.json")
    identity = metadata["identity"]
    if (metadata.get("complete") is not True or identity.get("schema") != SCHEMA
            or identity["problem"] != problem
            or metadata["identity_sha256"] != _json_sha(identity)
            or identity["winner_rule"] != WINNER_RULE or identity["label_splits"] != ["train"]):
        raise ValueError("Incomplete or changed R97 context identity")
    transform_meta = identity["transform"]
    transform = FeatureTransform(tuple(transform_meta["columns"]), tuple(transform_meta["kept_columns"]))
    if transform_meta != transform.to_dict():
        raise ValueError("Changed TRAIN-only feature transformation/kept-column SHA")
    pool = _names(identity["pool"])
    pool_ids = tuple(identity["pool_ids"])
    if (len(pool_ids) != len(pool) or len(set(pool_ids)) != len(pool_ids)
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in pool_ids)):
        raise ValueError("Invalid saved pool/global ID mapping")
    rows = _positive_int(identity["rows"], "context rows")
    if (identity["data_root"] == str(DEFAULT_DATA_ROOT)
            and (rows != 81000 or len(transform.columns) != 357)):
        raise ValueError("NEW90k contexts must contain all81000 TRAIN rows from raw357 features")
    specs = dict(x=("X.npy", (rows, len(transform.kept_columns)), np.dtype("float32")),
                 y=("y.npy", (rows,), np.dtype("int64")),
                 source_ids=("source_ids.npy", (rows,), np.dtype("int64")))
    arrays = {}
    try:
        for key, (filename, shape, dtype) in specs.items():
            spec = metadata["artifacts"][key]
            path = directory / filename
            if (spec["file"] != filename or spec["shape"] != list(shape)
                    or spec["dtype"] != dtype.name or path.stat().st_size != spec["size_bytes"]):
                raise ValueError("Context artifact schema/size mismatch")
            if verify_hashes and _file_identity(path)["sha256"] != spec["sha256"]:
                raise ValueError("Context artifact SHA mismatch; preserve existing output")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays[key] = array
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("Context array shape/dtype mismatch")
        for start in range(0, rows, 4096):
            if not np.isfinite(arrays["x"][start:start + 4096]).all():
                raise ValueError("Nonfinite prepared context features")
        _labels(arrays["y"], len(pool), rows)
        _source_ids(arrays["source_ids"])
        kept = np.asarray(transform.kept_columns, dtype=np.int64)
        kept.flags.writeable = False
        return ContextTable(**arrays, kept_columns=kept, pool=pool, pool_ids=pool_ids,
                            metadata=metadata, transform=transform)
    except BaseException:
        for array in arrays.values():
            _close(array)
        raise
