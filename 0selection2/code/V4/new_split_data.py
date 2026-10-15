"""CPU dataset reader for the locked new-only, B-filtered scenario.

Candidate pools come from the release manifest rather than the historical
seven-method B registry. Scenario aliases retain their source global IDs.
"""

import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import (
    GLOBAL_SOLVERS, P2I, S2I, constraint_bits, problem_descriptor,
)
from .r75_data import (
    RealTrainDataset, _CPUUnpickler, _check_split, _file_identity, _node_count,
    build_batch, dataset_sizes, lengths,
)


def scenario_solver_ids(manifest):
    aliases = manifest.get("solver_aliases", {})
    if any(alias in S2I or source not in S2I for alias, source in aliases.items()):
        raise ValueError("Scenario aliases must name existing solver deployments")
    names = manifest["global_solvers"]
    canonical = [aliases.get(name, name) for name in names]
    if (len(set(canonical)) != len(canonical) or any(name not in S2I for name in canonical)
            or canonical != sorted(canonical, key=S2I.get)):
        raise ValueError("Scenario solver identities must follow stable global IDs")
    if not aliases and names != GLOBAL_SOLVERS:
        raise ValueError("Global solver names/IDs differ from the released scenario")
    if manifest.get("model_global_solvers", GLOBAL_SOLVERS) != GLOBAL_SOLVERS:
        raise ValueError("The model's global solver ID space changed")
    identities = {name: S2I[source] for name, source in zip(names, canonical)}
    if manifest.get("global_solver_ids", identities) != identities:
        raise ValueError("Scenario global IDs disagree with their source deployments")
    return identities


class NewSplitDataset(RealTrainDataset):
    def __init__(self, problem, split, data_root, *, root=None):
        self.problem = str(problem).upper()
        self.split = "val" if str(split).lower() == "validation" else str(split).lower()
        _check_split(self.split, root)
        self.data_root = Path(data_root).expanduser().resolve()
        manifest_path = self.data_root / "scenario_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete" or manifest.get("format_version") != 1:
            raise ValueError("The new scenario release is not complete")
        identities = scenario_solver_ids(manifest)
        record = manifest["problems"][self.problem]
        self.pool = tuple(record["methods"])
        if (not self.pool or len(set(self.pool)) != len(self.pool)
                or any(name not in identities for name in self.pool)):
            raise ValueError("Invalid scenario candidate identities")
        self.pool_order = [identities[solver] for solver in self.pool]
        if self.pool_order != sorted(self.pool_order) or record["global_solver_ids"] != self.pool_order:
            raise ValueError("Scenario candidate order differs from its stable global IDs")
        self.mask = [int(index in self.pool_order) for index in range(len(GLOBAL_SOLVERS))]
        directory = self.data_root / (self.problem + self.split)
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        if (metadata["methods"] != list(self.pool)
                or metadata["global_solver_ids"] != self.pool_order):
            raise ValueError("Split cost-column identities disagree with the scenario")
        result_files = sorted((directory / "results").glob("result_*.txt"))
        if {p.name for p in result_files} != {"result_" + s + ".txt" for s in self.pool}:
            raise ValueError("Split result filenames disagree with the candidate pool")
        with (directory / "dataset.pkl").open("rb") as stream:
            self.instances = _CPUUnpickler(stream).load()
        with (directory / "raw_label.pkl").open("rb") as stream:
            raw = _CPUUnpickler(stream).load()
        self.base_N = self.N = self.size = len(self.instances)
        self.instance_ids = tuple(str(i) for i in range(self.size))
        if (self.size != metadata["instances"] or len(raw) != self.size
                or any(key not in raw for key in self.instance_ids)):
            raise ValueError("Dataset rows and contiguous label IDs disagree")
        self.labels = {key: dict(cost=raw[key]["cost"], ind=raw[key]["ind"])
                       for key in self.instance_ids}
        del raw
        self.K_p, self.coord_augment = len(self.pool), 0
        self.cbits = constraint_bits(self.problem)
        self.problem_desc = problem_descriptor(self.problem)
        self.pid = P2I[self.problem]
        self.raw_costs = np.asarray([self.labels[key]["cost"] for key in self.instance_ids],
                                   dtype=np.float64)
        self.raw_winner = np.asarray([self.labels[key]["ind"] for key in self.instance_ids],
                                    dtype=np.int64)
        if (self.raw_costs.shape != (self.size, self.K_p)
                or not np.isfinite(self.raw_costs).all()
                or not np.array_equal(self.raw_winner, self.raw_costs.argmin(axis=1))):
            raise ValueError("Invalid FP64 costs or recomputed winner")
        self.raw_costs.setflags(write=False)
        self.raw_winner.setflags(write=False)
        with np.load(directory / "input_metadata.npz", allow_pickle=False) as inputs:
            self.coord_dist = inputs["coord_dist"].astype(np.int64)
            self.source_indices = inputs["source_index"].astype(np.int64)
        if self.coord_dist.shape != (self.size,) or self.source_indices.shape != (self.size,):
            raise ValueError("Input metadata is not aligned with dataset rows")
        self.lengths = np.fromiter((_node_count(self.problem, x) for x in self.instances),
                                   dtype=np.int64, count=self.size)
        self.lengths.setflags(write=False)
        self.kind = "matrix" if self.problem == "ATSP" else "coord"
        self.d_in = None if self.kind == "matrix" else (
            2 if self.problem == "TSP" else 4 if self.problem == "CVRP" else 8)
        paths = [manifest_path, metadata_path, directory / "dataset.pkl",
                 directory / "raw_label.pkl", directory / "input_metadata.npz", *result_files]
        self.metadata = dict(
            problem=self.problem, problem_id=self.pid, split=self.split,
            data_root=str(self.data_root), rows=self.size,
            true_length_min=int(self.lengths.min()), true_length_max=int(self.lengths.max()),
            length_definition="model node count; includes depot for CVRP/MVRP",
            kind=self.kind, d_in=self.d_in, pool=list(self.pool), pool_ids=self.pool_order,
            raw_cost_dtype="float64", winner_source="recomputed FP64 winner in filtered candidate pool",
            coord_augment=0, coord_dist="actual source generation metadata; no index-based guess",
            solver_aliases={name: manifest.get("solver_aliases", {})[name]
                            for name in self.pool if name in manifest.get("solver_aliases", {})},
            scenario_id=manifest["scenario_id"], source_files=[_file_identity(p) for p in paths])

    def __getitem__(self, index):
        row = self._row_index(index)
        sample = super().__getitem__(row)
        sample["coord_dist"] = int(self.coord_dist[row])
        return sample


def load_dataset(problem, split="train", data_root=None, *, root=None):
    if data_root is None:
        raise ValueError("Specify the new scenario data_root explicitly")
    return NewSplitDataset(problem, split, data_root, root=root)
