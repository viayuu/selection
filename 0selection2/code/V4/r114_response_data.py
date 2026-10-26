"""TRAIN-only paired responses. Preparation is CPU-only; dispatch defaults dry."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import importlib.metadata
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[3]
DATA_ROOT = WORKSPACE / "selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018"
BRIDGE = WORKSPACE / "easynco_v3_bridge"
LEGACY_ROOT = BRIDGE / "exports/train_100k_20261002"
PREPARATION = Path(__file__).resolve().parent / "runs/R114_paired_responses/preparation"
PROBLEMS = ("TSP", "CVRP", "ATSP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
            "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW",
            "OVRPLTW", "VRPBLTW", "OVRPBLTW")
ELIGIBLE = tuple(p for p in PROBLEMS if p not in ("TSP", "ATSP"))
VERSION = "r114_train_paired_v1_seed2"
ADAPTER_ENTRY = "train_100k_sampling:make_confirmed_backend"
INTERVENTION = dict(proposals=256, disjoint_swaps=2, control_l1_relative_tolerance=0.25,
                    demand_geometry="first spatial PCA axis, positive x orientation",
                    tw_geometry="Euclidean distance to depot", b_roles="sign fixed at each node")
WINNER_POLICY = "native reported cost promoted to FP64; exact ties lowest stable global ID"


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_for(*parts):
    return int(canonical_hash(parts)[:8], 16)


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_new_json(path, value):
    """Idempotent immutable writes; never silently replace a data version."""
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Immutable artifact changed; choose a new version: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(content)


def load_pool(data_root, problem):
    meta = read_json(Path(data_root) / f"{problem}train/metadata.json")
    methods, ids = meta["methods"], meta["global_solver_ids"]
    if len(set(methods)) != len(methods) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate identity")
    if "RELD_MTL" in methods or "RELD_MOEL" in methods:
        raise ValueError("Frozen release must use the single reld_mt alias")
    if "B" in problem and set(methods) & {"RouteFinder", "MoSES_RF", "MoSES_CaDA"}:
        raise ValueError("Deleted B candidates must not reappear")
    order = sorted(zip(ids, methods))
    return [dict(method=m, global_id=int(i), source_method="RELD_MOEL" if m == "reld_mt" else m)
            for i, m in order]


def read_mapping(data_root, problem, split):
    directory = Path(data_root) / f"{problem}{split}"
    meta = read_json(directory / "metadata.json")
    path = directory / "index_mapping.csv"
    if file_hash(path) != meta["files"][path.name]["sha256"]:
        raise ValueError(f"Frozen membership changed: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != meta["instances"]:
        raise ValueError("Frozen membership count changed")
    if [int(r["local_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Non-contiguous frozen local indices")
    if len({int(r["source_index"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate source/base IDs in split")
    return rows, meta


def freeze_selection(data_root, version=VERSION, count=32, seed=2):
    if count < 1:
        raise ValueError("Base count must be positive")
    root = Path(data_root).resolve()
    scenario = read_json(root / "scenario_manifest.json")
    selected, sources, pools = [], {}, {}
    for problem in PROBLEMS:
        pools[problem] = load_pool(root, problem)
        if problem not in ELIGIBLE:
            continue
        rows, meta = read_mapping(root, problem, "train")
        heldout = set()
        for split in ("val", "test"):
            mapping, _ = read_mapping(root, problem, split)
            heldout.update(int(r["source_index"]) for r in mapping)
        if heldout & {int(r["source_index"]) for r in rows}:
            raise ValueError(f"TRAIN/holdout base overlap in {problem}")
        if count > len(rows):
            raise ValueError("Requested more bases than TRAIN membership")
        strata = defaultdict(list)
        for row in rows:
            strata[(row["distribution"], row["capacity_type"], int(row["scale"]) // 25)].append(row)
        rng = np.random.default_rng(seed_for(seed, problem, "base_selection"))
        keys = sorted(strata)
        rng.shuffle(keys)
        for group in strata.values():
            rng.shuffle(group)
        chosen = []
        while len(chosen) < count:
            for key in keys:
                if strata[key]:
                    chosen.append(strata[key].pop())
                    if len(chosen) == count:
                        break
        for slot, row in enumerate(chosen):
            source_index = int(row["source_index"])
            base_id = f"{scenario['scenario_id']}:{problem}:source{source_index}"
            selected.append(dict(problem=problem, split="train", slot=slot,
                local_index=int(row["local_index"]), source_index=source_index, base_id=base_id,
                source_base_id=f"{scenario['source_data']}:{problem}:{source_index}",
                scale=int(row["scale"]), distribution=row["distribution"],
                capacity_type=row["capacity_type"]))
        sources[problem] = dict(dataset=str(root / f"{problem}train/dataset.pkl"),
            dataset_sha256=meta["files"]["dataset.pkl"]["sha256"],
            mapping_sha256=meta["files"]["index_mapping.csv"]["sha256"],
            metadata_sha256=file_hash(root / f"{problem}train/metadata.json"),
            holdout_base_exclusion_checked=True)
    selection = dict(schema_version=1, version=version, seed=seed, split="train", data_root=str(root),
        scenario_sha256=file_hash(root / "scenario_manifest.json"), source_scenario=scenario["scenario_id"],
        base_count_per_problem=count, base_selection="input-only strata round-robin, immutable order",
        intervention=INTERVENTION, winner_policy=WINNER_POLICY, sources=sources, pools=pools,
        not_applicable={p: "No demand or TW attribute; do not invent either" for p in ("TSP", "ATSP")},
        bases=selected, original_costs_read=False, solver_execution=False)
    selection["selection_sha256"] = canonical_hash(selection)
    return selection


def verify_selection(selection):
    payload = dict(selection)
    recorded = payload.pop("selection_sha256")
    if canonical_hash(payload) != recorded or selection["split"] != "train":
        raise ValueError("Selection is changed or not TRAIN")


def fields(problem, item):
    if problem not in ELIGIBLE:
        raise ValueError("Requested intervention is not applicable to this problem")
    item = plain(item)
    if problem == "CVRP":
        depot = np.asarray(item["depot"], dtype=np.float64)
        xy = np.asarray(item["loc"], dtype=np.float64)
        q = np.asarray(item["demand"], dtype=np.float64)
        if depot.shape != (1, 1, 2) or xy.ndim != 3 or xy.shape[0] != 1 or q.shape != xy.shape[:2]:
            raise ValueError("Invalid native CVRP shape")
        return dict(xy=xy[0], depot=depot[0, 0], q=q[0], capacity=1.0)
    return dict(xy=np.asarray(item["node_xy"], dtype=np.float64),
                depot=np.asarray(item["depot_xy"], dtype=np.float64).reshape(-1),
                q=np.asarray(item["node_demand"], dtype=np.float64), capacity=float(item["capacity"]))


def validate_input(problem, item):
    """Independent constructive legality check, not a solver/label call."""
    item = plain(item)
    f = fields(problem, item)
    xy, depot, q = f["xy"], f["depot"], f["q"]
    n = len(q)
    required = {"loc", "depot", "demand"} if problem == "CVRP" else {
        "node_xy", "depot_xy", "node_demand", "capacity"}
    if "L" in problem:
        required.add("route_limit")
    if "TW" in problem:
        required.update(("tw_start", "tw_end", "service_time"))
    if set(item) != required or n < 1 or xy.shape != (n, 2) or depot.shape != (2,):
        raise ValueError("Unexpected native fields or coordinate/demand shape")
    if not all(np.isfinite(a).all() for a in (xy, depot, q)) or f["capacity"] != 1.0:
        raise ValueError("Nonfinite input or changed capacity")
    if (q == 0).any() or (np.abs(q) > 1.0 + 1e-7).any() or ("B" not in problem and (q < 0).any()):
        raise ValueError("Invalid demand under existing signed-load semantics")
    distance = np.linalg.norm(xy - depot, axis=1)
    opened = problem.startswith("OVRP")
    if "L" in problem:
        if item["route_limit"] != 3.0 or ((1 if opened else 2) * distance > 3.0 + 1e-5).any():
            raise ValueError("No singleton route-length feasibility certificate")
    if "TW" in problem:
        start, end, service = (np.asarray(item[k], dtype=np.float64)
                               for k in ("tw_start", "tw_end", "service_time"))
        if any(a.shape != (n,) or not np.isfinite(a).all() for a in (start, end, service)):
            raise ValueError("Invalid TW/service shape or values")
        if (start < 0).any() or (end > 3.0 + 1e-7).any() or (start > end).any() or (service < 0).any():
            raise ValueError("Invalid TW bounds or service duration")
        begin = np.maximum(distance, start)
        if (begin > end + 1e-5).any():
            raise ValueError("Unreachable singleton time window")
        if not opened and (begin + service + distance > 3.0 + 1e-5).any():
            raise ValueError("Singleton misses closed depot deadline=3")
    return dict(legal=True, certificate="singleton routes; all positive-demand nodes before negative-demand nodes",
                n=n, b_semantics="existing signed-load; no classical precedence added",
                depot_deadline=3.0 if "TW" in problem else None)


def apply_permutation(problem, item, family, permutation):
    changed = copy.deepcopy(plain(item))
    p = np.asarray(permutation, dtype=np.int64)
    if family == "demand":
        q = fields(problem, item)["q"][p]
        changed["demand" if problem == "CVRP" else "node_demand"] = (
            q[None, :].tolist() if problem == "CVRP" else q.tolist())
    elif family == "tw_center" and "TW" in problem:
        start, end = (np.asarray(item[k], dtype=np.float64) for k in ("tw_start", "tw_end"))
        centers, half_width = (start + end) / 2, (end - start) / 2
        changed["tw_start"] = (centers[p] - half_width).tolist()
        changed["tw_end"] = (centers[p] + half_width).tolist()
    else:
        raise ValueError("Intervention family is not applicable")
    return changed


def validate_intervention(problem, original, changed, family):
    validate_input(problem, original)
    certificate = validate_input(problem, changed)
    a, b = plain(original), plain(changed)
    fa, fb = fields(problem, a), fields(problem, b)
    allowed = {"demand" if problem == "CVRP" else "node_demand"} if family == "demand" else {"tw_start", "tw_end"}
    if family not in ("demand", "tw_center") or (family == "tw_center" and "TW" not in problem):
        raise ValueError("Intervention family is not applicable")
    for key in set(a) - allowed:
        if a[key] != b[key]:
            raise ValueError(f"Unrelated field changed: {key}")
    if family == "demand":
        if not np.array_equal(np.sort(fa["q"]), np.sort(fb["q"])):
            raise ValueError("Demand multiset changed")
        if "B" in problem and not np.array_equal(np.sign(fa["q"]), np.sign(fb["q"])):
            raise ValueError("B node role/sign changed")
    else:
        sa, ea = (np.asarray(a[k]) for k in ("tw_start", "tw_end"))
        sb, eb = (np.asarray(b[k]) for k in ("tw_start", "tw_end"))
        if not np.allclose(ea - sa, eb - sb, rtol=0, atol=1e-12):
            raise ValueError("Per-node window widths changed")
        if not np.allclose(np.sort(sa + ea), np.sort(sb + eb), rtol=0, atol=1e-12):
            raise ValueError("TW center multiset changed")
    if canonical_hash(a) == canonical_hash(b):
        raise ValueError("No-op intervention")
    return certificate


def intervention_pair(problem, item, family, seed):
    """Select a directed change and amplitude-matched random legal control."""
    item = plain(item)
    validate_input(problem, item)
    f = fields(problem, item)
    n = len(f["q"])
    if family == "demand":
        values = np.abs(f["q"])
        centered = f["xy"] - f["xy"].mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        axis = vectors[0]
        if axis[np.flatnonzero(np.abs(axis) > 1e-12)[0]] < 0:
            axis = -axis
        geometry = centered @ axis
    elif family == "tw_center" and "TW" in problem:
        values = (np.asarray(item["tw_start"]) + np.asarray(item["tw_end"])) / 2
        geometry = np.linalg.norm(f["xy"] - f["depot"], axis=1)
    else:
        raise ValueError("Intervention family is not applicable")
    legal_swaps = []
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] == values[j]:
                continue
            if family == "demand" and np.sign(f["q"][i]) != np.sign(f["q"][j]):
                continue
            if family == "tw_center":
                # Check affected recipients without a solver or geometric repair.
                start, end = np.asarray(item["tw_start"]), np.asarray(item["tw_end"])
                half = (end - start)[[i, j]] / 2
                centers = values[[j, i]]
                lo, hi = centers - half, centers + half
                dist = geometry[[i, j]]
                begin = np.maximum(dist, lo)
                service = np.asarray(item["service_time"])[[i, j]]
                if (lo < 0).any() or (hi > 3).any() or (begin > hi + 1e-5).any():
                    continue
                if not problem.startswith("OVRP") and (begin + service + dist > 3 + 1e-5).any():
                    continue
            legal_swaps.append((i, j))
    if len(legal_swaps) < 2:
        return dict(status="rejected", reason="fewer than two legal swaps")
    rng = np.random.default_rng(seed)
    proposals = {}
    for _ in range(INTERVENTION["proposals"]):
        first = legal_swaps[int(rng.integers(len(legal_swaps)))]
        second = None
        for _ in range(32):
            candidate = legal_swaps[int(rng.integers(len(legal_swaps)))]
            if not set(first) & set(candidate):
                second = candidate
                break
        if second is None:
            continue
        p = np.arange(n)
        for i, j in (first, second):
            p[i], p[j] = p[j], p[i]
        key = tuple(p.tolist())
        delta = values[p] - values
        proposals[key] = dict(permutation=p.tolist(), l1=float(np.abs(delta).sum()),
                              geometry_score=float(np.dot(geometry - geometry.mean(), delta)),
                              moved_customers=4)
    candidates = sorted(proposals.values(), key=lambda r: (-r["geometry_score"], r["permutation"]))
    for directed in candidates:
        if directed["geometry_score"] <= 0:
            break
        controls = [r for r in proposals.values() if r != directed and
                    not np.array_equal(values[r["permutation"]], values[directed["permutation"]]) and
                    abs(r["l1"] / directed["l1"] - 1) <= INTERVENTION["control_l1_relative_tolerance"]]
        if not controls:
            continue
        control_rng = np.random.default_rng(seed_for(seed, "random_control"))
        control = controls[int(control_rng.integers(len(controls)))]
        result = dict(status="ready", family=family, proposal_count=len(proposals), seed=seed)
        for arm, record in (("directed", directed), ("random", control)):
            changed = apply_permutation(problem, item, family, record["permutation"])
            check = validate_intervention(problem, item, changed, family)
            result[arm] = dict(record, item=changed, input_check=check)
        return result
    return dict(status="rejected", reason="no positive directed change with distinct matched control",
                proposal_count=len(proposals), seed=seed)


def check_route(problem, item, raw_tour, reported_cost):
    """Audit a deployment result without importing its environment or masks."""
    validate_input(problem, item)
    f = fields(problem, item)
    n = len(f["q"])
    raw = np.asarray(raw_tour)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("Noninteger or nonflat route")
    tour = raw.astype(np.int64).tolist()
    if any(x < 0 or x > n for x in tour) or sorted(x for x in tour if x) != list(range(1, n + 1)):
        raise ValueError("Route does not serve every customer exactly once")
    if not tour or tour[0] != 0:
        tour.insert(0, 0)
    if tour[-1] != 0:
        tour.append(0)
    boundaries = [i for i, v in enumerate(tour) if v == 0]
    xy = np.concatenate((f["depot"][None], f["xy"]))
    remaining = set((np.flatnonzero(f["q"] > 0) + 1).tolist())
    total = 0.0
    opened = problem.startswith("OVRP")
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        route = tour[left:right + 1]
        if len(route) < 3:
            continue
        load = 1.0 if remaining or "B" not in problem else 0.0
        clock = distance = 0.0
        previous = 0
        for node in route[1:-1]:
            edge = float(np.linalg.norm(xy[node] - xy[previous]))
            distance += edge
            load -= f["q"][node - 1]
            if load < -1e-5 or load > 1.0 + 1e-5:
                raise ValueError("Signed-load capacity violation")
            remaining.discard(node)
            if "TW" in problem:
                clock = max(clock + edge, item["tw_start"][node - 1])
                if clock > item["tw_end"][node - 1] + 1e-5:
                    raise ValueError("Service-start window violation")
                clock += item["service_time"][node - 1]
            previous = node
        if not opened:
            distance += float(np.linalg.norm(xy[previous] - xy[0]))
            if "TW" in problem and clock + np.linalg.norm(xy[previous] - xy[0]) > 3.0 + 1e-5:
                raise ValueError("Closed depot deadline violation")
        if "L" in problem and distance > item["route_limit"] + 1e-5:
            raise ValueError("Route-length limit violation")
        total += distance
    reported = float(reported_cost)
    if not np.isfinite(reported) or reported <= 0 or abs(total - reported) > 2e-4 + 2e-5 * abs(total):
        raise ValueError("Reported native cost disagrees with independently recomputed route")
    return dict(verified=True, native_cost=reported, recomputed_cost_fp64=total,
                absolute_error=abs(total - reported))


def prepare_inputs(selection, output, problems=None, limit=2):
    verify_selection(selection)
    if limit < 1 or limit > selection["base_count_per_problem"]:
        raise ValueError("Preparation limit must be within the pre-fixed base count")
    output = Path(output)
    wanted = set(problems or ELIGIBLE)
    if not wanted <= set(ELIGIBLE):
        raise ValueError("Only demand/TW eligible TRAIN tasks can be prepared")
    endpoints, pairs, rejections = [], [], []
    for problem in ELIGIBLE:
        if problem not in wanted:
            continue
        source = selection["sources"][problem]
        path = Path(source["dataset"])
        if file_hash(path) != source["dataset_sha256"]:
            raise ValueError("Frozen input dataset changed")
        with path.open("rb") as handle:
            items = pickle.load(handle)
        for base in [b for b in selection["bases"] if b["problem"] == problem and b["slot"] < limit]:
            item = plain(items[base["local_index"]])
            check = validate_input(problem, item)
            if check["n"] != base["scale"]:
                raise ValueError("Base/input metadata misalignment")
            prefix = f"{problem}_src{base['source_index']}"
            endpoint_items = [(f"{prefix}_original", "original", "original", item, check)]
            for family in ("demand", "tw_center") if "TW" in problem else ("demand",):
                seed = seed_for(selection["seed"], base["base_id"], family)
                group = intervention_pair(problem, item, family, seed)
                if group["status"] != "ready":
                    rejections.append(dict(base_id=base["base_id"], problem=problem, family=family, **group))
                    continue
                for arm in ("directed", "random"):
                    record = group[arm]
                    endpoint_id = f"{prefix}_{family}_{arm}"
                    endpoint_items.append((endpoint_id, family, arm, record["item"], record["input_check"]))
                    pairs.append(dict(pair_id=endpoint_id, problem=problem, base_id=base["base_id"],
                        original_id=f"{prefix}_original", changed_id=endpoint_id, family=family, arm=arm,
                        seed=seed, permutation=record["permutation"], l1=record["l1"],
                        geometry_score=record["geometry_score"], moved_customers=record["moved_customers"],
                        proposal_count=group["proposal_count"]))
            for endpoint_id, family, arm, endpoint_item, certificate in endpoint_items:
                input_path = output / "inputs" / f"{endpoint_id}.json"
                payload = dict(schema_version=1, version=selection["version"], endpoint_id=endpoint_id,
                    problem=problem, split="train", base_id=base["base_id"], item=endpoint_item)
                write_new_json(input_path, payload)
                endpoints.append(dict(endpoint_id=endpoint_id, problem=problem, base_id=base["base_id"],
                    source_base_id=base["source_base_id"], local_index=base["local_index"],
                    source_index=base["source_index"], family=family, arm=arm,
                    input_path=str(input_path.resolve()), input_sha256=file_hash(input_path),
                    item_sha256=canonical_hash(endpoint_item), input_check=certificate))
        del items
        gc.collect()
    tasks = []
    for endpoint in endpoints:
        for candidate in selection["pools"][endpoint["problem"]]:
            tasks.append(dict(task_id=f"{endpoint['endpoint_id']}__{candidate['method']}",
                endpoint_id=endpoint["endpoint_id"], problem=endpoint["problem"], base_id=endpoint["base_id"],
                input_sha256=endpoint["input_sha256"], method=candidate["method"],
                global_id=candidate["global_id"], source_method=candidate["source_method"],
                solver_seed=seed_for(selection["seed"], endpoint["base_id"], candidate["method"], "solver"),
                batch_size=1, rerun_original=True, status="not_run"))
    manifest = dict(schema_version=1, version=selection["version"], split="train",
        selection_sha256=selection["selection_sha256"], selected_problems=sorted(wanted),
        prepared_prefix_per_problem=limit, pools=selection["pools"],
        endpoints=endpoints, pairs=pairs, tasks=tasks, rejections=rejections,
        endpoint_count=len(endpoints), pair_count=len(pairs), task_count=len(tasks), solver_execution=False)
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_new_json(output / "manifest.json", manifest)
    return manifest


def verify_manifest(manifest):
    payload = dict(manifest)
    recorded = payload.pop("manifest_sha256")
    if canonical_hash(payload) != recorded or manifest["split"] != "train":
        raise ValueError("Manifest changed or is not TRAIN")
    endpoints = {e["endpoint_id"]: e for e in manifest["endpoints"]}
    if len(endpoints) != len(manifest["endpoints"]):
        raise ValueError("Duplicate endpoint identity")
    for endpoint in manifest["endpoints"]:
        payload = read_json(endpoint["input_path"])
        if (file_hash(endpoint["input_path"]) != endpoint["input_sha256"] or
                payload["endpoint_id"] != endpoint["endpoint_id"] or payload["split"] != "train" or
                payload["version"] != manifest["version"] or
                payload["base_id"] != endpoint["base_id"] or payload["problem"] != endpoint["problem"] or
                canonical_hash(payload["item"]) != endpoint["item_sha256"]):
            raise ValueError("Endpoint input/lineage changed")
        validate_input(endpoint["problem"], payload["item"])
    expected = {(e["endpoint_id"], c["method"]) for e in manifest["endpoints"]
                for c in manifest["pools"][e["problem"]]}
    actual = [(t["endpoint_id"], t["method"]) for t in manifest["tasks"]]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Task manifest does not cover the exact whole legal pool")
    if len({t["task_id"] for t in manifest["tasks"]}) != len(manifest["tasks"]):
        raise ValueError("Duplicate task identity")
    for task in manifest["tasks"]:
        endpoint = endpoints[task["endpoint_id"]]
        candidate = next(c for c in manifest["pools"][endpoint["problem"]] if c["method"] == task["method"])
        if (any(task[k] != endpoint[k] for k in ("problem", "base_id", "input_sha256")) or
                any(task[k] != candidate[k] for k in ("method", "global_id", "source_method")) or
                task["batch_size"] != 1 or not task["rerun_original"]):
            raise ValueError("Task identity/input/deployment mismatch")
    groups = defaultdict(set)
    for pair in manifest["pairs"]:
        original, changed = endpoints[pair["original_id"]], endpoints[pair["changed_id"]]
        if (original["arm"] != "original" or pair["arm"] not in ("directed", "random") or
                any(pair[k] != original[k] or pair[k] != changed[k] for k in ("base_id", "problem")) or
                pair["family"] != changed["family"] or pair["arm"] != changed["arm"]):
            raise ValueError("Paired endpoint/base lineage mismatch")
        validate_intervention(pair["problem"], read_json(original["input_path"])["item"],
                              read_json(changed["input_path"])["item"], pair["family"])
        groups[(pair["base_id"], pair["family"])].add(pair["arm"])
    if any(arms != {"directed", "random"} for arms in groups.values()):
        raise ValueError("Directed/random mechanism control is incomplete")


def runtime_description(overlay):
    wanted = {"numpy", "torch", "torchrl", "tensordict", "rl4co", "hydra-core", "lightning"}
    packages = {}
    for distribution in importlib.metadata.Distribution.discover(path=[str(overlay), *sys.path]):
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        if name in wanted and name not in packages:
            packages[name] = dict(version=distribution.version, root=str(distribution.locate_file("").resolve()))
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()), packages=packages)


def inventory_sources(data_root=DATA_ROOT, legacy_root=LEGACY_ROOT, verify_hashes=True):
    """Read actual production profiles/dependencies without loading a solver."""
    legacy_root = Path(legacy_root).resolve()
    scripts = legacy_root.parents[1] / "scripts"
    records, artifacts, cache = [], {}, {}

    def observe(path, expected=None):
        path = str(Path(path).resolve())
        if path not in cache:
            cache[path] = file_hash(path) if verify_hashes and Path(path).is_file() else None
        exists = Path(path).is_file()
        current = cache[path]
        return dict(path=path, exists=exists, expected_sha256=expected, current_sha256=current,
                    hash_matches=(current == expected) if verify_hashes and expected else None)

    for problem in PROBLEMS:
        for candidate in load_pool(data_root, problem):
            source = legacy_root / "labels" / problem / candidate["source_method"]
            paths = {name: source / name for name in ("profile.json", "dependency_contract.json", "profile_path_bindings.json")}
            missing_metadata = [str(p) for p in paths.values() if not p.is_file()]
            if missing_metadata:
                records.append(dict(problem=problem, **candidate, status="missing_metadata",
                                    missing=missing_metadata, changed=[]))
                continue
            profile = read_json(paths["profile.json"])
            dependencies = read_json(paths["dependency_contract.json"])
            bindings = read_json(paths["profile_path_bindings.json"])
            expected = dict(dependencies["artifacts"])
            for binding in bindings["bindings"].values():
                if binding["path"] in expected and expected[binding["path"]] != binding["sha256"]:
                    raise ValueError("Conflicting production dependency and profile binding")
                expected[binding["path"]] = binding["sha256"]
            observations = [observe(p, digest) for p, digest in expected.items()]
            missing = [r["path"] for r in observations if not r["exists"]]
            changed = [r["path"] for r in observations if r["hash_matches"] is False]
            for record in observations:
                artifacts[record["path"]] = record
            checkpoints = []
            if profile.get("checkpoint"):
                checkpoints.append(dict(path=profile["checkpoint"], sha256=profile.get("checkpoint_sha256")))
            checkpoints.extend(profile.get("checkpoints", {}).values())
            checkpoints.extend(dependencies.get("actual_scale_checkpoints", {}).values())
            checkpoints = [dict(path=p, sha256=h) for p, h in sorted({
                (r["path"], r.get("sha256")) for r in checkpoints})]
            records.append(dict(problem=problem, **candidate, backend=profile["backend"],
                status="missing_artifacts" if missing else "changed_artifacts" if changed else
                       "pinned_files_match" if verify_hashes else "files_present_hashes_unchecked",
                missing=missing, changed=changed, budget=profile.get("budget"), checkpoints=checkpoints,
                profile_path=str(paths["profile.json"]), profile_file_sha256=file_hash(paths["profile.json"]),
                profile_sha256=canonical_hash(profile), dependency_path=str(paths["dependency_contract.json"]),
                dependency_file_sha256=file_hash(paths["dependency_contract.json"]),
                bindings_path=str(paths["profile_path_bindings.json"]),
                bindings_file_sha256=file_hash(paths["profile_path_bindings.json"]),
                runtime_verified_this_turn=False, intervention_applicable=problem in ELIGIBLE))
    # These factory/decision files are essential even when omitted by a legacy dependency table.
    factory_files = ("train_100k_sampling.py", "train_100k_alias_weights.py", "train_100k_backends.py",
                     "train_100k_common.py")
    for name in factory_files:
        record = observe(scripts / name)
        artifacts[record["path"]] = record
    for path in (legacy_root / "recipe_decisions.json", scripts / "train_100k_environment.json"):
        record = observe(path)
        artifacts[record["path"]] = record
    decision = read_json(legacy_root / "recipe_decisions.json")
    compat = legacy_root / "python_compat"
    overlay_artifacts = {}
    if compat.is_dir() and verify_hashes:
        for path in sorted(compat.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".so"):
                overlay_artifacts[str(path.resolve())] = file_hash(path)
    payload = dict(schema_version=1, adapter_entry=ADAPTER_ENTRY, scripts_root=str(scripts),
        legacy_root=str(legacy_root), overlay_root=str(compat), overlay_exists=compat.is_dir(),
        overlay_artifacts=overlay_artifacts, hashes_verified=verify_hashes,
        runtime=runtime_description(compat),
        records=records, artifacts=artifacts, sampling_decision=decision["mvrp_decoder_strategy"],
        runtime_verified=False, solver_execution=False,
        summary=dict(legal_deployments=len(records), eligible_deployments=sum(r["problem"] in ELIGIBLE for r in records),
                     missing_deployments=sum(bool(r["missing"]) for r in records),
                     changed_deployments=sum(bool(r["changed"]) for r in records),
                     unique_artifacts=len(artifacts), overlay_files=len(overlay_artifacts)),
        interpretation="File evidence only. Not a runtime qualification or exact historical reproduction.")
    payload["inventory_sha256"] = canonical_hash(payload)
    return payload


def verify_inventory(inventory):
    payload = dict(inventory)
    recorded = payload.pop("inventory_sha256")
    if canonical_hash(payload) != recorded or not inventory["hashes_verified"]:
        raise ValueError("Inventory changed or hashes were never verified")
    if inventory["sampling_decision"] != "sampling" or not inventory["overlay_exists"]:
        raise ValueError("Confirmed sampling or isolated compatibility overlay missing")
    for record in inventory["records"]:
        if record["status"] != "pinned_files_match":
            raise ValueError(f"Unresolved deployment source: {record['problem']}/{record['method']}")


def source_lock(inventory, manifest):
    """CPU lock of a new production, deliberately NOT a historical replay claim."""
    verify_inventory(inventory)
    verify_manifest(manifest)
    fingerprints = {p: r["current_sha256"] for p, r in inventory["artifacts"].items()}
    fingerprints.update(inventory["overlay_artifacts"])
    records = {}
    for r in inventory["records"]:
        key = f"{r['problem']}/{r['method']}"
        records[key] = {k: r[k] for k in ("profile_path", "profile_sha256", "backend", "budget", "checkpoints")}
        for prefix in ("profile", "dependency", "bindings"):
            fingerprints[r[f"{prefix}_path"]] = r[f"{prefix}_file_sha256"]
    if any(v is None for v in fingerprints.values()):
        raise ValueError("A required current artifact was not hashed")
    fingerprints[str(Path(__file__).resolve())] = file_hash(__file__)
    lock = dict(schema_version=1, version=manifest["version"], manifest_sha256=manifest["manifest_sha256"],
        inventory_sha256=inventory["inventory_sha256"], adapter_entry=ADAPTER_ENTRY,
        scripts_root=inventory["scripts_root"], legacy_root=inventory["legacy_root"],
        overlay_root=inventory["overlay_root"], artifacts=fingerprints, deployments=records,
        runtime=inventory["runtime"],
        execution=dict(batch_size=1, fresh_process_per_task=True, seed="common per base and method; reset after model loading",
                       both_endpoints_fresh=True, old_cost_reuse=False),
        runtime_verified=False, historical_reproduction=False)
    lock["lock_sha256"] = canonical_hash(lock)
    return lock


def verify_lock(lock, manifest, verify_files=False):
    payload = dict(lock)
    recorded = payload.pop("lock_sha256")
    if (canonical_hash(payload) != recorded or lock["manifest_sha256"] != manifest["manifest_sha256"] or
            lock["version"] != manifest["version"] or lock["adapter_entry"] != ADAPTER_ENTRY):
        raise ValueError("Deployment lock mismatch")
    if verify_files:
        if runtime_description(lock["overlay_root"]) != lock["runtime"]:
            raise ValueError("Locked Python/package runtime changed")
        for path, digest in lock["artifacts"].items():
            if not Path(path).is_file() or file_hash(path) != digest:
                raise ValueError(f"Locked source/weight/runtime overlay changed: {path}")


def dispatch(manifest, output, task_id=None, execute=False, lock=None, device=None):
    """One task per fresh process, avoiding legacy global import/model caches."""
    verify_manifest(manifest)
    tasks = [t for t in manifest["tasks"] if task_id is None or t["task_id"] == task_id]
    if not tasks:
        raise ValueError("Unknown task ID")
    if not execute:
        plan = dict(version=manifest["version"], manifest_sha256=manifest["manifest_sha256"],
            solver_execution=False, task_count=len(tasks), adapter_entry=ADAPTER_ENTRY,
            required="explicit --execute, --task-id, --device and immutable --deployment-lock; resources released by R112 owner",
            tasks=tasks)
        write_new_json(output, plan)
        return plan
    if lock is None or device is None or task_id is None or len(tasks) != 1:
        raise ValueError("Execution requires one explicit task, device and source lock")
    verify_lock(lock, manifest, verify_files=True)
    task = tasks[0]
    endpoint = next(e for e in manifest["endpoints"] if e["endpoint_id"] == task["endpoint_id"])
    payload = read_json(endpoint["input_path"])
    result = dict(schema_version=1, version=manifest["version"], task_id=task["task_id"],
        endpoint_id=task["endpoint_id"], base_id=task["base_id"], problem=task["problem"],
        method=task["method"], input_sha256=task["input_sha256"], lock_sha256=lock["lock_sha256"],
        solver_seed=task["solver_seed"], status="error", source="fresh_solver",
        cost=None, tour=None, route_audit=None)
    started = time.monotonic()
    try:
        sys.path[:0] = [lock["overlay_root"], lock["scripts_root"]]
        import torch
        torch.set_num_threads(1)
        model_seed = seed_for(2, task["problem"], task["method"], "model")
        random.seed(model_seed)
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        factory = importlib.import_module("train_100k_sampling").make_confirmed_backend
        backend = factory(task["problem"], task["source_method"], device, Path(lock["legacy_root"]))
        profile = backend.profile()
        expected = lock["deployments"][f"{task['problem']}/{task['method']}"]
        if canonical_hash(profile) != expected["profile_sha256"]:
            raise ValueError("Actual initialized backend profile differs from locked recorded profile")
        item = payload["item"]
        if task["problem"] == "CVRP":
            item = {k: torch.tensor(v, dtype=torch.float32) for k, v in item.items()}
        random.seed(task["solver_seed"])
        np.random.seed(task["solver_seed"])
        torch.manual_seed(task["solver_seed"])
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(task["solver_seed"])
            torch.cuda.synchronize()
        solved = backend.run([item])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if len(solved.costs) != 1 or len(solved.tours) != 1:
            raise ValueError("Solver output count mismatch")
        audit = check_route(task["problem"], payload["item"], solved.tours[0], solved.costs[0])
        result.update(status="ok", cost=float(solved.costs[0]), tour=plain(solved.tours[0]),
                      route_audit=audit, profile_sha256=canonical_hash(profile), backend_audit=plain(solved.audit))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    result["wall_seconds_including_load"] = time.monotonic() - started
    write_new_json(output, result)
    return result


def endpoint_labels(pool, records):
    """Only independently verified fresh receipts are usable supervision."""
    order = sorted(pool, key=lambda r: r["global_id"])
    costs, valid, statuses = [], [], []
    for candidate in order:
        receipt = records.get(candidate["method"])
        status = receipt.get("status", "missing") if receipt else "missing"
        value = receipt.get("cost") if receipt else None
        good = bool(receipt and status == "ok" and receipt.get("source") == "fresh_solver" and
                    (receipt.get("route_audit") or {}).get("verified") is True and
                    value is not None and np.isfinite(float(value)) and float(value) > 0)
        costs.append(float(value) if good else None)
        valid.append(good)
        statuses.append(status if good or status != "ok" else "unverified")
    complete = all(valid)
    c = np.asarray([v if v is not None else np.nan for v in costs], dtype=np.float64)
    gap = [[float(c[i] - c[j]) if valid[i] and valid[j] else None
            for j in range(len(c))] for i in range(len(c))]
    winner = order[int(c.argmin())] if complete else None
    return dict(methods=[r["method"] for r in order], global_ids=[r["global_id"] for r in order],
        costs_fp64=costs, valid_mask=valid, failure_mask=[not v for v in valid], statuses=statuses,
        winner=winner["method"] if winner else None, winner_global_id=winner["global_id"] if winner else None,
        winner_mask=complete, relative_response=(c - c.mean()).tolist() if complete else [None] * len(c),
        relative_response_mask=[complete] * len(c), pair_gap=gap,
        pair_mask=[[a and b for b in valid] for a in valid], winner_policy=WINNER_POLICY)


def paired_labels(original, changed):
    if original["methods"] != changed["methods"] or original["global_ids"] != changed["global_ids"]:
        raise ValueError("Paired endpoints have different candidate identities/order")
    valid = [a and b for a, b in zip(original["valid_mask"], changed["valid_mask"])]
    n = len(valid)
    delta = [[(changed["pair_gap"][i][j] - original["pair_gap"][i][j]) if valid[i] and valid[j] else None
              for j in range(n)] for i in range(n)]
    full = original["winner_mask"] and changed["winner_mask"]
    return dict(methods=original["methods"], candidate_change=[
        changed["costs_fp64"][i] - original["costs_fp64"][i] if valid[i] else None for i in range(n)],
        candidate_change_mask=valid, failure_mask=[not v for v in valid], Delta_mn=delta,
        Delta_mask=[[a and b for b in valid] for a in valid],
        relative_response_change=[changed["relative_response"][i] - original["relative_response"][i]
                                  if full else None for i in range(n)],
        relative_response_change_mask=[full] * n, original_winner=original["winner"],
        changed_winner=changed["winner"], winner_pair_mask=full,
        winner_flip=(original["winner"] != changed["winner"]) if full else None,
        delta_definition="(C_after[m]-C_after[n])-(C_before[m]-C_before[n]); native cost units")


def assemble_labels(manifest, results_dir, output, lock=None):
    verify_manifest(manifest)
    if lock is not None:
        verify_lock(lock, manifest)
    results_dir = Path(results_dir)
    expected_files = {t["task_id"] for t in manifest["tasks"]}
    if results_dir.exists() and {p.stem for p in results_dir.glob("*.json")} - expected_files:
        raise ValueError("Unknown result receipts; do not mix versions or attempts")
    by_endpoint = defaultdict(dict)
    endpoints = {e["endpoint_id"]: e for e in manifest["endpoints"]}
    for task in manifest["tasks"]:
        path = results_dir / f"{task['task_id']}.json"
        if not path.exists():
            continue
        if lock is None:
            raise ValueError("Real result ingestion requires a deployment lock")
        receipt = read_json(path)
        expected = {k: task[k] for k in ("task_id", "endpoint_id", "base_id", "problem", "method", "input_sha256", "solver_seed")}
        expected.update(version=manifest["version"], lock_sha256=lock["lock_sha256"], source="fresh_solver")
        if any(receipt.get(k) != value for k, value in expected.items()):
            raise ValueError("Result receipt is stale, reused, or has wrong input/deployment provenance")
        receipt = copy.deepcopy(receipt)
        if receipt["status"] == "ok":
            profile = lock["deployments"][f"{task['problem']}/{task['method']}"]
            if receipt.get("profile_sha256") != profile["profile_sha256"]:
                raise ValueError("Receipt profile mismatch")
            item = read_json(endpoints[task["endpoint_id"]]["input_path"])["item"]
            try:
                receipt["route_audit"] = check_route(task["problem"], item, receipt["tour"], receipt["cost"])
            except (ValueError, TypeError) as error:
                receipt.update(status="invalid_route", route_audit=dict(verified=False), error=str(error))
        by_endpoint[task["endpoint_id"]][task["method"]] = receipt
    rows = {e["endpoint_id"]: dict(endpoint_id=e["endpoint_id"], base_id=e["base_id"],
        problem=e["problem"], source_index=e["source_index"],
        **endpoint_labels(manifest["pools"][e["problem"]], by_endpoint[e["endpoint_id"]]))
        for e in manifest["endpoints"]}
    pairs = [dict(pair, **paired_labels(rows[pair["original_id"]], rows[pair["changed_id"]]))
             for pair in manifest["pairs"]]
    payload = dict(schema_version=1, version=manifest["version"], manifest_sha256=manifest["manifest_sha256"],
        lock_sha256=lock["lock_sha256"] if lock else None, endpoints=list(rows.values()), pairs=pairs,
        summary=dict(endpoint_count=len(rows), full_pool_winners=sum(r["winner_mask"] for r in rows.values()),
                     pair_count=len(pairs), full_pool_pairs=sum(p["winner_pair_mask"] for p in pairs),
                     verified_candidate_costs=sum(sum(r["valid_mask"]) for r in rows.values())))
    write_new_json(output, payload)
    return payload


def cli_output_path(path):
    path = Path(path).resolve()
    if PREPARATION.resolve() not in path.parents:
        raise ValueError(f"R114 CLI writes only beneath {PREPARATION}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("schema", "inventory", "freeze", "prepare", "lock", "dispatch", "assemble"):
        child = sub.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        if name in ("inventory", "freeze"):
            child.add_argument("--data-root", type=Path, default=DATA_ROOT)
        if name == "inventory":
            child.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT)
        if name == "freeze":
            child.add_argument("--version", default=VERSION)
            child.add_argument("--count", type=int, default=32)
            child.add_argument("--seed", type=int, default=2)
        if name == "prepare":
            child.add_argument("--selection", type=Path, required=True)
            child.add_argument("--problems", nargs="+", choices=ELIGIBLE)
            child.add_argument("--limit", type=int, default=2)
        if name in ("lock", "dispatch", "assemble"):
            child.add_argument("--manifest", type=Path, required=True)
        if name == "lock":
            child.add_argument("--inventory", type=Path, required=True)
        if name in ("dispatch", "assemble"):
            child.add_argument("--deployment-lock", type=Path)
        if name == "dispatch":
            child.add_argument("--task-id")
            child.add_argument("--execute", action="store_true")
            child.add_argument("--device")
        if name == "assemble":
            child.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = cli_output_path(args.output)
    if args.command == "schema":
        write_new_json(output, dict(schema_version=1, split="train", solver_execution=False,
                                   winner_requires="verified fresh full legal pool"))
        return 0
    if args.command == "inventory":
        result = inventory_sources(args.data_root, args.legacy_root)
        write_new_json(output, result)
        print(json.dumps(result["summary"]))
    elif args.command == "freeze":
        result = freeze_selection(args.data_root, args.version, args.count, args.seed)
        write_new_json(output, result)
        print(json.dumps(dict(bases=len(result["bases"]), solver_execution=False)))
    elif args.command == "prepare":
        result = prepare_inputs(read_json(args.selection), output, args.problems, args.limit)
        print(json.dumps({k: result[k] for k in ("endpoint_count", "pair_count", "task_count")}))
    elif args.command == "lock":
        result = source_lock(read_json(args.inventory), read_json(args.manifest))
        write_new_json(output, result)
        print(json.dumps(dict(locked=True, runtime_verified=False, solver_execution=False)))
    elif args.command == "dispatch":
        result = dispatch(read_json(args.manifest), output, args.task_id, args.execute,
                          read_json(args.deployment_lock) if args.deployment_lock else None, args.device)
        print(json.dumps({k: result[k] for k in ("solver_execution", "task_count") if k in result}
                         or dict(task_id=result["task_id"], status=result["status"])))
        return 0 if result.get("status", "ok") == "ok" else 1
    elif args.command == "assemble":
        result = assemble_labels(read_json(args.manifest), args.results_dir, output,
                                 read_json(args.deployment_lock) if args.deployment_lock else None)
        print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
