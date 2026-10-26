"""Bounded, input-only CPU features for R115's TRAIN response pilot.

The caller passes only an R114 endpoint's ``item``, never its ID envelope.
Conversion follows r114_response_data.fields and UnifiedProblemDataset: demand
is already native (no capacity renormalization), depot attributes are zero
except xy/is_depot, and CVRP's four channels are padded to eight inside R92.
``coord_dist=0`` is a fixed missing-metadata placeholder for every problem,
NOT a measured uniform-distribution label or an inference from a source index.
No dataset, label, source-index lookup, fitted statistics, or GPU is used.
"""

from __future__ import annotations

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS, constraint_bits, problem_descriptor
from .local_geometry import local_geometry
from .r92_tree_features import COLUMNS, features


MAX_CUSTOMERS = 1000  # Bound the quadratic CPU local-geometry calculation.
_ELIGIBLE = frozenset(PROBLEMS) - {"TSP", "ATSP"}
_CVRP_KEYS = frozenset(("depot", "loc", "demand"))
_MVRP_KEYS = frozenset(("depot_xy", "node_xy", "node_demand", "capacity"))


def _array(item, key, shape=None):
    value = item[key]
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{key} must be CPU input")
        value = value.detach().numpy()
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric input: {key}") from exc
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError(f"Invalid shape or nonfinite input: {key}")
    return result


@torch.no_grad()
def native_phi(problem: str, item: dict) -> np.ndarray:
    """Return one finite CPU FP32 ``(357,)`` row in R92 ``COLUMNS`` order.

    Accept exactly the R114 native fields for CVRP or a registered MVRP task;
    L/TW fields are required only for the corresponding active constraints.
    Reject extra keys before inspecting any values. Inputs are never mutated.
    At most 1000 customers plus one depot are processed per call, unpadded.
    """
    if not isinstance(problem, str) or problem not in _ELIGIBLE:
        raise ValueError("Expected CVRP or a registered R114 MVRP problem")
    if not isinstance(item, dict):
        raise ValueError("Expected the native item dict, not an endpoint envelope")
    bits = constraint_bits(problem)
    expected = set(_CVRP_KEYS if problem == "CVRP" else _MVRP_KEYS)
    if bits[3]:
        expected.add("route_limit")
    if bits[4]:
        expected.update(("service_time", "tw_start", "tw_end"))
    if set(item) != expected:
        raise ValueError("Unexpected native fields (extra or missing keys)")

    demand = _array(item, "demand" if problem == "CVRP" else "node_demand")
    if problem == "CVRP":
        if demand.ndim != 2 or demand.shape[0] != 1:
            raise ValueError("Native CVRP demand must have shape (1, n)")
        demand = demand[0]
    elif demand.ndim != 1:
        raise ValueError("Native MVRP demand must have shape (n,)")
    customers = len(demand)
    if not 1 <= customers <= MAX_CUSTOMERS:
        raise ValueError(f"Expected 1..{MAX_CUSTOMERS} customers")

    if problem == "CVRP":
        xy = _array(item, "loc", (1, customers, 2))[0]
        depot = _array(item, "depot", (1, 1, 2))[0, 0]
        d_in = 4
    else:
        xy = _array(item, "node_xy", (customers, 2))
        depot = _array(item, "depot_xy")
        if depot.shape not in ((1, 2), (2,)):
            raise ValueError("Native MVRP requires one two-coordinate depot")
        depot = depot.reshape(2)
        if _array(item, "capacity", ()).item() <= 0:
            raise ValueError("Native capacity must be positive")
        d_in = 8

    n = customers + 1
    node = torch.zeros((1, n, d_in), dtype=torch.float32, device="cpu")
    node[0, 0, :2] = torch.from_numpy(depot)
    node[0, 0, 3] = 1
    node[0, 1:, :2] = torch.from_numpy(xy)
    node[0, 1:, 2] = torch.from_numpy(demand)
    if bits[3]:
        node[0, 1:, 4] = float(_array(item, "route_limit", ()))
    if bits[4]:
        for channel, key in enumerate(("service_time", "tw_start", "tw_end"), 5):
            node[0, 1:, channel] = torch.from_numpy(_array(item, key, (customers,)))
    mask = torch.ones((1, n), dtype=torch.bool, device="cpu")
    batch = dict(kind="coord", node=node, node_mask=mask, n=torch.tensor([n], device="cpu"),
                 d_in=d_in, node_geom=local_geometry(node[..., :2], mask).float(),
                 cbits=torch.tensor([bits], dtype=torch.float32, device="cpu"),
                 problem_desc=torch.tensor([problem_descriptor(problem)], dtype=torch.float32,
                                           device="cpu"),
                 coord_dist=torch.zeros(1, dtype=torch.long, device="cpu"))
    rows, columns = features(batch)
    if columns != COLUMNS or rows.shape != (1, 357):
        raise ValueError("R92 feature schema changed")
    return rows[0].copy()
