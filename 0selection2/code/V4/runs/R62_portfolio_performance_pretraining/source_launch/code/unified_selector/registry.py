"""Problem / solver registry for the unified selector.

The candidate pools must match the current dataset labels exactly.
Instead of freezing an old handcrafted solver list, we discover each
problem's pool from ``data/<problem>train/results/result_*.txt`` and fall
back to a known-safe default only when those files are unavailable.
"""
from __future__ import annotations
import re
from pathlib import Path

# All 18 problems in a canonical order (index = problem_id).
PROBLEMS = [
    "TSP", "CVRP", "ATSP",
    # 15 MVRP variants:
    "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW", "OVRPB", "OVRPL",
    "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW",
    "VRPBLTW", "OVRPBLTW",
]
P2I = {p: i for i, p in enumerate(PROBLEMS)}

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

# Safe fallback pools that match the current checked-in dataset layout.
_FALLBACK_POOLS = {
    "TSP": ["BQ", "DIFUSCO", "DIFUSCO500", "ELG", "LEHD", "OMNI", "T2T", "T2T500"],
    "CVRP": ["BQ", "ELG", "ICAM", "LEHD", "MVMOE", "MoSES_CaDA", "MoSES_RF", "OMNI", "RELD_CVRP", "RouteFinder"],
    "ATSP": ["GLOP", "ICAM_ATSP", "MATNET", "MATPOENET", "UNICO_MatPOENet"],
}
for _mvrp in PROBLEMS[3:]:
    _FALLBACK_POOLS[_mvrp] = ["MTPOMO", "MVMOE", "MoSES_CaDA", "MoSES_RF", "RELD_MOEL", "RELD_MTL", "RouteFinder"]


def _strip_result_name(name: str) -> str:
    return re.sub(r"^result_|\.txt$", "", name)


def _discover_pools() -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    for problem in PROBLEMS:
        result_dir = DATA_ROOT / f"{problem}train" / "results"
        files = sorted(result_dir.glob("result_*.txt")) if result_dir.exists() else []
        if not files:
            return dict(_FALLBACK_POOLS)
        pools[problem] = [_strip_result_name(path.name) for path in files]
    return pools


# Per-problem pool — ORDER MATCHES the raw_label.pkl cost vector
# (which follows sorted(result_*.txt) filenames).
POOLS = _discover_pools()

# Union solver vocabulary (index = solver_id). Sorted for stable IDs.
GLOBAL_SOLVERS = sorted({solver for pool in POOLS.values() for solver in pool})
S2I = {s: i for i, s in enumerate(GLOBAL_SOLVERS)}
M_GLOBAL = len(GLOBAL_SOLVERS)

# Constraint-bit vector (K=5): C, O, B, L, TW.  C=always-on routing constraint for VRP/MVRP; TSP/ATSP C=0.
CONSTRAINT_BITS = ["C", "O", "B", "L", "TW"]
K_CBITS = len(CONSTRAINT_BITS)

# Problem descriptor used by the zero-shot-friendly R25-ZS path.  Unlike
# ``problem_id``, these slots can be populated for unseen variants directly.
PROBLEM_DESCRIPTOR_FIELDS = [
    "C", "O", "B", "BP", "L", "TW", "MD", "A", "PD", "PC", "OP",
    "HAS_DEPOT", "IS_MATRIX", "IS_COORD", "DEPOT_COUNT_NORM",
    "ACTIVE_FEATURE_FRAC",
]
K_PROBLEM_DESC = len(PROBLEM_DESCRIPTOR_FIELDS)

def constraint_bits(problem: str):
    p = problem
    if p in ("TSP", "ATSP"):
        return [0, 0, 0, 0, 0]
    bits = [1, 0, 0, 0, 0]  # capacity always on for VRP/MVRP
    if "O" in p and (p.startswith("O") or p.startswith("VRPBL") or p.startswith("OVRP")):
        pass
    bits[1] = int(p.startswith("OVRP"))  # open route
    bits[2] = int("B" in p.replace("ATSP","") and p != "CVRP" and p != "TSP")  # backhaul flag
    # more precise: check explicit tokens
    toks = p.replace("OVRP", "OVRP_").replace("VRP", "VRP_")
    # Simpler rule: explicit subtring check on suffix letters after OVRP/VRP prefix
    suf = p
    for pre in ("OVRP", "VRP"):
        if suf.startswith(pre):
            suf = suf[len(pre):]
            break
    bits[1] = int(p.startswith("OVRP"))
    bits[2] = int("B" in suf)
    bits[3] = int("L" in suf)
    bits[4] = int("TW" in suf)
    return bits


def problem_descriptor(problem: str):
    """Return a descriptor for both seen and unseen routing variants.

    The vector intentionally avoids enumerating ``problem_id``.  It represents
    constraints/structure that are available for unseen tasks such as MD/BP
    variants, so a model trained with this descriptor can be evaluated without
    proxying the task to one of the 18 seen problem IDs.
    """
    p = str(problem).upper()
    low = p.lower()
    has_asym = p == "ATSP" or low.startswith("atsp") or p.startswith("A") or low.startswith("amd")
    is_tsp_like = p in {"TSP", "OP", "PCTSP", "SPCTSP"} or low.endswith("tsp")
    has_md = low.startswith("md") or low.startswith("amd")
    has_open = p.startswith("O") or low.startswith("mdoc") or low.startswith("aopd") or "ocvrp" in low
    has_bp = "bp" in low
    has_b = has_bp or ("b" in low and "atsp" not in low)
    has_l = "l" in low and "atsp" not in low
    has_tw = "tw" in low
    has_pd = "pd" in low
    has_pc = "pctsp" in low
    has_op = p == "OP" or p.startswith("OP") or "opd" in low
    has_capacity = ("cvrp" in low) or (p == "CVRP") or (p.startswith("VRP")) or (p.startswith("OVRP"))
    is_matrix = p == "ATSP" or low.startswith("atsp")
    is_coord = not is_matrix
    depot_count = 3.0 if has_md else (0.0 if (p in {"TSP", "ATSP"} or low.endswith("tsp")) else 1.0)
    has_depot = float(depot_count > 0 or has_op or has_pc)
    flags = [
        float(has_capacity),
        float(has_open),
        float(has_b),
        float(has_bp),
        float(has_l),
        float(has_tw),
        float(has_md),
        float(has_asym),
        float(has_pd),
        float(has_pc),
        float(has_op),
        has_depot,
        float(is_matrix),
        float(is_coord),
        min(depot_count, 5.0) / 5.0,
    ]
    active_frac = sum(flags[:11]) / 11.0
    flags.append(active_frac)
    return flags

def problem_to_pool_mask(problem: str):
    """Return (pool_order, global_mask).

    pool_order: list of global solver ids in the order cost[] stores them.
    global_mask: length-M_GLOBAL {0,1} vector, 1 iff solver is in pool.
    """
    order = [S2I[s] for s in POOLS[problem]]
    mask = [0] * M_GLOBAL
    for i in order:
        mask[i] = 1
    return order, mask

# COORD_DIST codes for MVRP (uniform=0, gaussian_mixture=1, clustered=2, ring=3). TSP/CVRP/ATSP → 4 = "other".
COORD_DIST = ["uniform", "gaussian_mixture", "clustered", "ring", "other"]
D_COORD = len(COORD_DIST)

def is_mvrp(problem: str) -> bool:
    return problem not in ("TSP", "CVRP", "ATSP")
