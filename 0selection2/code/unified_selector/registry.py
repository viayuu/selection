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
