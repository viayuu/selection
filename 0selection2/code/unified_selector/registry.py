"""Problem / solver registry for the unified selector.

18 problems × union of 18 global solvers. Per-problem mask tells which
solvers are in the candidate pool, plus the order in which raw_label.pkl's
`cost` vector lists them (= sorted filenames of results/result_*.txt).
"""
from __future__ import annotations

# All 18 problems in a canonical order (index = problem_id).
PROBLEMS = [
    "TSP", "CVRP", "ATSP",
    # 15 MVRP variants:
    "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW", "OVRPB", "OVRPL",
    "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW",
    "VRPBLTW", "OVRPBLTW",
]
P2I = {p: i for i, p in enumerate(PROBLEMS)}

# Union solver vocabulary (index = solver_id). sorted alphabetically.
GLOBAL_SOLVERS = [
    "DACT", "DIFUSCO", "ELG", "GLOP", "ICAM", "INVIT", "LEHD", "LIH",
    "MATNET", "MATPOENET", "MTPOMO", "MVMOE", "OMNI",
    "RELD_CVRP", "RELD_MOEL", "RELD_MTL", "T2T", "UDC",
]
S2I = {s: i for i, s in enumerate(GLOBAL_SOLVERS)}
M_GLOBAL = len(GLOBAL_SOLVERS)  # = 18

# Per-problem pool — ORDER MATCHES the raw_label.pkl cost vector
# (which follows sorted(result_*.txt) filenames).
POOLS = {
    "TSP":  ["DACT","DIFUSCO","ELG","GLOP","INVIT","LEHD","LIH","OMNI","T2T","UDC"],
    "CVRP": ["DACT","ELG","GLOP","ICAM","INVIT","LEHD","OMNI","RELD_CVRP","UDC"],
    "ATSP": ["GLOP","MATNET","MATPOENET"],
}
# All MVRP variants share the same 4-method pool
for mvrp in PROBLEMS[3:]:
    POOLS[mvrp] = ["MTPOMO","MVMOE","RELD_MOEL","RELD_MTL"]

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
