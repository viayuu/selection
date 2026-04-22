"""Unified dataset for the 18-problem selector.

Each __getitem__ returns a homogeneous sample; collate only groups samples
with the same problem_id (variable n, different input types).  Training loop
uses a ProblemBalancedBatchSampler to ensure each batch is single-problem.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import (
    PROBLEMS, P2I, POOLS, S2I, M_GLOBAL, K_CBITS, D_COORD,
    problem_to_pool_mask, constraint_bits, is_mvrp,
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


# -------- 8-fold D4 augmentation (Kwon et al. 2020, POMO / NSS dataset.py:23-42) --------
def _augment_xy(xy: torch.Tensor, fold: int) -> torch.Tensor:
    """Apply one of 8 D4 transforms to xy coordinates in [0,1]^2.  Costs are invariant.
    fold in [0,7]:
      0: (x,y)            identity
      1: (1-x,y)          reflect x
      2: (x,1-y)          reflect y
      3: (1-x,1-y)        rotate 180
      4: (y,x)            transpose (reflect main diag)
      5: (1-y,x)          rotate 90 ccw
      6: (y,1-x)          rotate 270 ccw (= 90 cw)
      7: (1-y,1-x)        reflect anti-diag
    """
    x, y = xy[..., 0], xy[..., 1]
    if   fold == 0: nx, ny = x, y
    elif fold == 1: nx, ny = 1.0 - x, y
    elif fold == 2: nx, ny = x, 1.0 - y
    elif fold == 3: nx, ny = 1.0 - x, 1.0 - y
    elif fold == 4: nx, ny = y, x
    elif fold == 5: nx, ny = 1.0 - y, x
    elif fold == 6: nx, ny = y, 1.0 - x
    elif fold == 7: nx, ny = 1.0 - y, 1.0 - x
    else:
        raise ValueError(f"bad fold {fold}")
    out = xy.clone()
    out[..., 0] = nx
    out[..., 1] = ny
    return out


def _coord_dist_code(problem: str, inst_idx: int, split: str, n: int) -> int:
    """MVRP stratification code.  For non-MVRP return 4 ("other").
    NOTE: the 'coord_dist' channel is a misnomer — the MVRP data (per `data/README.md` and
    verified empirically: OVRPtrain[0].n=50 → [2500].n=62 → [5000].n=75 → [7500].n=88)
    is stratified primarily by *instance size* (n), not by the four coordinate families.
    So this code actually encodes the size-quartile bin.  We keep the column name for
    backward compatibility but clarify the semantics here.
    Equal 4-bin split: ``(inst_idx * 4) // n``.
    """
    if problem in ("TSP", "CVRP", "ATSP"):
        return 4
    q = (inst_idx * 4) // max(1, n)
    return int(min(3, q))


class UnifiedProblemDataset(Dataset):
    """One-problem dataset: loads instances + labels for a single problem × split.
    If `aug_8fold=True`, training-time __getitem__ applies a random D4 transform to
    the xy columns.  Labels (costs) are invariant under D4 on unit square → reuse.
    ATSP is not augmented (matrix doesn't have a canonical 2D embedding).
    """

    def __init__(self, problem: str, split: str, aug_8fold: bool = False):
        self.problem = problem
        self.split = split
        self.aug_8fold = aug_8fold and split == "train" and problem != "ATSP"
        d = DATA_ROOT / f"{problem}{split}"
        with open(d / "dataset.pkl", "rb") as f:
            self.instances = pickle.load(f)
        with open(d / "raw_label.pkl", "rb") as f:
            raw = pickle.load(f)
        # Keep only cost + ind to save RAM
        self.labels = {k: {"cost": v["cost"], "ind": v["ind"]} for k, v in raw.items()}
        del raw
        self.pool_order, self.mask = problem_to_pool_mask(problem)  # len==M_GLOBAL mask; pool_order = cost-slot positions
        self.cbits = constraint_bits(problem)
        self.pid = P2I[problem]
        self.K_p = len(self.pool_order)
        self.N = len(self.instances)
        # Raw cost vectors in raw_label.pkl are ordered by sorted(results/result_*.txt)
        # filenames — which may include solvers beyond POOLS[problem] (e.g. additional
        # baselines added after R18 training).  Build a per-problem gather index so
        # raw_cost[cost_gather_idx] aligns with POOLS[problem] order.
        results_dir = d / "results"
        if results_dir.exists():
            solver_files = sorted(f.name for f in results_dir.iterdir() if f.name.startswith("result_") and f.name.endswith(".txt"))
            solver_names = [f[len("result_"):-len(".txt")] for f in solver_files]
            try:
                self.cost_gather_idx = [solver_names.index(s) for s in POOLS[problem]]
            except ValueError as e:
                raise RuntimeError(f"data alignment error: POOL solver missing in {results_dir}: {e}")
        else:
            # Fallback: assume first K_p of raw cost == POOLS order (legacy data layout).
            self.cost_gather_idx = list(range(self.K_p))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        inst = self.instances[idx]
        lbl = self.labels[str(idx)]
        raw_cost = lbl["cost"]
        costs = torch.tensor([raw_cost[j] for j in self.cost_gather_idx], dtype=torch.float32)  # (K_p,), POOLS order
        ind = int(lbl["ind"])                                                # scalar
        p = self.problem
        if p == "TSP":
            # tensor (1,n,2)
            x = inst[0]  # (n,2)
            node = torch.as_tensor(x, dtype=torch.float32)
            kind = "coord"
            mat = None
            n = node.shape[0]
        elif p == "ATSP":
            mat = torch.as_tensor(inst[0] if inst.ndim == 3 else inst, dtype=torch.float32)  # (n,n)
            node = None
            kind = "matrix"
            n = mat.shape[0]
        elif p == "CVRP":
            loc = torch.as_tensor(inst["loc"][0], dtype=torch.float32)    # (n,2)
            dem = torch.as_tensor(inst["demand"][0], dtype=torch.float32)  # (n,)
            dep = torch.as_tensor(inst["depot"][0], dtype=torch.float32)   # (1,2)
            # Concatenate depot as first node. Node features: (x,y,demand,is_depot)
            feats = torch.cat([
                torch.cat([dep, torch.zeros(1,1), torch.ones(1,1)], dim=1),
                torch.cat([loc, dem.unsqueeze(1), torch.zeros(loc.shape[0],1)], dim=1),
            ], dim=0)  # (1+n, 4)
            node = feats
            kind = "coord"
            mat = None
            n = node.shape[0]
        else:
            # MVRP
            dep = torch.tensor(inst["depot_xy"], dtype=torch.float32)       # (1,2)
            xy  = torch.tensor(inst["node_xy"], dtype=torch.float32)        # (n,2)
            dem = torch.tensor(inst["node_demand"], dtype=torch.float32)    # (n,)
            rlim = float(inst.get("route_limit", 0.0))
            st = inst.get("service_time", None)
            tws = inst.get("tw_start", None)
            twe = inst.get("tw_end", None)
            has_L = self.cbits[3]
            has_TW = self.cbits[4]
            # Node features: x,y,demand (negative=backhaul),is_depot, rlim(if L else 0), st,tws,twe(if TW else 0s)
            n_nodes = xy.shape[0]
            dep_feat = torch.zeros(1, 8)
            dep_feat[0, 0:2] = dep[0]
            dep_feat[0, 3] = 1.0   # is_depot
            cust_feat = torch.zeros(n_nodes, 8)
            cust_feat[:, 0:2] = xy
            cust_feat[:, 2]   = dem
            cust_feat[:, 3]   = 0.0
            if has_L:
                cust_feat[:, 4] = rlim
            if has_TW and st is not None:
                cust_feat[:, 5] = torch.tensor(st)
                cust_feat[:, 6] = torch.tensor(tws)
                cust_feat[:, 7] = torch.tensor(twe)
            node = torch.cat([dep_feat, cust_feat], dim=0)  # (1+n, 8)
            kind = "coord"
            mat = None
            n = node.shape[0]
        # Coord dist code
        cd = _coord_dist_code(p, idx, self.split, self.N)
        # 8-fold augmentation: apply a random D4 transform to xy columns (coord kind only)
        if self.aug_8fold and kind == "coord" and node is not None:
            fold = int(torch.randint(0, 8, (1,)).item())
            # xy always lives at columns [0:2] — TSP: (x,y); CVRP: (x,y,demand,is_depot); MVRP: (x,y,...)
            node = node.clone()
            node[:, 0:2] = _augment_xy(node[:, 0:2], fold)
        return {
            "problem_id": self.pid,
            "problem_name": p,
            "kind": kind,
            "node": node,     # (n, d_in) or None
            "matrix": mat,    # (n, n) or None
            "cbits": torch.tensor(self.cbits, dtype=torch.float32),
            "coord_dist": cd,
            "pool_global_ids": torch.tensor(self.pool_order, dtype=torch.long),  # (K_p,)
            "mask": torch.tensor(self.mask, dtype=torch.float32),                # (M_global,)
            "costs": costs,                                                       # (K_p,)
            "ind": ind,
            "n": n,
        }


def collate_single_problem(batch: List[dict]):
    """Collate items of the SAME problem_id, pad coord nodes by their max n.
    Returns a dict of batched tensors.
    """
    kind = batch[0]["kind"]
    pid = batch[0]["problem_id"]
    B = len(batch)
    K_p = batch[0]["pool_global_ids"].shape[0]
    cbits = torch.stack([b["cbits"] for b in batch])
    cd    = torch.tensor([b["coord_dist"] for b in batch], dtype=torch.long)
    pool_ids = batch[0]["pool_global_ids"].clone()
    mask = torch.stack([b["mask"] for b in batch])
    costs = torch.stack([b["costs"] for b in batch])
    ind = torch.tensor([b["ind"] for b in batch], dtype=torch.long)
    n_arr = torch.tensor([b["n"] for b in batch], dtype=torch.long)

    if kind == "coord":
        max_n = int(n_arr.max().item())
        d_in = batch[0]["node"].shape[1]
        node = torch.zeros(B, max_n, d_in)
        node_mask = torch.zeros(B, max_n, dtype=torch.bool)
        for i, b in enumerate(batch):
            nn = b["n"]
            node[i, :nn] = b["node"]
            node_mask[i, :nn] = True
        return {
            "problem_id": pid, "kind": kind,
            "node": node, "node_mask": node_mask, "d_in": d_in,
            "cbits": cbits, "coord_dist": cd,
            "pool_ids": pool_ids, "mask": mask, "costs": costs, "ind": ind, "n": n_arr,
        }
    else:  # matrix (ATSP)
        max_n = int(n_arr.max().item())
        mat = torch.zeros(B, max_n, max_n)
        node_mask = torch.zeros(B, max_n, dtype=torch.bool)
        for i, b in enumerate(batch):
            nn = b["n"]
            mat[i, :nn, :nn] = b["matrix"]
            node_mask[i, :nn] = True
        return {
            "problem_id": pid, "kind": kind,
            "matrix": mat, "node_mask": node_mask,
            "cbits": cbits, "coord_dist": cd,
            "pool_ids": pool_ids, "mask": mask, "costs": costs, "ind": ind, "n": n_arr,
        }
