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


def augment_xy_by_8_fold(xy: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """Apply the standard 8-fold POMO/NSS geometric augmentation on [0,1]^2 coords."""
    x = xy[..., 0:1]
    y = xy[..., 1:2]
    variants = (
        torch.cat([x, y], dim=-1),
        torch.cat([1 - x, y], dim=-1),
        torch.cat([x, 1 - y], dim=-1),
        torch.cat([1 - x, 1 - y], dim=-1),
        torch.cat([y, x], dim=-1),
        torch.cat([1 - y, x], dim=-1),
        torch.cat([y, 1 - x], dim=-1),
        torch.cat([1 - y, 1 - x], dim=-1),
    )
    return variants[int(aug_idx) % 8]


def _coord_dist_code(problem: str, inst_idx: int, split: str, n: int) -> int:
    """MVRP coord distribution code.  For non-MVRP return 4 ("other").
    For MVRP: per README.md, train has ~uniform 2503/2499, val/test 250 each.
    Encoding: stratified by index within sorted coord-distribution order
    (uniform=0, gaussian_mixture=1, clustered=2, ring=3). We approximate
    by stable slicing (assumes dataset was constructed with that order).
    """
    if problem in ("TSP", "CVRP", "ATSP"):
        return 4
    # 4-way split; 10000 train, 1000 val/test. Order per data spec.
    q = inst_idx // (n // 4 + 1)
    return int(min(3, q))


class UnifiedProblemDataset(Dataset):
    """One-problem dataset: loads instances + labels for a single problem × split."""

    def __init__(self, problem: str, split: str, coord_augment: int = 0):
        self.problem = problem
        self.split = split
        split_key = str(split)
        split_upper = split_key.upper()
        if split_upper == "TSPLIB":
            if problem != "TSP":
                raise ValueError("TSPLIB split is only valid for problem='TSP'")
            d = DATA_ROOT / "TSPLIB"
        elif split_upper == "CVRPLIB":
            if problem != "CVRP":
                raise ValueError("CVRPLIB split is only valid for problem='CVRP'")
            d = DATA_ROOT / "CVRPLIB"
        else:
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
        self.base_N = len(self.instances)
        self.coord_augment = int(coord_augment) if split == "train" and problem != "ATSP" else 0
        if self.coord_augment not in (0, 1, 8):
            raise ValueError(f"coord_augment must be one of {{0,1,8}}, got {self.coord_augment}")
        self.N = self.base_N * (self.coord_augment if self.coord_augment > 1 else 1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        base_idx = idx % self.base_N if self.coord_augment > 1 else idx
        aug_idx = idx // self.base_N if self.coord_augment > 1 else 0
        inst = self.instances[base_idx]
        lbl = self.labels[str(base_idx)]
        costs = torch.tensor(lbl["cost"][:self.K_p], dtype=torch.float32)   # (K_p,)
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
        if kind == "coord" and self.coord_augment > 1:
            node = node.clone()
            node[:, 0:2] = augment_xy_by_8_fold(node[:, 0:2], aug_idx)
        cd = _coord_dist_code(p, base_idx, self.split, self.base_N)
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
