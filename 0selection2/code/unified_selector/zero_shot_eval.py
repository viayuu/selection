"""Zero-shot evaluation for frozen selector checkpoints on unseen URS-style tasks.

This script evaluates an existing selector checkpoint on the zero13 bundle without
using zero-shot labels for training.  It adapts unseen problem instances into the
current R25-compatible batch schema and only uses labels to compute metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import augment_xy_by_8_fold
from .model import UnifiedSelector, remap_legacy_state_dict, shortlist_from_support
from .registry import D_COORD, GLOBAL_SOLVERS, K_CBITS, M_GLOBAL, P2I, PROBLEMS, S2I, problem_descriptor


ZERO13_PROBLEMS = [
    "ACVRP",
    "ACVRPTW",
    "MDCVRP",
    "MDCVRPTW",
    "AMDCVRP",
    "AMDCVRPTW",
    "PDCVRP",
    "OPDCVRP",
    "APDCVRP",
    "AOPDCVRP",
    "OP",
    "PCTSP",
    "SPCTSP",
]

ZERO_SOLVERS = ["MTPOMO", "MVMOE", "RELD"]


SEMANTIC_PROXY = {
    "ACVRP": "CVRP",
    "ACVRPTW": "VRPTW",
    "MDCVRP": "CVRP",
    "MDCVRPTW": "VRPTW",
    "AMDCVRP": "CVRP",
    "AMDCVRPTW": "VRPTW",
    "PDCVRP": "CVRP",
    "OPDCVRP": "OVRP",
    "APDCVRP": "CVRP",
    "AOPDCVRP": "OVRP",
    "OP": "TSP",
    "PCTSP": "TSP",
    "SPCTSP": "TSP",
}

MVRP_POOL_PROXY = {
    "ACVRP": "VRPL",
    "ACVRPTW": "VRPTW",
    "MDCVRP": "VRPL",
    "MDCVRPTW": "VRPTW",
    "AMDCVRP": "VRPL",
    "AMDCVRPTW": "VRPTW",
    "PDCVRP": "VRPL",
    "OPDCVRP": "OVRP",
    "APDCVRP": "VRPL",
    "AOPDCVRP": "OVRP",
    "OP": "TSP",
    "PCTSP": "TSP",
    "SPCTSP": "TSP",
}


def zero_constraint_bits(problem: str) -> List[int]:
    """R25-compatible C/O/B/L/TW signature for unseen zero-shot tasks."""
    if problem in {"OP", "PCTSP", "SPCTSP"}:
        return [0, 0, 0, 0, 0]
    bits = [1, 0, 0, 0, 0]
    bits[1] = int(problem.startswith("O") or "OPDCVRP" in problem or problem == "AOPDCVRP")
    bits[4] = int(problem.endswith("TW") or "TW" in problem)
    return bits


def proxy_problem(problem: str, strategy: str) -> str:
    if strategy == "semantic":
        return SEMANTIC_PROXY[problem]
    if strategy == "mvrp_pool":
        return MVRP_POOL_PROXY[problem]
    raise ValueError(f"Unknown proxy strategy: {strategy}")


def solver_alias_ids(reld_alias: str) -> List[int]:
    names = ["MTPOMO", "MVMOE", reld_alias]
    missing = [name for name in names if name not in S2I]
    if missing:
        raise ValueError(f"Solver alias not in selector vocabulary: {missing}; vocab={GLOBAL_SOLVERS}")
    return [S2I[name] for name in names]


def _as_float_tensor(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def build_coord_node(inst: dict, cbits: List[int], feature_mode: str) -> torch.Tensor:
    if "xy" in inst:
        xy = _as_float_tensor(inst["xy"])
    else:
        dep = _as_float_tensor(inst.get("depot_xy"))
        node = _as_float_tensor(inst.get("node_xy"))
        xy = torch.cat([dep, node], dim=0)
    n = xy.shape[0]
    feat = torch.zeros(n, 8, dtype=torch.float32)
    feat[:, 0:2] = xy[:, 0:2]

    depot_count = 1
    if "depot_xy" in inst:
        depot_count = int(_as_float_tensor(inst["depot_xy"]).shape[0])
    feat[:depot_count, 3] = 1.0

    if "demand" in inst:
        dem = _as_float_tensor(inst["demand"]).flatten()
        feat[: min(n, dem.numel()), 2] = dem[: min(n, dem.numel())]
    elif feature_mode == "udr_project":
        # R25 was not trained on prize/penalty fields; this projection is a
        # compatibility probe, not a trained URS-style representation.
        val = None
        if "prize" in inst:
            val = _as_float_tensor(inst["prize"]).flatten()
        if "penalty" in inst:
            pen = _as_float_tensor(inst["penalty"]).flatten()
            val = -pen if val is None else (val - pen)
        if val is not None:
            feat[: min(n, val.numel()), 2] = val[: min(n, val.numel())]

    if cbits[3] and "route_limit" in inst:
        feat[:, 4] = float(_as_float_tensor(inst["route_limit"]).reshape(-1)[0])
    if cbits[4] and {"tw_start", "tw_end"}.issubset(inst.keys()):
        tws = _as_float_tensor(inst["tw_start"]).flatten()
        twe = _as_float_tensor(inst["tw_end"]).flatten()
        if "service_time" in inst:
            st = _as_float_tensor(inst["service_time"]).flatten()
            feat[: min(n, st.numel()), 5] = st[: min(n, st.numel())]
        feat[: min(n, tws.numel()), 6] = tws[: min(n, tws.numel())]
        feat[: min(n, twe.numel()), 7] = twe[: min(n, twe.numel())]
    return feat


class ZeroShotProblemDataset(Dataset):
    def __init__(
        self,
        bundle_root: Path,
        problem: str,
        proxy_strategy: str,
        reld_alias: str,
        feature_mode: str,
    ):
        self.bundle_root = Path(bundle_root)
        self.problem = problem
        self.proxy = proxy_problem(problem, proxy_strategy)
        self.proxy_pid = P2I[self.proxy]
        self.cbits = zero_constraint_bits(problem)
        self.problem_desc = problem_descriptor(problem)
        self.solver_ids = solver_alias_ids(reld_alias)
        self.feature_mode = feature_mode
        d = self.bundle_root / "data" / f"{problem}test"
        with open(d / "dataset.pkl", "rb") as f:
            self.instances = pickle.load(f)
        with open(d / "raw_label.pkl", "rb") as f:
            raw = pickle.load(f)
        self.labels = {k: {"cost": v["cost"], "ind": v["ind"]} for k, v in raw.items()}

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        inst = self.instances[idx]
        lbl = self.labels[str(idx)]
        mask = torch.zeros(M_GLOBAL, dtype=torch.float32)
        mask[self.solver_ids] = 1.0
        input_kind = inst.get("input_kind", "matrix" if "dist" in inst else "coord")
        if input_kind == "matrix" or "dist" in inst:
            kind = "matrix"
            mat = _as_float_tensor(inst["dist"])
            node = None
            n = int(mat.shape[0])
        else:
            kind = "coord"
            node = build_coord_node(inst, self.cbits, self.feature_mode)
            mat = None
            n = int(node.shape[0])
        return {
            "problem_id": self.proxy_pid,
            "problem_name": self.problem,
            "proxy_problem": self.proxy,
            "kind": kind,
            "node": node,
            "matrix": mat,
            "cbits": torch.tensor(self.cbits, dtype=torch.float32),
            "problem_desc": torch.tensor(self.problem_desc, dtype=torch.float32),
            "coord_dist": D_COORD - 1,
            "pool_global_ids": torch.tensor(self.solver_ids, dtype=torch.long),
            "mask": mask,
            "costs": torch.tensor(lbl["cost"][: len(ZERO_SOLVERS)], dtype=torch.float32),
            "ind": int(lbl["ind"]),
            "n": n,
        }


def collate_zero(batch: List[dict]) -> dict:
    kind = batch[0]["kind"]
    pid = batch[0]["problem_id"]
    pool_ids = batch[0]["pool_global_ids"].clone()
    cbits = torch.stack([b["cbits"] for b in batch])
    problem_desc = torch.stack([b["problem_desc"] for b in batch])
    cd = torch.tensor([b["coord_dist"] for b in batch], dtype=torch.long)
    mask = torch.stack([b["mask"] for b in batch])
    costs = torch.stack([b["costs"] for b in batch])
    ind = torch.tensor([b["ind"] for b in batch], dtype=torch.long)
    n_arr = torch.tensor([b["n"] for b in batch], dtype=torch.long)
    base = {
        "problem_id": pid,
        "kind": kind,
        "cbits": cbits,
        "problem_desc": problem_desc,
        "coord_dist": cd,
        "pool_ids": pool_ids,
        "mask": mask,
        "costs": costs,
        "ind": ind,
        "n": n_arr,
    }
    if kind == "coord":
        max_n = int(n_arr.max().item())
        d_in = batch[0]["node"].shape[1]
        node = torch.zeros(len(batch), max_n, d_in, dtype=torch.float32)
        node_mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
        for i, b in enumerate(batch):
            nn = b["n"]
            node[i, :nn] = b["node"]
            node_mask[i, :nn] = True
        base.update({"node": node, "node_mask": node_mask, "d_in": d_in})
    else:
        max_n = int(n_arr.max().item())
        mat = torch.zeros(len(batch), max_n, max_n, dtype=torch.float32)
        node_mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
        for i, b in enumerate(batch):
            nn = b["n"]
            mat[i, :nn, :nn] = b["matrix"]
            node_mask[i, :nn] = True
        base.update({"matrix": mat, "node_mask": node_mask})
    return base


def apply_meta_only(batch: dict, meta_only: bool) -> dict:
    if not meta_only:
        return batch
    if batch["kind"] == "coord":
        batch["node"] = torch.zeros_like(batch["node"])
    else:
        batch["matrix"] = torch.zeros_like(batch["matrix"])
    return batch


def load_model(ckpt_path: Path, device: str) -> tuple[UnifiedSelector, dict, bool]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt.get("args", {})
    has_bias = "problem_solver_bias" in ckpt["model"]
    model = UnifiedSelector(
        d=model_args.get("d", 128),
        depth=model_args.get("depth", 4),
        dropout=model_args.get("dropout", 0.1),
        use_mvrp_factorized=not model_args.get("no_fact", False),
        use_problem_solver_bias=has_bias,
        use_problem_film=bool(model_args.get("problem_film", False)),
        use_size_feature=bool(model_args.get("size_feature", False)),
        rich_pool=bool(model_args.get("rich_pool", False)),
        use_global_stats=bool(model_args.get("global_stats", False)),
        use_manual_features=bool(model_args.get("manual_features", False)),
        use_constraint_experts=bool(model_args.get("constraint_experts", False)),
        use_problem_residual_head=bool(model_args.get("problem_residual_head", False)),
        use_problem_adapter=bool(model_args.get("problem_adapter", False)),
        use_support_head=bool(model_args.get("support_head", False)),
        support_hidden=int(model_args.get("support_hidden", 128)),
        support_generators=int(model_args.get("support_generators", 1)),
        adapter_hidden=int(model_args.get("adapter_hidden", 64)),
        coord_hier_pool=bool(model_args.get("coord_hier_pool", False)),
        coord_downsample_ratio=float(model_args.get("coord_downsample_ratio", 0.8)),
        deep_encoder_overhaul=bool(model_args.get("deep_encoder_overhaul", False)),
        encoder_rezero=bool(model_args.get("encoder_rezero", False)),
        encoder_constraint_experts=bool(model_args.get("encoder_constraint_experts", False)),
        encoder_constraint_hidden=int(model_args.get("encoder_constraint_hidden", 128)),
        use_problem_descriptor=bool(model_args.get("problem_descriptor", False)),
        use_descriptor_solver_bias=bool(model_args.get("descriptor_solver_bias", False)),
    ).to(device)
    missing, unexpected = model.load_state_dict(remap_legacy_state_dict(ckpt["model"]), strict=False)
    if missing or unexpected:
        print(f"[load] missing={missing} unexpected={unexpected}")
    model.eval()
    return model, model_args, bool(model_args.get("meta_only", False))


def evaluate_problem(
    model: UnifiedSelector,
    model_args: dict,
    ds: ZeroShotProblemDataset,
    batch_size: int,
    device: str,
    meta_only: bool,
    tta: int,
    no_support: bool,
) -> dict:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_zero)
    use_support_head = bool(model_args.get("support_head", False)) and not no_support
    support_threshold = float(model_args.get("support_threshold", 0.5))
    support_topk = int(model_args.get("support_topk", 3))
    all_pred, all_score, all_costs, all_shortlist = [], [], [], []
    for b in dl:
        for k, v in list(b.items()):
            if torch.is_tensor(v):
                b[k] = v.to(device)
        b = apply_meta_only(b, meta_only)
        if tta > 1 and b["kind"] == "coord":
            logits = None
            support_logits = None
            n_tta = min(max(1, tta), 8)
            for aug_idx in range(n_tta):
                aug_b = dict(b)
                aug_node = b["node"].clone()
                aug_node[:, :, 0:2] = augment_xy_by_8_fold(aug_node[:, :, 0:2], aug_idx)
                aug_b["node"] = aug_node
                if use_support_head:
                    cur, cur_support = model(aug_b, return_support=True)
                else:
                    cur = model(aug_b)
                    cur_support = None
                logits = cur if logits is None else logits + cur
                if cur_support is not None:
                    support_logits = cur_support if support_logits is None else support_logits + cur_support
            logits = logits / float(n_tta)
            if support_logits is not None:
                support_logits = support_logits / float(n_tta)
        else:
            if use_support_head:
                logits, support_logits = model(b, return_support=True)
            else:
                logits = model(b)
                support_logits = None
        logits_pool = logits[:, b["pool_ids"]]
        if support_logits is not None:
            support_pool = (
                support_logits[:, b["pool_ids"]]
                if support_logits.dim() == 2
                else support_logits[:, :, b["pool_ids"]]
            )
        else:
            support_pool = None
        effective_logits_pool, shortlist, _ = shortlist_from_support(
            logits_pool,
            support_pool,
            threshold=support_threshold,
            topk=support_topk,
        )
        pred = F.log_softmax(effective_logits_pool, dim=1).argmax(dim=1)
        all_pred.append(pred.detach().cpu().numpy())
        all_score.append(effective_logits_pool.detach().cpu().numpy())
        all_costs.append(b["costs"].detach().cpu().numpy())
        if shortlist is not None:
            all_shortlist.append(shortlist.detach().cpu().numpy())

    pred = np.concatenate(all_pred)
    score = np.concatenate(all_score)
    costs = np.concatenate(all_costs)
    shortlist = np.concatenate(all_shortlist) if all_shortlist else None
    n = costs.shape[0]
    k_p = costs.shape[1]
    true_rank = np.argsort(costs, axis=1)
    best_idx = true_rank[:, 0]
    pred_rank = np.argsort(-score, axis=1)
    top1 = float((pred == best_idx).mean())
    top2 = float(np.any(pred[:, None] == true_rank[:, : min(2, k_p)], axis=1).mean())
    top3 = float(np.any(pred[:, None] == true_rank[:, : min(3, k_p)], axis=1).mean())
    oracle_rank = np.empty(n, dtype=np.int64)
    for i in range(n):
        oracle_rank[i] = int(np.where(pred_rank[i] == best_idx[i])[0][0]) + 1
    sel_cost = costs[np.arange(n), pred]
    method_mean = {name: float(costs[:, i].mean()) for i, name in enumerate(ZERO_SOLVERS)}
    method_top1 = {name: float((best_idx == i).mean()) for i, name in enumerate(ZERO_SOLVERS)}
    sbs_idx = int(np.argmin([method_mean[name] for name in ZERO_SOLVERS]))
    sbs_name = ZERO_SOLVERS[sbs_idx]
    sbs_cost = method_mean[sbs_name]
    vbs_mean = float(costs.min(axis=1).mean())
    arm_dist = {name: float((pred == i).mean()) for i, name in enumerate(ZERO_SOLVERS)}
    hidden_winners = [name for name in ZERO_SOLVERS if method_top1[name] > 0 and arm_dist[name] <= 0]
    result = {
        "problem": ds.problem,
        "proxy_problem": ds.proxy,
        "kind": ds.instances[0].get("input_kind", "matrix" if "dist" in ds.instances[0] else "coord"),
        "N": int(n),
        "K_p": int(k_p),
        "pool_names": list(ZERO_SOLVERS),
        "global_solver_alias": {
            "MTPOMO": "MTPOMO",
            "MVMOE": "MVMOE",
            "RELD": GLOBAL_SOLVERS[ds.solver_ids[2]],
        },
        "sbs_name": sbs_name,
        "sbs_cost": float(sbs_cost),
        "vbs_mean": float(vbs_mean),
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "mean_cost": float(sel_cost.mean()),
        "vs_sbs_pct": float((sel_cost.mean() - sbs_cost) / (abs(sbs_cost) + 1e-9) * 100),
        "vbs_gap_closed_pct": float((sbs_cost - sel_cost.mean()) / (sbs_cost - vbs_mean + 1e-9) * 100)
        if sbs_cost != vbs_mean
        else 0.0,
        "method_top1": method_top1,
        "method_mean": method_mean,
        "arm_distribution": arm_dist,
        "final_arm_coverage": float(np.mean([v > 0 for v in arm_dist.values()])),
        "zero_pick_count": int(sum(v <= 0 for v in arm_dist.values())),
        "hidden_winners": hidden_winners,
        "hidden_winner_count": int(len(hidden_winners)),
        "hidden_winner_mass": float(sum(method_top1[name] for name in hidden_winners)),
        "oracle_rank_hist": {
            str(rank): int((oracle_rank == rank).sum())
            for rank in range(1, k_p + 1)
        },
    }
    if shortlist is not None:
        result["support_top1_recall"] = float(shortlist[np.arange(n), best_idx].mean())
        result["support_arm_coverage"] = float(np.mean(shortlist.mean(axis=0) > 0))
    return result


def macro_metrics(per_problem: Dict[str, dict]) -> dict:
    vals = list(per_problem.values())
    return {
        "macro_top1": float(np.mean([v["top1"] for v in vals])),
        "macro_top2": float(np.mean([v["top2"] for v in vals])),
        "macro_top3": float(np.mean([v["top3"] for v in vals])),
        "macro_vs_sbs_pct": float(np.mean([v["vs_sbs_pct"] for v in vals])),
        "macro_vbs_gap_closed_pct": float(np.mean([v["vbs_gap_closed_pct"] for v in vals])),
        "macro_final_arm_coverage": float(np.mean([v["final_arm_coverage"] for v in vals])),
        "macro_hidden_winner_mass": float(np.mean([v["hidden_winner_mass"] for v in vals])),
        "n_problems_beat_sbs": int(sum(v["vs_sbs_pct"] < 0 for v in vals)),
        "n_problems_match_sbs": int(sum(abs(v["vs_sbs_pct"]) < 0.02 for v in vals)),
        "micro_top1": float(
            sum(v["top1"] * v["N"] for v in vals) / max(1, sum(v["N"] for v in vals))
        ),
        "micro_mean_cost": float(
            sum(v["mean_cost"] * v["N"] for v in vals) / max(1, sum(v["N"] for v in vals))
        ),
    }


def write_csv(per_problem: Dict[str, dict], out: Path) -> None:
    fields = [
        "problem",
        "proxy_problem",
        "kind",
        "N",
        "sbs_name",
        "top1",
        "top2",
        "top3",
        "mean_cost",
        "sbs_cost",
        "vbs_mean",
        "vs_sbs_pct",
        "vbs_gap_closed_pct",
        "arm_distribution",
        "method_top1",
        "method_mean",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in per_problem.values():
            row = {k: r[k] for k in fields if k in r}
            for k in ("arm_distribution", "method_top1", "method_mean"):
                row[k] = json.dumps(row[k], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)


def write_markdown(payload: dict, out: Path) -> None:
    macro = payload["macro"]
    per = payload["per_problem"]
    lines = [
        f"# Zero-shot R25 Evaluation ({payload['run_name']})",
        "",
        "## Setup",
        "",
        f"- Checkpoint: `{payload['ckpt']}`",
        f"- Bundle: `{payload['bundle_root']}`",
        f"- Proxy strategy: `{payload['proxy_strategy']}`",
        f"- RELD alias: `{payload['reld_alias']}`",
        f"- Feature mode: `{payload['feature_mode']}`",
        f"- TTA: `{payload['tta']}`",
        "",
        "## Macro Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for k, v in macro.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.6f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.extend([
        "",
        "## Per-problem Metrics",
        "",
        "| problem | proxy | kind | top1 | top2 | mean_cost | SBS | sbs_cost | VBS | vs_sbs% | vbs_closed% | picks [MTPOMO/MVMOE/RELD] | oracle [MTPOMO/MVMOE/RELD] |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for p, r in per.items():
        picks = [r["arm_distribution"][s] for s in ZERO_SOLVERS]
        oracle = [r["method_top1"][s] for s in ZERO_SOLVERS]
        lines.append(
            f"| {p} | {r['proxy_problem']} | {r['kind']} | {r['top1']:.3f} | {r['top2']:.3f} "
            f"| {r['mean_cost']:.6f} | {r['sbs_name']} | {r['sbs_cost']:.6f} | {r['vbs_mean']:.6f} "
            f"| {r['vs_sbs_pct']:+.3f} | {r['vbs_gap_closed_pct']:+.2f} "
            f"| [{picks[0]:.3f},{picks[1]:.3f},{picks[2]:.3f}] "
            f"| [{oracle[0]:.3f},{oracle[1]:.3f},{oracle[2]:.3f}] |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- This is a frozen-checkpoint zero-shot evaluation: labels are used only for metrics.",
        "- `RELD` is not a native R25 solver token, so it is evaluated via the selected alias.",
        "- R25 is problem-id based; the proxy problem is a compatibility adapter, not URS-style learned conditioning.",
        "- For `matrix` zero-shot tasks, R25 currently consumes the distance matrix but not extra node attributes such as demand/TW.",
        "",
    ])
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle-root", default="zero_shot/selector_data/zero13")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--problems", nargs="*", default=None)
    ap.add_argument("--proxy-strategy", choices=["semantic", "mvrp_pool"], default="semantic")
    ap.add_argument("--reld-alias", choices=["RELD_MTL", "RELD_MOEL", "RELD_CVRP"], default="RELD_MTL")
    ap.add_argument("--feature-mode", choices=["drop_extra", "udr_project"], default="drop_extra")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--no-support", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, model_args, meta_only = load_model(Path(args.ckpt), args.device)
    problems = args.problems or list(ZERO13_PROBLEMS)
    per_problem: Dict[str, dict] = {}
    with torch.no_grad():
        for p in problems:
            ds = ZeroShotProblemDataset(
                Path(args.bundle_root),
                p,
                proxy_strategy=args.proxy_strategy,
                reld_alias=args.reld_alias,
                feature_mode=args.feature_mode,
            )
            r = evaluate_problem(
                model,
                model_args,
                ds,
                batch_size=args.batch_size,
                device=args.device,
                meta_only=meta_only,
                tta=args.tta,
                no_support=args.no_support,
            )
            per_problem[p] = r
            print(
                f"{p:>10} proxy={r['proxy_problem']:<6} top1={r['top1']:.3f} "
                f"top2={r['top2']:.3f} mean_cost={r['mean_cost']:.6f} "
                f"sbs={r['sbs_cost']:.6f} vs_sbs={r['vs_sbs_pct']:+.3f}% "
                f"picks={r['arm_distribution']}"
            )

    payload = {
        "run_name": out.name,
        "ckpt": str(args.ckpt),
        "bundle_root": str(args.bundle_root),
        "proxy_strategy": args.proxy_strategy,
        "reld_alias": args.reld_alias,
        "feature_mode": args.feature_mode,
        "tta": args.tta,
        "no_support": args.no_support,
        "per_problem": per_problem,
        "macro": macro_metrics(per_problem),
    }
    (out / "zero_shot_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_csv(per_problem, out / "zero_shot_per_problem.csv")
    write_markdown(payload, out / "zero_shot_summary.md")
    m = payload["macro"]
    print(
        f"\nMacro: top1={m['macro_top1']:.4f} top2={m['macro_top2']:.4f} "
        f"vs_sbs={m['macro_vs_sbs_pct']:+.3f}% vbs_closed={m['macro_vbs_gap_closed_pct']:+.2f}% "
        f"hidden_mass={m['macro_hidden_winner_mass']:.4f}"
    )
    print(f"Saved: {out / 'zero_shot_analysis.json'}")
    print(f"Saved: {out / 'zero_shot_summary.md'}")


if __name__ == "__main__":
    main()
