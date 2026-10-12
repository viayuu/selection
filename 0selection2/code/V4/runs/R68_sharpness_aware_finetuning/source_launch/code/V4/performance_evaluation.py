"""R39 evaluation keeps classification CE and the two decision policies separate."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import P2I
from .performance_targets import make_targets, read_raw_costs
from .tensor_loader import TensorBatchLoader


POLICY_METRICS = ("top1", "top2", "top3", "top1_tie_aware", "mean_cost", "vs_sbs_pct",
                  "vs_oracle_pct", "actual_regret_pct", "vbs_gap_closed_pct")


def decision_metrics(scores, raw, mask=None):
    costs, winner = raw["costs"], raw["winner"]
    if mask is None:
        mask = np.ones_like(costs, dtype=bool)
    if not mask[np.arange(len(winner)), winner].all():
        raise ValueError("Native winners must belong to the valid pool")
    scores = np.where(mask, scores, -np.inf)
    rank = np.argsort(-scores, axis=1, kind="stable")
    pred = rank[:, 0]
    selected = costs[np.arange(len(pred)), pred]
    oracle = np.where(mask, costs, np.inf).min(1)
    common = mask.all(0)
    if not common.any():
        raise ValueError("SBS requires at least one solver available on every evaluation instance")
    sbs_idx = int(np.where(common, np.where(mask, costs, 0).mean(0), np.inf).argmin())
    sbs, vbs, mean_cost = float(costs.mean(0)[sbs_idx]), float(oracle.mean()), float(selected.mean())
    result = dict(n=len(pred), mean_cost=mean_cost, sbs=sbs, vbs=vbs, oracle_cost=vbs, sbs_cost=sbs,
                  sbs_name=raw["pool"][sbs_idx], top1_tie_aware=float((selected == oracle).mean()),
                  vs_sbs_pct=(mean_cost / sbs - 1) * 100, vs_oracle_pct=(mean_cost / vbs - 1) * 100,
                  actual_regret_pct=float(((selected - oracle) / oracle * 100).mean()),
                  vbs_gap_closed_pct=(sbs - mean_cost) / (sbs - vbs) * 100 if sbs != vbs else 0.0,
                  **{f"top{k}": float((rank[:, :k] == winner[:, None]).any(1).mean()) for k in (1, 2, 3)})
    result["pick_dist"] = (np.bincount(pred, minlength=costs.shape[1]) / len(pred)).tolist()
    result["arm_distribution"] = dict(zip(raw["pool"], result["pick_dist"]))
    return result, pred


def metrics_from_outputs(logits, prediction, raw, scale, decision_head, mask=None):
    mask = np.ones_like(logits, dtype=bool) if mask is None else mask
    if not mask.any(1).all() or not np.isfinite(logits[mask]).all() or not np.isfinite(prediction[mask]).all():
        raise RuntimeError("Nonfinite R39 evaluation output")
    logits = np.where(mask, logits, -np.inf)
    ce = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(raw["winner"])))
    target = make_targets(raw["costs"], scale, mask).numpy()
    values = np.where(mask, prediction, 0)
    counts = mask.sum(1)
    centered = values - (values.sum(1) / counts)[:, None]
    prediction = np.where(mask, centered, np.inf)
    squared = np.where(mask, (np.where(mask, prediction, 0).astype(np.float64) - target.astype(np.float64)) ** 2, 0)
    mse = float((squared.sum(1) / counts).mean())
    target_energy = float((np.square(target.astype(np.float64)).sum(1) / counts).mean())
    classification, class_pred = decision_metrics(logits, raw, mask)
    performance, perf_pred = decision_metrics(-prediction, raw, mask)
    main = dict(classification if decision_head == "classification" else performance)
    main.update(ce=ce, performance_mse=mse, performance_zero_mse=target_energy,
                performance_r2=1 - mse / target_energy if target_energy > 0 else 0.0,
                decision_head=decision_head, pool=raw["pool"], classification=classification, performance=performance)
    main["method_top1"] = dict(zip(raw["pool"], (np.bincount(raw["winner"], minlength=len(raw["pool"])) / len(raw["winner"])).tolist()))
    main["method_mean_cost"] = {name: float(raw["costs"][:, j].mean()) for j, name in enumerate(raw["pool"]) if mask[:, j].all()}
    for name, result in (("classification", classification), ("performance", performance)):
        for key in POLICY_METRICS:
            main[f"{key}_{name}"] = result[key]
    main["top1_exceeds_methods"] = [name for name, value in main["method_top1"].items() if main["top1"] > value]
    main["top1_not_exceeds_methods"] = [name for name, value in main["method_top1"].items() if main["top1"] <= value]
    main["cost_exceeds_methods"] = [name for name, value in main["method_mean_cost"].items() if main["mean_cost"] < value]
    main["cost_not_exceeds_methods"] = [name for name, value in main["method_mean_cost"].items() if main["mean_cost"] >= value]
    return main, class_pred, perf_pred


@torch.no_grad()
def evaluate_one(model, loader, raw, problem, path=None):
    from .train import to_device
    if isinstance(loader, TensorBatchLoader):
        loader = TensorBatchLoader(loader.batch, loader.batch_size, False, False)
    logits, prediction, masks = [], [], []
    device = next(model.parameters()).device
    for batch in loader:
        out = model(to_device(batch, device))
        logits.append(out["logits"].float().cpu())
        prediction.append(out["pred_performance"].float().cpu())
        masks.append(out["solver_mask"].cpu())
    logits = torch.cat(logits).numpy()
    prediction = torch.cat(prediction).numpy()
    mask = torch.cat(masks).numpy()
    if len(logits) != len(raw["winner"]):
        raise ValueError("Evaluation must cover every instance, including the tail batch")
    scale = float(model.performance_scales[P2I[problem]])
    result, class_pred, perf_pred = metrics_from_outputs(logits, prediction, raw, scale, model.decision_head, mask)
    result["problem"] = problem
    if path is not None:
        main_pred = class_pred if model.decision_head == "classification" else perf_pred
        np.savez_compressed(path, indices=np.arange(len(logits)), logits=logits, pred_performance=prediction,
                            pred=main_pred, classification_pred=class_pred, performance_pred=perf_pred,
                            winner=raw["winner"], costs=raw["costs"], pool_ids=np.array(raw["pool_ids"]),
                            scale=np.array(scale), decision_head=np.array(model.decision_head), solver_mask=mask)
    return result


def aggregate(per_problem):
    keys = ("ce", "performance_mse", "performance_zero_mse", "performance_r2", *POLICY_METRICS)
    keys += tuple(f"{key}_{name}" for name in ("classification", "performance") for key in POLICY_METRICS)
    return {"macro_" + key: float(np.mean([r[key] for r in per_problem.values()])) for key in keys}


def family_metrics(per_problem):
    groups = {p: [p] for p in ("TSP", "CVRP", "ATSP") if p in per_problem}
    mvrp = [p for p in per_problem if p not in ("TSP", "CVRP", "ATSP")]
    if mvrp:
        groups["MVRP"] = mvrp
    return {name: aggregate({p: per_problem[p] for p in problems}) for name, problems in groups.items()}


@torch.no_grad()
def evaluate_performance(model, problems, split, batch_size, num_workers, device, cache_device=None,
                         loaders=None, raw=None, prediction_dir=None):
    from .train import make_loader
    was_training = model.training
    model.eval()
    if prediction_dir is not None:
        prediction_dir = Path(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        for problem in problems:
            loader = loaders[problem] if loaders is not None else make_loader(
                problem, split, batch_size, num_workers, shuffle=False, cache_device=cache_device)
            labels = raw[problem] if raw is not None else read_raw_costs(problem, split)
            path = None if prediction_dir is None else prediction_dir / f"{problem}.npz"
            results[problem] = evaluate_one(model, loader, labels, problem, path)
    finally:
        model.train(was_training)
    return results, aggregate(results)
