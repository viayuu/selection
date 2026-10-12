"""Train-only performance scales; raw FP64 differences precede dtype conversion."""

import pickle

import numpy as np
import torch

from code.unified_selector.registry import DATA_ROOT, POOLS, problem_to_pool_mask
from .multitask_probe import file_hash


def center_valid(values, mask=None):
    if mask is None:
        return values - values.mean(-1, keepdim=True)
    values = values.masked_fill(~mask, 0)
    mean = values.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
    return (values - mean).masked_fill(~mask, 0)


def performance_mse(prediction, target, mask=None):
    if mask is None:
        mask = torch.ones_like(prediction, dtype=torch.bool)
    prediction = center_valid(prediction.float(), mask)
    target = target.float().masked_fill(~mask, 0)
    squared = (prediction - target).square().masked_fill(~mask, 0)
    return (squared.sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def fit_scale(costs, mask=None):
    values = np.asarray(costs)
    if values.dtype != np.float64:
        raise ValueError("fit_scale requires original FP64 costs, not a recast FP32 cache")
    if mask is None:
        mask = np.ones_like(values, dtype=bool)
    counts = mask.sum(1)
    if (counts == 0).any() or not np.isfinite(values[mask]).all():
        raise ValueError("Every instance must have finite valid costs and a nonempty pool")
    mean = np.where(mask, values, 0).sum(1, keepdims=True) / counts[:, None]
    centered = np.where(mask, values - mean, 0)
    scale = float(np.sqrt((np.square(centered).sum(1) / counts).mean()))
    return scale if scale > 0 else 1.0


def make_targets(costs, scale, mask=None):
    if np.asarray(costs).dtype != np.float64 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Targets require original FP64 costs and a positive training scale")
    values = torch.from_numpy(np.asarray(costs))
    valid = None if mask is None else torch.from_numpy(mask)
    return (center_valid(values, valid) / scale).float()


def read_raw_costs(problem, split):
    directory = DATA_ROOT / f"{problem}{split}"
    path = directory / "raw_label.pkl"
    with path.open("rb") as stream:
        labels = pickle.load(stream)
    pool = list(POOLS[problem])
    rows = [labels[str(i)] for i in range(len(labels))]
    if any(len(row["cost"]) != len(pool) for row in rows):
        raise ValueError(f"{problem}/{split}: label lengths disagree with the current registry")
    result_files = sorted((directory / "results").glob("result_*.txt"))
    if result_files and [p.name[len("result_"):-len(".txt")] for p in result_files] != pool:
        raise ValueError(f"{problem}/{split}: result file order disagrees with the training pool")
    costs = np.array([row["cost"] for row in rows], dtype=np.float64)
    winner = np.array([row["ind"] for row in rows], dtype=np.int64)
    if not np.isfinite(costs).all() or (costs <= 0).any() or (winner < 0).any() or (winner >= len(pool)).any():
        raise ValueError(f"{problem}/{split}: invalid cost vector or native winner")
    return dict(costs=costs, winner=winner, pool=pool, pool_ids=problem_to_pool_mask(problem)[0],
                label_hash=file_hash(path), data_hash=file_hash(directory / "dataset.pkl"))


def fit_training_scales(problems):
    raw, metadata = {}, {}
    for problem in problems:
        raw[problem] = read_raw_costs(problem, "train")
        record = raw[problem]
        scale = fit_scale(record["costs"])
        metadata[problem] = dict(scale=scale, pool=record["pool"], pool_ids=record["pool_ids"],
                                 instances=len(record["winner"]), fitted_on="train", dtype="raw FP64",
                                 label_hash=record["label_hash"], data_hash=record["data_hash"])
    return raw, metadata
