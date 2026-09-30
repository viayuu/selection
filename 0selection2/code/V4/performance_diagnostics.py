"""Post-hoc validation diagnostics against train-only solver cost priors."""

import argparse
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .performance_analysis import write_csv
from .performance_evaluation import decision_metrics
from .performance_targets import read_raw_costs


def centered_targets(costs, scale):
    return (costs - costs.mean(1, keepdims=True)) / scale


def summarize(root):
    scales = json.loads((root / "performance_scales.json").read_text())
    raw = {p: read_raw_costs(p, "val") for p in PROBLEMS}
    priors, baseline = {}, []
    for p in PROBLEMS:
        train = read_raw_costs(p, "train")
        priors[p] = centered_targets(train["costs"], scales[p]["scale"]).mean(0)
        prediction = np.broadcast_to(priors[p], raw[p]["costs"].shape)
        metrics, _ = decision_metrics(-prediction, raw[p])
        target = centered_targets(raw[p]["costs"], scales[p]["scale"])
        baseline.append(dict(problem=p, top1=metrics["top1"], mean_cost=metrics["mean_cost"],
                             vs_sbs_pct=metrics["vs_sbs_pct"], mse=float(np.square(prediction - target).mean())))
    write_csv(root / "train_mean_baseline.csv", baseline)
    rows = []
    for directory in sorted(root.glob("*_seed*")):
        if not (directory / "result.json").exists():
            continue
        result = json.loads((directory / "result.json").read_text())
        for point in ("best", "final"):
            epoch = result[point]["epoch"]
            for p, fixed in zip(PROBLEMS, baseline):
                with np.load(directory / "predictions" / f"val_e{epoch:03d}" / f"{p}.npz") as values:
                    target = centered_targets(raw[p]["costs"], scales[p]["scale"])
                    prediction = values["pred_performance"].astype(np.float64)
                    prediction -= prediction.mean(1, keepdims=True)
                    mse = float(np.square(prediction - target).mean())
                    class_pred, perf_pred = values["classification_pred"], values["performance_pred"]
                    costs, winner = raw[p]["costs"], raw[p]["winner"]
                    indices = np.arange(len(winner))
                    class_cost, perf_cost = costs[indices, class_pred], costs[indices, perf_pred]
                    class_correct, perf_correct = class_pred == winner, perf_pred == winner
                    row = dict(run=directory.name, group=result["group"], seed=result["seed"], point=point,
                               epoch=epoch, problem=p, performance_head_trained=result["group"] == "B",
                               performance_mse=mse, train_mean_mse=fixed["mse"],
                               mse_vs_train_mean=mse - fixed["mse"],
                               r2_vs_train_mean=1 - mse / fixed["mse"] if fixed["mse"] else 0.,
                               policy_disagreement=float((class_pred != perf_pred).mean()),
                               performance_top1=float(perf_correct.mean()), classification_top1=float(class_correct.mean()),
                               performance_cost_minus_classification=float((perf_cost - class_cost).mean()),
                               recovered_classification_errors=int((~class_correct & perf_correct).sum()),
                               harmed_classification_correct=int((class_correct & ~perf_correct).sum()))
                    sorted_cost = np.sort(costs, axis=1)
                    gap_pct = (sorted_cost[:, 1] - sorted_cost[:, 0]) / sorted_cost[:, 0] * 100
                    for cutoff, name in ((.1, "gap_gt_01"), (.5, "gap_gt_05")):
                        selected = gap_pct > cutoff
                        row[name + "_n"] = int(selected.sum())
                        row[name + "_classification_top1"] = float(class_correct[selected].mean()) if selected.any() else None
                        row[name + "_performance_top1"] = float(perf_correct[selected].mean()) if selected.any() else None
                    rows.append(row)
    write_csv(root / "performance_diagnostics.csv", rows)
    dump(root / "performance_diagnostics.json", dict(test_read=False, train_prior_fitted_on="train raw FP64 costs only",
                                                     untrained_A_performance_head_is_not_a_candidate_policy=True,
                                                     baseline=baseline, rows=rows))
    print(f"[performance diagnostics] completed runs={len(rows)//36}, validation only", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("code/V4/runs/R39_performance_model"))
    summarize(parser.parse_args().root)
