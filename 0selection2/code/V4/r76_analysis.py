"""Summarize saved R76 metrics; never load a model or run another TEST."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code.unified_selector.registry import PROBLEMS


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(split, value, epoch):
    groups = [("ALL", value["macro"], value["native_total"], True)]
    for scope, metrics in value["families"].items():
        tasks = PROBLEMS[3:] if scope == "MVRP" else [scope]
        n = sum(value["per_problem"][p]["n"] for p in tasks)
        if scope == "MVRP":
            groups.append((scope, metrics, n, True))
    groups.extend((problem, metrics, metrics["n"], False)
                  for problem, metrics in value["per_problem"].items())
    rows = []
    for scope, metrics, n, macro in groups:
        prefix = "macro_" if macro else ""
        rows.append(dict(model="R76_R45A", split=split, scope=scope, n=n, epoch=epoch,
            top1_pct=100 * metrics[prefix + "top1"], top2_pct=100 * metrics[prefix + "top2"],
            top3_pct=100 * metrics[prefix + "top3"], ce=metrics[prefix + "ce"],
            mean_cost=metrics[prefix + "mean_cost"], vs_train_sbs_pct=metrics[prefix + "vs_sbs_pct"],
            vs_oracle_pct=metrics[prefix + "vs_oracle_pct"],
            actual_regret_pct=metrics[prefix + "actual_regret_pct"]))
    return rows


def fixed_baselines(root, baselines):
    directory = root / "evaluation/test/selected_best"
    grouped = {name: {} for name in ("train_sbs", "train_majority", "train_min_mean_regret")}
    fields = dict(train_sbs="sbs_solver", train_majority="majority_winner_solver",
                  train_min_mean_regret="minimum_mean_relative_regret_solver")
    for path in sorted(directory.glob("rank*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            for problem in PROBLEMS:
                prefix = problem + "__"
                if prefix + "costs" not in saved:
                    continue
                costs, winner = saved[prefix + "costs"], saved[prefix + "winner"]
                pool = saved[prefix + "pool"].tolist()
                oracle = costs.min(axis=1)
                sbs_cost = costs[:, pool.index(baselines[problem]["sbs_solver"])].mean()
                for strategy, field in fields.items():
                    name = baselines[problem][field]
                    slot = pool.index(name)
                    selected = costs[:, slot]
                    grouped[strategy][problem] = dict(
                        strategy=strategy, split="test", scope=problem, n=len(costs), solver=name,
                        top1_pct=float((winner == slot).mean() * 100), mean_cost=float(selected.mean()),
                        vs_train_sbs_pct=float((selected.mean() / sbs_cost - 1) * 100),
                        actual_regret_pct=float(((selected - oracle) / oracle).mean() * 100))
    rows = []
    for strategy, tasks in grouped.items():
        if set(tasks) != set(PROBLEMS):
            raise ValueError("The saved TEST predictions must cover all 18 tasks")
        rows.extend(tasks[problem] for problem in PROBLEMS)
        all_row = dict(strategy=strategy, split="test", scope="ALL", n=sum(r["n"] for r in tasks.values()),
                       solver="TRAIN-fitted method per problem")
        for key in ("top1_pct", "mean_cost", "vs_train_sbs_pct", "actual_regret_pct"):
            all_row[key] = float(np.mean([r[key] for r in tasks.values()]))
        rows.append(all_row)
    return rows


def method_observations(root, best, test, baselines):
    rows, diagnostics, plotted = [], [], {}
    tags = dict(val="epoch{:03d}".format(best["epoch"]), test="selected_best")
    values = dict(val=best["val"], test=test["result"])
    directory = root / "per_problem"
    directory.mkdir(exist_ok=True)
    for split, tag in tags.items():
        for path in sorted((root / "evaluation" / split / tag).glob("rank*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                for problem in PROBLEMS:
                    prefix = problem + "__"
                    if prefix + "costs" not in saved:
                        continue
                    costs, winner = saved[prefix + "costs"], saved[prefix + "winner"]
                    pool = saved[prefix + "pool"].tolist()
                    metrics = values[split]["per_problem"][problem]
                    ranking = np.argsort(costs, axis=1, kind="stable")
                    oracle = costs.min(axis=1)
                    mean = costs.mean(axis=0)
                    top1 = np.asarray([(winner == slot).mean() * 100 for slot in range(len(pool))])
                    for slot, solver in enumerate(pool):
                        rows.append(dict(problem=problem, split=split, solver=solver, n=len(costs),
                            top1_pct=float(top1[slot]),
                            true_cost_top2_pct=float((ranking[:, :min(2, len(pool))] == slot).any(axis=1).mean() * 100),
                            true_cost_top3_pct=float((ranking[:, :min(3, len(pool))] == slot).any(axis=1).mean() * 100),
                            mean_cost=float(mean[slot]),
                            actual_regret_pct=float(((costs[:, slot] - oracle) / oracle).mean() * 100),
                            selector_top1_pct=100 * metrics["top1"], selector_mean_cost=metrics["mean_cost"],
                            selector_exceeds_top1=bool(100 * metrics["top1"] > top1[slot]),
                            selector_exceeds_mean_cost=bool(metrics["mean_cost"] < mean[slot])))
                    posthoc = int(mean.argmin())
                    diagnostics.append(dict(problem=problem, split=split, n=len(costs),
                        train_sbs=baselines[problem]["sbs_solver"], posthoc_split_best=pool[posthoc],
                        posthoc_best_top1_pct=float(top1[posthoc]), posthoc_best_mean_cost=float(mean[posthoc]),
                        oracle_mean_cost=float(oracle.mean()),
                        exceeds_top1=";".join(pool[i] for i in range(len(pool)) if 100 * metrics["top1"] > top1[i]),
                        not_exceeds_top1=";".join(pool[i] for i in range(len(pool)) if 100 * metrics["top1"] <= top1[i]),
                        exceeds_mean_cost=";".join(pool[i] for i in range(len(pool)) if metrics["mean_cost"] < mean[i]),
                        not_exceeds_mean_cost=";".join(pool[i] for i in range(len(pool)) if metrics["mean_cost"] >= mean[i])))
                    plotted[problem, split] = dict(pool=pool, top1=top1, mean_cost=mean,
                        selected=np.asarray(metrics["pick_dist"]) * 100,
                        winners=np.bincount(winner, minlength=len(pool)) / len(winner) * 100,
                        selector_top1=100 * metrics["top1"], selector_cost=metrics["mean_cost"])
    write_csv(root / "method_metrics.csv", rows)
    write_csv(root / "method_comparison.csv", diagnostics)
    for problem in PROBLEMS:
        for kind, filename in (("top1", "top1_vs_single_methods.png"),
                               ("mean_cost", "mean_cost_vs_single_methods.png"),
                               ("distribution", "arm_distribution.png")):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            for axis, split in zip(axes, ("val", "test")):
                value = plotted[problem, split]
                x = np.arange(len(value["pool"]))
                if kind == "distribution":
                    axis.bar(x - .18, value["selected"], width=.36, label="Selector", color="#237c68")
                    axis.bar(x + .18, value["winners"], width=.36, label="Native winner", color="#6f8192")
                    axis.set_ylabel("Instances (%)")
                else:
                    axis.bar(x, value[kind], label="Fixed method", color="#84949e")
                    score = value["selector_top1"] if kind == "top1" else value["selector_cost"]
                    axis.axhline(score, label="Selector", color="#237c68", linewidth=2)
                    axis.set_ylabel("Native Top1 (%)" if kind == "top1" else "Mean cost")
                axis.set_xticks(x)
                axis.set_xticklabels(value["pool"], rotation=35, ha="right", fontsize=9)
                axis.set_title(problem + " " + split)
                axis.legend()
                axis.grid(axis="y", alpha=.2)
            destination = directory / problem
            destination.mkdir(exist_ok=True)
            fig.savefig(destination / filename, dpi=140)
            plt.close(fig)


def plot_history(root, history):
    plots = (("macro_top1", "Top1 (%)", 100), ("macro_ce", "CE", 1),
             ("macro_actual_regret_pct", "Actual regret (%)", 1))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (metric, label, factor) in zip(axes, plots):
        for split, color in (("train", "#237c68"), ("val", "#b84d48")):
            selected = [row for row in history if row.get(split) is not None]
            axis.plot([row["successful_updates"] for row in selected],
                      [factor * row[split]["macro"][metric] for row in selected],
                      label=split, color=color, marker="." if split == "train" else None)
        axis.set(xlabel="Successful optimizer updates", ylabel=label)
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(root / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = [row["successful_updates"] for row in history]
    for key in ("ce", "cmp", "risk", "kl"):
        if all(key in row["optimization"] for row in history):
            axes[0].plot(x, [row["optimization"][key] for row in history], label=key)
    axes[0].set(xlabel="Successful optimizer updates", ylabel="Unweighted loss component")
    axes[0].legend()
    axes[1].plot(x, [row["optimization"]["lr"] for row in history])
    axes[1].set(xlabel="Successful optimizer updates", ylabel="Learning rate")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(root / "optimization_curves.png", dpi=160)
    plt.close(fig)


def summarize(root):
    root = root.expanduser().resolve()
    outcome, test, config = read(root / "result.json"), read(root / "test_results.json"), read(root / "protocol.json")
    if not test.get("complete") or not (root / "test_complete.json").exists():
        raise ValueError("Analysis requires the one completed locked TEST")
    best, history = outcome["best"], read(root / "history.json")
    rows = []
    for split, value in (("train", best["train"]), ("val", best["val"]), ("test", test["result"])):
        rows.extend(metric_rows(split, value, best["epoch"]))
    write_csv(root / "comparison.csv", rows)
    write_csv(root / "learning_curves.csv", [dict(epoch=row["epoch"], successful_updates=row["successful_updates"],
        split=split, top1_pct=100 * row[split]["macro"]["macro_top1"],
        ce=row[split]["macro"]["macro_ce"], actual_regret_pct=row[split]["macro"]["macro_actual_regret_pct"])
        for row in history for split in ("train", "val") if row.get(split) is not None])
    baselines = fixed_baselines(root, config["baselines"])
    write_csv(root / "fixed_baselines.csv", baselines)
    method_observations(root, best, test, config["baselines"])
    plot_history(root, history)
    lines = ["# R76: R45A on the New Single-ReLD Scenario", "",
        "One from-scratch seed2 model. Model and loss are R45A; batches and execution are four-GPU distributed. "
        "No original trained selector weights or old candidate cache were reused.", "",
        "Data: " + config["data_root"], "",
        "18 tasks, 1,262,052 TRAIN / 70,110 validation / 70,110 TEST rows. "
        "reld_mt is the fixed MOEL deployment; no MTL column or best-of-two costs.", "",
        "## Locked Results", "",
        "| Split | N | Top1 % | Top2 % | Top3 % | CE | Mean cost | vs TRAIN SBS % | Actual regret % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "ALL":
            lines.append("| {split} | {n} | {top1_pct:.4f} | {top2_pct:.4f} | {top3_pct:.4f} | {ce:.6f} | "
                         "{mean_cost:.6f} | {vs_train_sbs_pct:.4f} | {actual_regret_pct:.6f} |".format(**row))
    lines += ["", "Selected epoch: {}. Completed epochs: {}. Successful updates: {}. Stop: {}.".format(
        best["epoch"], outcome["epochs_completed"], outcome["successful_updates"], outcome["stop_reason"]), "",
        "Selection used validation macro_vs_sbs_pct only, with the SBS identity fitted on TRAIN. "
        "The selected checkpoint was locked before TEST cache construction and received one full TEST. "
        "There is no TEST-based epoch/rule/task-wise model selection.", "",
        "## TRAIN-Fitted References on TEST", "",
        "| Reference | Top1 % | Mean cost | Actual regret % |", "|---|---:|---:|---:|"]
    for row in baselines:
        if row["scope"] == "ALL":
            lines.append("| {strategy} | {top1_pct:.4f} | {mean_cost:.6f} | {actual_regret_pct:.6f} |".format(**row))
    lines += ["", "## Batches and Scope", "",
        "| Problem | Rank-local microbatch caps (4090,4090,3090,3090) | Global microbatch capacity | Logical batch |",
        "|---|---|---:|---:|"]
    for problem, caps in config["batch_config"].items():
        lines.append("| {} | {} | {} | {} |".format(problem, ",".join(map(str, caps)), sum(caps),
                                                   config["logical_global_batch"]))
    lines += ["", "Large batches intentionally differ from historical R45A batch128 and change update counts. "
        "LR was not automatically scaled. Each epoch still retains every real TRAIN row and tail exactly once. "
        "The whole local objective, including KL, is weighted for uneven rank row counts.", "",
        "The logical global batch is uniform across tasks. TSP/CVRP use accumulation without parameter "
        "updates between microbatches; this does not multiply optimizer updates or task sampling weight.", "",
        "The two4090 and two3090 cards trained one synchronized model, not four seeds. "
        "Input preparation was parallel; node-local arrays and pinned prefetch reduced shared-filesystem waits. "
        "W&B mode and measured capacity/timing are recorded in the protocol and capacity receipts.", "",
        "Old R45A/R75 used different splits, candidate pools and/or deployment labels; their TEST figures "
        "are not matched controls for this release. Compare this model to the references on this same scenario. "
        "Removing candidates does not certify complete remaining constraint equivalence; R58 stays deferred.", "",
        "Saved predictions are in evaluation/<split>/<tag>/rank*.npz. Full per-task results are in comparison.csv; "
        "learning_curves.png and optimization_curves.png use successful optimizer updates.", "",
        "method_metrics.csv and method_comparison.csv retain each single method, the selector's gains/losses, "
        "and the separately labelled post-hoc split-best method. Per-problem Top1, mean-cost and arm-distribution "
        "plots compare saved validation and TEST predictions. Fixed-method true_cost_top2/3 mean membership "
        "in the true cost ranking; selector Top2/3 instead mean winner coverage in its predicted ranking. "
        "Do not mix these definitions or use the post-hoc split-best identity for checkpoint/policy selection."]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    stamp = dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                 selected_epoch=best["epoch"], successful_updates=outcome["successful_updates"],
                 full_tests=1, test_top1_pct=100 * test["result"]["macro"]["macro_top1"])
    (root / "pipeline_complete.json").write_text(json.dumps(stamp, indent=2) + "\n")
    print(json.dumps(stamp), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root)


if __name__ == "__main__":
    main()
