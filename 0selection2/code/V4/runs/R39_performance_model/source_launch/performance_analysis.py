"""Validation-only plots, paired checks and summary tables for R39."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump


METRICS = ("macro_top1", "macro_top2", "macro_top3", "macro_mean_cost", "macro_vs_sbs_pct",
           "macro_actual_regret_pct", "macro_ce", "macro_performance_mse", "macro_top1_classification",
           "macro_mean_cost_classification", "macro_vs_sbs_pct_classification")


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_run(history, directory):
    epochs = [r["epoch"] for r in history]
    plots = (
        ("loss_curve", [("train loss", [r["train"]["loss"] for r in history])]),
        ("ce_curve", [("train CE (training mode)", [r["train"]["ce"] for r in history]),
                      ("val classification CE", [r["val"]["macro_ce"] for r in history])]),
        ("top1_curve", [("main policy", [r["val"]["macro_top1"] for r in history]),
                        ("classification", [r["val"]["macro_top1_classification"] for r in history]),
                        ("performance", [r["val"]["macro_top1_performance"] for r in history])]),
        ("mean_cost_curve", [("main policy", [r["val"]["macro_mean_cost"] for r in history]),
                             ("classification", [r["val"]["macro_mean_cost_classification"] for r in history])]),
        ("vs_sbs_curve", [("main policy", [r["val"]["macro_vs_sbs_pct"] for r in history]),
                          ("classification", [r["val"]["macro_vs_sbs_pct_classification"] for r in history])]),
        ("performance_mse_curve", [("val centered MSE", [r["val"]["macro_performance_mse"] for r in history]),
                                   ("zero-prediction MSE", [r["val"]["macro_performance_zero_mse"] for r in history])]),
    )
    for filename, series in plots:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, values in series:
            ax.plot(epochs, values, label=label)
        ax.set(xlabel="Epoch", ylabel=filename.replace("_curve", ""), title=directory.name)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{filename}.png", dpi=150)
        plt.close(fig)
    for metric, filename in (("macro_top1", "family_top1"), ("macro_vs_sbs_pct", "family_vs_sbs")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for family in history[-1]["families"]:
            ax.plot(epochs, [r["families"][family][metric] for r in history], label=family)
        ax.set(xlabel="Epoch", ylabel=metric, title=directory.name)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{filename}.png", dpi=150)
        plt.close(fig)


def plot_methods(record, directory):
    for problem, r in record["per_problem"].items():
        output = directory / "method_plots" / problem
        output.mkdir(parents=True, exist_ok=True)
        names = [name for name in r["pool"] if name in r["method_mean_cost"]]
        for metric, filename in (("method_top1", "top1_vs_single_methods"), ("method_mean_cost", "mean_cost_vs_single_methods")):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(names, [r[metric][p] for p in names], color="#599eae")
            selected = r["top1"] if metric == "method_top1" else r["mean_cost"]
            ax.axhline(selected, color="#c45746", label="R39 actual policy")
            ax.set_title(f"{directory.name}: {problem}, best val epoch {record['epoch']}")
            ax.tick_params(axis="x", labelrotation=45)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"{filename}.png", dpi=150)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(names, [r["arm_distribution"][name] for name in names], color="#599eae")
        ax.set(title=f"{problem}: selected solver distribution", ylabel="Pick fraction")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(output / "arm_distribution.png", dpi=150)
        plt.close(fig)


def result_table(record):
    lines = ["| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs_SBS) | Oracle cost (vs_Oracle) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    macro = record["val"]
    per = record["per_problem"]
    sbs, oracle = np.mean([r["sbs"] for r in per.values()]), np.mean([r["vbs"] for r in per.values()])
    lines.append(f"| ALL | {macro['macro_top1']:.4f} | {macro['macro_top2']:.4f} | {macro['macro_top3']:.4f} | "
                 f"{macro['macro_mean_cost']:.6f} | {sbs:.6f} ({macro['macro_vs_sbs_pct']:+.4f}%) | "
                 f"{oracle:.6f} ({macro['macro_vs_oracle_pct']:+.4f}%) |")
    for p, r in per.items():
        lines.append(f"| {p} | {r['top1']:.4f} | {r['top2']:.4f} | {r['top3']:.4f} | {r['mean_cost']:.6f} | "
                     f"{r['sbs']:.6f} ({r['vs_sbs_pct']:+.4f}%) | {r['vbs']:.6f} ({r['vs_oracle_pct']:+.4f}%) |")
    return lines


def paired_checks(root, results):
    pairs = {}
    for seed in sorted({r["seed"] for r in results}):
        a = root / f"winner_cost_seed{seed}"
        b = root / f"performance_seed{seed}"
        if not (a / "result.json").exists() or not (b / "result.json").exists():
            continue
        aa, bb = read(a / "args.json"), read(b / "args.json")
        ha, hb = read(a / "history.json"), read(b / "history.json")
        common_keys = ("seed", "epochs", "batch_per_problem", "lr", "min_lr", "weight_decay", "warmup_epochs",
                       "lr_schedule", "coord_augment", "sampling", "performance_scales", "geometry_stats",
                       "source_hashes", "eval_precision", "amp_dtype", "sdpa", "grad_clip", "sbs_definition")
        model_a = {k: v for k, v in aa["model_params"].items() if k != "decision_head"}
        model_b = {k: v for k, v in bb["model_params"].items() if k != "decision_head"}
        checks = dict(same_initial_model=aa["model_initial_hash"] == bb["model_initial_hash"],
                      same_shared_configuration=all(aa[k] == bb[k] for k in common_keys) and model_a == model_b,
                      same_schedule=aa["schedule_hash"] == bb["schedule_hash"],
                      same_initial_task_rng=aa["dropout_rng_initial_hash"] == bb["dropout_rng_initial_hash"],
                      same_task_rng_each_epoch=all(x["task_rng_hashes"] == y["task_rng_hashes"] for x, y in zip(ha, hb)),
                      same_updates_each_epoch=all(x["updates_by_problem"] == y["updates_by_problem"] for x, y in zip(ha, hb)),
                      all_epochs_completed=len(ha) == len(hb) == aa["epochs"] == bb["epochs"],
                      no_test_read=not aa["test_read"] and not bb["test_read"],
                      scratch_initialization=not aa["weights_restored"] and not bb["weights_restored"])
        pairs[str(seed)] = checks
        if not all(checks.values()):
            raise RuntimeError(f"Paired R39 protocol check failed: seed={seed}, {checks}")
    dump(root / "paired_checks.json", pairs)
    return pairs


def summarize(root):
    root = Path(root)
    paths = sorted(root.glob("*_seed*/result.json"))
    results = [read(path) for path in paths]
    if not results:
        return
    rows = []
    for r in results:
        for point in ("best", "final", "last5"):
            values = r[point] if point == "last5" else r[point]["val"]
            rows.append(dict(group=r["group"], seed=r["seed"], point=point,
                             epoch="last5" if point == "last5" else r[point]["epoch"],
                             **{k: values[k] for k in METRICS}))
    write_csv(root / "comparison.csv", rows)
    family_rows = []
    for r in results:
        for point in ("best", "final", "last5"):
            families = r["last5_families"] if point == "last5" else r[point]["families"]
            for name, metrics in families.items():
                family_rows.append(dict(group=r["group"], seed=r["seed"], point=point, family=name,
                                        **{key: metrics[key] for key in METRICS}))
    write_csv(root / "family_comparison.csv", family_rows)
    pairs = paired_checks(root, results)
    differences = []
    for seed in (int(x) for x in pairs):
        for point in ("best", "final", "last5"):
            pair = {r["group"]: r for r in rows if r["seed"] == seed and r["point"] == point}
            differences.append(dict(seed=seed, point=point, **{k: pair["B"][k] - pair["A"][k] for k in METRICS}))
    write_csv(root / "paired_differences.csv", differences)
    lines = ["# R39 Performance Prediction", "", f"Completed runs: {len(results)}/6. All numbers below are validation, not test.",
             "A: winner-cost classification decision. B: CE + centered-cost MSE, performance decision.",
             "Classification Top1 is never combined with the performance policy's cost.",
             "The ALL percentage is a mean of per-problem percentages, not a ratio of aggregate costs.",
             "", "## Main Comparison", "", "| Group | Point | Seeds | Top1 | Mean cost | vs_SBS (%) | Actual regret (%) | Classification Top1 | Classification CE |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in ("A", "B"):
        for point in ("best", "final", "last5"):
            chosen = [r for r in rows if r["group"] == group and r["point"] == point]
            if not chosen:
                continue
            values = {k: float(np.mean([r[k] for r in chosen])) for k in METRICS}
            lines.append(f"| {group} | {point} | {len(chosen)} | {values['macro_top1']:.4f} | {values['macro_mean_cost']:.6f} | "
                         f"{values['macro_vs_sbs_pct']:+.4f} | {values['macro_actual_regret_pct']:.4f} | "
                         f"{values['macro_top1_classification']:.4f} | {values['macro_ce']:.5f} |")
    if differences:
        lines += ["", "## Paired B Minus A", "", "| Point | Top1 (pp) | Mean cost | vs_SBS (pp) | Actual regret (pp) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for point in ("best", "final", "last5"):
            chosen = [r for r in differences if r["point"] == point]
            means = {k: float(np.mean([r[k] for r in chosen])) for k in METRICS}
            lines.append(f"| {point} | {means['macro_top1']*100:+.4f} | {means['macro_mean_cost']:+.6f} | "
                         f"{means['macro_vs_sbs_pct']:+.4f} | {means['macro_actual_regret_pct']:+.4f} |")
        lines += ["", "Three seeds are independent from-scratch training runs. Paired per-seed differences are in paired_differences.csv."]
    lines += ["", "## Family Summary: Best Validation Checkpoints", "",
              "| Group | Family | Seeds | Main Top1 | Mean cost | vs_SBS (%) | Classification Top1 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in ("A", "B"):
        for family in ("TSP", "CVRP", "ATSP", "MVRP"):
            selected = [r for r in family_rows if r["group"] == group and r["family"] == family and r["point"] == "best"]
            if selected:
                mean = {key: float(np.mean([r[key] for r in selected])) for key in METRICS}
                lines.append(f"| {group} | {family} | {len(selected)} | {mean['macro_top1']:.4f} | {mean['macro_mean_cost']:.6f} | "
                             f"{mean['macro_vs_sbs_pct']:+.4f} | {mean['macro_top1_classification']:.4f} |")
    for r, path in zip(results, paths):
        lines += ["", f"## {path.parent.name}: Best Validation Epoch {r['best']['epoch']}", ""]
        lines.extend(result_table(r["best"]))
        lines += ["", "| Problem | Selected arm distribution |", "| --- | --- |"]
        for problem, v in r["best"]["per_problem"].items():
            text = "; ".join(f"{name}: {rate*100:.1f}%" for name, rate in v["arm_distribution"].items())
            lines.append(f"| {problem} | {text} |")
        if not (path.parent / "method_plots").exists():
            plot_methods(r["best"], path.parent)
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    for key, filename in (("macro_top1", "comparison_top1"), ("macro_vs_sbs_pct", "comparison_vs_sbs"),
                          ("macro_mean_cost", "comparison_mean_cost"), ("macro_ce", "comparison_ce"),
                          ("macro_performance_mse", "comparison_performance_mse")):
        fig, ax = plt.subplots(figsize=(8, 4))
        for group, color in (("A", "#c45746"), ("B", "#287e91")):
            curves = [read(path.parent / "history.json") for r, path in zip(results, paths) if r["group"] == group]
            if not curves:
                continue
            values = np.array([[h["val"][key] for h in history] for history in curves])
            epochs = np.arange(1, values.shape[1]+1)
            ax.plot(epochs, values.mean(0), color=color, label=f"{group}, n={len(curves)}")
            ax.fill_between(epochs, values.mean(0)-values.std(0), values.mean(0)+values.std(0), color=color, alpha=.15)
        ax.set(xlabel="Epoch", ylabel=key, title="R39 validation: mean +/- seed standard deviation")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / f"{filename}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("code/V4/runs/R39_performance_model"))
    summarize(parser.parse_args().root)
