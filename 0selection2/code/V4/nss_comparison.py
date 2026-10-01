"""Reuse frozen test predictions and evaluate missing NSS-comparison datasets."""

import argparse
import csv
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import DATA_ROOT, GLOBAL_SOLVERS, POOLS, problem_to_pool_mask
from .direct_selector import make_r40_model
from .performance_evaluation import decision_metrics
from .r41_frozen_evaluation import attach_geometry
from .solver_query import make_r42_model
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, make_loader


RUNS = Path("code/V4/runs")
OUTPUT = RUNS / "NSS_ppt_comparison"
DATASETS = {"TSP test": ("TSP", "test"), "CVRP test": ("CVRP", "test"),
            "TSPLIB": ("TSP", "TSPLIB"), "CVRPLIB": ("CVRP", "CVRPLIB")}
MODELS = [
    ("R25", "5/21 original method", "../unified_selector/runs/R25_newdata_oldcode_resume1_seed2/best.pt"),
    ("R32d", "Experiment 1: without solver features", "R32d_g0main_seqtopk_seed2_gpu0_b448/best.pt"),
    ("R32e", "Experiment 1: solver features", "R32e_solver_features_seed2_4090/best.pt"),
    ("R33a", "Experiment 2: dual-stream encoder", "R33a_dualstream_seed2_4090/best.pt"),
    ("R34", "Experiment 3: winner-cost", "R34_winner_cost_seed2_4090/best.pt"),
    ("R37A", "Experiment 5: redundant-input control", "R37_tsp_local_geometry/redundant_seed2/last.pt"),
    ("R37B", "Experiment 5: local geometry", "R37_tsp_local_geometry/geometry_seed2/last.pt"),
    ("R38A", "Experiment 5: frozen redundant-input control", "R38_tsp_frozen_geometry/redundant_seed2/last.pt"),
    ("R38B", "Experiment 5: frozen local geometry", "R38_tsp_frozen_geometry/geometry_seed2/last.pt"),
    ("R39A", "Experiment 6: winner-cost control", "R39_performance_model/winner_cost_seed2/best.pt"),
    ("R39B", "Experiment 6: performance prediction", "R39_performance_model/performance_seed2/best.pt"),
    ("R40A", "Experiment 7: small-batch dual stream", "R40_discriminative_training/dual_stream_seed2/best.pt"),
    ("R40B", "Experiment 7: direct classification", "R40_discriminative_training/direct_seed2/best.pt"),
    ("R42A", "Experiment 8: dual stream + R-Drop", "R42_relation_query/dual_stream_rdrop_seed2/best.pt"),
    ("R42B", "Experiment 8: relation graph + solver queries", "R42_relation_query/relation_query_rdrop_seed2/best.pt"),
]
REUSE = {name: RUNS / f"R41_signal_audit/test_predictions/{name}"
         for name in ("R34", "R39A", "R40A", "R40B")}
REUSE.update({"R42A": RUNS / "R42_relation_query/test_predictions/A",
              "R42B": RUNS / "R42_relation_query/test_predictions/B"})
CSV_FIELDS = ("experiment", "model_version", "checkpoint", "checkpoint_epoch", "updates", "dataset", "n",
              "top1", "top2", "top3", "mean_cost", "sbs_cost", "vs_sbs_pct", "oracle_cost", "vs_oracle_pct",
              "top1_native_ind", "top2_native_ind", "top3_native_ind", "winner_rule",
              "decision", "source", "prediction_file", "cost_difference_vs_nss")


class R25Selector(torch.nn.Module):
    def __init__(self, path, device):
        super().__init__()
        from code.unified_selector.zero_shot_eval import load_model
        self.model, self.settings, meta_only = load_model(path, device)
        if meta_only:
            raise ValueError("This comparison requires the original full R25 model")
        self.params = {"architecture": "r25"}

    def forward(self, batch):
        from code.unified_selector.model import shortlist_from_support
        logits, support = self.model(batch, return_support=True)
        ids = batch["pool_ids"]
        support = support[:, ids] if support.dim() == 2 else support[:, :, ids]
        scores, _, _ = shortlist_from_support(logits[:, ids], support,
                                              threshold=self.settings["support_threshold"],
                                              topk=self.settings["support_topk"])
        return {"logits": scores}


def read_costs(problem, split):
    directory = DATA_ROOT / (split if split.endswith("LIB") else problem + split)
    with (directory / "raw_label.pkl").open("rb") as stream:
        labels = pickle.load(stream)
    pool = list(POOLS[problem])
    files = sorted((directory / "results").glob("result_*.txt"))
    if [p.stem[len("result_"):] for p in files] != pool:
        raise ValueError(f"{directory}: solver names/order disagree with registry")
    costs = np.array([labels[str(i)]["cost"] for i in range(len(labels))], dtype=np.float64)
    winner = np.array([labels[str(i)]["ind"] for i in range(len(labels))], dtype=np.int64)
    if costs.shape != (len(winner), len(pool)) or not np.isfinite(costs).all():
        raise ValueError(f"{directory}: invalid cost table")
    return dict(costs=costs, winner=winner, pool=pool, pool_ids=problem_to_pool_mask(problem)[0],
                directory=str(directory))


def references(raw):
    means = raw["costs"].mean(0)
    idx = int(means.argmin())
    return dict(n=len(raw["winner"]), pool=raw["pool"], sbs_name=raw["pool"][idx],
                sbs_cost=float(means[idx]), oracle_cost=float(raw["costs"].min(1).mean()))


def comparable_metrics(scores, raw):
    native, pred = decision_metrics(scores, raw)
    winner = raw["costs"].astype(np.float32).argmin(1)
    result, _ = decision_metrics(scores, dict(raw, winner=winner))
    result.update({f"top{k}_native_ind": native[f"top{k}"] for k in (1, 2, 3)})
    return result, pred


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def row(name, experiment, path, dataset, metrics, source, prediction="", epoch=None, updates=None):
    return dict(experiment=experiment, model_version=name, checkpoint=str(path), checkpoint_epoch=epoch,
                updates=updates, dataset=dataset,
                **{k: metrics[k] for k in ("n", "top1", "top2", "top3", "mean_cost", "sbs_cost",
                                         "vs_sbs_pct", "oracle_cost", "vs_oracle_pct")},
                decision="performance argmin" if name == "R39B" else "greedy argmax",
                winner_rule="FP32 cost argmin (NSS historical convention)",
                **{f"top{k}_native_ind": metrics.get(f"top{k}_native_ind") for k in (1, 2, 3)},
                source=str(source), prediction_file=str(prediction),
                arm_distribution=metrics.get("arm_distribution", {}))


def historical_metrics(value, raw):
    ref = references(raw)
    if value.get("pool_names", raw["pool"]) != raw["pool"]:
        raise ValueError("Historical solver name order differs from comparison pool")
    for key, aliases in (("sbs_cost", ("sbs_cost", "sbs")), ("oracle_cost", ("oracle_cost", "vbs_mean", "vbs"))):
        stored = next((value[k] for k in aliases if k in value), ref[key])
        if abs(stored - ref[key]) > 1e-4:
            raise ValueError("Historical reference costs differ from comparison dataset")
    n = value.get("n", value.get("N", ref["n"]))
    if n != ref["n"]:
        raise ValueError("Historical instance count differs from comparison dataset")
    metrics = {k: value[k] for k in ("top1", "top2", "top3", "mean_cost")}
    metrics.update(ref, vs_sbs_pct=(value["mean_cost"] / ref["sbs_cost"] - 1) * 100,
                   vs_oracle_pct=(value["mean_cost"] / ref["oracle_cost"] - 1) * 100,
                   arm_distribution=value.get("arm_distribution", {}))
    return metrics


def write_reports(root, rows, raw, checkpoints):
    nss = {r["dataset"]: r["mean_cost"] for r in rows if r["model_version"] == "NSS"}
    for r in rows:
        r["cost_difference_vs_nss"] = r["mean_cost"] - nss[r["dataset"]]
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "results.json", rows)
    write_json(root / "checkpoints.json", checkpoints)
    write_json(root / "references.json", {d: references(r) for d, r in raw.items()})
    lines = ["# NSS 与现有模型：统一四项评估", "",
             "只补评估，不训练 selector、不重新运行底层 solver、不按 test/LIB 更换 checkpoint。",
             "全部采用 greedy 单方法选择；R39B 使用性能预测 argmin，其余使用分类/选择 logits argmax。",
             "R37/R38 仅评估 TSP test、TSPLIB，使用 seed=2 的固定 1000 更新终点；其他模型使用历史选定的 best.pt。",
             "R41 没有单独训练模型；R34/R39A/R40A/R40B 的 test 预测复用 R41，R42A/B 的 test 预测复用 R42。",
             "百分比为该数据集的 mean_cost / 参考 mean_cost − 1，不是逐实例相对损失的平均值；负 vs_sbs 表示更好。",
             "Top1/Top2/Top3 统一沿用 NSS 的 FP32 成本 argmin 标签（并列取候选顺序第一名）；",
             "Top-k 定义为模型分数前 k 名包含 oracle winner，不是模型 Top1 落在真实成本前 k 名。",
             "原生 ind 口径另外保存在 CSV 的 top1_native_ind/top2_native_ind/top3_native_ind 列；",
             "TSP test 有 3 例、TSPLIB 有 1 例的原生 ind 与 NSS 标签不同，CVRP 两项没有差异。",
             "成本按原始 FP64 表汇总。NSS 复用原始 JSON（含 FP32 舍入），公共参考值和百分比在此重算。",
             "R25 旧 Top2/Top3 定义不一致，因此本次补推理重算；不修改历史报告。", "",
             "## 公共数据与候选池", "",
             "| 数据集 | 实例数 | 候选数 | SBS 方法 | SBS cost | Oracle cost |",
             "| --- | ---: | ---: | --- | ---: | ---: |"]
    for dataset, values in raw.items():
        ref = references(values)
        lines.append(f"| {dataset} | {ref['n']} | {len(ref['pool'])} | {ref['sbs_name']} | "
                     f"{ref['sbs_cost']:.6f} | {ref['oracle_cost']:.6f} |")
    for problem in ("TSP", "CVRP"):
        lines += ["", f"- {problem} 候选顺序：`{' / '.join(POOLS[problem])}`"]
    for dataset in DATASETS:
        lines += ["", f"## {dataset}", "",
                  "| 模型 | top1 | top2 | top3 | mean_cost | SBS cost (vs_sbs) | Oracle cost (vs_Oracle) | cost − NSS |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            if r["dataset"] == dataset:
                lines.append(f"| {r['model_version']} | {r['top1']:.4f} | {r['top2']:.4f} | {r['top3']:.4f} | "
                             f"{r['mean_cost']:.6f} | {r['sbs_cost']:.6f} ({r['vs_sbs_pct']:+.4f}%) | "
                             f"{r['oracle_cost']:.6f} ({r['vs_oracle_pct']:+.4f}%) | {r['cost_difference_vs_nss']:+.6f} |")
    lines += ["", "## 固定权重与来源", "",
              "| 模型 | checkpoint | 历史轮次 / 更新数 |",
              "| --- | --- | --- |"]
    for name, record in checkpoints.items():
        progress = f"epoch {record['epoch']}" if record.get("epoch") is not None else str(record.get("updates", "历史汇总"))
        lines.append(f"| {name} | `{record['path']}` | {progress} |")
    lines += ["", "完整来源、每行实例数、决策规则、预测文件路径见 summary.csv。",
              "predictions/ 保存所有 Ours 的逐实例分数、选择及成本；NSS 只复用历史汇总，不重跑。", "",
              "## 主要观察", ""]
    by_key = {(r["model_version"], r["dataset"]): r for r in rows}
    if ("R42B", "CVRPLIB") in by_key:
        for dataset in DATASETS:
            better = [r for r in rows if r["dataset"] == dataset and r["model_version"] != "NSS"
                      and r["cost_difference_vs_nss"] < 0]
            text = "、".join(f"{r['model_version']} ({r['cost_difference_vs_nss']:+.6f})" for r in better)
            lines.append(f"- {dataset}：mean_cost 低于 NSS 的固定版本为 {text if text else '无'}。")
        lines += ["- R39B 的 TSP 成本下降伴随严格 Top1 降低，不能把其他分类头的 Top1 配给性能头成本。",
                  "- R42B 相比 R42A 的两项 LIB 均改善，但 CVRPLIB 仍高于 NSS 和 R25。",
                  "- R37B/R38B 的 TSP test 成本低于对应 A 对照，但 TSPLIB 未超过对应 A，未表现出稳定的跨分布收益。",
                  "- 这里只做固定权重的描述性比较，未计算显著性，也未按新评估结果重新选型。", ""]
    (root / "summary.md").write_text("\n".join(lines))
    with (root / "arm_distribution.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model_version", "dataset", "solver", "pick_rate"))
        for r in rows:
            writer.writerows((r["model_version"], r["dataset"], name, rate)
                            for name, rate in r["arm_distribution"].items())


@torch.inference_mode()
def infer(model, loader, batch_size, raw, path):
    scores, logits, performance = [], [], []
    for batch in TensorBatchLoader(loader.batch, batch_size, False, False):
        inputs = {k: v for k, v in batch.items() if k not in ("costs", "ind", "performance_target")}
        if model.params.get("architecture", "legacy") == "legacy":
            # Older selectors require this key only to check the candidate dimension.
            inputs["costs"] = torch.zeros_like(batch["costs"])
        out = model(inputs)
        score = out.get("selection_scores", out["logits"])
        scores.append(score.float().cpu())
        logits.append(out["logits"].float().cpu())
        if "pred_performance" in out:
            performance.append(out["pred_performance"].float().cpu())
    score = torch.cat(scores).numpy()
    if (score.shape != raw["costs"].shape or np.isnan(score).any() or np.isposinf(score).any()
            or not np.isfinite(score).any(1).all()):
        raise ValueError("Missing or nonfinite evaluation scores")
    metrics, pred = comparable_metrics(score, raw)
    data = dict(indices=np.arange(len(pred)), scores=score, logits=torch.cat(logits).numpy(), pred=pred,
                winner=raw["winner"], costs=raw["costs"], pool=np.array(raw["pool"]),
                pool_ids=np.array(raw["pool_ids"]), nodes=loader.lengths.numpy())
    if performance:
        data["pred_performance"] = torch.cat(performance).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    configure_torch()
    raw = {d: read_costs(*pair) for d, pair in DATASETS.items()}
    rows, checkpoints, loaders = [], {}, {}
    nss_path = Path("code/NSS_retrain/results/nss_retrain_rank_test_summary.json")
    nss = json.loads(nss_path.read_text())
    nss_root = nss_path.parents[1]
    nss_pools = json.loads((nss_root / "solver_order_ours.json").read_text())
    for problem in ("TSP", "CVRP"):
        if nss_pools[problem] != POOLS[problem]:
            raise ValueError("NSS candidate names/order differ from comparison registry")
        checkpoints[f"NSS_{problem}"] = dict(path=str(nss_root / "train_logs" /
             f"config_{problem}_ours_rank_2024/checkpoint_epoch_best.pt"), source="historical NSS results")
    for item in nss["rows"]:
        dataset = item["problem"] + " test" if item["problem"] in ("TSP", "CVRP") else item["problem"]
        metrics = historical_metrics(item, raw[dataset])
        counts = item.get("pick_counts", {})
        metrics["arm_distribution"] = {solver: counts.get(str(j), 0) / metrics["n"]
                                       for j, solver in enumerate(raw[dataset]["pool"])}
        problem = DATASETS[dataset][0]
        rows.append(row("NSS", "NSS retrained baseline", checkpoints[f"NSS_{problem}"]["path"], dataset,
                        metrics, nss_path))
    write_reports(root, rows, raw, checkpoints)

    for name, experiment, relative in MODELS:
        started = time.monotonic()
        path = Path("code/unified_selector/runs/R25_newdata_oldcode_resume1_seed2/best.pt") if name == "R25" else RUNS / relative
        ck = torch.load(path, map_location="cpu", weights_only=False)
        params = ck["args"].get("model_params", {"architecture": "r25"})
        params.setdefault("architecture", "legacy")
        feature_spec = params.get("solver_feature_spec")
        if feature_spec and feature_spec["solver_names"] != GLOBAL_SOLVERS:
            raise ValueError(f"{name}: checkpoint solver identities differ from current registry")
        if name == "R39B" and params.get("decision_head") != "performance":
            raise ValueError("R39B must retain its performance-head decision")
        if name == "R25":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            model = R25Selector(path, args.device)
        else:
            configure_torch()
            model = (make_r42_model(params) if params.get("architecture") == "relation_query"
                     else make_r40_model(params)).to(args.device)
            model.load_state_dict(ck["model"], strict=True)
        model.eval().requires_grad_(False)
        epoch = ck["epoch"] + 1 if "epoch" in ck else None
        updates = ck.get("updates", ck.get("successful_updates"))
        checkpoints[name] = dict(path=str(path), epoch=epoch, updates=updates,
                                 original_val=ck.get("macro"), decision=params.get("decision_head", "classification"))
        batch_size = ck["args"].get("eval_batch_size", ck["args"].get("batch_per_problem", 640))
        datasets = ("TSP test", "TSPLIB") if name.startswith(("R37", "R38")) else tuple(DATASETS)
        for dataset in datasets:
            problem, split = DATASETS[dataset]
            destination = root / "predictions" / name / (dataset.replace(" ", "_") + ".npz")
            previous = REUSE.get(name, Path("__not_available__")) / f"{problem}.npz"
            if destination.exists() or (split == "test" and previous.exists()):
                previous = destination if destination.exists() else previous
                with np.load(previous) as saved:
                    np.testing.assert_array_equal(saved["pool_ids"], raw[dataset]["pool_ids"])
                    np.testing.assert_array_equal(saved["winner"], raw[dataset]["winner"])
                    np.testing.assert_array_equal(saved["costs"], raw[dataset]["costs"])
                    metrics, _ = comparable_metrics(saved["scores"] if "scores" in saved else saved["logits"], raw[dataset])
                destination.parent.mkdir(parents=True, exist_ok=True)
                if previous != destination:
                    shutil.copyfile(previous, destination)
                source = str(REUSE[name] / f"{problem}.npz") if split == "test" and name in REUSE else "new frozen-checkpoint inference (saved predictions)"
            else:
                if dataset not in loaders:
                    loaders[dataset] = make_loader(problem, split, 640, 0, shuffle=False, cache_device=args.device)
                    np.testing.assert_array_equal(loaders[dataset].batch["pool_ids"].cpu(), raw[dataset]["pool_ids"])
                loader = loaders[dataset]
                if params.get("local_geometry") and "node_geom" not in loader.batch:
                    attach_geometry(loader, args.device)
                size = 32 if name == "R25" else (16 if split.endswith("LIB") else (batch_size or 640))
                metrics = infer(model, loader, size, raw[dataset], destination)
                source = "new frozen-checkpoint inference"
            rows.append(row(name, experiment, path, dataset, metrics, source, destination, epoch, updates))
            print(f"[{name} {dataset}] n={metrics['n']} top1={metrics['top1']:.4f} "
                  f"mean_cost={metrics['mean_cost']:.6f} vs_sbs={metrics['vs_sbs_pct']:+.4f}% "
                  f"vs_oracle={metrics['vs_oracle_pct']:+.4f}%", flush=True)
            write_reports(root, rows, raw, checkpoints)
        del model, ck
        torch.cuda.empty_cache()
        print(f"[done {name}] seconds={time.monotonic()-started:.1f}", flush=True)

    (root / "missing_items.md").write_text(
        "# 缺失项\n\n全部 56 行评估指标已齐全，没有需要补跑的模型或数据集。\n\n"
        "NSS 沿用历史评估 JSON，未补造逐实例预测；native ind 口径的 NSS Top-k 因无逐实例预测留空。\n"
        "所有 Ours 的 NSS 可比口径和 native ind 口径均已计算；R25 也补齐了逐实例预测。\n"
        "R41 是冻结评估阶段，不存在独立 R41 selector checkpoint；其四个模型已按各自版本纳入。\n"
        "R37/R38 不补 CVRP/CVRPLIB，是预定 TSP 专项范围，不属于缺失项。\n")
    print(f"[complete] {len(rows)} rows -> {root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
