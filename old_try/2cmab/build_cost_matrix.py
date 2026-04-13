"""
build_cost_matrix.py — 从 EasyNCO/results/test/ 构建 cost 矩阵

扫描评测结果目录，读取 instance_results.jsonl，
构建 [N_instances, 12] 的 cost 矩阵 (aug_score)。
结果保存为 npz 文件到 2cmab/data/。

用法:
  cd neural-solver-selection
  python 2cmab/build_cost_matrix.py
  # 或指定路径:
  python 2cmab/build_cost_matrix.py --results-dir EasyNCO/results/test --output-dir 2cmab/data
"""
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arm_config import ALL_METHODS, METHOD_TO_IDX, N_ARMS


def discover_results(results_dir: Path) -> dict:
    """
    扫描 EasyNCO/results/test/ 下的所有结果。
    目录结构: results/test/{method}_{problem}/{timestamp_run}/instance_results.jsonl
    返回: {(problem, method): {global_index: aug_score}}
    """
    all_scores = {}
    for method_problem_dir in sorted(results_dir.iterdir()):
        if not method_problem_dir.is_dir():
            continue
        name = method_problem_dir.name
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        method, problem = parts
        if problem not in ("tsp", "cvrp"):
            continue  # 只处理 TSP 和 CVRP

        # 找最新 run
        run_dirs = sorted([d for d in method_problem_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
        if not run_dirs:
            continue
        jsonl_path = run_dirs[-1] / "instance_results.jsonl"
        if not jsonl_path.exists():
            print(f"  警告: 缺少 {jsonl_path}")
            continue

        scores = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                scores[int(rec["global_index"])] = float(rec.get("aug_score", rec.get("no_aug_score", float("nan"))))
        all_scores[(problem, method)] = scores
        print(f"  {problem}/{method}: {len(scores)} 实例")
    return all_scores


def build_matrix(all_scores: dict, problem: str, methods: list) -> tuple:
    """构建 [N, N_ARMS] cost 矩阵，不可用方法填 NaN"""
    # 取所有方法的实例交集
    sets = [set(all_scores[(problem, m)].keys()) for m in methods if (problem, m) in all_scores]
    if not sets:
        return np.array([]), []
    common = sorted(sets[0].intersection(*sets[1:]))
    N = len(common)
    costs = np.full((N, N_ARMS), np.nan, dtype=np.float64)
    for m in methods:
        key = (problem, m)
        if key not in all_scores:
            continue
        col = METHOD_TO_IDX[m]
        for row, gidx in enumerate(common):
            costs[row, col] = all_scores[key].get(gidx, np.nan)
    print(f"  {problem}: {N} 个共有实例, {sum(1 for m in methods if (problem,m) in all_scores)} 个方法")
    return costs, common


def main():
    parser = argparse.ArgumentParser(description="构建 cost 矩阵")
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "EasyNCO" / "results" / "test"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "data"))
    args = parser.parse_args()

    results_dir, output_dir = Path(args.results_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"扫描: {results_dir}")
    all_scores = discover_results(results_dir)

    # TSP
    tsp_methods = [m for m in ALL_METHODS]  # 全部 12 个
    print("\n构建 TSP cost 矩阵...")
    tsp_costs, tsp_idx = build_matrix(all_scores, "tsp", tsp_methods)
    if tsp_costs.size > 0:
        np.savez(output_dir / "tsp100_costs.npz", costs=tsp_costs, indices=np.array(tsp_idx), method_names=np.array(ALL_METHODS))
        print(f"  -> {output_dir / 'tsp100_costs.npz'}: {tsp_costs.shape}")

    # CVRP
    cvrp_methods = ALL_METHODS[:8]  # 前 8 个
    print("\n构建 CVRP cost 矩阵...")
    cvrp_costs, cvrp_idx = build_matrix(all_scores, "cvrp", cvrp_methods)
    if cvrp_costs.size > 0:
        np.savez(output_dir / "cvrp100_costs.npz", costs=cvrp_costs, indices=np.array(cvrp_idx), method_names=np.array(ALL_METHODS))
        print(f"  -> {output_dir / 'cvrp100_costs.npz'}: {cvrp_costs.shape}")

    # 统计
    print("\n=== 统计 ===")
    for label, costs in [("TSP", tsp_costs), ("CVRP", cvrp_costs)]:
        if costs.size == 0:
            continue
        print(f"{label}: {costs.shape[0]} 实例")
        for i, m in enumerate(ALL_METHODS):
            col = costs[:, i]
            v = col[~np.isnan(col)]
            if len(v) > 0:
                print(f"  {m:20s}: mean={v.mean():.4f} std={v.std():.4f}")


if __name__ == "__main__":
    main()
