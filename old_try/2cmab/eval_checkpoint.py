"""
eval_checkpoint.py — 加载 checkpoint 在 test 集上评测

用法:
  python 2cmab/eval_checkpoint.py --checkpoint 2cmab/checkpoints/<run_name>/final.pt
  python 2cmab/eval_checkpoint.py --checkpoint 2cmab/checkpoints/<run_name>/step500.pt
  python 2cmab/eval_checkpoint.py --checkpoint 2cmab/checkpoints/<run_name>/final.pt --n-instances 1000

输出内容:
  1. 总体指标: mean_reward, top1/top2/top3 命中率, regret
  2. 按 TSP/CVRP 分开: mean_cost, oracle_cost, single_best_cost, top-k
  3. Baseline 对比: random, single_best_global, single_best_per_problem, oracle
  4. 各臂选择分布 + 各臂被选后的真实效果
  5. 逐实例明细 CSV
"""
import os
import sys
import argparse
import csv
import numpy as np
import torch
from pathlib import Path
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(__file__))
from arm_config import ALL_METHODS, N_ARMS, get_mask
from reward import rank_reward, compute_all_rewards
from baselines import (OracleSelector, RandomSelector,
                        SingleBestGlobalSelector, SingleBestPerProblemSelector)
from run_offline import load_cost_matrices, load_instance_coords, stratified_split


def compute_rank_of_arm(arm_cost, all_costs, mask):
    """计算某臂在可行臂中的排名 (1=最好), 支持 tie 用平均名次"""
    feasible = np.where(mask > 0)[0]
    costs = all_costs[feasible]
    valid = ~np.isnan(costs)
    costs_valid = costs[valid]
    if len(costs_valid) == 0:
        return np.nan
    ranks = rankdata(costs_valid, method="average")
    idx = np.argmin(np.abs(costs_valid - arm_cost))
    return float(ranks[idx])


def main():
    parser = argparse.ArgumentParser(description="加载 checkpoint 在 test 集上评测")
    parser.add_argument("--checkpoint", default="2cmab/checkpoints/0324_130801_neural_linucb_a1.0_q50_J10_bs512_s42/step14000.pt", help="checkpoint 文件路径")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42, help="必须和训练时一致")
    parser.add_argument("--n-instances", type=int, default=0, help="必须和训练时一致")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--output-csv", default="", help="逐实例明细输出路径")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    project_root = Path(__file__).resolve().parent.parent

    # ---- 1. 加载数据 ----
    print("加载数据...")
    all_costs, all_types, all_masks = load_cost_matrices(data_dir, args.n_instances)
    if len(all_costs) == 0:
        print("错误: 没有找到 cost 矩阵")
        return
    coords_dict = load_instance_coords(project_root, args.n_instances)
    if not coords_dict:
        print("错误: 没有找到坐标数据")
        return

    # ---- 2. 划分 ----
    train_idx, val_idx, test_idx = stratified_split(all_types, seed=args.seed)
    print(f"划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    tsp_indices = [i for i, t in enumerate(all_types) if t == "tsp"]
    cvrp_indices = [i for i, t in enumerate(all_types) if t == "cvrp"]
    def get_coords(global_idx):
        ptype = all_types[global_idx]
        if ptype == "tsp":
            return coords_dict["tsp"][tsp_indices.index(global_idx)].numpy()
        else:
            return coords_dict["cvrp"][cvrp_indices.index(global_idx)].numpy()

    # ---- 3. 加载 checkpoint (自动识别 NeuralLinUCB / NeuralTS) ----
    print(f"\n加载 checkpoint: {args.checkpoint}")
    from encoder import DualEncoder

    # 先 peek checkpoint 判断类型
    ckpt_peek = torch.load(args.checkpoint, map_location="cpu")
    is_neural_ts = "reward_nets" in ckpt_peek
    del ckpt_peek

    encoder = DualEncoder()
    if is_neural_ts:
        from neural_ts import NeuralTS
        selector = NeuralTS(n_arms=N_ARMS, encoder=encoder, arm_names=ALL_METHODS, device=args.device)
        selector.load_checkpoint(args.checkpoint)
        print(f"  [NeuralTS] 模型已加载, t={selector.t}, 各臂拉取次数: {selector.arm_pull_counts.tolist()}")
    else:
        from neural_linucb import NeuralLinUCB
        selector = NeuralLinUCB(n_arms=N_ARMS, encoder=encoder, arm_names=ALL_METHODS, device=args.device)
        selector.load_checkpoint(args.checkpoint)
        print(f"  [NeuralLinUCB] 模型已加载, t={selector.t}, 各臂拉取次数: {selector.arm_pull_counts.tolist()}")

    # ---- 4. 在 test 集上评测 ----
    print(f"\n在 test 集上评测 ({len(test_idx)} 实例)...")
    oracle = OracleSelector()
    details = []

    for ti, i in enumerate(test_idx):
        ptype = all_types[i]
        c = get_coords(i)
        arm = selector.select(c, None, ptype, 100, all_masks[i])
        oracle_arm = oracle.select_with_costs(all_costs[i], all_masks[i])
        reward = rank_reward(all_costs[i, arm], all_costs[i], all_masks[i])
        oracle_reward = rank_reward(all_costs[i, oracle_arm], all_costs[i], all_masks[i])
        chosen_rank = compute_rank_of_arm(all_costs[i, arm], all_costs[i], all_masks[i])

        details.append({
            "test_idx": ti,
            "global_idx": int(i),
            "problem": ptype,
            "chosen_arm": arm,
            "chosen_method": ALL_METHODS[arm],
            "chosen_cost": float(all_costs[i, arm]),
            "chosen_reward": float(reward),
            "chosen_rank": chosen_rank,
            "oracle_arm": oracle_arm,
            "oracle_method": ALL_METHODS[oracle_arm],
            "oracle_cost": float(all_costs[i, oracle_arm]),
            "oracle_reward": float(oracle_reward),
            "match": (arm == oracle_arm),
        })
        if (ti + 1) % 500 == 0:
            print(f"    已评测 {ti+1}/{len(test_idx)}...")

    # ================================================================
    #  5. 汇总统计
    # ================================================================
    n_test = len(details)

    def summarize(subset, label):
        """对一组 details 计算所有指标"""
        n = len(subset)
        if n == 0:
            return
        rewards = [d["chosen_reward"] for d in subset]
        oracle_rewards = [d["oracle_reward"] for d in subset]
        costs = [d["chosen_cost"] for d in subset]
        oracle_costs = [d["oracle_cost"] for d in subset]
        ranks = [d["chosen_rank"] for d in subset]
        matches = [d["match"] for d in subset]

        mean_reward = np.mean(rewards)
        mean_cost = np.mean(costs)
        oracle_mean_cost = np.mean(oracle_costs)
        regret = np.mean(oracle_rewards) - mean_reward
        top1 = np.mean(matches)
        top2 = np.mean([r <= 2.0 for r in ranks])
        top3 = np.mean([r <= 3.0 for r in ranks])
        mean_rank = np.mean(ranks)

        print(f"\n  {label} ({n} 实例):")
        print(f"    mean_reward:    {mean_reward:.4f}")
        print(f"    mean_rank:      {mean_rank:.2f}")
        print(f"    top1_hit_rate:  {top1:.4f}  ({int(top1*n)}/{n})")
        print(f"    top2_hit_rate:  {top2:.4f}  ({int(top2*n)}/{n})")
        print(f"    top3_hit_rate:  {top3:.4f}  ({int(top3*n)}/{n})")
        print(f"    regret:         {regret:.4f}")
        print(f"    mean_cost:      {mean_cost:.4f}  (模型选择)")
        print(f"    oracle_cost:    {oracle_mean_cost:.4f}  (每个实例选最好的)")

    print("\n" + "=" * 70)
    print("  评测结果")
    print("=" * 70)

    summarize(details, "ALL")
    for pt in ["tsp", "cvrp"]:
        pt_details = [d for d in details if d["problem"] == pt]
        summarize(pt_details, pt.upper())

    # ---- 6. Baseline 对比 ----
    print("\n" + "=" * 70)
    print("  Baseline 对比 (在同一 test 集上)")
    print("=" * 70)

    for pt in ["tsp", "cvrp"]:
        pt_test = [i for i in test_idx if all_types[i] == pt]
        pt_details = [d for d in details if d["problem"] == pt]
        if not pt_test or not pt_details:
            continue
        mask = get_mask(pt)
        feasible = np.where(mask > 0)[0]

        # 模型选择
        model_cost = np.mean([d["chosen_cost"] for d in pt_details])

        # Oracle: 每个实例选 cost 最低的臂
        oracle_cost = np.mean([d["oracle_cost"] for d in pt_details])

        # Single-best: 在 test 集上找平均 cost 最低的那个固定臂
        arm_mean_costs = {}
        for k in feasible:
            arm_costs = [all_costs[i, k] for i in pt_test if not np.isnan(all_costs[i, k])]
            if arm_costs:
                arm_mean_costs[k] = np.mean(arm_costs)
        sb_arm = min(arm_mean_costs, key=arm_mean_costs.get)
        sb_cost = arm_mean_costs[sb_arm]

        # Random: 每个实例对所有可行臂均匀随机, 期望 cost = 可行臂 cost 的均值
        random_cost = np.mean([
            np.nanmean([all_costs[i, k] for k in feasible])
            for i in pt_test
        ])

        print(f"\n  {pt.upper()} ({len(pt_test)} 实例):")
        print(f"    oracle mean_cost:            {oracle_cost:.4f}  (每实例选最好的, 上界)")
        print(f"    模型选择 mean_cost:          {model_cost:.4f}")
        print(f"    single_best mean_cost:       {sb_cost:.4f}  (固定选 {ALL_METHODS[sb_arm]})")
        print(f"    random mean_cost:            {random_cost:.4f}  (均匀随机的期望)")

    # ---- 7. 各臂选择分布 ----
    print("\n" + "=" * 70)
    print("  各臂选择分布")
    print("=" * 70)

    for pt in ["tsp", "cvrp", "all"]:
        subset = details if pt == "all" else [d for d in details if d["problem"] == pt]
        if not subset:
            continue
        n_sub = len(subset)
        print(f"\n  {pt.upper()} ({n_sub} 实例):")
        arm_counts = np.zeros(N_ARMS)
        for d in subset:
            arm_counts[d["chosen_arm"]] += 1
        for k in range(N_ARMS):
            if arm_counts[k] > 0:
                print(f"    {k:2d}({ALL_METHODS[k]:16s}): {int(arm_counts[k]):5d} 次 "
                      f"({arm_counts[k]/n_sub*100:5.1f}%)")

    # ---- 8. 各臂被选后的真实效果 ----
    print("\n" + "=" * 70)
    print("  各臂被选后的真实效果")
    print("=" * 70)

    for pt in ["tsp", "cvrp"]:
        subset = [d for d in details if d["problem"] == pt]
        if not subset:
            continue
        print(f"\n  {pt.upper()}:")
        print(f"    {'arm':20s} {'count':>6s} {'mean_cost':>10s} {'mean_reward':>12s} {'top1_rate':>10s} {'mean_rank':>10s}")
        print(f"    {'─'*20} {'─'*6} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")
        for k in range(N_ARMS):
            arm_subset = [d for d in subset if d["chosen_arm"] == k]
            if not arm_subset:
                continue
            ac = np.mean([d["chosen_cost"] for d in arm_subset])
            ar = np.mean([d["chosen_reward"] for d in arm_subset])
            at1 = np.mean([d["match"] for d in arm_subset])
            amr = np.mean([d["chosen_rank"] for d in arm_subset])
            print(f"    {k:2d}({ALL_METHODS[k]:16s}) {len(arm_subset):6d} {ac:10.4f} {ar:12.4f} {at1:10.4f} {amr:10.2f}")

    # ---- 9. Oracle 最优臂分布 ----
    print("\n" + "=" * 70)
    print("  Oracle 最优臂分布 (test 集上哪个方法最常是最优的)")
    print("=" * 70)

    for pt in ["tsp", "cvrp"]:
        subset = [d for d in details if d["problem"] == pt]
        if not subset:
            continue
        n_sub = len(subset)
        oracle_counts = np.zeros(N_ARMS)
        for d in subset:
            oracle_counts[d["oracle_arm"]] += 1
        print(f"\n  {pt.upper()} ({n_sub} 实例):")
        for k in np.argsort(-oracle_counts):
            if oracle_counts[k] > 0:
                print(f"    {k:2d}({ALL_METHODS[k]:16s}): {int(oracle_counts[k]):5d} 次 "
                      f"({oracle_counts[k]/n_sub*100:5.1f}%)")

    # ---- 10. 保存 CSV ----
    if args.output_csv:
        csv_path = Path(args.output_csv)
    else:
        csv_path = Path(args.checkpoint).parent / "test_details.csv"

    fieldnames = list(details[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)
    print(f"\n逐实例明细已保存到: {csv_path}")


if __name__ == "__main__":
    main()
