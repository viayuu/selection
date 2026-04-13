"""
run_offline.py — 主实验入口

用法:
  cd neural-solver-selection
  # 第一步: 构建 cost 矩阵 (只需运行一次)
  python 2cmab/build_cost_matrix.py

  # 第二步: 运行实验
  python 2cmab/run_offline.py --method linucb --alpha 1.0
  python 2cmab/run_offline.py --method neural_linucb --alpha 1.0
  python 2cmab/run_offline.py --method neural_ts --nu 1.0

流程:
  1. 加载 cost 矩阵 + 实例坐标
  2. 提取特征 (手工 or 神经网络)
  3. 70/15/15 分层划分
  4. 训练 selector
  5. 在 test 集上评测
  6. 对比 baselines 输出结果
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from arm_config import ALL_METHODS, N_ARMS, get_mask
from reward import rank_reward
from features import manual_features_single, build_interaction_features
from baselines import RandomSelector, SingleBestGlobalSelector, SingleBestPerProblemSelector, OracleSelector
import config as C  # 集中配置


def load_cost_matrices(data_dir: Path, n_instances: int = 0):
    """加载 cost 矩阵, 返回 (costs, problem_types, masks). n_instances>0 时每种类型只取前 n 条"""
    all_costs, all_types, all_masks = [], [], []

    for fname, ptype in [("tsp100_costs.npz", "tsp"), ("cvrp100_costs.npz", "cvrp")]:
        path = data_dir / fname
        if not path.exists():
            print(f"  警告: {path} 不存在, 跳过 {ptype.upper()}")
            continue
        data = np.load(path, allow_pickle=True)
        costs = data["costs"]  # [N, 12]
        n = min(costs.shape[0], n_instances) if n_instances > 0 else costs.shape[0]
        mask = get_mask(ptype)
        for i in range(n):
            all_costs.append(costs[i])
            all_types.append(ptype)
            all_masks.append(mask)
        print(f"  {ptype.upper()}: {n} 实例" + (f" (截取自 {costs.shape[0]})" if n < costs.shape[0] else ""))

    return np.array(all_costs), all_types, np.array(all_masks)


def load_instance_coords(data_dir: Path, n_instances: int = 0):
    """
    加载实例坐标 (用于 Neural 方法的 encoder 输入)。
    从 EasyNCO 的 test 数据集中加载。
    如果找不到, 返回 None。
    """
    easynco_data = data_dir / "EasyNCO" / "data" / "datasets"

    tsp_path = easynco_data / "test_dataset_tsp_uniform" / "test_tsp100_nums10000_uniform.pt"
    cvrp_path = easynco_data / "test_dataset_vrp_uniform" / "test_vrp100_capacity50_nums10000_uniform.pt"

    coords = {}
    if tsp_path.exists():
        tsp_data = torch.load(tsp_path, map_location="cpu")
        if isinstance(tsp_data, dict):
            coords["tsp"] = tsp_data.get("node_xy", tsp_data.get("data"))
        else:
            coords["tsp"] = tsp_data
        print(f"  TSP 坐标: {coords['tsp'].shape if coords.get('tsp') is not None else 'N/A'}")
        if n_instances > 0 and coords.get("tsp") is not None:
            coords["tsp"] = coords["tsp"][:n_instances]

    if cvrp_path.exists():
        cvrp_data = torch.load(cvrp_path, map_location="cpu")
        if isinstance(cvrp_data, dict):
            node_xy = cvrp_data.get("node_xy")
            demand = cvrp_data.get("node_demand")
            if node_xy is not None and demand is not None:
                # CVRP: 拼接坐标和需求 -> [N, nodes, 3]
                if demand.dim() == 2:
                    demand = demand.unsqueeze(-1)
                coords["cvrp"] = torch.cat([node_xy, demand], dim=-1)
            else:
                coords["cvrp"] = cvrp_data.get("data")
        else:
            coords["cvrp"] = cvrp_data
        print(f"  CVRP 坐标: {coords['cvrp'].shape if coords.get('cvrp') is not None else 'N/A'}")
        if n_instances > 0 and coords.get("cvrp") is not None:
            coords["cvrp"] = coords["cvrp"][:n_instances]

    return coords


def stratified_split(problem_types, train_ratio=0.7, val_ratio=0.15, seed=42):
    """按问题类型分层 70/15/15 划分"""
    rng = np.random.RandomState(seed)
    indices = {"tsp": [], "cvrp": []}
    for i, pt in enumerate(problem_types):
        indices[pt].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for pt in ["tsp", "cvrp"]:
        idx = np.array(indices[pt])
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def extract_all_features(all_costs, all_types, all_masks, coords_dict, scale=100):
    """提取手工特征 (34维, 含交互)"""
    print("  提取手工特征 (34维)...")
    features = []
    tsp_counter, cvrp_counter = 0, 0
    for i, ptype in enumerate(all_types):
        if ptype == "tsp" and coords_dict and "tsp" in coords_dict:
            nodes = coords_dict["tsp"][tsp_counter].numpy()
            tsp_counter += 1
        elif ptype == "cvrp" and coords_dict and "cvrp" in coords_dict:
            nodes = coords_dict["cvrp"][cvrp_counter].numpy()
            cvrp_counter += 1
        else:
            # 无坐标数据, 用零特征
            nodes = np.zeros((scale, 3 if ptype == "cvrp" else 2))
        base = manual_features_single(nodes, scale, ptype)
        features.append(build_interaction_features(base))
    return np.array(features)


def evaluate_selector(selector, all_costs, all_masks, all_types, test_idx,
                      features=None, method_name=""):
    """在 test 集上评测 selector, 返回指标字典"""
    oracle = OracleSelector()
    total_reward = 0.0
    total_oracle_reward = 0.0
    total_cost = 0.0
    total_oracle_cost = 0.0
    n_correct = 0  # top-1 命中率
    method_counts = np.zeros(N_ARMS)

    for i in test_idx:
        costs = all_costs[i]
        mask = all_masks[i]
        ptype = all_types[i]

        # selector 选择
        if method_name in ("linucb",):
            arm = selector.select(features[i], mask)
        elif method_name in ("random", "single_best_global"):
            arm = selector.select(None, mask)
        elif method_name == "single_best_per_problem":
            arm = selector.select(None, mask, problem_type=ptype)
        else:
            arm = selector.select(features[i], mask, problem_type=ptype)

        # oracle 选择
        oracle_arm = oracle.select_with_costs(costs, mask)

        # 奖励
        reward = rank_reward(costs[arm], costs, mask)
        oracle_reward = rank_reward(costs[oracle_arm], costs, mask)
        total_reward += reward
        total_oracle_reward += oracle_reward

        # cost
        total_cost += costs[arm] if not np.isnan(costs[arm]) else 0
        total_oracle_cost += costs[oracle_arm] if not np.isnan(costs[oracle_arm]) else 0

        # top-1 命中
        if arm == oracle_arm:
            n_correct += 1
        method_counts[arm] += 1

    n = len(test_idx)
    return {
        "mean_reward": total_reward / n,
        "oracle_reward": total_oracle_reward / n,
        "regret": (total_oracle_reward - total_reward) / n,
        "mean_cost": total_cost / n,
        "oracle_cost": total_oracle_cost / n,
        "top1_hit_rate": n_correct / n,
        "method_distribution": {ALL_METHODS[k]: int(method_counts[k]) for k in range(N_ARMS) if method_counts[k] > 0},
    }


def run_linucb(all_costs, all_masks, all_types, features, train_idx, val_idx, test_idx, alpha):
    """运行 LinUCB 实验"""
    from linucb import LinUCB

    print(f"\n=== LinUCB (alpha={alpha}) ===")
    selector = LinUCB(n_arms=N_ARMS, context_dim=features.shape[1], alpha=alpha)

    # 训练: 遍历训练集, 每个实例选臂并更新
    rng = np.random.RandomState(42)
    order = train_idx.copy()
    for epoch in range(3):  # 多轮遍历
        rng.shuffle(order)
        total_r = 0.0
        for i in order:
            arm = selector.select(features[i], all_masks[i])
            reward = rank_reward(all_costs[i, arm], all_costs[i], all_masks[i])
            selector.update(arm, features[i], reward)
            total_r += reward
        print(f"  epoch {epoch+1}: mean_reward={total_r/len(order):.4f}")

    # 评测
    result = evaluate_selector(selector, all_costs, all_masks, all_types, test_idx,
                               features=features, method_name="linucb")
    print(f"  test: mean_reward={result['mean_reward']:.4f}, top1_hit={result['top1_hit_rate']:.4f}, "
          f"regret={result['regret']:.4f}")
    return result, selector


def run_baselines(all_costs, all_masks, all_types, train_idx, test_idx):
    """运行所有 baseline"""
    results = {}

    # Random
    print("\n=== Random ===")
    random_sel = RandomSelector()
    results["random"] = evaluate_selector(random_sel, all_costs, all_masks, all_types, test_idx,
                                          method_name="random")
    print(f"  mean_reward={results['random']['mean_reward']:.4f}")

    # SingleBest Global
    print("\n=== SingleBest Global ===")
    sb_global = SingleBestGlobalSelector(N_ARMS)
    sb_global.fit(all_costs[train_idx], all_masks[train_idx])
    results["single_best_global"] = evaluate_selector(sb_global, all_costs, all_masks, all_types, test_idx,
                                                       method_name="single_best_global")
    best_method = ALL_METHODS[sb_global.best_arm]
    print(f"  best_method={best_method}, mean_reward={results['single_best_global']['mean_reward']:.4f}")

    # SingleBest Per Problem
    print("\n=== SingleBest Per Problem ===")
    sb_pp = SingleBestPerProblemSelector(N_ARMS)
    sb_pp.fit(all_costs[train_idx], all_masks[train_idx],
              [all_types[i] for i in train_idx])
    results["single_best_per_problem"] = evaluate_selector(sb_pp, all_costs, all_masks, all_types, test_idx,
                                                            method_name="single_best_per_problem")
    print(f"  TSP best={ALL_METHODS[sb_pp.best_arm_tsp]}, CVRP best={ALL_METHODS[sb_pp.best_arm_cvrp]}, "
          f"mean_reward={results['single_best_per_problem']['mean_reward']:.4f}")

    # Oracle
    print("\n=== Oracle ===")
    oracle = OracleSelector()
    total_r = 0.0
    for i in test_idx:
        arm = oracle.select_with_costs(all_costs[i], all_masks[i])
        total_r += rank_reward(all_costs[i, arm], all_costs[i], all_masks[i])
    results["oracle"] = {"mean_reward": total_r / len(test_idx)}
    print(f"  mean_reward={results['oracle']['mean_reward']:.4f}")

    return results


def main():
    # ---- 命令行参数 (默认值从 config.py 读取, 命令行可覆盖) ----
    parser = argparse.ArgumentParser(description="Contextual Bandit 求解器选择器")
    parser.add_argument("--method", default="linucb",
                        choices=["linucb", "neural_linucb", "neural_ts", "all"])
    parser.add_argument("--alpha", type=float, default=C.ALPHA)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--device", default=C.DEVICE)
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--n-instances", type=int, default=C.N_INSTANCES)
    parser.add_argument("--train-every", type=int, default=C.TRAIN_EVERY)
    parser.add_argument("--train-steps", type=int, default=C.TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--max-history", type=int, default=C.MAX_HISTORY)
    parser.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / C.LOG_DIR))
    parser.add_argument("--verbose", action="store_true", default=C.VERBOSE)
    parser.add_argument("--save-checkpoint", action="store_true", default=C.SAVE_CHECKPOINT)
    parser.add_argument("--checkpoint-dir",
                        default=str(Path(__file__).resolve().parent / C.CHECKPOINT_DIR))
    parser.add_argument("--checkpoint-every", type=int, default=C.CHECKPOINT_EVERY)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    project_root = Path(__file__).resolve().parent.parent

    # ---- 本次运行的唯一标识 (时间戳 + 关键超参) ----
    run_id = datetime.now().strftime("%m%d_%H%M%S")
    run_tag = f"{args.method}_a{args.alpha}_q{args.train_every}_J{args.train_steps}_bs{args.batch_size}_s{args.seed}"
    run_name = f"{run_id}_{run_tag}"
    print(f"运行标识: {run_name}")

    # ---- 设置日志 ----
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"

    nl_logger = logging.getLogger("neural_linucb")
    nl_logger.setLevel(logging.DEBUG)
    nl_logger.handlers.clear()
    # 文件 handler: 始终写入详细日志
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    nl_logger.addHandler(fh)
    # 终端 handler: 仅 --verbose 时输出
    if args.verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter("%(message)s"))
        nl_logger.addHandler(ch)

    print(f"日志将写入: {log_file}")

    # 1. 加载数据
    print("加载 cost 矩阵...")
    all_costs, all_types, all_masks = load_cost_matrices(data_dir, args.n_instances)
    if len(all_costs) == 0:
        print("错误: 没有找到 cost 矩阵。请先运行: python 2cmab/build_cost_matrix.py")
        return

    print(f"总计: {len(all_costs)} 实例 (TSP: {all_types.count('tsp')}, CVRP: {all_types.count('cvrp')})")

    # 2. 加载坐标 (如果可用)
    print("\n加载实例坐标...")
    coords_dict = load_instance_coords(project_root, args.n_instances)

    # 3. 划分
    train_idx, val_idx, test_idx = stratified_split(all_types, seed=args.seed)
    print(f"\n划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 4. 提取手工特征 (LinUCB 用)
    features = extract_all_features(all_costs, all_types, all_masks, coords_dict)

    # 5. 运行 baselines
    baseline_results = run_baselines(all_costs, all_masks, all_types, train_idx, test_idx)

    # 6. 运行选定的方法
    method_results = {}
    if args.method in ("linucb", "all"):
        result, _ = run_linucb(all_costs, all_masks, all_types, features, train_idx, val_idx, test_idx, args.alpha)
        method_results["linucb"] = result

    if args.method in ("neural_linucb", "all"):
        print(f"\n=== Online Neural-LinUCB (alpha={args.alpha}, device={args.device}) ===")
        if coords_dict and ("tsp" in coords_dict or "cvrp" in coords_dict):
            from encoder import DualEncoder
            from neural_linucb import NeuralLinUCB

            encoder = DualEncoder()
            nl = NeuralLinUCB(
                n_arms=N_ARMS, encoder=encoder,
                arm_names=ALL_METHODS,
                alpha=args.alpha,
                train_every=args.train_every, train_steps=args.train_steps,
                batch_size=args.batch_size,
                max_history=args.max_history,
                device=args.device,
            )

            # 建立实例索引映射 (全局 idx → 坐标)
            tsp_indices = [i for i, t in enumerate(all_types) if t == "tsp"]
            cvrp_indices = [i for i, t in enumerate(all_types) if t == "cvrp"]
            def get_coords(global_idx):
                ptype = all_types[global_idx]
                if ptype == "tsp":
                    local_idx = tsp_indices.index(global_idx)
                    return coords_dict["tsp"][local_idx].numpy()
                else:
                    local_idx = cvrp_indices.index(global_idx)
                    return coords_dict["cvrp"][local_idx].numpy()

            # 在线交互: 逐个实例 select → observe reward → update
            rng = np.random.RandomState(args.seed)
            order = train_idx.copy()
            rng.shuffle(order)
            print(f"  在线交互 {len(order)} 轮 (train_every={args.train_every}, train_steps={args.train_steps})")
            nl_logger = logging.getLogger("neural_linucb")
            nl_logger.info(f"=== 在线训练开始: {len(order)} 轮, alpha={args.alpha}, "
                           f"train_every={args.train_every}, train_steps={args.train_steps} ===")

            cumulative_reward = 0.0
            cumulative_oracle_reward = 0.0
            oracle_sel = OracleSelector()
            # checkpoint 目录预创建 (每次运行一个子目录)
            ckpt_dir = None
            if args.save_checkpoint:
                ckpt_dir = Path(args.checkpoint_dir) / run_name
                ckpt_dir.mkdir(parents=True, exist_ok=True)
            for step, i in enumerate(order):
                ptype = all_types[i]
                c = get_coords(i)
                arm = nl.select(c, None, ptype, 100, all_masks[i])
                reward = rank_reward(all_costs[i, arm], all_costs[i], all_masks[i])
                # oracle 对比
                oracle_arm = oracle_sel.select_with_costs(all_costs[i], all_masks[i])
                oracle_reward = rank_reward(all_costs[i, oracle_arm], all_costs[i], all_masks[i])
                nl_logger.info(
                    f"t={step} | TRAIN | {ptype.upper()} | "
                    f"chosen={arm}({ALL_METHODS[arm]}) cost={all_costs[i,arm]:.4f} reward={reward:.4f} | "
                    f"oracle={oracle_arm}({ALL_METHODS[oracle_arm]}) cost={all_costs[i,oracle_arm]:.4f} reward={oracle_reward:.4f} | "
                    f"match={'YES' if arm == oracle_arm else 'NO'}"
                )
                nl.update(arm, c, None, ptype, 100, reward)
                cumulative_reward += reward
                cumulative_oracle_reward += oracle_reward
                if (step + 1) % 500 == 0:
                    avg_r = cumulative_reward / (step + 1)
                    avg_or = cumulative_oracle_reward / (step + 1)
                    regret = avg_or - avg_r
                    msg = (f"    step {step+1}/{len(order)}, avg_reward={avg_r:.4f}, "
                           f"oracle={avg_or:.4f}, regret={regret:.4f}")
                    print(msg)
                    nl_logger.info(f"=== CHECKPOINT {msg.strip()} ===")
                # 周期性保存 checkpoint
                if (args.save_checkpoint and args.checkpoint_every > 0
                        and (step + 1) % args.checkpoint_every == 0):
                    ckpt_path = ckpt_dir / f"step{step+1}.pt"
                    nl.save_checkpoint(str(ckpt_path))
                    print(f"    checkpoint 已保存: {ckpt_path}")

            final_avg = cumulative_reward / len(order)
            print(f"  在线训练完成, 最终 avg_reward={final_avg:.4f}")
            nl_logger.info(f"=== 在线训练完成, avg_reward={final_avg:.4f} ===")

            # 评测: 在 test 集上逐实例评测
            print("  在 test 集上评测...")
            nl_logger.info(f"=== TEST 评测开始: {len(test_idx)} 实例 ===")
            total_reward, total_oracle_reward = 0.0, 0.0
            n_correct = 0
            oracle = OracleSelector()
            method_counts = np.zeros(N_ARMS)
            for ti, i in enumerate(test_idx):
                ptype = all_types[i]
                c = get_coords(i)
                arm = nl.select(c, None, ptype, 100, all_masks[i])
                oracle_arm = oracle.select_with_costs(all_costs[i], all_masks[i])
                reward = rank_reward(all_costs[i, arm], all_costs[i], all_masks[i])
                oracle_reward = rank_reward(all_costs[i, oracle_arm], all_costs[i], all_masks[i])
                total_reward += reward
                total_oracle_reward += oracle_reward
                hit = arm == oracle_arm
                if hit:
                    n_correct += 1
                method_counts[arm] += 1
                nl_logger.info(
                    f"TEST[{ti}] | {ptype.upper()} | "
                    f"chosen={arm}({ALL_METHODS[arm]}) cost={all_costs[i,arm]:.4f} reward={reward:.4f} | "
                    f"oracle={oracle_arm}({ALL_METHODS[oracle_arm]}) cost={all_costs[i,oracle_arm]:.4f} | "
                    f"match={'YES' if hit else 'NO'}"
                )

            n_test = len(test_idx)
            result = {
                "mean_reward": total_reward / n_test,
                "oracle_reward": total_oracle_reward / n_test,
                "regret": (total_oracle_reward - total_reward) / n_test,
                "top1_hit_rate": n_correct / n_test,
                "method_distribution": {ALL_METHODS[k]: int(method_counts[k])
                                        for k in range(N_ARMS) if method_counts[k] > 0},
            }
            print(f"  test: mean_reward={result['mean_reward']:.4f}, "
                  f"top1_hit={result['top1_hit_rate']:.4f}, regret={result['regret']:.4f}")
            nl_logger.info(
                f"=== TEST 结果: mean_reward={result['mean_reward']:.4f}, "
                f"top1_hit={result['top1_hit_rate']:.4f}, regret={result['regret']:.4f} ===\n"
                f"  method_distribution={result['method_distribution']}"
            )
            method_results["neural_linucb"] = result

            # 保存最终 checkpoint
            if args.save_checkpoint:
                ckpt_path = ckpt_dir / "final.pt"
                nl.save_checkpoint(str(ckpt_path))
                print(f"  最终模型已保存到: {ckpt_path}")
        else:
            print("  跳过: 未找到实例坐标数据")
            method_results["neural_linucb"] = {"status": "缺少坐标数据"}

    if args.method in ("neural_ts", "all"):
        print(f"\n=== NeuralTS (nu=1.0, device={args.device}) ===")
        if coords_dict and ("tsp" in coords_dict or "cvrp" in coords_dict):
            from encoder import DualEncoder
            from neural_ts import NeuralTS

            # 设置 neural_ts 日志 (和 neural_linucb 同样格式)
            ts_log_file = log_dir / f"{run_name}_ts.log"
            ts_logger = logging.getLogger("neural_ts")
            ts_logger.setLevel(logging.DEBUG)
            ts_logger.handlers.clear()
            ts_fh = logging.FileHandler(ts_log_file, mode="w", encoding="utf-8")
            ts_fh.setLevel(logging.DEBUG)
            ts_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
            ts_logger.addHandler(ts_fh)
            if args.verbose:
                ts_ch = logging.StreamHandler()
                ts_ch.setLevel(logging.DEBUG)
                ts_ch.setFormatter(logging.Formatter("%(message)s"))
                ts_logger.addHandler(ts_ch)
            print(f"  日志将写入: {ts_log_file}")

            ts_encoder = DualEncoder()
            nts = NeuralTS(
                n_arms=N_ARMS, encoder=ts_encoder,
                arm_names=ALL_METHODS,
                train_every=args.train_every, train_steps=100,
                max_history=args.max_history,
                device=args.device,
            )

            # 复用 get_coords
            if 'get_coords' not in dir():
                tsp_indices = [i for i, t in enumerate(all_types) if t == "tsp"]
                cvrp_indices = [i for i, t in enumerate(all_types) if t == "cvrp"]
                def get_coords(global_idx):
                    ptype = all_types[global_idx]
                    if ptype == "tsp":
                        return coords_dict["tsp"][tsp_indices.index(global_idx)].numpy()
                    else:
                        return coords_dict["cvrp"][cvrp_indices.index(global_idx)].numpy()

            # 在线交互
            rng_ts = np.random.RandomState(args.seed)
            order_ts = train_idx.copy()
            rng_ts.shuffle(order_ts)
            print(f"  在线交互 {len(order_ts)} 轮 (train_every={args.train_every})")
            ts_logger.info(f"=== 在线训练开始: {len(order_ts)} 轮, nu=1.0, "
                           f"train_every={args.train_every} ===")

            cumulative_reward_ts = 0.0
            cumulative_oracle_reward_ts = 0.0
            oracle_sel_ts = OracleSelector()
            # checkpoint 目录
            ckpt_dir_ts = None
            if args.save_checkpoint:
                ckpt_dir_ts = Path(args.checkpoint_dir) / (run_name + "_ts")
                ckpt_dir_ts.mkdir(parents=True, exist_ok=True)

            for step, i in enumerate(order_ts):
                ptype = all_types[i]
                c = get_coords(i)
                arm = nts.select(c, None, ptype, 100, all_masks[i])
                reward = rank_reward(all_costs[i, arm], all_costs[i], all_masks[i])
                # oracle 对比
                oracle_arm = oracle_sel_ts.select_with_costs(all_costs[i], all_masks[i])
                oracle_reward = rank_reward(all_costs[i, oracle_arm], all_costs[i], all_masks[i])
                ts_logger.info(
                    f"t={step} | TRAIN | {ptype.upper()} | "
                    f"chosen={arm}({ALL_METHODS[arm]}) cost={all_costs[i,arm]:.4f} reward={reward:.4f} | "
                    f"oracle={oracle_arm}({ALL_METHODS[oracle_arm]}) cost={all_costs[i,oracle_arm]:.4f} reward={oracle_reward:.4f} | "
                    f"match={'YES' if arm == oracle_arm else 'NO'}"
                )
                nts.update(arm, c, None, ptype, 100, reward)
                cumulative_reward_ts += reward
                cumulative_oracle_reward_ts += oracle_reward
                if (step + 1) % 500 == 0:
                    avg_r = cumulative_reward_ts / (step + 1)
                    avg_or = cumulative_oracle_reward_ts / (step + 1)
                    regret = avg_or - avg_r
                    msg = (f"    step {step+1}/{len(order_ts)}, avg_reward={avg_r:.4f}, "
                           f"oracle={avg_or:.4f}, regret={regret:.4f}")
                    print(msg)
                    ts_logger.info(f"=== CHECKPOINT {msg.strip()} ===")
                # 周期性保存 checkpoint
                if (args.save_checkpoint and args.checkpoint_every > 0
                        and (step + 1) % args.checkpoint_every == 0):
                    ckpt_path_ts = ckpt_dir_ts / f"step{step+1}.pt"
                    nts.save_checkpoint(str(ckpt_path_ts))
                    print(f"    checkpoint 已保存: {ckpt_path_ts}")

            final_avg_ts = cumulative_reward_ts / len(order_ts)
            print(f"  训练完成, avg_reward={final_avg_ts:.4f}")
            ts_logger.info(f"=== 在线训练完成, avg_reward={final_avg_ts:.4f} ===")

            # 评测
            print("  在 test 集上评测...")
            ts_logger.info(f"=== TEST 评测开始: {len(test_idx)} 实例 ===")
            total_r_ts, total_or_ts = 0.0, 0.0
            n_hit_ts = 0
            oracle_ts = OracleSelector()
            method_counts_ts = np.zeros(N_ARMS)
            for ti, i in enumerate(test_idx):
                ptype = all_types[i]
                c = get_coords(i)
                arm = nts.select(c, None, ptype, 100, all_masks[i])
                oracle_arm = oracle_ts.select_with_costs(all_costs[i], all_masks[i])
                reward = rank_reward(all_costs[i, arm], all_costs[i], all_masks[i])
                oracle_reward = rank_reward(all_costs[i, oracle_arm], all_costs[i], all_masks[i])
                total_r_ts += reward
                total_or_ts += oracle_reward
                hit = (arm == oracle_arm)
                if hit:
                    n_hit_ts += 1
                method_counts_ts[arm] += 1
                ts_logger.info(
                    f"TEST[{ti}] | {ptype.upper()} | "
                    f"chosen={arm}({ALL_METHODS[arm]}) cost={all_costs[i,arm]:.4f} reward={reward:.4f} | "
                    f"oracle={oracle_arm}({ALL_METHODS[oracle_arm]}) cost={all_costs[i,oracle_arm]:.4f} | "
                    f"match={'YES' if hit else 'NO'}"
                )

            n_t = len(test_idx)
            result_ts = {
                "mean_reward": total_r_ts / n_t,
                "oracle_reward": total_or_ts / n_t,
                "regret": (total_or_ts - total_r_ts) / n_t,
                "top1_hit_rate": n_hit_ts / n_t,
                "method_distribution": {ALL_METHODS[k]: int(method_counts_ts[k])
                                        for k in range(N_ARMS) if method_counts_ts[k] > 0},
            }
            print(f"  test: mean_reward={result_ts['mean_reward']:.4f}, "
                  f"top1_hit={result_ts['top1_hit_rate']:.4f}, regret={result_ts['regret']:.4f}")
            ts_logger.info(
                f"=== TEST 结果: mean_reward={result_ts['mean_reward']:.4f}, "
                f"top1_hit={result_ts['top1_hit_rate']:.4f}, regret={result_ts['regret']:.4f} ===\n"
                f"  method_distribution={result_ts['method_distribution']}"
            )
            method_results["neural_ts"] = result_ts

            # 保存最终 checkpoint
            if args.save_checkpoint:
                ckpt_path_ts = ckpt_dir_ts / "final.pt"
                nts.save_checkpoint(str(ckpt_path_ts))
                print(f"  最终模型已保存到: {ckpt_path_ts}")
        else:
            print("  跳过: 未找到实例坐标数据")
            method_results["neural_ts"] = {"status": "缺少坐标数据"}

    # 7. 汇总结果
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    all_results = {**baseline_results, **method_results}
    for name, res in all_results.items():
        if "mean_reward" in res:
            print(f"  {name:30s}: reward={res['mean_reward']:.4f}" +
                  (f", top1_hit={res['top1_hit_rate']:.4f}" if 'top1_hit_rate' in res else "") +
                  (f", regret={res['regret']:.4f}" if 'regret' in res else ""))
        elif "status" in res:
            print(f"  {name:30s}: {res['status']}")

    # 保存结果
    output_path = data_dir / f"results_{run_name}.json"
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                           for kk, vv in v.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
