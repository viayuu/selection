# 2cmab2 结果分析

## 已收集运行

- 运行数量: 2
- 方法列表: linucb, neural_linucb

## Selector Test Greedy 最优结果

- 按 `mean_reward` 最佳: `linucb` = 0.783550
- 按 `mean_regret` 最佳: `linucb` = 0.051070

## Baseline 对比

- baseline 按 `mean_reward` 最佳: `single_best_global` = 1.000000
- baseline 按 `mean_regret` 最佳: `oracle` = 0.000000

## Selector vs Baseline

- selector 最佳 `mean_reward` 与 baseline 最佳的差值: -0.216450
- selector 最佳 `mean_regret` 与 baseline 最佳的差值: 0.051070

## 输出文件

- `runs_manifest.csv`: 运行清单
- `selector_test_greedy.csv`: selector 的 greedy 测试指标
- `selector_test_ucb.csv`: selector 的 UCB 测试指标
- `baseline_test_greedy.csv`: baseline 测试指标
- `selector_overall_ranking.csv`: selector 总体排名表
- `best_summary.json`: 最佳结果摘要
