# 当前部分结果汇总

- run目录: `/public/home/zhoucl/shiys/2cmab2/outputs/2026-04-02-11-17-56_nss_neural_linucb_server`
- 当前用于分析的checkpoint: `best.pt` (step=6000)
- 当前best checkpoint在val集上的top1 accuracy: `0.435000`
- 当前best checkpoint在val集上的mean cost: `13.110217`

## periodic val中最好的top1

- mode=`val_ucb` | step=`6000` | top1=`0.443000` | mean_cost=`13.112447`

## periodic val中最好的mean cost

- mode=`val_greedy` | step=`6000` | mean_cost=`13.110217` | top1=`0.435000`

## top1 / mean cost 对比

- selector_best_checkpoint: top1=`0.435000` | mean_cost=`13.110217`
- single_best_global: top1=`0.358500` | mean_cost=`13.205983`
- single_best_per_problem: top1=`0.358500` | mean_cost=`13.205983`
- oracle: top1=`1.000000` | mean_cost=`13.023971`

## TSP 单方法对比

- selector mean_cost: `7.941977`
- selector top1_accuracy: `0.293000`
- 在 mean_cost 上超过的方法: `t2t500, difusco500, t2t, difusco, bq, lehd, elg`
- 在 mean_cost 上未超过的方法: `无`
- 在 top1_accuracy 上超过的方法: `t2t500, difusco500, t2t, difusco, lehd, elg`
- 在 top1_accuracy 上未超过的方法: `bq`

## CVRP 单方法对比

- selector mean_cost: `18.278457`
- selector top1_accuracy: `0.577000`
- 在 mean_cost 上超过的方法: `omni, bq, lehd, elg, mvmoe`
- 在 mean_cost 上未超过的方法: `无`
- 在 top1_accuracy 上超过的方法: `omni, bq, lehd, elg, mvmoe`
- 在 top1_accuracy 上未超过的方法: `无`
