# 历史结果：R64 / R65 / R75（限定现成报告）

只摘录每版 README、RESULTS、R75 final summary，以及报告直接引用的两份 method_metrics.csv；未读取权重、预测或大数据，未重放训练/TEST。以下“完成”有现成结果报告支撑，不再仅凭实现存在；未另审原始回执。R88/R89 不干预。

## 1. 完成情况与实际配置

| 正式组 | 完成情况 / 选中 epoch | 数据与候选 | 真实更新方式 / 优化器 |
|---|---|---|---|
| R64A / B | 均完成40轮；均选40；均有完整18题VAL/TEST | 原每题10k TRAIN、1k VAL、1k TEST；旧完整池 | 三任务同参数点梯度；A均值，B仅选定共享参数PCGrad；AdamW |
| R65A / B | 均完成40轮；A选38，B选40；均有完整18题VAL/TEST | 同旧划分、旧完整池 | 三任务梯度均值，无PCGrad；A AdamW，B官方AdamWScheduleFree、部署平均x权重 |
| R75 单组 | 完成40轮、选40；完整18题VAL/TEST | Oct13扩容快照：TRAIN 1,786,887，仍含原10k；旧VAL/TEST及旧池 | fresh R65B架构、三任务梯度均值、ScheduleFree平均x部署 |

三版均直接用 **score MLP 的 raw utility 做 native CE + 两次 dropout 对称KL**，不是 maximin CE；不含当前R77完整目标的 compare/risk。仍有完整双流、34维方法特征+ID，不是无方法编码的简化分类器。三任务在同一参数状态分别求梯度后汇总，一次clip/step，不是每个任务各step一次。

R64A/B 的对照实际检验 **PCGrad vs 均值**，不是 rawCE vs maximinCE；R65A/B 检验固定配方下的 **优化器/权重平均整包**，不是混合任务 vs 单任务轮转。不能把其结果算作已经完成当前 NEW90k、原完整loss 下的这两项单因素实验。

| 预算 | R64A/B（每组） | R65A/B（每组） | R75 |
|---|---:|---:|---:|
| 每任务batch / 常规有效batch | 256 / 768 | 256 / 768 | 384 / 1,152 |
| 成功更新 / presentations | 9,600 / 7.2M | 9,600 / 7.2M | 62,640 / 71.47548M |
| warmup更新 | 720 | 720 | 4,698 |
| LR控制 | 峰值1e-4；A平台控制，B重放A实际轨迹 | 3e-4 warmup后恒定；无控制器/早停 | 限读报告未明确数值LR，不从“沿用R65B”擅自认证 |

R64旧 batch42 分支为**中止**：A完成14轮+部分15轮、21,383次更新，B未启动、无TEST；不得与正式256版并列为完整结果或匹配batch实验。所有正式组均seed2、按VAL整数正确数选唯一checkpoint，非当前按VAL regret选型。

## 2. 已有完整18题结果

均为原native标签、18题等权；mean cost和TRAIN详见CSV，不混合不同任务raw cost作因果解释。

| 组 | VAL Top1% | TEST Top1% | VAL actual regret% | TEST actual regret% |
|---|---:|---:|---:|---:|
| R64A | 48.6944 | 47.3778 | 0.938619 | 0.981460 |
| R64B | 48.7611 | 47.4111 | 0.940973 | 0.977711 |
| R65A | 47.7722 | 46.4722 | 1.044606 | 1.070499 |
| R65B | 48.7222 | 47.6389 | 0.945528 | 0.978280 |
| R75 | 46.361111 | 44.866667 | 1.727817 | 1.700811 |

这是“已有训练完成且有结果”的证据，不是机制有效/无效的隔离证明。R64/R65还改变LR/控制器；R75改变数据来源、标签部署、规模覆盖、batch、更新数和TRAIN统计。R75不是后来 new-only / B-filtered / single-reld，更不是当前 NEW90k。三版均未解决旧B语义问题；不据跨cohort Top1断言更多数据有害、rawCE无效或准确率上限。

## 3. 现成零选择统计及缺失

只对两份已有标量CSV计数：**pick_fraction=0 且 method_top1>0** 的“问题×方法”组数；不是重放预测。保持原CSV的method_top1口径，不升级为已复核的unique-winner计数，不套用当前固定36targets。

| 模型 | TRAIN | VAL | TEST |
|---|---:|---:|---:|
| R64A/B | 缺失 | 缺失 | 缺失 |
| R65A | 48 | 34 | 35 |
| R65B | 55 | 40 | 41 |
| R75 | 41 | 47 | 48 |

每份可用CSV每划分128个方法组；CSV同时保留所有pick=0组数，包含该划分method_top1本身为0的组。R64在限读报告未见零选择汇总，不代表其没有，未补算。R65B VAL/TEST的零选择正Top1组包括 TSP/LEHD、DIFUSCO；R75仍有TSP/DIFUSCO、OMNI及CVRP/RouteFinder。

可支持的边界：**直接utility CE和同step多任务汇总，并未在这些已完成旧配方中自动消除零选择**；不能据此推出当前maximin/轮转是共同根因，也不能预测它们在当前数据、loss、预算上的重训收益。

来源：[R64结果:3](/public/home/shiys/0selection2/code/V4/runs/R64_pcgrad_full18/RESULTS.md:3)、[配方/中止:28](/public/home/shiys/0selection2/code/V4/runs/R64_pcgrad_full18/RESULTS.md:28)、[R64控制器:45](/public/home/shiys/0selection2/code/V4/R64_README.md:45)；[R65完成/对照:3](/public/home/shiys/0selection2/code/V4/runs/R65_schedule_free_full18/RESULTS.md:3)、[指标:20](/public/home/shiys/0selection2/code/V4/runs/R65_schedule_free_full18/RESULTS.md:20)、[CE/三任务:9](/public/home/shiys/0selection2/code/V4/R65_README.md:9)；[R75目标:33](/public/home/shiys/0selection2/code/V4/R75_README.md:33)、[指标:9](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/RESULTS.md:9)、[预算:44](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/FINAL_RUN_SUMMARY.md:44)、[数据限制:59](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/FINAL_RUN_SUMMARY.md:59)；零选择来源为各版RESULTS直接引用的[R65 method_metrics.csv](/public/home/shiys/0selection2/code/V4/runs/R65_schedule_free_full18/method_metrics.csv)与[R75 method_metrics.csv](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/method_metrics.csv)。

