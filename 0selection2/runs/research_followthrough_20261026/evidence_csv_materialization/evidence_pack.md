# 保存结果证据包：数值核验完成，CSV 发布格式已对账

七个唯一 VAL checkpoint 和280行指标沿用原 Sol/max 的实际核验，数值与科学结论未改。主线程另行确认92个CSV差异全部仅为CRLF到LF，实际Git blob与规范化工作文件相等，逐单元格解析内容也完全一致。严格字节SHA仍不同，不把它写成相同。

附加对账见 `publication_csv_materialization.json`；原严格核验、原发布清单、失败动作、checkpoint、预测、CSV和提交全部保留。当前完成口径是科学内容一致并明确记录发布格式转换，不是谎称原严格字节检查通过。不重训、不重测、不重新选型，也不为换行重复提交。

## 实际完成与保留的失败

**R109：** 两个完整模型已有 TRAIN/VAL/TEST、LIB、PDF 和实际提交 43e3432。原 watcher 仍为 `stopped_on_error` / `Publication requires canonical roots`；这是发布路径错误，后续是独立、CPU-only 的发布恢复，没有假称原 watcher 成功，也没有重训、重评 TEST/LIB 或重选权重。[结果][r109]、[恢复认证][r109recovery]。

**R112：** 两臂原 dispatch 仍为 failed，原 train/TEST/timing/LIB 的 rc0 与 report rc1 均保留；CPU 报告恢复成功、联合分析已保存，直接复用指定发布 446ac3c，不重复恢复或发布。恢复输入中两份原失败 dispatch 的 SHA 与现文件一致。[结果][r112]、[CPU 恢复][r112recovery]。

**R113：** Mencius 的 portable-inode 两卡队列下 A/B/C 三个独立 train/finalize 均 rc0；worker 实际绑定 gpu03/job1465 和两张指定 RTX3090 UUID。A 从头开始；B/C 从持久化 update20160 继续，optimizer/scaler/RNG/stream 保留；未保存的 B2286、C1872 次更新是重放，不是恢复成功的历史权重。B/C 的预训练 8640 更新已在历史 4090 上完成；历史 protocol 的 gpu_type=4090 为精确续跑保留，不能据此否认新 worker 的 RTX3090 证据。新队列总状态 blocked 来自 R114 原队列分支，不能替代三个模型各自的完整完成回执。旧四槽失败 controller 与冻结副本字节相同，原 B/C worker 与 ready 中历史 SHA 相同；两个 error.json 仍为 KeyboardInterrupt 且本次核验首尾字节一致。它们更早的独立 error.json SHA 未保存，不把本次首尾核对冒充更早历史哈希认证。[结果][r113]、[主线程 handoff][handoff]。

**R114：** 原 CVRP/ICAM cold receipt 仍失败，其 SHA `d6826cf0468bb946264e3af9f565b3187b0c0fa98a1f0ef87a67f1ea69c26be3` 与修正前 revision 的锚点相同。首个错误是 CUDA tensor 直接交给 NumPy 的公共 native audit 接口错误，不是候选能力失败，更不是响应学习失败。revision.json 仍是 prepared_not_integrated_not_executed；它只证明准备，不证明已修正标签。实际后续执行由另一套已锁定 CVRP scheduler、保存标签数组哈希、RTX3090 worker、combined finish/analysis 及实际提交 62fc8b6 证明。仅用纠正后的完整 CVRP 替换旧 CVRP，其他15题来源保持原样；父 scheduler 仍 blocked、子 scheduler complete，合并 lock_sha256=null，execution_lock 按题映射，没有虚构一个共同执行锁。[最终结果][r114]、[原失败与准备][r114revision]。

## 原假说、实际实现与对照边界

| 版本 | 原假说与实现 | 对照与不能声称的内容 |
| --- | --- | --- |
| R109 | 候选条件图计算的共享/私有化；原两 view CE/比较/relative-risk/R-Drop 复合目标，非纯 CE | 相同 fresh 原主干及分支原型；A 2,305,033/B 10,220,553 参数，不等参数/等 FLOPs |
| R112 | 保留合并前后的接口条件：4/2/1 区域层级，每区域3×3×4=36个潜接口条件，共252 token | A 平均后广播/B 保留条件；完整初始参数哈希和形状相同，各1,878,027参数；不是经认证的资源DP、真实子路线成本或可行性 |
| R113 | 由 TRAIN 合成过程预训练后持续执行预测状态；最多12个片段，非 native 完整求解程序 | 同 fresh 完整主干；A 无预训练、B 独立快照、C 反馈转移；B/C 同目标同预训练预算，A总预算不等；三臂部署均预测反馈 |
| R114 | 原路线是候选响应函数学习；当前仅 ACQUISITION/response pilot，尚未训练函数学习器 | 16个适用需求/TW任务，每题32个TRAIN base；同题/同family/同base的 directed-random全池交集；无TSP/ATSP新响应 |

原假说见[20261025综合研究][review1]及[大范式研究][review2]；实现细节见各 PLAN 与锁定 protocol。R112/R113 的 batch=512、seed2、最多57600成功更新；R109 batch=1024、seed2、同更新上限。均使用冻结 TRAIN81000/VAL4500/TEST4500 每题；R84 TRAIN-only 统计及 fresh 随机资产复用，不以旧训练 checkpoint 冒充 fresh。相同 seeded task/shuffle 协议不等于独立归档了每步全部样本ID。单 seed 的描述性结果不是显著性结论；历史 R94B/R89B 只是相同数据的完整模型参考，不是新鲜等预算因果对照。

**近似实现的阴性不能否定整个范式。完整响应标签也不能证明尚未实现的下游 response learning 成功或失败；不能声称三个范式已充分验证，不能声称 universal full18 improvement。** 本次没有产生任何新 NN arm 或训练计划。

## 七个固定模型的结果

`checkpoints.csv` 恰好7行；每臂一个整体 checkpoint。checkpoint 文件 SHA 与锁一致，protocol 规范化 SHA 与 state_hash 一致；保存的完整 VAL history 重现原预锁选择准则：R109 最小 whole18 VAL macro_vs_sbs_pct，精确并列保留最早；R112/R113 最大 whole18 VAL macro_top1，精确并列取最小 actual regret，再取最早。没有按 TEST/LIB 重选，也没有逐题拼装。

`metrics.csv` 恰好280行：每臂 VAL/TEST 各18题+ALL，以及 TSPLIB/TSP、CVRPLIB/CVRP。VAL/TEST 每题4500、ALL81000；全部采用 native int64 ind/原 FP64 winner，保留稳定候选身份和并列规则，未换成 FP32 argmin。数据版本严格为 `selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`；LIB 为 native_TSPLIB/native_CVRPLIB。候选池哈希来自每臂 protocol.candidate_pools 的 sort_keys 紧凑 JSON；八个含B题不恢复 RF/MoSES，不恢复 RELD_MTL，不合并两个 ReLD 最低成本，RELD_MOEL 原值仅命名 reld_mt。

SBS 在完整 TRAIN 拟合；已从保存 TRAIN 方法均值核对每题所选 SBS。Oracle 是逐例合法池最低成本。vs_SBS=100×(mean_cost/SBS_cost−1)，vs_Oracle 同理；actual regret=100×mean((所选成本−逐例Oracle)/逐例Oracle)。ALL 的7项数值均逐题等权平均；ALL的mean_cost只作描述，不能跨任务混合raw cost作净效益证据。

| ALL split | 实例数 | TRAIN拟合SBS cost | Oracle cost |
| --- | --- | --- | --- |
| VAL | 81000 | 11.427978 | 11.371692 |
| TEST | 81000 | 11.385299 | 11.329932 |

| 模型 | split | Top1 % | mean_cost | vs_SBS % | vs_Oracle % | actual regret % |
| --- | --- | --- | --- | --- | --- | --- |
| R109A | VAL | 71.102469 | 11.416634 | -0.192841 | 0.394804 | 0.404713 |
| R109A | TEST | 71.385185 | 11.373796 | -0.197930 | 0.385240 | 0.394037 |
| R109B | VAL | 71.175309 | 11.417059 | -0.189295 | 0.398365 | 0.408363 |
| R109B | TEST | 71.360494 | 11.373875 | -0.196958 | 0.386208 | 0.393914 |
| R112A | VAL | 71.060494 | 11.417398 | -0.187797 | 0.399854 | 0.410621 |
| R112A | TEST | 71.283951 | 11.374503 | -0.190906 | 0.392293 | 0.400308 |
| R112B | VAL | 71.024691 | 11.417900 | -0.183590 | 0.404089 | 0.415310 |
| R112B | TEST | 71.333333 | 11.374755 | -0.190510 | 0.392669 | 0.401151 |
| R113A | VAL | 70.927160 | 11.417807 | -0.185309 | 0.402390 | 0.413414 |
| R113A | TEST | 71.164198 | 11.374716 | -0.190165 | 0.393068 | 0.402034 |
| R113B | VAL | 70.835802 | 11.417715 | -0.185514 | 0.402182 | 0.412755 |
| R113B | TEST | 71.188889 | 11.374399 | -0.193219 | 0.389976 | 0.398368 |
| R113C | VAL | 70.809877 | 11.417839 | -0.184394 | 0.403311 | 0.414188 |
| R113C | TEST | 71.059259 | 11.374651 | -0.190058 | 0.393183 | 0.401596 |

代号与完整 arm 的一一映射：R109A=A_tied_late；R109B=B_private_late；R112A=A_collapsed；R112B=B_interfaces；R113A=A_scratch；R113B=B_snapshot；R113C=C_transition。

以下是同一组完整模型的逐题 TEST Top1%；每题全部成本、SBS、Oracle、比率和逐例 regret 同时保存在 metrics.csv，VAL逐题及两项LIB也完整列出。

| problem | R109A | R109B | R112A | R112B | R113A | R113B | R113C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TSP | 46.911 | 47.178 | 46.956 | 46.356 | 46.822 | 46.600 | 46.444 |
| CVRP | 48.311 | 48.933 | 48.067 | 47.733 | 46.778 | 47.489 | 46.933 |
| ATSP | 76.667 | 77.067 | 77.022 | 77.444 | 76.800 | 76.889 | 76.244 |
| OVRP | 40.689 | 40.667 | 40.644 | 40.778 | 40.267 | 40.622 | 40.022 |
| VRPB | 80.556 | 80.289 | 80.244 | 80.244 | 80.600 | 80.533 | 80.578 |
| VRPL | 38.111 | 37.000 | 37.844 | 38.000 | 37.911 | 37.267 | 37.000 |
| VRPTW | 81.133 | 81.089 | 81.089 | 81.044 | 80.933 | 80.911 | 80.911 |
| OVRPTW | 83.889 | 83.911 | 83.889 | 83.889 | 83.933 | 83.911 | 83.933 |
| OVRPB | 88.089 | 88.200 | 88.111 | 88.111 | 88.133 | 88.111 | 88.111 |
| OVRPL | 42.156 | 42.222 | 41.689 | 42.733 | 40.511 | 40.956 | 40.800 |
| VRPBL | 79.400 | 79.289 | 79.289 | 79.200 | 79.556 | 79.578 | 79.578 |
| VRPBTW | 78.822 | 78.889 | 78.778 | 78.756 | 78.911 | 78.778 | 78.778 |
| VRPLTW | 79.422 | 79.267 | 79.244 | 79.244 | 79.289 | 79.289 | 79.267 |
| OVRPBL | 89.267 | 89.311 | 89.200 | 89.267 | 89.311 | 89.289 | 89.311 |
| OVRPBTW | 84.889 | 84.800 | 84.956 | 84.889 | 84.800 | 84.844 | 84.844 |
| OVRPLTW | 82.978 | 82.889 | 82.889 | 82.911 | 82.933 | 82.911 | 82.867 |
| VRPBLTW | 79.911 | 79.822 | 79.689 | 79.933 | 79.844 | 79.844 | 79.844 |
| OVRPBLTW | 83.733 | 83.667 | 83.511 | 83.467 | 83.622 | 83.578 | 83.600 |

| 完整模型对照 | split | ΔTop1 pp | Δmean_cost | Δactual regret pp |
| --- | --- | --- | --- | --- |
| R109B−R109A | VAL | 0.072840 | 0.000425 | 0.003650 |
| R109B−R109A | TEST | -0.024691 | 0.000079 | -0.000123 |
| R112B−R112A | VAL | -0.035802 | 0.000502 | 0.004689 |
| R112B−R112A | TEST | 0.049383 | 0.000252 | 0.000843 |
| R113B−R113A | VAL | -0.091358 | -0.000093 | -0.000659 |
| R113B−R113A | TEST | 0.024691 | -0.000317 | -0.003666 |
| R113C−R113B | VAL | -0.025926 | 0.000124 | 0.001433 |
| R113C−R113B | TEST | -0.129630 | 0.000252 | 0.003228 |

R112 B 的 TEST Top1小幅增加，但 VAL Top1下降，VAL/TEST regret均更差；不构成整体联合改进。R113 C 相对B在VAL/TEST的Top1和regret均更差；B相对A仅有局部、小幅表现变化。R109 私有化在固定36组的TEST recall从0.674206%增至1.761635%，救57、丢7、产生105个新误选；同时整体TEST Top1下降0.024691pp、regret只下降约0.000123pp，VAL regret变差。固定36组的纠正/伤害与新误选不是互不重叠的全体错误分解，不能直接相减推出ALL收益。R94B 历史完整模型 TEST Top1=71.577778%、regret=0.379115%，当前七个完整模型均未同时超过这两个描述性参考值。这不是已知上限或 Bayes 最优性证据。[R109][r109]、[R112][r112]、[R113][r113]。

## 零选择与71%左右表现的证据边界

| 模型 | TRAIN Top1 % | VAL Top1 % | TEST Top1 % | TEST零选择但native赢家>0的题×候选组 |
| --- | --- | --- | --- | --- |
| R109A | 71.481824 | 71.102469 | 71.385185 | 22 |
| R109B | 71.996091 | 71.175309 | 71.360494 | 19 |
| R112A | 71.672085 | 71.060494 | 71.283951 | 20 |
| R112B | 71.483745 | 71.024691 | 71.333333 | 22 |
| R113A | 71.083608 | 70.927160 | 71.164198 | 35 |
| R113B | 71.100069 | 70.835802 | 71.188889 | 33 |
| R113C | 71.038272 | 70.809877 | 71.059259 | 35 |

这里统计的是当前各模型的**精确零选择**组，不是固定历史36组，也不是选择次数≤0.1%的近零组；各模型均覆盖同89个合法题×候选组。例如三个R113模型在TSP TEST都从未选择LEHD，但LEHD实际是762/4500例native winner。这两种频数来自selection_distribution.csv不同panel，不能把winner频率当作模型频率，更不能把真实winner系列标作100%模型准确率。已读取每臂18页同轴对照PDF；CSV逐面板计数、比例与样本数自洽，PDF的20个TEST/LIB“模型Top1=”标题与相应逐例一致率一致。

这些结果只说明当前实现仍有漏选及近似71%的表现聚集。**后验argmax的合法最优策略可以对某候选零选择**：其边际winner频率非零，不保证存在让它后验冠军概率最大的可识别区域；本次没有测到真实Bayes后验。反过来，零选择也不证明无信息或必然不可学；保存的rescued/harmed和双方均错成本变化提供的是特定已实现策略的可部署变化证据。TRAIN拟合的最低平均成本SBS、条件多数winner先验、最小期望相对成本是不同决策对象；不能把Top1的小变化代替逐题决策成本。任何“可学的漏选”判断都必须保留救回、伤害及题内成本，不能由零频数推断 Bayes 最优或信息不存在。[20261025综合研究第2、3、4、6节][review1]给出了这些区分；本次没有新增最优性或信息量测量。

## NSS：七个同版本模型各自对照

28项对照均有保存结果，无缺失。NSS复用R79历史权重，未在new90k重训；TSP/CVRP的NSS checkpoint分别不同，它们是两题历史参照，不能称一个新训练full18 NSS模型。TEST使用本次冻结新数据/合法池；LIB保留旧native数据。每一行当前模型均绑定本臂唯一checkpoint，不跨版本挑最佳。NSS表的R109未独列state/update字段，按current_checkpoint_sha256与该臂final lock绑定，没有杜撰额外原列。R112/R113保存的NSS对齐说明每数据集抽核3行、FP64成本按候选名称对齐；本次复用其对齐说明，未重做全量原始数据审计。

| 模型 | 数据集 | n | 模型Top1 % | 模型mean_cost | NSS Top1 % | NSS mean_cost |
| --- | --- | --- | --- | --- | --- | --- |
| R109A | TSP TEST | 4500 | 46.9111 | 7.971084 | 45.8889 | 7.970401 |
| R109A | CVRP TEST | 4500 | 48.3111 | 18.009787 | 45.9778 | 18.004802 |
| R109A | TSPLIB | 49 | 22.4490 | 8.215799 | 20.4082 | 8.217907 |
| R109A | CVRPLIB | 100 | 61.0000 | 67.190011 | 70.0000 | 67.101112 |
| R109B | TSP TEST | 4500 | 47.1778 | 7.971649 | 45.8889 | 7.970401 |
| R109B | CVRP TEST | 4500 | 48.9333 | 18.010666 | 45.9778 | 18.004802 |
| R109B | TSPLIB | 49 | 28.5714 | 8.205443 | 20.4082 | 8.217907 |
| R109B | CVRPLIB | 100 | 63.0000 | 67.227001 | 70.0000 | 67.101112 |
| R112A | TSP TEST | 4500 | 46.9556 | 7.972859 | 45.8889 | 7.970401 |
| R112A | CVRP TEST | 4500 | 48.0667 | 18.012468 | 45.9778 | 18.004802 |
| R112A | TSPLIB | 49 | 32.6531 | 8.196547 | 20.4082 | 8.217907 |
| R112A | CVRPLIB | 100 | 64.0000 | 67.242449 | 70.0000 | 67.101112 |
| R112B | TSP TEST | 4500 | 46.3556 | 7.973580 | 45.8889 | 7.970401 |
| R112B | CVRP TEST | 4500 | 47.7333 | 18.011106 | 45.9778 | 18.004802 |
| R112B | TSPLIB | 49 | 26.5306 | 8.199875 | 20.4082 | 8.217907 |
| R112B | CVRPLIB | 100 | 63.0000 | 67.256356 | 70.0000 | 67.101112 |
| R113A | TSP TEST | 4500 | 46.8222 | 7.972356 | 45.8889 | 7.970401 |
| R113A | CVRP TEST | 4500 | 46.7778 | 18.018578 | 45.9778 | 18.004802 |
| R113A | TSPLIB | 49 | 24.4898 | 8.209452 | 20.4082 | 8.217907 |
| R113A | CVRPLIB | 100 | 48.0000 | 67.599797 | 70.0000 | 67.101112 |
| R113B | TSP TEST | 4500 | 46.6000 | 7.971642 | 45.8889 | 7.970401 |
| R113B | CVRP TEST | 4500 | 47.4889 | 18.013341 | 45.9778 | 18.004802 |
| R113B | TSPLIB | 49 | 30.6122 | 8.191982 | 20.4082 | 8.217907 |
| R113B | CVRPLIB | 100 | 49.0000 | 67.547212 | 70.0000 | 67.101112 |
| R113C | TSP TEST | 4500 | 46.4444 | 7.972612 | 45.8889 | 7.970401 |
| R113C | CVRP TEST | 4500 | 46.9333 | 18.014831 | 45.9778 | 18.004802 |
| R113C | TSPLIB | 49 | 28.5714 | 8.195919 | 20.4082 | 8.217907 |
| R113C | CVRPLIB | 100 | 60.0000 | 67.295075 | 70.0000 | 67.101112 |

## R114：完整标签，不是已训练响应学习器

**CURRENT EXECUTION PHASE = ACQUISITION/response pilot；NOT a trained function learner。** 实际权威 closeout_kind=complete_labels；9600/9600候选成本、2048/2048全池winner、1536/1536全池响应对完整。对应16题×32个TRAIN基础实例；多端点/同base关联，不能当独立样本。无TSP/ATSP新响应；没有下游NN、TEST/LIB准确率、full18模型或普遍数据改进结论。完整标签不能建立下游响应学习的成功或失败。

| problem | 端点 | 实际/计划候选成本 | 全池winner | 完整/计划响应对 |
| --- | --- | --- | --- | --- |
| CVRP | 96 | 960/960 | 96 | 64/64 |
| OVRP | 96 | 576/576 | 96 | 64/64 |
| OVRPB | 96 | 288/288 | 96 | 64/64 |
| OVRPBL | 96 | 288/288 | 96 | 64/64 |
| OVRPBLTW | 160 | 480/480 | 160 | 128/128 |
| OVRPBTW | 160 | 480/480 | 160 | 128/128 |
| OVRPL | 96 | 576/576 | 96 | 64/64 |
| OVRPLTW | 160 | 960/960 | 160 | 128/128 |
| OVRPTW | 160 | 960/960 | 160 | 128/128 |
| VRPB | 96 | 288/288 | 96 | 64/64 |
| VRPBL | 96 | 288/288 | 96 | 64/64 |
| VRPBLTW | 160 | 480/480 | 160 | 128/128 |
| VRPBTW | 160 | 480/480 | 160 | 128/128 |
| VRPL | 96 | 576/576 | 96 | 64/64 |
| VRPLTW | 160 | 960/960 | 160 | 128/128 |
| VRPTW | 160 | 960/960 | 160 | 128/128 |

24个题×family条件均为32个同base全池配对；directed−random的RMS相对响应差18项正、6项负，winner-flip差16项正，其余含负值及零。不是所有条件都受益。平均幅度比范围1.078266–1.163273，所以没有严格匹配干预幅度，不能把差异都归因为“定向机制”。CVRP demand的RMS差为+0.424028pp、flip差+9.375pp；OVRPL demand的flip差−12.5pp；OVRPTW tw_center的flip差−15.625pp。响应存在、响应幅度、winner翻转和下游可泛化学习是四个不同命题。[完整结果与逐条件表][r114]；原失败不能混入候选能力或学习成败判断。

## 历史 R57 与运行条件

[R57已知边界][r57]必须保留：FP64成本/原生winner对齐不等于任务语义一致；历史B任务的signed-residual-load与classical backhaul可行域不同。R57静态检查覆盖128合法题×候选，但原输入runtime只完成31/128；没有任何一行被称作完整历史认证。TorchRL ABI、部分NSS导入、RF/MoSES绑定与历史resolved配置仍有未闭合项；992个原输入成本重现不能认证未存档的历史依赖修订。当前候选过滤和ReLD别名是已冻结的版本规则，不回填旧数据，也不把所有历史标签宣判错误。

R57旧VAL的R45A Top1=49.1167%，条件多数先验42.4667%，同题同size置换均值39.7110%；这是旧数据/候选版本上已存在实例信号的历史证据，不是对当前90k的71%上限估计。置换并非均匀随机排列，分位数非总体置信区间；小规模执行角色翻转也不解释整体平台。本次未重新遍历300000条路线、权重、solver或实例，更没有“修复native contract以促成R114成功”。

[20261025 running_snapshot.csv][oldrunning]中R109/R111的training状态是当时快照；本次以实际final lock/complete/recovery/publication认证，不重写旧状态。R112历史训练/计时在4090、CPU恢复复用旧计时；R113历史预训练4090与本轮两3090续跑分别记录。R109 timing.json只记录cuda:0，未独列device_name，不由此编造硬件实测；R112/R113的device_name分别记4090/3090。计时只是完整模型、GPU已驻留输入、首512例/题的保存测量，3次重复、1次warmup，排除I/O、CPU→GPU传输及求解器，不是端到端耗时；不同设备/预算不可直接作架构效率因果比较。本次不访问GPU、不测时、不更改现有队列。

**TEST/LIB已经参与历史研发，不是全新盲测。** 此次唯一VAL锁的重用并不消除这种研究暴露。whole18持续改进目标仍ACTIVE；本次换行对账不关闭目标；后续仍按已有研究授权推进。

## 可复核来源

每臂的`final_test_lock.json`、`protocol.json`、`final_metrics.json`、`history.json`及`full_model_statistics/{metrics_by_problem.csv,nss_comparison.csv,selection_distribution.csv,MODEL_STATISTICS.pdf,selection_distributions.pdf}`的实际绝对路径与SHA见[saved_results_authentication.json](saved_results_authentication.json)。已核验7个checkpoint字节/21份PDF；所有附加输入的核验首尾SHA相同。原始权重只计算文件SHA，没有反序列化，没有模型forward。

| 模型 | 实际run目录 |
| --- | --- |
| R109A | `/public/home/shiys/0selection2/code/V4/runs/R109_method_conditioned_nodes/A_tied_late` |
| R109B | `/public/home/shiys/0selection2/code/V4/runs/R109_method_conditioned_nodes/B_private_late` |
| R112A | `/public/home/shiys/0selection2/code/V4/runs/R112_region_interfaces/A_collapsed` |
| R112B | `/public/home/shiys/0selection2/code/V4/runs/R112_region_interfaces/B_interfaces` |
| R113A | `/public/home/shiys/0selection2/code/V4/runs/R113_executable_process/A_scratch` |
| R113B | `/public/home/shiys/0selection2/code/V4/runs/R113_executable_process/B_snapshot` |
| R113C | `/public/home/shiys/0selection2/code/V4/runs/R113_executable_process/C_transition` |

[r109]: /public/home/shiys/0selection2/code/V4/runs/R109_method_conditioned_nodes/RESULTS.md
[r109recovery]: /public/home/shiys/0selection2/code/V4/runs/R109_method_conditioned_nodes/preparation/publication_recovery_20261026/final_verification.json
[r112]: /public/home/shiys/0selection2/code/V4/runs/R112_region_interfaces/RESULTS.md
[r112recovery]: /public/home/shiys/0selection2/code/V4/runs/R112_region_interfaces/preparation/recovery_20261026/recovery_complete.json
[r113]: /public/home/shiys/0selection2/code/V4/runs/R113_executable_process/RESULTS.md
[handoff]: /public/home/shiys/0selection2/code/V4/runs/R113_executable_process/preparation/migration_3090_portable_inode_20261026/HANDOFF.md
[r114]: /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/finish/tensor_boundary_combined_20261026/RESULTS.md
[r114revision]: /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/revision.json
[r57]: /public/home/shiys/0selection2/code/V4/runs/R57_task_signal_audit/RESULTS.md
[review1]: /public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/REPORT.md
[review2]: /public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/REPORT.md
[oldrunning]: /public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/running_snapshot.csv
