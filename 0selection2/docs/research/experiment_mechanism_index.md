# R60-R79 现有机制索引

记录日期：2026-10-20。固定20个版本，每版本一行；机器可读文件为
[experiment_mechanism_index.csv](experiment_mechanism_index.csv)。

本次只读各版 README、RESULTS 或顶层最终摘要。未读取训练实现、预测数组、
大型权重，也未运行模型、GPU巡检、launcher、结果重算、论文调研或新实验规划。
这是一份防止机制重复的历史索引，不是启动 gate、代码审计或性能排行榜。
R64/R65/R75只登记机制与完成状态，详细数值由另一个独立整理负责。

## 字段口径

- **LEGACY18**：原18题、原完整合法池，通常每题 TRAIN10000、VAL1000、TEST1000；
  保留两种 ReLD，以及带B题的 RF/MoSES。该阶段文档披露的B语义问题没有在这些实验中修复。
- **NEW70114**：`selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014`；
  每题 TRAIN70114、VAL3895、TEST3895。八个带B题删除 RouteFinder、MoSES_RF、MoSES_CaDA；
  删除 RELD_MTL，RELD_MOEL 原成本列改名 `reld_mt`，不是两者取最小。
- **raw-utility CE**：CE直接使用未经 maximin 的utility/原始分类头logits。
  `yes` 不表示它是一次独立、严格匹配的CE接入点消融。`no` 也不表示没有分类监督。
- **mixed-task step**：同一次成功 optimizer.step 聚合多个问题的梯度。
  微批累积、两次dropout、同题多DDP rank，都不自动算跨任务step。
- **unknown**：上述小文档没有明确写出，未越界查代码补推断；不等于机制未尝试。
  **planned** 只指计划描述；**not_applicable** 指该版本没有对应训练目标或新step。
- 完成状态以已发布结果/最终状态优先于旧的启动计划。继承、微调、复用对照、
  无训练集成和局部诊断明确分开，不能都算成新训完整模型。各版数据、目标、预算不同，
  此表不支持跨版因果归因。

## 逐版本索引

| Version | 核心机制 | 数据/候选池 | Loss与训练调度 | raw-utility CE | mixed-task step | 实际TRAIN/TEST状态 | 小文件证据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R60 | 原完整选择器额外学习MOEL/MTL比较：旧TRAIN回放对新增独立实例；非局部专家 | LEGACY18；额外15类MVRP部分成本，不造全池winner | 原目标/R-Drop + 0.35辅助比较BCE；batch128/AdamW/warmup3；A控制共同LR和停止轴；VAL成本选型 | unknown：未单独说明原CE接入点 | unknown：未明确跨任务合step | A/B TRAIN/VAL完成；B的TEST后来由R63交付补做；A正式TEST证据unknown | [README](../../code/V4/R60_README.md)、[RESULTS](../../code/V4/runs/R60_expanded_partial_supervision/RESULTS.md)、[R63补做TEST](../../code/V4/runs/R63_recurrent_workspaces/RESULTS.md) |
| R61 | 保存路线后继/END预训练实例Encoder；只迁移Encoder，再训练全池选择器 | LEGACY18；旧路线回放对新增保存路线 | 两轮路线预训练；后续原比较/risk/R-Drop；batch128/AdamW；A控制微调轨迹；VAL成本选型 | no：阶段2明确maximin logits；路线CE不是utility CE | unknown | A/B预训练及微调完成；明确无TEST访问 | [README](../../code/V4/R61_README.md)、[RESULTS](../../code/V4/runs/R61_behavior_pretraining/RESULTS.md) |
| R62 | 源分离组合性能预训练；迁移主干、重置utility头；同源新10k回放对90k | LEGACY18下游；辅助成本缺失仅mask监督，不缩池或造winner | 观测方法对BCE预训练两遍；原R43/R-Drop微调；batch128/AdamW；VAL成本选型 | unknown：预训练为utility差BCE，微调CE接入未单列 | unknown | A/B预训练+40E微调完成；明确无TEST访问 | [README](../../code/V4/R62_README.md)、[RESULTS](../../code/V4/runs/R62_portfolio_performance_pretraining/RESULTS.md) |
| R63 | 候选专属节点workspace，邻接消息+GRU八次共享递推；初始邻居记忆对动态通信 | LEGACY18原池 | 两视图native CE+R-Drop；无pair/risk；micro16累计128；AdamW/warmup3；VAL Top1选型 | yes：直接分类头logits，明确非maximin | unknown：只说明原batch累积 | TRAIN/VAL/TEST两组完成；A40E、B按用户修订15E，非等预算对照 | [README](../../code/V4/R63_README.md)、[RESULTS](../../code/V4/runs/R63_recurrent_workspaces/RESULTS.md) |
| R64 | fresh direct-utility双流；三任务梯度均值对共享dense前缀PCGrad | LEGACY18原池 | native CE/R-Drop；大real batch重启；同状态三题梯度合并、clip一次；A控制共同调度；VAL整数Top1 | yes | yes：3任务/step | 正式重启A/B TRAIN/VAL/TEST完成；旧batch42归档不是完整结果 | [README](../../code/V4/R64_README.md)、[RESULTS](../../code/V4/runs/R64_pcgrad_full18/RESULTS.md) |
| R65 | 同初值NativeUtilitySelector：AdamW对official ScheduleFree，部署averaged-x | LEGACY18原池 | native CE/R-Drop；256×3；40E/9600step；720步warmup到3e-4后常数；VAL整数Top1 | yes | yes：3任务/step | fresh A/B TRAIN/VAL/锁定TEST完成 | [README](../../code/V4/R65_README.md)、[RESULTS](../../code/V4/runs/R65_schedule_free_full18/RESULTS.md) |
| R66 | 冻结R65B的F，训练新g；A训练g、B训练F+g，两者均部署F+g | LEGACY18原池；继承基模型 | native CE/R-Drop；ScheduleFree；256×3；40E/9600step；720步warmup；完整bundle按VAL整数Top1选型 | yes：g或F+g | yes：3任务/step | 两新成员及完整系统TEST完成；基模型属历史计算 | [README](../../code/V4/R66_README.md)、[RESULTS](../../code/V4/runs/R66_additive_correction_full18/RESULTS.md) |
| R67 | 同winner SupCon预训练对分类预训练；丢projector并重置整个utility MLP，再全参数微调 | LEGACY18原池；不新增标签 | 阶段1 A CE/R-Drop、B 0.1×SupCon温度0.1；阶段2 CE/R-Drop；ScheduleFree；256×3；40+40E | yes：A阶段1及两组阶段2；B阶段1无CE | yes：3任务合step；SupCon配对不跨题 | 两组两阶段及锁定TEST完成 | [README](../../code/V4/R67_README.md)、[RESULTS](../../code/V4/runs/R67_supervised_contrastive_full18/RESULTS.md) |
| R68 | 同一R67B预训练主干+fresh头：普通AdamW对SAM radius0.05 | LEGACY18原池；非从头Encoder | native CE/R-Drop；同RNG全三任务SAM重放；256×3；40E/9600step；warmup720 | yes | yes：3任务均值，SAM两遍梯度/step | A/B微调及锁定TEST完成；继承预训练另计 | [README](../../code/V4/R68_README.md)、[RESULTS](../../code/V4/runs/R68_sharpness_aware_finetuning/RESULTS.md) |
| R69 | 18个完整独立task-specific分支；仅新训B，复用R68A对照 | LEGACY18原池；原预训练主干+fresh头 | native CE/R-Drop；普通AdamW；256×3；40E/9600step；不活跃分支不更新；同一epoch整bank选型 | yes | yes：3个不同分支同step | 新B TRAIN/VAL/TEST完成；A只复用，不重训/重测 | [README](../../code/V4/R69_README.md)、[RESULTS](../../code/V4/runs/R69_task_specific_finetuning/RESULTS.md) |
| R70 | 18分支bank的latent mixup：encode_pairs后、完整score_head前混合 | LEGACY18原池；对照复用R69B | 0.5 clean CE+0.5 mixed CE+clean R-Drop；AdamW；256×3；40E/9600step；warmup720 | yes：clean/mixed均经utility头 | yes：3任务合step；mixup只在题内 | 新B TRAIN/VAL/TEST完成；A复用 | [README](../../code/V4/R70_README.md)、[RESULTS](../../code/V4/runs/R70_latent_mixup/RESULTS.md) |
| R71 | 8分量全协方差Gaussian mixture学习联合成本差分布；固定Sobol积分选赢家概率最大者 | LEGACY18原池；TRAIN标准化；继承主干，替换scalar头 | 仅joint NLL/(K-1)；AdamW；256×3；40E/9600step；warmup720；2048固定Sobol向量/分量 | not_applicable：winner CE仅报告 | yes：3任务/step | 新B TRAIN/VAL/TEST完成；A复用R70B；不是仅协方差消融 | [README](../../code/V4/R71_README.md)、[RESULTS](../../code/V4/runs/R71_joint_performance_mixture/RESULTS.md) |
| R72 | 固定冻结R60B/R67B/R70B异构三策略，整数midrank/Borda求和 | LEGACY18原池 | 零训练；每例三前向；Borda CE仅报告；固定VAL后锁定TEST | not_applicable | not_applicable | 固定策略VAL/新TEST完成；零新TRAIN，不能算新训模型 | [README](../../code/V4/R72_README.md)、[RESULTS](../../code/V4/runs/R72_heterogeneous_consensus/RESULTS.md) |
| R73 | 三策略教师在扩容输入生成hard伪标签，warm-start完整R67B共享student | LEGACY18原池；额外每题90k输入/共1.62M教师预测，不是实测新winner | native CE+R-Drop+pseudo CE；ScheduleFree；每题256 native+256 auxiliary，三题合step；40E/9600step | yes | yes：3任务/step | 教师生成及新B TRAIN/VAL/TEST完成；A复用R72；非纯覆盖因果对照 | [README](../../code/V4/R73_README.md)、[RESULTS](../../code/V4/runs/R73_expansion_teacher_training/RESULTS.md) |
| R74 | 冻结基础选择器/solver，通过前缀token、cross-attention、获取门学习有界信息获取与utility纠正 | LEGACY18原池；原TRAIN/VAL前缀观测计划 | 计划三叶期望native CE+R-Drop、ScheduleFree、256×3/40E；实际只少量更新 | planned_yes；实际部分运行接入unknown | planned_yes_3_tasks；实际部分运行unknown | **paused/进程已终止、磁盘保留**；日志91/耐久60；无完整policy VAL、选中锁或新TEST | [README](../../code/V4/R74_README.md) |
| R75 | 完整标签扩容，保留R65B/ScheduleFree；RAM packing/prefetch仅IO续跑 | 旧完整池；TRAIN1786887且各题不等量；原VAL/TEST不动 | native CE/R-Drop；fresh seed2；384×3；ScheduleFree；40E上限；成功更新warmup；VAL整数Top1 | yes | yes：3任务/step | 单组TRAIN/VAL/锁定TEST完成；IO续跑不是多组模型；详细数值另行整理 | [README](../../code/V4/R75_README.md)、[最终摘要](../../code/V4/runs/R75_large_complete_labels/FINAL_RUN_SUMMARY.md) |
| R76 | R45A完整手工34维+ID双流/maximin，新场景从头训练，四卡DDP大batch | NEW70114全18题；单一reld_mt | 0.35 CE+0.35 compare+0.02 risk+R-Drop；AdamW1e-4；global9856；warmup3E；平台LR/早停；VAL成本选型 | no：maximin分类路径 | no：同题DDP不算混题 | 单组TRAIN/VAL/TEST完成；26E/3744step；后续同锁LIB另有已发布评估 | [README](../../code/V4/R76_README.md)、[RESULTS](../../code/V4/runs/R76_R45A_single_reld_mt/RESULTS.md) |
| R77 | 嵌套分层10k对全70114的匹配成功更新预算对照；R45A模型/目标不变 | NEW70114同VAL/TEST；A/B各自TRAIN geometry；共同FULL TRAIN SBS | batch1024；各49680step；warmup3726/cosine到1e-6；每1242步VAL；不早停；VAL成本选型 | no：原ScoreDifference/maximin路径 | no：每step单题1024例 | A/B TRAIN/VAL/TEST完成；保留唯一R77B；B同锁TEST/两LIB交付完成 | [README](../../code/V4/R77_README.md)、[RESULTS](../../code/V4/runs/R77_sample_size_batch1024/RESULTS.md) |
| R78 | 六轮合并：冻结偏置CE/代价二次、风险单调续训、冻结代价头、固定概率均值、成本画像邻域续训 | NEW70114；001/002基于R76，其余基于R77 | CPU拟合/无训练集成；GPU沿用R77全模型续训14904步，改risk或加0.10邻域损失；batch1024；预定VAL gate | no：post-maximin CE或原R77；部分路线无CE | no：各题独立拟合或R77单题step；集成无step | 001/002 **inconclusive**；003-006有限预算negative；无新TEST、无接收候选；保留R77B | [README](../../code/V4/R78_README.md)、[六轮最终摘要](../../code/V4/runs/R78_zero_selection_goal/final_delivery/zero_selection_retrospective/RETROSPECTIVE.md) |
| R79 | 原NSS分层pooling分别重训TSP/CVRP；不使用V4双流或额外零选择惩罚 | NEW70114仅TSP8/CVRP10；原TSPLIB49/CVRPLIB100 | RankingLoss(K,K)；Adam1e-4/WD1e-6；batch1024增强样本、micro64；8倍XY增强×50E/27400step；无调度/AMP/早停；VALID gap选型 | not_applicable：ranking loss非CE | no：两题独立模型 | 两模型TRAIN/VALID/锁定TEST/LIB完成；非18任务模型；DP数值尝试零正式DP更新 | [README](../../code/V4/R79_NSS_README.md)、[REPORT](../../code/V4/runs/R79_NSS_new_data_batch1024/REPORT.md) |

## 复用索引时的边界

- 这里已明确记录 direct-utility CE、三任务合step、PCGrad、ScheduleFree、SAM、
  监督对比预训练、任务专属bank、latent mixup、性能分布、固定集成和教师伪标签，
  不能把它们再次描述成项目从未尝试。机制包不同，也不能据此断言严格匹配消融已做过。
- R60原结果是TRAIN/VAL；其B后来在R63交付时补做了完整TEST，不能简单写成R60从未TEST。
  R61/R62的无TEST记录没有因此改变。
- R72无新训练，R74未完成，R78包含校准/集成/续训及不收敛试验，R79只有两题。
  它们不构成额外的从头全18模型训练次数；R78机制已按主路线合并，未展开几十种变体。
- R79 README的最终口径明确区分native/FP64与历史FP32-argmin Top1；旧REPORT保留兼容口径。
  本索引不搬运数值，不将两种标签口径混用为一次性能结论。
- 所有状态只表达所引文档中的历史交付事实，未据此检查当前进程、重认证权重或修复历史基准。
  本次不提出新实验、恢复paused任务，也不新增启动前检查要求。
