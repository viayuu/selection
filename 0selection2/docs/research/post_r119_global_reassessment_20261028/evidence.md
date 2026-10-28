# 证据索引与科学解释边界

事实冻结于 **2026-10-28 13:27 CST**。R119完整选择器仍在运行；本文不使用此后进度或结果。状态标签：**完成**为已完成资产/结果；**快照**为用户及主线程提供的运行中观察；**实现**只证明代码表达的机制；**历史**为旧实验报告，其数据与控制范围需单独解释；**推断**不当作实验事实。

## 核心证据

|ID|来源与定位|状态|支持的事实/机制|不支持的结论|
|---|---|---|---|---|
|E01|[AGENTS.md](/public/home/shiys/0selection2/AGENTS.md)，2026-10-20数据/Batch及2026-10-22上下文条款|冻结约定|18题new90k；81,000/4,500/4,500；89合法部署；native winner；TRAIN-only上下文例外|不是新划分、改候选池、恢复R58或GPU授权|
|E02|[上次执行条件报告](/public/home/shiys/0selection2/docs/research/execution_conditions_and_symmetry_20261027/REPORT.md)|历史科学分析|固定部署D与运行实现Z的区分；物理角色；当时无就绪主方向|报告中的假设不自动成为新的实验证据；给定commit引用未用Git复核|
|E03|[R118 RESULTS](/public/home/shiys/0selection2/code/V4/runs/R118_native_trace_assets/RESULTS.md)；[observer](/public/home/shiys/0selection2/code/V4/r118_reld_trace.py:97)；[capture入口](/public/home/shiys/0selection2/code/V4/r118_trace_assets.py:113)|完成资产/实现|准确目录R118_native_trace_assets；144capture；136坐标capture成本接近；8ATSP不匹配；18教师部署；约973,743非终止概率行；CVRP观测列恢复|不是选择器结果；概率行不是独立实例；成本接近不保证同一轨迹|
|E04|[R119 PLAN](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/PLAN.md)；[锁定协议](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/protocol_lock_uuid_v2.json)；[annotations](/public/home/shiys/0selection2/code/V4/r119_annotations.py)|完成注释资产/实现|17,408既有TRAIN实例；15,232预训/2,176过程诊断；17,272新调用+136复用；输入分层采样；17个教师部署；不改历史成本/winner/split|不代表全部89候选行为；不保证优势区覆盖；fresh seed不是历史replay|
|E05|[targets](/public/home/shiys/0selection2/code/V4/native_process_targets.py:19)；[student](/public/home/shiys/0selection2/code/V4/native_process_student.py:132)；[model](/public/home/shiys/0selection2/code/V4/r119_model.py:83)|实现|三输入计数宏相位；POMO×节点pair；自身软状态反馈；candidate-specific full-memory读出40维；原始节点路径保留；ATSP矩阵路径无过程监督|不证明完整world model、离散native rollout、联合状态可行性或全部候选仿真；维度本身不证明信息不足|
|E06|[loss定义](/public/home/shiys/0selection2/code/V4/native_process_loss.py:38)；[预训练与诊断模式](/public/home/shiys/0selection2/code/V4/r119_experiment.py:595)；[A预训练收据](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/A_snapshot/pretrain/complete.json)；[B预训练收据](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/B_transition/pretrain/complete.json)；[过程CSV](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/process_statistics/process_diagnostic.csv)|完成预训练/诊断|各8,160步；A snapshot/B transition诊断；current=CE/logN，policy=KL/logN，资源归一化MSE；诊断TRAIN来源；全选择器无process aux|不是盲VAL、accuracy或最终状态保持证据；A/B未统一部署模式做过程诊断|
|E07|[REQUEST冻结快照](/public/home/shiys/0selection2/docs/research/post_r119_global_reassessment_20261028/REQUEST.md)及用户13:27消息；已读[R119 STATUS](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/STATUS.md)但其较旧记录不覆盖新快照|运行中快照|A17,280/B17,352；均选17,064；VAL Top1 71.216049/71.181481%；regret .407684/.414612%；无最终TEST/LIB|不宣称完成、优胜、因果增益；不因本文阅读时间较晚而更新截止|
|E08|[预算RESULTS](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/error_budget_reference/RESULTS.md)；[categories](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/error_budget_reference/categories.csv)；[定义代码](/public/home/shiys/0selection2/code/V4/r119_error_budget.py:118)|完成保存预测分析|R94B/R117B当前18×4,500；rank2占regret73.56/73.82%；rank是模型分数排名；macro relative regret分母|不是R119结果；不是cost rank；不是Top2可部署上限；两模型不等于多seed重复|
|E09|[partitions](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/error_budget_reference/partitions.csv)与[methods](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/error_budget_reference/methods.csv)|完成保存预测分析；少量CSV计数核对|固定36：R94B 4,481漏选、21.21% regret；当前严格零28总部署但22有winner，R117B 27总部署但21有winner；各6个无winner；1,405/1,035漏选不变|28/27不能转述为positive-winner部署；动态零组变小不等于真实恢复；与固定36不能相加|
|E10|[pairs](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/error_budget_reference/pairs.csv)与[method_pairs](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/error_budget_reference/method_pairs.csv)|完成保存预测分析|跨任务reld_mt→MVMOE占40.96% regret；前三跨任务53.10%；前三task-pair仅16.11%，17项覆盖50%|不支持只做一个task-pair专家；不证明13个任务具有相同错误机理；两种粒度不能相加|
|E11|[metrics](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/error_budget_reference/metrics.csv)；[source_manifest](/public/home/shiys/0selection2/code/V4/runs/R119_native_state_transfer/error_budget_reference/source_manifest.json)|完成保存预测分析|R94B TEST71.577778%/.379115%；R117B71.512346%/.387063%；完整18题表；来源为完成模型|历史参考非R119匹配对照；TEST已经研发暴露|
|E12|[R47 README](/public/home/shiys/0selection2/code/V4/R47_README.md)；[R71 README](/public/home/shiys/0selection2/code/V4/R71_README.md)；[机制索引](/public/home/shiys/0selection2/docs/research/experiment_mechanism_index.csv)|历史实现说明|R47可训练检索+成本/winner/mask+差分修正；R71八分量全协方差成本差混合、Sobol冠军积分、joint NLL|不是弱KNN或独立成本回归；旧协议负结果不覆盖所有检索/生成模型|
|E13|[综合方向回顾](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/REPORT.md)；[大范式回顾](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/REPORT.md)|历史综合证据|R53独立pair；R54–56路线/边监督与载体；R67对比；R78校准/成本/邻域；R81/82排序；R99全局输入竞争|不把每个历史负结果扩张成整个大类已被否定；未逐行重审全部旧代码|
|E14|[大范式结果](/public/home/shiys/0selection2/docs/research/post_major_paradigm_results_20261026/REPORT.md)；[R80–R102机制索引](/public/home/shiys/0selection2/docs/research/experiment_mechanism_index_R80_R102.csv)|历史结果|R94静态迁移；R101/102 hint及donor；R103条件输入信号；R104加宽加深；R105门控FFN；R106解冻；R107连接路径；R108条件先验；R109/110私有晚期计算和纯CE|不能仅凭编号记忆把R104称模块化、R107称任务干扰、R108称task batch；没有统一匹配架构排行榜|
|E15|[R116前方向报告](/public/home/shiys/0selection2/docs/research/post_r116_major_direction_20261027/REPORT.md)|历史结果/机制|R112层次latent接口不等于资源DP；R113合成片段预训及transition对snapshot负差|不等于真实教师过程已被否定；latent接口保留不等于物理兼容性已学会|
|E16|[R114/R115背景](/public/home/shiys/0selection2/docs/research/post_r116_major_direction_20261027/REPORT.md)；[R114完整池与旧神经提案](/public/home/shiys/0selection2/docs/research/post_major_paradigm_results_20261026/REPORT.md:172)；[R115实际目标](/public/home/shiys/0selection2/code/V4/r115_pilot.py:145)|历史完成结果/实现|16任务完整候选池响应；512base/2,048endpoint/9,600cost；R115为中心化全候选变化的多输出Ridge，四条件跨base劣于零/均值；714维/每折每题16训练base|不是单方法对预测；差分、全池及神经响应概念已有先例；失败是真负证据，小样本不构成无限扩量理由|
|E17|[R116方向报告](/public/home/shiys/0selection2/docs/research/post_r116_major_direction_20261027/REPORT.md)，R116机制与结果段|历史完成结果|两臂都有champion CE及ranking NLL；条件消去改变排名/冠军表现但未建立联合目标突破|不能误称只监督尾部排序；不支持“再加冠军loss就是新方向”|
|E18|[R117 RESULTS](/public/home/shiys/0selection2/code/V4/runs/R117_native_computation/RESULTS.md)；[R117科学判断](/public/home/shiys/0selection2/docs/research/post_r117_scientific_decision_20261027/REPORT.md)|历史完成结果|B-A TEST净53正确，1,123纠正/1,070误伤；regret上升；固定36 129命中/257选择；输入定义参考状态|不是已访问native前缀；不支持原生计算充分性，也不否定所有真实过程迁移|
|E19|[early graph context报告](/public/home/shiys/0selection2/docs/research/early_graph_context_followup_20261027/REPORT.md)|历史科学提案|早期原图结构/标签绑定与晚期摘要不同；强局部late对照必要；固定任务函数不因episode重组而改变|不是已完成新实验或early优于late的证据|

## 自行推导与假设登记

|结论|性质|依据与边界|
|---|---|---|
|R94B Top2 native-winner覆盖90.8%|简单算术推导|(57,978+15,570)/81,000；不是可部署Top1|
|严格零组完美无误伤修复的Top1上界1.734568pp|离线算术上界|1,405/81,000×100；固定36相应5.532099pp；不预测可达收益|
|两类微小/零cost错误占约1.46%|简单算术推导|(116+220)/23,022；不推断Bayes噪声或输入充分性|
|utility差maximin的argmax仍为最高utility|代数结论|合法候选数至少2；并列使用同一tie规则；不意味着概率及梯度相同|
|正文§7.2的0-1与regret代理上界|本文代数推导|被选动作的utility不低于其它动作，故相应softplus/log2至少1；成本regret非负；仅为上界，不保证优化或泛化|
|D1可能补充优势区域覆盖和学习归纳偏置|研究假设|错误预算+历史迁移反证+S05/S06/S07；尚未证明数据缺口是主因；R115为关键反证|
|平方响应loss只约束族内残差变化|本文代数推导|写为E范数平方(e端点2-e端点1)；族内公共候选偏差不受此项惩罚，故响应好不保证winner正确|
|D2可能减少代理与硬动作的失配|研究假设|现有全信息成本、双目标不等价、S03/S04；不能保证非凸确定性模型可同时改善|
|D3–D7的排序|研究判断|实际缺失对象、历史重复度、全18可用性及额外标签/计算成本；不是实验胜率估计|

## 阅读与未执行范围

已读取要求中的七组主要研究上下文及early-context后续，检查相关机制索引、R47/R71实现说明、R118结果、R119计划/状态/协议、预训练完成metadata、过程诊断和核心targets/student/loss/model/experiment/annotations/reporting相关实现。错误预算读取已完成CSV、报告和定义；没有访问R119未完成预测。

未加载checkpoint张量，未运行前向或solver，未启动/停止作业，未做GPU或重数据操作，未执行Git命令，未写入本研究五个指定文件以外的位置。原论文检索和阅读范围见`sources.csv`。本文不要求增加审计清单、重新哈希所有资产或重复计算现有指标。
