# R111：方法族条件化的完整选择

状态：代码准备，尚未启动。研究判断由 Astra/max 提供；协议、实现、审查与执行由 Sol/max 完成。R109/R110 健康任务不修改、不抢占。

## 唯一机制

共享原四层节点编码、两层双流、真实输入角色、ID+34 与几何输入。不同方法族在池化之前使用私有的两层完整节点 self-attention/FFN 更新，沿用输入可计算的距离 bias，再聚合形成族内决策证据；不是先池化再加一个注意力读出头。根分类器只读共享表示，不读这些私有证据。最终所有合法方法共同竞争：

`P(method | x) = P(source_group | x) * P(method | source_group, x)`。

这条路径不可被另一个图级分类头绕过。所有非退化合法族都计算，没有用真实 winner 决定运行哪个分支；族内只剩一个合法方法时条件概率恒为1，可跳过无效的族内图计算。训练只用未加权联合 NLL，即正确族的根损失加正确族内部的条件损失；不是人为平衡选择次数、给漏选方法加偏置或只重排 ReLD。

族按既有实现来源预先固定，不根据成本或留出集搜索：

| 族 | 方法 |
| --- | --- |
| diffusion | DIFUSCO、DIFUSCO500、T2T、T2T500 |
| reld | RELD_CVRP、RELD_MOEL（数据别名 reld_mt） |
| decomposition | GLOP |
| other_constructive | 其余合法候选；仅为固定分组，不声称这些架构相同 |

概率链本身不增加理想分类器的表达能力。待检验的是私有条件特征及其监督分配，而不是“分组必然提高准确率”。共享前段与根分支仍接收跨族梯度，不宣称完全消除梯度竞争。历史 R109/R110 的方法私有图分支仍属于重要近邻。

## 固定协议

- 一组，`A_family_conditional`，全部18题；seed2；从头初始化，保留 R84 原始随机参数和 TRAIN-only 归一化，不加载已训练 selector。
- 冻结 NEW90k：每题81,000 TRAIN、4,500 VAL、4,500 TEST；候选删除、单一 reld_mt、native/FP64 winner 不变。不运行 solver、不补标签。
- 两张原授权4090做 DDP，每 rank512，总逻辑 batch1024。TSP/CVRP micro256、ATSP128、其它512，按真实行数归一化；不把 batch 改为2048。
- 57,600 次成功更新，各题3,200次；warmup4,320；AdamW，学习率1e-4余弦至1e-6，weight decay1e-4，dropout0.1，clip1。AMP 溢出两 rank 同步重放同一批，不跳样本。
- 每1,440次完整 VAL，每7,200次完整 TRAIN；严格最低完整 VAL `macro_vs_sbs_pct` 选唯一 checkpoint，精确并列保留先到者。这不是 actual regret 选型，不混称。
- 锁定后重新保存文件绑定的完整 TRAIN/VAL，再做完整 TEST、两 LIB、完整模型推理计时及 PDF。TEST/LIB 不用于改变 checkpoint、组定义、阈值或推理规则。
- 历史 R94B/R102A/R110 是参照，不是新跑的参数量/FLOP/执行路径匹配对照。旧 TEST 已参与研发，不称为全新盲测。
- W&B 启用，沿用现有无凭证时的显式 offline，不另行发送通知。

## 资源与交付

等待 R110 两条完整管线正常释放 gpu05 原 allocation1475 的两张4090后开始，不停止其评测或发布。复用容器和已有 `/tmp/shiys_R84_new90k_cache`，不复制、不重建缓存。R109 继续使用两张3090。

仅做少量必要的 CPU 检查及一次 Sol 代码审查；不新增消融或系数搜索。评估完整18题 Top1、实际 regret、mean cost、TRAIN-SBS/Oracle，并保留固定36组真正救回、误选、净成本及非目标人群。零选择组数下降不是成功标准。交付 combined model/winner PDF、NSS四数据集对照、学习曲线、CSV；整版完成后集中提交一次 GitHub。
