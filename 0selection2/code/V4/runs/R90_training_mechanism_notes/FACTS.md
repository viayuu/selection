# R90：训练机制事实与局部梯度说明

范围：只读 V4 源码/README，以及 R84A 已保存的 TRAIN-selected/VAL 预测；不读 TEST/LIB，不加载权重，不推理、不训练、不修改 R89 或旧源码。本文不是实验计划，也不改变任何版本的选型或接收标准。

## 1. Maximin 保留排名，但不保留 softmax 或 CE 梯度

当前 ScoreDifferenceSelector 先输出 utility，再形成 `margin[m,k]=u_m-u_k`，对合法对手取最小值；因此

\[
z_m=u_m-\max_{k\ne m}u_k.
\]

代码依据：[pairwise_selector.py:11](/public/home/shiys/0selection2/code/V4/pairwise_selector.py:11)、[同文件:27](/public/home/shiys/0selection2/code/V4/pairwise_selector.py:27)。以下结论仅针对这种可由 utility 差表示的比较，不推广到任意独立 pair-margin 网络。

若唯一第一名为 t、第二名为 s，记 Δ=u_t-u_s>0，则 `z_t=Δ`，其它 `z_j=u_j-u_t`。所以完整排序和 argmax 与 u 相同，包括数学上的并列关系；同一冻结前向只把 maximin 改为 raw-utility argmax，不会救回零选择。单候选时源码令 z=0。

令 q=softmax(u)、p=softmax(z)。相对居中的 u，z 只把当前第一名额外提高 Δ：第一名的未归一化质量乘上 exp(Δ)，其它项不变。两个候选时等价于 softmax(2u)；超过两个候选时一般不是统一温度缩放。

对未加权单次 CE，令 g=p-onehot(y)。在唯一 t、s 的可微区域，Jacobian 给出

\[
\nabla_{u_t}CE(z)=2g_t,\quad
\nabla_{u_s}CE(z)=g_s-g_t,\quad
\nabla_{u_j}CE(z)=g_j\;(j\notin\{t,s\}).
\]

raw-utility CE 的梯度则为 q-onehot(y)，不是把上述 g 直接换个名字；概率本身也变了。精确 ties 处要按实现选取的次梯度处理，不能机械使用唯一 t、s 公式。

极小 CPU/FP64 autograd 实例：`u=[2,1,0]`，真实 winner 为第3项。`z=[1,-1,-2]`：

| 项 | raw utility CE | maximin CE |
|---|---|---|
| softmax | [0.665241, 0.244728, 0.090031] | [0.843795, 0.114195, 0.042010] |
| CE | 2.407606 | 3.169846 |
| 对 u 的梯度 | [0.665241, 0.244728, -0.909969] | [1.687589, -0.729600, -0.957990] |

因此真实 winner 不在 Top2 时，错误第二名确实也得到负梯度，即独立坐标梯度下降会提高它。但不能据此断言有害：此时 winner 的上推幅度为 `1-p_y`，第二名为 `p_t-p_s`，两者之差为 `2p_s+Σ_{j∉{t,s,y}}p_j>0`，所以局部独立 utility 坐标步仍改善 winner 相对第二名的差距。共享网络参数、其它 loss、裁剪和优化器下的实际变化不能从该坐标反例直接推出。四个极小 AD 实例均通过公式核对，见 `tiny_ad_examples.csv`。

## 2. 当前 loss 与更新路径；历史上并非从未用过 utility CE/多任务 step

R89 的工厂固定使用 R88 execution-role 模型，且直接返回原 `RDropObjective`：[r89_experiment.py:32](/public/home/shiys/0selection2/code/V4/r89_experiment.py:32)、[同文件:56](/public/home/shiys/0selection2/code/V4/r89_experiment.py:56)。R88 继承 ScoreDifferenceSelector，仅预处理输入：[r88_model.py:96](/public/home/shiys/0selection2/code/V4/r88_model.py:96)。

| 原目标 | 实际读取 | 证据 |
|---|---|---|
| CE，系数0.35 | maximin `output['logits']`，native `ind` 为目标 | [pairwise_objective.py:63](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:63) |
| compare，系数0.35 | `pair_margin`，即 u 的两两差；预测 Top3 聚焦来自两次 maximin logits | [同文件:39](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:39)、[同文件:45](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:45) |
| risk，系数0.02 | maximin softmax 加权 FP64 标签派生的 relative regret；逐例上限1，再平均除0.01 | [同文件:67](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:67) |
| R-Drop | 同一批两次 dropout 的 maximin logits，合法池对称 KL | [r76_experiment.py:164](/public/home/shiys/0selection2/code/V4/r76_experiment.py:164)、[r42_experiment.py:59](/public/home/shiys/0selection2/code/V4/r42_experiment.py:59) |

单次逻辑更新确实只包含一个任务的1024例：`task_order` 每18任务一轮打乱，并避免跨轮相邻任务相同；每任务独立 ShuffleStream，尾段接下一次 shuffle 补满，不因尾批丢例。请求按 `(problem, indices,1024)` 构造，然后逐请求更新，不是把18任务合成一次 step。microbatch 按真实样本数累积后，只 unscale/clip/step 一次。证据：[r77_experiment.py:52](/public/home/shiys/0selection2/code/V4/r77_experiment.py:52)、[同文件:66](/public/home/shiys/0selection2/code/V4/r77_experiment.py:66)、[r84_experiment.py:380](/public/home/shiys/0selection2/code/V4/r84_experiment.py:380)、[同文件:408](/public/home/shiys/0selection2/code/V4/r84_experiment.py:408)、[r76_experiment.py:222](/public/home/shiys/0selection2/code/V4/r76_experiment.py:222)。参数和优化器状态仍跨任务共享，“单任务 step”不等于训练独立模型。

限定检索已有 V4 源码/README，找到明确历史实现：R64 的 NativeUtilitySelector 直接用 score MLP 的 u 作 logits，目标为两次 raw-utility CE + KL；R64/R65 每次在相同参数点计算三个不同任务梯度，再平均/投影、裁剪并 step 一次。R65 README 还明确正式 A/B、256例/任务、有效768。证据：[r64_model.py:16](/public/home/shiys/0selection2/code/V4/r64_model.py:16)、[同文件:40](/public/home/shiys/0selection2/code/V4/r64_model.py:40)、[R64_README.md:23](/public/home/shiys/0selection2/code/V4/R64_README.md:23)、[r65_gradients.py:31](/public/home/shiys/0selection2/code/V4/r65_gradients.py:31)、[R65_README.md:7](/public/home/shiys/0selection2/code/V4/R65_README.md:7)。这不是当前 NEW90k、完整原 compare/risk loss 下的单因素对照；本次未重查其运行完成回执或成绩。R85 A/B 的 raw-utility OVR **BCE** 也不是多类 raw-utility **CE**：[R85_README.md:73](/public/home/shiys/0selection2/code/V4/R85_README.md:73)。

## 3. 已存 R84A 的范围证据，不是重训收益证据

直接复用同一模型状态、update37440 的 TRAIN `selected_best` 与 VAL `selected_best` logits、FP64 costs、native winner；36 targets 使用原冻结清单，未重新挑目标。每题分别81,000/4,500例；排名按分数降序、global-ID 升序打破精确并列。regret 为 `(selected_cost-Oracle)/Oracle`；ALL 保持18题等权。

| 划分/范围 | 漏选数 | winner rank≥3 | 占该范围漏选 | 占该范围 regret | 占全体 regret |
|---|---:|---:|---:|---:|---:|
| TRAIN 全体错例 | 419,264 | 136,178 | 32.48% | 26.49% | 26.49% |
| TRAIN 固定36目标 | 82,982 | 38,833 | 46.80% | 37.40% | 8.39% |
| VAL 全体错例 | 23,421 | 7,600 | 32.45% | 26.52% | 26.52% |
| VAL 固定36目标 | 4,614 | 2,113 | 45.80% | 35.64% | 7.99% |

表中固定目标范围只包含“真实 winner 属于该题冻结目标”的实例，不包含任意被选为目标的方法。36目标总体占 TRAIN/VAL regret 的22.44%/22.41%；其中 rank≥3 仅占全体8.39%/7.99%，不可将前述比例当作可恢复收益。例：VAL TSP/LEHD 的792个漏选，rank2为414、rank3为293、rank4为30、rank5为45、rank6/7各4、rank8为2。

仅在 rank≥3 且 top/second 唯一的行上，对照 CE 的局部 utility 梯度：错误 second 上推幅度 / true-winner 上推幅度的逐行比值均值，TRAIN/VAL 全体为0.2193/0.2273，36目标为0.3135/0.3205；相应 true-winner 梯度幅度 / 同前向重建 raw-CE 幅度均值为1.0182/1.0188、1.0223/1.0226。该子集没有 top/second ties。存档仅有 z，raw u 由 `u_t=0,u_j=z_j` 恢复至平移（浮点近似），未冒称保存了实际 raw utility。以上只计算未加权 CE；不含 compare/risk/KL、参数 Jacobian 或优化器步。

可反驳边界：用户此前 NSS 单任务训练也有零选择，故不能把当前 maximin 或任务轮转写成共同根因；排名统计只界定当前局部机制涉及的范围。它不证明该机制有害、解释平台，也不能从同前向梯度推导更换训练后的收益。

附件：`winner_rank_summary.csv`（各目标/各题/ALL）、`winner_rank_distribution.csv`（逐名次）、`ce_gradient_summary.csv`、`tiny_ad_examples.csv`、`archive_provenance.json`。CPU复算：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -m code.V4.runs.R90_training_mechanism_notes.summarize_saved`（先激活 easynco）。未运行任何 GPU 操作。
