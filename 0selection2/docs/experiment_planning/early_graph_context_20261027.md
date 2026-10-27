# 早期图上下文：whole18 两臂可行性判断

日期：2026-10-27。**建议：NO-GO，现在不启动训练；资源上有条件可实现，不是已证明不可行。**
仅交付本文；未运行 GPU、读取/改动 R117、分析其结果、调用子代理或提交代码。
用户指定 `gpt-6.1-sol / reasoning=max`；当前工具未暴露本会话底层模型标识，不能声称已核实。

## 1. 要回答的唯一问题

固定 solver 性能映射、同一冻结数据下，支持图局部结构与整图优势的绑定，
在 query 后续图内消息传递之前进入，是否优于**保留节点、有向边、多尺度信息的强晚期匹配读出**？
不制造新函数族、不新增标签、不调用求解器；整图成本不是边级真值，也不声称发现因果边贡献。
Astra 的 REPORT §5/§7 和 directions.csv 的 D01 定义了这个问题，并未推荐自动启动。

**判断分开：** 下述具体对象足以区分计算位置，值得作为一个有边界的实现候选；
但现有 API 尚不能保证完整 base-family 排除，端到端支持重算的预算也无实测依据。
不应为“上下文方向还没穷尽”直接投入一次两臂 whole18，更不以缩水实验填补证据。

## 2. 冻结的数据与合法支持

- 数据根：`/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`。
- 全18题，每题 TRAIN/VAL/TEST = 81,000/4,500/4,500；TRAIN 共 1,458,000 条。
- 候选身份、顺序、mask 完全沿用 release；八个 B 问题的删除规则及单一 `reld_mt` 不变。
- 支持只取同题冻结 TRAIN 的原始图；绑定 native winner、完整合法池 FP64 相对成本、solver ID/mask。
- 本 release 的 winner 口径是过滤合法池内的 FP64 argmin（原生 `ind`）；不从 FP32 成本重算。
- 复用 `BehaviorBank` 的 gap 定义及 `log1p(gap/.01)` 输入变换，但不复用其整图摘要来替代局部图。
- query 只见原始输入/约束。其成本、winner、排名、比较目标等仅进监督；`inputs_only` 后另接独立支持通道。
- 支持标签编码为按稳定全局 ID 排列的完整 gap、winner one-hot、合法 mask；不按 winner 分组求原型。
- 不使用学习过的 donor 表示、求解路线、轨迹、状态或新求解结果；不需要用户另许的新标签。

**固定检索：** 同题全部 81,000 TRAIN 是索引库，每个 query 取 S=4 个支持，排序仅依赖输入。
键固定由现有确定性 12 维 input stats、坐标题 `node_geom` 的逐维 mean/max、`log1p(n)` 组成37维；
ATSP 的几何槽置零，不虚构坐标；逐维标准化只拟合 TRAIN，常数维置零。
这是检索摘要，不是最终模型表示。CPU 分块精确距离检索，按距离及稳定 source ID 解并列。
TRAIN/VAL 两臂使用相同邻居表；TEST/LIB 只在 checkpoint 锁定后按同规则独立检索。
不得用标签挑邻居/改配额，不按成本接近程度组 episode，不联合编码一批 TEST query。

**排除先于 top4：** 排除自身行、同 source/base-family 及输入哈希识别出的关联副本，之后再取4个；不足则失败，不回填亲属。
`source_index` 只用于身份排除，不能作为预测特征。VAL/TEST/LIB 从不成为支持记录。
**未解前提：** 现读数据 API 仅暴露 source_index，没有通用 family ID。
`base_instance_ids` 对坐标做重排/D4 保守分组，但对 ATSP 仅哈希原矩阵字节，不能排除重编号同源矩阵。
必须先确定原始 source 到 base-family 的来源映射，或有生成来源保证一族一 ID；不能把精确去重冒称完整排除。
这是尚未证实的接口前提，不是已发现数据泄漏；不要求重划分或新建数据。

## 3. 最小但不丢图局部性的实现对象

两臂均保留每个 query/support 的全部原始节点、原始距离/有向矩阵、角色和约束；不跨图按节点编号对齐。
宽度 d=128，4头，FFN=512；现有4层 `InstanceEncoder` 拆为共享输入 stem 的前2层 P 和 query 后2层 T。
support 仅经 P；其权重与本臂 query 的 P 共享，每次按当前权重重算，不能缓存旧输出。
wrapper 须保留原 stem 的条件/统计特殊 token、mask 和 bias，不能直接拿现有最终 `node_only` 输出冒充层间接口。
这仍是 raw TRAIN 上下文；支持没有带标签执行图内编码，绑定发生在后述局部匹配器。

**两臂同等保留的局部对象：**
- 节点：原始属性、角色、P 输出及 T 输出；坐标题另保留现有 4/8/16 邻域的12维几何特征。
- 有向关系：坐标题保留带符号位移、距离、两端原始属性；ATSP 分开保留 D[a,b] 与 D[b,a]。
- 匹配边集合 E：每点16个 outgoing 近邻及其反向/incoming 集合，加远端和双向 depot 关系，最多约36N条。
- 边携带 4/8/16 的 outgoing/incoming rank 标记；ATSP 按有向成本各自排序，不对称化。
- 定长邻居的边界并列按原生输入顺序解；现有 tie-inclusive `node_geom` 不改写，不宣称 solver 对重排不变。
- 完整距离/有向成本仍进入 T 的稠密 relation bias，其它属性随节点保留；E 只限制跨图匹配，不裁原图。

**一个明确的新模块 C，而非未定义的“上下文 Transformer”：**
1. 用 query 节点与每张支持图节点的32维可训练投影求软对应，角色 mask 保留 depot/客户及 FROM/TO 语义。
2. 每个 query 端点在每张支持图保留至多 L=4 个合法匹配节点；depot 可仅1个，选择离散，选中键/值保留梯度。
3. 对 query 边 a->b，只比较端点对应的最多16个支持有向对 u->v；去掉 u=v，读取原始关系，不要求硬对应。
4. 关系 token 为 `Wsrc*h_u + Wdst*h_v + Wr*r_uv + Wscale*scale_uv`，宽度 c=32。
5. 将完整支持标签投影 b_s 加入该 token 后经 GELU、32维 key/value 投影；与 query 关系 token 匹配。
6. softmax 同时使用端点对应分数和 label-bound 关系分数；每张支持图先独立归一化，再学习图间权重。
7. 输出逐节点 M_a、逐有向边 M_ab；节点通道同样使用局部 token 与 b_s 的非线性绑定。

因此支持图多节点不是多份独立标签，图大小不会靠 token 数量自动获得更多支持权重。
b_s 进入局部匹配的 key/value，而非只把同一 winner 先验广播给所有节点。
匹配器还向读出提供 query-reference 的局部 token 差，避免删掉强晚期控制的差异校正能力。
S=4、端点 top4 是显式近似，不等于穷尽支持局部对应；阴性结果也不能否定所有早期上下文。

## 4. 只移动一次条件化位置

令 J 用相同投影将 M_a 加到节点，将 M_ab 投影为4头有向 relation-bias 残差；原始关系不被改写。
两臂各自训练，但参数清单、张量形状、初值完全相同；同名4层参数复制 R84 的**随机初始化**，不是已训练权重。
C/J/新读出用一个固定种子初始化并逐张量复制；本臂内 P 共享，跨臂不共享优化更新。

```text
共同：H0 = P(raw_query); S0 = P(raw_TRAIN_support); labels 另行绑定
早期：M = C(H0, S0, raw_relations, support_labels)
      H2 = T(J_node(H0,M), raw_bias + J_edge(M))
      Hread = H2
晚期：H2 = T(H0, raw_bias)
      M = C(H2, S0, raw_relations, support_labels)
      Hread = J_node(H2,M)
共同：edge_read = (raw_relations, raw_bias + J_edge(M), M)
共同读出：合法候选作为 query，读取 H0/H2/Hread、有向边/尺度 token、M 和局部差异，输出全池 scores
```

早期读出同样读取 J 后的边与 M；晚期也保留全部原始属性及多尺度 skip，不先均值池化。
共同读出固定为逐节点/边的 Linear-GELU token 投影（128维），以及两层候选交叉注意力（4头、FFN512）和标量评分头。
候选 token 更新，但节点/边 key-value 不互相更新；晚期不能在读出中偷做一次带标签的 query 图消息传递。
早期的标签绑定会改变 T 中的节点间传播及有向注意力；晚期只能在独立形成图内关系后联合判别。
两臂 C/J 各调用一次、T 各两层，全部参数在两臂均有有效用途；不是给早期多加一个 GNN。
匹配位置不同会自然改变对应和梯度路径，不能宣称两臂的函数类或 FLOPs 严格完全相同。
可识别的结论限于**这一局部匹配近似下的计算位置/归纳偏置**，不是普遍表达能力定理。
这里移动完整 label-bound 匹配器，不能仅凭两臂证明标签本身必要，也不能单独归因于纯标签注入时机。
若要求后者，必须先固定两臂同位置的无标签跨图交互、只移动标签依赖残差；本文未把这个更强设计冒称已解决。

## 5. 可复用 API 与必须新增的边界

- `new_split_data.NewSplitDataset` / `scenario_solver_ids`：冻结池与 source metadata；`r76_cache.ScenarioCache.batch`：原始输入缓存。
- `V4Model.InstanceEncoder` / `EncoderLayer` / `MultiHeadAttention(attn_bias=...)`：P/T、方向偏置及局部张量。
- `local_geometry` / R84 `geometry_stats.json`：现有坐标尺度与 TRAIN-only 归一化；不拿几何特征冒充 ATSP 边。
- `behavior_memory`：复用 gap/身份排除原则，不能直接复用旧 summary cache 作为本对象。
- 新增仅需独立 wrapper 拆 stem/layers、输入派生邻居索引、C/J、局部候选读出和支持 gather；不是现成开关。
- 输出维持 `logits/pair_margin/solver_mask`，以 score differences 接现有监督/评分接口。
- 现有 `inputs_only` 只是 label-key 黑名单；新支持字段须独立白名单，不能把 query targets 混入 support payload。

以上是接口可接性判断，不是已实现、已测试或已通过 gate；不修改公共旧模型及任何 R117 文件。

## 6. 跨支持成本：可计算，不是免费记忆

从 R84 TRAIN 缓存 metadata 实读：TSP 最大499节点、CVRP最大500、ATSP最大100，其余15题最大101。
以下取 query/support 都按 N=100/500 的两个规模端点，不是全18平均或严格性能上下界；矩阵/掩码和 kernel 开销另有增加。

| 项目 | 粗算与含义 |
| --- | --- |
| 4层原编码器前向 | `4*(24*N*d^2 + 4*N^2*d)`，约0.18/1.30 GFLOP；乘加算2 FLOP |
| 查询4层 + 4支持各2层 | 约原单查询编码3倍：0.54/3.90 GFLOP/query；不是仅多一个 readout |
| 稀疏边跨支持 | `S*|E|*L^2 <= 2304N` 个关系对；32维 K/V 投影约1/5 GFLOP/query，节点匹配另加 |
| 强晚期/共同读出 | 两层128维候选交叉注意力，主要 K/V 投影约0.5/2.4 GFLOP/query，另有 token 投影 |
| 总训练量级 | 前向约2–12 GFLOP/query；双前向 R-Drop、反传及 matcher checkpoint 后约15–85 GFLOP/query |
| 两臂完整预算 | 每臂58,982,400次 query 暴露；合计约1.8–10 EFLOP，非运行时预测 |
| 支持重算次数 | 每臂235,929,600条支持引用；双前向约471,859,200次2层支持编码，去重前 |
| 完整边对边注意力 | N=500、S=4 时约2500亿边对/query；仅FP16分数约500GB，明确不选 |

mu=4 的固定 query microbatch、AMP、edge chunk=512、匹配器激活 checkpoint；按真实样本累积至逻辑1024。
不复制 K 份 query 图，不把逻辑 batch1024 全部局部激活常驻。支持 gather 可在同一 microbatch 去重，不能复用陈旧计算图。
N=500 时单 query 的4支持节点对应分数约1M个；mu=4、4头单层 query attention 约4M个元素。
若把全部匹配 K/V 展开，mu=4 时仅这对张量合计约1.2GB（FP16），反传还有多份；按512边分块降至约34MB临时 K/V。
因此24GB有合理实现路径，但未测峰值；目标上限18GiB/卡，显式 bias 也不能假定一定走 Flash 快核。
两张卡各跑一臂，不把两张24GB误算成一张48GB，也不借用 R117 的显存/作业。

**数据 I/O：** 原始坐标缓存至多约80N字节/图，ATSP约4N²；每 query 的5图约40–200KB量级。
每臂完整57,600-update训练 gather 约2.4–11.8TB，尚未计 page/随机读放大；R-Drop 两次复用原始缓冲，不重复磁盘读。
TRAIN 原始 padded 输入上界约18–20GB；若缓存输入派生的 uint16 邻居表，另约15GB，VAL再按比例增加。
两臂共享只读 CPU mmap/页缓存，不复制原数据；主机可用内存和随机读能力本轮未核验。
37维检索键约216MB；TRAIN/VAL的 top4 行号约25MB。精确 TRAIN 检索约9 TFLOP的CPU距离计算，必须分块，不能存整张距离矩阵。

**缓存取舍不能偷换：**
- 可永久缓存：原始输入、确定性几何/邻居索引、标签及 input-only 检索表；不依赖训练参数。
- 本对象不可跨 update 缓存：P 的可训练支持节点、局部 K/V；更新后即陈旧，按 VAL 间隔刷新也不等价。
- FP16支持节点快照按各题 padding 上界约54GB，全有向边32维约3.4TB；冻结 learned cache 并非无代价。
- 冻结双方随机 P 可降低重算，但检验的是固定局部表示上的条件 tail；不等价于完整可学习的局部对应。
- 只冻结支持分支、使用训练过的 donor cache、只给晚期整图均值，均引入额外机制差异，不接受为主对照。

## 7. 有限封套与停止结论

**若后续另行决定投入，只保留这一对照，无 batch/LR 网格、预训练或补丁实验序列：**
seed2、逻辑1024、每臂57,600成功 updates（约40.45次 TRAIN 暴露）；沿用 R84 原 R-Drop 监督、
AdamW(peak1e-4, wd1e-4)、warmup4320、cosine至1e-6，VAL每1440；不另换目标替上下文背书。
同一 query 顺序、支持表、候选池和初始化；两臂各在一张授权3090上训练，前提是主线程释放相应资源。
资源硬封套为训练每臂144h、VAL/固定TEST/LIB及交付预留每臂24h，合计最多336 GPUh；不自动续预算。
这要求平均完整 update <=9秒，约114 query/s/卡及约1.7–9.7有效 TFLOP/s；是必要吞吐要求，不是实测速率。
40轮完整VAL共3.24M query/臂；检查点只按预先固定的 VAL macro_vs_SBS（TRAIN拟合SBS）选择一个。
锁定该 checkpoint 和推理规则后完整 TEST/LIB；LIB 原生尺寸本轮未核验，不能拿TRAIN的500节点上界承诺其显存/时间。
超时、OOM或身份来源无法保证则记为未完成，不缩 epoch、不砍题、不把短跑当机制结论；不另开填充实验。

完整比较仍需 VAL/TEST/LIB 并列，whole18等权 Top1、mean_cost、SBS/Oracle、vs_SBS/vs_Oracle、actual regret；
方法分布 PDF/CSV 沿用 AGENTS 交付规则，不在本文件重复开展 R117 结果分析或新增报告。
即使两臂都改善，也只能先归因于更强局部上下文；早期必须相对强晚期有整体可归因价值，不能靠个别题拼接。
历史留出已参与研发，不宣称盲测；本轮不报告任何 gate、准确率或性能改善。

**最终 NO-GO 理由：** 参数/局部性对齐可以具体化，但目前没有完整 base-family 接口保证，
也没有支撑上述吞吐、I/O及LIB交付封套的实测证据；整图优势能否支持可迁移局部对应仍只是高风险假说。
**首要未解设计阻塞：** source_index 与真实同源族的关系，特别是 ATSP 重编号；不能用现有精确哈希默默替代。
解决这一前提不自动获得科学推荐；若实际实现只能依赖冻结 learned cache 或弱晚期池化省预算，应放弃本次时机比较。

依据仅限 AGENTS、Astra 本次 REPORT/directions，以及上述本地模型/缓存/数据接口；未做外部论文调研或历史结果扫查。
