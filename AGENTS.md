# AGENTS.md

本文件是 `/public/home/zhoucl/shiys` 整个工作区的全局长期记忆。它的目标不是完整记录所有历史，而是帮助后续助手快速判断：

- 现在真正的研究主线是什么
- 哪些旧方向已经基本完成或暂时废弃
- `shiys/` 下各目录分别是什么
- 下一步建模时应优先参考哪些论文、代码和设计原则

除非某个更深层目录下有自己的 `AGENTS.md`，否则本文件对整个 `shiys/` 生效。

## 额外说明

- 本文件里关于“技术路线”的内容，当前只应理解为：
  - 基于现有阅读形成的**候选方向**
  - 后续深度调研与实验前的**工作假设**
- 这些内容**不是最终定案**。
- 后续如果新的论文阅读、实验结果或代码验证推翻了当前判断，应优先更新本文件，而不是被旧记忆反向约束。

## 最近一次 idea-discovery 结论

- 时间：`2026-04-14`
- 产物：
  - `/public/home/zhoucl/shiys/IDEA_REPORT.md`
  - `/public/home/zhoucl/shiys/refine-logs/FINAL_PROPOSAL.md`
  - `/public/home/zhoucl/shiys/refine-logs/EXPERIMENT_PLAN.md`
  - `/public/home/zhoucl/shiys/refine-logs/PIPELINE_SUMMARY.md`
- 当前最值得推进的 idea 不是简单的“一个模型支持多个问题”，而是：
  - **compositional problem-solver compatibility selector**
  - 也就是把问题实例、问题属性、求解方法能力都作为组合式对象来建模，再学统一的 `score(instance, solver)`
- 但这一点目前仍只应视为：
  - **当前最强候选主线**
  - 不是已经锁死的最终方案
- 当前最直接的下一步不是先写最终模型，而是：
  - 先用 `EasyNCO` 建立问题-方法覆盖表
  - 先跑小规模标签数据 smoke test
  - 先做 pooled baseline / per-problem baseline

## 0. 当前最重要的结论

### 0.1 研究方向已经切换

当前主线**不再**是：

- 强化学习
- contextual bandit / LinUCB / Neural-LinUCB
- “先做一个 per-problem selector，再扩展”的思路

当前主线**已经切换为**：

- 用**监督学习**做一个**统一的 selector / ranker**
- 让**一个模型**同时支持多种 routing problem
- 先覆盖：
  - `tsp`
  - `atsp`
  - `pctsp`
  - 用户当前指定的 `16` 种 `cvrp` 变体
- 核心目标是：
  - 不像 NSS 那样每个问题训练一个单独模型
  - 而是训练**一个跨问题共享的模型**

### 0.2 与 NSS 的关键区别

`NSS` 的核心思想是：

- 发现不同 solver 在实例级别有互补性
- 训练 selector 为每个实例选 solver
- 但本质上仍是**按问题分别建模 / 分别训练**

你接下来要做的方向是：

- 保留“实例级方法选择”这个问题意识
- 但改成：
  - **一个统一模型**
  - **多个问题共同训练**
  - **多个问题共享表示与参数**
  - 在统一框架内处理不同问题的输入和输出差异

### 0.3 现在最关键的三个科学问题

当前工作最核心的问题是：

1. **不同问题可选方法数不同，输出空间不一致**
   - 例如某些方法只支持 TSP，某些只支持 CVRP，某些支持 prize-collecting，某些不支持 asymmetry。
   - 这会带来输出不均衡和监督信号不均衡。

2. **模型如何知道一个实例属于什么问题**
   - 是靠显式 problem id？
   - 还是靠数据本身推断？
   - 还是两者都用？

3. **如何统一表示不同问题**
   - `tsp / atsp / pctsp / cvrp variants` 的节点属性、约束、解语义都不一样。
   - 需要一个统一输入表示，让一个模型可以真正共享底座。

## 1. 当前候选的总体建模方向

### 1.1 不要从 RL / bandit 再出发

旧方向已经基本达到阶段目标，可以作为背景、失败经验和论文素材保留，但**不要再默认往 RL / bandit 继续扩展**。

当前更值得优先调研的出发点是：

- **监督学习**
- **跨问题统一建模**
- **统一 scorer / ranker**

### 1.2 输出层的一个强候选方向：统一方法空间 + feasible mask

当前较值得优先考虑的思路不是：

- 为每个问题单独做一个 classifier head

而是：

- 建一个**全局方法池** `M`
- 每个方法对应一个 `method embedding`
- 每个实例额外带一个 `feasible mask`
  - 标出该问题上哪些方法可用，哪些不可用
- 模型学习一个统一评分函数：
  - `score(instance, method)`

这会把“不同问题方法数不一致”的问题，从“头大小不同”转化为：

- 同一个 scorer
- 不同的候选集合

这和推荐系统里“用户-候选 item 打分”比“每个场景单独 softmax 头”更自然，逻辑上非常接近。

### 1.3 输入层的一个强候选方向：统一数据表示，不靠纯 problem tag

当前较值得优先试验的输入设计是：

- 做一个类似 URS 的 **Unified Data Representation**
- 为不同问题定义统一的节点特征槽位
- 不存在的属性直接补零

当前可优先考虑的统一输入模板至少包含：

- 位置相关：
  - `x`
  - `y`
  - `eta` 或其它 asymmetric identifier
- 节点属性：
  - `demand`
  - `prize`
  - `penalty`
  - `service_time`
  - `tw_start`
  - `tw_end`
- 节点角色：
  - depot flag
  - pickup flag
  - delivery flag
  - open-route related flag
  - multi-route related flag

对当前阶段的 `tsp / atsp / pctsp / cvrp variants` 而言：

- `TSP`：大多数学槽位补零
- `ATSP`：重点用 `eta` / asymmetric distance representation
- `PCTSP`：需要 prize / penalty
- `CVRP` 变体：重点用 demand + 约束相关属性

### 1.4 关于“问题类型识别”的当前假设

当前更稳妥、但仍待验证的原则是：

- **不要强迫模型只靠原始数据自己猜问题类型**
- 最好同时提供：
  - 显式的 coarse problem family 信息
  - 隐式的 active-feature / multi-hot problem representation

也就是说，当前可优先尝试同时使用：

- `problem family embedding`
  - 例如 `tsp / atsp / pctsp / cvrp`
- `active-attribute multi-hot`
  - 当前实例启用了哪些特征槽位 / 约束属性

这比只给一个离散 problem id 更灵活，也比完全不告诉模型更稳定。

### 1.5 损失函数的当前优先调研顺序

当前可先尝试的优先级：

1. `masked listwise / ranking loss`
2. `pairwise ranking loss`
3. `masked cross-entropy`

原因：

- 不同问题的可行方法数不同
- 只学最优 one-hot label 会浪费次优方法的信息
- ranking 更适合“方法选择 / 方法排序”本质
- 这也是 NSS 里已经验证过的重要经验

### 1.6 如果共享模型出现负迁移，再考虑 shared/private experts

当前较自然的探索顺序是：

1. 先做一个**简单统一 baseline**
   - 统一输入表示
   - 统一实例编码器
   - 问题表示条件化
   - 全局方法池 + feasible mask
   - ranking loss

2. 如果发现明显负迁移，再引入更复杂结构：
   - shared-bottom + problem-specific adapters
   - MMoE / PLE 风格 shared/private experts
   - STAR 风格 shared center + domain private residual
   - CoEKS 风格按约束组合激活 experts

当前判断是**不要一上来就做太重的专家系统**，但这也不是硬限制；若后续调研表明专家结构是更好的第一步，可以调整。

## 2. 当前可优先调研的技术路线候选

### 2.1 一个最自然的统一 selector 形式

当前一个很自然的写法是：

- 输入：实例 `x`
- 候选：可行方法集合 `A(x)`
- 输出：对每个 `a in A(x)` 的分数 `s(x, a)`

而不是：

- 输入 `x`
- 直接输出固定长度分类结果

一个可行的最小实现候选是：

1. `instance encoder`
   - 编码统一后的图实例
2. `problem representation encoder`
   - 编码 problem family + active-attribute multi-hot
3. `method encoder`
   - 为每个方法学习 embedding
   - 可选地加入方法元信息
4. `pair scorer`
   - 对 `(instance, method)` 做兼容性打分
5. `feasible mask`
   - 只对当前问题可用的方法计算损失和排序

### 2.2 方法 embedding 的当前候选想法

当前较值得尝试的是：每个方法除了 learnable id embedding 外，再加入**方法元信息**：

- 支持哪些问题族
- 是否支持 asymmetry
- 是否支持 prize collecting
- 是否支持需求 / capacity
- 是否支持 open route / time window / backhaul
- 方法类别
  - construction / diffusion / autoregressive / heuristic-improvement
- 推理成本或运行时间先验

这和推荐系统里 item feature 的思路类似，也和 NSS 论文最后讨论“solver feature”是一致的。

### 2.3 输出不均衡的当前处理候选

对于“不同问题方法数量不同”的问题，当前优先可试的做法是：

- **masked normalization**
  - 只在可行方法上做 softmax / listwise ranking
- **per-instance loss normalization**
  - 不让候选多的问题天然贡献更大的 loss
- **balanced sampler**
  - batch 内控制不同 problem family 的比例
- **per-problem / per-method reweighting**
  - 避免热门问题、热门方法压制其它部分

如果以后问题种类继续扩张，再继续考虑：

- curriculum
- group DRO
- uncertainty weighting
- counterfactual augmentation / domain augmentation

### 2.4 “问题识别”的当前候选方案

对“如何区分实例属于什么问题”，当前较好的候选不是二选一，而是：

- 既给显式 problem token
- 又给 active-feature multi-hot
- 再让统一 encoder 读统一实例

也就是说，问题区分信息应同时从三处进入：

1. 原始统一实例表示
2. 显式问题族 embedding
3. 特征槽位激活 multi-hot

### 2.5 “统一表示”的当前候选方案

当前统一表示可优先遵循以下原则：

- **槽位固定**
  - 不要为每种问题改网络输入维度
- **缺失补零**
  - 允许新问题通过新增属性平滑扩展
- **约束尽量由 mask 或元信息处理**
  - 不强行把所有规则都塞进 decoder 输入
- **保留 problem-conditioned modulation**
  - 例如 bias modulation、adapter、FiLM、hypernetwork 小模块

## 3. 论文与代码调研后的关键启发

### 3.1 NSS 对当前方向的启发

来源：

- 本地笔记：`/public/home/zhoucl/shiys/已有文献/nss.md`
- 代码：`/public/home/zhoucl/shiys/9nss论文/neural-solver-selection`

核心启发：

- 实例级方法选择是有意义的，因为 solver 之间确实互补
- ranking 通常比只做 top-1 分类更稳
- hierarchical instance encoder 对泛化有帮助
- 但 NSS 的关键限制也很明确：
  - 本质上仍是 per-problem selector
  - 不是一个真正的 cross-problem unified model

对你现在的新方向而言：

- **保留 NSS 的“实例级选择 + ranking + encoder”**
- **放弃 NSS 的“每个问题单独训练一个模型”**

### 3.2 URS 对当前方向的启发

来源：

- 本地论文：`/public/home/zhoucl/shiys/已有文献/urs.md`
- 代码：`/public/home/zhoucl/shiys/a3_Revised_URS_FinalRefine_UnifiedEnv`

最值得借鉴的点：

1. **Unified Data Representation**
   - 不再显式枚举所有 problem tag 才能建模
   - 而是定义统一的数据槽位

2. **active-feature / multi-hot problem representation**
   - 问题的区别不仅是一个离散名字
   - 还可以是“哪些属性槽位被激活”

3. **problem-conditioned modulation**
   - 代码里通过 `problem_representation` 进入：
     - adaptation bias
     - hypernetwork 生成 decoder 参数
   - 说明共享底座之外，可以用很轻量的方式做条件化

4. **约束与可行性更多交给 mask**
   - 这点对跨问题统一很重要

对你当前工作而言，URS 最重要的不是它的 RL，而是：

- **统一表示**
- **问题条件信息**
- **共享底座 + 轻量条件化**

### 3.3 CoEKS / 本地 `moe.md` 对当前方向的启发

来源：

- 本地论文：`/public/home/zhoucl/shiys/已有文献/moe.md`

这篇论文最值得借鉴的不是“MoE”三个字本身，而是它的结构性观点：

- 不同任务不是离散孤岛，而是由**若干基础约束组合**构成
- 因此共享模型不一定只能是一个完全 dense 的 shared trunk
- 也可以是：
  - 部分共享
  - 部分约束/任务特化
  - 再通过组合机制融合

对你当前工作而言，它提供的不是第一步 baseline，而是：

- 当统一模型出现明显负迁移时
- 可以考虑把差异建模为：
  - problem-family experts
  - constraint experts
  - shared + private experts

### 3.4 推荐系统文献给当前方向的启发

推荐系统里，“一个模型服务多个域 / 多个场景 / 多种候选集合”的问题和你现在的问题非常接近。

当前最值得记住的几类思路：

1. **MMoE**
   - 论文：Ma et al., KDD 2018
   - 链接：
     - https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/
   - 启发：
     - 共享专家 + 任务独立 gate
     - 适合处理多任务间相关性不同的问题

2. **PLE**
   - 论文：Tang et al., RecSys 2020
   - 链接：
     - https://doi.org/10.1145/3383313.3412236
   - 启发：
     - 通过 progressively separated shared/private experts 缓解负迁移
     - 很适合“一个底座共享，但任务不完全同质”的场景

3. **STAR**
   - 论文：Sheng et al., CIKM 2021
   - 链接：
     - https://arxiv.org/abs/2101.11427
   - 启发：
     - 一个模型服务多个 domain
     - 用 shared center + domain-specific parameters 做适配
     - 很适合作为“统一模型 + 轻量问题特化”的参考

4. **YouTube DNN / large corpus item recommendation**
   - 论文：Covington et al., RecSys 2016
   - 链接：
     - https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
   - 启发：
     - 推荐系统通常不是“每个用户一个分类头”
     - 而是“用户表示 × item 表示”的匹配打分
     - 对你这里最重要的启发是：
       - **把方法当 item**
       - **把实例当 query**
       - 做统一 pair scoring

5. **Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations**
   - 论文：Yi et al., RecSys 2019
   - 链接：
     - https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
   - 启发：
     - 当候选空间很大或采样不均衡时，要注意训练分布偏差
     - 对你当前问题的启发是：
       - 如果全局方法池扩大、不同问题覆盖差异加大，要警惕训练采样偏差

6. **AREAD**
   - 论文：Adaptive REcommendation for All Domains with Counterfactual Augmentation, arXiv 2024
   - 链接：
     - https://arxiv.org/abs/2411.05489
   - 启发：
     - 面向很多 domain 的统一推荐
     - 关注 minor domain 与 domain imbalance
     - 如果未来问题种类继续扩展，这类工作会更 relevant

对当前阶段最重要的推荐系统结论可以浓缩成一句话：

- **不要把“不同问题候选方法数不同”理解成必须做多个输出头；更自然的做法是统一候选空间、做 query-item 打分、再加 mask。**

## 4. 当前建议的调研与实验优先级

### 4.1 第一阶段 baseline

第一阶段较值得先做：

- 一个统一实例编码器
- 一个问题表示编码器
- 一个方法 embedding 表
- 一个共享 scorer `s(x, a)`
- 一个 feasible mask
- 一个 masked ranking loss

这个 baseline 跑通以后，先回答：

- 单模型是否能在当前问题集合上稳定训练
- 相比 per-problem NSS 风格 baselines 是否有竞争力
- 不同问题之间是否出现明显负迁移

### 4.2 第二阶段：如果需要，再解决负迁移

如果第一阶段已经证明“统一模型可行，但有负迁移”，再考虑：

- shared trunk + problem-specific adapter
- MMoE / PLE 风格专家结构
- STAR 风格 domain residual
- CoEKS 风格 constraint-composition experts

### 4.3 当前不建议优先做的事情

当前不建议优先投入时间到：

- bandit / LinUCB / Neural-LinUCB 继续扩展
- 完整 RL 重新开线
- 每个问题继续各训一个 selector
- 过早追求极复杂的 expert 组合系统

## 5. 旧方向目前的定位

### 5.1 已经完成阶段目标、但不再是当前主线

以下内容仍有价值，但**现在主要作为背景材料**：

- `old_try/2cmab2/`
  - 旧的 contextual bandit / Neural-LinUCB 主线
- `old_try/2l2r/`
  - 并行监督排序分支
- `old_try/sustechthesis-1.3.9/`
  - 旧 thesis / slides 工程
- `old_try/1two_gate/`
  - 双 gate RL 原型
- `old_try/1step/`
  - 每步 operator 选择
- `old_try/1online/`
  - fully-online RL 原型

### 5.2 这些旧目录现在还能提供什么

- `2cmab2`
  - 提供 selector 数据协议、实例编码、ranking reward / discussion 的经验
- `2l2r`
  - 提供 supervised ranking 的实现经验
- `sustechthesis`
  - 提供已写成文的动机、实验和旧方向总结
- `1two_gate / 1step / 1online`
  - 提供为什么不继续做 RL 的证据和答辩素材

## 6. `shiys/` 全局目录地图

### 6.1 隐藏目录

- `.agents/`
  - 本地 agent skills
- `.bin/`
  - 个人脚本与启动器
- `.claude/`
  - Claude 本地状态、会话、日志、插件
- `.codex/`
  - Codex 本地状态、skills、memory、sessions
- `.cache/`
  - 缓存
- `.config/`
  - 配置
- `.local/`
  - 本地工具数据
- `.npm/`
  - npm 缓存
- `.vscode/`
  - 当前工作区 VSCode 配置

### 6.2 主要研究与代码目录

- `EasyNCO/`
  - 用户自己的统一 NCO 平台
  - 多问题、多方法训练 / 评测 / benchmark / exact solver 支撑

- `9nss论文/`
  - NSS 论文与代码相关目录
  - 当前做统一 selector 时，仍是重要参考

- `9Reld/`
  - ReLD 相关目录

- `a3_Revised_URS_FinalRefine_UnifiedEnv/`
  - URS 对应代码目录
  - 当前最重要的“统一表示 / 多问题单模型”参考实现

- `methods源代码和论文/`
  - 第三方 routing 方法源码与论文资料库
  - 后续构建全局方法池、方法元信息时会很重要

- `old_try/`
  - 旧主线和大量历史研究目录
  - 现在主要作为背景、代码参考和历史材料

- `已有文献/`
  - 当前最关键的文献笔记目录
  - 重点文件：
    - `nss.md`
    - `urs.md`
    - `moe.md`

- `plan/`
  - 使用说明、流程性文档

- `temp/`
  - 临时目录

### 6.3 独立工具 / 独立 git 子项目

- `Auto-claude-code-research-in-sleep/`
  - 独立 git 项目
- `codex-oauth-automation-extension-4.0.0/`
  - 独立 git 项目，浏览器扩展
- `old_try/Claudix/`
  - 独立 git 项目
- `old_try/9Auto-claude-code-research-in-sleep/`
  - 独立 git 项目

### 6.4 根目录重要文件

- `FULL_PROJECT_CONTEXT_FOR_NEXT_MODEL_2026-04-13.md`
  - 上一阶段旧主线的完整交接文档
  - 现在仍然有背景价值，但不是新方向的主入口
- `初始化数据-实验设计.md`
  - 相关实验设计笔记
- `平台方法统计统计.md`
  - solver 互补性 / motivation 相关材料
- `skills-lock.json`
  - skill 锁文件

## 7. 当前推荐阅读顺序

### 7.1 如果目标是新主线建模

1. 本文件
2. `已有文献/nss.md`
3. `已有文献/urs.md`
4. `已有文献/moe.md`
5. `a3_Revised_URS_FinalRefine_UnifiedEnv/Model.py`
6. `a3_Revised_URS_FinalRefine_UnifiedEnv/UNIEnv.py`
7. `a3_Revised_URS_FinalRefine_UnifiedEnv/multi_hot_set.py`
8. `9nss论文/neural-solver-selection/`
9. `old_try/2l2r/`
10. `methods源代码和论文/`

### 7.2 如果目标是继续看旧主线背景

1. `FULL_PROJECT_CONTEXT_FOR_NEXT_MODEL_2026-04-13.md`
2. `old_try/2cmab2/`
3. `old_try/sustechthesis-1.3.9/`

## 8. 运行与协作约定

- 当前是在服务器上直接工作，可以直接跑代码。
- Python / PyTorch 相关命令前默认：
  - `conda activate easynco_zhoucl`
- 长任务优先放 `tmux`
- 机器有直接 GPU：
  - `1x RTX 3090 24GB`

### 8.1 根级 git

- `/public/home/zhoucl/shiys` 已初始化为根级 git 总控仓库。
- 但内部仍有若干独立 git 子仓库。
- 不要把这些子仓库的历史和根仓库混为一谈。

## 9. 给后续助手的硬性提醒

- **不要默认把当前主线理解成 `2cmab2 + bandit`。**
- **不要默认继续往 RL / LinUCB 扩。**
- **当前最重要的是统一监督学习模型，不是旧 thesis 收尾。**
- **优先思考统一表示、统一 scorer、feasible mask、problem conditioning。**
- **如果需要借鉴旧工作，先借鉴 `NSS` 和 `URS`，再考虑更复杂的 MoE / 推荐系统结构。**
