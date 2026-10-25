# 综合质询记录

日期：2026-10-25。最终阶段：三镜头均已获用户正式完成通知，正文与来源记录已逐份全文阅读；最终综合见[REPORT.md](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/REPORT.md)及[research_directions.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/research_directions.csv)。

**这是实际阅读、质询与最终取舍记录，不是虚构的三方逐字对话。** 第2-5节保留背景问题框架，第7节保留01先完成阶段的分析；最终取舍在第6、8-10节。未收到作者针对综合者问题的回复，不将综合者自己的修正写成三方共识。

## 1. 职责与证据约束

- 仅作科学文献与下一步思路综合；读取实现用于辨认机制，不作工程代码审查、结果认证或执行预算。
- 不启动 agent、训练、solver、label generation；不修改运行代码、状态、协议或其它 worker 的文件。
- 请求模型为 `gpt-6-astra/max`。本地 `agent_trace.json` 记录调用方请求与角色；本会话没有独立认证服务端实际模型身份的接口，不把请求参数冒称为后端认证。
- 历史数字来自已有报告及机械汇总，不是本轮重新计算或独立复现。研究意见保持 same-family provisional。
- 当前 frozen NEW90k 为18题各81,000/4,500/4,500，logical batch1024、seed2；候选删除规则与 native/FP64 winner 不变。旧 TEST/LIB 已参与研发。
- 历史 TRAIN context 可以使用，但排除 query 同 base 实例及其关联副本；VAL/TEST/LIB 标签不进入 context。当前 query 标签、成本与求解结果不进入前向。
- 本任务不恢复暂停的自动实验目标，不重开 R58，不自行增加标签、solver 调用、划分或在线信息预算。

## 2. 背景阶段实际读过什么

| 材料 | 阅读目的 | 证据边界 |
| --- | --- | --- |
| [REQUEST](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/REQUEST.md)、[AGENTS](/public/home/shiys/0selection2/AGENTS.md) | 确定目标、角色与冻结协议 | 当前任务优先于历史运行指令 |
| [第一轮完整综述](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/REPORT.md)及[R35-R79历史稿](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/ASTRA_HISTORY_R35_R79.md) | 恢复旧假说与早期覆盖缺口 | R1-R34仍为partial；不把旧在跑状态当当前事实 |
| [v2完整综述](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/REPORT.md) | 恢复零选择预算、条件先验、纠正/伤害、实际路径及两条旧建议 | 定量分析主要对应R94B等旧锁定模型，不冒充R110逐例结果 |
| [post_R108](/public/home/shiys/0selection2/docs/research/post_r108_learning_mechanisms_20261024.md)、[post_R110](/public/home/shiys/0selection2/docs/research/post_r110_broad_mechanism_20261025.md) | 理解方法私有图更新与族条件特征机制 | 后者文末R110未完成的时态已过期，不沿用 |
| [artifact_index](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/artifact_index.csv)、[reported_all_metrics](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/reported_all_metrics.csv)、[running_snapshot](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/running_snapshot.csv) | 定位原始小报告与完成状态 | 文件存在及复制报告行不等于独立认证 |
| R105-R108固定36组B对A恢复CSV、[R106-R108摘要](/public/home/shiys/0selection2/code/V4/runs/continuation_20261024_results/RESULTS_BRIEF.md)、[R108恢复报告](/public/home/shiys/0selection2/code/V4/runs/R108_conditional_density_ratio/preparation/report_only_recovery/RESULTS.md) | 读取新结果及联合指标取舍 | 摘要中的旧发布阻塞由后续完成记录覆盖；不修复任何管线 |
| [R110整版结果](/public/home/shiys/0selection2/code/V4/runs/R110_method_conditioned_ce/RESULTS.md) | 更新纯CE下方法私有化判断 | 已完成并提交370e0b49；本轮不重新验证提交或预测 |
| [R47](/public/home/shiys/0selection2/code/V4/R47_README.md)、[R63](/public/home/shiys/0selection2/code/V4/R63_README.md)、[R69](/public/home/shiys/0selection2/code/V4/R69_README.md)、[R71](/public/home/shiys/0selection2/code/V4/R71_README.md) | 明确历史机制与控制关系 | 旧数据、继承预训练、不等预算和复用控制等限制保留 |
| [r47_model](/public/home/shiys/0selection2/code/V4/r47_model.py)、[r107_model](/public/home/shiys/0selection2/code/V4/r107_model.py)、[r109_model](/public/home/shiys/0selection2/code/V4/r109_model.py)、[r111_model](/public/home/shiys/0selection2/code/V4/r111_model.py)的相关前向 | 区分可训练检索、关系内容、参数私有和条件监督 | 仅机制阅读，没有运行、测试或代码质量认证 |

未开展额外独立文献检索。收到专题后，仅沿给出的URL定向复核9篇原论文的相关段落，范围见第7、8、9节；不把镜头的全部阅读声明写成主综合者已独立复核全部论文。最终还全文读了R97/R98/R100结果、R109/R110/R111方案，以落实局部函数最近历史与在途决策路径。

## 3. 新事实要求怎样更新旧建议

下表数字是既有报告的TEST ALL Top1百分数；不是跨版本匹配因果比较。

| 旧期待 | 新报告事实 | 需要更新的判断 |
| --- | --- | --- |
| 条件FFN专家可能让互补特征形成 | R105 static/input gate：71.2370/71.3395；固定36组正确18/20 | 输入路由有有限配置差异，但没有形成广泛恢复；普通MoE不能继续以未试机制优先推荐 |
| 解冻native donor可释放选择信号 | R106 frozen/finetune：71.5235/71.1605；regret为0.381866/0.390786；固定36组正确130/93 | 当前适配配方不支持该期待；不能因此否定所有迁移，R94的匹配小收益仍然存在 |
| 明确连接的短路径可能改善关系区分 | R107 factorized/connected：71.2605/71.3074；固定36组正确21/18 | v2关系提案已有直接实施近邻；后续必须提出不同的信息对象，不能只把相连路径重新命名 |
| 先验加密度比可改善条件外证据学习 | R108 prior-CE/ratio：71.3383/71.2173；固定36组正确102/63 | v2备用目标已测；该配置不支持优先继续，不能由略低regret宣称全面胜出 |
| 方法私有节点参数配纯CE可以恢复漏选 | R110 tied/private：71.3074/71.2333；固定36组召回1.5224/0.9569；regret为0.406464/0.399449 | TEST成本取舍与召回/Top1方向不同，VAL又较差；既非突破，也非全部风险全面退化 |
| 复合目标下的方法私有化、族条件监督 | R109/R111在2026-10-25 04:54 UTC快照仍在训练 | 等待完整锁定结果；不把中间成绩写成反证，不另开重复版本 |

R110 TEST的固定36组B对A为救回21、丢失47、新误选41。`21-47=-26`仅表示目标winner正确数的变化；新误选是另一个条件事件，未检查逐例交集前不能直接再减41来声称全体净准确率变化。成本还可能在双方都错的实例上变化。

## 4. 综合问题框架

### 4.1 两张因果图，而不是一个万能解释

零选择图要区分：合法候选及winner定义、局部优势支持、允许输入、表示可辨识性、特征学习与梯度路径、跨候选分数比较、最终argmax、固定组真实恢复与误选。

ALL平台图要区分：条件先验已解释的部分、清单外主要错误、每题竞争难度与样本支持、共同错误及错误交换、学习目标/选型目标/报告指标差异，以及18题等权聚合。不能把零选择组的局部解释外推为平台的全部原因。

每一层均须列出观测、机制假说、最强反方和未识别之处。小TRAIN-VAL gap不是Bayes上限或欠容量证书；旧小数据明显过拟合也不能外推到所有当前模型。

### 4.2 必须阻止的概念偷换

| 容易混同的概念 | 本轮要求 |
| --- | --- |
| 有winner支持与存在可恢复的决策区域 | 局部支持、可预测性、模型已学会及自然分布下净收益分别讨论 |
| 有完整原始图与计算保留全部相关结构 | 写明关系如何进入value/状态、如何被聚合及最终如何参与决策 |
| 方法状态私有、更新参数私有、梯度独立 | R63、R109/R110、R111分别说明，不互相替代 |
| 联合性能建模与冠军后验学习 | R71已做联合成本分布；新建议必须交代何种条件依赖或决策路径真正不同 |
| 目标公式不同与引入新信息 | surrogate重参数化不自动增加信息或改善学习可达性 |
| 原始成本均值与跨题结论 | ALL按问题等权；跨问题raw cost仅描述，不遮蔽逐题风险 |
| 答案感知union与可部署组合 | union只说明给定模型集合在已知答案下的覆盖，不是学习上限、部署性能或收益承诺 |

### 4.3 每个保留方向的必答项

最终最多三个方向，不要求凑满。每个方向必须同时写出数学目标、监督信号、表示对象、最终单方法决策、nearest历史实验、实质差别、成功机制、失败机制，以及哪项现有事实会使建议撤回。论文定理本身与本项目迁移假设分开标记。

## 5. 综合时采用的具体交叉问题

1. **新关系到底是什么？** R107已经保留同一中间节点的两边连接、方向/闭合及输入可计算的局部资源组合，并在两次关系轮中汇入节点流。若继续推荐关系表示，必须明确它能区分而这一具体计算仍会混淆的对象，并说明该对象为什么影响完整候选池的相对优势。更强图表达力定理本身不足以填补此桥梁。
2. **学习路径究竟新增在哪里？** R47已有共享可训练key、选中reference两侧梯度、同题TRAIN行为值和key差修正；R63已有候选节点状态；R69已有完整task-private bank；R71已有联合分布。不能分别用“可训练记忆”“候选专属”“任务独立”“不只学均值”等宽泛标签绕过近邻。
3. **R111提供哪种而非哪种隔离？** 实际前向中，根读取共享post-joint表示和族信息，族私有节点输出不回流到根；非winner族的私有参数没有该实例的条件NLL梯度，但共享编码、pool、score与context并未全部独立。根正确但族内失败、族内改善但根压低、两者改善却跨题代价增加，应分别引向何种后续判断？这只是等待结果的解释分支，不是新增消融矩阵。
4. **何种负面结果足以停止，而非永远换名续训？** R109/R111若仍仅交换错误，应收缩哪一类“共享导致平台”的假说？反之若有实质联合收益，能够支持到哪一层，哪些因果解释仍不能成立？

## 6. 三镜头分歧记录：最终取舍

| 镜头 | 实际主张与原文位置 | 冲突及综合取舍 | 是否推翻旧建议 |
| --- | --- | --- | --- |
| 01 决策与可辨识性 | 全文301行及28条来源记录已读，其中8篇一手论文；3.2节净收益、4.2节pair语义、4.3节R111、7节两方向 | 采纳为共同决策尺子，不单开诊断研究线；有限策略类边界只限制不可学主张。旧表示无见证不能否决新表示 | 将旧Bayes区域讨论推进为可部署策略相对反证；不支持继续以密度比为未试路线 |
| 02 表示与计算 | 全文206行及10篇来源记录已读；4节撤回泛关系，6节保留局部函数 | 采纳撤回；局部函数有条件保留并降至唯一探索机制。修正“不是加权读出”的过强对比，明确无局部正证据与支持缺失限制 | 撤回v2宽泛关系编码优先建议；不恢复泛memory/ICL路线 |
| 03 学习与优化 | 全文455行及13篇来源记录已读；4节有限类、6.4节WD、9节方向与停止条件 | 采纳在途收束及阴性终点；任务顺序/学习时间不升第三方向。proper-loss和WD用于排除错误解释，不另提新loss或减WD | 弱化复合目标/共享/强WD的充分性故事；不否认历史小幅匹配收益 |

三者并不自动构成共同推荐：01的方法论不是性能干预，02的新计算对象没有直接项目支持，03要求减少追加训练。最终只保留“一条在途裁决+一条新增机制”，而不是把各镜头保留项相加。

## 7. 镜头01全文阅读记录

### 7.1 阅读范围与暂时取舍

已读[01正文](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/01_decision_and_identifiability.md)全部301行，以及[01_sources.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/01_sources.csv)全部28条记录：8条一手论文、20条项目材料。以下形成于01先完成阶段，不是镜头02/03的意见；最终排序以第6、10节为准。

| 01原文位置 | 暂时采纳的增量 | 保留限制 |
| --- | --- | --- |
| 2节，三种风险 | 冠军概率、原始成本均值、相对成本均值所需统计对象不同；不必为每种决策拟合完整连续密度 | 这不是断言R71失败来自密度目标；更不是再开成本头的理由 |
| 3.2节，式(1)-(2) | 相对当前策略的正净收益足以反驳该策略在相应区域的最优性，不必先证明真实Bayes获胜区域 | 只能推翻当前策略；第三方法可能更好，未找到见证不能推出不存在 |
| 3.3节及6B | 全向量可靠性与利用完整输入是不同命题；预测残差可帮助定位问题 | 粗分箱内的输入相关残差仍有混杂，见7.3；有限检查不是输入充分性认证 |
| 4.1节 | AUC、precision与当前竞争者对应的动作收益不同；不能规定precision必须大于50% | 一个候选优于错误较多的基线，仍可能多数改选错误；需看有符号净变化与成本 |
| 4.2节，式(3) | 互斥冠军条件概率的pairwise coupling，不能套给R100的两成本胜率 | 已核对其三方法构造：两个分布所有pair胜率均为1/2，冠军向量分别为(5,3,4)/12与(3,5,4)/12；这是数学反例，不是项目分布 |
| 4.3节，式(4) | 固定根和其它族输出时，真实族总质量低于族外最大叶子概率，任何族内单独修复都无法救回 | 根family Top1不等于最终错误；这个诊断使用答案，不能变成可部署路由或效果上限 |
| 5节 | 同数据、同seed、共享代码和研发历史使多项结果相关；不能把版本数当独立反证次数 | 相关也不抹掉各项配置对“足够解决问题”这一具体猜想的负证据 |
| 6节及7节两个方向 | 决策收益见证、受限策略类遗漏价值能收紧研究问题 | 两张表不是必须新增的研究gate；受限价值边界不是性能干预，方向一尚需具体输入函数类和学习路径 |

### 7.2 将条件矩真正接到动作，而非只接到校准

01已经明确其式(5)非零不保证收益。为避免最终报告只写一句免责声明，保留以下直接恒等式作为更具体的解释。设旧动作由概率向量的固定argmax给出，区域成员资格为输入可计算的指示函数 `h(X)`：

```text
d0(X) = argmax_m q_m(X)
W_k(X,Y) = 1[Y=k] - 1[Y=d0(X)]
m_k(X) = q(X)[k] - q(X)[d0(X)] <= 0
residual_k = W_k - m_k

Top1_gain(h,k) = E[h * W_k]
              = E[h * residual_k] + E[h * m_k]

Positive_gain requires:
E[h * residual_k] > -E[h * m_k]
```

这里方括号表示在同一实例上取相应方法的概率分量。正的残差相关可能不足以弥补旧动作的预测优势。因此不能从“发现multicalibration违背”直接跳到“该区域应改选”；若旧分数不具有概率解释，不能使用这套后验措辞。以上是本轮阅读后的代数说明，不是新数据分析。

01式(1)-(2)同时加强了第3节对误选计数的提醒：原正确被伤害才直接减少Top1；新增false pick还可能来自双方都错的替换，此时Top1不变而成本可升可降。固定36组的恢复统计和全体动作变化不能不经集合区分就相加减。

### 7.3 “校准错误还是遗漏输入”还需保留一层识别限制

对01的6B建议暂不作因果归类。减去每个实例的预测差是必要修正，但在粗预测箱内仍不足以排除细粒度校准误差。假定 `d0` 是完整 `q` 的固定函数，令 `r_k(q)=E[W_k given q]`，则有：

```text
W_k - m_k(q) = (W_k - r_k(q)) + (r_k(q) - m_k(q))

E[h(X) * (W_k - m_k(q))]
  = E[h(X) * (W_k - r_k(q))]
  + E[h(X) * (r_k(q) - m_k(q))]
```

第二项纯粹来自给定完整预测状态时的概率差失配；若输入字段与箱内 `q` 的细微变化相关，它也能造成按输入切分后的残差差异。只有额外控制这一项，或具备足够可靠的完整预测状态条件结论，才可更有针对性地解释第一项为未利用的输入关联。若旧动作还依赖其它状态，条件变量需包含该状态，而不是只看 `q`。

01已承认有限摘要、有限样本不能证明充分性；这里将其残余歧义写得更明确，并不说01已经声称完成了因果识别。本轮未拟合 `r_k`、运行图表、增加校准网格或新划分。最终用实际动作W/G连接02，而不要求先完成这项条件识别。

### 7.4 定向原文复核

| 论文与一手URL | 主综合者本次实读位置 | 独立复核结论与限界 |
| --- | --- | --- |
| [Vaicenavicius等，Evaluating model calibration in classification](https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf) | 2节定义；3节Proposition 1及3.1节canonical calibration function | 粗化输入的真实条件概率可以可靠；常数先验是极端例子。支持可靠性不等于完整输入充分性，不为本项目估计任何后验 |
| [Wu-Lin-Weng，Probability Estimates for Multi-class Classification by Pairwise Coupling](https://jmlr.org/papers/volume5/wu04a/wu04a.pdf) | 1节输入对象；2.1-2.2节及式(2) | 论文耦合的是互斥类别条件概率，支持01对R100迁移对象的限制；没有复核其全部数值实验 |
| [Hebert-Johnson等，Multicalibration](https://proceedings.mlr.press/v80/hebert-johnson18a/hebert-johnson18a.pdf) | 3节adaptive generalization、Algorithm 1、Theorem 2；4节weak agnostic learner定义与Theorem 3 | 原文要求可计算子群、支持与受控自适应估计，确有弱学习联系；多类成本决策及节点关系学习是项目延伸，不是该论文现成保证 |

此阶段其余5篇保留01的具体阅读声明，不标记为主综合者独立全文复核；其中SPO的相关方法与条件在阅读03时另行定向核实。未改动01正文或其来源CSV。

### 7.5 当时的等待事项及最终处理

两条01方向当时分别归入“连接表示与净动作收益的研究方法”和“限定不可学结论的理论问题”。三份齐全后，前者成为REPORT第6节并直接连接局部函数策略，后者不占新方向名额。没有把诊断变成先于一切训练的无限gate。

## 8. 镜头03：采纳边界，拒绝再造训练技巧队列

完整阅读[03正文](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/03_learning_and_optimization.md)与[03来源表](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/03_sources.csv)。实质取舍如下。

| 03原文主张 | 综合者判断 | 最终落点 |
| --- | --- | --- |
| 4.1节：有限假设类中proper-loss最优不保证最佳Top1 | 采纳，保留其两个输入的数学构造；不据此诊断当前项目或另荐loss | REPORT第5.1节，限制“CE正确所以已到上限”的推论 |
| 5节：特征形成与读出利用不同，DFR是竞争解释 | 采纳，不把TRAIN漏选确诊为starvation或NTK-lazy；不引入VAL标签重训 | REPORT第3、5节 |
| 6.4节：当前WD直接乘法收缩至多0.0576% | 已读R110方案核对协议数值；采纳量级推导，只排除大幅直接收缩，不证明WD=0等价 | REPORT第4.3节 |
| 9.1/9.3节：等待R109/R111，阴性要真正降级 | 采纳为优先事项一，不启动重复版本 | REPORT第7、9节与CSV D1 |
| 9.2节：有条件研究任务顺序/可迁移优势形成 | 保留理论问题但不进入前三；目前无优势被其它任务更新抹掉的定向证据 | REPORT第7.2、10节，不成为共享私有化阴性的默认后继 |

定向复核范围：[H-consistency原文](https://proceedings.neurips.cc/paper_files/paper/2022/file/051f3997af1dd65da8e14397b6a72f8e-Paper-Conference.pdf)第2-3节的minimizability gap与定理条件；[DFR](https://arxiv.org/html/2204.02937)第4.1-4.2、5、6节的隐藏特征及实际重训数据制度；[L2动力学](https://papers.nips.cc/paper_files/paper/2020/file/32fcc8cfe1fa4c77b5c58dafd36d1a98-Paper.pdf)训练时钟、日程限制与第4节理论假设；[SPO v5 PDF](https://arxiv.org/pdf/1710.08005v5)定义、假设与一致性定理，并对照[HTML](https://arxiv.org/html/1710.08005v5)。

SPO的PDF使用Definition 3、Assumption 1、Theorem 1，HTML使用Definition 3.5、Theorem 4.2等编号；这是渲染版本编号差异，不当作两镜头引用冲突。主稿不依赖虚构统一页码。没有复核这些论文全部实验或附录。

## 9. 镜头02：保留一个对象，同时收窄其新意

完整阅读[02正文](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/02_representation_and_computation.md)与[02来源表](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/02_sources.csv)，并重读R47实际聚合及R97/R98/R100完整结果。

### 9.1 采用与修正

**采纳第4节的撤回。** R107已经保留连通中间节点、方向/闭合和资源组合；未给出完整前向仍混淆且影响候选相对优势的具体对象，泛高阶关系不应继续优先。

**有条件采纳第6节的局部函数对象。** `Phi.T W Y`与联合几何求解，区别于R47固定query/key条件下分离的标签value与位置修正聚合；等距记录交换例子只见证该受限计算差别，不证明真实数据存在对应规律，不证明R47全部可学习函数类较弱。

**收窄“拟合函数而不是加权读出”。** 综合者将其展开为` s = a_q.T Y_support `。仍是标签线性加权，区别在整个support几何决定系数且可为负，而不是脱离记忆读出或创造额外标签信息。REPORT第8.2节明确保留此修正。

**补足弱信号与支持限制。** 现有报告没有直接给出“同邻域局部斜率优于常数”的正证据。REPORT第8.3节的误差分解只用于解释偏差/方差，不是对学习后表示的项目泛化界；纯one-hot ridge对support中完全无winner的方法给零列，且总分为正，所以不能选择该方法。这是定义的必要限制，不是测得项目覆盖不足。

**明确学习目标而不再加方向。** 02给出局部ridge与TRAIN query/support episode，主综合进一步固定自然频率one-hot平方外层目标，使方法有完整数学定义；不声称胜过CE，不追加温度。常数/仿射比较共用外层目标，以免将目标差别解释为几何机制。此选择属于综合者，不伪称作者共同选择。

**计算成本写全。** 主稿计入query编码、bank检索、局部矩阵及多右端求解、bank存储和训练support编码；不把小全局参数量等同低时延，不给未经测量的秒数或GPU计划。

### 9.2 实际提出的质询与回复状态

三个专题完成后，综合者已通过用户可见commentary请求转02两问：

1. 是否同意局部ridge仍是`a_q.T Y_support`，真正差别是联合support几何，而不是“摆脱加权记忆”？
2. 除构造反例外，现有项目材料是否已有局部函数优于同邻域常数预测的直接正证据？若没有，是否应按尚无直接支持的有限机制检验表述？

本会话工具列表没有可用的`send_input`接口，因此没有虚构一次工具发送或作者回复。到用户要求收束时，尚未收到这两问的作者回应。最终科学判断由综合者依据已读文本与代数推导承担，不将其写成三方共识；这不是等待定稿的阻塞条件。

为核实最近文献差异，定向阅读[TabR第3.1-3.2节，式1-5](https://arxiv.org/html/2307.14338v2)与[DNNR第2-3节、算法1](https://proceedings.mlr.press/v162/nader22a/nader22a.pdf)。前者确实借鉴DNNR并使用标签value加key差修正；后者明确区别每query局部平面和其逐邻居导数/Taylor方案。本提案不冒称原样DNNR，也不继承其回归实验或完整误差界保证。未新增独立文献检索。

## 10. 最终排序、旧建议撤回与交付边界

| 最终项目 | 为什么保留或不保留 | 与其它镜头的衔接 |
| --- | --- | --- |
| D1，优先收束在途R109/R111 | 已有任务正好检验当前私有计算/条件监督主张；无需新版本 | 用01的净动作收益判断，用03的阴性停止条件收束，不以02的局部函数替它们提前判负 |
| D2，唯一新增局部函数机制 | 有明确中间对象和最近历史差别，但无直接项目正证据，只作条件性有限研究 | 02提供对象，综合者补足目标/等效权重/支持限制，01的W/G直接评价完整策略 |
| 01的单独诊断与受限不可学边界 | 不占新学习方向名额 | 保留判断工具，不要求所有旧分数诊断通过才允许检验D2 |
| 03的时间/顺序问题 | 当前定向证据不足，不升第三方向 | 若R109/R111阴性，不据此立即再开优化技巧路线 |

**明确推翻或降级的旧建议：**撤回v2泛关系计算的优先地位；R108之后不再把条件密度比列为未试备选；R105/R106/R110之后降低普通MoE、解冻donor、私有节点加纯CE的充分性期待。保留R94、R65、R67、R89、R101的小幅历史正结果，不将历史统一写成失败。

**不推翻的边界：**冻结18题、81k/4.5k/4.5k、logical1024、seed2、原生winner与候选删除；历史TRAIN可用但排除query/base关联，不用VAL/TEST/LIB标签作context；不恢复R58、不加标签或在线solver。TEST已参与研发，union和完美修复只作答案感知统计，不作可部署上限。

REPORT第9节列出完整结果的后续分支。R109/R111若仍只交换错误，停止以残余共享为由续命；D2若不胜常数或无完整策略价值，也不自动扩成高阶/更大邻域/新ICL/gate清单。一个小成本收益可以如实保留，但不能替代自然Top1与真实恢复主张。

三专题有31条一手论文阅读记录、标题去重30篇，SPO重复。综合者全文读完三专题与来源记录，定向复核9篇相关原文段落，没有逐页读完30篇的声明。本轮只创建/修改本人负责的REPORT、SYNTHESIS_DIALOGUE和research_directions.csv，不改专题、来源表、运行代码或状态，不启动agent、训练、solver或标签任务。最终科学主张是同族LLM provisional；Sol后续机械来源检查与PDF排版不等于独立复现实证。
