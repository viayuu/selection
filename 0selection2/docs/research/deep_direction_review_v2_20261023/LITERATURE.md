# 第二轮论文阅读：能借鉴什么，不能借鉴什么

2026-10-24。本文件只记录一手论文与项目研究映射；项目效果须以相邻实测证据为准。没有启动新训练。

## 1. 不能把“有 winner 但从不选择”直接认定为分类器损坏

多类0-1决策取条件后验最大者，不要求每个具有正边际概率的类都有非空决策区域。例如在当前表示能区分的所有区域，三类后验均为(0.45,0.35,0.20)，第二类占35%的真实标签，但最佳0-1决策仍不选择它。这个例子不是对本数据的估计；完整输入和固定部署若决定唯一结果，则完整输入上的标签可以是确定的。问题可能出在可学习表示、有限样本或未记录运行条件，不能把71%视为Bayes上限。

[Tewari & Bartlett, On the Consistency of Multiclass Classification Methods, JMLR 2007](https://www.jmlr.org/papers/volume8/tewari07a/tewari07a.pdf)研究代理风险与分类风险的一致性，前提不是任意有限网络已经达到代理总体最优。该理论不能认证现有零选择合理，也不能支持按winner频率分配选择名额。

## 2. 简单特征竞争：重要假说，但不是已经查实的诊断

[Pezeshki et al., Gradient Starvation, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf)，§3及式17：在线性化网络、二分类和特征耦合设定下分析特征竞争；以输出平方惩罚替代参数L2的谱解耦是其具体干预。论文§2也指出，在训练和测试共享强特征时，这种偏向可能具有正则化作用。

项目映射：CE输出误差不小，不保证经Jacobian形成有效的新特征；但不应把“有大量漏选”直接称为梯度饥饿。R93纯CE、R95曝光和R89优化器已经改变若干学习条件。下一步若再研究此机制，应说明哪一类区分被抑制、如何让它形成，而不是仅再加正则项。

[Tiwari & Shenoy, Overcoming Simplicity Bias in Deep Networks using a Feature Sieve, ICML 2023](https://proceedings.mlr.press/v202/tiwari23a/tiwari23a.pdf)，§3.2、Algorithm 1：辅助网络从中间表示识别简单信号，不向主干反传；随后固定辅助网络，以均匀目标对前部主干施加遗忘，并与正常训练交替。论文实验主要为有偏图像和OOD识别。

项目边界：真实规模、容量和约束是有用且合法的预测条件，并非理应删除的伪相关。直接让网络忘记这些条件可能损害已经取得的收益。若借鉴，应保留明确的先验通路，只检验剩余实例信息的学习，不能照搬均匀化或以若干新超参搜索代替研究。

## 3. 更细标签可能改变特征，但 runner-up 标签不是免费的新信息机制

[Li et al., Feature Averaging: An Implicit Bias of Gradient Descent Leading to Non-Robustness in Neural Networks, ICLR 2025](https://arxiv.org/pdf/2410.10322)，§4.1、§4.2、Theorem 4.7：在多簇分布和两层ReLU设定下，粗标签可能使特征平均化，更细的簇标签有助于区分簇并改善鲁棒性。论文不证明神经方法选择的自然Top1会提高。

项目映射：同一solver的获胜实例可能来自不同竞争模式，细化监督是值得思考的对象。但“预测winner与runner-up的联合分布”有严格边界：联合NLL始终等于winner边际NLL加上给定winner的runner-up条件NLL。即使使用非因式化联合logits，仍不能因此宣称发现了新的必经因果中介。相邻 `loss_math_checks.py` 给出了这个恒等式的CPU检查。

此外，R71已学习联合成本分布，R85已使用非winner成本画像，R43/R100已有pair比较。新方案不能仅因为改成K(K-1)个输出便宣称避免所有旧问题；runner-up近并列还可能使稀有winner的监督进一步碎片化。本轮没有将此候选列为优先开跑方案。

## 4. 条件密度比：一种具体、可证伪的剩余信号目标

[Durkan, Murray & Papamakarios, On Contrastive Learning for Likelihood-free Inference, ICML 2020](https://proceedings.mlr.press/v119/durkan20a.html)连接联合分布对照与密度比推断。借鉴的是概率比的构造，不是模拟器应用收益或该任务的理论保证。

项目候选推导：令g为推理前可得条件，正对来自P(X,Y|g)，负对来自P(X|g)P(Y|g)。在正负抽样各半且支持条件成立时，理想二分类logit为t(x,m,g)=log P(Y=m|x,g)-log P(Y=m|g)。最终得分为log pi_train(m|g)+t，而不是把t直接当冠军概率。若负正比为nu，需补回log nu。当前query标签只用于构造TRAIN损失，不能作为前向输入。

这与无条件AUC或同组regret排序不同：目标是恢复实例与方法之间超出条件先验的关联，而不是仅排好同方法的一列。理论最优与正确的多类CE一致，故它没有增加输入信息，也没有天然更强的Bayes表达力；可能仍只是难以优化的重参数化。需处理TRAIN-only平滑、空支持、分组支持不足以及自然分布下的误选成本。不得以均匀选择或强制召回作为成功条件。

## 5. 互补模型必须真有互补信号

[Allen-Zhu & Li, Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning, ICLR 2023](https://arxiv.org/pdf/2012.09816)，§2.3与理论设定：在特定multi-view分布、两层网络等条件下，独立网络可能学习不同特征，集成与蒸馏可利用这种互补。该机制依赖数据与学习过程，不是“多几个网络必然提升”。

项目边界：模型分数稍有不同或有共同错误，并不足以判定有/无可迁移互补。若多个已完成模型几乎都错在同一批目标winner上，盲目再蒸馏平均输出缺少恢复这些winner的直接证据。知道答案才能逐例选择正确模型的并集是诊断量，不是ensemble成绩或准确率上限。

## 6. 直接算法选择研究：上下文有效，不代表必须新增文本Embedding

[Lu et al., Learning Unified Graph and Language Representations for SMT Algorithm Selection, CP 2026](https://drops.dagstuhl.de/storage/00lipics/lipics-vol379-cp2026/LIPIcs.CP.2026.41/LIPIcs.CP.2026.41.pdf)，§3.2–3.4、§5.3：将公式AST图与推理前可得的应用/来源描述结合；其描述通常来自SMT-LIB头信息。说明结构与合法上下文可能互补，但方法、PAR-2运行时间目标和数据不同。

项目映射：必须先区分缺失运行角色/约束与已有但难学的几何关系。不是把文件名、实例索引或求解结果转成文本，更不能恢复R58已暂停的标注工程。新数据的分布标签是否应进入前向，需要先证明其推理可得性和独立增益，不能把同题同规模置换差异全部称为更细的组合结构信号。

## 7. NSS并不承诺高单方法准确率

[Gao et al., Neural Solver Selection for Combinatorial Optimization, ICML 2025](https://arxiv.org/html/2410.09693v2)，Appendix A.9 Table 8：原池TSP的分类/排序单方法准确率为36%/35%；CVRP为61%/62%。候选池与本项目不同，不可直接当作目标值。§3.3及A.8的Top-k、rejection、Top-p通过运行多个solver交换质量与时间，不是解决单方法Top1。

本次研究仍限定单方法选择。不能把多跑一个solver或根据真实成本重排的离线结果写成模型已经突破平台。

## 8. 路径身份不是更多邻居均值

[Michel et al., Path Neural Networks: Expressive and Accurate Graph Neural Networks, ICML 2023](https://proceedings.mlr.press/v202/michel23a/michel23a.pdf)，§3.1–3.3：比较单条最短路、全部最短路与有界简单路径集合，沿路径编码再聚合；理论区分了这些集合的表达能力。这里借鉴“保留哪些节点共同构成一条路径”，不是只在原GNN多传播两步。

边界尤其重要：论文以无向图和具体路径集合定义理论；本项目含有向带权矩阵、连续坐标与约束。完全欧氏图的最短路通常就是直接边，不能原样提取最短路便期待新增信息；有界简单路径又有密度和长度相关的组合开销。稀疏局部片段是本项目的工程近似，不继承论文全部表达性保证。即便区分更多图，也未证明这些区别对应solver相对优势。
