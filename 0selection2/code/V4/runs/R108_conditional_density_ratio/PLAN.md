# R108: 条件先验与实例证据的学习目标对照

2026-10-24。沿用Astra第二轮调研中的低优先、可精确定义候选，由Sol执行。不是已证实有效的优化，也不是新信息或新的Bayes上限。

## 两组只改变分类代理

- `A_prior_ce`：原R88B完整18题网络产生残余utility t；固定TRAIN拟合先验pi给出s=log(pi)+t，对s做普通native CE。
- `B_density_ratio`：完全相同的网络、先验、完整参数初值及推理；对t使用`0.5*softplus(-t_y)+0.5*sum_m pi_m*softplus(t_m)`。

负项包含真实y，不能排除它后仍声称估计独立乘积分布比。没有额外loss权重、温度、选择配额、均匀采样或新读出头。两组都单次Dropout前向，dropout0.1，去掉comparison、risk和R-Drop；这不是保留R77复合目标的隐式续训。固定纯分类对照用于不把目标间折衷混入密度比机制。不同代理的数值/梯度尺度不相同，不声称等梯度优化。

前向函数、完整初值和总参数一致，但不声称所有参数的数据梯度作用相同：CE对公共utility平移不变，末层共享标量bias的数据梯度消去；ratio代理会锚定t的绝对原点。这是学习目标干预的一部分，而不是为某方法单独添加偏置。

理想二分类logit为`log P(Y=m|x,g)-log pi(m|g)`，所以推理必须补回log pi。它与正确训练的多类CE有相同的理想冠军决策；当前实验仅检验有限模型和预算中的学习差异。若只改善AUC、选择次数或拟合，没有完整系统净收益，即保留阴性，不继续变换权重寻找好点。

## 固定先验是TRAIN学习得到的分类器参数

组仅为问题类型与真实节点数，均是当前输入已有且LIB可得的条件。全问题先验采用Laplace1；精确N先验向该问题先验固定收缩20例，未见N回退问题先验。只读冻结TRAIN的n/native winner计数；不读VAL/TEST，不依赖实例索引或当前query标签。前向只查询已固定的参数buffer，修改query成本/winner不能改变结果。它不是逐实例历史标签上下文或自包含记忆。

这些先验由同一个TRAIN一次生成并锁定，不按VAL/TEST拟合，不改变候选资格，不合并ReLD成本。完整prior buffers随同唯一checkpoint保存；恢复时不重新拟合先验。

## 训练与交付

冻结NEW90k全部18题，81,000/4,500/4,500每题；seed2，从原R84随机初值从头训练，batch1024。57,600次成功更新，warmup4,320；原AdamW、LR1e-4至1e-6、WD1e-4、clip1；完整VAL每1,440次，TRAIN诊断每7,200次。初拟TSP/CVRP micro512、其它1024、eval256；正式锁前只做一次必要的最长TRAIN输入显存见证。

主线程实测最长TRAIN输入单前向反传通过，正式锁定micro512/1024、eval256；TSP/CVRP峰值allocated13,327/13,367MiB，reserved17,278/17,366MiB。无需为占满显存而改变逻辑batch或增加成功更新预算。记录在`preparation/kernel_witness/A_single512_1024/complete.json`。

完整模型选型依旧为VAL macro_vs_sbs_pct严格最小、首个并列点；原始utility差与maximin用于最终完整池排名，winner由s的argmax决定，不改推理配额。报告的eval CE是maximin CE，与A的raw-s训练CE和B的ratio训练loss分别标注。锁定唯一权重后完整TEST/LIB；同模型NSS四集对比、全18题SBS/Oracle/regret、固定36组纠正/伤害及PDF全部保留。

## 卡位

gpu03两张3090保持R107；gpu05/4090_0在R106A完整GPU/报告结束后运行R108A；R108B排队等待原R106B完整交付，不停止微调组。只使用现有job1475容器和UUID/CPU锁，热复用R84缓存，不新建GPU分配、不重跑solver或R58。完成整个R108后集中一次提交GitHub。
