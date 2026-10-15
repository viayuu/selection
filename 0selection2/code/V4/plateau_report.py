"""Render the retrospective diagnosis from its CSVs, without fitting a model."""

import json
from pathlib import Path
import zipfile

import markdown
import numpy as np
import pandas as pd

from code.V4.plateau_analysis import ROOT, PROBLEMS


def table(frame, columns, decimals=3):
    selected=frame[list(columns)].rename(columns=columns)
    lines=['| '+' | '.join(selected.columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in selected.itertuples(index=False,name=None):
        values=[]
        for value in row:
            if pd.isna(value):value='未提供'
            elif isinstance(value,(float,np.floating)):value=f'{value:.{decimals}f}'
            values.append(str(value).replace('|',' / '))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def read(name):
    return pd.read_csv(ROOT/(name+'.csv'))


def fig(name,caption):
    return f'\n![{caption}](figures/{name}.png)\n\n*{caption}*\n'


def main():
    metrics=read('metrics'); signal=read('instance_signal'); comp=read('model_comparisons')
    profiles=read('data_profiles'); portfolios=read('fixed_portfolios')
    current=metrics.query("model == 'R70B' and split == 'test' and problem == 'ALL18'").iloc[0]
    churn=comp.query("reference == 'R45A' and model == 'R70B' and split == 'test' and problem == 'ALL18'").iloc[0]
    common=read('shared_errors').query("split == 'test'")
    common_regret=common.all_wrong_R70_regret_contribution_pp.sum()/18/current.actual_regret_pct*100
    topk=read('topk_error_decomposition').query("model == 'R70B' and split == 'test'")
    topk=topk.groupby('category',sort=False)[['count','regret_contribution_pp']].sum().reset_index()
    topk['error_share_pct']=topk['count']/topk['count'].sum()*100
    topk['regret_share_pct']=topk.regret_contribution_pp/topk.regret_contribution_pp.sum()*100
    gap=read('gap_error_budget').query("model == 'R70B' and split == 'test'")
    gap=gap.groupby('gap_bin',sort=False)[['n','error_count','regret_contribution_pp']].sum().reset_index()
    gap['error_share_pct']=gap.error_count/gap.error_count.sum()*100
    gap['regret_share_pct']=gap.regret_contribution_pp/gap.regret_contribution_pp.sum()*100
    anchors=['R34','R43A','R45A','R60B','R64A','R64B','R65B','R69B','R70B']
    history=metrics[(metrics.problem=='ALL18')&metrics.model.isin(anchors)&metrics.split.isin(['val','test'])]
    history=history.pivot(index='model',columns='split',values=['top1_pct','actual_regret_pct']).reindex(anchors)
    history.columns=['_'.join(c) for c in history.columns];history=history.reset_index()
    stats=signal[(signal.model.isin(['R45A','R70B','NSS']))&(signal.scope=='full')&(signal.split=='test')]
    compact=stats[stats.problem.isin(['ALL18','TSP','CVRP','ATSP','VRPTW','VRPLTW','OVRPTW','OVRPBLTW'])]
    joint=[]
    for p in PROBLEMS:
        a=metrics.query("model == 'R45A' and split == 'test' and problem == @p").iloc[0]
        b=metrics.query("model == 'R70B' and split == 'test' and problem == @p").iloc[0]
        s=stats[(stats.model=='R70B')&(stats.problem==p)].iloc[0]
        profile=profiles.query("split == 'test' and problem == @p").iloc[0]
        joint.append(dict(problem=p,R45_top1=a.top1_pct,R70_top1=b.top1_pct,R70_regret=b.actual_regret_pct,
                          permutation_top1=s.shuffled_top1_pct,permutation_regret=s.shuffled_regret_pct,
                          hindsight_constant=profile.hindsight_best_constant_regret_pct))
    sets=metrics[metrics.model.isin(['NSS','R25','R42B','R45A','R60B','R70B'])&metrics.problem.isin(['TSP','CVRP'])&metrics.split.isin(['test','LIB'])].copy()
    sets['dataset']=sets.problem+sets.split
    sets['dataset']=sets.dataset.replace({'TSPLIB':'TSPLIB','CVRPLIB':'CVRPLIB','TSPtest':'TSP test','CVRPtest':'CVRP test'})
    sets=sets.sort_values(['dataset','model'])
    nsstrain=metrics[metrics.model.eq('NSS')&metrics.split.isin(['train_sample','val','test'])]
    contrasts=comp.query("reference == 'NSS' and split == 'test' and model in ['R45A','R70B']")
    fixed=[]
    for p in ('TSP','CVRP','ATSP','VRPTW','OVRPTW','OVRPBLTW'):
        model=metrics.query("model == 'R70B' and split == 'test' and problem == @p").iloc[0]
        f=portfolios.query("problem == @p and split == 'test' and objective == 'winner_coverage'")
        fixed.append(dict(problem=p,fixed2=f[f.k==2].winner_coverage_pct.iloc[0],model2=model.top2_pct,
                          fixed3=f[f.k==3].winner_coverage_pct.iloc[0],model3=model.top3_pct))
    families=read('error_families').query("model == 'R70B' and split == 'test'").groupby('category')[['count','regret_contribution_pp']].sum().reset_index()
    families['regret_share_pct']=100*families.regret_contribution_pp/families.regret_contribution_pp.sum()
    common_methods=read('universal_error_solvers').query("split == 'test'").sort_values('R70_regret_contribution_pp',ascending=False).head(12)
    objective=read('solver_tradeoffs').query("problem == 'TSP' and split == 'test' and solver in ['BQ','LEHD','DIFUSCO','DIFUSCO500','T2T','T2T500']")
    neglected=read('solver_distributions').query("model == 'R70B' and split in ['train','test'] and ((problem == 'TSP' and solver in ['BQ','LEHD','T2T']) or (problem == 'VRPTW' and solver in ['MVMOE','RouteFinder']))")
    scope=read('scope_metrics').query("split == 'test' and model in ['R45A','R60B','R70B']")
    weight=read('cost_scale_weighting').query("model == 'NSS' and split == 'test' and problem == 'CVRP'")
    vote=read('unweighted_consensus').query("split == 'test'")
    all_perm=signal.query("problem == 'ALL18' and scope == 'full'")
    appendix_links='\n'.join(f'- [{p.name}]({p.name})' for p in sorted(ROOT.glob('*.csv')))
    notes=[]
    notes.append('''# 到 R70 为止，性能究竟卡在哪里？

日期：2026-10-13。范围：原有数据、冻结版本、NSS 本地重训权重。**这是回顾性诊断，不是新模型成绩。**

## 先看结论

**当前证据支持“已学会一些稳定的实例规律，但不同任务的剩余难点不同”，不支持“完全没有学到东西”，也不支持“已经找到一个共同大 bug 或准确率上限”。**

1. 同问题、同真实规模置换预测后，Ours 和 NSS 都明显退化。总体选择频率完全保留，结果却变差，因此频率相似不能解释全部成绩。
2. 平台包含两种不同困难：CVRP、闭合 TW 任务明显学会了实例相关的候选匹配；若干开放任务基本锁定少数方法，难点转为这几种方法的逐例胜负。
3. 多轮改动并非“不改变答案”：它们纠正与新增错误大量抵消，同时有一批长期共同错误。共同错误不是不可学的证明。
4. 残余 regret 并非主要由数值并列造成。不能用提高容忍度后的命中率替代原 Top1，也不能把小成本差直接称为标签噪声。
5. NSS 本地重训有相同类型的低 Top1、漏选部分方法与共享错例，但其成本表现常优于更高 Top1 的 Ours。原论文没有提供足以重做这些诊断的逐例输出，因此不能宣称原论文也存在完全相同机制。
6. 本报告的条件先验、浅层区域及分数条件表只用于定位可迁移信号，不发布新模型、不根据 test 重新选权重。是否继续研究，要看可重复的条件规律与实际收益，不是继续要求下一个 Encoder 必须突破。

## 1. 数据与解释边界

- 18 个原任务；各 train 10,000、validation/test 各 1,000。LIB 为 49 个 TSPLIB、100 个 CVRPLIB。
- 原始 FP64 成本、原生 `ind` 为主口径，按名称对齐候选；Top1 是命中这一原生 winner。另列 FP32 argmin 以桥接历史 NSS 对比。
- NSS 仅有 TSP/CVRP 两套模型，不计算它的“18 任务 ALL”。新导出完整 val/test/LIB 和各 2,000 个按规模分层的训练样本；**训练样本不是完整训练集**。
- 没有新训练神经网络、没有求解器复跑、没有读取 R71/未来版本 test、没有重新选 checkpoint。
- 全任务结果按问题等权宏平均；`actual regret` 是逐实例 `(selected - oracle) / oracle` 的均值。原始 mean cost、均值之比 gap 和平均 relative regret 分开。
- 本项目已经多轮查看历史 test；本轮是探索性分析，不能把显著-looking 的子群当成独立确认。配对 bootstrap 只描述固定预测的样本波动，不代表训练 seed 方差。
- 区间没有重建跨问题基础实例关联；若不同变体含同源实例，跨任务区间可能偏窄，不能把18,000行自动当成完全独立的样本或作严格显著性认证。
- 八个带 B 的旧场景有已知可行域语义冲突。本轮按要求不修复、不重标，相关结果是旧部署场景的统计，不是统一可行域的认证。
- R45A 是稳定历史锚点，R60B 是近期验证参考，R70B 是当前版本；不因本报告的 test 表再挑一个“最佳版本”。完整输入路径见 `prediction_manifest.csv`。
''')
    notes.append('## 2. 平台是真的，但并不是所有版本作出同样选择\n\n'+table(history,{
        'model':'模型','top1_pct_val':'Val Top1 %','top1_pct_test':'Test Top1 %','actual_regret_pct_val':'Val regret %','actual_regret_pct_test':'Test regret %'})+'\n')
    notes.append(f'''R70B 相对 R45A，18,000 个 test 中改选 **{int(churn.changed_count):,}** 例（{churn.changed_count/18000*100:.2f}%），纠正 **{int(churn.corrected):,}** 例，新增错误 **{int(churn.harmed):,}** 例；两者都错时，还有 {int(churn.both_wrong_cheaper)} 例变便宜、{int(churn.both_wrong_dearer)} 例变贵、1例改选但等成本。不是网络没有反应，而是变化没有形成可靠的净收益。

R60B 相对 R45A 的 test Top1 只提高 0.333 个百分点，配对区间包含零；regret 也没有清晰拉开。不要把不同版本、不同 epoch 当作多个独立 seed。

从 15 个有完整 test 逐例预测的版本看：共同选错 **{common.all_wrong_pct.mean():.2f}%** 的实例，贡献 R70B **{common_regret:.2f}%** 的 regret；另有大量错误在版本间交换。即使把这些版本简单多数投票，Top1 也只是 **{vote.top1_pct.mean():.2f}%**、regret **{vote.actual_regret_pct.mean():.4f}%**。这只是固定版本集合的诊断，未计多模型推理开销，不作为正式集成成绩。

“任意一个版本曾经选对”的事后并集是 {common.hindsight_union_accuracy_pct.mean():.2f}%，但需要先知道每例哪个版本正确，**既不可部署，也不是准确率上限，更不是已经证明 70% 可学**。
''')
    notes.append(fig('01_frozen_full18_trends','图1：冻结权重的验证与测试轨迹。版本横轴不是独立随机重复。'))
    notes.append(fig('07_corrections_vs_r45','图2：纠正与伤害、两者都错时的成本变化必须一起看。'))
    notes.append('''## 3. “我是不是只学了方法的总体分布？”答案是否定的

将完整分数向量在**同问题、同真实规模**内置换 100 次，保留选方法频率、候选池与分数分布。唯一被破坏的是实例与分数的对应。每组至少两例；单例规模不放宽，保持原预测，并另报可置换子集。
实现为节省计算只置换各分数行的argmax；对这里报告的Top1/regret，这与整行分数置换后再取argmax完全等价，不代表同时检查了置换后的全部排名或校准。
'''+table(compact,{'model':'模型','problem':'问题','actual_top1_pct':'原 Top1 %','shuffled_top1_pct':'置换 Top1 %','actual_regret_pct':'原 regret %','shuffled_regret_pct':'置换 regret %'},3))
    notes.append('''
TSP/CVRP test 可置换覆盖率分别为89.6%/90.3%，其余任务100%；ALL 为98.88%。主表中未置换的单例保持不动，因此不是放宽规模后的比较。完整可置换子集见 `instance_signal.csv`。

**最关键的反例是 CVRP 和闭合 TW：相同规模下，模型明显知道哪些实例应交给不同方法。** ATSP 的高准确率与规模条件更相关，其置换前后差距小。若干开放任务则只使用了较少的额外实例信号。这些结论说明信号使用量不均衡，不能证明剩余误差没有信号。

置换区间是重复随机置换的分布，不是总体置信区间或“准确率天花板”。
''')
    notes.append(fig('02_within_size_signal','图3：任务差异很大，ALL Top1 不能定位统一瓶颈。'))
    notes.append('## 4. 每个问题的机会与剩余损失\n\n'+table(pd.DataFrame(joint),{
        'problem':'问题','R45_top1':'R45A Top1 %','R70_top1':'R70B Top1 %','permutation_top1':'R70置换 Top1 %',
        'R70_regret':'R70 regret %','permutation_regret':'R70置换 regret %','hindsight_constant':'事后最佳常量 regret %'},3))
    notes.append('''
最后一列是在该评估集上事后挑出的“固定选一个方法”的最低平均 relative regret，**只描述候选池互补空间，不能作为训练得到的可部署基线**。训练选定的固定方法与条件先验另外列出。必须区分“总体就比较好的方法”与“给每个实例找到合适方法”。

- **CVRP、VRPTW、VRPLTW**：模型已经消除了大量实例依赖的成本风险。继续研究有任务价值，但要找到它尚未利用的优势条件，不能再以低 Top1 证明没学到。
- **ATSP**：高 Top1 的相当部分可由规模条件解释；相对粗先验的增量更重要，不能仅凭绝对准确率宣布表征最好。
- **OVRPTW、OVRPBLTW 等**：模型相对合适常量的成本优势有限。这里有真实的有限池 Oracle 空间，却尚未定位强而可迁移的逐例胜负规则。
- **TSP、VRPL**：多方法分担赢家，不能只研究 MOEL/MTL；静态常量、候选召回、风险偏好与精细排序需要分开。
''')
    notes.append('''## 5. 找到前几名，与排对第一名是两种能力

固定组合仅用 train 的 winner 频率确定，带到 test；表中的 Top2/3 是“包含 winner”的覆盖率，不是执行一次 solver 的成功率。
'''+table(pd.DataFrame(fixed),{'problem':'问题','fixed2':'固定 Top2覆盖 %','model2':'R70 Top2 %','fixed3':'固定 Top3覆盖 %','model3':'R70 Top3 %'},2))
    notes.append('''
OVRPTW、OVRPBLTW 的高 Top3 几乎可以由固定强方法解释；不能把接近100%的 Top3 自动归功于精细的实例排名。反过来，CVRP 和 VRPTW 有明显的动态覆盖增益。

另一个 train 定义的控制按最小平均 relative gap 选固定方法组合，结果在 `fixed_portfolios.csv`。频率最优组合与成本最优组合并不相同。组合内取最小成本都需要多个 solver 结果，仅作预算不同的诊断，不能冒充单方法成绩。

R70B 的全任务 test 错误分解如下：
'''+table(topk,{'category':'winner位置','count':'错误数','error_share_pct':'占错误 %','regret_share_pct':'占regret %'},2))
    notes.append(fig('11_fixed_vs_adaptive_top3','图4：动态候选覆盖与训练固定三方法覆盖的区别。'))
    notes.append(fig('04_topk_error_budget','图5：错误数量与错误成本要分别记账。'))
    notes.append('''## 6. 不能把剩余错误主要解释为近似并列

下面按**真实第一、第二方法的成本差**分桶，而不是模型置信度或模型选错后的 regret。
'''+table(gap,{'gap_bin':'真实前二差距','n':'实例数','error_count':'错误数','error_share_pct':'占错误 %','regret_share_pct':'占regret %'},2))
    notes.append('''
差距超过2%的实例贡献约一半剩余成本损失。因此，“Top1 低只是因为所有赢家成本太接近”并不能解释整个18任务。TSP 的确更细：其 test 前二差距中位数仅0.287%，69.9%的实例不超过0.5%；CVRP中位数0.556%。不能从 TSP 推广到全部 MVRP。

R70B 允许0.1% regret时命中49.69%，允许0.5%时59.99%，允许1%时70.37%。这些是额外的成本容忍指标，**严格 Top1 仍是46.72%**。数值很近也不等于求解随机或不可预测；大差距也不保证仅凭有限训练样本就容易预测。
''')
    notes.append(fig('03_margins_and_tolerance','图6：真实成本差分布与成本容忍曲线，不替代原始 Top1。'))
    notes.append('''## 7. 哪些方法与错误长期被忽略？

R70B test 的损失不仅来自一个方法对：
'''+table(families,{'category':'比较类别','count':'错误数','regret_share_pct':'占regret %'},2))
    notes.append('''
同族仅按 RELD、MoSES、DIFUSCO、T2T 名称前缀定义，是描述性分组。MOEL/MTL依然重要，但非该方法对的错误贡献约一半成本损失，不应再次把整体问题压缩成这一个二分类。

以下是15个完整版本**共同漏掉**的 winner。这里的成本贡献按各任务均值列出；转换成 ALL 贡献需除以18。
'''+table(common_methods,{'problem':'问题','solver':'真正winner','count':'共同漏选数','total_winner_count':'该方法赢家总数','R70_regret_contribution_pp':'任务内regret贡献 pp'},4))
    notes.append('''
这给出了比“再换一种结构”更具体的解释对象：例如 TSP 的 LEHD，以及闭合 TW 中的 RouteFinder/MoSES，哪些输入区域让它们具备稳定优势？但名单来自历史结果，不能据此在同一 test 设计规则后再宣布验证成功。
''')
    notes.append(fig('05_winners_vs_picks','图7：赢家频率、模型选择频率与成本相关的漏选不是同一个量。'))
    notes.append(fig('06_model_agreement','图8：预测相近及共同错误，不代表这些相关模型提供独立的不可学证据。'))
    notes.append('''## 8. 为什么不应该强制预测频率匹配 winner 频率？

频繁小幅获胜、但失败时损失较大的方法，可能被成本导向决策保守使用。TSP 的例子如下；“独有贡献”是移除该方法后 Oracle 平均变差多少，仍是事后有限池诊断。
'''+table(objective,{'solver':'方法','native_winner_pct':'赢家频率 %','mean_win_advantage_pct':'获胜时平均领先 %',
                     'mean_loss_when_not_best_pct':'落败时平均损失 %','loss_p95_pct':'固定选它的p95 regret %','recoverable_winner_gain_pp':'独有成本贡献 pp'},4))
    notes.append('''
R70B在TSP选BQ约65%，但BQ的winner频率28%；LEHD频率18.5%，模型只选约0.5%。NSS也几乎不选LEHD。**这证明行为偏好与真实winner分布不同，却不能仅凭频率表判定它是否合理。** 需要同时看误选它的代价、漏选它的代价，以及能否训练出可迁移的获胜条件。

同一分区内分别用“多数winner”和“最小平均relative gap”作决定，是控制信息相同后检验目标取舍的方式；见下文的训练区域对照。调整输出温度不会改变argmax，单纯改善校准也不会自动修正Top1。
''')
    notes.append('''### 一项额外定位：部分漏选在训练集就存在

以下使用R70B同一锁定权重的完整train/test输出，不是训练batch准确率。
'''+table(neglected,{'problem':'问题','split':'划分','solver':'方法','winner_pct':'赢家占比 %','pick_pct':'选择占比 %','winner_recall_pct':'该类召回 %'},3)+'''

LEHD、T2T以及部分MVMOE赢家在训练集也很少被选中。因此，不能把这些类的低召回都解释成“测试分布变化”或“训练已经充分分清，只有泛化失败”。它仍可能涉及条件信息不足、学得不足或目标取舍；仅靠这张表无法在三者中定因，也不支持直接均衡输出频率。
''')
    notes.append('''## 9. NSS 与 Ours：相同数据上的四项对照

以下全部重算到原始标签和FP64成本。R45A/R60B/R70B没有保存同口径LIB逐例预测，本轮**不伪造这些格子**；LIB使用已有R25/R42B。NSS CPU导出在历史FP32口径下复现四项历史Top1/Top3计数。
'''+table(sets,{'dataset':'数据集','model':'模型','n':'数量','top1_pct':'原生Top1 %','top3_pct':'原生Top3 %',
                'top1_fp32_pct':'历史口径Top1 %','top3_fp32_pct':'历史口径Top3 %','actual_regret_pct':'actual regret %','mean_cost':'mean cost'},4))
    notes.append('''
**并不是 Ours 的 Top1 稍高就全面超过 NSS。** 例如CVRP test，R45A比NSS多选对22例，却有更高regret；R70B比NSS多选对5例，成本也更高。TSP中两者同样有这种取舍。原生与FP32标签的几例差异不足以解释多轮平台。

同实例的配对比较：
'''+table(contrasts,{'problem':'问题','model':'相对NSS的模型','changed_count':'改选数','corrected':'纠正','harmed':'新增错误',
                     'delta_top1_pp':'Top1差 pp','delta_regret_pp':'regret差 pp'},4))
    notes.append('''
R45A与NSS的 test 错误集合 Jaccard：TSP约0.867、CVRP约0.782；不是所有错例相同，但共同难例很多。它们共享训练场景和候选池，不能当作独立证据证明这些实例无法学习。

### NSS 自身也不是只在测试集突然失败

训练样本是固定seed的分层抽样，不是挑容易实例；每个问题2,000例。
'''+table(nsstrain,{'problem':'问题','split':'划分','n':'数量','top1_pct':'Top1 %','top3_pct':'Top3 %','ce':'分数诱导CE','actual_regret_pct':'regret %'},4))
    notes.append('''
本地NSS的训练样本、val、test准确率接近。因此不能把其低test Top1统一归因于严重过拟合；也不能反过来认定它已充分优化或输入不含信息。它是单seed的本项目候选池复现，不是原论文五seed实验的严格重做。

实现保留NSS主体与排序目标，但数据、候选池变为本项目8个TSP/10个CVRP方法；历史validation缺少真实optimality-gap参照而使用raw mean cost选型。这与按实例relative gap等权选型不同。现有记录不足以量化另一种选型会得到什么最终test结果，本轮没有重选。

### 原论文能告诉我们什么？

NSS附录A.9/Table8给出classification/ranking的greedy Top1：TSP 36±1%/35±1%，CVRP61±1%/62±0%，TSPLIB40±3%/40±7%，CVRPLIB52±2%/56±3%，为五次运行。原池是7个TSP/5个CVRP方法；其gap参照最佳已知解，不是本项目成本列Oracle。论文A.6报告多方法执行的成本曲线，没有可直接抄作Top3 winner命中率的表。它讨论质量与执行预算取舍，但没有证明低Top1来自不可消除噪声或给出准确率上限。[NSS原文](https://arxiv.org/html/2410.09693v2#A9)

因此可以说“低greedy Top1并非只出现于本项目”，不能说“原论文已证明这批剩余实例不可区分”。

LIB很小，TSPLIB改变一例就是2.04个百分点，CVRPLIB是一百分点；不能把小数点后的差别当成稳健泛化结论。
''')
    notes.append(fig('08_four_dataset_accuracy','图9：四项数据集的Top1/Top3，缺失结果保持缺失。'))
    notes.append(fig('09_four_dataset_cost','图10：准确率优势不必然是选择成本优势。'))
    notes.append('''## 10. 规模权重和评估目标会改变“更好”的含义

若每例relative regret为r，Oracle成本为c*，则：

`均值成本之比 gap = sum(c* × r) / sum(c*)`，而 `actual regret = mean(r)`。

前者按参考成本加权；规模往往与成本相关，但不是严格按节点数加权。Top1则不区分误选代价。下面以NSS CVRP test按Oracle成本四等分作**描述性**比较，不是训练输入或分桶策略：
'''+table(weight,{'cost_quartile':'成本四分位','mean_true_size':'平均规模','mean_reference_cost':'平均Oracle成本',
                 'top1_pct':'Top1 %','regret_pct':'regret %','relative_gap_weight_share_pct':'均值之比权重 %',
                 'relative_error_share_pct':'relative误差贡献 %','absolute_error_share_pct':'绝对误差贡献 %'},3))
    notes.append('''
最高成本四分位占均值之比权重约39.75%，最低只占13.35%；而每例平均regret给两组各25%的实例权重。这是实质差别，但**不能据此宣称它造成了47%平台**。此外，R45A、R60B、R70B各自现有validation历史中，Top1最佳和regret最佳恰好同一轮，因此最近这些锁定结果也不能简单归咎于“只差一个选型指标”。
''')
    notes.append(fig('13_cost_scale_and_solver_value','图11：评价权重与不同方法的独有成本贡献。'))
    notes.append('''## 11. 训练曲线排除不了什么？

R70B锁定点完整train Top1约50.89%、val48.38%；末期train58.14%而val45.32%。R45A末期train50.22%、val49.02%。因此历史中既有显著过拟合，也有选中模型区分能力有限，不能把所有失败归为一种原因。

同样，“能把R49的2,000例背到100%”不等于当前完整系统在优化充分后只能到这个水平；“训练验证接近”也不等于没有可学信息。近期版本的初始化、正则化、任务组织和预算并不全相同，历史曲线是定位证据，不是一个干净的容量或loss消融。

置信度图只描述softmax分数诱导的行为。NSS排序分数不是已经校准的winner后验；ECE不同不能独立解释谁的argmax更好。
''')
    notes.append(fig('12_learning_and_selection','图12：优化曲线与冻结选型，不能把不同版本当独立seed。'))
    notes.append(fig('10_confidence_and_wrong_cost','图13：置信度与错误成本，仅为描述，不据此调阈值。'))
    region_report=ROOT/'regions/findings.md'
    notes.append('## 12. 哪些可观察条件能跨实例迁移？\n\n')
    if region_report.exists():
        base=pd.read_csv(ROOT/'regions/baselines.csv')
        policies=['majority','mean_gap','global_sbs','winner_tree_leaf_majority','gap_tree_leaf_mean_gap']
        b=base[base.policy.isin(policies)&base.split.isin(['val','test'])&base.problem.isin(PROBLEMS)]
        b=b.groupby(['policy','split'])[['top1','mean_relative_gap_pct']].mean().reset_index()
        b['top1_pct']=b.top1*100
        b=b.pivot(index='policy',columns='split',values=['top1_pct','mean_relative_gap_pct'])
        b.columns=['_'.join(v) for v in b.columns];b=b.reset_index()
        notes.append('''### 12.1 简单统计已经能解释一部分稳定规律

所有规则只使用train。规模先验按精确规模向训练四分位箱、再向问题总体平滑（伪计数20）；分布外规模回退问题总体。浅树预先固定深度3、最小叶节点200，不搜索深度或调参；每个问题分别拟合winner树和完整gap向量树，**不是一棵跨18任务共享的树**。它们是统计探针，不是本轮新增的神经模型，也不是与共享网络严格匹配容量/共享方式的结构对照。

下面的majority与mean_gap分别追求正确次数和relative成本，不混成一个基线：
'''+table(b,{'policy':'训练定义策略','top1_pct_val':'Val Top1 %','top1_pct_test':'Test Top1 %',
           'mean_relative_gap_pct_val':'Val regret %','mean_relative_gap_pct_test':'Test regret %'},4)+'''

深度3的winner树达到test 46.19%，比规模多数先验42.20%强，却仍未超过R45A 47.91%及其成本表现。这说明粗几何/约束统计能够利用真实信号，同时复杂模型相对于这种低成本描述的增量也没有想象中大。**这不是树已逼近理论极限，而是目前观察到的可利用规律有相当部分很粗粒度。**
''')
        transfers=pd.read_csv(ROOT/'regions/region_transfer.csv')
        examples=transfers[(transfers.problem.eq('CVRP')&transfers.scope.eq('demand_quantile')&transfers.region.isin([2,3,4,5]))|
                           (transfers.problem.eq('ATSP')&transfers.scope.eq('size_bin'))].copy()
        examples['region_name']=examples.problem+' / '+examples.scope+' '+examples.region.astype(str)
        notes.append('''### 12.2 能明确写出来的偏好反转

CVRP以容量归一化的平均需求0.0468134作train中位数边界，规模分箱也是train确定。规模162至277时，低需求区更偏RELD_CVRP，高需求区更偏OMNI；规模278至389也如此。下表正值表示A平均比B便宜，单位是相对完整候选池Oracle的百分点。
'''+table(examples,{'region_name':'区域','ref_a':'A','ref_b':'B','train_support':'Train数','val_support':'Val数','test_support':'Test数',
                    'train_mean':'Train优势 pp','val_mean':'Val优势 pp','test_mean':'Test优势 pp'},3)+'''

ATSP则有很强的规模分段偏好，解释了为什么规模先验已达约78%。这两种现象都是实际数据支持的“不同方法擅长不同区域”，不是从方法名称或论文印象猜出来的。

闭合VRPTW也有可迁移的粗结构：训练浅树按depot距离、客户规模、服务与时间余量划分，分别偏向RELD、MoSES、RouteFinder。仅此已将test Top1从规模先验27.8%提高到40.4%；R45A为43.5%。相反，OVRPTW同样深度的树只有44.8%，与规模多数先验44.4%接近。**任务之间的差别是真实的，不能统称“TW都容易/都困难”。**

完整规则与所有支持度见[区域表](regions/regions.csv)；不只保留看起来有效的区域。部分规则由train标签发现，其train效果会乐观；区域相互重叠，不能把多个一致方向当成独立重复实验。稳定平均优势也不等于每个实例都能选对，更不能证明当前模型遗漏了全部这些条件。
''')
        notes.append(fig('14_advantage_region_transfer','图14：固定训练树的叶区域在train、val、test上的迁移；不是在test重新划分。'))
        notes.append(fig('16_region_advantage','图15：需求/规模改变时，方法平均优势确实会反转。'))
        notes.append('''### 12.3 相同信息下，准确率目标与成本目标确实不同

固定同一套训练规模分箱，只改变格子里的决策目标：TSP test用mean-gap而非多数winner，Top1下降7.3个百分点，regret却降低0.507个百分点；CVRP对应下降2.8个百分点、regret降低0.711个百分点。VRPTW反而可以同时提高Top1与降低成本。可见目标冲突在一些区域存在，在另一些区域并非不可避免。

这比比较两个不同架构的最终分数更能隔离“目标不同”的影响，但仍然只在当前粗信息分区上成立，不能直接推出换某个训练loss一定有效。逐任务结果见[同区域决策取舍](regions/decision_tradeoffs.csv)。
''')
        shifts=pd.read_csv(ROOT/'regions/shift_summary.csv')
        short=shifts[shifts.problem.isin(['TSP','CVRP','ATSP','VRPTW','OVRPTW'])&shifts.split.ne('train')].copy()
        short['outside_size_pct']=short.extrapolated_size_fraction*100
        short['outside_feature_pct']=short.fraction_any_non_size_feature_outside_train*100
        notes.append('''### 12.4 常规留出与LIB需要分开解释

18任务常规val/test规模都在各自train范围内；所测非规模统计的最大标准化均值偏移不到0.10、最大KS距离约0.06。这不支持很大的低阶统计来源差异，**但不能证明联合分布一致或覆盖充分**。

LIB明显不同：TSPLIB有14.3%、CVRPLIB有32%的实例超出训练规模；至少一项非规模特征超出train范围的比例分别87.8%/52%。后者受特征个数影响，不能当作严格OOD概率，但与规模、SMD、KS一起看，分布外风险确实更大。
'''+table(short,{'problem':'问题','split':'划分','min_size':'最小规模','max_size':'最大规模','outside_size_pct':'规模外推 %',
                  'outside_feature_pct':'任一非规模特征越界 %','max_non_size_abs_smd':'最大绝对SMD','max_non_size_ks':'最大KS距离'},3)+'''

因此，CVRPLIB上R42B显著差于NSS/R25，不能只拿synthetic test接近就解释为偶然；需要保留独立的外推评价。不过LIB样本小，当前证据不允许指定是哪一项特征变化造成下降。

几何距离统计在大于128客户时使用按坐标固定选取的128客户；最近邻统计因此是该子集的统计，不假装完整图最近邻。所有原实例、标签、划分不变。完整英文方法说明与辅助表见[regions/findings.md](regions/findings.md)。
''')
    else:notes.append('区域统计仍在生成；最终交付前更新。\n')
    residual=ROOT/'residual_signal.csv'
    if residual.exists() and residual.stat().st_size>10:
        residual_df=pd.read_csv(residual)
        view=residual_df[residual_df.problem.eq('ALL18')]
        notes.append('''### 对剩余信号的一项受控检查

用R70训练分数建立条件表：训练规模四分位、**有序**预测Top2方法对、预测分数差三分位，保留原模型选择方向；再只增加一个预先固定的输入描述符。TW为局部有向slack均值，ATSP为矩阵不对称均值，其他为最近邻距离均值乘规模平方根。切分点来自train，不按test错误挑特征；每格至少40例、伪计数50。score与feature格均收缩到同一个规模/方法对父格，无支持的feature格退回score表；常量特征必须不改变预测，18任务均通过此性质检查。

在完全相同的格子里分别按**非并列两方法胜负频率**、平均相对成本差决定二者顺序，不新训神经网络。前者不是原生全池winner多数：即使真正winner在Top2外，两候选仍可有二者间胜负。原生winner目标对照使用上面的regions表；此处只比较pairwise决策。结果仍按完整候选池计Top1/regret：
'''+table(view,{'split':'划分','representation':'条件信息','objective':'决策目标','top1_pct':'Top1 %','actual_regret_pct':'regret %','own_cell_support_pct':'直接格子覆盖 %','fallback_pct':'父格回退 %'},4))
        notes.append('''
训练分数本身是样本内预测，所以条件表仍可能过拟合；稀疏区域回退率必须一起看。这个检查回答的是这些粗描述符是否有额外可迁移价值，**不是穷尽全部输入信息，也不是证明Encoder或标签的极限**。逐任务结果、条件平均差的方向迁移见 `residual_signal.csv`、`residual_region_transfer.csv`。

**这里有小幅可恢复的决策收益，不能一概说没有信号。** 仅用粗分数条件，mean-gap表在val从R70B的48.38%/0.9888%变为48.54%/0.9821%，test从46.72%/1.0052%变为47.37%/0.9859%。这提示同一模型的输出与实际成本关系仍有少量可利用空间，但仍低于R45A的47.91%/0.9432%，不构成整体突破，也不是一次正式新模型测试。

再增加描述符后，val成本略好但Top1下降，test两项都比score成本表差；pairwise频率分支也呈混合结果。86.27%的test实例有足够自身格子支持，但计数够不等于估计可靠。准确结论是“粗分数条件可恢复一小部分R70退化；这一个额外描述符未带来跨划分、跨指标一致的明显优势”，不是“原始输入不存在剩余规律”。

评审发现初版表丢掉Top2方向且多做一层收缩，已局部修正，未调整分桶、支持门槛或特征；初版留在 `residual_initial/`，不用于最终结论。原模型成绩、其余统计及checkpoint均未改。
''')
        notes.append(fig('15_residual_signal','图16：保留选择方向后，分数条件表有小幅收益；额外描述符结果混合。'))
    notes.append('''## 13. 已知数据边界不能被性能图掩盖

带B旧任务与不在该语义冲突范围内的10任务分开列出：
'''+table(scope,{'model':'模型','scope':'范围','top1_pct':'Test Top1 %','actual_regret_pct':'Test regret %'},4))
    notes.append('''
这不是在本轮排除B任务或修改标签；只是显示平台在非B部分也存在，不能把全部失败归结于已发现的语义冲突。反过来，也不能因为模型能学习旧标签，就宣称跨solver可行域已经统一。

已知与未知必须分开：目前没有全候选池完整重复求解分布，不能估算总体“标签噪声”；没有新的独立测试集，不能用多轮历史test声称严格确认；没有所有内部表示的充分性检测，不能定位为某一个具体网络模块。

## 14. 是否值得继续，以及该停止什么

**值得继续研究“在可比较的场景中利用实例差异做单方法选择”；不值得继续把同一批历史test的微小波动包装成结构突破。** 现有数据已经反驳了“完全没有实例信号”，但还没有证明剩余空间能被当前训练方式恢复。

后续投入应分问题、按证据排序：

1. **已有效的匹配区域**：CVRP、闭合TW保留为正对照。要求新思路超过训练定义的条件先验，并解释减少了哪类跨方法风险，不能只看ALL。
2. **长期漏选的有效方法**：依据train里稳定的获胜区域定义假设，在新冻结留出上检查；例如TSP LEHD、TW中的其他方法族。高winner频率本身不是强行均衡输出的理由。
3. **固定强候选但细粒度难分的区域**：开放TW等已有多轮阴性结果。若本报告的简单条件与残差检查也找不到可迁移增量，先承认“现有分析未找到”，不自动再加一个头或宣布随机。
4. **任务有效性**：旧B语义问题仍是研究结论的边界，遵照你的决定本轮只记录；未来要发表统一可行域的比较时仍需明确处理，不能由新网络消除。
5. **方法与目标的声明**：若研究目标是单方法原生Top1，就保持这一目标；若研究实际求解成本，必须同时报告误选的大小与推理预算。Top-k运行多个solver不是修好了单方法Top1。

本报告不推荐一个未经验证的新架构，也不承诺修一处就能到60%/70%。最值得提出的新假设应当写成“哪个train可观察条件、对哪组部署、预期减少哪类代价”，而不是“下一种Attention可能更强”。

## 15. 文件、复现和研究依据

- [完整图册 PDF](figures/charts.pdf)，每张图同时保留PNG/PDF。
- [预测来源清单](prediction_manifest.csv)、[缺失项](missing_predictions.csv)、[分析协议](analysis_protocol.json)。
- [NSS真实导出记录](nss_export_manifest.json)；未把2,000例训练抽样当全量train，未调用底层solver。
- [规划评审](../../../../review-stage/plateau_r70/PLAN_REVIEW.md)、[结果评审](../../../../review-stage/plateau_r70/RESULT_REVIEW.md)。
- NSS：原论文仅作为背景和口径依据，本地逐例结论由本次导出得到。
- ASlib强调固定场景、单最佳/逐例最佳参考和误选代价；本报告借用这种区分，不把事后Oracle空间当可预测性证明。[ASlib原论文](https://www.cs.ubc.ca/~kevinlb/papers/2016-AIJ-ASlib.pdf)
- Solver贡献应与单独性能区分，频繁winner与独有成本贡献可能不同。[Xu等，2012](https://www.cs.ubc.ca/~kevinlb/papers/2012-SAT-EvaluatingSolverContributions.pdf)

复现仅统计与渲染，不训练：

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -m code.V4.plateau_analysis
python -B -m code.V4.plateau_regions run
python -B -m code.V4.plateau_residual
python -B -m code.V4.plateau_figures
python -B -m code.V4.plateau_report
```

NSS已有缓存，不需要再次推理；缺缓存时使用 `plateau_nss_export.py --help` 中的冻结导出入口。主代码没有覆盖历史结果或checkpoint。

### 数据表索引

'''+appendix_links)
    text='\n\n'.join(notes)+'\n'
    (ROOT/'RESULTS.md').write_text(text)
    renderer=markdown.Markdown(extensions=['tables','fenced_code','toc'],extension_configs={'toc':{'toc_depth':'2-2'}})
    body=renderer.convert(text)
    body=body.replace('<h2 id=', '<nav aria-label="目录">'+renderer.toc+'</nav><h2 id=',1)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>性能平台诊断：R70 / NSS</title><style>
:root{color-scheme:light;--ink:#233238;--muted:#607077;--line:#cbd5d8;--accent:#167c80}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#f5f7f8;font:16px/1.8 "Noto Sans CJK SC","Microsoft Yahei",sans-serif;letter-spacing:0}
main{max-width:1180px;margin:auto;padding:40px 28px 80px;background:#fff}h1{font-size:30px;line-height:1.4}h2{font-size:24px;border-top:2px solid var(--line);margin-top:44px;padding-top:20px}h3{font-size:20px}a{color:var(--accent)}p,li{overflow-wrap:anywhere}nav{border-block:1px solid var(--line);padding:10px 0;font-size:14px}nav ul{columns:2;padding-left:22px}nav li{break-inside:avoid;margin-bottom:4px}img{display:block;max-width:100%;height:auto;margin:20px auto}table{display:block;overflow-x:auto;border-collapse:collapse;max-width:100%;font-size:14px;line-height:1.6}td,th{padding:8px 10px;border:1px solid var(--line);white-space:nowrap}th{background:#edf3f3}tr:nth-child(even){background:#f7f9fa}pre{background:#edf1f2;padding:16px;overflow:auto}code{font:13px/1.6 monospace}em{color:var(--muted)}@media(max-width:600px){main{padding:20px 14px}h1{font-size:26px}h2{font-size:22px}table{font-size:12px}nav ul{columns:1}}@media print{main{max-width:none;padding:0}body{background:white}img{max-height:25cm;object-fit:contain}h2{break-after:avoid}table{font-size:11px}}
</style><main>'''+body+'</main></html>'
    (ROOT/'report.html').write_text(html)
    (ROOT/'findings.json').write_text(json.dumps(dict(prediction_sets=json.loads((ROOT/'analysis_protocol.json').read_text())['prediction_sets'],
        full18_current_top1=float(current.top1_pct),full18_current_regret=float(current.actual_regret_pct),
        common_error_pct=float(common.all_wrong_pct.mean()),common_error_regret_share=float(common_regret),
        type='retrospective descriptive analysis, not confirmatory experiment'),indent=2)+'\n')
    files=[p for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ('.csv','.md','.html','.png','.pdf','.json'))
           and not any(s in p.parts for s in ('feature_cache','implement-stage','.mplconfig','nss_predictions'))
           and p.name!='nss_export_manifest.json']
    with zipfile.ZipFile(ROOT/'analysis_bundle.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for path in files:bundle.write(path,path.relative_to(ROOT))
        for name in ('plateau_analysis.py','plateau_regions.py','plateau_residual.py','plateau_figures.py','plateau_report.py','plateau_nss_export.py'):
            path=ROOT.parents[1]/name
            if path.exists():bundle.write(path,'scripts/'+name)
    print(f'[report] {len(text)} characters; {len(files)} report/data/figure files',flush=True)


if __name__=='__main__':main()
