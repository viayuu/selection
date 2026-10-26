"""Write the saved-results synthesis; fail closed on publication byte identity."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPO = Path('/public/home/shiys/0selection2')
ACTION = 'e1021815bf3f51b6f34424bf'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(columns, values):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(columns)) + ' |',
                      *['| ' + ' | '.join(map(str, value)) + ' |' for value in values]]) + '\n\n'


def number(value, precision=6):
    return f'{float(value):.{precision}f}'


def main():
    audit = read(HERE / 'saved_results_authentication.json')
    publication = read(HERE / 'publication_byte_verification.json')
    snapshot = read(HERE / 'source_snapshot.json')
    checkpoints = rows(HERE / 'checkpoints.csv')
    metrics = rows(HERE / 'metrics.csv')
    assert len(checkpoints) == 7 and len(metrics) == 280
    # This invocation may only report a failed strict publication gate. It does
    # not reinterpret EOL normalization as authorization to change native hashes.
    assert publication['strict_published_bytes_pass'] is False
    assert len(publication['mismatches']) == 92
    assert all(r['CRLF_to_LF_only'] for r in publication['mismatches'])
    model_names = [(r['experiment'], r['arm']) for r in checkpoints]
    short = {(v, a): v + a[0] for v, a in model_names}
    index = {(r['experiment'], r['arm'], r['split'], r['problem']): r for r in metrics}
    all_rows = {(r['experiment'], r['arm'], r['split']): r for r in audit['all_summary']}
    problems = ['TSP', 'CVRP', 'ATSP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW',
                'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW', 'OVRPBL',
                'OVRPBTW', 'OVRPLTW', 'VRPBLTW', 'OVRPBLTW']
    mismatch_counts = collections.Counter(r['experiment'] for r in publication['mismatches'])
    text = [
        '# 保存结果证据包：发布字节认证受阻\n\n',
        f'本次 action：`{ACTION}`；请求模型 `gpt-6.1-sol`，`reasoning_effort=max`。'
        '**状态为 blocked，verification.complete=false。** 七个唯一 VAL checkpoint、280 行同版本指标和保存报告已核验；'
        '没有签发整个证据包完成证明，也没有开启 Astra。\n\n',
        '四个指定提交在实际 origin/main 历史中均可见。550 个清单文件条目中，458 个文件的工作区字节与提交字节完全一致；'
        '**92 个 CSV 的发布清单 SHA256 与实际 Git blob SHA256 不同**，全部仅为 CRLF→LF，解析内容一致。'
        '这不是测得的科学退化或数值篡改；但不能把不同的字节 SHA 写成相同。逐文件双哈希及转换证明见'
        '[publication_byte_verification.json](publication_byte_verification.json)。原清单、回执、提交、原始产物均未修改。\n\n',
        '所需动作：由主线程/对应发布所有者对账这 92 项，提供版本化附加回执，显式绑定“原文件 SHA256 → 实际 commit/blob SHA256”及换行转换，'
        '明确权威发布字节身份并协调 monitor 的证据快照。当前核验不能改写其发布清单或原回执，也不会重复提交/推送。'
        '现成逐项对账材料已经保存，无需重新训练或求解。\n\n'
    ]
    text.append(table(['版本', '实际提交', '核对文件条目', '字节不同的 CSV'], [
        [p['experiment'], '`' + p['commit'] + '`', p['verified_asset_count'], mismatch_counts[p['experiment']]]
        for p in publication['publications']]))
    sample = next(r for r in publication['mismatches'] if 'metrics_by_problem.csv' in r['path'])
    text.append(f"例如 `{sample['path']}`：原文件 `{sample['working_sha256']}`；"
                f"提交 blob `{sample['git_blob_sha256']}`。全部 135 个 source_snapshot 文件仍与冻结 SHA 一致；"
                '快照对象的规范化 SHA 是 '
                f"`{audit['scientific_completion_gate_sha256']}`，不能与 JSON 文件本身的字节 SHA 混淆。\n\n")
    text.append('## 实际完成与保留的失败\n\n')
    text.append(
        '**R109：** 两个完整模型已有 TRAIN/VAL/TEST、LIB、PDF 和实际提交 43e3432。原 watcher 仍为 '
        '`stopped_on_error` / `Publication requires canonical roots`；这是发布路径错误，后续是独立、CPU-only 的发布恢复，'
        '没有假称原 watcher 成功，也没有重训、重评 TEST/LIB 或重选权重。[结果][r109]、'
        '[恢复认证][r109recovery]。\n\n'
        '**R112：** 两臂原 dispatch 仍为 failed，原 train/TEST/timing/LIB 的 rc0 与 report rc1 均保留；'
        'CPU 报告恢复成功、联合分析已保存，直接复用指定发布 446ac3c，不重复恢复或发布。'
        '恢复输入中两份原失败 dispatch 的 SHA 与现文件一致。[结果][r112]、[CPU 恢复][r112recovery]。\n\n'
        '**R113：** Mencius 的 portable-inode 两卡队列下 A/B/C 三个独立 train/finalize 均 rc0；'
        'worker 实际绑定 gpu03/job1465 和两张指定 RTX3090 UUID。A 从头开始；B/C 从持久化 update20160 继续，'
        'optimizer/scaler/RNG/stream 保留；未保存的 B2286、C1872 次更新是重放，不是恢复成功的历史权重。'
        'B/C 的预训练 8640 更新已在历史 4090 上完成；历史 protocol 的 gpu_type=4090 为精确续跑保留，'
        '不能据此否认新 worker 的 RTX3090 证据。新队列总状态 blocked 来自 R114 原队列分支，'
        '不能替代三个模型各自的完整完成回执。旧四槽失败 controller 与冻结副本字节相同，原 B/C worker 与 ready 中历史 SHA 相同；'
        '两个 error.json 仍为 KeyboardInterrupt 且本次核验首尾字节一致。它们更早的独立 error.json SHA 未保存，'
        '不把本次首尾核对冒充更早历史哈希认证。[结果][r113]、[主线程 handoff][handoff]。\n\n'
        '**R114：** 原 CVRP/ICAM cold receipt 仍失败，其 SHA '
        '`d6826cf0468bb946264e3af9f565b3187b0c0fa98a1f0ef87a67f1ea69c26be3` '
        '与修正前 revision 的锚点相同。首个错误是 CUDA tensor 直接交给 NumPy 的公共 native audit 接口错误，'
        '不是候选能力失败，更不是响应学习失败。revision.json 仍是 prepared_not_integrated_not_executed；'
        '它只证明准备，不证明已修正标签。实际后续执行由另一套已锁定 CVRP scheduler、保存标签数组哈希、RTX3090 worker、'
        'combined finish/analysis 及实际提交 62fc8b6 证明。仅用纠正后的完整 CVRP 替换旧 CVRP，其他15题来源保持原样；'
        '父 scheduler 仍 blocked、子 scheduler complete，合并 lock_sha256=null，execution_lock 按题映射，'
        '没有虚构一个共同执行锁。[最终结果][r114]、[原失败与准备][r114revision]。\n\n')
    text.append('## 原假说、实际实现与对照边界\n\n')
    text.append(table(['版本', '原假说与实现', '对照与不能声称的内容'], [
        ['R109', '候选条件图计算的共享/私有化；原两 view CE/比较/relative-risk/R-Drop 复合目标，非纯 CE',
         '相同 fresh 原主干及分支原型；A 2,305,033/B 10,220,553 参数，不等参数/等 FLOPs'],
        ['R112', '保留合并前后的接口条件：4/2/1 区域层级，每区域3×3×4=36个潜接口条件，共252 token',
         'A 平均后广播/B 保留条件；完整初始参数哈希和形状相同，各1,878,027参数；不是经认证的资源DP、真实子路线成本或可行性'],
        ['R113', '由 TRAIN 合成过程预训练后持续执行预测状态；最多12个片段，非 native 完整求解程序',
         '同 fresh 完整主干；A 无预训练、B 独立快照、C 反馈转移；B/C 同目标同预训练预算，A总预算不等；三臂部署均预测反馈'],
        ['R114', '原路线是候选响应函数学习；当前仅 ACQUISITION/response pilot，尚未训练函数学习器',
         '16个适用需求/TW任务，每题32个TRAIN base；同题/同family/同base的 directed-random全池交集；无TSP/ATSP新响应']]))
    text.append(
        '原假说见[20261025综合研究][review1]及[大范式研究][review2]；实现细节见各 PLAN 与锁定 protocol。'
        'R112/R113 的 batch=512、seed2、最多57600成功更新；R109 batch=1024、seed2、同更新上限。'
        '均使用冻结 TRAIN81000/VAL4500/TEST4500 每题；R84 TRAIN-only 统计及 fresh 随机资产复用，'
        '不以旧训练 checkpoint 冒充 fresh。相同 seeded task/shuffle 协议不等于独立归档了每步全部样本ID。'
        '单 seed 的描述性结果不是显著性结论；历史 R94B/R89B 只是相同数据的完整模型参考，不是新鲜等预算因果对照。\n\n'
        '**近似实现的阴性不能否定整个范式。完整响应标签也不能证明尚未实现的下游 response learning 成功或失败；'
        '不能声称三个范式已充分验证，不能声称 universal full18 improvement。** 本次没有产生任何新 NN arm 或训练计划。\n\n')
    text.append('## 七个固定模型的结果\n\n')
    text.append(
        '`checkpoints.csv` 恰好7行；每臂一个整体 checkpoint。checkpoint 文件 SHA 与锁一致，'
        'protocol 规范化 SHA 与 state_hash 一致；保存的完整 VAL history 重现原预锁选择准则：'
        'R109 最小 whole18 VAL macro_vs_sbs_pct，精确并列保留最早；R112/R113 最大 whole18 VAL macro_top1，'
        '精确并列取最小 actual regret，再取最早。没有按 TEST/LIB 重选，也没有逐题拼装。\n\n'
        '`metrics.csv` 恰好280行：每臂 VAL/TEST 各18题+ALL，以及 TSPLIB/TSP、CVRPLIB/CVRP。'
        'VAL/TEST 每题4500、ALL81000；全部采用 native int64 ind/原 FP64 winner，保留稳定候选身份和并列规则，'
        '未换成 FP32 argmin。数据版本严格为 '
        '`selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`；LIB 为 native_TSPLIB/native_CVRPLIB。'
        '候选池哈希来自每臂 protocol.candidate_pools 的 sort_keys 紧凑 JSON；八个含B题不恢复 RF/MoSES，'
        '不恢复 RELD_MTL，不合并两个 ReLD 最低成本，RELD_MOEL 原值仅命名 reld_mt。\n\n'
        'SBS 在完整 TRAIN 拟合；已从保存 TRAIN 方法均值核对每题所选 SBS。Oracle 是逐例合法池最低成本。'
        'vs_SBS=100×(mean_cost/SBS_cost−1)，vs_Oracle 同理；actual regret=100×mean((所选成本−逐例Oracle)/逐例Oracle)。'
        'ALL 的7项数值均逐题等权平均；ALL的mean_cost只作描述，不能跨任务混合raw cost作净效益证据。\n\n')
    baselines = []
    for split in ['VAL', 'TEST']:
        r = index[model_names[0][0], model_names[0][1], split, 'ALL']
        baselines.append([split, r['n'], number(r['sbs_cost']), number(r['oracle_cost'])])
    text.append(table(['ALL split', '实例数', 'TRAIN拟合SBS cost', 'Oracle cost'], baselines))
    text.append(table(['模型', 'split', 'Top1 %', 'mean_cost', 'vs_SBS %', 'vs_Oracle %', 'actual regret %'], [
        [short[v, a], s, *[number(index[v, a, s, 'ALL'][k]) for k in
                           ['top1_pct', 'mean_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct']]]
        for v, a in model_names for s in ['VAL', 'TEST']]))
    text.append('代号与完整 arm 的一一映射：' + '；'.join(short[v, a] + '=' + a for v, a in model_names) + '。\n\n')
    text.append('以下是同一组完整模型的逐题 TEST Top1%；每题全部成本、SBS、Oracle、比率和逐例 regret 同时保存在 metrics.csv，'
                'VAL逐题及两项LIB也完整列出。\n\n')
    text.append(table(['problem', *[short[k] for k in model_names]], [
        [p, *[number(index[v, a, 'TEST', p]['top1_pct'], 3) for v, a in model_names]] for p in problems]))
    pair_names = [('R109', 'B_private_late', 'A_tied_late'),
                  ('R112', 'B_interfaces', 'A_collapsed'),
                  ('R113', 'B_snapshot', 'A_scratch'),
                  ('R113', 'C_transition', 'B_snapshot')]
    pair_rows = []
    for v, current, baseline in pair_names:
        for split in ['VAL', 'TEST']:
            c, b = index[v, current, split, 'ALL'], index[v, baseline, split, 'ALL']
            pair_rows.append([short[v, current] + '−' + short[v, baseline], split,
                              *[number(float(c[k]) - float(b[k])) for k in ['top1_pct', 'mean_cost', 'actual_regret_pct']]])
    text.append(table(['完整模型对照', 'split', 'ΔTop1 pp', 'Δmean_cost', 'Δactual regret pp'], pair_rows))
    text.append(
        'R112 B 的 TEST Top1小幅增加，但 VAL Top1下降，VAL/TEST regret均更差；不构成整体联合改进。'
        'R113 C 相对B在VAL/TEST的Top1和regret均更差；B相对A仅有局部、小幅表现变化。'
        'R109 私有化在固定36组的TEST recall从0.674206%增至1.761635%，救57、丢7、产生105个新误选；'
        '同时整体TEST Top1下降0.024691pp、regret只下降约0.000123pp，VAL regret变差。'
        '固定36组的纠正/伤害与新误选不是互不重叠的全体错误分解，不能直接相减推出ALL收益。'
        'R94B 历史完整模型 TEST Top1=71.577778%、regret=0.379115%，当前七个完整模型均未同时超过这两个描述性参考值。'
        '这不是已知上限或 Bayes 最优性证据。[R109][r109]、[R112][r112]、[R113][r113]。\n\n')
    text.append('## 零选择与71%左右表现的证据边界\n\n')
    zero_counts = collections.Counter((r['experiment'], r['arm']) for r in audit['test_zero_selection_positive_native_winners'])
    text.append(table(['模型', 'TRAIN Top1 %', 'VAL Top1 %', 'TEST Top1 %', 'TEST零选择但native赢家>0的题×候选组'], [
        [short[v, a], *[number(all_rows[v, a, s]['top1_pct']) for s in ['TRAIN', 'VAL', 'TEST']], zero_counts[v, a]]
        for v, a in model_names]))
    tsp_winners = [(r['experiment'], r['arm'], r['native_winner_count'])
                   for r in audit['test_zero_selection_positive_native_winners']
                   if r['problem'] == 'TSP' and r['candidate'] == 'LEHD']
    assert len(tsp_winners) == 3 and all(n == 762 for _, _, n in tsp_winners)
    text.append(
        '这里统计的是当前各模型的**精确零选择**组，不是固定历史36组，也不是选择次数≤0.1%的近零组；各模型均覆盖同89个合法题×候选组。'
        '例如三个R113模型在TSP TEST都从未选择LEHD，但LEHD实际是762/4500例native winner。'
        '这两种频数来自selection_distribution.csv不同panel，不能把winner频率当作模型频率，更不能把真实winner系列标作100%模型准确率。'
        '已读取每臂18页同轴对照PDF；CSV逐面板计数、比例与样本数自洽，PDF的20个TEST/LIB“模型Top1=”标题与相应逐例一致率一致。\n\n'
        '这些结果只说明当前实现仍有漏选及近似71%的表现聚集。**后验argmax的合法最优策略可以对某候选零选择**：'
        '其边际winner频率非零，不保证存在让它后验冠军概率最大的可识别区域；本次没有测到真实Bayes后验。'
        '反过来，零选择也不证明无信息或必然不可学；保存的rescued/harmed和双方均错成本变化提供的是特定已实现策略的可部署变化证据。'
        'TRAIN拟合的最低平均成本SBS、条件多数winner先验、最小期望相对成本是不同决策对象；不能把Top1的小变化代替逐题决策成本。'
        '任何“可学的漏选”判断都必须保留救回、伤害及题内成本，不能由零频数推断 Bayes 最优或信息不存在。'
        '[20261025综合研究第2、3、4、6节][review1]给出了这些区分；本次没有新增最优性或信息量测量。\n\n')
    text.append('## NSS：七个同版本模型各自对照\n\n')
    text.append(
        '28项对照均有保存结果，无缺失。NSS复用R79历史权重，未在new90k重训；TSP/CVRP的NSS checkpoint分别不同，'
        '它们是两题历史参照，不能称一个新训练full18 NSS模型。TEST使用本次冻结新数据/合法池；LIB保留旧native数据。'
        '每一行当前模型均绑定本臂唯一checkpoint，不跨版本挑最佳。NSS表的R109未独列state/update字段，'
        '按current_checkpoint_sha256与该臂final lock绑定，没有杜撰额外原列。'
        'R112/R113保存的NSS对齐说明每数据集抽核3行、FP64成本按候选名称对齐；本次复用其对齐说明，未重做全量原始数据审计。\n\n')
    nss_values = []
    for v, a in model_names:
        for r in audit['nss_comparison']:
            if r['current_model'] != v + '/' + a:
                continue
            nss_values.append([short[v, a], r['dataset'], r['n'],
                               number(r['current_top1_native_pct'], 4), number(r['current_mean_cost']),
                               number(r['nss_top1_native_pct'], 4), number(r['nss_mean_cost'])])
    assert len(nss_values) == 28
    text.append(table(['模型', '数据集', 'n', '模型Top1 %', '模型mean_cost', 'NSS Top1 %', 'NSS mean_cost'], nss_values))
    text.append('## R114：完整标签，不是已训练响应学习器\n\n')
    text.append(
        '**CURRENT EXECUTION PHASE = ACQUISITION/response pilot；NOT a trained function learner。** '
        '实际权威 closeout_kind=complete_labels；9600/9600候选成本、2048/2048全池winner、1536/1536全池响应对完整。'
        '对应16题×32个TRAIN基础实例；多端点/同base关联，不能当独立样本。无TSP/ATSP新响应；'
        '没有下游NN、TEST/LIB准确率、full18模型或普遍数据改进结论。完整标签不能建立下游响应学习的成功或失败。\n\n')
    text.append(table(['problem', '端点', '实际/计划候选成本', '全池winner', '完整/计划响应对'], [
        [r['problem'], r['endpoints'], r['actual_verified_labels'] + '/' + r['planned_labels'], r['full_pool_winners'],
         r['full_pool_pairs'] + '/' + r['planned_pairs']] for r in audit['r114_counts']]))
    directed = audit['r114_directed_vs_random']
    rms_pos = sum(float(r['mean_rms_delta_difference_pp']) > 0 for r in directed)
    flip_pos = sum(float(r['winner_flip_difference_pp']) > 0 for r in directed)
    text.append(
        f'24个题×family条件均为32个同base全池配对；directed−random的RMS相对响应差{rms_pos}项正、{len(directed)-rms_pos}项负，'
        f'winner-flip差{flip_pos}项正，其余含负值及零。不是所有条件都受益。平均幅度比范围'
        f"{min(float(r['mean_amplitude_ratio']) for r in directed):.6f}–{max(float(r['mean_amplitude_ratio']) for r in directed):.6f}，"
        '所以没有严格匹配干预幅度，不能把差异都归因为“定向机制”。'
        'CVRP demand的RMS差为+0.424028pp、flip差+9.375pp；OVRPL demand的flip差−12.5pp；'
        'OVRPTW tw_center的flip差−15.625pp。响应存在、响应幅度、winner翻转和下游可泛化学习是四个不同命题。'
        '[完整结果与逐条件表][r114]；原失败不能混入候选能力或学习成败判断。\n\n')
    text.append('## 历史 R57 与运行条件\n\n')
    text.append(
        '[R57已知边界][r57]必须保留：FP64成本/原生winner对齐不等于任务语义一致；历史B任务的signed-residual-load与classical '
        'backhaul可行域不同。R57静态检查覆盖128合法题×候选，但原输入runtime只完成31/128；没有任何一行被称作完整历史认证。'
        'TorchRL ABI、部分NSS导入、RF/MoSES绑定与历史resolved配置仍有未闭合项；992个原输入成本重现不能认证未存档的历史依赖修订。'
        '当前候选过滤和ReLD别名是已冻结的版本规则，不回填旧数据，也不把所有历史标签宣判错误。\n\n'
        'R57旧VAL的R45A Top1=49.1167%，条件多数先验42.4667%，同题同size置换均值39.7110%；这是旧数据/候选版本上已存在实例信号的历史证据，'
        '不是对当前90k的71%上限估计。置换并非均匀随机排列，分位数非总体置信区间；小规模执行角色翻转也不解释整体平台。'
        '本次未重新遍历300000条路线、权重、solver或实例，更没有“修复native contract以促成R114成功”。\n\n'
        '[20261025 running_snapshot.csv][oldrunning]中R109/R111的training状态是当时快照；本次以实际final lock/complete/recovery/publication认证，'
        '不重写旧状态。R112历史训练/计时在4090、CPU恢复复用旧计时；R113历史预训练4090与本轮两3090续跑分别记录。'
        'R109 timing.json只记录cuda:0，未独列device_name，不由此编造硬件实测；R112/R113的device_name分别记4090/3090。'
        '计时只是完整模型、GPU已驻留输入、首512例/题的保存测量，3次重复、1次warmup，排除I/O、CPU→GPU传输及求解器，'
        '不是端到端耗时；不同设备/预算不可直接作架构效率因果比较。本次不访问GPU、不测时、不更改现有队列。\n\n'
        '**TEST/LIB已经参与历史研发，不是全新盲测。** 此次唯一VAL锁的重用并不消除这种研究暴露。'
        'whole18持续改进目标仍ACTIVE；当前有限核验受阻既不关闭目标，也不授权新的arm、求解或发布。\n\n')
    text.append('## 可复核来源\n\n')
    text.append(
        '每臂的`final_test_lock.json`、`protocol.json`、`final_metrics.json`、`history.json`及'
        '`full_model_statistics/{metrics_by_problem.csv,nss_comparison.csv,selection_distribution.csv,MODEL_STATISTICS.pdf,selection_distributions.pdf}`'
        '的实际绝对路径与SHA见[saved_results_authentication.json](saved_results_authentication.json)。'
        '已核验7个checkpoint字节/21份PDF；所有附加输入的核验首尾SHA相同。原始权重只计算文件SHA，没有反序列化，没有模型forward。\n\n')
    roots = {'R109': 'R109_method_conditioned_nodes', 'R112': 'R112_region_interfaces', 'R113': 'R113_executable_process'}
    text.append(table(['模型', '实际run目录'], [[short[v, a], '`' + str(REPO / 'code/V4/runs' / roots[v] / a) + '`']
                                               for v, a in model_names]))
    refs = {
        'r109': REPO / 'code/V4/runs/R109_method_conditioned_nodes/RESULTS.md',
        'r109recovery': REPO / 'code/V4/runs/R109_method_conditioned_nodes/preparation/publication_recovery_20261026/final_verification.json',
        'r112': REPO / 'code/V4/runs/R112_region_interfaces/RESULTS.md',
        'r112recovery': REPO / 'code/V4/runs/R112_region_interfaces/preparation/recovery_20261026/recovery_complete.json',
        'r113': REPO / 'code/V4/runs/R113_executable_process/RESULTS.md',
        'handoff': REPO / 'code/V4/runs/R113_executable_process/preparation/migration_3090_portable_inode_20261026/HANDOFF.md',
        'r114': REPO / 'code/V4/runs/R114_paired_responses/preparation/finish/tensor_boundary_combined_20261026/RESULTS.md',
        'r114revision': REPO / 'code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/revision.json',
        'r57': REPO / 'code/V4/runs/R57_task_signal_audit/RESULTS.md',
        'review1': REPO / 'docs/research/comprehensive_direction_review_20261025/REPORT.md',
        'review2': REPO / 'docs/research/major_paradigm_review_20261025/REPORT.md',
        'oldrunning': REPO / 'docs/research/comprehensive_direction_review_20261025/running_snapshot.csv'}
    text.append('\n'.join(f'[{key}]: {path}' for key, path in refs.items()) + '\n')
    (HERE / 'evidence_pack.md').write_text(''.join(text))
    negative = [
        {'fact': '92 CSV publication-manifest file hashes differ from the actual named commit blobs; all are CRLF-to-LF only. Strict exact-byte gate is blocked.',
         'sources': [str(HERE / 'publication_byte_verification.json')]},
        {'fact': 'R112 interfaces: VAL Top1 and VAL/TEST regret worse than collapsed; TEST Top1 alone slightly better. Not a joint whole18 improvement.', 'sources': [str(refs['r112'])]},
        {'fact': 'R113 transition C has lower VAL/TEST Top1 and higher regret than snapshot B; synthetic max12-fragment process is not a native complete program.', 'sources': [str(refs['r113'])]},
        {'fact': 'R114 old CVRP/ICAM receipt remains failed and hash-identical to the pre-correction revision anchor; original cause is CUDA-to-NumPy audit interface error, not solver capability or response-learning failure.', 'sources': [str(refs['r114revision']), str(refs['r114'])]},
        {'fact': 'R114 has complete_labels, 9600 costs/2048 full-pool winners/1536 full-pool pairs, but current phase is ACQUISITION/response pilot. No trained function learner, no TSP/ATSP new responses and no full18 downstream success/failure result.', 'sources': [str(refs['r114'])]},
        {'fact': 'R57 runtime certification was partial (31/128); cost/winner alignment did not certify common backhaul semantics or unrecorded historical dependencies.', 'sources': [str(refs['r57'])]},
        {'fact': 'Historical TEST/LIB have participated in development; single seed results do not establish Bayes optimality or an accuracy ceiling.', 'sources': [str(refs['review1']), str(refs['r112']), str(refs['r113'])]}]
    limits = {
        'R112': 'Four leaf regions, 4/2/1 hierarchy and 36 latent entry/exit/resource interface conditions per region (252 tokens); approximate port projection, not certified resource DP, physical conditional cost or feasibility. Its finite approximation cannot reject the full regional-interface paradigm.',
        'R113': 'TRAIN-only synthetic programs of at most12 fragments, not native complete solver programs. A has no pretraining; B/C share pretraining targets/budget and all deploy predicted feedback. Resource-checkpoint replay disclosed (B2286/C1872); no universal paradigm rejection or matched-total-compute claim.',
        'R114': 'CURRENT EXECUTION PHASE=ACQUISITION/response pilot, NOT a trained function learner. Covers only16 applicable demand/TW tasks,32 TRAIN bases/problem, no new TSP/ATSP responses. Complete labels cannot establish downstream response-learning success or failure, universal full18 data improvement or generalization. Related endpoints are not independent; directed/random amplitudes are not exactly matched.'}
    verification = {
        'complete': False, 'status': 'blocked', 'action_id': ACTION,
        'model': 'gpt-6.1-sol', 'reasoning_effort': 'max',
        'saved_results_only': True, 'unique_val_checkpoints': True,
        'same_version_metrics': True, 'test_development_exposed': True,
        'scientific_completion_gate_sha256': audit['scientific_completion_gate_sha256'],
        'verified_inputs': snapshot,
        'outputs': {name: digest(HERE / name) for name in ['evidence_pack.md', 'checkpoints.csv', 'metrics.csv']},
        'r114_closeout_kind': audit['r114_closeout_kind'],
        'r114_current_execution_phase': audit['r114_current_execution_phase'],
        'negative_partial_facts': negative, 'implemented_hypothesis_limits': limits,
        'strict_published_bytes_pass': False,
        'blocking_reason': '92 CSV manifest SHA256 values describe CRLF working files, while the actual published commit blobs contain LF bytes. Parsed content is identical, but exact-byte identity is false. Native publication receipts/manifests are owned by other agents and remain untouched.',
        'required_action': 'Main/publication owners must reconcile the supplied92-item dual-hash mapping in an additive versioned authoritative receipt, explicitly bind the original file and actual commit/blob hashes and line-ending transformation, and coordinate the monitor source snapshot. No retraining, solver rerun, blind recommit or repush is required.',
        'publication_byte_mismatch_count': len(publication['mismatches']),
        'publication_check': {'path': str(HERE / 'publication_byte_verification.json'), 'sha256': digest(HERE / 'publication_byte_verification.json')},
        'saved_result_check': {'path': str(HERE / 'saved_results_authentication.json'), 'sha256': digest(HERE / 'saved_results_authentication.json')},
        'counts': {'checkpoints': 7, 'metrics_rows': 280, 'source_snapshot_inputs': 135,
                   'publication_assets': 550, 'pdfs': 21, 'preserved_failure_records': 9},
        'model_identity_evidence': 'Native CLI record -m gpt-6.1-sol/model_reasoning_effort=max and live ancestry; self-reported JSON model is not independent provider-side routing proof.',
        'provider_route_verified': False, 'astra_probed': False,
        'original_artifacts_modified': False, 'gpu_used': False,
        'no_new_training_or_arms': True, 'git_mutations': False,
        'whole18_goal_remains_active': True,
        'historical_error_hash_limit': 'R113 old worker receipts and old controller have historical hash anchors; B/C error.json remain KeyboardInterrupt and are byte-identical across this read-only invocation, but no independently archived earlier error.json hash was available.',
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (HERE / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'blocked', 'made_progress': True, 'checkpoints': len(checkpoints),
                      'metrics': len(metrics), 'evidence_pack_bytes': (HERE / 'evidence_pack.md').stat().st_size,
                      'strict_publication_byte_mismatches': len(publication['mismatches'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
