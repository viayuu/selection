"""Summarize the completed R89 run using saved metrics only, without inference."""

import csv
import json
from pathlib import Path


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R89_optimizer_geometry'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def selected(run):
    lock = read(run / 'final_test_lock.json')
    history = read(run / 'history.json')
    val = next(row['val']['macro'] for row in history
               if row['successful_updates'] == lock['selected_update'])
    return lock, val, read(run / 'test_results.json')['result']['macro']


def main():
    names = [('R84A', V4 / 'runs/R84_new90k_attention/A_baseline'),
             ('R89A AdamW', ROOT / 'A_adamw'), ('R89B Muon', ROOT / 'B_muon_hybrid')]
    records = [(name, *selected(run)) for name, run in names]
    table = ['| 模型 | 锁定更新 | VAL Top1 | VAL regret | TEST Top1 | TEST regret |',
             '| --- | --- | --- | --- | --- | --- |']
    for name, lock, val, test in records:
        table.append(f"| {name} | {lock['selected_update']} | {100*val['macro_top1']:.4f}% | "
                     f"{val['macro_actual_regret_pct']:.6f}% | {100*test['macro_top1']:.4f}% | "
                     f"{test['macro_actual_regret_pct']:.6f}% |")
    recovery = rows(ROOT / 'B_muon_hybrid/full_model_statistics/target_recovery_summary.csv')
    a, b = records[1], records[2]
    gate = (b[2]['macro_top1'] > a[2]['macro_top1']
            and b[2]['macro_mean_cost'] <= a[2]['macro_mean_cost']
            and b[2]['macro_actual_regret_pct'] <= a[2]['macro_actual_regret_pct'])
    r84 = records[0]
    gate_r84 = (b[2]['macro_top1'] > r84[2]['macro_top1']
                and b[2]['macro_mean_cost'] <= r84[2]['macro_mean_cost']
                and b[2]['macro_actual_regret_pct'] <= r84[2]['macro_actual_regret_pct'])
    target = ['| 划分 | 固定清单 winner | 正确选择 | 救回 / 丢失 | 新增误选 | 召回率 |',
              '| --- | --- | --- | --- | --- | --- |']
    for row in recovery:
        target.append(f"| {row['split']} | {row['winner_count']} | {row['correct_count']} | "
                      f"{row['rescued']} / {row['lost']} | {row['new_false_pick_count']} | "
                      f"{float(row['recall_pct']):.4f}% |")
    text = '\n'.join([
        '# R89：完整18任务优化器对照', '',
        '两组均完成57,600次成功更新，按 validation 锁定唯一 checkpoint，'
        '完整 TEST、LIB、方法分布 PDF 与两组比较均已成功交付。', '',
        *table, '',
        f'B 相对匹配 A 的预设 VAL 三指标筛选通过：{gate}；'
        f'相对历史 R84A 的同方向筛选通过：{gate_r84}。'
        '这是单 seed、同一数据版本的较小改善，不代表漏选问题已解决。', '',
        f"TEST 相对 R84A 的 Top1 增加 {100*(b[3]['macro_top1']-r84[3]['macro_top1']):.4f} 个百分点；"
        f"actual regret 相对下降 {100*(r84[3]['macro_actual_regret_pct']-b[3]['macro_actual_regret_pct'])/r84[3]['macro_actual_regret_pct']:.3f}%。"
        'TEST 不参与本次权重或规则选取。', '',
        '## 固定36组漏选清单', '', *target, '',
        '以上救回、丢失与新增误选相对固定 R84A 计算。TEST 召回仍不足1%，'
        '新增选择不能全部当作正确恢复；未满足主要漏选问题的改善目标。', '',
        '## 证据边界', '',
        'A/B 的输入、模型、自然采样、完整 CE/比较/risk/R-Drop 与更新预算相同；'
        'B 只对预先指定的隐藏线性矩阵使用 Muon，其余参数使用 AdamW。'
        '没有改成仅 ReLD 二分类，也没有重新运行 solver、生成标签或恢复已删除候选。', '',
        '原始运行于2026-10-20结束，本摘要于2026-10-22补交。'
        'ALL 为18问题等权；成本、TRAIN-fitted SBS、Oracle、逐问题指标及 NSS 四数据集比较'
        '见各组 full_model_statistics/MODEL_STATISTICS.pdf。'
        '旧 TEST 已参与历轮研发，不能宣称盲测。', '',
        '后续优先检验有明确差异的大方向，不因该微小收益自动扩展一串优化器参数搜索。',
    ])
    (ROOT / 'RESULTS.md').write_text(text + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
    with PdfPages(ROOT / 'RESULTS.pdf') as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.05, .94, 'R89：完整18任务 AdamW / Muon 对照', fontproperties=font, fontsize=19)
        headers = ['模型', '锁定更新', 'VAL Top1', 'VAL regret', 'TEST Top1', 'TEST regret']
        cells = [[name, str(lock['selected_update']), f"{100*val['macro_top1']:.4f}%",
                  f"{val['macro_actual_regret_pct']:.6f}%", f"{100*test['macro_top1']:.4f}%",
                  f"{test['macro_actual_regret_pct']:.6f}%"] for name, lock, val, test in records]
        ax = fig.add_axes([.05, .57, .9, .29]); ax.axis('off')
        tab = ax.table(cellText=cells, colLabels=headers, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        for cell in tab.get_celld().values():
            cell.get_text().set_fontproperties(font); cell.get_text().set_fontsize(11)
        fig.text(.05, .50, 'B 通过 VAL 联合筛选；TEST 仅 +0.1840 个百分点，regret 相对下降约1.32%。',
                 fontproperties=font, fontsize=12)
        fig.text(.05, .44, '固定漏选清单：B 在 TEST 救回34、丢失2、新增误选42，真实 winner 召回0.9787%。',
                 fontproperties=font, fontsize=12)
        fig.text(.05, .38, '结论：存在小幅优化信号，但没有解决零／近零方法漏选，goal 仍未完成。',
                 fontproperties=font, fontsize=12)
        fig.text(.05, .29, '同一新90k划分、完整18任务、自然候选池、seed2、batch1024；每组唯一 VAL 锁定权重。',
                 fontproperties=font, fontsize=10)
        fig.text(.05, .23, '完整 TEST/LIB、NSS 对照、SBS/Oracle、并列方法分布图均在各组 full_model_statistics/。',
                 fontproperties=font, fontsize=10)
        fig.text(.05, .17, '不将微小变化视为突破，不使用 TEST 调参。旧 TEST 已参与研发，不宣称盲测。',
                 fontproperties=font, fontsize=10)
        fig.text(.05, .08, '运行结束：2026-10-20；补充摘要：2026-10-22。仅重用保存指标，无训练或推理。',
                 fontproperties=font, fontsize=9)
        pdf.savefig(fig); plt.close(fig)
    print(ROOT / 'RESULTS.pdf')


if __name__ == '__main__':
    main()
