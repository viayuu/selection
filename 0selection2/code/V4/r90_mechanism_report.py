"""CPU-only PDF of existing rank summaries; not a new trained R90 model."""

import argparse
import csv
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R90_training_mechanism_notes'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def pick(values, **fields):
    found = [value for value in values if all(value[key] == item for key, item in fields.items())]
    if len(found) != 1:
        raise ValueError('Expected one saved summary row: ' + repr(fields))
    return found[0]


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages

    font = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
        'font.size': 10, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    summaries = rows(root / 'winner_rank_summary.csv')
    distribution = rows(root / 'winner_rank_distribution.csv')
    output = root / 'MECHANISM_NOTES.pdf'

    def page(title, subtitle, number):
        figure = plt.figure(figsize=(11.7, 8.3))
        figure.text(.055, .945, title, fontsize=20, color='#064b50', weight='bold')
        figure.text(.055, .902, subtitle, fontsize=10, color='#40545a')
        figure.text(.055, .035, 'R90 机制梳理 | 未启动新训练 | 研究目标仍未解决', color='#40545a', fontsize=9)
        figure.text(.94, .035, str(number), ha='right', fontsize=9)
        return figure

    def table(figure, headers, values, box, widths):
        axis = figure.add_axes(box)
        axis.axis('off')
        item = axis.table(cellText=values, colLabels=headers, cellLoc='center',
            colWidths=widths, bbox=[0, 0, 1, 1])
        item.auto_set_font_size(False)
        item.set_fontsize(10)
        for (row, _), cell in item.get_celld().items():
            cell.set_edgecolor('#d7e2e5')
            cell.set_facecolor('#dbe9ed' if row == 0 else ('#f4f7f9' if row % 2 == 0 else 'white'))
        return item

    with PdfPages(output, metadata={'Title': 'R90 training mechanisms: evidence, not a new model'}) as pdf:
        figure = page('漏选发生在哪里：先界定机制覆盖范围',
            'R84A 唯一锁定权重 update37440；NEW90k；TRAIN/VAL 原生 FP64 winner；无新推理或求解', 1)
        data = []
        for split in ('train', 'val'):
            for scope, label in (('ALL_ERRORS', '全部错例'), ('FIXED_36_TARGETS', '固定36目标')):
                value = pick(summaries, split=split, scope=scope, problem='ALL', method='')
                missed, deep = int(value['missed']), int(value['rank_ge3'])
                data.append([split.upper(), label, f'{missed:,}', f'{missed-deep:,}',
                    f'{deep:,}', f"{float(value['rank_ge3_pct_of_missed']):.2f}%",
                    f"{float(value['rank_ge3_pct_of_all_regret']):.2f}%"])
        table(figure, ['划分', '范围', '漏选', '第2名', '第3名及以后', '深位/漏选', '深位/全体regret'],
            data, [.055, .56, .89, .27], [.10, .17, .13, .13, .13, .12, .22])
        figure.text(.06, .49, '固定目标中，约一半漏选仍排在第2名，而非全部被压到深位。', fontsize=14, color='#064b50')
        figure.text(.06, .425,
            '深位指预测名次 >=3，包含第3名。固定目标的 VAL 深位漏选占全体 regret 的7.99%。\n'
            '这只是当前错误的范围分解，\n'
            '不是某种新训练能够恢复的收益，更不是准确率上限。\n\n'
            'maximin 与 utility 保留同一排序，但 softmax、CE 梯度不同。真实 winner 不在\n'
            '前两名时，错误次名也收到上推梯度；真实 winner 相对次名的局部差距仍在改善。\n'
            '因此，“存在梯度耦合”不能直接写成“梯度有害”或“已找到共同根因”。',
            va='top', linespacing=1.8, fontsize=11)
        figure.text(.06, .105,
            '来源：winner_rank_summary.csv、FACTS.md；ALL regret 逐题等权。固定目标不包含任意误选为该方法的实例。\n'
            '历史 R64/R65 已实现 utility CE、多任务同一步梯度；先读已有结果，不将两者当作全新方案直接重跑。',
            fontsize=9, linespacing=1.6)
        pdf.savefig(figure)
        plt.close(figure)

        figure = page('三个已有支持的漏选方法：预测名次并不相同',
            '仅显示 VAL 真实 winner 属于该方法但未选中的实例；不是选择频率图，也不是 TEST 结果', 2)
        for index, (problem, method) in enumerate((('TSP', 'LEHD'), ('TSP', 'DIFUSCO'), ('ATSP', 'MATPOENET'))):
            summary = pick(summaries, split='val', scope='FIXED_TARGET', problem=problem, method=method)
            overall = pick(summaries, split='val', scope='ALL_ERRORS', problem=problem, method='')
            items = [item for item in distribution if item['split'] == 'val'
                and item['scope'] == 'FIXED_TARGET' and item['problem'] == problem and item['method'] == method]
            items.sort(key=lambda item: int(item['predicted_winner_rank']))
            items = [item for item in items if int(item['predicted_winner_rank']) > 1]
            axis = figure.add_axes([.10 + .30*index, .35, .235, .43])
            ranks = [int(item['predicted_winner_rank']) for item in items]
            shares = [float(item['pct_of_missed']) for item in items]
            axis.barh(range(len(items)), shares, color='#187d86', height=.64)
            axis.set_yticks(range(len(items)), [f'第{rank}名' for rank in ranks])
            axis.invert_yaxis()
            axis.set_xlim(0, 115)
            axis.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
            axis.set_xlabel('占该方法漏选实例的比例')
            axis.grid(axis='x', color='#e0e7e9')
            axis.set_axisbelow(True)
            for spine in ('top', 'right', 'left'):
                axis.spines[spine].set_visible(False)
            for row, (item, share) in enumerate(zip(items, shares)):
                axis.text(share+1, row, f"{share:.1f}% ({int(item['count']):,})", va='center', fontsize=9)
            top1 = 100*int(overall['correct'])/int(overall['true_winner_rows'])
            axis.set_title(f'{problem} / {method}\n模型Top1={top1:.2f}%\n漏选 n={int(summary["missed"]):,}',
                fontsize=11, pad=14)
        figure.text(.06, .215,
            '图头 Top1 是该问题全部4,500个 VAL 实例的模型准确率，不是该方法选择率或条件召回率。\n'
            'LEHD 的漏选既有第2名，也有更深位置。不同方法须分别观察，不能仅用零选择组数归纳。\n'
            '原 NSS 也出现零选择，但不共享当前 maximin/18任务训练链；这限制了共同根因解释。',
            va='top', fontsize=11, linespacing=1.8)
        pdf.savefig(figure)
        plt.close(figure)

        history = rows(root / 'historical_results.csv')
        figure = page('历史实现已经完成过：不能把重合机制当成新发现',
            'R64/R65：旧10k划分、旧完整候选池；R75：含旧10k的不均衡扩容；均非当前 NEW90k', 3)
        data = []
        for model in ('R64A', 'R64B', 'R65A', 'R65B', 'R75'):
            validation = pick(history, model=model, split='val')
            test = pick(history, model=model, split='test')
            data.append([model, f"{float(validation['top1_macro_pct']):.4f}",
                f"{float(test['top1_macro_pct']):.4f}",
                f"{float(validation['actual_regret_macro_pct']):.6f}",
                f"{float(test['actual_regret_macro_pct']):.6f}"])
        table(figure, ['旧模型', 'VAL Top1 %', 'TEST Top1 %', 'VAL regret %', 'TEST regret %'],
            data, [.055, .565, .89, .27], [.12, .22, .22, .22, .22])
        figure.text(.06, .505,
            '三版均为 raw-utility CE + R-Drop，并在同一参数状态汇总三个任务梯度。',
            fontsize=12, color='#064b50')
        figure.text(.06, .465,
            'R64 对照 PCGrad 与梯度均值；R65 对照 AdamW 与 ScheduleFree。\n'
            '它们不是当前数据、完整 compare/risk 目标下的严格 CE 接入点或任务混合消融。\n'
            '直接 utility CE 和多任务同一步更新，在这些旧配方中没有自动消除零选择。',
            fontsize=11, va='top', linespacing=1.8)
        zero_data = []
        for model in ('R65A', 'R65B', 'R75'):
            zero_data.append([model] + [pick(history, model=model, split=split)[
                'zero_selected_positive_saved_method_top1_groups'] for split in ('train', 'val', 'test')])
        table(figure, ['旧模型', 'TRAIN 零选择组', 'VAL 零选择组', 'TEST 零选择组'],
            zero_data, [.055, .225, .89, .14], [.16, .28, .28, .28])
        figure.text(.06, .17,
            '零组定义沿用各版已有 CSV：pick_fraction=0 且 method_top1>0；不是当前固定36目标。\n'
            '这里只引用历史报告，不重新认证其标签口径；R64 缺失的零选择统计不补造。\n'
            '不据跨数据集 Top1 推断机制无效、扩容有害或准确率上限。TEST 不改变 R88/R89 选型。',
            fontsize=9, va='top', linespacing=1.6)
        figure.text(.06, .075,
            '来源：HISTORICAL_RESULTS.md、historical_results.csv；论文建议及反例见 docs/research/r90_training_mechanisms.md。',
            fontsize=8)
        pdf.savefig(figure)
        plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(render(args.root))


if __name__ == '__main__':
    main()
