"""Saved-only closeout of the four principal Return18 experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REPO = Path('/public/home/shiys/0selection2')
RUNS = REPO / 'code/V4/runs'
OUTPUT = RUNS / 'Return18_post_R120/final_results'
R121 = RUNS / 'R121_structural_families/preparation/learner_v1'
DATA_NAME = 'selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def macro_rows(path, name):
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            split = row.get('split', '').upper()
            if row.get('problem') in ('ALL', 'ALL18') and split in ('VAL', 'TEST'):
                rows.append(dict(experiment=name, split=split,
                    **{key: float(row[key]) for key in ('top1_pct', 'mean_cost', 'sbs_cost',
                        'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')}))
    if {r['split'] for r in rows} != {'VAL', 'TEST'} or len(rows) != 2:
        raise ValueError('Exactly one full18 VAL and TEST macro row required: ' + str(path))
    return rows


def collect(r121_root):
    r121_root = Path(r121_root).resolve()
    principal = r121_root / 'principal'
    delivery = read(principal / 'delivery_manifest.json')
    if delivery.get('complete') is not True or delivery.get('all18') is not True:
        raise ValueError('R121 is not delivered; never close the goal using only the other three')
    paths = dict(
        R121=principal / 'full_model_statistics/metrics_by_problem.csv',
        R122=RUNS / 'R122_hard_decision_constraints/model/full_model_statistics/metrics_by_problem.csv',
        R123=RUNS / 'R123_heterogeneous_moe/finish/reports/metrics.csv',
        R124=RUNS / 'R124_top2_reranking/launch_v2/model/finish/reports/metrics.csv')
    rows = [row for name, path in paths.items() for row in macro_rows(path, name)]
    for split in ('VAL', 'TEST'):
        current = [row for row in rows if row['split'] == split]
        for key in ('sbs_cost', 'oracle_cost'):
            if max(row[key] for row in current) - min(row[key] for row in current) > 1e-9:
                raise ValueError('Different frozen cost populations, do not pool: ' + split + '/' + key)
    publication = {}
    for name, directory in (
            ('R122', RUNS / 'R122_hard_decision_constraints'),
            ('R123', RUNS / 'R123_heterogeneous_moe'),
            ('R124', RUNS / 'R124_top2_reranking/launch_v2')):
        record = read(directory / 'github_publication.json')
        if record.get('complete') is not True or record.get('push_returncode') != 0:
            raise ValueError('Required complete historical version not published: ' + name)
        publication[name] = record['commit']
    bank_path = RUNS / 'R123_heterogeneous_moe/finish/reports/bank_decomposition.csv'
    with bank_path.open(encoding='utf-8-sig', newline='') as stream:
        bank = next(r for r in csv.DictReader(stream) if r['problem'] == 'ALL18' and r['split'] == 'test')
    locks = {name: read(path) for name, path in (
        ('R121', principal / 'final_test_lock.json'),
        ('R122', RUNS / 'R122_hard_decision_constraints/model/final_test_lock.json'),
        ('R123', RUNS / 'R123_heterogeneous_moe/model/final_test_lock.json'),
        ('R124', RUNS / 'R124_top2_reranking/launch_v2/model/final_test_lock.json'))}
    evidence = dict(all_four_complete=True, data_version=DATA_NAME, original18=True,
        macro='equal problem weights; raw mean costs descriptive only', native_winner=True,
        publications=publication, r121_publication='next one-version publication, not yet asserted',
        checkpoint_sha256={name: lock['checkpoint_sha256'] for name, lock in locks.items()},
        selected_update={name: lock['selected_update'] for name, lock in locks.items()},
        metric_sources={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()},
        r123_bank=dict(path=str(bank_path), sha256=digest(bank_path), test_all18=bank),
        R58='deferred', test_development_exposed=True, matched_ablation=False,
        model_inference=False, solver_execution=False, checkpoint_selection=False)
    return rows, evidence


def render(output, rows, evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import font_manager
    fonts = [font for font in font_manager.fontManager.ttflist
        if font.name in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei')]
    if not fonts:
        raise ValueError('A local CJK font is required for the Chinese closeout PDF')
    plt.rcParams.update({'font.family': fonts[0].name, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    values = {(r['experiment'], r['split']): r for r in rows}
    bank = evidence['r123_bank']['test_all18']
    path = output / 'RESULTS.pdf'
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .92, '原18题：四条主要路线的完整结果', fontsize=22)
        fig.text(.06, .87, '相同冻结划分与候选池；每个版本只有一个 validation 选定的 checkpoint。', fontsize=11)
        axis = fig.add_axes([.06, .39, .88, .39]); axis.axis('off')
        headings = ['版本', 'VAL Top1 %', 'TEST Top1 %', 'TEST regret %', 'TEST mean cost']
        table = axis.table(cellText=[[name,
            f"{values[name, 'VAL']['top1_pct']:.4f}", f"{values[name, 'TEST']['top1_pct']:.4f}",
            f"{values[name, 'TEST']['actual_regret_pct']:.6f}", f"{values[name, 'TEST']['mean_cost']:.9f}"]
            for name in ('R121', 'R122', 'R123', 'R124')], colLabels=headings,
            cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(12)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor('#d3dce0')
            cell.set_facecolor('#14525a' if row == 0 else '#edf3f4' if row % 2 else 'white')
            if row == 0: cell.get_text().set_color('white')
        notes = [
            'R121：自然胜负 CE + 完整方法优势及结构族响应；R122：硬决策约束及固定漏选群体。',
            'R123：冻结历史完整模型之间路由；R124：独立的完整18题 Top2 精排。',
            '这不是更新数、训练人口严格匹配的消融表；不能据小差值宣布机制胜负。',
            '所有单版本完整任务、SBS、Oracle、NSS及LIB明细见各自正式PDF/CSV。',
            'TEST已参与历史研发，不称盲测；无TEST调温度、挑轮次或逐题拼接。',
            'R58仍延后：本轮不认证所有历史B部署属于同一规范可行域。']
        for i, text in enumerate(notes): fig.text(.06, .32-i*.037, text, fontsize=10)
        fig.text(.93, .035, '1 / 2', ha='right', fontsize=10)
        pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .92, '解释范围：不要把局部失败当成任务上限', fontsize=21)
        lines = [
            'R122：预设硬约束未达成，不能据此证明共同改进在数学上不可行。',
            f"R123：三个固定模型的 TEST Top1 联合覆盖仅 {float(bank['bank_native_coverage_pct']):.4f}%。",
            f"即使知道答案后总能选对专家，仍有 {float(bank['native_bank_miss_pp']):.4f} 个百分点错误不在硬路由动作支持内。",
            f"硬bank离线质量下界为 {float(bank['bank_quality_floor_pct']):.6f}% regret；它不是可部署成绩。",
            '这些界只约束这三个固定专家，不约束新模型、软混合或整个问题的可学性。',
            'R124：Top2召回空间大，但需要事前辨别；纠正与误伤必须同时报告。',
            'R121：优势/响应误差应对照零势函数和TRAIN均值，而不是只看损失下降。',
            '响应预测改善不自动等于自然实例选择改善；也不能把重复端点当独立新样本。',
            '固定漏选清单应比较真实命中、误选和成本；选择次数非零不等于问题已解决。',
            '下一步科学解释参考Astra的完整模型意见、凸可达性与行动风险报告；不自动启动新版本。',
            '当前比较以四项实际交付为基础，不将准备、排队或CPU fixture当训练成果。']
        for i, text in enumerate(lines): fig.text(.06, .83-i*.057, text, fontsize=11)
        fig.text(.06, .12, '数据：'+DATA_NAME, fontsize=8)
        fig.text(.06, .08, 'actual regret = 逐实例相对Oracle差再取任务宏平均；不是 mean cost 的均值比。', fontsize=10)
        fig.text(.93, .035, '2 / 2', ha='right', fontsize=10)
        pdf.savefig(fig); plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r121-root', type=Path, default=R121)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    rows, evidence = collect(args.r121_root)
    output = args.output_root.resolve()
    if output != OUTPUT:
        raise ValueError('Use the one owned saved-results closeout directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    pdf = render(output, rows, evidence)
    evidence['artifacts'] = {p.name: dict(path=str(p), sha256=digest(p)) for p in (pdf, output / 'comparison.csv')}
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(complete=True, output=str(output), all_four_complete=True,
        model_inference=False, solver_execution=False, goal_complete_not_asserted=True)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
