"""Small CPU-only conditional permutation of completed locked VAL predictions.

No training, checkpoint loading, GPU, TEST/LIB, solver or model rule changes.
Same-problem/exact-size permutations are shared across the three fixed models.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / 'code/V4/runs'
ROOT = RUNS / 'R103_conditional_signal_map'
DATA = '/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018'
MODELS = dict(R89B='R89_optimizer_geometry/B_muon_hybrid',
              R94A='R94_pretrained_node_fusion/A', R94B='R94_pretrained_node_fusion/B')
FIELDS = ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct',
          'vs_oracle_pct', 'actual_regret_pct')
IDENTITY = ('indices', 'source_indices', 'release_source_indices', 'nodes', 'pool', 'pool_ids', 'costs', 'winner')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_argmax(scores, ids):
    if scores.ndim != 2 or ids.shape != (scores.shape[1],) or not np.isfinite(scores).all():
        raise ValueError('Require finite complete legal score vectors and stable global IDs')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Global solver IDs must be unique')
    order = np.argsort(ids, kind='stable')
    return order[np.argmax(scores[:, order], axis=1)]


def grouped_permutations(nodes, repeats, seed):
    nodes = np.asarray(nodes)
    if nodes.ndim != 1 or nodes.dtype.kind not in 'iu' or (nodes < 1).any():
        raise ValueError('Require original positive integer real sizes, no inferred size bins')
    groups = [np.flatnonzero(nodes == size) for size in np.unique(nodes)]
    rng = np.random.default_rng(seed)
    permutations = np.tile(np.arange(len(nodes)), (repeats, 1))
    for row in permutations:
        for group in groups:
            if len(group) >= 2:
                row[group] = rng.permutation(group)
    eligible = sum(len(group) for group in groups if len(group) >= 2)
    return permutations, dict(n=len(nodes), groups=len(groups),
        singleton_groups=sum(len(group) == 1 for group in groups),
        eligible_rows=eligible, eligible_pct=100 * eligible / len(nodes),
        actually_moved_pct_mean=float(100 * (permutations != np.arange(len(nodes))).mean()))


def row_metrics(costs, winner, pred, sbs_slot, target_slots):
    n = len(winner)
    chosen = costs[np.arange(n), pred]
    oracle = costs.min(1)
    sbs = costs[:, sbs_slot].mean()
    target_truth = np.isin(winner, target_slots)
    target_pick = np.isin(pred, target_slots)
    correct = pred == winner
    return dict(n=n, top1_pct=float(100 * correct.mean()), mean_cost=float(chosen.mean()),
        sbs_cost=float(sbs), oracle_cost=float(oracle.mean()),
        vs_sbs_pct=float(100 * (chosen.mean() / sbs - 1)),
        vs_oracle_pct=float(100 * (chosen.mean() / oracle.mean() - 1)),
        actual_regret_pct=float(100 * ((chosen - oracle) / oracle).mean()),
        target_winners=int(target_truth.sum()), target_correct=int((correct & target_truth).sum()),
        target_selected=int(target_pick.sum()), target_false_picks=int((target_pick & ~correct).sum()),
        target_regret_contribution_pp=float(100 * (((chosen - oracle) / oracle) * target_truth).mean()))


def macro(rows):
    if len(rows) != 18:
        raise ValueError('ALL requires all18 problems, never an available-task subset')
    result = {field: float(np.mean([row[field] for row in rows])) for field in FIELDS}
    result.update(n=sum(row['n'] for row in rows))
    for key in ('target_winners', 'target_correct', 'target_selected', 'target_false_picks'):
        result[key] = sum(row[key] for row in rows)
    result['target_regret_contribution_pp'] = float(np.mean([row['target_regret_contribution_pp'] for row in rows]))
    return result


def summarize(samples):
    groups = {}
    for row in samples:
        groups.setdefault((row['model'], row['problem']), []).append(row)
    rows = []
    for (model, problem), values in groups.items():
        observed = next(row for row in values if row['mode'] == 'observed')
        permuted = [row for row in values if row['mode'] == 'permuted']
        result = dict(model=model, problem=problem, n=observed['n'], repeats=len(permuted))
        for field in (*FIELDS, 'target_correct', 'target_selected', 'target_false_picks', 'target_regret_contribution_pp'):
            result['observed_' + field] = observed[field]
            result['permuted_mean_' + field] = float(np.mean([row[field] for row in permuted]))
            result['permuted_sd_' + field] = float(np.std([row[field] for row in permuted], ddof=1))
        result.update(target_winners=observed['target_winners'],
            instance_top1_lift_pp=result['observed_top1_pct'] - result['permuted_mean_top1_pct'],
            instance_regret_benefit_pp=result['permuted_mean_actual_regret_pct'] - result['observed_actual_regret_pct'],
            instance_cost_benefit=result['permuted_mean_mean_cost'] - result['observed_mean_cost'])
        rows.append(result)
    return rows


def matched_comparisons(summary):
    index = {(row['model'], row['problem']): row for row in summary}
    rows = []
    for problem in dict.fromkeys(row['problem'] for row in summary):
        for reference in ('R94A', 'R89B'):
            current, old = index['R94B', problem], index[reference, problem]
            row = dict(candidate='R94B', reference=reference, problem=problem,
                observed_top1_delta_pp=current['observed_top1_pct'] - old['observed_top1_pct'],
                permuted_mean_top1_delta_pp=current['permuted_mean_top1_pct'] - old['permuted_mean_top1_pct'],
                observed_regret_delta_pp=current['observed_actual_regret_pct'] - old['observed_actual_regret_pct'],
                permuted_mean_regret_delta_pp=current['permuted_mean_actual_regret_pct'] - old['permuted_mean_actual_regret_pct'])
            row['additional_instance_top1_lift_pp'] = row['observed_top1_delta_pp'] - row['permuted_mean_top1_delta_pp']
            row['additional_instance_regret_benefit_pp'] = old['instance_regret_benefit_pp'] * -1 + current['instance_regret_benefit_pp']
            rows.append(row)
    return rows


def render(root, summary, coverage, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties, fontManager
    font_path = '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc'
    fontManager.addfont(font_path)
    font = FontProperties(fname=font_path)
    plt.rcParams.update({'font.family': font.get_name(), 'pdf.fonttype': 42, 'axes.unicode_minus': False})
    overall = [row for row in summary if row['problem'] == 'ALL']
    texts = [
        'R103 | 模型是否只记住了问题与规模？',
        '完整 validation：18题各4,500例；同问题、同精确规模置换20次；不训练、不读TEST/LIB',
        '',
        '模型          原对应Top1       置换Top1       差值(pp)        原regret           置换regret',
    ]
    for row in overall:
        texts.append(f"{row['model']:8s}    {row['observed_top1_pct']:8.4f}%      {row['permuted_mean_top1_pct']:8.4f}%     {row['instance_top1_lift_pp']:+7.4f}      {row['observed_actual_regret_pct']:.6f}%       {row['permuted_mean_actual_regret_pct']:.6f}%")
    texts += ['', '固定36组漏选清单：正确数 / 真实winner数；各模型目标方法选择次数在置换前后不变']
    for row in overall:
        texts.append(f"{row['model']}: 原对应正确{row['observed_target_correct']:,.0f}/{row['target_winners']:,.0f}，置换均值{row['permuted_mean_target_correct']:.2f}；误选数：原{row['observed_target_false_picks']:,.0f} / 置换{row['permuted_mean_target_false_picks']:.2f}")
    texts += ['', 'R94B 相对匹配的随机表示对照 R94A']
    pair = next(row for row in comparisons if row['problem'] == 'ALL' and row['reference'] == 'R94A')
    texts += [f"原对应Top1差：{pair['observed_top1_delta_pp']:+.6f} 个百分点",
              f"置换后Top1差：{pair['permuted_mean_top1_delta_pp']:+.6f} 个百分点",
              f"两模型实例对应增益之差：{pair['additional_instance_top1_lift_pp']:+.6f} 个百分点（不是新增可部署成绩）",
              '', '结论边界',
              '确有超出问题与精确规模的实例关联；没有控制分布、容量类型等其它因素。',
              '固定36组仍只正确126/4,625例；不能把机制信号当作已经解决漏选。',
              '置换仅为离线诊断；各组方法选择次数完全保留，单例组不放宽规模。',
              '均匀置换允许随机固定点；重复标准差不是置信区间或多个训练seed证据。',
              '这不是预训练的因果效应、准确率上限或新的模型接收判断。',
              '不重选checkpoint，不改标签、推理规则或四卡任务；后续仍看锁定模型正式结果。',
              f"可置换覆盖：{sum(row['eligible_rows'] for row in coverage):,.0f}/81,000例。",
              'native/FP64 ind；ALL为18题等权；SBS在TRAIN拟合；regret为逐实例相对Oracle损失。',
              '历史TEST已参与研发；本轮仅读取已保存的validation数组。']
    with PdfPages(root / 'CONDITIONAL_SIGNAL.pdf') as pdf:
        fig = plt.figure(figsize=(13, 9))
        fig.text(.045, .955, '\n'.join(texts), va='top', fontsize=10.5, fontproperties=font, linespacing=1.55)
        pdf.savefig(fig)
        fig.savefig(root / 'preview_page1.png', dpi=100)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 9))
        problems = [row['problem'] for row in coverage]
        colors = dict(R89B='#55657a', R94A='#c58a41', R94B='#137d83')
        index = {(row['model'], row['problem']): row for row in summary}
        y = np.arange(len(problems))
        for offset, (model, color) in zip((-.23, 0, .23), colors.items()):
            ax.barh(y + offset, [index[model, p]['instance_top1_lift_pp'] for p in problems],
                    height=.22, color=color, label=model)
        ax.set_yticks(y)
        ax.set_yticklabels(problems)
        ax.invert_yaxis()
        ax.axvline(0, color='#65717a', lw=.8)
        ax.set_xlabel('原对应 Top1 减同问题、同精确规模置换 Top1（百分点）')
        ax.set_title('各问题的实例对应增益 | 仅 validation，不是新模型成绩')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def run(root=ROOT):
    root = Path(root).resolve()
    if (root / 'complete.json').exists():
        raise RuntimeError('Preserve the completed diagnostic; no implicit rerun/overwrite')
    protocol = load_json(root / 'protocol_lock.json')
    if (protocol.get('experiment') != 'R103' or protocol.get('locked') is not True
            or protocol.get('models') != list(MODELS) or protocol.get('repeats') != 20
            or protocol.get('seed') != 2 or protocol.get('split') != 'val'):
        raise ValueError('Use the fixed CPU VAL-only diagnostic protocol')
    targets_path = RUNS / 'R84_new90k_attention/comparison_artifacts/target_list.json'
    targets = load_json(targets_path)['targets']
    if len(targets) != 36:
        raise ValueError('Use the existing fixed36 list, not a newly chosen target set')
    controls, archives, sources = {}, {}, []
    samples, coverages, summaries = [], [], []
    with ExitStack() as stack:
        for model, relative in MODELS.items():
            run_root = RUNS / relative
            config = load_json(run_root / 'protocol.json')
            lock = load_json(run_root / 'final_test_lock.json')
            if (config['data_root'] != DATA or config['rows_per_problem'] != dict(train=81000, val=4500, test=4500)
                    or not lock['selected_without_test'] or lock['per_problem_assembly']):
                raise ValueError('Require original same-release whole-model VAL lock')
            controls[model] = config
            path = run_root / 'evaluation/val/selected_best/rank000.npz'
            stat = path.stat()
            sources.append(dict(model=model, path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'],
                lock_sha256=digest(run_root / 'final_test_lock.json')))
            archives[model] = stack.enter_context(np.load(path, allow_pickle=False))
        problems = list(controls['R89B']['candidate_pools'])
        if len(problems) != 18:
            raise ValueError('Require all18 problems')
        for problem_i, problem in enumerate(problems):
            data = {}
            for model, archive in archives.items():
                current = {field: archive[problem + '__' + field] for field in IDENTITY}
                scores = archive[problem + '__logits']
                current['pred'] = archive[problem + '__pred']
                if (not np.array_equal(current['pred'], stable_argmax(scores, current['pool_ids']))
                        or current['costs'].dtype != np.float64 or scores.shape != current['costs'].shape
                        or scores.shape[0] != 4500 or not np.isfinite(current['costs']).all()
                        or (current['costs'] <= 0).any()
                        or not np.array_equal(current['indices'], np.arange(4500))
                        or not np.array_equal(current['costs'][np.arange(4500), current['winner']], current['costs'].min(1))):
                    raise ValueError('Invalid native/FP64 VAL or saved complete-score rule: ' + model + '/' + problem)
                data[model] = current
                if model != 'R89B' and any(not np.array_equal(current[k], data['R89B'][k]) for k in IDENTITY):
                    raise ValueError('Different row/node/candidate/cost identities')
                if controls[model]['baselines'][problem] != controls['R89B']['baselines'][problem]:
                    raise ValueError('Different frozen TRAIN SBS')
            common = data['R89B']
            permutations, info = grouped_permutations(common['nodes'], 20, 2 + problem_i)
            coverages.append(dict(problem=problem, **info))
            slots = [common['pool'].tolist().index(t['method']) for t in targets if t['problem'] == problem]
            sbs_slot = controls['R89B']['baselines'][problem]['sbs_slot']
            for model, current in data.items():
                for repeat in range(-1, 20):
                    pred = current['pred'] if repeat == -1 else current['pred'][permutations[repeat]]
                    row = row_metrics(common['costs'], common['winner'], pred, sbs_slot, slots)
                    samples.append(dict(model=model, problem=problem, mode='observed' if repeat == -1 else 'permuted', repeat=repeat, **row))
        for model in MODELS:
            for repeat in range(-1, 20):
                rows = [row for row in samples if row['model'] == model and row['repeat'] == repeat]
                samples.append(dict(model=model, problem='ALL', mode='observed' if repeat == -1 else 'permuted', repeat=repeat, **macro(rows)))
    for source in sources:
        stat = Path(source['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source['bytes'], source['mtime_ns']):
            raise ValueError('A completed source changed while reading it')
    summary = summarize(samples)
    comparisons = matched_comparisons(summary)
    for model, relative in MODELS.items():
        with (RUNS / relative / 'full_model_statistics/metrics_by_problem.csv').open(encoding='utf-8-sig', newline='') as f:
            saved = {row['problem']: row for row in csv.DictReader(f) if row['split'].upper() == 'VAL'}
        for row in (row for row in summary if row['model'] == model):
            for field in FIELDS:
                if not math.isclose(row['observed_' + field], float(saved[row['problem']][field]), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError('Observed metrics must reproduce existing locked CSV: ' + model + '/' + row['problem'] + '/' + field)
    save_csv(root / 'metric_samples.csv', samples)
    save_csv(root / 'summary.csv', summary)
    save_csv(root / 'matched_comparisons.csv', comparisons)
    save_csv(root / 'group_coverage.csv', coverages)
    render(root, summary, coverages, comparisons)
    overall = [row for row in summary if row['problem'] == 'ALL']
    payload = dict(experiment='R103', complete=True, kind='VAL_only_CPU_diagnostic',
        protocol_sha256=digest(root / 'protocol_lock.json'), script_sha256=digest(__file__),
        created_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        target_list_sha256=digest(targets_path), rows_per_problem=4500,
        summary_ALL=overall, goal_complete=False, TEST_LIB_read=False, GPU_or_training=False)
    (root / 'complete.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(complete=True, report=str(root / 'CONDITIONAL_SIGNAL.pdf'), ALL=overall)), flush=True)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    run(args.root)


if __name__ == '__main__':
    main()
