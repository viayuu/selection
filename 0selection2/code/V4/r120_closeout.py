"""CPU-only Chinese closeout from immutable R120 and historical saved scores."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import textwrap
import zipfile

import numpy as np

from .r120_reporting import (
    NONROUTING_PROBLEMS, PROBLEMS, ROUTING_PROBLEMS, aggregate_metrics,
    method_rows, native_task_metrics, nss_rows,
)


REPO = Path('/public/home/shiys/0selection2')
RUNS = REPO / 'code/V4/runs'
ROOT = RUNS / 'R120_multidomain_selector'
OUTPUT = RUNS / 'Return18_post_R120/R120_closeout'
EXPECTED_SHA = 'ff7b2f8f4b176bd0dd06c2bf61dda1b3007f9510a836385e7f8a6e7056aff6ad'
DATA = '/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018'
HISTORY = {'R84A': RUNS / 'R84_new90k_attention/A_baseline',
           'R94B': RUNS / 'R94_pretrained_node_fusion/B'}
FONT = Path('/usr/share/fonts/google-droid/DroidSansFallback.ttf')
SHORT_SBS = {'MIS': 'T2T_ER', 'JSSP': 'MWKR', 'FJSP': 'DAN 10x5+mix',
             'BPP': 'MBS_prime', 'MKP': 'LPDualGreedy', 'SMTWTP': 'ATC'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def check_close(actual, expected, name):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f'{name}: missing-value disagreement')
    elif not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise ValueError(f'{name}: {actual} != saved {expected}')


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple))
                             else value for key, value in row.items()})


def collect():
    lock = read(ROOT / 'final_test_lock.json')
    config = read(ROOT / 'protocol.json')
    train = read(ROOT / 'training_complete.json')
    test = read(ROOT / 'test_complete.json')
    report = read(ROOT / 'reports/report_complete.json')
    if (lock['checkpoint_sha256'] != EXPECTED_SHA or file_hash(ROOT / 'best.pt') != EXPECTED_SHA
            or lock['selected_update'] != 44640 or lock.get('selected_without_test') is not True
            or lock.get('per_problem_assembly') is not False or not train['complete']
            or train['successful_updates'] != 53280 or train['best']['update'] != 44640):
        raise ValueError('The explicitly requested R120 checkpoint/training identity changed')
    for receipt in (test, report):
        if not receipt['complete'] or receipt['checkpoint_sha256'] != EXPECTED_SHA or receipt['selected_update'] != 44640:
            raise ValueError('R120 TEST/report completion identity changed')
    if not report['lib_complete'] or not report['complete24']:
        raise ValueError('Complete full24 TEST and both LIB reports are required')
    state_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if lock['state_hash'] != state_hash or lock['inference_rule'] != config['inference_rule']:
        raise ValueError('Locked R120 protocol/inference identity changed')
    predictions, per_task, scope, support, source_files = {}, [], [], [], []
    summaries = {}
    for split in ('val', 'test', 'TSPLIB', 'CVRPLIB'):
        directory = ROOT / 'evaluation' / split / 'selected_best'
        summary = read(directory / 'metrics.json')
        identity = summary['identity']
        if identity['checkpoint_sha256'] != EXPECTED_SHA or identity['successful_updates'] != 44640:
            raise ValueError(f'{split}: saved metrics belong to another checkpoint')
        if split in ('val', 'test') and set(summary['per_problem']) != set(PROBLEMS):
            raise ValueError('The full24 saved evaluation is incomplete')
        current = {}
        for problem, saved_metrics in summary['per_problem'].items():
            path = directory / 'by_problem' / (problem + '.npz')
            data = archive(path)
            if not np.array_equal(data['indices'], np.arange(saved_metrics['n'])):
                raise ValueError('Saved score row order is not the frozen population order')
            reference = config['references'][problem]
            if reference['selection_split'] != 'train':
                raise ValueError('SBS must be chosen on TRAIN')
            metrics, pred = native_task_metrics(data['logits'], data['native_objective'], data['observed_mask'],
                str(data['objective_direction']), all_best_mask=data['all_best_mask'],
                native_winner=data.get('native_winner_index'), reference=reference)
            if not np.array_equal(pred, data['pred']):
                raise ValueError('Saved choices disagree with the locked logits argmax')
            for key in ('n', 'native_top1_pct', 'any_best_top1_pct', 'mean_objective', 'sbs_objective',
                        'oracle_objective', 'actual_regret_pct', 'vs_sbs_pct', 'vs_oracle_pct'):
                check_close(metrics[key], saved_metrics[key], f'{split}/{problem}/{key}')
            current[problem] = metrics
            predictions[split, problem] = data, metrics
            per_task.append(dict(split=split, problem=problem, checkpoint_sha256=EXPECTED_SHA,
                                 selected_update=44640, **metrics))
            source_files.append(path)
            if split == 'test':
                for row in method_rows(problem, split, data, metrics):
                    slot = data['pool'].tolist().index(row['solver'])
                    native = data.get('native_winner_index')
                    native_support = int((native == slot).sum()) if native is not None else None
                    row.update(native_winner_support_count=native_support,
                               native_support_rule='native/FP64 ind' if native is not None else 'not defined; new-task best-set labels',
                               full_population=True)
                    if row['strict_zero_selected']:
                        row['zero_group_type'] = ('unique_positive' if row['unique_best_count'] else
                            'tied_support_only' if row['all_best_support_count'] else 'empty_best_support')
                    else:
                        row['zero_group_type'] = 'not_zero_choice'
                    support.append(row)
        summaries[split] = summary
        if split in ('val', 'test'):
            aggregates = aggregate_metrics(current)
            for group, aggregate in aggregates.items():
                saved_aggregate = summary['aggregates'][group]
                for key in ('task_macro_native_top1_pct', 'task_macro_any_best_top1_pct',
                            'task_macro_positive_oracle_regret_pct', 'task_macro_train_scaled_excess'):
                    check_close(aggregate[key], saved_aggregate[key], f'{split}/{group}/{key}')
                scope.append(dict(split=split, group=group,
                    primary_top1_pct=aggregate['task_macro_native_top1_pct'] if group == 'ROUTING18'
                        else aggregate['task_macro_any_best_top1_pct'],
                    primary_top1_rule='native/FP64 ind' if group == 'ROUTING18' else 'native exact any-best',
                    checkpoint_sha256=EXPECTED_SHA, selected_update=44640, **aggregate))
    new_sbs, zero_oracle = [], []
    for problem in NONROUTING_PROBLEMS:
        data, metrics = predictions['test', problem]
        reference = config['references'][problem]
        slot = reference['sbs_slot']
        sbs_scores = np.zeros_like(data['logits'])
        sbs_scores[:, slot] = 1
        sbs, _ = native_task_metrics(sbs_scores, data['native_objective'], data['observed_mask'],
            metrics['objective_direction'], all_best_mask=data['all_best_mask'], reference=reference)
        check_close(sbs['mean_objective'], metrics['sbs_objective'], f'{problem}/TRAIN-SBS')
        new_sbs.append(dict(problem=problem, n=metrics['n'], direction=metrics['objective_direction'],
            model_any_best_top1_pct=metrics['any_best_top1_pct'], sbs_any_best_top1_pct=sbs['any_best_top1_pct'],
            model_objective=metrics['mean_objective'], sbs_objective=sbs['mean_objective'],
            oracle_objective=metrics['oracle_objective'], sbs_solver=reference['sbs_solver'],
            sbs_selection_split='train', sbs_train_n=reference['train_instances'],
            model_vs_sbs_pct=metrics['vs_sbs_pct'], model_actual_regret_pct=metrics['actual_regret_pct'],
            sbs_actual_regret_pct=sbs['actual_regret_pct'], tied_instances=metrics['tied_instances'],
            canonical_model_top1_pct=metrics['canonical_top1_pct'],
            positive_oracle_count=metrics['positive_oracle_count'], zero_oracle_count=metrics['zero_oracle_total'],
            model_equals_sbs_choice_pct=100. * float((data['pred'] == slot).mean()),
            selected_objective_missing=metrics['selected_objective_missing']))
        if problem == 'SMTWTP':
            values = data['native_objective']
            oracle = np.where(data['observed_mask'], values, np.inf).min(1)
            for name, pred in (('R120', data['pred']), ('TRAIN-SBS', np.full(len(values), slot))):
                excess = values[np.arange(len(values)), pred] - oracle
                for population, mask in (('positive_oracle', oracle > 0), ('zero_oracle', oracle == 0)):
                    zero_oracle.append(dict(problem=problem, model=name, population=population, n=int(mask.sum()),
                        actual_regret_pct=100. * float((excess[mask] / oracle[mask]).mean())
                            if population == 'positive_oracle' else None,
                        mean_absolute_excess=float(excess[mask].mean()),
                        total_absolute_excess=float(excess[mask].sum()),
                        positive_loss_count=int((excess[mask] > 0).sum()),
                        positive_loss_pct=100. * float((excess[mask] > 0).mean())))
    historical, alignment = [], []
    for model, root in HISTORY.items():
        protocol, old_lock, final = (read(root / name) for name in ('protocol.json', 'final_test_lock.json', 'final_metrics.json'))
        if (protocol['data_root'] != DATA or protocol['rows_per_problem'] != dict(train=81000, val=4500, test=4500)
                or not final['complete'] or not final['test_read'] or old_lock['selected_without_test'] is not True
                or old_lock['per_problem_assembly'] is not False
                or old_lock['checkpoint_sha256'] != final['checkpoint_sha256']):
            raise ValueError(f'{model}: historical data/lock provenance differs')
        summary = read(root / 'evaluation/test/selected_best/metrics.json')
        if summary['identity']['checkpoint_sha256'] != old_lock['checkpoint_sha256']:
            raise ValueError('Historical saved TEST uses another checkpoint')
        for problem in ROUTING_PROBLEMS:
            old_path = root / 'evaluation/test/selected_best/by_problem' / (problem + '.npz')
            old = archive(old_path)
            data, _ = predictions['test', problem]
            if (old['costs'].dtype != np.float64 or len(old['costs']) != 4500
                    or not np.array_equal(old['costs'], data['native_objective'])
                    or not np.array_equal(old['winner'], data['native_winner_index'])
                    or not np.array_equal(old['pool'], data['pool'])
                    or not np.array_equal(old['indices'], data['indices'])):
                raise ValueError(f'{model}/{problem}: EVAL costs, native labels, pool or row order differ')
            metrics, _ = native_task_metrics(old['logits'], old['costs'], np.ones(old['costs'].shape, dtype=bool),
                'min', native_winner=old['winner'], reference=config['references'][problem])
            saved = final['test']['per_problem'][problem]
            for key in ('mean_cost', 'actual_regret_pct', 'sbs_cost', 'oracle_cost'):
                check_close(metrics[key], saved[key], f'{model}/{problem}/{key}')
            check_close(metrics['native_top1_pct'], 100 * saved['top1'], f'{model}/{problem}/native Top1')
            historical.append(dict(model=model, problem=problem, n=4500,
                native_top1_pct=metrics['native_top1_pct'], any_best_top1_pct=metrics['any_best_top1_pct'],
                mean_cost=metrics['mean_cost'], actual_regret_pct=metrics['actual_regret_pct'],
                sbs_cost=metrics['sbs_cost'], oracle_cost=metrics['oracle_cost'],
                vs_sbs_pct=metrics['vs_sbs_pct'], vs_oracle_pct=metrics['vs_oracle_pct'],
                selected_update=old_lock['selected_update'], checkpoint_sha256=old_lock['checkpoint_sha256'],
                same_eval_costs_pool=True, matched_architecture_training=False))
            alignment.append(dict(model=model, problem=problem, n=4500, costs_equal=True, winners_equal=True,
                pool_equal=True, indices_equal=True, fp64_cost_sha256=array_hash(old['costs']),
                source_npz=str(old_path), comparison='entire small saved cost matrix; no feature arrays or GPU'))
            source_files.append(old_path)
        source_files.extend(root / name for name in ('protocol.json', 'final_test_lock.json', 'final_metrics.json'))
    for problem in ROUTING_PROBLEMS:
        metrics = predictions['test', problem][1]
        historical.append(dict(model='R120', problem=problem, n=metrics['n'],
            native_top1_pct=metrics['native_top1_pct'], any_best_top1_pct=metrics['any_best_top1_pct'],
            **{key: metrics[key] for key in ('mean_cost', 'actual_regret_pct', 'sbs_cost', 'oracle_cost',
                                           'vs_sbs_pct', 'vs_oracle_pct')},
            selected_update=44640, checkpoint_sha256=EXPECTED_SHA, same_eval_costs_pool=True,
            matched_architecture_training=False))
    for model in (*HISTORY, 'R120'):
        rows = [row for row in historical if row['model'] == model]
        aggregate = {key: float(np.mean([row[key] for row in rows])) for key in
            ('native_top1_pct', 'any_best_top1_pct', 'mean_cost', 'actual_regret_pct', 'sbs_cost',
             'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct')}
        historical.append(dict(model=model, problem='ALL_ROUTING18', n=81000, **aggregate,
            selected_update=rows[0]['selected_update'], checkpoint_sha256=rows[0]['checkpoint_sha256'],
            same_eval_costs_pool=True, matched_architecture_training=False))
    nss = nss_rows(predictions, RUNS / 'R84_new90k_attention/nss_reference', 'R120_best44640')
    saved_nss = list(csv.DictReader((ROOT / 'reports/nss_comparison.csv').open(encoding='utf-8-sig')))
    if len(nss) != 4 or len(saved_nss) != 4:
        raise ValueError('Four NSS EVAL datasets are required')
    for row, saved in zip(nss, saved_nss):
        if row['dataset'] != saved['dataset'] or not row['status'].startswith('same EVAL'):
            raise ValueError('NSS EVAL identities do not match R120')
        for key in ('nss_top1_pct', 'current_top1_pct', 'nss_mean_cost', 'current_mean_cost'):
            check_close(row[key], float(saved[key]), 'NSS/' + key)
        if row['nss_historical_train_rows_per_problem'] != 70114 or row['nss_retrained_on_new90k'] is not False:
            raise ValueError('Historical NSS TRAIN provenance changed')
    nss_root = RUNS / 'R84_new90k_attention/nss_reference'
    source_files.append(nss_root / 'protocol.json')
    source_files.extend(nss_root / problem / (split + '_predictions.npz') for problem, split in
                        (('TSP', 'test'), ('CVRP', 'test'), ('TSP', 'TSPLIB'), ('CVRP', 'CVRPLIB')))
    source_files.extend(ROOT / name for name in ('protocol.json', 'final_test_lock.json', 'training_complete.json',
        'test_complete.json', 'reports/report_complete.json', 'reports/nss_comparison.csv'))
    return dict(lock=lock, scope=scope, per_task=per_task, new_sbs=new_sbs, zero_oracle=zero_oracle,
                history=historical, alignment=alignment, methods=support, nss=nss,
                sources=source_files, summaries=summaries)


def number(value, digits=2):
    return '不适用' if value is None else f'{value:.{digits}f}'


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows])


def tables(result):
    scopes = {(row['split'], row['group']): row for row in result['scope']}
    names = {'ROUTING18': '原18题 native', 'NONROUTING6': '新6题 any-best', 'ALL24': '全部24题 any-best'}
    scope_rows = [[names[group], number(scopes['val', group]['primary_top1_pct']),
                   number(scopes['test', group]['primary_top1_pct']), f"{scopes['test', group]['n']:,}"]
                  for group in names]
    new_rows = [[row['problem'] + (' ↑' if row['direction'] == 'max' else ' ↓'), str(row['n']),
        number(row['model_any_best_top1_pct']), number(row['sbs_any_best_top1_pct']),
        number(row['model_objective'], 3), number(row['sbs_objective'], 3),
        number(row['oracle_objective'], 3), str(row['tied_instances'])] for row in result['new_sbs']]
    aggregate = {row['model']: row for row in result['history'] if row['problem'] == 'ALL_ROUTING18'}
    history_rows = [[model, number(row['native_top1_pct']), number(row['actual_regret_pct'], 4),
                     number(row['mean_cost'], 6), str(row['selected_update'])] for model, row in aggregate.items()]
    lookup = {(row['model'], row['problem']): row for row in result['history']}
    task_rows = []
    for problem in ROUTING_PROBLEMS:
        current, baseline = lookup['R120', problem], lookup['R94B', problem]
        task_rows.append([problem, number(current['native_top1_pct']),
            f"{current['native_top1_pct'] - baseline['native_top1_pct']:+.2f}",
            number(current['actual_regret_pct'], 3),
            f"{current['actual_regret_pct'] - baseline['actual_regret_pct']:+.3f}",
            f"{100 * (current['mean_cost'] / baseline['mean_cost'] - 1):+.3f}"])
    zeros = [row for row in result['methods'] if row['strict_zero_selected']]
    zero_summary = []
    for label, tasks in (('原18题', ROUTING_PROBLEMS), ('新6题', NONROUTING_PROBLEMS), ('全部24题', PROBLEMS)):
        subset = [row for row in zeros if row['problem'] in tasks]
        zero_summary.append([label, str(len(subset)), str(sum(row['unique_best_count'] > 0 for row in subset)),
            str(sum(row['zero_group_type'] == 'tied_support_only' for row in subset)),
            str(sum(row['zero_group_type'] == 'empty_best_support' for row in subset))])
    focus = [('TSP', 'LEHD'), ('TSP', 'DIFUSCO'), ('TSP', 'DIFUSCO500'), ('TSP', 'OMNI'),
             ('ATSP', 'MATPOENET'), ('ATSP', 'GLOP'), ('CVRP', 'RouteFinder'), ('OVRPBL', 'MTPOMO')]
    methods = {(row['problem'], row['solver']): row for row in result['methods']}
    focus_rows = []
    for problem, solver in focus:
        row = methods[problem, solver]
        if not row['strict_zero_selected']:
            raise ValueError('The requested strict-zero support example changed')
        focus_rows.append([problem + '/' + solver, str(row['native_winner_support_count']),
            str(row['unique_best_count']), str(row['all_best_support_count']),
            str(row['positive_cost_missed_best_count'])])
    nss_rows_display = [[row['dataset'], number(row['nss_top1_pct']), number(row['current_top1_pct']),
                        number(row['nss_mean_cost'], 4), number(row['current_mean_cost'], 4)] for row in result['nss']]
    return dict(scope=scope_rows, new=new_rows, history=history_rows, tasks=task_rows,
                zero_summary=zero_summary, focus=focus_rows, nss=nss_rows_display, aggregate=aggregate)


def narrative(result, table):
    aggregate = table['aggregate']
    current, old, strong = (aggregate[model] for model in ('R120', 'R84A', 'R94B'))
    zero = next(row for row in result['zero_oracle'] if row['model'] == 'R120' and row['population'] == 'zero_oracle')
    positive = next(row for row in result['zero_oracle'] if row['model'] == 'R120' and row['population'] == 'positive_oracle')
    zeros = [row for row in result['methods'] if row['strict_zero_selected']]
    empty = '、'.join(row['problem'] + '/' + ('SPT' if row['problem'] == 'FJSP' else row['solver'])
                      for row in zeros if not row['all_best_support_count'])
    return dict(
        identity='唯一 VAL 锁定 best=44640；训练于 53280 步早停。全部 VAL/TEST/LIB 复用此模型，未拼接 checkpoint。',
        scope='Top1 均为逐题等权。原18题使用 native/FP64 ind；新6题及 ALL24 使用精确 any-best，不能互称 native Top1。',
        units='新6题 TEST 共 9100 例；并列最优全计正确。↑ 越大越好，↓ 越小越好；原生目标只在同题比较，不报告 ALL24 混合成本。',
        new='新6题中 MIS、FJSP、SMTWTP 的平均原生目标优于 TRAIN 拟合 SBS；JSSP、BPP、MKP 与 SBS 持平，不能把高 Top1 都归为选择收益。',
        sbs='TRAIN 拟合 SBS：MIS=T2T_ER，JSSP=MWKR，FJSP=DAN 10x5+mix，BPP=MBS_prime，MKP=LPDualGreedy，SMTWTP=ATC；各用 10000 个 TRAIN 实例选定。',
        zero=(f"SMTWTP：仅 {positive['n']} 个正 Oracle 实例计算逐实例 regret={positive['actual_regret_pct']:.3f}%。"
              f"另 {zero['n']} 个零 Oracle 实例不做除法，平均绝对损失 {zero['mean_absolute_excess']:.4f}，"
              f"有损失 {zero['positive_loss_count']} 例（{zero['positive_loss_pct']:.3f}%）。"),
        ratios='actual regret 是逐实例有向损失/正 Oracle 的均值；vs_Oracle、vs_SBS 是同题均值比。两者不可替代；SMTWTP 的均值比仅 0.1197%，不是 15.509% 的逐实例 regret。',
        historical=(f"原18题：R120 比 R84A Top1 {current['native_top1_pct'] - old['native_top1_pct']:+.3f} 个百分点，"
            f"比 R94B {current['native_top1_pct'] - strong['native_top1_pct']:+.3f} 个百分点；"
            f"regret 分别高 {current['actual_regret_pct'] - old['actual_regret_pct']:.4f}、"
            f"{current['actual_regret_pct'] - strong['actual_regret_pct']:.4f} 个百分点。没有超越历史强基线。"),
        mismatch='三者 Routing 均用冻结90k（TRAIN 81000、VAL/TEST 各4500）及同一合法候选池；已核对全部小型成本矩阵、native winner、候选顺序和行序。',
        architecture='这不是匹配消融：R120 是从头训练的24题 typed 模型、无 R-Drop、按 VAL any-best 选型；R84A 是18题 scratch；R94B 有冻结预训练 ReLD 节点特征。后两者按 VAL vs_SBS 选型并使用 R-Drop，训练均至57600步。',
        interpretation='因此只能比较已交付模型，不能把差异归因于新增6题、某个模块或统一训练本身。单 seed=2；TEST 曾参与研发，不是全新盲测。',
        zero_groups='严格零选定义为整个 TEST 的选择次数恰为0，不是“少选”。正唯一支持是真实遗漏机会；仅并列支持可能由其它最优解覆盖；空最优组不能算漏选。',
        zero_detail='表中各组模型选择均为0；Routing 每题4500例。原18题的32个零选组中，26个具有正 native winner 支持，其余6个无最优支持。原生 winner 与含并列支持分别统计；新6题没有独立 native ind，不能用 canonical 首项冒充原生 winner。',
        empty='空最优组：' + empty + '。',
        tie='BPP/BFD 与 FFD 均无唯一最优，但各有126例并列支持；模型各漏掉9例正损失，事件可能重叠，不得把两组损失直接相加。',
        nss='NSS 复用 R79 历史权重：每题 TRAIN 70114 例，不是新90k/81000重训。四套 EVAL 成本、native winner 和候选池一致；训练协议不匹配。Top1 越高越好，成本越低越好。',
        bottom='结论：新任务已形成可比较结果，但部分任务仅复现 SBS；Routing 未超过 R94B，且 TSP LEHD/DIFUSCO 等正支持组仍零选。后续方向应围绕实际遗漏与成本收益，不靠并列或空组制造改进。')


def make_markdown(result, table, text, output):
    sections = [
        '# R120 锁定模型简报', text['identity'],
        f"checkpoint SHA256：`{EXPECTED_SHA}`。R120 已发布 `2330198`；本 closeout 属于 Return18，不再提交 Git。",
        '## 1. 同一模型在 VAL / TEST 上怎样？',
        md_table(['范围与 Top1 口径', 'VAL Top1(%)', 'TEST Top1(%)', '每个划分实例数'], table['scope']),
        text['scope'], '原18题 any-best：VAL 71.1123%，TEST 71.3765%。ALL24 原生目标单位不同，mean_cost/SBS cost/Oracle cost 不定义。',
        '## 2. 新6题真的超过 TRAIN-SBS 吗？',
        md_table(['问题/方向', 'TEST n', '模型Top1(%)', 'SBS Top1(%)', '模型原生目标', 'SBS原生目标', 'Oracle原生目标', '并列例'], table['new']),
        text['units'], text['new'], text['sbs'], text['zero'], text['ratios'],
        '## 3. 原18题与 R84A、R94B 相比？',
        md_table(['模型', 'native Top1(%)', 'actual regret(%)', '逐题等权 mean_cost', '锁定步'], table['history']),
        text['historical'], text['mismatch'], text['architecture'], text['interpretation'],
        md_table(['问题', 'R120 Top1(%)', '较R94B Top1(pp)', 'R120 regret(%)', '较R94B regret(pp)', '较R94B成本(%)'], table['tasks']),
        '完整18题、三个模型的成本/SBS/Oracle/vs_SBS/vs_Oracle/actual regret 见 routing18_history.csv；差值正负不代表统一好坏：Top1 越大好，成本及 regret 越小好。',
        '## 4. 零选择是不是漏掉真实机会？',
        md_table(['范围', '严格零选组', '唯一最优>0', '仅并列支持', '空最优组'], table['zero_summary']),
        md_table(['零选择组', '原生winner数', '唯一最优数', '含并列支持数', '错失正损失例'], table['focus']),
        text['zero_groups'], text['zero_detail'], text['empty'], text['tie'],
        '全部方法的完整 TEST native/唯一/并列支持、选择数及损失见 method_support_test.csv；空组和零 native 但有并列支持的组分别保留。',
        '## 5. 四套数据集上的 NSS 对照',
        md_table(['数据集', 'NSS Top1(%)', 'R120 Top1(%)', 'NSS mean_cost', 'R120 mean_cost'], table['nss']),
        text['nss'], 'TSP/CVRP TEST 各4500例，TSPLIB 49例，CVRPLIB 100例。R120 始终是 best44640。',
        '## 结论', text['bottom'],
        '## 证据与范围',
        '只读已完成的 selected_best scores/metrics；不加载模型推理，不训练，不重选模型，不更改旧报告或 transport。成本对齐仅检查保存的小型 FP64 成本矩阵，不读取输入特征大数组。',
        'CSV 为完整数值交付，evidence.json 记录来源及哈希。PDF 内嵌中文字体，不依赖外部图片或 HTML。',
    ]
    (output / 'RESULTS.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')


def make_pdf(table, text, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    if not FONT.is_file():
        raise FileNotFoundError('The verified CJK font is unavailable')
    font = font_manager.FontProperties(fname=str(FONT))
    font_manager.fontManager.addfont(str(FONT))
    matplotlib.rcParams.update({'font.family': font.get_name(), 'pdf.fonttype': 42, 'axes.unicode_minus': False})
    ink, accent = '#19343b', '#176b75'

    def page(title, number):
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='#fcfcfa')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        ax.text(.06, .957, title, fontsize=17, color=ink, fontproperties=font, va='top')
        ax.plot([.06, .94], [.923, .923], color=accent, linewidth=1.5)
        ax.text(.06, .025, 'R120 best44640 | 仅复用保存预测 | Return18 closeout', fontsize=7.5, color='#62747a')
        ax.text(.94, .025, f'{number} / 3', ha='right', fontsize=8, color='#62747a')
        return fig, ax

    def paragraph(ax, y, value, size=9, width=65):
        lines = textwrap.wrap(value, width, break_long_words=True, break_on_hyphens=False)
        ax.text(.06, y, '\n'.join(lines), fontsize=size, linespacing=1.55, color=ink,
                va='top', fontproperties=font)
        return y - len(lines) * .019 - .014

    def heading(ax, y, value):
        ax.text(.06, y, value, fontsize=11.5, color=accent, va='top', fontproperties=font)
        return y - .035

    def draw_table(ax, y, headers, rows, widths, height=.032, size=9):
        widths = np.asarray(widths, dtype=float)
        widths = widths / widths.sum() * .88
        hh = .044 if any('\n' in item for item in headers) else .034
        for index, row in enumerate([headers] + rows):
            h = hh if index == 0 else height
            bottom, x = y - h, .06
            ax.add_patch(Rectangle((x, bottom), .88, h, facecolor='#dcebed' if index == 0
                else '#f0f5f4' if index % 2 else '#ffffff', edgecolor='#dce4e4', linewidth=.35))
            for value, width in zip(row, widths):
                ax.text(x + .007, bottom + h / 2, str(value), fontsize=size if index else size - .3,
                        va='center', color=ink, fontproperties=font, linespacing=1.25)
                x += width
            y = bottom
        return y - .018

    with PdfPages(output / 'RESULTS.pdf', metadata={'Title': 'R120 锁定模型简报', 'Author': 'Return18 CPU closeout'}) as pdf:
        fig, ax = page('同一模型：24题结果与新任务收益', 1)
        y = paragraph(ax, .903, text['identity'], width=70)
        y = paragraph(ax, y, 'SHA256: ' + EXPECTED_SHA, size=7.6, width=100)
        y = heading(ax, y, '1  VAL / TEST：先分清标签口径')
        y = draw_table(ax, y, ['范围', 'VAL Top1(%)', 'TEST Top1(%)', '划分实例数'], table['scope'], [3.3, 2, 2, 2], height=.034)
        y = paragraph(ax, y, text['scope'], size=8.5)
        y = heading(ax, y, '2  新6题：模型是否优于 TRAIN-SBS？')
        y = draw_table(ax, y, ['问题/方向', '例数', '模型\nTop1(%)', 'SBS\nTop1(%)',
                              '模型\n原生目标', 'SBS\n原生目标', 'Oracle\n原生目标', '并列例'],
                       table['new'], [1.55, .85, 1.05, 1.05, 1.4, 1.4, 1.4, .8], height=.034, size=8.3)
        y = paragraph(ax, y, text['units'], size=8.5)
        y = paragraph(ax, y, text['new'], size=8.8)
        y = paragraph(ax, y, text['zero'], size=8.7)
        y = paragraph(ax, y, text['ratios'], size=8.2)
        paragraph(ax, y, '完整 SBS 名称、每题 regret 与零 Oracle 明细见附属 CSV。新6题 SBS 均在各10000例 TRAIN 上拟合，不是 TEST 事后选取。', size=8.2)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = page('原18题：与历史强基线同口径比较', 2)
        y = paragraph(ax, .903, text['mismatch'], size=9)
        y = draw_table(ax, y, ['模型', 'native\nTop1(%)', 'actual\nregret(%)', 'mean_cost', '锁定步'],
                       table['history'], [1.1, 1.5, 1.7, 2.1, 1.1], height=.03)
        y = paragraph(ax, y, text['historical'], size=8.7)
        y = paragraph(ax, y, '非匹配消融：R120为24题typed/scratch，无R-Drop；R84A为18题scratch；R94B含冻结预训练ReLD节点特征。训练目标、VAL选型和步数也不一致，不能归因新增任务。', size=8.2, width=71)
        y = heading(ax, y, '各题 TEST：R120 数值及相对 R94B 差值')
        y = draw_table(ax, y, ['问题', 'R120\nTop1(%)', 'Top1差\n百分点', 'R120\nregret(%)', 'regret差\n百分点', '成本差\n(%)'],
                       table['tasks'], [1.7, 1.25, 1.3, 1.35, 1.35, 1.2], height=.02, size=8.2)
        y = paragraph(ax, y, 'Top1差正为好，regret/成本差负为好。成本仅沿用Routing18逐题等权口径；三模型完整逐题指标见CSV。', size=8.2)
        paragraph(ax, y, '单seed=2；TEST已参与研发，不是全新盲测。不能据此作模块或新增任务的因果归因。', size=8.2)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = page('零选择的真实遗漏与 NSS 四数据集对照', 3)
        y = paragraph(ax, .903, text['zero_groups'], size=8.7)
        y = draw_table(ax, y, ['范围', '零选组', '唯一最优>0', '仅并列支持', '空最优组'],
                       table['zero_summary'], [1.7, 1.2, 1.8, 1.8, 1.5], height=.027, size=8.8)
        y = draw_table(ax, y, ['零选择组', '原生\nwinner数', '唯一\n最优数', '含并列\n支持数', '正损失\n遗漏例'],
                       table['focus'], [3.25, 1.25, 1.25, 1.4, 1.4], height=.023, size=8.5)
        y = paragraph(ax, y, '各组模型选择均为0；Routing每题4500例。原生、唯一及并列支持分开；新6题无独立原生标签。', size=8.2)
        y = paragraph(ax, y, text['empty'], size=8, width=82)
        y = paragraph(ax, y, 'BPP/BFD与FFD均无唯一最优，各有126例并列支持、9例正损失；事件重叠，不能相加。', size=8.2)
        y = heading(ax, y, 'NSS：同一 EVAL，不是同一训练协议')
        y = draw_table(ax, y, ['数据集', 'NSS\nTop1(%)', 'R120\nTop1(%)', 'NSS成本', 'R120成本'],
                       table['nss'], [2, 1.4, 1.4, 1.8, 1.8], height=.026, size=8.6)
        y = paragraph(ax, y, text['nss'], size=8.3)
        paragraph(ax, y, text['bottom'], size=8.5)
        pdf.savefig(fig)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT.resolve()):
        parser.error('Outputs must remain within the owned R120_closeout directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(output / 'mplconfig')
    result = collect()
    table = tables(result)
    text = narrative(result, table)
    files = {'scope_val_test.csv': result['scope'], 'per_problem_metrics.csv': result['per_task'],
             'new6_test_vs_train_sbs.csv': result['new_sbs'], 'smtwtp_zero_oracle.csv': result['zero_oracle'],
             'routing18_history.csv': result['history'], 'cost_pool_alignment.csv': result['alignment'],
             'method_support_test.csv': result['methods'], 'nss_four_datasets.csv': result['nss']}
    for name, rows in files.items():
        csv_write(output / name, rows)
    make_markdown(result, table, text, output)
    make_pdf(table, text, output)
    evidence = dict(complete=True, generated_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=EXPECTED_SHA, selected_update=44640, train_stop=53280,
        original_r120_publication='2330198 (user-supplied; no Git operation performed)',
        output_owner='Return18 next version closeout', cpu_only=True, new_inference=False,
        gpu_launch=False, training=False, modified_historical_sources=False,
        pdf_pages=3, font=str(FONT), r120_metrics_reproduced=True,
        historical_small_cost_matrix_checks=len(result['alignment']),
        nss_eval_populations_matched=4, nss_train_rows=70114, nss_retrained_on_90k=False,
        source_sha256={str(path): file_hash(path) for path in result['sources']},
        generator_sha256=file_hash(Path(__file__)),
        outputs={name: len(rows) for name, rows in files.items()})
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    with zipfile.ZipFile(output / 'CSV.zip', 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in files:
            bundle.write(output / name, arcname=name)
    print(json.dumps(dict(output=str(output), tables=evidence['outputs'], checkpoint_sha256=EXPECTED_SHA), indent=2))


if __name__ == '__main__':
    main()
