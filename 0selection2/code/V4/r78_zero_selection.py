"""R78-A: CPU-only TRAIN-fitted intercepts on locked, saved R76 scores.

This is the authorized round001 method, not a permanent CPU/head-only limit.
Only unplanned broad retraining is disallowed: later user-authorized all18
training based on an Astra plan remains permitted within the resource policy.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np
import torch
from torch.nn import functional as F

from code.unified_selector.registry import PROBLEMS
from .r40_experiment import aggregate, classification_metrics
from .r76_probability_diagnostics import file_hash, load_shards, read_json, score_ranks


BASE = Path(__file__).resolve().parent / 'runs/R76_R45A_single_reld_mt'
ROOT = Path(__file__).resolve().parent / 'runs/R78_zero_selection_goal'
FIT = dict(seed=2, batch_size=1024, max_updates=30000, check_every=1000,
           initial_lr=.05, minimum_lr=1e-6, gradient_tolerance=1e-5,
           objective_tolerance=1e-7, successive_checks=2, prior_precision=1.)
SQUARED_FIT = dict(seed=2, batch_size=1024, max_updates=30000, check_every=1000,
                   gradient_tolerance=1e-5, objective_tolerance=1e-7,
                   reference_tolerance=1e-6, successive_checks=2,
                   prior_precision=1., score_multiplier=1., margin=1., step_divisor=8.)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def progress(root, phase, round_number=1, **values):
    value = dict(updated_utc=utc(), phase=phase, round=round_number, **values)
    write_json(root / 'progress.json', value)
    status_path = root.parent / 'STATUS.json'
    if status_path.exists():
        status = read_json(status_path)
        status['recorded_at'] = utc()
        status['file_based_goal'].update(status='active', round=round_number, phase=phase, complete=False)
        key = f'round{round_number:03d}'
        status[key] = dict(status.get(key, {}), **value)
        if 'next_action' in values:
            status['next_action'] = values['next_action']
        write_json(status_path, status)


def batches(size, seed=2, batch_size=1024):
    if size < 1 or batch_size < 1:
        raise ValueError('Positive TRAIN size and batch size are required')
    rng = np.random.default_rng(seed)
    remaining = np.empty(0, dtype=np.int64)
    while True:
        parts, needed = [], batch_size
        while needed:
            if not len(remaining):
                remaining = rng.permutation(size)
            take = min(needed, len(remaining))
            parts.append(remaining[:take])
            remaining = remaining[take:]
            needed -= take
        yield np.concatenate(parts)


def objective(scores, winner, bias, batch_size=1024):
    scores = np.asarray(scores, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    n, k = scores.shape
    gradient = np.zeros(k, dtype=np.float64)
    loss = 0.
    for start in range(0, n, batch_size):
        z = scores[start:start + batch_size] + bias
        labels = winner[start:start + batch_size]
        maximum = z.max(1)
        exp = np.exp(z - maximum[:, None])
        total = exp.sum(1)
        loss += float((maximum + np.log(total) - z[np.arange(len(z)), labels]).sum())
        gradient += (exp / total[:, None]).sum(0)
        gradient -= np.bincount(labels, minlength=k)
    ce = loss / n
    penalty = float(np.square(bias).sum() / (2 * n))
    gradient = gradient / n + bias / n
    gradient -= gradient.mean()
    return dict(ce=ce, penalty=penalty, objective=ce + penalty,
                gradient_inf=float(np.abs(gradient).max()), gradient=gradient)


def fit_intercepts(value, config=None, callback=None):
    config = FIT if config is None else config
    if value['split'] != 'train':
        raise ValueError('Only TRAIN may enter the intercept optimizer')
    scores = np.asarray(value['logits'], dtype=np.float64)
    winner = np.asarray(value['winner'], dtype=np.int64)
    n, k = scores.shape
    if (winner.shape != (n,) or n == 0 or (winner < 0).any() or (winner >= k).any()
            or np.isnan(scores).any() or np.isposinf(scores).any()
            or not np.isfinite(scores[np.arange(n), winner]).all()):
        raise ValueError('Invalid TRAIN scores or native winners')
    if config['batch_size'] != 1024 or config['prior_precision'] != 1.:
        raise ValueError('The authorized batch and shrinkage are fixed')
    tensor = torch.from_numpy(scores)
    labels = torch.from_numpy(winner)
    bias = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=config['initial_lr'])
    indices = batches(n, config['seed'], config['batch_size'])
    history, streak = [], 0
    started = time.monotonic()
    previous = objective(scores, winner, bias.detach().numpy())

    def record(step, checked, lr, change):
        row = dict(problem=value['problem'], successful_updates=step, batch_size=1024,
                   optimizer_examples=step * 1024, lr=lr, seconds=time.monotonic() - started,
                   objective_change=change, **{key: checked[key] for key in
                                              ('ce', 'penalty', 'objective', 'gradient_inf')})
        history.append(row)
        if callback is not None:
            callback(row)

    record(0, previous, config['initial_lr'], 0.)
    for step in range(1, config['max_updates'] + 1):
        ind = next(indices)
        lr = max(config['minimum_lr'], config['initial_lr'] *
                 (1 - (step - 1) / config['max_updates']) ** 2)
        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(tensor[ind] + bias, labels[ind]) + bias.square().sum() / (2 * n)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite TRAIN objective')
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            bias.sub_(bias.mean())
        if step % config['check_every'] and step != config['max_updates']:
            continue
        checked = objective(scores, winner, bias.detach().numpy())
        change = abs(checked['objective'] - previous['objective'])
        good = (checked['gradient_inf'] <= config['gradient_tolerance']
                and change <= config['objective_tolerance'])
        streak = streak + 1 if good else 0
        record(step, checked, lr, change)
        previous = checked
        if streak >= config['successive_checks']:
            break
    result = bias.detach().numpy().copy()
    converged = streak >= config['successive_checks']
    return dict(problem=value['problem'], pool=value['pool'].tolist(),
                pool_ids=value['pool_ids'].tolist(), bias=result.tolist(), train_rows=n,
                successful_updates=step, optimizer_examples=step * 1024,
                converged=converged, status='converged' if converged else 'inconclusive_budget',
                final={key: previous[key] for key in ('ce', 'penalty', 'objective', 'gradient_inf')},
                history=history)


def squared_inputs(value):
    if value['split'] != 'train':
        raise ValueError('Only TRAIN may define squared-surrogate weights')
    z = np.asarray(value['logits'], dtype=np.float64)
    costs = np.asarray(value['costs'])
    if (costs.dtype != np.float64 or costs.shape != z.shape or z.ndim != 2
            or not z.size or not np.isfinite(z).all() or not np.isfinite(costs).all()):
        raise ValueError('Round002 requires finite complete scores and original FP64 costs')
    n, k = z.shape
    minimum = costs.min(1)
    if (minimum <= 0).any():
        raise ValueError('Round002 requires positive row-minimum costs')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        regret = (costs - minimum[:, None]) / minimum[:, None]
        c = float(regret.mean(dtype=np.float64))
        c = 1. if c == 0. else c
        weights = regret / c
        centered = z - z.mean(1, keepdims=True)
        mean_w, mean_wh, constant = np.zeros(k), np.zeros(k), 0.
        for start in range(0, n, 1024):
            w = weights[start:start + 1024]
            target = 1. + centered[start:start + 1024]
            mean_w += w.sum(0, dtype=np.float64) / n
            mean_wh += (w * target).sum(0, dtype=np.float64) / n
            constant += float((w * target ** 2).sum(dtype=np.float64) / (n * k))
        d, a = 2 * mean_w / k + 1 / n, 2 * mean_wh / k
        multiplier = -float((a / d).sum() / (1 / d).sum())
        reference = -(a + multiplier) / d
        reference -= reference.mean()
        reference_objective = constant + float(a @ reference + .5 * (d * reference ** 2).sum())
        lipschitz = float(2 * weights.max() / k + 1 / n)
    return dict(n=n, k=k, h=centered, w=weights, c=c, d=d, a=a,
                reference=reference, reference_objective=reference_objective,
                L=lipschitz, eta=1 / (8 * lipschitz), statistics_full_passes=2)


def squared_snapshot(data, bias):
    n, k = data['n'], data['k']
    loss, gradient = 0., np.zeros(k, dtype=np.float64)
    for start in range(0, n, 1024):
        w = data['w'][start:start + 1024]
        score = 1. + data['h'][start:start + 1024] + bias
        loss += float((w * score ** 2).sum(dtype=np.float64) / (n * k))
        gradient += (w * score).sum(0, dtype=np.float64) * (2 / (n * k))
    penalty = float(bias @ bias / (2 * n))
    gradient += bias / n
    projected = gradient - gradient.mean()
    delta = bias - data['reference']
    exact_gap = float(.5 * (data['d'] * delta ** 2).sum())
    value = loss + penalty
    gap_residual = value - data['reference_objective'] - exact_gap
    gap_tolerance = 128 * np.finfo(np.float64).eps * max(1., abs(value), abs(data['reference_objective']), exact_gap)
    return dict(surrogate_loss=loss, penalty=penalty, objective=value,
                gradient=gradient, gradient_inf=float(np.abs(projected).max()),
                reference_error_inf=float(np.abs(delta).max()), exact_objective_gap=exact_gap,
                audit_gap_residual=float(gap_residual), audit_gap_tolerance=float(gap_tolerance),
                bias_sum=float(bias.sum()))


def svrg_direction(data, bias, snapshot_bias, snapshot_gradient, indices):
    # For this quadratic, the batch gradient difference is exactly diagonal.
    diagonal = 2 * data['w'][indices].mean(0, dtype=np.float64) / data['k'] + 1 / data['n']
    return diagonal * (bias - snapshot_bias) + snapshot_gradient


def fit_cost_intercepts(value, config=None, callback=None):
    config = SQUARED_FIT if config is None else config
    for key in ('batch_size', 'prior_precision', 'score_multiplier', 'margin', 'step_divisor'):
        if config[key] != SQUARED_FIT[key]:
            raise ValueError('Round002 batch, prior, score scale, margin and step rule are fixed')
    data = squared_inputs(value)
    bias = np.zeros(data['k'], dtype=np.float64)
    rng = np.random.default_rng(config['seed'])
    history, streak, snapshot_passes = [], 0, 1
    started = time.monotonic()
    previous = squared_snapshot(data, bias)
    snapshot_bias, snapshot_gradient = bias.copy(), previous['gradient'].copy()

    def record(step, checked, change, qualifies):
        row = dict(problem=value['problem'], successful_updates=step, batch_size=1024,
                   optimizer_examples=step * 1024, full_gradient_snapshot_passes=snapshot_passes,
                   snapshot_rows_read=snapshot_passes * data['n'], statistics_full_passes=2,
                   seconds=time.monotonic() - started, eta=data['eta'], L=data['L'], c=data['c'],
                   objective_change=change, qualifies=qualifies,
                   **{key: checked[key] for key in ('surrogate_loss', 'penalty', 'objective',
                                                   'gradient_inf', 'reference_error_inf',
                                                   'exact_objective_gap', 'audit_gap_residual',
                                                   'audit_gap_tolerance', 'bias_sum')})
        history.append(row)
        if callback is not None:
            callback(row)

    record(0, previous, 0., False)
    for step in range(1, config['max_updates'] + 1):
        ind = rng.integers(data['n'], size=1024)
        bias -= data['eta'] * svrg_direction(data, bias, snapshot_bias, snapshot_gradient, ind)
        bias -= bias.mean()
        if not np.isfinite(bias).all():
            raise FloatingPointError('Non-finite Round002 optimizer update')
        if step % config['check_every'] and step != config['max_updates']:
            continue
        checked = squared_snapshot(data, bias)
        snapshot_passes += 1
        change = abs(checked['objective'] - previous['objective'])
        qualifies = (checked['gradient_inf'] <= config['gradient_tolerance']
                     and change <= config['objective_tolerance']
                     and checked['reference_error_inf'] <= config['reference_tolerance']
                     and abs(checked['audit_gap_residual']) <= checked['audit_gap_tolerance'])
        streak = streak + 1 if qualifies else 0
        record(step, checked, change, qualifies)
        previous = checked
        if streak >= config['successive_checks']:
            break
        snapshot_bias, snapshot_gradient = bias.copy(), checked['gradient'].copy()
    converged = streak >= config['successive_checks']
    return dict(problem=value['problem'], pool=value['pool'].tolist(), pool_ids=value['pool_ids'].tolist(),
                bias=bias.tolist(), train_rows=data['n'], free_parameters=data['k'] - 1,
                status='converged' if converged else 'inconclusive_budget', converged=converged,
                successful_updates=step, optimizer_examples=step * 1024, batch_size=1024,
                sampling='uniform_with_replacement', initialization='zero_not_analytical_reference',
                gap_dtype='float64', loss_dtype='float64', c=data['c'], L=data['L'], eta=data['eta'],
                curvature_min=float(data['d'].min()), curvature_max=float(data['d'].max()),
                full_gradient_snapshot_passes=snapshot_passes, snapshot_rows_read=snapshot_passes * data['n'],
                statistics_full_passes=2, analytical_reference_role='TRAIN_convergence_audit_only',
                final={key: previous[key] for key in ('surrogate_loss', 'penalty', 'objective',
                                                     'gradient_inf', 'reference_error_inf',
                                                     'exact_objective_gap', 'audit_gap_residual', 'bias_sum')},
                history=history)


def align_bias(value, fitted):
    names = value['pool'].tolist()
    identities = dict(zip(fitted['pool'], fitted['pool_ids']))
    if (len(set(fitted['pool'])) != len(fitted['pool']) or set(names) != set(identities)
            or any(identities[name] != int(gid) for name, gid in zip(names, value['pool_ids']))):
        raise ValueError('Candidate names or global IDs differ from the TRAIN lock')
    offsets = dict(zip(fitted['pool'], fitted['bias']))
    bias = np.asarray([offsets[name] for name in names], dtype=np.float64)
    if not np.isfinite(bias).all() or abs(bias.sum()) > 1e-9:
        raise ValueError('Offsets must be finite and sum-zero')
    return bias


def metrics(value, bias, baseline):
    scores = value['logits'].astype(np.float64) + bias
    order = np.argsort(value['pool_ids'], kind='stable')
    raw = dict(costs=value['costs'][:, order], winner=np.argsort(order)[value['winner']],
               pool=value['pool'][order].tolist(), pool_ids=value['pool_ids'][order].tolist())
    result, selected = classification_metrics(scores[:, order], raw, value['mask'][:, order])
    selected = order[selected]
    result['pool'] = value['pool'].tolist()
    result['pick_dist'] = np.asarray(result['pick_dist'])[np.argsort(order)].tolist()
    slot = result['pool'].index(baseline)
    fixed = float(value['costs'][:, slot].mean(dtype=np.float64))
    result.update(sbs_name=baseline, sbs_cost=fixed, sbs=fixed, sbs_selection_split='train',
                  vs_sbs_pct=(result['mean_cost'] / fixed - 1) * 100)
    rows = np.arange(value['n'])
    cost = value['costs'][rows, selected]
    oracle = np.where(value['mask'], value['costs'], np.inf).min(1)
    sample = dict(top1=(selected == value['winner']).astype(float), mean_cost=cost,
                  actual_regret_pct=(cost - oracle) / oracle * 100,
                  vs_sbs_pct=(cost / fixed - 1) * 100)
    return result, selected, sample


def compare_split(predictions, fitted, baselines, include_tails=False):
    reports = {rule: {} for rule in ('identity', 'candidate')}
    paired, methods, changed = {}, [], []
    for problem in PROBLEMS:
        value = predictions[problem]
        bias = align_bias(value, fitted[problem])
        old = metrics(value, np.zeros_like(bias), baselines[problem])
        new = metrics(value, bias, baselines[problem])
        paired[problem] = {key: new[2][key] - old[2][key] for key in old[2]}
        for rule, offsets, evaluated in (('identity', np.zeros_like(bias), old),
                                         ('candidate', bias, new)):
            reports[rule][problem] = evaluated[0]
            scores = value['logits'].astype(np.float64) + offsets
            exp = np.exp(scores - scores.max(1, keepdims=True))
            probabilities = exp / exp.sum(1, keepdims=True)
            selected = evaluated[1]
            costs = value['costs']
            cost_order = np.argsort(costs[:, np.argsort(value['pool_ids'])], axis=1, kind='stable')
            canonical = np.argsort(value['pool_ids'])
            for slot, name in enumerate(value['pool']):
                wins, picks = value['winner'] == slot, selected == slot
                hits = int((wins & picks).sum())
                margin = scores[:, slot] - np.max(np.delete(scores, slot, axis=1), axis=1) if value['k'] > 1 else np.zeros(value['n'])
                finite = np.isfinite(margin)
                methods.append(dict(problem=problem, split=value['split'], rule=rule,
                                    solver=str(name), solver_id=int(value['pool_ids'][slot]),
                                    winner_count=int(wins.sum()), selected_count=int(picks.sum()),
                                    precision=hits / int(picks.sum()) if picks.any() else None,
                                    recall=hits / int(wins.sum()) if wins.any() else None,
                                    mean_probability=float(probabilities[:, slot].mean()),
                                    fixed_top1=float(wins.mean()),
                                    fixed_top2=float(np.any(canonical[cost_order[:, :2]] == slot, axis=1).mean()),
                                    fixed_top3=float(np.any(canonical[cost_order[:, :3]] == slot, axis=1).mean()),
                                    fixed_mean_cost=float(costs[:, slot].mean()),
                                    mean_margin_wins=float(margin[wins & finite].mean()) if (wins & finite).any() else None,
                                    mean_margin_losses=float(margin[~wins & finite].mean()) if (~wins & finite).any() else None))
        change = old[1] != new[1]
        delta = paired[problem]['actual_regret_pct']
        changed.append(dict(problem=problem, split=value['split'], changed_decisions=int(change.sum()),
                            repaired_top1=int(((old[2]['top1'] == 0) & (new[2]['top1'] == 1)).sum()),
                            broken_top1=int(((old[2]['top1'] == 1) & (new[2]['top1'] == 0)).sum()),
                            regret_gain_sum_pp=float(delta[change & (delta < 0)].sum()),
                            regret_loss_sum_pp=float(delta[change & (delta > 0)].sum()),
                            candidate_p95_regret_pct=float(np.quantile(new[2]['actual_regret_pct'], .95))))
        if include_tails:
            for rule, evaluated in (('identity', old), ('candidate', new)):
                regret = evaluated[2]['actual_regret_pct']
                count = max(1, int(np.ceil(len(regret) * .1)))
                tail = np.argsort(-regret, kind='stable')[:count]
                total = float(regret.sum())
                changed[-1].update({rule + '_p95_regret_pct': float(np.quantile(regret, .95)),
                                   rule + '_max_regret_pct': float(regret.max()),
                                   rule + '_upper_decile_rows': count,
                                   rule + '_upper_decile_regret_contribution': float(regret[tail].sum() / total) if total else 0.})
    result = {rule: dict(per_problem=per, macro=aggregate(per)) for rule, per in reports.items()}
    result['macro_delta'] = {key: result['candidate']['macro'][key] - result['identity']['macro'][key]
                             for key in result['identity']['macro']}
    result['taskwise_harms'] = [p for p in PROBLEMS if
                               reports['candidate'][p]['mean_cost'] > reports['identity'][p]['mean_cost']
                               or reports['candidate'][p]['actual_regret_pct'] > reports['identity'][p]['actual_regret_pct']
                               or reports['candidate'][p]['top1'] < reports['identity'][p]['top1']]
    return result, paired, methods, changed


def val_gate(identity, candidate, converged):
    checks = dict(all_train_fits_converged=bool(converged),
                  all18_macro_top1_improved=candidate['macro_top1'] > identity['macro_top1'],
                  macro_mean_cost_not_worse=candidate['macro_mean_cost'] <= identity['macro_mean_cost'],
                  macro_actual_regret_not_worse=candidate['macro_actual_regret_pct'] <= identity['macro_actual_regret_pct'])
    passed = all(checks.values())
    return dict(accepted=passed, status='accepted' if passed else 'negative' if converged else 'inconclusive',
                checks=checks, tolerance=0., selected_rule='candidate' if passed else 'identity',
                macro_vs_sbs_role='reporting_only', test_labels_used_for_selection=False)


def paired_intervals(paired, replicates=500):
    rng = np.random.default_rng(2)
    keys = ('top1', 'mean_cost', 'actual_regret_pct', 'vs_sbs_pct')
    draws = np.zeros((replicates, len(keys)))
    for problem in PROBLEMS:
        values = np.column_stack([paired[problem][key] for key in keys])
        for rep in range(replicates):
            ind = rng.integers(len(values), size=len(values))
            draws[rep] += values[ind].mean(0) / len(PROBLEMS)
    return dict(seed=2, replicates=replicates, unit='instance within each fixed task',
                assumption='Independent instances; generation-group dependence is not identified by these NPZ files.',
                used_for_acceptance=False,
                intervals={key: dict(lower=float(np.quantile(draws[:, i], .025)),
                                     upper=float(np.quantile(draws[:, i], .975))) for i, key in enumerate(keys)})


def test_once_guard(root, decision, lock):
    if not decision['accepted']:
        write_json(root / 'test_skipped.json', dict(reason=decision['status'], test_opened=False))
        return False
    with (root / 'test_started.json').open('x') as stream:
        json.dump(dict(started_utc=utc(), candidate_sha256=file_hash(root / 'train_locked_candidate.json')),
                  stream, indent=2)
    write_json(root / 'formal_candidate_lock.json', lock)
    return True


def export(root, split, predictions, fitted, baselines, include_tails=False):
    result, paired, methods, changed = compare_split(predictions, fitted, baselines, include_tails)
    write_json(root / f'{split}_comparison.json', result)
    write_csv(root / f'methods_{split}.csv', methods)
    write_csv(root / f'changed_decisions_{split}.csv', changed)
    return result, paired


def figures(root, train_report, val_report, fitted, test_report=None,
            loss_label='Full TRAIN CE + fixed penalty'):
    import matplotlib.pyplot as plt
    from .winner_distribution_figures import style
    style()
    reports = [('train', train_report)]
    if val_report is not None:
        reports.append(('val', val_report))
    if test_report is not None:
        reports.append(('test', test_report))
    for split, report in reports:
        for problem in PROBLEMS:
            identity = report['identity']['per_problem'][problem]
            candidate = report['candidate']['per_problem'][problem]
            pool = candidate['pool']
            folder = root / 'per_problem' / problem / split
            folder.mkdir(parents=True, exist_ok=True)
            for key, file, label in (('method_top1', 'top1_vs_single_methods.png', 'Native Top1'),
                                     ('method_mean_cost', 'mean_cost_vs_single_methods.png', 'Mean cost')):
                fig, ax = plt.subplots(figsize=(9, 4.2))
                ax.bar(np.arange(len(pool)), [candidate[key][name] for name in pool], color='#76969b', label='Fixed method')
                metric = 'top1' if key == 'method_top1' else 'mean_cost'
                ax.axhline(identity[metric], color='#b44c4c', label='Identity')
                ax.axhline(candidate[metric], color='#207855', label='Intercept')
                ax.set_xticks(np.arange(len(pool)), pool, rotation=40, ha='right')
                ax.set_ylabel(label)
                ax.set_title(f'{problem} / {split}')
                ax.legend()
                fig.tight_layout()
                fig.savefig(folder / file, dpi=160)
                plt.close(fig)
            fig, ax = plt.subplots(figsize=(9, 4.2))
            x = np.arange(len(pool))
            ax.bar(x - .2, identity['pick_dist'], .4, label='Identity', color='#b44c4c')
            ax.bar(x + .2, candidate['pick_dist'], .4, label='Intercept', color='#207855')
            ax.set_xticks(x, pool, rotation=40, ha='right')
            ax.set_ylabel('Selection fraction')
            ax.legend()
            fig.tight_layout()
            fig.savefig(folder / 'arm_distribution.png', dpi=160)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for problem in PROBLEMS:
        history = fitted[problem]['history']
        ax.plot([row['successful_updates'] for row in history], [row['objective'] for row in history], label=problem)
    ax.set_xlabel('Optimizer updates (1024 examples each)')
    ax.set_ylabel(loss_label)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'train_loss_curve.png', dpi=160)
    plt.close(fig)


def run(source=BASE, root=ROOT / 'round001', wandb_mode='auto'):
    torch.set_num_threads(2)
    torch.manual_seed(2)
    source, root = Path(source), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        raise FileExistsError('Round001 already started; do not silently refit or reopen TEST')
    lock = read_json(source / 'training_complete.json')['selected_best']
    if file_hash(Path(lock['checkpoint'])) != lock['checkpoint_sha256']:
        raise ValueError('Locked R76 checkpoint hash changed')
    protocol = dict(created_utc=utc(), round=1, model='post-maximin z+b', fit=FIT,
                    cpu_only=True, gpu_used=False, new_forward_passes=0,
                    base_checkpoint=lock, train_split='train/selected_best',
                    val_split=f"val/epoch{lock['epoch']:03d}", prior='mean CE + ||b||^2/(2*N)',
                    selection='one all18 macro Top1/mean-cost/actual-regret gate; VS reporting only',
                    score_precision='cached FP32 promoted to FP64', tie_policy='ascending global solver ID',
                    test_policy='only once after accepted VAL and locked complete candidate')
    write_json(root / 'protocol.json', protocol)
    progress(root, 'loading_train_cached_scores', pid=os.getpid(), gpu_used=False)
    expected = read_json(source / 'evaluation/train/selected_best/metrics.json')
    if set(expected['per_problem']) != set(PROBLEMS):
        raise ValueError('TRAIN must cover all 18 tasks')
    train, paths = load_shards(source / 'evaluation/train/selected_best', 'train', expected['per_problem'])
    write_json(root / 'train_sources.json', [dict(path=str(path), sha256=file_hash(path)) for path in paths])
    baselines = {p: expected['per_problem'][p]['sbs_name'] for p in PROBLEMS}
    zeros = {p: dict(pool=train[p]['pool'].tolist(), pool_ids=train[p]['pool_ids'].tolist(),
                    bias=[0.] * train[p]['k']) for p in PROBLEMS}
    identity, _, _, _ = compare_split(train, zeros, baselines)
    write_json(root / 'identity_step0.json', identity['identity'])
    tracker = None
    try:
        import wandb
        mode = ('online' if os.environ.get('WANDB_API_KEY') else 'offline') if wandb_mode == 'auto' else wandb_mode
        tracker = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                             name='R78_round001_train_intercepts_seed2', mode=mode, dir=str(root), config=protocol,
                             settings=wandb.Settings(init_timeout=15))
    except Exception as error:
        write_json(root / 'wandb_error.json', dict(error=str(error), local_logs_authoritative=True))

    def log(row):
        nonlocal tracker
        with (root / 'loss.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print('[R78 fit] ' + json.dumps(row), flush=True)
        progress(root, 'fitting_train_intercepts', pid=os.getpid(), gpu_used=False, **row)
        if tracker is not None:
            try:
                tracker.log({row['problem'] + '/' + key: row[key] for key in
                             ('objective', 'ce', 'gradient_inf', 'successful_updates', 'lr')})
            except Exception as error:
                write_json(root / 'wandb_error.json', dict(error=str(error), local_logs_authoritative=True))
                tracker = None

    fitted = {}
    for problem in PROBLEMS:
        fitted[problem] = fit_intercepts(train[problem], callback=log)
        write_json(root / 'fit' / f'{problem}.json', fitted[problem])
    candidate_lock = dict(locked_utc=utc(), base_checkpoint_sha256=lock['checkpoint_sha256'],
                          fit_split='train', problems=fitted, score_precision=protocol['score_precision'],
                          tie_policy=protocol['tie_policy'], baselines=baselines)
    write_json(root / 'train_locked_candidate.json', candidate_lock)
    train_report, _ = export(root, 'train', train, fitted, baselines)
    progress(root, 'evaluating_one_locked_candidate_on_val', pid=os.getpid(), gpu_used=False)
    val_dir = source / protocol['val_split'].replace('val/', 'evaluation/val/', 1)
    val_expected = read_json(val_dir / 'metrics.json')
    val, val_paths = load_shards(val_dir, 'val', val_expected['per_problem'])
    write_json(root / 'val_sources.json', [dict(path=str(path), sha256=file_hash(path)) for path in val_paths])
    val_report, paired = export(root, 'val', val, fitted, baselines)
    decision = val_gate(val_report['identity']['macro'], val_report['candidate']['macro'],
                        all(item['converged'] for item in fitted.values()))
    decision.update(candidate_sha256=file_hash(root / 'train_locked_candidate.json'),
                    taskwise_harms=val_report['taskwise_harms'])
    write_json(root / 'val_decision.json', decision)
    write_json(root / 'paired_val_intervals.json', paired_intervals(paired))
    print('[R78 VAL] ' + json.dumps(dict(decision=decision, delta=val_report['macro_delta'])), flush=True)
    test_report = None
    if test_once_guard(root, decision, candidate_lock):
        progress(root, 'formal_test_cached_score_transform_once', pid=os.getpid(), gpu_used=False)
        test_dir = source / 'evaluation/test/selected_best'
        test_expected = read_json(test_dir / 'metrics.json')
        test, _ = load_shards(test_dir, 'test', test_expected['per_problem'])
        test_report, _ = export(root, 'test', test, fitted, baselines)
        write_json(root / 'test_complete.json', dict(completed_utc=utc(), macro=test_report['candidate']['macro']))
    lines = ['# R78 Round001 TRAIN/VAL Handoff', '',
             f"Outcome: **{decision['status']}**. No coefficients or taskwise rules were searched.",
             'R76 own-TRAIN frozen-score head fit; VAL already selected the backbone. Not independent confirmation.',
             '', '| Split | Rule | Top1 % | Mean cost | Actual regret % | VS TRAIN SBS % |',
             '| --- | --- | --- | --- | --- | --- |']
    for split, report in (('TRAIN', train_report), ('VAL', val_report)):
        for rule in ('identity', 'candidate'):
            m = report[rule]['macro']
            lines.append(f"| {split} | {rule} | {100*m['macro_top1']:.6f} | {m['macro_mean_cost']:.9f} | {m['macro_actual_regret_pct']:.9f} | {m['macro_vs_sbs_pct']:.9f} |")
    lines += ['', 'Convergence, taskwise effects, changed decisions and all method counts are retained in the adjacent JSON/CSV files.',
              f"VAL taskwise adverse effects: {', '.join(decision['taskwise_harms']) or 'none'}.",
              'A negative converged screen rejects this prescribed CE intercept fit, not all possible offsets or learnability.',
              'If negative/inconclusive, keep identity and request the next Astra TRAIN/VAL-only research recommendation; do not select rules on TEST.']
    (root / 'RESEARCH_HANDOFF.md').write_text('\n'.join(lines) + '\n')
    progress(root, 'rendering_observation_figures', pid=os.getpid(), gpu_used=False, outcome=decision['status'])
    figures(root, train_report, val_report, fitted, test_report)
    if tracker is not None:
        tracker.log({'val/' + key: value for key, value in val_report['candidate']['macro'].items()})
        tracker.finish()
    progress(root, 'awaiting_astra_round2' if not decision['accepted'] else 'round001_formal_candidate_evaluated',
             outcome=decision['status'], process_terminal=True, gpu_used=False,
             test_opened=decision['accepted'], research_handoff=str(root / 'RESEARCH_HANDOFF.md'))
    write_json(root / 'complete.json', dict(completed_utc=utc(), outcome=decision['status'],
                                          test_opened=decision['accepted'], goal_complete=False))
    return decision


def round002_train_qualified(fitted):
    return set(fitted) == set(PROBLEMS) and all(item['converged'] for item in fitted.values())


def round002_report(root, fitted, decision, train_report, val_report):
    summary = dict(schema_version=1, round=2, outcome=decision['status'],
                   selected_rule=decision['selected_rule'], improvement_certified=False,
                   val_evaluated=val_report is not None, test_opened=decision.get('test_opened', False),
                   tasks=18, converged_count=sum(item['converged'] for item in fitted.values()),
                   total_optimizer_updates=sum(item['successful_updates'] for item in fitted.values()),
                   effective_batch_size=1024, seed=2, cpu_only=True,
                   full_gradient_snapshot_passes=sum(item.get('full_gradient_snapshot_passes', 0) for item in fitted.values()),
                   statistics_and_snapshots_are_not_optimizer_updates=True,
                   analytical_reference_deployed=False, gate=decision,
                   metrics={split: None if report is None else
                            {rule: report[rule]['macro'] for rule in ('identity', 'candidate')}
                            for split, report in (('train', train_report), ('val', val_report))})
    write_json(root / 'result_summary.json', summary)
    lines = ['# R78 Round002 Results', '', f"Outcome: **{decision['status']}**; selected rule: **{decision['selected_rule']}**.",
             'Frozen R76 epoch18; FP64 cost gaps and centered scores; K-1 intercepts per task.',
             'Natural uniform-with-replacement TRAIN sampling, seed2, projected SVRG, exactly1024 rows per parameter update.',
             'Margin1, score multiplier1, prior1, task-wide TRAIN c, eta=1/(8L); no scans or clipping.',
             'Full-gradient snapshots and TRAIN quadratic references are not optimizer updates or deployed candidates.',
             '', '| Split | Rule | Top1 % | CE (descriptive) | Mean cost | Actual regret % | VS TRAIN SBS % |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for split, report in (('TRAIN', train_report), ('VAL', val_report)):
        if report is None:
            lines.append(f'| {split} | Not evaluated | - | - | - | - | - |')
            continue
        for rule in ('identity', 'candidate'):
            m = report[rule]['macro']
            lines.append(f"| {split} | {rule} | {100*m['macro_top1']:.6f} | {m['macro_ce']:.9f} | {m['macro_mean_cost']:.9f} | {m['macro_actual_regret_pct']:.9f} | {m['macro_vs_sbs_pct']:.9f} |")
    lines += ['', '| Task | Updates | Snapshots | L | Projected gradient | Reference error | Exact objective gap | Converged |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for problem in PROBLEMS:
        item = fitted[problem]
        f = item.get('final') or {}
        lines.append(f"| {problem} | {item['successful_updates']} | {item.get('full_gradient_snapshot_passes', 0)} | {item.get('L', '-')} | {f.get('gradient_inf', '-')} | {f.get('reference_error_inf', '-')} | {f.get('exact_objective_gap', '-')} | {item['converged']} |")
    lines += ['', f"Converged tasks: {summary['converged_count']}/18. TEST opened: {summary['test_opened']}.",
              'Stopping requires two successive TRAIN checks: projected gradient<=1e-5, objective change<=1e-7, reference error<=1e-6 and quadratic-gap agreement at FP64 rounding tolerance.',
              'Numerical budget exhaustion is inconclusive, not rejection of this fully converged objective. If any task fails, the complete candidate is not evaluated on VAL.',
              'Native macro Top1 must strictly improve; raw macro mean cost and actual regret must not increase. VS-SBS and NLL are descriptive only.',
              'All18 task details, method comparisons and signed changed-decision costs are saved beside this report. Upper-decile summaries, when present, use the worst ceil(0.1*N) regrets under each rule with stable row-order ties.',
              'No temperature, margin, prior, optimizer or tolerance was adapted using VAL. No TEST/LIB labels were used for fitting or selection.',
              'The continuing file-based goal stays active; this report is not goal completion or an independent confirmation claim.']
    text = '\n'.join(lines) + '\n'
    (root / 'RESULTS.md').write_text(text)
    (root / 'RESEARCH_HANDOFF.md').write_text(text)
    return summary


def run_round002(source=BASE, root=ROOT / 'round002', wandb_mode='auto'):
    torch.set_num_threads(2)
    source, root = Path(source), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        raise FileExistsError('Round002 already started; no silent refit, budget extension or repeated VAL')
    plan = ROOT / 'review-stage/ROUND002_RESEARCH_PLAN.md'
    lock = read_json(source / 'training_complete.json')['selected_best']
    expected_sha = '5ba5777e437235e22682c4d54cfd8cdefca6fb53c27204a8e1e3f753ea9043c9'
    if lock['epoch'] != 18 or lock['checkpoint_sha256'] != expected_sha or file_hash(Path(lock['checkpoint'])) != expected_sha:
        raise ValueError('Round002 must use the same locked R76 epoch18 checkpoint')
    protocol = dict(created_utc=utc(), round=2, plan=str(plan), plan_sha256=file_hash(plan),
                    fit=SQUARED_FIT, base_checkpoint=lock, cpu_only=True, gpu_used=False,
                    new_forward_passes=0, model='centered frozen scores + K-1 sum-zero intercepts',
                    objective='mean_{N*K}(r/c * (1+h+b)^2) + ||b||^2/(2*N)',
                    gap_dtype='float64', loss_dtype='float64', cost_scale='TRAIN mean regret; c=1 only if all zero',
                    sampling='1024 independent uniform TRAIN draws with replacement per optimizer update',
                    full_gradient_snapshots='initial and each1000 updates, separately counted',
                    closed_form_role='TRAIN convergence audit only; never initialization or deployed candidate',
                    val_policy='No VAL read unless all18 tasks pass TRAIN convergence, then one locked candidate comparison',
                    gate='delta native macro Top1>0, delta raw macro mean_cost<=0, delta actual_regret<=0',
                    test_policy='once only after passing locked VAL; otherwise never opened')
    write_json(root / 'protocol.json', protocol)
    progress(root, 'round002_loading_train_cached_scores', round_number=2, pid=os.getpid(), gpu_used=False)
    expected = read_json(source / 'evaluation/train/selected_best/metrics.json')
    if set(expected['per_problem']) != set(PROBLEMS):
        raise ValueError('Round002 requires all18 TRAIN tasks')
    train, paths = load_shards(source / 'evaluation/train/selected_best', 'train', expected['per_problem'])
    write_json(root / 'train_sources.json', [dict(path=str(path), sha256=file_hash(path)) for path in paths])
    baselines = {p: expected['per_problem'][p]['sbs_name'] for p in PROBLEMS}
    tracker = None
    if wandb_mode != 'disabled':
        try:
            import wandb
            mode = ('online' if os.environ.get('WANDB_API_KEY') else 'offline') if wandb_mode == 'auto' else wandb_mode
            tracker = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                                 name='R78_round002_squared_offsets_seed2', mode=mode, dir=str(root), config=protocol,
                                 settings=wandb.Settings(init_timeout=15))
        except Exception as error:
            write_json(root / 'wandb_error.json', dict(error=str(error), local_logs_authoritative=True))

    def log(row):
        nonlocal tracker
        with (root / 'loss.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print('[R78 round002 fit] ' + json.dumps(row), flush=True)
        progress(root, 'round002_fitting_train_squared_svrg', round_number=2,
                 pid=os.getpid(), gpu_used=False, **row)
        if tracker is not None:
            try:
                tracker.log({row['problem'] + '/' + key: row[key] for key in
                             ('objective', 'gradient_inf', 'reference_error_inf', 'exact_objective_gap',
                              'successful_updates', 'full_gradient_snapshot_passes')})
            except Exception as error:
                write_json(root / 'wandb_error.json', dict(error=str(error), local_logs_authoritative=True))
                tracker = None

    fitted = {}
    for problem in PROBLEMS:
        try:
            fitted[problem] = fit_cost_intercepts(train[problem], callback=log)
        except (ValueError, FloatingPointError) as error:
            value = train[problem]
            fitted[problem] = dict(problem=problem, pool=value['pool'].tolist(),
                                    pool_ids=value['pool_ids'].tolist(), bias=[0.] * value['k'],
                                    status='inconclusive_input_or_numerical', converged=False,
                                    successful_updates=0, history=[], final=None, error=str(error))
        write_json(root / 'fit' / f'{problem}.json', fitted[problem])
    write_json(root / 'train_fit_snapshot.json', dict(base_checkpoint_sha256=expected_sha, problems=fitted,
                                                    not_a_deployed_candidate=True))
    numerical_input_failure = any(item['status'] == 'inconclusive_input_or_numerical' for item in fitted.values())
    train_report = None
    if not numerical_input_failure:
        train_report, _ = export(root, 'train', train, fitted, baselines, include_tails=True)
        write_json(root / 'identity_step0.json', train_report['identity'])
    val_report, test_report = None, None
    qualified = round002_train_qualified(fitted)
    if not qualified:
        decision = dict(accepted=False, status='inconclusive', selected_rule='identity',
                        reason='At least one TRAIN task failed the fixed input/convergence/budget requirement',
                        val_evaluated=False, test_opened=False,
                        failed_tasks=[p for p in PROBLEMS if not fitted[p]['converged']])
        write_json(root / 'val_skipped.json', decision)
        test_once_guard(root, decision, {})
    else:
        candidate_lock = dict(locked_utc=utc(), base_checkpoint_sha256=expected_sha,
                              fit_split='train', problems=fitted, baselines=baselines,
                              deployment='argmax(z+b), equivalent to argmax(h+b)',
                              tie_policy='ascending global solver ID', loss_dtype='float64',
                              analytical_reference_deployed=False)
        write_json(root / 'train_locked_candidate.json', candidate_lock)
        progress(root, 'round002_evaluating_one_locked_candidate_on_val', round_number=2,
                 pid=os.getpid(), gpu_used=False)
        val_dir = source / 'evaluation/val/epoch018'
        val_expected = read_json(val_dir / 'metrics.json')
        val, val_paths = load_shards(val_dir, 'val', val_expected['per_problem'])
        write_json(root / 'val_sources.json', [dict(path=str(path), sha256=file_hash(path)) for path in val_paths])
        val_report, paired = export(root, 'val', val, fitted, baselines, include_tails=True)
        decision = val_gate(val_report['identity']['macro'], val_report['candidate']['macro'], True)
        decision.update(val_evaluated=True, test_opened=False,
                        candidate_sha256=file_hash(root / 'train_locked_candidate.json'),
                        taskwise_harms=val_report['taskwise_harms'])
        write_json(root / 'paired_val_intervals.json', paired_intervals(paired))
        if test_once_guard(root, decision, candidate_lock):
            progress(root, 'round002_formal_test_cached_score_transform_once', round_number=2,
                     pid=os.getpid(), gpu_used=False)
            test_dir = source / 'evaluation/test/selected_best'
            test_expected = read_json(test_dir / 'metrics.json')
            test, _ = load_shards(test_dir, 'test', test_expected['per_problem'])
            test_report, _ = export(root, 'test', test, fitted, baselines, include_tails=True)
            decision['test_opened'] = True
            write_json(root / 'test_complete.json', dict(completed_utc=utc(), macro=test_report['candidate']['macro']))
    write_json(root / 'val_decision.json', decision)
    summary = round002_report(root, fitted, decision, train_report, val_report)
    print('[R78 round002 result] ' + json.dumps(summary), flush=True)
    if train_report is not None:
        progress(root, 'round002_rendering_observation_figures', round_number=2, pid=os.getpid(), gpu_used=False)
        figures(root, train_report, val_report, fitted, test_report,
                loss_label='Full TRAIN weighted squared surrogate + fixed prior')
    if tracker is not None:
        tracker.finish()
    progress(root, 'awaiting_astra_round3' if not decision['accepted'] else 'round002_formal_candidate_evaluated',
             round_number=2, outcome=decision['status'], process_terminal=True, gpu_used=False,
             val_evaluated=decision['val_evaluated'], test_opened=decision['test_opened'],
             research_handoff=str(root / 'RESEARCH_HANDOFF.md'),
             next_action='Round002 is terminal; send its TRAIN convergence and available VAL evidence to Astra. No unplanned refit, relaxed tolerance, TEST tuning or goal completion.')
    write_json(root / 'complete.json', dict(completed_utc=utc(), outcome=decision['status'],
                                          val_evaluated=decision['val_evaluated'],
                                          test_opened=decision['test_opened'], goal_complete=False))
    return decision


def round002_reference_readout(source=BASE, root=ROOT / 'round002'):
    """One authorized TRAIN-only decision comparison; no optimization or VAL."""
    torch.set_num_threads(2)
    source, root = Path(source), Path(root)
    output = root / 'train_reference_readout.json'
    if output.exists():
        raise FileExistsError('The one-off TRAIN reference readout is already saved')
    snapshot = read_json(root / 'train_fit_snapshot.json')
    historical = read_json(root / 'result_summary.json')
    train_dir = source / 'evaluation/train/selected_best'
    expected = read_json(train_dir / 'metrics.json')
    fitted = snapshot['problems']
    if set(fitted) != set(PROBLEMS) or set(expected['per_problem']) != set(PROBLEMS):
        raise ValueError('The readout requires the existing complete18 TRAIN snapshot')
    train, paths = load_shards(train_dir, 'train', expected['per_problem'])
    rules = ('identity', 'fitted', 'reference')
    reports = {rule: {} for rule in rules}
    metric_rows, change_rows, margin_rows, changed_details = [], [], [], []
    references = {}
    for problem in PROBLEMS:
        value = train[problem]
        baseline = expected['per_problem'][problem]['sbs_name']
        bias = align_bias(value, fitted[problem])
        reference = squared_inputs(value)['reference']
        references[problem] = dict(pool=value['pool'].tolist(), pool_ids=value['pool_ids'].tolist(),
                                   bias=reference.tolist(), diagnostic_only=True)
        evaluated = {}
        for rule, offsets in zip(rules, (np.zeros_like(bias), bias, reference)):
            result, selected, sample = metrics(value, offsets, baseline)
            reports[rule][problem] = result
            evaluated[rule] = (selected, sample)
            metric_rows.append(dict(problem=problem, rule=rule, rows=value['n'],
                                    top1_pct=100 * result['top1'], mean_cost=result['mean_cost'],
                                    actual_regret_pct=result['actual_regret_pct'],
                                    vs_sbs_pct=result['vs_sbs_pct']))
        for old_rule, new_rule in (('identity', 'fitted'), ('identity', 'reference'),
                                   ('fitted', 'reference')):
            old_selected, old_sample = evaluated[old_rule]
            new_selected, new_sample = evaluated[new_rule]
            changed = old_selected != new_selected
            old_correct, new_correct = old_selected == value['winner'], new_selected == value['winner']
            cost_delta = new_sample['mean_cost'] - old_sample['mean_cost']
            regret_delta = new_sample['actual_regret_pct'] - old_sample['actual_regret_pct']
            change_rows.append(dict(problem=problem, comparison=f'{new_rule}_vs_{old_rule}',
                                    rows=value['n'], changed_predictions=int(changed.sum()),
                                    corrected_top1=int((~old_correct & new_correct).sum()),
                                    broken_top1=int((old_correct & ~new_correct).sum()),
                                    cost_better_count=int((cost_delta < 0).sum()),
                                    cost_worse_count=int((cost_delta > 0).sum()),
                                    top1_delta_pp=100 * float((new_correct.astype(float) - old_correct).mean()),
                                    mean_cost_delta=float(cost_delta.mean(dtype=np.float64)),
                                    actual_regret_delta_pp=float(regret_delta.mean(dtype=np.float64)),
                                    signed_cost_delta_sum=float(cost_delta.sum(dtype=np.float64)),
                                    signed_regret_delta_sum_pct=float(regret_delta.sum(dtype=np.float64))))
        epsilon = float(np.abs(bias - reference).max())
        scores = np.where(value['mask'], value['logits'].astype(np.float64) + bias, -np.inf)
        top_two = np.partition(scores, -2, axis=1)[:, -2:]
        margin = top_two[:, 1] - top_two[:, 0]
        stable = margin > 2 * epsilon
        changed = evaluated['fitted'][0] != evaluated['reference'][0]
        margin_rows.append(dict(problem=problem, rows=value['n'], full_K_epsilon_inf=epsilon,
                                twice_epsilon=2 * epsilon, fitted_margin_min=float(margin.min()),
                                fitted_margin_mean=float(margin.mean()),
                                fitted_margin_p05=float(np.quantile(margin, .05)),
                                fitted_margin_median=float(np.median(margin)),
                                fitted_margin_max=float(margin.max()),
                                certified_stable_count=int(stable.sum()),
                                not_certified_stable_count=int((~stable).sum()),
                                changed_predictions=int(changed.sum()),
                                changed_above_stability_bound=int((changed & stable).sum())))
        for row in np.flatnonzero(changed):
            old_slot, new_slot = int(evaluated['fitted'][0][row]), int(evaluated['reference'][0][row])
            changed_details.append(dict(problem=problem, train_index=int(value['indices'][row]),
                                        winner_global_id=int(value['pool_ids'][value['winner'][row]]),
                                        fitted_global_id=int(value['pool_ids'][old_slot]),
                                        reference_global_id=int(value['pool_ids'][new_slot]),
                                        fitted_top2_margin=float(margin[row]), full_K_epsilon_inf=epsilon,
                                        fitted_cost=float(value['costs'][row, old_slot]),
                                        reference_cost=float(value['costs'][row, new_slot]),
                                        regret_delta_pp=float(evaluated['reference'][1]['actual_regret_pct'][row]
                                                              - evaluated['fitted'][1]['actual_regret_pct'][row])))
        print(f'[R78 TRAIN reference] {problem}: fitted/reference changed={int(changed.sum())}', flush=True)
    macros = {rule: aggregate(reports[rule]) for rule in rules}
    comparisons = {}
    for name in ('fitted_vs_identity', 'reference_vs_identity', 'reference_vs_fitted'):
        rows = [row for row in change_rows if row['comparison'] == name]
        comparisons[name] = {key: sum(row[key] for row in rows) for key in
                             ('rows', 'changed_predictions', 'corrected_top1', 'broken_top1',
                              'cost_better_count', 'cost_worse_count')}
        comparisons[name].update({key: float(np.mean([row[key] for row in rows])) for key in
                                 ('top1_delta_pp', 'mean_cost_delta', 'actual_regret_delta_pp')})
    converged = [p for p in PROBLEMS if fitted[p]['converged']]
    bounds = dict(fixed_converged_tasks=len(converged), remaining_tasks=len(PROBLEMS) - len(converged),
                  best_possible_macro_top1_pct=100 * (sum(reports['fitted'][p]['top1'] for p in converged)
                                                       + len(PROBLEMS) - len(converged)) / len(PROBLEMS),
                  best_possible_macro_actual_regret_pct=sum(reports['fitted'][p]['actual_regret_pct']
                                                           for p in converged) / len(PROBLEMS))
    identity, reference_macro = macros['identity'], macros['reference']
    negative = (reference_macro['macro_top1'] < identity['macro_top1']
                and reference_macro['macro_mean_cost'] > identity['macro_mean_cost']
                and reference_macro['macro_actual_regret_pct'] > identity['macro_actual_regret_pct'])
    disposition = 'route_closed_negative' if negative else 'research_reinterpretation_required'
    result = dict(schema_version=1, completed_utc=utc(), split='train', tasks=len(PROBLEMS),
                  rows=sum(value['n'] for value in train.values()),
                  historical_execution_outcome=historical['outcome'], research_decision=disposition,
                  source_train_directory=str(train_dir), train_shards=[str(path) for path in paths],
                  base_checkpoint_sha256=snapshot['base_checkpoint_sha256'],
                  additional_optimizer_updates=0, new_backbone_forwards=0, gpu_used=False,
                  val_read=False, test_read=False, plan_gate_changed=False,
                  reference_role='Existing closed-form TRAIN diagnostic only; not a deployed or retroactively qualified candidate',
                  tie_policy='ascending global solver ID', macro_aggregation='equal weight per task',
                  signed_difference='new rule minus old rule; negative cost/regret is better',
                  metrics={rule: dict(macro=macros[rule], per_problem=reports[rule]) for rule in rules},
                  prediction_comparisons=comparisons, per_problem_prediction_changes=change_rows,
                  margins=margin_rows, fitted_reference_changed_rows=changed_details,
                  reference_intercepts=references, fixed14_fitted_arithmetic_bounds=bounds,
                  reference_worsens_all_three_task_count=sum(
                      reports['reference'][p]['top1'] < reports['identity'][p]['top1']
                      and reports['reference'][p]['mean_cost'] > reports['identity'][p]['mean_cost']
                      and reports['reference'][p]['actual_regret_pct'] > reports['identity'][p]['actual_regret_pct']
                      for p in PROBLEMS),
                  next_phase='waiting_R77_final_trainval_distributions', goal_complete=False)
    write_csv(root / 'train_reference_metrics.csv', metric_rows)
    write_csv(root / 'train_reference_changes.csv', change_rows)
    write_csv(root / 'train_reference_margins.csv', margin_rows)
    lines = ['# Round002: Authorized TRAIN Reference Readout', '',
             f"Historical execution: {historical['outcome']}. Research decision: {disposition}.",
             'One existing TRAIN-cache readout; zero optimizer updates, new forwards, VAL or TEST reads.',
             'Reference b* is diagnostic only. Original outputs and the PLAN gate are unchanged.',
             '', '| TRAIN rule | Native Top1 % | Raw macro mean cost | Actual regret % |',
             '|---|---:|---:|---:|']
    for rule in rules:
        m = macros[rule]
        lines.append(f"| {rule} | {100*m['macro_top1']:.9f} | {m['macro_mean_cost']:.12f} | {m['macro_actual_regret_pct']:.9f} |")
    lines += ['', '| Comparison (new minus old) | Changed | Corrected Top1 | Broken Top1 | Cost delta | Regret delta pp |',
              '|---|---:|---:|---:|---:|---:|']
    for name, row in comparisons.items():
        lines.append(f"| {name} | {row['changed_predictions']} | {row['corrected_top1']} | {row['broken_top1']} | {row['mean_cost_delta']:.12g} | {row['actual_regret_delta_pp']:.12g} |")
    lines += ['', 'Full-K epsilon=||b-b*||_inf. Fitted top-two margin>2*epsilon certifies an unchanged winner.',
              'This is a descriptive stability bound, not a new acceptance gate.',
              '', '| Task | Full-K epsilon | Mean fitted margin | Rows margin>2epsilon | Changed | Changed above bound |',
              '|---|---:|---:|---:|---:|---:|']
    for row in margin_rows:
        lines.append(f"| {row['problem']} | {row['full_K_epsilon_inf']:.9g} | {row['fitted_margin_mean']:.9g} | {row['certified_stable_count']} | {row['changed_predictions']} | {row['changed_above_stability_bound']} |")
    lines += ['', '| Task | Rule | Native Top1 % | Mean cost | Actual regret % |',
              '|---|---|---:|---:|---:|']
    for row in metric_rows:
        lines.append(f"| {row['problem']} | {row['rule']} | {row['top1_pct']:.9f} | {row['mean_cost']:.12f} | {row['actual_regret_pct']:.9f} |")
    lines += ['', f"Fixed14 fitted arithmetic bounds: Top1 <= {bounds['best_possible_macro_top1_pct']:.9f}%; "
              f"actual regret >= {bounds['best_possible_macro_actual_regret_pct']:.9f}%.",
              'These bounds hold those14 fitted rules fixed; they are not a claim of fitted/reference prediction equality.',
              'No further fixed-R76 intercept, scale, margin, prior or numerical-precision trial is authorized.',
              'Goal stays active, waiting_R77_final_trainval_distributions; R76 identity remains incumbent.']
    (root / 'TRAIN_REFERENCE_READOUT.md').write_text('\n'.join(lines) + '\n')
    write_json(output, result)
    print('[R78 TRAIN reference result] ' + json.dumps(dict(research_decision=disposition,
                                                           macro=macros, changes=comparisons)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=BASE)
    parser.add_argument('--round', type=int, choices=(1, 2), default=1)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--wandb-mode', choices=('auto', 'offline', 'online', 'disabled'), default='auto')
    parser.add_argument('--round002-train-reference-readout', action='store_true',
                        help='Authorized one-off TRAIN comparison only; no optimizer, VAL or TEST')
    args = parser.parse_args()
    if args.round002_train_reference_readout:
        round002_reference_readout(args.source_root, args.root or ROOT / 'round002')
        return
    root = args.root or ROOT / f'round{args.round:03d}'
    execute = run if args.round == 1 else run_round002
    execute(args.source_root, root, args.wandb_mode)


if __name__ == '__main__':
    main()
