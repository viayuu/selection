"""R78 round004: fixed-budget CPU conditional-cost heads on frozen B caches."""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .performance_evaluation import decision_metrics
from .r40_experiment import aggregate
from .r76_probability_diagnostics import file_hash, masked_softmax_fp32, read_json, score_ranks
from .r78_zero_selection import batches, progress, utc, write_csv, write_json


GOAL = Path(__file__).resolve().parent / 'runs/R78_zero_selection_goal'
ROOT = GOAL / 'round004'
PLAN = GOAL / 'review-stage/ROUND004_FROZEN_SCORE_COST_HEAD_PLAN.md'
BASELINE = GOAL / 'round003/baseline_B_train_val.json'
BASE_SHA = '31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b'
UPDATES = 1380
BATCH = 1024
SEED = 2


def learning_rate(step):
    return .01 * (1 + math.cos(math.pi * step / UPDATES)) / 2


def train_targets(costs, mask):
    if costs.dtype != np.float64:
        raise ValueError('Original FP64 costs are required')
    oracle = np.where(mask, costs, np.inf).min(1)
    if not np.isfinite(oracle).all() or (oracle <= 0).any():
        raise ValueError('Relative regret requires positive finite valid minimum costs')
    regret = np.where(mask, (costs - oracle[:, None]) / oracle[:, None], 0.)
    scale = float((regret.sum(1) / mask.sum(1)).mean(dtype=np.float64))
    return regret / scale if scale > 0 else np.zeros_like(regret), scale


def head(probabilities, matrix):
    p = probabilities.to(dtype=matrix.dtype)
    return 1 - p + F.linear(p, matrix)


def objective(probabilities, targets, mask, matrix, train_rows):
    residual = (head(probabilities, matrix) - targets).square().masked_fill(~mask, 0.)
    mse = (residual.sum(1) / mask.sum(1)).mean()
    prior = matrix.square().sum() / (2 * train_rows)
    return mse + prior


def predictions(probabilities, matrix, batch_size=BATCH):
    with torch.no_grad():
        weight = torch.from_numpy(matrix)
        return np.concatenate([head(torch.from_numpy(probabilities[start:start + batch_size]), weight).numpy()
                               for start in range(0, len(probabilities), batch_size)])


def head_order(values, logits, mask, pool_ids):
    # Lexicographic exact ties: predicted cost, original B logits, global ID.
    ids = np.broadcast_to(pool_ids, values.shape)
    return np.lexsort((ids, -logits, np.where(mask, values, np.inf)), axis=1)


def load_scores(directory, problem, expected):
    parts = []
    for path in sorted(Path(directory).glob('rank*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            prefix = problem + '__'
            if prefix + 'logits' in saved:
                parts.append({key: saved[prefix + key] for key in
                              ('logits', 'costs', 'winner', 'indices', 'pool', 'pool_ids')})
    if not parts:
        raise FileNotFoundError(f'{directory}: missing {problem} scores')
    value = {key: np.concatenate([part[key] for part in parts]) for key in
             ('logits', 'costs', 'winner', 'indices')}
    value.update(pool=parts[0]['pool'], pool_ids=parts[0]['pool_ids'])
    if any(not np.array_equal(part[key], value[key]) for part in parts for key in ('pool', 'pool_ids')):
        raise ValueError('Shard candidate identity mismatch')
    order = np.argsort(value['indices'], kind='stable')
    for key in ('logits', 'costs', 'winner', 'indices'):
        value[key] = value[key][order]
    n, k = value['logits'].shape
    if (n != expected['n'] or value['pool'].tolist() != expected['pool']
            or not np.array_equal(value['indices'], np.arange(n))
            or value['costs'].shape != (n, k) or value['costs'].dtype != np.float64
            or value['logits'].dtype != np.float32):
        raise ValueError('Incomplete or changed native frozen-score cache')
    value['mask'] = ~np.isneginf(value['logits'])
    value['p'], _ = masked_softmax_fp32(value['logits'], value['mask'])
    value['B_choice'], _ = score_ranks(value['logits'], value['pool_ids'], value['mask'])
    value.update(n=n, k=k)
    return value


def fit_task(problem, value, root, state, log):
    target, scale = train_targets(value['costs'], value['mask'])
    matrix = torch.zeros(value['k'], value['k'], dtype=torch.float64, requires_grad=True)
    receipt = dict(problem=problem, train_rows=value['n'], candidates=value['k'], a_t=scale,
                   train_target_dtype='float64', probability_dtype='float32', matrix_loss_dtype='float64',
                   successful_updates=0, degenerate_TRAIN_task=scale == 0)
    if scale == 0:
        return matrix.detach().numpy(), receipt
    optimizer = torch.optim.AdamW([matrix], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    receipt['fresh_optimizer_state_entries'] = len(optimizer.state)
    p, y, mask = torch.from_numpy(value['p']), torch.from_numpy(target), torch.from_numpy(value['mask'])
    stream = batches(value['n'], seed=SEED, batch_size=BATCH)
    for step in range(UPDATES):
        index = next(stream)
        optimizer.param_groups[0]['lr'] = learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(p[index], y[index], mask[index], matrix, value['n'])
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(matrix.grad).all():
            raise FloatingPointError('Nonfinite minibatch loss or gradient; no retry or budget extension')
        optimizer.step()
        if not torch.isfinite(matrix).all():
            raise FloatingPointError('Nonfinite head update; preserve incomplete attempt')
        state['successful_updates'] += 1
        receipt['successful_updates'] = step + 1
        if state['successful_updates'] == 1:
            write_json(root / 'first_batch_check.json', dict(problem=problem, batch_size=len(index),
                successful_update=1, included_in_declared_budget=True, finite_loss=float(loss.detach()),
                finite_gradient=True, finite_matrix=True, CPU_only=True))
        if step == 0 or (step + 1) % 200 == 0 or step + 1 == UPDATES:
            log(dict(problem=problem, task_successful_updates=step + 1,
                     successful_updates=state['successful_updates'], minibatch_loss=float(loss.detach()),
                     lr=learning_rate(step), a_t=scale))
    receipt.update(completed_utc=utc(), optimizer='fresh AdamW', weight_decay=0.,
                   last_update_lr=learning_rate(UPDATES - 1), boundary_lr=learning_rate(UPDATES),
                   rows_drawn=UPDATES * BATCH, extended_budget=False, optimum_certificate=False)
    return matrix.detach().numpy().copy(), receipt


def fitted_metrics(value, order, baseline):
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(value['k'])[None, :], axis=1)
    raw = dict(costs=value['costs'], winner=value['winner'], pool=value['pool'].tolist())
    result, choice = decision_metrics(-ranks.astype(np.float64), raw, value['mask'])
    fixed, oracle = baseline['sbs_cost'], result['oracle_cost']
    result.update(ce=baseline['ce'], ce_source='unchanged frozen B; not softmax(-g)', pool=raw['pool'],
                  native_correct=int((choice == value['winner']).sum()), sbs_name=baseline['sbs_name'],
                  sbs=fixed, sbs_cost=fixed, sbs_selection_split='train',
                  vs_sbs_pct=(result['mean_cost'] / fixed - 1) * 100,
                  vbs_gap_closed_pct=(fixed - result['mean_cost']) / (fixed - oracle) * 100 if fixed != oracle else 0.)
    return result, choice


def evaluate_split(split, data, matrices, receipts, baseline):
    per, choices, mse = {}, {}, {}
    for problem in PROBLEMS:
        value = data[problem]
        g = predictions(value['p'], matrices[problem])
        order = (head_order(g, value['logits'], value['mask'], value['pool_ids'])
                 if receipts[problem]['a_t'] > 0 else
                 np.argsort(-np.where(value['mask'], value['logits'], -np.inf)[:,
                            np.argsort(value['pool_ids'], kind='stable')], axis=1, kind='stable'))
        if receipts[problem]['a_t'] == 0:
            order = np.argsort(value['pool_ids'], kind='stable')[order]
        per[problem], choices[problem] = fitted_metrics(value, order, baseline[split]['per_problem'][problem])
        if split == 'train':
            target, _ = train_targets(value['costs'], value['mask'])
            squared = np.where(value['mask'], (g - target) ** 2, 0.)
            mse[problem] = dict(mse=float((squared.sum(1) / value['mask'].sum(1)).mean()),
                prior=float(np.square(matrices[problem]).sum() / (2 * value['n'])),
                statistics_only_no_optimizer_or_gradient=True)
    report = dict(per_problem=per, macro=aggregate(per),
                  native_total=sum(item['n'] for item in per.values()),
                  native_correct=sum(item['native_correct'] for item in per.values()),
                  CE_definition='Frozen B CE unchanged; the cost-head scores are not winner probabilities')
    if mse:
        report['TRAIN_fit_statistics'] = mse
    return report, choices


def method_rows(split, data, choices):
    rows = []
    for rule, by_problem in choices.items():
        for problem, value in data.items():
            support = np.bincount(value['winner'], minlength=value['k'])
            picks = np.bincount(by_problem[problem], minlength=value['k'])
            for slot, method in enumerate(value['pool']):
                rows.append(dict(split=split, rule=rule, problem=problem, method=str(method),
                    solver_id=int(value['pool_ids'][slot]), winner_count=int(support[slot]),
                    selected_count=int(picks[slot]), rows=value['n'],
                    zero_with_winner=bool(support[slot] > 0 and picks[slot] == 0)))
    return rows


def changed_rows(split, data, choices):
    rows, categories = [], []
    for rule in ('fitted', 'fallback_B'):
        for problem, value in data.items():
            old, new, winner = value['B_choice'], choices[rule][problem], value['winner']
            changed = old != new
            fixed = (old != winner) & (new == winner)
            harmed = (old == winner) & (new != winner)
            wrong = changed & (old != winner) & (new != winner)
            index = np.arange(value['n'])
            delta = value['costs'][index, new] - value['costs'][index, old]
            oracle = np.where(value['mask'], value['costs'], np.inf).min(1)
            rows.append(dict(split=split, problem=problem, comparison=rule + '_vs_B',
                changed_predictions=int(changed.sum()), corrected=int(fixed.sum()), harmed=int(harmed.sum()),
                wrong_to_wrong=int(wrong.sum()), exact_cost_tie_changes=int((changed & (delta == 0)).sum()),
                negative_cost_change_sum=float(delta[delta < 0].sum(dtype=np.float64)),
                positive_cost_change_sum=float(delta[delta > 0].sum(dtype=np.float64)),
                mean_cost_delta=float(delta.mean(dtype=np.float64)),
                actual_regret_delta_pp=float((delta / oracle * 100).mean(dtype=np.float64))))
            for category, keep in (('corrected', fixed), ('harmed', harmed), ('wrong_to_wrong', wrong)):
                d = delta[keep]
                categories.append(dict(split=split, problem=problem, comparison=rule + '_vs_B', category=category,
                    rows=int(keep.sum()), cost_decrease_count=int((d < 0).sum()), cost_increase_count=int((d > 0).sum()),
                    negative_cost_change_sum=float(d[d < 0].sum(dtype=np.float64)),
                    positive_cost_change_sum=float(d[d > 0].sum(dtype=np.float64))))
    return rows, categories


def run(root):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        raise FileExistsError('Round004 already started: no silent refit, retry or extra budget')
    baseline = read_json(BASELINE)
    if (baseline['checkpoint_sha256'] != BASE_SHA or baseline['source_epoch'] != 28
            or baseline['logical_update'] != 34776 or set(baseline['train']['per_problem']) != set(PROBLEMS)
            or set(baseline['val']['per_problem']) != set(PROBLEMS)):
        raise ValueError('Wrong locked B TRAIN/VAL metadata')
    protocol = dict(created_utc=utc(), plan=str(PLAN), plan_sha256=file_hash(PLAN), source=str(BASELINE),
        source_checkpoint_sha256=BASE_SHA, source_epoch=28, source_update=34776,
        inputs='saved frozen B logits only; original logits also resolve exact cost-head ties',
        probability='valid masked FP32 softmax, T=1', target='TRAIN FP64 relative regret / TRAIN a_t',
        model='g=1-p+D_t p; D_t initialized to zero; no separate intercept', matrix_loss_dtype='float64',
        prior='||D_t||_F^2/(2*N_t), explicitly included once per minibatch',
        optimizer='fresh AdamW betas(.9,.999) eps1e-8 weight_decay0', seed=SEED,
        updates_per_nondegenerate_task=UPDATES, logical_batch_size=BATCH, maximum_updates=len(PROBLEMS) * UPDATES,
        sampler='per-task numpy default_rng(seed2), continuous natural permutations with boundary tail fill',
        lr='0.01*(1+cos(pi*s/1380))/2, s=0..1379; zero at boundary1380, no warmup',
        VAL_endpoints=1, VAL_selection='one frozen all18 head; no taskwise mixture; exact B macro gate',
        GPU_used=False, backbone_passes=0, solver_calls=0, API_calls=0, TEST_read=False, LIB_read=False,
        wandb_mode='offline', additional_optimizer_budget=False, optimum_certificate=False)
    write_json(root / 'protocol.json', protocol)
    state = dict(successful_updates=0, completed_tasks=0)
    receipt = dict(pid=os.getpid(), host=socket.gethostname(), started_utc=utc(), cpu_threads=2,
                   CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], log=str(root / 'train.log'))
    write_json(root / 'launch_receipt.json', receipt)
    progress(root, 'round004_loading_TRAIN_scores', round_number=4, **receipt,
             optimizer_updates_executed=0, max_optimizer_updates=len(PROBLEMS) * UPDATES,
             next_action='Execute only the approved round004 fixed CPU budget, then one complete18 VAL; no TEST/LIB.')
    tracker = None
    try:
        import wandb
        tracker = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
            name='R78_round004_frozen_B_cost_head_seed2', mode='offline', dir=str(root), config=protocol,
            settings=wandb.Settings(init_timeout=15, _disable_stats=True))
    except Exception as error:
        write_json(root / 'wandb_error.json', dict(error=str(error), local_logs_authoritative=True))
    try:
        train = {p: load_scores(baseline['prediction_directories']['train'], p, baseline['train']['per_problem'][p])
                 for p in PROBLEMS}
        params = sum(value['k'] ** 2 for value in train.values())
        if params != 513 or sum(value['k'] for value in train.values()) != 89:
            raise ValueError('Locked full18 pool capacity must be 513 parameters /89 candidates')
        matrices, receipts = {}, {}
        started = time.monotonic()

        def log(row):
            elapsed = time.monotonic() - started
            total = state['successful_updates']
            eta = elapsed / total * (len(PROBLEMS) * UPDATES - total) if total >= 200 else None
            row.update(updated_utc=utc(), completed_tasks=state['completed_tasks'], elapsed_seconds=elapsed,
                       training_eta_seconds=eta)
            with (root / 'loss.jsonl').open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            print('[R78 round004 fit] ' + json.dumps(row), flush=True)
            progress(root, 'round004_fitting_TRAIN_heads', round_number=4, pid=os.getpid(),
                optimizer_updates_executed=total, max_optimizer_updates=len(PROBLEMS) * UPDATES,
                parameters=params, GPU_used=False, TEST_read=False, LIB_read=False,
                **{key: value for key, value in row.items() if key != 'updated_utc'})
            if tracker is not None:
                tracker.log({'fit/' + key: value for key, value in row.items() if isinstance(value, (int, float))}, step=total)

        for problem in PROBLEMS:
            matrices[problem], receipts[problem] = fit_task(problem, train[problem], root, state, log)
            state['completed_tasks'] += 1
        payload = {p + '__D': matrix for p, matrix in matrices.items()}
        for problem, value in train.items():
            payload.update({problem + '__pool': value['pool'], problem + '__pool_ids': value['pool_ids']})
        np.savez_compressed(root / 'locked_heads.npz', **payload)
        lock = dict(locked_utc=utc(), matrices=str(root / 'locked_heads.npz'),
            matrices_sha256=file_hash(root / 'locked_heads.npz'), tasks=18, parameters=params,
            source_checkpoint_sha256=BASE_SHA, successful_updates=state['successful_updates'], tasks_fitted=receipts,
            frozen_before_VAL=True, retraining_or_rule_revision_after_VAL=False,
            formal_TEST_LIB_authorized=False)
        write_json(root / 'head_lock.json', lock)
        progress(root, 'round004_heads_locked_reading_single_VAL_endpoint', round_number=4,
            pid=os.getpid(), optimizer_updates_executed=state['successful_updates'], head_lock=str(root / 'head_lock.json'),
            GPU_used=False, TEST_read=False, LIB_read=False)
        fitted_train, train_choices = evaluate_split('train', train, matrices, receipts, baseline)
        val = {p: load_scores(baseline['prediction_directories']['val'], p, baseline['val']['per_problem'][p])
               for p in PROBLEMS}
        fitted_val, val_choices = evaluate_split('val', val, matrices, receipts, baseline)
        b, fitted = baseline['val']['macro'], fitted_val['macro']
        gate = dict(native_top1_improves=fitted['macro_top1'] > b['macro_top1'],
                    raw_mean_cost_not_worse=fitted['macro_mean_cost'] <= b['macro_mean_cost'],
                    actual_regret_not_worse=fitted['macro_actual_regret_pct'] <= b['macro_actual_regret_pct'])
        passed = all(gate.values())
        reports = {'train': dict(B=baseline['train'], fitted=fitted_train, fallback_B=baseline['train']),
                   'val': dict(B=baseline['val'], fitted=fitted_val, fallback_B=baseline['val'])}
        tasks, methods, changes, categories = [], [], [], []
        for split, data, fitted_choices in (('train', train, train_choices), ('val', val, val_choices)):
            old = {p: value['B_choice'] for p, value in data.items()}
            choices = dict(B=old, fitted=fitted_choices, fallback_B=old)
            methods.extend(method_rows(split, data, choices))
            c, group = changed_rows(split, data, choices)
            changes.extend(c)
            categories.extend(group)
            for rule, report in reports[split].items():
                for problem, value in report['per_problem'].items():
                    tasks.append(dict(split=split, rule=rule, problem=problem, top1_pct=100 * value['top1'],
                        mean_cost=value['mean_cost'], actual_regret_pct=value['actual_regret_pct'],
                        frozen_B_ce=value['ce'], vs_sbs_pct=value['vs_sbs_pct']))
        cold = {split: {rule: dict(zero_with_winner_pairs=sum(r['zero_with_winner'] for r in methods
                        if r['split'] == split and r['rule'] == rule),
                    zero_winner_count=sum(r['winner_count'] for r in methods if r['split'] == split
                        and r['rule'] == rule and r['zero_with_winner'])) for rule in ('B', 'fitted', 'fallback_B')}
                for split in ('train', 'val')}
        for split, rules in cold.items():
            n = baseline[split]['native_total']
            for value in rules.values():
                value['zero_winner_coverage_pct'] = 100 * value['zero_winner_count'] / n
        outcome = ('exploratory_VAL_pass_pending_formal_release' if passed else
                   'finite_frozen_score_cost_head_trial_closed_negative')
        summary = dict(completed_utc=utc(), outcome=outcome, gate=gate, passed=passed,
            selected_rule='fitted' if passed else 'fallback_B', reports=reports, cold_methods=cold,
            head_lock=lock, optimizer_updates_executed=state['successful_updates'],
            TEST_read=False, LIB_read=False, formal_TEST_LIB_authorized=False, goal_complete=False,
            interpretation='Fixed finite Adam endpoint, not an exact optimum or information-limit certificate')
        write_json(root / 'results.json', summary)
        write_csv(root / 'all18_TRAIN_VAL_metrics.csv', tasks)
        write_csv(root / 'native_method_counts.csv', methods)
        write_csv(root / 'decision_changes.csv', changes)
        write_csv(root / 'decision_change_costs_by_category.csv', categories)
        lines = ['# R78 Round004 Fixed Endpoint', '', 'Outcome: ' + outcome + '.',
                 'B, actual fitted heads and whole-version B fallback are separate. No taskwise mixture.',
                 'CE below is unchanged frozen-B CE, not softmax(-g) or a head-calibration metric.', '',
                 '| Split | Rule | Top1 % | Raw mean cost | Actual regret % | Frozen B CE | Zero/winner pairs |',
                 '|---|---|---:|---:|---:|---:|---:|']
        for split, rules in reports.items():
            for rule, report in rules.items():
                m = report['macro']
                lines.append(f"| {split} | {rule} | {100*m['macro_top1']:.9f} | {m['macro_mean_cost']:.12f} | "
                    f"{m['macro_actual_regret_pct']:.9f} | {m['macro_ce']:.9f} | {cold[split][rule]['zero_with_winner_pairs']} |")
        lines += ['', 'Gate: ' + json.dumps(gate) + '.', 'Selected whole18 rule: ' + summary['selected_rule'] + '.',
                  f"Successful optimizer updates: {state['successful_updates']}; fixed1380 per nondegenerate task, batch1024.",
                  'Native method counts and corrected/harmed/wrong-to-wrong FP64 cost changes are saved as CSV.',
                  'All18 matrices were locked before the only candidate VAL endpoint.',
                  'No TEST/LIB permission, optimum certificate, parameter scan, budget extension or automatic next trial.',
                  'Ongoing file goal remains active. Native/legacy app goal statuses are untouched.']
        (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
        progress(root, 'round004_VAL_pass_waiting_formal_release' if passed else 'round004_closed_negative_waiting_Astra',
            round_number=4, pid=os.getpid(), execution_complete=True, optimizer_updates_executed=state['successful_updates'],
            successful_updates_by_task={p: r['successful_updates'] for p, r in receipts.items()}, parameters=params,
            outcome=outcome, gate=gate, selected_rule=summary['selected_rule'], head_lock=str(root / 'head_lock.json'),
            results=str(root / 'results.json'), report=str(root / 'RESULTS.md'), GPU_used=False,
            TEST_read=False, LIB_read=False, goal_complete=False, native_goal_status_mutated=False,
            next_action='Deliver the one locked fitted candidate for explicit formal release approval.' if passed else
                        'Close this finite head route, retain B and wait for Astra single next direction; no extra training or TEST/LIB.')
        print('[R78 round004 result] ' + json.dumps(dict(outcome=outcome, gate=gate,
            successful_updates=state['successful_updates'], fitted_train=fitted_train['macro'], fitted_val=fitted_val['macro'])), flush=True)
        if tracker is not None:
            tracker.log({'result/passed': int(passed), **{'VAL/' + k: v for k, v in fitted.items()}}, step=state['successful_updates'])
        return summary
    except Exception as error:
        write_json(root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc(), **state))
        progress(root, 'round004_incomplete_attempt_retaining_B', round_number=4, pid=os.getpid(),
            optimizer_updates_executed=state['successful_updates'], error=str(error), GPU_used=False,
            TEST_read=False, LIB_read=False, goal_complete=False, next_action='Preserve incomplete attempt; no automatic retry or extra budget.')
        raise
    finally:
        if tracker is not None:
            tracker.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)


if __name__ == '__main__':
    main()
