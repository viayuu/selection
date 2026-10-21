"""R82 same-lock reports and within-context continuous-regret diagnostics."""

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .r80_reporting import (
    check_identity, file_hash, json_hash, macro_row, metric_row as generic_metric_row,
    nss_comparisons, read_json, reporting_helpers, selected_val_path, utc, write_csv, write_json,
)


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R82_context_matched_ranking'
GROUP = 'B_context_matched_rank_seed2'
MODEL = 'R82B_context_matched_rank_seed2'
LIBS = {'TSPLIB': 'TSP', 'CVRPLIB': 'CVRP'}
WINNER_RULE = 'native/FP64 ind primary; FP32 argmin compatibility only'
CAUSAL_NOTE = 'Historical R77B reference only. Matched R82A not run; no strict A/B causal conclusion.'


def competitive_scores(utility):
    if utility.ndim != 2 or not np.isfinite(utility).all():
        raise ValueError('Need finite unmodified raw utilities [n,k]')
    if utility.shape[1] == 1:
        return np.zeros_like(utility, dtype=np.float64)
    u = utility.astype(np.float64)
    return np.stack([u[:, slot] - np.logaddexp.reduce(np.delete(u, slot, axis=1), axis=1)
                     for slot in range(u.shape[1])], axis=1)


def binary_diagnostics(score, pos, neg):
    """Exact empirical ranking: half credit for ties; ROC never splits a tie."""
    score, pos, neg = np.asarray(score), np.asarray(pos), np.asarray(neg)
    if (score.ndim != 1 or pos.dtype != bool or neg.dtype != bool or score.shape != pos.shape
            or neg.shape != pos.shape or np.any(pos & neg) or not np.isfinite(score[pos | neg]).all()):
        raise ValueError('Invalid diagnostic scores or disjoint applicable P/N masks')
    positives, negatives = score[pos], score[neg]
    p, n = len(positives), len(negatives)
    row = dict(positive_count=p, negative_count=n, auc=None, tpr_fpr_le_1pct=None, tpr_fpr_le_5pct=None,
        fpr_at_1pct_operating_point=None, fpr_at_5pct_operating_point=None,
        applicability='complete' if p and n else 'missing_positive_or_negative')
    for label, values in (('positive', positives), ('negative', negatives)):
        for name, q in (('q01', .01), ('q05', .05), ('q25', .25), ('q50', .5), ('q75', .75), ('q95', .95), ('q99', .99)):
            row[label + '_' + name] = float(np.quantile(values, q)) if len(values) else None
    if not p or not n:
        return row
    ordered_neg = np.sort(negatives)
    lower, upper = (np.searchsorted(ordered_neg, positives, side=side) for side in ('left', 'right'))
    row['auc'] = float(np.mean(lower + .5*(upper-lower))/n)
    values = np.concatenate((positives, negatives))
    labels = np.concatenate((np.ones(p, dtype=np.int64), np.zeros(n, dtype=np.int64)))
    order = np.argsort(-values, kind='stable')
    values, labels = values[order], labels[order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values)-1]
    tp = np.r_[0, np.cumsum(labels)[ends]] / p
    fp = np.r_[0, np.cumsum(1-labels)[ends]] / n
    for suffix, cap in (('1pct', .01), ('5pct', .05)):
        valid = np.flatnonzero(fp <= cap)
        point = valid[np.argmax(tp[valid])]
        row['tpr_fpr_le_'+suffix] = float(tp[point])
        row['fpr_at_'+suffix+'_operating_point'] = float(fp[point])
    return row


def regret_concordance(score, regret):
    """Exact empirical C-index over unequal regrets, half credit for score ties."""
    from scipy.stats import kendalltau
    score, regret = np.asarray(score), np.asarray(regret)
    if score.ndim != 1 or regret.shape != score.shape or not np.isfinite(score).all() or not np.isfinite(regret).all():
        raise ValueError('Concordance needs aligned finite one-dimensional scores and FP64 regrets')
    total = len(score) * (len(score)-1) // 2
    choose = lambda counts: int(np.sum(counts * (counts-1)) // 2)
    tied_r = choose(np.unique(regret, return_counts=True)[1])
    comparable = total - tied_r
    if not comparable:
        return dict(regret_pairs=0, concordance=None, concordant_pairs=0, discordant_pairs=0, tied_score_pairs=0)
    tied_s = choose(np.unique(score, return_counts=True)[1])
    tied_both = choose(np.unique(np.column_stack((score,regret)), axis=0, return_counts=True)[1])
    non_tied = total - tied_r - tied_s + tied_both
    if total == tied_s:
        signed = 0
    else:
        tau = float(kendalltau(regret, -score, variant='b').statistic)
        signed = int(round(tau * np.sqrt(float(comparable) * (total-tied_s))))
    concordant, discordant = (non_tied+signed)//2, (non_tied-signed)//2
    score_ties = tied_s-tied_both
    return dict(regret_pairs=comparable, concordance=float((concordant+.5*score_ties)/comparable),
        concordant_pairs=concordant, discordant_pairs=discordant, tied_score_pairs=score_ties)


def context_metadata(data_root, problem, split, indices):
    with np.load(Path(data_root)/(problem+split)/'input_metadata.npz', allow_pickle=False) as saved:
        metadata = {key:saved[key] for key in ('scale','distribution','capacity_type') if key in saved}
    return {key:value[indices] for key,value in metadata.items()}


def write_cmr_diagnostics(prediction_path, root, split, output):
    from .r82_experiment import native_targets
    from .r82_context import apply_context, relative_regret
    support = read_json(root / 'support/manifest.json')['problems']
    prepared = read_json(root / 'prepared/prepared.json')
    rows, methods, coverage, problems = [], [], [], []
    with np.load(prediction_path, allow_pickle=False) as saved:
        for problem in PROBLEMS:
            get = lambda key: saved[problem + '__' + key]
            utility, logits, costs = (get(key) for key in ('utility', 'logits', 'costs'))
            names, ids = get('pool').tolist(), get('pool_ids').tolist()
            if names != support[problem]['pool'] or ids != support[problem]['pool_ids']:
                raise ValueError('CMR diagnostic candidate axis differs from fixed TRAIN support')
            groups = apply_context(context_metadata(prepared['data_root'], problem, split.lower(), get('indices')),
                support[problem]['context'])
            regrets = relative_regret(costs)
            pos, neg, _ = native_targets(costs)
            score = competitive_scores(utility)
            for slot, name in enumerate(names):
                group_rows = []
                for group in np.unique(groups[groups >= 0]):
                    mask = groups == group
                    row = dict(experiment='R82', model=MODEL, split=split, problem=problem,
                        method=name, solver_id=ids[slot], group_id=int(group), n=int(mask.sum()),
                        **regret_concordance(score[mask,slot],regrets[mask,slot]))
                    conditional = binary_diagnostics(score[mask,slot],pos[mask,slot],neg[mask,slot])
                    row.update({'conditional_winner_'+key:value for key,value in conditional.items()})
                    rows.append(row)
                    group_rows.append(row)
                valid = [row for row in group_rows if row['concordance'] is not None]
                binary = [row for row in group_rows if row['conditional_winner_auc'] is not None]
                methods.append(dict(experiment='R82', model=MODEL, split=split, problem=problem,
                    method=name, solver_id=ids[slot], n=len(costs), matched_context_rows=int((groups>=0).sum()),
                    continuous_group_count=len(valid), regret_pairs=sum(row['regret_pairs'] for row in valid),
                    within_context_regret_concordance=float(np.mean([row['concordance'] for row in valid])) if valid else None,
                    conditional_winner_group_count=len(binary),
                    conditional_winner_auc=float(np.mean([row['conditional_winner_auc'] for row in binary])) if binary else None))
            coverage.append(dict(problem=problem,n=len(costs),matched_context_rows=int((groups>=0).sum()),
                unseen_train_context_rows=int((groups<0).sum()),matched_context_fraction=float(np.mean(groups>=0))))
            local = [row for row in methods if row['problem']==problem and row['within_context_regret_concordance'] is not None]
            conditional = [row for row in methods if row['problem']==problem and row['conditional_winner_auc'] is not None]
            problems.append(dict(problem=problem,
                within_context_regret_concordance=float(np.mean([row['within_context_regret_concordance'] for row in local])) if local else None,
                conditional_winner_auc=float(np.mean([row['conditional_winner_auc'] for row in conditional])) if conditional else None))
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'context_group_diagnostics.csv'
    write_csv(path, rows)
    write_csv(output / 'method_context_diagnostics.csv',methods)
    write_csv(output / 'context_coverage.csv',coverage)
    write_csv(output / 'context_diagnostics_by_problem.csv',problems)
    applicable = [row for row in problems if row['within_context_regret_concordance'] is not None]
    conditional = [row for row in problems if row['conditional_winner_auc'] is not None]
    summary = dict(path=str(path.resolve()), rows=len(rows), method_rows=len(methods),
        label_rule='continuous relative regret from native FP64 costs; lower regret should have higher support',
        score_rule='raw FP32 utility; competitive score calculated in FP64 for offline diagnostics',
        continuous_statistic='within-context regret concordance, NOT global winner AUC; unequal regret pairs only; score ties half credit',
        aggregation='mean nonempty groups within method, mean methods within problem, equal-problem macro',
        conditional_winner_rule='within TRAIN-defined contexts only; unique FP64 minimum P vs strict loser N; tied minima excluded',
        context_policy='apply frozen TRAIN bins/categories unchanged; unseen combinations excluded and counted, no coarse fallback',
        macro_within_context_regret_concordance=float(np.mean([row['within_context_regret_concordance'] for row in applicable])) if applicable else None,
        macro_conditional_winner_auc=float(np.mean([row['conditional_winner_auc'] for row in conditional])) if conditional else None,
        coverage=coverage, selection_or_training_influence=False, split=split)
    write_json(output / 'cmr_diagnostics.json', summary)
    return summary


def rescue_rows(current, historical, methods, split):
    """Native-label rescue and cost deltas on the frozen diagnostic method sets."""
    rows = []
    for problem, data in current.items():
        prior = historical.get(problem)
        if prior is not None:
            if data.pool != prior['pool'].tolist():
                raise ValueError('Historical rescue candidate axis mismatch')
            np.testing.assert_array_equal(data.winner, prior['winner'])
            np.testing.assert_array_equal(data.costs, prior['costs'])
        selected_cost = data.costs[np.arange(data.n), data.pred]
        prior_cost = data.costs[np.arange(data.n), prior['pred']] if prior is not None else None
        for record in (row for row in methods if row['problem'] == problem and row['near_zero']):
            slot = data.pool.index(record['method'])
            won, picked = data.winner == slot, data.pred == slot
            correct, false = won & picked, ~won & picked
            rescue = correct & (prior['pred'] != data.winner) if prior is not None else correct
            both_wrong = (data.pred != data.winner) & (prior['pred'] != data.winner) if prior is not None else None
            relevant = won | picked | (prior['pred'] == slot) if prior is not None else won | picked
            relevant_wrong = both_wrong & relevant if both_wrong is not None else None
            oracle = data.costs.min(1)
            rows.append(dict(model=MODEL, split=split, problem=problem, method=record['method'],
                frozen_set='zero' if record['zero'] else 'near_zero_nonzero', n=data.n,
                native_winner_support=int(won.sum()), selected_count=int(picked.sum()),
                correct_selected_count=int(correct.sum()), true_rescue_count=int(rescue.sum()),
                precision=float(correct.sum()/picked.sum()) if picked.any() else None,
                recall=float(correct.sum()/won.sum()) if won.any() else None,
                false_selection_count=int(false.sum()), false_selection_excess_cost_sum=float((selected_cost-oracle)[false].sum()),
                false_selection_excess_cost_mean=float((selected_cost-oracle)[false].mean()) if false.any() else None,
                false_selection_actual_regret_pct=float(((selected_cost-oracle)/oracle*100)[false].mean()) if false.any() else None,
                false_selection_regret_contribution_pp=float(((selected_cost-oracle)/oracle*100)[false].sum()/data.n),
                both_wrong_relevant_n=int(relevant_wrong.sum()) if prior is not None else None,
                both_wrong_net_cost_saving_sum=float((prior_cost-selected_cost)[relevant_wrong].sum()) if prior is not None else None,
                historical_cost_saving_sum=float((prior_cost-selected_cost)[relevant].sum()) if prior is not None else None,
                historical_comparison='aligned historical R77B '+split if prior is not None else 'missing aligned historical predictions; current descriptive only',
                matched_r82a='not_run', causal_claim=False))
    return rows


def rescue_set_totals(current, historical, methods, split):
    totals = []
    for kind in ('zero', 'near_zero'):
        aggregate = dict(split=split, frozen_set=kind, native_winner_support=0, selected_count=0,
            correct_selected_count=0, true_rescue_count=0, false_selection_count=0,
            false_selection_relative_regret_sum_pct=0., both_wrong_relevant_n=0,
            both_wrong_net_cost_saving_sum=0., historical_pair_available=bool(historical), matched_r82a='not_run')
        for problem, data in current.items():
            names = [row['method'] for row in methods if row['problem']==problem and row[kind]]
            slots = [data.pool.index(name) for name in names]
            won, picked = np.isin(data.winner,slots), np.isin(data.pred,slots)
            correct, false = (data.pred==data.winner)&won, (data.pred!=data.winner)&picked
            prior = historical.get(problem)
            rescue = correct&(prior['pred']!=data.winner) if prior is not None else correct
            selected = data.costs[np.arange(data.n),data.pred]
            oracle = data.costs.min(1)
            for key,mask in (('native_winner_support',won),('selected_count',picked),('correct_selected_count',correct),
                    ('true_rescue_count',rescue),('false_selection_count',false)):
                aggregate[key] += int(mask.sum())
            aggregate['false_selection_relative_regret_sum_pct'] += float(((selected-oracle)/oracle*100)[false].sum())
            if prior is not None:
                relevant = won|picked|np.isin(prior['pred'],slots)
                both_wrong = relevant&(data.pred!=data.winner)&(prior['pred']!=data.winner)
                prior_cost = data.costs[np.arange(data.n),prior['pred']]
                aggregate['both_wrong_relevant_n'] += int(both_wrong.sum())
                aggregate['both_wrong_net_cost_saving_sum'] += float((prior_cost-selected)[both_wrong].sum())
        correct, support, picks = (aggregate[key] for key in ('correct_selected_count','native_winner_support','selected_count'))
        aggregate['recall'] = correct/support if support else None
        aggregate['precision'] = correct/picks if picks else None
        false = aggregate['false_selection_count']
        aggregate['false_selection_actual_regret_pct'] = aggregate['false_selection_relative_regret_sum_pct']/false if false else None
        aggregate['cost_aggregation_note'] = 'raw cost deltas descriptive only across problems; do not infer cross-problem performance'
        if not historical:
            aggregate['both_wrong_relevant_n'] = None
            aggregate['both_wrong_net_cost_saving_sum'] = None
        totals.append(aggregate)
    return totals


def locked_inputs(root):
    from .r82_experiment import MAX_UPDATES
    run = root / GROUP
    config, complete = read_json(run / 'protocol.json'), read_json(run / 'training_complete.json')
    lock = read_json(run / 'final_test_lock.json')
    if (config.get('experiment') != 'R82' or config.get('group') != GROUP
            or lock.get('experiment') != 'R82' or lock.get('group') != GROUP
            or complete.get('experiment') != 'R82' or complete.get('successful_updates') != MAX_UPDATES
            or config.get('max_successful_updates') != MAX_UPDATES or lock.get('state_hash') != json_hash(config)
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False
            or Path(lock['checkpoint']).resolve() != (run / 'best.pt').resolve()
            or file_hash(run / 'best.pt') != lock['checkpoint_sha256']
            or lock.get('rule') != config.get('checkpoint_rule') or lock.get('inference_rule') != config.get('inference_rule')
            or not 0 < lock.get('selected_update', 0) <= MAX_UPDATES):
        raise ValueError('Require completed R82B and unchanged unique VAL checkpoint lock')
    check_identity(complete['selected_best'], lock, 'R82 training completion')
    test, final = read_json(run / 'test_results.json'), read_json(run / 'final_metrics.json')
    if test.get('complete') is not True:
        raise ValueError('Full TEST is incomplete; reporter never starts TEST or training')
    check_identity(test, lock, 'R82 full TEST')
    check_identity(final, lock, 'R82 locked VAL')
    for result in (test['result'], final['val']):
        if set(result['per_problem']) != set(PROBLEMS):
            raise ValueError('Need all 18 problems in VAL and TEST')
    if set(config.get('baselines', {})) != set(PROBLEMS) or any(
            row.get('selection_split') != 'train' for row in config['baselines'].values()):
        raise ValueError('SBS identities must be frozen on FULL TRAIN')
    return run, config, lock, test['result'], final['val']


def metric_row(problem, split, data, saved, config, lock, source):
    row = generic_metric_row(problem, split, data, saved, config['baselines'][problem], config, lock, source)
    row['model'] = MODEL
    row['winner_rule'] = WINNER_RULE
    return row


def load_saved_splits(run, config, lock, test, val):
    helper = reporting_helpers()
    predictions, metrics, methods = {}, [], []
    model_hash = test['identity']['model_state_sha256']
    if val['identity']['model_state_sha256'] != model_hash:
        raise ValueError('VAL/TEST use different model states')
    for split, result, path in (
            ('VAL', val, selected_val_path(run, lock)),
            ('TEST', test, run / 'evaluation/test/selected_best/rank000.npz')):
        identity = read_json(path.parent / 'identity.json')
        if (identity != result['identity'] or identity.get('experiment') != 'R82'
                or identity.get('group') != GROUP or identity.get('split') != split.lower()
                or identity.get('successful_updates') != lock['selected_update']
                or identity.get('state_hash') != lock['state_hash'] or identity.get('model_state_sha256') != model_hash
                or (split == 'TEST' and identity.get('checkpoint_sha256') != lock['checkpoint_sha256'])):
            raise ValueError(f'Invalid locked R82 {split} prediction identity')
        rows = []
        with np.load(path, allow_pickle=False) as saved:
            for problem in PROBLEMS:
                data = helper.load_predictions(saved, path, problem+'__')
                if data.n != result['per_problem'][problem]['n'] or data.n != config['data_manifest'][split.lower()][problem]['rows']:
                    raise ValueError('Incomplete saved prediction population')
                if saved[problem+'__utility'].shape != data.costs.shape:
                    raise ValueError('Missing utility in locked prediction archive')
                np.testing.assert_array_equal(data.indices, np.arange(data.n))
                np.testing.assert_array_equal(data.pool_ids, config['data_manifest'][split.lower()][problem]['pool_ids'])
                np.testing.assert_array_equal(data.pool, config['data_manifest'][split.lower()][problem]['pool'])
                predictions[split, problem] = data
                rows.append(metric_row(problem, split, data, result['per_problem'][problem], config, lock, path))
                methods.extend(dict(row, model=MODEL) for row in helper.method_rows(
                    problem, split, result['per_problem'][problem], data, Path(config['data_root']).name))
        metrics.extend([dict(macro_row(split, rows, result), model=MODEL), *rows])
    return predictions, metrics, methods


def r82_lib(run, config, lock, device_name, batch_size, phase, model_hash):
    import torch
    from .pairwise_objective import inputs_only
    from .r41_frozen_evaluation import attach_geometry
    from .r76_experiment import fixed_sbs_metrics
    from .r77_data import parameter_hash
    from .r77_lib import SCHEMA, checked_predictions, checked_raw, summarize
    from .r82_experiment import load_checkpoint, save_npz
    from .tensor_loader import TensorBatchLoader
    from .train import configure_torch, make_loader
    configure_torch()
    torch.set_num_threads(2)
    device = torch.device(device_name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use CPU or an explicitly assigned GPU for LIB inference')
    directory = run / 'lib_evaluation'
    directory.mkdir(exist_ok=True)
    model, buffer_reference, results = None, None, {}
    for dataset, problem in LIBS.items():
        phase('lib_evaluation', dataset=dataset)
        raw = checked_raw(problem, dataset, config)
        path = directory / 'predictions' / (dataset + '.npz')
        protocol_path = directory / (dataset + '_protocol.json')
        started = time.monotonic()
        identity = dict(experiment='R82', group=GROUP, arm=GROUP, dataset=dataset, problem=problem,
            checkpoint_sha256=lock['checkpoint_sha256'], epoch=lock['epoch'], selected_update=lock['selected_update'],
            state_hash=lock['state_hash'], model_state_sha256=model_hash, candidate_pool=raw['pool'], pool_ids=raw['pool_ids'],
            winner_rule=WINNER_RULE, model_factory='code.V4.r82_experiment.load_checkpoint',
            inference_rule=config['inference_rule'], normalization='all checkpoint TRAIN buffers restored; no refit',
            data_version='original native LIB', batch_size=batch_size, device=str(device), training=False,
            solver_execution=False, embedding_api=False, checkpoint_reselection=False,
            sources={name: dict(path=str(Path(raw['directory'])/name), sha256=file_hash(Path(raw['directory'])/name))
                     for name in ('dataset.pkl', 'raw_label.pkl')},
            prediction_schema=dict(SCHEMA, utility='float32 [n,k], actual raw score-head utilities'))
        if protocol_path.exists():
            previous = read_json(protocol_path)
            if any(previous.get(key) != value for key, value in identity.items() if key != 'device'):
                raise ValueError('Existing LIB predictions differ from R82 model/data/protocol lock')
            if not path.exists() and previous.get('device') != str(device):
                raise ValueError('Retry unfinished LIB inference on its recorded device')
        else:
            if path.exists():
                raise ValueError('Unattributed LIB predictions; refuse reuse')
            write_json(protocol_path, dict(identity, created_utc=utc()))
        reused = path.exists()
        if not reused:
            if model is None:
                model, state = load_checkpoint(run / 'best.pt', device)
                if (state['args'] != config or state['successful_updates'] != lock['selected_update']
                        or state['epoch'] != lock['epoch'] or parameter_hash(state['model']) != model_hash):
                    raise ValueError('R82 factory returned another locked model state')
                del state
                model.eval().requires_grad_(False)
                buffer_reference = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
            with torch.inference_mode():
                loader = make_loader(problem, dataset, batch_size, 0, coord_augment=0, shuffle=False, cache_device=device)
                if loader.size != len(raw['winner']):
                    raise ValueError('LIB loader lost native rows')
                np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), raw['pool_ids'])
                np.testing.assert_array_equal(loader.batch['ind'].cpu(), raw['winner'])
                np.testing.assert_array_equal(loader.batch['costs'].cpu(), raw['costs'].astype(np.float32))
                attach_geometry(loader, device)
                scores, utilities = [], []
                for batch in TensorBatchLoader(loader.batch, batch_size, False, False):
                    output = model(inputs_only(batch))
                    scores.append(output['logits'].float().cpu().numpy())
                    utilities.append(output['utility'].float().cpu().numpy())
                scores = np.concatenate(scores)
                _, pred = fixed_sbs_metrics(scores, raw, config['baselines'][problem])
                data = dict(scores=scores, logits=scores.copy(), utility=np.concatenate(utilities), pred=pred,
                    costs=raw['costs'], winner=raw['winner'], pool=np.asarray(raw['pool']),
                    pool_ids=np.asarray(raw['pool_ids']), nodes=loader.lengths.cpu().numpy(),
                    indices=np.arange(len(pred), dtype=np.int64), winner_fp32_argmin=raw['costs'].astype(np.float32).argmin(1),
                    checkpoint_sha256=np.array(lock['checkpoint_sha256']), checkpoint_epoch=np.array(lock['epoch']),
                    checkpoint_update=np.array(lock['selected_update']))
                temporary = path.with_name(f'.{dataset}.{os.getpid()}.npz')
                try:
                    save_npz(temporary, data)
                    checked_predictions(temporary, raw, lock)
                    for name, value in model.named_buffers():
                        torch.testing.assert_close(value.cpu(), buffer_reference[name], rtol=0, atol=0)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                del loader
        data = checked_predictions(path, raw, lock)
        if data.get('utility', np.empty(0)).shape != raw['costs'].shape or not np.isfinite(data['utility']).all():
            raise ValueError('Missing native LIB utility archive')
        result = dict(**summarize(data, raw, config['baselines'][problem]),
            protocol=read_json(protocol_path), inference_reused=reused, seconds=time.monotonic()-started, completed_utc=utc())
        write_json(directory / (dataset + '_results.json'), result)
        results[dataset] = result
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during LIB inference')
    return results


def page(helper, title, subtitle, lock, number):
    fig = helper.new_page(title, subtitle, number)
    fig.texts[-2].set_text(f"{MODEL} | checkpoint {lock['checkpoint_sha256'][:16]} | native/FP64 ind")
    return fig


def distribution_figure(helper, problem, pool, panels, version, lock):
    fig = helper.plot_distribution(problem, pool, panels, version, lock)
    for text in fig.texts:
        value = text.get_text().replace('R77B', MODEL).replace(helper.SHA256[:16], lock['checkpoint_sha256'][:16])
        value = value.replace('No training, inference, checkpoint reselection or Rxx mixing.',
            'Saved same-lock predictions. No training, checkpoint reselection or Rxx mixing.')
        text.set_text(value)
    return fig


def make_panels(problem, test, lib, canonical_pool):
    sources = [(dataset,lib) for dataset,p in LIBS.items() if p==problem]+[('TEST',test)]
    panels = []
    for split,data in sources:
        if data is None or set(data.pool)!=set(canonical_pool):
            raise ValueError('Distribution requires checked same-lock data and matching candidate names')
        order = [data.pool.index(name) for name in canonical_pool]
        for field,kind in (('pred','model_selection'),('winner','true_winner')):
            counts = np.bincount(getattr(data,field),minlength=len(data.pool))[order]
            if int(counts.sum()) != data.n:
                raise ValueError('Distribution counts must equal actual dataset population')
            panels.append(dict(split=split,kind=kind,n=data.n,counts=counts,top1=data.top1_pct if field=='pred' else None))
    return panels


def historical_test_predictions(path):
    helper = reporting_helpers()
    result = {}
    with np.load(path, allow_pickle=False) as saved:
        for problem in PROBLEMS:
            data = helper.load_predictions(saved, path, problem+'__')
            result[problem] = dict(pred=data.pred, winner=data.winner, costs=data.costs,
                pool=np.asarray(data.pool), pool_ids=data.pool_ids)
    return result


def mechanism_reports(root, run, predictions, lock):
    output = run / 'full_model_statistics'
    cmr = {}
    for split in ('VAL', 'TEST'):
        path = next(data.path for (s, _), data in predictions.items() if s == split)
        cmr[split] = write_cmr_diagnostics(path, root, split, output / split.lower())
    baseline_path = root / 'diagnostic/historical_r77b_val.json'
    baseline = read_json(baseline_path) if baseline_path.exists() else dict(status='missing', methods=[])
    historical = {}
    if baseline.get('status') == 'complete':
        archive = Path(baseline['frozen_archive'])
        if file_hash(archive) != baseline['frozen_archive_sha256']:
            raise ValueError('Frozen historical VAL reference changed')
        with np.load(archive, allow_pickle=False) as saved:
            historical = {problem: {key: saved[problem+'__'+key] for key in ('pred', 'winner', 'costs', 'pool', 'pool_ids')}
                          for problem in PROBLEMS}
    histories = {'VAL':historical,'TEST':{}}
    historical_test = Path(baseline.get('r77_root','/missing_R77'))/'B_full_seed2/evaluation/test/selected_best/rank000.npz'
    if historical_test.is_file():
        histories['TEST'] = historical_test_predictions(historical_test)
    rescue, totals, paired = [], [], []
    for split in ('VAL', 'TEST'):
        current = {p: predictions[split, p] for p in PROBLEMS}
        prior = histories[split]
        rescue.extend(rescue_rows(current, prior, baseline['methods'], split))
        totals.extend(rescue_set_totals(current,prior,baseline['methods'],split))
        for problem,data in current.items():
            if problem not in prior:
                continue
            old = prior[problem]
            np.testing.assert_array_equal(data.winner,old['winner'])
            np.testing.assert_array_equal(data.costs,old['costs'])
            np.testing.assert_array_equal(data.pool_ids,old['pool_ids'])
            chosen = data.costs[np.arange(data.n),data.pred]
            old_cost = data.costs[np.arange(data.n),old['pred']]
            oracle = data.costs.min(1)
            both_wrong = (data.pred!=data.winner)&(old['pred']!=data.winner)
            paired.append(dict(model=MODEL,reference='historical R77B',split=split,problem=problem,n=data.n,
                corrected=int(((data.pred==data.winner)&(old['pred']!=data.winner)).sum()),
                harmed=int(((data.pred!=data.winner)&(old['pred']==data.winner)).sum()),
                both_wrong_cheaper=int((both_wrong&(chosen<old_cost)).sum()),
                both_wrong_dearer=int((both_wrong&(chosen>old_cost)).sum()),
                both_wrong_net_cost_saving_sum=float((old_cost-chosen)[both_wrong].sum()),
                net_mean_cost_change=float((chosen-old_cost).mean()),
                actual_regret_change_pp=float(((chosen-old_cost)/oracle*100).mean()),
                top1_change_pp=float(((data.pred==data.winner).mean()-(old['pred']==data.winner).mean())*100),
                matched_r82a='not_run',causal_claim=False))
    if baseline.get('status')=='complete':
        if totals[0]['native_winner_support']!=baseline['zero_support']:
            raise ValueError('Fixed historical VAL zero-list support changed; do not alter its definition')
    write_csv(output / 'historical_zero_rescue.csv', rescue)
    write_csv(output / 'historical_zero_rescue_totals.csv', totals)
    write_csv(output / 'historical_corrected_harmed.csv',paired)
    return dict(cmr=cmr, rescue_rows=len(rescue), historical_zero_support=baseline.get('zero_support'),
        matched_r82a='not_run', causal_note=CAUSAL_NOTE, rescue_csv=str(output / 'historical_zero_rescue.csv'),
        rescue_totals=totals,paired_corrections=paired)


def build_reports(root, run, config, lock, test, val, lib_results, predictions, metrics, methods, phase, nss_root=None):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from .r80_reporting import diagnostic_pdf
    helper = reporting_helpers()
    helper.setup_style()
    output = run / 'full_model_statistics'
    output.mkdir(exist_ok=True)
    for dataset, problem in LIBS.items():
        path = run / 'lib_evaluation/predictions' / (dataset + '.npz')
        with np.load(path, allow_pickle=False) as saved:
            data = helper.load_predictions(saved, path)
        predictions[dataset, problem] = data
        metrics.append(metric_row(problem, dataset, data, lib_results[dataset]['train_sbs_reference'], config, lock, path))
        methods.extend(dict(row, model=MODEL) for row in helper.method_rows(problem, dataset,
            lib_results[dataset]['train_sbs_reference'], data, 'original native LIB'))
    comparisons, alignment = nss_comparisons(root, config, lock, predictions, nss_root)
    for row in comparisons:
        row['current_model'] = MODEL
        row['winner_rule'] = WINNER_RULE
    distributions, panel_sets = [], {}
    for problem in PROBLEMS:
        data = predictions['TEST', problem]
        dataset = next((key for key, value in LIBS.items() if value == problem), None)
        panels = make_panels(problem, data, predictions.get((dataset, problem)), data.pool)
        panel_sets[problem] = panels
        distributions.extend(dict(row, model=MODEL, checkpoint_sha256=lock['checkpoint_sha256'], winner_rule=WINNER_RULE)
            for row in helper.distribution_rows(problem, panels, data.pool, Path(config['data_root']).name))
    for name, rows in (('metrics_by_problem.csv', metrics), ('nss_comparison.csv', comparisons),
                       ('selection_distribution.csv', distributions), ('single_method_comparison.csv', methods)):
        write_csv(output / name, rows)
    phase('mechanism_diagnostics')
    mechanism = mechanism_reports(root, run, predictions, lock)
    headers = ['Split / Problem', 'Top1 (%)', 'SBS Top1 (%)', 'mean_cost', 'SBS cost', 'Oracle cost',
               'vs_SBS (%)', 'vs_Oracle (%)', 'actual regret (%)']
    pdf_path, distribution_path = output / 'MODEL_STATISTICS.pdf', output / 'selection_distributions.pdf'
    temporary = pdf_path.with_name(f'.{pdf_path.stem}.{os.getpid()}.pdf')
    distribution_tmp = distribution_path.with_name(f'.{distribution_path.stem}.{os.getpid()}.pdf')
    phase('pdf_generation')
    try:
        with PdfPages(temporary, metadata=dict(Title=MODEL+' locked full-model statistics', Author='R82 same-lock reporting')) as pdf:
            fig = page(helper, MODEL+' | Locked Full-Model Statistics',
                'Requested_R80B_context_matched_residual_ranking; native FP64 labels; no TEST/LIB checkpoint or inference reselection.', lock, 1)
            fig.text(.045, .85, f"Seed2 scratch | update {lock['selected_update']} | SHA256 {lock['checkpoint_sha256']}", fontsize=10)
            all_rows = [next(row for row in metrics if row['split'] == split and row['problem'] == 'ALL') for split in ('VAL', 'TEST')]
            helper.add_table(fig, headers, [helper.metric_cells(row, row['split']+' ALL') for row in all_rows], [.045,.65,.91,.13], font=10)
            cells = [[r['dataset'], str(r['n']), 'MISSING' if r['nss_top1_native_pct'] is None else f"{r['nss_top1_native_pct']:.4f}",
                f"{r['current_top1_native_pct']:.4f}", 'MISSING' if r['nss_mean_cost'] is None else f"{r['nss_mean_cost']:.8f}",
                f"{r['current_mean_cost']:.8f}"] for r in comparisons]
            helper.add_table(fig, ['Dataset','n','NSS Top1 (%)','R82B Top1 (%)','NSS mean_cost','R82B mean_cost'], cells, [.045,.35,.91,.23], font=10)
            notes = [CAUSAL_NOTE, f"Frozen R77B VAL zero support: {mechanism['historical_zero_support']}; near-zero <=0.1%; diagnostics only.",
                'SBS identities: FULL TRAIN fit. Oracle: per-instance legal-pool minimum. ALL: equal-problem means.',
                'vs_SBS/vs_Oracle are ratios of means; actual regret is the mean per-instance relative regret.',
                'Earlier TEST results informed multi-round R&D. This is not an untouched blind evaluation.',
                'Missing NSS remains missing; no retraining, no per-dataset Rxx or checkpoint mixing.']
            for index, text in enumerate(notes):
                fig.text(.045, .285-index*.035, text, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            for number, split in enumerate(('VAL', 'TEST'), 2):
                fig = page(helper, MODEL+' | '+split+': ALL + 18 Problems', Path(config['data_root']).name, lock, number)
                rows = [row for row in metrics if row['split'] == split]
                helper.add_table(fig, headers, [helper.metric_cells(row, row['problem']) for row in rows], [.045,.20,.91,.65], font=10, bold_first=True)
                fig.text(.045,.12,'ALL uses equal problem weights; raw cross-problem mean costs are descriptive only.',fontsize=11)
                pdf.savefig(fig)
                plt.close(fig)
            fig = page(helper, MODEL+' | Same Checkpoint On Native LIB',
                'R82 checkpoint factory restores all TRAIN geometry/ID/feature buffers. No historical LIB-score substitution.', lock, 4)
            helper.add_table(fig, headers, [helper.metric_cells(row, f"{row['split']} n={row['n']}") for row in metrics if row['split'] in LIBS],
                [.045,.66,.91,.16],font=10)
            for i, text in enumerate((config['inference_rule'], 'Primary: native/FP64 ind. FP32 argmin is compatibility-only.',
                    'Candidate names/IDs and instance identities are checked before reusing predictions.',
                    'Continuous-regret concordance compares only rows in the same TRAIN-defined context.',
                    'TPR@FPR<=1%,5% uses whole tied-score groups, without interpolation or threshold fitting.',
                    'Utility diagnostics do not change maximin inference. Zero/near-zero sets never influence training.', CAUSAL_NOTE)):
                fig.text(.045,.59-i*.05,text,fontsize=10.5)
            pdf.savefig(fig)
            plt.close(fig)
            with PdfPages(distribution_tmp) as distribution_pdf:
                for problem in PROBLEMS:
                    fig = distribution_figure(helper, problem, predictions['TEST',problem].pool,
                        panel_sets[problem], Path(config['data_root']).name, lock)
                    distribution_pdf.savefig(fig)
                    pdf.savefig(fig)
                    plt.close(fig)
        temporary.replace(pdf_path)
        distribution_tmp.replace(distribution_path)
    finally:
        temporary.unlink(missing_ok=True)
        distribution_tmp.unlink(missing_ok=True)
    original_diagnostics = diagnostic_pdf(run, output, predictions, methods, MODEL, lock)
    cmr_pdf = output / 'cmr_training_and_mechanism.pdf'
    with PdfPages(cmr_pdf) as pdf:
        fig = page(helper, MODEL+' | Base + Context-Matched Ranking', 'Successful updates only; lambda ramp fixed before training.', lock, 1)
        axes = fig.subplots(2,2)
        fig.subplots_adjust(top=.80,bottom=.12,hspace=.35,wspace=.25)
        history_path = run / 'history.json'
        history = read_json(history_path) if history_path.exists() else []
        for ax, keys, title in zip(axes.flat,
                (('loss','base_loss','cmr_loss','auxiliary_loss'), ('cmr_weight','lambda_cmr'), ('cmr_row_coverage','cmr_effective_methods'), ('lr',)),
                ('Loss components','Fixed CMR ramp','CMR coverage / effective methods','Learning rate')):
            for key in keys:
                points = [(row['successful_updates'],row['optimization'][key]) for row in history if key in row.get('optimization',{})]
                if points:
                    ax.plot(*zip(*points),label=key)
            ax.set_title(title)
            if ax.lines:
                ax.legend(fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        fig = page(helper, MODEL+' | Mechanism Evaluation', CAUSAL_NOTE, lock, 2)
        rows = [[split, mechanism['cmr'][split]['macro_within_context_regret_concordance'],
            mechanism['cmr'][split]['macro_conditional_winner_auc'], mechanism['cmr'][split]['rows']] for split in ('VAL','TEST')]
        helper.add_table(fig,['Split','Within-context regret concordance','Conditional winner AUC','Context/method rows'],rows,[.045,.62,.91,.18],font=11)
        for i,text in enumerate(('Continuous regret concordance is not global AUC. Conditional winner AUC is reported separately.',
                'Zero rescue uses native winner agreement, precision, and false-selection excess cost.',
                'Both-wrong net saving = historical cost minus current cost on the same relevant rows.',
                'Unseen TRAIN contexts are counted and excluded; conditions are never silently broadened.',
                'No matched R82A exists. No strict A/B causal claim.')):
            fig.text(.045,.53-i*.055,text,fontsize=11)
        fmt = lambda value: 'N/A' if value is None else f'{100*value:.2f}'
        rescue_cells = [[row['split']+'/'+row['frozen_set'],row['native_winner_support'],row['correct_selected_count'],
            row['true_rescue_count'],fmt(row['recall']),fmt(row['precision']),
            'N/A' if row['false_selection_actual_regret_pct'] is None else f"{row['false_selection_actual_regret_pct']:.3f}"]
            for row in mechanism['rescue_totals']]
        fig.text(.045,.275,'Fixed method-set totals; recall = native correct / native winner support.',fontsize=10)
        helper.add_table(fig,['Split / frozen set','Support','Correct','Rescued','Recall (%)','Precision (%)','False-pick regret (%)'],
            rescue_cells,[.045,.075,.91,.18],font=10)
        pdf.savefig(fig)
        plt.close(fig)
        for number,split in enumerate(('VAL','TEST'),3):
            paired = [row for row in mechanism['paired_corrections'] if row['split']==split]
            fig = page(helper, MODEL+' | '+split+' vs Historical R77B', CAUSAL_NOTE, lock, number)
            if paired:
                cells = [[row['problem'],row['corrected'],row['harmed'],row['both_wrong_cheaper'],
                    row['both_wrong_dearer'],f"{row['top1_change_pp']:+.4f}",
                    f"{row['actual_regret_change_pp']:+.6f}",f"{row['net_mean_cost_change']:+.8f}"] for row in paired]
                helper.add_table(fig,['Problem','Corrected','Harmed','Both wrong / cheaper','Both wrong / dearer',
                    'Top1 change (pp)','Regret change (pp)','Mean cost change'],cells,[.045,.22,.91,.60],font=9)
                macro_top1 = float(np.mean([row['top1_change_pp'] for row in paired]))
                macro_regret = float(np.mean([row['actual_regret_change_pp'] for row in paired]))
                fig.text(.045,.14,f'Equal-problem ALL: Top1 {macro_top1:+.4f} pp; actual regret {macro_regret:+.6f} pp.',fontsize=11)
            else:
                fig.text(.045,.68,'MISSING aligned historical predictions; no paired improvement claim.',fontsize=12)
            fig.text(.045,.10,'Changes = R82B minus R77B; negative regret/cost is better. Historical, not matched A/B.',fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
    missing = [row['dataset'] for row in comparisons if row['status'] != 'complete']
    record = dict(experiment='R82', alias=config.get('alias'), group=GROUP, created_utc=utc(), checkpoint_lock=lock,
        model_factory='code.V4.r82_experiment.load_checkpoint', training=False, checkpoint_reselection=False,
        complete=not missing, missing_nss_datasets=missing, winner_rule=WINNER_RULE,
        data_root=config['data_root'], candidate_order={p:predictions['TEST',p].pool for p in PROBLEMS},
        nss_alignment=alignment, lib_protocols={name:result['protocol'] for name,result in lib_results.items()},
        mechanism=mechanism, causal_note=CAUSAL_NOTE, matched_r82a='not_run', test_not_untouched_blind=True,
        model_statistics_pdf=str(pdf_path), selection_distributions_pdf=str(distribution_path),
        original_diagnostics=original_diagnostics, cmr_diagnostics_pdf=str(cmr_pdf), main_pdf_pages=22, distribution_pages=18)
    write_json(output / 'protocol.json',record)
    return record


def report(args):
    root = args.root.resolve()
    run = root / GROUP
    if not run.is_dir():
        raise FileNotFoundError('R82B has not run; reporter never starts training')
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(experiment='R82', group=GROUP, started_utc=utc(), pid=os.getpid(),
            training_retained=True, training_started_by_reporter=False)

        def phase(name, **fields):
            status.update(phase=name,updated_utc=utc(),**fields)
            write_json(run / 'postprocess_status.json',status)

        try:
            phase('checking_unique_r82_val_lock_and_completed_test')
            run,config,lock,test,val = locked_inputs(root)
            phase('loading_saved_predictions',checkpoint_sha256=lock['checkpoint_sha256'])
            predictions,metrics,methods = load_saved_splits(run,config,lock,test,val)
            libraries = r82_lib(run,config,lock,args.device,args.lib_batch_size,phase,test['identity']['model_state_sha256'])
            result = build_reports(root,run,config,lock,test,val,libraries,predictions,metrics,methods,phase,args.nss_root)
            if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
                raise ValueError('R82 checkpoint changed while reporting')
            phase('complete' if result['complete'] else 'complete_with_missing_NSS',**result,completed_utc=utc())
            print(json.dumps(result,indent=2))
            return result
        except BaseException as error:
            phase('postprocess_failed',failed_stage=status.get('phase'),error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc(),retry='reporting only; retain training/TEST')
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    result.add_argument('--group','--arm',choices=(GROUP,),default=GROUP)
    result.add_argument('--device',default='cpu')
    result.add_argument('--lib-batch-size',type=int,choices=(1,2),default=2)
    result.add_argument('--nss-root',type=Path)
    result.add_argument('--status',action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.status:
        run = args.root / GROUP
        print(json.dumps({name:read_json(run/name) if (run/name).exists() else None
                         for name in ('status.json','postprocess_status.json')},indent=2))
        return 0
    try:
        report(args)
        return 0
    except Exception as error:
        print(f'R82 reporting failed: {error}. Completed training/TEST retained.',file=sys.stderr,flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
