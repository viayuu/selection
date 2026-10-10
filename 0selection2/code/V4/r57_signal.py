"""Train-only conditional priors and exact-size validation score permutations."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from code.unified_selector.registry import DATA_ROOT, PROBLEMS
from .multitask_probe import dump
from .pair_specialist import ordinal_scores, stable_order
from .performance_evaluation import decision_metrics
from .performance_targets import read_raw_costs
from .r41_label_stability import proportional_indices
from .r48_common import write_csv
from .r53_data import read_baseline


ROOT = Path('code/V4/runs/R57_task_signal_audit')
SCALARS = ('capacity', 'route_limit', 'speed', 'backhaul_class', 'open_route', 'depot_tw_end')
PSEUDOCOUNT = 20.
METRICS = ('top1', 'top2', 'top3', 'mean_cost', 'actual_regret_pct',
           'vs_sbs_pct', 'vs_oracle_pct', 'vbs_gap_closed_pct')


def load_instances(problem, split):
    if split not in ('train', 'val'):
        raise ValueError('R57 does not read test')
    with (DATA_ROOT / (problem + split) / 'dataset.pkl').open('rb') as stream:
        return pickle.load(stream)


def true_size(problem, instance):
    if problem in ('TSP', 'ATSP'):
        return int(instance.shape[-2])
    return len(instance['node_xy']) if problem not in ('CVRP',) else int(instance['loc'].shape[-2])


def scalar(instance, name):
    if not isinstance(instance, dict) or name not in instance:
        return None
    value = np.asarray(instance[name])
    if value.size != 1:
        raise ValueError(f'{name} is not a global scalar')
    result = float(value.item())
    if not np.isfinite(result):
        raise ValueError(f'Nonfinite input condition {name}')
    return result


def metadata(problem, instances, fields):
    sizes = np.asarray([true_size(problem, x) for x in instances], np.int64)
    conditions = [tuple(scalar(x, name) for name in fields) for x in instances]
    return sizes, conditions


def fit_prior(costs, winner, sizes, conditions, pseudocount=PSEUDOCOUNT):
    gaps = (costs - costs.min(1, keepdims=True)) / costs.min(1, keepdims=True)
    labels = np.eye(costs.shape[1], dtype=np.float64)[winner]
    global_winner, global_gap = labels.mean(0), gaps.mean(0)
    groups, sizes_only = {}, {}
    for i, (n, condition) in enumerate(zip(sizes, conditions)):
        groups.setdefault((int(n), condition), []).append(i)
        sizes_only.setdefault(int(n), []).append(i)

    def summarize(rows):
        return ((labels[rows].sum(0) + pseudocount * global_winner) / (len(rows) + pseudocount),
                (gaps[rows].sum(0) + pseudocount * global_gap) / (len(rows) + pseudocount))

    return dict(global_value=(global_winner, global_gap),
                conditions={key: summarize(rows) for key, rows in groups.items()},
                sizes={key: summarize(rows) for key, rows in sizes_only.items()},
                counts={key: len(rows) for key, rows in groups.items()})


def prior_scores(fitted, sizes, conditions):
    majority, cost, fallbacks = [], [], []
    for n, condition in zip(sizes, conditions):
        key = (int(n), condition)
        if key in fitted['conditions']:
            values, level = fitted['conditions'][key], 'exact_conditions'
        elif int(n) in fitted['sizes']:
            values, level = fitted['sizes'][int(n)], 'size_only'
        else:
            values, level = fitted['global_value'], 'problem_only'
        majority.append(values[0])
        cost.append(-values[1])
        fallbacks.append(level)
    return np.asarray(majority), np.asarray(cost), fallbacks


def deranged_mapping(sizes, rng):
    """Shuffle whole score rows within exact size; singleton rows remain visible."""
    mapping = np.arange(len(sizes))
    eligible = np.zeros(len(sizes), dtype=bool)
    for n in np.unique(sizes):
        rows = np.flatnonzero(sizes == n)
        if len(rows) < 2:
            continue
        shuffled = rng.permutation(rows)
        mapping[shuffled] = np.roll(shuffled, int(rng.integers(1, len(rows))))
        eligible[rows] = True
    return mapping, eligible


def metrics(scores, raw, baseline_cost, subset=None):
    if subset is None:
        subset = np.arange(len(raw['winner']))
    labels = dict(raw, costs=raw['costs'][subset], winner=raw['winner'][subset])
    order = stable_order(scores[subset], raw['pool_ids'])
    result, pred = decision_metrics(ordinal_scores(order), labels)
    selected = labels['costs'][np.arange(len(pred)), pred]
    result['net_cost_vs_R45A'] = float((selected - baseline_cost[subset]).mean())
    result['net_relative_cost_vs_R45A_pp'] = float(((selected - baseline_cost[subset]) /
                                                  labels['costs'].min(1) * 100).mean())
    return result, pred


def summary_rows(rows):
    result = []
    for key in sorted({(row['strategy'], row['scope'], row['repeat']) for row in rows}):
        group = [row for row in rows if (row['strategy'], row['scope'], row['repeat']) == key]
        if len(group) != len(PROBLEMS):
            continue
        values = dict(problem='ALL', strategy=key[0], scope=key[1], repeat=key[2],
                      n=sum(row['n'] for row in group),
                      coverage_pct=float(np.mean([row['coverage_pct'] for row in group])))
        for name in (*METRICS, 'net_cost_vs_R45A', 'net_relative_cost_vs_R45A_pp'):
            values[name] = float(np.mean([row[name] for row in group]))
        result.append(values)
    return result


def run(root=ROOT, repeats=100):
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=2, splits=['train', 'val'], baseline='R45A validation-cost-best; fixed',
        training=False, priors=dict(conditions='problem, exact true size, scalar globals varying in train',
            pseudocount=PSEUDOCOUNT, smoothing='problem-global training distribution',
            fallback=['exact_conditions', 'size_only', 'problem_only'], tie_break='smallest global solver ID'),
        permutation=dict(repeats=repeats, seed=2, strata='problem x exact true size',
            operation='derangement of complete score rows; unchanged singleton rows explicitly reported',
            widening=False, singleton_policy='full-scope metrics keep them unchanged; eligible-only metrics exclude them'),
        metrics='per-problem metrics, ALL is equal-weight 18-problem macro; FP64 raw costs',
        sample=dict(train=16, val=16, seed=2, strata='train-size quartiles with proportional quotas',
                    labels_used=False), no_test=True)
    dump(root / 'signal_protocol.json', protocol)
    raw_val, saved = read_baseline('val', root)
    rows, coverage, input_rows, prior_dump, indices, predictions, shuffle_maps = [], [], [], {}, {}, {}, {}
    for p in PROBLEMS:
        raw_train = read_raw_costs(p, 'train')
        train = load_instances(p, 'train')
        fields = [name for name in SCALARS if len({scalar(x, name) for x in train}) > 1]
        train_sizes, train_conditions = metadata(p, train, fields)
        cuts = np.quantile(train_sizes, [.25, .5, .75]).astype(int)
        val = load_instances(p, 'val')
        val_sizes, val_conditions = metadata(p, val, fields)
        raw = raw_val[p]
        if saved[p]['nodes'] is not None:
            expected = val_sizes + (p not in ('TSP', 'ATSP'))
            np.testing.assert_array_equal(saved[p]['nodes'], expected)
        indices[p] = {}
        for split, instances, sizes, source in (('train', train, train_sizes, raw_train),
                                                 ('val', val, val_sizes, raw)):
            chosen = proportional_indices(sizes, cuts, 16, 2)[0]
            indices[p][split] = dict(indices=chosen.tolist(), true_size=sizes[chosen].tolist(),
                data_hash=source['data_hash'], label_hash=source['label_hash'])
            keys = list(instances[0]) if isinstance(instances[0], dict) else ['matrix' if p == 'ATSP' else 'xy']
            scalar_values = {name: sorted({scalar(x, name) for x in instances}, key=str) for name in SCALARS
                             if isinstance(instances[0], dict) and name in instances[0]}
            input_rows.append(dict(problem=p, split=split, n=len(instances),
                min_size=int(sizes.min()), max_size=int(sizes.max()), distinct_sizes=len(np.unique(sizes)),
                raw_fields=json.dumps(keys), scalar_values=json.dumps(scalar_values),
                varying_train_condition_fields=json.dumps(fields), data_hash=source['data_hash'],
                label_hash=source['label_hash']))
        fitted = fit_prior(raw_train['costs'], raw_train['winner'], train_sizes, train_conditions)
        majority, mean_gap, fallbacks = prior_scores(fitted, val_sizes, val_conditions)
        prior_dump[p] = dict(pool=raw['pool'], pool_ids=list(map(int, raw['pool_ids'])), fields=fields,
            global_majority=fitted['global_value'][0].tolist(), global_mean_gap=fitted['global_value'][1].tolist(),
            groups=[dict(size=key[0], conditions=list(key[1]), n=fitted['counts'][key],
                         winner_probability=value[0].tolist(), mean_gap=value[1].tolist())
                    for key, value in sorted(fitted['conditions'].items(), key=str)],
            validation_fallback_counts={level: fallbacks.count(level) for level in set(fallbacks)})
        baseline_pred = saved[p]['order'][:, 0]
        baseline_cost = raw['costs'][np.arange(len(val)), baseline_pred]
        rng = np.random.default_rng(np.random.SeedSequence([2, PROBLEMS.index(p)]))
        mapping, eligible = deranged_mapping(val_sizes, rng)
        coverage.append(dict(problem=p, n=len(val), eligible=int(eligible.sum()),
            singleton_rows=int((~eligible).sum()), exact_size_coverage_pct=float(eligible.mean() * 100),
            singleton_sizes=','.join(map(str, np.unique(val_sizes[~eligible])))))
        policies = {'R45A': saved[p]['logits'], 'conditional_majority': majority,
                    'conditional_mean_gap': mean_gap}
        per_pred = dict(indices=np.arange(len(val)), true_size=val_sizes, winner=raw['winner'],
                        costs=raw['costs'], pool=np.asarray(raw['pool']), pool_ids=np.asarray(raw['pool_ids']),
                        permutation_eligible=eligible, R45A_scores=saved[p]['logits'])
        maps = []
        for repeat in range(repeats):
            if repeat:
                mapping, _ = deranged_mapping(val_sizes, rng)
            maps.append(mapping)
            policies_now = dict(policies) if repeat == 0 else {}
            policies_now['permuted_R45A'] = saved[p]['logits'][mapping]
            for strategy, scores in policies_now.items():
                for scope, subset in (('full', None), ('permutable_only', np.flatnonzero(eligible))):
                    if subset is not None and not len(subset):
                        continue
                    result, pred = metrics(scores, raw, baseline_cost, subset)
                    rows.append(dict(problem=p, strategy=strategy, scope=scope,
                        repeat=repeat if strategy == 'permuted_R45A' else -1,
                        coverage_pct=float(eligible.mean() * 100),
                        **{name: result[name] for name in ('n', *METRICS, 'net_cost_vs_R45A', 'net_relative_cost_vs_R45A_pp')}))
                    if scope == 'full' and repeat == 0:
                        per_pred[strategy + '_pred'] = pred
        shuffle_maps[p] = np.asarray(maps, np.int32)
        predictions[p] = per_pred
        print(f'[R57 signal] {p}: train={len(train)} val={len(val)} permutation coverage={eligible.mean():.1%}', flush=True)
        del train, val
    rows.extend(summary_rows(rows))
    write_csv(root / 'signal_metrics.csv', rows)
    write_csv(root / 'permutation_coverage.csv', coverage)
    write_csv(root / 'input_inventory.csv', input_rows)
    dump(root / 'conditional_priors.json', prior_dump)
    dump(root / 'audit_indices.json', dict(seed=2, label_stratification=False, datasets=indices))
    folder = root / 'signal_predictions'
    folder.mkdir(exist_ok=True)
    for p, values in predictions.items():
        np.savez_compressed(folder / (p + '.npz'), **values)
    np.savez_compressed(root / 'permutation_maps.npz', **shuffle_maps)
    summary = []
    for p in (*PROBLEMS, 'ALL'):
        for scope in ('full', 'permutable_only'):
            for strategy in ('R45A', 'conditional_majority', 'conditional_mean_gap', 'permuted_R45A'):
                group = [row for row in rows if row['problem'] == p and row['scope'] == scope and row['strategy'] == strategy]
                if not group:
                    continue
                entry = dict(problem=p, scope=scope, strategy=strategy, repeats=len(group), n=group[0]['n'],
                             coverage_pct=group[0]['coverage_pct'])
                for name in (*METRICS, 'net_cost_vs_R45A', 'net_relative_cost_vs_R45A_pp'):
                    values = np.asarray([row[name] for row in group])
                    entry[name] = float(values.mean())
                    if strategy == 'permuted_R45A':
                        entry[name + '_sd'] = float(values.std())
                        entry[name + '_q025'] = float(np.quantile(values, .025))
                        entry[name + '_q975'] = float(np.quantile(values, .975))
                summary.append(entry)
    write_csv(root / 'signal_comparison.csv', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--permutations', type=int, default=100)
    args = parser.parse_args()
    run(args.root, args.permutations)
