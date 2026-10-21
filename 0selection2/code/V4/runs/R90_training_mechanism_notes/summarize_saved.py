"""CPU-only arithmetic on R84A's existing selected TRAIN/VAL archives."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from code.V4.pairwise_selector import maximin_logits


OUT = Path(__file__).resolve().parent
BASE = OUT.parent / 'R84_new90k_attention'


def write_csv(name, rows):
    if rows:
        with (OUT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def softmax(values):
    exp = np.exp(values - values.max(1, keepdims=True))
    return exp / exp.sum(1, keepdims=True)


def tiny_ad():
    rows = []
    for values, target in (([2., 1., 0.], 0), ([2., 1., 0.], 1),
                           ([2., 1., 0.], 2), ([1., 0.], 1)):
        u = torch.tensor([values], dtype=torch.float64, requires_grad=True)
        mask = torch.ones_like(u, dtype=torch.bool)
        z = maximin_logits(u[:, :, None] - u[:, None, :], mask)
        y = torch.tensor([target])
        ce_raw = torch.nn.functional.cross_entropy(u, y)
        ce_z = torch.nn.functional.cross_entropy(z, y)
        grad_raw = torch.autograd.grad(ce_raw, u, retain_graph=True)[0]
        grad_z = torch.autograd.grad(ce_z, u)[0]
        g = z.softmax(-1).detach()[0]
        g[target] -= 1
        t, s = u.detach()[0].argsort(descending=True)[:2].tolist()
        expected = g.clone()
        expected[t], expected[s] = 2 * g[t], g[s] - g[t]
        error = float((expected - grad_z[0]).abs().max())
        assert error < 1e-12
        rows.append(dict(utility=json.dumps(values), target_zero_based=target,
            maximin=json.dumps(z.detach()[0].tolist()), raw_ce=float(ce_raw.detach()),
            maximin_ce=float(ce_z.detach()), raw_probability=json.dumps(u.softmax(-1).detach()[0].tolist()),
            maximin_probability=json.dumps(z.softmax(-1).detach()[0].tolist()),
            raw_utility_ce_gradient=json.dumps(grad_raw[0].tolist()),
            maximin_ce_gradient_wrt_utility=json.dumps(grad_z[0].tolist()),
            formula_max_abs_error=error))
    write_csv('tiny_ad_examples.csv', rows)


def main():
    torch.set_num_threads(1)
    tiny_ad()
    target_doc = json.loads((BASE / 'comparison_artifacts/target_list.json').read_text())
    targets = target_doc['targets']
    assert len(targets) == 36
    target_pairs = {(t['problem'], t['method']) for t in targets}
    summaries, ranks, gradients, identities = [], [], [], {}
    for split in ('train', 'val'):
        directory = BASE / 'A_baseline/evaluation' / split / 'selected_best'
        identity = json.loads((directory / 'identity.json').read_text())
        identities[split] = identity
        with np.load(directory / 'rank000.npz', allow_pickle=False) as saved:
            problems = [key[:-len('__logits')] for key in saved.files if key.endswith('__logits')]
            all_data = {}
            for problem in problems:
                z = saved[problem + '__logits'].astype(np.float64)
                costs = saved[problem + '__costs']
                y = saved[problem + '__winner']
                pool = saved[problem + '__pool'].tolist()
                ids = saved[problem + '__pool_ids']
                by_id = np.argsort(ids, kind='stable')
                order = by_id[np.argsort(-z[:, by_id], axis=1, kind='stable')]
                row = np.arange(len(y))
                rank = np.argsort(order, axis=1)[row, y] + 1
                regret = (costs[row, order[:, 0]] - costs.min(1)) / costs.min(1)
                unique = z[row, order[:, 0]] > z[row, order[:, 1]]
                if z.shape[1] > 2:
                    unique &= z[row, order[:, 1]] > z[row, order[:, 2]]
                # For a unique leader, z determines centered u: leader=0, others=z.
                u = z.copy()
                u[row, order[:, 0]] = 0.
                p, q = softmax(z), softmax(u)
                all_data[problem] = dict(rank=rank, regret=regret, y=y, pool=pool,
                    order=order, unique=unique, p=p, q=q)
        assert len(problems) == 18
        assert len({len(d['y']) for d in all_data.values()}) == 1
        n_per_problem = len(next(iter(all_data.values()))['y'])
        denominator = sum(d['regret'].sum() for d in all_data.values())
        groups = [('ALL_ERRORS', 'ALL', '', 'all', {p: np.ones(len(d['y']), dtype=bool)
                   for p, d in all_data.items()}),
                  ('FIXED_36_TARGETS', 'ALL', '', 'all', {
                      p: np.isin(d['y'], [i for i, m in enumerate(d['pool']) if (p, m) in target_pairs])
                      for p, d in all_data.items()})]
        groups += [('ALL_ERRORS', p, '', 'all', {p: np.ones(len(d['y']), dtype=bool)})
                   for p, d in all_data.items()]
        for target in targets:
            p, method = target['problem'], target['method']
            d = all_data[p]
            groups.append(('FIXED_TARGET', p, method, target['category'],
                           {p: d['y'] == d['pool'].index(method)}))
        for scope, problem, method, category, members in groups:
            r = np.concatenate([all_data[p]['rank'][mask] for p, mask in members.items()])
            c = np.concatenate([all_data[p]['regret'][mask] for p, mask in members.items()])
            missed, ge3 = r > 1, r >= 3
            cost_sum, cost_ge3 = float(c[missed].sum()), float(c[ge3].sum())
            common = dict(split=split, scope=scope, problem=problem, method=method, category=category)
            summaries.append(dict(**common, true_winner_rows=len(r), correct=int((r == 1).sum()),
                missed=int(missed.sum()), rank_ge3=int(ge3.sum()),
                rank_ge3_pct_of_missed=100 * ge3.sum() / max(1, missed.sum()),
                missed_regret_sum=cost_sum, rank_ge3_regret_sum=cost_ge3,
                rank_ge3_pct_of_scope_regret=100 * cost_ge3 / cost_sum if cost_sum else 0.,
                missed_pct_of_all_regret=100 * cost_sum / denominator if denominator else 0.,
                rank_ge3_pct_of_all_regret=100 * cost_ge3 / denominator if denominator else 0.,
                missed_macro_regret_contribution_pctpoints=100 * cost_sum / (18 * n_per_problem),
                rank_ge3_macro_regret_contribution_pctpoints=100 * cost_ge3 / (18 * n_per_problem)))
            max_rank = max(len(all_data[p]['pool']) for p in members)
            for k in range(1, max_rank + 1):
                hit = r == k
                ranks.append(dict(**common, predicted_winner_rank=k, count=int(hit.sum()),
                    pct_of_true_winner_rows=100 * hit.sum() / max(1, len(r)),
                    pct_of_missed=100 * hit.sum() / max(1, missed.sum()) if k > 1 else 0.,
                    regret_sum=float(c[hit].sum()),
                    pct_of_all_regret=100 * c[hit].sum() / denominator if denominator else 0.))
            second_up, winner_up, winner_raw_up = [], [], []
            excluded = 0
            for p, mask in members.items():
                d = all_data[p]
                population = mask & (d['rank'] >= 3)
                excluded += int((population & ~d['unique']).sum())
                keep = np.flatnonzero(population & d['unique'])
                top, second = d['order'][keep, 0], d['order'][keep, 1]
                second_up.extend(d['p'][keep, top] - d['p'][keep, second])
                winner_up.extend(1 - d['p'][keep, d['y'][keep]])
                winner_raw_up.extend(1 - d['q'][keep, d['y'][keep]])
            su, wu, raw = (np.asarray(v) for v in (second_up, winner_up, winner_raw_up))
            ratios = su / wu if len(su) else np.array([])
            wr = wu / raw if len(wu) else np.array([])
            gradients.append(dict(**common, winner_rank_ge3_rows=int(ge3.sum()),
                unique_top_and_second_rows=len(su), ties_excluded=excluded,
                wrong_second_negative_gradient_count=int((su > 0).sum()),
                mean_wrong_second_up_gradient=float(su.mean()) if len(su) else '',
                mean_true_winner_up_gradient=float(wu.mean()) if len(wu) else '',
                wrong_second_vs_winner_abs_gradient_ratio_mean=float(ratios.mean()) if len(ratios) else '',
                wrong_second_vs_winner_abs_gradient_ratio_median=float(np.median(ratios)) if len(ratios) else '',
                wrong_second_vs_winner_abs_gradient_ratio_p90=float(np.quantile(ratios, .9)) if len(ratios) else '',
                true_winner_maximin_vs_reconstructed_raw_ce_abs_gradient_ratio_mean=float(wr.mean()) if len(wr) else ''))
        print(json.dumps(dict(split=split, n_per_problem=n_per_problem,
            summary=summaries[-len(groups)], targets_summary=summaries[-len(groups)+1],
            all_gradient=gradients[-len(groups)], targets_gradient=gradients[-len(groups)+1])))
    assert identities['train']['model_state_sha256'] == identities['val']['model_state_sha256']
    write_csv('winner_rank_summary.csv', summaries)
    write_csv('winner_rank_distribution.csv', ranks)
    write_csv('ce_gradient_summary.csv', gradients)
    (OUT / 'archive_provenance.json').write_text(json.dumps(dict(
        baseline='R84A', identities=identities, target_identity=target_doc['identity'],
        archives={s: str(BASE / 'A_baseline/evaluation' / s / 'selected_best/rank000.npz')
                  for s in ('train', 'val')}, targets=36, inference_executed=False,
        test_or_lib_read=False, gradient_scope='unweighted single-view CE only; centered raw utility reconstructed from maximin logits; ties excluded',
        aggregation='18 tasks equal weight; same rows per task within each split'), indent=2) + '\n')


if __name__ == '__main__':
    main()
