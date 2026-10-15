"""CPU-only native winner counts, using TRAIN caches and saved heldout reports."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', type=Path, default=Path('/tmp/shiys_r75_large_complete_labels_cache/train'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    references = root.parent / 'R65_schedule_free_full18'
    reference = json.loads((references / 'B_schedule_free_seed2/best_eval.json').read_text())
    test = json.loads((references / 'test_results.json').read_text())['models']['B']
    rows, summary, distributions = [], [], {}
    for problem, old_val in reference['val']['per_problem'].items():
        cache = args.train_cache / problem
        metadata = json.loads((cache / 'metadata.json').read_text())
        winner = np.load(cache / 'winner.npy', mmap_mode='r', allow_pickle=False)
        pool = metadata['pool']
        if pool != old_val['pool'] or pool != test['per_problem'][problem]['pool']:
            raise ValueError('Align solver identities by name: ' + problem)
        counts = {
            'big_train': np.bincount(winner, minlength=len(pool)),
            'retained_first10k': np.bincount(winner[:10000], minlength=len(pool)),
            'added_train': np.bincount(winner[10000:], minlength=len(pool)),
        }
        sizes = dict(big_train=len(winner), retained_first10k=10000, added_train=len(winner)-10000)
        distributions[problem] = {name: dict(zip(pool, (value / sizes[name]).tolist()))
                                  for name, value in counts.items()}
        for name, entry in (('old_train', reference['train']['per_problem'][problem]),
                            ('val', old_val), ('test', test['per_problem'][problem])):
            sizes[name] = entry['n']
            distributions[problem][name] = entry['method_top1']
        for column, solver in enumerate(pool):
            row = dict(problem=problem, solver=solver)
            for split in ('old_train', 'retained_first10k', 'added_train', 'big_train', 'val', 'test'):
                fraction = distributions[problem][split][solver]
                row[split + '_n'] = sizes[split]
                row[split + '_winner_count'] = (int(counts[split][column]) if split in counts
                                                else int(round(fraction * sizes[split])))
                row[split + '_winner_pct'] = fraction * 100
            rows.append(row)
        values = distributions[problem]
        summary.append(dict(problem=problem,
            big_train_val_tv_pct=50*sum(abs(values['big_train'][s]-values['val'][s]) for s in pool),
            big_train_test_tv_pct=50*sum(abs(values['big_train'][s]-values['test'][s]) for s in pool),
            added_train_val_tv_pct=50*sum(abs(values['added_train'][s]-values['val'][s]) for s in pool),
            old_train_val_tv_pct=50*sum(abs(values['old_train'][s]-values['val'][s]) for s in pool),
            val_test_tv_pct=50*sum(abs(values['val'][s]-values['test'][s]) for s in pool),
            retained_first10k_winner_counts_match_old=(
                values['retained_first10k'] == values['old_train'])))
    for name, items in (('winner_distribution_comparison.csv', rows),
                        ('winner_distribution_shift.csv', summary)):
        with (root / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='native winner marginal distribution, not R75 model TEST evaluation',
        training_source=str(args.train_cache),
        heldout_source='saved R65 full validation and previously completed TEST reports',
        new_test_data_opened=False, model_forwards=0, solver_runs=0,
        includes_all18_problems=True, solver_rows=len(rows), distributions=distributions, summary=summary)
    (root / 'winner_distribution_analysis.json').write_text(json.dumps(receipt, indent=2) + '\n')
    lines = ['# R75 Native Winner Distribution', '',
        'All values are the percentage of instances whose original native winner is the named solver. '
        'No new TEST inference or solver execution was performed. TEST columns reuse the saved '
        'previously completed original full TEST report. Added TRAIN excludes the first10000 rows.', '',
        'TV is half the sum of absolute solver-frequency differences; it describes marginal '
        'distribution shift, not an accuracy ceiling or evidence of its cause.', '',
        '| Problem | Big TRAIN vs VAL TV % | Big TRAIN vs TEST TV % | Old TRAIN vs VAL TV % | VAL vs TEST TV % |',
        '|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append('| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(
            row['problem'], row['big_train_val_tv_pct'], row['big_train_test_tv_pct'],
            row['old_train_val_tv_pct'], row['val_test_tv_pct']))
    for problem in distributions:
        lines += ['', '## ' + problem, '',
            '| Solver | Old TRAIN % | Added TRAIN % | Big TRAIN % | VAL % | TEST % |',
            '|---|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['problem'] == problem:
                lines.append('| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(
                    row['solver'], *(row[s + '_winner_pct'] for s in
                                     ('old_train', 'added_train', 'big_train', 'val', 'test'))))
    lines += ['', 'Retained-first10000 label frequency counts match the old TRAIN report for '
              + str(sum(r['retained_first10k_winner_counts_match_old'] for r in summary))
              + '/18 tasks. This is a count comparison, not a per-instance provenance audit.', '']
    (root / 'WINNER_DISTRIBUTION.md').write_text('\n'.join(lines))
    print(json.dumps(dict(solver_rows=len(rows), summary=summary,
                         key_distributions={p: distributions[p] for p in
                                            ('TSP', 'VRPTW', 'VRPBTW', 'VRPLTW', 'VRPBLTW')}), indent=2))


if __name__ == '__main__':
    main()
