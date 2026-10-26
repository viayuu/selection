"""Describe fresh paired candidate responses without fitting or selecting models."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def build_tables(manifest, labels):
    if (labels.get('version') != manifest.get('version')
            or labels.get('manifest_sha256') != manifest.get('manifest_sha256')):
        raise ValueError('Labels and paired-input version/manifest must match')
    endpoints = {row['endpoint_id']: row for row in labels['endpoints']}
    planned = {row['pair_id']: row for row in manifest['pairs']}
    if len(planned) != len(manifest['pairs']) or set(planned) != {r['pair_id'] for r in labels['pairs']}:
        raise ValueError('Keep every planned pair, including missing/failed full-pool rows')
    pairs, methods = [], []
    for response in labels['pairs']:
        proposal = planned[response['pair_id']]
        old, new = endpoints[proposal['original_id']], endpoints[proposal['changed_id']]
        if old['base_id'] != new['base_id'] or old['base_id'] != proposal['base_id']:
            raise ValueError('A response pair must retain its common base identity')
        if old['methods'] != new['methods'] or response['methods'] != old['methods']:
            raise ValueError('The full candidate pool cannot change between endpoints')
        valid = np.asarray(old['valid_mask'], bool) & np.asarray(new['valid_mask'], bool)
        complete = bool(valid.all())
        if response['winner_pair_mask'] != complete:
            raise ValueError('A partial candidate pool cannot supply a full-pool winner')
        row = dict(pair_id=response['pair_id'], base_id=proposal['base_id'],
            problem=proposal['problem'], family=proposal['family'], arm=proposal['arm'],
            valid_candidates=int(valid.sum()), candidates=len(valid), full_pool_pair=complete,
            winner_flip=None, original_winner=old['winner'] if complete else None,
            changed_winner=new['winner'] if complete else None,
            rms_delta_pct=None, max_abs_delta_pct=None, common_cost_change_pct=None,
            proposal_l1=float(proposal['l1']), moved_customers=int(proposal['moved_customers']))
        if complete:
            original = np.asarray(old['costs_fp64'], np.float64)
            changed = np.asarray(new['costs_fp64'], np.float64)
            if (not np.isfinite(original).all() or not np.isfinite(changed).all()
                    or min(original.min(), changed.min()) <= 0):
                raise ValueError('Full-pool native costs must be finite and positive')
            scale = float(original.min())
            change = changed - original
            delta = change[:, None] - change[None, :]
            observed = np.asarray(response['Delta_mn'], np.float64)
            if not np.allclose(delta, observed, atol=1e-12, rtol=1e-12):
                raise ValueError('Saved Delta disagrees with native FP64 endpoint costs')
            off_diagonal = delta[np.triu_indices(len(delta), 1)]
            row.update(winner_flip=bool(old['winner'] != new['winner']),
                rms_delta_pct=float(np.sqrt(np.mean(off_diagonal**2)) / scale * 100),
                max_abs_delta_pct=float(np.max(np.abs(off_diagonal)) / scale * 100),
                common_cost_change_pct=float(change.mean() / scale * 100))
            centered = (change - change.mean()) / scale * 100
            for index, method in enumerate(old['methods']):
                methods.append(dict(pair_id=row['pair_id'], base_id=row['base_id'],
                    problem=row['problem'], family=row['family'], arm=row['arm'], method=method,
                    cost_change=float(change[index]), centered_change_pct=float(centered[index]),
                    relative_benefit=bool(centered[index] < 0)))
        pairs.append(row)

    groups = defaultdict(list)
    for row in pairs:
        groups[row['problem'], row['family'], row['arm']].append(row)
    summary = []
    for (problem, family, arm), rows in sorted(groups.items()):
        full = [row for row in rows if row['full_pool_pair']]
        summary.append(dict(problem=problem, family=family, arm=arm, planned_pairs=len(rows),
            planned_bases=len({r['base_id'] for r in rows}), full_pool_pairs=len(full),
            missing_full_pool_pairs=len(rows) - len(full),
            winner_flip_pct=100 * mean(full, 'winner_flip') if full else None,
            mean_rms_delta_pct=mean(full, 'rms_delta_pct'),
            mean_max_abs_delta_pct=mean(full, 'max_abs_delta_pct'),
            mean_common_cost_change_pct=mean(full, 'common_cost_change_pct')))

    matched = defaultdict(dict)
    for row in pairs:
        key = row['problem'], row['family'], row['base_id']
        if row['arm'] in matched[key]:
            raise ValueError('At most one intervention per arm/family/base is allowed')
        matched[key][row['arm']] = row
    matched_rows = []
    for (problem, family, base), arms in sorted(matched.items()):
        full = set(arms) == {'directed', 'random'} and all(r['full_pool_pair'] for r in arms.values())
        row = dict(problem=problem, family=family, base_id=base, matched_full_pool=full,
            rms_delta_difference_pp=None, winner_flip_difference=None, amplitude_ratio=None)
        if full:
            directed, random = arms['directed'], arms['random']
            row.update(rms_delta_difference_pp=directed['rms_delta_pct'] - random['rms_delta_pct'],
                winner_flip_difference=int(directed['winner_flip']) - int(random['winner_flip']),
                amplitude_ratio=directed['proposal_l1'] / random['proposal_l1']
                    if random['proposal_l1'] > 0 else None)
        matched_rows.append(row)
    matched_summary = []
    for problem, family in sorted({(r['problem'], r['family']) for r in matched_rows}):
        rows = [r for r in matched_rows if (r['problem'], r['family']) == (problem, family)]
        full = [r for r in rows if r['matched_full_pool']]
        matched_summary.append(dict(problem=problem, family=family, planned_bases=len(rows),
            matched_full_pool_bases=len(full), mean_rms_delta_difference_pp=mean(full, 'rms_delta_difference_pp'),
            winner_flip_difference_pp=100 * mean(full, 'winner_flip_difference') if full else None,
            mean_amplitude_ratio=mean(full, 'amplitude_ratio')))
    return dict(pairs=pairs, summary=summary, method_responses=methods,
                matched=matched_rows, matched_summary=matched_summary)


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def fmt(value, digits=4):
    return 'MISSING' if value is None else f'{value:.{digits}f}'


def render_pdf(path, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for title, rows, fields in (
            ('Fresh paired candidate responses', tables['summary'],
             ['problem', 'family', 'arm', 'full_pool_pairs', 'missing_full_pool_pairs',
              'winner_flip_pct', 'mean_rms_delta_pct', 'mean_common_cost_change_pct']),
            ('Directed minus random, matched by TRAIN base', tables['matched_summary'],
             ['problem', 'family', 'matched_full_pool_bases', 'mean_rms_delta_difference_pp',
              'winner_flip_difference_pp', 'mean_amplitude_ratio'])):
            for offset in range(0, max(1, len(rows)), 20):
                fig, axis = plt.subplots(figsize=(16, 9))
                axis.axis('off')
                axis.set_title(title, loc='left', fontsize=19, pad=30)
                batch = rows[offset:offset+20]
                values = [[fmt(row.get(key)) if isinstance(row.get(key), float)
                           or row.get(key) is None else str(row[key]) for key in fields] for row in batch]
                if values:
                    table = axis.table(cellText=values, colLabels=[key.replace('_', '\n') for key in fields],
                        loc='center', cellLoc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 1.65)
                else:
                    axis.text(.05, .5, 'No usable rows; missing is not a zero response.', fontsize=16)
                fig.text(.025, .045,
                    'Native full-pool costs only. Percent response uses the ORIGINAL endpoint oracle as a descriptive scale.\n'
                    'No model training, no accuracy claim, no independent-sample claim for related endpoints. '
                    'TSP/ATSP interventions not applicable.', fontsize=9)
                fig.tight_layout(rect=(.02, .11, .98, .94))
                pdf.savefig(fig)
                plt.close(fig)


def analyze(manifest_path, labels_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Keep prior response analysis unchanged: ' + str(output))
    manifest, labels = read(manifest_path), read(labels_path)
    tables = build_tables(manifest, labels)
    output.mkdir(parents=True)
    names = dict(pairs='pair_responses.csv', summary='response_summary.csv',
        method_responses='method_responses.csv', matched='matched_bases.csv',
        matched_summary='directed_vs_random.csv')
    for key, filename in names.items():
        write_csv(output / filename, tables[key])
    render_pdf(output / 'RESPONSE_DIAGNOSIS.pdf', tables)
    full = sum(row['full_pool_pair'] for row in tables['pairs'])
    total = len(tables['pairs'])
    note = ('No full-pool response pairs were established. Missing labels cannot be interpreted as stability.'
            if not full else 'These are descriptive paired-response measurements, not selection improvements.')
    markdown = ('# R114 Paired Response Diagnosis\n\n'
        f'Complete full-pool pairs: **{full}/{total}**. {note}\n\n'
        'Only fixed TRAIN bases were used. Base and intervention costs were freshly solved under the same locked deployment. '
        'A missing candidate masks the full-pool winner; partial candidate deltas never manufacture a winner.\n\n'
        'The main comparison is directed minus random on the SAME problem/family/base intersection. '
        'All candidate differences are in native cost units before descriptive scaling by the original endpoint oracle. '
        'A common shift to all methods yields zero relative response. '
        'Related endpoints/pairs are not counted as independent instances; no significance or causal-internals claim is made.\n\n'
        'No model was trained here. Natural full18 TEST/LIB performance is not inferred from intervention winner flips. '
        'The frozen original dataset and its labels are unchanged.\n')
    (output / 'RESULTS.md').write_text(markdown)
    receipt = dict(experiment='R114_response_diagnosis', complete=True,
        manifest_path=str(Path(manifest_path).resolve()), manifest_sha256=digest(manifest_path),
        labels_path=str(Path(labels_path).resolve()), labels_sha256=digest(labels_path),
        planned_pairs=total, full_pool_pairs=int(full), model_training=False,
        files={p.name: digest(p) for p in output.iterdir() if p.is_file()})
    (output / 'analysis_complete.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.manifest, args.labels, args.output)
    print(json.dumps({k: result[k] for k in ('complete', 'planned_pairs', 'full_pool_pairs')}))


if __name__ == '__main__':
    main()
