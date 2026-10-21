"""Publish the completed new 90k rows while preserving existing holdouts."""

import argparse
import csv
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS, S2I
from .export_new_split_data import (
    B_PROBLEMS, COORD_DIST, REMOVED, derive_targets, file_hash, filtered_pool,
    train_baselines, utc, write_json, write_pickle,
)
from .export_single_reld_data import single_reld_pool
from .new_split_data import scenario_solver_ids
from .r75_data import _CPUUnpickler, _node_count
from .r75_new_split_trial import proportional_counts, tv_pct, write_csv


SOURCE = Path('/public/home/shiys/\u8bad\u7ec3\u6570\u636e_\u6bcf\u7c7b10\u4e07')
PREVIOUS = Path('/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014')
OUTPUT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
SCENARIO = 'new90k_seed2_B_filtered_MOEL_as_reld_mt_v1'
SIZES = dict(train=81000, val=4500, test=4500)
GENERATION = '\u751f\u6210\u4fe1\u606f.json'
LABEL_COMPLETION = '\u6807\u7b7e\u5b8c\u6210.json'
ACCEPTANCE = '\u5b8c\u6574\u9a8c\u6536.json'
SEED = 2


def read_json(path):
    return json.loads(path.read_text())


def load_pickle(path):
    with path.open('rb') as stream:
        return _CPUUnpickler(stream).load()


def identity(path, expected=None):
    actual = file_hash(path)
    if expected is not None and actual != expected:
        raise ValueError('Source hash mismatch: ' + str(path))
    return dict(path=str(path), bytes=path.stat().st_size, sha256=actual)


def source_inputs(generation):
    n, prefix = int(generation['target_instances']), int(generation['original_instances'])
    if (generation['status'] != 'complete' or n != 100000 or prefix != 10000
            or int(generation['generated_instances']) != n):
        raise ValueError('Expected authoritative complete 100k, including original 10k')
    arrays = dict(scale=np.full(n, -1, dtype=np.int32),
                  coord_dist=np.full(n, -1, dtype=np.int8),
                  distribution=np.full(n, '', dtype='U32'),
                  capacity_type=np.full(n, '', dtype='U32'))
    cursor = prefix
    for shard in generation['shards']:
        start, count = int(shard['start_index']), int(shard['count'])
        stop = start + count
        if start != cursor or count <= 0 or stop > n:
            raise ValueError('Generation shards do not exactly cover the new rows')
        arrays['scale'][start:stop] = int(shard['scale'])
        arrays['distribution'][start:stop] = shard['distribution']
        arrays['capacity_type'][start:stop] = shard['capacity_type']
        arrays['coord_dist'][start:stop] = COORD_DIST[shard['distribution']]
        cursor = stop
    if cursor != n or np.any(arrays['scale'][prefix:] <= 0):
        raise ValueError('Incomplete 90k generation metadata')
    return arrays, prefix


def verify_old_lineage(old, generation):
    if (old['index_mapping'] != 'local_index == original_global_index'
            or int(old['original_instances']) != int(generation['original_instances'])
            or int(old['global_index_start']) != 0
            or int(old['global_index_stop']) != int(old['instances'])):
        raise ValueError('Unsupported previous source index mapping')
    current = {shard['id']: shard for shard in generation['shards']}
    cursor = int(old['original_instances'])
    for shard in old['included_new_shards']:
        other = current.get(shard['id'])
        if other is None or any(shard[key] != other[key] for key in (
                'sha256', 'start_index', 'scale', 'distribution', 'capacity_type', 'seed')):
            raise ValueError('Old and completed generation shards differ')
        count = int(shard['count'])
        if int(shard['start_index']) != cursor or not 0 < count <= int(other['count']):
            raise ValueError('Old shard coverage is not a prefix of the same source')
        cursor += count
    if cursor != int(old['instances']):
        raise ValueError('Old source lineage does not cover its instances')


def release_pool(problem, source_methods):
    filtered, first_kept = filtered_pool(problem, source_methods)
    methods, second_kept, ids = single_reld_pool(filtered)
    return methods, [first_kept[k] for k in second_kept], ids


def extend_splits(inputs, old, prefix, rng, sizes=None):
    sizes = SIZES if sizes is None else sizes
    n = len(inputs['scale'])
    previous = np.concatenate([old[key] for key in ('train', 'val', 'test')])
    if (len(np.unique(previous)) != len(previous) or previous.min() < prefix
            or previous.max() >= n):
        raise ValueError('Previous splits overlap or reference excluded/missing rows')
    used = np.zeros(n, dtype=bool)
    used[:prefix] = True
    used[previous] = True
    unused = np.flatnonzero(~used)
    groups = {}
    for index in unused:
        key = (int(inputs['scale'][index]), str(inputs['distribution'][index]),
               str(inputs['capacity_type'][index]))
        groups.setdefault(key, []).append(int(index))
    keys = sorted(groups)
    capacities = np.array([len(groups[key]) for key in keys])
    nv, nt = sizes['val'] - len(old['val']), sizes['test'] - len(old['test'])
    if nv < 0 or nt < 0 or nv + nt > len(unused):
        raise ValueError('Insufficient previously unused rows for holdout extensions')
    val_counts = proportional_counts(capacities, nv, rng)
    test_counts = proportional_counts(capacities - val_counts, nt, rng)
    extensions = dict(val=[], test=[])
    for column, key in enumerate(keys):
        candidates = rng.permutation(groups[key])
        a, b = int(val_counts[column]), int(test_counts[column])
        extensions['val'].extend(candidates[:a])
        extensions['test'].extend(candidates[a:a + b])
    for split in extensions:
        extensions[split] = rng.permutation(extensions[split]).astype(np.int32)
    heldout = np.concatenate([old['val'], extensions['val'], old['test'], extensions['test']])
    is_holdout = np.zeros(n, dtype=bool)
    is_holdout[heldout] = True
    additions = unused[~is_holdout[unused]]
    splits = dict(
        train=np.concatenate([old['train'], rng.permutation(additions)]).astype(np.int32),
        val=np.concatenate([old['val'], extensions['val']]).astype(np.int32),
        test=np.concatenate([old['test'], extensions['test']]).astype(np.int32))
    all_rows = np.concatenate(list(splits.values()))
    if not np.array_equal(np.sort(all_rows), np.arange(prefix, n)):
        raise ValueError('Splits must partition every new row exactly once')
    if any(len(splits[key]) != sizes[key] for key in sizes):
        raise ValueError('Incorrect new split sizes')
    if np.intersect1d(old['train'], heldout).size:
        raise ValueError('Previously trained examples entered a heldout split')
    return splits, extensions, unused


def check_costs(directory, methods, completion, acceptance):
    labels = load_pickle(directory / 'raw_label.pkl')
    n = int(completion['instances'])
    if n != 100000 or len(labels) != n or any(str(i) not in labels for i in range(n)):
        raise ValueError('Incomplete contiguous native labels: ' + str(directory))
    costs = np.asarray([labels[str(i)]['cost'] for i in range(n)], dtype=np.float64)
    times = np.asarray([labels[str(i)]['time'] for i in range(n)], dtype=np.float64)
    if (costs.shape != (n, len(methods)) or times.shape != costs.shape
            or not np.isfinite(costs).all() or not np.isfinite(times).all()
            or np.any(costs <= 0)):
        raise ValueError('Missing, nonfinite or invalid native cost columns: ' + str(directory))
    if {p.name for p in (directory / 'results').glob('result_*.txt')} != {
            'result_' + name + '.txt' for name in methods}:
        raise ValueError('Native result methods differ from the declared pool')
    checks = {row['method']: row for row in acceptance['label_methods']}
    evidence = {}
    for column, name in enumerate(methods):
        path = directory / 'results' / ('result_' + name + '.txt')
        evidence[name] = identity(path, checks[name]['result_sha256'])
        records = np.loadtxt(path, delimiter=',', dtype=np.float64, ndmin=2)
        if (records.shape[0] != n or records.shape[1] not in (2, 3)
                or not np.array_equal(records[:, 0], np.arange(n))
                or not np.array_equal(records[:, 1], costs[:, column])):
            raise ValueError('Native results/raw label costs disagree: ' + str(path))
    raw_identity = identity(directory / 'raw_label.pkl', completion['raw_label_sha256'])
    return labels, costs, times, dict(raw_label=raw_identity, results=evidence,
        rows=n, finite_cost_values=int(costs.size), cost_dtype='float64')


def inspect_source(source, previous):
    acceptance = read_json(source / ACCEPTANCE)
    if (not acceptance.get('complete') or acceptance.get('data_only') is not False
            or not acceptance.get('full_problem_scope')):
        raise ValueError('The source has not passed full label acceptance')
    accepted = {row['problem']: row for row in acceptance['problems']}
    if set(accepted) != set(PROBLEMS):
        raise ValueError('Acceptance does not cover all 18 tasks')
    parent = read_json(previous / 'scenario_manifest.json')
    original = read_json(previous / 'original_split_manifest.json')
    if parent['status'] != 'complete':
        raise ValueError('Previous release is incomplete')
    scenario_solver_ids(parent)
    plans, saved, extension_saved = {}, {}, {}
    with np.load(previous / 'split_indices.npz', allow_pickle=False) as z:
        for problem_id, problem in enumerate(PROBLEMS):
            directory = source / (problem + 'train')
            generation, completion = read_json(directory / GENERATION), read_json(directory / LABEL_COMPLETION)
            accepted_problem = accepted[problem]
            if (generation['methods'] != completion['methods']
                    or generation['methods'] != accepted_problem['methods']
                    or accepted_problem['instances'] != 100000
                    or accepted_problem['new_instances'] != 90000
                    or accepted_problem['retained_instances'] != 10000
                    or generation['problem'] != problem or completion['problem'] != problem):
                raise ValueError('Native generation/label/acceptance declarations differ: ' + problem)
            inputs, prefix = source_inputs(generation)
            old_metadata_path = Path(original['problems'][problem]['source_metadata'])
            old_metadata = read_json(old_metadata_path)
            verify_old_lineage(old_metadata, generation)
            old = {split: z[problem + '_' + ('validation' if split == 'val' else split)].copy()
                   for split in SIZES}
            for split, values in old.items():
                name = 'validation' if split == 'val' else split
                if (len(values) != parent['split_sizes_per_problem'][split]
                        or hashlib.sha256(values.tobytes()).hexdigest()
                        != original['problems'][problem]['index_sha256'][name]):
                    raise ValueError('Previous split indices changed: ' + problem)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, problem_id]))
            splits, extensions, unused = extend_splits(inputs, old, prefix, rng)
            methods, kept, ids = release_pool(problem, generation['methods'])
            if methods != parent['problems'][problem]['methods']:
                raise ValueError('Candidate rules changed from the previous release')
            for split, values in splits.items():
                saved[problem + '_' + ('validation' if split == 'val' else split)] = values
            for split, values in extensions.items():
                extension_saved[problem + '_' + split] = values
            plans[problem] = dict(
                generation=generation, completion=completion, acceptance=accepted_problem,
                inputs=inputs, splits=splits, extensions=extensions, methods=methods,
                kept=kept, ids=ids, old=old, unused_count=len(unused),
                old_metadata_identity=identity(old_metadata_path),
                generation_identity=identity(directory / GENERATION),
                completion_identity=identity(directory / LABEL_COMPLETION))
    print('SOURCE_METADATA_COMPLETE tasks=18 rows_per_task=100000 new_per_task=90000 '
          'retained_holdouts=3895+3895 extensions=605+605 unused_per_task=12096', flush=True)
    cost_summary = {}
    for problem, plan in plans.items():
        labels, costs, times, evidence = check_costs(source / (problem + 'train'),
            plan['generation']['methods'], plan['completion'], plan['acceptance'])
        plan['cost_evidence'] = evidence
        cost_summary[problem] = dict(rows=len(costs), methods=costs.shape[1],
            finite_cost_values=int(costs.size), retained_candidates=len(plan['methods']))
        print('COST_COMPLETE {} rows={} source_methods={} retained={}'.format(
            problem, len(costs), costs.shape[1], len(plan['methods'])), flush=True)
        del labels, costs, times
        gc.collect()
    return plans, saved, extension_saved, parent, cost_summary


def export_split(destination, instances, costs, times, source_winner, inputs,
                 indices, methods, ids, kept, baseline, problem, split, source_methods):
    destination.mkdir()
    (destination / 'results').mkdir()
    selected = costs[indices][:, kept]
    targets = derive_targets(selected)
    selected_times = times[indices][:, kept]
    raw = {}
    for row in range(len(indices)):
        raw[str(row)] = dict(cost=selected[row].tolist(), time=selected_times[row].tolist(),
            ind=int(targets['winner'][row]), gap=targets['gap_pct'][row].tolist(),
            rank=targets['rank'][row].tolist(), ranking=targets['ranking'][row].tolist(),
            oracle_cost=float(targets['oracle_cost'][row]),
            relative_regret=targets['relative_regret'][row].tolist())
    write_pickle(destination / 'dataset.pkl', [instances[int(i)] for i in indices])
    write_pickle(destination / 'raw_label.pkl', raw)
    del raw
    targets['sbs_train_cost'] = selected[:, baseline['sbs_slot']]
    np.savez_compressed(destination / 'targets.npz', **targets)
    np.savez_compressed(destination / 'input_metadata.npz', source_index=indices,
                        **{key: value[indices] for key, value in inputs.items()})
    for slot, solver in enumerate(methods):
        np.savetxt(destination / 'results' / ('result_' + solver + '.txt'),
            np.column_stack([np.arange(len(indices)), selected[:, slot], selected_times[:, slot]]),
            delimiter=',', fmt=('%d', '%.17g', '%.17g'))
    with (destination / 'index_mapping.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('local_index', 'source_index', 'scale', 'distribution', 'capacity_type'))
        writer.writerows((row, int(index), int(inputs['scale'][index]),
            str(inputs['distribution'][index]), str(inputs['capacity_type'][index]))
            for row, index in enumerate(indices))
    counts = np.bincount(targets['winner'], minlength=len(methods))
    split_sbs = int(selected.mean(axis=0).argmin())
    summary = dict(problem=problem, split=split, instances=len(indices), candidates=len(methods),
        finite_cost_values=int(selected.size), nonfinite_cost_values=0,
        oracle_mean_cost=float(targets['oracle_cost'].mean()),
        train_sbs_solver=baseline['sbs_solver'], train_sbs_mean_cost=float(targets['sbs_train_cost'].mean()),
        train_sbs_actual_regret_pct=float(targets['relative_regret'][:, baseline['sbs_slot']].mean() * 100),
        train_majority_solver=baseline['majority_winner_solver'],
        train_majority_top1_pct=float(counts[methods.index(baseline['majority_winner_solver'])] / len(indices) * 100),
        posthoc_split_sbs_solver=methods[split_sbs],
        posthoc_split_sbs_mean_cost=float(selected[:, split_sbs].mean()),
        exact_min_cost_ties=int(((selected == targets['oracle_cost'][:, None]).sum(axis=1) > 1).sum()),
        winner_changed_from_source_pool=int((np.asarray(kept)[targets['winner']] != source_winner[indices]).sum()))
    metadata = dict(summary, scenario_id=SCENARIO, methods=methods, global_solver_ids=ids,
        source_methods=source_methods, retained_source_slots=kept,
        solver_aliases={'reld_mt': 'RELD_MOEL'} if 'reld_mt' in methods else {},
        cost_dtype='float64', source_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        rank_definition='zero-based ordinal rank; ranking gives best-to-worst candidate slots',
        tie_break='lowest stable global solver ID for exactly equal FP64 costs',
        winner_policy='FP64 argmin in retained pool; stored native ind rebuilt after filtering',
        gap_definition='percent above retained-candidate Oracle; not global-optimum gap',
        time_definition='source compatibility placeholders, not measured runtimes')
    metadata['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p))
                         for p in destination.iterdir() if p.is_file()}
    metadata['result_files'] = {p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p))
                              for p in (destination / 'results').iterdir()}
    write_json(destination / 'metadata.json', metadata)
    return summary, counts


def stratum_counts(problem, inputs, splits, old):
    keys = sorted(set(zip(inputs['scale'][10000:].tolist(),
                          inputs['distribution'][10000:].tolist(), inputs['capacity_type'][10000:].tolist())))
    all_keys = list(zip(inputs['scale'].tolist(), inputs['distribution'].tolist(), inputs['capacity_type'].tolist()))
    from collections import Counter
    counts = {split: Counter(all_keys[int(i)] for i in values) for split, values in splits.items()}
    previous = {split: Counter(all_keys[int(i)] for i in values) for split, values in old.items()}
    return [dict(problem=problem, scale=int(key[0]), distribution=key[1], capacity_type=key[2],
                 **{split + '_n': int(counts[split][key]) for split in SIZES},
                 **{'retained_' + split + '_n': int(previous[split][key]) for split in SIZES}) for key in keys]


def write_report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter

    manifest, stats = read_json(output / 'scenario_manifest.json'), read_json(output / 'release_statistics.json')
    with (output / 'winner_distribution.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 10})
    colors = dict(train='#167e83', val='#c34e58', test='#436fa4')
    with PdfPages(output / 'split_winner_distributions.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        ax.set_title('New90k data release | 18 tasks | seed=2', loc='left', fontsize=19, pad=20)
        lines = [
            'Per task: TRAIN 81,000 | VALIDATION 4,500 | TEST 4,500',
            'Total: TRAIN 1,458,000 | VALIDATION 81,000 | TEST 81,000',
            'The original 10,000 rows are excluded using generation metadata.',
            'All previous 3,895 VAL + 3,895 TEST members are retained in their split.',
            'Each holdout adds 605 input-stratified, previously unused rows.',
            'All previous TRAIN rows remain TRAIN. No solver execution or training.',
            'Candidates: remove RF/MoSES from B tasks; MOEL is reld_mt; MTL removed.',
            'Winner: exact FP64 costs, stable global-ID tie break.',
            'SBS is fitted on TRAIN, then fixed for VAL/TEST.',
            'These figures show true winners only, not model selections or model Top1.',
            'Previous holdouts have been exposed during research; not a new blind test.',
            'B candidate filtering does not certify equivalence of all remaining solvers.',
            '',
            'Mean winner-frequency TV: train/val {:.3f}%, train/test {:.3f}%, val/test {:.3f}%'.format(
                *(stats['distribution_similarity'][key]['mean'] for key in (
                    'train_validation_tv_pct', 'train_test_tv_pct', 'validation_test_tv_pct'))),
        ]
        ax.text(0, .95, '\n\n'.join(lines), va='top', fontsize=11, transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        for problem in PROBLEMS:
            methods = manifest['problems'][problem]['methods']
            by_key = {(row['split'], row['solver']): row for row in rows if row['problem'] == problem}
            fig, ax = plt.subplots(figsize=(12, max(5, .68 * len(methods) + 2.5)))
            positions = np.arange(len(methods))
            for offset, split in zip((-.25, 0, .25), SIZES):
                fractions = np.array([float(by_key[(split, name)]['winner_pct']) for name in methods])
                bars = ax.barh(positions + offset, fractions, height=.23,
                    color=colors[split], label='{} n={:,}'.format(split.upper(), SIZES[split]))
                for bar, name in zip(bars, methods):
                    row = by_key[(split, name)]
                    value = float(row['winner_pct'])
                    label = '{:.2f}% ({:,})'.format(value, int(row['winner_count']))
                    inside = value >= 78
                    ax.text(value - .8 if inside else value + .65,
                        bar.get_y() + bar.get_height() / 2, label, va='center',
                        ha='right' if inside else 'left', color='white' if inside else colors[split], fontsize=9)
            ax.set_yticks(positions)
            ax.set_yticklabels(methods)
            ax.invert_yaxis()
            ax.set_xlim(0, 115)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.xaxis.set_major_formatter(PercentFormatter(100))
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlabel('Share of split instances (%)')
            ax.set_title(problem + ' | true winner distribution across splits\n' +
                         'New90k | native FP64 winner after candidate filtering', loc='left', pad=38)
            ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def build(source, previous, output):
    if output.exists():
        raise FileExistsError('Preserve existing releases; output already exists: ' + str(output))
    started = time.monotonic()
    plans, saved, extensions, parent, cost_summary = inspect_source(source, previous)
    output.mkdir()
    shutil.copy2(previous / 'scenario_manifest.json', output / 'previous_scenario_manifest.json')
    shutil.copy2(previous / 'split_indices.npz', output / 'previous_split_indices.npz')
    shutil.copy2(source / ACCEPTANCE, output / 'source_acceptance.json')
    np.savez_compressed(output / 'split_indices.npz', **saved)
    np.savez_compressed(output / 'holdout_extensions.npz', **extensions)
    names = ['reld_mt' if name == 'RELD_MOEL' else name for name in GLOBAL_SOLVERS if name != 'RELD_MTL']
    identities = {name: S2I['RELD_MOEL' if name == 'reld_mt' else name] for name in names}
    manifest = dict(format_version=1, scenario_id=SCENARIO, status='building', created_utc=utc(),
        seed=SEED, source_data=str(source), previous_data=str(previous),
        previous_manifest_sha256=file_hash(previous / 'scenario_manifest.json'),
        source_acceptance=identity(source / ACCEPTANCE), split_sizes_per_problem=SIZES,
        total_instances=90000 * len(PROBLEMS), global_solvers=names, global_solver_ids=identities,
        model_global_solvers=GLOBAL_SOLVERS, solver_aliases={'reld_mt': 'RELD_MOEL'},
        filtered_problems=sorted(B_PROBLEMS), removed_methods=sorted(set(REMOVED) | {'RELD_MTL'}),
        retained_reld_deployment='RELD_MOEL', reld_removed_from_all_problem_pools='RELD_MTL',
        reserved_inactive_global_solver_ids=[S2I['RELD_MTL']],
        cost_combination="none; retain the chosen deployment's native FP64 costs",
        solver_runs=0, original_files_modified=False, inputs_modified=False, fresh_training_required=True,
        heldout_history='Old 3895-member VAL/TEST sets are development-exposed; not newly blind.',
        remaining_semantic_limits='B filtering does not certify equivalent semantics of remaining deployments.',
        problems={})
    scenario_solver_ids(manifest)
    split_manifest = dict(seed=SEED, source_data=str(source), previous_data=str(previous),
        split_sizes_per_problem=SIZES, total_selected=manifest['total_instances'],
        index_format='split_indices.npz keys <problem>_{train,validation,test}; source-global int32 indices',
        selection='Preserve old split membership and order; add exact input-stratified holdouts from previously unused rows only.',
        labels_used_for_selection=False, split_seed_search=False, problems={})
    for problem, plan in plans.items():
        record = dict(source_directory=str(source / (problem + 'train')), source_instances=100000,
            excluded_old_instances=int(plan['generation']['original_instances']), eligible_new_instances=90000,
            source_metadata=plan['generation_identity'], label_completion=plan['completion_identity'],
            previous_source_lineage=plan['old_metadata_identity'],
            previous_membership_preserved={key: len(plan['old'][key]) for key in SIZES},
            new_holdout_members={key: len(plan['extensions'][key]) for key in ('val', 'test')},
            previously_unused_new_instances=plan['unused_count'], previously_train_in_new_holdouts=0,
            index_sha256={key: hashlib.sha256(values.tobytes()).hexdigest() for key, values in plan['splits'].items()})
        split_manifest['problems'][problem] = record
    write_json(output / 'split_manifest.json', split_manifest)
    write_json(output / 'scenario_manifest.json', manifest)
    write_json(output / 'source_cost_completeness.json', dict(complete=True, problems=cost_summary,
        source_method_columns=sum(row['methods'] for row in cost_summary.values()),
        finite_cost_values=sum(row['finite_cost_values'] for row in cost_summary.values())))
    write_json(output / 'export_command.json', dict(command=[sys.executable, '-m', 'code.V4.r84_data_release',
        'build', '--source', str(source), '--previous', str(previous), '--output', str(output)],
        created_utc=utc(), code_sha256=file_hash(Path(__file__))))
    summaries, winners, similarities, strata, baselines = [], [], [], [], {}
    for problem, plan in plans.items():
        step_started = time.monotonic()
        directory = source / (problem + 'train')
        print('LOAD ' + problem, flush=True)
        dataset_identity = identity(directory / 'dataset.pkl', plan['generation']['output_sha256'])
        instances, labels = load_pickle(directory / 'dataset.pkl'), load_pickle(directory / 'raw_label.pkl')
        if len(instances) != 100000 or len(labels) != 100000:
            raise ValueError('Native row count changed during export: ' + problem)
        costs = np.asarray([labels[str(i)]['cost'] for i in range(len(instances))], dtype=np.float64)
        times = np.asarray([labels[str(i)]['time'] for i in range(len(instances))], dtype=np.float64)
        native_winner = np.asarray([labels[str(i)]['ind'] for i in range(len(instances))], dtype=np.int64)
        if not np.array_equal(native_winner, costs.argmin(axis=1)):
            raise ValueError('Native winner does not match FP64 cost argmin: ' + problem)
        del labels
        lengths = np.fromiter((_node_count(problem, item) for item in instances), dtype=np.int32, count=100000)
        expected = plan['inputs']['scale'][10000:] + (0 if problem in ('TSP', 'ATSP') else 1)
        if not np.array_equal(lengths[10000:], expected):
            raise ValueError('Input scale metadata disagrees with actual node counts: ' + problem)
        baseline = train_baselines(costs[plan['splits']['train']][:, plan['kept']], plan['methods'])
        baselines[problem] = baseline
        fractions = {}
        for split in SIZES:
            summary, counts = export_split(output / (problem + split), instances, costs, times,
                native_winner, plan['inputs'], plan['splits'][split], plan['methods'], plan['ids'],
                plan['kept'], baseline, problem, split, plan['generation']['methods'])
            summaries.append(summary)
            fractions[split] = counts / SIZES[split]
            for slot, method in enumerate(plan['methods']):
                winners.append(dict(problem=problem, split=split, solver=method,
                    global_solver_id=plan['ids'][slot], instances=SIZES[split],
                    winner_count=int(counts[slot]), winner_pct=float(fractions[split][slot] * 100)))
        similarities.append(dict(problem=problem,
            train_validation_tv_pct=tv_pct(fractions['train'], fractions['val']),
            train_test_tv_pct=tv_pct(fractions['train'], fractions['test']),
            validation_test_tv_pct=tv_pct(fractions['val'], fractions['test'])))
        strata.extend(stratum_counts(problem, plan['inputs'], plan['splits'], plan['old']))
        manifest['problems'][problem] = dict(methods=plan['methods'], global_solver_ids=plan['ids'],
            source_methods=plan['generation']['methods'], retained_source_slots=plan['kept'],
            removed_methods=[name for i, name in enumerate(plan['generation']['methods']) if i not in plan['kept']],
            splits={key: problem + key for key in SIZES}, train_sbs=baseline,
            source_dataset=dataset_identity, source_costs=plan['cost_evidence'],
            source_generation=plan['generation_identity'], source_label_completion=plan['completion_identity'])
        write_json(output / 'scenario_manifest.json', manifest)
        write_json(output / 'status.json', dict(phase='exporting', completed_tasks=len(manifest['problems']),
            total_tasks=len(PROBLEMS), last_problem=problem, elapsed_seconds=time.monotonic() - started, updated_utc=utc()))
        print('EXPORTED {} splits=81000/4500/4500 candidates={} seconds={:.1f}'.format(
            problem, len(plan['methods']), time.monotonic() - step_started), flush=True)
        del instances, costs, times, native_winner, lengths
        gc.collect()
    write_json(output / 'baselines.json', baselines)
    write_csv(output / 'split_statistics.csv', summaries)
    write_csv(output / 'winner_distribution.csv', winners)
    write_csv(output / 'distribution_similarity.csv', similarities)
    write_csv(output / 'input_strata.csv', strata)
    statistics = dict(total_instances=manifest['total_instances'], tasks=18, splits=54,
        sizes_per_task=SIZES, sizes_all_tasks={key: size * 18 for key, size in SIZES.items()},
        finite_cost_values=sum(row['finite_cost_values'] for row in summaries), nonfinite_cost_values=0,
        previous_train_moved_to_heldout=0,
        retained_old_holdout_members_per_task=dict(val=3895, test=3895),
        added_holdout_members_per_task=dict(val=605, test=605),
        distribution_similarity={key: dict(mean=float(np.mean([row[key] for row in similarities])),
            maximum=max(row[key] for row in similarities)) for key in (
                'train_validation_tv_pct', 'train_test_tv_pct', 'validation_test_tv_pct')})
    write_json(output / 'release_statistics.json', statistics)
    manifest.update(status='complete', completed_utc=utc(), total_split_directories=54,
        total_problem_solver_combinations=sum(len(row['methods']) for row in manifest['problems'].values()))
    write_json(output / 'scenario_manifest.json', manifest)
    write_report(output)
    lines = ['# Frozen New90k Release', '',
        'Source: ' + str(source), '',
        'Previous release: ' + str(previous), '',
        '18 tasks, each with 81,000 TRAIN / 4,500 VAL / 4,500 TEST (1,620,000 rows total).', '',
        'The original 10,000 instances are excluded using original_instances and the complete generation shard mapping. '
        'Existing TRAIN, VAL and TEST membership and relative order are preserved. '
        'VAL and TEST each add 605 seed=2, scale/distribution/capacity-stratified rows from the '
        '12,096 previously unused new instances. No previous TRAIN row enters a holdout.', '',
        'The previous holdouts have been used during research. The enlarged sets are not a newly blind evaluation.', '',
        'B tasks exclude RouteFinder, MoSES_RF and MoSES_CaDA. All tasks drop RELD_MTL; '
        'RELD_MOEL is retained exactly as reld_mt with original global ID 16. '
        'No best-of-two costs, solver runs, data generation or source modifications occurred. '
        'Remaining B implementations are not certified semantically identical.', '',
        'Each split contains dataset.pkl, raw_label.pkl, targets.npz, results/, input_metadata.npz, '
        'index_mapping.csv and metadata.json with SHA256 hashes. FP64 winner, rank/ranking, Oracle, '
        'relative regret and TRAIN-fitted SBS are recomputed in stable global-ID order. '
        'Native compatibility time values are not measured runtimes; gap is a percent above the retained-pool Oracle.', '',
        'Load using code.V4.new_split_data.load_dataset(problem, split, data_root=THIS_DIRECTORY). '
        'TEST loading still requires the training run final_test_lock.json. '
        'Use a new data cache for this release; never reuse the R76/R77 cache. '
        'Train from fresh initialization and select one checkpoint using VAL before TEST/LIB.', '',
        'Readiness is recorded in readiness.json. Split/winner statistics are in CSV and '
        'split_winner_distributions.pdf (true label frequencies only; no model was evaluated).', '',
        '## Candidate Pools', '', '| Problem | Candidates |', '|---|---|']
    lines.extend('| {} | {} |'.format(problem, ', '.join(row['methods'])) for problem, row in manifest['problems'].items())
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    (output / 'export_code').mkdir()
    for name in ('r84_data_release.py', 'export_new_split_data.py', 'export_single_reld_data.py', 'new_split_data.py'):
        shutil.copy2(Path(__file__).with_name(name), output / 'export_code' / name)
    write_json(output / 'readiness.json', dict(status='ready', data_root=str(output), scenario_id=SCENARIO,
        split_sizes_per_problem=SIZES, total_instances=manifest['total_instances'],
        complete_tasks=18, complete_splits=54, labels_recomputed=True, train_sbs_recomputed=True,
        all_previous_membership_preserved=True, previous_train_moved_to_heldout=0,
        source_files_modified=False, solver_runs=0, training_runs=0,
        loader='code.V4.new_split_data.load_dataset', fresh_training_required=True,
        manifest_sha256=file_hash(output / 'scenario_manifest.json'),
        split_manifest_sha256=file_hash(output / 'split_manifest.json'),
        elapsed_seconds=time.monotonic() - started, completed_utc=utc()))
    write_json(output / 'status.json', dict(phase='complete', completed_tasks=18, total_tasks=18,
        elapsed_seconds=time.monotonic() - started, updated_utc=utc()))
    print('RELEASE_READY ' + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'build', 'report'))
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    source, previous, output = (path.expanduser().resolve() for path in (args.source, args.previous, args.output))
    if args.command == 'report':
        write_report(output)
    elif args.command == 'inspect':
        inspect_source(source, previous)
    else:
        try:
            build(source, previous, output)
        except Exception as error:
            if output.is_dir() and (output / 'export_command.json').exists() and not (output / 'readiness.json').exists():
                write_json(output / 'status.json', dict(phase='failed', error=str(error), updated_utc=utc()))
            raise


if __name__ == '__main__':
    main()
