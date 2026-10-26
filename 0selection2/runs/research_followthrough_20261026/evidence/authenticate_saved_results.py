"""Bounded, CPU-only authentication of this invocation's existing saved results.

No experiment modules, tensors, model forwards, solvers, launches or Git writes.
Only this evidence directory is writable. Native receipts are never rewritten.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import re
import subprocess

REPO = Path('/public/home/shiys/0selection2')
GIT = REPO.parent
HERE = Path(__file__).resolve().parent
MONITOR = HERE.parent
ACTION = 'e1021815bf3f51b6f34424bf'
EXPECTED_GATE = '4f74c4802781fff805a62c709c377baa58feabc1315d297a3a764aa95444fd89'
DATA = 'selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018'
ROOTS = {'R109': 'R109_method_conditioned_nodes', 'R112': 'R112_region_interfaces',
         'R113': 'R113_executable_process', 'R114': 'R114_paired_responses'}
ARMS = {'R109': ['A_tied_late', 'B_private_late'],
        'R112': ['A_collapsed', 'B_interfaces'],
        'R113': ['A_scratch', 'B_snapshot', 'C_transition']}
NUMBERS = ['top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct',
           'vs_oracle_pct', 'actual_regret_pct']
CHECKPOINT_COLUMNS = ['experiment', 'arm', 'checkpoint_sha256', 'state_hash',
                      'selected_update', 'selection_split', 'inference_rule']
METRIC_COLUMNS = ['experiment', 'arm', 'split', 'problem', 'n', 'checkpoint_sha256',
                  *NUMBERS, 'data_version', 'candidate_pool_sha256', 'winner_rule',
                  'sbs_fit_split', 'all_aggregation']
INPUTS = {}
FAILURES = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def remember(path, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected is not None:
        assert actual == expected, ('hash mismatch', str(path), actual, expected)
    INPUTS[str(path)] = actual
    return actual


def read(path):
    remember(path)
    return json.loads(Path(path).read_text())


def rows(path):
    remember(path)
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def command(argv, *, binary=False):
    return subprocess.run(argv, cwd=GIT, capture_output=True, text=not binary,
                          check=True, timeout=60).stdout


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n')


def write_csv(name, columns, values):
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def equal(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), rel_tol=2e-10, abs_tol=2e-9), (a, b)


def pdf(path):
    remember(path)
    info = command(['pdfinfo', str(path)])
    text = command(['pdftotext', '-layout', str(path), '-'])
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
    assert pages and text.strip(), ('unreadable PDF', str(path))
    return {'path': str(path), 'sha256': INPUTS[str(Path(path).resolve())],
            'pages': pages, 'extracted_characters': len(text),
            'text_sha256': hashlib.sha256(text.encode()).hexdigest()}, text


def failed(path, expected, kind, *, message=None):
    digest = remember(path, expected)
    v = read(path)
    if kind == 'keyboard_interrupt':
        assert v.get('error_type') == 'KeyboardInterrupt'
    elif kind == 'r109_publication':
        assert v.get('state') == 'stopped_on_error' and v.get('exit_status') == 1
    elif kind == 'r114_interface':
        assert 'can\'t convert cuda:0 device type tensor to numpy' in json.dumps(v)
        assert v.get('status') != 'complete' and v.get('state') != 'complete'
    else:
        assert v.get('state') in ('failed', 'blocked') or v.get('exit_status', 0) != 0
    FAILURES.append({'path': str(Path(path).resolve()), 'sha256': digest,
                     'kind': kind, 'state': v.get('state', v.get('status')),
                     'error_type': v.get('error_type'),
                     'historical_hash_anchor': message or 'native saved receipt',
                     'unchanged_before_after': True})


def main():
    control = MONITOR / 'control.json'
    if control.exists():
        assert read(control).get('paused') is not True, 'Explicit new monitor pause'
    request = read(MONITOR / 'actions' / ACTION / 'request.json')
    assert request['model'] == 'gpt-6.1-sol' and request['reasoning_effort'] == 'max'
    context = request['context']
    snapshot = read(HERE / 'source_snapshot.json')
    assert canonical(snapshot) == EXPECTED_GATE == context['source_snapshot_sha256']
    for path, digest in snapshot.items():
        remember(path, digest)
    observation = context['observation']
    assert observation['complete'] is True
    cli = read(MONITOR / 'actions' / ACTION / 'codex.json')
    args = cli['argv']
    assert args[args.index('-m') + 1] == 'gpt-6.1-sol'
    assert 'model_reasoning_effort="max"' in args
    assert 'approval_policy="never"' in args and 'danger-full-access' in args

    # Compare every small published manifest asset with both working bytes and
    # the actual named commit tree. Read operations only, no index changes.
    publications = []
    publication_issues = []
    for version in ROOTS:
        task = observation['tasks'][version]
        receipt = task['publication']
        commit = receipt['commit']
        root = REPO / 'code/V4/runs' / ROOTS[version]
        if version == 'R109':
            manifest = root / 'preparation/publication_recovery_20261026/publication_manifest.json'
            m = read(manifest)
            hashes = {x['path']: x['sha256'] for x in m['records']}
        else:
            manifest = (Path(task['finish_dir']) / 'publication_manifest.json'
                        if version == 'R114' else root / 'publish_manifest.json')
            m = read(manifest)
            hashes = m['sha256']
        hashes = dict(hashes)
        hashes[str(manifest.relative_to(GIT))] = sha(manifest)
        records = []
        for relative, digest in hashes.items():
            p = GIT / relative
            remember(p, digest)
            data = command(['git', 'show', f'{commit}:{relative}'], binary=True)
            actual = hashlib.sha256(data).hexdigest()
            record = {'path': relative, 'working_sha256': digest,
                      'git_blob_sha256': actual, 'git_blob_bytes': len(data),
                      'working_bytes': p.stat().st_size,
                      'exact_bytes_match': actual == digest}
            if actual != digest:
                local = p.read_bytes()
                record['CRLF_to_LF_only'] = local.replace(b'\r\n', b'\n') == data
                publication_issues.append(dict(experiment=version, commit=commit, **record))
            records.append(record)
        parent = command(['git', 'show', '-s', '--format=%P', commit]).strip()
        expected_parent = receipt.get('parent_commit', receipt.get('parent'))
        assert parent == expected_parent
        publications.append({'experiment': version, 'commit': commit, 'parent': parent,
                             'manifest': str(manifest), 'manifest_sha256': sha(manifest),
                             'verified_asset_count': len(records),
                             'verified_asset_bytes': sum(x['git_blob_bytes'] for x in records),
                             'assets': records})
    remote = command(['git', 'ls-remote', '--exit-code', 'origin', 'refs/heads/main']).strip()
    remote_head = remote.split()[0]
    for pub in publications:
        command(['git', 'merge-base', '--is-ancestor', pub['commit'], remote_head])
    dump('publication_byte_verification.json', {
        'action_id': ACTION, 'complete': not publication_issues,
        'strict_published_bytes_pass': not publication_issues,
        'origin_main_checked': remote_head, 'publications': publications,
        'mismatches': publication_issues,
        'original_artifacts_modified': False, 'git_mutations': False})

    locks, metrics, checks, zeroes, nss, pdfs, summary = [], [], [], [], [], [], []
    expected_problems = None
    for version, arms in ARMS.items():
        root = REPO / 'code/V4/runs' / ROOTS[version]
        remember(root / 'RESULTS.md')
        if (root / 'science_boundary.json').exists():
            read(root / 'science_boundary.json')
        for arm in arms:
            run = root / arm
            lock = read(run / 'final_test_lock.json')
            assert lock == observation['tasks'][version]['locks'][arm]
            assert lock['selected_without_test'] is True and lock['per_problem_assembly'] is False
            remember(run / 'best.pt', lock['checkpoint_sha256'])
            protocol = read(run / 'protocol.json')
            assert canonical(protocol) == lock['state_hash']
            pools = protocol['candidate_pools']
            problems = list(pools)
            if expected_problems is None:
                expected_problems = set(problems)
            assert len(problems) == 18 and set(problems) == expected_problems
            for problem, pool in pools.items():
                assert 'RELD_MTL' not in pool and 'RELD_MOEL' not in pool
                if 'B' in problem:
                    assert not set(pool) & {'RouteFinder', 'MoSES_RF', 'MoSES_CaDA'}
            final = read(run / 'final_metrics.json')
            assert final['complete'] is True
            for k in ['checkpoint_sha256', 'state_hash', 'selected_update']:
                assert final[k] == lock[k]
            table = rows(run / 'full_model_statistics/metrics_by_problem.csv')
            indices = {(r['split'], r['problem']): r for r in table}
            assert len(indices) == len(table) == 59
            required = {(s, p) for s in ['VAL', 'TEST'] for p in [*problems, 'ALL']}
            required |= {('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')}
            assert len(required) == 40 and required <= set(indices)
            locks.append({k: lock[k] for k in CHECKPOINT_COLUMNS if k != 'selection_split'})
            locks[-1]['selection_split'] = 'VAL'
            for split in ['VAL', 'TEST']:
                saved = final[split.lower()]
                assert 'TRAIN' in saved['sbs_reference']
                assert 'FP64' in lock['winner_rule'] and 'never FP32' in lock['winner_rule']
                for problem in problems:
                    original = saved['per_problem'][problem]
                    row = indices[split, problem]
                    assert int(row['n']) == original['n'] == 4500
                    for metric in NUMBERS:
                        source = original['top1'] * 100 if metric == 'top1_pct' else original[metric]
                        equal(row[metric], source)
                    assert json.loads(row['candidate_pool']) == pools[problem]
                    train = final['train']['per_problem'][problem]
                    train_costs = train['method_mean_cost']
                    best = min(train_costs.values())
                    assert train_costs[row['sbs_name']] <= best + 1e-12
                    assert row['sbs_name'] == original['sbs_name']
                    equal(row['vs_sbs_pct'], 100 * (float(row['mean_cost']) / float(row['sbs_cost']) - 1))
                    equal(row['vs_oracle_pct'], 100 * (float(row['mean_cost']) / float(row['oracle_cost']) - 1))
                allrow = indices[split, 'ALL']
                assert int(allrow['n']) == 81000
                for metric in NUMBERS:
                    equal(allrow[metric], sum(float(indices[split, p][metric]) for p in problems) / 18)
                    mk = 'macro_top1' if metric == 'top1_pct' else 'macro_' + metric
                    equal(allrow[metric], saved['macro'][mk] * (100 if metric == 'top1_pct' else 1))
                for k, v in lock['val'].items():
                    equal(final['val']['macro'][k], v)
            for split, problem in sorted(required):
                source = indices[split, problem]
                assert source['checkpoint_sha256'] == lock['checkpoint_sha256']
                assert 'TRAIN' in source['sbs_selection_split']
                if problem != 'ALL':
                    assert json.loads(source['candidate_pool']) == pools[problem]
                row = {k: source[k] for k in ['split', 'problem', 'n', *NUMBERS]}
                row.update(experiment=version, arm=arm,
                           checkpoint_sha256=lock['checkpoint_sha256'],
                           data_version=DATA if split in ['VAL', 'TEST'] else 'native_' + split,
                           candidate_pool_sha256=canonical(pools), winner_rule=lock['winner_rule'],
                           sbs_fit_split='TRAIN',
                           all_aggregation='equal_problem' if problem == 'ALL' else 'single_problem')
                if split in ['VAL', 'TEST']:
                    assert source['data_version'] == DATA
                for metric in NUMBERS:
                    assert math.isfinite(float(row[metric]))
                metrics.append(row)
            distribution = rows(run / 'full_model_statistics/selection_distribution.csv')
            grouped = collections.defaultdict(dict)
            for row in distribution:
                assert row['checkpoint_sha256'] == lock['checkpoint_sha256']
                assert 'native' in row['winner_rule'] and 'FP64' in row['winner_rule']
                key = row['split'], row['problem'], row['panel']
                assert row['candidate'] not in grouped[key]
                grouped[key][row['candidate']] = row
            for (split, problem, panel), values in grouped.items():
                assert panel in ['model_selection', 'true_winner']
                assert set(values) == set(pools[problem])
                n = int(indices[split, problem]['n'])
                assert sum(int(r['count']) for r in values.values()) == n
                for order, name in enumerate(pools[problem]):
                    row = values[name]
                    assert int(row['candidate_order']) == order and int(row['n']) == n
                    equal(row['percentage'], 100 * int(row['count']) / n)
                    if panel == 'model_selection':
                        equal(row['model_top1_native_pct'], indices[split, problem]['top1_pct'])
            assert len(grouped) == 40
            for problem in problems:
                chosen = grouped['TEST', problem, 'model_selection']
                winners = grouped['TEST', problem, 'true_winner']
                for name in pools[problem]:
                    if int(chosen[name]['count']) == 0 and int(winners[name]['count']) > 0:
                        zeroes.append({'experiment': version, 'arm': arm, 'problem': problem,
                                       'candidate': name, 'selected_count': 0,
                                       'native_winner_count': int(winners[name]['count']), 'n': 4500})
            nr = rows(run / 'full_model_statistics/nss_comparison.csv')
            assert len(nr) == 4 and {r['dataset'] for r in nr} == {'TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB'}
            for row in nr:
                assert row['status'] == 'complete'
                assert row['current_checkpoint_sha256'] == lock['checkpoint_sha256']
                if 'current_state_hash' in row:
                    assert row['current_state_hash'] == lock['state_hash']
                if 'current_selected_update' in row:
                    assert int(row['current_selected_update']) == lock['selected_update']
                split = 'TEST' if row['dataset'].endswith(' TEST') else row['dataset']
                current = indices[split, row['problem']]
                equal(row['current_top1_native_pct'], current['top1_pct'])
                equal(row['current_mean_cost'], current['mean_cost'])
                assert json.loads(row['candidate_pool']) == pools[row['problem']]
                assert int(row['n']) == int(current['n'])
                nss.append(row)
            stats, text = pdf(run / 'full_model_statistics/MODEL_STATISTICS.pdf')
            assert arm in text and lock['checkpoint_sha256'][:16] in text
            pdfs.append(stats)
            stats, text = pdf(run / 'full_model_statistics/selection_distributions.pdf')
            assert stats['pages'] == 18 and '真实最优方法' in text and '模型选择' in text
            assert arm in text and lock['checkpoint_sha256'][:16] in text
            labels = re.findall(r'模型Top1\s*=\s*([0-9.]+)%', text)
            expected_labels = [f"{float(indices[s, p]['top1_pct']):.2f}" for s, p in required
                               if s in ['TSPLIB', 'CVRPLIB'] or s == 'TEST' and p != 'ALL']
            assert sorted(labels) == sorted(expected_labels), (version, arm, len(labels))
            stats['native_top1_labels_checked'] = len(labels)
            pdfs.append(stats)
            history = read(run / 'history.json')
            assert history
            best = min(history, key=(lambda h: (h['val']['macro']['macro_vs_sbs_pct'], h['successful_updates']))
                       if version == 'R109' else
                       (lambda h: (-h['val']['macro']['macro_top1'],
                                   h['val']['macro']['macro_actual_regret_pct'], h['successful_updates'])))
            assert best['successful_updates'] == lock['selected_update']
            for split in ['TRAIN', 'VAL', 'TEST']:
                source = indices[split, 'ALL']
                summary.append({'experiment': version, 'arm': arm, 'split': split,
                                **{k: source[k] for k in NUMBERS}})
            checks.append({'experiment': version, 'arm': arm,
                           'checkpoint_sha256': lock['checkpoint_sha256'],
                           'checkpoint_file_bytes': (run / 'best.pt').stat().st_size,
                           'protocol_state_hash_verified': True,
                           'selected_update': lock['selected_update'],
                           'prelocked_VAL_criterion_reproduced_from_saved_history': True,
                           'candidate_pool_sha256': canonical(pools),
                           'same_checkpoint_metric_rows': 40,
                           'TRAIN_SBS_method_choices_verified': 18,
                           'selection_and_true_winner_groups_verified': 40})

    # Preserve original failure meanings and historical hash anchors.
    r109 = REPO / 'code/V4/runs' / ROOTS['R109']
    path = r109 / 'goal_watch_receipt.json'
    failed(path, snapshot[str(path)], 'r109_publication', message='source_snapshot; original failed watcher')
    recovery = read(REPO / 'code/V4/runs' / ROOTS['R112'] / 'preparation/recovery_20261026/recovery_complete.json')
    assert recovery['state'] == 'complete' and recovery['exit_status'] == 0
    assert recovery['gpu_compute'] is False and recovery['model_inference'] is False
    for arm in ARMS['R112']:
        path = REPO / 'code/V4/runs' / ROOTS['R112'] / 'preparation/dispatch' / (arm + '.json')
        failed(path, recovery['inputs'][str(path)], 'r112_report', message='CPU recovery input SHA256')
        dispatch = read(path)
        assert all(x['exit_status'] == 0 for x in dispatch['stages'] if x['name'] in ['train', 'test', 'timing', 'lib'])
        assert dispatch['stages'][-1]['name'] == 'report' and dispatch['stages'][-1]['exit_status'] != 0
    r113 = REPO / 'code/V4/runs' / ROOTS['R113']
    ready = read(r113 / 'preparation/migration_3090_portable_inode_20261026/ready.json')
    assert ready['main_approved'] is True
    completion = read(r113 / 'finish_complete.json')
    for arm in ARMS['R113']:
        job = completion['completion'][arm]
        assert job['state'] == 'complete' and job['allocation'] == 1465
        assert job['gpu_uuid'] in context['allowed_gpu_uuids'] and job['actual_gpu_type'] == '3090'
        path = Path(job['attempt_dir']) / 'worker.json'
        worker = read(path)
        remember(path, job['new_worker_sha256'])
        assert worker['state'] == 'complete' and worker['exit_status'] == 0
        assert worker['checks']['gpu']['uuid'] == job['gpu_uuid']
        assert 'RTX 3090' in worker['checks']['gpu']['name']
        if arm != 'A_scratch':
            link = ready['links'][arm]
            assert link['checkpoint_update'] == 20160
            assert all(link[x + '_retained'] for x in ['optimizer', 'scaler', 'rng', 'stream'])
            assert link['unpersisted_updates_at_signal'] == (2286 if arm == 'B_snapshot' else 1872)
            failed(link['source_worker'], link['old_worker_sha256'], 'r113_interrupted_worker', message='main-approved ready.json old_worker_sha256')
            path = r113 / arm / 'error.json'
            failed(path, None, 'keyboard_interrupt', message='read-only start/end byte comparison; historical worker independently hash-anchored')
    old_controller = r113 / 'preparation/migration_3090_20261026/controller.json'
    frozen = MONITOR / 'actions/a853b6e63ff2a817b151fa60/r113_repair_inspection/original_snapshots/queue_controller.json'
    failed(old_controller, sha(frozen), 'r113_old_controller', message=str(frozen))

    r114 = REPO / 'code/V4/runs' / ROOTS['R114'] / 'preparation'
    revision = read(r114 / 'tensor_boundary_recovery_20261026/revision.json')
    original = revision['original_failure']
    failed(original['path'], original['sha256'], 'r114_interface', message='pre-correction revision.json original_failure.sha256')
    assert revision['status'] == 'prepared_not_integrated_not_executed'
    finish = Path(observation['tasks']['R114']['finish_dir'])
    closeout = read(finish / 'closeout.json')
    assert closeout['kind'] == 'complete_labels' and closeout['labeling_complete'] is True
    assert closeout['trained_function_learner'] is False and closeout['model_training'] is False
    assert closeout['all18_model_performance_claim'] is False and closeout['TSP_ATSP_response_claim'] is False
    assert len(closeout['execution_lock']) == 16 and closeout['lock_sha256'] is None
    assert closeout['parent_scheduler_status'] == 'blocked' and closeout['child_scheduler_status'] == 'complete'
    terminal = read(finish / 'finish_complete.json')
    for name, digest in terminal['outputs'].items():
        remember(finish / name, digest)
    provenance = read(finish / 'input_provenance.json')
    for branch in provenance['execution_provenance'].values():
        for name in ['full_manifest_file', 'labels_file', 'progress_file']:
            remember(branch[name]['path'], branch[name]['sha256'])
    for path, digest in terminal['inputs'].items():
        if '/attempts/runtime_group__CVRP__ICAM/' in path or path.endswith('/scheduler_runs/run_0001/result.json'):
            remember(path, digest)
    counts = rows(finish / 'counts.csv')
    assert len(counts) == 16 and not {'TSP', 'ATSP'} & {r['problem'] for r in counts}
    assert all(int(r['planned_labels']) == int(r['actual_verified_labels']) and
               int(r['full_pool_winners']) == int(r['endpoints']) and
               int(r['full_pool_pairs']) == int(r['planned_pairs']) for r in counts)
    assert sum(int(r['actual_verified_labels']) for r in counts) == closeout['actual_verified_labels'] == 9600
    assert sum(int(r['full_pool_winners']) for r in counts) == 2048
    assert sum(int(r['full_pool_pairs']) for r in counts) == 1536
    directed = rows(finish / 'response_analysis/directed_vs_random.csv')
    assert len(directed) == 24
    assert all(float(r['planned_bases']) == float(r['matched_full_pool_bases']) == 32 for r in directed)
    for name in ['RESULTS.pdf', 'response_analysis/RESPONSE_DIAGNOSIS.pdf']:
        stats, text = pdf(finish / name)
        pdfs.append(stats)
    for version in ['R112', 'R113']:
        stats, text = pdf(REPO / 'code/V4/runs' / ROOTS[version] / 'RESULTS.pdf')
        pdfs.append(stats)
    stats, text = pdf(r109 / 'MODEL_COMPARISON.pdf')
    pdfs.append(stats)
    references = [REPO / 'code/V4/runs/R57_task_signal_audit/RESULTS.md']
    for folder in ['comprehensive_direction_review_20261025', 'major_paradigm_review_20261025']:
        base = REPO / 'docs/research' / folder
        references.append(base / 'REPORT.md')
        stats, text = pdf(base / 'REPORT.pdf')
        pdfs.append(stats)
    references.append(REPO / 'docs/research/comprehensive_direction_review_20261025/running_snapshot.csv')
    for path in references:
        remember(path)

    # Start/end byte checks also cover the few preserved historical error files
    # that have no separately archived pre-migration hash of their own.
    for path, digest in INPUTS.copy().items():
        remember(path, digest)
    assert len(locks) == 7 and len(metrics) == 280
    write_csv('checkpoints.csv', CHECKPOINT_COLUMNS, locks)
    write_csv('metrics.csv', METRIC_COLUMNS, metrics)
    result = {'action_id': ACTION, 'saved_results_only': True,
              'scientific_completion_gate_sha256': canonical(snapshot),
              'source_snapshot_file_sha256': sha(HERE / 'source_snapshot.json'),
              'snapshot_entries_verified': len(snapshot),
              'model_invocation': {'requested_model': 'gpt-6.1-sol', 'reasoning_effort': 'max',
                                   'cli_record': str(MONITOR / 'actions' / ACTION / 'codex.json'),
                                   'provider_route_verified': False,
                                   'proof_limit': 'Exact -m/effort flags and live CLI ancestry; no independent provider generation-route proof'},
              'origin_main_checked': remote_head, 'publications': publications,
              'strict_published_bytes_pass': not publication_issues,
              'publication_byte_mismatches': publication_issues,
              'model_checks': checks, 'preserved_failures': FAILURES, 'pdf_checks': pdfs,
              'all_summary': summary, 'test_zero_selection_positive_native_winners': zeroes,
              'nss_comparison': nss, 'r114_closeout_kind': closeout['kind'],
              'r114_current_execution_phase': 'ACQUISITION/response pilot; NOT a trained function learner',
              'r114_counts': counts, 'r114_directed_vs_random': directed,
              'additional_verified_inputs': INPUTS,
              'native_raw_instances_reaudited': False, 'weights_deserialized': False,
              'model_or_solver_executed': False, 'gpu_used': False,
              'original_artifacts_modified': False, 'git_mutations': False}
    dump('saved_results_authentication.json', result)
    print(json.dumps({'saved_model_metrics_authenticated': True,
                      'strict_published_bytes_pass': not publication_issues,
                      'publication_byte_mismatches': len(publication_issues),
                      'models': len(locks), 'metrics_rows': len(metrics),
                      'published_assets': sum(x['verified_asset_count'] for x in publications),
                      'pdfs': len(pdfs), 'preserved_failures': len(FAILURES)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
