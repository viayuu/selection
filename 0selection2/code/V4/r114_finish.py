"""Read-only terminal R114 closeout; explicit one-attempt Git publication only.

Independent sidecar, deliberately absent from the frozen runtime source closure.
No solver/model imports, GPU access, data rebuilds, route retries or analysis runs.
"""

import argparse
import csv
import fcntl
import json
import math
import shutil
import sys
import time
from pathlib import Path

from . import r112_finish as common


ROOT_REL = Path('0selection2/code/V4/runs/R114_paired_responses')
CODE_NAMES = (
    'r114_response_data.py', 'r114_label_queue.py', 'r114_group_worker.py',
    'r114_run_queue.py', 'r114_response_analysis.py', 'r114_finish.py',
    'test_r114_response_data.py', 'test_r114_run_queue.py',
    'test_r114_response_analysis.py', 'test_r114_finish.py')
PREPARATION_FILES = (
    'README.md', 'schema.json', 'base_selection.json', 'acquisition_plan.json',
    'deployment_sources.json', 'grouped_execution/READY_RECEIPT.json',
    'grouped_execution/ASSUMPTIONS.md', 'grouped_execution/schema.json',
    'production_execution/QUEUE_HANDOFF.md', 'production_execution/ASSUMPTIONS.md',
    'production_execution/BUILD_NOTE.md', 'production_execution/conversion_complete.json',
    'production_execution/deployment_sources.production.json',
    'production_execution/test_label_queue.py')
ANALYSIS_FILES = (
    'RESPONSE_DIAGNOSIS.pdf', 'RESULTS.md', 'pair_responses.csv',
    'response_summary.csv', 'method_responses.csv', 'matched_bases.csv',
    'directed_vs_random.csv')
OUTPUT_FILES = ('RESULTS.md', 'counts.csv', 'closeout.json', 'data_provenance.json',
                'source_provenance.json', 'server_artifacts.sha256.json',
                'deployed_lock_summary.json', 'finish_complete.json')
TERMINAL = {'complete', 'blocked', 'failed', 'interrupted'}
SMALL_LIMIT = 8 * 1024 * 1024
TOTAL_LIMIT = 64 * 1024 * 1024
git = common.git
publication_mutex = common.publication_mutex
dependency_files = common.dependency_files
NotReady = common.NotReady


def within(path, parent):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(parent).resolve()):
        raise ValueError('Receipt path escapes its R114 scope: ' + str(path))
    return path


def sealed(record, field):
    if common.json_hash({k: v for k, v in record.items() if k != field}) != record.get(field):
        raise ValueError('Changed canonical ' + field)


def completed(root):
    """Small receipt polling only. Never scan inputs, weights, routes or Git."""
    root = Path(root).resolve()
    ready = common.read(root / 'preparation/grouped_execution/READY_RECEIPT.json')
    plan_path = within(ready['plan_path'], root)
    queue_root = plan_path.parent
    path = queue_root / 'scheduler_progress.json'
    if not path.is_file():
        raise NotReady('Awaiting R114 scheduler_progress.json; no launch/retry')
    status = common.read(path)
    if status.get('status') not in TERMINAL:
        raise NotReady('R114 scheduler status=' + str(status.get('status')))
    if status.get('execution') is not True:
        raise ValueError('Dry-run metadata is not an executed terminal result')
    run = within(status['run_directory'], queue_root / 'scheduler_runs')
    result_path = run / 'result.json'
    if not result_path.is_file() or common.read(result_path) != status:
        raise NotReady('Awaiting immutable terminal result matching scheduler progress')
    labels = status.get('labels') or {}
    analysis = status.get('analysis') or {}
    if not labels.get('path') or not analysis.get('output') or analysis.get('status') != 'complete':
        raise NotReady('Terminal result has no assembled labels + completed analysis; no automatic repair')
    labels_path = within(labels['path'], run)
    analysis_dir = within(analysis['output'], run)
    receipt_path = analysis_dir / 'analysis_complete.json'
    if not labels_path.is_file() or not receipt_path.is_file() or common.read(receipt_path).get('complete') is not True:
        raise NotReady('Awaiting existing assembly and analysis_complete.json; no rerun')
    return dict(root=root, ready=ready, plan_path=plan_path, queue_root=queue_root,
                progress_path=path, status=status, run=run, labels_path=labels_path,
                analysis_dir=analysis_dir, analysis_receipt_path=receipt_path)


def file_record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=common.digest(path))


def problem_reasons(context, plan):
    reasons = {p: [] for p in context['status'].get('stopped_problems', [])}
    evidence = {}
    for job in plan['runtime_jobs'] + plan['label_jobs']:
        if job['problem'] not in reasons:
            continue
        for attempt in sorted((context['queue_root'] / 'attempts' / job['job_id']).glob('attempt_*')):
            found = False
            for name in ('worker_result.json', 'problem_stop.json', 'queue_problem_stop.json', 'job_result.json'):
                path = attempt / name
                if not path.is_file():
                    continue
                row = common.read(path)
                evidence[str(path.resolve())] = common.digest(path)
                detail = row.get('error') or row.get('reason')
                if detail:
                    reasons[job['problem']].append(dict(job_id=job['job_id'], evidence=str(path), detail=detail))
                    found = True
            if not found and not (attempt / 'job_result.json').is_file():
                reasons[job['problem']].append(dict(job_id=job['job_id'], evidence=str(attempt),
                    detail='Incomplete attempt: no completion receipt; never retried'))
    for problem, records in reasons.items():
        if not records:
            records.append(dict(detail='Source-route deployment blocked before execution' if
                problem in plan.get('route_source_blocked_problems', []) else
                'Scheduler stopped this problem; detailed failure receipt unavailable (not inferred)'))
    return reasons, evidence


def validate(context):
    """Verify saved lineage/counts; never re-audit tours or re-open solver weights."""
    root, ready, status = context['root'], context['ready'], context['status']
    plan = common.read(context['plan_path'])
    sealed(plan, 'plan_sha256')
    if (plan['plan_sha256'] != ready['plan_sha256'] or status.get('plan_sha256') != plan['plan_sha256'] or
            Path(plan['queue_root']).resolve() != context['queue_root']):
        raise ValueError('Terminal result/ready/queue plan mismatch')
    manifest_path = within(plan['manifest_path'], root / 'preparation/production_inputs')
    lock_path = within(plan['deployment_lock_path'], context['queue_root'])
    reference_path = within(plan['reference_lock_path'], root / 'preparation')
    manifest, lock, reference = map(common.read, (manifest_path, lock_path, reference_path))
    for row, key in ((manifest, 'manifest_sha256'), (lock, 'lock_sha256'), (reference, 'lock_sha256')):
        sealed(row, key)
    if (manifest['manifest_sha256'] != plan['manifest_sha256'] or manifest['split'] != 'train' or
            status.get('manifest_sha256') != plan['manifest_sha256'] or
            lock['lock_sha256'] != plan['lock_sha256'] or status.get('lock_sha256') != plan['lock_sha256'] or
            lock['manifest_sha256'] != plan['manifest_sha256'] or
            reference['lock_sha256'] != plan['reference_lock_sha256'] or
            reference['deployments'] != lock['deployments'] or
            lock['execution'].get('old_cost_reuse') is not False):
        raise ValueError('Wrong TRAIN manifest/native deployment protocol provenance')
    selection_path = root / 'preparation/base_selection.json'
    selection = common.read(selection_path)
    sealed(selection, 'selection_sha256')
    if selection['selection_sha256'] != manifest['selection_sha256'] or selection['split'] != 'train':
        raise ValueError('Wrong fixed TRAIN selection')
    binding_path = context['queue_root'] / 'execution_binding.json'
    binding = common.read(binding_path)
    sealed(binding, 'binding_sha256')
    if binding.get('plan_sha256') != plan['plan_sha256'] or not binding.get('gpu_uuid', '').startswith('GPU-'):
        raise ValueError('Missing executed physical GPU binding')
    labels = common.read(context['labels_path'])
    if (labels.get('version') != manifest['version'] or labels.get('manifest_sha256') != plan['manifest_sha256'] or
            labels.get('lock_sha256') != plan['lock_sha256']):
        raise ValueError('Assembly is from another data/deployment version')
    endpoints = {e['endpoint_id']: e for e in manifest['endpoints']}
    rows = {e['endpoint_id']: e for e in labels['endpoints']}
    if len(rows) != len(labels['endpoints']) or set(rows) != set(endpoints):
        raise ValueError('Assembly must retain every endpoint, including masks')
    stopped = set(status.get('stopped_problems', []))
    counts = {p: dict(problem=p, endpoints=0, actual_verified_labels=0, full_pool_winners=0,
        planned_pairs=0, full_pool_pairs=0, stopped=p in stopped) for p in manifest['selected_problems']}
    bases = {(b['problem'], b['base_id']) for b in selection['bases']}
    for key, row in rows.items():
        endpoint = endpoints[key]
        problem = endpoint['problem']
        if problem in ('TSP', 'ATSP') or (problem, endpoint['base_id']) not in bases or row['base_id'] != endpoint['base_id'] or row['problem'] != problem:
            raise ValueError('Invalid intervention problem or fixed base lineage')
        pool = sorted(manifest['pools'][problem], key=lambda c: c['global_id'])
        if row['methods'] != [c['method'] for c in pool] or row['global_ids'] != [c['global_id'] for c in pool]:
            raise ValueError('Candidate pool/order differs from the frozen legal pool')
        costs, valid = row['costs_fp64'], row['valid_mask']
        if len(costs) != len(pool) or len(valid) != len(pool) or any(type(v) is not bool for v in valid):
            raise ValueError('Invalid candidate masks')
        for cost, usable in zip(costs, valid):
            if usable and (not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or cost <= 0):
                raise ValueError('Verified native FP64 cost must be finite and positive')
        complete = all(valid) and bool(valid)
        if bool(row['winner_mask']) != complete or problem in stopped and complete:
            raise ValueError('Stopped/partial problem cannot claim a full-pool winner')
        winner = pool[min(range(len(pool)), key=lambda i: costs[i])] if complete else None
        if row['winner'] != (winner['method'] if winner else None) or row.get('winner_global_id') != (winner['global_id'] if winner else None):
            raise ValueError('Winner must use native FP64/stable-global-ID argmin, never FP32')
        counts[problem]['endpoints'] += 1
        counts[problem]['actual_verified_labels'] += sum(valid)
        counts[problem]['full_pool_winners'] += int(complete)
    pairs = {r['pair_id']: r for r in labels['pairs']}
    if len(pairs) != len(labels['pairs']) or set(pairs) != {r['pair_id'] for r in manifest['pairs']}:
        raise ValueError('Assembly must retain every planned pair')
    for pair in manifest['pairs']:
        complete = bool(rows[pair['original_id']]['winner_mask'] and rows[pair['changed_id']]['winner_mask'])
        if pairs[pair['pair_id']]['winner_pair_mask'] != complete:
            raise ValueError('Partial endpoint cannot supply a full-pool paired winner')
        counts[pair['problem']]['planned_pairs'] += 1
        counts[pair['problem']]['full_pool_pairs'] += int(complete)
    totals = dict(endpoint_count=len(rows), pair_count=len(pairs),
        verified_candidate_costs=sum(c['actual_verified_labels'] for c in counts.values()),
        full_pool_winners=sum(c['full_pool_winners'] for c in counts.values()),
        full_pool_pairs=sum(c['full_pool_pairs'] for c in counts.values()))
    if labels['summary'] != totals or status['labels']['summary'] != totals:
        raise ValueError('Saved actual verified label/winner counts disagree')
    labeling_complete = status['status'] == 'complete'
    if status.get('complete') is not labeling_complete or labeling_complete and (
            stopped or totals['verified_candidate_costs'] != len(manifest['tasks']) or
            totals['full_pool_winners'] != len(rows) or totals['full_pool_pairs'] != len(pairs)):
        raise ValueError('Cannot declare the complete labeling version from partial results')
    receipt = common.read(context['analysis_receipt_path'])
    if (receipt.get('experiment') != 'R114_response_diagnosis' or receipt.get('complete') is not True or
            receipt.get('model_training') is not False or
            Path(receipt['manifest_path']).resolve() != manifest_path or
            Path(receipt['labels_path']).resolve() != context['labels_path'] or
            receipt['manifest_sha256'] != common.digest(manifest_path) or
            receipt['labels_sha256'] != common.digest(context['labels_path']) or
            receipt['planned_pairs'] != len(pairs) or receipt['full_pool_pairs'] != totals['full_pool_pairs'] or
            set(receipt['files']) != set(ANALYSIS_FILES)):
        raise ValueError('Analysis receipt is incomplete or not bound to the actual assembly')
    for name, expected in receipt['files'].items():
        if common.digest(context['analysis_dir'] / name) != expected:
            raise ValueError('Saved analysis changed: ' + name)
    pdf = (context['analysis_dir'] / 'RESPONSE_DIAGNOSIS.pdf').read_bytes()
    if not pdf.startswith(b'%PDF-') or b'%%EOF' not in pdf[-1024:]:
        raise ValueError('Analysis PDF is incomplete')
    sources = {}
    v4 = root.parent.parent
    for name, expected in ready['source_sha256'].items():
        if name not in CODE_NAMES:
            raise ValueError('Unexpected ready source; never auto-expand publication scope')
        if common.digest(v4 / name) != expected:
            raise ValueError('Frozen R114 source changed: ' + name)
        sources[str(v4 / name)] = expected
    for name in CODE_NAMES:
        sources[str(v4 / name)] = common.digest(v4 / name)
    reasons, reason_files = problem_reasons(context, plan)
    inputs = {str(p): common.digest(p) for p in (context['progress_path'], context['run'] / 'result.json',
        context['plan_path'], manifest_path, lock_path, reference_path, selection_path, binding_path,
        context['labels_path'], context['analysis_receipt_path'], root / 'preparation/grouped_execution/READY_RECEIPT.json')}
    inputs.update(reason_files)
    inputs.update({str(context['analysis_dir'] / n): common.digest(context['analysis_dir'] / n) for n in ANALYSIS_FILES})
    return dict(context=context, plan=plan, manifest=manifest, lock=lock, selection=selection,
        binding=binding, totals=totals, per_problem=list(counts.values()), stopped_problem_reasons=reasons,
        labeling_complete=labeling_complete, sources=sources, inputs=inputs,
        closeout_kind='complete_labels' if labeling_complete else 'partial_negative_closeout')


def markdown_table(rows):
    if not rows:
        return 'MISSING (not zero).\n'
    fields = list(rows[0])
    clean = lambda value: str(value).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join('---' for _ in fields) + ' |',
        *['| ' + ' | '.join(clean(row.get(k, 'MISSING')) for k in fields) + ' |' for row in rows]]) + '\n'


def prepare(root):
    value = validate(completed(root))
    context = value['context']
    root = context['root']
    suffix = common.digest(context['progress_path'])[:12] + '_' + common.digest(__file__)[:8]
    output = root / 'preparation/finish' / (context['run'].name + '_' + suffix)
    if (output / 'finish_complete.json').is_file():
        return verify_delivery(root, output)
    output.mkdir(parents=True, exist_ok=False)
    closeout = dict(experiment='R114_paired_response_pilot', closeout_ready=True,
        closeout_kind=value['closeout_kind'], labeling_complete=value['labeling_complete'],
        scheduler_status=context['status']['status'], actual_verified_labels=value['totals']['verified_candidate_costs'],
        full_pool_winners=value['totals']['full_pool_winners'], full_pool_pairs=value['totals']['full_pool_pairs'],
        planned_label_calls=len(value['manifest']['tasks']), totals=value['totals'], per_problem=value['per_problem'],
        stopped_problem_reasons=value['stopped_problem_reasons'], scheduler_error=context['status'].get('error'),
        scheduler_committed_label_receipts=context['status'].get('committed_label_receipts'),
        count_boundary='Actual verified labels are valid candidate costs in the released assembly; stopped-problem raw attempts remain server-only and are not recounted.',
        model_training=False, all18_model_performance_claim=False, gpu_or_solver_execution=False,
        winner_policy='native float(cost) promoted to FP64; exact ties by stable global ID',
        no_automatic_retry=True, output=str(output))
    common.write(output / 'closeout.json', closeout)
    common.write_csv(output / 'counts.csv', value['per_problem'])
    manifest = value['manifest']
    data_provenance = dict(version=manifest['version'], split='train',
        selection_sha256=manifest['selection_sha256'], manifest_sha256=manifest['manifest_sha256'],
        full_manifest_file=file_record(value['plan']['manifest_path']),
        selection_sources=value['selection']['sources'], data_root=value['selection']['data_root'],
        base_count=len({e['base_id'] for e in manifest['endpoints']}),
        pools={p: manifest['pools'][p] for p in manifest['selected_problems']},
        full_inputs_and_costs_retained_on_server=True, original_dataset_unchanged=True,
        train_only_pilot_no_TEST_LIB_claim=True)
    common.write(output / 'data_provenance.json', data_provenance)
    common.write(output / 'source_provenance.json', dict(source_sha256=value['sources'],
        plan_sha256=value['plan']['plan_sha256'], lock_sha256=value['plan']['lock_sha256'],
        reference_lock_sha256=value['plan']['reference_lock_sha256'], binding=value['binding'],
        finisher_independent_of_runtime_freeze=True))
    common.write(output / 'deployed_lock_summary.json', dict(lock_sha256=value['lock']['lock_sha256'],
        reference_lock_sha256=value['plan']['reference_lock_sha256'],
        execution=value['lock']['execution'], runtime=value['lock'].get('runtime'),
        deployments=value['lock']['deployments'], source_and_weight_sha256=value['lock']['artifacts'],
        weights_rehashed_by_finisher=False, protocol_changed=False))
    common.write(output / 'server_artifacts.sha256.json', dict(
        retained_files=[file_record(p) for p in (value['plan']['manifest_path'], context['labels_path'])],
        endpoint_inputs=[{k: e[k] for k in ('endpoint_id', 'base_id', 'problem', 'input_path', 'input_sha256')}
                         for e in manifest['endpoints']],
        attempt_roots=[str(context['queue_root'] / 'attempts' / j['job_id'])
                       for j in value['plan']['runtime_jobs'] + value['plan']['label_jobs']],
        durable_receipt_sha256={p: h for p, h in value['inputs'].items() if '/attempts/' in p},
        raw_routes_inputs_weights_caches_and_full_manifest_excluded_from_git=True,
        endpoint_hashes_are_frozen_manifest_witnesses_not_a_new_input_audit=True))
    analysis_output = output / 'response_analysis'
    analysis_output.mkdir()
    for name in (*ANALYSIS_FILES, 'analysis_complete.json'):
        shutil.copyfile(context['analysis_dir'] / name, analysis_output / name)
    summary = ('# R114 Paired Response Pilot Closeout\n\n'
        f'Closeout: **{value["closeout_kind"]}**; scheduler={context["status"]["status"]}; '
        f'labeling_complete={value["labeling_complete"]}.\n\n'
        f'Actual verified candidate labels in assembly: **{closeout["actual_verified_labels"]}/{closeout["planned_label_calls"]}**; '
        f'full-pool winner rows: **{closeout["full_pool_winners"]}/{len(manifest["endpoints"])}**; '
        f'full-pool pairs: **{closeout["full_pool_pairs"]}/{len(manifest["pairs"])}**.\n\n'
        'A partial/negative closeout is NOT labeling finished. Missing costs/routes mask full-pool winners; '
        'raw stopped-problem attempts stay on the server. No solver retries, repairs or new analysis runs.\n\n'
        + markdown_table(value['per_problem']) + '\n## Stopped Problems\n\n'
        + '```json\n' + json.dumps(value['stopped_problem_reasons'], indent=2) + '\n```\n\n'
        + '## Directed Versus Random\n\n'
        + 'Same problem/family/TRAIN-base full-pool intersection only; missing is not a zero response.\n\n')
    with (analysis_output / 'directed_vs_random.csv').open(newline='') as stream:
        summary += markdown_table(list(csv.DictReader(stream)))
    summary += ('\n## Provenance And Scope\n\n'
        f'Data version: {manifest["version"]}; fixed TRAIN bases: {data_provenance["base_count"]}.\n\n'
        f'Manifest canonical SHA256: {manifest["manifest_sha256"]}.\n\n'
        f'Full server manifest file SHA256: {data_provenance["full_manifest_file"]["sha256"]}.\n\n'
        f'Deployed lock SHA256: {value["plan"]["lock_sha256"]}; selection SHA256: {manifest["selection_sha256"]}.\n\n'
        'Native float(cost) is promoted to FP64; exact ties use stable global ID. '
        'Both endpoints were required to be fresh under the same locked native protocol. '
        'This is a 16-applicable-problem/512-base pilot design, not an all18 model evaluation; '
        'TSP/ATSP interventions are not applicable. No training, accuracy, TEST/LIB or generalization claim. '
        'All counts, failure reasons and matched comparisons above are self-contained; no external images are required.\n')
    (output / 'RESULTS.md').write_text(summary, encoding='utf-8')
    for path, expected in value['inputs'].items():
        if common.digest(path) != expected:
            raise ValueError('Terminal evidence changed during closeout; no publication')
    outputs = {str(p.relative_to(output)): common.digest(p) for p in output.rglob('*') if p.is_file()}
    receipt = dict(closeout_ready=True, labeling_complete=value['labeling_complete'],
        closeout_kind=value['closeout_kind'], output=str(output), inputs=value['inputs'],
        source_sha256=value['sources'], outputs=outputs, generator_sha256=common.digest(__file__))
    common.write(output / 'finish_complete.json', receipt)
    return receipt


def verify_delivery(root, output):
    context = completed(root)
    receipt = common.read(Path(output) / 'finish_complete.json')
    if not receipt.get('closeout_ready') or receipt.get('generator_sha256') != common.digest(__file__):
        raise ValueError('Closeout sidecar changed or incomplete')
    if receipt['inputs'].get(str(context['progress_path'])) != common.digest(context['progress_path']):
        raise ValueError('Scheduler resumed/changed after this closeout; no stale publication')
    for path, expected in {**receipt['inputs'], **receipt['source_sha256']}.items():
        if common.digest(path) != expected:
            raise ValueError('Closeout provenance changed: ' + path)
    for name, expected in receipt['outputs'].items():
        if common.digest(within(Path(output) / name, output)) != expected:
            raise ValueError('Closeout output changed: ' + name)
    return receipt


def publication_paths(root, delivery):
    root, repository = Path(root).resolve(), common.GIT_ROOT.resolve()
    if root != (repository / ROOT_REL).resolve():
        raise ValueError('Publisher owns only the named R114 root')
    output = within(delivery['output'], root / 'preparation/finish')
    context = completed(root)
    plan = common.read(context['plan_path'])
    paths = [root.parent.parent / n for n in CODE_NAMES]
    paths += [root / 'preparation' / n for n in PREPARATION_FILES]
    paths += [context['plan_path'], Path(plan['deployment_lock_path']), Path(plan['reference_lock_path']),
              context['plan_path'].parent / 'route_sources.json', context['run'] / 'result.json',
              context['queue_root'] / 'execution_binding.json']
    paths += [output / n for n in OUTPUT_FILES]
    paths += [output / 'response_analysis' / n for n in (*ANALYSIS_FILES, 'analysis_complete.json')]
    paths = sorted(set(paths))
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repository):
            raise ValueError('Allowlist requires ordinary exact in-repository files: ' + str(path))
        if path.name == 'manifest.json' or path.stat().st_size > SMALL_LIMIT:
            raise ValueError('Raw/full-manifest/large file forbidden from publication: ' + str(path))
    if sum(p.stat().st_size for p in paths) > TOTAL_LIMIT:
        raise ValueError('R114 publication exceeds its bounded total size')
    return paths


def upstream():
    branch = git('symbolic-ref', '--quiet', '--short', 'HEAD')
    remote = git('config', '--get', 'branch.' + branch + '.remote')
    merge = git('config', '--get', 'branch.' + branch + '.merge')
    if not remote or remote == '.' or not merge.startswith('refs/heads/'):
        raise ValueError('Require the existing named remote/upstream; never infer a push destination')
    parent = git('rev-parse', 'HEAD')
    rows = git('ls-remote', '--heads', remote, merge).splitlines()
    if len(rows) != 1 or rows[0].split()[0] != parent:
        raise ValueError('Dynamic remote upstream must equal current HEAD; reconcile manually, no retry')
    return dict(branch=branch, remote=remote, merge=merge, parent=parent)


def publication_preflight(root, delivery):
    verify_delivery(root, delivery['output'])
    if Path(git('rev-parse', '--show-toplevel')).resolve() != common.GIT_ROOT.resolve():
        raise ValueError('Unexpected Git root')
    if git('diff', '--cached', '--name-only', '-z'):
        raise ValueError('Pre-existing staged changes; index untouched')
    paths = publication_paths(root, delivery)
    names = [str(p.relative_to(common.GIT_ROOT)) for p in paths]
    dependencies = dependency_files([p for p in paths if p.suffix == '.py'])
    deps = [str(p.relative_to(common.GIT_ROOT)) for p in dependencies]
    tracked = set(filter(None, git('ls-files', '-z', '--', *deps).split('\0')))
    if set(deps) - tracked - set(names):
        raise ValueError('Unpublished dependencies outside exact R114 allowlist; never auto-add')
    outside = sorted(set(deps) - set(names))
    if outside and git('diff', 'HEAD', '--name-only', '-z', '--', *outside):
        raise ValueError('Dirty external dependencies must be published independently; never auto-add')
    return dict(paths=names, sha256={n: common.digest(common.GIT_ROOT / n) for n in names},
        dependency_sha256={n: common.digest(common.GIT_ROOT / n) for n in deps}, **upstream())


def publish(root, message=None):
    """One commit and one push maximum per version; preserve an uncertain attempt."""
    root = Path(root).resolve()
    delivery = prepare(root)
    finish_root = root / 'preparation/finish'
    with publication_mutex() as mutex, (finish_root / 'publication.lock').open('a') as claim:
        if Path(mutex).resolve() != (common.GIT_ROOT / '.git/codex-publication.lock').resolve():
            raise ValueError('Require the shared /public/home/shiys/.git/codex-publication.lock')
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = finish_root / 'publish_receipt.json'
        if path.exists():
            previous = common.read(path)
            if previous.get('state') == 'complete' and previous.get('delivery') == delivery['output']:
                verify_delivery(root, delivery['output'])
                return previous
            raise RuntimeError('Publication attempt already recorded; no automatic commit/push retry')
        plan = publication_preflight(root, delivery)
        manifest_path = Path(delivery['output']) / 'publication_manifest.json'
        common.write(manifest_path, dict(experiment='R114', labeling_complete=delivery['labeling_complete'],
            closeout_kind=delivery['closeout_kind'], **plan))
        names = [*plan['paths'], str(manifest_path.relative_to(common.GIT_ROOT))]
        receipt = dict(state='prepared', labeling_complete=delivery['labeling_complete'],
            closeout_kind=delivery['closeout_kind'], delivery=delivery['output'],
            parent=plan['parent'], remote=plan['remote'], merge=plan['merge'],
            allowlist=names, global_mutex=str(mutex), commit_attempts=0, push_attempts=0,
            automatic_retry=False)
        with path.open('x') as stream:
            json.dump(receipt, stream, indent=2)
        try:
            verify_delivery(root, delivery['output'])
            for name, expected in {**plan['sha256'], **plan['dependency_sha256']}.items():
                if common.digest(common.GIT_ROOT / name) != expected:
                    raise ValueError('Publication/dependency source changed before staging')
            if git('diff', '--cached', '--name-only', '-z') or upstream() != {k: plan[k] for k in ('branch', 'remote', 'merge', 'parent')}:
                raise ValueError('Index or dynamic parent/upstream changed; no staging')
            git('add', '-f', '--', *names)
            staged = set(filter(None, git('diff', '--cached', '--name-only', '-z').split('\0')))
            if not staged or staged - set(names):
                raise ValueError('Staged payload escapes the exact R114 allowlist; no commit')
            blobs = {name: git('rev-parse', ':' + name) for name in names}
            receipt.update(state='commit_started', commit_attempts=1)
            common.write(path, receipt)
            title = 'Publish complete R114 paired-response pilot' if delivery['labeling_complete'] else 'Publish R114 partial negative closeout; labels incomplete'
            git('commit', '-m', message or title, '--only', '--', *names)
            commit = git('rev-parse', 'HEAD')
            receipt.update(state='committed', commit=commit)
            common.write(path, receipt)
            changed = set(git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).splitlines())
            if not changed or changed - set(names) or git('rev-parse', 'HEAD^') != plan['parent'] or any(
                    git('rev-parse', commit + ':' + n) != b for n, b in blobs.items()):
                raise ValueError('Commit differs from exact R114 staged payload; no push')
            rows = git('ls-remote', '--heads', plan['remote'], plan['merge']).splitlines()
            if len(rows) != 1 or rows[0].split()[0] != plan['parent']:
                raise ValueError('Remote moved after local commit; no push/retry')
            receipt.update(state='push_started', push_attempts=1)
            common.write(path, receipt)
            git('push', '--porcelain', plan['remote'], commit + ':' + plan['merge'])
            receipt.update(state='complete', publication_complete=True)
            common.write(path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(state='failed', error_type=type(error).__name__, error=str(error),
                           manual_recovery_required=True)
            common.write(path, receipt)
            raise


def wait(root, *, timeout_hours=72., poll_seconds=60., publish_after=False, message=None):
    if not math.isfinite(timeout_hours) or timeout_hours <= 0 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError('Wait timeout and poll interval must be finite and positive')
    deadline, previous = time.monotonic() + timeout_hours * 3600, None
    while True:
        try:
            completed(root)
            break
        except NotReady as error:
            if time.monotonic() >= deadline:
                raise TimeoutError('R114 sidecar wait timed out; no solver/analysis/publication retry')
            if str(error) != previous:
                print(str(error), flush=True)
                previous = str(error)
            time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))
    # Only readiness is polled. Never retry preparation or a publication attempt.
    return publish(root, message) if publish_after else prepare(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'prepare', 'wait', 'publish'))
    parser.add_argument('--root', type=Path, default=common.GIT_ROOT / ROOT_REL)
    parser.add_argument('--publish', action='store_true', help='wait only: explicitly allow one commit/push')
    parser.add_argument('--timeout-hours', type=float, default=72.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--message')
    args = parser.parse_args(argv)
    if args.publish and args.command != 'wait':
        parser.error('--publish is a wait-only opt-in; otherwise use publish')
    try:
        if args.command == 'status':
            result = completed(args.root)['status']
        elif args.command == 'prepare':
            result = prepare(args.root)
        elif args.command == 'publish':
            result = publish(args.root, args.message)
        else:
            result = wait(args.root, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds,
                          publish_after=args.publish, message=args.message)
    except NotReady as error:
        print(str(error), file=sys.stderr)
        return 2
    except TimeoutError as error:
        print(str(error), file=sys.stderr)
        return 124
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps({k: result[k] for k in ('status', 'state', 'closeout_kind', 'labeling_complete',
                                          'output', 'commit', 'commit_attempts', 'push_attempts') if k in result}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
