"""CPU-only saved-label closeout for original R114 plus corrected CVRP.

No solver imports, launches, repairs or source edits. Publication is an explicit
opt-in using a private copy of the existing one-attempt R114 publisher.
"""

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys
import textwrap
import time
import types

from . import r114_finish as parent
from . import r114_response_analysis as analysis


common = parent.common
NotReady = parent.NotReady
ROOT = common.GIT_ROOT / parent.ROOT_REL
RECOVERY = Path('preparation/tensor_boundary_recovery_20261026')
OUTPUT = Path('preparation/finish/tensor_boundary_combined_20261026')
REVISION = 'r114_cvrp_host_audit_acquisition_v1_20261026'
NEW_CODE = ('r114_recovery_finish.py', 'test_r114_recovery_finish.py',
            'r114_tensor_boundary_recovery.py', 'r114_cvrp_recovery.py',
            'r114_recovery_launch.py', 'test_r114_recovery_launch.py')
REPORT_FILES = ('RESULTS.md', 'RESULTS.pdf', 'counts.csv', 'closeout.json',
                'input_provenance.json', 'source_provenance.json', 'finish_complete.json')
ACQUISITION_FILES = ('CPU_READY.json', 'HANDOFF.md', 'source_bindings.json',
    'cold.lock.json', 'queue/plan.json', 'queue/deployments.grouped.lock.json',
    'queue/route_sources.json', 'profiles.original.snapshot.json',
    'profiles.corrected.snapshot.json', 'run_native.sh')


def unique(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate ' + key)
    return result


def check_subset(original, corrected):
    """Compare complete metadata, not just counts or human-readable IDs."""
    for manifest in (original, corrected):
        parent.sealed(manifest, 'manifest_sha256')
    expected = copy.deepcopy(original)
    expected.update(selected_problems=['CVRP'], pools={'CVRP': original['pools']['CVRP']},
        acquisition_revision=REVISION, parent_manifest_sha256=original['manifest_sha256'])
    for key in ('endpoints', 'pairs', 'tasks', 'rejections'):
        expected[key] = [r for r in original[key] if r['problem'] == 'CVRP']
    for key in ('endpoint', 'pair', 'task'):
        expected[key + '_count'] = len(expected[key + 's'])
    expected.pop('manifest_sha256')
    expected['manifest_sha256'] = common.json_hash(expected)
    if corrected != expected:
        raise ValueError('Corrected manifest must be the exact unchanged parent CVRP subset')
    counts = (len(corrected['pools']['CVRP']),
        len({r['base_id'] for r in corrected['endpoints']}),
        len(corrected['endpoints']), len(corrected['pairs']), len(corrected['tasks']))
    if counts != (10, 32, 96, 64, 960):
        raise ValueError('Require full CVRP 10 methods / 32 bases / 96 endpoints / 64 pairs / 960 tasks')
    for key, identifier in (('endpoints', 'endpoint_id'), ('pairs', 'pair_id'), ('tasks', 'task_id')):
        unique(corrected[key], identifier)


def check_assembly(manifest, labels, lock_sha256):
    if (labels.get('version') != manifest['version'] or
            labels.get('manifest_sha256') != manifest['manifest_sha256'] or
            labels.get('lock_sha256') != lock_sha256):
        raise ValueError('Assembly version/manifest/execution lock mismatch')
    rows = unique(labels['endpoints'], 'endpoint_id')
    planned = unique(manifest['endpoints'], 'endpoint_id')
    pairs = unique(labels['pairs'], 'pair_id')
    if set(rows) != set(planned) or set(pairs) != set(unique(manifest['pairs'], 'pair_id')):
        raise ValueError('Keep every actual endpoint and pair, including partial rows')
    counts = {p: dict(problem=p, endpoints=0, planned_labels=0, actual_verified_labels=0,
        full_pool_winners=0, planned_pairs=0, full_pool_pairs=0) for p in manifest['selected_problems']}
    for key, row in rows.items():
        endpoint = planned[key]
        if any(row.get(k) != endpoint[k] for k in ('base_id', 'problem', 'source_index')):
            raise ValueError('Endpoint metadata identity changed: ' + key)
        pool = sorted(manifest['pools'][row['problem']], key=lambda r: r['global_id'])
        if (row['methods'] != [r['method'] for r in pool] or
                row['global_ids'] != [r['global_id'] for r in pool]):
            raise ValueError('Candidate pool identity/order changed')
        costs, valid = row['costs_fp64'], row['valid_mask']
        if len(costs) != len(pool) or len(valid) != len(pool) or any(type(v) is not bool for v in valid):
            raise ValueError('Invalid actual cost/mask dimensions')
        for cost, usable in zip(costs, valid):
            if usable:
                if type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0:
                    raise ValueError('Verified native FP64 costs must be finite and positive')
            elif cost is not None:
                raise ValueError('Missing costs must remain null, never zero or an inferred value')
        full = bool(valid) and all(valid)
        winner = pool[min(range(len(pool)), key=lambda i: costs[i])] if full else None
        if (type(row['winner_mask']) is not bool or row['winner_mask'] != full or
                row['failure_mask'] != [not v for v in valid] or
                row['winner'] != (winner['method'] if winner else None) or
                row['winner_global_id'] != (winner['global_id'] if winner else None)):
            raise ValueError('Winner/failure mask must follow the actual full legal pool')
        count = counts[row['problem']]
        count['endpoints'] += 1
        count['planned_labels'] += len(pool)
        count['actual_verified_labels'] += sum(valid)
        count['full_pool_winners'] += int(full)
    for proposal in manifest['pairs']:
        row = pairs[proposal['pair_id']]
        if any(row.get(k) != v for k, v in proposal.items()):
            raise ValueError('Pair metadata identity changed: ' + proposal['pair_id'])
        old, new = rows[proposal['original_id']], rows[proposal['changed_id']]
        valid = [a and b for a, b in zip(old['valid_mask'], new['valid_mask'])]
        full = bool(valid) and all(valid)
        matrix = [[a and b for b in valid] for a in valid]
        if (row['methods'] != old['methods'] or row['candidate_change_mask'] != valid or
                row['failure_mask'] != [not v for v in valid] or row['Delta_mask'] != matrix or
                type(row['winner_pair_mask']) is not bool or row['winner_pair_mask'] != full or
                row['winner_flip'] != ((old['winner'] != new['winner']) if full else None) or
                row['original_winner'] != old['winner'] or row['changed_winner'] != new['winner']):
            raise ValueError('Paired winner/response masks disagree with real endpoint arrays')
        changes = row['candidate_change']
        delta = row['Delta_mn']
        if len(changes) != len(valid) or len(delta) != len(valid) or any(len(r) != len(valid) for r in delta):
            raise ValueError('Invalid saved response dimensions')
        for i, usable in enumerate(valid):
            expected = new['costs_fp64'][i] - old['costs_fp64'][i] if usable else None
            if not same_number(changes[i], expected):
                raise ValueError('Saved candidate change differs from actual costs')
            for j in range(len(valid)):
                expected = ((new['costs_fp64'][i] - new['costs_fp64'][j]) -
                    (old['costs_fp64'][i] - old['costs_fp64'][j])) if matrix[i][j] else None
                if not same_number(delta[i][j], expected):
                    raise ValueError('Saved Delta differs from actual masked native costs')
        count = counts[proposal['problem']]
        count['planned_pairs'] += 1
        count['full_pool_pairs'] += int(full)
    totals = dict(endpoint_count=len(rows), pair_count=len(pairs),
        verified_candidate_costs=sum(r['actual_verified_labels'] for r in counts.values()),
        full_pool_winners=sum(r['full_pool_winners'] for r in counts.values()),
        full_pool_pairs=sum(r['full_pool_pairs'] for r in counts.values()))
    if labels['summary'] != totals:
        raise ValueError('Assembly summary differs from real arrays')
    return totals, list(counts.values())


def same_number(actual, expected):
    return (actual is None if expected is None else type(actual) in (int, float)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))


def merge_labels(manifest, child_manifest, original, corrected, provenance):
    check_subset(manifest, child_manifest)
    for key, source, labels in (('parent', manifest, original), ('corrected_CVRP', child_manifest, corrected)):
        record = provenance[key]
        if record['manifest_sha256'] != source['manifest_sha256'] or not record['lock_sha256']:
            raise ValueError('Source provenance must identify each real manifest and execution lock')
        check_assembly(source, labels, record['lock_sha256'])
    if provenance['parent']['lock_sha256'] == provenance['corrected_CVRP']['lock_sha256']:
        raise ValueError('The host-audit correction requires its own execution lock')
    merged = copy.deepcopy(original)
    for field, identifier in (('endpoints', 'endpoint_id'), ('pairs', 'pair_id')):
        replacements = unique(corrected[field], identifier)
        merged[field] = [copy.deepcopy(replacements[r[identifier]]) if r['problem'] == 'CVRP'
                         else copy.deepcopy(r) for r in original[field]]
    merged.update(lock_sha256=None,
        execution_lock={p: provenance['corrected_CVRP' if p == 'CVRP' else 'parent']['lock_sha256']
                        for p in manifest['selected_problems']},
        per_problem_provenance={p: 'corrected_CVRP' if p == 'CVRP' else 'parent'
                                for p in manifest['selected_problems']},
        execution_provenance=copy.deepcopy(provenance), posthoc_merge=True)
    merged['summary'] = dict(endpoint_count=len(merged['endpoints']), pair_count=len(merged['pairs']),
        verified_candidate_costs=sum(sum(r['valid_mask']) for r in merged['endpoints']),
        full_pool_winners=sum(r['winner_mask'] for r in merged['endpoints']),
        full_pool_pairs=sum(r['winner_pair_mask'] for r in merged['pairs']))
    check_assembly(manifest, merged, None)
    return merged


def child_completed(root, plan_path):
    acquisition = Path(root).resolve() / RECOVERY / 'acquisition_v1'
    plan_path = parent.within(plan_path, acquisition)
    if not plan_path.is_file():
        raise NotReady('Awaiting corrected CVRP plan: ' + str(plan_path))
    plan = common.read(plan_path)
    parent.sealed(plan, 'plan_sha256')
    queue = parent.within(plan['queue_root'], acquisition)
    if queue != plan_path.parent:
        raise ValueError('Child plan must reside at its own queue root')
    progress_path = queue / 'scheduler_progress.json'
    if not progress_path.is_file():
        raise NotReady('Awaiting corrected CVRP scheduler_progress.json (CPU preparation is not labels)')
    status = common.read(progress_path)
    if status.get('status') not in parent.TERMINAL:
        raise NotReady('Corrected CVRP scheduler status=' + str(status.get('status')))
    run = parent.within(status['run_directory'], queue / 'scheduler_runs')
    if not (run / 'result.json').is_file() or common.read(run / 'result.json') != status:
        raise NotReady('Awaiting immutable corrected CVRP terminal result')
    if status.get('execution') is not True:
        raise ValueError('A dry-run is not corrected CVRP acquisition evidence')
    label_info, info = status.get('labels') or {}, status.get('analysis') or {}
    if not label_info.get('path') or not info.get('output') or info.get('status') != 'complete':
        raise NotReady('Awaiting actual corrected CVRP assembly AND completed saved analysis (--finalize)')
    labels_path = parent.within(label_info['path'], run)
    analysis_dir = parent.within(info['output'], run)
    receipt_path = analysis_dir / 'analysis_complete.json'
    if not labels_path.is_file() or not receipt_path.is_file():
        raise NotReady('Awaiting corrected CVRP labels and analysis_complete.json')
    manifest_path = parent.within(plan['manifest_path'], acquisition)
    lock_path = parent.within(plan['deployment_lock_path'], queue)
    reference_path = parent.within(plan['reference_lock_path'], acquisition)
    manifest, lock, reference = map(common.read, (manifest_path, lock_path, reference_path))
    for value, field in ((manifest, 'manifest_sha256'), (lock, 'lock_sha256'), (reference, 'lock_sha256')):
        parent.sealed(value, field)
    if (manifest['split'] != 'train' or plan['manifest_sha256'] != manifest['manifest_sha256'] or
            lock['manifest_sha256'] != manifest['manifest_sha256'] or
            plan['lock_sha256'] != lock['lock_sha256'] or
            plan['reference_lock_sha256'] != reference['lock_sha256'] or
            reference['manifest_sha256'] != manifest['manifest_sha256'] or
            lock['deployments'] != reference['deployments'] or
            lock['execution'].get('old_cost_reuse') is not False or
            any(status.get(k) != plan[k] for k in ('plan_sha256', 'manifest_sha256', 'lock_sha256'))):
        raise ValueError('Corrected CVRP terminal manifest/lock lineage mismatch')
    labels = common.read(labels_path)
    totals, _ = check_assembly(manifest, labels, plan['lock_sha256'])
    full = totals['verified_candidate_costs'] == len(manifest['tasks'])
    if (label_info['summary'] != totals or status.get('complete') is not (status['status'] == 'complete') or
            status['status'] == 'complete' and (not full or status.get('stopped_problems'))):
        raise ValueError('Corrected CVRP scheduler/actual coverage mismatch')
    receipt = common.read(receipt_path)
    if receipt.get('complete') is not True:
        raise NotReady('Awaiting completed corrected CVRP analysis')
    if (receipt.get('experiment') != 'R114_response_diagnosis' or receipt.get('model_training') is not False or
            Path(receipt['manifest_path']).resolve() != manifest_path or
            Path(receipt['labels_path']).resolve() != labels_path or
            receipt['manifest_sha256'] != common.digest(manifest_path) or
            receipt['labels_sha256'] != common.digest(labels_path) or
            receipt['planned_pairs'] != len(manifest['pairs']) or receipt['full_pool_pairs'] != totals['full_pool_pairs'] or
            set(receipt['files']) != set(parent.ANALYSIS_FILES)):
        raise ValueError('Child analysis is not bound to the actual corrected assembly')
    inputs = {str(p): common.digest(p) for p in (plan_path, progress_path, run / 'result.json',
        run / 'request.json', manifest_path, lock_path, reference_path, labels_path, receipt_path)}
    for name, expected in receipt['files'].items():
        path = analysis_dir / name
        if common.digest(path) != expected:
            raise ValueError('Saved corrected CVRP analysis changed: ' + name)
        inputs[str(path)] = expected
    pdf = (analysis_dir / 'RESPONSE_DIAGNOSIS.pdf').read_bytes()
    if not pdf.startswith(b'%PDF-') or b'%%EOF' not in pdf[-1024:]:
        raise ValueError('Saved corrected CVRP analysis PDF is incomplete')
    sources = {}
    for prefix in ('queue', 'group_worker', 'scheduler', 'analysis'):
        path = Path(plan[prefix + '_source']).resolve()
        expected = plan[prefix + ('_source_sha256' if prefix == 'queue' else '_sha256')]
        if common.digest(path) != expected:
            raise ValueError('Frozen child source changed: ' + str(path))
        sources[str(path)] = expected
    binding_path = queue / 'execution_binding.json'
    binding = common.read(binding_path)
    parent.sealed(binding, 'binding_sha256')
    request = common.read(run / 'request.json')
    if (binding.get('plan_sha256') != plan['plan_sha256'] or request.get('binding') != binding or
            request.get('plan_sha256') != plan['plan_sha256'] or request.get('finalize') is not True):
        raise ValueError('Child binding/finalized invocation mismatch')
    inputs[str(binding_path)] = common.digest(binding_path)
    return dict(plan=plan, manifest=manifest, labels=labels, status=status, run=run,
        plan_path=plan_path, progress_path=progress_path, labels_path=labels_path,
        manifest_path=manifest_path, binding=binding, inputs=inputs, sources=sources)


def launch_evidence(root, child):
    folder = Path(root).resolve() / RECOVERY / 'launch_3090'
    if (folder / 'attempt_0002_script_entry').is_dir():
        folder = folder / 'attempt_0002_script_entry'
    ready_name = 'ready_worker_nvml.json' if (folder / 'ready_worker_nvml.json').exists() else 'ready.json'
    names = (ready_name, 'request.json', 'entry.json', 'worker.json', 'controller.json')
    for name in names:
        if not (folder / name).is_file():
            raise NotReady('Awaiting actual corrected CVRP launch evidence: ' + name)
    ready, request, entry, worker, controller = [common.read(folder / n) for n in names]
    for row in (entry, worker, controller):
        if row.get('state') not in ('complete', 'blocked', 'failed', 'returned'):
            raise NotReady('Awaiting terminal corrected CVRP launch/worker receipts')
    checks = worker.get('resource_checks') or {}
    gpu, binding = checks.get('gpu') or {}, child['binding']
    uuid = binding.get('gpu_uuid', '')
    if (ready.get('main_approved') is not True or ready.get('slot') != '3090_1' or
            Path(ready['plan']).resolve() != child['plan_path'] or
            request.get('ready_sha256') != common.digest(folder / ready_name) or
            not uuid.startswith('GPU-') or ready.get('gpu_uuid') != uuid or
            gpu.get('uuid') != uuid or 'RTX 3090' not in gpu.get('name', '') or
            worker.get('gpu_uuid') != uuid or worker.get('actual_gpu_type') != '3090' or
            worker.get('initialized') is not True or checks.get('allocation') != 1465 or
            checks.get('uid') != 2022 or not checks.get('container_id') or
            binding.get('host') != checks['container_id'][:12] or
            checks['container_id'] not in checks.get('cgroup', '') or
            '/job_1465/' not in entry.get('cgroup', '').strip() + '/' or
            entry.get('container', {}).get('id') != checks['container_id'] or
            entry.get('cpus') != checks.get('cpus') or
            not set(binding['cpu_ids']) <= set(checks['cpus']) or
            worker.get('acquisition_status') != child['status']['status'] or
            Path(worker.get('progress_path', '')).resolve() != child['progress_path'] or
            controller.get('gpu_uuid') != uuid or
            Path(controller.get('worker_path', '')).resolve() != folder / 'worker.json'):
        raise ValueError('Require honest actual RTX3090 worker/container/UUID/binding proof')
    argv = worker.get('argv') or []
    required = {'--plan': str(child['plan_path']), '--manifest': child['plan']['manifest_path'],
        '--deployment-lock': child['plan']['deployment_lock_path'], '--host': binding['host'],
        '--gpu-uuid': uuid, '--cpu-ids': ','.join(map(str, binding['cpu_ids'])),
        '--timeout-seconds': str(binding['timeout_seconds']), '-m': 'code.V4.r114_run_queue'}
    if '--execute' not in argv or '--finalize' not in argv or any(
            argv.count(k) != 1 or argv.index(k) + 1 >= len(argv) or argv[argv.index(k) + 1] != v
            for k, v in required.items()):
        raise ValueError('Actual child argv must be the bound existing runner with --execute --finalize')
    if any(r.get('exit_status') != worker.get('exit_status') for r in (entry, controller)):
        raise ValueError('Child launch and worker exit receipts disagree')
    inputs = {str(folder / n): common.digest(folder / n) for n in names}
    for path, expected in ready['sources'].items():
        if common.digest(path) != expected:
            raise ValueError('Launch-bound source changed: ' + path)
        inputs[path] = expected
    # Never copy ready.json: it contains the dispatch token.
    return dict(actual_gpu_type='3090', gpu=gpu, binding=binding,
                worker_state=worker['state'], worker_exit_status=worker['exit_status'],
                container_id=checks['container_id'], allocation=checks['allocation'],
                receipt_sha256={str(folder / n): inputs[str(folder / n)] for n in names}), inputs


def completed(root, plan_path):
    root = Path(root).resolve()
    context = parent.completed(root)
    if (context['progress_path'] != root / 'preparation/grouped_execution/queue_v1/scheduler_progress.json' or
            Path(common.read(context['plan_path'])['manifest_path']).resolve() != root / 'preparation/production_inputs/manifest.json'):
        raise ValueError('Require the exact original R114 v1 progress and full parent manifest')
    child = child_completed(root, plan_path)
    check_subset(common.read(root / 'preparation/production_inputs/manifest.json'), child['manifest'])
    hardware, launch_inputs = launch_evidence(root, child)
    return context, child, hardware, launch_inputs


def source_record(plan, manifest_path, labels_path, progress_path):
    return dict(plan_sha256=plan['plan_sha256'], manifest_sha256=plan['manifest_sha256'],
        lock_sha256=plan['lock_sha256'], full_manifest_file=parent.file_record(manifest_path),
        labels_file=parent.file_record(labels_path), progress_file=parent.file_record(progress_path))


def acquisition_paths(root):
    acquisition = Path(root).resolve() / RECOVERY / 'acquisition_v1'
    paths = [acquisition / n for n in ACQUISITION_FILES]
    paths.append(acquisition / 'scripts/train_100k_backends.py')
    lock = common.read(acquisition / 'queue/deployments.grouped.lock.json')
    for deployment in lock['deployments'].values():
        profile = parent.within(deployment['profile_path'], acquisition / 'profiles')
        paths += [profile, profile.with_name('dependency_contract.json'),
                  profile.with_name('profile_path_bindings.json')]
    return sorted(set(paths))


def source_bindings(root):
    acquisition = Path(root).resolve() / RECOVERY / 'acquisition_v1'
    bindings = common.read(acquisition / 'source_bindings.json')
    sources, ordinary = {}, []
    for entry in bindings:
        path = Path(entry['path'])
        if (not path.is_relative_to(acquisition / 'scripts') or
                str(path.resolve()) != entry['resolved_path'] or
                path.is_symlink() is not entry['symlink'] or common.digest(path) != entry['sha256'] or
                common.digest(entry['original_path']) != entry['original_sha256']):
            raise ValueError('Private script/source-target binding changed: ' + str(path))
        if not path.is_symlink():
            ordinary.append(path)
        sources[str(path)] = sources[entry['resolved_path']] = entry['sha256']
    if ordinary != [acquisition / 'scripts/train_100k_backends.py']:
        raise ValueError('Require exactly one ordinary corrected script; never stage linked originals')
    revision = common.read(Path(root).resolve() / RECOVERY / 'revision.json')
    if (sources[str(ordinary[0])] != revision['patch']['revised_sha256'] or
            common.digest(revision['patch']['path']) != revision['patch']['sha256']):
        raise ValueError('Corrected script differs from the recorded host-audit-only patch')
    return sources


def render_pdf(path, closeout, provenance, matched):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        fig, axis = plt.subplots(figsize=(12, 9))
        axis.axis('off')
        lines = [
            'R114: original other 15 problems + separately locked CVRP correction',
            'Closeout: ' + closeout['kind'], 'Labeling complete: ' + str(closeout['labeling_complete']),
            f"Verified costs: {closeout['actual_verified_labels']}/{closeout['planned_label_calls']}",
            f"Full-pool winners: {closeout['totals']['full_pool_winners']}/{closeout['totals']['endpoint_count']}",
            f"Full-pool pairs: {closeout['totals']['full_pool_pairs']}/{closeout['totals']['pair_count']}",
            'CVRP scope: 10 methods, 32 TRAIN bases, 96 endpoints, 64 pairs, 960 tasks.',
            'The original ICAM CVRP failure was a CUDA-to-NumPy host-audit boundary error,',
            'not evidence that ICAM cannot solve CVRP. Original failed attempts are preserved.',
            'Actual corrected execution: RTX3090; physical UUID: ' + closeout['hardware']['binding']['gpu_uuid'],
            'Merged lock_sha256 = null. Execution locks are mapped per problem, not unified.',
            'Native FP64 costs; exact winner ties by stable global ID. Missing is not zero.',
            'Directed versus random uses the same problem/family/TRAIN-base full-pool intersection.',
            'No trained function learner, TSP/ATSP response, full18 model, TEST/LIB or accuracy claim.',
            'Related endpoints are not independent samples. No solver or GPU is used by this finisher.',
            'A terminal partial/negative closeout is not labeling finished.', '',
            'Parent full manifest file SHA256: ' + provenance['parent']['full_manifest_file']['sha256'],
            'Parent execution lock: ' + provenance['parent']['lock_sha256'],
            'Corrected CVRP execution lock: ' + provenance['corrected_CVRP']['lock_sha256']]
        axis.text(0, 1, '\n'.join('\n'.join(textwrap.wrap(line, 104)) if line else '' for line in lines),
                  va='top', fontsize=10, linespacing=1.7, family='DejaVu Sans')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        for title, rows, fields in (
            ('Actual coverage by problem', closeout['per_problem'],
             ['problem', 'endpoints', 'planned_labels', 'actual_verified_labels', 'full_pool_winners', 'planned_pairs', 'full_pool_pairs']),
            ('Directed minus random: matched TRAIN bases', matched,
             ['problem', 'family', 'matched_full_pool_bases', 'mean_rms_delta_difference_pp', 'winner_flip_difference_pp', 'mean_amplitude_ratio'])):
            for start in range(0, max(1, len(rows)), 20):
                fig, axis = plt.subplots(figsize=(16, 9))
                axis.axis('off')
                axis.set_title(title, loc='left', pad=25)
                batch = rows[start:start + 20]
                if batch:
                    values = [[analysis.fmt(r.get(k)) if isinstance(r.get(k), float) or r.get(k) is None
                               else str(r[k]) for k in fields] for r in batch]
                    table = axis.table(cellText=values, colLabels=[k.replace('_', '\n') for k in fields], loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 1.8)
                else:
                    axis.text(.05, .5, 'MISSING, not zero. No usable response rows.')
                fig.text(.03, .04, 'Descriptive native full-pool responses only; no trained selector or model-performance claim.')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


def prepare(root, plan_path):
    root = Path(root).resolve()
    output = root / OUTPUT
    if (output / 'finish_complete.json').is_file():
        return verify_delivery(root, plan_path, output)
    context, child, hardware, launch_inputs = completed(root, plan_path)
    # This prepares/verifies the baseline, but NEVER publishes the parent alone.
    baseline = parent.prepare(root)
    manifest_path = root / 'preparation/production_inputs/manifest.json'
    manifest = common.read(manifest_path)
    original = common.read(context['labels_path'])
    original_plan = common.read(context['plan_path'])
    provenance = dict(parent=source_record(original_plan, manifest_path, context['labels_path'], context['progress_path']),
        corrected_CVRP=source_record(child['plan'], child['manifest_path'], child['labels_path'], child['progress_path']))
    merged = merge_labels(manifest, child['manifest'], original, child['labels'], provenance)
    totals, counts = check_assembly(manifest, merged, None)
    full = (totals['verified_candidate_costs'] == len(manifest['tasks']) and
            totals['full_pool_winners'] == len(manifest['endpoints']) and
            totals['full_pool_pairs'] == len(manifest['pairs']))
    kind = 'complete_labels' if full else 'partial_negative_closeout'
    inputs = {**baseline['inputs'], **child['inputs'], **launch_inputs,
        str(Path(baseline['output']) / 'finish_complete.json'): common.digest(Path(baseline['output']) / 'finish_complete.json')}
    recovery = root / RECOVERY
    for path in (recovery / 'revision.json', recovery / 'launch_3090/publication_handoff.json',
                 recovery / 'acquisition_v1/CPU_READY.json', recovery / 'acquisition_v1/source_bindings.json'):
        inputs[str(path)] = common.digest(path)
    revision = common.read(recovery / 'revision.json')
    for path, expected in revision['frozen_evidence_sha256'].items():
        if common.digest(path) != expected:
            raise ValueError('Original frozen evidence changed: ' + path)
        inputs[path] = expected
    sources = {**baseline['source_sha256'], **child['sources']}
    sources.update(source_bindings(root))
    locked_artifacts = common.read(child['plan']['deployment_lock_path'])['artifacts']
    for path in acquisition_paths(root):
        digest = common.digest(path)
        if str(path) in locked_artifacts and locked_artifacts[str(path)] != digest:
            raise ValueError('Locked private profile/contract/source changed: ' + str(path))
        inputs[str(path)] = digest
    for path, digest in sources.items():
        if path in locked_artifacts and locked_artifacts[path] != digest:
            raise ValueError('Execution-bound private source changed: ' + path)
    for name in NEW_CODE:
        path = root.parent.parent / name
        sources[str(path)] = common.digest(path)
    output.mkdir(parents=True, exist_ok=False)
    # Server-only merged assembly. It is hashed, but excluded from publication.
    labels_path = output / 'combined_labels.json'
    common.write(labels_path, merged)
    analysis.analyze(manifest_path, labels_path, output / 'response_analysis')
    with (output / 'response_analysis/directed_vs_random.csv').open(newline='') as stream:
        matched = [{k: (None if v == '' else v if k in ('problem', 'family') else float(v))
                    for k, v in row.items()} for row in csv.DictReader(stream)]
    closeout = dict(experiment='R114_tensor_boundary_combined', closeout_ready=True,
        kind=kind, closeout_kind=kind, labeling_complete=full, output=str(output),
        actual_verified_labels=totals['verified_candidate_costs'], planned_label_calls=len(manifest['tasks']),
        totals=totals, per_problem=counts, lock_sha256=None, execution_lock=merged['execution_lock'],
        per_problem_provenance=merged['per_problem_provenance'], execution_provenance=provenance,
        hardware=hardware, baseline_output=baseline['output'],
        parent_scheduler_status=context['status']['status'], child_scheduler_status=child['status']['status'],
        original_failed_attempts_preserved=True, model_training=False, trained_function_learner=False,
        all18_model_performance_claim=False, TSP_ATSP_response_claim=False, gpu_or_solver_execution=False,
        count_boundary='Actual separately assembled native cost arrays; no raw-receipt recount or missing-as-zero.',
        winner_policy='native FP64; exact ties by stable global ID')
    common.write(output / 'closeout.json', closeout)
    common.write_csv(output / 'counts.csv', counts)
    common.write(output / 'input_provenance.json', dict(inputs=inputs, execution_provenance=provenance,
        merged_labels_server_only=parent.file_record(labels_path), baseline_output=baseline['output']))
    common.write(output / 'source_provenance.json', dict(source_sha256=sources,
        source_bindings=common.read(recovery / 'acquisition_v1/source_bindings.json'),
        original_failed_attempts_preserved=True, single_source_lock_claim=False))
    markdown = ('# R114 Combined Saved-Label Closeout\n\n'
        f'**{kind}**; labeling_complete={full}. Actual verified costs: '
        f'{totals["verified_candidate_costs"]}/{len(manifest["tasks"])}; full-pool winners: '
        f'{totals["full_pool_winners"]}/{totals["endpoint_count"]}; full-pool pairs: '
        f'{totals["full_pool_pairs"]}/{totals["pair_count"]}.\n\n'
        'Only CVRP endpoint/pair rows were replaced by the separately locked correction '
        '(10 methods, 32 bases, 96 endpoints, 64 pairs, 960 tasks). The original other 15 problems '
        'and every metadata identity remain unchanged. Original ICAM/CVRP failed at a CUDA-to-NumPy '
        'host-audit boundary, not solver capability. Original failed attempts remain untouched.\n\n'
        'Both schedulers are terminal; both actual assemblies and saved analyses were verified. '
        'Corrected execution has actual RTX3090 worker/physical-UUID/binding evidence. '
        'Missing costs stay null with false masks; a partial-negative closeout is NOT labeling finished.\n\n'
        + parent.markdown_table(counts) + '\n## Matched Directed Minus Random\n\n'
        + parent.markdown_table(matched) + '\n## Provenance And Scope\n\n'
        + '```json\n' + json.dumps(provenance, indent=2) + '\n```\n\n'
        + 'Merged `lock_sha256=null`; `execution_lock` is mapped per problem. The full parent manifest '
        'file SHA256 and both source manifests/labels/progress hashes are recorded above. '
        'Server-only merged labels are excluded from publication. Baseline reports remain at '
        + baseline['output'] + '.\n\n'
        'Native FP64/stable-global-ID winners only. Same problem/family/TRAIN-base full-pool '
        'intersection only; missing response is not zero. Related endpoints are not independent samples. '
        'No trained function learner, TSP/ATSP response, full18 model, accuracy, TEST/LIB or generalization '
        'claim. This finisher executes no solver/model/GPU code. RESULTS.pdf is the primary delivery.\n')
    (output / 'RESULTS.md').write_text(markdown, encoding='utf-8')
    render_pdf(output / 'RESULTS.pdf', closeout, provenance, matched)
    for path, expected in {**inputs, **sources}.items():
        if common.digest(path) != expected:
            raise ValueError('Evidence changed during preparation; no publication: ' + path)
    outputs = {str(p.relative_to(output)): common.digest(p) for p in output.rglob('*') if p.is_file()}
    delivery = dict(closeout_ready=True, kind=kind, closeout_kind=kind, labeling_complete=full,
        output=str(output), inputs=inputs, source_sha256=sources, outputs=outputs,
        baseline_output=baseline['output'], child_plan=str(child['plan_path']),
        generator_sha256=common.digest(__file__))
    common.write(output / 'finish_complete.json', delivery)
    return delivery


def verify_delivery(root, plan_path, output=None):
    root = Path(root).resolve()
    output = Path(output or root / OUTPUT).resolve()
    if output != root / OUTPUT:
        raise ValueError('Monitor accepts only the fixed combined closeout directory')
    context, child, _, _ = completed(root, plan_path)
    receipt = common.read(output / 'finish_complete.json')
    if (receipt.get('closeout_ready') is not True or receipt.get('output') != str(output) or
            receipt.get('generator_sha256') != common.digest(__file__) or
            receipt.get('child_plan') != str(child['plan_path']) or
            receipt.get('kind') not in ('complete_labels', 'partial_negative_closeout') or
            receipt.get('labeling_complete') is not (receipt['kind'] == 'complete_labels') or
            receipt.get('closeout_kind') != receipt['kind']):
        raise ValueError('Combined closeout is incomplete, stale, or from another sidecar')
    for progress in (context['progress_path'], child['progress_path']):
        if receipt['inputs'].get(str(progress)) != common.digest(progress):
            raise ValueError('Scheduler changed after combined closeout')
    for path, expected in {**receipt['inputs'], **receipt['source_sha256']}.items():
        if common.digest(path) != expected:
            raise ValueError('Combined source/input changed: ' + path)
    for name, expected in receipt['outputs'].items():
        if common.digest(parent.within(output / name, output)) != expected:
            raise ValueError('Combined output changed: ' + name)
    source_bindings(root)
    parent.verify_delivery(root, receipt['baseline_output'])
    pdf = (output / 'RESULTS.pdf').read_bytes()
    if not pdf.startswith(b'%PDF-') or b'%%EOF' not in pdf[-1024:]:
        raise ValueError('Combined primary PDF is incomplete')
    return receipt


def publication_paths(root, delivery):
    root = Path(root).resolve()
    output = root / OUTPUT
    if delivery['output'] != str(output):
        raise ValueError('Refuse parent-only or alternate-directory publication')
    baseline = common.read(Path(delivery['baseline_output']) / 'finish_complete.json')
    paths = parent.publication_paths(root, baseline)
    recovery = root / RECOVERY
    paths += [root.parent.parent / n for n in NEW_CODE]
    paths += [common.GIT_ROOT / '0selection2' / n for n in
              ('test_r114_tensor_boundary_recovery.py', 'test_r114_cvrp_recovery.py')]
    paths += [output / n for n in REPORT_FILES]
    paths += [output / 'response_analysis' / n for n in (*parent.ANALYSIS_FILES, 'analysis_complete.json')]
    paths += [recovery / n for n in ('RECOVERY_PLAN.md', 'revision.json',
        'source_revision/train_100k_backends.py.patch', 'launch_3090/publication_handoff.json',
        'combined_closeout_prep/README.md')]
    paths += [recovery / 'launch_3090' / n for n in
              ('preinit_admin_nvml_failure.json', 'controller_before_admin_nvml_fix.log',
               'launcher_before_admin_nvml_fix.py', 'preinit_script_entry_failure.json',
               'launcher_before_script_entry_fix.py')]
    paths.append(root.parent.parent / 'run_v4_r114_recovery.sh')
    paths += acquisition_paths(root)
    child = child_completed(root, delivery['child_plan'])
    paths += [child['run'] / 'result.json', child['run'] / 'request.json',
              Path(child['plan']['queue_root']) / 'execution_binding.json']
    paths = sorted(set(paths))
    for path in paths:
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(common.GIT_ROOT) or
                path.name in ('manifest.json', 'labels.json', 'combined_labels.json', 'ready.json') or
                path.stat().st_size > parent.SMALL_LIMIT):
            raise ValueError('Only exact small ordinary non-secret publication files: ' + str(path))
    if sum(p.stat().st_size for p in paths) > parent.TOTAL_LIMIT:
        raise ValueError('Combined publication exceeds the existing total size limit')
    return paths


def r113_published(root):
    path = Path(root).parent / 'R113_executable_process/publish_receipt.json'
    if not path.is_file() or common.read(path).get('state') != 'complete':
        raise NotReady('Awaiting independent R113 publication; never add dirty external dependencies')


def publish(root, plan_path, message=None):
    r113_published(root)
    delivery = prepare(root, plan_path)
    # A private namespace leaves every original publisher global and source untouched.
    private = types.ModuleType(__package__ + '._r114_combined_publisher')
    private.__package__, private.__file__ = __package__, parent.__file__
    exec(compile(Path(parent.__file__).read_text(), parent.__file__, 'exec'), private.__dict__)
    private.prepare = lambda r: verify_delivery(r, plan_path)
    private.verify_delivery = lambda r, out: verify_delivery(r, plan_path, out)
    private.publication_paths = publication_paths
    return private.publish(root, message or ('Publish R114 combined labels and CVRP host-audit correction'
        if delivery['labeling_complete'] else 'Publish R114 combined partial negative closeout; labels incomplete'))


def wait(root, plan_path, *, timeout_hours=336., poll_seconds=60., publish_after=False, message=None):
    if not math.isfinite(timeout_hours) or timeout_hours <= 0 or not math.isfinite(poll_seconds) or poll_seconds < 5:
        raise ValueError('Use a positive finite timeout and a quiet poll interval >=5 seconds')
    deadline, previous = time.monotonic() + timeout_hours * 3600, None
    while True:
        try:
            completed(root, plan_path)
            if publish_after:
                r113_published(root)
            break
        except NotReady as error:
            if time.monotonic() >= deadline:
                raise TimeoutError('Combined R114 wait timed out; no solver, preparation or publication retry')
            if str(error) != previous:
                print(str(error), flush=True)
                previous = str(error)
            time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))
    return publish(root, plan_path, message) if publish_after else prepare(root, plan_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'prepare', 'verify', 'wait', 'publish'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--publish', action='store_true', help='wait only: explicit one-commit/one-push opt-in')
    parser.add_argument('--timeout-hours', type=float, default=336.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--message')
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error("CPU-only sidecar requires CUDA_VISIBLE_DEVICES=''")
    if args.publish and args.command != 'wait':
        parser.error('--publish is only valid with wait')
    try:
        if args.command == 'status':
            completed(args.root, args.plan)
            result = dict(status='ready_for_saved_label_merge', closeout_ready=False, output=str(args.root.resolve() / OUTPUT))
        elif args.command == 'prepare':
            result = prepare(args.root, args.plan)
        elif args.command == 'verify':
            result = verify_delivery(args.root, args.plan)
        elif args.command == 'publish':
            result = publish(args.root, args.plan, args.message)
        else:
            result = wait(args.root, args.plan, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds,
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
    print(json.dumps({k: result[k] for k in ('status', 'state', 'kind', 'closeout_kind', 'closeout_ready',
        'labeling_complete', 'output', 'delivery', 'commit', 'commit_attempts', 'push_attempts') if k in result}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
