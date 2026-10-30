"""One-attempt R121 continuation after the complete, single-root native export.

The waiting controller is CPU-only. GPU work goes through the existing pinned
Return18 transport, never through direct SSH or a second allocation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


REPO = Path('/public/home/shiys/0selection2')
PREPARATION = REPO / 'code/V4/runs/R121_structural_families/preparation'
ROOT = PREPARATION / 'followup_principal_v1'
COLLECTION = PREPARATION / 'collection_explicit_cpu_v2'
LEARNER = PREPARATION / 'learner_v1'
DISPATCH = REPO / 'code/V4/runs/Return18_post_R120/dispatch'
PLAN_PATH = DISPATCH / 'main_approved_r121_principal_v1.json'
NODE_CACHE = '/tmp/shiys_R84_new90k_cache'
SLOT = 'gpu05_1'
EXPECTED_LABELS, EXPECTED_DEPLOYMENTS, EXPECTED_ENDPOINTS = 113920, 89, 23040
SOURCES = (
    'code/V4/r121_followup.py', 'code/V4/r121_experiment.py',
    'code/V4/r121_model.py', 'code/V4/r121_finish.py',
    'code/V4/r121_structural_data.py', 'code/V4/r121_collection.py',
    'code/V4/r121_universal_collection.py', 'code/V4/r121_input_boundary.py',
    'code/V4/return18_results.py',
    'code/V4/return18_launch.py', 'code/V4/run_v4_return18.sh',
    'code/V4/r120_launch.py', 'code/V4/r81_queue.py',
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, data, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(body)
    else:
        temporary = path.with_name('.' + path.name + '.' + str(os.getpid()))
        temporary.write_text(body)
        os.replace(temporary, path)


def source_hashes():
    return {name: sha(REPO / name) for name in SOURCES}


def verify_sources(contract):
    for name, digest in contract['source_sha256'].items():
        if sha(REPO / name) != digest:
            raise ValueError('Frozen source changed; preserve this attempt: ' + name)
    for item in contract['immutable_inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed: ' + item['path'])


def label_progress(contract):
    collection = Path(contract['collection_root'])
    rows, failed, uncertain = {}, [], []

    def count(path, expected):
        try:
            value = read(path)['completed'] if path.exists() else 0
        except (json.JSONDecodeError, FileNotFoundError) as error:
            # A collector may be replacing its cursor. Reobserve, never relaunch.
            uncertain.append(dict(path=str(path), error=type(error).__name__))
            return 0
        if type(value) is not int or not 0 <= value <= expected:
            raise ValueError('Invalid collection cursor: ' + str(path))
        return value

    for key, expected in contract['deployment_counts'].items():
        slug = key.replace('/', '__')
        progress = collection / 'progress' / (slug + '.json')
        value = count(progress, expected)
        unique = max([value, *[count(Path(source) / 'progress' / (slug + '.json'), expected)
            for source in contract.get('progress_source_roots', [])]])
        failure = collection / 'unavailable' / (slug + '.json')
        if failure.exists():
            failed.append(str(failure))
        rows[key] = dict(completed=value, collected_unique=unique, planned=expected,
            preflight=(collection / 'preflight' / (slug + '.json')).is_file())
    lanes = [i for i in range(4) if (collection / 'bootstrap' / ('lane' + str(i) + '.json')).is_file()]
    return dict(completed_labels=sum(r['completed'] for r in rows.values()),
        collected_unique_labels=sum(r['collected_unique'] for r in rows.values()),
        planned_labels=contract['planned_labels'], bootstrapped_lanes=lanes,
        full89_complete=len(lanes) == 4 and not failed and all(
            r['completed'] == r['planned'] and r['preflight'] for r in rows.values()),
        failures=failed, observation_uncertainty=uncertain, deployments=rows)


def create(root):
    root = Path(root).resolve()
    if root != ROOT:
        raise ValueError('This controller has one literal owned continuation root')
    if (root / 'contract.json').exists():
        raise ValueError('Existing contract is preserved; use status or run')
    config = read(COLLECTION / 'protocol.json')
    with Path(config['input_manifest']).open() as stream:
        manifest = json.load(stream)
    counts = Counter(t['problem'] + '/' + t['method'] for t in manifest['tasks'])
    if (len(counts) != EXPECTED_DEPLOYMENTS or sum(counts.values()) != EXPECTED_LABELS
            or len(manifest['endpoints']) != EXPECTED_ENDPOINTS
            or set(counts) != set(config['group_order'])):
        raise ValueError('Only the original complete full18/89 family population is authorized')
    ready_path = PREPARATION / 'learner_READY.json'
    ready = read(ready_path)
    if ready.get('learner_interface_ready') is not True:
        raise ValueError('Learner interface is not ready')
    for name, digest in ready['source_sha256'].items():
        if sha(REPO / 'code/V4' / name) != digest:
            raise ValueError('Learner READY no longer describes its source: ' + name)
    if any((LEARNER / 'principal' / name).exists() for name in
           ('last.pt', 'best.pt', 'training_complete.json')):
        raise ValueError('Never attach a new controller to an existing training attempt')
    files = (ready_path, COLLECTION / 'protocol.json', COLLECTION / 'universal.json',
        Path(config['input_manifest']), PREPARATION / 'learner_natural_isolation_v1.json',
        PREPARATION / 'collection_persistent_v1/main_r120_release.json',
        PREPARATION / 'collection_persistent_v1/main_binding_lane0.json')
    contract = dict(schema='r121-principal-followup-v1', main_approved=True,
        approved_by='main thread: user active goal, one full18 D1 principal; no partial pool or new ablation',
        approved_utc=utc(), root=str(root), collection_root=str(COLLECTION), learner_root=str(LEARNER),
        progress_source_roots=[str(PREPARATION / name) for name in
            ('collection_persistent_v1', 'collection_input_boundary_v1')],
        source_sha256=source_hashes(), immutable_inputs=[dict(path=str(p), sha256=sha(p)) for p in files],
        deployment_counts=dict(counts), planned_labels=EXPECTED_LABELS, endpoints=EXPECTED_ENDPOINTS,
        slot=SLOT, seed=2, logical_batch=1024, successful_updates=14400,
        micro_cap=352, eval_batch_size=256, lib_batch_size=1,
        wandb=True, wandb_mode='offline', wandb_reason='Existing allocations have no configured online credentials',
        resource_plan=str(PLAN_PATH), automatic_retry=False, automatic_resume=False,
        existing_native_jobs_interrupted=False, historical_costs_replaced=False,
        stages=['full89_export', 'family_cpu_features', 'freeze_learner_protocol',
            'train', 'test', 'lib', 'locked_train_and_family_capture', 'saved_only_finish',
            'four_route_analysis', 'publish'])
    write(root / 'contract.json', contract, exclusive=True)
    return contract


def common_args(contract):
    return ['--root', contract['learner_root'], '--device', 'cuda:0', '--gpu-type', '4090',
        '--cache-root', NODE_CACHE, '--test-cache-root', NODE_CACHE,
        '--family-labels', str(COLLECTION / 'family_labels.jsonl'),
        '--family-cache', str(LEARNER / 'family_cache'),
        '--release', str(PREPARATION / 'collection_persistent_v1/main_r120_release.json'),
        '--binding', str(PREPARATION / 'collection_persistent_v1/main_binding_lane0.json'),
        '--micro-cap', str(contract['micro_cap']), '--eval-batch-size', str(contract['eval_batch_size']),
        '--lib-batch-size', str(contract['lib_batch_size']), '--wandb-mode', 'offline']


def learner_namespace(contract):
    from . import r121_experiment as api
    return SimpleNamespace(root=LEARNER, arm=api.ARM, experiment='R121',
        data_root=api.DEFAULT_DATA_ROOT, prepared_root=api.DEFAULT_PREPARED_ROOT,
        cache_root=Path(NODE_CACHE), test_cache_root=Path(NODE_CACHE),
        family_labels=COLLECTION / 'family_labels.jsonl', family_cache=LEARNER / 'family_cache',
        family_input_manifest=PREPARATION / 'full18_inputs_v1/manifest.json',
        natural_isolation=api.DEFAULT_ISOLATION, device='cpu',
        release=PREPARATION / 'collection_persistent_v1/main_r120_release.json',
        binding=PREPARATION / 'collection_persistent_v1/main_binding_lane0.json',
        gpu_type='4090', micro_cap=contract['micro_cap'], eval_batch_size=contract['eval_batch_size'],
        lib_batch_size=contract['lib_batch_size'], resume=False, wandb_mode='offline',
        nss_root=REPO / 'code/V4/runs/R84_new90k_attention/nss_reference')


def freeze_protocol(contract):
    from . import r121_experiment as api
    args = learner_namespace(contract)
    ready, manifest, geometry, baselines = api.natural.prepared_config(args)
    rows, receipt, family_manifest = api.family_rows(args.family_labels)
    _, isolation = api.save_isolation(args, family_manifest)
    initial = api.torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    model, params, _ = api.model_api.initialize_shared(initial, geometry)
    config = api.protocol(args, ready, manifest, geometry, baselines, params, receipt,
        api.family_population(rows, isolation))
    del initial, model, rows
    path = LEARNER / api.ARM / 'protocol.json'
    if path.exists() and api.read_json(path) != config:
        raise ValueError('Preserve an existing different learner protocol')
    path.parent.mkdir(parents=True, exist_ok=True)
    api.write_json(path, config)
    return dict(path=str(path.relative_to(REPO)), sha256=sha(path), state_hash=api.json_hash(config))


def resource_plan(contract, protocol):
    arguments = common_args(contract)
    stages = [dict(id=name, kind=name, argv=['python', '-B', '-u', '-m',
        'code.V4.r121_experiment', name, *arguments]) for name in ('train', 'test', 'lib')]
    stages.append(dict(id='locked_train_and_family_capture', kind='report',
        argv=['python', '-B', '-u', '-m', 'code.V4.r121_finish', 'capture-train', *arguments]))
    return dict(schema='return18-resource-v1', name='r121_principal_full18_v1', main_approved=True,
        approved_by=contract['approved_by'], approved_utc=contract['approved_utc'],
        protocol={k: protocol[k] for k in ('path', 'sha256')}, source_sha256=contract['source_sha256'],
        jobs=[dict(id='R121_principal', arm='R121', slot=contract['slot'],
            root=str(LEARNER.relative_to(REPO)), stages=stages)])


def command(root, name, argv):
    logfile = root / (name + '.log')
    started = utc()
    state = dict(stage=name, state='running', argv=argv, started_utc=started, parent_pid=os.getpid())
    write(root / 'status.json', state)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with logfile.open('x') as stream:
        process = subprocess.Popen(argv, cwd=REPO, env=environment, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        state['pid'] = process.pid
        write(root / 'status.json', state)
        code = process.wait()
    result = dict(state, state='complete' if code == 0 else 'failed', exit_status=code, ended_utc=utc())
    write(root / (name + '.json'), result, exclusive=True)
    if code:
        raise RuntimeError('Preserved failed stage, no automatic retry: ' + name + '; see ' + str(logfile))
    return result


def run(root, poll_seconds):
    root = Path(root).resolve()
    contract = read(root / 'contract.json')
    with (root / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / 'claim.json', dict(pid=os.getpid(), started_utc=utc(),
            contract_sha256=sha(root / 'contract.json'), automatic_retry=False), exclusive=True)
        try:
            verify_sources(contract)
            while True:
                progress = label_progress(contract)
                if progress['failures']:
                    raise RuntimeError('Native deployment failed; do not train a partial pool: ' + str(progress['failures']))
                write(root / 'status.json', dict(state='waiting_full89_native_labels', pid=os.getpid(),
                    updated_utc=utc(), **progress))
                if progress['full89_complete']:
                    break
                time.sleep(poll_seconds)
            verify_sources(contract)
            command(root, 'full89_export', [sys.executable, '-B', '-u', '-m',
                'code.V4.r121_universal_collection', 'export', '--root', str(COLLECTION)])
            command(root, 'family_cpu_features', [sys.executable, '-B', '-u', '-m',
                'code.V4.r121_experiment', 'prepare-family', '--root', str(LEARNER), '--device', 'cpu',
                '--family-labels', str(COLLECTION / 'family_labels.jsonl'),
                '--family-cache', str(LEARNER / 'family_cache')])
            verify_sources(contract)
            protocol = freeze_protocol(contract)
            plan = resource_plan(contract, protocol)
            write(PLAN_PATH, plan, exclusive=True)
            from . import return18_launch as transport
            _, digest = transport.validate_plan(PLAN_PATH)
            write(root / 'learner_launch.json', dict(protocol=protocol, plan=str(PLAN_PATH), plan_sha256=digest))
            command(root, 'principal_gpu_stages', [sys.executable, '-B', '-u', '-m',
                'code.V4.return18_launch', 'dispatch', '--plan', str(PLAN_PATH), '--wait', '--poll-seconds', '30'])
            verify_sources(contract)
            command(root, 'saved_only_finish', [sys.executable, '-B', '-u', '-m',
                'code.V4.r121_finish', 'finish', '--root', str(LEARNER), '--device', 'cpu'])
            command(root, 'four_route_analysis', [sys.executable, '-B', '-u', '-m',
                'code.V4.return18_results', '--r121-root', str(LEARNER)])
            command(root, 'publish', [sys.executable, '-B', '-u', '-m', 'code.V4.return18_publish',
                '--experiment', 'R121', '--root', str(LEARNER), '--apply'])
            complete = dict(complete=True, updated_utc=utc(), experiment='R121', root=str(root),
                learner_root=str(LEARNER), automatic_retry=False, research_goal_complete=False)
            write(root / 'complete.json', complete, exclusive=True)
            write(root / 'status.json', complete)
            return complete
        except BaseException as error:
            failure = dict(state='failed', pid=os.getpid(), error=f'{type(error).__name__}: {error}',
                updated_utc=utc(), automatic_retry=False, automatic_resume=False)
            write(root / 'failure.json', failure, exclusive=True)
            write(root / 'status.json', failure)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('create', 'run', 'status'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    args = parser.parse_args(argv)
    if args.poll_seconds < 10:
        parser.error('Use a bounded polling interval of at least10 seconds')
    if args.command == 'create':
        result = create(args.root)
    elif args.command == 'run':
        result = run(args.root, args.poll_seconds)
    else:
        result = read(args.root / 'status.json')
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
