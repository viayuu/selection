"""One-shot frozen R97 inference on the existing R94B allocation slot.

Scientific settings belong to main's protocol and runner. Preparation and parser
checks are CPU-only; resource inspection is available only after explicit ready.
Dependency imports use protocol library_root and optional launch.pythonpath.
"""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R97_launch'
MODEL_ROOT = REPO / 'code/V4/runs/R97_tabular_in_context'
PROTOCOL_LOCK = MODEL_ROOT / 'protocol_lock.json'
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
PREPARED = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
FEATURE_ROOT = REPO / 'code/V4/runs/R92_broad_learning/features'
CACHE_SOURCE = MODEL_ROOT / 'context'
CACHE_ROOT = Path('/tmp/shiys_R97_tabular_context')
WRAPPER = REPO / 'code/V4/run_v4_r97_launch.sh'
TOKEN_ENV = 'R97_LAUNCH_TOKEN'
ARMS = ('A',)
ARM_SLOTS = dict(A='4090_1')
SLOT = ARM_SLOTS['A']
PREDECESSOR_LAUNCH = REPO / 'code/V4/runs/R94_launch/B'
PREDECESSOR_MODEL = REPO / 'code/V4/runs/R94_pretrained_node_fusion/B'
ATTEMPT_FILES = ('start.json', 'attempt.json', 'claim.json', 'controller.json', 'tmux.json')
STAMP = '.r97_context_identity.json'


def _private(name, filename):
    spec = importlib.util.spec_from_file_location('code.V4.' + name, REPO / 'code/V4' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Rebind only these private objects, never imported R84/R94/R95 namespaces.
transport = _private('_r97_transport', 'r81_queue.py')
base = _private('_r97_resources', 'r84_launch.py')
base.transport = transport
INVENTORY_SLOTS = {name: dict(item) for name, item in base.SLOTS.items()}
SLOTS = {SLOT: dict(INVENTORY_SLOTS[SLOT])}
_prepare, _check_ready = base.prepare, base.check_ready
_environment, _container_command = base.environment, base.container_command
_load_spec, _worker, _cache_manifest = base.load_spec, base.worker, base.cache_manifest


def require_confirmation(evidence, tested):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True:
        raise RuntimeError('Explicit main-ready evidence and runner-tests-passed are required')


def locked_protocol():
    if not PROTOCOL_LOCK.is_file():
        raise RuntimeError('Waiting for main R97 protocol_lock.json')
    value = base.read(PROTOCOL_LOCK)
    expected = dict(experiment='R97', locked=True, task_scope='full18',
                    data_root=str(DATA), prepared_root=str(PREPARED))
    if (any(value.get(key) != item for key, item in expected.items())
            or value.get('arm', 'A') != 'A'
            or value.get('arms', [value.get('arm')]) != ['A']):
        raise ValueError('Main must lock the R97 A/full18 NEW90k/R84 protocol')
    return value


def environment(slot=None, token=None, protocol=None):
    protocol = locked_protocol() if protocol is None else protocol
    roots = protocol.get('launch', {}).get('pythonpath', [])
    if isinstance(roots, list) and protocol.get('library_root') is not None:
        roots = [protocol['library_root'], *roots]
    if not isinstance(roots, list) or any(not isinstance(p, str) or not Path(p).is_absolute()
                                          or ':' in p for p in roots):
        raise ValueError('library_root/launch.pythonpath must use absolute dependency import roots')
    result = dict(_environment(slot, token), WANDB_MODE='offline')
    result['PYTHONPATH'] = ':'.join([*roots, result['PYTHONPATH']])
    return result


def self_command(stage, spec=None, slot=None, token=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r97_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            argv += [flag, str(value)]
    return argv


def container_command(stage, slot, spec=None, token=None):
    if slot != SLOT or spec is None:
        raise ValueError('R97 container commands require the fixed slot and prepared spec')
    argv = _container_command(stage, slot, spec, token)
    argv[argv.index(str(REPO / 'code/V4/run_v4_r84_launch.sh'))] = str(WRAPPER)
    index = argv.index(transport.CONTAINERS[SLOTS[slot]['node']][1])
    argv[index:index] = ['-e', 'WANDB_MODE=offline']
    return argv


def no_symlinks(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Context and frozen sources must not use symlinks: ' + str(path))


def context_manifest(source):
    no_symlinks(source)
    manifest = _cache_manifest(source)
    if not any(row['path'].endswith('.json') for row in manifest['files']):
        raise ValueError('TRAIN context requires its JSON provenance manifests')
    if any(row['path'] == STAMP for row in manifest['files']):
        raise ValueError('Source context cannot contain a node-local staging stamp')
    return manifest


def _sources(protocol):
    """Discover once before ready; future source files do not change a saved spec."""
    directory = REPO / 'code/V4'
    paths = {p for pattern in ('r97*.py', 'test_r97*.py') for p in directory.glob(pattern)}
    paths.update(directory / name for name in ('r97_experiment.py', 'r97_reporting.py', 'r97_launch.py',
        'test_r97_launch.py', 'r84_launch.py', 'r81_queue.py'))
    paths.update((PROTOCOL_LOCK, WRAPPER, directory / 'run_v4_r84_launch.sh'))
    paths.update(CACHE_SOURCE.rglob('*.json'))
    paths.update(Path(p) if Path(p).is_absolute() else REPO / p
                 for p in protocol.get('launch', {}).get('source_files', []))
    pending, seen = list(paths), set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file() or path.suffix != '.py':
            continue
        seen.add(path)
        package = list(path.relative_to(REPO).parts[:-1]) if path.is_relative_to(REPO) else []
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [entry.name.split('.') for entry in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = prefix + (node.module.split('.') if node.module else [])
                modules = [module, *(module + entry.name.split('.') for entry in node.names)]
            for module in modules:
                if not module or module[0] != 'code':
                    continue
                candidate = REPO.joinpath(*module)
                candidate = candidate / '__init__.py' if candidate.is_dir() else candidate.with_suffix('.py')
                if candidate.is_file() and candidate not in paths:
                    paths.add(candidate)
                    pending.append(candidate)
    return sorted(str(p) for p in paths)


def template(arm='A', protocol=None, source_files=None):
    if arm != 'A':
        raise ValueError('R97 has exactly one frozen inference arm A')
    run = ['{python}', '-B', '-u', '-m', 'code.V4.r97_experiment', 'run',
        '--root', '{common_root}', '--data-root', '{data_root}',
        '--prepared-root', '{prepared_root}', '--feature-root', '{feature_root}',
        '--context-root', '{cache_root}', '--protocol-lock', str(PROTOCOL_LOCK),
        '--device', 'cuda:0', '--wandb-mode', 'offline']
    report = ['{python}', '-B', '-u', '-m', 'code.V4.r97_experiment', 'report',
        '--root', '{common_root}', '--protocol-lock', str(PROTOCOL_LOCK), '--device', 'cpu']
    return dict(schema_version=1, run_id='A', run_dir=str(MODEL_ROOT / 'A'),
        common_root=str(MODEL_ROOT), data_root=str(DATA), prepared_root=str(PREPARED),
        feature_root=str(FEATURE_ROOT), cache_source=str(CACHE_SOURCE), cache_root=str(CACHE_ROOT),
        slots=[SLOT], cpu_count=4, memory_gib=9, wandb_mode='offline',
        source_files=_sources(protocol or {}) if source_files is None else source_files,
        protocol=dict(mode='frozen_public_weight_inference', sgd_updates=0,
            context_split='train', current_query_labels=False,
            run_order=['resource_witness_full_TSP81000_32_unlabeled_VAL', 'full_VAL',
                       'unique_bank_lock', 'full_TEST', 'LIB']),
        stages=[dict(name='run', device='gpu', argv=run), dict(name='PDF', device='cpu', argv=report)])


def commands(spec, slot):
    if slot != SLOT or spec['slots'] != [SLOT] or spec['run_id'] != 'A':
        raise ValueError('R97 cannot fall back to the healthy R95B GPU')
    return [dict(stage, argv=[arg.format_map(dict(spec, python=sys.executable))
                             for arg in stage['argv']]) for stage in spec['stages']]


def load_spec(path):
    no_symlinks(path)
    value = _load_spec(path)
    sources = value.get('source_files', [])
    if (not isinstance(sources, list) or len(set(sources)) != len(sources)
            or any(not isinstance(p, str) or not Path(p).is_absolute() for p in sources)
            or value != template(source_files=sources)):
        raise ValueError('R97 spec differs from the fixed one-arm allocation/runner contract')
    locked_protocol()
    return value


def _parse_runtime(argv, protocol):
    if not (REPO / 'code/V4/r97_experiment.py').is_file():
        raise RuntimeError('Waiting for main R97 run/report parser')
    script = ('import json,sys; from code.V4 import r97_experiment as r; '
              'print(json.dumps(vars(r.parser().parse_args(json.loads(sys.argv[1]))), default=str))')
    result = base.subprocess.run([sys.executable, '-B', '-c', script, json.dumps(argv)],
        cwd=REPO, env=environment(protocol=protocol), text=True, capture_output=True,
        timeout=90, check=False)
    if result.returncode:
        raise RuntimeError('Waiting for compatible R97 CLI: ' + result.stderr[-2000:])
    return json.loads(result.stdout)


def parse_commands(protocol=None, spec=None):
    protocol = locked_protocol() if protocol is None else protocol
    spec = template(source_files=[]) if spec is None else spec
    rows = []
    for stage in commands(spec, SLOT):
        argv = stage['argv'][5:]
        parsed = _parse_runtime(argv, protocol)
        expected = dict(command=argv[0], root=str(MODEL_ROOT), protocol_lock=str(PROTOCOL_LOCK),
                        device='cuda:0' if stage['device'] == 'gpu' else 'cpu')
        if stage['name'] == 'run':
            expected.update(data_root=str(DATA), prepared_root=str(PREPARED),
                feature_root=str(FEATURE_ROOT), context_root=str(CACHE_ROOT), wandb_mode='offline')
        if any(str(parsed.get(key)) != value for key, value in expected.items()):
            raise RuntimeError('R97 parser differs from the locked run/report contract')
        if parsed.get('resume'):
            raise RuntimeError('R97 never implicitly resumes or retries')
        rows.append(dict(stage=stage['name'], argv=argv, parsed=True))
    return dict(commands=rows, parsed_count=2, resource_probe_performed=False, training_started=False)


def generate_spec(evidence=None, tested=False):
    require_confirmation(evidence, tested)
    protocol = locked_protocol()
    result = parse_commands(protocol)
    path, spec = ROOT / 'specs/A.json', template(protocol=protocol)
    if path.exists() and base.read(path) != spec:
        raise RuntimeError('Existing R97 spec differs; no implicit overwrite')
    if not path.exists():
        base.save(path, spec, exclusive=True)
    return dict(result, specs=[str(path)], main_ready_evidence=evidence)


write_specs = generate_spec


def stages_complete(record, names):
    stages = record.get('stages', [])
    by_name = {stage['name']: stage for stage in stages}
    return len(by_name) == len(stages) and all(
        by_name.get(name, {}).get('state') == 'complete'
        and by_name[name].get('exit_status') == 0 for name in names)


def predecessor_gate(arm='A'):
    if arm != 'A':
        raise ValueError('R97 A waits for R94B only')
    result = dict(experiment='R94', arm='B', complete=False, failed=False)
    try:
        for name, required in (('worker', ('input_features', 'train', 'TEST', 'timing', 'LIB')),
                ('controller', ('allocation_GPU_pipeline', 'PDF'))):
            record = base.read(PREDECESSOR_LAUNCH / (name + '.json'))
            if (record.get('state') == 'failed' or record.get('exit_status') not in (None, 0)
                    or any(s.get('state') == 'failed' or s.get('exit_status') not in (None, 0)
                           for s in record.get('stages', []))):
                raise ValueError('R94B failed; no automatic repair or retry')
            if (record.get('state') != 'complete' or record.get('exit_status') != 0
                    or not stages_complete(record, required)
                    or any(s.get('state') != 'complete' or s.get('exit_status') != 0
                           for s in record.get('stages', []))):
                raise FileNotFoundError('Waiting for R94B full worker/controller TEST/LIB/PDF rc0')
        lock = base.read(PREDECESSOR_MODEL / 'final_test_lock.json')
        if (lock.get('experiment') != 'R94' or lock.get('arm') != 'B'
                or not lock.get('checkpoint_sha256') or lock.get('selected_without_test') is not True
                or lock.get('per_problem_assembly') is not False):
            raise ValueError('R94B unique VAL lock is inconsistent')
        for name in ('test_results.json', 'lib_stage_complete.json'):
            record = base.read(PREDECESSOR_MODEL / name)
            if record.get('complete') is not True or record.get('checkpoint_sha256') != lock['checkpoint_sha256']:
                raise ValueError('R94B TEST/LIB is incomplete or uses another lock')
        post = base.read(PREDECESSOR_MODEL / 'postprocess_status.json')
        if post.get('complete') is not True or post.get('phase') != 'complete' or post.get('exit_status') != 0:
            raise ValueError('R94B PDF delivery is not complete rc0')
        for key in ('model_statistics_pdf', 'selection_distributions_pdf', 'target_recovery_pdf'):
            pdf = Path(post[key])
            if not pdf.is_relative_to(PREDECESSOR_MODEL) or not pdf.is_file() or not pdf.stat().st_size:
                raise FileNotFoundError('Waiting for R94B PDF: ' + str(pdf))
        result.update(complete=True, reason='R94B same-lock full delivery rc0; idle resources still required')
    except FileNotFoundError as error:
        result['reason'] = str(error)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update(failed=True, reason=str(error))
    return result


def prepare(path, evidence=None, tested=False):
    require_confirmation(evidence, tested)
    spec = load_spec(path)
    output = base.receipts(spec)
    with base.lock(output / 'prepare.lock'):
        if any((output / name).exists() for name in (*ATTEMPT_FILES, 'ready.json')):
            raise RuntimeError('R97 already ready/attempted; no implicit replacement')
        protocol = locked_protocol()
        if not FEATURE_ROOT.is_dir():
            raise RuntimeError('Waiting for existing R92 input-only features')
        parsed = parse_commands(protocol, spec)
        sources = set(spec['source_files']) | set(_sources(protocol))
        for source in sources:
            no_symlinks(source)
            if not Path(source).is_file():
                raise RuntimeError('Waiting for final R97 source before readiness: ' + source)
        result = _prepare(path)
        result['source_sha256'].update({source: base.sha(source) for source in sorted(sources)})
        result.update(experiment='R97', main_thread_ready=True, main_ready_evidence=evidence,
            runner_tests_passed=True, automatic_retry=False, resource_probe_performed=False,
            cache_mode='TRAIN_context_stage_once', cli=parsed, predecessor=predecessor_gate())
        base.save(output / 'plan.json', result)
        base.save(output / 'ready.json', dict(main_thread_ready=True, evidence=evidence,
            runner_tests_passed=True, confirmed_utc=base.utc(), plan_sha256=base.sha(output / 'plan.json')),
            exclusive=True)
        check_ready(path)
    return result


def check_ready(path):
    spec, plan = _check_ready(path)
    if context_manifest(spec['cache_source']) != plan['cache']:
        raise RuntimeError('Frozen TRAIN context size/manifest identity changed after ready')
    return spec, plan


def queue_prepared(path, evidence, tested):
    require_confirmation(evidence, tested)
    spec, plan = check_ready(path)
    if plan.get('main_ready_evidence') != evidence:
        raise RuntimeError('Queue requires the same explicit main-ready evidence as prepare')
    return spec, plan


def start(path, evidence=None, tested=False):
    spec, _ = queue_prepared(path, evidence, tested)
    gate = predecessor_gate()
    if not gate['complete']:
        raise RuntimeError(gate['reason'])
    if base.socket.gethostname().split('.')[0] != 'admin01' or not base.FONT.is_file():
        raise RuntimeError('Controller/PDF requires admin01 CPU with CJK fonts')
    output = base.receipts(spec)
    with base.lock(output / 'start.lock'):
        if any((output / name).exists() for name in ATTEMPT_FILES):
            raise RuntimeError('R97 already attempted; no automatic retry')
        base.save(output / 'start.json', dict(started_utc=base.utc(), evidence=evidence,
            automatic_retry=False, predecessor=gate), exclusive=True)
        session = 'r97_A'
        command = shlex.join(['bash', '-lc', base.activated(self_command('controller', path))])
        base.subprocess.run(['tmux', 'new-session', '-d', '-s', session, command], check=True)
        base.subprocess.run(['tmux', 'set-option', '-t', session, 'remain-on-exit', 'on'], check=True)
        base.save(output / 'tmux.json', dict(session=session, command=command), exclusive=True)
    return dict(session=session, receipts=str(output), automatic_retry=False)


def queue(path, evidence=None, tested=False, interval=60):
    if interval < 30:
        raise ValueError('Use a quiet queue interval of at least 30 seconds')
    spec, _ = queue_prepared(path, evidence, tested)
    output = base.receipts(spec)
    with base.lock(output / 'queue.lock'):
        if any((output / name).exists() for name in ATTEMPT_FILES):
            raise RuntimeError('R97 already attempted; no automatic retry')
        while True:
            queue_prepared(path, evidence, tested)
            gate = predecessor_gate()
            base.save(output / 'queue_status.json', dict(experiment='R97', arm='A',
                state='ready' if gate['complete'] else 'failed' if gate['failed'] else 'waiting_predecessor',
                updated_utc=base.utc(), predecessor=gate, resource_probe_performed=False,
                training_started=False, automatic_retry=False))
            if gate['failed']:
                raise RuntimeError(gate['reason'])
            if gate['complete']:
                return start(path, evidence, tested)
            time.sleep(interval)


def stage_cache(spec, manifest):
    destination = Path(spec['cache_root'])
    stamp = destination / STAMP
    with base.lock(destination.with_name(destination.name + '.r97_stage.lock'), blocking=True):
        no_symlinks(destination)
        if destination.exists() and any(p.is_symlink() for p in destination.rglob('*')):
            raise ValueError('Node-local context must not contain symlinks')
        if context_manifest(spec['cache_source']) != manifest:
            raise RuntimeError('Prepared TRAIN context source changed')
        if stamp.is_file():
            identity = base.read(stamp)
            if any(identity.get(key) != manifest[key] for key in ('source', 'bytes', 'identity_sha256')):
                raise RuntimeError('Existing node-local context belongs to another source/manifest')
        else:
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError('Nonempty unstamped context requires explicit review')
            if base.shutil.disk_usage(destination.parent).free < manifest['bytes'] + 2**30:
                raise RuntimeError('Insufficient node-local space for TRAIN context')
            for row in manifest['files']:
                target = destination / row['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.r97_tmp')
                base.shutil.copyfile(Path(spec['cache_source']) / row['path'], temporary)
                temporary.replace(target)
        for row in manifest['files']:
            target = destination / row['path']
            if not target.is_file() or target.stat().st_size != row['bytes']:
                raise RuntimeError('Node-local context is missing or has changed size')
            if 'sha256' in row and base.sha(target) != row['sha256']:
                raise RuntimeError('Node-local context JSON manifest identity changed')
        expected = {row['path'] for row in manifest['files']} | {STAMP}
        if any(str(p.relative_to(destination)) not in expected for p in destination.rglob('*') if p.is_file()):
            raise RuntimeError('Unexpected files in node-local TRAIN context')
        if not stamp.exists():
            base.save(stamp, dict(source=manifest['source'], bytes=manifest['bytes'],
                identity_sha256=manifest['identity_sha256'], staged_utc=base.utc()), exclusive=True)
    return dict(root=str(destination), bytes=manifest['bytes'], files=len(manifest['files']),
                identity_sha256=manifest['identity_sha256'], mode='TRAIN_context_stage_once')


def reserved_cpus(node, exclude):
    occupied = set()
    roots = (ROOT, *(REPO / ('code/V4/runs/R' + number + '_launch') for number in ('92', '93', '94', '95')))
    for root in roots:
        for path in root.glob('*/claim.json'):
            if path.parent == exclude:
                continue
            claim = base.read(path)
            controller = path.with_name('controller.json')
            record = base.read(controller) if controller.is_file() else {}
            # A failed/reporting/complete label alone is not evidence the GPU step ended.
            if claim.get('node') == node and not stages_complete(record, ('allocation_GPU_pipeline',)):
                occupied.update(claim['cpus'])
    return occupied


def authorize_resources(path, slot, token=None, claimed=False):
    if path is None or slot != SLOT:
        raise RuntimeError('No resource inspection without explicit-ready fixed-slot authorization')
    spec, plan = check_ready(path)
    gate = predecessor_gate()
    if not gate['complete']:
        raise RuntimeError(gate['reason'])
    if claimed:
        claim = base.read(base.receipts(spec) / 'claim.json')
        if not token or claim['slot'] != slot or claim['token'] != token:
            raise RuntimeError('R97 worker claim/token mismatch')
    return spec, plan


def container_resources(path, slot, token=None):
    authorize_resources(path, slot, token)
    identity = base.assert_container(slot)
    return dict(container_id=identity, cpus=sorted(os.sched_getaffinity(0)),
        memory=base.memory_inventory(identity), gpus={name: transport.gpu_inventory(item)
        for name, item in INVENTORY_SLOTS.items() if item['node'] == SLOTS[slot]['node']})


def capture(argv, timeout=90):
    return re.sub(r'\ADistribute Task Container\.\r?\ntask_id:\d+\r?\n', '',
                  transport.capture(argv, timeout=timeout))


def node_resources(path, slot, token=None):
    authorize_resources(path, slot, token)
    item = SLOTS[slot]
    if (base.socket.gethostname().split('.')[0] != item['node']
            or os.environ.get('SLURM_JOB_ID') != str(item['job'])
            or f'/job_{item["job"]}/' not in Path('/proc/self/cgroup').read_text()):
        raise RuntimeError('Node inspection requires the existing allocation CPU-only step')
    allocation, container = transport.allocation(item), transport.container_inspect(item)
    nested = json.loads(capture(container_command('container-resources', slot, path, token), timeout=60))
    if nested['container_id'] != container['id'] or not set(nested['cpus']) <= set(allocation['cpus']):
        raise RuntimeError('Allocation-owned container identity or CPU set changed')
    gpu_pids = {int(row[1]) for gpu in nested['gpus'].values() for row in gpu['compute_processes']}
    processes, found = [], set()
    for process in Path('/proc').glob('[0-9]*'):
        try:
            details = dict(line.split(':', 1) for line in (process / 'status').read_text().splitlines() if ':' in line)
            pids = {int(value) for value in details.get('NSpid', process.name).split()}
            matched = gpu_pids & pids
            if not matched and container['id'] not in (process / 'cgroup').read_text():
                continue
            argv = [v for v in (process / 'cmdline').read_bytes().decode(errors='replace').split('\0') if v]
            if not argv or 'container-resources' in argv or 'node-resources' in argv:
                continue
            if any(Path(v).name == 'conda' for v in argv) and 'hook' in argv:
                continue
            if not matched and not Path(argv[0]).name.startswith('python'):
                continue
            found.update(matched)
            processes.append(dict(pid=int(process.name), cpus=sorted(os.sched_getaffinity(int(process.name)))))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if gpu_pids - found:
        raise RuntimeError('Cannot attribute live GPU processes to CPU affinity')
    return dict(node=item['node'], allocation=allocation, container=container, memory=nested['memory'],
        processes=processes, gpus=nested['gpus'],
        allowed_cpus=sorted(set(nested['cpus']) & set(container['allowed_cpus'])))


def entry(path, slot, token):
    spec, _ = authorize_resources(path, slot, token, claimed=True)
    claim = base.read(base.receipts(spec) / 'claim.json')
    probe = node_resources(path, slot, token)
    gpu = probe['gpus'][slot]
    occupied = {cpu for row in probe['processes'] for cpu in row['cpus']}
    occupied |= reserved_cpus(SLOTS[slot]['node'], base.receipts(spec))
    if (gpu['compute_processes'] or gpu['memory_used_mib'] > 512 or gpu['utilization_pct']
            or set(claim['cpus']) & occupied
            or not set(claim['cpus']) <= set(probe['allowed_cpus'])
            or probe['memory']['headroom_excluding_reclaimable_cache_bytes'] < 9 * 2**30):
        raise RuntimeError('Claimed resources became busy; no retry or interruption')
    base.save(base.receipts(spec) / 'entry.json', probe, exclusive=True)
    return base.subprocess.run(container_command('worker', slot, path, token), check=False).returncode


def worker(path, slot, token):
    authorize_resources(path, slot, token, claimed=True)
    return _worker(path, slot, token)


def controller(path):
    if base.socket.gethostname().split('.')[0] != 'admin01' or not base.FONT.is_file():
        raise RuntimeError('Controller/PDF requires admin01 CPU with CJK fonts')
    spec, _ = authorize_resources(path, SLOT)
    output, item = base.receipts(spec), SLOTS[SLOT]
    record = dict(started_utc=base.utc(), slot=SLOT, node=item['node'], state='claimed',
                  automatic_retry=False, stages=[])
    with base.lock(output / 'controller.lock'):
        if any((output / name).exists() for name in ('attempt.json', 'claim.json', 'controller.json')):
            raise RuntimeError('No duplicate launch or automatic retry')
        base.save(output / 'attempt.json', dict(started_utc=base.utc(), automatic_retry=False), exclusive=True)
        base.save(output / 'controller.json', record, exclusive=True)
        try:
            mutex = transport.ROOT / f'queue_gpu_{item["uuid"]}.json'
            with base.lock(mutex, existing=mutex.exists()):
                with base.lock(transport.ROOT / (item['node'] + '.cpu.lock')):
                    probe = json.loads(capture(base.srun_command('node-resources', SLOT, path)))
                    gpu = probe['gpus'][SLOT]
                    occupied = {cpu for row in probe['processes'] for cpu in row['cpus']}
                    occupied |= reserved_cpus(item['node'], output)
                    cpus = sorted(set(probe['allowed_cpus']) - occupied)[:4]
                    if (gpu['compute_processes'] or gpu['memory_used_mib'] > 512 or gpu['utilization_pct']
                            or len(cpus) != 4
                            or probe['memory']['headroom_excluding_reclaimable_cache_bytes'] < 9 * 2**30):
                        raise RuntimeError('Fixed GPU/CPU/RAM busy; leave healthy runs alone; no retry')
                    authorize_resources(path, SLOT)
                    claim = dict(slot=SLOT, node=item['node'], gpu_uuid=item['uuid'], cpus=cpus,
                        token=base.uuid.uuid4().hex, claimed_utc=base.utc(), resources=probe)
                    base.save(output / 'claim.json', claim, exclusive=True)
                stage = dict(name='allocation_GPU_pipeline', device='gpu',
                    argv=base.srun_command('entry', SLOT, path, claim['token'], cpus=4, memory='9G'))
                base.run_stage(stage, spec, record, output / 'controller.json', environment())
                completed = base.read(output / 'worker.json')
                if (completed.get('state') != 'complete' or completed.get('exit_status') != 0
                        or not stages_complete(completed, ('run',))):
                    raise RuntimeError('CPU PDF requires rc0 resource witness/VAL/bank lock/TEST/LIB run')
                record.update(state='gpu_complete', gpu_finished_utc=base.utc())
                base.save(output / 'controller.json', record)
            # Release the exact-UUID mutex and allocation step before CPU reporting.
            check_ready(path)
            for stage in commands(spec, SLOT):
                if stage['device'] == 'cpu':
                    base.run_stage(stage, spec, record, output / 'controller.json', environment())
            record.update(state='complete', exit_status=0, finished_utc=base.utc())
            base.save(output / 'controller.json', record)
            return 0
        except BaseException as error:
            record.update(state='failed', exit_status=1, error=str(error), finished_utc=base.utc())
            base.save(output / 'controller.json', record)
            raise


base.ROOT, base.SLOTS, base.TOKEN_ENV = ROOT, SLOTS, TOKEN_ENV
base.self_command, base.load_spec, base.commands = self_command, load_spec, commands
base.environment, base.check_ready = environment, check_ready
base.container_command, base.cache_manifest, base.stage_cache = container_command, context_manifest, stage_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('template', 'parse', 'generate-spec', 'specs', 'prepare',
        'start', 'queue', 'gates', 'controller', 'entry', 'worker', 'node-resources', 'container-resources'))
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--slot', choices=(SLOT,))
    parser.add_argument('--token')
    parser.add_argument('--main-ready-evidence')
    parser.add_argument('--runner-tests-passed', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args(argv)
    simple = dict(template=template, parse=parse_commands, gates=predecessor_gate)
    if args.stage in simple:
        result = simple[args.stage]()
    elif args.stage in ('generate-spec', 'specs'):
        result = generate_spec(args.main_ready_evidence, args.runner_tests_passed)
    else:
        if args.spec is None:
            parser.error('--spec is required')
        if args.stage in ('prepare', 'start', 'queue'):
            kwargs = dict(interval=args.interval) if args.stage == 'queue' else {}
            result = globals()[args.stage](args.spec, args.main_ready_evidence, args.runner_tests_passed, **kwargs)
        elif args.stage == 'controller':
            return controller(args.spec)
        else:
            if args.slot is None or (args.stage in ('entry', 'worker') and not args.token):
                parser.error('--slot required; entry/worker also require --token')
            result = globals()[args.stage.replace('-', '_')](args.spec, args.slot, args.token)
            if args.stage in ('entry', 'worker'):
                return result
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
