"""Explicit, one-shot R93 A/B launches in two existing allocation-owned slots."""

import argparse
import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R93_launch'
MODEL_ROOT = REPO / 'code/V4/runs/R93_plain_classification'
PROTOCOL_LOCK = MODEL_ROOT / 'protocol_lock.json'
WRAPPER = REPO / 'code/V4/run_v4_r93_launch.sh'
TOKEN_ENV = 'R93_LAUNCH_TOKEN'
ARMS = ('A', 'B')
ARM_SLOTS = dict(A='3090_1', B='4090_1')
PREDECESSORS = dict(A='C', B='D')
R92_LAUNCH = REPO / 'code/V4/runs/R92_launch'
R92_MODELS = REPO / 'code/V4/runs/R92_broad_learning'

# R92 already isolates the R84/R85 helpers and transport. Patch only this copy.
_spec = importlib.util.spec_from_file_location('code.V4._r93_adapter', REPO / 'code/V4/r92_launch.py')
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)
base, transport = adapter.base, adapter.transport
DATA, PREPARED = adapter.DATA, adapter.PREPARED
CACHE_SOURCE, CACHE_ROOT, CACHE_EVIDENCE = adapter.CACHE_SOURCE, adapter.CACHE_ROOT, adapter.CACHE_EVIDENCE
INVENTORY_SLOTS = {name: dict(item) for name, item in adapter.SLOTS.items()}
SLOTS = {name: dict(INVENTORY_SLOTS[name]) for name in ARM_SLOTS.values()}
_start, _worker, _check_ready = adapter._start, adapter.engine._worker, adapter._check_ready
_environment = adapter.environment
_alias, _arm_setting, _stages_complete = adapter._alias, adapter._arm_setting, adapter._stages_complete


def self_command(stage, spec=None, slot=None, token=None):
    command = [sys.executable, '-B', '-u', '-m', 'code.V4.r93_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            command += [flag, str(value)]
    return command


def environment(slot=None, token=None):
    return dict(_environment(slot, token), WANDB_MODE='offline')


def container_command(stage, slot, spec=None, token=None):
    item = SLOTS[slot]
    command = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid())]
    for name, value in environment(slot, token).items():
        if name in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'PYTHONUNBUFFERED', 'MPLBACKEND',
                'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_PROJECT', 'WANDB_ENTITY',
                'WANDB_MODE', 'SLURM_JOB_ID', TOKEN_ENV):
            command += ['-e', name + '=' + value]
    return command + [transport.CONTAINERS[item['node']][1], 'bash', str(WRAPPER),
                      *self_command(stage, spec, slot, token)[5:]]


def arm_budget(protocol, arm):
    if arm not in ARMS:
        raise ValueError('R93 uses stable neural arms A/B only')
    budget = adapter.arm_budget(protocol, arm)
    expected = dict(unit='successful_updates', count=57600, warmup_updates=4320,
                    evaluation_every_updates=1440)
    if budget != expected:
        raise ValueError('Both R93 neural arms require successful_updates57600/warm4320/VAL1440')
    return budget


def locked_protocol():
    if not PROTOCOL_LOCK.is_file():
        raise RuntimeError('Waiting for main final R93 protocol_lock.json')
    value = base.read(PROTOCOL_LOCK)
    if value.get('experiment') != 'R93' or value.get('locked') is not True or tuple(value.get('arms', ())) != ARMS:
        raise ValueError('Main must lock R93 with stable arms A/B')
    if (value.get('seed') != 2 or value.get('task_scope') != 'full18'
            or _alias(value, ('logical_batch', 'logical_global_batch', 'global_batch_size', 'batch_size')) != 1024):
        raise ValueError('R93 requires full18, seed2 and logical batch1024')
    if (value.get('data_root') != str(DATA) or value.get('prepared_root') != str(PREPARED)
            or value.get('rows_per_problem') != dict(train=81000, val=4500, test=4500)):
        raise ValueError('R93 reuses frozen NEW90k data and R84 preparation')
    if value.get('wandb_mode') != 'offline' or value.get('dropout') != dict(A=0.1, B=0.0):
        raise ValueError('R93 requires explicit offline W&B and dropout A=0.1/B=0')
    for arm in ARMS:
        arm_budget(value, arm)
        if (_arm_setting(value, arm, ('large_micro_cap', 'large_instance_micro_cap')) != 352
                or _arm_setting(value, arm, ('eval_batch_size', 'evaluation_batch')) != 256
                or _arm_setting(value, arm, ('dropout',)) != value['dropout'][arm]):
            raise ValueError('R93 requires locked micro352/eval256 and consistent per-arm dropout')
    return value


def _sources(protocol):
    """Freeze local Python dependencies, not dataset arrays or live R92 state."""
    directory = REPO / 'code/V4'
    paths = set(directory.glob('r93*.py'))
    paths.update(directory / name for name in ('r93_experiment.py', 'r93_reporting.py',
        'r92_launch.py', 'r87_launch.py', 'r85_launch.py', 'r84_launch.py', 'r81_queue.py',
        'r84_experiment.py', 'r84_reporting.py', 'r84_prepare.py', 'r84_nss_reference.py'))
    for name in protocol.get('launch', {}).get('source_files', []):
        path = Path(name)
        path = path if path.is_absolute() else REPO / path
        if not path.resolve().is_relative_to(directory) or path.suffix != '.py':
            raise ValueError('Additional runtime sources must be V4 Python modules')
        paths.add(path)
    pending, seen = list(paths), set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        package = list(path.relative_to(REPO).parts[:-1])
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
    paths.update(PREPARED / name for name in ('prepared.json', 'geometry_stats.json',
        'common_full_train_sbs.json', 'identicalinit_seed2.pt'))
    paths.update((PROTOCOL_LOCK, WRAPPER, directory / 'test_r93_launch.py', CACHE_EVIDENCE,
                  directory / 'run_v4_r84_launch.sh'))
    return sorted(str(path) for path in paths)


def template(arm, protocol=None):
    if arm not in ARMS:
        raise ValueError('R93 uses stable neural arms A/B only')
    protocol = {} if protocol is None else protocol
    train = dict(name='train', device='gpu', argv=[
        '{python}', '-B', '-u', '-m', 'code.V4.r93_experiment', 'train',
        '--root', '{common_root}', '--arm', arm, '--prepared-root', '{prepared_root}',
        '--data-root', '{data_root}', '--cache-root', '{cache_root}',
        '--protocol-lock', str(PROTOCOL_LOCK), '--gpu-type', SLOTS[ARM_SLOTS[arm]]['gpu_type'],
        '--large-micro-cap', '352', '--eval-batch-size', '256',
        '--device', '{device}', '--wandb-mode', 'offline'])
    test = dict(name='TEST', device='gpu', argv=list(train['argv']))
    test['argv'][5] = 'test'
    return dict(schema_version=1, run_id=arm, run_dir=str(MODEL_ROOT / arm),
        common_root=str(MODEL_ROOT), data_root=str(DATA), prepared_root=str(PREPARED),
        cache_source=str(CACHE_SOURCE), cache_root=str(CACHE_ROOT), slots=[ARM_SLOTS[arm]],
        cpu_count=4, memory_gib=9, wandb_mode='offline', source_files=_sources(protocol),
        protocol=dict(initialization='scratch', training_splits=['train', 'val'], seed=2,
            logical_batch=1024, task_scope='full18', objective='plain_native_CE_only',
            dropout=0.1 if arm == 'A' else 0.0, arm_budget=arm_budget(protocol, arm) if protocol else
            dict(unit='successful_updates', count=57600, warmup_updates=4320, evaluation_every_updates=1440),
            large_instance_micro_cap=352, evaluation_batch=256),
        stages=[train, test,
            dict(name='LIB', device='gpu', argv=['{python}', '-B', '-u', '-m',
                'code.V4.r93_reporting', 'lib', '--root', '{common_root}', '--arm', arm,
                '--device', '{device}', '--lib-batch-size', '2']),
            dict(name='PDF', device='cpu', argv=['{python}', '-B', '-u', '-m',
                'code.V4.r93_reporting', 'report', '--root', '{common_root}', '--arm', arm,
                '--device', '{device}'])])


def commands(spec, slot):
    if ARM_SLOTS.get(spec['run_id']) != slot or spec['slots'] != [slot]:
        raise ValueError('R93 arms have fixed slots; no fallback GPU or healthy R92 slot')
    values = dict(spec, python=sys.executable)
    return [dict(stage, argv=[arg.format_map(dict(values,
        device='cpu' if stage['device'] == 'cpu' else 'cuda:0')) for arg in stage['argv']])
        for stage in spec['stages']]


def comparison_command():
    return [sys.executable, '-B', '-u', '-m', 'code.V4.r93_reporting', 'compare',
            '--root', str(MODEL_ROOT), '--arm', 'A', '--device', 'cpu']


def load_spec(path):
    value = adapter.engine.saved._load_spec(path)
    if value != template(value['run_id'], locked_protocol()):
        raise ValueError('R93 spec differs from final main lock, fixed routing or CPU PDF policy')
    return value


def runtime_modules():
    names = ('r93_experiment', 'r93_reporting')
    for name in names:
        if not (REPO / 'code/V4' / (name + '.py')).is_file():
            raise RuntimeError('Waiting for final runtime: ' + name)
    modules = [importlib.import_module('code.V4.' + name) for name in names]
    if any(tuple(module.ARMS) != ARMS for module in modules):
        raise RuntimeError('Final runner/reporter ARMS must remain A/B')
    return modules


def parse_commands(protocol=None):
    protocol = locked_protocol() if protocol is None else protocol
    runner, reporter = runtime_modules()
    rows = []
    for arm, slot in ARM_SLOTS.items():
        for stage in commands(template(arm, protocol), slot):
            argv = stage['argv'][5:]
            module = runner if stage['name'] in ('train', 'TEST') else reporter
            parsed = module.parser().parse_args(argv)
            expected = dict(command=argv[0], root=MODEL_ROOT, arm=arm,
                            device='cpu' if stage['device'] == 'cpu' else 'cuda:0')
            if module is runner:
                expected.update(prepared_root=PREPARED, data_root=DATA, cache_root=CACHE_ROOT,
                    protocol_lock=PROTOCOL_LOCK, gpu_type=SLOTS[slot]['gpu_type'],
                    large_micro_cap=352, eval_batch_size=256, wandb_mode='offline')
                if getattr(parsed, 'resume', False):
                    raise RuntimeError('R93 cannot implicitly resume a trained checkpoint')
            elif stage['name'] == 'LIB':
                expected['lib_batch_size'] = 2
            if any(getattr(parsed, key, None) != value for key, value in expected.items()):
                raise RuntimeError('Final runtime parser differs from R93 stage contract')
            rows.append(dict(arm=arm, slot=slot, command=argv[0], argv=argv, parsed=True))
    argv = comparison_command()[5:]
    parsed = reporter.parser().parse_args(argv)
    if (parsed.command, parsed.root, parsed.arm, parsed.device) != ('compare', MODEL_ROOT, 'A', 'cpu'):
        raise RuntimeError('Final comparison parser differs from R93 CPU contract')
    rows.append(dict(arm=None, slot=None, command='compare', argv=argv, parsed=True))
    return dict(commands=rows, parsed_count=len(rows), training_started=False,
                source_hashes_frozen=False, protocol_written=False, resource_probe_performed=False)


def require_confirmation(evidence, tested):
    return adapter.require_confirmation(evidence, tested)


def write_specs(evidence=None, tested=False):
    require_confirmation(evidence, tested)
    protocol = locked_protocol()
    parsed = parse_commands(protocol)
    specs = {ROOT / 'specs' / (arm + '.json'): template(arm, protocol) for arm in ARMS}
    for path, value in specs.items():
        if path.exists() and base.read(path) != value:
            raise RuntimeError('Existing R93 spec differs; no implicit overwrite')
    for path, value in specs.items():
        if not path.exists():
            base.save(path, value, exclusive=True)
    return dict(parsed, specs=[str(path) for path in specs], main_ready_evidence=evidence, slots=ARM_SLOTS)


def prepare(path, evidence=None, tested=False):
    require_confirmation(evidence, tested)
    parse_commands()
    result = adapter.engine._prepare(path)
    result.update(experiment='R93', cache_mode='existing_read_only', automatic_retry=False,
        main_thread_ready=False, main_ready_evidence=evidence,
        predecessor=predecessor_gate(result['spec']['run_id']), resource_probe_performed=False)
    base.save(base.receipts(result['spec']) / 'plan.json', result)
    return result


def _checkpoint(model, arm):
    lock = base.read(model / 'final_test_lock.json')
    if (lock.get('experiment') != 'R92' or lock.get('arm') != arm
            or lock.get('checkpoint') != str(model / 'best.pt')
            or lock.get('checkpoint_sha256') != base.sha(model / 'best.pt')
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False):
        raise ValueError('R92 current best/VAL lock identity is invalid')
    training = base.read(model / 'training_complete.json')
    if (training.get('full18') is not True or len(training.get('tree_counts', {})) != 18
            or training.get('selected_best') != lock):
        raise ValueError('R92 predecessor must finish all18 with one unchanged VAL lock')
    return lock


def _post_complete(model, lock):
    test = base.read(model / 'test_results.json')
    lib = base.read(model / 'lib_stage_complete.json')
    post = base.read(model / 'postprocess_status.json')
    if (test.get('complete') is not True or test.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or lib.get('complete') is not True or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or set(lib.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}
            or post.get('complete') is not True or post.get('phase') != 'complete'):
        raise FileNotFoundError('Waiting for same-lock R92 TEST/LIB/postprocess complete')
    for key, name in (('model_statistics_pdf', 'MODEL_STATISTICS.pdf'),
                      ('selection_distributions_pdf', 'selection_distributions.pdf'),
                      ('target_recovery_pdf', 'target_recovery_and_train.pdf')):
        pdf = model / 'full_model_statistics' / name
        if post.get(key) != str(pdf) or not pdf.is_file() or not pdf.stat().st_size:
            raise FileNotFoundError('Waiting for completed R92 PDF: ' + str(pdf))


def _c_recovery(model, directory, worker, controller):
    receipt_path = model / 'recovery_complete.json'
    receipt = base.read(receipt_path)
    if (receipt.get('complete') is not True or receipt.get('state') != 'complete'
            or receipt.get('protected_unchanged') is not True or receipt.get('CPU_only') is not True
            or receipt.get('TEST_LIB_PDF_complete') is not True
            or receipt.get('original_failure_preserved') is not True
            or receipt.get('optimizer_updates') != 0 or receipt.get('fitted_trees') != 0
            or any(receipt.get(key) is not False for key in
                   ('automatic_retry', 'model_checkpoint_replaced', 'inference_rule_changed'))):
        raise ValueError('R92C requires explicit completed zero-fit CPU recovery, not retraining')
    before, after = receipt.get('protected_before'), receipt.get('protected_after')
    required = [model / name for name in ('best.pt', 'protocol.json', 'error.json')]
    required += [directory / name for name in ('worker.json', 'controller.json')]
    required += list((model / 'tree_models').glob('*'))
    if not before or before != after or any(str(path) not in before for path in required):
        raise ValueError('R92C recovery lacks unchanged original failure/model SHA evidence')
    for name, digest in before.items():
        path = Path(name)
        if (not any(path.resolve().is_relative_to(root.resolve()) for root in (model, directory))
                or base.sha(path) != digest):
            raise ValueError('R92C original failure/model SHA changed: ' + str(path))
    error = base.read(model / 'error.json')
    if (error.get('message') != 'Selected-bank VAL disagrees with staged CPU prediction selection'
            or worker.get('state') != 'failed' or worker.get('exit_status') != 1
            or controller.get('state') != 'failed' or controller.get('exit_status') != 1
            or len(worker.get('stages', [])) != 1
            or worker['stages'][0].get('name') != 'train'
            or worker['stages'][0].get('state') != 'failed' or worker['stages'][0].get('exit_status') != 1
            or len(controller.get('stages', [])) != 1
            or controller['stages'][0].get('name') != 'allocation_GPU_pipeline'
            or controller['stages'][0].get('state') != 'failed'
            or controller['stages'][0].get('exit_status') != 1):
        raise ValueError('Only the preserved pre-VAL-lock R92C consistency failure is recoverable')
    lock = _checkpoint(model, 'C')
    stage = receipt.get('test_stage', {})
    if (receipt.get('selected_best') != lock or stage.get('complete') is not True
            or stage.get('protected_unchanged') is not True or stage.get('CPU_only') is not True
            or stage.get('formal_VAL_lock_sha256') != base.sha(model / 'final_test_lock.json')
            or stage.get('optimizer_updates') != 0 or stage.get('fitted_trees') != 0):
        raise ValueError('R92C recovery selected best/TEST lock SHA is not the current unchanged lock')
    return dict(path=str(receipt_path), checkpoint_sha256=lock['checkpoint_sha256'],
                original_failures_preserved=True), lock


def predecessor_gate(arm):
    if arm not in ARMS:
        raise ValueError('R93 uses stable neural arms A/B only')
    previous = PREDECESSORS[arm]
    directory, model = R92_LAUNCH / previous, R92_MODELS / previous
    result = dict(predecessor=previous, experiment='R92', complete=False, failed=False)
    try:
        worker, controller = (base.read(directory / name) for name in ('worker.json', 'controller.json'))
        if arm == 'A':
            result['recovery'], lock = _c_recovery(model, directory, worker, controller)
        else:
            for record, names in ((worker, ('train', 'TEST', 'LIB')),
                                  (controller, ('allocation_GPU_pipeline', 'PDF'))):
                if (record.get('state') == 'failed' or record.get('exit_status') not in (None, 0)
                        or any(stage.get('state') == 'failed' or stage.get('exit_status') not in (None, 0)
                               for stage in record.get('stages', []))):
                    raise ValueError('R92D failed; no automatic repair/retry')
                if record.get('state') != 'complete' or record.get('exit_status') != 0 or not _stages_complete(record, names):
                    raise FileNotFoundError('Waiting for R92D train/TEST/LIB and controller PDF rc0')
            lock = _checkpoint(model, 'D')
        _post_complete(model, lock)
        result.update(complete=True, reason='R92' + previous + ' full delivery verified')
    except FileNotFoundError as error:
        result['reason'] = str(error)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update(failed=True, reason=str(error))
    return result


def require_predecessor(path):
    gate = predecessor_gate(load_spec(path)['run_id'])
    if not gate['complete']:
        raise RuntimeError(gate['reason'])


def check_ready(path):
    return _check_ready(path)


def authorize_resources(path, slot, token=None, claimed=False):
    if path is None:
        raise RuntimeError('No resource probe before explicit main-ready authorization')
    spec, plan = check_ready(path)
    if spec['slots'] != [slot] or ARM_SLOTS[spec['run_id']] != slot:
        raise RuntimeError('Resource authorization is restricted to the fixed R93 slot')
    require_predecessor(path)
    if claimed:
        claim = base.read(base.receipts(spec) / 'claim.json')
        if not token or claim['slot'] != slot or claim['token'] != token:
            raise RuntimeError('R93 claimed-slot token mismatch')
    return spec, plan


def container_resources(path, slot, token=None):
    authorize_resources(path, slot, token)
    identity = base.assert_container(slot)
    return dict(container_id=identity, cpus=sorted(os.sched_getaffinity(0)),
        memory=base.memory_inventory(identity),
        gpus={name: transport.gpu_inventory(item) for name, item in INVENTORY_SLOTS.items()
              if item['node'] == SLOTS[slot]['node']})


def node_resources(path, slot, token=None):
    authorize_resources(path, slot, token)
    item = SLOTS[slot]
    if (base.socket.gethostname().split('.')[0] != item['node']
            or os.environ.get('SLURM_JOB_ID') != str(item['job'])
            or f'/job_{item["job"]}/' not in Path('/proc/self/cgroup').read_text()):
        raise RuntimeError('Node entry requires the existing authorized allocation CPU step')
    allocation, container = transport.allocation(item), transport.container_inspect(item)
    nested = json.loads(transport.capture(container_command('container-resources', slot, path, token), timeout=60))
    if (nested['container_id'] != container['id'] or not set(nested['cpus']) <= set(allocation['cpus'])):
        raise RuntimeError('Container allocation identity or CPU set changed')
    gpu_pids = {int(row[1]) for gpu in nested['gpus'].values() for row in gpu['compute_processes']}
    processes, found = [], set()
    for process in Path('/proc').glob('[0-9]*'):
        try:
            details = dict(line.split(':', 1) for line in (process / 'status').read_text().splitlines() if ':' in line)
            pids = {int(value) for value in details.get('NSpid', process.name).split()}
            matched = gpu_pids & pids
            if not matched and container['id'] not in (process / 'cgroup').read_text():
                continue
            argv = [value for value in (process / 'cmdline').read_bytes().decode(errors='replace').split('\0') if value]
            if not argv or 'container-resources' in argv or 'node-resources' in argv:
                continue
            if any(Path(value).name == 'conda' for value in argv) and 'hook' in argv:
                continue
            if not matched and not Path(argv[0]).name.startswith('python'):
                continue
            found.update(matched)
            processes.append(dict(pid=int(process.name), namespace_pids=sorted(pids),
                cpus=sorted(os.sched_getaffinity(int(process.name))), command=argv))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if gpu_pids - found:
        raise RuntimeError('Cannot attribute live GPU processes to CPU affinity')
    return dict(captured_utc=base.utc(), node=item['node'], allocation=allocation,
        container=container, memory=nested['memory'], processes=processes, gpus=nested['gpus'],
        allowed_cpus=sorted(set(nested['cpus']) & set(container['allowed_cpus'])))


def start(path, evidence, tested):
    require_confirmation(evidence, tested)
    parse_commands()
    require_predecessor(path)
    return _start(path, evidence, tested)


def entry(path, slot, token):
    spec, _ = authorize_resources(path, slot, token, claimed=True)
    claim = base.read(base.receipts(spec) / 'claim.json')
    probe = node_resources(path, slot, token)
    gpu = probe['gpus'][slot]
    occupied = {cpu for row in probe['processes'] for cpu in row['cpus']}
    if gpu['compute_processes'] or gpu['memory_used_mib'] > 512 or set(claim['cpus']) & occupied:
        raise RuntimeError('Claimed GPU/CPU became busy before entry; no retry')
    base.save(base.receipts(spec) / 'entry.json', probe, exclusive=True)
    return base.subprocess.run(container_command('worker', slot, path, token), check=False).returncode


def worker(path, slot, token):
    spec, _ = authorize_resources(path, slot, token, claimed=True)
    base.assert_container(slot)
    if os.environ.get(TOKEN_ENV) != token or os.environ.get('CUDA_VISIBLE_DEVICES') != SLOTS[slot]['uuid']:
        raise RuntimeError('R93 worker requires exact UUID and its main-authorized token')
    base.save(base.receipts(spec) / 'wandb_lookup.json', dict(selected_mode='offline',
        credential_lookup_performed=False, reason='Explicit offline policy; no credential hunt'), exclusive=True)
    return _worker(path, slot, token)


def reserved_cpus(node, exclude):
    occupied = set()
    for root in (ROOT, R92_LAUNCH):
        for path in root.glob('*/claim.json'):
            if path.parent == exclude:
                continue
            claim = base.read(path)
            controller_path = path.with_name('controller.json')
            record = base.read(controller_path) if controller_path.is_file() else {}
            if (claim.get('node') == node and not record.get('gpu_finished_utc')
                    and record.get('state') not in ('gpu_complete', 'reporting', 'complete', 'failed')):
                occupied.update(claim['cpus'])
    return occupied


def pipeline(path, spec, slot, claim):
    output = base.receipts(spec)
    record = dict(started_utc=base.utc(), slot=slot, node=SLOTS[slot]['node'],
                  state='claimed', stages=[], automatic_retry=False)
    base.save(output / 'controller.json', record, exclusive=True)
    try:
        stage = dict(name='allocation_GPU_pipeline', device='gpu',
            argv=base.srun_command('entry', slot, path, claim['token'], cpus=4, memory='9G'))
        base.run_stage(stage, spec, record, output / 'controller.json', environment())
        worker_record = base.read(output / 'worker.json')
        if (worker_record.get('state') != 'complete' or worker_record.get('exit_status') != 0
                or not _stages_complete(worker_record, ('train', 'TEST', 'LIB'))):
            raise RuntimeError('CPU PDF requires completed train/TEST/LIB rc0')
        record.update(state='gpu_complete', exit_status=0, gpu_finished_utc=base.utc())
        base.save(output / 'controller.json', record)
        return record
    except BaseException as error:
        record.update(state='failed', exit_status=1, error=str(error), finished_utc=base.utc())
        base.save(output / 'controller.json', record)
        raise


def controller(path):
    if base.socket.gethostname().split('.')[0] != 'admin01' or not base.FONT.is_file():
        raise RuntimeError('Controller/PDF requires ADMIN01 CPU with its CJK font')
    spec, _ = check_ready(path)
    slot = ARM_SLOTS[spec['run_id']]
    authorize_resources(path, slot)
    output, item = base.receipts(spec), SLOTS[slot]
    with base.lock(output / 'controller.lock'):
        if any((output / name).exists() for name in ('claim.json', 'controller.json', 'attempt.json')):
            raise RuntimeError('No duplicate launch or implicit retry')
        base.save(output / 'attempt.json', dict(started_utc=base.utc(), automatic_retry=False), exclusive=True)
        # Use the original cross-experiment UUID mutex inode, never a new GPU pool.
        mutex = transport.ROOT / f'queue_gpu_{item["uuid"]}.json'
        with base.lock(mutex, existing=mutex.exists()):
            with base.lock(transport.ROOT / (item['node'] + '.cpu.lock')):
                probe = json.loads(transport.capture(base.srun_command('node-resources', slot, path), timeout=90))
                gpu = probe['gpus'][slot]
                occupied = {cpu for row in probe['processes'] for cpu in row['cpus']}
                occupied |= reserved_cpus(item['node'], output)
                cpus = sorted(set(probe['allowed_cpus']) - occupied)[:4]
                if (gpu['compute_processes'] or gpu['memory_used_mib'] > 512 or gpu['utilization_pct']
                        or len(cpus) != 4
                        or probe['memory']['headroom_excluding_reclaimable_cache_bytes'] < 9 * 2**30):
                    raise RuntimeError('Fixed R93 GPU/CPU/memory busy; leave healthy runs alone; no retry')
                check_ready(path)
                require_predecessor(path)
                claim = dict(slot=slot, node=item['node'], gpu_uuid=item['uuid'], cpus=cpus,
                    token=base.uuid.uuid4().hex, claimed_utc=base.utc(), resources=probe)
                base.save(output / 'claim.json', claim, exclusive=True)
            record = pipeline(path, spec, slot, claim)
        # The allocation step exited, all GPU work ended, and UUID flock is gone.
        try:
            check_ready(path)
            record['state'] = 'reporting'
            base.save(output / 'controller.json', record)
            for stage in commands(spec, slot):
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
base.container_command = container_command
base.start, base.controller, base.entry, base.worker = start, controller, entry, worker
base.node_resources, base.container_resources = node_resources, container_resources
base.pipeline, base.reserved_cpus = pipeline, reserved_cpus
base.session_name = lambda value: 'r93_' + value['run_id']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('template', 'parse', 'specs', 'prepare', 'start',
        'controller', 'entry', 'worker', 'gates', 'node-resources', 'container-resources'))
    parser.add_argument('--arm', choices=ARMS, default='A')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--slot', choices=tuple(SLOTS))
    parser.add_argument('--token')
    parser.add_argument('--main-ready-evidence')
    parser.add_argument('--runner-tests-passed', action='store_true')
    args = parser.parse_args(argv)
    if args.stage in ('prepare', 'start', 'controller', 'entry', 'worker', 'node-resources', 'container-resources') and not args.spec:
        parser.error('--spec required; resource probes require main-ready authorization')
    if args.stage in ('entry', 'worker', 'node-resources', 'container-resources') and not args.slot:
        parser.error('--slot required')
    if args.stage in ('entry', 'worker') and not args.token:
        parser.error('--token required')
    if args.stage == 'template':
        value = template(args.arm)
    elif args.stage == 'parse':
        value = parse_commands()
    elif args.stage == 'specs':
        value = write_specs(args.main_ready_evidence, args.runner_tests_passed)
    elif args.stage == 'prepare':
        value = prepare(args.spec, args.main_ready_evidence, args.runner_tests_passed)
    elif args.stage == 'start':
        value = start(args.spec, args.main_ready_evidence, args.runner_tests_passed)
    elif args.stage == 'gates':
        value = {arm: predecessor_gate(arm) for arm in ARMS}
    elif args.stage == 'node-resources':
        value = node_resources(args.spec, args.slot, args.token)
    elif args.stage == 'container-resources':
        value = container_resources(args.spec, args.slot, args.token)
    else:
        return dict(controller=lambda: controller(args.spec),
            entry=lambda: entry(args.spec, args.slot, args.token),
            worker=lambda: worker(args.spec, args.slot, args.token))[args.stage]()
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
