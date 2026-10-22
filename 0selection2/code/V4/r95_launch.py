"""One-shot R95 queues over the existing allocation-owned R92 A/B slots.

Only a private R93 adapter is rebound. Templates/parsing never authorize GPU
probes; main's scientific lock and explicit readiness remain prerequisites.
"""

import argparse
import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R95_launch'
MODEL_ROOT = DEFAULT_ROOT = REPO / 'code/V4/runs/R95_adaptive_example_sampling'
PROTOCOL_LOCK = MODEL_ROOT / 'protocol_lock.json'
WRAPPER = REPO / 'code/V4/run_v4_r95_launch.sh'
TOKEN_ENV = 'R95_LAUNCH_TOKEN'
ARMS = ('A', 'B')
ARM_SLOTS = dict(A='3090_0', B='4090_0')
PREDECESSORS = dict(A='A', B='B')
R92_LAUNCH = REPO / 'code/V4/runs/R92_launch'
R92_MODELS = REPO / 'code/V4/runs/R92_broad_learning'
SAMPLING = dict(warm_natural_updates=7200, refresh_every_updates=7200,
    proposal_uniform_mixture=.5, proxy='deterministic_full_loss_final_affine_gradient_norm',
    correction='1/(N*q)', normalization='arithmetic_batch_mean_no_self_normalization')
DROPOUT = dict(A=.1, B=.1)
TEST_SOURCES = ('test_r95_launch.py', 'test_r95_reporting.py', 'test_r95_goal.py',
                'test_r95_experiment.py',
                'test_r95_objective.py', 'test_r95_sampling.py')

_spec = importlib.util.spec_from_file_location('code.V4._r95_resource_adapter',
                                             REPO / 'code/V4/r93_launch.py')
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)
base, transport = adapter.base, adapter.transport
DATA, PREPARED = adapter.DATA, adapter.PREPARED
CACHE_SOURCE, CACHE_ROOT, CACHE_EVIDENCE = adapter.CACHE_SOURCE, adapter.CACHE_ROOT, adapter.CACHE_EVIDENCE
INVENTORY_SLOTS = adapter.INVENTORY_SLOTS
SLOTS = {name: dict(INVENTORY_SLOTS[name]) for name in ARM_SLOTS.values()}
_alias, _arm_setting = adapter._alias, adapter._arm_setting
require_confirmation = adapter.require_confirmation


def self_command(stage, spec=None, slot=None, token=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r95_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            argv += [flag, str(value)]
    return argv


def arm_budget(protocol, arm):
    if arm not in ARMS:
        raise ValueError('R95 uses stable neural arms A/B only')
    actual = adapter.adapter.arm_budget(protocol, arm)
    expected = dict(unit='successful_updates', count=57600, warmup_updates=4320,
                    evaluation_every_updates=1440)
    if actual != expected:
        raise ValueError('R95 requires successful_updates57600/warm4320/VAL1440')
    return actual


def locked_protocol():
    if not PROTOCOL_LOCK.is_file():
        raise RuntimeError('Waiting for main final R95 protocol_lock.json')
    value = base.read(PROTOCOL_LOCK)
    required = dict(experiment='R95', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', rows_per_problem=dict(train=81000, val=4500, test=4500),
        batch_size=1024, logical_global_batch=1024, max_updates=57600,
        warmup_updates=4320, eval_every=1440, large_micro_cap=352,
        eval_batch_size=256, dropout=DROPOUT, sampling=SAMPLING, wandb_mode='offline',
        data_root=str(DATA), prepared_root=str(PREPARED))
    if any(value.get(name) != expected for name, expected in required.items()):
        raise ValueError('Use main final R95 original-R77 matched sampling protocol')
    if _alias(value, ('logical_batch', 'logical_global_batch', 'global_batch_size', 'batch_size')) != 1024:
        raise ValueError('Conflicting R95 logical batch controls')
    for arm in ARMS:
        arm_budget(value, arm)
        for names, expected in ((('dropout',), .1),
                (('large_micro_cap', 'large_instance_micro_cap'), 352),
                (('eval_batch_size', 'evaluation_batch'), 256)):
            actual = _arm_setting(value, arm, names)
            if names == ('dropout',) and isinstance(actual, dict):
                actual = actual.get(arm)
            if actual != expected:
                raise ValueError('R95 per-arm dropout/micro/evaluation controls disagree')
    return value


def _sources(protocol):
    """Freeze R95 runtime plus source dependencies, never a future test glob."""
    directory = REPO / 'code/V4'
    paths = set(directory.glob('r95*.py'))
    paths.update(directory / name for name in ('r95_experiment.py', 'r95_reporting.py',
        'r95_goal.py', 'r95_launch.py', 'r93_launch.py', 'r92_launch.py', 'r87_launch.py',
        'r85_launch.py', 'r84_launch.py', 'r81_queue.py', 'r84_experiment.py',
        'r84_reporting.py', 'r84_prepare.py', 'r84_nss_reference.py'))
    paths.update(directory / name for name in TEST_SOURCES)
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
    paths.update((PROTOCOL_LOCK, WRAPPER, CACHE_EVIDENCE, directory / 'run_v4_r84_launch.sh'))
    return sorted(str(path) for path in paths)


def template(arm, protocol=None):
    if arm not in ARMS:
        raise ValueError('R95 uses arms A/B only')
    train = dict(name='train', device='gpu', argv=[
        '{python}', '-B', '-u', '-m', 'code.V4.r95_experiment', 'train',
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
        cpu_count=4, memory_gib=9, wandb_mode='offline', source_files=_sources(protocol or {}),
        protocol=dict(initialization='scratch', training_splits=['train', 'val'], seed=2,
            logical_batch=1024, task_scope='full18', objective='original_R77_CE_comparison_risk_RDrop',
            dropout=.1, sampling=dict(SAMPLING),
            sampling_arm='uniform_replacement' if arm == 'A' else 'inverse_corrected_adaptive',
            arm_budget=arm_budget(protocol, arm) if protocol else dict(unit='successful_updates',
                count=57600, warmup_updates=4320, evaluation_every_updates=1440),
            large_instance_micro_cap=352, evaluation_batch=256),
        stages=[train, test,
            dict(name='LIB', device='gpu', argv=['{python}', '-B', '-u', '-m',
                'code.V4.r95_reporting', 'lib', '--root', '{common_root}', '--arm', arm,
                '--device', '{device}', '--lib-batch-size', '2']),
            dict(name='PDF', device='cpu', argv=['{python}', '-B', '-u', '-m',
                'code.V4.r95_reporting', 'report', '--root', '{common_root}', '--arm', arm,
                '--device', '{device}'])])


def commands(spec, slot):
    if ARM_SLOTS.get(spec['run_id']) != slot or spec['slots'] != [slot]:
        raise ValueError('R95 has fixed slots; never use healthy R94 A/B GPUs')
    values = dict(spec, python=sys.executable)
    return [dict(stage, argv=[arg.format_map(dict(values,
        device='cpu' if stage['device'] == 'cpu' else 'cuda:0')) for arg in stage['argv']])
        for stage in spec['stages']]


def load_spec(path):
    value = adapter.adapter.engine.saved._load_spec(path)
    if value != template(value['run_id'], locked_protocol()):
        raise ValueError('R95 spec differs from final lock, fixed routing or CPU PDF policy')
    return value


def runtime_modules():
    names = ('r95_experiment', 'r95_reporting')
    for name in names:
        if not (REPO / 'code/V4' / (name + '.py')).is_file():
            raise RuntimeError('Waiting for final runtime: ' + name)
    modules = [importlib.import_module('code.V4.' + name) for name in names]
    if any(tuple(module.ARMS) != ARMS for module in modules):
        raise RuntimeError('Final R95 runner/reporter ARMS must remain A/B')
    return modules


def comparison_command():
    return [sys.executable, '-B', '-u', '-m', 'code.V4.r95_reporting', 'compare',
            '--root', str(MODEL_ROOT), '--arm', 'A', '--device', 'cpu']


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
                    raise RuntimeError('R95 cannot implicitly resume a trained checkpoint')
            elif stage['name'] == 'LIB':
                expected['lib_batch_size'] = 2
            if any(getattr(parsed, name, None) != value for name, value in expected.items()):
                raise RuntimeError('Final runtime parser differs from R95 stage contract')
            rows.append(dict(arm=arm, slot=slot, command=argv[0], argv=argv, parsed=True))
    argv = comparison_command()[5:]
    parsed = reporter.parser().parse_args(argv)
    if (parsed.command, parsed.root, parsed.arm, parsed.device) != ('compare', MODEL_ROOT, 'A', 'cpu'):
        raise RuntimeError('Final comparison parser differs from R95 CPU contract')
    rows.append(dict(arm=None, slot=None, command='compare', argv=argv, parsed=True))
    return dict(commands=rows, parsed_count=len(rows), training_started=False,
        resource_probe_performed=False, source_hashes_frozen=False, protocol_written=False)


def predecessor_gate(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R95 arm')
    model, directory = R92_MODELS / PREDECESSORS[arm], R92_LAUNCH / PREDECESSORS[arm]
    result = dict(experiment='R92', predecessor=PREDECESSORS[arm], complete=False, failed=False)
    try:
        for name, stages in (('worker.json', ('train', 'TEST', 'LIB')),
                ('controller.json', ('allocation_GPU_pipeline', 'PDF'))):
            record = base.read(directory / name)
            if (record.get('state') == 'failed' or record.get('exit_status') not in (None, 0)
                    or any(s.get('state') == 'failed' or s.get('exit_status') not in (None, 0)
                           for s in record.get('stages', []))):
                raise ValueError('R92 predecessor failed; no implicit repair/retry')
            if (record.get('state') != 'complete' or record.get('exit_status') != 0
                    or not adapter._stages_complete(record, stages)):
                raise FileNotFoundError('Waiting for R92' + arm + ' full train/TEST/LIB/PDF controller rc0')
        lock = base.read(model / 'final_test_lock.json')
        training = base.read(model / 'training_complete.json')
        if (lock.get('experiment') != 'R92' or lock.get('arm') != arm
                or lock.get('checkpoint') != str(model / 'best.pt')
                or lock.get('checkpoint_sha256') != base.sha(model / 'best.pt')
                or lock.get('selected_without_test') is not True
                or lock.get('per_problem_assembly') is not False
                or training.get('successful_updates') != 57600
                or training.get('selected_best') != lock
                or training.get('state_hash') != lock.get('state_hash')):
            raise ValueError('R92 predecessor fresh neural budget/unique VAL lock is inconsistent')
        adapter._post_complete(model, lock)
        result.update(complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
            reason='R92' + arm + ' same-lock full delivery verified; actual idle GPU still required')
    except FileNotFoundError as error:
        result['reason'] = str(error)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update(failed=True, reason=str(error))
    return result


def prepare(path, evidence=None, tested=False):
    require_confirmation(evidence, tested)
    parse_commands()
    result = adapter.adapter.engine._prepare(path)
    result.update(experiment='R95', cache_mode='existing_read_only', automatic_retry=False,
        main_thread_ready=False, main_ready_evidence=evidence,
        predecessor=predecessor_gate(result['spec']['run_id']), resource_probe_performed=False)
    base.save(base.receipts(result['spec']) / 'plan.json', result)
    return result


def reserved_cpus(node, exclude):
    occupied = set()
    for root in (ROOT, R92_LAUNCH, REPO / 'code/V4/runs/R93_launch', REPO / 'code/V4/runs/R94_launch'):
        for path in root.glob('*/claim.json'):
            if path.parent == exclude:
                continue
            claim = base.read(path)
            controller = path.with_name('controller.json')
            record = base.read(controller) if controller.is_file() else {}
            if (claim.get('node') == node and not record.get('gpu_finished_utc')
                    and record.get('state') not in ('gpu_complete', 'reporting', 'complete', 'failed')):
                occupied.update(claim['cpus'])
    return occupied


def queue_prepared(path, evidence, tested):
    require_confirmation(evidence, tested)
    spec = load_spec(path)
    plan = base.read(base.receipts(spec) / 'plan.json')
    if plan.get('spec') != spec or plan.get('main_ready_evidence') != evidence:
        raise RuntimeError('Queue requires the final main-authorized prepared CLI')
    if any(base.sha(name) != digest for name, digest in plan['source_sha256'].items()):
        raise RuntimeError('Prepared source changed before queuing')
    return spec, plan


def queue(path, evidence, tested, interval=60):
    if interval < 30:
        raise ValueError('Use a quiet queue interval of at least30 seconds')
    spec, _ = queue_prepared(path, evidence, tested)
    output = base.receipts(spec)
    with base.lock(output / 'queue.lock'):
        if any((output / name).exists() for name in ('attempt.json', 'controller.json', 'claim.json')):
            raise RuntimeError('R95 arm already attempted; no automatic retry')
        while True:
            gate = predecessor_gate(spec['run_id'])
            base.save(output / 'queue_status.json', dict(experiment='R95', arm=spec['run_id'],
                state='ready' if gate['complete'] else 'failed' if gate['failed'] else 'waiting_predecessor',
                updated_utc=base.utc(), predecessor=gate, training_started=False, automatic_retry=False))
            if gate['failed']:
                raise RuntimeError(gate['reason'])
            if gate['complete']:
                return start(path, evidence, tested)
            time.sleep(interval)


def start(path, evidence=None, tested=False):
    require_confirmation(evidence, tested)
    parse_commands()
    adapter.require_predecessor(path)
    return adapter._start(path, evidence, tested)


def write_specs(evidence=None, tested=False):
    return adapter.write_specs(evidence, tested)


for _name, _value in dict(ROOT=ROOT, MODEL_ROOT=MODEL_ROOT, PROTOCOL_LOCK=PROTOCOL_LOCK,
        WRAPPER=WRAPPER, TOKEN_ENV=TOKEN_ENV, ARMS=ARMS, ARM_SLOTS=ARM_SLOTS, SLOTS=SLOTS,
        self_command=self_command, locked_protocol=locked_protocol, arm_budget=arm_budget,
        template=template, _sources=_sources, commands=commands, load_spec=load_spec,
        runtime_modules=runtime_modules, comparison_command=comparison_command,
        parse_commands=parse_commands, predecessor_gate=predecessor_gate,
        reserved_cpus=reserved_cpus, prepare=prepare, start=start).items():
    setattr(adapter, _name, _value)
base.ROOT, base.SLOTS, base.TOKEN_ENV = ROOT, SLOTS, TOKEN_ENV
base.self_command, base.load_spec, base.commands = self_command, load_spec, commands
base.reserved_cpus, base.start = reserved_cpus, start
base.session_name = lambda value: 'r95_' + value['run_id']


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == 'queue':
        parser = argparse.ArgumentParser(description='Wait for fixed R92 predecessor, then start once')
        parser.add_argument('stage', choices=('queue',))
        parser.add_argument('--spec', type=Path, required=True)
        parser.add_argument('--main-ready-evidence', required=True)
        parser.add_argument('--runner-tests-passed', action='store_true')
        parser.add_argument('--interval', type=int, default=60)
        args = parser.parse_args(values)
        return queue(args.spec, args.main_ready_evidence, args.runner_tests_passed, args.interval)
    return adapter.main(values)


if __name__ == '__main__':
    raise SystemExit(main())
