"""One-shot queued R94 launches, reusing only existing allocation-owned slots.

The private resource adapter retains the established UUID/cgroup/CPU/memory
guards. No active R92/R93 module or process is changed by this adapter.
"""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R94_launch'
MODEL_ROOT = REPO / 'code/V4/runs/R94_pretrained_node_fusion'
PROTOCOL_LOCK = MODEL_ROOT / 'protocol_lock.json'
WRAPPER = REPO / 'code/V4/run_v4_r94_launch.sh'
TOKEN_ENV = 'R94_LAUNCH_TOKEN'
ARMS = ('A', 'B')
ARM_SLOTS = dict(A='3090_1', B='4090_1')
PREDECESSOR_ROOT = REPO / 'code/V4/runs/R93_plain_classification'
PREDECESSOR_LAUNCH = REPO / 'code/V4/runs/R93_launch'

_spec = importlib.util.spec_from_file_location('code.V4._r94_resource_adapter',
                                             REPO / 'code/V4/r93_launch.py')
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)
base, transport = adapter.base, adapter.transport
DATA, PREPARED = adapter.DATA, adapter.PREPARED
CACHE_SOURCE, CACHE_ROOT = adapter.CACHE_SOURCE, adapter.CACHE_ROOT
SLOTS = {name: dict(adapter.INVENTORY_SLOTS[name]) for name in ARM_SLOTS.values()}
_sources_previous = adapter._sources
_start, _check_ready = adapter._start, adapter._check_ready
_controller, _parse_main = adapter.controller, adapter.main


def self_command(stage, spec=None, slot=None, token=None):
    command = [sys.executable, '-B', '-u', '-m', 'code.V4.r94_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            command += [flag, str(value)]
    return command


def locked_protocol():
    value = base.read(PROTOCOL_LOCK)
    required = dict(experiment='R94', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', batch_size=1024, logical_global_batch=1024,
        max_updates=57600, warmup_updates=4320, eval_every=1440,
        large_micro_cap=352, eval_batch_size=256,
        donor_variants=dict(A='random', B='pretrained'), dropout=dict(A=.1, B=.1),
        rows_per_problem=dict(train=81000, val=4500, test=4500), wandb_mode='offline')
    if any(value.get(name) != expected for name, expected in required.items()):
        raise ValueError('Use the final R94 paired early-fusion protocol')
    if value.get('data_root') != str(DATA) or value.get('prepared_root') != str(PREPARED):
        raise ValueError('R94 must preserve the frozen NEW90k release and R84 preparation')
    return value


def _sources(protocol):
    paths = set(_sources_previous(protocol))
    paths.update(str(path) for pattern in ('r94*.py', 'test_r94*.py')
                 for path in (REPO / 'code/V4').glob(pattern))
    paths.update((str(PROTOCOL_LOCK), str(WRAPPER)))
    # Native encoder sources are runtime dependencies, not generated labels.
    native = Path('/public/home/shiys/reld-nco-main/Multi-Task')
    paths.update(str(native / path) for path in
        ('models/MOEModel_Light.py', 'models/MOELayer.py', 'models/MTLModel.py'))
    return sorted(paths)


def template(arm, protocol=None):
    if arm not in ARMS:
        raise ValueError('R94 has exactly the two prelocked donor arms')
    feature_directory = '/tmp/shiys_R94_' + arm + '_node_features'
    common = ['{python}', '-B', '-u', '-m', 'code.V4.r94_experiment', 'train',
        '--root', '{common_root}', '--arm', arm, '--prepared-root', '{prepared_root}',
        '--data-root', '{data_root}', '--cache-root', '{cache_root}',
        '--feature-root', feature_directory, '--protocol-lock', str(PROTOCOL_LOCK),
        '--gpu-type', SLOTS[ARM_SLOTS[arm]]['gpu_type'], '--large-micro-cap', '352',
        '--eval-batch-size', '256', '--device', '{device}', '--wandb-mode', 'offline']
    stages = []
    for name, command in (('input_features', 'prepare'), ('train', 'train'), ('TEST', 'test'),
                          ('timing', 'timing')):
        argv = list(common)
        argv[5] = command
        stages.append(dict(name=name, device='gpu', argv=argv))
    stages += [dict(name='LIB', device='gpu', argv=['{python}', '-B', '-u', '-m',
        'code.V4.r94_reporting', 'lib', '--root', '{common_root}', '--arm', arm,
        '--device', '{device}', '--lib-batch-size', '2']),
        dict(name='PDF', device='cpu', argv=['{python}', '-B', '-u', '-m',
        'code.V4.r94_reporting', 'report', '--root', '{common_root}', '--arm', arm,
        '--device', '{device}'])]
    return dict(schema_version=1, run_id=arm, run_dir=str(MODEL_ROOT / arm),
        common_root=str(MODEL_ROOT), data_root=str(DATA), prepared_root=str(PREPARED),
        cache_source=str(CACHE_SOURCE), cache_root=str(CACHE_ROOT), slots=[ARM_SLOTS[arm]],
        cpu_count=4, memory_gib=9, wandb_mode='offline', source_files=_sources(protocol or {}),
        protocol=dict(initialization='scratch', training_splits=['train', 'val'], seed=2,
            logical_batch=1024, task_scope='full18',
            arm_budget=dict(unit='successful_updates', count=57600, warmup_updates=4320,
                            evaluation_every_updates=1440),
            donor_variant='random' if arm == 'A' else 'pretrained', dropout=.1,
            large_instance_micro_cap=352, evaluation_batch=256), stages=stages)


def commands(spec, slot):
    if ARM_SLOTS.get(spec['run_id']) != slot or spec['slots'] != [slot]:
        raise ValueError('No fallback to a healthy predecessor GPU')
    values = dict(spec, python=sys.executable)
    return [dict(stage, argv=[arg.format_map(dict(values,
        device='cpu' if stage['device'] == 'cpu' else 'cuda:0')) for arg in stage['argv']])
        for stage in spec['stages']]


def load_spec(path):
    value = adapter.adapter.engine.saved._load_spec(path)
    if value != template(value['run_id'], locked_protocol()):
        raise ValueError('R94 launch specification changed after locking')
    return value


def runtime_modules():
    return [importlib.import_module('code.V4.' + name)
            for name in ('r94_experiment', 'r94_reporting')]


def comparison_command():
    return [sys.executable, '-B', '-u', '-m', 'code.V4.r94_reporting', 'compare',
            '--root', str(MODEL_ROOT), '--arm', 'A', '--device', 'cpu']


def parse_commands(protocol=None):
    protocol = locked_protocol() if protocol is None else protocol
    runner, reporter = runtime_modules()
    rows = []
    for arm, slot in ARM_SLOTS.items():
        for stage in commands(template(arm, protocol), slot):
            module = reporter if stage['name'] in ('LIB', 'PDF') else runner
            argv = stage['argv'][5:]
            parsed = module.parser().parse_args(argv)
            if (parsed.command != argv[0] or parsed.root != MODEL_ROOT or parsed.arm != arm
                    or parsed.device != ('cpu' if stage['device'] == 'cpu' else 'cuda:0')):
                raise ValueError('R94 stage parser differs from the locked launch arguments')
            if module is runner and (parsed.resume or parsed.large_micro_cap != 352
                    or parsed.eval_batch_size != 256 or parsed.wandb_mode != 'offline'):
                raise ValueError('R94 cannot resume or change its micro/evaluation controls')
            rows.append(dict(arm=arm, slot=slot, command=parsed.command, parsed=True, argv=argv))
    reporter.parser().parse_args(comparison_command()[5:])
    return dict(commands=rows, parsed_count=len(rows) + 1, training_started=False,
                resource_probe_performed=False, source_hashes_frozen=False)


def predecessor_gate(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R94 arm')
    root, directory = PREDECESSOR_ROOT / arm, PREDECESSOR_LAUNCH / arm
    result = dict(experiment='R93', predecessor=arm, complete=False, failed=False)
    try:
        for name, names in (('worker.json', ('train', 'TEST', 'LIB')),
                            ('controller.json', ('allocation_GPU_pipeline', 'PDF'))):
            record = base.read(directory / name)
            if (record.get('state') == 'failed' or record.get('exit_status') not in (None, 0)
                    or any(stage.get('state') == 'failed' or stage.get('exit_status') not in (None, 0)
                           for stage in record.get('stages', []))):
                raise ValueError('Predecessor failed; no implicit repair or restart')
            if record.get('state') != 'complete' or not adapter._stages_complete(record, names):
                raise FileNotFoundError('Waiting for R93' + arm + ' full train/TEST/LIB/PDF delivery')
        lock, complete = base.read(root / 'final_test_lock.json'), base.read(root / 'training_complete.json')
        test, lib = base.read(root / 'test_results.json'), base.read(root / 'lib_stage_complete.json')
        post = base.read(root / 'postprocess_status.json')
        if (lock.get('experiment') != 'R93' or lock.get('arm') != arm
                or lock.get('selected_without_test') is not True
                or lock.get('per_problem_assembly') is not False
                or complete.get('successful_updates') != 57600
                or not lock.get('checkpoint_sha256')
                or test.get('complete') is not True or lib.get('complete') is not True
                or post.get('complete') is not True or post.get('phase') != 'complete'
                or test.get('checkpoint_sha256') != lock['checkpoint_sha256']
                or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']):
            raise ValueError('Predecessor same-lock full delivery is inconsistent')
        result.update(complete=True, reason='R93' + arm + ' complete; actual idle GPU still required')
    except FileNotFoundError as error:
        result['reason'] = str(error)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update(failed=True, reason=str(error))
    return result


def prepare(path, evidence=None, tested=False):
    adapter.require_confirmation(evidence, tested)
    parse_commands()
    result = adapter.adapter.engine._prepare(path)
    result.update(experiment='R94', cache_mode='existing_read_only', automatic_retry=False,
        main_thread_ready=False, main_ready_evidence=evidence,
        predecessor=predecessor_gate(result['spec']['run_id']), resource_probe_performed=False)
    base.save(base.receipts(result['spec']) / 'plan.json', result)
    return result


def reserved_cpus(node, exclude):
    occupied = set()
    for root in (ROOT, PREDECESSOR_LAUNCH, REPO / 'code/V4/runs/R92_launch'):
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
    adapter.require_confirmation(evidence, tested)
    spec = load_spec(path)
    plan = base.read(base.receipts(spec) / 'plan.json')
    if plan.get('spec') != spec or plan.get('main_ready_evidence') != evidence:
        raise RuntimeError('Queue requires the final main-authorized prepared CLI')
    if any(base.sha(name) != digest for name, digest in plan['source_sha256'].items()):
        raise RuntimeError('Prepared source changed before queuing')
    # ready.json is deliberately created only by start(), after the predecessor.
    return spec, plan


def queue(path, evidence, tested, interval=60):
    spec, _ = queue_prepared(path, evidence, tested)
    output = base.receipts(spec)
    with base.lock(output / 'queue.lock'):
        if any((output / name).exists() for name in ('attempt.json', 'controller.json', 'claim.json')):
            raise RuntimeError('R94 arm already attempted; no automatic retry')
        while True:
            gate = predecessor_gate(spec['run_id'])
            base.save(output / 'queue_status.json', dict(experiment='R94', arm=spec['run_id'],
                state='ready' if gate['complete'] else 'failed' if gate['failed'] else 'waiting_predecessor',
                updated_utc=base.utc(), predecessor=gate, training_started=False,
                automatic_retry=False))
            if gate['failed']:
                raise RuntimeError(gate['reason'])
            if gate['complete']:
                return adapter.start(path, evidence, tested)
            time.sleep(interval)


# Rebind only this private adapter, including the lower-level transport hooks.
for name, value in dict(ROOT=ROOT, MODEL_ROOT=MODEL_ROOT, PROTOCOL_LOCK=PROTOCOL_LOCK,
        WRAPPER=WRAPPER, TOKEN_ENV=TOKEN_ENV, ARMS=ARMS, ARM_SLOTS=ARM_SLOTS,
        SLOTS=SLOTS, self_command=self_command, locked_protocol=locked_protocol,
        template=template, _sources=_sources, commands=commands, load_spec=load_spec,
        runtime_modules=runtime_modules, comparison_command=comparison_command,
        parse_commands=parse_commands, predecessor_gate=predecessor_gate,
        reserved_cpus=reserved_cpus, prepare=prepare).items():
    setattr(adapter, name, value)
base.ROOT, base.SLOTS, base.TOKEN_ENV = ROOT, SLOTS, TOKEN_ENV
base.self_command, base.load_spec, base.commands = self_command, load_spec, commands
base.reserved_cpus = reserved_cpus
base.session_name = lambda value: 'r94_' + value['run_id']


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == 'queue':
        parser = argparse.ArgumentParser(description='Wait quietly for the fixed predecessor, then start once')
        parser.add_argument('stage', choices=('queue',))
        parser.add_argument('--spec', type=Path, required=True)
        parser.add_argument('--main-ready-evidence', required=True)
        parser.add_argument('--runner-tests-passed', action='store_true')
        parser.add_argument('--interval', type=int, default=60)
        args = parser.parse_args(values)
        if args.interval < 30:
            parser.error('Use a quiet interval of at least30 seconds')
        return queue(args.spec, args.main_ready_evidence, args.runner_tests_passed, args.interval)
    return _parse_main(values)


if __name__ == '__main__':
    raise SystemExit(main())
