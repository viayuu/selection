"""Shared one-shot dispatch for the post-R97 research wave.

Only existing allocation-owned containers are used. Each supplied scientific
protocol, exact argv and source set is frozen before its independent launch.
The transport is a private copy: historical launchers are never rebound.
"""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R98_parallel_dispatch'
WRAPPER = REPO / 'code/V4/run_v4_r98_launch.sh'
TOKEN_ENV = 'R98_DISPATCH_TOKEN'


def private(name, filename):
    spec = importlib.util.spec_from_file_location('code.V4.' + name,
                                                  REPO / 'code/V4' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = private('_r98_transport_resources', 'r84_launch.py')
base.transport = private('_r98_gpu_mutexes', 'r81_queue.py')
base.ROOT = ROOT
base.TOKEN_ENV = TOKEN_ENV
SLOTS = base.SLOTS
original_environment = base.environment
original_load_spec = base.load_spec
original_prepare = base.prepare
original_check_ready = base.check_ready
original_stage_cache = base.stage_cache


def self_command(stage, spec=None, slot=None, token=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r98_launch', stage]
    for key, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            argv.extend((key, str(value)))
    return argv


def environment(slot=None, token=None):
    result = original_environment(slot, token)
    # No credentials are searched; this wave retains the explicit offline mode.
    result['WANDB_MODE'] = 'offline'
    return result


def container_command(stage, slot, spec=None, token=None):
    item = SLOTS[slot]
    argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid())]
    env = environment(slot, token)
    keys = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'PYTHONUNBUFFERED', 'MPLBACKEND',
            'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_PROJECT', 'WANDB_ENTITY',
            'WANDB_MODE', 'SLURM_JOB_ID')
    for key in (*keys, *((TOKEN_ENV,) if token else ())):
        argv.extend(('-e', key + '=' + env[key]))
    identity = base.transport.CONTAINERS[item['node']][1]
    return argv + [identity, 'bash', str(WRAPPER),
                   *self_command(stage, spec, slot, token)[5:]]


def load_spec(path):
    spec = original_load_spec(path)
    if (not spec['run_id'].startswith(('R98_', 'R99_', 'R100_'))
            or len(spec['slots']) != 1 or spec.get('wandb_mode') != 'offline'):
        raise ValueError('Use one fixed GPU and the explicit offline post-R97 wave')
    protocol = Path(spec['science_protocol'])
    if not protocol.is_absolute() or not protocol.is_file():
        raise ValueError('A predeclared scientific protocol is required')
    science = base.read(protocol)
    if (science.get('locked') is not True or science.get('seed') != 2
            or science.get('task_scope') != 'full18'
            or science.get('data_root') != spec['data_root']
            or science.get('current_query_labels_in_forward', False) is not False):
        raise ValueError('Scientific protocol must retain full18 and query-label isolation')
    if spec.get('cache_kind') not in ('r84_inputs', 'r97_context'):
        raise ValueError('Only immutable existing input/context caches are supported')
    return spec


def source_closure(paths):
    paths = {Path(path).resolve() for path in paths}
    pending, visited = list(paths), set()
    while pending:
        path = pending.pop()
        if path in visited or path.suffix != '.py' or not path.is_file():
            continue
        visited.add(path)
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
    return sorted(str(path) for path in paths)


def parse_commands(spec):
    result = []
    for stage in base.commands(spec, spec['slots'][0]):
        argv = stage['argv']
        if '-m' not in argv:
            raise ValueError('Runner stages must expose a Python module parser')
        index = argv.index('-m')
        module, arguments = argv[index + 1], argv[index + 2:]
        program = ('import importlib,json,sys; m=importlib.import_module(sys.argv[1]); '
                   'print(json.dumps(vars(m.parser().parse_args(json.loads(sys.argv[2]))),default=str))')
        process = base.subprocess.run([sys.executable, '-B', '-c', program,
            module, json.dumps(arguments)], cwd=REPO, env=environment(),
            text=True, capture_output=True, timeout=90)
        if process.returncode:
            raise RuntimeError('Exact stage CLI is not ready: ' + process.stderr[-3000:])
        parsed = json.loads(process.stdout)
        if parsed.get('resume'):
            raise ValueError('Fresh experiments must not implicitly resume')
        result.append(dict(stage=stage['name'], module=module, argv=arguments, parsed=parsed))
    return result


def prepare(path, evidence, tested):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True:
        raise ValueError('Main-thread CPU-test evidence is required')
    spec = load_spec(path)
    output = base.receipts(spec)
    if (output / 'ready.json').exists():
        raise RuntimeError('Already armed; no replacement or automatic retry')
    parsed = parse_commands(spec)
    sources = source_closure([__file__, WRAPPER, spec['science_protocol'],
                              *spec.get('source_files', [])])
    record = original_prepare(path)
    record['source_sha256'].update({name: base.sha(name) for name in sources})
    record.update(main_thread_ready=True, evidence=evidence, exact_cli=parsed,
                  automatic_retry=False, cache_preprocessing=False)
    base.save(output / 'plan.json', record)
    base.save(output / 'ready.json', dict(main_thread_ready=True, evidence=evidence,
        runner_tests_passed=True, confirmed_utc=base.utc(),
        plan_sha256=base.sha(output / 'plan.json')), exclusive=True)
    original_check_ready(path)
    return dict(receipts=str(output), commands=len(parsed), gpu_started=False)


def stage_cache(spec, manifest):
    if spec['cache_kind'] == 'r84_inputs':
        return original_stage_cache(spec, manifest)
    # This helper stages frozen files only; it does not start the old R97 runner.
    from .r97_launch import stage_cache as context_stage
    return context_stage(spec, manifest)


def reserved_cpus(node, exclude):
    occupied = set()
    for root in (ROOT, *(REPO / f'code/V4/runs/R{version}_launch'
                         for version in ('92', '93', '94', '95', '97'))):
        for path in root.glob('*/claim.json'):
            if path.parent == exclude:
                continue
            claim = base.read(path)
            controller_path = path.with_name('controller.json')
            state = base.read(controller_path).get('state') if controller_path.exists() else None
            if claim.get('node') == node and state not in ('complete', 'failed'):
                occupied.update(claim['cpus'])
    return occupied


def start(path, evidence, tested):
    if not evidence or tested is not True:
        raise ValueError('Explicit main-ready evidence is required')
    spec, plan = original_check_ready(path)
    output = base.receipts(spec)
    if evidence != plan.get('evidence'):
        raise ValueError('Start evidence must match the armed launch plan')
    if base.socket.gethostname().split('.')[0] != 'admin01' or not base.FONT.is_file():
        raise RuntimeError('Controller and PDF stages must remain on admin01')
    with base.lock(output / 'start.lock'):
        if any((output / name).exists() for name in ('start.json', 'claim.json', 'controller.json', 'tmux.json')):
            raise RuntimeError('A one-shot start was already attempted')
        run = Path(spec['run_dir'])
        if any((run / name).exists() for name in ('status.json', 'attempt.json', 'checkpoint.json',
                'last.pt', 'best.pt', 'final_test_lock.json', 'run_complete.json', 'evaluation')):
            raise RuntimeError('Preserve existing model outputs; no implicit retry/resume')
        base.save(output / 'start.json', dict(started_utc=base.utc(), automatic_retry=False,
                                              main_ready_evidence=evidence), exclusive=True)
        session = 'goal_' + spec['run_id'].lower()
        command = shlex.join(['bash', '-lc', base.activated(self_command('controller', path))])
        base.subprocess.run(['tmux', 'new-session', '-d', '-s', session, command], check=True)
        base.subprocess.run(['tmux', 'set-option', '-t', session, 'remain-on-exit', 'on'], check=True)
        base.save(output / 'tmux.json', dict(session=session, command=command), exclusive=True)
    return dict(session=session, receipts=str(output), automatic_retry=False)


base.self_command = self_command
base.container_command = container_command
base.environment = environment
base.load_spec = load_spec
base.stage_cache = stage_cache
base.reserved_cpus = reserved_cpus


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('stage', choices=('prepare', 'start', 'controller', 'node-resources',
                                         'container-resources', 'entry', 'worker'))
    result.add_argument('--spec', type=Path)
    result.add_argument('--slot', choices=tuple(SLOTS))
    result.add_argument('--token')
    result.add_argument('--evidence')
    result.add_argument('--runner-tests-passed', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == 'prepare':
        value = prepare(args.spec, args.evidence, args.runner_tests_passed)
    elif args.stage == 'start':
        value = start(args.spec, args.evidence, args.runner_tests_passed)
    elif args.stage == 'controller':
        return base.controller(args.spec)
    elif args.stage == 'node-resources':
        value = base.node_resources(args.slot)
    elif args.stage == 'container-resources':
        value = base.container_resources(args.slot)
    elif args.stage == 'entry':
        return base.entry(args.spec, args.slot, args.token)
    else:
        return base.worker(args.spec, args.slot, args.token)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
