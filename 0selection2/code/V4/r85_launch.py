"""R85 launch adapter: existing allocations, read-only R84 cache, explicit ready."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R85_launch'
R84_ROOT = REPO / 'code/V4/runs/R84_new90k_launch'
PREPARED = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
CACHE_SOURCE = PREPARED.parent / 'input_cache'
CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
WRAPPER = REPO / 'code/V4/run_v4_r85_launch.sh'
BASE_SOURCE = REPO / 'code/V4/r84_launch.py'
TOKEN_ENV = 'R85_LAUNCH_TOKEN'
ARM_SLOTS = dict(A_ovr_control='3090_0', B_gradient_balance='3090_1',
                 C_cost_profile='4090_0', D_candidate_subset='4090_1')

# A private module namespace keeps the historical launcher and its receipts intact.
_spec = importlib.util.spec_from_file_location('code.V4._r85_resource_base', BASE_SOURCE)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
SLOTS = base.SLOTS
_load_spec = base.load_spec
_prepare = base.prepare
_node_resources = base.node_resources
_container_command = base.container_command


def self_command(stage, spec=None, slot=None, token=None):
    command = [sys.executable, '-B', '-u', '-m', 'code.V4.r85_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            command += [flag, str(value)]
    return command


def container_command(stage, slot, spec=None, token=None):
    command = _container_command(stage, slot, spec, token)
    command[command.index(str(REPO / 'code/V4/run_v4_r84_launch.sh'))] = str(WRAPPER)
    return command


def load_spec(path):
    spec = _load_spec(path)
    for key, expected in dict(data_root=DATA, prepared_root=PREPARED,
                              cache_source=CACHE_SOURCE, cache_root=CACHE_ROOT).items():
        if Path(spec[key]) != expected:
            raise ValueError(f'R85 reuses the existing NEW90k {key}: {expected}')
    if (spec.get('protocol', {}).get('initialization') != 'scratch'
            or spec['protocol'].get('training_splits') != ['train', 'val']):
        raise ValueError('Main must declare fresh weights and TRAIN/VAL-only training')
    if Path(spec['run_dir']) == PREPARED.parent or PREPARED.parent in Path(spec['run_dir']).parents:
        raise ValueError('R85 model output must not overwrite historical R84 outputs')
    return spec


def cache_manifest(source):
    if Path(source) != CACHE_SOURCE:
        raise ValueError('Reuse the completed R84 cache source')
    prior = base.read(R84_ROOT / 'A_baseline/plan.json')['cache']
    files = [row for row in prior['files']
             if Path(row['path']).parts[0] in ('train', 'val')
             and Path(row['path']).suffix in ('.json', '.npy')]
    files.sort(key=lambda row: row['path'])
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    snapshot = base.read(ROOT / 'resources.json')
    for node in ('gpu03', 'gpu05'):
        checked = snapshot['nodes'][node]['reuse_check']['cache']
        if (identity != checked['identity_sha256'] or len(files) != checked['files']
                or sum(row['bytes'] for row in files) != checked['bytes']):
            raise RuntimeError('Prior completed TRAIN/VAL cache evidence does not match')
    return dict(source=str(source), files=files, bytes=sum(row['bytes'] for row in files),
                identity_sha256=identity, splits=['train', 'val'], array_contents_read=False,
                validation_snapshot=str(ROOT / 'resources.json'),
                validation_snapshot_utc=snapshot['captured_utc'])


def stage_cache(spec, manifest):
    """Reuse completed validation; only check the existing cache identity stamp."""
    destination = Path(spec['cache_root'])
    stamp = base.read(destination / '.r84_cache_identity.json')
    if (Path(stamp.get('data_root', '')) != Path(spec['data_root'])
            or Path(stamp.get('source', '')) != Path(spec['cache_source'])):
        raise RuntimeError('Existing allocation cache does not identify the NEW90k release')
    for row in manifest['files']:
        if Path(row['path']).parts[0] not in ('train', 'val'):
            raise RuntimeError('R85 training cache verification excludes TEST')
    if any(not (destination / split).is_dir() for split in ('train', 'val')):
        raise RuntimeError('Existing cache is missing; no automatic rebuild')
    return dict(root=str(destination), bytes=manifest['bytes'], files=len(manifest['files']),
                identity_sha256=manifest['identity_sha256'], splits=['train', 'val'],
                mode='existing_read_only', copied_bytes=0, preprocessing=False,
                array_contents_read=False, test_cache_ignored=True,
                cache_audit_repeated=False, validation_snapshot=manifest.get('validation_snapshot'))


def container_reuse_check(slot):
    base.assert_container(slot)
    manifest = cache_manifest(CACHE_SOURCE)
    cache = stage_cache(dict(cache_root=str(CACHE_ROOT), data_root=str(DATA),
                            cache_source=str(CACHE_SOURCE)), manifest)
    workers = {}
    for directory in sorted(R84_ROOT.glob('*')):
        receipt = directory / 'worker.json'
        claim = directory / 'claim.json'
        if not receipt.is_file() or not claim.is_file():
            continue
        if base.read(claim).get('node') != SLOTS[slot]['node']:
            continue
        record = base.read(receipt)
        stages = []
        for stage in record.get('stages', []):
            pid = stage.get('pid')
            command_path = Path(f'/proc/{pid}/cmdline')
            try:
                command = command_path.read_bytes().decode(errors='replace').replace('\0', ' ').strip()
            except (FileNotFoundError, ProcessLookupError):
                command = ''
            same_worker = bool(command and ('code.V4.r84_' in command or '/r84_' in command))
            stages.append(dict(name=stage['name'], recorded_pid=pid,
                               recorded_state=stage.get('state'), exit_status=stage.get('exit_status'),
                               same_worker_alive=same_worker, pid_exists=bool(command),
                               current_command=command or None))
        workers[directory.name] = dict(recorded_state=record.get('state'),
            recorded_exit_status=record.get('exit_status'), stages=stages,
            terminal_confirmed=record.get('state') in ('complete', 'failed')
                and all(not row['same_worker_alive'] for row in stages))
    return dict(captured_utc=base.utc(), cache=cache, R84_workers=workers)


def node_resources(slot):
    # Claim-time probes inspect current allocation/process/GPU state only.
    return _node_resources(slot)


def resources():
    result = dict(captured_utc=base.utc(), nodes={}, training_started=False,
                  existing_processes_stopped=False, no_GPU_context_created=True)
    for slot in ('3090_0', '4090_0'):
        result['nodes'][SLOTS[slot]['node']] = json.loads(base.transport.capture(
            base.srun_command('node-resources', slot), timeout=55))
    base.save(ROOT / 'resources_live.json', result)
    return result


def prepare(path):
    result = _prepare(path)
    for source in (Path(__file__).resolve(), WRAPPER, ROOT / 'resources.json'):
        result['source_sha256'][str(source)] = base.sha(source)
    result.update(cache_mode='existing_read_only', training_splits=['train', 'val'],
                  main_thread_ready=False)
    base.save(base.receipts(result['spec']) / 'plan.json', result)
    return result


def template(run_id='MAIN_SUPPLIES_RUN_ID'):
    return dict(schema_version=1, run_id=run_id,
        run_dir=str(REPO / 'code/V4/runs/R85_zero_selection_research' / run_id), data_root=str(DATA),
        prepared_root=str(PREPARED), cache_source=str(CACHE_SOURCE), cache_root=str(CACHE_ROOT),
        slots=list(SLOTS), cpu_count=4, memory_gib=9, source_files=[],
        protocol=dict(initialization='scratch', training_splits=['train', 'val'],
                      seed=2, logical_batch=1024, max_successful_updates=57600,
                      main_protocol='MAIN_SUPPLIES_FINAL_PROTOCOL'),
        stages=[dict(name='train_VALlock_TEST', device='gpu', argv=['{python}', '-B', '-u',
            '-m', 'MAIN_SUPPLIES_RUNNER', 'train', '--data-root', '{data_root}',
            '--prepared-root', '{prepared_root}', '--cache-root', '{cache_root}',
            '--run-dir', '{run_dir}', '--device', '{device}']),
            dict(name='LIB', device='gpu', argv=['{python}', '-B', '-u', '-m', 'MAIN_SUPPLIES_LIB_RUNNER']),
            dict(name='PDF', device='cpu', argv=['{python}', '-B', '-u', '-m', 'MAIN_SUPPLIES_PDF_RUNNER'])])


def final_spec(arm):
    if arm not in ARM_SLOTS:
        raise ValueError('Use one of the four main-locked R85 arms')
    value = template(arm)
    value.update(common_root=str(REPO / 'code/V4/runs/R85_zero_selection_research'),
                 slots=[ARM_SLOTS[arm]])
    value['protocol'].update(main_protocol='main_locked_four_R85_arms',
        warmup_updates=4320, evaluation_every_updates=1440,
        training_includes_locked_TEST=True,
        large_instance_micro_cap=256 if arm == 'D_candidate_subset' else SLOTS[ARM_SLOTS[arm]]['cap'])
    value['source_files'] = [str(REPO / 'code/V4' / name) for name in (
        'r85_experiment.py', 'r85_objectives.py', 'r85_candidate_selector.py', 'r85_reporting.py',
        'r84_experiment.py', 'r84_prepare.py', 'r84_reporting.py', 'r84_nss_reference.py',
        'r76_experiment.py', 'r77_experiment.py', 'r80_experiment.py', 'r80_reporting.py',
        'pairwise_selector.py', 'pairwise_objective.py', 'solver_code_encoder.py')]
    value['source_files'] += [str(PREPARED / name) for name in (
        'prepared.json', 'geometry_stats.json', 'common_full_train_sbs.json', 'identicalinit_seed2.pt')]
    value['source_files'].append(value['common_root'] + '/protocol_lock.json')
    value['stages'] = [dict(name='train_VALlock_TEST', device='gpu', argv=[
        '{python}', '-B', '-u', '-m', 'code.V4.r85_experiment', 'train',
        '--root', '{common_root}', '--arm', arm, '--prepared-root', '{prepared_root}',
        '--data-root', '{data_root}', '--cache-root', '{cache_root}',
        '--protocol-lock', '{common_root}/protocol_lock.json', '--device', '{device}',
        '--gpu-type', '{gpu_type}', '--large-micro-cap',
        '256' if arm == 'D_candidate_subset' else '{micro_cap}',
        '--eval-batch-size', '256', '--wandb-mode', 'offline']),
        dict(name='LIB', device='gpu', argv=['{python}', '-B', '-u', '-m',
            'code.V4.r85_reporting', 'lib', '--root', '{common_root}', '--arm', arm,
            '--device', '{device}', '--lib-batch-size', '2']),
        dict(name='PDF', device='cpu', argv=['{python}', '-B', '-u', '-m',
            'code.V4.r85_reporting', 'report', '--root', '{common_root}', '--arm', arm,
            '--device', '{device}', '--lib-batch-size', '2'])]
    return value


def write_specs():
    runner = importlib.import_module('code.V4.r85_experiment')
    reporter = (importlib.import_module('code.V4.r85_reporting')
                if (REPO / 'code/V4/r85_reporting.py').is_file() else None)
    if tuple(runner.ARMS) != tuple(ARM_SLOTS):
        raise RuntimeError('Runner arm names differ from main lock')
    parsed = []
    report_parsed = []
    protocol = None
    for arm, slot in ARM_SLOTS.items():
        spec = final_spec(arm)
        path = ROOT / 'specs' / (arm + '.json')
        if path.is_file() and base.read(path) != spec:
            raise RuntimeError('Existing spec differs; explicit main update required')
        command = base.commands(spec, slot)[0]['argv']
        for mode in ('train', 'test'):
            argv = [mode, *command[6:]]
            args = runner.parser().parse_args(argv)
            if (args.arm != arm or args.root != Path(spec['common_root']) or args.resume
                    or args.eval_batch_size != 256 or args.cache_root != CACHE_ROOT
                    or args.gpu_type != SLOTS[slot]['gpu_type']
                    or args.large_micro_cap != spec['protocol']['large_instance_micro_cap']):
                raise RuntimeError('Parsed runner command does not match the fixed arm')
            if protocol is None:
                protocol = runner.protocol_lock(args)
            parsed.append(dict(arm=arm, command=mode, argv=argv, parsed=True,
                               slot=slot, gpu_type=args.gpu_type, micro_cap=args.large_micro_cap))
        if reporter is not None:
            for stage in base.commands(spec, slot)[1:]:
                argv = stage['argv'][5:]
                args = reporter.parser().parse_args(argv)
                expected_device = 'cuda:0' if stage['device'] == 'gpu' else 'cpu'
                if (args.arm != arm or args.root != Path(spec['common_root'])
                        or args.device != expected_device or args.lib_batch_size != 2):
                    raise RuntimeError('Parsed reporter command does not match the fixed arm')
                report_parsed.append(dict(arm=arm, command=args.command, argv=argv, parsed=True))
        if not path.is_file():
            base.save(path, spec, exclusive=True)
    result = dict(created_utc=base.utc(), state='specs_parsed_waiting_runner_ready',
        train_test_argv=parsed, train_test_parsed_count=len(parsed),
        protocol_lock=protocol, slot_mapping=ARM_SLOTS,
        LIB_PDF_parser_confirmed=reporter is not None,
        LIB_PDF_argv=report_parsed, LIB_PDF_parsed_count=len(report_parsed),
        main_thread_ready=False, training_started=False,
        source_hashes_frozen=False, cache_audit_repeated=False,
        specs=[str(ROOT / 'specs' / (arm + '.json')) for arm in ARM_SLOTS])
    base.save(ROOT / 'spec_parse.json', result)
    return result


base.ROOT = ROOT
base.TOKEN_ENV = TOKEN_ENV
base.self_command = self_command
base.container_command = container_command
base.load_spec = load_spec
base.cache_manifest = cache_manifest
base.stage_cache = stage_cache
base.node_resources = node_resources
base.session_name = lambda spec: 'r85_' + spec['run_id']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('resources', 'template', 'specs', 'prepare', 'start', 'controller',
        'node-resources', 'container-resources', 'container-reuse-check', 'entry', 'worker'))
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--slot', choices=tuple(SLOTS))
    parser.add_argument('--run-id', default='MAIN_SUPPLIES_RUN_ID')
    parser.add_argument('--token')
    parser.add_argument('--main-ready-evidence')
    parser.add_argument('--runner-tests-passed', action='store_true')
    args = parser.parse_args(argv)
    if args.stage in ('prepare', 'start', 'controller', 'entry', 'worker') and not args.spec:
        parser.error('--spec from main is required')
    if args.stage in ('node-resources', 'container-resources', 'container-reuse-check', 'entry', 'worker') and not args.slot:
        parser.error('--slot is required')
    if args.stage == 'resources':
        value = resources()
    elif args.stage == 'template':
        value = template(args.run_id)
    elif args.stage == 'specs':
        value = write_specs()
    elif args.stage == 'prepare':
        value = prepare(args.spec)
    elif args.stage == 'start':
        value = base.start(args.spec, args.main_ready_evidence, args.runner_tests_passed)
    elif args.stage == 'controller':
        return base.controller(args.spec)
    elif args.stage == 'node-resources':
        value = node_resources(args.slot)
    elif args.stage == 'container-resources':
        value = base.container_resources(args.slot)
    elif args.stage == 'container-reuse-check':
        value = container_reuse_check(args.slot)
    elif args.stage == 'entry':
        return base.entry(args.spec, args.slot, args.token)
    else:
        return base.worker(args.spec, args.slot, args.token)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
