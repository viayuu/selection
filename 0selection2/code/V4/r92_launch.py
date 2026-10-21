"""Shared R92 plumbing; scientific controls come only from main's final lock."""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R92_launch'
MODEL_ROOT = REPO / 'code/V4/runs/R92_broad_learning'
PROTOCOL_LOCK = MODEL_ROOT / 'protocol_lock.json'
WRAPPER = REPO / 'code/V4/run_v4_r92_launch.sh'
TOKEN_ENV = 'R92_LAUNCH_TOKEN'
ARMS = ('A', 'B', 'C', 'D')
ARM_SLOTS = dict(zip(ARMS, ('3090_0', '4090_0', '3090_1', '4090_1')))
PREDECESSORS = dict(A='A_utility_ce', B='A_adamw',
                    C='B_utility_probabilities', D='B_muon_hybrid')
PREVIOUS_LAUNCH = {name: REPO / f'code/V4/runs/{name}_launch' for name in ('R91', 'R89')}
PREVIOUS_MODELS = dict(R91=REPO / 'code/V4/runs/R91_probability_link_control',
                       R89=REPO / 'code/V4/runs/R89_optimizer_geometry')
R91_RECOVERY_EVIDENCE = PREVIOUS_MODELS['R91'] / 'report_only_recovery_complete.json'


def _private(name, path):
    spec = importlib.util.spec_from_file_location('code.V4._r92_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# All monkeypatches below target private namespaces, never frozen live helpers.
engine = _private('helpers', REPO / 'code/V4/r87_launch.py')
base = engine.base
transport = _private('transport', Path(base.transport.__file__))
base.transport = transport
DATA, PREPARED = engine.DATA, engine.PREPARED
CACHE_SOURCE, CACHE_ROOT = engine.CACHE_SOURCE, engine.CACHE_ROOT
CACHE_EVIDENCE = engine.saved.ROOT / 'resources.json'
SLOTS = {slot: {key: value for key, value in engine.saved.SLOTS[slot].items()
                if key != 'cap'} for slot in ARM_SLOTS.values()}
_environment, _capture = base.environment, transport.capture
_check_ready, _start = base.check_ready, base.start
_controller, _worker, _entry = engine._controller, engine._worker, engine._entry


def self_command(stage, spec=None, slot=None, token=None):
    command = [sys.executable, '-B', '-u', '-m', 'code.V4.r92_launch', stage]
    for flag, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            command += [flag, str(value)]
    return command


def environment(slot=None, token=None):
    return dict(_environment(slot, token), WANDB_MODE='offline')


def capture(command, timeout=25):
    # capture() already separates stderr and rejects nonzero rc. Strip only the
    # platform's known stdout prefix, not arbitrary warnings or JSON-like stderr.
    raw = _capture(command, timeout=timeout)
    return re.sub(r'\ADistribute Task Container\.\r?\ntask_id:\d+\r?\n', '', raw)


def _alias(value, names, default=None):
    found = [value[name] for name in names if name in value]
    if found and any(item != found[0] for item in found[1:]):
        raise ValueError('Conflicting protocol aliases: ' + ', '.join(names))
    return found[0] if found else default


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError('Main must lock a positive integer ' + name)
    return value


def _arm_setting(protocol, arm, names, default=None):
    launch = protocol.get('launch', {})
    groups = (launch.get('arms', {}), protocol.get('arm_descriptions', {}),
              protocol.get('arms', {}))
    scopes = [group[arm] for group in groups if isinstance(group, dict)
              and isinstance(group.get(arm), dict)]
    if isinstance(protocol.get(arm), dict):
        scopes.append(protocol[arm])
    for scope in (*scopes, launch, protocol):
        value = _alias(scope, names)
        if value is not None:
            return value
    return default


def arm_budget(protocol, arm):
    """Translate main's per-arm lock without equating NN updates and trees."""
    if arm not in ARMS:
        raise ValueError('Choose stable generic arm A/B/C/D')
    budgets = protocol.get('arm_budgets', protocol.get('arm_budget', {}))
    budget = budgets.get(arm, {})
    if type(budget) is int:
        budget = dict(max_successful_updates=budget) if arm in ('A', 'B') else dict(trees_per_task=budget)
    if not isinstance(budget, dict):
        raise ValueError('Main must supply a per-arm budget description')
    if arm in ('A', 'B'):
        count = _alias(budget, ('max_updates', 'max_successful_updates'),
            _arm_setting(protocol, arm, ('max_updates', 'max_successful_updates')))
        warmup = budget.get('warmup_updates', _arm_setting(protocol, arm, ('warmup_updates',)))
        cadence = _alias(budget, ('eval_every', 'evaluation_every_updates'),
            _arm_setting(protocol, arm, ('eval_every', 'evaluation_every_updates')))
        count = _positive(count, 'NN update budget for ' + arm)
        warmup, cadence = _positive(warmup, 'warmup_updates'), _positive(cadence, 'eval_every')
        if warmup >= count or cadence > count:
            raise ValueError('Main-locked warmup/evaluation cadence exceeds the NN update budget')
        return dict(unit='successful_updates', count=count, warmup_updates=warmup,
                    evaluation_every_updates=cadence)
    count = _alias(budget, ('trees_per_task', 'iterations', 'max_trees', 'tree_iterations',
                           'max_successful_updates'),
        _arm_setting(protocol, arm, ('trees_per_task', 'iterations', 'max_trees', 'tree_iterations')))
    return dict(unit='trees_per_task', count=_positive(count, 'tree budget for ' + arm))


def locked_protocol():
    if not PROTOCOL_LOCK.is_file():
        raise RuntimeError('Waiting for main final protocol_lock.json; no scientific defaults')
    value = base.read(PROTOCOL_LOCK)
    if value.get('experiment') != 'R92' or value.get('locked') is not True:
        raise ValueError('Main must explicitly lock the R92 protocol')
    if tuple(value.get('arms', ())) != ARMS:
        raise ValueError('R92 uses stable generic arms A/B/C/D in order')
    if value.get('seed') != 2 or _alias(value, (
            'logical_batch', 'logical_global_batch', 'global_batch_size', 'batch_size')) != 1024:
        raise ValueError('R92 requires logical batch1024 and seed2')
    if (value.get('task_scope') != 'full18'
            and not str(value.get('tasks', '')).startswith(('all18', 'full18'))):
        raise ValueError('Main must lock original full18 scope')
    if (value.get('data_root') != str(DATA) or value.get('prepared_root') != str(PREPARED)
            or value.get('rows_per_problem') != dict(train=81000, val=4500, test=4500)):
        raise ValueError('Use frozen original full18 NEW90k and R84 preparation')
    for arm in ARMS:
        arm_budget(value, arm)
        cap = _positive(_arm_setting(value, arm,
            ('large_micro_cap', 'large_instance_micro_cap'), 352), 'large_micro_cap for ' + arm)
        evaluation = _arm_setting(value, arm, ('eval_batch_size', 'evaluation_batch'), 256)
        if cap > 1024 or type(evaluation) is not int or evaluation not in (128, 256):
            raise ValueError('Micro-cap must fit logical1024; eval256 or explicitly locked128')
    return value


def _sources(protocol):
    names = ('r92_experiment.py', 'r92_reporting.py', 'r92_tree.py', 'r92_tree_features.py', 'r88_model.py',
        'r88_experiment.py', 'r88_reporting.py', 'r87_model.py', 'r87_experiment.py',
        'r87_reporting.py', 'r85_candidate_selector.py', 'r85_objectives.py',
        'r85_experiment.py', 'r85_reporting.py', 'r84_experiment.py', 'r84_reporting.py',
        'r84_prepare.py', 'r84_nss_reference.py', 'r76_experiment.py',
        'r77_experiment.py', 'r80_experiment.py', 'r80_reporting.py',
        'r77_best_model_report.py', 'r42_experiment.py', 'pairwise_selector.py',
        'pairwise_objective.py', 'solver_code_encoder.py')
    paths = [REPO / 'code/V4' / name for name in names]
    paths += sorted((REPO / 'code/V4').glob('r92_*.py'))
    for name in protocol.get('launch', {}).get('source_files', []):
        path = Path(name)
        path = path if path.is_absolute() else REPO / path
        if not path.resolve().is_relative_to(REPO / 'code/V4') or path.suffix != '.py':
            raise ValueError('Additional runtime sources must be V4 Python modules')
        paths.append(path)
    paths += [PREPARED / name for name in ('prepared.json', 'geometry_stats.json',
        'common_full_train_sbs.json', 'identicalinit_seed2.pt')]
    paths.append(PROTOCOL_LOCK)
    return list(dict.fromkeys(str(path) for path in paths))


def template(arm, protocol=None):
    if arm not in ARMS:
        raise ValueError('Choose stable generic arm A/B/C/D')
    protocol = {} if protocol is None else protocol
    cap = _arm_setting(protocol, arm, ('large_micro_cap', 'large_instance_micro_cap'),
                       352)
    evaluation = _arm_setting(protocol, arm, ('eval_batch_size', 'evaluation_batch'), 256)
    budget = arm_budget(protocol, arm) if protocol else dict(
        unit='successful_updates' if arm in ('A', 'B') else 'trees_per_task',
        count='MAIN_SUPPLIES_ARM_BUDGET')
    train = dict(name='train', device='gpu', argv=[
        '{python}', '-B', '-u', '-m', 'code.V4.r92_experiment', 'train',
        '--root', '{common_root}', '--arm', arm, '--prepared-root', '{prepared_root}',
        '--data-root', '{data_root}', '--cache-root', '{cache_root}',
        '--protocol-lock', str(PROTOCOL_LOCK), '--gpu-type', SLOTS[ARM_SLOTS[arm]]['gpu_type'],
        '--large-micro-cap', str(cap), '--eval-batch-size', str(evaluation),
        '--device', '{device}', '--wandb-mode', 'offline'])
    test = dict(name='TEST', device='gpu', argv=list(train['argv']))
    test['argv'][5] = 'test'
    return dict(schema_version=1, run_id=arm, run_dir=str(MODEL_ROOT / arm),
        common_root=str(MODEL_ROOT), data_root=str(DATA), prepared_root=str(PREPARED),
        cache_source=str(CACHE_SOURCE), cache_root=str(CACHE_ROOT), slots=[ARM_SLOTS[arm]],
        cpu_count=4, memory_gib=9, wandb_mode='offline', source_files=_sources(protocol),
        protocol=dict(initialization='scratch', training_splits=['train', 'val'],
            seed=2, logical_batch=1024, task_scope='full18', arm_budget=budget,
            large_instance_micro_cap=cap, evaluation_batch=evaluation),
        stages=[train, test,
            dict(name='LIB', device='gpu', argv=['{python}', '-B', '-u', '-m',
                'code.V4.r92_reporting', 'lib', '--root', '{common_root}', '--arm', arm,
                '--device', '{device}', '--lib-batch-size', '2']),
            dict(name='PDF', device='cpu', argv=['{python}', '-B', '-u', '-m',
                'code.V4.r92_reporting', 'report', '--root', '{common_root}', '--arm', arm,
                '--device', '{device}'])])


def commands(spec, slot):
    if ARM_SLOTS.get(spec['run_id']) != slot or spec['slots'] != [slot]:
        raise ValueError('R92 arms have fixed slots; no fallback GPU')
    # No slot-level cap: the exact main-locked value is already in each argv.
    values = dict(spec, python=sys.executable, device='cuda:0')
    return [dict(stage, argv=[arg.format_map(dict(values,
        device='cpu' if stage['device'] == 'cpu' else 'cuda:0')) for arg in stage['argv']])
        for stage in spec['stages']]


def load_spec(path):
    value = engine.saved._load_spec(path)
    if value != template(value['run_id'], locked_protocol()):
        raise ValueError('R92 spec differs from the final main lock, fixed routing or CPU PDF policy')
    return value


def runtime_modules():
    for name in ('r92_experiment.py', 'r92_reporting.py'):
        if not (REPO / 'code/V4' / name).is_file():
            raise RuntimeError('Waiting for final runtime: ' + name)
    modules = [importlib.import_module('code.V4.' + name)
               for name in ('r92_experiment', 'r92_reporting')]
    if any(tuple(module.ARMS) != ARMS for module in modules):
        raise RuntimeError('Final runner/reporter ARMS must remain A/B/C/D')
    return modules


def parse_commands(protocol=None):
    protocol = locked_protocol() if protocol is None else protocol
    runner, reporter = runtime_modules()
    rows = []
    for arm, slot in ARM_SLOTS.items():
        spec = template(arm, protocol)
        for stage in commands(spec, slot):
            argv = stage['argv'][5:]
            module = runner if stage['name'] in ('train', 'TEST') else reporter
            parsed = module.parser().parse_args(argv)
            expected = dict(command=argv[0], root=MODEL_ROOT, arm=arm,
                            device='cpu' if stage['device'] == 'cpu' else 'cuda:0')
            if module is runner:
                expected.update(prepared_root=PREPARED, data_root=DATA, cache_root=CACHE_ROOT,
                    protocol_lock=PROTOCOL_LOCK, gpu_type=SLOTS[slot]['gpu_type'],
                    large_micro_cap=spec['protocol']['large_instance_micro_cap'],
                    eval_batch_size=spec['protocol']['evaluation_batch'], wandb_mode='offline')
                if getattr(parsed, 'resume', False):
                    raise RuntimeError('R92 cannot implicitly resume a trained checkpoint')
            elif stage['name'] == 'LIB':
                expected['lib_batch_size'] = 2
            if any(getattr(parsed, key, None) != value for key, value in expected.items()):
                raise RuntimeError('Final runtime parser differs from R92 stage contract')
            rows.append(dict(arm=arm, slot=slot, command=argv[0], argv=argv, parsed=True))
    return dict(commands=rows, parsed_count=len(rows), training_started=False,
                source_hashes_frozen=False, protocol_written=False)


def require_confirmation(evidence, tested):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True:
        raise RuntimeError('Main must provide explicit ready evidence and focused CPU test confirmation')


def write_specs(evidence=None, tested=False):
    require_confirmation(evidence, tested)
    protocol = locked_protocol()
    parsed = parse_commands(protocol)
    values = {arm: template(arm, protocol) for arm in ARMS}
    for arm, value in values.items():
        path = ROOT / 'specs' / (arm + '.json')
        if path.exists() and base.read(path) != value:
            raise RuntimeError('Existing R92 spec differs; no implicit overwrite')
    for arm, value in values.items():
        path = ROOT / 'specs' / (arm + '.json')
        if not path.exists():
            base.save(path, value, exclusive=True)
    return dict(parsed, specs=[str(ROOT / 'specs' / (arm + '.json')) for arm in ARMS],
                main_ready_evidence=evidence, slots=ARM_SLOTS)


def prepare(path, evidence=None, tested=False):
    require_confirmation(evidence, tested)
    parse_commands()
    result = engine._prepare(path)
    for source in (Path(__file__).resolve(), WRAPPER, REPO / 'code/V4/r87_launch.py',
            REPO / 'code/V4/r85_launch.py', REPO / 'code/V4/test_r92_launch.py', CACHE_EVIDENCE):
        result['source_sha256'][str(source)] = base.sha(source)
    result.update(experiment='R92', cache_mode='existing_read_only', automatic_retry=False,
                  main_thread_ready=False, main_ready_evidence=evidence,
                  predecessor=predecessor_gate(result['spec']['run_id']))
    base.save(base.receipts(result['spec']) / 'plan.json', result)
    return result


def _stages_complete(record, names):
    stages = record.get('stages', [])
    by_name = {stage['name']: stage for stage in stages}
    return len(by_name) == len(stages) and all(
        by_name.get(name, {}).get('state') == 'complete'
        and by_name[name].get('exit_status') == 0 for name in names)


def _r91_recovery(previous, directory, model):
    path = R91_RECOVERY_EVIDENCE
    if not path.is_file():
        raise FileNotFoundError('Waiting for explicit R91 report-only recovery completion')
    receipt = base.read(path)
    if (receipt.get('experiment') != 'R91' or receipt.get('complete') is not True
            or receipt.get('state') != 'recovered_report_only'
            or receipt.get('old_failure_unchanged') is not True
            or receipt.get('evaluation_unchanged') is not True or receipt.get('host') != 'admin01'):
        raise ValueError('R91 recovery-complete receipt is not a completed CPU-only recovery')
    detail_path = Path(receipt['recovery_receipt'])
    if not detail_path.resolve().is_relative_to(PREVIOUS_MODELS['R91'] / 'report_only_recovery'):
        raise ValueError('R91 detailed recovery receipt is outside the explicit recovery root')
    detail = base.read(detail_path)
    if (detail.get('complete') is not True or detail.get('state') != 'recovered_report_only'
            or detail.get('recovery_kind') != 'report_only'
            or detail.get('protected_unchanged') is not True or detail.get('host') != 'admin01'):
        raise ValueError('Detailed R91 report-only recovery is not complete and unchanged')
    for record in (receipt, detail):
        before, after = record.get('protected_sha256_before'), record.get('protected_sha256_after')
        if not before or before != after:
            raise ValueError('R91 recovery changed protected inputs')
        for key in ('training', 'inference', 'test_rerun', 'lib_rerun',
                    'checkpoint_reselection', 'automatic_retry', 'frozen_reporter_modified'):
            if record.get(key) is not False:
                raise ValueError('R91 recovery must explicitly forbid ' + key)
        stages = record.get('report_commands', record.get('commands', []))
        if len(stages) != 3 or any(stage.get('exit_status') != 0 for stage in stages):
            raise ValueError('R91 both reports and comparison must complete rc0')
    if detail['protected_sha256_before'] != receipt['protected_sha256_before']:
        raise ValueError('R91 root and detailed recovery hashes disagree')
    failure = detail['original_failures'][previous]['original_postprocess']
    if (failure.get('phase') != 'postprocess_failed'
            or 'NotoSansCJK-Regular.ttc' not in failure.get('error', '')
            or base.read(model / 'postprocess_status.json') != failure):
        raise ValueError('Only preserved original font-only PDF failure can be recovered')
    lock = base.read(model / 'final_test_lock.json')
    report = receipt['both_arm_report_paths'][previous]
    if report['checkpoint_lock'] != lock:
        raise ValueError('R91 recovery must retain the original checkpoint lock')
    protected = [directory / name for name in ('worker.json', 'controller.json', 'PDF.log')]
    protected += [model / name for name in ('best.pt', 'final_test_lock.json', 'postprocess_status.json')]
    for source in protected:
        if receipt['protected_sha256_before'].get(str(source)) != base.sha(source):
            raise ValueError('R91 original checkpoint or failed receipt changed: ' + str(source))
    if (lock.get('checkpoint') != str(model / 'best.pt')
            or lock.get('checkpoint_sha256') != receipt['protected_sha256_before'][str(model / 'best.pt')]):
        raise ValueError('R91 checkpoint identity differs from the unchanged recovery')
    pdf = model / 'full_model_statistics/MODEL_STATISTICS.pdf'
    if report.get('MODEL_STATISTICS_pdf') != str(pdf) or not pdf.is_file() or not pdf.stat().st_size:
        raise ValueError('R91 completed report PDF is missing')
    return dict(path=str(path), detail=str(detail_path), state=receipt['state'],
                checkpoint_sha256=lock['checkpoint_sha256'], original_failures_preserved=True)


def predecessor_gate(arm):
    if arm not in ARMS:
        raise ValueError('Choose stable generic arm A/B/C/D')
    experiment = 'R91' if arm in ('A', 'C') else 'R89'
    previous = PREDECESSORS[arm]
    directory = PREVIOUS_LAUNCH[experiment] / previous
    model = PREVIOUS_MODELS[experiment] / previous
    result = dict(predecessor=previous, experiment=experiment, complete=False, failed=False, records={})
    try:
        worker, controller = (base.read(directory / name) for name in ('worker.json', 'controller.json'))
        result['records'] = dict(worker=worker, controller=controller)
        if experiment == 'R89':
            for record, names in ((worker, ('train_VALlock_TEST', 'LIB')),
                                 (controller, ('allocation_GPU_pipeline', 'PDF'))):
                if (record.get('state') == 'failed' or record.get('exit_status') not in (None, 0)
                        or any(stage.get('state') == 'failed' or stage.get('exit_status') not in (None, 0)
                               for stage in record.get('stages', []))):
                    raise ValueError('R89 predecessor failed; no automatic repair or retry')
                if record.get('state') != 'complete' or record.get('exit_status') != 0 or not _stages_complete(record, names):
                    raise FileNotFoundError('Waiting for R89 train/TEST/LIB/PDF complete rc0')
            report = base.read(model / 'postprocess_status.json')
            if 'failed' in report.get('phase', '') or report.get('exit_status') not in (None, 0):
                raise ValueError('R89 predecessor report failed; no automatic repair or retry')
            if report.get('phase') != 'complete' or report.get('complete') is not True:
                raise FileNotFoundError('Waiting for R89 completed report receipt')
        else:
            if not _stages_complete(worker, ('train', 'TEST', 'LIB')):
                bad = any(stage.get('name') in ('train', 'TEST', 'LIB')
                    and (stage.get('state') == 'failed' or stage.get('exit_status') not in (None, 0))
                    for stage in worker.get('stages', []))
                if bad:
                    raise ValueError('R91 train/TEST/LIB failed; report recovery cannot bypass it')
                raise FileNotFoundError('Waiting for R91 train/TEST/LIB complete rc0')
            pdf = next((stage for stage in worker.get('stages', []) if stage.get('name') == 'PDF'), {})
            if (worker.get('state') != 'failed' or worker.get('exit_status') != 1
                    or controller.get('state') != 'failed' or controller.get('exit_status') != 1
                    or pdf.get('state') != 'failed' or pdf.get('exit_status') != 1
                    or set(stage['name'] for stage in worker['stages']) != {'train', 'TEST', 'LIB', 'PDF'}
                    or any(stage.get('name') != 'PDF' and stage.get('state') != 'complete'
                           for stage in worker.get('stages', []))):
                raise ValueError('R91 gate expects intentionally preserved font-only failure receipts')
            result['recovery'] = _r91_recovery(previous, directory, model)
        result.update(complete=True, reason=experiment + ' full predecessor delivery confirmed')
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


def start(path, evidence, tested):
    require_confirmation(evidence, tested)
    require_predecessor(path)
    return _start(path, evidence, tested)


def entry(path, slot, token):
    require_predecessor(path)
    return _entry(path, slot, token)


def worker(path, slot, token):
    require_predecessor(path)
    base.assert_container(slot)
    spec, _ = check_ready(path)
    claim = base.read(base.receipts(spec) / 'claim.json')
    if claim['slot'] != slot or claim['token'] != token or os.environ.get(TOKEN_ENV) != token:
        raise RuntimeError('R92 worker requires its fixed claimed slot/token')
    base.save(base.receipts(spec) / 'wandb_lookup.json', dict(selected_mode='offline',
        credential_lookup_performed=False, reason='Explicit offline policy; no credential hunt'), exclusive=True)
    return _worker(path, slot, token)


def pipeline(path, spec, slot, claim):
    """Called under the inherited exact-UUID flock; run GPU work only here."""
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
        return 0
    except BaseException as error:
        record.update(state='failed', exit_status=1, error=str(error), finished_utc=base.utc())
        base.save(output / 'controller.json', record)
        raise


def reserved_cpus(node, exclude):
    occupied = set()
    for path in ROOT.glob('*/claim.json'):
        if path.parent == exclude:
            continue
        claim = base.read(path)
        controller_path = path.with_name('controller.json')
        record = base.read(controller_path) if controller_path.is_file() else {}
        if (claim.get('node') == node and not record.get('gpu_finished_utc')
                and record.get('state') not in ('gpu_complete', 'reporting', 'complete', 'failed')):
            occupied.update(claim['cpus'])
    return occupied


def controller(path):
    require_predecessor(path)
    # The inherited controller returns out of the UUID-lock context after this
    # GPU-only pipeline. PDF is deliberately outside that context and container.
    rc = _controller(path)
    if rc != 0:
        raise RuntimeError('GPU pipeline failed; no PDF or automatic retry')
    spec, _ = check_ready(path)
    output = base.receipts(spec)
    record = base.read(output / 'controller.json')
    try:
        if base.socket.gethostname().split('.')[0] != 'admin01' or not base.FONT.is_file():
            raise RuntimeError('PDF requires ADMIN01 CPU with its CJK font, never the container')
        record['state'] = 'reporting'
        base.save(output / 'controller.json', record)
        for stage in commands(spec, ARM_SLOTS[spec['run_id']]):
            if stage['device'] == 'cpu':
                base.run_stage(stage, spec, record, output / 'controller.json', environment())
        record.update(state='complete', exit_status=0, finished_utc=base.utc())
        base.save(output / 'controller.json', record)
        return 0
    except BaseException as error:
        record.update(state='failed', exit_status=1, error=str(error), finished_utc=base.utc())
        base.save(output / 'controller.json', record)
        raise


engine.ROOT, engine.MODEL_ROOT = ROOT, MODEL_ROOT
engine.WRAPPER, engine.TOKEN_ENV = WRAPPER, TOKEN_ENV
base.ROOT, base.SLOTS, base.TOKEN_ENV = ROOT, SLOTS, TOKEN_ENV
base.self_command, base.load_spec, base.commands = self_command, load_spec, commands
base.environment, base.check_ready = environment, check_ready
base.container_command = engine.container_command
base.pipeline, base.reserved_cpus = pipeline, reserved_cpus
base.session_name = lambda value: 'r92_' + value['run_id']
transport.capture = capture


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
    if args.stage in ('prepare', 'start', 'controller', 'entry', 'worker') and not args.spec:
        parser.error('--spec required')
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
        value = base.node_resources(args.slot)
    elif args.stage == 'container-resources':
        value = base.container_resources(args.slot)
    else:
        return dict(controller=lambda: controller(args.spec),
            entry=lambda: entry(args.spec, args.slot, args.token),
            worker=lambda: worker(args.spec, args.slot, args.token))[args.stage]()
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
