"""Main-only R108 dispatch, privately ported from the R107 working pattern.

Draft specs are not readiness. Reuse the existing gpu05 allocation/container,
physical slot mutexes and warm raw R84 cache; never create another allocation.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / 'code/V4/runs/R108_conditional_density_ratio'
ROOT = MODEL_ROOT / 'preparation/dispatch'
PREDECESSOR_ROOT = REPO / 'code/V4/runs/R106_native_encoder_adaptation'
WRAPPER = REPO / 'code/V4/run_v4_r108_launch.sh'
TOKEN_ENV = 'R108_DISPATCH_TOKEN'
ARMS = ('A_prior_ce', 'B_density_ratio')
ARM_SLOTS = dict(zip(ARMS, ('4090_0', '4090_1')))
PREDECESSORS = dict(zip(ARMS, ('A_frozen', 'B_finetune')))
SLOT_CPUS = {'4090_0': [0, 1, 2, 3], '4090_1': [4, 5, 72, 73]}
DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
PREPARED_ROOT = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
CACHE_SOURCE = REPO / 'code/V4/runs/R84_new90k_attention/input_cache'
CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
DRAFT_CAPS = dict(large_micro_cap=512, other_micro_cap=1024, eval_batch_size=256)
TRAINABLE_PARAMETERS = 1842953
ORIGINAL_FRESH_SHA256 = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
STAGES = ('train', 'TEST', 'timing', 'LIB', 'PDF')
SOURCE_NAMES = ('r108_launch.py', 'r108_wave.py', 'r108_goal.py',
    'r108_model.py', 'r108_objective.py', 'r108_priors.py',
    'r108_experiment.py', 'r108_reporting.py', 'r108_memory_witness.py',
    'run_v4_r108_witness.sh',
    'test_r108_launch.py', 'test_r108_goal.py', 'test_r108_model.py',
    'test_r108_experiment.py', 'test_r108_objective.py', 'test_r108_reporting.py',
    'r106_goal.py', 'r106_launch.py', 'r105_goal.py', 'r104_goal.py', 'r98_launch.py', 'r98_goal.py',
    'r84_launch.py', 'r81_queue.py', 'run_v4_r84_launch.sh', 'run_v4_r81.sh')
PUBLICATION_EXTRAS = (
    'code/V4/runs/R108_conditional_density_ratio/prepared/priors.json',
    'docs/research/deep_direction_review_v2_20261023/REPORT.md',
    'docs/research/deep_direction_review_v2_20261023/REPORT.pdf',
    'docs/research/deep_direction_review_v2_20261023/LITERATURE.md',
)


def private(name, filename):
    spec = importlib.util.spec_from_file_location('code.V4.' + name, REPO / 'code/V4' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = private('_r108_dispatch', 'r98_launch.py')
predecessor_goal = private('_r108_predecessor_admin', 'r106_goal.py')
base = adapter.base
SLOTS = {slot: dict(adapter.SLOTS[slot], cap=DRAFT_CAPS['large_micro_cap']) for slot in ARM_SLOTS.values()}
original_start, original_node_resources = adapter.start, base.node_resources
original_check_ready, original_parse_commands = base.check_ready, adapter.parse_commands


def run_id(arm):
    if arm not in ARMS:
        raise ValueError('Choose A_prior_ce or B_density_ratio')
    return 'R108_' + arm


def arm_of(spec):
    for arm in ARMS:
        if spec.get('run_id') == run_id(arm):
            return arm
    raise ValueError('R108 owns only its two conditional-prior classification arms')


def self_command(stage, spec=None, slot=None, token=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r108_launch', stage]
    for key, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            argv.extend((key, str(value)))
    return argv


def microcaps(science=None):
    """Main-confirmed shared witness caps; never automatic shrink."""
    if science is None:
        return dict(DRAFT_CAPS)
    caps = {key: science.get(key) for key in DRAFT_CAPS}
    if caps != DRAFT_CAPS or any(type(value) is not int for value in caps.values()):
        raise ValueError('Main confirmed shared witness caps512/1024/eval256; no automatic shrink')
    return caps


def locked_protocol(path, spec):
    path = Path(path)
    if path != MODEL_ROOT / 'protocol_lock.json' or not path.is_file():
        raise ValueError('Wait for main R108 protocol_lock.json; draft is not ready')
    science = base.read(path)
    required = dict(experiment='R108', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', data_root=spec['data_root'], prepared_root=spec['prepared_root'],
        primary_cache=spec['cache_root'], rows_per_problem=dict(train=81000, val=4500, test=4500),
        batch_size=1024, logical_global_batch=1024, max_updates=57600, warmup_updates=4320,
        eval_every=1440, train_eval_every=7200, optimizer='AdamW', peak_lr=1e-4,
        model_role_arm='B_execution_anchor', role_feature_slot=3,
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        external_pretraining=False, end_lr=1e-6, dropout=.1,
        loss=dict(ce_or_density_ratio=1., comparison=0., risk=0., kl=0., dropout_passes=1),
        trainable_parameters=dict(original=TRAINABLE_PARAMETERS, **dict.fromkeys(ARMS, TRAINABLE_PARAMETERS)),
        identical_complete_fresh_parameters=True, original_fresh_trainable_sha256=ORIGINAL_FRESH_SHA256,
        priors_path=str(MODEL_ROOT / 'prepared/priors.json'))
    if not isinstance(science, dict) or any(science.get(k) != v for k, v in required.items()):
        raise ValueError('Require main-locked R108 full18 NEW90k seed2 logical1024/57600 scratch R84 recipe')
    microcaps(science)
    prior = MODEL_ROOT / 'prepared/priors.json'
    if not prior.is_file() or science.get('priors_sha256') != base.sha(prior):
        raise ValueError('Main must generate and lock the small TRAIN-only priors.json before final specs')
    return science


def expected_stages(arm, science=None):
    run_id(arm)
    caps = microcaps(science)
    result = []
    for name, command in zip(STAGES, ('train', 'test', 'timing', 'lib', 'report')):
        reporter = name in ('LIB', 'PDF')
        options = ['--root', str(MODEL_ROOT), '--arm', arm]
        if not reporter:
            options += ['--prepared-root', str(PREPARED_ROOT), '--data-root', str(DATA_ROOT),
                '--cache-root', '{cache_root}', '--protocol-lock', str(MODEL_ROOT / 'protocol_lock.json'),
                '--gpu-type', '4090', '--large-micro-cap', str(caps['large_micro_cap']),
                '--eval-batch-size', str(caps['eval_batch_size']), '--wandb-mode', 'offline']
        options += ['--device', '{device}']
        if name == 'LIB':
            options += ['--lib-batch-size', '2']
        result.append(dict(name=name, device='cpu' if name == 'PDF' else 'gpu',
            argv=['{python}', '-B', '-u', '-m',
                'code.V4.r108_reporting' if reporter else 'code.V4.r108_experiment', command, *options]))
    return result


def load_spec(path):
    spec = adapter.original_load_spec(path)
    arm = arm_of(spec)
    if spec.get('spec_draft') is not False or spec.get('main_thread_ready') is not False:
        raise ValueError('Only a final spec may be prepared; a spec itself is never main-ready')
    science = locked_protocol(spec['science_protocol'], spec)
    expected = dict(run_dir=str(MODEL_ROOT / arm), common_root=str(MODEL_ROOT),
        science_protocol=str(MODEL_ROOT / 'protocol_lock.json'), data_root=str(DATA_ROOT),
        prepared_root=str(PREPARED_ROOT), cache_source=str(CACHE_SOURCE), cache_root=str(CACHE_ROOT),
        cache_kind='r84_inputs', wandb_mode='offline', slots=[ARM_SLOTS[arm]],
        cpu_count=4, memory_gib=9, cpus=SLOT_CPUS[ARM_SLOTS[arm]], stages=expected_stages(arm, science))
    forbidden = ('feature_root', 'routing_feature_root', 'routing_stage_argv', 'routing_stage_json',
        'teacher_root', 'context_root', 'preprocessing', 'feature_stage', 'donor_cache', 'solver_root')
    if any(spec.get(k) != v for k, v in expected.items()) or any(k in spec for k in forbidden):
        raise ValueError('Use exact R108 train/TEST/timing/LIB2/CPU-PDF, gpu05 CPU quartets and raw R84 reuse')
    return spec


def declared_sources():
    return [REPO / 'code/V4' / name for name in SOURCE_NAMES] + [WRAPPER,
        MODEL_ROOT / 'protocol_lock.json', MODEL_ROOT / 'PLAN.md', MODEL_ROOT / 'RUNNER_PREP.md',
        MODEL_ROOT / 'MODEL_PREP.md',
        MODEL_ROOT / 'REPORT_PREP.md', MODEL_ROOT / 'preparation/LAUNCH_PLAN.md',
        MODEL_ROOT / 'prepared/priors.json',
        *(PREPARED_ROOT / name for name in
            ('prepared.json', 'geometry_stats.json', 'source_cache_manifest.json', 'identicalinit_seed2.pt'))]


def source_files(spec):
    sources = declared_sources()
    declared = {str(Path(p).resolve()) for p in sources}
    supplied = spec.get('source_files', [])
    if not isinstance(supplied, list) or any(str(Path(p).resolve()) not in declared for p in supplied):
        raise ValueError('R108 source declarations accept only exact named dependencies')
    missing = [str(p) for p in sources if not Path(p).is_file()]
    if missing:
        raise RuntimeError('Sources pending; not main-ready: ' + ', '.join(missing))
    tests = {str(Path(p).resolve()) for p in sources if Path(p).name.startswith('test_')}
    runtime = [p for p in sources if str(Path(p).resolve()) not in tests]
    return sorted(set(adapter.source_closure(runtime)) | tests)


def explicit_ready(evidence, tested):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True:
        raise ValueError('Explicit main-ready evidence and passed CPU tests are required')


def parse_commands(spec):
    parsed = original_parse_commands(spec)
    program = ('import json,sys; from code.V4 import r108_experiment as e, r108_reporting as r; '
        'a=e.parser().parse_args(json.loads(sys.argv[1])); e.protocol_lock(a); '
        'r.parser().parse_args(["compare","--root",str(a.root),"--device","cpu"]); '
        'print("R108 main-lock and compare CLI verified")')
    process = base.subprocess.run([sys.executable, '-B', '-c', program, json.dumps(parsed[0]['argv'])],
        cwd=REPO, env=environment(), capture_output=True, text=True, timeout=90)
    if process.returncode:
        raise RuntimeError('R108 runner science/compare CLI is not ready: ' + process.stderr[-3000:])
    return parsed


def check_ready(path):
    spec, plan = original_check_ready(path)
    ready = base.read(base.receipts(spec) / 'ready.json')
    if (plan.get('main_thread_ready') is not True or plan.get('evidence') != ready.get('evidence')
            or ready.get('source_freeze_ready') is not True
            or plan.get('primary_cache_reuse_only') is not True
            or plan.get('cache_copy') is not False or plan.get('cache_preprocessing') is not False
            or plan.get('automatic_retry') is not False
            or not plan.get('source_sha256') or set(plan['source_sha256']) != set(source_files(spec))):
        raise RuntimeError('Explicit main-ready exact source set required; no automatic refreeze')
    return spec, plan


def prepare(path, evidence=None, tested=False):
    explicit_ready(evidence, tested)
    spec = load_spec(path)
    sources = source_files(spec)
    result = adapter.prepare(path, evidence, tested)
    output = base.receipts(spec)
    plan = base.read(output / 'plan.json')
    plan.update(source_sha256={name: base.sha(name) for name in sources},
        primary_cache_reuse_only=True, cache_preprocessing=False, cache_copy=False,
        automatic_retry=False, predecessor='R106_' + PREDECESSORS[arm_of(spec)])
    base.save(output / 'plan.json', plan)
    ready = base.read(output / 'ready.json')
    ready.update(plan_sha256=base.sha(output / 'plan.json'), source_freeze_ready=True)
    base.save(output / 'ready.json', ready)
    check_ready(path)
    return dict(result, source_freeze_ready=True, cache_copied=False)


def stage_cache(spec, manifest):
    root = Path(spec['cache_root'])
    stamp = root / '.r84_cache_identity.json'
    with base.lock(root.with_name(root.name + '.r84_stage.lock'), blocking=True):
        if not stamp.is_file() or base.read(stamp).get('identity_sha256') != manifest['identity_sha256']:
            raise RuntimeError('Original R84 local cache required; no preprocessing, rebuild or copy')
        for row in manifest['files']:
            path = root / row['path']
            if (not path.is_file() or path.is_symlink() or path.stat().st_size != row['bytes']
                    or ('sha256' in row and base.sha(path) != row['sha256'])):
                raise RuntimeError('Original R84 cache incomplete; no repair or copy')
    return dict(root=str(root), reused=True, identity_sha256=manifest['identity_sha256'], copied=False)


def predecessor_gate(arm):
    run_id(arm)
    prior_arm = PREDECESSORS[arm]
    model = PREDECESSOR_ROOT
    prior_dispatch = model / 'preparation/dispatch'
    run, parent_id = model / prior_arm, 'R106_' + prior_arm
    output = prior_dispatch / parent_id
    current = predecessor_goal.status('R106', model, prior_dispatch, persist=False)
    row = next(r for r in current['arms'] if r['arm'] == prior_arm)
    checks = dict(row['checks'])
    spec = predecessor_goal.optional(prior_dispatch / 'specs' / (parent_id + '.json'))
    checks['assigned_predecessor'] = (spec.get('run_id') == parent_id
        and spec.get('run_dir') == str(run) and spec.get('slots') == [ARM_SLOTS[arm]])
    checks['exact_pipeline'] = [(s.get('name'), s.get('device')) for s in spec.get('stages', [])] == [
        (name, 'cpu' if name == 'PDF' else 'gpu') for name in STAGES]
    for name, stages in (('worker.json', ['train', 'TEST', 'timing', 'LIB']),
                         ('controller.json', ['allocation_GPU_pipeline', 'PDF'])):
        record = predecessor_goal.optional(output / name)
        checks['original_' + name.removesuffix('.json') + '_rc0'] = (
            predecessor_goal.stages_ready(record, stages)
            and not predecessor_goal.receipt_errors(output / name, record))
    errors = []
    for error in [*current.get('errors', []), *current.get('invalid_receipts', [])]:
        path = error.get('path') if isinstance(error, dict) else None
        if path is None or Path(path).is_relative_to(run) or Path(path).is_relative_to(output):
            errors.append(error)
    return dict(arm=arm, predecessor=parent_id, complete=not errors and all(checks.values()),
        checks=checks, errors=errors, healthy_runs_stopped=False, automatic_retry=False)


def start(path, evidence=None, tested=False):
    explicit_ready(evidence, tested)
    spec, plan = check_ready(path)
    if plan.get('evidence') != evidence:
        raise ValueError('Start evidence must match main-ready plan')
    gate = predecessor_gate(arm_of(spec))
    if gate['errors'] or not gate['complete']:
        raise RuntimeError('Wait for assigned original R106 same-lock TRAIN/VAL/TEST/timing/LIB/PDF and rc0 receipts')
    return original_start(path, evidence, tested)


def queue(path, evidence=None, tested=False, interval=60):
    explicit_ready(evidence, tested)
    if interval <= 0:
        raise ValueError('Queue interval must be positive')
    spec, plan = check_ready(path)
    if plan.get('evidence') != evidence:
        raise ValueError('Queue evidence must match main-ready plan')
    output = base.receipts(spec)
    with base.lock(output / 'queue.lock'):
        receipt_path = output / 'queue_receipt.json'
        if receipt_path.exists():
            raise RuntimeError('Queue already attempted; no replacement or retry')
        receipt = dict(state='waiting_predecessor', cpu_only=True, automatic_retry=False,
            started_utc=base.utc(), main_ready_evidence=evidence)
        base.save(receipt_path, receipt, exclusive=True)
        try:
            while True:
                check_ready(path)
                gate = predecessor_gate(arm_of(spec))
                if gate['errors']:
                    raise RuntimeError('Assigned R106 predecessor failed: ' + json.dumps(gate['errors']))
                if gate['complete']:
                    result = start(path, evidence, tested)
                    receipt.update(state='complete', exit_status=0, completed_utc=base.utc(), start=result)
                    base.save(receipt_path, receipt)
                    return result
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=base.utc())
            base.save(receipt_path, receipt)
            raise


def controller(path):
    spec, _ = check_ready(path)
    gate = predecessor_gate(arm_of(spec))
    if gate['errors'] or not gate['complete']:
        raise RuntimeError('Controller cannot bypass its original R106 completion gate')
    return base.controller(path)


def node_resources(slot):
    probe = original_node_resources(slot)
    cpus = SLOT_CPUS[slot]
    if not set(cpus) <= set(probe['allowed_cpus']):
        raise RuntimeError('Original gpu05 CPU quartet is outside allocation/container affinity')
    return dict(probe, allowed_cpus=list(cpus))


# Only private instances are rebound; historical/live launchers keep their globals.
adapter.SLOTS = base.SLOTS = SLOTS
adapter.ROOT = base.ROOT = ROOT
adapter.TOKEN_ENV = base.TOKEN_ENV = TOKEN_ENV
adapter.WRAPPER = WRAPPER
adapter.self_command = base.self_command = self_command
adapter.load_spec = base.load_spec = load_spec
adapter.parse_commands = parse_commands
base.check_ready = check_ready
base.container_command, base.environment = adapter.container_command, adapter.environment
base.reserved_cpus, base.stage_cache = adapter.reserved_cpus, stage_cache
base.node_resources = node_resources
base.session_name = lambda spec: 'goal_' + spec['run_id'].lower()
environment, container_command = adapter.environment, adapter.container_command


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('prepare', 'queue', 'start', 'predecessor', 'controller',
        'node-resources', 'container-resources', 'entry', 'worker'))
    value.add_argument('--spec', type=Path)
    value.add_argument('--arm', choices=ARMS)
    value.add_argument('--slot', choices=tuple(SLOTS))
    value.add_argument('--token')
    value.add_argument('--evidence')
    value.add_argument('--runner-tests-passed', action='store_true')
    value.add_argument('--interval', type=int, default=60)
    return value


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.stage == 'predecessor':
        if not (args.arm or args.spec):
            cli.error('predecessor requires --arm or --spec')
        result = predecessor_gate(args.arm or arm_of(load_spec(args.spec)))
    elif args.stage in ('node-resources', 'container-resources'):
        if not args.slot:
            cli.error(args.stage + ' requires --slot')
        result = getattr(base, args.stage.replace('-', '_'))(args.slot)
    else:
        if args.spec is None:
            cli.error(args.stage + ' requires --spec')
        if args.stage in ('prepare', 'start'):
            result = globals()[args.stage](args.spec, args.evidence, args.runner_tests_passed)
        elif args.stage == 'queue':
            result = queue(args.spec, args.evidence, args.runner_tests_passed, args.interval)
        elif args.stage == 'controller':
            return controller(args.spec)
        else:
            if not (args.slot and args.token):
                cli.error(args.stage + ' requires --slot and --token')
            return getattr(base, args.stage)(args.spec, args.slot, args.token)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
