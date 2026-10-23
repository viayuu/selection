"""R106-only, main-authorized one-shot dispatch. Imports never probe or launch.

The stable R98/R84 transport is privately loaded. Shared allocation, container
and mutex identities survive; scientific sources and historical workers do not
get rebound. A draft is deliberately not sufficient for prepare/start.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import uuid


REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / 'code/V4/runs/R106_native_encoder_adaptation'
ROOT = MODEL_ROOT / 'preparation/dispatch'
SHARED_DISPATCH = REPO / 'code/V4/runs/R98_parallel_dispatch'
WRAPPER = REPO / 'code/V4/run_v4_r106_launch.sh'
TOKEN_ENV = 'R106_DISPATCH_TOKEN'
ARMS = ('A_frozen', 'B_finetune')
ARM_SLOTS = dict(zip(ARMS, ('4090_0', '4090_1')))
SLOT_CPUS = {'4090_0': [0, 1, 2, 3], '4090_1': [4, 5, 72, 73]}
CONTAINER = '76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6'
DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
PREPARED_ROOT = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
CACHE_SOURCE = REPO / 'code/V4/runs/R84_new90k_attention/input_cache'
CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
STAGES = ('train', 'TEST', 'timing', 'LIB', 'PDF')
SOURCE_NAMES = ('r106_launch.py', 'r106_goal.py', 'r106_wave.py', 'r106_donor.py',
    'r106_model.py', 'r106_experiment.py', 'r106_reporting.py', 'r106_memory_witness.py', 'test_r106_launch.py',
    'test_r106_model.py', 'test_r106_experiment.py', 'test_r106_reporting.py',
    'r98_launch.py', 'r84_launch.py', 'r81_queue.py', 'r98_goal.py',
    'run_v4_r84_launch.sh', 'run_v4_r81.sh')
OPTIONAL_SOURCE_NAMES = ('test_r106_donor.py',)
HOME = Path('/public/home/shiys')
NATIVE_SOURCES = (HOME / 'methods\u8bba\u6587_mineru_output/source_code/ELG/TSP/models.py',
    HOME / 'EasyNCO/neural_solvers/methods/matnet/matnet_encoder.py',
    HOME / 'EasyNCO/neural_solvers/backbones/Transformer/attr_component.py',
    HOME / 'reld-nco-main/Multi-Task/models/MOEModel_Light.py',
    HOME / 'reld-nco-main/Multi-Task/models/MOELayer.py')


def private(name, filename):
    spec = importlib.util.spec_from_file_location('code.V4.' + name, REPO / 'code/V4' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = private('_r106_dispatch', 'r98_launch.py')
base = adapter.base
SLOTS = {slot: dict(adapter.SLOTS[slot], cap=64) for slot in ARM_SLOTS.values()}
original_node_resources = base.node_resources
original_check_ready = base.check_ready
original_start = adapter.start
original_parse_commands = adapter.parse_commands


def run_id(arm):
    if arm not in ARMS:
        raise ValueError('Choose A_frozen or B_finetune')
    return 'R106_' + arm


def arm_of(spec):
    for arm in ARMS:
        if spec.get('run_id') == run_id(arm):
            return arm
    raise ValueError('R106 owns only its two native-adaptation arms')


def normalize_gpu_uuid(value):
    """Torch may omit GPU-; require the complete UUID, never a prefix match."""
    if isinstance(value, bytes):
        value = value.decode('ascii')
    value = str(value).strip()
    if value.lower().startswith('gpu-'):
        value = value[4:]
    if not re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value):
        raise ValueError('A complete CUDA/NVML GPU UUID is required, not a string prefix')
    return 'GPU-' + str(uuid.UUID(value))


def verify_uuid_binding(slot, torch_uuid, nvml_uuid):
    expected = normalize_gpu_uuid(SLOTS[slot]['uuid'])
    if not str(nvml_uuid.decode('ascii') if isinstance(nvml_uuid, bytes) else nvml_uuid).lower().startswith('gpu-'):
        raise ValueError('NVML must supply its true full GPU- UUID')
    if normalize_gpu_uuid(torch_uuid) != expected or normalize_gpu_uuid(nvml_uuid) != expected:
        raise ValueError('Torch/NVML UUID does not match the assigned R106 GPU')
    return dict(slot=slot, expected_uuid=expected, torch_uuid=expected, nvml_uuid=expected)


def self_command(stage, spec=None, slot=None, token=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r106_launch', stage]
    for key, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            argv.extend((key, str(value)))
    return argv


def microcaps(science=None):
    science = science or {}
    return {key: science.get(key, value) for key, value in (
        ('large_micro_cap', 64), ('atsp_micro_cap', 128),
        ('other_micro_cap', 256), ('eval_batch_size', 128))}


def locked_protocol(path, spec):
    path = Path(path)
    if path != MODEL_ROOT / 'protocol_lock.json' or not path.is_file():
        raise ValueError('Wait for the main R106 protocol lock; draft is not ready')
    science = base.read(path)
    required = dict(experiment='R106', locked=True, arms=list(ARMS), seed=2, task_scope='full18',
        data_root=spec['data_root'], prepared_root=spec['prepared_root'],
        rows_per_problem=dict(train=81000, val=4500, test=4500),
        warmup_updates=4320, eval_every=1440, train_eval_every=7200,
        current_query_labels_in_forward=False)
    logical = science.get('logical_global_batch', science.get('global_batch_size', science.get('batch_size')))
    budget = science.get('max_updates', science.get('max_successful_updates'))
    if (any(science.get(k) != v for k, v in required.items()) or logical != 1024 or budget != 57600
            or science.get('primary_cache', science.get('cache_root')) != spec['cache_root']
            or any(type(science.get(k)) is not int or science[k] <= 0 for k in microcaps())):
        raise ValueError('Require main-locked R106 full18 NEW90k seed2 logical1024/57600 and final positive microcaps')
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
                '--gpu-type', '4090', '--wandb-mode', 'offline']
            for key, value in caps.items():
                options += ['--' + key.replace('_', '-'), str(value)]
        options += ['--device', '{device}']
        if name == 'LIB':
            options += ['--lib-batch-size', '2']
        result.append(dict(name=name, device='cpu' if name == 'PDF' else 'gpu',
            argv=['{python}', '-B', '-u', '-m',
                'code.V4.r106_reporting' if reporter else 'code.V4.r106_experiment', command, *options]))
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
    forbidden = ('feature_root', 'routing_feature_root', 'feature_stage', 'preprocessing',
        'teacher_root', 'context_root', 'routing_stage_argv', 'routing_stage_json', 'donor_cache')
    if any(spec.get(k) != v for k, v in expected.items()) or any(k in spec for k in forbidden):
        raise ValueError('Use exact R106 online train/TEST/timing/LIB/CPU-PDF, assigned4090 CPU4/mem9 and raw R84 reuse')
    return spec


def declared_sources():
    return [REPO / 'code/V4' / name for name in SOURCE_NAMES] + [
        REPO / 'code/V4' / name for name in OPTIONAL_SOURCE_NAMES
        if (REPO / 'code/V4' / name).is_file()] + [WRAPPER,
        MODEL_ROOT / 'protocol_lock.json', MODEL_ROOT / 'PLAN.md',
        MODEL_ROOT / 'RUNNER_PREP.md', MODEL_ROOT / 'preparation/LAUNCH_PLAN.md', *NATIVE_SOURCES,
        *(PREPARED_ROOT / name for name in ('prepared.json', 'geometry_stats.json',
            'source_cache_manifest.json', 'identicalinit_seed2.pt'))]


def source_files(spec):
    sources = declared_sources()
    declared = {str(Path(p).resolve()) for p in sources}
    supplied = spec.get('source_files', [])
    if not isinstance(supplied, list) or any(str(Path(p).resolve()) not in declared for p in supplied):
        raise ValueError('Source declarations accept only the exact named R106 dependencies')
    missing = [str(p) for p in sources if not Path(p).is_file()]
    if missing:
        raise RuntimeError('Sources pending; not main-ready: ' + ', '.join(missing))
    tests = {str(Path(p).resolve()) for p in sources if Path(p).name.startswith('test_')}
    return sorted(set(adapter.source_closure([p for p in sources if str(Path(p).resolve()) not in tests])) | tests)


def explicit_ready(evidence, tested):
    if not isinstance(evidence, str) or not evidence.strip() or tested is not True:
        raise ValueError('Explicit main evidence and passed CPU tests required; no implicit readiness')


def parse_commands(spec):
    parsed = original_parse_commands(spec)
    train = parsed[0]['argv']
    program = ('import json,sys; from code.V4 import r106_experiment as e, r106_reporting as r; '
        'a=e.parser().parse_args(json.loads(sys.argv[1])); e.protocol_lock(a); '
        'r.parser().parse_args(["compare","--root",str(a.root),"--device","cpu"]); '
        'print("R106 full main-lock and compare CLI verified")')
    process = base.subprocess.run([sys.executable, '-B', '-c', program, json.dumps(train)],
        cwd=REPO, env=environment(), capture_output=True, text=True, timeout=90)
    if process.returncode:
        raise RuntimeError('R106 runner science/compare CLI is not ready: ' + process.stderr[-3000:])
    return parsed


def check_ready(path):
    spec, plan = original_check_ready(path)
    ready = base.read(base.receipts(spec) / 'ready.json')
    if (plan.get('main_thread_ready') is not True or plan.get('evidence') != ready.get('evidence')
            or ready.get('source_freeze_ready') is not True
            or plan.get('primary_cache_reuse_only') is not True or plan.get('cache_copy') is not False
            or plan.get('cache_preprocessing') is not False or plan.get('automatic_retry') is not False
            or set(plan.get('source_sha256', {})) != set(source_files(spec))):
        raise RuntimeError('Main-ready exact source freeze required; no automatic refreeze')
    return spec, plan


def prepare(path, evidence=None, tested=False):
    explicit_ready(evidence, tested)
    spec = load_spec(path)
    sources = source_files(spec)
    result = adapter.prepare(path, evidence, tested)
    output = base.receipts(spec)
    plan = base.read(output / 'plan.json')
    plan.update(source_sha256={name: base.sha(name) for name in sources},
        primary_cache_reuse_only=True, cache_copy=False, cache_preprocessing=False,
        automatic_retry=False, native_execution='online_all_splits_both_arms')
    base.save(output / 'plan.json', plan)
    ready = base.read(output / 'ready.json')
    ready.update(plan_sha256=base.sha(output / 'plan.json'), source_freeze_ready=True)
    base.save(output / 'ready.json', ready)
    check_ready(path)
    return dict(result, source_freeze_ready=True, gpu_started=False, cache_copied=False)


def stage_cache(spec, manifest):
    root = Path(spec['cache_root'])
    stamp = root / '.r84_cache_identity.json'
    with base.lock(root.with_name(root.name + '.r84_stage.lock'), blocking=True):
        if root.is_symlink() or not stamp.is_file() or base.read(stamp).get('identity_sha256') != manifest['identity_sha256']:
            raise RuntimeError('Original frozen R84 cache required; no generation or copy')
        for row in manifest['files']:
            path = root / row['path']
            if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
                    or not path.is_file() or path.stat().st_size != row['bytes']
                    or ('sha256' in row and base.sha(path) != row['sha256'])):
                raise RuntimeError('Frozen R84 cache incomplete; no repair or copy')
    return dict(root=str(root), reused=True, identity_sha256=manifest['identity_sha256'], copied=False)


def node_resources(slot):
    probe = original_node_resources(slot)
    if not set(SLOT_CPUS[slot]) <= set(probe['allowed_cpus']):
        raise RuntimeError('Assigned gpu05 CPU quartet is outside allocation/container affinity')
    return dict(probe, allowed_cpus=list(SLOT_CPUS[slot]))


def reserved_cpus(node, exclude):
    occupied = set()
    for root in (ROOT, SHARED_DISPATCH, *(REPO / ('code/V4/runs/R' + v + '_launch')
            for v in ('84', '92', '93', '94', '95', '97'))):
        for path in root.glob('*/claim.json'):
            if path.parent == exclude:
                continue
            claim = base.read(path)
            controller = path.with_name('controller.json')
            state = base.read(controller).get('state') if controller.is_file() else None
            if claim.get('node') == node and state not in ('complete', 'failed'):
                occupied.update(claim['cpus'])
    return occupied


def start(path, evidence=None, tested=False):
    explicit_ready(evidence, tested)
    _, plan = check_ready(path)
    if plan.get('evidence') != evidence:
        raise ValueError('Start evidence must match the main-ready freeze')
    return original_start(path, evidence, tested)


def controller(path):
    if base.socket.gethostname().split('.')[0] != 'admin01' or not base.FONT.is_file():
        raise RuntimeError('Controller/PDF must remain on admin01 with CJK fonts')
    spec, _ = check_ready(path)
    output = base.receipts(spec)
    with base.lock(output / 'controller.lock'):
        if any((output / name).exists() for name in ('claim.json', 'controller.json')):
            raise RuntimeError('No duplicate launch or implicit resume/retry')
        slot = spec['slots'][0]
        item = SLOTS[slot]
        while True:
            claimed = False
            try:
                gpu_mutex = base.transport.gpu_lock_path(item['uuid'])
                with base.lock(gpu_mutex, existing=gpu_mutex.exists()):
                    with base.lock(SHARED_DISPATCH / (item['node'] + '.lock')):
                        probe = json.loads(base.transport.capture(base.srun_command('node-resources', slot), timeout=90))
                        gpu = probe['gpus'][slot]
                        occupied = {cpu for row in probe['processes'] for cpu in row['cpus']}
                        occupied |= reserved_cpus(item['node'], output)
                        cpus = SLOT_CPUS[slot]
                        available = (not gpu['compute_processes'] and gpu['memory_used_mib'] <= 512
                            and not gpu['utilization_pct'] and not (set(cpus) & occupied)
                            and set(cpus) <= set(probe['allowed_cpus'])
                            and probe['memory']['headroom_excluding_reclaimable_cache_bytes'] >= 9 * 2**30)
                        if available:
                            check_ready(path)
                            claim = dict(slot=slot, node=item['node'], gpu_uuid=item['uuid'], cpus=cpus,
                                token=uuid.uuid4().hex, claimed_utc=base.utc(), resources=probe)
                            base.save(output / 'claim.json', claim, exclusive=True)
                            claimed = True
                    if claimed:
                        return base.pipeline(path, spec, slot, claim)
            except BlockingIOError:
                if claimed:
                    raise
            base.save(output / 'status.json', dict(state='waiting_resources', updated_utc=base.utc(),
                training_started=False, tmux=base.session_name(spec)))
            time.sleep(30)


# Rebind private instances only; retain the original GPU and shared node mutexes.
adapter.SLOTS = base.SLOTS = SLOTS
adapter.ROOT = base.ROOT = ROOT
adapter.TOKEN_ENV = base.TOKEN_ENV = TOKEN_ENV
adapter.WRAPPER = WRAPPER
adapter.self_command = base.self_command = self_command
adapter.load_spec = base.load_spec = load_spec
adapter.parse_commands = parse_commands
base.check_ready = check_ready
base.environment, base.container_command = adapter.environment, adapter.container_command
base.stage_cache, base.node_resources, base.reserved_cpus = stage_cache, node_resources, reserved_cpus
base.session_name = lambda spec: 'goal_' + spec['run_id'].lower()
environment, container_command = adapter.environment, adapter.container_command


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('prepare', 'start', 'controller', 'node-resources',
        'container-resources', 'entry', 'worker'))
    value.add_argument('--spec', type=Path)
    value.add_argument('--slot', choices=tuple(SLOTS))
    value.add_argument('--token')
    value.add_argument('--evidence')
    value.add_argument('--runner-tests-passed', action='store_true')
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage in ('prepare', 'start'):
        result = globals()[args.stage](args.spec, args.evidence, args.runner_tests_passed)
    elif args.stage == 'controller':
        return controller(args.spec)
    elif args.stage in ('entry', 'worker'):
        return getattr(base, args.stage)(args.spec, args.slot, args.token)
    else:
        result = getattr(base, args.stage.replace('-', '_'))(args.slot)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
