"""Conditional R74 delivery and its single locked, live-acquisition TEST."""

import argparse
import csv
import importlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from .r73_frozen_evaluation import append, capture, exclusive, now, read, write


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'code/V4/runs/R74_adaptive_incomplete_acquisition'
RUN = ROOT / 'B_adaptive_seed2'
CONTROL = WORKSPACE / 'code/V4/runs/R73_expansion_teacher_training'
CONTROL_RUN = CONTROL / 'B_teacher_training_seed2'
GPU_UUID = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'
OWNED = ('r74_delivery.py', 'r74_analysis.py', 'run_v4_r74_delivery.sh')
FAILED = ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED')
LAYOUT = dict(base_batch=128, base_task_tail=104, base_batches=144, probe_batch=1,
    policy_queries=18000, original_order=True, live_selected_paths_only=True,
    retain_native_state_when_deepening=True, crop_base_nodes_to_true_node_count=True,
    counterfactual_TEST_cache=False, completed_solver_executions=0)


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def delivery_hashes():
    from .multitask_probe import file_hash
    paths = [WORKSPACE / 'code/V4' / name for name in OWNED] + [ROOT / 'DELIVERY_PROTOCOL.md']
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def require_preparation():
    authorization, preparation = read(ROOT / 'delivery_authorization.json'), read(ROOT / 'delivery_preparation.json')
    if (authorization['scope'] != 'conditional_original_R74_delivery' or authorization['original_step'] != '1465.111'
            or authorization['allowed_new_TEST_passes'] != {'A': 0, 'B': 1}
            or authorization['allowed_analysis_invocations'] != 1 or canonical(authorization['test_layout']) != LAYOUT
            or not preparation['syntax_passed'] or preparation['syntax_check']['returncode'] != 0
            or preparation['delivery_source_identities'] != delivery_hashes()):
        raise ValueError('Require the unchanged parent conditional authorization and one passed delivery syntax invocation')
    return authorization, preparation


def original_identity():
    handoff, launch = read(ROOT / 'execution_handoff.json'), read(ROOT / 'trainer_launch.json')
    if (handoff['slurm_step'] != '1465.111' or launch['slurm_step'] != '1465.111'
            or handoff['trainer_pid'] != 580952 or launch['trainer_pid'] != 580952
            or handoff['tmux_pane'] != '%129' or launch['tmux_pane'] != '%129'
            or str(handoff['tmux_pane_pid']) != '3450348' or launch['tmux_pane_pid'] != '3450348'
            or launch['gpu_uuid'] != GPU_UUID or launch['gpu_index'] != 1
            or launch['host'].split('.')[0] != 'gpu03' or launch['tmux_session'] != 'r74_train'):
        raise ValueError('Only the original recorded R74 formal process is authorized')
    return launch


def absent(observation):
    return observation['returncode'] == 1 and any(text in observation['stderr'] for text in
        ("can't find session", 'no server running', 'no sessions'))


def terminal(step, session):
    from .r71_frozen_evaluation import terminal_observation
    return terminal_observation(step, session)


def require_terminal(receipt, session, step):
    from .r69_frozen_evaluation import require_scheduler_terminal
    require_scheduler_terminal(receipt, session, step)


def wait_original():
    launch = original_identity()
    while True:
        current, pipeline = read(ROOT / 'current_status.json'), read(ROOT / 'pipeline_status.json')
        owned = capture(['tmux', 'has-session', '-t', '=r74_train'])
        failures = [name for name in ('failures.jsonl', 'bootstrap_failures.jsonl') if (ROOT / name).exists()]
        last = current.get('last_batch', current.get('last_metrics', {}))
        observation = dict(utc=now(), stage='waiting_original_formal_process', phase=current['stage'],
            status_utc=current['updated_utc'], original_step=launch['slurm_step'],
            original_trainer_pid=launch['trainer_pid'], owned_training_tmux=owned,
            successful_updates=current.get('successful_updates', last.get('successful_update', 0)),
            generation_counters=last.get('counters', current.get('counters')),
            native_presentations=last.get('native_presentations'),
            successful_leaf_view_rows=last.get('successful_leaf_view_rows'),
            pipeline_stage=pipeline['stage'], formal_failure_files=failures, scheduler_queried=False,
            GPU_queries=0, model_loads=0, array_loads=0, evaluation_forwards=0)
        append(ROOT / 'delivery_monitor.jsonl', observation)
        write(ROOT / 'delivery_current_status.json', observation)
        if not (ROOT / 'delivery_first_wait.json').exists():
            exclusive(ROOT / 'delivery_first_wait.json', observation)
        print('[R74 delivery wait] '+json.dumps({key: observation[key] for key in
            ('phase', 'successful_updates', 'generation_counters', 'formal_failure_files')}), flush=True)
        if failures or pipeline['stage'] == 'failed_preserved_no_retry':
            actual = terminal(launch['slurm_step'], 'r74_train')
            exclusive(ROOT / 'delivery_formal_failure_observation.json', dict(actual, failure_files=failures,
                no_stop_or_restart=True))
            raise RuntimeError('Original formal failure preserved; no generation/training restart or recipe change')
        if owned['returncode'] != 0 and not absent(owned):
            raise RuntimeError('Cannot establish owned training tmux state; preserve observation')
        if absent(owned):
            actual = terminal(launch['slurm_step'], 'r74_train')
            append(ROOT / 'delivery_terminal_observations.jsonl', actual)
            if actual['state'].startswith(FAILED) or (actual['state'] == 'COMPLETED' and actual['exit_code'] != '0:0'):
                exclusive(ROOT / 'training_terminal_receipt.json', actual)
                raise RuntimeError('Original formal Slurm step failed; no restart')
            if actual['state'] == 'COMPLETED' and actual['tmux_terminal']:
                require_terminal(actual, 'r74_train', launch['slurm_step'])
                actual.update(original_trainer_pid=launch['trainer_pid'], original_pane_pid=launch['tmux_pane_pid'],
                    formal_process_gone=True, formal_process_evidence='Original exec Python belongs to the terminal '
                    'COMPLETED/0:0 step; independent owned launcher tmux absence confirms its exit')
                if (ROOT / 'training_terminal_receipt.json').exists():
                    require_terminal(read(ROOT / 'training_terminal_receipt.json'), 'r74_train', launch['slurm_step'])
                    exclusive(ROOT / 'delivery_training_terminal_observation.json', actual)
                else:
                    exclusive(ROOT / 'training_terminal_receipt.json', actual)
                completed_training()
                return actual
        time.sleep(240)


def completed_training():
    from code.unified_selector.registry import PROBLEMS, problem_to_pool_mask
    from .r69_frozen_evaluation import require_evaluation
    from .r74_model import INFERENCE_RULE, PROBE_NAMES

    launch = original_identity()
    require_terminal(read(ROOT / 'training_terminal_receipt.json'), 'r74_train', launch['slurm_step'])
    protocol, generation = read(ROOT / 'protocol.json'), read(ROOT / 'feature_cache_complete.json')
    complete, released = read(ROOT / 'training_complete.json'), read(ROOT / 'probes_released.json')
    history, result = read(RUN / 'history.json'), read(RUN / 'result.json')
    candidates, proposal = read(ROOT / 'validation_candidates.json'), read(RUN / 'best_checkpoint.json')
    if (launch['status'] != 'complete' or not generation['complete'] or generation['executed_solvers'] != 0
            or generation['completed_solver_labels_added'] != 0 or not released['before_optimizer_construction']
            or not released['no_live_probe_states'] or set(generation['probes']) != set(PROBE_NAMES)
            or protocol['inference_rule'] != INFERENCE_RULE or protocol['new_arms'] != ['B']
            or protocol['fixed_control']['native_correct'] != 8959
            or read(ROOT / 'source_launch/identities.json') != protocol['source_identities']
            or protocol['test_authorized'] or not protocol['parent_freeze_pending']):
        raise ValueError('Require original complete generation, released probes and unchanged R74 recipe')
    totals = generation['counters']
    expected = dict(base_query_rows=198000, train_rows=180000, val_rows=18000,
                    train_initializations=330000, val_initializations=33000,
                    trajectory_initializations=363000, completed_trajectories=0)
    if any(totals[key] != value for key, value in expected.items()):
        raise ValueError('Original full18 generation rows and all bounded counterfactual trajectories must complete')
    manifest = read(ROOT / 'data_manifest.json')
    for split, rows in (('train', 10000), ('val', 1000)):
        for problem in PROBLEMS:
            value = generation['per_task'][split+'/'+problem]
            multiplier = 1 if problem in ('TSP', 'CVRP', 'ATSP') else 2
            if (not value['complete'] or value['rows'] != rows or value['counters']['rows'] != rows
                    or value['counters']['trajectory_initializations'] != multiplier*rows
                    or sum(group['rows'] for group in value['groups']) != rows
                    or any(not group['complete'] for group in value['groups'])
                    or value['source_identity'] != manifest[split][problem]
                    or value['source_identity']['pool_ids'] != problem_to_pool_mask(problem)[0]
                    or value['labels_used_in_features']):
                raise ValueError('Generation must preserve every original task/row/full legal pool')
    expected_budget = (40, 9600, 7200000, 38400000)
    keys = ('epochs_completed', 'successful_updates', 'native_presentations', 'successful_leaf_view_rows')
    if (tuple(complete[key] for key in keys) != expected_budget
            or tuple(result[key] for key in keys) != expected_budget or len(history) != 40
            or [row['epoch'] for row in history] != list(range(1, 41))
            or [row['global_epoch'] for row in history] != list(range(41, 81))
            or result['final'] != history[-1] or result['inference_rule'] != INFERENCE_RULE):
        raise ValueError('Require all original40E/9600 updates/7.2M native rows/38.4M leaf-view rows')
    for row in history:
        require_evaluation(row['val'], 1000)
        if (not row['val']['hard_policy'] or not row['val']['fp32'] or row['val']['query_batches'] != 144
                or row['successful_updates'] != 240*row['epoch']
                or row['native_presentations'] != 180000*row['epoch']
                or row['successful_leaf_view_rows'] != 960000*row['epoch']
                or row['parameter_mode'] != 'averaged x'):
            raise ValueError('Only original full18 FP32 hard-policy validation may select B')
        if row['train'] is not None:
            require_evaluation(row['train'], 10000)
        if row['epoch'] % 5 == 0 and row['train'] is None:
            raise ValueError('Required actual full native TRAIN measurement is missing')
    selected = max(history, key=lambda row: (row['val']['native_correct'], -row['epoch']))
    identity = result['best_checkpoint']
    official = 'B' if selected['val']['native_correct'] > 8959 else 'A'
    if (identity != candidates['B'] or result['best']['epoch'] != selected['epoch']
            or result['best']['val'] != selected['val']
            or identity['epoch'] != selected['epoch'] or identity['global_epoch'] != selected['global_epoch']
            or identity['native_correct'] != selected['val']['native_correct'] or identity['native_total'] != 18000
            or any(proposal[key] != identity[key] for key in ('path', 'epoch', 'global_epoch', 'native_correct', 'native_total'))
            or result['proposed_official_arm'] != official or candidates['proposed_official_arm'] != official
            or complete['best_checkpoint'] != identity or complete['proposed_official_arm'] != official):
        raise ValueError('Saved unique whole-policy proposal must be the earliest maximum integer hard-VAL count')
    require_evaluation(result['best']['train'], 10000)
    return protocol, generation, history, result, identity, official


def copy_exclusive(source, destination):
    with Path(source).open('rb') as incoming, Path(destination).open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def check_state(checkpoint, protocol, identity):
    import torch
    from .r74_model import INFERENCE_RULE
    if (checkpoint['epoch'] != identity['epoch'] or checkpoint['global_epoch'] != identity['global_epoch']
            or checkpoint['native_correct'] != identity['native_correct'] or checkpoint['native_total'] != 18000
            or checkpoint['parameter_mode'] != 'averaged x' or checkpoint['checkpoint_mode'] != 'eval'
            or not checkpoint['complete_policy'] or not checkpoint['frozen_base_included']
            or checkpoint['inference_rule'] != INFERENCE_RULE
            or canonical(checkpoint['args']) != canonical(protocol)):
        raise ValueError('Require the complete selected averaged-x R74 policy with its original frozen base')
    state = checkpoint['model']
    if (not any(key.startswith('base.score_head.') for key in state)
            or not any(key.startswith('policy.root_gate.') for key in state)
            or not any(key.startswith('policy.second_gate.') for key in state)
            or any(not key.startswith(('base.', 'policy.')) for key in state)
            or not all(torch.isfinite(value).all() for value in state.values())):
        raise ValueError('Complete finite base and policy state required; no taskwise assembly or head transplant')
    for field in ('mean', 'std'):
        if not torch.equal(state['base.instance_encoder.geometry_residual.'+field],
                           torch.tensor(protocol['geometry_stats'][field], dtype=torch.float32)):
            raise ValueError('Original frozen TRAIN normalization buffers must be retained exactly')
    return state


def freeze():
    require_preparation()
    protocol, generation, history, result, identity, official = completed_training()
    if (ROOT / 'final_test_lock.json').exists() or (ROOT / 'final_test_started.json').exists():
        raise ValueError('Preserve the original unique lock/start; never reselect or restart TEST')
    import torch
    from .multitask_probe import file_hash
    from .r74_acquisition import recipe_records
    from .r74_model import PROBE_NAMES
    selected = Path(identity['path'])
    expected = RUN / f'best_checkpoints/epoch{identity["epoch"]:03d}.pt'
    if selected != expected:
        raise ValueError('Only the unique original whole-policy checkpoint may be frozen')
    checkpoint = torch.load(selected, map_location='cpu', weights_only=False)
    state = check_state(checkpoint, protocol, identity)
    assets = ROOT / 'locked_inference'
    assets.mkdir(exist_ok=False)
    model_path = assets / 'complete_policy.pt'
    copy_exclusive(selected, model_path)
    model_record = dict(identity, inference_path=str(model_path), inference_sha256=file_hash(model_path),
        model_params=protocol['model_params'], complete_policy=True, frozen_base_included=True,
        state_shapes={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in state.items()})
    del checkpoint, state
    probes, init, recipes = {}, read(ROOT / 'probe_initialization.json'), recipe_records()
    for name in PROBE_NAMES:
        record = init[name]
        if record != generation['probes'][name]:
            raise ValueError('The complete formal probe snapshots/configurations must remain unchanged')
        payload = torch.load(record['state'], map_location='cpu', weights_only=False)
        if (not payload['fp32'] or not payload['frozen'] or canonical(payload['recipe']) != canonical(record['recipe'])
                or any(not torch.isfinite(value).all() for value in payload['model'].values())
                or any(canonical(record['recipe'][key]) != canonical(value) for key, value in recipes[name].items())):
            raise ValueError('A formal probe state or its locked recipe differs; no substitute weights/config')
        destination = assets / (name+'.pt')
        copy_exclusive(record['state'], destination)
        probes[name] = dict(path=str(destination), sha256=file_hash(destination), recipe=record['recipe'],
            parameters=record['parameters'], state_shapes={key: list(value.shape) for key, value in payload['model'].items()})
        del payload
    lock = dict(locked_utc=now(), selection_split='original native full18 VAL only', official_arm=official,
        native_correct=identity['native_correct'], native_total=18000, selected_epoch=identity['epoch'],
        selected_global_epoch=identity['global_epoch'], earliest_exact_tie=True,
        rule=protocol['official_arm_rule'], inference_rule=protocol['inference_rule'], models={'B': model_record},
        probes=probes, full_legal_pools={task: value['pool_ids'] for task, value in read(ROOT / 'data_manifest.json')['train'].items()},
        source_identities=protocol['source_identities'], delivery_source_identities=delivery_hashes(),
        geometry_stats=protocol['geometry_stats'], complete_recipe=protocol, test_layout=LAYOUT,
        A=dict(name='R73B23/global63', validation_native_correct=8959, validation_native_total=18000,
               test_path=str(CONTROL / 'test_results.json'), test_json_pointer='/models/B',
               unchanged_saved_object_only=True, new_TEST_passes=0, original_TEST_passes_already_executed=1),
        allowed_new_TEST_passes={'A': 0, 'B': 1}, allowed_analysis_invocations=1,
        source_and_model_changes_after_lock=False, TEST_read_before_lock=False,
        final_deployment='F + selected incomplete acquisitions/gates + final utility + one full restarted chosen solver',
        completed_solver_executions_during_evaluation=0, parent_reviews_and_objective_audit_pending=True)
    exclusive(ROOT / 'final_test_lock.json', lock)
    exclusive(ROOT / 'freeze_receipt.json', dict(utc=now(), lock=str(ROOT / 'final_test_lock.json'),
        selected_epoch=identity['epoch'], selected_global_epoch=identity['global_epoch'],
        native_correct=identity['native_correct'], official_arm=official, no_validation_replay=True,
        no_A_weight_load=True, TEST_read_or_executed=False))
    return lock


def validate_lock():
    require_preparation()
    lock = read(ROOT / 'final_test_lock.json')
    protocol, _, _, _, identity, official = completed_training()
    if (lock['official_arm'] != official or lock['models']['B']['path'] != identity['path']
            or lock['models']['B']['native_correct'] != identity['native_correct']
            or canonical(lock['complete_recipe']) != canonical(protocol) or lock['test_layout'] != LAYOUT
            or lock['delivery_source_identities'] != delivery_hashes()
            or lock['inference_rule'] != protocol['inference_rule']):
        raise ValueError('The original whole-policy lock, official arm and exact inference recipe must remain unchanged')
    return protocol, lock


def assigned_idle():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03'):
        raise ValueError('Only existing1465/gpu03/assigned index1 UUID is authorized')
    while True:
        gpu = capture(['nvidia-smi', '--id='+GPU_UUID,
            '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'])
        apps = capture(['nvidia-smi', '--id='+GPU_UUID,
            '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'])
        rows = list(csv.reader(gpu['stdout'].splitlines()))
        if gpu['returncode'] or apps['returncode'] or len(rows) != 1:
            raise RuntimeError('Assigned UUID idle observation failed; preserve evidence without switching')
        index, uuid, name, used, total = [value.strip() for value in rows[0]]
        if index != '1' or uuid != GPU_UUID:
            raise ValueError('Assigned physical index/UUID mismatch')
        clients = [row for row in csv.reader(apps['stdout'].splitlines()) if row and row[0].strip() == GPU_UUID]
        observation = dict(utc=now(), gpu_query=gpu, assigned_client_query=apps, gpu_uuid=uuid,
            gpu_index=1, gpu_name=name, memory_used_mib=int(used), memory_total_mib=int(total),
            compute_apps=clients, idle=int(used)<500 and not clients, only_assigned_UUID_queried=True)
        append(ROOT / 'test_availability.jsonl', observation)
        if observation['idle']:
            launch = dict(observation, host=socket.gethostname(), evaluator_pid=os.getpid(),
                slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                tmux_session='r74_final_test', tmux_pane=os.environ['R74_TEST_TMUX_PANE_ID'],
                tmux_pane_pid=os.environ['R74_TEST_TMUX_PANE_PID'],
                command='bash code/V4/run_v4_r74_delivery.sh test')
            exclusive(ROOT / 'test_launch_receipt.json', launch)
            return launch
        print('[R74 TEST wait] assigned UUID occupied; waiting30s on the same device', flush=True)
        time.sleep(30)


def input_only_native_functions():
    """Reuse the native input construction/collation, removing only label reads/outputs."""
    import ast
    import copy
    from code.unified_selector import data
    tree = ast.parse(Path(data.__file__).read_text(), filename=data.__file__)
    dataset = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'UnifiedProblemDataset')
    item = copy.deepcopy(next(node for node in dataset.body if isinstance(node, ast.FunctionDef) and node.name == '__getitem__'))
    collate = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'collate_single_problem'))
    for function, names in ((item, {'lbl', 'costs', 'ind'}), (collate, {'costs', 'ind'})):
        removed = [node for node in function.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
        if {node.targets[0].id for node in removed} != names or len(removed) != len(names):
            raise ValueError('Original native input facade changed; do not improvise preprocessing')
        function.body = [node for node in function.body if node not in removed]
    class WithoutTargets(ast.NodeTransformer):
        def visit_Dict(self, node):
            pairs = [(key, value) for key, value in zip(node.keys, node.values)
                     if not (isinstance(key, ast.Constant) and key.value in ('costs', 'ind'))]
            node.keys, node.values = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            return self.generic_visit(node)
    functions = ast.Module(body=[item, collate], type_ignores=[])
    functions = ast.fix_missing_locations(WithoutTargets().visit(functions))
    if any(isinstance(node, ast.Attribute) and node.attr == 'labels' for node in ast.walk(functions)):
        raise ValueError('The input-only facade must have no supervision accessor')
    namespace = dict(vars(data))
    exec(compile(functions, data.__file__, 'exec'), namespace)
    return namespace['__getitem__'], namespace['collate_single_problem']


class OriginalTestInputs:
    def __init__(self, problem, functions):
        import pickle
        from code.unified_selector.data import DATA_ROOT
        from code.unified_selector.registry import P2I, constraint_bits, problem_descriptor, problem_to_pool_mask
        self.problem, self.split, self.coord_augment = problem, 'test', 0
        with (DATA_ROOT / (problem+'test') / 'dataset.pkl').open('rb') as stream:
            self.instances = pickle.load(stream)
        self.pool_order, self.mask = problem_to_pool_mask(problem)
        self.cbits, self.problem_desc = constraint_bits(problem), problem_descriptor(problem)
        self.pid, self.K_p = P2I[problem], len(self.pool_order)
        self.base_N = self.N = len(self.instances)
        self.item, self.collate = functions
        if self.N != 1000:
            raise ValueError('Original full native TEST must contain1000 rows per task')

    def gather(self, rows, device):
        import torch
        batch = self.collate([self.item(self, int(row)) for row in rows])
        if set(batch) & {'costs', 'ind', 'winner', 'labels'}:
            raise ValueError('Actual costs/winners are excluded from input-derived prediction')
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def locked_probes(lock, device):
    import torch
    from .multitask_probe import file_hash
    from .r74_acquisition import BQ_ROOT, BRIDGE, EASY_ROOT, HOME, RELD_ROOT, ProbeRecipes, Trajectory, load_source, source_namespace
    from .r74_model import PROBE_NAMES

    class LockedProbes(ProbeRecipes):
        def __init__(self):
            super().__init__(device, {}, lambda *_: None)
            self.recipes = {name: lock['probes'][name]['recipe'] for name in PROBE_NAMES}
            self.loading_seconds, self.loads = 0., []

        def ensure_probe(self, problem, probe):
            if probe in self.models:
                return
            started = time.monotonic()
            name, record = PROBE_NAMES[probe], lock['probes'][PROBE_NAMES[probe]]
            if file_hash(Path(record['path'])) != record['sha256']:
                raise ValueError('A uniquely locked formal probe snapshot changed')
            payload = torch.load(record['path'], map_location='cpu', weights_only=False)
            if canonical(payload['recipe']) != canonical(record['recipe']) or not payload['fp32'] or not payload['frozen']:
                raise ValueError('Locked formal probe metadata differs; no alternate checkpoint/default')
            params = payload['recipe']['params']
            if probe == 0:
                with source_namespace(BQ_ROOT, ('model', 'utils', 'learning')):
                    model = importlib.import_module('model.model').BQModel(**params)
                    self.helpers[0] = load_source('r74_locked_bq_decoding', BQ_ROOT / 'learning/tsp/decoding.py')
            elif probe == 1:
                with source_namespace(RELD_ROOT / 'CVRP', ('utils', 'CVRPModel', 'CVRPEnv')):
                    model = importlib.import_module('CVRPModel').CVRPModel(**params)
                    self.envs[1] = importlib.import_module('CVRPEnv').CVRPEnv
                    self.helpers[1] = load_source('r74_locked_cvrp_bridge', BRIDGE / 'run_reld_cvrp_single_bridge.py')
            elif probe == 2:
                from .r57_probe import torchrl_compat
                torchrl_compat()
                with source_namespace(HOME, ()):
                    from EasyNCO.neural_solvers.methods.matnet.policy import ATSPPolicy
                    from EasyNCO.neural_solvers.envs.ATSPEnv import ATSPEnv
                    model = ATSPPolicy(**params)
                    self.envs[2] = ATSPEnv
                model.set_decoder_strategy('greedy')
            else:
                from code.unified_selector.registry import PROBLEMS
                from .r41_label_stability import relocated_reld_module
                with source_namespace(RELD_ROOT / 'Multi-Task', ('utils', 'models', 'envs')):
                    bridge = relocated_reld_module(BRIDGE / 'reld_bridge_common.py')
                    utils, _, _ = bridge.import_reld_modules()
                    model = utils.get_model('MTL' if probe == 3 else 'MOE_LIGHT')(**dict(params, device=torch.device(device)))
                    self.helpers[probe] = bridge
                    self.envs.update({task: utils.get_env(task)[0] for task in PROBLEMS[3:]})
            model.load_state_dict(payload['model'], strict=True)
            self.models[probe] = model.to(device).float().requires_grad_(False).eval()
            torch.cuda.synchronize()
            seconds = time.monotonic()-started
            self.loading_seconds += seconds
            actual = sum(parameter.numel() for parameter in model.parameters())
            if actual != record['parameters']:
                raise ValueError('Strictly restored probe parameter count differs from formal generation')
            receipt = dict(utc=now(), probe=name, parameters=actual, seconds=seconds, checkpoint=record['path'],
                strict_restore=True, original_mutable_weight_pointer_loaded=False, FP32=True, eval=True)
            self.loads.append(receipt)
            append(ROOT / 'test_probe_loading.jsonl', receipt)
            del payload

        def start(self, problem, instances, rows, probe):
            self.ensure_probe(problem, probe)
            return Trajectory(self, problem, instances, rows, probe)

    return LockedProbes()


class LiveObservations:
    """One current query; native context survives from shallow to deep, without slicing."""
    def __init__(self, probes, problem, instances, row, nodes, n, starts):
        self.probes, self.problem, self.instances, self.row = probes, problem, instances, row
        self.nodes, self.n, self.starts = nodes, n, starts
        self.trajectories, self.cache, self.requests, self.receipts = {}, {}, [], []
        self.acquisition_seconds = 0.

    def __call__(self, probe, checkpoint, positions=None):
        import torch
        if positions is not None and positions.detach().cpu().tolist() != [0]:
            raise ValueError('Locked live TEST provider has exactly one current query, without state slicing')
        key = (probe, checkpoint)
        if key in self.cache:
            return self.cache[key]
        if checkpoint not in (0, 1) or (checkpoint == 1 and (probe, 0) not in self.cache):
            raise ValueError('No future/deep observation may be acquired before its current shallow prefix')
        self.probes.ensure_probe(self.problem, probe)
        torch.cuda.synchronize()
        started = time.monotonic()
        if probe not in self.trajectories:
            if checkpoint != 0:
                raise ValueError('A deep request must continue its actual retained native trajectory')
            self.trajectories[probe] = self.probes.start(self.problem, self.instances, [self.row], probe)
        trajectory = self.trajectories[probe]
        if trajectory.n != self.n or trajectory.starts != self.starts or trajectory.node_count != self.nodes.shape[1]:
            raise ValueError('Actual native true-size/start/node roles differ from the original current input')
        value = trajectory.acquire_increment(self.nodes)
        start = self.nodes.gather(1, value['start'][..., None].expand(-1, -1, 128))
        current = self.nodes.gather(1, value['current'][..., None].expand(-1, -1, 128))
        identity = self.nodes.new_zeros(1, self.starts, 7)
        identity[..., probe], identity[..., 5+checkpoint] = 1., 1.
        tokens = torch.cat((start, current, value['visited_mean'], value['scalars'], identity), -1).float()
        if tokens.shape != (1, self.starts, 403) or not torch.isfinite(tokens).all():
            raise ValueError('Every actual per-start live token must match the frozen403-dimensional protocol')
        self.cache[key] = tokens, torch.ones(1, self.starts, dtype=torch.bool, device=tokens.device)
        self.requests.append(key)
        torch.cuda.synchronize()
        seconds = time.monotonic()-started
        self.acquisition_seconds += seconds
        self.receipts.append(dict(value['receipt'], checkpoint=checkpoint, seconds=seconds))
        return self.cache[key]

    def summary(self):
        counts = Counter(probe_initializations=len(self.trajectories), per_start_states=0,
            native_model_action_calls=0, native_decoder_calls=0, per_start_selections=0,
            incremental_selection_calls=0, per_start_incremental_selections=0, probe_encoder_calls=0,
            observed_checkpoints=len(self.cache), completed_probe_trajectories=0)
        for probe, trajectory in self.trajectories.items():
            counts.update(per_start_states=trajectory.starts,
                native_model_action_calls=trajectory.model_calls, native_decoder_calls=trajectory.decoder_calls,
                per_start_selections=trajectory.selections*trajectory.starts,
                incremental_selection_calls=trajectory.network_steps,
                per_start_incremental_selections=trajectory.network_steps*trajectory.starts,
                probe_encoder_calls=trajectory.model_calls if probe == 0 else 1)
        return dict(counters=dict(counts), requests=[list(key) for key in self.requests],
                    observations=self.receipts, acquisition_seconds=self.acquisition_seconds)


def final_test():
    protocol, lock = validate_lock()
    if (ROOT / 'final_test_started.json').exists():
        raise ValueError('Original exclusive TEST start exists; no automatic retry')
    launch = assigned_idle()
    exclusive(ROOT / 'final_test_started.json', dict(utc=now(), lock=lock, launch=launch,
        allowed_attempts=1, new_TEST_passes={'A': 0, 'B': 1}, automatic_retry=False))
    whole_started = time.monotonic()
    import numpy as np
    import torch
    from code.unified_selector.registry import P2I, PROBLEMS
    from .multitask_probe import file_hash
    from .performance_targets import read_raw_costs
    from .r40_experiment import aggregate, families
    from .r43_experiment import policy_metrics
    from .r69_frozen_evaluation import require_evaluation
    from .r74_data import geometry_of_instance, probe_ids
    from .r74_experiment import memory
    from .r74_model import R74Model
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    base_loading_started = time.monotonic()
    record = lock['models']['B']
    if file_hash(Path(record['inference_path'])) != record['inference_sha256']:
        raise ValueError('The unique complete locked inference checkpoint changed')
    checkpoint = torch.load(record['inference_path'], map_location='cpu', weights_only=False)
    state = check_state(checkpoint, protocol, record)
    model = R74Model(record['model_params'])
    model.load_state_dict(state, strict=True)
    model.to('cuda:0').float().requires_grad_(False).eval()
    del state, checkpoint
    torch.cuda.synchronize()
    base_loading_seconds = time.monotonic()-base_loading_started
    counts = read(ROOT / 'initialization.json')
    if (sum(p.numel() for p in model.parameters()) != counts['full_model_parameters']
            or any(p.requires_grad for p in model.parameters())):
        raise ValueError('Only the complete frozen selected R74 policy may perform TEST')
    probes = locked_probes(lock, 'cuda:0')
    functions = input_only_native_functions()
    counters = Counter(query_rows=0, base_forward_batches=0, base_forward_rows=0, policy_queries=0,
                       tails104=0, completed_probe_trajectories=0, completed_solver_executions=0,
                       new_final_solver_labels=0)
    predictions, paths, results, batch_timings, per_task_usage = {}, Counter(), {}, [], {}
    totals = Counter(base_forward_seconds=0., acquisition_seconds=0., policy_exclusive_seconds=0., input_loading_seconds=0.)
    directory = ROOT / 'test_predictions'
    directory.mkdir(exist_ok=False)
    import wandb
    wb = wandb.init(**protocol['wandb'], name='R74_ONE_locked_live_hard_policy_TEST', dir=str(ROOT),
        id='r74-test-'+launch['slurm_step'].replace('.', '-'), config=dict(lock=lock, new_tests={'A': 0, 'B': 1}))
    evaluation_started = time.monotonic()
    try:
        with torch.no_grad():
            for problem in PROBLEMS:
                loading = time.monotonic()
                inputs = OriginalTestInputs(problem, functions)
                totals['input_loading_seconds'] += time.monotonic()-loading
                if inputs.pool_order != lock['full_legal_pools'][problem]:
                    raise ValueError('Retain the entire original legal candidate pool and its cost-column identities')
                scores, predicted, true_nodes, hard_paths, query_records = [], [], [], [], []
                usage = Counter()
                for start in range(0, 1000, 128):
                    stop = min(start+128, 1000)
                    torch.cuda.synchronize()
                    batch_started = time.monotonic()
                    batch = inputs.gather(range(start, stop), 'cuda:0')
                    torch.cuda.synchronize()
                    before = time.monotonic()
                    base = model.extract_base(batch)
                    torch.cuda.synchronize()
                    base_seconds = time.monotonic()-before
                    totals['base_forward_seconds'] += base_seconds
                    counters.update(base_forward_batches=1, base_forward_rows=stop-start, tails104=int(stop-start == 104))
                    batch_acquisition, batch_policy, cold_before = 0., 0., probes.loading_seconds
                    for offset, row in enumerate(range(start, stop)):
                        n, starts, _ = geometry_of_instance(problem, inputs.instances[row])
                        node_count = n+int(problem not in ('TSP', 'ATSP'))
                        if int(base['node_mask'][offset].sum()) != node_count:
                            raise ValueError('Original physical node roles and true-node count must be retained')
                        provider = LiveObservations(probes, problem, inputs.instances, row,
                            base['nodes'][offset:offset+1, :node_count], n, starts)
                        query = {key: base[key][offset:offset+1] for key in ('features', 'h', 'base_logits', 'solver_mask')}
                        query.update(problem_id=P2I[problem], mvrp=problem not in ('TSP', 'CVRP', 'ATSP'),
                            probe_id=probe_ids(problem)[0], pool_ids=batch['pool_ids'], observations=provider)
                        torch.cuda.synchronize()
                        before, prior_loading = time.monotonic(), probes.loading_seconds
                        output = model(query)
                        torch.cuda.synchronize()
                        policy_wall = time.monotonic()-before
                        observation = provider.summary()
                        acquisition = observation['acquisition_seconds']
                        cold = probes.loading_seconds-prior_loading
                        exclusive_policy = max(0., policy_wall-acquisition-cold)
                        path = int(output['hard_paths'][0])
                        expected_keys = ([(probe_ids(problem)[0], 0), (probe_ids(problem)[0], 1)]
                            if not query['mvrp'] else [[(3, 0), (3, 1)], [(4, 0), (4, 1)], [(3, 0), (4, 0)]][path])
                        if set(provider.requests) != set(expected_keys) or len(provider.requests) != 2:
                            raise ValueError('Only the hard chosen acquisition path may execute; never acquire counterfactual TEST states')
                        logits = output['logits'].cpu().numpy()[0]
                        if logits.dtype != np.float32 or not output['solver_mask'].all() or not np.isfinite(logits).all():
                            raise ValueError('Every original legal utility must be finite FP32')
                        local, global_id = int(output['pred'][0]), int(output['pred_global_id'][0])
                        scores.append(logits)
                        predicted.append(local)
                        true_nodes.append(n)
                        hard_paths.append(path)
                        paths[problem+'/'+str(path)] += 1
                        usage.update(observation['counters'])
                        counters.update(observation['counters'])
                        counters.update(query_rows=1, policy_queries=1)
                        totals.update(acquisition_seconds=acquisition, policy_exclusive_seconds=exclusive_policy)
                        batch_acquisition += acquisition
                        batch_policy += exclusive_policy
                        row_record = dict(utc=now(), problem=problem, source_index=row, true_n=n, starts=starts,
                            pool_ids=inputs.pool_order, logits=logits.tolist(), pred=local, pred_global_id=global_id,
                            hard_path=path, gate_probabilities={key: float(value[0]) for key, value in output['gate_probabilities'].items()},
                            policy_wall_seconds=policy_wall, policy_exclusive_seconds=exclusive_policy,
                            probe_loading_seconds=cold, **observation)
                        append(ROOT / 'test_queries.jsonl', row_record)
                        query_records.append(row_record)
                        del provider, query, output
                    torch.cuda.synchronize()
                    timing = dict(utc=now(), problem=problem, first_source=start, rows=stop-start,
                        base_seconds=base_seconds, acquisition_seconds=batch_acquisition,
                        policy_exclusive_seconds=batch_policy, probe_loading_seconds=probes.loading_seconds-cold_before,
                        whole_batch_seconds=time.monotonic()-batch_started, counters=dict(counters), **memory('cuda:0'))
                    batch_timings.append(timing)
                    append(ROOT / 'test_batches.jsonl', timing)
                    write(ROOT / 'test_current_status.json', dict(stage='live_selected_acquisition_TEST', **timing))
                    if counters['base_forward_batches'] == 1:
                        exclusive(ROOT / 'first_test_batch.json', dict(timing, launch=launch, actual_layout=LAYOUT,
                            finite=True, measured_partial_solver_runs=True, completed_solver_executions=0))
                    print('[R74 live TEST] '+problem+' rows='+str(counters['query_rows'])+'/18000', flush=True)
                    del batch, base
                predictions[problem] = dict(logits=np.asarray(scores, dtype=np.float32), pred=np.asarray(predicted),
                    true_nodes=np.asarray(true_nodes), hard_paths=np.asarray(hard_paths), records=query_records,
                    pool_ids=np.asarray(inputs.pool_order), source_indices=np.arange(1000))
                per_task_usage[problem] = dict(usage)
                del inputs
            # Supervision is opened only after every input-derived prediction has been recorded.
            manifest = {}
            for problem in PROBLEMS:
                raw, saved = read_raw_costs(problem, 'test'), predictions[problem]
                if (len(raw['winner']) != 1000 or raw['costs'].dtype != np.float64
                        or raw['pool_ids'] != lock['full_legal_pools'][problem]):
                    raise ValueError('Detached report scoring must use original18x1000 native winners/raw FP64 costs/full pools')
                metrics, pred = policy_metrics(saved['logits'], raw)
                if not np.array_equal(pred, saved['pred']):
                    raise ValueError('Detached scoring must preserve the original global-ID utility decision')
                metrics['native_correct'] = int(np.count_nonzero(pred == raw['winner']))
                results[problem] = metrics
                records = saved.pop('records')
                np.savez_compressed(directory / (problem+'.npz'), **saved, winner=raw['winner'], costs=raw['costs'],
                    pool=np.asarray(raw['pool']), pred_global_id=np.asarray(raw['pool_ids'])[pred],
                    acquisition_seconds=np.asarray([row['acquisition_seconds'] for row in records]),
                    policy_seconds=np.asarray([row['policy_exclusive_seconds'] for row in records]),
                    probe_initializations=np.asarray([row['counters']['probe_initializations'] for row in records]),
                    per_start_states=np.asarray([row['counters']['per_start_states'] for row in records]),
                    native_decoder_calls=np.asarray([row['counters']['native_decoder_calls'] for row in records]),
                    per_start_selections=np.asarray([row['counters']['per_start_selections'] for row in records]))
                manifest[problem] = dict(rows=1000, pool=raw['pool'], pool_ids=raw['pool_ids'],
                    data_hash=raw['data_hash'], label_hash=raw['label_hash'], raw_cost_dtype='float64',
                    original_nodes_order_roles_and_ATSP_direction=True, labels_opened_after_all_predictions=True)
            if (counters['query_rows'] != 18000 or counters['policy_queries'] != 18000
                    or counters['base_forward_batches'] != 144 or counters['base_forward_rows'] != 18000
                    or counters['tails104'] != 18 or counters['completed_probe_trajectories'] != 0
                    or counters['completed_solver_executions'] != 0 or counters['new_final_solver_labels'] != 0):
                raise ValueError('The one live full18 TEST must preserve exact original rows/base batches and noncompletion')
            correct = sum(value['native_correct'] for value in results.values())
            macro = aggregate(results)
            macro['macro_top1'] = correct/18000
            B = dict(per_problem=results, macro=macro, families=families(results), native_correct=correct,
                native_total=18000, seconds=time.monotonic()-evaluation_started, timings=dict(totals),
                base_model_loading_seconds=base_loading_seconds, probe_loading_seconds=probes.loading_seconds,
                probe_loads=probes.loads, counters=dict(counters), hard_path_counts=dict(paths),
                per_task_acquisition_counters=per_task_usage, parameter_counts=counts, trainable_parameters=0,
                actual_test_layout=LAYOUT, query_batches=144, ordinary_batch_seconds=[row['whole_batch_seconds'] for row in batch_timings],
                **memory('cuda:0'), timing_scope='Whole live TEST including base batches, selected native prefixes, '
                'policy, loading, input preparation and detached report scoring; not singleton/completed-solver latency',
                partial_solver_runs=True, completed_solver_executions=0, new_final_solver_labels=0,
                inference_rule=lock['inference_rule'], hard_policy=True, FP32=True)
            require_evaluation(B, 1000)
            source = CONTROL / 'test_results.json'
            historical = read(source)
            A = historical['models']['B']
            require_evaluation(A, 1000)
            if not historical['complete']:
                raise ValueError('Only the previously completed original R73B TEST object may be reused')
            provenance = dict(path=str(source), json_pointer='/models/B', file_sha256=file_hash(source),
                object_reused_unchanged=True, new_A_TEST_passes=0, original_A_TEST_passes_already_executed=1,
                no_weight_load_or_forward_or_recount=True, historical_TEST_exposure_disclosed=True)
            write(ROOT / 'test_data_manifest.json', manifest)
            exclusive(ROOT / 'control_test_reuse.json', provenance)
            exclusive(ROOT / 'test_results.json', dict(complete=True, completed_utc=now(), models={'A': A, 'B': B},
                official_arm=lock['official_arm'], official_arm_fixed_before_TEST=True, lock=lock,
                control_provenance=provenance, new_TEST_passes={'A': 0, 'B': 1},
                wall_seconds=time.monotonic()-whole_started, counterfactual_TEST_cache=False,
                goal_status='ACTIVE; parent whole-objective audit pending'))
            exclusive(ROOT / 'test_complete.json', dict(utc=now(), slurm_step=launch['slurm_step'],
                native_correct=correct, native_total=18000, new_TEST_passes={'A': 0, 'B': 1},
                counters=dict(counters), process_success=True, scheduler_terminal_still_required=True))
            wb.log({'test/native_correct': correct, **{'test/'+key: value for key, value in macro.items()},
                    **{'acquisition/'+key: value for key, value in counters.items()}})
    finally:
        wb.finish()


def completed_test():
    from .r69_frozen_evaluation import require_evaluation
    lock, launch = read(ROOT / 'final_test_lock.json'), read(ROOT / 'test_launch_receipt.json')
    receipt = read(ROOT / 'test_terminal_receipt.json')
    require_terminal(receipt, 'r74_final_test', launch['slurm_step'])
    test, complete = read(ROOT / 'test_results.json'), read(ROOT / 'test_complete.json')
    if (not test['complete'] or test['lock'] != lock or test['official_arm'] != lock['official_arm']
            or test['new_TEST_passes'] != {'A': 0, 'B': 1} or complete['new_TEST_passes'] != {'A': 0, 'B': 1}
            or test['counterfactual_TEST_cache'] or launch['gpu_uuid'] != GPU_UUID
            or launch['gpu_index'] != 1 or not launch['slurm_step'].startswith('1465.')
            or launch['slurm_step'] == '1465.111'):
        raise ValueError('Require the ONE original locked B TEST, independent terminal0, and unchanged official arm')
    for arm in ('A', 'B'):
        require_evaluation(test['models'][arm], 1000)
    if test['models']['A'] != read(CONTROL / 'test_results.json')['models']['B']:
        raise ValueError('Fixed A historical TEST object must remain unchanged; no recount/reference forward')
    B = test['models']['B']
    if (B['actual_test_layout'] != LAYOUT or B['counters'] != complete['counters']
            or any(B['counters'][key] != expected for key, expected in dict(query_rows=18000,
                base_forward_batches=144, base_forward_rows=18000, policy_queries=18000, tails104=18,
                completed_probe_trajectories=0, completed_solver_executions=0, new_final_solver_labels=0).items())):
        raise ValueError('Actual live TEST counters do not match the frozen original full18 execution layout')
    return test


def run_test():
    exclusive(ROOT / 'test_dispatch_started.json', dict(utc=now(), attempts=1, session='r74_final_test', automatic_retry=False))
    command = 'bash code/V4/run_v4_r74_delivery.sh test >> code/V4/runs/R74_adaptive_incomplete_acquisition/test_bootstrap.log 2>&1'
    dispatch = capture(['tmux', 'new-session', '-d', '-s', 'r74_final_test', '-c', str(WORKSPACE), command])
    exclusive(ROOT / 'test_tmux_dispatch.json', dispatch)
    if dispatch['returncode'] != 0:
        raise RuntimeError('ONE TEST dispatch failed; preserve existing session/marker without retry')
    while not (ROOT / 'test_launch_receipt.json').exists():
        owned = capture(['tmux', 'has-session', '-t', '=r74_final_test'])
        if absent(owned):
            raise RuntimeError('Original TEST launcher exited before assigned-UUID launch; preserve bootstrap log, no retry')
        time.sleep(30)
    step = read(ROOT / 'test_launch_receipt.json')['slurm_step']
    if not step.startswith('1465.') or step == '1465.111':
        raise ValueError('Require one actual new TEST step in the same original allocation')
    while True:
        owned = capture(['tmux', 'has-session', '-t', '=r74_final_test'])
        if absent(owned):
            actual = terminal(step, 'r74_final_test')
            append(ROOT / 'delivery_test_terminal_observations.jsonl', actual)
            if actual['state'].startswith(FAILED) or (actual['state'] == 'COMPLETED' and actual['exit_code'] != '0:0'):
                exclusive(ROOT / 'test_terminal_receipt.json', actual)
                raise RuntimeError('Original ONE TEST failed; preserve all partial artifacts and do not retry')
            if actual['state'] == 'COMPLETED':
                require_terminal(actual, 'r74_final_test', step)
                exclusive(ROOT / 'test_terminal_receipt.json', actual)
                completed_test()
                return actual
        elif owned['returncode'] != 0:
            raise RuntimeError('Cannot establish owned TEST tmux state; preserve evidence')
        time.sleep(30)


def controller():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('R74_CONDITIONAL_DELIVERY') != 'authorized':
        raise ValueError('Independent conditional controller must remain CPU-only and parent-authorized')
    require_preparation()
    pane = os.environ.get('TMUX_PANE', '')
    identity = capture(['tmux', 'display-message', '-p', '-t', pane, '#{session_name}|#{pane_id}|#{pane_pid}'])
    if identity['returncode'] != 0 or not identity['stdout'].startswith('r74_delivery_controller|'):
        raise ValueError('Only owned independent tmux r74_delivery_controller is authorized')
    exclusive(ROOT / 'delivery_started.json', dict(utc=now(), pid=os.getpid(), host=socket.gethostname(),
        tmux='r74_delivery_controller', pane=pane, pane_pid=int(identity['stdout'].strip().split('|')[2]),
        tmux_identity=identity, original_formal_process=original_identity(),
        command='bash code/V4/run_v4_r74_delivery.sh controller',
        controller='python -B -u -m code.V4.r74_delivery --stage controller', CPU_only=True,
        monitor_interval_seconds=240, future_execution='UNVERIFIED', automatic_retry=False,
        conditional_authorization_is_not_a_VAL_freeze_or_TEST=True))
    training = wait_original()
    lock = freeze()
    test = run_test()
    if (ROOT / 'analysis_started.json').exists():
        raise RuntimeError('Preserve original ONCE analysis marker; never regenerate reports')
    analysis = capture([sys.executable, '-B', '-u', '-m', 'code.V4.r74_analysis', '--authorize-analysis'])
    exclusive(ROOT / 'analysis_invocation_receipt.json', analysis)
    if analysis['returncode'] != 0:
        raise RuntimeError('ONCE CPU analysis failed; preserve invocation/partial reports without retry')
    receipt = read(ROOT / 'analysis_receipt.json')
    if not receipt['complete'] or receipt['invocation'] != 1:
        raise ValueError('Require the actual single completed CPU analysis')
    exclusive(ROOT / 'delivery_receipt.json', dict(completed_utc=now(), complete=True,
        original_training_terminal=training, new_B_TEST_terminal=test, unique_VAL_lock=str(ROOT / 'final_test_lock.json'),
        selected_epoch=lock['selected_epoch'], selected_global_epoch=lock['selected_global_epoch'],
        validation_native_correct=lock['native_correct'], validation_native_total=18000,
        official_arm=lock['official_arm'], official_arm_fixed_before_TEST=True, new_TEST_passes={'A': 0, 'B': 1},
        analysis=receipt, analysis_command=analysis, owned_training_tmux_gone=training['tmux_terminal'],
        owned_test_tmux_gone=test['tmux_terminal'], owned_TEST_and_analysis_work_terminated=True,
        controller_current_process_will_exit_after_receipt=True, no_Git_or_review_execution=True,
        goal_status='ACTIVE; parent whole-objective audit pending',
        parent_obligations='Actual primary/fresh reviews, persistent memory, objective audit, one completed-version commit/push'))
    write(ROOT / 'delivery_current_status.json', dict(utc=now(), stage='delivery_complete_parent_review_pending'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('controller', 'test'), required=True)
    args = parser.parse_args()
    try:
        controller() if args.stage == 'controller' else final_test()
    except BaseException as error:
        failure = dict(utc=now(), stage=args.stage, pid=os.getpid(), error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(), automatic_retry=False,
            slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''))
        path = ROOT / ('delivery_failure.json' if args.stage == 'controller' else 'test_failure.json')
        if not path.exists():
            exclusive(path, failure)
        append(ROOT / 'delivery_failures.jsonl', failure)
        write(ROOT / 'delivery_current_status.json', dict(failure, status='failed_preserved_no_retry'))
        raise


if __name__ == '__main__':
    main()
