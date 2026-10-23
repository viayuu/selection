"""R106 online native adaptation through a process-private original R84 engine."""

import argparse
import copy
import fcntl
import importlib
import importlib.util
import json
import sys
import time
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .pairwise_objective import inputs_only
from .r76_experiment import RDropObjective, file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import selection_counts, zero_summary


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R106_native_encoder_adaptation'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
ARMS = NEURAL_ARMS = ('A_frozen', 'B_finetune')
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES, EVAL_EVERY = 1024, 57600, 4320, 1440
TRAIN_EVERY, PEAK_LR, END_LR, NATIVE_LR_SCALE = 7200, 1e-4, 1e-6, .1
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
CORE_PARAMETER_COUNT = 1941257
NATIVE_PARAMETER_COUNT, TOTAL_PARAMETER_COUNT = 10023840, 11965097
PARAMETER_COUNTS = dict(original=1842953, core=CORE_PARAMETER_COUNT, native=NATIVE_PARAMETER_COUNT,
    total=TOTAL_PARAMETER_COUNT, A_frozen=CORE_PARAMETER_COUNT, B_finetune=TOTAL_PARAMETER_COUNT)
ORIGINAL_FRESH_SHA256 = 'dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93'
MODEL_FACTORY = 'code.V4.r106_experiment.load_checkpoint'
INITIALIZATION = 'fresh_r84_seed2_plus_r102_projections_and_pinned_r102b_native'
MODEL_ROLE_ARM, SELECTION_HEAD = 'B_execution_anchor', 'original_score_mlp_maximin'
CHECKPOINT_RULE = 'strict minimum whole18 VAL macro_vs_sbs_pct; first exact tie; common full-TRAIN SBS'
INFERENCE_RULE = ('eval online native ELG/MatNet/ReLD plus original full18 selector; native FP32 '
    'until normal fusion; physical TSP node0 slot3, instance4/joint2, methodID+hand34, '
    'score MLP/utility differences/maximin; ascending-global-solver-ID exact ties; '
    'no cached native features, calibration, quotas or per-problem assembly')
VAL_BINDING_RULE = 'exact_selected_checkpoint_model_state_protocol_state_and_successful_update'
PAIR_CONTEXT_RULE = 'one_input_only_native_graph_per_microbatch_shared_by_original_two_dropout_forwards'
WINNER_RULE = engine.WINNER_RULE
learning_rate, consistency_weight = engine.learning_rate, engine.consistency_weight
task_order, json_hash, status_update = engine.task_order, engine.json_hash, engine.status_update
DIMENSIONS = dict(embedding_dim=128, encoder_layer_num=4, joint_layer_num=2,
    head_num=4, qkv_dim=32, ff_hidden_dim=512, head_hidden_dim=256, solver_feature_hidden=128)


def model_module():
    module = importlib.import_module('code.V4.r106_model')
    if tuple(module.ARMS) != ARMS:
        raise ValueError('R106 model/runner arm interface differs')
    return module


def micro_caps(gpu_type='4090', large_cap=None, *, atsp_cap=128, other_cap=256):
    large_cap = 64 if large_cap is None else large_cap
    if (gpu_type not in ('3090', '4090') or any(type(v) is not int or not 0 < v <= BATCH_SIZE
            for v in (large_cap, atsp_cap, other_cap))):
        raise ValueError('R106 requires positive main-locked microcaps, without automatic shrink')
    return {p: large_cap if p in ('TSP', 'CVRP') else atsp_cap if p == 'ATSP' else other_cap
            for p in PROBLEMS}


def required_lock_fields(large_micro_cap=64, atsp_micro_cap=128, other_micro_cap=256,
                         eval_batch_size=128):
    """Main owns this lock; call with the caps from its one hardware witness."""
    micro_caps(large_cap=large_micro_cap, atsp_cap=atsp_micro_cap, other_cap=other_micro_cap)
    if type(eval_batch_size) is not int or not 0 < eval_batch_size <= BATCH_SIZE:
        raise ValueError('Invalid main-locked evaluation cap')
    return dict(experiment='R106', locked=True, arms=list(ARMS), seed=2, task_scope='full18',
        rows_per_problem=dict(SPLIT_ROWS), batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        train_eval_every=TRAIN_EVERY, large_micro_cap=large_micro_cap,
        atsp_micro_cap=atsp_micro_cap, other_micro_cap=other_micro_cap, eval_batch_size=eval_batch_size,
        model_role_arm=MODEL_ROLE_ARM, role_feature_slot=3, embedding_dim=128,
        encoder_layer_num=4, joint_layer_num=2, solver_features=34,
        selector_initialization='fresh_R84_seed2', native_initialization='R102B_original_native_pretrained',
        identical_complete_fresh_parameters=True, original_fresh_trainable_sha256=ORIGINAL_FRESH_SHA256,
        native_execution='online_all_splits', native_feature_cache=False, native_eval=True,
        native_pair_context=PAIR_CONTEXT_RULE, native_lr_scale=NATIVE_LR_SCALE,
        trainable_parameter_matched=False, equal_training_compute_claim=False,
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        external_pretraining=True, new_targets=False, auxiliary_loss=False, optimizer='AdamW',
        peak_lr=PEAK_LR, end_lr=END_LR, weight_decay=1e-4, grad_clip=1., dropout=.1,
        loss=dict(CE=.35, comparison=.35, relative_risk=.02, RDrop_final=.35),
        checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE, R58_operations=False)


def _execution(args):
    return {k: getattr(args, k) for k in
            ('large_micro_cap', 'atsp_micro_cap', 'other_micro_cap', 'eval_batch_size')}


def protocol_lock(args, *, execution=None):
    path = Path(args.protocol_lock or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    expected = required_lock_fields(**(_execution(args) if execution is None else execution))
    mismatches = [k for k, v in expected.items() if record.get(k) != v]
    if mismatches:
        raise ValueError('R106 main scientific lock differs: ' + ', '.join(mismatches))
    for name in ('data_root', 'prepared_root'):
        if Path(record[name]).resolve() != Path(getattr(args, name)).resolve():
            raise ValueError('R106 main lock names another ' + name)
    if Path(record['primary_cache']).resolve() != Path(args.cache_root or DEFAULT_CACHE_ROOT).resolve():
        raise ValueError('R106 must directly reuse the locked raw R84 cache in place')
    return dict(path=str(path), sha256=file_hash(path), record=record)


def prepared_config(args):
    protocol_lock(args)
    if args.cache_root is None:
        args.cache_root = DEFAULT_CACHE_ROOT
    return engine.prepared_config(args)


def parameter_counts(model, arm):
    if arm not in ARMS:
        raise ValueError('Unknown R106 arm')
    native = [p for p in model.parameters() if getattr(p, '_r106_native', False)]
    core = [p for p in model.parameters() if not getattr(p, '_r106_native', False)]
    if (not native or not core or any(not p.requires_grad for p in core)
            or any(p.requires_grad != (arm == ARMS[1]) for p in native)):
        raise ValueError('R106 native marker/trainability or common selector trainability differs')
    nc, cc = sum(p.numel() for p in native), sum(p.numel() for p in core)
    return dict(total_parameter_count=nc + cc, trainable_parameter_count=cc + (nc if arm == ARMS[1] else 0),
        core_parameter_count=cc, native_parameter_count=nc,
        native_trainable_parameter_count=nc if arm == ARMS[1] else 0)


def make_model(params, arm, state_dict=None):
    return model_module().make_model(params, arm, state_dict=state_dict)


def initialize_shared(initial, geometry, arm, device='cpu'):
    if initial.get('trainable_sha256') != ORIGINAL_FRESH_SHA256:
        raise ValueError('R106 requires the original fresh R84 seed2 initialization, not selector continuation')
    model, params, receipt = model_module().initialize_shared(initial, geometry, arm, device)
    counts = parameter_counts(model, arm)
    if counts['core_parameter_count'] != CORE_PARAMETER_COUNT or params.get('dropout') != .1:
        raise ValueError('R106 must retain original selector and exactly three projections')
    if counts['native_parameter_count'] != NATIVE_PARAMETER_COUNT:
        raise ValueError('R106 must retain the three complete original native encoders')
    state_hash = parameter_hash(model.state_dict())
    if params.get('r106_initial_state_sha256') != state_hash:
        raise ValueError('R106 saved initial-state signature differs from actual complete fresh tensors')
    receipt.update(**counts, geometry_fitted_on='frozen R84 new81000 TRAIN only',
        common_fresh_model_state_sha256=state_hash,
        complete_state_equality_required=True, identical_complete_fresh_parameters=True,
        native_lr_scale=NATIVE_LR_SCALE, native_eval=True,
        native_pretraining_is_not_selector_continuation=True)
    return model, params, receipt


class NativePairRDropObjective(RDropObjective):
    """Reuse only the deterministic native graph, not either dropout forward."""

    def forward(self, batch, consistency_weight):
        if batch is None:
            return super().forward(batch, consistency_weight)
        with self.selector.native_pair_context(inputs_only(batch)):
            return super().forward(batch, consistency_weight)


def make_objective(selector, arm):
    if arm not in ARMS or not callable(getattr(selector, 'native_pair_context', None)):
        raise ValueError('R106 requires the input-only same-microbatch native pair context')
    return NativePairRDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not NativePairRDropObjective:
        raise TypeError('R106 requires the thin original two-forward RDropObjective wrapper')
    if getattr(objective.selector, '_native_pair', None) is not None:
        raise ValueError('An R106 native pair graph must never survive into a checkpoint boundary')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('No mutable objective/teacher state belongs to R106')


class NativeScaledAdamW(torch.optim.AdamW):
    """Keep R84's common update-axis LR while applying native .1 at step time."""

    def __init__(self, params, *args, **kwargs):
        parameters = list(params)
        if not parameters or any(not isinstance(p, torch.Tensor) for p in parameters):
            raise ValueError('R106 optimizer expects the complete model parameter iterator')
        groups = []
        for native, role, scale in ((False, 'core', 1.), (True, 'native', NATIVE_LR_SCALE)):
            chosen = [p for p in parameters if bool(getattr(p, '_r106_native', False)) == native]
            if not chosen:
                raise ValueError('R106 optimizer requires both core and marked native parameters')
            groups.append(dict(params=chosen, r106_role=role, lr_scale=scale))
        super().__init__(groups, *args, **kwargs)
        self.r106_successful_steps = 0
        self.r106_first_step_witness = None

    def group_structure(self):
        return [dict(role=g['r106_role'], lr_scale=g['lr_scale'],
            shapes=[list(p.shape) for p in g['params']],
            trainable=[bool(p.requires_grad) for p in g['params']]) for g in self.param_groups]

    def step(self, closure=None):
        common = [g['lr'] for g in self.param_groups]
        witness = None
        if self.r106_successful_steps == 0:
            native = self.param_groups[1]['params']
            gradients = [p for p in native if p.grad is not None]
            if any(not p.requires_grad for p in gradients):
                raise ValueError('Frozen native parameters must not receive an optimizer update')
            if any(p.requires_grad for p in native) and not gradients:
                raise ValueError('B must receive a native parameter gradient on its first update')
            sample = next((p for p in gradients if p.grad.detach().ne(0).any()), None)
            before = sample.detach().reshape(-1)[:32].clone() if sample is not None else None
            witness = dict(native_gradient_tensors=len(gradients),
                native_gradient_finite=all(bool(torch.isfinite(p.grad).all()) for p in gradients),
                native_trainable=any(p.requires_grad for p in native), native_effective_lr=common[1] * .1,
                common_lr=common[0], selection_effectiveness_evidence=False)
        try:
            for group, lr in zip(self.param_groups, common):
                group['lr'] = lr * group['lr_scale']
            result = super().step(closure)
        finally:
            for group, lr in zip(self.param_groups, common):
                group['lr'] = lr
        self.r106_successful_steps += 1
        if witness is not None:
            witness.update(successful_update=1, native_sample_changed=(
                bool(sample.detach().reshape(-1)[:32].ne(before).any()) if before is not None else False),
                sample_is_not_full_parameter_change_audit=True)
            self.r106_first_step_witness = witness
        return result

    def state_dict(self):
        state = super().state_dict()
        state.update(r106_group_structure=self.group_structure(),
            r106_successful_steps=self.r106_successful_steps,
            r106_first_step_witness=copy.deepcopy(self.r106_first_step_witness))
        return state

    def load_state_dict(self, state_dict):
        if state_dict.get('r106_group_structure') != self.group_structure():
            raise ValueError('R106 optimizer resume group roles/scales/shapes/trainability differ')
        for group, expected in zip(state_dict.get('param_groups', []), self.param_groups):
            if any(group.get(k) != expected[k] for k in ('r106_role', 'lr_scale')):
                raise ValueError('R106 optimizer saved native LR multiplier differs')
        super().load_state_dict(state_dict)
        self.r106_successful_steps = state_dict['r106_successful_steps']
        self.r106_first_step_witness = copy.deepcopy(state_dict['r106_first_step_witness'])


class _Facade:
    def __init__(self, source, **overrides):
        self._source, self._overrides = source, overrides

    def __getattr__(self, name):
        return self._overrides[name] if name in self._overrides else getattr(self._source, name)


def private_module(source, name):
    spec = importlib.util.spec_from_file_location('code.V4._r106_' + name, source.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def private_engine():
    module = private_module(engine, 'private_engine')
    module.micro_caps, module.write_json = micro_caps, _engine_write_json
    module._r106_caps = micro_caps()
    module.torch = _Facade(torch, optim=_Facade(torch.optim, AdamW=NativeScaledAdamW))
    return module


@lru_cache(maxsize=1)
def private_evaluator():
    module = private_module(engine, 'private_evaluator')
    # Evaluation uses its own locked cap, not min(eval cap, TRAIN microcap).
    module.micro_caps = lambda *args, **kwargs: dict.fromkeys(PROBLEMS, BATCH_SIZE)
    return module


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem,
                 max_retries=20):
    if (problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE
            or cap != private_engine()._r106_caps[problem]
            or any(k.startswith(('teacher_', 'donor_', 'nco_', 'route_target', 'context_')) for k in batch)):
        raise ValueError('R106 requires original logical1024 raw inputs and main-locked microcaps')
    return private_engine().update_batch(objective, optimizer, scaler, batch, weight,
                                        runtime, cap, max_retries=max_retries)


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    expected_caps = micro_caps(args.gpu_type, args.large_micro_cap,
        atsp_cap=args.atsp_micro_cap, other_cap=args.other_micro_cap)
    if caps != expected_caps:
        raise ValueError('R106 training caps differ from the final main hardware lock')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R106', r106_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, main_protocol_lock=protocol_lock(args),
        model_spec=copy.deepcopy(model_module().SPEC), core_parameter_count=CORE_PARAMETER_COUNT,
        total_parameter_count=TOTAL_PARAMETER_COUNT, trainable_parameter_count=PARAMETER_COUNTS[args.arm],
        native_parameter_count=NATIVE_PARAMETER_COUNT,
        native_trainable_parameter_count=NATIVE_PARAMETER_COUNT if args.arm == ARMS[1] else 0,
        initial_model_state_sha256=params['r106_initial_state_sha256'],
        **_execution(args), train_eval_every_updates=TRAIN_EVERY, role_arm=MODEL_ROLE_ARM,
        selection_head=SELECTION_HEAD, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        selected_val_binding_rule=VAL_BINDING_RULE, native_pair_context=PAIR_CONTEXT_RULE,
        native_execution='online_all_splits', native_feature_cache=False, native_eval=True,
        native_trainable=args.arm == ARMS[1], native_lr_scale=NATIVE_LR_SCALE,
        optimizer_groups=dict(core=dict(lr_scale=1.), native=dict(lr_scale=.1, marker='_r106_native')),
        identical_complete_fresh_parameters=True, trainable_parameter_matched=False,
        equal_training_compute_claim=False, native_pretraining_is_not_selector_continuation=True,
        objective=dict(classification='original R77 native CE', forward_passes=2,
            comparison=True, relative_risk=True, symmetric_rdrop=True, mutable_objective_state=False),
        external_pretraining=True, external_pretraining_data_overlap='unknown',
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        new_targets=False, auxiliary_loss=False, solver_quotas=False, R58_operations=False,
        cache_policy='direct frozen raw R84 reuse; no copy/build/native feature cache',
        oom_policy='fail without automatic cap shrink; one main input-only hardware witness before TRAIN',
        timing_scope='all18 online native+selector, GPU-resident batch128 amortized; not singleton or I/O end-to-end',
        historical_references=['R94B', 'R102B'], historical_references_matched=False,
        source_sha256={name: file_hash(V4 / name) for name in
            ('r106_experiment.py', 'r106_model.py', 'r106_donor.py', 'r106_reporting.py',
             'r84_experiment.py', 'r84_prepare.py', 'r88_model.py', 'r102_model.py',
             'r102_donor.py', 'r94_donor.py', 'r76_cache.py', 'r75_cache.py', 'pairwise_selector.py',
             'pairwise_objective.py', 'dual_stream.py', 'V4Model.py', 'r76_experiment.py', 'r77_experiment.py')})
    validate_checkpoint_config(config)
    return config


def validate_checkpoint_config(config):
    arm = config.get('arm')
    expected = dict(experiment='R106', r106_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, seed=2, solver_names=list(GLOBAL_SOLVERS), rows_per_problem=SPLIT_ROWS,
        global_batch_size=1024, logical_global_batch=1024, max_successful_updates=57600,
        warmup_updates=4320, eval_every_updates=1440, train_eval_every_updates=7200,
        optimizer='AdamW', weight_decay=1e-4, grad_clip=1., dropout=.1,
        rdrop_weight=.35, rdrop_ramp_updates=4320, loss=engine.LOSS_RECIPE,
        role_arm=MODEL_ROLE_ARM, selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE,
        checkpoint_rule=CHECKPOINT_RULE, selected_val_binding_rule=VAL_BINDING_RULE,
        model_spec=model_module().SPEC, native_pair_context=PAIR_CONTEXT_RULE,
        native_execution='online_all_splits', native_feature_cache=False, native_eval=True,
        native_lr_scale=.1, core_parameter_count=CORE_PARAMETER_COUNT,
        total_parameter_count=TOTAL_PARAMETER_COUNT, native_parameter_count=NATIVE_PARAMETER_COUNT,
        identical_complete_fresh_parameters=True, trainable_parameter_matched=False,
        equal_training_compute_claim=False, external_pretraining=True,
        current_query_labels_in_forward=False, historical_TRAIN_label_context_used=False,
        new_targets=False, auxiliary_loss=False, solver_quotas=False, R58_operations=False)
    if arm not in ARMS or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('R106 online/full18/objective/initialization protocol changed')
    execution = {k: config[k] for k in ('large_micro_cap', 'atsp_micro_cap', 'other_micro_cap', 'eval_batch_size')}
    record = config.get('main_protocol_lock', {}).get('record', {})
    params = config.get('model_params', {})
    if (config.get('native_trainable') != (arm == ARMS[1])
            or config.get('trainable_parameter_count') != PARAMETER_COUNTS[arm]
            or config.get('native_trainable_parameter_count') != (NATIVE_PARAMETER_COUNT if arm == ARMS[1] else 0)
            or any(params.get(k) != v for k, v in DIMENSIONS.items())
            or params.get('dropout') != .1 or params.get('role_arm') != MODEL_ROLE_ARM
            or params.get('r106_arm') != arm or params.get('model_spec') != model_module().SPEC
            or params.get('native_donor', {}).get('trainable') != (arm == ARMS[1])
            or params.get('native_donor', {}).get('mode') != 'eval'
            or not params.get('r106_initial_state_sha256')
            or config.get('initial_model_state_sha256') != params['r106_initial_state_sha256']
            or config.get('microbatch_caps') != micro_caps(config['gpu_type'], execution['large_micro_cap'],
                atsp_cap=execution['atsp_micro_cap'], other_cap=execution['other_micro_cap'])
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)
            or config.get('optimizer_groups') != dict(core=dict(lr_scale=1.), native=dict(lr_scale=.1, marker='_r106_native'))
            or any(record.get(k) != v for k, v in required_lock_fields(**execution).items())):
        raise ValueError('R106 model dimensions, LR groups or embedded main lock differ')
    for field, locked in (('data_root', 'data_root'), ('prepared_root', 'prepared_root'), ('cache_root', 'primary_cache')):
        if Path(config[field]).resolve() != Path(record[locked]).resolve():
            raise ValueError('R106 embedded path differs: ' + field)


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R106 checkpoint protocol or stateless loss identity differs')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    counts = parameter_counts(model, config['arm'])
    if (any(counts[k] != config[k] for k in counts)
            or parameter_hash(model.state_dict()) != parameter_hash(state['model'])):
        raise ValueError('R106 factory must restore every native/selector parameter and buffer exactly')
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


def selected_capture_proof(run, summary, lock, split, state=None):
    """A nullable capture SHA binds to parent lock via exact state/protocol/update."""
    run = Path(run)
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
        raise ValueError('Selected capture requires the exact immutable parent checkpoint')
    if state is None:
        state = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    capture = copy.deepcopy(summary.get('archive_capture_identity', summary['identity']))
    path = Path(summary['prediction_archive']).resolve()
    if not path.is_relative_to((run / 'evaluation' / split).resolve()):
        raise ValueError('Selected capture archive is outside this run/split')
    archived = read_json(path.parent / 'identity.json')
    model_hash = parameter_hash(state['model'])
    if (capture != archived or capture.get('experiment') != 'R106'
            or capture.get('arm') != lock.get('arm') or lock.get('arm') not in ARMS
            or state['args'].get('experiment') != 'R106' or state['args'].get('arm') != lock['arm']
            or capture.get('split') != split
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or capture.get('model_state_sha256') != model_hash
            or capture.get('state_hash') != lock['state_hash']
            or state.get('state_hash') != lock['state_hash'] or json_hash(state['args']) != lock['state_hash']
            or state.get('successful_updates') != lock['selected_update']
            or capture.get('successful_updates') != lock['selected_update']):
        raise ValueError('Selected capture differs from exact parent model state, protocol or update')
    return capture, dict(rule=VAL_BINDING_RULE, checkpoint_sha256=lock['checkpoint_sha256'],
        model_state_sha256=model_hash, state_hash=lock['state_hash'], selected_update=lock['selected_update'],
        archive_capture_checkpoint_sha256=capture.get('checkpoint_sha256'), archive_identity_unchanged=True)


def bind_selected_val(run, final):
    lock = read_json(Path(run) / 'final_test_lock.json')
    if any(final.get(k) != lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')):
        raise ValueError('Final summary differs from the unique VAL parent lock')
    result = copy.deepcopy(final)
    val = result['val']
    capture, binding = selected_capture_proof(run, val, lock, 'val')
    official = dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
    if (('checkpoint_binding' in val and val['checkpoint_binding'] != binding)
            or val['identity'] not in (capture, official)):
        raise ValueError('Selected VAL binding differs from its immutable capture proof')
    val.update(archive_capture_identity=capture, identity=official, checkpoint_binding=binding)
    return result


def _engine_write_json(path, value):
    path = Path(path)
    if path.name == 'final_metrics.json' and 'val' in value:
        value = bind_selected_val(path.parent, value)
    return write_json(path, value)


def evaluate(*args, **kwargs):
    return private_evaluator().evaluate(*args, **kwargs)


def authorize_test(args):
    lock, config = private_engine().authorize_test(args)
    validate_checkpoint_config(config)
    execution = {k: config[k] for k in ('large_micro_cap', 'atsp_micro_cap', 'other_micro_cap', 'eval_batch_size')}
    if (protocol_lock(args, execution=execution) != config['main_protocol_lock']
            or lock['inference_rule'] != INFERENCE_RULE):
        raise PermissionError('R106 TEST differs from the immutable main/VAL/inference lock')
    for k, value in execution.items():
        setattr(args, k, value)
    return lock, config


def ensure_test_cache(args, lock):
    verified, config = authorize_test(args)
    if verified != lock:
        raise PermissionError('R106 TEST cache requested under another lock')
    manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
    tasks = manifest.get('splits', {}).get('test', {})
    if set(tasks) != set(PROBLEMS) or any(tasks[p].get('rows') != SPLIT_ROWS['test']
            or tasks[p].get('complete') is not True for p in PROBLEMS):
        raise PermissionError('R106 needs preexisting complete full18 raw R84 TEST arrays; never build/copy')
    return manifest


def train(args, device):
    protocol_lock(args)
    local = private_engine()
    local.micro_caps = partial(micro_caps, atsp_cap=args.atsp_micro_cap, other_cap=args.other_micro_cap)
    local._r106_caps = local.micro_caps(args.gpu_type, args.large_micro_cap)
    hooks = sys.modules[__name__]
    last = args.root / args.arm / 'last.pt'
    if args.resume and last.is_file():
        # R84 requests initialization even on resume. Restore it from the complete
        # R106 state instead of needlessly rereading original native weight files.
        def resume_initialization(initial, geometry, arm, where):
            if (initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
                    or initial.get('trainable_sha256') != ORIGINAL_FRESH_SHA256
                    or parameter_hash(initial['trainable']) != ORIGINAL_FRESH_SHA256):
                raise ValueError('R106 resume still requires the exact frozen fresh R84 identity')
            model, saved = load_checkpoint(last, where)
            if saved['args']['arm'] != arm or saved['args']['geometry_stats'] != geometry:
                raise ValueError('R106 resume initial metadata differs from frozen TRAIN/arm')
            return model, copy.deepcopy(saved['args']['model_params']), copy.deepcopy(saved['initialization'])
        hooks = _Facade(hooks, initialize_shared=resume_initialization)
    return local.train(args, device, integration=hooks)


def test(args, device):
    lock, config = authorize_test(args)
    path = args.root / args.arm / 'test_results.json'
    if path.exists():
        saved = read_json(path)
        identity = saved.get('result', {}).get('identity', {})
        expected = dict(experiment='R106', arm=args.arm, split='test',
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            successful_updates=lock['selected_update'])
        val = read_json(args.root / args.arm / 'final_metrics.json')['val']
        if (any(identity.get(k) != v for k, v in expected.items())
                or identity.get('model_state_sha256') != val['identity']['model_state_sha256']
                or set(saved.get('result', {}).get('per_problem', {})) != set(PROBLEMS)
                or any(r.get('n') != SPLIT_ROWS['test'] for r in saved['result']['per_problem'].values())):
            raise PermissionError('Saved R106 TEST identity/population differs from the immutable parent lock')
    return private_engine().test(args, device, integration=sys.modules[__name__])


def raw_inputs(batch):
    names = ('node', 'matrix', 'node_mask', 'n', 'node_geom', 'kind', 'problem_id',
             'coord_dist', 'cbits', 'problem_desc', 'd_in', 'pool_ids', 'solver_mask', 'mask')
    return {key: batch[key] for key in names if key in batch}


def cached_input_batch(config, manifest, split, problem, device, count=128):
    """Raw input arrays only; never open labels, costs or native feature arrays."""
    from .r75_cache import _close, _open_arrays
    from .r76_cache import _validate_record
    root = Path(config['cache_root'])
    directory = next((p for p in (root / split / problem, root / (problem + split)) if p.is_dir()), None)
    if directory is None:
        raise FileNotFoundError('Existing raw input cache required: ' + problem + '/' + split)
    record = read_json(directory / 'metadata.json')
    specs = _validate_record(record, directory)
    expected = manifest['splits'][split][problem]
    if (record['problem'] != problem or record['split'] != split or record['rows'] != SPLIT_ROWS[split]
            or record['rows'] != expected['rows'] or record['rows'] < count
            or record['pool'] != config['candidate_pools'][problem]
            or record['pool_ids'] != config['candidate_pool_ids'][problem]
            or record['pool'] != expected['pool'] or record['pool_ids'] != expected['pool_ids']
            or Path(record['data_root']).resolve() != Path(config['data_root']).resolve()):
        raise ValueError('R106 input-only raw cache provenance differs')
    keys = ('n', 'coord_dist', 'node', 'node_geom') if record['kind'] == 'coord' else ('n', 'coord_dist', 'matrix')
    arrays = _open_arrays(directory, {k: specs[k] for k in keys})
    try:
        n = torch.from_numpy(np.array(arrays['n'][:count], copy=True))
        maximum = int(n.max())
        batch = dict(problem_id=record['problem_id'], kind=record['kind'], n=n,
            node_mask=torch.arange(maximum)[None] < n[:, None],
            cbits=torch.tensor(record['cbits'], dtype=torch.float32).repeat(count, 1),
            problem_desc=torch.tensor(record['problem_desc'], dtype=torch.float32).repeat(count, 1),
            coord_dist=torch.from_numpy(np.array(arrays['coord_dist'][:count], copy=True)),
            pool_ids=torch.tensor(record['pool_ids'], dtype=torch.long),
            mask=torch.tensor(record['mask'], dtype=torch.float32).repeat(count, 1),
            solver_mask=torch.ones(count, len(record['pool']), dtype=torch.bool))
        if record['kind'] == 'coord':
            batch['d_in'] = record['d_in']
            for key in ('node', 'node_geom'):
                batch[key] = torch.from_numpy(np.array(arrays[key][:count, :maximum], copy=True))
        else:
            batch['matrix'] = torch.from_numpy(np.array(arrays['matrix'][:count, :maximum, :maximum], copy=True))
    finally:
        _close(arrays)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def measure_inputs(model, batch, device, repetitions=3):
    raw = raw_inputs(batch)
    donor = getattr(model, 'donor', None)
    if not isinstance(donor, torch.nn.Module):
        raise ValueError('Online timing requires the real native donor module, not cached features')
    calls, dtypes = [], []
    def witness(module, args, output):
        if any(k not in raw for k in args[0]) or any(k.startswith('nco_') for k in args[0]):
            raise ValueError('Native timing forward received non-whitelist inputs')
        calls.append(1)
        if not torch.is_tensor(output) or output.dtype != torch.float32:
            raise ValueError('Native timing must witness FP32 native node output')
        dtypes.append(str(output.dtype))
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    handle = donor.register_forward_hook(witness)
    samples, invocations = [], []
    try:
        model(raw)
        sync()
        if len(calls) != 1:
            raise ValueError('Warmup must invoke the online native donor exactly once')
        for _ in range(repetitions):
            before = len(calls)
            sync()
            started = time.perf_counter()
            output = model(raw)
            sync()
            samples.append(time.perf_counter() - started)
            invocations.append(len(calls) - before)
            if (invocations[-1] != 1
                    or not torch.isfinite(output['logits'][output['solver_mask']]).all()):
                raise ValueError('Timing requires finite real online native+selector calls')
    finally:
        handle.remove()
    return dict(batch_seconds=samples, mean_seconds_per_instance=sum(samples) / (repetitions * len(raw['node_mask'])),
        donor_calls=invocations, warmup_donor_calls=1, native_output_dtype=sorted(set(dtypes)))


@torch.no_grad()
def timing(args, device):
    if device.type != 'cuda':
        raise ValueError('Formal R106 timing requires the launcher-assigned CUDA device')
    lock, config = authorize_test(args)
    run = args.root / args.arm
    record = read_json(run / 'test_results.json')
    if (record.get('complete') is not True or record.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or record.get('state_hash') != lock['state_hash']):
        raise PermissionError('R106 timing requires completed same-lock full TEST')
    destination = run / 'inference_timing.json'
    if destination.exists():
        from .r106_reporting import inference_timing
        return inference_timing(run, lock)
    model, state = load_checkpoint(run / 'best.pt', device)
    model_hash = parameter_hash(state['model'])
    if model_hash != record['result']['identity']['model_state_sha256']:
        raise ValueError('R106 timing model differs from locked TEST')
    counts = parameter_counts(model, args.arm)
    del state
    model.eval()
    manifest = ensure_test_cache(args, lock)
    rows = []
    for problem in PROBLEMS:
        raw = cached_input_batch(config, manifest, 'test', problem, device)
        rows.append(dict(problem=problem, instances=128, online_native_plus_selector=measure_inputs(model, raw, device)))
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256'] or parameter_hash(model.state_dict()) != model_hash:
        raise ValueError('Timing changed the unique R106 checkpoint/model')
    value = dict(experiment='R106', arm=args.arm, complete=True, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=model_hash, model_factory=MODEL_FACTORY,
        inference_rule=INFERENCE_RULE, device=str(device), repetitions=3, warmup_batches=1,
        sample_policy='first128 frozen TEST inputs per problem', **counts,
        scope='online_native_plus_selector_GPU_resident_inputs', batch128_amortized=True, singleton_measured=False,
        all_current_model_included=True, donor_present=True, cached_feature_latency_used=False,
        end_to_end_measured=False, input_labels_read=False, model_reselection=False,
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver decoding/execution', per_problem=rows)
    write_json(destination, value)
    return value


def describe(args):
    return dict(experiment='R106', arm=args.arm, required_protocol_lock=required_lock_fields(**_execution(args)),
        microbatch_caps=micro_caps(args.gpu_type, args.large_micro_cap,
            atsp_cap=args.atsp_micro_cap, other_cap=args.other_micro_cap),
        eval_batch_size=args.eval_batch_size, cache_root=str(args.cache_root or DEFAULT_CACHE_ROOT),
        model_factory=MODEL_FACTORY, inference_rule=INFERENCE_RULE, wandb_mode=args.wandb_mode,
        protocol_lock_owner='main; describe never creates a lock or readiness receipt')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'timing', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='4090')
    result.add_argument('--large-micro-cap', type=int, default=64)
    result.add_argument('--atsp-micro-cap', type=int, default=128)
    result.add_argument('--other-micro-cap', type=int, default=256)
    result.add_argument('--eval-batch-size', type=int, default=128)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R106')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'prepared_root', 'data_root', 'cache_root', 'protocol_lock'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command in ('describe', 'status'):
        value = describe(args) if args.command == 'describe' else (
            read_json(args.root / args.arm / 'status.json') if (args.root / args.arm / 'status.json').exists()
            else dict(experiment='R106', phase='not_started'))
        print(json.dumps(value, indent=2))
        return value
    if args.command in ('train', 'test'):
        return private_engine().main(argv, integration=sys.modules[__name__])
    run = args.root / args.arm
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        engine.configure_torch()
        return timing(args, torch.device(args.device))


if __name__ == '__main__':
    main()
