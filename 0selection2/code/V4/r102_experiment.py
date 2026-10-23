"""Future R102 full-type early fusion; no launch occurs on import or describe.

The unchanged R84 update loop is loaded privately. Only TSP/ATSP donor weights
differ between arms; both routing branches use the same frozen R94B donor.
"""

import argparse
import copy
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .pairwise_objective import inputs_only
from .r76_experiment import PART_KEYS, file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import selection_counts, zero_summary


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R102_full_type_pretrained_fusion'
ARMS = NEURAL_ARMS = ('A', 'B')
ARM_VARIANTS = VARIANTS = dict(A='extra_random', B='full_pretrained')
INITIALIZATION = 'fresh_r84_plus_full_type_fusion'
MODEL_FACTORY = 'code.V4.r102_experiment.load_checkpoint'
MODEL_ROLE_ARM, ROLE_SLOT = 'B_execution_anchor', 3
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES, EVAL_EVERY = 1024, 57600, 4320, 1440
TRAIN_EVERY, PEAK_LR, END_LR = 7200, 1e-4, 1e-6
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
ORIGINAL_PARAMETERS, NEW_PARAMETERS, PARAMETER_COUNT = 1842953, 98304, 1941257
PROJECTION_SHAPES = {
    'instance_encoder.nco_projections.TSP.weight': (128, 128),
    'instance_encoder.nco_projections.ATSP.weight': (128, 512),
    'instance_encoder.nco_projections.VRP.weight': (128, 128),
}
PROJECTION_NAMES = tuple(PROJECTION_SHAPES)
WIDTH_BY_PROBLEM = {p: 512 if p == 'ATSP' else 128 for p in PROBLEMS}
ROUTING = tuple(p for p in PROBLEMS if p not in ('TSP', 'ATSP'))
R94B_CHECKPOINT_SHA256 = '2ce5fb49950c4c9f51d2015a125744be9960aa0c60c609cfe235e74d44ad4be3'
SELECTION_HEAD = 'original_score_mlp_maximin'
WINNER_RULE = engine.WINNER_RULE
CHECKPOINT_RULE = ('strict minimum full VAL macro_vs_sbs_pct with common new81000 '
                   'TRAIN-fitted SBS; first wins ties')
INFERENCE_RULE = (
    'eval frozen TSP ELG128 / ATSP MatNet row256+col256 / routing ReLD128; '
    'ATSP fixed identity onehot columns firstN, not historical RNG certification; '
    'FP32 donor output quantized FP16; affine-free normalization and three projections '
    'before original four instance layers; original physical TSP node0 slot3 raw role; '
    'hand34/methodID, joint2, score MLP/pair differences/maximin; '
    'stable ascending-global-solver-ID ties; all legal pools; no calibration')
TIMING_SCOPES = ('donor_plus_selector', 'cache_hit_selector')
prepared_config, micro_caps = engine.prepared_config, engine.micro_caps
json_hash, task_order, learning_rate = engine.json_hash, engine.task_order, engine.learning_rate
status_update = engine.status_update


def model_spec():
    return copy.deepcopy(importlib.import_module('code.V4.r102_model').SPEC)


def make_model(params, arm, state_dict=None, *, donor=None):
    return importlib.import_module('code.V4.r102_model').make_model(
        params, arm, state_dict=state_dict, donor=donor)


def feature_root(args):
    return Path(getattr(args, 'feature_root', None) or
                ('/tmp/shiys_R102_' + args.arm + '_extra_nodes')).expanduser()


def routing_feature_root(args):
    return Path(getattr(args, 'routing_feature_root', None) or
                '/tmp/shiys_R94_B_node_features').expanduser()


def required_lock_fields():
    return dict(experiment='R102', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', rows_per_problem=dict(SPLIT_ROWS), batch_size=BATCH_SIZE,
        logical_global_batch=BATCH_SIZE, max_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        train_eval_every=TRAIN_EVERY, large_micro_cap=352, eval_batch_size=256,
        donor_variants=ARM_VARIANTS.copy())


def protocol_lock(args):
    path = Path(args.protocol_lock or args.root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    if any(record.get(k) != v for k, v in required_lock_fields().items()):
        raise ValueError('R102 requires the main-owned final full18 matched protocol lock')
    for name in ('data_root', 'prepared_root'):
        if not record.get(name) or Path(record[name]).resolve() != getattr(args, name).resolve():
            raise ValueError('R102 lock names a different ' + name)
    return dict(path=str(path), sha256=file_hash(path), record=record)


def objective_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R102 arm')
    return dict(main='unchanged R77 CE/comparison/risk plus symmetric R-Drop',
        classification='original R77 maximin', coefficients=dict(ce=.35, comparison=.35, risk=.02),
        rdrop='linear ramp to .35 over4320 updates', forward_passes=2,
        donor_preparations_per_pair=1, shared_safe_nco_fields=True,
        auxiliary_loss=False, new_target=False, route_targets=False,
        cross_instance_objective=False, mutable_objective_state=False)


def validate_donor_provenance(record, arm):
    if arm not in ARMS or not isinstance(record, dict):
        raise ValueError('R102 donor provenance must name all three components')
    components = record.get('components', {})
    if (set(components) != {'TSP', 'ATSP', 'VRP'} or record.get('arm') != arm
            or record.get('variant') != ARM_VARIANTS[arm]
            or record.get('supported_problems') != list(PROBLEMS)
            or record.get('widths') != WIDTH_BY_PROBLEM):
        raise ValueError('R102 donor provenance must retain the full18 component interface')
    for name, component in components.items():
        variant = ('random' if arm == 'A' else 'pretrained') if name != 'VRP' else 'pretrained'
        if not isinstance(component, dict) or component.get('variant') != variant:
            raise ValueError('Only TSP/ATSP initialization may differ; routing is pretrained in BOTH arms')
    if components['VRP'].get('checkpoint_sha256') != R94B_CHECKPOINT_SHA256:
        raise ValueError('R102 routing donor must be the pinned unchanged R94B donor')
    return copy.deepcopy(record)


def donor_record(donor, arm):
    value = validate_donor_provenance(donor.provenance(), arm)
    if donor.variant != ARM_VARIANTS[arm] or any(p.requires_grad for p in donor.parameters()):
        raise ValueError('FullTypeDonor must have the arm variant and remain completely frozen')
    return value


def initialize_shared(initial, geometry, arm, device='cpu'):
    reference = initial['trainable']
    if (arm not in ARMS or initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or parameter_hash(reference) != initial.get('trainable_sha256')
            or sum(t.numel() for t in reference.values()) != ORIGINAL_PARAMETERS):
        raise ValueError('R102 requires the original fresh R84 seed2 state, never a trained selector')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        model = make_model(initial['model_params'], arm)
    actual = {n: p for n, p in model.named_parameters() if p.requires_grad}
    retained = {n: p for n, p in actual.items() if n in reference and p.shape == reference[n].shape}
    added = {n: p for n, p in actual.items() if n not in retained}
    if (set(retained) != set(reference) or set(added) != set(PROJECTION_NAMES)
            or any(tuple(added[n].shape) != shape for n, shape in PROJECTION_SHAPES.items())
            or sum(p.numel() for p in actual.values()) != PARAMETER_COUNT):
        raise ValueError('R102 must retain the backbone and add exactly the three bias-free projections')
    for name, shape in PROJECTION_SHAPES.items():
        group = name.split('.')[-2]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(model_spec()['projection_seeds'][group])
            expected = torch.nn.Linear(shape[1], shape[0], bias=False).weight
        if not torch.equal(added[name].detach().cpu(), expected.detach()):
            raise ValueError('R102 projection did not use the main-owned isolated seed: ' + name)
    provenance = donor_record(model.donor, arm)
    with torch.no_grad():
        for name, tensor in retained.items():
            tensor.copy_(reference[name])
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    common_hash, new_hash, full_hash = map(parameter_hash, (retained, added, actual))
    if common_hash != initial['trainable_sha256']:
        raise ValueError('R102 original R84 parameter initialization changed')
    params = copy.deepcopy(model.params)
    params.update(r102_fresh_trainable_sha256=full_hash,
        r102_common_parameter_sha256=common_hash, r102_new_parameter_sha256=new_hash)
    receipt = dict(seed=2, initialization=INITIALIZATION,
        historical_selector_checkpoint_loaded=False, r77_fresh_trainable_sha256=common_hash,
        external_solver_pretraining=True, extra_type_pretraining=arm == 'B',
        common_parameter_names=sorted(retained), common_parameter_sha256=common_hash,
        common_parameter_count=ORIGINAL_PARAMETERS, new_parameter_names=sorted(added),
        new_parameter_shapes={n: list(v.shape) for n, v in added.items()},
        projection_shapes={n: list(v.shape) for n, v in added.items()},
        new_parameter_sha256=new_hash, new_parameter_count=NEW_PARAMETERS,
        trainable_sha256=full_hash, trainable_parameter_count=PARAMETER_COUNT,
        frozen_parameter_count=sum(p.numel() for p in model.donor.parameters()),
        frozen_donor=provenance, removed_parameter_names=[],
        geometry_fitted_on='frozen R84 new81000 TRAIN only')
    return model.to(device), params, receipt


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    if (args.large_micro_cap != 352 or args.eval_batch_size != 256
            or caps != micro_caps('3090', 352) or args.wandb_mode != 'offline'):
        raise ValueError('R102 fixes original per-task microbatches, eval256 and explicit offline W&B')
    prepared = read_json(args.root / args.arm / 'input_features_ready.json')
    provenance = validate_donor_provenance(prepared['donor_provenance'], args.arm)
    if params.get('frozen_donor') != dict(variant=ARM_VARIANTS[args.arm], provenance=provenance):
        raise ValueError('Prepared features and initialized model use different frozen donor identities')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R102', r102_schema_version=1, initialization=INITIALIZATION,
        model_factory=MODEL_FACTORY, model_spec=model_spec(), main_protocol_lock=protocol_lock(args),
        objective=objective_spec(args.arm), donor_variant=ARM_VARIANTS[args.arm],
        donor_variants=ARM_VARIANTS.copy(), donor_provenance=provenance,
        frozen_donor=dict(frozen=True, eval_only=True, embedded_in_checkpoint=True),
        role_arm=MODEL_ROLE_ARM, input_roles=dict(arm=MODEL_ROLE_ARM, problem='TSP',
            node_column=ROLE_SLOT, assignment='physical input node0', label_dependent=False),
        node_fusion_initialization_sha256=params['r102_fresh_trainable_sha256'],
        retained_parameter_sha256=params['r102_common_parameter_sha256'],
        new_parameter_sha256=params['r102_new_parameter_sha256'],
        trainable_parameter_count=PARAMETER_COUNT, original_parameter_count=ORIGINAL_PARAMETERS,
        new_parameter_count=NEW_PARAMETERS, projection_shapes={n: list(s) for n, s in PROJECTION_SHAPES.items()},
        feature_cache_root=str(feature_root(args).resolve()),
        routing_feature_cache_root=str(routing_feature_root(args).resolve()),
        routing_features='validated read-only DIRECT reuse of original R94B cache; same in A and B',
        feature_precision='native FP32 frozen donors then FP16 nodes; same online quantization',
        source_preparation_batch=128, train_eval_every_updates=TRAIN_EVERY,
        selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE,
        initialization_receipt='initialization.json', auxiliary_loss=False, new_target=False,
        external_pretraining=True, checkpoint_donor_restore='embedded only; initialize_from_checkpoint=False',
        external_solver_pretraining=True, extra_type_pretraining=args.arm == 'B',
        historical_reference=dict(primary='R94B', secondary='R84A', retrained=False, matched=False),
        wandb_project='selector', wandb_entity='yjkds-southern-university-of-science-technology',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r102_experiment.py', 'r102_model.py', 'r102_objective.py', 'r102_donor.py',
             'r102_features.py', 'r84_experiment.py', 'r84_prepare.py', 'r88_model.py',
             'V4Model.py', 'pairwise_selector.py', 'pairwise_objective.py', 'r76_experiment.py')})
    validate_checkpoint_config(config)
    return config


def validate_checkpoint_config(config):
    arm = config.get('arm')
    expected = dict(experiment='R102', r102_schema_version=1, seed=2, initialization=INITIALIZATION,
        model_factory=MODEL_FACTORY, model_spec=model_spec(), solver_names=list(GLOBAL_SOLVERS),
        rows_per_problem=SPLIT_ROWS, global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, train_eval_every_updates=TRAIN_EVERY, eval_batch_size=256,
        optimizer='AdamW', weight_decay=1e-4, grad_clip=1., dropout=.1, rdrop_weight=.35,
        rdrop_ramp_updates=WARMUP_UPDATES, microbatch_caps=micro_caps('3090', 352),
        donor_variants=ARM_VARIANTS, selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE,
        checkpoint_rule=CHECKPOINT_RULE, early_stop=False, fixed_budget=True, winner_rule=WINNER_RULE,
        trainable_parameter_count=PARAMETER_COUNT, original_parameter_count=ORIGINAL_PARAMETERS,
        new_parameter_count=NEW_PARAMETERS, projection_shapes={n: list(s) for n, s in PROJECTION_SHAPES.items()},
        frozen_donor=dict(frozen=True, eval_only=True, embedded_in_checkpoint=True),
        source_preparation_batch=128, auxiliary_loss=False, new_target=False, external_pretraining=True,
        external_solver_pretraining=True,
        wandb=True, wandb_mode='offline', wandb_project='selector',
        wandb_entity='yjkds-southern-university-of-science-technology',
        checkpoint_donor_restore='embedded only; initialize_from_checkpoint=False',
        role_arm=MODEL_ROLE_ARM, input_roles=dict(arm=MODEL_ROLE_ARM, problem='TSP',
            node_column=ROLE_SLOT, assignment='physical input node0', label_dependent=False))
    if arm not in ARMS or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('Not the locked fresh full18 R102 architecture/training/inference protocol')
    validate_donor_provenance(config.get('donor_provenance'), arm)
    if (config.get('donor_variant') != ARM_VARIANTS[arm]
            or config.get('extra_type_pretraining') != (arm == 'B')
            or config.get('objective') != objective_spec(arm) or config.get('loss') != engine.LOSS_RECIPE
            or config.get('lr') != dict(rule='successful-update warmup then cosine, independent of VAL',
                peak=PEAK_LR, end=END_LR, warmup_updates=WARMUP_UPDATES, max_updates=MAX_UPDATES)
            or config.get('task_order_sha256') != hashlib.sha256(
                '\n'.join(task_order(MAX_UPDATES)).encode('ascii')).hexdigest()):
        raise ValueError('R102 original loss, donor variant, task order or LR changed')
    params = config.get('model_params', {})
    if params.get('frozen_donor') != dict(variant=ARM_VARIANTS[arm], provenance=config['donor_provenance']):
        raise ValueError('R102 saved model donor metadata differs from the protocol')
    for key, param in (('node_fusion_initialization_sha256', 'r102_fresh_trainable_sha256'),
                       ('retained_parameter_sha256', 'r102_common_parameter_sha256'),
                       ('new_parameter_sha256', 'r102_new_parameter_sha256')):
        if not config.get(key) or config[key] != params.get(param):
            raise ValueError('R102 fresh initialization hashes missing or inconsistent')
    if config['retained_parameter_sha256'] != config.get('common_trainable_sha256'):
        raise ValueError('R102 original trainables must equal fresh R84 initialization')
    embedded = config.get('main_protocol_lock', {})
    if (not embedded.get('sha256') or any(embedded.get('record', {}).get(k) != v
            for k, v in required_lock_fields().items())):
        raise ValueError('R102 embedded main-owned protocol lock changed')
    for name in ('data_root', 'prepared_root'):
        if embedded['record'].get(name) != config.get(name):
            raise ValueError('R102 embedded lock data/preparation identity changed')


def load_checkpoint(path, device='cpu'):
    """Restore all donor tensors from best.pt; never reopen external weights."""
    from .r102_donor import FullTypeDonor
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R102 checkpoint protocol or stateless objective changed')
    with torch.random.fork_rng(devices=[]):
        donor = FullTypeDonor(config['arm'], initialize_from_checkpoint=False,
                              expected_provenance=copy.deepcopy(config['donor_provenance']))
        model = make_model(config['model_params'], config['arm'], state_dict=state['model'], donor=donor)
    if donor_record(model.donor, config['arm']) != config['donor_provenance']:
        raise ValueError('Restored R102 donor provenance differs from the checkpoint')
    actual = model.state_dict()
    if set(actual) != set(state['model']) or not any(k.startswith('donor.') for k in actual):
        raise ValueError('R102 checkpoint must embed the entire frozen donor and original state')
    for name, value in actual.items():
        if not torch.equal(value.detach().cpu(), state['model'][name].detach().cpu()):
            raise ValueError('R102 factory failed complete state restoration: ' + name)
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


def make_objective(selector, arm):
    objective_spec(arm)
    return importlib.import_module('code.V4.r102_objective').RDropObjective(selector)


def checkpoint_extras(objective):
    return {}


def restore_objective(objective, checkpoint):
    if 'objective_state' in checkpoint:
        raise ValueError('R102 has no auxiliary or mutable objective state')


def inputs_with_donor(batch):
    clean = inputs_only({k: v for k, v in batch.items() if not k.startswith('nco_')})
    clean.update({k: batch[k] for k in ('nco_nodes', 'nco_node_mask') if k in batch})
    return clean


def raw_inputs(batch):
    # Explicit removal of BOTH fields is essential: either can bypass online extraction.
    clean = dict(batch)
    clean.pop('nco_nodes', None)
    clean.pop('nco_node_mask', None)
    return inputs_only({k: v for k, v in clean.items() if not k.startswith('nco_')})


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem, max_retries=20):
    if (problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE
            or cap != micro_caps('3090', 352)[problem]
            or not all(k in batch for k in ('nco_nodes', 'nco_node_mask'))):
        raise ValueError('R102 requires original full1024 row-weighted micros and mandatory donor cache')
    parts, retries, _ = engine.update_batch(objective, optimizer, scaler, batch, weight,
                                           runtime, cap, max_retries=max_retries)
    return parts, retries, {}


def offline_wandb(args, config):
    import wandb
    if args.wandb_mode != 'offline':
        raise ValueError('R102 permits explicit offline W&B only')
    run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
        mode='offline', name='R102_' + args.arm, dir=str(args.root / args.arm), config=config,
        id='r102-' + hashlib.sha256((str(args.root) + args.arm).encode('utf-8')).hexdigest()[:16])
    write_json(args.root / args.arm / 'wandb_receipt.json', dict(requested='offline', mode='offline',
        project='selector', entity='yjkds-southern-university-of-science-technology',
        run_id=run.id, url=None, updated_utc=utc()))
    return run


def lock_best(root, best, config, state_hash):
    """Original first-tie VAL selection, with explicit R102 feature authorization."""
    record = dict(experiment='R102', arm=config['arm'], checkpoint=str((root / 'best.pt').resolve()),
        checkpoint_sha256=file_hash(root / 'best.pt'), state_hash=state_hash,
        epoch=best['epoch'], selected_update=best['successful_updates'], rule=config['checkpoint_rule'],
        selection_split='val', selected_without_test=True, per_problem_assembly=False,
        val=best['val']['macro'], inference_rule=config['inference_rule'], winner_rule=WINNER_RULE)
    path = root / 'final_test_lock.json'
    if path.exists():
        previous = read_json(path)
        if any(previous.get(k) != v for k, v in record.items()):
            raise PermissionError('R102 unique whole-model VAL/inference lock cannot change')
        return previous
    record['locked_utc'] = utc()
    write_json(path, record)
    return record


@lru_cache(maxsize=4)
def private_engine(arm, feature_directory, routing_feature_directory, provenance_json):
    from .r102_features import FrozenFeatureCache
    if arm not in ARMS:
        raise ValueError('Unknown R102 arm')
    provenance = validate_donor_provenance(json.loads(provenance_json), arm)
    identity = hashlib.sha256((feature_directory + routing_feature_directory + provenance_json).encode()).hexdigest()[:12]
    name = 'code.V4._r102_engine_' + arm + '_' + identity
    spec = importlib.util.spec_from_file_location(name, engine.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    def open_with_features(cache_root, split, problem, manifest, data_root, **kwargs):
        base = engine.open_cache(cache_root, split, problem, manifest, data_root, **kwargs)
        try:
            return FrozenFeatureCache(base, Path(feature_directory) / split / problem,
                expected_variant=ARM_VARIANTS[arm], expected_provenance=copy.deepcopy(provenance),
                routing_feature_directory=Path(routing_feature_directory) / split / problem)
        except BaseException:
            base.close()
            raise

    module.open_cache, module.inputs_only, module.wandb_run = open_with_features, inputs_with_donor, offline_wandb
    module.lock_best = lock_best
    return module


def feature_receipts(record, split, args, provenance):
    tasks = record.get('splits', {}).get(split, record)
    if not isinstance(tasks, dict) or set(tasks) != set(PROBLEMS):
        raise ValueError('R102 features require complete all18 receipts, including validated routing reuse')
    for problem, receipt in tasks.items():
        identity = receipt.get('identity', receipt)
        shape = receipt.get('shape', [])
        width = receipt.get('width', shape[-1] if shape else None)
        if (receipt.get('complete') is not True or receipt.get('rows_written') != SPLIT_ROWS[split]
                or identity.get('problem') != problem or identity.get('split') != split
                or identity.get('rows') != SPLIT_ROWS[split] or width != WIDTH_BY_PROBLEM[problem]
                or receipt.get('dtype') != 'float16' or identity.get('input_only') is not True
                or identity.get('labels_opened') is True):
            raise ValueError('Incomplete or incompatible R102 feature receipt: ' + split + '/' + problem)
        directory = Path(receipt.get('directory', '')).resolve()
        # Routing receipts live in the extra-root too; only the referenced arrays
        # stay in the common transported R94B routing directory.
        root = feature_root(args)
        if directory != (root / split / problem).resolve():
            raise ValueError('R102 feature receipt uses the wrong extra/routing root')
        if problem in ROUTING and (receipt.get('source_kind') != 'r94b_reference'
                or Path(receipt.get('routing_directory', '')).resolve()
                != (routing_feature_root(args) / split / problem).resolve()):
            raise ValueError('R102 routing must reference the identical original pretrained R94B cache')
        component = 'VRP' if problem in ROUTING else problem
        donor = identity.get('donor', receipt.get('donor'))
        if donor not in (provenance, provenance['components'][component]):
            raise ValueError('R102 feature receipt donor identity changed: ' + problem)
    return tasks


def prepare_input_features(args, donor, device, *, split=None, test_lock=None):
    from .r102_features import prepare_features
    if split not in (None, 'train', 'val', 'test') or split == 'test' and test_lock is None:
        raise PermissionError('R102 TEST features require the completed unique VAL lock')
    provenance = donor_record(donor, args.arm)
    records = {}
    for name in ((split,) if split is not None else ('train', 'val')):
        record = prepare_features(args.cache_root, feature_root(args), name, donor, device,
            routing_feature_root=routing_feature_root(args), batch_size=128,
            manifest=args.prepared_root / 'source_cache_manifest.json', data_root=args.data_root,
            test_lock=test_lock)
        records[name] = feature_receipts(record, name, args, provenance)
    return records


def _module(args, provenance):
    return private_engine(args.arm, str(feature_root(args).resolve()),
        str(routing_feature_root(args).resolve()), json.dumps(provenance, sort_keys=True))


def prepare(args, device):
    from .r102_donor import FullTypeDonor
    lock = protocol_lock(args)
    prepared_config(args)
    started = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        donor = FullTypeDonor(args.arm).to(device)
    provenance = donor_record(donor, args.arm)
    receipts = prepare_input_features(args, donor, device)
    value = dict(experiment='R102', arm=args.arm, variant=ARM_VARIANTS[args.arm], complete=True,
        completed_utc=utc(), main_protocol_lock_sha256=lock['sha256'], splits=['train', 'val'],
        feature_root=str(feature_root(args).resolve()), routing_feature_root=str(routing_feature_root(args).resolve()),
        cache_root=str(args.cache_root.resolve()), donor_provenance=provenance,
        feature_receipts=receipts, seconds=time.monotonic() - started, preparation_batch_size=128,
        new_feature_problems=['TSP', 'ATSP'], reused_feature_problems=list(ROUTING),
        labels_used=False, solver_executed=False, new_target=False)
    path = args.root / args.arm / 'input_features_ready.json'
    if path.exists():
        prior = read_json(path)
        if any(prior.get(k) != v for k, v in value.items() if k not in ('seconds', 'completed_utc')):
            raise ValueError('R102 preparation identity changed; preserve existing run')
        return prior
    write_json(path, value)
    return value


def train(args, device):
    lock = protocol_lock(args)
    prepared_config(args)
    ready = read_json(args.root / args.arm / 'input_features_ready.json')
    expected = dict(experiment='R102', arm=args.arm, variant=ARM_VARIANTS[args.arm], complete=True,
        main_protocol_lock_sha256=lock['sha256'], splits=['train', 'val'], preparation_batch_size=128,
        cache_root=str(args.cache_root.resolve()), feature_root=str(feature_root(args).resolve()),
        routing_feature_root=str(routing_feature_root(args).resolve()))
    if any(ready.get(k) != v for k, v in expected.items()):
        raise ValueError('Prepare matching R102 TRAIN/VAL extra features and routing reuse first')
    provenance = validate_donor_provenance(ready.get('donor_provenance'), args.arm)
    for split in ('train', 'val'):
        feature_receipts(ready.get('feature_receipts', {}).get(split, {}), split, args, provenance)
    return _module(args, provenance).train(args, device, integration=sys.modules[__name__])


def authorize_test(args):
    lock, config = engine.authorize_test(args)
    validate_checkpoint_config(config)
    if (lock.get('inference_rule') != config['inference_rule']
            or lock.get('selection_split') != 'val'
            or protocol_lock(args) != config['main_protocol_lock']
            or feature_root(args).resolve() != Path(config['feature_cache_root']).resolve()
            or routing_feature_root(args).resolve() != Path(config['routing_feature_cache_root']).resolve()
            or args.cache_root is not None and args.cache_root.resolve() != Path(config['cache_root']).resolve()):
        raise PermissionError('R102 TEST data/cache/inference identity changed after the unique VAL lock')
    return lock, config


def ensure_test_cache(args, lock):
    """Only consume the existing R84 TEST source cache, after the unique lock."""
    from .r85_experiment import ensure_test_cache as existing_test_cache
    verified, _ = authorize_test(args)
    if verified != lock:
        raise PermissionError('R102 TEST preparation was given a different lock')
    return existing_test_cache(args, lock)


def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    provenance = donor_record(selector.donor, args.arm)
    if split == 'test':
        lock, _ = authorize_test(args)
        started = time.monotonic()
        receipts = prepare_input_features(args, selector.donor, next(selector.parameters()).device,
                                          split='test', test_lock=lock)
        write_json(args.root / args.arm / 'test_input_features_ready.json', dict(experiment='R102',
            arm=args.arm, complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=lock['state_hash'], feature_receipts=receipts,
            seconds=time.monotonic() - started, labels_used=False, preparation_batch_size=128))
    return _module(args, provenance).evaluate(selector, args, manifest, baselines,
        split, tag, step, state_hash, checkpoint_sha256)


def test(args, device):
    return engine.test(args, device, integration=sys.modules[__name__])


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_inputs(selector, base_batch, cached_batch, device, repetitions=3):
    """Hook-verified online donor+selector versus a separate cache-hit scope."""
    if repetitions != 3:
        raise ValueError('R102 locks three timed repetitions')
    raw, cached = raw_inputs(base_batch), inputs_with_donor(cached_batch)
    if (not all(k in cached for k in ('nco_nodes', 'nco_node_mask'))
            or len(raw['node_mask']) != 128 or not torch.equal(raw['node_mask'], cached['node_mask'])):
        raise ValueError('R102 timing requires aligned raw/cached first128 inputs')
    calls = 0

    def counted(module, inputs, output):
        nonlocal calls
        calls += 1

    handle = selector.donor.register_forward_hook(counted)
    values = {}
    try:
        for scope in TIMING_SCOPES:
            start_count, samples = calls, []
            for index in range(repetitions + 1):
                # Rebuild input every call, including warmup; never reuse prepared online H.
                batch = raw_inputs(raw) if scope == 'donor_plus_selector' else dict(cached)
                before = calls
                _synchronize(device)
                started = time.perf_counter()
                output = selector(batch)
                _synchronize(device)
                elapsed = time.perf_counter() - started
                expected = 1 if scope == 'donor_plus_selector' else 0
                if calls - before != expected:
                    raise ValueError('Online timing bypassed the donor or cached timing executed it')
                if not torch.isfinite(output['logits'][output['solver_mask']]).all():
                    raise ValueError('Nonfinite locked R102 timing logits')
                if index:
                    samples.append(elapsed)
            values[scope] = dict(batch_seconds=samples,
                mean_seconds_per_instance=sum(samples) / (repetitions * 128),
                donor_forward_calls=calls - start_count, warmup_batches=1)
    finally:
        handle.remove()
    return dict(**values, online_donor_verified=True,
        online_fields_removed=['nco_nodes', 'nco_node_mask'])


@torch.no_grad()
def timing(args, device):
    lock, config = authorize_test(args)
    saved = read_json(args.root / args.arm / 'test_results.json')
    if (saved.get('complete') is not True or saved.get('state_hash') != lock['state_hash']
            or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise PermissionError('R102 timing follows completed locked TEST, never selection')
    path = args.root / args.arm / 'inference_timing.json'
    if path.exists():
        from .r102_reporting import inference_timing
        inference_timing(path.parent, lock)
        return read_json(path)
    selector, state = load_checkpoint(args.root / args.arm / 'best.pt', device)
    del state
    selector.eval()
    args.cache_root = Path(config['cache_root']) if args.cache_root is None else args.cache_root
    manifest = read_json(args.prepared_root / 'source_cache_manifest.json')
    module = _module(args, config['donor_provenance'])
    rows = []
    for problem in PROBLEMS:
        base, cached = engine.open_cache(args.cache_root, 'test', problem, manifest, args.data_root), None
        try:
            cached = module.open_cache(args.cache_root, 'test', problem, manifest, args.data_root)
            if base.size < 128 or cached.size != base.size:
                raise ValueError('R102 timing source rows are incomplete or misaligned')
            indices = torch.arange(128)
            rows.append(dict(problem=problem, instances=128,
                **measure_inputs(selector, base.batch(indices, device=device),
                    cached.batch(indices, device=device), device)))
        finally:
            base.close()
            if cached is not None:
                cached.close()
    value = dict(experiment='R102', arm=args.arm, variant=ARM_VARIANTS[args.arm], complete=True,
        completed_utc=utc(), state_hash=lock['state_hash'], checkpoint_sha256=lock['checkpoint_sha256'],
        inference_rule=INFERENCE_RULE, device=str(device), repetitions=3, warmup_batches=1,
        scope='first128 TEST rows/task; GPU-resident batch128 amortized synchronized wall time; not singleton',
        excludes='dataset I/O, host transfer, one-time cache preparation, decoder and route solving',
        raw_input_source='original_base_cache_without_both_nco_fields',
        includes_native_encoder=True, online_donor_verified=True, all18_donor_coverage=True,
        includes_solver_decoder=False, full_system_measured=False, end_to_end_measured=False,
        singleton_measured=False, cache_hit_is_total=False, model_reselection=False,
        preprocessing=read_json(args.root / args.arm / 'input_features_ready.json'),
        test_preprocessing=read_json(args.root / args.arm / 'test_input_features_ready.json'), per_problem=rows)
    write_json(path, value)
    return value


def describe(args):
    return dict(experiment='R102', arm=args.arm, variant=ARM_VARIANTS[args.arm], task_scope='full18',
        root=str(args.root), feature_root=str(feature_root(args)), routing_feature_root=str(routing_feature_root(args)),
        required_protocol_lock=required_lock_fields(), required_lock_paths=['data_root', 'prepared_root'],
        initialization=INITIALIZATION, trainable_parameter_count=PARAMETER_COUNT,
        projection_shapes={n: list(s) for n, s in PROJECTION_SHAPES.items()}, objective=objective_spec(args.arm),
        inference_rule=INFERENCE_RULE, seed=2, batch_size=BATCH_SIZE, max_updates=MAX_UPDATES,
        routing_donor='identical pinned pretrained R94B in BOTH arms',
        new_feature_problems=['TSP', 'ATSP'], reused_feature_problems=list(ROUTING),
        new_solver_runs=False, new_performance_labels=False, training=False, inference=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('prepare', 'train', 'test', 'timing', 'describe'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--feature-root', type=Path)
    result.add_argument('--routing-feature-root', type=Path)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='4090')
    result.add_argument('--large-micro-cap', type=int, default=352)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--wandb-mode', choices=('offline',), default='offline')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R102')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'prepared_root', 'data_root', 'cache_root', 'feature_root',
                 'routing_feature_root', 'protocol_lock'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command == 'describe':
        value = describe(args)
        print(json.dumps(value, indent=2))
        return value
    if args.command in ('train', 'test'):
        return engine.main(argv, integration=sys.modules[__name__])
    run = args.root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            device = torch.device(args.device)
            if device.type != 'cuda':
                raise ValueError('Formal R102 preparation/timing requires an explicitly allocated CUDA device')
            torch.set_num_threads(2)
            torch.set_num_interop_threads(2)
            return prepare(args, device) if args.command == 'prepare' else timing(args, device)
        except BaseException as error:
            status_update(args, 'failed', error=dict(type=type(error).__name__, message=str(error)),
                          automatic_retry=False)
            raise


if __name__ == '__main__':
    main()
