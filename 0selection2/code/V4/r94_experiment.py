"""R94: full18 selection with frozen, early-fused route-pretrained node features.

Only the donor differs between the two freshly initialized selectors. The
private R84 engine below keeps active historical runners and their globals
untouched. Cached features contain inputs only, never route/cost targets.
"""

import argparse
import copy
import importlib.util
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import PART_KEYS, file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r85_experiment import ensure_test_cache, selection_counts, zero_summary


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R94_pretrained_node_fusion'
ARMS = NEURAL_ARMS = ('A', 'B')
VARIANTS = dict(A='random', B='pretrained')
BATCH_SIZE, MAX_UPDATES = engine.BATCH_SIZE, engine.MAX_UPDATES
WARMUP_UPDATES, EVAL_EVERY = engine.WARMUP_UPDATES, engine.EVAL_EVERY
PEAK_LR, END_LR = engine.PEAK_LR, engine.END_LR
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
WINNER_RULE, MODEL_FACTORY = engine.WINNER_RULE, 'code.V4.r94_experiment.load_checkpoint'
MODEL_ROLE_ARM, ROLE_SLOT = 'B_execution_anchor', 3
SELECTION_HEAD = 'original_score_mlp_maximin'
prepared_config, micro_caps = engine.prepared_config, engine.micro_caps
json_hash, task_order, learning_rate = engine.json_hash, engine.task_order, engine.learning_rate
status_update = engine.status_update
INFERENCE_RULE = ('eval frozen native static encoder on exact real nodes; early node fusion; '
    'original four-layer instance processor, two joint layers, score MLP/pair differences/maximin; '
    'stable ascending-global-solver-ID ties; TSP/ATSP raw-only; no calibration')


def feature_root(args):
    value = getattr(args, 'feature_root', None)
    return Path(value) if value is not None else Path('/tmp/shiys_R94_' + args.arm + '_node_features')


def donor_record(donor):
    value = donor.provenance
    value = value() if callable(value) else value
    if not isinstance(value, dict):
        raise TypeError('Frozen donor must expose explicit provenance')
    return copy.deepcopy(value)


def initialize_shared(initial, geometry, arm, device='cpu'):
    from .r94_model import make_model
    reference = initial['trainable']
    if (arm not in ARMS or initial.get('seed') != 2
            or initial.get('initialization') != 'scratch'
            or parameter_hash(reference) != initial.get('trainable_sha256')):
        raise ValueError('R94 uses the frozen fresh R84 seed2 initialization, not a trained selector')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(940003)
        model = make_model(initial['model_params'], arm)
    actual = {name: value for name, value in model.named_parameters() if value.requires_grad}
    retained = {name: value for name, value in actual.items()
                if name in reference and value.shape == reference[name].shape}
    if set(retained) != set(reference):
        raise ValueError('R94 must retain every original selector parameter')
    added = {name: value for name, value in actual.items() if name not in retained}
    if not added or any(not name.startswith('instance_encoder.') for name in added):
        raise ValueError('Only the new pre-processor node-fusion projection may be trainable')
    with torch.no_grad():
        for name, value in retained.items():
            value.copy_(reference[name])
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    if any(value.requires_grad for value in model.donor.parameters()):
        raise ValueError('The external solver encoder must remain frozen')
    params = copy.deepcopy(model.params)
    params['r94_fresh_trainable_sha256'] = parameter_hash(actual)
    receipt = dict(seed=2, initialization='scratch_shared_r77_plus_node_fusion',
        historical_selector_checkpoint_loaded=False, external_solver_pretraining=arm == 'B',
        r77_fresh_trainable_sha256=initial['trainable_sha256'],
        common_parameter_sha256=parameter_hash(retained),
        trainable_sha256=parameter_hash(actual), new_parameter_sha256=parameter_hash(added),
        new_parameter_names=sorted(added), new_common_seed=940003,
        trainable_parameter_count=sum(value.numel() for value in actual.values()),
        frozen_parameter_count=sum(value.numel() for value in model.donor.parameters()),
        frozen_donor=donor_record(model.donor),
        geometry_fitted_on='frozen R84 new81000 TRAIN only')
    return model.to(device), params, receipt


def protocol_lock(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    record = read_json(path)
    required = dict(experiment='R94', locked=True, arms=list(ARMS), seed=2,
        task_scope='full18', rows_per_problem=SPLIT_ROWS, batch_size=BATCH_SIZE,
        logical_global_batch=BATCH_SIZE, max_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        large_micro_cap=352, eval_batch_size=256, donor_variants=VARIANTS)
    if any(record.get(name) != value for name, value in required.items()):
        raise ValueError('R94 requires the final paired early-fusion full18 protocol')
    for name in ('data_root', 'prepared_root'):
        if Path(record.get(name, '')).resolve() != getattr(args, name).resolve():
            raise ValueError('R94 lock names a different ' + name)
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def objective_spec(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R94 arm')
    return dict(main='unchanged R77 CE/comparison/risk plus symmetric R-Drop',
        coefficients=dict(ce=.35, comparison=.35, risk=.02),
        rdrop='linear ramp to .35 over4320 updates', forward_passes=2,
        teacher_gradients=False, teacher_auxiliary_loss=False,
        route_targets=False, cross_instance_objective=False, mutable_objective_state=False)


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    from .r94_model import SPEC
    if args.large_micro_cap != 352 or args.eval_batch_size != 256:
        raise ValueError('R94 retains per-task micros TSP/CVRP352, others1024; eval256')
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R94', r94_schema_version=1, model_factory=MODEL_FACTORY,
        initialization='scratch_shared_r77_plus_node_fusion',
        main_protocol_lock=protocol_lock(args), objective=objective_spec(args.arm),
        model_spec=copy.deepcopy(SPEC), role_arm=MODEL_ROLE_ARM,
        frozen_donor=dict(variant=VARIANTS[args.arm], frozen=True, eval_only=True,
            pretraining_source=('existing R48 pinned ReLD-MOEL solver encoder only'
                                if args.arm == 'B' else 'none: fixed random donor seed940002'),
            known_unknowns=('original pretraining instance manifest, training CLI and overlap are unavailable'
                            if args.arm == 'B' else 'not applicable: no pretrained weights loaded')),
        node_fusion_initialization_sha256=params['r94_fresh_trainable_sha256'],
        feature_cache_root=str(feature_root(args).resolve()),
        feature_precision='native static encoder FP32, per-node output stored FP16; same online quantization',
        feature_cache_identity='input cache identity + donor specification; no performance targets',
        dropout=.1, selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE,
        input_roles=dict(arm=MODEL_ROLE_ARM, problem='TSP', node_column=ROLE_SLOT,
            assignment='physical input node0', label_dependent=False),
        historical_reference=dict(primary='R89B', secondary='R84A', retrained=False,
            matched_control='R94A random frozen donor',
            external_pretraining_budget_matched_to_scratch=False),
        evidence_boundary='One early node-fusion mechanism, with a random-frozen donor control. External solver pretraining is additional knowledge/computation. Partial16-task donor coverage, raw TSP/ATSP paths; full18 final selector. Not R50 pool-first readout, a route-solving system, or a B-semantics repair.',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r94_experiment.py', 'r94_model.py', 'r94_objective.py', 'r94_donor.py',
             'r94_features.py', 'r84_experiment.py', 'r84_prepare.py', 'r88_model.py',
             'V4Model.py', 'pairwise_selector.py', 'pairwise_objective.py', 'r76_experiment.py')})
    return config


def make_objective(selector, arm):
    from .r94_objective import NodeFusionObjective
    objective_spec(arm)
    return NodeFusionObjective(selector)


def checkpoint_extras(objective):
    return {}


def restore_objective(objective, checkpoint):
    if 'objective_state' in checkpoint:
        raise ValueError('R94 has no mutable objective state')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem,
                 max_retries=20):
    if (problem not in PROBLEMS or len(batch['ind']) != BATCH_SIZE
            or cap != micro_caps('3090', 352)[problem]):
        raise ValueError('R94 logical batch and per-task micros changed')
    return engine.update_batch(objective, optimizer, scaler, batch, weight, runtime,
                               cap, max_retries=max_retries)


@lru_cache(maxsize=4)
def private_engine(arm, feature_directory, provenance_json=None):
    from .r94_donor import SUPPORTED_PROBLEMS
    from .r94_features import FrozenFeatureCache
    from .r94_model import inputs_with_donor
    name = 'code.V4._r94_engine_' + arm
    spec = importlib.util.spec_from_file_location(name, engine.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    def open_with_features(cache_root, split, problem, manifest, data_root, **kwargs):
        base = engine.open_cache(cache_root, split, problem, manifest, data_root, **kwargs)
        if problem not in SUPPORTED_PROBLEMS:
            return base
        try:
            return FrozenFeatureCache(base, Path(feature_directory) / split / problem,
                expected_variant=VARIANTS[arm],
                expected_provenance=json.loads(provenance_json) if provenance_json else None)
        except BaseException:
            base.close()
            raise

    module.open_cache, module.inputs_only = open_with_features, inputs_with_donor
    return module


def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash,
             checkpoint_sha256=None):
    if split == 'test':
        lock, _ = authorize_test(args)
        prepare_input_features(args, selector.donor, split='test', test_lock=lock)
    module = private_engine(args.arm, str(feature_root(args).resolve()),
                            json.dumps(donor_record(selector.donor), sort_keys=True))
    return module.evaluate(selector, args, manifest, baselines, split, tag, step,
                           state_hash, checkpoint_sha256)


def validate_checkpoint_config(config):
    from .r94_model import SPEC
    arm = config.get('arm')
    required = dict(experiment='R94', r94_schema_version=1, seed=2,
        initialization='scratch_shared_r77_plus_node_fusion', model_factory=MODEL_FACTORY,
        solver_names=list(GLOBAL_SOLVERS), rows_per_problem=SPLIT_ROWS,
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE,
        max_successful_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every_updates=EVAL_EVERY, eval_batch_size=256, optimizer='AdamW',
        weight_decay=1e-4, grad_clip=1., dropout=.1, rdrop_weight=.35,
        rdrop_ramp_updates=WARMUP_UPDATES, model_spec=SPEC,
        selection_head=SELECTION_HEAD, inference_rule=INFERENCE_RULE, winner_rule=WINNER_RULE)
    if arm not in ARMS or any(config.get(name) != value for name, value in required.items()):
        raise ValueError('Not an unchanged fresh R94 checkpoint')
    if (config.get('objective') != objective_spec(arm)
            or config.get('frozen_donor', {}).get('variant') != VARIANTS[arm]):
        raise ValueError('R94 donor or supervision changed')


def load_checkpoint(path, device='cpu'):
    from .r94_model import make_model
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_checkpoint_config(config)
    if state.get('state_hash') != json_hash(config) or 'objective_state' in state:
        raise ValueError('R94 checkpoint protocol or stateless objective changed')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


def authorize_test(args):
    lock, config = engine.authorize_test(args)
    validate_checkpoint_config(config)
    if lock.get('inference_rule') != config['inference_rule']:
        raise PermissionError('R94 TEST inference differs from the unique VAL lock')
    return lock, config


def prepare_input_features(args, donor, *, split=None, test_lock=None):
    from .r94_features import prepare_features
    splits = (split,) if split is not None else ('train', 'val')
    for name in splits:
        prepare_features(cache_root=args.cache_root, feature_root=feature_root(args),
            split=name, donor=donor, device=next(donor.parameters()).device,
            batch_size=128, test_lock=test_lock,
            manifest=args.prepared_root / 'source_cache_manifest.json', data_root=args.data_root)


def prepare(args, device):
    from .r94_donor import StaticReLDDonor
    protocol_lock(args)
    prepared_config(args)
    started = time.monotonic()
    status_update(args, 'input_feature_preparation', 0,
        feature_root=str(feature_root(args)), decoder_executed=False, splits=['train', 'val'])
    donor = StaticReLDDonor(VARIANTS[args.arm]).to(device)
    prepare_input_features(args, donor)
    write_json(args.root / args.arm / 'input_features_ready.json', dict(experiment='R94',
        arm=args.arm, complete=True, completed_utc=utc(), feature_root=str(feature_root(args)),
        frozen_donor=donor_record(donor), splits=['train', 'val'],
        seconds=time.monotonic() - started,
        labels_used=False, decoder_executed=False, new_cost_labels=False))
    status_update(args, 'input_features_complete', 0, feature_root=str(feature_root(args)),
                  seconds=time.monotonic() - started)


def train(args, device):
    protocol_lock(args)
    ready = read_json(args.root / args.arm / 'input_features_ready.json')
    if not ready.get('complete') or ready.get('arm') != args.arm:
        raise ValueError('Prepare frozen input-only TRAIN/VAL features first')
    return private_engine(args.arm, str(feature_root(args).resolve()),
        json.dumps(ready['frozen_donor'], sort_keys=True)).train(
        args, device, integration=sys.modules[__name__])


def test(args, device):
    return engine.test(args, device, integration=sys.modules[__name__])


@torch.no_grad()
def timing(args, device):
    """Fixed-input latency diagnostic, never a new model-selection criterion."""
    from .r94_model import inputs_with_donor
    lock, config = authorize_test(args)
    saved_test = read_json(args.root / args.arm / 'test_results.json')
    if (saved_test.get('complete') is not True
            or saved_test.get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise PermissionError('R94 latency measurement follows the complete locked TEST')
    destination = args.root / args.arm / 'inference_timing.json'
    if destination.exists():
        saved = read_json(destination)
        if not saved.get('complete') or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']:
            raise ValueError('Preserve an existing incompatible timing receipt')
        return saved
    selector, state = load_checkpoint(args.root / args.arm / 'best.pt', device)
    del state
    selector.eval()
    manifest = read_json(args.prepared_root / 'source_cache_manifest.json')
    module = private_engine(args.arm, str(feature_root(args).resolve()),
                            json.dumps(donor_record(selector.donor), sort_keys=True))
    rows = []
    for problem in PROBLEMS:
        cache = module.open_cache(args.cache_root, 'test', problem, manifest, args.data_root)
        try:
            count = min(128, cache.size)
            # Inputs only. These fixed rows do not select a model or its rule.
            batch = cache.batch(torch.arange(count), device=device)
            raw, cached = engine.inputs_only(batch), inputs_with_donor(batch)
            measurements = {}
            for name, query in (('donor_plus_selector', raw), ('cache_hit_selector', cached)):
                selector(query)
                torch.cuda.synchronize(device)
                samples = []
                for _ in range(3):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    result = selector(query)
                    torch.cuda.synchronize(device)
                    samples.append(time.perf_counter() - start)
                    if not torch.isfinite(result['logits'][result['solver_mask']]).all():
                        raise ValueError('Nonfinite locked inference in timing diagnostic')
                measurements[name] = dict(batch_seconds=samples,
                    mean_seconds_per_instance=sum(samples) / (len(samples) * count))
            rows.append(dict(problem=problem, instances=count, **measurements))
        finally:
            cache.close()
    value = dict(experiment='R94', arm=args.arm, complete=True, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        device=str(device), repetitions=3, warmup_batches=1,
        scope='same first128 TEST input rows per task; inputs already on GPU; synchronized wall time',
        includes_native_encoder=True, includes_solver_decoder=False,
        excludes='dataset I/O and one-time feature cache preparation; cache-hit latency is not full inference',
        pretraining_budget_matched_to_scratch=False, per_problem=rows,
        inference_rule=config['inference_rule'], model_reselection=False)
    write_json(destination, value)
    return value


def describe(args):
    return dict(experiment='R94', arm=args.arm, donor=VARIANTS[args.arm],
        scope='full18', rows=SPLIT_ROWS, batch_size=BATCH_SIZE, seed=2,
        max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        objective=objective_spec(args.arm), inference_rule=INFERENCE_RULE,
        new_solver_runs=False, new_performance_labels=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('prepare', 'train', 'test', 'timing', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--feature-root', type=Path)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--gpu-type', choices=('3090', '4090'), default='3090')
    result.add_argument('--large-micro-cap', type=int, default=352)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='offline')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R94')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command not in ('prepare', 'timing'):
        return engine.main(argv, integration=sys.modules[__name__])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Formal feature preparation uses only a launcher-authorized GPU')
    return prepare(args, device) if args.command == 'prepare' else timing(args, device)


if __name__ == '__main__':
    main()
