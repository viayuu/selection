"""Fresh R83 model-only selector with MoSES_RF removed, retaining BQ/LEHD."""

import argparse
import importlib.util
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import RDropObjective, file_hash, utc, write_json
from .r83_experiment import (SOURCE_SCOPE, initialize_shared as initialize_r83,
    load_bundle, validate_model_only_bundle)
from .solver_code_embeddings import bundle_provenance
from .solver_code_encoder import make_r45_model


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R126_R83_no_moses_rf'
DEFAULT_DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_no_moses_rf_seed2_20261112')
DEFAULT_PREPARED_ROOT = DEFAULT_ROOT / 'prepared'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
DEFAULT_EMBEDDING = V4 / 'runs/R83_model_only_source_embedding/model_only_vectors.pt'
ARM, ARMS = 'model_only_seed2', ('model_only_seed2',)
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES, EVAL_EVERY = 1024, 57600, 4320, 1440
MODEL_FACTORY = 'code.V4.r126_experiment.load_checkpoint'
WINNER_RULE = engine.WINNER_RULE
CHECKPOINT_RULE = 'strict minimum full18 VAL macro_vs_sbs_pct; no-MoSES_RF full TRAIN SBS; first wins ties'
INFERENCE_RULE = 'original utility score differences and maximin; no MoSES_RF; ascending-global-ID tie break'
CAPS = dict(large_micro_cap=256, eval_batch_size=128, other_micro_cap=1024)


def micro_caps(gpu_type='4090', large_cap=None):
    cap = CAPS['large_micro_cap'] if large_cap is None else large_cap
    if gpu_type not in ('4090', '3090') or not 1 <= cap <= 352:
        raise ValueError('Use a safe explicit microbatch; the logical batch remains1024')
    return {p: cap if p in ('TSP', 'CVRP') else CAPS['other_micro_cap'] for p in PROBLEMS}


def prepared_config(args):
    from .r126_data import prepared_config as load_prepared
    return load_prepared(args)


def initialize_shared(initial, geometry, arm, device='cpu'):
    if arm != ARM:
        raise ValueError('R126 trains one fresh R83 group, not a continued selector')
    model, params, receipt = initialize_r83(initial, geometry, device=device,
        bundle=load_bundle(DEFAULT_EMBEDDING))
    receipt.update(experiment='R126', architecture_reference='R83 model-only source embedding',
        source_scope=SOURCE_SCOPE, removed_candidates=['MoSES_RF'], retained_candidates=['BQ', 'LEHD'],
        external_embedding_recomputed=False,
        geometry_fitted_on='unchanged frozen81000 TRAIN input rows only',
        trainable_parameter_count=sum(p.numel() for p in model.parameters()))
    return model, params, receipt


def make_objective(selector, arm):
    if arm != ARM:
        raise ValueError('Unknown R126 group')
    return RDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not RDropObjective:
        raise TypeError('Use the original stateless two-forward R-Drop objective')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('R126 does not introduce a teacher or mutable context memory')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem,
                 max_retries=20):
    if len(batch['ind']) != BATCH_SIZE:
        raise ValueError('Retain complete logical1024 batches and former MoSES_RF winner rows')
    return private_engine().update_batch(objective, optimizer, scaler, batch, weight,
        runtime, cap, max_retries=max_retries)


def parameter_count(params):
    with torch.random.fork_rng(devices=[]):
        model = make_r45_model(params, bundle=load_bundle(DEFAULT_EMBEDDING))
    return sum(p.numel() for p in model.parameters())


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    bundle = load_bundle(DEFAULT_EMBEDDING)
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R126', model_factory=MODEL_FACTORY, initialization='scratch_shared_r77',
        architecture_reference='R83 model-only source embedding plus original complete dual stream',
        source_scope=SOURCE_SCOPE, model_only_scope=bundle['model_only_scope'],
        role_policy=dict(encoder=True, decision=True, inference=False, config=False),
        role_policy_exception='GLOP insertion deployment has no neural model and uses its original ID only',
        removed_candidates=['MoSES_RF'], retained_candidates=['BQ', 'LEHD'],
        candidate_filter_applied_to='forward, all losses, native winner, ranking and evaluation',
        source_dataset_root=manifest['source_data_root'], unchanged_instance_membership=True,
        data_preparation=ready, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        source_vectors=True, frozen_embedding_path=str(DEFAULT_EMBEDDING.resolve()),
        frozen_embedding_file_sha256=file_hash(DEFAULT_EMBEDDING),
        embedding_provenance=bundle_provenance(bundle), train_eval_every_updates=7200,
        selection_head='original_score_mlp_maximin', trainable_parameter_count=parameter_count(params),
        objective=dict(CE=.35, comparison=.35, relative_risk=.02, RDrop_final=.35, forward_passes=2),
        nss_reference='aligned saved historical NSS scores restricted to the same pool; no retraining',
        historical_comparison_boundary='R83 used older70k data; R125 uses four source roles and a different pool; neither is a matched single-factor control',
        interpretation='removing a candidate changes the Oracle; compare against same-pool TRAIN-fitted baselines',
        new_solver_runs=False, R58_operations=False, new_instances=False, external_embedding_recomputed=False,
        source_sha256={name: file_hash(V4 / name) for name in (
            'r126_experiment.py', 'r126_data.py', 'r83_experiment.py', 'r84_experiment.py',
            'r80_experiment.py', 'solver_code_encoder.py', 'solver_code_embeddings.py',
            'pairwise_selector.py', 'pairwise_objective.py', 'r76_experiment.py', 'r77_experiment.py')})
    validate_config(config)
    return config


def validate_config(config):
    if (config.get('experiment') != 'R126' or config.get('arm') != ARM
            or config.get('seed') != 2 or config.get('model_factory') != MODEL_FACTORY
            or config.get('solver_names') != GLOBAL_SOLVERS
            or config.get('removed_candidates') != ['MoSES_RF']
            or config.get('retained_candidates') != ['BQ', 'LEHD']
            or config.get('global_batch_size') != BATCH_SIZE
            or config.get('max_successful_updates') != MAX_UPDATES
            or config.get('checkpoint_rule') != CHECKPOINT_RULE
            or config.get('inference_rule') != INFERENCE_RULE
            or config.get('source_scope') != SOURCE_SCOPE
            or config.get('role_policy') != dict(encoder=True, decision=True, inference=False, config=False)
            or config.get('model_params', {}).get('solver_representation') != 'code'
            or config.get('loss') != engine.LOSS_RECIPE):
        raise ValueError('Checkpoint differs from the fixed fresh R126 R83 model-only protocol')
    for problem in PROBLEMS:
        names, ids = config['candidate_pools'][problem], config['candidate_pool_ids'][problem]
        if 'MoSES_RF' in names or ids != sorted(set(ids)) or len(names) != len(ids):
            raise ValueError('Removed method or altered stable candidate order in checkpoint')
        if problem in ('TSP', 'CVRP') and not {'BQ', 'LEHD'} <= set(names):
            raise ValueError('R126 must retain BQ and LEHD in both of their legal task pools')


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    validate_config(config)
    if state.get('state_hash') != engine.json_hash(config):
        raise ValueError('Checkpoint protocol changed after selection')
    prefix = 'solver_encoder.'
    validate_model_only_bundle(dict(config['embedding_provenance'], **{
        key: state['model'][prefix + key] for key in
        ('code_vectors', 'role_mask', 'chunk_vectors', 'chunk_indices')
    }, vectors=state['model'][prefix + 'code_vectors']))
    model = make_r45_model(config['model_params'], state_dict=state['model'])
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


@lru_cache(maxsize=1)
def private_engine():
    from .r126_data import open_cache
    name = 'code.V4._r126_private_engine'
    spec = importlib.util.spec_from_file_location(name, engine.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.micro_caps, module.open_cache = micro_caps, open_cache
    return module


def evaluate(*args, **kwargs):
    return private_engine().evaluate(*args, **kwargs)


def authorize_test(args):
    lock, config = private_engine().authorize_test(args)
    validate_config(config)
    return lock, config


def ensure_test_cache(args, lock):
    verified, _ = authorize_test(args)
    if verified != lock:
        raise PermissionError('TEST uses only the unique VAL-best checkpoint')
    _, manifest, _, _ = prepared_config(args)
    if set(manifest['splits'].get('test', {})) != set(PROBLEMS):
        raise PermissionError('Use the existing complete frozen TEST inputs')
    return manifest


def train(args, device):
    return private_engine().train(args, device, integration=sys.modules[__name__])


def test(args, device):
    return private_engine().test(args, device, integration=sys.modules[__name__])


def describe(args):
    return dict(experiment='R126', arm=ARM, seed=2, batch=BATCH_SIZE,
        max_updates=MAX_UPDATES, warmup=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        data_root=str(args.data_root), architecture='R83 model-only source embedding plus original dual stream',
        removed_candidates=['MoSES_RF'], retained_candidates=['BQ', 'LEHD'], new_solver_runs=False)


def timing(args, device):
    from .pairwise_objective import inputs_only
    from .r126_data import open_cache
    lock, _ = authorize_test(args)
    _, manifest, _, _ = prepared_config(args)
    model, _ = load_checkpoint(args.root / args.arm / 'best.pt', device)
    model.eval()
    rows = []
    for problem in PROBLEMS:
        cache = open_cache(args.cache_root, 'val', problem, manifest, args.data_root)
        try:
            count = min(128, cache.size)
            batch = inputs_only(cache.batch(np.arange(count), device=device))
            with torch.no_grad():
                model(batch)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                for _ in range(5):
                    model(batch)
                torch.cuda.synchronize(device)
            seconds = (time.perf_counter() - start) / 5
            rows.append(dict(problem=problem, batch=count, seconds=seconds, ms_per_instance=seconds * 1000 / count))
        finally:
            cache.close()
    result = dict(experiment='R126', checkpoint_sha256=lock['checkpoint_sha256'],
        includes='source lookup and learned projections, complete instance/method interaction, decoder and score head',
        excludes='disk input staging; no solver or external Embedding API at inference',
        per_problem=rows, device=torch.cuda.get_device_name(device), updated_utc=utc())
    write_json(args.root / args.arm / 'inference_timing.json', result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'timing', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, default=ARM)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('4090', '3090'), default='4090')
    result.add_argument('--large-micro-cap', type=int, default=CAPS['large_micro_cap'])
    result.add_argument('--eval-batch-size', type=int, default=CAPS['eval_batch_size'])
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R126')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'timing':
        torch.set_num_threads(2)
        from .train import configure_torch
        configure_torch()
        return timing(args, torch.device(args.device))
    return private_engine().main(argv, integration=sys.modules[__name__])


if __name__ == '__main__':
    main()
