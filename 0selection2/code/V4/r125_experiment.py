"""One fresh classic source-embedding selector without BQ/LEHD candidates."""

import argparse
import importlib.util
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import RDropObjective, file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r80_experiment import initialize_shared as initialize_source
from .solver_code_embeddings import bundle_provenance, load_bundle
from .solver_code_encoder import make_r45_model


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R125_no_bq_lehd_source_embedding'
DEFAULT_DATA_ROOT = Path('/public/home/shiys/selection_data_new90k_no_bq_lehd_seed2_20261112')
DEFAULT_PREPARED_ROOT = DEFAULT_ROOT / 'prepared'
DEFAULT_CACHE_ROOT = Path('/tmp/shiys_R84_new90k_cache')
DEFAULT_EMBEDDING = V4 / 'runs/R45_solver_code_embeddings/solver_code_vectors_atlas.pt'
ARM = 'code_seed2'
ARMS = (ARM,)
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES, EVAL_EVERY = 1024, 57600, 4320, 1440
MODEL_FACTORY = 'code.V4.r125_experiment.load_checkpoint'
WINNER_RULE = engine.WINNER_RULE
CHECKPOINT_RULE = 'strict minimum full18 VAL macro_vs_sbs_pct; reduced-pool full TRAIN SBS; first wins ties'
INFERENCE_RULE = 'original utility score differences and maximin; reduced candidates; ascending-global-ID tie break'
CAPS = dict(large_micro_cap=256, eval_batch_size=128, other_micro_cap=1024)


def micro_caps(gpu_type='4090', large_cap=None):
    large = CAPS['large_micro_cap'] if large_cap is None else large_cap
    if gpu_type not in ('4090', '3090') or not 1 <= large <= 352:
        raise ValueError('Use an explicit safe microbatch cap, with logical batch1024 unchanged')
    return {p: large if p in ('TSP', 'CVRP') else CAPS['other_micro_cap'] for p in PROBLEMS}


def prepared_config(args):
    from .r125_data import prepared_config as load_prepared
    return load_prepared(args)


def initialize_shared(initial, geometry, arm, device='cpu'):
    if arm != ARM:
        raise ValueError('R125 owns exactly one fresh source-embedding group')
    model, params, receipt = initialize_source(initial, geometry, 'D_code_embedding',
        device=device, bundle=load_bundle(DEFAULT_EMBEDDING))
    receipt.update(experiment='R125', architecture_reference='R80D classic four-view source embedding',
        geometry_fitted_on='unchanged frozen new81000 TRAIN inputs only',
        removed_candidates=['BQ', 'LEHD'], external_embedding_recomputed=False,
        trainable_parameter_count=sum(p.numel() for p in model.parameters()))
    return model, params, receipt


def make_objective(selector, arm):
    if arm != ARM:
        raise ValueError('Unknown R125 group')
    return RDropObjective(selector)


def checkpoint_extras(objective):
    if type(objective) is not RDropObjective:
        raise TypeError('Only the unchanged original two-forward R-Drop objective is allowed')
    return {}


def restore_objective(objective, checkpoint):
    checkpoint_extras(objective)
    if 'objective_state' in checkpoint:
        raise ValueError('No teacher, context memory, or mutable objective state belongs to R125')


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap, *, problem,
                 max_retries=20):
    if len(batch['ind']) != BATCH_SIZE:
        raise ValueError('Keep the full logical batch, including samples formerly won by removed methods')
    removed_ids = {GLOBAL_SOLVERS.index(name) for name in ('BQ', 'LEHD')}
    if any(int(value) in removed_ids for value in batch['pool_ids']):
        raise ValueError('Removed methods must not enter the forward candidate set')
    return private_engine().update_batch(objective, optimizer, scaler, batch, weight,
        runtime, cap, max_retries=max_retries)


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config['loss'].pop('focal_extra', None)
    config.update(experiment='R125', model_factory=MODEL_FACTORY, initialization='scratch_shared_r77',
        architecture_reference='R80D classic source embedding, not an expert or pretrained selector',
        removed_candidates=['BQ', 'LEHD'], candidate_filter_applied_to='forward, all losses, ranking and evaluation',
        source_dataset_root=manifest['source_data_root'], unchanged_instance_membership=True,
        data_preparation=ready, checkpoint_rule=CHECKPOINT_RULE, inference_rule=INFERENCE_RULE,
        source_vectors=True, frozen_embedding_path=str(DEFAULT_EMBEDDING.resolve()),
        frozen_embedding_file_sha256=file_hash(DEFAULT_EMBEDDING),
        embedding_provenance=bundle_provenance(load_bundle(DEFAULT_EMBEDDING)),
        train_eval_every_updates=7200, selection_head='original_score_mlp_maximin',
        objective=dict(CE=.35, comparison=.35, relative_risk=.02, RDrop_final=.35, forward_passes=2),
        trainable_parameter_count=sum(t.numel() for t in initialize_for_count(params).parameters()),
        nss_reference='same-pool fixed historical NSS scores only if saved aligned predictions exist; no retraining',
        historical_comparison_boundary='R80D has the same architecture but older data; not a matched causal control',
        interpretation='candidate deletion changes Oracle and task difficulty; Top1 alone cannot establish architecture-family imbalance',
        new_solver_runs=False, R58_operations=False, new_instances=False, external_embedding_recomputed=False,
        source_sha256={name: file_hash(V4 / name) for name in (
            'r125_experiment.py', 'r125_data.py', 'r84_experiment.py', 'r80_experiment.py',
            'solver_code_encoder.py', 'pairwise_selector.py', 'pairwise_objective.py',
            'r76_experiment.py', 'r77_experiment.py')})
    validate_config(config)
    return config


def initialize_for_count(params):
    # Counting must not consume the training RNG after the common initialization.
    with torch.random.fork_rng(devices=[]):
        return make_r45_model(params, bundle=load_bundle(DEFAULT_EMBEDDING))


def validate_config(config):
    if (config.get('experiment') != 'R125' or config.get('arm') != ARM
            or config.get('seed') != 2 or config.get('model_factory') != MODEL_FACTORY
            or config.get('solver_names') != GLOBAL_SOLVERS
            or config.get('removed_candidates') != ['BQ', 'LEHD']
            or config.get('global_batch_size') != BATCH_SIZE
            or config.get('max_successful_updates') != MAX_UPDATES
            or config.get('checkpoint_rule') != CHECKPOINT_RULE
            or config.get('inference_rule') != INFERENCE_RULE
            or config.get('model_params', {}).get('solver_representation') != 'code'
            or config.get('loss') != engine.LOSS_RECIPE):
        raise ValueError('Checkpoint is not the locked R125 source-embedding experiment')
    for problem in PROBLEMS:
        names, ids = config['candidate_pools'][problem], config['candidate_pool_ids'][problem]
        if set(names) & {'BQ', 'LEHD'} or ids != sorted(ids) or len(names) != len(ids):
            raise ValueError('Removed candidates or changed candidate ordering in checkpoint')


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_config(state['args'])
    if state.get('state_hash') != engine.json_hash(state['args']):
        raise ValueError('Checkpoint protocol changed after selection')
    model = make_r45_model(state['args']['model_params'], state_dict=state['model'])
    return model.to(device), state


load_model = from_checkpoint = load_checkpoint


@lru_cache(maxsize=1)
def private_engine():
    from .r125_data import open_cache
    name = 'code.V4._r125_private_engine'
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
    verified, config = authorize_test(args)
    if verified != lock:
        raise PermissionError('TEST must use the unique VAL-best model')
    ready, manifest, _, _ = prepared_config(args)
    if set(manifest['splits'].get('test', {})) != set(PROBLEMS):
        raise PermissionError('The frozen TEST input cache must already exist')
    return manifest


def train(args, device):
    return private_engine().train(args, device, integration=sys.modules[__name__])


def test(args, device):
    return private_engine().test(args, device, integration=sys.modules[__name__])


def describe(args):
    return dict(experiment='R125', arm=ARM, seed=2, batch=BATCH_SIZE,
        max_updates=MAX_UPDATES, warmup=WARMUP_UPDATES, eval_every=EVAL_EVERY,
        data_root=str(args.data_root), architecture='R80D classic source embedding + original dual stream',
        removed_candidates=['BQ', 'LEHD'], new_solver_runs=False)


def timing(args, device):
    from .pairwise_objective import inputs_only
    from .r125_data import open_cache
    lock, config = authorize_test(args)
    _, manifest, _, _ = prepared_config(args)
    model, _ = load_checkpoint(args.root / args.arm / 'best.pt', device)
    model.eval()
    rows = []
    for problem in PROBLEMS:
        cache = open_cache(args.cache_root, 'val', problem, manifest, args.data_root)
        try:
            count = min(128, cache.size)
            # Exclude input staging and every label from the timed forward.
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
    result = dict(experiment='R125', checkpoint_sha256=lock['checkpoint_sha256'],
        includes='frozen source lookup, learned projections, instance encoder, dual stream, decoder and score head',
        excludes='disk input staging; no solver execution or external Embedding API at inference',
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
    result.set_defaults(experiment='R125')
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
