"""R85 NEW90k objectives on the unchanged R77 selector and maximin inference.

Reuses the completed R84 preparation and its training engine through explicit
callbacks. No cache construction, solver execution or training launch occurs
when importing this module.
"""

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as engine
from .r76_experiment import (
    PART_KEYS, Runtime, capture_rank_rng, file_hash, read_json, restore_rank_rng,
    slice_batch, write_json,
)
from .r77_data import parameter_hash
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .solver_code_encoder import make_r45_model


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R85_zero_selection_research'
ARMS = ('A_ovr_control', 'B_gradient_balance', 'C_cost_profile', 'D_candidate_subset')
BATCH_SIZE, EVAL_EVERY = engine.BATCH_SIZE, engine.EVAL_EVERY
WARMUP_UPDATES, MAX_UPDATES = engine.WARMUP_UPDATES, engine.MAX_UPDATES
PEAK_LR, END_LR = engine.PEAK_LR, engine.END_LR
WINNER_RULE, MODEL_FACTORY = engine.WINNER_RULE, 'code.V4.r85_experiment.load_checkpoint'
learning_rate, consistency_weight = engine.learning_rate, engine.consistency_weight
micro_caps, task_order, json_hash = engine.micro_caps, engine.task_order, engine.json_hash
status_update, prepared_config = engine.status_update, engine.prepared_config
ARM_OBJECTIVES = {
    'A_ovr_control': dict(classification='OVR BCE on raw utility', coefficient=.35,
        target='one-hot saved native winner', class_weights='none',
        reduction='per-row valid-method mean times log(K)/log(2), then mean real rows'),
    'B_gradient_balance': dict(classification='OVR BCE on raw utility', coefficient=.35,
        target='one-hot saved native winner', class_weights='per-problem/method EQLv2',
        initial_positive_negative_gradient_ratio=100., gamma=12., mu=.8, alpha=4., epsilon=1e-10,
        reduction='per-row valid-method mean times log(K)/log(2), then mean real rows',
        state_commit='successful update only'),
    'C_cost_profile': dict(classification='original native CE', coefficient=.35,
        auxiliary='KL(q_nonwinner || p_nonwinner)', auxiliary_coefficient=.35,
        teacher='softmax(-FP64-derived relative gap / .01), legal nonwinners only',
        student='softmax(final maximin scores), legal nonwinners only',
        excluded='saved native winner', minimum_legal_candidates=3, temperature=.01),
    'D_candidate_subset': dict(classification='original native CE', coefficient=.35,
        auxiliary='CE on the actual reduced-candidate interaction context',
        auxiliary_coefficient=.35, subset_size='uniform integer 2..K-1',
        subset_refresh='once per logical batch; fixed through microbatches/AMP retries',
        target='restricted native FP64 comparison labels; global-ID tie break',
        forced_winner_or_method_quota=False),
}


def make_model(params, arm, state_dict=None):
    if arm not in ARMS:
        raise ValueError('Unknown R85 arm')
    if arm == 'D_candidate_subset':
        from .r85_candidate_selector import make_candidate_model
        return make_candidate_model(params, state_dict=state_dict)
    return make_r45_model(params, state_dict=state_dict)


def initialize_shared(initial, geometry, arm, device='cpu'):
    return engine.initialize_shared(initial, geometry, arm, device, model_factory=make_model)


def make_objective(selector, arm):
    module = importlib.import_module('code.V4.r85_objectives')
    if arm in ('A_ovr_control', 'B_gradient_balance'):
        objective = module.OVRRDropObjective(selector,
            mode='natural' if arm == 'A_ovr_control' else 'eql_v2')
    elif arm == 'C_cost_profile':
        objective = module.CostProfileKLRDropObjective(selector)
    elif arm == 'D_candidate_subset':
        objective = module.CandidateSubsetRDropObjective(selector)
    else:
        raise ValueError('Unknown R85 arm')
    for name in ('begin_attempt', 'accumulate', 'finish_attempt',
                 'training_state_dict', 'load_training_state_dict'):
        if not callable(getattr(objective, name, None)):
            raise TypeError('R85 objective must expose ' + name)
    return objective


def _cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_state(item) for item in value)
    return copy.deepcopy(value)


def checkpoint_extras(objective):
    return dict(objective_state=_cpu_state(objective.training_state_dict()))


def restore_objective(objective, checkpoint):
    if 'objective_state' not in checkpoint:
        raise ValueError('R85 resume requires the objective state from the same successful update')
    if ('successful_updates' in checkpoint and checkpoint['objective_state'].get('successful_steps')
            != checkpoint['successful_updates']):
        raise ValueError('Objective state does not match the checkpoint successful-update boundary')
    objective.load_training_state_dict(checkpoint['objective_state'])


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap,
                 max_retries=20, *, problem):
    """One logical context; failed AMP attempts discard all pending EQL state."""
    count = len(batch['ind'])
    if runtime.world_size != 1 or count <= 0 or cap <= 0 or problem not in PROBLEMS:
        raise ValueError('R85 requires a real, single-task batch on one assigned device')
    rng = capture_rank_rng(runtime.device)
    for retry in range(max_retries):
        restore_rank_rng(rng, runtime.device)
        optimizer.zero_grad(set_to_none=True)
        attempt_batch = objective.begin_attempt(batch, weight)
        if not isinstance(attempt_batch, dict) or len(attempt_batch['ind']) != count:
            raise ValueError('begin_attempt must return the complete logical batch with its frozen context')
        parts_sum = torch.zeros(len(PART_KEYS) - 1, dtype=torch.float64, device=runtime.device)
        bad = torch.zeros((), dtype=torch.int32, device=runtime.device)
        for start in range(0, count, cap):
            stop = min(start + cap, count)
            loss, parts = objective(slice_batch(attempt_batch, start, stop), weight)
            scaler.scale(loss * ((stop - start) / count)).backward()
            bad |= (~torch.isfinite(loss.detach())).to(torch.int32)
            parts_sum.add_(parts.detach().double(), alpha=stop - start)
            objective.accumulate(stop - start)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.)
        bad |= (~torch.isfinite(grad_norm)).to(torch.int32)
        if bad.item():
            objective.finish_attempt(False)
            if scaler.is_enabled():
                scaler.update(new_scale=scaler.get_scale() * .5)
            continue
        scaler.step(optimizer)
        scaler.update()
        metrics = objective.finish_attempt(True)
        if set(metrics) & set(PART_KEYS):
            raise ValueError('Auxiliary metrics must not overwrite the original objective components')
        metrics = {key: torch.as_tensor(value, device=runtime.device).detach()
                   for key, value in metrics.items()}
        if any(value.numel() != 1 or not torch.isfinite(value).item() for value in metrics.values()):
            raise ValueError('R85 objective metrics must be finite detached scalars')
        return torch.cat((parts_sum / count, grad_norm.detach().reshape(1))), retry, metrics
    raise RuntimeError('Nonfinite gradients persisted; no successful EQL/context commit or skipped rows')


def protocol_lock(args):
    path = args.protocol_lock or args.root / 'protocol_lock.json'
    record = read_json(path)
    expected = dict(experiment='R85', locked=True, arms=list(ARMS), seed=2,
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES, warmup_updates=WARMUP_UPDATES,
        eval_every=EVAL_EVERY)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('R85 requires main\'s final four-arm protocol lock before training')
    return dict(path=str(path.resolve()), sha256=file_hash(path), record=record)


def protocol(args, ready, manifest, geometry, baselines, params, order, caps):
    config = engine.protocol(args, ready, manifest, geometry, baselines, params, order, caps)
    config.update(experiment='R85', model_factory=MODEL_FACTORY,
        objective=copy.deepcopy(ARM_OBJECTIVES[args.arm]), main_protocol_lock=protocol_lock(args),
        historical_reference=dict(model='R84A', data_match='exact same NEW90k release',
            run=str((engine.DEFAULT_ROOT / 'A_baseline').resolve()), retrained=False),
        objective_state='atomic objective_state alongside model, optimizer, scaler and RNG',
        gradients='real micro rows / logical1024; one clip/step; same subset and frozen EQL weights across AMP retries',
        source_sha256={name: file_hash(V4 / name) for name in
            ('r85_experiment.py', 'r85_objectives.py', 'r85_candidate_selector.py',
             'r84_experiment.py', 'r84_prepare.py', 'pairwise_selector.py',
             'pairwise_objective.py', 'r76_experiment.py', 'r77_experiment.py')},
        nss_reference='reuse R84 exact4500 predictions from locked R79 weights; validate population/pool/hash')
    config['loss'].pop('focal_extra', None)
    config['loss'].update(classification=copy.deepcopy(ARM_OBJECTIVES[args.arm]),
        route_auxiliary=False, focal=False,
        ce_component='native maximin CE; diagnostic only for OVR arms A/B')
    return config


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R85' or config.get('arm') not in ARMS
            or config.get('seed') != 2 or config.get('initialization') != 'scratch_shared_r77'
            or config.get('solver_names') != GLOBAL_SOLVERS or state.get('state_hash') != json_hash(config)
            or config.get('max_successful_updates') != MAX_UPDATES):
        raise ValueError('Checkpoint is not an unchanged fresh R85 state')
    model = make_model(config['model_params'], config['arm'], state_dict=state['model'])
    return model.to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def selection_counts(archive, prefix, problem, split):
    winner, pred = np.asarray(archive[prefix + 'winner']), np.asarray(archive[prefix + 'pred'])
    pool = archive[prefix + 'pool'].astype(str).tolist()
    rows = []
    for index, solver in enumerate(pool):
        truth, selected = winner == index, pred == index
        correct = int((truth & selected).sum())
        support, chosen = int(truth.sum()), int(selected.sum())
        rows.append(dict(problem=problem, split=split.upper(), solver=solver, n=len(winner),
            winner_count=support, selected_count=chosen, correct_count=correct,
            missed_winner_count=support - correct, false_pick_count=chosen - correct,
            winner_share_pct=100. * support / len(winner), selected_share_pct=100. * chosen / len(winner),
            recall_pct=100. * correct / support if support else None,
            precision_pct=100. * correct / chosen if chosen else None))
    return rows


def zero_summary(rows):
    problems = sorted({row['problem'] for row in rows}, key=PROBLEMS.index)
    values = {}
    for problem in problems:
        current = [row for row in rows if row['problem'] == problem]
        zero = [row for row in current if row['winner_count'] > 0 and row['selected_count'] == 0]
        near = [row for row in current if row['winner_count'] > 0 and row['selected_share_pct'] <= .1]
        values[problem] = dict(n=current[0]['n'], zero_with_winner_groups=len(zero),
            zero_winner_mass=sum(row['winner_count'] for row in zero),
            near_zero_with_winner_groups=len(near), near_zero_winner_mass=sum(row['winner_count'] for row in near),
            total_missed_winner_count=sum(row['missed_winner_count'] for row in current),
            native_correct=sum(row['correct_count'] for row in current))
    return dict(per_problem=values, all={key: sum(value[key] for value in values.values()) for key in
        ('n', 'zero_with_winner_groups', 'zero_winner_mass', 'near_zero_with_winner_groups',
         'near_zero_winner_mass', 'total_missed_winner_count', 'native_correct')},
        definition='zero group: native winner_count > 0 and selected_count == 0; winner mass is not all misses')


def evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256=None):
    result = engine.evaluate(selector, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha256)
    if 'zero_selection' not in result:
        rows = []
        with np.load(result['prediction_archive'], allow_pickle=False) as archive:
            for problem in PROBLEMS:
                rows.extend(selection_counts(archive, problem + '__', problem, split))
        result['zero_selection'] = zero_summary(rows)
        directory = args.root / args.arm / 'evaluation' / split / tag
        write_json(directory / 'method_selection.json', dict(methods=rows, summary=result['zero_selection']))
        write_json(directory / 'metrics.json', result)
    return result


def authorize_test(args):
    return engine.authorize_test(args)


def ensure_test_cache(args, lock):
    """R84 completed TEST features are shared read-only; never rebuild here."""
    manifest = read_json(args.prepared_root / 'source_cache_manifest.json')
    records = manifest['splits'].get('test', {})
    if set(records) != set(PROBLEMS):
        raise ValueError('R85 requires the already completed full18 R84 TEST cache')
    for problem in PROBLEMS:
        metadata = read_json(args.cache_root / 'test' / problem / 'metadata.json')
        if (records[problem]['rows'] != SPLIT_ROWS['test'] or not metadata.get('complete')
                or metadata.get('rows') != SPLIT_ROWS['test']):
            raise ValueError('Incomplete existing R84 TEST cache: ' + problem)
    return manifest


def train(args, device):
    protocol_lock(args)
    return engine.train(args, device, integration=sys.modules[__name__])


def test(args, device):
    return engine.test(args, device, integration=sys.modules[__name__])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('train', 'test', 'describe', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    result.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--protocol-lock', type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--gpu-type', choices=('4090', '3090'), default='4090')
    result.add_argument('--large-micro-cap', type=int)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--log-every', type=int, default=18)
    result.add_argument('--wandb-mode', choices=('auto', 'online', 'offline'), default='auto')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--state-hash', '--expected-state-hash')
    result.set_defaults(experiment='R85')
    return result


def describe(args):
    return dict(experiment='R85', arm=args.arm, objective=ARM_OBJECTIVES[args.arm],
        rows=SPLIT_ROWS, seed=2, batch_size=BATCH_SIZE, max_updates=MAX_UPDATES,
        eval_every=EVAL_EVERY, warmup_updates=WARMUP_UPDATES,
        model_factory=MODEL_FACTORY, inference='original R77 score MLP then maximin',
        data_root=str(args.data_root), prepared_root=str(args.prepared_root),
        microbatch_caps=micro_caps(args.gpu_type, args.large_micro_cap),
        training_requires='main-locked protocol_lock.json; code readiness is not a GPU launch')


def main(argv=None):
    return engine.main(argv, integration=sys.modules[__name__])


if __name__ == '__main__':
    main()
