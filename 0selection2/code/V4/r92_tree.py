"""Independent full18 R92 CatBoost arms, with one bank-wide VAL tree count.

Importing this module never trains, launches a worker, or opens a dataset.
The parent runner supplies tree_protocol/evaluate/authorize_test callbacks.
"""

import copy
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
import torch
from torch import nn

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .r76_experiment import file_hash, fixed_sbs_metrics, open_cache, read_json, save_checkpoint, utc, write_json
from .r77_data import parameter_hash
from .r80_experiment import json_hash, lock_best, wandb_run
from .r84_prepare import SPLIT_ROWS
from .r92_tree_features import COLUMNS, FEATURE_DEFINITION, features, input_batch


ARMS = ('C', 'D')
ITERATIONS, VAL_PERIOD = 3000, 100
TREE_PARAMETERS = dict(iterations=ITERATIONS, depth=8, learning_rate=.03, l2_leaf_reg=3.,
    border_count=128, random_seed=2, task_type='GPU', devices='0', thread_count=2,
    gpu_ram_part=.85, boosting_type='Plain', allow_writing_files=False, nan_mode='Forbidden')


def _parent():
    return importlib.import_module('code.V4.r92_experiment')


def _args(args, device):
    for name in ('root', 'prepared_root', 'data_root', 'cache_root'):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value))
    if args.arm not in ARMS:
        raise ValueError('Tree runner owns only R92 C and D')
    args.experiment, args.device = 'R92', str(device)
    return args


def _status(args, phase, **values):
    path = args.root / args.arm / 'status.json'
    record = read_json(path) if path.exists() else {}
    record.update(experiment='R92', arm=args.arm, phase=phase, pid=os.getpid(),
        host=socket.gethostname(), updated_utc=utc(), device=str(args.device),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'external allocation'),
        max_successful_updates=ITERATIONS, training_unit='trees_per_task',
        sample_exposures=0, nn_update_budget_matched=False)
    record.update(values)
    write_json(path, record)


def _config(args):
    config = copy.deepcopy(_parent().tree_protocol(args))
    if (config.get('experiment') != 'R92' or config.get('arm') != args.arm
            or config.get('seed') != 2 or config.get('max_successful_updates') != ITERATIONS
            or config.get('solver_names') != GLOBAL_SOLVERS
            or config.get('rows_per_problem') != SPLIT_ROWS):
        raise ValueError('Parent tree_protocol must lock full18 NEW90k and 3000 trees')
    for key in ('candidate_pools', 'candidate_pool_ids', 'baselines'):
        if set(config[key]) != set(PROBLEMS):
            raise ValueError('Tree bank requires all18 problems: ' + key)
    if any(b.get('selection_split') != 'train' for b in config['baselines'].values()):
        raise ValueError('Use TRAIN-fitted SBS for global VAL selection')
    config.update(feature_definition=FEATURE_DEFINITION,
        feature_source_sha256=file_hash(Path(__file__).with_name('r92_tree_features.py')),
        tree_source_sha256=file_hash(Path(__file__)),
        tree_parameters=dict(TREE_PARAMETERS, loss_function='MultiClass' if args.arm == 'C' else 'MultiRMSE'),
        tree_validation_period=VAL_PERIOD, training_unit='trees_per_task',
        tree_iteration_grid=list(range(VAL_PERIOD, ITERATIONS+1, VAL_PERIOD)),
        tree_selection='ONE common ntree_end for the entire full18 bank; strict full18 VAL macro_vs_sbs_pct; first exact tie',
        tree_training='full-dataset boosting; no SGD minibatches; not budget-matched to NN updates',
        tree_targets=('native FP64 ind, classes_count=full legal K, natural class frequency'
                      if args.arm == 'C' else 'FP32((raw FP64 legal costs - per-row FP64 minimum) / per-row FP64 minimum)'),
        tree_inference=('native-class probability/log-probability argmax' if args.arm == 'C'
                        else 'argmin predicted relative regret; no observed costs or Oracle at inference'),
        tree_early_stopping=False, tree_class_weights=None, tree_method_quotas=False,
        tree_cost_normalizer='TRAIN labels only, per-row legal minimum; no feature normalization from labels',
        train_eval='full18 VAL every100 common trees after each task finishes3000; full TRAIN only at selected common count',
        nn_update_budget_matched=False, total_sample_exposures=0)
    return config


def relative_regret_targets(costs):
    costs = np.asarray(costs)
    if costs.dtype != np.float64 or costs.ndim != 2 or not costs.shape[1]:
        raise ValueError('D supervision requires the full legal FP64 cost matrix')
    if not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('GPU MultiRMSE requires finite positive costs, without missing targets')
    oracle = costs.min(1, keepdims=True)
    targets = np.asarray((costs-oracle)/oracle, dtype=np.float32, order='C')
    if not np.isfinite(targets).all():
        raise ValueError('Relative regret overflowed FP32')
    return targets


def choose_global_iteration(stages):
    """stages[problem][iteration] holds a TRAIN-SBS-referenced VAL metric."""
    if set(stages) != set(PROBLEMS):
        raise ValueError('Do not select an incomplete full18 bank')
    grid = tuple(range(VAL_PERIOD, ITERATIONS+1, VAL_PERIOD))
    if any(set(row) != set(grid) for row in stages.values()):
        raise ValueError('Every problem must use the same prelocked tree-count grid')
    means = {t: float(np.mean([stages[p][t]['vs_sbs_pct'] for p in PROBLEMS])) for t in grid}
    if not all(math.isfinite(v) for v in means.values()):
        raise ValueError('Nonfinite global validation metric')
    return min(grid, key=means.__getitem__), means


def _validate_test_lock(run, verify_artifacts=True):
    run = Path(run).resolve()
    needed = ('protocol.json', 'training_complete.json', 'final_test_lock.json', 'best.pt')
    if not all((run/name).is_file() for name in needed):
        raise PermissionError('No TEST access before a completed unique full18 VAL lock')
    config, done, lock = (read_json(run/name) for name in needed[:3])
    if (config.get('experiment') != 'R92' or config.get('arm') not in ARMS
            or lock.get('experiment') != 'R92' or lock.get('arm') != config['arm']
            or done.get('successful_updates') != ITERATIONS
            or config.get('max_successful_updates') != ITERATIONS
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False
            or lock.get('state_hash') != json_hash(config)
            or Path(lock.get('checkpoint', '')).resolve() != run/'best.pt'
            or lock.get('checkpoint_sha256') != file_hash(run/'best.pt')
            or lock.get('selected_update') not in range(VAL_PERIOD, ITERATIONS+1, VAL_PERIOD)):
        raise PermissionError('Changed or incomplete tree VAL lock')
    # best.pt locks external tree artifacts as well as the common inference count.
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    entries = state['tree_bank']['entries']
    if (state['args'] != config or state['successful_updates'] != lock['selected_update']
            or state['tree_bank']['ntree_end'] != lock['selected_update']
            or set(entries) != set(PROBLEMS)):
        raise PermissionError('Tree checkpoint differs from the unique VAL lock')
    if verify_artifacts:
        for entry in entries.values():
            if entry['kind'] == 'constant':
                continue
            path = (run/entry['path']).resolve()
            if not path.is_relative_to(run) or file_hash(path) != entry['sha256']:
                raise PermissionError('A locked tree artifact changed before TEST')
    return lock, config


def feature_cache(args, manifest, problem, splits, device):
    """One flat input-only array per problem/phase, shared through flock.

    Original R84 caches are only opened read-only. No audit, copy, or rebuild.
    TEST authorization happens before even checking a cached TEST feature file.
    """
    splits = tuple(splits)
    if splits == ('test',):
        _validate_test_lock(args.root/args.arm, verify_artifacts=False)
        phase = 'TEST'
    elif splits == ('train', 'val'):
        phase = 'TRAIN_VAL'
    else:
        raise ValueError('Extract TRAIN/VAL together, or TEST only after locking')
    directory = args.root/'features'/phase
    directory.mkdir(parents=True, exist_ok=True)
    path, receipt = directory/(problem+'.npy'), directory/(problem+'.json')
    identity = dict(feature_definition=FEATURE_DEFINITION,
        feature_source_sha256=file_hash(Path(__file__).with_name('r92_tree_features.py')),
        data_root=str(args.data_root.resolve()),
        preparation_sha256=file_hash(args.prepared_root/'prepared.json'), problem=problem,
        splits=list(splits), rows={s: manifest['splits'][s][problem]['rows'] for s in splits},
        sources={s: manifest['splits'][s][problem] for s in splits})
    total = sum(identity['rows'].values())
    if any(identity['rows'][s] != SPLIT_ROWS[s] for s in splits):
        raise ValueError('Never truncate the frozen split')
    with (directory/(problem+'.lock')).open('a') as claim:
        _status(args, 'waiting_for_features', feature_problem=problem, feature_phase=phase)
        fcntl.flock(claim, fcntl.LOCK_EX)
        if receipt.exists():
            saved = read_json(receipt)
            if saved.get('identity') != identity or not saved.get('complete') or not path.is_file():
                raise ValueError('Changed feature-cache identity; preserve existing cache')
        elif path.exists():
            raise ValueError('Unattributed feature array; preserve it rather than overwrite')
        else:
            temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
            array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                             shape=(total, len(COLUMNS)))
            offset = 0
            try:
                for split in splits:
                    cache = open_cache(args.cache_root, split, problem, manifest, args.data_root)
                    try:
                        cap = 256 if cache.metadata['true_length_max'] >= 500 or cache.kind == 'matrix' else 1024
                        for start in range(0, cache.size, cap):
                            stop = min(start+cap, cache.size)
                            batch = input_batch(cache.batch(np.arange(start, stop), device='cpu'))
                            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                            rows, columns = features(batch)
                            if columns != COLUMNS:
                                raise ValueError('Feature schema changed while extracting')
                            array[offset+start:offset+stop] = rows
                            _status(args, 'extracting_features', feature_problem=problem,
                                    feature_split=split, feature_rows=stop, feature_total=cache.size)
                        offset += cache.size
                    finally:
                        cache.close()
                array.flush()
            finally:
                array._mmap.close()
            temporary.replace(path)
            write_json(receipt, dict(identity=identity, complete=True, shape=[total, len(COLUMNS)],
                                     dtype='float32', completed_utc=utc()))
        result = np.load(path, mmap_mode='r', allow_pickle=False)
        if result.shape != (total, len(COLUMNS)) or result.dtype != np.float32 or not np.isfinite(result).all():
            result._mmap.close()
            raise ValueError('Feature-cache shape/dtype/finiteness mismatch')
        return result


def _catboost():
    import catboost
    if catboost.__version__ != '1.2.8':
        raise RuntimeError('R92 requires the installed catboost==1.2.8; do not change packages here')
    return catboost


def _load_artifact(run, entry):
    if entry['kind'] == 'constant':
        return None
    path = (run/entry['path']).resolve()
    if not path.is_relative_to(run.resolve()) or file_hash(path) != entry['sha256']:
        raise PermissionError('A locked tree artifact changed or escaped the arm directory')
    catboost = _catboost()
    model = catboost.CatBoostClassifier() if entry['arm'] == 'C' else catboost.CatBoostRegressor()
    model.load_model(str(path))
    if model.tree_count_ != ITERATIONS:
        raise ValueError('Every nonconstant tree ensemble must contain all3000 trees')
    return model


def prediction_scores(raw, entry, classes=None):
    """Convert predictions to finite, local-pool FP32 utility, retaining all K."""
    raw = np.asarray(raw, dtype=np.float64)
    k = len(entry['pool'])
    if entry['kind'] == 'constant':
        count = len(raw)
        scores = np.full((count, k), np.log(np.finfo(np.float32).tiny), dtype=np.float32)
        scores[:, entry['constant_class']] = 0.
        return scores if entry['arm'] == 'C' else np.zeros_like(scores)
    if entry['arm'] == 'D':
        raw = raw.reshape(-1, k)
        scores = np.asarray(-raw, dtype=np.float32)
    else:
        classes = np.asarray(classes, dtype=np.int64)
        if raw.ndim != 2 or raw.shape[1] != len(classes) or len(set(classes.tolist())) != len(classes):
            raise ValueError('CatBoost class outputs cannot be mapped to the complete legal pool')
        if (classes < 0).any() or (classes >= k).any():
            raise ValueError('CatBoost returned a class outside the original legal pool')
        if np.isnan(raw).any() or np.isposinf(raw).any() or not np.isfinite(raw).any(1).all():
            raise ValueError('Invalid classifier raw predictions')
        maximum = raw.max(1, keepdims=True)
        log_prob = raw-maximum-np.log(np.exp(raw-maximum).sum(1, keepdims=True))
        floor = np.log(np.finfo(np.float32).tiny)
        scores = np.full((len(raw), k), floor, dtype=np.float32)
        scores[:, classes] = np.maximum(log_prob, floor).astype(np.float32)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite tree prediction')
    return scores


class TreeBank(nn.Module):
    """CPU CatBoost inference with GPU/CPU input-only phi and stable state hashes.

    forward() follows existing R84 local-pool logits. predict_batch() expands to
    GLOBAL_SOLVERS columns, while exposing the corresponding global legal mask.
    """

    def __init__(self, run, config, entries, ntree_end):
        super().__init__()
        self.run, self.config = Path(run).resolve(), config
        self.entries = entries
        self.selected_iteration = int(ntree_end)
        if set(entries) != set(PROBLEMS) or ntree_end not in range(VAL_PERIOD, ITERATIONS+1, VAL_PERIOD):
            raise ValueError('Require a full18 bank and one common locked tree count')
        self.device_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        fingerprints = []
        for problem in PROBLEMS:
            entry = entries[problem]
            if (entry['pool'] != config['candidate_pools'][problem]
                    or entry['pool_ids'] != config['candidate_pool_ids'][problem]
                    or entry['arm'] != config['arm']):
                raise ValueError('Tree artifact pool/arm differs from locked protocol')
            digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode('ascii')).digest()
            fingerprints.append(list(digest))
        self.register_buffer('model_fingerprints', torch.tensor(fingerprints, dtype=torch.uint8))
        self.register_buffer('common_ntree_end', torch.tensor(ntree_end, dtype=torch.int64))
        self.models = {p: _load_artifact(self.run, entries[p]) for p in PROBLEMS}
        self.inference_head = 'native_probability' if config['arm'] == 'C' else 'negative_predicted_relative_regret'
        self.params = dict(training_family='CatBoost', arm=config['arm'],
                           feature_definition=FEATURE_DEFINITION, selected_iteration=ntree_end)

    def predict_features(self, problem, rows):
        entry, model = self.entries[problem], self.models[problem]
        if rows.shape[1] != len(COLUMNS):
            raise ValueError('Wrong feature columns')
        if model is None:
            return prediction_scores(np.zeros(len(rows)), entry)
        raw = model.predict(rows, prediction_type='RawFormulaVal', ntree_end=self.selected_iteration,
                            thread_count=2, task_type='CPU')
        return prediction_scores(raw, entry, getattr(model, 'classes_', None))

    @torch.no_grad()
    def forward(self, batch):
        pid = batch['problem_id']
        if torch.is_tensor(pid):
            unique = pid.unique()
            if unique.numel() != 1:
                raise ValueError('Tree inference expects a single task per batch')
            pid = int(unique.item())
        problem = PROBLEMS[int(pid)] if not isinstance(pid, str) else pid
        entry = self.entries[problem]
        if 'pool_ids' in batch:
            ids = torch.as_tensor(batch['pool_ids']).detach().cpu().numpy().reshape(-1)
            if not np.array_equal(ids, entry['pool_ids']):
                raise ValueError('Inference candidate identities differ from the locked legal pool')
        rows, _ = features(batch)
        output_device = batch['n'].device
        utility = torch.from_numpy(self.predict_features(problem, rows)).to(output_device)
        mask = torch.ones_like(utility, dtype=torch.bool)
        if 'mask' in batch:
            mask &= batch['mask'].to(output_device).bool()[:, entry['pool_ids']]
        if 'solver_mask' in batch:
            mask &= batch['solver_mask'].to(output_device).bool()
        if not mask.any(1).all():
            raise ValueError('No legal candidate at inference')
        margin = (utility[:, :, None]-utility[:, None, :]).masked_fill(~(mask[:, :, None]&mask[:, None, :]), 0.)
        return dict(logits=utility.masked_fill(~mask, -float('inf')), utility=utility,
                    pair_margin=margin, solver_mask=mask)

    @torch.no_grad()
    def predict_batch(self, batch):
        local = self(batch)
        ids = torch.as_tensor(batch['pool_ids'], device=local['logits'].device).long()
        shape = (len(local['logits']), len(GLOBAL_SOLVERS))
        logits = local['logits'].new_full(shape, -float('inf'))
        utility = local['utility'].new_zeros(shape)
        mask = torch.zeros(shape, dtype=torch.bool, device=logits.device)
        logits[:, ids], utility[:, ids], mask[:, ids] = local['logits'], local['utility'], local['solver_mask']
        margin = (utility[:, :, None]-utility[:, None, :]).masked_fill(~(mask[:, :, None]&mask[:, None, :]), 0.)
        return dict(logits=logits, utility=utility, pair_margin=margin, solver_mask=mask,
                    inference_head=self.inference_head)


def load_checkpoint(path, device='cpu'):
    path = Path(path).resolve()
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R92' or config.get('arm') not in ARMS
            or config.get('max_successful_updates') != ITERATIONS
            or config.get('solver_names') != GLOBAL_SOLVERS
            or state.get('state_hash') != json_hash(config)
            or state.get('successful_updates') != state['tree_bank']['ntree_end']
            or state.get('epoch') != state['successful_updates']//VAL_PERIOD
            or config.get('feature_definition') != FEATURE_DEFINITION):
        raise ValueError('Not an unchanged R92 full18 tree-bank checkpoint')
    bank = TreeBank(path.parent, config, state['tree_bank']['entries'], state['tree_bank']['ntree_end'])
    if parameter_hash(bank.state_dict()) != parameter_hash(state['model']):
        raise ValueError('Tree-bank fingerprints/common iteration differ from checkpoint state')
    bank.load_state_dict(state['model'], strict=True)
    return bank.to(device), state


def load_tree_bank(run_dir, device='cpu'):
    return load_checkpoint(Path(run_dir)/'best.pt', device)[0]


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def _fit_problem(args, config, manifest, problem, device):
    run, state_hash = args.root/args.arm, json_hash(config)
    directory = run/'tree_models'
    directory.mkdir(exist_ok=True)
    receipt, path = directory/(problem+'.json'), directory/(problem+'.cbm')
    if receipt.exists():
        entry = read_json(receipt)
        if entry.get('state_hash') != state_hash:
            raise ValueError('Completed tree belongs to another protocol; refusing overwrite')
        return entry, _load_artifact(run, entry)
    if path.exists():
        raise ValueError('Unattributed trained tree file; preserve rather than overwrite')
    rows = feature_cache(args, manifest, problem, ('train', 'val'), device)
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    cache = open_cache(args.cache_root, 'train', problem, manifest, args.data_root)
    try:
        pool, ids = list(cache.pool), list(cache.pool_ids)
        if pool != config['candidate_pools'][problem] or ids != config['candidate_pool_ids'][problem]:
            raise ValueError('Original legal pool differs from parent protocol')
        winner = np.asarray(cache.raw['winner']).copy()
        if not np.array_equal(winner, cache.raw['costs'].argmin(1)):
            raise ValueError('Retain native FP64 ind, not an FP32 argmin')
        entry = dict(problem=problem, arm=args.arm, pool=pool, pool_ids=ids,
            state_hash=state_hash, train_rows=cache.size, requested_iterations=ITERATIONS,
            winner_counts=np.bincount(winner, minlength=len(pool)).tolist())
        classes = np.unique(winner)
        constant = (args.arm == 'C' and len(classes) == 1) or len(pool) == 1
        if constant:
            entry.update(kind='constant', constant_class=int(classes[0]), actual_tree_count=0,
                constant_reason='single observed native TRAIN class' if args.arm == 'C' else 'single legal method',
                legal_pool_preserved=True, gpu_tree_fit=False)
            model = None
        else:
            catboost = _catboost()
            params = dict(TREE_PARAMETERS)
            if args.arm == 'C':
                params.update(loss_function='MultiClass', classes_count=len(pool))
                model, target = catboost.CatBoostClassifier(**params), winner
            else:
                params.update(loss_function='MultiRMSE')
                model, target = catboost.CatBoostRegressor(**params), relative_regret_targets(cache.raw['costs'])
            _status(args, 'fitting_tree', training_problem=problem, successful_updates=0,
                    target_dimensions=len(pool), planned_trees=ITERATIONS)
            # No eval_set, early stopping, quotas, class weights, or per-task search.
            model.fit(rows[:SPLIT_ROWS['train']], target, verbose=100)
            if model.tree_count_ != ITERATIONS:
                raise RuntimeError('Fit did not complete the prelocked3000 trees')
            temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
            model.save_model(str(temporary))
            temporary.replace(path)
            entry.update(kind='catboost', path=str(path.relative_to(run)), sha256=file_hash(path),
                         actual_tree_count=ITERATIONS, gpu_tree_fit=True,
                         parameters=params, observed_classes=classes.tolist())
        write_json(receipt, entry)
        return entry, model
    finally:
        cache.close()
        rows._mmap.close()


def validation_predictions(model, rows, entry, iterations=ITERATIONS, period=VAL_PERIOD):
    """Exact prefixes avoid CatBoost1.2.8 staged_predict's absent-class bug."""
    for iteration in range(period, iterations+1, period):
        raw = (np.zeros(len(rows)) if model is None else model.predict(rows,
            prediction_type='RawFormulaVal', ntree_end=iteration, thread_count=2, task_type='CPU'))
        yield prediction_scores(raw, entry, getattr(model, 'classes_', None))


def _staged_validation(args, config, manifest, problem, entry, model, device):
    run = args.root/args.arm
    directory = run/'staged_val'
    directory.mkdir(exist_ok=True)
    path = directory/(problem+'.json')
    identity = dict(state_hash=json_hash(config), tree_entry=entry, period=VAL_PERIOD)
    if path.exists():
        saved = read_json(path)
        if saved['identity'] != identity:
            raise ValueError('Staged VAL belongs to a different fixed tree model')
        return {int(k): v for k, v in saved['metrics'].items()}
    array = feature_cache(args, manifest, problem, ('train', 'val'), device)
    cache = open_cache(args.cache_root, 'val', problem, manifest, args.data_root)
    try:
        predictions = validation_predictions(model, array[SPLIT_ROWS['train']:], entry)
        result = {}
        for iteration, scores in zip(range(VAL_PERIOD, ITERATIONS+1, VAL_PERIOD), predictions):
            result[iteration], _ = fixed_sbs_metrics(scores, cache.raw, config['baselines'][problem])
        if len(result) != ITERATIONS//VAL_PERIOD:
            raise ValueError('CatBoost did not emit every common VAL tree count')
        write_json(path, dict(identity=identity, metrics=result, test_read=False))
        return result
    finally:
        cache.close()
        array._mmap.close()


def train(args, device):
    """Called only by an explicitly launched, allocated parent worker."""
    args = _args(args, device)
    if torch.device(device).type != 'cuda':
        raise ValueError('Actual R92 tree fitting requires the parent-allocated GPU')
    config = _config(args)
    (args.root/args.arm).mkdir(parents=True, exist_ok=True)
    tracker = wandb_run(args, config)
    try:
        return _train_bank(args, device, config, tracker)
    finally:
        tracker.finish()


def _train_bank(args, device, config, tracker):
    torch.set_num_threads(2)
    run = args.root/args.arm
    run.mkdir(parents=True, exist_ok=True)
    with (run/'tree_train.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = run/'protocol.json'
        if protocol_path.exists() and read_json(protocol_path) != config:
            raise ValueError('Tree protocol changed; do not overwrite a previous arm')
        write_json(protocol_path, config)
        if (run/'training_complete.json').exists():
            _validate_test_lock(run)
        else:
            from .r84_experiment import prepared_config, summarize
            _, manifest, _, baselines = prepared_config(args)
            if baselines != config['baselines']:
                raise ValueError('Parent SBS differs from frozen TRAIN statistics')
            entries, stages = {}, {}
            order = list(PROBLEMS) if args.arm == 'C' else list(reversed(PROBLEMS))
            for problem in order:
                entry, model = _fit_problem(args, config, manifest, problem, device)
                entries[problem] = entry
                stages[problem] = _staged_validation(args, config, manifest, problem, entry, model, device)
                for iteration, metrics in stages[problem].items():
                    tracker.log({'tree/completed_tasks': len(entries), 'tree/common_ntree_end': iteration,
                        f'val_task/{problem}/vs_sbs_pct': metrics['vs_sbs_pct'],
                        f'val_task/{problem}/mean_cost': metrics['mean_cost'],
                        f'val_task/{problem}/native_top1_pct': 100.*metrics['native_correct']/metrics['n']})
                del model
                _status(args, 'task_tree_complete', training_problem=problem,
                    completed_problems=len(entries), successful_updates=ITERATIONS if len(entries) == len(PROBLEMS) else 0)
            selected, means = choose_global_iteration(stages)
            history = []
            for t in means:
                val = summarize({p: stages[p][t] for p in PROBLEMS})
                history.append(dict(epoch=t//VAL_PERIOD, evaluation_point=t//VAL_PERIOD,
                    successful_updates=t, common_ntree_end=t, evaluation_tag=f'trees{t:04d}',
                    sample_exposures=0, training_unit='trees_per_task', train=None, val=val))
                tracker.log(dict({'tree/common_ntree_end': t, 'tree/full18': True},
                                 **{'val/'+k: v for k, v in val['macro'].items()}))
            best = copy.deepcopy(history[selected//VAL_PERIOD-1])
            best['evaluation_tag'] = 'selected_best'
            bank = TreeBank(run, config, entries, selected).to(device)
            state = dict(args=config, model={k: v.detach().cpu() for k, v in bank.state_dict().items()},
                tree_bank=dict(entries=entries, ntree_end=selected), optimizer=None, scaler=None,
                state_hash=json_hash(config), successful_updates=selected, epoch=selected//VAL_PERIOD,
                training_complete_trees_per_task=ITERATIONS, best=best)
            checkpoint = run/'best.pt'
            if checkpoint.exists():
                prior = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if prior['args'] != config or prior['tree_bank'] != state['tree_bank']:
                    raise PermissionError('Existing selected checkpoint differs; refusing overwrite')
            else:
                save_checkpoint(checkpoint, state)
            digest, state_hash = file_hash(checkpoint), state['state_hash']
            parent = _parent()
            best['val'] = parent.evaluate(bank, args, manifest, baselines, 'val', 'selected_best',
                                          selected, state_hash, digest)
            observed = best['val']['macro']['macro_vs_sbs_pct']
            if not math.isclose(observed, means[selected], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Selected-bank VAL disagrees with staged CPU prediction selection')
            lock = lock_best(run, best, config, state_hash)
            best['train'] = parent.evaluate(bank, args, manifest, baselines, 'train', 'selected_best',
                                            selected, state_hash, digest)
            for split in ('train', 'val'):
                tracker.log(dict({'tree/selected_ntree_end': selected},
                    **{split+'/'+k: v for k, v in best[split]['macro'].items()}))
            history[selected//VAL_PERIOD-1] = best
            write_json(run/'history.json', history)
            write_json(run/'best_eval.json', best)
            write_json(run/'result.json', dict(best=best, final=history[-1],
                successful_updates=ITERATIONS, common_ntree_end=selected, state_hash=state_hash,
                training_unit='trees_per_task', nn_update_budget_matched=False,
                stop_reason='all full18 task ensembles completed fixed3000 trees; no early stopping', test_read=False))
            write_json(run/'final_metrics.json', dict(epoch=best['epoch'], selected_update=selected,
                checkpoint_sha256=digest, state_hash=state_hash, train=best['train'], val=best['val'], test_read=False))
            write_json(run/'training_complete.json', dict(completed_utc=utc(), successful_updates=ITERATIONS,
                training_unit='trees_per_task', full18=True, selected_best=lock,
                tree_counts={p: entries[p]['actual_tree_count'] for p in PROBLEMS},
                common_ntree_end=selected, state_hash=state_hash, test_read=False))
            _status(args, 'training_complete', successful_updates=ITERATIONS, selected_update=selected)
    return test(args, device, tracker=tracker)


def test(args, device, *, tracker=None):
    """Idempotent TEST, gated before any original TEST input/label/feature read."""
    args = _args(args, device)
    run = args.root/args.arm
    lock, config = _validate_test_lock(run)
    parent_lock, parent_config = _parent().authorize_test(args)
    if parent_lock != lock or parent_config != config:
        raise PermissionError('Parent and tree TEST authorization disagree')
    args.cache_root = Path(config['cache_root']) if args.cache_root is None else args.cache_root
    args.prepared_root = Path(config['prepared_root'])
    args.gpu_type, args.eval_batch_size = config['gpu_type'], config['eval_batch_size']
    args.large_micro_cap = config['microbatch_caps']['TSP']
    expected = getattr(args, 'state_hash', None)
    if expected is not None and expected != lock['state_hash']:
        raise PermissionError('Caller expected a different frozen tree state')
    result_path = run/'test_results.json'
    with (run/'tree_test.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result_path.exists():
            saved = read_json(result_path)
            if (saved.get('complete') is not True or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
                    or saved.get('state_hash') != lock['state_hash']):
                raise PermissionError('Saved TEST belongs to a different tree lock')
            return saved['result']
        own_tracker = tracker is None
        if own_tracker:
            tracker = wandb_run(args, config)
        try:
            return _test_bank(args, device, run, lock, config, result_path, tracker)
        finally:
            if own_tracker:
                tracker.finish()


def _test_bank(args, device, run, lock, config, result_path, tracker):
    from .r85_experiment import ensure_test_cache
    manifest = ensure_test_cache(args, lock)
    for problem in PROBLEMS:
        array = feature_cache(args, manifest, problem, ('test',), device)
        array._mmap.close()
    bank, state = load_checkpoint(run/'best.pt', device)
    write_json(run/'test_started.json', dict(started_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256']))
    result = _parent().evaluate(bank, args, manifest, config['baselines'], 'test', 'selected_best',
                                state['successful_updates'], lock['state_hash'], lock['checkpoint_sha256'])
    _validate_test_lock(run)
    tracker.log(dict({'tree/selected_ntree_end': lock['selected_update']},
                     **{'test/'+k: v for k, v in result['macro'].items()}))
    write_json(result_path, dict(complete=True, completed_utc=utc(), result=result,
        epoch=lock['epoch'], selected_update=lock['selected_update'], state_hash=lock['state_hash'],
        checkpoint_sha256=lock['checkpoint_sha256']))
    final = read_json(run/'final_metrics.json')
    final.update(test=result, test_read=True)
    write_json(run/'final_metrics.json', final)
    _status(args, 'complete', successful_updates=ITERATIONS, selected_update=lock['selected_update'],
            test=result['macro'])
    return result
