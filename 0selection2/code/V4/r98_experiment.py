"""Frozen full18 per-method expected-relative-cost inference, never SGD.

``prepare(args)`` returns the protocol for main to lock; it does not create
protocol_lock.json. Only historical TRAIN targets enter the frozen regressor.
All three commands share the same explicit path arguments; ROOT is the actual
run directory, not ROOT/A. Runtime/data/model implementation belongs to Curie.
"""

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .r80_reporting import file_hash, json_hash, read_json, utc, write_json
from . import r97_experiment as previous
from .r98_reporting import (INFERENCE_RULE, SELECTION_RULE, descriptive_gap_view,
                            metric_row, stable_gap_rank)


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R98_performance_context'
CONTEXT = V4 / 'runs/R97_tabular_in_context/context'
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
PREPARED = V4 / 'runs/R84_new90k_attention/prepared'
FEATURES = V4 / 'runs/R92_broad_learning/features'
ROWS = dict(train=81000, val=4500, test=4500)
TARGET_RULE = 'raw FP64 relative gap (c-cmin)/cmin, per-method mean regression'


def status(root, phase, **fields):
    path = Path(root) / 'status.json'
    record = read_json(path) if path.is_file() else {}
    record.update(experiment='R98', arm='A', phase=phase, pid=os.getpid(),
        host=socket.gethostname(), updated_utc=utc(), optimizer_updates=0,
        automatic_retry=False, training_unit='frozen_per_method_TRAIN_context', **fields)
    write_json(path, record)


def write_once(path, record):
    path = Path(path)
    if path.exists():
        if read_json(path) != record:
            raise ValueError('Preserve existing identity; use a new root: ' + str(path))
    else:
        write_json(path, record)


def runtime_metadata(root):
    """Consume Curie's immutable, config-ready runtime receipt, not R97 weights."""
    path = Path(root) / 'runtime/runtime_metadata.json'
    record = read_json(path)
    for key in ('regressor_parameters', 'library_root', 'public_checkpoint_sha256',
                'package_version', 'source_revision'):
        if key not in record:
            raise ValueError('Incomplete R98 runtime receipt: ' + key)
    record = dict(record, runtime_metadata_path=str(path.resolve()),
                  runtime_metadata_sha256=file_hash(path))
    check_parameters(record)
    return record


def check_parameters(config):
    from .r98_model import WEIGHT_NAME, WEIGHT_SHA256
    params = config['regressor_parameters']
    if (params.get('n_estimators') != 1 or params.get('batch_size') != 1
            or params.get('norm_methods') != ['none']
            or params.get('allow_auto_download') is not False
            or params.get('offload_mode') != 'gpu'
            or params.get('random_state') != 2 or params.get('use_fa3') is not False
            or params.get('use_amp') is not True or params.get('checkpoint_version') != WEIGHT_NAME):
        raise ValueError('Require the fixed single-estimator, no-normalization GPU regressor')
    managers = params.get('inference_config', {})
    if any(managers.get(key, {}).get('offload') != 'gpu'
           for key in ('COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG')):
        raise ValueError('CPU/disk offload must be disabled for every inference stage')
    digest = config['public_checkpoint_sha256']
    if (digest != WEIGHT_SHA256
            or file_hash(params['model_path']) != digest):
        raise ValueError('Public regressor checkpoint differs from the runtime SHA256')


def context_identity(context, problem, args, fingerprint):
    from .r92_tree_features import COLUMNS, FEATURE_DEFINITION
    identity = context.metadata['identity']
    if (identity.get('problem') != problem or identity.get('seed') != 2
            or identity.get('data_root') != str(args.data_root)
            or identity.get('rows') != ROWS['train']
            or identity.get('label_splits') != ['train']
            or identity.get('context_subsampling') is not False
            or identity.get('class_quotas') is not False
            or identity.get('row_order') != 'natural frozen TRAIN row order'
            or identity.get('feature_definition') != FEATURE_DEFINITION
            or tuple(context.transform.columns) != tuple(COLUMNS)
            or context.x.shape != (ROWS['train'], len(context.kept_columns))
            or context.y.shape != (ROWS['train'],)
            or context.source_ids.shape != (ROWS['train'],)):
        raise ValueError('Require the immutable R97 full81000 natural-order TRAIN context')
    sources = identity['source_files']
    if (sources['canonical_features']['sha256'] != fingerprint['feature_sha256']
            or sources['feature_receipt']['sha256'] != fingerprint['receipt_sha256']
            or sources['prepared']['sha256'] != file_hash(args.prepared_root / 'prepared.json')):
        raise ValueError('R97 context and current canonical R92/R84 inputs differ')
    return dict(metadata_sha256=json_hash(context.metadata), rows=len(context.y),
                columns=len(context.kept_columns))


def prepare(args):
    from .r97_data import load_context
    from .r98_data import prepare_targets, load_targets
    args = normalize_args(args)
    if args.target_root == args.context_root or args.target_root.is_relative_to(args.context_root):
        raise ValueError('Historical R97 context is immutable; targets need a separate root')
    prepared = read_json(args.prepared_root / 'prepared.json')
    manifest = read_json(args.data_root / 'scenario_manifest.json')
    source = read_json(args.prepared_root / 'source_cache_manifest.json')
    baselines = read_json(args.prepared_root / 'common_full_train_sbs.json')
    if (prepared.get('data_root') != str(args.data_root) or prepared.get('seed') != 2
            or prepared.get('rows_per_problem') != ROWS
            or manifest.get('status') != 'complete' or set(manifest['problems']) != set(PROBLEMS)
            or source.get('complete') is not True or source.get('data_root') != str(args.data_root)
            or any(set(source['splits'][s]) != set(PROBLEMS) for s in ('train', 'val'))
            or set(baselines) != set(PROBLEMS)):
        raise ValueError('Require frozen NEW90k seed2 full18 R84 preparation and TRAIN SBS')
    runtime = runtime_metadata(args.root)
    inputs = {p: previous.query_fingerprint(p, 'val', args) for p in PROBLEMS}
    contexts, pools, ids = {}, {}, {}
    for problem in PROBLEMS:
        context = load_context(args.context_root, problem)
        try:
            contexts[problem] = context_identity(context, problem, args, inputs[problem])
            pools[problem], ids[problem] = list(context.pool), list(context.pool_ids)
            released = manifest['problems'][problem]
            if (pools[problem] != released['methods'] or ids[problem] != released['global_solver_ids']
                    or ids[problem] != sorted(set(ids[problem]))
                    or 'RELD_MTL' in pools[problem] or 'RELD_MOEL' in pools[problem]
                    or ('B' in problem and set(pools[problem]) & {'RouteFinder', 'MoSES_RF', 'MoSES_CaDA'})
                    or baselines[problem].get('selection_split') != 'train'
                    or baselines[problem].get('train_rows') != 81000
                    or pools[problem][baselines[problem]['sbs_slot']] != baselines[problem]['sbs_solver']):
                raise ValueError('Frozen legal pool or full-TRAIN SBS differs: ' + problem)
        finally:
            context.close()
    prepare_targets(args.data_root, args.context_root, args.target_root, problems=PROBLEMS)
    targets = {}
    for problem in PROBLEMS:
        table = load_targets(args.target_root, problem)
        try:
            if (table.g.shape != (81000, len(pools[problem])) or table.g.dtype != np.float64
                    or not np.isfinite(table.g).all() or np.any(table.g < 0)
                    or np.any(table.g.min(1) != 0)):
                raise ValueError('Require raw FP64 full81000 per-method relative-gap targets')
            targets[problem] = dict(metadata_sha256=json_hash(table.metadata),
                rows=len(table.g), methods=table.g.shape[1])
        finally:
            table.close()
    protocol = dict(runtime, experiment='R98', arm='A', arms=['A'], seed=2,
        locked=True, task_scope='full18', data_root=str(args.data_root),
        prepared_root=str(args.prepared_root), feature_root=str(args.feature_root),
        context_root=str(args.context_root), target_root=str(args.target_root),
        rows_per_problem=ROWS, contexts=contexts, targets=targets, baselines=baselines,
        candidate_pools=pools, candidate_pool_ids=ids, solver_names=GLOBAL_SOLVERS,
        query_inputs=dict(val=inputs), inference_rule=INFERENCE_RULE, target_rule=TARGET_RULE,
        selection_rule=SELECTION_RULE, winner_rule=previous.WINNER_RULE,
        optimizer_updates=0, fine_tuning=False, historical_TRAIN_labels_in_context=True,
        current_query_labels_in_forward=False, context_subsampling=False,
        context_rows='all81000/task, natural order and frequency, no quota/subset',
        feature_transform='R97 task-TRAIN-constant drop only; R92 canonical357; no refitting',
        query_batch_rule='one complete named dataset per predict; each method fits all81000 sequentially',
        checkpoint_rule='one frozen predeclared full18 bank; no VAL grid or task-wise assembly',
        prediction_statistic='official TabICLRegressor mean', constant_shortcut='exact constant only',
        descriptive_probabilities=dict(rule='normalized exp(-raw predicted gap/0.01)',
            temperature=.01, calibrated=False, used_for_decisions=False),
        kv_cache=False, allow_cpu_offload=False, flash_attention_only=True,
        resource_guard=dict(max_rss_gib=8.5, max_seconds_per_call=1800),
        resource_witness='first full TSP VAL, before query supervision; no extra subset call',
        pretraining_budget_matched=False, test_development_exposed=True, wandb_mode='offline',
        identities=dict(prepared_sha256=file_hash(args.prepared_root / 'prepared.json'),
            scenario_manifest_sha256=file_hash(args.data_root / 'scenario_manifest.json'),
            source_cache_manifest_sha256=file_hash(args.prepared_root / 'source_cache_manifest.json'),
            train_sbs_sha256=file_hash(args.prepared_root / 'common_full_train_sbs.json')))
    write_once(args.root / 'prepare_identity.json', protocol)
    if args.protocol_lock.exists() and read_json(args.protocol_lock) != protocol:
        raise ValueError('Main protocol lock differs; preserve it')
    print(json.dumps(dict(experiment='R98', cli_ready=True, tasks=18, optimizer_updates=0,
        identity=str(args.root / 'prepare_identity.json'), protocol_lock=str(args.protocol_lock),
        protocol_lock_written=False)), flush=True)
    return protocol


def validate_protocol(path):
    config = read_json(path)
    if (config.get('experiment') != 'R98' or config.get('seed') != 2
            or config.get('rows_per_problem') != ROWS or config.get('optimizer_updates') != 0
            or config.get('fine_tuning') is not False
            or config.get('historical_TRAIN_labels_in_context') is not True
            or config.get('current_query_labels_in_forward') is not False
            or config.get('inference_rule') != INFERENCE_RULE or config.get('target_rule') != TARGET_RULE
            or config.get('prediction_statistic') != 'official TabICLRegressor mean'
            or config.get('constant_shortcut') != 'exact constant only'
            or config.get('kv_cache') is not False or config.get('allow_cpu_offload') is not False
            or config.get('flash_attention_only') is not True
            or config.get('resource_guard') != dict(max_rss_gib=8.5, max_seconds_per_call=1800)
            or any(set(config.get(key, {})) != set(PROBLEMS) for key in
                   ('contexts', 'targets', 'baselines', 'candidate_pools', 'candidate_pool_ids'))
            or any(c['rows'] != 81000 for c in config['contexts'].values())
            or any(t['rows'] != 81000 for t in config['targets'].values())
            or set(config.get('query_inputs', {}).get('val', {})) != set(PROBLEMS)):
        raise ValueError('Require the predeclared full18 frozen expected-cost protocol')
    check_parameters(config)
    if file_hash(config['runtime_metadata_path']) != config['runtime_metadata_sha256']:
        raise ValueError('Pinned regressor runtime receipt changed')
    return config


def require_test_lock(root, config):
    lock = previous.require_test_lock(Path(root), config)
    bank = read_json(Path(root) / 'checkpoint.json')
    if (lock.get('selection_rule') != SELECTION_RULE
            or bank.get('selection_rule') != SELECTION_RULE
            or bank.get('weights_updated') is not False
            or bank.get('protocol_sha256') != json_hash(config)
            or any(bank.get(key) != config[key] for key in
                   ('contexts', 'targets', 'public_checkpoint_sha256', 'inference_rule'))):
        raise PermissionError('Require the unchanged unique R98 expected-cost bank before TEST/LIB')
    return lock


def query_inputs(problem, split, args, config, context):
    if split == 'test':
        require_test_lock(args.root, config)
    return previous.query_inputs(problem, split, args, config, context)


def guarded_call(function, config, torch, *, flash=False):
    if flash:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        kernel = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    else:
        kernel = nullcontext()
    with previous.call_guard(config, torch) as timing:
        with kernel:
            result = function()
    return result, timing


class GuardedRegressor:
    """Guard and time each sequential full-context method fit and mean call."""
    def __init__(self, regressor, slot, model, config, torch):
        self.regressor, self.slot, self.model = regressor, slot, model
        self.config, self.torch = config, torch

    def fit(self, x, y, *args, **kwargs):
        rows = self.config['rows_per_problem']['train']
        if len(x) != rows or np.shape(y) != (rows,) or np.asarray(y).dtype != np.float64:
            raise ValueError('Every nonconstant method must fit all historical TRAIN rows')
        if kwargs.get('kv_cache') is not False:
            raise ValueError('Every scalar fit requires explicit kv_cache=False')
        _, timing = guarded_call(lambda: self.regressor.fit(x, y, *args, **kwargs),
                                self.config, self.torch, flash=True)
        self.model.call_records.append(dict(timing, stage='fit_method', method_slot=self.slot,
                                            train_rows=rows))
        return self

    def predict(self, *args, **kwargs):
        if kwargs.get('output_type') != 'mean':
            raise ValueError('Use the official regressor mean, never quantiles or a classifier')
        result, timing = guarded_call(lambda: self.regressor.predict(*args, **kwargs),
                                     self.config, self.torch, flash=True)
        self.model.call_records.append(dict(timing, stage='predict_method', method_slot=self.slot,
            train_rows=self.config['rows_per_problem']['train'], prediction_statistic='mean'))
        return result

    def __getattr__(self, name):
        return getattr(self.regressor, name)


def fitted(problem, args, config, torch):
    from .r97_data import load_context
    from .r98_data import load_targets
    from .r98_model import FrozenPerformanceContext, public_regressor
    started = time.monotonic()
    context = load_context(args.context_root, problem)
    targets = None
    try:
        context_seconds = time.monotonic() - started
        target_started = time.monotonic()
        targets = load_targets(args.target_root, problem)
        target_seconds = time.monotonic() - target_started
        record, target_record = config['contexts'][problem], config['targets'][problem]
        if (json_hash(context.metadata) != record['metadata_sha256']
                or context.x.shape != (record['rows'], record['columns'])
                or list(context.pool) != config['candidate_pools'][problem]
                or list(context.pool_ids) != config['candidate_pool_ids'][problem]
                or json_hash(targets.metadata) != target_record['metadata_sha256']
                or targets.g.shape != (record['rows'], len(context.pool))):
            raise ValueError('Context/relative-gap targets changed after locking')
        constant = np.all(targets.g == targets.g[0], axis=0)
        slots = iter(np.flatnonzero(~constant).tolist())
        holder = {}

        def factory():
            return GuardedRegressor(public_regressor(config, args.device), next(slots),
                                    holder['model'], config, torch)

        model = FrozenPerformanceContext(context, targets, factory)
        model.call_records = []
        model.constant_method_slots = np.flatnonzero(constant).tolist()
        holder['model'] = model
        return model, dict(problem=problem, stage='load_context', seconds=time.monotonic()-started,
            context_read_and_verify_seconds=context_seconds, target_read_and_verify_seconds=target_seconds,
            train_rows=len(context.y), methods=len(context.pool),
            exact_constant_method_slots=model.constant_method_slots)
    except BaseException:
        if targets is not None:
            targets.close()
        context.close()
        raise


def predict(model, x, ids, config, torch):
    if np.intersect1d(ids, model.context.source_ids).size:
        raise ValueError('Query/base-instance IDs overlap historical TRAIN context')
    started = time.monotonic()
    gaps = np.asarray(model.predict_features(x, ids), dtype=np.float64)
    stable_gap_rank(gaps, model.context.pool_ids)
    if gaps.shape != (len(x), len(model.context.pool)):
        raise ValueError('Regressor mean must retain every query and every legal method')
    expected = set(range(len(model.context.pool))) - set(model.constant_method_slots)
    for stage in ('fit_method', 'predict_method'):
        slots = [r['method_slot'] for r in model.call_records if r['stage'] == stage]
        if len(slots) != len(expected) or set(slots) != expected:
            raise ValueError('Require one sequential full-context fit/mean call per nonconstant method')
    return gaps, dict(seconds=time.monotonic()-started, stage='predict',
        train_rows=len(model.context.y), query_rows=len(x), methods=len(model.context.pool),
        sequential_fullcontext_fits=len(expected), sequential_mean_calls=len(expected),
        exact_constant_method_slots=model.constant_method_slots, kv_cache=False,
        context_reencoded=True, excludes_graph_to_statistics_time=True,
        includes_all_sequential_method_fits=True)


def saved_prediction(problem, gaps, raw, source_ids, nodes, config):
    from .r98_reporting import load_predictions
    gaps = np.asarray(gaps, dtype=np.float64)
    rank = stable_gap_rank(gaps, config['candidate_pool_ids'][problem])
    logits, probabilities = descriptive_gap_view(gaps)
    rows = len(gaps)
    data = dict(expected_gaps=gaps, predG=gaps, logits=logits, scores=logits,
        probabilities=probabilities, pred=rank[:, 0].astype(np.int64),
        costs=np.asarray(raw['costs']), winner=np.asarray(raw['winner']), pool=np.asarray(raw['pool']),
        pool_ids=np.asarray(config['candidate_pool_ids'][problem], dtype=np.int64),
        indices=np.arange(rows, dtype=np.int64), source_indices=np.arange(rows, dtype=np.int64),
        release_source_indices=np.asarray(source_ids, dtype=np.int64), nodes=np.asarray(nodes, dtype=np.int64))
    view = load_predictions(data, '<in-memory R98>')
    row = metric_row(view, config['baselines'][problem], problem, 'pending')
    from .r97_reporting import SAVED_KEYS
    metrics = {name: row[key] / 100. if key.startswith('top') else row[key]
               for key, name in SAVED_KEYS.items()}
    metrics.update(n=rows, native_correct=row['native_correct'], sbs_name=row['sbs_name'],
        sbs_selection_split='train', winner_rule=previous.WINNER_RULE,
        inference_rule=INFERENCE_RULE, optimizer_updates=0)
    return metrics, data


def summarize(results):
    from .r97_reporting import SAVED_KEYS
    macro = {'macro_' + name: float(np.mean([v[name] for v in results.values()]))
             for name in SAVED_KEYS.values()}
    return dict(per_problem=results, macro=macro,
        native_total=sum(v['n'] for v in results.values()),
        native_correct=sum(v['native_correct'] for v in results.values()),
        all_aggregation='equal weight per problem; mixed raw mean_cost descriptive only')


def close_model(model, torch):
    model.targets.close()
    model.context.close()
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_split(split, args, config, torch, timings):
    if split == 'test':
        require_test_lock(args.root, config)
    packed, results = {}, {}
    directory = args.root / 'evaluation' / split / 'selected_best'
    directory.mkdir(parents=True, exist_ok=False)
    for problem in PROBLEMS:
        model, loading = fitted(problem, args, config, torch)
        timings.append(dict(loading, split=split))
        try:
            feature_started = time.monotonic()
            x, ids = query_inputs(problem, split, args, config, model.context)
            timings.append(dict(problem=problem, split=split, stage='cached_query_inputs',
                seconds=time.monotonic()-feature_started, query_rows=len(x),
                includes_input_fingerprint_verification=True, fresh_graph_to_statistics_measured=False))
            status(args.root, 'predicting', problem=problem, split=split, query_rows=len(x))
            gaps, timing = predict(model, x, ids, config, torch)
            timings.extend(dict(r, problem=problem, split=split) for r in model.call_records)
            timings.append(dict(timing, problem=problem, split=split))
            if problem == 'TSP' and split == 'val':
                write_json(args.root / 'resource_witness.json', dict(complete=True,
                    labels_loaded=False, accuracy_computed=False, context_rows=81000,
                    query_rows=len(x), expected_gap_shape=list(gaps.shape),
                    checkpoint_sha256=config['public_checkpoint_sha256'],
                    context_loading=loading, predict=timing, extra_subset_call=False))
            # Inference has returned and its resource receipt is persisted before labels open.
            raw = previous.native_labels(problem, split, args, config)
            all_x = np.load(args.feature_root / ('TRAIN_VAL' if split == 'val' else 'TEST') /
                            (problem + '.npy'), mmap_mode='r', allow_pickle=False)
            offset = 81000 if split == 'val' else 0
            metrics, data = saved_prediction(problem, gaps, raw, ids,
                                             all_x[offset:offset+len(ids), 0], config)
            results[problem] = metrics
            np.savez_compressed(directory / (problem + '.npz'), **data)
            packed.update({problem + '__' + key: value for key, value in data.items()})
            print('[R98]', split, problem, json.dumps(metrics), flush=True)
            del x, ids, gaps, raw, all_x, data
        finally:
            close_model(model, torch)
            del model
            write_json(args.root / 'timing.json', dict(records=timings))
    result = dict(summarize(results), complete=True, experiment='R98', arm='A', split=split,
        state_hash=json_hash(config), optimizer_updates=0,
        sbs_reference='R84 full81000 TRAIN-fitted, frozen', winner_rule=previous.WINNER_RULE)
    np.savez_compressed(directory / 'rank000.npz', **packed)
    write_json(args.root / (split + '_results.json'), result)
    return result


def lib_query_inputs(problem, dataset, args, config, context, torch):
    """Decode native LIB inputs only; never instantiate a label-opening loader."""
    from code.unified_selector.registry import DATA_ROOT, constraint_bits, problem_descriptor
    from code.unified_selector.data import _coord_dist_code
    from .r75_data import _CPUUnpickler
    from .r92_tree_features import features
    from .local_geometry import local_geometry
    lock = require_test_lock(args.root, config)
    path = DATA_ROOT / dataset / 'dataset.pkl'
    identity = dict(dataset=dataset, problem=problem, path=str(path.resolve()),
        sha256=file_hash(path), checkpoint_sha256=lock['checkpoint_sha256'], labels_loaded=False)
    write_once(args.root / 'lib_input_locks' / (dataset + '.json'), identity)
    with path.open('rb') as stream:
        instances = _CPUUnpickler(stream).load()
    rows, sizes = [], []
    for start in range(0, len(instances), 16):
        nodes = []
        for inst in instances[start:start+16]:
            if problem == 'TSP':
                node = torch.as_tensor(inst[0], dtype=torch.float32)
            elif problem == 'CVRP':
                loc = torch.as_tensor(inst['loc'][0], dtype=torch.float32)
                depot = torch.as_tensor(inst['depot'][0], dtype=torch.float32)
                demand = torch.as_tensor(inst['demand'][0], dtype=torch.float32)
                node = torch.cat((torch.cat((depot, torch.zeros(1, 1), torch.ones(1, 1)), 1),
                    torch.cat((loc, demand[:, None], torch.zeros(len(loc), 1)), 1)), 0)
            else:
                raise ValueError('Only native TSPLIB and CVRPLIB are declared')
            nodes.append(node)
        lengths = [len(node) for node in nodes]
        batch_size, maximum = len(nodes), max(lengths)
        node = torch.zeros(batch_size, maximum, nodes[0].shape[1], device=args.device)
        mask = torch.zeros(batch_size, maximum, dtype=torch.bool, device=args.device)
        for i, values in enumerate(nodes):
            node[i, :len(values)] = values.to(args.device)
            mask[i, :len(values)] = True
        batch = dict(kind='coord', node=node, node_mask=mask,
            n=torch.tensor(lengths, dtype=torch.long, device=args.device),
            cbits=torch.tensor([constraint_bits(problem)] * batch_size, device=args.device),
            problem_desc=torch.tensor([problem_descriptor(problem)] * batch_size, device=args.device),
            coord_dist=torch.tensor([_coord_dist_code(problem, start+i, dataset, len(instances))
                                    for i in range(batch_size)], device=args.device))
        batch['node_geom'] = local_geometry(node[..., :2], mask).float()
        x, _ = features(batch)
        rows.append(x)
        sizes.extend(lengths)
    if file_hash(path) != identity['sha256']:
        raise ValueError('Native LIB input changed while extracting statistics')
    x = np.concatenate(rows)
    # LIB is a separate base-instance namespace; Curie's source-ID validator is nonnegative.
    namespace = 2**62 + (0 if dataset == 'TSPLIB' else 2**32)
    ids = namespace + np.arange(len(x), dtype=np.int64)
    if np.intersect1d(ids, context.source_ids).size:
        raise ValueError('LIB input namespace overlaps historical TRAIN base IDs')
    return np.ascontiguousarray(x[:, context.kept_columns]), ids, np.asarray(sizes)


def evaluate_libs(args, config, torch, timings):
    from .r77_lib import checked_raw
    require_test_lock(args.root, config)
    results = {}
    for dataset, problem in (('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')):
        model, loading = fitted(problem, args, config, torch)
        timings.append(dict(loading, split=dataset))
        try:
            started = time.monotonic()
            x, ids, nodes = lib_query_inputs(problem, dataset, args, config, model.context, torch)
            timings.append(dict(problem=problem, split=dataset, stage='native_input_to_statistics',
                seconds=time.monotonic()-started, query_rows=len(x), includes_native_input_loading=True,
                labels_loaded=False, fresh_graph_to_statistics_measured=True))
            gaps, timing = predict(model, x, ids, config, torch)
            timings.extend(dict(r, problem=problem, split=dataset) for r in model.call_records)
            timings.append(dict(timing, problem=problem, split=dataset))
            raw = checked_raw(problem, dataset, config)
            if (raw['pool'] != config['candidate_pools'][problem]
                    or raw['pool_ids'] != config['candidate_pool_ids'][problem]):
                raise ValueError('Native LIB candidates differ from the single frozen bank')
            metrics, data = saved_prediction(problem, gaps, raw, ids, nodes, config)
            directory = args.root / 'lib_evaluation/predictions'
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / (dataset + '.npz'), **data)
            results[dataset] = dict(metrics, problem=problem, complete=True,
                source_directory=str(raw['directory']), state_hash=json_hash(config))
            del x, ids, gaps, raw, data
        finally:
            close_model(model, torch)
            del model
            write_json(args.root / 'timing.json', dict(records=timings))
    write_json(args.root / 'lib_results.json', results)
    return results


def ensure_fresh_run(root):
    names = ('attempt.json', 'status.json', 'protocol.json', 'checkpoint.json',
        'final_test_lock.json', 'test_input_lock.json', 'evaluation', 'val_results.json',
        'test_results.json', 'lib_evaluation', 'lib_results.json', 'run_complete.json',
        'postprocess_status.json', 'full_model_statistics', 'RESULTS.md', 'RESULTS.pdf')
    present = [name for name in names if (Path(root) / name).exists()]
    if present:
        raise RuntimeError('Preserve existing R98 output; no implicit retry: ' + ', '.join(present))


def run(args):
    import torch
    args = normalize_args(args)
    config = validate_protocol(args.protocol_lock)
    if args.device != 'cuda:0' or not torch.cuda.is_available():
        raise RuntimeError('Frozen inference needs the exact launcher-assigned CUDA device')
    for key in ('data_root', 'prepared_root', 'feature_root', 'target_root'):
        if str(getattr(args, key)) != config[key]:
            raise ValueError('Runtime path differs from the locked protocol: ' + key)
    # A launcher may stage immutable context files locally; fitted() verifies content hashes.
    ensure_fresh_run(args.root)
    write_json(args.root / 'attempt.json', dict(started_utc=utc(), automatic_retry=False))
    write_json(args.root / 'protocol.json', config)
    tracker, timings = None, []
    try:
        import wandb
        tracker = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
            name='R98_frozen_expected_cost_context', config=config, mode='offline', dir=str(args.root))
        write_json(args.root / 'wandb_receipt.json', dict(enabled=True, mode='offline',
            project='selector', entity='yjkds-southern-university-of-science-technology',
            run_id=tracker.id, path=str(tracker.dir), optimizer_updates=0))
        val = evaluate_split('val', args, config, torch, timings)
        bank = dict(experiment='R98', protocol_sha256=json_hash(config), contexts=config['contexts'],
            targets=config['targets'], public_checkpoint_sha256=config['public_checkpoint_sha256'],
            inference_rule=INFERENCE_RULE, weights_updated=False, selection_rule=SELECTION_RULE)
        write_once(args.root / 'checkpoint.json', bank)
        write_once(args.root / 'final_test_lock.json', dict(experiment='R98', arm='A',
            checkpoint=str(args.root / 'checkpoint.json'), checkpoint_sha256=file_hash(args.root / 'checkpoint.json'),
            state_hash=json_hash(config), selected_without_test=True, per_problem_assembly=False,
            locked_utc=utc(), selection_rule=SELECTION_RULE, validation_macro=val['macro']))
        require_test_lock(args.root, config)
        previous.lock_test_inputs(args, config)
        test = evaluate_split('test', args, config, torch, timings)
        libs = evaluate_libs(args, config, torch, timings)
        tracker.log({'VAL/' + k: v for k, v in val['macro'].items()})
        tracker.log({'TEST/' + k: v for k, v in test['macro'].items()})
        write_json(args.root / 'run_complete.json', dict(complete=True, completed_utc=utc(),
            full18_VAL=True, full18_TEST=True, LIB=list(libs), optimizer_updates=0,
            checkpoint_sha256=file_hash(args.root / 'checkpoint.json')))
        status(args.root, 'complete')
    except BaseException as error:
        status(args.root, 'failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if tracker is not None:
            tracker.finish()


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('command', choices=('prepare', 'run', 'report'))
    value.add_argument('--root', type=Path, default=ROOT)
    value.add_argument('--context-root', type=Path, default=CONTEXT)
    value.add_argument('--target-root', type=Path)
    value.add_argument('--feature-root', type=Path, default=FEATURES)
    value.add_argument('--data-root', type=Path, default=DATA)
    value.add_argument('--prepared-root', type=Path, default=PREPARED)
    value.add_argument('--protocol-lock', type=Path)
    value.add_argument('--device', default='cpu')
    value.add_argument('--wandb-mode', choices=('offline',), default='offline')
    return value


def normalize_args(args):
    args.root = Path(args.root).resolve()
    args.context_root = Path(args.context_root).resolve()
    args.target_root = Path(getattr(args, 'target_root', None) or args.root / 'targets').resolve()
    args.protocol_lock = Path(getattr(args, 'protocol_lock', None) or args.root / 'protocol_lock.json').resolve()
    for key in ('data_root', 'prepared_root', 'feature_root'):
        setattr(args, key, Path(getattr(args, key)).resolve())
    return args


def main(argv=None):
    args = normalize_args(parser().parse_args(argv))
    if args.command == 'report':
        from .r98_reporting import report
        return report(args)
    return {'prepare': prepare, 'run': run}[args.command](args)


if __name__ == '__main__':
    main()
