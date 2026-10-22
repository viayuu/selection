"""R100 full18 pairwise-binary frozen TabICL: prepare, run, report.

prepare is CPU-only. run is owned by the common launcher, never implicit.
One predeclared bank: full VAL -> unique context/weight lock -> bind TEST
inputs -> full TEST -> native LIB. No SGD, retries or post-TEST selection.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r97_experiment as previous
from .r80_reporting import file_hash, json_hash, read_json, utc, write_json
from .r97_data import _close, load_context
from .r100_data import LABEL_RULE, load_relations, prepare_relations, validate_relation_binding
from .r100_model import (
    INFERENCE_RULE, PROBABILITY_SEMANTICS, FrozenPairwiseContext,
    formal_score_view, mean_opponent_win, public_pair_classifier, stable_mean_win_rank,
)


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R100_pairwise_context'
CONTEXT = previous.ROOT / 'context'
ROWS = dict(train=81000, val=4500, test=4500)
SELECTION_RULE = 'frozen_single_predeclared_bank_no_VAL_search'
BANK_KEYS = ('contexts', 'relations', 'public_checkpoint_sha256', 'inference_rule', 'source_sha256')


def write_once(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != record:
            raise ValueError('Preserve existing immutable R100 artifact: ' + str(path))
        return
    # Exclusive creation prevents two preparers/runners silently replacing a lock.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def status(root, phase, **fields):
    path = Path(root) / 'status.json'
    record = read_json(path) if path.is_file() else {}
    record.update(experiment='R100', arm='A', phase=phase, pid=os.getpid(), host=socket.gethostname(),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES', ''), updated_utc=utc(), optimizer_updates=0,
        training_unit='frozen_strict_pair_TRAIN_context', automatic_retry=False, **fields)
    write_json(path, record)


def normalize_args(args):
    for key in ('root', 'context_root', 'feature_root', 'data_root', 'prepared_root'):
        setattr(args, key, Path(getattr(args, key)).resolve())
    args.protocol_lock = Path(args.protocol_lock or args.root / 'protocol_lock.json').resolve()
    args.relation_root = args.root / 'prep/relations'
    if (args.root == args.context_root or args.root.is_relative_to(args.context_root)
            or args.root == previous.ROOT or args.root.is_relative_to(previous.ROOT)
            or any(args.root == getattr(args, key) or args.root.is_relative_to(getattr(args, key))
                   for key in ('data_root', 'prepared_root', 'feature_root'))):
        raise ValueError('R100 output must be separate from historical inputs/results')
    if not args.protocol_lock.is_relative_to(args.root):
        raise ValueError('The R100 protocol lock must live inside its fresh output root')
    return args


def runtime_config():
    path = previous.ROOT / 'protocol_lock.json'
    old = read_json(path)
    params = old['classifier_parameters']
    required = dict(n_estimators=1, batch_size=1, norm_methods=['none'], use_amp=True,
        use_fa3=False, offload_mode='gpu', random_state=2, allow_auto_download=False,
        support_many_classes=False, checkpoint_version=previous.WEIGHT_NAME)
    library = previous.ROOT / 'runtime/python'
    weight = previous.ROOT / 'runtime/weights' / previous.WEIGHT_NAME
    if (old.get('experiment') != 'R97' or old.get('public_checkpoint_sha256') != previous.WEIGHT_SHA
            or old.get('library_root') != str(library) or params.get('model_path') != str(weight)
            or any(params.get(k) != v for k, v in required.items())
            or old.get('checkpoint_config', {}).get('max_classes') != 10
            or old.get('kv_cache') is not False or old.get('allow_cpu_offload') is not False
            or file_hash(weight) != previous.WEIGHT_SHA):
        raise ValueError('Require R97 original public classifier runtime, not regression or altered max_classes')
    keys = ('library_root', 'classifier_parameters', 'checkpoint_config', 'public_checkpoint_sha256',
        'public_checkpoint_revision', 'package_version', 'source_revision', 'runtime_wheels')
    result = {key: old[key] for key in keys}
    result['r97_protocol_path'] = str(path)
    result['r97_protocol_sha256'] = file_hash(path)
    paths = [path, weight, *sorted(library.rglob('*.py')),
        *sorted((previous.ROOT / 'runtime/wheels').glob('*.whl'))]
    if not library.is_dir() or not list(library.rglob('classifier.py')):
        raise ValueError('Missing pinned isolated TabICL classifier runtime')
    result['runtime_source_sha256'] = {str(p): file_hash(p) for p in paths}
    return result


def source_hashes():
    names = ('r100_data.py', 'r100_model.py', 'r100_experiment.py', 'r100_reporting.py',
        'r97_data.py', 'r97_model.py', 'r97_experiment.py', 'r97_reporting.py',
        'r75_data.py', 'r77_lib.py', 'r80_reporting.py', 'r92_tree_features.py', 'local_geometry.py')
    paths = [V4 / name for name in names]
    paths.extend([V4.parent / 'unified_selector' / name for name in ('registry.py', 'data.py')])
    return {str(path): file_hash(path) for path in paths}


def context_identity(context, problem, args, fingerprint):
    from .r92_tree_features import COLUMNS, FEATURE_DEFINITION
    identity, sources = context.metadata['identity'], context.metadata['identity']['source_files']
    if (identity.get('problem') != problem or identity.get('seed') != 2
            or identity.get('data_root') != str(args.data_root) or identity.get('rows') != 81000
            or identity.get('label_splits') != ['train'] or identity.get('context_subsampling') is not False
            or identity.get('class_quotas') is not False or identity.get('row_order') != 'natural frozen TRAIN row order'
            or identity.get('feature_definition') != FEATURE_DEFINITION or tuple(context.transform.columns) != tuple(COLUMNS)
            or context.x.shape != (81000, len(context.kept_columns))
            or sources['canonical_features']['sha256'] != fingerprint['feature_sha256']
            or sources['feature_receipt']['sha256'] != fingerprint['receipt_sha256']
            or sources['prepared']['sha256'] != file_hash(args.prepared_root / 'prepared.json')):
        raise ValueError('Require immutable full81000 natural-frequency R97 TRAIN X/transform/source IDs')
    return dict(metadata_sha256=json_hash(context.metadata), rows=len(context.x), columns=len(context.kept_columns))


def prepare(args):
    args = normalize_args(args)
    ensure_fresh_run(args.root)
    prepared = read_json(args.prepared_root / 'prepared.json')
    manifest = read_json(args.data_root / 'scenario_manifest.json')
    source = read_json(args.prepared_root / 'source_cache_manifest.json')
    baselines = read_json(args.prepared_root / 'common_full_train_sbs.json')
    if (prepared.get('data_root') != str(args.data_root) or prepared.get('seed') != 2
            or prepared.get('rows_per_problem') != ROWS or manifest.get('status') != 'complete'
            or set(manifest['problems']) != set(PROBLEMS) or source.get('complete') is not True
            or source.get('data_root') != str(args.data_root) or set(baselines) != set(PROBLEMS)
            or any(set(source['splits'][split]) != set(PROBLEMS) for split in ('train', 'val'))):
        raise ValueError('Require frozen NEW90k full18 R84 preparation and full TRAIN SBS')
    runtime = runtime_config()
    inputs = {p: previous.query_fingerprint(p, 'val', args) for p in PROBLEMS}
    contexts, pools, ids = {}, {}, {}
    for problem in PROBLEMS:
        with load_context(args.context_root, problem) as context:
            contexts[problem] = context_identity(context, problem, args, inputs[problem])
            pools[problem], ids[problem] = list(context.pool), list(context.pool_ids)
            released = manifest['problems'][problem]
            baseline = baselines[problem]
            if (pools[problem] != released['methods'] or ids[problem] != released['global_solver_ids']
                    or ids[problem] != sorted(set(ids[problem]))
                    or set(pools[problem]) & {'RELD_MTL', 'RELD_MOEL'}
                    or ('B' in problem and set(pools[problem]) & {'RouteFinder', 'MoSES_RF', 'MoSES_CaDA'})
                    or baseline.get('selection_split') != 'train' or baseline.get('train_rows') != 81000
                    or pools[problem][baseline['sbs_slot']] != baseline['sbs_solver']):
                raise ValueError('Changed frozen candidate pool/TRAIN SBS: ' + problem)
    prepare_relations(args.data_root, args.context_root, args.relation_root, problems=PROBLEMS)
    relations, support = {}, []
    for problem in PROBLEMS:
        with load_relations(args.relation_root, problem) as table:
            relations[problem] = dict(metadata_sha256=json_hash(table.metadata), rows=len(table.labels),
                pairs=table.labels.shape[1], pair_support=table.metadata['pair_support'])
            support.extend(dict(problem=problem, **row) for row in table.metadata['pair_support'])
    protocol = dict(runtime, experiment='R100', arm='A', arms=['A'], seed=2, locked=True, task_scope='full18',
        root=str(args.root), data_root=str(args.data_root), prepared_root=str(args.prepared_root),
        feature_root=str(args.feature_root), context_root=str(args.context_root), relation_root=str(args.relation_root),
        rows_per_problem=ROWS, contexts=contexts, relations=relations, baselines=baselines,
        candidate_pools=pools, candidate_pool_ids=ids, solver_names=GLOBAL_SOLVERS,
        query_inputs=dict(val=inputs), label_rule=LABEL_RULE, inference_rule=INFERENCE_RULE,
        probability_semantics=PROBABILITY_SEMANTICS, selection_rule=SELECTION_RULE, winner_rule=previous.WINNER_RULE,
        optimizer_updates=0, fine_tuning=False, historical_TRAIN_labels_in_context=True,
        current_query_labels_in_forward=False, context_subsampling=False, full_pool_winner_filter=False,
        query_overlap_policy='reject against complete TRAIN base IDs before any pair mask or shortcut',
        constant_shortcut='exact one class or no strict rows only; no strict rows -> .5',
        feature_transform='immutable R97 task-TRAIN-constant drop only; canonical R92 raw357',
        query_batch_rule='one complete named dataset; each pair independently fits all its strict TRAIN rows',
        context_rows='all81000/task in natural frequency/order before pair-only exact FP64 tie exclusion',
        aggregate_search=False, calibration=False, coupling=False, cost_sensitive_claim=False,
        formal_score_view='logsoftmax(mean opponent win) for formal CE/report only; not winner probability; not ranking',
        kv_cache=False, allow_cpu_offload=False, flash_attention_only=True,
        resource_guard=dict(max_rss_gib=8.5, max_seconds_per_call=1800),
        resource_witness='first full TSP VAL before supervision; no extra subset call',
        context_runtime_cost='K*(K-1)/2 independent pair contexts, one resident model; shortcuts explicitly counted',
        fresh_TEST_graph_statistics_measured=False, pretraining_budget_matched=False,
        test_development_exposed=True, wandb_mode='offline', source_sha256=source_hashes(),
        identities={str(path): file_hash(path) for path in (args.prepared_root / 'prepared.json',
            args.data_root / 'scenario_manifest.json', args.prepared_root / 'source_cache_manifest.json',
            args.prepared_root / 'common_full_train_sbs.json')})
    write_once(args.root / 'prepare_identity.json', protocol)
    write_once(args.protocol_lock, protocol)
    from .r80_reporting import write_csv
    write_csv(args.root / 'prep/pair_support.csv', support)
    ready = dict(experiment='R100', cli_ready=True, preparation_complete=True, tasks=18,
        protocol_lock=str(args.protocol_lock), protocol_sha256=json_hash(protocol),
        optimizer_updates=0, gpu_launched=False, automatic_retry=False,
        entrypoint='code.V4.r100_experiment', root=str(args.root),
        pairs_per_named_full18_split=sum(v['pairs'] for v in relations.values()))
    write_once(args.root / 'prep/cli_ready.json', ready)
    print(json.dumps(ready), flush=True)
    return protocol


def validate_protocol(path):
    config = read_json(path)
    if (config.get('experiment') != 'R100' or config.get('seed') != 2 or config.get('arm') != 'A'
            or config.get('rows_per_problem') != ROWS or config.get('optimizer_updates') != 0
            or config.get('fine_tuning') is not False or config.get('historical_TRAIN_labels_in_context') is not True
            or config.get('current_query_labels_in_forward') is not False
            or config.get('label_rule') != LABEL_RULE or config.get('inference_rule') != INFERENCE_RULE
            or config.get('probability_semantics') != PROBABILITY_SEMANTICS
            or config.get('selection_rule') != SELECTION_RULE or config.get('kv_cache') is not False
            or config.get('allow_cpu_offload') is not False or config.get('flash_attention_only') is not True
            or config.get('full_pool_winner_filter') is not False or config.get('context_subsampling') is not False
            or any(config.get(key) is not False for key in ('aggregate_search', 'calibration', 'coupling', 'cost_sensitive_claim'))
            or config.get('constant_shortcut') != 'exact one class or no strict rows only; no strict rows -> .5'
            or config.get('resource_guard') != dict(max_rss_gib=8.5, max_seconds_per_call=1800)
            or any(set(config.get(key, {})) != set(PROBLEMS) for key in
                   ('contexts', 'relations', 'baselines', 'candidate_pools', 'candidate_pool_ids'))
            or any(v['rows'] != 81000 for v in config['contexts'].values())
            or any(v['rows'] != 81000 for v in config['relations'].values())
            or set(config.get('query_inputs', {}).get('val', {})) != set(PROBLEMS)):
        raise ValueError('Require predeclared frozen full18 strict-pair binary R100 protocol')
    runtime = runtime_config()
    if any(config.get(key) != value for key, value in runtime.items()):
        raise ValueError('Original R97 public classifier parameters/runtime/source/weights changed')
    for key in ('source_sha256', 'runtime_source_sha256', 'identities'):
        for source, digest in config[key].items():
            if file_hash(source) != digest:
                raise ValueError('Scientific-lock source identity changed: ' + source)
    if source_hashes() != config['source_sha256']:
        raise ValueError('R100 implementation source set changed after preparation')
    return config


def require_test_lock(root, config):
    root = Path(root)
    lock = previous.require_test_lock(root, config)
    bank = read_json(root / 'checkpoint.json')
    if (bank.get('experiment') != 'R100' or bank.get('weights_updated') is not False
            or bank.get('protocol_sha256') != json_hash(config) or bank.get('selection_rule') != SELECTION_RULE
            or lock.get('selection_rule') != SELECTION_RULE
            or any(bank.get(key) != config.get(key) for key in BANK_KEYS)):
        raise PermissionError('Require unique unchanged full18 R100 VAL context/weight lock')
    return lock


def preflight_inputs(args, config):
    """Complete scientific binding before importing CUDA or constructing a model."""
    for key in ('root', 'data_root', 'prepared_root', 'feature_root', 'relation_root'):
        if str(getattr(args, key)) != config[key]:
            raise ValueError('Runtime path differs from locked protocol: ' + key)
    if read_json(args.root / 'prepare_identity.json') != config:
        raise ValueError('Protocol differs from CPU scientific preparation')
    for problem in PROBLEMS:
        fingerprint = previous.query_fingerprint(problem, 'val', args)
        previous.verify_query_fingerprint(fingerprint, config['query_inputs']['val'][problem])
        with load_context(args.context_root, problem) as context, load_relations(args.relation_root, problem) as relations:
            validate_relation_binding(context, relations.metadata)
            if (context_identity(context, problem, args, fingerprint) != config['contexts'][problem]
                    or json_hash(relations.metadata) != config['relations'][problem]['metadata_sha256']):
                raise ValueError('Context/strict-label scientific lock changed')
            for item in relations.metadata['identity']['source_files'].values():
                if file_hash(item['path']) != item['sha256']:
                    raise ValueError('Original TRAIN label/context source changed: ' + item['path'])


def query_inputs(problem, split, args, config, context):
    if split == 'test':
        require_test_lock(args.root, config)
    return previous.query_inputs(problem, split, args, config, context)


def guarded_call(function, config, torch):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with previous.call_guard(config, torch) as timing:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result = function()
    return result, timing


@contextmanager
def fitted(problem, args, config, torch):
    started = time.monotonic()
    context = load_context(args.context_root, problem)
    relations = None
    try:
        context_seconds = time.monotonic()-started
        labels_started = time.monotonic()
        relations = load_relations(args.relation_root, problem)
        relation_seconds = time.monotonic()-labels_started
        if (json_hash(context.metadata) != config['contexts'][problem]['metadata_sha256']
                or json_hash(relations.metadata) != config['relations'][problem]['metadata_sha256']):
            raise ValueError('Context/strict labels changed after lock')
        model = FrozenPairwiseContext(context, relations, lambda: public_pair_classifier(config, args.device),
            cleanup=torch.cuda.empty_cache, call=lambda function: guarded_call(function, config, torch))
        yield model, dict(problem=problem, stage='load_context', seconds=time.monotonic()-started,
            context_read_and_verify_seconds=context_seconds, relation_read_and_verify_seconds=relation_seconds,
            train_rows=len(context.x), methods=len(context.pool), pairs=len(model.pairs))
    finally:
        if relations is not None:
            relations.close()
        context.close()
        gc.collect()
        torch.cuda.empty_cache()


def predict(model, x, ids, config, torch):
    started = time.monotonic()
    probabilities = model.predict_features(x, ids)
    support = model.relations.metadata['pair_support']
    expected = {r['pair_slot'] for r in support if r['shortcut'] == 'none'}
    for stage in ('fit_pair', 'predict_pair'):
        actual = [r['pair_slot'] for r in model.call_records if r['stage'] == stage]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Require exactly one independent fit/predict for each nonconstant pair')
    return probabilities, dict(stage='predict', seconds=time.monotonic()-started,
        train_rows=len(model.context.x), query_rows=len(x), methods=len(model.context.pool),
        pairs=len(model.pairs), sequential_pair_fits=len(expected), sequential_pair_predictions=len(expected),
        constant_pair_shortcuts=len(support)-len(expected), kv_cache=False, context_reencoded=True,
        includes_all_sequential_pair_fits=True, excludes_graph_to_statistics_time=True)


def saved_prediction(problem, probabilities, raw, source_ids, nodes, config):
    from .r100_reporting import load_predictions, metric_row
    from .r97_reporting import SAVED_KEYS
    scores = mean_opponent_win(probabilities)
    rank = stable_mean_win_rank(scores, config['candidate_pool_ids'][problem])
    logits, formal = formal_score_view(scores)
    rows = len(scores)
    data = dict(pair_win_probabilities=probabilities, mean_opponent_win=scores,
        logits=logits, scores=scores, formal_score_softmax=formal, pred=rank[:, 0].astype(np.int64),
        costs=np.asarray(raw['costs']), winner=np.asarray(raw['winner']), pool=np.asarray(raw['pool']),
        pool_ids=np.asarray(config['candidate_pool_ids'][problem], dtype=np.int64),
        indices=np.arange(rows, dtype=np.int64), source_indices=np.arange(rows, dtype=np.int64),
        release_source_indices=np.asarray(source_ids, dtype=np.int64), nodes=np.asarray(nodes, dtype=np.int64))
    view = load_predictions(data, '<in-memory R100>')
    row = metric_row(view, config['baselines'][problem], problem, 'pending')
    metrics = {name: row[key]/100. if key.startswith('top') else row[key] for key, name in SAVED_KEYS.items()}
    metrics.update(n=rows, native_correct=row['native_correct'], sbs_name=row['sbs_name'],
        sbs_selection_split='train', winner_rule=previous.WINNER_RULE, inference_rule=INFERENCE_RULE,
        formal_ce=row['formal_ce'], optimizer_updates=0)
    return metrics, data


def summarize(results):
    from .r97_reporting import SAVED_KEYS
    macro = {'macro_'+name: float(np.mean([v[name] for v in results.values()])) for name in SAVED_KEYS.values()}
    return dict(per_problem=results, macro=macro, native_total=sum(v['n'] for v in results.values()),
        native_correct=sum(v['native_correct'] for v in results.values()),
        all_aggregation='equal weight per problem; mixed raw mean_cost descriptive only')


def evaluate_split(split, args, config, torch, timings):
    if split == 'test':
        require_test_lock(args.root, config)
    packed, results = {}, {}
    directory = args.root / 'evaluation' / split / 'selected_best'
    directory.mkdir(parents=True, exist_ok=False)
    for problem in PROBLEMS:
        with fitted(problem, args, config, torch) as (model, loading):
            timings.append(dict(loading, split=split))
            try:
                started = time.monotonic()
                x, ids = query_inputs(problem, split, args, config, model.context)
                timings.append(dict(problem=problem, split=split, stage='cached_query_inputs',
                    seconds=time.monotonic()-started, query_rows=len(x),
                    fresh_graph_to_statistics_measured=False, includes_input_fingerprint_verification=True))
                status(args.root, 'predicting_pairs', problem=problem, split=split, query_rows=len(x))
                probabilities, timing = predict(model, x, ids, config, torch)
                timings.extend(dict(r, problem=problem, split=split) for r in model.call_records)
                timings.append(dict(timing, problem=problem, split=split))
                if problem == 'TSP' and split == 'val':
                    write_once(args.root / 'resource_witness.json', dict(complete=True, labels_loaded=False,
                        accuracy_computed=False, context_rows=81000, query_rows=len(x),
                        pair_probability_shape=list(probabilities.shape), extra_subset_call=False,
                        checkpoint_sha256=config['public_checkpoint_sha256'], context_loading=loading, predict=timing))
                # The complete unlabeled inference returns before evaluation supervision is opened.
                raw = previous.native_labels(problem, split, args, config)
                all_x = np.load(args.feature_root / ('TRAIN_VAL' if split == 'val' else 'TEST') /
                    (problem+'.npy'), mmap_mode='r', allow_pickle=False)
                try:
                    offset = 81000 if split == 'val' else 0
                    metrics, data = saved_prediction(problem, probabilities, raw, ids, all_x[offset:offset+len(ids), 0], config)
                finally:
                    _close(all_x)
                results[problem] = metrics
                np.savez_compressed(directory / (problem+'.npz'), **data)
                packed.update({problem+'__'+key: value for key, value in data.items()})
                print('[R100]', split, problem, json.dumps(metrics), flush=True)
                del x, ids, probabilities, raw, data
            finally:
                write_json(args.root / 'timing.json', dict(records=timings))
    result = dict(summarize(results), complete=True, experiment='R100', arm='A', split=split,
        state_hash=json_hash(config), optimizer_updates=0, winner_rule=previous.WINNER_RULE,
        inference_rule=INFERENCE_RULE, sbs_reference='R84 full81000 TRAIN-fitted, frozen')
    np.savez_compressed(directory / 'rank000.npz', **packed)
    write_once(args.root / (split+'_results.json'), result)
    return result


def lib_query_inputs(problem, dataset, args, config, context, torch):
    """Private input-only R98 adaptation, never a supervision-opening loader."""
    from code.unified_selector.registry import DATA_ROOT, constraint_bits, problem_descriptor
    from code.unified_selector.data import _coord_dist_code
    from .r75_data import _CPUUnpickler
    from .r92_tree_features import features
    from .local_geometry import local_geometry
    lock = require_test_lock(args.root, config)
    path = DATA_ROOT / dataset / 'dataset.pkl'
    binding = dict(dataset=dataset, problem=problem, path=str(path.resolve()), sha256=file_hash(path),
        checkpoint_sha256=lock['checkpoint_sha256'], labels_loaded=False)
    write_once(args.root / 'lib_input_locks' / (dataset+'.json'), binding)
    with path.open('rb') as stream:
        instances = _CPUUnpickler(stream).load()
    rows, sizes = [], []
    for start in range(0, len(instances), 16):
        nodes = []
        for inst in instances[start:start+16]:
            if problem == 'TSP':
                node = torch.as_tensor(inst[0], dtype=torch.float32)
            elif problem == 'CVRP':
                loc, depot, demand = (torch.as_tensor(inst[key][0], dtype=torch.float32)
                                      for key in ('loc', 'depot', 'demand'))
                node = torch.cat((torch.cat((depot, torch.zeros(1, 1), torch.ones(1, 1)), 1),
                    torch.cat((loc, demand[:, None], torch.zeros(len(loc), 1)), 1)), 0)
            else:
                raise ValueError('Only native TSPLIB/CVRPLIB declared')
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
            cbits=torch.tensor([constraint_bits(problem)]*batch_size, device=args.device),
            problem_desc=torch.tensor([problem_descriptor(problem)]*batch_size, device=args.device),
            coord_dist=torch.tensor([_coord_dist_code(problem, start+i, dataset, len(instances))
                                    for i in range(batch_size)], device=args.device))
        batch['node_geom'] = local_geometry(node[..., :2], mask).float()
        x, _ = features(batch)
        rows.append(x)
        sizes.extend(lengths)
    if file_hash(path) != binding['sha256']:
        raise ValueError('LIB input changed during feature extraction')
    x = np.concatenate(rows)
    ids = 2**62 + (0 if dataset == 'TSPLIB' else 2**32) + np.arange(len(x), dtype=np.int64)
    if np.intersect1d(ids, context.source_ids).size:
        raise ValueError('LIB namespace overlaps complete TRAIN base IDs')
    return np.ascontiguousarray(x[:, context.kept_columns]), ids, np.asarray(sizes, dtype=np.int64)


def evaluate_libs(args, config, torch, timings):
    from .r77_lib import checked_raw
    require_test_lock(args.root, config)
    results = {}
    for dataset, problem in (('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')):
        with fitted(problem, args, config, torch) as (model, loading):
            timings.append(dict(loading, split=dataset))
            try:
                started = time.monotonic()
                x, ids, nodes = lib_query_inputs(problem, dataset, args, config, model.context, torch)
                timings.append(dict(problem=problem, split=dataset, stage='native_input_to_statistics',
                    seconds=time.monotonic()-started, query_rows=len(x), includes_native_input_loading=True,
                    labels_loaded=False, fresh_graph_to_statistics_measured=True))
                probabilities, timing = predict(model, x, ids, config, torch)
                timings.extend(dict(r, problem=problem, split=dataset) for r in model.call_records)
                timings.append(dict(timing, problem=problem, split=dataset))
                raw = checked_raw(problem, dataset, config)
                if raw['pool'] != config['candidate_pools'][problem] or raw['pool_ids'] != config['candidate_pool_ids'][problem]:
                    raise ValueError('Native LIB pool differs from single frozen bank')
                metrics, data = saved_prediction(problem, probabilities, raw, ids, nodes, config)
                directory = args.root / 'lib_evaluation/predictions'
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / (dataset+'.npz'), **data)
                results[dataset] = dict(metrics, problem=problem, complete=True,
                    source_directory=str(raw['directory']), state_hash=json_hash(config))
                del x, ids, probabilities, raw, data
            finally:
                write_json(args.root / 'timing.json', dict(records=timings))
    write_once(args.root / 'lib_results.json', results)
    return results


def ensure_fresh_run(root):
    names = ('attempt.json', 'status.json', 'protocol.json', 'checkpoint.json', 'final_test_lock.json',
        'test_input_lock.json', 'evaluation', 'val_results.json', 'test_results.json', 'lib_evaluation',
        'lib_results.json', 'run_complete.json', 'resource_witness.json', 'timing.json', 'wandb_receipt.json',
        'postprocess_status.json', 'full_model_statistics', 'RESULTS.md', 'RESULTS.pdf')
    present = [name for name in names if os.path.lexists(Path(root) / name)]
    if present:
        raise RuntimeError('Preserve existing R100 output; no implicit retry: ' + ', '.join(present))


def run(args):
    args = normalize_args(args)
    config = validate_protocol(args.protocol_lock)
    ensure_fresh_run(args.root)
    preflight_inputs(args, config)
    import torch
    if args.device != 'cuda:0' or not torch.cuda.is_available():
        raise RuntimeError('Use exact common-launcher assigned cuda:0; no CPU inference fallback')
    write_once(args.root / 'attempt.json', dict(started_utc=utc(), automatic_retry=False))
    write_once(args.root / 'protocol.json', config)
    timings, tracker = [], None
    try:
        import wandb
        tracker = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
            name='R100_frozen_pairwise_binary_context', config=config, mode=args.wandb_mode, dir=str(args.root))
        write_once(args.root / 'wandb_receipt.json', dict(enabled=True, mode='offline', project='selector',
            entity='yjkds-southern-university-of-science-technology', run_id=tracker.id, path=str(tracker.dir), optimizer_updates=0))
        val = evaluate_split('val', args, config, torch, timings)
        bank = dict(experiment='R100', protocol_sha256=json_hash(config), weights_updated=False,
            selection_rule=SELECTION_RULE, **{key: config[key] for key in BANK_KEYS})
        write_once(args.root / 'checkpoint.json', bank)
        write_once(args.root / 'final_test_lock.json', dict(experiment='R100', arm='A',
            checkpoint=str(args.root / 'checkpoint.json'), checkpoint_sha256=file_hash(args.root / 'checkpoint.json'),
            state_hash=json_hash(config), selected_without_test=True, per_problem_assembly=False,
            locked_utc=utc(), selection_rule=SELECTION_RULE, validation_macro=val['macro']))
        require_test_lock(args.root, config)
        previous.lock_test_inputs(args, config)
        test = evaluate_split('test', args, config, torch, timings)
        libs = evaluate_libs(args, config, torch, timings)
        tracker.log({'VAL/'+key: value for key, value in val['macro'].items()})
        tracker.log({'TEST/'+key: value for key, value in test['macro'].items()})
        write_once(args.root / 'run_complete.json', dict(complete=True, completed_utc=utc(),
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
    value.add_argument('--feature-root', type=Path, default=previous.FEATURES)
    value.add_argument('--data-root', type=Path, default=previous.DATA)
    value.add_argument('--prepared-root', type=Path, default=previous.PREPARED)
    value.add_argument('--protocol-lock', type=Path)
    value.add_argument('--device', default='cpu')
    value.add_argument('--wandb-mode', choices=('offline',), default='offline')
    return value


def main(argv=None):
    args = normalize_args(parser().parse_args(argv))
    if args.command == 'prepare':
        return prepare(args)
    if args.command == 'report':
        from .r100_reporting import report
        return report(args)
    return run(args)


if __name__ == '__main__':
    main()
