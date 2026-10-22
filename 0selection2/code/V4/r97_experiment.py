"""One predeclared frozen TabICL bank, TRAIN context only, full18 evaluation.

The resource witness never opens query supervision. TEST/LIB access follows
one full-bank VAL lock; no task-specific setting or checkpoint selection.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import threading
import time
import traceback

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .r80_reporting import file_hash, json_hash, read_json, utc, write_json
from .r97_model import FrozenContextClassifier, public_classifier


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R97_tabular_in_context'
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
PREPARED = V4 / 'runs/R84_new90k_attention/prepared'
FEATURES = V4 / 'runs/R92_broad_learning/features'
WEIGHT_NAME = 'tabicl-classifier-v2-20260212.ckpt'
WEIGHT_SHA = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
WINNER_RULE = 'native/FP64 raw_label ind; never FP32 argmin'


def status(root, phase, **fields):
    path = Path(root) / 'status.json'
    record = read_json(path) if path.is_file() else {}
    record.update(experiment='R97', arm='A', phase=phase, pid=os.getpid(),
        host=socket.gethostname(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        updated_utc=utc(), optimizer_updates=0, training_unit='frozen_labeled_TRAIN_context',
        automatic_retry=False, **fields)
    write_json(path, record)


def prepare(args):
    from .r97_data import prepare_contexts, load_context
    root = args.root.resolve()
    prepare_contexts(args.prepared_root, args.data_root, args.feature_root, root / 'context')
    runtime = root / 'runtime'
    weights = runtime / 'weights' / WEIGHT_NAME
    if file_hash(weights) != WEIGHT_SHA:
        raise ValueError('Public frozen checkpoint SHA256 mismatch')
    import torch
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    del checkpoint
    if config.get('max_classes') != 10 or config.get('embed_dim') != 128:
        raise ValueError('Pinned TabICL checkpoint architecture changed')
    mgr = dict(offload='gpu', safety_factor=.1, min_batch_size=1,
               use_async=False, max_pinned_memory_mb=0.)
    params = dict(n_estimators=1, batch_size=1, norm_methods=['none'],
        model_path=str(weights), allow_auto_download=False, checkpoint_version=WEIGHT_NAME,
        use_amp=True, use_fa3=False, offload_mode='gpu', random_state=2,
        n_jobs=None, verbose=True, support_many_classes=False,
        inference_config={key: dict(mgr) for key in ('COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG')})
    contexts, pools, ids = {}, {}, {}
    for problem in PROBLEMS:
        context = load_context(root / 'context', problem)
        contexts[problem] = dict(metadata_sha256=json_hash(context.metadata),
            rows=len(context.y), columns=len(context.kept_columns))
        pools[problem], ids[problem] = list(context.pool), list(context.pool_ids)
        context.close()
        del context
    protocol = dict(experiment='R97', arm='A', arms=['A'], seed=2, locked=True,
        task_scope='full18', data_root=str(args.data_root.resolve()),
        prepared_root=str(args.prepared_root.resolve()), feature_root=str(args.feature_root.resolve()),
        library_root=str(runtime / 'python'), rows_per_problem=dict(train=81000, val=4500, test=4500),
        baselines=read_json(args.prepared_root / 'common_full_train_sbs.json'),
        candidate_pools=pools, candidate_pool_ids=ids, solver_names=GLOBAL_SOLVERS,
        contexts=contexts, classifier_parameters=params,
        query_inputs={'val': {problem: query_fingerprint(problem, 'val', args)
                              for problem in PROBLEMS}},
        checkpoint_config=config, public_checkpoint_sha256=WEIGHT_SHA,
        public_checkpoint_revision='4dcd344ece2c00be9e831fdd35bed57b5ad83e19',
        package_version='tabicl==2.0.2', source_revision='ba4bdbc6bb46b9307da3dbc1685109371ec5085f',
        runtime_wheels={p.name: file_hash(p) for p in sorted((runtime / 'wheels').glob('*.whl'))},
        launch=dict(pythonpath=[str(runtime/'python')], source_files=[
            str(p) for p in sorted((runtime/'python').rglob('*.py'))] + [
            str(p) for p in sorted((runtime/'wheels').glob('*.whl'))] + [str(weights)]),
        winner_rule=WINNER_RULE, optimizer_updates=0, fine_tuning=False,
        historical_TRAIN_labels_in_context=True, current_query_labels_in_forward=False,
        context_rows='all81000/task, natural order and frequency, no quota/subset',
        feature_transform='only task-TRAIN-constant columns removed, original order retained',
        query_batch_rule='one complete named dataset per predict call; no query labels or mixed splits',
        checkpoint_rule='one frozen predeclared full18 bank; no VAL hyperparameter search',
        inference_rule='legal-class probability argmax; ascending global-ID exact ties',
        kv_cache=False, allow_cpu_offload=False, flash_attention_only=True,
        resource_guard=dict(max_rss_gib=8.5, max_seconds_per_call=1800),
        witness=dict(problem='TSP', train_rows=81000, query_split='val', query_rows=32,
                     labels_loaded=False, accuracy_computed=False),
        pretraining_budget_matched=False, test_development_exposed=True,
        wandb_mode='offline')
    path = root / 'protocol_lock.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Existing R97 protocol differs; do not overwrite')
    if not path.exists():
        write_json(path, protocol)
    print(json.dumps(dict(ready_for_resource_witness=True, protocol=str(path),
                         context_rows=81000, tasks=18)), flush=True)
    return protocol


def validate_protocol(path):
    config = read_json(path)
    if (config.get('experiment') != 'R97' or config.get('seed') != 2
            or config.get('fine_tuning') is not False
            or config.get('current_query_labels_in_forward') is not False
            or config.get('rows_per_problem') != dict(train=81000, val=4500, test=4500)
            or set(config.get('contexts', {})) != set(PROBLEMS)
            or any(x['rows'] != 81000 for x in config['contexts'].values())
            or config.get('public_checkpoint_sha256') != WEIGHT_SHA):
        raise ValueError('Use the predeclared full18 TRAIN-only frozen context protocol')
    if file_hash(config['classifier_parameters']['model_path']) != WEIGHT_SHA:
        raise ValueError('Public weights changed')
    return config


@contextmanager
def call_guard(config, torch):
    """Self-only sampled RAM guard plus a bounded call deadline, no sibling control."""
    import psutil
    process, stop, peak = psutil.Process(), threading.Event(), [0]
    previous = signal.getsignal(signal.SIGALRM)

    def interrupt(signum, frame):
        raise RuntimeError('R97 resource/time guard stopped its own call')

    def monitor():
        while not stop.wait(.1):
            peak[0] = max(peak[0], process.memory_info().rss)
            if peak[0] > config['resource_guard']['max_rss_gib'] * 2**30:
                os.kill(os.getpid(), signal.SIGALRM)
                return

    signal.signal(signal.SIGALRM, interrupt)
    signal.alarm(config['resource_guard']['max_seconds_per_call'])
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    record = dict(started_utc=utc())
    started = time.monotonic()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield record
        torch.cuda.synchronize()
    finally:
        signal.alarm(0)
        stop.set()
        thread.join(timeout=2)
        signal.signal(signal.SIGALRM, previous)
        record.update(seconds=time.monotonic()-started,
            peak_sampled_rss_bytes=max(peak[0], process.memory_info().rss),
            process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            sampled_guard_is_not_a_hard_memory_limit=True)


def install_gpu_only_offload_guard(config):
    directory = config['library_root']
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module = importlib.import_module('tabicl.model.inference')
    original = module.InferenceManager._resolve_offload_mode

    def resolve(self, *args, **kwargs):
        mode, reason = original(self, *args, **kwargs)
        if mode != module.OffloadMode.GPU:
            raise RuntimeError('CPU/disk fallback rejected before allocation: ' + str(reason))
        return mode, reason

    def allocate(self, mode, shape, dtype):
        if mode != module.OffloadMode.GPU or self.exe_device.type != 'cuda':
            raise RuntimeError('R97 output allocation must stay on the assigned GPU')
        # Upstream catches a failed CUDA allocation and then allocates on CPU.
        # Keep its successful GPU path, but propagate failure before that fallback.
        out = module.torch.empty(shape, dtype=dtype, device=self.exe_device)
        return out, dict(mode=mode.name, shape=shape, dtype=str(dtype))

    module.InferenceManager._resolve_offload_mode = resolve
    module.InferenceManager._allocate_output_buffer = allocate


def fitted(problem, args, config, torch):
    from .r97_data import load_context
    started = time.monotonic()
    context = load_context(args.context_root, problem)
    context_seconds = time.monotonic()-started
    record = config['contexts'][problem]
    if (json_hash(context.metadata) != record['metadata_sha256']
            or len(context.y) != record['rows'] or len(context.kept_columns) != record['columns']
            or list(context.pool) != config['candidate_pools'][problem]
            or list(context.pool_ids) != config['candidate_pool_ids'][problem]):
        raise ValueError('Historical context changed after locking')
    model = FrozenContextClassifier(context, lambda: public_classifier(config, args.device))
    status(args.root, 'fitting_context', problem=problem)
    with call_guard(config, torch) as timing:
        model.fit()
    return model, dict(timing, problem=problem, stage='fit_context', train_rows=len(context.y),
        context_read_and_verify_seconds=context_seconds, total_seconds=time.monotonic()-started)


def query_fingerprint(problem, split, args):
    from .r92_tree_features import COLUMNS, FEATURE_DEFINITION
    phase = 'TRAIN_VAL' if split == 'val' else 'TEST'
    metadata_path = args.feature_root / phase / (problem+'.json')
    array_path = metadata_path.with_suffix('.npy')
    source_path = args.data_root / (problem+split) / 'input_metadata.npz'
    metadata = read_json(metadata_path)
    identity = metadata['identity']
    splits = ['train', 'val'] if split == 'val' else ['test']
    rows = dict(train=81000, val=4500) if split == 'val' else dict(test=4500)
    if (metadata.get('complete') is not True or identity.get('splits') != splits
            or identity.get('rows') != rows or identity.get('problem') != problem
            or identity.get('data_root') != str(args.data_root.resolve())
            or identity.get('feature_definition') != FEATURE_DEFINITION
            or metadata.get('shape') != [sum(rows.values()), len(COLUMNS)]
            or metadata.get('dtype') != 'float32'):
        raise ValueError('Canonical query split/schema/source differs from the frozen release')
    with np.load(source_path, allow_pickle=False) as source:
        ids = source['source_index'].copy()
    if ids.shape != (4500,) or ids.dtype.kind not in 'iu':
        raise ValueError('Query source/base-instance IDs must retain frozen row order')
    ids = ids.astype(np.int64)
    return dict(receipt_sha256=file_hash(metadata_path), feature_sha256=file_hash(array_path),
        input_metadata_sha256=file_hash(source_path),
        ordered_source_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())


def verify_query_fingerprint(actual, expected):
    if actual != expected:
        raise ValueError('Query inputs/source-index order changed after locking')


def lock_test_inputs(args, config):
    lock = require_test_lock(args.root, config)
    path = args.root/'test_input_lock.json'
    if path.exists():
        raise RuntimeError('TEST input binding already exists; no implicit replacement')
    write_json(path, dict(checkpoint_sha256=lock['checkpoint_sha256'],
        inputs={problem: query_fingerprint(problem, 'test', args) for problem in PROBLEMS},
        labels_loaded=False, created_after_unique_VAL_lock=True))


def query_inputs(problem, split, args, config, context):
    if split == 'test':
        lock = require_test_lock(args.root, config)
        bound = read_json(args.root/'test_input_lock.json')
        if bound.get('checkpoint_sha256') != lock['checkpoint_sha256']:
            raise ValueError('TEST input binding belongs to another locked bank')
        expected = bound['inputs'][problem]
    else:
        expected = config['query_inputs']['val'][problem]
    verify_query_fingerprint(query_fingerprint(problem, split, args), expected)
    phase, offset = ('TRAIN_VAL', 81000) if split == 'val' else ('TEST', 0)
    metadata = read_json(args.feature_root / phase / (problem + '.json'))
    if (metadata.get('complete') is not True
            or metadata['identity']['data_root'] != config['data_root']
            or metadata['identity']['problem'] != problem):
        raise ValueError('Use the canonical input-only feature cache for this release')
    x = np.load(args.feature_root / phase / (problem + '.npy'), mmap_mode='r', allow_pickle=False)
    rows = config['rows_per_problem'][split]
    result = np.ascontiguousarray(x[offset:offset+rows, context.kept_columns])
    with np.load(args.data_root / (problem+split) / 'input_metadata.npz', allow_pickle=False) as source:
        ids = source['source_index'].astype(np.int64)
    if result.shape != (rows, len(context.kept_columns)) or ids.shape != (rows,):
        raise ValueError('Canonical query row order/shape changed')
    if hashlib.sha256(ids.tobytes()).hexdigest() != expected['ordered_source_ids_sha256']:
        raise ValueError('Query source-index order changed while loading')
    return result, ids


def predict(model, x, ids, config, torch):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with call_guard(config, torch) as timing:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            probabilities = model.predict_features(x, ids)
    return probabilities, dict(timing, query_rows=len(x), train_rows=len(model.context.y),
        kv_cache=False, context_reencoded=True,
        excludes_graph_to_statistics_time=True)


def native_labels(problem, split, args, config):
    from .r75_data import _CPUUnpickler
    directory = args.data_root / (problem+split)
    metadata = read_json(directory / 'metadata.json')
    if metadata['methods'] != config['candidate_pools'][problem]:
        raise ValueError('Native cost columns differ from the frozen legal candidates')
    with (directory / 'raw_label.pkl').open('rb') as stream:
        labels = _CPUUnpickler(stream).load()
    n = config['rows_per_problem'][split]
    costs = np.asarray([labels[str(i)]['cost'] for i in range(n)], dtype=np.float64)
    winner = np.asarray([labels[str(i)]['ind'] for i in range(n)], dtype=np.int64)
    if not np.array_equal(winner, costs.argmin(1)) or not np.isfinite(costs).all():
        raise ValueError('Native FP64 labels are not aligned')
    return dict(costs=costs, winner=winner, pool=metadata['methods'],
                pool_ids=config['candidate_pool_ids'][problem])


def saved_prediction(problem, probabilities, raw, source_ids, nodes, config):
    from .r76_experiment import fixed_sbs_metrics
    logits = np.log(np.maximum(probabilities, 1e-300))
    metrics, pred = fixed_sbs_metrics(logits, raw, config['baselines'][problem])
    rows = len(probabilities)
    data = dict(logits=logits, scores=logits, probabilities=probabilities, pred=pred,
        costs=raw['costs'], winner=raw['winner'], pool=np.asarray(raw['pool']),
        pool_ids=np.asarray(config['candidate_pool_ids'][problem], dtype=np.int64),
        indices=np.arange(rows), source_indices=np.arange(rows), release_source_indices=source_ids,
        nodes=np.asarray(nodes, dtype=np.int64))
    return metrics, data


def evaluate_split(split, args, config, torch, timings, initial=None):
    from .r80_experiment import summarize
    packed, results = {}, {}
    for problem in PROBLEMS:
        model, fit_timing = initial.pop() if initial and problem == 'TSP' else fitted(problem, args, config, torch)
        timings.append(dict(fit_timing, split=split))
        x, ids = query_inputs(problem, split, args, config, model.context)
        status(args.root, 'predicting', problem=problem, split=split, query_rows=len(x))
        probabilities, timing = predict(model, x, ids, config, torch)
        timings.append(dict(timing, problem=problem, split=split, stage='predict'))
        # Query supervision is opened only after inference, solely for reporting.
        raw = native_labels(problem, split, args, config)
        all_x = np.load(args.feature_root / ('TRAIN_VAL' if split == 'val' else 'TEST') /
                        (problem+'.npy'), mmap_mode='r')
        offset = 81000 if split == 'val' else 0
        metrics, data = saved_prediction(problem, probabilities, raw, ids,
            all_x[offset:offset+len(ids), 0], config)
        results[problem] = metrics
        directory = args.root / 'evaluation' / split / 'selected_best'
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / (problem+'.npz'), **data)
        packed.update({problem+'__'+key: value for key, value in data.items()})
        print('[R97]', split, problem, json.dumps(metrics), flush=True)
        del model, x, all_x, probabilities, raw, data
        gc.collect()
        torch.cuda.empty_cache()
        write_json(args.root / 'timing.json', dict(records=timings))
    result = summarize(results)
    result.update(complete=True, experiment='R97', arm='A', split=split,
        state_hash=json_hash(config), sbs_reference='R84 full81000 TRAIN-fitted, frozen',
        training_unit='frozen_context_no_SGD', optimizer_updates=0)
    np.savez_compressed(directory / 'rank000.npz', **packed)
    write_json(args.root / (split+'_results.json'), result)
    return result


def require_test_lock(root, config):
    lock = read_json(root / 'final_test_lock.json')
    if (lock.get('selected_without_test') is not True
            or lock.get('per_problem_assembly') is not False
            or lock.get('state_hash') != json_hash(config)
            or lock.get('checkpoint_sha256') != file_hash(root / 'checkpoint.json')):
        raise PermissionError('TEST/LIB require the unique unchanged full18 frozen bank VAL lock')
    return lock


def evaluate_libs(args, config, torch, timings):
    from .r77_lib import checked_raw
    from .train import make_loader
    from .r41_frozen_evaluation import attach_geometry
    from .r92_tree_features import features, input_batch
    require_test_lock(args.root, config)
    results = {}
    for dataset, problem in (('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')):
        model, timing = fitted(problem, args, config, torch)
        timings.append(dict(timing, split=dataset))
        # Loader has supervision for evaluation; whitelist it out before phi.
        feature_started = time.monotonic()
        loader = make_loader(problem, dataset, 1024, 0, coord_augment=0,
                             shuffle=False, cache_device=torch.device(args.device))
        attach_geometry(loader, torch.device(args.device))
        x, _ = features(input_batch(loader.batch))
        timings.append(dict(problem=problem, split=dataset, stage='native_input_to_statistics',
            seconds=time.monotonic()-feature_started, query_rows=len(x), includes_native_input_loading=True))
        query = np.ascontiguousarray(x[:, model.context.kept_columns])
        ids = -1-np.arange(len(query), dtype=np.int64)
        status(args.root, 'predicting', problem=problem, split=dataset)
        probabilities, timing = predict(model, query, ids, config, torch)
        timings.append(dict(timing, problem=problem, split=dataset, stage='predict'))
        raw = checked_raw(problem, dataset, config)
        metrics, data = saved_prediction(problem, probabilities, raw, ids,
                                         loader.batch['n'].cpu().numpy(), config)
        directory = args.root / 'lib_evaluation/predictions'
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / (dataset+'.npz'), **data)
        results[dataset] = dict(metrics, problem=problem, complete=True,
            source_directory=str(raw['directory']), state_hash=json_hash(config))
        del model, loader, query, x, probabilities
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.root / 'lib_results.json', results)
    write_json(args.root / 'timing.json', dict(records=timings))
    return results


def ensure_fresh_run(root):
    names = ('attempt.json', 'status.json', 'checkpoint.json', 'final_test_lock.json',
        'test_input_lock.json', 'evaluation', 'val_results.json', 'test_results.json',
        'lib_evaluation', 'lib_results.json', 'run_complete.json', 'postprocess_status.json',
        'full_model_statistics', 'RESULTS.md', 'RESULTS.pdf')
    present = [name for name in names if (root/name).exists()]
    if present:
        raise RuntimeError('Preserve existing R97 output; no implicit retry/overwrite: ' + ', '.join(present))


def run(args):
    import torch
    config = validate_protocol(args.protocol_lock)
    if args.device != 'cuda:0' or not torch.cuda.is_available():
        raise RuntimeError('Use the exact launcher-assigned CUDA device')
    ensure_fresh_run(args.root)
    write_json(args.root / 'attempt.json', dict(started_utc=utc(), automatic_retry=False))
    write_json(args.root / 'protocol.json', config)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    install_gpu_only_offload_guard(config)
    tracker, timings = None, []
    try:
        import wandb
        tracker = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
            name='R97_frozen_tabular_context', config=config, mode='offline', dir=str(args.root))
        write_json(args.root/'wandb_receipt.json', dict(enabled=True, mode='offline',
            project='selector', entity='yjkds-southern-university-of-science-technology',
            run_id=tracker.id, path=str(tracker.dir), reason='predeclared isolated offline protocol'))
        status(args.root, 'resource_witness')
        model, fit_timing = fitted('TSP', args, config, torch)
        x, ids = query_inputs('TSP', 'val', args, config, model.context)
        probabilities, timing = predict(model, x[:32], ids[:32], config, torch)
        write_json(args.root / 'resource_witness.json', dict(complete=True, labels_loaded=False,
            accuracy_computed=False, context_rows=len(model.context.y), columns=x.shape[1],
            probability_shape=list(probabilities.shape), fit=fit_timing, predict=timing,
            checkpoint_config=config['checkpoint_config'], checkpoint_sha256=WEIGHT_SHA))
        del x, ids, probabilities
        initial = [(model, fit_timing)]
        del model
        val = evaluate_split('val', args, config, torch, timings, initial=initial)
        del initial
        bank = dict(experiment='R97', protocol_sha256=json_hash(config),
            public_checkpoint_sha256=WEIGHT_SHA, contexts=config['contexts'],
            inference_rule=config['inference_rule'], weights_updated=False,
            selection_rule='frozen_single_predeclared_bank_no_VAL_search')
        write_json(args.root / 'checkpoint.json', bank)
        write_json(args.root / 'final_test_lock.json', dict(experiment='R97', arm='A',
            checkpoint=str((args.root/'checkpoint.json').resolve()),
            checkpoint_sha256=file_hash(args.root/'checkpoint.json'), state_hash=json_hash(config),
            selected_without_test=True, per_problem_assembly=False, locked_utc=utc(),
            selection_rule=bank['selection_rule'], validation_macro=val['macro']))
        lock_test_inputs(args, config)
        test = evaluate_split('test', args, config, torch, timings)
        libs = evaluate_libs(args, config, torch, timings)
        tracker.log({'VAL/'+k: v for k, v in val['macro'].items()})
        tracker.log({'TEST/'+k: v for k, v in test['macro'].items()})
        write_json(args.root / 'run_complete.json', dict(complete=True, completed_utc=utc(),
            full18_VAL=True, full18_TEST=True, LIB=list(libs), optimizer_updates=0,
            checkpoint_sha256=file_hash(args.root/'checkpoint.json')))
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
    value.add_argument('--data-root', type=Path, default=DATA)
    value.add_argument('--prepared-root', type=Path, default=PREPARED)
    value.add_argument('--feature-root', type=Path, default=FEATURES)
    value.add_argument('--context-root', type=Path, default=ROOT/'context')
    value.add_argument('--protocol-lock', type=Path, default=ROOT/'protocol_lock.json')
    value.add_argument('--device', default='cpu')
    value.add_argument('--wandb-mode', choices=('offline',), default='offline')
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'prepare':
        return prepare(args)
    if args.command == 'report':
        from .r97_reporting import report
        return report(args)
    return run(args)


if __name__ == '__main__':
    main()
