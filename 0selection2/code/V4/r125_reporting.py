"""R125 same-lock LIB inference and CPU-only PDF reporting.

After the runner has completed its locked TEST::

    python -m code.V4.r125_reporting all --device cpu

``lib_stage(args)`` and ``report(args)`` accept the namespace from ``parser``.
Only LIB may run inference, using r125_experiment.load_checkpoint. NSS is
strictly a saved-score, reduced-pool diagnostic: no training or inference.
Neither stage modifies the runner's VAL/TEST archives or shared modules.
"""

import argparse
import copy
import fcntl
import importlib
import json
import os
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from code.unified_selector.registry import DATA_ROOT, GLOBAL_SOLVERS, POOLS, PROBLEMS, S2I
from . import r80_reporting as common


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R125_no_bq_lehd_source_embedding'
DEFAULT_NSS_ROOT = V4 / 'runs/R84_new90k_attention/nss_reference'
ARM, ARMS, MAX_UPDATES = 'code_seed2', ('code_seed2',), 57600
FACTORY = 'code.V4.r125_experiment'
LIBS = {'TSPLIB': 'TSP', 'CVRPLIB': 'CVRP'}
LIB_ROWS = {'TSPLIB': 49, 'CVRPLIB': 100}
SPLIT_ROWS = {'train': 81000, 'val': 4500, 'test': 4500}
REMOVED = frozenset(('BQ', 'LEHD'))
ORIGINAL_POOLS = {problem: tuple(POOLS[problem]) for problem in PROBLEMS}
WINNER_RULE = 'native FP64 costs, argmin after deleting BQ/LEHD; not FP32 argmin'
NSS_DESCRIPTION = ('Fixed historical NSS logits, reduced-pool diagnostic only; '
                   'NOT trained on the R125 reduced pool; no retraining or new inference')
read_json, write_json, file_hash = common.read_json, common.write_json, common.file_hash


def expected_pool(problem):
    removed = set(REMOVED) | {'RELD_MTL'}
    if 'B' in problem:
        removed.update(('RouteFinder', 'MoSES_RF', 'MoSES_CaDA'))
    return ['reld_mt' if name == 'RELD_MOEL' else name
            for name in ORIGINAL_POOLS[problem] if name not in removed]


def validate_config(config):
    if (config.get('experiment') != 'R125' or config.get('arm') != ARM
            or config.get('model_factory') != FACTORY + '.load_checkpoint'
            or config.get('seed') != 2 or config.get('rows_per_problem') != SPLIT_ROWS
            or config.get('logical_global_batch') != 1024
            or config.get('max_successful_updates') != MAX_UPDATES
            or config.get('warmup_updates') != 4320 or config.get('eval_every_updates') != 1440
            or set(config.get('candidate_pools', {})) != set(PROBLEMS)):
        raise ValueError('Require the frozen R125 full18 single-group protocol')
    aliases = config.get('scenario_solver_aliases', {})
    for problem in PROBLEMS:
        pool, ids = config['candidate_pools'][problem], config['candidate_pool_ids'][problem]
        canonical = [aliases.get(name, name) for name in pool]
        baseline = config['baselines'][problem]
        if (pool != expected_pool(problem) or ids != sorted(set(ids))
                or canonical != [GLOBAL_SOLVERS[i] for i in ids]
                or baseline.get('selection_split') != 'train'
                or set(baseline['mean_cost_by_solver']) != set(pool)
                or baseline['sbs_solver'] not in pool
                or baseline['sbs_slot'] != pool.index(baseline['sbs_solver'])):
            raise ValueError('Changed R125 candidates or non-TRAIN SBS: ' + problem)


def bind_selected_val(val, lock, test_identity):
    """Bind a capture to the official lock in a copy, never rewrite the archive."""
    capture = val.get('archive_capture_identity', val['identity'])
    if (capture.get('experiment') != 'R125' or capture.get('arm') != ARM
            or capture.get('split') != 'val'
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or capture.get('successful_updates') != lock['selected_update']
            or capture.get('state_hash') != lock['state_hash']
            or not capture.get('model_state_sha256')
            or capture['model_state_sha256'] != test_identity.get('model_state_sha256')
            or test_identity.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or test_identity.get('state_hash') != lock['state_hash']
            or test_identity.get('successful_updates') != lock['selected_update']
            or test_identity.get('experiment') != 'R125' or test_identity.get('arm') != ARM
            or test_identity.get('split') != 'test'):
        raise ValueError('Selected VAL capture differs from the unique locked TEST model state')
    official = copy.deepcopy(val)
    official['archive_capture_identity'] = copy.deepcopy(capture)
    official['identity'] = dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
    official['checkpoint_binding'] = dict(checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], selected_update=lock['selected_update'],
        model_state_sha256=capture['model_state_sha256'], archive_identity_unchanged=True)
    return official


def locked_inputs(root, arm=ARM):
    run, config, lock, test, val = common.locked_inputs(Path(root), arm,
        experiment='R125', arms={ARM: MAX_UPDATES})
    validate_config(config)
    return run, config, lock, test, bind_selected_val(val, lock, test['identity'])


def validate_prediction(data, problem, config, *, expected_rows=None):
    if (data.pool != config['candidate_pools'][problem]
            or data.pool_ids.tolist() != config['candidate_pool_ids'][problem]
            or set(data.pool) & REMOVED or expected_rows is not None and data.n != expected_rows):
        raise ValueError('R125 prediction population/candidates differ: ' + problem)
    np.testing.assert_array_equal(data.indices, np.arange(data.n))
    np.testing.assert_array_equal(data.winner, data.costs.argmin(1),
        err_msg='Recompute winner from the reduced FP64 pool, never FP32 argmin')


def load_saved_splits(run, config, lock, test, val):
    helper = common.reporting_helpers()
    predictions, metrics, methods = {}, [], []
    for split, result, path in (
            ('VAL', val, common.selected_val_path(run, lock)),
            ('TEST', test, run / 'evaluation/test/selected_best/rank000.npz')):
        captured = result.get('archive_capture_identity', result['identity'])
        if read_json(path.parent / 'identity.json') != captured:
            raise ValueError('Saved ' + split + ' identity differs from its selected summary')
        rows = []
        with np.load(path, allow_pickle=False) as archive:
            for problem in PROBLEMS:
                data = helper.load_predictions(archive, path, problem + '__')
                validate_prediction(data, problem, config, expected_rows=SPLIT_ROWS[split.lower()])
                key = problem + '__release_source_indices'
                data.release_source_indices = archive[key] if key in archive else None
                predictions[split, problem] = data
                saved = result['per_problem'][problem]
                if saved['n'] != data.n:
                    raise ValueError('Saved metrics and predictions have different row counts')
                rows.append(common.metric_row(problem, split, data, saved,
                    config['baselines'][problem], config, lock, path))
                methods.extend(dict(row, model='R125/' + ARM) for row in helper.method_rows(
                    problem, split, saved, data, Path(config['data_root']).name))
        metrics.extend([common.macro_row(split, rows, result), *rows])
    for row in metrics:
        row['winner_rule'] = WINNER_RULE
    return predictions, metrics, methods


def reduced_raw(raw, problem, config):
    """Slice existing native FP64 labels; never open solver result contents."""
    pool = config['candidate_pools'][problem]
    if (set(pool) & REMOVED or list(raw['pool']) != list(ORIGINAL_POOLS[problem])
            or raw['costs'].dtype != np.float64 or not np.isfinite(raw['costs']).all()
            or np.any(raw['costs'] <= 0)
            or raw['costs'].shape != (len(raw['winner']), len(raw['pool']))):
        raise ValueError('Invalid original native LIB pool or FP64 costs')
    columns = [raw['pool'].index(name) for name in pool]
    costs = raw['costs'][:, columns].copy()
    return dict(raw, pool=list(pool), pool_ids=list(config['candidate_pool_ids'][problem]),
        costs=costs, winner=costs.argmin(1).astype(np.int64), kept_columns=columns)


def native_lib(problem, dataset, config):
    directory = DATA_ROOT / dataset
    with (directory / 'raw_label.pkl').open('rb') as stream:
        labels = pickle.load(stream)
    raw = dict(directory=str(directory), pool=list(ORIGINAL_POOLS[problem]),
        costs=np.asarray([labels[str(i)]['cost'] for i in range(len(labels))], dtype=np.float64),
        winner=np.asarray([labels[str(i)]['ind'] for i in range(len(labels))], dtype=np.int64))
    if len(labels) != LIB_ROWS[dataset]:
        raise ValueError('Native LIB population changed: ' + dataset)
    return reduced_raw(raw, problem, config)


def lib_loader(problem, dataset, raw, batch_size, device):
    import torch
    from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem
    from .tensor_loader import TensorBatchLoader
    from .r41_frozen_evaluation import attach_geometry

    data = UnifiedProblemDataset(problem, dataset, coord_augment=0)
    # Own dataset and batch: no cached-batch or registry mutation. The native
    # file still has its original columns even if a factory restores a new pool.
    data.pool_order = [S2I[name] for name in ORIGINAL_POOLS[problem]]
    data.K_p = len(data.pool_order)
    data.mask = [int(i in data.pool_order) for i in range(len(GLOBAL_SOLVERS))]
    batch = collate_single_problem([data[i] for i in range(len(data))])
    np.testing.assert_array_equal(batch['costs'][:, raw['kept_columns']].numpy(),
        raw['costs'].astype(np.float32))
    if len(data) != len(raw['winner']):
        raise ValueError('LIB input and label rows differ')
    batch['pool_ids'] = torch.tensor(raw['pool_ids'], dtype=torch.long)
    batch['mask'] = torch.zeros_like(batch['mask'])
    batch['mask'][:, raw['pool_ids']] = 1
    batch['costs'] = torch.from_numpy(raw['costs'].astype(np.float32))
    batch['ind'] = torch.from_numpy(raw['winner'])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    loader = TensorBatchLoader(batch, batch_size, False, False)
    attach_geometry(loader, device)
    return loader


def evaluate_libs(run, config, lock, test, device_name='cpu', batch_size=2, phase=None):
    import torch
    from .nss_comparison import infer
    from .r76_experiment import fixed_sbs_metrics
    from .r77_data import parameter_hash
    from .r77_lib import checked_predictions
    from .train import configure_torch

    device = torch.device(device_name)
    if device.type not in ('cpu', 'cuda') or batch_size not in (1, 2):
        raise ValueError('Use CPU or an explicitly assigned CUDA device and LIB batch size 1/2')
    configure_torch()
    torch.set_num_threads(2)
    directory = run / 'lib_evaluation'
    directory.mkdir(exist_ok=True)
    model, results = None, {}
    for dataset, problem in LIBS.items():
        if phase:
            phase('lib_evaluation', dataset=dataset)
        raw = native_lib(problem, dataset, config)
        prediction = directory / 'predictions' / (dataset + '.npz')
        protocol_path = directory / (dataset + '_protocol.json')
        result_path = directory / (dataset + '_results.json')
        identity = dict(experiment='R125', arm=ARM, dataset=dataset, problem=problem,
            checkpoint_sha256=lock['checkpoint_sha256'], epoch=lock['epoch'],
            selected_update=lock['selected_update'], state_hash=lock['state_hash'],
            model_state_sha256=test['identity']['model_state_sha256'], candidate_pool=raw['pool'],
            pool_ids=raw['pool_ids'], model_factory=FACTORY + '.load_checkpoint',
            winner_rule=WINNER_RULE, removed_candidates=sorted(REMOVED),
            data_version='original native LIB; existing FP64 costs sliced to R125 pool',
            sources={name: dict(path=str(Path(raw['directory']) / name),
                sha256=file_hash(Path(raw['directory']) / name)) for name in ('dataset.pkl', 'raw_label.pkl')},
            inference_rule=config['inference_rule'], batch_size=batch_size,
            training=False, solver_execution=False, new_solver_answers=False, checkpoint_reselection=False)
        if protocol_path.exists():
            previous = read_json(protocol_path)
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ValueError('Existing LIB protocol changed: ' + dataset)
            if not prediction.is_file() and previous['device'] != str(device):
                raise ValueError('Retry unfinished LIB on its recorded device')
        else:
            if prediction.exists() or result_path.exists():
                raise ValueError('Unattributed existing LIB output: ' + dataset)
            write_json(protocol_path, dict(identity, device=str(device), created_utc=common.utc()))
        started, reused = time.monotonic(), prediction.is_file()
        if reused:
            saved = checked_predictions(prediction, raw, lock)
        else:
            if model is None:
                factory = importlib.import_module(FACTORY)
                model, state = factory.load_checkpoint(run / 'best.pt', device)
                if (state.get('args') != config or state.get('epoch') != lock['epoch']
                        or state.get('successful_updates') != lock['selected_update']
                        or parameter_hash(model.state_dict()) != identity['model_state_sha256']):
                    raise ValueError('R125 factory did not restore the exact locked model/buffers')
                model.eval().requires_grad_(False)
                del state
            loader = lib_loader(problem, dataset, raw, batch_size, device)
            prediction.parent.mkdir(parents=True, exist_ok=True)
            temporary = prediction.with_name(f'.{dataset}.{os.getpid()}.npz')
            try:
                with torch.inference_mode():
                    infer(model, loader, batch_size, raw, temporary)
                with np.load(temporary, allow_pickle=False) as archive:
                    saved = {key: archive[key] for key in archive.files}
                saved.update(winner_fp32_argmin=raw['costs'].astype(np.float32).argmin(1),
                    checkpoint_sha256=np.asarray(lock['checkpoint_sha256']),
                    checkpoint_epoch=np.asarray(lock['epoch']), checkpoint_update=np.asarray(lock['selected_update']))
                np.savez_compressed(temporary, **saved)
                checked_predictions(temporary, raw, lock)
                if parameter_hash(model.state_dict()) != identity['model_state_sha256']:
                    raise ValueError('LIB inference modified locked model parameters/buffers')
                temporary.replace(prediction)
            finally:
                temporary.unlink(missing_ok=True)
            del loader
        fixed, pred = fixed_sbs_metrics(saved['scores'], raw, config['baselines'][problem])
        np.testing.assert_array_equal(pred, saved['pred'])
        result = dict(protocol=read_json(protocol_path), train_sbs_reference=fixed,
            inference_reused=reused, seconds=time.monotonic() - started, completed_utc=common.utc())
        write_json(result_path, result)
        results[dataset] = result
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
        raise ValueError('R125 checkpoint changed during LIB evaluation')
    return results


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root.resolve(), args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(phase=name, updated_utc=common.utc(), **fields))

        try:
            results = evaluate_libs(run, config, lock, test, args.device, args.lib_batch_size, phase)
            write_json(run / 'lib_stage_complete.json', dict(complete=True, datasets=list(results),
                checkpoint_sha256=lock['checkpoint_sha256'], nss_inference=False, updated_utc=common.utc()))
            phase('complete')
            return results
        except Exception as error:
            phase('failed', error=str(error), retry='lib only; retain training and TEST')
            raise


def saved_libs(run, config, lock, test, predictions, metrics, methods):
    helper = common.reporting_helpers()
    for dataset, problem in LIBS.items():
        path = run / 'lib_evaluation/predictions' / (dataset + '.npz')
        result = read_json(run / 'lib_evaluation' / (dataset + '_results.json'))
        protocol = result['protocol']
        common.check_identity(protocol, lock, dataset)
        if (protocol.get('experiment') != 'R125' or protocol.get('arm') != ARM
                or protocol.get('dataset') != dataset
                or protocol.get('model_factory') != FACTORY + '.load_checkpoint'
                or protocol.get('model_state_sha256') != test['identity']['model_state_sha256']
                or protocol.get('candidate_pool') != config['candidate_pools'][problem]
                or protocol.get('pool_ids') != config['candidate_pool_ids'][problem]
                or protocol.get('winner_rule') != WINNER_RULE):
            raise ValueError('LIB is not from the locked R125 model/reduced pool')
        with np.load(path, allow_pickle=False) as archive:
            data = helper.load_predictions(archive, path)
            if (str(archive['checkpoint_sha256'].item()) != lock['checkpoint_sha256']
                    or int(archive['checkpoint_epoch']) != lock['epoch']
                    or int(archive['checkpoint_update']) != lock['selected_update']):
                raise ValueError('LIB archive checkpoint identity differs')
        validate_prediction(data, problem, config, expected_rows=LIB_ROWS[dataset])
        predictions[dataset, problem] = data
        fixed = result['train_sbs_reference']
        row = common.metric_row(problem, dataset, data, fixed, config['baselines'][problem], config, lock, path)
        row.update(winner_rule=WINNER_RULE, data_version=protocol['data_version'])
        metrics.append(row)
        methods.extend(dict(row, model='R125/' + ARM) for row in helper.method_rows(
            problem, dataset, fixed, data, protocol['data_version']))


def reselect_nss(archive, path, current):
    """Reorder aligned saved rows/columns and reselect logits in the legal pool."""
    helper = common.reporting_helpers()
    pool, costs, indices = archive['pool'].tolist(), archive['costs'], archive['indices']
    scores = archive['logits'] if 'logits' in archive else archive['scores']
    if (len(pool) != len(set(pool)) or set(pool) - set(current.pool) != set(REMOVED)
            or not set(current.pool).issubset(pool) or costs.dtype != np.float64
            or costs.shape != (current.n, len(pool)) or scores.shape != costs.shape
            or not np.isfinite(costs).all() or np.any(costs <= 0) or not np.isfinite(scores).all()
            or indices.shape != (current.n,) or len(set(indices.tolist())) != current.n
            or set(indices.tolist()) != set(current.indices.tolist())):
        raise ValueError('NSS diagnostic requires aligned saved logits/FP64 costs on the same full population')
    lookup = {int(index): row for row, index in enumerate(indices)}
    rows = np.asarray([lookup[int(index)] for index in current.indices])
    columns = [pool.index(name) for name in current.pool]
    reduced_costs, reduced_scores = costs[rows][:, columns], scores[rows][:, columns]
    np.testing.assert_allclose(reduced_costs, current.costs, rtol=0, atol=1e-12,
        err_msg='NSS must use the same frozen instances and original FP64 cost columns')
    release = getattr(current, 'release_source_indices', None)
    if release is not None:
        if 'source_indices' not in archive:
            raise ValueError('NSS is missing frozen release source IDs')
        np.testing.assert_array_equal(archive['source_indices'][rows], release)
    order = np.argsort(current.pool_ids, kind='stable')
    pred = order[np.argmax(reduced_scores[:, order], axis=1)]
    return helper.Predictions(path=path, pool=list(current.pool), pred=pred,
        winner=reduced_costs.argmin(1).astype(np.int64), costs=reduced_costs,
        indices=current.indices.copy(), source_indices=None, pool_ids=current.pool_ids.copy(),
        score_ties=int(((reduced_scores == reduced_scores.max(1, keepdims=True)).sum(1) > 1).sum()))


def nss_comparisons(config, lock, predictions, nss_root=DEFAULT_NSS_ROOT):
    nss_root = Path(nss_root)
    protocol_path = nss_root / 'protocol.json'
    plan = read_json(protocol_path) if protocol_path.is_file() else {}
    source_root = Path(config.get('source_dataset_root', config['data_root'])).resolve()
    rows, alignment, diagnostics = [], {}, {}
    for dataset, problem, split in (('TSP TEST', 'TSP', 'TEST'), ('CVRP TEST', 'CVRP', 'TEST'),
                                    ('TSPLIB', 'TSP', 'TSPLIB'), ('CVRPLIB', 'CVRP', 'CVRPLIB')):
        current = predictions[split, problem]
        name = 'test' if split == 'TEST' else split
        path = nss_root / problem / (name + '_predictions.npz')
        metadata_path = path.with_name(name + '_metrics.json')
        row = dict(dataset=dataset, problem=problem, n=current.n,
            data_version=Path(config['data_root']).name if split == 'TEST' else 'original native LIB',
            source_data_version=source_root.name, candidate_pool=json.dumps(current.pool),
            nss_model=NSS_DESCRIPTION, current_model='R125/' + ARM,
            nss_top1_native_pct=None, nss_mean_cost=None, current_top1_native_pct=current.top1_pct,
            current_mean_cost=current.mean_cost, current_checkpoint_sha256=lock['checkpoint_sha256'],
            nss_checkpoint_sha256=None, winner_rule=WINNER_RULE, nss_prediction_path=str(path),
            current_prediction_path=str(current.path), status='missing_saved_NSS',
            missing_reason='Saved same-population logits, metadata or protocol unavailable',
            nss_retrained=False, nss_reduced_pool_training=False, nss_new_inference=False,
            score_tie_rule='ascending global solver ID after pool reduction (diagnostic, not original NSS topk)')
        if plan and path.is_file() and metadata_path.is_file():
            expected = plan.get('checkpoints', {}).get(problem, {}).get('sha256')
            metadata = read_json(metadata_path)
            if not expected or metadata.get('checkpoint', {}).get('sha256') != expected:
                raise ValueError('Saved NSS checkpoint does not match its historical protocol')
            if split == 'TEST' and (Path(plan.get('data_root', '')).resolve() != source_root
                    or current.n != SPLIT_ROWS['test']):
                row['missing_reason'] = 'No saved NSS logits on the same frozen 4500-row TEST release'
            else:
                with np.load(path, allow_pickle=False) as archive:
                    if len(archive['indices']) != current.n:
                        row['missing_reason'] = 'Saved NSS population differs; old3895 TEST is not reusable'
                    elif 'logits' not in archive and 'scores' not in archive:
                        row['missing_reason'] = 'No saved NSS logits/scores; fixed old pred alone cannot be reselected'
                    else:
                        data = reselect_nss(archive, path, current)
                        diagnostics[split, problem] = data
                        alignment[dataset] = dict(checked_rows=current.n, all_FP64_cost_columns_equal=True,
                            original_candidate_pool=archive['pool'].tolist(), reduced_candidate_pool=data.pool,
                            source_prediction_sha256=file_hash(path), same_release_source_ids=
                                getattr(current, 'release_source_indices', None) is not None)
                        row.update(nss_top1_native_pct=data.top1_pct, nss_mean_cost=data.mean_cost,
                            nss_checkpoint_sha256=expected, status='fixed_NSS_reduced_pool_diagnostic',
                            missing_reason='', alignment_checked_rows=current.n)
        rows.append(row)
    return rows, alignment, diagnostics


def make_panels(problem, test, lib=None):
    pool = test.pool if test is not None else lib.pool
    sources = [('TEST', test, test.n if test is not None else SPLIT_ROWS['test'])]
    if problem in LIBS.values():
        dataset = next(name for name, task in LIBS.items() if task == problem)
        sources.append((dataset, lib, LIB_ROWS[dataset]))
    panels = []
    for split, data, n in sources:
        if data is not None and data.pool != pool:
            raise ValueError('Distribution candidate order differs: ' + problem)
        for kind, field in (('model_selection', 'pred'), ('true_winner', 'winner')):
            counts = None if data is None else np.bincount(getattr(data, field), minlength=len(pool))
            if counts is not None and int(counts.sum()) != n:
                raise ValueError('Distribution count does not cover the dataset')
            panels.append(dict(split=split, kind=kind, n=n, counts=counts,
                top1=data.top1_pct if data is not None and field == 'pred' else None))
    return panels


def distribution_figure(problem, pool, panels, version, model, lock, *, diagnostic=False):
    helper = common.reporting_helpers()
    fig = common._distribution(helper, problem, pool, panels, version, model, lock)
    for label in fig.texts:
        value = label.get_text()
        if 'native/FP64 ind' in value:
            label.set_text(value.replace('native/FP64 ind', 'FP64 argmin after BQ/LEHD deletion'))
    if diagnostic:
        fig.texts[-1].set_text('Fixed NSS reduced-pool diagnostic; not reduced-pool training; no new inference.')
    else:
        fig.texts[-1].set_text('R125: BQ/LEHD deleted; same VAL lock; ascending global-ID score ties; no Rxx mixing.')
    return fig


def learning_figure(run, lock):
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('R125/' + ARM + ' | one locked VAL checkpoint')
    for key in ('loss', 'ce', 'cmp', 'risk', 'kl'):
        points = [(row['successful_updates'], row['optimization'][key]) for row in history
                  if key in row.get('optimization', {})]
        if points:
            axes[0, 0].plot(*zip(*points), label=key)
    axes[0, 0].set_title('TRAIN interval mean objectives')
    for ax, key, factor, title in ((axes[0, 1], 'macro_top1', 100., 'Top1 (%)'),
            (axes[1, 0], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
            (axes[1, 1], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
        for split in ('train', 'val'):
            points = [(row['successful_updates'], row[split]['macro'][key] * factor)
                      for row in history if row.get(split) is not None and key in row[split].get('macro', {})]
            if points:
                ax.plot(*zip(*points), label=split)
        ax.set_title(title)
    for ax in axes.flat:
        ax.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
        ax.set_xlabel('Successful optimizer updates')
        ax.grid(alpha=.2)
        if ax.lines and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return fig


def build_reports(run, config, lock, test, val, predictions, metrics, methods,
                  comparisons, alignment, nss_predictions=None):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    output = run / 'full_model_statistics'
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    model, version = 'R125/' + ARM, Path(config['data_root']).name
    for name, rows in (('metrics_by_problem.csv', metrics), ('single_method_comparison.csv', methods),
                       ('nss_comparison.csv', comparisons)):
        common.write_csv(output / name, rows)
    panels, distributions = {}, []
    for problem in PROBLEMS:
        lib = next((dataset for dataset, task in LIBS.items() if task == problem), None)
        data = predictions['TEST', problem]
        panels[problem] = make_panels(problem, data, predictions.get((lib, problem)))
        distributions.extend(dict(row, model=model, checkpoint_sha256=lock['checkpoint_sha256'],
            winner_rule=WINNER_RULE) for row in helper.distribution_rows(problem, panels[problem], data.pool, version))
    common.write_csv(output / 'selection_distribution.csv', distributions)
    pdf, dist_pdf = output / 'MODEL_STATISTICS.pdf', output / 'selection_distributions.pdf'
    temporary = pdf.with_name(f'.model.{os.getpid()}.pdf')
    dist_temporary = dist_pdf.with_name(f'.distributions.{os.getpid()}.pdf')
    headings = ['Split / Problem', 'Top1 (%)', 'SBS Top1 (%)', 'mean_cost', 'SBS cost',
                'Oracle cost', 'vs_SBS (%)', 'vs_Oracle (%)', 'actual regret (%)']
    try:
        with PdfPages(temporary) as report, PdfPages(dist_temporary) as distribution:
            fig = common._page(helper, model + ' | Single VAL-best full18 results',
                'Frozen new90k, BQ/LEHD deleted; FP64 reduced-pool argmin; no TEST/LIB reselection.', model, lock, 1)
            fig.text(.045, .85, f'epoch {lock["epoch"]} | update {lock["selected_update"]} | seed2 | fresh initialization', fontsize=12)
            fig.text(.045, .81, 'Data: ' + version, fontsize=9)
            overall = [next(row for row in metrics if row['split'] == split and row['problem'] == 'ALL')
                       for split in ('VAL', 'TEST')]
            helper.add_table(fig, headings, [helper.metric_cells(row, row['split'] + ' ALL') for row in overall],
                [.045, .63, .91, .13], font=10)
            number = lambda value, places: 'MISSING' if value is None else f'{value:.{places}f}'
            helper.add_table(fig, ['Dataset', 'n', 'Fixed NSS Top1 (%)', 'R125 Top1 (%)', 'Fixed NSS cost', 'R125 cost'],
                [[row['dataset'], str(row['n']), number(row['nss_top1_native_pct'], 4),
                  number(row['current_top1_native_pct'], 4), number(row['nss_mean_cost'], 8),
                  number(row['current_mean_cost'], 8)] for row in comparisons], [.045, .33, .91, .23], font=10)
            notes = [
                'NSS: fixed historical logits reselected after deleting BQ/LEHD; NOT a matched reduced-pool training control.',
                'Only existing aligned4500-row TEST / original LIB predictions are reused; unavailable entries are MISSING.',
                'SBS: reduced-pool full81000 TRAIN only. Oracle: minimum legal FP64 cost per instance.',
                'ALL: equal weight per problem. Macro raw costs are descriptive, not cross-problem pooled costs.',
                'TRAIN81000 / VAL4500 / TEST4500 per problem. Retained holdouts informed earlier R&D, not fully blind.',
                'R80D: architecture reference only; older data / original pool, NOT a matched experimental control.',
                'Deletion changes Oracle and winner labels; a Top1 change alone does not establish an architectural cause.']
            for index, note in enumerate(notes):
                fig.text(.045, .27 - index * .035, note, fontsize=9.5)
            report.savefig(fig)
            plt.close(fig)
            for page, split in enumerate(('VAL', 'TEST'), 2):
                fig = common._page(helper, model + ' | ' + split + ': ALL and18 tasks',
                    'Top1 higher is better; cost/regret lower is better. Same checkpoint for every task.', model, lock, page)
                rows = [row for row in metrics if row['split'] == split]
                helper.add_table(fig, headings + ['TRAIN SBS'],
                    [helper.metric_cells(row, row['problem']) + [row['sbs_name']] for row in rows],
                    [.045, .22, .91, .62], font=10, bold_first=True)
                for index, note in enumerate((
                        'vs_SBS=100*(mean_cost/SBS_cost-1); vs_Oracle=100*(mean_cost/Oracle_cost-1).',
                        'actual regret=mean_i[100*(selected_i-oracle_i)/oracle_i]; NOT a ratio of means.',
                        'Native FP64 argmin after candidate deletion; never FP32 cost argmin.')):
                    fig.text(.045, .17 - index * .04, note, fontsize=10)
                report.savefig(fig)
                plt.close(fig)
            fig = common._page(helper, model + ' | Native LIB and provenance',
                'Original LIB inputs/costs; BQ/LEHD removed; winner recomputed in FP64; no new solver answers.', model, lock, 4)
            helper.add_table(fig, headings, [helper.metric_cells(row, row['split']) for row in metrics if row['split'] in LIBS],
                [.045, .66, .91, .17], font=10)
            for index, note in enumerate(('SHA256: ' + lock['checkpoint_sha256'],
                    'Selection: ' + lock['rule'], 'Factory: ' + FACTORY + '.load_checkpoint',
                    'Inference: ' + config['inference_rule'], 'TSP6 / CVRP8; other16 task pools unchanged.',
                    'TRAIN-only normalization restored from the same best.pt; no LIB refit.')):
                fig.text(.045, .59 - index * .06, note, fontsize=9)
            report.savefig(fig)
            plt.close(fig)
            curve = learning_figure(run, lock)
            curve.savefig(output / 'learning_curves.pdf')
            curve.savefig(output / 'learning_curves.png', dpi=140)
            report.savefig(curve)
            plt.close(curve)
            for problem in PROBLEMS:
                data = predictions['TEST', problem]
                fig = distribution_figure(problem, data.pool, panels[problem], version, model, lock)
                fig.savefig(figures / (problem + '_selection_distribution.pdf'))
                fig.savefig(figures / (problem + '_selection_distribution.png'), dpi=140)
                report.savefig(fig)
                distribution.savefig(fig)
                plt.close(fig)
        temporary.replace(pdf)
        dist_temporary.replace(dist_pdf)
    finally:
        temporary.unlink(missing_ok=True)
        dist_temporary.unlink(missing_ok=True)
    nss_distributions = []
    if nss_predictions:
        with PdfPages(output / 'nss_selection_distributions.pdf') as diagnostic:
            for problem in ('TSP', 'CVRP'):
                dataset = next(name for name, task in LIBS.items() if task == problem)
                test_data, lib_data = nss_predictions.get(('TEST', problem)), nss_predictions.get((dataset, problem))
                if test_data is None and lib_data is None:
                    continue
                data = test_data if test_data is not None else lib_data
                nss_row = next(row for row in comparisons if row['problem'] == problem and row['nss_checkpoint_sha256'])
                historical_lock = dict(lock, checkpoint_sha256=nss_row['nss_checkpoint_sha256'],
                    epoch='historical VAL-best', selected_update='historical (not R125)')
                diagnostic_panels = make_panels(problem, test_data, lib_data)
                nss_distributions.extend(dict(row, model=NSS_DESCRIPTION, winner_rule=WINNER_RULE,
                    checkpoint_sha256=nss_row['nss_checkpoint_sha256'])
                    for row in helper.distribution_rows(problem, diagnostic_panels, data.pool, version))
                fig = distribution_figure(problem, data.pool, diagnostic_panels, version,
                    'Fixed NSS reduced-pool diagnostic', historical_lock, diagnostic=True)
                diagnostic.savefig(fig)
                plt.close(fig)
        common.write_csv(output / 'nss_selection_distribution.csv', nss_distributions)
    diagnostics = common.diagnostic_pdf(run, output, predictions, methods, model, lock)
    write_json(output / 'protocol.json', dict(experiment='R125', arm=ARM, complete=True,
        checkpoint_lock=lock, selected_val_binding=val.get('checkpoint_binding'), model_factory=FACTORY + '.load_checkpoint',
        data_root=config['data_root'], source_dataset_root=config.get('source_dataset_root'),
        rows_per_problem=SPLIT_ROWS, candidate_pools=config['candidate_pools'],
        candidate_pool_ids=config['candidate_pool_ids'], winner_rule=WINNER_RULE,
        removed_candidates=sorted(REMOVED), sbs_selection_split='full81000 TRAIN, reduced pool',
        all_aggregation='equal-problem macro; raw cost means descriptive only',
        metric_formulas=dict(vs_sbs_pct='100*(mean_cost/SBS_cost-1)',
            vs_oracle_pct='100*(mean_cost/Oracle_cost-1)',
            actual_regret_pct='mean_i[100*(selected_i-oracle_i)/oracle_i]'),
        nss_reference=NSS_DESCRIPTION, nss_alignment=alignment, nss_comparisons=comparisons,
        training=False, inference=False, new_solver_answers=False, checkpoint_reselection=False,
        archive_identity_unchanged=True, distribution_pages=len(PROBLEMS), main_pdf_pages=5 + len(PROBLEMS),
        diagnostics=diagnostics, created_utc=common.utc()))
    lines = ['# R125/' + ARM, '', 'Single VAL-best checkpoint: update ' + str(lock['selected_update']), '',
        '| Split | Top1 (%) | mean_cost | actual regret (%) |', '| --- | ---: | ---: | ---: |']
    for row in overall:
        lines.append(f'| {row["split"]} | {row["top1_pct"]:.4f} | {row["mean_cost"]:.8f} | {row["actual_regret_pct"]:.4f} |')
    lines += ['', 'FP64 winner after deleting BQ/LEHD. SBS: full TRAIN only. ALL: equal-problem macro.',
        NSS_DESCRIPTION + '.', 'Missing NSS entries: ' + ', '.join(row['dataset'] for row in comparisons
            if row['nss_top1_native_pct'] is None),
        'TEST informed earlier R&D; not fully blind. No TEST/LIB checkpoint or inference-rule selection.',
        'Primary: MODEL_STATISTICS.pdf; selection_distributions.pdf; learning_curves.pdf.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(complete=True, output=str(output), model_statistics_pdf=str(pdf),
        selection_distributions_pdf=str(dist_pdf), learning_curves_pdf=str(output / 'learning_curves.pdf'))


def report(args):
    run, config, lock, test, val = locked_inputs(args.root.resolve(), args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(phase='loading_saved_predictions', updated_utc=common.utc()))
            predictions, metrics, methods = load_saved_splits(run, config, lock, test, val)
            saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment, nss = nss_comparisons(config, lock, predictions,
                getattr(args, 'nss_root', None) or DEFAULT_NSS_ROOT)
            result = build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment, nss)
            write_json(run / 'postprocess_status.json', dict(phase='complete', completed_utc=common.utc(), **result))
            return result
        except Exception as error:
            write_json(run / 'postprocess_status.json', dict(phase='postprocess_failed', updated_utc=common.utc(),
                error=str(error), traceback=traceback.format_exc(), retry='report only; retain training/TEST/LIB'))
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('lib', 'report', 'all', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, default=ARM)
    result.add_argument('--device', default='cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path, default=DEFAULT_NSS_ROOT,
        help='Existing full-pool NSS predictions only; unavailable items are MISSING, never recomputed')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.root, args.nss_root = args.root.resolve(), args.nss_root.resolve()
    if args.command == 'status':
        run = args.root / args.arm
        result = {name: read_json(run / name) if (run / name).is_file() else {'phase': 'not_started'}
                  for name in ('lib_stage_status.json', 'postprocess_status.json')}
    elif args.command == 'lib':
        result = lib_stage(args)
    elif args.command == 'report':
        result = report(args)
    else:
        lib_stage(args)
        result = report(args)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    main()
