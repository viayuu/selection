"""Same-checkpoint R112 LIB inference and offline full18 PDF/CSV delivery.

Reuses only saved historical NSS predictions. A missing reference is explicitly
MISSING in PDF and CSV, never a reason to train NSS or rerun current TEST.
"""

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from . import r80_reporting, r84_reporting
from . import r112_experiment as runner
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import parameter_hash


ARMS, BUDGETS = runner.ARMS, runner.BUDGETS
DEFAULT_ROOT, FACTORY = runner.DEFAULT_ROOT, 'code.V4.r112_experiment'
PROBLEMS = tuple(runner.PROBLEMS)
common = runner.private_module(r80_reporting, 'report_common')
shared = runner.private_module(r84_reporting, 'report_shared')
shared.common = common
for module in (common, shared):
    module.DEFAULT_ROOT, module.ARMS, module.BUDGETS, module.FACTORY = DEFAULT_ROOT, ARMS, BUDGETS, FACTORY
LIBS = shared.LIBS
# Same four-dataset contract as R84/R109, without importing unfinished reporters.
NSS_DATASETS = (('TSP TEST', 'TSP', 'TEST'), ('CVRP TEST', 'CVRP', 'TEST'),
                ('TSPLIB', 'TSP', 'TSPLIB'), ('CVRPLIB', 'CVRP', 'CVRPLIB'))


def write_csv(path, rows):
    """R109's small BOM-free atomic writer, adapted locally for publication."""
    if not rows:
        return
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    try:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


common.write_csv = write_csv


def _nss_display(value, precision):
    return 'MISSING' if value is None else format(value, '.' + str(precision) + 'f')


shared._nss_display = _nss_display
runner.adapt_function(shared, r84_reporting.build_reports, (
    ("f'{r[\"nss_top1_native_pct\"]:.6f}'", '_nss_display(r.get("nss_top1_native_pct"), 6)'),
    ("f'{r[\"nss_mean_cost\"]:.9f}'", '_nss_display(r.get("nss_mean_cost"), 9)'),
    ('NSS TEST predictions were freshly computed on these4500 instances; old3895 TEST metrics are not reused.',
     'NSS: saved exact4500/native LIB references only; MISSING entries are not inferred or retrained.')))


def locked_inputs(root, arm):
    run, config, lock, test, val = shared.locked_inputs(Path(root), arm, experiment='R112', arms=BUDGETS)
    runner.validate_checkpoint_config(config)
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    model_hash = parameter_hash(state['model'])
    final = read_json(run / 'final_metrics.json')
    if (final.get('complete') is not True or final.get('test') != test
            or test['identity'].get('model_state_sha256') != model_hash
            or test['identity'].get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError('R112 report requires completed same-model full TEST')
    val = copy.deepcopy(val)
    val['identity'] = runner.selected_capture_proof(run, val, lock, 'val', state)
    runner.selected_capture_proof(run, final['train'], lock, 'train', state)
    return run, config, lock, test, val


def inference_timing(run, config, lock, model_hash):
    value = read_json(Path(run) / 'inference_timing.json')
    count = config['eval_batch_size']
    required = dict(experiment='R112', arm=config['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=model_hash,
        model_factory=runner.MODEL_FACTORY, inference_rule=runner.INFERENCE_RULE,
        parameter_count=config['trainable_parameter_count'], eval_batch_size=count,
        repetitions=3, warmup_batches=1, all_current_model_included=True,
        scope='complete_standalone_model_original_GPU_resident_inputs',
        sample_policy=f'first{count} frozen TEST inputs per problem',
        end_to_end_measured=False, input_labels_read=False, model_reselection=False,
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver execution')
    if any(value.get(k) != v for k, v in required.items()) or not str(value.get('device')).startswith('cuda'):
        raise ValueError('R112 timing must describe the complete same-checkpoint input-only model')
    rows = value.get('per_problem', [])
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('R112 timing must cover all18 problems')
    for row in rows:
        samples = row.get('standalone_model', {}).get('batch_seconds', [])
        mean = row.get('standalone_model', {}).get('mean_seconds_per_instance')
        numbers = [*samples, mean]
        if (row.get('instances') != count or len(samples) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in numbers)
                or not math.isclose(mean, sum(samples) / (3 * count), rel_tol=1e-10)):
            raise ValueError('Invalid timing sample count or per-instance denominator')
    return value


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    model_hash = test['identity']['model_state_sha256']
    inference_timing(run, config, lock, model_hash)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R112', phase=name,
                arm=args.arm, updated_utc=utc(), **fields))

        batch = (min(config['eval_batch_size'], config['microbatch_caps']['TSP'],
                     config['microbatch_caps']['CVRP']) if args.lib_batch_size is None else args.lib_batch_size)
        if batch < 1:
            raise ValueError('LIB batch size must be positive')
        results = common.r80_lib(args.root, run, config, lock, args.device, batch,
            phase, model_hash, factory_module=FACTORY)
        if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during LIB inference')
        write_json(run / 'lib_stage_complete.json', dict(experiment='R112', arm=args.arm,
            complete=True, completed_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=lock['state_hash'], selected_update=lock['selected_update'],
            model_factory=runner.MODEL_FACTORY, model_state_sha256=model_hash, datasets=list(results)))
        phase('complete')
        return results


def nss_comparisons(config, lock, predictions, nss_root):
    helper, rows, alignment = common.reporting_helpers(), [], {}
    root = Path(nss_root)
    plan = read_json(root / 'protocol.json') if (root / 'protocol.json').is_file() else None
    if plan is not None and (Path(plan['data_root']).resolve() != Path(config['data_root']).resolve()
                             or plan.get('retrained_on_new90k') is not False):
        raise ValueError('NSS reference is not saved historical weights on the same data')
    for dataset, problem, split in NSS_DATASETS:
        current = predictions[split, problem]
        name = 'test' if split == 'TEST' else split
        path = root / problem / (name + '_predictions.npz')
        row = dict(dataset=dataset, problem=problem, n=current.n,
            current_model='R112/' + config['arm'], current_model_factory=runner.MODEL_FACTORY,
            current_top1_native_pct=current.top1_pct, current_mean_cost=current.mean_cost,
            current_checkpoint_sha256=lock['checkpoint_sha256'], current_state_hash=lock['state_hash'],
            current_selected_update=lock['selected_update'], current_prediction_path=str(current.path),
            nss_model='R79 saved historical weights; not retrained on new90k',
            data_version=Path(config['data_root']).name if split == 'TEST' else 'original native ' + dataset,
            candidate_pool=json.dumps(list(current.pool)), winner_rule=runner.WINNER_RULE)
        try:
            if plan is None:
                raise FileNotFoundError(root / 'protocol.json')
            if problem not in plan.get('checkpoints', {}):
                raise FileNotFoundError('NSS checkpoint lock missing: ' + problem)
            metadata = read_json(path.with_name(name + '_metrics.json'))
            expected = plan['checkpoints'][problem]['sha256']
            if metadata.get('checkpoint', {}).get('sha256') != expected:
                raise ValueError('Saved NSS checkpoint identity differs: ' + dataset)
            with np.load(path, allow_pickle=False) as archive:
                nss = helper.load_predictions(archive, path, r77=False)
            checks = helper.check_comparison_alignment(current, nss)
            np.testing.assert_allclose(nss.top1_pct, metadata['top1_native_ind'] * 100, rtol=0, atol=1e-9)
            np.testing.assert_allclose(nss.mean_cost, metadata['mean_cost'], rtol=0, atol=1e-9)
        except FileNotFoundError as error:
            row.update(status='missing', nss_top1_native_pct=None, nss_mean_cost=None,
                nss_checkpoint_sha256=None, nss_prediction_path=str(path), missing_reason=str(error))
            alignment[dataset] = dict(status='missing', reason=str(error))
        else:
            row.update(status='complete', nss_top1_native_pct=nss.top1_pct, nss_mean_cost=nss.mean_cost,
                nss_checkpoint_sha256=expected, nss_prediction_path=str(path))
            alignment[dataset] = checks
        rows.append(row)
    return rows, alignment


def report(args):
    """CPU-only rendering: no factory construction, inference or reselection."""
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'report.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lib = read_json(run / 'lib_stage_complete.json')
        required = dict(experiment='R112', arm=args.arm, complete=True,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], model_factory=runner.MODEL_FACTORY,
            model_state_sha256=test['identity']['model_state_sha256'])
        if any(lib.get(k) != v for k, v in required.items()) or set(lib.get('datasets', [])) != set(LIBS):
            raise ValueError('PDF requires both LIBs from the unique selected model')
        timing = inference_timing(run, config, lock, test['identity']['model_state_sha256'])
        predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
        shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
        training = read_json(run / 'final_metrics.json')['train']
        train_rows, helper = [], common.reporting_helpers()
        path = Path(training['prediction_archive'])
        with np.load(path, allow_pickle=False) as archive:
            for problem in PROBLEMS:
                data = helper.load_predictions(archive, path, problem + '__')
                shared.validate_population(data, problem, 'TRAIN', config)
                train_rows.append(common.metric_row(problem, 'TRAIN', data, training['per_problem'][problem],
                    config['baselines'][problem], config, lock, path))
        metrics = [common.macro_row('TRAIN', train_rows, training), *train_rows, *metrics]
        comparisons, alignment = nss_comparisons(config, lock, predictions, args.nss_root)
        result = shared.build_reports(run, config, lock, test, val, predictions,
            metrics, methods, comparisons, alignment)
        output = run / 'full_model_statistics'
        common.write_csv(output / 'inference_timing.csv', [dict(problem=r['problem'], instances=r['instances'],
            milliseconds_per_instance=1000 * r['standalone_model']['mean_seconds_per_instance'])
            for r in timing['per_problem']])
        provenance = read_json(output / 'protocol.json')
        provenance.update(checkpoint_rule=runner.CHECKPOINT_RULE, main_protocol_lock=config['main_protocol_lock'],
            model_params=config['model_params'], trainable_parameter_count=config['trainable_parameter_count'],
            inference_timing=timing, nss_reference=comparisons,
            evidence_boundary='TEST development-exposed; VAL-only checkpoint selection; ALL problem-equal; '
                'TRAIN-only SBS; missing NSS references are not retrained or inferred')
        write_json(output / 'protocol.json', provenance)
        write_json(run / 'report_complete.json', dict(result, experiment='R112', arm=args.arm, complete=True,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], completed_utc=utc()))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('lib', 'report'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lib-batch-size', type=int)
    parser.add_argument('--nss-root', type=Path, default=runner.V4 / 'runs/R84_new90k_attention/nss_reference')
    args = parser.parse_args(argv)
    args.root, args.nss_root = args.root.expanduser().resolve(), args.nss_root.expanduser().resolve()
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()
