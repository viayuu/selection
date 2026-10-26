"""Saved-only R113 joint closeout and one explicitly requested publication.

No torch/model imports, forward passes, solvers, checkpoint selection or launches.
Only this module's receipts/reports are written. R112 helpers are read-only.

Interface assumptions: the wave plan supplies three independent R113 job
receipts (R114 is not a closeout prerequisite); missing historical/target36
evidence is reported, never substituted or reselected. A publication attempt is
durable before staging and can never be automatically retried.
"""

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np

from . import r112_finish as common


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R113_executable_process'
GIT_ROOT = common.GIT_ROOT
ARMS = ('A_scratch', 'B_snapshot', 'C_transition')
PROBLEMS, REFERENCES = common.PROBLEMS, common.REFERENCES
METRICS, COUNTS, DELTAS = common.METRICS, common.COUNTS, common.DELTAS
NotReady = common.NotReady
read, write, digest, remember = common.read, common.write, common.digest, common.remember
json_hash, write_csv, utc = common.json_hash, common.write_csv, common.utc
load_run, saved_split, paired = common.load_run, common.saved_split, common.paired
metric_row, macro = common.metric_row, common.macro
git, publication_mutex, dependency_files = common.git, common.publication_mutex, common.dependency_files
TARGET_LIST = V4 / 'runs/R84_new90k_attention/comparison_artifacts/target_list.json'
CODE_NAMES = ('r113_finish.py', 'test_r113_finish.py', 'r113_model.py', 'r113_processor.py',
    'r113_process_data.py', 'r113_torch_inputs.py', 'r113_pretrain.py', 'r113_experiment.py',
    'r113_reporting.py', 'r113_diagnostics.py', 'r113_witness.py', 'r113_wave.py',
    'test_r113_model.py', 'test_r113_process.py', 'test_r113_torch_inputs.py',
    'test_r113_experiment.py', 'test_r113_wave.py', 'run_v4_r113_wave.sh')
OUTPUTS = ('RESULTS.md', 'RESULTS.pdf', 'metrics.csv', 'comparison.csv', 'events.csv',
    'historical_references.csv', 'science_boundary.json', 'target_recovery.csv',
    'target_recovery_summary.csv', 'target_status.json', 'process_comparison.csv')
SMALL_LIMIT, TOTAL_LIMIT = common.SMALL_LIMIT, common.TOTAL_LIMIT
MESSAGE = 'Publish complete R113 executable-process models and joint analysis'
LOADED_SHA256 = digest(Path(__file__))


def wave_plan(root):
    return Path(root) / 'preparation/wave_plan.json'


def required(path):
    if not Path(path).is_file():
        raise NotReady('Awaiting ' + str(path))
    return read(path)


def completion_evidence(root, *, plan_path=None):
    """Read wave receipts only; never wait for the independent R114 job."""
    root = Path(root).resolve()
    plan_path = Path(plan_path or wave_plan(root)).resolve()
    plan = required(plan_path)
    inputs = {str(plan_path): digest(plan_path)}
    if Path(plan.get('root', '')).resolve() != root:
        raise ValueError('Wave plan names another R113 root')
    jobs = plan.get('jobs', [])
    if len(jobs) != 4 or {j.get('job_id') for j in jobs} != {*ARMS, 'R114_labels'}:
        raise ValueError('Expected three R113 jobs and independent R114_labels')
    dispatch = root / 'preparation/wave_dispatch'
    controller = required(dispatch / 'controller.json')
    projection = {}
    for arm in ARMS:
        job = controller.get('jobs', {}).get(arm, {})
        if job.get('state') == 'failed':
            raise RuntimeError('R113 '+arm+' failed; no restart: '+str(job.get('reason', job)))
        if job.get('state') != 'complete':
            raise NotReady('R113 '+arm+' state='+str(job.get('state', 'pending')))
        attempts = job.get('attempts')
        if type(attempts) is not int or attempts < 1 or job.get('exit_status') != 0:
            raise ValueError('Completed R113 controller job must have rc0 and an exact attempt')
        directory = dispatch / arm / f'attempt_{attempts:04d}'
        if common.inside(job.get('attempt_dir', ''), root) != directory:
            raise ValueError('R113 completion attempt path differs from the controller')
        path = directory / 'worker.json'
        row = required(path)
        if row.get('state') == 'failed' or row.get('exit_status') not in (None, 0):
            raise RuntimeError('R113 ' + arm + ' failed; no restart: ' + str(row.get('error', row)))
        if row.get('state') != 'complete':
            raise NotReady('R113 ' + arm + ' state=' + str(row.get('state')))
        stages = row.get('stages', [])
        expected = ['train', 'finalize'] if arm == ARMS[0] else ['pretrain', 'train', 'finalize']
        names = [s.get('name') for s in stages]
        if (row.get('exit_status') != 0 or names != ['capacity', *expected]
                or any(s.get('state') != 'complete' or s.get('exit_status') != 0 for s in stages)
                or row.get('job') != arm or row.get('slot') != job.get('slot')
                or row.get('gpu_uuid') != job.get('gpu_uuid')):
            raise ValueError('R113 requires independent rc0 train/finalize pipelines: ' + arm)
        remember(inputs, path)
        projection[arm] = job
    # controller.json remains mutable while R114 runs. Freeze only R113 jobs.
    return inputs, projection


def completed(root, *, plan_path=None):
    return completion_evidence(root, plan_path=plan_path)[0]


def receipt_identity(value, model, *, label):
    expected = dict(experiment='R113', arm=model['config']['arm'], complete=True,
        **{k: model['lock'][k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')})
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError(label + ' must bind to the unique R113 checkpoint')


def saved_lib(model, dataset, inputs):
    """The R112 LIB helper is version-specific; keep this binding R113-only."""
    run, lock = model['run'], model['lock']
    problem = {'TSPLIB': 'TSP', 'CVRPLIB': 'CVRP'}[dataset]
    result = read(remember(inputs, run / 'lib_evaluation' / (dataset + '_results.json')))
    expected = dict(experiment='R113', arm=model['config']['arm'], dataset=dataset,
        model_state_sha256=model['test']['identity']['model_state_sha256'],
        **{k: lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')})
    if any(result['protocol'].get(k) != v for k, v in expected.items()):
        raise ValueError('LIB differs from the same whole-model R113 checkpoint')
    path = remember(inputs, run / 'lib_evaluation/predictions' / (dataset + '.npz'))
    with np.load(path, allow_pickle=False) as archive:
        if (str(archive['checkpoint_sha256'].item()) != lock['checkpoint_sha256']
                or int(archive['checkpoint_update']) != lock['selected_update']):
            raise ValueError('LIB prediction checkpoint binding differs')
        return problem, common.load_predictions(archive, '', model['config'], problem,
                                               result['train_sbs_reference']['n'])


def postlock(model, inputs):
    """Verify saved process evidence without importing its torch-based producer."""
    run = model['run']
    for name in ('report_complete.json', 'lib_stage_complete.json', 'inference_timing.json'):
        receipt_identity(read(remember(inputs, run / name)), model, label=name)
    frozen = read(remember(inputs, run.parent / 'process_diagnostic_inputs.json'))
    if (frozen.get('experiment') != 'R113' or frozen.get('complete') is not True
            or frozen.get('main_protocol_lock_sha256') != model['config']['main_protocol_lock']['sha256']
            or frozen.get('subset_sha256') != json_hash({k: v for k, v in frozen.items() if k != 'subset_sha256'})
            or set(frozen.get('sources', {})) != set(PROBLEMS)):
        raise ValueError('Invalid common prelaunch process-diagnostic subset')
    receipt = read(remember(inputs, run / 'process_diagnostics_complete.json'))
    receipt_identity(receipt, model, label='process diagnostics')
    if (receipt.get('subset_sha256') != frozen['subset_sha256']
            or receipt.get('diagnostic_source_sha256') != digest(V4 / 'r113_diagnostics.py')):
        raise ValueError('Process diagnostic source/subset differs')
    for suffix in ('json', 'csv', 'pdf'):
        path = remember(inputs, run / 'process_diagnostics' / ('process_diagnostics.' + suffix))
        if receipt.get('files', {}).get(suffix, {}).get('sha256') != digest(path):
            raise ValueError('Process diagnostic file differs: ' + str(path))
    value = read(run / 'process_diagnostics/process_diagnostics.json')
    receipt_identity(value, model, label='process diagnostic data')
    if (value.get('inputs') != frozen or value.get('model_reselection') is not False
            or value.get('policy') != frozen.get('policy') or value.get('version') != receipt.get('version')
            or value.get('teacher_in_model_input') is not False
            or value.get('teacher_used_for_training') is not False
            or value.get('cost_or_winner_arrays_opened') is not False
            or value.get('native_solver_calls') != 0
            or value.get('model_state_sha256') != model['test']['identity']['model_state_sha256']):
        raise ValueError('Post-lock process teacher may not influence model selection/deployment')
    rows = [dict(row, model='R113/' + model['config']['arm'],
        checkpoint_sha256=model['lock']['checkpoint_sha256'], subset_sha256=frozen['subset_sha256'])
        for row in value['metrics'] if row['field'] == 'ALL']
    if ({(r['problem'], r['component']) for r in rows}
            != {(p, c) for p in PROBLEMS for c in ('scalar', 'binary', 'pointer', 'compatible', 'reached', 'frontier')}
            or len(rows) != 18 * 6):
        raise ValueError('Process diagnostics must cover every task/component')
    return rows


def frozen_sources(root, inputs, plan_path=None):
    """Use launch-owned freeze metadata, not a new finish-time source freeze."""
    path = Path(plan_path or wave_plan(root))
    plan = read(remember(inputs, path))
    ready = read(remember(inputs, Path(root) / 'preparation/wave_dispatch/ready.json'))
    expected_plan = dict(plan, plan_path=str(path.resolve()))
    sources = ready.get('source_sha256', {})
    if (ready.get('main_approved') is not True or ready.get('runner_tests_passed') is not True
            or ready.get('plan') != expected_plan or ready.get('source_set_sha256') != json_hash(sources)
            or ready.get('protocol_sha256') != digest(Path(root) / 'protocol_lock.json')):
        raise ValueError('Require the main-approved, hash-bound Jason wave source freeze')
    if not sources:
        raise ValueError('Wave ready receipt lacks launch-frozen source_sha256')
    necessary = {str(V4 / n) for n in CODE_NAMES if n.endswith('.py')
                 and not n.startswith('test_') and n != 'r113_finish.py'}
    necessary |= {str(V4 / 'run_v4_r113_wave.sh'), str(Path(root) / 'protocol_lock.json'), str(path.resolve())}
    config = read(Path(root) / ARMS[0] / 'protocol.json')
    necessary |= {str((V4 / n).resolve()) for n in config.get('source_sha256', {})}
    resolved = {}
    for name in necessary:
        if name not in sources:
            raise ValueError('Wave freeze must bind final R113 runtime: '+name)
        expected = sources[name]
        path = Path(name)
        if not path.is_absolute():
            path = V4 / path
        path = path.resolve()
        if not path.is_relative_to(GIT_ROOT):
            raise ValueError('Launch source escapes repository')
        if digest(remember(inputs, path)) != expected:
            raise ValueError('Launch-frozen source changed: ' + str(path))
        resolved[str(path)] = expected
    return resolved


def science_boundary(root, models, inputs, *, plan_path=None):
    """Same fresh model, not identical post-pretraining B/C parameters."""
    main_path = Path(root) / 'protocol_lock.json'
    main = read(remember(inputs, main_path))
    if (main.get('experiment') != 'R113' or main.get('locked') is not True
            or main.get('arms') != list(ARMS) or main.get('seed') != 2
            or main.get('logical_global_batch') != 512 or main.get('max_successful_updates') != 57600
            or main.get('pretraining', {}).get('max_successful_updates') != 8640
            or main['pretraining'].get('logical_global_batch') != 512
            or main['pretraining'].get('teacher_forcing') is not False
            or main['pretraining'].get('processor_only') is not True):
        raise ValueError('R113 requires frozen seed2 batch512/57600 and TRAIN process8640')
    sources = frozen_sources(root, inputs, plan_path)
    common_fields = ('seed', 'global_batch_size', 'logical_global_batch', 'task_order_sha256',
        'max_successful_updates', 'lr', 'optimizer', 'weight_decay', 'loss', 'dropout', 'grad_clip',
        'microbatch_caps', 'eval_batch_size', 'checkpoint_rule', 'inference_rule', 'model_spec',
        'main_protocol_lock', 'source_sha256', 'initialization', 'fresh_state_sha256',
        'fresh_trainable_sha256', 'fresh_nonprocessor_sha256', 'initial_parameter_layout_sha256',
        'trainable_parameter_count', 'deployment_mode', 'original_fresh_reference_sha256')
    first = models[ARMS[0]]['config']
    receipts, pretraining = {}, {}
    identity_keys = ('fresh_state_sha256', 'fresh_trainable_sha256', 'fresh_nonprocessor_sha256',
        'initial_processor_sha256', 'parameter_layout_sha256', 'fresh_reference_sha256',
        'trainable_parameter_count')
    for arm, model in models.items():
        config = model['config']
        common.same_data(first, config)
        if any(k not in first or config.get(k) != first[k] for k in common_fields):
            raise ValueError('R113 joint training/init protocol differs beyond processor pretraining')
        if (config['global_batch_size'] != 512 or config['logical_global_batch'] != 512
                or config['max_successful_updates'] != 57600 or config['seed'] != 2
                or config['deployment_mode'] != 'transition'
                or config['model_spec'].get('deployment') != 'self_initialized_predicted_feedback'
                or config['current_query_labels_in_forward'] is not False
                or any(config.get(k) != main.get(k) for k in ('checkpoint_rule', 'inference_rule', 'microbatch_caps', 'eval_batch_size'))
                or config['main_protocol_lock']['record'] != main
                or config['main_protocol_lock']['sha256'] != digest(main_path)):
            raise ValueError('Final R113 model must always use input-only self-predicted feedback')
        for name, expected in config['source_sha256'].items():
            if sources.get(str((V4 / name).resolve())) != expected:
                raise ValueError('Runtime protocol not bound by wave source freeze: ' + name)
        initial = read(remember(inputs, model['run'] / 'initialization.json'))
        receipts[arm] = initial
        if (any(k not in initial for k in identity_keys)
                or initial.get('fresh_initialization') is not True
                or initial.get('historical_checkpoint_loaded') is not False
                or initial['trainable_sha256'] != config['initial_trainable_sha256']
                or initial['parameter_layout_sha256'] != config['initial_parameter_layout_sha256']
                or any(initial[k] != config[k] for k in ('fresh_state_sha256', 'fresh_trainable_sha256',
                    'fresh_nonprocessor_sha256', 'trainable_parameter_count'))):
            raise ValueError('Initial full-model receipt differs from its frozen protocol')
        if arm == ARMS[0]:
            if (initial.get('processor_pretraining_applied') is not False
                    or config.get('processor_pretraining') is not None
                    or initial.get('processor_pretraining') is not None
                    or initial['trainable_sha256'] != initial['fresh_trainable_sha256']):
                raise ValueError('Scratch must retain its entire fresh model')
            continue
        mode = {'B_snapshot': 'snapshot', 'C_transition': 'transition'}[arm]
        path = Path(main['pretrain_root']) / arm
        if path.resolve() != (Path(root) / 'pretrain' / arm).resolve():
            raise ValueError('R113 publication expects its own root/pretrain exports')
        pre = read(remember(inputs, path / 'protocol.json'))
        done = read(remember(inputs, path / 'completion.json'))
        applied = initial.get('processor_pretraining')
        if (not isinstance(applied, dict) or applied != config.get('processor_pretraining')
                or initial.get('processor_pretraining_applied') is not True
                or pre.get('experiment') != 'R113_pretrain' or pre.get('arm') != arm or pre.get('seed') != 2
                or pre.get('mode') != mode or pre.get('logical_global_batch') != 512
                or pre.get('max_successful_updates') != 8640
                or pre.get('main_protocol_lock') != config['main_protocol_lock']
                or pre.get('initial_full_state_sha256') != initial['fresh_state_sha256']
                or pre.get('initial_processor_sha256') != initial['initial_processor_sha256']
                or pre.get('teacher_forcing') is not False or pre.get('processor_only') is not True
                or done.get('complete') is not True or done.get('experiment') != 'R113_pretrain'
                or done.get('arm') != arm or done.get('successful_updates') != 8640
                or done.get('state_hash') != json_hash(pre)
                or done.get('dataset') != pre.get('dataset') or applied.get('dataset') != pre.get('dataset')
                or applied.get('mode') != mode or applied.get('successful_updates') != 8640
                or done.get('processor_sha256') != applied.get('processor_sha256')
                or done.get('processor_file_sha256') != applied.get('file_sha256')
                or done.get('source_sha256') != pre.get('source_sha256')
                or applied.get('source_sha256') != pre.get('source_sha256')
                or applied.get('state_hash') != done.get('state_hash')
                or pre.get('dataset', {}).get('manifest_sha256') != main.get('process_manifest_sha256')
                or pre['dataset'].get('split') != 'train'
                or pre['dataset'].get('rows') != 18 * 81000
                or done.get('updates_by_problem') != dict.fromkeys(PROBLEMS, 480)):
            raise ValueError('Own new TRAIN-only processor pretraining identity differs: ' + arm)
        pretraining[arm] = pre
        if (any(pre.get(k) != v for k, v in main['pretraining'].items())
                or pre.get('microbatch_caps') != main.get('pretrain_microbatch_caps')
                or pre.get('query_labels_in_forward') is not False or pre.get('native_solver_calls') != 0
                or done.get('initial_full_state_sha256') != initial['fresh_state_sha256']
                or done.get('initial_processor_sha256') != initial['initial_processor_sha256']
                or any(sources.get(str(V4 / n)) != h for n, h in pre['source_sha256'].items())):
            raise ValueError('B/C frozen TRAIN-only process source/budget differs')
    for arm in ARMS[1:]:
        if any(receipts[arm][k] != receipts[ARMS[0]][k] for k in identity_keys):
            raise ValueError('All three arms require identical complete fresh tensors/layouts')
    for key in ('dataset', 'initial_full_state_sha256', 'initial_processor_sha256',
                'processor_parameter_schema', 'task_order_sha256', 'source_sha256'):
        if key not in pretraining[ARMS[1]] or pretraining[ARMS[1]][key] != pretraining[ARMS[2]].get(key):
            raise ValueError('B/C TRAIN process pretraining identity differs: ' + key)
    process_root = Path(root) / 'process_train_v1'
    manifest = read(remember(inputs, process_root / 'manifest.json'))
    completion = read(remember(inputs, process_root / 'completion_receipt.json'))
    source = read(remember(inputs, process_root / 'source_manifest.json'))
    schema = remember(inputs, process_root / 'schema.json')
    if (manifest.get('status') != 'complete' or manifest.get('split') != 'train' or manifest.get('seed') != 2
            or digest(process_root / 'manifest.json') != main['process_manifest_sha256']
            or completion.get('status') != 'complete'
            or completion.get('manifest_sha256') != main['process_manifest_sha256']
            or completion.get('per_problem_rows') != dict.fromkeys(PROBLEMS, 81000)
            or completion.get('source_manifest_sha256') != digest(process_root / 'source_manifest.json')
            or manifest.get('source_manifest_sha256') != completion['source_manifest_sha256']
            or source.get('split') != 'train' or source.get('seed') != 2
            or completion.get('schema_sha256') != digest(schema)):
        raise ValueError('Complete process schema/source/manifest metadata differs')
    dataset = pretraining[ARMS[1]]['dataset']
    if (dataset.get('source_manifest_sha256') != completion['source_manifest_sha256']
            or dataset.get('tensor_schema_sha256') != digest(schema)):
        raise ValueError('B/C processor dataset differs from completed process metadata')
    return dict(initialization={a: {k: receipts[a][k] for k in identity_keys} for a in ARMS},
        models={a: dict(selected_update=m['lock']['selected_update'], checkpoint_sha256=m['lock']['checkpoint_sha256'],
            data_version=Path(m['config']['data_root']).name) for a, m in models.items()},
        source_sha256=sources, common_training_fields=list(common_fields), logical_batch=512,
        process_updates=8640, selector_updates=57600,
        mechanism='same fresh full model; only own TRAIN-pretrained processor differs; final mode always predicted feedback',
        order_boundary='same seeded task/shuffle protocol; no claim of independently archived selector per-update row IDs',
        statistical_boundary='single seed2; descriptive evidence, no multi-seed significance or automatic promotion',
        historical_boundary='whole saved exact-data references; not fresh parameter/budget matched controls',
        test_boundary='development-exposed TEST; no TEST/LIB checkpoint selection, tuning or per-task assembly',
        metric_boundary='ALL equal-problem; raw cost descriptive; ratios of means are not per-instance actual regret',
        W_definition='100*mean(current_cost < baseline_cost)',
        G_definition='100*mean((baseline_cost-current_cost)/baseline_cost)',
        process_boundary='post-lock fixed input-only VAL subset; teacher evaluation only, never query/model selection')


def target_groups(path, config, inputs):
    path = Path(path)
    value = read(remember(inputs, path))
    identity = value.get('identity', {})
    targets = [(r['problem'], r['method']) for r in value.get('targets', [])]
    if (len(targets) != 36 or len(set(targets)) != 36
            or Path(identity['data_root']).resolve() != Path(config['data_root']).resolve()
            or identity.get('threshold_chosen_pct') != .1
            or 'VAL' not in identity.get('rule', '')
            or any(p not in PROBLEMS or s not in config['candidate_pools'][p] for p, s in targets)):
        raise ValueError('Fixed historical VAL target36 incompatible; never reselect targets')
    return targets, dict(status='complete', path=str(path), sha256=digest(path), identity=identity, count=36)


def target_recovery(old, new, targets, comparison, split):
    rows = []
    by_problem = {p: [s for problem, s in targets if problem == p] for p in PROBLEMS}
    for p in PROBLEMS:
        left, right = old[p], new[p]
        paired(left, right, comparison, split, p)
        costs, winner, before, after = right['costs'], right['winner'], left['pred'], right['pred']
        ix, n = np.arange(len(winner)), len(winner)
        delta = costs[ix, after] - costs[ix, before]
        oracle = costs.min(1)
        slots = [right['pool'].index(s) for s in by_problem[p]]
        groups = [(s, winner == right['pool'].index(s), after == right['pool'].index(s),
                   before == right['pool'].index(s), 'fixed36_method') for s in by_problem[p]]
        groups += [(name, mask, None, None, 'winner_partition') for name, mask in
                   (('FIXED36', np.isin(winner, slots)), ('OUTSIDE36', ~np.isin(winner, slots)))]
        for name, truth, chosen, was, kind in groups:
            if kind == 'winner_partition':
                rescued = truth & (before != winner) & (after == winner)
                lost = truth & (before == winner) & (after != winner)
                incoming = truth & (after != winner) & (after != before)
                false = truth & (after != winner)
                selection = truth
            else:
                rescued, lost = truth & ~was & chosen, truth & was & ~chosen
                incoming, false, selection = ~truth & ~was & chosen, ~truth & chosen, chosen
            rows.append(dict(comparison=comparison, split=split, problem=p, solver=name, scope=kind, n=n,
                winner_count=int(truth.sum()), selected_count=int(selection.sum()),
                corrected=int(rescued.sum()), harmed=int(lost.sum()), net_correct=int(rescued.sum()-lost.sum()),
                false_pick_count=int(false.sum()), new_false_pick_count=int(incoming.sum()),
                new_false_pick_cost_delta_contribution=float(delta[incoming].sum()/n),
                new_false_pick_regret_contribution_pp=100*float(((costs[ix, after]-oracle)/oracle)[incoming].sum()/n),
                mean_cost_delta_contribution=float(delta[truth].sum()/n),
                actual_regret_delta_contribution_pp=100*float((delta[truth]/oracle[truth]).sum()/n),
                aggregation='problem denominator; winner partitions sum to full problem; method false picks overlap partitions'))
    totals = []
    for scope in ('FIXED36', 'OUTSIDE36'):
        group = [r for r in rows if r['scope'] == 'winner_partition' and r['solver'] == scope]
        totals.append(macro(group, ('new_false_pick_cost_delta_contribution', 'new_false_pick_regret_contribution_pp',
            'mean_cost_delta_contribution', 'actual_regret_delta_contribution_pp'),
            ('winner_count', 'selected_count', 'corrected', 'harmed', 'net_correct', 'false_pick_count', 'new_false_pick_count')))
    return rows, totals


def add_metrics(model, data, split, label, metrics):
    rows = []
    for problem in PROBLEMS:
        row = metric_row(data[problem], model['config']['baselines'][problem], model=label,
            split=split, problem=problem, checkpoint=model['lock']['checkpoint_sha256'])
        row['data_version'] = Path(model['config']['data_root']).name
        saved = (model['test'] if split == 'TEST' else model['final']['val'])['per_problem'][problem]
        for key in METRICS:
            expected = saved['top1']*100 if key == 'top1_pct' else saved[key]
            if not math.isclose(row[key], expected, rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError('Saved metric differs from locked predictions: ' + key)
        rows.append(row)
    metrics.extend([*rows, macro(rows, METRICS, ('native_correct',))])


def compare(old, new, label, split, comparisons, events):
    rows = []
    for problem in PROBLEMS:
        row, details = paired(old[problem], new[problem], label, split, problem)
        rows.append(row)
        events.extend(details)
    comparisons.extend([*rows, macro(rows, DELTAS, COUNTS)])


def render(root, metrics, comparisons, references, boundary, target_status, targets, process):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
            + ['| '+' | '.join(map(str, row))+' |' for row in rows])

    mh = ['Model', 'Split', 'Problem', 'Top1 %', 'mean cost', 'SBS', 'Oracle', 'vs SBS %', 'vs Oracle %', 'Regret %']
    ph = ['Comparison', 'Split', 'Problem', 'Rescued', 'Harmed', 'Wrong cheaper', 'Wrong dearer', 'dCost', 'Wrong dCost', 'dRegret pp', 'W %', 'G %']
    cells = lambda r: [r['model'], r['split'], r['problem'], *[f'{r[k]:.6f}' for k in METRICS]]
    changes = lambda r: [r['comparison'], r['split'], r['problem'], r['corrected'], r['harmed'],
        r['both_wrong_cheaper'], r['both_wrong_dearer'], *[f'{r[k]:.6f}' for k in
            ('mean_cost_delta', 'both_wrong_cost_delta', 'actual_regret_delta_pp', 'W_pct', 'G_pct')]]
    overall = [r for r in metrics if r['problem'] == 'ALL']
    pair_all = [r for r in comparisons if r['problem'] == 'ALL']
    lines = ['# R113 Joint Closeout', '', '## What Completed?',
        'All three independent rc0 train/finalize pipelines; full18 VAL/TEST, native LIB, PDFs and post-lock process diagnostics.',
        'Saved predictions only. One unique whole-model VAL-selected checkpoint per arm; no model execution or TEST tuning.',
        table(['Arm', 'Selected update', 'Checkpoint SHA256'], [[a, boundary['models'][a]['selected_update'],
            boundary['models'][a]['checkpoint_sha256']] for a in ARMS]),
        'Data: '+boundary['models'][ARMS[0]]['data_version'], '', '## Results', table(mh, [cells(r) for r in overall]),
        'Native/FP64 ind (not FP32 argmin). SBS fitted on frozen full TRAIN; Oracle is the legal-pool per-instance minimum.',
        boundary['metric_boundary']+'.', '', '## Paired Changes', table(ph, [changes(r) for r in pair_all]),
        'Rescued=baseline wrong/current correct; harmed=the reverse. dCost/dRegret=current-baseline.',
        'W='+boundary['W_definition']+'; G='+boundary['G_definition']+'. Positive G is improvement.',
        'Both-wrong net cost/regret and gross gain/harm are separate comparison.csv columns, normalized by all problem rows.',
        '', '## Historical References', table(['Model', 'Status', 'Reason'],
            [[r['model'], r['status'], r.get('reason', '')] for r in references]),
        'Only complete exact-data whole historical models; no available-task subsets or best-of-version stitching.',
        '', '## Fixed Historical Target36', json.dumps(target_status, sort_keys=True),
        'Target membership is frozen from historical VAL, not newly chosen using TEST or R113 predictions.',
        'target_recovery.csv: per-method rescues/harms/new false picks; winner-partition FIXED36 + OUTSIDE36 exactly cover every row.',
        'New false means a changed prediction to a non-winner; winner partitions classify these by true-winner membership, not chosen-method membership.',
        '', '## Scientific Boundary', boundary['mechanism']+'.', boundary['statistical_boundary']+'.',
        boundary['historical_boundary']+'.', boundary['test_boundary']+'.', boundary['order_boundary']+'.',
        boundary['process_boundary']+'.', '', '## Full18 And LIB Tables']
    for arm in ARMS:
        lines += ['', '### '+arm, table(mh, [cells(r) for r in metrics if r['model'] == 'R113/'+arm])]
    lines += ['', 'Full precision: metrics.csv / comparison.csv / target_recovery*.csv / process_comparison.csv.',
        'Per-arm method-distribution and NSS four-dataset tables remain in full_model_statistics/.', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    temporary = root / '.RESULTS.finish.pdf'
    try:
        with PdfPages(temporary) as pdf:
            def page(title, headers, rows, note):
                for start in range(0, max(1, len(rows)), 20):
                    fig, ax = plt.subplots(figsize=(17, 9))
                    try:
                        ax.axis('off')
                        fig.suptitle(title, fontsize=14)
                        fig.text(.025, .90, note[:235], fontsize=8)
                        tab = ax.table(cellText=rows[start:start+20] or [['MISSING']+['']*(len(headers)-1)],
                            colLabels=headers, loc='center', cellLoc='center')
                        tab.auto_set_font_size(False)
                        tab.set_fontsize(6.5)
                        tab.scale(1, 1.6)
                        fig.text(.025, .035, 'R113 | native/FP64 ind | frozen TRAIN SBS | equal-problem ALL | unique VAL checkpoint; no TEST/LIB tuning', fontsize=8)
                        pdf.savefig(fig)
                    finally:
                        plt.close(fig)
            page('R113: Locked Whole-Model Results', mh, [cells(r) for r in overall], boundary['statistical_boundary'])
            page('R113: Rescues, Harms And Net Cost', ph, [changes(r) for r in pair_all],
                'W=cost win rate; G=mean relative cost improvement; both-wrong contributions use all problem rows.')
            for arm in ARMS:
                for split in ('VAL', 'TEST', 'LIB'):
                    rows = [r for r in metrics if r['model'] == 'R113/'+arm and
                        (r['split'] in ('TSPLIB', 'CVRPLIB') if split == 'LIB' else r['split'] == split)]
                    page(arm+' / '+split, mh, [cells(r) for r in rows], boundary['metric_boundary'])
            for label in dict.fromkeys(r['comparison'] for r in comparisons):
                for split in ('VAL', 'TEST', 'LIB'):
                    rows = [r for r in comparisons if r['comparison'] == label and
                        (r['split'] in ('TSPLIB', 'CVRPLIB') if split == 'LIB' else r['split'] == split)]
                    if rows:
                        page(label+' / '+split, ph, [changes(r) for r in rows], boundary['test_boundary'])
            page('Historical References And Fixed36 Status', ['Item', 'Status', 'Reason'],
                [[r['model'], r['status'], r.get('reason', '')[:150]] for r in references]
                + [['Fixed36', target_status['status'], target_status.get('reason', '')[:150]]], boundary['historical_boundary'])
            page('Fixed36 Versus Outside: TEST Winner Partition',
                ['Comparison', 'Scope', 'Winners', 'Rescued', 'Harmed', 'New false', 'dCost contribution', 'dRegret pp'],
                [[r['comparison'], r['solver'], r['winner_count'], r['corrected'], r['harmed'], r['new_false_pick_count'],
                  f"{r['mean_cost_delta_contribution']:.7f}", f"{r['actual_regret_delta_contribution_pp']:.7f}"]
                 for r in targets if r['split'] == 'TEST'],
                'Winner partitions sum to the full18 cost/regret delta; per-method new false picks are reported separately in CSV.')
            for component in ('scalar', 'binary', 'pointer', 'compatible', 'reached', 'frontier'):
                page('Post-Lock VAL Process / '+component,
                    ['Arm', 'Problem', 'Valid', 'NMSE', 'Accuracy', 'Precision', 'Recall', 'All-zero accuracy'],
                    [[r['model'], r['problem'], r.get('valid_count'), *[str(r.get(k, 'N/A'))[:12] for k in
                        ('normalized_mse', 'accuracy', 'precision', 'recall', 'all_zero_accuracy')]]
                     for r in process if r['component'] == component], boundary['process_boundary'])
            page('Source, Initialization And Protocol Boundary', ['Item', 'Evidence'],
                [['Mechanism', boundary['mechanism']], ['Logical batch / updates', '512 /57600; B/C TRAIN process8640'],
                 ['Fresh full state', boundary['initialization'][ARMS[0]]['fresh_state_sha256']],
                 *[[a+' checkpoint', boundary['models'][a]['checkpoint_sha256']] for a in ARMS],
                 ['Frozen runtime files', len(boundary['source_sha256'])],
                 ['TEST', boundary['test_boundary']], ['Statistics', boundary['statistical_boundary']]],
                'Fresh identity, source and processor provenance checks only; no rereading raw process shards or model deserialization.')
        temporary.replace(root / 'RESULTS.pdf')
    finally:
        temporary.unlink(missing_ok=True)


def analyze(root, references=None, *, plan_path=None, target_path=TARGET_LIST):
    if digest(Path(__file__)) != LOADED_SHA256:
        raise ValueError('Loaded R113 finisher source changed; restart the saved-only watcher explicitly')
    root = Path(root).resolve()
    inputs, completion = completion_evidence(root, plan_path=plan_path)
    remember(inputs, Path(common.__file__))
    with (root / 'finish.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        models = {a: load_run(root / a, inputs, current=False, expected=('R113', a)) for a in ARMS}
        boundary = science_boundary(root, models, inputs, plan_path=plan_path)
        metrics, comparisons, events, historical, recoveries, recovery_summary, process = [], [], [], [], [], [], []
        try:
            targets, target_status = target_groups(target_path, models[ARMS[0]]['config'], inputs)
        except (OSError, ValueError, KeyError) as error:
            targets, target_status = [], dict(status='missing_or_incompatible', path=str(target_path), reason=str(error))
        for arm in ARMS:
            process.extend(postlock(models[arm], inputs))
        all_data = {}
        for split in ('VAL', 'TEST'):
            data = {a: saved_split(models[a], split, inputs) for a in ARMS}
            all_data[split] = data
            for arm in ARMS:
                add_metrics(models[arm], data[arm], split, 'R113/'+arm, metrics)
            for left, right in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
                label = right+' vs '+left
                compare(data[left], data[right], label, split, comparisons, events)
                if targets:
                    rows, totals = target_recovery(data[left], data[right], targets, label, split)
                    recoveries.extend(rows)
                    recovery_summary.extend(totals)
        for dataset in ('TSPLIB', 'CVRPLIB'):
            data = {a: saved_lib(models[a], dataset, inputs) for a in ARMS}
            for arm in ARMS:
                problem, item = data[arm]
                row = metric_row(item, models[arm]['config']['baselines'][problem], model='R113/'+arm,
                    split=dataset, problem=problem, checkpoint=models[arm]['lock']['checkpoint_sha256'])
                metrics.append(dict(row, data_version='original native '+dataset))
            for left, right in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
                row, detail = paired(data[left][1], data[right][1], right+' vs '+left, dataset, data[left][0])
                comparisons.append(row)
                events.extend(detail)
        for name, expected in (('R94B', ('R94', 'B')), ('R89B', ('R89', 'B_muon_hybrid'))):
            path = Path((REFERENCES if references is None else references).get(name, REFERENCES[name]))
            reference_inputs = {}
            try:
                old = load_run(path, reference_inputs, current=False, expected=expected)
                common.same_data(models[ARMS[0]]['config'], old['config'])
                saved = {s: saved_split(old, s, reference_inputs) for s in ('VAL', 'TEST')}
                for split in saved:
                    for arm in ARMS:
                        for p in PROBLEMS:
                            paired(saved[split][p], all_data[split][arm][p], name, split, p)
            except (OSError, ValueError, KeyError, AssertionError) as error:
                historical.append(dict(model=name, status='missing_or_incompatible', run=str(path), reason=str(error),
                    whole_model_only=True, stitched=False))
            else:
                for split in saved:
                    add_metrics(old, saved[split], split, name, metrics)
                    for arm in ARMS:
                        label = arm+' vs '+name
                        compare(saved[split], all_data[split][arm], label, split, comparisons, events)
                        if targets and name == 'R89B':
                            rows, totals = target_recovery(saved[split], all_data[split][arm], targets, label, split)
                            recoveries.extend(rows)
                            recovery_summary.extend(totals)
                historical.append(dict(model=name, status='complete', run=str(path), reason='',
                    checkpoint_sha256=old['lock']['checkpoint_sha256'], whole_model_only=True, stitched=False))
            inputs.update({p: h for p, h in reference_inputs.items() if Path(p).is_file()})
        write_csv(root / 'metrics.csv', metrics)
        write_csv(root / 'comparison.csv', comparisons)
        write_csv(root / 'events.csv', events, common.EVENT_FIELDS)
        write_csv(root / 'historical_references.csv', historical)
        write_csv(root / 'target_recovery.csv', recoveries, None if recoveries else ['comparison', 'split', 'problem', 'solver', 'status'])
        write_csv(root / 'target_recovery_summary.csv', recovery_summary, None if recovery_summary else ['comparison', 'split', 'solver', 'status'])
        write_csv(root / 'process_comparison.csv', process)
        write(root / 'science_boundary.json', boundary)
        write(root / 'target_status.json', target_status)
        render(root, metrics, comparisons, historical, boundary, target_status, recovery_summary, process)
        for path, expected in inputs.items():
            if digest(path) != expected:
                raise ValueError('Analysis input changed during collection: ' + path)
        value = dict(experiment='R113', complete=True, analysis_rc=0, completed_utc=utc(),
            model_reselection=False, model_inference=False, inputs=inputs,
            outputs={n: digest(root / n) for n in OUTPUTS}, checkpoints={a: models[a]['lock'] for a in ARMS},
            generator_sha256=LOADED_SHA256, historical=historical, targets=target_status,
            plan_path=str(Path(plan_path or wave_plan(root)).resolve()),
            completion=completion,
            event_count=len(events), metric_rows=len(metrics), comparison_rows=len(comparisons))
        write(root / 'finish_complete.json', value)
        return value


def verify_delivery(root):
    root = Path(root).resolve()
    value = read(root / 'finish_complete.json')
    _, completion = completion_evidence(root, plan_path=value.get('plan_path'))
    if (value.get('experiment') != 'R113' or value.get('complete') is not True or value.get('analysis_rc') != 0
            or value.get('model_reselection') is not False or value.get('model_inference') is not False
            or value.get('completion') != completion or set(value.get('checkpoints', {})) != set(ARMS)
            or set(value.get('outputs', {})) != set(OUTPUTS)
            or value.get('generator_sha256') != digest(Path(__file__))):
        raise ValueError('Run current R113 analyze successfully before publication')
    for path, expected in value['inputs'].items():
        if digest(path) != expected:
            raise ValueError('Frozen analysis input changed: ' + path)
    for name, expected in value['outputs'].items():
        if digest(root / name) != expected:
            raise ValueError('Closeout output changed: ' + name)
    pdf = (root / 'RESULTS.pdf').read_bytes()
    if not pdf.startswith(b'%PDF-') or b'%%EOF' not in pdf[-1024:]:
        raise ValueError('Joint PDF incomplete')
    return value


def publication_paths(root):
    """Finite literal artifact names; no globs, weights, archives or raw NPZs."""
    root = Path(root).resolve()
    if root != (GIT_ROOT / '0selection2/code/V4/runs/R113_executable_process').resolve():
        raise ValueError('Publication owns only the named R113 root')
    paths = [root.parent.parent / name for name in CODE_NAMES]
    paths += [root / n for n in (*OUTPUTS, 'finish_complete.json', 'protocol_lock.json', 'PLAN.md',
        'process_diagnostic_inputs.json', 'preparation/wave_plan.json', 'preparation/LAUNCH.md')]
    paths += [root / 'process_train_v1' / n for n in ('schema.json', 'source_manifest.json',
        'completion_receipt.json', 'run_config.json', 'manifest.json', 'README.md')]
    for arm in ARMS:
        paths += [root / arm / n for n in ('protocol.json', 'initialization.json', 'final_test_lock.json',
            'training_complete.json', 'report_complete.json', 'lib_stage_complete.json', 'inference_timing.json',
            'process_diagnostics_complete.json')]
        paths += [root / arm / 'process_diagnostics' / ('process_diagnostics.'+suffix) for suffix in ('json', 'csv', 'pdf')]
        paths += [root / arm / 'lib_evaluation' / (dataset+'_results.json') for dataset in ('TSPLIB', 'CVRPLIB')]
        paths += [root / arm / 'full_model_statistics' / n for n in ('MODEL_STATISTICS.pdf',
            'selection_distributions.pdf', 'learning_curves.pdf', 'training_and_method_diagnostics.pdf',
            'RESULTS.md', 'protocol.json', 'metrics_by_problem.csv', 'nss_comparison.csv',
            'selection_distribution.csv', 'single_method_comparison.csv', 'inference_timing.csv')]
        if arm != ARMS[0]:
            paths += [root / 'pretrain' / arm / n for n in ('protocol.json', 'completion.json')]
    # Only the exact three R113 completion receipts named in the frozen plan.
    _, jobs = completion_evidence(root)
    for arm, job in jobs.items():
        directory = common.inside(job['attempt_dir'], root)
        paths += [directory / n for n in ('worker.json', 'request.json', 'entry.json')]
        paths.append(root / 'preparation' / ('capacity_'+arm+'.json'))
    included, excluded = [], []
    optional = {'PLAN.md', 'README.md', 'LAUNCH.md', 'single_method_comparison.csv', 'inference_timing.csv'}
    for path in dict.fromkeys(paths):
        if not path.exists() and path.name in optional:
            continue
        if (path.is_symlink() or not path.is_file() or path.resolve() != path
                or not path.is_relative_to(GIT_ROOT.resolve())
                or path.suffix in ('.pt', '.npz', '.npy')):
            raise ValueError('Publication requires an ordinary exact-allowlisted small artifact: '+str(path))
        if path.stat().st_size > SMALL_LIMIT:
            if path == root / 'events.csv':
                excluded.append(dict(path=str(path.relative_to(GIT_ROOT)), reason='large event CSV retained locally',
                    bytes=path.stat().st_size, sha256=digest(path)))
                continue
            raise ValueError('R113 small-file limit exceeded: '+str(path))
        included.append(path)
    if sum(p.stat().st_size for p in included) > TOTAL_LIMIT:
        raise ValueError('R113 bounded publication total exceeded; no staging')
    return included, excluded


def publication_preflight(root):
    value = verify_delivery(root)
    if Path(git('rev-parse', '--show-toplevel')).resolve() != GIT_ROOT.resolve():
        raise ValueError('Expected Git root /public/home/shiys')
    paths, excluded = publication_paths(root)
    names = [str(p.relative_to(GIT_ROOT)) for p in paths]
    if git('diff', '--cached', '--name-only', '-z'):
        raise ValueError('Pre-existing staged changes; index untouched')
    frozen_sources(root, {}, value['plan_path'])
    # R114 is an independent queue dependency, not a model/report dependency.
    # Do not let its unfinished source enter this publication via the launcher.
    runtime_names = set(CODE_NAMES)-{'r113_wave.py'}
    deps = dependency_files([p for p in paths if p.suffix == '.py' and p.name in runtime_names
                             and not p.name.startswith('test_') and p.parent == Path(root).parent.parent])
    dep_names = [str(p.relative_to(GIT_ROOT)) for p in deps]
    tracked = set(filter(None, git('ls-files', '-z', '--', *dep_names).split('\0')))
    missing = set(dep_names)-tracked-set(names)
    if missing:
        raise ValueError('Unpublished dependencies outside exact R113 allowlist; never auto-add: '+', '.join(sorted(missing)))
    branch = git('symbolic-ref', '--quiet', '--short', 'HEAD')
    remote, merge = (git('config', '--get', 'branch.'+branch+'.'+key) for key in ('remote', 'merge'))
    if remote == '.' or not merge.startswith('refs/heads/'):
        raise ValueError('Require an existing named remote/upstream')
    parent = git('rev-parse', 'HEAD')
    rows = git('ls-remote', '--heads', remote, merge).splitlines()
    if len(rows) != 1 or rows[0].split()[0] != parent:
        raise ValueError('Current HEAD must equal upstream before the one R113 commit; reconcile manually')
    return dict(paths=names, excluded=excluded, parent=parent, remote=remote, merge=merge, branch=branch,
        sha256={n: digest(GIT_ROOT / n) for n in names}, dependency_paths=dep_names)


def publish(root, message=MESSAGE):
    """At most one commit and one push; capture current parent under the mutex."""
    root = Path(root).resolve()
    with publication_mutex() as mutex, (root / 'finish_publish.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = root / 'publish_receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt.get('state') == 'complete':
                return receipt
            raise RuntimeError('Publication already attempted; manual recovery required, no automatic retry')
        plan = publication_preflight(root)
        manifest = root / 'publish_manifest.json'
        write(manifest, dict(experiment='R113', prepared_utc=utc(), **plan))
        names = [*plan['paths'], str(manifest.relative_to(GIT_ROOT))]
        receipt = dict(experiment='R113', state='prepared', started_utc=utc(), parent=plan['parent'],
            remote=plan['remote'], merge=plan['merge'], global_mutex=str(mutex), automatic_retry=False,
            allowlist=names, commit_attempts=0, push_attempts=0)
        with receipt_path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2)
        try:
            for path, expected in plan['sha256'].items():
                if digest(GIT_ROOT / path) != expected:
                    raise ValueError('Publication input changed before staging: '+path)
            if git('rev-parse', 'HEAD') != plan['parent'] or git('diff', '--cached', '--name-only', '-z'):
                raise ValueError('HEAD/index changed before staging; preserve unrelated work')
            git('add', '-f', '--', *names)
            staged = set(filter(None, git('diff', '--cached', '--name-only', '-z').split('\0')))
            if not staged or staged-set(names):
                raise ValueError('Staged set must be a nonempty subset of exact R113 allowlist')
            blobs = {n: git('rev-parse', ':'+n) for n in names}
            receipt.update(state='commit_started', commit_attempts=1)
            write(receipt_path, receipt)
            git('commit', '-m', message, '--only', '--', *names)
            commit = git('rev-parse', 'HEAD')
            receipt.update(state='committed', commit=commit)
            write(receipt_path, receipt)
            changed = set(filter(None, git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).splitlines()))
            if (git('rev-parse', 'HEAD^') != plan['parent'] or changed-set(names)
                    or any(git('rev-parse', commit+':'+n) != blob for n, blob in blobs.items())):
                raise ValueError('Committed payload differs from exact R113 allowlist; no push')
            receipt.update(state='push_started', push_attempts=1)
            write(receipt_path, receipt)
            git('push', '--porcelain', plan['remote'], commit+':'+plan['merge'])
            receipt.update(state='complete', exit_status=0, completed_utc=utc())
            write(receipt_path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error_type=type(error).__name__, error=str(error),
                failed_utc=utc(), manual_recovery_required=True)
            write(receipt_path, receipt)
            raise


def wait(root, *, timeout_hours=168., poll_seconds=60., publish_after=False, message=MESSAGE,
         references=None, plan_path=None, target_path=TARGET_LIST):
    if any(not math.isfinite(v) or v <= 0 for v in (timeout_hours, poll_seconds)):
        raise ValueError('Wait timeout and poll interval must be finite and positive')
    deadline, previous = time.monotonic()+timeout_hours*3600, None
    while True:
        try:
            completed(root, plan_path=plan_path)
            break
        except NotReady as error:
            if time.monotonic() >= deadline:
                raise TimeoutError('R113 finish timed out; no launch/restart')
            if str(error) != previous:
                print(str(error), flush=True)
                previous = str(error)
            time.sleep(min(poll_seconds, max(0., deadline-time.monotonic())))
    value = analyze(root, references, plan_path=plan_path, target_path=target_path)
    return publish(root, message) if publish_after else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analyze', 'wait', 'publish'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--wave-plan', type=Path)
    parser.add_argument('--target-list', type=Path, default=TARGET_LIST)
    parser.add_argument('--r94b-root', type=Path, default=REFERENCES['R94B'])
    parser.add_argument('--r89b-root', type=Path, default=REFERENCES['R89B'])
    parser.add_argument('--timeout-hours', type=float, default=168.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--publish', action='store_true', help='one commit/push after complete saved-only analysis')
    parser.add_argument('--message', default=MESSAGE)
    args = parser.parse_args(argv)
    if args.publish and args.command != 'wait':
        parser.error('--publish belongs to wait; use the publish command explicitly otherwise')
    options = dict(references={'R94B': args.r94b_root, 'R89B': args.r89b_root},
        plan_path=args.wave_plan, target_path=args.target_list)
    if args.command == 'wait':
        value = wait(args.root, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds,
            publish_after=args.publish, message=args.message, **options)
    elif args.command == 'analyze':
        value = analyze(args.root, **options)
    else:
        value = publish(args.root, args.message)
    print(json.dumps({k: v for k, v in value.items() if k not in ('inputs', 'outputs', 'checkpoints', 'allowlist')}, indent=2))
    return value


if __name__ == '__main__':
    main()
