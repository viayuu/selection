"""R116 same-checkpoint PDFs, post-query likelihood curves and complete-DP p95 timing.

The historical PDF/layout callbacks are privately rebound, not copied or edited.
Reports never launch arms, refit a model, or select a checkpoint from TEST/LIB.
"""

import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import r116_experiment as runner
from . import r80_reporting as reporting_template
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r116_objective import PART_KEYS, inputs_only


_callbacks = runner.private_module(runner.V4 / 'r112_reporting.py', 'report_callbacks')
ARMS, DEFAULT_ROOT = runner.ARMS, runner.DEFAULT_ROOT
PROBLEMS, LIBS = runner.PROBLEMS, _callbacks.LIBS
common, shared = _callbacks.common, _callbacks.shared
write_csv, locked_inputs, nss_comparisons = _callbacks.write_csv, _callbacks.locked_inputs, _callbacks.nss_comparisons
REPETITIONS, WARMUPS = 10, 2
TIMING_SCOPE = 'GPU-resident fixed input batch; complete graph encoder, all subset heads and exact champion DP'


@torch.no_grad()
def _measure_model(model, batch, device):
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    clean = inputs_only(batch)
    for _ in range(WARMUPS):
        model(clean)
    sync()
    samples = []
    for _ in range(REPETITIONS):
        sync()
        started = time.perf_counter()
        output = model(clean)
        sync()
        samples.append(time.perf_counter() - started)
        if not bool(torch.isfinite(output['logits']).all()):
            raise ValueError('Nonfinite input-only timing prediction')
    count = len(clean['node_mask'])
    return dict(batch_seconds=samples, mean_seconds_per_instance=float(np.mean(samples)) / count,
                p95_seconds_per_instance=float(np.percentile(samples, 95)) / count)


@torch.no_grad()
def measure_timing(args, device):
    if device.type != 'cuda':
        raise ValueError('Formal timing requires the main-assigned 3090')
    lock, config = runner.authorize_test(args)
    run = args.root / args.arm
    saved = read_json(run / 'test_results.json')
    if (saved.get('complete') is not True or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or saved.get('state_hash') != lock['state_hash']):
        raise PermissionError('Timing must follow complete immutable TEST')
    model_hash = saved['result']['identity']['model_state_sha256']
    if (run / 'inference_timing.json').exists():
        return inference_timing(run, config, lock, model_hash)
    model, state = runner.load_checkpoint(run / 'best.pt', device)
    if parameter_hash(state['model']) != model_hash:
        raise ValueError('Timing and TEST model identities differ')
    del state
    initial = torch.load(Path(config['prepared_root']) / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    baseline, _, baseline_receipt = runner.engine.initialize_shared(initial, config['geometry_stats'], 'A_baseline', device)
    del initial
    baseline.eval()
    model.eval()
    manifest, rows = runner.ensure_test_cache(args, lock), []
    for problem in PROBLEMS:
        count = min(config['eval_batch_size'], config['microbatch_caps'][problem])
        batch = runner.cached_input_batch(config, manifest, 'test', problem, device, count=count)
        current = _measure_model(model, batch, device)
        point = _measure_model(baseline, batch, device)
        ratio = current['p95_seconds_per_instance'] / point['p95_seconds_per_instance']
        rows.append(dict(problem=problem, instances=count, standalone_model=current,
                         point_selector=point, p95_ratio=ratio, p95_ratio_le_2=ratio <= 2.))
        del batch
    if (file_hash(run / 'best.pt') != lock['checkpoint_sha256']
            or parameter_hash(model.state_dict()) != model_hash):
        raise ValueError('Input-only timing changed the selected checkpoint')
    value = dict(experiment='R116', arm=args.arm, complete=True, completed_utc=utc(),
        **{k: lock[k] for k in ('checkpoint_sha256', 'state_hash', 'selected_update')},
        model_state_sha256=model_hash, model_factory=runner.MODEL_FACTORY,
        inference_rule=runner.INFERENCE_RULE, device=str(device), device_name=torch.cuda.get_device_name(device),
        repetitions=REPETITIONS, warmup_batches=WARMUPS, eval_batch_size=config['eval_batch_size'],
        parameter_count=config['trainable_parameter_count'], scope=TIMING_SCOPE,
        sample_policy='first min(eval_batch_size, locked task microcap) frozen TEST inputs per task',
        p95_definition='numpy percentile95 of 10 synchronized full-batch forwards, divided by real batch rows',
        all_current_model_included=True, all_subsets_and_exact_dp_included=True,
        end_to_end_measured=False, input_labels_read=False, model_reselection=False,
        point_selector=dict(factory='code.V4.r84_experiment.make_model', arm='A_baseline',
            initialization='fresh never-trained R84 seed2; timing only; identical inputs and batch sizes',
            trainable_sha256=baseline_receipt['trainable_sha256'], checkpoint_used=False),
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver execution', per_problem=rows,
        max_task_p95_ratio=max(r['p95_ratio'] for r in rows), p95_ratio_le_2=all(r['p95_ratio_le_2'] for r in rows))
    write_json(run / 'inference_timing.json', value)
    return inference_timing(run, config, lock, model_hash)


def inference_timing(run, config, lock, model_hash):
    value = read_json(Path(run) / 'inference_timing.json')
    expected = dict(experiment='R116', arm=config['arm'], complete=True,
        **{k: lock[k] for k in ('checkpoint_sha256', 'state_hash', 'selected_update')},
        model_state_sha256=model_hash, model_factory=runner.MODEL_FACTORY,
        inference_rule=runner.INFERENCE_RULE, repetitions=REPETITIONS, warmup_batches=WARMUPS,
        eval_batch_size=config['eval_batch_size'], parameter_count=config['trainable_parameter_count'],
        scope=TIMING_SCOPE, all_current_model_included=True, all_subsets_and_exact_dp_included=True,
        input_labels_read=False, model_reselection=False, end_to_end_measured=False)
    rows = value.get('per_problem', [])
    if (any(value.get(k) != v for k, v in expected.items()) or not str(value.get('device')).startswith('cuda')
            or len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS)
            or value.get('point_selector', {}).get('factory') != 'code.V4.r84_experiment.make_model'
            or value.get('point_selector', {}).get('checkpoint_used') is not False):
        raise ValueError('R116 timing must include all subset DP and a fresh point baseline on identical inputs')
    for row in rows:
        count = min(config['eval_batch_size'], config['microbatch_caps'][row['problem']])
        if row['instances'] != count:
            raise ValueError('Timing must retain locked per-task batch size')
        for name in ('standalone_model', 'point_selector'):
            measured = row[name]
            samples = measured['batch_seconds']
            if (len(samples) != REPETITIONS or any(not math.isfinite(v) or v <= 0 for v in samples)
                    or not math.isclose(measured['mean_seconds_per_instance'], float(np.mean(samples)) / count, rel_tol=1e-10)
                    or not math.isclose(measured['p95_seconds_per_instance'], float(np.percentile(samples, 95)) / count, rel_tol=1e-10)):
                raise ValueError('Invalid p95 timing samples or real-row denominator')
        ratio = row['standalone_model']['p95_seconds_per_instance'] / row['point_selector']['p95_seconds_per_instance']
        if not math.isclose(row['p95_ratio'], ratio, rel_tol=1e-10) or row['p95_ratio_le_2'] != (ratio <= 2):
            raise ValueError('Invalid fixed point-selector p95 ratio')
    if (value['p95_ratio_le_2'] != all(r['p95_ratio_le_2'] for r in rows)
            or value['max_task_p95_ratio'] != max(r['p95_ratio'] for r in rows)):
        raise ValueError('Timing acceptance must cover every task')
    return value


def learning_curves(run, output, model, lock, **unused):
    import matplotlib.pyplot as plt
    history = read_json(Path(run) / 'history.json')
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(model + ' | one forward; champion CE + mean full-ranking NLL')
    for key in PART_KEYS[:-1]:
        points = [(r['successful_updates'], r['optimization'][key]) for r in history]
        if points:
            axes[0, 0].plot(*zip(*points), label=key)
    axes[0, 0].set_title('TRAIN optimization interval means')
    traces = ((axes[0, 1], 'macro_champion_nll', 1., 'Champion NLL / native CE'),
              (axes[0, 2], 'macro_ranking_nll', 1., 'Ranking NLL (mean K-1 steps)'),
              (axes[1, 0], 'macro_top1', 100., 'Native Top1 (%)'),
              (axes[1, 1], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
              (axes[1, 2], 'macro_vs_sbs_pct', 1., 'vs frozen TRAIN SBS (%)'))
    for ax, key, factor, title in traces:
        for split in ('train', 'val'):
            points = [(r['successful_updates'], r[split]['macro'][key] * factor)
                      for r in history if r.get(split) is not None]
            if points:
                ax.plot(*zip(*points), label=split)
        ax.set_title(title)
    for ax in axes.flat:
        ax.axvline(lock['selected_update'], color='#b44432', linestyle='--', linewidth=1)
        ax.set_xlabel('Successful optimizer updates')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(Path(output) / 'learning_curves.pdf')
    fig.savefig(Path(output) / 'learning_curves.png', dpi=140)
    plt.close(fig)


# Keep the established full18 paired charts/NSS/cost tables; replace only real loss traces.
shared.learning_curves = learning_curves
runner.adapt_function(common, reporting_template.diagnostic_pdf, (
    ('for name in ("loss", "ce", "cmp", "risk", "kl"):',
     'for name in ("loss", "champion_nll", "ranking_nll"):'),))
_callbacks.inference_timing = inference_timing


def lib_stage(args):
    # Same immutable factory/lock and raw LIB loader; no historical model fallback.
    runner.protocol_lock(args)
    return _callbacks.lib_stage(args)


def report(args):
    result = _callbacks.report(args)
    run = args.root / args.arm
    config = read_json(run / 'protocol.json')
    lock = read_json(run / 'final_test_lock.json')
    final = read_json(run / 'final_metrics.json')
    output = run / 'full_model_statistics'
    rows = []
    for split in ('train', 'val', 'test'):
        value = final[split]
        rows.append(dict(split=split.upper(), problem='ALL',
            **{k: value['macro']['macro_' + k] for k in PART_KEYS[:-1]}, aggregation='equal problem mean'))
        rows.extend(dict(split=split.upper(), problem=p,
            **{k: value['per_problem'][p][k] for k in PART_KEYS[:-1]}, aggregation='real instance mean') for p in PROBLEMS)
    write_csv(output / 'objective_diagnostics.csv', rows)
    timing = inference_timing(run, config, lock, final['test']['identity']['model_state_sha256'])
    write_csv(output / 'inference_timing.csv', [dict(problem=r['problem'], instances=r['instances'],
        mean_milliseconds=1000 * r['standalone_model']['mean_seconds_per_instance'],
        p95_milliseconds=1000 * r['standalone_model']['p95_seconds_per_instance'],
        point_p95_milliseconds=1000 * r['point_selector']['p95_seconds_per_instance'],
        p95_ratio=r['p95_ratio'], p95_ratio_le_2=r['p95_ratio_le_2']) for r in timing['per_problem']])
    provenance = read_json(output / 'protocol.json')
    provenance.update(objective=config['objective'], historical_comparison=config['historical_comparison'],
        all_subset_dp_timing=timing, objective_diagnostics='objective_diagnostics.csv',
        no_rdrop=True, no_old_maximin=True)
    write_json(output / 'protocol.json', provenance)
    return result


def main(argv=None):
    parser = runner.parser()
    parser.set_defaults(command='finalize')
    args = parser.parse_args(argv)
    if args.command != 'finalize':
        raise ValueError('Use runner train/test/timing; reporting finalize renders saved same-lock results')
    args.root, args.nss_root = args.root.expanduser().resolve(), args.nss_root.expanduser().resolve()
    return report(args)


if __name__ == '__main__':
    main()
