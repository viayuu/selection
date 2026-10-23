"""R105 frozen-factory LIB and saved-evidence full18 PDF/CSV reporting."""

import argparse
import copy
import fcntl
import json
import math
import os
import textwrap
from pathlib import Path

import numpy as np

from . import r80_reporting, r84_reporting, r85_reporting, r87_reporting
from . import r105_experiment as runner
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import parameter_hash


ARMS, BUDGETS, DEFAULT_ROOT = runner.ARMS, runner.BUDGETS, runner.DEFAULT_ROOT
FACTORY = 'code.V4.r105_experiment'
PROBLEMS = tuple(runner.PROBLEMS)
DEFAULT_R84, DEFAULT_NSS = r87_reporting.DEFAULT_R84, r87_reporting.DEFAULT_NSS
DEFAULT_R89 = DEFAULT_ROOT.parent / 'R89_optimizer_geometry'
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
                 'training_and_method_diagnostics.pdf', 'target_recovery_and_train.pdf')
BOUNDARY = ('R105 replaces only the four instance FFNs with four dense FF experts per layer. '
    'A learns input-independent bias-softmax mixing through zero router inputs; it is not fixed '
    'uniform after initialization. B routes from current post-attention node representations. '
    'Both start with exactly the same complete fresh tensors and uniform gates, and both compute '
    'all four experts. Each has3,425,561 total parameters; A has2,048 intentionally inactive '
    'zero-gradient router feature weights, so active functional freedom is not matched. '
    'No sparse-compute or equal-FLOPs claim. Original R88B physical TSP node0 role, methodID+34, '
    'geometry/distance, score MLP/maximin and original CE/comparison/risk/two-forward R-Drop remain. '
    'No teacher, source vectors, context labels, new targets, routing calibration, quota or auxiliary '
    'loss. R89A AdamW, R89B Muon and R84A are saved historical references, not rerun or newly '
    'parameter/FLOP-matched controls. Fixed36 rescues, losses, false picks and aligned net costs '
    'are separate from choice frequency. Native/FP64 ind and legal pools; TRAIN-fitted SBS; ALL '
    'task-equal. TEST is development-exposed; one whole-model full18 VAL checkpoint; no TEST/LIB '
    'reselection. First128 VAL/task routing is input-only post-lock descriptive evidence, not '
    'efficacy or proof against coarse shortcuts. Actual standalone GPU-resident input timing '
    'excludes dataset I/O, CPU transfer and solvers. No online W&B equality/availability is asserted.')


# Function globals belong to these private copies, never the historical modules.
common = runner.private_module(r80_reporting, 'report_common')
shared = runner.private_module(r84_reporting, 'report_shared')
previous = runner.private_module(r85_reporting, 'report_previous')
helpers = runner.private_module(r87_reporting, 'report_helpers')
shared.common = common
previous.shared, previous.common = shared, common
helpers.shared, helpers.previous, helpers.common = shared, previous, common
LIBS = shared.LIBS
for module in (common, shared, previous, helpers):
    module.DEFAULT_ROOT, module.ARMS, module.BUDGETS, module.FACTORY = DEFAULT_ROOT, ARMS, BUDGETS, FACTORY
_template_selected_train = previous.selected_train


def selected_train(run, config, lock):
    result, path = _template_selected_train(run, config, lock)
    if config['experiment'] == 'R105':
        runner.selected_capture_proof(run, result, lock, 'train')
    return result, path


previous.selected_train = selected_train


def capture_val(val, lock):
    """Use preserved capture identity only after checking the official binding."""
    binding = val.get('checkpoint_binding', {})
    expected = dict(rule=runner.VAL_BINDING_RULE, checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], selected_update=lock['selected_update'], archive_identity_unchanged=True)
    capture, identity = val.get('archive_capture_identity'), val.get('identity', {})
    if (not isinstance(capture, dict) or any(binding.get(k) != v for k, v in expected.items())
            or capture.get('experiment') != 'R105' or capture.get('arm') != lock['arm']
            or capture.get('split') != 'val'
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or binding.get('archive_capture_checkpoint_sha256') != capture.get('checkpoint_sha256')
            or identity != dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
            or capture.get('state_hash') != lock['state_hash']
            or capture.get('successful_updates') != lock['selected_update']
            or not capture.get('model_state_sha256')
            or capture['model_state_sha256'] != binding.get('model_state_sha256')):
        raise ValueError('R105 selected VAL summary lacks its exact-state official binding')
    result = copy.deepcopy(val)
    result['identity'] = copy.deepcopy(capture)
    return result


def locked_inputs(root, arm):
    run, config, lock, test, val = shared.locked_inputs(Path(root), arm, experiment='R105', arms=BUDGETS)
    model, state = runner.load_checkpoint(run / 'best.pt', 'cpu')
    if (state['args'] != config or state['successful_updates'] != lock['selected_update']
            or state.get('epoch') != lock['epoch'] or config['model_factory'] != FACTORY + '.load_checkpoint'):
        raise ValueError('R105 reporter factory differs from the unique per-arm model lock')
    model_hash = parameter_hash(model.state_dict())
    _, binding = runner.selected_capture_proof(run, val, lock, 'val', state)
    if binding != val.get('checkpoint_binding'):
        raise ValueError('R105 official VAL binding differs from the actual factory/checkpoint proof')
    val = capture_val(val, lock)
    if model_hash != test['identity']['model_state_sha256'] or model_hash != val['identity']['model_state_sha256']:
        raise ValueError('R105 VAL/TEST/frozen-factory model states differ')
    del model, state
    return run, config, lock, test, val


def same_data(left, right):
    for key in ('data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids', 'baselines', 'winner_rule'):
        if key not in left or left[key] != right.get(key):
            raise ValueError('R105 comparison requires identical ' + key)


def selected_predictions(run, config, lock, val, test):
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    name = config['experiment'] + '/' + config['arm']
    return packed, [dict(r, model=name) for r in methods], [dict(r, model=name) for r in summaries], results, paths


def references(config, args):
    result = {}
    for name, path, experiment, budgets, optimizer in (
            ('R89A', args.r89a_root, 'R89', dict(A_adamw=57600, B_muon_hybrid=57600), 'AdamW'),
            ('R89B', args.r89_root, 'R89', dict(A_adamw=57600, B_muon_hybrid=57600), 'Muon hybrid'),
            ('R84A', args.historical_root, 'R84', dict(A_baseline=57600), 'AdamW')):
        path = Path(path)
        run, old, lock, test, val = shared.locked_inputs(path.parent, path.name, experiment=experiment, arms=budgets)
        same_data(config, old)
        result[name] = dict(run=run, config=old, lock=lock, test=test, val=val,
            provenance=dict(model=name, run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
                same_data=True, retrained=False, newly_matched_control=False, optimizer=optimizer,
                parameter_matched=False, equal_flops=False, identical_full_initialization=False))
    return result


def load_targets(config, old_config, old_lock):
    targets, identity = helpers.load_targets(config, old_config, old_lock)
    if len(targets) != 36:
        raise ValueError('R105 retains frozen R84 target36 without reselection')
    return targets, dict(identity, target_count=36, sha256=file_hash(Path(identity['path'])))


def comparison_tables(old, new, targets, identity, model, baseline):
    pairs, recovery, totals = helpers.comparison_tables(old, new, targets, identity, model, baseline)
    for row in recovery:
        left, right = old[row['split']][row['problem']], new[row['split']][row['problem']]
        slot, index = right.pool.index(row['solver']), np.arange(right.n)
        truth, was, chosen = right.winner == slot, left.pred == slot, right.pred == slot
        incoming, outgoing = ~truth & ~was & chosen, ~truth & was & ~chosen
        delta = right.costs[index, right.pred] - left.costs[index, left.pred]
        row.update(net_rescues=row['rescued'] - row['lost'],
            new_false_pick_cost_delta_contribution=float(delta[incoming].sum() / right.n),
            removed_false_pick_count=int(outgoing.sum()),
            removed_false_pick_cost_delta_contribution=float(delta[outgoing].sum() / right.n),
            target_winner_cost_delta_contribution=float(delta[truth].sum() / right.n),
            full_problem_mean_cost_delta=float(delta.mean()))
    for row in totals:
        row['net_rescues'] = row['rescued'] - row['lost']
        for key in ('new_false_pick_cost_delta_contribution', 'removed_false_pick_cost_delta_contribution',
                    'target_winner_cost_delta_contribution'):
            row[key] = sum(r[key] for r in recovery if r['split'] == row['split']) / len(PROBLEMS)
        row['full18_net_mean_cost_delta'] = next(r['mean_cost_delta'] for r in pairs
            if r['split'] == row['split'] and r['problem'] == 'ALL')
    return pairs, recovery, totals


def diagnostic_tables(run, config, lock, val, test, args):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    refs = references(config, args)
    targets, identity = load_targets(config, refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = {}, {}, {}
    for name, ref in refs.items():
        old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
        pairs[name], recovery[name], totals[name] = comparison_tables(old, packed, targets, identity,
            'R105/' + config['arm'], name)
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results, paths=paths,
        pairs=pairs, recovery=recovery, recovery_summary=totals, target_identity=identity,
        historical={name: ref['provenance'] for name, ref in refs.items()})


def _receipt_identity(value, lock):
    required = dict(experiment='R105', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_factory=runner.MODEL_FACTORY,
        inference_rule=runner.INFERENCE_RULE, model_reselection=False, routing_calibration=False)
    if (any(value.get(k) != v for k, v in required.items()) or lock['arm'] not in ARMS
            or not isinstance(value.get('model_state_sha256'), str) or not value['model_state_sha256']):
        raise ValueError('R105 diagnostic/timing identity differs from the unique per-arm model')


def inference_timing(run, lock):
    value = read_json(Path(run) / 'inference_timing.json')
    _receipt_identity(value, lock)
    required = dict(repetitions=3, warmup_batches=1,
        scope='complete_standalone_model_original_GPU_resident_inputs', all_current_model_included=True,
        all_four_experts_computed=True, donor_present=False, cached_feature_latency_used=False,
        end_to_end_measured=False, input_labels_read=False,
        sample_policy='first128 frozen TEST inputs per problem')
    if (any(value.get(k) != v for k, v in required.items())
            or value.get('parameter_count') != runner.PARAMETER_COUNTS[lock['arm']]
            or not str(value.get('device', '')).startswith('cuda')
            or value.get('excludes') != 'dataset I/O, CPU-to-GPU transfer, training, routing diagnostic and solver execution'):
        raise ValueError('R105 requires actual complete standalone GPU-resident model timing')
    rows = value.get('per_problem', [])
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('R105 timing must cover all18 tasks')
    for row in rows:
        samples = row.get('standalone_model', {}).get('batch_seconds', [])
        mean = row.get('standalone_model', {}).get('mean_seconds_per_instance')
        if (row.get('instances') != 128 or len(samples) != 3
                or any(not math.isfinite(float(s)) or s <= 0 for s in samples)
                or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0
                or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-10)):
            raise ValueError('Invalid R105 latency samples/denominator')
    return value


def routing_receipt(run, lock):
    value = read_json(Path(run) / 'routing_diagnostic.json')
    _receipt_identity(value, lock)
    if (any(value.get(k) != v for k, v in runner.ROUTING_DIAGNOSTIC.items())
            or any(value.get(k) is not False for k in ('labels_read', 'targets_read', 'auxiliary_loss', 'solver_quotas'))
            or 'not mean node entropy' not in value.get('aggregation', '')):
        raise ValueError('R105 routing must be input-only, post-lock, descriptive exact-N VAL evidence')
    rows = value.get('per_problem', [])
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('R105 routing must cover first128 VAL inputs of all18 tasks')
    for row in rows:
        counts = row.get('exact_n_counts', {})
        sizes = {int(k): v for k, v in counts.items()}
        if (row.get('instances') != 128 or row.get('sample_indices') != list(range(128))
                or not sizes or any(k <= 0 or not isinstance(v, int) or v <= 0 for k, v in sizes.items())
                or sum(sizes.values()) != 128 or len(row.get('layers', [])) != 4
                or {r['layer'] for r in row['layers']} != set(range(4))
                or len(row.get('within_exact_n', [])) != 4 * len(sizes)
                or {(r['exact_n'], r['layer']) for r in row['within_exact_n']} !=
                   {(n, layer) for n in sizes for layer in range(4)}):
            raise ValueError('R105 routing sample/exact-N coverage is incomplete')
        for layer in [*row['layers'], *row['within_exact_n']]:
            count = sizes[layer['exact_n']] if 'exact_n' in layer else 128
            weights, coverage = layer.get('mean_expert_weights', []), layer.get('top1_instance_counts', [])
            entropy = layer.get('mean_graph_mixture_entropy_nats')
            if (layer.get('instances') != count or len(weights) != 4 or len(coverage) != 4
                    or any(not math.isfinite(float(w)) or not 0 <= w <= 1 for w in weights)
                    or not math.isclose(sum(weights), 1., abs_tol=1e-6)
                    or any(not isinstance(c, int) or c < 0 for c in coverage) or sum(coverage) != count
                    or not isinstance(entropy, (int, float)) or not math.isfinite(entropy)
                    or not -1e-6 <= entropy <= math.log(4) + 1e-6
                    or layer.get('variation_estimable') != (count > 1)):
                raise ValueError('R105 routing weights, entropy or coverage invalid')
            for key in ('instance_weight_std', 'instance_weight_range'):
                variation = layer.get(key)
                if ((count == 1 and variation is not None) or (count > 1 and
                        (not isinstance(variation, list) or len(variation) != 4
                         or any(not math.isfinite(float(v)) or v < 0 for v in variation)))):
                    raise ValueError('R105 within-exact-N variation is missing/invalid')
    return value


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        timing, routing = inference_timing(run, lock), routing_receipt(run, lock)
        expected_hash = test['identity']['model_state_sha256']
        if timing['model_state_sha256'] != expected_hash or routing['model_state_sha256'] != expected_hash:
            raise ValueError('R105 LIB requires timing/routing from the same locked TEST model')
        reference = previous.validate_nss_reference(config, args.nss_root)
        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R105', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))
        results = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
            phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        write_json(run / 'lib_stage_complete.json', dict(experiment='R105', arm=args.arm, complete=True,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            model_factory=runner.MODEL_FACTORY, nss_reference=reference,
            datasets=list(results), completed_utc=utc()))
        phase('complete')
        return results


def diagnostics_pdf(output, config, lock, tables, timing, routing):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, model = common.reporting_helpers(), 'R105/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, model + ' | Dense experts and actual latency',
                'Complete standalone model; synchronized GPU-resident inputs; no sparse-compute claim.', model, lock, 1)
            rows = [[r['problem'], '128', f"{1000 * r['standalone_model']['mean_seconds_per_instance']:.6f}"]
                    for r in timing['per_problem']]
            helper.add_table(fig, ['Problem', 'Instances', 'Milliseconds / instance'], rows, [.05, .37, .9, .46], font=9)
            fig.text(.05, .32, f"Total parameters: {config['trainable_parameter_count']:,}; inactive gate feature W: {config['inactive_router_feature_parameters']:,}", fontsize=10)
            fig.text(.05, .27, '\n'.join(textwrap.wrap(BOUNDARY, 128)), fontsize=7, va='top')
            pdf.savefig(fig)
            plt.close(fig)
            headings = ['Problem', 'Top1 (%)', 'SBS Top1 (%)', 'mean_cost', 'SBS cost',
                        'Oracle cost', 'vs_SBS (%)', 'vs_Oracle (%)', 'actual regret (%)']
            fig = common._page(helper, model + ' | Same selected model: full TRAIN',
                'Diagnostic only; TRAIN was never used to select the checkpoint; full TRAIN-fitted SBS.', model, lock, 2)
            rows = helpers.train_metric_rows(config, lock, tables)
            helper.add_table(fig, headings, [helper.metric_cells(r, r['problem']) for r in rows],
                [.045, .22, .91, .62], font=9, bold_first=True)
            fig.text(.05, .15, 'ALL task-equal; ratios of mean costs differ from per-instance actual regret.', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            for page, ref in enumerate(('R89A', 'R89B', 'R84A'), 3):
                fig = common._page(helper, model + ' | Fixed target36 vs ' + ref,
                    'True rescues/losses, false picks and aligned net cost; fixed targets, no reselection.', model, lock, page)
                helper.add_table(fig, ['Split', 'Winners', 'Rescued', 'Lost', 'New false', 'Net cost delta'],
                    [[r['split'], str(r['winner_count']), str(r['rescued']), str(r['lost']),
                      str(r['new_false_pick_count']), f"{r['full18_net_mean_cost_delta']:.9f}"]
                     for r in tables['recovery_summary'][ref]], [.05, .57, .9, .25], font=10)
                helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                    [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                      f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                     for r in tables['pairs'][ref] if r['problem'] == 'ALL'], [.05, .25, .9, .22], font=10)
                fig.text(.05, .16, 'Historical only, not retrained/matched; R89A AdamW, R89B Muon, R84A AdamW.', fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
            for offset, problem in enumerate(PROBLEMS, 6):
                row = next(r for r in routing['per_problem'] if r['problem'] == problem)
                fig = common._page(helper, model + ' | Input-only routing: ' + problem,
                    'First128 VAL after lock; entropy of graph-mean mixture, not mean node entropy.', model, lock, offset)
                cells = []
                for r in row['within_exact_n']:
                    variation = r['instance_weight_range']
                    cells.append([str(r['exact_n']), str(r['instances']), str(r['layer']),
                        '/'.join(f'{w:.4f}' for w in r['mean_expert_weights']),
                        f"{r['mean_graph_mixture_entropy_nats']:.5f}",
                        '/'.join(str(c) for c in r['top1_instance_counts']),
                        f'{max(variation):.5g}' if variation is not None else 'n=1: unavailable'])
                # Exact-size groups may be numerous; paginate rather than omit any.
                for start in range(0, len(cells), 28):
                    if start:
                        fig = common._page(helper, model + ' | Routing: ' + problem + ' (continued)',
                            'Same fixed input-only sample; no new sampling or selection.', model, lock, offset)
                    helper.add_table(fig, ['Exact N', 'n', 'Layer', 'Mean expert weights', 'Entropy', 'Top1 coverage', 'Max range'],
                        cells[start:start + 28], [.04, .22, .92, .62], font=8)
                    fig.text(.05, .14, 'Graph means cover valid physical nodes only; special tokens/padding excluded.', fontsize=9)
                    fig.text(.05, .10, 'Diversity/coverage is descriptive, not efficacy or evidence against coarse shortcuts.', fontsize=9)
                    pdf.savefig(fig)
                    plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def report(args):
    if args.device != 'cpu':
        raise ValueError('R105 report consumes saved evidence on CPU; no inference/reselection')
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R105', phase='loading_saved_predictions'))
            lib = read_json(run / 'lib_stage_complete.json')
            if (lib.get('complete') is not True or lib.get('arm') != args.arm
                    or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']
                    or lib.get('state_hash') != lock['state_hash'] or lib.get('model_factory') != runner.MODEL_FACTORY
                    or set(lib.get('datasets', [])) != set(LIBS)):
                raise ValueError('R105 report needs complete same-checkpoint/factory LIB evidence')
            timing, routing = inference_timing(run, lock), routing_receipt(run, lock)
            expected_hash = test['identity']['model_state_sha256']
            if timing['model_state_sha256'] != expected_hash or routing['model_state_sha256'] != expected_hash:
                raise ValueError('R105 routing/timing and locked predictions use different models')
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            result = shared.build_reports(run, config, lock, test, val,
                predictions, metrics, methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            for name, rows in (('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', helpers.train_metric_rows(config, lock, tables)),
                    ('routing_by_problem_exact_n_layer.csv', routing_csv(routing)),
                    ('inference_timing.csv', [dict(problem=r['problem'], instances=r['instances'],
                        milliseconds_per_instance=1000 * r['standalone_model']['mean_seconds_per_instance'],
                        parameter_count=config['trainable_parameter_count']) for r in timing['per_problem']])):
                common.write_csv(output / name, rows)
            for ref in tables['historical']:
                for stem, key in (('paired_changes', 'pairs'), ('target_recovery', 'recovery'),
                                  ('target_recovery_summary', 'recovery_summary')):
                    common.write_csv(output / (stem + '_vs_' + ref + '.csv'), tables[key][ref])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables, timing, routing),
                inference_timing=timing, routing_diagnostic=routing,
                historical_reference=tables['historical'], nss_reference=reference)
            _, train_proof = runner.selected_capture_proof(run, tables['results']['TRAIN'], lock, 'train')
            provenance = read_json(output / 'protocol.json')
            provenance.update(model_spec=config['model_spec'], gate=config['gate'],
                trainable_parameter_count=config['trainable_parameter_count'],
                inactive_router_feature_parameters=config['inactive_router_feature_parameters'],
                initialization=read_json(run / 'initialization.json'), target_identity=tables['target_identity'],
                historical_reference=tables['historical'], inference_timing=timing,
                routing_diagnostic=routing, evidence_boundary=BOUNDARY,
                selected_train_capture_proof=train_proof,
                selected_val_binding=read_json(run / 'final_metrics.json')['val']['checkpoint_binding'])
            write_json(output / 'protocol.json', provenance)
            for name in REQUIRED_PDFS:
                if not (output / name).is_file() or not (output / name).stat().st_size:
                    raise ValueError('R105 required PDF missing: ' + name)
            write_json(run / 'postprocess_status.json', dict(experiment='R105', phase='complete',
                exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R105', phase='postprocess_failed',
                exit_status=1, error=str(error), automatic_retry=False, updated_utc=utc()))
            raise


def routing_csv(routing):
    rows = []
    for problem in routing['per_problem']:
        for layer in [*problem['layers'], *problem['within_exact_n']]:
            row = dict(problem=problem['problem'], exact_n=layer.get('exact_n', 'ALL'),
                layer=layer['layer'], instances=layer['instances'], valid_physical_nodes=layer['valid_physical_nodes'],
                mean_graph_mixture_entropy_nats=layer['mean_graph_mixture_entropy_nats'],
                variation_estimable=layer['variation_estimable'])
            for i in range(4):
                row.update({f'expert{i}_mean': layer['mean_expert_weights'][i],
                    f'expert{i}_top1_count': layer['top1_instance_counts'][i],
                    f'expert{i}_std': layer['instance_weight_std'][i] if layer['variation_estimable'] else None,
                    f'expert{i}_range': layer['instance_weight_range'][i] if layer['variation_estimable'] else None})
            rows.append(row)
    return rows


def compare(args):
    if args.device != 'cpu':
        raise ValueError('R105 comparison is saved-evidence CPU only')
    selected, records, initializations = {}, {}, {}
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        selected[arm] = selected_predictions(run, config, lock, val, test)[0]
        records[arm], initializations[arm] = (config, lock), read_json(run / 'initialization.json')
    same_data(records[ARMS[0]][0], records[ARMS[1]][0])
    a, b = (initializations[arm] for arm in ARMS)
    if (a.get('common_fresh_model_state_sha256') != b.get('common_fresh_model_state_sha256')
            or not a.get('common_fresh_model_state_sha256')
            or any(r.get('cross_arm_complete_state_verified') is not True for r in (a, b))):
        raise ValueError('R105 compare requires verified complete fresh tensor equality, not just counts')
    refs = references(records[ARMS[0]][0], args)
    targets, identity = load_targets(records[ARMS[0]][0], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = comparison_tables(selected[ARMS[0]], selected[ARMS[1]],
        targets, identity, 'R105/' + ARMS[1], 'R105/' + ARMS[0])
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('paired_changes_B_vs_A.csv', pairs), ('target_recovery_B_vs_A.csv', recovery),
                       ('target_recovery_summary_B_vs_A.csv', totals)):
        common.write_csv(output / name, rows)
    result = dict(experiment='R105', complete=True, total_parameter_matched=True,
        active_parameter_matched=False, equal_flops_claim=False, identical_complete_fresh_parameters=True,
        A_inactive_router_feature_parameters=2048, evidence_boundary=BOUNDARY,
        checkpoints={arm: records[arm][1] for arm in ARMS}, training=False, inference=False)
    write_json(output / 'protocol.json', result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('lib', 'report', 'compare', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS)
    result.add_argument('--device', default='cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path, default=DEFAULT_NSS)
    result.add_argument('--historical-root', type=Path, default=DEFAULT_R84 / 'A_baseline')
    result.add_argument('--r89-root', type=Path, default=DEFAULT_R89 / 'B_muon_hybrid')
    result.add_argument('--r89a-root', type=Path, default=DEFAULT_R89 / 'A_adamw')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root', 'r89_root', 'r89a_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.command == 'compare':
        return compare(args)
    if args.arm is None:
        raise ValueError('--arm is required for R105 lib/report/status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        value = read_json(path) if path.exists() else dict(experiment='R105', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()
