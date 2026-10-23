"""R104 saved-evidence full18 PDFs/NSS4/recovery, plus separate native LIB stage."""

import argparse
import copy
import fcntl
import json
import math
import os
import textwrap
from pathlib import Path

import numpy as np

from . import r87_reporting as helpers
from . import r93_reporting as historical
from . import r104_experiment as runner
from .r76_experiment import file_hash, read_json, utc, write_json


ARMS, BUDGETS, DEFAULT_ROOT = runner.ARMS, runner.BUDGETS, runner.DEFAULT_ROOT
FACTORY = 'code.V4.r104_experiment'
shared, previous, common = helpers.shared, helpers.previous, helpers.common
PROBLEMS, LIBS = tuple(helpers.PROBLEMS), shared.LIBS
DEFAULT_R84, DEFAULT_NSS = helpers.DEFAULT_R84, helpers.DEFAULT_NSS
DEFAULT_R89 = DEFAULT_ROOT.parent / 'R89_optimizer_geometry'
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
                 'training_and_method_diagnostics.pdf', 'target_recovery_and_train.pdf')
BOUNDARY = ('Width256 instance4/joint2 and width128 instance8/joint4 are two capacity axes, '
    'not equal-parameter, equal-FLOPs or identical-full-initialization A/B controls. Same-name/shape '
    'fresh R84 seed2 parameters are copied; resized/new parameters use isolated seed104002. '
    'Original R88B physical TSP node0, raw inputs/geometry/distance, methodID+34, original full '
    'score MLP, utility differences/maximin and R77 CE/comparison/risk/two-view R-Drop remain. '
    'No teacher, context labels, features, routes or new targets. R89A AdamW, R89B Muon and '
    'R84A are historical same-data references, not freshly matched controls; R89B differs in '
    'optimizer. Hardware and microbatch differences preclude a clean capacity-only causal claim. '
    'Fixed R84 target36, true rescues/losses, false picks and net cost are separate from selection '
    'frequency. Native/FP64 ind and legal pools; TRAIN-fitted SBS; ALL task-equal. TEST is '
    'development-exposed; one whole-model minimum full18 VAL lock, no TEST/LIB reselection. '
    'Timing includes the complete current standalone model on original GPU-resident inputs, '
    'not donor/cached-feature latency; excludes dataset I/O, transfer and solver execution.')


def capture_val(val, lock):
    """Validate official summary binding, then preserve archive identity for readers."""
    binding = val.get('checkpoint_binding', {})
    expected = dict(rule=runner.VAL_BINDING_RULE, checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], selected_update=lock['selected_update'], archive_identity_unchanged=True)
    capture, identity = val.get('archive_capture_identity'), val.get('identity', {})
    if (not isinstance(capture, dict) or any(binding.get(k) != v for k, v in expected.items())
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or identity != dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
            or capture.get('state_hash') != lock['state_hash']
            or capture.get('successful_updates') != lock['selected_update']
            or capture.get('model_state_sha256') != binding.get('model_state_sha256')):
        raise ValueError('R104 selected VAL summary lacks its exact-state official binding')
    result = copy.deepcopy(val)
    result['identity'] = copy.deepcopy(capture)
    return result


def locked_inputs(root, arm):
    run, config, lock, test, val = shared.locked_inputs(Path(root), arm, experiment='R104', arms=BUDGETS)
    runner.validate_checkpoint_config(config)
    val = capture_val(val, lock)
    if val['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('R104 VAL/TEST model states differ')
    return run, config, lock, test, val


def same_data(left, right):
    for key in ('data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids', 'baselines', 'winner_rule'):
        if key not in left or left[key] != right.get(key):
            raise ValueError('R104 historical comparison requires identical ' + key)


def selected_predictions(run, config, lock, val, test):
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    model = config['experiment'] + '/' + config['arm']
    return packed, [dict(r, model=model) for r in methods], [dict(r, model=model) for r in summaries], results, paths


def references(config, args):
    result = historical.references(config, args)
    path = Path(args.r89a_root)
    run, old_config, lock, test, val = shared.locked_inputs(path.parent, path.name,
        experiment='R89', arms=dict(A_adamw=57600, B_muon_hybrid=57600))
    same_data(config, old_config)
    result['R89A'] = dict(run=run, config=old_config, lock=lock, test=test, val=val,
        provenance=dict(model='R89A', run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
            same_data=True, retrained=False, newly_matched_control=False, optimizer='AdamW'))
    result['R89B']['provenance'].update(optimizer='Muon hybrid', optimizer_matched=False)
    for ref in result.values():
        ref['provenance'].update(parameter_matched=False, equal_flops=False, identical_full_initialization=False)
    return result


def load_targets(config, old_config, old_lock):
    targets, identity = helpers.load_targets(config, old_config, old_lock)
    if len(targets) != 36:
        raise ValueError('R104 retains the frozen R84 target36 without reselection')
    return targets, dict(identity, target_count=36, sha256=file_hash(Path(identity['path'])))


def comparison_tables(old, new, targets, identity, model, baseline):
    pairs, recovery, totals = helpers.comparison_tables(old, new, targets, identity, model, baseline)
    for row in recovery:
        left, right = old[row['split']][row['problem']], new[row['split']][row['problem']]
        slot, index = right.pool.index(row['solver']), np.arange(right.n)
        truth, was, chosen = right.winner == slot, left.pred == slot, right.pred == slot
        incoming_false = ~truth & ~was & chosen
        outgoing_false = ~truth & was & ~chosen
        delta = right.costs[index, right.pred] - left.costs[index, left.pred]
        row.update(net_rescues=row['rescued'] - row['lost'],
            new_false_pick_cost_delta_contribution=float(delta[incoming_false].sum() / right.n),
            removed_false_pick_count=int(outgoing_false.sum()),
            removed_false_pick_cost_delta_contribution=float(delta[outgoing_false].sum() / right.n),
            target_winner_cost_delta_contribution=float(delta[truth].sum() / right.n),
            full_problem_mean_cost_delta=float(delta.mean()))
    for row in totals:
        row['net_rescues'] = row['rescued'] - row['lost']
        for key in ('new_false_pick_cost_delta_contribution', 'removed_false_pick_cost_delta_contribution',
                    'target_winner_cost_delta_contribution'):
            row[key] = sum(r[key] for r in recovery if r['split'] == row['split']) / len(PROBLEMS)
        # Full-population net cost comes from aligned paired predictions, not target counts.
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
            'R104/' + config['arm'], name)
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results, paths=paths,
        pairs=pairs, recovery=recovery, recovery_summary=totals, target_identity=identity,
        historical={name: ref['provenance'] for name, ref in refs.items()})


def inference_timing(run, lock):
    value = read_json(Path(run) / 'inference_timing.json')
    required = dict(experiment='R104', complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], repetitions=3, warmup_batches=1,
        scope='complete_standalone_model_original_GPU_resident_inputs', all_current_model_included=True,
        donor_present=False, cached_feature_latency_used=False, end_to_end_measured=False)
    if (any(value.get(k) != v for k, v in required.items()) or value.get('arm') not in ARMS
            or value.get('parameter_count') != runner.PARAMETER_COUNTS[value['arm']]
            or value.get('inference_rule') != runner.INFERENCE_RULE
            or not str(value.get('device', '')).startswith('cuda')):
        raise ValueError('R104 requires actual complete standalone GPU-resident model timing')
    rows = value.get('per_problem', [])
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('R104 timing must cover all18 tasks')
    for row in rows:
        samples = row.get('standalone_model', {}).get('batch_seconds', [])
        mean = row.get('standalone_model', {}).get('mean_seconds_per_instance')
        if (row.get('instances') != 128 or len(samples) != 3
                or any(not math.isfinite(float(s)) or s <= 0 for s in samples)
                or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0
                or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-10)):
            raise ValueError('Invalid R104 latency samples or denominator')
    return value


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference = previous.validate_nss_reference(config, args.nss_root)
        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R104', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))
        results = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
            phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        write_json(run / 'lib_stage_complete.json', dict(experiment='R104', complete=True,
            checkpoint_sha256=lock['checkpoint_sha256'], nss_reference=reference,
            datasets=list(results), completed_utc=utc()))
        phase('complete')
        return results


def diagnostics_pdf(output, config, lock, tables, timing):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, model = common.reporting_helpers(), 'R104/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    with PdfPages(temporary) as pdf:
        fig = common._page(helper, model + ' | Capacity and actual latency',
            'Complete standalone model; synchronized GPU-resident original inputs; no donor.', model, lock, 1)
        rows = [[p, '128', f"{1000 * r['standalone_model']['mean_seconds_per_instance']:.6f}"]
                for p in PROBLEMS for r in timing['per_problem'] if r['problem'] == p]
        helper.add_table(fig, ['Problem', 'Instances', 'Milliseconds / instance'], rows, [.05, .34, .9, .5], font=9)
        fig.text(.05, .29, f"Parameters: {config['trainable_parameter_count']:,}; copied fresh: {config['copied_parameter_count']:,}; independent: {config['independent_capacity_parameter_count']:,}", fontsize=10)
        fig.text(.05, .24, '\n'.join(textwrap.wrap(BOUNDARY, 120)), fontsize=7, va='top')
        pdf.savefig(fig)
        plt.close(fig)
        for page, ref in enumerate(('R89A', 'R89B', 'R84A'), 2):
            fig = common._page(helper, model + ' | Fixed target36 vs ' + ref,
                'True rescues, losses, false picks and aligned full-population net cost; no reselection.', model, lock, page)
            helper.add_table(fig, ['Split', 'Winners', 'Rescued', 'Lost', 'New false', 'Net cost delta'],
                [[r['split'], str(r['winner_count']), str(r['rescued']), str(r['lost']),
                  str(r['new_false_pick_count']), f"{r['full18_net_mean_cost_delta']:.9f}"]
                 for r in tables['recovery_summary'][ref]], [.05, .57, .9, .25], font=10)
            helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                 for r in tables['pairs'][ref] if r['problem'] == 'ALL'], [.05, .25, .9, .22], font=10)
            fig.text(.05, .17, 'Historical only, not retrained/matched; R89A AdamW, R89B Muon, R84A original AdamW.', fontsize=9)
            fig.text(.05, .12, 'ALL net cost is the equal-task mean of aligned instance cost differences, not target frequency.', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
    temporary.replace(path)
    return str(path)


def report(args):
    if args.device != 'cpu':
        raise ValueError('R104 report uses saved predictions on CPU; inference is a separate stage')
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R104', phase='loading_saved_predictions'))
            lib = read_json(run / 'lib_stage_complete.json')
            if lib.get('complete') is not True or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']:
                raise ValueError('R104 report needs complete same-checkpoint LIB evidence')
            timing = inference_timing(run, lock)
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
                    ('inference_timing.csv', [dict(problem=r['problem'], instances=r['instances'],
                        milliseconds_per_instance=1000 * r['standalone_model']['mean_seconds_per_instance'],
                        parameter_count=config['trainable_parameter_count']) for r in timing['per_problem']])):
                common.write_csv(output / name, rows)
            for ref in tables['historical']:
                for stem, key in (('paired_changes', 'pairs'), ('target_recovery', 'recovery'),
                                  ('target_recovery_summary', 'recovery_summary')):
                    common.write_csv(output / (stem + '_vs_' + ref + '.csv'), tables[key][ref])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables, timing),
                inference_timing=timing, historical_reference=tables['historical'], nss_reference=reference)
            provenance = read_json(output / 'protocol.json')
            provenance.update(model_spec=config['model_spec'], capacity=config['capacity'],
                trainable_parameter_count=config['trainable_parameter_count'],
                initialization=read_json(run / 'initialization.json'), target_identity=tables['target_identity'],
                historical_reference=tables['historical'], inference_timing=timing, evidence_boundary=BOUNDARY,
                selected_val_binding=read_json(run / 'final_metrics.json')['val']['checkpoint_binding'])
            write_json(output / 'protocol.json', provenance)
            for name in REQUIRED_PDFS:
                if not (output / name).is_file() or not (output / name).stat().st_size:
                    raise ValueError('R104 report PDF missing: ' + name)
            write_json(run / 'postprocess_status.json', dict(experiment='R104', phase='complete',
                exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R104', phase='postprocess_failed',
                exit_status=1, error=str(error), automatic_retry=False, updated_utc=utc()))
            raise


def compare(args):
    if args.device != 'cpu':
        raise ValueError('R104 comparison is saved-evidence CPU only')
    selected, records = {}, {}
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        selected[arm] = selected_predictions(run, config, lock, val, test)[0]
        records[arm] = (config, lock)
    same_data(records[ARMS[0]][0], records[ARMS[1]][0])
    refs = references(records[ARMS[0]][0], args)
    targets, identity = load_targets(records[ARMS[0]][0], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = comparison_tables(selected[ARMS[0]], selected[ARMS[1]],
        targets, identity, 'R104/' + ARMS[1], 'R104/' + ARMS[0])
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('paired_changes_B_vs_A.csv', pairs), ('target_recovery_B_vs_A.csv', recovery),
                       ('target_recovery_summary_B_vs_A.csv', totals)):
        common.write_csv(output / name, rows)
    result = dict(experiment='R104', complete=True, matched=False, parameter_matched=False,
        equal_flops=False, identical_full_initialization=False, evidence_boundary=BOUNDARY,
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
        raise ValueError('--arm is required for R104 lib/report/status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        value = read_json(path) if path.exists() else dict(experiment='R104', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()
