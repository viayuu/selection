"""CPU-only R91 report recovery, without changing frozen sources or receipts."""

import argparse
import csv
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import traceback
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs/R91_probability_link_control'
ARMS = ('A_utility_ce', 'B_utility_probabilities')
CPU_ENV = dict(CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_only():
    if socket.gethostname().split('.')[0] != 'admin01':
        raise RuntimeError('This bounded recovery is authorized on admin01 CPU only')
    if os.environ.get('CONDA_DEFAULT_ENV') != 'easynco':
        raise RuntimeError('Activate easynco before report recovery')
    if any(os.environ.get(key) != value for key, value in CPU_ENV.items()):
        raise RuntimeError('Launch with the documented CPU-only environment')


def protected_files(root):
    paths = [root / 'protocol_lock.json']
    for arm in ARMS:
        run = root / arm
        paths.extend(run / name for name in (
            'protocol.json', 'final_test_lock.json', 'best.pt', 'best_eval.json',
            'training_complete.json', 'test_results.json', 'test_complete.json',
            'final_metrics.json', 'lib_stage_complete.json', 'lib_stage_status.json',
            'postprocess_status.json', 'status.json', 'result.json'))
        launch = root.parent / 'R91_launch' / arm
        paths.extend(launch / name for name in ('worker.json', 'controller.json', 'PDF.log'))
    for name in ('r91_experiment.py', 'r91_objective.py', 'r91_reporting.py',
                 'r84_reporting.py', 'r85_reporting.py', 'r87_reporting.py',
                 'r80_reporting.py', 'r77_best_model_report.py'):
        paths.append(HERE / name)
    return {str(path): sha(path) for path in paths}


def original_failures(root):
    result = {}
    for arm in ARMS:
        run = root / arm
        failure = read(run / 'postprocess_status.json')
        worker_path = root.parent / 'R91_launch' / arm / 'worker.json'
        controller_path = worker_path.with_name('controller.json')
        worker, controller = read(worker_path), read(controller_path)
        stages = {stage['name']: stage for stage in worker['stages']}
        if (failure['phase'] != 'postprocess_failed'
                or 'NotoSansCJK-Regular.ttc' not in failure['error']
                or any(stages[name]['exit_status'] != 0 for name in ('train', 'TEST', 'LIB'))
                or stages['PDF']['exit_status'] != 1):
            raise ValueError('Expected completed train/TEST/LIB and original font-only PDF failure')
        result[arm] = dict(original_failure_utc=failure['updated_utc'],
            original_postprocess=failure, original_worker_path=str(worker_path),
            original_controller_path=str(controller_path),
            original_pdf_stage=stages['PDF'], original_controller_state=controller['state'],
            original_controller_exit_status=controller['exit_status'])
    return result


def forbidden(*args, **kwargs):
    raise RuntimeError('Report-only recovery forbids training, inference and checkpoint loading')


@contextmanager
def report_context(reporting, root, receipt_dir):
    """Redirect only reporter status; keep original failure receipts byte-identical."""
    import torch
    from . import r91_experiment

    original_write = reporting.write_json
    destinations = {root / arm / 'postprocess_status.json':
                    receipt_dir / arm / 'postprocess_status.json' for arm in ARMS}

    def recovery_write(path, value):
        path = Path(path).resolve()
        if path in destinations:
            value = dict(value, recovery_kind='report_only', original_receipt_preserved=True)
            return write(destinations[path], value)
        outputs = [root / arm / 'full_model_statistics' for arm in ARMS]
        outputs.append(root / 'comparison_artifacts')
        if not any(path.is_relative_to(output) for output in outputs):
            raise RuntimeError('Unexpected reporter write outside report output: ' + str(path))
        return original_write(path, value)

    with ExitStack() as stack:
        stack.enter_context(patch.object(reporting, 'write_json', recovery_write))
        stack.enter_context(patch.object(torch, 'load', forbidden))
        stack.enter_context(patch.object(torch.nn.Module, '__call__', forbidden))
        for name in ('train', 'test', 'load_checkpoint', 'load_model', 'from_checkpoint', 'make_model'):
            stack.enter_context(patch.object(r91_experiment, name, forbidden))
        stack.enter_context(patch.object(reporting.common, 'r80_lib', forbidden))
        yield


def run_stage(args):
    from . import r91_reporting as reporting
    namespace = reporting.parser().parse_args([
        args.command, '--root', str(args.root), '--device', 'cpu',
        *(['--arm', args.arm] if args.arm else [])])
    before = protected_files(args.root)
    if args.command == 'report':
        run, config, lock, _, _ = reporting.locked_inputs(args.root, args.arm)
        for name in ('r91_reporting.py', 'r84_reporting.py', 'r85_reporting.py',
                     'r87_reporting.py', 'r80_reporting.py'):
            if sha(HERE / name) != config['source_sha256'][name]:
                raise ValueError('Frozen reporting source hash changed: ' + name)
        train, path = reporting.previous.selected_train(run, config, lock)
        if not path.is_file():
            raise FileNotFoundError('No cached selected TRAIN; inference is not authorized')
        print(json.dumps(dict(arm=args.arm, selected_update=lock['selected_update'],
            checkpoint_sha256=lock['checkpoint_sha256'], selected_train=str(path),
            selected_train_cached=True, train_macro=train['macro'])), flush=True)
    with report_context(reporting, args.root, args.receipt_dir):
        result = getattr(reporting, args.command)(namespace)
    if protected_files(args.root) != before:
        raise RuntimeError('Protected original receipt, checkpoint or source changed during recovery')
    print(json.dumps(dict(command=args.command, arm=args.arm, result=result)), flush=True)


def deliver(args):
    receipt_dir = args.receipt_dir
    receipt_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = receipt_dir / 'recovery_receipt.json'
    record = dict(experiment='R91', state='running_report_only', recovery_kind='report_only',
        started_utc=utc(), host=socket.gethostname(), environment=CPU_ENV,
        original_failures=original_failures(args.root),
        protected_sha256_before=protected_files(args.root), commands=[],
        training=False, inference=False, test_rerun=False, lib_rerun=False,
        checkpoint_reselection=False, automatic_retry=False, frozen_reporter_modified=False)
    write(receipt_path, record)
    try:
        stages = [('report', arm) for arm in ARMS] + [('compare', None)]
        for command, arm in stages:
            argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r91_reporting_repair',
                    command, '--root', str(args.root), '--receipt-dir', str(receipt_dir)]
            if arm:
                argv.extend(('--arm', arm))
            log = receipt_dir / ((arm or 'round') + '_' + command + '.log')
            item = dict(argv=argv, shell_command=' '.join(
                shlex.quote(key + '=' + value) for key, value in CPU_ENV.items())
                + ' ' + shlex.join(argv), started_utc=utc(), log=str(log))
            record['commands'].append(item)
            write(receipt_path, record)
            print('START ' + command + ' ' + str(arm), flush=True)
            with log.open('x') as stream:
                completed = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                    cwd=HERE.parent.parent, env=os.environ.copy(), check=False)
            item.update(finished_utc=utc(), exit_status=completed.returncode)
            write(receipt_path, record)
            print('EXIT ' + str(completed.returncode) + ': ' + str(log), flush=True)
            if completed.returncode:
                raise RuntimeError('Report recovery command failed; no automatic retry: ' + str(log))
        record['protected_sha256_after'] = protected_files(args.root)
        record['protected_unchanged'] = record['protected_sha256_before'] == record['protected_sha256_after']
        if not record['protected_unchanged']:
            raise RuntimeError('Protected input changed')
        record.update(state='recovered_report_only', complete=True)
    except BaseException:
        record.update(state='report_only_recovery_failed', complete=False,
                      traceback=traceback.format_exc())
        raise
    finally:
        record['finished_utc'] = utc()
        write(receipt_path, record)
        print('RECEIPT ' + str(receipt_path), flush=True)


def complete(args):
    record = read(args.receipt_dir / 'recovery_receipt.json')
    if (record.get('state') != 'recovered_report_only' or not record.get('complete')
            or len(record['commands']) != 3
            or any(item['exit_status'] != 0 for item in record['commands'])
            or protected_files(args.root) != record['protected_sha256_before']):
        raise ValueError('Both reports and compare must complete with original inputs unchanged')
    reports, evaluations = {}, {}
    started = datetime.fromisoformat(record['started_utc']).timestamp()
    for arm in ARMS:
        run = args.root / arm
        output = run / 'full_model_statistics'
        names = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf',
                 'target_recovery_and_train.pdf', 'metrics_by_problem.csv',
                 'nss_comparison.csv', 'train_metrics_by_problem.csv',
                 'target_recovery_summary_vs_R84A.csv', 'target_recovery_summary_vs_R88B.csv')
        for name in names:
            if not (output / name).is_file() or (output / name).stat().st_size == 0:
                raise FileNotFoundError(output / name)
        reports[arm] = dict(output=str(output),
            **{name.replace('.', '_'): str(output / name) for name in names},
            checkpoint_lock=read(run / 'final_test_lock.json'),
            joint_val_pass=read(output / 'protocol.json')['val_empirical_screen'][0]['joint_val_pass'])
        for path in [run / 'evaluation' / split / 'selected_best/rank000.npz'
                     for split in ('train', 'val', 'test')] + [
                         run / 'lib_evaluation/predictions' / (name + '.npz')
                         for name in ('TSPLIB', 'CVRPLIB')]:
            stat = path.stat()
            if stat.st_mtime > started:
                raise ValueError('Cached evaluation archive changed after recovery began: ' + str(path))
            evaluations[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                predates_report_recovery=True)
    receipt = dict(experiment='R91', complete=True, state='recovered_report_only',
        completed_utc=utc(), host=socket.gethostname(), both_arm_report_paths=reports,
        comparison_pdf=str(args.root / 'comparison_artifacts/COMPARISON.pdf'),
        recovery_receipt=str(args.receipt_dir / 'recovery_receipt.json'),
        report_commands=record['commands'], original_failures=record['original_failures'],
        old_failure_unchanged=True, evaluation_unchanged=True,
        evaluation_verification='Original result/lock/checkpoint SHA256 unchanged; cached archive sizes/mtimes recorded and predate recovery. No NPZ hash rescan.',
        cached_evaluation_archives=evaluations,
        protected_sha256_before=record['protected_sha256_before'],
        protected_sha256_after=protected_files(args.root),
        training=False, inference=False, test_rerun=False, lib_rerun=False,
        checkpoint_reselection=False, automatic_retry=False,
        frozen_reporter_modified=False, goal_or_index_modified=False)
    path = args.root / 'report_only_recovery_complete.json'
    write(path, receipt)
    print(str(path), flush=True)


def summary(args):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from . import r91_reporting as reporting

    completion = read(args.root / 'report_only_recovery_complete.json')
    if not completion['complete']:
        raise ValueError('Full reports must be ready before the round summary')

    def rows(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    macro, gates, recovery = [], [], []
    runs = [('R84A', reporting.DEFAULT_R84 / 'A_baseline'),
            ('R88B', reporting.DEFAULT_R88B),
            *[(f'R91{letter}', args.root / arm) for letter, arm in zip('AB', ARMS)]]
    for name, run in runs:
        val = read(run / 'final_metrics.json')['val']['macro']
        test = read(run / 'test_results.json')['result']['macro']
        for split, values in (('VAL', val), ('TEST', test)):
            macro.append(dict(model=name, split=split, top1_pct=100 * values['macro_top1'],
                **{key: values['macro_' + key] for key in ('mean_cost', 'sbs_cost',
                   'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')}))
    for letter, arm in zip('AB', ARMS):
        output = args.root / arm / 'full_model_statistics'
        gates.extend(dict(row, arm=letter) for row in rows(output / 'val_empirical_screen.csv'))
        for reference in ('R84A', 'R88B'):
            recovery.extend(dict(row, arm=letter, reference=reference) for row in
                rows(output / ('target_recovery_summary_vs_' + reference + '.csv'))
                if row['split'] in ('VAL', 'TEST'))
    paired = [row for row in rows(args.root / 'comparison_artifacts/paired_changes_B_vs_A.csv')
              if row['problem'] == 'ALL' and row['split'] in ('VAL', 'TEST')]
    reporting.common.write_csv(args.root / 'RESULTS_macro.csv', macro)
    reporting.common.write_csv(args.root / 'RESULTS_target36.csv', recovery)
    helper = reporting.common.reporting_helpers()
    helper.setup_style()

    def page(title, number):
        fig = plt.figure(figsize=(13.5, 9))
        fig.text(.045, .95, title, fontsize=21, fontweight='bold', color='#093f45')
        fig.text(.045, .905, 'R91A: utility CE only | R91B: utility links also for risk/R-Drop | seed2, logical batch1024', fontsize=11)
        fig.text(.045, .025, f'R91 report-only recovery | {number}/2 | admin01 CPU | no inference, retraining or reselection', fontsize=9)
        return fig

    pdf_path = args.root / 'RESULTS.pdf'
    temporary = pdf_path.with_name(f'.RESULTS.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = page('R91 | Locked outcomes: delivery recovered, research gate not passed', 1)
            helper.add_table(fig, ['Model / split', 'Top1 (%)', 'mean_cost', 'SBS cost',
                'Oracle cost', 'vs SBS (%)', 'vs Oracle (%)', 'actual regret (%)'],
                [[row['model'] + ' / ' + row['split'], f"{row['top1_pct']:.6f}",
                  *[f"{row[key]:.9f}" for key in ('mean_cost', 'sbs_cost', 'oracle_cost')],
                  *[f"{row[key]:.6f}" for key in ('vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')]]
                 for row in macro], [.045, .55, .91, .29], font=9.5)
            fig.text(.045, .50, 'Equal weight across18 problems. SBS fitted on TRAIN81000; Oracle = per-instance legal minimum; native/FP64 ind.', fontsize=10)
            fig.text(.045, .46, 'Joint VAL gate: Top1 strictly increases, cost and actual regret do not increase versus BOTH historical references.', fontsize=10)
            helper.add_table(fig, ['Arm / reference (VAL)', 'Top1 delta pp', 'mean_cost delta',
                'regret delta pp', 'Reference pass', 'Joint pass'],
                [[row['arm'] + ' / ' + row['baseline'], f"{float(row['top1_delta_pp']):+.6f}",
                  f"{float(row['mean_cost_delta']):+.9f}", f"{float(row['actual_regret_delta_pp']):+.6f}",
                  row['pass_reference'], row['joint_val_pass']] for row in gates],
                [.045, .235, .91, .18], font=10)
            for y, text in ((.18, 'Both arms FAIL the fixed joint VAL gate. B raises TEST Top1 but worsens TEST cost/regret versus A and R84A.'),
                    (.135, 'R84A and same-role R88B are historical references, not newly matched original-loss controls; single seed only.'),
                    (.09, 'All18 use frozen NEW90k splits81000/4500/4500. TEST informed prior R&D; these are not untouched blind results.'),
                    (.05, 'A lock: update37440 / fc889238acd827fa | B lock: update40320 / cdcb5f1da1e3c30c. Both trained57600 updates.')):
                fig.text(.045, y, text, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            fig = page('R91 | Frozen target36 recovery and matched B-minus-A tradeoff', 2)
            helper.add_table(fig, ['Arm / ref / split', 'Winners', 'Correct', 'Selected', 'Recall %',
                'Precision %', 'Rescued', 'Lost', 'New false', 'New false regret pp', 'Target regret delta pp'],
                [[row['arm'] + '/' + row['reference'] + '/' + row['split'], row['winner_count'],
                  row['correct_count'], row['selected_count'], f"{float(row['recall_pct']):.4f}",
                  f"{float(row['precision_pct']):.4f}", row['rescued'], row['lost'], row['new_false_pick_count'],
                  f"{float(row['new_false_pick_regret_contribution_pp']):.7f}",
                  f"{float(row['target_regret_delta_pp']):+.7f}"] for row in recovery],
                [.045, .52, .91, .32], font=8.5)
            fig.text(.045, .47, 'Same36 R84A VAL-defined method/problem groups; never reselected. VAL4625 / TEST4598 true target winners.', fontsize=10)
            fig.text(.045, .43, 'B recovers more true winners, but TEST recall remains0.5655%; target-local regret gains do not imply global gains.', fontsize=10)
            helper.add_table(fig, ['B minus A', 'Top1 delta pp', 'Corrected', 'Harmed', 'mean_cost delta', 'actual regret delta pp'],
                [[row['split'], f"{float(row['top1_delta_pp']):+.6f}", row['corrected'], row['harmed'],
                  f"{float(row['mean_cost_delta']):+.9f}", f"{float(row['actual_regret_delta_pp']):+.6f}"]
                 for row in paired], [.045, .25, .91, .13], font=10)
            for y, text in ((.20, 'Recommendation: retain R84A as reference; neither R91 arm establishes joint cost/regret plus recovery improvement.'),
                    (.155, 'Full reports: each arm/full_model_statistics/MODEL_STATISTICS.pdf (NSS4 + ALL/per-task SBS/Oracle metrics).'),
                    (.11, 'selection_distributions.pdf:18 pages, paired selection/winner bars; target_recovery_and_train.pdf: cached TRAIN/VAL/TEST.'),
                    (.065, 'Original font failure, workers/controllers, checkpoint locks and evaluations remain unchanged; see report_only_recovery_complete.json.')):
                fig.text(.045, y, text, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
        temporary.replace(pdf_path)
    finally:
        temporary.unlink(missing_ok=True)
    completion.update(summary_pdf=str(pdf_path), summary_markdown=str(args.root / 'RESULTS.md'),
        summary_macro_csv=str(args.root / 'RESULTS_macro.csv'),
        summary_target36_csv=str(args.root / 'RESULTS_target36.csv'), summary_completed_utc=utc())
    write(args.root / 'report_only_recovery_complete.json', completion)
    print(str(pdf_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('deliver', 'report', 'compare', 'complete', 'summary'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--receipt-dir', type=Path)
    args = parser.parse_args()
    cpu_only()
    args.root = args.root.resolve()
    if args.receipt_dir is None:
        if args.command != 'deliver':
            parser.error('--receipt-dir is required for a report/compare child')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        args.receipt_dir = args.root / 'report_only_recovery' / stamp
    args.receipt_dir = args.receipt_dir.resolve()
    if args.command == 'report' and args.arm is None:
        parser.error('report requires --arm')
    if args.command in ('deliver', 'complete', 'summary'):
        globals()[args.command](args)
    else:
        run_stage(args)


if __name__ == '__main__':
    main()
