"""Explicit C evaluation recovery; frozen R92 training sources are not edited.

Trees were fitted/VAL-selected on the already materialized input statistics.
Recomputing those FP32 statistics inside another forward can cross a tree border.
Use the canonical input-only feature rows for known-split evaluation, not costs
or labels as inputs. The untouched raw-input factory remains the LIB factory.
"""

import argparse
import copy
import fcntl
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from . import r92_experiment as runner
from . import r92_tree as tree
from .r76_experiment import file_hash, fixed_sbs_metrics, open_cache, read_json, utc, write_json
from .r77_data import parameter_hash
from .r80_experiment import save_npz, wandb_run
from .r85_experiment import selection_counts, zero_summary


ROOT, LAUNCH = runner.DEFAULT_ROOT, runner.DEFAULT_ROOT.parent / 'R92_launch'
SHARED_CACHE = runner.DEFAULT_PREPARED_ROOT.parent / 'input_cache'


def cached_evaluate(bank, args, manifest, baselines, split, tag, step,
                    state_hash, checkpoint_sha256=None):
    if args.arm != 'C' or not isinstance(bank, tree.TreeBank):
        raise ValueError('This recovery is restricted to the completed C tree bank')
    directory = args.root / args.arm / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment='R92', arm='C', split=split, tag=tag,
        successful_updates=step, state_hash=state_hash,
        model_state_sha256=parameter_hash(bank.state_dict()),
        checkpoint_sha256=checkpoint_sha256, winner_rule=runner.WINNER_RULE)
    if (directory / 'identity.json').exists() and read_json(directory / 'identity.json') != identity:
        raise ValueError('Recovery cannot overwrite another evaluation identity')
    if (directory / 'metrics.json').exists() and (directory / 'rank000.npz').exists():
        return read_json(directory / 'metrics.json')
    write_json(directory / 'identity.json', identity)
    started, packed, per_problem, method_rows, feature_sources = time.monotonic(), {}, {}, [], {}
    for problem in PROBLEMS:
        tree._status(args, 'cached_feature_evaluation', successful_updates=step,
            split=split, tag=tag, evaluation_problem=problem, CPU_only_recovery=True)
        phase = ('test',) if split == 'test' else ('train', 'val')
        phase_name = 'TEST' if split == 'test' else 'TRAIN_VAL'
        metadata_path = args.root / 'features' / phase_name / (problem + '.json')
        if not metadata_path.is_file():
            raise RuntimeError('Recovery requires the existing canonical input features, never a rebuild')
        rows = tree.feature_cache(args, manifest, problem, phase, torch.device('cpu'))
        cache = open_cache(args.cache_root, split, problem, manifest, args.data_root)
        try:
            offset = 81000 if split == 'val' else 0
            scores = bank.predict_features(problem, rows[offset:offset + cache.size])
            with np.load(args.data_root / (problem + split) / 'input_metadata.npz', allow_pickle=False) as source:
                release_indices = source['source_index'].copy()
            data = dict(logits=scores, indices=np.arange(cache.size, dtype=np.int64),
                source_indices=np.arange(cache.size, dtype=np.int64), release_source_indices=release_indices,
                costs=cache.raw['costs'].copy(), winner=cache.raw['winner'].copy(),
                pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids, dtype=np.int64),
                nodes=np.asarray(cache.lengths).copy())
            result, data['pred'] = fixed_sbs_metrics(scores, cache.raw, baselines[problem])
            per_problem[problem] = result
            save_npz(directory / 'by_problem' / (problem + '.npz'), data)
            prefix = problem + '__'
            packed.update({prefix + name: value for name, value in data.items()})
            method_rows.extend(selection_counts(packed, prefix, problem, split))
            feature_sources[problem] = dict(path=str(metadata_path), sha256=file_hash(metadata_path),
                input_only=True, feature_columns=len(tree.COLUMNS), reencoded=False)
        finally:
            cache.close()
            rows._mmap.close()
    save_npz(directory / 'rank000.npz', packed)
    result = runner.engine.summarize(per_problem)
    result.update(seconds=time.monotonic() - started, tag=tag, successful_updates=step,
        sample_exposures=None, training_unit='trees_per_task', identity=identity,
        prediction_archive=str((directory / 'rank000.npz').resolve()),
        own_train_sbs=runner.engine.summarize(per_problem), zero_selection=zero_summary(method_rows),
        input_pipeline='canonical materialized input-only357 structural features; no FP32 re-encoding',
        feature_sources=feature_sources, recovery_module_sha256=file_hash(Path(__file__)))
    write_json(directory / 'method_selection.json', dict(methods=method_rows, summary=result['zero_selection']))
    write_json(directory / 'metrics.json', result)
    print('[R92 C cached evaluation]', split, json.dumps(result['macro']), flush=True)
    return result


def protected(root):
    run, launch = root / 'C', LAUNCH / 'C'
    paths = [run / 'best.pt', run / 'protocol.json', run / 'error.json',
             launch / 'worker.json', launch / 'controller.json']
    paths += sorted((run / 'tree_models').glob('*'))
    return {str(path): file_hash(path) for path in paths if path.is_file()}


def finish_test(root, cache_root):
    """Only the missing TEST stage, in the original cache-owning container."""
    root = Path(root).resolve()
    run, folder = root / 'C', root / 'C/recovery'
    if not (run / 'training_complete.json').is_file() or not (run / 'final_test_lock.json').is_file():
        raise RuntimeError('Finish TEST requires the already completed global VAL lock')
    from .r92_launch import base
    base.assert_container('3090_1')
    before = protected(root)
    lock_hash = file_hash(run / 'final_test_lock.json')
    args = runner.parser().parse_args(['test', '--arm', 'C', '--root', str(root),
        '--cache-root', str(cache_root), '--device', 'cpu', '--wandb-mode', 'offline'])
    args.experiment = 'R92'
    torch.set_num_threads(2)
    with patch.object(runner, 'evaluate', side_effect=cached_evaluate):
        result = tree.test(args, torch.device('cpu'))
    if protected(root) != before or file_hash(run / 'final_test_lock.json') != lock_hash:
        raise RuntimeError('TEST recovery changed protected model or VAL lock')
    write_json(folder / 'TEST_recovery_complete.json', dict(complete=True, completed_utc=utc(),
        CPU_only=True, allocated_container=True, runtime_cache=str(cache_root),
        protected_unchanged=True, formal_VAL_lock_sha256=lock_hash,
        optimizer_updates=0, fitted_trees=0, test=result['macro']))
    return result


def finish_report(root):
    """CPU report continuation after the allocated-cache TEST stage succeeds."""
    root = Path(root).resolve()
    run, folder = root / 'C', root / 'C/recovery'
    stage = read_json(folder / 'TEST_recovery_complete.json')
    receipt = read_json(folder / 'recovery_receipt.json')
    if not stage.get('complete') or protected(root) != receipt['protected_before']:
        raise RuntimeError('Report needs successful TEST and unchanged original models/failed receipts')
    from . import r92_reporting as reporting
    args = reporting.parser().parse_args(['lib', '--root', str(root), '--arm', 'C', '--device', 'cpu'])
    reporting.lib_stage(args)
    args.command = 'report'
    reporting.report(args)
    if protected(root) != receipt['protected_before']:
        raise RuntimeError('Reporting changed protected originals')
    receipt.update(state='complete', complete=True, completed_utc=utc(), protected_after=protected(root),
        protected_unchanged=True, TEST_LIB_PDF_complete=True,
        selected_best=read_json(run / 'final_test_lock.json'),
        first_recovery_stage='canonical VAL/TRAIN completed on ADMIN01; TEST path unavailable there',
        test_stage=stage, final_recovery_source_sha256=file_hash(Path(__file__)))
    write_json(folder / 'recovery_completion_receipt.json', receipt)
    write_json(run / 'recovery_complete.json', receipt)
    print('C_RECOVERY_COMPLETE', receipt['selected_best']['selected_update'], flush=True)
    return receipt


def recover(root=ROOT):
    root = Path(root).resolve()
    run, folder = root / 'C', root / 'C/recovery'
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / 'recovery_receipt.json'
    if receipt_path.exists():
        raise RuntimeError('Recovery was already attempted; no implicit retry')
    if any((run / name).exists() for name in ('training_complete.json', 'final_test_lock.json', 'test_started.json')):
        raise RuntimeError('This recovery is only for the pre-lock C evaluation failure')
    config = read_json(run / 'protocol.json')
    if (read_json(run / 'error.json')['message'] != 'Selected-bank VAL disagrees with staged CPU prediction selection'
            or len(list((run / 'tree_models').glob('*.json'))) != 18):
        raise RuntimeError('Unexpected failure or incomplete tree fitting')
    # Retain every original wrong-path prediction, receipt and provisional state.
    before = protected(root)
    original_val = run / 'evaluation/val/selected_best'
    original_val.rename(folder / 'original_recomputed_VAL')
    shutil.copyfile(run / 'status.json', folder / 'original_failed_status.json')
    receipt = dict(state='recovering', started_utc=utc(), protected_before=before,
        automatic_retry=False, optimizer_updates=0, fitted_trees=0, CPU_only=True,
        model_checkpoint_replaced=False, inference_rule_changed=False,
        change='known split evaluation reuses canonical input-only feature cache; original raw-input LIB forward unchanged',
        original_failure_preserved=True, source_sha256=file_hash(Path(__file__)))
    receipt['observed_mismatch'] = dict(problem='TSP', instances=[458, 1770, 3099],
        staged_native_correct=2080, recomputed_native_correct=2080,
        staged_mean_cost=8.026598762905289, recomputed_mean_cost=8.026721756430689,
        interpretation='small FP32 input-statistic differences can cross tree thresholds; not changed costs, labels or weights')
    write_json(receipt_path, receipt)
    args = runner.parser().parse_args(['train', '--arm', 'C', '--root', str(root),
        '--cache-root', str(SHARED_CACHE), '--device', 'cpu', '--wandb-mode', 'offline'])
    torch.set_num_threads(2)
    args.experiment = 'R92'
    tracker = wandb_run(args, config)
    try:
        def forbidden(*args, **kwargs):
            raise AssertionError('No CatBoost fitting is allowed in C evaluation recovery')
        import catboost
        with patch.object(catboost.CatBoostClassifier, 'fit', side_effect=forbidden), \
                patch.object(catboost.CatBoostRegressor, 'fit', side_effect=forbidden), \
                patch.object(tree, 'features', side_effect=AssertionError('No feature re-encoding during split evaluation recovery')), \
                patch.object(runner, 'evaluate', side_effect=cached_evaluate):
            tree._train_bank(args, torch.device('cpu'), config, tracker)
    finally:
        tracker.finish()
    if protected(root) != before:
        raise RuntimeError('Protected checkpoint/tree/original failure evidence changed')
    from . import r92_reporting as reporting
    report_args = reporting.parser().parse_args(['lib', '--root', str(root), '--arm', 'C', '--device', 'cpu'])
    reporting.lib_stage(report_args)
    report_args.command = 'report'
    reporting.report(report_args)
    if protected(root) != before:
        raise RuntimeError('Report changed protected original inputs')
    receipt.update(state='complete', complete=True, completed_utc=utc(), protected_after=protected(root),
        protected_unchanged=True, TEST_LIB_PDF_complete=True,
        selected_best=read_json(run / 'final_test_lock.json'),
        runtime_cache_override=str(SHARED_CACHE), runtime_cache_semantics='same frozen R84 input caches, CPU-readable source path')
    write_json(receipt_path, receipt)
    write_json(run / 'recovery_complete.json', receipt)
    print('C_RECOVERED_WITHOUT_FITTING', receipt['selected_best']['selected_update'], flush=True)
    return receipt


def recovered_watch(root=ROOT, interval=120):
    from . import r92_goal as goal
    root = Path(root).resolve()
    recovery = read_json(root / 'C/recovery_complete.json')
    if not (recovery.get('complete') and recovery.get('protected_unchanged')
            and recovery.get('optimizer_updates') == 0 and recovery.get('fitted_trees') == 0
            and recovery.get('TEST_LIB_PDF_complete')):
        raise RuntimeError('Completion watcher requires the explicit no-fitting C recovery receipt')
    original_status = goal.status

    def status(*args, **kwargs):
        result = original_status(*args, **kwargs)
        ignored = {str(LAUNCH / 'C' / name) for name in ('controller.json', 'worker.json')}
        expected = recovery['protected_after']
        if any(path not in expected or file_hash(Path(path)) != expected[path] for path in ignored):
            raise RuntimeError('Original C failed receipt changed after recovery; do not suppress this failure')
        result['errors'] = [row for row in result['errors'] if row.get('path') not in ignored]
        result['C_recovery'] = str(root / 'C/recovery_complete.json')
        result['original_C_failure_receipts_preserved'] = True
        result['phase'] = 'failed' if result['errors'] else 'comparison_complete' if result['comparison_complete'] else 'waiting_for_arms'
        write_json(root / 'goal_progress.json', result)
        return result

    original_publish = goal.publish

    def publish(directory):
        recovery_note(Path(directory))
        git_dir = Path(subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'],
            cwd=goal.REPO, text=True).strip())
        with (git_dir / 'codex-publication.lock').open('a') as claim:
            fcntl.flock(claim, fcntl.LOCK_EX)
            return original_publish(directory)

    goal.status, goal.publish = status, publish
    return goal.watch(root, LAUNCH, interval, publish_github=True)


def recovery_note(root):
    """Readable companion; never hides the original failed pipeline records."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties
    folder = Path(root) / 'C/recovery'
    receipt = read_json(Path(root) / 'C/recovery_complete.json')
    text = ['# R92C: Evaluation Recovery', '',
        'All18 classifiers were already trained for3000 trees/task. No tree was refitted.',
        'Canonical input statistics and recomputed FP32 statistics changed three TSP validation selections',
        '(indices458,1770,3099). Correct count stayed2080, but mean cost differed.',
        'No tolerance was loosened. Known-split evaluation now reuses the canonical357-column input-only feature cache.',
        'The same previously selected800-tree checkpoint and original protocol were retained.',
        'The formal global validation lock preceded TEST. No labels, costs, Oracle or indices enter prediction features.',
        'Raw-input LIB uses the untouched factory and is a separately disclosed numerical path;',
        'it is not claimed bitwise equivalent to cached split features.', '',
        'VAL/TRAIN were recovered on ADMIN01 CPU. TEST ran CPU-only in the original allocation-owned container',
        'because its existing node cache is under container /tmp. LIB/PDF then ran on ADMIN01 CPU.',
        'The first CPU recovery reached the VAL lock, then failed on that unavailable ADMIN01 TEST-cache path;',
        'the explicit post-lock continuation did not rerun selection or training.',
        'Original failed worker/controller/error records and wrong-path VAL predictions are preserved.',
        f"Protected originals unchanged: {receipt['protected_unchanged']}; fitted trees during recovery:0.",
        'Authoritative completion: C/recovery_complete.json and C/recovery/recovery_completion_receipt.json.',
        'Original launch failure receipts remain failed; the recovery is not a forged original rc0.']
    (folder / 'RECOVERY.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    font = FontProperties(fname='/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
    with PdfPages(folder / 'RECOVERY.pdf') as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.05, .94, 'R92C | Evaluation-only recovery, no retraining', fontsize=18, fontproperties=font)
        for index, line in enumerate([line for line in text[2:] if line]):
            fig.text(.05, .87-index*.037, line, fontsize=9, fontproperties=font)
        pdf.savefig(fig)
        plt.close(fig)
    results = Path(root) / 'RESULTS.md'
    if results.exists() and 'Evaluation Recovery' not in results.read_text():
        with results.open('a', encoding='utf-8') as stream:
            stream.write('\n## R92C Evaluation Recovery\n\nSee C/recovery/RECOVERY.pdf. '
                'Original failed receipts are preserved; no fitting or checkpoint reselection.\n')


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('recover', 'finish-test', 'finish-report', 'note', 'watch'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cache-root', type=Path, default=Path('/tmp/shiys_R84_new90k_cache'))
    parser.add_argument('--interval', type=int, default=120)
    args = parser.parse_args(argv)
    if args.command == 'recover':
        return recover(args.root)
    if args.command == 'finish-test':
        return finish_test(args.root, args.cache_root)
    if args.command == 'finish-report':
        return finish_report(args.root)
    if args.command == 'note':
        return recovery_note(args.root)
    return recovered_watch(args.root, args.interval)


if __name__ == '__main__':
    main()
