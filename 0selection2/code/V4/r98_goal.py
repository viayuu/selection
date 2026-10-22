"""Quiet CPU end-of-version gate and opt-in, one-shot R98/R99/R100 publication.

Run after activating easynco, e.g. ``python -B -m code.V4.r98_goal
--version R98 --watch --publish-github``. Main starts watchers, not imports.
Only saved JSON/PDF receipts are inspected. R99 may run its saved-data CPU
``compare`` once after both complete arm reports; no model/report repair,
training, inference, launch or failed-stage retry is permitted here.

Git uses the same <absolute-git-dir>/codex-publication.lock as r97_goal and
r94_goal. Only the completed version's curated assets are staged explicitly.
Pipeline/publication completion NEVER marks the research goal complete.
"""

import argparse
import ast
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .r80_reporting import file_hash, json_hash
from .r93_goal import PROBLEMS, _sensitive_json, cpu_environment, optional, save, utc


REPO = Path(__file__).resolve().parents[2]
LAUNCH = REPO / 'code/V4/runs/R98_parallel_dispatch'
VERSIONS = dict(R98='R98_performance_context', R99='R99_global_competition_input',
                R100='R100_pairwise_context')
FROZEN_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'target_recovery.pdf')
R99_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'target_recovery_and_train.pdf',
            'learning_curves.pdf', 'training_and_method_diagnostics.pdf')
REFERENCE_FLAGS = ('nss_root', 'historical_root', 'r89_root')
MAX_ASSET_BYTES = 32 * 1024**2
PRUNED = {'python', 'wheels', 'weights', 'wandb', 'evaluation', 'predictions', 'outcomes', 'outcome',
          'input_cache', 'feature_cache', 'features', 'checkpoints', '__pycache__',
          'site-packages', 'by_problem', 'logs'}


def model_root(version):
    return REPO / 'code/V4/runs' / VERSIONS[version]


def arms(version):
    return ('A', 'B') if version == 'R99' else ('A',)


def run_root(version, root, arm):
    return root / arm if version == 'R99' else root


def pdf_ready(path):
    if not Path(path).is_file() or Path(path).stat().st_size < 5:
        return False
    with Path(path).open('rb') as stream:
        return stream.read(5) == b'%PDF-'


def receipt_errors(path, record):
    errors = []
    if not isinstance(record, dict):
        return [dict(path=str(path), error='Expected a JSON receipt object')]
    phases = [str(record.get(key, '')).lower() for key in ('state', 'phase')]
    rc = record.get('exit_status', record.get('returncode', record.get('exit_code')))
    if any('failed' in phase or phase in ('error', 'stopped_on_error') for phase in phases) or rc not in (None, 0):
        errors.append(dict(path=str(path), error=record.get('error', f'phase={phases}; rc={rc}')))
    for child in record.get('stages', []):
        errors.extend(receipt_errors(path, child))
    return errors


def rc0(record):
    return (record.get('state') == 'complete' and record.get('exit_status') == 0
            and not isinstance(record.get('exit_status'), bool))


def stages_ready(record, expected):
    stages = record.get('stages', [])
    names = [stage.get('name') for stage in stages]
    return (rc0(record) and bool(expected) and len(names) == len(set(names))
            and set(names) == set(expected) and all(rc0(stage) for stage in stages)
            and not receipt_errors('<stages>', record))


def same_state(record, lock):
    identity = record.get('identity', record)
    return (identity.get('state_hash') == lock.get('state_hash')
            and identity.get('checkpoint_sha256', lock.get('checkpoint_sha256')) == lock.get('checkpoint_sha256'))


def split_ready(result, rows, lock):
    per_problem = result.get('per_problem', {})
    return (isinstance(per_problem, dict) and set(per_problem) == set(PROBLEMS)
            and all(isinstance(row, dict) and row.get('n') == rows for row in per_problem.values())
            and isinstance(result.get('macro'), dict) and bool(result['macro']) and same_state(result, lock))


def native_rule(config):
    rule = str(config.get('winner_rule', '')).lower()
    return 'fp64' in rule and ('native' in rule or 'ind' in rule)


def common_checks(version, arm, run, records):
    config, lock = records['protocol.json'], records['final_test_lock.json']
    return dict(protocol=config.get('experiment') == version and config.get('arm') == arm
        and config.get('seed') == 2 and native_rule(config)
        and config.get('rows_per_problem') == dict(train=81000, val=4500, test=4500)
        and set(config.get('candidate_pools', {})) == set(PROBLEMS)
        and set(config.get('baselines', {})) == set(PROBLEMS)
        and all(b.get('selection_split') == 'train' and b.get('train_rows') == 81000
                for b in config.get('baselines', {}).values()),
        unique_VAL_lock=bool(lock.get('checkpoint_sha256'))
        and config.get('experiment') == version and lock.get('experiment') == version
        and lock.get('arm') == arm and lock.get('state_hash') == json_hash(config)
        and lock.get('selected_without_test') is True and lock.get('per_problem_assembly') is False)


def frozen_checks(version, arm, run, records):
    checks = common_checks(version, arm, run, records)
    config, lock, bank, post, complete = (records[key] for key in
        ('protocol.json', 'final_test_lock.json', 'checkpoint.json', 'postprocess_status.json', 'run_complete.json'))
    checkpoint = run / 'checkpoint.json'
    keys = ('contexts', 'targets') if version == 'R98' else ('contexts', 'relations', 'source_sha256')
    checks['frozen_bank'] = (checkpoint.is_file() and file_hash(checkpoint) == lock.get('checkpoint_sha256')
        and bank.get('protocol_sha256') == lock.get('state_hash') and bank.get('weights_updated') is False
        and bool(config.get('inference_rule')) and bank.get('inference_rule') == config.get('inference_rule')
        and bank.get('public_checkpoint_sha256') == config.get('public_checkpoint_sha256')
        and bank.get('selection_rule') == lock.get('selection_rule') == config.get('selection_rule')
        and bool(bank.get('selection_rule')) and all(key in config and bank.get(key) == config[key] for key in keys)
        and config.get('optimizer_updates') == 0 and config.get('fine_tuning') is False
        and config.get('current_query_labels_in_forward') is False
        and all(set(config.get(key, {})) == set(PROBLEMS) for key in keys[:2])
        and all(c.get('rows') == 81000 for c in config.get('contexts', {}).values()))
    checks['run_complete'] = (complete.get('complete') is True and complete.get('full18_VAL') is True
        and complete.get('full18_TEST') is True and set(complete.get('LIB', [])) == {'TSPLIB', 'CVRPLIB'}
        and complete.get('checkpoint_sha256') == lock.get('checkpoint_sha256'))
    for split in ('val', 'test'):
        result = records[split + '_results.json']
        checks['full_' + split.upper()] = result.get('complete') is True and split_ready(result, 4500, lock)
    bound = records['test_input_lock.json']
    checks['TEST_input_lock'] = (bound.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and bound.get('labels_loaded') is False and bound.get('created_after_unique_VAL_lock') is True
        and set(bound.get('inputs', {})) == set(PROBLEMS))
    libs = records['lib_results.json']
    checks['LIB'] = all(libs.get(dataset, {}).get('complete') is True
        and libs[dataset].get('problem') == problem and same_state(libs[dataset], lock)
        for dataset, problem in (('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')))
    provenance = records['full_model_statistics/provenance.json']
    checks['CPU_report'] = (post.get('complete') is True and post.get('phase') == 'complete'
        and post.get('exit_status', 0) == 0 and same_state(post, lock)
        and post.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and provenance.get('experiment') == version and same_state(provenance, lock)
        and provenance.get('checkpoint_sha256') == lock.get('checkpoint_sha256'))
    return checks


def r99_checks(arm, run, records):
    checks = common_checks('R99', arm, run, records)
    config, lock, train, final, test, lib, post, report = (records[key] for key in
        ('protocol.json', 'final_test_lock.json', 'training_complete.json', 'final_metrics.json',
         'test_results.json', 'lib_stage_complete.json', 'postprocess_status.json', 'full_model_statistics/protocol.json'))
    budget, selected = config.get('max_successful_updates'), lock.get('selected_update')
    checks['training_complete'] = (budget == 57600 and train.get('successful_updates') == budget
        and train.get('state_hash') == lock.get('state_hash')
        and all(train.get('updates_by_problem', {}).get(p) == 3200 for p in PROBLEMS))
    checks['unique_VAL_lock'] &= (isinstance(selected, int) and not isinstance(selected, bool)
        and 0 < selected <= 57600 and bool(config.get('inference_rule'))
        and lock.get('inference_rule') == config.get('inference_rule')
        and lock.get('rule') == config.get('checkpoint_rule') and 'macro_vs_sbs' in str(lock.get('rule', '')))
    checks['full_VAL'] = (final.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and same_state(final, lock) and final.get('selected_update') == selected
        and isinstance(final.get('val'), dict) and split_ready(final['val'], 4500, lock))
    result = test.get('result', {})
    checks['full_TEST'] = (test.get('complete') is True and same_state(test, lock)
        and test.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and isinstance(result, dict) and split_ready(result, 4500, lock))
    checks['LIB'] = (lib.get('complete') is True and lib.get('exit_status') == 0 and same_state(lib, lock)
        and lib.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and set(lib.get('datasets', [])) == {'TSPLIB', 'CVRPLIB'})
    checks['CPU_report'] = (post.get('complete') is True and post.get('phase') == 'complete'
        and post.get('exit_status') == 0 and report.get('complete') is True
        and report.get('experiment') == 'R99' and report.get('checkpoint_lock') == lock)
    timing = records['inference_timing.json']
    path = run / 'inference_timing.json'
    checks['timing_bound_to_report'] = (path.is_file() and timing.get('complete') is True
        and same_state(timing, lock) and timing.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and report.get('inference_timing', {}).get('sha256') == file_hash(path))
    return checks


def comparison_available():
    path = REPO / 'code/V4/r99_reporting.py'
    return path.is_file() and any(isinstance(node, ast.FunctionDef) and node.name == 'compare'
        for node in ast.parse(path.read_text(encoding='utf-8')).body)


def status(version='R98', root=None, launch_root=LAUNCH, *, persist=True):
    root, launch_root = Path(root or model_root(version)).resolve(), Path(launch_root).resolve()
    names = ('protocol.json', 'status.json', 'checkpoint.json', 'final_test_lock.json', 'run_complete.json',
        'val_results.json', 'test_results.json', 'test_input_lock.json', 'lib_results.json',
        'training_complete.json', 'final_metrics.json', 'lib_stage_complete.json', 'lib_stage_status.json',
        'inference_timing.json', 'postprocess_status.json', 'full_model_statistics/protocol.json',
        'full_model_statistics/provenance.json')
    rows, errors, locks = [], [], {}
    for arm in arms(version):
        run, run_id = run_root(version, root, arm), version + '_' + arm
        records = {name: optional(run / name) for name in names}
        launch = launch_root / run_id
        controller, worker = optional(launch / 'controller.json'), optional(launch / 'worker.json')
        spec = optional(launch_root / 'specs' / (run_id + '.json'))
        checks = r99_checks(arm, run, records) if version == 'R99' else frozen_checks(version, arm, run, records)
        gpu_stages = [stage['name'] for stage in spec.get('stages', []) if stage.get('device') == 'gpu']
        cpu_stages = [stage['name'] for stage in spec.get('stages', []) if stage.get('device') == 'cpu']
        checks.update(launch_spec=spec.get('run_id') == run_id and spec.get('run_dir') == str(run)
            and bool(gpu_stages) and bool(cpu_stages),
            controller_rc0=stages_ready(controller, ['allocation_GPU_pipeline', *cpu_stages]),
            worker_all_GPU_stages_rc0=stages_ready(worker, gpu_stages))
        for name in R99_PDFS if version == 'R99' else FROZEN_PDFS:
            checks[name] = pdf_ready(run / 'full_model_statistics' / name)
        if version != 'R99':
            checks['RESULTS.pdf'] = pdf_ready(run / 'RESULTS.pdf')
        for name, record in records.items():
            errors.extend(receipt_errors(run / name, record))
        for name in ('controller.json', 'worker.json', 'queue_status.json'):
            errors.extend(receipt_errors(launch / name, optional(launch / name)))
        locks[arm] = records['final_test_lock.json']
        rows.append(dict(arm=arm, run_id=run_id, root=str(run), checks=checks,
            missing=[key for key, value in checks.items() if not value], ready=all(checks.values())))
    for name in ('goal_watch_receipt.json', 'goal_compare_receipt.json', 'github_publication.json'):
        errors.extend(receipt_errors(root / name, optional(root / name)))
    ready = all(row['ready'] for row in rows)
    comparison = optional(root / 'comparison_artifacts/protocol.json') if version == 'R99' else {}
    compared = version != 'R99' or (comparison.get('experiment') == 'R99' and comparison.get('complete') is True
        and comparison.get('exit_status') == 0 and comparison.get('checkpoints') == locks
        and pdf_ready(root / 'comparison_artifacts/COMPARISON.pdf') and pdf_ready(root / 'RESULTS.pdf'))
    complete = ready and compared and not errors
    result = dict(experiment=version, arms=rows, errors=errors, all_arms_ready=ready,
        comparison_complete=compared, complete=complete,
        phase='failed' if errors else 'complete' if complete else 'waiting_for_comparison' if ready else 'waiting_for_arms',
        cpu_only=True, automatic_retry=False, goal_complete=False, goal_status='active_unresolved',
        note='Saved completion evidence only; no live GPU or research-success claim.')
    if persist:
        path = root / 'goal_progress.json'
        prior = optional(path)
        if {k: v for k, v in prior.items() if k != 'updated_utc'} == result:
            return prior
        result['updated_utc'] = utc()
        save(path, result)
    return result


def compare_once(root, launch_root=LAUNCH, **references):
    root = Path(root).resolve()
    current = status('R99', root, launch_root)
    if current['errors'] or not current['all_arms_ready'] or current['comparison_complete']:
        raise RuntimeError('CPU compare requires both complete R99 arm pipelines, without a comparison or failures')
    if not comparison_available():
        raise RuntimeError('R99 CPU compare is not implemented; leave the version unresolved')
    path = root / 'goal_compare_receipt.json'
    with (root / 'goal_compare.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise RuntimeError('Comparison already attempted; preserve receipt, no automatic retry')
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r99_reporting', 'compare',
                '--root', str(root), '--device', 'cpu']
        for key in REFERENCE_FLAGS:
            if references.get(key) is not None:
                argv.extend(['--' + key.replace('_', '-'), str(Path(references[key]).resolve())])
        receipt = dict(state='running', started_utc=utc(), argv=argv, cpu_only=True, automatic_retry=False)
        save(path, receipt)
        try:
            with (root / 'goal_compare.log').open('a', encoding='utf-8') as log:
                process = subprocess.run(argv, cwd=REPO, env=cpu_environment(),
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            if process.returncode or not status('R99', root, launch_root)['complete']:
                raise RuntimeError('CPU comparison failed or lacks same-lock PDF outputs; no retry')
            receipt.update(state='complete', exit_status=0, completed_utc=utc())
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=utc())
            raise
        finally:
            save(path, receipt)


def _safe_json(record):
    if _sensitive_json(record):
        return False
    if isinstance(record, dict):
        array_keys = {'costs', 'winner', 'pred', 'logits', 'expected_gaps', 'probabilities', 'g', 'x', 'y'}
        return all(str(key).lower() not in {'token', 'authorization'}
                   and not (str(key).lower() in array_keys and isinstance(value, list)) and _safe_json(value)
                   for key, value in record.items())
    if isinstance(record, list):
        return all(_safe_json(value) for value in record)
    return not isinstance(record, str) or '--token' not in record


def publication_paths(version, root, launch_root, git_root):
    """Metadata/report allowlist, no dependency closure or other-version glob."""
    root, launch_root, git_root = (Path(p).absolute() for p in (root, launch_root, git_root))
    if root != model_root(version).absolute() or launch_root != LAUNCH.absolute():
        raise RuntimeError('Publication accepts only canonical version and dispatch roots')
    code = REPO / 'code/V4'
    prefix = version.lower()
    candidates = [p for pattern in (prefix + '_*.py', 'test_' + prefix + '*.py', 'run_v4_' + prefix + '*.sh')
                  for p in code.glob(pattern)]
    candidates += [code / name for name in ('r98_goal.py', 'test_r98_goal.py', 'r98_launch.py',
                                            'test_r98_launch.py', 'run_v4_r98_launch.sh')]
    metadata_roots = [root, *([root / arm for arm in arms(version)] if version == 'R99' else [])]
    for run in metadata_roots:
        candidates += [p for p in run.glob('*') if p.suffix.lower() in {'.json', '.md', '.pdf', '.csv'}
                       and not p.name.startswith(('goal_', 'github_', 'prediction', 'outcome', 'wandb'))]
        for directory in ('preparation', 'prep', 'full_model_statistics', 'comparison_artifacts'):
            for parent, directories, files in os.walk(run / directory):
                directories[:] = [name for name in directories if name not in PRUNED and not (Path(parent) / name).is_symlink()]
                candidates += [Path(parent) / name for name in files if Path(name).suffix.lower() in {'.json', '.md', '.pdf', '.csv'}]
        candidates += [run / 'runtime' / name for name in
                       ('runtime_metadata.json', 'provenance.json', 'official_checkpoint_api.json')]
        config = optional(run / 'protocol.json') or optional(run / 'protocol_lock.json')
        banks = [run / name for name in ('context', 'contexts', 'targets', 'relations')]
        banks += [Path(config[key]) for key in ('context_root', 'target_root', 'relation_root', 'relations_root')
                  if isinstance(config.get(key), str)]
        for bank in banks:
            for problem in PROBLEMS:
                candidates.append(bank / problem / 'identity.json')
    for arm in arms(version):
        run_id = version + '_' + arm
        candidates.append(launch_root / 'specs' / (run_id + '.json'))
        candidates += [launch_root / run_id / name for name in
                       ('plan.json', 'ready.json', 'controller.json', 'worker.json')]
    result = set()
    for path in candidates:
        if (not path.is_file() or path.is_symlink() or path.name.lower().startswith('wandb')
                or any(parent.is_symlink() for parent in path.parents if parent != git_root)):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(REPO.resolve()) or not resolved.is_relative_to(git_root.resolve()):
            raise RuntimeError('Curated asset escapes repository: ' + str(path))
        if path.stat().st_size > MAX_ASSET_BYTES:
            raise RuntimeError('Curated asset unexpectedly exceeds 32MiB: ' + str(path))
        if path.suffix.lower() == '.json' and not _safe_json(optional(path)):
            continue
        result.add(str(resolved.relative_to(git_root.resolve())))
    if not result:
        raise RuntimeError('No curated assets to publish')
    return sorted(result)


def git_output(argv, cwd):
    return subprocess.check_output(['git', *argv], cwd=cwd, env=cpu_environment(), text=True).strip()


def publish(version='R98', root=None, launch_root=LAUNCH):
    root, launch_root = Path(root or model_root(version)).resolve(), Path(launch_root).resolve()
    if root != model_root(version).resolve() or launch_root != LAUNCH.resolve():
        raise RuntimeError('Publication requires canonical roots')
    path = root / 'github_publication.json'
    if path.exists():
        prior = optional(path)
        if prior.get('complete') is True:
            return prior
        raise RuntimeError('Publication already attempted; no staging/commit/push retry')
    if not status(version, root, launch_root)['complete']:
        raise RuntimeError('Publish requires the complete whole version, not an individual arm')
    git_root = Path(git_output(['rev-parse', '--show-toplevel'], REPO)).resolve()
    git_dir = Path(git_output(['rev-parse', '--absolute-git-dir'], REPO)).resolve()
    with (git_dir / 'codex-publication.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        if path.exists():
            prior = optional(path)
            if prior.get('complete') is True:
                return prior
            raise RuntimeError('Publication already attempted; preserve its receipt')
        current = status(version, root, launch_root)
        if not current['complete']:
            raise RuntimeError('Completion gate changed while waiting for the shared Git mutex')
        if git_output(['branch', '--show-current'], git_root) != 'main':
            raise RuntimeError('Expected main; branch/index unchanged')
        if git_output(['diff', '--cached', '--name-only', '-z'], git_root):
            raise RuntimeError('Pre-existing staged changes; index left untouched')
        head = git_output(['rev-parse', 'HEAD'], git_root)
        files = publication_paths(version, root, launch_root, git_root)
        receipt = dict(experiment=version, complete=False, state='staging', started_utc=utc(), files=files,
            global_mutex=str(git_dir / 'codex-publication.lock'), remote='origin/main', curated_only=True,
            automatic_retry=False, goal_complete=False, goal_status='active_unresolved')
        save(path, receipt)
        try:
            for start in range(0, len(files), 80):
                subprocess.run(['git', 'add', '-f', '--', *files[start:start+80]], cwd=git_root,
                               env=cpu_environment(), check=True)
            staged = set(filter(None, git_output(['diff', '--cached', '--name-only', '-z'], git_root).split('\0')))
            if (not staged or not staged.issubset(set(files))
                    or git_output(['branch', '--show-current'], git_root) != 'main'
                    or git_output(['rev-parse', 'HEAD'], git_root) != head):
                raise RuntimeError('Unexpected index/branch/HEAD mutation; no reset or revert performed')
            subprocess.run(['git', 'commit', '-m', version.lower() + ': publish completed full18 version'],
                           cwd=git_root, env=cpu_environment(), check=True)
            receipt.update(state='committed_pending_push', commit=git_output(['rev-parse', 'HEAD'], git_root))
            save(path, receipt)
            subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/main'], cwd=git_root,
                           env=cpu_environment(), check=True)
            receipt.update(complete=True, state='complete', exit_status=0, completed_utc=utc())
            save(path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=utc())
            save(path, receipt)
            raise


def watch(version='R98', root=None, launch_root=LAUNCH, interval=120, publish_github=False, **references):
    if interval <= 0:
        raise ValueError('Watcher interval must be positive')
    root, launch_root = Path(root or model_root(version)).resolve(), Path(launch_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'goal_watch.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / 'goal_watch_receipt.json'
        if path.exists():
            prior = optional(path)
            if prior.get('state') == 'complete' and prior.get('exit_status') == 0:
                return 0
            raise RuntimeError('Watcher already attempted; preserve original receipt, no automatic retry')
        receipt = dict(experiment=version, state='watching', pid=os.getpid(), started_utc=utc(),
            cpu_only=True, github_on_completion=publish_github, automatic_retry=False,
            goal_complete=False, goal_status='active_unresolved')
        save(path, receipt)
        previous = None
        try:
            while True:
                current = status(version, root, launch_root)
                signature = json_hash({k: v for k, v in current.items() if k != 'updated_utc'})
                if signature != previous:
                    print(json.dumps(dict(version=version, phase=current['phase'],
                        missing={row['arm']: row['missing'] for row in current['arms']}, errors=current['errors'])), flush=True)
                    previous = signature
                if current['errors']:
                    raise RuntimeError('Stage failure; preserve receipts and outputs; no retry or model completion')
                if version == 'R99' and current['all_arms_ready'] and not current['comparison_complete']:
                    if not comparison_available():
                        raise RuntimeError('R99 compare unavailable; version remains unresolved')
                    compare_once(root, launch_root, **references)
                    continue
                if current['complete']:
                    if publish_github:
                        publish(version, root, launch_root)
                    receipt.update(state='complete', pipeline_complete=True, exit_status=0, completed_utc=utc())
                    save(path, receipt)
                    return 0
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='stopped_on_error', exit_status=1, error=str(error), completed_utc=utc(),
                           traceback=traceback.format_exc())
            save(path, receipt)
            return 1


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--version', choices=tuple(VERSIONS), default='R98')
    value.add_argument('--watch', action='store_true')
    value.add_argument('--publish-github', action='store_true')
    value.add_argument('--root', type=Path)
    value.add_argument('--launch-root', type=Path, default=LAUNCH)
    value.add_argument('--interval', type=int, default=120)
    for flag in REFERENCE_FLAGS:
        value.add_argument('--' + flag.replace('_', '-'), type=Path)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.interval <= 0:
        raise ValueError('Watcher interval must be positive')
    if args.watch:
        return watch(args.version, args.root, args.launch_root, args.interval, args.publish_github,
                     **{key: getattr(args, key) for key in REFERENCE_FLAGS})
    if args.publish_github:
        publish(args.version, args.root, args.launch_root)
    else:
        print(json.dumps(status(args.version, args.root, args.launch_root), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
