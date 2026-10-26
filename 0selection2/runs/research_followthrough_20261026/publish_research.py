"""One exact-path publication, with a durable intent and the global Git mutex.

Run after render_research.py and visual PDF inspection. No model/experiment is
executed. An uncertain transaction is reconciled from existing commits only;
it never automatically commits or pushes a second time.
"""

import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import re
import subprocess

from render_research import ACTION, ROOT, REPO, active, check, read, sha, verify_inputs

MODELS = dict(R109A=('R109', 'A_tied_late'), R109B=('R109', 'B_private_late'),
              R112A=('R112', 'A_collapsed'), R112B=('R112', 'B_interfaces'),
              R113A=('R113', 'A_scratch'), R113B=('R113', 'B_snapshot'),
              R113C=('R113', 'C_transition'))
METRICS = ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct',
           'vs_oracle_pct', 'actual_regret_pct')
TEXT_SUFFIXES = {'.py', '.sh', '.md', '.csv', '.json'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def git(root, *args, allow_failure=False):
    result = subprocess.run(['git', '-C', str(root), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=60)
    if not allow_failure and result.returncode:
        raise RuntimeError('Git failed: ' + ' '.join(args[:3]) + ': ' + result.stderr.decode(errors='replace'))
    return result


def output(root, *args):
    return git(root, *args).stdout.decode().strip()


def expected_bytes(path):
    payload = path.read_bytes()
    return payload.replace(b'\r\n', b'\n') if path.suffix in TEXT_SUFFIXES else payload


def assert_clean_index(root):
    staged = git(root, 'diff', '--cached', '--name-only', '-z').stdout
    check(not staged, 'Unrelated staged changes: main must reconcile the index; nothing was staged/reset')
    check(not git(root, 'ls-files', '-u', '-z').stdout, 'Unmerged Git index requires main reconciliation')


def close_enough(actual, expected, digits=6):
    check(abs(float(actual) - float(expected)) <= 0.51 * 10 ** -digits,
          'Research/evidence numeric contradiction: ' + str(actual) + ' versus ' + str(expected))


def scientific_tables(request):
    context = request['context']
    evidence = Path(context['evidence_dir'])
    records = list(csv.DictReader(io.StringIO((evidence / 'metrics.csv').read_text())))
    check(len(records) == 280, 'Expected 280 fixed-model metric rows')
    indexed = {(r['experiment'], r['arm'], r['split'], r['problem']): r for r in records}
    check(len(indexed) == 280, 'Duplicate evidence metric rows')
    auth = read(evidence / 'saved_results_authentication.json')
    nss = {(r['current_model'], r['dataset']): r for r in auth['nss_comparison']}
    text = (Path(context['research_dir']) / 'REPORT.md').read_text()
    all_rows = problem_rows = lib_rows = difference_rows = 0
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        cells = [x.strip().replace('−', '-').replace('**', '') for x in line.strip('|').split('|')]
        if len(cells) == 9 and cells[0] in MODELS and cells[1] in ('VAL', 'TEST'):
            key = (*MODELS[cells[0]], cells[1], 'ALL')
            for value, metric in zip(cells[2:], METRICS):
                close_enough(value, indexed[key][metric])
            all_rows += 1
        if len(cells) == 8 and (*MODELS['R109A'], 'TEST', cells[0]) in indexed:
            for value, metric in zip(cells[1:], METRICS):
                close_enough(value, indexed[(*MODELS['R109A'], 'TEST', cells[0])][metric])
            problem_rows += 1
        if len(cells) == 5 and cells[0] in MODELS and all('/' in c for c in cells[1:]):
            for cell, dataset in zip(cells[1:], ('TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB')):
                row = nss[('/'.join(MODELS[cells[0]]), dataset)]
                top1, cost = cell.split('/')
                close_enough(top1, row['current_top1_native_pct'], 4)
                close_enough(cost, row['current_mean_cost'])
                lib_rows += 1
        if len(cells) == 4 and (*MODELS['R109A'], 'TEST', cells[0]) in indexed:
            for cell, (a, b) in zip(cells[1:], (('R109B', 'R109A'), ('R112B', 'R112A'), ('R113C', 'R113B'))):
                ra, rb = (indexed[(*MODELS[m], 'TEST', cells[0])] for m in (a, b))
                for value, metric in zip(cell.split('/'), ('top1_pct', 'actual_regret_pct')):
                    close_enough(value, float(ra[metric]) - float(rb[metric]))
                difference_rows += 1
    check((all_rows, problem_rows, lib_rows, difference_rows) == (14, 18, 28, 54),
          'Missing or inconsistent research metric tables')
    check(all(r['sbs_fit_split'] == 'TRAIN' and 'FP64' in r['winner_rule'] for r in records),
          'Native winner/SBS definitions differ from the research report')
    return dict(all_model_split_rows=all_rows, anchor_problem_rows=problem_rows,
                current_model_nss_pairs=lib_rows, per_problem_contrasts=difference_rows,
                native_winner='locked native int64 ind verified against original FP64 argmin',
                sbs_fit='full frozen TRAIN', all_aggregation='equal weight across 18 problems',
                r114_phase=auth['r114_current_execution_phase'], no_scientific_contradictions=True)


def curated_paths(folder):
    paths = [folder / name for name in ('REPORT.md', 'REPORT.pdf', 'sources.csv', 'directions.csv', 'DELIVERY.md')]
    paths += [ROOT / name for name in (
        'PLAN.md', 'monitor_plan.json', 'queue_handoff.json', 'response.schema.json',
        'evidence_csv_reconciliation.json', 'render_research.py', 'publish_research.py',
        'test_publish_research.py', 'render_notes.json', 'prepublication_checks.json')]
    paths += [ROOT / 'prompts' / (name + '.md') for name in ('common', 'execution', 'verify', 'research', 'publish')]
    for subdir, names in (
        ('evidence', ('evidence_pack.md', 'metrics.csv', 'checkpoints.csv', 'source_snapshot.json',
                      'verification.json', 'publication_byte_verification.json', 'saved_results_authentication.json',
                      'pack_gate_check.json', 'authenticate_saved_results.py', 'write_evidence_pack.py')),
        ('evidence_csv_materialization', ('evidence_pack.md', 'metrics.csv', 'checkpoints.csv',
                                        'source_snapshot.json', 'verification.json', 'publication_byte_verification.json',
                                        'saved_results_authentication.json', 'publication_csv_materialization.json')),
    ):
        paths += [ROOT / subdir / name for name in names]
    paths += [REPO / 'code/V4' / name for name in (
        'research_completion_monitor.py', 'test_research_completion_monitor.py',
        'run_research_completion_monitor.sh', 'reconcile_publication_csv.py')]
    check(len(set(paths)) == len(paths), 'Duplicate publication path')
    check(all(p.is_file() and not p.is_symlink() for p in paths), 'Missing/non-regular curated file')
    return paths


def remote_head(root, remote, ref):
    rows = output(root, 'ls-remote', '--exit-code', remote, ref).splitlines()
    check(len(rows) == 1 and rows[0].split()[1] == ref, 'Remote reference is ambiguous')
    return rows[0].split()[0]


def committed_hashes(root, commit, paths):
    result = {}
    for path in paths:
        relative = str(path.relative_to(root))
        result[str(path)] = sha(git(root, 'cat-file', 'blob', commit + ':' + relative).stdout)
    return result


def reconcile_attempt(root, attempt, remote, ref):
    candidates = [attempt['commit']] if attempt.get('commit') else output(
        root, 'rev-list', '--all', '--max-count=30').splitlines()
    matches = []
    for commit in candidates:
        parents = output(root, 'rev-list', '--parents', '-n', '1', commit).split()[1:]
        if parents != [attempt['parent']]:
            continue
        changed = output(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', commit).splitlines()
        if sorted(changed) != attempt['changed_paths']:
            continue
        try:
            actual = committed_hashes(root, commit, [Path(p) for p in attempt['expected_hashes']])
        except RuntimeError:
            continue
        if actual == attempt['expected_hashes']:
            matches.append(commit)
    check(len(matches) == 1,
          'Uncertain prior publication: main must reconcile the saved attempt/index; no duplicate commit/push')
    commit = matches[0]
    check(remote_head(root, remote, ref) == commit,
          'Prior commit exists but remote confirmation is absent; main must reconcile the saved push attempt')
    return commit


def main():
    root = Path(output(REPO, 'rev-parse', '--show-toplevel')).resolve()
    git_dir = Path(output(REPO, 'rev-parse', '--absolute-git-dir')).resolve()
    mutex = git_dir / 'codex-publication.lock'
    check(str(mutex) == '/public/home/shiys/.git/codex-publication.lock', 'Unexpected global Git mutex')
    with mutex.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active()
        request, inputs = verify_inputs()
        folder = Path(request['context']['research_dir'])
        attempt_path = folder / 'publication_attempt.json'
        receipt_path = folder / 'publication_receipt.json'
        branch = output(root, 'symbolic-ref', '--short', 'HEAD')
        check(branch == 'main', 'Only the actual authorized main branch may be published')
        remote = output(root, 'config', '--get', 'branch.' + branch + '.remote')
        ref = output(root, 'config', '--get', 'branch.' + branch + '.merge')
        check((remote, ref) == ('origin', 'refs/heads/main'), 'Unexpected publication upstream')

        if attempt_path.exists():
            # Reconcile an interrupted commit/push without issuing either again.
            attempt = read(attempt_path)
            commit = reconcile_attempt(root, attempt, remote, ref)
        else:
            assert_clean_index(root)
            check(not output(root, 'log', '--all', '-1', '--format=%H', '--', str(folder.relative_to(root))),
                  'Research path already has Git history without an attempt receipt: reconcile before publication')
            notes = read(ROOT / 'render_notes.json')
            check(sha((folder / 'REPORT.pdf').read_bytes()) == notes['pdf']['sha256'], 'Rendered PDF changed')
            checks = read(ROOT / 'prepublication_checks.json')
            check(checks.get('complete') is True and checks.get('visual_inspection', {}).get('complete') is True,
                  'PDF visual inspection and prepublication checks are required')
            table_checks = scientific_tables(request)
            paths = curated_paths(folder)
            check(all(sha(p.read_bytes()) == checks['curated_input_sha256'][str(p)] for p in paths
                      if p.name != 'prepublication_checks.json'), 'Curated input changed after final inspection')
            parent = output(root, 'rev-parse', 'HEAD')
            before = remote_head(root, remote, ref)
            check(before == parent, 'Local main and actual remote differ; main must reconcile without a blind commit/push')
            manifest = dict(version=folder.name, action_id=ACTION, complete_version=True,
                            scientific_checks=table_checks, inputs=inputs,
                            files=[dict(path=str(p.relative_to(root)), source_sha256=sha(p.read_bytes()),
                                        committed_sha256=sha(expected_bytes(p)),
                                        transform='CRLF-to-LF only' if p.read_bytes() != expected_bytes(p) else 'identity')
                                   for p in paths],
                            source_evidence_unchanged=True, runtime_actions_excluded=True,
                            manifest_self_excluded=True, whole18_goal_remains_active=True)
            manifest_path = folder / 'publication_manifest.json'
            save(manifest_path, manifest)
            paths.append(manifest_path)
            expected = {str(p): sha(expected_bytes(p)) for p in paths}
            changed = []
            for p in paths:
                old = git(root, 'cat-file', 'blob', parent + ':' + str(p.relative_to(root)), allow_failure=True)
                if old.returncode or sha(old.stdout) != expected[str(p)]:
                    changed.append(str(p.relative_to(root)))
            check(changed, 'No completed version to publish')
            attempt = dict(state='prepared', action_id=ACTION, model='gpt-6.1-sol', reasoning_effort='max',
                           started_utc=utc(), parent=parent, remote=remote, ref=ref, remote_before=before,
                           global_mutex=str(mutex), expected_hashes=expected, changed_paths=sorted(changed),
                           manifest_sha256=sha(expected_bytes(manifest_path)), commit_attempts=0,
                           push_attempts=0, automatic_retry=False)
            save(attempt_path, attempt)  # Durable write-ahead record BEFORE staging.
            active()
            assert_clean_index(root)
            git(root, 'add', '--force', '--', *[str(p.relative_to(root)) for p in paths])
            staged = sorted(x.decode() for x in git(root, 'diff', '--cached', '--name-only', '-z').stdout.split(b'\0') if x)
            check(staged == sorted(changed), 'Index differs from the exact curated paths: preserve it for main')
            check(committed_hashes(root, '', paths) == expected, 'Staged Git blob differs from normalized manifest')
            check(output(root, 'rev-parse', 'HEAD') == parent, 'HEAD changed during publication')
            active()
            attempt.update(state='commit_started', commit_attempts=1)
            save(attempt_path, attempt)
            git(root, 'commit', '-m', 'Publish completed post-paradigm research report (2026-10-26)')
            commit = output(root, 'rev-parse', 'HEAD')
            attempt.update(state='committed', commit=commit)
            save(attempt_path, attempt)
            check(output(root, 'rev-parse', commit + '^') == parent, 'Unexpected publication commit parent')
            actual_changed = output(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', commit).splitlines()
            check(sorted(actual_changed) == sorted(changed), 'Commit includes unexpected changes; stop before push')
            check(committed_hashes(root, commit, paths) == expected, 'Committed artifact identity mismatch')
            active()
            attempt.update(state='push_started', push_attempts=1)
            save(attempt_path, attempt)
            git(root, 'push', remote, commit + ':' + ref)

        after = remote_head(root, remote, ref)
        check(after == commit, 'Push outcome is uncertain: preserve attempt and reconcile actual remote')
        all_hashes = committed_hashes(root, commit, [Path(p) for p in attempt['expected_hashes']])
        check(all_hashes == attempt['expected_hashes'], 'Actual commit hashes differ from publication intent')
        core = [folder / name for name in ('REPORT.md', 'REPORT.pdf', 'sources.csv', 'directions.csv')]
        notes = read(ROOT / 'render_notes.json')
        receipt = dict(state='complete', exit_status=0, action_id=ACTION, commit=commit,
                       model='gpt-6.1-sol', reasoning_effort='max', global_mutex=str(mutex),
                       completed_utc=utc(), published_hashes={str(p): all_hashes[str(p)] for p in core},
                       committed_artifact_sha256=all_hashes,
                       git_blob_ids={str(p): output(root, 'rev-parse', commit + ':' + str(p.relative_to(root))) for p in core},
                       source_snapshot_sha256=request['context']['source_snapshot_sha256'],
                       source_snapshot_file_sha256=inputs['source_snapshot_file_sha256'],
                       pdf_metadata=notes, publication_manifest_sha256=attempt['manifest_sha256'],
                       remote_confirmation=dict(remote=remote, ref=ref, advertised_head=after,
                                                commit_received=True, exact_artifact_hashes_verified=True,
                                                method='Independent git ls-remote; exact advertised commit tree/blob SHA256'),
                       provider_route_verified=False,
                       model_identity_evidence='Exact requested CLI -m/max record and live ancestry; not independent provider-side routing proof',
                       commit_attempts=attempt['commit_attempts'], push_attempts=attempt['push_attempts'],
                       source_evidence_unchanged=True, scientific_text_changed=False,
                       unrelated_changes_staged=False, whole18_goal_remains_active=True)
        save(receipt_path, receipt)  # Success receipt is written only AFTER remote confirmation.
        attempt.update(state='complete', commit=commit, remote_after=after, completed_utc=utc())
        save(attempt_path, attempt)
        print(json.dumps(dict(state='complete', commit=commit, artifacts=len(all_hashes), receipt=str(receipt_path))))


if __name__ == '__main__':
    main()
