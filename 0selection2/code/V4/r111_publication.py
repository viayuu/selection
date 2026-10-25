"""Read-only R111 publication inventory; never import or execute experiment code.

build prints a proposed manifest, verify checks its small files, and allowlist
prints NUL-delimited Git-root-relative paths. Only main may perform Git writes.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'code/V4/runs/R111_family_conditional_features'
RUN = ROOT / 'A_family_conditional'
MANIFEST = ROOT / 'publication_manifest.json'
SEEDS = tuple('code/V4/' + name for name in (
    'r111_experiment.py', 'r111_model.py', 'r111_reporting.py', 'r111_launch.py',
    'test_r111_model.py', 'test_r111_reporting.py', 'test_r111_launch.py',
    'r111_publication.py')) + tuple('code/NSS_retrain/' + name for name in (
        'utils.py', 'dataset.py', 'model.py', 'loss.py'))
REPORTS = (
    'MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
    'training_and_method_diagnostics.pdf', 'target_recovery_and_train.pdf',
    'RESULTS.md', 'protocol.json', 'metrics_by_problem.csv', 'nss_comparison.csv',
    'historical_R110_summary.csv', 'method_selection_train_val_test.csv',
    'paired_changes_vs_R84A.csv', 'paired_changes_vs_R94B.csv',
    'target_recovery_summary_vs_R84A.csv', 'target_recovery_summary_vs_R94B.csv',
    'target_recovery_vs_R84A.csv', 'target_recovery_vs_R94B.csv',
    'zero_selection_summary.csv', 'selection_distribution.csv',
    'single_method_comparison.csv', 'train_metrics_by_problem.csv')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(['git', '-C', str(WORKSPACE), *args],
        env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'), text=True).strip()


def local_source_closure():
    """Conservative imports plus literal factories and four NSS file imports.

    All function bodies are scanned, including inactive historical helpers.
    Private spec_from_file_location copies refer to already imported sources.
    This inventories source only, not solver corpora, data, caches or weights.
    """
    pending, seen = list(SEEDS), set()

    def module_file(name):
        while name == 'code' or name.startswith('code.'):
            path = WORKSPACE.joinpath(*name.split('.'))
            for candidate in (path.with_suffix('.py'), path / '__init__.py'):
                if candidate.is_file():
                    return candidate.relative_to(WORKSPACE).as_posix()
            name = name.rpartition('.')[0]
        return None

    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        path = WORKSPACE / relative
        package = list(Path(relative).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                base += node.module.split('.') if node.module else []
                names = ['.'.join(base)] + ['.'.join(base + [a.name])
                    for a in node.names if a.name != '*']
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith('code.') and all(
                        part.isidentifier() for part in node.value.split('.')):
                    names = [node.value]
            for name in names:
                dependency = module_file(name)
                if dependency:
                    pending.append(dependency)
        for parent in path.parents:
            if parent == WORKSPACE:
                break
            init = parent / '__init__.py'
            if init.is_file():
                pending.append(init.relative_to(WORKSPACE).as_posix())
    return sorted(WORKSPACE / name for name in seen)


def completion():
    dispatch = ROOT / 'preparation/dispatch_recovery_01'
    records = {name: read(dispatch / (name + '.json'))
               for name in ('queue', 'entry', 'worker')}
    expected = dict(queue=['allocation_pipeline', 'report'], entry=['container_exec'],
                    worker=['train', 'finalize', 'timing', 'lib'])
    for name, record in records.items():
        require(record['state'] == 'complete' and record['exit_status'] == 0,
                name + ' did not complete at rc0')
        require([s['name'] for s in record['stages']] == expected[name],
                name + ' stage order differs')
        require(all(s['state'] == 'complete' and s['exit_status'] == 0
                    for s in record['stages']), name + ' has an incomplete stage')
    final, lock = read(RUN / 'final_metrics.json'), read(RUN / 'final_test_lock.json')
    require(final['complete'] is True and final['selected_update'] == 41760,
            'not the completed R111 selected checkpoint')
    digest = lock['checkpoint_sha256']
    require(final['checkpoint_sha256'] == digest, 'final checkpoint differs')
    for split in ('train', 'val', 'test'):
        value = final[split]
        require(len(value['per_problem']) == 18, split + ' is not full18')
        require(all(p['n'] == (81000 if split == 'train' else 4500)
                    for p in value['per_problem'].values()), split + ' row counts differ')
        require(value['identity']['checkpoint_sha256'] == digest
                and value['identity']['successful_updates'] == lock['selected_update']
                and value['identity']['state_hash'] == lock['state_hash'],
                split + ' identity differs')
    for filename in ('postprocess_status.json', 'lib_stage_complete.json'):
        value = read(RUN / filename)
        require(value['complete'] is True and value['checkpoint_sha256'] == digest,
                filename + ' is not complete for this checkpoint')
    require(read(RUN / 'training_complete.json')['successful_updates'] == 57600,
            'training budget incomplete')
    require(read(RUN / 'test_complete.json')['full_tests'] == 1, 'TEST receipt differs')
    require(read(RUN / 'full_model_statistics/protocol.json')['selected_archives_file_bound'] is True,
            'existing report does not attest selected archive binding')
    return dict(state='completed_rc0', finished_utc=records['queue']['finished_utc'],
        selected_update=lock['selected_update'], checkpoint_sha256=digest,
        state_hash=lock['state_hash'], checkpoint_file_rehashed=False,
        prediction_arrays_read=False,
        evidence_paths=[str((dispatch / (n + '.json')).relative_to(WORKSPACE))
                        for n in ('queue', 'worker', 'entry')])


def artifact_paths():
    paths = [ROOT / n for n in ('RESULTS.md', 'PLAN.md', 'LAUNCH.md',
                               'launch_ready.json', 'launch_startup_check.json')]
    paths += [RUN / n for n in ('protocol.json', 'initialization.json', 'final_metrics.json',
        'final_test_lock.json', 'best_eval.json', 'test_results.json', 'training_complete.json',
        'test_complete.json', 'lib_stage_complete.json', 'postprocess_status.json',
        'inference_timing.json', 'metrics.csv')]
    paths += [RUN / 'full_model_statistics' / n for n in REPORTS]
    paths += [RUN / 'lib_evaluation' / (dataset + suffix)
              for dataset in ('TSPLIB', 'CVRPLIB')
              for suffix in ('_protocol.json', '_results.json')]
    paths += [ROOT / 'preparation/review.md', ROOT / 'preparation/recovery_01/RECOVERY.md',
              ROOT / 'preparation/recovery_01/r111_launch.py',
              ROOT / 'preparation/recovery_01/launch_ready.json']
    for directory in ('dispatch', 'dispatch_recovery_01'):
        paths += [ROOT / 'preparation' / directory / (name + '.json')
                  for name in ('queue', 'worker', 'entry', 'claim')]
    return sorted(paths)


def build():
    done = completion()
    git_root = Path(git('rev-parse', '--show-toplevel'))
    sources = local_source_closure() + [WORKSPACE / 'code/V4/run_v4_r111.sh']
    artifacts = artifact_paths()
    paths = sorted(set(sources + artifacts))
    relative = [p.relative_to(git_root).as_posix() for p in paths]
    tracked = set(git('ls-tree', '-r', '--full-tree', '--name-only', 'HEAD', '--',
                      *map(str, paths)).splitlines())
    changed = set(git('diff', '--no-relative', '--name-only', 'HEAD', '--',
                      *map(str, paths)).splitlines())
    entries = []
    for path, name in zip(paths, relative):
        require(path.is_file() and not path.is_symlink(), 'missing or symlink: ' + name)
        data = path.read_bytes()
        require(len(data) < 1024 * 1024, 'unexpected large publication file: ' + name)
        if path.suffix == '.pdf':
            require(data.startswith(b'%PDF-'), 'not a PDF: ' + name)
        entries.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
            kind='source' if path in sources else 'report_or_evidence',
            in_head=name in tracked, changed_from_head=name in changed))
    frozen = read(ROOT / 'launch_ready.json')['source_sha256']
    current = {str(git_root / e['path']): e['sha256'] for e in entries}
    require(all(current.get(path) == digest for path, digest in frozen.items()),
            'a launch-frozen R111 source/PLAN changed')
    return dict(schema_version=1, experiment='R111', kind='publication_preparation_only',
        prepared_utc=datetime.now(timezone.utc).isoformat(), git_root=str(git_root),
        workspace=str(WORKSPACE), parent_commit=git('rev-parse', 'HEAD'),
        git_lock_for_main=str(git_root / '.git/codex-publication.lock'),
        staged=False, committed=False, pushed=False, completion=done,
        validation_scope='Small saved JSON/CSV/PDF and source hashes only; no model imports, '
            'training, TEST/LIB, timing, replay, array reads or full audit.',
        source_closure_policy='Conservative AST imports in all bodies plus literal local factories, '
            'package initializers, R111 wrapper and four dynamic NSS_retrain files; '
            'historical private module copies use their original source files.',
        files=entries,
        stage_paths=sorted([e['path'] for e in entries if not e['in_head'] or e['changed_from_head']]
                           + [MANIFEST.relative_to(git_root).as_posix()]),
        manifest_self_hash='omitted to avoid a self-reference; verify before main stages',
        r109=dict(status='unfinished_per_main_handoff_not_reaudited', results_included=False,
            source_only_paths=[e['path'] for e in entries if '/r109_' in e['path']
                               or '/test_r109_' in e['path']],
            note='Dependency source only, never R109 completion or a result publication.'),
        historical_results='R94B and both R110 arms are saved same-data historical references, '
            'not newly trained, parameter/FLOP-matched controls; TEST development-exposed.',
        external_prerequisites=dict(
            data_root=read(RUN / 'protocol.json')['data_root'],
            prepared_root=read(RUN / 'protocol.json')['prepared_root'],
            cache_root=read(RUN / 'protocol.json')['cache_root'],
            required='Frozen data/labels, R84 TRAIN-only statistics, identicalinit_seed2.pt, '
                'solver ID+34 assets, exact input caches, best.pt and saved prediction archives '
                'remain external. Exact report regeneration also needs local history.json '
                'and historical R84/R94/R110/NSS saved artifacts. This is not a standalone '
                'weights/data release. Identities and paths are in committed protocols.'),
        exclusions=['*.pt', '*.pth', '*.npz', '*.npy', 'cache directories', 'wandb directories/receipts',
            'history.json', 'updates.jsonl', 'training logs', '*.png', 'per-problem figure directories',
            'R109 result/runtime directories', 'R112 and other concurrent work', 'unrelated old figures'],
        missing_or_not_computed=[
            'No R111-vs-R110 paired rescue/harm CSV exists; not computed from prediction arrays.',
            'No multi-seed uncertainty or newly matched causal control.',
            'No new review/model-backend attestation was performed in this read-only closeout.'])


def verify():
    manifest = read(MANIFEST)
    require(completion() == manifest['completion'], 'completion evidence changed')
    require(git('rev-parse', 'HEAD') == manifest['parent_commit'],
            'HEAD moved; main must rebase the publication inventory under its Git lock')
    root = Path(manifest['git_root'])
    for entry in manifest['files']:
        data = (root / entry['path']).read_bytes()
        require(len(data) == entry['bytes']
                and hashlib.sha256(data).hexdigest() == entry['sha256'],
                'publication file changed: ' + entry['path'])
    expected = sorted([e['path'] for e in manifest['files']
                       if not e['in_head'] or e['changed_from_head']]
                      + [MANIFEST.relative_to(root).as_posix()])
    require(manifest['stage_paths'] == expected, 'allowlist differs from inventoried files')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'verify', 'allowlist'))
    command = parser.parse_args().command
    if command == 'build':
        print(json.dumps(build(), indent=2, ensure_ascii=True, allow_nan=False))
    else:
        manifest = verify()
        if command == 'allowlist':
            sys.stdout.buffer.write(b'\0'.join(p.encode('utf-8')
                for p in manifest['stage_paths']) + b'\0')
        else:
            print(json.dumps(dict(verified=True, files=len(manifest['files']),
                stage_paths=len(manifest['stage_paths']), completion=manifest['completion'])))


if __name__ == '__main__':
    main()
