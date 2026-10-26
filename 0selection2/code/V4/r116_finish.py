"""Finite saved-only R116 closeout and optional one-attempt curated publication.

wait --root PATH --publish-after waits for both independent pipelines. It never
starts/restarts an arm. Budget truncation is incomplete, not a negative result.
"""

import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np

from . import r112_finish as original


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R116_subset_elimination'
ARMS = ('A_static', 'B_subset')
PROBLEMS = original.PROBLEMS
GIT_ROOT = original.GIT_ROOT
NotReady = original.NotReady
read, write, digest, utc = original.read, original.write, original.digest, original.utc


def _private_callbacks():
    module = types.ModuleType('code.V4._r116_finish_callbacks')
    module.__file__, module.__package__ = __file__, 'code.V4'
    sys.modules[module.__name__] = module
    source = Path(original.__file__).read_text(encoding='utf-8')
    for old, new in (('r112', 'r116'), ('R112', 'R116'), ('A_collapsed', 'A_static'),
                     ('B_interfaces', 'B_subset'), ('region-interface', 'subset-elimination'),
                     ('collapse versus retained interface conditions', 'static versus subset-conditioned exact elimination'),
                     ("else ('R89', 'B_muon_hybrid')", "else ('R77', 'B')")):
        source = source.replace(old, new)
    exec(compile(source, str(original.__file__) + ':R116_private', 'exec'), module.__dict__)
    module.ROOT, module.ARMS = ROOT, ARMS
    module.REFERENCES = dict(R77=V4 / 'runs/R77_sample_size_batch1024/B_full_seed2',
                            R94B=V4 / 'runs/R94_pretrained_node_fusion/B')
    return module


_callbacks = _private_callbacks()
BASE_OUTPUTS = _callbacks.OUTPUTS
OUTPUTS = (*BASE_OUTPUTS, 'decision_gate.json')
CODE_NAMES = ('r116_experiment.py', 'r116_objective.py', 'r116_model.py', 'r116_subset.py',
              'r116_reporting.py', 'r116_finish.py', 'r116_capacity.py', 'r116_launch.py',
              'run_v4_r116.sh', 'test_r116_objective.py', 'test_r116_subset.py')


class IncompleteBudget(RuntimeError):
    pass


def completed(root):
    root = Path(root).resolve()
    inputs = {}
    for arm in ARMS:
        path = root / 'preparation/resources/training' / arm / 'worker.json'
        if not path.is_file():
            raise NotReady('Awaiting independent R116 worker receipt: ' + arm)
        receipt = read(path)
        state = receipt.get('state')
        if state == 'incomplete_budget':
            raise IncompleteBudget('R116 ' + arm + ': GPU wall cap reached; retained partial state, not negative evidence')
        if state == 'failed':
            raise RuntimeError('R116 ' + arm + ' failed; no restart: ' + str(receipt.get('error', receipt.get('reason'))))
        if state != 'complete':
            raise NotReady('Awaiting R116 ' + arm + ': ' + str(state))
        stages = receipt.get('stages', [])
        if (receipt.get('exit_status') != 0 or [s.get('stage') for s in stages] != ['train', 'finalize']
                or any(s.get('state') != 'complete' or s.get('exit_status') != 0 for s in stages)):
            raise ValueError('Both R116 pipelines require rc0 train/finalize, never a partial snapshot')
        inputs[str(path)] = digest(path)
        for name in ('training_complete.json', 'test_complete.json', 'lib_stage_complete.json', 'report_complete.json'):
            path = root / arm / name
            if not path.is_file():
                raise NotReady('Awaiting ' + str(path))
            inputs[str(path)] = digest(path)
    return inputs


def decision_gate(root):
    root = Path(root)
    finals = {a: read(root / a / 'final_metrics.json') for a in ARMS}
    timings = {a: read(root / a / 'inference_timing.json') for a in ARMS}
    left, right = (finals[a]['val'] for a in ARMS)
    top1 = 100 * (right['macro']['macro_top1'] - left['macro']['macro_top1'])
    regret = right['macro']['macro_actual_regret_pct'] - left['macro']['macro_actual_regret_pct']
    sbs = right['macro']['macro_vs_sbs_pct'] - left['macro']['macro_vs_sbs_pct']
    harms = {p: right['per_problem'][p]['actual_regret_pct'] - left['per_problem'][p]['actual_regret_pct'] for p in PROBLEMS}
    values = [top1, regret, sbs, *harms.values()]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Gate requires finite complete whole18 native metrics')
    science = top1 >= .20 and regret <= .005 and sbs <= 0 and max(harms.values()) <= .05
    speed = timings[ARMS[1]]['p95_ratio_le_2']
    return dict(experiment='R116', complete=True, evaluated_on='locked whole18 VAL only',
        matched_comparison='B_subset vs A_static; same new objective; not an R77 loss ablation',
        top1_delta_pp=top1, actual_regret_delta_pp=regret, vs_sbs_delta_pp=sbs,
        per_problem_regret_delta_pp=harms, worst_regret_harm_pp=max(harms.values()),
        thresholds=dict(top1_gain_pp=.20, macro_regret_harm_pp=.005, max_task_regret_harm_pp=.05,
                        vs_sbs_must_not_worsen=True, max_p95_ratio=2),
        scientific_gate_passed=science, deployment_p95_gate_passed=speed,
        recommend_deployment=science and speed, single_seed=True,
        max_task_p95_ratio={a: timings[a]['max_task_p95_ratio'] for a in ARMS},
        TEST_used_for_gate=False, checkpoint_reselection=False, automatic_next_arms=False)


_render = _callbacks.render


def render(root, metrics, comparisons, references, boundary):
    gate = decision_gate(root)
    write(Path(root) / 'decision_gate.json', gate)
    _render(root, metrics, comparisons, references, boundary)
    path = Path(root) / 'RESULTS.md'
    with path.open('a', encoding='utf-8') as stream:
        stream.write('\n## Predeclared VAL Gate\n\n'
            f"B-A Top1: {gate['top1_delta_pp']:.6f} pp; actual regret: {gate['actual_regret_delta_pp']:.6f} pp; "
            f"vsSBS: {gate['vs_sbs_delta_pp']:.6f} pp.\n\n"
            f"Scientific gate: {gate['scientific_gate_passed']}; complete-DP p95 <=2x point selector: "
            f"{gate['deployment_p95_gate_passed']}. No TEST-based selection or new arms.\n\n"
            'Both arms use champion CE plus full-ranking NLL. Historical R77 is not an isolated loss ablation.\n')


def analyze(root, references=None):
    root = Path(root).resolve()
    return _callbacks.analyze(root, references)


def publication_paths(root):
    """Curated small files only; no arrays, weights, credentials, bulk logs or other versions."""
    root = Path(root).resolve()
    if root != ROOT.resolve():
        raise ValueError('Publication owns only the named in-repository R116 root')
    required = [V4 / name for name in CODE_NAMES]
    required += [root / name for name in (*OUTPUTS, 'finish_complete.json', 'protocol_lock.json', 'PLAN.md')]
    required += [root / 'preparation/training_ready.json']
    for arm in ARMS:
        required += [root / 'preparation/resources/training' / arm / 'worker.json']
        required += [root / arm / name for name in ('protocol.json', 'initialization.json', 'final_test_lock.json',
            'training_complete.json', 'report_complete.json', 'lib_stage_complete.json', 'inference_timing.json')]
        required += [root / arm / 'full_model_statistics' / name for name in
            ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
             'training_and_method_diagnostics.pdf', 'RESULTS.md', 'protocol.json', 'metrics_by_problem.csv',
             'nss_comparison.csv', 'selection_distribution.csv', 'inference_timing.csv', 'objective_diagnostics.csv')]
    # Local static imports plus the two explicitly source-adapted callback templates.
    dependencies = original.dependency_files([p for p in required if p.suffix == '.py'])
    required += [V4 / 'r112_experiment.py', V4 / 'r112_reporting.py', V4 / 'r112_finish.py']
    required += dependencies
    included, excluded = [], []
    for path in dict.fromkeys(required):
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(GIT_ROOT.resolve())):
            raise ValueError('Required ordinary in-repository publication file: ' + str(path))
        if path.stat().st_size > original.SMALL_LIMIT:
            if path == root / 'events.csv':
                excluded.append(dict(path=str(path.relative_to(GIT_ROOT)), sha256=digest(path),
                                     reason='large instance events remain local'))
                continue
            raise ValueError('R116 small-file limit exceeded: ' + str(path))
        included.append(path)
    if sum(p.stat().st_size for p in included) > original.TOTAL_LIMIT:
        raise ValueError('Bounded R116 publication exceeds total size limit')
    return included, excluded


def publication_preflight(root):
    _callbacks.verify_delivery(root)
    git = original.git
    if Path(git('rev-parse', '--show-toplevel')).resolve() != GIT_ROOT.resolve():
        raise ValueError('Publication Git root differs')
    if git('diff', '--cached', '--name-only', '-z'):
        raise ValueError('Preexisting staged changes; index untouched')
    ready = read(Path(root) / 'preparation/training_ready.json')
    for path, expected in ready['sources'].items():
        if digest(path) != expected:
            raise ValueError('Launch-frozen source changed: ' + path)
    if digest(ready['protocol_lock']) != ready['protocol_lock_sha256']:
        raise ValueError('Main launch protocol changed')
    # Require the complete runtime/report source identity embedded before training.
    for arm in ARMS:
        config = read(Path(root) / arm / 'protocol.json')
        for name, expected in config['source_sha256'].items():
            if digest(V4 / name) != expected:
                raise ValueError('Runtime source changed after training: ' + name)
    paths, excluded = publication_paths(root)
    names = [str(p.relative_to(GIT_ROOT)) for p in paths]
    branch = git('symbolic-ref', '--quiet', '--short', 'HEAD')
    remote = git('config', '--get', 'branch.' + branch + '.remote')
    merge = git('config', '--get', 'branch.' + branch + '.merge')
    if remote == '.' or not merge.startswith('refs/heads/'):
        raise ValueError('Require existing named remote/upstream; no inferred destination')
    tip = git('rev-parse', 'HEAD')
    upstream = git('ls-remote', '--heads', remote, merge).splitlines()
    if len(upstream) != 1 or upstream[0].split()[0] != tip:
        raise ValueError('HEAD must equal remote upstream; reconcile manually before one R116 publication')
    # Existing unrelated dirty dependency files are never silently swept into this version.
    closure_names = [n for n in names if n not in {str((V4 / p).relative_to(GIT_ROOT)) for p in CODE_NAMES}
                     and not (GIT_ROOT / n).is_relative_to(Path(root))]
    dirty = git('status', '--porcelain', '--', *closure_names) if closure_names else ''
    if dirty:
        raise ValueError('Dirty historical dependencies require main review, not automatic staging: ' + dirty[:1200])
    return dict(paths=names, excluded=excluded, remote=remote, merge=merge, branch=branch, parent=tip,
                sha256={n: digest(GIT_ROOT / n) for n in names})


_callbacks.completed, _callbacks.render = completed, render
_callbacks.OUTPUTS, _callbacks.CODE_NAMES = OUTPUTS, CODE_NAMES
_callbacks.publication_paths, _callbacks.publication_preflight = publication_paths, publication_preflight
publish = _callbacks.publish


def wait(root, *, timeout_hours=32., poll_seconds=60., publish_after=False, message=None):
    if not math.isfinite(timeout_hours) or timeout_hours <= 0 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError('Finite positive timeout and poll interval required')
    root = Path(root).resolve()
    deadline, previous = time.monotonic() + timeout_hours * 3600, None
    while True:
        try:
            completed(root)
            break
        except IncompleteBudget as error:
            value = dict(experiment='R116', complete=False, state='incomplete_budget', reason=str(error),
                interpretation='incomplete; not negative evidence', automatic_retry=False, publication=False)
            write(root / 'finish_status.json', value)
            return value
        except NotReady as error:
            if time.monotonic() >= deadline:
                value = dict(experiment='R116', complete=False, state='wait_timeout', reason=str(error),
                    interpretation='incomplete; not negative evidence', automatic_retry=False, publication=False)
                write(root / 'finish_status.json', value)
                return value
            if str(error) != previous:
                print(str(error), flush=True)
                previous = str(error)
            time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))
    value = analyze(root)
    return publish(root, message or 'Publish complete R116 matched elimination models and reports') if publish_after else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('wait', 'analyze', 'publish', 'status'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--publish-after', action='store_true')
    parser.add_argument('--timeout-hours', type=float, default=32.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--message', default='Publish complete R116 matched elimination models and reports')
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.command == 'wait':
        return wait(root, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds,
                    publish_after=args.publish_after, message=args.message)
    if args.command == 'publish':
        return publish(root, args.message)
    if args.command == 'analyze':
        return analyze(root)
    path = root / 'finish_complete.json'
    value = read(path) if path.exists() else dict(experiment='R116', complete=False, state='not_finished')
    print(json.dumps(value, indent=2))
    return value


if __name__ == '__main__':
    main()
