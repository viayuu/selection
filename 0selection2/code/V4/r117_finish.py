"""Saved-only R117 whole-model comparison, fixed36 recovery and opt-in publication.

Wait observes two actual completed worker receipts, never stale launch snapshots.
No GPU launches, retries, checkpoint changes or publication happen on import.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
import types

from . import r112_finish as original
from . import r116_mechanism_finish as fixed_template


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R117_native_computation'
ARMS = ('A', 'B')
PROBLEMS, GIT_ROOT = original.PROBLEMS, original.GIT_ROOT
NotReady = original.NotReady
read, write, digest, utc = original.read, original.write, original.digest, original.utc
CODE_NAMES = ('r117_experiment.py', 'r117_objective.py', 'r117_model.py', 'r117_native.py',
              'r117_features.py', 'r117_reporting.py', 'r117_finish.py', 'test_r117_experiment.py',
              'r117_launch.py', 'r117_capacity.py', 'run_v4_r117.sh',
              'test_r117_native.py', 'test_r117_model.py', 'test_r117_features.py')
EXTRA_OUTPUTS = (*fixed_template.EXTRA_OUTPUTS, 'decision_gate.json')
OUTPUTS = (*original.OUTPUTS, *EXTRA_OUTPUTS)
_context = {}


def _private(source, name):
    module = types.ModuleType('code.V4._r117_finish_' + name)
    module.__file__, module.__package__ = __file__, 'code.V4'
    sys.modules[module.__name__] = module
    text = Path(source.__file__).read_text(encoding='utf-8')
    for old, new in (('r112', 'r117'), ('R112', 'R117'), ('r116', 'r117'), ('R116', 'R117'),
                     ('A_collapsed', 'A'), ('B_interfaces', 'B'), ('A_static', 'A'), ('B_subset', 'B'),
                     ('R117_region_interfaces', 'R117_native_computation'),
                     ('R117_subset_elimination', 'R117_native_computation'),
                     ('collapse versus retained interface conditions', 'random versus pretrained frozen native additions'),
                     ('region-interface', 'native-computation')):
        text = text.replace(old, new)
    # The fixed36 template's generic helper import remains the read-only original.
    text = text.replace('from . import r117_finish as common', 'from . import r112_finish as common')
    exec(compile(text, str(source.__file__) + ':R117_private', 'exec'), module.__dict__)
    return module


_callbacks = _private(original, 'callbacks')
_fixed = _private(fixed_template, 'fixed36')
_callbacks.ROOT, _callbacks.ARMS = ROOT, ARMS
_callbacks.REFERENCES = {'R94B': V4 / 'runs/R94_pretrained_node_fusion/B'}
_callbacks.OUTPUTS, _callbacks.CODE_NAMES = OUTPUTS, CODE_NAMES


class IncompleteBudget(RuntimeError):
    pass


def completed(root):
    root, inputs = Path(root).resolve(), {}
    for arm in ARMS:
        path = root / 'preparation/resources/training' / arm / 'worker.json'
        if not path.is_file():
            raise NotReady('Awaiting independent R117 worker receipt: ' + arm)
        receipt = read(path)
        state = receipt.get('state')
        if state == 'incomplete_budget':
            raise IncompleteBudget('R117 ' + arm + ': 14h wall cap reached; incomplete, not negative evidence')
        if state == 'failed':
            raise RuntimeError('R117 ' + arm + ' failed; no restart: ' + str(receipt.get('error', receipt.get('reason'))))
        if state != 'complete':
            raise NotReady('Awaiting R117 ' + arm + ': ' + str(state))
        stages = receipt.get('stages', [])
        if (receipt.get('exit_status') != 0
                or [s.get('stage', s.get('name')) for s in stages] != ['train', 'finalize']
                or any(s.get('state') != 'complete' or s.get('exit_status') != 0 for s in stages)):
            raise ValueError('Require both independent rc0 train/finalize pipelines, never partial snapshots')
        inputs[str(path)] = digest(path)
        for name in ('training_complete.json', 'test_complete.json', 'lib_stage_complete.json',
                     'inference_timing.json', 'inference_device.json', 'report_complete.json'):
            path = root / arm / name
            if not path.is_file():
                raise NotReady('Awaiting ' + str(path))
            inputs[str(path)] = digest(path)
    for path in (Path(__file__), Path(original.__file__), Path(fixed_template.__file__),
                 Path(_fixed.recovery.__file__), _fixed.TARGET_LIST):
        original.remember(inputs, path)
    _context.clear()
    _context.update(root=root, inputs=inputs, models={}, saved={})
    return inputs


def decision_gate(root):
    finals = {a: read(Path(root) / a / 'final_metrics.json') for a in ARMS}
    left, right = (finals[a]['val'] for a in ARMS)
    if any(set(v['per_problem']) != set(PROBLEMS) for v in (left, right)):
        raise ValueError('The gate requires all18 tasks, not a favorable subset')
    top1 = 100 * (right['macro']['macro_top1'] - left['macro']['macro_top1'])
    regret = right['macro']['macro_actual_regret_pct'] - left['macro']['macro_actual_regret_pct']
    sbs = right['macro']['macro_vs_sbs_pct'] - left['macro']['macro_vs_sbs_pct']
    harms = {p: right['per_problem'][p]['actual_regret_pct'] - left['per_problem'][p]['actual_regret_pct'] for p in PROBLEMS}
    if not all(math.isfinite(v) for v in (top1, regret, sbs, *harms.values())):
        raise ValueError('VAL gate requires finite native metrics')
    passed = top1 >= .20 and regret <= 0 and sbs <= 0 and max(harms.values()) <= .05
    return dict(experiment='R117', complete=True, evaluated_on='locked whole18 VAL only',
        matched_comparison='B pretrained native additions vs A random frozen additions; common pretrained ReLD base',
        top1_delta_pp=top1, actual_regret_delta_pp=regret, vs_sbs_delta_pp=sbs,
        per_problem_regret_delta_pp=harms, worst_regret_harm_pp=max(harms.values()),
        thresholds=dict(top1_gain_pp=.20, macro_regret_harm_pp=0., max_task_regret_harm_pp=.05,
                        vs_sbs_must_not_worsen=True),
        scientific_gate_passed=passed, engineering_continuation_gate=True,
        not_a_significance_threshold=True, single_seed=True, TEST_used_for_gate=False,
        checkpoint_reselection=False, automatic_next_arms=False,
        timing_gate='none predeclared; online native-plus-selector and cache-hit latency reported separately')


def zero_selection_pdf(root, details, summary, status, gate):
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    target = Path(root) / 'ZERO_SELECTION.pdf'
    temporary = target.with_name('.ZERO_SELECTION.r117.pdf')
    try:
        with PdfPages(temporary) as pdf:
            def page(title, headers, cells, note):
                fig, ax = plt.subplots(figsize=(17., 10.5))
                ax.axis('off')
                fig.suptitle(title, fontsize=14)
                fig.text(.03, .91, note, fontsize=8)
                table = ax.table(cellText=cells, colLabels=headers, bbox=[0, .05, 1, .85], cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(6.5 if len(cells) > 25 else 8)
                fig.text(.03, .035, 'Native/FP64 ind; fixed R84 VAL target36; selected counts are not success; no inference/reselection.', fontsize=8)
                pdf.savefig(fig)
                plt.close(fig)

            page('R117 | Predeclared VAL Gate', ['Field', 'Result'], [
                ['Scientific gate', str(gate['scientific_gate_passed'])],
                ['B-A Top1 (pp)', f"{gate['top1_delta_pp']:.6f}"],
                ['B-A actual regret (pp)', f"{gate['actual_regret_delta_pp']:.6f}"],
                ['B-A vs SBS (pp)', f"{gate['vs_sbs_delta_pp']:.6f}"],
                ['Worst task regret harm (pp)', f"{gate['worst_regret_harm_pp']:.6f}"],
                ['Thresholds', str(gate['thresholds'])],
                ['Historical reference', 'R94B, not a matched control'],
                ['Timing', 'online native + selector separately from cached selector; no speed success threshold'],
                ['Fixed target provenance', status['identity']['baseline_val_prediction_sha256']]],
                'Single seed2 engineering continuation gate, not statistical significance; TEST never selects weights or targets.')
            for comparison in dict.fromkeys(r['comparison'] for r in details):
                for split in ('VAL', 'TEST'):
                    rows = [r for r in details if r['comparison'] == comparison and r['split'] == split]
                    page(split + ' Fixed36 | ' + comparison,
                        ['Task', 'Method', 'Winners', 'TP old', 'TP new', 'Recall %', 'Precision %', 'New FP',
                         'Rescues', 'Lost', 'dCost*', 'dRegret pp*'],
                        [[r['problem'], r['solver'], r['winner_count'], r['baseline_true_tp'], r['current_true_tp'],
                          _fixed._fmt(r['current_recall_pct']), _fixed._fmt(r['current_selection_precision_pct']),
                          r['new_false_pick_count'], r['true_rescues'], r['lost_true_picks'],
                          format(r['mean_cost_delta_contribution'], '.5g'),
                          format(r['actual_regret_delta_contribution_pp'], '.5g')] for r in rows],
                        '*Cost/regret contributions use the full task denominator; undefined ratios are blank, not zero.')
            for split in ('VAL', 'TEST'):
                rows = [r for r in summary if r['problem'] == 'ALL' and r['split'] == split]
                page(split + ' | Fixed36 / Outside36 / Full18',
                    ['Comparison', 'Block', 'TP old', 'TP new', 'New FP', 'Rescues', 'Lost', 'dCost', 'dRegret pp'],
                    [[r['comparison'], r['solver'], r['baseline_true_tp'], r['current_true_tp'],
                      r['new_false_pick_count'], r['true_rescues'], r['lost_true_picks'],
                      format(r['mean_cost_delta_contribution'], '.5g'),
                      format(r['actual_regret_delta_contribution_pp'], '.5g')] for r in rows],
                    'FIXED36 + OUTSIDE36 reconstruct FULL18. ALL cost/regret are problem-equal; raw costs descriptive only.')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


_fixed.zero_selection_pdf = zero_selection_pdf
_load_run, _saved_split, _render = _callbacks.load_run, _callbacks.saved_split, _callbacks.render


def load_run(*args, **kwargs):
    model = _load_run(*args, **kwargs)
    _context['models'][str(model['run'])] = model
    return model


def saved_split(model, split, inputs):
    data = _saved_split(model, split, inputs)
    _context['saved'][str(model['run']), split] = data
    return data


def render(root, metrics, comparisons, references, boundary):
    root = Path(root).resolve()
    if _context.get('root') != root:
        raise ValueError('Fixed36 analysis requires both real completed worker receipts')
    gate = decision_gate(root)
    write(root / 'decision_gate.json', gate)
    _render(root, metrics, comparisons, references, boundary)
    if not _fixed.FONT.is_file():
        from matplotlib.font_manager import findfont
        _fixed.FONT = Path(findfont('DejaVu Sans'))
    _fixed._supplement(root, _context['models'], _context['saved'], _context['inputs'], gate)
    path = root / 'RESULTS.md'
    text = path.read_text(encoding='utf-8')
    text += ('\n## Predeclared VAL Gate\n\n'
        f"B-A Top1={gate['top1_delta_pp']:.6f} pp; actual regret={gate['actual_regret_delta_pp']:.6f} pp; "
        f"vs_SBS={gate['vs_sbs_delta_pp']:.6f} pp. Gate passed: {gate['scientific_gate_passed']}.\n\n"
        'Frozen R84 fixed36 true recovery, false picks and outside36/full18 partitions: ZERO_SELECTION.pdf and target_recovery*.csv.\n'
        'Pretrained ReLD static base is identical in both arms; only added frozen native computation weights differ.\n'
        'Cache-hit selector time is not deployed native-computation latency. No TEST/LIB-based reselection.\n')
    path.write_text(text, encoding='utf-8')


_callbacks.completed, _callbacks.load_run = completed, load_run
_callbacks.saved_split, _callbacks.render = saved_split, render


def analyze(root, references=None):
    return _callbacks.analyze(Path(root).resolve(), references)


def publication_paths(root):
    root = Path(root).resolve()
    if root != ROOT.resolve():
        raise ValueError('Publication owns only the in-repository R117 root')
    required = [V4 / n for n in CODE_NAMES]
    required += [root / n for n in (*OUTPUTS, 'finish_complete.json', 'protocol_lock.json', 'PLAN.md',
                                  'preparation/training_ready.json')]
    required.append(_fixed.TARGET_LIST)
    for arm in ARMS:
        required.append(root / 'preparation/resources/training' / arm / 'worker.json')
        required += [root / arm / n for n in ('protocol.json', 'initialization.json', 'final_test_lock.json',
            'training_complete.json', 'test_complete.json', 'report_complete.json', 'lib_stage_complete.json',
            'inference_timing.json', 'inference_device.json')]
        required += [root / arm / 'full_model_statistics' / n for n in
            ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
             'training_and_method_diagnostics.pdf', 'RESULTS.md', 'protocol.json', 'metrics_by_problem.csv',
             'nss_comparison.csv', 'selection_distribution.csv', 'inference_timing.csv', 'objective_diagnostics.csv')]
        required += [V4 / n for n in read(root / arm / 'protocol.json')['source_sha256']]
    for path in read(root / 'preparation/training_ready.json').get('sources', {}):
        candidate = Path(path)
        if candidate.suffix in ('.py', '.sh'):
            required.append(candidate)
    required += original.dependency_files([p for p in required if p.suffix == '.py'])
    included, excluded = [], []
    for path in dict.fromkeys(required):
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(GIT_ROOT.resolve())):
            raise ValueError('Publication requires ordinary in-repository files: ' + str(path))
        if path.stat().st_size > original.SMALL_LIMIT:
            if path == root / 'events.csv':
                excluded.append(dict(path=str(path.relative_to(GIT_ROOT)), sha256=digest(path), reason='large events remain local'))
                continue
            raise ValueError('R117 small-file limit exceeded: ' + str(path))
        included.append(path)
    if sum(p.stat().st_size for p in included) > original.TOTAL_LIMIT:
        raise ValueError('Bounded curated R117 publication exceeds total size limit')
    return included, excluded


def publication_preflight(root):
    root = Path(root).resolve()
    _callbacks.verify_delivery(root)
    git = original.git
    if Path(git('rev-parse', '--show-toplevel')).resolve() != GIT_ROOT.resolve():
        raise ValueError('Publication Git root differs')
    if git('diff', '--cached', '--name-only', '-z'):
        raise ValueError('Preexisting staged changes; index untouched')
    ready = read(root / 'preparation/training_ready.json')
    for path, expected in ready['sources'].items():
        if digest(path) != expected:
            raise ValueError('Launch-frozen source changed: ' + path)
    if digest(ready['protocol_lock']) != ready['protocol_lock_sha256']:
        raise ValueError('Main launch protocol changed')
    for arm in ARMS:
        for name, expected in read(root / arm / 'protocol.json')['source_sha256'].items():
            if digest(V4 / name) != expected:
                raise ValueError('Runtime/report source changed after training: ' + name)
    paths, excluded = publication_paths(root)
    names = [str(p.relative_to(GIT_ROOT)) for p in paths]
    branch = git('symbolic-ref', '--quiet', '--short', 'HEAD')
    remote = git('config', '--get', 'branch.' + branch + '.remote')
    merge = git('config', '--get', 'branch.' + branch + '.merge')
    if remote == '.' or not merge.startswith('refs/heads/'):
        raise ValueError('Require existing named remote/upstream; never infer a publication destination')
    tip, upstream = git('rev-parse', 'HEAD'), git('ls-remote', '--heads', remote, merge).splitlines()
    if len(upstream) != 1 or upstream[0].split()[0] != tip:
        raise ValueError('HEAD must equal remote upstream; reconcile manually before one R117 publication')
    unrelated = [n for n in names if not (GIT_ROOT / n).is_relative_to(root)
                 and not ((GIT_ROOT / n).parent == V4 and (GIT_ROOT / n).name.startswith(('r117_', 'test_r117_', 'run_v4_r117')))]
    dirty = git('status', '--porcelain', '--', *unrelated) if unrelated else ''
    if dirty:
        raise ValueError('Dirty historical dependencies require parent review, never automatic staging: ' + dirty[:1200])
    return dict(paths=names, excluded=excluded, remote=remote, merge=merge, branch=branch, parent=tip,
                sha256={n: digest(GIT_ROOT / n) for n in names})


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
        except (IncompleteBudget, NotReady) as error:
            state = 'incomplete_budget' if isinstance(error, IncompleteBudget) else 'wait_timeout'
            if isinstance(error, IncompleteBudget) or time.monotonic() >= deadline:
                value = dict(experiment='R117', complete=False, state=state, reason=str(error),
                    interpretation='incomplete; not negative evidence', automatic_retry=False, publication=False)
                write(root / 'finish_status.json', value)
                return value
            if str(error) != previous:
                print(str(error), flush=True)
                previous = str(error)
            time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))
    value = analyze(root)
    return publish(root, message or 'Publish complete R117 native computation models and fixed36 reports') if publish_after else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('wait', 'analyze', 'publish', 'status'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--publish-after', action='store_true')
    parser.add_argument('--timeout-hours', type=float, default=32.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--message', default='Publish complete R117 native computation models and fixed36 reports')
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.publish_after and args.command != 'wait':
        raise ValueError('--publish-after belongs only to wait')
    if args.command == 'wait':
        return wait(root, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds,
            publish_after=args.publish_after, message=args.message)
    if args.command == 'analyze':
        return analyze(root)
    if args.command == 'publish':
        return publish(root, args.message)
    path = root / 'finish_complete.json'
    value = read(path) if path.exists() else dict(experiment='R117', complete=False, state='not_finished')
    print(json.dumps(value, indent=2))
    return value


if __name__ == '__main__':
    main()
