"""CPU-only support geometry of the completed, immutable R123 prediction bank.

No model loader, query input, solver, optimizer or deployable weight policy is
used here. Native answers identify an offline feasibility question only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from scipy.optimize import linprog

from . import r123_finish as saved


VERSION = 'return18_moe_support_v1'
ROOT = saved.ROOT.resolve()
OUTPUT = (Path(__file__).parent / 'runs/Return18_post_R120/moe_support').resolve()
SCIENCE = Path(__file__).resolve().parents[2] / 'docs/research/frozen_selector_mixture_information_20261030/REPORT.md'
TARGETS = saved.REFERENCE_ROOT / 'comparison_artifacts/target_list.json'
FONT = Path('/usr/share/fonts/google-droid/DroidSansFallback.ttf')
TOL = 1e-8
LP_TOL = 1e-9
HARD, REACHABLE, UNREACHABLE, BOUNDARY = range(4)
MODES = ('probability', 'logits')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def pin_cpu():
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    return cpu


def softmax64(logits):
    value = np.asarray(logits, dtype=np.float64)
    value = np.exp(value - value.max(axis=-1, keepdims=True))
    return value / value.sum(axis=-1, keepdims=True)


def differences(scores, winner):
    rivals = np.flatnonzero(np.arange(scores.shape[1]) != winner)
    delta = (scores[:, winner, None] - scores[:, rivals]).T
    scale = float(np.max(np.abs(delta)))
    return rivals, delta / (scale or 1.), scale or 1.


def margin_lp(delta, strict=None):
    """Three simplex weights plus one max-min margin; one bounded LP call."""
    strict = np.ones(len(delta), dtype=bool) if strict is None else strict
    matrix = np.column_stack((-delta, strict.astype(float)))
    return linprog([0., 0., 0., -1.], A_ub=matrix, b_ub=np.zeros(len(delta)),
        A_eq=[[1., 1., 1., 0.]], b_eq=[1.],
        bounds=[(0., 1.)] * 3 + [(None, None)], method='highs-ds',
        options={'primal_feasibility_tolerance': LP_TOL,
                 'dual_feasibility_tolerance': LP_TOL,
                 'maxiter': 1000, 'time_limit': 2.})


def witness(result, delta):
    if not result.success or result.x is None:
        return None
    weights = result.x[:3]
    if not np.isfinite(weights).all() or weights.min() < -LP_TOL:
        return None
    weights = np.maximum(weights, 0.)
    weights /= weights.sum()
    return weights, delta @ weights


def solve_support(scores, winner, pool_ids):
    """Certify positive witnesses/negative dual bounds; do not round ties up."""
    rivals, delta, scale = differences(scores, winner)
    result = margin_lp(delta)
    record = dict(lp_calls=1, lp_status=int(result.status), scale=scale,
                  reason='numerical_boundary', state=BOUNDARY)
    proof = witness(result, delta)
    if proof is None:
        record['reason'] = 'lp_unresolved'
        return record
    weights, margins = proof
    record.update(margin_lower=float(margins.min()),
                  weights=weights.tolist(), margin_upper=None)
    # A dual mixture of rivals provides a directly checked impossibility proof.
    beta = -np.asarray(result.ineqlin.marginals)
    if np.isfinite(beta).all() and beta.min() >= -LP_TOL and beta.sum() > 0:
        beta = np.maximum(beta, 0.)
        beta /= beta.sum()
        record['margin_upper'] = float(np.max(delta.T @ beta))
    action = int(saved.hard_actions((weights @ scores)[None], pool_ids)[0])
    if margins.min() > TOL and action == winner:
        record.update(state=REACHABLE, reason='positive_margin')
        return record
    if record['margin_upper'] is not None and record['margin_upper'] < -TOL:
        record.update(state=UNREACHABLE, reason='convex_rival_certificate')
        return record
    lower_id = pool_ids[rivals] < pool_ids[winner]
    # Strict inequality is required against EVERY lower-ID rival. Exact weak
    # dominance by one such rival blocks the native winner even on a tied face.
    if np.any(lower_id & np.all(delta <= 0., axis=1)):
        record.update(state=UNREACHABLE, reason='exact_lower_ID_block')
        return record
    if lower_id.any():
        tied = margin_lp(delta, lower_id)
        record['lp_calls'] += 1
        proof = witness(tied, delta)
        if proof is not None:
            weights, margins = proof
    # Equality is accepted only with a direct nonnegative witness, strict
    # lower-ID separation and an actual ID-tied argmax. Near-zero negatives
    # are NOT silently repaired with a tolerance.
    action = int(saved.hard_actions((weights @ scores)[None], pool_ids)[0])
    if (margins.min() == 0. and (not lower_id.any() or margins[lower_id].min() > TOL)
            and action == winner):
        record.update(state=REACHABLE, reason='verified_ID_tie_face',
                      weights=weights.tolist(), margin_lower=float(margins.min()))
    return record


def support(scores, winner, pool_ids, hard_covered):
    actions = saved.hard_actions(scores, pool_ids)
    covered = (actions == winner[:, None]).any(axis=1)
    if not np.array_equal(covered, hard_covered):
        raise ValueError('FP64 softmax did not preserve the saved hard support')
    state = np.full(len(winner), HARD, dtype=np.int8)
    reason = np.full(len(winner), 'hard_vertex', dtype=object)
    rows = np.flatnonzero(~covered)
    query = scores[rows]
    delta = query[np.arange(len(rows)), :, winner[rows]][:, :, None] - query
    scale = np.max(np.abs(delta), axis=(1, 2))
    scale[scale == 0.] = 1.
    dominated = (delta.max(axis=1) < -TOL * scale[:, None]).any(axis=1)
    state[rows[dominated]] = UNREACHABLE
    reason[rows[dominated]] = 'common_strict_dominance'
    cases, calls = [], 0
    for index in rows[~dominated]:
        record = solve_support(scores[index], int(winner[index]), pool_ids)
        state[index], reason[index] = record['state'], record['reason']
        calls += record['lp_calls']
        cases.append(dict(local_row=int(index), **record))
    if not np.array_equal(state == HARD, hard_covered):
        raise AssertionError('Support partition changed saved hard actions')
    return dict(state=state, reason=reason, cases=cases, lp_calls=calls)


def validate_archive(data, capture, *, protocol, manifest, split, problem, expert):
    meta = json.loads(str(data['metadata_json']))
    bank = next(x for x in protocol['expert_bank'] if x['expert_id'] == expert)
    required = dict(schema='r123_shared_expert_npz_v1', complete=True, rows=4500,
        split=split, problem=problem, expert_id=expert, bank_order=list(saved.EXPERTS),
        checkpoint_path=bank['checkpoint_path'], checkpoint_sha256=bank['checkpoint_sha256'],
        factory=bank['factory'], meta_preparation_identity=manifest['preparation_identity'],
        pool=protocol['candidate_pools'][problem], pool_ids=protocol['candidate_pool_ids'][problem],
        helper_sha256=protocol['shared_prediction_cache_sha256'],
        score_ties='ascending global solver ID', query_labels_stored=False)
    if any(meta.get(k) != v for k, v in required.items()):
        raise ValueError('Bank schema/checkpoint/pool/source version differs')
    n, k = len(capture['winner']), len(capture['pool'])
    if (n != 4500 or data['logits'].dtype != np.float32
            or data['logits'].shape != (n, k) or not np.isfinite(data['logits']).all()
            or data['pool'].tolist() != required['pool']
            or data['pool_ids'].tolist() != required['pool_ids']):
        raise ValueError('Incomplete/nonfinite/misaligned bank')
    for field in ('logits', 'row_indices', 'source_indices', 'source_identity'):
        if array_hash(data[field]) != meta[field + '_sha256']:
            raise ValueError('Bank content hash differs: ' + field)
    for actual, expected in (('row_indices', 'indices'),
                             ('source_indices', 'release_source_indices'),
                             ('source_identity', 'source_identity'),
                             ('pool', 'pool'), ('pool_ids', 'pool_ids')):
        if not np.array_equal(data[actual], capture[expected]):
            raise ValueError('Bank/capture row or candidate join differs: ' + actual)
    return meta


def counts(result, mask):
    state, reason = result['state'][mask], result['reason'][mask]
    return dict(hard_covered=int(np.sum(state == HARD)),
        additional_reachable=int(np.sum(state == REACHABLE)),
        unreachable=int(np.sum(state == UNREACHABLE)), boundary=int(np.sum(state == BOUNDARY)),
        common_dominated=int(np.sum(reason == 'common_strict_dominance')),
        convex_rival_blocked=int(np.sum(reason == 'convex_rival_certificate')),
        ID_tie_blocked=int(np.sum(reason == 'exact_lower_ID_block')),
        verified_tie_reachable=int(np.sum(reason == 'verified_ID_tie_face')))


def summarize(capture, actions, result, *, problem, split, mode, target_keys):
    winner, costs, pred = capture['winner'], capture['costs'], capture['pred']
    n = len(winner)
    covered = (actions == winner[:, None]).any(axis=1)
    metrics = saved.bank_decomposition(capture, actions)
    row = dict(problem=problem, split=split, pool_mode=mode, n=n,
        actual_router_correct=int(np.sum(pred == winner)), hard_uncovered=int(np.sum(~covered)),
        router_missed_covered=int(np.sum(covered & (pred != winner))),
        **counts(result, np.ones(n, dtype=bool)),
        actual_router_top1_pct=100. * float(np.mean(pred == winner)),
        hard_support_pct=metrics['bank_native_coverage_pct'],
        offline_additional_pp=100. * float(np.mean(result['state'] == REACHABLE)),
        offline_supported_pct=100. * float(np.mean(result['state'] <= REACHABLE)),
        bank_quality_floor_pct=metrics['bank_quality_floor_pct'],
        actual_regret_pct=metrics['actual_regret_pct'], lp_calls=result['lp_calls'],
        is_deployable=False)
    methods = []
    tied = np.sum(costs == costs.min(axis=1, keepdims=True), axis=1) > 1
    for slot, method in enumerate(capture['pool']):
        won = winner == slot
        selected = int(np.sum(pred == slot))
        methods.append(dict(problem=problem, split=split, pool_mode=mode,
            method=str(method), global_id=int(capture['pool_ids'][slot]), n=n,
            fixed36=(problem, str(method)) in target_keys,
            selected_count=selected, native_winner_count=int(won.sum()),
            native_unique_cost_count=int(np.sum(won & ~tied)),
            native_tied_cost_count=int(np.sum(won & tied)),
            missed_native_count=int(np.sum(won & (pred != winner))),
            hard_uncovered=int(np.sum(won & ~covered)),
            empty_native_group=not bool(won.any()),
            zero_selected_positive_native=selected == 0 and bool(won.any()),
            near_zero_selected_positive_native=0 < 100. * selected / n <= .1 and bool(won.any()),
            **counts(result, won)))
    if row['hard_covered'] + row['additional_reachable'] + row['unreachable'] + row['boundary'] != n:
        raise AssertionError('Support states do not partition the population')
    return row, methods, metrics


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def verify_decomposition(actual, expected):
    for key, value in actual.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and expected.get(key, '') != '':
            np.testing.assert_allclose(value, float(expected[key]), atol=1e-10, rtol=1e-10,
                                       err_msg='Existing bank decomposition differs: ' + key)


def macro(rows, split, mode):
    rows = [r for r in rows if r['split'] == split and r['pool_mode'] == mode]
    out = dict(problem='ALL18', split=split, pool_mode=mode,
               aggregation='equal problem weight; all tasks n=4500', is_deployable=False)
    for key, value in rows[0].items():
        if isinstance(value, int) and not isinstance(value, bool):
            out[key] = sum(r[key] for r in rows)
        elif isinstance(value, float):
            out[key] = float(np.mean([r[key] for r in rows]))
    return out


def analyze(root, output):
    start, cpu_start = time.perf_counter(), time.process_time()
    cpu = pin_cpu()
    root, output = Path(root).resolve(), Path(output).resolve()
    if output != OUTPUT:
        raise ValueError('Writes are restricted to Return18_post_R120/moe_support')
    if (output / 'evidence.json').exists():
        raise FileExistsError('Completed sidecar already exists; no silent overwrite')
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    receipt_path = root / 'finish/reports/receipt.json'
    receipt = saved.read_json(receipt_path)
    saved.remember(inputs, receipt_path)
    if receipt.get('complete') is not True:
        raise ValueError('Only a completed locked R123 report can be diagnosed')

    def checked(path):
        path = Path(path).resolve()
        digest = saved.file_hash(path)
        if receipt['input_files_sha256'].get(str(path)) != digest:
            raise ValueError('Saved completed-report source changed: ' + str(path))
        inputs[str(path)] = digest
        return path

    protocol = saved.read_json(checked(root / 'model/protocol.json'))
    lock = saved.read_json(checked(root / 'model/final_test_lock.json'))
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    if (lock['protocol_sha256'] != identity or receipt['protocol_sha256'] != identity
            or lock['checkpoint_sha256'] != receipt['checkpoint_sha256']
            or lock['selected_update'] != receipt['selected_update']
            or lock['selected_without_test'] is not True or lock['completed_updates'] != 7200
            or protocol['expert_order'] != list(saved.EXPERTS)):
        raise ValueError('Unique completed checkpoint/protocol/bank lock differs')
    for name in ('r123_finish.py', 'r123_prediction_cache.py', 'r123_model.py', 'r123_experiment.py'):
        checked(Path(__file__).with_name(name))
    manifest_path = Path(protocol['meta_preparation']) / 'manifest.json'
    manifest = saved.read_json(manifest_path)
    saved.remember(inputs, manifest_path)
    target_path = checked(TARGETS)
    targets = saved.read_json(target_path)
    target_keys = {(r['problem'], r['method']) for r in targets['targets']}
    if len(target_keys) != 36 or targets['identity']['data_root'] != protocol['data_root']:
        raise ValueError('Original fixed36/data identity differs')
    old_path = root / 'finish/reports/bank_decomposition.csv'
    if saved.file_hash(old_path) != receipt['artifacts'][old_path.name]:
        raise ValueError('Original bank decomposition changed')
    saved.remember(inputs, old_path)
    old = {(r['split'], r['problem']): r for r in read_csv(old_path)}
    saved.remember(inputs, SCIENCE)
    summaries, methods, cases, decompositions, bank_metadata = [], [], [], [], []
    read_bytes = 0
    for split in ('val', 'test'):
        metrics = saved.read_json(checked(root / f'model/evaluation/{split}/selected_best/metrics.json'))
        if (metrics.get('experiment') != 'R123' or metrics.get('split') != split
                or metrics.get('successful_updates') != lock['selected_update']
                or set(metrics['per_problem']) != set(saved.PROBLEMS)):
            raise ValueError('Saved evaluation checkpoint identity differs')
        for problem in saved.PROBLEMS:
            capture_path = checked(root / f'model/evaluation/{split}/selected_best/by_problem/{problem}.npz')
            capture = saved.read_npz(capture_path)
            saved.validate_prediction(capture)
            archives = []
            for expert in saved.EXPERTS:
                path = checked(root / f'prediction_cache/{split}/{problem}/{expert}.npz')
                archive = saved.read_npz(path)
                metadata = validate_archive(archive, capture, protocol=protocol, manifest=manifest,
                                            split=split, problem=problem, expert=expert)
                archives.append(archive)
                bank_metadata.append(dict(path=str(path), metadata=metadata))
                read_bytes += path.stat().st_size
            bank = SimpleNamespace(logits=np.stack([a['logits'] for a in archives], axis=1),
                row_indices=archives[0]['row_indices'], source_indices=archives[0]['source_indices'],
                source_identity=archives[0]['source_identity'])
            actions = saved.check_bank_join(capture, bank)
            hard = (actions == capture['winner'][:, None]).any(axis=1)
            for mode in MODES:
                scores = softmax64(bank.logits) if mode == 'probability' else bank.logits.astype(np.float64)
                result = support(scores, capture['winner'], capture['pool_ids'], hard)
                row, detail, budget = summarize(capture, actions, result, problem=problem,
                    split=split, mode=mode, target_keys=target_keys)
                summaries.append(row)
                methods.extend(detail)
                if mode == MODES[0]:
                    verify_decomposition(budget, old[split, problem])
                    np.testing.assert_allclose(row['actual_router_top1_pct'],
                        100. * metrics['per_problem'][problem]['top1'], rtol=0, atol=1e-10)
                    decompositions.append(dict(problem=problem, split=split, **budget))
                for case in result['cases']:
                    index = case['local_row']
                    cases.append(dict(split=split, problem=problem, pool_mode=mode,
                        native_method=str(capture['pool'][capture['winner'][index]]),
                        source_identity=bytes(capture['source_identity'][index]).decode(), **case))
                print(json.dumps(dict(split=split, problem=problem, pool_mode=mode,
                    hard_uncovered=row['hard_uncovered'], reachable=row['additional_reachable'],
                    unreachable=row['unreachable'], boundary=row['boundary'], lp_calls=result['lp_calls'])), flush=True)
            read_bytes += capture_path.stat().st_size
        for mode in MODES:
            overall = macro(summaries, split, mode)
            summaries.append(overall)
            verify_decomposition(dict(bank_native_coverage_pct=overall['hard_support_pct'],
                bank_quality_floor_pct=overall['bank_quality_floor_pct'],
                actual_regret_pct=overall['actual_regret_pct']), old[split, 'ALL18'])
    fixed = [r for r in methods if r['fixed36']]
    if len(fixed) != 36 * 2 * len(MODES):
        raise AssertionError('Fixed36 was not preserved in both splits and both geometries')
    for name, rows in (('support_summary.csv', summaries), ('method_support.csv', methods),
                       ('fixed36_support.csv', fixed), ('verified_bank_decomposition.csv', decompositions)):
        saved.write_csv(output / name, rows)
    # Only the small non-dominated LP residual is retained, not another full bank.
    saved.write_json(output / 'lp_residual_cases.json', cases)
    timing = dict(cpu_id=cpu, affinity=sorted(os.sched_getaffinity(0)),
        omp_threads=1, analyzed_query_rows=162000, reused_expert_rows=486000,
        existing_archive_bytes=read_bytes, lp_calls=sum(r['lp_calls'] for r in summaries if r['problem'] != 'ALL18'),
        analysis_wall_seconds=time.perf_counter() - start,
        analysis_cpu_seconds=time.process_time() - cpu_start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    evidence = dict(schema=VERSION, complete=True, checkpoint=lock,
        expert_bank=protocol['expert_bank'], data_root=protocol['data_root'],
        population='full18 frozen VAL4500/task + TEST4500/task; no TRAIN or LIB added',
        score_geometry=dict(probability='FP64 stable softmax of original FP32 logits, temperature=1',
                            logits='original FP32 logits cast to FP64, no per-expert rescaling'),
        numerical_policy=dict(normalized_margin_tolerance=TOL, highs_feasibility_tolerance=LP_TOL,
            scale='per-query max absolute target-minus-rival score difference; zero -> 1',
            lp='max-min margin, 3 nonnegative expert weights summing to 1; highs-ds; maxiter1000, 2s/call',
            certificates='positive primal witness or negative directly checked rival-mixture dual bound',
            boundary='neither certified reachable nor unreachable; never counted as rescued',
            ties='equal scores: lowest global solver ID; lower-ID rivals require STRICT separation',
            boundary_extra_lp='maximize lower-ID margin subject to nonnegative gaps to every other rival'),
        winner='saved native int64 winner; exact FP64 cost minimum verified; cost ties NOT relabeled by ID',
        interpretation=dict(deployable=False, model_accuracy=False, bayes_or_general_architecture_upper_bound=False,
            per_query_oracle_weights_use_native_answers=True, weights_or_policy_fitted=False,
            support_only='closed-simplex geometry for this frozen bank and specified score pool',
            query_costs_used='offline validation/decomposition only, never a forward input',
            historical_test_development_exposure=True,
            source_novelty='source-record certificate only; exhaustive same-base/content novelty unproven',
            capture_attribution=receipt['existing_runner_capture_attribution'],
            three_expert_precompute_cost=receipt['bank_precompute_accounting']),
        execution=timing, input_files_sha256=inputs, bank_metadata=bank_metadata,
        no_nn_inference=True, no_gpu=True, no_solver=True, no_training=True,
        torch_imported='torch' in sys.modules, source_sha256=saved.file_hash(__file__))
    render_pdf(output / 'RESULTS.pdf', summaries, methods, fixed, evidence)
    evidence['execution'].update(total_wall_seconds=time.perf_counter() - start,
        total_cpu_seconds=time.process_time() - cpu_start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    evidence['artifacts_sha256'] = {p.name: saved.file_hash(p) for p in output.iterdir()
        if p.is_file() and p.suffix in ('.csv', '.pdf', '.json') and p.name != 'evidence.json'}
    saved.write_json(output / 'evidence.json', evidence)
    print(json.dumps(dict(complete=True, output=str(output), execution=evidence['execution'])), flush=True)
    return evidence


def render_pdf(path, summaries, methods, fixed, evidence):
    os.environ['MPLCONFIGDIR'] = str(path.parent / 'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    if not FONT.is_file():
        raise FileNotFoundError('Verified embedded CJK font missing')
    font_manager.fontManager.addfont(str(FONT))
    font = font_manager.FontProperties(fname=str(FONT))
    matplotlib.rcParams.update({'font.family': font.get_name(), 'pdf.fonttype': 42,
                                'axes.unicode_minus': False})
    ink, accent = '#19343b', '#176b75'
    lookup = {(r['split'], r['pool_mode'], r['problem']): r for r in summaries}

    def page(title, number):
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='#fcfcfa')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        ax.text(.06, .958, title, fontsize=17, color=ink, va='top')
        ax.plot([.06, .94], [.921, .921], color=accent, linewidth=1.5)
        ax.text(.06, .025, 'R123 冻结三专家 | 仅离线支持域诊断，不是部署成绩', fontsize=8, color=ink)
        ax.text(.94, .025, f'{number} / 4', ha='right', fontsize=8, color=ink)
        return fig, ax

    def text(ax, y, lines, size=10):
        if isinstance(lines, str):
            lines = lines.split('\n')
        ax.text(.06, y, '\n'.join(lines), fontsize=size, color=ink, va='top', linespacing=1.6)
        return y - len(lines) * .022 - .023

    def heading(ax, y, title):
        ax.text(.06, y, title, fontsize=12, color=accent, va='top')
        return y - .037

    def table(ax, y, headers, rows, widths, height=.034, size=9):
        widths = np.asarray(widths) / sum(widths) * .88
        for i, row in enumerate([headers] + rows):
            h = .047 if i == 0 else height
            ax.add_patch(Rectangle((.06, y - h), .88, h,
                facecolor='#dcebed' if i == 0 else '#f0f5f4' if i % 2 else '#ffffff',
                edgecolor='#dce4e4', linewidth=.35))
            x = .06
            for value, width in zip(row, widths):
                ax.text(x + .006, y - h / 2, str(value), fontsize=size, va='center',
                        color=ink, linespacing=1.2)
                x += width
            y -= h
        return y - .024

    with PdfPages(path, metadata={'Title': '冻结三专家：硬路由与共同凸池支持域',
                                 'Author': 'Return18 CPU-only saved-bank analysis'}) as pdf:
        fig, ax = page('失败来自哪里：硬动作支持域与门选错', 1)
        y = text(ax, .895, ['同一个已完成模型，冻结完整专家 R77B / R80A / R83。',
            '每题 VAL、TEST 各4500例；18题等权，不混合跨问题 raw cost。'])
        rows = []
        for split in ('val', 'test'):
            r = lookup[split, 'logits', 'ALL18']
            rows.append([split.upper(), f"{r['actual_router_top1_pct']:.4f}",
                f"{r['hard_support_pct']:.4f}", f"{100*r['hard_uncovered']/r['n']:.4f}",
                f"{100*r['router_missed_covered']/r['n']:.4f}"])
        y = heading(ax, y, '1  原报告硬路由分解已从保存数组复现')
        y = table(ax, y, ['划分', '实际模型\nTop1 %', '硬bank\n覆盖 %', 'bank未覆盖\n百分点', '可覆盖但门错\n百分点'],
                  rows, [1., 1.6, 1.6, 1.9, 2.1])
        r = lookup['test', 'logits', 'ALL18']
        y = text(ax, y, [f"TEST：硬bank质量floor {r['bank_quality_floor_pct']:.8f}%，实际regret {r['actual_regret_pct']:.8f}%。",
            '质量floor按逐例最低成本硬bank动作计算；不等于native错误率。'], size=9.5)
        y = heading(ax, y, '2  软池能否让原先三专家都漏掉的winner登顶？')
        rows = []
        for split in ('val', 'test'):
            for mode, label in (('probability', '概率凸池'), ('logits', 'logits凸池')):
                r = lookup[split, mode, 'ALL18']
                rows.append([split.upper(), label, r['hard_uncovered'], r['additional_reachable'],
                             r['unreachable'], r['boundary']])
        y = table(ax, y, ['划分', '共同权重池', '原hard\n未覆盖', '新增\n可达', '仍然\n不可达', '数值\n边界'],
                  rows, [1., 1.9, 1.3, 1.1, 1.3, 1.1], height=.04)
        p, l = lookup['test', 'probability', 'ALL18'], lookup['test', 'logits', 'ALL18']
        y = text(ax, y, [f"TEST新增几何支持：概率池 {p['offline_additional_pp']:.4f} 个百分点，",
            f"原logits池 {l['offline_additional_pp']:.4f} 个百分点。边界样本不计入可救。",
            '这只说明固定三专家的共同凸池很难覆盖既有硬bank漏项；',
            '它不否定更自由的候选再判别，也不证明任何权重规则可学习。'])
        text(ax, y, ['逐查询使用真winner问“是否存在权重”，属于不可部署oracle。',
            '可达率不是新模型准确率，也不是Bayes或其它架构上限。',
            '没有调温度、换pool、重训gate或新执行计划。'], size=9.5)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = page('逐题 TEST：新增支持域有多大？', 2)
        y = text(ax, .895, ['P=softmax概率池，L=原logits池；新增仅统计原硬bank未覆盖样本。',
            '两池对同一候选共用同一组三专家非负权重，权重和为1。'], size=9.5)
        rows = []
        for problem in saved.PROBLEMS:
            p, l = lookup['test', 'probability', problem], lookup['test', 'logits', problem]
            rows.append([problem, f"{l['actual_router_top1_pct']:.2f}",
                f"{l['hard_support_pct']:.2f}", l['hard_uncovered'],
                p['additional_reachable'], l['additional_reachable'], f"{p['boundary']}/{l['boundary']}"])
        y = table(ax, y, ['问题', '模型\nTop1 %', '硬支持\n%', '硬bank\n漏例', 'P新增\n例', 'L新增\n例', 'P/L\n边界例'],
                  rows, [1.9, 1.35, 1.25, 1.25, 1.1, 1.1, 1.1], height=.031, size=9)
        text(ax, y, ['每题4500例。仍不可达例数=硬bank漏例-该池新增例-该池边界例。',
            'VAL逐题、两池不可达/支配/同分分解均见 support_summary.csv。',
            '模型成绩来自原锁定模型，不把oracle权重当作新的预测。'], size=9.5)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = page('真实漏选：固定36组与 LEHD 等方法', 3)
        y = text(ax, .895, ['固定36组沿用原冻结VAL目标清单，不按本次TEST重新挑选。',
            '以下“新增”都是 native winner 的离线可达性，不是出现次数配额。'], size=9.5)
        rows = []
        for split in ('val', 'test'):
            p = [r for r in fixed if r['split'] == split and r['pool_mode'] == 'probability']
            l = [r for r in fixed if r['split'] == split and r['pool_mode'] == 'logits']
            rows.append([split.upper(), sum(r['native_winner_count'] for r in p),
                sum(r['missed_native_count'] for r in p), sum(r['hard_uncovered'] for r in p),
                sum(r['additional_reachable'] for r in p), sum(r['additional_reachable'] for r in l),
                f"{sum(r['boundary'] for r in p)}/{sum(r['boundary'] for r in l)}"])
        y = table(ax, y, ['固定36', 'native\nwinner例', '模型\n漏例', '硬bank\n漏例', 'P新增\n例', 'L新增\n例', 'P/L\n边界'],
                  rows, [1.2, 1.4, 1.2, 1.2, 1., 1., 1.1], height=.038)
        detail = {(r['pool_mode'], r['problem'], r['method']): r for r in methods if r['split'] == 'test'}
        keys = [('TSP', k) for k in ('LEHD', 'DIFUSCO', 'DIFUSCO500', 'OMNI')]
        remaining = [r for r in methods if r['split'] == 'test' and r['pool_mode'] == 'logits'
                     and (r['zero_selected_positive_native'] or r['near_zero_selected_positive_native'])
                     and (r['problem'], r['method']) not in keys]
        keys += [(r['problem'], r['method']) for r in sorted(remaining,
                    key=lambda r: (-r['missed_native_count'], r['problem'], r['method']))[:8]]
        rows = []
        for problem, method in keys:
            p, l = detail['probability', problem, method], detail['logits', problem, method]
            rows.append([problem + '/' + method, l['selected_count'], l['native_winner_count'],
                l['hard_uncovered'], p['additional_reachable'], l['additional_reachable'],
                f"{p['boundary']}/{l['boundary']}"])
        y = heading(ax, y, 'TEST：零/近零选择且有真实winner的重点组')
        y = table(ax, y, ['问题/方法', '模型\n选例', 'native\nwinner例', '硬bank\n漏例', 'P新增', 'L新增', 'P/L\n边界'],
                  rows, [3.5, .9, 1.2, 1.2, .85, .85, .95], height=.031, size=8.5)
        all_test = [r for r in methods if r['split'] == 'test' and r['pool_mode'] == 'logits']
        zero = sum(r['zero_selected_positive_native'] for r in all_test)
        empty = sum(r['empty_native_group'] for r in all_test)
        y = text(ax, y, [f"TEST共{zero}个“零选择且native支持为正”组；另有{empty}个native空winner组。",
            '空winner组不是漏选证据；FP64成本并列也不等于native标签相同。',
            'method_support.csv 单列唯一成本最优/成本并列的native支持；',
            'fixed36_support.csv保留全部36组和VAL/TEST，不删除合法候选。'], size=9.5)
        text(ax, y, ['本页按原模型选择数展示漏项，不把真实winner次数当模型选择次数。'], size=9)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = page('判定方法、证据与不能推出的结论', 4)
        y = heading(ax, .895, '3  有界离线几何，不训练新权重规则')
        y = text(ax, y, ['每个查询先检查各专家实际Top1；已覆盖由单纯形顶点作证。',
            '对漏例先找同一个竞争方法是否在三个专家坐标严格支配winner。',
            '其余用 scipy.linprog 求 max min_j 权重·(winner分数-竞争分数)。',
            '正primal margin验证可达；负rival混合dual bound验证不可达。',
            '概率池：原FP32 logits转FP64后稳定softmax，温度恒为1。',
            'logits池：只转FP64，不归一化专家尺度；两者不是同一个凸池。'], size=10)
        y = heading(ax, y, '4  同分和数值边界必须独立保留')
        y = text(ax, y, ['分数同分按最低global ID；native winner仍使用保存的int64标签。',
            '须严格胜过所有更低ID；对更高ID允许相等。margin接近0不自动可救。',
            '先按每例最大绝对分差缩放，分类容差1e-8，LP可行容差1e-9。',
            '边界额外LP只检查低ID分离；仅直接非负可行见证才接受同分可达。',
            '其它未证实样本保留boundary；LP失败也保留，不重试或扩大扫描。'], size=9.5)
        y = heading(ax, y, '5  来源、身份和开销')
        run = evidence['execution']
        y = text(ax, y, [f"模型唯一锁定update {evidence['checkpoint']['selected_update']}，完成7200更新；SHA256：",
            evidence['checkpoint']['checkpoint_sha256'],
            '三bank及VAL/TEST captures与已完成R123报告输入哈希逐文件一致；',
            '逐行source identity、pool/global ID、所选专家logits与native成本均验证。',
            f"复用162000查询/486000专家行，已有小数组约{run['existing_archive_bytes']/2**20:.1f} MiB；",
            f"1 CPU、OMP1，LP调用{run['lp_calls']}次；分析{run['analysis_wall_seconds']:.2f}s wall / {run['analysis_cpu_seconds']:.2f}s CPU。",
            '未做NN推理、solver、训练或GPU调用；既有三专家预计算并非免费，',
            '其原始GPU构建时间缺失，不能把本次CPU耗时当成部署总开销。'], size=9)
        y = heading(ax, y, '6  只解释支持域，不部署答案oracle')
        text(ax, y, ['TEST已开发暴露；当前诊断又使用真winner，不能称新盲测。',
            '来源记录证书不证明同base/内容副本完全独立。原VAL/TEST captures',
            '未嵌入checkpoint签名，本次沿用原报告锁定/精确数组归属，不补造签名。',
            '可达不等于可学习，更不等于会改善Top1或成本。没有新R125执行计划。',
            '完整哈希/容差/逐LP见证：evidence.json、lp_residual_cases.json。'], size=9.5)
        pdf.savefig(fig)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    analyze(args.root, args.output)


if __name__ == '__main__':
    main()
