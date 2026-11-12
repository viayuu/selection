"""R126 R83 model-only reporting, with MoSES_RF removed and BQ/LEHD retained.

After the runner's unique VAL lock and full TEST, use ``lib``, then ``report``
or ``all``. Defaults are CPU, model_only_seed2, and runs/R126_R83_no_moses_rf.
Only LIB inference loads a model, exclusively r126_experiment.load_checkpoint.
NSS only reuses aligned saved scores; it never trains or runs inference.

R125 is a read-only implementation dependency. Its functions are compiled in a
private module, with literal captions retargeted and R126-specific hooks below.
No R125 module globals, files, archives, or running processes are modified.
"""

import ast
import hashlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R126_R83_no_moses_rf'
ARM, ARMS, MAX_UPDATES = 'model_only_seed2', ('model_only_seed2',), 57600
FACTORY = 'code.V4.r126_experiment'
REMOVED = frozenset(('MoSES_RF',))
ROLE_POLICY = dict(encoder=True, decision=True, inference=False, config=False)
SOURCE_SCOPE = 'solver MODEL nn.Module code only; no config/train/solver inference/environment code'
WINNER_RULE = ('native FP64 argmin in the legal pool after removing MoSES_RF where present; '
               'exact ties use ascending global solver ID; not FP32 argmin')
NSS_DESCRIPTION = ('Fixed historical NSS scores restricted to the same legal pool; '
                   'TSP unchanged8 / CVRP9; NOT a matched R126 training control; '
                   'no retraining or new inference')


def _caption(value):
    exact = {
        'TSP6 / CVRP8; other16 task pools unchanged.':
            'TSP8 / CVRP9 / ATSP5 / non-B5 / B3; BQ and LEHD retained.',
        'Frozen new90k, BQ/LEHD deleted; FP64 reduced-pool argmin; no TEST/LIB reselection.':
            'R83 model-only source; MoSES_RF removed; BQ/LEHD retained; same VAL lock; FP64 argmin.',
        'R80D: architecture reference only; older data / original pool, NOT a matched experimental control.':
            'R83 model-only architecture reference: older data / original pool, NOT a matched training control.',
        'TRAIN-only normalization restored from the same best.pt; no LIB refit.':
            'encoder+decision active; inference/config absent; GLOP ID-only. TRAIN buffers; no LIB refit.',
    }
    if value in exact:
        return exact[value]
    return (value.replace('R125_no_bq_lehd_source_embedding', 'R126_R83_no_moses_rf')
            .replace('R125', 'R126').replace('r125_experiment', 'r126_experiment')
            .replace('r125_reporting', 'r126_reporting').replace('code_seed2', ARM)
            .replace('BQ/LEHD', 'MoSES_RF'))


class _Captions(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=_caption(node.value)), node)
        return node


def _private_template():
    path = V4 / 'r125_reporting.py'
    source = path.read_bytes()
    tree = _Captions().visit(ast.parse(source, filename=str(path)))
    module = ModuleType('code.V4._r126_reporting_template')
    module.__file__, module.__package__ = str(path), 'code.V4'
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    module.DEFAULT_ROOT, module.ARM, module.ARMS = DEFAULT_ROOT, ARM, ARMS
    module.FACTORY, module.REMOVED = FACTORY, REMOVED
    module.WINNER_RULE, module.NSS_DESCRIPTION = WINNER_RULE, NSS_DESCRIPTION
    return module, hashlib.sha256(source).hexdigest()


_ENGINE, TEMPLATE_SHA256 = _private_template()
common = _ENGINE.common
read_json, write_json, file_hash = _ENGINE.read_json, _ENGINE.write_json, _ENGINE.file_hash
LIBS, LIB_ROWS, SPLIT_ROWS = _ENGINE.LIBS, _ENGINE.LIB_ROWS, _ENGINE.SPLIT_ROWS
PROBLEMS, ORIGINAL_POOLS, GLOBAL_SOLVERS = _ENGINE.PROBLEMS, _ENGINE.ORIGINAL_POOLS, _ENGINE.GLOBAL_SOLVERS
DEFAULT_NSS_ROOT = _ENGINE.DEFAULT_NSS_ROOT
expected_pool = _ENGINE.expected_pool
_validate_template, _compare_template = _ENGINE.validate_config, _ENGINE.nss_comparisons
_build_template, _distribution_template = _ENGINE.build_reports, _ENGINE.distribution_figure


def validate_config(config):
    _validate_template(config)
    if (config.get('removed_candidates') != ['MoSES_RF']
            or config.get('retained_candidates') != ['BQ', 'LEHD']
            or config.get('role_policy') != ROLE_POLICY
            or config.get('source_scope') != SOURCE_SCOPE
            or config.get('model_only_scope', {}).get('no_neural_model_solvers') != ['GLOP']
            or config.get('model_params', {}).get('solver_representation') != 'code'):
        raise ValueError('R126 requires R83 model-only source roles and the GLOP ID-only exception')
    counts = {'TSP': 8, 'CVRP': 9, 'ATSP': 5}
    for problem in PROBLEMS:
        expected = counts.get(problem, 3 if 'B' in problem else 5)
        if len(config['candidate_pools'][problem]) != expected:
            raise ValueError('Unexpected R126 pool size: ' + problem)


def reselect_nss(archive, path, current):
    """Allow unchanged TSP8 as well as CVRP10-to9, never R125's TSP6/CVRP8."""
    helper = common.reporting_helpers()
    pool, costs, indices = archive['pool'].tolist(), archive['costs'], archive['indices']
    scores = archive['logits'] if 'logits' in archive else archive['scores']
    if (len(pool) != len(set(pool)) or not set(current.pool).issubset(pool)
            or not (set(pool) - set(current.pool)).issubset(REMOVED)
            or costs.dtype != np.float64 or costs.shape != (current.n, len(pool))
            or scores.shape != costs.shape or not np.isfinite(costs).all()
            or np.any(costs <= 0) or not np.isfinite(scores).all()
            or indices.shape != (current.n,) or len(set(indices.tolist())) != current.n
            or set(indices.tolist()) != set(current.indices.tolist())):
        raise ValueError('NSS requires same-population scores with every R126 candidate, including BQ/LEHD')
    lookup = {int(index): row for row, index in enumerate(indices)}
    rows = np.asarray([lookup[int(index)] for index in current.indices])
    columns = [pool.index(name) for name in current.pool]
    retained_costs, retained_scores = costs[rows][:, columns], scores[rows][:, columns]
    np.testing.assert_allclose(retained_costs, current.costs, rtol=0, atol=1e-12,
        err_msg='NSS and R126 must use the same frozen rows and original FP64 costs')
    release = getattr(current, 'release_source_indices', None)
    if release is not None:
        if 'source_indices' not in archive:
            raise ValueError('NSS is missing frozen release source IDs')
        np.testing.assert_array_equal(archive['source_indices'][rows], release)
    order = np.argsort(current.pool_ids, kind='stable')
    pred = order[np.argmax(retained_scores[:, order], axis=1)]
    return helper.Predictions(path=path, pool=list(current.pool), pred=pred,
        winner=retained_costs.argmin(1).astype(np.int64), costs=retained_costs,
        indices=current.indices.copy(), source_indices=None, pool_ids=current.pool_ids.copy(),
        score_ties=int(((retained_scores == retained_scores.max(1, keepdims=True)).sum(1) > 1).sum()))


def nss_comparisons(config, lock, predictions, nss_root=DEFAULT_NSS_ROOT):
    rows, alignment, diagnostics = _compare_template(config, lock, predictions, nss_root)
    for row in rows:
        if row['nss_top1_native_pct'] is not None:
            check = alignment[row['dataset']]
            removed = sorted(set(check['original_candidate_pool']) - set(check['reduced_candidate_pool']))
            row.update(status='fixed_NSS_same_pool_diagnostic', nss_removed_candidates=json.dumps(removed),
                nss_pool_changed=bool(removed), nss_training_matched=False)
            check['removed_candidates'] = removed
        else:
            row.update(nss_training_matched=False, nss_pool_changed=None)
    return rows, alignment, diagnostics


def distribution_figure(problem, pool, panels, version, model, lock, *, diagnostic=False):
    fig = _distribution_template(problem, pool, panels, version, model, lock, diagnostic=diagnostic)
    if not diagnostic:
        fig.texts[-1].set_text('R126: BQ/LEHD retained; MoSES_RF removed where present; same VAL lock; no Rxx mixing.'
            if problem != 'ATSP' else 'R126 ATSP: GLOP has no source roles; original trainable solver ID only.')
    return fig


def build_reports(run, config, lock, test, val, predictions, metrics, methods,
                  comparisons, alignment, nss_predictions=None):
    result = _build_template(run, config, lock, test, val, predictions, metrics, methods,
        comparisons, alignment, nss_predictions)
    output = Path(result['output'])
    protocol = read_json(output / 'protocol.json')
    protocol.update(retained_candidates=['BQ', 'LEHD'], architecture_reference='R83 model-only source embedding',
        source_scope=config['source_scope'], role_policy=config['role_policy'],
        model_only_scope=config['model_only_scope'],
        role_policy_exception='GLOP: all source roles absent; original trainable solver ID only',
        candidate_counts={problem: len(pool) for problem, pool in config['candidate_pools'].items()},
        changed_task_pools=['CVRP'] + [problem for problem in PROBLEMS[3:] if 'B' not in problem],
        historical_reference='R83 older-data model: architecture reference, NOT a matched training control',
        nss_training_matched=False, report_template=dict(path=str(V4 / 'r125_reporting.py'),
            sha256=TEMPLATE_SHA256, isolation='private module; no shared globals or file edits'),
        reporting_source_sha256=file_hash(Path(__file__)))
    write_json(output / 'protocol.json', protocol)
    return result


_ENGINE.validate_config = validate_config
_ENGINE.reselect_nss = reselect_nss
_ENGINE.nss_comparisons = nss_comparisons
_ENGINE.distribution_figure = distribution_figure
_ENGINE.build_reports = build_reports

# These functions resolve every hook and constant in the private namespace.
bind_selected_val, locked_inputs = _ENGINE.bind_selected_val, _ENGINE.locked_inputs
validate_prediction, load_saved_splits = _ENGINE.validate_prediction, _ENGINE.load_saved_splits
reduced_raw, native_lib, lib_loader = _ENGINE.reduced_raw, _ENGINE.native_lib, _ENGINE.lib_loader
evaluate_libs, lib_stage, saved_libs = _ENGINE.evaluate_libs, _ENGINE.lib_stage, _ENGINE.saved_libs
make_panels, learning_figure = _ENGINE.make_panels, _ENGINE.learning_figure
report, parser = _ENGINE.report, _ENGINE.parser


def main(argv=None):
    args = parser().parse_args(argv)
    args.root, args.nss_root = args.root.resolve(), args.nss_root.resolve()
    if args.command == 'status':
        run = args.root / args.arm
        result = {name: read_json(run / name) if (run / name).is_file() else {'phase': 'not_started'}
                  for name in ('lib_stage_status.json', 'postprocess_status.json')}
    elif args.command == 'lib':
        result = lib_stage(args)
    elif args.command == 'report':
        result = report(args)
    else:
        lib_stage(args)
        result = report(args)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    main()
