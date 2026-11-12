"""R127 R83 model-only reports: remove only MVMOE, retain legal MoSES_RF.

After the unique VAL-best lock and runner TEST, ``lib`` evaluates the original
LIB inputs using r127_experiment.load_checkpoint. ``report`` reads saved
predictions only; ``all`` performs both. CPU is the default. NSS is an aligned
fixed-score, same-pool diagnostic, never retrained or inferred here.

R126 and its R125 dependency are read-only templates compiled into private
namespaces. No running experiment's files, globals, or archives are changed.
All eight B tasks have two candidates: Top2/Top3 saturate at min(k, K)=2.
"""

import ast
import hashlib
import json
from pathlib import Path
from types import ModuleType


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R127_R83_no_mvmoe'
ARM, ARMS, MAX_UPDATES = 'model_only_seed2', ('model_only_seed2',), 57600
FACTORY = 'code.V4.r127_experiment'
REMOVED = frozenset(('MVMOE',))
RETAINED = ['BQ', 'LEHD', 'MoSES_RF']
TOPK_NOTE = ('B tasks K=2: Top2/Top3 both cover all candidates (min(k,K)); '
             '100% is not extra ranking evidence.')


def _retarget(value):
    return (value.replace('R126_R83_no_moses_rf', 'R127_R83_no_mvmoe')
            .replace('R126', 'R127').replace('r126', 'r127')
            .replace('MoSES_RF', 'MVMOE').replace('non-B5 / B3', 'non-B5 / B2'))


class _Literals(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=_retarget(node.value)), node)
        return node


def _private_template():
    path = V4 / 'r126_reporting.py'
    source = path.read_bytes()
    tree = _Literals().visit(ast.parse(source, filename=str(path)))
    module = ModuleType('code.V4._r127_reporting_reuse')
    module.__file__, module.__package__ = str(path), 'code.V4'
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


_REUSE, TEMPLATE_SHA256 = _private_template()
_ENGINE = _REUSE._ENGINE
PROBLEMS, ORIGINAL_POOLS, GLOBAL_SOLVERS = _REUSE.PROBLEMS, _REUSE.ORIGINAL_POOLS, _REUSE.GLOBAL_SOLVERS
ROLE_POLICY, SOURCE_SCOPE = _REUSE.ROLE_POLICY, _REUSE.SOURCE_SCOPE
WINNER_RULE, NSS_DESCRIPTION = _REUSE.WINNER_RULE, _REUSE.NSS_DESCRIPTION
LIBS, LIB_ROWS, SPLIT_ROWS = _REUSE.LIBS, _REUSE.LIB_ROWS, _REUSE.SPLIT_ROWS
DEFAULT_NSS_ROOT = _REUSE.DEFAULT_NSS_ROOT
read_json, write_json, file_hash = _REUSE.read_json, _REUSE.write_json, _REUSE.file_hash
expected_pool = _REUSE.expected_pool
_template_build, _template_distribution = _REUSE.build_reports, _REUSE.distribution_figure

# The facade allows report-only captions without replacing any shared helper.
_shared_common = _REUSE.common
common = ModuleType('code.V4._r127_reporting_common')
common.__dict__.update(_shared_common.__dict__)
_REUSE.common = _ENGINE.common = common


def _page(helper, title, subtitle, model, lock, page):
    fig = _shared_common._page(helper, title, subtitle, model, lock, page)
    if '| VAL:' in title or '| TEST:' in title:
        fig.text(.045, .045, TOPK_NOTE, fontsize=8.5)
    return fig


common._page = _page


def validate_config(config):
    # The base gate checks all frozen row counts, stable IDs, aliases, the
    # unique R127 identity/budget, and reduced-pool FULL TRAIN SBS identities.
    _REUSE._validate_template(config)
    if (config.get('removed_candidates') != ['MVMOE']
            or config.get('retained_candidates') != RETAINED
            or config.get('role_policy') != ROLE_POLICY
            or config.get('source_scope') != SOURCE_SCOPE
            or config.get('model_only_scope', {}).get('no_neural_model_solvers') != ['GLOP']
            or config.get('model_params', {}).get('solver_representation') != 'code'):
        raise ValueError('R127 requires R83 model-only roles, GLOP ID-only, and only MVMOE deletion')
    for problem in PROBLEMS:
        expected = {'TSP': 8, 'CVRP': 9, 'ATSP': 5}.get(problem, 2 if 'B' in problem else 5)
        if len(config['candidate_pools'][problem]) != expected:
            raise ValueError('Unexpected R127 legal candidate count: ' + problem)


def pool_statistics(config):
    rows = []
    for problem in PROBLEMS:
        k = len(config['candidate_pools'][problem])
        rows.append(dict(problem=problem, candidate_count=k, structural_unordered_pairs=k * (k - 1) // 2,
            top2_effective_k=min(2, k), top3_effective_k=min(3, k),
            top2_all_candidates=k <= 2, top3_all_candidates=k <= 3,
            candidate_pool=json.dumps(config['candidate_pools'][problem]),
            note=TOPK_NOTE if k == 2 else 'Top-k native winner coverage uses min(k,K).'))
    return rows


def distribution_figure(problem, pool, panels, version, model, lock, *, diagnostic=False):
    fig = _template_distribution(problem, pool, panels, version, model, lock, diagnostic=diagnostic)
    if not diagnostic and problem != 'ATSP':
        fig.texts[-1].set_text('R127: only MVMOE deleted; BQ/LEHD and originally legal MoSES_RF retained; same VAL lock.')
    return fig


def build_reports(run, config, lock, test, val, predictions, metrics, methods,
                  comparisons, alignment, nss_predictions=None):
    result = _template_build(run, config, lock, test, val, predictions, metrics, methods,
        comparisons, alignment, nss_predictions)
    output = Path(result['output'])
    protocol = read_json(output / 'protocol.json')
    protocol.update(retained_candidates=list(RETAINED),
        retained_candidate_scope='Only in originally legal pools; the eight B-task filters are not reversed.',
        changed_task_pools=['CVRP'] + list(PROBLEMS[3:]),
        two_candidate_problems=[problem for problem in PROBLEMS if len(config['candidate_pools'][problem]) == 2],
        topk_definition='Native winner in score rank[:min(k,K)]; K=2 makes Top2 and Top3 100% by construction.',
        pairwise_definition='K*(K-1)/2 structural unordered pairs; K=2 has one; exact cost ties are not valid comparison labels.',
        candidate_pool_statistics=pool_statistics(config),
        report_template=dict(path=str(V4 / 'r126_reporting.py'), sha256=TEMPLATE_SHA256,
            isolation='private modules and common facade; no shared globals or file edits'),
        report_dependencies=[dict(path=str(V4 / 'r126_reporting.py'), sha256=TEMPLATE_SHA256),
            dict(path=str(V4 / 'r125_reporting.py'), sha256=_REUSE.TEMPLATE_SHA256)],
        reporting_source_sha256=file_hash(Path(__file__)))
    write_json(output / 'protocol.json', protocol)
    common.write_csv(output / 'candidate_pool_statistics.csv', pool_statistics(config))
    summary = output / 'RESULTS.md'
    with summary.open('a', encoding='utf-8') as stream:
        stream.write('\nBQ/LEHD and originally legal MoSES_RF are retained; only MVMOE is removed.\n' + TOPK_NOTE + '\n')
    return result


_REUSE.validate_config = _ENGINE.validate_config = validate_config
_REUSE.distribution_figure = _ENGINE.distribution_figure = distribution_figure
_REUSE.build_reports = _ENGINE.build_reports = build_reports
_ENGINE.DEFAULT_ROOT, _ENGINE.ARM, _ENGINE.ARMS = DEFAULT_ROOT, ARM, ARMS
_ENGINE.FACTORY, _ENGINE.REMOVED, _ENGINE.__doc__ = FACTORY, REMOVED, __doc__

bind_selected_val, locked_inputs = _REUSE.bind_selected_val, _REUSE.locked_inputs
validate_prediction, load_saved_splits = _REUSE.validate_prediction, _REUSE.load_saved_splits
reduced_raw, native_lib, lib_loader = _REUSE.reduced_raw, _REUSE.native_lib, _REUSE.lib_loader
evaluate_libs, lib_stage, saved_libs = _REUSE.evaluate_libs, _REUSE.lib_stage, _REUSE.saved_libs
reselect_nss, nss_comparisons = _REUSE.reselect_nss, _REUSE.nss_comparisons
make_panels, learning_figure = _REUSE.make_panels, _REUSE.learning_figure
report, parser = _REUSE.report, _REUSE.parser


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
