"""Prepare source-backed assumed defaults without rewriting historical evidence."""

import argparse
import ast
import copy
import json
from pathlib import Path

import yaml

from .solver_code_corpus import (ASSUMPTION_POLICY, MANIFEST, ROOT, build_corpus, digest,
                                ModelTokenizer, semantic_config, source_path, symbol_table, write_json)


ALIASES = (
    ('embedding_dim', 'embed_dim', 'dim_emb'),
    ('ff_hidden_dim', 'feedforward_hidden', 'dim_ff'),
    ('encoder_layer_num', 'num_encoder_layers', 'nb_layers_encoder'),
    ('head_num', 'num_heads', 'nb_heads'),
)

# Only representation-relevant fields are read; training/data/device options are excluded.
PROFILES = {
    'bq': ('sources:bq-nco/args.py', 'arguments', 'add_common_args'),
    'difusco': ('easynco:configs_old/tsp_difusco_config.yaml', 'yaml', 'model'),
    't2t': ('easynco:configs_old/tsp_t2t_config.yaml', 'yaml', 'model'),
    'elg': ('easynco:settings/elg_settings.yaml', 'yaml', 'model'),
    'lehd': ('easynco:neural_solvers/methods/lehd/policy.py', 'constructor', 'LEHDPolicy.__init__'),
    'omni': ('easynco:neural_solvers/methods/omni/policy.py', 'constructor', 'OMNI_POMO_Policy.__init__'),
    'icam': ('easynco:neural_solvers/methods/icam/policy.py', 'constructor', 'CVRPICAMPolicy.__init__'),
    'mvmoe': ('easynco:settings/mvmoe_settings.yaml', 'yaml', 'model'),
    'routefinder': ('sources:routefinder/routefinder/models/policy.py', 'constructor', 'RouteFinderPolicy.__init__'),
    'moses_rf': ('sources:moses_vrp/models/policy.py', 'constructor', 'MultiLoRAPolicy.__init__'),
    'moses_cada': ('sources:moses_vrp/models/policy.py', 'constructor', 'CadaMultiLoRAPolicy.__init__'),
}
ARGUMENT_FIELDS = {'dim_emb', 'dim_ff', 'nb_layers_encoder', 'nb_heads', 'dropout', 'batchnorm',
                   'activation_ff', 'activation_attention', 'beam_size', 'knns'}
OMIT = {'self', 'phase', 'env_name', 'problem_type', 'use_activation_checkpoint', 'node_dim',
        'sdpa_fn', 'check_nan', 'extra_encoder_kwargs'}


def fill_missing(known, defaults):
    """Preserve recorded values, including equivalent parameter names and nested keys."""
    result, added = copy.deepcopy(known), {}
    for key, value in defaults.items():
        aliases = next((group for group in ALIASES if key in group), (key,))
        existing = next((name for name in aliases if name in result), None)
        if existing is None:
            result[key], added[key] = copy.deepcopy(value), copy.deepcopy(value)
        elif isinstance(result[existing], dict) and isinstance(value, dict):
            result[existing], extra = fill_missing(result[existing], value)
            if extra:
                added[existing] = extra
    return result, added


def literal_defaults(source, kind, symbol):
    tree = ast.parse(source)
    node = symbol_table(tree)[symbol]
    if kind == 'arguments':
        result = {}
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != 'add_argument':
                continue
            if not call.args:
                continue
            name = ast.literal_eval(call.args[0]).lstrip('-')
            keywords = {kw.arg: kw.value for kw in call.keywords}
            name = ast.literal_eval(keywords['dest']) if 'dest' in keywords else name
            if name not in ARGUMENT_FIELDS:
                continue
            if 'default' in keywords:
                result[name] = ast.literal_eval(keywords['default'])
            elif 'action' in keywords and ast.literal_eval(keywords['action']) == 'store_true':
                result[name] = False
        return result
    names = node.args.posonlyargs + node.args.args
    values = list(zip(names[len(names) - len(node.args.defaults):], node.args.defaults))
    values += list(zip(node.args.kwonlyargs, node.args.kw_defaults))
    result = {}
    for arg, value in values:
        if value is None or arg.arg in OMIT:
            continue
        try:
            decoded = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        if decoded is not None and not arg.arg.endswith(('_embedding', '_path')):
            result[arg.arg] = decoded
    return result


def read_profile(manifest, profile):
    reference, kind, symbol = profile
    path = source_path(manifest, reference, ('.py', '.yaml', '.yml'))
    data = path.read_bytes()
    if kind == 'yaml':
        values = yaml.safe_load(data)[symbol]
        values = {key: value for key, value in values.items()
                  if not key.startswith('_') and key not in OMIT and '${' not in str(value)}
    else:
        values = literal_defaults(data.decode('utf-8-sig'), kind, symbol)
    semantic_config(values)
    return values, dict(file=reference, sha256=digest(data), selector=symbol,
                        basis='current source defaults' if kind != 'yaml' else 'checked-in problem-specific template')


def prepare_defaults(manifest):
    result, rows, hashes = copy.deepcopy(manifest), [], {}
    for solver in result['solvers']:
        for index, deployment in enumerate(solver['deployments']):
            implementation = deployment['implementation']
            defaults, sources = {}, []
            if implementation in PROFILES:
                defaults, evidence = read_profile(manifest, PROFILES[implementation])
                sources.append(evidence)
            if implementation in ('mtpomo', 'mvmoe'):
                values, evidence = read_profile(manifest, ('easynco:phases/rl/ar_reinforce.py',
                                                          'constructor', 'ARREINFORCELightning.__init__'))
                defaults['decoder_strategy'] = values['decoder_strategy']
                sources.append(evidence)
            if not sources:
                selected = manifest['implementations'][implementation]['views']['encoder']['sources'][0]
                path = source_path(manifest, selected['file'])
                sources.append(dict(file=selected['file'], sha256=digest(path.read_bytes()),
                                    selector=selected['symbols'], basis='current implementation; historical version unknown'))
            _, assumed = fill_missing(deployment.get('resolved_config', {}), defaults)
            deployment.update(assumed_config=assumed, assumption_policy=ASSUMPTION_POLICY,
                              assumption_sources=sources)
            for source in sources:
                hashes[source['file']] = source['sha256']
            rows.append(dict(solver=solver['solver_name'], deployment_index=index,
                             implementation=implementation, assumed_config=assumed, sources=sources,
                             preserved_gaps=deployment.get('gaps', [])))
    result['default_source_hashes'] = hashes
    result['representation_policy'] = dict(name=ASSUMPTION_POLICY, base_manifest_sha256=digest(manifest),
                                         historical_bindings_verified=False,
                                         statement='Experimental source representation uses explicitly assumed defaults; historical gaps remain unknown.')
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, default=ROOT / 'corpus_atlas')
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--legacy-chunks', action='store_true', help='Use the historical 4096-token preparation mode')
    args = parser.parse_args()
    original = json.loads(args.manifest.read_text())
    derived, rows = prepare_defaults(original)
    corpus = build_corpus(derived, ModelTokenizer(args.tokenizer), allow_assumptions=True, full_views=not args.legacy_chunks)
    write_json(args.output / 'source_manifest.json', derived)
    write_json(args.output / 'source_snapshot.json', derived)
    write_json(args.output / 'corpus.json', corpus)
    write_json(args.output / 'assumed_defaults.json', rows)
    lines = ['# R45 Explicit Default Assumptions', '', 'Ready for embedding: True',
             'Historical bindings verified: False', '',
             'Recorded values and historical gaps were preserved. No training/API/solver run was performed.', '',
             'Only missing semantic parameters use the following defaults; revisions, hashes and RNG history are not invented.', '']
    for row in rows:
        lines += [f"## {row['solver']} / deployment {row['deployment_index']}", '',
                  'Assumed values (empty means only the historical binding is waived):', '',
                  '```json', json.dumps(row['assumed_config'], indent=2), '```', '', 'Default sources:']
        lines += ['- ' + source['file'] + ' / ' + str(source['selector']) for source in row['sources']]
        lines += ['', 'Preserved historical gaps:'] + ['- ' + gap for gap in row['preserved_gaps']] + ['']
    (args.output / 'assumed_defaults.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(solvers=len(corpus['solvers']), deployments=len(rows),
                          deployments_with_default_fields=sum(bool(row['assumed_config']) for row in rows),
                          chunks=len(corpus['chunks']), tokens=sum(chunk['tokens'] for chunk in corpus['chunks']),
                          ready=corpus['ready'], historical_bindings_verified=False, output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
