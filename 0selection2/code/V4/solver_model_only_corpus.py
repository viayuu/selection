"""R83: neural model definitions only, never solver execution/configuration."""

import argparse
import ast
import copy
import json
from pathlib import Path

from code.unified_selector.registry import GLOBAL_SOLVERS
from .solver_code_corpus import (
    MANIFEST, ModelTokenizer, CleanAST, check_sensitive, digest, source_path, symbol_table, write_json,
)
from .solver_code_encoder import ROLES
from .solver_code_embeddings import assemble, generate, load_bundle
from .solver_code_views import CONTEXT_TOKENS, extract_components, pack_view, calls_in


ROOT = Path(__file__).resolve().parent / 'runs/R83_model_only_source_embedding'
VERSION = 'r83-model-only-ast-v1'
EXCLUDED_METHODS = {
    'training_step', 'categorical_training_step', 'gaussian_training_step', 'configure_optimizers',
    'set_decoder_strategy', 'set_decoder_type', 'set_eval_type', 'initialize', 'iteration',
    'one_step_rollout', 'select_next_node', 'load_basis_policy_weights', 'fix_basis_policy_weights',
    'lora_fixed_params', 'lora_trainable_params', 'load_lora_modules', 'save_numpy_heatmap',
}
SCOPE = dict(kind='model_only', active_roles=['encoder', 'decision'],
    excluded_roles=['inference', 'config'], no_neural_model_solvers=['GLOP'],
    excluded_content=['training loops/loss/optimizer', 'solver rollout/search/postprocessing',
                      'environment state transitions/masks', 'external configuration', 'bridge input conversion'],
    included_content='Neural encoder/decoder/layers, model forward and network cache helpers; model default arguments remain code',
    glop_reason='Actual labeled GLOP deployment is random insertion initialization, with no neural model; retain ID only')


def selection(file, *symbols):
    return dict(file=file, symbols=list(symbols) or ['*'])


def model_manifest(original):
    manifest = copy.deepcopy(original)
    manifest['schema'] = VERSION
    manifest['model_only_scope'] = SCOPE
    for name, impl in manifest['implementations'].items():
        for role in ('encoder', 'decision'):
            kept = []
            for entry in impl['views'][role].get('sources', []):
                ref = entry['file']
                if ref.startswith(('bridge:', 'insertion:')) or '/learning/' in ref or '/Diffusion/' in ref:
                    continue
                if ref.endswith('/base_methods.py'):
                    continue
                if ref.endswith('/matpoenet/utils.py'):
                    entry = selection(ref, 'make_positional_encoding_cosh_recur')
                if ref.endswith('/policy.py'):
                    continue
                kept.append(entry)
            impl['views'][role] = dict(status='present', sources=kept)
        for role in ('inference', 'config'):
            impl['views'][role] = dict(status='absent', sources=[], reason='Excluded by model-only scope')
    impls = manifest['implementations']
    impls['bq']['views']['encoder']['sources'] = [
        selection('sources:bq-nco/model/encoder.py'), selection('sources:bq-nco/model/attention.py')]
    impls['bq']['views']['decision']['sources'] = [selection('sources:bq-nco/model/model.py', 'BQModel')]
    gnn = 'easynco:neural_solvers/backbones/GNN/AGNNDifusco.py'
    for name in ('difusco', 't2t'):
        impls[name]['views']['encoder']['sources'] = [selection(gnn,
            'GNNLayer', 'PositionEmbeddingSine', 'ScalarEmbeddingSine', 'ScalarEmbeddingSine1D',
            'run_sparse_layer', 'normalize', 'zero_module')]
        impls[name]['views']['decision']['sources'] = [selection(gnn, 'GNNEncoder')]
    # The common backbone and input embeddings are network layers, not adapters.
    for name in ('elg', 'omni'):
        impls[name]['views']['encoder']['sources'] += [
            selection('easynco:neural_solvers/backbones/Transformer/attention.py'),
            selection('easynco:neural_solvers/utils/static_embedding.py', 'TSPInitEmbedding', 'VRPInitEmbedding')]
    impls['elg']['views']['encoder']['sources'] = [entry for entry in impls['elg']['views']['encoder']['sources']
                                                   if not entry['file'].endswith('/elg/policy.py')]
    impls['elg']['views']['decision']['sources'] += [selection(
        'easynco:neural_solvers/methods/elg/policy.py', 'local_policy_att_tsp', 'local_policy_att_cvrp')]
    impls['icam']['views']['encoder']['sources'] = [selection(
        'easynco:neural_solvers/methods/icam/icam_encoder.py',
        'CVRPICAMEncoder', 'EncoderLayer', 'adaptation_attention_free_module')]
    impls['icam']['views']['decision']['sources'] = [selection(
        'easynco:neural_solvers/methods/icam/icam_decoder.py', 'CVRPICAMDecoder')]
    unico = 'sources:UniCO/MatPOENet/ATSPModel.py'
    impls['unico_matpoenet']['views']['encoder']['sources'].append(selection(unico, 'PEEncodingBlock'))
    for name in ('routefinder', 'moses_rf', 'moses_cada'):
        prefix = 'sources:routefinder/routefinder' if name == 'routefinder' else 'sources:moses_vrp'
        policy_name = dict(routefinder='RouteFinderPolicy', moses_rf='MultiLoRAPolicy',
                           moses_cada='CadaMultiLoRAPolicy')[name]
        impls[name]['views']['decision']['sources'].append(selection(prefix + '/models/policy.py', policy_name))
    # Helpers in these original Model.py files perform the neural attention math.
    for name, file in (
        ('icam_atsp', 'sources:ICAM/ICAM_ATSP/ATSPModel_ICAM.py'),
        ('unico_matpoenet', unico),
        ('reld_cvrp', 'reld:CVRP/CVRPModel.py'),
        ('reld_moel', 'reld:Multi-Task/models/MOEModel_Light.py'),
        ('reld_mtl', 'reld:Multi-Task/models/MTLModel.py'),
    ):
        tree = ast.parse(source_path(manifest, file).read_text(encoding='utf-8-sig'))
        functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        if functions:
            impls[name]['views']['decision']['sources'].append(selection(file, *functions))
    for solver in manifest['solvers']:
        for deployment in solver['deployments']:
            # These values remain in original provenance, never model-only text or cache context.
            for key in ('resolved_config', 'assumed_config', 'assumption_sources'):
                deployment.pop(key, None)
    return manifest


def clean_components(source, symbols, reference, sha):
    components = extract_components(source, symbols, reference, sha)
    for component in components:
        node = component['_node']
        if isinstance(node, ast.ClassDef):
            node.body = [child for child in node.body if not (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in EXCLUDED_METHODS)]
            if not node.body:
                node.body = [ast.Pass()]
            component['aliases'] = [node.name] + [node.name + '.' + child.name for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
        component['text'] = ast.unparse(node)
        component['calls'] = calls_in(node)
    return components


def build(original, tokenizer):
    manifest = model_manifest(original)
    chunks, hashes, solvers, views, warnings = {}, {}, [], [], []
    for solver in manifest['solvers']:
        variants = []
        for deployment in solver['deployments']:
            roles = {role: [] for role in ROLES}
            if solver['solver_name'] != 'GLOP':
                impl = manifest['implementations'][deployment['implementation']]
                for role in SCOPE['active_roles']:
                    components = []
                    for entry in impl['views'][role]['sources']:
                        ref = entry['file']
                        path = source_path(manifest, ref)
                        raw = path.read_bytes()
                        hashes[ref] = digest(raw)
                        components += clean_components(raw.decode('utf-8-sig'), entry['symbols'], ref, hashes[ref])
                    if not components:
                        raise ValueError(f"No actual model code for {solver['solver_name']}/{role}")
                    packed, tokens = pack_view(components, tokenizer, role)
                    for part in packed:
                        key = digest(dict(text=part['text'], sources=part['provenance'], version=VERSION,
                                          tokenizer=tokenizer.fingerprint, max_tokens=CONTEXT_TOKENS))
                        check_sensitive(part['text'])
                        chunks.setdefault(key, dict(id=key, role=role, text_sha256=digest(part['text'].encode()),
                                                    config_context={}, **part))
                        roles[role].append(key)
                    roles[role] = list(dict.fromkeys(roles[role]))
                    views.append(dict(solver_id=solver['solver_id'], solver_name=solver['solver_name'],
                        role=role, full_view_tokens=tokens, chunk_ids=roles[role],
                        complete=len(packed) == 1 and packed[0]['complete_view']))
            variant = dict(id=digest(roles), roles=roles)
            if variant['id'] not in [v['id'] for v in variants]:
                variants.append(variant)
            if deployment.get('binding_status') != 'confirmed' or deployment.get('gaps'):
                warnings.append(dict(solver=solver['solver_name'], implementation=deployment['implementation'],
                    binding_status=deployment.get('binding_status'), gaps=deployment.get('gaps', [])))
        solvers.append(dict(solver_id=solver['solver_id'], solver_name=solver['solver_name'], variants=variants))
    corpus = dict(schema='r45-corpus-v2', preprocessing=VERSION, model='voyage-code-4', roles=list(ROLES),
        max_tokens=CONTEXT_TOKENS, tokenizer_sha256=tokenizer.fingerprint, manifest_sha256=digest(manifest),
        source_hashes=hashes, ready=True, issues=[], solvers=solvers, chunks=list(chunks.values()),
        semantic_views=views, api=dict(provider='mongodb_atlas', endpoint='https://ai.mongodb.com/v1/embeddings'),
        model_only_scope=SCOPE, historical_bindings_verified=not warnings, provenance_warnings=warnings,
        chunking='Whole neural-model role; AST/call-aware splitting only beyond 32000 tokens')
    corpus['sha256'] = digest(corpus)
    return corpus, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--stage', choices=('prepare', 'plan', 'embed', 'assemble', 'all'), default='prepare')
    parser.add_argument('--allow-upload', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=400000)
    parser.add_argument('--api-key-file', type=Path)
    args = parser.parse_args()
    corpus_path = args.root / 'model_only_corpus/corpus.json'
    if args.stage in ('prepare', 'all') or not corpus_path.exists():
        corpus, manifest = build(json.loads(args.manifest.read_text()), ModelTokenizer(args.tokenizer))
        write_json(corpus_path, corpus)
        write_json(corpus_path.with_name('source_snapshot.json'), manifest)
        text_dir = corpus_path.parent / 'texts'
        text_dir.mkdir(parents=True, exist_ok=True)
        for chunk in corpus['chunks']:
            (text_dir / (chunk['id'] + '.py.txt')).write_text(chunk['text'], encoding='utf-8')
    else:
        corpus = json.loads(corpus_path.read_text())
    if args.stage in ('embed', 'all') and not args.allow_upload:
        parser.error('Embedding requires explicit --allow-upload')
    if args.stage in ('prepare', 'plan', 'embed', 'all'):
        plan = generate(corpus, args.root / 'embedding_cache', args.stage in ('embed', 'all'),
                        args.max_tokens, args.api_key_file)
        write_json(args.root / 'embedding_plan.json', plan)
        print(json.dumps(plan, indent=2), flush=True)
    output = args.root / 'model_only_vectors.pt'
    if args.stage in ('assemble', 'all'):
        bundle = load_bundle(output) if output.exists() else assemble(corpus, args.root / 'embedding_cache', output)
        print(json.dumps(dict(output=str(output), shape=list(bundle['vectors'].shape),
            active_roles=bundle['role_mask'].sum(0).tolist(), model_only_scope=bundle['model_only_scope']), indent=2))


if __name__ == '__main__':
    main()
