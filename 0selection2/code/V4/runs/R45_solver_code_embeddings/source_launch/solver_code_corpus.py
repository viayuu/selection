"""Explicit source allowlist -> cleaned AST units -> bounded embedding requests.

No solver modules are imported or executed. Provenance and upload text are kept
separate: paths, checkpoint identities, and audit observations never enter text.
"""

import argparse
import ast
import copy
import hashlib
import json
import re
import textwrap
from pathlib import Path

from code.unified_selector.registry import GLOBAL_SOLVERS
from .solver_code_encoder import ROLES


MANIFEST = Path(__file__).with_name('solver_source_manifest.json')
ROOT = Path('code/V4/runs/R45_solver_code_embeddings')
PREPROCESS_VERSION = 'r45-ast-v1'
MAX_TOKENS = 4096
MODEL = 'voyage-code-4'
ASSUMPTION_POLICY = 'r45-explicit-defaults-v1'


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')
    temporary.replace(path)


class ModelTokenizer:
    def __init__(self, path=None):
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError('Install tokenizers in easynco; no approximate tokenizer is allowed') from error
        self.backend = Tokenizer.from_file(str(path)) if path else Tokenizer.from_pretrained('voyageai/' + MODEL)
        self.fingerprint = digest(self.backend.to_str().encode())

    def count(self, text):
        return len(self.backend.encode(text, add_special_tokens=True).ids)


def check_sensitive(text):
    patterns = (r'\b(?:sk|pa|voyage)-[A-Za-z0-9_-]{16,}',
                r'(?i)(?:api[_-]?key|secret|access[_-]?token|password)[\'\"]?\s*[=:]\s*[\'\"][^\'\"]{8,}[\'\"]',
                r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError('Potential credential found in allowlisted content; upload refused')


class CleanAST(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # Strip docstrings; comments disappear during AST parsing.
        if isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func).lower()
            if name == 'print' or any(part in name for part in ('logger.', 'logging.', '.save', '.write', '.dump')):
                return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        value = node.value
        if isinstance(value, str):
            if value.startswith(('/', '~/', 'http://', 'https://')) or re.search(r'\.(?:pt|pth|ckpt|pkl|npz|npy|log)$', value):
                return ast.copy_location(ast.Constant('<external-path>'), node)
            if value in ('cuda', 'cuda:0', 'cuda:1'):
                return ast.copy_location(ast.Constant('<device>'), node)
        return node


def symbol_table(tree, prefix=''):
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            result[name] = node
            if isinstance(node, ast.ClassDef):
                result.update(symbol_table(node, name + '.'))
    return result


def ast_units(source, selected):
    check_sensitive(source)
    tree = ast.parse(source)
    table = symbol_table(tree)
    if selected == ['*']:
        selected = [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    result = []
    for name in selected:
        if name not in table:
            raise ValueError(f'Missing explicitly selected symbol: {name}')
        original = table[name]
        cleaned = CleanAST().visit(copy.deepcopy(original))
        for node in ast.walk(cleaned):
            if hasattr(node, 'body') and not node.body:
                node.body = [ast.Pass()]
        # Split classes into complete methods, retaining the class signature.
        if isinstance(cleaned, ast.ClassDef):
            header = ast.unparse(ast.ClassDef(name=cleaned.name, bases=cleaned.bases, keywords=cleaned.keywords,
                                            body=[ast.Pass()], decorator_list=cleaned.decorator_list)).split('\n')[0]
            for child in cleaned.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    previous = table.get(name + '.' + child.name, original)
                    result.append(dict(symbol=name + '.' + child.name, start=previous.lineno, end=previous.end_lineno,
                                       text=header + '\n' + textwrap.indent(ast.unparse(child), '    ')))
            attributes = [n for n in cleaned.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if attributes:
                result.append(dict(symbol=name + '.__class__', start=original.lineno, end=original.end_lineno,
                                   text=header + '\n' + '\n'.join(ast.unparse(n) for n in attributes)))
        else:
            result.append(dict(symbol=name, start=original.lineno, end=original.end_lineno, text=ast.unparse(cleaned)))
    return result


def statement_parts(text):
    """Prefer whole AST statements; line splitting is only the oversized-statement fallback."""
    try:
        node = ast.parse(text).body[0]
        if isinstance(node, ast.ClassDef) and len(node.body) == 1:
            node = node.body[0]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature = copy.deepcopy(node)
            signature.body = [ast.Pass()]
            header = ast.unparse(signature).rsplit('\n', 1)[0] + '\n'
            return [header] + [ast.unparse(statement) + '\n' for statement in node.body]
    except (SyntaxError, IndexError):
        pass
    return text.splitlines(keepends=True)


def bounded_parts(text, tokenizer, limit=MAX_TOKENS, context=''):
    """Keep complete units when possible, split oversized units without dropping a tail."""
    prefix = context + '\n' if context else ''
    if tokenizer.count(prefix + text) <= limit:
        return [prefix + text]
    if tokenizer.count(prefix) >= limit - 64:
        raise ValueError('Configuration context leaves no space for code')
    # Prefer statement/line boundaries; an unusually long literal/expression is split last.
    header = text.split('\n', 1)[0][:300]
    prefix += 'Continuation of: ' + header + '\n'
    output, pending = [], ''
    pieces = []
    for statement in statement_parts(text):
        pieces.extend(statement.splitlines(keepends=True) if tokenizer.count(prefix + statement) > limit else [statement])
    for line in pieces:
        if tokenizer.count(prefix + pending + line) <= limit:
            pending += line
            continue
        if pending:
            output.append(prefix + pending)
            pending = ''
        while tokenizer.count(prefix + line) > limit:
            low, high = 1, len(line)
            while low < high:
                middle = (low + high + 1) // 2
                if tokenizer.count(prefix + line[:middle]) <= limit:
                    low = middle
                else:
                    high = middle - 1
            if tokenizer.count(prefix + line[:low]) > limit:
                raise ValueError('Cannot fit even one source character with this context')
            output.append(prefix + line[:low])
            line = line[low:]
        pending = line
    if pending:
        output.append(prefix + pending)
    if not all(tokenizer.count(part) <= limit for part in output):
        raise ValueError('Tokenizer limit violated')
    return output


def semantic_config(config):
    forbidden = {'costs', 'mean_cost', 'winner', 'oracle', 'gap', 'rankings', 'instance_id', 'global_index', 'test_results'}
    def check(value):
        if isinstance(value, dict):
            if forbidden.intersection(str(key).lower() for key in value):
                raise ValueError('Observed labels/results are not solver configuration input')
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, str) and (value.startswith(('/', '~/')) or re.search(r'\.(?:pt|pth|ckpt|pkl|npz)$', value)):
            raise ValueError('Checkpoint/data paths belong in provenance, not semantic configuration')
    check(config)
    text = json.dumps(config, sort_keys=True, indent=2)
    check_sensitive(text)
    return text


def pack_units(units, tokenizer, context):
    pending, output = '', []
    for unit in dict.fromkeys(units):
        for part in bounded_parts(unit, tokenizer, context=context):
            joined = pending + '\n\n' + part if pending else part
            if pending and tokenizer.count(joined) > MAX_TOKENS:
                output.append(pending)
                pending = part
            else:
                pending = joined
    if pending:
        output.append(pending)
    return output


def source_path(manifest, reference, extensions=('.py',)):
    root, relative = reference.split(':', 1)
    base = Path(manifest['roots'][root]).expanduser().resolve()
    path = (base / relative).resolve()
    if base not in path.parents or path.suffix not in extensions or any(x in path.parts for x in ('results', 'logs', 'data', 'datasets')):
        raise ValueError(f'Outside the Python source allowlist: {reference}')
    return path


def build_corpus(manifest, tokenizer, draft=False, allow_assumptions=False, full_views=False):
    from .solver_code_views import CONTEXT_TOKENS, VIEW_VERSION, extract_components, pack_view
    if [row['solver_name'] for row in manifest['solvers']] != GLOBAL_SOLVERS:
        raise ValueError('Manifest rows must follow the current GLOBAL_SOLVERS exactly')
    chunks, solvers, issues, source_hashes, semantic_views = {}, [], [], {}, []
    warnings, assumptions = [], []
    for solver_id, solver in enumerate(manifest['solvers']):
        if solver['solver_id'] != solver_id:
            raise ValueError('Manifest solver ID mismatch')
        if not solver['deployments']:
            raise ValueError('Every solver needs an explicitly documented deployment')
        variants = []
        for deployment_index, deployment in enumerate(solver['deployments']):
            impl = manifest['implementations'][deployment['implementation']]
            gaps = deployment.get('gaps', [])
            accepted = (allow_assumptions and deployment.get('assumption_policy') == ASSUMPTION_POLICY
                        and bool(deployment.get('assumption_sources')))
            if deployment['binding_status'] != 'confirmed' or gaps:
                warning = dict(solver=solver['solver_name'], deployment=deployment['implementation'],
                               deployment_index=deployment_index,
                               gaps=gaps or ['Historical source/config/checkpoint binding is unconfirmed'])
                (warnings if accepted else issues).append(warning)
            if deployment.get('assumed_config') and not accepted:
                issues.append(dict(solver=solver['solver_name'], role='config',
                                   gaps=['Assumed defaults require explicit --allow-assumptions approval']))
            if accepted:
                assumptions.append(dict(solver=solver['solver_name'], deployment_index=deployment_index,
                                        implementation=deployment['implementation'],
                                        historical_binding_status=deployment['binding_status'],
                                        assumed_config=deployment.get('assumed_config', {}),
                                        sources=deployment['assumption_sources']))
            config = copy.deepcopy(deployment.get('resolved_config', {}))
            if accepted:
                from .solver_code_defaults import fill_missing
                config = fill_missing(config, deployment.get('assumed_config', {}))[0]
            views = {}
            for role in ROLES:
                spec = impl['views'][role]
                if spec['status'] not in ('present', 'absent', 'missing'):
                    raise ValueError('Role status must be present/absent/missing')
                if spec['status'] == 'missing':
                    issues.append(dict(solver=solver['solver_name'], role=role, gaps=[spec.get('reason', 'Source is missing')]))
                if spec['status'] == 'absent':
                    if role in ROLES[:2] or not spec.get('reason'):
                        raise ValueError('A core role cannot be declared absent')
                    views[role] = []
                    continue
                units, provenance, components = [], [], []
                for selection in spec.get('sources', []):
                    path = source_path(manifest, selection['file'])
                    if not path.exists():
                        issues.append(dict(solver=solver['solver_name'], role=role, gaps=['Missing ' + selection['file']]))
                        continue
                    data = path.read_bytes()
                    source_hashes[selection['file']] = digest(data)
                    if full_views:
                        components.extend(extract_components(data.decode('utf-8-sig'), selection['symbols'],
                                                             selection['file'], digest(data)))
                        continue
                    extracted = ast_units(data.decode('utf-8-sig'), selection['symbols'])
                    for unit in extracted:
                        provenance.append(dict(file=selection['file'], sha256=digest(data),
                                               **{k: unit[k] for k in ('symbol', 'start', 'end')}))
                        units.append(unit['text'])
                if role == 'config' and config:
                    if full_views:
                        components.append(dict(symbol='resolved_configuration', text=semantic_config(config),
                                               calls=[], provenance=[]))
                    else:
                        units.append(semantic_config(config))
                if not (components if full_views else units):
                    issues.append(dict(solver=solver['solver_name'], role=role, gaps=['No verified content for required role']))
                role_chunks = []
                if full_views:
                    packed, full_tokens = pack_view(components, tokenizer, role)
                    for part in packed:
                        text = part['text']
                        check_sensitive(text)
                        key = digest(dict(text=text, sources=part['provenance'], version=VIEW_VERSION,
                                          tokenizer=tokenizer.fingerprint, max_tokens=CONTEXT_TOKENS))
                        chunks.setdefault(key, dict(id=key, role=role, text_sha256=digest(text.encode()),
                                                    config_context=config, **part))
                        role_chunks.append(key)
                    views[role] = list(dict.fromkeys(role_chunks))
                    semantic_views.append(dict(solver_id=solver_id, solver_name=solver['solver_name'],
                                               deployment_index=deployment_index, role=role,
                                               full_view_tokens=full_tokens, chunk_ids=views[role],
                                               complete=bool(packed) and len(packed) == 1 and packed[0]['complete_view']))
                    continue
                for text in pack_units(units, tokenizer, context='Solver implementation role: ' + role):
                    check_sensitive(text)
                    key = digest(dict(text=text, sources=provenance, version=PREPROCESS_VERSION,
                                      tokenizer=tokenizer.fingerprint, max_tokens=MAX_TOKENS))
                    chunks.setdefault(key, dict(id=key, text=text, role=role, tokens=tokenizer.count(text),
                                                text_sha256=digest(text.encode()), provenance=provenance,
                                                config_context=config))
                    role_chunks.append(key)
                views[role] = list(dict.fromkeys(role_chunks))
            variant_id = digest(views)  # Identical deployments do not acquire extra averaging weight.
            if variant_id not in [v['id'] for v in variants]:
                variants.append(dict(id=variant_id, roles=views))
        solvers.append(dict(solver_id=solver_id, solver_name=solver['solver_name'], variants=variants))
    result = dict(schema='r45-corpus-v1', preprocessing=PREPROCESS_VERSION, roles=list(ROLES), model=MODEL,
                  tokenizer_sha256=tokenizer.fingerprint, max_tokens=MAX_TOKENS,
                  manifest_sha256=digest(manifest), source_hashes=source_hashes,
                  ready=not issues, issues=issues, solvers=solvers, chunks=list(chunks.values()))
    if full_views:
        result.update(schema='r45-corpus-v2', preprocessing=VIEW_VERSION, max_tokens=CONTEXT_TOKENS,
                      semantic_views=semantic_views, api=dict(provider='mongodb_atlas', endpoint='https://ai.mongodb.com/v1/embeddings'),
                      chunking='Complete semantic role first; split only overflow at AST/call boundaries')
    if allow_assumptions:
        result.update(uses_assumptions=bool(assumptions), assumptions_accepted=True,
                      historical_bindings_verified=not warnings and not issues,
                      provenance_warnings=warnings, assumed_configurations=assumptions,
                      config_source_hashes=manifest.get('default_source_hashes', {}))
    result['sha256'] = digest(result)
    if issues and not draft:
        raise ValueError('Source bindings are incomplete; run --draft to inspect the exact gaps without uploading')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tokenizer', type=Path, help='Official voyage-code-4 tokenizer.json; otherwise download from its official Hugging Face repo')
    parser.add_argument('--draft', action='store_true', help='Extract real source, but mark unresolved deployment evidence as non-production')
    parser.add_argument('--allow-assumptions', action='store_true', help='Accept explicitly documented defaults for experimental representation, not historical verification')
    parser.add_argument('--legacy-chunks', action='store_true', help='Rebuild the historical 4096-token format, not the Atlas path')
    args = parser.parse_args()
    args.output = args.output or ROOT / ('corpus_legacy' if args.legacy_chunks else
                                        'corpus_draft_atlas' if args.draft else 'corpus_atlas')
    manifest = json.loads(args.manifest.read_text())
    corpus = build_corpus(manifest, ModelTokenizer(args.tokenizer), args.draft, args.allow_assumptions, full_views=not args.legacy_chunks)
    write_json(args.output / 'corpus.json', corpus)
    write_json(args.output / 'source_snapshot.json', manifest)
    lines = ['# R45 source coverage', '', f"Ready for upload: {corpus['ready']}", '']
    lines += [f"- {row['solver']} / {row.get('role', row.get('deployment', 'source'))}: " + '; '.join(row['gaps'])
              for row in corpus['issues']]
    if corpus.get('uses_assumptions'):
        lines += ['', 'Historical bindings verified: False', '',
                  'Ready means usable with explicitly accepted assumptions, not reproduced historical runs.', '',
                  '## Preserved Historical Gaps', '']
        lines += [f"- {row['solver']} / deployment {row['deployment_index']}: " + '; '.join(row['gaps'])
                  for row in corpus['provenance_warnings']]
    (args.output / 'missing_sources.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(solvers=len(corpus['solvers']), chunks=len(corpus['chunks']),
                          unique_tokens=sum(c['tokens'] for c in corpus['chunks']), ready=corpus['ready'],
                          output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
