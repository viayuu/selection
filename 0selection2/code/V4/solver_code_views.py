"""Whole-role inputs; AST/call-aware splitting only at the actual model limit."""

import ast
import copy

from .solver_code_corpus import CleanAST, check_sensitive, digest, symbol_table


CONTEXT_TOKENS = 32000
VIEW_VERSION = 'r45-semantic-views-v2'


def calls_in(node):
    return sorted({ast.unparse(child.func) for child in ast.walk(node) if isinstance(child, ast.Call)})


def extract_components(source, selected, reference, sha):
    check_sensitive(source)
    tree = ast.parse(source)
    table = symbol_table(tree)
    if selected == ['*']:
        selected = [n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    result = []
    for name in selected:
        if name not in table:
            raise ValueError(f'Missing explicitly selected symbol: {name}')
        if any(name.startswith(parent + '.') and isinstance(table[parent], ast.ClassDef)
               for parent in selected if parent != name and parent in table):
            continue
        original = table[name]
        node = CleanAST().visit(copy.deepcopy(original))
        for child in ast.walk(node):
            if hasattr(child, 'body') and not child.body:
                child.body = [ast.Pass()]
        aliases = [name]
        if isinstance(node, ast.ClassDef):
            aliases += [name + '.' + child.name for child in node.body
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
        result.append(dict(symbol=name, origin_symbol=name, aliases=aliases, text=ast.unparse(node), calls=calls_in(node),
                           provenance=[dict(file=reference, sha256=sha, symbol=name,
                                            start=original.lineno, end=original.end_lineno)], _node=node))
    return result


def render(component):
    return '# Component: ' + component['symbol'] + '\n' + component['text']


def call_relations(components):
    relations = []
    for index, component in enumerate(components):
        for target, other in enumerate(components):
            if index == target or component.get('origin_symbol', component['symbol']) == other.get('origin_symbol', other['symbol']):
                continue
            names = set(other.get('aliases', [other.get('origin_symbol', other['symbol'])]))
            names.update(name.rsplit('.', 1)[-1] for name in list(names))
            if any(call in names or call.rsplit('.', 1)[-1] in names for call in component['calls']):
                relations.append((index, target))
    return relations


def dependency_order(components):
    """Keep selected callers/callees together; do not recursively collect new files."""
    edges = call_relations(components)
    adjacency = {i: [] for i in range(len(components))}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)
    output, visited = [], set()

    for i in range(len(components)):
        stack = [i]
        while stack:
            index = stack.pop()
            if index in visited:
                continue
            visited.add(index)
            output.append(components[index])
            stack.extend(sorted(set(adjacency[index]), reverse=True))
    return output


def ast_fragments(node):
    """Split statement blocks while retaining enclosing signatures/control flow."""
    children = [(field, index, child) for field, value in ast.iter_fields(node) if isinstance(value, list)
                for index, child in enumerate(value) if isinstance(child, (ast.stmt, ast.ExceptHandler))
                or type(child).__name__ == 'match_case']
    if not children:
        return []
    base = copy.deepcopy(node)
    for field, value in ast.iter_fields(base):
        if not isinstance(value, list) or not value:
            continue
        if field in ('body', 'orelse', 'finalbody'):
            setattr(base, field, [ast.Pass()])
        elif field in ('handlers', 'cases'):
            for item in value:
                item.body = [ast.Pass()]
    output = []
    for field, index, child in children:
        fragments = [(child, child)] if len(children) > 1 else ast_fragments(child)
        for fragment, leaf in fragments:
            wrapper = copy.deepcopy(base)
            if field in ('handlers', 'cases'):
                getattr(wrapper, field)[index] = copy.deepcopy(fragment)
            else:
                setattr(wrapper, field, [copy.deepcopy(fragment)])
            output.append((wrapper, leaf))
    return output


def split_component(component, tokenizer, prefix, limit):
    if tokenizer.count(prefix + render(component)) <= limit:
        return [component]
    node = component.get('_node')
    fragments = ast_fragments(node) if node is not None else []
    if not fragments:
        raise ValueError(f"Atomic statement {component['symbol']} exceeds {limit} tokens; no character truncation is allowed")
    output = []
    for index, (wrapper, leaf) in enumerate(fragments):
        text = ast.unparse(wrapper)
        symbol = component['symbol'] + f'.part_{index + 1}'
        provenance = [dict(p, symbol=symbol, start=getattr(leaf, 'lineno', p['start']),
                           end=getattr(leaf, 'end_lineno', p['end'])) for p in component['provenance']]
        part = dict(component, symbol=symbol, text=text, calls=calls_in(wrapper), provenance=provenance, _node=wrapper)
        output.extend(split_component(part, tokenizer, prefix, limit))
    return output


def call_context(components):
    # Catalog complete callable names before any split; fragment names never
    # replace the identities that appear at call sites.
    catalog = []
    for component in components:
        node = component.get('_node')
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = component['symbol'] + '.' + child.name
                    catalog.append(dict(component, symbol=name, origin_symbol=name, aliases=[name], calls=calls_in(child)))
        else:
            catalog.append(component)
    pairs = sorted({(catalog[i]['symbol'], catalog[j]['symbol']) for i, j in call_relations(catalog)})
    return '\n'.join(i + ' -> ' + j for i, j in pairs) or '(none)'


def class_context(components, tokenizer, header, limit):
    context = []
    for component in components:
        node = component.get('_node')
        if not isinstance(node, ast.ClassDef) or tokenizer.count(header + render(component)) <= limit:
            continue
        summary = copy.deepcopy(node)
        for child in summary.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name != '__init__':
                child.body = [ast.Pass()]
        context.append(ast.unparse(summary))
    return '\n\n'.join(context)


def pack_view(components, tokenizer, role, limit=CONTEXT_TOKENS):
    # Source aliases can select the same class repeatedly; include it only once.
    unique = {}
    for component in components:
        key = digest(dict(text=component['text'], provenance=component['provenance']))
        unique.setdefault(key, component)
    components = list(unique.values())
    if not components:
        return [], 0
    header = 'Solver implementation role: ' + role + '\n\n'
    whole = header + '\n\n'.join(render(c) for c in components)
    full_tokens = tokenizer.count(whole)
    groups = [components]
    if full_tokens > limit:
        prefix = header + 'Semantic continuation; selected call relations:\n' + call_context(components) + '\n\n'
        initialization = class_context(components, tokenizer, header, limit)
        if initialization:
            prefix += 'Shared class initialization/signatures:\n' + initialization + '\n\n'
        if tokenizer.count(prefix) >= limit:
            raise ValueError('Shared call/class context exceeds the model limit; narrow the source selection')
        parts = []
        for component in components:
            parts.extend(split_component(component, tokenizer, prefix, limit))
        parts = dependency_order(parts)
        groups, pending = [], []
        for component in parts:
            candidate = pending + [component]
            text = prefix + '\n\n'.join(render(c) for c in candidate)
            if pending and tokenizer.count(text) > limit:
                groups.append(pending)
                pending = [component]
            else:
                pending = candidate
        if pending:
            groups.append(pending)
        header = prefix
    result = []
    for group in groups:
        text = header + '\n\n'.join(render(c) for c in group)
        count = tokenizer.count(text)
        if count > limit:
            raise ValueError('Semantic chunk exceeds the model context; no truncated input was produced')
        provenance = [p for c in group for p in c['provenance']]
        result.append(dict(text=text, tokens=count, provenance=provenance,
                           components=[dict(symbol=c['symbol'], origin_symbol=c.get('origin_symbol', c['symbol']),
                                            calls=c['calls']) for c in group],
                           complete_view=full_tokens <= limit, full_view_tokens=full_tokens))
    return result, full_tokens
