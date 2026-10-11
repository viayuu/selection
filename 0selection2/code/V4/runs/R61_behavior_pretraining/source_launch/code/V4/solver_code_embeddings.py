"""Explicit, budgeted Voyage calls and immutable offline solver embedding bundles."""

import argparse
from collections import deque
import datetime
from email.utils import parsedate_to_datetime
import json
import math
import os
import re
import stat
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS
from .solver_code_corpus import ROOT, MODEL, digest, check_sensitive, source_path, write_json
from .solver_code_encoder import CODE_DIM, ROLES, check_vectors, check_components


SETTINGS = dict(model=MODEL, output_dimension=CODE_DIM, output_dtype='float', input_type=None, truncation=False)
ENDPOINT = 'https://ai.mongodb.com/v1/embeddings'
ENDPOINTS = dict(mongodb_atlas=ENDPOINT, voyage='https://api.voyageai.com/v1/embeddings')
REQUEST_TOKENS = 320000
COMPONENT_FIELDS = ('chunk_vectors', 'chunk_indices')


def provider_for(corpus):
    return corpus.get('api', {}).get('provider', 'voyage')


def bundle_provenance(bundle):
    return {k: v for k, v in bundle.items() if k not in ('vectors', 'role_mask', *COMPONENT_FIELDS)}


class RequestLimiter:
    """Enforce both request and input-token budgets in a rolling minute."""
    def __init__(self, requests_per_minute, tokens_per_minute):
        if requests_per_minute < 1 or tokens_per_minute < 1:
            raise ValueError('Per-minute request/token limits must be positive')
        self.rpm, self.tpm = requests_per_minute, tokens_per_minute
        self.history = deque()

    def wait(self, tokens):
        if tokens > self.tpm:
            raise ValueError('A chunk exceeds --tokens-per-minute; increase the limit or rebuild smaller chunks')
        while True:
            now = time.monotonic()
            while self.history and now - self.history[0][0] >= 61:
                self.history.popleft()
            if len(self.history) < self.rpm and sum(n for _, n in self.history) + tokens <= self.tpm:
                self.history.append((now, tokens))
                return
            delay = max(0.01, self.history[0][0] + 61 - now)
            print(f'[embedding] Rate limit pacing: waiting {delay:.1f}s', flush=True)
            time.sleep(delay)


def error_details(error):
    # Inspect structured error fields, but never print remote bodies or credentials.
    if error.code != 429 or error.fp is None:
        return False, {}
    try:
        body = json.loads(error.read(16384))
        detail = json.dumps({k: body[k] for k in ('detail', 'message', 'error') if k in body}).lower()
    except (ValueError, TypeError, AttributeError, OSError):
        detail = ''
    quota = any(word in detail for word in ('insufficient_quota', 'quota_exceeded', 'quota exceeded',
                'insufficient credit', 'insufficient balance', 'billing_hard_limit',
                'payment_required', 'payment required', 'credit balance exhausted'))
    limits = {}
    for unit in ('rpm', 'tpm'):
        match = re.search(r'([0-9][0-9,]*)\s*' + unit + r'\b', detail)
        if match:
            limits[unit] = int(match.group(1).replace(',', ''))
    return quota, limits


def retry_delay(error, attempt):
    value = error.headers.get('Retry-After') if error.headers else None
    if value is not None:
        try:
            delay = float(value)
        except ValueError:
            try:
                deadline = parsedate_to_datetime(value)
                delay = deadline.timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                delay = -1
        if math.isfinite(delay) and delay >= 0:
            return max(1., delay)
    return 61. if error.code == 429 else min(60., 2. ** (attempt + 1))


def validate_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get('kind') != 'production' or bundle.get('provider') not in ENDPOINTS:
        raise ValueError('Only real production Voyage bundles are accepted; no mock/random fallback')
    provider = bundle['provider']
    if bundle.get('settings') != SETTINGS or bundle.get('roles') != list(ROLES) or bundle.get('solver_names') != GLOBAL_SOLVERS:
        raise ValueError('Embedding model/settings/role order/solver identity mismatch')
    if bundle.get('corpus_ready') is not True or not bundle.get('calls'):
        raise ValueError('Production vectors require an approved source corpus and request provenance')
    if bundle.get('uses_assumptions') and (bundle.get('historical_bindings_verified') is not False
            or not bundle.get('provenance_warnings') or not bundle.get('assumed_configurations')):
        raise ValueError('Assumed-default vectors must preserve unverified historical provenance')
    source = 'mongodb-atlas-http' if provider == 'mongodb_atlas' else 'voyage-http'
    if any(call.get('source') != source or not call.get('timestamp_utc') for call in bundle['calls']):
        raise ValueError('Mock/missing API request provenance is not a production embedding')
    if provider == 'mongodb_atlas':
        if bundle.get('endpoint') != ENDPOINT or any(call.get('endpoint') != ENDPOINT for call in bundle['calls']):
            raise ValueError('Atlas vectors require Atlas endpoint provenance, not a Voyage cache')
        check_components(bundle['chunk_vectors'], bundle['chunk_indices'], bundle['role_mask'])
        if len(bundle.get('chunk_provenance', [])) != len(bundle['chunk_vectors']):
            raise ValueError('Every component embedding needs its source provenance')
    for key in ('corpus_sha256', 'cache_sha256'):
        if len(bundle.get(key, '')) != 64:
            raise ValueError('Missing locked corpus/cache provenance')
    check_vectors(bundle['vectors'], bundle['role_mask'])


def load_bundle(path):
    path = Path(path)
    lock = json.loads(path.with_suffix('.lock.json').read_text())
    if lock.get('sha256') != digest(path.read_bytes()):
        raise ValueError('Embedding bundle changed after it was locked')
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    validate_bundle(bundle)
    return bundle


def api_key(path=None):
    if path is None:
        key = os.environ.get('VOYAGE_API_KEY', '').strip()
    else:
        path = Path(path).expanduser().resolve()
        project = Path(__file__).resolve().parents[2]
        metadata = path.stat()
        if path.is_relative_to(project) or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError('Key file must be private (mode600), owned by you, and outside this Git repository')
        key = path.read_text().strip()
    if not key or '\n' in key:
        raise ValueError('Set VOYAGE_API_KEY to your Atlas Model API Key or supply a private --api-key-file; no request was sent')
    return key


def request_embeddings(texts, key, *, token_count=0, limiter=None, provider='mongodb_atlas'):
    for text in texts:
        check_sensitive(text)
    endpoint = ENDPOINTS[provider]
    if not 1 <= len(texts) <= 1000 or token_count > REQUEST_TOKENS:
        raise ValueError('Atlas requests allow at most 1000 texts / 320000 tokens')
    request = urllib.request.Request(endpoint, data=json.dumps(dict(SETTINGS, input=texts)).encode(),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    for attempt in range(5):
        if limiter is not None:
            limiter.wait(token_count)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
                request_id = response.headers.get('x-request-id')
            break
        except urllib.error.HTTPError as error:
            quota, limits = error_details(error)
            if error.code == 429:
                reported = ', '.join(f'{name.upper()}={limit}' for name, limit in limits.items())
                dashboard = 'MongoDB Atlas AI Model APIs' if provider == 'mongodb_atlas' else 'Voyage dashboard'
                hint = f' Check {dashboard} Billing and organization/project Rate Limits; completed chunks are retained.'
                if quota:
                    raise RuntimeError('Voyage HTTP 429: service reports an account quota/payment limit.' + hint) from None
                if any(limit < 1 for limit in limits.values()):
                    raise RuntimeError('Voyage HTTP 429: service reports a zero rate limit for this account/model.' + hint) from None
                if limits.get('tpm', token_count) < token_count:
                    raise RuntimeError(f'Voyage HTTP 429: request has {token_count} tokens, reported {reported}. '
                                       'Rerun with --batch-size 1 and --tokens-per-minute set to your dashboard limit.' + hint) from None
                if limiter is not None:
                    limiter.rpm = min(limiter.rpm, limits.get('rpm', limiter.rpm))
                    limiter.tpm = min(limiter.tpm, limits.get('tpm', limiter.tpm))
                if attempt == 4:
                    raise RuntimeError('Voyage HTTP 429: rate limit persists after 5 attempts. ' + reported + hint) from None
            elif error.code not in (500, 502, 503, 504) or attempt == 4:
                raise RuntimeError(f'Voyage HTTP {error.code}; no backend substitution was attempted') from None
            delay = retry_delay(error, attempt)
            if delay > 600:
                raise RuntimeError(f'Voyage HTTP {error.code}: Retry-After exceeds 10 minutes; rerun later, cached chunks are retained') from None
            print(f'[embedding] HTTP {error.code}: retry {attempt + 1}/4 in {delay:.1f}s', flush=True)
            time.sleep(delay)
        except urllib.error.URLError:
            raise RuntimeError('Voyage network request failed; cached completed chunks are retained') from None
    if result.get('model', MODEL) != MODEL:
        raise ValueError('Server returned a different embedding model')
    items = sorted(result['data'], key=lambda row: row['index'])
    if [row['index'] for row in items] != list(range(len(texts))):
        raise ValueError('API response indices do not match the request')
    vectors = np.asarray([row['embedding'] for row in items], dtype=np.float32)
    if vectors.shape != (len(texts), CODE_DIM) or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=-1) == 0).any():
        raise ValueError('Invalid Voyage vectors; not cached')
    metadata = dict(source='mongodb-atlas-http' if provider == 'mongodb_atlas' else 'voyage-http',
                    endpoint=endpoint, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    request_id=request_id, model=result.get('model', MODEL), usage=result.get('usage', {}),
                    revision=result.get('revision'), revision_note='Provider may not expose a pinned model revision')
    return vectors, metadata


def cache_key(chunk, corpus, provider=None):
    provider = provider or provider_for(corpus)
    identity = dict(chunk_id=chunk['id'], text_sha256=chunk['text_sha256'], settings=SETTINGS,
                    preprocessing=corpus['preprocessing'], tokenizer=corpus['tokenizer_sha256'],
                    max_tokens=corpus['max_tokens'], manifest=corpus['manifest_sha256'])
    # Keep the original identity for old artifacts; Atlas never shares that namespace.
    if provider != 'voyage':
        identity.update(provider=provider, endpoint=ENDPOINTS[provider])
    return digest(identity)


def validate_corpus(corpus, snapshot=None):
    expected = digest({k: v for k, v in corpus.items() if k != 'sha256'})
    limit = 32000 if corpus.get('schema') == 'r45-corpus-v2' else 4096
    if corpus.get('sha256') != expected or corpus.get('model') != MODEL or corpus.get('max_tokens') != limit:
        raise ValueError('Corpus hash/model/chunking mismatch; rebuild the corpus')
    if limit == 32000 and corpus.get('api') != dict(provider='mongodb_atlas', endpoint=ENDPOINT):
        raise ValueError('The complete-view corpus must declare the MongoDB Atlas endpoint')
    if corpus.get('roles') != list(ROLES) or [s['solver_name'] for s in corpus['solvers']] != GLOBAL_SOLVERS:
        raise ValueError('Corpus solver/role order mismatch')
    if corpus.get('uses_assumptions') and (not corpus.get('assumptions_accepted')
            or corpus.get('historical_bindings_verified') is not False
            or not corpus.get('provenance_warnings') or not corpus.get('assumed_configurations')):
        raise ValueError('Assumptions need explicit approval and preserved historical warnings')
    ids = [c['id'] for c in corpus['chunks']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate chunk identifiers')
    for chunk in corpus['chunks']:
        check_sensitive(chunk['text'])
        if chunk['text_sha256'] != digest(chunk['text'].encode()) or not 0 < chunk['tokens'] <= limit:
            raise ValueError('Chunk content/token count changed')
    if snapshot is not None:
        if digest(snapshot) != corpus['manifest_sha256']:
            raise ValueError('Source manifest changed; rebuild the corpus')
        for name, sha in corpus['source_hashes'].items():
            if digest(source_path(snapshot, name).read_bytes()) != sha:
                raise ValueError(f'Allowlisted source changed: {name}; rebuild the corpus')
        for name, sha in corpus.get('config_source_hashes', {}).items():
            if digest(source_path(snapshot, name, ('.py', '.yaml', '.yml')).read_bytes()) != sha:
                raise ValueError(f'Default configuration source changed: {name}; rebuild the corpus')


def read_cache(path, expected_key, provider='voyage'):
    value = json.loads(path.read_text())
    if value.get('key') != expected_key or value.get('settings') != SETTINGS or value.get('provider') != provider:
        raise ValueError('Cache provenance/settings mismatch')
    vector = np.asarray(value['embedding'], dtype=np.float32)
    if vector.shape != (CODE_DIM,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError('Invalid cached embedding')
    if (value.get('embedding_sha256') != digest(vector.tobytes()) or not value.get('call', {}).get('timestamp_utc')
            or value.get('call', {}).get('source') != ('mongodb-atlas-http' if provider == 'mongodb_atlas' else 'voyage-http')):
        raise ValueError('Cached vector changed or has no request provenance')
    if provider == 'mongodb_atlas' and value['call'].get('endpoint') != ENDPOINT:
        raise ValueError('Atlas cache is missing its endpoint identity')
    return vector, value['call']


def l2mean(values):
    values = np.asarray(values, dtype=np.float64)
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    result = values.mean(axis=0)
    length = np.linalg.norm(result)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('Degenerate mean code vector')
    return (result / length).astype(np.float32)


def component_bundle(corpus, entries):
    ids = list(entries)
    lookup = {key: i for i, key in enumerate(ids)}
    vectors = np.stack([entries[key] for key in ids])
    vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    variants = max(len(s['variants']) for s in corpus['solvers'])
    chunks = max(len(v['roles'][r]) for s in corpus['solvers'] for v in s['variants'] for r in ROLES)
    indices = torch.full((len(GLOBAL_SOLVERS), len(ROLES), variants, chunks), -1, dtype=torch.long)
    for s, solver in enumerate(corpus['solvers']):
        for v, variant in enumerate(solver['variants']):
            for r, role in enumerate(ROLES):
                selected = variant['roles'][role]
                indices[s, r, v, :len(selected)] = torch.tensor([lookup[key] for key in selected], dtype=torch.long)
    provenance = [{key: value for key, value in chunk.items() if key != 'text'} for chunk in corpus['chunks']]
    return dict(schema='r45-embeddings-v2', endpoint=ENDPOINT, chunk_vectors=torch.from_numpy(vectors),
                chunk_indices=indices, chunk_provenance=provenance,
                semantic_views=corpus.get('semantic_views', []), context_tokens=32000,
                aggregation='Whole view unchanged; overflow: trainable role-shared attention + max-summary fusion; '
                            'equal distinct variant mean after projection. vectors is only a legacy reference summary.')


def assemble(corpus, cache_dir, output):
    validate_corpus(corpus)
    if not corpus['ready']:
        raise ValueError('Unresolved actual solver bindings cannot become production embeddings')
    provider = provider_for(corpus)
    entries, calls, cache_hashes = {}, {}, {}
    for chunk in corpus['chunks']:
        key = cache_key(chunk, corpus)
        path = cache_dir / (key + '.json')
        entries[chunk['id']], call = read_cache(path, key, provider)
        calls[digest(call)] = call
        cache_hashes[key] = digest(path.read_bytes())
    vectors = np.zeros((len(GLOBAL_SOLVERS), len(ROLES), CODE_DIM), dtype=np.float32)
    mask = np.zeros(vectors.shape[:2], dtype=bool)
    for row, solver in enumerate(corpus['solvers']):
        for column, role in enumerate(ROLES):
            variants = []
            for variant in solver['variants']:
                ids = variant['roles'][role]
                if ids:
                    variants.append(l2mean([entries[key] for key in ids]))
            if variants:
                vectors[row, column] = l2mean(variants)
                mask[row, column] = True
    bundle = dict(schema='r45-embeddings-v1', kind='production', provider=provider, settings=SETTINGS,
                  solver_names=GLOBAL_SOLVERS, roles=list(ROLES), vectors=torch.from_numpy(vectors),
                  role_mask=torch.from_numpy(mask), corpus_sha256=corpus['sha256'], corpus_ready=True,
                  cache_sha256=digest(cache_hashes), calls=list(calls.values()),
                  aggregation='L2 chunks -> equal chunk mean/L2 per variant -> equal distinct variant mean/L2')
    if provider == 'mongodb_atlas':
        bundle.update(component_bundle(corpus, entries))
    if corpus.get('uses_assumptions'):
        bundle.update({key: corpus[key] for key in ('uses_assumptions', 'historical_bindings_verified',
                                                  'provenance_warnings', 'assumed_configurations')})
    validate_bundle(bundle)
    output = Path(output)
    if output.exists() or output.with_suffix('.lock.json').exists():
        raise ValueError('Refusing to overwrite a locked embedding bundle; choose a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    torch.save(bundle, temporary)
    temporary.replace(output)
    write_json(output.with_suffix('.lock.json'), dict(sha256=digest(output.read_bytes()),
               corpus_sha256=corpus['sha256'], settings=SETTINGS, cache_sha256=bundle['cache_sha256']))
    return bundle


def generate(corpus, cache_dir, allow_upload=False, max_tokens=None, key_file=None, batch_size=16,
             requests_per_minute=2000, tokens_per_minute=8000000, provider=None):
    validate_corpus(corpus)
    provider = provider or provider_for(corpus)
    if provider != provider_for(corpus):
        raise ValueError('Provider differs from corpus preparation; rebuild the Atlas corpus instead of reusing legacy chunks')
    pending = []
    for chunk in corpus['chunks']:
        key = cache_key(chunk, corpus)
        path = cache_dir / (key + '.json')
        if path.exists():
            read_cache(path, key, provider)
        else:
            pending.append((chunk, key, path))
    plan = dict(total_chunks=len(corpus['chunks']), uncached_chunks=len(pending),
                uncached_tokens=sum(c['tokens'] for c, _, _ in pending), settings=SETTINGS, ready=corpus['ready'],
                provider=provider, endpoint=ENDPOINTS[provider], context_tokens=corpus['max_tokens'],
                requests_per_minute=requests_per_minute, tokens_per_minute=tokens_per_minute)
    if corpus.get('semantic_views'):
        plan.update(semantic_views=len(corpus['semantic_views']),
                    split_views=sum(not view['complete'] for view in corpus['semantic_views']),
                    largest_view_tokens=max(view['full_view_tokens'] for view in corpus['semantic_views']))
    if corpus.get('uses_assumptions'):
        plan.update(uses_assumptions=True, historical_bindings_verified=False,
                    preserved_historical_gaps=len(corpus['provenance_warnings']))
    if not allow_upload:
        return plan
    if not corpus['ready']:
        raise ValueError('Draft corpus has unresolved source/config bindings; upload refused')
    if max_tokens is None or plan['uncached_tokens'] > max_tokens:
        raise ValueError('Explicit --max-tokens budget is required and must cover the uncached input')
    if not 1 <= batch_size <= 1000:
        raise ValueError('Request batch must be 1..1000 texts')
    limiter = RequestLimiter(requests_per_minute, tokens_per_minute)
    if any(chunk['tokens'] > tokens_per_minute for chunk, _, _ in pending):
        raise ValueError('A chunk exceeds --tokens-per-minute; increase the limit or rebuild smaller chunks')
    if not pending:
        return plan
    key = api_key(key_file)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[embedding] {len(pending)} pending chunks / {plan['uncached_tokens']} tokens; "
          f'batch<={batch_size}, RPM<={requests_per_minute}, TPM<={tokens_per_minute}', flush=True)
    offset = 0
    while offset < len(pending):
        items, token_count = [], 0
        for item in pending[offset:offset + batch_size]:
            if items and token_count + item[0]['tokens'] > min(limiter.tpm, REQUEST_TOKENS):
                break
            items.append(item)
            token_count += item[0]['tokens']
        vectors, call = request_embeddings([c['text'] for c, _, _ in items], key,
                                          token_count=token_count, limiter=limiter, provider=provider)
        for (chunk, identity, path), vector in zip(items, vectors):
            write_json(path, dict(key=identity, provider=provider, settings=SETTINGS, call=call,
                                 text_sha256=chunk['text_sha256'], embedding=vector.tolist(),
                                 embedding_sha256=digest(vector.tobytes())))
        offset += len(items)
        print(f"[embedding] {offset}/{len(pending)} uncached chunks", flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'embed', 'assemble', 'all'), default='plan')
    parser.add_argument('--corpus', type=Path, default=ROOT / 'corpus_atlas/corpus.json')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'embedding_cache_atlas')
    parser.add_argument('--output', type=Path, default=ROOT / 'solver_code_vectors_atlas.pt')
    parser.add_argument('--provider', choices=tuple(ENDPOINTS), default='mongodb_atlas')
    parser.add_argument('--allow-upload', action='store_true')
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--batch-size', type=int, default=16, help='Maximum texts per request (1..1000); total capped at 320K tokens')
    parser.add_argument('--requests-per-minute', type=int, default=2000)
    parser.add_argument('--tokens-per-minute', type=int, default=8000000)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    validate_corpus(corpus, json.loads(args.corpus.with_name('source_snapshot.json').read_text()))
    if args.provider != provider_for(corpus):
        parser.error('Provider differs from corpus preparation; use the matching provider/corpus')
    if args.stage in ('embed', 'all') and not args.allow_upload:
        parser.error('--stage embed/all requires --allow-upload (source code is sent to Voyage)')
    if args.stage in ('plan', 'embed', 'all'):
        print(json.dumps(generate(corpus, args.cache_dir, args.stage != 'plan', args.max_tokens, args.api_key_file,
                                  batch_size=args.batch_size, requests_per_minute=args.requests_per_minute,
                                  tokens_per_minute=args.tokens_per_minute, provider=args.provider), indent=2))
    if args.stage in ('assemble', 'all'):
        bundle = assemble(corpus, args.cache_dir, args.output)
        print(f"Locked offline vectors: {args.output}, shape={tuple(bundle['vectors'].shape)}")


if __name__ == '__main__':
    main()
