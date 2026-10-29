"""TRAIN-only source-membership preparation and frozen historical expert adapters.

Source-record certification uses immutable generator-shard SHA256 plus the
row inside that shard, not bare split/global indices. It is not an exhaustive
content/base-duplicate certificate. No held-out scores select experts here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS


V4 = Path(__file__).resolve().parent
PREPARATION = V4 / 'runs/R123_heterogeneous_moe/preparation'
CURRENT_ROOT = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
OLD_ROOT = Path('/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014')
R77_ROOT = V4 / 'runs/R77_sample_size_batch1024'
DEFAULT_EXPERTS = ('R77B', 'R80A', 'R83')
BASE_COMMIT = '04866414831d25330b0d2e575e4b63c68b4196d0'
EXPERTS = {
    'R77B': ('R77_sample_size_batch1024/B_full_seed2', 'r77_experiment',
             'handcrafted representation; original composite objective and R-Drop'),
    'R80A': ('R80_training_and_code_ablation/A_ce_only', 'r80_experiment',
             'handcrafted representation; CE-only objective'),
    'R80C': ('R80_training_and_code_ablation/C_lr3e4', 'r80_experiment',
             'handcrafted representation; composite objective; changed learning rate'),
    'R80D': ('R80_training_and_code_ablation/D_code_embedding', 'r80_experiment',
             'complete source-code representation; composite objective'),
    'R83': ('R83_model_only_source_embedding/model_only_seed2', 'r83_experiment',
            'model-only source-code representation; composite objective'),
}
INPUT_FIELDS = frozenset(('problem_id', 'kind', 'n', 'node', 'matrix', 'node_mask',
    'cbits', 'problem_desc', 'coord_dist', 'node_geom', 'pool_ids', 'solver_mask', 'mask', 'd_in'))
SOURCE_FILES = ('r77_experiment.py', 'r80_experiment.py', 'r83_experiment.py',
    'solver_code_encoder.py', 'solver_code_embeddings.py', 'pairwise_selector.py',
    'dual_stream.py', 'V4Model.py', 'local_geometry.py', 'pairwise_objective.py',
    'r76_cache.py', 'r76_experiment.py', 'new_split_data.py')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


class Evidence:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Frozen metadata hash changed: {path}')
        previous = self.files.get(str(path))
        if previous is not None and previous['sha256'] != digest:
            raise ValueError(f'Provenance changed while preparing: {path}')
        self.files[str(path)] = dict(sha256=digest, bytes=path.stat().st_size)
        return digest


def validate_lineage(old, generation):
    if (old['index_mapping'] != 'local_index == original_global_index'
            or old['original_instances'] != generation['original_instances']
            or old['global_index_start'] != 0 or old['global_index_stop'] != old['instances']):
        raise ValueError('Old source-row mapping is not established')
    current = {row['id']: row for row in generation['shards']}
    cursor = old['original_instances']
    for row in old['included_new_shards']:
        other = current.get(row['id'])
        keys = ('sha256', 'start_index', 'scale', 'distribution', 'capacity_type', 'seed')
        if other is None or any(row[key] != other[key] for key in keys):
            raise ValueError('Old and current sources do not reference the same immutable shard')
        if row['start_index'] != cursor or not 0 < row['count'] <= other['count']:
            raise ValueError('Old included-shard membership is not a declared source prefix')
        cursor += row['count']
    if cursor != old['instances']:
        raise ValueError('Old source membership has an uncovered tail')


def source_record_ids(problem, source_indices, generation):
    """Explicit generator lookup; global integers alone never establish identity."""
    indices = np.asarray(source_indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise ValueError('Source row references must be a one-dimensional integer array')
    shards = generation['shards']
    starts = np.asarray([s['start_index'] for s in shards], dtype=np.int64)
    sizes = np.asarray([s['count'] for s in shards], dtype=np.int64)
    prefix = int(generation['original_instances'])
    if (not len(shards) or starts[0] != prefix or np.any(sizes <= 0)
            or not np.array_equal(starts[1:], (starts + sizes)[:-1])):
        raise ValueError('Generator shards have a gap/overlap or missing explicit prefix')
    slots = np.searchsorted(starts, indices, side='right') - 1
    if (np.any(slots < 0) or np.any(indices >= (starts + sizes)[slots])):
        raise ValueError('A referenced input has no declared generator shard identity')
    ids = []
    for index, slot in zip(indices, slots):
        digest = shards[int(slot)]['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Generator shard has no valid SHA256 identity')
        ids.append(f'{problem}:{digest}:{int(index - starts[slot])}')
    return np.asarray(ids, dtype='S100')


def unseen_intersection(current_ids, expert_memberships):
    """Missing identity evidence certifies zero rows instead of assuming unseen."""
    current_ids = np.asarray(current_ids)
    if current_ids.ndim != 1 or len(np.unique(current_ids)) != len(current_ids):
        raise ValueError('Current source-record identities must be distinct')
    if not expert_memberships:
        raise ValueError('At least one explicitly identified historical expert is required')
    eligible, gaps = np.ones(len(current_ids), dtype=bool), []
    for expert, trained in expert_memberships.items():
        if trained is None:
            gaps.append(f'{expert}: historical TRAIN identity evidence missing')
            eligible[:] = False
        else:
            trained = np.asarray(trained)
            if trained.ndim != 1:
                raise ValueError('Historical identities must be one-dimensional')
            eligible &= ~np.isin(current_ids, trained)
    return np.flatnonzero(eligible).astype(np.int64), gaps


def _canonical_candidates(names, aliases):
    result = tuple(aliases.get(str(name), str(name)) for name in names)
    if len(set(result)) != len(result):
        raise ValueError('Candidate aliases collide; never merge multiple deployments')
    return result


def candidate_permutation(source_names, target_names, aliases=None):
    aliases = aliases or {}
    source, target = (_canonical_candidates(names, aliases) for names in (source_names, target_names))
    if set(source) != set(target):
        raise ValueError('Expert and target must retain exactly the same legal candidate deployments')
    return np.asarray([source.index(name) for name in target], dtype=np.int64)


def align_candidate_scores(scores, source_names, target_names, aliases=None):
    scores = np.asarray(scores)
    if scores.ndim < 1 or scores.shape[-1] != len(source_names):
        raise ValueError('Score columns and expert candidate names differ')
    return scores[..., candidate_permutation(source_names, target_names, aliases)]


def normalize_scores(scores, mode='raw'):
    """Explicit label-free per-query transform, never a fitted VAL/TEST scaler."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 1 or not np.isfinite(scores).all():
        raise ValueError('Expert scores must be finite')
    if mode == 'raw':
        return scores.copy()
    centered = scores - scores.mean(axis=-1, keepdims=True)
    if mode == 'center':
        return centered
    if mode == 'zscore':
        scale = scores.std(axis=-1, keepdims=True)
        return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)
    raise ValueError('Normalization must be raw, center, or zscore')


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key in INPUT_FIELDS}


def load_expert(expert_id, device='cpu'):
    relative, module, _ = EXPERTS[expert_id]
    loader = importlib.import_module(f'code.V4.{module}').load_checkpoint
    model, state = loader(V4 / 'runs' / relative / 'best.pt', device=device)
    model.eval().requires_grad_(False)
    return model, state


class ExpertAdapter:
    def __init__(self, model, global_solver_names, candidate_pools, aliases=None):
        self.model = model.eval().requires_grad_(False)
        self.solver_names = tuple(global_solver_names)
        self.candidate_pools, self.aliases = candidate_pools, aliases or {}

    def predict(self, problem, batch, target_names=None, normalization='raw'):
        import torch
        names = self.candidate_pools[problem]
        target_names = names if target_names is None else target_names
        indices = candidate_permutation(names, target_names, self.aliases)
        device = next(self.model.parameters()).device
        inputs = model_inputs(batch)
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        canonical = _canonical_candidates(names, self.aliases)
        inputs['pool_ids'] = torch.tensor([self.solver_names.index(name) for name in canonical],
                                         dtype=torch.long, device=device)
        inputs['solver_mask'] = torch.ones((len(inputs['n']), len(names)), dtype=torch.bool, device=device)
        with torch.no_grad():
            scores = self.model(inputs)['logits'].float().cpu().numpy()
        return normalize_scores(scores[:, indices], normalization)


def _train_rows(root, problem, evidence, split_snapshot):
    directory = root / (problem + 'train')
    metadata_path = directory / 'metadata.json'
    evidence.bind(metadata_path)
    meta = read_json(metadata_path)
    path = directory / 'input_metadata.npz'
    evidence.bind(path, meta['files']['input_metadata.npz']['sha256'])
    with np.load(path, allow_pickle=False) as stream:
        indices = stream['source_index'].copy()
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    if (len(indices) != meta['instances'] or len(np.unique(indices)) != len(indices)
            or digest != meta['source_index_sha256']
            or digest != split_snapshot['index_sha256']['train']):
        raise ValueError('TRAIN identity manifest, metadata and source indices disagree')
    return indices, meta


def _probe_expert(expert_id, evidence, train_cache):
    import torch
    relative, module, distinction = EXPERTS[expert_id]
    root = V4 / 'runs' / relative
    protocol_path, checkpoint = root / 'protocol.json', root / 'best.pt'
    evidence.bind(protocol_path)
    config = read_json(protocol_path)
    result = dict(expert_id=expert_id, distinction=distinction, selected=expert_id in DEFAULT_EXPERTS,
        run_root=str(root), protocol_path=str(protocol_path), checkpoint_path=str(checkpoint),
        checkpoint_sha256=evidence.bind(checkpoint),
        factory=f'code.V4.{module}.load_checkpoint', data_root=config['data_root'],
        declared_train_rows_per_problem=config['train_rows_per_problem'],
        loss=config['loss'], initialization=config['initialization'],
        preparation_sha256=config['preparation_sha256'], loader_status='not_checked')
    try:
        model, state = load_expert(expert_id)
        args = state['args']
        if (args['solver_names'] != list(GLOBAL_SOLVERS) or args['data_root'] != config['data_root']
                or args['train_rows_per_problem'] != config['train_rows_per_problem']):
            raise ValueError('Checkpoint deployment/training contract differs from its saved protocol')
        result.update(loader_status='loaded_strictly_on_cpu',
            checkpoint_selected_updates=int(state['successful_updates']),
            model_class=f'{type(model).__module__}.{type(model).__name__}',
            model_params=args['model_params'],
            input_normalization='checkpoint own historical TRAIN-fitted geometry buffers; never refit',
            checkpoint_training_problem_count=len(state['updates_by_problem']))
        geometry = model.instance_encoder.geometry_residual
        result['geometry_mean'] = geometry.mean.cpu().tolist()
        result['geometry_std'] = geometry.std.cpu().tolist()
        result['smoke_forwards'] = []
        from .r76_cache import ScenarioCache
        pools = {p: train_cache['splits']['train'][p]['pool'] for p in PROBLEMS}
        adapter = ExpertAdapter(model, args['solver_names'], pools,
                                train_cache['scenario_solver_aliases'])
        for problem in ('TSP', 'ATSP', 'VRPB'):
            cache = ScenarioCache(Path(train_cache['splits']['train'][problem]['directory']))
            try:
                scores = adapter.predict(problem, cache.batch(np.asarray([0]), device='cpu'))
                if scores.shape != (1, len(pools[problem])) or not np.isfinite(scores).all():
                    raise ValueError('Nonfinite/wrong-shape CPU expert forward')
                result['smoke_forwards'].append(dict(problem=problem, rows=1, candidate_count=scores.shape[1]))
            finally:
                cache.close()
        del adapter, model, state
        last = root / 'last.pt'
        evidence.bind(last)
        state = torch.load(last, map_location='cpu', weights_only=False)
        result['final_successful_updates'] = int(state['successful_updates'])
        result['full18_training_budget_complete'] = (
            result['final_successful_updates'] == config['max_successful_updates']
            and set(state['updates_by_problem']) == set(PROBLEMS))
        del state
    except Exception as error:
        result.update(loader_status='failed', load_error=f'{type(error).__name__}: {error}',
                      full18_training_budget_complete=False)
    gc.collect()
    return result


def prepare(output=PREPARATION, selected=DEFAULT_EXPERTS):
    output = Path(output).resolve()
    if output != PREPARATION.resolve() and PREPARATION.resolve() not in output.parents:
        raise ValueError('Preparation output must stay in the owned R123 preparation directory')
    if len(set(selected)) != len(selected) or not set(selected) <= set(EXPERTS):
        raise ValueError('Select distinct inventoried historical experts')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise FileExistsError('Preserve the existing preparation; choose a new owned subdirectory')
    evidence = Evidence()
    current_split_path = CURRENT_ROOT / 'split_manifest.json'
    old_split_path = OLD_ROOT / 'original_split_manifest.json'
    for path in (current_split_path, old_split_path, CURRENT_ROOT / 'scenario_manifest.json',
                 OLD_ROOT / 'scenario_manifest.json'):
        evidence.bind(path)
    current_split, old_split = read_json(current_split_path), read_json(old_split_path)
    current_scenario, old_scenario = (read_json(root / 'scenario_manifest.json') for root in (CURRENT_ROOT, OLD_ROOT))
    ready_path, cache_path = R77_ROOT / 'prepared.json', R77_ROOT / 'source_cache_manifest.json'
    evidence.bind(ready_path)
    ready = read_json(ready_path)
    evidence.bind(cache_path, ready['artifact_sha256']['source_cache_manifest.json'])
    train_cache = read_json(cache_path)
    if set(train_cache['splits']['train']) != set(PROBLEMS):
        raise ValueError('Historical experts must cover the complete 18-problem TRAIN registry')
    selected_bindings = {}
    for expert_id in selected:
        protocol_path = V4 / 'runs' / EXPERTS[expert_id][0] / 'protocol.json'
        evidence.bind(protocol_path)
        config = read_json(protocol_path)
        selected_bindings[expert_id] = (
            Path(config['data_root']).resolve() == OLD_ROOT.resolve()
            and config['train_rows_per_problem'] == 70114
            and config['preparation_sha256'] == evidence.files[str(ready_path.resolve())]['sha256']
            and config['initialization'] in ('scratch_identicalinit', 'scratch_shared_r77'))
    arrays, problems, catalogues = {}, {}, {}
    aliases = dict(current_scenario.get('solver_aliases', {}))
    aliases.update(old_scenario.get('solver_aliases', {}))
    for problem in PROBLEMS:
        try:
            new_info, old_info = current_split['problems'][problem], old_split['problems'][problem]
            generation_path = Path(new_info['source_metadata']['path'])
            lineage_path = Path(old_info['source_metadata'])
            evidence.bind(generation_path, new_info['source_metadata']['sha256'])
            evidence.bind(lineage_path, new_info['previous_source_lineage']['sha256'])
            generation, lineage = read_json(generation_path), read_json(lineage_path)
            if old_info['source_dataset_sha256'] != lineage['dataset_sha256']:
                raise ValueError('Historical source dataset is not bound to the declared subset lineage')
            validate_lineage(lineage, generation)
            current, new_meta = _train_rows(CURRENT_ROOT, problem, evidence, new_info)
            old, old_meta = _train_rows(OLD_ROOT, problem, evidence, old_info)
            if len(current) != 81000 or len(old) != 70114:
                raise ValueError('Unexpected current/historical TRAIN population')
            for row in (new_meta, old_meta):
                candidate_permutation(row['methods'], new_meta['methods'], aliases)
                if row['global_solver_ids'] != new_meta['global_solver_ids']:
                    raise ValueError('Source candidate global IDs differ')
            cached = train_cache['splits']['train'][problem]
            candidate_permutation(cached['pool'], new_meta['methods'], aliases)
            if cached['rows'] != len(old) or cached['pool_ids'] != new_meta['global_solver_ids']:
                raise ValueError('Historical model cache does not use the full legal candidate pool')
            for source in cached['source_files']:
                if Path(source['path']).name in ('metadata.json', 'input_metadata.npz'):
                    stat = Path(source['path']).stat()
                    if stat.st_size != source['size_bytes'] or stat.st_mtime_ns != source['mtime_ns']:
                        raise ValueError('TRAIN membership metadata changed since the frozen expert cache')
            current_ids = source_record_ids(problem, current, generation)
            old_ids = source_record_ids(problem, old, generation)
            local, gaps = unseen_intersection(current_ids,
                {key: old_ids if selected_bindings[key] else None for key in selected})
            if gaps:
                raise ValueError('; '.join(gaps))
            prefix = problem + '__'
            arrays.update({prefix + 'current_train_source_index': current,
                prefix + 'historical_train_source_index': old,
                prefix + 'meta_train_local_indices': local,
                prefix + 'meta_train_source_index': current[local],
                prefix + 'meta_train_source_identity': current_ids[local]})
            problems[problem] = dict(status='source_identity_certified', current_train_count=len(current),
                historical_train_count=len(old), certified_source_unseen_count=len(local),
                eligible_local_indices_sha256=hashlib.sha256(local.tobytes()).hexdigest(),
                meta_train_source_identity_sha256=hashlib.sha256(current_ids[local].tobytes()).hexdigest(),
                candidates=new_meta['methods'], global_solver_ids=new_meta['global_solver_ids'],
                source_identity_scope='problem + frozen generator shard SHA256 + row within shard',
                source_input_version_match='same immutable source records; current release extends old TRAIN',
                strict_content_or_base_unseen_count=None,
                gaps=['No explicit semantic base-group/content-hash inventory; source-record novelty only.',
                      'Generator shard hashes are declared in frozen metadata; large source bytes not rehashed.'])
            catalogues[problem] = dict(generation_metadata=str(generation_path),
                old_subset_metadata=str(lineage_path), shards=generation['shards'])
            print(f'{problem}: source-certified metaTRAIN {len(local)}/{len(current)}', flush=True)
        except Exception as error:
            problems[problem] = dict(status='evidence_gap', certified_source_unseen_count=0,
                                     strict_content_or_base_unseen_count=None,
                                     gaps=[f'{type(error).__name__}: {error}'])
            print(f'{problem}: EVIDENCE_GAP {error}', flush=True)
    experts = {}
    for expert_id in EXPERTS:
        result = _probe_expert(expert_id, evidence, train_cache)
        result['selected'] = expert_id in selected
        if (result['data_root'] != str(OLD_ROOT) or result['declared_train_rows_per_problem'] != 70114
                or result['preparation_sha256'] != evidence.files[str(ready_path.resolve())]['sha256']):
            result['membership_evidence_gap'] = 'Expert is not bound to the verified historical TRAIN release'
        experts[expert_id] = result
        print(expert_id, result['loader_status'], flush=True)
    negative_path = V4 / 'runs/R94_pretrained_node_fusion/A/protocol.json'
    evidence.bind(negative_path)
    negative = read_json(negative_path)
    if (Path(negative['data_root']).resolve() != CURRENT_ROOT.resolve()
            or negative['train_rows_per_problem'] != 81000):
        raise ValueError('R94 negative-control full81k exposure contract changed')
    negative_control = dict(expert_id='R94A', protocol_path=str(negative_path),
        data_root=negative['data_root'], declared_train_rows_per_problem=negative['train_rows_per_problem'],
        included=False, source_unseen_current_train_count=0,
        reason='Full current TRAIN81k is an exposure population, including TRAIN-fitted normalization.')
    for name in SOURCE_FILES:
        evidence.bind(V4 / name)
    evidence.bind(Path(__file__))
    evidence.bind(V4.parent / 'unified_selector/registry.py')
    np.savez_compressed(output / 'meta_train_indices.npz', **arrays)
    _write_json(output / 'source_catalogues.json', catalogues)
    _write_json(output / 'experts.json', experts)
    ready_for_meta = (len(problems) == len(PROBLEMS)
        and all(row['status'] == 'source_identity_certified' for row in problems.values())
        and all(experts[key]['loader_status'] == 'loaded_strictly_on_cpu'
                and experts[key]['full18_training_budget_complete']
                and not experts[key].get('membership_evidence_gap') for key in selected))
    manifest = dict(schema_version='r123_meta_source_v1', base_commit=BASE_COMMIT,
        current_train_data_root=str(CURRENT_ROOT), historical_train_data_root=str(OLD_ROOT),
        selected_experts=list(selected), selection_basis='provisional objective/representation diversity; no held-out scores',
        meta_training_ready_at_source_record_scope=ready_for_meta,
        certified_source_unseen_total=sum(row['certified_source_unseen_count'] for row in problems.values()),
        strict_content_or_base_unseen_total=None, problems=problems, experts=experts,
        candidate_aliases=aliases, negative_exposure_control=negative_control,
        membership_policy='conservative union of full historical TRAIN, including normalization exposure',
        historical_holdouts='VAL/TEST have participated in prior research; not fresh blind sets',
        heldout_data_used_for_meta_training=False, test_metrics_or_predictions_read=False,
        calibration_fitted=False, default_score_normalization='raw',
        r50_two_method_experts_included=False, scientific_mechanism='pending Astra; no gate/loss trained',
        evidence_files=evidence.files,
        artifacts={name: dict(path=str(output / name), sha256=sha256(output / name))
                   for name in ('meta_train_indices.npz', 'source_catalogues.json', 'experts.json')})
    manifest['preparation_identity'] = json_hash(manifest)
    _write_json(output / 'manifest.json', manifest)
    _write_summary(output, manifest)
    return manifest


def load_meta_indices(problem, preparation=PREPARATION, *, require_usable=True):
    preparation = Path(preparation)
    manifest = read_json(preparation / 'manifest.json')
    if require_usable and not manifest['meta_training_ready_at_source_record_scope']:
        raise ValueError('Preparation has unresolved selected-expert/identity evidence gaps')
    if manifest['problems'][problem]['status'] != 'source_identity_certified':
        raise ValueError('This task has no certified source-record membership intersection')
    record = manifest['artifacts']['meta_train_indices.npz']
    if sha256(record['path']) != record['sha256']:
        raise ValueError('Frozen metaTRAIN indices changed')
    with np.load(record['path'], allow_pickle=False) as stream:
        return stream[problem + '__meta_train_local_indices'].copy()


def _write_summary(output, manifest):
    rows = ['# R123 CPU data/model preparation', '',
        'Provisional trio: ' + ', '.join(manifest['selected_experts']) + '.',
        'Source-record scope only; no mathematical-base/content-duplicate guarantee.', '',
        '| Problem | Source-certified metaTRAIN | Current TRAIN | Status |', '| --- | ---: | ---: | --- |']
    for problem, info in manifest['problems'].items():
        rows.append(f'| {problem} | {info["certified_source_unseen_count"]} | '
                    f'{info.get("current_train_count", "unknown")} | {info["status"]} |')
    rows += ['', '## Expert interfaces', '', '| Expert | Loader | Complete full18 budget | Role |',
             '| --- | --- | --- | --- |']
    for key, info in manifest['experts'].items():
        rows.append(f'| {key} | {info["loader_status"]} | '
                    f'{info.get("full18_training_budget_complete", False)} | {info["distinction"]} |')
    rows += ['', '## Usage', '', '```python',
        'from code.V4.r123_meta_data import load_meta_indices, load_expert, ExpertAdapter',
        "indices = load_meta_indices('TSP')  # positions in current frozen TRAIN81k, not source-global IDs",
        "model, state = load_expert('R77B', device='cpu')", '# Keep labels/IDs outside ExpertAdapter.predict inputs.',
        '```', '',
        'Full paths/hashes and each checkpoint model_params are in experts.json/manifest.json.',
        'source_catalogues.json binds generated source shards and old subset membership.',
        'meta_train_indices.npz also stores both complete TRAIN source-reference lists and meta source identities.',
        'Default score normalization is raw; center/zscore are optional per-query, label-free transforms.',
        'R94A is excluded: it declares the full current TRAIN81k exposure population.',
        'VAL/TEST are not meta training or expert-selection inputs. Old holdouts were exposed during research.',
        'No GPU, optimizer, training, R120 modification, or Git operation occurred.', '']
    (output / 'README.md').write_text('\n'.join(rows))


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PREPARATION)
    parser.add_argument('--experts', nargs='+', choices=tuple(EXPERTS), default=DEFAULT_EXPERTS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = prepare(args.output, tuple(args.experts))
    print(json.dumps(dict(source_certified_total=manifest['certified_source_unseen_total'],
        ready_at_source_scope=manifest['meta_training_ready_at_source_record_scope'],
        manifest=str(args.output / 'manifest.json'))), flush=True)


if __name__ == '__main__':
    main()
