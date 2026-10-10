"""R49 grouped, size-stratified holdouts of existing OVRPTW training data."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .behavior_memory import base_instance_ids
from .local_geometry import FIELDS
from .multitask_probe import dump, file_hash, state_hash
from .r48_common import binary_labels, load_split, tree_features, write_csv
from .train import make_loader


ROOT = Path('code/V4/runs/R49_existing_data_diagnosis')
GROUPS = ('training_pool', 'development', 'internal', 'original_val')
MODELS = ('A_2000_seed2', 'B_8000_seed2')


def grouped_partition(rows, sizes, base_ids, fraction, seed=2):
    rows = np.asarray(rows, dtype=np.int64)
    groups = np.unique(base_ids[rows])
    strata = []
    for group in groups:
        values = np.unique(sizes[rows[base_ids[rows] == group]])
        if len(values) != 1:
            raise ValueError('Related instances have inconsistent customer counts')
        strata.append(values[0])
    first, second = train_test_split(groups, train_size=fraction, stratify=strata, random_state=seed)
    return (np.sort(rows[np.isin(base_ids[rows], first)]),
            np.sort(rows[np.isin(base_ids[rows], second)]))


def make_splits(sizes, base_ids, val_base_ids):
    if np.intersect1d(base_ids, val_base_ids).size:
        raise ValueError('Original validation contains a training-related base instance')
    rows = np.arange(len(sizes))
    pool, holdout = grouped_partition(rows, sizes, base_ids, .8)
    development, internal = grouped_partition(holdout, sizes, base_ids, .5)
    small, _ = grouped_partition(pool, sizes, base_ids, .25)
    splits = dict(training_pool=pool, development=development, internal=internal,
                  original_val=np.arange(len(val_base_ids)), A_2000=small)
    for first, second in (('training_pool', 'development'), ('training_pool', 'internal'),
                          ('development', 'internal')):
        if np.intersect1d(base_ids[splits[first]], base_ids[splits[second]]).size:
            raise ValueError('A base instance crossed a holdout boundary')
    if not np.isin(small, pool).all() or len(np.union1d(np.union1d(pool, development), internal)) != len(rows):
        raise ValueError('Nested subset or complete split coverage failed')
    return splits


def load_data(device='cpu'):
    data, loaders = {}, {}
    cache_root = Path('code/V4/runs/R39_performance_model/geometry_cache')
    cache_info = json.loads((cache_root/'stats.json').read_text())
    if cache_info['definition_hash'] != file_hash(Path(__file__).parent/'local_geometry.py'):
        raise ValueError('Geometry cache definition changed')
    for split in ('train', 'val'):
        instances, raw, costs, sizes = load_split(split)
        loader = make_loader('OVRPTW', split, 128, 0, shuffle=False, cache_device=device)
        loader.drop_last = False
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), raw['pool_ids'])
        np.testing.assert_array_equal(loader.batch['costs'].cpu(), raw['costs'].astype(np.float32))
        np.testing.assert_array_equal(loader.lengths, sizes+1)
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), raw['winner'])
        if cache_info['sources'][split+'/OVRPTW'] != raw['data_hash']:
            raise ValueError('Geometry cache instance order changed')
        geometry = torch.load(cache_root/('OVRPTW_'+split+'.pt'), map_location='cpu', weights_only=True)
        if geometry.shape != (*loader.batch['node'].shape[:2], len(FIELDS)):
            raise ValueError('Geometry cache has a different node layout')
        loader.batch['node_geom'] = geometry.to(device)
        loader.batch['binary_label'] = torch.as_tensor(binary_labels(costs), device=device)
        identities = base_instance_ids(loader.batch).cpu().numpy()
        data[split] = dict(instances=instances, raw=raw, costs=costs, sizes=sizes,
                           labels=binary_labels(costs), base_ids=identities)
        loaders[split] = loader
    return data, loaders


def fit_geometry_stats(loader, rows):
    rows = torch.as_tensor(rows, dtype=torch.long, device=loader.batch['node_geom'].device)
    geometry = loader.batch['node_geom'][rows]
    mask = loader.batch['node_mask'][rows]
    valid = geometry[mask].double()
    return dict(mean=valid.mean(0).cpu().tolist(),
        std=valid.std(0, unbiased=False).clamp_min(1e-6).cpu().tolist(),
        valid_nodes=len(valid), fields=list(FIELDS), training_indices=rows.cpu().tolist(),
        training_indices_sha256=state_hash(rows.cpu()),
        fitted_on='Only actual non-tie training subset; no development/internal/original validation nodes')


def prepare(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    data, _ = load_data('cpu')
    train, val = data['train'], data['val']
    split_rows = make_splits(train['sizes'], train['base_ids'], val['base_ids'])
    fingerprints = {s:{k:d['raw'][k] for k in ('data_hash', 'label_hash', 'pool', 'pool_ids')}
                    for s,d in data.items()}
    sets = {}
    for name, rows in split_rows.items():
        source = 'val' if name == 'original_val' else 'train'
        labels = data[source]['labels'][rows]
        sets[name] = dict(source_split=source, indices=rows.tolist(), n_total=len(rows),
            n_non_ties=int((labels>=0).sum()), n_exact_ties=int((labels<0).sum()),
            n_base_groups=len(np.unique(data[source]['base_ids'][rows])),
            size_counts={str(n):int((data[source]['sizes'][rows]==n).sum()) for n in np.unique(data[source]['sizes'][rows])},
            base_ids=data[source]['base_ids'][rows].tolist())
    manifest = dict(seed=2, problem='OVRPTW', sources=fingerprints, sets=sets,
        requested_nominal_counts=dict(training_pool=8000, development=1000, internal=1000, original_val=1000, A_2000=2000),
        grouping='Existing input-derived base IDs: node-order/D4 canonicalized, attributes retained, rounded to 1e-6; no labels',
        explicit_base_id_metadata=False, stratification='Exact true customer count on base-instance groups; no winner stratification',
        distribution_metadata=None, distribution_metadata_reason='All instances have only depot/coordinate/demand/capacity/service/TW fields',
        unique_original_train_groups=len(np.unique(train['base_ids'])),
        unique_original_val_groups=len(np.unique(val['base_ids'])),
        associated_copies_can_cross_sets=False, original_validation_used_for_selection=False,
        test_read=False, input_field_names=list(train['instances'][0]))
    if len(np.unique(train['base_ids'])) == len(train['sizes']):
        for name, expected in manifest['requested_nominal_counts'].items():
            if sets[name]['n_total'] != expected:
                raise ValueError('Unique-group split has incorrect nominal count')
    path = root/'split_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Refusing to replace a locked split manifest')
    dump(path, manifest)
    feature_cache = {}
    for source, values in data.items():
        dictionaries = [tree_features(x) for x in values['instances']]
        fields = list(dictionaries[0])
        feature_cache[source] = np.array([[row[k] for k in fields] for row in dictionaries], dtype=np.float64)
    np.savez_compressed(root/'input_statistics.npz', **feature_cache, fields=np.asarray(fields))
    comparisons=[]
    reference = feature_cache['train'][split_rows['training_pool']]
    for name, rows in split_rows.items():
        source = sets[name]['source_split']
        values = feature_cache[source][rows]
        for j, field in enumerate(fields):
            column = values[:,j]
            pooled = np.sqrt((column.var()+reference[:,j].var())/2)
            comparisons.append(dict(set=name, feature=field, n=len(column), mean=float(column.mean()),
                std=float(column.std()), q10=float(np.quantile(column,.1)), median=float(np.median(column)),
                q90=float(np.quantile(column,.9)), minimum=float(column.min()), maximum=float(column.max()),
                standardized_mean_difference_vs_training_pool=float((column.mean()-reference[:,j].mean())/max(pooled,1e-12)),
                used_as_new_model_input=False))
        labels=data[source]['labels'][rows]
        strict=labels>=0
        for klass, name_suffix in ((0,'MOEL_win_fraction'), (1,'MTL_win_fraction')):
            comparisons.append(dict(set=name,feature=name_suffix,n=int(strict.sum()),
                mean=float((labels[strict]==klass).mean()),used_as_new_model_input=False))
    write_csv(root/'split_comparison.csv', comparisons)
    print('[R49 splits] '+json.dumps({k:(v['n_total'],v['n_non_ties']) for k,v in sets.items()}), flush=True)
    return manifest


def load_manifest(root, data):
    manifest = json.loads((root/'split_manifest.json').read_text())
    for source, values in data.items():
        for field in ('data_hash', 'label_hash'):
            if manifest['sources'][source][field] != values['raw'][field]:
                raise ValueError('Locked data or labels changed')
    for name, values in manifest['sets'].items():
        rows = np.asarray(values['indices'])
        np.testing.assert_array_equal(data[values['source_split']]['base_ids'][rows], values['base_ids'])
    return manifest


class ShuffledIndexStream:
    """Concatenate shuffled complete passes; carry tails into full optimizer batches."""

    def __init__(self, rows, seed=2):
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        if self.rows.numel() == 0 or self.rows.unique().numel() != self.rows.numel():
            raise ValueError('Sampling stream requires distinct training rows')
        self.generator = torch.Generator().manual_seed(seed)
        self.remaining = torch.empty(0,dtype=torch.long)
        self.passes_started = 0

    def next_batch(self, batch_size=128):
        pieces=[]
        needed=batch_size
        while needed:
            if not len(self.remaining):
                self.remaining=self.rows[torch.randperm(len(self.rows),generator=self.generator)]
                self.passes_started+=1
            take=min(needed,len(self.remaining))
            pieces.append(self.remaining[:take])
            self.remaining=self.remaining[take:]
            needed-=take
        return torch.cat(pieces)
