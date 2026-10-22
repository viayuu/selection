"""Narrow CPU fixtures: strict native FP64 labels and TRAIN-only provenance."""

import copy
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import r97_data as old
from . import r100_data as data


def tables(costs=None):
    if costs is None:
        costs = np.array([[2., 3., 1.], [4., 2., 1.], [1., 1., 2.],
            [1.+2.**-40, 1., 2.], [3., 1., 2.], [2., 3., 1.]])
    rows = len(costs)
    context = SimpleNamespace(x=np.arange(rows*2, dtype=np.float32).reshape(rows, 2),
        source_ids=np.arange(rows, dtype=np.int64)//2+100, kept_columns=np.array([0, 2]),
        pool=('C', 'A', 'reld_mt'), pool_ids=(2, 0, 3))
    original = dict(problem='TSP', data_root='/fixture', label_splits=['train'])
    context.metadata = dict(identity=original, identity_sha256=old._json_sha(original))
    labels = data.strict_pair_labels(costs, context.pool_ids)
    support = data.pair_support(labels, context.pool, context.pool_ids)
    identity = dict(schema=data.SCHEMA, problem='TSP', data_root='/fixture', rows=rows,
        pool=list(context.pool), pool_ids=list(context.pool_ids), pairs=[list(p) for p in data.legal_pairs(context.pool_ids)],
        label_splits=['train'], row_order=data.ROW_ORDER, label_rule=data.LABEL_RULE,
        context_subsampling=False, full_pool_winner_filter=False,
        context_metadata_sha256=old._json_sha(context.metadata), context_identity_sha256=context.metadata['identity_sha256'],
        source_ids_values_sha256=data.source_ids_sha256(context.source_ids),
        kept_columns_sha256=old._json_sha(context.kept_columns.tolist()), pair_support_sha256=old._json_sha(support))
    metadata = dict(complete=True, identity=identity, identity_sha256=old._json_sha(identity), pair_support=support)
    return context, data.RelationTable(labels, metadata)


def make_fixture(root):
    release, contexts, output = (root / name for name in ('release', 'contexts', 'relations'))
    native = release / 'TSPtrain'
    native.mkdir(parents=True)
    context, relations = tables()
    costs = np.array([[2., 3., 1.], [4., 2., 1.], [1., 1., 2.],
        [1.+2.**-40, 1., 2.], [3., 1., 2.], [2., 3., 1.]])
    winners = costs.argmin(1).astype(np.int64)
    np.savez(native / 'input_metadata.npz', source_index=context.source_ids)
    with (native / 'raw_label.pkl').open('wb') as stream:
        pickle.dump({str(i): dict(cost=costs[i], ind=int(winners[i])) for i in reversed(range(len(costs)))}, stream)
    metadata = dict(problem='TSP', split='train', instances=len(costs), cost_dtype='float64',
        methods=list(context.pool), global_solver_ids=list(context.pool_ids),
        files={name: dict(sha256=old._file_identity(native / name)['sha256'])
               for name in ('raw_label.pkl', 'input_metadata.npz')})
    old._atomic_json(native / 'metadata.json', metadata)
    cache = root / 'train_cache_metadata.json'
    old._atomic_json(cache, dict(pool=list(context.pool), pool_ids=list(context.pool_ids),
        raw_cost_dtype='float64', source_files=[old._file_identity(native / 'raw_label.pkl')]))
    directory = contexts / 'TSP'
    directory.mkdir(parents=True)
    transform = old.FeatureTransform(('a', 'constant', 'b'), (0, 2))
    identity = dict(schema=old.SCHEMA, problem='TSP', data_root=str(release), rows=len(costs),
        transform=transform.to_dict(), pool=list(context.pool), pool_ids=list(context.pool_ids),
        label_splits=['train'], winner_rule=old.WINNER_RULE, row_order=data.ROW_ORDER,
        context_subsampling=False, class_quotas=False, source_files=dict(
            train_input_metadata=old._file_identity(native / 'input_metadata.npz'),
            train_cache_metadata=old._file_identity(cache)))
    artifacts = {key: old._write_array(directory / name, values.shape, values.dtype, [(0, values)])
        for key, name, values in (('x', 'X.npy', context.x), ('y', 'y.npy', winners),
                                  ('source_ids', 'source_ids.npy', context.source_ids))}
    old._atomic_json(directory / 'identity.json', dict(complete=True, identity=identity,
        identity_sha256=old._json_sha(identity), artifacts=artifacts))
    for split in ('val', 'test', 'LIB'):
        forbidden = release / ('TSP'+split)
        forbidden.mkdir()
        (forbidden / 'raw_label.pkl').write_bytes(b'FORBIDDEN')
    return release, contexts, output, costs


class RelationTests(unittest.TestCase):
    def test_fp64_exact_pair_ties_third_winners_and_global_orientation(self):
        context, table = tables()
        self.assertEqual(data.legal_pairs(context.pool_ids), ((1, 0), (1, 2), (0, 2)))
        self.assertEqual(table.labels[0, 0], 1)  # Third method wins the full pool; row retained.
        self.assertEqual(table.labels[2, 0], -1)
        self.assertEqual(table.labels[3, 0], 0)  # FP32 would turn this strict relation into a tie.
        self.assertEqual(np.float32(1.+2.**-40), np.float32(1.))
        self.assertEqual(table.metadata['pair_support'][0]['tie_rows'], 1)
        with self.assertRaisesRegex(ValueError, 'FP64'):
            data.strict_pair_labels(np.ones((3, 3), dtype=np.float32), context.pool_ids)

    def test_binding_rejects_source_reorder_columns_pool_and_heldout_labels(self):
        context, table = tables()
        for name, value in (('source_ids', context.source_ids[::-1]), ('kept_columns', np.array([2, 0])),
                            ('pool', context.pool[::-1])):
            changed = copy.copy(context)
            setattr(changed, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'binding'):
                data.validate_relation_binding(changed, table.metadata)
        changed = copy.deepcopy(table.metadata)
        changed['identity']['label_splits'] = ['val']
        changed['identity_sha256'] = old._json_sha(changed['identity'])
        with self.assertRaisesRegex(ValueError, 'identity'):
            data.validate_relation_binding(context, changed)

    def test_cpu_preparation_train_only_and_hash_bound_without_fp32_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            release, contexts, output, costs = make_fixture(Path(temp))
            original, opened = Path.open, []
            def guarded(path, *args, **kwargs):
                opened.append(path)
                if any(part in ('TSPval', 'TSPtest', 'TSPLIB') for part in path.parts) or path.name in ('costs.npy', 'dataset.pkl'):
                    raise AssertionError('Opened held-out supervision or FP32/graph cache')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded):
                data.prepare_relations(release, contexts, output)
            with data.load_relations(output, 'TSP') as table, old.load_context(contexts, 'TSP') as context:
                data.validate_relation_binding(context, table.metadata)
                self.assertFalse(table.labels.flags.writeable)
                np.testing.assert_array_equal(table.labels, data.strict_pair_labels(costs, context.pool_ids))
            self.assertIn(release / 'TSPtrain/raw_label.pkl', opened)
            before = old._file_identity(output / 'TSP/relations.npy')
            data.prepare_relations(release, contexts, output)
            self.assertEqual(before, old._file_identity(output / 'TSP/relations.npy'))
            with (release / 'TSPtrain/raw_label.pkl').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'frozen cache'):
                data.prepare_relations(release, contexts, output)
            self.assertEqual(before, old._file_identity(output / 'TSP/relations.npy'))

    def test_saved_relation_load_does_not_reopen_sources_and_fails_on_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            release, contexts, output, _ = make_fixture(Path(temp))
            data.prepare_relations(release, contexts, output)
            original = Path.open
            def guarded(path, *args, **kwargs):
                if path.is_relative_to(release) or path.is_relative_to(contexts):
                    raise AssertionError('Saved relation load reopened sources')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded), data.load_relations(output, 'TSP'):
                pass
            path = output / 'TSP/relations.npy'
            array = np.load(path).copy()
            array[0, 0] = 0
            np.save(path, array)
            with self.assertRaisesRegex(ValueError, 'SHA'):
                data.load_relations(output, 'TSP')


if __name__ == '__main__':
    unittest.main()
