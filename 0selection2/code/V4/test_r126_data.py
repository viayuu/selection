"""Small CPU fixtures for R126, with no production data writes or GPU work."""

import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, P2I, POOLS, S2I, constraint_bits, problem_descriptor
from . import r126_data as data
from .local_geometry import FIELDS, SCALES
from .r75_cache import _specs
from .r76_cache import CACHE_FORMAT, CACHE_VERSION


class SmallRelease:
    """Two rows per task/split, with deliberately unreadable input pickles."""

    def __init__(self, directory, caches=False):
        self.source, self.output = directory / 'original', directory / 'r126'
        self.cache, self.prepared = directory / 'cache', directory / 'source_prepared'
        self.root = directory / 'prepared'
        self.source.mkdir()
        self.sizes = dict(train=2, val=2, test=2)
        names = ['reld_mt' if name == 'RELD_MOEL' else name
                 for name in GLOBAL_SOLVERS if name != 'RELD_MTL']
        ids = {name: S2I[data.ALIASES.get(name, name)] for name in names}
        self.manifest = dict(format_version=1, status='complete', scenario_id=data.SOURCE_SCENARIO_ID,
            seed=2, split_sizes_per_problem=self.sizes, total_instances=18 * 6,
            global_solvers=names, global_solver_ids=ids, model_global_solvers=list(GLOBAL_SOLVERS),
            solver_aliases=dict(data.ALIASES), reserved_inactive_global_solver_ids=[S2I['RELD_MTL']],
            problems={})
        baselines, records = {}, {split: {} for split in self.sizes}
        for problem in data.PROBLEMS:
            native = list(POOLS[problem])
            kept = [i for i, name in enumerate(native) if name != 'RELD_MTL' and not (
                problem in data.B_PROBLEMS and name in ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA'))]
            methods = ['reld_mt' if native[i] == 'RELD_MOEL' else native[i] for i in kept]
            pool_ids = [ids[name] for name in methods]
            self.manifest['problems'][problem] = dict(methods=methods, global_solver_ids=pool_ids,
                source_methods=native, retained_source_slots=kept)
            costs = np.tile(np.arange(1., len(methods) + 1.), (2, 1))
            if 'MoSES_RF' in methods:
                costs[:, methods.index('MoSES_RF')] = [.5, .25]
            baselines[problem] = data.fit_sbs(costs, methods, costs.argmin(axis=1))
            for split in self.sizes:
                source = self.source / (problem + split)
                source.mkdir()
                values = costs.copy()
                if split != 'train':
                    values[:, -1] = .125
                targets = data.derive_targets(values)
                targets['sbs_train_cost'] = values[:, baselines[problem]['sbs_slot']]
                np.savez(source / 'targets.npz', **targets)
                data.write_pickle(source / 'raw_label.pkl', {str(i): dict(
                    cost=values[i].tolist(), time=[0.] * len(methods), ind=int(targets['winner'][i]))
                    for i in range(2)})
                (source / 'dataset.pkl').write_bytes(b'not a pickle: inputs must never be loaded')
                (source / 'index_mapping.csv').write_text('local_index,source_index\n0,10001\n1,10003\n')
                np.savez(source / 'input_metadata.npz', source_index=[10001, 10003], coord_dist=[0, 1])
                (source / 'results').mkdir()
                for slot, method in enumerate(methods):
                    (source / 'results' / ('result_' + method + '.txt')).write_text(
                        '\n'.join(f'{i},{values[i, slot]:.17g},0' for i in range(2)) + '\n')
                metadata = dict(problem=problem, split=split, instances=2, methods=methods,
                    global_solver_ids=pool_ids, source_index_sha256='same-frozen-members',
                    files={name: data._identity(source / name) for name in
                        ('dataset.pkl', 'input_metadata.npz', 'index_mapping.csv')},
                    result_files={'result_' + name + '.txt': {} for name in methods})
                data.write_json(source / 'metadata.json', metadata)
                if caches:
                    records[split][problem] = self._cache_record(problem, split, methods, pool_ids, values, ids)
        data.write_json(self.source / 'scenario_manifest.json', self.manifest)
        data.write_json(self.source / 'baselines.json', baselines)
        np.savez(self.source / 'split_indices.npz', train=[10001, 10003])
        data.write_json(self.source / 'split_manifest.json', {'members': 'unchanged'})
        if caches:
            self.prepared.mkdir()
            data.write_json(self.prepared / 'geometry_stats.json', {'fitted_on': 'TRAIN only', 'mean': [0.]})
            (self.prepared / 'identicalinit_seed2.pt').write_bytes(b'fixture scratch parameters, never loaded')
            data.write_json(self.prepared / 'source_cache_manifest.json', dict(splits=records))
            data.write_json(self.prepared / 'prepared.json', dict(experiment='R84', seed=2,
                data_root=str(self.source), scenario_sha256=data.file_hash(self.source / 'scenario_manifest.json'),
                rows_per_problem=self.sizes, cache_source=str(self.cache), model_params={'fixture': True},
                trainable_sha256='fixture-scratch-hash', artifact_sha256={
                    name: data.file_hash(self.prepared / name)
                    for name in ('geometry_stats.json', 'identicalinit_seed2.pt')}))

    def _cache_record(self, problem, split, methods, ids, costs, global_ids):
        directory = self.cache / split / problem
        directory.mkdir(parents=True)
        kind = 'matrix' if problem == 'ATSP' else 'coord'
        d_in = None if kind == 'matrix' else 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
        specs = _specs(2, 3, kind, d_in, len(methods))
        for key, spec in specs.items():
            values = np.zeros(spec['shape'], dtype=spec['dtype'])
            if key == 'costs':
                values[:] = costs
            elif key == 'winner':
                values[:] = costs.argmin(axis=1)
            elif key == 'n':
                values[:] = [2, 3]
            elif key in ('node', 'node_geom', 'matrix'):
                values[:] = np.arange(values.size).reshape(values.shape)
            np.save(directory / spec['file'], values)
        record = dict(problem=problem, split=split, problem_id=P2I[problem],
            rows=2, rows_written=2, complete=True, directory=str(directory),
            data_root=str(self.source), cache_format=CACHE_FORMAT, cache_version=CACHE_VERSION,
            pool=methods, pool_ids=ids, scenario_id=data.SOURCE_SCENARIO_ID,
            scenario_global_solvers=list(global_ids), scenario_global_solver_ids=global_ids,
            model_global_solvers=list(GLOBAL_SOLVERS), scenario_solver_aliases=dict(data.ALIASES),
            solver_aliases={name: data.ALIASES[name] for name in methods if name in data.ALIASES},
            mask=[int(i in ids) for i in range(len(GLOBAL_SOLVERS))],
            true_length_min=2, true_length_max=3, kind=kind, d_in=d_in, arrays=specs,
            cbits=constraint_bits(problem), problem_desc=problem_descriptor(problem),
            geometry_fields=list(FIELDS), geometry_scales=list(SCALES),
            geometry_calculation_dtype='float64', geometry_cache_dtype='float32')
        data.write_json(directory / 'metadata.json', record)
        return record

    def build(self):
        with patch.object(data, 'SPLIT_ROWS', self.sizes), contextlib.redirect_stdout(io.StringIO()):
            return data.build_release(self.source, self.output)


class TestR126Data(unittest.TestCase):
    def test_filter_only_rf_and_preserve_bq_lehd_global_ids(self):
        methods = ['BQ', 'ELG', 'LEHD', 'MoSES_RF', 'OMNI']
        ids = [S2I[name] for name in methods]
        self.assertEqual(data.filter_pool('CVRP', methods, ids), (
            ['BQ', 'ELG', 'LEHD', 'OMNI'], [S2I[name] for name in ['BQ', 'ELG', 'LEHD', 'OMNI']], [0, 1, 2, 4]))
        tsp = ['BQ', 'ELG', 'LEHD', 'OMNI']
        self.assertEqual(data.filter_pool('TSP', tsp, [S2I[name] for name in tsp])[0], tsp)
        with self.assertRaises(ValueError):
            data.filter_pool('CVRP', ['ELG', 'MoSES_RF'], [S2I['ELG'], S2I['MoSES_RF']])
        with self.assertRaises(ValueError):
            data.filter_pool('VRPB', ['MoSES_RF', 'reld_mt'], [13, 16])

    def test_fp64_ties_ranks_and_migration_instance_mean(self):
        original = np.array([[2., 1., 3.], [3., 2., 3.]], dtype=np.float64)
        targets = data.derive_targets(original[:, [0, 2]])
        baseline = data.fit_sbs(targets['costs'], ['MTPOMO', 'reld_mt'], targets['winner'])
        rows, summary = data._migration(original, targets, ['MTPOMO', 'MoSES_RF', 'reld_mt'],
            ['MTPOMO', 'reld_mt'], baseline, 'OVRP', 'train')
        self.assertEqual(sum(row['count'] for row in rows), 2)
        self.assertEqual(summary['removed_old_winner_count'], 2)
        self.assertEqual(summary['oracle_mean_cost_delta'], 1.)
        self.assertEqual(summary['oracle_mean_relative_increase_pct'], 75.)
        self.assertEqual(summary['train_sbs_winner_pct'], 100.)
        np.testing.assert_array_equal(targets['rank'], [[0, 1], [0, 1]])
        precise = data.derive_targets(np.array([[1.00000002, 1.00000001]], dtype=np.float64))
        self.assertEqual(int(precise['winner'][0]), 1)
        self.assertEqual(precise['costs'].astype(np.float32).argmin(axis=1)[0], 0)

    def test_full18_build_links_and_train_only_sbs(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SmallRelease(Path(temporary))
            original_bytes = (fixture.source / 'scenario_manifest.json').read_bytes()
            manifest = fixture.build()
            self.assertEqual(manifest['experiment'], 'R126')
            self.assertEqual(manifest['solver_aliases'], data.ALIASES)
            self.assertEqual(manifest['model_global_solvers'], list(GLOBAL_SOLVERS))
            self.assertIn('BQ', manifest['global_solvers'])
            self.assertIn('LEHD', manifest['global_solvers'])
            self.assertNotIn('MoSES_RF', manifest['global_solvers'])
            self.assertEqual(len(manifest['model_global_solvers']), 22)
            baselines = data.read_json(fixture.output / 'baselines.json')
            self.assertEqual(baselines['CVRP']['sbs_solver'], 'BQ')
            self.assertEqual(baselines['CVRP']['selection_split'], 'train')
            for problem in data.PROBLEMS:
                for split in fixture.sizes:
                    directory = fixture.output / (problem + split)
                    self.assertEqual((directory / 'dataset.pkl').resolve(),
                        fixture.source / (problem + split) / 'dataset.pkl')
                    self.assertEqual((directory / 'input_metadata.npz').resolve(),
                        fixture.source / (problem + split) / 'input_metadata.npz')
                    if problem not in data.CHANGED_PROBLEMS:
                        self.assertTrue(directory.is_symlink())
                    else:
                        self.assertTrue((directory / 'dataset.pkl').is_symlink())
                        old = data._source_costs(fixture.source / (problem + split))
                        kept = manifest['problems'][problem]['retained_parent_slots']
                        costs = np.load(directory / 'costs.npy')
                        np.testing.assert_array_equal(costs, old[:, kept])
                        self.assertEqual(costs.dtype, np.float64)
                        with (directory / 'raw_label.pkl').open('rb') as stream:
                            labels = data.pickle.load(stream)
                        np.testing.assert_array_equal([labels[str(i)]['ind'] for i in range(2)], costs.argmin(axis=1))
            with (fixture.output / 'candidate_filter_summary.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 24)
            self.assertEqual((fixture.source / 'scenario_manifest.json').read_bytes(), original_bytes)
            self.assertEqual(fixture.build(), manifest)

    def test_reject_r125_source_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SmallRelease(Path(temporary))
            derived = copy.deepcopy(fixture.manifest)
            derived.update(experiment='R125', scenario_id='R125_derived')
            data.write_json(fixture.source / 'scenario_manifest.json', derived)
            with patch.object(data, 'SPLIT_ROWS', fixture.sizes), self.assertRaises(ValueError):
                data.build_release(fixture.source, fixture.output)
            self.assertFalse(fixture.output.exists())

    def test_prepare_api_and_memmaps_without_label_or_input_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SmallRelease(Path(temporary), caches=True)
            fixture.build()
            args = SimpleNamespace(prepared_root=fixture.root, data_root=fixture.output,
                cache_root=fixture.cache, source_prepared_root=fixture.prepared)
            with patch.object(data, 'SPLIT_ROWS', fixture.sizes):
                ready, manifest, geometry, baselines = data.prepared_config(args)
                self.assertEqual(data.prepared_config(args)[0], ready)
            self.assertEqual(ready['experiment'], 'R126')
            self.assertTrue(ready['test_inputs_ready'])
            self.assertEqual(manifest['scenario_solver_aliases'], {'reld_mt': 'RELD_MOEL'})
            self.assertEqual(geometry['fitted_on'], 'TRAIN only')
            self.assertEqual(baselines['CVRP']['sbs_solver'], 'BQ')
            self.assertEqual((fixture.root / 'identicalinit_seed2.pt').resolve(),
                fixture.prepared / 'identicalinit_seed2.pt')
            for problem in ('TSP', 'CVRP', 'OVRP', 'VRPB'):
                cache = data.open_cache(fixture.cache, 'train', problem, manifest, fixture.output)
                try:
                    batch = cache.batch(torch.tensor([1, 0]), device='cpu')
                    self.assertEqual(Path(cache._arrays['node'].filename), fixture.cache / 'train' / problem / 'node.npy')
                    self.assertEqual(cache.raw['costs'].dtype, np.float64)
                    self.assertEqual(cache.raw['winner'].dtype, np.int64)
                    self.assertEqual(len(cache.metadata['model_global_solvers']), 22)
                    self.assertEqual(batch['mask'][0, S2I['MoSES_RF']], 0)
                    self.assertNotIn('MoSES_RF', cache.pool)
                    if problem in ('TSP', 'CVRP'):
                        for method in ('BQ', 'LEHD'):
                            self.assertIn(method, cache.pool)
                            self.assertEqual(batch['mask'][0, S2I[method]], 1)
                    expected = data.comparison_targets(cache.raw['costs'])
                    for key, values in expected.items():
                        torch.testing.assert_close(batch[key], values[[1, 0]])
                    np.testing.assert_array_equal(batch['ind'], cache.raw['winner'][[1, 0]])
                finally:
                    cache.close()
                self.assertFalse(cache._arrays)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
