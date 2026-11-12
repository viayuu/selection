"""Small CPU fixtures for the candidate overlay, never production input scans."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, P2I, S2I, constraint_bits, problem_descriptor
from . import r125_data as data
from .local_geometry import FIELDS, SCALES
from .r75_cache import _specs
from .r76_cache import CACHE_FORMAT, CACHE_VERSION


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_filter_retains_order_ids_and_rejects_illegal_removals():
    methods = ['BQ', 'ELG', 'LEHD', 'OMNI']
    assert data.filter_pool('TSP', methods, [S2I[n] for n in methods]) == (
        ['ELG', 'OMNI'], [S2I['ELG'], S2I['OMNI']], [1, 3])
    assert data.filter_pool('OVRP', ['MTPOMO', 'reld_mt'], [10, 16]) == (
        ['MTPOMO', 'reld_mt'], [10, 16], [0, 1])
    with unittest.TestCase().assertRaises(ValueError):
        data.filter_pool('OVRP', methods, [S2I[n] for n in methods])


def test_fp64_targets_rebuild_ties_ranks_oracle_and_train_sbs():
    costs = np.array([[1., 1.00000002, 1.00000001, 3.], [1., 6., 2., 2.]], dtype=np.float64)
    kept = costs[:, [1, 2, 3]]
    targets = data.derive_targets(kept)
    np.testing.assert_array_equal(targets['winner'], [1, 1])
    np.testing.assert_array_equal(targets['ranking'], [[1, 0, 2], [1, 2, 0]])
    np.testing.assert_array_equal(targets['rank'], [[1, 0, 2], [2, 0, 1]])
    np.testing.assert_array_equal(targets['oracle_cost'], [1.00000001, 2.])
    assert kept[0].astype(np.float32).argmin() == 0
    baseline = data.fit_sbs(kept, ['ELG', 'OMNI', 'T2T'], targets['winner'])
    assert baseline['sbs_solver'] == 'OMNI'
    assert baseline['selection_split'] == 'train'


def test_export_keeps_inputs_and_results_linked_and_raw_labels_aligned(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'overlay'
    source.mkdir()
    methods = ['BQ', 'ELG', 'LEHD', 'OMNI']
    original = np.array([[1., 4., 2., 3.], [1., 2., 3., 2.]], dtype=np.float64)
    np.savez(source / 'targets.npz', costs=original)
    with (source / 'raw_label.pkl').open('wb') as stream:
        pickle.dump({str(i): dict(time=[0.] * 4) for i in range(2)}, stream)
    for name in ('dataset.pkl', 'index_mapping.csv', 'input_metadata.npz'):
        (source / name).write_bytes(b'unchanged input fixture')
    (source / 'results').mkdir()
    for method in methods:
        (source / 'results' / ('result_' + method + '.txt')).write_text('source result')
    metadata = dict(instances=2, methods=methods, source_index_sha256='fixed-members',
        files={n: {'sha256': 'fixed'} for n in ('dataset.pkl', 'input_metadata.npz', 'index_mapping.csv')},
        result_files={'result_' + n + '.txt': {} for n in methods})
    save_json(source / 'metadata.json', metadata)
    targets = data.derive_targets(original[:, [1, 3]])
    baseline = data.fit_sbs(targets['costs'], ['ELG', 'OMNI'], targets['winner'])
    result = data._export_changed(source, destination, 'TSP', 'test', ['ELG', 'OMNI'],
        [S2I['ELG'], S2I['OMNI']], [1, 3], baseline, 'fixture-r125')
    assert (destination / 'dataset.pkl').is_symlink()
    assert (destination / 'input_metadata.npz').resolve() == source / 'input_metadata.npz'
    assert sorted(p.name for p in (destination / 'results').iterdir()) == [
        'result_ELG.txt', 'result_OMNI.txt']
    with (destination / 'raw_label.pkl').open('rb') as stream:
        labels = pickle.load(stream)
    assert [labels[str(i)]['ind'] for i in range(2)] == [1, 0]
    assert labels['0']['rank'] == [1, 0]
    np.testing.assert_array_equal(np.load(destination / 'costs.npy'), original[:, [1, 3]])
    assert np.load(destination / 'costs.npy').dtype == np.float64
    assert json.loads((source / 'metadata.json').read_text()) == metadata
    assert sum(row['count'] for row in result['winner_migration']) == 2
    assert result['migration_summary']['removed_old_winner_count'] == 2
    assert result['migration_summary']['oracle_mean_cost_delta'] == 1.5
    assert result['migration_summary']['oracle_mean_relative_increase_pct'] == 150.


def cache_fixture(tmp_path, problem='TSP'):
    cache_root, scenario_root = tmp_path / 'cache', tmp_path / 'scenario'
    directory = cache_root / 'train' / problem
    directory.mkdir(parents=True)
    changed = problem in data.CHANGED_PROBLEMS
    methods = ['BQ', 'ELG', 'LEHD', 'OMNI'] if changed else ['MTPOMO', 'reld_mt']
    ids = [S2I['RELD_MOEL' if n == 'reld_mt' else n] for n in methods]
    costs = np.array([[1., 3., 2., 2.], [1., 2., 3., 2.]]) if changed else np.array([[2., 1.], [1., 2.]])
    specs = _specs(2, 3, 'coord', 2 if problem == 'TSP' else 8, len(methods))
    for key, spec in specs.items():
        array = np.zeros(spec['shape'], dtype=spec['dtype'])
        if key == 'n':
            array[:] = [2, 3]
        elif key == 'costs':
            array[:] = costs
        elif key == 'winner':
            array[:] = costs.argmin(axis=1)
        elif key in ('node', 'node_geom'):
            array[:] = np.arange(array.size).reshape(array.shape)
        np.save(directory / spec['file'], array)
    names = ['reld_mt' if name == 'RELD_MOEL' else name for name in GLOBAL_SOLVERS if name != 'RELD_MTL']
    global_ids = {name: S2I['RELD_MOEL' if name == 'reld_mt' else name] for name in names}
    source = dict(problem=problem, problem_id=P2I[problem], split='train', rows=2, rows_written=2,
        data_root=str(tmp_path / 'parent'), complete=True, cache_format=CACHE_FORMAT,
        cache_version=CACHE_VERSION, true_length_min=2, true_length_max=3, kind='coord',
        d_in=2 if problem == 'TSP' else 8, pool=methods, pool_ids=ids,
        scenario_id='source', scenario_global_solvers=names, scenario_global_solver_ids=global_ids,
        model_global_solvers=list(GLOBAL_SOLVERS), scenario_solver_aliases={'reld_mt': 'RELD_MOEL'},
        solver_aliases={} if changed else {'reld_mt': 'RELD_MOEL'},
        mask=[int(i in ids) for i in range(len(GLOBAL_SOLVERS))], cbits=constraint_bits(problem),
        problem_desc=problem_descriptor(problem), arrays=specs, geometry_fields=list(FIELDS),
        geometry_scales=list(SCALES), geometry_calculation_dtype='float64', geometry_cache_dtype='float32')
    save_json(directory / 'metadata.json', source)
    pool, pool_ids, kept = data.filter_pool(problem, methods, ids)
    scenario = dict(scenario_id='r125-fixture', global_solvers=[n for n in names if n not in data.REMOVED],
        global_solver_ids={n: i for n, i in global_ids.items() if n not in data.REMOVED},
        solver_aliases={'reld_mt': 'RELD_MOEL'}, problems={problem: dict(methods=pool, global_solver_ids=pool_ids)})
    record = data._overlay_record(source, scenario, scenario_root, directory)
    if changed:
        labels = scenario_root / (problem + 'train')
        labels.mkdir(parents=True)
        np.save(labels / 'costs.npy', np.ascontiguousarray(costs[:, kept]))
        np.save(labels / 'winner.npy', costs[:, kept].argmin(axis=1))
    manifest = dict(candidate_overlay='r125', data_root=str(scenario_root), splits={'train': {problem: record}})
    return cache_root, scenario_root, manifest, source


def test_open_cache_reuses_input_memmaps_and_refreshes_supervision(tmp_path, problem):
    cache_root, scenario_root, manifest, source = cache_fixture(tmp_path, problem)
    cache = data.open_cache(cache_root, 'train', problem, manifest, scenario_root)
    try:
        batch = cache.batch(torch.tensor([1, 0]), device='cpu')
        assert Path(cache._arrays['node'].filename) == cache_root / 'train' / problem / 'node.npy'
        assert batch['node'].shape[0] == 2
        assert len(batch['pool_ids']) == 2
        assert batch['solver_mask'].shape == (2, 2)
        assert batch['mask'][0, S2I['BQ']] == 0
        assert batch['mask'][0, S2I['LEHD']] == 0
        assert cache.raw['costs'].dtype == np.float64
        assert cache.raw['winner'].dtype == np.int64
        assert cache.metadata['model_global_solvers'] == list(GLOBAL_SOLVERS)
        assert len(cache.metadata['model_global_solvers']) == 22
        assert cache.metadata['scenario_solver_aliases'] == {'reld_mt': 'RELD_MOEL'}
        np.testing.assert_array_equal(batch['ind'], cache.raw['winner'][[1, 0]])
        expected = data.comparison_targets(cache.raw['costs'])
        for key, values in expected.items():
            torch.testing.assert_close(batch[key], values[[1, 0]])
        assert json.loads((cache_root / 'train' / problem / 'metadata.json').read_text()) == source
    finally:
        cache.close()
    assert not cache._arrays


def test_prepared_config_returns_runner_tuple_without_step_checks(tmp_path):
    root = tmp_path / 'prepared'
    root.mkdir()
    records = dict(source_cache_manifest={'complete': True}, geometry_stats={'fitted_on': 'train'},
        common_full_train_sbs={'TSP': {'selection_split': 'train'}})
    for name, value in records.items():
        save_json(root / (name + '.json'), value)
    with patch.object(data, 'prepare', return_value={'experiment': 'R125'}):
        ready, manifest, geometry, baselines = data.prepared_config(SimpleNamespace(prepared_root=root))
    assert ready['experiment'] == 'R125'
    assert manifest['complete']
    assert geometry['fitted_on'] == 'train'
    assert baselines['TSP']['selection_split'] == 'train'


class TestR125Data(unittest.TestCase):
    def test_pool(self):
        test_filter_retains_order_ids_and_rejects_illegal_removals()

    def test_targets(self):
        test_fp64_targets_rebuild_ties_ranks_oracle_and_train_sbs()

    def test_migration_relative_change_is_instance_mean(self):
        original = np.array([[1., 2., 4., 3.], [5., 4., 2., 3.]])
        targets = data.derive_targets(original[:, [1, 3]])
        methods = ['ELG', 'OMNI']
        baseline = data.fit_sbs(targets['costs'], methods, targets['winner'])
        rows, summary = data._migration(original, targets, ['BQ', 'ELG', 'LEHD', 'OMNI'],
            methods, baseline, 'TSP', 'train')
        self.assertEqual(sum(row['count'] for row in rows), 2)
        self.assertEqual(summary['removed_old_winner_count'], 2)
        self.assertEqual(summary['oracle_mean_cost_delta'], 1.)
        self.assertEqual(summary['oracle_mean_relative_increase_pct'], 75.)
        self.assertEqual(summary['train_sbs_winner_pct'], 50.)
        self.assertEqual(summary['train_majority_winner_pct'], 50.)
        self.assertEqual({(row['old_winner'], row['new_winner']): row['count']
            for row in rows if row['count']}, {('BQ', 'ELG'): 1, ('LEHD', 'OMNI'): 1})

    def test_export(self):
        with tempfile.TemporaryDirectory() as root:
            test_export_keeps_inputs_and_results_linked_and_raw_labels_aligned(Path(root))

    def test_changed_cache(self):
        with tempfile.TemporaryDirectory() as root:
            test_open_cache_reuses_input_memmaps_and_refreshes_supervision(Path(root), 'TSP')

    def test_unchanged_cache(self):
        with tempfile.TemporaryDirectory() as root:
            test_open_cache_reuses_input_memmaps_and_refreshes_supervision(Path(root), 'OVRP')

    def test_prepared_config(self):
        with tempfile.TemporaryDirectory() as root:
            test_prepared_config_returns_runner_tuple_without_step_checks(Path(root))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
