"""Small CPU cache-identity fixtures, with NO real data or supervision files."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from . import r102_features as module
from .r102_donor import PROBLEM_NAMES, ROUTING_PROBLEMS, SPEC, VARIANTS, WIDTH_BY_PROBLEM
from .r102_transfer import RELD_SHA256, digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def val_lock(arm='A'):
    return dict(experiment='R102', arm=arm, selection_split='val', selected_without_test=True,
                per_problem_assembly=False, selected_update=1440,
                checkpoint='/synthetic/R102/' + arm + '/best.pt', checkpoint_sha256='a' * 64,
                inference_rule='fixed score_difference argmin; no per-task assembly')


class TinyDonor(nn.Module):
    def __init__(self, arm='A'):
        super().__init__()
        self.arm, self.variant, self.calls = arm, VARIANTS[arm], []
        self.register_buffer('placeholder', torch.zeros(1))
        vrp = dict(variant='pretrained', output_dim=128, supported_problems=list(ROUTING_PROBLEMS),
                   checkpoint_sha256=RELD_SHA256)
        self.identity = dict(format=SPEC['format'], arm=arm, variant=self.variant,
            supported_problems=list(PROBLEM_NAMES), widths=WIDTH_BY_PROBLEM,
            external_pretraining_by_component=dict(TSP=arm == 'B', ATSP=arm == 'B', VRP=True),
            components={p: dict(variant='pretrained' if arm == 'B' else 'random',
                                output_dim=WIDTH_BY_PROBLEM[p]) for p in ('TSP', 'ATSP')}, spec=SPEC)
        self.identity['components']['VRP'] = vrp

    def provenance(self):
        return copy.deepcopy(self.identity)

    def forward(self, batch):
        problem = batch['problem_name']
        if problem not in ('TSP', 'ATSP'):
            raise AssertionError('Routing features must never be extracted')
        if len(torch.unique(batch['n'])) != 1:
            raise AssertionError('Native batches must group exact N')
        self.calls.append((problem, batch['n'].tolist()))
        raw = batch['node'][..., :1] if problem == 'TSP' else batch['matrix'].sum(-1, keepdim=True)
        return raw.expand(-1, -1, WIDTH_BY_PROBLEM[problem]).float().contiguous()


class BaseCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text())
        self.problem, self.split, self.size = (self.metadata[k] for k in ('problem', 'split', 'rows'))
        self.kind, self.pool, self.pool_ids = (self.metadata[k] for k in ('kind', 'pool', 'pool_ids'))
        self.lengths = np.load(self.directory / 'n.npy', mmap_mode='r', allow_pickle=False)
        key = 'matrix' if self.kind == 'matrix' else 'node'
        self.key, self.values = key, np.load(self.directory / (key + '.npy'), mmap_mode='r', allow_pickle=False)
        self.raw, self.closed = object(), False

    def batch(self, rows, device='cpu'):
        lengths = torch.from_numpy(np.array(self.lengths[rows], copy=True))
        maximum = int(lengths.max())
        view = self.values[:, :maximum, :maximum] if self.key == 'matrix' else self.values[:, :maximum]
        result = dict(problem_id=self.metadata['problem_id'], problem_name=self.problem,
                      kind=self.kind, d_in=self.metadata['d_in'], n=lengths.to(device),
                      node_mask=(torch.arange(maximum)[None] < lengths[:, None]).to(device),
                      untouched=self.raw)
        result[self.key] = torch.from_numpy(np.array(view[rows], copy=True)).to(device)
        return result

    def close(self):
        if not self.closed:
            self.values._mmap.close()
            self.lengths._mmap.close()
            self.closed = True


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw, self.original, self.routing, self.features = (self.root / p for p in ('raw', 'original', 'routing', 'features'))
        self.ready = self.root / 'original_R94_B_ready.json'
        self.donor = TinyDonor()
        self.manifest = self.sources('train')
        for name, value in (('R84_CACHE_ROOT', self.original), ('R94_READY', self.ready),
                            ('R84_MANIFEST', self.root / 'frozen_manifest.json')):
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def sources(self, split, canonical_metadata=True):
        records = {}
        lengths = np.array([5, 3, 5, 4], dtype=np.int64)
        for problem in PROBLEM_NAMES:
            directory = self.raw / split / problem
            canonical = self.original / split / problem
            directory.mkdir(parents=True)
            canonical.mkdir(parents=True)
            width = None if problem == 'ATSP' else 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
            shape = [4, 5, 5] if problem == 'ATSP' else [4, 5, width]
            key = 'matrix' if problem == 'ATSP' else 'node'
            arrays = dict(n=dict(file='n.npy', shape=[4], dtype='int64'),
                costs=dict(file='costs.npy', shape=[4, 2], dtype='float64'),
                winner=dict(file='winner.npy', shape=[4], dtype='int64'),
                coord_dist=dict(file='coord_dist.npy', shape=[4], dtype='int64'))
            arrays[key] = dict(file=key + '.npy', shape=shape, dtype='float32')
            if problem != 'ATSP':
                arrays['node_geom'] = dict(file='node_geom.npy', shape=[4, 5, 12], dtype='float32')
            record = dict(problem=problem, problem_id=PROBLEM_NAMES.index(problem), split=split,
                data_root='/do_not_open_sources', rows=4, true_length_min=3, true_length_max=5,
                length_definition='original model nodes', kind='matrix' if problem == 'ATSP' else 'coord',
                d_in=width, pool=['a', 'reld_mt'], pool_ids=[0, 1], raw_cost_dtype='float64',
                winner_source='native FP64 filtered pool', coord_augment=0, coord_dist='original metadata',
                solver_aliases={}, scenario_id='synthetic_frozen_r84',
                source_files=[dict(path='/forbidden/dataset.pkl'), dict(path='/forbidden/raw_label.pkl')],
                scenario_global_solvers=['a', 'reld_mt'], scenario_global_solver_ids={'a': 0, 'reld_mt': 1},
                model_global_solvers=['a', 'reld_mt'], scenario_solver_aliases={},
                directory=str(directory), cache_root=str(self.raw), cache_format='r76_scenario_v1',
                cache_version=1, complete=True, rows_written=4, created_utc='frozen', seconds=1.,
                feature_sources=[dict(path='/forbidden/original_cache_code.py')], arrays=arrays,
                cbits=[0, 0, 0, 0, 0], problem_desc=[0.] * 16, mask=[1, 1],
                geometry_calculation_dtype='float64', geometry_cache_dtype='float32',
                geometry_fields=[f'k{k}_{n}' for k in (4, 8, 16)
                                 for n in ('radius', 'mean_distance', 'distance_cv', 'anisotropy')],
                geometry_scales=[4, 8, 16], geometry_moments=dict(count=0, sum=[0.] * 12, squared_sum=[0.] * 12),
                test_lock_root=None, reuse_policy='frozen', reused_input_rows=0,
                newly_computed_geometry_rows=4, completed_utc='frozen', storage_bytes=1)
            for path in ((directory, canonical) if canonical_metadata else (directory,)):
                write_json(path / 'metadata.json', record)
                np.save(path / 'n.npy', lengths)
            raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100
            np.save(directory / (key + '.npy'), raw)
            records[problem] = copy.deepcopy(record)
            if problem in ROUTING_PROBLEMS:
                target = self.routing / split / problem
                source = dict(directory=str(directory), metadata_sha256=digest(record), inputs={})
                for name in ('n', 'node'):
                    path = directory / (name + '.npy')
                    source['inputs'][name] = dict(path=str(path), bytes=path.stat().st_size,
                                                  mtime_ns=path.stat().st_mtime_ns, spec=arrays[name])
                old_lock = None
                if split == 'test':
                    old = dict(val_lock('B'), experiment='R94')
                    old_lock = dict(record=old, sha256=digest(old))
                write_json(target / 'metadata.json', dict(format='r94_frozen_nodes_v1',
                    problem=problem, split=split, rows=4, rows_written=4, complete=True,
                    shape=[4, 5, 128], dtype='float16', file='nco_nodes.npy',
                    input_only=True, source_cache=source, donor=self.donor.provenance()['components']['VRP'],
                    extract_order=module.EXTRACT_ORDER, test_lock=old_lock))
                values = np.arange(4 * 5 * 128, dtype=np.float16).reshape(4, 5, 128)
                values[np.arange(5)[None] >= lengths[:, None]] = 0
                np.save(target / 'nco_nodes.npy', values)
        write_json(self.ready, dict(experiment='R94', arm='B', complete=True,
            labels_used=False, decoder_executed=False, splits=['train', 'val'],
            frozen_donor=self.donor.provenance()['components']['VRP'], feature_root=str(self.routing)))
        return dict(splits={split: records})

    def prepare(self, **overrides):
        options = dict(cache_root=self.raw, feature_root=self.features, split='train',
                       donor=self.donor, device='cpu', routing_feature_root=self.routing,
                       batch_size=2, manifest=self.manifest)
        options.update(overrides)
        return module.prepare_features(**options)

    def wrapper(self, problem, split='train', **overrides):
        base = BaseCache(self.raw / split / problem)
        self.addCleanup(base.close)
        wrapper = module.FrozenFeatureCache(base, self.features / split / problem,
            expected_variant='extra_random', expected_provenance=self.donor.provenance(),
            routing_feature_directory=self.routing / split / problem, **overrides)
        self.addCleanup(wrapper.close)
        return wrapper

    def test_all18_receipts_native_only_reads_and_original_r94_bytes_untouched(self):
        before = {path: path.read_bytes() for path in self.routing.rglob('*') if path.is_file()}
        real_load, opened = np.load, []

        def guarded(path, *args, **kwargs):
            name = Path(path).name
            self.assertIn(name, ('n.npy', 'node.npy', 'matrix.npy', 'nco_nodes.npy'))
            if name in ('node.npy', 'matrix.npy'):
                self.assertIn(Path(path).parent.name, ('TSP', 'ATSP'))
            opened.append(name)
            return real_load(path, *args, **kwargs)

        with mock.patch.object(np, 'load', side_effect=guarded):
            receipt = self.prepare()
        self.assertEqual(set(receipt['splits']['train']), set(PROBLEM_NAMES))
        self.assertFalse(receipt['routing_reextracted'])
        self.assertEqual({p for p, _ in self.donor.calls}, {'TSP', 'ATSP'})
        self.assertEqual(opened.count('node.npy'), 1)
        self.assertEqual(opened.count('matrix.npy'), 1)
        self.assertFalse(list(self.root.rglob('costs.npy')))
        self.assertFalse(list(self.root.rglob('winner.npy')))
        for path, payload in before.items():
            self.assertEqual(path.read_bytes(), payload)
        for problem in ROUTING_PROBLEMS:
            self.assertFalse((self.features / 'train' / problem / 'nco_nodes.npy').exists())
            self.assertEqual(receipt['splits']['train'][problem]['source_kind'], 'r94b_reference')
        with mock.patch.object(self.donor, 'forward', side_effect=AssertionError('No regeneration')):
            self.prepare()

    def test_wrappers_widths_row_identity_prefetch_mask_and_readonly_close(self):
        self.prepare()
        for problem in ('TSP', 'ATSP', 'CVRP'):
            cache = self.wrapper(problem)
            self.assertIs(cache.raw, cache.base_cache.raw)
            self.assertIs(cache.pool_ids, cache.base_cache.pool_ids)
            self.assertFalse(cache._features.flags.writeable)
            with mock.patch.object(module.os, 'posix_fadvise', wraps=os.posix_fadvise) as advice:
                batch = cache.batch(['1', '3', '1'])
            width, array = WIDTH_BY_PROBLEM[problem], cache._features
            self.assertEqual(tuple(batch['nco_nodes'].shape), (3, 4, width))
            self.assertIs(batch['nco_node_mask'], batch['node_mask'])
            self.assertIs(batch['untouched'], cache.raw)
            self.assertEqual(batch['nco_nodes'][~batch['node_mask']].count_nonzero(), 0)
            torch.testing.assert_close(batch['n'], torch.tensor([3, 4, 3]))
            torch.testing.assert_close(batch['nco_nodes'][0], batch['nco_nodes'][2], rtol=0, atol=0)
            self.assertEqual([call.args for call in advice.call_args_list], [
                (cache._feature_fd, int(array.offset) + row * int(array.strides[0]),
                 4 * width * 2, os.POSIX_FADV_WILLNEED) for row in (1, 3)])
            if problem in ('TSP', 'ATSP'):
                expected = self.donor(cache.base_cache.batch([1])).half()
                torch.testing.assert_close(cache.batch([1])['nco_nodes'], expected, rtol=0, atol=0)
            fd = cache._feature_fd
            cache.close()
            self.assertTrue(cache.base_cache.closed)
            with self.assertRaises(OSError):
                os.fstat(fd)
            with self.assertRaises(RuntimeError):
                cache.batch([0])

    def test_missing_full_header_and_permuted_n_fail_even_with_matching_manifest(self):
        path = self.raw / 'train' / 'TSP' / 'metadata.json'
        original = json.loads(path.read_text())
        broken = copy.deepcopy(original)
        broken.pop('feature_sources')
        write_json(path, broken)
        matching = copy.deepcopy(self.manifest)
        matching['splits']['train']['TSP'] = broken
        with self.assertRaisesRegex(ValueError, 'missing fields'):
            self.prepare(manifest=matching)
        write_json(path, original)
        np.save(self.raw / 'train' / 'TSP' / 'n.npy', np.array([3, 5, 5, 4], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, 'full n vector'):
            self.prepare()
        self.assertFalse(self.features.exists())

    def test_partial_and_identity_change_fail_preserving_every_byte(self):
        with mock.patch.object(self.donor, 'forward', side_effect=RuntimeError('tiny interruption')):
            with self.assertRaisesRegex(RuntimeError, 'tiny interruption'):
                self.prepare()
        before = {p: p.read_bytes() for p in self.features.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.prepare()
        for path, value in before.items():
            self.assertEqual(path.read_bytes(), value)
        other = self.root / 'different_complete_root'
        self.prepare(feature_root=other)
        with self.assertRaisesRegex(ValueError, 'identity-changing'):
            self.prepare(feature_root=other, donor=TinyDonor('B'))

    def test_test_access_requires_new_matching_r102_val_lock_not_old_r94(self):
        with self.assertRaises(PermissionError):
            self.prepare(split='test')
        with self.assertRaises(PermissionError):
            self.prepare(split='test', test_lock=dict(val_lock(), experiment='R94'))
        with self.assertRaises(PermissionError):
            self.prepare(split='test', test_lock=val_lock('B'))
        test_manifest = self.sources('test', canonical_metadata=False)
        write_json(module.R84_MANIFEST, test_manifest)
        result = self.prepare(split='test', manifest=test_manifest, test_lock=val_lock())
        self.assertEqual(result['splits']['test']['CVRP']['test_lock']['record']['experiment'], 'R102')
        original = json.loads((self.routing / 'test' / 'CVRP' / 'metadata.json').read_text())
        self.assertEqual(original['test_lock']['record']['experiment'], 'R94')
        cache = self.wrapper('CVRP', split='test')
        self.assertEqual(tuple(cache.batch([0])['nco_nodes'].shape), (1, 5, 128))
        changed = dict(val_lock(), selected_update=2880)
        with self.assertRaisesRegex(ValueError, 'identity-changing'):
            self.prepare(split='test', manifest=test_manifest, test_lock=changed)

    def test_estimate_cli_is_metadata_only_and_variant_tags_are_strict(self):
        with mock.patch.object(np, 'load', side_effect=AssertionError('No array reads')), \
                mock.patch.object(module, 'npy_header', side_effect=AssertionError('No payload headers')):
            value = module.estimate_features(self.raw, ('train',), self.manifest)
        self.assertEqual(len(value['tasks']), 18)
        self.assertEqual(value['new_payload_bytes'], 4 * 5 * (128 + 512) * 2)
        self.assertTrue(value['metadata_only'])
        args = module.parser().parse_args(['estimate', '--cache-root', str(self.raw)])
        self.assertEqual(args.command, 'estimate')
        self.prepare()
        base = BaseCache(self.raw / 'train' / 'ATSP')
        self.addCleanup(base.close)
        with self.assertRaisesRegex(ValueError, 'expected_variant'):
            module.FrozenFeatureCache(base, self.features / 'train' / 'ATSP', expected_variant='pretrained')


if __name__ == '__main__':
    unittest.main()
