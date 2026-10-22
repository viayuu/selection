"""Small synthetic CPU sidecars only; no released data or supervision files."""

import copy
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from . import r94_donor as donor_module
from . import r94_features as module
from .r94_donor import PROBLEM_NAMES, SUPPORTED_PROBLEMS, StaticReLDDonor
from .r94_features import FrozenFeatureCache, estimate_features, prepare_features
from .test_r94_donor import StubEncoder, synthetic_batch


def _write_json(path, record):
    path.write_text(json.dumps(record) + '\n')


def _sources(root, split='train', arrays=True):
    records = {}
    for problem in SUPPORTED_PROBLEMS:
        directory = root / split / problem
        directory.mkdir(parents=True, exist_ok=True)
        width = 4 if problem == 'CVRP' else 8
        record = dict(cache_format='r76_scenario_v1', cache_version=1, complete=True,
                      rows_written=2, rows=2, problem=problem,
                      problem_id=PROBLEM_NAMES.index(problem), split=split,
                      kind='coord', d_in=width, true_length_min=3, true_length_max=5,
                      scenario_id='synthetic_fixed_inputs', data_root=str(root / 'never_open_source'),
                      pool=['candidate_a', 'reld_mt'], pool_ids=[1, 16],
                      source_files=[dict(path='/forbidden/raw_label.pkl')],
                      arrays=dict(n=dict(file='n.npy', shape=[2], dtype='int64'),
                                  node=dict(file='node.npy', shape=[2, 5, width], dtype='float32')))
        _write_json(directory / 'metadata.json', record)
        if arrays:
            batch = synthetic_batch(problem, lengths=(3, 5), padded=5)
            np.save(directory / 'n.npy', batch['n'].numpy())
            np.save(directory / 'node.npy', batch['node'].numpy())
        records[problem] = dict(record, directory=str(directory))
    return dict(splits={split: records})


class BaseCache:
    """ScenarioCache-shaped input stub; raw supervision is an opaque sentinel."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text())
        record = self.metadata
        self.problem, self.split, self.size = record['problem'], record['split'], record['rows']
        self.kind, self.pool, self.pool_ids = record['kind'], record['pool'], record['pool_ids']
        self.raw = object()
        self.lengths = np.load(self.directory / 'n.npy', mmap_mode='r')
        self.nodes = np.load(self.directory / 'node.npy', mmap_mode='r')
        self.closed = False

    def batch(self, indices, device='cpu'):
        rows = np.asarray(indices, dtype=np.int64)
        lengths = torch.from_numpy(self.lengths[rows])
        maximum = int(lengths.max())
        return dict(problem_id=self.metadata['problem_id'], kind='coord',
                    d_in=self.metadata['d_in'], n=lengths.to(device),
                    node=torch.from_numpy(np.array(self.nodes[:, :maximum][rows], copy=True)).to(device),
                    node_mask=(torch.arange(maximum)[None] < lengths[:, None]).to(device),
                    untouched=self.raw)

    def close(self):
        if not self.closed:
            self.lengths._mmap.close()
            self.nodes._mmap.close()
            self.closed = True


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache_root, self.feature_root = self.root / 'inputs', self.root / 'features'
        self.manifest = _sources(self.cache_root)
        mock.patch.object(donor_module, '_build_encoder', side_effect=StubEncoder).start()
        mock.patch.object(donor_module, '_native_sources', return_value=('a' * 64, {'stub.py': 'b' * 64})).start()
        self.addCleanup(mock.patch.stopall)
        self.donor = StaticReLDDonor('random')

    def prepare(self, **kwargs):
        options = dict(cache_root=self.cache_root, feature_root=self.feature_root,
                       split='train', donor=self.donor, device='cpu', batch_size=2,
                       manifest=self.manifest)
        options.update(kwargs)
        return prepare_features(**options)

    def wrapper(self, problem='OVRPBLTW'):
        base = BaseCache(self.cache_root / 'train' / problem)
        self.addCleanup(base.close)
        wrapper = FrozenFeatureCache(base, self.feature_root / 'train' / problem)
        self.addCleanup(wrapper.close)
        return wrapper

    def test_preparation_opens_only_raw_inputs_and_writes_fp16_nodes(self):
        original = np.load
        opened = []

        def guarded(path, *args, **kwargs):
            self.assertIn(Path(path).name, ('n.npy', 'node.npy', 'nco_nodes.npy', '.nco_nodes.partial.npy'))
            self.assertIn(kwargs.get('mmap_mode'), ('r', 'r+'))
            opened.append(Path(path).name)
            return original(path, *args, **kwargs)

        with mock.patch.object(np, 'load', side_effect=guarded):
            manifest = self.prepare()
        self.assertTrue(manifest['complete'])
        self.assertEqual(set(manifest['splits']['train']), set(SUPPORTED_PROBLEMS))
        self.assertEqual(opened.count('n.npy'), 16)
        self.assertEqual(opened.count('node.npy'), 16)
        self.assertFalse((self.feature_root / 'train' / 'TSP').exists())
        self.assertFalse((self.feature_root / 'train' / 'ATSP').exists())
        for problem in SUPPORTED_PROBLEMS:
            path = self.feature_root / 'train' / problem
            metadata = json.loads((path / 'metadata.json').read_text())
            self.assertTrue(metadata['complete'])
            self.assertEqual(metadata['rows_written'], 2)
            self.assertTrue(metadata['input_only'])
            self.assertEqual(metadata['extract_order'], module.EXTRACT_ORDER)
            self.assertEqual(metadata['donor']['variant'], 'random')
            self.assertFalse((path / '.nco_nodes.partial.npy').exists())
            values = np.load(path / 'nco_nodes.npy', mmap_mode='r')
            self.assertEqual(values.dtype, np.float16)
            self.assertEqual(values.shape, (2, 5, 128))
            self.assertEqual(int(np.count_nonzero(values[0, 3:])), 0)
            values._mmap.close()

    def test_wrapper_proxies_raw_and_crops_duplicate_out_of_order_rows(self):
        self.prepare()
        wrapper = self.wrapper()
        base = wrapper.base_cache
        self.assertIs(wrapper.raw, base.raw)
        self.assertIs(wrapper.lengths, base.lengths)
        self.assertIs(wrapper.pool, base.pool)
        self.assertIs(wrapper.pool_ids, base.pool_ids)
        self.assertIs(wrapper.metadata, base.metadata)
        self.assertEqual(wrapper.size, 2)
        self.assertIsInstance(wrapper._features, np.memmap)
        self.assertFalse(wrapper._features.flags.writeable)
        batch = wrapper.batch(['1', '0', '1'], 'cpu')
        self.assertIs(batch['nco_node_mask'], batch['node_mask'])
        self.assertIs(batch['untouched'], base.raw)
        self.assertEqual(batch['nco_nodes'].dtype, torch.float16)
        self.assertEqual(batch['nco_nodes'].shape, (3, 5, 128))
        self.assertEqual(wrapper.batch([0])['nco_nodes'].shape, (1, 3, 128))
        self.assertTrue(batch['nco_nodes'].is_contiguous())
        expected = self.donor(batch).half()
        torch.testing.assert_close(batch['nco_nodes'], expected, rtol=0, atol=0)
        wrapper.close()
        wrapper.close()
        self.assertTrue(base.closed)
        with self.assertRaises(RuntimeError):
            wrapper.batch([0])

    def test_precise_async_prefetch_preserves_values_order_and_file_bytes(self):
        self.prepare()
        wrapper = self.wrapper('CVRP')
        path = self.feature_root / 'train/CVRP/nco_nodes.npy'
        before = path.read_bytes()
        fd = wrapper._feature_fd
        with mock.patch.object(module.os, 'posix_fadvise', wraps=module.os.posix_fadvise) as advice:
            batch = wrapper.batch([1, 0, 1])
        calls = [call.args for call in advice.call_args_list]
        stride = int(wrapper._features.strides[0])
        offset = int(wrapper._features.offset)
        self.assertEqual(calls, [(fd, offset, stride, os.POSIX_FADV_WILLNEED),
                                 (fd, offset + stride, stride, os.POSIX_FADV_WILLNEED)])
        torch.testing.assert_close(batch['nco_nodes'], self.donor(batch).half(), rtol=0, atol=0)
        self.assertEqual(before, path.read_bytes())
        wrapper.close()
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_completed_tasks_resume_without_forward_and_manifest_test_append_is_stable(self):
        self.prepare()
        changed = copy.deepcopy(self.manifest)
        changed['splits']['test'] = {'some_later_record': {'complete': True}}
        changed['created_after_test'] = True
        with mock.patch.object(self.donor, 'forward', side_effect=AssertionError('unnecessary extraction')):
            result = self.prepare(manifest=changed)
        self.assertTrue(result['complete'])

    def test_shared_manifest_does_not_override_node_local_runtime_cache(self):
        local = self.root / 'node_local'
        shutil.copytree(self.cache_root, local)
        self.prepare(cache_root=local, manifest=self.manifest)
        target = self.feature_root / 'train' / 'CVRP'
        metadata = json.loads((target / 'metadata.json').read_text())
        self.assertEqual(metadata['source_cache']['directory'], str(local / 'train' / 'CVRP'))
        base = BaseCache(local / 'train' / 'CVRP')
        wrapper = FrozenFeatureCache(base, target, expected_variant='random',
                                     expected_provenance=self.donor.provenance())
        try:
            torch.testing.assert_close(wrapper.batch([1, 0])['nco_nodes'],
                                       self.donor(base.batch([1, 0])).half(), rtol=0, atol=0)
        finally:
            wrapper.close()

    def test_partial_task_resumes_from_flushed_sequential_cursor(self):
        forward = self.donor.forward
        calls = []

        def interrupt(batch):
            calls.append(batch['node'].clone())
            if len(calls) == 2:
                raise RuntimeError('synthetic interruption')
            return forward(batch)

        with mock.patch.object(self.donor, 'forward', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                self.prepare(batch_size=1)
        target = self.feature_root / 'train' / 'CVRP'
        progress = json.loads((target / 'metadata.json').read_text())
        self.assertFalse(progress['complete'])
        self.assertEqual(progress['rows_written'], 1)
        self.assertFalse((target / 'nco_nodes.npy').exists())
        base = BaseCache(self.cache_root / 'train' / 'CVRP')
        try:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                FrozenFeatureCache(base, target)
        finally:
            base.close()
        with mock.patch.object(self.donor, 'forward', wraps=forward) as resumed:
            self.prepare(batch_size=1)
        self.assertEqual(resumed.call_count, 31)
        torch.testing.assert_close(resumed.call_args_list[0].args[0]['node'], calls[1], rtol=0, atol=0)

    def test_global_exact_n_order_interleaved_rows_and_partial_resume_alignment(self):
        directory = self.cache_root / 'train' / 'CVRP'
        batch = synthetic_batch('CVRP', lengths=(5, 3, 5, 3, 4, 3), padded=5)
        record = json.loads((directory / 'metadata.json').read_text())
        record.update(rows=6, rows_written=6)
        record['arrays']['n']['shape'] = [6]
        record['arrays']['node']['shape'] = [6, 5, 4]
        _write_json(directory / 'metadata.json', record)
        np.save(directory / 'n.npy', batch['n'].numpy())
        np.save(directory / 'node.npy', batch['node'].numpy())
        self.manifest['splits']['train']['CVRP'] = dict(record, directory=str(directory))
        forward, seen = self.donor.forward, []

        def interrupt(group):
            seen.append(group['node'].clone())
            self.assertTrue((group['n'] == group['n'][0]).all())
            self.assertLessEqual(len(group['n']), 2)
            if len(seen) == 2:
                raise RuntimeError('interleaved interruption')
            return forward(group)

        with mock.patch.object(self.donor, 'forward', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'interleaved interruption'):
                self.prepare()
        torch.testing.assert_close(seen[0], batch['node'][[1, 3], :3], rtol=0, atol=0)
        target = self.feature_root / 'train' / 'CVRP'
        partial = json.loads((target / 'metadata.json').read_text())
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['rows_written'], 2)
        self.assertEqual(partial['extract_order'], module.EXTRACT_ORDER)
        with mock.patch.object(self.donor, 'forward', wraps=forward) as resumed:
            self.prepare()
        torch.testing.assert_close(resumed.call_args_list[0].args[0]['node'], batch['node'][[5], :3], rtol=0, atol=0)
        torch.testing.assert_close(resumed.call_args_list[1].args[0]['node'], batch['node'][[4], :4], rtol=0, atol=0)
        torch.testing.assert_close(resumed.call_args_list[2].args[0]['node'], batch['node'][[0, 2]], rtol=0, atol=0)
        wrapper = self.wrapper('CVRP')
        actual = wrapper.batch([5, 0, 3, 4, 1, 2])
        torch.testing.assert_close(actual['nco_nodes'], forward(actual).half(), rtol=0, atol=0)

    def test_atomic_final_rename_recovery(self):
        self.prepare()
        path = self.feature_root / 'train' / 'CVRP' / 'metadata.json'
        record = json.loads(path.read_text())
        record['complete'] = False
        _write_json(path, record)
        with mock.patch.object(self.donor, 'forward', side_effect=AssertionError('already renamed')):
            self.prepare()
        self.assertTrue(json.loads(path.read_text())['complete'])

    def test_stale_source_and_donor_variant_rejected_without_overwriting(self):
        self.prepare()
        path = self.feature_root / 'train' / 'CVRP' / 'metadata.json'
        before = path.read_bytes()
        other = StaticReLDDonor('random')
        other._provenance['variant'] = 'pretrained'
        other._provenance['checkpoint_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.prepare(donor=other)
        self.assertEqual(path.read_bytes(), before)
        source = self.cache_root / 'train' / 'CVRP' / 'metadata.json'
        metadata = json.loads(source.read_text())
        metadata['scenario_id'] = 'changed'
        _write_json(source, metadata)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.prepare(manifest=None)
        self.assertEqual(path.read_bytes(), before)

    def test_wrapper_binds_expected_variant_and_full_model_provenance(self):
        self.prepare()
        base = BaseCache(self.cache_root / 'train' / 'CVRP')
        target = self.feature_root / 'train' / 'CVRP'
        try:
            with self.assertRaisesRegex(ValueError, 'requested arm'):
                FrozenFeatureCache(base, target, expected_variant='pretrained')
            changed = self.donor.provenance()
            changed['native_source_sha256'] = 'f' * 64
            with self.assertRaisesRegex(ValueError, 'attached model donor'):
                FrozenFeatureCache(base, target, expected_provenance=changed)
            wrapper = FrozenFeatureCache(base, target, expected_variant='random',
                                         expected_provenance=self.donor.provenance())
            wrapper.close()
        finally:
            base.close()

    def test_changed_input_array_identity_rejected(self):
        self.prepare()
        array_path = self.cache_root / 'train' / 'CVRP' / 'node.npy'
        nodes = np.load(array_path)
        nodes[0, 1, 0] += .125
        np.save(array_path, nodes)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.prepare()

    def test_test_lock_gate_before_any_source_or_donor_access(self):
        for lock in (None, {}, dict(experiment='R93', selected_without_test=True),
                     dict(experiment='R94', selected_without_test=False)):
            with mock.patch.object(module, '_source_record', side_effect=AssertionError('source before lock')):
                with self.assertRaises(PermissionError):
                    self.prepare(split='test', test_lock=lock)
        lock = dict(experiment='R94', arm='A', selected_update=1440,
                    selected_without_test=True, per_problem_assembly=False,
                    checkpoint='/locked/selector.pt', checkpoint_sha256='c' * 64,
                    inference_rule=dict(rule='argmax', temperature=1.))
        test_manifest = _sources(self.cache_root, split='test')
        result = self.prepare(split='test', manifest=test_manifest, test_lock=lock)
        self.assertTrue(result['complete'])
        self.assertEqual(result['splits']['test']['CVRP']['test_lock']['record'], lock)
        lock['inference_rule']['temperature'] = 2.
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.prepare(split='test', manifest=test_manifest, test_lock=lock)

    def test_test_wrapper_requires_intact_lock(self):
        manifest = _sources(self.cache_root, split='test')
        lock = dict(experiment='R94', arm='A', selected_update=1440,
                    selected_without_test=True, per_problem_assembly=False,
                    checkpoint='/locked/model.pt', checkpoint_sha256='c' * 64, inference_rule='argmax')
        self.prepare(split='test', manifest=manifest, test_lock=lock)
        base = BaseCache(self.cache_root / 'test' / 'CVRP')
        target = self.feature_root / 'test' / 'CVRP'
        wrapper = FrozenFeatureCache(base, target)
        wrapper.close()
        path = target / 'metadata.json'
        record = json.loads(path.read_text())
        record['test_lock']['record']['selected_without_test'] = False
        _write_json(path, record)
        base = BaseCache(self.cache_root / 'test' / 'CVRP')
        try:
            with self.assertRaises(PermissionError):
                FrozenFeatureCache(base, target)
        finally:
            base.close()

    def test_estimate_is_metadata_only_even_without_arrays(self):
        metadata_root = self.root / 'metadata_only'
        _sources(metadata_root, 'train', arrays=False)
        _sources(metadata_root, 'val', arrays=False)
        with mock.patch.object(np, 'load', side_effect=AssertionError('estimate opened an array')), mock.patch.object(module, 'StaticReLDDonor', side_effect=AssertionError('estimate constructed donor')):
            estimate = estimate_features(metadata_root)
            self.assertEqual(estimate['payload_bytes'], 16 * 2 * 2 * 5 * 128 * 2)
            output = io.StringIO()
            with redirect_stdout(output):
                module.main(['estimate', '--cache-root', str(metadata_root)])
            self.assertEqual(json.loads(output.getvalue())['payload_bytes'], estimate['payload_bytes'])

    def test_missing_supported_cache_never_falls_back_to_extraction(self):
        with self.assertRaises(FileNotFoundError):
            self.wrapper()

    def test_raw_tsp_atsp_pass_through_without_sidecar(self):
        for problem in ('TSP', 'ATSP'):
            base = mock.Mock(problem=problem, size=2)
            raw = dict(raw_only=object())
            base.batch.return_value = raw
            wrapper = FrozenFeatureCache(base, self.root / 'missing')
            self.assertIs(wrapper.batch([1]), raw)
            self.assertIsNone(wrapper._features)
            wrapper.close()
            base.close.assert_called_once()

    def test_invalid_indices_and_data_or_manifest_mismatch(self):
        self.prepare()
        wrapper = self.wrapper()
        for rows in ([], [-1], [2], ['01'], np.array([[0]]), torch.tensor([[0]])):
            with self.assertRaises((ValueError, IndexError, KeyError)):
                wrapper.batch(rows)
        with self.assertRaisesRegex(ValueError, 'data_root'):
            self.prepare(data_root=self.root / 'wrong')
        changed = copy.deepcopy(self.manifest)
        changed['splits']['train']['CVRP']['rows'] = 999
        with self.assertRaisesRegex(ValueError, 'per-task manifest'):
            self.prepare(manifest=changed)
        for size in (0, -1, True):
            with self.assertRaises(ValueError):
                self.prepare(batch_size=size)

    def test_nonfinite_features_are_not_marked_complete(self):
        def bad(batch):
            values = torch.full((len(batch['n']), batch['node'].shape[1], 128), float('inf'))
            return values

        with mock.patch.object(self.donor, 'forward', side_effect=bad):
            with self.assertRaisesRegex(ValueError, 'finite'):
                self.prepare()
        record = json.loads((self.feature_root / 'train' / 'CVRP' / 'metadata.json').read_text())
        self.assertFalse(record['complete'])
        self.assertEqual(record['rows_written'], 0)


if __name__ == '__main__':
    unittest.main()
