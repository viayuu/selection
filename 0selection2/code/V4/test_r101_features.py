"""Tiny CPU fixtures only. No supervision files, donors, or GPUs are used."""

import hashlib
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

from . import r101_features as module
from .r94_donor import OUTPUT_DIM, PROBLEM_NAMES, SUPPORTED_PROBLEMS
from .r94_features import EXTRACT_ORDER, FEATURE_FORMAT, _digest


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class HintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw, self.teacher, self.original_r94 = self.root / 'r84', self.root / 'r94', self.root / 'receipts'
        self.donors = {}
        for arm, variant in (('A', 'random'), ('B', 'pretrained')):
            donor = dict(variant=variant, output_dim=OUTPUT_DIM,
                         supported_problems=list(SUPPORTED_PROBLEMS), checkpoint_sha256=None if arm == 'A' else 'a' * 64,
                         known_unknown_pretraining_source=dict(unknown=['training instance manifest']))
            self.donors[arm] = donor
            _write(self.original_r94 / arm / 'input_features_ready.json', dict(
                experiment='R94', arm=arm, complete=True, splits=['train', 'val'],
                labels_used=False, decoder_executed=False, frozen_donor=donor))
        for problem in SUPPORTED_PROBLEMS:
            directory = self.raw / 'train' / problem
            directory.mkdir(parents=True)
            width = 4 if problem == 'CVRP' else 8
            arrays = dict(n=dict(file='n.npy', shape=[3], dtype='int64'),
                          node=dict(file='node.npy', shape=[3, 5, width], dtype='float32'),
                          node_geom=dict(file='node_geom.npy', shape=[3, 5, 12], dtype='float32'),
                          coord_dist=dict(file='coord_dist.npy', shape=[3], dtype='int64'),
                          costs=dict(file='costs.npy', shape=[3, 2], dtype='float64'),
                          winner=dict(file='winner.npy', shape=[3], dtype='int64'))
            metadata = dict(cache_format='r76_scenario_v1', cache_version=1, complete=True,
                            rows_written=3, rows=3, problem=problem, problem_id=PROBLEM_NAMES.index(problem),
                            split='train', kind='coord', d_in=width, true_length_min=3, true_length_max=5,
                            scenario_id='synthetic_original_full_train', data_root='/do_not_open',
                            pool=['a', 'reld_mt'], pool_ids=[0, 2], mask=[1, 0, 1],
                            cbits=[1, 0, 1, 0, 0], problem_desc=[.25, .5], arrays=arrays,
                            source_files=[dict(path='/forbidden/raw_label.pkl')])
            _write(directory / 'metadata.json', metadata)
            np.save(directory / 'n.npy', np.array([5, 3, 4], dtype=np.int64))
            np.save(directory / 'coord_dist.npy', np.array([2, 0, 1], dtype=np.int64))
            for key in ('node', 'node_geom'):
                shape = arrays[key]['shape']
                np.save(directory / arrays[key]['file'], np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
            source = dict(directory=str(directory), metadata_sha256=_digest(metadata), inputs={})
            for key in ('n', 'node'):
                path = directory / arrays[key]['file']
                source['inputs'][key] = dict(path=str(path), bytes=path.stat().st_size,
                                             mtime_ns=path.stat().st_mtime_ns, spec=arrays[key])
            target = self.teacher / 'train' / problem
            _write(target / 'metadata.json', dict(
                format=FEATURE_FORMAT, problem=problem, split='train', rows=3, shape=[3, 5, 128],
                dtype='float16', source_cache=source, donor=self.donors['A'], input_only=True,
                test_lock=None, extract_order=EXTRACT_ORDER, complete=True, rows_written=3, file='nco_nodes.npy'))
            nodes = np.arange(3 * 5 * 128, dtype=np.float16).reshape(3, 5, 128)
            nodes[1, 3:] = np.nan
            np.save(target / 'nco_nodes.npy', nodes)
        self.local = self.root / 'local_r84'
        shutil.copytree(self.raw, self.local)
        for path in self.local.glob('train/*/*.npy'):
            os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1000000))
        for name, value in (('R84_CACHE_ROOT', self.raw), ('R94_ROOT', self.original_r94), ('TRAIN_ROWS', 3)):
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def cache(self, **kwargs):
        options = dict(cache_root=self.local, teacher_root=self.teacher, problem='CVRP', variant='A')
        options.update(kwargs)
        result = module.HintInputCache(**options)
        self.addCleanup(result.close)
        return result

    def test_no_supervision_files_exist_or_are_opened(self):
        self.assertFalse(list(self.root.rglob('costs.npy')))
        self.assertFalse(list(self.root.rglob('winner.npy')))
        load = np.load
        opened = []

        def guarded(path, *args, **kwargs):
            self.assertIn(Path(path).name, ('n.npy', 'node.npy', 'node_geom.npy', 'coord_dist.npy', 'nco_nodes.npy'))
            self.assertEqual(kwargs['mmap_mode'], 'r')
            opened.append(str(path))
            return load(path, *args, **kwargs)

        with mock.patch.object(np, 'load', side_effect=guarded):
            batch = self.cache().batch([2, 0, 2])
        self.assertTrue(opened)
        self.assertEqual(set(batch), {'problem_id', 'problem_name', 'kind', 'd_in', 'n', 'node_mask',
                                     'cbits', 'problem_desc', 'coord_dist', 'pool_ids', 'mask', 'solver_mask',
                                     'node', 'node_geom', 'teacher_nodes', 'teacher_node_mask'})
        for key in ('costs', 'winner', 'ind', 'comp_target', 'targets'):
            self.assertNotIn(key, batch)

    def test_strict_mask_row_order_duplicates_and_conditions(self):
        cache = self.cache()
        batch = cache.batch(['1', '0', '1'])
        self.assertIs(batch['teacher_node_mask'], batch['node_mask'])
        self.assertEqual(batch['teacher_nodes'].shape, (3, 5, 128))
        self.assertEqual(batch['teacher_nodes'].dtype, torch.float16)
        torch.testing.assert_close(batch['n'], torch.tensor([3, 5, 3]))
        torch.testing.assert_close(batch['coord_dist'], torch.tensor([0, 2, 0]))
        self.assertTrue(torch.isfinite(batch['teacher_nodes']).all())
        self.assertEqual(torch.count_nonzero(batch['teacher_nodes'][~batch['node_mask']]), 0)
        self.assertEqual(batch['teacher_nodes'][0, 0, 0].item(), 640)
        self.assertEqual(batch['teacher_nodes'][1, 0, 0].item(), 0)
        torch.testing.assert_close(batch['teacher_nodes'][0], batch['teacher_nodes'][2])
        torch.testing.assert_close(batch['cbits'][0], torch.tensor(cache.metadata['cbits'], dtype=torch.float32))
        self.assertEqual(cache.batch([1])['teacher_nodes'].shape, (1, 3, 128))
        self.assertTrue(cache.identity['transport_observations']['node']['mtime_differs'])

    def test_precise_prefetch_spans_and_close(self):
        cache = self.cache()
        arrays, fds = dict(cache._arrays), dict(cache._fds)
        with mock.patch.object(module.os, 'posix_fadvise', wraps=os.posix_fadvise) as advice:
            cache.batch([2, 1, 2])
        calls = [call.args for call in advice.call_args_list]
        for key, array in arrays.items():
            wanted = array.dtype.itemsize if array.ndim == 1 else 4 * array.strides[1]
            expected = [(fds[key], int(array.offset) + row * int(array.strides[0]), int(wanted),
                         os.POSIX_FADV_WILLNEED) for row in (1, 2)]
            self.assertEqual([call for call in calls if call[0] == fds[key]], expected)
            self.assertFalse(array.flags.writeable)
        cache.close()
        cache.close()
        for fd in fds.values():
            with self.assertRaises(OSError):
                os.fstat(fd)
        self.assertTrue(all(array._mmap.closed for array in arrays.values()))
        with self.assertRaises(RuntimeError):
            cache.batch([0])

    def test_wrong_variant_or_original_teacher_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError, 'variant/provenance'):
            self.cache(variant='B')
        path = self.teacher / 'train/CVRP/metadata.json'
        metadata = json.loads(path.read_text())
        metadata['donor']['checkpoint_sha256'] = 'forged'
        _write(path, metadata)
        with self.assertRaisesRegex(ValueError, 'variant/provenance'):
            self.cache()

    def test_metadata_hash_and_pool_change_rejected(self):
        path = self.local / 'train/CVRP/metadata.json'
        record = json.loads(path.read_text())
        record['pool'] = ['other', 'reld_mt']
        _write(path, record)
        with self.assertRaisesRegex(ValueError, 'metadata hash'):
            self.cache()

    def test_actual_n_row_permutation_rejected(self):
        np.save(self.local / 'train/CVRP/n.npy', np.array([3, 5, 4], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, 'row alignment'):
            self.cache()

    def test_incomplete_or_nontrain_teacher_rejected(self):
        path = self.teacher / 'train/CVRP/metadata.json'
        original = json.loads(path.read_text())
        for patch in (dict(split='val'), dict(complete=False), dict(rows_written=2),
                      dict(extract_order='sorted_not_original'), dict(input_only=False)):
            with self.subTest(patch=patch):
                _write(path, dict(original, **patch))
                with self.assertRaises(ValueError):
                    self.cache()
        with self.assertRaisesRegex(ValueError, 'TRAIN only'):
            self.cache(teacher_root=self.teacher / 'test')
        with self.assertRaises(ValueError):
            self.cache(problem='TSP')

    def test_inspection_reads_no_teacher_payload_and_reports_exact_size(self):
        with mock.patch.object(module, '_hash_file', side_effect=AssertionError('large hash')):
            plan = module.inspect_teacher(self.teacher, 'A', cache_root=self.local)
        expected = sum(path.stat().st_size for path in self.teacher.glob('train/*/*'))
        self.assertEqual(plan['bytes'], expected)
        self.assertEqual(len(plan['tasks']), 16)
        self.assertFalse(plan['teacher_payload_read'])
        self.assertFalse(plan['labels_read'])

    def test_metadata_discovery_does_not_open_stat_or_hash_arrays(self):
        with (mock.patch.object(np, 'load', side_effect=AssertionError('array open')),
              mock.patch.object(module, '_stat', side_effect=AssertionError('array stat')),
              mock.patch.object(module, '_hash_file', side_effect=AssertionError('array hash'))):
            receipt = module.inspect_teacher_metadata(self.teacher, 'A')
        self.assertEqual(receipt['format'], 'r101_teacher_metadata_only_v1')
        self.assertFalse(receipt['arrays_opened'])
        self.assertFalse(receipt['array_hashes_computed'])
        self.assertEqual(len(receipt['tasks']), 16)
        self.assertIn('teacher_metadata_sha256', receipt['tasks']['CVRP'])
        with self.assertRaisesRegex(ValueError, 'source receipt'):
            self.cache(source_receipt=receipt)

    def test_stream_stage_hashes_only_selected_files_and_preserves_source(self):
        (self.teacher / 'train/CVRP/do_not_copy.txt').write_text('not part of teacher')
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.teacher.glob('train/*/*')}
        target = self.root / 'staged'
        receipt_path = self.root / 'stage_receipt.json'
        with redirect_stdout(io.StringIO()), mock.patch.object(
                module.os, 'posix_fadvise', wraps=os.posix_fadvise) as advice:
            receipt = module.stage_teacher(self.teacher, target, 'A', cache_root=self.local,
                                           receipt_path=receipt_path, buffer_bytes=1024)
        self.assertEqual(receipt['root'], str(target / 'train'))
        self.assertTrue(receipt['payload_sha256_verified'])
        self.assertFalse(receipt['automatic_retries'])
        self.assertEqual(len(list(target.glob('train/*/*'))), 32)
        self.assertFalse((target / 'train/CVRP/do_not_copy.txt').exists())
        self.assertFalse((target / 'val').exists())
        for problem in SUPPORTED_PROBLEMS:
            for filename, entry in receipt['tasks'][problem]['files'].items():
                expected = hashlib.sha256((target / 'train' / problem / filename).read_bytes()).hexdigest()
                self.assertEqual(entry['source_sha256'], expected)
                self.assertEqual(entry['target_sha256'], expected)
                self.assertEqual(entry['source_passes'], 1)
                self.assertEqual(entry['target_verification_passes'], 1)
        self.assertTrue(any(call.args[-1] == os.POSIX_FADV_SEQUENTIAL for call in advice.call_args_list))
        self.assertTrue(any(call.args[-1] == os.POSIX_FADV_DONTNEED for call in advice.call_args_list))
        for path, (content, mtime) in before.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
        cache = self.cache(teacher_root=target, source_receipt=receipt_path)
        self.assertEqual(cache.batch([1])['teacher_nodes'].shape, (1, 3, 128))
        with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
            module.stage_teacher(self.teacher, target, 'A', cache_root=self.local)

    def test_source_receipt_metadata_hash_tampering_rejected(self):
        receipt = module.stage_teacher(self.teacher, variant='A', cache_root=self.local)
        receipt['tasks']['CVRP']['teacher_metadata_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'receipt metadata hash'):
            self.cache(source_receipt=receipt)

    def test_reuse_does_not_copy_or_hash_and_discloses_unknown_provenance(self):
        for path in self.teacher.glob('train/*/metadata.json'):
            metadata = json.loads(path.read_text())
            metadata['donor'] = self.donors['B']
            _write(path, metadata)
        with (mock.patch.object(module, '_copy_file', side_effect=AssertionError('copy')),
              mock.patch.object(module, '_hash_file', side_effect=AssertionError('hash'))):
            receipt = module.stage_teacher(self.teacher, variant='B', cache_root=self.local)
        self.assertFalse(receipt['copied'])
        self.assertFalse(receipt['payload_sha256_verified'])
        self.assertEqual(receipt['teacher_provenance']['known_unknown_pretraining_source']['unknown'],
                         ['training instance manifest'])
        self.cache(variant='B', source_receipt=receipt).batch([0])

    def test_buffer_limit_and_bad_header_and_rows_rejected(self):
        with self.assertRaises(ValueError):
            module.stage_teacher(self.teacher, self.root / 'never_created', 'A',
                                 cache_root=self.local, buffer_bytes=module.MAX_BUFFER_BYTES + 1)
        self.assertFalse((self.root / 'never_created').exists())
        for rows in ([], [-1], [3], ['01']):
            with self.subTest(rows=rows), self.assertRaises((ValueError, IndexError, KeyError)):
                self.cache().batch(rows)
        np.save(self.teacher / 'train/CVRP/nco_nodes.npy', np.zeros((3, 5, 128), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'shape, dtype'):
            self.cache()


if __name__ == '__main__':
    unittest.main()
