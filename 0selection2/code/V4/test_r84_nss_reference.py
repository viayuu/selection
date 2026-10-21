"""CPU fixtures only; never open actual R84 TEST or launch NSS inference."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from code.unified_selector.registry import S2I
from . import r84_nss_reference as reference
from .r80_reporting import check_saved_coverage, file_hash, json_hash, read_json, write_json


class R84ReferenceTests(unittest.TestCase):
    def locked_fixture(self, root):
        run, data = root / 'model', root / 'data'
        run.mkdir()
        checkpoint = run / 'best.pt'
        checkpoint.write_bytes(b'fixture weights, never loaded as a model')
        config = dict(experiment='R84', arm='attention_seed2', data_root=str(data),
            max_successful_updates=100, checkpoint_rule='fixed validation rule', inference_rule='direct attention')
        lock = dict(experiment='R84', arm=config['arm'], epoch=2, selected_update=40,
            selected_without_test=True, per_problem_assembly=False,
            state_hash=json_hash(config), rule=config['checkpoint_rule'], inference_rule=config['inference_rule'],
            checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint))
        write_json(run / 'protocol.json', config)
        write_json(run / 'final_test_lock.json', lock)
        return run, data, config, lock

    def test_missing_val_lock_stops_before_import_or_any_test_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(model_run=root / 'absent', data_root=root / 'new_test', root=root / 'output')
            with patch.object(reference.importlib, 'import_module') as load_nss:
                with self.assertRaises(FileNotFoundError):
                    reference.run_reference(args)
                load_nss.assert_not_called()
            self.assertFalse(args.root.exists())

    def test_gate_checks_model_hash_data_identity_and_unique_validation_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, data, config, lock = self.locked_fixture(Path(temporary))
            self.assertEqual(reference.require_r84_val_lock(run, data), (config, lock))
            with self.assertRaises(PermissionError):
                reference.require_r84_val_lock(run, data / 'different_release')
            for changes in ({'selected_without_test': False}, {'per_problem_assembly': True},
                            {'checkpoint_sha256': 'wrong'}, {'selected_update': 0}, {'state_hash': 'changed'}):
                write_json(run / 'final_test_lock.json', dict(lock, **changes))
                with self.subTest(changes=changes), self.assertRaises(PermissionError):
                    reference.require_r84_val_lock(run, data)

    def test_native_fp64_near_tie_not_replaced_by_fp32_argmin(self):
        costs = np.array([[1. + 5e-10, 1.], [2., 3.]], dtype=np.float64)
        metrics = reference.native_metrics(np.array([[2., 0.], [2., 0.]]), np.array([0, 0]),
                                           costs, np.array([1, 0]), ['BQ', 'LEHD'], 'BQ')
        self.assertEqual(metrics['top1_native_ind'], .5)
        self.assertEqual(metrics['top1_fp32_argmin'], 1.)
        self.assertEqual(metrics['fp32_native_winner_disagreements'], 1)
        self.assertGreater(metrics['actual_regret_pct'], 0.)
        self.assertEqual(metrics['sbs_selection_split'], 'R84 TRAIN')
        with self.assertRaises(ValueError):
            reference.native_metrics(np.ones((2, 2)), np.array([0, 0]), costs.astype(np.float32),
                                     np.array([0, 0]), ['BQ', 'LEHD'], 'BQ')

    def test_report_coverage_accepts_alias_by_stable_id_and_manifest_population(self):
        ids = [S2I['MTPOMO'], S2I['RELD_MOEL']]
        data = SimpleNamespace(n=4500, pool=['MTPOMO', 'reld_mt'], pool_ids=np.array(ids))
        record = dict(rows=4500, pool=data.pool, pool_ids=ids)
        config = dict(data_manifest={'val': {'OVRP': record}})
        check_saved_coverage(data, dict(n=4500), config, 'VAL', 'OVRP')
        current = dict(rows_per_problem=dict(val=4500, test=4500),
                       candidate_pools={'OVRP': data.pool}, candidate_pool_ids={'OVRP': ids})
        check_saved_coverage(data, dict(n=4500), current, 'TEST', 'OVRP')
        with self.assertRaises(ValueError):
            check_saved_coverage(data, dict(n=4500), dict(current, heldout_rows_per_problem=3895,
                rows_per_problem={}), 'TEST', 'OVRP')
        with self.assertRaises(ValueError):
            check_saved_coverage(data, dict(n=3895), config, 'VAL', 'OVRP')
        with self.assertRaises(ValueError):
            check_saved_coverage(data, dict(n=4500),
                dict(data_manifest={'val': {'OVRP': dict(record, rows=3895)}}), 'VAL', 'OVRP')
        wrong = SimpleNamespace(n=4500, pool=['MTPOMO', 'RELD_MTL'], pool_ids=np.array(ids))
        with self.assertRaises(ValueError):
            check_saved_coverage(wrong, dict(n=4500), config, 'VAL', 'OVRP')

    def test_new_test_cache_uses_native_inputs_labels_and_source_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'TSPtest'
            directory.mkdir()
            write_json(directory / 'metadata.json', dict(instances=2, methods=['BQ', 'LEHD']))
            np.savez(directory / 'input_metadata.npz', source_index=np.array([10000, 99999]))
            instances = [torch.zeros(1, 3, 2), torch.ones(1, 5, 2)]
            labels = {'0': dict(cost=[1. + 5e-10, 1.], ind=1), '1': dict(cost=[2., 3.], ind=0)}
            nss = SimpleNamespace(load_pickle=Mock(side_effect=[instances, labels]))
            cache = reference.load_test_cache(Path(temporary), 'TSP', ['BQ', 'LEHD'], nss, expected_rows=2)
            torch.testing.assert_close(cache['source_indices'], torch.tensor([10000, 99999]))
            torch.testing.assert_close(cache['lengths'], torch.tensor([3, 5]))
            torch.testing.assert_close(cache['offsets'], torch.tensor([0, 3, 8]))
            self.assertEqual(cache['costs'].dtype, torch.float64)
            self.assertEqual(cache['winner'].tolist(), [1, 0])
            self.assertEqual(tuple(cache['points'].shape), (8, 2))
            self.assertEqual(nss.load_pickle.call_count, 2)

    def test_lib_reuses_exact_saved_predictions_and_rejects_unbound_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'old' / 'TSP'
            original.mkdir(parents=True)
            source = original / 'TSPLIB_predictions.npz'
            n = 49
            np.savez(source, scores=np.tile([2., 0.], (n, 1)), pred=np.zeros(n, dtype=np.int64),
                costs=np.tile([1., 2.], (n, 1)).astype(np.float64), winner=np.zeros(n, dtype=np.int64),
                pool=np.array(['BQ', 'LEHD']), indices=np.arange(n), source_indices=np.arange(n))
            frozen = dict(sha256='fixed historical checkpoint')
            write_json(original / 'TSPLIB_metrics.json', dict(checkpoint=frozen))
            write_json(original / 'TSPLIB_once.json', dict(state='complete', checkpoint_sha256=frozen['sha256'],
                                                         predictions_sha256=file_hash(source)))
            result = reference.reuse_lib(root / 'new', root / 'old', 'TSP', frozen, 'BQ')
            self.assertTrue(result['inference_reused'])
            self.assertFalse(result['new_inference'])
            self.assertEqual(result['n'], 49)
            self.assertEqual(result['top1_native_ind'], 1.)
            self.assertEqual(file_hash(source), file_hash(root / 'new/TSP/TSPLIB_predictions.npz'))
            write_json(original / 'TSPLIB_once.json', dict(state='complete', checkpoint_sha256='wrong',
                                                         predictions_sha256=file_hash(source)))
            with self.assertRaises(ValueError):
                reference.reuse_lib(root / 'new', root / 'old', 'TSP', frozen, 'BQ')

    def test_protocol_reuse_is_idempotent_but_does_not_silently_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'protocol.json'
            reference.check_or_write(path, dict(data='new TEST', trained=False))
            reference.check_or_write(path, dict(data='new TEST', trained=False))
            with self.assertRaises(ValueError):
                reference.check_or_write(path, dict(data='old TEST', trained=False))
            self.assertFalse(read_json(path)['trained'])

    def test_pipeline_cli_has_explicit_lock_and_new_dataset_defaults(self):
        args = reference.parser().parse_args(['--model-run', '/fixture/R84/attention_seed2', '--device', 'cuda:0'])
        self.assertEqual(args.data_root, reference.DEFAULT_DATA)
        self.assertEqual(args.root, reference.DEFAULT_ROOT)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.device, 'cuda:0')
        self.assertEqual(reference.TEST_ROWS, 4500)

    def test_multiple_locked_arms_share_reference_without_changing_its_first_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = dict(data_root='same new release', checkpoints={'TSP': 'fixed'},
                current_model_lock=dict(path='/A/lock', checkpoint_sha256='A', selected_update=20))
            second = dict(first, current_model_lock=dict(path='/B/lock', checkpoint_sha256='B', selected_update=40))
            self.assertEqual(reference.bind_reference_protocol(root, first), first)
            self.assertEqual(reference.bind_reference_protocol(root, second), first)
            self.assertEqual(read_json(root / 'protocol.json'), first)
            self.assertTrue((root / 'val_lock_consumers/A.json').is_file())
            self.assertTrue((root / 'val_lock_consumers/B.json').is_file())
            with self.assertRaises(ValueError):
                reference.bind_reference_protocol(root, dict(second, data_root='different TEST'))


if __name__ == '__main__':
    unittest.main()
