"""Small no-GPU witnesses for metric semantics and the held-out access gate."""

from pathlib import Path
from enum import Enum
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .r97_experiment import (ensure_fresh_run, install_gpu_only_offload_guard,
                             parser, query_fingerprint, query_inputs, saved_prediction,
                             verify_query_fingerprint)


class ExperimentTests(unittest.TestCase):
    def test_fp64_winner_and_full_pool_regret(self):
        raw = dict(costs=np.asarray([[1., 1.+1e-10], [3., 2.]], dtype=np.float64),
                   winner=np.asarray([0, 1]), pool=['A', 'B'], pool_ids=[0, 7])
        config = dict(baselines={'TSP': dict(selection_split='train', sbs_solver='A',
            sbs_slot=0, mean_cost_by_solver={'A': 2., 'B': 2.})},
            candidate_pool_ids={'TSP': [0, 7]})
        metrics, data = saved_prediction('TSP', np.asarray([[.4, .6], [.6, .4]]),
            raw, np.asarray([900, 901]), np.asarray([50, 60]), config)
        self.assertEqual(metrics['top1'], 0.)
        self.assertEqual(metrics['native_correct'], 0)
        self.assertAlmostEqual(metrics['actual_regret_pct'], 25.+5e-9)
        np.testing.assert_array_equal(data['winner'], [0, 1])
        self.assertEqual(data['costs'].dtype, np.float64)

    def test_test_lock_precedes_query_feature_access(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(root=Path(directory),
                feature_root=Path(directory)/'nonexistent', data_root=Path(directory)/'nonexistent')
            with self.assertRaises(FileNotFoundError) as error:
                query_inputs('TSP', 'test', args, {}, None)
            self.assertIn('final_test_lock.json', str(error.exception))

    def test_parser_frozen_run_and_cpu_report(self):
        self.assertEqual(parser().parse_args(['run', '--device', 'cuda:0']).device, 'cuda:0')
        self.assertEqual(parser().parse_args(['report']).device, 'cpu')
        with self.assertRaises(SystemExit):
            parser().parse_args(['run', '--resume'])

    def test_cuda_allocation_failure_cannot_fall_back_to_cpu(self):
        class Modes(Enum):
            GPU = 1
            CPU = 2

        class Manager:
            exe_device = SimpleNamespace(type='cuda')

            def _resolve_offload_mode(self):
                return Modes.GPU, 'requested'

        calls = []

        def failed_cuda(shape, **kwargs):
            calls.append(kwargs['device'].type)
            raise RuntimeError('synthetic CUDA OOM')

        module = SimpleNamespace(InferenceManager=Manager, OffloadMode=Modes,
                                 torch=SimpleNamespace(empty=failed_cuda))
        with patch('code.V4.r97_experiment.importlib.import_module', return_value=module):
            install_gpu_only_offload_guard({'library_root': '/tmp'})
        manager = Manager()
        with self.assertRaisesRegex(RuntimeError, 'synthetic CUDA OOM'):
            manager._allocate_output_buffer(Modes.GPU, (2, 3), 'float32')
        with self.assertRaisesRegex(RuntimeError, 'assigned GPU'):
            manager._allocate_output_buffer(Modes.CPU, (2, 3), 'float32')
        self.assertEqual(calls, ['cuda'])

    def test_same_shape_source_reordering_is_rejected(self):
        import hashlib
        original = np.asarray([900, 901], dtype=np.int64)
        expected = dict(feature_sha256='same', receipt_sha256='same', input_metadata_sha256='same',
            ordered_source_ids_sha256=hashlib.sha256(original.tobytes()).hexdigest())
        changed = dict(expected, ordered_source_ids_sha256=hashlib.sha256(original[::-1].tobytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'source-index order'):
            verify_query_fingerprint(changed, expected)

    def test_actual_output_root_must_be_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'context').mkdir()
            ensure_fresh_run(root)
            (root/'checkpoint.json').touch()
            with self.assertRaisesRegex(RuntimeError, 'checkpoint.json'):
                ensure_fresh_run(root)

    def test_same_columns_incompatible_feature_definition_is_rejected(self):
        from .r92_tree_features import FEATURE_DEFINITION
        from .r80_reporting import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(feature_root=root/'features', data_root=root/'data')
            definition = dict(FEATURE_DEFINITION, version='incompatible_semantics')
            record = dict(complete=True, dtype='float32', shape=[85500, 357],
                identity=dict(splits=['train', 'val'], rows=dict(train=81000, val=4500),
                    problem='TSP', data_root=str(args.data_root.resolve()), feature_definition=definition))
            write_json(args.feature_root/'TRAIN_VAL/TSP.json', record)
            with self.assertRaisesRegex(ValueError, 'split/schema/source'):
                query_fingerprint('TSP', 'val', args)


if __name__ == '__main__':
    unittest.main()
