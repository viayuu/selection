"""Four bounded CPU sidecar contracts; no native weights, GPU or publication."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from . import r117_experiment as runner
from . import r117_finish as finish
from . import r117_model as models
from .r94_donor import CHECKPOINT_SHA256


class FakeNative(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.register_buffer('marker', torch.tensor([1. if arm == 'A' else 2.]))
        self.record = dict(arm=arm, variant=runner.ARM_VARIANTS[arm],
            components=dict(base=dict(variant='pretrained', checkpoint_sha256=CHECKPOINT_SHA256),
                **{p: dict(variant='pretrained' if arm == 'B' else 'random') for p in ('TSP', 'CVRP', 'VRP', 'ATSP')}),
            external_pretraining_by_component=dict(base=True, TSP=arm == 'B', CVRP=arm == 'B',
                                                   VRP=arm == 'B', ATSP=arm == 'B'))

    def provenance(self):
        return copy.deepcopy(self.record)

    def forward(self, batch):
        raise AssertionError('CPU runner contracts must never execute a native donor')


def fake_factory(params, arm, state_dict=None, *, donor=None):
    return models.make_model(params, arm, state_dict=state_dict, donor=donor or FakeNative(arm))


def fixture_lock(root, arm='A'):
    args = runner.parser().parse_args(['describe', '--root', str(root), '--arm', arm])
    caps = {p: 256 if p in ('TSP', 'CVRP', 'ATSP') else 512 for p in runner.PROBLEMS}
    witness = root / 'capacity.json'
    runner.write_json(witness, dict(complete=True, microbatch_caps=caps, eval_batch_size=256,
                                    native_extraction_batch_size=128))
    record = dict(runner.required_lock_fields(), data_root=str(args.data_root),
        prepared_root=str(args.prepared_root), primary_cache=str(args.cache_root),
        microbatch_caps=caps, eval_batch_size=256, lib_batch_size=128,
        capacity_witness=str(witness), capacity_witness_sha256=runner.file_hash(witness),
        feature_roots={a: '/tmp/shiys_R117_' + a + '_native_features' for a in runner.ARMS},
        common_feature_root=str(args.common_root))
    runner.write_json(root / 'protocol_lock.json', record)
    (root / 'PLAN.md').write_text('CPU fixture only; no training or inference.\n', encoding='ascii')
    return args, record


class RunnerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initialization_protocol_and_embedded_restore(self):
        initial = torch.load(runner.DEFAULT_PREPARED_ROOT / 'identicalinit_seed2.pt',
                             map_location='cpu', weights_only=False)
        geometry = runner.read_json(runner.DEFAULT_PREPARED_ROOT / 'geometry_stats.json')
        manifest = runner.read_json(runner.DEFAULT_PREPARED_ROOT / 'source_cache_manifest.json')
        baselines = runner.read_json(runner.DEFAULT_PREPARED_ROOT / 'common_full_train_sbs.json')
        receipts = []
        with tempfile.TemporaryDirectory(prefix='r117_cpu_') as directory, patch.object(runner, 'make_model', fake_factory):
            root = Path(directory)
            for arm in runner.ARMS:
                args, record = fixture_lock(root, arm)
                run = root / arm
                run.mkdir()
                model, params, receipt = runner.initialize_shared(initial, geometry, arm)
                self.assertEqual(receipt['trainable_parameter_count'], 1941257)
                self.assertEqual(receipt['new_parameter_count'], 98304)
                self.assertEqual(receipt['common_parameter_sha256'], initial['trainable_sha256'])
                self.assertEqual(set(receipt['new_parameter_names']), set(runner.PROJECTION_SHAPES))
                receipts.append(receipt)
                runner.write_json(run / 'input_features_ready.json', dict(donor_provenance=receipt['frozen_donor']))
                config = runner.protocol(args, dict(trainable_sha256=initial['trainable_sha256']), manifest,
                    geometry, baselines, params, runner.task_order(runner.MAX_UPDATES), record['microbatch_caps'],
                    model=model, initialization=receipt)
                state = dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict(), successful_updates=1422)
                torch.save(state, run / 'cpu.pt')
                restored, saved = runner.load_checkpoint(run / 'cpu.pt')
                self.assertEqual(runner.parameter_hash(restored.state_dict()), runner.parameter_hash(state['model']))
                self.assertEqual(saved['args']['model_factory'], runner.MODEL_FACTORY)
                damaged = copy.deepcopy(config)
                damaged['logical_global_batch'] = 512
                with self.assertRaises(ValueError):
                    runner.validate_checkpoint_config(damaged)
                del model, restored, state, saved
            for key in ('trainable_sha256', 'new_parameter_sha256', 'parameter_layout_sha256'):
                self.assertEqual(receipts[0][key], receipts[1][key])

    def test_source_cache_and_private_engine_contract(self):
        from . import r84_experiment
        from .r117_objective import DONOR_FIELDS, INPUT_KEYS, inputs_only
        frozen = {n: runner.file_hash(runner.V4 / n) for n in
                  ('r84_experiment.py', 'r112_experiment.py', 'r112_reporting.py', 'r112_finish.py')}
        with tempfile.TemporaryDirectory(prefix='r117_cpu_') as directory:
            args, record = fixture_lock(Path(directory))
            local = runner._private_engine(json.dumps(dict(record=record, feature_root=str(runner.feature_root(args)),
                common_root=str(args.common_root), provenance={})))
            self.assertEqual((local.BATCH_SIZE, local.MAX_UPDATES, local.EVAL_EVERY), (1024, 28440, 1422))
            self.assertEqual(local.learning_rate(4266), 1e-4)
            self.assertEqual(local.learning_rate(28440), 1e-6)
            self.assertEqual(local.consistency_weight(4266), .35)
            self.assertIs(local.lock_best, runner.lock_best)
            self.assertIn('selection_key', local.train.__code__.co_names)
            self.assertEqual(r84_experiment.MAX_UPDATES, 57600)
            self.assertEqual(set(local.micro_caps('3090', 256)), set(runner.PROBLEMS))
            with self.assertRaises(ValueError):
                local.micro_caps('3090', 128)
            self.assertEqual(runner.protocol_lock(args)['record'], record)
            args.eval_batch_size = 128
            with self.assertRaises(ValueError):
                runner.protocol_lock(args)
        self.assertTrue(set(DONOR_FIELDS) <= INPUT_KEYS)
        self.assertEqual(INPUT_KEYS, models.INPUT_KEYS)
        query = dict(node=1, costs=2, costs_fp64=3, winner=4, ind=5, arbitrary_label=6,
                     nco_nodes=7, nco_node_mask=8, native_nodes=9, native_node_mask=10)
        self.assertEqual(set(inputs_only(query)), {'node', *DONOR_FIELDS})
        self.assertTrue(all((runner.V4 / n).is_file() for n in runner.SOURCE_FILES))
        self.assertEqual(frozen, {n: runner.file_hash(runner.V4 / n) for n in frozen})
        from . import r117_reporting
        self.assertEqual(r117_reporting._callbacks.FACTORY, 'code.V4.r117_experiment')
        self.assertEqual(r117_reporting._callbacks.runner.ARMS, ('A', 'B'))

    def test_cli_without_gpu_or_external_weights(self):
        with tempfile.TemporaryDirectory(prefix='r117_cpu_') as directory:
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                value = runner.main(['describe', '--root', str(root), '--arm', 'B'])
                status = runner.main(['status', '--root', str(root), '--arm', 'A'])
                closed = finish.main(['status', '--root', str(root)])
            self.assertEqual(value['required_protocol_lock']['max_successful_updates'], 28440)
            self.assertIsNone(value['main_protocol_lock'])
            self.assertEqual(status['phase'], 'not_started')
            self.assertEqual(closed['state'], 'not_finished')
            self.assertFalse(list(root.iterdir()))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parser().parse_args(['train', '--arm', 'A', '--gpu-type', '4090'])

    def test_strict_selection_unique_val_lock_and_full18_gate(self):
        def validation(top1, regret):
            return dict(macro=dict(macro_top1=top1, macro_actual_regret_pct=regret))

        self.assertLess(runner.selection_key(validation(.5, 9.)), runner.selection_key(validation(.49, 0.)))
        self.assertLess(runner.selection_key(validation(.5, .1)), runner.selection_key(validation(.5, .2)))
        self.assertFalse(runner.selection_key(validation(.5, .1)) < runner.selection_key(validation(.5, .1)))
        with self.assertRaises(ValueError):
            runner.selection_key(validation(float('nan'), .1))
        with tempfile.TemporaryDirectory(prefix='r117_cpu_') as directory:
            root = Path(directory)
            for arm in runner.ARMS:
                run = root / arm
                run.mkdir()
                val = validation(.5 if arm == 'A' else .503, 1.)
                val['macro']['macro_vs_sbs_pct'] = -.1
                val['per_problem'] = {p: dict(actual_regret_pct=1.) for p in runner.PROBLEMS}
                runner.write_json(run / 'final_metrics.json', dict(val=val))
            self.assertTrue(finish.decision_gate(root)['scientific_gate_passed'])
            final = runner.read_json(root / 'B/final_metrics.json')
            final['val']['macro']['macro_actual_regret_pct'] += .00001
            runner.write_json(root / 'B/final_metrics.json', final)
            self.assertFalse(finish.decision_gate(root)['scientific_gate_passed'])
            run = root / 'A'
            (run / 'best.pt').write_bytes(b'CPU lock fixture; not a model')
            best = dict(epoch=1, successful_updates=1422, val=validation(.5, .1))
            lock = runner.lock_best(run, best, dict(arm='A'), 'fixture-state-hash')
            self.assertEqual(lock['selection_split'], 'val')
            from .r117_features import _test_permission
            self.assertTrue(_test_permission('test', lock))
            with self.assertRaises(PermissionError):
                runner.lock_best(run, dict(best, successful_updates=2844), dict(arm='A'), 'fixture-state-hash')
            with self.assertRaises(finish.NotReady):
                finish.completed(root)


if __name__ == '__main__':
    unittest.main()
