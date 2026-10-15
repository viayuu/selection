"""Small CPU witnesses for R80; no GPU, dataset scans, solver/API or training launches."""

import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from code.unified_selector.registry import GLOBAL_SOLVERS, M_GLOBAL, PROBLEMS
from . import r80_experiment as r80
from .r76_experiment import PART_KEYS, Runtime, update_batch, write_json
from .r77_data import parameter_hash
from .solver_code_encoder import CODE_DIM, ROLES
from .test_r76 import CountingSGD, fixture_batch, fixture_model
from .test_r77 import ScalarObjective


def fresh_fixture():
    model, params = fixture_model()
    values = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
    initial = dict(model_params=params, trainable=values, trainable_sha256=parameter_hash(values),
        seed=2, initialization='scratch')
    size = len(model.instance_encoder.geometry_residual.mean)
    return initial, dict(mean=[2.] * size, std=[3.] * size)


def source_fixture():
    generator = torch.Generator().manual_seed(80)
    vectors = F.normalize(torch.randn(M_GLOBAL, len(ROLES), CODE_DIM, generator=generator), dim=-1)
    return dict(vectors=vectors, role_mask=torch.ones(M_GLOBAL, len(ROLES), dtype=torch.bool))


class RecordingScores(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10)
        self.dropout = nn.Dropout(.1)
        self.outputs = []

    def forward(self, batch):
        if set(batch) != {'x'}:
            raise AssertionError('Labels leaked into the scoring path')
        scores = self.dropout(batch['x']) @ self.weight
        self.outputs.append(scores)
        return dict(logits=scores)


class R80Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.cuda_guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not access CUDA'))
        self.cuda_guard.start()
        self.addCleanup(self.cuda_guard.stop)

    def test_ce_exact_two_pass_mean_without_other_labels(self):
        model = RecordingScores().train()
        batch = dict(x=torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10,
            ind=torch.tensor([0, 1, 2, 0, 1]), costs=torch.full((5, 3), float('nan')))
        loss, parts = r80.CEOnlyObjective(model)(batch, 999.)
        self.assertEqual(len(model.outputs), 2)
        ce = .5 * sum(F.cross_entropy(scores.float(), batch['ind']) for scores in model.outputs)
        torch.testing.assert_close(loss, .35 * ce, rtol=0., atol=0.)
        torch.testing.assert_close(parts[1], ce.detach(), rtol=0., atol=0.)
        self.assertEqual(torch.count_nonzero(parts[2:]).item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())

    def test_fixed_budgets_exposure_lr_and_ramp(self):
        a, b = r80.arm_spec('A_ce_only'), r80.arm_spec('B_batch512')
        self.assertEqual((a.max_updates, a.warmup_updates, a.eval_every), (49680, 3726, 1242))
        self.assertEqual((b.max_updates, b.warmup_updates, b.eval_every), (99360, 7452, 2484))
        for arm in r80.ARMS:
            spec = r80.arm_spec(arm)
            self.assertEqual(spec.max_updates * spec.batch_size, 50872320)
            self.assertEqual(spec.eval_every * spec.batch_size, 1271808)
            self.assertEqual(r80.learning_rate(spec.warmup_updates, spec), spec.peak_lr)
            self.assertEqual(r80.learning_rate(spec.max_updates, spec), 1e-6)
        for step in (1, 18, 1242, 3726, 4000, 49680):
            self.assertEqual(r80.learning_rate(step, a), r80.learning_rate(2 * step, b))
            self.assertEqual(r80.consistency_weight(step, 'C_lr3e4'), r80.consistency_weight(2 * step, b))
        self.assertEqual(r80.consistency_weight(100, a), 0.)
        self.assertEqual(r80.learning_rate(3726, 'C_lr3e4'), 3e-4)
        with self.assertRaises(ValueError):
            r80.learning_rate(0, a)

    def test_same_task_source_requests_split_for_b(self):
        order = r80.task_order(updates=18)
        streams = lambda: {p: r80.ShuffleStream(71, 2 + i) for i, p in enumerate(PROBLEMS)}
        a_streams, b_streams = streams(), streams()
        a = r80.interval_requests(order, a_streams, 0, 18, r80.arm_spec('A_ce_only'))
        b = r80.interval_requests(order, b_streams, 0, 36, r80.arm_spec('B_batch512'))
        self.assertEqual(Counter(p for p, _, _ in a), Counter(dict.fromkeys(PROBLEMS, 1)))
        for index, (problem, rows, count) in enumerate(a):
            halves = b[2 * index:2 * index + 2]
            self.assertEqual([entry[0] for entry in halves], [problem, problem])
            self.assertEqual([entry[2] for entry in halves], [512, 512])
            torch.testing.assert_close(rows, torch.cat([entry[1] for entry in halves]))
            self.assertEqual(count, 1024)
        self.assertEqual({p: s.draws for p, s in a_streams.items()}, {p: s.draws for p, s in b_streams.items()})

    def test_shared_fresh_parameters_and_full_train_geometry(self):
        initial, geometry = fresh_fixture()
        for arm in r80.ARMS[:3]:
            model, params, receipt = r80.initialize_shared(initial, geometry, arm)
            self.assertEqual(parameter_hash(dict(model.named_parameters())), initial['trainable_sha256'])
            self.assertFalse(receipt['historical_checkpoint_loaded'])
            self.assertEqual(params['dropout'], .1)
            torch.testing.assert_close(model.instance_encoder.geometry_residual.mean, torch.tensor(geometry['mean']))

    def test_code_source_frozen_but_projections_recomputed_and_trainable(self):
        initial, geometry = fresh_fixture()
        bundle = source_fixture()
        with patch('code.V4.solver_code_embeddings.validate_bundle'):
            model, params, receipt = r80.initialize_shared(initial, geometry, 'D_code_embedding', bundle=bundle)
        common = receipt['common_parameter_names']
        for name, value in model.named_parameters():
            if name in common:
                torch.testing.assert_close(value, initial['trainable'][name], rtol=0., atol=0.)
        self.assertIn('solver_encoder.solver_emb.weight', common)
        self.assertEqual(params['solver_representation'], 'code')
        before = r80.frozen_source_hash(model)
        encoder = model.solver_encoder
        self.assertFalse(encoder.code_vectors.requires_grad)
        self.assertNotIn('code_vectors', dict(encoder.named_parameters()))
        optimizer = torch.optim.SGD(encoder.parameters(), lr=.03)
        original = encoder.role_proj[0][1].weight.detach().clone()
        calls = []
        hook = encoder.role_proj[0].register_forward_hook(lambda *unused: calls.append(1))
        ids = torch.arange(M_GLOBAL)
        for _ in range(2):
            optimizer.zero_grad()
            scores = encoder(ids)
            (scores[:, 0] ** 2).mean().backward()
            optimizer.step()
        hook.remove()
        self.assertEqual(len(calls), 2)
        self.assertFalse(torch.equal(original, encoder.role_proj[0][1].weight))
        self.assertEqual(r80.frozen_source_hash(model), before)
        torch.testing.assert_close(encoder.code_vectors, bundle['vectors'])

    def test_same_score_factory_and_native_checkpoint_roundtrip(self):
        initial, geometry = fresh_fixture()
        model, params, receipt = r80.initialize_shared(initial, geometry, 'A_ce_only')
        model.eval()
        batch, _ = fixture_batch('TSP', 3)
        config = dict(experiment='R80', arm='A_ce_only', initialization='scratch_shared_r77',
            solver_names=GLOBAL_SOLVERS, model_params=params)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            torch.save(dict(args=config, state_hash=r80.json_hash(config), model=model.state_dict(), initialization=receipt), path)
            loaded, _ = r80.load_checkpoint(path)
            loaded.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(r80.inputs_only(batch))['logits'], loaded(r80.inputs_only(batch))['logits'])
            self.assertIs(r80.load_checkpoint, r80.load_model)

    def test_microcaps_keep_mvrp_atsp_whole(self):
        for arm in r80.ARMS:
            for gpu, cap in (('4090', 352), ('3090', 384)):
                caps = r80.micro_caps(arm, gpu)
                self.assertEqual(caps['TSP'], cap)
                self.assertEqual(caps['CVRP'], cap)
                self.assertEqual(caps['ATSP'], r80.arm_spec(arm).batch_size)
                self.assertTrue(all(caps[p] == r80.arm_spec(arm).batch_size for p in PROBLEMS[3:]))

    def test_actual_row_micro_weight_single_clip_step_and_amp_rng_replay(self):
        batch = dict(ind=torch.zeros(5, dtype=torch.long), x=torch.arange(5) * .1, y=torch.ones(5) * .2)
        objectives = [ScalarObjective(), ScalarObjective()]
        for objective, cap in zip(objectives, (5, 2)):
            optimizer = CountingSGD(objective.parameters())
            scaler = torch.amp.GradScaler('cpu', enabled=False)
            with patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
                _, retries = update_batch(objective, optimizer, scaler, batch, 0., 5, 5,
                    Runtime(0, 1, torch.device('cpu')), microbatch_cap=cap)
            self.assertEqual((optimizer.steps, retries, clip.call_count), (1, 0, 1))
        torch.testing.assert_close(objectives[0].weight, objectives[1].weight)
        objective = ScalarObjective()
        optimizer = CountingSGD(objective.parameters())
        scaler = torch.amp.GradScaler('cpu', init_scale=8., growth_interval=100)
        injected = [False]

        def overflow(gradient):
            if not injected[0]:
                injected[0] = True
                return torch.full_like(gradient, float('nan'))
            return gradient

        hook = objective.weight.register_hook(overflow)
        _, retries = update_batch(objective, optimizer, scaler, batch, 0., 5, 5,
            Runtime(0, 1, torch.device('cpu')), microbatch_cap=2, max_retries=3)
        hook.remove()
        self.assertEqual((optimizer.steps, retries), (1, 1))
        self.assertEqual(objective.draws[:3], objective.draws[3:])

    def test_resume_pending_interval_and_source_cursor(self):
        spec = r80.arm_spec('B_batch512')
        config = dict(arm=spec.arm)
        counts = dict.fromkeys(PROBLEMS, 138)
        state = dict(args=config, state_hash=r80.json_hash(config), successful_updates=2484,
            pending_evaluation=dict(loss=1.), history=[], updates_by_problem=counts,
            stream_draws={p: count * 512 for p, count in counts.items()})
        r80.validate_resume(state, config, spec)
        state['stream_draws']['TSP'] += 1
        with self.assertRaises(ValueError):
            r80.validate_resume(state, config, spec)

    def test_test_guard_and_saved_predictions_not_blocked_by_old_started_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(root=Path(directory), arm='A_ce_only', data_root=Path('/fixture'),
                state_hash=None, gpu_type='4090', device='cpu')
            root = args.root / args.arm
            root.mkdir()
            with self.assertRaises(PermissionError):
                r80.authorize_test(args)
            write_json(root / 'test_started.json', dict(checkpoint_sha256='fixed'))
            result = dict(macro=dict(macro_top1=.5))
            write_json(root / 'test_results.json', dict(complete=True, checkpoint_sha256='fixed', state_hash='state', result=result))
            write_json(root / 'final_metrics.json', dict(val={}))
            with patch.object(r80, 'authorize_test', return_value=(dict(checkpoint_sha256='fixed', state_hash='state',
                    selected_update=1242), {})), patch.object(r80, 'load_checkpoint', side_effect=AssertionError('No repeated inference')):
                self.assertEqual(r80.test(args, torch.device('cpu')), result)
            self.assertTrue((root / 'test_complete.json').exists())

    def test_cli_ready_with_early_progress(self):
        args = r80.parser().parse_args(['train', '--arm', 'A_ce_only', '--gpu-type', '3090'])
        self.assertEqual(args.log_every, 18)
        self.assertEqual(args.wandb_mode, 'auto')
        self.assertEqual(args.eval_batch_size, 128)


if __name__ == '__main__':
    unittest.main()
