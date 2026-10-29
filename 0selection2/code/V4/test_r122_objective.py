"""Small deterministic checks for the new hard-action objective components."""

import unittest

import torch

from .r122_objective import HardDecisionController, hard_action_terms


def fixture(rows=19, candidates=5):
    generator = torch.Generator().manual_seed(2)
    utility = torch.randn(rows, candidates, generator=generator).requires_grad_()
    costs = 1. + torch.rand(rows, candidates, generator=generator, dtype=torch.float64)
    winner = costs.argmin(-1)
    mask = torch.ones_like(costs, dtype=torch.bool)
    ids = torch.arange(candidates - 1, -1, -1)
    fixed = torch.arange(rows) % 3 == 0
    return utility, costs, winner, mask, ids, fixed


def controller(problems=('TSP', 'CVRP'), **kwargs):
    return HardDecisionController(problems, train_regret_budget=.02,
        train_fixed_miss_budget=.1, train_fixed_population=.2, **kwargs)


class HardActionTests(unittest.TestCase):
    def test_hard_bounds_and_finite_final_score_gradients(self):
        args = fixture()
        terms = hard_action_terms(*args)
        self.assertTrue((terms['classification_proxy'].double() + 1e-6 >= terms['hard_error']).all())
        self.assertTrue((terms['regret_proxy'].double() + 1e-6 >= terms['hard_regret']).all())
        self.assertTrue((terms['missed_proxy'].double() + 1e-6 >= terms['hard_fixed_miss']).all())
        loss, _ = controller().primal_loss(terms)
        loss.backward()
        self.assertTrue(torch.isfinite(args[0].grad).all())
        self.assertGreater(float(args[0].grad.abs().sum()), 0.)

    def test_original_precision_native_targets_and_stable_id_ties(self):
        utility = torch.zeros(1, 2, requires_grad=True)
        costs = torch.tensor([[1. + 1e-10, 1.]], dtype=torch.float64)
        terms = hard_action_terms(utility, costs, torch.tensor([1]),
            torch.ones(1, 2, dtype=torch.bool), torch.tensor([2, 7]), torch.tensor([True]))
        self.assertEqual(int(terms['prediction'][0]), 0)
        self.assertEqual(float(terms['hard_error'][0]), 1.)
        self.assertGreater(float(terms['hard_regret'][0]), 0.)
        with self.assertRaises(ValueError):
            hard_action_terms(utility, costs.float(), torch.tensor([1]),
                torch.ones(1, 2, dtype=torch.bool), torch.tensor([2, 7]), torch.tensor([True]))

    def test_illegal_nan_slots_have_no_effect_or_gradient(self):
        utility = torch.tensor([[.4, -.7, float('nan')]], requires_grad=True)
        costs = torch.tensor([[1.2, 1., float('nan')]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        terms = hard_action_terms(utility, costs, torch.tensor([1]), mask,
                                  torch.tensor([2, 7, 9]), torch.tensor([True]))
        loss, _ = controller().primal_loss(terms)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(utility.grad).all())
        self.assertEqual(float(utility.grad[0, 2]), 0.)

    def test_single_legal_candidate(self):
        for scores, mask, ids, winner in (
            ([[.2]], [[True]], [3], [0]),
            ([[float('nan'), .2]], [[False, True]], [1, 3], [1]),
        ):
            utility = torch.tensor(scores, requires_grad=True)
            costs = torch.full_like(utility, 2., dtype=torch.float64)
            terms = hard_action_terms(utility, costs, torch.tensor(winner),
                torch.tensor(mask), torch.tensor(ids), torch.tensor([True]))
            loss, _ = controller().primal_loss(terms)
            loss.backward()
            self.assertTrue(torch.isfinite(utility.grad).all())
            self.assertEqual(float(terms['hard_regret'].sum()), 0.)
            self.assertEqual(float(terms['regret_proxy'].sum()), 0.)

    def test_microbatch_loss_and_gradients_match_full_logical_batch(self):
        args = fixture()
        full = hard_action_terms(*args)
        state = controller()
        loss, _ = state.primal_loss(full)
        expected, = torch.autograd.grad(loss, args[0])
        utility = args[0].detach().clone().requires_grad_()
        micro_loss = 0.
        for start, stop in ((0, 3), (3, 11), (11, 19)):
            parts = hard_action_terms(utility[start:stop], args[1][start:stop],
                args[2][start:stop], args[3][start:stop], args[4], args[5][start:stop])
            value, _ = state.primal_loss(parts)
            micro_loss = micro_loss + value * (stop - start) / 19.
        actual, = torch.autograd.grad(micro_loss, utility)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertEqual(int(state.committed_updates), 0)
        self.assertEqual(int(state.completed_cycles), 0)


class ControllerTests(unittest.TestCase):
    def test_budget_source_cannot_be_validation_or_test(self):
        for split in ('val', 'test', 'lib'):
            with self.assertRaises(ValueError):
                controller(budget_source_split=split)

    def test_complete_task_cycles_and_correct_feedback_direction(self):
        state = controller(ema_decay=0., dual_lr=1.)
        self.assertFalse(state.commit('TSP', hard_regret=.04, hard_fixed_miss=.3, successful_update=1))
        torch.testing.assert_close(state.multipliers, torch.ones(2, dtype=torch.float64))
        self.assertTrue(state.commit('CVRP', hard_regret=.04, hard_fixed_miss=.3, successful_update=2))
        high = state.multipliers.clone()
        self.assertTrue((high > 1).all())
        state.commit('CVRP', hard_regret=0., hard_fixed_miss=0., successful_update=3)
        state.commit('TSP', hard_regret=0., hard_fixed_miss=0., successful_update=4)
        self.assertTrue((state.multipliers < high).all())
        self.assertEqual(int(state.completed_cycles), 2)

    def test_replay_eval_and_resume_do_not_silently_advance_feedback(self):
        state = controller()
        state.commit('TSP', hard_regret=.04, hard_fixed_miss=.3, successful_update=1)
        before = {key: value.clone() for key, value in state.state_dict().items()}
        with self.assertRaises(ValueError):
            state.commit('TSP', hard_regret=.04, hard_fixed_miss=.3, successful_update=1)
        with self.assertRaises(RuntimeError):
            state.eval().commit('CVRP', hard_regret=.04, hard_fixed_miss=.3, successful_update=2)
        for key, value in state.state_dict().items():
            torch.testing.assert_close(value, before[key])
        resumed = controller()
        resumed.load_state_dict(before)
        self.assertTrue(resumed.commit('CVRP', hard_regret=.04, hard_fixed_miss=.3, successful_update=2))
        self.assertEqual(int(resumed.completed_cycles), 1)


if __name__ == '__main__':
    unittest.main()
