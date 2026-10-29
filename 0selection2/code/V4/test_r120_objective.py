import pytest
import torch

from .r120_objective import native_targets, safe_model_inputs, single_loss


def test_native_max_direction_and_fp64_exact_ties():
    labels = dict(native_objective=torch.tensor([[9., 10., 10.]], dtype=torch.float64),
                  observed_mask=torch.ones(1, 3, dtype=torch.bool), objective_direction='max')
    best, advantage, excess = native_targets(labels, 10.)
    assert best.tolist() == [[False, True, True]]
    torch.testing.assert_close(excess, torch.tensor([[.1, 0., 0.]]))
    torch.testing.assert_close(advantage.sum(-1), torch.zeros(1), atol=1e-7, rtol=0)


def test_zero_optimum_uses_fixed_training_unit_and_set_ce_accepts_all_ties():
    labels = dict(native_objective=torch.tensor([[0., 0., 0.]], dtype=torch.float64),
                  observed_mask=torch.ones(1, 3, dtype=torch.bool), objective_direction='min',
                  all_best_mask=torch.ones(1, 3, dtype=torch.bool))
    output = dict(logits=torch.tensor([[5., -2., 1.]], requires_grad=True),
                  solver_mask=torch.ones(1, 3, dtype=torch.bool),
                  advantage=torch.zeros(1, 3, requires_grad=True))
    loss, parts = single_loss(output, labels, 100.)
    assert float(loss) == 0.
    assert float(parts['ce']) == 0.
    loss.backward()
    assert torch.isfinite(output['logits'].grad).all()
    labels['native_objective'] = torch.tensor([[0., 5., 10.]], dtype=torch.float64)
    _, _, excess = native_targets(labels, 100.)
    torch.testing.assert_close(excess, torch.tensor([[0., .05, .1]]))


def test_native_routing_winner_is_not_silently_replaced_by_tie_mask():
    labels = dict(native_objective=torch.tensor([[2., 2.]], dtype=torch.float64),
                  observed_mask=torch.ones(1, 2, dtype=torch.bool), objective_direction='min',
                  native_winner_index=torch.tensor([1]), all_best_mask=torch.ones(1, 2, dtype=torch.bool))
    output = dict(logits=torch.tensor([[2., -2.]], requires_grad=True),
                  solver_mask=torch.ones(1, 2, dtype=torch.bool), advantage=torch.zeros(1, 2))
    _, parts = single_loss(output, labels, 2.)
    assert parts['ce'] > 4.


def test_missing_nan_is_not_zero_cost_and_label_namespace_is_excluded():
    labels = dict(native_objective=torch.tensor([[3., float('nan')]], dtype=torch.float64),
                  observed_mask=torch.tensor([[True, False]]), objective_direction='min')
    best, advantage, excess = native_targets(labels, 3.)
    assert best.tolist() == [[True, False]]
    assert torch.isfinite(advantage).all() and torch.isfinite(excess).all()
    batch = dict(problem='MIS', inputs={}, pool_ids=torch.tensor([0]), labels=labels,
                 metadata={'instance_id': 'private'}, costs=torch.tensor([0.]))
    assert set(safe_model_inputs(batch)) == {'problem', 'inputs', 'pool_ids'}
    with pytest.raises(ValueError, match='FP64'):
        native_targets(dict(labels, native_objective=labels['native_objective'].float()), 3.)
