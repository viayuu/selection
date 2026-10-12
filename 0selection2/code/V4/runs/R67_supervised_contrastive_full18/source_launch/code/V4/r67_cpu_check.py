"""One focused synthetic CPU check of Eq.2 and the actual stage boundary."""

import copy
import math

import torch
import torch.nn.functional as F

from code.unified_selector.registry import K_CBITS, K_PROBLEM_DESC
from .local_geometry import FIELDS
from .multitask_probe import state_hash
from .r65_optimizer import configure_update_lr, evaluation_mode, make_optimizer, observed_lr, training_mode
from .r67_model import INPUT_KEYS, PretrainingSelector, cpu_state, model_inputs, shared_initialization, transition_to_stage2
from .r67_objective import GradientLayout, contrastive_masks, supervised_contrastive, update_triple


def scalar_reference(first, second, winner):
    features = torch.cat((first, second)).float()
    labels = torch.cat((winner, winner))
    anchors = []
    for anchor in range(len(features)):
        other = [index for index in range(len(features)) if index != anchor]
        denominator = torch.logsumexp(torch.stack([
            (features[anchor]*features[index]).sum()/.1 for index in other]), dim=0)
        positives = [index for index in other if labels[index] == labels[anchor]]
        anchors.append(torch.stack([
            (features[anchor]*features[index]).sum()/.1-denominator for index in positives]).mean())
    return -.1*torch.stack(anchors).mean()


def focused_check():
    torch.set_num_threads(1)
    torch.manual_seed(670001)
    winner = torch.tensor([0, 1, 1, 2])
    first = F.normalize(torch.randn(4, 7), dim=-1).requires_grad_()
    second = F.normalize(torch.randn(4, 7), dim=-1).requires_grad_()
    loss, parts = supervised_contrastive(first, second, winner)
    ref_first, ref_second = first.detach().clone().requires_grad_(), second.detach().clone().requires_grad_()
    reference = scalar_reference(ref_first, ref_second, winner)
    torch.testing.assert_close(loss, reference, rtol=1e-6, atol=1e-6)
    loss.backward()
    reference.backward()
    for actual, expected in ((first.grad, ref_first.grad), (second.grad, ref_second.grad)):
        assert actual is not None and torch.isfinite(actual).all() and actual.norm() > 0
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-6)
    positive, nonself = contrastive_masks(winner)
    assert not positive.diag().any() and not nonself.diag().any()
    assert torch.equal(positive.sum(1), torch.tensor([1, 3, 3, 1, 1, 3, 3, 1]))
    assert all(positive[index, index+4] and positive[index+4, index] for index in range(4))
    assert torch.equal(nonself.sum(1), torch.full((8,), 7))

    geometry = dict(mean=[0.]*len(FIELDS), std=[1.]*len(FIELDS))
    params, states, initialization = shared_initialization(geometry)
    arms = {arm: PretrainingSelector(params, states) for arm in ('A', 'B')}
    assert state_hash(arms['A'].state_dict()) == state_hash(arms['B'].state_dict())
    fixture = dict(problem_id=0, kind='coord', node=torch.rand(4, 8, 8),
        node_mask=torch.ones(4, 8, dtype=torch.bool), n=torch.full((4,), 8),
        cbits=torch.zeros(4, K_CBITS), problem_desc=torch.zeros(4, K_PROBLEM_DESC),
        coord_dist=torch.zeros(4, dtype=torch.long), pool_ids=torch.tensor([0, 1, 3, 4]),
        solver_mask=torch.tensor([[True, True, True, False]]*4),
        node_geom=torch.zeros(4, 8, len(FIELDS)), ind=winner,
        costs=torch.full((4, 4), float('nan')), winner='forbidden',
        routes='forbidden', performance_target='forbidden')
    seen, calls = [], []
    model = arms['A']
    original_encode = model.selector.encode_pairs

    def counted_encode(batch):
        assert set(batch) <= INPUT_KEYS
        assert {'node', 'node_mask', 'n', 'cbits', 'problem_desc', 'coord_dist', 'pool_ids', 'node_geom', 'solver_mask'} <= set(batch)
        seen.append(set(batch))
        calls.append(1)
        return original_encode(batch)

    model.selector.encode_pairs = counted_encode
    model.eval()
    with torch.no_grad():
        initial = model(fixture)
        changed = model(dict(fixture, ind=torch.tensor([2, 0, 0, 1]), costs=torch.randn(4, 4),
                             winner='changed', routes='changed', performance_target='changed'))
    assert len(calls) == 2
    for key in ('logits', 'representation', 'projection'):
        torch.testing.assert_close(initial[key], changed[key], rtol=0, atol=0)
    assert initial['representation'].shape == (4, 384) and initial['projection'].shape == (4, 128)
    torch.testing.assert_close(initial['representation'].norm(dim=-1), torch.ones(4))
    torch.testing.assert_close(initial['projection'].norm(dim=-1), torch.ones(4))
    assert initial['logits'][:, -1].isneginf().all()
    del model.selector.encode_pairs

    transitions, new_heads, first_lrs = {}, {}, []
    for arm, current in arms.items():
        optimizer = make_optimizer(current, 'B')
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        layout = GradientLayout(current)
        inactive_module = current.projector if arm == 'A' else current.selector.score_head
        inactive_before = state_hash(inactive_module.state_dict())
        triples = [dict(fixture, problem_id=task) for task in range(3)]
        for successful in range(3):
            training_mode(current, optimizer, 'B')
            expected = configure_update_lr(optimizer, 'B', successful)
            torch.manual_seed(670100+successful)
            values = update_triple(current, optimizer, scaler, triples, .35*(successful+1)/720,
                                   layout, arm, 1)
            assert values['finite_loss'] and values['finite_gradient']
            lr = observed_lr(optimizer, 'B', expected)
            if successful == 0:
                first_lrs.append(lr)
            assert all(p.grad is None for p in inactive_module.parameters())
            assert all('z' not in optimizer.state.get(p, {}) for p in inactive_module.parameters())
        assert state_hash(inactive_module.state_dict()) == inactive_before
        y = cpu_state(current.selector)
        evaluation_mode(current, optimizer, 'B')
        x = cpu_state(current.selector)
        assert any(not torch.equal(y[key], x[key]) for key in x if not key.startswith('score_head.'))
        old_head = copy.deepcopy(current.selector.score_head)
        old_head[0].weight.data.fill_(123.)
        old_head[-1].bias.data.fill_(456.)
        current.selector.score_head = old_head
        selector, fresh_optimizer, fresh_scaler, receipt = transition_to_stage2(current, optimizer,
                                                                              states['stage2_head'], amp=False)
        for key, value in selector.state_dict().items():
            if not key.startswith('score_head.'):
                torch.testing.assert_close(value, x[key], rtol=0, atol=0)
        assert not hasattr(selector, 'projector')
        assert state_hash(selector.score_head.state_dict()) == initialization['hashes']['stage2_head']
        assert fresh_optimizer is not optimizer and len(fresh_optimizer.state) == 0
        assert all(group['k'] == 0 and group['weight_sum'] == 0 for group in fresh_optimizer.param_groups)
        assert fresh_scaler is not scaler and fresh_scaler.state_dict() == {}
        training_mode(selector, fresh_optimizer, 'B')
        expected = configure_update_lr(fresh_optimizer, 'B', 0)
        assert math.isclose(expected, 3e-4/720, rel_tol=1e-15) and receipt['stage_rdrop_updates'] == 0
        torch.manual_seed(670400)
        stage2_values = update_triple(selector, fresh_optimizer, fresh_scaler,
            [model_inputs(batch) | {'ind': batch['ind']} for batch in triples], .35/720,
            GradientLayout(selector), arm, 2)
        assert stage2_values['finite_gradient'] and any(p.grad is not None for p in selector.score_head[0].parameters())
        transitions[arm], new_heads[arm] = receipt, receipt['stage2_head_hash']
    assert new_heads['A'] == new_heads['B'] and all(math.isclose(lr, 3e-4/720, rel_tol=1e-15) for lr in first_lrs)
    assert states['stage2_head']['0.weight'].count_nonzero() and states['stage2_head']['3.weight'].count_nonzero()
    return dict(passed=True, focused_cpu_checks=1, gpu_used=False, synthetic_fixtures_only=True,
        eq2_scalar_reference_loss=float(reference.detach()), eq2_loss=float(loss.detach()),
        positive_self_and_counterpart_masks=True, both_view_gradients_match_scalar_reference=True,
        fixed_temperature=.1, fixed_project_loss_multiplier=.1, target_isolated_forward=True,
        encode_pairs_once_per_view=True, original_input_fields_preserved=True,
        identical_shared_fresh_selector_projector=True, unused_heads_grad_none_and_unchanged=True,
        terminal_x_trunk_transfer=True, entire_head_reset_identical=True, projector_discarded=True,
        optimizer_scaler_and_stage_counters_fresh=True, first_effective_lr=first_lrs[0],
        shared_initialization=initialization, stage_transitions=transitions, test_read=False,
        no_training_overfit_or_timing_pilot=True)
