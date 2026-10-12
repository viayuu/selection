"""One focused CPU check of R66's offset, initialization, boundary and modes."""

import copy
import io

import torch

from code.unified_selector.registry import K_CBITS, K_PROBLEM_DESC
from .multitask_probe import state_hash
from .r65_gradients import GradientLayout, update_triple
from .r65_optimizer import (configure_update_lr, evaluation_mode, make_optimizer,
                            observed_lr, training_mode)
from .r66_model import (BASE_SHA256, CACHE_INPUT, INPUT_KEYS, build_selector,
                        load_base_caches, load_base_checkpoint, shared_initialization)
from .r66_objective import objective


def focused_check():
    checkpoint = load_base_checkpoint()
    params, geometry = checkpoint['args']['model_params'], checkpoint['args']['geometry_stats']
    initial, initialization = shared_initialization(params, geometry)
    caches, cache_receipt = load_base_caches()
    del caches
    base = torch.tensor([[1.7, -.8, .2], [-.5, .4, 1.1]], requires_grad=True)
    mask, winner = torch.ones(2, 3, dtype=torch.bool), torch.tensor([0, 1])
    gradients, losses = {}, {}
    for arm in ('A', 'B'):
        first = torch.tensor([[.2, -.1, .3], [.1, .2, -.2]], requires_grad=True)
        second = (first.detach()+torch.tensor([[.1, .2, -.1], [-.1, .1, .2]])).requires_grad_()
        loss, parts = objective(dict(logits=first, solver_mask=mask),
            dict(logits=second, solver_mask=mask), base, winner, .35, arm)
        loss.backward()
        assert first.grad is not None and second.grad is not None and base.grad is None
        gradients[arm], losses[arm] = first.grad.clone(), float(loss.detach())
        offset = 0. if arm == 'A' else base.detach()
        expected = .5*(torch.nn.functional.cross_entropy(first+offset, winner)
                       +torch.nn.functional.cross_entropy(second+offset, winner))
        torch.testing.assert_close(parts['ce'], expected, rtol=0, atol=0)
    assert not torch.equal(gradients['A'], gradients['B'])
    model = build_selector(params, checkpoint['model'], initial, 'B', cached=True)
    base_hash = state_hash(model.base.state_dict())
    optimizer = make_optimizer(model.member, 'B')
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {
        id(p) for p in model.member.parameters()}
    seen = []

    def boundary(module, args):
        assert set(args[0]) <= INPUT_KEYS
        seen.append('base' if module is model.base else 'member')

    hooks = [module.register_forward_pre_hook(boundary) for module in (model.base, model.member)]
    fixture = dict(problem_id=0, kind='coord', node=torch.rand(2, 8, 8),
        node_mask=torch.ones(2, 8, dtype=torch.bool), n=torch.full((2,), 8),
        cbits=torch.zeros(2, K_CBITS), problem_desc=torch.zeros(2, K_PROBLEM_DESC),
        coord_dist=torch.zeros(2, dtype=torch.long), pool_ids=torch.tensor([0, 1, 3]),
        node_geom=torch.zeros(2, 8, len(geometry['mean'])),
        ind=winner, costs=torch.full((2, 3), float('nan')), winner='forbidden',
        routes='forbidden', pred='forbidden', **{CACHE_INPUT: base.detach()})
    training_mode(model, optimizer, 'B')
    assert not model.base.training and all(not p.requires_grad for p in model.base.parameters())
    assert all(not module.training for module in model.base.modules())
    output = model(fixture)
    torch.testing.assert_close(output['logits'], base.detach(), rtol=0, atol=0)
    torch.nn.functional.cross_entropy(output['logits'], winner).backward()
    assert not model.member.instance_encoder.coord_proj.weight.grad.count_nonzero()
    assert model.member.score_head[-1].weight.grad.norm() > 0
    optimizer.zero_grad(set_to_none=True)
    triples = [dict(fixture, problem_id=task) for task in range(3)]
    expected = configure_update_lr(optimizer, 'B', 0)
    values = update_triple(model, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                           triples, .35/720, GradientLayout(model))
    lr = observed_lr(optimizer, 'B', expected)
    assert values['finite_loss'] and values['finite_gradient']
    for successful in range(1, 3):
        configure_update_lr(optimizer, 'B', successful)
        update_triple(model, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                      triples, .35*(successful+1)/720, GradientLayout(model))
    y = {key: value.detach().clone() for key, value in model.member.state_dict().items()}
    evaluation_mode(model, optimizer, 'B')
    x = {key: value.detach().clone() for key, value in model.member.state_dict().items()}
    assert any(not torch.equal(x[key], y[key]) for key in x)
    assert all(not group['train_mode'] for group in optimizer.param_groups)
    stream = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location='cpu', weights_only=False)
    resumed = build_selector(params, checkpoint['model'], initial, 'B', cached=True)
    resumed_optimizer = make_optimizer(resumed.member, 'B')
    resumed.load_state_dict(saved['model'])
    resumed_optimizer.load_state_dict(saved['optimizer'])
    training_mode(model, optimizer, 'B')
    training_mode(resumed, resumed_optimizer, 'B')
    for successful in (3,):
        for current, opt in ((model, optimizer), (resumed, resumed_optimizer)):
            torch.manual_seed(660100+successful)
            configure_update_lr(opt, 'B', successful)
            update_triple(current, opt, torch.amp.GradScaler('cuda', enabled=False),
                          triples, .35*(successful+1)/720, GradientLayout(current))
            evaluation_mode(current, opt, 'B')
    assert state_hash(model.state_dict()) == state_hash(resumed.state_dict())
    assert state_hash(optimizer.state_dict()) == state_hash(resumed_optimizer.state_dict())
    seen.clear()
    deployed = model.deployment_forward(fixture)
    assert seen == ['base', 'member']
    clean = {key: value for key, value in fixture.items() if key in INPUT_KEYS}
    with torch.no_grad():
        expected_logits = model.base(clean)['logits']+model.member(clean)['logits']
    torch.testing.assert_close(deployed['logits'], expected_logits, rtol=0, atol=0)
    independent = copy.deepcopy(model)
    independent.arm = 'A'
    torch.testing.assert_close(independent.deployment_forward(fixture)['logits'], deployed['logits'], rtol=0, atol=0)
    independent.train()
    assert not independent.base.training
    assert state_hash(model.base.state_dict()) == base_hash
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
    for hook in hooks:
        hook.remove()
    return (dict(passed=True, focused_cpu_checks=1, gpu_used=False, synthetic_fixtures_only=True,
        real_train_val_cache_alignment=True, cache_fields_only=['logits', 'indices', 'pool_ids', 'pool'],
        original_indices_names_and_pool_ids=True, base_sha256=BASE_SHA256,
        offset_objectives_and_both_branches=True, objective_fixture_losses=losses,
        identical_explicit_copied_initialization=True, zero_scalar_head=True,
        expected_initial_zero_encoder_gradient=True, frozen_base_no_grad_eval=True,
        no_label_cost_winner_route_forward_inputs=True, composite_live_deployment_both_arms=True,
        optimizer_only_member=True, optimizer='exact official R65B AdamWScheduleFree',
        y_train_x_eval_save=True, exact_resume_model_and_optimizer=True,
        reused_whole_triple_update_finite=True, first_effective_lr=lr,
        no_external_warmup=True, test_read=False),
        checkpoint, initial, initialization, cache_receipt)
