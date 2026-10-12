"""One focused synthetic check of R68 initialization, SAM and whole-update retry."""

import math

import torch
from torch import nn

from .local_geometry import FIELDS
from .multitask_probe import state_hash
from .r64_model import initialize
from .r67_model import score_head
from .r68_gradient import GradientLayout, SavedWeights, update_triple
from .r68_model import INPUT_KEYS, cpu_state, make_optimizer, make_selector, starting_state, update_lr
from .training_monitor import capture_rng


class ToySelector(nn.Module):
    def __init__(self, fail_on=None, failure='nonfinite'):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4)/20)
        self.bias = nn.Parameter(torch.zeros(4))
        self.unused = nn.Parameter(torch.tensor([3.]))
        self.dropout = nn.Dropout(.25)
        self.masks, self.weights, self.rows = [], [], []
        self.fail_on, self.failure, self.observer = fail_on, failure, None

    def forward(self, batch):
        assert set(batch) <= INPUT_KEYS and 'ind' not in batch and 'costs' not in batch
        if self.observer is not None:
            self.observer()
        mask = self.dropout(torch.ones_like(batch['node']))
        self.masks.append(mask.detach().clone())
        self.weights.append(self.weight.detach().clone())
        self.rows.append(batch['node'].detach().clone())
        logits = (batch['node']*mask)@self.weight+self.bias
        if len(self.masks) == self.fail_on:
            if self.failure == 'exception':
                raise RuntimeError('injected second-pass exception')
            logits = logits*float('nan')
        return dict(logits=logits.float(), solver_mask=batch['solver_mask'])


def focused_check():
    torch.set_num_threads(1)
    torch.manual_seed(680001)
    fixture = dict(node=torch.tensor([[.2, .5, .8], [.7, .4, .1], [.9, .3, .6], [.1, .8, .4]]),
        solver_mask=torch.ones(4, 4, dtype=torch.bool), pool_ids=torch.arange(4),
        ind=torch.tensor([0, 1, 2, 1]), costs=torch.full((4, 4), float('nan')),
        performance_target='forbidden', routes='forbidden')
    triples = [dict(fixture, problem_id=task) for task in range(3)]
    toy = ToySelector()
    layout = GradientLayout(toy)
    saved = SavedWeights(layout)
    gradient = torch.randn(sum(p.numel() for p in layout.parameters))
    active = [name != 'unused' for name in layout.names]
    for section, used in zip(layout.slices, active):
        if not used:
            gradient[section] = 0
    effective, requested = saved.perturb(gradient, active)
    assert math.isclose(effective, .05, abs_tol=2e-7)
    assert math.isclose(requested, .05, abs_tol=2e-7)
    assert any(not torch.equal(p, original) for p, original in zip(layout.parameters, saved.originals))
    saved.restore()
    assert all(torch.equal(p, original) for p, original in zip(layout.parameters, saved.originals))

    retry_model = ToySelector(fail_on=8)
    original = cpu_state(retry_model)
    optimizer = make_optimizer(retry_model)
    step_calls = []

    def before_step(opt, args, kwargs):
        assert len(opt.state) == 0
        assert state_hash(cpu_state(retry_model)) == state_hash(original)

    optimizer.register_step_pre_hook(before_step)
    optimizer.register_step_post_hook(lambda opt, args, kwargs: step_calls.append(1))

    def during_pass():
        assert len(optimizer.state) == 0
        assert optimizer.param_groups[0]['lr'] == 3e-4

    retry_model.observer = during_pass
    scaler = torch.amp.GradScaler('cpu', enabled=True, init_scale=8., growth_interval=2000)
    torch.manual_seed(680002)
    values = update_triple(retry_model, optimizer, scaler, triples, .35/720,
                           GradientLayout(retry_model), True, update_lr(0), growth_successes=1999)
    assert values['finite_gradient'] and values['finite_loss'] and values['retries'] == 1
    assert values['retry_attempts'][0]['pass_name'] == 'perturbed'
    assert values['retry_attempts'][0]['task'] == 0 and values['scale'] == 4.
    assert values['scale_growth_successes'] == 1 and len(step_calls) == 1
    assert len(retry_model.masks) == 20
    masks = retry_model.masks
    assert all(torch.equal(masks[i], masks[6+i]) for i in range(2))
    assert all(torch.equal(masks[i], masks[8+i]) for i in range(6))
    assert all(torch.equal(masks[8+i], masks[14+i]) for i in range(6))
    assert torch.equal(retry_model.weights[0], retry_model.weights[8])
    assert all(torch.equal(retry_model.rows[i], retry_model.rows[8+i]) for i in range(6))
    assert all(torch.equal(retry_model.rows[8+i], retry_model.rows[14+i]) for i in range(6))
    assert math.isclose(values['effective_perturbation_norm'], .05, abs_tol=2e-7)
    assert not values['first_gradient_clipped'] and values['perturbation_restored_exactly']
    assert values['successful_view_rows'] == 48 and values['attempted_view_rows'] == 80
    assert retry_model.unused.grad is None and retry_model.unused not in optimizer.state
    assert torch.equal(retry_model.unused, original['unused'])
    assert all(int(state['step']) == 1 for state in optimizer.state.values())
    assert optimizer.param_groups[0]['lr'] == update_lr(0)

    exception_model = ToySelector(fail_on=8, failure='exception')
    exception_optimizer = make_optimizer(exception_model)
    exception_state = cpu_state(exception_model)
    exception_rng = capture_rng()
    try:
        update_triple(exception_model, exception_optimizer, torch.amp.GradScaler('cpu', enabled=False),
            triples, .35/720, GradientLayout(exception_model), True, update_lr(0))
        raise AssertionError('Injected exception must propagate after restoration')
    except RuntimeError as error:
        assert str(error) == 'injected second-pass exception'
    assert state_hash(cpu_state(exception_model)) == state_hash(exception_state)
    assert state_hash(capture_rng()) == state_hash(exception_rng)
    assert len(exception_optimizer.state) == 0
    assert all(p.grad is None for p in exception_model.parameters())

    geometry = dict(mean=[0.]*len(FIELDS), std=[1.]*len(FIELDS))
    terminal_selector, params = initialize(geometry, seed=2)
    with torch.no_grad():
        for p in terminal_selector.score_head.parameters():
            p.fill_(123.)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(670003)
        fresh = cpu_state(score_head(params))
    terminal = dict(model=cpu_state(terminal_selector), args=dict(model_params=params, arm='B'),
        stage=1, epoch=40, parameter_mode='averaged x', selection='fixed terminal',
        successful_updates=9600, presentations=7200000)
    actual_params, initial, initialization = starting_state(terminal, fresh, state_hash(fresh))
    arms = {arm: make_selector(actual_params, initial) for arm in ('A', 'B')}
    assert state_hash(arms['A'].state_dict()) == state_hash(arms['B'].state_dict())
    for model in arms.values():
        assert not hasattr(model, 'projector') and all(p.requires_grad for p in model.parameters())
        assert state_hash(model.score_head.state_dict()) == state_hash(fresh)
        assert not torch.equal(model.score_head[0].weight, terminal_selector.score_head[0].weight)
        for key, value in model.state_dict().items():
            if not key.startswith('score_head.'):
                assert torch.equal(value, terminal['model'][key])
        assert len(make_optimizer(model).state) == 0
    assert math.isclose(update_lr(0), 3e-4/720, rel_tol=1e-15)
    assert update_lr(719) == update_lr(9599) == 3e-4
    return dict(passed=True, focused_cpu_checks=1, synthetic_fixtures_only=True, gpu_used=False,
        global_sam_radius=.05, effective_perturbation_norm=effective,
        whole_update_retry_effective_perturbation_norm=values['effective_perturbation_norm'],
        exact_weight_copy_restoration=True, second_pass_identical_dropout_masks_and_rows=True,
        injected_second_pass_nonfinite_retry=True, failed_attempt_optimizer_lr_state_unchanged=True,
        successful_base_optimizer_steps=len(step_calls), scale_backoff_and_growth_reset=True,
        exception_restores_weights_rng_and_clears_gradients=True, unused_grad_none_and_unchanged=True,
        target_isolated_forward=True, explicit_identical_complete_initialization=True,
        inherited_trunk_only=True, entire_fresh_head_copied=True, no_projector=True,
        optimizer_and_scaler_fresh=True, first_lr=update_lr(0), initialization=initialization,
        test_read=False, training_pilot=False)
