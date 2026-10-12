"""One focused synthetic R69 check; no dataset, GPU, reference inference or pilot."""

import copy
import io
import math

import torch
from torch import nn

from .local_geometry import FIELDS
from .multitask_probe import state_hash
from .r64_model import classification_objective, initialize
from .r67_model import score_head
from .r68_model import starting_state
from .r69_gradients import GradientLayout, update_triple
from .r69_model import INPUT_KEYS, TaskSpecificSelector, cpu_state, make_bank, make_optimizer, update_lr
from .training_monitor import capture_rng, restore_rng


class ToySelector(nn.Module):
    def __init__(self, fail_on=None):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4)/20)
        self.bias = nn.Parameter(torch.zeros(4))
        self.unused = nn.Parameter(torch.tensor([3.]))
        self.dropout = nn.Dropout(.25)
        self.fail_on, self.observer = fail_on, None
        self.masks, self.rows, self.pools = [], [], []

    def forward(self, batch):
        assert set(batch) <= INPUT_KEYS and 'ind' not in batch and 'costs' not in batch
        if self.observer is not None:
            self.observer()
        mask = self.dropout(torch.ones_like(batch['node']))
        self.masks.append(mask.detach().clone())
        self.rows.append(batch['node'].detach().clone())
        self.pools.append(batch['pool_ids'].detach().clone())
        logits = (batch['node']*mask)@self.weight+self.bias
        if len(self.masks) == self.fail_on:
            logits = logits*float('nan')
        return dict(logits=logits.float(), solver_mask=batch['solver_mask'])


def focused_check():
    torch.set_num_threads(1)
    torch.manual_seed(690001)
    fixture = dict(node=torch.tensor([[.2, .5, .8], [.7, .4, .1], [.9, .3, .6], [.1, .8, .4]]),
        solver_mask=torch.ones(4, 4, dtype=torch.bool), pool_ids=torch.tensor([7, 2, 9, 1]),
        ind=torch.tensor([0, 1, 2, 1]), costs=torch.full((4, 4), float('nan')),
        performance_target='forbidden', routes='forbidden')
    triples = [dict(fixture, problem_id=index) for index in range(3)]
    toy = TaskSpecificSelector([ToySelector() for _ in range(18)])
    toy.eval()
    for index in range(18):
        before = [len(branch.masks) for branch in toy.branches]
        first = toy(dict(fixture, problem_id=index))['logits']
        poisoned = dict(fixture, problem_id=index, ind=fixture['ind'].roll(1),
                        costs=torch.full((4, 4), float('inf')), performance_target=object())
        second = toy(poisoned)['logits']
        assert torch.equal(first, second)
        assert [len(branch.masks)-count for branch, count in zip(toy.branches, before)] == [
            2 if branch_id == index else 0 for branch_id in range(18)]
        assert torch.equal(toy.branches[index].pools[-1], fixture['pool_ids'])
    toy.train()
    reference = copy.deepcopy(toy)
    torch.manual_seed(690002)
    rng = capture_rng()
    expected = {}
    for batch in triples:
        reference.zero_grad(set_to_none=True)
        loss, _ = classification_objective(reference(batch), reference(batch), batch['ind'], .35/720)
        loss.backward()
        branch_id = batch['problem_id']
        for name, parameter in reference.branches[branch_id].named_parameters():
            if parameter.grad is not None:
                expected[f'branches.{branch_id}.{name}'] = parameter.grad.detach().clone()/3
    expected_norm = torch.cat([value.reshape(-1) for value in expected.values()]).norm()
    assert float(expected_norm) < 1.
    restore_rng(rng)
    optimizer = make_optimizer(toy)
    original = cpu_state(toy)
    step_calls = []

    def check_divided_gradient(opt, args, kwargs):
        for name, parameter in toy.named_parameters():
            if name in expected:
                torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-6, atol=2e-8)
            else:
                assert parameter.grad is None
        step_calls.append(1)

    hook = optimizer.register_step_pre_hook(check_divided_gradient)
    values = update_triple(toy, optimizer, torch.amp.GradScaler('cpu', enabled=False), triples,
                           .35/720, GradientLayout(toy), update_lr(0))
    hook.remove()
    assert len(step_calls) == 1 and values['global_clip_calls'] == 1
    assert math.isclose(values['grad_norm'], float(expected_norm), rel_tol=2e-6)
    for index, branch in enumerate(toy.branches):
        if index >= 3:
            assert state_hash(branch.state_dict()) == state_hash({key.removeprefix(f'branches.{index}.'): value
                for key, value in original.items() if key.startswith(f'branches.{index}.')})
            assert all(parameter.grad is None and parameter not in optimizer.state for parameter in branch.parameters())
        assert branch.unused.grad is None and branch.unused not in optimizer.state
    old_weights = [cpu_state(branch) for branch in toy.branches[:3]]
    old_states = {name: copy.deepcopy(optimizer.state[parameter]) for name, parameter in toy.named_parameters()
                  if parameter in optimizer.state}
    update_triple(toy, optimizer, torch.amp.GradScaler('cpu', enabled=False),
        [dict(fixture, problem_id=index) for index in range(3, 6)], .35*2/720,
        GradientLayout(toy), update_lr(1))
    for index in range(3):
        assert state_hash(toy.branches[index].state_dict()) == state_hash(old_weights[index])
        assert all(parameter.grad is None for parameter in toy.branches[index].parameters())
    for name, parameter in toy.named_parameters():
        if name in old_states:
            assert state_hash(optimizer.state[parameter]) == state_hash(old_states[name])
    assert all(int(state['step']) == 1 for state in optimizer.state.values())

    retry_model = TaskSpecificSelector([ToySelector(fail_on=2 if index == 2 else None) for index in range(18)])
    retry_original = cpu_state(retry_model)
    retry_optimizer = make_optimizer(retry_model)
    retry_steps = []

    def during_pass():
        assert not retry_optimizer.state
        assert retry_optimizer.param_groups[0]['lr'] == 3e-4
        assert state_hash(retry_model.state_dict()) == state_hash(retry_original)

    for branch in retry_model.branches:
        branch.observer = during_pass
    retry_optimizer.register_step_post_hook(lambda opt, args, kwargs: retry_steps.append(1))
    scaler = torch.amp.GradScaler('cpu', enabled=True, init_scale=8., growth_interval=2000)
    torch.manual_seed(690003)
    retried = update_triple(retry_model, retry_optimizer, scaler, triples, .35/720,
        GradientLayout(retry_model), update_lr(0), growth_successes=1999)
    assert retried['finite_gradient'] and retried['finite_loss'] and retried['retries'] == 1
    assert retried['retry_attempts'][0]['task'] == 2 and retried['scale'] == 4.
    assert retried['scale_growth_successes'] == 1 and len(retry_steps) == 1
    assert retried['successful_view_rows'] == 24 and retried['attempted_view_rows'] == 48
    for branch in retry_model.branches[:3]:
        assert len(branch.masks) == 4
        assert all(torch.equal(branch.masks[index], branch.masks[index+2]) for index in range(2))
        assert all(torch.equal(branch.rows[index], branch.rows[index+2]) for index in range(2))
    assert all(int(state['step']) == 1 for state in retry_optimizer.state.values())
    assert all(parameter.grad is None and parameter not in retry_optimizer.state
               for branch in retry_model.branches[3:] for parameter in branch.parameters())

    geometry = dict(mean=[0.]*len(FIELDS), std=[1.]*len(FIELDS))
    terminal_selector, params = initialize(geometry, seed=2)
    with torch.no_grad():
        for parameter in terminal_selector.score_head.parameters():
            parameter.fill_(123.)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(670003)
        fresh = cpu_state(score_head(params))
    terminal = dict(model=cpu_state(terminal_selector), args=dict(model_params=params, arm='B'),
        stage=1, epoch=40, parameter_mode='averaged x', selection='fixed terminal',
        successful_updates=9600, presentations=7200000)
    actual_params, initial, initialization = starting_state(terminal, fresh, state_hash(fresh))
    bank = make_bank(actual_params, initial)
    expected_hash = state_hash(initial)
    assert all(state_hash(branch.state_dict()) == expected_hash for branch in bank.branches)
    assert len({next(branch.parameters()).data_ptr() for branch in bank.branches}) == 18
    assert all(parameter.requires_grad for parameter in bank.parameters())
    assert all(state_hash(branch.score_head.state_dict()) == state_hash(fresh) for branch in bank.branches)
    assert not hasattr(bank, 'projector') and not make_optimizer(bank).state
    parameter_count = sum(parameter.numel() for parameter in bank.parameters())
    assert parameter_count == 33173154 and parameter_count//18 == 1842953
    with torch.no_grad():
        bank.branches[17].score_head[-1].bias.add_(.25)
    bundle = io.BytesIO()
    torch.save(dict(model=cpu_state(bank), epoch=1, global_epoch=41, problem_names=bank.problem_names), bundle)
    bundle.seek(0)
    loaded = torch.load(bundle, map_location='cpu', weights_only=False)
    restored = make_bank(actual_params, initial)
    restored.load_state_dict(loaded['model'], strict=True)
    assert state_hash(restored.state_dict()) == state_hash(bank.state_dict())
    assert all(any(key.startswith(f'branches.{index}.') for key in loaded['model']) for index in range(18))
    assert loaded['epoch'] == 1 and loaded['global_epoch'] == 41
    assert math.isclose(update_lr(0), 3e-4/720, rel_tol=1e-15)
    assert update_lr(719) == update_lr(9599) == 3e-4
    return dict(passed=True, focused_cpu_checks=1, synthetic_fixtures_only=True, gpu_used=False,
        identical_initialization_all18=True, independent_complete_parameter_storage=True,
        all_parameters_trainable=True, inherited_trunk_only=True, entire_fresh_head_copied=True,
        routing_all18_by_known_problem_id=True, target_isolation=True, pool_order_preserved=True,
        gradient_divided_by3_matches_mean_objective=True, one_global_clip_and_step=True,
        inactive_weights_gradients_adam_moments_steps_and_decay_unchanged=True,
        previously_active_branches_immutable_when_inactive=True,
        injected_third_task_whole_update_amp_retry=True, identical_retry_rows_and_dropout=True,
        failed_attempt_no_optimizer_lr_state_advancement=True, scale_backoff_and_growth_reset=True,
        injected_retries=retried['retries'], full18_common_epoch_bundle_save_load=True,
        shared_parameters=parameter_count//18, bank_parameters=parameter_count,
        initialization=initialization, first_lr=update_lr(0), test_read=False, training_pilot=False)
