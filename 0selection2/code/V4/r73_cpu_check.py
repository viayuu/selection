"""One synthetic CPU invocation for R73's target boundary and update recovery."""

import copy
import time

import torch
import torch.nn.functional as F
from torch import nn

from .r64_model import classification_objective
from .r65_gradients import GradientLayout
from .r67_model import model_inputs
from .r73_data import global_to_local
from .r73_gradients import auxiliary_rng, update_triple
from .train import set_seed
from .training_monitor import capture_rng, restore_rng


class SyntheticSelector(nn.Module):
    def __init__(self, fail_call=None):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 7), nn.GELU(), nn.Dropout(.25), nn.Linear(7, 4))
        self.fail_call, self.calls = fail_call, []

    def forward(self, inputs):
        if set(inputs) - {'node', 'n', 'pool_ids', 'solver_mask', 'problem_id'}:
            raise AssertionError('A target or metadata leaked into the forward')
        self.calls.append(dict(problem_id=inputs['problem_id'], rng=torch.get_rng_state().clone(),
                               node=inputs['node'].clone()))
        scores = self.layers(inputs['node'][:, 0])
        if len(self.calls) == self.fail_call:
            scores = scores * float('nan')
        mask = inputs['solver_mask']
        return dict(logits=scores.masked_fill(~mask, float('-inf')), solver_mask=mask)


class SyntheticScaler:
    """Loss scaling/backoff fixture; this check performs no CPU AMP benchmark."""
    def __init__(self):
        self.value = 8.

    def is_enabled(self):
        return False

    def get_scale(self):
        return self.value

    def scale(self, loss):
        return loss * self.value

    def get_backoff_factor(self):
        return .5

    def update(self, new_scale):
        self.value = new_scale


class RecordingSGD(torch.optim.SGD):
    def __init__(self, model):
        super().__init__(model.parameters(), lr=.01)
        self.steps, self.saved_gradient = 0, None

    def step(self, closure=None):
        self.saved_gradient = torch.cat([p.grad.flatten().clone() for group in self.param_groups for p in group['params']])
        self.steps += 1
        return super().step(closure)


def focused_check():
    started = time.perf_counter()
    torch.set_num_threads(1)
    set_seed(730001)
    ids = torch.tensor([8, 2, 5, 19])
    mask = torch.tensor([[True, True, True, False]]).expand(4, -1).clone()
    targets = torch.tensor([5, 2, 8, 5])
    assert global_to_local(targets, ids, mask).tolist() == [2, 1, 0, 2]
    row_ids = ids[None].expand(4, -1)[:, torch.tensor([2, 0, 3, 1])]
    row_mask = mask[:, torch.tensor([2, 0, 3, 1])]
    assert global_to_local(targets, row_ids, row_mask).tolist() == [0, 3, 1, 0]
    for invalid in (torch.tensor([19, 2, 8, 5]), torch.tensor([21, 2, 8, 5])):
        try:
            global_to_local(invalid, ids, mask)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal or absent teacher target was accepted')
    batches, auxiliary = [], []
    for problem_id in range(3):
        clean = dict(node=torch.randn(4, 1, 4), n=torch.ones(4, dtype=torch.long),
                     pool_ids=ids.clone(), solver_mask=mask.clone(), problem_id=problem_id)
        batches.append(dict(clean, ind=torch.tensor([0, 1, 2, 0]), costs=torch.full((4, 4), float('nan')),
                            teacher_global_id=targets.clone(), source_id=torch.arange(10000, 10004)))
        auxiliary.append((dict(clean, node=torch.randn(4, 1, 4)), targets.clone()))
    original = SyntheticSelector()
    baseline, recovering, reference = copy.deepcopy(original), copy.deepcopy(original), copy.deepcopy(original)
    recovering.fail_call = 9
    weight = .35 * 17 / 720
    set_seed(12345)
    initial_rng = capture_rng()
    with auxiliary_rng(17, 2):
        private_draw = torch.rand(9)
    assert torch.equal(torch.get_rng_state(), initial_rng['torch'])
    with auxiliary_rng(17, 2):
        torch.testing.assert_close(torch.rand(9), private_draw, rtol=0, atol=0)
    reference.zero_grad(set_to_none=True)
    for batch, (aux, target) in zip(batches, auxiliary):
        first, second = reference(model_inputs(batch)), reference(model_inputs(batch))
        native, _ = classification_objective(first, second, batch['ind'], weight)
        (native / 3).backward()
        del first, second, native
        with auxiliary_rng(17, batch['problem_id']):
            output = reference(model_inputs(aux))
            pseudo = F.cross_entropy(output['logits'], global_to_local(target, aux['pool_ids'], output['solver_mask']))
            (pseudo / 3).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    expected = torch.cat([p.grad.flatten() for p in reference.parameters()])
    restore_rng(initial_rng)
    optimizer = RecordingSGD(baseline)
    ordinary = update_triple(baseline, optimizer, SyntheticScaler(), batches,
        lambda task: auxiliary[task], weight, GradientLayout(baseline), 17)
    torch.testing.assert_close(optimizer.saved_gradient, expected, rtol=3e-6, atol=3e-7)
    ordinary_rng = capture_rng()
    restore_rng(initial_rng)
    recovering_optimizer = RecordingSGD(recovering)
    scaler = SyntheticScaler()
    recovered = update_triple(recovering, recovering_optimizer, scaler, batches,
        lambda task: auxiliary[task], weight, GradientLayout(recovering), 17)
    assert recovered['retries'] == 1 and recovering_optimizer.steps == 1 and scaler.get_scale() == 4
    assert len(recovering.calls) == 18
    for first, second in zip(recovering.calls[:9], recovering.calls[9:]):
        assert first['problem_id'] == second['problem_id']
        assert torch.equal(first['rng'], second['rng']) and torch.equal(first['node'], second['node'])
    assert torch.equal(torch.get_rng_state(), ordinary_rng['torch'])
    torch.testing.assert_close(recovering_optimizer.saved_gradient, optimizer.saved_gradient, rtol=0, atol=0)
    for first, second in zip(baseline.parameters(), recovering.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in recovering.parameters())
    assert float(optimizer.saved_gradient.norm()) > 0
    assert ordinary['successful_native_view_rows'] == 24 and ordinary['successful_auxiliary_view_rows'] == 12
    assert recovered['attempted']['native_view_rows'] == 48 and recovered['attempted']['auxiliary_view_rows'] == 24
    try:
        wrong = dict(auxiliary[0][0], problem_id=2)
        wrong_model = copy.deepcopy(original)
        update_triple(wrong_model, RecordingSGD(wrong_model), SyntheticScaler(),
            batches, lambda task: (wrong, targets), weight, GradientLayout(wrong_model), 17)
    except ValueError as error:
        assert 'same task' in str(error)
    else:
        raise AssertionError('Cross-task pseudo-target batch accepted')
    return dict(passed=True, invocations=1, synthetic_only=True, cuda_used=False,
        real_checkpoint_loads=0, real_data_reads=0, seconds=time.perf_counter() - started,
        global_local_legal_alignment=True, column_permutation_alignment=True,
        target_isolation=True, combined_three_task_gradient_matches_reference=True,
        two_native_and_one_auxiliary_paths=True, private_rng_restored=True,
        injected_whole_update_retry=True, retry_rows_and_draws_identical=True,
        optimizer_steps_after_recovery=1, loss_scaling_fixture='synthetic FP32 scale/backoff; no CPU AMP claim')
