"""The single combined synthetic R74 boundary/objective/recovery invocation."""

import copy
import time

import torch
from torch import nn

from .r74_acquisition import guard_incomplete, quarter_increment
from .r74_gradients import GradientLayout, leaf_rows, update_triple
from .r74_model import AcquisitionPolicy, choose_global
from .training_monitor import capture_rng, restore_rng


class SyntheticObservations:
    def __init__(self, rows):
        self.tokens = {(probe, checkpoint): torch.randn(rows, 3, 403)
                       for probe in (3, 4) for checkpoint in (0, 1)}
        self.calls, self.forbidden = [], set()

    def __call__(self, probe, checkpoint, positions=None):
        key = (probe, checkpoint)
        if key in self.forbidden:
            raise AssertionError('A future or unchosen observation was accessed')
        self.calls.append(key)
        tokens = self.tokens[key]
        if positions is not None:
            tokens = tokens[positions]
        return tokens, torch.ones(tokens.shape[:2], dtype=torch.bool)


class RecordingPolicy(AcquisitionPolicy):
    def __init__(self, fail_call=None):
        super().__init__()
        self.calls, self.fail_call = [], fail_call

    def leaf_logits(self, batch, leaf, positions=None):
        self.calls.append(dict(problem_id=batch['problem_id'], leaf=leaf,
                               rng=torch.get_rng_state().clone()))
        output = super().leaf_logits(batch, leaf, positions)
        if len(self.calls) == self.fail_call:
            return output * float('nan')
        return output


class SyntheticScaler:
    def __init__(self):
        self.value = 8.
    def is_enabled(self):
        return False
    def get_scale(self):
        return self.value
    def scale(self, loss):
        return loss*self.value
    def get_backoff_factor(self):
        return .5
    def update(self, new_scale):
        self.value = new_scale


class RecordingSGD(torch.optim.SGD):
    def __init__(self, model):
        super().__init__(model.parameters(), lr=.01)
        self.steps, self.gradient = 0, None
    def step(self, closure=None):
        self.steps += 1
        self.gradient = torch.cat([parameter.grad.flatten().clone()
                                   for group in self.param_groups for parameter in group['params']])
        return super().step(closure)


def synthetic_batch(problem_id, rows=3):
    return dict(problem_id=problem_id, mvrp=True, h=torch.randn(rows, 128),
        features=torch.randn(rows, 4, 384), base_logits=torch.randn(rows, 4),
        solver_mask=torch.tensor([True, True, False, True])[None].expand(rows, -1),
        pool_ids=torch.tensor([8, 2, 19, 5]), observations=SyntheticObservations(rows),
        winner=torch.tensor([0, 1, 3])[:rows])


def focused_check():
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(740001)
    for n in (4, 8, 12, 20, 50, 75, 100, 500):
        h = quarter_increment(n)
        for vrp in (False, True):
            visited = torch.zeros(2, 3, n+int(vrp), dtype=torch.bool)
            if vrp:
                visited[..., 0] = True
            first = int(vrp)
            visited[..., first:first+1+2*h] = True
            assert guard_incomplete(n, visited, vrp) == 1+2*h <= n//4
            for done, finished in ((True, None), (False, torch.ones(2, 3, dtype=torch.bool))):
                try:
                    guard_incomplete(n, visited, vrp, done, finished)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Completion guard accepted a finished trajectory')
            visited[..., first:first+n//4+1] = True
            try:
                guard_incomplete(n, visited, vrp)
            except ValueError:
                pass
            else:
                raise AssertionError('Quarter boundary accepted an extra required node')
    for n in (1, 2, 3):
        try:
            quarter_increment(n)
        except ValueError:
            pass
        else:
            raise AssertionError('Tiny invalid size accepted')

    batch = synthetic_batch(3)
    policy = AcquisitionPolicy().eval()
    assert torch.count_nonzero(policy.correction.weight) == 0
    assert torch.count_nonzero(policy.root_gate[-1].weight) == 0
    assert torch.count_nonzero(policy.second_gate[-1].weight) == 0
    batch['observations'].forbidden = {(4, 0), (4, 1)}
    with torch.no_grad():
        output = policy(batch)
        assert output['hard_paths'].tolist() == [0, 0, 0]
        torch.testing.assert_close(output['logits'][batch['solver_mask']], batch['base_logits'][batch['solver_mask']])
        original = output['logits'].clone()
        batch['costs'], batch['native_winner'], batch['source_ids'] = torch.full((3, 4), float('nan')), [-9]*3, [-100]*3
        batch['observations'].tokens[(4, 0)].fill_(float('nan'))
        batch['observations'].tokens[(4, 1)].fill_(float('nan'))
        torch.testing.assert_close(policy(batch)['logits'], original)
        # Nonzero TESTFIXTURE gate weights make the isolation assertion nontrivial.
        nn.init.normal_(policy.root_gate[-1].weight, std=.03)
        nn.init.normal_(policy.second_gate[-1].weight, std=.03)
        shallow, mask = batch['observations'](3, 0)
        root_before = policy.root(batch['h'])
        second_before = policy.second(batch['h'], shallow, mask, 0)
        batch['observations'].tokens[(3, 1)].mul_(1000)
        torch.testing.assert_close(policy.root(batch['h']), root_before)
        torch.testing.assert_close(policy.second(batch['h'], shallow, mask, 0), second_before)
    assert set(batch['observations'].calls) <= {(3, 0), (3, 1)}
    tied = torch.tensor([[0., 0., float('nan'), 0.]])
    local, global_id = choose_global(tied, batch['pool_ids'], batch['solver_mask'][:1])
    assert local.item() == 1 and global_id.item() == 2

    a = torch.tensor([.65, .2], requires_grad=True)
    b = torch.tensor([.3, .7], requires_grad=True)
    c = torch.tensor([.8, .4], requires_grad=True)
    weights = torch.stack((a*b, (1-a)*c, a*(1-b)+(1-a)*(1-c)), -1)
    first = torch.randn(2, 3, 4, requires_grad=True)
    second = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False, True]]).expand(2, -1)
    target = torch.tensor([0, 3])
    ce, kl = zip(*(leaf_rows(first[:, leaf], second[:, leaf], target, mask) for leaf in range(3)))
    ce, kl = torch.stack(ce, -1), torch.stack(kl, -1)
    loss = (weights*ce).sum(-1).mean() + .35*(weights.detach()*kl).sum(-1).mean()
    ga, gb, gc, gf, gs = torch.autograd.grad(loss, (a, b, c, first, second))
    expected_a = (b*ce[:, 0]-c*ce[:, 1]+(c-b)*ce[:, 2])/2
    expected_b = a*(ce[:, 0]-ce[:, 2])/2
    expected_c = (1-a)*(ce[:, 1]-ce[:, 2])/2
    for actual, expected in ((ga, expected_a), (gb, expected_b), (gc, expected_c)):
        torch.testing.assert_close(actual, expected)
    assert torch.isfinite(loss) and gf.abs().sum() > 0 and gs.abs().sum() > 0
    assert torch.count_nonzero(gf[:, :, 2]) == 0 and torch.count_nonzero(gs[:, :, 2]) == 0

    initial = RecordingPolicy().train()
    with torch.no_grad():
        nn.init.normal_(initial.correction.weight, std=.03)
        nn.init.normal_(initial.root_gate[-1].weight, std=.01)
        nn.init.normal_(initial.second_gate[-1].weight, std=.01)
    clean, recovered = copy.deepcopy(initial), copy.deepcopy(initial)
    recovered.fail_call = 8
    batches = [synthetic_batch(problem_id) for problem_id in (3, 4, 5)]
    reference_optimizer, recovered_optimizer = RecordingSGD(clean), RecordingSGD(recovered)
    rng = capture_rng()
    reference = update_triple(clean, reference_optimizer, SyntheticScaler(), batches, .1, GradientLayout(clean))
    reference_rng = capture_rng()
    restore_rng(rng)
    recovery = update_triple(recovered, recovered_optimizer, SyntheticScaler(), batches, .1, GradientLayout(recovered))
    assert reference_optimizer.steps == recovered_optimizer.steps == 1
    assert reference['retries'] == 0 and recovery['retries'] == 1 and recovery['scale'] == 4.
    assert len(clean.calls) == 18 and len(recovered.calls) == 26
    for left, right in zip(clean.calls[:8], recovered.calls[:8]):
        assert left['problem_id'] == right['problem_id'] and left['leaf'] == right['leaf']
        assert torch.equal(left['rng'], right['rng'])
    for left, right in zip(clean.calls, recovered.calls[8:]):
        assert left['problem_id'] == right['problem_id'] and left['leaf'] == right['leaf']
        assert torch.equal(left['rng'], right['rng'])
    torch.testing.assert_close(reference_optimizer.gradient, recovered_optimizer.gradient, rtol=1e-5, atol=2e-7)
    for left, right in zip(clean.parameters(), recovered.parameters()):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-7)
    assert torch.equal(torch.get_rng_state(), reference_rng['torch'])
    assert recovery['successful_leaf_view_rows'] == 54
    return dict(passed=True, seconds=time.perf_counter()-started, synthetic_only=True,
        checks=['quarter boundary and zero completion', 'target and future-observation isolation',
            'hard-path access and exact gate/global-ID ties', 'exact expected CE and detached-weight KL gate gradients',
            'both dropout-view gradient paths', 'same-state FP32 mean/clip/one step',
            'injected whole-triple retry preserves rows and ordinary RNG'],
        injected_retries=1, production_initialization_unchanged=True, real_model_or_data_preflight=False,
        solver_executions=0, GPU_operations=0)
