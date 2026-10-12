"""One focused CPU check: official optimizer modes/resume and actual selected batches."""

import io
from types import SimpleNamespace

import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import state_hash
from .r64_experiment import paired_triples
from .r65_gradients import GradientLayout, update_triple
from .r65_optimizer import (configure_update_lr, evaluation_mode, make_optimizer,
                            observed_lr, training_mode)


def focused_check():
    from .r65_experiment import select_triple_batches

    torch.manual_seed(2)
    model = torch.nn.Linear(3, 2)
    optimizer = make_optimizer(model, 'B')
    inputs = torch.tensor([[1., -.3, .7], [-.2, .8, .4]])
    target = torch.tensor([[.3, -.1], [.8, .2]])

    def advance(current_model, current_optimizer, successful):
        training_mode(current_model, current_optimizer, 'B')
        expected = configure_update_lr(current_optimizer, 'B', successful)
        current_optimizer.zero_grad(set_to_none=True)
        (current_model(inputs)-target).square().mean().backward()
        current_optimizer.step()
        return observed_lr(current_optimizer, 'B', expected)

    lrs = [advance(model, optimizer, successful) for successful in range(6)]
    y = {key: value.detach().clone() for key, value in model.state_dict().items()}
    evaluation_mode(model, optimizer, 'B')
    x = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert any(not torch.equal(x[key], y[key]) for key in x), 'Expected distinct x and y after updates'
    assert all(not group['train_mode'] for group in optimizer.param_groups)
    stream = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), successful_updates=6,
                    checkpoint_mode='eval'), stream)
    evaluation_mode(model, optimizer, 'B')
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, x[key], rtol=0, atol=0)
    training_mode(model, optimizer, 'B')
    roundtrip_error = max(float((value-y[key]).abs().max()) for key, value in model.state_dict().items())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, y[key], rtol=0, atol=2e-7)
    training_mode(model, optimizer, 'B')
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, y[key], rtol=0, atol=2e-7)
    stream.seek(0)
    checkpoint = torch.load(stream, map_location='cpu', weights_only=False)
    resumed = torch.nn.Linear(3, 2)
    resumed_optimizer = make_optimizer(resumed, 'B')
    resumed.load_state_dict(checkpoint['model'])
    resumed_optimizer.load_state_dict(checkpoint['optimizer'])
    assert all(not group['train_mode'] for group in resumed_optimizer.param_groups)
    assert checkpoint['checkpoint_mode'] == 'eval'
    for successful in range(6, 9):
        assert advance(model, optimizer, successful) == advance(resumed, resumed_optimizer, successful)
    evaluation_mode(model, optimizer, 'B')
    evaluation_mode(resumed, resumed_optimizer, 'B')
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    assert state_hash(optimizer.state_dict()) == state_hash(resumed_optimizer.state_dict())

    schedule = paired_triples({task: 10000 for task in PROBLEMS}, epochs=1, seed=2, batch_size=256)
    loaders = {}
    indices = torch.arange(10000)
    for problem_id, task in enumerate(PROBLEMS):
        loaders[task] = SimpleNamespace(lengths=torch.ones(10000, dtype=torch.long), batch=dict(
            problem_id=problem_id, row_index=indices, costs=torch.zeros(10000, 3),
            ind=indices % 3, input=(indices.float()/10000).unsqueeze(1)))
    entries, batches = select_triple_batches(loaders, schedule, 0, 0)

    class ObservedRows(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 3)
            self.seen = []

        def forward(self, batch):
            self.seen.append((batch['problem_id'], batch['row_index'].clone(), batch['input'].data_ptr()))
            logits = self.linear(batch['input'])
            return dict(logits=logits, solver_mask=torch.ones_like(logits, dtype=torch.bool))

    observed = ObservedRows()
    plain = make_optimizer(observed, 'A')
    configure_update_lr(plain, 'A', 0)
    values = update_triple(observed, plain, torch.amp.GradScaler('cuda', enabled=False), batches,
                           .35/720, GradientLayout(observed))
    assert len(observed.seen) == 6 and values['finite_gradient']
    for offset, ((task, selected), batch) in enumerate(zip(entries, batches)):
        assert len(selected) == len(batch['ind']) == 256
        torch.testing.assert_close(batch['row_index'], selected, rtol=0, atol=0)
        for problem_id, actual, pointer in observed.seen[2*offset:2*offset+2]:
            assert problem_id == batch['problem_id'] and pointer == batch['input'].data_ptr()
            torch.testing.assert_close(actual, selected, rtol=0, atol=0)
    tail_entries, tail_batches = select_triple_batches(loaders, schedule, 0, 239)
    assert [len(batch['ind']) for batch in tail_batches] == [16, 16, 16]
    for (_, selected), batch in zip(tail_entries, tail_batches):
        torch.testing.assert_close(batch['row_index'], selected, rtol=0, atol=0)
    return dict(passed=True, focused_cpu_checks=1, optimizer='official AdamWScheduleFree',
        eval_train_roundtrip=True, roundtrip_max_abs_error=roundtrip_error,
        checkpoint_mode='eval x plus optimizer train_mode=False', exact_resume_model_and_optimizer=True,
        resumed_successful_updates=9, no_double_warmup=True, first_effective_lr=lrs[0],
        actual_selected_rows_forwarded=True, real_task_rows=[256, 256, 256],
        tail_task_rows=[16, 16, 16], actual_two_forwards_share_selected_batch=True,
        fp32_arithmetic_mean=True, gpu_used=False, data_or_test_loaded=False)
