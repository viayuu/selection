"""R69: eighteen complete native selectors routed only by known problem ID."""

import copy
import operator

from torch import nn

from code.unified_selector.registry import PROBLEMS
from .r68_model import (INPUT_KEYS, OPTIMIZER_CONFIG, PARAMETER_MODE, cpu_state,
                        make_optimizer, make_selector, model_inputs, update_lr)


class TaskSpecificSelector(nn.Module):
    def __init__(self, branches):
        super().__init__()
        if len(branches) != len(PROBLEMS) or len(PROBLEMS) != 18:
            raise ValueError('R69 requires eighteen complete selectors in canonical problem-ID order')
        self.branches = nn.ModuleList(branches)
        self.problem_names = tuple(PROBLEMS)

    def branch_index(self, problem_id):
        index = operator.index(problem_id)
        if isinstance(problem_id, bool) or not 0 <= index < len(self.branches):
            raise ValueError('R69 routing requires an original known problem ID')
        return index

    def forward(self, batch):
        clean = model_inputs(batch)
        return self.branches[self.branch_index(clean['problem_id'])](clean)


def make_bank(params, initial, device='cpu'):
    first = make_selector(params, initial)
    bank = TaskSpecificSelector([first] + [copy.deepcopy(first) for _ in range(17)])
    bank.requires_grad_(True)
    return bank.to(device)
