import abc

from typing import Optional, Tuple

import torch
from torch import Tensor

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


def get_post_search_strategy(decoding_strategy, probs, **config):
    strategy_registry = {
        "greedy": greedy_search,
        "sampling": sampling_search,
        "beam_search": beam_search,
    }

    if decoding_strategy not in strategy_registry:
        logger.warning(
            f"Unknown decode type '{decoding_strategy}'. Available decode types: {strategy_registry.keys()}. Defaulting to Sampling."
        )
    if decoding_strategy == 'beam_search':
        assert config.get('beam_size', None) is not None, "The beam size must be provided for beam search"


    return strategy_registry.get(decoding_strategy, sampling_search)(probs, **config)

def greedy_search(probs: Tensor)-> Tuple[Tensor, Tensor]:
    # probs: (batch_size, n_start, problem_size)

    selected = torch.argmax(probs, dim=-1)
    prob = probs.gather(dim=-1, index=selected.unsqueeze(dim=-1)).squeeze(dim=-1)

    return selected, prob   # (batch_size, n_start)

def sampling_search(probs: Tensor)-> Tuple[Tensor, Tensor]:
    # probs: (batch_size, n_start, problem_size)

    batch_size, n_start, problem_size = probs.size()
    # Check if sampling went OK, can go wrong due to bug on GPU
    # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
    # to fix pytorch.multinomial bug on selecting 0 probability elements
    while True:
        selected = probs.reshape(batch_size * n_start, -1).multinomial(1) \
            .squeeze(dim=1).reshape(batch_size, n_start)
            # shape: (batch, n_start)

        prob = probs.gather(dim=-1, index=selected.unsqueeze(dim=-1)).squeeze(dim=-1)
        assert prob.size() == (batch_size, n_start), f"prob.size(): {prob.size()}. Expected: {(batch_size, n_start)}"
        # shape: (batch, n_start)

        if (prob != 0).all():
            break

    return selected, prob

def beam_search(probs: Tensor, beam_size: int)-> Tuple[Tensor, Tensor]:
    pass


