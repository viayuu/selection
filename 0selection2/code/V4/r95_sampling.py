"""R95 matched sampling; proposal production and weighted objectives live upstream.

The provider must return the arm's own full-TRAIN proposal at ``source_step``;
this module never loads labels, evaluates a model, or consults held-out data.
Sampling metadata is supervision, not model input: the caller must remove it
from the forward batch separately from applying the weighted objective.
"""

from dataclasses import dataclass
from operator import index

import numpy as np
import torch

from .r77_experiment import ShuffleStream


BATCH_SIZE = 1024
REFRESH_UPDATES = 7200
WARM_DRAWS = 400 * BATCH_SIZE
RNG_OFFSET = 9500003


@dataclass(frozen=True)
class WeightedRows:
    indices: torch.Tensor
    weights: torch.Tensor
    source_step: int

    def __len__(self):
        return len(self.indices)


def make_proposal(scores, alpha=.5):
    """Mix uniform mass with nonnegative scores; ``alpha`` is the score fraction.

    ``0 <= alpha < 1`` preserves full support. All-zero scores give uniform
    probabilities. Scaling before summation avoids overflow without modifying
    the input, including read-only proposal-score tables.
    """
    scores = np.asarray(scores, dtype=np.float64)
    alpha = float(alpha)
    if (scores.ndim != 1 or not scores.size or not np.isfinite(scores).all()
            or np.any(scores < 0)):
        raise ValueError('Proposal scores must be a nonempty finite nonnegative vector')
    if not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError('The score fraction must satisfy 0 <= alpha < 1')
    maximum = scores.max()
    if maximum == 0 or alpha == 0:
        return np.full(scores.size, 1. / scores.size, dtype=np.float64)
    scaled = scores / maximum
    return (1. - alpha) / scores.size + alpha * scaled / scaled.sum(dtype=np.float64)


class AdaptiveStream:
    """Original R77 warmup, then coupled IID replacement in exact logical batches.

    ``seed`` is the caller's per-task seed, identical across arms. Only the
    replacement stream uses PCG64; warmup never consumes its uniforms. Resume
    requires just the draw cursor and the persisted proposal for its source
    step, not a replay of earlier proposals or batches.
    """

    def __init__(self, size, seed, draws=0, *, arm, problem, proposal_provider):
        self.size, self.seed, self.draws = index(size), index(seed), index(draws)
        if self.size <= 0 or self.seed < 0 or self.draws < 0:
            raise ValueError('Use a positive population and nonnegative seed and cursor')
        if arm not in ('A', 'B'):
            raise ValueError('R95 arms are A (uniform) and B (importance)')
        if arm == 'B' and not callable(proposal_provider):
            raise TypeError('Arm B requires a TRAIN proposal provider')
        self.arm, self.problem = arm, problem
        self.proposal_provider = proposal_provider
        self._shuffle = ShuffleStream(self.size, self.seed, draws=min(self.draws, WARM_DRAWS))
        self._rng = np.random.Generator(np.random.PCG64(self.seed + RNG_OFFSET))
        self._rng.bit_generator.advance(max(0, self.draws - WARM_DRAWS))
        self._source_step, self._q, self._cdf = None, None, None

    def _load_proposal(self, source_step):
        if self._source_step == source_step:
            return
        if self.arm == 'A':
            q = np.full(self.size, 1. / self.size, dtype=np.float64)
        else:
            q = np.array(self.proposal_provider(self.problem, source_step),
                dtype=np.float64, copy=True)
        if (q.shape != (self.size,) or not np.isfinite(q).all() or np.any(q <= 0)
                or not np.isclose(q.sum(dtype=np.float64), 1., rtol=1e-12, atol=1e-12)):
            raise ValueError('A proposal must contain one positive normalized probability per TRAIN row')
        cdf = np.cumsum(q, dtype=np.float64)
        # Close only the floating-point CDF endpoint, never renormalize q or weights.
        cdf[-1] = 1.
        q.setflags(write=False)
        cdf.setflags(write=False)
        self._q, self._cdf, self._source_step = q, cdf, source_step

    def take(self, count=BATCH_SIZE):
        count = index(count)
        if count != BATCH_SIZE:
            raise ValueError('R95 requires exactly 1024 rows per logical batch')
        if self.draws % WARM_DRAWS + count > WARM_DRAWS:
            raise ValueError('A logical batch cannot cross warmup or proposal-refresh boundaries')
        if self.draws < WARM_DRAWS:
            rows = self._shuffle.take(count)
            self.draws += count
            return WeightedRows(rows, torch.ones(count, dtype=torch.float32), 0)
        source_step = (self.draws // WARM_DRAWS) * REFRESH_UPDATES
        self._load_proposal(source_step)
        rows = np.searchsorted(self._cdf, self._rng.random(count), side='right')
        weights = (1. / (self.size * self._q[rows])).astype(np.float32)
        self.draws += count
        return WeightedRows(torch.from_numpy(rows.astype(np.int64, copy=False)),
            torch.from_numpy(weights), source_step)


class WeightedCache:
    """Pass through original gathers, attaching supervision only for weighted rows."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def batch(self, rows, device='cpu'):
        if not isinstance(rows, WeightedRows):
            return self.base.batch(rows, device=device)
        batch = dict(self.base.batch(rows.indices, device=device))
        batch['sampling_weight'] = rows.weights.to(device=device, dtype=torch.float32)
        batch['sampling_row'] = rows.indices.to(device=device, dtype=torch.long)
        return batch

    def close(self):
        return self.base.close()
