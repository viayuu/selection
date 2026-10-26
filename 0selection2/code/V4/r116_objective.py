"""One-forward R116 champion CE plus full-ranking likelihood, shared by both arms."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .pairwise_objective import inputs_only as historical_inputs_only
from .r76_cache import ScenarioCache, _check_split, _open_arrays, _read, _row_indices, _validate_record


PART_KEYS = ('loss', 'champion_nll', 'ranking_nll', 'grad_norm')
LOSS_RECIPE = dict(champion_nll_weight=.5, ranking_nll_weight=.5,
    classification='native CE on exact champion log probabilities',
    champion_target='native ind', ranking_target='original FP64 cost, native winner first in minimum tie, global ID ties',
    ranking_reduction='per instance mean over K-1 eliminations, then mean over real rows',
    forward_passes=1, rdrop=False, comparison=False, relative_risk=False, quotas=False)
INPUT_KEYS = frozenset(('node', 'matrix', 'node_mask', 'n', 'node_geom', 'kind', 'problem_id',
    'coord_dist', 'cbits', 'problem_desc', 'd_in', 'pool_ids', 'solver_mask', 'mask'))


def inputs_only(batch):
    # The historical blacklist predates costs_fp64. Allowlisting also excludes future targets.
    return {key: value for key, value in historical_inputs_only(batch).items() if key in INPUT_KEYS}


class FP64Cache(ScenarioCache):
    """Same read-only R84 arrays and gathers, without unused pairwise target preprocessing."""

    def __init__(self, directory):
        from pathlib import Path
        self.directory = Path(directory).expanduser().resolve()
        self._arrays, self._targets, self.raw, self.lengths = {}, {}, {}, None
        self.metadata = record = _read(self.directory / 'metadata.json')
        _check_split(record['split'], record.get('test_lock_root'))
        specs = _validate_record(record, self.directory)
        self.problem, self.split = record['problem'], record['split']
        self.size, self.kind = record['rows'], record['kind']
        self.pool, self.pool_ids = list(record['pool']), list(record['pool_ids'])
        try:
            self._arrays = _open_arrays(self.directory, specs)
            self.lengths = self._arrays['n']
            self.raw = dict(costs=self._arrays['costs'], winner=self._arrays['winner'],
                            pool=self.pool, pool_ids=self.pool_ids)
            if (self.raw['costs'].dtype != np.float64 or self.raw['winner'].dtype != np.int64
                    or int(self.lengths.min()) != record['true_length_min']
                    or int(self.lengths.max()) != record['true_length_max']
                    or not np.array_equal(self.raw['winner'], self.raw['costs'].argmin(1))):
                raise ValueError('R116 requires original FP64 costs, native winners and frozen lengths')
        except BaseException:
            self.close()
            raise

    def batch(self, indices, device='cpu'):
        batch = super().batch(indices, device)
        rows = _row_indices(indices, self.size)
        batch['costs_fp64'] = torch.from_numpy(np.array(self.raw['costs'][rows], copy=True)).to(device)
        return batch


def per_instance_losses(output, batch):
    """Targets are consumed only after the query forward has returned."""
    from .r116_subset import full_ranking_nll, stable_cost_order
    logits, subset = output['logits'].float(), output['subset_logits'].float()
    costs, ind = batch['costs_fp64'], batch['ind']
    if (costs.dtype != torch.float64 or ind.dtype != torch.int64 or logits.shape != costs.shape
            or ind.shape != logits.shape[:1] or output['solver_mask'].shape != logits.shape
            or output['solver_mask'].dtype != torch.bool or not bool(output['solver_mask'].all())
            or not 1 <= logits.shape[1] <= 10
            or subset.shape != (len(ind), 2 ** logits.shape[1] - logits.shape[1] - 1, logits.shape[1])):
        raise ValueError('R116 needs homogeneous all-legal champion/subset outputs and original FP64 targets')
    order = stable_cost_order(costs, ind, batch['pool_ids'])
    champion = F.cross_entropy(logits, ind, reduction='none')
    ranking = full_ranking_nll(subset, order)
    if ranking.shape != champion.shape:
        raise ValueError('Ranking API must return one K-1-averaged NLL per instance')
    return dict(loss=.5 * champion + .5 * ranking, champion_nll=champion, ranking_nll=ranking)


class EliminationObjective(nn.Module):
    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def forward(self, batch, unused_weight=0.):
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            output = self.selector(inputs_only(batch))
        parts = {key: value.mean() for key, value in per_instance_losses(output, batch).items()}
        return parts['loss'], torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])
