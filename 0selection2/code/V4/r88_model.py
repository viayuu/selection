"""R88: a TSP customer execution role in the original zero-padded slot 3.

Runner API::

    model = make_model(params, arm, state_dict=None)
    model, params, receipt = r84.initialize_shared(
        initial, geometry, arm, model_factory=make_model)

Arms: A_canonical_anchor and B_execution_anchor. Every parameter, buffer,
score MLP and maximin operation is the original ScoreDifferenceSelector.
Only TSP's input-only preprocessing changes. Raw TSP nodes contain xy; the
existing InstanceEncoder._pad_node fills slots 2..7 with zeros. Slot 3 is not
read by _coord_stats. In other problems it may denote a depot, so their node
attributes are never modified. The TSP role is a CUSTOMER role, not a depot.

prepare_role_inputs(batch, arm) pads batch['node'] to eight attributes, marks
legal role nodes, removes supervision through inputs_only, and never mutates
batch tensors. All-zero role rows are unmarked. A's explicit binary role must
equal its complete canonical coordinate tie set; B's must be legal one-hot.
Valid explicit roles are preserved after synchronous node relabeling. Unknown
slot values, inconsistent roles, roles on padding or empty TSP graphs are
rejected. No additional cache fields are required.

A marks ALL customers at the lexicographic minimum (x, then y) over valid
customers, including exact coordinate ties. B derives the first valid node,
representing original physical node 0 (slot 0 without leading padding).
For a pure relabeling, call prepare_role_inputs before permuting node, mask
and cached node_geom together. Relabeling an unmarked B input changes the
physical start when its first valid customer changes.

The role enters existing attribute projections, including the raw attribute
part of GeometryResidual.projection. Coordinates, distance bias, physical
geometry/cache, normalization, constraints and non-TSP inputs do not change.
There are no new parameters, objectives or inference calibration rules.
"""

import torch
import torch.nn.functional as F

from code.unified_selector.registry import P2I
from .pairwise_objective import inputs_only
from .pairwise_selector import ScoreDifferenceSelector


ARMS = ('A_canonical_anchor', 'B_execution_anchor')
ROLE_SLOT = 3
ROLE_SPEC = dict(problem='TSP', role_slot=ROLE_SLOT, node_width=8,
                 semantics='customer_anchor_not_depot',
                 canonical_rule='lexicographic_min_x_then_y_among_valid_nodes',
                 exact_coordinate_tie_rule='mark_all_valid_customers_at_minimum_xy',
                 execution_rule='original_physical_node0_first_valid_slot_if_unmarked',
                 explicit_role_rule='A_exact_canonical_tie_set_B_legal_one_hot',
                 non_tsp_rule='unchanged')
SELECTION_HEAD = 'original_score_mlp_maximin'


def prepare_role_inputs(batch, arm):
    """Return input-only TSP attributes with validated role bits, without mutation."""
    if arm not in ARMS:
        raise ValueError('Unknown R88 role arm')
    prepared = inputs_only(batch)
    if prepared['problem_id'] != P2I['TSP']:
        return prepared
    if prepared['kind'] != 'coord':
        raise ValueError('The TSP execution role requires coordinate inputs')
    node, mask = prepared['node'], prepared['node_mask']
    if (node.ndim != 3 or not 2 <= node.shape[-1] <= 8
            or mask.dtype != torch.bool or mask.shape != node.shape[:2]
            or mask.device != node.device):
        raise ValueError('Expected TSP node [B, N, 2..8] and matching boolean node_mask')
    if not mask.any(-1).all() or not torch.isfinite(node[..., :2][mask]).all():
        raise ValueError('Each TSP instance needs at least one finite valid customer')
    node = F.pad(node, (0, 8 - node.shape[-1])) if node.shape[-1] < 8 else node.clone()
    role = node[..., ROLE_SLOT]
    binary = ((role == 0) | (role == 1)).all()
    count = (role == 1).sum(-1)
    if not binary or ((role != 0) & ~mask).any():
        raise ValueError('TSP role slot 3 must be binary and legal; unknown values are invalid')
    if arm == 'A_canonical_anchor':
        x = node[..., 0].masked_fill(~mask, float('inf'))
        leftmost = mask & (x == x.amin(-1, keepdim=True))
        y = node[..., 1].masked_fill(~leftmost, float('inf'))
        derived = (leftmost & (y == y.amin(-1, keepdim=True))).to(role.dtype)
        if ((count > 0) & (role != derived).any(-1)).any():
            raise ValueError('Explicit canonical roles must mark the complete exact minimum-xy tie set')
    else:
        if (count > 1).any():
            raise ValueError('An explicit execution role must be legal one-hot')
        anchor = mask.long().argmax(-1)
        derived = torch.zeros_like(role).scatter_(1, anchor[:, None], 1.)
    node[..., ROLE_SLOT] = torch.where(count[:, None] > 0, role, derived)
    prepared['node'] = node
    return prepared


class RoleInputSelector(ScoreDifferenceSelector):
    def __init__(self, arm, **params):
        params = dict(params)
        if arm not in ARMS:
            raise ValueError('Unknown R88 role arm')
        if (params.get('solver_representation', 'handcrafted') != 'handcrafted'
                or params.get('pair_mode', 'score_difference') != 'score_difference'
                or params.get('selection_head', SELECTION_HEAD) != SELECTION_HEAD
                or any(name.startswith('residual_') for name in params)):
            raise ValueError('R88 requires the original R77 score MLP/maximin without residual heads')
        if params.get('role_arm', arm) != arm or params.get('role_spec', ROLE_SPEC) != ROLE_SPEC:
            raise ValueError('R88 role arm or fixed slot3 input specification changed')
        params.update(selection_head=SELECTION_HEAD, role_arm=arm, role_spec=dict(ROLE_SPEC))
        super().__init__(**params)
        self.role_arm = arm

    def prepare_role_inputs(self, batch):
        return prepare_role_inputs(batch, self.role_arm)

    def forward(self, batch):
        return super().forward(self.prepare_role_inputs(batch))


def make_model(params, arm, state_dict=None):
    """Construct the unchanged backbone, optionally strictly restoring its state."""
    model = RoleInputSelector(arm, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
