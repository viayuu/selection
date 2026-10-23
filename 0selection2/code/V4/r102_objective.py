"""Unmodified R77 loss; R94's two safe cached-node fields are representation-only.

The inherited forward prepares frozen inputs once and shares them across both
dropout views. No native decoder, auxiliary objective or new loss weight.
"""

from .r94_objective import RDropObjective as _FrozenNodeObjective


class RDropObjective(_FrozenNodeObjective):
    """The same CE, comparison, relative risk and symmetric current-step R-Drop."""


NodeFusionObjective = RDropObjective
