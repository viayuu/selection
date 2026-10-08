"""
Unified Mask System for 107+ VRP Variants

This package contains mask functions for all problem variants.
Each problem has its own mask file defining the specific constraints.
"""

from .mask_registry import mask_registry, register_mask

# Import all mask modules to register them
from . import mask_atsp
from . import mask_tsp
from . import mask_cvrp
from . import mask_acvrp
from . import mask_op
from . import mask_pctsp
from . import mask_pdp
from . import mask_cvrptw
from . import mask_ovrp
from . import mask_vrpb
from . import mask_ovrptw
from . import mask_spctsp
from . import mask_sdvrp
from . import mask_vrpl
from . import mask_vrpbl
from . import mask_vrpltw
from . import mask_vrpbltw
from . import mask_ovrpb
from . import mask_ovrpl
from . import mask_ovrpbl
from . import mask_ovrpltw
from . import mask_ovrpbltw
from . import mask_vrpbtw
from . import mask_ovrpbtw

# Multi-depot VRP with Backhaul-Pickup variants
from . import mask_mdvrpbp
from . import mask_mdovrpbp
from . import mask_mdvrpbpl
from . import mask_mdovrpbptw
from . import mask_mdvrpbpltw
from . import mask_mdvrpbptw
from . import mask_mdovrpbpl
from . import mask_mdovrpbpltw

__all__ = ['mask_registry', 'register_mask']
