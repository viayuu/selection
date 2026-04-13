from EasyNCO.neural_solvers.methods.matnet.policy import MatNetGLOPPolicy

from EasyNCO.neural_solvers.methods.am.policy import AttentionModelPolicy

from EasyNCO.neural_solvers.backbones.GNN.partition_net import glop_partition_net

import torch.nn as nn


class GLOPPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.problem = kwargs.get("problem")
        self.global_params=kwargs.get("global_params")
        self.k_sparse_cvrp_leq1000=kwargs.get("k_sparse_cvrp_leq1000",100)
        self.k_sparse_cvrp_gt1000=kwargs.get("k_sparse_cvrp_gt1000",200)
        self.k_sparse_pctsp_leq500=kwargs.get("k_sparse_pctsp_leq500",50)
        self.k_sparse_pctsp_leq1000=kwargs.get("k_sparse_pctsp_leq1000",100)
        self.k_sparse_pctsp_gt1000=kwargs.get("k_sparse_pctsp_gt1000",200)
        # Upper model initialization
        if self.problem in ["tsp", "atsp"]:
            self.upper_model = None
        elif self.problem in ["cvrp", "pctsp"]:
            upper_param = kwargs.get("upper", {})
            self.upper_model = glop_partition_net(**upper_param)

        # Lower model initialization with merged sub-models
        self.lower_model = nn.ModuleDict()  # Using ModuleDict to properly register all sub-models

        if self.problem in ["tsp", "cvrp", "pctsp"]:
            lower_param = kwargs.get("lower", {})

            # Create and store all attention model variants
            self.lower_model["sub_20"] = AttentionModelPolicy(**lower_param)
            self.lower_model["sub_50"] = AttentionModelPolicy(**lower_param)
            self.lower_model["sub_100"] = AttentionModelPolicy(**lower_param)

        elif self.problem == "atsp":
            lower_param = kwargs.get("atsp_lower", {})

            # Create and store all MatNet model variants
            self.lower_model["sub_20"] = MatNetGLOPPolicy(**lower_param)
            self.lower_model["sub_50"] = MatNetGLOPPolicy(**lower_param)
            self.lower_model["sub_100"] = MatNetGLOPPolicy(**lower_param)