from EasyNCO.neural_solvers.pipeline import ARInitialization
import torch.nn as nn

class ICAMInitialization(ARInitialization):
    """
    Initialization class for ICAM model
    """
    def __init__(self, policy: nn.Module):
        super().__init__(policy.model)
