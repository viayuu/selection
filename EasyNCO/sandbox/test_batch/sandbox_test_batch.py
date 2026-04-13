from abc import ABC, abstractmethod
from typing import Union
from torch import Tensor

class SandboxTestBatch(ABC):
    @abstractmethod
    def run(self, batch: Union[Tensor, dict, list]) -> dict:
        """
        This function is used to run the test batch.
        args:
            - batch: The batch of data to be used.
        returns:
            - The result of the test batch.
        """
        raise NotImplementedError("Implement me in subclass!")
