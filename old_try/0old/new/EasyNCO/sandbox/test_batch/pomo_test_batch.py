from typing import Union
from EasyNCO.sandbox.test_batch import SandboxTestBatch
from torch import Tensor
from EasyNCO.sandbox.POMO.NEW_py_ver.TSP.POMO.test_n100 import env_params, model_params, tester_params
from EasyNCO.sandbox.POMO.NEW_py_ver.TSP.POMO.TSPTester import TSPTester

class POMOTestBatch(SandboxTestBatch):
    def run(self, batch: Union[Tensor, dict, list]) -> dict:
        tester = TSPTester(env_params=env_params,
                           model_params=model_params,
                           tester_params=tester_params)
        no_aug_score, aug_score = tester._test_one_batch(batch)
        return {"no_aug_score": no_aug_score, "aug_score": aug_score}
