import importlib.util
import sys
from pathlib import Path
from typing import Union

from EasyNCO.sandbox.test_batch import SandboxTestBatch
from torch import Tensor


MODULE_DIR = (
    Path(__file__).resolve().parent.parent
    / "NCO_code"
    / "single_objective"
    / "UDC-Large-scale-CO-master"
    / "UDC"
    / "TSP-AGNN-ICAM"
)


def _import_module(module_name: str, file_name: str):
    module_dir = MODULE_DIR
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)
    file_path = module_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{module_name}' from '{file_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

test_rrc = _import_module("test_rrc", "test-rrc.py")
env_params = test_rrc.env_params
model_params = test_rrc.model_params
model_p_params = test_rrc.model_p_params
tester_params = test_rrc.tester_params
Tester = test_rrc.Tester

class UDCTestBatch(SandboxTestBatch):
    def run(self, batch: Union[Tensor, dict, list]) -> dict:
        tester = Tester(env_params=env_params,
                      model_params=model_params,
                      model_p_params=model_p_params,
                      tester_params=tester_params)
        no_aug_score, aug_score = tester.run(batch)
        return {"no_aug_score": no_aug_score, "aug_score": aug_score}
