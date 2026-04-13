import importlib.util
import sys
from pathlib import Path
from typing import Union
from EasyNCO.sandbox.test_batch import SandboxTestBatch
from torch import Tensor
MODULE_DIR = (
    Path(__file__).resolve().parent.parent / "bq-nco"
)

# Pre-register top-level packages to enable imports across dynamically loaded modules
def _register_package(package_name: str):
    """Register a package in sys.modules by loading its __init__.py"""
    if package_name not in sys.modules:
        package_path = MODULE_DIR / package_name
        init_file = package_path / '__init__.py'
        spec = importlib.util.spec_from_file_location(package_name, init_file)
        if spec is not None and spec.loader is not None:
            package_module = importlib.util.module_from_spec(spec)
            package_module.__path__ = [str(package_path)]
            sys.modules[package_name] = package_module
            spec.loader.exec_module(package_module)

# Add bq-nco directory to sys.path and register top-level packages
module_dir_str = str(MODULE_DIR)
if module_dir_str not in sys.path:
    sys.path.insert(0, module_dir_str)

# Register all top-level packages that modules might import from
for pkg in ['utils', 'learning', 'model', 'data']:
    _register_package(pkg)

def _import_module(module_name: str, file_path_str: str):
    module_dir = MODULE_DIR
    
    # Register parent packages in sys.modules to enable nested imports
    parts = module_name.split('.')
    for i in range(len(parts) - 1):  # Don't include the final module, only packages
        parent_name = '.'.join(parts[:i+1])
        if parent_name not in sys.modules:
            import types
            parent_module = types.ModuleType(parent_name)
            parent_path = module_dir / '/'.join(parts[:i+1])
            parent_module.__path__ = [str(parent_path)]
            parent_module.__file__ = str(parent_path / '__init__.py')
            parent_module.__package__ = parent_name
            sys.modules[parent_name] = parent_module
    
    file_path = module_dir / file_path_str
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{module_name}' from '{file_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Pre-load commonly imported utility and model modules to avoid import errors in nested imports
try:
    _import_module("utils.misc", "utils/misc.py")
    _import_module("utils.sampler", "utils/sampler.py")
    _import_module("utils.chekpointer", "utils/chekpointer.py")
    _import_module("utils.multi_class_loss", "utils/multi_class_loss.py")
    _import_module("model.attention", "model/attention.py")
    _import_module("model.encoder", "model/encoder.py")
    _import_module("model.model", "model/model.py")
    _import_module("utils.exp", "utils/exp.py")
except Exception as e:
    pass  # Some modules may not exist or may have dependencies

test_tsp_module = _import_module("test_tsp", "test_tsp.py")
args = test_tsp_module.args
exp_module = _import_module("utils.exp", "utils/exp.py")
setup_exp = exp_module.setup_exp
data_iterator_module = _import_module("learning.tsp.data_iterator", "learning/tsp/data_iterator.py")
DataIterator = data_iterator_module.DataIterator
traj_learner_module = _import_module("learning.tsp.traj_learner", "learning/tsp/traj_learner.py")
TrajectoryLearner = traj_learner_module.TrajectoryLearner

class BQTestBatch(SandboxTestBatch):
    def run(self, batch: Union[Tensor, dict, list]) -> dict:
        net, module, device, _, checkpointer, _ = setup_exp(args, is_test=True)
        data_iterator = DataIterator(args, batch)
        traj_learner = TrajectoryLearner(args, net, module, device, data_iterator, checkpointer=checkpointer)
        no_aug_score, aug_score = traj_learner.val_test()
        return {"no_aug_score": no_aug_score, "aug_score": aug_score}
