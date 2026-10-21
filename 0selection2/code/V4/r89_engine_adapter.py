"""Private immutable-engine optimizer injection without shared Torch mutation."""

import importlib.util
import itertools
import sys
from pathlib import Path

import torch

from . import r84_experiment
from .r76_experiment import file_hash, read_json, write_json


_SERIAL = itertools.count()


def _json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(name): _json_value(item) for name, item in value.items()}
    return str(value)


class _OptimizerFacade:
    def __init__(self, adapter):
        self._adapter = adapter

    def __getattr__(self, name):
        return getattr(torch.optim, name)

    def AdamW(self, parameters, lr, weight_decay, **kwargs):
        if kwargs:
            raise ValueError('The frozen R84 constructor contract unexpectedly changed')
        return self._adapter.construct(parameters, lr, weight_decay)


class _TorchFacade:
    def __init__(self, adapter):
        self.optim = _OptimizerFacade(adapter)

    def __getattr__(self, name):
        return getattr(torch, name)


class PrivateEngineAdapter:
    """One private engine, one exact bound selector, one actual optimizer."""

    def __init__(self, arm, run_dir=None):
        if arm not in ('A_adamw', 'B_muon_hybrid'):
            raise ValueError('Unknown R89 optimizer arm')
        self.arm, self.run_dir = arm, None if run_dir is None else Path(run_dir)
        self.selector, self.optimizer, self.successful_steps = None, None, 0
        self._real_adamw = torch.optim.AdamW
        name = 'code.V4._r89_private_engine_' + str(next(_SERIAL))
        path = Path(r84_experiment.__file__).resolve()
        specification = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        try:
            specification.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        module.torch = _TorchFacade(self)
        self.engine, self.engine_source = module, path
        self._assert_isolated()

    def _assert_isolated(self):
        if (torch.optim.AdamW is not self._real_adamw or r84_experiment.torch is not torch
                or r84_experiment.torch.optim.AdamW is not self._real_adamw
                or self.engine.torch is torch):
            raise RuntimeError('R89 optimizer injection must remain private to its engine instance')

    def bind(self, selector):
        if selector is not self.selector:
            self.selector, self.optimizer, self.successful_steps = selector, None, 0
        self._assert_isolated()

    def prepare_optimizer(self, lr, weight_decay):
        if self.selector is None:
            raise RuntimeError('Bind the actual R89 selector before optimizer construction')
        if self.optimizer is None:
            from .r89_optim import build_optimizer
            self.optimizer = build_optimizer(self.selector, self.arm, lr=lr, weight_decay=weight_decay)
            if self.arm == 'A_adamw' and type(self.optimizer) is not self._real_adamw:
                raise TypeError('R89A must construct the actual original Torch AdamW')
            if not isinstance(self.optimizer, torch.optim.Optimizer):
                raise TypeError('R89 must expose the real optimizer interface for AMP/checkpointing')
            grouped = [id(value) for group in self.optimizer.param_groups for value in group['params']]
            expected = [id(value) for value in self.selector.parameters()]
            if len(grouped) != len(set(grouped)) or set(grouped) != set(expected):
                raise ValueError('Every model parameter must belong to exactly one actual optimizer group')
            if any(group['lr'] != lr or group['weight_decay'] != weight_decay
                   for group in self.optimizer.param_groups):
                raise ValueError('All groups must use the exact same engine LR and weight decay')
            self.optimizer.register_step_post_hook(self._after_step)
        self._assert_isolated()
        return self.optimizer

    def construct(self, parameters, lr, weight_decay):
        if self.selector is None or {id(value) for value in parameters} != {
                id(value) for value in self.selector.parameters()}:
            raise ValueError('Private optimizer constructor received another selector\'s parameters')
        optimizer = self.prepare_optimizer(lr, weight_decay)
        if any(group['lr'] != lr or group['weight_decay'] != weight_decay for group in optimizer.param_groups):
            raise ValueError('Private optimizer construction controls differ from the locked plan')
        return optimizer

    def _after_step(self, optimizer, args, kwargs):
        self.successful_steps += 1

    def static_receipt(self):
        if self.optimizer is None:
            raise RuntimeError('Actual optimizer has not been constructed')
        names = {id(value): name for name, value in self.selector.named_parameters()}
        groups = []
        for number, group in enumerate(self.optimizer.param_groups):
            groups.append(dict(index=number, name=group.get('name', group.get('algorithm', group.get('group_name', f'group{number}'))),
                settings={key: _json_value(value) for key, value in group.items() if key != 'params'},
                parameter_names=[names[id(value)] for value in group['params']],
                parameter_shapes=[list(value.shape) for value in group['params']],
                parameter_count=sum(value.numel() for value in group['params'])))
        optimizer_source = Path(__file__).with_name('r89_optim.py')
        from .r89_optim import optimizer_manifest
        return dict(arm=self.arm, optimizer_class=type(self.optimizer).__module__ + '.' + type(self.optimizer).__qualname__,
            groups=groups, implementation_source=str(optimizer_source.resolve()),
            implementation_sha256=file_hash(optimizer_source),
            optimizer_manifest=optimizer_manifest(self.optimizer),
            private_engine_source=str(self.engine_source), private_engine_sha256=file_hash(self.engine_source),
            shared_torch_adamw_unchanged=torch.optim.AdamW is self._real_adamw,
            shared_r84_engine_unchanged=r84_experiment.torch is torch,
            step_boundary='one actual optimizer.step after complete logical gradient unscale/clip',
            model_forward_dtype='original FP16 autocast', new_neural_parameters=0)

    def checkpoint_receipt(self):
        receipt = self.static_receipt()
        histograms = []
        for index, group in enumerate(self.optimizer.param_groups):
            values = []
            for parameter in group['params']:
                step = self.optimizer.state.get(parameter, {}).get('step')
                if step is not None:
                    values.append(int(step.item()) if torch.is_tensor(step) else int(step))
            histograms.append(dict(index=index, initialized_parameter_states=len(values),
                minimum=min(values) if values else None, maximum=max(values) if values else None))
        return dict(receipt, successful_optimizer_steps=self.successful_steps, parameter_state_steps=histograms)

    def write_initial_receipt(self):
        receipt = dict(self.static_receipt(), successful_optimizer_steps=0)
        if self.run_dir is not None:
            path = self.run_dir / 'optimizer_initialization.json'
            if path.exists():
                if read_json(path) != receipt:
                    raise ValueError('Existing optimizer initialization receipt differs from the locked constructor')
            else:
                write_json(path, receipt)
        return receipt

    def restore_step_receipt(self, checkpoint):
        recorded = checkpoint.get('optimizer_provenance')
        if recorded is None or recorded.get('arm') != self.arm:
            raise ValueError('R89 resume requires optimizer provenance from the same arm')
        if (recorded.get('optimizer_class') != self.static_receipt()['optimizer_class']
                or recorded.get('implementation_sha256') != self.static_receipt()['implementation_sha256']
                or recorded.get('successful_optimizer_steps') != checkpoint['successful_updates']):
            raise ValueError('R89 optimizer identity or successful-update boundary changed on resume')
        self.successful_steps = checkpoint['successful_updates']

    def step_metrics(self):
        from .r89_optim import optimizer_metrics
        metrics = optimizer_metrics(self.optimizer)
        if not isinstance(metrics, dict) or 'muon_step_applied' not in metrics:
            raise RuntimeError('Actual hybrid optimizer must expose per-step Muon application/direction statistics')
        result = dict(metrics, optimizer_step_applied=1.)
        if not any('rms' in name.lower() for name in result):
            raise RuntimeError('Actual hybrid optimizer must expose its measured Muon direction RMS')
        return result


class TrainingHooks:
    """Explicit engine callbacks with only optimizer binding/provenance added."""

    def __init__(self, integration, adapter):
        self.integration, self.adapter = integration, adapter

    def __getattr__(self, name):
        return getattr(self.integration, name)

    def initialize_shared(self, initial, geometry, arm, device='cpu'):
        selector, params, receipt = self.integration.initialize_shared(initial, geometry, arm, device)
        self.adapter.bind(selector)
        self.adapter.prepare_optimizer(self.integration.PEAK_LR, 1e-4)
        receipt['optimizer_initialization'] = self.adapter.write_initial_receipt()
        return selector, params, receipt

    def load_checkpoint(self, path, device='cpu'):
        selector, state = self.integration.load_checkpoint(path, device)
        self.adapter.bind(selector)
        self.adapter.prepare_optimizer(self.integration.PEAK_LR, 1e-4)
        return selector, state

    def protocol(self, *args, **kwargs):
        return self.integration.protocol(*args, optimizer_receipt=self.adapter.static_receipt(), **kwargs)

    def checkpoint_extras(self, objective):
        result = self.integration.checkpoint_extras(objective)
        return dict(result, optimizer_provenance=self.adapter.checkpoint_receipt())

    def restore_objective(self, objective, checkpoint):
        self.integration.restore_objective(objective, checkpoint)
        self.adapter.restore_step_receipt(checkpoint)

    def update_batch(self, *args, **kwargs):
        parts, retries, extra = self.integration.update_batch(*args, **kwargs)
        metrics = dict(extra, **self.adapter.step_metrics())
        return parts, retries, metrics
