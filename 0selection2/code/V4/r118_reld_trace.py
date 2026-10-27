"""Observe one original_reld run; the caller must supply a frozen TRAIN item.

Step arrays have axes (step, batch=1, physical POMO column[, native node ID]).
`step` is the pre-action selected_count; trajectory_id/pomo_idx never reorder
columns. start_node is the actual second action, not an assumed arange.
Absent time/length and forced decoder rows contain NaN, with explicit presence
and decoder_valid masks. Only decoder_valid rows are probability observations.
Native masks are observations, not independent feasibility certificates.
"""

import importlib
import math

import numpy as np
import torch


_DTYPES = {torch.bool: np.bool_, torch.int64: np.int64, torch.int32: np.int32,
           torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64}


class _Trace:
    def __init__(self, max_bytes):
        self.max_bytes, self.nbytes = max_bytes, 0
        self.env = self.shape = self.pending = None
        self.rows, self.arrays = [], {}
        self.terminal = False

    def reserve(self, shape, dtype):
        size = math.prod(shape) * np.dtype(dtype).itemsize
        # Reserve a second payload for mask/cast scratch and final stacking.
        if 2 * (self.nbytes + size) > self.max_bytes:
            raise MemoryError("RELD trace exceeds max_bytes (including copy/stack reserve)")
        self.nbytes += size

    def filled(self, shape, value, dtype):
        self.reserve(shape, dtype)
        return np.full(shape, value, dtype=dtype)

    def copy(self, value, shape, dtype=None, transform=None):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Unrecognized native tensor dimensions; expected {shape}")
        dtype = value.dtype if dtype is None else dtype
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported native dtype: {dtype}")
        self.reserve(shape, _DTYPES[dtype])
        if transform is not None:
            value = transform(value)
        return value.detach().to(device="cpu", dtype=dtype, copy=True).numpy()

    def geometry(self, env):
        mask = env.ninf_mask
        if (not isinstance(mask, torch.Tensor) or mask.ndim != 3
                or mask.shape[0] != 1 or mask.shape[1] < 1
                or mask.shape[2] != env.problem_size + 1):
            raise ValueError("Expected singleton native (batch, POMO, full nodes) mask")
        shape = tuple(mask.shape)
        if self.shape is not None and self.shape != shape:
            raise ValueError("Native trajectory dimensions changed")
        self.shape = shape
        return shape[:2]

    def snapshot(self, env, prefix):
        shape = self.shape[:2]
        if env.load.dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError("Unrecognized native load dtype")
        if env.finished.dtype != torch.bool:
            raise ValueError("Unrecognized native finished dtype")
        row = {prefix + "load": self.copy(env.load, shape),
               prefix + "finished": self.copy(env.finished, shape)}
        current = env.current_node
        if current is None and env.selected_count != 0:
            raise ValueError("Missing current_node after a native action")
        if current is not None and current.dtype != torch.int64:
            raise ValueError("Unrecognized native current_node dtype")
        row[prefix + "current_node"] = (self.filled(shape, -1, np.int64) if current is None
                                         else self.copy(current, shape))
        for name, field in (("time", "current_time"), ("length", "length")):
            value = getattr(env, field, None)
            row[prefix + name + "_present"] = self.filled(shape, value is not None, np.bool_)
            row[prefix + name] = (self.filled(shape, np.nan, _DTYPES[env.load.dtype])
                                  if value is None else self.copy(value, shape))
        return row

    def decoder(self, module, arguments, output):
        env = self.env
        if env is None or self.terminal or env.selected_count != len(self.rows):
            raise RuntimeError("Decoder event outside the active native step")
        self.geometry(env)
        if self.pending is not None:
            raise RuntimeError("Multiple decoder events for one native step")
        probabilities = output[0] if isinstance(output, tuple) else output
        if not isinstance(probabilities, torch.Tensor) or not probabilities.is_floating_point():
            raise ValueError("Expected native decoder probability tensor")
        self.pending = (env.selected_count, self.copy(probabilities, self.shape, torch.float32))

    def before(self, env, selected):
        shape = self.geometry(env)
        count = env.selected_count
        if type(count) is not int or count != len(self.rows) or self.terminal:
            raise RuntimeError("Nonsequential native selected_count")
        if selected.dtype != torch.int64:
            raise ValueError("Expected native int64 actions")
        if count == 0:
            self.arrays["pomo_idx"] = self.copy(env.POMO_IDX, shape)
            for name, size, dims in (("trajectory_id", shape[1], shape),
                                      ("node_id", self.shape[2], (self.shape[2],))):
                self.reserve(dims, np.int64)
                self.arrays[name] = np.arange(size, dtype=np.int64).reshape(dims)
        if count == 1:
            self.arrays["start_node"] = self.copy(selected, shape)
        row = self.snapshot(env, "pre_")
        row["step"] = self.filled((), count, np.int64)
        row["action"] = self.copy(selected, shape)
        row["visited_node"] = self.copy(env.visited_ninf_flag, self.shape, torch.bool, torch.isneginf)
        row["feasible"] = self.copy(env.ninf_mask, self.shape, torch.bool, torch.isfinite)
        valid = self.pending is not None
        if valid and self.pending[0] != count:
            raise RuntimeError("Stale native decoder event")
        row["decoder_probabilities"] = (self.pending[1] if valid else
                                         self.filled(self.shape, np.nan, np.float32))
        row["decoder_valid"] = self.filled(shape, valid, np.bool_)
        row["forced"] = self.filled(shape, not valid, np.bool_)
        self.pending = None
        return row

    def after(self, env, row, result):
        if env.selected_count != len(self.rows) + 1:
            raise RuntimeError("Native step did not increment selected_count exactly once")
        self.geometry(env)
        row.update(self.snapshot(env, "post_"))
        row["post_visited_node"] = self.copy(env.visited_ninf_flag, self.shape, torch.bool, torch.isneginf)
        row["post_feasible"] = self.copy(env.ninf_mask, self.shape, torch.bool, torch.isfinite)
        if self.rows and any(row[key].dtype != self.rows[0][key].dtype for key in row):
            raise ValueError("Native observation dtype changed")
        self.rows.append(row)
        if bool(result[-1]):
            self.terminal = True
            shape = self.shape[:2]
            reward = self.copy(result[1], shape)
            self.arrays["all_start_reward"] = reward
            self.reserve(shape, reward.dtype)
            self.arrays["all_start_cost"] = -reward
            self.arrays["all_start_tours"] = self.copy(env.selected_node_list,
                                                      shape + (len(self.rows),))

    def finish(self):
        if not self.terminal or self.pending is not None or "start_node" not in self.arrays:
            raise RuntimeError("Incomplete native trajectory capture")
        if not any((row["decoder_valid"] & ~row["pre_finished"]).any() for row in self.rows):
            raise RuntimeError("No genuine nonterminal decoder event captured")
        arrays = dict(self.arrays)
        for key in self.rows[0]:
            arrays[key] = np.stack([row[key] for row in self.rows])
        return arrays


def capture_reld(backend, items, *, max_bytes=384 * 2**20):
    """Return (unchanged InferenceResult, numpy arrays, info) from ONE native run.

    No input copies, policy/budget/RNG changes, or historical bitwise replay
    claim. max_bytes bounds observer array payload plus copy/stack reserve,
    not backend-owned allocations or Python object overhead. A limit failure
    aborts the capture without retry or reduced recording. Use one capture per
    process: CVRP's native class binding is temporarily process-global.
    """
    if len(items) != 1:
        raise ValueError("First finite RELD capture requires singleton items")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if backend.method not in {"RELD_CVRP", "RELD_MTL", "RELD_MOEL"}:
        raise ValueError("Expected an original_reld backend")
    bindings = []
    if backend.method == "RELD_CVRP":
        module = importlib.import_module("CVRPEnv")
        original = module.CVRPEnv
        bindings.append((module, "CVRPEnv", original))
        if hasattr(backend.module, "CVRPEnv") and backend.module is not module:
            if backend.module.CVRPEnv is not original:
                raise ValueError("Unrecognized CVRP bridge environment binding")
            bindings.append((backend.module, "CVRPEnv", original))
    else:
        original = backend.env_class
        bindings.append((backend, "env_class", original))
    trace = _Trace(max_bytes)

    class ObserverEnv(original):
        def __init__(environment, *args, **kwargs):
            if trace.env is not None:
                raise RuntimeError("Multiple native environments in one capture")
            super().__init__(*args, **kwargs)
            trace.env = environment

        def step(environment, selected, *args, **kwargs):
            row = trace.before(environment, selected)
            result = super().step(selected, *args, **kwargs)
            trace.after(environment, row, result)
            return result

    handle = None
    try:
        handle = backend.model.decoder.register_forward_hook(trace.decoder)
        for owner, name, _ in bindings:
            setattr(owner, name, ObserverEnv)
        with torch.inference_mode():
            result = backend.run(items)
        arrays = trace.finish()
    finally:
        if handle is not None:
            handle.remove()
        for owner, name, value in reversed(bindings):
            setattr(owner, name, value)
        trace.env = None
    info = dict(backend="original_reld", method=backend.method, problem=backend.problem,
                native_profile="unchanged backend.run(items)", caller_must_supply_frozen_train=True,
                historical_bitwise_replay=False, native_mask_is_feasibility_certificate=False,
                steps=len(trace.rows), starts=trace.shape[1], nodes=trace.shape[2],
                array_bytes=trace.nbytes, peak_array_bytes_bound=2 * trace.nbytes,
                max_bytes=max_bytes, forced_definition="no decoder forward event at this selected_count")
    return result, arrays, info
