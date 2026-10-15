"""One-group-ahead CPU gathers; no CUDA, pinning or random-state changes."""

import operator
from concurrent.futures import ThreadPoolExecutor


class CpuGroupPrefetch:
    """Gather ``plan[step]`` entries in their original task/index order.

    Loaders map problem names to caches exposing ``batch_cpu(indices)``. Keep
    loaders and plan unchanged until close, and call take sequentially from
    start_step. Only one future group exists; consuming it submits the next.
    Worker exceptions propagate through take. Close/join before closing caches.
    """

    def __init__(self, loaders, plan, start_step=0):
        start_step = operator.index(start_step)
        if not 0 <= start_step <= len(plan):
            raise ValueError('start_step must be within [0, len(plan)]')
        self._loaders, self._plan = loaders, plan
        self._next_step, self._closed = start_step, False
        self._future = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='r75-cpu-input')
        try:
            if start_step < len(plan):
                self._future = self._executor.submit(self._load_group, start_step)
        except BaseException:
            self.close()
            raise

    def _load_group(self, step):
        return [self._loaders[problem].batch_cpu(indices)
                for problem, indices in self._plan[step]]

    def take(self, step):
        """Obtain one CPU group, then schedule only its immediate successor."""
        if self._closed:
            raise RuntimeError('CpuGroupPrefetch is closed')
        step = operator.index(step)
        if step != self._next_step:
            raise ValueError(f'Expected step {self._next_step}, got {step}')
        if self._future is None:
            raise IndexError('No remaining prefetch groups')
        try:
            batches = self._future.result()
        except BaseException:
            self.close()
            raise
        self._future = None
        self._next_step += 1
        if self._next_step < len(self._plan):
            self._future = self._executor.submit(self._load_group, self._next_step)
        return batches

    def close(self):
        """Cancel queued work, join running work, then release cache references."""
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self._future = self._executor = None
            self._loaders = self._plan = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
