"""Bounded, input-only mmap advice probe inside an existing job container."""

import argparse
import gc
import json
import mmap
import os
from pathlib import Path
import time

import numpy as np


def read_bytes():
    values = Path('/proc/self/io').read_text().splitlines()
    return int(next(line.split(':')[1] for line in values if line.startswith('read_bytes:')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-file', type=Path, required=True)
    parser.add_argument('--expected-container', required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--range-prefetch-only', action='store_true')
    args = parser.parse_args()
    cgroup = Path('/proc/self/cgroup').read_text()
    if args.expected_container not in cgroup or args.cpu not in os.sched_getaffinity(0):
        raise RuntimeError('Use the authorized allocation-owned container and its unused CPU')
    os.sched_setaffinity(0, {args.cpu})
    features = np.load(args.feature_file, mmap_mode='r', allow_pickle=False)
    if features.dtype != np.float16 or features.ndim != 3 or features.shape[-1] != 128:
        raise ValueError('Expected one existing FP16 node-feature mapping')
    seed = 5723 if args.range_prefetch_only else 5722
    count = 2 if args.range_prefetch_only else 4
    rows = np.random.default_rng(seed).choice(len(features), size=count * 1024, replace=False).reshape(count, 1024)
    fd = os.open(args.feature_file, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    records = []
    modes = ('range_prefetch',) * 2 if args.range_prefetch_only else ('normal', 'random', 'normal', 'random')
    for mode, indices in zip(modes, rows):
        features._mmap.madvise(mmap.MADV_NORMAL if mode == 'normal' else mmap.MADV_RANDOM)
        before, start = read_bytes(), time.perf_counter()
        if mode == 'range_prefetch':
            stride = int(features.strides[0])
            for row in np.sort(indices):
                os.posix_fadvise(fd, int(features.offset) + int(row) * stride,
                                 stride, os.POSIX_FADV_WILLNEED)
        values = np.ascontiguousarray(features[indices])
        elapsed, physical = time.perf_counter() - start, read_bytes() - before
        records.append(dict(mode=mode, seconds=elapsed, physical_read_bytes=physical,
                            requested_bytes=values.nbytes, shape=list(values.shape)))
        del values
        gc.collect()
    os.close(fd)
    print(json.dumps(dict(CPU_input_only=True, GPU_work=False, cgroup=cgroup,
                          cpus=sorted(os.sched_getaffinity(0)), records=records)))


if __name__ == '__main__':
    main()
