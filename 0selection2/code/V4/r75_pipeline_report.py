"""Compare consecutive official R75 update windows; no additional GPU work."""

import json
from datetime import datetime

from .r75_capacity import ROOT, utc, write_json
from .r75_runtime_report import sample


def window(rows, begin, end):
    chosen = [row for row in rows if begin < row['successful_update'] <= end]
    previous = next(row for row in rows if row['successful_update'] == begin)
    if len(chosen) != end - begin:
        raise ValueError('Require all completed updates in the observed window')
    result = sample(chosen)
    wall = (datetime.fromisoformat(chosen[-1]['utc']) - datetime.fromisoformat(previous['utc'])).total_seconds()
    result.update(first_update=begin + 1, last_update=end, wall_seconds=wall,
                  wall_rows_per_second=sum(row['rows'] for row in chosen) / wall)
    return result


def main():
    directory = ROOT / 'runtime_pipeline_continuation2'
    boundary = json.loads((directory / 'quiesced.json').read_text())
    update = boundary['successful_updates']
    rows = []
    with (ROOT / 'single_seed2/updates.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                break
            row = json.loads(line)
            if update - 120 <= row['successful_update'] <= update + 126:
                rows.append(row)
    before, after = window(rows, update - 120, update), window(rows, update + 6, update + 126)
    if before['tasks'] != after['tasks'] or any(len(row['tasks']) != 3 for row in rows):
        raise ValueError('Only compare balanced full three-task windows')
    result = dict(utc=utc(), source='official existing training updates; no extra benchmarks',
        boundary_update=update, before=before, after=after,
        update_speedup=before['mean_step_seconds'] / after['mean_step_seconds'],
        wall_throughput_ratio=after['wall_rows_per_second'] / before['wall_rows_per_second'],
        window_rule='120 updates before; first6 after excluded, then120 updates; identical task counts',
        batches_paired=False, loss_and_rng_and_update_math_unchanged=True,
        extra_gpu_batch_buffers=0, cpu_prefetch_groups=1,
        scientific_source_files_modified=False, logged_successful_updates_replayed=0,
        utilization_not_guaranteed_100_percent=True)
    gpu = directory / 'gpu_samples.json'
    if gpu.exists():
        result['gpu_short_sample'] = json.loads(gpu.read_text())
    write_json(directory / 'optimization_measurements.json', result)
    lines = ['# R75 CPU/GPU Pipeline Continuation', '',
        f'The same seed2 run continues from logged/durable update {update}; '
        'no logged successful update was lost or replayed. The original model, optimizer, '
        'AMP retry path, RNG, data plan, batch384 and scientific protocol are unchanged.', '',
        'One CPU thread prepares exactly the next real task group. Only the main thread '
        'uses pinned staging and CUDA. No additional group is prefetched into GPU memory. '
        'An isolated runtime copy removes two timing-only global CUDA barriers; '
        'the original scientific source files are unchanged.', '',
        '| Measurement | RAM input before | CPU prefetch after |', '|---|---:|---:|',
        f"| Mean update seconds | {before['mean_step_seconds']:.4f} | {after['mean_step_seconds']:.4f} |",
        f"| Update rows/s | {before['rows_per_second']:.1f} | {after['rows_per_second']:.1f} |",
        f"| Wall rows/s, including intervening logging/checkpoints | {before['wall_rows_per_second']:.1f} | {after['wall_rows_per_second']:.1f} |", '',
        f"AMP retries in the two windows: {before['retries']} before, {after['retries']} after. "
        'All reported successful updates had finite loss and gradients.', '',
        f"Observed update speedup: {result['update_speedup']:.3f}x; "
        f"wall throughput ratio: {result['wall_throughput_ratio']:.3f}x. "
        'These are balanced120-update observed windows with different batches, not a paired '
        'controlled benchmark or a guarantee of constant99-100% GPU utilization.', '',
        'The first observed windows do not establish an additional material speedup: '
        'update time is almost unchanged and wall throughput did not improve. '
        'The earlier lossless RAM fix remains the main measured acceleration. '
        'Do not attribute the retry-count difference to prefetching; batches differ.', '',
        'The resumed epoch is mixed pre/post-continuation; its optimization.seconds only '
        'covers the latest process segment. Use complete later epochs for epoch comparisons. '
        'The final completion metadata records both runtime continuations and partial timing scopes.', '',
        'Added execution files: r75_cpu_prefetch.py, r75_pipeline_runner.py, '
        'r75_pipeline_boundary.py, run_v4_r75_pipeline.sh and r75_pipeline_report.py. '
        'The running command is `bash code/V4/run_v4_r75_pipeline.sh` in tmux '
        '`r75_train_pipeline2`; do not launch a duplicate. The original native70 goal stays paused.', '']
    (directory / 'PIPELINE_IO.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
