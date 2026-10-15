"""Summarize observed R75 IO speedup without running another model or replay."""

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from .r75_capacity import ROOT, utc, write_json


def sample(rows):
    seconds = sum(row['seconds'] for row in rows)
    return dict(updates=len(rows), mean_step_seconds=seconds / len(rows),
        rows_per_second=sum(row['rows'] for row in rows) / seconds,
        tasks=dict(Counter(task for row in rows for task in row['tasks'])),
        retries=sum(row['retries'] for row in rows),
        all_finite=all(row['finite_loss'] and row['finite_gradient'] for row in rows))


def main():
    root, directory = ROOT, ROOT / 'runtime_io_continuation1'
    boundary = json.loads((directory / 'quiesced.json').read_text())
    before_io = json.loads((root / 'profile_io_before.json').read_text())
    after_io = json.loads((directory / 'profile_io_after.json').read_text())
    gpu = json.loads((directory / 'gpu_samples.json').read_text())
    update = boundary['successful_updates']
    before, after, last = [], [], None
    with (root / 'single_seed2/updates.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                break
            row = json.loads(line)
            if update - 120 < row['successful_update'] <= update:
                before.append(row)
            if update + 6 < row['successful_update'] <= update + 126:
                after.append(row)
            last = row
    a, b = sample(before), sample(after)
    if len(before) != 120 or len(after) != 120 or a['tasks'] != b['tasks']:
        raise ValueError('Require balanced observed windows with identical task counts')
    stages = [json.loads(line) for line in (directory / 'stage_timings.jsonl').read_text().splitlines()]
    fields = ('gather_seconds', 'pin_seconds', 'h2d_seconds', 'update_wall_seconds',
              'update_stream_span_seconds')
    warm = stages[6:]
    result = dict(utc=utc(), source='actual official training updates, no extra benchmark updates',
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        boundary_update=update, existing_logged_updates_replayed=0,
        discarded_uncommitted_inflight_groups_upper_bound=1,
        before=a, after=b, speedup=a['mean_step_seconds'] / b['mean_step_seconds'],
        window_rule='last120 before versus120 after excluding first6 cold updates; full three-task groups; all18 tasks20 times each',
        paired_batches=False, stage_samples=len(stages), warm_stage_samples=len(warm),
        stage_means_seconds={key: statistics.fmean(row[key] for row in stages) for key in fields},
        warm_stage_means_seconds={key: statistics.fmean(row[key] for row in warm) for key in fields},
        stream_span_not_pure_kernel_busy_time=True,
        gpu_utilization_sample_mean_percent={name: statistics.fmean(gpu[name]['utilization_percent'])
                                             for name in ('before', 'after')},
        gpu_utilization_scope='12 one-second observations per window; not a full-run guarantee',
        major_faults={'before': before_io['major_faults'], 'after': after_io['major_faults']},
        input_disk_read_bytes={'before': before_io['io_delta']['read_bytes'],
                               'after': after_io['io_delta']['read_bytes']},
        probe_seconds={'before': before_io['elapsed_seconds'], 'after': after_io['elapsed_seconds']},
        ram_input_bytes=after_io['packed_total_bytes'],
        observed_latest_update=last['successful_update'],
        remaining_training_hours_estimate=(62640 - last['successful_update']) * b['mean_step_seconds'] / 3600,
        estimate_excludes_future_full_train_evaluations_and_final_cache_reload=True,
        unchanged=['batch384', 'seed2', 'model', 'loss', 'optimizer state', 'RNG and sample order',
                   'geometry normalization', 'precision', '40-epoch budget', 'VAL selection', 'one locked TEST'],
        changed=['lossless compact anonymous CPU RAM inputs', 'bounded per-problem host pinned staging',
                 'Slurm step16GiB to19GiB', '24-update runtime timing wrapper'],
        files_added=['r75_fast_input.py', 'r75_fast_runner.py', 'run_v4_r75_fast.sh',
                     'r75_checkpoint_boundary.py', 'r75_runtime_probe.py', 'r75_runtime_report.py'],
        original_scientific_source_files_modified=False,
        mixed_epoch_note='Epoch11 spans old/new IO. Its saved optimization.seconds is post-resume wall only; do not use it as a whole-epoch timing.')
    write_json(directory / 'optimization_measurements.json', result)
    lines = ['# R75 Runtime IO Optimization', '',
        'This resumes the existing one seed2 run at successful update16860. '
        'No logged successful updates were discarded or replayed; at most one uncommitted in-flight group '
        'may have been discarded. The checkpoint, optimizer, scaler, RNG and original logs are preserved.', '',
        '| Measurement | Before | After |', '|---|---:|---:|',
        f"| Mean full three-task update (s) | {a['mean_step_seconds']:.4f} | {b['mean_step_seconds']:.4f} |",
        f"| Rows/s | {a['rows_per_second']:.1f} | {b['rows_per_second']:.1f} |",
        f"| Mean GPU utilization, short sample (%) | {result['gpu_utilization_sample_mean_percent']['before']:.2f} | {result['gpu_utilization_sample_mean_percent']['after']:.2f} |",
        f"| Major faults in process probe | {before_io['major_faults']} | {after_io['major_faults']} |",
        f"| Process input disk read in probe (bytes) | {before_io['io_delta']['read_bytes']} | {after_io['io_delta']['read_bytes']} |", '',
        f"Observed update speedup: {result['speedup']:.2f}x. The120-update windows contain each of18 tasks20 times; "
        'the first6 cold restart updates are excluded. Batches differ, so this is an observed runtime comparison, '
        'not an exactly paired benchmark. GPU utilization is12 one-second samples, not sustained100% utilization.', '',
        'The16GiB old process hit its memory ceiling and repeatedly faulted the22.11GB padded mmap inputs. '
        'The new19GiB process owns14.03GB losslessly packed TRAIN/VAL inputs in anonymous RAM, '
        'with small bounded pinned staging buffers. No quantization, relabeling or candidate changes are involved.', '',
        '## Warm Stage Timing', '', '| Stage | Mean Seconds |', '|---|---:|']
    lines.extend(f'| {key} | {value:.6f} |' for key, value in result['warm_stage_means_seconds'].items())
    lines += ['', 'CUDA stream span includes CPU launch gaps; it is not pure kernel busy time. '
        'The24 instrumented updates are official training updates, not an additional experiment.', '',
        '## Execution', '', 'Current tmux: `r75_train_fast1`; monitor: `r75_monitor`; '
        'Slurm step1465.125 on the same approved GPU UUID. Source scientific files are unchanged; '
        'the IO wrapper is separately hashed in `runtime_started.json`. '
        'Four real CPU batches matched every field, dtype and padding exactly.', '',
        'Epoch11 is a mixed pre/post-resume epoch. Its saved optimization.seconds reports only '
        'post-resume wall time; the saved prefix step time is in `quiesced.json`. '
        'Final compute metadata separates the last process segment from end-to-end elapsed time.', '',
        '## Added Files', '', '| File | Purpose |', '|---|---|',
        '| r75_fast_input.py | Lossless packed RAM batches and bounded pinned staging |',
        '| r75_fast_runner.py | Resume existing run, replace input factory, profile24 real updates |',
        '| run_v4_r75_fast.sh | Same GPU/allocator/environment;19GiB step |',
        '| r75_checkpoint_boundary.py | Capture a logged durable state without truncation/replay |',
        '| r75_runtime_probe.py | Read-only12-second process/IO samples and memory sizing |',
        '| r75_runtime_report.py | Generate this report from saved observations; no forwards/replays |', '']
    (directory / 'RUNTIME_IO.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
