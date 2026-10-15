"""Render saved round006 results; optional assigned-UUID read-only exit receipt."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
UUID = 'GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be'
METRICS = ('top1', 'ce', 'mean_cost', 'actual_regret_pct', 'vs_sbs_pct')


def read(path):
    return json.loads(path.read_text())


def write(path, record):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render():
    result = read(ROOT / 'results.json')
    receipt = read(ROOT / 'execution_receipt.json')
    lock = read(ROOT / 'candidate_lock.json')
    if not receipt['complete'] or receipt['successful_updates'] != 14904:
        raise ValueError('Only report a completed actual14904 endpoint as complete')
    methods, tasks, aggregate, lines = [], [], {}, []
    for split in ('train', 'val'):
        changes = read(ROOT / 'evaluation' / split / 'final14904/method_changes.json')
        totals = dict(changed=0, corrected=0, harmed=0, wrong_to_wrong=0,
                      corrected_cost_sum=0., harmed_cost_sum=0., wrong_to_wrong_cost_sum=0.)
        for task, change in changes.items():
            candidate = result['fitted'][split]['per_problem'][task]
            baseline = result['B'][split]['per_problem'][task]
            row = dict(split=split, task=task, n=candidate['n'])
            for name in METRICS:
                row.update({f'B_{name}': baseline[name], f'candidate_{name}': candidate[name],
                            f'delta_{name}': candidate[name] - baseline[name]})
            row['changed'] = change['changed']
            totals['changed'] += change['changed']
            for group in ('corrected', 'harmed', 'wrong_to_wrong'):
                value = change['groups'][group]
                row[group] = value['n']
                row[group + '_signed_cost_sum'] = value['signed_cost_sum']
                row[group + '_signed_cost_mean'] = value['signed_cost_mean']
                row[group + '_signed_regret_sum_pct'] = value['signed_regret_sum_pct']
                totals[group] += value['n']
                totals[group + '_cost_sum'] += value['signed_cost_sum']
            row['zero_winner_methods_B'] = '|'.join(change['zero_winner_methods_B'])
            row['zero_winner_methods_candidate'] = '|'.join(change['zero_winner_methods_fitted'])
            tasks.append(row)
            methods.extend(dict(split=split, task=task, **value) for value in change['native_methods'])
        totals['zero_winner_method_pairs_B'] = sum(len(c['zero_winner_methods_B']) for c in changes.values())
        totals['zero_winner_method_pairs_candidate'] = sum(len(c['zero_winner_methods_fitted']) for c in changes.values())
        aggregate[split] = totals
    write_csv(ROOT / 'per_task_comparison.csv', tasks)
    write_csv(ROOT / 'native_method_counts.csv', methods)
    write(ROOT / 'change_summary.json', aggregate)
    lines += ['# R78 Round006 Final Results', '',
        'Owner: Solmax, 6.1Sol max. Original fixed protocol; no additional configurations.', '',
        f"- Actual successful updates: {receipt['successful_updates']} (828 per task,18 tasks).",
        '- Logical batch1024; seed2; one approved4090. Update1 is formal budget update1.',
        f"- Runtime: {receipt['elapsed_seconds']:.6f}s ({receipt['elapsed_seconds'] / 3600:.6f} GPUh), cap4h.",
        f"- Checkpoint SHA256: `{lock['checkpoint_sha256']}`; locked before candidate VAL.",
        f"- Gate: `{json.dumps(result['gate'], sort_keys=True)}`; pass={result['gate_passed']}.",
        f"- Selected rule: `{result['selected_rule']}`. TEST completed={receipt['test_completed']}.",
        '- Actual candidate metrics below are NOT the fallback metrics.', '',
        '| Split | Model | Top1 % | CE | Raw mean cost | Actual regret % | Macro VS % |',
        '|---|---|---:|---:|---:|---:|---:|']
    for split in ('train', 'val'):
        for source, name in (('B', 'locked B'), ('fitted', 'actual candidate'), ('fallback_B', 'whole B fallback')):
            m = result[source][split]['macro']
            lines.append(f"| {split.upper()} | {name} | {100*m['macro_top1']:.10f} | {m['macro_ce']:.12f} | "
                         f"{m['macro_mean_cost']:.15f} | {m['macro_actual_regret_pct']:.12f} | {m['macro_vs_sbs_pct']:.12f} |")
    lines += ['', '## Choice Changes', '',
        '| Split | Changed | Corrected | Harmed | Wrong-to-wrong | B zero winner-method pairs | Candidate zero pairs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split, a in aggregate.items():
        lines.append(f"| {split.upper()} | {a['changed']} | {a['corrected']} | {a['harmed']} | {a['wrong_to_wrong']} | "
                     f"{a['zero_winner_method_pairs_B']} | {a['zero_winner_method_pairs_candidate']} |")
    lines += ['', 'Full89 native method counts per split are in `native_method_counts.csv` (178 rows).',
        'All18 task metrics and corrected/harmed/wrong-to-wrong signed FP64 cost changes',
        'per split are in `per_task_comparison.csv` (36 rows); aggregate changes are in',
        '`change_summary.json`. Cost sign is candidate minus B: negative is improvement.', '',
        '## Historical Matched Control Limits', '',
        'The previously completed actual final14904 control VAL has Top1=70.81443446013407%,',
        'raw mean cost=11.395855343252101 and actual regret=0.41092598253845325%.',
        'Its actual endpoint, not its selected B fallback, is the descriptive comparator.',
        'No control TRAIN metrics were available or invented, and the baseline was not rerun.',
        'This is one seed with reused VAL and a historical, not fresh randomized, control.',
        'Comparison with B confounds extra training with the new objective; kernel/code drift',
        'also limits causal attribution. No feature-starvation, universal task-safety,',
        'statistical non-inferiority or paper-level mechanism claim follows from this screen.', '',
        '## Scope And Receipts', '',
        'No NSS TEST/LIB values, other TEST/LIB artifacts or new raw-data audits informed the rule.',
        'A failed gate closes this fixed trial; no rescue, extra seed/configuration or budget.',
        'The overall R78 file goal stays active; native/legacy application goals are untouched.',
        'Authoritative complete outputs: `results.json`, `candidate_lock.json`,',
        '`execution_receipt.json`, `update1_receipt.json`, `cpu_checks.json`,',
        '`evaluation/{train,val}/final14904/` and `train.log`.',
        'Actual worker/Slurm exit and assigned-card release evidence: `PROCESS_EXIT_RECEIPT.json`',
        '(recorded after the GPU worker exits; allocation1475 is retained, not cancelled).', '']
    (ROOT / 'FINAL_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(gate=result['gate'], gate_passed=result['gate_passed'],
        successful_updates=receipt['successful_updates'], elapsed_seconds=receipt['elapsed_seconds'],
        macros={source: {split: result[source][split]['macro'] for split in ('train', 'val')}
                for source in ('fitted', 'B', 'fallback_B')}, changes=aggregate), indent=2))


def gpu_exit():
    if (os.environ.get('SLURM_JOB_ID') != '1475' or not os.environ.get('SLURM_STEP_ID')
            or socket.gethostname().split('.')[0] != 'gpu05'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != UUID):
        raise ValueError('Read only the assigned UUID in the existing exact Slurm allocation')
    query = ['nvidia-smi', '-i', UUID, '-q', '-x']
    output = subprocess.run(query, check=True, text=True, capture_output=True).stdout
    tree = ET.fromstring(output)
    gpu = tree.find('gpu')
    if gpu is None or gpu.findtext('uuid') != UUID:
        raise ValueError('NVML receipt must refer to exactly the assigned UUID')
    processes = [{child.tag: child.text for child in process}
                 for process in gpu.findall('processes/process_info')]
    write(ROOT / 'gpu_exit_observation.json', dict(captured_utc=datetime.now(timezone.utc).isoformat(),
        node='gpu05', allocation='1475', query_step=os.environ['SLURM_STEP_ID'], command=query,
        gpu_uuid=UUID, gpu_util=gpu.findtext('utilization/gpu_util'),
        memory_used=gpu.findtext('fb_memory_usage/used'), processes=processes,
        no_compute_processes=not any(p.get('type') in ('C', 'C+G') for p in processes),
        other_gpus_queried=False, training_or_inference_repeated=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('report', 'gpu-exit'))
    args = parser.parse_args()
    render() if args.stage == 'report' else gpu_exit()
