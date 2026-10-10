"""CPU-only post-run audit; does not launch models or read label/cost archives."""

import json
from pathlib import Path

from code.V4.r58_scenario import check_route, file_hash
from code.V4.r58_throughput import (GPU, HERE, OUTPUT, ROOT, compare_equivalence,
                                    load_train, read_json, write_json)


def audit():
    manifests = [read_json(OUTPUT / f'{prefix}_selection.json') for prefix in ['aug1', 'aug1_moses_rf']]
    assert manifests[0]['datasets'] == manifests[1]['datasets']
    assert manifests[1]['methods'] == ['MoSES_RF']
    assert file_hash(OUTPUT / 'aug1_selection.json') == manifests[1]['parent_manifest_sha256']
    assert file_hash(OUTPUT / 'aug1_source/r58_throughput.py') == manifests[0]['pilot_source_sha256']
    assert file_hash(HERE / 'r58_throughput.py') == manifests[1]['pilot_source_sha256']
    for name, digest in manifests[0]['frozen_source_sha256'].items():
        assert file_hash(HERE / name) == digest, name
    items = {}
    for problem, data in manifests[0]['datasets'].items():
        items[problem], path = load_train(problem)
        assert file_hash(path) == data['input_sha256']
    counts = dict(measured_routes=0, warmup_routes=0, phases=0, gpu_samples=0)
    phases, aggregates = [], []
    for prefix, manifest in zip(['aug1', 'aug1_moses_rf'], manifests):
        state = read_json(OUTPUT / f'{prefix}_state.json')
        assert state['status'] == 'complete' and not state['failed_phases']
        assert not state['frozen_sources_changed']
        assert state['pilot_source_sha256'] == manifest['pilot_source_sha256']
        assert (OUTPUT / f'{prefix}_exit_code').read_text().strip() == '0'
        summary = read_json(OUTPUT / f'{prefix}_summary.json')
        aggregates.extend(summary['aggregates'])
        for listed in summary['phases']:
            phase = read_json(Path(listed['directory']) / 'result.json')
            directory = Path(phase['directory'])
            problem, method = phase['problem'], phase['method']
            data = manifest['datasets'][problem]
            assert phase['status'] == 'complete' and phase['merged']['complete']
            assert all(code == 0 for code in phase['returncodes'])
            assert listed['equivalence_to_aug1_serial']['equivalent']
            assert listed['equivalence_to_aug1_serial']['raw_tour_differences'] == 0
            task = read_json(directory / 'task.json')
            assert sorted(i for group in task['assignments'] for i in group) == data['indices']
            assert task['warmup_indices'] == data['warmup_indices']
            baseline = read_json(ROOT / 'preflight' / f'{problem}__{method}.json')['deployment']
            rows = list(phase['merged']['rows'])
            for worker in range(phase['workers']):
                worker_state = read_json(directory / f'worker_{worker}.state.json')
                profile = read_json(directory / f'worker_{worker}.deployment.json')
                assert worker_state['status'] == 'complete'
                assert profile['checkpoints'] == baseline['checkpoints']
                assert profile['source_sha256'] == baseline['source_sha256']
                for key in ['loader_sha256', 'checkpoint_alias_loader_sha256', 'constraint_adapter_sha256',
                            'r58_tensor_adapter_sha256', 'checkpoint_loading']:
                    assert profile[key] == baseline[key], key
                assert profile['budget']['augmentation'] == 1
                assert profile['budget']['precision'] == baseline['budget']['precision']
                assert [row['index'] for row in worker_state['warmups']] == data['warmup_indices']
                rows.extend(worker_state['warmups'])
                counts['warmup_routes'] += len(worker_state['warmups'])
            for row in rows:
                checked = check_route(problem, items[problem][row['index']], row['route'], row['reported_cost'])
                assert checked['feasible'] and checked['cost_matches'] and checked['cost'] == row['cost']
                assert row['seed'] == 2 and row['inference_batch_size'] == 1
                n = row['true_size']
                assert row['rf_budget_witness'] == dict(augmentation=1, original_orientation=True,
                    policy_input_shape=[1, n + 1, 2], starts=n, checkpoint_bucket=50 if n <= 75 else 100,
                    precision='FP32', inference_batch_size=1, decoding='multistart_greedy')
            owned = {worker['pid'] for worker in phase['worker_states']}
            for sample in read_json(directory / 'gpu_samples.json'):
                assert sample['uuid'] == GPU
                assert all(int(line.split(',')[0]) in owned for line in sample['contexts'].splitlines())
                counts['gpu_samples'] += 1
            counts['measured_routes'] += len(phase['merged']['rows'])
            counts['phases'] += 1
            phases.append(phase)
    assert counts['phases'] == 36
    assert counts['measured_routes'] == 432 and counts['warmup_routes'] == 252
    assert all(row['complete'] for row in aggregates)
    for phase in phases:
        serial = next(p for p in phases if (p['problem'], p['method'], p['repeat'], p['workers']) ==
                      (phase['problem'], phase['method'], phase['repeat'], 1))
        assert compare_equivalence(serial['merged'], phase['merged'])['equivalent']
    state = read_json(OUTPUT / 'aug1_moses_rf_state.json')
    assert state['previous_single_gpu_wall_seconds'] == read_json(OUTPUT / 'aug1_state.json')['combined_single_gpu_wall_seconds']
    assert state['combined_single_gpu_wall_seconds'] <= 10800
    assert read_json(ROOT / 'pipeline_state.json')['status'] == 'skipped_over_budget'
    assert not (ROOT / 'deployments.lock.json').exists()
    assert not list(ROOT.glob('**/columns/*.jsonl'))
    assert not list(ROOT.glob('**/performance.npz'))
    report = dict(status='passed', **counts, methods=sorted({p['method'] for p in phases}),
        frozen_four_unchanged=True, checkpoints_and_source_unchanged=True, original_orientation_all_N_starts_FP32=True,
        all_paid_costs_and_physical_and_raw_tours_equal_aug1_serial=True, all_sampled_contexts_owned=True,
        no_val_test_or_archive_cost_reads=True, no_deployment_lock_or_generated_columns=True,
        previous_audit='verification.json (original concurrency plus diffusion retry, unchanged)',
        combined_single_gpu_wall_seconds=state['combined_single_gpu_wall_seconds'])
    write_json(OUTPUT / 'aug1_verification.json', report)
    write_json(OUTPUT / 'aug1_combined_summary.json', dict(protocol=manifests[0]['protocol'],
        manifests=['aug1_selection.json', 'aug1_moses_rf_selection.json'], aggregates=aggregates,
        repeat_aggregation='Both repetitions, ratio of summed actual completion-barrier wall times; never best/min',
        algorithm_equivalence_to_aug8=False, full128_qualification=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    audit()
