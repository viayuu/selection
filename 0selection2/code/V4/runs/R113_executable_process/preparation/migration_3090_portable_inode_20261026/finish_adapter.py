"""Saved-only R113 finisher hooks for the independent two-3090 migration.

Only a private in-memory copy of the unforked analysis/publisher is adapted.
Old failed attempts stay failed. No GPU, checkpoints, predictions or protocol
are created, changed, selected or evaluated here.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from code.V4 import r113_3090_queue as queue


ROOT, HERE = queue.ROOT, queue.HERE


def private_finisher():
    path = queue.REPO / 'code/V4/r113_finish.py'
    spec = importlib.util.spec_from_file_location('code.V4._r113_3090_saved_finish', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hooks():
    finish = private_finisher()
    original_sources = finish.frozen_sources
    original_boundary = finish.science_boundary

    def completion_evidence(root, *, plan_path=None):
        if Path(root).resolve() != ROOT or (plan_path is not None and Path(plan_path).resolve() != queue.PLAN):
            raise ValueError('Migration finisher only accepts the new R113 two-slot plan/root')
        ready = finish.required(HERE / 'ready.json')
        if ready.get('main_approved') is not True:
            raise finish.NotReady('Awaiting independent 3090 main-ready receipt')
        ready = queue.validated_ready(queue.PLAN)
        state = finish.required(HERE / 'controller.json')
        inputs = {}
        for path in (queue.PLAN, HERE / 'ready.json', Path(ready['plan']['operations_receipt']), Path(__file__)):
            finish.remember(inputs, path)
        links = queue.operations(ready['plan'], verify_checkpoints=False)
        if links != ready['links']:
            raise ValueError('Original interruption/pretraining evidence changed after approval')
        projection = {}
        for arm in queue.ARMS:
            job = state.get('jobs', {}).get(arm, {})
            if job.get('state') in ('failed', 'blocked'):
                raise RuntimeError('New R113 migration ' + arm + ' failed; never retry: ' + str(job))
            if job.get('state') != 'complete':
                raise finish.NotReady('New 3090 R113 ' + arm + ' state=' + str(job.get('state', 'pending')))
            number, slot = job.get('attempts'), job.get('slot')
            if type(number) is not int or number < 1 or job.get('exit_status') != 0:
                raise ValueError('Completed migration job must name its exact rc0 attempt')
            item = queue.require_slot(slot)
            folder = queue.attempt_path(ready, arm, number)
            if finish.common.inside(job.get('attempt_dir', ''), ROOT) != folder:
                raise ValueError('Migration completion attempt path mismatch')
            worker_path = folder / 'worker.json'
            worker = finish.required(worker_path)
            request, entry = finish.required(folder / 'request.json'), finish.required(folder / 'entry.json')
            stages = worker.get('stages', [])
            names = [s.get('name') for s in stages]
            expected = ready['commands'][arm][slot]
            formal = [s for s in stages if s.get('name') != 'capacity']
            gpu = worker.get('checks', {}).get('gpu', {})
            if (worker.get('state') != 'complete' or worker.get('exit_status') != 0
                    or worker.get('job') != arm or worker.get('slot') != slot
                    or worker.get('gpu_uuid') != item['uuid'] or job.get('gpu_uuid') != item['uuid']
                    or worker.get('gpu_type') != '3090' or worker.get('actual_gpu_type') != '3090'
                    or gpu.get('uuid') != item['uuid'] or 'RTX 3090' not in gpu.get('name', '')
                    or worker.get('checks', {}).get('allocation') != 1465
                    or worker.get('checks', {}).get('container_id') != queue.CONTAINER
                    or job.get('allocation') != 1465
                    or names not in (['train', 'finalize'], ['capacity', 'train', 'finalize'])
                    or any(s.get('state') != 'complete' or s.get('exit_status') != 0 for s in stages)
                    or [dict(name=s['name'], argv=s['argv']) for s in formal] != expected
                    or request.get('job') != arm or request.get('slot') != slot
                    or request.get('ready_sha256') != queue.digest(HERE / 'ready.json')
                    or entry.get('state') != 'complete' or entry.get('exit_status') != 0
                    or entry.get('allocation', {}).get('job') != 1465
                    or entry.get('container', {}).get('id') != queue.CONTAINER):
                raise ValueError('Require new independent real-3090 rc0 train/finalize pipeline: ' + arm)
            capacity = queue.validate_capacity(ready)
            if worker.get('capacity_evidence') != capacity:
                raise ValueError('Migration worker lacks the shared same-cap 3090 memory proof')
            finish.remember(inputs, Path(capacity['path']))
            for path in (worker_path, folder / 'request.json', folder / 'entry.json'):
                finish.remember(inputs, path)
            config = finish.read(ROOT / arm / 'protocol.json')
            if arm == 'A_scratch':
                if config.get('gpu_type') != '3090' or worker.get('resume_link'):
                    raise ValueError('A must be fresh, with an honest 3090 protocol')
            else:
                link = links[arm]
                if (worker.get('resume_link') != link or config.get('gpu_type') != '4090'
                        or queue.canonical(config) != link['state_hash']):
                    raise ValueError('B/C must explicitly link unchanged historical protocol + approved last.pt')
                finish.remember(inputs, Path(link['source_worker']))
                finish.remember(inputs, ROOT / 'pretrain' / arm / 'completion.json')
            projection[arm] = dict(job, actual_gpu_type='3090', hardware_explanation=queue.HARDWARE_NOTE,
                original_interruption=links.get(arm), new_worker_sha256=queue.digest(worker_path))
        # Neither mutable controller JSON nor independent R114 state is frozen.
        return inputs, projection

    def frozen_sources(root, inputs, plan_path=None):
        if plan_path is not None and Path(plan_path).resolve() != queue.PLAN:
            raise ValueError('Migration analysis must bind the new queue plan')
        ready = queue.validated_ready(queue.PLAN)
        # Keep the original science freeze, plus independent posthoc resource hooks.
        result = original_sources(root, inputs, finish.wave_plan(root))
        for path in (Path(__file__), queue.PLAN, HERE / 'ready.json',
                     queue.REPO / 'code/V4/r113_3090_queue.py', queue.WRAPPER):
            finish.remember(inputs, path)
            expected = ready['source_sha256'].get(str(path.resolve()))
            if path.name != 'ready.json' and expected != queue.digest(path):
                raise ValueError('New migration source hook is not launch-frozen: ' + str(path))
            result[str(path.resolve())] = queue.digest(path)
        return result

    def science_boundary(root, models, inputs, *, plan_path=None):
        value = original_boundary(root, models, inputs, plan_path=plan_path)
        ready = queue.validated_ready(queue.PLAN)
        value['resource_migration'] = dict(actual_execution='RTX3090 only, gpu03/job1465',
            hardware_explanation=queue.HARDWARE_NOTE, resumed=ready['links'],
            gpu_type_not_a_scientific_comparison_field=True,
            original_attempts_rewritten=False, pretraining_repeated=False)
        value['mechanism'] += ('; resource-only migration: B/C processor8640 completed on 4090, '
            'selector resumed at persisted20160 on 3090; unsaved B2286/C1872 updates replayed')
        return value

    def publication_paths(root):
        root = Path(root).resolve()
        _, jobs = completion_evidence(root, plan_path=queue.PLAN)
        paths = [finish.V4 / name for name in finish.CODE_NAMES]
        paths += [root / n for n in (*finish.OUTPUTS, 'finish_complete.json', 'protocol_lock.json',
            'PLAN.md', 'process_diagnostic_inputs.json', 'preparation/wave_plan.json', 'preparation/LAUNCH.md')]
        paths += [root / 'process_train_v1' / n for n in ('schema.json', 'source_manifest.json',
            'completion_receipt.json', 'run_config.json', 'manifest.json', 'README.md')]
        paths += [queue.REPO / 'code/V4' / n for n in
                  ('r113_3090_queue.py', 'test_r113_3090_queue.py', 'run_v4_r113_3090.sh',
                   'resource_3090_migration.py', 'run_resource_3090_migration.sh',
                   'resource_font_3090.sh')]
        paths += [HERE / n for n in ('finish_adapter.py', 'plan.json', 'ready.json', 'cpu_tests.json',
            'capacity_attempt.json', 'capacity_3090.json', 'HANDOFF.md', 'font_dependency.json')]
        ready = queue.validated_ready(queue.PLAN)
        paths += [Path(ready['plan']['operations_receipt'])]
        for arm in queue.ARMS:
            paths += [root / arm / n for n in ('protocol.json', 'initialization.json', 'final_test_lock.json',
                'training_complete.json', 'report_complete.json', 'lib_stage_complete.json', 'inference_timing.json',
                'process_diagnostics_complete.json')]
            paths += [root / arm / 'process_diagnostics' / ('process_diagnostics.' + suffix) for suffix in ('json', 'csv', 'pdf')]
            paths += [root / arm / 'lib_evaluation' / (ds + '_results.json') for ds in ('TSPLIB', 'CVRPLIB')]
            paths += [root / arm / 'full_model_statistics' / n for n in ('MODEL_STATISTICS.pdf',
                'selection_distributions.pdf', 'learning_curves.pdf', 'training_and_method_diagnostics.pdf',
                'RESULTS.md', 'protocol.json', 'metrics_by_problem.csv', 'nss_comparison.csv',
                'selection_distribution.csv', 'single_method_comparison.csv', 'inference_timing.csv')]
            folder = Path(jobs[arm]['attempt_dir'])
            paths += [folder / n for n in ('worker.json', 'request.json', 'entry.json')]
            if arm != 'A_scratch':
                paths += [root / 'pretrain' / arm / n for n in ('protocol.json', 'completion.json')]
                old = Path(ready['links'][arm]['source_worker']).parent
                paths += [old / n for n in ('worker.json', 'request.json', 'entry.json')]
                paths.append(root / 'preparation' / ('capacity_' + arm + '.json'))
        included, excluded = [], []
        optional = {'PLAN.md', 'README.md', 'LAUNCH.md', 'single_method_comparison.csv', 'inference_timing.csv'}
        for path in dict.fromkeys(paths):
            if not path.exists() and path.name in optional:
                continue
            if (path.is_symlink() or not path.is_file() or path.resolve() != path
                    or not path.is_relative_to(finish.GIT_ROOT.resolve())
                    or path.suffix in ('.pt', '.npz', '.npy')):
                raise ValueError('Migration publication requires exact ordinary small artifact: ' + str(path))
            if path.stat().st_size > finish.SMALL_LIMIT:
                if path == root / 'events.csv':
                    excluded.append(dict(path=str(path.relative_to(finish.GIT_ROOT)), reason='large event CSV retained locally',
                        bytes=path.stat().st_size, sha256=queue.digest(path)))
                    continue
                raise ValueError('Migration small-file publication limit exceeded: ' + str(path))
            included.append(path)
        if sum(p.stat().st_size for p in included) > finish.TOTAL_LIMIT:
            raise ValueError('Migration bounded publication total exceeded; no staging')
        return included, excluded

    finish.completion_evidence = completion_evidence
    finish.frozen_sources = frozen_sources
    finish.science_boundary = science_boundary
    finish.publication_paths = publication_paths
    return finish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'analyze', 'wait', 'publish'))
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--timeout-hours', type=float, default=336)
    args = parser.parse_args(argv)
    if args.publish and args.command != 'wait':
        parser.error('--publish only belongs to wait')
    finish = hooks()
    if args.command == 'status':
        try:
            _, jobs = finish.completion_evidence(ROOT, plan_path=queue.PLAN)
            value = dict(complete=True, jobs=jobs, launches_work=False)
        except finish.NotReady as error:
            value = dict(complete=False, reason=str(error), launches_work=False)
    elif args.command == 'analyze':
        value = finish.analyze(ROOT, plan_path=queue.PLAN)
    elif args.command == 'publish':
        value = finish.publish(ROOT)
    else:
        value = finish.wait(ROOT, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds,
            publish_after=args.publish, plan_path=queue.PLAN)
    print(json.dumps({k: v for k, v in value.items() if k not in ('inputs', 'outputs', 'checkpoints', 'allowlist')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
