"""Synthetic CPU fixtures test integration rules, never experimental evidence."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import r114_recovery_finish as finish


def seal(value, field):
    value = copy.deepcopy(value)
    value.pop(field, None)
    value[field] = finish.common.json_hash(value)
    return value


def manifest_fixture():
    manifest = dict(version='synthetic_not_experiment', split='train', schema_version=1,
        selection_sha256='fixture-selection', selected_problems=['CVRP', 'OVRP'],
        pools={p: [dict(method=f'm{i}', global_id=i, source_method=f'm{i}') for i in range(n)]
               for p, n in (('CVRP', 10), ('OVRP', 2))},
        endpoints=[], pairs=[], tasks=[], rejections=[])
    for problem, bases in (('CVRP', 32), ('OVRP', 1)):
        for index in range(bases):
            base = f'{problem}-base-{index}'
            for arm in ('original', 'directed', 'random'):
                endpoint = dict(endpoint_id=f'{base}-{arm}', base_id=base, problem=problem,
                    source_index=index, input_sha256='input-' + base + arm,
                    input_path='/synthetic/' + base + arm, arm=arm)
                manifest['endpoints'].append(endpoint)
                for candidate in manifest['pools'][problem]:
                    manifest['tasks'].append(dict(task_id=endpoint['endpoint_id'] + candidate['method'],
                        endpoint_id=endpoint['endpoint_id'], base_id=base, problem=problem,
                        input_sha256=endpoint['input_sha256'], method=candidate['method'], solver_seed=index))
                if arm != 'original':
                    manifest['pairs'].append(dict(pair_id=endpoint['endpoint_id'], problem=problem,
                        base_id=base, original_id=base + '-original', changed_id=endpoint['endpoint_id'],
                        family='demand', arm=arm, l1=1., moved_customers=1))
    for key in ('endpoint', 'pair', 'task'):
        manifest[key + '_count'] = len(manifest[key + 's'])
    return seal(manifest, 'manifest_sha256')


def subset_fixture(manifest):
    child = copy.deepcopy(manifest)
    child.update(selected_problems=['CVRP'], pools={'CVRP': child['pools']['CVRP']},
        acquisition_revision=finish.REVISION, parent_manifest_sha256=manifest['manifest_sha256'])
    for key in ('endpoints', 'pairs', 'tasks', 'rejections'):
        child[key] = [r for r in child[key] if r['problem'] == 'CVRP']
    for key in ('endpoint', 'pair', 'task'):
        child[key + '_count'] = len(child[key + 's'])
    return seal(child, 'manifest_sha256')


def labels_fixture(manifest, lock, *, failed_parent=False, missing=None):
    endpoints = []
    for endpoint in manifest['endpoints']:
        pool = sorted(manifest['pools'][endpoint['problem']], key=lambda r: r['global_id'])
        valid = [not (failed_parent and endpoint['problem'] == 'CVRP') for _ in pool]
        if endpoint['endpoint_id'] == missing:
            valid[-1] = False
        offset = {'original': 0., 'directed': .1, 'random': .2}[endpoint['arm']]
        costs = [i + 1. + offset if v else None for i, v in enumerate(valid)]
        full = all(valid)
        endpoints.append(dict(endpoint_id=endpoint['endpoint_id'], base_id=endpoint['base_id'],
            problem=endpoint['problem'], source_index=endpoint['source_index'],
            methods=[r['method'] for r in pool], global_ids=[r['global_id'] for r in pool],
            valid_mask=valid, failure_mask=[not v for v in valid], costs_fp64=costs,
            winner_mask=full, winner=pool[0]['method'] if full else None,
            winner_global_id=pool[0]['global_id'] if full else None))
    rows = {r['endpoint_id']: r for r in endpoints}
    pairs = []
    for proposal in manifest['pairs']:
        old, new = rows[proposal['original_id']], rows[proposal['changed_id']]
        valid = [a and b for a, b in zip(old['valid_mask'], new['valid_mask'])]
        matrix = [[a and b for b in valid] for a in valid]
        full = all(valid)
        delta = [[((new['costs_fp64'][i] - new['costs_fp64'][j]) -
                   (old['costs_fp64'][i] - old['costs_fp64'][j])) if matrix[i][j] else None
                  for j in range(len(valid))] for i in range(len(valid))]
        pairs.append(dict(proposal, methods=old['methods'], candidate_change_mask=valid,
            candidate_change=[new['costs_fp64'][i] - old['costs_fp64'][i] if v else None
                              for i, v in enumerate(valid)],
            failure_mask=[not v for v in valid], Delta_mask=matrix, Delta_mn=delta,
            original_winner=old['winner'], changed_winner=new['winner'],
            winner_pair_mask=full, winner_flip=False if full else None))
    return dict(version=manifest['version'], manifest_sha256=manifest['manifest_sha256'],
        lock_sha256=lock, endpoints=endpoints, pairs=pairs,
        summary=dict(endpoint_count=len(endpoints), pair_count=len(pairs),
            verified_candidate_costs=sum(sum(r['valid_mask']) for r in endpoints),
            full_pool_winners=sum(r['winner_mask'] for r in endpoints),
            full_pool_pairs=sum(r['winner_pair_mask'] for r in pairs)))


class CombinedCloseoutTests(unittest.TestCase):
    def setUp(self):
        self.parent = manifest_fixture()
        self.child = subset_fixture(self.parent)
        self.original = labels_fixture(self.parent, 'original-lock', failed_parent=True)
        self.corrected = labels_fixture(self.child, 'corrected-lock')
        self.provenance = {key: dict(lock_sha256=lock, manifest_sha256=manifest['manifest_sha256'],
            full_manifest_file=dict(path='/synthetic/' + key, sha256='file-hash-' + key))
            for key, lock, manifest in (('parent', 'original-lock', self.parent),
                                        ('corrected_CVRP', 'corrected-lock', self.child))}

    def merge(self, corrected=None):
        return finish.merge_labels(self.parent, self.child, self.original,
                                   corrected or self.corrected, self.provenance)

    def test_exact_identity_replace_only_cvrp_and_preserve_inputs(self):
        original = copy.deepcopy(self.original)
        merged = self.merge()
        for field, key in (('endpoints', 'endpoint_id'), ('pairs', 'pair_id')):
            corrected = {r[key]: r for r in self.corrected[field]}
            for before, after in zip(self.original[field], merged[field]):
                self.assertEqual(after, corrected[before[key]] if before['problem'] == 'CVRP' else before)
        self.assertEqual(self.original, original)
        for section, key in (('tasks', 'solver_seed'), ('endpoints', 'input_sha256'), ('pairs', 'l1')):
            changed = copy.deepcopy(self.child)
            changed[section][0][key] = 'changed'
            changed = seal(changed, 'manifest_sha256')
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, 'exact unchanged'):
                finish.check_subset(self.parent, changed)
        changed = copy.deepcopy(self.corrected)
        changed['endpoints'][0]['source_index'] = -1
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.merge(changed)

    def test_partial_costs_keep_null_masks_and_no_synthetic_winner(self):
        missing = self.child['endpoints'][1]['endpoint_id']
        partial = labels_fixture(self.child, 'corrected-lock', missing=missing)
        merged = self.merge(partial)
        endpoint = next(r for r in merged['endpoints'] if r['endpoint_id'] == missing)
        self.assertIsNone(endpoint['costs_fp64'][-1])
        self.assertIsNone(endpoint['winner'])
        self.assertFalse(endpoint['winner_mask'])
        totals, _ = finish.check_assembly(self.parent, merged, None)
        self.assertEqual(totals['verified_candidate_costs'], len(self.parent['tasks']) - 1)
        tables = finish.analysis.build_tables(self.parent, merged)
        pair = next(r for r in tables['pairs'] if r['pair_id'] == missing)
        self.assertFalse(pair['full_pool_pair'])
        self.assertIsNone(pair['winner_flip'])
        self.assertIsNone(pair['rms_delta_pct'])
        changed = copy.deepcopy(partial)
        changed['endpoints'][1]['costs_fp64'][-1] = 0.
        with self.assertRaisesRegex(ValueError, 'remain null'):
            self.merge(changed)
        changed = copy.deepcopy(partial)
        changed['pairs'][0]['winner_pair_mask'] = True
        with self.assertRaisesRegex(ValueError, 'masks disagree'):
            self.merge(changed)

    def test_separate_lock_provenance_readiness_and_no_parent_publication(self):
        merged = self.merge()
        self.assertIsNone(merged['lock_sha256'])
        self.assertEqual(merged['execution_lock'], {'CVRP': 'corrected-lock', 'OVRP': 'original-lock'})
        self.assertEqual(merged['execution_provenance'], self.provenance)
        self.assertEqual(merged['per_problem_provenance'], {'CVRP': 'corrected_CVRP', 'OVRP': 'parent'})
        self.assertEqual(merged['manifest_sha256'], self.parent['manifest_sha256'])
        changed = copy.deepcopy(self.provenance)
        changed['corrected_CVRP']['lock_sha256'] = 'original-lock'
        with self.assertRaises(ValueError):
            finish.merge_labels(self.parent, self.child, self.original, self.corrected, changed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(finish.NotReady):
                finish.child_completed(root, root / finish.RECOVERY / 'acquisition_v1/queue/plan.json')
            with patch.object(finish.parent, 'prepare') as baseline, patch.object(finish.parent, 'publish') as publisher:
                with patch.object(finish, 'completed', side_effect=finish.NotReady('labels absent')):
                    with self.assertRaises(finish.NotReady):
                        finish.prepare(root, root / 'plan.json')
                with self.assertRaises(finish.NotReady):
                    finish.publish(root, root / 'plan.json')
                baseline.assert_not_called()
                publisher.assert_not_called()
            progress = root / 'parent_progress.json'
            child_progress = root / 'child_progress.json'
            finish.common.write(progress, dict(status='blocked', synthetic=True))
            finish.common.write(child_progress, dict(status='complete', synthetic=True))
            output = root / finish.OUTPUT
            output.mkdir(parents=True)
            totals, counts = finish.check_assembly(self.parent, merged, None)
            closeout = dict(kind='complete_labels', labeling_complete=True, totals=totals,
                actual_verified_labels=totals['verified_candidate_costs'],
                planned_label_calls=len(self.parent['tasks']), per_problem=counts,
                hardware=dict(binding=dict(gpu_uuid='GPU-SYNTHETIC-NOT-EVIDENCE')))
            with patch.dict('os.environ', {'MPLCONFIGDIR': str(root / 'mpl_cache')}):
                finish.render_pdf(output / 'RESULTS.pdf', closeout, self.provenance, [])
            plan = root / 'plan.json'
            receipt = dict(closeout_ready=True, output=str(output), child_plan=str(plan),
                kind='complete_labels', closeout_kind='complete_labels', labeling_complete=True,
                generator_sha256=finish.common.digest(finish.__file__), source_sha256={},
                inputs={str(p): finish.common.digest(p) for p in (progress, child_progress)},
                outputs={'RESULTS.pdf': finish.common.digest(output / 'RESULTS.pdf')},
                baseline_output=str(root / 'baseline'))
            finish.common.write(output / 'finish_complete.json', receipt)
            with patch.object(finish, 'completed', return_value=(dict(progress_path=progress),
                    dict(progress_path=child_progress, plan_path=plan), {}, {})), \
                    patch.object(finish.parent, 'verify_delivery'), \
                    patch.object(finish, 'source_bindings', return_value={}):
                self.assertEqual(finish.verify_delivery(root, plan), receipt)
                finish.common.write(progress, dict(status='resumed', synthetic=True))
                with self.assertRaisesRegex(ValueError, 'Scheduler changed'):
                    finish.verify_delivery(root, plan)
            # Synthetic Docker cgroups need not contain Slurm's outer job path.
            folder = root / finish.RECOVERY / 'launch_3090'
            folder.mkdir(parents=True)
            container = 'synthetic-container-not-evidence'
            binding = dict(host=container[:12], gpu_uuid='GPU-SYNTHETIC-NOT-EVIDENCE',
                           cpu_ids=[0, 1], timeout_seconds=3600)
            child = dict(plan_path=plan, progress_path=child_progress, binding=binding,
                plan=dict(manifest_path='/synthetic/manifest', deployment_lock_path='/synthetic/lock'),
                status=dict(status='complete'))
            argv = ['/synthetic/python', '-m', 'code.V4.r114_run_queue', '--plan', str(plan),
                '--manifest', '/synthetic/manifest', '--deployment-lock', '/synthetic/lock',
                '--host', binding['host'], '--gpu-uuid', binding['gpu_uuid'],
                '--cpu-ids', '0,1', '--timeout-seconds', '3600', '--execute', '--finalize']
            finish.common.write(folder / 'ready.json', dict(main_approved=True, slot='3090_1',
                plan=str(plan), gpu_uuid=binding['gpu_uuid'], sources={}, token='SYNTHETIC-SECRET'))
            finish.common.write(folder / 'request.json', dict(ready_sha256=finish.common.digest(folder / 'ready.json')))
            finish.common.write(folder / 'entry.json', dict(state='complete', exit_status=0,
                container=dict(id=container), cpus=[0, 1, 2, 3], cgroup='0::/slurm/job_1465/step_1'))
            worker = dict(state='complete', exit_status=0, initialized=True, actual_gpu_type='3090',
                gpu_uuid=binding['gpu_uuid'], acquisition_status='complete', progress_path=str(child_progress),
                argv=argv, resource_checks=dict(allocation=1465, uid=2022, container_id=container,
                    cgroup='0::/docker/' + container, cpus=[0, 1, 2, 3],
                    gpu=dict(uuid=binding['gpu_uuid'], name='SYNTHETIC RTX 3090')))
            finish.common.write(folder / 'worker.json', worker)
            finish.common.write(folder / 'controller.json', dict(state='complete', exit_status=0,
                gpu_uuid=binding['gpu_uuid'], worker_path=str(folder / 'worker.json')))
            hardware, inputs = finish.launch_evidence(root, child)
            self.assertEqual(hardware['binding'], binding)
            self.assertEqual(inputs[str(folder / 'ready.json')], finish.common.digest(folder / 'ready.json'))
            self.assertNotIn('SYNTHETIC-SECRET', str(hardware))
            worker['resource_checks']['gpu']['name'] = 'SYNTHETIC RTX 4090'
            finish.common.write(folder / 'worker.json', worker)
            with self.assertRaisesRegex(ValueError, 'actual RTX3090'):
                finish.launch_evidence(root, child)


if __name__ == '__main__':
    unittest.main()
