"""Exactly two small CPU fixtures; Git and publication transport are mocked."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import r114_finish as finish


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    finish.common.write(path, value)


def seal(value, key):
    value[key] = finish.common.json_hash(value)
    return value


class Fixture:
    def __init__(self, directory, status='complete'):
        self.repo = Path(directory)
        (self.repo / '.git').mkdir()
        self.root = self.repo / finish.ROOT_REL
        self.v4 = self.root.parent.parent
        self.prep = self.root / 'preparation'
        self.queue = self.prep / 'grouped_execution/queue_v1'
        self.run = self.queue / 'scheduler_runs/run_0001'
        for name in finish.CODE_NAMES:
            path = self.v4 / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Synthetic fixture only; not a runtime or solver.\n')
        for name in finish.PREPARATION_FILES:
            write(self.prep / name, dict(synthetic_test_only=True))
        self.helper = self.v4 / 'r112_finish.py'
        self.helper.write_text('# Read-only helper fixture, separately tracked.\n')
        pools = {p: [dict(method='BQ', global_id=0), dict(method='MVMOE', global_id=1)] for p in ('CVRP', 'VRPB')}
        selection = seal(dict(version='synthetic_test_only', split='train', data_root='/server/frozen/train',
            sources={p: dict(dataset='/server/' + p, dataset_sha256='declared_not_rehashed') for p in pools},
            bases=[dict(problem=p, base_id=p + '_base') for p in pools]), 'selection_sha256')
        write(self.prep / 'base_selection.json', selection)
        endpoints, pairs, tasks = [], [], []
        for p in pools:
            for arm in ('original', 'directed', 'random'):
                key = p + '_' + arm
                endpoints.append(dict(endpoint_id=key, base_id=p + '_base', problem=p,
                    input_path='/server/inputs/' + key + '.json', input_sha256='frozen_endpoint_hash'))
                tasks.extend(dict(task_id=key + '__' + c['method'], problem=p) for c in pools[p])
                if arm != 'original':
                    pairs.append(dict(pair_id=key, problem=p, original_id=p + '_original', changed_id=key))
        self.manifest = seal(dict(version='synthetic_test_only', split='train', selection_sha256=selection['selection_sha256'],
            selected_problems=list(pools), pools=pools, endpoints=endpoints, pairs=pairs, tasks=tasks), 'manifest_sha256')
        self.manifest_path = self.prep / 'production_inputs/manifest.json'
        write(self.manifest_path, self.manifest)
        deployments = {p + '/' + c['method']: dict(backend='synthetic_test_only', budget=dict(decoding='fixture'),
            checkpoints=[], profile_sha256='fixture') for p in pools for c in pools[p]}
        self.reference = seal(dict(deployments=deployments), 'lock_sha256')
        self.reference_path = self.prep / 'production_execution/deployments.production.lock.json'
        write(self.reference_path, self.reference)
        self.lock = seal(dict(manifest_sha256=self.manifest['manifest_sha256'], deployments=deployments,
            execution=dict(old_cost_reuse=False, batch_size=1), artifacts={}, runtime=dict(fixture=True)), 'lock_sha256')
        self.lock_path = self.queue / 'deployments.grouped.lock.json'
        write(self.lock_path, self.lock)
        jobs = [dict(job_id='runtime_group__' + p + '__' + c['method'], problem=p, method=c['method'])
                for p in pools for c in pools[p]]
        self.plan = seal(dict(queue_root=str(self.queue), manifest_path=str(self.manifest_path),
            manifest_sha256=self.manifest['manifest_sha256'], deployment_lock_path=str(self.lock_path),
            lock_sha256=self.lock['lock_sha256'], reference_lock_path=str(self.reference_path),
            reference_lock_sha256=self.reference['lock_sha256'], runtime_jobs=jobs,
            label_jobs=[dict(j, job_id=j['job_id'].replace('runtime_group', 'label_group')) for j in jobs],
            route_source_blocked_problems=[]), 'plan_sha256')
        write(self.queue / 'plan.json', self.plan)
        write(self.queue / 'route_sources.json', dict(source_only=True))
        write(self.prep / 'grouped_execution/READY_RECEIPT.json', dict(plan_path=str(self.queue / 'plan.json'),
            plan_sha256=self.plan['plan_sha256'], source_sha256={n: finish.common.digest(self.v4 / n)
                for n in finish.CODE_NAMES if 'finish' not in n}))
        write(self.queue / 'execution_binding.json', seal(dict(plan_sha256=self.plan['plan_sha256'],
            gpu_uuid='GPU-synthetic-fixture-only', cpu_ids=[0], host='fixture'), 'binding_sha256'))
        self.labels_path = self.run / 'labels.json'
        self.analysis = self.run / 'response_analysis'
        self.analysis.mkdir(parents=True)
        for name in finish.ANALYSIS_FILES:
            path = self.analysis / name
            if name.endswith('.pdf'):
                path.write_bytes(b'%PDF-1.4\nSynthetic signature fixture only.\n%%EOF\n')
            elif name.endswith('.csv'):
                path.write_text('problem,family,matched_full_pool_bases\nCVRP,demand,1\n')
            else:
                path.write_text('# Synthetic read-only response diagnosis\n')
        self.set_terminal(status)

    def set_terminal(self, status):
        partial = status != 'complete'
        rows = []
        for e in self.manifest['endpoints']:
            valid = not partial or e['problem'] != 'VRPB'
            rows.append(dict(endpoint_id=e['endpoint_id'], base_id=e['base_id'], problem=e['problem'],
                methods=['BQ', 'MVMOE'], global_ids=[0, 1], costs_fp64=[1., 2.] if valid else [None, None],
                valid_mask=[valid, valid], winner_mask=valid, winner='BQ' if valid else None,
                winner_global_id=0 if valid else None))
        summary = dict(endpoint_count=6, pair_count=4, verified_candidate_costs=6 if partial else 12,
                       full_pool_winners=3 if partial else 6, full_pool_pairs=2 if partial else 4)
        labels = dict(version=self.manifest['version'], manifest_sha256=self.manifest['manifest_sha256'],
            lock_sha256=self.lock['lock_sha256'], endpoints=rows,
            pairs=[dict(pair_id=p['pair_id'], winner_pair_mask=not partial or p['problem'] != 'VRPB') for p in self.manifest['pairs']],
            summary=summary)
        write(self.labels_path, labels)
        write(self.analysis / 'analysis_complete.json', dict(experiment='R114_response_diagnosis', complete=True,
            manifest_path=str(self.manifest_path), manifest_sha256=finish.common.digest(self.manifest_path),
            labels_path=str(self.labels_path), labels_sha256=finish.common.digest(self.labels_path),
            planned_pairs=4, full_pool_pairs=summary['full_pool_pairs'], model_training=False,
            files={n: finish.common.digest(self.analysis / n) for n in finish.ANALYSIS_FILES}))
        self.status = dict(status=status, execution=True, complete=not partial, run_directory=str(self.run),
            plan_sha256=self.plan['plan_sha256'], manifest_sha256=self.manifest['manifest_sha256'],
            lock_sha256=self.lock['lock_sha256'], stopped_problems=['VRPB'] if partial else [],
            error='fixture interruption' if status == 'interrupted' else None,
            committed_label_receipts=summary['verified_candidate_costs'],
            labels=dict(path=str(self.labels_path), summary=summary),
            analysis=dict(status='complete', output=str(self.analysis)))
        write(self.run / 'result.json', self.status)
        write(self.queue / 'scheduler_progress.json', self.status)
        if partial:
            write(self.queue / 'attempts/runtime_group__VRPB__BQ/attempt_0001/worker_result.json',
                dict(error=dict(stage='grouped_singletons', task_id='VRPB_original__BQ',
                                error='route_unavailable: real tour missing')))


class FakeGit:
    def __init__(self, fixture):
        self.repo = fixture.repo
        self.parent = self.head = self.remote = 'dynamic-parent-after-other-watchers'
        self.staged, self.changed, self.calls = set(), set(), []
        self.fail_push = False

    def __call__(self, *args, **_kwargs):
        self.calls.append(args)
        if args[:2] == ('rev-parse', '--absolute-git-dir'):
            return str(self.repo / '.git')
        if args[:2] == ('rev-parse', '--show-toplevel'):
            return str(self.repo)
        if args[:2] == ('diff', '--cached'):
            return '\0'.join(sorted(self.staged))
        if args[:2] == ('diff', 'HEAD'):
            return ''
        if args[0] == 'symbolic-ref':
            return 'main'
        if args[0] == 'config':
            return 'origin' if args[-1].endswith('.remote') else 'refs/heads/main'
        if args[0] == 'ls-remote':
            return self.remote + '\trefs/heads/main'
        if args[0] == 'ls-files':
            return '\0'.join(args[args.index('--') + 1:])
        if args[0] == 'rev-parse':
            name = args[1]
            if name == 'HEAD':
                return self.head
            if name == 'HEAD^':
                return self.parent
            return 'blob-' + finish.common.digest(self.repo / name.split(':', 1)[1])
        if args[0] == 'add':
            self.staged = set(args[args.index('--') + 1:])
            return ''
        if args[0] == 'commit':
            self.changed, self.staged = set(self.staged), set()
            self.head = 'single-r114-commit'
            return self.head
        if args[0] == 'diff-tree':
            return '\n'.join(sorted(self.changed))
        if args[0] == 'push':
            if self.fail_push:
                raise RuntimeError('synthetic uncertain push; no retry')
            self.remote = self.head
            return 'mock push complete'
        raise AssertionError('Unexpected Git operation ' + repr(args))


class FinishTests(unittest.TestCase):
    def patches(self, fixture, fake, stack):
        stack.enter_context(patch.object(finish.common, 'GIT_ROOT', fixture.repo))
        stack.enter_context(patch.object(finish, 'git', side_effect=fake))
        stack.enter_context(patch.object(finish.common, 'git', side_effect=fake))
        stack.enter_context(patch.object(finish, 'dependency_files', side_effect=lambda roots: [*roots, fixture.helper]))

    def test_complete_wait_exact_allowlist_dynamic_parent_and_one_publication(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            fixture = Fixture(directory)
            fake = FakeGit(fixture)
            self.patches(fixture, fake, stack)
            write(fixture.queue / 'scheduler_progress.json', dict(fixture.status, status='running'))
            with self.assertRaises(finish.NotReady):
                finish.completed(fixture.root)
            self.assertFalse((fixture.prep / 'finish').exists())
            self.assertEqual(fake.calls, [])
            with patch.object(finish.time, 'sleep', side_effect=lambda _: write(fixture.queue / 'scheduler_progress.json', fixture.status)):
                delivery = finish.wait(fixture.root, timeout_hours=.01, poll_seconds=.001)
            self.assertTrue(delivery['labeling_complete'])
            self.assertEqual(fake.calls, [])
            closeout = finish.common.read(Path(delivery['output']) / 'closeout.json')
            self.assertEqual(closeout['actual_verified_labels'], 12)
            self.assertEqual(closeout['full_pool_winners'], 6)
            fake.staged = {'unrelated.txt'}
            with self.assertRaisesRegex(ValueError, 'Pre-existing staged'):
                finish.publication_preflight(fixture.root, delivery)
            fake.staged.clear()
            fake.remote = 'other-head'
            with self.assertRaisesRegex(ValueError, 'Dynamic remote'):
                finish.publication_preflight(fixture.root, delivery)
            fake.remote = fake.head
            receipt = finish.publish(fixture.root)
            self.assertEqual(receipt['state'], 'complete')
            self.assertEqual(receipt['parent'], fake.parent)
            self.assertEqual(receipt['commit_attempts'], 1)
            self.assertEqual(receipt['push_attempts'], 1)
            self.assertEqual(receipt['global_mutex'], str(fixture.repo / '.git/codex-publication.lock'))
            self.assertNotIn(str(fixture.manifest_path.relative_to(fixture.repo)), receipt['allowlist'])
            self.assertNotIn(str(fixture.labels_path.relative_to(fixture.repo)), receipt['allowlist'])
            self.assertTrue(all('inputs/' not in n and 'attempts/' not in n and not n.endswith('.pt') for n in receipt['allowlist']))
            self.assertTrue(any(n.endswith('server_artifacts.sha256.json') for n in receipt['allowlist']))
            self.assertTrue(any(n.endswith('RESPONSE_DIAGNOSIS.pdf') for n in receipt['allowlist']))
            finish.publish(fixture.root)
            self.assertEqual(sum(c[0] == 'commit' for c in fake.calls), 1)
            self.assertEqual(sum(c[0] == 'push' for c in fake.calls), 1)
            self.assertTrue(all('-A' not in c for c in fake.calls))

    def test_partial_negative_closeout_masks_analysis_gate_and_uncertain_push_never_retries(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            fixture = Fixture(directory, 'blocked')
            fake = FakeGit(fixture)
            self.patches(fixture, fake, stack)
            analysis_receipt = fixture.analysis / 'analysis_complete.json'
            saved = analysis_receipt.read_bytes()
            analysis_receipt.unlink()
            with self.assertRaises(finish.NotReady):
                finish.completed(fixture.root)
            analysis_receipt.write_bytes(saved)
            for status in ('blocked', 'failed', 'interrupted'):
                fixture.set_terminal(status)
                delivery = finish.prepare(fixture.root)
                closeout = finish.common.read(Path(delivery['output']) / 'closeout.json')
                self.assertFalse(delivery['labeling_complete'])
                self.assertEqual(delivery['closeout_kind'], 'partial_negative_closeout')
                self.assertEqual(closeout['actual_verified_labels'], 6)
                self.assertEqual(closeout['full_pool_winners'], 3)
                self.assertEqual(closeout['full_pool_pairs'], 2)
                self.assertFalse(closeout['all18_model_performance_claim'])
                text = (Path(delivery['output']) / 'RESULTS.md').read_text()
                self.assertIn('NOT labeling finished', text)
                self.assertIn('route_unavailable', text)
            bad = finish.common.read(fixture.labels_path)
            bad['endpoints'][-1]['winner_mask'] = True
            write(fixture.labels_path, bad)
            with self.assertRaisesRegex(ValueError, 'full-pool winner'):
                finish.validate(finish.completed(fixture.root))
            fixture.set_terminal('interrupted')
            fake.fail_push = True
            with self.assertRaisesRegex(RuntimeError, 'uncertain push'):
                finish.publish(fixture.root)
            with self.assertRaisesRegex(RuntimeError, 'no automatic'):
                finish.publish(fixture.root)
            self.assertEqual(sum(c[0] == 'commit' for c in fake.calls), 1)
            self.assertEqual(sum(c[0] == 'push' for c in fake.calls), 1)
            receipt = finish.common.read(fixture.prep / 'finish/publish_receipt.json')
            self.assertEqual(receipt['state'], 'failed')
            self.assertFalse(receipt['labeling_complete'])
            self.assertTrue(receipt['manual_recovery_required'])


if __name__ == '__main__':
    unittest.main()
