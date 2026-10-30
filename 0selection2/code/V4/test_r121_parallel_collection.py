"""CPU-only process-pool contract tests; no native session or GPU is created."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import r121_parallel_collection as parallel


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'parent'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(parallel.base.data, 'ROOT', Path(self.temporary.name)))
        self.stack.enter_context(mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        self.stack.enter_context(mock.patch.object(parallel, 'assert_no_old_producer'))
        self.stack.enter_context(mock.patch.object(parallel.time, 'sleep'))
        self.protocol = dict(mode='cached_native_v1', lanes=4, batch_size=1, native_budget_changes=False,
            deployment_seed_changes=False, group_order=[f'P{i:02d}/method' for i in range(89)],
            planned_labels=113920, plan_sha256='plan', collection_sha256='collection')
        self.groups = {key: [dict(task_id=key + f'_{i}', endpoint_id=key + '_original') for i in range(1280)]
                       for key in self.protocol['group_order']}
        self.manifest = dict(endpoints=[dict(endpoint_id=key + '_original', cell='original') for key in self.groups])
        self.inventory = dict(deployments={key: dict(status='ready', profile_sha256='profile') for key in self.groups})
        self.stack.enter_context(mock.patch.object(parallel.base, 'load', return_value=(self.protocol, self.manifest, self.inventory)))
        self.stack.enter_context(mock.patch.object(parallel.base, 'task_groups', return_value=self.groups))
        self.stack.enter_context(mock.patch.object(parallel.base, 'validate_receipt'))
        self.stack.enter_context(mock.patch.object(parallel.base.data, 'require_release', return_value={}))
        self.stack.enter_context(mock.patch.object(parallel.base.data, 'require_binding', return_value='binding'))
        self.binding = self.root / 'binding.json'
        self.fixture(self.binding, dict(cpu_ids=[0, 1, 2, 3]))
        self.release = self.root / 'release.json'
        self.fixture(self.release, {})
        for key in self.groups:
            self.fixture(self.root / 'preflight' / (parallel.base.slug(key) + '.json'),
                         dict(profile_sha256='profile', binding_sha256='binding'))

    @staticmethod
    def fixture(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def invoke(self, maximum=1280, describe=False):
        return parallel.run(self.root, 0, 2, maximum, 'cuda:0', self.release, self.binding, describe=describe)

    def fake_pool(self, failure_key=None):
        state = dict(live=set(), max_live=0, keys=[], budgets=[], phase='label')
        def spawn(command, **kwargs):
            key = parallel.option(command, '--deployment')
            self.assertNotIn(key, state['live'])
            state['live'].add(key)
            state['max_live'] = max(state['max_live'], len(state['live']))
            self.assertLessEqual(len(state['live']), 2)
            state['keys'].append(key)
            budget = int(parallel.option(command, '--max-jobs'))
            state['budgets'].append(budget)
            self.assertLessEqual(budget, 1280)
            self.assertEqual(command[command.index('-m') + 1], 'code.V4.r121_collection')
            self.assertEqual(parallel.option(command, '--phase'), 'label')
            self.assertEqual(parallel.option(command, '--device'), 'cuda:0')
            self.assertEqual(parallel.option(command, '--binding'), str(self.binding))
            self.assertEqual(parallel.option(command, '--release'), str(self.release))
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')
            polled, done = 0, False
            def poll():
                nonlocal polled, done
                polled += 1
                if polled == 1:
                    return None
                if done:
                    return 2 if key == failure_key else 0
                done = True
                state['live'].remove(key)
                result = Path(parallel.option(command, '--result'))
                if key == failure_key:
                    self.fixture(result, dict(status='unavailable', deployment=key, phase='label',
                        attempted=1, completed=0, reused=0, first_error='first native failure'))
                    self.fixture(self.root / 'unavailable' / (parallel.base.slug(key) + '.json'), dict(first_error='first native failure'))
                    return 2
                start = parallel.base.progress(self.root, key)['next_index']
                self.fixture(self.root / 'progress' / (parallel.base.slug(key) + '.json'),
                    dict(next_index=start + budget, completed=start + budget,
                         collection_sha256='collection', status='complete' if start + budget == 1280 else 'running'))
                self.fixture(result, dict(status='finite_budget_complete', deployment=key, phase='label',
                    attempted=budget, completed=budget, reused=0))
                return 0
            return SimpleNamespace(pid=len(state['keys']) + 1000, poll=poll, wait=lambda: poll())
        return state, spawn

    def test_pool_max_two_distinct_keys_fixed_order_and_full_pool_false(self):
        state, spawn = self.fake_pool()
        with mock.patch.object(parallel.subprocess, 'Popen', side_effect=spawn):
            result, code = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(state['max_live'], 2)
        self.assertEqual(state['keys'], self.protocol['group_order'][::4])
        self.assertEqual(len(state['keys']), len(set(state['keys'])))
        self.assertTrue(result['qualified_segment_complete'])
        self.assertFalse(result['full89_complete'])
        self.assertFalse(result['export_performed'])
        self.assertEqual(state['budgets'], [1280] * 23)

    def test_skip_recorded_failure_and_no_receipt_overwrite(self):
        key = self.protocol['group_order'][0]
        failure = self.root / 'unavailable' / (parallel.base.slug(key) + '.json')
        self.fixture(failure, dict(first_error='preserved first fatal'))
        original = failure.read_bytes()
        state, spawn = self.fake_pool()
        with mock.patch.object(parallel.subprocess, 'Popen', side_effect=spawn):
            result, code = self.invoke()
            before = (self.root / parallel.VERSION / 'lane0/controller.json').read_bytes()
            with self.assertRaises(FileExistsError):
                self.invoke()
        self.assertEqual(code, 0)
        self.assertNotIn(key, state['keys'])
        self.assertEqual(result['deployments'][key]['status'], 'needs_other_recovery_root')
        self.assertEqual(failure.read_bytes(), original)
        self.assertEqual((self.root / parallel.VERSION / 'lane0/controller.json').read_bytes(), before)
        self.assertEqual(len(result['group_order']), 89)
        self.assertFalse(result['full89_complete'])

    def test_new_failure_is_nonzero_first_error_and_no_new_scheduling(self):
        key = self.protocol['group_order'][0]
        state, spawn = self.fake_pool(failure_key=key)
        with mock.patch.object(parallel.subprocess, 'Popen', side_effect=spawn):
            result, code = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(state['keys'], self.protocol['group_order'][0:8:4])
        self.assertEqual(result['first_error']['first_error'], 'first native failure')
        self.assertFalse(result['qualified_segment_complete'])
        self.assertFalse(result['full89_complete'])

    def test_remaining_budget_and_small_prefix_not_false_segment_complete(self):
        key = self.protocol['group_order'][0]
        self.fixture(self.root / 'progress' / (parallel.base.slug(key) + '.json'),
                     dict(next_index=1279, completed=1279, collection_sha256='collection', status='running'))
        state, spawn = self.fake_pool()
        with mock.patch.object(parallel.subprocess, 'Popen', side_effect=spawn):
            result, code = self.invoke(maximum=3)
        self.assertEqual(code, 0)
        self.assertEqual(state['budgets'][0], 1)
        self.assertTrue(all(value <= 3 for value in state['budgets']))
        self.assertFalse(result['qualified_segment_complete'])
        self.assertFalse(result['full89_complete'])

    def test_unqualified_keys_are_not_run_and_describe_never_spawns(self):
        key = self.protocol['group_order'][0]
        (self.root / 'preflight' / (parallel.base.slug(key) + '.json')).unlink()
        with mock.patch.object(parallel.subprocess, 'Popen', side_effect=AssertionError('native launch')):
            result, code = self.invoke(describe=True)
        self.assertEqual(code, 0)
        self.assertEqual(result['deployments'][key]['status'], 'qualification_pending')
        self.assertFalse(result['solver_execution'])

    def test_binding_mismatch_and_invalid_finite_budget_fail_before_children(self):
        for maximum in (0, -1, 1281):
            with self.assertRaises(ValueError):
                self.invoke(maximum=maximum, describe=True)
        key = self.protocol['group_order'][0]
        self.fixture(self.root / 'preflight' / (parallel.base.slug(key) + '.json'),
                     dict(profile_sha256='profile', binding_sha256='other'))
        with mock.patch.object(parallel.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                self.invoke()
            spawn.assert_not_called()

    def test_old_same_lane_producer_blocks_before_controller_claim(self):
        with mock.patch.object(parallel, 'assert_no_old_producer', side_effect=parallel.base.NotReady('old worker active')), \
                mock.patch.object(parallel.subprocess, 'Popen') as spawn:
            with self.assertRaises(parallel.base.NotReady):
                self.invoke()
            spawn.assert_not_called()
        self.assertFalse((self.root / parallel.VERSION / 'lane0/controller.json').exists())

    def test_all_unqualified_never_claims_qualified_segment_or_full_pool(self):
        with mock.patch.object(parallel.base, 'validate_receipt'), \
                mock.patch.object(parallel.subprocess, 'Popen') as spawn:
            for key in self.protocol['group_order'][::4]:
                self.fixture(self.root / 'unavailable' / (parallel.base.slug(key) + '.json'), dict(first_error='known failure'))
            result, code = self.invoke()
            spawn.assert_not_called()
        self.assertEqual(code, 0)
        self.assertFalse(result['qualified_segment_complete'])
        self.assertFalse(result['full89_complete'])


if __name__ == '__main__':
    unittest.main()
