"""Small CPU checks for the one-attempt, full-pool continuation barrier."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import r121_followup as api


class FollowupTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        contract = dict(collection_root=str(root), planned_labels=6,
            deployment_counts={'TSP/BQ': 3, 'CVRP/BQ': 3})
        for key in contract['deployment_counts']:
            slug = key.replace('/', '__')
            api.write(root / 'preflight' / (slug + '.json'), {'status': 'native_success'})
            api.write(root / 'progress' / (slug + '.json'), {'completed': 3})
        for lane in range(4):
            api.write(root / 'bootstrap' / ('lane' + str(lane) + '.json'), {'complete': True})
        return contract

    def test_complete_requires_every_deployment_and_lane(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = self.fixture(directory)
            self.assertTrue(api.label_progress(contract)['full89_complete'])
            Path(directory, 'bootstrap/lane3.json').unlink()
            self.assertFalse(api.label_progress(contract)['full89_complete'])

    def test_partial_new_root_is_not_completed_by_old_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self.fixture(root / 'new')
            contract['progress_source_roots'] = [str(root / 'old')]
            api.write(root / 'new/progress/TSP__BQ.json', {'completed': 2})
            self.fixture(root / 'old')
            result = api.label_progress(contract)
            self.assertFalse(result['full89_complete'])
            self.assertEqual(result['completed_labels'], 5)
            self.assertEqual(result['collected_unique_labels'], 6)

    def test_truncated_live_cursor_is_reobserved_not_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = self.fixture(directory)
            Path(directory, 'progress/TSP__BQ.json').write_text('{')
            result = api.label_progress(contract)
            self.assertFalse(result['full89_complete'])
            self.assertEqual(result['failures'], [])
            self.assertEqual(len(result['observation_uncertainty']), 1)

    def test_qualification_and_failure_are_not_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = self.fixture(directory)
            Path(directory, 'preflight/TSP__BQ.json').unlink()
            self.assertFalse(api.label_progress(contract)['full89_complete'])
            api.write(Path(directory, 'unavailable/CVRP__BQ.json'), {'error': 'native failure'})
            self.assertEqual(len(api.label_progress(contract)['failures']), 1)

    def test_no_stages_run_before_full_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api.write(root / 'contract.json', {'fixture': True})
            waiting = dict(full89_complete=False, failures=[], completed_labels=5)
            with patch.object(api, 'verify_sources'), patch.object(api, 'label_progress', return_value=waiting), \
                    patch.object(api.time, 'sleep', side_effect=KeyboardInterrupt), patch.object(api, 'command') as execute:
                with self.assertRaises(KeyboardInterrupt):
                    api.run(root, 60.)
                execute.assert_not_called()
            self.assertFalse(json.loads((root / 'failure.json').read_text())['automatic_retry'])

    def test_resource_plan_keeps_one_root_and_explicit_stages(self):
        contract = dict(learner_root=str(api.LEARNER), micro_cap=352, eval_batch_size=256,
            lib_batch_size=1, approved_by='main fixture', approved_utc='2026-10-29T00:00:00+00:00',
            source_sha256={'fixture': '0' * 64}, slot='gpu05_1')
        plan = api.resource_plan(contract, {'path': 'code/V4/fixture.json', 'sha256': '0' * 64})
        job = plan['jobs'][0]
        self.assertEqual(job['slot'], 'gpu05_1')
        self.assertEqual([s['id'] for s in job['stages']],
            ['train', 'test', 'lib', 'locked_train_and_family_capture'])
        for stage in job['stages']:
            self.assertEqual(stage['argv'].count('--root'), 1)
            self.assertEqual(stage['argv'][stage['argv'].index('--root') + 1], str(api.LEARNER))
            self.assertNotIn('--resume', stage['argv'])
            self.assertNotIn('--auto-retry', stage['argv'])


if __name__ == '__main__':
    unittest.main()
