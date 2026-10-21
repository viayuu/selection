"""Small receipt/target/arithmetic checks; no real inference or training."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import r85_goal as goal


class R85GoalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.launch = self.base/'R85', self.base/'launch'

    def ready_arm(self, arm):
        run = self.root/arm
        for name, record in (
                ('status.json', dict(phase='training_complete', successful_updates=57600)),
                ('training_complete.json', dict(successful_updates=57600)),
                ('final_test_lock.json', dict(selected_without_test=True, per_problem_assembly=False,
                                             selected_update=1440, val=dict(macro_top1=.7))),
                ('test_results.json', dict(complete=True)),
                ('lib_stage_complete.json', dict(complete=True)),
                ('postprocess_status.json', dict(complete=True, phase='complete')),
                ('full_model_statistics/protocol.json', dict(complete=True))):
            goal.save(run/name, record)
        for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf'):
            (run/'full_model_statistics'/name).write_bytes(b'%PDF-fixture')

    def test_status_requires_all_artifacts_and_preserves_goal(self):
        for arm in goal.ARMS:
            self.ready_arm(arm)
        with patch.object(goal.receipts, 'file_hash', side_effect=AssertionError('poll must not hash predictions')):
            result = goal.status(self.root, self.launch)
        self.assertTrue(result['all_arms_ready'])
        self.assertFalse(result['goal_complete'])
        self.assertFalse(result['automatic_retry'])
        (self.root/goal.ARMS[2]/'lib_stage_complete.json').unlink()
        result = goal.status(self.root, self.launch)
        self.assertFalse(result['all_arms_ready'])
        goal.save(self.launch/goal.ARMS[1]/'worker.json', dict(state='failed', error='fixture exit'))
        self.assertTrue(goal.status(self.root, self.launch)['error_flags'])

    def test_frozen_published_targets_cannot_be_redefined(self):
        historical = self.base/'R84'/'A_baseline'
        identity = dict(baseline_checkpoint_sha256='checkpoint', baseline_state_hash='protocol', data_root='/new90k')
        targets = [dict(problem='TSP', method='LEHD', baseline_val_winners=3,
                        baseline_val_chosen=0, category='zero'),
                   dict(problem='CVRP', method='RouteFinder', baseline_val_winners=1,
                        baseline_val_chosen=1, category='near_zero')]
        goal.save(historical/'final_test_lock.json', dict(checkpoint_sha256='checkpoint', state_hash='protocol'))
        goal.save(historical/'protocol.json', dict(data_root='/new90k'))
        source = historical.parent/'comparison_artifacts/target_list.json'
        goal.save(source, dict(identity=identity, targets=targets))
        frozen = goal.freeze_targets(self.root, historical)
        self.assertEqual((frozen['groups'], frozen['baseline_val_winners']), (2, 4))
        self.assertFalse(frozen['affects_training'])
        self.assertEqual(goal.freeze_targets(self.root, historical), frozen)
        targets[0]['baseline_val_winners'] = 4
        goal.save(source, dict(identity=identity, targets=targets))
        with self.assertRaisesRegex(ValueError, 'cannot be redefined'):
            goal.freeze_targets(self.root, historical)

    def test_real_report_helpers_count_rescues_and_false_picks_on_two_tasks(self):
        import numpy as np
        from . import r85_reporting as api
        def prediction(selected):
            return SimpleNamespace(pool=['m0', 'm1'], pool_ids=np.array([0,1]),
                                   costs=np.array([[2.,1.],[1.,2.],[1.,3.]], dtype=np.float64),
                                   winner=np.array([1,0,0]), pred=np.array(selected),
                                   indices=np.arange(3), source_indices=np.arange(3), n=3)
        old = {p: prediction([0,0,0]) for p in ('TSP','CVRP')}
        new = {p: prediction([1,1,0]) for p in ('TSP','CVRP')}
        targets = [(p,'m1') for p in old]
        with patch.object(goal, 'PROBLEMS', ('TSP','CVRP')), patch.object(api, 'PROBLEMS', ('TSP','CVRP')):
            rows = api.fixed_target_recovery(old,new,targets,'R85B','R84A','VAL')
            summary = goal.summarize_targets(rows,'R85B','R84A','VAL','zero_near_zero')
            pairs = api.paired_comparison(old,new,'R85B','R84A','VAL')
        self.assertEqual((summary['rescued'], summary['new_false_pick_count']), (2,2))
        self.assertEqual(summary['precision_pct'],50.)
        self.assertEqual(summary['recall_pct'],100.)
        total = pairs[-1]
        self.assertEqual((total['corrected'],total['harmed']),(2,2))
        self.assertEqual(total['actual_regret_delta_pp'],0.)

    def test_unrounded_validation_gate(self):
        baseline = dict(top1_pct=71., mean_cost=11., actual_regret_pct=.4)
        tiny = dict(baseline, top1_pct=71.+1e-10)
        self.assertTrue(goal.passes(tiny,baseline))
        self.assertFalse(goal.passes(dict(tiny,actual_regret_pct=.4000000001),baseline))
        self.assertFalse(goal.passes(baseline,baseline))

    def test_error_stops_watcher_without_retry_or_report(self):
        with patch.object(goal, 'status', return_value=dict(errors=[dict(error='fixture')])) as status, \
                patch.object(goal, 'report') as report, patch.object(goal.time, 'sleep') as sleep:
            self.assertEqual(goal.watch(self.root,self.launch,max_polls=1),1)
        self.assertEqual(status.call_count,1)
        report.assert_not_called(); sleep.assert_not_called()
        receipt = goal.read(self.root/'goal_watch_receipt.json')
        self.assertEqual(receipt['state'],'stopped_on_error')
        self.assertFalse(receipt['autonomous_training'])


if __name__ == '__main__':
    unittest.main()
