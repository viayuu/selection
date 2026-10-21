"""Focused report population/alias/plot tests; no TEST files or inference."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from code.unified_selector.registry import S2I
from . import r84_reporting as reporter


class R84ReportingTests(unittest.TestCase):
    def test_release_count_and_alias_use_deployment_ids(self):
        ids = [S2I['MTPOMO'], S2I['RELD_MOEL']]
        data = SimpleNamespace(n=4500, pool=['MTPOMO', 'reld_mt'], pool_ids=np.array(ids), indices=np.arange(4500))
        config = dict(rows_per_problem=dict(val=4500, test=4500),
            candidate_pools={'OVRP': data.pool}, candidate_pool_ids={'OVRP': ids},
            scenario_solver_aliases={'reld_mt': 'RELD_MOEL'})
        reporter.validate_population(data, 'OVRP', 'TEST', config)
        data.n = 3895
        with self.assertRaises(ValueError):
            reporter.validate_population(data, 'OVRP', 'TEST', config)
        data.n = 4500
        with self.assertRaises(ValueError):
            reporter.validate_population(data, 'OVRP', 'TEST', dict(config, scenario_solver_aliases={}))

    def test_lib_and_cpu_report_cli_share_new_reference_root(self):
        for arm in reporter.ARMS:
            for command, device in (('lib', 'cuda:0'), ('report', 'cpu')):
                args = reporter.parser().parse_args([command, '--arm', arm, '--device', device])
                self.assertEqual((args.command, args.device, args.lib_batch_size), (command, device, 2))
        with patch.object(reporter, 'report') as render:
            reporter.main(['report', '--root', '/tmp/synthetic-r84', '--arm', 'A_baseline'])
        self.assertEqual(render.call_args.args[0].nss_root, Path('/tmp/synthetic-r84/nss_reference'))

    def test_one_test_axes_contains_both_selection_and_winner_bars(self):
        helper = reporter.common.reporting_helpers()
        helper.setup_style()
        data = helper.Predictions(path=Path('/tmp/synthetic.npz'), pool=['MTPOMO', 'reld_mt'],
            pred=np.array([0, 0], np.int64), winner=np.array([0, 1], np.int64),
            costs=np.array([[1., 2.], [2., 1.]], np.float64), indices=np.arange(2),
            source_indices=np.arange(2), pool_ids=np.array([S2I['MTPOMO'], S2I['RELD_MOEL']]), score_ties=0)
        panels = helper.make_panels('OVRP', data, None, data.pool)
        lock = dict(epoch=1, selected_update=1440, checkpoint_sha256='f' * 64)
        fig = reporter.common._distribution(helper, 'OVRP', data.pool, panels,
            'synthetic-new90k', 'R84/A_baseline', lock)
        try:
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(len(fig.axes[0].containers), 2)
            self.assertIn('50.00%', fig.axes[0].get_title())
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
