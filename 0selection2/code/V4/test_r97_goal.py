"""No Git/GPU execution: publication boundaries on temporary output trees."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import r97_goal as goal


class GoalTests(unittest.TestCase):
    def test_early_receipts_do_not_publish_incomplete_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root, launch = Path(directory)/'model', Path(directory)/'launch'
            root.mkdir()
            goal.write_json(root/'run_complete.json', {'complete': True})
            result = goal.status(root, launch)
            self.assertFalse(result['complete'])
            self.assertFalse(result['required']['controller_rc0'])

    def test_runtime_weights_raw_predictions_and_wandb_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            git_root = Path(directory)
            repo = git_root/'0selection2'
            root, launch = repo/'model', repo/'launch'
            for name in ('runtime/key.json', 'wandb/run.json', 'evaluation/prediction.json',
                         'prediction.npz', 'model.pt', 'RESULTS.pdf', 'context/TSP/identity.json'):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'x')
            with patch.object(goal, 'REPO', repo):
                paths = goal.publication_paths(root, launch, git_root)
            self.assertEqual(paths, ['0selection2/model/RESULTS.pdf', '0selection2/model/context/TSP/identity.json'])


if __name__ == '__main__':
    unittest.main()
