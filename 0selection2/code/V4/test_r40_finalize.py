import json
import tempfile
import unittest
from pathlib import Path

from .multitask_probe import dump
from .r40_verify import expected_epochs


class StopEndpointTests(unittest.TestCase):
    def test_original_budget_without_user_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(expected_epochs(Path(folder)), {'A': 60, 'B': 60})

    def test_only_explicit_A_checkpoint_changes_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stop = dict(group='A', requested_epochs=60, checkpoint_epoch=56, checkpoint_updates=151200)
            dump(root / 'user_stop.json', stop)
            self.assertEqual(expected_epochs(root), {'A': 56, 'B': 60})
            self.assertEqual(json.loads((root / 'user_stop.json').read_text()), stop)

    def test_invalid_stop_cannot_relax_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for changes in (dict(group='B'), dict(checkpoint_updates=162000), dict(checkpoint_epoch=60)):
                stop = dict(group='A', requested_epochs=60, checkpoint_epoch=56, checkpoint_updates=151200)
                dump(root / 'user_stop.json', dict(stop, **changes))
                with self.assertRaises(ValueError):
                    expected_epochs(root)


if __name__ == '__main__':
    unittest.main()
