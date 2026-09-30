import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("study_eval", Path(__file__).with_name("evaluate_study.py"))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class EvaluationGuardTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "study"
        self.root.mkdir()
        for mocked in (patch.object(study, "ROOT", self.root), patch.object(study, "configure_torch")):
            mocked.start()
            self.addCleanup(mocked.stop)
        self.load = patch.object(study, "load_model", side_effect=AssertionError("Evaluation must not start"))
        self.load_model = self.load.start()
        self.addCleanup(self.load.stop)

    def make_runs(self, updates=12000, finished=True):
        (self.root / "exit_code").write_text("0\n")
        for mode in ("original", "ce", "winner_cost"):
            run = self.root.parent / f"R34_{mode}_seed2_4090"
            run.mkdir()
            if finished:
                (run / "finished_at.txt").write_text("finished\n")
            (run / "history.json").write_text(json.dumps([dict(updates=updates)]))
            torch.save(dict(epoch=0, successful_updates=updates,
                            macro=dict(macro_top1=.5, macro_vs_sbs_pct=-1.)), run / "best.pt")

    def test_failed_queue_is_rejected(self):
        (self.root / "exit_code").write_text("1\n")
        with self.assertRaisesRegex(RuntimeError, "finish successfully"):
            study.main()
        self.load_model.assert_not_called()

    def test_unfinished_run_is_rejected(self):
        self.make_runs(finished=False)
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            study.main()
        self.load_model.assert_not_called()

    def test_insufficient_budget_is_rejected(self):
        self.make_runs(updates=11999)
        with self.assertRaisesRegex(RuntimeError, "budget not met"):
            study.main()
        self.load_model.assert_not_called()

    def test_changed_checkpoint_is_rejected(self):
        source = self.root / "source.pt"
        source.write_bytes(b"original")
        frozen, _ = study.freeze_checkpoint("model", source)
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            study.freeze_checkpoint("model", source)
        self.assertEqual(frozen.read_bytes(), b"original")

    def test_changed_selection_is_rejected(self):
        self.make_runs()
        (self.root / "validation_selection.json").write_text(json.dumps({"different": True}))
        with self.assertRaisesRegex(RuntimeError, "selection is frozen"):
            study.main()
        self.load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
