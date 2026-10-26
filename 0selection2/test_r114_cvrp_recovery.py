"""Focused CPU checks of the staged acquisition revision; no solver calls."""

import copy
import unittest
from pathlib import Path

from code.V4 import r114_cvrp_recovery as recovery
from code.V4 import r114_label_queue as queue
from code.V4 import r114_response_data as data


class CVRPRecoveryTests(unittest.TestCase):
    def test_frozen_subset_preserves_all_inputs_tasks_pairs_and_solver_seeds(self):
        parent = data.read_json(recovery.PARENT_MANIFEST)
        staged = recovery.subset_manifest(parent)
        for field in ("endpoints", "tasks", "pairs", "rejections"):
            self.assertEqual(staged[field], [e for e in parent[field] if e["problem"] == "CVRP"])
        self.assertEqual(staged["pools"], {"CVRP": parent["pools"]["CVRP"]})
        for kind in ("endpoint", "task", "pair"):
            self.assertEqual(staged[kind + "_count"],
                             sum(row["problem"] == "CVRP" for row in parent[kind + "s"]))
        self.assertEqual(recovery.cvrp_counts(staged), recovery.cvrp_counts(parent))
        data.verify_manifest(staged)
        # An alternate parent metadata fixture catches fixed-count assumptions.
        # It reuses existing inputs in memory only; no production rows are changed.
        smaller = copy.deepcopy(parent)
        base_id = staged["endpoints"][0]["base_id"]
        for field in ("endpoints", "tasks", "pairs", "rejections"):
            smaller[field] = [row for row in smaller[field]
                              if row["problem"] != "CVRP" or row.get("base_id") == base_id]
        for kind in ("endpoint", "task", "pair"):
            smaller[kind + "_count"] = len(smaller[kind + "s"])
        smaller.pop("manifest_sha256")
        smaller["manifest_sha256"] = data.canonical_hash(smaller)
        subset = recovery.subset_manifest(smaller)
        self.assertLess(subset["endpoint_count"], staged["endpoint_count"])
        self.assertEqual(recovery.cvrp_counts(subset), recovery.cvrp_counts(smaller))
        data.verify_manifest(subset)
        altered = copy.deepcopy(parent)
        altered["tasks"][0]["solver_seed"] += 1
        with self.assertRaisesRegex(ValueError, "Parent manifest metadata changed"):
            recovery.subset_manifest(altered)

    def test_actual_profiles_private_source_bindings_and_host_boundary(self):
        root = recovery.ROOT
        old = data.read_json(root / "profiles.original.snapshot.json")
        new = data.read_json(root / "profiles.corrected.snapshot.json")
        self.assertEqual(old["profiles"], new["profiles"])
        self.assertEqual(new["workspace"], str(data.WORKSPACE))
        self.assertFalse(new["native_qualified"])
        bindings = data.read_json(root / "source_bindings.json")
        changed = [e for e in bindings if e["sha256"] != e["original_sha256"]]
        self.assertEqual([Path(e["path"]).name for e in changed], ["train_100k_backends.py"])
        for entry in bindings:
            self.assertEqual(str(Path(entry["path"]).resolve()), entry["resolved_path"])
            self.assertEqual(data.file_hash(entry["path"]), entry["sha256"])
        original = Path(changed[0]["original_path"]).read_text()
        corrected = Path(changed[0]["path"]).read_text()
        self.assertEqual(corrected, recovery.boundary.revised_source(original))
        self.assertEqual(new["modules"]["common"]["resolved_path"],
                         str(Path(old["modules"]["common"]["path"]).resolve()))

    def test_real_lock_topology_and_native_cli_parser_without_execution(self):
        plan, manifest, lock = recovery.verify()
        self.assertEqual(len(plan["runtime_jobs"]), len(manifest["pools"]["CVRP"]))
        self.assertEqual(sum(len(j["task_ids"]) for j in plan["label_jobs"]), len(manifest["tasks"]))
        self.assertEqual(set(lock["deployments"]), {f"CVRP/{c['method']}" for c in manifest["pools"]["CVRP"]})
        self.assertEqual(lock["scripts_root"], str(recovery.ROOT / "scripts"))
        self.assertFalse((recovery.ROOT / "queue/attempts").exists())
        self.assertFalse((recovery.ROOT / "queue/execution_binding.json").exists())
        arguments = recovery.parse_runner_argv(recovery.runner_argv(plan, execute=True))
        self.assertTrue(arguments.execute)
        self.assertFalse(arguments.finalize)
        self.assertEqual(arguments.plan, recovery.ROOT / "queue/plan.json")
        altered = copy.deepcopy(lock)
        altered["scripts_root"] = "/not/the/locked/scripts"
        with self.assertRaisesRegex(ValueError, "Deployment lock mismatch"):
            data.verify_lock(altered, manifest)
        reference = data.read_json(plan["reference_lock_path"])
        altered = copy.deepcopy(plan)
        altered["label_jobs"][0]["requires"] = []
        with self.assertRaisesRegex(ValueError, "every legal method"):
            queue.validate_group_topology(altered, manifest, lock, reference)


if __name__ == "__main__":
    unittest.main()
