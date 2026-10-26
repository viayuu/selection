"""CPU-only gate tests; all mock solver receipts live in deleted temp fixtures."""

import copy
import json
import os
import socket
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from code.V4 import r114_label_queue as queue
from code.V4 import r114_response_data as data
from code.V4 import test_r114_response_data as fixtures


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ManifestTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        selection = self.fixture.selection()
        self.manifest = data.prepare_inputs(selection, self.root / "prepared", ["CVRP", "VRPB", "VRPBTW"], 1)
        self.manifest_path = self.root / "prepared/manifest.json"
        deployments = {}
        for problem in ("CVRP", "VRPB", "VRPBTW"):
            for candidate in self.manifest["pools"][problem]:
                key = f"{problem}/{candidate['method']}"
                deployments[key] = dict(backend="synthetic_test_only_" + candidate["method"],
                    profile_sha256="synthetic_test_only_" + key, budget=dict(decoding="synthetic_test_only"), checkpoints=[])
        self.lock = dict(version=self.manifest["version"], manifest_sha256=self.manifest["manifest_sha256"],
                         adapter_entry=data.ADAPTER_ENTRY, runtime=data.runtime_description(self.root / "no_overlay"),
                         overlay_root=str(self.root / "no_overlay"), artifacts={}, deployments=deployments)
        self.lock["lock_sha256"] = data.canonical_hash(self.lock)
        self.lock_path = self.root / "source.lock.json"
        data.write_new_json(self.lock_path, self.lock)
        self.plan = queue.build_plan(self.manifest_path, self.lock_path, self.root / "queue")
        self.plan_path = self.root / "queue/plan.json"
        data.write_new_json(self.plan_path, self.plan)
        self.tasks = {t["task_id"]: t for t in self.manifest["tasks"]}
        self.endpoints = {e["endpoint_id"]: e for e in self.manifest["endpoints"]}

    def tearDown(self):
        self.fixture.tearDown()

    def receipt(self, task_id):
        task = self.tasks[task_id]
        item = data.read_json(self.endpoints[task["endpoint_id"]]["input_path"])["item"]
        f = data.fields(task["problem"], item)
        order = np.argsort(-np.sign(f["q"]), kind="stable") + 1
        tour = [0]
        for node in order:
            tour.extend((int(node), 0))
        cost = float(2 * np.linalg.norm(f["xy"] - f["depot"], axis=1).sum())
        receipt = {k: task[k] for k in ("task_id", "endpoint_id", "base_id", "problem", "method", "input_sha256", "solver_seed")}
        receipt.update(version=self.manifest["version"], source="fresh_solver", status="ok", cost=cost, tour=tour,
            lock_sha256=self.lock["lock_sha256"],
            profile_sha256=self.lock["deployments"][f"{task['problem']}/{task['method']}"]["profile_sha256"])
        return receipt

    def pass_job(self, job):
        # TEST FIXTURE ONLY: simulate completed GPU receipts to exercise gates on CPU.
        attempt = Path(self.plan["queue_root"]) / "attempts" / job["job_id"] / "attempt_0001"
        binding = self.binding()
        data.write_new_json(Path(self.plan["queue_root"]) / "execution_binding.json", binding)
        hashes = {}
        for task_id in job["task_ids"]:
            path = attempt / f"{task_id}.json"
            data.write_new_json(path, self.receipt(task_id))
            hashes[task_id] = data.file_hash(path)
        result = dict(job_id=job["job_id"], plan_sha256=self.plan["plan_sha256"], status="passed",
                      executed=True, binding_sha256=binding["binding_sha256"],
                      lock_sha256=self.lock["lock_sha256"], receipt_sha256=hashes)
        data.write_new_json(attempt / "job_result.json", result)
        return attempt

    def pass_all_runtime(self):
        for job in self.plan["runtime_jobs"]:
            self.pass_job(job)

    def binding(self):
        return queue.make_binding(self.plan, socket.gethostname().split(".")[0], "GPU-test-fixture-only",
                                  sorted(os.sched_getaffinity(0))[:1], 30)

    def fake_success(self, command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        output = Path(command[command.index("--output") + 1])
        data.write_new_json(output, self.receipt(task_id))
        return subprocess.CompletedProcess(command, 0)

    def test_initial_queue_never_certifies_missing_runtime(self):
        self.assertIsNone(self.plan["gpu_binding"])
        self.assertFalse(self.plan["execution"])
        self.assertEqual(self.plan["summary"]["legal_problem_candidates"], 7)
        self.assertEqual(self.plan["summary"]["family_protocol_gates"], 4)
        status = queue.queue_status(self.plan, self.manifest, self.lock)
        self.assertEqual(status["summary"]["runtime_passed"], 0)
        self.assertEqual(status["summary"]["label_ready"], 0)
        self.assertIsNone(queue.next_job(self.plan, self.manifest, self.lock, phase="label")["job"])

    def test_dry_run_and_unbound_execution_never_spawn_worker(self):
        with patch.object(queue.subprocess, "run", side_effect=AssertionError("no process allowed")):
            result = queue.run_one(self.plan, self.manifest, self.lock, None)
            self.assertEqual(result["job"]["phase"], "family")
            self.assertFalse(result["executable"])
            with self.assertRaises(ValueError):
                queue.run_one(self.plan, self.manifest, self.lock, None, execute=True)
        self.assertFalse((Path(self.plan["queue_root"]) / "attempts").exists())

    def test_family_barrier_and_exact_deployment_gate(self):
        family_jobs = [j for j in self.plan["runtime_jobs"] if j["phase"] == "family"]
        self.pass_job(family_jobs[0])
        self.assertIsNone(queue.next_job(self.plan, self.manifest, self.lock, phase="label")["job"])
        for job in family_jobs[1:]:
            self.pass_job(job)
        status = queue.queue_status(self.plan, self.manifest, self.lock)
        self.assertTrue(status["all_family_protocols_passed"])
        self.assertGreater(status["summary"]["label_ready"], 0)
        self.assertLess(status["summary"]["label_ready"], len(self.plan["label_jobs"]))
        pending = next(j for j in self.plan["runtime_jobs"] if j["phase"] == "deployment")
        label = next(j for j in self.plan["label_jobs"] if pending["job_id"] in j["requires"])
        self.assertIsNone(queue.next_job(self.plan, self.manifest, self.lock, job_id=label["job_id"])["job"])
        self.pass_job(pending)
        self.assertIsNotNone(queue.next_job(self.plan, self.manifest, self.lock, job_id=label["job_id"])["job"])

    def test_pass_claim_without_real_receipts_stays_failed(self):
        job = self.plan["runtime_jobs"][0]
        path = Path(self.plan["queue_root"]) / "attempts" / job["job_id"] / "attempt_0001/job_result.json"
        data.write_new_json(path, dict(job_id=job["job_id"], plan_sha256=self.plan["plan_sha256"],
            lock_sha256=self.lock["lock_sha256"], executed=True, status="passed", receipt_sha256={}))
        self.assertEqual(queue.job_state(self.plan, job, self.manifest, self.lock)["status"], "failed")

    def test_tampered_cost_or_profile_never_certifies_gate(self):
        job = self.plan["runtime_jobs"][0]
        receipt = self.receipt(job["task_ids"][0])
        task = self.tasks[receipt["task_id"]]
        receipt["profile_sha256"] = "wrong_budget_or_weight"
        with self.assertRaisesRegex(ValueError, "profile"):
            queue.validate_receipt(receipt, task, self.manifest, self.lock)
        receipt = self.receipt(task["task_id"])
        receipt["cost"] += 2
        with self.assertRaisesRegex(ValueError, "cost"):
            queue.validate_receipt(receipt, task, self.manifest, self.lock)

    def test_explicit_mock_worker_uses_exact_source_and_separate_probe_paths(self):
        with patch.object(queue.subprocess, "run", side_effect=self.fake_success) as invoked:
            result = queue.run_one(self.plan, self.manifest, self.lock, self.binding(), execute=True)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(invoked.call_count, 2)
        for call in invoked.call_args_list:
            argv, environment = call.args[0], call.kwargs["env"]
            self.assertIn(self.plan["deployment_lock_path"], argv)
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-test-fixture-only")
            self.assertEqual(environment["OMP_NUM_THREADS"], "1")
        self.assertEqual(queue.export_labels(self.plan, self.manifest, self.lock, self.root / "export")["exported_label_receipts"], 0)

    def test_failed_attempt_not_retried_or_overwritten_automatically(self):
        job = queue.next_job(self.plan, self.manifest, self.lock)["job"]
        with patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 17)):
            failed = queue.run_one(self.plan, self.manifest, self.lock, self.binding(), job_id=job["job_id"], execute=True)
        self.assertEqual(failed["status"], "failed")
        first = Path(self.plan["queue_root"]) / "attempts" / job["job_id"] / "attempt_0001/job_result.json"
        fingerprint = data.file_hash(first)
        self.assertIsNone(queue.next_job(self.plan, self.manifest, self.lock, job_id=job["job_id"])["job"])
        with patch.object(queue.subprocess, "run", side_effect=self.fake_success):
            retried = queue.run_one(self.plan, self.manifest, self.lock, self.binding(),
                job_id=job["job_id"], retry_failed=True, execute=True)
        self.assertEqual(retried["status"], "passed")
        self.assertEqual(data.file_hash(first), fingerprint)
        self.assertTrue(first.parent.with_name("attempt_0002").exists())

    def test_all_runtime_then_label_execution_and_export(self):
        self.pass_all_runtime()
        status = queue.queue_status(self.plan, self.manifest, self.lock)
        self.assertEqual(status["summary"]["label_ready"], len(self.plan["label_jobs"]))
        with patch.object(queue.subprocess, "run", side_effect=self.fake_success):
            result = queue.run_one(self.plan, self.manifest, self.lock, self.binding(), phase="label", execute=True)
        self.assertEqual(result["status"], "passed")
        exported = queue.export_labels(self.plan, self.manifest, self.lock, self.root / "export")
        self.assertEqual(exported["exported_label_receipts"], 1)
        self.assertEqual(exported["runtime_probe_costs_exported"], 0)

    def test_timeout_keeps_failure_receipts(self):
        job = queue.next_job(self.plan, self.manifest, self.lock)["job"]
        with patch.object(queue.subprocess, "run", side_effect=subprocess.TimeoutExpired("synthetic", 30)):
            result = queue.run_one(self.plan, self.manifest, self.lock, self.binding(), execute=True)
        self.assertEqual(result["status"], "failed")
        attempt = Path(self.plan["queue_root"]) / "attempts" / job["job_id"] / "attempt_0001"
        receipt = data.read_json(attempt / f"{job['task_ids'][0]}.json")
        self.assertEqual(receipt["status"], "timeout")
        self.assertIsNone(receipt["cost"])

    def test_runtime_success_does_not_transfer_to_different_binding(self):
        with patch.object(queue.subprocess, "run", side_effect=self.fake_success):
            queue.run_one(self.plan, self.manifest, self.lock, self.binding(), execute=True)
        changed = queue.make_binding(self.plan, socket.gethostname().split(".")[0], "GPU-other-test-only",
                                     self.binding()["cpu_ids"], 30)
        with patch.object(queue.subprocess, "run", side_effect=AssertionError("must not start on another binding")):
            with self.assertRaisesRegex(ValueError, "Immutable"):
                queue.run_one(self.plan, self.manifest, self.lock, changed, execute=True)

    def test_checkpoint_buckets_and_related_candidates_are_not_merged(self):
        task = dict(problem="CVRP", method="MoSES_RF")
        self.assertEqual(queue.deployment_key(task, dict(input_check=dict(n=75)))[1], "50")
        self.assertEqual(queue.deployment_key(task, dict(input_check=dict(n=76)))[1], "100")
        self.assertEqual(queue.family_id("MoSES_RF"), queue.family_id("MoSES_CaDA"))
        self.assertNotEqual("MoSES_RF", "MoSES_CaDA")

    def test_load_plan_integrity(self):
        queue.load_plan(self.plan_path)
        changed = copy.deepcopy(self.plan)
        changed["gpu_binding"] = "implicitly_allocated"
        self.plan_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "Queue plan changed"):
            queue.load_plan(self.plan_path)


if __name__ == "__main__":
    unittest.main()
