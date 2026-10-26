"""Bounded CPU fixtures only. No fixture is a real route qualification."""

import copy
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from . import r114_group_worker as worker
from . import r114_label_queue as queue
from . import r114_response_data as data
from . import r114_run_queue as scheduler
from . import test_r114_response_data as fixtures


class GroupQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.ManifestTests()
        cls.fixture.setUp()
        cls.base = cls.fixture.root
        cls.manifest = data.prepare_inputs(cls.fixture.selection(), cls.base / "inputs", ["CVRP", "VRPB", "VRPBTW"], 2)
        cls.manifest_path = cls.base / "inputs/manifest.json"
        cls.reference = dict(version=cls.manifest["version"], manifest_sha256=cls.manifest["manifest_sha256"],
            adapter_entry=data.ADAPTER_ENTRY, overlay_root=str(cls.base / "no_overlay"),
            runtime=data.runtime_description(cls.base / "no_overlay"),
            artifacts={str(Path(data.__file__).resolve()): data.file_hash(data.__file__)},
            execution=dict(batch_size=1, seed="common per base and method; reset after model loading", old_cost_reuse=False),
            deployments={f"{p}/{c['method']}": dict(backend="synthetic_test_only", profile_sha256=f"test_{p}_{c['method']}",
                budget=dict(decoding="synthetic_test_only"), checkpoints=[])
                for p in cls.manifest["selected_problems"] for c in cls.manifest["pools"][p]})
        cls.reference["lock_sha256"] = data.canonical_hash(cls.reference)
        cls.lock_path = cls.base / "reference.lock.json"
        data.write_new_json(cls.lock_path, cls.reference)
        cls.parent_path = cls.base / "parent/plan.json"
        data.write_new_json(cls.parent_path, queue.build_plan(cls.manifest_path, cls.lock_path, cls.parent_path.parent))
        cls.tasks = {t["task_id"]: t for t in cls.manifest["tasks"]}
        cls.endpoints = {e["endpoint_id"]: e for e in cls.manifest["endpoints"]}

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDown()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.base)
        self.root = Path(self.temp.name)
        routes = dict(source_route_paths=7, records=[])
        with patch.object(worker, "route_inventory", return_value=routes):
            self.plan = queue.build_group_plan(self.parent_path, self.root / "queue")
        self.plan_path = Path(self.plan["queue_root"]) / "plan.json"
        data.write_new_json(self.plan_path, self.plan)
        self.lock = data.read_json(self.plan["deployment_lock_path"])
        self.binding = queue.make_binding(self.plan, socket.gethostname().split(".")[0],
            "GPU-synthetic-test-only", sorted(os.sched_getaffinity(0))[:1], 10)
        self.analysis_mock = patch.object(scheduler, "run_analysis", return_value=dict(status="synthetic_test_only"))
        self.analysis_mock.start()

    def tearDown(self):
        self.analysis_mock.stop()
        self.temp.cleanup()

    def receipt(self, task_id, lock=None):
        task = self.tasks[task_id]
        item = data.read_json(self.endpoints[task["endpoint_id"]]["input_path"])["item"]
        fields = data.fields(task["problem"], item)
        tour = [0]
        for node in np.argsort(-np.sign(fields["q"]), kind="stable") + 1:
            tour.extend((int(node), 0))
        cost = float(2 * np.linalg.norm(fields["xy"] - fields["depot"], axis=1).sum())
        lock = lock or self.lock
        return dict({k: task[k] for k in ("task_id", "endpoint_id", "base_id", "problem", "method", "input_sha256", "solver_seed")},
            version=self.manifest["version"], lock_sha256=lock["lock_sha256"], source="fresh_solver", status="ok",
            profile_sha256=lock["deployments"][f"{task['problem']}/{task['method']}"]["profile_sha256"],
            cost=cost, tour=tour)

    def session(self, *_args):
        return Mock(solve=lambda t: self.receipt(t))

    def run_worker(self, job, output=None, session=None):
        return worker.run_group(self.plan, self.manifest, self.lock, job, output or self.root / "worker",
            execute=True, device="cpu", session_factory=session or self.session,
            cold_runner=lambda t: self.receipt(t, self.reference))

    def fake_process(self, command, **_kwargs):
        job_id = command[command.index("--job-id") + 1]
        job = next(j for j in self.plan["runtime_jobs"] + self.plan["label_jobs"] if j["job_id"] == job_id)
        output = Path(command[command.index("--output") + 1])
        result = self.run_worker(job, output)
        return subprocess.CompletedProcess(command, int(result["status"] != "passed"))

    def test_dry_run_has_no_binding_or_solver_process(self):
        with patch.object(queue.subprocess, "Popen", side_effect=AssertionError("no worker")), \
             patch.object(data.importlib, "import_module", side_effect=AssertionError("no backend")):
            result = scheduler.run_queue(self.plan, self.manifest, self.lock)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["committed_label_receipts"], 0)
        self.assertFalse((self.plan_path.parent / "execution_binding.json").exists())
        self.assertFalse((self.plan_path.parent / "attempts").exists())

    def test_whole_pool_and_independent_base_probe_topology(self):
        self.assertEqual(self.plan["summary"]["label_worker_groups"], 7)
        self.assertEqual(sum(len(j["task_ids"]) for j in self.plan["label_jobs"]), len(self.tasks))
        for job in self.plan["runtime_jobs"]:
            self.assertGreaterEqual(len({self.tasks[t]["base_id"] for t in job["task_ids"]}), 2)
            self.assertEqual(len(job["task_ids"]), 3)
        broken = copy.deepcopy(self.plan)
        broken["label_jobs"][0]["requires"] = []
        with self.assertRaisesRegex(ValueError, "every legal method"):
            queue.validate_group_topology(broken, self.manifest, self.lock, self.reference)

    def test_plan_and_profile_provenance_cannot_drift(self):
        queue.load_plan(self.plan_path)
        scheduler.check_paths(self.plan, self.manifest_path, self.plan["deployment_lock_path"])
        with self.assertRaisesRegex(ValueError, "pinned"):
            scheduler.check_paths(self.plan, self.manifest_path, self.lock_path)
        broken = copy.deepcopy(self.lock)
        next(iter(broken["deployments"].values()))["budget"] = dict(decoding="changed")
        with self.assertRaisesRegex(ValueError, "native source"):
            queue.validate_group_topology(self.plan, self.manifest, broken, self.reference)

    def test_one_persistent_backend_per_problem_method(self):
        job = self.plan["label_jobs"][0]
        factory = Mock(side_effect=self.session)
        result = self.run_worker(job, session=factory)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(result["completed_task_ids"], job["task_ids"])
        worker.validate_equivalence(self.plan, job, self.root / "worker", self.manifest, self.lock)

    def test_cold_reference_and_replay_are_real_required_receipts(self):
        job = self.plan["runtime_jobs"][0]
        result = self.run_worker(job)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["cold_reference_sha256"]), len(job["cold_task_ids"]))
        self.assertEqual(len(result["replay_checks"]), len(job["replay_task_ids"]))
        worker.validate_equivalence(self.plan, job, self.root / "worker", self.manifest, self.lock)
        (self.root / "worker/replay" / f"{job['replay_task_ids'][0]}.json").unlink()
        with self.assertRaises(OSError):
            worker.validate_equivalence(self.plan, job, self.root / "worker", self.manifest, self.lock)

    def test_exact_fp64_response_equivalence_not_fp32_tolerance(self):
        task_id = self.plan["runtime_jobs"][0]["task_ids"][0]
        original = self.receipt(task_id)
        changed = dict(original, cost=np.nextafter(original["cost"], np.inf))
        self.assertEqual(np.float32(changed["cost"]), np.float32(original["cost"]))
        with self.assertRaisesRegex(ValueError, "exact native"):
            worker.same_response(original, changed)

    def test_cold_lazy_qualification_needs_only_grouped_route_and_seed_replay(self):
        job = dict(self.plan["runtime_jobs"][0], cold_task_ids=[])
        result = self.run_worker(job)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["cold_reference_sha256"], {})
        worker.validate_equivalence(self.plan, job, self.root / "worker", self.manifest, self.lock)

    def test_analysis_uses_actual_assembled_json_and_cpu_only_exact_cli(self):
        self.analysis_mock.stop()
        output = self.root / "unique_analysis"
        labels = self.root / "actual_assembled.json"
        with patch.object(scheduler.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as invoked:
            result = scheduler.run_analysis(self.plan, labels, output, self.binding)
        command = invoked.call_args.args[0]
        self.assertEqual(command[command.index("--labels") + 1], str(labels))
        self.assertEqual(command[command.index("--manifest") + 1], str(self.manifest_path))
        self.assertEqual(command[command.index("--output") + 1], str(output))
        self.assertIn("-B", command)
        self.assertEqual(invoked.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(result["status"], "complete")

    def test_first_unusable_route_leaves_entire_unrun_tail_absent(self):
        job = self.plan["label_jobs"][0]
        called = []
        def solve(task_id):
            called.append(task_id)
            return dict(self.receipt(task_id), tour=None)
        result = self.run_worker(job, session=lambda *_: Mock(solve=solve))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(called, job["task_ids"][:1])
        self.assertEqual(result["completed_task_ids"], [])
        for task_id in job["task_ids"][1:]:
            self.assertFalse((self.root / "worker" / f"{task_id}.json").exists())

    def test_bad_cold_reference_never_initializes_grouped_backend(self):
        job = self.plan["runtime_jobs"][0]
        factory = Mock(side_effect=AssertionError("must stop before group"))
        result = worker.run_group(self.plan, self.manifest, self.lock, job, self.root / "worker",
            execute=True, device="cpu", session_factory=factory,
            cold_runner=lambda t: dict(self.receipt(t, self.reference), cost=None))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(factory.call_count, 0)
        self.assertEqual(len(result["cold_reference_sha256"]), 0)

    def test_lazy_cache_reset_and_per_endpoint_rng_reset(self):
        backend = Mock(pool=Mock(_cache={50: object()}), validated_buckets={50})
        worker.reset_lazy_runner(backend, "MoSES_RF")
        self.assertEqual(backend.pool._cache, {})
        self.assertEqual(backend.validated_buckets, set())
        torch = Mock()
        with patch.object(worker.random, "seed") as pyseed, patch.object(worker.np.random, "seed") as npseed:
            worker.reset_rng(torch, 123, "cuda:0")
        pyseed.assert_called_once_with(123)
        npseed.assert_called_once_with(123)
        torch.manual_seed.assert_called_once_with(123)
        torch.cuda.manual_seed_all.assert_called_once_with(123)

    def test_scheduler_executes_resume_and_complete_masks_without_probes_as_labels(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=self.binding["gpu_uuid"]), \
             patch.object(queue, "supervised_group", side_effect=self.fake_process):
            first = scheduler.run_queue(self.plan, self.manifest, self.lock, self.binding, execute=True, max_jobs=1)
            self.assertEqual(first["status"], "yielded")
            final = scheduler.run_queue(self.plan, self.manifest, self.lock, self.binding, execute=True, finalize=True)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["committed_label_receipts"], len(self.tasks))
        self.assertEqual(final["export"]["runtime_probe_costs_exported"], 0)
        self.assertEqual(final["labels"]["summary"]["full_pool_winners"], len(self.endpoints))
        self.assertEqual(len(list((self.plan_path.parent / "scheduler_runs").glob("run_*"))), 2)
        self.assertTrue(all(len(list(p.glob("attempt_*"))) == 1 for p in (self.plan_path.parent / "attempts").iterdir()))

    def test_failed_problem_stops_without_retries_other_problems_can_finish(self):
        failed_problem = queue.next_job(self.plan, self.manifest, self.lock)["job"]["problem"]
        def process(command, **kwargs):
            if failed_problem in command[command.index("--job-id") + 1]:
                return subprocess.CompletedProcess(command, 9)
            return self.fake_process(command, **kwargs)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=self.binding["gpu_uuid"]), \
             patch.object(queue, "supervised_group", side_effect=process) as called:
            result = scheduler.run_queue(self.plan, self.manifest, self.lock, self.binding, execute=True, finalize=True)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stopped_problems"], [failed_problem])
        self.assertEqual(sum(failed_problem in c.args[0][c.args[0].index("--job-id") + 1] for c in called.call_args_list), 1)
        labels = data.read_json(result["labels"]["path"])
        self.assertTrue(all(not row["winner_mask"] for row in labels["endpoints"] if row["problem"] == failed_problem))
        self.assertFalse(result["export"]["complete_release"])

    def test_interruption_still_assembles_masked_snapshot(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=self.binding["gpu_uuid"]), \
             patch.object(queue, "supervised_group", side_effect=KeyboardInterrupt):
            result = scheduler.run_queue(self.plan, self.manifest, self.lock, self.binding, execute=True, finalize=True)
        self.assertEqual(result["status"], "interrupted")
        self.assertTrue(Path(result["labels"]["path"]).is_file())
        self.assertEqual(result["labels"]["summary"]["full_pool_winners"], 0)
        self.assertFalse((self.plan_path.parent / "labels.json").exists())

    def test_incomplete_attempt_never_retries_or_qualifies(self):
        job = self.plan["runtime_jobs"][0]
        attempt = self.plan_path.parent / "attempts" / job["job_id"] / "attempt_0001"
        attempt.mkdir(parents=True)
        status = queue.queue_status(self.plan, self.manifest, self.lock)
        self.assertIn(job["problem"], status["stopped_problems"])
        self.assertNotEqual(queue.next_job(self.plan, self.manifest, self.lock)["job"]["problem"], job["problem"])
        self.assertEqual(queue.job_state(self.plan, job, self.manifest, self.lock)["status"], "incomplete")

    def test_wrong_physical_gpu_binding_never_spawns(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"), patch.object(queue, "supervised_group") as called:
            with self.assertRaisesRegex(ValueError, "physical GPU"):
                scheduler.run_queue(self.plan, self.manifest, self.lock, self.binding, execute=True)
        self.assertEqual(called.call_count, 0)

    def test_stalled_group_kills_process_group_with_short_cpu_timeout(self):
        with (self.root / "watchdog.log").open("w") as log, \
             patch.object(queue.os, "killpg", wraps=os.killpg) as killed:
            with self.assertRaises(subprocess.TimeoutExpired):
                queue.supervised_group([sys.executable, "-c", "import time; time.sleep(10)"],
                    cwd=str(self.root), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdout=log,
                    timeout=0.1, progress=self.root / "no_progress", stage_limit=2)
        self.assertEqual(killed.call_args.args[1], signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
