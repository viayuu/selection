"""CPU-only conversion/contract tests. Synthetic receipts are test fixtures only."""

import copy
import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import r114_response_data as r


def vrp(problem="VRPTW"):
    xy = [[0.10, 0.05], [0.15, 0.07], [0.20, 0.09],
          [0.25, 0.11], [0.30, 0.13], [0.35, 0.15]]
    demand = [0.4, 0.1, 0.3, 0.2, -0.4 if "B" in problem else 0.2,
              -0.2 if "B" in problem else 0.4]
    if problem == "CVRP":
        return dict(depot=[[[0.0, 0.0]]], loc=[xy], demand=[demand])
    item = dict(depot_xy=[[0.0, 0.0]], node_xy=xy, node_demand=demand, capacity=1.0)
    if "L" in problem:
        item["route_limit"] = 3.0
    if "TW" in problem:
        item.update(service_time=[0.2] * 6, tw_start=[1.2, 0.8, 1.1, 0.9, 1.0, 0.7],
                    tw_end=[1.8, 1.5, 1.9, 1.6, 1.9, 1.4])
    return item


POOL = [dict(method="MVMOE", global_id=11, source_method="MVMOE"),
        dict(method="MTPOMO", global_id=10, source_method="MTPOMO"),
        dict(method="reld_mt", global_id=16, source_method="RELD_MOEL")]


def synthetic_verified(cost, status="ok"):
    return dict(cost=cost, status=status, source="fresh_solver", route_audit=dict(verified=True))


class InterventionTests(unittest.TestCase):
    def test_all_eligible_input_types(self):
        for problem in r.ELIGIBLE:
            with self.subTest(problem=problem):
                self.assertTrue(r.validate_input(problem, vrp(problem))["legal"])

    def test_demand_pair_reproducible_and_no_mutation(self):
        item = vrp("CVRP")
        saved = copy.deepcopy(item)
        pair = r.intervention_pair("CVRP", item, "demand", 2)
        self.assertEqual(pair["status"], "ready")
        self.assertEqual(pair, r.intervention_pair("CVRP", item, "demand", 2))
        self.assertEqual(item, saved)
        self.assertNotEqual(pair["directed"]["item"], pair["random"]["item"])
        for arm in ("directed", "random"):
            changed = pair[arm]["item"]
            self.assertEqual(sorted(changed["demand"][0]), sorted(item["demand"][0]))
            self.assertEqual(changed["loc"], item["loc"])
            self.assertEqual(sum(a != b for a, b in zip(item["demand"][0], changed["demand"][0])), 4)
        self.assertLessEqual(abs(pair["random"]["l1"] / pair["directed"]["l1"] - 1), 0.25)

    def test_all_b_roles_and_other_fields_preserved(self):
        for problem in (p for p in r.ELIGIBLE if "B" in p):
            original = vrp(problem)
            pair = r.intervention_pair(problem, original, "demand", 18)
            self.assertEqual(pair["status"], "ready")
            for arm in ("directed", "random"):
                changed = pair[arm]["item"]
                np.testing.assert_array_equal(np.sign(original["node_demand"]), np.sign(changed["node_demand"]))
                r.validate_intervention(problem, original, changed, "demand")

    def test_tw_centers_and_per_node_widths_preserved(self):
        for problem in (p for p in r.ELIGIBLE if "TW" in p):
            original = vrp(problem)
            pair = r.intervention_pair(problem, original, "tw_center", 2)
            self.assertEqual(pair["status"], "ready")
            for arm in ("directed", "random"):
                changed = pair[arm]["item"]
                r.validate_intervention(problem, original, changed, "tw_center")
                np.testing.assert_allclose(np.asarray(changed["tw_end"]) - changed["tw_start"],
                                           np.asarray(original["tw_end"]) - original["tw_start"], atol=1e-12)
                self.assertEqual(changed["node_xy"], original["node_xy"])
                self.assertEqual(changed["node_demand"], original["node_demand"])

    def test_illegal_tw_swap_rejected_without_clipping(self):
        item = vrp("VRPTW")
        item["tw_start"][0], item["tw_end"][0] = 0.11, 0.12
        r.validate_input("VRPTW", item)
        changed = r.apply_permutation("VRPTW", item, "tw_center", [5, 1, 2, 3, 4, 0])
        with self.assertRaises(ValueError):
            r.validate_intervention("VRPTW", item, changed, "tw_center")

    def test_corrupted_interventions_rejected(self):
        item = vrp("VRPBTW")
        for mutate in (lambda x: x["node_demand"].__setitem__(0, 0.9),
                       lambda x: x["node_xy"][0].__setitem__(0, 0.2),
                       lambda x: x.__setitem__("capacity", 2)):
            changed = copy.deepcopy(item)
            mutate(changed)
            with self.assertRaises(ValueError):
                r.validate_intervention("VRPBTW", item, changed, "demand")
        changed = copy.deepcopy(item)
        changed["node_demand"][0], changed["node_demand"][4] = changed["node_demand"][4], changed["node_demand"][0]
        with self.assertRaisesRegex(ValueError, "role/sign"):
            r.validate_intervention("VRPBTW", item, changed, "demand")

    def test_width_or_center_multiset_corruption_rejected(self):
        item = vrp()
        changed = copy.deepcopy(item)
        changed["tw_end"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "width"):
            r.validate_intervention("VRPTW", item, changed, "tw_center")
        changed = copy.deepcopy(item)
        changed["tw_end"][0] += 0.01
        changed["tw_start"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "center multiset"):
            r.validate_intervention("VRPTW", item, changed, "tw_center")

    def test_no_attribute_and_noop_are_not_labels(self):
        with self.assertRaises(ValueError):
            r.intervention_pair("TSP", [], "demand", 2)
        with self.assertRaises(ValueError):
            r.intervention_pair("CVRP", vrp("CVRP"), "tw_center", 2)
        with self.assertRaisesRegex(ValueError, "No-op"):
            r.validate_intervention("VRPTW", vrp(), vrp(), "demand")
        item = vrp("CVRP")
        item["demand"] = [[0.1] * 6]
        self.assertEqual(r.intervention_pair("CVRP", item, "demand", 2)["status"], "rejected")


class RouteAndLabelTests(unittest.TestCase):
    def test_independent_route_cost_and_mixed_signed_load(self):
        item = vrp("VRPB")
        # This route mixes pickup/delivery, legal in the existing signed-load model.
        tour = [0, 1, 2, 5, 3, 4, 0, 6, 0]
        xy = np.asarray(item["depot_xy"] + item["node_xy"])
        cost = float(np.linalg.norm(xy[tour[1:]] - xy[tour[:-1]], axis=1).sum())
        self.assertTrue(r.check_route("VRPB", item, tour, cost)["verified"])
        with self.assertRaisesRegex(ValueError, "capacity"):
            r.check_route("VRPB", item, [0, 5, 1, 2, 3, 4, 6, 0], cost)
        with self.assertRaisesRegex(ValueError, "cost"):
            r.check_route("VRPB", item, tour, cost + 1)

    def test_open_and_closed_costs_distinct(self):
        for p in ("VRPTW", "OVRPTW", "VRPBTW", "OVRPBLTW"):
            item = vrp(p)
            order = np.argsort(-np.sign(item["node_demand"]), kind="stable") + 1
            tour = [0]
            for node in order:
                tour.extend((int(node), 0))
            distance = np.linalg.norm(np.asarray(item["node_xy"]), axis=1).sum()
            cost = float(distance * (1 if p.startswith("OVRP") else 2))
            self.assertTrue(r.check_route(p, item, tour, cost)["verified"])

    def test_invalid_route_is_not_truth(self):
        item = vrp("CVRP")
        for tour in ([0, 1, 0], [0, 1.5, 2, 3, 4, 5, 6, 0], [0, 1, 2, 3, 4, 5, 8, 0]):
            with self.assertRaises(ValueError):
                r.check_route("CVRP", item, tour, 2)

    def test_fp64_winner_order_and_exact_ties(self):
        records = {"MTPOMO": synthetic_verified(1 + 1e-10), "MVMOE": synthetic_verified(1),
                   "reld_mt": synthetic_verified(2)}
        labels = r.endpoint_labels(POOL, records)
        self.assertEqual(labels["winner"], "MVMOE")
        self.assertEqual(labels["methods"], ["MTPOMO", "MVMOE", "reld_mt"])
        records["MTPOMO"]["cost"] = 1
        self.assertEqual(r.endpoint_labels(POOL, records)["winner"], "MTPOMO")

    def test_partial_pool_never_produces_winner_or_centered_response(self):
        for status in ("timeout", "error", "invalid_route"):
            records = {"MTPOMO": synthetic_verified(1), "MVMOE": synthetic_verified(2, status)}
            labels = r.endpoint_labels(POOL, records)
            self.assertFalse(labels["winner_mask"])
            self.assertIsNone(labels["winner"])
            self.assertEqual(labels["failure_mask"], [False, True, True])
            self.assertEqual(labels["relative_response"], [None] * 3)
            self.assertIsNone(labels["costs_fp64"][1])

    def test_old_costs_and_pseudo_receipts_masked(self):
        for source in ("old_label", "teacher", "predicted", "solver_default"):
            record = synthetic_verified(1)
            record["source"] = source
            label = r.endpoint_labels(POOL, {"MTPOMO": record})
            self.assertTrue(all(label["failure_mask"]))

    def test_nonfinite_cost_or_missing_route_audit_is_not_success(self):
        for value in (float("nan"), float("inf"), 0, -1):
            labels = r.endpoint_labels(POOL, {"MTPOMO": synthetic_verified(value)})
            self.assertTrue(all(labels["failure_mask"]))
        receipt = synthetic_verified(1)
        receipt["route_audit"] = None
        self.assertTrue(all(r.endpoint_labels(POOL, {"MTPOMO": receipt})["failure_mask"]))

    def test_delta_and_partial_pair_masks(self):
        a = r.endpoint_labels(POOL, {m: synthetic_verified(c) for m, c in zip(
            ("MTPOMO", "MVMOE", "reld_mt"), (10, 12, 13))})
        b = r.endpoint_labels(POOL, {m: synthetic_verified(c) for m, c in zip(
            ("MTPOMO", "MVMOE", "reld_mt"), (14, 13, 17))})
        pair = r.paired_labels(a, b)
        self.assertEqual(pair["Delta_mn"][0][1], 3)
        self.assertEqual(pair["Delta_mn"][1][0], -3)
        self.assertTrue(pair["winner_flip"])
        shifted = r.endpoint_labels(POOL, {m: synthetic_verified(c + 4) for m, c in zip(
            ("MTPOMO", "MVMOE", "reld_mt"), (10, 12, 13))})
        self.assertEqual(r.paired_labels(a, shifted)["Delta_mn"], [[0] * 3 for _ in range(3)])
        incomplete = r.endpoint_labels(POOL, {"MTPOMO": synthetic_verified(3), "MVMOE": synthetic_verified(4)})
        pair = r.paired_labels(a, incomplete)
        self.assertIsNotNone(pair["Delta_mn"][0][1])
        self.assertIsNone(pair["Delta_mn"][0][2])
        self.assertFalse(pair["winner_pair_mask"])
        self.assertEqual(pair["relative_response_change"], [None] * 3)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "frozen"
        self.data.mkdir()
        r.write_new_json(self.data / "scenario_manifest.json", dict(scenario_id="synthetic_test_only", source_data="test_only"))
        for problem in r.PROBLEMS:
            for split, ids in (("train", range(12)), ("val", range(20, 22)), ("test", range(30, 32))):
                directory = self.data / f"{problem}{split}"
                directory.mkdir()
                mapping = directory / "index_mapping.csv"
                with mapping.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["local_index", "source_index", "scale", "distribution", "capacity_type"])
                    writer.writeheader()
                    for local, source in enumerate(ids):
                        writer.writerow(dict(local_index=local, source_index=source, scale=6,
                                             distribution="synthetic_test_only", capacity_type="normalized"))
                with (directory / "dataset.pkl").open("wb") as handle:
                    pickle.dump([vrp(problem) if problem in r.ELIGIBLE else [] for _ in ids], handle)
                pool = POOL if "B" in problem else [dict(method="BQ", global_id=0, source_method="BQ")]
                r.write_new_json(directory / "metadata.json", dict(problem=problem, split=split, instances=len(ids),
                    methods=[c["method"] for c in pool], global_solver_ids=[c["global_id"] for c in pool],
                    files={name: dict(sha256=r.file_hash(directory / name)) for name in ("dataset.pkl", "index_mapping.csv")}))

    def tearDown(self):
        self.temp.cleanup()

    def selection(self):
        return r.freeze_selection(self.data, version="synthetic_test_only", count=2)

    def test_freeze_tracks_source_ids_and_only_train(self):
        selected = self.selection()
        self.assertEqual(selected, self.selection())
        self.assertEqual(len(selected["bases"]), 32)
        self.assertFalse(selected["original_costs_read"])
        for b in selected["bases"]:
            self.assertEqual(b["split"], "train")
            self.assertLess(b["source_index"], 12)
            self.assertIn(str(b["source_index"]), b["source_base_id"])
        self.assertEqual(set(selected["not_applicable"]), {"TSP", "ATSP"})

    def test_membership_corruption_and_deleted_candidates_fail(self):
        with (self.data / "VRPBtrain/index_mapping.csv").open("a") as handle:
            handle.write("12,999,6,synthetic_test_only,normalized\n")
        with self.assertRaisesRegex(ValueError, "membership"):
            self.selection()
        meta_path = self.data / "VRPBTWtrain/metadata.json"
        meta = r.read_json(meta_path)
        meta["methods"].append("MoSES_RF")
        meta["global_solver_ids"].append(13)
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Deleted B"):
            r.load_pool(self.data, "VRPBTW")

    def test_immutable_artifacts(self):
        path = self.root / "immutable.json"
        r.write_new_json(path, dict(value=1))
        r.write_new_json(path, dict(value=1))
        with self.assertRaisesRegex(ValueError, "Immutable"):
            r.write_new_json(path, dict(value=2))

    def test_small_cpu_preparation_all_pool_and_lineage(self):
        selection = self.selection()
        manifest = r.prepare_inputs(selection, self.root / "prepared", ["VRPBTW"], 1)
        self.assertEqual(manifest["endpoint_count"], 5)
        self.assertEqual(manifest["pair_count"], 4)
        self.assertEqual(manifest["task_count"], 15)
        self.assertEqual({t["source_method"] for t in manifest["tasks"]}, {"MTPOMO", "MVMOE", "RELD_MOEL"})
        self.assertEqual(len({e["base_id"] for e in manifest["endpoints"]}), 1)
        for method in ("MTPOMO", "MVMOE", "reld_mt"):
            self.assertEqual(len({t["solver_seed"] for t in manifest["tasks"] if t["method"] == method}), 1)
        r.verify_manifest(manifest)
        masked = r.assemble_labels(manifest, self.root / "nonexistent_results", self.root / "masked.json")
        self.assertEqual(masked["summary"]["full_pool_winners"], 0)
        self.assertEqual(masked["summary"]["verified_candidate_costs"], 0)
        self.assertTrue(all(all(row["failure_mask"]) for row in masked["endpoints"]))

    def test_dry_run_does_not_import_backend(self):
        manifest = r.prepare_inputs(self.selection(), self.root / "prepared", ["VRPBTW"], 1)
        with patch.object(r.importlib, "import_module", side_effect=AssertionError("must not import solver")):
            result = r.dispatch(manifest, self.root / "dry.json")
        self.assertFalse(result["solver_execution"])
        with self.assertRaisesRegex(ValueError, "Execution requires"):
            r.dispatch(manifest, self.root / "result.json", execute=True)

    def test_tampered_input_and_incomplete_task_pool_fail(self):
        manifest = r.prepare_inputs(self.selection(), self.root / "prepared", ["VRPBTW"], 1)
        broken = copy.deepcopy(manifest)
        broken["tasks"].pop()
        broken.pop("manifest_sha256")
        broken["manifest_sha256"] = r.canonical_hash(broken)
        with self.assertRaisesRegex(ValueError, "whole legal pool"):
            r.verify_manifest(broken)
        path = Path(manifest["endpoints"][0]["input_path"])
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "lineage"):
            r.verify_manifest(manifest)

    def test_cross_base_pair_and_wrong_candidate_source_rejected(self):
        manifest = r.prepare_inputs(self.selection(), self.root / "prepared", ["VRPBTW"], 2)
        for kind in ("pair", "source"):
            broken = copy.deepcopy(manifest)
            if kind == "pair":
                broken["pairs"][0]["original_id"] = broken["endpoints"][5]["endpoint_id"]
            else:
                broken["tasks"][0]["source_method"] = "RELD_MTL"
            broken.pop("manifest_sha256")
            broken["manifest_sha256"] = r.canonical_hash(broken)
            with self.assertRaises(ValueError):
                r.verify_manifest(broken)

    def test_cli_rejects_writes_outside_r114(self):
        with self.assertRaisesRegex(ValueError, "writes only"):
            r.cli_output_path(self.data / "new_labels.json")

    def test_ingestion_requires_lock_and_reaudits_route(self):
        manifest = r.prepare_inputs(self.selection(), self.root / "prepared", ["VRPTW"], 1)
        lock = dict(version=manifest["version"], manifest_sha256=manifest["manifest_sha256"], adapter_entry=r.ADAPTER_ENTRY,
                    deployments={"VRPTW/BQ": dict(profile_sha256="synthetic_test_only")})
        lock["lock_sha256"] = r.canonical_hash(lock)
        results = self.root / "results"
        results.mkdir()
        task = manifest["tasks"][0]
        receipt = {k: task[k] for k in ("task_id", "endpoint_id", "base_id", "problem", "method", "input_sha256", "solver_seed")}
        receipt.update(version=manifest["version"], lock_sha256=lock["lock_sha256"], source="fresh_solver",
                       status="ok", cost=1, tour=[0, 1, 0], route_audit=dict(verified=True), profile_sha256="synthetic_test_only")
        r.write_new_json(results / f"{task['task_id']}.json", receipt)
        with self.assertRaisesRegex(ValueError, "requires a deployment lock"):
            r.assemble_labels(manifest, results, self.root / "never.json")
        labels = r.assemble_labels(manifest, results, self.root / "reaudited.json", lock)
        self.assertEqual(labels["endpoints"][0]["statuses"], ["invalid_route"])
        self.assertEqual(labels["summary"]["verified_candidate_costs"], 0)

    def test_complete_ingestion_and_provenance_mismatch(self):
        manifest = r.prepare_inputs(self.selection(), self.root / "prepared", ["VRPTW"], 1)
        lock = dict(version=manifest["version"], manifest_sha256=manifest["manifest_sha256"], adapter_entry=r.ADAPTER_ENTRY,
                    deployments={"VRPTW/BQ": dict(profile_sha256="synthetic_test_only")})
        lock["lock_sha256"] = r.canonical_hash(lock)
        results = self.root / "results"
        endpoints = {e["endpoint_id"]: e for e in manifest["endpoints"]}
        for task in manifest["tasks"]:
            item = r.read_json(endpoints[task["endpoint_id"]]["input_path"])["item"]
            f = r.fields(task["problem"], item)
            tour = [0]
            for node in range(1, len(f["q"]) + 1):
                tour.extend((node, 0))
            cost = float(2 * np.linalg.norm(f["xy"] - f["depot"], axis=1).sum())
            receipt = {k: task[k] for k in ("task_id", "endpoint_id", "base_id", "problem", "method", "input_sha256", "solver_seed")}
            receipt.update(version=manifest["version"], lock_sha256=lock["lock_sha256"], source="fresh_solver",
                           status="ok", cost=cost, tour=tour, profile_sha256="synthetic_test_only")
            r.write_new_json(results / f"{task['task_id']}.json", receipt)
        labels = r.assemble_labels(manifest, results, self.root / "complete.json", lock)
        self.assertEqual(labels["summary"]["full_pool_winners"], 5)
        self.assertEqual(labels["summary"]["full_pool_pairs"], 4)
        self.assertTrue(all(row["winner"] == "BQ" for row in labels["endpoints"]))
        self.assertTrue(all(pair["Delta_mn"] == [[0.0]] for pair in labels["pairs"]))
        receipt["input_sha256"] = "wrong_input"
        (results / f"{task['task_id']}.json").write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "provenance"):
            r.assemble_labels(manifest, results, self.root / "never.json", lock)

    def test_source_inventory_and_lock_detect_changed_weights(self):
        legacy = self.root / "bridge/exports/test_only"
        scripts = legacy.parents[1] / "scripts"
        scripts.mkdir(parents=True)
        for name in ("train_100k_sampling.py", "train_100k_alias_weights.py", "train_100k_backends.py", "train_100k_common.py"):
            (scripts / name).write_text("# Synthetic fixture only; not a solver.\n")
        r.write_new_json(scripts / "train_100k_environment.json", dict(synthetic_test_only=True))
        r.write_new_json(legacy / "recipe_decisions.json", dict(mvrp_decoder_strategy="sampling"))
        (legacy / "python_compat").mkdir()
        weight = self.root / "synthetic_test_only_weight.bin"
        weight.write_bytes(b"synthetic_test_only")
        for problem in r.PROBLEMS:
            for candidate in r.load_pool(self.data, problem):
                root = legacy / "labels" / problem / candidate["source_method"]
                profile = dict(backend="synthetic_test_only", method=candidate["source_method"],
                               checkpoint=str(weight), checkpoint_sha256=r.file_hash(weight),
                               budget=dict(decoding="synthetic_test_only"))
                r.write_new_json(root / "profile.json", profile)
                r.write_new_json(root / "dependency_contract.json", dict(artifacts={str(weight): r.file_hash(weight)}))
                r.write_new_json(root / "profile_path_bindings.json", dict(bindings={}))
        inventory = r.inventory_sources(self.data, legacy)
        self.assertEqual(inventory["summary"]["missing_deployments"], 0)
        self.assertEqual(inventory["summary"]["changed_deployments"], 0)
        self.assertEqual(inventory["adapter_entry"], "train_100k_sampling:make_confirmed_backend")
        manifest = r.prepare_inputs(self.selection(), self.root / "prepared", ["VRPTW"], 1)
        lock = r.source_lock(inventory, manifest)
        r.verify_lock(lock, manifest, verify_files=True)
        weight.write_bytes(b"changed_synthetic_test_only")
        with self.assertRaisesRegex(ValueError, "changed"):
            r.verify_lock(lock, manifest, verify_files=True)
        changed = r.inventory_sources(self.data, legacy)
        self.assertGreater(changed["summary"]["changed_deployments"], 0)
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            r.source_lock(changed, manifest)


if __name__ == "__main__":
    unittest.main()
