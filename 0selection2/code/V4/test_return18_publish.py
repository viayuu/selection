"""Bounded CPU fixtures. All Git operations and model gates are mocked."""

import csv
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import return18_publish as pub


class PublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.git_root = Path(temporary.name)
        self.repo = self.git_root / '0selection2'
        self.v4 = self.repo / 'code/V4'
        self.shared = self.v4 / 'runs/Return18_post_R120'
        self.root = self.v4 / 'runs/R122_hard_decision_constraints'
        (self.git_root / '.git').mkdir()
        self.root.mkdir(parents=True)
        for name, value in dict(GIT_ROOT=self.git_root, REPO=self.repo, V4=self.v4, RETURN18=self.shared).items():
            active = patch.object(pub, name, value)
            active.start()
            self.addCleanup(active.stop)
        self.lock = dict(checkpoint_sha256='a' * 64, state_hash='b' * 64, selected_update=14400)
        self.pdf = self.put(self.root / 'model/full_model_statistics/REPORT.pdf', 'PDF fixture')
        self.files = {str(self.pdf.relative_to(self.git_root)): pub.sha(self.pdf)}

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return path

    def ready(self):
        for target, value in (('completion_gate', (self.lock, [self.pdf])), ('curated_paths', self.files)):
            active = patch.object(pub, target, return_value=value)
            active.start()
            self.addCleanup(active.stop)

    def r121_fixture(self, numeric=False):
        root = self.v4 / 'runs/R121_structural_families/preparation/learner_v1'
        run, output = root / 'principal', root / 'principal/full_model_statistics'
        collection = root.parent / ('collection_numeric_boundary_v3' if numeric else 'collection_explicit_cpu_v2')
        layer, version, followup_name, immutable_key = pub.R121_COLLECTIONS[collection.name]
        layer_key = layer + '_sha256'
        for name in (pub.CODE['R121'] + ' ' + pub.COMMON).split():
            self.put(self.v4 / name, 'frozen source fixture')
        for name in ('CHECKS.json', 'evidence.json'):
            self.put(self.shared / 'R120_closeout' / name, {})
        science = self.repo / 'docs/research/return18_major_routes_20261030'
        for name in ('REPORT.md', 'HANDOFF.md', 'sources.csv', 'directions.csv', 'ROUTING.json',
                     'routing_receipt.json', 'astra_followup.md', 'astra_followup_sources.csv'):
            self.put(science / name, 'identity caveat unchanged')
        science = self.repo / 'docs/research/frozen_selector_mixture_information_20261030'
        for name in pub.R121_SCIENCE:
            self.put(science / name, 'fixed small science delivery')
        self.put(science / 'formula_assets/raw.png', 'excluded original figure')
        def seal(value, key):
            digest = pub.hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            return dict(value, **{key: digest})
        def verify(value, key):
            fields = {k: v for k, v in value.items() if k != key}
            if seal(fields, key)[key] != value[key]:
                raise ValueError('Fixture seal changed')
        pools = {p: ['m' + str(j) for j in range(4 if i == 17 else 5)]
                 for i, p in enumerate(['TSP', 'CVRP'] + ['TASK' + str(i) for i in range(16)])}
        groups = sorted(p + '/' + m for p, pool in pools.items() for m in pool)
        raw = self.put(collection / 'family_labels.jsonl', 'raw labels must never be staged')
        input_manifest = self.put(root.parent / 'inputs/manifest.json', 'DO NOT PARSE OR STAGE THIS INPUT MANIFEST')
        protocol = seal(dict(group_order=groups, planned_labels=113920, lanes=4, source_sha256={},
                            manifest_sha256='input-seal', input_manifest=str(input_manifest)), 'collection_sha256')
        metadata = seal(dict(version=version,
            native_collection_sha256=protocol['collection_sha256'], source_sha256={},
            **{immutable_key: {}}, checker_version='r121_independent_L_fp32_plus_one_ulp_v3_20261030',
            source_root_by_deployment={g: str(root.parent / 'collection_persistent_v1') for g in groups}), layer_key)
        self.put(collection / 'protocol.json', protocol)
        self.put(collection / (layer + '.json'), metadata)
        export = dict(complete=True, full18=True, candidate_pool_unchanged=True, historical_labels_replaced=False,
            labels=113920, endpoints=23040, path=str(raw), sha256=pub.sha(raw), native_receipt_root=str(collection),
            **{layer_key: metadata[layer_key], layer + '_version': version},
            independent_numeric_version=metadata['checker_version'], source_policy=metadata['source_root_by_deployment'],
            collection_sha256=protocol['collection_sha256'], manifest_sha256='input-seal', family_input_manifest=str(input_manifest))
        self.put(collection / 'family_labels.receipt.json', export)
        if numeric:
            for name in pub.R121_NUMERIC_METADATA:
                if not (root.parent / name).exists():
                    self.put(root.parent / name, dict(diagnostic_only=True) if name.endswith('.json') else 'numeric boundary evidence')
            self.put(collection / 'labels/raw_task.json', dict(raw_tour=[0, 1, 0]))
            for lane in (2, 3):
                self.put(self.shared / 'dispatch' / ('r121_universal_parallel_lane' + str(lane) + '_v2') /
                    'main_pending_only_replacement_v3.json', dict(token='do-not-publish-transport-token'))
        for lane in range(4):
            self.put(collection / 'bootstrap' / ('lane' + str(lane) + '.json'), seal(dict(
                status='bootstrap_complete', lane=lane, collection_sha256=export['collection_sha256'],
                **{layer_key: export[layer_key]}), 'bootstrap_sha256'))
        for group in groups:
            name = group.replace('/', '__') + '.json'
            self.put(collection / 'progress' / name, dict(completed=1280, next_index=1280,
                status='complete', collection_sha256=export['collection_sha256']))
            self.put(collection / 'preflight' / name, {})
        config = dict(candidate_pools=pools, source_sha256={'r121_experiment.py': pub.sha(self.v4 / 'r121_experiment.py')},
            data_root='/frozen/new90k', model_factory='code.V4.r121_experiment.load_checkpoint', family_labels=export,
            family_population=dict(family_endpoints=23040, strict_unseen_base=False),
            train_diagnostic=dict(population='fixed eligible TRAIN subset, NOT complete TRAIN', max_rows_per_problem=1024),
            natural_ce_eligible_rows_per_problem={p: 80000 for p in pools},
            loss=dict(structural='unchanged'), exposures=dict(logical_input_slots=14745600), historical_control='not matched')
        lock = dict(self.lock, epoch=20, inference_rule='original stable-ID maximin')
        self.put(run / 'protocol.json', config)
        self.put(run / 'training_complete.json', dict(complete=True, selected_best=lock, exposure=config['exposures']))
        results = {}
        for split in ('val', 'test'):
            identity = dict(experiment='R121', arm='principal', split=split, successful_updates=14400,
                state_hash=lock['state_hash'], model_state_sha256='one-full-model',
                checkpoint_sha256=lock['checkpoint_sha256'] if split == 'test' else None)
            results[split] = dict(identity=identity, macro=dict(macro_top1=.7), per_problem={p: dict(n=4500) for p in pools})
            self.put(run / 'evaluation' / split / 'selected_best/identity.json', identity)
            self.put(run / 'evaluation' / split / 'selected_best/rank000.npz', 'never-stage predictions')
        self.put(run / 'lib_stage_complete.json', dict(complete=True, experiment='R121',
            checkpoint_sha256=lock['checkpoint_sha256'], datasets=['TSPLIB', 'CVRPLIB']))
        libs = {d: dict(protocol=dict(model_factory=config['model_factory'], locked_inference_rule=lock['inference_rule']))
                for d in ('TSPLIB', 'CVRPLIB')}
        for dataset, value in libs.items():
            self.put(run / 'lib_evaluation' / (dataset + '_results.json'), value)
        report_protocol = dict(config, experiment='R121', arm='principal', complete=True,
                               checkpoint_lock=lock, exposure=config['exposures'])
        for name in pub.R121_REPORTS:
            self.put(output / name, report_protocol if name == 'protocol.json' else '%PDF-fixture' if name.endswith('.pdf') else 'metric,value\ntop1,70\n')
        receipt = dict(lock, experiment='R121', arm='principal', complete=True, all18=True, native_libs=['TSPLIB', 'CVRPLIB'],
            VAL=results['val']['macro'], TEST=results['test']['macro'], model_factory=config['model_factory'],
            family_export=export, family_population=config['family_population'], loss=config['loss'],
            train_diagnostic=config['train_diagnostic'],
            exposure=config['exposures'], historical_control=config['historical_control'],
            artifacts={name: dict(path=str(output / name), sha256=pub.sha(output / name), bytes=(output / name).stat().st_size) for name in pub.R121_REPORTS})
        self.put(run / 'delivery_manifest.json', receipt)
        def check_identity(record, expected, *args, **kwargs):
            if any(record[k] != expected[k] for k in ('checkpoint_sha256', 'state_hash', 'selected_update')):
                raise ValueError('Delivery checkpoint identity changed')
        learner = SimpleNamespace(locked_training=Mock(return_value=(run, config, lock)), structural=SimpleNamespace(verify_seal=verify))
        common = SimpleNamespace(locked_inputs=Mock(return_value=(run, config, lock, results['test'], results['val'])),
            check_identity=check_identity, selected_val_path=lambda r, k: run / 'evaluation/val/selected_best/rank000.npz')
        reports = SimpleNamespace(saved_libs=Mock(return_value=libs))
        modules = {'code.V4.r121_experiment': learner, 'code.V4.r80_reporting': common, 'code.V4.r84_reporting': reports}
        def artifact(path):
            path = Path(path).resolve()
            return dict(path=str(path), sha256=pub.sha(path), bytes=path.stat().st_size)
        def finish_identity(lock, model_hash):
            return dict(lock, schema='r121_learner_finish_v1', experiment='R121', arm='principal', model_state_sha256=model_hash)
        def finish_check(record, expected):
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError('Capture model differs')
        def completed(path, expected):
            value = pub.read(path)
            finish_check(value, expected)
            if (value.get('complete') is not True or any(pub.sha(p) != h for p, h in value['input_files_sha256'].items())
                    or any(artifact(item['path']) != item for item in value['artifacts'].values())):
                raise ValueError('Supplemental saved artifact/input changed or incomplete')
            return value
        modules['code.V4.r121_finish'] = SimpleNamespace(REPORT_FILES=pub.R121_SUPPLEMENT, artifact=artifact,
            identity_for=finish_identity, check_identity=finish_check, completed_receipt=completed)
        supplement = run / 'learner_finish'
        identity = finish_identity(lock, 'one-full-model')
        raw_capture = self.put(supplement / 'family_predictions/TSP.npz', 'excluded family predictions')
        inputs = {str(raw_capture): pub.sha(raw_capture), str(run / 'delivery_manifest.json'): pub.sha(run / 'delivery_manifest.json')}
        family = dict(identity, complete=True, all18=True, phase='locked_family_capture', endpoints=23040,
            family_baselines=['zero_potential', 'family_train_mean_potential'], input_files_sha256=inputs,
            artifacts={'TSP.npz': artifact(raw_capture)})
        train = dict(identity, complete=True, all18=True, phase='locked_full_train_capture', train_complete=True,
            train_rows_per_problem=81000, train_eligible_rows_per_problem=config['natural_ce_eligible_rows_per_problem'],
            TRAIN=dict(macro_top1=.72), input_files_sha256=inputs, artifacts={})
        self.put(supplement / 'capture_train_receipt.json', train)
        self.put(supplement / 'family_capture_receipt.json', family)
        for name in pub.R121_SUPPLEMENT:
            self.put(supplement / 'reports' / name, identity if name == 'protocol.json' else 'signed supplemental fixture')
        supplemental_receipt = dict(identity, complete=True, all18=True, phase='complete_post_TEST_learner_finish',
            train_complete=True, train_rows_per_problem=81000,
            train_eligible_rows_per_problem=config['natural_ce_eligible_rows_per_problem'], TRAIN=train['TRAIN'],
            VAL=receipt['VAL'], TEST=receipt['TEST'], family_diagnostics_complete=True, fixed36_count=36,
            family_baselines=family['family_baselines'], model_inference=False, GPU_execution=False,
            solver_execution=False, training=False, checkpoint_reselection=False, research_goal_complete=False,
            source_sha256={'r121_finish.py': pub.sha(self.v4 / 'r121_finish.py')}, input_files_sha256=inputs,
            base_delivery_manifest=artifact(run / 'delivery_manifest.json'),
            capture_train_receipt=artifact(supplement / 'capture_train_receipt.json'),
            family_capture_receipt=artifact(supplement / 'family_capture_receipt.json'),
            artifacts={name: artifact(supplement / 'reports' / name) for name in pub.R121_SUPPLEMENT})
        self.put(supplement / 'reports/receipt.json', supplemental_receipt)
        followup = root.parent / followup_name
        plan = self.put(self.shared / 'dispatch/main_approved_r121_principal_v1.json', dict(main_approved=True))
        self.put(followup / 'README.md', 'frozen principal followup')
        self.put(followup / 'contract.json', dict(main_approved=True, root=str(followup), learner_root=str(root),
            collection_root=str(collection), planned_labels=113920, endpoints=23040, resource_plan=str(plan),
            source_sha256={'code/V4/' + ('r121_followup_numeric.py' if numeric else 'r121_followup.py'):
                pub.sha(self.v4 / ('r121_followup_numeric.py' if numeric else 'r121_followup.py'))}, immutable_inputs=[]))
        self.put(followup / 'learner_launch.json', dict(plan=str(plan), plan_sha256=pub.sha(plan), protocol=dict(
            path=str((run / 'protocol.json').relative_to(self.repo)), sha256=pub.sha(run / 'protocol.json'), state_hash=lock['state_hash'])))
        for name in pub.R121_STAGES:
            self.put(followup / (name + '.json'), dict(stage=name, state='complete', exit_status=0))
        for name in ('status.json', 'publish.json', 'complete.json', 'claim.json'):
            self.put(followup / name, dict(token='never-stage'))
        rows = [dict(experiment=exp, split=split, top1_pct=70.) for exp in ('R121', 'R122', 'R123', 'R124') for split in ('VAL', 'TEST')]
        current = dict(all_four_complete=True, checkpoint_sha256={exp: lock['checkpoint_sha256'] for exp in ('R121', 'R122', 'R123', 'R124')})
        final_output = self.shared / 'final_results'
        self.put(final_output / 'RESULTS.pdf', 'four locked versions fixture')
        with (final_output / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        self.put(final_output / 'evidence.json', dict(current, artifacts={name: dict(path=str(final_output / name),
            sha256=pub.sha(final_output / name)) for name in ('RESULTS.pdf', 'comparison.csv')}))
        modules['code.V4.return18_results'] = SimpleNamespace(collect=Mock(return_value=(rows, current)))
        self.put(self.v4 / 'runs/R123_heterogeneous_moe/model/final_test_lock.json', lock)
        moe = self.shared / 'moe_support'
        for name in (*pub.MOE_REPORTS, 'lp_residual_cases.json', 'CHECKS.json'):
            self.put(moe / name, 'small frozen support fixture')
        self.put(moe / 'evidence.json', dict(complete=True, checkpoint=lock, data_root=config['data_root'],
            source_sha256=pub.sha(self.v4 / 'return18_moe_support.py'), no_nn_inference=True, no_gpu=True,
            no_solver=True, no_training=True, interpretation=dict(deployable=False, model_accuracy=False),
            artifacts_sha256={name: pub.sha(moe / name) for name in (*pub.MOE_REPORTS, 'lp_residual_cases.json')}))
        return root, collection, groups, modules

    def test_r121_saved_delivery_gate_and_principal_only_allowlist(self):
        root, collection, groups, modules = self.r121_fixture()
        with patch.object(pub.importlib, 'import_module', side_effect=modules.__getitem__), patch.object(pub, 'git') as git:
            result = pub.publish(root, 'R121')
            self.assertTrue(result['ready'])
            self.assertFalse(result['applied'])
            included = {str(self.git_root / p) for p in result['files']}
            self.assertIn(str(root / 'principal/delivery_manifest.json'), included)
            self.assertIn(str(collection / 'family_labels.receipt.json'), included)
            for path in (root / 'principal/learner_finish/reports/receipt.json',
                         root.parent / 'followup_principal_v1/four_route_analysis.json',
                         self.shared / 'final_results/RESULTS.pdf', self.shared / 'moe_support/fixed36_support.csv',
                         self.v4 / 'return18_moe_support.py', self.v4 / 'r121_followup.py',
                         self.repo / 'docs/research/frozen_selector_mixture_information_20261030/REPORT.pdf'):
                self.assertIn(str(path), included)
            self.assertNotIn(str(collection / 'family_labels.jsonl'), included)
            self.assertNotIn(str(root.parent / 'inputs/manifest.json'), included)
            self.assertFalse(any('/progress/' in p or '/preflight/' in p or p.endswith('.npz') for p in included))
            self.assertFalse(any('r122_' in p or 'r123_' in p or 'r124_' in p for p in included))
            self.assertFalse(any('/formula_assets/' in p or p.endswith('/lp_residual_cases.json') for p in included))
            self.assertFalse(any(p.endswith('/' + n) for p in included for n in ('status.json', 'claim.json', 'publish.json', 'complete.json')))
            modules['code.V4.r84_reporting'].saved_libs.assert_called_once()
            delivery_path = root / 'principal/delivery_manifest.json'
            delivery = pub.read(delivery_path)
            self.put(delivery_path, dict(delivery, checkpoint_sha256='other-checkpoint'))
            with self.assertRaisesRegex(ValueError, 'checkpoint identity'):
                pub.publish(root, 'R121')
            self.put(delivery_path, delivery)
            lib_path = root / 'principal/lib_stage_complete.json'
            lib = pub.read(lib_path)
            self.put(lib_path, dict(lib, datasets=['TSPLIB']))
            with self.assertRaisesRegex(ValueError, 'both completed'):
                pub.publish(root, 'R121')
            self.put(lib_path, lib)
            source = self.v4 / 'r121_experiment.py'
            frozen = source.read_text()
            source.write_text('changed learner')
            with self.assertRaisesRegex(ValueError, 'frozen principal source'):
                pub.publish(root, 'R121')
            source.write_text(frozen)
            state_path = collection / 'progress' / (groups[0].replace('/', '__') + '.json')
            state = pub.read(state_path)
            self.put(state_path, dict(state, completed=1279))
            with self.assertRaisesRegex(ValueError, 'deployment collection is incomplete'):
                pub.publish(root, 'R121')
            self.put(state_path, state)
            export_path = collection / 'family_labels.receipt.json'
            export = pub.read(export_path)
            self.put(export_path, dict(export, native_receipt_root=str(collection.parent / 'old_overlay')))
            with self.assertRaisesRegex(ValueError, 'single-root'):
                pub.publish(root, 'R121')
            self.put(export_path, export)
            report = root / 'principal/full_model_statistics/structural_objectives.pdf'
            report.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                pub.publish(root, 'R121')
            git.assert_not_called()
            self.assertFalse((root / 'github_publication.json').exists())

    def test_r121_new_capture_followup_and_saved_report_gates_fail_closed(self):
        root, _, _, modules = self.r121_fixture()
        run = root / 'principal'
        config = pub.read(run / 'protocol.json')
        lock = dict(self.lock, epoch=20, inference_rule='original stable-ID maximin')
        with patch.object(pub.importlib, 'import_module', side_effect=modules.__getitem__), patch.object(pub, 'git') as git:
            path = run / 'learner_finish/reports/receipt.json'
            receipt = pub.read(path)
            for changes in (dict(complete=False), dict(model_state_sha256='different-model'), dict(train_rows_per_problem=1024)):
                self.put(path, dict(receipt, **changes))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    pub.publish(root, 'R121')
            self.put(path, receipt)
            stage = root.parent / 'followup_principal_v1/saved_only_finish.json'
            completed_stage = pub.read(stage)
            self.put(stage, dict(completed_stage, state='failed', exit_status=1))
            with self.assertRaisesRegex(ValueError, 'stage is incomplete'):
                pub.publish(root, 'R121')
            self.put(stage, completed_stage)
            report = self.shared / 'final_results/RESULTS.pdf'
            original = report.read_text()
            report.write_text('changed final PDF')
            with self.assertRaisesRegex(ValueError, 'analysis output changed'):
                pub.r121_followup_files(root, config, lock)
            report.write_text(original)
            report = self.shared / 'moe_support/support_summary.csv'
            report.write_text('changed support table')
            with self.assertRaisesRegex(ValueError, 'support report artifact changed'):
                pub.r121_moe_files(config)
            git.assert_not_called()

    def test_r121_missing_delivery_never_imports_learner_or_calls_git(self):
        root = self.v4 / 'runs/R121_structural_families/preparation/learner_v1'
        with patch.object(pub.importlib, 'import_module') as loader, patch.object(pub, 'git') as git:
            with self.assertRaises(FileNotFoundError):
                pub.publish(root, 'R121')
            loader.assert_not_called()
            git.assert_not_called()

    def test_r121_numeric_v3_paths_and_v2_followup_preserve_small_allowlist(self):
        root, collection, _, modules = self.r121_fixture(numeric=True)
        with patch.object(pub.importlib, 'import_module', side_effect=modules.__getitem__), patch.object(pub, 'git') as git:
            result = pub.publish(root, 'R121')
            included = {str(self.git_root / name) for name in result['files']}
            for name in ('r121_numeric_collection.py', 'test_r121_numeric_collection.py',
                         'r121_parallel_collection_v3.py', 'test_r121_parallel_collection_v3.py',
                         'r121_followup_numeric.py', 'test_r121_followup_numeric.py'):
                self.assertIn(str(self.v4 / name), included)
            for name in pub.R121_NUMERIC_METADATA:
                self.assertIn(str(root.parent / name), included)
            self.assertIn(str(root.parent / 'followup_principal_v2/contract.json'), included)
            self.assertIn(str(root.parent / 'followup_principal_v2/four_route_analysis.json'), included)
            self.assertNotIn(str(collection / 'family_labels.jsonl'), included)
            self.assertNotIn(str(collection / 'labels/raw_task.json'), included)
            self.assertFalse(any('/followup_principal_v1/' in name or 'main_pending_only_replacement_v3.json' in name for name in included))
            self.assertFalse(any(name.endswith(('.npz', '/publish.json', '/status.json', '/claim.json')) for name in included))
            config = pub.read(root / 'principal/protocol.json')
            wrong = dict(config, family_labels=dict(config['family_labels'], native_receipt_root=str(root.parent / 'collection_unknown_v4')))
            with self.assertRaisesRegex(ValueError, 'supported single-root'):
                pub.r121_collection_gate(root, wrong, modules['code.V4.r121_experiment'].structural)
            stage = root.parent / 'followup_principal_v2/full89_export.json'
            self.put(stage, dict(stage='full89_export', state='running', exit_status=None))
            with self.assertRaisesRegex(ValueError, 'stage is incomplete'):
                pub.publish(root, 'R121')
            git.assert_not_called()

    def test_read_only_r122_allowlist_excludes_other_versions_and_raw_inputs(self):
        for name in (pub.CODE['R122'] + ' ' + pub.COMMON).split():
            self.put(self.v4 / name, 'source')
        for name in ('CHECKS.json', 'evidence.json'):
            self.put(self.shared / 'R120_closeout' / name, {})
        science = self.repo / 'docs/research/return18_major_routes_20261030'
        for name in ('REPORT.md', 'HANDOFF.md', 'sources.csv', 'directions.csv', 'ROUTING.json',
                     'routing_receipt.json', 'astra_followup.md', 'astra_followup_sources.csv'):
            self.put(science / name, 'identity caveat unchanged')
        unwanted = [self.put(self.v4 / 'r123_experiment.py', 'other dirty source'),
                    self.put(self.root / 'model/best.pt', 'weights'),
                    self.put(self.root / 'preparation/manifest.json', 'raw manifest'),
                    self.put(self.root / 'model/wandb/raw_label.json', 'raw labels'),
                    self.put(self.shared / 'dispatch/main_approved_r123_v1.json', {}),
                    self.put(self.shared / 'dispatch/runtime/worker.json', dict(token='never-publish'))]
        recovery_plan = self.put(self.shared / 'dispatch/main_approved_r122_test_resource_recovery_v1.json', {})
        finish_plan = self.put(self.shared / 'dispatch/main_approved_r122_finish_v1.json', {})
        with patch.object(pub, 'completion_gate', return_value=(self.lock, [self.pdf])), patch.object(pub, 'git') as git:
            result = pub.publish(self.root, 'R122')
            self.assertTrue(result['ready'])
            self.assertFalse(result['applied'])
            self.assertTrue(all(str(p.relative_to(self.git_root)) not in result['files'] for p in unwanted))
            self.assertTrue(all(str(p.relative_to(self.git_root)) in result['files'] for p in (recovery_plan, finish_plan)))
            git.assert_not_called()
            self.assertFalse((self.root / 'github_publication.json').exists())
            for path in unwanted[1:4]:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    pub.curated_paths(self.root, 'R122', [self.pdf, path])
        r123 = self.v4 / 'runs/R123_heterogeneous_moe'
        manifest = self.put(r123 / 'preparation/manifest.json', dict(bank='small typed metadata'))
        r123_pdf = self.put(r123 / 'finish/reports/REPORT.pdf', 'PDF fixture')
        for name in pub.CODE['R123'].split():
            self.put(self.v4 / name, 'source')
        selected = pub.curated_paths(r123, 'R123', [r123_pdf])
        self.assertIn(str(manifest.relative_to(self.git_root)), selected)
        other_manifest = self.put(r123 / 'inputs/manifest.json', {})
        self.assertNotIn(str(other_manifest.relative_to(self.git_root)), selected)
        with self.assertRaises(ValueError):
            pub.curated_paths(r123, 'R123', [r123_pdf, other_manifest])
        with manifest.open('wb') as stream:
            stream.truncate(125 * 1024**2)
        with self.assertRaises(ValueError):
            pub.curated_paths(r123, 'R123', [r123_pdf])

    def test_real_r122_report_layout_and_completion_gate_fail_closed(self):
        output = self.pdf.parent
        csv = self.put(output / 'metrics.csv', 'metric,value\ntop1,68\n')
        artifacts = {p.name: pub.sha(p) for p in (self.pdf, csv)}
        protocol = self.put(output / 'protocol.json', dict(checkpoint_lock=self.lock, outputs=artifacts, source_sha256={}))
        path = self.put(self.root / 'model/report_complete.json', dict(self.lock, complete=True,
                        outputs=artifacts, report_protocol_sha256=pub.sha(protocol)))
        finish = SimpleNamespace(locked_inputs=Mock(return_value=dict(lock=self.lock)), load_libs=Mock(), same_identity=Mock())
        with patch.object(pub.importlib, 'import_module', return_value=finish), patch.object(pub, 'git') as git:
            lock, reports = pub.completion_gate(self.root, 'R122')
            self.assertEqual(lock, self.lock)
            self.assertIn(path, reports)
            finish.load_libs.assert_called_once()
            value = pub.read(path)
            self.put(path, dict(value, complete=False))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                pub.completion_gate(self.root, 'R122')
            self.put(path, value)
            csv.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                pub.completion_gate(self.root, 'R122')
            finish.load_libs.side_effect = FileNotFoundError('LIB not complete')
            with self.assertRaises(FileNotFoundError):
                pub.completion_gate(self.root, 'R122')
            git.assert_not_called()

    def test_existing_staged_changes_are_never_added_to_or_committed(self):
        self.ready()
        def fake(*args):
            if args == ('rev-parse', '--show-toplevel'):
                return str(self.git_root)
            if args == ('branch', '--show-current'):
                return 'main'
            if args == ('diff', '--cached', '--name-only'):
                return 'somebody_else.py'
            self.fail('Unexpected Git operation: ' + repr(args))
        with patch.object(pub, 'git', side_effect=fake), self.assertRaisesRegex(RuntimeError, 'staged changes'):
            pub.publish(self.root, 'R122', apply=True)
        self.assertFalse((self.root / 'github_publication.json').exists())

    def test_r124_pdf_only_receipt_plus_real_csvs_have_explicit_late_hashes(self):
        root = self.v4 / 'runs/R124_top2_reranking/launch_v2'
        run, output = root / 'model', root / 'model/finish/reports'
        config = dict(data_root='/frozen/new90k', candidate_pools={'TSP': ['a'], 'CVRP': ['b']},
                      candidate_pool_ids={'TSP': [1], 'CVRP': [2]})
        identity = dict(self.lock, experiment='R124', winner_rule='native FP64', first_stage_checkpoint_sha256='stage1')
        outputs = {key: str(self.put(output / name, 'PDF fixture')) for key, name in
                   (('report_pdf', 'R124_FULL18.pdf'), ('distributions_pdf', 'selection_distributions.pdf'))}
        receipt = self.put(output / 'finish_receipt.json', dict(complete=True, identity=identity, inputs={},
            outputs=outputs, output_sha256={k: pub.sha(v) for k, v in outputs.items()}))
        receipt_sha = pub.sha(receipt)
        lib = {}
        for dataset in ('TSPLIB', 'CVRPLIB'):
            prediction = self.put(run / 'finish/lib_evaluation/predictions' / (dataset + '.npz'), 'pinned fixture')
            lib[dataset] = dict(complete=True, protocol=identity, predictions_sha256=pub.sha(prediction))
        self.put(run / 'finish/lib_complete.json', dict(complete=True, identity=identity, datasets=lib))
        self.put(run / 'finish/identity.json', identity)
        groups = [('VAL', 'TSP', 4500), ('VAL', 'CVRP', 4500), ('TEST', 'TSP', 4500), ('TEST', 'CVRP', 4500),
                  ('TSPLIB', 'TSP', 49), ('CVRPLIB', 'CVRP', 100)]
        for name, fields in pub.R124_CSV_FIELDS.items():
            nss, methods = name == 'nss_four_datasets.csv', 'method' in fields.split()
            headers = sorted(set(fields.split()) | set(('winner_rule data_version' if nss else 'split problem n winner_rule data_version').split()))
            selected = [g for g in groups if g[0] != 'VAL'] if nss else list(groups)
            if not nss and not methods:
                selected += [('VAL', 'ALL', 9000), ('TEST', 'ALL', 9000)]
            with (output / name).open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                for split, problem, n in selected:
                    row = dict.fromkeys(headers, '0')
                    row.update(n=n, winner_rule=identity['winner_rule'], data_version='new90k' if split in ('VAL', 'TEST') else 'original native ' + (split if nss else 'LIB'))
                    if nss:
                        row.update(dataset=problem + ' TEST' if split == 'TEST' else split,
                            r124_checkpoint_sha256=identity['checkpoint_sha256'], r77b_checkpoint_sha256='stage1', nss_status='complete')
                    else:
                        row.update(split=split, problem=problem)
                        if methods:
                            row.update(method=config['candidate_pools'][problem][0], global_id=config['candidate_pool_ids'][problem][0])
                        if name == 'r77b_comparison.csv':
                            row['first_stage_checkpoint_sha256'] = 'stage1'
                    writer.writerow(row)
        finish = SimpleNamespace(locked_inputs=Mock(return_value=(run, config, self.lock, {}, identity)))
        with patch.object(pub.importlib, 'import_module', return_value=finish), patch.object(pub, 'git') as git:
            with patch.object(pub, 'curated_paths', side_effect=lambda r, e, reports: {str(p): pub.sha(p) for p in reports}):
                result = pub.publish(root, 'R124')
            late = result['late_publication_hashes']
            self.assertTrue(result['ready'])
            self.assertEqual(late['kind'], 'LATE_PUBLICATION_HASH')
            self.assertFalse(late['original_finish_receipt_signed_these_csvs'])
            self.assertEqual({Path(p).name for p in late['csv_sha256']}, set(pub.R124_CSV_FIELDS))
            self.assertEqual(pub.sha(receipt), receipt_sha)
            (output / 'metrics.csv').write_text('wrong_header\n0\n')
            with self.assertRaisesRegex(ValueError, 'headers'):
                pub.completion_gate(root, 'R124')
            git.assert_not_called()

    def test_one_commit_failed_push_retry_and_failed_commit_never_recommits(self):
        self.ready()
        state = dict(head='base', index=set(), push_fail=True, commit_fail=False)
        calls = []
        def fake(*args):
            calls.append(args)
            if args == ('rev-parse', '--show-toplevel'):
                return str(self.git_root)
            if args == ('branch', '--show-current'):
                return 'main'
            if args == ('diff', '--cached', '--name-only'):
                return '\n'.join(sorted(state['index']))
            if args == ('rev-parse', 'HEAD'):
                return state['head']
            if args == ('rev-parse', 'refs/remotes/origin/main'):
                return 'base'
            if args[:3] == ('add', '-f', '--'):
                state['index'].update(args[3:])
            elif args[0] == 'commit':
                if state['commit_fail']:
                    raise subprocess.CalledProcessError(1, ['git', *args])
                state.update(head='new-commit', index=set())
            elif args[0] == 'push':
                if state['push_fail']:
                    raise subprocess.CalledProcessError(128, ['git', *args])
            else:
                self.fail('Unexpected Git operation: ' + repr(args))
            return ''
        receipt = self.root / 'github_publication.json'
        with patch.object(pub, 'git', side_effect=fake):
            with self.assertRaises(subprocess.CalledProcessError):
                pub.publish(self.root, 'R122', apply=True)
            failed = pub.read(receipt)
            self.assertFalse(failed['complete'])
            self.assertEqual(failed['commit'], 'new-commit')
            state['push_fail'] = False
            result = pub.publish(self.root, 'R122', apply=True)
            self.assertTrue(result['complete'])
            self.assertEqual(result['push_returncode'], 0)
            self.assertEqual(result['failures'][0]['returncode'], 128)
            pub.publish(self.root, 'R122', apply=True)
            self.assertEqual(sum(c[0] == 'commit' for c in calls), 1)
            self.assertEqual(sum(c[0] == 'push' for c in calls), 2)
            receipt.unlink()
            state.update(head='base', commit_fail=True)
            with self.assertRaises(subprocess.CalledProcessError):
                pub.publish(self.root, 'R122', apply=True)
            state['index'].clear()
            count = len(calls)
            with self.assertRaisesRegex(RuntimeError, 'no extra auto-commit'):
                pub.publish(self.root, 'R122', apply=True)
            self.assertFalse(any(c[0] in ('add', 'commit', 'push') for c in calls[count:]))


if __name__ == '__main__':
    unittest.main()
