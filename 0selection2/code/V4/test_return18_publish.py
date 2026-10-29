"""Four bounded CPU tests. All Git operations and model gates are mocked."""

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
