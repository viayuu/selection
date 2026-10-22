"""Small synthetic CPU receipts only. Every Git/compare command is mocked."""

from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r98_goal as goal


def pdf(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'%PDF-synthetic CPU fixture')


def fixture(repo, version):
    root = repo / 'code/V4/runs' / goal.VERSIONS[version]
    launch = repo / 'code/V4/runs/R98_parallel_dispatch'
    locks = {}
    for arm in goal.arms(version):
        run = goal.run_root(version, root, arm)
        config = dict(experiment=version, arm=arm, seed=2, winner_rule='native FP64 ind',
            rows_per_problem=dict(train=81000, val=4500, test=4500),
            candidate_pools={p: ['alpha', 'beta'] for p in goal.PROBLEMS},
            baselines={p: dict(selection_split='train', train_rows=81000) for p in goal.PROBLEMS},
            contexts={p: dict(rows=81000) for p in goal.PROBLEMS},
            context_root=str(root.parent / 'R97_tabular_in_context/context'),
            target_root=str(root / 'targets'),
            optimizer_updates=0, fine_tuning=False, current_query_labels_in_forward=False,
            inference_rule='fixed native decision', public_checkpoint_sha256='w'*64,
            selection_rule='frozen_single_predeclared_bank_no_VAL_search')
        key = 'targets' if version == 'R98' else 'relations'
        config[key] = {p: dict(rows=81000) for p in goal.PROBLEMS}
        config['source_sha256'] = {'fixture.py': 'source'}
        if version == 'R99':
            config.update(max_successful_updates=57600, checkpoint_rule='minimum full18 VAL macro_vs_sbs_pct')
        state = goal.json_hash(config)
        goal.save(run / 'protocol.json', config)
        bank = dict(protocol_sha256=state, weights_updated=False, selection_rule=config['selection_rule'],
                    **{k: config[k] for k in ('contexts', key, 'source_sha256',
                                              'public_checkpoint_sha256', 'inference_rule')})
        goal.save(run / 'checkpoint.json', bank)
        lock = dict(experiment=version, arm=arm, state_hash=state,
            checkpoint_sha256=goal.file_hash(run / 'checkpoint.json'), checkpoint=str(run / 'checkpoint.json'),
            selected_without_test=True, per_problem_assembly=False, selection_rule=config['selection_rule'],
            inference_rule=config['inference_rule'], rule=config.get('checkpoint_rule'), selected_update=1440)
        locks[arm] = lock

        def result(rows):
            return dict(complete=True, state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'],
                macro=dict(macro_top1=.5), per_problem={p: dict(n=rows) for p in goal.PROBLEMS})

        val, test = result(4500), result(4500)
        for name, record in {
            'final_test_lock.json': lock, 'val_results.json': val,
            'test_results.json': dict(test, result=test) if version == 'R99' else test,
            'run_complete.json': dict(complete=True, full18_VAL=True, full18_TEST=True,
                LIB=['TSPLIB', 'CVRPLIB'], checkpoint_sha256=lock['checkpoint_sha256']),
            'test_input_lock.json': dict(checkpoint_sha256=lock['checkpoint_sha256'], labels_loaded=False,
                created_after_unique_VAL_lock=True, inputs={p: {} for p in goal.PROBLEMS}),
            'lib_results.json': {d: dict(complete=True, problem=p, state_hash=state)
                for d, p in (('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP'))},
            'postprocess_status.json': dict(complete=True, phase='complete', exit_status=0,
                checkpoint_sha256=lock['checkpoint_sha256'], state_hash=state),
            'full_model_statistics/provenance.json': dict(experiment=version, state_hash=state,
                checkpoint_sha256=lock['checkpoint_sha256']),
        }.items():
            goal.save(run / name, record)
        if version == 'R99':
            (run / 'run_complete.json').unlink()
            timing = dict(complete=True, checkpoint_sha256=lock['checkpoint_sha256'], state_hash=state)
            goal.save(run / 'inference_timing.json', timing)
            for name, record in {
                'training_complete.json': dict(successful_updates=57600, state_hash=state,
                    updates_by_problem=dict.fromkeys(goal.PROBLEMS, 3200)),
                'final_metrics.json': dict(checkpoint_sha256=lock['checkpoint_sha256'], state_hash=state,
                    selected_update=1440, val=val),
                'lib_stage_complete.json': dict(complete=True, exit_status=0,
                    checkpoint_sha256=lock['checkpoint_sha256'], state_hash=state, datasets=['TSPLIB', 'CVRPLIB']),
                'full_model_statistics/protocol.json': dict(experiment='R99', complete=True, checkpoint_lock=lock,
                    inference_timing=dict(sha256=goal.file_hash(run / 'inference_timing.json'))),
            }.items():
                goal.save(run / name, record)
        for name in goal.R99_PDFS if version == 'R99' else goal.FROZEN_PDFS:
            pdf(run / 'full_model_statistics' / name)
        pdf(run / 'RESULTS.pdf')
        run_id = version + '_' + arm
        goal.save(launch / 'specs' / (run_id + '.json'), dict(run_id=run_id, run_dir=str(run),
            stages=[dict(name='GPU_pipeline', device='gpu'), dict(name='PDF', device='cpu')]))
        worker = dict(state='complete', exit_status=0,
            stages=[dict(name='GPU_pipeline', device='gpu', state='complete', exit_status=0)])
        controller = dict(state='complete', exit_status=0,
            stages=[dict(name=name, state='complete', exit_status=0) for name in ('allocation_GPU_pipeline', 'PDF')])
        goal.save(launch / run_id / 'worker.json', worker)
        goal.save(launch / run_id / 'controller.json', controller)
    return root, launch, locks


def comparison(root, locks):
    goal.save(root / 'comparison_artifacts/protocol.json', dict(experiment='R99', complete=True,
        exit_status=0, checkpoints=locks))
    pdf(root / 'comparison_artifacts/COMPARISON.pdf')
    pdf(root / 'RESULTS.pdf')


class GoalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r98-goal-cpu-')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / '0selection2'

    def test_exact_flag_cli_and_import_are_cpu_only(self):
        args = goal.parser().parse_args(['--version', 'R100', '--watch', '--publish-github'])
        self.assertTrue(args.watch and args.publish_github)
        self.assertEqual(args.version, 'R100')
        self.assertEqual(goal.LAUNCH.name, 'R98_parallel_dispatch')
        script = '''
import importlib.abc, os, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'numpy', 'tabicl', 'wandb'):
            raise AssertionError('heavy/model import: ' + name)
sys.meta_path.insert(0, NoModels())
before = dict(os.environ)
from code.V4 import r98_goal
assert before == dict(os.environ)
assert 'code.V4.r98_experiment' not in sys.modules
assert 'code.V4.r99_reporting' not in sys.modules
'''
        process = subprocess.run([sys.executable, '-B', '-c', script], cwd=goal.REPO,
                                 capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_frozen_versions_require_all_gpu_stages_native_full18_and_bound_pdf(self):
        for version in ('R98', 'R100'):
            root, launch, locks = fixture(self.repo, version)
            current = goal.status(version, root, launch, persist=False)
            self.assertTrue(current['complete'], current)
            self.assertFalse(current['goal_complete'])
            self.assertEqual(current['goal_status'], 'active_unresolved')
            worker_path = launch / (version+'_A') / 'worker.json'
            worker = goal.optional(worker_path)
            goal.save(worker_path, dict(worker, stages=[]))
            self.assertFalse(goal.status(version, root, launch, persist=False)['complete'])
            goal.save(worker_path, worker)
            result_path = root / 'test_results.json'
            result = goal.optional(result_path)
            bad = copy.deepcopy(result)
            bad['per_problem'].pop('ATSP')
            goal.save(result_path, bad)
            self.assertFalse(goal.status(version, root, launch, persist=False)['complete'])
            goal.save(result_path, result)
            provenance_path = root / 'full_model_statistics/provenance.json'
            provenance = goal.optional(provenance_path)
            goal.save(provenance_path, dict(provenance, checkpoint_sha256='other'))
            self.assertFalse(goal.status(version, root, launch, persist=False)['complete'])
            goal.save(provenance_path, provenance)
            (root / 'RESULTS.pdf').write_bytes(b'not a PDF')
            self.assertFalse(goal.status(version, root, launch, persist=False)['complete'])

    def test_r99_original_report_schema_waits_for_both_arms_then_cpu_compare_once(self):
        root, launch, locks = fixture(self.repo, 'R99')
        current = goal.status('R99', root, launch, persist=False)
        self.assertTrue(current['all_arms_ready'], current)
        self.assertFalse(current['complete'])
        self.assertFalse((root / 'A/run_complete.json').exists())
        failed_path = launch / 'R99_B/worker.json'
        worker = goal.optional(failed_path)
        goal.save(failed_path, dict(worker, state='running'))
        with patch.object(goal.subprocess, 'run') as command:
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, launch)
        command.assert_not_called()
        goal.save(failed_path, worker)

        def run(argv, **kwargs):
            self.assertEqual(argv[4:6], ['code.V4.r99_reporting', 'compare'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertIn('--r89-root', argv)
            comparison(root, locks)
            return SimpleNamespace(returncode=0)

        with patch.object(goal, 'comparison_available', return_value=True), patch.object(goal.subprocess, 'run', side_effect=run) as command:
            goal.compare_once(root, launch, r89_root=self.base / 'r89')
            self.assertTrue(goal.status('R99', root, launch, persist=False)['complete'])
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, launch)
        self.assertEqual(command.call_count, 1)

    def test_quiet_unchanged_status_and_watch_preserve_failure_receipts(self):
        root, launch, _ = fixture(self.repo, 'R98')
        first = goal.status('R98', root, launch)
        path = root / 'goal_progress.json'
        timestamp = path.stat().st_mtime_ns
        self.assertEqual(goal.status('R98', root, launch), first)
        self.assertEqual(path.stat().st_mtime_ns, timestamp)
        waiting = dict(first, complete=False, all_arms_ready=False, phase='waiting_for_arms')
        output = io.StringIO()
        with patch.object(goal, 'status', side_effect=[waiting, dict(waiting, updated_utc='later'), first]), \
                patch.object(goal.time, 'sleep') as sleep, redirect_stdout(output):
            self.assertEqual(goal.watch('R98', root, launch, interval=1), 0)
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertEqual(sleep.call_count, 2)
        self.assertFalse(goal.optional(root / 'goal_watch_receipt.json')['goal_complete'])
        other_root, other_launch, _ = fixture(self.repo, 'R100')
        failure = other_launch / 'R100_A/worker.json'
        record = goal.optional(failure)
        record['stages'][0].update(state='failed', exit_status=7)
        goal.save(failure, record)
        original = failure.read_bytes()
        with patch.object(goal.subprocess, 'run') as command, patch.object(goal.time, 'sleep') as sleep, redirect_stdout(io.StringIO()):
            self.assertEqual(goal.watch('R100', other_root, other_launch), 1)
            receipt = (other_root / 'goal_watch_receipt.json').read_bytes()
            with self.assertRaises(RuntimeError):
                goal.watch('R100', other_root, other_launch)
            self.assertEqual((other_root / 'goal_watch_receipt.json').read_bytes(), receipt)
        self.assertEqual(failure.read_bytes(), original)
        command.assert_not_called()
        sleep.assert_not_called()

    def test_curated_files_include_metadata_but_not_runtime_arrays_or_future_versions(self):
        root, launch, _ = fixture(self.repo, 'R98')
        good = ['code/V4/r98_model.py', 'code/V4/test_r98_goal.py',
            'code/V4/runs/R98_performance_context/runtime/runtime_metadata.json',
            'code/V4/runs/R98_performance_context/runtime/provenance.json',
            'code/V4/runs/R98_performance_context/targets/TSP/identity.json',
            'code/V4/runs/R97_tabular_in_context/context/TSP/identity.json']
        bad = ['code/V4/r99_model.py', 'code/V4/r100_model.py',
            'code/V4/runs/R98_performance_context/runtime/python/package/metadata.json',
            'code/V4/runs/R98_performance_context/runtime/weights/public.ckpt',
            'code/V4/runs/R98_performance_context/targets/TSP/G.npy',
            'code/V4/runs/R98_performance_context/predictions.npz',
            'code/V4/runs/R98_performance_context/predictions.json',
            'code/V4/runs/R98_performance_context/outcome/data.json',
            'code/V4/runs/R98_performance_context/wandb_receipt.json',
            'code/V4/runs/R98_performance_context/wandb/config.json']
        for name in [*good, *bad]:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}' if path.suffix == '.json' else 'synthetic', encoding='utf-8')
        goal.save(root / 'secret.json', dict(access_token='never publish'))
        goal.save(root / 'raw_array.json', dict(costs=[[1., 2.]]))
        goal.save(launch / 'R98_A/controller.json', dict(argv=['--token', 'secret']))
        with patch.object(goal, 'REPO', self.repo), patch.object(goal, 'LAUNCH', launch):
            paths = goal.publication_paths('R98', root, launch, self.base)
        expected = {'0selection2/' + name for name in good}
        self.assertTrue(expected.issubset(set(paths)), expected-set(paths))
        self.assertFalse(set(paths) & {'0selection2/' + name for name in bad})
        self.assertFalse(any('secret.json' in name or 'raw_array.json' in name or name.endswith('controller.json') for name in paths))

    def publication_mocks(self, root, launch, fail_push=False, staged_before=False):
        git_dir = self.base / '.git'
        git_dir.mkdir(exist_ok=True)
        state = dict(staged=['unrelated.txt'] if staged_before else [], head='before', commands=[])

        def output(argv, cwd):
            if argv == ['rev-parse', '--show-toplevel']:
                return str(self.base)
            if argv == ['rev-parse', '--absolute-git-dir']:
                return str(git_dir)
            if argv == ['branch', '--show-current']:
                return 'main'
            if argv == ['diff', '--cached', '--name-only', '-z']:
                return '\0'.join(state['staged'])
            if argv == ['rev-parse', 'HEAD']:
                return state['head']
            raise AssertionError(argv)

        def command(argv, **kwargs):
            self.assertEqual(argv[0], 'git')
            state['commands'].append(argv)
            if argv[1] == 'add':
                self.assertEqual(argv[2:4], ['-f', '--'])
                state['staged'].extend(argv[4:])
            elif argv[1] == 'commit':
                state['head'] = 'committed'
            elif argv[1] == 'push' and fail_push:
                raise subprocess.CalledProcessError(1, argv)
            return SimpleNamespace(returncode=0)
        return state, output, command

    def test_publish_is_curated_once_per_version_under_r97_shared_mutex(self):
        root, launch, _ = fixture(self.repo, 'R98')
        state, output, command = self.publication_mocks(root, launch)
        with patch.object(goal, 'REPO', self.repo), patch.object(goal, 'LAUNCH', launch), \
                patch.object(goal, 'git_output', side_effect=output), patch.object(goal.subprocess, 'run', side_effect=command):
            result = goal.publish('R98', root, launch)
            self.assertEqual(goal.publish('R98', root, launch), result)
        self.assertEqual(result['global_mutex'], str(self.base / '.git/codex-publication.lock'))
        self.assertEqual(sum(c[1] == 'commit' for c in state['commands']), 1)
        self.assertEqual(sum(c[1] == 'push' for c in state['commands']), 1)
        self.assertTrue(result['complete'])
        self.assertFalse(result['goal_complete'])
        self.assertFalse(any('-A' in c for c in state['commands']))

    def test_existing_index_and_failed_push_are_never_reset_or_retried(self):
        root, launch, _ = fixture(self.repo, 'R98')
        state, output, command = self.publication_mocks(root, launch, staged_before=True)
        with patch.object(goal, 'REPO', self.repo), patch.object(goal, 'LAUNCH', launch), \
                patch.object(goal, 'git_output', side_effect=output), patch.object(goal.subprocess, 'run', side_effect=command):
            with self.assertRaisesRegex(RuntimeError, 'staged changes'):
                goal.publish('R98', root, launch)
        self.assertEqual(state['commands'], [])
        state, output, command = self.publication_mocks(root, launch, fail_push=True)
        with patch.object(goal, 'REPO', self.repo), patch.object(goal, 'LAUNCH', launch), \
                patch.object(goal, 'git_output', side_effect=output), patch.object(goal.subprocess, 'run', side_effect=command):
            with self.assertRaises(subprocess.CalledProcessError):
                goal.publish('R98', root, launch)
            receipt = (root / 'github_publication.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                goal.publish('R98', root, launch)
            self.assertEqual((root / 'github_publication.json').read_bytes(), receipt)
        self.assertEqual(sum(c[1] == 'commit' for c in state['commands']), 1)
        self.assertEqual(sum(c[1] == 'push' for c in state['commands']), 1)
        self.assertEqual(goal.optional(root / 'github_publication.json')['commit'], 'committed')

    def test_compare_rc0_without_outputs_fails_once_without_touching_gpu_receipts(self):
        root, launch, _ = fixture(self.repo, 'R99')
        worker = launch / 'R99_A/worker.json'
        original = worker.read_bytes()
        with patch.object(goal, 'comparison_available', return_value=True), \
                patch.object(goal.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as command:
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, launch)
            receipt = (root / 'goal_compare_receipt.json').read_bytes()
            with self.assertRaises(RuntimeError):
                goal.compare_once(root, launch)
            self.assertEqual((root / 'goal_compare_receipt.json').read_bytes(), receipt)
        self.assertEqual(command.call_count, 1)
        self.assertEqual(worker.read_bytes(), original)
        self.assertFalse(goal.status('R99', root, launch, persist=False)['complete'])


if __name__ == '__main__':
    unittest.main()
