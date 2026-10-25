"""Small CPU contracts; all synthetic reports and Git remotes live in temp dirs."""

import copy
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import r112_finish as finish


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def prediction(variant='A', n=4500):
    costs = np.tile(np.array([[1., 2., 3.]], dtype=np.float64), (n, 1))
    pred = np.zeros(n, dtype=np.int64)
    pred[:6] = [1, 0, 1, 2, 0, 2] if variant == 'A' else [0, 1, 2, 1, 0, 2]
    return dict(costs=costs, winner=np.zeros(n, dtype=np.int64), pred=pred,
        pool=['s0', 's1', 's2'], pool_ids=np.arange(3), indices=np.arange(n),
        source_indices=np.arange(n), release_source_indices=np.arange(n) + 100)


def packed(data):
    result = dict(data)
    logits = np.zeros_like(data['costs'], dtype=np.float32)
    logits[np.arange(len(data['pred'])), data['pred']] = 1.
    result['logits'] = logits
    result['pool'] = np.asarray(data['pool'])
    return result


def model_fixture(run, main_path, variant, experiment='R112', arm=None):
    run = Path(run)
    run.mkdir(parents=True)
    arm = arm or run.name
    (run / 'best.pt').write_bytes(b'SYNTHETIC_NOT_MODEL_WEIGHTS_' + arm.encode())
    checkpoint = finish.digest(run / 'best.pt')
    base = dict(selection_split='train', sbs_solver='s1')
    main = finish.read(main_path)
    config = dict(experiment=experiment, arm=arm, seed=2, data_root=str(main_path.parent / 'synthetic_release'),
        preparation_sha256='synthetic_preparation', rows_per_problem=dict(train=81000, val=4500, test=4500),
        candidate_pools={p: ['s0', 's1', 's2'] for p in finish.PROBLEMS},
        candidate_pool_ids={p: [0, 1, 2] for p in finish.PROBLEMS},
        baselines={p: base for p in finish.PROBLEMS}, winner_rule='native/FP64 ind',
        global_batch_size=512, logical_global_batch=512, task_order_sha256='synthetic_order',
        max_successful_updates=57600, lr={'peak': 1e-4}, optimizer='AdamW', weight_decay=1e-4,
        loss={'ce': .35}, dropout=.1, microbatch_caps={p: 384 for p in finish.PROBLEMS},
        eval_batch_size=512, checkpoint_rule='synthetic VAL-only locked rule', inference_rule='stable argmax',
        model_spec={'synthetic': True}, source_sha256={'synthetic.py': 'source'},
        main_protocol_lock=dict(record=main, sha256=finish.digest(main_path)),
        trainable_parameter_count=3, initial_trainable_sha256='synthetic_fresh_parameters')
    state_hash = finish.json_hash(config)
    step = 2880 if variant == 'A' else 5760
    lock = dict(experiment=experiment, arm=arm, checkpoint=str(run / 'best.pt'), checkpoint_sha256=checkpoint,
        state_hash=state_hash, selected_update=step, epoch=1, selected_without_test=True,
        per_problem_assembly=False, rule=config['checkpoint_rule'], inference_rule=config['inference_rule'])
    put(run / 'protocol.json', config)
    put(run / 'final_test_lock.json', lock)
    put(run / 'training_complete.json', {'successful_updates': 57600})
    put(run / 'initialization.json', dict(trainable_sha256='synthetic_fresh_parameters',
        parameter_layout_sha256='synthetic_shapes', trainable_parameter_count=3, fresh_reference_sha256='source'))
    data = prediction(variant)
    summaries = {}
    for split in ('VAL', 'TEST'):
        directory = run / 'evaluation' / split.lower() / 'selected_best'
        directory.mkdir(parents=True)
        identity = dict(experiment=experiment, arm=arm, split=split.lower(), state_hash=state_hash,
            successful_updates=step, model_state_sha256='synthetic_state_' + arm,
            checkpoint_sha256=None if split == 'VAL' else checkpoint)
        put(directory / 'identity.json', identity)
        arrays = {p + '__' + k: v for p in finish.PROBLEMS for k, v in packed(data).items()}
        np.savez(directory / 'rank000.npz', **arrays)
        metrics = {}
        for problem in finish.PROBLEMS:
            row = finish.metric_row(data, base, model=arm, split=split, problem=problem, checkpoint=checkpoint)
            metrics[problem] = dict(n=4500, top1=row['top1_pct'] / 100,
                **{k: row[k] for k in finish.METRICS if k != 'top1_pct'})
        summaries[split] = dict(identity=identity, prediction_archive=str(directory / 'rank000.npz'), per_problem=metrics)
    test = dict(complete=True, result=summaries['TEST'], **{k: lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')})
    final = dict(complete=True, val=summaries['VAL'], test=summaries['TEST'],
        **{k: lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')})
    put(run / 'test_results.json', test)
    put(run / 'final_metrics.json', final)
    put(run / 'report_complete.json', dict(complete=True, experiment=experiment, arm=arm,
        **{k: lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')}))
    stats = run / 'full_model_statistics'
    put(stats / 'protocol.json', dict(complete=True, checkpoint_lock=lock))
    for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf', 'training_and_method_diagnostics.pdf'):
        (stats / name).write_bytes(b'%PDF-1.4\nSYNTHETIC_TEST_ONLY\n%%EOF\n')
    for dataset, problem in (('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')):
        directory = run / 'lib_evaluation/predictions'
        directory.mkdir(parents=True, exist_ok=True)
        values = packed(prediction(variant, 6))
        np.savez(directory / (dataset + '.npz'), **values,
                 checkpoint_sha256=np.array(checkpoint), checkpoint_update=np.array(step))
        put(run / 'lib_evaluation' / (dataset + '_results.json'), dict(
            protocol=dict(experiment=experiment, arm=arm, dataset=dataset,
                model_state_sha256='synthetic_state_' + arm,
                **{k: lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')}),
            train_sbs_reference={'n': 6}))
    return config


def complete_fixture(root):
    put(root / 'protocol_lock.json', dict(experiment='R112', locked=True, logical_global_batch=512))
    dispatch = root / 'preparation/dispatch'
    put(dispatch / 'launch.json', dict(state='complete', exit_status=0,
        stages=[dict(name='allocation_pipeline', state='complete', exit_status=0)]))
    for arm in finish.ARMS:
        put(dispatch / (arm + '.json'), dict(state='complete', exit_status=0,
            stages=[dict(name=s, state='complete', exit_status=0) for s in finish.STAGES]))
        model_fixture(root / arm, root / 'protocol_lock.json', 'A' if arm == finish.ARMS[0] else 'B')


def command(directory, *args):
    result = subprocess.run(['git', '-C', str(directory), *args], capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


def git_fixture(directory):
    remote, repo = directory / 'remote.git', directory / 'repo'
    remote.mkdir()
    repo.mkdir()
    command(remote, 'init', '--bare')
    command(repo, 'init', '-b', 'main')
    command(repo, 'config', 'user.name', 'R112 synthetic test')
    command(repo, 'config', 'user.email', 'fixture@example.invalid')
    (repo / 'base.txt').write_text('synthetic base\n')
    command(repo, 'add', '--', 'base.txt')
    command(repo, 'commit', '-m', 'synthetic base')
    command(repo, 'remote', 'add', 'origin', str(remote))
    command(repo, 'push', '-u', 'origin', 'main')
    v4 = repo / '0selection2/code/V4'
    v4.mkdir(parents=True)
    for name in finish.CODE_NAMES:
        (v4 / name).write_text('# synthetic publication fixture\n')
    research = repo / '0selection2/docs/research'
    for group, names in finish.RESEARCH_FILES.items():
        for name in names:
            path = research / group / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic research delivery\n')
    layout = research / finish.REPORT_LAYOUT
    layout.parent.mkdir(parents=True)
    layout.write_text('# synthetic shared layout dependency\n')
    root = v4 / 'runs/R112_region_interfaces'
    root.mkdir(parents=True)
    for name in (*finish.OUTPUTS, 'finish_complete.json', 'protocol_lock.json'):
        (root / name).write_text('synthetic publication delivery\n')
    paths = [v4 / n for n in ('r112_model.py', 'r112_experiment.py', 'r112_reporting.py', 'r112_launch.py', 'run_v4_r112.sh')]
    paths.append(root / 'protocol_lock.json')
    put(root / 'preparation/dispatch/claim.json', dict(source_sha256={str(p): finish.digest(p) for p in paths}))
    return repo, root


class FinishTests(unittest.TestCase):
    def test_event_partition_metrics_and_equal_problem_macro(self):
        old, new = prediction('A', 6), prediction('B', 6)
        row, events = finish.paired(old, new, 'B vs A', 'TEST', 'TSP')
        self.assertEqual([row[k] for k in ('corrected', 'harmed', 'both_wrong_cheaper', 'both_wrong_dearer')], [1, 1, 1, 1])
        self.assertEqual(len(events), row['changed'])
        self.assertEqual(row['net_correct'], 0)
        self.assertAlmostEqual(row['mean_cost_delta'], 0.)
        self.assertAlmostEqual(row['W_pct'], 100 / 3)
        self.assertLess(row['G_pct'], 0.)
        self.assertAlmostEqual(row['gross_harm_cost'] - row['gross_gain_cost'], row['mean_cost_delta'])
        rows = [dict(row, problem=p, n=6 if p == 'TSP' else 60) for p in finish.PROBLEMS]
        overall = finish.macro(rows, finish.DELTAS, finish.COUNTS)
        self.assertAlmostEqual(overall['G_pct'], row['G_pct'])
        self.assertEqual(overall['corrected'], 18)
        with self.assertRaises(ValueError):
            finish.macro(rows[:-1], finish.DELTAS)
        damaged = copy.deepcopy(new)
        damaged['release_source_indices'][0] += 1
        with self.assertRaises(AssertionError):
            finish.paired(old, damaged, 'bad', 'TEST', 'TSP')

    def test_complete_saved_only_delivery_and_whole_reference_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'R112_region_interfaces'
            complete_fixture(root)
            ref = Path(directory) / 'R94B'
            model_fixture(ref, root / 'protocol_lock.json', 'A', 'R94', 'B')
            bad = Path(directory) / 'R89B'
            model_fixture(bad, root / 'protocol_lock.json', 'A', 'R89', 'B_muon_hybrid')
            path = bad / 'evaluation/test/selected_best/rank000.npz'
            with np.load(path, allow_pickle=False) as saved:
                arrays = {k: saved[k] for k in saved.files if k != 'ATSP__costs'}
            np.savez(path, **arrays)
            receipt = finish.analyze(root, {'R94B': ref, 'R89B': bad})
            self.assertEqual(receipt['analysis_rc'], 0)
            self.assertEqual(receipt['comparison_rows'], 76)
            self.assertEqual([r['status'] for r in receipt['historical']], ['complete', 'missing_or_incompatible'])
            self.assertTrue((root / 'RESULTS.pdf').read_bytes().startswith(b'%PDF-'))
            self.assertEqual(finish.verify_delivery(root)['outputs'], receipt['outputs'])
            with (root / 'comparison.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertFalse(any('R89B' in r['comparison'] for r in rows))
            self.assertEqual({r['problem'] for r in rows if r['comparison'] == 'B_interfaces vs A_collapsed'}, set(finish.PROBLEMS) | {'ALL'})
            self.assertNotIn('torch', sys.modules)
            (root / 'comparison.csv').write_text('tampered')
            with self.assertRaises(ValueError):
                finish.verify_delivery(root)

    def test_wait_stops_on_failure_or_timeout_without_analysis_or_git(self):
        with patch.object(finish, 'analyze', side_effect=AssertionError), patch.object(finish, 'git', side_effect=AssertionError):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                put(finish.launch_receipt(root), {'state': 'failed', 'error': 'synthetic fatal'})
                with self.assertRaisesRegex(RuntimeError, 'no restart'):
                    finish.wait(root)
            with patch.object(finish, 'completed', side_effect=finish.NotReady('pending')), \
                    patch.object(finish.time, 'monotonic', side_effect=[0., 3601.]):
                with self.assertRaises(TimeoutError):
                    finish.wait(Path('/synthetic'), timeout_hours=1.)

    def test_publication_refuses_other_index_and_commits_pushes_only_once(self):
        for dirty in (True, False):
            with self.subTest(dirty=dirty), tempfile.TemporaryDirectory() as directory:
                repo, root = git_fixture(Path(directory))
                (repo / 'unrelated.txt').write_text('must not publish\n')
                if dirty:
                    command(repo, 'add', '--', 'unrelated.txt')
                before = (repo / '.git/index').read_bytes()
                if not dirty:
                    # Another complete version may advance HEAD before R112 obtains the mutex.
                    (repo / 'r111_completed.txt').write_text('synthetic completed R111\n')
                    command(repo, 'add', '--', 'r111_completed.txt')
                    command(repo, 'commit', '-m', 'synthetic completed R111')
                    command(repo, 'push', 'origin', 'main')
                current_head = command(repo, 'rev-parse', 'HEAD')
                original_preflight = finish.publication_preflight

                def locked_preflight(root):
                    with (repo / '.git/codex-publication.lock').open('a') as mutex:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return original_preflight(root)

                with patch.object(finish, 'GIT_ROOT', repo), patch.object(finish, 'verify_delivery', return_value={}):
                    with patch.object(finish, 'publication_preflight', side_effect=locked_preflight):
                        if dirty:
                            with self.assertRaisesRegex(ValueError, 'staged changes'):
                                finish.publish(root)
                            self.assertEqual(before, (repo / '.git/index').read_bytes())
                            self.assertFalse((root / 'publish_receipt.json').exists())
                        else:
                            receipt = finish.publish(root)
                            self.assertEqual(receipt['state'], 'complete')
                            self.assertEqual(receipt['parent'], current_head)
                            self.assertEqual(receipt['global_mutex'], str(repo / '.git/codex-publication.lock'))
                            self.assertEqual((receipt['commit_attempts'], receipt['push_attempts']), (1, 1))
                            self.assertEqual(finish.publish(root)['commit'], receipt['commit'])
                            self.assertEqual(command(repo, 'rev-list', '--count', 'HEAD'), '3')
                            self.assertNotIn('unrelated.txt', command(repo, 'ls-tree', '-r', '--name-only', 'HEAD').splitlines())
                            changed = command(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').splitlines()
                            self.assertNotIn('r111_completed.txt', changed)
                            self.assertFalse(any('/R113' in p or '/R114' in p or '/R77' in p for p in changed))
                            self.assertEqual(command(repo, 'ls-remote', 'origin', 'refs/heads/main').split()[0], receipt['commit'])

    def test_unpublished_dependency_and_incomplete_delivery_block_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            path = repo / '0selection2/code/V4/r112_model.py'
            path.parent.mkdir(parents=True)
            path.write_text('from .test_r109_model import input_batch\n')
            with patch.object(finish, 'GIT_ROOT', repo):
                with self.assertRaisesRegex(ValueError, 'R109/R111 hard dependency'):
                    finish.dependency_files([path])
            with patch.object(finish, 'git', side_effect=AssertionError('No Git before successful delivery')):
                with self.assertRaises(finish.NotReady):
                    finish.publication_preflight(repo)


if __name__ == '__main__':
    unittest.main()
