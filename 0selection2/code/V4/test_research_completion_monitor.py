"""CPU-only fixtures: all subprocesses, model calls, signals and GPU work mocked."""

import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import research_completion_monitor as monitor


def text_file(path, value='fixture\n'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding='utf-8')


def pdf(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'%PDF-1.4\nfixture\n%%EOF\n')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def version_root(repo, version):
    return Path(repo) / 'code/V4/runs' / monitor.ROOTS[version]


def handoff_fixture(repo, state='running', pilot='running'):
    root = Path(repo) / monitor.RUN_REL
    source = monitor.REPO / monitor.RUN_REL
    for name in ('monitor_plan.json', 'response.schema.json'):
        text_file(root / name, (source / name).read_text())
    for path in (source / 'prompts').glob('*.md'):
        text_file(root / 'prompts' / path.name, path.read_text())
    migration = version_root(repo, 'R113') / 'preparation/migration_3090_portable_inode_20261026'
    queue_path = migration / 'controller.json'
    progress = version_root(repo, 'R114') / 'preparation/grouped_execution/queue_v1/scheduler_progress.json'
    value = dict(ready=True, owner='Mencius', queue_module=monitor.QUEUE_MODULE,
                 host='gpu03', allocation_job=1465, gpu_uuids=list(monitor.UUIDS),
                 queue_state_path=str(queue_path), owner_ready_path=str(migration / 'ready.json'),
                 state_field='state', failure_states=['failed', 'RESOURCEERROR', 'resource_error'],
                 r114_progress_path=str(progress), r114_terminal_authoritative=False,
                 recovery_argv=[], closeout_argv=[['python', '-B', str(migration / 'finish_adapter.py'), 'analyze']],
                 finish_adapter_path=str(migration / 'finish_adapter.py'),
                 resumes={a: dict(kind='fresh') if a == 'A_scratch' else dict(
                     kind='resource_checkpoint_resume', update=20160, optimizer_retained=True, rng_retained=True)
                     for a in monitor.ARMS['R113']})
    monitor.save(root / 'queue_handoff.json', value)
    monitor.save(migration / 'ready.json', dict(main_approved=True))
    jobs = {a: dict(state='complete' if state in ('complete', 'blocked') else 'waiting_resources',
                    exit_status=0 if state in ('complete', 'blocked') else None) for a in monitor.ARMS['R113']}
    jobs['R114_labels'] = dict(state=pilot, exit_status=2 if pilot == 'blocked' else 0)
    monitor.save(queue_path, dict(state=state, jobs=jobs))
    return value


def models_fixture(repo, version):
    root, locks = version_root(repo, version), {}
    pools = {p: ['solver_a', 'solver_b'] for p in monitor.PROBLEMS}
    for arm in monitor.ARMS[version]:
        run = root / arm
        checkpoint = monitor.json_hash([version, arm, 'checkpoint'])
        final_lock = dict(experiment=version, arm=arm, checkpoint_sha256=checkpoint,
                          state_hash=monitor.json_hash([version, arm, 'protocol']), selected_update=20160,
                          selection_split='val', selected_without_test=True, per_problem_assembly=False,
                          rule='strict maximum full18 VAL macro_top1; first tie', inference_rule='frozen maximin')
        locks[arm] = final_lock
        monitor.save(run / 'protocol.json', dict(max_successful_updates=57600,
                     candidate_pools=pools, inference_rule='frozen maximin'))
        monitor.save(run / 'final_test_lock.json', final_lock)
        monitor.save(run / 'training_complete.json', dict(successful_updates=57600))
        monitor.save(run / 'test_results.json', dict(complete=True, checkpoint_sha256=checkpoint))
        monitor.save(run / 'lib_stage_complete.json', dict(complete=True, checkpoint_sha256=checkpoint))
        name = 'postprocess_status.json' if version == 'R109' else 'report_complete.json'
        monitor.save(run / name, dict(complete=True))
        monitor.save(run / 'full_model_statistics/protocol.json', dict(complete=True, checkpoint_lock=final_lock))
        for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf'):
            pdf(run / 'full_model_statistics' / name)
        for name in ('final_metrics.json', 'inference_timing.json'):
            monitor.save(run / name, dict(complete=True, checkpoint_sha256=checkpoint))
        for name in ('RESULTS.md', 'metrics_by_problem.csv', 'selection_distribution.csv', 'nss_comparison.csv'):
            text_file(run / 'full_model_statistics' / name)
    publication = dict(state='complete', exit_status=0, commit='c' * 40)
    if version == 'R109':
        monitor.save(root / 'goal_progress.json', dict(complete=True, comparison_complete=True, errors=[]))
        monitor.save(root / 'github_publication.json', publication)
    else:
        monitor.save(root / 'finish_complete.json', dict(complete=True, analysis_rc=0, checkpoints=locks))
        monitor.save(root / 'publish_receipt.json', publication)
    if version == 'R112':
        monitor.save(root / 'preparation/recovery_20261026/recovery_complete.json',
                     dict(experiment='R112', kind='cpu_report_recovery', state='complete', exit_status=0,
                          gpu_compute=False, model_inference=False, checkpoint_reselection=False))
    return locks


def full_fixture(repo, partial=True):
    handoff = handoff_fixture(repo, state='blocked' if partial else 'complete', pilot='blocked' if partial else 'complete')
    for version in monitor.ARMS:
        models_fixture(repo, version)
    root = version_root(repo, 'R114')
    analysis = root / 'preparation/grouped_execution/queue_v1/scheduler_runs/fixture/analysis'
    progress = dict(status='blocked' if partial else 'complete', execution=True,
                    analysis=dict(status='complete', output=str(analysis)), stopped_problems=['VRPBTW'] if partial else [])
    monitor.save(handoff['r114_progress_path'], progress)
    monitor.save(analysis / 'analysis_complete.json', dict(complete=True))
    finish_dir = root / 'preparation/finish/fixture'
    kind = 'partial_negative_closeout' if partial else 'complete_labels'
    monitor.save(finish_dir / 'finish_complete.json', dict(closeout_ready=True, closeout_kind=kind,
                 labeling_complete=not partial, inputs={handoff['r114_progress_path']: monitor.digest(handoff['r114_progress_path'])}))
    monitor.save(finish_dir / 'publish_receipt.json', dict(state='complete', commit='b' * 40))
    monitor.save(finish_dir / 'closeout.json', dict(closeout_kind=kind, full18_model_claim=False,
                 execution_phase='acquisition_only', reasons=['unsupported native contract'] if partial else []))
    text_file(finish_dir / 'RESULTS.md')
    text_file(finish_dir / 'counts.csv')
    return handoff


def pack_fixture(instance, observation, snapshot):
    folder = instance.root / 'evidence'
    text_file(folder / 'evidence_pack.md', '# Actual fixture negative and partial evidence\n')
    checkpoints, metrics = [], []
    for version in monitor.ARMS:
        for arm, final_lock in observation['tasks'][version]['locks'].items():
            checkpoints.append(dict(experiment=version, arm=arm, selection_split='VAL',
                                   **{k: final_lock[k] for k in ('checkpoint_sha256', 'state_hash', 'selected_update', 'inference_rule')}))
            config = monitor.read(version_root(instance.repo, version) / arm / 'protocol.json')
            datasets = [(s, p) for s in ('VAL', 'TEST') for p in (*monitor.PROBLEMS, 'ALL')]
            datasets += [('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')]
            for split, problem in datasets:
                metrics.append(dict(experiment=version, arm=arm, split=split, problem=problem,
                    n=(81000 if problem == 'ALL' else 4500) if split in ('VAL', 'TEST') else 100,
                    checkpoint_sha256=final_lock['checkpoint_sha256'], top1_pct=70., mean_cost=11.,
                    sbs_cost=12., oracle_cost=10., vs_sbs_pct=-1., vs_oracle_pct=2., actual_regret_pct=.4,
                    data_version=monitor.DATA_VERSION if split in ('VAL', 'TEST') else 'native_' + split,
                    candidate_pool_sha256=monitor.json_hash(config['candidate_pools']), winner_rule='native FP64 ind',
                    sbs_fit_split='TRAIN', all_aggregation='equal_problem' if problem == 'ALL' else ''))
    write_csv(folder / 'checkpoints.csv', checkpoints)
    write_csv(folder / 'metrics.csv', metrics)
    value = dict(complete=True, model=monitor.SOL, reasoning_effort='max', saved_results_only=True,
                 unique_val_checkpoints=True, same_version_metrics=True, test_development_exposed=True,
                 verified_inputs=snapshot, scientific_completion_gate_sha256=monitor.json_hash(snapshot),
                 outputs={n: monitor.digest(folder / n) for n in ('evidence_pack.md', 'checkpoints.csv', 'metrics.csv')},
                 r114_closeout_kind=observation['tasks']['R114']['closeout_kind'],
                 negative_partial_facts=['acquisition only; not a trained function learner'],
                 implemented_hypothesis_limits={v: 'bounded approximation' for v in ('R112', 'R113', 'R114')})
    monitor.save(folder / 'verification.json', value)
    return value


def research_fixture(repo):
    root = Path(repo) / monitor.RESEARCH_REL
    text_file(root / 'REPORT.md', '# Full research fixture\n')
    write_csv(root / 'sources.csv', [dict(source_id=str(i), title='primary paper', url='https://example.org/paper/' + str(i),
              primary='true', browsed='true', sections='methods and limitations') for i in range(8)])
    write_csv(root / 'directions.csv', [dict(id=str(i), name='Broad paradigm', hypothesis='testable', scope='all18',
              decisive_control='matched finite control', falsifier='no net gain', finite_plan='finite experiment',
              gpu_hours_2x3090='12', label_cost='0', stop_rule='one completed comparison', counterevidence='actual negative') for i in range(3)])


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='research-monitor-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / 'repo'
        self.repo.mkdir()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.popen = self.stack.enter_context(patch.object(monitor.subprocess, 'Popen'))
        self.run_command = self.stack.enter_context(patch.object(monitor.subprocess, 'run'))
        self.run_command.return_value = Mock(returncode=1)
        self.stack.enter_context(patch.object(monitor, 'processes', return_value=[]))
        self.stack.enter_context(patch.object(monitor, 'process_identity', return_value=None))
        self.kill = self.stack.enter_context(patch.object(monitor.os, 'kill'))
        self.killpg = self.stack.enter_context(patch.object(monitor.os, 'killpg'))
        self.popen.return_value.pid = 12345
        self.popen.return_value.poll.return_value = None
        self.output = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.output))

    def instance(self):
        return monitor.Monitor(self.repo)

    def complete_action(self, instance, key, **extra):
        action = instance.state['actions'][key]
        monitor.save(Path(action['attempt']) / 'exit.json', dict(state='complete', exit_status=0, **extra))

    def test_status_and_unapproved_start_never_launch(self):
        with patch.object(monitor, 'REPO', self.repo):
            self.assertEqual(monitor.main(['status']), 0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                monitor.main(['start'])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                monitor.main(['run'])
        self.popen.assert_not_called()
        self.run_command.assert_not_called()
        self.assertFalse((self.repo / monitor.RUN_REL).exists())

    def test_pending_handoff_quiet_and_idempotent(self):
        instance = self.instance()
        for _ in range(12):
            self.assertFalse(instance.tick())
        self.assertEqual(len((instance.root / 'events.jsonl').read_text().splitlines()), 1)
        self.assertEqual(instance.state['actions'], {})
        self.popen.assert_not_called()
        self.run_command.assert_not_called()

    def test_old_failures_and_native_goal_pause_are_ignored(self):
        handoff_fixture(self.repo)
        models_fixture(self.repo, 'R112')
        old = version_root(self.repo, 'R113')
        monitor.save(old / 'preparation/wave_dispatch/controller.json', dict(state='failed', error='User resource interrupt'))
        for arm in ('B_snapshot', 'C_transition'):
            monitor.save(old / arm / 'error.json', dict(error='KeyboardInterrupt'))
            monitor.save(old / arm / 'status.json', dict(phase='training', gpu_type='4090'))
        monitor.save(version_root(self.repo, 'R112') / 'preparation/dispatch/launch.json', dict(state='failed', error='PDF font missing'))
        monitor.save(self.repo / 'old_native_goal.json', dict(status='paused'))
        instance = self.instance()
        instance.tick()
        self.assertEqual(instance.state['observation']['tasks']['R112']['state'], 'complete')
        self.assertEqual(instance.state['observation']['tasks']['R113']['state'], 'waiting_queue')
        self.assertFalse(instance.state['observation']['complete'])
        self.assertEqual(instance.state['actions'], {})
        self.assertNotIn('KeyboardInterrupt', self.output.getvalue())
        self.popen.assert_not_called()

    def test_actual_pending_main_ready_is_not_a_failed_queue(self):
        value = handoff_fixture(self.repo, state='failed')
        Path(value['owner_ready_path']).unlink()
        source = monitor.Sources(self.repo, self.repo / monitor.RUN_REL / 'queue_handoff.json')
        self.assertEqual(source.queue()['state'], 'waiting_main')
        self.instance().tick()
        self.popen.assert_not_called()

    def test_resource_scope_rejects_extra_or_4090_allocation(self):
        value = handoff_fixture(self.repo)
        for key, bad in [('allocation_job', 1475), ('gpu_uuids', list(monitor.UUIDS) + ['GPU-4090']), ('host', 'gpu04')]:
            modified = dict(value, **{key: bad})
            monitor.save(self.repo / monitor.RUN_REL / 'queue_handoff.json', modified)
            result = self.instance().sources.observe()
            self.assertEqual(result['queue']['state'], 'needs_action')
            self.assertFalse(result['complete'])
        self.popen.assert_not_called()

    def test_partial_pilot_is_legitimate_complete_pipeline(self):
        full_fixture(self.repo, partial=True)
        observation = self.instance().sources.observe()
        self.assertTrue(observation['queue']['complete'], observation)
        self.assertTrue(observation['complete'], observation)
        self.assertEqual(observation['tasks']['R114']['closeout_kind'], 'partial_negative_closeout')
        self.assertFalse(observation['tasks']['R114']['labeling_complete'])

    def test_full_acquisition_also_terminal_but_not_function_learning_claim(self):
        full_fixture(self.repo, partial=False)
        observation = self.instance().sources.observe()
        self.assertTrue(observation['complete'])
        self.assertEqual(observation['tasks']['R114']['closeout_kind'], 'complete_labels')

    def test_r114_failed_native_contract_not_automatic_model_repair(self):
        value = full_fixture(self.repo)
        queue = monitor.read(value['queue_state_path'])
        queue['state'] = 'failed'
        queue['jobs']['R114_labels']['state'] = 'failed'
        queue['jobs']['R114_labels']['reason'] = 'Unsupported native contract'
        monitor.save(value['queue_state_path'], queue)
        source = self.instance().sources
        self.assertTrue(source.queue()['complete'])
        self.assertFalse(source.queue()['failed'])

    def test_complete_queue_word_alone_cannot_open_scientific_gate(self):
        value = full_fixture(self.repo)
        queue = monitor.read(value['queue_state_path'])
        queue['jobs']['B_snapshot']['state'] = 'running'
        monitor.save(value['queue_state_path'], queue)
        self.assertFalse(self.instance().sources.observe()['complete'])

    def test_pilot_shared_publisher_receipt_must_match_exact_delivery(self):
        full_fixture(self.repo)
        root = version_root(self.repo, 'R114')
        delivery = root / 'preparation/finish/fixture'
        receipt = monitor.read(delivery / 'publish_receipt.json')
        (delivery / 'publish_receipt.json').unlink()
        receipt['delivery'] = str(delivery)
        monitor.save(root / 'preparation/finish/publish_receipt.json', receipt)
        source = self.instance().sources
        self.assertEqual(source.observe()['tasks']['R114']['state'], 'complete')
        receipt['delivery'] += '_other'
        monitor.save(root / 'preparation/finish/publish_receipt.json', receipt)
        self.assertNotEqual(source.observe()['tasks']['R114']['state'], 'complete')

    def test_pilot_waits_for_explicit_interface_recovery_closeout(self):
        full_fixture(self.repo)
        path = self.repo / monitor.RUN_REL / 'queue_handoff.json'
        value = monitor.read(path)
        value['r114_required_finish_dir'] = str(version_root(self.repo, 'R114') /
                                               'preparation/finish/future_corrected_closeout')
        monitor.save(path, value)
        self.assertEqual(self.instance().sources.observe()['tasks']['R114']['state'], 'waiting_owner')

    def test_r109_additive_publication_recovery_preserves_failed_watcher(self):
        handoff_fixture(self.repo)
        models_fixture(self.repo, 'R109')
        root = version_root(self.repo, 'R109')
        (root / 'github_publication.json').unlink()
        failed = dict(state='stopped_on_error', error='historical canonical-root failure')
        monitor.save(root / 'goal_watch_receipt.json', failed)
        folder = root / 'preparation/publication_recovery_20261026'
        receipt = dict(state='complete', exit_status=0, commit='d' * 40,
                       kind='explicit_additive_publication_recovery',
                       original_failure_preserved=True, canonical_main_remote_verified=True,
                       unrelated_changes_staged=False, remote_after='d' * 40)
        monitor.save(folder / 'publication_receipt.json', receipt)
        verification = dict(state='complete', actual_commit='d' * 40, actual_origin_main='d' * 40,
                            publication_receipt_sha256=monitor.digest(folder / 'publication_receipt.json'))
        monitor.save(folder / 'final_verification.json', verification)
        source = self.instance().sources
        self.assertEqual(source.observe()['tasks']['R109']['state'], 'complete')
        self.assertEqual(monitor.read(root / 'goal_watch_receipt.json'), failed)
        verification['actual_origin_main'] = 'e' * 40
        monitor.save(folder / 'final_verification.json', verification)
        self.assertNotEqual(source.observe()['tasks']['R109']['state'], 'complete')

    def test_new_failure_calls_sol_once_despite_changed_timestamps(self):
        value = handoff_fixture(self.repo)
        queue = monitor.read(value['queue_state_path'])
        queue.update(state='failed', error=dict(kind='RESOURCEERROR', stage='finalize', message='resource unavailable'))
        monitor.save(value['queue_state_path'], queue)
        instance = self.instance()
        instance.tick()
        self.assertEqual(self.popen.call_count, 1)
        key = next(iter(instance.state['actions']))
        self.assertTrue(key.startswith('repair:'))
        self.assertEqual(instance.state['actions'][key]['model'], monitor.SOL)
        for i in range(10):
            queue['updated_utc'] = str(i)
            monitor.save(value['queue_state_path'], queue)
            instance.tick()
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(len(instance.state['actions']), 1)

    def test_per_failure_and_total_recovery_bounds(self):
        handoff_fixture(self.repo)
        instance = self.instance()
        for i in range(monitor.MAX_RECOVERIES + 3):
            instance.dispatch('repair:' + str(i), 'execution', {})
        self.assertEqual(self.popen.call_count, monitor.MAX_RECOVERIES)
        instance.dispatch('repair:0', 'execution', {})
        self.assertEqual(self.popen.call_count, monitor.MAX_RECOVERIES)

    def test_active_external_owner_is_not_duplicated(self):
        full_fixture(self.repo)
        (version_root(self.repo, 'R113') / 'publish_receipt.json').unlink()
        with patch.object(monitor, 'path_running', return_value=True):
            self.instance().tick()
        self.popen.assert_not_called()

    def test_adopts_live_claim_across_controller_restart(self):
        handoff_fixture(self.repo)
        instance = self.instance()
        instance.dispatch('repair:test', 'execution', {})
        action = instance.state['actions']['repair:test']
        identity = dict(pid=12345, start_ticks='1', boot_id='boot', argv=['worker'])
        monitor.save(Path(action['attempt']) / 'worker.json', dict(identity=identity))
        with patch.object(monitor, 'alive', return_value=True):
            restarted = self.instance()
            restarted.tick()
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(restarted.state['actions']['repair:test']['state'], 'running')

    def test_launch_crash_gap_discovers_owned_orphan_no_duplicate(self):
        handoff_fixture(self.repo)
        instance = self.instance()
        instance.dispatch('repair:test', 'execution', {})
        attempt = instance.state['actions']['repair:test']['attempt']
        with patch.object(monitor, 'processes', return_value=[dict(argv=['python', '-m', monitor.MODULE, '_worker', '--attempt-dir', attempt])]):
            self.assertTrue(self.instance().reconcile())
        self.assertEqual(self.popen.call_count, 1)

    def test_dead_or_reused_pid_is_not_restarted(self):
        handoff_fixture(self.repo)
        instance = self.instance()
        instance.dispatch('repair:test', 'execution', {})
        for _ in range(5):
            self.instance().tick()
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.instance().state['actions']['repair:test']['state'], 'needs_action')
        identity = dict(pid=1, start_ticks='old', boot_id='boot', argv=['ours'])
        with patch.object(monitor, 'process_identity', return_value=dict(identity, start_ticks='new')):
            self.assertFalse(monitor.alive(identity))

    def test_pause_only_stops_owned_pid_never_external_owners(self):
        handoff_fixture(self.repo)
        instance = self.instance()
        instance.dispatch('repair:test', 'execution', {})
        attempt = Path(instance.state['actions']['repair:test']['attempt'])
        identity = dict(pid=12345, start_ticks='1', boot_id='boot', argv=['ours'])
        monitor.save(attempt / 'worker.json', dict(identity=identity))
        monitor.save(instance.root / 'control.json', dict(paused=True))
        with patch.object(monitor, 'alive', return_value=True), patch.object(monitor.os, 'getpgid', return_value=12345):
            instance.tick()
        self.killpg.assert_called_once_with(12345, monitor.signal.SIGTERM)
        self.kill.assert_not_called()
        self.assertEqual(instance.state['phase'], 'paused')
        self.assertEqual(self.popen.call_count, 1)

    def test_disabled_manifest_no_actions(self):
        full_fixture(self.repo)
        instance = self.instance()
        monitor.save(instance.root / 'monitor_plan.json', dict(enabled=False))
        instance.tick()
        self.assertEqual(instance.state['phase'], 'paused')
        self.popen.assert_not_called()

    def test_orphaned_model_has_controller_six_hour_deadline(self):
        handoff_fixture(self.repo)
        instance = self.instance()
        instance.dispatch('repair:test', 'execution', {})
        action = instance.state['actions']['repair:test']
        action['started_utc'] = '2000-01-01T00:00:00+00:00'
        identity = dict(pid=666, start_ticks='1', boot_id='boot', argv=['owned_codex'])
        monitor.save(Path(action['attempt']) / 'codex.json', dict(identity=identity))
        with patch.object(monitor, 'alive', return_value=True), patch.object(monitor.os, 'getpgid', return_value=666):
            self.assertFalse(instance.reconcile())
        self.killpg.assert_called_once_with(666, monitor.signal.SIGTERM)
        self.assertEqual(action['state'], 'needs_action')
        self.assertIn('Six-hour', action['reason'])
        self.assertEqual(self.popen.call_count, 1)

    def test_r109_font_repair_requires_all_four_compute_stages_rc0(self):
        handoff_fixture(self.repo)
        root = version_root(self.repo, 'R109')
        monitor.save(root / 'goal_progress.json', dict(phase='failed', errors=['PDF font missing']))
        for arm in monitor.ARMS['R109']:
            folder = root / 'preparation/dispatch' / ('R109_' + arm)
            monitor.save(folder / 'worker.json', dict(stages=[dict(name=n, state='complete', exit_status=0)
                         for n in ('train', 'TEST', 'timing', 'LIB')]))
            monitor.save(folder / 'controller.json', dict(state='failed', stages=[dict(name='PDF', state='failed', exit_status=1)]))
        instance = self.instance()
        instance.tick()
        key = next(iter(instance.state['actions']))
        self.assertTrue(key.startswith('repair:R109_pdf:'))
        request = monitor.read(Path(instance.state['actions'][key]['attempt']) / 'request.json')
        self.assertEqual(request['context']['mode'], 'report_only_recovery')
        stages_path = root / 'preparation/dispatch/R109_A_tied_late/worker.json'
        value = monitor.read(stages_path)
        value['stages'][1]['exit_status'] = 1
        monitor.save(stages_path, value)
        self.assertEqual(instance.sources.observe()['tasks']['R109']['state'], 'needs_action')

    def test_scientific_gate_dispatches_sol_first_not_astra(self):
        full_fixture(self.repo)
        instance = self.instance()
        instance.tick()
        self.assertEqual(list(instance.state['actions']), ['verify-results'])
        self.assertEqual(instance.state['actions']['verify-results']['model'], monitor.SOL)
        self.assertTrue((instance.root / 'evidence/source_snapshot.json').is_file())
        self.assertEqual(self.popen.call_count, 1)

    def test_pack_rejects_test_selection_stitching_missing_or_mixed_metrics(self):
        full_fixture(self.repo)
        instance = self.instance()
        observation = instance.sources.observe()
        snapshot = instance.sources.snapshot(observation)
        pack_fixture(instance, observation, snapshot)
        monitor.verify_pack(instance.root / 'evidence', snapshot, observation, self.repo)
        folder = instance.root / 'evidence'
        rows = monitor.csv_rows(folder / 'metrics.csv')
        for column, bad in [('checkpoint_sha256', 'd' * 64), ('candidate_pool_sha256', 'e' * 64),
                            ('data_version', 'old10k'), ('sbs_fit_split', 'TEST'), ('top1_pct', 'nan')]:
            modified = [dict(r) for r in rows]
            modified[0][column] = bad
            write_csv(folder / 'metrics.csv', modified)
            attestation = monitor.read(folder / 'verification.json')
            attestation['outputs']['metrics.csv'] = monitor.digest(folder / 'metrics.csv')
            monitor.save(folder / 'verification.json', attestation)
            with self.assertRaises((monitor.NeedAction, ValueError)):
                monitor.verify_pack(folder, snapshot, observation, self.repo)
        write_csv(folder / 'metrics.csv', rows[:-1])
        attestation['outputs']['metrics.csv'] = monitor.digest(folder / 'metrics.csv')
        monitor.save(folder / 'verification.json', attestation)
        with self.assertRaises(monitor.NeedAction):
            monitor.verify_pack(folder, snapshot, observation, self.repo)

    def test_pack_rejects_wrong_equal_problem_all_and_changed_source(self):
        full_fixture(self.repo)
        instance = self.instance()
        observation = instance.sources.observe()
        snapshot = instance.sources.snapshot(observation)
        pack_fixture(instance, observation, snapshot)
        folder = instance.root / 'evidence'
        rows = monitor.csv_rows(folder / 'metrics.csv')
        next(r for r in rows if r['problem'] == 'ALL')['mean_cost'] = 1000
        write_csv(folder / 'metrics.csv', rows)
        attestation = monitor.read(folder / 'verification.json')
        attestation['outputs']['metrics.csv'] = monitor.digest(folder / 'metrics.csv')
        monitor.save(folder / 'verification.json', attestation)
        with self.assertRaises(monitor.NeedAction):
            monitor.verify_pack(folder, snapshot, observation, self.repo)
        pack_fixture(instance, observation, snapshot)
        path = Path(next(iter(snapshot)))
        path.write_text(path.read_text() + '\n')
        with self.assertRaises(monitor.NeedAction):
            monitor.verify_pack(folder, snapshot, observation, self.repo)

    def test_full_mock_pipeline_sol_astra_sol_exactly_once(self):
        full_fixture(self.repo)
        instance = self.instance()
        instance.tick()
        snapshot = monitor.read(instance.root / 'evidence/source_snapshot.json')
        pack_fixture(instance, instance.state['observation'], snapshot)
        self.complete_action(instance, 'verify-results')
        instance.tick()
        self.assertEqual(instance.state['actions']['deep-research']['model'], monitor.ASTRA)
        research_fixture(self.repo)
        self.complete_action(instance, 'deep-research')
        instance.tick()
        self.assertEqual(instance.state['actions']['publish-research']['model'], monitor.SOL)
        root = self.repo / monitor.RESEARCH_REL
        pdf(root / 'REPORT.pdf')
        paths = ['REPORT.md', 'REPORT.pdf', 'sources.csv', 'directions.csv']
        monitor.save(root / 'publication_receipt.json', dict(state='complete', commit='a' * 40,
                     model=monitor.SOL, reasoning_effort='max', global_mutex=str(self.repo.parent / '.git/codex-publication.lock'),
                     published_hashes={str(root / p): (monitor.digest(root / p) if p.endswith('.pdf')
                         else monitor.publication_text_hash(root / p)) for p in paths}))
        self.complete_action(instance, 'publish-research')
        self.assertTrue(instance.tick())
        self.assertEqual(instance.state['phase'], 'complete')
        self.assertEqual(self.popen.call_count, 3)
        instance.tick()
        self.assertEqual(self.popen.call_count, 3)

    def test_invalid_pack_latches_no_repeated_audit_or_astra(self):
        full_fixture(self.repo)
        instance = self.instance()
        instance.tick()
        self.complete_action(instance, 'verify-results')
        with patch.object(monitor, 'verify_pack', side_effect=monitor.NeedAction('Actual evidence missing')) as verify:
            for _ in range(8):
                instance.tick()
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(self.popen.call_count, 1)
        self.assertIn('research_gate_blocked', instance.state)

    def test_cli_flags_exact_model_max_never_preserve_provider(self):
        for model in (monitor.SOL, monitor.ASTRA):
            argv = monitor.codex_argv(model, self.repo, self.repo / 'response.json', self.repo / 'schema.json')
            self.assertEqual(argv[argv.index('-m') + 1], model)
            self.assertIn('model_reasoning_effort="max"', argv)
            self.assertIn('approval_policy="never"', argv)
            self.assertEqual('--search' in argv, model == monitor.ASTRA)
            self.assertEqual(argv[argv.index('--sandbox') + 1],
                             'workspace-write' if model == monitor.ASTRA else 'danger-full-access')
            self.assertEqual(argv[argv.index('-C') + 1],
                             str(self.repo / monitor.RESEARCH_REL if model == monitor.ASTRA else self.repo))
            if model == monitor.ASTRA:
                for setting in ('writable_roots=[]', 'exclude_tmpdir_env_var=true',
                                'exclude_slash_tmp=true', 'network_access=true'):
                    self.assertIn('sandbox_workspace_write.' + setting, argv)
            self.assertNotIn('--add-dir', argv)
            self.assertNotIn('--ignore-user-config', argv)
            self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', argv)
            self.assertFalse(any('model_provider' in s for s in argv))

    def test_research_directory_exists_before_supervisor_spawn(self):
        handoff_fixture(self.repo)
        folder = self.repo / monitor.RESEARCH_REL
        def spawn(*_, **__):
            self.assertTrue(folder.is_dir())
            return Mock(pid=12345)
        self.popen.side_effect = spawn
        instance = self.instance()
        instance.dispatch('deep-research', 'research', {})
        request = monitor.read(Path(instance.state['actions']['deep-research']['attempt']) / 'request.json')
        self.assertEqual((request['model'], request['reasoning_effort']), (monitor.ASTRA, 'max'))
        self.popen.assert_called_once()

    def test_old_needs_action_ledgers_survive_new_failure_and_restart(self):
        value = handoff_fixture(self.repo)
        instance = self.instance()
        old_receipts = {}
        for key in ('repair:old', 'closeout:R109'):
            instance.dispatch(key, 'execution', {})
            attempt = Path(instance.state['actions'][key]['attempt'])
            monitor.save(attempt / 'exit.json', dict(state='needs_action', reason='old read-only invocation'))
            old_receipts.update({str(p): monitor.digest(p) for p in attempt.glob('*') if p.is_file()})
        instance.reconcile()
        old_actions = json.loads(json.dumps(instance.state['actions']))
        queue = monitor.read(value['queue_state_path'])
        queue.update(state='failed', error=dict(stage='bootstrap', message='new authoritative wrapper failure'))
        monitor.save(value['queue_state_path'], queue)
        restarted = self.instance()
        for _ in range(4):
            restarted.tick()
        self.assertEqual(self.popen.call_count, 3)
        self.assertEqual(len(restarted.state['actions']), 3)
        for key, action in old_actions.items():
            self.assertEqual(restarted.state['actions'][key], action)
        for path, expected in old_receipts.items():
            self.assertEqual(monitor.digest(path), expected)

    def worker_fixture(self, model=monitor.SOL, stage='execution'):
        path = self.repo / monitor.RUN_REL / 'actions/fixture'
        monitor.save(path / 'request.json', dict(action_id='fixture', model=model, stage=stage,
                     repo=str(self.repo), schema=str(self.repo / 'schema.json')))
        text_file(path / 'prompt.md')
        return path

    def test_worker_success_bounded_and_not_provider_proof(self):
        path = self.worker_fixture()
        def finish(**kwargs):
            self.assertEqual(kwargs['timeout'], monitor.MODEL_TIMEOUT_SECONDS)
            monitor.save(path / 'response.json', dict(action_id='fixture', model=monitor.SOL,
                         reasoning_effort='max', status='complete', made_progress=True, summary='Saved real artifact'))
            return 0
        self.popen.return_value.wait.side_effect = finish
        self.assertEqual(monitor.worker(path), 0)
        exit_record = monitor.read(path / 'exit.json')
        self.assertEqual(exit_record['requested_model'], monitor.SOL)
        self.assertEqual(exit_record['requested_reasoning_effort'], 'max')
        self.assertIn('danger-full-access', exit_record['argv'])
        self.assertEqual(exit_record['cwd'], str(self.repo))
        self.assertFalse(exit_record['provider_route_verified'])
        environment = self.popen.call_args.kwargs['env']
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(monitor.worker(path), 0)
        self.assertEqual(self.popen.call_count, 1)

    def test_astra_worker_uses_only_research_cwd_and_records_actual_argv(self):
        path = self.worker_fixture(monitor.ASTRA, 'research')
        folder = self.repo / monitor.RESEARCH_REL
        def finish(**_):
            self.assertTrue(folder.is_dir())
            monitor.save(path / 'response.json', dict(action_id='fixture', model=monitor.ASTRA,
                         reasoning_effort='max', status='complete', made_progress=True, summary='Saved research prose'))
            return 0
        self.popen.return_value.wait.side_effect = finish
        self.assertEqual(monitor.worker(path), 0)
        self.assertEqual(self.popen.call_args.kwargs['cwd'], folder)
        argv = self.popen.call_args.args[0]
        self.assertEqual(argv[argv.index('-C') + 1], str(folder))
        self.assertIn('workspace-write', argv)
        self.assertIn('--search', argv)
        self.assertEqual(monitor.read(path / 'codex.json')['argv'], argv)
        exit_record = monitor.read(path / 'exit.json')
        self.assertEqual(exit_record['argv'], argv)
        self.assertFalse(exit_record['provider_route_verified'])

    def test_worker_rejects_astra_execution_without_substitution(self):
        path = self.worker_fixture(monitor.ASTRA, 'execution')
        self.assertEqual(monitor.worker(path), 1)
        self.assertIn('no fallback', monitor.read(path / 'exit.json')['reason'])
        self.popen.assert_not_called()

    def test_worker_missing_or_no_progress_output_is_not_retried(self):
        path = self.worker_fixture()
        self.popen.return_value.wait.return_value = 0
        self.assertEqual(monitor.worker(path), 1)
        self.assertEqual(monitor.read(path / 'exit.json')['state'], 'needs_action')
        monitor.worker(path)
        self.assertEqual(self.popen.call_count, 1)

    def test_worker_timeout_terminates_only_owned_cli(self):
        path = self.worker_fixture()
        self.popen.return_value.wait.side_effect = [subprocess.TimeoutExpired('mocked_codex', monitor.MODEL_TIMEOUT_SECONDS), 0]
        self.assertEqual(monitor.worker(path), 1)
        self.popen.return_value.terminate.assert_called_once()
        self.assertIn('six hours', monitor.read(path / 'exit.json')['reason'])
        self.kill.assert_not_called()
        self.killpg.assert_not_called()

    def test_pause_in_spawn_gap_prevents_model_call(self):
        path = self.worker_fixture()
        monitor.save(self.repo / monitor.RUN_REL / 'control.json', dict(paused=True))
        self.assertEqual(monitor.worker(path), 1)
        self.assertIn('paused before model launch', monitor.read(path / 'exit.json')['reason'])
        self.popen.assert_not_called()

    def test_model_unavailable_halts_future_model_calls_no_fallback(self):
        handoff_fixture(self.repo)
        instance = self.instance()
        instance.dispatch('closeout:R112', 'execution', {})
        action = instance.state['actions']['closeout:R112']
        monitor.save(Path(action['attempt']) / 'exit.json', dict(state='needs_action', model_unavailable=True,
                     reason='Requested model unavailable; no fallback'))
        instance.tick()
        self.assertEqual(instance.state['model_unavailable'], monitor.SOL)
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(instance.state['phase'], 'needs_action')

    def test_worker_detects_model_not_found_without_model_probe(self):
        path = self.worker_fixture()
        def finish(**_):
            text_file(path / 'stderr.log', 'model_not_found: requested model does not exist\n')
            return 1
        self.popen.return_value.wait.side_effect = finish
        self.assertEqual(monitor.worker(path), 1)
        self.assertTrue(monitor.read(path / 'exit.json')['model_unavailable'])
        self.assertEqual(self.popen.call_count, 1)

    def test_start_never_blindly_duplicates_tmux_or_pid(self):
        handoff_fixture(self.repo)
        self.run_command.return_value = Mock(returncode=0)
        with self.assertRaises(monitor.NeedAction):
            monitor.start(self.repo, self.repo / monitor.RUN_REL / 'queue_handoff.json')
        self.assertEqual(self.run_command.call_count, 1)
        monitor.save(self.repo / monitor.RUN_REL / 'state.json', dict(controller=dict(pid=12345)))
        with patch.object(monitor, 'alive', return_value=True):
            result = monitor.start(self.repo, self.repo / monitor.RUN_REL / 'queue_handoff.json')
        self.assertEqual(result['state'], 'already_running')
        self.assertEqual(self.run_command.call_count, 1)

    def test_run_sleeps_300_seconds_without_busy_checks(self):
        handoff_fixture(self.repo)
        instance = self.instance()
        instance.wakeup = Mock()
        with patch.object(instance, 'tick', side_effect=[False, True]), patch.object(monitor.signal, 'signal'):
            self.assertEqual(instance.run(), 0)
        instance.wakeup.wait.assert_called_once_with(300)
        self.popen.assert_not_called()

    def test_source_symlinks_and_bad_receipts_fail_closed(self):
        handoff_fixture(self.repo)
        path = self.repo / monitor.RUN_REL / 'queue_handoff.json'
        path.write_text('{invalid')
        self.assertEqual(self.instance().sources.observe()['queue']['state'], 'needs_action')
        outside = Path(self.temp.name) / 'outside.json'
        monitor.save(outside, dict(state='complete'))
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaises(monitor.NeedAction):
            monitor.inside(path, self.repo)
        self.popen.assert_not_called()

    def test_research_prompts_have_scientific_scope_and_no_extra_arms(self):
        folder = monitor.REPO / monitor.RUN_REL / 'prompts'
        for name in ('verify.md', 'research.md'):
            content = (folder / name).read_text()
            self.assertIn('ACQUISITION/response pilot', content)
            self.assertIn('function learner', content)
            self.assertIn('R57', content)
        research = (folder / 'research.md').read_text()
        for phrase in ('36 latent', '12 fragments', '32 TRAIN bases', 'posterior', 'primary papers', 'No additional NN arms'):
            self.assertIn(phrase, research)
        self.assertIn('absolute invocation repo', research)
        self.assertIn('Never add the repository or monitor directory as a writable root', research)

    def test_bootstrap_prompt_allows_one_proved_wrapper_repair_not_old_retries(self):
        content = (monitor.REPO / monitor.RUN_REL / 'prompts/execution.md').read_text()
        for phrase in ('Empty recovery_argv alone is NOT', 'ONE bounded bootstrap repair',
                       'before entry/worker/CUDA initialization', 'model/source/checkpoints are untouched',
                       'Missing worker files alone are not proof', 'Claim that one repair durably BEFORE',
                       'new\nversioned wrapper and approval receipt', 'no second repair',
                       'do NOT duplicate that fix', "Do not edit main's queue_handoff/state"):
            self.assertIn(phrase, content)


if __name__ == '__main__':
    unittest.main()
