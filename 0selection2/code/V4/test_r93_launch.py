"""Focused CPU-only R93 launcher tests; external execution fails closed."""

import argparse
from contextlib import ExitStack
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from . import r93_launch as launch


def protocol_fixture():
    return dict(experiment='R93', locked=True, arms=['A', 'B'], seed=2,
        logical_global_batch=1024, batch_size=1024, task_scope='full18',
        data_root=str(launch.DATA), prepared_root=str(launch.PREPARED),
        rows_per_problem=dict(train=81000, val=4500, test=4500),
        max_updates=57600, warmup_updates=4320, eval_every=1440,
        arm_budgets=dict(A=57600, B=57600), large_micro_cap=352, eval_batch_size=256,
        dropout=dict(A=0.1, B=0.0), arm_descriptions=dict(A=dict(dropout=0.1), B=dict(dropout=0.0)),
        wandb_mode='offline')


def parser_stub(runner):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'test') if runner else ('lib', 'report', 'compare'))
    for name in (('root', 'prepared-root', 'data-root', 'cache-root', 'protocol-lock') if runner else ('root',)):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=launch.ARMS, required=True)
    parser.add_argument('--device', required=True)
    if runner:
        parser.add_argument('--gpu-type', choices=('3090', '4090'), required=True)
        parser.add_argument('--large-micro-cap', type=int, required=True)
        parser.add_argument('--eval-batch-size', type=int, required=True)
        parser.add_argument('--wandb-mode', choices=('offline',), required=True)
        parser.add_argument('--resume', action='store_true')
    else:
        parser.add_argument('--lib-batch-size', type=int, default=2)
    return parser


def runtime_stubs():
    def forbidden(*args, **kwargs):
        raise AssertionError('Parser verification must not train, report, or write a protocol')
    return [SimpleNamespace(ARMS=launch.ARMS, parser=lambda: parser_stub(True),
                            train=forbidden, test=forbidden, protocol_lock=forbidden),
            SimpleNamespace(ARMS=launch.ARMS, parser=lambda: parser_stub(False),
                            lib=forbidden, report=forbidden, compare=forbidden)]


class R93LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ('run', 'Popen'):
            self.stack.enter_context(mock.patch.object(launch.base.subprocess, name,
                side_effect=AssertionError('No external command in focused CPU tests')))
        # Use tiny source lists except for the dedicated real dependency test.
        self.real_sources = launch._sources
        self.stack.enter_context(mock.patch.object(launch, '_sources',
            return_value=[str(Path(launch.__file__).resolve())]))

    def patch(self, target, name, *args, **kwargs):
        return self.stack.enter_context(mock.patch.object(target, name, *args, **kwargs))

    def final_lock(self, value=None):
        path = self.root / 'protocol_lock.json'
        launch.base.save(path, protocol_fixture() if value is None else value)
        self.patch(launch, 'PROTOCOL_LOCK', path)
        return path

    def predecessor_fixture(self):
        models, receipts = self.root / 'R92_models', self.root / 'R92_launch'
        self.patch(launch, 'R92_MODELS', models)
        self.patch(launch, 'R92_LAUNCH', receipts)
        for arm in ('C', 'D'):
            model, directory = models / arm, receipts / arm
            model.mkdir(parents=True)
            (model / 'best.pt').write_bytes(b'synthetic-unchanged-checkpoint')
            lock = dict(experiment='R92', arm=arm, checkpoint=str(model / 'best.pt'),
                checkpoint_sha256=launch.base.sha(model / 'best.pt'), selected_without_test=True,
                per_problem_assembly=False, state_hash='state', selected_update=800)
            launch.base.save(model / 'final_test_lock.json', lock)
            launch.base.save(model / 'training_complete.json', dict(full18=True, selected_best=lock,
                tree_counts={str(index): 3000 for index in range(18)}))
            launch.base.save(model / 'test_results.json', dict(complete=True,
                checkpoint_sha256=lock['checkpoint_sha256'], state_hash='state', selected_update=800))
            launch.base.save(model / 'lib_stage_complete.json', dict(complete=True,
                checkpoint_sha256=lock['checkpoint_sha256'], datasets=['TSPLIB', 'CVRPLIB']))
            post = dict(phase='complete', complete=True)
            for key, name in (('model_statistics_pdf', 'MODEL_STATISTICS.pdf'),
                              ('selection_distributions_pdf', 'selection_distributions.pdf'),
                              ('target_recovery_pdf', 'target_recovery_and_train.pdf')):
                path = model / 'full_model_statistics' / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b'%PDF-synthetic-CPU-fixture')
                post[key] = str(path)
            launch.base.save(model / 'postprocess_status.json', post)
            if arm == 'D':
                for name, stages in (('worker', ('train', 'TEST', 'LIB')),
                                      ('controller', ('allocation_GPU_pipeline', 'PDF'))):
                    launch.base.save(directory / (name + '.json'), dict(state='complete', exit_status=0,
                        stages=[dict(name=stage, state='complete', exit_status=0) for stage in stages]))
                continue
            launch.base.save(model / 'protocol.json', dict(arm='C'))
            launch.base.save(model / 'error.json', dict(type='ValueError',
                message='Selected-bank VAL disagrees with staged CPU prediction selection'))
            for name, stage in (('worker', 'train'), ('controller', 'allocation_GPU_pipeline')):
                launch.base.save(directory / (name + '.json'), dict(state='failed', exit_status=1,
                    stages=[dict(name=stage, state='failed', exit_status=1)]))
            paths = [model / name for name in ('best.pt', 'protocol.json', 'error.json')]
            paths += [directory / name for name in ('worker.json', 'controller.json')]
            for index in range(18):
                for suffix in ('.cbm', '.json'):
                    path = model / 'tree_models' / (str(index) + suffix)
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b'synthetic-frozen-tree')
                    paths.append(path)
            protected = {str(path): launch.base.sha(path) for path in paths}
            stage = dict(complete=True, protected_unchanged=True, CPU_only=True,
                formal_VAL_lock_sha256=launch.base.sha(model / 'final_test_lock.json'),
                optimizer_updates=0, fitted_trees=0)
            launch.base.save(model / 'recovery_complete.json', dict(state='complete', complete=True,
                protected_unchanged=True, CPU_only=True, TEST_LIB_PDF_complete=True,
                original_failure_preserved=True, optimizer_updates=0, fitted_trees=0,
                automatic_retry=False, model_checkpoint_replaced=False, inference_rule_changed=False,
                protected_before=protected, protected_after=protected, selected_best=lock, test_stage=stage))
        return models, receipts

    def test_fixed_two_slots_never_claim_healthy_R92_slots(self):
        self.assertEqual(launch.ARMS, ('A', 'B'))
        self.assertEqual(launch.ARM_SLOTS, dict(A='3090_1', B='4090_1'))
        self.assertEqual(set(launch.SLOTS), {'3090_1', '4090_1'})
        for arm, slot, uuid, job, node in (
            ('A', '3090_1', 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d', 1465, 'gpu03'),
            ('B', '4090_1', 'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d', 1475, 'gpu05')):
            spec = launch.template(arm)
            item = launch.SLOTS[slot]
            self.assertEqual((item['uuid'], item['job'], item['node']), (uuid, job, node))
            self.assertEqual((spec['cpu_count'], spec['memory_gib']), (4, 9))
            self.assertEqual(launch.base.session_name(spec), 'r93_' + arm)
            self.assertEqual(spec['slots'], [slot])
            self.assertNotIn('cap', item)
        self.assertEqual(launch.MODEL_ROOT.name, 'R93_plain_classification')

    def test_template_before_runtime_or_gates_is_read_only(self):
        self.patch(launch, 'runtime_modules', side_effect=AssertionError('runtime not ready'))
        self.patch(launch, 'predecessor_gate', side_effect=AssertionError('not needed for template'))
        self.patch(launch.base, 'save', side_effect=AssertionError('no writes'))
        for arm in launch.ARMS:
            spec = launch.template(arm)
            self.assertEqual(spec['protocol']['arm_budget']['unit'], 'successful_updates')
            self.assertEqual(spec['protocol']['arm_budget']['count'], 57600)
            self.assertEqual(spec['protocol']['dropout'], 0.1 if arm == 'A' else 0)

    def test_current_protocol_and_both_NN_budgets(self):
        self.final_lock()
        value = launch.locked_protocol()
        self.assertEqual([launch.arm_budget(value, arm)['count'] for arm in launch.ARMS], [57600, 57600])
        for key, bad in (('locked', False), ('experiment', 'R92'), ('arms', ['A', 'B', 'C']),
                         ('batch_size', 512), ('seed', 1), ('task_scope', 'subset'),
                         ('data_root', '/old'), ('prepared_root', '/old'),
                         ('rows_per_problem', dict(train=8100)), ('wandb_mode', 'online'),
                         ('arm_budgets', dict(A=3000, B=57600)), ('warmup_updates', 4000),
                         ('eval_every', 1000), ('large_micro_cap', 176), ('eval_batch_size', 128),
                         ('dropout', dict(A=0, B=0))):
            changed = dict(protocol_fixture(), **{key: bad})
            launch.base.save(launch.PROTOCOL_LOCK, changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.locked_protocol()

    def test_parse_all_stage_and_comparison_CLIs_before_predecessor(self):
        self.final_lock()
        self.patch(launch, 'runtime_modules', return_value=runtime_stubs())
        self.patch(launch, 'predecessor_gate', side_effect=AssertionError('parsing precedes gates'))
        self.patch(launch.base, 'save', side_effect=AssertionError('parse cannot write'))
        result = launch.parse_commands()
        self.assertEqual(result['parsed_count'], 9)
        self.assertEqual([row['command'] for row in result['commands']],
                         ['train', 'test', 'lib', 'report'] * 2 + ['compare'])
        for key in ('training_started', 'source_hashes_frozen', 'protocol_written', 'resource_probe_performed'):
            self.assertFalse(result[key])

    def test_parser_mismatch_and_unready_runtime_never_prepare(self):
        self.final_lock()
        modules = runtime_stubs()
        parser = parser_stub(True)
        parser.set_defaults(resume=True)
        modules[0].parser = lambda: parser
        self.patch(launch, 'runtime_modules', return_value=modules)
        with self.assertRaisesRegex(RuntimeError, 'resume'):
            launch.parse_commands()
        self.patch(launch, 'runtime_modules', side_effect=RuntimeError('Waiting for final runtime'))
        prepare = self.patch(launch.adapter.engine, '_prepare')
        with self.assertRaisesRegex(RuntimeError, 'final runtime'):
            launch.prepare(self.root / 'spec.json', 'main ready', True)
        prepare.assert_not_called()

    def test_confirmation_precedes_write_start_or_probe(self):
        parse = self.patch(launch, 'parse_commands')
        start = self.patch(launch, '_start')
        save = self.patch(launch.base, 'save')
        for evidence, tested in (('', False), (' ', True), ('main ready', False)):
            for action in (launch.start, launch.prepare):
                with self.assertRaisesRegex(RuntimeError, 'explicit ready'):
                    action(self.root / 'spec.json', evidence, tested)
            with self.assertRaises(RuntimeError):
                launch.write_specs(evidence, tested)
        parse.assert_not_called()
        start.assert_not_called()
        save.assert_not_called()

    def test_specs_and_exact_stage_contract_no_implicit_overwrite(self):
        self.final_lock()
        self.patch(launch, 'ROOT', self.root / 'launch')
        self.patch(launch, 'runtime_modules', return_value=runtime_stubs())
        result = launch.write_specs('main final runtime ready', True)
        self.assertEqual(len(result['specs']), 2)
        path = Path(result['specs'][0])
        spec = launch.load_spec(path)
        self.assertEqual([stage['device'] for stage in spec['stages']], ['gpu', 'gpu', 'gpu', 'cpu'])
        spec['stages'][-1]['device'] = 'gpu'
        launch.base.save(path, spec)
        with self.assertRaisesRegex(ValueError, 'CPU PDF'):
            launch.load_spec(path)
        with self.assertRaisesRegex(RuntimeError, 'overwrite'):
            launch.write_specs('main final runtime ready', True)
        with self.assertRaisesRegex(ValueError, 'fixed slots'):
            launch.commands(launch.template('A'), '3090_0')

    def test_transport_exact_allocation_container_UUID_and_offline(self):
        for slot, item in launch.SLOTS.items():
            step = launch.base.srun_command('entry', slot, '/shared/spec.json', 'token', cpus=4, memory='9G')
            for flag in (f'--jobid={item["job"]}', f'--nodelist={item["node"]}', '--overlap',
                         '--exact', '--gres=none', '--cpus-per-task=4', '--mem=9G'):
                self.assertIn(flag, step)
            self.assertIn('conda activate easynco', step[-1])
            self.assertIn('code.V4.r93_launch entry', step[-1])
            command = launch.container_command('worker', slot, '/shared/spec.json', 'token')
            self.assertIn(launch.transport.CONTAINERS[item['node']][1], command)
            self.assertIn('CUDA_VISIBLE_DEVICES=' + item['uuid'], command)
            self.assertIn('R93_LAUNCH_TOKEN=token', command)
            self.assertIn('WANDB_MODE=offline', command)
            self.assertIn(str(launch.WRAPPER), command)
            self.assertFalse(any(word in step or word in command for word in ('ssh', 'salloc', 'sbatch')))
        with mock.patch.dict(launch.os.environ, WANDB_MODE='online'):
            self.assertEqual(launch.environment()['WANDB_MODE'], 'offline')
            self.assertEqual(launch.environment()['CUDA_VISIBLE_DEVICES'], '')

    def test_private_adapter_does_not_mutate_R92_or_frozen_helpers(self):
        from . import r92_launch, r84_launch, r81_queue
        self.assertIsNot(launch.adapter, r92_launch)
        for module in (r92_launch.base, r84_launch):
            self.assertIsNot(module, launch.base)
            self.assertNotEqual(module.TOKEN_ENV, launch.TOKEN_ENV)
        self.assertEqual(r92_launch.ARMS, ('A', 'B', 'C', 'D'))
        self.assertEqual(r92_launch.ARM_SLOTS['A'], '3090_0')
        self.assertIsNot(launch.transport, r81_queue)
        self.assertEqual(launch.adapter.engine.saved.ROOT, launch.CACHE_EVIDENCE.parent)

    def test_source_freeze_includes_R93_dependencies_R84_and_cache_evidence(self):
        sources = self.real_sources(protocol_fixture())
        for name in ('r93_launch.py', 'r93_experiment.py', 'r93_reporting.py', 'r93_model.py',
                     'r92_launch.py', 'r84_experiment.py', 'r84_prepare.py', 'test_r93_launch.py'):
            self.assertIn(str(launch.REPO / 'code/V4' / name), sources)
        for name in ('prepared.json', 'geometry_stats.json', 'common_full_train_sbs.json', 'identicalinit_seed2.pt'):
            self.assertIn(str(launch.PREPARED / name), sources)
        self.assertIn(str(launch.CACHE_EVIDENCE), sources)
        self.assertFalse(any(Path(path).suffix == '.npy' for path in sources))

    def test_cache_reuse_never_reads_arrays_copies_or_rebuilds(self):
        for split in ('train', 'val'):
            (self.root / split).mkdir()
        launch.base.save(self.root / '.r84_cache_identity.json',
            dict(data_root=str(launch.DATA), source=str(launch.CACHE_SOURCE)))
        self.patch(launch.base.shutil, 'copyfile', side_effect=AssertionError('no copy'))
        self.patch(launch.base, 'sha', side_effect=AssertionError('no array hashing'))
        manifest = dict(files=[dict(path='train/TSP/nodes.npy', bytes=10)], bytes=10, identity_sha256='prior')
        result = launch.base.stage_cache(dict(cache_root=str(self.root), data_root=str(launch.DATA),
            cache_source=str(launch.CACHE_SOURCE)), manifest)
        self.assertEqual(result['copied_bytes'], 0)
        for key in ('preprocessing', 'array_contents_read', 'cache_audit_repeated'):
            self.assertFalse(result[key])

    def test_C_recovery_and_D_full_delivery_pass_without_changing_old_failures(self):
        models, receipts = self.predecessor_fixture()
        before = {path: launch.base.sha(path) for path in (receipts / 'C').glob('*.json')}
        for arm in launch.ARMS:
            result = launch.predecessor_gate(arm)
            self.assertTrue(result['complete'], result['reason'])
        result = launch.predecessor_gate('A')
        self.assertEqual(result['recovery']['path'], str(models / 'C/recovery_complete.json'))
        self.assertTrue(result['recovery']['original_failures_preserved'])
        self.assertEqual(before, {path: launch.base.sha(path) for path in before})
        self.assertEqual(launch.base.read(receipts / 'C/worker.json')['state'], 'failed')

    def test_C_original_failure_alone_never_authorizes_A(self):
        models, _ = self.predecessor_fixture()
        (models / 'C/recovery_complete.json').unlink()
        gate = launch.predecessor_gate('A')
        self.assertFalse(gate['complete'])
        self.assertFalse(gate['failed'])
        self.assertTrue(launch.predecessor_gate('B')['complete'])

    def test_C_recovery_flags_no_fitting_and_current_lock_are_required(self):
        models, _ = self.predecessor_fixture()
        path = models / 'C/recovery_complete.json'
        original = launch.base.read(path)
        for key, bad in (('complete', False), ('protected_unchanged', False),
                         ('TEST_LIB_PDF_complete', False), ('CPU_only', False),
                         ('optimizer_updates', 1), ('fitted_trees', 1),
                         ('automatic_retry', True), ('original_failure_preserved', False)):
            launch.base.save(path, dict(original, **{key: bad}))
            gate = launch.predecessor_gate('A')
            self.assertFalse(gate['complete'], key)
            self.assertTrue(gate['failed'], key)
        changed = copy.deepcopy(original)
        changed['selected_best']['selected_update'] = 1
        launch.base.save(path, changed)
        self.assertTrue(launch.predecessor_gate('A')['failed'])

    def test_C_changed_original_failure_or_best_SHA_blocks_A(self):
        models, receipts = self.predecessor_fixture()
        for path in (models / 'C/best.pt', receipts / 'C/worker.json', receipts / 'C/controller.json'):
            before = path.read_bytes()
            path.write_bytes(b'{}')
            gate = launch.predecessor_gate('A')
            self.assertFalse(gate['complete'])
            self.assertTrue(gate['failed'])
            path.write_bytes(before)

    def test_D_requires_PDF_rc0_and_same_lock_TEST_LIB_post(self):
        models, receipts = self.predecessor_fixture()
        path = receipts / 'D/controller.json'
        original = launch.base.read(path)
        for state, rc in (('running', None), ('failed', 1)):
            changed = dict(original, state=state, exit_status=rc)
            changed['stages'] = original['stages'][:1]
            launch.base.save(path, changed)
            self.assertFalse(launch.predecessor_gate('B')['complete'])
        launch.base.save(path, original)
        (models / 'D/lib_stage_complete.json').unlink()
        self.assertFalse(launch.predecessor_gate('B')['complete'])

    def test_start_requires_predecessor_before_tmux(self):
        self.patch(launch, 'parse_commands')
        self.patch(launch, 'load_spec', return_value=dict(run_id='A'))
        self.patch(launch, 'predecessor_gate', return_value=dict(complete=False, reason='waiting recovery'))
        start = self.patch(launch, '_start')
        with self.assertRaisesRegex(RuntimeError, 'waiting recovery'):
            launch.start(self.root / 'spec.json', 'main ready', True)
        start.assert_not_called()

    def test_resource_probes_require_main_ready_before_external_actions(self):
        self.patch(launch, 'check_ready', side_effect=RuntimeError('main not ready'))
        gpu = self.patch(launch.transport, 'gpu_inventory')
        allocation = self.patch(launch.transport, 'allocation')
        for function in (launch.node_resources, launch.container_resources):
            with self.assertRaisesRegex(RuntimeError, 'authorization'):
                function(None, '3090_1')
            with self.assertRaisesRegex(RuntimeError, 'main not ready'):
                function(self.root / 'spec.json', '3090_1')
        gpu.assert_not_called()
        allocation.assert_not_called()

    def test_healthy_R92_CPU_reservations_are_disjoint_and_PDF_releases_them(self):
        self.patch(launch, 'ROOT', self.root / 'R93')
        self.patch(launch, 'R92_LAUNCH', self.root / 'R92')
        for arm, node, cpus in (('A', 'gpu03', [0, 1, 2, 3]), ('B', 'gpu05', [4, 5, 6, 7])):
            directory = launch.R92_LAUNCH / arm
            launch.base.save(directory / 'claim.json', dict(node=node, cpus=cpus))
            launch.base.save(directory / 'controller.json', dict(state='running'))
        self.assertEqual(launch.reserved_cpus('gpu03', self.root / 'new'), {0, 1, 2, 3})
        launch.base.save(launch.R92_LAUNCH / 'A/controller.json', dict(state='reporting', gpu_finished_utc='done'))
        self.assertEqual(launch.reserved_cpus('gpu03', self.root / 'new'), set())

    def controller_fixture(self):
        spec, slot = launch.template('A'), '3090_1'
        self.patch(launch, 'ROOT', self.root / 'launch')
        self.patch(launch.base, 'ROOT', launch.ROOT)
        self.patch(launch.transport, 'ROOT', self.root / 'global_mutex')
        self.patch(launch.base.socket, 'gethostname', return_value='admin01')
        self.patch(launch.base, 'FONT', launch.WRAPPER)
        self.patch(launch, 'check_ready', return_value=(spec, {}))
        self.patch(launch, 'authorize_resources', return_value=(spec, {}))
        self.patch(launch, 'require_predecessor')
        probe = dict(gpus={slot: dict(compute_processes=[], memory_used_mib=0, utilization_pct=0)},
            processes=[dict(cpus=[0, 1, 2, 3])], allowed_cpus=list(range(8)),
            memory=dict(headroom_excluding_reclaimable_cache_bytes=20 * 2**30))
        self.patch(launch.transport, 'capture', return_value=json.dumps(probe))
        return spec, slot, probe

    def test_real_global_UUID_flock_released_before_ADMIN01_PDF(self):
        spec, slot, _ = self.controller_fixture()
        output = launch.ROOT / 'A'
        mutex = launch.transport.ROOT / ('queue_gpu_' + launch.SLOTS[slot]['uuid'] + '.json')
        seen = []

        def run(stage, _, record, __, env):
            seen.append(stage['name'])
            if stage['name'] == 'allocation_GPU_pipeline':
                with self.assertRaises(BlockingIOError), launch.base.lock(mutex, existing=True):
                    pass
                claim = launch.base.read(output / 'claim.json')
                self.assertEqual(claim['cpus'], [4, 5, 6, 7])
                launch.base.save(output / 'worker.json', dict(state='complete', exit_status=0,
                    stages=[dict(name=name, state='complete', exit_status=0) for name in ('train', 'TEST', 'LIB')]))
            else:
                with launch.base.lock(mutex, existing=True):
                    self.assertEqual(stage['name'], 'PDF')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(env['WANDB_MODE'], 'offline')
                self.assertEqual(stage['argv'][-1], 'cpu')
        self.patch(launch.base, 'run_stage', side_effect=run)
        self.assertEqual(launch.controller(self.root / 'spec.json'), 0)
        self.assertEqual(seen, ['allocation_GPU_pipeline', 'PDF'])
        self.assertEqual(launch.base.read(output / 'controller.json')['state'], 'complete')
        with self.assertRaisesRegex(RuntimeError, 'retry'):
            launch.controller(self.root / 'spec.json')

    def test_busy_slot_or_failed_GPU_pipeline_never_retries_or_runs_PDF(self):
        _, slot, probe = self.controller_fixture()
        probe['gpus'][slot]['memory_used_mib'] = 4096
        capture = self.patch(launch.transport, 'capture', return_value=json.dumps(probe))
        stage = self.patch(launch.base, 'run_stage')
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            launch.controller(self.root / 'spec.json')
        self.assertEqual(capture.call_count, 1)
        stage.assert_not_called()
        self.assertFalse((launch.ROOT / 'A/claim.json').exists())

    def test_GPU_pipeline_failure_stops_before_PDF(self):
        self.controller_fixture()
        stage = self.patch(launch.base, 'run_stage', side_effect=RuntimeError('GPU failed'))
        with self.assertRaisesRegex(RuntimeError, 'GPU failed'):
            launch.controller(self.root / 'spec.json')
        self.assertEqual(stage.call_count, 1)
        self.assertEqual(launch.base.read(launch.ROOT / 'A/controller.json')['state'], 'failed')

    def test_worker_failure_blocks_later_stages_without_retry(self):
        spec, slot, token = launch.template('A'), '3090_1', 'test-token'
        cpus = [12, 13, 14, 15]
        self.patch(launch.base, 'check_ready', return_value=(spec, dict(cache={})))
        self.patch(launch.base, 'assert_container', return_value='owned')
        self.patch(launch.base, 'stage_cache', return_value=dict(copied_bytes=0))
        self.patch(launch.base.os, 'sched_getaffinity', return_value=set(cpus))
        self.patch(launch.base.os, 'sched_setaffinity')
        self.stack.enter_context(mock.patch.dict(launch.os.environ, {launch.TOKEN_ENV: token,
            'CUDA_VISIBLE_DEVICES': launch.SLOTS[slot]['uuid']}))
        for failing in ('train', 'TEST', 'LIB'):
            self.patch(launch.base, 'ROOT', self.root / failing)
            output = launch.base.ROOT / 'A'
            launch.base.save(output / 'claim.json', dict(slot=slot, token=token, cpus=cpus))
            seen = []

            def run(stage, *args):
                seen.append(stage['name'])
                if stage['name'] == failing:
                    raise RuntimeError('simulated failure')

            with mock.patch.object(launch.base, 'run_stage', side_effect=run):
                with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                    launch._worker(self.root / 'spec.json', slot, token)
            order = ['train', 'TEST', 'LIB']
            self.assertEqual(seen, order[:order.index(failing) + 1])
            self.assertFalse(launch.base.read(output / 'worker.json')['automatic_retry'])

    def test_wrapper_only_dispatches_requested_stage_in_easynco(self):
        wrapper = launch.WRAPPER.read_text()
        self.assertIn('conda activate easynco', wrapper)
        self.assertIn('export WANDB_MODE=offline', wrapper)
        self.assertIn('code.V4.r93_launch "$@"', wrapper)
        self.assertNotIn('tmux', wrapper)
        self.assertNotIn('nvidia-smi', wrapper)


if __name__ == '__main__':
    unittest.main()
