"""Small CPU-only administrative fixtures. No live probes, launches or Git."""

from contextlib import redirect_stdout
import copy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r106_launch as launch
from . import r106_wave as wave


def launch_fixture(repo, arm=launch.ARMS[0]):
    root = repo / 'R106_native_encoder_adaptation'
    science = dict(experiment='R106', locked=True, arms=list(launch.ARMS), seed=2, task_scope='full18',
        data_root=str(launch.DATA_ROOT), prepared_root=str(launch.PREPARED_ROOT),
        primary_cache=str(launch.CACHE_ROOT), rows_per_problem=dict(train=81000, val=4500, test=4500),
        logical_global_batch=1024, max_updates=57600, warmup_updates=4320, eval_every=1440,
        train_eval_every=7200, current_query_labels_in_forward=False, **launch.microcaps())
    launch.base.save(root / 'protocol_lock.json', science)
    with patch.object(launch, 'MODEL_ROOT', root):
        spec = wave.make_spec(arm, final=True)
    path = repo / (launch.run_id(arm) + '.json')
    launch.base.save(path, spec)
    return root, path, spec, science


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='r106-admin-cpu-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)

    def test_private_transports_preserve_other_agents_and_shared_resources(self):
        from . import r98_launch, r105_launch, r84_launch, r81_queue
        self.assertIsNot(launch.adapter, r98_launch)
        self.assertIsNot(launch.base, r84_launch)
        self.assertIsNot(launch.base.transport, r81_queue)
        self.assertNotEqual(launch.ROOT, r98_launch.ROOT)
        self.assertEqual(launch.SHARED_DISPATCH, r98_launch.ROOT)
        self.assertEqual(launch.base.transport.ROOT, r81_queue.ROOT)
        self.assertEqual(r98_launch.TOKEN_ENV, 'R98_DISPATCH_TOKEN')
        self.assertEqual(r105_launch.ARM_SLOTS, dict(A_static_gate='3090_0', B_input_gate='3090_1'))
        self.assertEqual(set(launch.SLOTS), {'4090_0', '4090_1'})
        self.assertFalse(set(launch.SLOT_CPUS['4090_0']) & set(launch.SLOT_CPUS['4090_1']))
        for slot, expected in zip(launch.ARM_SLOTS.values(), (
                'GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be', 'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d')):
            item = launch.SLOTS[slot]
            self.assertEqual((item['job'], item['node'], item['uuid']), (1475, 'gpu05', expected))
            with patch.object(launch.base.subprocess, 'run', side_effect=AssertionError('No processes')):
                argv = launch.container_command('worker', slot, '/tmp/spec', 'fixture')
                srun = launch.base.srun_command('entry', slot, '/tmp/spec', cpus=4, memory='9G')
            self.assertIn(launch.CONTAINER, argv)
            self.assertIn(str(launch.WRAPPER), argv)
            self.assertNotIn('ssh', argv)
            for flag in ('--jobid=1475', '--nodelist=gpu05', '--cpus-per-task=4', '--mem=9G', '--gres=none'):
                self.assertIn(flag, srun)
            env = launch.environment(slot, 'fixture')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], expected)
            self.assertEqual(env['WANDB_MODE'], 'offline')
            self.assertEqual(env['WANDB_PROJECT'], 'selector')

    def test_uuid_full_normalization_and_true_nvml_binding(self):
        for slot in launch.SLOTS:
            expected = launch.SLOTS[slot]['uuid']
            for value in (expected, expected[4:], expected[4:].upper(), expected.encode('ascii')):
                self.assertEqual(launch.normalize_gpu_uuid(value), expected)
                self.assertEqual(launch.verify_uuid_binding(slot, value, expected)['torch_uuid'], expected)
            for bad in (expected[:12], expected + 'suffix', '', '0', expected[4:].replace('-', '')):
                with self.assertRaises(ValueError):
                    launch.verify_uuid_binding(slot, bad, expected)
            with self.assertRaises(ValueError):
                launch.verify_uuid_binding(slot, expected, expected[4:])
            other = next(key for key in launch.SLOTS if key != slot)
            with self.assertRaises(ValueError):
                launch.verify_uuid_binding(slot, launch.SLOTS[other]['uuid'], expected)

    def test_draft_is_possible_without_science_but_never_preparable(self):
        with patch.object(launch, 'MODEL_ROOT', self.repo / 'absent'):
            spec = wave.make_spec(launch.ARMS[0])
            self.assertTrue(spec['spec_draft'])
            self.assertFalse(spec['main_thread_ready'])
            path = self.repo / 'draft.json'
            launch.base.save(path, spec)
            with self.assertRaisesRegex(ValueError, 'final spec'):
                launch.load_spec(path)
            with self.assertRaisesRegex(ValueError, 'main R106 protocol'):
                wave.make_spec(launch.ARMS[0], final=True)

    def test_exact_commands_caps_paths_and_no_routing_cache_stage(self):
        root, path, spec, science = launch_fixture(self.repo)
        with patch.object(launch, 'MODEL_ROOT', root):
            self.assertEqual(launch.load_spec(path), spec)
            commands = launch.base.commands(spec, '4090_0')
            self.assertEqual(tuple(stage['name'] for stage in commands), launch.STAGES)
            for stage, positional in zip(commands[:3], ('train', 'test', 'timing')):
                args = stage['argv']
                self.assertEqual(args[args.index('-m') + 1:args.index('-m') + 3], ['code.V4.r106_experiment', positional])
                self.assertEqual(args[args.index('--cache-root') + 1], str(launch.CACHE_ROOT))
                self.assertEqual(args[args.index('--gpu-type') + 1], '4090')
                for key, value in launch.microcaps(science).items():
                    self.assertEqual(args[args.index('--' + key.replace('_', '-')) + 1], str(value))
                self.assertNotIn('--feature-root', args)
                self.assertNotIn('--routing-feature-root', args)
            self.assertEqual(commands[-1]['argv'][-2:], ['--device', 'cpu'])
            for changes in (dict(slots=['3090_0']), dict(memory_gib=10), dict(cpus=[0, 1]),
                    dict(donor_cache='/tmp/new'), dict(stages=spec['stages'][1:])):
                launch.base.save(path, dict(spec, **changes))
                with self.assertRaises(ValueError):
                    launch.load_spec(path)
            launch.base.save(path, spec)
            for changes in (dict(logical_global_batch=64), dict(max_updates=2), dict(atsp_micro_cap=0)):
                launch.base.save(root / 'protocol_lock.json', dict(science, **changes))
                with self.assertRaises(ValueError):
                    launch.load_spec(path)

    def test_final_cap_changes_require_exact_new_scientific_spec(self):
        root, path, spec, science = launch_fixture(self.repo)
        science['large_micro_cap'] = 32
        launch.base.save(root / 'protocol_lock.json', science)
        with patch.object(launch, 'MODEL_ROOT', root):
            with self.assertRaises(ValueError):
                launch.load_spec(path)
            final = wave.make_spec(launch.ARMS[0], final=True)
            launch.base.save(path, final)
            self.assertEqual(launch.load_spec(path), final)

    def test_missing_team_sources_block_before_prepare_or_ready(self):
        root, path, _, _ = launch_fixture(self.repo)
        with patch.object(launch, 'MODEL_ROOT', root), \
                patch.object(launch, 'declared_sources', return_value=[self.repo / 'pending.py']), \
                patch.object(launch.adapter, 'prepare') as actual:
            launch.base.save(path, wave.make_spec(launch.ARMS[0], final=True))
            with self.assertRaisesRegex(RuntimeError, 'pending; not main-ready'):
                launch.prepare(path, 'main CPU proof', True)
        actual.assert_not_called()

    def test_explicit_main_evidence_and_immutable_draft_final_files(self):
        for action in (launch.prepare, launch.start):
            for evidence, tested in ((None, True), (' ', True), ('proof', False)):
                with self.assertRaises(ValueError):
                    action('/absent', evidence, tested)
        root, _, _, _ = launch_fixture(self.repo)
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch, 'ROOT', self.repo / 'dispatch'):
            draft = wave.write_spec(launch.ARMS[0])
            final = wave.write_spec(launch.ARMS[0], final=True)
            self.assertNotEqual(draft.parent, final.parent)
            before = final.read_bytes()
            self.assertEqual(wave.write_spec(launch.ARMS[0], final=True).read_bytes(), before)
            launch.base.save(final, dict(unrelated=True))
            with self.assertRaisesRegex(RuntimeError, 'Preserve existing'):
                wave.write_spec(launch.ARMS[0], final=True)

    def test_source_freeze_is_exact_and_changes_never_refreeze(self):
        root, path, spec, _ = launch_fixture(self.repo)
        output = self.repo / 'dispatch' / spec['run_id']
        source = self.repo / 'entry.py'
        source.write_text('VALUE = 1\n')
        plan = dict(spec=spec, main_thread_ready=True, evidence='main CPU proof',
            primary_cache_reuse_only=True, cache_copy=False, cache_preprocessing=False,
            automatic_retry=False, source_sha256={str(source): launch.base.sha(source)})
        launch.base.save(output / 'plan.json', plan)
        launch.base.save(output / 'ready.json', dict(main_thread_ready=True, runner_tests_passed=True,
            source_freeze_ready=True, evidence='main CPU proof', plan_sha256=launch.base.sha(output / 'plan.json')))
        with patch.object(launch, 'MODEL_ROOT', root), patch.object(launch.base, 'ROOT', output.parent), \
                patch.object(launch, 'source_files', return_value=[str(source)]):
            launch.check_ready(path)
            source.write_text('VALUE = 2\n')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                launch.check_ready(path)
        code = self.repo / 'code/V4'
        code.mkdir(parents=True)
        entry, dep, test = (code / name for name in ('r106_entry.py', 'dependency.py', 'test_exact.py'))
        entry.write_text('from .dependency import VALUE\n')
        dep.write_text('VALUE = 1\n')
        test.write_text('from .unrelated import VALUE\n')
        (code / 'unrelated.py').write_text('VALUE = 0\n')
        with patch.object(launch, 'declared_sources', return_value=[entry, test]), \
                patch.object(launch.adapter, 'REPO', self.repo), \
                patch.object(Path, 'glob', side_effect=AssertionError('No source globs')):
            self.assertEqual(set(launch.source_files(dict(source_files=[]))), {str(p) for p in (entry, dep, test)})

    def test_cache_must_exist_identity_match_and_never_repair(self):
        root = self.repo / 'cache'
        root.mkdir()
        manifest = dict(identity_sha256='frozen', files=[dict(path='tiny.bin', bytes=3)])
        launch.base.save(root / '.r84_cache_identity.json', dict(identity_sha256='frozen'))
        (root / 'tiny.bin').write_bytes(b'abc')
        self.assertFalse(launch.stage_cache(dict(cache_root=str(root)), manifest)['copied'])
        (root / 'tiny.bin').unlink()
        with self.assertRaisesRegex(RuntimeError, 'no repair or copy'):
            launch.stage_cache(dict(cache_root=str(root)), manifest)
        self.assertFalse((root / 'tiny.bin').exists())

    def test_affinity_and_cpu_reservations_include_shared_and_new_dispatch(self):
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=list(range(80)))):
            self.assertEqual(launch.node_resources('4090_1')['allowed_cpus'], [4, 5, 72, 73])
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=[4, 5])):
            with self.assertRaisesRegex(RuntimeError, 'affinity'):
                launch.node_resources('4090_1')
        shared, own = self.repo / 'shared', self.repo / 'own'
        for root, name, cpus in ((shared, 'old', [0, 1, 2, 3]), (own, 'new', [4, 5, 72, 73])):
            launch.base.save(root / name / 'claim.json', dict(node='gpu05', cpus=cpus))
        with patch.object(launch, 'ROOT', own), patch.object(launch, 'SHARED_DISPATCH', shared), \
                patch.object(launch, 'REPO', self.repo):
            self.assertEqual(launch.reserved_cpus('gpu05', own / 'new'), {0, 1, 2, 3})

    def test_start_does_not_wait_for_or_touch_running_r105(self):
        _, _, spec, _ = launch_fixture(self.repo)
        with patch.object(launch, 'check_ready', return_value=(spec, dict(evidence='main'))), \
                patch.object(launch, 'original_start', return_value=dict(mocked=True)) as start:
            self.assertTrue(launch.start('/spec', 'main', True)['mocked'])
            with self.assertRaises(ValueError):
                launch.start('/spec', 'different', True)
        start.assert_called_once_with('/spec', 'main', True)

    def test_admin_imports_and_status_have_no_torch_probe_or_subprocess(self):
        program = """import subprocess, sys
def forbidden(*a, **k): raise AssertionError('No subprocess/probes')
subprocess.run = subprocess.check_output = subprocess.Popen = forbidden
import code.V4.r106_launch, code.V4.r106_wave
import code.V4.r106_goal as goal
result = goal.status(root='/tmp/absent-r106-cpu-fixture', launch_root='/tmp/absent-r106-dispatch')
assert result['complete'] is False and result['errors'] == []
assert 'torch' not in sys.modules
print('CPU administrative import/status only')
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-B', '-c', program], cwd=launch.REPO,
            env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrapper_and_cpu_cli_only(self):
        text = launch.WRAPPER.read_text()
        self.assertLess(text.index('source '), text.index('conda activate easynco'))
        self.assertIn('exec python -B -u -m code.V4.r106_launch', text)
        self.assertNotIn('r105_launch', text)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(wave.main([launch.ARMS[0], '--print-only']), 0)
        self.assertTrue(json.loads(output.getvalue())['spec_draft'])

    def test_actual_runner_cli_cpu_contract_without_model_or_data(self):
        if not (launch.REPO / 'code/V4/r106_experiment.py').is_file():
            self.skipTest('Runner absent; not ready')
        program = """import json,sys,tempfile
from pathlib import Path
import torch
def forbidden(*a, **k): raise AssertionError('No GPU probe or work')
torch.cuda.is_available = torch.cuda.device_count = torch.cuda.get_device_properties = forbidden
from code.V4 import r106_experiment as runner
from code.V4 import r106_launch as launch, r106_wave as wave
assert 'code.V4.r106_model' not in sys.modules
for arm in launch.ARMS:
    spec = wave.make_spec(arm)
    for stage in launch.base.commands(spec, launch.ARM_SLOTS[arm])[:3]:
        args = stage['argv']; args = args[args.index('-m') + 2:]
        parsed = runner.parser().parse_args(args)
        assert parsed.arm == arm and not parsed.resume
        caps = {key:getattr(parsed,key) for key in launch.microcaps()}
        required = runner.required_lock_fields(**caps)
        assert required['native_lr_scale'] == .1 and required['native_feature_cache'] is False
with tempfile.TemporaryDirectory(prefix='r106-lock-cpu-') as directory:
    path=Path(directory)/'protocol_lock.json'
    record=dict(runner.required_lock_fields(),data_root=str(launch.DATA_ROOT),
        prepared_root=str(launch.PREPARED_ROOT),primary_cache=str(launch.CACHE_ROOT))
    path.write_text(json.dumps(record))
    args=runner.parser().parse_args(['train','--arm',launch.ARMS[0],'--protocol-lock',str(path)])
    assert runner.protocol_lock(args)['record'] == record
    path.write_text(json.dumps(dict(record,native_lr_scale=.2)))
    try:
        runner.protocol_lock(args)
    except ValueError:
        pass
    else:
        raise AssertionError('Detailed scientific mismatch was accepted')
print('six runner stage CLIs and full lock contract parsed, no model/data/GPU')
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-B', '-c', program], cwd=launch.REPO,
            env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_reporter_cli_if_landed_never_calls_stage(self):
        if not (launch.REPO / 'code/V4/r106_reporting.py').is_file():
            self.skipTest('Reporter source pending; five-stage main readiness is NOT established')
        program = """import torch
def forbidden(*a, **k): raise AssertionError('No GPU probe or work')
torch.cuda.is_available = torch.cuda.device_count = torch.cuda.get_device_properties = forbidden
from code.V4 import r106_reporting as reporting, r106_launch as launch, r106_wave as wave
for arm in launch.ARMS:
    for stage in launch.base.commands(wave.make_spec(arm),launch.ARM_SLOTS[arm])[-2:]:
        args=stage['argv']; args=args[args.index('-m')+2:]
        parsed=reporting.parser().parse_args(args)
        assert parsed.arm == arm
reporting.parser().parse_args(['compare','--root',str(launch.MODEL_ROOT),'--device','cpu'])
print('reporter CLI only')
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-B', '-c', program], cwd=launch.REPO,
            env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)


def goal_fixture(repo):
    from . import r106_goal as goal
    root, dispatch = repo / 'R106_native_encoder_adaptation', repo / 'dispatch'
    goal.save(root / 'protocol_lock.json', dict(experiment='R106', locked=True,
        original_fresh_trainable_sha256='a' * 64))
    locks = {}
    for index, arm in enumerate(goal.ARMS):
        run = root / arm
        config = dict(experiment='R106', arm=arm, seed=2, winner_rule='native FP64 ind',
            data_root='/frozen-new90k', prepared_root='/frozen-r84',
            rows_per_problem=dict(train=81000, val=4500, test=4500),
            candidate_pools={problem: ['a', 'b'] for problem in goal.PROBLEMS},
            baselines={problem: dict(selection_split='train', train_rows=81000) for problem in goal.PROBLEMS},
            max_successful_updates=57600, checkpoint_rule='strict minimum whole18 VAL macro_vs_sbs_pct; first exact tie',
            selected_val_binding_rule='exact_selected_checkpoint_model_state_protocol_state_and_successful_update',
            inference_rule='online native plus original selector maximin',
            main_protocol_lock=dict(sha256=goal.file_hash(root / 'protocol_lock.json')),
            native_execution='online_all_splits', native_feature_cache=False, native_eval=True,
            current_query_labels_in_forward=False, core_parameter_count=1941257,
            native_trainable=index == 1, native_lr_scale=.1, microbatch_caps=dict(TSP=64),
            task_order_sha256='order', native_pair_context='one microbatch native graph for two dropout forwards')
        goal.save(run / 'protocol.json', config)
        state, signature = goal.json_hash(config), str(index + 5) * 64
        lock = dict(experiment='R106', arm=arm, state_hash=state, checkpoint_sha256=str(index + 1) * 64,
            selection_split='val', selected_without_test=True, per_problem_assembly=False, selected_update=1440,
            rule=config['checkpoint_rule'], inference_rule=config['inference_rule'])
        locks[arm] = lock
        result = dict(identity=dict(experiment='R106', arm=arm, state_hash=state, successful_updates=1440,
            checkpoint_sha256=lock['checkpoint_sha256'], model_state_sha256=signature),
            macro=dict(top1=.5, actual_regret=.1), per_problem={p: dict(n=4500) for p in goal.PROBLEMS})
        val, train = copy.deepcopy(result), copy.deepcopy(result)
        proof = dict(rule=config['selected_val_binding_rule'], checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=state, selected_update=1440, model_state_sha256=signature,
            archive_capture_checkpoint_sha256=None, archive_identity_unchanged=True)
        for split, current in (('val', val), ('train', train)):
            current['identity'].update(split=split, checkpoint_sha256=None)
            if split == 'train':
                current['per_problem'] = {p: dict(n=81000) for p in goal.PROBLEMS}
            current['prediction_archive'] = str(run / 'evaluation' / split / 'selected_best/predictions.npz')
            goal.save(Path(current['prediction_archive']).parent / 'identity.json', current['identity'])
        val['archive_capture_identity'] = copy.deepcopy(val['identity'])
        val['identity']['checkpoint_sha256'] = lock['checkpoint_sha256']
        val['checkpoint_binding'] = proof
        final = dict(state_hash=state, selected_update=1440, checkpoint_sha256=lock['checkpoint_sha256'], val=val, train=train)
        counts = dict(total=11965097, selector=1941257, native=10023840, selector_trainable=1941257,
            native_trainable=10023840 if index else 0, trainable=11965097 if index else 1941257)
        init = dict(seed=2, historical_selector_checkpoint_loaded=False, parameter_counts=counts,
            total_parameter_count=counts['total'], trainable_parameter_count=counts['trainable'],
            initial_state_sha256='b' * 64, common_fresh_model_state_sha256='b' * 64,
            native_state_sha256='c' * 64, selector_parameter_sha256='d' * 64,
            common_parameter_sha256='a' * 64, identical_complete_fresh_parameters=True)
        timing = dict(experiment='R106', arm=arm, complete=True, state_hash=state,
            checkpoint_sha256=lock['checkpoint_sha256'], selected_update=1440, model_state_sha256=signature,
            inference_rule=lock['inference_rule'], device='cuda:0', repetitions=3, warmup_batches=1,
            all_current_model_included=True, donor_present=True, cached_feature_latency_used=False,
            end_to_end_measured=False, input_labels_read=False, model_reselection=False,
            scope='online_native_plus_selector_GPU_resident_inputs', excludes='I/O, H2D and solvers',
            per_problem=[dict(problem=p, instances=128, online_native_plus_selector=dict(batch_seconds=[.128]*3,
                mean_seconds_per_instance=.001, donor_calls=[1, 1, 1], warmup_donor_calls=1,
                native_output_dtype=['torch.float32'])) for p in goal.PROBLEMS])
        records = {'initialization.json': init, 'final_test_lock.json': lock, 'final_metrics.json': final,
            'training_complete.json': dict(successful_updates=57600, state_hash=state,
                updates_by_problem=dict.fromkeys(goal.PROBLEMS, 3200), stage='train'),
            'test_results.json': dict(complete=True, state_hash=state, checkpoint_sha256=lock['checkpoint_sha256'], result=result),
            'inference_timing.json': timing,
            'lib_stage_complete.json': dict(experiment='R106', complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
                datasets=['TSPLIB', 'CVRPLIB'], native_counts=dict(TSPLIB=49, CVRPLIB=100)),
            'postprocess_status.json': dict(complete=True, phase='complete', exit_status=0),
            'full_model_statistics/protocol.json': dict(experiment='R106', complete=True, checkpoint_lock=lock,
                inference_timing=timing, target_identity=dict(target_count=36, sha256='f' * 64),
                selected_val_binding=proof)}
        for name, record in records.items():
            goal.save(run / name, record)
        for name in goal.adapter.R99_PDFS:
            (run / 'full_model_statistics' / name).write_bytes(b'%PDF-synthetic-CPU-fixture')
        header = 'dataset,current_checkpoint_sha256,current_top1_native_pct,current_mean_cost,nss_top1_native_pct,nss_mean_cost,status,data_version,candidate_pool,winner_rule\n'
        body = ''.join(f'{dataset},{lock["checkpoint_sha256"]},50,10,,,missing_saved_NSS,frozen,new,native FP64 ind\n'
            for dataset in ('TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB'))
        (run / 'full_model_statistics/nss_comparison.csv').write_text(header + body)
        goal.save(dispatch / 'specs' / (launch.run_id(arm) + '.json'), dict(run_id=launch.run_id(arm),
            run_dir=str(run), slots=[launch.ARM_SLOTS[arm]], spec_draft=False,
            cpus=launch.SLOT_CPUS[launch.ARM_SLOTS[arm]], stages=[dict(name=name,
                device='cpu' if name == 'PDF' else 'gpu') for name in launch.STAGES]))
        for name, stages in (('worker', ['train', 'TEST', 'timing', 'LIB']),
                ('controller', ['allocation_GPU_pipeline', 'PDF'])):
            goal.save(dispatch / launch.run_id(arm) / (name + '.json'), dict(state='complete', exit_status=0,
                stages=[dict(name=stage, state='complete', exit_status=0) for stage in stages]))
    return root, dispatch, locks


def goal_comparison(root, locks):
    from . import r106_goal as goal
    goal.save(root / 'comparison_artifacts/protocol.json', dict(experiment='R106', complete=True,
        checkpoints=locks, training=False, inference=False))
    for name in ('paired_changes_B_vs_A.csv', 'target_recovery_B_vs_A.csv', 'target_recovery_summary_B_vs_A.csv'):
        (root / 'comparison_artifacts' / name).write_text('problem,n\nTSP,4500\n')


class GoalTests(unittest.TestCase):
    def setUp(self):
        from . import r106_goal as goal
        self.goal = goal
        temp = tempfile.TemporaryDirectory(prefix='r106-goal-cpu-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)

    def test_whole_version_same_init_but_distinct_trainables_and_no_goal_claim(self):
        root, dispatch, locks = goal_fixture(self.repo)
        result = self.goal.status(root=root, launch_root=dispatch)
        self.assertTrue(result['all_arms_ready'], result)
        self.assertFalse(result['complete'])
        goal_comparison(root, locks)
        self.assertTrue(self.goal.status(root=root, launch_root=dispatch)['complete'])
        self.assertFalse(result['goal_complete'])
        self.assertFalse((root / 'goal_progress.json').exists())

    def test_pending_report_stage_strings_and_declared_history_are_not_failures(self):
        root, dispatch, _ = goal_fixture(self.repo)
        for name in ('status.json', 'postprocess_status.json'):
            self.goal.save(root / 'A_frozen' / name, dict(phase='pending_report', stage='PDF',
                history=[dict(state='failed', exit_status=1, error='declared historical failure')],
                errors='previously declared history, not live'))
        result = self.goal.status(root=root, launch_root=dispatch)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['phase'], 'waiting_for_arms')
        self.assertFalse(result['complete'])
        self.assertEqual(self.goal.receipt_errors('science', dict(stage='train', stages=['train', 'val'])), [])

    def test_only_true_failures_scalar_list_and_nested_rc_are_terminal(self):
        for error in ('real current failure', ['real current failure', 'no retry']):
            self.assertIn('real current failure', str(self.goal.receipt_errors('fixture', dict(state='failed', errors=error))))
        self.assertTrue(self.goal.receipt_errors('fixture', dict(state='running', stages=[dict(name='train', exit_status=2)])))
        self.assertFalse(self.goal.receipt_errors('fixture', dict(phase='pending_report', stage='train')))
        self.assertFalse(self.goal.receipt_errors('fixture', dict(phase='running', exit_status=False)))
        root, dispatch, _ = goal_fixture(self.repo)
        self.goal.save(root / 'A_frozen/status.json', ['invalid receipt'])
        result = self.goal.status(root=root, launch_root=dispatch)
        self.assertFalse(result['complete'])
        self.assertEqual(result['errors'], [])
        self.assertTrue(result['invalid_receipts'])

    def test_null_capture_sha_requires_exact_parent_state_step_signature_and_proof(self):
        root, dispatch, _ = goal_fixture(self.repo)
        path = root / 'A_frozen/final_metrics.json'
        original = self.goal.optional(path)
        archive = root / 'A_frozen/evaluation/val/selected_best/identity.json'
        before = archive.read_bytes()
        for target, key, value in (('train', 'state_hash', None), ('train', 'successful_updates', 2880),
                ('train', 'model_state_sha256', None), ('val', 'model_state_sha256', 'other')):
            changed = copy.deepcopy(original)
            changed[target]['identity'][key] = value
            self.goal.save(path, changed)
            self.assertFalse(self.goal.status(root=root, launch_root=dispatch)['all_arms_ready'], (target, key))
        self.goal.save(path, original)
        self.assertTrue(self.goal.status(root=root, launch_root=dispatch)['all_arms_ready'])
        self.assertEqual(archive.read_bytes(), before)
        original['val']['checkpoint_binding']['archive_identity_unchanged'] = False
        self.goal.save(path, original)
        self.assertFalse(self.goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_online_timing_cannot_be_cache_hit_or_missing_donor_call(self):
        root, dispatch, _ = goal_fixture(self.repo)
        path = root / 'B_finetune/inference_timing.json'
        original = self.goal.optional(path)
        for changes in (dict(donor_present=False), dict(cached_feature_latency_used=True),
                dict(model_state_sha256='wrong'), dict(selected_update=2880)):
            self.goal.save(path, dict(original, **changes))
            self.assertFalse(self.goal.status(root=root, launch_root=dispatch)['all_arms_ready'])
        changed = copy.deepcopy(original)
        changed['per_problem'][0]['online_native_plus_selector']['donor_calls'] = [0, 0, 0]
        self.goal.save(path, changed)
        self.assertFalse(self.goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_all_pdfs_lib_counts_and_same_checkpoint_nss4_required(self):
        root, dispatch, locks = goal_fixture(self.repo)
        goal_comparison(root, locks)
        path = root / 'A_frozen/full_model_statistics/nss_comparison.csv'
        path.write_text(path.read_text().replace(locks['A_frozen']['checkpoint_sha256'], 'wrong'))
        self.assertFalse(self.goal.status(root=root, launch_root=dispatch)['complete'])
        (root / 'B_finetune/full_model_statistics/learning_curves.pdf').unlink()
        self.assertFalse(self.goal.status(root=root, launch_root=dispatch)['all_arms_ready'])

    def test_compare_and_publication_require_main_authorization_before_subprocess(self):
        root, dispatch, _ = goal_fixture(self.repo)
        with patch.object(self.goal.subprocess, 'run') as process:
            for action in (self.goal.compare_once, self.goal.publish, self.goal.watch):
                with self.assertRaises(ValueError):
                    action(root, dispatch)
        process.assert_not_called()

    def test_saved_cpu_compare_one_shot_and_preserved_failure(self):
        root, dispatch, _ = goal_fixture(self.repo)
        with patch.object(self.goal, 'require_ready'), \
                patch.object(self.goal.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as process:
            with self.assertRaisesRegex(RuntimeError, 'no retry'):
                self.goal.compare_once(root, dispatch, 'main')
            before = (root / 'goal_compare_receipt.json').read_bytes()
            with self.assertRaises(RuntimeError):
                self.goal.compare_once(root, dispatch, 'main')
        self.assertEqual(process.call_count, 1)
        self.assertEqual(process.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(before, (root / 'goal_compare_receipt.json').read_bytes())

    def test_quiet_watch_pending_then_complete_only_explicit_publish(self):
        root = self.repo / 'root'
        root.mkdir()
        pending = dict(errors=[], invalid_receipts=[], all_arms_ready=False, comparison_complete=False, complete=False)
        complete = dict(pending, all_arms_ready=True, comparison_complete=True, complete=True)
        with patch.object(self.goal, 'require_ready'), \
                patch.object(self.goal, 'status', side_effect=[pending, complete]), \
                patch.object(self.goal.time, 'sleep') as sleep, patch.object(self.goal, 'publish') as publish, \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.goal.watch(root, self.repo / 'dispatch', 1, False, 'main'), 0)
        self.assertEqual(len(output.getvalue().strip().splitlines()), 1)
        sleep.assert_called_once_with(1)
        publish.assert_not_called()

    def test_curated_paths_exclude_other_versions_arrays_tokens_and_secrets(self):
        root, dispatch, _ = goal_fixture(self.repo)
        code = self.repo / 'code/V4'
        code.mkdir(parents=True)
        for name in ('r106_goal.py', 'r105_goal.py', 'r106_unknown_future.py'):
            (code / name).write_text('FIXTURE = True\n')
        self.goal.save(root / 'preparation/unsafe.json', dict(token='fixture-not-a-real-token'))
        self.goal.save(root / 'preparation/raw_cost_array.json', dict(costs=[1, 2]))
        (root / 'preparation/secret.txt').write_text('fixture')
        (root / 'preparation/weights.pt').write_bytes(b'fixture')
        with patch.object(self.goal, 'ROOT', root), patch.object(self.goal, 'LAUNCH', dispatch), \
                patch.object(self.goal, 'REPO', self.repo), patch.object(self.goal, 'require_ready'):
            paths = self.goal.publication_paths('R106', root, dispatch, self.repo)
        self.assertIn('code/V4/r106_goal.py', paths)
        self.assertNotIn('code/V4/r105_goal.py', paths)
        self.assertNotIn('code/V4/r106_unknown_future.py', paths)
        self.assertFalse(any('unsafe' in path or 'raw_cost' in path or 'weights' in path for path in paths))

    def test_mocked_whole_version_publication_one_commit_push_and_shared_mutex(self):
        root, dispatch, locks = goal_fixture(self.repo)
        goal_comparison(root, locks)
        git_dir = self.repo / '.git'
        git_dir.mkdir()
        commands = []
        asset = 'code/V4/r106_goal.py'

        def git_output(args, cwd):
            if args == ['rev-parse', '--show-toplevel']:
                return str(self.repo)
            if args == ['rev-parse', '--absolute-git-dir']:
                return str(git_dir)
            if args == ['branch', '--show-current']:
                return 'main'
            if args == ['rev-parse', 'HEAD']:
                return 'new-head' if any(cmd[:2] == ['git', 'commit'] for cmd in commands) else 'old-head'
            if args == ['diff', '--cached', '--name-only', '-z']:
                return asset if any(cmd[:2] == ['git', 'add'] for cmd in commands) else ''
            raise AssertionError(args)

        with patch.object(self.goal, 'ROOT', root), patch.object(self.goal, 'LAUNCH', dispatch), \
                patch.object(self.goal.adapter, 'LAUNCH', dispatch), patch.object(self.goal, 'require_ready'), \
                patch.object(self.goal, 'publication_paths', return_value=[asset]), \
                patch.object(self.goal.adapter, 'publication_paths', return_value=[asset]), \
                patch.object(self.goal.adapter, 'git_output', side_effect=git_output), \
                patch.object(self.goal.adapter.subprocess, 'run', side_effect=lambda cmd, **kwargs: commands.append(cmd)):
            self.assertTrue(self.goal.publish(root, dispatch, 'main')['complete'])
            self.assertTrue(self.goal.publish(root, dispatch, 'main')['complete'])
        self.assertEqual(sum(cmd[:2] == ['git', 'commit'] for cmd in commands), 1)
        self.assertEqual(sum(cmd[:2] == ['git', 'push'] for cmd in commands), 1)
        self.assertTrue((git_dir / 'codex-publication.lock').is_file())
        self.assertFalse(any('reset' in cmd or 'checkout' in cmd for cmd in commands))

    def test_mocked_publication_preserves_unrelated_staged_work(self):
        root, dispatch, locks = goal_fixture(self.repo)
        goal_comparison(root, locks)
        git_dir = self.repo / '.git'
        git_dir.mkdir()

        def git_output(args, cwd):
            if args == ['rev-parse', '--show-toplevel']:
                return str(self.repo)
            if args == ['rev-parse', '--absolute-git-dir']:
                return str(git_dir)
            if args == ['branch', '--show-current']:
                return 'main'
            if args == ['diff', '--cached', '--name-only', '-z']:
                return 'unrelated-user-work.py'
            raise AssertionError(args)

        with patch.object(self.goal, 'ROOT', root), patch.object(self.goal.adapter, 'LAUNCH', dispatch), \
                patch.object(self.goal, 'require_ready'), patch.object(self.goal.adapter, 'git_output', side_effect=git_output), \
                patch.object(self.goal.adapter.subprocess, 'run') as process:
            with self.assertRaisesRegex(RuntimeError, 'staged'):
                self.goal.publish(root, dispatch, 'main')
        process.assert_not_called()
        self.assertFalse((root / 'github_publication.json').exists())


if __name__ == '__main__':
    unittest.main()
