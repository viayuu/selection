"""Three bounded CPU mocks; never create a production root or launch a child."""

from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from . import r121_followup_numeric as api


def contracts():
    old = json.loads(api.LEGACY_CONTRACT.read_text())
    groups = list(old['deployment_counts'])
    special = groups[:5]
    source_map = {key: str(api.PREPARATION / ('collection_input_boundary_v1' if key in special else
        'collection_explicit_cpu_v2' if index % 4 == 1 else 'collection_persistent_v1')) for index, key in enumerate(groups)}
    numeric = dict(version=api.NUMERIC_VERSION, source_policy=api.POLICY, input_boundary_keys=special,
        source_root_by_deployment=source_map, numeric_sha256='fixture-numeric',
        candidate_pool_unchanged=True, historical_labels_changed=False, solver_model_weights_seeds_budget_unchanged=True)
    return old, groups, numeric


def mocked_create(ns):
    """Run the real private create body with every write and manifest read intercepted."""
    old, groups, numeric = contracts()
    fixture_manifest = api.PREPARATION / '_MOCK_ONLY_manifest.json'
    metadata = {api.LEGACY_CONTRACT: old, api.REPLACEMENT: dict(main_owned=True, observer_pid=1202384),
        api.COLLECTION / 'protocol.json': dict(input_manifest=str(fixture_manifest), group_order=groups),
        api.COLLECTION / 'numeric.json': numeric,
        api.PREPARATION / 'learner_READY.json': dict(learner_interface_ready=True, source_sha256={})}
    pinned_files = {Path(item['path']): item['sha256'] for item in old['immutable_inputs']}
    pinned_files.update({api.REPO / key: value for key, value in old['source_sha256'].items()})
    pinned_files.update({api.REPO / key: value for key, value in api.EXTRA_PINS.items()})
    pinned_files[api.REPO / api.SELF_SOURCE] = api._private()['sha'](api.REPO / api.SELF_SOURCE)
    opened, writes = [], []
    def open_mock(path, mode='r', *args, **kwargs):
        path = Path(path)
        opened.append((path, mode))
        if path == fixture_manifest and mode == 'r':
            return io.StringIO(json.dumps(dict(tasks=[], endpoints=list(range(23040)))))
        if path == api.ROOT / 'contract.json' and mode == 'x':
            class Saved(io.StringIO):
                def __exit__(self, *args):
                    writes.append(json.loads(self.getvalue()))
                    return super().__exit__(*args)
            return Saved()
        raise AssertionError('Unexpected filesystem open: ' + str(path))
    with ExitStack() as stack:
        stack.enter_context(patch.object(api, '_unstarted'))
        stack.enter_context(patch.object(Path, 'exists', return_value=False))
        stack.enter_context(patch.object(Path, 'mkdir'))
        stack.enter_context(patch.object(Path, 'open', open_mock))
        stack.enter_context(patch.dict(ns, read=lambda p: copy.deepcopy(metadata[Path(p)]),
            sha=lambda p: pinned_files.get(Path(p), 'fixture-file-hash'),
            Counter=lambda rows: copy.deepcopy(old['deployment_counts'])))
        result = ns['create'](api.ROOT)
    return result, writes, opened


class NumericFollowupTests(unittest.TestCase):
    def test_private_compile_and_full_contract_preserve_original_pins(self):
        before = api.LEGACY_SOURCE.read_bytes()
        ns = api._private()
        result, writes, opened = mocked_create(ns)
        old, _, numeric = contracts()
        self.assertEqual(api.LEGACY_SOURCE.read_bytes(), before)
        self.assertEqual(result, writes[0])
        self.assertEqual(result['schema'], api.SCHEMA)
        self.assertEqual(result['first_NN_attempt'], 'principal_v1')
        self.assertEqual((len(result['deployment_counts']), result['planned_labels'], result['endpoints']), (89, 113920, 23040))
        self.assertEqual((result['successful_updates'], result['logical_batch'], result['seed']), (14400, 1024, 2))
        self.assertEqual((result['micro_cap'], result['eval_batch_size'], result['lib_batch_size'], result['wandb_mode']), (352, 256, 1, 'offline'))
        self.assertEqual(result['collection_root'], str(api.COLLECTION))
        self.assertEqual(result['source_root_by_deployment'], numeric['source_root_by_deployment'])
        for key, digest in old['source_sha256'].items():
            self.assertEqual(result['source_sha256'][key], digest)
        self.assertEqual(set(result['source_sha256']) - set(old['source_sha256']), {*api.EXTRA_PINS, api.SELF_SOURCE})
        inputs = {item['path'] for item in result['immutable_inputs']}
        self.assertIn(str(api.COLLECTION / 'numeric.json'), inputs)
        self.assertNotIn(str(api.COLLECTION / 'universal.json'), inputs)
        self.assertIn(str(api.REPLACEMENT), inputs)
        self.assertEqual(len(result['progress_source_roots']), 3)
        self.assertEqual([p for p, mode in opened if mode != 'r'], [api.ROOT / 'contract.json'])
        args = ns['common_args'](result)
        self.assertEqual(args[args.index('--family-labels') + 1], str(api.COLLECTION / 'family_labels.jsonl'))
        self.assertEqual(args[args.index('--root') + 1], str(api.LEARNER))
        with patch.object(api.PLAN_PATH.__class__, 'exists', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Never|never'):
                api._unstarted()

    def test_only_numeric_progress_and_qualifications_can_release_barrier(self):
        ns = api._private()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current, historical = root / 'numeric', root / 'history'
            contract = dict(collection_root=str(current), deployment_counts={'TSP/BQ': 1280},
                            planned_labels=1280, progress_source_roots=[str(historical)])
            for base, count in ((current, 1279), (historical, 1280)):
                (base / 'progress').mkdir(parents=True)
                (base / 'progress/TSP__BQ.json').write_text(json.dumps(dict(completed=count)))
            (current / 'bootstrap').mkdir()
            (current / 'preflight').mkdir()
            for lane in range(4):
                (current / 'bootstrap' / ('lane' + str(lane) + '.json')).write_text('{}')
            qualification = current / 'preflight/TSP__BQ.json'
            qualification.write_text('{}')
            result = ns['label_progress'](contract)
            self.assertFalse(result['full89_complete'])
            self.assertEqual((result['completed_labels'], result['collected_unique_labels']), (1279, 1280))
            (current / 'progress/TSP__BQ.json').write_text(json.dumps(dict(completed=1280)))
            self.assertTrue(ns['label_progress'](contract)['full89_complete'])
            qualification.unlink()
            self.assertFalse(ns['label_progress'](contract)['full89_complete'])
            qualification.write_text('{}')
            (historical / 'progress/TSP__BQ.json').write_text(json.dumps(dict(completed=99999)))
            result = ns['label_progress'](contract)
            self.assertTrue(result['full89_complete'])
            self.assertIn('historical_observation_error', result)

    def test_inherited_run_stage_order_export_and_single_principal_plan(self):
        ns = api._private()
        contract, _, _ = mocked_create(ns)
        old, _, _ = contracts()
        events, saved = [], {}
        progress = iter([dict(full89_complete=False, failures=[]), dict(full89_complete=True, failures=[])])
        def read_mock(path):
            if Path(path) == api.LEGACY_CONTRACT:
                return copy.deepcopy(old)
            if Path(path) == api.ROOT / 'contract.json':
                return copy.deepcopy(contract)
            raise AssertionError(path)
        def command(root, name, argv):
            events.append((name, argv))
            return dict(exit_status=0)
        fake_transport = ModuleType('code.V4.return18_launch')
        fake_transport.validate_plan = Mock(return_value=({}, 'fixture-plan-digest'))
        package = sys.modules['code.V4']
        with tempfile.TemporaryFile() as lockfile, ExitStack() as stack:
            stack.enter_context(patch.object(api, '_unstarted'))
            stack.enter_context(patch.object(Path, 'open', lambda *a, **k: lockfile))
            stack.enter_context(patch.object(ns['fcntl'], 'flock'))
            stack.enter_context(patch.dict(sys.modules, {'code.V4.return18_launch': fake_transport}))
            stack.enter_context(patch.object(package, 'return18_launch', fake_transport, create=True))
            stack.enter_context(patch.dict(ns, read=read_mock,
                source_hashes=lambda: dict(contract['source_sha256']), verify_sources=Mock(),
                sha=lambda path: contract['source_sha256'].get(str(Path(path).relative_to(api.REPO)), 'fixture-hash'),
                write=lambda path, data, exclusive=False: saved.__setitem__(Path(path), copy.deepcopy(data)),
                label_progress=lambda record: next(progress),
                freeze_protocol=lambda record: dict(path='fixture/protocol.json', sha256='fixture-protocol'),
                command=command))
            stack.enter_context(patch.object(ns['time'], 'sleep', side_effect=lambda seconds: events.append(('wait', seconds))))
            result = ns['run'](api.ROOT, 10)
        self.assertTrue(result['complete'])
        self.assertEqual([name for name, _ in events], ['wait', 'full89_export', 'family_cpu_features',
            'principal_gpu_stages', 'saved_only_finish', 'four_route_analysis', 'publish'])
        self.assertIn('code.V4.r121_numeric_collection', events[1][1])
        self.assertIn(str(api.COLLECTION / 'family_labels.jsonl'), events[2][1])
        plan = saved[api.PLAN_PATH]
        self.assertEqual([s['id'] for s in plan['jobs'][0]['stages']], ['train', 'test', 'lib', 'locked_train_and_family_capture'])
        self.assertEqual(plan['jobs'][0]['root'], str(api.LEARNER.relative_to(api.REPO)))
        self.assertEqual(plan['source_sha256'], contract['source_sha256'])
        self.assertEqual(set(saved) - {api.PLAN_PATH}, {api.ROOT / name for name in
            ('claim.json', 'status.json', 'learner_launch.json', 'complete.json')})
        for stage in plan['jobs'][0]['stages']:
            argv = stage['argv']
            self.assertEqual(argv[argv.index('--family-labels') + 1], str(api.COLLECTION / 'family_labels.jsonl'))
            self.assertNotIn('--resume', argv)


if __name__ == '__main__':
    unittest.main()
