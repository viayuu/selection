"""Small CPU checks using accepted real native inputs, without any solver replay."""

import copy
import itertools
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .nonrouting_data import (
    DIRECTIONS, INPUT_FIELDS, PREPARED_ROOT, PROBLEMS, READY_PROBLEMS, SCHEMA_VERSION,
    SOURCE_ROOT, IdentityRegistry, NonRoutingDataset, collate_nonrouting, instance_key,
    load_manifest, load_nonrouting_datasets, make_sample, native_labels, typed_inputs,
    write_prepared_split, _check_labels,
)


def real_records(problem, count=2):
    split = 'benchmark' if problem in ('MKP', 'SMTWTP') else 'train'
    directory = SOURCE_ROOT / 'runs/20261025_构造法首批/实例标签' / problem / split
    path = sorted(directory.glob('*.jsonl'))[0]
    with path.open() as stream:
        return [(json.loads(line), dict(path=str(path), line=row))
                for row, line in enumerate(itertools.islice(stream, count), 1)]


def real_samples(problem):
    records = real_records(problem)
    solver_ids = tuple(sorted(label['solver_id'] for label in records[0][0]['labels']))
    return [make_sample(record['instance'], record['labels'], solver_ids,
                        source_provenance=reference) for record, reference in records]


class NativeLabelsTest(unittest.TestCase):
    def test_min_max_ties_and_missing(self):
        ids = ('a', 'b', 'c', 'd')
        for direction, expected in (('min', [False, True, True, False]),
                                    ('max', [True, False, False, False])):
            labels = [dict(solver_id=s, objective=v, objective_direction=direction,
                           feasible=True, cost=-12345)
                      for s, v in zip(ids, (7., 4., 4.))]
            result = native_labels(labels, ids, direction)
            self.assertEqual(result['native_objective'].dtype, np.float64)
            np.testing.assert_array_equal(result['all_best_mask'], expected)
            np.testing.assert_array_equal(result['observed_mask'], [True, True, True, False])
            self.assertTrue(np.isnan(result['native_objective'][3]))
            self.assertEqual(int(result['canonical_winner_index']), 1 if direction == 'min' else 0)

    def test_fp64_ties_zero_and_no_observations(self):
        labels = [dict(solver_id=s, objective=v, objective_direction='min', feasible=True)
                  for s, v in zip(('a', 'b', 'c'), (1.00000002, 1.00000001, 1.00000001))]
        result = native_labels(labels, ('a', 'b', 'c'), 'min')
        np.testing.assert_array_equal(result['all_best_mask'], [False, True, True])
        labels[0]['objective'] = 0
        self.assertEqual(native_labels(labels, ('a', 'b', 'c'), 'min')['native_objective'][0], 0.)
        result = native_labels([], ('a',), 'max')
        self.assertEqual(int(result['canonical_winner_index']), -1)
        self.assertFalse(result['all_best_mask'].any())
        self.assertTrue(np.isnan(result['native_objective']).all())

    def test_invalid_deployment_or_direction_is_not_silently_merged(self):
        label = dict(solver_id='a', objective=1, objective_direction='max', feasible=True)
        with self.assertRaises(ValueError):
            native_labels([label, label], ('a',), 'max')
        with self.assertRaises(ValueError):
            native_labels([label], ('a',), 'min')
        with self.assertRaises(ValueError):
            native_labels([label], ('b',), 'max')

    def test_export_rejects_broadcasting_or_incomplete_tie_masks(self):
        labels = native_labels([dict(solver_id='a', objective=3., objective_direction='min', feasible=True),
                                dict(solver_id='b', objective=3., objective_direction='min', feasible=True)],
                               ('a', 'b'), 'min')
        invalid = copy.deepcopy(labels)
        invalid['native_objective'] = np.asarray([3.], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, 'shape'):
            _check_labels(invalid, 2, 'min')
        invalid = copy.deepcopy(labels)
        invalid['all_best_mask'][1] = False
        with self.assertRaisesRegex(ValueError, 'all-best'):
            _check_labels(invalid, 2, 'min')


class IdentityTest(unittest.TestCase):
    def test_id_collision_and_bare_index_is_not_join_key(self):
        instance = real_records('MIS', 1)[0][0]['instance']
        registry = IdentityRegistry()
        registry.add(instance)
        changed = copy.deepcopy(instance)
        changed['input_identity'] = 'different'
        with self.assertRaises(ValueError):
            registry.add(changed)
        other = copy.deepcopy(instance)
        other['split'] = 'test'
        other['instance_id'] = other['instance_id'].replace('/train/', '/test/')
        self.assertNotEqual(instance_key(instance), instance_key(other))
        with self.assertRaises(ValueError):
            registry.add(other)
        invalid = dict(instance, split='test')
        with self.assertRaises(ValueError):
            instance_key(invalid)

    def test_base_group_cannot_cross_splits(self):
        first = copy.deepcopy(real_records('JSSP', 1)[0][0]['instance'])
        first['base_group_id'] = 'shared-base'
        second = copy.deepcopy(first)
        second.update(input_identity='different', split='test')
        second['instance_id'] = second['instance_id'].replace('/train/', '/test/')
        registry = IdentityRegistry()
        registry.add(first)
        with self.assertRaises(ValueError):
            registry.add(second)


class RealTypedSpineTest(unittest.TestCase):
    def test_all_six_real_types_and_label_isolation(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                samples = real_samples(problem)
                batch = collate_nonrouting(samples)
                self.assertTrue(set(INPUT_FIELDS[problem]) <= set(batch['inputs']))
                self.assertEqual(batch['labels']['native_objective'].dtype, torch.float64)
                self.assertEqual(batch['labels']['objective_direction'], DIRECTIONS[problem])
                self.assertTrue(all(isinstance(value, torch.Tensor) for value in batch['inputs'].values()))
                for forbidden in ('labels', 'objective', 'native_objective', 'solution',
                                  'all_best_mask', 'observed_mask', 'input_identity', 'candidate_solver_ids'):
                    self.assertNotIn(forbidden, batch['inputs'])
                self.assertEqual(batch['metadata'][0]['input_identity'], samples[0]['metadata']['input_identity'])
                poisoned = copy.deepcopy(samples)
                poisoned[0]['labels']['native_objective'][:] = 999999
                clean_inputs = collate_nonrouting(poisoned)['inputs']
                for key in batch['inputs']:
                    self.assertTrue(torch.equal(batch['inputs'][key], clean_inputs[key]))

    def test_raw_native_values_and_padding(self):
        for problem in PROBLEMS:
            record = real_records(problem, 1)[0][0]
            data, inputs = record['instance']['data'], typed_inputs(record['instance'])
            if problem == 'MIS':
                np.testing.assert_array_equal(inputs['edge_index'].T, data['edges'])
            elif problem in ('JSSP', 'FJSP'):
                self.assertEqual(len(inputs['operation_job']), sum(map(len, data['jobs'])))
                self.assertEqual(inputs['precedence_edge_index'].shape[1],
                                 len(inputs['operation_job']) - len(data['jobs']))
                if problem == 'FJSP':
                    self.assertTrue((inputs['processing_time'][~inputs['compatibility']] == -1).all())
                    self.assertTrue((inputs['processing_time'][inputs['compatibility']] > 0).all())
                else:
                    np.testing.assert_array_equal(inputs['duration'],
                                                  [op[0][1] for job in data['jobs'] for op in job])
                    np.testing.assert_array_equal(inputs['machine'],
                                                  [op[0][0] for job in data['jobs'] for op in job])
            elif problem == 'BPP':
                np.testing.assert_array_equal(inputs['sizes'], data['sizes'])
            elif problem == 'MKP':
                np.testing.assert_array_equal(inputs['resources'].T, data['weights'])
            else:
                np.testing.assert_array_equal(inputs['d'], data['due_dates'])
        batch = collate_nonrouting(real_samples('FJSP'))
        valid = batch['inputs']['operation_mask'][:, :, None] & batch['inputs']['machine_mask'][:, None, :]
        self.assertFalse(batch['inputs']['compatibility'][~valid].any())
        self.assertTrue((batch['inputs']['processing_time'][~valid] == -1).all())
        with self.assertRaises(ValueError):
            collate_nonrouting([real_samples('MIS')[0], real_samples('BPP')[0]])

    def test_real_six_type_ragged_roundtrip_and_missing_tail(self):
        scratch = PREPARED_ROOT / '_cpu_checks'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='raw_spine_', dir=scratch) as temporary:
            root, entries = Path(temporary), {}
            for problem in PROBLEMS:
                samples = real_samples(problem)
                split = samples[0]['metadata']['split']
                ids = samples[0]['metadata']['candidate_solver_ids']
                descriptor = write_prepared_split(root / problem, iter(samples), count=len(samples),
                                                  problem=problem, split=split, candidate_solver_ids=ids)
                entries[problem] = dict(status='ready' if problem in READY_PROBLEMS else 'incomplete',
                                        ready_for_selector_training=problem in READY_PROBLEMS,
                                        candidate_solver_ids=ids, splits={split: descriptor})
            (root / 'manifest.json').write_text(json.dumps(dict(schema_version=SCHEMA_VERSION,
                                                               nonrouting=entries)))
            for problem in PROBLEMS:
                samples = real_samples(problem)
                split = samples[0]['metadata']['split']
                if problem not in READY_PROBLEMS:
                    with self.assertRaises(ValueError):
                        NonRoutingDataset(root, problem, split)
                dataset = NonRoutingDataset(root, problem, split, allow_incomplete=True)
                self.assertEqual(len(dataset), 2)
                for row, expected in enumerate(samples):
                    actual = dataset[row]
                    for key in expected['inputs']:
                        np.testing.assert_array_equal(actual['inputs'][key], expected['inputs'][key])
                    np.testing.assert_array_equal(actual['labels']['native_objective'],
                                                  expected['labels']['native_objective'])
                collate_nonrouting([dataset[0], dataset[-1]])
                restored = pickle.loads(pickle.dumps(dataset))
                self.assertTrue(isinstance(restored._raw[next(iter(restored._raw))], np.memmap))
                np.testing.assert_array_equal(restored[-1]['labels']['native_objective'],
                                              dataset[-1]['labels']['native_objective'])
                restored.close()
                with self.assertRaises(IndexError):
                    dataset[2]
                dataset.close()
            samples = real_samples('MIS')
            with self.assertRaisesRegex(ValueError, 'Missing tail'):
                write_prepared_split(root / 'truncated', samples, count=3, problem='MIS', split='train',
                                     candidate_solver_ids=samples[0]['metadata']['candidate_solver_ids'])
            with self.assertRaises(FileExistsError):
                write_prepared_split(root / 'MIS', samples, count=2, problem='MIS', split='train',
                                     candidate_solver_ids=samples[0]['metadata']['candidate_solver_ids'])


@unittest.skipUnless((PREPARED_ROOT / 'manifest.json').is_file(), 'Ready export not built yet')
class PreparedReadyCoverageTest(unittest.TestCase):
    def test_all_ready_splits_masks_and_real_tail_objectives(self):
        manifest = load_manifest()
        self.assertEqual(manifest['exported_inputs'], 37200)
        self.assertEqual(manifest['exported_labels'], 234320)
        self.assertFalse(manifest['all_six_training_pools_ready'])
        for problem, columns, pilot in (('MIS', 3, 360), ('JSSP', 6, 80), ('FJSP', 10, 160)):
            entry = manifest['nonrouting'][problem]
            self.assertEqual(len(entry['candidate_solver_ids']), columns)
            for split, count in (('train', 10000), ('val', 1000), ('test', 1000),
                                 ('pilot_dev', pilot), ('pilot_val', pilot)):
                dataset = NonRoutingDataset(PREPARED_ROOT, problem, split)
                self.assertEqual(len(dataset), count)
                self.assertEqual(dataset[-1]['metadata']['global_index'], count - 1)
                expected_id = f'nonrouting_v2_20261025/{problem}/{split}/{count - 1:06d}'
                self.assertEqual(dataset[-1]['metadata']['instance_id'], expected_id)
                objectives = dataset.label_arrays['native_objective']
                observed, best = dataset.label_arrays['observed_mask'], dataset.label_arrays['all_best_mask']
                self.assertEqual(objectives.dtype, np.float64)
                self.assertTrue(observed.all())
                optimum = (np.min if DIRECTIONS[problem] == 'min' else np.max)(objectives, axis=1)
                np.testing.assert_array_equal(best, objectives == optimum[:, None])
                np.testing.assert_array_equal(dataset.label_arrays['canonical_winner_index'],
                                              np.argmax(best, axis=1))
                if split == 'train':
                    tail = dataset[-1]
                    for column, reference in enumerate(tail['metadata']['source_provenance']['labels']):
                        file_id, line = reference
                        with Path(manifest['source_files'][file_id]['path']).open() as stream:
                            source = json.loads(next(itertools.islice(stream, line - 1, line)))
                        if 'labels' in source:
                            source = next(label for label in source['labels']
                                          if label['solver_id'] == dataset.candidate_solver_ids[column])
                        self.assertEqual(tail['labels']['native_objective'][column], source['objective'])
                collate_nonrouting([dataset[0], dataset[-1]])
                dataset.close()

    def test_registry_pending_and_reference_only_routing(self):
        manifest = load_manifest()
        datasets = load_nonrouting_datasets(splits=('val',))
        self.assertEqual(set(datasets), set(READY_PROBLEMS))
        self.assertTrue(manifest['routing']['reference_only'])
        self.assertEqual(len(manifest['routing']['problems']), 18)
        for problem in ('BPP', 'MKP', 'SMTWTP'):
            self.assertEqual(manifest['nonrouting'][problem]['status'], 'pending')
            self.assertEqual(manifest['nonrouting'][problem]['splits'], {})
            with self.assertRaisesRegex(ValueError, 'pending'):
                NonRoutingDataset(PREPARED_ROOT, problem, 'train', allow_incomplete=True)
        for task in datasets.values():
            for dataset in task.values():
                dataset.close()


if __name__ == '__main__':
    unittest.main()
