"""Small real integer/certificate and immutable-schema checks; no GPU work."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import nonrouting_supplement as s


class GenerationChecks(unittest.TestCase):
    def test_signed_interval_before_clip(self):
        for total in range(1, 31):
            for tf, rdd in itertools.product(range(1, 6), repeat=2):
                low, high = s.signed_due_bounds(total, tf, rdd)
                a, b = total * (10 - 2 * tf - rdd), total * (10 - 2 * tf + rdd)
                self.assertTrue(10 * (low - 1) < a <= 10 * low)
                self.assertTrue(10 * high <= b < 10 * (high + 1))
        zeros = 0
        for index in range(75):
            instance, witness = s._make_instance('SMTWTP', 'witness', index)
            self.assertEqual(instance, s.make_instance('SMTWTP', 'witness', index))
            data = instance['data']
            self.assertTrue(all(1 <= x <= 100 for x in data['processing']))
            self.assertTrue(all(1 <= x <= 10 for x in data['weights']))
            self.assertEqual(data['due_dates'], [max(0, x) for x in witness['signed_due_draws']])
            zeros += sum(x < 0 for x in witness['signed_due_draws'])
        self.assertGreater(zeros, 0)
        self.assertEqual({s.make_instance('SMTWTP', 'witness', i)['cell_index'] for i in range(75)}, set(range(75)))

    def test_mkp_integer_grid(self):
        seen = set()
        for index in range(27):
            instance, witness = s._make_instance('MKP', 'witness', index)
            seen.add(instance['cell_index'])
            data, p = instance['data'], instance['parameters']
            weights = np.asarray(data['weights'], dtype=np.int64)
            self.assertTrue(np.all((weights >= 1) & (weights <= 999)))
            np.testing.assert_array_equal(data['profits'], weights.sum(0) // p['dimensions'] + witness['profit_noise'])
            np.testing.assert_array_equal(data['capacities'], p['tightness_numerator'] * weights.sum(1) // 4)
            self.assertTrue(np.all(weights <= np.asarray(data['capacities'])[:, None]))
            self.assertEqual(instance['generation']['accepted_attempt'], 0)
        self.assertEqual(seen, set(range(27)))

    def test_independent_splits_and_bpp_witness(self):
        identities = set()
        for problem in s.COUNTS:
            for split in ('val', 'test'):
                for index in range(len(s.cells(problem))):
                    instance, witness = s._make_instance(problem, split, index)
                    self.assertNotIn(instance['input_identity'], identities)
                    identities.add(instance['input_identity'])
                    self.assertTrue(instance['source_id'].startswith('synthetic://'))
                    if problem == 'BPP' and instance['data']['capacity'] == 1000:
                        optimum = s.validate_solution(instance, {'bins': witness['hidden_triplet_bins']})
                        self.assertEqual(optimum, instance['parameters']['n'] // 3)
                        self.assertTrue(all(4 * x > 1000 for x in instance['data']['sizes']))
        with self.assertRaisesRegex(ValueError, 'read-only'):
            s.make_instance('BPP', 'train', 0)


class CertificateChecks(unittest.TestCase):
    def test_mbs_full_one_bin_not_greedy(self):
        rng = np.random.default_rng(119)
        for _ in range(100):
            capacity = int(rng.integers(9, 31))
            n = int(rng.integers(2, 11))
            sizes = rng.integers(1, capacity + 1, size=n).tolist()
            remaining = sorted(range(n), key=lambda j: (-sizes[j], j))
            group, nodes = s.mbs_one_bin(sizes, remaining, capacity)
            seed = remaining[0]
            oracle_load = max(sum(sizes[j] for j in [seed, *subset])
                for r in range(n) for subset in itertools.combinations(remaining[1:], r)
                if sum(sizes[j] for j in [seed, *subset]) <= capacity)
            self.assertEqual(sum(sizes[j] for j in group), oracle_load)
            self.assertEqual(group[0], seed)
            self.assertGreaterEqual(nodes, 1)
        solution, _ = s.mbs_prime(dict(sizes=[5, 5, 5, 3, 3, 3], capacity=9))
        self.assertEqual(len(solution['bins']), 3)

    def test_exact_legacy_functions_and_objectives(self):
        functions = s.legacy_functions()
        for problem in s.COUNTS:
            instance = s.make_instance(problem, 'witness', 0)
            for name in s.CLASSIC[problem]:
                label = s.solve_instance(instance, name)
                if problem == 'SMTWTP':
                    expected = functions['single_machine'](instance['data'], name)
                elif problem == 'BPP':
                    expected = functions['bin_packing'](instance['data'], name == 'BFD')
                else:
                    expected = functions['surrogate_greedy'](instance['data'])
                self.assertEqual(label['solution'], expected)
                self.assertEqual(label['objective'], s.validate_solution(instance, expected))
                self.assertGreater(label['runtime_s'], 0)
        invalid = s.make_instance('SMTWTP', 'witness', 0)
        with self.assertRaises(ValueError):
            s.validate_solution(invalid, {'order': [0] * len(invalid['data']['processing'])})

    def test_real_lp_certificate_and_dual(self):
        instance = s.make_instance('MKP', 'witness', 0)
        label = s.solve_instance(instance, s.LP_ID)
        self.assertEqual(label['objective'], s.validate_solution(instance, label['solution']))
        extra = label['execution_details']
        self.assertEqual(extra['lp_status'], 0)
        self.assertTrue(all(x >= 0 for x in extra['resource_shadow_prices']))
        self.assertLessEqual(label['objective'], extra['lp_objective'] + 1e-6)
        self.assertAlmostEqual(extra['bounded_lp_dual_objective'], extra['lp_objective'], places=5)

    def test_native_as_real_cpu_without_gpu(self):
        import torch
        initialized = torch.cuda.is_initialized()
        index = next(i for i in range(27) if s.make_instance('MKP', 'witness', i)['cell_index'] == 0)
        instance = s.make_instance('MKP', 'witness', index)
        solution, extra = s.native_ant_system(instance, 'cpu', iterations=2)
        self.assertEqual(s.validate_solution(instance, solution), extra['native_objective'])
        self.assertEqual(extra['n_ants'], 20)
        self.assertFalse(extra['production_budget'])
        self.assertEqual(torch.cuda.is_initialized(), initialized)
        repeat, _ = s.native_ant_system(instance, 'cpu', iterations=2)
        self.assertEqual(solution, repeat)
        for cell in (1, 2, 26):
            index = next(i for i in range(27) if s.make_instance('MKP', 'witness', i)['cell_index'] == cell)
            instance = s.make_instance('MKP', 'witness', index)
            solution, extra = s.native_ant_system(instance, 'cpu', iterations=1)
            self.assertEqual(s.validate_solution(instance, solution), extra['native_objective'])
            self.assertGreaterEqual(extra['minimum_integer_slack'], 0)

    def test_benchmark_exclusion_catalogue(self):
        identities = s.excluded_identities()
        self.assertEqual(len(identities), 270 + 375 + 240)
        self.assertNotIn(s.make_instance('SMTWTP', 'train', 0)['input_identity'], identities)
        self.assertNotIn(s.make_instance('MKP', 'train', 0)['input_identity'], identities)


class SchemaChecks(unittest.TestCase):
    def test_zero_objective_ties_and_missing(self):
        instance = s.make_instance('SMTWTP', 'witness', 0)
        instance['data']['due_dates'] = [100000] * len(instance['data']['processing'])
        instance['input_identity'] = s.digest(dict(problem='SMTWTP', data=instance['data']))
        labels = [s.solve_instance(instance, name) for name in s.CLASSIC['SMTWTP']]
        sample = s.as_sample(dict(instance=instance, labels=labels))
        self.assertEqual(set(sample), {'inputs', 'labels', 'metadata'})
        self.assertTrue(sample['labels']['all_best_mask'].all())
        self.assertTrue(np.all(sample['labels']['native_objective'] == 0))
        self.assertEqual(sample['labels']['canonical_winner_index'], 0)
        self.assertEqual(sample['labels']['objective_direction'], 'min')
        self.assertEqual(set(sample['inputs']), {'p', 'w', 'd'})
        mkp = s.make_instance('MKP', 'witness', 0)
        partial = s.as_sample(dict(instance=mkp, labels=[s.solve_instance(mkp, 'SurrogateGreedy')]))
        self.assertEqual(partial['labels']['observed_mask'].sum(), 1)
        self.assertTrue(np.isnan(partial['labels']['native_objective'][~partial['labels']['observed_mask']]).all())
        self.assertEqual(partial['inputs']['resources'].shape, (mkp['parameters']['n'], mkp['parameters']['dimensions']))
        self.assertNotIn('input_identity', partial['inputs'])

    def test_readonly_bpp_view(self):
        path = s.bpp_source_paths()[0]
        before = s.file_sha(path)
        with path.open() as stream:
            source = json.loads(stream.readline())
        instance = source['instance']
        ref = dict(path=str(path), sha256=before, line=1, global_index=0,
                   instance_id=instance['instance_id'], input_identity=instance['input_identity'])
        record = dict(instance_ref=ref, labels=[s.solve_instance(instance, s.MBS_ID)])
        resolved = s.resolve_record(record)
        self.assertEqual(resolved['instance'], instance)
        self.assertEqual(resolved['labels'][:2], source['labels'])
        self.assertEqual(s.resolve_record(resolved), resolved)
        self.assertEqual(len(s.as_sample(record)['metadata']['labels_with_certificates']), 3)
        self.assertEqual(len(s.as_sample(resolved)['metadata']['labels_with_certificates']), 3)
        self.assertEqual(s.file_sha(path), before)

    def test_prepared_schema_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix='unittest_', dir=s.ROOT) as temporary:
            directory = Path(temporary) / 'SMTWTP'
            records = []
            for i in range(2):
                instance = s.make_instance('SMTWTP', 'val', i)
                records.append(dict(instance=instance, labels=[s.solve_instance(instance, n) for n in s.CLASSIC['SMTWTP']]))
            descriptor = s._write_prepared_split(directory, iter(records), 2, 'SMTWTP', 'val', s.solver_ids('SMTWTP'))
            self.assertTrue(descriptor['measured_pool_complete'])
            self.assertEqual(descriptor['schema_version'], 'nonrouting_raw_v1')
            for key, spec in descriptor['input_fields'].items():
                raw = np.memmap(directory / ('input_' + key + '.bin'), dtype=spec['dtype'], mode='r')
                offsets = np.load(directory / ('input_' + key + '_offsets.npy'))
                shapes = np.load(directory / ('input_' + key + '_shapes.npy'))
                actual = raw[offsets[0]:offsets[1]].reshape(tuple(shapes[0]))
                np.testing.assert_array_equal(actual, s.as_sample(records[0])['inputs'][key])
            with (directory / 'metadata.jsonl').open() as stream:
                meta = json.loads(stream.readline())
            self.assertEqual(meta['candidate_solver_ids'], s.solver_ids('SMTWTP'))
            self.assertNotIn('labels_with_certificates', meta)


if __name__ == '__main__':
    unittest.main()
