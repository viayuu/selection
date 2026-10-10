"""Small CPU fixtures for pilot alignment, exact equivalence and actual args."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .r58_throughput import (compare_equivalence, diffusion_witness, merge_outputs,
                             override_diffusion, preflight_gate, premature_worker_exit, stratified_indices)


class ThroughputTests(unittest.TestCase):
    def merged(self, rows, expected=(9, 2)):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, group in enumerate(rows):
                path = Path(directory) / f'worker_{i}.jsonl'
                path.write_text(''.join(json.dumps(row) + '\n' for row in group))
                paths.append(path)
            return merge_outputs(expected, paths)

    def row(self, index, cost=1., route=None):
        return dict(index=index, cost=cost, route=route or [0, 1, 2, 0], status='ok')

    def test_alignment_by_original_index_not_completion_order(self):
        merged = self.merged([[self.row(9)], [self.row(2)]])
        self.assertTrue(merged['complete'])
        self.assertEqual([r['index'] for r in merged['rows']], [2, 9])

    def test_errors_missing_duplicate_and_unexpected_are_not_success(self):
        failed = self.merged([[dict(index=9, status='error', error='OOM', error_type='OutOfMemoryError')]])
        self.assertFalse(failed['complete'])
        self.assertEqual(failed['missing'], [2])
        self.assertEqual(failed['successful'], 0)
        duplicate = self.merged([[self.row(9), self.row(2)], [self.row(9), self.row(88)]])
        self.assertFalse(duplicate['complete'])
        self.assertEqual(duplicate['successful'], 1)

    def test_missing_file_and_truncated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'truncated.jsonl'
            path.write_text('{"index":')
            merged = merge_outputs([9], [path, Path(directory) / 'absent'])
            self.assertFalse(merged['complete'])
            self.assertEqual(merged['missing'], [9])
            self.assertEqual(len(merged['errors']), 2)

    def test_exact_cost_criterion_and_documented_trailing_padding_only(self):
        serial = self.merged([[self.row(9), self.row(2)]])
        padded = self.merged([[self.row(2)], [self.row(9, route=[0, 1, 2, 0, 0])]])
        result = compare_equivalence(serial, padded)
        self.assertTrue(result['equivalent'])
        self.assertEqual(result['raw_tour_differences'], 1)
        changed = self.merged([[self.row(9, cost=1. + 1e-12), self.row(2)]])
        self.assertFalse(compare_equivalence(serial, changed)['equivalent'])
        internal = self.merged([[self.row(9, route=[0, 1, 0, 2, 0]), self.row(2)]])
        self.assertEqual(compare_equivalence(serial, internal)['physical_tour_differences'], 1)

    def test_partial_matching_rows_are_not_equivalent(self):
        serial = self.merged([[self.row(9), self.row(2)]])
        partial = self.merged([[self.row(9)]])
        self.assertFalse(compare_equivalence(serial, partial)['equivalent'])

    def test_normal_early_worker_exit_is_not_a_barrier_failure(self):
        states = [dict(status='complete'), dict(status='ready')]
        self.assertFalse(premature_worker_exit(states, [0, None], 'complete'))
        self.assertTrue(premature_worker_exit(states, [1, None], 'complete'))
        self.assertTrue(premature_worker_exit(states, [0, 0], 'complete'))

    def test_budget_skip_preserves_completed_preflight_but_running_does_not(self):
        from . import r58_throughput as pilot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = dict(preflight_status='preflight_complete', preflight_finished_unix=20,
                preflight_pid_absent_on_gpu03=True, preflight_tmux_handle_absent=True, preflight_pid=123)
            state = dict(stage='labels', status='skipped_over_budget', finished=30,
                budget_coverage=dict(current_passing_deployments=128, expected_deployments=128,
                                     all_current_passing=True, complete_budget=True))
            (root / 'gate_evidence.json').write_text(json.dumps(proof))
            (root / 'selection.json').write_text(json.dumps(dict(frozen_source_sha256={'fixture': 'same'})))
            (root / 'pipeline_state.json').write_text(json.dumps(state))
            with patch.object(pilot, 'ROOT', root), patch.object(pilot, 'OUTPUT', root), \
                    patch('code.V4.r58_scenario.file_hash', return_value='same'), \
                    patch.object(pilot.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
                self.assertEqual(preflight_gate()['status'], 'preflight_complete')
                state['status'] = 'running'
                (root / 'pipeline_state.json').write_text(json.dumps(state))
                with self.assertRaisesRegex(RuntimeError, 'No completed preflight evidence'):
                    preflight_gate()

    def test_deterministic_size_selection_and_disjoint_warmup(self):
        sizes = [50] * 20 + [75] * 20 + [100] * 20
        selected = stratified_indices(sizes, 8)
        self.assertEqual(selected, stratified_indices(sizes, 8))
        self.assertEqual(selected['scales'], [50, 75, 100])
        self.assertEqual(len(selected['indices']), 24)
        self.assertFalse(set(selected['indices']) & set(selected['warmup_indices']))

    def backend(self, is_t2t):
        parameters = dict(inference_diffusion_steps=50, two_opt_iterations=100,
            rewrite_steps=3, inference_steps=10, rewrite_ratio=.25 if is_t2t else .4,
            rewrite=is_t2t, parallel_sampling=1, sequential_sampling=1)
        model = SimpleNamespace(args=SimpleNamespace(**parameters),
            categorical_denoise_step=lambda: None, guided_categorical_denoise_step=lambda: None)
        module = SimpleNamespace(batched_two_opt_torch=lambda **kwargs: None)
        return SimpleNamespace(parameters=parameters, model=model, module=module, is_t2t=is_t2t)

    def test_overrides_both_parameter_copies_and_preserves_ratio(self):
        for is_t2t in [False, True]:
            backend = self.backend(is_t2t)
            ratio = backend.parameters['rewrite_ratio']
            override_diffusion(backend)
            self.assertEqual(backend.parameters['inference_diffusion_steps'], 20)
            self.assertEqual(backend.model.args.inference_diffusion_steps, 20)
            self.assertEqual(backend.model.args.rewrite_steps, int(is_t2t))
            self.assertEqual(backend.model.args.inference_steps, 10)
            self.assertEqual(backend.model.args.rewrite_ratio, ratio)

    def test_actual_budget_witness_and_restore_on_failure(self):
        backend = self.backend(True)
        original = backend.model.categorical_denoise_step
        with diffusion_witness(backend) as witness:
            for _ in range(20):
                backend.model.categorical_denoise_step()
            for _ in range(10):
                backend.model.guided_categorical_denoise_step()
            for _ in range(2):
                backend.module.batched_two_opt_torch(max_iterations=100)
        self.assertEqual(witness['guided_calls'], 10)
        self.assertIs(backend.model.categorical_denoise_step, original)
        with self.assertRaisesRegex(ValueError, 'Actual solver budget differs'):
            with diffusion_witness(backend):
                backend.model.categorical_denoise_step()
        self.assertIs(backend.model.categorical_denoise_step, original)


if __name__ == '__main__':
    unittest.main()
