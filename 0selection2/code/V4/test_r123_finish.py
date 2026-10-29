"""Small CPU-only fixtures; no real model, dataset, TEST or GPU is opened."""

import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import r123_finish as finish


def fixture():
    actions = np.array([[1, 1, 2], [1, 2, 1], [0, 1, 2], [0, 0, 0]], dtype=np.int64)
    scores = np.full((4, 3, 3), -1., dtype=np.float32)
    np.put_along_axis(scores, actions[..., None], 1., axis=2)
    chosen = np.array([0, 0, 1, 0], dtype=np.int64)
    data = dict(costs=np.array([[1., 1., 3.], [1., 2., 4.], [1., 2., 5.], [1., 3., 4.]], dtype=np.float64),
        winner=np.zeros(4, dtype=np.int64), pred=actions[np.arange(4), chosen],
        logits=scores[np.arange(4), chosen], indices=np.array([9, 21, 35, 40], dtype=np.int64),
        release_source_indices=np.array([100, 102, 107, 108]),
        source_identity=np.array(['TSP:sha:100', 'TSP:sha:102', 'TSP:sha:107', 'TSP:sha:108'], dtype='S100'),
        pool=np.array(['A', 'B', 'C']), pool_ids=np.array([0, 2, 6], dtype=np.int64), chosen_expert=chosen)
    bank = SimpleNamespace(logits=scores, row_indices=data['indices'],
        source_indices=data['release_source_indices'], source_identity=data['source_identity'])
    return data, actions, bank


def test_native_tie_and_two_nonduplicated_error_budgets():
    data, actions, _ = fixture()
    assert finish.validate_prediction(data) == 4
    result = finish.bank_decomposition(data, actions)
    assert result['native_error_pct'] == 75.
    assert result['native_bank_miss_pp'] == 50.
    assert result['native_router_miss_pp'] == 25.
    assert result['bank_quality_floor_pct'] == 25.
    assert result['routing_quality_excess_pct'] == 25.
    assert result['actual_regret_pct'] == 50.
    assert result['native_error_zero_quality_count'] == 1
    baseline = dict(selection_split='train', sbs_solver='A')
    metrics = finish.metric_row(data, baseline, problem='TSP', split='test')
    assert metrics['mean_cost'] == 1.5
    assert metrics['oracle_cost'] == metrics['sbs_cost'] == 1.
    assert metrics['actual_regret_pct'] == metrics['vs_oracle_pct'] == 50.
    unequal = dict(costs=np.array([[1., 2.], [100., 200.]], dtype=np.float64),
        winner=np.zeros(2, dtype=np.int64), pred=np.array([1, 0]), pool=np.array(['A', 'B']))
    metrics = finish.metric_row(unequal, baseline, problem='TSP', split='val')
    assert metrics['actual_regret_pct'] == 50.
    np.testing.assert_allclose(metrics['vs_oracle_pct'], (51. / 50.5 - 1.) * 100.)


def test_fixed_solver_id_ties_and_identity_join_not_local_index_guess():
    assert finish.hard_actions(np.array([[1., 1., 0.]]), np.array([8, 2, 9])).tolist() == [1]
    data, actions, bank = fixture()
    np.testing.assert_array_equal(finish.check_bank_join(data, bank), actions)
    changed = dict(data, release_source_indices=data['release_source_indices'][::-1].copy())
    with unittest.TestCase().assertRaisesRegex(ValueError, 'identity mismatch'):
        finish.check_bank_join(changed, bank)
    changed = dict(data, source_identity=np.repeat(data['source_identity'][:1], 4))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'identity mismatch|Duplicate'):
        finish.check_bank_join(changed, bank)
    changed = dict(data, logits=data['logits'] + np.float32(.25))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'exact selected-expert'):
        finish.check_bank_join(changed, bank)


def test_train_only_priors_equal_task_weight_and_fixed_ties():
    values = {'A': dict(split='train', actions=np.tile([0, 1, 1], (100, 1)), winner=np.zeros(100)),
              'B': dict(split='train', actions=np.array([[1, 0, 1]]), winner=np.zeros(1))}
    refs = finish.fit_train_references(values)
    assert refs['global_fixed_expert'] == 0
    assert refs['task_prior_expert'] == {'A': 0, 'B': 1}
    assert refs['equal_task_correctness_prior'] == [.5, .5, 0.]
    for split in ('val', 'test'):
        with unittest.TestCase().assertRaisesRegex(ValueError, 'No VAL/TEST'):
            finish.fit_train_references({'A': dict(values['A'], split=split)})


def test_zero_rows_all_retained_and_missed_cost_charged_once():
    data, _, _ = fixture()
    rows = finish.method_rows(data, problem='TSP', split='train')
    assert len(rows) == 3
    assert rows[2]['selected_count'] == rows[2]['true_winner_count'] == 0
    assert not rows[2]['zero_selected_with_winner']
    assert rows[0]['missed_winner_count'] == 3
    assert sum(r['missed_actual_regret_contribution_pp'] for r in rows) == 50.
    many = dict(data, winner=np.zeros(1000, dtype=np.int64),
                pred=np.r_[0, np.ones(999, dtype=np.int64)], costs=np.tile([1., 2., 3.], (1000, 1)))
    tiny = finish.method_rows(many, problem='TSP', split='test')
    assert tiny[0]['near_zero_selected_with_winner']
    assert not tiny[0]['zero_selected_with_winner']


def test_all18_is_equal_task_macro_and_requires_complete_problem_set():
    rows = [dict(problem=p, n=i + 1, value=float(i)) for i, p in enumerate(finish.PROBLEMS)]
    macro = finish.macro_row(rows, ['value'])
    assert macro['value'] == (len(rows) - 1.) / 2.
    assert macro['n'] == sum(r['n'] for r in rows)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'all18'):
        finish.macro_row(rows[:-1], ['value'])


def test_live_dispatch_is_one_expert_per_query_and_label_free():
    import torch

    class Expert(torch.nn.Module):
        def forward(self, batch):
            return dict(logits=torch.zeros(len(batch['n']), 3))

    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([Expert() for _ in range(3)])

        def forward(self, batch):
            assert not {'costs', 'winner', 'ind', 'performance_target'}.intersection(batch)
            chosen = batch['n'] % 3
            scores = torch.empty(len(chosen), 3)
            for i, expert in enumerate(self.experts):
                rows = chosen == i
                if rows.any():
                    scores[rows] = expert({'n': batch['n'][rows]})['logits']
            return dict(logits=scores, chosen_expert=chosen, reliability=torch.zeros(len(chosen), 3))

    batch = dict(n=torch.tensor([3, 4, 5, 6]), pool_ids=torch.tensor([0, 2, 6]),
        costs=torch.ones(4, 3), winner=torch.zeros(4), ind=torch.zeros(4), performance_target=torch.ones(4))
    output, result = finish.live_forward(Router(), [batch], 'cpu')
    assert result['expert_executed_rows'] == [2, 1, 1]
    assert result['total_expert_executed_rows'] == result['n'] == 4
    assert output['chosen_expert'].tolist() == [0, 1, 2, 0]


def test_describe_is_no_results_no_cuda_and_no_lock():
    output = StringIO()
    with tempfile.TemporaryDirectory() as directory, redirect_stdout(output):
        with patch.object(finish, 'locked', side_effect=AssertionError('must not read a real run')):
            result = finish.main(['describe', '--root', directory])
        assert result['run'] == str(Path(directory) / 'model')
    assert not result['test_results_read']
    assert not result['cuda_initialized']
    assert 'metaTRAIN_per_task' in output.getvalue()


def test_live_full_router_retains_pool_problem_and_masks_without_labels():
    import torch
    from .r123_model import HistoricalModelRouter

    forbidden = {'costs', 'winner', 'ind', 'performance_target', 'labels', 'expert_logits'}

    class Gate(torch.nn.Module):
        expert_ids = finish.EXPERTS

        def forward(self, batch):
            assert not forbidden.intersection(batch)
            logits = torch.zeros(len(batch['n']), 3)
            logits[:, 0] = 1.
            return dict(reliability_logits=logits, reliability=logits.sigmoid())

    class Expert(torch.nn.Module):
        def forward(self, batch):
            assert not forbidden.intersection(batch)
            assert batch['pool_ids'].tolist() == [0, 2, 6]
            assert batch['problem_id'] == 0
            assert batch['solver_mask'].shape == batch['mask'].shape == (2, 3)
            assert batch['node_mask'].shape == (2, 4)
            return dict(logits=torch.zeros(len(batch['n']), 3))

    router = HistoricalModelRouter(Gate(), [Expert() for _ in finish.EXPERTS], {'TSP': [0, 2, 6]})
    batch = dict(n=torch.tensor([3, 4]), problem_id=0, pool_ids=torch.tensor([0, 2, 6]),
        node_mask=torch.ones(2, 4, dtype=torch.bool), solver_mask=torch.ones(2, 3, dtype=torch.bool),
        mask=torch.ones(2, 3, dtype=torch.bool), **{key: torch.ones(2, 3) for key in forbidden})
    output, counts = finish.live_forward(router, [batch], 'cpu', warmup=False)
    assert output['logits'].shape == (2, 3)
    assert counts['expert_executed_rows'] == [2, 0, 0]


def test_lib_str_directory_metadata_and_same_lock_capture_reuse():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for name in ('dataset.pkl', 'raw_label.pkl'):
            (directory / name).write_bytes(b'CPU metadata fixture only')
        data, _, _ = fixture()
        data = dict(data, indices=np.arange(4, dtype=np.int64))
        raw = dict(costs=data['costs'], winner=data['winner'], pool=data['pool'],
                   pool_ids=data['pool_ids'], directory=str(directory))
        sources = finish.lib_sources(raw)
        assert sources == finish.lib_sources(dict(raw, directory=directory))
        assert set(sources) == {str(directory / name) for name in ('dataset.pkl', 'raw_label.pkl')}
        path = directory / 'TSPLIB.npz'
        finish.save_npz(path, data)
        digest, modified = finish.file_hash(path), path.stat().st_mtime_ns
        model = dict(lock=dict(checkpoint_sha256='c' * 64, protocol_sha256='d' * 64, selected_update=720))
        with unittest.TestCase().assertRaisesRegex(ValueError, 'same-lock reuse requires'):
            finish.lib_capture_receipt(path, raw, model)
        receipt = finish.lib_capture_receipt(str(path), raw, model, allow_legacy=True)
        assert receipt['timing']['forward_seconds'] is None
        assert receipt['sources'] == sources
        sidecar = directory / 'TSPLIB_receipt.json'
        finish.write_json(sidecar, receipt)
        assert finish.require_identity(sidecar, model, 'live_sparse_LIB')['dataset'] == 'TSPLIB'
        assert finish.file_hash(path) == digest and path.stat().st_mtime_ns == modified


def test_historical_nss_saved_schema_alignment_preserves_original_ties():
    current, _, _ = fixture()
    rows, columns = np.array([2, 0, 3, 1]), np.array([2, 0, 1])
    inverse = np.argsort(columns)
    # Actual four-archive schema: scores and saved pred, no public pool_ids.
    other = dict(scores=np.zeros((4, 3), dtype=np.float32),
        pred=inverse[current['pred'][rows]], costs=current['costs'][rows][:, columns],
        winner=inverse[current['winner'][rows]], indices=current['indices'][rows],
        source_indices=current['release_source_indices'][rows], pool=current['pool'][columns])
    original = {key: value.copy() for key, value in other.items()}
    aligned = finish.align_nss_prediction(other, current)
    for name in ('costs', 'winner', 'indices', 'pool', 'pool_ids', 'pred'):
        np.testing.assert_array_equal(aligned[name], current[name])
    np.testing.assert_array_equal(aligned['scores'], np.zeros((4, 3), dtype=np.float32))
    np.testing.assert_array_equal(aligned['source_indices'], current['release_source_indices'])
    assert not np.array_equal(aligned['pred'], finish.hard_actions(aligned['scores'], aligned['pool_ids']))
    for key in other:
        np.testing.assert_array_equal(other[key], original[key])
    changed = dict(other, costs=other['costs'].copy())
    changed['costs'][0, 0] += 1.
    with unittest.TestCase().assertRaisesRegex(ValueError, 'aligned native FP64'):
        finish.align_nss_prediction(changed, current)


def test_portable_distribution_pdf_fixture():
    if not finish.FONT.is_file():
        raise unittest.SkipTest('Installed CJK font unavailable on this test host')
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt
    data, _, _ = fixture()
    finish.set_style()
    model = dict(lock=dict(checkpoint_sha256='0' * 64, selected_update=0),
                 config=dict(data_root='/CPU_FIXTURE_ONLY_NOT_TRAINED_RESULTS'))
    fig = finish.distribution_figure('TSP', [('TEST', data), ('TSPLIB', data)], model)
    fig.text(.02, .99, 'CPU FIXTURE ONLY - NOT REAL MODEL RESULTS', color='red', weight='bold')
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(os.environ.get('R123_FINISH_FIXTURE_DIR', temporary))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'learner_finish_CPU_FIXTURE_ONLY.pdf'
        with PdfPages(path) as pdf:
            pdf.savefig(fig)
        fig.savefig(path.with_suffix('.png'), dpi=80)
        assert path.read_bytes().startswith(b'%PDF-')
        assert path.stat().st_size > 1000
    plt.close(fig)


def load_tests(_loader, _tests, _pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith('test_') and callable(value))


if __name__ == '__main__':
    unittest.main()
