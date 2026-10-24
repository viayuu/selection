"""Small synthetic CPU checks; no real caches/checkpoints, CUDA or launches."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, P2I, POOLS, PROBLEMS, S2I, constraint_bits, problem_descriptor
from .pairwise_objective import LABEL_KEYS
from .pairwise_selector import maximin_logits
from .r77_data import parameter_hash
from .r88_model import RoleInputSelector
from .r108_model import ARMS, FRESH_REFERENCE_SHA256, ORIGINAL, PARAMETER_COUNTS, PRIOR_BUFFERS, initialize_shared, make_model
from .r108_priors import DEFAULT_DATA_ROOT, build_priors, prepare, validate_priors
from .V4Model import get_default_model_params


def full18_params():
    params = get_default_model_params()
    params.update(architecture='pairwise', joint_layer_num=2, local_geometry=True,
        geometry_mode='real', ignore_coord_dist=True, native_winner=True,
        decision_head='classification', pair_mode='score_difference',
        solver_representation='handcrafted', solver_feature_weight=1., sdpa=True)
    return params


def frozen_pool(problem):
    pool = [s for s in POOLS[problem] if s != 'RELD_MTL']
    if problem not in ('TSP', 'CVRP', 'ATSP') and 'B' in problem:
        pool = [s for s in pool if s not in ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA')]
    return ['reld_mt' if s == 'RELD_MOEL' else s for s in pool]


def synthetic_cache(root):
    """Only tiny generated TRAIN files, not research data or source datasets."""
    names = ['reld_mt' if s == 'RELD_MOEL' else s for s in GLOBAL_SOLVERS if s != 'RELD_MTL']
    ids = {s: S2I['RELD_MOEL' if s == 'reld_mt' else s] for s in names}
    for problem in PROBLEMS:
        directory = root / 'train' / problem
        directory.mkdir(parents=True)
        pool = frozen_pool(problem)
        pool_ids = [ids[s] for s in pool]
        record = dict(problem=problem, problem_id=P2I[problem], split='train', rows=8,
            rows_written=8, complete=True, data_root=str(DEFAULT_DATA_ROOT),
            cache_format='r76_scenario_v1', cache_version=1, raw_cost_dtype='float64',
            winner_source='recomputed FP64 winner in filtered candidate pool',
            model_global_solvers=list(GLOBAL_SOLVERS), scenario_global_solvers=names,
            scenario_global_solver_ids=ids, scenario_solver_aliases={'reld_mt': 'RELD_MOEL'},
            pool=pool, pool_ids=pool_ids, mask=[int(i in pool_ids) for i in range(len(GLOBAL_SOLVERS))],
            true_length_min=4, true_length_max=6, scenario_id='synthetic_frozen_TRAIN',
            source_files=[dict(path=str(DEFAULT_DATA_ROOT / 'scenario_manifest.json'), size_bytes=1, mtime_ns=2),
                          dict(path=str(DEFAULT_DATA_ROOT / (problem + 'train') / 'metadata.json'), size_bytes=1, mtime_ns=2)],
            arrays={s: dict(file=s + '.npy', shape=[8], dtype='int64') for s in ('n', 'winner')})
        (directory / 'metadata.json').write_text(json.dumps(record))
        np.save(directory / 'n.npy', np.array([4] * 4 + [6] * 4, dtype=np.int64))
        np.save(directory / 'winner.npy', np.arange(8, dtype=np.int64) % len(pool))


def input_batch(problem):
    generator = torch.Generator().manual_seed(108)
    node = torch.rand(2, 6, 8, generator=generator)
    mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
    node[..., 3:] = 0.
    if problem == 'TSP':
        node = node[..., :2].clone()
    elif problem != 'ATSP':
        node[:, 0, 3] = 1.
    matrix = torch.rand(2, 6, 6, generator=generator)
    matrix.diagonal(dim1=1, dim2=2).zero_()
    ids = [S2I['RELD_MOEL' if s == 'reld_mt' else s] for s in frozen_pool(problem)]
    return dict(problem_id=P2I[problem], pool_ids=torch.tensor(ids),
        kind='matrix' if problem == 'ATSP' else 'coord', node=node.masked_fill(~mask[..., None], 0.),
        matrix=matrix, node_mask=mask, n=mask.sum(-1), coord_dist=torch.full((2,), 4),
        cbits=torch.tensor([constraint_bits(problem)] * 2, dtype=torch.float32),
        problem_desc=torch.tensor([problem_descriptor(problem)] * 2), node_geom=torch.zeros(2, 6, 12))


class R108ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.cache = Path(cls.temporary.name) / 'input_cache'
        synthetic_cache(cls.cache)
        cls.priors = build_priors(cls.cache, expected_rows=8)
        cls.params = full18_params()
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2)
            original = RoleInputSelector('B_execution_anchor', **cls.params)
        trainable = {k: v.detach().clone() for k, v in original.named_parameters()}
        cls.initial = dict(seed=2, initialization='scratch', model_params=cls.params,
            trainable=trainable, trainable_sha256=parameter_hash(trainable))
        cls.geometry = dict(mean=[.21] * 12, std=[1.31] * 12, split='train')
        cls.models = {arm: initialize_shared(cls.initial, cls.geometry, arm, priors=cls.priors)[0] for arm in ARMS}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.threads)

    def setUp(self):
        for model in self.models.values():
            model.eval().zero_grad(set_to_none=True)
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU checks cannot initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_builder_train_only_sequential_smoothing_and_alias(self):
        opened, read_metadata = [], []
        load, read_bytes = np.load, Path.read_bytes
        def observed_load(path, **kwargs):
            opened.append(Path(path).relative_to(self.cache).as_posix())
            return load(path, **kwargs)
        def observed_read(path):
            read_metadata.append(path.relative_to(self.cache).as_posix())
            return read_bytes(path)
        with patch('numpy.load', side_effect=observed_load), patch.object(Path, 'read_bytes', observed_read):
            record = build_priors(self.cache, expected_rows=8)
        self.assertEqual(opened, [f'train/{p}/{key}.npy' for p in PROBLEMS for key in ('n', 'winner')])
        self.assertEqual(read_metadata, [f'train/{p}/metadata.json' for p in PROBLEMS])
        self.assertEqual(record, self.priors)
        validate_priors(record)
        for p, task in record['tasks'].items():
            counts = np.asarray(task['counts_by_n'])
            global_prior = (counts.sum(0) + 1.) / (8 + len(task['pool']))
            expected = (counts + 20 * global_prior) / (counts.sum(1)[:, None] + 20)
            np.testing.assert_array_equal(task['task_prior'], global_prior)
            np.testing.assert_array_equal(task['conditional_prior'], expected)
            if p not in ('TSP', 'CVRP', 'ATSP'):
                self.assertEqual(task['pool_ids'][task['pool'].index('reld_mt')], 16)
                self.assertNotIn(17, task['pool_ids'])
        output = Path(self.temporary.name) / 'run'
        self.assertEqual(prepare(self.cache, output, expected_rows=8), record)
        self.assertEqual(prepare(self.cache, output, expected_rows=8), record)

    def test_builder_rejects_source_pool_and_native_winner_changes(self):
        for field, replacement in [('split', 'val'), ('rows', 9), ('scenario_id', 'another'),
                                   ('scenario_solver_aliases', {'reld_mt': 'RELD_MTL'})]:
            path = self.cache / 'train/VRPB/metadata.json'
            previous = path.read_text()
            try:
                record = json.loads(previous)
                record[field] = replacement
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    build_priors(self.cache, expected_rows=8)
            finally:
                path.write_text(previous)
        path = self.cache / 'train/VRPB/winner.npy'
        previous = np.load(path)
        try:
            np.save(path, np.full(8, 3, dtype=np.int64))
            with self.assertRaises(ValueError):
                build_priors(self.cache, expected_rows=8)
        finally:
            np.save(path, previous)

    def test_exact_fresh_full_state_rng_and_no_factory_io(self):
        self.assertEqual(self.initial['trainable_sha256'], FRESH_REFERENCE_SHA256)
        self.assertEqual(parameter_hash(self.models[ARMS[0]].state_dict()), parameter_hash(self.models[ARMS[1]].state_dict()))
        rng = torch.get_rng_state().clone()
        with patch('builtins.open', side_effect=AssertionError('factory file read')), \
             patch.object(Path, 'open', side_effect=AssertionError('factory path read')):
            model = make_model(self.params, ARMS[0], priors=self.priors)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), PARAMETER_COUNTS['original'])
        self.assertEqual(set(dict(model.named_parameters())), set(self.initial['trainable']))
        for k, v in model.named_parameters():
            torch.testing.assert_close(v, self.initial['trainable'][k], rtol=0, atol=0)
        for k, v in ORIGINAL.items():
            self.assertEqual(model.params[k], v)
        self.assertLess(len(json.dumps(model.params['r108_prior'])), 1200)
        self.assertEqual(set(PRIOR_BUFFERS) - set(model.state_dict()), set())

    @torch.no_grad()
    def test_all18_identical_forward_original_t_maximin_and_query_isolation(self):
        a, b = [self.models[arm] for arm in ARMS]
        original = RoleInputSelector('B_execution_anchor', **self.params).eval()
        original.load_state_dict({k: v for k, v in a.state_dict().items() if k not in PRIOR_BUFFERS}, strict=True)
        for problem in PROBLEMS:
            batch = input_batch(problem)
            output = a(batch)
            self.assertEqual(set(output), {'utility', 'residual_utility', 'conditional_prior', 'pair_margin', 'solver_mask', 'logits'})
            for key, value in b(batch).items():
                torch.testing.assert_close(output[key], value, rtol=0, atol=0)
            torch.testing.assert_close(output['residual_utility'], original(batch)['utility'], rtol=0, atol=0)
            torch.testing.assert_close(output['utility'], output['residual_utility'] + output['conditional_prior'].log(), rtol=0, atol=0)
            torch.testing.assert_close(output['logits'], maximin_logits(output['pair_margin'], output['solver_mask']), rtol=0, atol=0)
            self.assertTrue(torch.equal(output['utility'].argmax(-1), output['logits'].argmax(-1)))
            contaminated = dict(batch, **{k: object() for k in LABEL_KEYS},
                                winner=object(), instance_id=object(), solver_outputs=object())
            for key, value in a(contaminated).items():
                torch.testing.assert_close(output[key], value, rtol=0, atol=0)

    @torch.no_grad()
    def test_unseen_n_fallback_and_solver_node_permutations(self):
        model = self.models[ARMS[0]]
        for problem in ('TSP', 'VRPB', 'ATSP'):
            batch = input_batch(problem)
            batch['node_mask'][0, 4] = True
            batch['n'] = batch['node_mask'].sum(-1)
            output = model(batch)
            task = self.priors['tasks'][problem]
            torch.testing.assert_close(output['conditional_prior'][0], torch.tensor(task['task_prior']), rtol=0, atol=0)
            torch.testing.assert_close(output['conditional_prior'][1], torch.tensor(task['conditional_prior'][1]), rtol=0, atol=0)
            order = torch.arange(len(batch['pool_ids']) - 1, -1, -1)
            permuted = dict(batch, pool_ids=batch['pool_ids'][order])
            torch.testing.assert_close(model(permuted)['utility'], output['utility'][:, order], rtol=1e-5, atol=2e-6)
            batch = model.prepare_role_inputs(batch)
            order = torch.tensor([3, 0, 5, 1, 4, 2])
            permuted = dict(batch)
            for key in ('node', 'node_mask', 'node_geom'):
                permuted[key] = batch[key][:, order]
            permuted['matrix'] = batch['matrix'][:, order][:, :, order]
            torch.testing.assert_close(model(permuted)['utility'], output['utility'], rtol=1e-5, atol=2e-6)

    def test_strict_file_free_restore_all_buffers_and_no_prior_tampering(self):
        model = self.models[ARMS[1]]
        state = copy.deepcopy(model.state_dict())
        state['score_head.3.bias'].add_(.3)
        with patch('builtins.open', side_effect=AssertionError('restore file read')), \
             patch('torch.load', side_effect=AssertionError('restore checkpoint read')):
            restored = make_model(model.params, ARMS[1], state)
        self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(state))
        self.assertTrue(restored.initialization_receipt['state_dict_restored'])
        self.assertFalse(restored.initialization_receipt['fresh_initialization'])
        for key in (*PRIOR_BUFFERS, 'instance_encoder.geometry_residual.mean'):
            missing = dict(state)
            del missing[key]
            with self.assertRaises(RuntimeError):
                make_model(model.params, ARMS[1], missing)
        bad = copy.deepcopy(state)
        bad['prior_conditional'][0, 0, 0] += .01
        with self.assertRaises(ValueError):
            make_model(model.params, ARMS[1], bad)
        bad = copy.deepcopy(self.initial)
        bad['initialization'] = 'trained'
        with self.assertRaises(ValueError):
            initialize_shared(bad, self.geometry, ARMS[0], priors=self.priors)
        with self.assertRaises(ValueError):
            make_model(self.params, ARMS[0])


if __name__ == '__main__':
    unittest.main()
