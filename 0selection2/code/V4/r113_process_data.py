"""TRAIN-only, input-derived R113 program supervision, never solver trajectories.

Run ``python -B -m code.V4.r113_process_data --help``. The input reader deliberately
does not use ScenarioCache/NewSplitDataset: those readers also open native labels.
``compile_inputs`` is deployment-safe; ``execute_program`` is a TRAIN teacher only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch


VERSION = 'r113_typed_fragment_v1'
SCENARIO = 'new90k_seed2_B_filtered_MOEL_as_reld_mt_v1'
V4 = Path(__file__).resolve().parent
DEFAULT_CACHE = V4 / 'runs/R84_new90k_attention/input_cache'
DEFAULT_DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
TOLERANCE = 2e-6
TASK_TYPES = {
    'TSP': (0, 0, 0, 0, 0), 'CVRP': (1, 0, 0, 0, 0),
    'ATSP': (0, 0, 0, 0, 0), 'OVRP': (1, 1, 0, 0, 0),
    'VRPB': (1, 0, 1, 0, 0), 'VRPL': (1, 0, 0, 1, 0),
    'VRPTW': (1, 0, 0, 0, 1), 'OVRPTW': (1, 1, 0, 0, 1),
    'OVRPB': (1, 1, 1, 0, 0), 'OVRPL': (1, 1, 0, 1, 0),
    'VRPBL': (1, 0, 1, 1, 0), 'VRPBTW': (1, 0, 1, 0, 1),
    'VRPLTW': (1, 0, 0, 1, 1), 'OVRPBL': (1, 1, 1, 1, 0),
    'OVRPBTW': (1, 1, 1, 0, 1), 'OVRPLTW': (1, 1, 0, 1, 1),
    'VRPBLTW': (1, 0, 1, 1, 1), 'OVRPBLTW': (1, 1, 1, 1, 1),
}
PROBLEMS = tuple(TASK_TYPES)
GRAPH_FIELDS = ('structural', 'signed_interval', 'classical_B')
TASK_FIELDS = ('capacity', 'open', 'backhaul', 'length', 'time', 'matrix', 'has_depot')
NODE_FIELDS = ('x', 'y', 'signed_demand', 'is_depot', 'route_limit', 'service',
               'window_start', 'window_end', 'from_anchor', 'to_anchor',
               'mean_outgoing', 'mean_incoming')
SCALAR_FIELDS = (
    'demand_sum', 'demand_prefix_min', 'demand_prefix_max',
    'initial_remaining_lo', 'initial_remaining_hi', 'delivery_sum', 'pickup_sum',
    'path_length', 'billed_fragment_length', 'arrival', 'service_start',
    'departure', 'waiting_sum', 'min_window_slack', 'route_limit_slack',
    'remaining_if_start_full',
)
BINARY_FIELDS = (
    'signed_interval_ok', 'signed_full_start_ok', 'classical_capacity_ok',
    'classical_precedence_ok', 'time_prefix_ok', 'length_prefix_ok',
    'classical_B_resource_ok',
)
INPUT_KEYS = (
    'nodes', 'anchor', 'edge_features', 'topology', 'node_mask', 'sequence',
    'anchor_index', 'steps', 'n', 'problem_id', 'coord_dist', 'task_flags',
    'graph_mask', 'scalar_mask', 'binary_mask',
)
TARGET_KEYS = ('scalar', 'binary', 'pointer', 'reached', 'frontier', 'compatible',
               'scalar_delta', 'state_mask', 'transition_mask')
RAW_KEYS = {'kind', 'node', 'matrix', 'n', 'node_mask', 'problem_id', 'cbits',
            'problem_desc', 'coord_dist', 'd_in', 'pool_ids', 'mask', 'solver_mask'}


@dataclass(frozen=True)
class ProgramSpec:
    max_steps: int = 12
    neighbors: int = 2

    def __post_init__(self):
        if not 1 <= self.max_steps <= 128 or not 1 <= self.neighbors <= self.max_steps:
            raise ValueError('Require 1 <= neighbors <= max_steps <= 128')


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def compose_resource_envelopes(left, right):
    """Ordered signed-demand envelope composition; zero prefix is included."""
    s, lo, hi = (left[..., i] for i in range(3))
    u, low, high = (right[..., i] for i in range(3))
    if torch.is_tensor(left):
        return torch.stack((s + u, torch.minimum(lo, s + low), torch.maximum(hi, s + high)), -1)
    return np.stack((s + u, np.minimum(lo, s + low), np.maximum(hi, s + high)), -1)


def propagate_time(departure, travel, window_start, service):
    arrival = departure + travel
    start = (torch.maximum(arrival, window_start) if torch.is_tensor(arrival)
             else np.maximum(arrival, window_start))
    return arrival, start, start + service


def propagate_compatibility(reached, compatible, topology):
    """One directed Boolean propagation round, with source axis before target."""
    messages = reached[..., :, None] & compatible & topology
    incoming = messages.any(dim=-2) if torch.is_tensor(messages) else messages.any(axis=-2)
    updated = reached | incoming
    return updated, updated & ~reached


def _type_masks(problem):
    c, o, b, length, tw = TASK_TYPES[problem]
    scalar = np.zeros(len(SCALAR_FIELDS), dtype=bool)
    scalar[:7], scalar[15] = bool(c), bool(c)
    scalar[7:9], scalar[9:14], scalar[14] = True, bool(tw), bool(length)
    binary = np.array([c, c, b, b, tw, length, b], dtype=bool)
    return (np.array([c, o, b, length, tw, problem == 'ATSP', c], dtype=bool),
            np.array([not c, bool(c), bool(b)], dtype=bool), scalar, binary)


def compile_inputs(raw, problem, spec=ProgramSpec()):
    """Compile input only, on CPU, then return tensors on the input's device.

    Unknown keys, including any supervision, are rejected rather than silently
    forwarded. Canonical tie-breaking uses original index, not solver outcomes.
    The program is bounded fragment arithmetic, not a route construction solver.
    """
    problem = str(problem).upper()
    if problem not in TASK_TYPES:
        raise ValueError(f'Unknown task type: {problem}')
    if set(raw) - RAW_KEYS:
        raise ValueError(f'Remove non-input keys before compilation: {sorted(set(raw) - RAW_KEYS)}')
    kind = 'matrix' if problem == 'ATSP' else 'coord'
    if raw.get('kind', kind) != kind:
        raise ValueError('Problem and input representation disagree')
    if 'problem_id' in raw and not np.all(_numpy(raw['problem_id']) == PROBLEMS.index(problem)):
        raise ValueError('Problem identity disagrees with raw batch')
    if 'cbits' in raw and not np.all(_numpy(raw['cbits']) == np.array(TASK_TYPES[problem])):
        raise ValueError('Constraint bits disagree with the explicit task type')
    values = raw['matrix' if kind == 'matrix' else 'node']
    device = values.device if torch.is_tensor(values) else torch.device('cpu')
    values, sizes = _numpy(values), _numpy(raw['n'])
    if sizes.ndim != 1 or not np.issubdtype(sizes.dtype, np.integer):
        raise ValueError('n must be an integer [B] vector')
    batch, width = len(sizes), spec.max_steps
    expected_dim = 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
    if values.ndim != 3 or values.shape[0] != batch or batch == 0:
        raise ValueError('Expected nonempty batched raw inputs')
    if kind == 'coord' and values.shape[2] != expected_dim:
        raise ValueError(f'{problem}: expected {expected_dim} raw fields')
    if kind == 'matrix' and values.shape[1] != values.shape[2]:
        raise ValueError('Directed matrices must be square')
    if 'node_mask' in raw:
        mask = _numpy(raw['node_mask'])
        if not np.array_equal(mask, np.arange(values.shape[1])[None] < sizes[:, None]):
            raise ValueError('node_mask must agree with cropped true n')
    task, graphs, scalar_mask, binary_mask = _type_masks(problem)
    c, _, backhaul, length, tw = TASK_TYPES[problem]
    output = dict(
        nodes=np.zeros((batch, width, len(NODE_FIELDS)), np.float32),
        anchor=np.zeros((batch, len(NODE_FIELDS)), np.float32),
        edge_features=np.zeros((batch, width, width, 2), np.float32),
        topology=np.zeros((batch, width, width), bool),
        node_mask=np.zeros((batch, width), bool),
        sequence=np.full((batch, width), -1, np.int64),
        anchor_index=np.zeros(batch, np.int64), steps=np.zeros(batch, np.int64),
        n=sizes.astype(np.int64).copy(),
        problem_id=np.full(batch, PROBLEMS.index(problem), np.int64),
        coord_dist=np.asarray(_numpy(raw.get('coord_dist', np.full(batch, 4))), dtype=np.int64),
        task_flags=np.tile(task, (batch, 1)), graph_mask=np.tile(graphs, (batch, 1)),
        scalar_mask=np.tile(scalar_mask, (batch, 1)), binary_mask=np.tile(binary_mask, (batch, 1)),
    )
    if output['coord_dist'].shape != (batch,):
        raise ValueError('coord_dist must be [B] input metadata')
    for row, size in enumerate(sizes):
        n = int(size)
        if not 2 <= n <= values.shape[1]:
            raise ValueError('Each instance needs at least an anchor and one vertex')
        x = values[row, :n, :n] if kind == 'matrix' else values[row, :n]
        if not np.isfinite(x).all():
            raise ValueError('Non-finite valid raw inputs')
        feats = np.zeros((n, len(NODE_FIELDS)), np.float64)
        if kind == 'matrix':
            distance = x.astype(np.float64)
            if (distance < 0).any():
                raise ValueError('Negative travel is outside this program')
        else:
            feats[:, :expected_dim] = x
            xy = x[:, :2].astype(np.float64)
            distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        diagonal = np.diag(distance)
        feats[:, 10] = (distance.sum(1) - diagonal) / (n - 1)
        feats[:, 11] = (distance.sum(0) - diagonal) / (n - 1)
        indices = np.arange(n)
        if c:
            if not np.isin(feats[:, 3], (0., 1.)).all() or (feats[:, 3] == 1).sum() != 1:
                raise ValueError('Exactly one physical depot is required')
            anchor = int(np.flatnonzero(feats[:, 3] == 1)[0])
            if feats[anchor, 2] != 0:
                raise ValueError('Depot demand must be zero')
            if not backhaul and (feats[:, 2] < 0).any():
                raise ValueError('Negative demand on an explicitly non-B task')
            customers = indices != anchor
            if length and (not (feats[customers, 4] > 0).all()
                           or not np.all(feats[customers, 4] == feats[customers, 4][0])):
                raise ValueError('L must have one positive customer-stored route limit')
            if tw and ((feats[:, 5:7] < 0).any() or
                       (feats[customers, 7] < feats[customers, 6]).any()):
                raise ValueError('Invalid customer time/service fields')
        else:
            anchor = None
        if kind == 'matrix':
            order = np.lexsort((indices, feats[:, 11], feats[:, 10]))
        else:
            order = np.lexsort((indices, feats[:, 2], feats[:, 1], feats[:, 0],
                                feats[:, 7] if tw else np.zeros(n)))
        if anchor is None:
            anchor = int(order[0])
        order = order[order != anchor]
        count = min(width, len(order), math.ceil(math.sqrt(len(order))))
        selected = order[:count]
        feats[:, 8], feats[:, 9] = distance[anchor], distance[:, anchor]
        output['nodes'][row, :count] = feats[selected]
        output['anchor'][row] = feats[anchor]
        # L is customer-stored; copy its declared constant to the anchor register.
        if length:
            output['anchor'][row, 4] = feats[selected[0], 4]
        d = distance[np.ix_(selected, selected)]
        output['edge_features'][row, :count, :count, 0] = d
        output['edge_features'][row, :count, :count, 1] = d.T
        for source in range(count):
            rank = np.lexsort((np.arange(count), d[source]))
            neighbors = rank[rank != source][:spec.neighbors]
            output['topology'][row, source, neighbors] = True
        output['node_mask'][row, :count] = True
        output['sequence'][row, :count] = selected
        output['anchor_index'][row], output['steps'][row] = anchor, count
    return {key: torch.from_numpy(np.ascontiguousarray(value)).to(device) for key, value in output.items()}


def initial_state(program):
    """Exact input-only initialization; no teacher states or teacher lengths."""
    nodes, masks = program['nodes'], program['scalar_mask']
    batch, width, _ = nodes.shape
    scalar = nodes.new_zeros(batch, len(SCALAR_FIELDS))
    scalar[:, 4] = masks[:, 4]
    scalar[:, 15] = masks[:, 15]
    scalar[:, 14] = program['anchor'][:, 4] * masks[:, 14]
    binary = program['binary_mask'].to(nodes.dtype)
    pointer = nodes.new_zeros(batch, width + 1)
    pointer[:, width] = 1.
    reached = nodes.new_zeros(batch, len(GRAPH_FIELDS), width)
    reached[:, :, 0] = program['graph_mask'].to(nodes.dtype)
    return dict(scalar=scalar, binary=binary, pointer=pointer,
                reached=reached, frontier=reached.clone(),
                compatible=nodes.new_zeros(batch, len(GRAPH_FIELDS), width, width))


def _pair_compatibility(nodes, edge, task, graph_mask, valid):
    c, opened, _, length, tw, _, _ = task
    pair = valid[:, None] & valid[None] & ~np.eye(len(valid), dtype=bool)
    q = nodes[:, 2]
    qi, qj = q[:, None], q[None]
    total = qi + qj
    lo = np.minimum(np.minimum(0., qi), total)
    hi = np.maximum(np.maximum(0., qi), total)
    signed = hi <= 1. + lo + TOLERANCE
    classical = ((np.maximum(qi, 0.) + np.maximum(qj, 0.) <= 1. + TOLERANCE)
                 & (np.maximum(-qi, 0.) + np.maximum(-qj, 0.) <= 1. + TOLERANCE)
                 & ~((qi < 0.) & (qj > 0.)))
    common = pair.copy()
    if tw:
        first_start = np.maximum(nodes[:, 8], nodes[:, 6])
        second_start = np.maximum(first_start[:, None] + nodes[:, 5, None] + edge[..., 0],
                                  nodes[None, :, 6])
        common &= ((first_start <= nodes[:, 7] + TOLERANCE)[:, None]
                   & (second_start <= nodes[None, :, 7] + TOLERANCE))
    if length:
        billed = nodes[:, 8, None] + edge[..., 0]
        if not opened:
            billed = billed + nodes[None, :, 9]
        common &= billed <= nodes[0, 4] + TOLERANCE
    result = np.stack((pair, common & signed, common & classical))
    return result & graph_mask[:, None, None]


def execute_program(program):
    """FP64 reference execution after FP32 compilation; use only as TRAIN targets.

    No feasibility-based early stop. Inactive typed fields are zero and masked.
    This function does not fit candidates, read routes, optimize, or call solvers.
    """
    p = {key: _numpy(value) for key, value in program.items()}
    batch, width, _ = p['nodes'].shape
    initial = {key: _numpy(value) for key, value in initial_state(program).items()}
    scalar = np.zeros((batch, width + 1, len(SCALAR_FIELDS)), np.float64)
    binary = np.zeros((batch, width + 1, len(BINARY_FIELDS)), bool)
    pointer = np.full((batch, width + 1), width, np.int64)
    reached = np.zeros((batch, width + 1, len(GRAPH_FIELDS), width), bool)
    frontier = np.zeros_like(reached)
    compatible = np.zeros((batch, len(GRAPH_FIELDS), width, width), bool)
    scalar[:, 0], binary[:, 0] = initial['scalar'], initial['binary'].astype(bool)
    reached[:, 0], frontier[:, 0] = initial['reached'].astype(bool), initial['frontier'].astype(bool)
    for row in range(batch):
        nodes = p['nodes'][row].astype(np.float64)
        edge = p['edge_features'][row].astype(np.float64)
        task, count = p['task_flags'][row], int(p['steps'][row])
        c, opened, backhaul, length, tw, _, _ = task
        compatible[row] = _pair_compatibility(nodes, edge, task, p['graph_mask'][row], p['node_mask'][row])
        envelope = np.zeros(3, np.float64)
        delivery = pickup = path = clock = wait = 0.
        slack = float('inf')
        seen_pickup, precedence_ok, time_ok = False, True, True
        for step in range(width):
            if step >= count:
                for array in (scalar, binary, pointer, reached, frontier):
                    array[row, step + 1] = array[row, step]
                continue
            q = nodes[step, 2]
            envelope = compose_resource_envelopes(envelope, np.array([q, min(0., q), max(0., q)]))
            delivery += max(q, 0.)
            pickup += max(-q, 0.)
            precedence_ok &= not (seen_pickup and q > 0.)
            seen_pickup |= q < 0.
            travel = nodes[step, 8] if step == 0 else edge[step - 1, step, 0]
            path += travel
            billed = path + (0. if opened else nodes[step, 9])
            arrival, start, departure = propagate_time(clock, travel, nodes[step, 6], nodes[step, 5])
            wait += start - arrival
            clock = departure
            slack = min(slack, nodes[step, 7] - start)
            time_ok &= start <= nodes[step, 7] + TOLERANCE
            s, lo, hi = envelope
            current = np.array([s, lo, hi, hi, 1. + lo, delivery, pickup,
                                path, billed, arrival, start, departure, wait, slack,
                                p['anchor'][row, 4] - billed, 1. - s])
            scalar[row, step + 1] = current * p['scalar_mask'][row]
            capacity_ok = delivery <= 1. + TOLERANCE and pickup <= 1. + TOLERANCE
            flags = np.array([hi <= 1. + lo + TOLERANCE,
                              hi <= 1. + TOLERANCE and lo >= -TOLERANCE,
                              capacity_ok, precedence_ok, time_ok,
                              billed <= p['anchor'][row, 4] + TOLERANCE,
                              capacity_ok and precedence_ok])
            binary[row, step + 1] = flags & p['binary_mask'][row]
            pointer[row, step + 1] = step
            reached[row, step + 1], frontier[row, step + 1] = propagate_compatibility(
                reached[row, step], compatible[row], p['topology'][row][None])
    transition_mask = np.arange(width)[None] < p['steps'][:, None]
    result = dict(scalar=scalar.astype(np.float32), binary=binary, pointer=pointer,
                  reached=reached, frontier=frontier, compatible=compatible,
                  scalar_delta=np.diff(scalar, axis=1).astype(np.float32),
                  state_mask=np.arange(width + 1)[None] <= p['steps'][:, None],
                  transition_mask=transition_mask)
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in result.items()}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class TrainInputCache:
    """Whitelist-only R84 TRAIN input reader with stable source IDs.

    Only metadata.json, n.npy, coord_dist.npy, node.npy OR matrix.npy, and TRAIN
    input_metadata.npz are opened. No labels, costs, geometry, result or route file.
    """

    def __init__(self, cache_root, problem, data_root=DEFAULT_DATA):
        if problem not in TASK_TYPES:
            raise ValueError('Unknown problem')
        self.problem = problem
        self.directory = directory = (Path(cache_root) / 'train' / problem).resolve()
        if directory.parent.name != 'train' or directory.name != problem:
            raise PermissionError('Resolved source must remain a TRAIN/problem directory')
        metadata_path = directory / 'metadata.json'
        self.metadata = meta = json.loads(metadata_path.read_text())
        expected_kind = 'matrix' if problem == 'ATSP' else 'coord'
        expected_dim = None if expected_kind == 'matrix' else 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
        if (meta.get('split') != 'train' or meta.get('problem') != problem
                or meta.get('scenario_id') != SCENARIO or meta.get('cache_format') != 'r76_scenario_v1'
                or meta.get('cache_version') != 1 or not meta.get('complete')
                or meta.get('rows_written') != meta.get('rows') or meta.get('rows', 0) <= 0
                or meta.get('problem_id') != PROBLEMS.index(problem)
                or meta.get('kind') != expected_kind or meta.get('d_in') != expected_dim
                or meta.get('cbits') != list(TASK_TYPES[problem])
                or Path(meta['data_root']).resolve() != Path(data_root).resolve()):
            raise PermissionError('Require complete frozen R84 TRAIN identity and explicit raw task type')
        self.key = 'matrix' if expected_kind == 'matrix' else 'node'
        self.arrays = {}
        self.opened_paths = [str(metadata_path)]
        try:
            for key in ('n', 'coord_dist', self.key):
                path = directory / (key + '.npy')
                if path.resolve().parent != directory:
                    raise PermissionError('Input array symlink escapes the TRAIN directory')
                self.arrays[key] = array = np.load(path, mmap_mode='r', allow_pickle=False)
                spec = meta['arrays'][key]
                if list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']:
                    raise ValueError('Input array differs from R84 metadata')
                self.opened_paths.append(str(path))
            source_path = (Path(data_root) / (problem + 'train') / 'input_metadata.npz').resolve()
            if source_path.parent.name != problem + 'train':
                raise PermissionError('Source IDs must come from TRAIN')
            with np.load(source_path, allow_pickle=False) as source:
                self.source_ids = source['source_index'].astype(np.int64)
            if self.source_ids.shape != (meta['rows'],) or len(np.unique(self.source_ids)) != meta['rows']:
                raise ValueError('Distinct stable source IDs must align with TRAIN rows')
            self.opened_paths.append(str(source_path))
            self.identity = dict(
                problem=problem, split='train', rows=meta['rows'], scenario_id=SCENARIO,
                cache_directory=str(directory), data_root=str(Path(data_root).resolve()),
                metadata_sha256=_sha256(metadata_path), source_ids_sha256=_sha256(source_path),
                input_files=[dict(path=str(directory / (key + '.npy')),
                                  size_bytes=(directory / (key + '.npy')).stat().st_size,
                                  mtime_ns=(directory / (key + '.npy')).stat().st_mtime_ns)
                             for key in self.arrays],
                opened_paths=list(self.opened_paths),
                row_hash='sha256(task UTF8 + true_n int64 + cropped cached raw FP32 C bytes)',
            )
        except BaseException:
            self.close()
            raise

    def batch(self, rows):
        rows = np.asarray(rows)
        if (rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer)
                or (rows < 0).any() or (rows >= self.metadata['rows']).any()):
            raise ValueError('Expected nonempty valid TRAIN row indices')
        n = np.array(self.arrays['n'][rows], copy=True)
        maximum = int(n.max())
        values = self.arrays[self.key]
        selected = (values[:, :maximum, :maximum][rows] if self.key == 'matrix'
                    else values[:, :maximum][rows])
        selected = np.ascontiguousarray(selected)
        hashes = []
        for row, length in enumerate(n):
            crop = selected[row, :int(length), :int(length)] if self.key == 'matrix' else selected[row, :int(length)]
            hashes.append(hashlib.sha256(self.problem.encode() + np.int64(length).tobytes()
                                         + np.ascontiguousarray(crop).tobytes()).hexdigest())
        raw = dict(kind=self.metadata['kind'], n=torch.from_numpy(n),
                   coord_dist=torch.from_numpy(np.array(self.arrays['coord_dist'][rows], copy=True)),
                   **{self.key: torch.from_numpy(selected)})
        provenance = dict(row_index=rows.astype(np.int64), source_index=self.source_ids[rows],
                          input_sha256=np.asarray(hashes, dtype='S64'))
        return raw, provenance

    def close(self):
        for array in getattr(self, 'arrays', {}).values():
            mapping = getattr(array, '_mmap', None)
            if mapping is not None:
                mapping.close()
        self.arrays = {}

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def schema():
    return dict(version=VERSION, split='train', graphs=list(GRAPH_FIELDS),
                task_fields=list(TASK_FIELDS), node_fields=list(NODE_FIELDS),
                edge_fields=['directed_travel', 'reverse_travel'],
                scalar_fields=list(SCALAR_FIELDS), binary_fields=list(BINARY_FIELDS),
                task_types={key: list(value) for key, value in TASK_TYPES.items()},
                inputs=list(INPUT_KEYS), targets=list(TARGET_KEYS),
                defaults=asdict(ProgramSpec()), tolerance=TOLERANCE,
                source_semantics='declared input-derived fragment programs, not native solver trajectories',
                forbidden_sources=['costs', 'winner', 'raw_label', 'results', 'query routes', 'VAL', 'TEST', 'LIB'])


def build_assets(args):
    spec = ProgramSpec(args.max_steps, args.neighbors)
    problems = PROBLEMS if args.problems == ['ALL'] else tuple(args.problems)
    if not problems or len(set(problems)) != len(problems) or any(p not in TASK_TYPES for p in problems):
        raise ValueError('Choose distinct typed tasks, or ALL')
    if args.rows_per_problem <= 0 or args.start_row < 0 or args.shard_size <= 0:
        raise ValueError('Positive row/shard counts and nonnegative start row are required')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=VERSION, split='train', status='building', schema=schema(),
                    spec=asdict(spec), sources=[], shards=[], native_solver_calls=0,
                    code_sha256={Path(__file__).name: _sha256(__file__)})
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    started = time.perf_counter()
    for problem in problems:
        task_started = time.perf_counter()
        with TrainInputCache(args.cache_root, problem, args.data_root) as source:
            stop = args.start_row + args.rows_per_problem
            if stop > source.metadata['rows']:
                raise ValueError('Requested rows exceed frozen TRAIN')
            manifest['sources'].append(source.identity)
            for start in range(args.start_row, stop, args.shard_size):
                end = min(stop, start + args.shard_size)
                raw, provenance = source.batch(np.arange(start, end, dtype=np.int64))
                program = compile_inputs(raw, problem, spec)
                targets = execute_program(program)
                arrays = {**{'input__' + k: _numpy(v) for k, v in program.items()},
                          **{'target__' + k: _numpy(v) for k, v in targets.items()}, **provenance,
                          'version': np.array(VERSION), 'split': np.array('train')}
                name = f'{problem}_train_{start:06d}_{end:06d}.npz'
                path = output / name
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                manifest['shards'].append(dict(path=name, problem=problem, rows=end-start,
                    start_row=start, stop_row=end, sha256=_sha256(path),
                    bytes=path.stat().st_size, tensor_bytes=sum(v.nbytes for v in arrays.values())))
        print(json.dumps(dict(problem=problem, rows=args.rows_per_problem,
                              seconds=round(time.perf_counter()-task_started, 6))), flush=True)
    manifest.update(status='complete', seconds=time.perf_counter()-started,
                    rows=sum(s['rows'] for s in manifest['shards']),
                    bytes=sum(s['bytes'] for s in manifest['shards']),
                    tensor_bytes=sum(s['tensor_bytes'] for s in manifest['shards']))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ('status', 'rows', 'seconds', 'bytes', 'tensor_bytes')}))
    return manifest


def load_shard(path, device='cpu'):
    """Load one bounded shard, returning (input tensors, target tensors, IDs)."""
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['version']) != VERSION or str(archive['split']) != 'train':
            raise PermissionError('Only declared R113 TRAIN assets can be loaded')
        program = {key: torch.from_numpy(archive['input__' + key].copy()).to(device) for key in INPUT_KEYS}
        targets = {key: torch.from_numpy(archive['target__' + key].copy()).to(device) for key in TARGET_KEYS}
        provenance = {key: archive[key].copy() for key in ('row_index', 'source_index', 'input_sha256')}
    return program, targets, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('schema', help='Print semantic field/type schema; no data access')
    build = commands.add_parser('build', help='Generate declared programs from TRAIN inputs only, never train or solve')
    build.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE)
    build.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    build.add_argument('--output', type=Path, required=True, help='A new directory; existing paths are refused')
    build.add_argument('--problems', nargs='+', default=['ALL'])
    build.add_argument('--rows-per-problem', type=int, default=4)
    build.add_argument('--start-row', type=int, default=0)
    build.add_argument('--shard-size', type=int, default=64)
    build.add_argument('--max-steps', type=int, default=12)
    build.add_argument('--neighbors', type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == 'schema':
        print(json.dumps(schema(), indent=2))
    else:
        build_assets(args)


if __name__ == '__main__':
    main()
