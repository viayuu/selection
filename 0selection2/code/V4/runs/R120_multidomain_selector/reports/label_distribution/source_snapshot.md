# Native label distribution: source snapshot

- Generated UTC: 2026-10-29T13:01:08.084849+00:00
- Source script: `/public/home/shiys/0selection2/code/V4/r120_label_distribution.py`
- Script SHA256: `87fae6d90b101516c9cbcf7bfcb073ec7d4df0182f97feb9c3b26f1a8122f4e7`
- Input manifest: `/public/home/shiys/0selection2/code/V4/runs/R120_multidomain_selector/nonrouting_manifest.json`
- Manifest SHA256: `af01b38cac77c4cdaa9893de1422161cefecd503c4aefbd79ac23ab0967875a1`
- Embedded PDF font: `/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc`
- No model predictions, GPU, training, solvers, or label changes.
- Native objectives are FP64; exact equality only, no tolerance/FP32 argmin.
- TRAIN/VAL/TEST only; pilot/benchmark populations are not concatenated.
- Canonical rule: first exact native FP64 best in frozen candidate order; not a unique winner.
- Fractional rule: unique=1; each member of an exact k-way best set=1/k; not real winner counts.
- Co-best support includes unique wins and all ties; percentages may total over 100%.
- All declared candidates, including every zero count, remain in fixed order.
- Small checks: count sums, fractional totals, saved best/canonical consistency.
- Recorded array hashes below are reused from split descriptors, not a new full audit.

## Input splits

| Problem | Split | N | Direction | Dataset version | Matrix SHA256 (recorded) |
| --- | --- | ---: | --- | --- | --- |
| MIS | train | 10000 | max | nonrouting_v2_20261025 | `c42a0925038fcb746ec2249064271df1cc5849634b3dd6ac600c3c555a11266a` |
| MIS | val | 1000 | max | nonrouting_v2_20261025 | `55aab603509c6ff26bdb25b3a3ed87d048704b5af651b945780358eb98610f7b` |
| MIS | test | 1000 | max | nonrouting_v2_20261025 | `27df8e49f19c8224864d7a163db06c17fac75881848d1be33a1f32d8c807a3ff` |
| JSSP | train | 10000 | min | nonrouting_v2_20261025 | `556bcc4fd89ba2620af1e9bee04a356fe5d75d267761c3fff2ef637034c1ba84` |
| JSSP | val | 1000 | min | nonrouting_v2_20261025 | `f1971ae325bf0b596273e8ec11e06d8ac2b8a78b94899d91022001dc6e5fc524` |
| JSSP | test | 1000 | min | nonrouting_v2_20261025 | `8fd30660448254de68836fc41db19f0326596f42b5e5fbf5cfa38d2bf292d953` |
| FJSP | train | 10000 | min | nonrouting_v2_20261025 | `37b2f7fd0866a75f9d293049253f66e166e781d480a5e55fd880715c23cf2bb7` |
| FJSP | val | 1000 | min | nonrouting_v2_20261025 | `50e678a600e96c7def40c8a233f42093501523ab52397e92e20b3c021b6ccd63` |
| FJSP | test | 1000 | min | nonrouting_v2_20261025 | `dece83564e2b0103ca7e7149411418129b701bd14591743f58c6857a2b2f6736` |
| BPP | train | 10000 | min | nonrouting_v2_20261025 | `407e5599ef17fb1223ef6b387aba9a2c453d8cffdf55ab6b142239daaae8f426` |
| BPP | val | 1000 | min | nonrouting_supplement_v1_20261029 | `5853a4208faca793e49c874a1ca6bb1227758d667fc873f4fb8cd3067532a1a6` |
| BPP | test | 1000 | min | nonrouting_supplement_v1_20261029 | `6db193f9938a9a0539acf897a524d734fe97c62e0132d46f288e2f4a4be23d36` |
| MKP | train | 10000 | max | nonrouting_supplement_v1_20261029 | `c62a0d01dd82094e7c90f3d52770d1ac6b40235be9cabab9d7dbba428a5d6e8e` |
| MKP | val | 1350 | max | nonrouting_supplement_v1_20261029 | `fed96966141b987f4d228b5a381d8fb845ae36b59da455bafc7b26f0cd40065d` |
| MKP | test | 1350 | max | nonrouting_supplement_v1_20261029 | `4d028a384e39b29ad6ea8c604ca4214de9df69468117806bb2548fb43a336f38` |
| SMTWTP | train | 10000 | min | nonrouting_supplement_v1_20261029 | `c2189d4284ec03b4cf26f5e07a6dab904a3f5678a4bbb3131acc0c8d88c5bd3d` |
| SMTWTP | val | 3750 | min | nonrouting_supplement_v1_20261029 | `46486183a635aa8de403f46641bc349ada5d42c7a12f24a1ae504d5458d7e9b6` |
| SMTWTP | test | 3750 | min | nonrouting_supplement_v1_20261029 | `b4765943ac9251ed95166ce54d7f0b4792e7c9c2a52ddaead32de91e1555e43a` |

# TRAIN native-label distribution (not model accuracy)

| Problem | Dominant canonical allocation | Unique wins of that method | Tie rate |
| --- | --- | --- | --- |
| MIS | T2T_ER: 5,055/10,000 (50.55%) | 5,055 (50.55%) | 22.42% |
| JSSP | MWKR: 6,042/10,000 (60.42%) | 6,027 (60.27%) | 2.61% |
| FJSP | DAN SD1 20x10: 2,727/10,000 (27.27%) | 2,590 (25.90%) | 8.74% |
| BPP | MBS': 8,646/10,000 (86.46%) | 8,646 (86.46%) | 13.54% |
| MKP | LPDualGreedy: 9,276/10,000 (92.76%) | 9,262 (92.62%) | 0.14% |
| SMTWTP | ATC: 9,202/10,000 (92.02%) | 7,840 (78.40%) | 13.62% |

Canonical: first exact native FP64 best in frozen candidate order; not a unique winner.
Fractional weights: unique=1; each member of an exact k-way best set=1/k; not real winner counts.
Full zero-preserving distributions and solver IDs are in method_distribution.csv.

## Source directories

```json
[
  {
    "problem": "MIS",
    "split": "train",
    "n": 10000,
    "direction": "max",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/selection_data_multidomain_seed2_20261029/nonrouting/MIS/train",
    "objective_sha256_recorded": "c42a0925038fcb746ec2249064271df1cc5849634b3dd6ac600c3c555a11266a"
  },
  {
    "problem": "MIS",
    "split": "val",
    "n": 1000,
    "direction": "max",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/selection_data_multidomain_seed2_20261029/nonrouting/MIS/val",
    "objective_sha256_recorded": "55aab603509c6ff26bdb25b3a3ed87d048704b5af651b945780358eb98610f7b"
  },
  {
    "problem": "MIS",
    "split": "test",
    "n": 1000,
    "direction": "max",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/selection_data_multidomain_seed2_20261029/nonrouting/MIS/test",
    "objective_sha256_recorded": "27df8e49f19c8224864d7a163db06c17fac75881848d1be33a1f32d8c807a3ff"
  },
  {
    "problem": "JSSP",
    "split": "train",
    "n": 10000,
    "direction": "min",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/selection_data_multidomain_seed2_20261029/nonrouting/JSSP/train",
    "objective_sha256_recorded": "556bcc4fd89ba2620af1e9bee04a356fe5d75d267761c3fff2ef637034c1ba84"
  },
  {
    "problem": "JSSP",
    "split": "val",
    "n": 1000,
    "direction": "min",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/selection_data_multidomain_seed2_20261029/nonrouting/JSSP/val",
    "objective_sha256_recorded": "f1971ae325bf0b596273e8ec11e06d8ac2b8a78b94899d91022001dc6e5fc524"
  },
  {
    "problem": "JSSP",
    "split": "test",
    "n": 1000,
    "direction": "min",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/selection_data_multidomain_seed2_20261029/nonrouting/JSSP/test",
    "objective_sha256_recorded": "8fd30660448254de68836fc41db19f0326596f42b5e5fbf5cfa38d2bf292d953"
  },
  {
    "problem": "FJSP",
    "split": "train",
    "n": 10000,
    "direction": "min",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/selection_data_multidomain_seed2_20261029/nonrouting/FJSP/train",
    "objective_sha256_recorded": "37b2f7fd0866a75f9d293049253f66e166e781d480a5e55fd880715c23cf2bb7"
  },
  {
    "problem": "FJSP",
    "split": "val",
    "n": 1000,
    "direction": "min",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/selection_data_multidomain_seed2_20261029/nonrouting/FJSP/val",
    "objective_sha256_recorded": "50e678a600e96c7def40c8a233f42093501523ab52397e92e20b3c021b6ccd63"
  },
  {
    "problem": "FJSP",
    "split": "test",
    "n": 1000,
    "direction": "min",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/selection_data_multidomain_seed2_20261029/nonrouting/FJSP/test",
    "objective_sha256_recorded": "dece83564e2b0103ca7e7149411418129b701bd14591743f58c6857a2b2f6736"
  },
  {
    "problem": "BPP",
    "split": "train",
    "n": 10000,
    "direction": "min",
    "dataset_versions": "nonrouting_v2_20261025",
    "directory": "/public/home/shiys/nonrouting_supplement_20261029/prepared_with_as/nonrouting/BPP/train",
    "objective_sha256_recorded": "407e5599ef17fb1223ef6b387aba9a2c453d8cffdf55ab6b142239daaae8f426"
  },
  {
    "problem": "BPP",
    "split": "val",
    "n": 1000,
    "direction": "min",
    "dataset_versions": "nonrouting_supplement_v1_20261029",
    "directory": "/public/home/shiys/nonrouting_supplement_20261029/prepared_with_as/nonrouting/BPP/val",
    "objective_sha256_recorded": "5853a4208faca793e49c874a1ca6bb1227758d667fc873f4fb8cd3067532a1a6"
  },
  {
    "problem": "BPP",
    "split": "test",
    "n": 1000,
    "direction": "min",
    "dataset_versions": "nonrouting_supplement_v1_20261029",
    "directory": "/public/home/shiys/nonrouting_supplement_20261029/prepared_with_as/nonrouting/BPP/test",
    "objective_sha256_recorded": "6db193f9938a9a0539acf897a524d734fe97c62e0132d46f288e2f4a4be23d36"
  },
  {
    "problem": "MKP",
    "split": "train",
    "n": 10000,
    "direction": "max",
    "dataset_versions": "nonrouting_supplement_v1_20261029",
    "directory": "/public/home/shiys/nonrouting_supplement_20261029/prepared_with_as/nonrouting/MKP/train",
    "objective_sha256_recorded": "c62a0d01dd82094e7c90f3d52770d1ac6b40235be9cabab9d7dbba428a5d6e8e"
  },
  {
    "problem": "MKP",
    "split": "val",
    "n": 1350,
    "direction": "max",
    "dataset_versions": "nonrouting_supplement_v1_20261029",
    "directory": "/public/home/shiys/nonrouting_supplement_20261029/prepared_with_as/nonrouting/MKP/val",
    "objective_sha256_recorded": "fed96966141b987f4d228b5a381d8fb845ae36b59da455bafc7b26f0cd40065d"
  },
  {
    "problem": "MKP",
    "split": "test",
    "n": 1350,
    "direction": "max",
    "dataset_versions": "nonrouting_supplement_v1_20261029",
    "directory": "/public/home/shiys/nonrouting_supplement_20261029/prepared_with_as/nonrouting/MKP/test",
    "objective_sha256_recorded": "4d028a384e39b29ad6ea8c604ca4214de9df69468117806bb2548fb43a336f38"
  },
  {
    "problem": "SMTWTP",
    "split": "train",
    "n": 10000,
    "direction": "min",
    "dataset_versions": "nonrouting_supplement_v1_20261029",
    "directory": "/public/home/shiys/nonrouting_supplement_20261029/prepared_with_as/nonrouting/SMTWTP/train",
    "objective_sha256_recorded": "c2189d4284ec03b4cf26f5e07a6dab904a3f5678a4bbb3131acc0c8d88c5bd3d"
  },
  {
    "problem": "SMTWTP",
    "split": "val",
    "n": 3750,
    "direction": "min",
    "dataset_versions": "nonrouting_supplement_v1_20261029",
    "directory": "/public/home/shiys/nonrouting_supplement_20261029/prepared_with_as/nonrouting/SMTWTP/val",
    "objective_sha256_recorded": "46486183a635aa8de403f46641bc349ada5d42c7a12f24a1ae504d5458d7e9b6"
  },
  {
    "problem": "SMTWTP",
    "split": "test",
    "n": 3750,
    "direction": "min",
    "dataset_versions": "nonrouting_supplement_v1_20261029",
    "directory": "/public/home/shiys/nonrouting_supplement_20261029/prepared_with_as/nonrouting/SMTWTP/test",
    "objective_sha256_recorded": "b4765943ac9251ed95166ce54d7f0b4792e7c9c2a52ddaead32de91e1555e43a"
  }
]
```

## Complete script source

```python
"""Read-only native FP64 label distributions for the six R120 new problems.

Canonical means the first exact best in the frozen candidate order, not a
unique/native solver winner or model accuracy. Fractional weight assigns 1/k
to every member of a k-way exact best set; it is not a real winner count.
Only aggregate arithmetic is checked here, not solver/certificate audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .nonrouting_data import NonRoutingDataset, load_manifest


PROBLEMS = ('MIS', 'JSSP', 'FJSP', 'BPP', 'MKP', 'SMTWTP')
SPLITS = ('train', 'val', 'test')
RUN_ROOT = Path(__file__).resolve().parent / 'runs/R120_multidomain_selector'
DEFAULT_MANIFEST = RUN_ROOT / 'nonrouting_manifest.json'
OUTPUT_ROOT = RUN_ROOT / 'reports/label_distribution'
FONT_PATH = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
CANONICAL_RULE = 'first exact native FP64 best in frozen candidate order; not a unique winner'
FRACTIONAL_RULE = 'unique=1; each member of an exact k-way best set=1/k; not real winner counts'
COLORS = dict(unique='#147d83', tied='#d79a38', text='#20313b', muted='#657581', grid='#dde4e8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def short_name(solver_id):
    problem, family, variant = solver_id.split('/', 2)
    if family == 'L2D':
        fields = variant.split('/')[0].split('_')
        return f'L2D {fields[0]}x{fields[1]}'
    if family == 'DAN':
        variant = variant.split('/')[0]
        return f'DAN {"SD2" if "+mix" in variant else "SD1"} {variant.replace("+mix", "")}'
    return dict(MBS_prime="MBS'", LPDualGreedy='LPDualGreedy',
                AntSystem='AS 20x100 FP64').get(family, family)


def label_counts(objective, observed, direction):
    if objective.dtype != np.float64 or objective.ndim != 2 or objective.shape != observed.shape:
        raise ValueError('Expected aligned native FP64 objective/mask matrices')
    if not objective.shape[0] or not observed.all() or not np.isfinite(objective).all():
        raise ValueError('This report requires completed, finite, fully observed labels')
    if direction not in ('min', 'max'):
        raise ValueError('Expected native min/max objective direction')
    optimum = (np.min if direction == 'min' else np.max)(objective, axis=1)
    best = objective == optimum[:, None]
    size = best.sum(axis=1)
    canonical = best.argmax(axis=1)
    columns = objective.shape[1]
    counts = np.bincount(canonical, minlength=columns)
    unique = (best & (size == 1)[:, None]).sum(axis=0)
    tied_allocation = counts - unique
    support = best.sum(axis=0)
    fractional = (best / size[:, None]).sum(axis=0, dtype=np.float64)
    sets, set_counts = np.unique(best, axis=0, return_counts=True)
    n = len(objective)
    if not (int(counts.sum()) == n and int(unique.sum()) == int((size == 1).sum())
            and int(tied_allocation.sum()) == int((size > 1).sum())
            and int(support.sum()) == int(size.sum()) and int(set_counts.sum()) == n
            and np.isclose(fractional.sum(), n, rtol=0., atol=1e-6)):
        raise ValueError('Label distribution totals do not add up')
    return dict(n=n, best=best, size=size, canonical=counts, unique=unique,
                tied_allocation=tied_allocation, support=support, fractional=fractional,
                sets=sets, set_counts=set_counts, optimum=optimum)


def self_check():
    objective = np.asarray([[5., 5., 4.], [5., 4., 4.], [3., 3., 3.]], np.float64)
    result = label_counts(objective, np.ones_like(objective, dtype=bool), 'max')
    np.testing.assert_array_equal(result['canonical'], [3, 0, 0])
    np.testing.assert_array_equal(result['unique'], [1, 0, 0])
    np.testing.assert_array_equal(result['support'], [3, 2, 1])
    np.testing.assert_allclose(result['fractional'], [1 + 1 / 2 + 1 / 3, 1 / 2 + 1 / 3, 1 / 3])
    precise = np.asarray([[1., 1. + 1e-8]], np.float64)
    result = label_counts(precise, np.ones_like(precise, dtype=bool), 'min')
    np.testing.assert_array_equal(result['unique'], [1, 0])


def collect(manifest_path):
    manifest = load_manifest(manifest_path)
    method_rows, summary_rows, set_rows, panels, sources = [], [], [], {}, []
    for problem in PROBLEMS:
        panels[problem] = {}
        for split in SPLITS:
            dataset = NonRoutingDataset(manifest_path, problem, split)
            try:
                objective = dataset.label_arrays['native_objective']
                observed = dataset.label_arrays['observed_mask']
                direction = dataset.objective_direction
                ids = dataset.candidate_solver_ids
                names = [short_name(solver_id) for solver_id in ids]
                if len(set(names)) != len(names):
                    raise ValueError('Short method names must map unambiguously to solver IDs')
                counts = label_counts(objective, observed, direction)
                if not np.array_equal(counts['best'], dataset.label_arrays['all_best_mask']):
                    raise ValueError('Stored best mask differs from native exact FP64 comparisons')
                if not np.array_equal(counts['best'].argmax(1), dataset.label_arrays['canonical_winner_index']):
                    raise ValueError('Stored canonical order differs from frozen candidate order')
                n, k = counts['n'], len(ids)
                versions = '|'.join(dataset.descriptor['dataset_versions'])
                for column, (solver_id, name) in enumerate(zip(ids, names)):
                    method_rows.append(dict(problem=problem, split=split, n=n,
                        objective_direction=direction, dataset_versions=versions,
                        candidate_order=column, short_name=name, solver_id=solver_id,
                        canonical_count=int(counts['canonical'][column]),
                        canonical_pct=100. * counts['canonical'][column] / n,
                        canonical_unique_count=int(counts['unique'][column]),
                        canonical_tie_allocated_count=int(counts['tied_allocation'][column]),
                        canonical_tie_allocated_pct=100. * counts['tied_allocation'][column] / n,
                        unique_win_count=int(counts['unique'][column]),
                        unique_win_pct=100. * counts['unique'][column] / n,
                        co_best_support_count=int(counts['support'][column]),
                        co_best_support_pct=100. * counts['support'][column] / n,
                        tie_fractional_share_equivalent_weight=float(counts['fractional'][column]),
                        tie_fractional_share_pct=100. * counts['fractional'][column] / n,
                        canonical_rule=CANONICAL_RULE, fractional_rule=FRACTIONAL_RULE))
                dominant = int(counts['canonical'].argmax())
                unique_dominant = int(counts['unique'].argmax())
                summary = dict(problem=problem, split=split, n=n, candidate_count=k,
                    objective_direction=direction, dataset_versions=versions,
                    unique_instance_count=int((counts['size'] == 1).sum()),
                    unique_instance_pct=100. * float((counts['size'] == 1).mean()),
                    tied_instance_count=int((counts['size'] > 1).sum()),
                    tie_rate_pct=100. * float((counts['size'] > 1).mean()),
                    max_exact_best_set_size=int(counts['size'].max()),
                    distinct_exact_best_sets=len(counts['sets']),
                    canonical_sum=int(counts['canonical'].sum()),
                    co_best_support_sum=int(counts['support'].sum()),
                    co_best_support_pct_sum=100. * counts['support'].sum() / n,
                    fractional_weight_sum=float(counts['fractional'].sum()),
                    zero_native_best_count=int((counts['optimum'] == 0).sum()),
                    dominant_canonical_short_name=names[dominant],
                    dominant_canonical_solver_id=ids[dominant],
                    dominant_canonical_count=int(counts['canonical'][dominant]),
                    dominant_canonical_pct=100. * counts['canonical'][dominant] / n,
                    dominant_unique_short_name=names[unique_dominant] if counts['unique'].any() else '',
                    dominant_unique_win_count=int(counts['unique'][unique_dominant]),
                    dominant_unique_win_pct=100. * counts['unique'][unique_dominant] / n,
                    canonical_rule=CANONICAL_RULE)
                summary_rows.append(summary)
                for mask, number in zip(counts['sets'], counts['set_counts']):
                    columns = np.flatnonzero(mask)
                    set_rows.append(dict(problem=problem, split=split, n=n,
                        objective_direction=direction, dataset_versions=versions,
                        exact_best_bitmask_in_candidate_order=''.join('1' if flag else '0' for flag in mask),
                        exact_best_set_size=len(columns), count=int(number), pct=100. * number / n,
                        exact_best_solver_ids=json.dumps([ids[c] for c in columns]),
                        exact_best_short_names=json.dumps([names[c] for c in columns]),
                        canonical_solver_id=ids[columns[0]], canonical_rule=CANONICAL_RULE))
                # Keep only small aggregate arrays for plotting, not the input/label matrices.
                panels[problem][split] = {key: counts[key] for key in
                    ('canonical', 'unique', 'tied_allocation', 'support', 'fractional')}
                panels[problem][split].update(summary=summary, ids=ids, names=names)
                sources.append(dict(problem=problem, split=split, n=n,
                    direction=direction, dataset_versions=versions,
                    directory=str(dataset.directory),
                    objective_sha256_recorded=dataset.descriptor['files']['native_objective.npy']['sha256']))
                print(f'{problem}/{split}: N={n}, K={k}, ties={summary["tie_rate_pct"]:.2f}%', flush=True)
            finally:
                dataset.close()
    return method_rows, summary_rows, set_rows, panels, sources


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    if not FONT_PATH.is_file():
        raise FileNotFoundError(FONT_PATH)
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    matplotlib.rcParams.update({'font.family': family, 'font.size': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.labelcolor': COLORS['muted'],
        'text.color': COLORS['text'], 'xtick.color': COLORS['muted'],
        'ytick.color': COLORS['text'], 'figure.facecolor': 'white'})
    return plt


def draw_page(plt, problem, panels):
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(16.54, 11.69))
    grid = fig.add_gridspec(3, 2, left=.15, right=.97, top=.845, bottom=.23,
                           hspace=.62, wspace=.12, width_ratios=(2.35, 1.))
    direction = panels['train']['summary']['objective_direction']
    fig.text(.04, .965, f'{problem}  |  \u539f\u751f FP64 \u539f\u6807\u7b7e\u6700\u4f18\u65b9\u6cd5\u5206\u5e03',
             fontsize=21, weight='bold', va='top')
    fig.text(.04, .922, '\u6807\u7b7e\u7edf\u8ba1\uff0c\u4e0d\u662f\u6a21\u578b\u51c6\u786e\u7387 | '
             f'native objective: {direction} | exact equality, no FP32 rounding / tolerance', fontsize=11)
    fig.legend(handles=[Patch(facecolor=COLORS['unique'], label='\u552f\u4e00\u6700\u4f18\uff08unique\uff09'),
                        Patch(facecolor=COLORS['tied'], label='\u5e76\u5217\u6700\u4f18\u6309\u56fa\u5b9a\u5019\u9009\u987a\u5e8f\u5206\u914d')],
               loc='upper left', bbox_to_anchor=(.035, .902), ncol=2, frameon=False, fontsize=10)
    for row, split in enumerate(SPLITS):
        panel = panels[split]
        summary, n, names = panel['summary'], panel['summary']['n'], panel['names']
        y = np.arange(len(names))
        ax, table = fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1])
        unique = panel['unique'] * 100. / n
        assigned = panel['tied_allocation'] * 100. / n
        ax.barh(y, unique, color=COLORS['unique'], height=.66)
        ax.barh(y, assigned, left=unique, color=COLORS['tied'], height=.66)
        ax.set_yticks(y, names, fontsize=9)
        ax.set_ylim(len(names) - .45, -.55)
        ax.set_xlim(0, 125)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.grid(True, color=COLORS['grid'], linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0, pad=8)
        ax.spines['bottom'].set_color(COLORS['grid'])
        for index, count in enumerate(panel['canonical']):
            pct = 100. * count / n
            ax.text(pct + 1.2, index, f'{count:,} ({pct:.2f}%)', va='center', fontsize=8.5)
        ax.set_title(f'{split.upper()}  |  N={n:,}  |  \u5e76\u5217\u7387={summary["tie_rate_pct"]:.2f}%'
                     f' ({summary["tied_instance_count"]:,}/{n:,})',
                     fontsize=11, loc='left', pad=12, weight='bold')
        if row == 2:
            ax.set_xlabel('canonical \u5206\u914d\u6bd4\u4f8b (%)\uff1b\u5e76\u5217\u7834\u5e8f\u4e0d\u662f\u552f\u4e00 winner', fontsize=9)
        table.set_xlim(0, 1)
        table.set_ylim(ax.get_ylim())
        table.axis('off')
        for x, text, align in ((.01, 'co-best \u652f\u6301 n (%)', 'left'),
                               (.62, '\u552f\u4e00 n', 'center'), (.99, '\u5206\u644a (%)', 'right')):
            table.text(x, 1.06, text, transform=table.transAxes, ha=align,
                       va='bottom', fontsize=9, weight='bold')
        for index in range(len(names)):
            support = int(panel['support'][index])
            table.text(.01, index, f'{support:,} ({100. * support / n:.2f}%)', va='center', fontsize=8.5)
            table.text(.62, index, f'{panel["unique"][index]:,}', ha='center', va='center', fontsize=8.5)
            table.text(.99, index, f'{100. * panel["fractional"][index] / n:.2f}%', ha='right', va='center', fontsize=8.5)
        table.text(.01, -.11, f'co-best \u5408\u8ba1 {summary["co_best_support_pct_sum"]:.2f}%\uff08\u53ef >100%\uff09',
                   transform=table.transAxes, fontsize=8, color=COLORS['muted'])
    ids, names = panels['train']['ids'], panels['train']['names']
    fig.text(.04, .185, '\u77ed\u540d\u2192\u5b8c\u6574 solver_id\uff08\u6309\u56fa\u5b9a\u5019\u9009\u987a\u5e8f\uff0c\u542b\u5168\u90e8\u96f6\u9879\uff09', fontsize=9, weight='bold')
    half = (len(ids) + 1) // 2
    for index, (name, solver_id) in enumerate(zip(names, ids)):
        column, line = divmod(index, half)
        fig.text(.04 + .49 * column, .163 - .017 * line, f'{index + 1:02d}  {name}  =  {solver_id}', fontsize=7.2)
    versions = ', '.join(sorted({p['summary']['dataset_versions'] for p in panels.values()}))
    fig.text(.04, .052, f'\u6570\u636e\u7248\u672c: {versions}', fontsize=8, color=COLORS['muted'])
    fig.text(.04, .031, 'canonical=\u56fa\u5b9a\u5019\u9009\u987a\u5e8f\u7b2c\u4e00\u4e2a exact-best\uff0c\u975e\u552f\u4e00 winner\u3002'
             '\u5206\u644a\u6743\u91cd=\u72ec\u8d62 1\uff0ck \u5e76\u5217\u5404 1/k\uff0c\u4e0d\u662f\u771f\u5b9e winner \u6b21\u6570\u3002',
             fontsize=8, color=COLORS['muted'])
    return fig


def train_summary(method_rows, summary_rows):
    lines = ['# TRAIN native-label distribution (not model accuracy)', '',
             '| Problem | Dominant canonical allocation | Unique wins of that method | Tie rate |',
             '| --- | --- | --- | --- |']
    for problem in PROBLEMS:
        summary = next(row for row in summary_rows if row['problem'] == problem and row['split'] == 'train')
        rows = [row for row in method_rows if row['problem'] == problem and row['split'] == 'train']
        dominant = max(rows, key=lambda row: row['canonical_count'])
        lines.append(f'| {problem} | {dominant["short_name"]}: {dominant["canonical_count"]:,}/{summary["n"]:,}'
                     f' ({dominant["canonical_pct"]:.2f}%) | {dominant["unique_win_count"]:,}'
                     f' ({dominant["unique_win_pct"]:.2f}%) | {summary["tie_rate_pct"]:.2f}% |')
    lines += ['', f'Canonical: {CANONICAL_RULE}.', f'Fractional weights: {FRACTIONAL_RULE}.',
              'Full zero-preserving distributions and solver IDs are in method_distribution.csv.', '']
    return '\n'.join(lines)


def run(manifest_path=DEFAULT_MANIFEST, output_root=OUTPUT_ROOT):
    manifest_path, output_root = Path(manifest_path).resolve(), Path(output_root).resolve()
    allowed = OUTPUT_ROOT.resolve()
    if output_root != allowed and allowed not in output_root.parents:
        raise ValueError('Outputs must stay in the owned label_distribution directory')
    self_check()
    manifest_hash = sha256(manifest_path)
    method_rows, summary_rows, set_rows, panels, sources = collect(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / 'method_distribution.csv', method_rows)
    write_csv(output_root / 'split_summary.csv', summary_rows)
    write_csv(output_root / 'exact_best_sets.csv', set_rows)
    plt = plotting()
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = output_root / 'label_distribution.pdf'
    with PdfPages(pdf, metadata={'Title': 'R120 native FP64 label distributions - not model accuracy',
                                'Author': 'R120 CPU label-distribution script'}) as pages:
        for problem in PROBLEMS:
            figure = draw_page(plt, problem, panels[problem])
            pages.savefig(figure)
            if problem in ('FJSP', 'BPP'):
                figure.savefig(output_root / f'preview_{problem}.png', dpi=100)
            plt.close(figure)
    if sha256(manifest_path) != manifest_hash:
        raise ValueError('The source manifest changed during this read-only report')
    summary = train_summary(method_rows, summary_rows)
    (output_root / 'train_summary.md').write_text(summary, encoding='utf-8')
    source_path = Path(__file__).resolve()
    snapshot = [
        '# Native label distribution: source snapshot', '',
        f'- Generated UTC: {datetime.now(timezone.utc).isoformat()}',
        f'- Source script: `{source_path}`', f'- Script SHA256: `{sha256(source_path)}`',
        f'- Input manifest: `{manifest_path}`', f'- Manifest SHA256: `{manifest_hash}`',
        f'- Embedded PDF font: `{FONT_PATH}`',
        '- No model predictions, GPU, training, solvers, or label changes.',
        '- Native objectives are FP64; exact equality only, no tolerance/FP32 argmin.',
        '- TRAIN/VAL/TEST only; pilot/benchmark populations are not concatenated.',
        f'- Canonical rule: {CANONICAL_RULE}.', f'- Fractional rule: {FRACTIONAL_RULE}.',
        '- Co-best support includes unique wins and all ties; percentages may total over 100%.',
        '- All declared candidates, including every zero count, remain in fixed order.',
        '- Small checks: count sums, fractional totals, saved best/canonical consistency.',
        '- Recorded array hashes below are reused from split descriptors, not a new full audit.', '',
        '## Input splits', '',
        '| Problem | Split | N | Direction | Dataset version | Matrix SHA256 (recorded) |',
        '| --- | --- | ---: | --- | --- | --- |',
    ]
    for row in sources:
        snapshot.append(f'| {row["problem"]} | {row["split"]} | {row["n"]} | {row["direction"]} | '
                        f'{row["dataset_versions"]} | `{row["objective_sha256_recorded"]}` |')
    snapshot += ['', summary, '## Source directories', '', '```json',
                 json.dumps(sources, indent=2), '```', '',
                 '## Complete script source', '', '```python', source_path.read_text(encoding='utf-8'), '```', '']
    (output_root / 'source_snapshot.md').write_text('\n'.join(snapshot), encoding='utf-8')
    print(summary, flush=True)
    print(f'PDF: {pdf}; {len(method_rows)} method rows, {len(summary_rows)} split rows, '
          f'{len(set_rows)} exact-best-set rows', flush=True)
    return output_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.manifest, args.output_root)


if __name__ == '__main__':
    main()

```
