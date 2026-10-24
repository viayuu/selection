# R108 Model Preparation

Scope: only r108_model.py, r108_objective.py, r108_priors.py, their two CPU
test modules and this note. No runners, live R106/R107, launches or Git edits.

## Locked Contract

- Arms: A_prior_ce and B_density_ratio; identical original R88B forward,
  scratch R84 seed2 trainables (1,842,953), dropout 0.1, no new learnable head.
- Both expose utility=s=log(pi)+t, residual_utility=t, conditional_prior=pi,
  original legal differences, maximin logits and final argmax ordering.
- A: ordinary legal CE(s). B: 0.5*softplus(-t_y) plus
  0.5*sum_legal pi_m*softplus(t_m), including the true winner in the sum.
- Only batch.ind is supervision. No queue, risk, pair, R-Drop, entropy,
  cost objective or consistency weight; one forward per objective call.
- This is the low-priority precise falsification in Astra REPORT section 9.1,
  not certified efficacy or a claim of a better Bayes rule/optimization.

## Interface Decisions

| Decision | Choice | Source |
| --- | --- | --- |
| Factory restore | Compact prior dimensions/identity in params; full persistent buffers in state; no file reads | user |
| Objective diagnostics | parts.loss is actual A/B loss; parts.ce is detached CE(s); other original R93 PART_KEYS zero | user |
| Prior artifact | build_priors(cache_root, data_root=..., expected_rows=81000); prepare(cache_root, root, output=None) | default interface |
| Prior provenance | Frozen cache metadata identities and actual n/winner content hashes; never open referenced source data | default interface |
| N semantics | Exact cached model node count, including routing depot; unseen N uses task prior | user and R84 cache metadata |

## Prior CLI

```bash
conda activate easynco
python -m code.V4.r108_priors prepare --cache-root /tmp/shiys_R84_new90k_cache --root code/V4/runs/R108_conditional_density_ratio
```

Alternatively use the original R84 input_cache root. Reads only metadata.json,
n.npy and winner.npy under train for all 18 tasks sequentially, 81,000 rows/task.
Output is root/prepared/priors.json. An identical existing artifact is retained;
a different existing artifact is never overwritten. No VAL/TEST arrays, raw
datasets, solver results, IDs or costs are used. Global task prior has Laplace1;
each exact-N count shrinks by 20 toward that prior. reld_mt retains original
RELD_MOEL global ID 16; deleted candidates are not restored.

## Status

All six owned files implemented. Exports:

```python
make_model(params, arm, state=None, *, priors=None)
initialize_shared(initial, geometry, arm, device='cpu', *, priors)
ConditionalPriorObjective(selector, arm)
make_objective(selector, arm)
build_priors(cache_root, *, data_root=DEFAULT_DATA_ROOT, expected_rows=81000)
prepare(cache_root, root=DEFAULT_ROOT, *, output=None, **kwargs)
```

ARMS, SPEC, PRIOR_SPEC, PARAMETER_COUNTS, ORIGINAL and
FRESH_REFERENCE_SHA256 are exported by r108_model. PARAMETER_COUNTS is 1,842,953
for original and both named arms. No extra trainables; all prior lookup tables,
task fallback, legal sets, exact N keys and identity are persistent buffers.
The compact r108_prior params contain dimensions/provenance/buffer hashes,
not tables. Strict restore needs only params and the complete in-memory state.
Missing or changed prior buffers fail closed. Factories preserve caller CPU RNG.

CPU verification (synthetic fixtures only):

```bash
conda activate easynco
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest code.V4.test_r108_model code.V4.test_r108_objective -v
```

12 tests passed, 1.524 seconds. Checks include sequential TRAIN-only reads,
Laplace1/shrink20, alias/global-ID alignment, rejected cache identity changes,
all18 original utility preservation and identical A/B forward, complete fresh
state, input-label isolation, unseen-N fallback, candidate/node permutation,
file-free strict restore, exact CE/NCE and analytical gradients, eta/pi optimum,
true-winner inclusion in product noise, poisoned unused costs, ignored
consistency weight, illegal nonfinite masking, row-weighted microbatch math,
idle batch and no mutable objective state. CLI help and syntax checks passed.

Main reported successful actual TRAIN aggregation with identity
690a8623d0f1f3ea85392aa4f5c1360ec29409b8b2e4c8c21784992bb00feb82;
that aggregation was not rerun by this implementation worker. Main's 4090
memory witness, runner/report/admin work and formal experiments are separate.
No GPU work, launch or Git operation was performed by this worker. No measured
efficacy is claimed. No blocking model/objective source issue found in these
bounded CPU checks; this is not an independent efficacy/integrity certification.
