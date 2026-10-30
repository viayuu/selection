# Versioned Native Output Serialization Recovery

Only NEW `code/V4/r121_collection_recovery.py` and this sibling recovery tree are
used. The frozen876 collector, structural_data, native profiles, two parent failure
records,21 qualifications and LIVE parent production labels are never overwritten.

## Bounded Diagnosis And Change

The frozen check_tour already plain-converts its tour argument. Its cost argument
still reaches np.isfinite directly. CPU CUDA-like cost/tour witnesses reproduce
both ATSP/UNICO_MatPOENet and TSP/OMNI failures there. The new layer normalizes BOTH
native cost and tour to CPU/plain BEFORE independent audit. This is an output-only
repair, not proof of the actual GPU call stack; main's two real reruns confirm it.

The helper executes frozen solve_native with exactly two checked substitutions:
normalize outputs before check_tour, and save these same normalized outputs.
Native input construction, backend.run, profiles, reset_lazy_runner, model/endpoint
seeds, algorithms and all native budgets remain verbatim. Recovery provenance pins
the old collector, helper, normalizer and NEW wrapper source hashes. Independent
route/cost audits are not weakened. Two CPU cases PASS; no GPU/solver was launched
by this sidecar.

## Ready Root

`/public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_serialization_recovery_v1`

CPU create (already submitted by the sidecar; see collection_recovery_create.log):

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m code.V4.r121_collection_recovery create --parent /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1
```

recovery.json records the two preserved first failures and references the21
successful qualifications by path, SHA, profile/input/task checks and physical
binding. Parent production remains live and is read at final federated export;
there is no stale copied snapshot of its labels. Native collection/recipe identity
stays the exact parent protocol; the output layer has its own sealed version/SHA.

## Main Reruns ONLY Two Failed Originals

Use the EXISTING main_r120_release.json and SAME main_binding_lane0.json. The actual
worker must retain GPU UUID, container hostname and CPU[4,5,72,73] binding exactly.
The following command is main-owned, not executed by the sidecar:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES="$R121_GPU_UUID" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 4,5,72,73 python -B -u -m code.V4.r121_collection_recovery wave --phase preflight --lane 0 --max-jobs 2 --device cuda:0 --release /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_r120_release.json --binding /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_binding_lane0.json --execute
```

There are ONLY two executable deployment keys in this overlay. Their task IDs must
equal the original failed originals. Already successful overlay preflights reuse
saved receipts; a new first failure is terminal, not generic retry. For a finite
production observation, change to --phase label --max-jobs8 after main approves.
Each deployment child holds the native adapter across its finite label shard.
All other87 deployments continue with the ORIGINAL frozen collector/parent root.
No all89/all-pool completion or full113920-label budget is implied by two repairs.

## Complete Full89 Export, Not A Reduced Pool

The original parent exporter intentionally remains incomplete for the two failed
methods. After all parent87 plus recovered2 deployments complete, use the NEW
saved-only federated exporter, which independently checks every native certificate:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m code.V4.r121_collection_recovery export
```

For each endpoint,87-method contributions come from parent labels and the two
recovered methods from overlay labels, according to the exact legal per-task pool.
No partial export or pool drop is accepted. The family export retains the existing
schema/parent collection identity and adds sealed output-layer provenance and both
receipt roots. Learner must use this NEW family_labels.jsonl via --family-labels.
This approach preserves the parent's concurrent8-label production observations.
