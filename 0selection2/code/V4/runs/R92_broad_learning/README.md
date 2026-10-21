# R92: Broad Learning Mechanisms

All four arms use the same frozen NEW90k full18 release and original legal
candidate pools. No solver reruns, relabeling, R58 repair, ReLD-only specialist,
new data split, output quota, or test-driven tuning is included.

| Arm | Mechanism | Fixed training budget |
| --- | --- | --- |
| A | Input-distance hard graph coarsening; message passing at fine/32/8 resolutions | 57,600 successful updates, logical batch1024 |
| B | Learned graph coarsening; graph relations are pooled and processed again | Same; input-only link0.1 and entropy0.01 |
| C | Independent input-statistic GPU CatBoost winner classifier for every task | 3,000 trees/task, depth8 |
| D | Same input features, complete legal-pool relative-regret MultiRMSE | 3,000 trees/task, depth8 |

A/B preserve solver ID and34 features, two joint layers, the original score MLP,
maximin inference and R77 selection losses. Both use the known physical node0
execution role for TSP. The old global instance attention is replaced, not run
before a renamed pooling head. B is DiffPool-inspired, not an exact reproduction;
it reconstructs input relations, not solver routes. Shared new weights have one
fresh initialization; assignment-only parameters exist only in B.

C/D are independent model-family tests, not equal-SGD-budget comparisons. They
use no neural embeddings or performance labels as forward inputs. Eighteen task
models form one complete selector bank. A single common tree count is selected
from full18 validation; selecting a separate best count for each task is forbidden.

Every arm locks one model/rule by validation macro_vs_sbs_pct using TRAIN-fitted
SBS. The same lock is evaluated on full TEST, TSPLIB and CVRPLIB. Historical R84A
and R89B are references, not newly matched controls. Existing TEST has informed
development and is not claimed blind. Report true missed-winner recovery,
new false picks, costs and task-equal macro metrics, not just zero-count changes.

## Execution

Only the main launcher starts workers inside existing authorized allocations.
GPU stages run in each allocation-owned container; PDF generation runs on CPU
on admin01 after GPU release, using the installed CJK font.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python -m code.V4.r92_launch parse
```

Per-arm logs and machine-readable progress live in A/B/C/D below this directory.
The launch controller records exact UUID/cgroup/CPU binding under ../R92_launch.
W&B is explicitly offline because credentials were previously unavailable.

Finish a version with all locked evaluations/PDFs before publishing its curated
code, protocol, metrics and reports once to GitHub. Do not publish weights,
raw data, caches, credentials, W&B stores or individual prediction blobs.
