# Resource-Only Cache Transport Fix

The original `prepared.json`, `constraint_reference.json` and CPU receipt remain
unchanged as the first preparation attempt. Current default preparation is the
versioned `preparation/v2` subdirectory, with the updated runner source hash.

Training/TEST prefer `/tmp/shiys_R84_new90k_cache` when present; `--cache-root`
may name any transport-created copy. Exact original metadata (including source
identities), all array shapes/dtypes/file lengths, and native FP64 costs/int64
winner SHA256 are validated. No feature rebuilding or cache copying is done.
The protocol binds the canonical frozen cache, not its resource-only location;
the actual training path is recorded separately in `model/cache_transport.json`.

TEST prefers a complete node-local copy and otherwise reads the already copied
original `runs/R120_multidomain_selector/existing_test_cache/test`. Its metadata
is validated against the frozen R84 source manifest, and copied costs/winners
are bound to that shared original copy. No TEST labels are read by prepare.

The objective, budgets, selection rule, model, initialization, update counts,
and checkpoint/inference constraints are unchanged. GPU/solver/Git not run.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m code.V4.r122_experiment prepare --device cpu
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && PYTHONDONTWRITEBYTECODE=1 python -m code.V4.r122_experiment train --device cuda:0 --gpu-type 4090 --cache-root /tmp/shiys_R84_new90k_cache
```

The second command is for the parent launcher; it has not been executed here.
