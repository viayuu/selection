# Advantage Region Diagnosis

Implement and run CPU-only, train-fitted interpretable solver preference diagnostics on all 18 tasks, with val/test and TSP/CVRP LIB holdouts. Inputs are existing dataset.pkl/raw_label.pkl and current registry pools, not README pool lists. Outputs are CSV region/size statistics, baseline NPZ arrays, methodology.json and findings.md under regions/. Only plateau_regions.py and this output directory are writable. No neural training, GPU, commits, or external model/error/permutation analysis.

Base commit: 9512d912cc2d19ec0347e2420cead8f03cc1ef16 (worktree already dirty).

Success command: `python -B -m code.V4.plateau_regions run`, followed by `python -B -m code.V4.plateau_regions verify`.

NPZ: original-order indices, native truth/winner, FP64 costs, registry pool/pool_ids, sizes, train-only baseline predictions/scores, diagnostic tree predictions and feature/range metadata. LIB is supplemental, excluded from 18-task macro summaries.
