# R57 Focused Independent Review

A separate read-only reviewer inspected the R57 implementation and artifacts.
The reviewer did not edit files, launch solver jobs, or use a GPU. This is a
same-family review, not an independent reproduction on another machine.

## Verified Evidence

- Reconciled 301,920 route checks and independently recomputed all 1,920 saved
  validation routes. Classical backhaul and signed-residual-load checks are
  correctly separated. Open returns and absent depot deadlines are explicit.
- Checked all 1,800 saved score-row mappings: exact-size bijections, no fixed
  points for eligible rows, and unchanged singleton rows.
- Checked all 206 ALL metric rows against equal-weight 18-problem macros.
- Found no blocking numerical, route-check, or label-leakage error in the
  inspected train/validation analysis.

## Findings And Resolution

1. Failed native preflights were absent from the original blocked-runtime
   ledger. The ledger now joins actual preflight status and log paths, separately
   from recipe recoverability; analysis refreshes it after runtime probes.
2. Selector replay lacked the native workers' locked input-hash check. Both now
   use the same guard, and selector replay was repeated after the correction.

Two focused regression tests cover these fixes, in addition to the original18
tests. The test output is retained in `unit_tests.log`.

## Limits And Access Disclosure

The cyclic-shift sampler is a valid nonuniform derangement sampler, not a
uniform permutation test. Quantiles are empirical perturbation ranges only.
Preserve/reassign shuffles coincide for non-B ReLD/CVRP; changes to actual start
sets occur only where the multi-task B start truncation makes them different.

During an earlier identity sweep, the reviewer read full bytes of128 historical
test cost files through SHA256 hashing. No numeric test costs were parsed and no
test metrics or instances were analyzed. This was content access, not merely
metadata. Numerical results and runtime jobs remain train/validation only.
