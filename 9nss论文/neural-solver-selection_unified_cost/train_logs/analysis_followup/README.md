# Analysis Follow-up

This directory contains the post-review follow-up analyses requested after the full-scale comparison.

## Files

- `bootstrap_summary.csv`
  - Paired bootstrap confidence intervals for:
    - `true baseline` vs `balanced+stats`
    - `true baseline` vs `best single`
    - `balanced+stats` vs `best single`
  - Computed on full `test` and `benchmark`.

- `test_paired_rows.csv`
  - Per-instance aligned rows for the full `test` split.

- `benchmark_paired_rows.csv`
  - Per-instance aligned rows for the full `benchmark` split.

- `benchmark_cvrp_shift_by_size_run_a.csv`
- `benchmark_cvrp_shift_by_solver_run_a.csv`
- `benchmark_cvrp_shift_worst_run_a.csv`
  - Focused `CVRP/LIB` shift analysis for `run_a`.
  - `run_a` is the full `true baseline`.

- `benchmark_cvrp_shift_by_size_run_b.csv`
- `benchmark_cvrp_shift_by_solver_run_b.csv`
- `benchmark_cvrp_shift_worst_run_b.csv`
  - Focused `CVRP/LIB` shift analysis for `run_b`.
  - `run_b` is the full `balanced+stats` variant.

- `benchmark_cvrp_shift_by_size_bin.csv`
  - Coarser size-bin view for `CVRP/LIB`:
    - `<300`
    - `300-599`
    - `600-899`
    - `>=900`

## Main Takeaways

- On full `test`, the simple unified selector (`true baseline`) is significantly better than the bundled `balanced+stats` variant.
- On full `test`, the simple unified selector is also significantly better than the per-problem best-single baseline.
- On full `benchmark`, neither learned selector beats the best single method.
- The main benchmark weakness is concentrated in `CVRP/LIB`, especially on larger instances.
