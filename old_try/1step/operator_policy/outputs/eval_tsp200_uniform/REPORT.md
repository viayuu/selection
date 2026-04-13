# Single-scale evaluation

- Checkpoint: `1step/operator_policy/outputs/new/selector_best.pt`
- Problem size: `200`
- Dataset: `EasyNCO/data/datasets/test_dataset_tsp_uniform/test_tsp200_nums128_uniform.pt`
- Dataset size: `128`
- Evaluated items: `128`
- Eval batches: `4`
- Rollout steps: `100`
- Init zoo: `lehd, elg, difusco`
- Operator zoo: `two_opt, lehd_rrc_step, dact_2opt_step`

## Metrics

- Init length: `10.964002`
- Final length: `10.719034`
- Improvement: `0.244968`
- Elapsed seconds: `382.94`

## Action distribution

- Init `lehd`: `0.0000`
- Init `elg`: `1.0000`
- Init `difusco`: `0.0000`
- Operator `two_opt`: `0.0000`
- Operator `lehd_rrc_step`: `1.0000`
- Operator `dact_2opt_step`: `0.0000`

## Warning

- This selector checkpoint was trained with `problem_size=100`, but evaluation uses `problem_size=200`.
- Results are out-of-distribution and should be interpreted cautiously.
