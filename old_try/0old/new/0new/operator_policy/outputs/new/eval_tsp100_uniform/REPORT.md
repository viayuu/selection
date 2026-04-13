# Single-scale evaluation

- Checkpoint: `0new/operator_policy/outputs/new/selector_best.pt`
- Problem size: `100`
- Dataset: `EasyNCO/data/datasets/test_dataset_tsp_uniform/test_tsp100_nums10000_uniform.pt`
- Dataset size: `10000`
- Evaluated items: `10000`
- Eval batches: `313`
- Rollout steps: `100`
- Init zoo: `lehd, elg, difusco`
- Operator zoo: `two_opt, lehd_rrc_step, dact_2opt_step`

## Metrics

- Init length: `7.815282`
- Final length: `7.765323`
- Improvement: `0.049958`
- Elapsed seconds: `12594.79`

## Action distribution

- Init `lehd`: `0.0000`
- Init `elg`: `1.0000`
- Init `difusco`: `0.0000`
- Operator `two_opt`: `0.0000`
- Operator `lehd_rrc_step`: `1.0000`
- Operator `dact_2opt_step`: `0.0000`
