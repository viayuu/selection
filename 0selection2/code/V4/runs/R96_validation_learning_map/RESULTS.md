# R96: Validation Learning Map

VAL-only descriptive overlap of 21 previously locked completed models. An instance counts as covered if any saved model predicted its native winner. Identifying that model uses the answer: this is NOT deployable selection, a new ensemble result, or an accuracy ceiling. No TEST/LIB data, new fit, new checkpoint, per-task model assembly or inference-rule changes.

No new neural/tree model was trained. No TEST/LIB predictions or metrics were used.
All21 catalog entries were locked before this analysis. R92A/R94/R95 are excluded because they were unfinished when the catalog was fixed.

## Observations

- Reference R89B VAL Top1: 71.248148%; actual regret: 0.400676%.
- 17,332/23,289 reference errors (74.42%) are missed by every catalog model.
- These common misses contribute 75.90% of R89B VAL regret, not an estimate of irreducible loss.
- Fixed36 winner cases: 4,625; R89B correct 52; any catalog model correct 146; all21 miss 4,479.
- Only 3.16% of those target winner cases were recovered by even one catalog entry; 96.84% were missed by all21. This is a catalog finding, not an accuracy ceiling.
- Catalog coverage is answer-dependent descriptive evidence. Weak models can add rare recall while harming overall cost; per-model false picks and net regret are reported separately.
- Correlated failure does not prove random labels, a Bayes accuracy ceiling, insufficient inputs, or a universal Encoder defect.

## Locked Model Metrics

| Model | VAL Top1 (%) | Actual regret (%) | Rescued / Harmed vs R89B |
| --- | ---: | ---: | ---: |
| R84A | 71.0852 | 0.403494 | 1332 / 1464 |
| R84B | 71.0000 | 0.404999 | 1300 / 1501 |
| R84C | 71.0235 | 0.408296 | 1262 / 1444 |
| R84D | 70.9543 | 0.408097 | 1384 / 1622 |
| R85A | 70.9679 | 0.406901 | 1523 / 1750 |
| R85B | 70.9358 | 0.405866 | 1496 / 1749 |
| R85C | 71.0086 | 0.397969 | 1366 / 1560 |
| R85D | 71.1272 | 0.403383 | 1198 / 1296 |
| R87A | 71.1481 | 0.406346 | 1224 / 1305 |
| R87B | 71.0790 | 0.402061 | 1160 / 1297 |
| R88A | 71.0580 | 0.405442 | 1301 / 1455 |
| R88B | 71.0580 | 0.405128 | 1223 / 1377 |
| R89A | 71.0506 | 0.405681 | 1332 / 1492 |
| R89B | 71.2481 | 0.400676 | 0 / 0 |
| R91A | 71.0728 | 0.405976 | 1324 / 1466 |
| R91B | 71.0914 | 0.408374 | 1184 / 1311 |
| R92B | 69.5580 | 0.437580 | 2904 / 4273 |
| R92C | 71.2864 | 0.403236 | 1616 / 1585 |
| R92D | 69.4247 | 0.430660 | 2747 / 4224 |
| R93A | 71.1259 | 0.407874 | 1202 / 1301 |
| R93B | 71.1099 | 0.407919 | 1303 / 1415 |

## Interpretation Boundary

This catalog asks where existing learned mappings disagree or repeatedly fail. It does not construct a deployable ensemble or choose a model per problem/instance. Current R94 representation-transfer and R95 exposure experiments continue unchanged. Do not expand a small calibration/penalty grid on the strength of this descriptive map.

Winner labels are native FP64 argmin. Exact ties may count as classification misses with zero regret; the separate zero-regret common-miss count prevents confusing those losses.

Reproduce (CPU only, easynco):

```bash
python -B -m code.V4.diagnostics.validation_learning_map
```

Primary report: `VALIDATION_LEARNING_MAP.pdf`; machine-readable tables and source hashes are adjacent.
