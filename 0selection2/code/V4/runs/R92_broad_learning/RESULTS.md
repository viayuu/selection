# R92: Full18 Broad Learning Mechanisms

A: hard graph hierarchy; B: learned graph hierarchy; C: winner trees; D: relative-regret trees.
One validation-selected lock per version; no test-based reselection or task-wise best-model assembly.

| Model | Split | Top1 (%) | Mean cost | Actual regret (%) | Selected step / trees |
| --- | --- | --- | --- | --- | --- |
| R92/A | VAL | 70.9123 | 11.417302768 | 0.408460 | 57600 |
| R92/A | TEST | 71.1395 | 11.374428572 | 0.399837 | 57600 |
| R92/B | VAL | 69.5580 | 11.419872283 | 0.437580 | 51840 |
| R92/B | TEST | 69.7877 | 11.376775960 | 0.429720 | 51840 |
| R92/C | VAL | 71.2864 | 11.416956692 | 0.403236 | 800 |
| R92/C | TEST | 71.3901 | 11.374192262 | 0.394833 | 800 |
| R92/D | VAL | 69.4247 | 11.419639878 | 0.430660 | 100 |
| R92/D | TEST | 69.5716 | 11.376628156 | 0.424277 | 100 |

## Evidence Boundary

Historical same-data R84A/R89B references. Tree fitting is not an equal-compute neural control; all18 task trees use one common VAL-selected tree count. No TEST selection or per-task best assembly.
Neural arms use57600 successful logical-batch1024 updates; tree arms use3000 trees per task.
All18 tasks have equal macro weight. Historical TEST has informed development and is not blind.
Each arm full_model_statistics contains SBS/Oracle, all-problem metrics, NSS4, paired method distributions,
fixed target36 recovery, lost correct predictions, new false picks and net costs.
Fewer zero-selected methods alone is not successful recovery, and no outcome automatically completes the goal.

## R92C Evaluation Recovery

See C/recovery/RECOVERY.pdf. Original failed receipts are preserved; no fitting or checkpoint reselection.
