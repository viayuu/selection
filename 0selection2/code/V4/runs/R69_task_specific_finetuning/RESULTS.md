# R69 Results

The authorized delivery phases completed successfully: original TRAIN, validation-only freeze, ONE new B TEST, and ONCE CPU analysis. The validation-frozen official model remains **reused R68A28/global68**. Actual primary and fresh Astra/max reviews found no actionable integrity defects. The parent whole-objective audit records `passed=false`: the official TEST is below70%. Publication is parent-owned; the goal remains active and unachieved.

R69 executed one new arm, `B_task_specific_seed2`, with 18 complete ordinary-AdamW selector branches. Fixed A received zero new training, checkpoint loads, inference, recounts, reselection or TEST. Its one original historical TEST and saved TRAIN/VAL records are reused unchanged. This reuse establishes protocol equivalence, not an independent replication.

## Validation Freeze

All 40 full18 validation epochs and the published B result were required before freeze. B's earliest maximum integer validation count is **8604/18000**, at **epoch20/global60**. Its published `best.pt` and unique `best_checkpoints/epoch020.pt` agree, with all18 branches from that one common epoch.

- B inference bundle: `B_task_specific_seed2/best_checkpoints/epoch020.pt`.
- B checkpoint SHA256: `8029e3125a1e8a52389abca64f508dbd8bc8516f66bbcf8bb352621afcb3b315`.
- B model-state SHA256: `fa1bc7d09fc83590640edb097b776c8fa97cd03354e0c137ae48f6552157541a`.
- Fixed A identity: R68A28/global68, **8644/18000** validation correct, unchanged from R68's existing lock.
- Official choice: B requires strictly more than8644; B is40 below A, so official A remains. Exact ties also retain A.
- New tests required/executed: A=false/0, B=true/1. A's one historical original TEST is retained separately.

[final_test_lock.json](final_test_lock.json) was written before the new TEST. Costs, TEST outcomes and individual task results did not select the checkpoint or official arm. There was no task-wise checkpoint assembly, averaging, head transplant or optimizer-mode transition.

## Selected Original18 Metrics

ALL is the original equal18-task macro. TRAIN has10000 rows/task; VAL and TEST have1000 rows/task. A is reused R68A; only B's TEST was newly evaluated.

| Model | Split | Native Correct | Top1 % | Top2 % | Top3 % | CE | Mean Cost | vsSBS % | Actual Regret % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reused R68A28 | TRAIN | 89252/180000 | 49.584444 | 82.443889 | 93.171667 | 1.065074291 | 11.176339280 | -1.079288307 | 0.936864681 |
| Reused R68A28 | VAL | 8644/18000 | 48.022222 | 81.283333 | 91.983333 | 1.119214873 | 11.174141584 | -1.024455199 | 0.974169165 |
| Reused R68A28 | TEST | 8435/18000 | 46.861111 | 80.494444 | 91.844444 | 1.133649776 | 11.178087126 | -0.960929170 | 1.015239038 |
| New R69B20 | TRAIN | 90596/180000 | 50.331111 | 83.065000 | 93.573333 | 1.050461746 | 11.179512737 | -1.062171017 | 0.953169317 |
| New R69B20 | VAL | 8604/18000 | 47.800000 | 81.361111 | 92.072222 | 1.119309624 | 11.184300592 | -0.965643837 | 1.042268989 |
| New R69B20 | TEST | 8338/18000 | 46.322222 | 80.800000 | 91.705556 | 1.127788530 | 11.184551713 | -0.922807027 | 1.061528592 |

B minus matched reused A is **-0.222222pp VAL** and **-0.538889pp TEST**. Against descriptive fixed historical R60B, B is **-1.561111pp VAL** and **-1.922222pp TEST**. R60B was neither reselected nor retested and is not this experiment's matched control.

| B Minus Reused A | TSP Top1 pp | CVRP Top1 pp | ATSP Top1 pp | MVRP Top1 pp | Task Gains/Regressions/Ties |
| --- | ---: | ---: | ---: | ---: | --- |
| VAL | +0.700000 | +0.400000 | +0.900000 | -0.400000 | 8/10/0 |
| TEST | +1.100000 | -0.300000 | +1.500000 | -0.800000 | 8/10/0 |

MVRP includes15 tasks. Against fixed R60B, B gains/regresses/ties4/14/0 tasks on VAL and1/17/0 on TEST. All task/family differences are descriptive; none chooses a branch or changes the frozen rule. The fixed full-bank recipe does not beat matched A on the predeclared validation target.

At B's final epoch40/global80, TRAIN is107974/180000 (**59.985556%**, CE0.871882843), while VAL is8369/18000 (**46.494444%**, CE1.273051418). These are actual final observations, not the selected model's TEST.

| Model | Epochs36-40 VAL Mean Top1 % | Population SD pp | Minimum % | Maximum % |
| --- | ---: | ---: | ---: | ---: |
| Reused R68A | 47.431111 | 0.240308 | 47.050000 | 47.783333 |
| New R69B | 46.181111 | 0.357108 | 45.666667 | 46.600000 |

Last5 summaries are neither seed uncertainty nor averaged checkpoints. TRAIN curves use actual every5/final/selected-best measurements only. Missing historical TRAIN stays missing; no dense TRAIN curve or inference-filled observation was invented.

## Actual Compute and Initialization

| Component | Successful Updates | Selected Rows | Successful View Rows | Attempted View Rows | AMP Retries | Arm Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEW R69B | 9600 | 7200000 | 14400000 | 14405152 | 5 | 3554.574466 |
| HISTORICAL reused R68A | 9600 | 7200000 | 14400000 | 14405632 | 5 | 3002.024237 |
| HISTORICAL inherited R67B stage1, ONCE | 9600 | 7200000 | See original stage1 record | See original stage1 record | 1 | 2764.217798 |

Each B branch has1600 active updates,400000 selected rows and800000 successful dropout-view rows. Actual B capacity is33173154 trainable parameters versus1842953 shared A parameters. Capacity and per-parameter optimizer histories are **not matched**. Unused input-path tensors can retain zero Adam steps; actual branch activations and parameter counters are saved in [branch_compute.csv](branch_compute.csv) and [parameter_optimizer_steps.csv](parameter_optimizer_steps.csv).

Saved records confirm identical initialization of all18 complete selectors from the exact R68 pre-fine-tuning state: R67B terminal stage1 averaged-x trunk plus the entire fresh head670003. No trained R68/R67-stage2 state initialized B. The actual task/row/dropout/LR order matches reused A, and every saved compute consistency check passed. These are inherited encoders, not from-scratch training.

The fixed schedule, objective and update semantics stayed unchanged: seed2/global41-80,40epochs, real256x3 and16x3 tails, two native-CE/legal-RDrop forwards, fresh ordinary AdamW, common successful-update LR clock, each active FP32 task gradient divided by3, one combined clip1 and one optimizer step. Inactive branches retain `grad=None` and do not advance parameters, moments, counters or weight decay. Five actual whole-triple AMP retries are counted separately from successful views.

B's peak allocated/reserved/physical memory was **15.661316/21.376953/21.702393GiB**. The focused synthetic CPU check remained the original one successful invocation; no CPU check was rerun in delivery. Delivery freeze, TEST and analysis each succeeded on their first actual invocation; no phase was retried.

## Terminal and Execution Receipts

| Phase | Actual Handle | Outcome | Measured Scope |
| --- | --- | --- | --- |
| Original TRAIN | `1465.101`, `r69_train`, pane`%82`, panePID505438, trainingPID269032 | Slurm`COMPLETED/0:0`; tmux gone | Slurm3604s; B arm3554.574466s |
| CPU freeze | [freeze_execution_receipt.json](freeze_execution_receipt.json) | Actual process rc0 | 3.298435s; finished2026-10-12T21:23:24.455620UTC |
| ONE new B TEST | `1465.102`, `r69_final_test`, pane`%85`, panePID745609, evaluatorPID1635387 | Actual process rc0; Slurm`COMPLETED/0:0`; tmux gone | Slurm17s; evaluation8.123364s; TEST procedure12.847509s |
| ONCE CPU analysis | [analysis_execution_receipt.json](analysis_execution_receipt.json) | Actual process rc0 | Analysis56.874977s; outer process58.697669s; finished2026-10-12T21:26:31.376222UTC |

Both GPU phases used the same existing allocation1465/nodegpu03/physicalindex1 UUID`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`. TEST's one immediate assigned-UUID idle check observed2MiB used and no assigned-UUID compute clients. No other GPU/client/queue/allocation was queried, used or stopped.

Actual original TRAIN terminal evidence is in [training_terminal_receipt.json](training_terminal_receipt.json), observed2026-10-12T21:23:04.586980UTC. Actual TEST terminal evidence is in [test_terminal_receipt.json](test_terminal_receipt.json), observed2026-10-12T21:24:57.655072UTC. Process exit alone was not accepted as terminal success; the original owned scheduler rows and owned tmux absence were recorded separately. Scheduler timestamps in those receipts are explicitly Asia/Shanghai.

The original trainer was preserved until its successful exit. No duplicate trainer/test/executor was created, and no GPU phase was stopped or restarted. The parent-created CPU `r69_delivery` executor performed this handoff. All foreground execution sessions created for freeze and analysis settled with rc0.

## Delivery Assets and Limits

[REPORT.md](REPORT.md), [selected_metrics.csv](selected_metrics.csv) and [comparison.csv](comparison.csv) provide full task/family/ALL TRAIN/VAL/TEST Top1/2/3, CE, mean_cost, vsSBS and actual regret. [comparison_deltas.csv](comparison_deltas.csv), [task_gain_regression.csv](task_gain_regression.csv) and [metric_availability.csv](metric_availability.csv) preserve fixed-reference differences and missing-data disclosures. [stage2_summary.csv](stage2_summary.csv), [last5_validation.csv](last5_validation.csv) and [matched_common_epochs.csv](matched_common_epochs.csv) preserve best/final/last5/common-actual-epoch observations.

The analysis produced874 comparison rows,18 root CSV tables and332 figures under `curves/delivery/`, including [learning curves](curves/delivery/learning_curves.png), [epoch optimization](curves/delivery/optimization_epoch.png), [update optimization](curves/delivery/optimization_update.png), [task curves](curves/delivery/problem_top1.png) and [family curves](curves/delivery/family_top1.png). [compute_summary.json](compute_summary.json), [reference_provenance.json](reference_provenance.json), [analysis_receipt.json](analysis_receipt.json) and [delivery_receipt.json](delivery_receipt.json) retain actual compute, provenance, return codes and limitations.

The one new B TEST evaluated all18000 original rows, FP32batch128 with104-row tails, original full legal pools, native winners and ascending-global-ID utility ties. Exactly144 wrapper batches and144 corresponding active-branch batches were recorded. Known problem identity routes one full selector forward/query. Deployed inference then completes one selected solver; delivery evaluation executed **zero solvers**. There was no augmentation, sampling, reference inference, prediction replay or separate singleton latency benchmark. Full evaluation walltime includes transfer, metrics and NPZ saving and is not singleton latency.

All18 B prediction files are present in `B_task_specific_seed2/test_predictions/`, containing live TEST logits, selected/native-winner local indices, rawFP64 costs, original row indices, instance sizes and original global pool IDs/names. Weights, NPZ, raw updates, W&B, cache and regenerable schedules remain local under user ignore. Git alone does not contain all inference assets; no Git action or force-add was performed.

Historical TEST exposure remains disclosed. Increased capacity, fewer active parameter updates and changed Adam histories prevent a pure causal task-interference claim. No result establishes an accuracy ceiling, label noise or a70% promise, and the new B TEST does not guide planning. Official validation-frozen original18 TEST is8435/18000 (**46.861111%**), below70%; [objective_audit.json](objective_audit.json) therefore records whole-goal failure, not completion. Actual primary and fresh Astra/max reviews both score scoped integrity9/10, ready, and performance4/10, not ready. Review independence is same-family and acceptance provisional. Verbatim responses and traces are in `reviews/`; the primary next-plan review explicitly excluded NEW R69 TEST and analysis. R58 is fully deferred. The eight legacy B-task feasible-domain mismatches remain disclosed/unfixed and are not a numeric native70 veto. Parent owns state/memory and one completed-version publication.
