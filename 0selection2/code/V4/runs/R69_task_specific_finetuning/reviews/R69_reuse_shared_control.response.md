`review_independence=same-family; acceptance_status=provisional`

**Allow reuse. Rerunning A is unnecessary. Amended R69 plan: 8/10, ready, PLAN ONLY.**

The saved [protocol](/public/home/shiys/0selection2/code/V4/runs/R68_sharpness_aware_finetuning/protocol.json), initialization/schedule records and [training path](/public/home/shiys/0selection2/code/V4/r68_experiment.py:215) match the proposed shared control: identical starting state, data/dropout order, objective, AdamW settings, gradient reduction, clipping, budget and integer validation selection. I found no scientific protocol difference requiring another A run. This establishes protocol equivalence, not an independent replication.

Amend execution as follows:

1. **Reuse fixed A.** Reference R68A’s complete TRAIN/validation history and immutable epoch28 checkpoint: **8644/18000 validation correct, 48.022222%**. Preserve its existing identity and selection. Do not retrain, recount, reselect or retest A.

2. **Train only B.** Initialize all18 complete selectors from the common **pre-fine-tuning** state: R67B terminal stage1 trunk plus fresh head670003. Do not initialize from trained A28. Retain the adopted schedule, native CE/RDrop, ordinary AdamW, real256×3 batches and16×3 tails. Each active branch receives its task gradient divided by3; clip the combined active gradient once at1 and perform one AdamW step. Inactive branches remain `grad=None`; retain whole-update AMP recovery.

3. **Separate expenditure.**

   | Component | Updates | Selected-row presentations | Accounting |
   |---|---:|---:|---|
   | New B training | 9,600 logical | 7.2M | Measure actual runtime/retries |
   | Reused R68A | 9,600 historical | 7.2M historical | Existing3,002.024237s;5 retries |
   | Shared inherited pretraining | 9,600 historical | 7.2M historical | Count once separately |

   B runs40 epochs and14.4M successful dropout-view rows; each branch receives1,600 active updates/400,000 rows. Describe R69 as **one new arm compared with a reused control**, not two newly executed arms. Capacity and per-parameter update counts remain unmatched.

4. **Freeze by validation only.** Evaluate B’s complete18-task validation each epoch. Select maximum integer correct count, earliest exact tie, saving all18 branches from that common epoch. Official B requires **more than8644 correct**; otherwise official A remains. Preserve the full-pool utility/global-ID-tie inference rule and one completed solver per query.

5. **Test only new B.** After the parent freezes B and the official arm, execute exactly one full original18-task TEST for B. Reuse A’s already-saved official TEST report unchanged, without new inference. Historical TEST exposure must remain disclosed; its values cannot guide B’s design or selection.

This amendment adds no verification gate. Existing R69 implementation requirements, sole assigned UUID/allocation1465/gpu03 restriction and one completed-version publication remain. R68’s negative performance conclusion is unchanged. No TEST artifacts, replay, hash verification or GPU operations were used here.

**Memory update:** R69 shared-control reuse is allowed after direct protocol/source inspection confirmed equivalence with completed R68A. Keep A28 fixed at8644/18000 validation correct; reuse its complete history, checkpoint and existing official TEST without rerunning or reselection. Execute only new task-specific B from the common initial trunk/fresh-head state, retaining the adopted40-epoch schedule and update semantics. Separate new B expenditure, historical A expenditure and inherited pretraining. Select one common-epoch B bundle by integer full18 validation count; official-arm tie favors fixed A. Only B receives a new frozen full18 TEST. No new R68 TEST informed this amendment; historical exposure remains disclosed. R58 remains deferred and the70% goal remains active and unachieved.