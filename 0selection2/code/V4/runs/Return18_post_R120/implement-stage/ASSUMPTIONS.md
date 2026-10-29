# Assumptions Before Implementation

ASK mode: never. The current active user goal authorizes the four routing
research lines, additional labels where needed and the four existing GPUs.

| ID | Decision | Chosen | Class | Source |
| --- | --- | --- | --- | --- |
| A01 | Existing experiment | Finish and summarize R120 without changing its model, loss, selected checkpoint or training budget | semantic | user |
| A02 | Subsequent scope | Original 18 routing tasks and frozen new90k pools; R58 and B-table replacement remain deferred | semantic | user |
| A03 | Version separation | R121 D1; R122 hard constraints; R123 heterogeneous MoE; R124 Top-2. No per-problem best-model stitching | interface | default |
| A04 | Scientific division | Astra/max handles papers and scientific mechanisms; Sol/max handles protocols, data/code checks, implementation, results and execution | interface | user |
| A05 | R122 numerical target | Native winner classification error and FP64 relative regret over the full legal pool. Historical fixed missed-winner groups stay frozen, not dynamically replaced by current zero groups | semantic | post-R119 Astra HANDOFF |
| A06 | R122 proxy | CE/log(2) for native classification; sum of nonnegative regret times softplus winner margins/log(2) for a hard-regret upper bound | semantic | post-R119 Astra HANDOFF |
| A07 | R122 updates | Dual feedback uses detached hard TRAIN actions only after a successful optimizer update, averaged over complete task cycles. Forward/evaluation and AMP replay cannot advance it | interface | default |
| A08 | R122 units | Risk is scaled by a fixed 1% unit; missed-winner risk by a fixed TRAIN eligible-population rate. Budget provenance must be TRAIN, never VAL/TEST. Exact controller/runner settings are locked before training | semantic | default |
| A09 | Meta-learning isolation | Historical expert reuse requires recoverable training identities and genuinely unseen current TRAIN examples, or OOF retraining. A path or chronological version alone does not prove non-overlap | semantic | user/data isolation |

The R122 proxy is an engineering use of the cited bound, not an implementation
of Cotter's theoretical game algorithm or a guarantee for a deterministic
checkpoint. GPU execution is not authorized by a ready flag alone: R120's
existing pipeline must finish its evaluation/report first.
