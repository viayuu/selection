# R78: Historical Fixed A/B Probability-Average Screen

Date: 2026-10-15. Final status: `fixed_equal_probability_route_closed_negative`.
Owner: papers/ideas only. Execution/checks: Sol. R78 remains active.

Round005 has completed its one70110-row CPU screen. Actual ensemble Top1 and
both costs fail the original VAL gate despite lower CE. Retain locked B;
no reweighting, changed ensemble rule, new training, taskwise rescue or sixth
similar postprocessor. See [final research closure](../RESEARCH.md#19-round005-negative-closure-and-evidence-wait)
and [execution result](../round005/RESULTS.md). Await NSS locked TRAIN/VAL.

The following proposal, approval and pre-screen review are historical, not
renewed execution instructions. Their conditional formal TEST/dual-forward
measurement procedure was not activated; no accepted release artifact exists
for this ensemble. Rounds001-004 also remain closed.

## Evidence And Decision

Read only [Sol's VAL handoff](R77_AB_VAL_COMPLEMENTARITY.md) and its
[summary CSV](R77_AB_VAL_COMPLEMENTARITY.csv). No linked source code, raw
cache, manifest, checkpoint, TEST/LIB or model was read by this researcher.
Sol reports aligned70110 VAL instances,3895/task, identical candidate IDs,
native labels and costs. This report is accepted as owner-supplied context,
not independently code-checked or re-audited.

| Existing VAL statistic | A | B |
|---|---:|---:|
| Native Top1 % | 70.199686207 | 70.894308943 |
| Raw macro mean cost | 11.397558589697 | 11.395558343544 |
| Actual macro regret % | 0.418612588 | 0.401326745 |
| Correct while other member is wrong | 1565 | 2052 |
| Cheaper while both members are wrong | 821 | 961 |

Choices differ on5399/70110 (7.7008%). Both are wrong on18841;17059 of those
share the same choice and cost. Disagreements concentrate in CVRP/OVRP/VRPL/
OVRPL, but the proposed rule applies to all18 without taskwise exceptions.

**Worth one cheap cache-only screen, not an expectation of improvement.**
There are observed A-only corrections and strict cost advantages on some
both-wrong cases, while B is better overall. These are genuine opportunities
and risks, not realized ensemble gains. The fixed rule's outcome remains
unknown without reading its decisions. No additional pre-screen statistics,
confidence-margin analysis, oracle construction or audit is requested.

The mechanism differs from fitting another function of p_B alone: a second
already-trained mapping supplies p_A. This does not establish incremental
conditional information, independent errors, new independent labels or a
causal data-size effect. Shared initialization/architecture, overlapping data,
different optimization histories and reuse of VAL remain limitations.

## Primary-Paper Basis

- Breiman (1996), [Bagging Predictors](https://link.springer.com/content/pdf/10.1007/BF00058655.pdf),
  Machine Learning24:123-140, DOI10.1007/BF00058655. The original paper links
  useful aggregation to training-set perturbation and learner instability;
  improvement is not universal. A10k/Bfull are not bootstrap replications, and
  squared-error averaging arguments do not guarantee better greedy cost.
- Lakshminarayanan, Pritzel and Blundell (2017),
  [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf),
  section2.4. The original prediction rule uniformly averages class
  probabilities. Its networks use different random initializations/shuffles;
  bagging worsened its experiments. This supports the rule's definition, not
  a guarantee for same-initialization R77 or a request to retrain members.

These are full-text primary sources, not an expanded paper search. Neither
paper establishes that lower CE implies better routing decisions here.

## Sole Rule And Finite Budget

Freeze A ep13/u16146 and B ep28/u34776, with each member's own original
TRAIN-derived preprocessing. Reuse Sol's existing aligned VAL caches; no new
model/solver inference, TRAIN full readout, training, GPU or model API calls.

For each valid task candidate j, define exactly:

```text
p_A = masked_softmax(A's cached native final logits, T=1)
p_B = masked_softmax(B's cached native final logits, T=1)
p_E(j | x) = 0.5 * p_A(j | x) + 0.5 * p_B(j | x)
decision = argmax over original valid candidates of p_E
tie rule = ascending original global method ID
```

Use the native probability definition associated with each locked artifact,
not utilities/ranks substituted for final logits. Preserve original candidate
IDs/masks; invalid candidates never enter the probability distribution.
No logit averaging, voting, restandardization, calibration, fitted bias,
temperature/weight search, new checkpoints, taskwise gate or frequency quota.

Budget: one deterministic CPU evaluation of one complete all18 candidate on
the existing70110 VAL rows, in chunks1024. There are zero optimizer updates;
1024 is a cache-processing chunk size, not new training. The existing seed2
checkpoints stay fixed; no seeds or training budgets are introduced.

Equal probability weights do not equalize confidence or make A and B equally
reliable. A's possibly harmful confidence remains part of this fixed test,
not a reason to fit a temperature after observing the result.

## Gate, Stop And Formal Artifact

Keep the inherited unrounded locked-B reference and zero-tolerance gate:

1. Native all18 macro Top1 strictly increases.
2. Existing raw macro mean cost does not increase.
3. Existing actual macro relative regret does not increase.

All three are required simultaneously. The rounded table above does not
redefine thresholds. CE/zero counts are descriptive, never substitute gates.
Sol reports actual ensemble metrics separately from a B fallback, with native
per-task results and adverse costs visible; no taskwise salvage and no new
precision, convergence, uncertainty or implementation-review loop requested.

If any condition fails, close this finite equal-weight route, retain B and
await the already-planned NSS locked TRAIN/VAL handoff. Do not try another
weight, temperature, logits rule, checkpoint or selected subset to rescue it.

If all pass, record exploratory VAL eligibility. The user has approved the
conditional formal-release procedure below; no TEST precedes a VAL pass and
the complete artifact lock. Both members and the decision to screen were
informed by this same VAL; this is not fresh or independent confirmation. The macro gate does not
ensure every task improves. Because costs may tie and are only required to
not increase, passing need not establish any strictly lower routing cost.

Before a formal version is evaluated, lock one complete artifact: A and B
checkpoint identities, each frozen preprocessing state, original pool/deployment
identities, candidate mapping and masks, native probability definition/T=1,
weights0.5/0.5 and global-ID tie rule. This is one complete two-model release
artifact, not a single-network checkpoint; its metadata and cost accounting
must represent both members. Only that complete locked version gets one full
TEST evaluation; do not read completed R77 TEST to design or choose it. LIB
uses the identical bundle.

Real inference requires both model forwards and both sets of weights, roughly
twice the single-model network arithmetic, not necessarily twice wall-clock
latency or peak memory. Cheap cached screening is not cheap deployment, and
the routing-cost gate does not account for selector compute or storage. A
pass alone does not establish that the double-model operational cost is worth
paying. The user requires actual dual-forward overhead reporting by Sol only
after a VAL pass and artifact lock, alongside the one formal evaluation. No
extra model forward, profiling sweep or hardware work belongs to this CPU
screen or this research agent.

## Interpretation Boundaries

Native correctness can differ while costs tie. Counts alone do not quantify
cost savings. Averaging probabilities preserves a shared top1 under a common
tie order in exact arithmetic; disagreement rows can select a third method
that neither member chose. Consequently a member-picking oracle is neither
the deployed rule nor a strict performance bound for soft averaging.

Jensen gives `CE_E <= (CE_A + CE_B)/2`, not `CE_E <= CE_B`, and says nothing
about a Top1 or cost improvement. No independence assumption is needed to
define the average, and no diversity-to-gain guarantee is claimed.

## Research Criticism

One bounded gpt-6-astra/max review completed, CLI session
01a13d20-9161-7b43-a90a-e0beaba2ab63, exit0:
[request](AB_EQUAL_WEIGHT_ASTRA_MAX_REQUEST.md),
[verbatim response](AB_EQUAL_WEIGHT_ASTRA_MAX_RESPONSE.md).
It supports GO-ONE-SCREEN and emphasizes tied-cost Top1 gains, taskwise harms,
reused VAL and unpriced dual-model inference. Same-family/provisional research
criticism based on the supplied briefing, not independent result verification
or code checking. No follow-up review loop.

R78 stays active. Old app goal remains paused/unmodified; STATUS.json remains
owned by the resource agent. No screen was executed by this research agent.
