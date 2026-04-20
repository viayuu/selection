# Architecture Refresh for Solver Selector

Generated on 2026-04-20 while current experiments continued running. No active training jobs were interrupted.

## Goal

Find architecture changes that are more likely than small tuning to improve:

1. exact `top1`,
2. mean cost / `vs_sbs`,
3. rare-solver coverage, especially zero-pick collapse.

This note combines:

- current project evidence from [literature/CoEKS.md](/public/home/zhoucl/shiys/0selection2/literature/CoEKS.md), [literature/urs.md](/public/home/zhoucl/shiys/0selection2/literature/urs.md), [literature/LITERATURE_REVIEW_v2.md](/public/home/zhoucl/shiys/0selection2/literature/LITERATURE_REVIEW_v2.md), and [review-stage/AUTO_REVIEW.md](/public/home/zhoucl/shiys/0selection2/review-stage/AUTO_REVIEW.md),
- external literature search via `arxiv`, `deepxiv`, `exa`, and `semantic_scholar` fallback attempts,
- local critical review.

`oracle-pro` review was attempted, but the current environment is missing `OPENAI_API_KEY`, so the review portion below is a local fallback.

## Current empirical reading

The bottleneck is now fairly clear.

- Best completed lines are still around `macro_top1 ~= 0.53` and `macro_vs_sbs ~= -0.118%`.
- `macro_top3` is already very high, so the project is not mainly failing because the candidate set is always wrong.
- Many solver arms are still never picked, especially rare winners in `TSP`, `ATSP`, and multiple MVRP families.
- The heavy deep encoder overhaul is a clear negative result:
  - [R22 eval_epoch0](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R22_deep_overhaul_gaprank_seed2/eval_epoch0.json): `macro_top1 = 0.4451`, `macro_vs_sbs = +0.4789%`
  - [R22 eval_epoch1](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R22_deep_overhaul_gaprank_seed2/eval_epoch1.json): `macro_top1 = 0.4385`, `macro_vs_sbs = +0.4804%`
- A plain clean reranker is still fragile:
  - `r23e` was killed before evaluation, so it did not provide positive evidence.
- The newest active branch `R24` is directionally more sensible:
  - fix only `oracle_in_topk & base_wrong`,
  - keep a do-no-harm anchor on non-fixable examples.

So the real problem is not "make the encoder bigger". It is:

1. the model still collapses many rare solver modes before the final choice,
2. among the plausible top candidates, the final scorer still does not separate the winner reliably enough,
3. the architecture does not yet explicitly encode diverse proposal generation plus safe top-k refinement.

## External papers most relevant to this phase

| Paper | Main signal | Why it matters here |
|---|---|---|
| `REMI` (Xie et al., 2023) https://arxiv.org/abs/2302.14532 | multi-interest models fail from easy negatives and routing collapse; fix with hard negatives + routing regularization | your zero-pick collapse looks much closer to routing collapse than to a pure capacity limit |
| `kNN-Embed` (El-Kishky et al., 2022) https://arxiv.org/abs/2205.06205 | represent users with mixtures over multiple interests to improve recall and diversity | maps naturally to multi-proposal solver shortlist generation |
| `ComiRec` (Cen et al., 2020) https://arxiv.org/abs/2005.09347 | retrieve with multiple interest embeddings, aggregate with controllable diversity-accuracy tradeoff | suggests shortlist should be multi-interest and explicitly diversity-controlled |
| `LongRetriever` (2025) https://arxiv.org/abs/2508.15486 | decompose behavior into multiple contexts and retrieve per context | strengthens the case for multi-context proposal heads instead of one shortlist head |
| `Comprehensive List Generation for Multi-Generator Reranking` (2025) https://arxiv.org/abs/2504.15625 | multiple complementary generators before evaluator improve list comprehensiveness | directly supports multi-generator solver proposal, not a single support head |
| `DivSPA` (2023) https://arxiv.org/abs/2308.07629 | use multiple retrieval strategies and self-distillation to augment sparse positives | good template for rare-winner augmentation without inventing fake labels |
| `Direct Learning to Rank and Rerank` (Rudin and Wang, 2018) https://arxiv.org/abs/1802.07400 | direct rank objectives can matter more than proxy losses when top positions matter most | relevant because your task is entirely about exact top-1 among a tiny candidate set |
| `Class Incremental Learning for Algorithm Selection` (2025) https://arxiv.org/abs/2506.01545 | rehearsal-based memory helps when new algorithm classes appear | useful for future new-solver support and for rare-arm memory even before new solvers are added |
| `Safe Policy Improvement with an Estimated Baseline Policy` (Simao et al., 2019) https://arxiv.org/abs/1909.05236 | only deviate from baseline where data supports safe improvement | conceptually the cleanest justification for your current `R24` direction |

Already relevant local architecture references remain:

- [literature/CoEKS.md](/public/home/zhoucl/shiys/0selection2/literature/CoEKS.md)
- [literature/urs.md](/public/home/zhoucl/shiys/0selection2/literature/urs.md)

## Critical review of what to stop doing

### 1. Stop treating bigger monolithic encoders as the default next move

`R22` already says the current deep-overhaul path is not just "not tuned yet"; it is materially worse. A larger unified encoder without better proposal diversity or safer scoring is the wrong abstraction.

### 2. Stop spending search budget on candidate-source / gate micro-sweeps

The clean reviewer conclusion is right: after `R21` and `R23d`, the main bottleneck is no longer "which gate knob" or "which small top-k variant". The architecture needs a better proposal mechanism and a better residual scoring rule.

### 3. Stop using a single shortlist head as if recall were the only issue

Your evidence already says the shortlist problem is not simple low recall. The issue is that the proposal process is not diverse enough, and the final scorer is not robust enough when the right solver is present.

## Ranked architecture proposals

## 1. Multi-proposal shortlist with routing regularization

### Core idea

Replace the single shortlist/support head with `K` proposal heads that each represent a different latent "interest" or solver-selection mode, then union their candidates before final evaluation.

### Why this fits the current evidence

- It directly targets the zero-pick collapse problem.
- It matches the strongest external pattern from `REMI`, `kNN-Embed`, `ComiRec`, `LongRetriever`, and `CLIG`.
- It fixes the most likely failure mode of the current shortlist: one head collapses onto dominant solver families and leaves rare winners uncovered.

### Minimum viable implementation

In [model.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/model.py):

- add `num_proposal_heads = 4` or `6`,
- create learnable proposal queries or solver-interest queries,
- produce `K` different solver proposal distributions over the masked pool,
- keep a small learned solver embedding table and concatenate solver meta features when available.

In [train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/train.py):

- replace single support BCE with:
  - per-head support supervision,
  - routing entropy / load-balance regularization,
  - coverage loss over hidden winners,
  - interest-aware hard negative mining inside the available solver pool.

In [rerank_train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/rerank_train.py):

- rerank over the union of `top-m` from each proposal head plus base top-k,
- log:
  - proposal-head arm coverage,
  - hidden-winner recall per head,
  - union recall,
  - overlap between heads.

### What to avoid

- Do not let all heads share the exact same logits without regularization.
- Do not optimize only final shortlist BCE; that will collapse again.
- Do not make this a huge new backbone. This change should sit on top of the current stronger base line.

### Why it is the best next new branch

It is the only proposal that directly addresses both of your main complaints at once:

- rare arms not selected,
- top-1 still too low even when good candidates exist.

## 2. Progressive shared plus constraint-expert scorer

### Core idea

Keep a shared instance encoder, but add shallow `shared + constraint-specific + solver-family-specific` expert blocks only in upper scoring layers, closer to `PLE / MMoE / CoEKS` than to a fully rewritten encoder.

### Why this fits the current evidence

- `R22` suggests the full deep rewrite is too disruptive.
- `CoEKS` and `URS` both argue that constraints matter structurally, but your task is selector scoring, not full route generation.
- Negative transfer across routing families likely survives in the current dense scorer, especially for MVRP rare winners.

### Minimum viable implementation

In [model.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/model.py):

- keep the current instance encoder mostly intact,
- insert 1 or 2 small expert FFN blocks after pooled instance features:
  - one shared expert bank,
  - one constraint-specific expert bank keyed by constraint bitvector / problem signature,
  - optional solver-family expert bank.

- gate these experts with a lightweight router from:
  - problem signature,
  - global instance statistics,
  - maybe pooled geometry features.

- fuse with residual connections, not replacement.

In [train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/train.py):

- add weak load balancing,
- add small expert-distillation / knowledge-sharing loss only at the first expert block,
- keep the rest of the objective simple.

### What to avoid

- Do not repeat the `R22` pattern of deep random-initialized encoder replacement.
- Do not apply strong distillation at all layers; `CoEKS` already warns that over-sharing harms specialization.
- Do not put experts at the node-routing level for this project phase; use them at the scorer level first.

### Why this is second, not first

It is plausible and better grounded than the previous deep-overhaul attempt, but it still changes more of the model than Proposal 1 and does not directly enforce proposal diversity.

## 3. Baseline-anchored set-conditioned scorer

### Core idea

Turn reranking into a true set-conditioned scorer over solver tokens, with pairwise/listwise top-k supervision and an explicit "stay close to baseline unless clearly fixable" anchor.

### Why this fits the current evidence

- This is the cleanest response to the fact that `top3` is already high.
- It matches your `R24` intuition and the `SPIBB` style safe-improvement idea.
- It treats the task as comparing solvers inside a candidate set, not as global class prediction.

### Minimum viable implementation

In [rerank_train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/rerank_train.py):

- continue the `R24` path,
- replace simple candidate token MLP scoring with a deeper set-conditioned interaction:
  - cross-attention or transformer over solver tokens,
  - solver token = base score + support score + solver embedding + solver meta.

- train with:
  - pairwise Bradley-Terry / listwise PL style loss on fixable examples,
  - `do-no-harm` anchor on safe examples,
  - switch-gain weighting.

- evaluate not just `macro_top1`, but:
  - `top1_given_fixable`,
  - safe-example drift,
  - harm on base-correct examples.

### What to avoid

- Do not go back to pure gate search as the main thing.
- Do not apply global long-tail reweighting over unavailable solvers.
- Do not use `fixable_only` without the anchor; `R23d` already showed that failure mode.

### Why this is third

It is very likely necessary, but by itself it still depends on candidate quality. If proposal diversity is not fixed, this line can only help so much.

## Reserve ideas

## 4. Rare-winner rehearsal memory

Maintain a small exemplar memory of instances where rare solvers are oracle winners, and periodically replay them during shortlist / scorer training.

This is inspired by `Class Incremental Learning for Algorithm Selection`. It is not the first new branch I would run, but it is a strong add-on once Proposal 1 or 3 is in place.

## 5. Diversified positive augmentation for solver winners

Use `DivSPA`-style multi-source positive augmentation inside the solver space:

- base-neighbor positives,
- problem-neighbor positives,
- rare-winner memory positives,
- then self-distill to a cleaned positive target.

This is useful if rare-winner supervision is still too sparse after Proposal 1.

## Concrete recommendation for what to do next

Without interrupting current jobs:

1. Let `R24` finish. It is the right thing to test now.
2. If `R24` improves exact top-1 without damaging `vs_sbs`, keep Proposal 3 as the active scoring path.
3. The next genuinely new architecture branch should be Proposal 1:
   - multi-proposal shortlist,
   - routing regularization,
   - hard negatives,
   - union candidate evaluator.
4. Only after that, add Proposal 2 as a shallow scorer-level expertization path.

In short:

- `R24` is worth finishing.
- The most promising new architecture is not "deeper encoder v2".
- It is "multi-proposal shortlist + regularized routing + safe residual top-k scorer."

## Downloads added this round

Saved to [literature/downloads](/public/home/zhoucl/shiys/0selection2/literature/downloads):

- [2302.14532.pdf](/public/home/zhoucl/shiys/0selection2/literature/downloads/2302.14532.pdf)
- [2205.06205.pdf](/public/home/zhoucl/shiys/0selection2/literature/downloads/2205.06205.pdf)
- [2005.09347.pdf](/public/home/zhoucl/shiys/0selection2/literature/downloads/2005.09347.pdf)
- [1802.07400.pdf](/public/home/zhoucl/shiys/0selection2/literature/downloads/1802.07400.pdf)
- [2308.07629.pdf](/public/home/zhoucl/shiys/0selection2/literature/downloads/2308.07629.pdf)
