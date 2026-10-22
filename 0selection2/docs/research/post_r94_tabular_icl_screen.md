# Post-R94: pretrained tabular ICL screen

Date: 2026-10-22. Paper feasibility only; no local code, results, data, hardware, or model inspection, and no experiment prescription.

**Supplied context, not independently verified:** full18 frozen pools; per task 81K TRAIN / 4.5K VAL / 4.5K TEST; 2-10 native-winner classes; 357 input-only geometry/demand/TW/constraint statistics. GPU CatBoost reportedly reaches approximately 71.39% TEST without recovering most zero-method winners; the multilayer graph selector approximately 71.4%. These are not paper findings.

## Conditional Decision

**YES, conditionally: one broad idea is a frozen, synthetic-pretrained TabICL-family classifier on the existing statistics, using labeled TRAIN examples as context.** This is a genuinely different adaptation regime, not a new graph encoder, scoring MLP, R69 taskbank, or repetition of the supplied loss/optimizer history.

PFNs learn prediction algorithms across synthetic tables; downstream adaptation conditions fixed weights on examples rather than updating task-specific parameters. This remains a pretrained neural method, not proof that ICL cannot emulate optimization. See [TabPFNv2, "Principled in-context learning"](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/#Sec2). TabICL's full-context attention needs no nearest-neighbor retrieval; its internal table representations add no graph embeddings or instance-level routing information. See [TabICL, Sec. 3](https://arxiv.org/html/2502.05564v2#S3).

**Hard condition:** ICL necessarily places TRAIN labels inside the inference forward pass. That is legitimate supervised context, not a query feature or TEST-label leakage, but conflicts with a literal blanket ban on labels entering forward computation. If that ban is non-negotiable, **DECLINE**; this memo does not authorize an exception.

## Primary Evidence

The following are three verified full-text papers and two official release records, not a comprehensive leaderboard survey.

| Primary paper and sections | Actual scale evidence, not a universal context cap | External pretraining |
| --- | --- | --- |
| Qu et al., [TabICL](https://arxiv.org/html/2502.05564v2), ICML 2025; Secs. 3.2-4.4, 5.1; Apps. E.2, H | Pretraining: <=60K total rows, <=100 features; native <=10 classes. App. E.2: 80K context + 20K queries / 500 features, approximately 5GB GPU / 25GB RAM (Sec. 4.4: 32GB RAM); 400K context + 100K queries / 500 features, below 14GB GPU / approximately 120GB RAM, with offloading. | Synthetic only; approximately 83M tables, 60 A100 GPU-days. |
| Qu et al., [TabICLv2](https://arxiv.org/html/2602.11139v2), 2026 preprint, revision 2; Secs. 3-6; Apps. A.4, H.2 | Pretraining remains at most 60K rows, 100 features; largest-stage context is 80%, at most 48K. Native 10-class output. The million-row/500-feature demonstration means 800K context + 200K queries: approximately 50GB GPU, not 24GB. Disk offloading uses approximately 250GB disk, under 24GB host memory, approximately 450 seconds on H100. | Approximately 35M synthetic tables; 24.5 H100 GPU-days per model. |
| Hollmann et al., [TabPFNv2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/), Nature 2025; Methods: "Training details", "User guide", "Details on the neural architecture"; "Code availability" | Pretraining: up to 2,048 context rows plus 128 queries, 1-160 features. Main evaluated envelope: 10K rows, 500 features, 10 classes, not a hard architectural cap. The 50M-cell H100 capacity claim does not establish 81K-context performance or 24GB feasibility. | Approximately 130M synthetic tables/model; eight RTX 2080 Ti GPUs for two weeks. Original synthetic generator was not released with the paper. |

Downstream fit/predict prepares or caches labeled context and then performs contextual inference, not local SGD. Caching can move computation into fit; it does not erase inference costs (TabPFNv2, "An architecture designed for tables").

**Resource inference, not measurement:** 81K x 357 FP32 inputs occupy 0.116GB; adding 4.5K queries, the 128-dimensional cell tensor occupies approximately 15.6GB before workspaces/caches/ensembles (TabICL, Sec. 3.2/App. E.2). Both TabICL versions cost `O(n*m^2+n^2)` for n rows/m features; 81K context means 6.56 billion context-context pairs/head (TabICLv2, Sec. 3). FlashAttention saves score-matrix storage, not quadratic arithmetic. Offloading makes 24GB plausible, not project-verified; 81K/357 still exceeds pretraining dimensions.

Shuffling/ensembling matters: original TabICL used 32 column/class/preprocessing variants; TabICLv2 uses 8 in TabArena and 32 in TALENT, despite an 8-member default. Feature-order sensitivity is partly averaged, not eliminated. These are reported assumptions, not proposed settings; benchmark times are not single-query selector latency. See the respective papers' Sec. 5.1/App. H and Sec. 6/App. L.1.

- **Official TabICL release:** [jingang/TabICL, Files and license metadata](https://huggingface.co/jingang/TabICL/tree/main) publicly lists v1, v1.1, and v2 classifier weights, approximately 108-110MB, under BSD-3-Clause. Availability is verified, not downloaded or executed.
- **Official TabPFNv2 release:** [Prior-Labs/TabPFN-v2-clf, Model Details](https://huggingface.co/Prior-Labs/TabPFN-v2-clf) supplies public weights. Its [bundled license, Sec. 10](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt) is Prior Labs License 1.1, Apache-derived with additional attribution, not plain Apache-2.0. Do not transfer licenses or capacity claims between later TabPFN versions.

## Scientific Boundaries

The biggest scientific barrier is **feature sufficiency**: summary statistics may discard geometry-demand-TW alignments and combinatorial routing structure that distinguish winners. A different learned prior cannot recreate missing information. Neither these papers nor the supplied CatBoost tie proves recoverability of rarely selected methods.

Winner classification supplies neither a guarantee of zero-method recovery nor automatic joint Top1/cost improvement: winner labels omit cost-gap magnitude. Class permutations are not class balancing; forcing uniform method frequencies is unjustified.

Class IDs are task-local because legal pools differ; pooling identically numbered labels would change their meaning. Task-specific context is not a novel independently trained taskbank. Reject TEST-label-dependent approaches, solver portfolios/multiple executions, and any change from selecting one legal solver. No extra solver execution is intrinsic to ICL.

Synthetic-only pretraining is an author-reported provenance boundary, not an independent contamination audit; real benchmark feedback can still shape model development. Imported pretraining compute must be disclosed separately from downstream fit/inference, not presented as equal-budget training from scratch.

This is a family-level feasibility recommendation only. Operational choices and any rule clarification remain with Sol/user; no Astra model invocation is claimed or independently verified.
