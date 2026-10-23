# Capacity Scaling for Solver Discrimination

Date: 2026-10-23. Status: bounded papers/ideas memo, not an experiment proposal.

**Recommendation:** retain substantially greater trainable width and independent
depth as a genuine parallel research direction. Prior shallow objective changes,
private branches, recurrence, and hierarchy do not settle this question. The
literature supports conditional mechanisms, not a measured benefit for this
selector, a target parameter count, or a promised Top1 improvement. Existing
R99/R101 directions and the pending R102 donor extension remain unchanged.

## Scope And Provenance

- Requested identity: `gpt-6-astra`, `reasoning_effort=max`. Independent backend
  attestation is unavailable; this memo is not represented as an independently
  verified Astra/max run. No other model or subagent was invoked.
- Used the `research-lit` workflow with web sources only. Four primary papers
  were selected; relevant full-text sections and supplements were read, rather
  than relying on abstracts. This is a mechanism-focused review, not an
  exhaustive survey or a claim that these are the latest papers.
- Project context below is supplied by the user/Sol, not independently checked.
  The two permitted mechanism indexes were read only to identify overlapping
  families; no result/code links were followed.
- Only this document is owned. No project code, results, data, jobs, or weights
  were inspected. No GPU/Git operations, experiment rules, architecture
  specification, resource allocation, or result audit are part of this memo.

**Given context.** The original scratch full18 selector has approximately 1.84M
trainable parameters, hidden width 128, four instance layers, two joint layers,
and score-MLP/maximin processing. User/Sol reports NEW90k TRAIN and VAL near 71%
with a small aggregate gap, while some methods with thousands of true winners
are selected zero or almost zero times. Prior CE/dropout/focal/OVR,
gradient-balancing, probability-link, private-residual, importance-sampling,
ICL, and old eight-step GRU variants largely lacked joint gains. R94 frozen
native routing knowledge reportedly adds a small gain and distinct true
rescues; R102 extends numerical donor coverage and must continue. R99
input-local/global processing and R101 hint pretraining with the original
downstream model also remain current. These are supplied facts, not refreshed
job statuses or new performance findings.

## Four-Paper Evidence

| Paper | Evidence type | Relevance and principal limit |
| --- | --- | --- |
| Levine et al., NeurIPS 2020, [Limits to Depth-Efficiencies of Self-Attention](https://proceedings.neurips.cc/paper/2020/file/ff4dfdf5904e920ce52b48c1cef97829-Paper.pdf) | Simplified expressivity analysis plus language-model comparisons | Width and depth interact; neither the theoretical threshold nor language-model scaling transfers directly to solver discrimination. |
| Rampasek et al., NeurIPS 2022, [Recipe for a General, Powerful, Scalable Graph Transformer](https://proceedings.neurips.cc/paper_files/paper/2022/file/5d4834a159f1547b267a05a4e2b7cf5e-Paper-Conference.pdf) | Supervised graph benchmarks and component ablations | Relevant small-label graph evidence, but not a demonstration that pure capacity growth solves minority-winner discrimination. |
| Xiong et al., ICML 2020, [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf) | Initialization analysis and NLP experiments | Explains a trainability mechanism, not a graph-task accuracy guarantee. |
| Dong, Cordonnier, and Loukas, ICML 2021, [Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth](https://proceedings.mlr.press/v139/dong21a/dong21a.pdf) | Conditional rank analysis and architectural interventions | Explains why added mixing can destroy distinctions; not a diagnosis of this selector. |

### 1. Width Can Enable Useful Depth

Levine et al. analyze input-dependency expressivity using separation rank.
Sections 4.1-4.2 distinguish a depth-efficient regime from one where width
limits depth's relative advantage. Section 5 and Figure 2 show corresponding
width/depth trade-offs in ordinary Transformers. Crucially, Sections 2.2-2.3
remove softmax, layer normalization, and ReLU from the theoretical model; its
function remains nonlinear. Separation rank is not classification accuracy,
sample complexity, or proof that training finds a useful function. Do not plug
the selector's width into the paper's logarithmic threshold to prescribe its
depth. [Primary text, Secs. 2.2-2.3, 3, 4.1-4.2, 5.1-5.3](https://proceedings.neurips.cc/paper/2020/file/ff4dfdf5904e920ce52b48c1cef97829-Paper.pdf).

The empirical setting is autoregressive language modeling on English Wikipedia,
BookCorpus, and OpenWebText, not modest-label graph selection.
[Supplement, Sec. D](https://proceedings.neurips.cc/paper_files/paper/2020/file/ff4dfdf5904e920ce52b48c1cef97829-Supplemental.pdf).
The arXiv title *The Depth-to-Width Interplay in Self-Attention* refers to this
same work, not a fifth independent source.

**Project inference:** if winner distinctions require composing multiple
relations, independently learned stages could help; insufficient intermediate
width could prevent those stages from retaining the distinctions. This makes
joint consideration of width and depth more defensible than assuming either
one alone is universally preferable. It does not establish that such a
bottleneck exists here.

### 2. Modest Labels Permit Learning, Not Arbitrary Scaling

GraphGPS combines structural/positional information, local message passing,
global attention, and nonlinear processing. Section 3.3 motivates avoiding
information loss before global processing. Its supplementary Figure D.1 uses
residual connections followed by **BatchNorm**, not Xiong's Pre-LN recipe.
[Main text, Sec. 3.3](https://proceedings.neurips.cc/paper_files/paper/2022/file/5d4834a159f1547b267a05a4e2b7cf5e-Paper-Conference.pdf),
[supplement, Sec. D.1](https://proceedings.neurips.cc/paper_files/paper/2022/file/5d4834a159f1547b267a05a4e2b7cf5e-Supplemental-Conference.pdf).

ZINC has 10,000 training graphs. In Table B.1, the hybrid has 423,717 parameters
and MAE 0.070; removing global attention leaves 257,317 parameters and the same
rounded MAE. This is **not** a width/depth-controlled comparison, but clearly
limits a blanket benefit-from-more-computation story. The authors relate the
absence of an attention benefit to ZINC's local-substructure target.
[Supplement, Sec. A.1 and Table B.1](https://proceedings.neurips.cc/paper_files/paper/2022/file/5d4834a159f1547b267a05a4e2b7cf5e-Supplemental-Conference.pdf),
[main text, Sec. 4.1](https://proceedings.neurips.cc/paper_files/paper/2022/file/5d4834a159f1547b267a05a4e2b7cf5e-Paper-Conference.pdf).

Conversely, on large-scale PCQM4Mv2, GPS-medium improves reported validation
MAE over GPS-small. Architecture and training settings also differ; the
original validation set serves as the authors' evaluation set. This is neither
isolated capacity evidence nor a modest-label result.
[Main text, Table 5](https://proceedings.neurips.cc/paper_files/paper/2022/file/5d4834a159f1547b267a05a4e2b7cf5e-Paper-Conference.pdf),
[supplement, Sec. A.1 and Table A.4](https://proceedings.neurips.cc/paper_files/paper/2022/file/5d4834a159f1547b267a05a4e2b7cf5e-Supplemental-Conference.pdf).

**Project inference:** the useful question is whether additional capacity can
learn repeatable winner-relevant structure from the permitted inputs. A few
thousand winners may cover a coherent pattern, or many fragmented regimes.
Neither label count nor parameter count alone resolves that distinction.

### 3. Normalization Changes Trainability

Xiong et al. distinguish normalization after residual addition from
normalization inside the residual branch. Sections 3.3-3.4 analyze and measure
initialization-time gradient scales; their Pre-LN configuration has more
manageable gradients than Post-LN in the studied setting. The theory uses
simplifications including single-head attention, special initialization, and
distributional assumptions. Section 4 validates optimization behavior in NLP,
not deeper graph classification with minority winners.
[Primary text, Fig. 1, Table 1, Secs. 3.3-3.4 and 4](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf).

**Project inference:** usable depth depends on the normalization/residual
path, not just the presence of a normalization operation. Pre-LN is a useful
mechanistic reference, not a universal winner over graph-specific alternatives.
This does not establish a defect in the current implementation, prescribe
training settings, or imply that stable gradients yield useful new features.

### 4. Preserving Distinctions Is Separate From Stable Gradients

Dong et al. bound token homogenization in pure attention without residual
connections or MLPs. Their convergence statements depend on weight-norm and
input-residual conditions. Section 3.1 shows how skip paths can retain
nonuniform information, but does not guarantee preservation for every learned
parameterization. Section 3.2 shows MLPs can counteract contraction; this is not
a claim that arbitrary FFN enlargement cures collapse. Section 3.3 does not
find normalization alone to be an antidote in its analysis. Section 4.1 also
acknowledges distribution shift when residual connections are removed from
trained networks.
[Primary text, Secs. 2.2-2.3, 3.1-3.3 and 4.1](https://proceedings.mlr.press/v139/dong21a/dong21a.pdf).

**Project inference:** additional graph mixing must preserve distinctions that
could identify solver-specific advantages. Trainability and representation
preservation are different issues. Token-rank collapse is an informative
analogy, not proof of oversmoothing in this model. Residual bypasses also mean
nominal depth need not equal effectively used computational depth.

## What The Small TRAIN/VAL Gap Does Not Establish

The following is logical interpretation of the supplied observation, not an
audit or a paper-derived diagnosis. Assuming comparable accuracy definitions,
TRAIN and VAL near 71% indicate similar aggregate errors on seen and held-out
instances. They do not identify why either error remains.

- **Not proof of insufficient capacity.** Optimization limitations, objective
  or decision-rule trade-offs, input-information limitations, shared-task
  interference, and target ambiguity could also produce similar errors.
- **Not proof of sufficient capacity.** A restricted learned representation can
  underperform similarly on both splits; changing loss weights without
  substantially changing that representation need not resolve it.
- **Not proof of random labels or a 71% ceiling.** Neither follows from two
  aggregate accuracies. Deterministic recorded labels can still encode a very
  difficult input-to-winner relationship.
- **Not absence of overfitting everywhere.** Aggregate agreement can conceal
  method- or task-specific differences. Nor does it show that a larger model
  would remain resistant to memorization.
- **Not evidence of successful discrimination.** Having many true winners but
  few model choices identifies suppression, not its cause. Increasing a
  method's selection frequency is not equivalent to recovering its true wins.

Thus capacity limitation remains an admissible explanation, not a diagnosis.
The low gap weakens a simple account based entirely on unseen-data overfitting;
it neither rejects nor establishes the value of substantial scaling.

## Separate Mechanism Families

This classification uses only the two permitted
[historical mechanism](experiment_mechanism_index.md) and
[R80-R102 mechanism](experiment_mechanism_index_R80_R102.md) indexes. It does not
assert that every earlier comparison isolated its stated mechanism.

| Family | What changes conceptually | Relationship to this capacity question |
| --- | --- | --- |
| Greater shared width / independent depth | Representation space and separately learnable transformations | The newly authorized hypothesis; not merely a new name for the following families. |
| R63 shared eight-step GRU recurrence | Repeated application of shared dynamics | More sequential computation, but not the same parameterization as independent layers. Earlier recurrence does not settle independent-depth scaling. |
| R92 graph hierarchy | Coarsening and information routing | Changes computational organization, not just representational size. |
| R69 task banks / R87 private residuals | Partitioned task- or method-specific capacity | Total parameters and per-instance shared capacity are different objects; these are relevant cautions, not exhaustive scaling evidence. |
| R99 local/global input processing; R101 hint pretraining | Information exposure or representation learning before the original downstream decision | Existing directions remain intact; do not repackage a GraphGPS-like local/global change as pure capacity. |
| R94/R102 frozen native donors; historical ICL | Imported learned knowledge or historical TRAIN context | Not evidence that extra scratch trainables alone work. Frozen donor parameters and trainable selector parameters must not be conflated. |

## Recommendation, Risks, And Strongest Counterargument

**Conceptual recommendation:** keep substantial width and independent depth
alongside, not instead of, the existing routes. The hypothesis is that the
current legal inputs contain reusable discriminating relations that a small
shared representation cannot adequately compose or retain. Treat adequate
representation width, nonlinear composition, and trainable residual paths as
distinct considerations, rather than equating capacity with a raw parameter
total. No particular size, layer allocation, normalization choice, or training
recipe is selected here.

If attention already offers all-to-all access, greater depth would chiefly
offer additional composition, not first-time access to distant nodes. That is
a conditional distinction, not a finding about the current code. Likewise,
widening a downstream mapping cannot recover distinctions already discarded by
its input representation; this does not locate such a bottleneck in this model.

**Strongest counterargument:** the suppressed winners may not be reliably
identifiable from the available input at the available supervision density.
Missing informative structure, fragmented decision regions, or weakly
predictable near-ties could make a larger model memorize without learning a
transferable boundary. None is established here. The reported R94 rescues are
compatible with a useful learned-representation contribution, but cannot tell
us whether capacity alone would reproduce it. This is a stronger objection
than declaring 1.84M parameters inherently sufficient.

The corresponding risks are memorization, more expensive inference, impaired
optimization, excessive mixing, and additional parameters that reinforce
dominant-task shortcuts rather than useful minority distinctions. These are
possibilities, not measured outcomes or reasons to terminate pending routes.

**Main research gap:** none of these four papers establishes that substantial
scratch width/depth scaling improves full18 solver-winner discrimination with
this input, supervision, and decision rule. The most useful conceptual
questions are whether finer shared representations can expose the suppressed
advantages, whether extra independent stages perform useful composition, and
whether capacity and native knowledge address complementary limitations.
These questions motivate the parallel direction without specifying experiments
or promising that either literature mechanisms or theorems transfer.
