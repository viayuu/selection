# R78: Primary Papers And One High-Level Direction

Scope: literature and mechanism hypotheses only. The task description supplied
by the user is the sole project-specific evidence used here. No local project
code, experiment reports, arrays or TRAIN/VAL/TEST files were inspected.
Existing research notes are unchanged. This note contains no experiment design
or execution instructions.

## Main Judgment

One defensible direction is **learning instance representations whose geometry
reflects the whole candidate pool's relative performance profiles**. The
mechanism to investigate is whether easy, broadly predictive input features
suppress the structural distinctions needed to recognize conditional solver
advantages. This is a literature-supported hypothesis, not a diagnosis of the
present models or evidence that another model would improve selection.

A marginal winner frequency of15-20% does not imply that a candidate should
be the preferred decision anywhere given the information available to a
selector. Positive predicted probability also does not establish conditional
dominance. Winner probability and expected decision cost need not favor the
same action. The meaningful question is whether observable instance structure
identifies a region of genuine relative advantage that the representation
fails to preserve. Neither class imbalance nor label noise follows from the
reported phenomenon.

## Three Primary Papers

### 1. Gradient Starvation: A Learning Proclivity in Neural Networks

Mohammad Pezeshki, Sekou-Oumar Kaba, Yoshua Bengio, Aaron Courville,
Doina Precup and Guillaume Lajoie. NeurIPS2021.

[Official paper and metadata](https://proceedings.neurips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html)
| [Original full text](https://proceedings.neurips.cc/paper_files/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf).

Sections2-3 explain how strongly predictive, easier-to-learn features can
suppress the learning of other predictive features under cross-entropy.
The central theoretical setting is a linearized network, binary classification
and interactions between feature directions; the paper also supplies broader
empirical illustrations. This makes feature competition a plausible mechanism
distinct from low class counts. Its spectral-decoupling treatment concerns
learning dynamics; this note does not propose copying its regularizer.

Transfer limit: the theorem does not diagnose a multicandidate, composite-loss
algorithm selector. Suppressing weaker features can sometimes be useful
regularization, as the paper itself discusses. Recovering more features need
not improve cost, and a zero-selection count does not demonstrate starvation.

### 2. Correct-N-Contrast: a Contrastive Approach for Improving Robustness to Spurious Correlations

Michael Zhang, Nimit S. Sohoni, Hongyang R. Zhang, Chelsea Finn and
Christopher Re. ICML2022, PMLR162:26484-26516.

[Official paper and metadata](https://proceedings.mlr.press/v162/zhang22z.html)
| [Author full text on arXiv](https://arxiv.org/html/2203.01517).

Sections3-4 study representation alignment: examples sharing a class but
different spurious attributes should retain their class-relevant similarity,
while superficially similar examples with different classes remain distinct.
The method uses initial-model predictions as proxies for unobserved attributes.
Its analysis connects within-class alignment to the gap between worst-group
and average error. This is evidence for changing what an encoder represents,
rather than only correcting output probabilities.

Transfer limit: the original setting has semantic classes and a specific
spurious-group structure. A solver winner is not necessarily such a semantic
class. Pulling together every instance with the same winning solver can erase
different runner-up orderings and cost gaps. Conversely, geometry or constraint
features that affect solver performance must not be declared spurious merely
because they correlate with an existing prediction. The proxy-group assumption
may fail, especially with collapsed predictions. No such groups are established
for this project by the abstract problem statement.

### 3. Rank-N-Contrast: Learning Continuous Representations for Regression

Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang and Dina Katabi. NeurIPS2023.

[Official paper and metadata](https://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)
| [Author full text, arXiv v2](https://arxiv.org/html/2210.01189v2).

Section3 contrasts instances according to distances between their continuous
targets, making representation neighborhoods reflect target-space relationships.
Its formulation permits vector-valued targets. This supplies the most direct
mechanism for using a complete candidate-performance profile rather than a
winner ID to organize instance representations. Section4's ordering results
concern the specified representation objective and its optimization conditions;
they are not guarantees of correct downstream decisions.

Transfer limit: the empirical tasks are generic regression, not solver
selection. Section6 explicitly says the method cannot distinguish incidental
from genuine input-target correlations and does not assure subgroup reliability.
Lower regression error or a better-ordered embedding need not improve the
argmin of a cost vector, particularly near decision boundaries.

## One Priority Mechanism Hypothesis

The proposed research question is whether **relative candidate-performance
structure should directly organize the instance encoder**. Instances with
similar patterns of solver advantages and disadvantages would have related
representations even when easy surface characteristics differ. Instances
that look similar but reverse economically meaningful solver preferences
would retain a representation distinction. The relevant relation includes
the other candidates and advantage magnitudes, not just the winner name.

The conceptual departure is across-instance geometry: relationships between
different instances' performance profiles guide which input distinctions
are retained. Within-instance ranking instead compares candidates for one
instance. The former is not implied merely by having the latter. This could
help a model recognize several distinct structural routes to the same
solver's advantage without treating that solver as a class that needs a quota.

The three papers play different evidential roles: Gradient Starvation motivates
the feature-competition possibility; Correct-N-Contrast illustrates that
cross-instance alignment can change learned information; Rank-N-Contrast
provides the continuous-target representation mechanism. This is one direction,
not a proposal to combine their objectives or methods mechanically.

This would change representation learning from the instance input. It does
not rely solely on a fixed existing score vector. The supervision remains
the same candidate costs; the proposed difference is an inductive constraint,
not new labels or newly acquired information. Its use would cover the complete
candidate pool within each problem, rather than a local solver-versus-rest
classifier or a universal numerical geometry shared across unlike tasks.

The supplied failures already include representation/ranking variants and
training that could change features. They weaken a generic claim that simply
learning more features will solve the problem. The particular hypothesis here
is the value of explicit cross-instance performance-profile geometry. Without
implementation inspection, this note cannot claim that such relations are
absent from the existing system. It also cannot infer that previous failures
were caused by feature starvation.

## Preconditions And Risks

- **Observable signal:** relative solver advantage must depend on information
  present in the input and learnable across instances. Representation learning
  cannot manufacture unobserved deployment conditions or solver behavior.
- **Meaningful profile geometry:** similarities must reflect decision-relevant
  relative advantages. Raw overall problem difficulty, objective scale or
  one frequently disastrous candidate could dominate profile distance while
  contributing little to the actual best-method boundary.
- **Local decision detail:** smooth profile neighborhoods can still hide narrow
  or discontinuous winner boundaries. Aggressive alignment can remove exactly
  the distinctions needed for a useful solver switch.
- **Correct notion of invariance:** label-preserving transformations of the
  optimization problem need not preserve every deployed algorithm's behavior.
  No node-order, start-role, orientation or other execution invariance is
  assumed here.
- **Average cost remains a separate question:** the cited guarantees concern
  feature dynamics, classification groups or representation order. None
  proves lower aggregate solver cost, improved Top1, or a reason to choose
  every candidate somewhere.

There is sufficient primary-paper support to retain this as a high-level
direction. There is not sufficient project-specific evidence to call it the
cause of the residual phenomenon or to recommend immediate execution. No
additional result inspection, new solver run or modification of the ongoing
NSS work is requested by this literature note.
