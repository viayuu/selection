# Assumptions

ASK mode: never. Latest user request explicitly resumes implementation and
authorizes two RTX3090 plus two RTX4090. Earlier preparation-only scope is
superseded for the work needed to complete this request.

| ID | Decision | Chosen | Class | Source |
| --- | --- | --- | --- | --- |
| A01 | Dataset version | Frozen routing new90k in place; accepted nonrouting v2 in place; missing data and labels in a separate 20261029 supplement | semantic | user/default |
| A02 | Task scope | All 18 routing tasks and six named nonrouting families; readiness is reported per task | semantic | user |
| A03 | Deployment actions | Keep every L2D/DAN checkpoint as a distinct selectable deployment; never take a free per-instance minimum across checkpoints | semantic | source protocol |
| A04 | Native objectives | MIS/MKP maximize; other new tasks minimize. Preserve FP64 objective values, observed masks and all exact best candidates | semantic | source protocol |
| A05 | Missing labels | Missing candidates remain unobserved; never fill with zero or silently change a query's legal pool after solving | semantic | source protocol |
| A06 | Evaluation | Routing retains native-winner Top1 and cost metrics. New tasks report any-best Top1, explicit canonical Top1, native objective and directed regret. Zero-optimum examples have separate absolute-regret reporting | semantic | default |
| A07 | Input interface | Typed graph, operation, resource and item fields; labels and solution certificates stay outside the forward-input namespace | interface | default |
| A08 | Data completion | Implement necessary generators and candidates in the new supplement; public benchmarks remain external evaluation and do not become synthetic TRAIN | semantic | user/default |
| A09 | Model selection | TRAIN updates parameters; a single predeclared validation rule selects a unique checkpoint before full TEST/LIB. No per-task checkpoint assembly | semantic | user |
| A10 | Initialization | Seed 2, fresh trainable parameters; existing frozen input statistics/caches may be reused, trained selector weights are not scratch initialization | semantic | user |

| A11 | Architecture | Astra's typed relational selector: typed raw entity/factor adapters, four shared relation/global layers, two shared node/candidate interaction layers, shared full-candidate scoring. Keep routing distance/matrix and geometry inputs | semantic | Astra HANDOFF 20261029 |
| A12 | Deployment representation | Separate deployment ID and method family embeddings; routing's 34 known features remain available with an explicit availability bit; new methods do not acquire fabricated routing features | interface | default |
| A13 | New objective | Set-selection CE (routing uses native winner), centered complete-candidate advantage Huber with weight 0.25, and expected directed TRAIN-scale excess with weight 0.10. No independent route/process supervision | semantic | Astra/default |
| A14 | Objective unit | Each task uses its TRAIN-fitted SBS mean native objective as its fixed positive loss scale. A zero SMTWTP scale falls back to TRAIN mean[(sum p)(sum w)]. Neither is reported as oracle-relative regret | semantic | Astra HANDOFF 20261029 |
| A15 | Training mixture | Uniform task cycles; each task keeps its own natural shuffled stream with no winner balancing. Small task populations repeat more often, explicitly reported | semantic | default |
| A16 | First full training | Seed 2, logical batch 1024, at most 57600 successful updates, 4320-update warmup, AdamW 1e-4 to 1e-6 cosine, weight decay 1e-4, dropout 0.1; validation every 1440 updates; patience six validations after warmup | semantic | default |
| A17 | Checkpoint rule | Maximize full-pool macro any-best validation accuracy, then minimize macro TRAIN-scale excess, then keep the earliest update. Also report routing-native Top1 independently | semantic | default |
| A18 | Hardware | One distributed model may use both allocated GPUs on gpu03 and both on gpu05; logical batch stays 1024 across all ranks. Use measured per-domain microbatch limits | interface | user/default |
| A19 | Routing arc values | All real directed ATSP arcs and all real coordinate distances also enter each layer's gated value/message sum, alongside the retained original attention bias. Dense aggregation uses exact matrix products; no nearest-neighbor pruning | semantic | Astra/default |
| A20 | Nonrouting execution roles | Preserve input-order priorities used by fixed native tie rules: vertex/item/job ranks, and scheduling job/machine ranks. These are numeric input execution roles, not instance IDs or learned arbitrary positional tables. Explicit roles can be carried through a pure relabeling | semantic | native rules and Astra input-role boundary |
| A21 | Distributed network transport | Run the full four-rank model inside existing Slurm GPU steps on the allocation hosts, using both assigned GPUs per node and host bond0 networking. Preserve the zero-update container-rendezvous attempt; reuse frozen input-cache files on node-local storage, without generating inputs or labels again | interface | observed bridge-network isolation; original R76 host DDP transport |

The Astra scientific handoff is
`docs/research/multidomain_selection_design_20261029/HANDOFF.md`.
An existing minimum-size readiness task is not permission to describe a
partial pool as complete. The new model/objective is a multidomain baseline,
not a matched claim of architectural improvement over R94B.
