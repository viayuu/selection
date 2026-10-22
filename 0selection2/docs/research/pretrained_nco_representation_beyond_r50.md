# Frozen NCO Node Representations Beyond R50

Date: 2026-10-22. Paper/idea research only. Project history is supplied by Sol, not independently reanalysed. No local code/results, checkpoints, or source adapters inspected; no experiments proposed or executed.

## Judgment

**Conditional yes: early node-level representation fusion is meaningfully different from R50, but substantial improvement remains unsubstantiated.** Prioritize one mechanism: frozen, task-compatible NCO encoder features fused with raw attributes **before** the selector's trainable inter-node processor. RouteFinder is the strongest architectural motivation among these sources, not a verified universal donor. This is a materially different transfer hypothesis, not a claim that feature fusion itself is novel.

## Five Primary Sources

Full-text evidence, with the relevant sections distinguished from our inference:

| Source | Evidence and limitation |
| --- | --- |
| Berto et al., [RouteFinder: Towards Foundation Models for Vehicle Routing Problems](https://arxiv.org/html/2406.15007v4), TMLR 2025; Sec. 4.2, Apps. B.2, D.6 | The static transformer processes customer coordinates, demands, time windows/service times and depot/global attributes, including O/L. Layer-wise t-SNE separates variants increasingly with depth. This supports constraint-conditioned representations, **not** within-task winner discrimination or frozen transfer. Adaptation to new attributes involves fine-tuning, not evidence for an unchanged universal encoder. |
| Zhou et al., [MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts](https://arxiv.org/html/2405.01029v2), ICML 2024; Sec. 4.1, Apps. A, B.1-B.2 | Attention/MoE encoders contextualize coordinates, demands and time-window endpoints; six trained variants generalize to ten others as full solvers. Open-route and route-state information enter the decoder. Consequently, identical static inputs cannot acquire different O/L-conditioned embeddings merely because the complete solver supports those tasks. Expert specialization is not evidence of algorithm-selection knowledge. |
| Huang et al., [Rethinking Light Decoder-based Solvers for Vehicle Routing Problems](https://proceedings.iclr.cc/paper_files/paper/2025/file/447ac93bf22099aa346a45577376492d-Paper-Conference.pdf), ICLR 2025; Sec. 2.3, Tables 1-2, Apps. F-G | POMO/AM static embeddings remain useful for subproblem solving after encoding additional irrelevant nodes; richer downstream decoding can help. However, decoder-only fine-tuning underperforms encoder fine-tuning on larger instances, and extra encoder depth does not reliably improve generalization. Thus reusable information exists, but neither sufficiency nor easy extraction follows. These are solver experiments, not frozen-selector transfer. |
| Luo et al., [Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/file/1c10d0c087c14689628124bbc8fa69f6-Paper-Conference.pdf), NeurIPS 2023; Secs. 3.1-3.2 | LEHD has one static attention layer and a deep decoder that re-embeds available nodes with endpoint context. Its TSP/CVRP performance does not establish a strong frozen encoder. The encoder is not knowledge-free, but discarding the decoder discards much of the learned computation. |
| Drakulic et al., [BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/file/f445ba15f0f05c26e1d24f908ea78d60-Paper-Conference.pdf), NeurIPS 2023; Secs. 3.2, 4 | BQ applies its network to successive residual problems, with origin/destination markers; it has no separate deep, once-per-instance encoder. ATSP uses directed costs through graph convolution rather than Euclidean coordinates. Calling its policy network a transferable static encoder obscures this distinction. No residual-state extraction is proposed here. |

## One Mechanism

Conceptually:

```text
H = stop_gradient(E_pretrained(X_native))
Z_i = fuse_trainable(raw_node_i, H_i)
scores = existing_method_interaction(trainable_instance_processor(Z), method_attributes)
```

Retain node alignment and raw/global attributes; fuse **before the first trainable inter-node layer**, not after pooling. This reuses the existing four-layer instance processor, rather than proposing additional depth. It can learn relations using both raw evidence and externally learned contextual coordinates, although its existing raw-input capacity may make those features redundant. Existing method descriptors remain available. No decoder, route prefix, action logits, trajectory targets, generated instances, or new solver-cost labels are involved; previous selector checkpoints are not pretraining sources.

**Research inference:** route-trained attention may organize spatial-demand-time interactions into features that make solver-sensitive configurations easier to learn from existing labels. The downstream processor can recombine those features differently from the donor's route policy. Neither interpretable motifs nor candidate-specific failure signatures have been established by these papers. Variant recognition alone would be redundant with known task attributes.

Because frozen H is a deterministic transformation of the input given external weights, this adds an **inductive bias**, not new instance information or inference-time access to solver costs. It is not data-budget-matched to scratch: the weights embody external pretraining data and computation.

## R50 Boundary

Sol's R50 used two frozen ReLD encoders on old, single-task OVRPTW MOEL-versus-MTL data, with node MLP, mean/max, depot and size readout. Random-versus-pretrained controls found partial validation signal, but not the required improvement on both held-outs; rapid near-perfect training accuracy demonstrated serious overfitting.

That result is adverse evidence against easily transferable winner information, **not proof that deeper processing will fix it**. R50 already contained contextual node features; the proposed difference is **supervised inter-node recomputation on raw-plus-pretrained features before compression**, not discovering graphs or adding an MLP. Full18, more data, another donor, or the word "foundation" alone would be scope changes/rebranding. Pool-first fusion remains inside the failed boundary. No route-successor, partial-cost, reconstruction, or prefix mechanism is revived.

## Compatibility and Limits

| Task group | Honest coverage |
| --- | --- |
| CVRP + 15 MVRP | RouteFinder/MVMoE cover corresponding attribute families, not automatically identical semantics. RouteFinder explicitly distinguishes its backhaul convention from prior MVMoE-style definitions (App. A.2). Therefore matching task names does not certify all 16 tasks. Frozen legal candidate pools remain unchanged. |
| TSP | LEHD provides native TSP representations, with the weak-static-encoder caveat above. Turning TSP into an artificial CVRP is not established TSP pretraining transfer. |
| ATSP | Requires directed-matrix-aware features. ReLD App. F describes a separate MatNet-derived encoder using both D and its transpose; this is **not** its VRP checkpoint. BQ's ATSP support likewise does not supply a shared VRP encoder. |

These are architectural possibilities, not verified available-weight compatibility. The complete18 selector must retain its raw-input path wherever no suitable donor exists; claim partial pretrained coverage, not one public model covering all18. If pretrained coverage of every task is mandatory now, this review cannot justify that claim.

**Representation correctness:** compute on the true node set, including actual depots, using native field meanings, units/normalization, capacity, service times, limits and disabled-constraint conventions. Missing fields cannot be fabricated. Padding is acceptable only if equivalent to exact-N execution throughout attention, node-set normalization and any shared MoE routing; cropping H or masking only final pooling is insufficient. This follows from the operations above, not a reported padding benchmark. ReLD's irrelevant-node robustness is not padding invariance.

**Hard ceiling:** solver-quality features need not distinguish competing solvers; freezing cannot recover discarded information, and greater downstream capacity can worsen R50-style overfitting. Domain shift, redundant cues and additional encoder inference cost remain real risks. Nothing establishes improved rare-winner recall, Top1 or regret.

**Recommendation:** retain early node fusion as one defensible next research direction, conditional on compatible, route-free static representations. Reject any implementation that reduces it to a pooled readout or presents broader scope as evidence of gain. The rationale is substantial; the expected gain is unknown.
