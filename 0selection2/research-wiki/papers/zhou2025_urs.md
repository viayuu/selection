---
type: paper
node_id: paper:zhou2025_urs
title: "URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization"
authors: ["Changliang Zhou", "Canhong Yu", "Shunyu Yao", "Xi Lin", "Zhenkun Wang", "Yu Zhou", "Qingfu Zhang"]
year: 2025
venue: arXiv (ICML-sub)
external_ids: {arxiv: "2509.23413"}
tags: [unified-solver, vrp, zero-shot, masking, hypernetwork]
relevance: core
origin_skill: research-lit
---

# One-line thesis
Single model solves 110 VRP variants by unifying data representation (constraints → masks) + mixed-bias encoder + problem-conditioned parameter generator.

## Method
UDR: every node gets `{positional, unified-attributes, node-type}` — constraints never modify the input, only the masking function. Parameter generator adapts decoder weights to problem tag. Mixed Bias Module captures geometric / relational priors.

## Reusable Ingredients
UDR schema, mask-as-constraint design, param generator as problem conditioning.

## Relevance to This Project
Direct answer to challenge 2 (disambiguation) and challenge 3 (unified repr). Selector can sit on top of UDR encoder with a problem-id token.
