---
type: paper
node_id: paper:sheng2021_star
title: "One Model to Serve All: Star Topology Adaptive Recommender"
authors: ["Xiang-Rong Sheng", "et al."]
year: 2021
venue: CIKM
tags: [recsys, multi-domain, star]
relevance: related
---

# One-line thesis
Shared centre network + per-domain tower whose weights are element-wise added onto the centre — cheap, trainable per-problem specialization.

## Relevance to This Project
Cheapest structural fix for 18-problem heads: one shared selector MLP + 18 small STAR-style residual towers.
