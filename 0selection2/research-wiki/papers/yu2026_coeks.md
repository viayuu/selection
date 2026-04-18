---
type: paper
node_id: paper:yu2026_coeks
title: "Combination-of-Experts with Knowledge Sharing for Cross-Task Vehicle Routing Problems"
authors: ["Zikang Yu", "Jinbiao Chen", "Jiahai Wang"]
year: 2026
venue: ICLR
tags: [moe, constraint-experts, ood, vrp]
relevance: core
---

# One-line thesis
One FFN expert per basic constraint (C/O/B/L/TW); active experts combined per task; mutual distillation shares general knowledge.

## Reusable Ingredients
Combiner softmax over active experts; MDis auxiliary loss; plug-in new experts for new constraints.

## Relevance to This Project
Structural template for "one head per candidate solver, activate subset with mask"; MDis transfers neatly to "shared knowledge across selectors for overlapping problems".
