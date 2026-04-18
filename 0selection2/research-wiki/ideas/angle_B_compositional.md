---
type: idea
node_id: idea:angle_B_compositional
title: "Constraint-compositional zero-shot selector"
origin_skill: research-lit
stage: proposed
tags: [zero-shot, compositionality]
---
# Idea
Train on some MVRP constraint combinations, test on unseen combinations. Selection generalizes compositionally even when underlying solvers do not.

## Target gaps
- G3' (constraint-compositional zero-shot selection)

## Experiment sketch
Hold out specific constraint combinations (e.g. OVRP+TW, OVRP+MD); compare problem-ID, constraint-bitvector, DSL-prompt conditioning.
