---
type: idea
node_id: idea:selective_abstaining_selector
title: "Selective / Abstaining Unified Selector for OOD Routing Instances"
stage: proposed
outcome: unknown
based_on:
  - paper:nss_anchor
  - paper:urs_anchor
target_gaps:
  - gap:G1
  - gap:G2
created_at: 2026-04-15T19:13:45+08:00
updated_at: 2026-04-15T19:13:45+08:00
---

# Hypothesis

On benchmark/OOD routing instances, the right selector should not always force a top-1 decision; uncertainty-aware fallback to top-k or a safe solver can reduce costly confident mistakes.

## Proposed Method

Calibrate the current selector's confidence and add a selective decision rule: top-1 when confident, otherwise abstain to a safe solver or a small top-k shortlist.

## Expected Outcome

- Better benchmark cost under controlled inference budget.
- Risk-coverage curves that expose OOD brittleness more honestly.

## Why This Idea Matters

The current benchmark weakness may stem from wrong but confident decisions. This idea turns unified selection into a risk-controlled decision system rather than only a forced classification/ranking model.

## Cheapest Validation

Use the current trained model logits, perform post-hoc calibration, and evaluate fallback strategies on CVRPLIB and other benchmark subsets.

## Main Reviewer Objection

This may be viewed as a selection strategy paper rather than a core representation-learning contribution.

## Failure Notes / Lessons

Best used as an inference-time companion to the stronger invariant-preference main direction, not necessarily as the sole paper headline.
