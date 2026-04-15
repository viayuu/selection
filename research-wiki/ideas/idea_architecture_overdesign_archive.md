---
type: idea
node_id: idea:architecture_overdesign_archive
title: "Archive: Heavy Expertized Unified Selector as Main Direction"
stage: archived
outcome: negative
based_on:
  - paper:urs_anchor
  - paper:routefinder_anchor
target_gaps:
  - gap:G1
created_at: 2026-04-15T19:13:45+08:00
updated_at: 2026-04-15T19:13:45+08:00
---

# Hypothesis

A significantly larger shared/private expertized selector architecture (MoE, adapters, PLE/STAR/MMoE-style routing, heavier solver-conditioned interaction) would materially improve the current project.

## Proposed Method

Use expert decomposition, domain-specific adapters, heavier conditional routing, and more complex interaction mechanisms as the paper headline.

## Expected Outcome

Originally expected to reduce negative transfer and improve all problems jointly.

## Why This Idea Is Archived

Current project evidence does **not** support making this the next main direction:

- the simple unified selector is already strong on IID test;
- more complex conditionalization did not clearly and stably outperform the simple baseline;
- the main remaining weakness is benchmark/OOD, especially CVRPLIB, which architecture inflation does not directly explain;
- as a paper story, this now risks looking like overdesign or borrowed recommender-system machinery.

## Main Reviewer Objection

"Why is a much bigger architecture necessary if the simple pooled selector already works and the real issue is OOD transfer?"

## Failure Notes / Lessons

**Failure:** treating architecture size/complexity as the main next-step novelty.

**Lesson:** the next paper should focus on transferability of solver preference, hard competitive subsets, and benchmark robustness — not on a larger selector trunk.
