# R102 Postprocess Recovery: Engineering Review Round 01

## Judgment

Engineering score: **9/10**. Verdict: **ready for the parent's explicitly named second preflight attempt**, not an automatic retry of the failed entry. One actual P2 was found after the initial CPU-only review, patched within the user's expanded ownership, and covered by a targeted CPU regression test. No additional concrete P1/P2 was found in the bounded helper review.

Requested reviewer: Sol/max only. This is a same-family provisional review; the requested model identity and reasoning setting were not independently attested. No secondary agent or research reviewer was used. This is not efficacy evidence or a claim that TEST, timing, LIB or delivery completed.

## Priority Findings

### P1: None Found

No concrete prediction, protocol, checkpoint-state or original-evidence corruption issue was established in the reviewed recovery path.

### P2: Actual UUID Preflight Rejected Textual Representations

**Status: minimal helper patch applied; CPU regression passed; actual second GPU preflight remains parent-owned.**

The pre-fix `r102_postprocess_recovery.py:387-388` compared `str(props.uuid)` directly against the allocation's `GPU-...` string. For example, `71479e7e-f8bd-e430-66ed-f8db9f0d47be` and `GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be` identify the same complete UUID but failed this comparison. Both actual `A_gpu.log` and `B_gpu.log` terminate at that comparison. Their separate `gpu_entry.json` files are failed/rc1, and the arm directories contained only those entry receipts, not `origin.json`, shadow or TEST output. The logs do not expose the raw Torch UUID, so the exact observed representation is not independently established here; environment equality alone is not hardware-identity proof.

The authorized patch adds `normalize_gpu_uuid`, parsing with standard-library `uuid.UUID` after removing only an optional leading `GPU-`. `require_gpu` compares complete parsed identities, still rejects another GPU, and records the unmodified Torch UUID string as `hardware.torch_uuid_raw`. The exact CUDA visibility, expected UUID, CPU affinity, container, one-visible-device and actual-device-property checks remain. No donor, numerical setting, model factory, input, inference rule or original source pin was changed.

The added CPU-only regression invokes the real `require_gpu` with mocked CUDA properties and checks prefixed/bare/case-varied equivalent identities, raw-value recording, a different full UUID, truncated UUIDs and foreign/repeated prefixes. The existing CUDA initialization tripwire remains active. This closes the missing successful actual-property-check coverage without using a real GPU.

**Parent handoff:** preserve the failed canonical `gpu_entry.json` files and logs verbatim. Use a newly named, explicitly authorized preflight continuation receipt, linked to the failed attempt and unchanged original lock/evidence. Parent owns the corresponding entry/status/completion-consumer integration. Merely adding an unused receipt name is insufficient: the reviewed admin otherwise consumes canonical `gpu_entry.json` and counts its failed state as a current error. This reviewer did not change administration or launch a continuation. No general retry/resume behavior is authorized.

## Bounded Core Checks

- `validate_proof` and `HeaderCompatibility` retain original schema/legal-pool validation, all task/row/typed-array constraints and exact proved header values. Only `created_utc`, `completed_utc`, `seconds`, `test_lock_root` receive the process-local cross-node exception; `RUNTIME_FIELDS` and original source files are not edited. The supplied actual proof covers 107 arrays / 1,000,309,696 bytes per node, including raw inputs, geometry, FP64 costs and native winners. Main supplied the successful actual `describe` validation; this review did not rehash payloads or regenerate costs.
- `shadow` copies only bounded metadata, symlinks original `best.pt` and the read-only-use TRAIN/VAL archives, omits old failure/first-TEST markers and does not symlink `evaluation/test`. `runtime_args` retains canonical original runtime authorization and redirects new TEST sidecars only. The original engine writes its TEST/status/final-metrics outputs into the new view, not original arms.
- `original_records`, `verify_origin` and strict restore bind original source pins, both 57600-update completions, unchanged failure/worker/controller/watcher evidence, A update33120/SHA10694b4d0fed..., B update20160/SHA7df434c762a0..., and the selected VAL full model state including buffers. Existing null VAL capture-time checkpoint SHAs are disclosed, not fabricated or rewritten.
- The original TEST, timing and LIB implementations run in that order. TEST Torch setup and separate-entry timing defaults are retained. Online/cached alignment is observational and fails on feature, mask, valid-score or pick drift; it does not substitute inference or regroup inputs. No TRAIN call, optimizer construction, checkpoint reselection, solver or new labels are introduced.
- Existing first-error and stage-start guards preserve failed/partial attempts. Successful science receipts bind the original checkpoint/protocol, outputs and private view; report/compare consume saved evidence. The finalized administrative entry has no readiness-JSON or publisher prerequisite for GPU work. The earlier WIP two signature TypeErrors were fixed and are not retained as findings.
- Reviewed resource checks retain original job1475/gpu05 container, the two original 4090 UUIDs, CPU quartets A0,1,2,3 and B4,5,72,73, and the existing GPU mutex inode. Main owns CPU4/mem9G step allocation. This review did not inspect or touch healthy R105 work.

## Verification And Provenance

All shell commands used the activated `easynco` environment after shell initialization. No Git command, allocation, GPU probe/launch, production inference or dataset edit was performed by this reviewer. The one new regression was added only after the actual finding and explicit ownership expansion.

```text
Before actual failure: existing helper 15/15 CPU tests passed.
Final original combined suites: 30/30 passed in 1.002s; wrapper bash -n passed.
After UUID patch: targeted CPU regression passed in 0.012s.
After UUID patch: combined helper/admin suites 31/31 passed in 0.963s.
```

Post-patch source SHA256:

```text
r102_postprocess_recovery.py
728fba9b75bd79687b3b6225fe1cd4869823a5c79ed4d45fb78a4f35f52a9fd3
test_r102_postprocess_recovery.py
d35b4bb493d11b26f35ca86cf7e53dd336f6a98676d48bda1a044528229f13dc
```

Failed entry SHA256 values matched before and after the patch:

```text
A/gpu_entry.json a3daf752fb9f13646a28ee86fc1622159c2f99ffbbeed3c56439ca1071cb3a01
B/gpu_entry.json 560c466c1ffc33563912bee86225e5b2c30153160735eb6a31f2e4007af71de8
```

## Warnings, Not Additional Gates

The CPU suites mock GPU dispatch and do not establish actual post-patch UUID acceptance, numerical timing alignment or completed TEST/LIB/PDFs. Those remain normal main-owned execution evidence, not a demand for another audit or production-PDF wait. No further tests, hash framework, metadata hardening or research review is requested before the authorized next preflight.

The earlier verbal ready judgment was superseded by the real UUID failure, then narrowed to the patched helper and explicit parent-owned continuation described above. Original training and the initial failed TEST attempt are not reclassified as successful.

Reviewer writes are limited to this review, sibling `REVIEWER_MEMORY.md`, and the explicitly authorized UUID helper/`require_gpu` change plus its targeted test. Administrative implementation and every original/failed receipt remain outside this reviewer's write ownership.
