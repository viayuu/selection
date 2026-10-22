# R94 Online Timing Erratum

Status: MEASURED_ONLINE. Additive correction to frozen version cf2f06bc4.

The old nominal donor+selector branch retained nco_nodes/nco_node_mask from FrozenFeatureCache.batch. pairwise_objective.inputs_only removes labels but not those fields; R94.prepare_inputs calls the donor only when they are absent. Both old covered-task columns therefore timed cached-feature selector calls. Their difference is NOT donor overhead. The flags includes_native_encoder=true and donor_plus_selector_measured=true are withdrawn for that path.

Old JSON/CSV/PDF files are preserved byte-for-byte, including old timing numbers below. Their online-donor timing claims are superseded by this erratum. TSP/ATSP are raw-only by design; donor time is not applicable (zero executed calls).

No corrected GPU number exists unless marked MEASURED_ONLINE. CPU smoke numbers are never reported as GPU latency. Predictive metrics, checkpoint locks, selection rules and saved predictions are unchanged. A and B are both complete; the older B/ROUND_ANALYSIS.md statement that A was still running is historical. This erratum does not re-evaluate the pretrained-vs-random quality claim.

## Fixed Measurement Protocol
Each task: original TEST rows 0..127 in order, no label/winner/cost reads; one warmup and three repetitions per scope, same per-arm VAL-locked checkpoint. A update40320: 872315c1bded11e1de128c5006b683b47b9aa30b9314d793bb3d0e871e6b5c42; B update54720: 8e797150d22ea138db246536f36a1b7dfa6ef3b516ee065a3303a6b6e579c3ee.

Native donor: unmodified frozen FP32 ReLD encoder wrapper, including its validation/packing but excluding the FP16 cast. Selector: unmodified selector given resident FP16 features freshly produced by that donor, not a disk sidecar. Total: separately measured unmodified model(raw), including donor, original FP16 quantization and selector. Total is not the sum of independent means.

Synchronized perf_counter wall time; GPU inputs already resident. Integer donor and native-encoder forward-hook counters verify each warmup/repetition (one encoder call per unique real node length); hook overhead is included. Output/equivalence checks are outside the timed span. Batch time /128 is amortized throughput cost, NOT single-query latency. Excludes dataset/checkpoint I/O, hashing, parsing/collation, FP64 local-geometry preparation, CPU-to-GPU transfer, warmup and post-timing validation. No solver decoder or full-system/I/O-inclusive end-to-end latency.

Preprocessing exclusion includes CPU reconstruction of the original input fields and frozen FP64 local geometry because the old container /tmp caches are not available here. No new donor feature cache is prepared or read.

## Preserved And Corrected Values
All values below are ms/instance. PENDING means unmeasured, not zero. Do not infer an A/B speed advantage from the old hardware-dependent numbers.

| Arm | Problem | Old nominal (cached) | Old cached | Native donor | Selector | Online total |
| --- | --- | --- | --- | --- | --- | --- |
| A | TSP | 0.403637 | 0.393278 | 0.000000 | 0.262299 | 0.256919 |
| A | CVRP | 0.488799 | 0.488662 | 8.714714 | 0.319110 | 9.046175 |
| A | ATSP | 0.071193 | 0.070898 | 0.000000 | 0.066141 | 0.065667 |
| A | OVRP | 0.073456 | 0.072977 | 3.576302 | 0.070246 | 3.637875 |
| A | VRPB | 0.073150 | 0.072927 | 3.979869 | 0.070200 | 4.072113 |
| A | VRPL | 0.073301 | 0.072990 | 4.034958 | 0.070244 | 4.130403 |
| A | VRPTW | 0.073165 | 0.073401 | 3.918851 | 0.070102 | 3.976312 |
| A | OVRPTW | 0.073171 | 0.073107 | 3.754352 | 0.070392 | 3.821210 |
| A | OVRPB | 0.073385 | 0.072881 | 3.993343 | 0.071334 | 4.095717 |
| A | OVRPL | 0.073080 | 0.073038 | 4.047617 | 0.070196 | 4.120938 |
| A | VRPBL | 0.073631 | 0.073041 | 3.725867 | 0.070370 | 3.799032 |
| A | VRPBTW | 0.072915 | 0.072817 | 4.078873 | 0.070244 | 4.144940 |
| A | VRPLTW | 0.073276 | 0.073000 | 4.003690 | 0.070389 | 4.145005 |
| A | OVRPBL | 0.072864 | 0.072992 | 4.122929 | 0.070417 | 4.137808 |
| A | OVRPBTW | 0.073097 | 0.072877 | 3.924431 | 0.070520 | 4.071150 |
| A | OVRPLTW | 0.073228 | 0.073196 | 3.848053 | 0.070674 | 3.908250 |
| A | VRPBLTW | 0.072942 | 0.072968 | 3.643619 | 0.070311 | 3.712529 |
| A | OVRPBLTW | 0.073125 | 0.073002 | 3.998434 | 0.071636 | 4.059226 |
| B | TSP | 0.262168 | 0.261500 | 0.000000 | 0.262171 | 0.257076 |
| B | CVRP | 0.317999 | 0.317428 | 9.141558 | 0.321195 | 9.441915 |
| B | ATSP | 0.064552 | 0.064129 | 0.000000 | 0.065647 | 0.065796 |
| B | OVRP | 0.068539 | 0.068381 | 3.731810 | 0.069801 | 3.787790 |
| B | VRPB | 0.068998 | 0.068472 | 4.170819 | 0.070203 | 4.206541 |
| B | VRPL | 0.068735 | 0.068517 | 4.208422 | 0.069823 | 4.272145 |
| B | VRPTW | 0.068802 | 0.068693 | 4.105517 | 0.069673 | 4.154369 |
| B | OVRPTW | 0.068672 | 0.068616 | 3.893741 | 0.070583 | 3.980544 |
| B | OVRPB | 0.068922 | 0.068748 | 4.141208 | 0.070093 | 4.213735 |
| B | OVRPL | 0.068784 | 0.068531 | 4.219307 | 0.070181 | 4.283638 |
| B | VRPBL | 0.069443 | 0.068753 | 3.861358 | 0.069855 | 3.936100 |
| B | VRPBTW | 0.068777 | 0.068460 | 4.243019 | 0.069919 | 4.321653 |
| B | VRPLTW | 0.068952 | 0.068537 | 4.135353 | 0.069791 | 4.212814 |
| B | OVRPBL | 0.068727 | 0.067991 | 4.238570 | 0.069785 | 4.277524 |
| B | OVRPBTW | 0.068671 | 0.068324 | 4.105317 | 0.070499 | 4.143570 |
| B | OVRPLTW | 0.068635 | 0.068589 | 3.968782 | 0.070234 | 4.057266 |
| B | VRPBLTW | 0.068858 | 0.068624 | 3.832422 | 0.069981 | 3.876881 |
| B | OVRPBLTW | 0.068739 | 0.068639 | 4.055712 | 0.069986 | 4.133151 |

## Scope And Provenance
Instrumentation: only diagnostics/r94_online_timing_recovery.py and files inside timing_correction/. Frozen r94 sources are unchanged. No training, model selection, new predictions files, Git commit or publishing. legacy_timing_preserved.json retains both complete old receipts and hashes; preservation_before.json hashes the original sources/checkpoints/reports and checks size/mtime identities of prediction arrays without rereading their payloads.
B/ROUND_ANALYSIS.md is reused only as evidence for the call-path error and unchanged quality-evidence boundary. No external reviewer/model was invoked; this script makes no claim about verified gpt-6.1-sol routing.
