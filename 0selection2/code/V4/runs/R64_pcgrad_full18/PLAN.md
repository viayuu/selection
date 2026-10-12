# R64 adopted larger real-batch restart

Both arms restart fresh at seed2. Old run is aborted and immutable: `/public/home/shiys/0selection2/code/V4/runs/R64_pcgrad_full18_aborted_batch42_20261012T140928`.
Adopted amendment: `/public/home/shiys/0selection2/.aris/traces/R64_large_batch_plan.response.md`; verbatim response stored alongside this plan.
Astra/max supplied text/context planning, not a direct local-code audit.

Locked per-task batch 256, effective batch 768; 40 all18-task cycles, six triples each, 240 updates/epoch. Final task tail 16, final effective batch 48. All180000 rows/epoch retained.
Warmup/R-Drop ramp 720 successful updates (three epochs). Max40 epochs/9600 updates/7.2M presentations per arm. The reduced optimizer-update budget is part of this amendment.

Unchanged LR1e-4, AdamW WD1e-4/betas(.9,.999)/eps1e-8, global clip1, nativeCE + .35 symmetric R-Drop; no microbatching/accumulation or automatic LR scaling. Same R45A architecture, original full18/fullpools/native winners and directed ATSP inputs. A averages; B alone projects the adopted shared dense prefixes. A controls actual LR/horizon; B replays exactly. Integer validation correct counts choose best; exact ties earlier, official arm tieA.

Selection used only the predeclared bounded TRAIN capacity/throughput pass, two warmup plus three timed complete updates per arm/size, at most three sizes. Eligibility requires at least15% physical memory headroom; highest PCGrad throughput with3% timing ties toward smaller batch. All calibration states/updates are discarded and costed separately.

Full validation every epoch; eval-mode full TRAIN every5 and termination/selected checkpoint. No automatic TEST/analysis/git/review gate. Parent freezes both completed validation-selected models, inference and official arm before authorizing one full TEST/model; existing R45A/R60B tests reused. R58 and unresolved legacy B semantics remain deferred; single-solver70% goal active.
