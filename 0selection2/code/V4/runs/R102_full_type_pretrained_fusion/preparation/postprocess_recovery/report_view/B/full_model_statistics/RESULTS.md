# R102/B

Unique VAL checkpoint: update 20160

| Split | Top1 (%) | mean_cost | actual regret (%) |
| --- | ---: | ---: | ---: |
| VAL | 71.132099 | 11.416322668 | 0.400845 |
| TEST | 71.225926 | 11.373165192 | 0.391959 |

SBS was fitted on new81000 TRAIN. Native FP64 winner labels; equal task weights.
NSS uses historical R79 weights, not a matched new90k training control.
No checkpoint, temperature, or per-problem assembly was selected using TEST/LIB.
PDF: MODEL_STATISTICS.pdf; distributions: selection_distributions.pdf.

R102 extends R94 donor coverage, not a new architecture family. Matched A/B differ ONLY in frozen TSP ELG / ATSP MatNet initialization: A extra_random, B full_pretrained. BOTH use the identical PRETRAINED R94B routing donor and existing routing cache for CVRP+15MVRP. All18 tasks; TSP128, ATSP FROM256+TO256=512, routing128; separate bias-free projections and affine-free norms;1941257 trainables=1842953 original+98304. Fresh R84 seed2 initialization/geometry, physical TSP node0 raw role, original four instance layers/two joint layers/hand34+ID/score MLP/maximin. Fixed ATSP identity onehot firstN roles are NOT historical RNG certification. Original R77 CE=.35, comparison=.35, risk=.02, R-Drop=.35, dropout=.1;57600 successful logical1024 updates, warm4320, VAL1440, TRAIN7200, AdamW1e-4 cosine1e-6, WD1e-4/clip1. Frozen donors are embedded in checkpoint; no external weight reads on restore. No decoder/routes, new labels or auxiliary loss. Native/FP64 ind winners, all legal pools, frozen R84 TRAIN SBS; ALL is equal-problem macro. Fixed R84 target36: genuine rescue/loss/new false picks and original-cost contributions, not selection-frequency improvement. R94B is a historical reference, not matched; unknown external-pretraining data/overlap and unmatched source-pretraining compute remain disclosed. Development-exposed TEST; one unique minimum full18 VAL macro_vs_sbs_pct checkpoint, first ties; no TEST/LIB tuning or per-problem checkpoint assembly. Native TSPLIB49/CVRPLIB100 and same-bank NSS4. Timing is GPU-resident batch128 amortized, NOT singleton or I/O-inclusive end-to-end: BOTH cached nco fields removed on every online call, donor hook proves execution for all18; cached-selector timing is a separate scope, never actual total.
