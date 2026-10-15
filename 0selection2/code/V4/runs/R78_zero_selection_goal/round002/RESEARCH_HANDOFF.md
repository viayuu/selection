# R78 Round002 Results

Outcome: **inconclusive**; selected rule: **identity**.
Frozen R76 epoch18; FP64 cost gaps and centered scores; K-1 intercepts per task.
Natural uniform-with-replacement TRAIN sampling, seed2, projected SVRG, exactly1024 rows per parameter update.
Margin1, score multiplier1, prior1, task-wide TRAIN c, eta=1/(8L); no scans or clipping.
Full-gradient snapshots and TRAIN quadratic references are not optimizer updates or deployed candidates.

| Split | Rule | Top1 % | CE (descriptive) | Mean cost | Actual regret % | VS TRAIN SBS % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | identity | 69.369646 | 0.752703080 | 11.373004840 | 0.460650092 | -0.143845417 |
| TRAIN | candidate | 40.849743 | 1.428347126 | 11.653491213 | 2.826220296 | 1.982256025 |
| VAL | Not evaluated | - | - | - | - | - |

| Task | Updates | Snapshots | L | Projected gradient | Reference error | Exact objective gap | Converged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TSP | 29000 | 30 | 24.014952309904526 | 3.9308426874606894e-08 | 4.418386548854514e-07 | 1.0136041105508459e-14 | True |
| CVRP | 30000 | 31 | 32.1882984332568 | 9.754324442931561e-06 | 0.0001818516011824478 | 1.1734727479401422e-09 | False |
| ATSP | 7000 | 8 | 4.699804018452712 | 9.084236106404742e-10 | 9.452442828461471e-09 | 6.2370509416357174e-18 | True |
| OVRP | 30000 | 31 | 22.62877579395495 | 4.820809839573004e-06 | 0.00012933978147611214 | 4.746028848078029e-10 | False |
| VRPB | 10000 | 11 | 24.70095236970742 | 6.26256677904724e-08 | 2.056001888739445e-07 | 1.0238768234773994e-14 | True |
| VRPL | 30000 | 31 | 18.64494162284708 | 1.6291630754727837e-06 | 4.750621927052112e-05 | 6.750093720788918e-11 | False |
| VRPTW | 8000 | 9 | 3.455589435867823 | 1.8335272345543219e-09 | 2.8463388090926856e-08 | 3.221782523676204e-17 | True |
| OVRPTW | 9000 | 10 | 3.703564379464218 | 5.337374899688774e-09 | 9.2252675720772e-08 | 3.038189218343745e-16 | True |
| OVRPB | 13000 | 14 | 29.70598755917515 | 7.598787220286507e-08 | 2.711394411214485e-07 | 1.6756389993463432e-14 | True |
| OVRPL | 30000 | 31 | 17.445548148589328 | 4.5401718443138783e-07 | 1.1391802237081805e-05 | 4.020280802869648e-12 | False |
| VRPBL | 9000 | 10 | 21.737651716899915 | 1.761675978012578e-08 | 5.446580808587953e-08 | 7.433340217057601e-16 | True |
| VRPBTW | 6000 | 7 | 9.757634768407122 | 1.4805397775818907e-09 | 5.942370151501564e-09 | 7.76482481565688e-18 | True |
| VRPLTW | 8000 | 9 | 3.5102182587666837 | 2.6769473027155044e-09 | 4.186171675968353e-08 | 6.954941925047308e-17 | True |
| OVRPBL | 9000 | 10 | 20.84355375400881 | 3.921799476142018e-08 | 1.3178116820000696e-07 | 4.124519768105393e-15 | True |
| OVRPBTW | 5000 | 6 | 7.154278266583421 | 1.3234283885790732e-09 | 5.995727470065049e-09 | 7.270743561555507e-18 | True |
| OVRPLTW | 9000 | 10 | 3.6763685932842396 | 5.204658337122636e-09 | 9.043835458300009e-08 | 2.9044623726174074e-16 | True |
| VRPBLTW | 6000 | 7 | 10.08107792348183 | 2.7291022219877448e-09 | 1.0266127947744508e-08 | 2.2358788137865037e-17 | True |
| OVRPBLTW | 5000 | 6 | 6.412341050947638 | 2.010812618843949e-10 | 8.312444066405078e-10 | 1.4386101184288646e-19 | True |

Converged tasks: 14/18. TEST opened: False.
Stopping requires two successive TRAIN checks: projected gradient<=1e-5, objective change<=1e-7, reference error<=1e-6 and quadratic-gap agreement at FP64 rounding tolerance.
Numerical budget exhaustion is inconclusive, not rejection of this fully converged objective. If any task fails, the complete candidate is not evaluated on VAL.
Native macro Top1 must strictly improve; raw macro mean cost and actual regret must not increase. VS-SBS and NLL are descriptive only.
All18 task details, method comparisons and signed changed-decision costs are saved beside this report. Upper-decile summaries, when present, use the worst ceil(0.1*N) regrets under each rule with stable row-order ties.
No temperature, margin, prior, optimizer or tolerance was adapted using VAL. No TEST/LIB labels were used for fitting or selection.
The continuing file-based goal stays active; this report is not goal completion or an independent confirmation claim.
