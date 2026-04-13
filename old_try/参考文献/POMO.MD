A richly detailed academic paper network architecture diagram in the modern style of top-tier ICLR / NeurIPS / ICML 2024-2025 figures. The figure illustrates an action-conditioned Neural-LinUCB architecture for one-shot initialization method selection on two combinatorial optimization problems: TSP and CVRP.

CANVAS: pure flat white (#FFFFFF), no gradient, no tint, no grey background.

PANELS: 3 large rounded white panels with very subtle soft drop-shadow only (3px blur, 1px y-offset, rgba(0,0,0,0.06)), floating on the white canvas. Rounded corners about 20px. No thick borders. No nested boxes. Elements float directly on the panel surfaces.

FONT: friendly rounded geometric sans-serif (Nunito / Poppins), semi-bold titles, regular body text, italic serif for formulas. All text must be in English, publication-ready, concise, and readable.

COLOR SCHEME: Warm ML pastel palette.
Soft blue #BBDEFB with border #90CAF9,
soft pink #FFD0D0 with border #EF9A9A,
soft yellow #FFF3C4 with border #FFE082,
soft purple #E1BEE7 with border #CE93D8,
soft green #C8E6C9 with border #A5D6A7.
Keyword text accents:
coral #E05555,
teal #1A9988,
purple #6A5ACD,
green #3A8F3A.

CONTENT DENSITY: information-rich but uncluttered, packed with arrows, tensor shapes, formulas, tiny labels, and small pastel feature tokens. Micro-spacing 8-12px. Clean vector infographic style.

LAYOUT: asymmetric modern research layout.
A large wide panel on the left for the problem-dependent graph encoder.
A second wide panel in the center for context fusion and action-conditioned representation.
A tall narrower panel on the right for the bandit scoring, arm selection, linear-head update, and periodic hidden-layer training.
Thin dark grey arrows (#555555, 1.2px) connect the three panels from left to right.

=== PANEL 1: Problem-Dependent Hierarchical Graph Encoder ===

White panel, subtle shadow, rounded corners.
Title in bold purple text: "Hierarchical Graph Encoder".

Top half of the panel shows two parallel input branches:
1) TSP branch:
a soft blue rounded token group labeled "TSP nodes [B, N, 2]"
followed by a small floating label "2D coordinates"
then a pastel blue block "Input projection -> 128"

2) CVRP branch:
a soft green rounded token group labeled "CVRP nodes [B, N+1, 3]"
followed by a small floating label "coordinates + demand"
then a pastel green block "Input projection -> 128"

The two branches visually merge into a shared encoder pipeline:
- a pastel purple block "2 x MHA + FFN"
- a small floating pill "pooling ratio 0.8"
- tensor label "[B, 0.8N, 128]"
- another pastel purple block "2 x MHA + FFN"
- a small floating pill "pooling ratio 0.64"
- tensor label "[B, 0.64N, 128]"

Below the shared encoder flow, show the readout stage:
- two small floating feature groups: "Mean Pool" and "Max Pool"
- then a compact fusion node "Concat + GELU"
- output label "[B, 256]"
- then a tiny yellow token "scale"
- final graph embedding label "[B, 257]"

Add subtle tiny annotations:
"problem-specific first projection"
"shared attention + pooling backbone"
"hierarchical coarsened graph readout"

Use pastel token squares and small floating arrows to emphasize the dimensional changes.
Keep the panel visually similar to a modern ML backbone diagram, airy and elegant.

=== PANEL 2: Context Fusion and Action-Conditioned Representation ===

White panel, subtle shadow.
Title in bold coral text: "Context and Action Fusion".

Start from the left with the incoming graph embedding:
"[B, 257] graph embedding"

Then show feature fusion as a horizontal flow:
- a soft yellow token strip labeled "manual_raw (11)"
- a small pastel pill "problem one-hot (2)"
- fused context label in teal: "c(x) = [g(x), m(x), o(p)]"
- output tensor label "[B, 270]"

Next, show action identity injection:
- a soft pink pill label "arm embedding e(a) in R^16"
- a small note "shared arm id space, feasible mask by problem"
- concatenate context and arm embedding:

Then show the action-conditioned MLP:
- a pastel coral block "Linear"
- a floating tiny formula chip "286 -> hidden"
- a pastel yellow block "GELU"
- a pastel coral block "Linear"
- a floating tiny formula chip "hidden -> 64"

Final output:
a prominent soft blue rounded rectangle labeled
"phi(x, a) in R^64"
and a small subtitle
"action-conditioned representation"


Make the tensor progression very explicit:
[B,257] + [11] + [2] -> [B,270]
[B,270] + [M,16] -> [M,286]
[M,286] -> [M,64]

=== PANEL 3: UCB Scoring, Arm Selection, and Online Update ===

White tall vertical panel, subtle shadow.
Title in bold teal text: "Neural-LinUCB Decision and Update".

At the top, show a stacked candidate feature block:
"Feasible arms: phi(x,a1), phi(x,a2), ..., phi(x,aM)"
with tensor label "[M, 64]"
Use small pastel cards stacked vertically.

Under it, show three floating formula chips in one row:
1) "mean(a) = theta_p^T phi(x,a)"
2) "bonus(a) = sqrt(phi(x,a)^T A_p^{-1} phi(x,a))"
3) "score(a) = mean(a) + alpha * bonus(a)"

Then a bold rounded node:
"Select best feasible arm"

Under selection, show:
- a small pastel green chip "rank-based reward r(x,a)"
- a small note "higher reward = lower rank"

Then the linear head update block:
"Update problem-specific linear head"
with tiny sublabels:
"A_p^{-1}  |  b_p  |  theta_p"
and a small floating annotation:
"Sherman-Morrison update"

At the bottom, show a periodic update block:
"Every H steps: fit representation network"
with the loss written clearly in small serif math:
"L = (1/B) sum_i (theta_i^pull^T phi(x_i,a_i) - r_i)^2"

Add a tiny side note:
"mini-batch training"

Use a clean downward flow with slim arrows:
candidate phi -> compute mean/bonus/score -> select arm -> update linear head -> periodic hidden-layer training

=== STYLE SPECIFICATIONS ===

Pure white canvas only.
White panels only, separated by subtle shadow.
Rounded friendly font.
Flat vector illustration.
No 3D.
No gradients on canvas or panels.
No dark theme.
No heavy borders.
No nested framed boxes inside panels.
Use floating pills, feature tokens, formulas, and arrows.
All labels in English.
Publication-ready, elegant, modern, airy, information-dense.
Resolution at least 1800 x 1100.