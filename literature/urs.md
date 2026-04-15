# URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization
Anonymous Authors1 
# Abstract
Multi-task neural routing solvers have emerged as a promising paradigm for their ability to solve multiple vehicle routing problems (VRPs) using a single model. However, existing neural solvers typically rely on predefined problem constraints or require per-problem fine-tuning, which substantially limits their zero-shot generalization ability to unseen VRP variants. To address this critical bottleneck, we propose URS, a unified neural routing solver that achieves zero-shot generalization across a wide range of unseen VRPs with a single model. We propose a unified data representation (UDR) that replaces problem enumeration with data unification, thereby broadening the problem coverage and reducing reliance on domain expertise. In addition, we introduce a Mixed Bias Module (MBM) during encoding to improve node embeddings, which efficiently captures multiple priors inherent to various problems. On top of the UDR, we develop a problem-conditioned parameter generator to further improve zero-shot generalization. Extensive experiments show that URS consistently produces high-quality solutions for 110 VRP variants (including 99 unseen variants) while demonstrating impressive scalability to large-scale instances with up to 7000 nodes. To the best of our knowledge, URS is the first neural solver to handle over 100 VRP variants with a single model. 
# 1. Introduction
The Vehicle Routing Problem (VRP) is an essential class of combinatorial optimization problems (COPs) with extensive applications in logistics and supply chain management (Tiwari & Sharma, 2023; Sar & Ghadimi, 2023). Solving VRPs 
1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>. 
Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute. 
efficiently is challenging due to their NP-hard nature. While exact solvers can produce optimal solutions, they often become computationally prohibitive for real-world scenarios. In recent decades, classical heuristic solvers (Helsgaun, 2017; Vidal, 2022) have achieved impressive performance within acceptable timeframes. Nevertheless, these methods require considerable domain expertise to design specialist rules for each routing problem. Given the growing diversity of VRP variants in real-world applications, manually crafting tailored rules for every case has become impractical. 
In recent years, neural combinatorial optimization (NCO) methods have attracted substantial attention for their potential to reduce reliance on handcrafted rules while maintaining competitive solution quality (Bengio et al., 2021). This has led to the development of many high-performing neural routing solvers that automatically learn implicit problemspecific rules from data under different training paradigms such as supervised learning (SL) (Drakulic et al., 2023; Luo et al., 2023; 2025b), reinforcement learning (RL) (Bello et al., 2016; Kool et al., 2019; Zhou et al., 2024a), and self-improved learning (SIL) (Luo et al., 2025a; Pirnay & Grimm, 2024). Although these solvers have demonstrated impressive performance on specific problems (Huang et al., 2025; Zhou et al., 2025), they typically require architectural customization and per-problem retraining to accommodate the distinct constraints and features of different VRP variants. These limitations increase overall training costs and hinder practical deployment to new problems. 
To address the challenge of cross-problem generalization, growing attention has been directed toward multi-task learning capable of handling diverse routing problems. As summarized in Table 1, existing efforts largely fall into two categories: 1) constraint combination-based multi-task learning and 2) adapter-based fine-tuning. In the first approach, VRP variants are treated as different combinations of constraint attributes, and a unified model is trained across these combinations to enable knowledge sharing for problems with seen constraints (Liu et al., 2024; Zhou et al., 2024b; Li et al., 2025a; Berto et al., 2025; Zheng et al., 2025). The second approach builds a shared model backbone for all problems and incorporates problem-specific input/output adapters, thereby reducing retraining costs (Lin et al., 2024; Drakulic et al., 2025; Wang et al., 2025a). 

URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 

Table 1. Comparison between our URS and existing neural solvers with multi-task learning. Note that ”#VRP Variants” and ”Generalizable Scale” refer to the total number of tested problems and the maximum scale reported in the original papers, respectively.

<table><tr><td>Multi-task Neural Routing Solver</td><td>Training Paradigm</td><td>#VRP Variants</td><td>Symmetric VRPs</td><td>Asymmetric VRPs</td><td>Pickup and Delivery Problems</td><td>Zero-shot Generalization</td><td>Training Scale</td><td>Generalizable Scale</td><td>Remarks</td></tr><tr><td>MTPOMO (Liu et al., 2024)</td><td>RL</td><td>16</td><td>?</td><td>×</td><td>×</td><td>?</td><td>100</td><td>1000</td><td>Constraint Combination</td></tr><tr><td>MVMoE (Zhou et al., 2024b)</td><td>RL</td><td>16</td><td>?</td><td>×</td><td>×</td><td>?</td><td>100</td><td>1000</td><td>Constraint Combination</td></tr><tr><td>RouteFinder (Berto et al., 2025)</td><td>RL</td><td>48</td><td>?</td><td>×</td><td>×</td><td>?</td><td>100</td><td>1000</td><td>Constraint Combination</td></tr><tr><td>CaDA (Li et al., 2025a)</td><td>RL</td><td>16</td><td>?</td><td>×</td><td>×</td><td>?</td><td>100</td><td>200</td><td>Constraint Combination</td></tr><tr><td>MTL-KD (Zheng et al., 2025)</td><td>RL+KD</td><td>16</td><td>?</td><td>×</td><td>×</td><td>?</td><td>100</td><td>1000</td><td>Constraint Combination</td></tr><tr><td>TSP-FT (Lin et al., 2024)</td><td>RL</td><td>4</td><td>?</td><td>×</td><td>×</td><td>×</td><td>100</td><td>1000</td><td>Adapter-based Fine-tuning</td></tr><tr><td>MTL-MAB (Wang et al., 2025a)</td><td>RL</td><td>3</td><td>?</td><td>×</td><td>×</td><td>×</td><td>100</td><td>1000</td><td>Adapter-based Fine-tuning</td></tr><tr><td>GOAL (Drakulic et al., 2025)</td><td>SL</td><td>10</td><td>?</td><td>?</td><td>×</td><td>×</td><td>100</td><td>1000</td><td>Adapter-based Fine-tuning</td></tr><tr><td>URS (Ours)</td><td>RL</td><td>110</td><td>?</td><td>?</td><td>?</td><td>?</td><td>100</td><td>7000</td><td>Unified Data Representation</td></tr></table>
Despite the above advancements, existing methods still fall short in zero-shot generalization. The problem coverage of constraint combination-based approaches is inherently bounded by their manually specified constraint sets, while adapter-based approaches require fine-tuning and thus cannot perform zero-shot generalization. Moreover, both strategies fundamentally rely on the explicit enumeration of problems through a predefined set of problem tags. This practice is problematic because the constraint space of routing problems is open-ended and compositional, making any fixed tagging scheme fundamentally limited. More critically, creating and maintaining such a problem taxonomy requires considerable domain expertise, which NCO always aims to avoid. A detailed review of related work on single-task and multi-task learning is provided in Appendix A. 
In this paper, we propose a powerful Unified Neural Routing Solver (URS) to improve the cross-problem zero-shot generalization ability for NCO methods. Our contributions can be summarized as follows: (1) We propose a unified data representation (UDR) that replaces existing problem enumeration with data unification, significantly broadening problem coverage while mitigating reliance on domain-specific expertise; (2) We introduce a Mixed Bias Module (MBM) during encoding to improve node embeddings, which efficiently captures multiple priors inherent to various problems while reducing architectural redundancy; (3) On top of the UDR, we develop a problem-conditioned parameter generator to further improve zero-shot generalization; (4) Extensive experiments on 110 VRP variants show that URS not only achieves competitive performance against specialist neural solvers on seen variants but also exhibits strong zero-shot generalization across 99 unseen variants. Notably, URS demonstrates impressive scalability to large-scale instances with up to 7000 nodes. To the best of our knowledge, URS is the first neural solver to efficiently solve over 100 VRP variants with a single model, without retraining or fine-tuning. 
# 2. Preliminaries
In this section, we first introduce the definition of VRP, then provide an overview of recent constructive neural solvers for solution generation (Kool et al., 2019; Kwon et al., 2020). 
Vehicle Routing Problems Consider a VRP instance with an optional depot indexed by 0 and $n$ customers indexed by $\{ 1 , 2 , \ldots , n \}$ , such as in the Capacitated VRP (CVRP). The instance can be represented as a graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where the node set $\mathcal { V } = \{ v _ { i } \} _ { i = 0 } ^ { n }$ has a total size of $| \nu | = 1 + n$ unless the variant has no depot (e.g., the Traveling Salesman Problem), and the edge set is $\mathcal { E } = \{ e ( v _ { i } , v _ { j } ) \mid v _ { i } , v _ { j } \in$ $\nu , v _ { i } \neq v _ { j } \}$ . The travel cost between nodes is given by a distance matrix $D = \{ d _ { i j } | \forall i , j \in 0 , \ldots , n \}$ . In VRP, each node $v _ { i } \in \mathcal V$ includes node coordinates $\{ x _ { i } , y _ { i } \}$ when available and problem-specific attributes (e.g., demands in CVRP). A feasible solution calls for the determination of a set of routes, each performed by a single vehicle that starts and ends at its own depot or starting node, all constraints are satisfied, denoted as $\pi = ( \pi _ { 1 } , \pi _ { 2 } , . . . , \pi _ { m } )$ (Toth & Vigo, 2002) (i.e., a finite sequence), which is a permutation of $m$ nodes. Let $\Omega$ denote the feasible sequence set, given an objective function $f ( \pi | \mathcal { G } )$ , we aim to search for a sequence $\pi ^ { * }$ with the maximum objective, i.e., $\pi ^ { * } =$ arg $\operatorname* { m a x } _ { \pi \in \Omega } f ( \pi | { \mathcal { G } } )$ $f ( \pi | \mathcal { G } )$ . In most VRPs, $f ( \pi | \mathcal { G } )$ can be defined as the negative value of the total distance of $\pi$ . 
In this paper, we consider 110 VRP variants which may simultaneously have one or more constraints from the following categories: (1) Capacity (C); (2) Open Route (O); (3) Backhaul (B); (4) Backhaul and Priority (BP); (5) Duration Limit (L); (6) Time Windows (TW); (7) Multi-Depot (MD); (8) Prize Collecting (PC); (9) Asymmetric (A); and (10) Pickup and Delivery (PD). Detailed definitions of these constraints are provided in Appendix B. 
Constructive Neural Routing Solver Most constructive NCO methods employ an encoder-decoder architecture for solution construction (Luo et al., 2023; Kwon et al., 2020). Let learnable model parameters be denoted as $\pmb { \theta } = \{ \pmb { \theta } _ { e n c } , \pmb { \theta } _ { d e c } \}$ . Without loss of generality, we present the prevailing autoregressive construction pipeline using AM (Kool et al., 2019). Given an instance $\mathcal { G }$ , raw node features are first mapped by a linear projection to initial embeddings $H ^ { ( 0 ) } = \{ \bar { \mathbf { h } } _ { i } ^ { ( 0 ) } \} _ { i = 0 } ^ { n }$ } i=0. They are then processed by an encoder $\pmb { \theta } _ { e n c }$ with $L$ attention layers to produce refined node embeddings $H ^ { ( L ) } = \{ { \bf h } _ { i } ^ { ( L ) } \} _ { i = 0 } ^ { n } \dot { \bf \Phi }$ . At each decoding step $t$ 
2 
110   
111   
112   
113   
114   
115   
116   
117   
118   
119   
120   
121   
122   
123   
124   
125   
126   
127   
128   
129   
130   
131   
132   
133   
134   
135   
136   
137   
138   
139   
140   
141   
142   
143   
144   
145   
146   
147   
148   
149   
150   
151   
152   
153   
154   
155   
156   
157   
158   
159   
160   
161   
162   
163   
164 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/90b59445-9787-40c5-9062-a92383c20222/92767a300c4991aff9404de3fd4d4b0b7c42b7d2165f4e39140f7f56a484e35b.jpg)


Figure 1. The pipeline of our URS for solving 110 VRP variants using a single model without any fine-tuning, illustrated on APDTSP.

the decoder $\theta _ { d e c }$ sequentially selects a node to append to the current partial solution $\pi _ { 1 : t - 1 } = ( \pi _ { 1 } , \pi _ { 2 } , . . . , \pi _ { t - 1 } )$ , where $\pi _ { 1 } , \pi _ { t - 1 } \in \mathcal { V }$ are the first and last visited node, respectively. This sequential process continues until a complete solution is formed. To guarantee the feasibility of generated solutions, a masking function $\mathcal { M } _ { t }$ sets the selection probabilities of the following nodes to $- \infty$ during the construction process: nodes that (1) are visited or (2) violate problem-specific constraints (e.g., capacity). 
# 3. Methodology
In this section, as illustrated in Figure 1, we propose a powerful Unified Neural Routing Solver (URS), which significantly improves cross-problem zero-shot generalization for NCO methods. Its key components are elaborated below. 
# 3.1. Unified Data Representation
Existing multi-task neural solvers (Zhou et al., 2024b; Drakulic et al., 2025) fundamentally rely on explicit problem enumeration via a predefined set of problem tags, which is challenging because the constraint space of VRPs is open-ended and compositional, rendering any fixed tagging scheme inherently limited. More critically, creating and maintaining such a problem taxonomy requires considerable domain expertise, which NCO always aims to avoid. 
From a data perspective, despite the substantial diversity in constraints, their instances share a common instance representation, e.g., all instances under the current constraint combinations (Liu et al., 2024; Zhou et al., 2024b) can be formulated as instances of either CVRP or CVRPTW. Leveraging this commonality, we propose a Unified Data 
Representation (UDR) ${ \cal U } = \{ { \bf u } _ { i } \} _ { i = 0 } ^ { n }$ , which replaces problem enumeration with data unification, broadening problem coverage and reducing reliance on domain expertise. 
Instead of relying on a predefined problem taxonomy with a discrete set of problem tags, UDR decouples instance representation from constraint definitions and operates in a unified space characterized by both continuous and discrete features. It allows a single neural solver to address a much larger, open-ended space of VRP variants, as new problems can be seamlessly incorporated into this unified representation. Problem-specific constraints are delegated to the model-agnostic masking function $\mathcal { M } _ { t }$ . For example, for all CVRP variants, only the remaining load is explicitly provided as input, while diverse additional constraints (e.g., time windows) are enforced implicitly by masking infeasible candidate nodes during solution construction, rather than being fully enumerated in the decoder’s input. Thus, UDR enables a single model to handle various VRP variants while significantly reducing reliance on domain expertise. Specifically, for an instance $\mathcal { G }$ , each $\mathbf { u } _ { i } = \{ \rho _ { i } , \ : \omega _ { i } , \ : \pmb { \xi } _ { i } \}$ , where $\rho _ { i }$ , $\omega _ { i }$ , $\pmb { \xi } _ { i }$ are a positional identifier, unified attribute set, and node-type indicator, respectively. 
Positional Identifier $( \rho _ { i } )$ We define a unified positional identifier for each node as $\pmb { \rho } _ { i } = \{ \eta _ { i } , x _ { i } , y _ { i } \} \in [ 0 , 1 ] ^ { 3 }$ . The $\eta _ { i } \sim \mathrm { U n i f o r m } ( 0 , 1 ) \in \mathbb { R } ^ { 1 }$ is a sampled scalar, which is used to address asymmetric problems, following Drakulic et al. (2023). For symmetric cases, we simply set $\eta _ { i } = 0$ . 
Unified Attribute Set $( \omega _ { i } )$ We define the unified attribute set as $\omega _ { i } = \{ \delta _ { i } , \epsilon _ { i } , \mu _ { i } , e _ { i } , l _ { i } , s _ { i } \}$ , where they correspond to demand, prize, penalty, earliest arrival time, latest arrival time, and service time, respectively. The unification 
3 
165   
166   
167   
168   
169   
170   
171   
172   
173   
174   
175   
176   
177   
178   
179   
180   
181   
182   
183   
184   
185   
186   
187   
188   
189   
190   
191   
192   
193   
194   
195   
196   
197   
198   
199   
200   
201   
202   
203   
204   
205   
206   
207   
208   
209   
210   
211   
212   
213   
214   
215   
216   
217   
218   
219 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
empowers the model to learn the relative importance of individual attributes in a shared embedding space. It also supports effortless attribute extension or ablation via simple zero-filling, eliminating the need to change the architecture. 
Node-type Indicator $( \xi _ { i } )$ We provide an additional 5- dimensional binary vector $\pmb { \xi } _ { i } \in \{ 0 , 1 \} ^ { 5 }$ for each node, encoding: depot, pickup node, delivery node, node in subroutes, node in open route. The nodes in sub-routes and open routes mean the solution $\pi$ can have sub-routes or be an open route. We introduce explicit structural roles via $\pmb { \xi } _ { i }$ that help the model generalize over diverse variants. Further ablation analysis concerning $\pmb { \xi } _ { i }$ is provided in Appendix J.1. 
Multi-hot Problem Representation $( \lambda _ { i } )$ Through the UDR, each problem instance activates only a specific subset of the unified feature space. For any instance $\mathcal { G } _ { i }$ drawn from different problems, a corresponding multi-hot problem representation $\lambda _ { i }$ is obtained by indicating active (nonzero) feature slots with a value of 1. This representation captures only the presence of features, in contrast to their raw values, and it subsequently provides conditional guidance to the position bias weight (see Equation (6)) and the adaptive decoder parameters $\pmb { \theta } _ { d e c } ( \lambda )$ (see Equation (9) and Equation (10)), thereby helping the model in distinguishing problem variants and refining its node selection process. 
A more detailed description of $U$ and $\boldsymbol { \lambda }$ is provided in Appendix C and Appendix D, respectively. 
# 3.2. Model Architecture
As shown in Figure 1, we adopt AM (Kool et al., 2019) as our basic model. While UDR provides cross-problem consistency, different problems have diverse geometric priors. To efficiently learn the multiple priors inherent in various problems and obtain better cross-problem zero-shot generalization, we propose two key enhancements: (1) a Mixed Bias Module (MBM) that captures multiple priors inherent to various problems, and (2) problem-conditioned parameter generators conditioned on the UDR. Their implementations are detailed below. 
Embedding Layer Given an instance $\mathcal { G }$ , for each $\mathbf { u } _ { i } =$ $\{ \rho _ { i } , \omega _ { i } , \xi _ { i } \}$ , we project it into $d$ -dimensional embeddings through three separate linear transformations: 
$$
\mathbf {h} _ {i} ^ {(0)} = \rho_ {i} W _ {\boldsymbol {\rho}} + \omega_ {i} W _ {\boldsymbol {\omega}} + \xi_ {i} W _ {\boldsymbol {\xi}}, \tag {1}
$$
where $W _ { \pmb { \rho } } \in \mathbb { R } ^ { 3 \times d }$ , $W _ { \omega } \in \mathbb { R } ^ { 6 \times d }$ , $W _ { \xi } \in \mathbb { R } ^ { 5 \times d }$ are learnable matrices. Then we obtain a set of initial embeddings $H ^ { ( 0 ) } = \{ { \bf h } _ { i } ^ { ( 0 ) } \} _ { i = 0 } ^ { n }$ } ni =0 for all nodes in instance $\mathcal { G }$ . This embedding is the initial input of encoder $\pmb { \theta } _ { e n c }$ . After passing through into adv $L$ stacked attention layced node embeddings $H ^ { ( 0 ) }$ rmed. The $H ^ { ( L ) } = \{ { \bf h } _ { i } ^ { ( L ) } \} _ { i = 0 } ^ { n }$ detailed encoding process is provided in Appendix E.1. 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/90b59445-9787-40c5-9062-a92383c20222/01f410b89f02aa227d27772c37732d516d8c7d26e3f22a0100c6b7725803f6f4.jpg)


Figure 2. The structure of the proposed Mixed Bias Module, explained on a 4-node APDTSP instance. We replace plain attention with MBM during encoding to improve node embeddings, which effectively captures multiple priors inherent to various problems while reducing architectural redundancy.

Mixed Bias Module Given the distinct priors across VRP variants, efficiently capturing the intrinsic geometric bias is pivotal for enhancing the cross-problem generalization of RL-based multi-task neural solvers. Current leading RL-based models generally adopt a heavy encoder and light decoder architecture, where the quality of the encodergenerated node embeddings plays a key role in overall performance. While prior studies like MatNet (Kwon et al., 2021) have attempted to handle asymmetry by integrating directional distance biases via parallel layers, such designs incur high model complexity and lack universality. Consequently, effectively encoding multiple inherent biases, such as symmetric and asymmetric distances, and relational dependencies, remains challenging, hindering the acquisition of high-quality solutions across a wide range of problems. 
To address the limitation, we propose the mixed bias module (MBM) to replace the vanilla attention mechanism (Vaswani et al., 2017), which enhances cross-problem generalization while reducing architectural redundancy. As shown in Figure 2, MBM can efficiently capture three problemspecific prior matrices via three separate attention calculations: (1) an outgoing distance matrix $_ { D }$ ; (2) an incoming distance matrix $D ^ { \mathrm { T } }$ ; and (3) an optional relation matrix $\pmb { R } = \{ r _ { i j } | \forall i , j \in 0 , \dots , n \}$ , where $r _ { i j } = 0$ if a predefined relation exists (e.g., pickup–delivery pairing) and 1 otherwise. This ensures that smaller values are preferred, aligning with the distance metric, where smaller values correspond to a larger selection bias. For the $\ell$ -th layer, the MBM output $\hat { \mathbf { h } } _ { i } ^ { ( \bar { \ell } ) }$ can be expressed as: 
$$
\bar {\mathbf {h}} _ {i} ^ {(0)} = \operatorname {A t t e n t i o n} \left(\mathbf {h} _ {i} ^ {(\ell - 1)}, H ^ {(\ell - 1)}, f (\alpha , N, \boldsymbol {D} _ {i})\right), \tag {2}
$$
4 
220   
221   
222   
223   
224   
225   
226   
227   
228   
229   
230   
231   
232   
233   
234   
235   
236   
237   
238   
239   
240   
241   
242   
243   
244   
245   
246   
247   
248   
249   
250   
251   
252   
253   
254   
255   
256   
257   
258   
259   
260   
261   
262   
263   
264   
265   
266   
267   
268   
269   
270   
271   
272   
273 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
$$
\bar {\mathbf {h}} _ {i} ^ {(1)} = \operatorname {A t t e n t i o n} \left(\mathbf {h} _ {i} ^ {(\ell - 1)}, H ^ {(\ell - 1)}, f (\alpha , N, \boldsymbol {D} _ {i} ^ {\mathrm {T}})\right), \tag {3}
$$
$$
\begin{array}{l} \bar {\mathbf {h}} _ {i} ^ {(2)} = \left\{ \begin{array}{l l} \operatorname {A t t e n t i o n} \left(\mathbf {h} _ {i} ^ {(\ell - 1)}, H ^ {(\ell - 1)}, f (\alpha , \boldsymbol {R} _ {i})\right) & \text {i f} \boldsymbol {R} \neq \emptyset \\ \mathbf {0} & \text {o t h e r w i s e}, \end{array} \right. (4) \\ \hat {\mathbf {h}} _ {i} ^ {(\ell)} = \left[ \bar {\mathbf {h}} _ {i} ^ {(0)}, \bar {\mathbf {h}} _ {i} ^ {(1)}, \bar {\mathbf {h}} _ {i} ^ {(2)} \right] W ^ {O}, (5) \\ \end{array}
$$
where $[ \cdot , \cdot ]$ denotes the horizontal concatenation operator, $W ^ { O } \in \bar { \mathbb { R } } ^ { ( 3 \times d ) \times d }$ is a learnable matrix. For Attention(·), we adopt the attention mechanism introduced by Zhou et al. (2024a), which boosts the perception of diverse geometric patterns through an adaptation function $f ( \alpha , N , d _ { i j } )$ with bias weight $\alpha$ (see Appendix F for implementation details). Note that due to the relation $r _ { i j }$ is scale-independent, we remove scale $N$ in adaptation function for $\pmb { R }$ , and if $\pmb { R }$ is absent, we substitute a zero vector for hˉ(2)i . $\bar { \mathbf { h } } _ { i } ^ { ( 2 ) }$ 
Problem-Conditioned Parameter Generation Using a single static set of parameters to solve 110 VRP variants is a significant challenge. To mitigate this, we introduce a problem-conditioned mechanism based on problem representation $\lambda$ , which consists of two components: (1) BIAS(λ) adjusts the degree to which different priors are integrated into the attention; and (2) WEIGHT(λ) generates decoder parameters for each problem. 
As shown in Figure 3, unlike existing adapter-based finetuning methods (Lin et al., 2024; Drakulic et al., 2025), we provide a unified decoding entrance and enable URS to adaptively generate decoder parameters for each problem. This conditional weight generation removes problem-specific adapters, enabling zero-shot generalization to unseen problems while maintaining solution quality. 
For MBM in Equation (5), we generate $\alpha$ via a lightweight bias network BIAS(λ): 
$$
\operatorname {B I A S} (\boldsymbol {\lambda}) = \max  \left(1, \left(\boldsymbol {\lambda} W _ {1} + \mathbf {b} _ {1}\right) W _ {2} + \mathbf {b} _ {2}\right), \tag {6}
$$
where $W _ { 1 } \in \mathbb { R } ^ { | \lambda | \times d }$ , $W _ { 2 } \in \mathbb { R } ^ { d \times 1 }$ , $\mathbf { b } _ { 1 } \in \mathbb { R } ^ { d }$ , $\mathbf { b } _ { 2 } \in \mathbb { R } ^ { 1 }$ are learnable parameters. We set the minimum bias to 1, resulting in faster model convergence. 
For decoder $\pmb { \theta } _ { d e c } ( \lambda )$ , all parameters are generated based on $\boldsymbol { \lambda }$ . Following Kwon et al. (2020), given the first and last visited node embeddings ${ \bf h } _ { \pi _ { 1 } } ^ { ( L ) }$ and $\mathbf { h } _ { \pi _ { t - 1 } } ^ { ( L ) }$ , and an optional constraint state $C _ { t } \in \mathbb { R } ^ { 1 }$ (e.g., remaining load), the context embedding $\mathbf { h } _ { ( C ) } ^ { t }$ can be expressed as: 
$$
\mathbf {h} _ {(C)} ^ {t} = \left[ \mathbf {h} _ {\pi_ {1}} ^ {(L)}, \mathbf {h} _ {\pi_ {t - 1}} ^ {(L)}, C _ {t} \right] W _ {Q} (\boldsymbol {\lambda}), \tag {7}
$$
where $[ \cdot , \cdot ]$ is the horizontal concatenation operator, $W _ { Q } ( \lambda ) \ { \bf \bar { \Big { \in } } } \ { \bf \bar { \mathbb { R } } } ^ { ( 2 d + 1 ) \times d }$ is a linear projection matrix conditioned on $\boldsymbol { \lambda }$ . Missing $C _ { t }$ is zero-padded (For implementation details of each problem, please refer to Appendix 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/90b59445-9787-40c5-9062-a92383c20222/2602c57be286b63e8dd86327434fe96f737a2692808154f8938985ce217af762.jpg)


Figure 3. Comparison between our URS and existing adapterbased fine-tuning frameworks.

E.3.). The new context embedding $\hat { \mathbf { h } } _ { ( C ) } ^ { t }$ is obtained via Attention(·) (Zhou et al., 2024a) on $\mathbf { h } _ { ( C ) } ^ { t }$ and $H ^ { ( L ) }$ : 
$$
K = H ^ {(L)} W _ {K} (\boldsymbol {\lambda}), \quad V = H ^ {(L)} W _ {V} (\boldsymbol {\lambda}), \tag {8}
$$
$$
\hat {\mathbf {h}} _ {(C)} ^ {t} = \operatorname {A t t e n t i o n} \left(\mathbf {h} _ {(C)} ^ {t}, K, V, \mathcal {M} _ {t}, f (\alpha , N, d _ {t, i})\right), \tag {9}
$$
where $W _ { K } ( \pmb { \lambda } ) , W _ { V } ( \pmb { \lambda } ) \in \mathbb { R } ^ { d \times d }$ are linear projection matrices conditioned on $\lambda$ . Finally, we compute probabilities for feasible nodes $p _ { \pmb { \theta } } ( \pi _ { t } = i \mid \mathcal { G } , \pi _ { 1 : t - 1 } ) = \mathrm { s o f t m a x } ( \mathbf { u } )$ based on the improved compatibility (Zhou et al., 2024a): 
$$
u _ {i} ^ {t} = \left\{ \begin{array}{l l} \zeta \cdot \tanh  \left(\frac {\hat {\mathbf {h}} _ {(C)} ^ {t} \left(\mathbf {h} _ {i} ^ {(L)}\right) ^ {\mathrm {T}}}{\sqrt {d}} + f (\alpha , N, d _ {t, i})\right) & \text {i f} i \notin \left\{\pi_ {1: t - 1} \right\} \\ - \infty & \text {o t h e r w i s e}, \end{array} \right. \tag {10}
$$
where $\zeta$ is the clipping parameter. Inspired by Lin et al. (2022), we use a simple MLP hypernetwork WEIGHT $( \lambda )$ to generate $\{ W _ { Q } ( \lambda )$ , $W _ { K } ( \lambda )$ , $W _ { V } ( \lambda ) \}$ (see Appendix E.2 for implementation details). For $\alpha$ in Equation (9) and Equation (10), we generate them via BIAS(λ) in Equation (6). 
# 4. Experiments
In this section, we present a comprehensive evaluation of URS against both classical and neural solvers across 110 variants. For URS, we focus on four key aspects: (1) performance on seen problems; (2) zero-shot generalization on 99 unseen problems; (3) scalability to large-scale VRP variants; and (4) results on benchmark datasets. All experiments are conducted on a single NVIDIA GeForce RTX 4090 GPU. 
# 4.1. Experimental Setup
Problem Setting We train URS on 11 mixed variants to ensure that most features in UDR have been seen at least twice, which prevents URS from overfitting a specific feature to a single problem. Training problems and related data generations are as follows: (1) ATSP (Kwon et al., 2021);(2) TSP (Kool et al., 2019); (3) CVRP (Kool et al., 2019); (4) ACVRP (Kwon et al., 2021; Kool et al., 
5 
275   
276   
277   
278   
279   
280   
281   
282   
283   
284   
285   
286   
287   
288   
289   
290   
291   
292   
293   
294   
295   
296   
297   
298   
299   
300   
301   
302   
303   
304   
305   
306   
307   
308   
309   
310   
311   
312   
313   
314   
315   
316   
317   
318   
319   
320   
321   
322   
323   
324   
325   
326   
327   
328   
329 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 

Table 2. Performance comparison on 8 seen less-explored variants.

<table><tr><td rowspan="2">Method</td><td colspan="2">OP100</td><td colspan="2">PCTSP100</td><td colspan="2">PDTSP100</td><td colspan="2">ACVRP100</td><td colspan="2">CVRPTW100</td><td colspan="2">CVRPB100</td><td colspan="2">OCVRP100</td><td colspan="2">OCVRPTW100</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>1.5m</td><td>0.00%</td><td>1.2h</td><td>0.00%</td><td>9.8m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>19.6m</td><td>0.00%</td><td>20.8m</td><td>0.00%</td><td>5.3m</td><td>0.00%</td><td>20.8m</td></tr><tr><td>Sym-NCO</td><td>0.68%</td><td>15s</td><td>0.14%</td><td>16s</td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td></tr><tr><td>BQ</td><td>0.63%</td><td>7s</td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td></tr><tr><td>Heter-AM</td><td>-</td><td></td><td>-</td><td></td><td>6.61%</td><td>5m</td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td></tr><tr><td>MVMoE</td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>4.90%</td><td>15s</td><td>1.28%</td><td>11s</td><td>3.14%</td><td>13s</td><td>3.85%</td><td>15s</td></tr><tr><td>MTPOMO</td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>5.31%</td><td>8s</td><td>1.67%</td><td>5s</td><td>3.46%</td><td>6s</td><td>4.41%</td><td>7s</td></tr><tr><td>ReLD-MTL</td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>4.56%</td><td>10s</td><td>0.90%</td><td>6s</td><td>2.32%</td><td>7s</td><td>3.10%</td><td>9s</td></tr><tr><td>GOAL-MTL</td><td>1.20%</td><td>38s</td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td><td>4.66%</td><td>42s</td><td>-</td><td></td><td>-</td><td></td><td>-</td><td></td></tr><tr><td>URS-STL</td><td>0.11%</td><td>4s</td><td>-0.09%</td><td>5s</td><td>3.22%</td><td>4s</td><td>2.01%</td><td>1.4m</td><td>4.50%</td><td>8s</td><td>1.59%</td><td>6s</td><td>2.25%</td><td>6s</td><td>3.22%</td><td>8s</td></tr><tr><td>URS</td><td>0.45%</td><td>4s</td><td>1.06%</td><td>5s</td><td>4.98%</td><td>4s</td><td>3.06%</td><td>1.4m</td><td>6.13%</td><td>8s</td><td>1.46%</td><td>6s</td><td>3.24%</td><td>6s</td><td>5.07%</td><td>8s</td></tr></table>

Table 3. Experimental results on seen ATSP, TSP, and CVRP.

<table><tr><td rowspan="2">Method</td><td colspan="2">ATSP100</td><td colspan="2">TSP100</td><td colspan="2">CVRP100</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>2m</td><td>0.00%</td><td>6m</td><td>0.00%</td><td>9.1m</td></tr><tr><td>Sym-NCO</td><td>-</td><td></td><td>0.14%</td><td>5s</td><td>1.46%</td><td>6s</td></tr><tr><td>BQ</td><td>1.27%</td><td>1m</td><td>0.35%</td><td>7s</td><td>3.24%</td><td>8s</td></tr><tr><td>LEHD</td><td>-</td><td></td><td>0.59%</td><td>2s</td><td>3.68%</td><td>2s</td></tr><tr><td>POMO</td><td>-</td><td></td><td>0.13%</td><td>5s</td><td>1.25%</td><td>6s</td></tr><tr><td>ICAM</td><td>3.05%</td><td>1m</td><td>0.15%</td><td>4s</td><td>2.04%</td><td>4s</td></tr><tr><td>MatNet</td><td>0.95%</td><td>6.5m</td><td>-</td><td></td><td>-</td><td></td></tr><tr><td>MVMoE</td><td>-</td><td></td><td>-</td><td></td><td>1.65%</td><td>12s</td></tr><tr><td>MTPOMO</td><td>-</td><td></td><td>-</td><td></td><td>1.85%</td><td>6s</td></tr><tr><td>ReLD-MTL</td><td>-</td><td></td><td>-</td><td></td><td>1.42%</td><td>8s</td></tr><tr><td>GOAL-MTL</td><td>1.77%</td><td>1m</td><td>-</td><td></td><td>4.22%</td><td>48s</td></tr><tr><td>URS-STL</td><td>0.62%</td><td>1.1m</td><td>0.08%</td><td>6s</td><td>1.63%</td><td>6s</td></tr><tr><td>URS</td><td>2.26%</td><td>1.1m</td><td>0.57%</td><td>6s</td><td>1.81%</td><td>6s</td></tr></table>
2019); (5) Orienteering Problem (OP) (Kool et al., 2019); (6) PCTSP (Kool et al., 2019); (7) PDTSP (Li et al., 2021); (8) CVRPTW (Zhou et al., 2024b); (9) OCVRP (Zhou et al., 2024b); (10) CVRPB (Zhou et al., 2024b), and (11) OCVRPTW (Zhou et al., 2024b). We provide further discussion of problem selection in Appendix I.1. To further highlight its broad applicability, we construct 99 unseen variants, each containing 1,000 instances. All oracle solvers and problem representations are available in Appendix D. 
Model & Training Setting We use an embedding dimension $d$ of 128 and a feed-forward layer dimension of 512. We set the number of attention layers in the encoder to 12. The clipping parameter $\zeta = 5 0$ in Equation (10). URS is trained by the REINFORCE (Williams, 1992) algorithm, and the data is generated on the fly (see Appendix E.4 for more details). We train URS for 500 epochs with 2,000 batches per epoch, and each batch has a size of 128. At each training step, we randomly select one problem from 11 problems. We employ the AdamW optimizer (Loshchilov & Hutter, 2017) with an initial learning rate of $1 0 ^ { - 4 }$ , which is decayed by a factor of 0.1 at the 451st epoch. Weight decay is set to $1 0 ^ { - 6 }$ . More details are provided in Appendix G. 
Baseline We compare URS with the following methods: (1)Classical Solvers: PyVRP (Wouda et al., 2024), LKH3 (Helsgaun, 2017), and OR-Tools (Perron & Furnon, 2023). We run LKH3 (Helsgaun, 2017) with 10000 trials and 1 run (Kool et al., 2019). For PyVRP and OR-Tools, we run them on a single CPU core with time limits of 20s (Li et al., 2025a); (2)Neural Solvers: (i) specialist solvers: Sym-NCO (Kim et al., 2022), BQ (Drakulic et al., 2023), LEHD (Luo et al., 2023), ICAM (Zhou et al., 2024a), POMO (Kwon et al., 2020), MatNet (Kwon et al., 2021), and Heter-AM (Li et al., 2021). ; (ii) generalist neural solvers: MTPOMO (Liu et al., 2024), MVMoE (Zhou et al., 2024b), ReLD-MTL (Huang et al., 2025), and GOAL-MTL (Drakulic et al., 2025). RouteFinder (Berto et al., 2025) and CaDA (Li et al., 2025a) are excluded to keep consistent problem settings in training for fair comparison. 
Metrics and Inference We report the optimality gap (Gap) and total inference time (Time) for each method. The optimality gap quantifies the discrepancy between the solutions generated by the corresponding methods and the near-optimal solutions obtained with classical solvers. For most NCO baseline methods, we execute the source code provided by the authors using default settings. We compare the performance of URS with relevant baselines for each problem, as well as its single-task version (i.e., URS-STL). The URS-STL variant uses exactly the same architecture and hyperparameters (see Appendix G.2 for more details). For symmetric instances, we report the best result with $\times 8$ instance augmentation, following Kwon et al. (2020). For asymmetric instances, we report the best result with $\times 1 2 8$ instance augmentation, following Kwon et al. (2021). 
# 4.2. Performance Evaluation
Performance on Seen VRPs The experimental results on trained VRPs with scale 100 are reported in Table 2 and Table 3. We observe that among comparable neural solvers, URS-STL achieves the best performance on 8 of the 11 routing problems, while also offering remarkably fast inference, highlighting the strength of the URS architecture 
6 
330   
331   
332   
333   
334   
335   
336   
337   
338   
339   
340   
341   
342   
343   
344   
345   
346   
347   
348   
349   
350   
351   
352   
353   
354   
355   
356   
357   
358   
359   
360   
361   
362   
363   
364   
365   
366   
367   
368   
369   
370   
371   
372   
373   
374   
375   
376   
377   
378   
379   
380   
381   
382   
383   
384 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 

Table 4. Zero-shot generalization on 1K test instances of 32 unseen VRPs $N = 1 0 0$ ). Due to page limit, complete experimental results (including all 99 unseen variants) are deferred to Appendix H.

<table><tr><td rowspan="2">Method</td><td colspan="2">CVRPBP</td><td colspan="2">CVRPBL</td><td colspan="2">MDOCVRPBP</td><td colspan="2">MDOCVRPBPTW</td><td colspan="2">MDOCVRPBPL</td><td colspan="2">MDOCVRPBPLTW</td><td colspan="2">MDOCVRP</td><td colspan="2">MDOCVRP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>MVMoE</td><td>13.95%</td><td>14s</td><td>13.38%</td><td>14s</td><td>56.60%</td><td>33s</td><td>63.77%</td><td>38s</td><td>56.26%</td><td>34s</td><td>62.47%</td><td>40s</td><td>33.06%</td><td>31s</td><td>29.08%</td><td>31s</td></tr><tr><td>MTPOMO</td><td>13.71%</td><td>7s</td><td>13.44%</td><td>8s</td><td>42.94%</td><td>24s</td><td>47.09%</td><td>29s</td><td>41.98%</td><td>27s</td><td>46.26%</td><td>32s</td><td>23.38%</td><td>23s</td><td>27.46%</td><td>23s</td></tr><tr><td>ReLD-MTL</td><td>13.57%</td><td>9s</td><td>12.96%</td><td>10s</td><td>37.74%</td><td>27s</td><td>45.16%</td><td>34s</td><td>37.45%</td><td>29s</td><td>44.74%</td><td>36s</td><td>18.39%</td><td>25s</td><td>22.32%</td><td>25s</td></tr><tr><td>URS</td><td>12.95%</td><td>7s</td><td>12.65%</td><td>8s</td><td>24.44%</td><td>22s</td><td>26.31%</td><td>26s</td><td>24.22%</td><td>24s</td><td>26.19%</td><td>28s</td><td>15.05%</td><td>23s</td><td>22.01%</td><td>23s</td></tr><tr><td rowspan="2">Method</td><td colspan="2">MDCVRPL</td><td colspan="2">MDOCVRPTW</td><td colspan="2">MDOCVRPB</td><td colspan="2">MDOCVRPBL</td><td colspan="2">MDOCVRPLTW</td><td colspan="2">SPCTSP</td><td colspan="2">PDCVRP</td><td colspan="2">OPDCVRP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0,00%</td><td>1.6h</td><td>0.00%</td><td>1.6h</td><td>0.00%</td><td>1.6h</td></tr><tr><td>MVMoE</td><td>33.42%</td><td>34s</td><td>51.57%</td><td>40s</td><td>28.21%</td><td>26s</td><td>28.70%</td><td>28s</td><td>51.09%</td><td>43s</td><td colspan="2">-</td><td colspan="2">-</td><td colspan="2">-</td></tr><tr><td>MTPOMO</td><td>24.32%</td><td>25s</td><td>35.71%</td><td>29s</td><td>25.06%</td><td>20s</td><td>25.01%</td><td>21s</td><td>35.39%</td><td>31s</td><td colspan="2">-</td><td colspan="2">-</td><td colspan="2">-</td></tr><tr><td>ReLD-MTL</td><td>18.48%</td><td>28s</td><td>33.55%</td><td>33s</td><td>18.50%</td><td>21s</td><td>18.50%</td><td>22s</td><td>33.26%</td><td>35s</td><td colspan="2">-</td><td colspan="2">-</td><td colspan="2">-</td></tr><tr><td>URS</td><td>15.32%</td><td>26s</td><td>26.05%</td><td>25s</td><td>15.17%</td><td>18s</td><td>14.90%</td><td>20s</td><td>25.59%</td><td>27s</td><td>-2.37%</td><td>5s</td><td>-1.47%</td><td>5s</td><td>4.93%</td><td>5s</td></tr><tr><td rowspan="2">Method</td><td colspan="2">ACVRPL</td><td colspan="2">ACVRPBL</td><td colspan="2">ACVRPB</td><td colspan="2">ACVRPBTW</td><td colspan="2">APDTSP</td><td colspan="2">AMDCVRPL</td><td colspan="2">AMDCVRP</td><td colspan="2">APDCVRP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>5.1m</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>-2.24%</td><td>1.9m</td><td>6.18%</td><td>2.4m</td><td>7.10%</td><td>2.1m</td><td>9.95%</td><td>2.2m</td><td>6.21%</td><td>1m</td><td>7.15%</td><td>6.1m</td><td>8.11%</td><td>5.3m</td><td>7.03%</td><td>1.2m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">ACVRPLTW</td><td colspan="2">ACVRPBLTW</td><td colspan="2">ACVRPBLPTW</td><td colspan="2">AOCVRPBLTW</td><td colspan="2">ACVRPTW</td><td colspan="2">ACVRPBTW</td><td colspan="2">AOPDCVRP</td><td colspan="2">AOP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>5.1</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>12.22%</td><td>2.7m</td><td>10.18%</td><td>2.4m</td><td>13.82%</td><td>2.8m</td><td>14.32%</td><td>2.5m</td><td>11.28%</td><td>2.5m</td><td>14.52%</td><td>2.6m</td><td>11.13%</td><td>1.2m</td><td>-5.47%</td><td>1.2m</td></tr></table>

Table 5. Generalization on large-scale instances of different VRP variants. All models are trained on instances of size $N = 1 0 0$ and employ instance augmentation $\times 8$ . HGS-PyVRP can process all instances of a given size concurrently to ensure a fair comparison.

<table><tr><td>Problem</td><td>Method</td><td>N = 1000
Obj.(Gap)</td><td>Time</td><td>N = 2000
Obj.(Gap)</td><td>Time</td><td>N = 3000
Obj.(Gap)</td><td>Time</td><td>N = 4000
Obj.(Gap)</td><td>Time</td><td>N = 5000
Obj.(Gap)</td><td>Time</td></tr><tr><td rowspan="5">CVRPTW</td><td>HGS-PyVRP</td><td>159.35(0.00%)</td><td>20m</td><td>337.15(0.00%)</td><td>40m</td><td>516.97(0.00%)</td><td>1h</td><td>618.58(0.00%)</td><td>2.7h</td><td>787.25(0.00%)</td><td>3.3h</td></tr><tr><td>MTPOMO</td><td>219.22(37.57%)</td><td>1.7m</td><td>488.40(44.86%)</td><td>14.6m</td><td>773.00(49.53%)</td><td>49.5m</td><td>975.61(57.72%)</td><td>1.9h</td><td>OOM</td><td></td></tr><tr><td>MVMoE</td><td>233.53(46.55%)</td><td>1.7m</td><td>624.73(85.30%)</td><td>14.6m</td><td>1094.25(111.67%)</td><td>51.9m</td><td>1486.34(140.28%)</td><td>2h</td><td>OOM</td><td></td></tr><tr><td>ReLD-MTL</td><td>183.31(15.04%)</td><td>1.6m</td><td>396.28(17.54%)</td><td>13.4m</td><td>613.58(18.69%)</td><td>45.7m</td><td>748.65(21.03%)</td><td>1.8h</td><td>OOM</td><td></td></tr><tr><td>URS</td><td>172.58(8.30%)</td><td>1.3m</td><td>365.27(8.34%)</td><td>11.1m</td><td>559.03(8.14%)</td><td>30.1m</td><td>673.63(8.90%)</td><td>1.2h</td><td>853.79(8.45%)</td><td>2.3h</td></tr><tr><td rowspan="5">CVRPB</td><td>HGS-PyVRP</td><td>36.52(0.00%)</td><td>20m</td><td>51.80(0.00%)</td><td>40m</td><td>75.12(0.00%)</td><td>1h</td><td>93.52(0.00%)</td><td>2.7h</td><td>112.21(0.00%)</td><td>3.3h</td></tr><tr><td>MTPOMO</td><td>45.92(25.77%)</td><td>1.1m</td><td>83.50(61.20%)</td><td>8.8m</td><td>136.52(81.73%)</td><td>31.4m</td><td>197.99(111.70%)</td><td>1.2h</td><td>OOM</td><td></td></tr><tr><td>MVMoE</td><td>98.60(170.03%)</td><td>1m</td><td>250.47(383.57%)</td><td>8.1m</td><td>332.93(343.19%)</td><td>28.9m</td><td>423.19(352.49%)</td><td>1.1h</td><td>OOM</td><td></td></tr><tr><td>ReLD-MTL</td><td>38.52(5.48%)</td><td>1m</td><td>59.21(14.30%)</td><td>8m</td><td>88.36(17.62%)</td><td>28.3m</td><td>114.88(22.83%)</td><td>1.1h</td><td>OOM</td><td></td></tr><tr><td>URS</td><td>34.86(-4.54%)</td><td>40s</td><td>50.39(-2.72%)</td><td>5.1m</td><td>72.01(-4.15%)</td><td>17.3m</td><td>91.74(-1.91%)</td><td>41.2m</td><td>111.93(-0.24%)</td><td>1.3h</td></tr><tr><td rowspan="5">OCVRPTW</td><td>HGS-PyVRP</td><td>90.91(0.00%)</td><td>20m</td><td>164.00(0.00%)</td><td>40m</td><td>224.16(0.00%)</td><td>1h</td><td>299.45(0.00%)</td><td>2.7h</td><td>367.86(0.00%)</td><td>3.3h</td></tr><tr><td>MTPOMO</td><td>147.14(61.85%)</td><td>1.6m</td><td>297.53(81.43%)</td><td>12.8m</td><td>437.73(95.28%)</td><td>44.3m</td><td>601.63(100.91%)</td><td>1.7h</td><td>OOM</td><td></td></tr><tr><td>MVMoE</td><td>147.35(62.09%)</td><td>1.6m</td><td>389.60(137.57%)</td><td>12.9m</td><td>634.11(182.88%)</td><td>44.7m</td><td>899.18(200.28%)</td><td>1.7h</td><td>OOM</td><td></td></tr><tr><td>ReLD-MTL</td><td>110.00(21.01%)</td><td>1.5m</td><td>208.60(27.20%)</td><td>12.3m</td><td>292.70(30.57%)</td><td>42m</td><td>394.42(31.72%)</td><td>1.6h</td><td>OOM</td><td></td></tr><tr><td>URS</td><td>98.76(8.63%)</td><td>1m</td><td>178.07(8.58%)</td><td>7.6m</td><td>243.88(8.80%)</td><td>25.4m</td><td>325.64(8.74%)</td><td>1h</td><td>398.72(8.39%)</td><td>1.9h</td></tr></table>
when fully specialized. Meanwhile, mixing all problems into the training of URS incurs only a minor performance drop, yet still delivers competitive performance across all problems, demonstrating that potential connections among different problems are effectively captured by URS. 
Generalization to Unseen VRPs We evaluate our URS model on zero-shot generalization across 44 well-studied and 55 less-explored VRP variants. The detailed results are presented in Table 4 and Appendix H. On widely-studied constraints (i.e., C, O, L, TW, MD, B, BP), URS continues to deliver high-quality solutions. Notably, URS merely selects the next node from a given feasible candidates without any additional domain knowledge, making its performance especially meaningful. Benefiting from our UDR, URS 
can directly address more complex variants while maintaining solution quality, which compounds asymmetry and PD relations. Without any fine-tuning, URS still produces highquality solutions for 99 unseen VRP variants, highlighting its strong zero-shot generalization. 
Scalability to Larger-scale VRP Variants We conduct experiments on large-scale instances to validate URS’s crossproblem scalability. We generate a new large-scale (1K-5K nodes) synthetic dataset for distinct problems (CVRPTW, CVRPB, and OCVRPTW). Each dataset includes 16 instances. We adhere to the capacity constraints specified in TAM (Hou et al., 2022). As shown in Table 5, URS consistently achieves the best solution quality, complemented by remarkably fast inference times, across various 
7 
385   
386   
387   
388   
389   
390   
391   
392   
393   
394   
395   
396   
397   
398   
399   
400   
401   
402   
403   
404   
405   
406   
407   
408   
409   
410   
411   
412   
413   
414   
415   
416   
417   
418   
419   
420   
421   
422   
423   
424   
425   
426   
427   
428   
429   
430   
431   
432   
433   
434   
435   
436   
437   
438   
439 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 

Table 6. Comparison on Set-XXL with $N \in [ 3 0 0 0 , 7 0 0 0 ]$ . All models are trained on instances of size $N = 1 0 0$ .

<table><tr><td>Type</td><td>Method</td><td>Leuven1 (N=3000)</td><td>Leuven2 (N=4000)</td><td>Antwerp1 (N=6000)</td><td>Antwerp2 (N=7000)</td><td>Avg.gap</td></tr><tr><td rowspan="4">Specialist</td><td>LEHD greedy</td><td>16.60%</td><td>34.86%</td><td>14.66%</td><td>22.77%</td><td>22.22%</td></tr><tr><td>BQ greedy</td><td>18.53%</td><td>30.70%</td><td>16.48%</td><td>27.67%</td><td>23.34%</td></tr><tr><td>POMO aug×8</td><td>460.32%</td><td>202.17%</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td>ELG aug×8</td><td>12.12%</td><td>21.52%</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td rowspan="8">Generalist</td><td>MTPOMO aug×8</td><td>67.72%</td><td>87.31%</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td>MVMoE/4E aug×8</td><td>299.10%</td><td>170.10%</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td>MVMoE/4E-L aug×8</td><td>182.28%</td><td>127.07%</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td>RF-MVMoE aug×8</td><td>57.30%</td><td>160.27%</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td>RF-TF aug×8</td><td>26.90%</td><td>45.56%</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td>CaDA aug×8</td><td>1035.51%</td><td>OOM</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td>ReLD-MTL aug×8</td><td>17.00%</td><td>30.59%</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td>GOAL greedy</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>-</td></tr><tr><td></td><td>URS aug×8</td><td>11.57%</td><td>17.80%</td><td>9.23%</td><td>14.95%</td><td>13.39%</td></tr></table>
problem instances among all comparable multi-task neural solvers. Remarkably, URS already outperforms the strong HGS-PyVRP with a carefully manual design on large-scale CVRPB instances. These extensive results show URS’s strong scalability to large-scale complex VRP variants. 
Results on Benchmark Dataset We further evaluate URS on benchmark instances from CVRPLIB Set-X (Uchoa et al., 2017) and CVRPLib XXL (Arnold et al., 2019). We compare URS against several representative specialist and generalist solvers. As presented in Table 6, URS can significantly outperform existing generalist neural routing solvers in large-scale benchmark instances. Notably, as a generalist solver, URS also greatly surpasses many representative specialist models (e.g., LEHD, BQ, and ELG). In addition, the detailed results (Appendix I.2) on the Set-X dataset show that URS still maintains its position as the best overall performance. These results further underscore the practical applicability of URS in real-world large-scale scenarios. 
# 5. Ablation Study
In this section, we conduct an ablation study across 11 seen VRP variants to validate the effect of key components of URS, mainly including: (1) without node-type indicator $\boldsymbol { \xi }$ of UDR; (2) without problem state $C _ { t }$ ; (3) effects of different priors in MBM ; (4) effects of alternative decoder architectures. As detailed in Figure 4. URS maintains its competitive performance without $C _ { t }$ . And its effectiveness is further improved by the inclusion of a node-type indicator $\boldsymbol { \xi }$ . For MBM, the results reveal that three priors $\{ D , D ^ { \mathrm { T } } , R \}$ are complementary, and their combined use allows MBM to effectively learn the geometric and relational biases inherent in various problems, resulting in superior overall performance. Furthermore, replacing the decoder with POMO or ReLD confirms the strong cross-problem robustness of URS, while integrating the decoder introduced in Zhou et al. (2024a) further improves cross-problem performance. More detailed results and analyses can be found in Appendix J. 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-13/90b59445-9787-40c5-9062-a92383c20222/60fad4f798e381d516ffb2c47e127df6c191ba6e128b48ac224d5aaeb7bf4f70.jpg)


Figure 4. Effects of different components of URS.


Table 7. Results comparing performance with and without MBM, WEIGHT(λ), and $\bar { \mathrm { B I A S } } ( \bar { \lambda } )$ on seen and unseen variants.

<table><tr><td rowspan="2">MBM</td><td rowspan="2">WEIGHT(λ)</td><td rowspan="2">BIAS(λ)</td><td colspan="2">Avg.gap</td><td rowspan="2">Best Solution</td></tr><tr><td>Seen (11)</td><td>Unseen (11)</td></tr><tr><td>×</td><td>?</td><td>?</td><td>6.33%</td><td>10.41%</td><td>3/22</td></tr><tr><td>?</td><td>×</td><td>×</td><td>4.20%</td><td>8.12%</td><td>0/22</td></tr><tr><td>?</td><td>×</td><td>?</td><td>3.84%</td><td>8.02%</td><td>1/22</td></tr><tr><td>?</td><td>?</td><td>×</td><td>3.63%</td><td>7.81%</td><td>1/22</td></tr><tr><td>?</td><td>?</td><td>?</td><td>2.74%</td><td>7.05%</td><td>17/22</td></tr></table>
To further evaluate the necessity of MBM, WEIGHT(λ), and BIAS(λ), we conduct tests on 11 seen problems and 11 additional unseen CVRP variants widely studied in (Liu et al., 2024). As shown in Table 7, the full URS model achieves the lowest average gap on both seen and unseen problems, and provides the best solution on the majority of tasks (17/22). This proves that the synergy of these components is essential for robust cross-problem zero-shot generalization. We provide detailed results in Appendix J.5. 
# 6. Conclusion, Limitation, and Future Work
In this work, we propose a novel RL-based URS for crossproblem zero-shot generalization. The core of URS is a Unified Data Representation (UDR), which broadens problem coverage by substituting problem enumeration with data unification. Its generalization capability is further enhanced by a Mixed Bias Module (MBM) and a problem-conditioned parameter generator. Extensive experiments show that URS can produce high-quality solutions for 110 VRP variants without any fine-tuning, including 99 unseen variants, significantly broadening the problem coverage and reducing reliance on domain expertise. 
Limitation and Future Work While URS achieves impressive results on a wide range of VRP variants, its current training scheme does not accommodate their inherent differences in complexity. In the future, we aim to explore more effective strategies to improve training efficiency. 
8 
440   
441   
442   
443   
444   
445   
446   
447   
448   
449   
450   
451   
452   
453   
454   
455   
456   
457   
458   
459   
460   
461   
462   
463   
464   
465   
466   
467   
468   
469   
470   
471   
472   
473   
474   
475   
476   
477   
478   
479   
480   
481   
482   
483   
484   
485   
486   
487   
488   
489   
490   
491   
492   
493   
494 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# Impact Statement
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 
# References


Arnold, F., Gendreau, M., and Sorensen, K. Efficiently ¨ solving very large-scale routing problems. Computers & operations research, 107:32–42, 2019. 




Bello, I., Pham, H., Le, Q. V., Norouzi, M., and Bengio, S. Neural combinatorial optimization with reinforcement learning. arXiv preprint arXiv:1611.09940, 2016. 




Bengio, Y., Lodi, A., and Prouvost, A. Machine learning for combinatorial optimization: a methodological tour d’horizon. European Journal of Operational Research, 290(2):405–421, 2021. 




Berto, F., Hua, C., Zepeda, N. G., Hottung, A., Wouda, N., Lan, L., Park, J., Tierney, K., and Park, J. Routefinder: Towards foundation models for vehicle routing problems. Transactions on Machine Learning Research, 2025. 




Chen, Y., Chen, R., Luo, F., and Wang, Z. Improving generalization of neural combinatorial optimization for vehicle routing problems via test-time projection learning. Advances in Neural Information Processing Systems, 2025. 




Drakulic, D., Michel, S., Mai, F., Sors, A., and Andreoli, J.- M. Bq-nco: Bisimulation quotienting for efficient neural combinatorial optimization. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 




Drakulic, D., Michel, S., and Andreoli, J.-M. Goal: A generalist combinatorial optimization agent learner. In The Thirteenth International Conference on Learning Representations, 2025. 




Fang, H., Song, Z., Weng, P., and Ban, Y. Invit: A generalizable routing problem solver with invariant nested view transformer. In International Conference on Machine Learning, 2024. 




Fu, Z.-H., Qiu, K.-B., and Zha, H. Generalize a small pre-trained model to arbitrarily large tsp instances. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 7474–7482, 2021. 




Gao, C., Shang, H., Xue, K., Li, D., and Qian, C. Towards generalizable neural solvers for vehicle routing problems via ensemble with transferrable local policy. In International Joint Conference on Artificial Intelligence, 2024. 




Ha, D., Dai, A., and Le, Q. V. Hypernetworks. arXiv preprint arXiv:1609.09106, 2016. 




He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016. 




Helsgaun, K. An extension of the lin-kernighan-helsgaun tsp solver for constrained traveling salesman and vehicle routing problems. Roskilde: Roskilde University, 12, 2017. 




Hou, Q., Yang, J., Su, Y., Wang, X., and Deng, Y. Generalize learned heuristics to solve large-scale vehicle routing problems in real-time. In The Eleventh International Conference on Learning Representations, 2022. 




Huang, Z., Zhou, J., Cao, Z., and Xu, Y. Rethinking light decoder-based solvers for vehicle routing problems. International Conference on Learning Representations, 2025. 




Kim, M., Park, J., and Park, J. Sym-nco: Leveraging symmetricity for neural combinatorial optimization. Advances in Neural Information Processing Systems, 35: 1936–1949, 2022. 




Kong, D., Ma, Y., Cao, Z., Yu, T., and Xiao, J. Efficient neural collaborative search for pickup and delivery problems. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 




Kool, W., van Hoof, H., and Welling, M. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019. 




Kwon, Y.-D., Choo, J., Kim, B., Yoon, I., Gwon, Y., and Min, S. Pomo: Policy optimization with multiple optima for reinforcement learning. Advances in Neural Information Processing Systems, 33:21188–21198, 2020. 




Kwon, Y.-D., Choo, J., Yoon, I., Park, M., Park, D., and Gwon, Y. Matrix encoding networks for neural combinatorial optimization. Advances in Neural Information Processing Systems, 34:5138–5149, 2021. 




Li, H., Liu, F., Zheng, Z., Zhang, Y., and Wang, Z. Cada: Cross-problem routing solver with constraint-aware dualattention. In Proceedings of the 42nd International Conference on Machine Learning, 2025a. 




Li, J., Xin, L., Cao, Z., Lim, A., Song, W., and Zhang, J. Heterogeneous attentions for solving pickup and delivery problem via deep reinforcement learning. IEEE Transactions on Intelligent Transportation Systems, 23 (3):2306–2315, 2021. 




Li, J., Ma, Y., Gao, R., Cao, Z., Lim, A., Song, W., and Zhang, J. Deep reinforcement learning for solving the heterogeneous capacitated vehicle routing problem. IEEE Transactions on Cybernetics, 52(12):13572–13585, 2022. 


9 
495   
496   
497   
498   
499   
500   
501   
502   
503   
504   
505   
506   
507   
508   
509   
510   
511   
512   
513   
514   
515   
516   
517   
518   
519   
520   
521   
522   
523   
524   
525   
526   
527   
528   
529   
530   
531   
532   
533   
534   
535   
536   
537   
538   
539   
540   
541   
542   
543   
544   
545   
546   
547   
548   
549 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 


Li, K., Liu, F., Wang, Z., and Zhang, Q. Destroy and repair using hyper-graphs for routing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 18341–18349, 2025b. 




Li, Y., Guo, J., Wang, R., and Yan, J. T2t: From distribution learning in training to gradient search in testing for combinatorial optimization. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 




Lin, X., Yang, Z., and Zhang, Q. Pareto set learning for neural multi-objective combinatorial optimization. In 10th International Conference on Learning Representations, 2022. 




Lin, Z., Wu, Y., Zhou, B., Cao, Z., Song, W., Zhang, Y., and Jayavelu, S. Cross-problem learning for solving vehicle routing problems. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pp. 6958–6966, 2024. 




Liu, F., Lin, X., Wang, Z., Zhang, Q., Xialiang, T., and Yuan, M. Multi-task learning for routing problem with cross-problem zero-shot generalization. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1898–1908, 2024. 




Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 




Luo, F., Lin, X., Liu, F., Zhang, Q., and Wang, Z. Neural combinatorial optimization with heavy decoder: Toward large scale generalization. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 




Luo, F., Lin, X., Wu, Y., Wang, Z., Xialiang, T., Yuan, M., and Zhang, Q. Boosting neural combinatorial optimization for large-scale vehicle routing problems. In The Thirteenth International Conference on Learning Representations, 2025a. 




Luo, F., Lin, X., Zhong, M., Liu, F., Wang, Z., Sun, J., and Zhang, Q. Learning to insert for constructive neural vehicle routing solver. Advances in Neural Information Processing Systems, 2025b. 




Luo, F., Wu, Y., Zheng, Z., and Wang, Z. Rethinking neural combinatorial optimization for vehicle routing problems with different constraint tightness degrees. Advances in Neural Information Processing Systems, 2025c. 




Nazari, M., Oroojlooy, A., Snyder, L., and Takac, M. Rein-′ forcement learning for solving the vehicle routing problem. Advances in Neural Information Processing Systems, 31, 2018. 




Perron, L. and Furnon, V. Or-tools, 2023. URL https: //developers.google.com/optimization/. 




Pirnay, J. and Grimm, D. G. Self-improvement for neural combinatorial optimization: Sample without replacement, but improvement. Transactions on Machine Learning Research, 2024. 




Qiu, R., Sun, Z., and Yang, Y. Dimes: A differentiable meta solver for combinatorial optimization problems. Advances in Neural Information Processing Systems, 35: 25531–25546, 2022. 




Sar, K. and Ghadimi, P. A systematic literature review of the vehicle routing problem in reverse logistics operations. Computers & Industrial Engineering, 177:109011, 2023. 




Sun, R., Zheng, Z., and Wang, Z. Learning encodings for constructive neural combinatorial optimization needs to regret. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 20803–20811, 2024. 




Sun, Z. and Yang, Y. Difusco: Graph-based diffusion solvers for combinatorial optimization. Advances in Neural Information Processing Systems, 36:3706–3731, 2023. 




Tiwari, K. V. and Sharma, S. K. An optimization model for vehicle routing problem in last-mile delivery. Expert Systems with Applications, 222:119789, 2023. 




Toth, P. and Vigo, D. The vehicle routing problem. SIAM, 2002. 




Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., and Subramanian, A. New benchmark instances for the capacitated vehicle routing problem. European Journal of Operational Research, 257(3):845–858, 2017. 




Ulyanov, D., Vedaldi, A., and Lempitsky, V. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022, 2016. 




Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, ?., and Polosukhin, I. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017. 




Vidal, T. Hybrid genetic search for the cvrp: Open-source implementation and swap* neighborhood. Computers & Operations Research, 140:105643, 2022. 




Vinyals, O., Fortunato, M., and Jaitly, N. Pointer networks. Advances in Neural Information Processing Systems, 28, 2015. 




Wang, C., Fu, Z.-H., Lu, P., and Yu, T. Efficient training of multi-task neural solver for combinatorial optimization. Transactions on Machine Learning Research, 2025a. 




Wang, Y., Jia, Y.-H., Chen, W.-N., and Mei, Y. Distanceaware attention reshaping for enhancing generalization of neural solvers. IEEE Transactions on Neural Networks and Learning Systems, 2025b. 


550   
551   
552   
553   
554   
555   
556   
557   
558   
559   
560   
561   
562   
563   
564   
565   
566   
567   
568   
569   
570   
571   
572   
573   
574   
575   
576   
577   
578   
579   
580   
581   
582   
583   
584   
585   
586   
587   
588   
589   
590   
591   
592   
593   
594   
595   
596   
597   
598   
599   
600   
601   
602   
603   
604 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 


Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256, 1992. 




Wouda, N. A., Lan, L., and Kool, W. Pyvrp: A highperformance vrp solver package. INFORMS Journal on Computing, 36(4):943–955, 2024. 




Xiao, Y., Wang, D., Li, B., Chen, H., Pang, W., Wu, X., Li, H., Xu, D., Liang, Y., and Zhou, Y. Reinforcement learning-based nonautoregressive solver for traveling salesman problems. IEEE Transactions on Neural Networks and Learning Systems, 2024a. 




Xiao, Y., Wang, D., Li, B., Wang, M., Wu, X., Zhou, C., and Zhou, Y. Distilling autoregressive models to obtain high-performance non-autoregressive solvers for vehicle routing problems with faster inference speed. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 20274–20283, 2024b. 




Xin, L., Song, W., Cao, Z., and Zhang, J. Step-wise deep learning models for solving routing problems. IEEE Transactions on Industrial Informatics, 17(7):4861–4871, 2020. 




Ye, H., Wang, J., Cao, Z., Liang, H., and Li, Y. Deepaco: Neural-enhanced ant systems for combinatorial optimization. Advances in neural information processing systems, 36:43706–43728, 2023. 




Ye, H., Wang, J., Liang, H., Cao, Z., Li, Y., and Li, F. Glop: Learning global partition and local construction for solving large-scale routing problems in real-time. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 20284–20292, 2024. 




Zheng, Y., Luo, F., Wang, Z., Wu, Y., and Zhou, Y. Mtlkd: Multi-task learning via knowledge distillation for generalizable neural vehicle routing solver. Advances in Neural Information Processing Systems, 2025. 




Zheng, Z., Yao, S., Wang, Z., Tong, X., Yuan, M., and Tang, K. Dpn: Decoupling partition and navigation for neural solvers of min-max vehicle routing problems. In Proceedings of the 41st International Conference on Machine Learning, pp. 61559–61592, 2024a. 




Zheng, Z., Zhou, C., Xialiang, T., Yuan, M., and Wang, Z. Udc: A unified neural divide-and-conquer framework for large-scale combinatorial optimization problems. In Thirty-eighth Conference on Neural Information Processing Systems, 2024b. 




Zhou, C., Lin, X., Wang, Z., Tong, X., Yuan, M., and Zhang, Q. Instance-conditioned adaptation for large-scale generalization of neural routing solver. arXiv preprint arXiv:2405.01906, 2024a. 




Zhou, C., Lin, X., Wang, Z., and Zhang, Q. Learning to reduce search space for generalizable neural routing solver. arXiv preprint arXiv:2503.03137, 2025. 




Zhou, J., Cao, Z., Wu, Y., Song, W., Ma, Y., Zhang, J., and Xu, C. Mvmoe: multi-task vehicle routing solver with mixture-of-experts. In Proceedings of the 41st International Conference on Machine Learning, pp. 61804– 61824, 2024b. 


11 
605   
606   
607   
608   
609   
610   
611   
612   
613   
614   
615   
616   
617   
618   
619   
620   
621   
622   
623   
624   
625   
626   
627   
628   
629   
630   
631   
632   
633   
634   
635   
636   
637   
638   
639   
640   
641   
642   
643   
644   
645   
646   
647   
648   
649   
650   
651   
652   
653   
654   
655   
656   
657   
658   
659 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# A. Related Work
# A.1. Neural Routing Solvers with Single-Task Learning
The NCO methods have emerged as a promising paradigm for tackling various routing problems, achieving impressive empirical performance. Early seminal work based on Pointer Networks ignited interest in learning end-to-end construction policies for TSP and CVRP (Vinyals et al., 2015; Bello et al., 2016; Nazari et al., 2018). Subsequently, the Transformer-style heavy-encoder and light-decoder auto-regressive architecture has become the dominant paradigm (Kool et al., 2019), while POMO (Kwon et al., 2020) exploits permutation symmetry to enhance exploration diversity. Many advancements within this paradigm have pushed promising results on small instances (e.g., 100-node TSPs) (Xin et al., 2020; Kim et al., 2022; Xiao et al., 2024a;b; Sun et al., 2024). However, models trained on a fixed instance scale often degrade sharply when evaluated out of scale. To mitigate the poor generalization performance, recent studies introduce auxiliary local policies (Gao et al., 2024), adaptive perception across instance scales (Zhou et al., 2024a), or different search space reduction methods (Fang et al., 2024; Zhou et al., 2025; Chen et al., 2025). In parallel, ”heavy decoder” designs that dynamically recompute embeddings at each construction step (Drakulic et al., 2023; Luo et al., 2023) demonstrate strong generalization on large-scale instances. To improve the performance on large-scale routing instances, complementary directions are reducing solving difficulty via problem decomposition (Zheng et al., 2024b; Ye et al., 2024; Li et al., 2025b) or adopting non-autoregressive generation augmented with additional searches (e.g., Monte Carlo tree search (MCTS) and 2-opt) (Fu et al., 2021; Sun & Yang, 2023; Qiu et al., 2022; Li et al., 2023; Ye et al., 2023). Moreover, heterogeneous vehicle routing (Li et al., 2022), asymmetric distance settings (Kwon et al., 2021), multiple VRPs (Zheng et al., 2024a), and pickup–delivery problems (Li et al., 2021; Kong et al., 2024) have each motivated tailored neural designs, which have yielded competitive results. Despite these advancements, they still train a specialist model per problem type, limiting cross-problem transfer. We instead focus on a unified neural routing solver that aims to generalize across multiple routing problems without retraining from scratch for each specification. 
# A.2. Neural Routing Solvers with Multi-Task Learning
To address the challenge of cross-problem generalization, growing attention has shifted to multi-task learning capable of adapting to diverse routing problems. Existing efforts can broadly fall into two categories: (1) constraint combination-based multi-task learning and (2) adapter-based fine-tuning frameworks. The former line (e.g., MTPOMO (Liu et al., 2024)) treats VRP variants as combinations of a predefined set of attributes and trains a shared backbone, achieving promising results on up to 16 VRP variants. Follow-up work extends this paradigm by introducing mixture-of-experts (MoE) modules (Zhou et al., 2024b), heavy decoder designs plus knowledge distillation (Zheng et al., 2025), and constraint-aware dual-attention (Li et al., 2025a) or updated Transformer structure (Berto et al., 2025) to improve model performance. While this enables knowledge sharing across problems, its coverage is inherently bounded by the hand-specified attribute set. The second family trains a shared backbone and performs adaptation to each variant via lightweight adapters. For instance, Lin et al. (2024) pretrain a TSP backbone and adapt to new variants through adapter-based fine-tuning. Drakulic et al. (2025) integrate edge features and adopt supervised multi-task pretraining for obtaining a powerful backbone before fine-tuning on novel tasks. Wang et al. (2025a) propose a multi-armed bandit strategy to realize more efficient multi-task training. Although these reduce retraining cost, they fail to achieve zero-shot generalization. In contrast, we focus on a constructive neural routing solver supported with zero-shot capability: once trained, it can directly generate high-quality solutions for previously unseen problems without any fine-tuning. 
12 
660   
661   
662   
663   
664   
665   
666   
667   
668   
669   
670   
671   
672   
673   
674   
675   
676   
677   
678   
679   
680   
681   
682   
683   
684   
685   
686   
687   
688   
689   
690   
691   
692   
693   
694   
695   
696   
697   
698   
699   
700   
701   
702   
703   
704   
705   
706   
707   
708   
709   
710   
711   
712   
713   
714 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# B. Setups of VRP Variants
In this paper, we consider 110 VRP variants which may simultaneously have one or more constraints from the following categories: (1) Capacity (C); (2) Open Route (O); (3) Backhaul (B); (4) Backhaul and Priority (BP); (5) Duration Limit (L); (6) Time Windows (TW); (7) Multi-Depot (MD); (8) Prize Collecting (PC); (9) Asymmetric (A); (10) Pickup and Delivery (PD). The node coordinates are randomly sampled from the unit square when available (in symmetric geometric cases, i.e., $d _ { i j } = d _ { j i } )$ ), following AM (Kool et al., 2019). Below, we provide a comprehensive description of the data generation process for each constraint. 
Capacity (C) The selection of the vehicle capacity $\mathcal { C }$ is critical, as prior work (Luo et al., 2025c) demonstrates its significant impact on the solution structure. For PDVRP variants (e.g., PDCVRP, OPDCVRP, APDCVRP, and AOPDCVRP), an excessively large $\mathcal { C }$ causes the solution to approximate that of a PDTSP instance. Conversely, an overly restrictive $\mathcal { C }$ promotes a simplistic, repetitive Pickup-Delivery sequence. To ensure structurally sound solutions, we set $\scriptstyle { \mathcal { C } } = 2 0$ for PDCVRP variants. For all other VRP variants, we adopt the convention from AM (Kool et al., 2019), setting $\scriptstyle { \mathcal { C } } = 5 0$ . 
Open Route (O) The open route setting implies that vehicles are not required to return to the depot after completing their service tours. This does not alter the data generation, and it specifically affects the final cost evaluation by excluding the travel distance from the last customer back to the depot. 
Backhaul (B) The standard CVRP involves vehicles delivering goods from a depot to a set of customers with positive demands. The introduction of backhauls extends this formulation to the CVRP with Backhauls (CVRPB), where vehicles must also load goods from backhaul customers and return them to the depot, in addition to serving the standard linehaul (delivery) customers. Following the experimental setup of MTPOMO (Liu et al., 2024), we generate customer demands by sampling from a discrete uniform distribution over the integers $\{ 1 , \ldots , 9 \}$ . Subsequently, $20 \%$ of these nodes are randomly designated as backhaul customers, and their corresponding demand values are negative, while the remainder are classified as linehaul customers. Notably, no precedence constraints are imposed between the servicing of linehaul and backhaul customers.To ensure the feasibility of the solutions constructed by our neural solver, we adhere to the MVMoE (Zhou et al., 2024b) framework’s policy that the first customer visited must be a linehaul node. 
Backhaul and Priority (BP) This constraint imposes a strict precedence requirement on the standard CVRPB, dictating that all linehaul customers must be serviced prior to any backhaul customers. Consequently, once a vehicle serves a backhaul customer, it is prohibited from visiting any subsequent linehaul nodes within the same route. If the first customer served is a backhaul node, the vehicle’s load is initialized to zero. This ensures the neural solver can construct a feasible solution. 
Duration Limit (L) Following MTPOMO (Liu et al., 2024), we impose a duration limit (i.e., the maximum length of each vehicle route) of 3 in symmetric cases. Since the node coordinates are randomly sampled from the unit square, the maximum possible distance between any two nodes is $\sqrt { 2 }$ . The setting is sufficient to ensure a vehicle can deliver goods to at least one customer and return to the depot, thus guaranteeing the feasibility of the solution. For the setting of asymmetric cases, we provide details in the description of the following asymmetric instances. 
Time Windows (TW) Consistent with the MVMoE (Zhou et al., 2024b), we set time windows of the depot to $[ e _ { 0 } , l _ { 0 } ] { = } [ 0 , 3 ]$ and its service time to 0. The service time $s _ { i }$ for each customer node is set to 0.2. For the time windows for each customer node, we generate them following the methodology outlined in MVMoE (Zhou et al., 2024b). 
Multi-Depot (MD) The generation of coordinates for the multiple depots aligns with that of other nodes, with the number of depots set to 3 (Berto et al., 2025). A vehicle is required to return to the same depot it departed from. To guarantee that each starting node is considered in this multi-depot variant, we set the number of trajectories for an instance to be the product of the number of depots and the number of customer nodes, maximizing the exploration of high-quality solutions. 
Prize Collecting (PC) Under the Prize Collecting condition, each node (excluding the depot) is assigned both a prize and a penalty. A prize is collected for each visited node, while a penalty is incurred for each unvisited node. The objective is to minimize the total travel distance plus the sum of penalties for all unvisited nodes. A key constraint is that a minimum amount of total prize must be collected before the vehicle is permitted to return to the depot and terminate its route. Consistent 
13 
715   
716   
717   
718   
719   
720   
721   
722   
723   
724   
725   
726   
727   
728   
729   
730   
731   
732   
733   
734   
735   
736   
737   
738   
739   
740   
741   
742   
743   
744   
745   
746   
747   
748   
749   
750   
751   
752   
753   
754   
755   
756   
757   
758   
759   
760   
761   
762   
763   
764   
765   
766   
767   
768   
769 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
with the AM (Kool et al., 2019) framework, the prizes and penalties are generated as follows: prize $\sim$ Uniform $\textstyle \left( 0 , { \frac { 4 } { n } } \right)$ , an d penalty $\sim$ Uniform $\textstyle ( 0 , 3 \cdot { \frac { k _ { n } } { n } } )$ , where $n = 1 0 0$ and $k _ { n } = 4$ in our experiments. 
Asymmetric (A) We adopt the data generation method from MatNet (Kwon et al., 2021) to create an asymmetric distance matrix. Due to the small pairwise distances in the generated distance matrix, the parameters for duration limit and time windows set for symmetric cases are not applicable to the asymmetric problems. To ensure the constraints remain valid, we adjust the duration limit and depot end time to 0.6 and 1.0, respectively. 
Pickup and Delivery (PD) For PDP variants, we adhere to the methodology in Li et al. (2021). For a problem of size $N$ , node 0 is set to the depot, nodes 1 to $N / 2$ are designated as pickup nodes, while nodes from $N / 2 + 1$ to $N$ are designated as delivery nodes. For each pickup node $v _ { i }$ , its corresponding delivery node is $v _ { i + N / 2 }$ . A strict precedence constraint is imposed in PDP cases, requiring each vehicle to visit a pickup node before its corresponding delivery node. For PDCVRP variants, the demand for each delivery node is randomly sampled from a discrete uniform distribution over $\{ 1 , 2 , \ldots , 9 \}$ . The demand for each pickup node is then set to the negative of its corresponding delivery node’s demand. To ensure the feasibility of the constraints, the vehicle capacity is set to 20. 
14 
770   
771   
772   
773   
774   
775   
776   
777   
778   
779   
780   
781   
782   
783   
784   
785   
786   
787   
788   
789   
790   
791   
792   
793   
794   
795   
796   
797   
798   
799   
800   
801   
802   
803   
804   
805   
806   
807   
808   
809   
810   
811   
812   
813   
814   
815   
816   
817   
818   
819   
820   
821   
822   
823   
824 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# C. Unified Data Representation
In this section, we provide a detailed description of our UDR $U = \{ \mathbf { u } _ { i } \} _ { i = 0 } ^ { n }$ (see Table 8). Each node $\mathbf { u } _ { i }$ consists of three components, i.e., $\mathbf { u } _ { i } = \{ \rho _ { i } , \ : \omega _ { i } , \ : \xi _ { i } \}$ . where $\rho _ { i }$ , $\omega _ { i }$ , $\pmb { \xi } _ { i }$ are a positional identifier, unified attribute set, and node-type indicator, respectively. To distinguish linehaul/delivery from backhaul/pickup nodes, we assign negative demand values to backhaul/pickup nodes (positive for linehaul/delivery), making demand the only signed component, following (Liu et al., 2024; Zhou et al., 2024b). 

Table 8. Detailed description of each component in UDR.

<table><tr><td>Attribute Name</td><td>Symbol</td><td>Range</td><td>Description</td><td>Examples</td></tr><tr><td>Random Identifier</td><td>η</td><td>[0,1]</td><td>Random scalar for asymmetric problems.</td><td>ATSP</td></tr><tr><td>Node Coordinates</td><td>{x,y}</td><td>[0,1]2</td><td>2D coordinates of the node.</td><td>TSP, CVRP</td></tr><tr><td>Demand</td><td>δ</td><td>[-1,1]</td><td>Node demand (+ for delivery, ? for pickup).</td><td>CVRP, PDCVRP</td></tr><tr><td>Prize</td><td>ε</td><td>[0,1]</td><td>Prize collected for each node.</td><td>OP, PCTSP</td></tr><tr><td>Penalty</td><td>μ</td><td>[0,1]</td><td>Penalty incurred for each node.</td><td>PCTSP</td></tr><tr><td>Earliest Arrival Time</td><td>e</td><td>[0,1]</td><td>Earliest permissible arrival time.</td><td>CVRPTW</td></tr><tr><td>Latest Arrival Time</td><td>l</td><td>[0,1]</td><td>Latest permissible arrival time (deadline).</td><td>CVRPTW</td></tr><tr><td>Service Time</td><td>s</td><td>[0,1]</td><td>Time required for service at the node.</td><td>CVRPTW</td></tr><tr><td>Depot</td><td>-</td><td>{0,1}</td><td>Indicates if the node is a depot.</td><td>CVRP</td></tr><tr><td>Pickup Node</td><td>-</td><td>{0,1}</td><td>Indicates if the node is a pickup location.</td><td>PDTSP</td></tr><tr><td>Delivery Node</td><td>-</td><td>{0,1}</td><td>Indicates if the node is a delivery location.</td><td>CVRP, PDTSP</td></tr><tr><td>Node in Sub-routes</td><td>-</td><td>{0,1}</td><td>The solution π can have sub-routes.</td><td>CVRP</td></tr><tr><td>Node in Open Route</td><td>-</td><td>{0,1}</td><td>Vehicles need not return to the depot.</td><td>OCVRP</td></tr></table>
825   
826   
827   
828   
829   
830   
831   
832   
833   
834   
835   
836   
837   
838   
839   
840   
841   
842   
843   
844   
845   
846   
847   
848   
849   
850   
851   
852   
853   
854   
855   
856   
857   
858   
859   
860   
861   
862   
863   
864   
865   
866   
867   
868   
869   
870   
871   
872   
873   
874   
875   
876   
877   
878   
879 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# D. Multi-hot Problem Representation
Leveraging the UDR, each problem instance activates only a subset of the unified feature space. Thus, for any instance $\mathcal { G } _ { i }$ from different problems, we simply mark active (non-zero) feature slots with 1 to obtain a corresponding multi-hot problem representation $\lambda _ { i }$ . Different from raw feature values, it encodes presence only and provides conditional signals to the following position bias weight (see Equation (6)) and adaptive decoder (see Equation (9) and Equation (10)), helping the model distinguish variants and refine node selection. Here, we provide detailed representations for each problem in our experiments, and each problem is a 13-dimensional multi-hot vector. 
As shown in Table 9, we train URS on eleven classic and varied routing problems to ensure that each feature in the UDR has been seen at least once, i.e., $\mathcal { P } = \{ P _ { i } \} _ { i = 1 } ^ { 1 1 }$ . Our selection of the 11 training problems is guided by a ”minimum redundancy” principle: we ensure that most attributes in UDR appear in at least two distinct problem types during training, which helps prevent the model from overfitting a specific attribute to a single problem type. The only exception is the Penalty attribute, as preliminary experiments have shown that exposure in a single problem type is sufficient for generalization to variants such as PCTSP and SPCTSP. For more detailed analysis and experimental results, please see Appendix I.1. 
Then, we evaluate generalization on unseen problems on different combinations of ten constraint attributes (see Appendix B), and the detailed representations of each unseen problem can be found in Table 10 (unseen well-studied VRP variants) and Table 11 (unseen less-explored VRP variants). 

Table 9. The representations for seen routing problems in our experiments. The total number of problems is 11. In training, each feature in the UDR has been seen at least once. Here, RI represents the random identifier, and EAT, LAT, and ST indicate the earliest arrival time, the latest arrival time, and the service time, respectively.

<table><tr><td>Problem</td><td>Oracle</td><td>RI</td><td>Coord.</td><td>Demand</td><td>Prize</td><td>Penalty</td><td>EAT</td><td>LAT</td><td>ST</td><td>Depot</td><td>Pickup</td><td>Delivery</td><td>Sub-routes</td><td>Open Route</td></tr><tr><td>ATSP</td><td>LKH-3</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TSP</td><td>LKH-3</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OP</td><td>Compass</td><td></td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td>?</td><td></td><td></td><td></td><td></td></tr><tr><td>PCTSP</td><td>ILS</td><td></td><td>?</td><td></td><td>?</td><td>?</td><td></td><td></td><td></td><td>?</td><td></td><td></td><td></td><td></td></tr><tr><td>PDTSP</td><td>LKH-3</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td></td><td></td></tr><tr><td>ACVRP</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>CVRP</td><td>PyVRP</td><td></td><td>?</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>CVRPTW</td><td>PyVRP</td><td></td><td>?</td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>CVRPB</td><td>OR-Tools</td><td></td><td>?</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>OCVRP</td><td>LKH-3</td><td></td><td>?</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>OCVRPTW</td><td>OR-Tools</td><td></td><td>?</td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr></table>
16 
880   
881   
882   
883   
884   
885   
886   
887   
888   
889   
890   
891   
892   
893   
894   
895   
896   
897   
898   
899   
900   
901   
902   
903   
904   
905   
906   
907   
908   
909   
910   
911   
912   
913   
914   
915   
916   
917   
918   
919   
920   
921   
922   
923   
924   
925   
926   
927   
928   
929   
930   
931   
932   
933   
934 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 

Table 10. The problem representations of unseen 44 well-studied VRP variants. Here, RI represents the random identifier, and EAT, LAT, and ST indicate the earliest arrival time, the latest arrival time, and the service time, respectively.

<table><tr><td>Problem</td><td>Oracle</td><td>RI</td><td>Coord.</td><td>Demand</td><td>Prize</td><td>Penalty</td><td>EAT</td><td>LAT</td><td>ST</td><td>Depot</td><td>Pickup</td><td>Delivery</td><td>Sub-routes</td><td>Open Route</td></tr><tr><td>CVRPL</td><td>LKH-3</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>OCVRPB</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>CVRPBL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>CVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>OCVRPBTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>CVRPBLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>OCVRPL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>CVRPBTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>OCVRPBL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>OCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>OCVRPBLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>CVRPBP</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>OCVRPBP</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>CVRPBPL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>OCVRPBPTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>CVRPBLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>CVRPBPTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>OCVRPBPL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>OCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDCVRP</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>MDCVRPTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>MDOCVRP</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDCVRPL</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>MDCVRPB</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>MDOCVRPTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDOCVRPB</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDCVRPLTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>MDOCVRPBTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDOCVRBLTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>MDOCVRPL</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDOCVRPTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>MDOCVRPLB</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDOCVRPBLTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDOCVRBP</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>MDOCVRBPB</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDOCVRPLB</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>MDOCVRPBTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDOCVRBLTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>MDOCVRPTW</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>MDOCVRPBL</td><td>PyVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr></table>
17 
935   
936   
937   
938   
939   
940   
941   
942   
943   
944   
945   
946   
947   
948   
949   
950   
951   
952   
953   
954   
955   
956   
957   
958   
959   
960   
961   
962   
963   
964   
965   
966   
967   
968   
969   
970   
971   
972   
973   
974   
975   
976   
977   
978   
979   
980   
981   
982   
983   
984   
985   
986   
987   
988   
989 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 

Table 11. The problem representations of unseen 55 less-explored VRP variants. Here, RI represents the random identifier, and EAT, LAT, and ST indicate the earliest arrival time, the latest arrival time, and the service time, respectively.

<table><tr><td>Problem</td><td>Oracle</td><td>RI</td><td>Coord.</td><td>Demand</td><td>Prize</td><td>Penalty</td><td>EAT</td><td>LAT</td><td>ST</td><td>Depot</td><td>Pickup</td><td>Delivery</td><td>Sub-routes</td><td>Open Route</td></tr><tr><td>ACVRPTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AOCVRP</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>ACVRPL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>ACVRPB</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>AOCVRPTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>AOCVRPB</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>ACVRBL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>ACVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AOCVRPTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>AOCVRBL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>AOCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>AOCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>AOCVRBP</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>AOCVRBP</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>AOCVRBL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>AOCVRPTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>ACVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>ACVRPTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>AOCVRBL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>AOCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>AMDCVRP</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDCVRPTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRP</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>AMDCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRBL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDCVRBL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRBL</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRBLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRBLTW</td><td>OR-Tools</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPLTW</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRBL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPLTW</td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRBL</td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPLTW</td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRBL</td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRPLTW</td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>AMDOCVRBL</td><td></td><td></td><td></td><td></td><td>?</td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>APDTSP</td><td>OR-Tools</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td></td><td></td></tr><tr><td>APDCVRP</td><td>OR-Tools</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>AOPDCVRP</td><td>OR-Tools</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>PDCVRP</td><td>OR-Tools</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>OPDCVRP</td><td>OR-Tools</td><td></td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>AOP</td><td>OR-Tools</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td></td><td></td><td></td></tr><tr><td>APCTSP</td><td>OR-Tools</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td></td><td></td><td></td></tr><tr><td>ASPECTSP</td><td>N/A</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>?</td><td></td><td></td><td></td><td></td></tr></table>
18 
990   
991   
992   
993   
994   
995   
996   
997   
998   
999   
1000   
1001   
1002   
1003   
1004   
1005   
1006   
1007   
1008   
1009   
1010   
1011   
1012   
1013   
1014   
1015   
1016   
1017   
1018   
1019   
1020   
1021   
1022   
1023   
1024   
1025   
1026   
1027   
1028   
1029   
1030   
1031   
1032   
1033   
1034   
1035   
1036   
1037   
1038   
1039   
1040   
1041   
1042   
1043   
1044 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# E. Model Architecture and Training
# E.1. Encoder
Similar to previous work (Kool et al., 2019; Kwon et al., 2020), we also use the encoder composed of $L$ stacked attention layers, where each attention layer consists of two sub-layers: an attention sub-layer and a Feed-Forward (FF) sub-layer, both of which use Instance Normalization (Ulyanov et al., 2016) and skip-connection (He et al., 2016). 
The encoder $\pmb { \theta } _ { e n c }$ with $L$ stacked attention layers transforms $H ^ { ( 0 ) }$ into advanced node embeddings $H ^ { ( L ) } = \{ { \bf h } _ { i } ^ { ( L ) } \} _ { i = 0 } ^ { n }$ Let com $H ^ { ( \ell - 1 ) } = \{ \mathbf { h } _ { i } ^ { ( \ell - 1 ) } \} _ { i = 0 } ^ { n }$ denote the input to the $\ell$ -th attention layer for $\ell = 1 , \ldots , L$ . The outputs for the $i$ -th node are 
$$
\widetilde {\mathbf {h}} _ {i} ^ {(\ell)} = \mathrm {I N} ^ {(\ell)} \left(\mathbf {h} _ {i} ^ {(\ell - 1)} + \operatorname {M B M} ^ {(\ell)} \left(\mathbf {h} _ {i} ^ {(\ell - 1)}, H ^ {(\ell - 1)}, \boldsymbol {D}, \boldsymbol {D} ^ {\mathrm {T}}, \boldsymbol {R}\right)\right), \tag {11}
$$
$$
\mathbf {h} _ {i} ^ {(\ell)} = \mathrm {I N} ^ {(\ell)} \left(\widetilde {\mathbf {h}} _ {i} ^ {(\ell)} + \mathrm {F F} ^ {(\ell)} \left(\widetilde {\mathbf {h}} _ {i} ^ {(\ell)}\right)\right), \tag {12}
$$
where $\operatorname { I N } ( \cdot )$ denotes instance normalization (Ulyanov et al., 2016), MBM represents the adopted mixed bias module (see Equation (5) for details), and FF(·) in Equation (12) corresponds to a fully connected neural network with ReLU activation. After $L$ attention layers, the final node embeddings $H ^ { ( L ) } = \{ { \bf h } _ { i } ^ { ( L ) } \} _ { i = 0 } ^ { n }$ encapsulate the advanced feature representations of each node. 
Notably, to ensure dimensional consistency across bias channels, we normalize $_ D$ and $D ^ { \mathbf { T } }$ by dividing their maximum value before passing the MBM. 
# E.2. Decoder
In URS, all parameters $\pmb { \theta } _ { d e c } ( \lambda )$ are adaptively generated conditioned on the multi-hot problem representation $\boldsymbol { \lambda }$ (see Appendix D for detailed problem representations). $\pmb { \theta } _ { d e c } ( \lambda )$ includes two MLP networks, which are WEIGHT $( \lambda )$ and BIAS(λ). WEIGHT(λ) is used to generate linear projection matrices $[ W _ { Q } ( \lambda )$ , $W _ { K } ( \lambda )$ , $W _ { V } ( \lambda ) ]$ . BIAS(λ) in Equation (6) is used to generate bias weights for Attention(·)(see Equation (9)) and compatibility (see Equation (10)). 
For WEIGHT(λ), we use a multi-layer MLP hypernetwork conditioned on a 13-dimensional problem representation $\boldsymbol { \lambda }$ to generate problem-specific adaptive parameters. Note that we factor $W _ { Q } ( \pmb { \lambda } ) \in \mathbb { R } ^ { ( 2 d + 1 ) \times d }$ into three vertically concatenated blocks $W _ { \mathrm { f i r s t } } ( \lambda ) \in \mathbb { R } ^ { d \times d }$ , $W _ { \mathrm { l a s t } } ( \pmb { \lambda } ) \in \mathbb { R } ^ { d \times d }$ and $W _ { C } ( \pmb { \lambda } ) \in \mathbb { R } ^ { 1 \times d }$ , applied respectively to the first visited node ${ \bf h } _ { \pi _ { 1 } } ^ { ( L ) }$ , the last visited node $\mathbf { h } _ { \pi _ { t - 1 } } ^ { ( L ) }$ , and a scalar state feature $C _ { t } \in \mathbb { R } ^ { 1 }$ (see Appendix E.3 for more details). Thus WEIGHT $( \lambda )$ outputs the linear projection matrices $\{ W _ { \mathrm { f i r s t } } ( \lambda ) , W _ { \mathrm { l a s t } } ( \lambda ) , W _ { C } ( \lambda ) , W _ { K } ( \lambda ) , W _ { V } ( \lambda ) \}$ . 
Furthermore, we adopt the parameter compression technique of Ha et al. (2016) to control model size. Specifically, we first generate an advanced embedding matrix $H _ { \mathrm { d e c } } ( \pmb { \lambda } ) = [ \pmb { h } _ { \mathrm { d e c } } ^ { i } ( \pmb { \lambda } ) | i = 1 , 2 , \dots , 5 ] \in \mathbb { R } ^ { 5 \times | \pmb { \lambda } | }$ whose five row embeddings each parameterize one target projection matrix. It can be computed via three consecutive linear projections: 
$$
\boldsymbol {H} _ {\mathrm {d e c}} (\boldsymbol {\lambda}) = \left(\left(\boldsymbol {\lambda} W _ {1} + \mathbf {b} _ {1}\right) W _ {2} + \mathbf {b} _ {2}\right) W _ {3} + b _ {3}, \tag {13}
$$
where $W _ { 1 } \in \mathbb { R } ^ { | \lambda | \times d _ { h } }$ , $W _ { 2 } \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ , $W _ { 3 } \in \mathbb { R } ^ { d _ { h } \times ( 5 \times | \lambda | ) }$ , $\mathbf { b } _ { 1 } \in \mathbb { R } ^ { d _ { h } }$ , b2 ∈ Rdh , b3 ∈ R(5×|λ|) are learnable parameters, $d _ { h }$ is 256 in this work, following Lin et al. (2022). Next, we use five different linear projections to map the new representation $h _ { \mathrm { d e c } } ^ { i } ( \lambda )$ of each row to the corresponding decoder parameters. The computation can be expressed as: 
$$
W _ {\text {f i r s t}} (\boldsymbol {\lambda}) = \boldsymbol {h} _ {\mathrm {d e c}} ^ {1} (\boldsymbol {\lambda}) W _ {\mathrm {h y p e r}} ^ {1}, W _ {\text {l a s t}} (\boldsymbol {\lambda}) = \boldsymbol {h} _ {\mathrm {d e c}} ^ {2} (\boldsymbol {\lambda}) W _ {\mathrm {h y p e r}} ^ {2}, W _ {C} (\boldsymbol {\lambda}) = \boldsymbol {h} _ {\mathrm {d e c}} ^ {3} (\boldsymbol {\lambda}) W _ {\mathrm {h y p e r}} ^ {3}, \tag {14}
$$
$$
W _ {K} (\boldsymbol {\lambda}) = \mathbf {h} _ {\mathrm {d e c}} ^ {4} (\boldsymbol {\lambda}) W _ {\text {h y p e r}} ^ {4}, W _ {V} (\boldsymbol {\lambda}) = \mathbf {h} _ {\mathrm {d e c}} ^ {5} (\boldsymbol {\lambda}) W _ {\text {h y p e r}} ^ {5}, \tag {15}
$$
where $W _ { \mathbf { h y p e r } } ^ { 1 } \in \mathbb { R } ^ { | \lambda | \times ( d \times d ) }$ , $W _ { \mathbf { h y p e r } } ^ { 2 } \in \mathbb { R } ^ { | \lambda | \times ( d \times d ) }$ , $W _ { \mathbf { h y p e r } } ^ { 3 } \in \mathbb { R } ^ { | \lambda | \times ( 1 \times d ) }$ , $W _ { \mathbf { h y p e r } } ^ { 4 } \in \mathbb { R } ^ { | \lambda | \times ( d \times d ) }$ , $W _ { \mathbf { h y p e r } } ^ { 5 } \in \mathbb { R } ^ { | \lambda | \times ( d \times d ) }$ are learnable parameters. In this way, we use a simple MLP hypernetwork $\mathrm { W E I G H T } ( \lambda )$ to generate used linear projection matrices $\{ W _ { Q } ( \lambda )$ , $W _ { K } ( \lambda )$ , $W _ { V } ( \lambda ) \}$ conditioned on the multi-hot problem representation $\lambda$ , enabling the adaptive construction of high-quality solutions for different problems. 
19 
1045   
1046   
1047   
1048   
1049   
1050   
1051   
1052   
1053   
1054   
1055   
1056   
1057   
1058   
1059   
1060   
1061   
1062   
1063   
1064   
1065   
1066   
1067   
1068   
1069   
1070   
1071   
1072   
1073   
1074   
1075   
1076   
1077   
1078   
1079   
1080   
1081   
1082   
1083   
1084   
1085   
1086   
1087   
1088   
1089   
1090   
1091   
1092   
1093   
1094   
1095   
1096   
1097   
1098   
1099 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# E.3. Problem State $C _ { t }$
In this section, we detail the problems that require additional state signals. If a problem is not mentioned in this section, it means we will not apply any extra state to it, i.e., $C _ { t } = 0$ . 
For all problems with the capacity constraint, we explicitly impose only remaining load, while diverse additional constraints (e.g., time windows, duration limit, and backhaul) are enforced implicitly as $\mathcal { M } _ { t }$ masks infeasible candidate nodes to guarantee valid solutions, instead of being fully enumerated in the input of the decoder. 
For OP and AOP, we keep track of the remaining maximum length at time $t$ , and $C _ { t }$ represents the remaining routing length that can be moved, and we normalize it to [0, 1] by dividing by the maximum length (i.e., 4.0 in OP100 and 1.0 in AOP100). The setting is the same as AM (Kool et al., 2019). 
For (A)PCTSP and (A)SPCTSP, $C _ { t }$ is the remaining prize to collect, and we do not provide any information about the penalties, as this is irrelevant for the remaining decisions, following Kool et al. (2019). 
For the problem state, we provide an ablation study in Appendix J.2. 
# E.4. Training
Following Kwon et al. (2020), we use $n$ trajectories with distinct starting nodes in training. URS is trained by the REINFORCE (Williams, 1992) algorithm with a shared baseline (Kwon et al., 2020): 
$$
\nabla_ {\theta} \mathcal {L} (\boldsymbol {\theta}) = \mathbb {E} _ {p (\boldsymbol {\pi} | \mathcal {G}, \boldsymbol {\theta})} [ (f (\boldsymbol {\pi} | \mathcal {G}) - \frac {1}{n} \sum_ {i = 1} ^ {n} f \left(\boldsymbol {\pi} ^ {i} | \mathcal {G}\right)) \nabla_ {\boldsymbol {\theta}} \log p _ {\boldsymbol {\theta}} (\boldsymbol {\pi} | \mathcal {G}) ], \tag {16}
$$
$$
p _ {\boldsymbol {\theta}} (\boldsymbol {\pi} \mid \mathcal {G}) = \prod_ {t = 2} ^ {n} p _ {\boldsymbol {\theta}} \left(\pi_ {t} \mid \mathcal {G}, \pi_ {1: t - 1}\right), \tag {17}
$$
where the objective function $f ( \pi | \mathcal { G } )$ represents the total reward (e.g., the negative value of tour length) of instance $\mathcal { G }$ given a specific solution $\pi$ . 
# F. Adaptation Attention Free Module
Following Zhou et al. (2024a), we implement Attention $( \cdot )$ using an adaptation attention free module (AAFM) to enhance geometric pattern recognition for routing problems. Given the input $X$ , AAFM first transforms it into Q, K, and $V$ through corresponding linear projection operations: 
$$
Q = X W ^ {Q}, \quad K = X W ^ {K}, \quad V = X W ^ {V}, \tag {18}
$$
where $W ^ { Q }$ , $W ^ { K }$ , and $W ^ { V }$ are learnable matrices. The AAFM computation is then expressed as: 
$$
\operatorname {A t t e n t i o n} (Q, K, V, A) = \sigma (Q) \odot \frac {\exp (A) (\exp (K) \odot V)}{\exp (A) \exp (K)}, \tag {19}
$$
where $\sigma$ denotes the sigmoid function, $\odot$ represents the element-wise product, and $A = \{ a _ { i j } \}$ denotes the pair-wise adaptation bias. For distance matrices, the corresponding adaptation bias $f \left( \alpha , N , d _ { i j } \right) = - \alpha \cdot \log _ { 2 } N \cdot d _ { i , j }$ . Here, $N$ denotes the total number of nodes (i.e., problem size), and $\alpha$ is generated via a lightweight network $\mathrm { B I A S } ( \lambda )$ in our paper (see Equation (6)). Notably, for the relation matrix, we remove the scale $N$ in the calculation of adaptation bias because the relation is scale-independent. 
Compared to multi-head attention (MHA) (Vaswani et al., 2017), AAFM enables the model to explicitly capture instancespecific knowledge by updating pair-wise adaptation biases while exhibiting lower computational overhead. Further details are provided in the related work section mentioned above. 
20 
1100   
1101   
1102   
1103   
1104   
1105   
1106   
1107   
1108   
1109   
1110   
1111   
1112   
1113   
1114   
1115   
1116   
1117   
1118   
1119   
1120   
1121   
1122   
1123   
1124   
1125   
1126   
1127   
1128   
1129   
1130   
1131   
1132   
1133   
1134   
1135   
1136   
1137   
1138   
1139   
1140   
1141   
1142   
1143   
1144   
1145   
1146   
1147   
1148   
1149   
1150   
1151   
1152   
1153   
1154 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# G. Training and Model Settings
# G.1. Multi-Task Version
The detailed information about the hyperparameter settings can be found in Table 12. The URS training is conducted on a single NVIDIA GeForce RTX 4090 GPU (24GB of memory). The training procedure requires only about 95 hours $\approx 4$ days) in total (500 epochs at approximately 12 minutes each) to obtain a unified parameter set capable of addressing a broad class of routing problems. 

Table 12. The hyperparameter settings of URS.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Clipping parameterζ</td><td>50</td></tr><tr><td>Initial learning rate</td><td>10-4</td></tr><tr><td>Epochs of learning rate decay</td><td>[451]</td></tr><tr><td>Factor of learning rate decay</td><td>0.1</td></tr><tr><td>Weight decay</td><td>10-6</td></tr><tr><td>The number of encoder layersL</td><td>12</td></tr><tr><td>Embedding dimensiond</td><td>128</td></tr><tr><td>Feed forward dimension</td><td>512</td></tr><tr><td>Dimension dh in Equation (13)</td><td>256</td></tr><tr><td>Training scale</td><td>100</td></tr><tr><td>Batches of each epoch</td><td>2,000</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Training Epochs</td><td>500</td></tr></table>
# G.2. Single-Task Version
URS-STL is trained on data from only a single problem type (e.g., ”URS-STL (CVRP)” is trained only on CVRP). The URS-STL variant uses exactly the same model architecture, hyperparameters (e.g., embedding dimension, number of layers, optimizer), and training setup (e.g., initial learning rate, batch size) as the multi-task URS described in Section 4.1 and Appendix G.1. The only modification we made is to the training duration. We observe that the single-task models converge significantly faster than the multi-task model. Therefore, we train URS-STL models for 300 epochs, with a learning rate decay applied at the 251st epoch. This contrasts with the multi-task URS, which is trained for 500 epochs with decay applied at the 451st epoch. This adjustment is made solely to reflect the faster convergence of the single-task training. All other hyperparameters and model architecture remain identical. 
# H. Generalization to Unseen VRPs
To comprehensively evaluate URS’s zero-shot generalization, we test its performance on a diverse set of 99 unseen problems. This set is designed to cover both established and novel challenges, including 44 well-studied VRP variants and 55 lessexplored VRP variants. The detailed results are presented in Table 13 and Table 14. On widely-studied constraints (i.e., C, O, L, TW, MD, B, BP), URS continues to deliver high-quality solutions. Notably, URS merely selects the next node from a given feasible candidates without any additional domain knowledge, making its performance especially meaningful. Benefiting from our UDR, URS can directly address more complex variants while maintaining solution quality, which compounds asymmetry and PD relations. Without any fine-tuning, URS still produces high-quality solutions for them, thereby strengthening its broad problem generality and deployment efficiency. 
21 
1155   
1156   
1157   
1158   
1159   
1160   
1161   
1162   
1163   
1164   
1165   
1166   
1167   
1168   
1169   
1170   
1171   
1172   
1173   
1174   
1175   
1176   
1177   
1178   
1179   
1180   
1181   
1182   
1183   
1184   
1185   
1186   
1187   
1188   
1189   
1190   
1191   
1192   
1193   
1194   
1195   
1196   
1197   
1198   
1199   
1200   
1201   
1202   
1203   
1204   
1205   
1206   
1207   
1208   
1209 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 

Table 13. Zero-shot generalization performance across 44 unseen well-studied VRP variants $N = 1 0 0$

<table><tr><td rowspan="2">Method</td><td colspan="2">CVRPBP</td><td colspan="2">OCVRPBP</td><td colspan="2">CVRPBL</td><td colspan="2">OCVRPBTW</td><td>CVRPBLT</td><td></td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>MVMoE</td><td>13.95%</td><td>14s</td><td>18.27%</td><td>14s</td><td>13.38%</td><td>14s</td><td>9.42%</td><td>15s</td><td>16.66%</td><td>16s</td></tr><tr><td>MTPOMO</td><td>13.71%</td><td>7s</td><td>17.90%</td><td>8s</td><td>13.44%</td><td>8s</td><td>9.94%</td><td>9s</td><td>17.15%</td><td>10s</td></tr><tr><td>ReLD-MTL</td><td>13.57%</td><td>9s</td><td>15.81%</td><td>9s</td><td>12.96%</td><td>10s</td><td>8.43%</td><td>11s</td><td>15.95%</td><td>12s</td></tr><tr><td>URS</td><td>12.95%</td><td>7s</td><td>16.96%</td><td>7s</td><td>12.65%</td><td>8s</td><td>10.05%</td><td>8s</td><td>18.33%</td><td>10s</td></tr><tr><td rowspan="2"></td><td colspan="2">CVRPBPWT</td><td colspan="2">OCVRPBL</td><td colspan="2">OCVRPBLTW</td><td colspan="2">CVRPL</td><td colspan="2">CVRPBL</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>16m</td><td>0.00%</td><td>3.5h</td></tr><tr><td>MVMoE</td><td>16.86%</td><td>15s</td><td>17.75%</td><td>14s</td><td>8.87%</td><td>16s</td><td>0.26%</td><td>12s</td><td>1.35%</td><td>12s</td></tr><tr><td>MTPOMO</td><td>17.22%</td><td>9s</td><td>17.31%</td><td>8s</td><td>9.47%</td><td>10s</td><td>0.48%</td><td>6s</td><td>1.79%</td><td>6s</td></tr><tr><td>ReLD-MTL</td><td>16.09%</td><td>11s</td><td>15.21%</td><td>10s</td><td>8.06%</td><td>11s</td><td>0.02%</td><td>8s</td><td>1.01%</td><td>7s</td></tr><tr><td>URS</td><td>18.15%</td><td>9s</td><td>16.37%</td><td>8s</td><td>9.51%</td><td>9s</td><td>0.43%</td><td>7s</td><td>1.65%</td><td>6s</td></tr><tr><td rowspan="2"></td><td colspan="2">CVRPBTW</td><td colspan="2">OCVRPBL</td><td colspan="2">OCVRPLTW</td><td colspan="2">OCVRPBLTW</td><td colspan="2">MDCVRPBP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>3.5h</td><td>0.00%</td><td>3.5h</td><td>0.00%</td><td>3.5h</td><td>0.00%</td><td>3.5h</td><td>0.00%</td><td>6.6m</td></tr><tr><td>MVMoE</td><td>7.08%</td><td>15s</td><td>7.12%</td><td>12s</td><td>3.90%</td><td>15s</td><td>10.01%</td><td>15s</td><td>65.98%</td><td>33s</td></tr><tr><td>MTPOMO</td><td>7.41%</td><td>7s</td><td>7.34%</td><td>7s</td><td>4.37%</td><td>8s</td><td>10.50%</td><td>7s</td><td>45.77%</td><td>24s</td></tr><tr><td>ReLD-MTL</td><td>6.74%</td><td>8s</td><td>5.41%</td><td>7s</td><td>3.16%</td><td>9s</td><td>9.22%</td><td>8s</td><td>40.70%</td><td>27s</td></tr><tr><td>URS</td><td>8.94%</td><td>8s</td><td>9.47%</td><td>7s</td><td>5.12%</td><td>9s</td><td>13.72%</td><td>8s</td><td>35.44%</td><td>24s</td></tr><tr><td rowspan="2"></td><td colspan="2">MDOCVRPBP</td><td colspan="2">MDCVRPBL</td><td colspan="2">MDOCVRPBTW</td><td colspan="2">MDCVRPBTW</td><td colspan="2">MDCVRPBTW</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>MVMoE</td><td>56.60%</td><td>33s</td><td>65.81%</td><td>36s</td><td>63.77%</td><td>38s</td><td>68.49%</td><td>42s</td><td>69.00%</td><td>39s</td></tr><tr><td>MTPOMO</td><td>42.94%</td><td>24s</td><td>45.98%</td><td>26s</td><td>47.09%</td><td>29s</td><td>50.45%</td><td>32s</td><td>50.74%</td><td>30s</td></tr><tr><td>ReLD-MTL</td><td>37.74%</td><td>27s</td><td>39.57%</td><td>29s</td><td>45.16%</td><td>34s</td><td>45.23%</td><td>37s</td><td>45.62%</td><td>35s</td></tr><tr><td>URS</td><td>24.44%</td><td>22s</td><td>35.07%</td><td>26s</td><td>26.31%</td><td>26s</td><td>36.48%</td><td>32s</td><td>36.69%</td><td>30s</td></tr><tr><td rowspan="2"></td><td colspan="2">MDOCVRPBL</td><td colspan="2">MDOCVRPBLTW</td><td colspan="2">MDOCVRPPTW</td><td colspan="2">MDOCVRPPTW</td><td colspan="2">MDOCVRPPTW</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>MVMoE</td><td>56.26%</td><td>34s</td><td>62.47%</td><td>40s</td><td>33.06%</td><td>31s</td><td>53.64%</td><td>41s</td><td>29.08%</td><td>31s</td></tr><tr><td>MTPOMO</td><td>41.98%</td><td>27s</td><td>46.26%</td><td>32s</td><td>23.38%</td><td>23s</td><td>36.10%</td><td>31s</td><td>27.46%</td><td>23s</td></tr><tr><td>ReLD-MTL</td><td>37.45%</td><td>29s</td><td>44.74%</td><td>36s</td><td>18.39%</td><td>25s</td><td>30.90%</td><td>34s</td><td>22.32%</td><td>25s</td></tr><tr><td>URS</td><td>24.22%</td><td>24s</td><td>26.19%</td><td>28s</td><td>15.05%</td><td>23s</td><td>37.26%</td><td>32s</td><td>22.01%</td><td>23s</td></tr><tr><td rowspan="2"></td><td colspan="2">MDCVRPL</td><td colspan="2">MDCVRPB</td><td colspan="2">MDOCVRPTW</td><td colspan="2">MDOCVRPB</td><td colspan="2">MDCVRPBL</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>MVMoE</td><td>33.42%</td><td>34s</td><td>21.70%</td><td>26s</td><td>51.57%</td><td>40s</td><td>28.21%</td><td>26s</td><td>20.35%</td><td>29s</td></tr><tr><td>MTPOMO</td><td>24.32%</td><td>25s</td><td>12.89%</td><td>19s</td><td>35.71%</td><td>29s</td><td>25.06%</td><td>20s</td><td>11.10%</td><td>20s</td></tr><tr><td>ReLD-MTL</td><td>18.48%</td><td>28s</td><td>9.38%</td><td>21s</td><td>33.55%</td><td>33s</td><td>18.50%</td><td>21s</td><td>8.26%</td><td>23s</td></tr><tr><td>URS</td><td>15.32%</td><td>26s</td><td>13.14%</td><td>19s</td><td>26.05%</td><td>25s</td><td>15.17%</td><td>18s</td><td>11.19%</td><td>21s</td></tr><tr><td rowspan="2"></td><td colspan="2">MDOCVRPLTW</td><td colspan="2">MDOCVRPBTW</td><td colspan="2">MDOCVRPBLTW</td><td colspan="2">MDOCVRPL</td><td colspan="2">MDOCVRPBTW</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>MVMoE</td><td>55.55%</td><td>44s</td><td>62.47%</td><td>35s</td><td>63.18%</td><td>37s</td><td>29.33%</td><td>33s</td><td>65.90%</td><td>35s</td></tr><tr><td>MTPOMO</td><td>37.34%</td><td>32s</td><td>45.55%</td><td>25s</td><td>45.07%</td><td>27s</td><td>27.21%</td><td>25s</td><td>45.94%</td><td>25s</td></tr><tr><td>ReLD-MTL</td><td>31.56%</td><td>36s</td><td>43.36%</td><td>28s</td><td>40.57%</td><td>30s</td><td>22.05%</td><td>27s</td><td>41.28%</td><td>28s</td></tr><tr><td>URS</td><td>38.04%</td><td>35s</td><td>30.32%</td><td>22s</td><td>37.70%</td><td>27s</td><td>22.15%</td><td>25s</td><td>37.78%</td><td>25s</td></tr><tr><td rowspan="2"></td><td colspan="2">MDOCVRPLB</td><td colspan="2">MDOCVRPLTW</td><td colspan="2">MDOCVRPBLTW</td><td colspan="2">SPCTSP</td><td></td><td></td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td></td><td></td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>1.6h</td><td></td><td></td></tr><tr><td>MVMoE</td><td>28.70%</td><td>28s</td><td>51.09%</td><td>43s</td><td>62.10%</td><td>37s</td><td></td><td></td><td></td><td></td></tr><tr><td>MTPOMO</td><td>25.01%</td><td>21s</td><td>35.39%</td><td>31s</td><td>45.48%</td><td>27s</td><td></td><td></td><td></td><td></td></tr><tr><td>ReLD-MTL</td><td>18.50%</td><td>22s</td><td>33.26%</td><td>35s</td><td>43.27%</td><td>30s</td><td></td><td></td><td></td><td></td></tr><tr><td>URS</td><td>14.90%</td><td>20s</td><td>25.59%</td><td>27s</td><td>30.41%</td><td>24s</td><td></td><td></td><td></td><td></td></tr></table>
1210   
1211   
1212   
1213   
1214   
1215   
1216   
1217   
1218   
1219   
1220   
1221   
1222   
1223   
1224   
1225   
1226   
1227   
1228   
1229   
1230   
1231   
1232   
1233   
1234   
1235   
1236   
1237   
1238   
1239   
1240   
1241   
1242   
1243   
1244   
1245   
1246   
1247   
1248   
1249   
1250   
1251   
1252   
1253   
1254   
1255   
1256   
1257   
1258   
1259   
1260   
1261   
1262   
1263   
1264 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 

Table 14. Zero-shot generalization performance across 55 unseen less-explored VRP variants $N = 1 0 0$ ).

<table><tr><td rowspan="2">Method</td><td colspan="2">ACVRPL</td><td colspan="2">ACVRPBL</td><td colspan="2">ACVRPLTW</td><td colspan="2">ACVRPBLTW</td><td colspan="2">AOCVRPL</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>-2.24%</td><td>1.9m</td><td>6.18%</td><td>2.4m</td><td>12.22%</td><td>2.7m</td><td>10.18%</td><td>2.4m</td><td>37.36%</td><td>3.1m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">AOCVRPBL</td><td colspan="2">AOCVRPLTW</td><td colspan="2">AOCVRPBLTW</td><td colspan="2">ACVRPBL</td><td colspan="2">ACVRPBLTW</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>17.23%</td><td>2.7m</td><td>16.35%</td><td>2.8m</td><td>14.32%</td><td>2.5m</td><td>24.03%</td><td>2.6m</td><td>13.82%</td><td>2.8m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">AOCVRPBL</td><td colspan="2">AOCVRPBLTW</td><td colspan="2">ACVRPBL</td><td colspan="2">ACVRPBL</td><td colspan="2">ACVRPBL</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>32.50%</td><td>2.7m</td><td>16.43%</td><td>2.7m</td><td>7.10%</td><td>2.1m</td><td>11.28%</td><td>2.5m</td><td>37.63%</td><td>2.9m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">AOCVRPTW</td><td colspan="2">AOCVRPB</td><td colspan="2">AOCVRPBTW</td><td colspan="2">ACVRPBTW</td><td colspan="2">ACVRPBP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>17.47%</td><td>2.6m</td><td>18.05%</td><td>2.5m</td><td>14.45%</td><td>2.3m</td><td>9.95%</td><td>2.2m</td><td>24.20%</td><td>2.4m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">AOCVRPBP</td><td colspan="2">AOCVRPBTW</td><td colspan="2">ACVRPBTW</td><td colspan="2">APDTSP</td><td colspan="2">AMDCVRPL</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>32.58%</td><td>2.5m</td><td>17.45%</td><td>2.5m</td><td>14.52%</td><td>2.6m</td><td>6.21%</td><td>1m</td><td>7.15%</td><td>6.1m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">AMDCVRPBL</td><td colspan="2">AMDCVRPLTW</td><td colspan="2">AMDCVRPBLTW</td><td colspan="2">AMDOCVRPL</td><td colspan="2">AMDOCVRPL</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>17.51%</td><td>6.5m</td><td>19.98%</td><td>8.8m</td><td>16.73%</td><td>7.5m</td><td>47.80%</td><td>8.6m</td><td>27.23%</td><td>7.1m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">AMDOCVRPLTW</td><td colspan="2">AMDOCVRPBLTW</td><td colspan="2">AMDOCVRPBL</td><td colspan="2">AMDOCVRPBLTW</td><td colspan="2">AMDOCVRPBL</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>24.90%</td><td>7.9m</td><td>22.84%</td><td>6.7m</td><td>35.00%</td><td>7.2m</td><td>20.50%</td><td>8.3m</td><td>38.78%</td><td>7.3m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">AMDOCVRPBLTW</td><td colspan="2">AMDCVRP</td><td colspan="2">AMDCVRPTW</td><td colspan="2">AMDOCVRP</td><td colspan="2">AMDOCVRP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>24.05%</td><td>7.5m</td><td>8.11%</td><td>5.3m</td><td>20.96%</td><td>8.1m</td><td>48.02%</td><td>7.8m</td><td>17.35%</td><td>5.8m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">AMDOCVRPTW</td><td colspan="2">AMDOCVRPB</td><td colspan="2">AMDOCVRPBTW</td><td colspan="2">AMDOCVRPBTW</td><td colspan="2">AMDOCVRPBP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>26.02%</td><td>7.3m</td><td>27.78%</td><td>6.5m</td><td>23.67%</td><td>6.2m</td><td>17.52%</td><td>6.8m</td><td>36.18%</td><td>6.5m</td></tr><tr><td rowspan="2">Method</td><td colspan="2">AMDOCVRPBP</td><td colspan="2">AMDOCVRPBLPW</td><td colspan="2">AMDOCVRPBLPW</td><td colspan="2">PDCVRP</td><td colspan="2">OPDCVRP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td></tr><tr><td>URS</td><td>39.31%</td><td>6.7m</td><td>24.99%</td><td>6.9m</td><td>20.98%</td><td>7.6m</td><td>-1.47%</td><td>4.3s</td><td>4.93%</td><td>4.3s</td></tr><tr><td rowspan="2">Method</td><td colspan="2">APDCVRP</td><td colspan="2">AOPDCVRP</td><td colspan="2">AOP</td><td colspan="2">APCTSP</td><td colspan="2">ASPECTSP</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Value</td><td>Time</td></tr><tr><td>Oracle</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>0.00%</td><td>6.6m</td><td>-</td><td>-</td></tr><tr><td>URS</td><td>7.03%</td><td>1.2m</td><td>11.13%</td><td>1.2m</td><td>-5.47%</td><td>1.2m</td><td>43.79%</td><td>1.2m</td><td>2.22</td><td>1.2m</td></tr></table>
1265   
1266   
1267   
1268   
1269   
1270   
1271   
1272   
1273   
1274   
1275   
1276   
1277   
1278   
1279   
1280   
1281   
1282   
1283   
1284   
1285   
1286   
1287   
1288   
1289   
1290   
1291   
1292   
1293   
1294   
1295   
1296   
1297   
1298   
1299   
1300   
1301   
1302   
1303   
1304   
1305   
1306   
1307   
1308   
1309   
1310   
1311   
1312   
1313   
1314   
1315   
1316   
1317   
1318   
1319 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# I. More Experiments
# I.1. Training Problem Selection
To empirically validate our selection strategy of training problems, we train two additional variants and compare them with our original URS: URS-Diff (Different Selection) and URS-Expand (Broader Problems). The first variant, URS-Diff, uses a different problem set where OCVRPTW is replaced by attribute-independent CVRPL. This modification means that the Open Route and Time Window attributes each appear in only one training problem. The second variant, URS-Expand, is trained on a broader set of 22 problems, incorporating all variants evaluated in MTPOMO (Liu et al., 2024) and MVMoE (Zhou et al., 2024b). 
As shown in Table 15, URS-Diff performs significantly worse on unseen problems involving Open Route and Time Window constraints. Meanwhile, URS-Expand achieves a slightly better performance than URS due to seeing more problem variants during training. However, the original URS remains highly competitive. Remarkably, despite being trained on only half the number of problem types (11 vs. 22), URS achieves the best solution on 12 out of the 22 evaluated problems. 
These results demonstrate that the choice of our selected 11 problems offers a balanced trade-off, providing sufficient attribute coverage for strong generalization while maintaining training efficiency. Expanding the training set yields only marginal gains, while a poorly selected set leads to poor cross-problem generalization. In future work, we plan to explore more effective training problem selection strategies to further enhance model performance and training efficiency. 

Table 15. Comparison under different training problem settings. The symbol * indicates the original problem selection, and $\Delta$ represents a different problem selection of equal number. All test problems are encompassed within URS-Expand.

<table><tr><td>Method</td><td>ATSP (* Δ)</td><td>TSP(* Δ)</td><td>OP(* Δ)</td><td>PCTSP(* Δ)</td><td>PDTSP(* Δ)</td><td>ACVRP(*Δ)</td></tr><tr><td>URS-Diff</td><td>4.52%</td><td>1.05%</td><td>0.82%</td><td>1.31%</td><td>6.09%</td><td>5.10%</td></tr><tr><td>URS-Expand</td><td>4.77%</td><td>0.93%</td><td>0.99%</td><td>1.41%</td><td>5.85%</td><td>5.48%</td></tr><tr><td>URS</td><td>2.26%</td><td>0.57%</td><td>0.45%</td><td>1.06%</td><td>4.98%</td><td>3.06%</td></tr><tr><td></td><td>CVRP(* Δ)</td><td>CVRPTW(* Δ)</td><td>CVRPB(* Δ)</td><td>OCVRP(* Δ)</td><td>OCVRPTW(*)</td><td>CVRPL(Δ)</td></tr><tr><td>URS-Diff</td><td>1.97%</td><td>6.26%</td><td>1.80%</td><td>3.75%</td><td>15.45%</td><td>0.58%</td></tr><tr><td>URS-Expand</td><td>2.04%</td><td>5.62%</td><td>1.77%</td><td>3.45%</td><td>4.34%</td><td>0.65%</td></tr><tr><td>URS</td><td>1.81%</td><td>6.13%</td><td>1.46%</td><td>3.24%</td><td>5.07%</td><td>0.43%</td></tr><tr><td></td><td>OCVRPB</td><td>CVRPBL</td><td>CVRPLTW</td><td>OCVRPBTW</td><td>CVRPBLTW</td><td>OCVRPL</td></tr><tr><td>URS-Diff</td><td>7.55%</td><td>2.02%</td><td>2.76%</td><td>27.48%</td><td>10.25%</td><td>3.73%</td></tr><tr><td>URS-Expand</td><td>5.39%</td><td>1.88%</td><td>2.03%</td><td>9.13%</td><td>6.29%</td><td>3.41%</td></tr><tr><td>URS</td><td>9.35%</td><td>1.65%</td><td>2.67%</td><td>13.77%</td><td>9.18%</td><td>3.22%</td></tr><tr><td></td><td>CVRPBTW</td><td>OCVRPBL</td><td>OCVRPLTW</td><td>OCVRPBLTW</td><td>Avg.gap</td><td>Best Sol.</td></tr><tr><td>URS-Diff</td><td>10.00%</td><td>7.56%</td><td>15.61%</td><td>27.80%</td><td>7.43%</td><td>0/22</td></tr><tr><td>URS-Expand</td><td>6.07%</td><td>5.47%</td><td>4.38%</td><td>9.23%</td><td>4.12%</td><td>10/22</td></tr><tr><td>URS</td><td>8.94%</td><td>9.47%</td><td>5.12%</td><td>13.72%</td><td>4.89%</td><td>12/22</td></tr></table>
1320   
1321   
1322   
1323   
1324   
1325   
1326   
1327   
1328   
1329   
1330   
1331   
1332   
1333   
1334   
1335   
1336   
1337   
1338   
1339   
1340   
1341   
1342   
1343   
1344   
1345   
1346   
1347   
1348   
1349   
1350   
1351   
1352   
1353   
1354   
1355   
1356   
1357   
1358   
1359   
1360   
1361   
1362   
1363   
1364   
1365   
1366   
1367   
1368   
1369   
1370   
1371   
1372   
1373   
1374 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# I.2. Detailed Results on CVRPLib Set-X Dataset
We further evaluate the generalization ability of URS on benchmark instances from CVRPLIB Set-X (Uchoa et al., 2017) with scale $\in [ 5 0 0 , 1 0 0 0 ]$ . Our test results on CVRPLib Set-X datasets are presented in Table 16. All models are trained on instances of size $N = 1 0 0$ . In the Set-X dataset, some results for comparative models are sourced from MVMoE (Zhou et al., 2024b) and RouteFinder (Berto et al., 2025). As shown in Table 16, URS maintains its position as the best overall performance across instances of varying scales. Notably, RL-based URS surpasses the SL-based representative specialist model LEHD $8 . 6 7 8 \%$ vs. $1 2 . 8 3 6 \%$ ) and many representative multi-task neural solvers (e.g., MVMoE and RouteFinder), underscoring its practical applicability in real-world scenarios. 

Table 16. Results on large-scale CVRPLIB instances (Set-X) (Uchoa et al., 2017) $( 5 0 0 \leq N \leq 1 0 0 0 )$ . All models are trained on instances of size $N = 1 0 0$ .

<table><tr><td rowspan="2">Set-X
Instance</td><td rowspan="2">Opt.</td><td colspan="2">POMO</td><td colspan="2">LEHD</td><td colspan="2">MTPOMO</td><td colspan="2">MVMoE/4E</td><td colspan="2">MVMoE/4E-L</td><td colspan="2">RF-MVMoE</td><td colspan="2">RF-TE</td><td colspan="2">URS</td></tr><tr><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td></tr><tr><td>X-n502-k39</td><td>69226</td><td>75617</td><td>9.232%</td><td>71438</td><td>3.195%</td><td>77284</td><td>11.640%</td><td>73533</td><td>6.222%</td><td>74429</td><td>7.516%</td><td>76338</td><td>10.274%</td><td>71791</td><td>3.705%</td><td>71281</td><td>2.969%</td></tr><tr><td>X-n513-k21</td><td>24201</td><td>30518</td><td>26.102%</td><td>25624</td><td>5.880%</td><td>28510</td><td>17.805%</td><td>32102</td><td>32.647%</td><td>31231</td><td>29.048%</td><td>32639</td><td>34.866%</td><td>28465</td><td>17.619%</td><td>26166</td><td>8.119%</td></tr><tr><td>X-n524-k153</td><td>154593</td><td>201877</td><td>30.586%</td><td>280556</td><td>81.480%</td><td>192249</td><td>24.358%</td><td>186540</td><td>20.665%</td><td>182392</td><td>17.982%</td><td>170999</td><td>10.612%</td><td>174381</td><td>12.800%</td><td>175250</td><td>13.362%</td></tr><tr><td>X-n536-k96</td><td>94846</td><td>106073</td><td>11.837%</td><td>103785</td><td>9.425%</td><td>106514</td><td>12.302%</td><td>109581</td><td>15.536%</td><td>108543</td><td>14.441%</td><td>105847</td><td>11.599%</td><td>103272</td><td>8.884%</td><td>102969</td><td>8.564%</td></tr><tr><td>X-n548-k50</td><td>86700</td><td>103093</td><td>18.908%</td><td>90644</td><td>4.549%</td><td>94562</td><td>9.068%</td><td>95894</td><td>10.604%</td><td>95917</td><td>10.631%</td><td>104289</td><td>20.287%</td><td>100956</td><td>16.443%</td><td>89768</td><td>3.539%</td></tr><tr><td>X-n561-k42</td><td>42717</td><td>49370</td><td>15.575%</td><td>44728</td><td>4.708%</td><td>47846</td><td>12.007%</td><td>56008</td><td>31.114%</td><td>51810</td><td>21.287%</td><td>53383</td><td>24.969%</td><td>49454</td><td>15.771%</td><td>45964</td><td>7.601%</td></tr><tr><td>X-n573-k30</td><td>50673</td><td>83545</td><td>64.871%</td><td>53482</td><td>5.543%</td><td>60913</td><td>20.208%</td><td>59473</td><td>17.366%</td><td>57042</td><td>12.569%</td><td>61524</td><td>21.414%</td><td>55952</td><td>10.418%</td><td>54361</td><td>7.278%</td></tr><tr><td>X-n586-k159</td><td>190316</td><td>229887</td><td>20.792%</td><td>232867</td><td>22.358%</td><td>208893</td><td>9.761%</td><td>215668</td><td>13.321%</td><td>214577</td><td>12.748%</td><td>212151</td><td>11.473%</td><td>205575</td><td>8.018%</td><td>202645</td><td>6.478%</td></tr><tr><td>X-n599-k92</td><td>108451</td><td>150572</td><td>38.839%</td><td>115377</td><td>6.386%</td><td>120333</td><td>10.956%</td><td>128949</td><td>18.901%</td><td>125279</td><td>15.517%</td><td>126578</td><td>16.714%</td><td>116560</td><td>7.477%</td><td>114423</td><td>5.507%</td></tr><tr><td>X-n613-k62</td><td>59535</td><td>68451</td><td>14.976%</td><td>62484</td><td>4.953%</td><td>67984</td><td>14.192%</td><td>82586</td><td>38.718%</td><td>74945</td><td>25.884%</td><td>73456</td><td>23.383%</td><td>67267</td><td>12.987%</td><td>65901</td><td>10.693%</td></tr><tr><td>X-n627-k43</td><td>62164</td><td>84434</td><td>35.825%</td><td>67568</td><td>8.693%</td><td>73060</td><td>17.528%</td><td>70987</td><td>14.193%</td><td>70905</td><td>14.061%</td><td>70414</td><td>13.271%</td><td>67572</td><td>8.700%</td><td>66499</td><td>6.973%</td></tr><tr><td>X-n641-k35</td><td>63682</td><td>75573</td><td>18.672%</td><td>68249</td><td>7.172%</td><td>72643</td><td>14.071%</td><td>75329</td><td>18.289%</td><td>72655</td><td>14.090%</td><td>71975</td><td>13.023%</td><td>70831</td><td>11.226%</td><td>67005</td><td>5.218%</td></tr><tr><td>X-n655-k131</td><td>106780</td><td>127211</td><td>19.134%</td><td>117532</td><td>10.069%</td><td>116988</td><td>9.560%</td><td>117678</td><td>10.206%</td><td>118475</td><td>10.952%</td><td>119057</td><td>11.497%</td><td>112202</td><td>5.078%</td><td>110237</td><td>3.237%</td></tr><tr><td>X-n670-k130</td><td>146332</td><td>208079</td><td>42.197%</td><td>220927</td><td>50.977%</td><td>190118</td><td>29.922%</td><td>197695</td><td>35.100%</td><td>183447</td><td>25.364%</td><td>168226</td><td>14.962%</td><td>168999</td><td>15.490%</td><td>184010</td><td>25.748%</td></tr><tr><td>X-n685-k75</td><td>68205</td><td>79482</td><td>16.534%</td><td>72946</td><td>6.951%</td><td>80892</td><td>18.601%</td><td>97388</td><td>42.787%</td><td>89441</td><td>31.136%</td><td>82269</td><td>20.620%</td><td>77847</td><td>14.137%</td><td>75942</td><td>11.344%</td></tr><tr><td>X-n701-k44</td><td>81923</td><td>97843</td><td>19.433%</td><td>86327</td><td>5.376%</td><td>92075</td><td>12.392%</td><td>98469</td><td>20.197%</td><td>94924</td><td>15.870%</td><td>90189</td><td>10.090%</td><td>89932</td><td>9.776%</td><td>86038</td><td>5.023%</td></tr><tr><td>X-n716-k35</td><td>43373</td><td>51381</td><td>18.463%</td><td>46502</td><td>7.214%</td><td>52709</td><td>21.525%</td><td>56773</td><td>30.895%</td><td>52305</td><td>20.593%</td><td>52250</td><td>20.467%</td><td>49669</td><td>14.516%</td><td>46496</td><td>7.200%</td></tr><tr><td>X-n733-k159</td><td>136187</td><td>159098</td><td>16.823%</td><td>149115</td><td>9.493%</td><td>161961</td><td>18.925%</td><td>178322</td><td>30.939%</td><td>167477</td><td>22.976%</td><td>156387</td><td>14.833%</td><td>148463</td><td>9.014%</td><td>147743</td><td>8.485%</td></tr><tr><td>X-n749-k98</td><td>77269</td><td>87786</td><td>13.611%</td><td>83439</td><td>7.985%</td><td>90582</td><td>17.229%</td><td>100438</td><td>29.985%</td><td>94497</td><td>22.296%</td><td>92147</td><td>19.255%</td><td>85171</td><td>10.227%</td><td>83759</td><td>8.399%</td></tr><tr><td>X-n766-k71</td><td>114417</td><td>135464</td><td>18.395%</td><td>131487</td><td>14.919%</td><td>144041</td><td>25.891%</td><td>152352</td><td>33.155%</td><td>136255</td><td>19.086%</td><td>130505</td><td>14.061%</td><td>129935</td><td>13.563%</td><td>139371</td><td>21.810%</td></tr><tr><td>X-n783-k48</td><td>72386</td><td>90289</td><td>24.733%</td><td>76766</td><td>6.051%</td><td>83169</td><td>14.897%</td><td>100383</td><td>38.677%</td><td>92960</td><td>28.423%</td><td>96336</td><td>33.087%</td><td>83185</td><td>14.919%</td><td>77437</td><td>6.978%</td></tr><tr><td>X-n801-k40</td><td>73305</td><td>124278</td><td>69.536%</td><td>77546</td><td>5.785%</td><td>85077</td><td>16.059%</td><td>91560</td><td>24.903%</td><td>87662</td><td>19.585%</td><td>87118</td><td>18.843%</td><td>86164</td><td>17.542%</td><td>77369</td><td>5.544%</td></tr><tr><td>X-n819-k171</td><td>158121</td><td>193451</td><td>22.344%</td><td>178558</td><td>12.925%</td><td>177157</td><td>12.039%</td><td>183599</td><td>16.113%</td><td>185832</td><td>17.525%</td><td>179596</td><td>13.581%</td><td>174441</td><td>10.321%</td><td>171024</td><td>8.160%</td></tr><tr><td>X-n837-k142</td><td>193737</td><td>237884</td><td>22.787%</td><td>207709</td><td>7.212%</td><td>214207</td><td>10.566%</td><td>229526</td><td>18.473%</td><td>221286</td><td>14.220%</td><td>230362</td><td>18.904%</td><td>208528</td><td>7.635%</td><td>203457</td><td>5.017%</td></tr><tr><td>X-n856-k95</td><td>88965</td><td>152528</td><td>71.447%</td><td>92936</td><td>4.464%</td><td>101774</td><td>14.398%</td><td>99129</td><td>11.425%</td><td>106816</td><td>20.065%</td><td>105801</td><td>18.924%</td><td>98291</td><td>10.483%</td><td>94547</td><td>6.274%</td></tr><tr><td>X-n876-k59</td><td>99299</td><td>119764</td><td>20.609%</td><td>104183</td><td>4.918%</td><td>116617</td><td>17.440%</td><td>119619</td><td>20.463%</td><td>114333</td><td>15.140%</td><td>114016</td><td>14.821%</td><td>107416</td><td>8.174%</td><td>105417</td><td>6.161%</td></tr><tr><td>X-n895-k37</td><td>53860</td><td>70245</td><td>30.421%</td><td>58028</td><td>7.739%</td><td>65587</td><td>21.773%</td><td>79018</td><td>46.710%</td><td>64310</td><td>19.402%</td><td>69099</td><td>28.294%</td><td>64871</td><td>20.444%</td><td>58137</td><td>7.941%</td></tr><tr><td>X-n916-k207</td><td>329179</td><td>399372</td><td>21.324%</td><td>385208</td><td>17.021%</td><td>361719</td><td>9.885%</td><td>383681</td><td>16.557%</td><td>374016</td><td>13.621%</td><td>373600</td><td>13.494%</td><td>352998</td><td>7.236%</td><td>346556</td><td>5.279%</td></tr><tr><td>X-n936-k151</td><td>132715</td><td>237625</td><td>79.049%</td><td>196547</td><td>48.097%</td><td>186262</td><td>40.347%</td><td>220926</td><td>66.466%</td><td>190407</td><td>43.471%</td><td>161343</td><td>21.571%</td><td>163162</td><td>22.942%</td><td>172675</td><td>30.110%</td></tr><tr><td>X-n957-k87</td><td>85465</td><td>130850</td><td>53.104%</td><td>90295</td><td>5.651%</td><td>98198</td><td>14.898%</td><td>113882</td><td>33.250%</td><td>105629</td><td>23.593%</td><td>123633</td><td>44.659%</td><td>102689</td><td>20.153%</td><td>90485</td><td>5.874%</td></tr><tr><td>X-n979-k58</td><td>118976</td><td>147687</td><td>24.132%</td><td>127972</td><td>7.561%</td><td>138092</td><td>16.067%</td><td>146347</td><td>23.005%</td><td>139682</td><td>17.404%</td><td>131754</td><td>10.740%</td><td>129952</td><td>9.225%</td><td>125353</td><td>5.360%</td></tr><tr><td>X-n1001-k43</td><td>72355</td><td>100399</td><td>38.759%</td><td>76689</td><td>5.990%</td><td>87660</td><td>21.153%</td><td>114448</td><td>58.176%</td><td>94734</td><td>30.929%</td><td>88969</td><td>22.962%</td><td>85929</td><td>18.760%</td><td>77739</td><td>7.441%</td></tr><tr><td colspan="2">Avg. Gap</td><td colspan="2">29.658%</td><td colspan="2">12.836%</td><td colspan="2">16.796%</td><td colspan="2">26.408%</td><td colspan="2">19.607%</td><td colspan="2">18.795%</td><td colspan="2">12.303%</td><td colspan="2">8.678%</td></tr></table>
25 
1375   
1376   
1377   
1378   
1379   
1380   
1381   
1382   
1383   
1384   
1385   
1386   
1387   
1388   
1389   
1390   
1391   
1392   
1393   
1394   
1395   
1396   
1397   
1398   
1399   
1400   
1401   
1402   
1403   
1404   
1405   
1406   
1407   
1408   
1409   
1410   
1411   
1412   
1413   
1414   
1415   
1416   
1417   
1418   
1419   
1420   
1421   
1422   
1423   
1424   
1425   
1426   
1427   
1428   
1429 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# J. Ablation Study
In this section, we conduct a detailed ablation study and analysis to demonstrate the effectiveness and robustness of URS. Please note that, unless stated otherwise, the results presented in the ablation study reflect the best result from multiple trajectories, and we employ instance augmentation to improve performance in the ablation study. We adopt the widely used 100-node setting as our primary evaluation scenario for URS. 
# J.1. Effects of Node-type Indicator

Table 17. The ablation study of the node-type indicator on trained VRPs

<table><tr><td>Method</td><td>ATSP</td><td>TSP</td><td>OP</td><td>PCTSP</td><td>PDTSP</td><td>ACVRP</td><td>CVRP</td><td>CVRPTW</td><td>CVRPB</td><td>OCVRP</td><td>OCVRPTW</td><td>Avg.gap</td></tr><tr><td>w/oξi</td><td>2.14%</td><td>0.48%</td><td>0.45%</td><td>1.06%</td><td>9.28%</td><td>2.84%</td><td>3.66%</td><td>6.61%</td><td>1.57%</td><td>7.87%</td><td>5.66%</td><td>3.78%</td></tr><tr><td>w/ξi</td><td>2.26%</td><td>0.57%</td><td>0.45%</td><td>1.06%</td><td>4.98%</td><td>3.06%</td><td>1.81%</td><td>6.13%</td><td>1.46%</td><td>3.24%</td><td>5.07%</td><td>2.74%</td></tr></table>
To evaluate the effectiveness of node-type indicator $\boldsymbol { \xi }$ , we conduct an ablation study about including and excluding $\boldsymbol { \xi }$ . Table 17 shows that adding the node-type indicator reduces the average optimality gap from $3 . 7 8 \%$ to $2 . 7 4 \%$ , with large gains on the routing problem with relatively rich instance node roles, such as PDTSP, where the coexistence of a depot and paired pickup–delivery nodes (with inherent precedence coupling) allows the explicit node-role encoding to further boost performance. Meanwhile, URS maintains a competitive performance in addressing the problem of single-node roles. 
# J.2. Effects of Problem State

Table 18. Performance impact of removing the problem state feature $C _ { t }$

<table><tr><td>Method</td><td>ATSP</td><td>TSP</td><td>OP</td><td>PCTSP</td><td>PDTSP</td><td>ACVRP</td><td>CVRP</td><td>CVRPTW</td><td>CVRPB</td><td>OCVRP</td><td>OCVRPTW</td><td>Avg.gap</td></tr><tr><td>w/o Ct</td><td>2.09%</td><td>0.45%</td><td>0.60%</td><td>1.20%</td><td>4.59%</td><td>4.03%</td><td>2.70%</td><td>5.71%</td><td>2.64%</td><td>4.02%</td><td>4.62%</td><td>2.97%</td></tr><tr><td>w/ Ct</td><td>2.26%</td><td>0.57%</td><td>0.45%</td><td>1.06%</td><td>4.98%</td><td>3.06%</td><td>1.81%</td><td>6.13%</td><td>1.46%</td><td>3.24%</td><td>5.07%</td><td>2.74%</td></tr></table>
As described in Appendix E.3, the majority of VRP variants are already solved without supplying an additional problem state $C _ { t }$ . To test whether the remaining use of $C _ { t }$ matters, we conduct an ablation in which we eliminate it entirely, enforcing $C _ { t } \equiv 0$ for all problems. The results in Table 18 show that URS maintains its competitive performance even in the absence of $C _ { t }$ . This indicates that explicit constraint-state features are not essential for the effectiveness of URS. Interestingly, on (O)CVRPTW instances, the model without any explicit constraint features outperforms the version that supplies only capacity information. We guess that capacity and time windows information are equally important, so enforcing only capacity creates a bias toward load considerations. 
# J.3. Effects of Mixed Bias Module

Table 19. Ablation of prior components in the Mixed Bias Module (MBM).

<table><tr><td>D</td><td>DT</td><td>R</td><td>ATSP</td><td>TSP</td><td>OP</td><td>PCTSP</td><td>PDTSP</td><td>ACVRP</td><td>CVRP</td><td>CVRPTW</td><td>CVRPB</td><td>OCVRP</td><td>OCVRPTW</td><td>Avg.gap</td></tr><tr><td>?</td><td>×</td><td>×</td><td>16.64%</td><td>0.44%</td><td>0.25%</td><td>0.67%</td><td>7.33%</td><td>16.62%</td><td>2.42%</td><td>6.44%</td><td>3.96%</td><td>9.44%</td><td>5.47%</td><td>6.33%</td></tr><tr><td>?</td><td>?</td><td>×</td><td>2.09%</td><td>0.45%</td><td>0.48%</td><td>0.82%</td><td>7.66%</td><td>9.11%</td><td>2.52%</td><td>6.89%</td><td>2.68%</td><td>5.70%</td><td>6.36%</td><td>4.07%</td></tr><tr><td>?</td><td>×</td><td>?</td><td>16.60%</td><td>0.46%</td><td>0.49%</td><td>0.94%</td><td>4.49%</td><td>26.22%</td><td>4.16%</td><td>6.57%</td><td>3.07%</td><td>12.23%</td><td>5.53%</td><td>7.34%</td></tr><tr><td>?</td><td>?</td><td>?</td><td>2.26%</td><td>0.57%</td><td>0.45%</td><td>1.06%</td><td>4.98%</td><td>3.06%</td><td>1.81%</td><td>6.13%</td><td>1.46%</td><td>3.24%</td><td>5.07%</td><td>2.74%</td></tr></table>
To further assess the effectiveness of MBM, we conduct an ablation study that incorporates different combinations of problem-specific priors into the MBM. Owing to its flexibility, the MBM permits imposing multiple priors on the same shared attention layer. The results in Appendix J.3 underscore the complementary roles of the three prior components: (1) The transposed distance matrix $D ^ { \mathbf { T } }$ captures directional asymmetry absent from the raw distance matrix, and its inclusion markedly stabilizes optimization on asymmetric variants; (2) Removing the pickup–delivery relation matrix $\pmb { R }$ degrades performance on PDTSPs, confirming that explicit relational coupling is indispensable for these instances; (3) When all priors are fused, MBM can simultaneously learn the geometric and relational biases inherent in various problems, yielding the best overall performance. 
26 
1430   
1431   
1432   
1433   
1434   
1435   
1436   
1437   
1438   
1439   
1440   
1441   
1442   
1443   
1444   
1445   
1446   
1447   
1448   
1449   
1450   
1451   
1452   
1453   
1454   
1455   
1456   
1457   
1458   
1459   
1460   
1461   
1462   
1463   
1464   
1465   
1466   
1467   
1468   
1469   
1470   
1471   
1472   
1473   
1474   
1475   
1476   
1477   
1478   
1479   
1480   
1481   
1482   
1483   
1484 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# J.4. Effects of Different Decoder Architectures

Table 20. Effect of different decoder architectures on URS cross-problem performance

<table><tr><td>Decoder</td><td>ATSP</td><td>TSP</td><td>OP</td><td>PCTSP</td><td>PDTSP</td><td>ACVRP</td><td>CVRP</td><td>CVRPTW</td><td>CVRPB</td><td>OCVRP</td><td>OCVRPTW</td><td>Avg.gap</td></tr><tr><td>URS-POMO</td><td>4.83%</td><td>0.74%</td><td>0.72%</td><td>1.29%</td><td>5.45%</td><td>5.09%</td><td>1.99%</td><td>6.42%</td><td>1.83%</td><td>3.62%</td><td>5.31%</td><td>3.39%</td></tr><tr><td>URS-ReLD</td><td>6.48%</td><td>0.90%</td><td>0.86%</td><td>1.51%</td><td>5.81%</td><td>6.51%</td><td>2.33%</td><td>6.56%</td><td>2.33%</td><td>3.62%</td><td>5.64%</td><td>3.87%</td></tr><tr><td>URS</td><td>2.26%</td><td>0.57%</td><td>0.45%</td><td>1.06%</td><td>4.98%</td><td>3.06%</td><td>1.81%</td><td>6.13%</td><td>1.46%</td><td>3.24%</td><td>5.07%</td><td>2.74%</td></tr></table>
We adopt an AM (Kool et al., 2019) as our basic encoder-decoder model. Multiple decoder refinements have been proposed around AM (Kwon et al., 2020; Huang et al., 2025; Zhou et al., 2024a). To assess how decoder structure impacts multi-task performance, we additionally train two URS variants whose decoders follow POMO and ReLD, denoted URS-POMO and URS-ReLD. For stronger and fairer baselines, we apply Distance-aware Attention Reshaping (DAR) (Wang et al., 2025b) to both, and all decoder parameters are generated by a hypernetwork conditioned on the problem representation $\boldsymbol { \lambda }$ (see Appendix E.2). As shown in Table 20, URS-POMO and URS-ReLD still exhibit strong cross-problem robustness, while replacing the decoder with ICAM (Zhou et al., 2024a) yields a further overall improvement. 
# J.5. Effects of Different Components on Zero-shot Generalization

Table 21. Ablation of different components in the model architecture, which includes MBM, WEIGHT $( \lambda )$ , and BIAS(λ). The symbol (*) indicates the seen problems in training.

<table><tr><td>MBM</td><td>WEIGHT(λ)</td><td>BIAS(λ)</td><td>ATSP(*)</td><td>TSP(*)</td><td>OP(*)</td><td>PCTSP(*)</td><td>PDTSP(*)</td><td>ACVRP(*)</td></tr><tr><td>×</td><td>?</td><td>?</td><td>16.64%</td><td>0.44%</td><td>0.25%</td><td>0.67%</td><td>7.33%</td><td>16.62%</td></tr><tr><td>?</td><td>×</td><td>×</td><td>4.24%</td><td>1.10%</td><td>1.04%</td><td>1.42%</td><td>6.38%</td><td>4.07%</td></tr><tr><td>?</td><td>×</td><td>?</td><td>3.83%</td><td>1.16%</td><td>1.09%</td><td>1.52%</td><td>6.65%</td><td>4.27%</td></tr><tr><td>?</td><td>?</td><td>×</td><td>2.79%</td><td>1.58%</td><td>1.39%</td><td>1.92%</td><td>7.05%</td><td>3.39%</td></tr><tr><td>?</td><td>?</td><td>?</td><td>2.26%</td><td>0.57%</td><td>0.45%</td><td>1.06%</td><td>4.98%</td><td>3.06%</td></tr><tr><td>MBM</td><td>WEIGHT(λ)</td><td>BIAS(λ)</td><td>CVRP(*)</td><td>CVRPTW(*)</td><td>CVRPB(*)</td><td>OCVRP(*)</td><td>OCVRPTW(*)</td><td>CVRPL</td></tr><tr><td>×</td><td>?</td><td>?</td><td>2.42%</td><td>6.44%</td><td>3.96%</td><td>9.44%</td><td>5.47%</td><td>0.98%</td></tr><tr><td>?</td><td>×</td><td>×</td><td>6.23%</td><td>7.22%</td><td>5.00%</td><td>3.99%</td><td>5.41%</td><td>2.92%</td></tr><tr><td>?</td><td>×</td><td>?</td><td>4.94%</td><td>6.08%</td><td>3.00%</td><td>4.61%</td><td>5.21%</td><td>4.79%</td></tr><tr><td>?</td><td>?</td><td>×</td><td>2.90%</td><td>6.60%</td><td>3.12%</td><td>4.47%</td><td>4.76%</td><td>1.46%</td></tr><tr><td>?</td><td>?</td><td>?</td><td>1.81%</td><td>6.13%</td><td>1.46%</td><td>3.24%</td><td>5.07%</td><td>0.43%</td></tr><tr><td>MBM</td><td>WEIGHT(λ)</td><td>BIAS(λ)</td><td>OCVRPB</td><td>CVRPBL</td><td>CVRPLTW</td><td>OCVRPBTW</td><td>CVRPBLTW</td><td>OCVRPL</td></tr><tr><td>×</td><td>?</td><td>?</td><td>15.95%</td><td>4.18%</td><td>2.89%</td><td>18.78%</td><td>10.93%</td><td>9.51%</td></tr><tr><td>?</td><td>×</td><td>×</td><td>9.54%</td><td>5.30%</td><td>2.73%</td><td>14.37%</td><td>9.80%</td><td>4.64%</td></tr><tr><td>?</td><td>×</td><td>?</td><td>9.58%</td><td>3.25%</td><td>3.07%</td><td>14.16%</td><td>9.91%</td><td>3.88%</td></tr><tr><td>?</td><td>?</td><td>×</td><td>9.92%</td><td>3.34%</td><td>3.08%</td><td>14.02%</td><td>9.92%</td><td>4.52%</td></tr><tr><td>?</td><td>?</td><td>?</td><td>9.35%</td><td>1.65%</td><td>2.67%</td><td>13.77%</td><td>9.18%</td><td>3.22%</td></tr><tr><td>MBM</td><td>WEIGHT(λ)</td><td>BIAS(λ)</td><td>CVRPBTW</td><td>OCVRPBL</td><td>OCVRPLTW</td><td>OCVRPBLTW</td><td>Avg.gap</td><td>Best Sol.</td></tr><tr><td>×</td><td>?</td><td>?</td><td>10.74%</td><td>16.14%</td><td>5.57%</td><td>18.80%</td><td>8.37%</td><td>3/22</td></tr><tr><td>?</td><td>×</td><td>×</td><td>9.67%</td><td>10.67%</td><td>5.23%</td><td>14.49%</td><td>6.16%</td><td>0/22</td></tr><tr><td>?</td><td>×</td><td>?</td><td>9.63%</td><td>10.42%</td><td>5.51%</td><td>13.99%</td><td>5.93%</td><td>1/22</td></tr><tr><td>?</td><td>?</td><td>×</td><td>9.79%</td><td>10.02%</td><td>5.81%</td><td>13.98%</td><td>5.72%</td><td>1/22</td></tr><tr><td>?</td><td>?</td><td>?</td><td>8.94%</td><td>9.47%</td><td>5.12%</td><td>13.72%</td><td>4.89%</td><td>17/22</td></tr></table>
To empirically evaluate the necessity of these components, we have conducted tests on the 11 seen problems as well as 11 additional unseen CVRP variants widely investigated in current multi-task research (e.g., MVMoE (Zhou et al., 2024b)). The results are presented in Table 21. We can observe that removing MBM leads to a significant performance drop on problems with complex geometric properties, such as ATSP, ACVRP, and PDTSP, which confirms MBM is critical for perceiving specific geometric biases. In addition, removing WEIGHT(λ) or BIAS(λ) also results in performance degradation across most tasks. In terms of overall performance, URS significantly outperforms any model where a single module is removed. The full URS model achieves the lowest average gap and provides the best solution on the majority of tasks (17/22). This demonstrates that the synergy of these components is essential for robust cross-problem generalization. 
27 
1485   
1486   
1487   
1488   
1489   
1490   
1491   
1492   
1493   
1494   
1495   
1496   
1497   
1498   
1499   
1500   
1501   
1502   
1503   
1504   
1505   
1506   
1507   
1508   
1509   
1510   
1511   
1512   
1513   
1514   
1515   
1516   
1517   
1518   
1519   
1520   
1521   
1522   
1523   
1524   
1525   
1526   
1527   
1528   
1529   
1530   
1531   
1532   
1533   
1534   
1535   
1536   
1537   
1538   
1539 
URS: A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization 
# K. Licenses for Used Resources

Table 22. List of licenses for the codes and datasets we used in this work.

<table><tr><td>Resource</td><td>Type</td><td>Link</td><td>License</td></tr><tr><td>HGS-PyVRP (Wouda et al., 2024)</td><td>Code</td><td>https://github.com/PyVRP/PyVRP</td><td>MIT License</td></tr><tr><td>LKH3 (Helsgaun, 2017)</td><td>Code</td><td>http://webhotel4.ruc.dk/~keld/research/LKH-3/</td><td>Available for academic research use</td></tr><tr><td>OR-Tools (Perron &amp; Furnon, 2023)</td><td>Code</td><td>https://github.com/google/or-tools</td><td>Apache-2.0 License</td></tr><tr><td>POMO (Kwon et al., 2020)</td><td>Code</td><td>https://github.com/yd-kwon/POMO/tree/master/NEW_PY_ver</td><td>MIT License</td></tr><tr><td>Sym-NCO (Kim et al., 2022)</td><td>Code</td><td>https://github.com/almstn12088/Sym-NCO</td><td>Available for any non-commercial use</td></tr><tr><td>LEHD (Luo et al., 2023)</td><td>Code</td><td>https://github.com/CTAM-Group/NCO_code/tree/main/single_objective/LEHD</td><td>Available for any non-commercial use</td></tr><tr><td>BQ (Drakulic et al., 2023)</td><td>Code</td><td>https://github.com/naver/bq-nco</td><td>CC BY-NC-SA 4.0 license</td></tr><tr><td>ICAM (Zhou et al., 2024a)</td><td>Code</td><td>https://github.com/CTAM-Group/ICAM</td><td>MIT License</td></tr><tr><td>ELG (Gao et al., 2024)</td><td>Code</td><td>https://github.com/gaocrr/ELG</td><td>MIT License</td></tr><tr><td>MatNet (Kwon et al., 2021)</td><td>Code</td><td>https://github.com/yd-kwon/MatNet/tree/main</td><td>MIT License</td></tr><tr><td>Heter-AM (Li et al., 2021)</td><td>Code</td><td>https://github.com/jingwenli0312/Heterogeneous-Attentions-PDP-DRL</td><td>MIT License</td></tr><tr><td>MTPOMO (Liu et al., 2024)</td><td>Code</td><td>https://github.com/FeiLiu36/MTNCO</td><td>MIT License</td></tr><tr><td>MVMoE (Zhou et al., 2024b)</td><td>Code</td><td>https://github.com/RoyalSkye/Routing-MVMoE</td><td>MIT License</td></tr><tr><td>RouteFinder (Berto et al., 2025)</td><td>Code</td><td>https://github.com/ai4co/routefinder</td><td>MIT License</td></tr><tr><td>CaDA (Li et al., 2025a)</td><td>Code</td><td>https://github.com/CTAM-Group/CaDA</td><td>MIT License</td></tr><tr><td>ReLD (Huang et al., 2025)</td><td>Code</td><td>https://github.com/ziweileonhuang/reld-nco</td><td>MIT License</td></tr><tr><td>GOAL (Drakulic et al., 2025)</td><td>Code</td><td>https://github.com/naver/goal-co</td><td>Available for any non-commercial use</td></tr><tr><td>CVRPLIB Set-X (Uchoa et al., 2017)</td><td>Dataset</td><td>http://vrp.galgos.inf.puc-rio.br/index.php/en/</td><td>Available for academic research use</td></tr><tr><td>CVRPLIB Set-XXL (Arnold et al., 2019)</td><td>Dataset</td><td>http://vrp.galgos.inf.puc-rio.br/index.php/en/</td><td>Available for academic research use</td></tr></table>
We list the used existing codes and datasets in Table 22, and all of them are open-sourced resources for academic usage. 
28 