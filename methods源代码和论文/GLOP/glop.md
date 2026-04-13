# GLOP: Learning Global Partition and Local Construction for Solving Large-scale Routing Problems in Real-time



Authors: Haoran Ye¹, Jiarui Wang¹, Helan Liang¹, Zhiguang Cao², Yong Li³, Fanzhang Li¹ 1

Affiliations:

¹Soochow University, China 2²Singapore Management University, Singapore 3³Tsinghua University, China 4Contact: {hrye, jrwangfurffico}@stu.suda.edu.cn, {hlliang,lfzh}@suda.edu.cn, zgcao@smu.edu.sg, liyong07@tsinghua.edu.cn 5

------

## Abstract

The recent end-to-end neural solvers have shown promise for small-scale routing problems but suffered from limited real-time scaling-up performance. 6This paper proposes GLOP (Global and Local Optimization Policies), a unified hierarchical framework that efficiently scales toward large-scale routing problems. 7GLOP partitions large routing problems into Travelling Salesman Problems (TSPs) and TSPs into Shortest Hamiltonian Path Problems. 8For the first time, we hybridize non-autoregressive neural heuristics for coarse-grained problem partitions and autoregressive neural heuristics for fine-grained route constructions, leveraging the scalability of the former and the meticulousness of the latter. 9Experimental results show that GLOP achieves competitive and state-of-the-art real-time performance on large-scale routing problems, including TSP, ATSP, CVRP, and PCTSP. 10Our code is available: https://github.com/henry-yeh/GLOP. 11



------

## 1 Introduction

Routing problems pervade logistics, supply chain, transportation, robotic systems, etc. 12Modern industries have witnessed ever-increasing demands for the massive and expeditious routing of goods, services, and people. 13Traditional solvers based on mathematical programming or iterative heuristics struggle to keep pace with such growing complexity and real-time requirements. 14



Recent advances in Neural Combinatorial Optimization (NCO) seek end-to-end solutions for routing problems, where neural solvers are exploited and empowered by massive training while enjoying potentially efficient inference. 15However, most existing NCO methods still struggle with real-time scaling-up performance; 16they are unable to solve routing problems involving thousands or tens of thousands of nodes in seconds, falling short of the needs of modern industries. 17



In answer to that, this work proposes GLOP (Global and Local Optimization Policies) which partitions a large routing problem into sub-Travelling Salesman Problems (TSPs) and further partitions potentially large (sub-)TSPs into small Shortest Hamiltonian Path Problems (SHPPs). 18GLOP hybridizes non-autoregressive (NAR) global partition and autoregressive (AR) local construction policies, where the global policy learns the first partition and the local policy learns to solve SHPPs. 19We intend to integrate the strengths while circumventing the drawbacks of NAR and AR paradigms. 20In particular, partitioning nodes into subsets (each corresponding to a TSP) well suits NAR heuristics, because it is a large-scale but coarse-grained task agnostic of within-subset node ordering. 21On the other hand, solving SHPPs could be efficiently handled with the AR heuristic because it is a small-scale but fine-grained task. 22



The solution pipeline of GLOP is applicable to variations of routing problems, such as those tackled in recent literature. 23We evaluate GLOP on canonical TSP, Asymmetric TSP (ATSP), Capacitated Vehicle Routing Problem (CVRP), and Prize Collecting TSP (PCTSP). 24GLOP for (A)TSP, as opposed to most methods that require scale-specific and distribution-specific training, can perform consistently and competitively across scales, across distributions, and on real-world benchmarks, using the same set of local policies. 25Notably, it is the first neural solver to effectively scale to TSP100K, obtaining a 5.1% optimality gap and a 174x speedup compared with 1-run 1-trial LKH-3. 26GLOP for CVRP clearly outperforms prior state-of-the-art (SOTA) real-time solvers while using 10x less execution time. 27On PCTSP, GLOP surpasses both recent neural solvers and conventional solvers. 28



Accordingly, we summarize our contributions as follows:

- We propose GLOP, a versatile framework that extends existing neural solvers to large-scale problems. 29

  

  

- To our knowledge, it makes the first effective attempt at hybridizing NAR and AR end-to-end NCO paradigms. 30

  

  

- We propose to learn global partition heatmaps for decomposing large-scale routing problems, leveraging NAR heatmap learning in a novel way. 31

  

  

- We propose a one-size-fits-all real-time (A)TSP solver that learns small SHPP solution construction for arbitrarily large (A)TSP. 32

  

  

- We dispense with learning upper-level TSP policies suggested in previous works while achieving better performance. 33

  

  

- On (A)TSP, GLOP delivers competitive scaling-up and cross-distribution performance and is the first neural solver to scale to TSP100K effectively. 34On CVRP and PCTSP, GLOP achieves SOTA real-time performance. 35

  

  

------

## 2 Background and related work

### 2.1 Neural Combinatorial Optimization (NCO)

Recent advances in NCO show promise for solving combinatorial optimization problems in an end-to-end manner. 36The end-to-end neural routing solvers can be categorized into two paradigms: AR solution construction and NAR heatmap generation coupled with subsequent decoding. 37We defer further discussions to Appendix C. 38



### 2.2 Divide and conquer for VRP

The idea of "divide and conquer" has long been applied to VRP variants in traditional (meta) heuristics. 39Recently, such an idea has been introduced in neural routing solvers. 40Li, Yan, and Wu (2021) propose learning to delegate (L2D) the improvement of subtours to LKH-3. 41 Zong et al. (2022a) introduce Rewriting-by-Generating (RBG) framework that involves repeated learning-based merging and rule-based decomposition. 42However, both methods rely on iterative refinement, therefore holding back the real-time performance. 43 More related to GLOP, Hou et al. (2023) present a Two-stage Divide Method (TAM), the prior SOTA real-time neural solver for large-scale CVRP, where a dividing model learns to partition CVRP into sub-TSPs autoregressively, and the sub-TSPs are then solved by sub-solvers such as Attention Model (AM) or LKH-3. 44Unlike other prior works, both TAM and GLOP target large-scale CVRP under real-time settings. 45By comparison, GLOP outperforms TAM on CVRP by leveraging more effective global representations and better neural sub-TSP solvers, and can also handle routing problems that TAM does not address. 46



### 2.3 Local construction for TSP

Learning local subtour reconstruction for TSP is initially introduced by Kim, Park, and Kim (2021) in Learning Collaborative Policy (LCP). 47LCP generates diversified initial solutions (seeds) with neural models (seeders), then repeatedly decomposes and reconstructs them. 48However, LCP, limited mainly by the design of seeders, can hardly scale up to TSP with hundreds of nodes. 49 More recently, Pan et al. (2023) propose H-TSP, a hierarchical TSP solver interleaving forming open-loop TSP (a.k.a., SHPP) with upper-level policies and conquering it. 50By comparison, GLOP dispenses with learning any upper-level TSP policy but is able to outperform H-TSP. 51Another concurrent work, namely select-and-optimize (SO) (Cheng et al. 2023), utilizes a TSP solution pipeline similar to GLOP. 52But SO heavily relies on sophisticated heuristics specific to TSP, resulting in prolonged computational time. 53By comparison, GLOP achieves competitive solutions while being hundreds of times more efficient. 54



------

## 3 Methodology overview

GLOP is schematically illustrated in Figure 1. 55It aims to provide a unified and scalable framework for heterogeneous vehicle routing problems. 56To this end, our design targets three representative problem settings: (1) large-scale TSP alone, (2) large-scale CVRP requiring problem partitioning and solving multiple small sub-TSPs, and (3) large-scale PCTSP requiring problem partitioning and solving a single large sub-TSP. 57We defer the detailed explanations of these problems to Appendix D. 58



In general, GLOP learns local policies for (sub-)TSP and global policies for partitioning general routing problems into sub-TSPs. 59Our (sub-)TSP solver generates initial TSP tours using Random Insertion, divides the complete tours into independent subtours, and learns to reconstruct them for improvements. 60Our general routing solver additionally learns to perform node clustering or subsetting that generates sub-TSP(s). 61We elaborate on our local policy and global policy in Section 4 and Section 5, respectively, and provide more details in Appendix A. 62





**Figure 1: The pipeline of GLOP.** 63



- 

  **(Sub-)TSP Solver**: Involves Insert (Random Insertion), Divide (into SHPP subtours), Conquer (Neural heuristic with local policy), and Compose. 64

  

  

- **General Routing Solver**: Involves Partition (using Partition heatmap parameterized by Neural heuristic with global policy), followed by solving Sub-TSPs using the (Sub-)TSP Solver. This covers Clustering (multiple Sub-TSPs) and Subsetting (single Sub-TSP). 65

  

  

------

## 4 (Sub-)TSP solver

### 4.1 Inference pipeline

GLOP learns local policies to improve a TSP solution by decomposing and reconstructing it. 66



- 

  **Initialization**: GLOP generates an initial TSP tour with Random Insertion (RI), a simple and generic heuristic. 67RI greedily picks the insertion place that minimizes the insertion cost for each node. 68

  

  

- 

  **Revisions**: Then, GLOP performs improvements on the initial tour. 69Following Kim, Park, and Kim (2021), we refer to a round of improvement as a "revision"; we refer to a local policy parameterized by an autoregressive NN and trained to solve SHPPn (SHPP of n nodes) as "Reviser-n". 70A revision involves decomposing and reconstructing the initial tour, comprising four sequential steps outlined below. 71

  

  

- 

  **Decomposition**: When improved by Reviser-n, a complete tour with $N$ nodes is randomly decomposed into $\lfloor\frac{N}{n}\rfloor$ subtours, each with $n$ nodes. 72There is no overlap between every two subtours. 73A "tail subtour" with $N \mod n$ nodes, if any, is left untouched until composition. 74Each subtour corresponds to an SHPP graph, and reconstructing a subtour is equivalent to solving an SHPP instance. 75We pick the decomposition positions uniformly when performing repeated revisions. 76

  

  

- 

  **Transformation and augmentation**: To improve the predictability and homogeneity of the model inputs, we apply Min-max Normalization and an optional rotation to the SHPP graphs. 77They scale the x-axis coordinates to the range [0,1] and set the lower bound of the y-axis to 0. 78In addition, we augment the SHPP instances by flipping the node coordinates to enhance the model performance. 79

  

  

- 

  **Solving SHPPs with local policies**: We autoregressively reconstruct the subtours (i.e., solve the SHPP instances) with trainable revisers. 80Any SHPP solutions that are worse than the current ones will be discarded. 81

  

  

- 

  **Composition**: The $\lfloor\frac{N}{n}\rfloor$ reconstructed (or original) subtours and a tail subtour, if any, compose an improved complete tour by connecting the starting/terminating nodes of SHPPs in their original order. 82

  

  

GLOP can apply multiple revisers to solve a problem from different angles. 83Also, a single reviser can decompose the tour at different points and repeat its revisions. 84After all revisions, GLOP outputs the improved tour as its final solution. 85Notably, GLOP allows applying a single set of small-SHPP-trained models for arbitrarily large TSPs. 86



### 4.2 Solving SHPP with local policy

- 

  **Problem formulation and motivation**: SHPP is also referred to as open-loop TSP. 87With starting/terminating nodes fixed, it aims to minimize the length of a Hamiltonian path visiting all nodes in between exactly once. 88Solving small SHPPs with neural networks, instead of directly solving TSP or with traditional heuristics, makes GLOP a highly parallelizable one-size-fits-all solution. 89

  

  

- 

  **Model**: We parameterize our local policies based on Attention Model (AM). 90To apply it to SHPP, we adjust its context embedding and leverage the solution symmetries by autoregressively constructing solutions from both starting/determining nodes. 91

  

  

- 

  **Local policy**: Given an SHPP instance $s$ with starting/terminating node 1 and $n$, our stochastic local policy $p_{\theta}(\omega_{fd},\omega_{bd}|s)$, parameterized by the neural model $\theta$, denotes the conditional probability of constructing forward and backward-decoded solutions $\omega_{fd}$ and $\omega_{bd}$. 92We let $\omega_{1:t-1}$ denote the partial solution at time step $t$, then the local policy can be factorized into probability distribution of per-step construction: 93

  

  

  $$p_{\theta}(\omega_{fd},\omega_{bd}|s)=p_{\theta}(\omega_{fd}|s)\times p_{\theta}(\omega_{bd}|s) = \prod_{t=1}^{n-2}p_{\theta}(\omega_{t}|s,\omega_{1:t-1},n)\times p_{\theta}(\omega_{t}|s,\omega_{1:t-1},1) \quad (1)$$

  94We accept the better one between $\omega_{fd}$ and $\omega_{bd}$ during inference while making use of both for training. 95

  

  

### 4.3 Training algorithm

We train our parameterized local policy, i.e., a reviser, by minimizing the expected length of its constructed SHPP solutions: 96



$$\text{minimize } \mathcal{L}(\theta|s)=\mathbb{E}_{\omega_{fd},\omega_{bd}\sim p_{\theta}(\omega_{fd},\omega_{bd}|s)} [f_{SHPP}(\omega_{fd},s)+f_{SHPP}(\omega_{bd},s)], \quad (2)$$

97where $f_{SHPP}$ maps an SHPP solution to its length. 98We apply the REINFORCE-based gradient estimator using the average path length of two greedy rollouts as a baseline. 99This training algorithm doubles the experience learned on each instance and enables a more reliable baseline by weighing the greedy rollouts of both directions. 100





**Two-stage curriculum learning**: According to our coordinate transformation, we design a two-stage curriculum to improve the homogeneity and consistency between training and inference instances. 101We are motivated by the following observation: the inputs to revisers are 1) SHPP graphs with y-axis upper bounds ranging from 0 to 1 after our coordinate transformation, and also 2) the outputs of its preceding module. 102Therefore, stage 1 in our curriculum trains revisers using multi-distribution SHPPs with varied y-axis upper bounds, and stage 2 collaboratively fine-tunes all revisers following the inference pipeline. 103



------

## 5 General routing solver

Many routing problems can be formulated hierarchically, which requires node clustering (e.g. CVRP, mTSP, Capacitated Arc Routing Problem) or node subsetting (e.g. PCTSP, Orienteering Problem, Covering Salesman Problem), followed by solving multiple sub-TSPs or a single sub-TSP, respectively. 104For these general routing problems, GLOP involves an additional global partition policy defined by a parameterized partition heatmap and trained with parallel on-policy sampling without costly step-by-step neural decoding. 105The applicability of our global policy is also discussed. 106



### 5.1 Global policy as partition heatmap

- 

  **Partition heatmap**: We introduce a parameterized partition heatmap $\mathcal{H}_{\phi}(\rho)=[h_{ij}(\rho)]_{(n+1)\times(n+1)}$ where $\rho$ is the input instance with $n+1$ nodes including node 0 as the depot. 107$h_{ij}\in\mathbb{R}^{+}$ represents the unnormalized probability of nodes $i$ and $j$ belonging to the same subset. 108

  

  

- 

  **Model and input graph**: The partition heatmap is parameterized by an isomorphic GNN. 109Inputs to the model are sparsified graphs with features designed separately for different problems. 110We defer the full details to Appendix A.4. 111

  

  

- 

  **Global policy**: For node clustering, GLOP partitions all nodes into multiple subsets, each corresponding to a sub-TSP to solve. 112For node subsetting, GLOP partitions all nodes into two subsets, i.e., the to-visit subset and the others, where the to-visit subset forms a sub-TSP to solve. 113Let $\pi=\{\pi^{r}\}_{r=1}^{|\pi|}$ denote a complete partition and $\pi^{r}=\{\pi_{t}^{r}\}_{t=1}^{|\pi^{r}|}$ the $r$-th subset containing both regular nodes and the depot. 114 Each subset begins and terminates at the depot; that is, $\pi_{1}^{r}=\pi_{|\pi^{r}|}^{r}=0$. Given $\mathcal{H}_{\phi}(\rho)$, our global policy partitions all nodes into subsets by sequentially sampling nodes while satisfying problem-specific constraints: 115

  

  

  $$p_{\phi}(\pi|\rho)=\prod_{\tau=1}^{|\pi|} \prod_{i=1}^{|\pi^{r}|-1} \frac{h_{\pi_{i}^{r},\pi_{i+1}^{r}}(\rho)}{\sum_{k\in \mathcal{N}(\pi^{\rho})}h_{\pi_{i}^{r},k}(\rho)} \quad \text{if } \pi \in \Theta, \quad \text{otherwise } 0 \quad (3)$$

  116where $\mathcal{N}(\pi^{\rho})$ is the set of feasible actions given the current partial partition. 117For our benchmark problems, the applied constraints are given in Appendix A.4. 118

  

  

- 

  **Decoding**: We apply greedy (GLOP-G) and sampling (GLOP-S) heatmap decoding to draw the node partitions following our global policy. 119

  

  

### 5.2 Training algorithm

We train our global policy to output partitions that could lead to the best-performing final solutions after solving sub-TSPs. 120Given each instance $\rho$, the training algorithm infers partition heatmap $\mathcal{H}_{\phi}(\rho)$, samples node partitions in parallel, feeds the sampled partitions into GLOP for sub-TSP solutions, and optimizes the expected final performance: 121



$$\text{minimize } \mathcal{L}(\phi|\rho)=\mathbb{E}_{\pi\sim p_{\phi}(\pi|\rho)}[\sum_{r=1}^{|\pi|}f_{TSP}(GLOP_{\theta}(\pi^{r},\rho))], \quad (4)$$

122where $f_{TSP}$ is a mapping from a sub-TSP solution to its length, and $GLOP_{\theta}$ generates sub-TSP solutions with well-trained local policies. 123We apply the REINFORCE algorithm with the averaged reward of sampled solutions for the same instance as a baseline. 124The baseline is respectively computed for each instance within each training batch. 125GLOP for sub-TSP, i.e. $GLOP_{\theta}$, enables efficient training of our global policy on large-scale problems due to its parallelizability and scalability. 126



### 5.3 Applicability

Many routing problems can be formulated hierarchically, involving node clustering and/or node subsetting, depending on the problem formulation. 127Node clustering is used when the problem requires formulating multiple routes that cover all nodes, while node subsetting is used when the problem requires formulating a single route that covers a subset of nodes. 128In some cases, a routing problem may require both subsetting and clustering. 129Our global policy offers a unified formulation for all these scenarios. 130Additionally, it can easily handle constraints via masking if they can be anticipated while constructing node subsets. 131To handle more complex constraints, one can assign a large negative value to the rewards of infeasible solutions or apply post-processing techniques before solution evaluation. 132



------

## 6 Experimentation

### 6.1 Experimental Setup

- 

  **Datasets**: We refer the readers to Kool, van Hoof, and Welling (2019) for more specific definitions of benchmark problems. 133

  

  

  - (1) We evaluate GLOP on uniformly sampled large-scale TSP instances (i.e., TSP500, 1K, and 10K) used in Fu, Qiu, and Zha (2021) and an additionally generated TSP100K instance. 134We perform a cross-distribution evaluation on test instances used in (Bi et al. 2022). 135For evaluation on real-world benchmarks, we draw all 49 symmetric TSP instances featuring EUC 2D and containing fewer than 1000 nodes (since most baselines cannot process larger-scale instances) from TSPLIB and map all instances to the [0, 1]² square through Min-max Normalization. 136 The test datasets of ATSP are generated following Kwon et al. (2021). 137

    

    

  - (2) For CVRP, we adhere to the settings in TAM (Hou et al. 2023) and use the code of AM (Kool, van Hoof, and Welling 2019) to generate test datasets on CVRP1K, 2K, 5K, and 7K, each containing 100 instances. 138138138138We also evaluate GLOP on several large-scale CVRPLIB instances. 139

    

    

  - (3) For PCTSP, we follow the settings in AM (Kool, van Hoof, and Welling 2019) for data generation on PCTSP500, 1K, and 5K. 140As suggested by Kool, van Hoof, and Welling (2019), we specify $K^{n}=9,12,20$ to sample prizes for $n=500,$ 1K, 5K, respectively, i.e., $\beta_{i}\sim Uniform(0,3\frac{K^{n}}{n})$. 141

    

    

- 

  **Baselines**: Evaluating an NCO method typically involves two metrics: the objective value and the runtime. 142To ensure the validity of our comparisons, we select SOTA baselines with adjustable runtime that can match GLOP. 143We defer detailed implementations of the baselines to Appendix F. 144

  

  

- 

  **Hardware**: Unless otherwise stated, GLOP and the baselines are executed on a 12-core Intel(R) Xeon(R) Platinum 8255C CPU and an NVIDIA RTX 3090 Graphics Card. 145

  

  

### 6.2 Travelling Salesman Problem



**Large-scale TSP**: Comparison results on large-scale TSP are shown in Table 1 and Table 2. 146For GCN+MCTS and DIMES+MCTS, we use all 12 CPU cores for MCTS and limit its running time to ensure comparable results. 147





**Table 1: Comparison results on 128 TSP500, 128 TSP1K, and 16 TSP10K.** 148



| **Method**            | **Obj. TSP500** | **Gap(%)** | **Time**  | **Obj. TSP1K** | **Gap(%)** | **Time**  | **Obj. TSP10K** | **Gap(%)** | **Time**  |
| --------------------- | --------------- | ---------- | --------- | -------------- | ---------- | --------- | --------------- | ---------- | --------- |
| Concorde (2006)       | 16.55           | 0.00       | 40.9m     | 23.12          | 0.00       | 8.2h      | 71.77           | 0.00       | 13h       |
| LKH-3 (2017)          | 16.55           | 0.00       | 5.5m      | 23.12          | 0.00       | 24m       | -               | -          | -         |
| Random Insertion      | 18.59           | 12.3       | <1s       | 26.12          | 13.0       | <1s       | 81.84           | 14.0       | 5.7s      |
| AM (2019)             | 22.60           | 36.6       | 5.8m      | 42.53          | 84.0       | 22m       | 430             | 499        | 3.5m      |
| LCP (2021)            | 20.82           | 25.8       | 29m       | 36.34          | 57.2       | 34m       | 357             | 397        | 4.3m      |
| GCN+MCTS x12 (2021)   | 16.96           | 2.48       | 2.4m+33s  | 23.86          | 3.20       | 4.9m+1.2m | 75.73           | 5.50       | 7.1m+6.0m |
| POMO-EAS (2022)       | 24.04           | 45.3       | 1.0h      | 47.79          | 107        | 8.6h      | 86.25           | 20.0       | 3.1m      |
| DIMES+S (2022)        | 19.06           | 15.0       | 2.0m      | 26.96          | 16.1       | 2.4m      | 76.02           | 5.90       | 13.7m+20m |
| DIMES+MCTS x12 (2022) | 17.01           | 2.78       | 1.0m+2.1m | 23.86          | 3.20       | 2.6m+1.0m | OOM             | OOM        | -         |
| Tspformer (2023)      | 17.57           | 5.97       | 3.1m      | 27.02          | 16.9       | 5.0m      | -               | -          | -         |
| H-TSP (2023)          | 17.14           | 3.56       | 1.0m      | 24.65          | 6.62       | 47s       | 77.75           | 7.32       | 48s       |
| Pointerformer (2023)  | 16.94           | 2.36       | 4.3m      | 24.80          | 7.30       | 6.5m      | -               | -          | -         |
| DeepACO (2023)        | -               | -          | -         | 23.85          | 3.16       | 1.1h      | -               | -          | -         |
| GLOP                  | 17.07           | 3.14       | 19s       | 24.01          | 3.85       | 34s       | 75.62           | 5.36       | 32s       |
| GLOP (more revisions) | 16.91           | 1.99       | 1.5m      | 23.84          | 3.11       | 3.0m      | 75.29           | 4.90       | 1.8m      |

*Note*: OOM: out of our graphics memory (24GB). *: Results are drawn from the original literature with runtime proportionally adjusted. 149





**Table 2: Comparison results on a TSP100K instance.** 150



| **Method**            | **Obj.** | **Gap(%)** | **Time** |
| --------------------- | -------- | ---------- | -------- |
| LKH-3 {T=1}           | 226.4    | 0.00       | 8.1h     |
| Random Insertion      | 258.5    | 14.2       | 1.7m     |
| AM                    | OOM      | -          | -        |
| LCP                   | OOM      | -          | -        |
| GCN+MCTS x12          | OOM      | -          | -        |
| POMO-EAS              | OOM      | -          | -        |
| DIMES+MCTS x12        | OOM      | -          | -        |
| DIMES+S               | 286.1    | 26.4       | 2.0m     |
| H-TSP                 | OOM      | -          | -        |
| Pointerformer         | OOM      | -          | -        |
| GLOP                  | 240.0    | 6.01       | 1.8m     |
| GLOP (more revisions) | 238.0    | 5.10       | 2.8m     |



**Table 3: Comparison results on TSPLIB instances.** 151



| **Method**            | **Avg. gap(%)** | **Time** |
| --------------------- | --------------- | -------- |
| LCP {M=1280}          | 99.9            | 3.6m     |
| DACT {T=1K}           | 865             | 50m      |
| GCN+MCTS x1           | 1.10            | 7.5m     |
| {T=60} POMO-EAS       | 18.8            | 20m      |
| DIMES+MCTS x1         | 2.21            | 7.4m     |
| AMDKD+EAS {T=100}     | 7.86*           | 48m      |
| Pointerformer         | 6.04            | 48s      |
| GLOP                  | 1.53            | 42s      |
| GLOP (more revisions) | 0.69            | 2.6m     |

From the results, GLOP is highly efficient due to its decomposed solution scheme. 152Furthermore, the memory consumption of GLOP can be basically invariant of the problem scale if reconstructing the subtours using a fixed batch size. 153Hence, it is the first neural solver to effectively scale to TSP100K, obtaining a 5.1% optimality gap and a 174x speed-up compared to LKH-3. 154Compared with LCP and H-TSP, GLOP dispenses with learning upper-level TSP policies while achieving better performance. 155Compared with NAR methods conducting MCTS refinements, GLOP generates reasonable solutions even before they have finished initialization or produced prerequisite heatmaps for solution decoding. 156Hence, GLOP exhibits clear advantages for real-time applications. 157





**Cross-distribution TSP**: Table 5 gathers the comparison between GLOP and two baselines specially devised for cross-distribution performance on four TSP100 datasets with different distributions, i.e., uniform, expansion, explosion, and implosion. 158 Recall that we use uniformly distributed samples to train GLOP. Hence, the latter three datasets contain out-of-distribution (OoD) instances. 159Results show that GLOP obtains smaller gaps and less Det. on most OoD datasets. 160We argue that the holistic solution scheme of GLOP is the main contributor to its cross-distribution performance. 161





**Table 5: Comparison results on the OoD datasets.** 162



| **Method**            | **Time** | **Uniform Gap(%)** | **Expansion Gap(%) Det.(%)** | **Explosion Gap(%) Det.(%)** | **Implosion Gap(%) Det.(%)** |
| --------------------- | -------- | ------------------ | ---------------------------- | ---------------------------- | ---------------------------- |
| AM                    | 0.5h     | 2.310              | 17.97 / 678                  | 3.817 / 65                   | 2.431 / 5.2                  |
| AM+HAC                | 0.5h     | 2.484              | 3.997 / 61                   | 3.084 / 24                   | 2.595 / 4.5                  |
| GLOP                  | 0.091    | 0.166 / 82         | 0.066 / -27                  | 0.082 / -9.9                 |                              |
| AMDKD+EAS             | 2.0h     | 0.078              | 0.165 / 112                  | 0.048 / -39                  | 0.079 / 1.3                  |
| GLOP (more revisions) | 0.048    | 0.076 / 60         | 0.028 / -41                  | 0.044 / -8.3                 |                              |



**Real-world TSPLIB**: We evaluate GLOP and the baselines on real-world TSPLIB instances and collect the results in Table 3, where GLOP performs favorably against the baselines due to its consistent performance across scales and distributions. 163





**Asymmetric TSP**: GLOP is compatible with any neural architecture and can be extended to asymmetric distance. 164Table 4 exemplifies this flexibility on ATSP, where we replace AM with MatNet which is specially designed for ATSP. 165The results validate that GLOP can successfully extend MatNet to solve large ATSP instances. 166Note that MatNet is limited to problem scales no larger than 256 due to its one-hot initialization while there is no such limitation for GLOP-empowered MatNet. 167167167167





**Table 4: Comparison results on ATSP.** 168



| **Method** | **MatNet**  | **GLOP**    |
| ---------- | ----------- | ----------- |
| ATSP150    | 2.88 (7.2s) | 1.89 (8.2s) |
| ATSP250    | 4.49 (12s)  | 2.04 (9.3s) |
| ATSP1000   | -           | 2.33 (15s)  |

### 6.3 Capacitated Vehicle Routing Problem

GLOP for CVRP involves node clustering with our global policy, followed by solving the produced sub-TSPs with our sub-TSP solver. 169





**Large-scale CVRP**: Table 6 summarizes the results of the comparison in large-scale CVRP. 170Here, we apply the global policy trained on CVRP2K to both CVRP5K and CVRP7K, verifying the generalization performance of GLOP. 171Compared to the methods that entail iterative solution refinement, both TAM and GLOP can deliver more real-time solutions. 172Compared with prior SOTA real-time solver TAM, GLOP learns more effective global/local policies and enables higher decoding efficiency. 173Hence, GLOP outperforms TAM regarding both solution quality and efficiency. 174





**Table 6: Comparison results on large-scale CVRP following the settings in (Hou et al. 2023).** 175



| **Method**     | **CVRP1K Obj. / Time(s)** | **CVRP2K Obj. / Time(s)** | **CVRP5K Obj. / Time(s)** | **CVRP7K Obj. / Time(s)** |
| -------------- | ------------------------- | ------------------------- | ------------------------- | ------------------------- |
| LKH-3          | 46.4 / 6.2                | 64.9 / 20                 | 175.7 / 152               | 245.0 / 501               |
| AM             | 61.4 / 0.6                | 114.4 / 1.9               | 257.1 / 12                | 354.3 / 26                |
| L2I            | 93.2 / 6.3                | 138.8 / 25                | 172.2 / 12                | 233.4 / 26                |
| NLNS           | 53.5 / 198                | 65.2 / 38                 | 144.6 / 17                | 196.9 / 33                |
| L2D            | 46.3 / 1.5                | 137.6 / 42                | 142.8 / 30                | 193.6 / 52                |
| RBG            | 74.0 / 13                 | 74.3 / 2.2                | -                         | -                         |
| TAM-AM         | 50.1 / 0.8                | 64.8 / 1.8                | -                         | -                         |
| TAM-LKH3       | 46.3 / 5.6                | -                         | -                         | -                         |
| TAM-HGS        | -                         | -                         | -                         | -                         |
| GLOP-G         | 47.1 / 0.4                | 63.5 / 1.2                | 141.9 / 1.7               | 191.7 / 2.4               |
| GLOP-G (LKH-3) | 45.9 / 1.1                | 63.0 / 1.5                | 140.6 / 4.0               | 191.2 / 5.8               |



**Real-world CVRPLIB**: We test GLOP on large-scale CVRPLIB instances and present results in Table 7. We generalize the model trained on CVRP2000 to these large instances and compare GLOP to prior SOTA real-time solver TAM (Hou et al. 2023). 176The comparison also demonstrates the superiority of GLOP in both solution quality and efficiency, especially for very large instances. 177





**Table 7: Comparison results on large-scale CVRPLIB instances.** 178



| **Instance** | **Scale** | **AM Gap/Time** | **TAM-AM Gap/Time** | **LKH-3 Gap/Time** | **TAM-LKH3 Gap/Time** | **GLOP Gap/Time** | **GLOP-LKH3 Gap/Time** |
| ------------ | --------- | --------------- | ------------------- | ------------------ | --------------------- | ----------------- | ---------------------- |
| LEUVEN1      | 3001      | 46.9/10s        | 20.2/10s            | 18.1/69s           | 19.3/16s              | 16.9/2s           | 16.6/8s                |
| LEUVEN2      | 4001      | 53.3/13s        | 38.6/14s            | 22.1/74s           | 15.9/24s              | 21.8/3s           | 21.1/3s                |
| ANTWERP1     | 6001      | 39.3/13s        | 24.9/13s            | 24.2/596s          | 24.0/25s              | 20.3/3s           | 19.3/14s               |
| ANTWERP2     | 7001      | 50.3/15s        | 33.2/15s            | 31.1/479s          | 22.6/32s              | 19.4/4s           | 19.4/7s                |
| GHENT1       | 10001     | 46.9/21s        | 30.2/22s            | 29.5/37s           | 20.3/5s               | 18.3/22s          | -                      |
| GHENT2       | 11001     | 52.2/39s        | 33.3/38s            | 23.7/56s           | 19.8/6s               | 18.1/8s           | -                      |
| BRUSSELS1    | 15001     | 52.4/131s       | 43.4/139s           | 27.2/167s          | 27.6/8s               | 27.5/26s          | -                      |
| BRUSSELS2    | 16001     | 52.4/166s       | 39.0/159s           | 37.1/187s          | 22.4/9s               | 20.1/14s          | -                      |

### 6.4 Prize Collecting Travelling Salesman Problem

GLOP for PCTSP involves node subsetting with our global policies, followed by solving a sub-TSP instance with our sub-TSP solvers. 179Large-scale PCTSP entails both a scalable global policy to generate a promising node partition and a scalable local policy to tackle the equivalently large sub-TSP. 180We evaluate greedy (GLOP-G) and sampling (GLOP-S) decoding for our global policy. 181The comparison results on large-scale PCTSP are displayed in Table 8, where GLOP surpasses recent neural solvers and conventional solvers in terms of both solution quality and efficiency. 182





**Table 8: Comparison results of GLOP and the baselines on 128 PCTSP500, 1K, and 5K.** 183



| **Method**                 | **PCTSP500 Obj./Time** | **PCTSP1K Obj./Time** | **PCTSP5K Obj./Time** |
| -------------------------- | ---------------------- | --------------------- | --------------------- |
| OR Tools                   | 15.0 / 1h              | 24.9 / 1h             | 63.3 / 7h             |
| OR Tools (more iterations) | 14.4 / 16h             | 20.6 / 16h            | 54.4 / 16h            |
| AM (2019)                  | 19.3 / 14m             | 34.8 / 23m            | 175 / 21m             |
| MDAM (2021)                | 14.8 / 2.8m            | 22.2 / 17m            | 58.9 / 3h             |
| GLOP-G                     | 14.6 / 26s             | 20.0 / 47s            | 46.0 / 3.7m           |
| GLOP-S                     | 14.3 / 1.5m            | 19.8 / 2.5m           | 44.9 / 16m            |

------

## 7 Conclusion and limitation

This paper proposes GLOP to learn global policies for coarse-grained problem partitioning and local policies for fine-grained route construction. 184GLOP leverages the scalability of the NAR paradigm and meticulousness of the AR paradigm, making the first effective attempt at hybridizing them. 185Extensive evaluations on large-scale TSP, ATSP, CVRP, and PCTSP demonstrate its competitive and SOTA real-time performance. 186However, GLOP might be less competitive in application scenarios where prolonged execution time is allowed. 187In terms of its ability to trade off execution time for solution quality, GLOP might be inferior to the methods based on iterative solution refinement (further discussed in Appendix E.1). 188Our future focus will be on addressing this limitation. 189In addition, we plan to investigate the emerging possibilities that arise when viewing AR and NAR methods from a unified perspective. 190We believe it is also promising to exploit unsupervised Deep Graph Clustering techniques or to formulate node classification tasks to solve large-scale routing problems hierarchically. 191



------

## Acknowledgments

The authors appreciate the helpful discussions with Juntao Li, Yu Hong, and anonymous reviewers. 192Yu Hu, Jiusi Yin, and Tao Yu also contributed to this work. 193This research was supported by the National Natural Science Foundation of China (NSFC); the National Key R&D Program of China; the Singapore Ministry of Education (MOE) Academic Research Fund (AcRF) Tier 1 grant; the Undergraduate Training Program for Innovation and Entrepreneurship, Soochow University; the Priority Academic Program Development of Jiangsu Higher Education Institutions, China; and Provincial Key Laboratory for Computer Information Processing Technology, Soochow University. 194



------

## A Details of GLOP

### A.1 Pseudo code of solving (sub-)TSP

We present the pseudo code of solving (sub-)TSP in Algorithm 1. Note that we can generate multiple initial tours and pick the best one after all revisions. 195In line 12, the strategy for shifting the decomposition point is to minimize the subtour overlap between revisions. 196





**Algorithm 1: Solving (sub-)TSP** 197



1. **Input**: A TSP instance *tsp* with $N$ nodes; trained Revisers and a list of their sizes $RS$ (e.g., {100, 50, 20}); the number of initial tours $W$; revision iterations $I_n, \forall n \in RS$
2. **Output**: The best solution $\pi^*$
3. Generate the initial tours: $\{\pi_1, ..., \pi_W\} \leftarrow$ RandomInsertion(*tsp*, $W$)
4. for $n \in RS$ do
5. Initialize the decomposition point: $p \leftarrow 0$
6. Calculate the number of subtours decomposed from a tour: $K_n = \lfloor\frac{N}{n}\rfloor$
7. for iter = $1 \rightarrow I_n$ do
8. subtours $\leftarrow$ Decompose tours into subtours based on $p$ and $n$.
9. Apply coordinate transformation and instance augmentation to subtours.
10. subtours $\leftarrow$ Reviser-$n$(subtours, *tsp*)
11. $\{\pi_1, ..., \pi_W\} \leftarrow$ Composition(subtours)
12. Shift the decomposition point: $p \leftarrow p + \max(1, \lfloor\frac{n}{I_n}\rfloor)$
13. end for
14. end for
15. $\pi^* =$ PickBest(*tsp*, $\{\pi_1, ..., \pi_W\}$)

### A.2 Coordinate transformation

Recall that we solve SHPPs with local policies parameterized by deep neural models, i.e., revisers, and the coordinates of a subtour are the inputs to a reviser. 198In the inference phase, we apply a coordinate transformation to improve the predictability and homogeneity of the model inputs. 199 It facilitates training and benefits inference performance. Let $(x'_i, y'_i)$ denote the coordinates of the $i$-th node after transformation; $x_{max}, x_{min}, y_{max}, \text{and } y_{min}$ denote the bounds of an SHPP graph. 200 Then the coordinate transformation is formulated as:



$$x'_i = sc(x_i - x_{min}), y'_i = sc(y_i - y_{min}) \quad \text{if } x_{max} - x_{min} > y_{max} - y_{min}$$



Otherwise coordinates are swapped and scaled. The scale coefficient $sc = \frac{1}{\max(x_{max}-x_{min}, y_{max}-y_{min})}$. 201With the above Min-max Normalization and an optional graph rotation, we scale the x-axis coordinates to [0, 1] and set the y-axis lower bound to 0. Nevertheless, the revisers need to handle inputs of varied y-axis upper bound. 202It motivates us to develop the multi-distribution curriculum. 203

### A.3 Curriculum learning



Stage 1: multi-distribution training: The first curriculum stage trains all revisers using multi-distribution SHPPs. 204For each training instance, we first sample a y-axis upper bound: $y_{max} \sim (0, 1]$, then the instance: $(x, y) \sim [0, 1] \times [0, y_{max}]$, both uniformly. 205

Stage 2: collaborative training: We first generate TSP instances with nodes sampled uniformly in [0, 1]², then decompose their insertion-generated TSP tours into SHPPs. These SHPPs are used to fine-tune the first reviser. Each fine-tuned reviser is applied to infer its training instances, and the output subtours are decomposed into smaller-scale SHPPs to fine-tune the next reviser. 206

### A.4 Details of our global policy

**Constraints** $\Theta$ in Eq. (3) requires specification for certain problems. For CVRP, we constrain $|\pi|$ to the maximum number of vehicles by using the mask function suggested by Hou et al. (2023) and prohibit revisiting the same nodes or exceeding vehicle capacity. For PCTSP, we set $|\pi|$ to 1 and prohibit revisiting the same nodes or violating the minimum prize constraint. 207





**Inputs**: For the CVRP input graph, the node features are the normalized demand and polar coordinates w.r.t. the depot; the edge attributes are the relative Euclidean distance and polar angle. 208We sparsify the input graph by restricting each node to only connect with its $k$ nearest neighbors based on their polar angles. 209$k$ is set to 100 for CVRP1K and 200 for the rest. 210 For the PCTSP input graph, the node features are the normalized prize and penalty; the edge attribute is the relative Euclidean distance. We sparsify the graph based on the Euclidean distance, setting $k$ to 50, 100, and 200 for PCTSP500, 1K, and 5K, respectively. 211



### A.5 Model architecture and training settings



**Local policy**: We deploy revisers of four scales: Reviser-100, 50, 20, and 10. The hyperparameter configurations of our revisers are the same as (Kool, van Hoof, and Welling 2019), except that we use 6 encoder layers instead of 3. We apply Adam optimizer. 212The 1st-stage curriculum trains Reviser-10 for 100 epochs, while others for 200 epochs. 213The 2nd-stage curriculum trains all revisers for 300 epochs. 214**Global policy**: We employ the same model architecture as (Qiu, Sun, and Yang 2022), except that we set the dimension of node embedding to 48 on CVRP. 215We apply AdamW optimizer. 216



### A.6 Inference settings



TSP: The used hyperparameters are gathered in Table 9. 217

CVRP: Our global policies implement greedy decoding for CVRP. The hyperparameter settings for the local policies are displayed in Table 10. 218

PCTSP: On PCTSP, our local policies apply both greedy and sampling decoding. For the latter, the number of sampled partitions is set to 10. 219



**Table 9: Hyperparameter settings used for TSP inference.** 220



|         | **GLOP for TSP** |      |      |      |      | **GLOP (more revisions)** |      |      |      |      |
| ------- | ---------------- | ---- | ---- | ---- | ---- | ------------------------- | ---- | ---- | ---- | ---- |
|         | W                | I10  | I20  | I50  | I100 | W                         | I10  | I20  | I50  | I100 |
| TSP100  | 35               | 5    | 10   | 10   | 20   | 140                       | 5    | 10   | 10   | 20   |
| TSP500  | 1                | -    | 5    | 25   | 20   | 10                        | -    | 5    | 25   | 20   |
| TSP1K   | 1                | -    | 5    | 25   | 20   | 10                        | -    | 5    | 25   | 20   |
| TSP10K  | 1                | -    | 5    | 20   | 10   | 1                         | -    | 5    | 25   | 50   |
| TSP100K | 1                | -    | 5    | 5    | 5    | 1                         | -    | 5    | 25   | 50   |



**Table 10: Hyperparameter settings used for CVRP inference.** 221



|        | **W** | **I20** | **I50** |
| ------ | ----- | ------- | ------- |
| CVRP1K | 1     | 5       | -       |
| CVRP2K | 1     | 5       | 5       |
| CVRP5K | 1     | 5       | -       |
| CVRP7K | 1     | 5       | -       |

------

## B Discussing two paradigms

From a unified perspective, both autoregressive (AR) and non-autoregressive (NAR) paradigms can be viewed as learning a probabilistic construction graph. 222The AR heuristics usually learn "fine-grained" construction graphs in the sense that per-step construction is heavily conditioned on the obtained partial solution. 223However, it entails accurate context representation for effective actions and costly step-by-step neural decoding. 224The NAR heuristics usually learn "coarse-grained" construction graphs only conditioned on the input problem instance. 225It enables construction graph (heatmap) generations in one shot. 226While being much more scalable, they underperform with vanilla sampling-based decoding due to the paucity of rich decoding context. 227In this sense, those NAR heuristics need to incorporate additional solution refinements to trade off execution time for higher solution quality, making them less suitable for real-time applications. 228



------

## C Additional related work

The end-to-end NCO solvers can be categorized into two paradigms, i.e., autoregressive (AR) solution construction and non-autoregressive (NAR) heatmap generation coupled with subsequent decoding. 229AR heuristics allow the neural models to output the assignments to decision variables sequentially. 230Learning global AR methods suffers from poor scaling-up generalization. 231By contrast, NAR heuristics typically apply GNNs in one shot to output a heatmap. 232Despite recent success in scaling NAR methods to large-scale problems, they underperform with vanilla sampling-based decoding and have to be coupled with iterative solution refinement such as Monte Carlo Tree Search (MCTS), falling short of real-time needs. 233



To the best of our knowledge, GLOP is the first neural solver to effectively integrate both AR and NAR components. 234Regarding TSP solution quality, GLOP is not in competition with the methods equipped with sophisticated and iterative improvement operators such as MCTS and LKH-3. 235Instead, it is a complement for scenarios requiring highly real-time solutions. 236Moreover, GLOP performs SOTA consistency across TSP scales and distributions. 237



------

## D Problem definitions



**TSP**: Given a set of cities and the distances between each pair of cities, the objective of the Traveling Salesman Problem is to find the shortest possible tour that visits each city exactly once and returns to the original city. 238**SHPP**: The Shortest Hamiltonian Path Problem (SHPP) is also referred to as the open-loop Traveling Salesman Problem (TSP). 239In the SHPP, the goal is to find the shortest Hamiltonian path in a given graph. 240**CVRP**: In Capacitated Vehicle Routing Problem (CVRP), a set of customers, each with a specific demand for goods, must be serviced by a fleet of vehicles originating from a central depot. 241CVRP determines the optimal sequence in which all customers are visited and serviced, while ensuring that the sum of their demands along each route does not exceed the vehicle capacity. 242**PCTSP**: The setup of Prize Collecting Traveling Salesman Problem (PCTSP) follows that in (Kool, van Hoof, and Welling 2019). 243PCTSP aims to minimize the total tour length plus penalties for unvisited nodes, while ensuring collecting a minimum total prize. 244



------

## E Extended evaluations

### E.1 Further comparison with baselines using MCTS

GLOP compares favorably with baselines that implement MCTS (Fu, Qiu, and Zha 2021; Qiu, Sun, and Yang 2022) when instant solutions are desired. 245Specifically, GLOP can produce reasonable solutions before the baselines obtain heatmaps, while the latter can eventually outperform GLOP due to the continued improvements provided by MCTS. 246





**Figure 2**: Further comparison with two strong baselines that implement MCTS. 247The curves of GCN+MCTS and DIMES start when they finish heatmap generation and the first MCTS iteration. 248



### E.2 Ablation study



**Neural heuristics against LKH-3 in local policy**: We use neural heuristics as SHPP solvers for parallelizability and efficiency. 249 Our local policies involve solving many SHPP instances. Neural heuristics can solve thousands of SHPPs simultaneously by leveraging the parallel computing power of GPUs. 250We evaluate neural heuristics against LKH-3 in our local policy to validate this motivation. 251Implementing neural heuristics shows clear advantage in terms of solution efficiency. 252





**Table 11: Neural heuristics against LKH-3 as SHPP solver.** 253



|          | **w/ LKH-3 Obj. / Time** | **w/ neural heuristics (GLOP) Obj. / Time** |
| -------- | ------------------------ | ------------------------------------------- |
| TSP500   | 16.86 / 34m              | 17.07 / 19s (x107)                          |
| TSP1K    | 23.74 / 68m              | 24.01 / 34s (x120)                          |
| TSP10K   | 74.64 / 50m              | 75.62 / 32s (x95)                           |
| PCTSP500 | 14.5 / 17m               | 14.6 / 26s (x39)                            |
| PCTSP1K  | 19.8 / 25m               | 20.0 / 47s (x32)                            |
| PCTSP5K  | 45.7 / 1.7h              | 46.0 / 3.7m (x27)                           |



**Components in local policy**: We conduct ablation studies of the components in our inference pipeline for (sub-)TSP. 254The results in Table 12 validate the design of each component. 255





**Table 12: Ablation studies of our local policy.** 256



| **Components** | **TSP500** | **TSP1K** | **BD** | **CT** | **CL1** | **CL2** |
| -------------- | ---------- | --------- | ------ | ------ | ------- | ------- |
| x              | 16.88      | 23.85     |        |        |         |         |
| x              | 16.92      | 24.17     |        |        |         |         |
| x              | 16.91      | 23.89     |        |        |         |         |
| x              | 16.94      | 23.95     |        |        |         |         |
| GLOP           | 16.80      | 23.73     |        |        |         |         |



**Global policy**: On CVRP, we demonstrate the advantage of our NAR global policy over AR dividing policy by comparing GLOP-G (LKH3) with TAM-LKH3 in Table 6. On PCTSP, Table 13 further verifies the importance of our global policy by comparing it with random partition. 257





**Table 13: Ablation studies of our global policy.** 258



| **Problem** | **GLOP-RP** | **GLOP-G** |
| ----------- | ----------- | ---------- |
| PCTSP500    | 19.0        | 14.6       |
| PCTSP1K     | 26.1        | 20.0       |
| PCTSP5K     | 53.0        | 46.0       |

### E.3 Discussing cross-distribution performance of local policies



**Training schemes**: In Table 14, we investigate four training schemes. 259Compared with the cross-distribution baseline, GLOP shows lower Det. under all settings. 260It verifies that the holistic solution process of dividing and conquering is the main contributor to the cross-distribution performance. 261





**Table 14: The impact of different training schemes on the cross-distribution performance of GLOP.** 262



| **Method**     | **Uniform Gap(%)** | **Expansion Gap(%) Det.(%)** | **Explosion Gap(%) Det.(%)** | **Implosion Gap(%) Det.(%)** |
| -------------- | ------------------ | ---------------------------- | ---------------------------- | ---------------------------- |
| AM+HAC         | 2.484              | 3.997 / 61                   | 3.084 / 24                   | 2.595 / 4.5                  |
| GLOP (Uniform) | 0.215              | 0.235 / 9.3                  | 0.119 / -44                  | 0.197 / -8.4                 |
| GLOP (C1)      | 0.183              | 0.189 / 3.3                  | 0.089 / -51                  | 0.162 / -11                  |
| GLOP (C2)      | 0.180              | 0.260 / 44                   | 0.105 / -42                  | 0.159 / -12                  |
| GLOP (C1+C2)   | 0.091              | 0.166 / 82                   | 0.066 / -27                  | 0.082 / -9.9                 |



**More revisers and more revisions**: Table 15 demonstrates that more revisers or more revisions can effectively reduce "Gap" but have less impact on "Det.". 263We further conclude that the holistic solution process of dividing and conquering is the main contributor to the cross-distribution performance. 264





**Table 15: The impact of more revisers or more revisions on the cross-distribution performance of GLOP.** 265



| **Method**     | **Uniform Gap(%)** | **Expansion Gap(%) Det.(%)** | **Explosion Gap(%) Det.(%)** | **Implosion Gap(%) Det.(%)** |
| -------------- | ------------------ | ---------------------------- | ---------------------------- | ---------------------------- |
| AM+HAC         | 2.484              | 3.997 / 61                   | 3.084 / 24                   | 2.595 / 4.5                  |
| GLOP (RS={50}) | 0.380              | 0.740 / 94                   | 0.285 / -25                  | 0.362 / -4.8                 |
| GLOP (In=1)    | 0.738              | 1.194 / 62                   | 0.621 / -16                  | 0.693 / -6.1                 |
| GLOP           | 0.091              | 0.166 / 82                   | 0.066 / -27                  | 0.082 / -9.9                 |

### E.4 Stability analysis of GLOP

Figure 3 showcases the stability of GLOP on TSP500 and TSP1K, illustrating the box plots of the objective values for 10 independent runs. 266The results suggest that implementing more revisions effectively guarantees better stability. 267



### E.5 Discussing time complexity

In GLOP, a revision decomposes a TSP into a batch of SHPPs and exploits NN to solve them simultaneously. 268Since the scale of SHPP is a constant for an arbitrary TSP, a revision enjoys linear time complexity. 269Table 16 presents the time needed for solving a single instance of different scales but under identical settings. 270It shows a near-constant empirical complexity when the problem scale is less than 10K and the time of revisions dominates. 271





**Table 16: Time needed for solving a single instance of different scales.** 272



| **Scale** | **TSP200** | **TSP500** | **TSP1K** | **TSP2K** | **TSP5K** | **TSP10K** |
| --------- | ---------- | ---------- | --------- | --------- | --------- | ---------- |
| Time (s)  | 6.3        | 6.3        | 6.9       | 7.3       | 7.8       | 8.6        |

------

## F Implementation details of baselines

Our experiments involve the following neural baselines:

- 

  **TSP**: AM, LCP, GCN+MCTS, POMO-EAS, DIMES, Tspformer, H-TSP, Pointerformer, DACT, AMDKD+EAS, AM+HAC, and MatNet. 273

  

  

- 

  **CVRP**: AM, L2I, NLNS, L2D, RBG, and TAM. 274

  

  

- 

  **PCTSP**: AM and MDAM. 275

  

  

Most baselines are reproduced by strictly following their open-sourced implementation. 276However, some others have to be tuned and adapted for proper and comparable evaluations. 277



- 

  **Non-learning baselines**: Most results are reproduced following Kool, van Hoof, and Welling (2019). 278

  

  

- 

  **AM**: The checkpoints trained on TSP/PCTSP100 are generalized for comparison. 279

  

  

- 

  **POMO-EAS**: We implement EAS-Tab using the POMO checkpoint trained on TSP100 and set T = 60. 280

  

  

- 

  **DACT**: For the TSPLIB benchmarks, we apply the model trained on TSP20, TSP50, and TSP100 appropriately. 281

  

  

- **GCN+MCTS**: We implement the CPU-version MCTS. The parameter T is set smaller for a fair comparison. 282

  

  

- 

  **DIMES**: We follow the original implementations except adjusting the MCTS execution time T. 283

  

  

- 

  **AM+HAC**: We fine-tune the AM trained on TSP100 with 100 epochs of Hardness-adaptive Curriculum (HAC). 284

  

  

- 

  **AMDKD+EAS**: We set the EAS iterations T to 100. 285

  

  

- 

  **L2D and RBG**: Their provided checkpoints are directly adapted for the evaluation in this work. 286

  

  

------

## G Discussing connections with non-learning OR methods

NCO solvers show promise due to their parallelizability and data-driven nature in design automation, and they are gradually closing the gap with traditional OR methods. 287The idea of solving routing problems hierarchically has long been utilized in non-learning OR methods. 288A seminal work of this kind is POPMUSIC for TSP. 289Our (Sub-)TSP solver shares the ideology of POPMUSIC. 290Nevertheless, we discuss that NCO solvers and POPMUSIC have complementary natures. 291Employing the sophisticated initialization of POPMUSIC can improve the performance of NCO solvers. 292On the other hand, if replacing LKH with NCO solvers, POPMUSIC can gain from the high parallelizability of NCO at little cost of solution quality. 293