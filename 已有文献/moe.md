Published as a conference paper at ICLR 2026 
# COMBINATION-OF-EXPERTS WITH KNOWLEDGE SHARING FOR CROSS-TASK VEHICLE ROUTING PROB-LEMS
Zikang $\mathbf { Y u } ^ { 1 }$ , Jinbiao Chen2 , Jiahai Wang1 
1School of Computer Science and Engineering, Sun Yat-sen University, P.R. China 2Department of Industrial Systems Engineering and Management, National University of Singapore yuzk6@mail2.sysu.edu.cn, bill.cjb@nus.edu.sg, wangjiah@mail.sysu.edu.cn 
# ABSTRACT
Recent neural methods have shown promise in generalizing across various vehicle routing problems (VRPs). These methods adopt either a fully-shared dense model across all VRP tasks (i.e., variants) or a mixture-of-experts model that assigns node embeddings within each task instance to different experts. However, they both struggle to generalize from training tasks with basic constraints to out-ofdistribution (OOD) tasks involving unseen constraint combinations and new basic constraints, as they overlook the fact that each VRP task is defined by a combination of multiple basic constraints. To address this, this paper proposes a novel model, combination-of-experts with knowledge sharing (CoEKS), which leverages the structural characteristic of VRP tasks. CoEKS enhances generalization to constraint combinations via two complementary components: a combinationof-experts architecture enabling flexible combinations via prior assignment of constraint-specific experts, and a knowledge sharing strategy strengthening generalization via automatic learning of transferable general knowledge across constraints. Moreover, CoEKS allows new experts to be plugged into the trained model for rapid adaptation to new constraints. Experiments demonstrate that Co-EKS outperforms state-of-the-art methods on in-distribution tasks and delivers greater gains on OOD tasks, including unseen constraint combinations (relative improvement of $12 \%$ over SOTA) and new constraints $2 5 \%$ improvement). 
# 1 INTRODUCTION
Combinatorial Optimization (CO) plays a pivotal role in numerous real-world applications, such as logistics (Zong et al., 2022), transportation (Fu et al., 2025), supply chain management (Tirkolaee et al., 2020), and resource allocation (Heydaribeni et al., 2024). The vehicle routing problem (VRP) stands as one of the most fundamental yet challenging CO problems, requiring the determination of optimal routes for a vehicle fleet serving a set of customers while satisfying multiple operational constraints. Despite decades of algorithmic progress, traditional methods face significant limitations: exact approaches are computationally infeasible for large-scale instances due to the NP-hard nature of VRP (Wu et al., 2024), while heuristic approaches heavily rely on handcrafted expert knowledge and time-consuming iterative search from scratch for each new instance (Bogyrbayeva et al., 2024). 
Recent advances in deep learning have introduced neural methods for VRPs, which autonomously learn heuristic policies from massive data end-to-end. These methods not only circumvent the dependency on expert domain knowledge but also produce high-quality solutions within short solving time (Chen et al., 2025a; Li et al., 2025b; Goh et al., 2024; Chen et al., 2023b; Zhang et al., 2023; Liao et al., 2025; Fang et al., 2026; Xiao et al., 2024). However, despite the promising performance, most existing methods adopt a task-specific learning paradigm, necessitating a separate neural model for each VRP task. This lack of cross-task generalization capability incurs costly retraining and deployment overhead when adapting to new tasks. 
?Corresponding author. 
1 
Published as a conference paper at ICLR 2026 
More recently, some efforts have focused on developing unified models for cross-task VRPs. Despite demonstrating feasibility, current methods still perform suboptimally, especially under out-ofdistribution (OOD) generalization scenarios: 1) tasks with unseen combinations of basic constraints, and 2) tasks involving new basic constraints. According to their model architectures, they fall into two categories: a task-shared dense model and a node-level mixture-of-experts (MoE) model. The task-shared dense model (Liu et al., 2024b; Berto et al., 2024; Li et al., 2025a), whose parameters are fully-shared across all tasks, overemphasizes coupled representations but neglects task-specific ones, resulting in negative transfer among tasks. This leads to particularly poor OOD generalization. As an alternative, the node-level MoE model (Zhou et al., 2024a; Huang et al., 2025) employs a gating mechanism to assign each node embedding within a task instance to different experts, fostering node-specialized experts. However, this gating mechanism restricts expert vision to a narrow node subset, which weakens experts’ cognition of task-level knowledge. 
Unlike general multi-task learning, we observe that VRP tasks often involve combinations of multiple basic constraints. This motivates us to develop an innovative model architecture, called combination-of-experts with knowledge sharing (CoEKS). On the one hand, CoEKS utilizes constraint-specific experts to facilitate learning dedicated knowledge for every basic constraint. This enables flexible combinations of experts to manage diverse VRP tasks with constraint combinations and allows further plugging in of new experts to adapt even to unseen basic constraints. On the other hand, CoEKS transfers general knowledge across constraints to foster collaboration among experts and enhance OOD generalization. 
The contributions of this paper can be summarized as follows: 1) We introduce a novel CoEKS model, building on the recognition of the prior structural characteristics of VRPs. This model is designed to enhance OOD generalization for VRPs with constraint combinations via two complementary components and adapts to unseen constraints by plugging in new experts. 2) We design a combination-of-experts (CoE) architecture to acquire specialized constraint-level knowledge, in which each expert specializes in a basic constraint, enabling the model to effectively solve diverse VRPs by flexibly combining corresponding experts. 3) We propose a multi-view knowledge sharing strategy to transfer general knowledge across constraints, utilizing mutual distillation and shared transformation layers to automatically learn coordination among experts. 4) We demonstrate that CoEKS can be deployed on both state-of-the-art (SOTA) and classic backbones to show its universality. Extensive experiments indicate that CoEKS outperforms SOTA cross-task neural methods for in-distribution (ID) and particularly OOD generalization. 
# 2 RELATED WORKS
Neural combinatorial optimization for VRPs. Existing neural methods for VRPs can be primarily categorized into two distinct groups: 1) Neural construction methods leverage deep neural networks to generate feasible solutions in an end-to-end manner. Some works (Vinyals et al., 2015; Bello et al., 2017; Nazari et al., 2018) pioneer this direction to address the Traveling Salesman Problem (TSP) and VRP. Attention Model (Kool et al., 2019), which combines Transformer with reinforcement learning, is regarded as a milestone. Policy Optimization with Multiple Optima (POMO) (Kwon et al., 2020) leverages solution symmetry to improve policy learning and has become a typical backbone model for many subsequent extensions (Bi et al., 2024; Chen et al., 2025b; 2023a; Zhou et al., 2024b; Hou et al., 2023; Fang et al., 2024) due to its prominent performance and flexibility. Recently, RELD (Huang et al., 2025) has emerged as the SOTA backbone model by incorporating a feedforward neural network (FFN) and identity mapping (IDT) into the decoder. 2) Neural improvement methods employ neural networks to replace handcrafted rules, iteratively refining an initial solution to meet specified requirements (Wu et al., 2021; Ma et al., 2021; Luo et al., 2025; Ma et al., 2023). Although such methods often yield superior solutions, their computational overhead is considerable. Consequently, this paper focuses on neural construction methods. 
Some subsequent works enhance the generalization of neural construction methods across problem sizes (Luo et al., 2023; Pan et al., 2025), node distributions (Bi et al., 2022; Liu et al., 2024a) and their interactions (Manchanda et al., 2022; Wang et al., 2024). Our work targets a more challenging and underexplored scenario: generalization across diverse VRP tasks. 
Cross-task generalization for VRPs. Recent studies (Drakulic et al., 2024; Wang & Yu, 2023; Jiang et al., 2024b) investigate a unified model for cross-task CO problems, but focus only on ID 
Published as a conference paper at ICLR 2026 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/d3ed2dc24d8680bace224e603feb8c470461d4554a7818661bc7aa6dfcecee4d.jpg)


Figure 1: Illustrations of feasible solutions with various constraints.

generalization. In addition, Lin et al. (2024) develop fine-tuning adapters on a pre-trained TSP model for new tasks. These methods fall short of zero-shot OOD generalization across diverse VRPs. 
Existing approaches to zero-shot OOD generalization for diverse VRPs with constraint combinations can be divided into two categories. One type, such as POMO-MTL (Liu et al., 2024b), RouteFinder (Berto et al., 2024), and CaDA (Li et al., 2025a), adopts a task-shared dense model, which overemphasizes coupled representations at the expense of task-specific ones, leading to negative transfer among tasks. The other type employs a node-level MoE model to route each node embedding to different experts, like MVMoE (Zhou et al., 2024a). ReLD-MoEL (Huang et al., 2025) integrates MVMoE with the ReLD backbone, achieving the SOTA performance on OOD generalization across VRP tasks. Nevertheless, these node-level MoE models suffer from narrow expert vision limited to a node subset, which deteriorates task-level generalization (Further comparisons with the MoE models are provided in Appendix E.1). In summary, existing models overlook the structural characteristic of VRPs, which motivates us to propose a novel architecture, CoEKS. Along another orthogonal lines of research, Liu et al. (2025) propose a pre-training paradigm for VRPs to improve generalization and Goh et al. (2025) study the multi-task multi-distribution VRPs using a mixture-of-depths architecture with clustering. 
# 3 PRELIMINARIES
# 3.1 VEHICLE ROUTING PROBLEMS
Formally, VRP is modeled on a complete graph $G = ( V , E )$ , where $V = V _ { 0 } \cup V _ { c }$ includes a depot node $V _ { 0 }$ and customer nodes $V _ { c } = \{ v _ { 1 } , \ldots , v _ { n } \}$ . Each node $v _ { i } \in V$ is associated with 2D coordinate $x _ { i } \in [ 0 , 1 ] ^ { 2 }$ . Edge set $E$ connects all node pairs, with each edge $( i , j ) \in E$ associated with cost $c _ { i j }$ , measured by the Euclidean distance. Each customer node $v _ { i } \in V _ { c }$ has a non-negative demand $d _ { i } ~ \geq ~ 0$ , while the depot has zero demand. A VRP solution $\tau$ consists of a set of routes, each executed by a vehicle, that collectively visit all customers exactly once while satisfying task-specific constraints. The objective is to minimize total cost: $\mathrm { m i n } _ { \tau \in \Phi } c ( \tau )$ , where $\begin{array} { r } { c ( \tau ) = \sum _ { r \in \tau } \bar { \sum } _ { ( i , j ) \in r } c _ { i j } } \end{array}$ $\Phi$ denotes the set of feasible solutions, and $r$ represents a route assigned to a vehicle. 
VRP tasks are defined by applying different sets of practical constraints (see Figure 1) to reflect diverse real-world operational requirements. This paper focuses on six basic constraints from recent studies (Berto et al., 2024; Zhou et al., 2024a). 1) Capacity (C): Each vehicle has a maximum capacity $Q$ , i.e. the total demand of customers along any route must not exceed $Q$ . 2) Open Route (O): Vehicles are allowed to end their routes at the last customer instead of returning to the depot. 3) Backhaul (B): Customers are divided into linehaul nodes that require goods from the depot (delivery demand $d _ { i }$ ) and backhaul nodes that need goods to return to the depot (pickup demand $p _ { i } )$ . Vehicles serve both on a single route, but all linehaul deliveries must precede backhaul pickups. 4) Duration Limit (L): Each route is subject to a maximum duration or distance limit $L$ , ensuring balanced workloads and operational feasibility. 5) Time Window (TW): Each customer $n _ { i }$ has a time window $[ e _ { i } , l _ { i } ]$ and service duration $s _ { i }$ . Service must start within $[ e _ { i } , l _ { i } ]$ . Early arrivals must wait, and service is not allowed after $l _ { i }$ . Additionally, all vehicles must return to the depot before a global time limit $T _ { \mathrm { m a x } }$ . 6) Mixed Backhauls (MB): Unlike backhaul, MB relaxes the strict linehaulbefore-backhaul priority, allowing flexible sequences. 
Each basic constraint can exist individually or in combination, resulting in a rich set of VRP tasks. Their interplay introduces significant task diversity, requiring flexible and generalizable methods. More details on the VRP configurations and data generation process are provided in Appendix A. 
3 
Published as a conference paper at ICLR 2026 
# 3.2 NEURAL CONSTRUCTION METHODS
Neural construction methods represent a cutting-edge approach to solving VRPs, using deep reinforcement learning to construct solutions in an autoregressive manner, eliminating the need for precomputed labels or handcrafted heuristics. Typically, such methods adopt a deep neural network $\theta$ with an encoder-decoder architecture to parameterize a stochastic policy. The encoder processes static VRP features (e.g., node coordinates and demands) to produce node embeddings. At each step, the decoder integrates these embeddings with dynamic context (e.g., remaining vehicle capacity, current route length) to output a probability distribution over unvisited nodes, from which the next node is sampled. This process repeats until all customers have been visited, forming a complete solution. The solution construction is modeled as a Markov Decision Process, where state consists of the instance and current partial solution, and the action comprises the set of selectable nodes. Given a graph $G$ , the policy network $\theta$ specifies the probability of a solution $\tau$ , expressed autoregressively as $\begin{array} { r } { \dot { \mathbf { \rho } } _ { p _ { \theta } \left( \tau \vert G \right) } = \prod _ { t = 1 } ^ { \bar { T } } p _ { \theta } ( a _ { t } \vert s _ { t } ) } \end{array}$ , where $a _ { t }$ and $s _ { t }$ are the action and state at step $t$ , respectively. $T$ is the total number of decoding steps. The reward is defined as the negative cost of tour $\tau$ , i.e., $r ( \tau ) = - c ( \tau )$ . The task loss ${ \mathcal { L } } _ { p }$ is defined as the expected total cost. The policy is optimized via REINFORCE with a shared baseline $b ( G )$ , defined as the average reward over multiple sampled trajectories per instance (Kwon et al., 2020). The policy gradient is estimated as: 
$$
\nabla_ {\theta} \mathcal {L} _ {p} (\theta | G) = \mathbb {E} _ {p _ {\theta} (\tau | G)} \left[ (r (\tau) - b (G)) \nabla_ {\theta} \log p _ {\theta} (\tau | G) \right]. \tag {1}
$$
# 4 METHODOLOGY
This section presents the proposed combination-of-experts with knowledge sharing (CoEKS), which is tailored for cross-task generalization for VRPs with basic constraints and their combinations. The overall model structure is shown in Figure 2, where CoEKS is employed in the encoder (see Appendix B for details). CoEKS addresses this challenge through two complementary components: 1) a combination-of-experts (CoE) model that learns specialized knowledge and enables adaptive combinations of constraint-specific experts to handle diverse VRP tasks; and 2) a multi-view knowledge sharing strategy that enhances the model’s learning of transferable general knowledge across different constraints, thereby improving cross-task generalization for VRPs. 
# 4.1 COE MODEL
The CoE model extends the transformer-based architecture by introducing expert and combiner modules in the encoder. Each expert specializes in a basic constraint and adaptively aggregate their expertise through combiners, enabling efficient handling of VRPs with combinations of constraints. 
Constraint-specific expert. In a standard transformer block, the FFN processes node embeddings to capture complex relationships. CoEKS replace the FFN into a pool of constraint-specific experts (i.e., FFNs), where each expert (for $j \in \mathcal { E } = \mathsf { \bar { \{ C , O , B , L , T W \} } }$ specializes in a specific constraint: capacity $( E _ { C } )$ , open route $( E _ { O } )$ , backhaul $( E _ { B } )$ , duration limit $( E _ { L } )$ , and time window $( E _ { T W } )$ . For a given VRP instance with constraint set $C S \subseteq { \mathcal { E } }$ , only the corresponding experts are activated, with outputs defined as: 
$$
O _ {j} ^ {E} (h) = \left\{ \begin{array}{l l} E _ {j} (h), & \text {i f} j \in C S, \\ 0, & \text {o t h e r w i s e}, \end{array} \right. \tag {2}
$$
where $E _ { j } ( h ) = \mathrm { F F N } _ { j } ( h ) \in \mathbb { R } ^ { d }$ is the output of the $j$ -th expert, and $h \in \mathbb { R } ^ { d }$ is the input embedding. Since the capacity $C$ is a fundamental constraint underlying all VRP tasks, a shared expert mechanism (Dai et al., 2024) is employed, where the expert $E _ { C }$ is always activated as a shared expert corresponding to CVRP. This design ensures universal problem-solving capability while facilitating expert specialization. 
Combiner. To adaptively the combine outputs of activated experts, we introduce a combiner for each expert $E _ { j }$ , parameterized by $W _ { j } \in \mathbb { R } ^ { 1 \times d }$ . The raw values of activated combiners are $s _ { j } ( h ) = W _ { j }$ · h, $j \in \bar { C } S$ , and then weights corresponding to the experts are calculated by softmax normalization: 
$$
S _ {j} (h) = \frac {\exp \left(s _ {j} (h)\right)}{\sum_ {k \in C S} \exp \left(s _ {k} (h)\right)}, \quad j \in C S, \tag {3}
$$
4 
Published as a conference paper at ICLR 2026 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/f90f76a27e8e152c8406d0605919436119c48937645aaea9a0d61f122b2868db.jpg)


Figure 2: Workflow of the cross-task VRP method with CoEKS: Sampling an OVRP instance from the training set (gray parts indicate inactive constraints), the encoder generates node embeddings, and the decoder constructs a feasible solution. CoEKS output is determined by the activated experts and combiners, with mutual distillation (MDis) among the activated experts. The trained policy then generalizes to OOD tasks. Add represents residual connections and Norm denotes normalization.

with $S _ { k } ( h ) = 0$ for inactive experts $( k \notin C S )$ . Therefore, the final output of CoE can be obtained by a weighted combination of active experts: 
$$
O (h) = \sum_ {j \in C S} E _ {j} (h) \cdot S _ {j} (h). \tag {4}
$$
# 4.2 MULTI-VIEW KNOWLEDGE SHARING STRATEGY
The CoE model effectively specializes experts in distinct constraints, serving as a foundation for handling diverse VRP tasks. To complement this, a multi-view knowledge sharing strategy is proposed to enhance the model’s learning of transferable knowledge across different constraints, thereby improving OOD generalization. This strategy operates in two views: expert-view and combiner-view. 
Expert-view knowledge sharing. To broaden the expert vision and strengthen their comprehensive understanding of VRPs, we introduce mutual distillation (MDis), where active experts exchange knowledge to capture shared patterns across constraints. Unlike traditional knowledge distillation, which transfers knowledge from a teacher to a student model, MDis encourages peer-to-peer learning among experts (Xie et al., 2024). An auxiliary loss ${ \mathcal { L } } _ { m d }$ is incorporated to facilitate this process. The overall loss function is defined as follows, where ${ \mathcal { L } } _ { p }$ denotes the primary loss of the task, and $\alpha$ controls the distillation strength. 
$$
\mathcal {L} = \mathcal {L} _ {p} + \alpha \cdot \mathcal {L} _ {m d}, \tag {5}
$$
${ \mathcal { L } } _ { m d }$ is calculated as: 
$$
\mathcal {L} _ {m d} = \left\{ \begin{array}{l l} 0, & K = 1, \\ \operatorname {M S E} \left(E _ {1} (h), E _ {2} (h)\right), & K = 2, \\ \frac {1}{K} \sum_ {i = 1} ^ {K} \operatorname {M S E} \left(E _ {i} (h), E _ {\text {a v g}} (h)\right), & K > 2, \end{array} \right. \tag {6}
$$
where $K$ is the number of active experts, $E _ { i } ( h ) \in \mathbb { R } ^ { d }$ is the output of the $i$ -th expert, $E _ { \mathrm { a v g } } ( h ) =$ $\begin{array} { r } { \frac { 1 } { K } \sum _ { i = 1 } ^ { K } E _ { i } ( h ) } \end{array}$ is a virtual expert averaging active expert outputs, and MSE is the mean squared error. For $K = 1$ (i.e., CVRP with only $E _ { C }$ ), $\mathcal { L } _ { m d } = 0$ . The virtual expert simplifies computation for $K > 2$ by reducing the complexity from $O ( K ^ { 2 } )$ (pairwise comparisons) to $O ( K )$ , while still guiding experts toward a consensus by minimizing the variance of their outputs. 
To realize the trade-off between specialized and general knowledge, MDis is employed in the lower encoder layer (e.g., the first layer). This design choice is inspired by the property that lower layers in neural networks tend to capture general features, while higher layers focus on task-specific knowledge (Long et al., 2017). By localizing knowledge sharing to these early representations, our 
5 
Published as a conference paper at ICLR 2026 
method promotes the exchange of broadly useful information among experts without causing a homogenization of their expertise. The similarity of expert representations in the lower encoder layers is empirically validated via the t-SNE analysis in Appendix E.2. 
Combiner-view knowledge sharing. To further enhance generalization to unseen constraint combinations, we introduce a combiner-view knowledge sharing mechanism. Specifically, a shared transformation layer $f _ { s }$ is applied to the input embedding $h$ before it reaches the combiners introduced in Section 4.1. Therefore, $f _ { s }$ can inject cross-task knowledge for all combiners, enabling them to make informed weighting decisions across diverse VRP tasks. Given the importance of nonlinearity in modeling complex functions, we introduce nonlinearity to improve the representation of $f _ { s }$ . For simplicity, $f _ { s }$ is implemented as a low-rank multilayer perceptron (MLP) with a residual connection: 
$$
f _ {s} (h) = W _ {2} \cdot \operatorname {R e L U} \left(W _ {1} \cdot h\right) + h, \tag {7}
$$
where $W _ { 1 } \in \mathbb { R } ^ { d \times r }$ and $W _ { 2 } \in \mathbb { R } ^ { r \times d }$ are weight matrices forming a bottleneck structure with $r \ll d$ , enhancing parameter efficiency while preserving expressiveness. The final output of CoEKS is: 
$$
O _ {\mathrm {C o E K S}} (h) = \sum_ {j \in C S} E _ {j} (h) \cdot S _ {j} \left(f _ {s} (h)\right). \tag {8}
$$
# 4.3 INFERENCE FOR CONSTRAINT COMBINATIONS AND ADAPTATION TO NEW CONSTRAINTS
During inference, CoEKS addresses diverse VRPs by activating experts corresponding to specific constraint combinations, as illustrated in Figure 2. It achieves zero-shot OOD generalization for unseen combinations by flexibly combining constraint-specific experts. When encountering unseen basic constraints, new experts are plugged into the trained model and fine-tuned in isolation, with all existing parameters frozen to prevent catastrophic forgetting. This design maintains acquired knowledge and enables continuous rapid adaptation to new constraints, facilitating scalable deployment. 
# 5 EXPERIMENTS
In this section, extensive experiments are conducted on 48 VRP tasks. All experiments are carried out on an NVIDIA RTX 3090 GPU and an AMD Ryzen 5 3600. Our code and data are publicly available at https://github.com/yuzikang0/CoEKS. 
We aim to answer the following research questions: Q1. Does CoEKS achieve superior ID and OOD generalization for tasks with unseen constraint combinations? Q2. Can CoEKS show superior scalability in OOD tasks with new constraints? Q3. Is universal CoEKS consistently effective across different backbones? Q4. How effective is the knowledge sharing strategy in CoEKS? 
Baselines. 1) Traditional methods. Two heuristic solvers are employed in this study: the state-ofthe-art PyVRP (Wouda et al., 2024) and Google OR-Tools (Furnon & Perron, 2023). Both methods use a single CPU core to solve each instance. For node sizes $n = 5 0$ and $n = 1 0 0$ , the time limits are 10 and 20 seconds. 2) Neural methods. Recent representative cross-task VRP methods are considered, including POMO-MTL (Liu et al., 2024b), RF-TE (Berto et al., 2024), MVMoE (Zhou et al., 2024a), CaDA (Li et al., 2025a), and ReLD-MoEL (Huang et al., 2025). RF-TE and ReLD-MoEL are the strongest variants reported in RouteFinder (Berto et al., 2024) and ReLD (Huang et al., 2025). CoEKS is implemented on the SOTA ReLD backbone (see Appendix B for more details). 
Training. Our settings mostly follow RouteFinder (Berto et al., 2024). Each model is trained for 300 epochs, with each epoch containing 100K VRP instances. The Adam optimizer is used with a learning rate of $3 \times 1 0 ^ { - 4 }$ and batch sizes are set to 256 and 128 for $n = 5 0$ and $n = 1 0 0$ , respectively. The learning rate is multiplied by 0.1 at epochs 270 and 295. Our training task set is similar to MVMoE, including CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW, and OVRPL (see Appendix C.1 for further discussion). CoEKS adopts the mixed batch training and reward regularization scheme from RouteFinder, with the distillation strength $\alpha$ set to 0.01. MDis is employed in the first encoder layer. For all neural methods, the rest of the settings follow their original papers. 
Inference & Metrics. For all neural methods, a greedy rollout with $\times 8$ instance augmentation (Zhou et al., 2024a) is employed. The test set is obtained through random sampling, with 1000 instances per VRP task to reduce the impact of randomness. We show the average results of the tests, including the objective value (total cost), the gap to the best traditional solver, and the total test time. 
6 
Published as a conference paper at ICLR 2026 

Table 1: Performance on 1K test instances of ID VRP tasks.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">n = 50</td><td colspan="2">n = 100</td><td rowspan="2">Time</td><td rowspan="2">Method</td><td colspan="2">n = 50</td><td colspan="2">n = 100</td><td>Time</td></tr><tr><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Time</td><td>Obj.</td><td>Gap</td></tr><tr><td rowspan="8">CVRP</td><td>HGS-PyVRP#</td><td>10.372</td><td>*</td><td>10.4m</td><td>15.628</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>16.031</td><td>*</td><td>10.4m</td><td>25.423</td></tr><tr><td>OR-Tools#</td><td>10.572</td><td>1.907%</td><td>10.4m</td><td>16.280</td><td>4.178%</td><td>20.8m</td><td>OR-Tools#</td><td>16.089</td><td>0.347%</td><td>10.4m</td><td>25.814</td></tr><tr><td>POMO-MTL</td><td>10.502</td><td>1.257%</td><td>1s</td><td>15.875</td><td>1.617%</td><td>7s</td><td>POMO-MTL</td><td>16.428</td><td>2.471%</td><td>1s</td><td>26.487</td></tr><tr><td>MVMoE</td><td>10.482</td><td>1.059%</td><td>2s</td><td>15.841</td><td>1.399%</td><td>9s</td><td>MVMoE</td><td>16.439</td><td>2.550%</td><td>2s</td><td>26.472</td></tr><tr><td>RF-TE</td><td>10.497</td><td>1.213%</td><td>1s</td><td>15.829</td><td>1.327%</td><td>6s</td><td>RF-TE</td><td>16.390</td><td>2.237%</td><td>1s</td><td>26.283</td></tr><tr><td>CaDA</td><td>10.491</td><td>1.148%</td><td>3s</td><td>15.822</td><td>1.277%</td><td>11s</td><td>CaDA</td><td>16.297</td><td>1.651%</td><td>2s</td><td>26.119</td></tr><tr><td>ReLD-MoEL</td><td>10.467</td><td>0.920%</td><td>2s</td><td>15.797</td><td>1.116%</td><td>9s</td><td>ReLD-MoEL</td><td>16.414</td><td>2.386%</td><td>2s</td><td>26.388</td></tr><tr><td>CoEKS</td><td>10.464</td><td>0.891%</td><td>2s</td><td>15.787</td><td>1.057%</td><td>9s</td><td>CoEKS</td><td>16.361</td><td>2.050%</td><td>2s</td><td>26.300</td></tr><tr><td rowspan="8">OVRP</td><td>HGS-PyVRP#</td><td>6.507</td><td>*</td><td>10.4m</td><td>9.725</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>10.587</td><td>*</td><td>10.4m</td><td>25.766</td></tr><tr><td>OR-Tools#</td><td>6.553</td><td>0.686%</td><td>10.4m</td><td>9.995</td><td>2.732%</td><td>20.8m</td><td>OR-Tools#</td><td>10.570</td><td>2.343%</td><td>10.4m</td><td>16.466</td></tr><tr><td>POMO-MTL</td><td>6.706</td><td>3.025%</td><td>1s</td><td>10.173</td><td>4.592%</td><td>6s</td><td>POMO-MTL</td><td>10.756</td><td>1.550%</td><td>1s</td><td>16.090</td></tr><tr><td>MVMoE</td><td>6.685</td><td>2.697%</td><td>2s</td><td>10.138</td><td>4.226%</td><td>8s</td><td>MVMoE</td><td>10.736</td><td>1.362%</td><td>2s</td><td>16.053</td></tr><tr><td>RF-TE</td><td>6.678</td><td>2.595%</td><td>1s</td><td>10.097</td><td>3.813%</td><td>6s</td><td>RF-TE</td><td>10.742</td><td>1.434%</td><td>1s</td><td>16.017</td></tr><tr><td>CaDA</td><td>6.683</td><td>2.668%</td><td>2s</td><td>10.105</td><td>3.882%</td><td>11s</td><td>CaDA</td><td>10.729</td><td>1.317%</td><td>2s</td><td>16.014</td></tr><tr><td>ReLD-MoEL</td><td>6.661</td><td>2.343%</td><td>2s</td><td>10.073</td><td>3.559%</td><td>9s</td><td>ReLD-MoEL</td><td>10.713</td><td>1.153%</td><td>2s</td><td>15.998</td></tr><tr><td>CoEKS</td><td>6.648</td><td>2.138%</td><td>2s</td><td>10.046</td><td>3.290%</td><td>8s</td><td>CoEKS</td><td>10.712</td><td>1.152%</td><td>2s</td><td>15.997</td></tr><tr><td rowspan="8">VRP#</td><td>HGS-PyVRP#</td><td>9.687</td><td>*</td><td>10.4m</td><td>14.377</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>10.510</td><td>*</td><td>10.4m</td><td>26.926</td></tr><tr><td>OR-Tools#</td><td>9.802</td><td>1.159%</td><td>10.4m</td><td>14.933</td><td>3.853%</td><td>20.8m</td><td>OR-Tools#</td><td>10.519</td><td>0.078%</td><td>10.4m</td><td>17.027</td></tr><tr><td>POMO-MTL</td><td>9.995</td><td>3.177%</td><td>1s</td><td>14.989</td><td>4.279%</td><td>7s</td><td>POMO-MTL</td><td>10.691</td><td>1.700%</td><td>2s</td><td>17.500</td></tr><tr><td>MVMoE</td><td>9.966</td><td>2.867%</td><td>2s</td><td>14.952</td><td>4.025%</td><td>9s</td><td>MVMoE</td><td>10.696</td><td>1.747%</td><td>2s</td><td>17.485</td></tr><tr><td>RF-TE</td><td>9.984</td><td>3.050%</td><td>1s</td><td>14.926</td><td>3.838%</td><td>6s</td><td>RF-TE</td><td>10.675</td><td>1.542%</td><td>1s</td><td>17.363</td></tr><tr><td>CaDA</td><td>9.965</td><td>2.860%</td><td>2s</td><td>14.906</td><td>3.699%</td><td>11s</td><td>CaDA</td><td>10.626</td><td>1.084%</td><td>3s</td><td>17.267</td></tr><tr><td>ReLD-MoEL</td><td>9.936</td><td>2.557%</td><td>2s</td><td>14.877</td><td>3.496%</td><td>9s</td><td>ReLD-MoEL</td><td>10.682</td><td>1.613%</td><td>2s</td><td>17.429</td></tr><tr><td>CoEKS</td><td>9.930</td><td>2.497%</td><td>2s</td><td>14.854</td><td>3.338%</td><td>9s</td><td>CoEKS</td><td>10.659</td><td>1.393%</td><td>2s</td><td>17.376</td></tr><tr><td rowspan="8">OVRP#</td><td>HGS-PyVRP#</td><td>6.507</td><td>*</td><td>10.4m</td><td>9.724</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>10.029</td><td>*</td><td>10.4m</td><td>25.367</td></tr><tr><td>OR-Tools#</td><td>6.552</td><td>0.668%</td><td>10.4m</td><td>10.001</td><td>2.791%</td><td>20.8m</td><td>OR-Tools#</td><td>10.094</td><td>1.574%</td><td>10.4m</td><td>15.788</td></tr><tr><td>POMO-MTL</td><td>6.709</td><td>3.070%</td><td>1s</td><td>10.177</td><td>4.625%</td><td>6s</td><td>POMO-MTL</td><td>10.255</td><td>2.321%</td><td>1s</td><td>15.899</td></tr><tr><td>MVMoE</td><td>6.687</td><td>2.737%</td><td>2s</td><td>10.140</td><td>4.244%</td><td>9s</td><td>MVMoE</td><td>10.242</td><td>2.146%</td><td>2s</td><td>15.869</td></tr><tr><td>RF-TE</td><td>6.678</td><td>2.606%</td><td>1s</td><td>10.096</td><td>3.803%</td><td>6s</td><td>RF-TE</td><td>10.235</td><td>2.097%</td><td>1s</td><td>15.802</td></tr><tr><td>CaDA</td><td>6.684</td><td>2.685%</td><td>2s</td><td>10.106</td><td>3.900%</td><td>11s</td><td>CaDA</td><td>10.211</td><td>1.916%</td><td>2s</td><td>15.763</td></tr><tr><td>ReLD-MoEL</td><td>6.661</td><td>2.341%</td><td>2s</td><td>10.075</td><td>3.583%</td><td>9s</td><td>ReLD-MoEL</td><td>10.219</td><td>1.902%</td><td>2s</td><td>15.805</td></tr><tr><td>CoEKS</td><td>6.648</td><td>2.135%</td><td>2s</td><td>10.047</td><td>3.298%</td><td>9s</td><td>CoEKS</td><td>10.203</td><td>1.751%</td><td>2s</td><td>15.773</td></tr></table>

(ID Avg.): Average performance across ID VRP tasks. bold: Best results among learning-based methods. #: Results are adopted from Berto et al. (2024) for the convenience of comparison. *: Best traditional method, taken as the baseline for gap calculation. 

(Q1) Generalization for ID and OOD VRPs. Table 1 presents the results on ID tasks. Across different problem scales, CoEKS outperforms all neural methods on ID average gap (ID Avg.) and achieves the smallest gaps in 10 out of 14 cases. To evaluate their zero-shot OOD generalization performance, all methods are examined on 9 VRP tasks with unseen constraint combinations. As shown in Table 2, CoEKS consistently achieves the best performance, demonstrating its ability to effectively handle unseen constraint combinations by adaptively combining experts. For $n = 5 0$ and $n = 1 0 0$ , CoEKS outperforms all other neural methods in the overall OOD average gap (OOD Avg.), with relative improvements of at least $1 8 . 3 \%$ and $13 \%$ , respectively. Moreover, compared with the traditional solvers, CoEKS achieves competitive results with substantially lower solving time, offering notable efficiency for practical applications. We further conduct supplementary experiments on the OOD large-scale real-world instances of the CVRPLIB benchmark dataset $\mathit { \Delta } n > 5 0 0 $ ), where CoEKS consistently delivers superior generalization results (see Appendix E.4). 
(Q2) Scalability for adaptation to unseen constraints by plugging in new experts. An unseen constraint MB is considered in VRP tasks, following the challenging setting in RouteFinder. Co-EKS plugs a new MB-specific expert into the trained model and only fine-tunes this expert. Lin et al. (2024) introduce task-specific adapter layers (AL) for fine-tuning. RouteFinder proposes efficient adaptation layers (EAL) (Berto et al., 2024), which extend the weight matrix with zeropadding to support new constraints. We compare: 1) AL-based methods, including RF-TE-AL and ReLD-MoEL-AL; 2) EAL-based methods, including RF-TE-EAL and ReLD-MoEL-EAL; 3) two variants of our method. $\mathrm { C o E K S ^ { + } }$ , where the new expert and combiner are randomly initialized; and ${ \mathrm { C o E K S } } ^ { c + }$ , which reuses shared modules ( $E _ { c }$ and $S _ { c . }$ ) for initialization to accelerate learning, inspired by Jiang et al. (2024a). For simplicity, our variants both use EAL to adapt to new constraint attributes. Aligning with RouteFinder, fine-tuning is conducted for 10 epochs with 10K instances per epoch on tasks including VRPMB and VRPMBTW (see Appendix C.2 for more discussions). 
As shown in Table 3, our methods consistently outperforms all baselines, where ${ \mathrm { C o E K S } } ^ { c + }$ exploits the knowledge of shared modules in CoE and performs best. This demonstrates that CoEKS has superior scalability in adapting to a new constraint by flexibly plugging in a new expert and combiner. This advantage is further amplified in OOD tasks with more constraints, where the extended model capability allows CoEKS to better handle complex generalization challenges (see Appendix E.3 for further validation on scalability to new Multi-Depot (MD) constraint). 
7 
Published as a conference paper at ICLR 2026 

Table 2: Performance on 1K test instances of OOD VRP tasks.

<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="3">n = 50</td><td colspan="3">n = 100</td><td rowspan="2">Method</td><td colspan="3">n = 50</td><td colspan="3">n = 100</td></tr><tr><td>Obj.</td><td>Gap</td><td>Time</td><td>Obj.</td><td>Gap</td><td>Time</td><td>Obj.</td><td>Gap</td><td>Time</td><td>Obj.</td><td>Gap</td><td>Time</td></tr><tr><td rowspan="8">OVRPB</td><td>HGS-PyVRP#</td><td>6.898</td><td>*</td><td>10.4m</td><td>10.335</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>10.186</td><td>*</td><td>10.4m</td><td>14.779</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>6.928</td><td>0.412%</td><td>10.4m</td><td>10.577</td><td>2.315%</td><td>20.8m</td><td>OR-Tools#</td><td>10.331</td><td>1.390%</td><td>10.4m</td><td>15.426</td><td>4.338%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>7.447</td><td>7.886%</td><td>1s</td><td>12.091</td><td>16.926%</td><td>7s</td><td>POMO-MTL</td><td>10.743</td><td>5.284%</td><td>1s</td><td>15.875</td><td>7.246%</td><td>7s</td></tr><tr><td>MVMoE</td><td>7.371</td><td>6.797%</td><td>2s</td><td>11.720</td><td>13.305%</td><td>9s</td><td>MVMoE</td><td>10.709</td><td>4.936%</td><td>2s</td><td>15.792</td><td>6.699%</td><td>9s</td></tr><tr><td>RF-TE</td><td>7.378</td><td>6.900%</td><td>1s</td><td>11.840</td><td>14.520%</td><td>7s</td><td>RF-TE</td><td>10.777</td><td>5.685%</td><td>1s</td><td>17.011</td><td>15.149%</td><td>6s</td></tr><tr><td>CaDA</td><td>7.701</td><td>11.549%</td><td>2s</td><td>11.796</td><td>14.074%</td><td>12s</td><td>CaDA</td><td>10.794</td><td>5.705%</td><td>2s</td><td>15.883</td><td>7.399%</td><td>12s</td></tr><tr><td>ReLD-MoEL</td><td>7.335</td><td>6.272%</td><td>2s</td><td>11.446</td><td>10.691%</td><td>10s</td><td>ReLD-MoEL</td><td>10.621</td><td>4.130%</td><td>2s</td><td>15.639</td><td>5.694%</td><td>9s</td></tr><tr><td>CoEKS</td><td>7.241</td><td>4.913%</td><td>2s</td><td>11.251</td><td>8.811%</td><td>10s</td><td>CoEKS</td><td>10.650</td><td>4.387%</td><td>2s</td><td>15.636</td><td>5.691%</td><td>9s</td></tr><tr><td rowspan="8">VRPBTW</td><td>HGS-PyVRP#</td><td>18.292</td><td>*</td><td>10.4m</td><td>29.467</td><td>*</td><td>20.8m</td><td rowspan="8">VRPBTW</td><td>16.356</td><td>*</td><td>10.4m</td><td>25.757</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>18.366</td><td>0.383%</td><td>10.4m</td><td>29.945</td><td>1.597%</td><td>20.8m</td><td>16.441</td><td>0.499%</td><td>10.4m</td><td>26.259</td><td>1.899%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>19.105</td><td>4.409%</td><td>1s</td><td>31.419</td><td>6.592%</td><td>8s</td><td>16.864</td><td>3.057%</td><td>1s</td><td>27.041</td><td>4.929%</td><td>7s</td></tr><tr><td>MVMoE</td><td>18.976</td><td>3.717%</td><td>2s</td><td>31.441</td><td>6.675%</td><td>10s</td><td>16.868</td><td>3.088%</td><td>2s</td><td>26.996</td><td>4.765%</td><td>10s</td></tr><tr><td>RF-TE</td><td>19.029</td><td>4.011%</td><td>1s</td><td>31.383</td><td>6.479%</td><td>7s</td><td>16.865</td><td>3.069%</td><td>1s</td><td>27.055</td><td>4.983%</td><td>7s</td></tr><tr><td>CaDA</td><td>19.118</td><td>4.491%</td><td>3s</td><td>31.568</td><td>7.117%</td><td>13s</td><td>16.780</td><td>2.514%</td><td>2s</td><td>26.853</td><td>4.194%</td><td>13s</td></tr><tr><td>ReLD-MoEL</td><td>18.994</td><td>3.821%</td><td>2s</td><td>31.218</td><td>5.920%</td><td>10s</td><td>16.951</td><td>3.599%</td><td>2s</td><td>27.222</td><td>5.663%</td><td>10s</td></tr><tr><td>CoEKS</td><td>18.882</td><td>3.210%</td><td>2s</td><td>31.168</td><td>5.746%</td><td>10s</td><td>16.847</td><td>2.949%</td><td>2s</td><td>26.891</td><td>4.355%</td><td>10s</td></tr><tr><td rowspan="8">OVRPBLT</td><td>HGS-PyVRP#</td><td>6.899</td><td>*</td><td>10.4m</td><td>10.335</td><td>*</td><td>20.8m</td><td rowspan="8">OVRPBTW</td><td>11.669</td><td>*</td><td>10.4m</td><td>19.156</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>6.927</td><td>0.386%</td><td>10.4m</td><td>10.582</td><td>2.363%</td><td>20.8m</td><td>11.682</td><td>0.109%</td><td>10.4m</td><td>19.303</td><td>0.757%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>7.451</td><td>7.937%</td><td>1s</td><td>12.135</td><td>17.341%</td><td>7s</td><td>12.094</td><td>3.596%</td><td>1s</td><td>20.255</td><td>5.690%</td><td>8s</td></tr><tr><td>MVMoE</td><td>7.440</td><td>7.778%</td><td>2s</td><td>11.823</td><td>14.289%</td><td>10s</td><td>12.027</td><td>3.038%</td><td>2s</td><td>20.270</td><td>5.759%</td><td>11s</td></tr><tr><td>RF-TE</td><td>7.638</td><td>10.632%</td><td>1s</td><td>11.876</td><td>14.864%</td><td>6s</td><td>12.088</td><td>3.548%</td><td>1s</td><td>20.314</td><td>5.986%</td><td>8s</td></tr><tr><td>CaDA</td><td>7.699</td><td>11.505%</td><td>2s</td><td>11.791</td><td>14.024%</td><td>12s</td><td>12.143</td><td>4.005%</td><td>3s</td><td>20.401</td><td>6.430%</td><td>13s</td></tr><tr><td>ReLD-MoEL</td><td>7.344</td><td>6.380%</td><td>2s</td><td>11.427</td><td>10.506%</td><td>10s</td><td>12.039</td><td>3.125%</td><td>2s</td><td>20.159</td><td>5.192%</td><td>11s</td></tr><tr><td>CoEKS</td><td>7.230</td><td>4.747%</td><td>2s</td><td>11.245</td><td>8.755%</td><td>9s</td><td>11.972</td><td>2.566%</td><td>2s</td><td>20.114</td><td>4.957%</td><td>10s</td></tr><tr><td rowspan="8">OVRPBLTW</td><td>HGS-PyVRP#</td><td>10.51</td><td>*</td><td>10.4m</td><td>16.926</td><td>*</td><td>20.8m</td><td rowspan="8">OVRPBLTW</td><td>18.361</td><td>*</td><td>10.4m</td><td>29.026</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>10.497</td><td>0.114%</td><td>10.4m</td><td>17.023</td><td>0.728%</td><td>20.8m</td><td>18.422</td><td>0.332%</td><td>10.4m</td><td>29.830</td><td>2.770%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>10.695</td><td>1.735%</td><td>1s</td><td>17.508</td><td>3.415%</td><td>7s</td><td>19.482</td><td>4.746%</td><td>1s</td><td>31.940</td><td>7.081%</td><td>7s</td></tr><tr><td>MVMoE</td><td>10.725</td><td>2.015%</td><td>2s</td><td>17.489</td><td>3.300%</td><td>10s</td><td>19.361</td><td>4.119%</td><td>2s</td><td>31.996</td><td>7.286%</td><td>10s</td></tr><tr><td>RF-TE</td><td>10.706</td><td>1.837%</td><td>1s</td><td>17.500</td><td>3.363%</td><td>7s</td><td>19.410</td><td>4.379%</td><td>1s</td><td>31.936</td><td>7.089%</td><td>7s</td></tr><tr><td>CaDA</td><td>10.627</td><td>1.097%</td><td>2s</td><td>17.273</td><td>2.027%</td><td>13s</td><td>19.472</td><td>4.713%</td><td>3s</td><td>32.447</td><td>8.832%</td><td>14s</td></tr><tr><td>ReLD-MoEL</td><td>10.702</td><td>1.791%</td><td>2s</td><td>17.462</td><td>3.144%</td><td>10s</td><td>19.563</td><td>5.212%</td><td>2s</td><td>32.222</td><td>8.056%</td><td>11s</td></tr><tr><td>CoEKS</td><td>10.681</td><td>1.603%</td><td>2s</td><td>17.402</td><td>2.785%</td><td>10s</td><td>19.283</td><td>3.714%</td><td>2s</td><td>31.738</td><td>6.423%</td><td>10s</td></tr><tr><td rowspan="8">OVRPBLTW</td><td>HGS-PyVRP#</td><td>11.668</td><td>*</td><td>10.4m</td><td>19.156</td><td>*</td><td>20.8m</td><td rowspan="8">OVRPBLTW</td><td>12.315</td><td>*</td><td>10.4m</td><td>19.437</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>11.681</td><td>0.106%</td><td>10.4m</td><td>19.305</td><td>0.767%</td><td>20.8m</td><td>12.364</td><td>0.415%</td><td>10.4m</td><td>19.806</td><td>1.948%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>12.101</td><td>3.659%</td><td>1s</td><td>20.287</td><td>5.854%</td><td>8s</td><td>12.887</td><td>4.701%</td><td>1s</td><td>20.950</td><td>8.341%</td><td>8s</td></tr><tr><td>MVMoE</td><td>12.100</td><td>3.655%</td><td>2s</td><td>20.306</td><td>5.949%</td><td>10s</td><td>12.842</td><td>4.349%</td><td>2s</td><td>20.870</td><td>7.559%</td><td>10s</td></tr><tr><td>RF-TE</td><td>12.063</td><td>3.342%</td><td>1s</td><td>20.291</td><td>5.887%</td><td>7s</td><td>12.884</td><td>4.823%</td><td>1s</td><td>21.023</td><td>8.702%</td><td>7s</td></tr><tr><td>CaDA</td><td>12.129</td><td>3.889%</td><td>3s</td><td>20.356</td><td>6.206%</td><td>13s</td><td>12.940</td><td>5.496%</td><td>2s</td><td>20.930</td><td>7.812%</td><td>13s</td></tr><tr><td>ReLD-MoEL</td><td>12.082</td><td>3.490%</td><td>2s</td><td>20.266</td><td>5.747%</td><td>11s</td><td>12.848</td><td>4.202%</td><td>2s</td><td>20.784</td><td>6.735%</td><td>11s</td></tr><tr><td>CoEKS</td><td>11.999</td><td>2.797%</td><td>2s</td><td>20.158</td><td>5.186%</td><td>10s</td><td>12.754</td><td>3.432%</td><td>2s</td><td>20.623</td><td>5.857%</td><td>10s</td></tr></table>

(OOD Avg.): Average performance across OOD VRP tasks. bold: Best results among learning-based methods. #: Results are adopted from Berto et al. (2024) for the convenience of comparison. *: Best traditional method, taken as the baseline for gap calculation. 

(Q3) Universality across backbone models. we implement CoEKS on both the classic POMO (Kwon et al., 2020) and the SOTA ReLD (Huang et al., 2025) backbones. The POMObased methods include POMO-MTL (Liu et al., 2024b), RF-TE (Berto et al., 2024), MVMoE (Zhou et al., 2024a), and our POMO-CoEKS. The ReLD-based methods include ReLD-MTL, ReLD-RF, ReLD-MoEL (Huang et al., 2025), and our ReLD-CoEKS (i.e., the original CoEKS implementation). MVMoE and ReLD-MoEL retain the MoE modules in their original decoders. Experiments are conducted on $n = 5 0$ , and all new methods are trained using the original settings of their corresponding baselines. As illustrated in Figures 3 (a-b), CoEKS consistently delivers the best average gap on both ID and OOD tasks under both backbone models. These results highlight the universality and consistent superiority of CoEKS. 
(Q4) Effectiveness of knowledge sharing strategy. This section investigates the impact of the proposed multi-view knowledge sharing strategy. Ablation experiments are conducted on 16 VRP tasks with $n = 5 0$ , under consistent training settings. The SOTA ReLD-MoEL serves as the baseline for comparison. Specifically, three ablated variants are examined: without (w/o) MDis, w/o shared transformation layer, and w/o both. Figures 3 (c-d) show the average gap for all variants on ID and OOD instances. It demonstrates that the multi-view knowledge sharing strategy significantly enhances model performance, particularly in OOD generalization. The results verify that multiview knowledge sharing contributes to the learning of transferable knowledge across constraints. 
Position of MDis. The effect of mutual distillation (MDis) among experts at different encoder layers is studied. Starting from a CoEKS variant w/o MDis, the MDis mechanism is gradually introduced from lower to higher encoder layers, until enabled throughout. As shown in Figures 4 (ab), incorporating MDis is generally beneficial for ID generalization unless applied to all layers. This suggests that moderate expert collaboration may promote their comprehensive understanding of the training task. For OOD generalization, improvements are observed only when MDis is applied to the 
8 
Published as a conference paper at ICLR 2026 

Table 3: Fine-tuning on VRPMB and VRPMBTW at $n { = } 5 0$ .

<table><tr><td rowspan="2">Method</td><td colspan="2">VRPMB</td><td colspan="2">OVRPMB</td><td colspan="2">VRPMBL</td><td colspan="2">VRPMBTW</td><td colspan="2">OVRPMBL</td><td colspan="2">OVRPMBTW</td><td colspan="2">VRPMBLTW</td><td colspan="2">OVRPMBLTW</td></tr><tr><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td></tr><tr><td>HGS-PyVRP</td><td>9.09</td><td>*</td><td>6.11</td><td>*</td><td>16.31</td><td>*</td><td>9.49</td><td>*</td><td>6.11</td><td>*</td><td>10.47</td><td>*</td><td>16.01</td><td>*</td><td>10.47</td><td>*</td></tr><tr><td>RF-TE-AL</td><td>11.31</td><td>24.69%</td><td>9.03</td><td>47.92%</td><td>22.32</td><td>137.22%</td><td>19.20</td><td>20.32%</td><td>13.75</td><td>125.04%</td><td>14.22</td><td>36.37%</td><td>19.95</td><td>22.81%</td><td>14.48</td><td>38.90%</td></tr><tr><td>RF-TE-EAL</td><td>9.25</td><td>1.76%</td><td>6.38</td><td>4.42%</td><td>9.73</td><td>2.54%</td><td>16.36</td><td>2.16%</td><td>6.39</td><td>4.48%</td><td>10.71</td><td>2.24%</td><td>16.80</td><td>2.97%</td><td>10.88</td><td>3.81%</td></tr><tr><td>ReLD-MoEL-AL</td><td>10.65</td><td>17.26%</td><td>8.45</td><td>38.47%</td><td>11.37</td><td>19.91%</td><td>18.18</td><td>13.65%</td><td>8.61</td><td>41.06%</td><td>12.88</td><td>23.25%</td><td>18.84</td><td>15.65%</td><td>13.04</td><td>24.78%</td></tr><tr><td>ReLD-MoEL-EAL</td><td>9.32</td><td>2.59%</td><td>6.43</td><td>5.09%</td><td>9.71</td><td>2.35%</td><td>16.38</td><td>2.30%</td><td>6.43</td><td>5.17%</td><td>10.67</td><td>1.88%</td><td>16.93</td><td>3.80%</td><td>10.70</td><td>2.14%</td></tr><tr><td>CoEKS+</td><td>9.24</td><td>1.61%</td><td>6.37</td><td>4.19%</td><td>9.71</td><td>2.23%</td><td>16.33</td><td>1.92%</td><td>6.35</td><td>3.84%</td><td>10.64</td><td>1.57%</td><td>16.80</td><td>2.97%</td><td>10.66</td><td>1.75%</td></tr><tr><td>CoEKS++</td><td>9.23</td><td>1.56%</td><td>6.33</td><td>3.52%</td><td>9.70</td><td>2.12%</td><td>16.33</td><td>1.97%</td><td>6.33</td><td>3.48%</td><td>10.65</td><td>1.62%</td><td>16.79</td><td>2.92%</td><td>10.66</td><td>1.74%</td></tr></table>
![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/bb22a62ecc1954e4771669b8ab444dd430cd220681e9e4cc7dac3be8ca5aae33.jpg)


(a) Verification of universality

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/8103debd0f0a5081a7aaa2262aaf1589266ecbf7ba5e0a1725d48f6868ecb79d.jpg)


(b) Verification of universality

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/15601b99831f135fe38cbfcbc2a59fb15e2e7b28f2b7b364f0e7e7a3539fad12.jpg)


(c) Ablation study on CoEKS

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/61231dc2fec259d31bdfa96dd183439d282732c1f2f025a0a7ee20d20d60b401.jpg)


(d) Ablation study on CoEKS


Figure 3: Left two panels: universality tests both (a) ID and (b) OOD tasks. Right two panels: ablation Study both (c) ID and (d) OOD tasks.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/da6403d21ee272eb0e3690963b40040780c02db54a9f0115527edf80b7d1dbc4.jpg)


(a) Position of MDis

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/f71f6d2485cef9d38d764d5bd8e57a86f41da548fef8f473c905ce4a1e5b79c0.jpg)


(b) Position of MDis

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/be98d51c7ae2846b6f0a4db9d15096c64848286e3f46cebf49249ff107b40763.jpg)


(c) Distillation strength

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/2f054d85cf8463e73b1e16e986059fe1e9e3a81a820eb087548e0bdb4624fb64.jpg)


(d) Distillation strength


Figure 4: Left two panels: effect of MDis position on (a) ID and (b) OOD performance. The horizontal axis is the encoder layer number. Right two panels: effect of MDis strength on (c) ID and (d) OOD performance. The $\mathbf { X }$ -axis is the weight $\alpha$ .

lower layers. The findings imply that deeper layers encode more task-specific patterns. Applying MDis on these layers may lead to homogenization of knowledge specialized in constraints, thus preventing model generalization to new VRPs with unseen constraint combinations. These results validate our choice of applying MDis only at the first encoder layer, which achieves a good balance between expert interaction and specialization. 
Distillation Strength. We further assess the impact of the distillation coefficient $\alpha$ by measuring average gaps on both ID and OOD tasks. Figures 4 (c-d) show that a properly chosen $\alpha$ enhances overall performance. However, when $\alpha$ increases beyond a certain threshold, the experts tend to produce overly similar outputs, resulting in no performance gains. This supports our use of $\alpha = 0 . 0 1$ as a simple and effective setting. 
# 6 CONCLUSION
This paper presents CoEKS, a novel model that leverages the structural characteristic of VRPs to address cross-task challenges. CoEKS integrates two complementary components: a combinationof-experts architecture that adaptively combines constraint-specific experts for diverse VRPs, and a multi-view knowledge sharing strategy that automatically learns transferable knowledge to enhance cross-task generalization. In addition, new experts can be seamlessly plugged into the trained model to handle unseen constraints. Extensive evaluations on 24 VRP tasks demonstrate that CoEKS achieves SOTA performance on ID tasks and yields even greater gains on OOD scenarios, including unseen constraint combinations and new constraints. Furthermore, CoEKS exhibits consistent superiority across backbone models, highlighting its universality. 
9 
Published as a conference paper at ICLR 2026 
A current limitation is that handling more constraints inevitably increases the number of parameters. However, this trade-off is natural, since more complex problems with more constraints demand stronger model capability. A promising future direction is to explore more efficient expert-sharing mechanisms enabling a single expert to serve multiple similar constraints, or more efficient parameterization strategies to scale model capability. 
# ACKNOWLEDGMENTS
This work is supported by the National Natural Science Foundation of China (62472461), and the Guangdong Basic and Applied Basic Research Foundation (2025A1515010129). 
# REFERENCES


Irwan Bello, Hieu Pham, Quoc V Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning. In International Conference on Learning Representations, 2017. 




Federico Berto, Chuanbo Hua, Nayeli Gast Zepeda, Andre Hottung, Niels Wouda, Leon Lan, Kevin ′ Tierney, and Jinkyoo Park. Routefinder: Towards foundation models for vehicle routing problems. In ICML 2024 Workshop on Foundation Models in the Wild, 2024. 




Jieyi Bi, Yining Ma, Jiahai Wang, Zhiguang Cao, Jinbiao Chen, Yuan Sun, and Yeow Meng Chee. Learning generalizable models for vehicle routing problems via knowledge distillation. In Advances in Neural Information Processing Systems, volume 35, pp. 31226–31238, 2022. 




Jieyi Bi, Yining Ma, Jianan Zhou, Wen Song, Zhiguang Cao, Yaoxin Wu, and Jie Zhang. Learning to handle complex constraints for vehicle routing problems. In Advances in Neural Information Processing Systems, volume 37, pp. 93479–93509, 2024. 




Aigerim Bogyrbayeva, Meraryslan Meraliyev, Taukekhan Mustakhov, and Bissenbay Dauletbayev. Machine learning to solve vehicle routing problems: A survey. IEEE Transactions on Intelligent Transportation Systems, 25(6):4754–4772, 2024. 




Jinbiao Chen, Jiahai Wang, Zizhen Zhang, Zhiguang Cao, Te Ye, and Siyuan Chen. Efficient meta neural heuristic for multi-objective combinatorial optimization. In Advances in Neural Information Processing Systems, volume 36, pp. 56825–56837, 2023a. 




Jinbiao Chen, Zizhen Zhang, Zhiguang Cao, Yaoxin Wu, Yining Ma, Te Ye, and Jiahai Wang. Neural multi-objective combinatorial optimization with diversity enhancement. In Advances in Neural Information Processing Systems, pp. 39176–39188, 2023b. 




Jinbiao Chen, Zhiguang Cao, Jiahai Wang, Yaoxin Wu, Hanzhang Qin, Zizhen Zhang, and Yue-Jiao Gong. Rethinking neural multi-objective combinatorial optimization via neat weight embedding. In International Conference on Learning Representations, 2025a. 




Jinbiao Chen, Jiahai Wang, Zhiguang Cao, and Yaoxin Wu. Neural multi-objective combinatorial optimization via graph-image multimodal fusion. In International Conference on Learning Representations, 2025b. 




Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1280–1297, 2024. 




Darko Drakulic, Sofia Michel, and Jean-Marc Andreoli. Goal: A generalist combinatorial optimization agent learner. In International Conference on Learning Representations, 2024. 




Han Fang, Zhihao Song, Paul Weng, and Yutong Ban. Invit: A generalizable routing problem solver with invariant nested view transformer. In International Conference on Machine Learning, pp. 12973–12992, 2024. 


10 
Published as a conference paper at ICLR 2026 


Zhanhong Fang, Debing Wang, Jinbiao Chen, Jiahai Wang, and Zizhen Zhang. Ucpo: A universal constrained combinatorial optimization method via preference optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026. 




Weigang Fu, Jiawei Li, Zhe Liao, and Yaoming Fu. A bi-objective optimization approach for scheduling electric ground-handling vehicles in an airport. Complex & Intelligent Systems, 11 (4):1–27, 2025. 




Vincent Furnon and Laurent Perron. Or-tools routing library, 2023. URL https:// developers.google.com/optimization/routing. 




Yong Liang Goh, Zhiguang Cao, Yining Ma, Yanfei Dong, Mohammed Haroon Dupty, and Wee Sun Lee. Hierarchical neural constructive solver for real-world tsp scenarios. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 884–895, 2024. 




Yong Liang Goh, Yining Ma, Jianan Zhou, Zhiguang Cao, Mohammed Haroon Dupty, and Wee Sun Lee. Shield: Multi-task multi-distribution vehicle routing solver with sparsity & hierarchy in efficiently layered decoder. In International Conference on Machine Learning, 2025. 




Nasimeh Heydaribeni, Xinrui Zhan, Ruisi Zhang, Tina Eliassi-Rad, and Farinaz Koushanfar. Distributed constrained combinatorial optimization leveraging hypergraph neural networks. Nature Machine Intelligence, 6(6):664–672, 2024. 




Qingchun Hou, Jingwei Yang, Yiqiang Su, Xiaoqing Wang, and Yuming Deng. Generalize learned heuristics to solve large-scale vehicle routing problems in real-time. In International Conference on Learning Representations, 2023. 




Ziwei Huang, Jianan Zhou, Zhiguang Cao, and Yixin Xu. Rethinking light decoder-based solvers for vehicle routing problems. In International Conference on Learning Representations, 2025. 




Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lelio Renard Lavaud, Lucile Saulnier, Marie- ′ Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Theophile Gervet, Thibaut Lavril, Thomas Wang, Timoth ′ ee Lacroix, and William El Sayed. ′ Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024a. 




Xia Jiang, Yaoxin Wu, Yuan Wang, and Yingqian Zhang. Unco: Towards unifying neural combinatorial optimization through large language model. arXiv preprint arXiv:2408.12214, 2024b. 




Wouter Kool, Herke Van Hoof, and Max Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019. 




Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, and Seungjai Min. Pomo: Policy optimization with multiple optima for reinforcement learning. In Advances in Neural Information Processing Systems, volume 33, pp. 21188–21198, 2020. 




Han Li, Fei Liu, Zhi Zheng, Yu Zhang, and Zhenkun Wang. Cada: Cross-problem routing solver with constraint-aware dual-attention. In International Conference on Learning Representations, 2025a. 




Qi Li, Zhiguang Cao, Yining Ma, Yaoxin Wu, and Yue-Jiao Gong. Diversity optimization for travelling salesman problem via deep reinforcement learning. arXiv preprint arXiv:2501.00884, 2025b. 




Zijun Liao, Jinbiao Chen, Debing Wang, Zizhen Zhang, and Jiahai Wang. Bopo: Neural combinatorial optimization via best-anchored and objective-guided preference optimization. In International Conference on Machine Learning, 2025. 




Zhuoyi Lin, Yaoxin Wu, Bangjian Zhou, Zhiguang Cao, Wen Song, Yingqian Zhang, and Jayavelu Senthilnath. Cross-problem learning for solving vehicle routing problems. In The 33rd International Joint Conference on Artificial Intelligence (IJCAI-24), 2024. 


Published as a conference paper at ICLR 2026 


Fei Liu, Xi Lin, Weiduo Liao, Zhenkun Wang, Qingfu Zhang, Xialiang Tong, and Mingxuan Yuan. Prompt learning for generalized vehicle routing. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pp. 6976–6984, 2024a. 




Fei Liu, Xi Lin, Zhenkun Wang, Qingfu Zhang, Tong Xialiang, and Mingxuan Yuan. Multi-task learning for routing problem with cross-problem zero-shot generalization. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1898–1908, 2024b. 




Suyu Liu, Zhiguang Cao, Shanshan Feng, and Yew-Soon Ong. A mixed-curvature based pre-training paradigm for multi-task vehicle routing solver. In International Conference on Machine Learning, 2025. 




Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I Jordan. Deep transfer learning with joint adaptation networks. In International Conference on Machine Learning, pp. 2208–2217, 2017. 




Fu Luo, Xi Lin, Fei Liu, Qingfu Zhang, and Zhenkun Wang. Neural combinatorial optimization with heavy decoder: Toward large scale generalization. In Advances in Neural Information Processing Systems, volume 36, pp. 8845–8864, 2023. 




Fu Luo, Xi Lin, Yaoxin Wu, Zhenkun Wang, Tong Xialiang, Mingxuan Yuan, and Qingfu Zhang. Boosting neural combinatorial optimization for large-scale vehicle routing problems. In International Conference on Learning Representations, 2025. 




Yining Ma, Jingwen Li, Zhiguang Cao, Wen Song, Le Zhang, Zhenghua Chen, and Jing Tang. Learning to iteratively solve routing problems with dual-aspect collaborative transformer. In Advances in Neural Information Processing Systems, volume 34, pp. 11096–11107, 2021. 




Yining Ma, Zhiguang Cao, and Yeow Meng Chee. Learning to search feasible and infeasible regions of routing problems with flexible neural k-opt. In Advances in Neural Information Processing Systems, volume 36, pp. 49555–49578, 2023. 




Sahil Manchanda, Sofia Michel, Darko Drakulic, and Jean-Marc Andreoli. On the generalization of neural combinatorial optimization heuristics. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 426–442. Springer, 2022. 




Mohammadreza Nazari, Afshin Oroojlooy, Lawrence Snyder, and Martin Takac. Reinforcement ′ learning for solving the vehicle routing problem. In Advances in Neural Information Processing Systems, pp. 9861–9871, 2018. 




Yuxin Pan, Ruohong Liu, Yize Chen, Zhiguang Cao, and Fangzhen Lin. Hierarchical learning-based graph partition for large-scale vehicle routing problems. arXiv preprint arXiv:2502.08340, 2025. 




Erfan Babaee Tirkolaee, Alireza Goli, Amin Faridnia, Mehdi Soltani, and Gerhard-Wilhelm Weber. Multi-objective optimization for the reliable pollution-routing problem with cross-dock selection using pareto-based algorithms. Journal of Cleaner Production, 276:122927, 2020. 




Eduardo Uchoa, Diego Pecin, Artur Alves Pessoa, Marcus Poggi, Thibaut Vidal, and Anand Subramanian. New benchmark instances for the capacitated vehicle routing problem. European Journal of Operational Research, 257(3):845–858, 2017. 




Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, ?ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. 




Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. In Advances in Neural Information Processing Systems, volume 28, 2015. 




Chenguang Wang and Tianshu Yu. Efficient training of multi-task neural solver with multi-armed bandits. arXiv preprint arXiv:2305.06361, 2023. 




Chenguang Wang, Zhouliang Yu, Stephen McAleer, Tianshu Yu, and Yaodong Yang. Asp: Learn a universal neural solver! IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(6): 4102–4114, 2024. 


12 
Published as a conference paper at ICLR 2026 


Niels A. Wouda, Leon Lan, and Wouter Kool. Pyvrp: A high-performance VRP solver package. INFORMS Journal on Computing, 36(4):943–955, 2024. 




Xuan Wu, Di Wang, Lijie Wen, Yubin Xiao, Chunguo Wu, Yuesong Wu, Chaoyu Yu, Douglas L Maskell, and You Zhou. Neural combinatorial optimization algorithms for solving vehicle routing problems: A comprehensive survey with perspectives. arXiv preprint arXiv:2406.00415, 2024. 




Yaoxin Wu, Wen Song, Zhiguang Cao, Jie Zhang, and Andrew Lim. Learning improvement heuristics for solving routing problems. IEEE Transactions on Neural Networks and Learning Systems, 33(9):5057–5069, 2021. 




Pei Xiao, Zizhen Zhang, Jinbiao Chen, Jiahai Wang, and Zhenzhen Zhang. Neural combinatorial optimization for robust routing problem with uncertain travel times. In Advances in Neural Information Processing Systems, 2024. 




Zhitian Xie, Yinger Zhang, Chenyi Zhuang, Qitao Shi, Zhining Liu, Jinjie Gu, and Guannan Zhang. Mode: A mixture-of-experts model with mutual distillation among the experts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 16067–16075, 2024. 




Zizhen Zhang, Zhiyuan Wu, Hang Zhang, and Jiahai Wang. Meta-learning-based deep reinforcement learning for multiobjective optimization problems. IEEE Transactions on Neural Networks and Learning Systems, 34(10):7978–7991, 2023. 




Jianan Zhou, Zhiguang Cao, Yaoxin Wu, Wen Song, Yining Ma, Jie Zhang, and Chi Xu. Mvmoe: multi-task vehicle routing solver with mixture-of-experts. In International Conference on Machine Learning, pp. 61804–61824, 2024a. 




Jianan Zhou, Yaoxin Wu, Zhiguang Cao, Wen Song, Jie Zhang, and Zhiqi Shen. Collaboration! towards robust neural methods for routing problems. In Advances in Neural Information Processing Systems, pp. 121731–121764, 2024b. 




Zefang Zong, Hansen Wang, Jingwei Wang, Meng Zheng, and Yong Li. RBG: hierarchically solving large-scale routing problems in logistic systems via reinforcement learning. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4648–4658, 2022. 


13 
Published as a conference paper at ICLR 2026 
# Combination-of-Experts with Knowledge Sharing for Cross-Task Vehicle Routing Problems (Appendix)
# A DETAILS OF VRPS
This section provides details on the 6 basic constraints described in Section 3.1. By combining these constraints with the base CVRP task, a total of 48 VRP tasks are constructed. CoEKS is evaluated on 16 of these tasks as introduced in MVMoE (Zhou et al., 2024a). To further assess the scalability of the model to new constraints, scalability experiments are conducted on the remaining 32 VRP tasks with Mixed Backhauls or Multi-Depots or both (see Table 4), following RouteFinder (Berto et al., 2024). The data generation process for VRP tasks is detailed below. 
Node Coordinates. The single depot and all customer nodes are uniformly sampled within the unit square $[ 0 , 1 ] ^ { 2 }$ . 
Capacity (C). Following RouteFinder and MVMoE, the vehicle capacity $C$ is set to 40 for $n = 5 0$ and 50 for $n ~ = ~ 1 0 0$ . For customer $i$ , the linehaul demand $d _ { i }$ is sampled from the integer set $\{ 1 , 2 , \ldots , 9 \}$ . 
Backhaul (B). $20 \%$ of customers are randomly selected to sample their backhaul demand from the integer set $\{ 1 , 2 , \ldots , 9 \}$ , while the rest are set to 0. For the selected customers, the linehaul demand is set to 0. As a result, each customer has only one type of demand. 
For non-backhaul instances, all backhaul demands are set to 0. Before passing into the model, all linehaul and backhaul demands are normalized by vehicle capacity $C$ to $[ 0 , 1 ]$ . 
Duration Limit (L). This constraint imposes a maximum route length $L$ per vehicle. Following RouteFinder, $L$ is sampled from $U ( 2 \operatorname* { m a x } ( c _ { 0 i } ) , 3 . 0 )$ , where $c _ { 0 i }$ is the distance from the depot to customer $i$ . 
Time Window (TW). Following RouteFinder, the time window and service time for customer $i$ are generated through a multi-step process to ensure feasibility and diversity: 
1. Service time: Sample service time $s _ { i } \sim U [ 0 . 1 5 , 0 . 1 8 ]$ ]. 
2. Window length: Sample time window length $\Delta t _ { i } \sim U [ 0 . 1 8 , 0 . 2 ]$ ]. 
3. Upper bound: Calculate the upper bound for start time as $\begin{array} { r } { u _ { i } = \frac { T _ { \mathrm { m a x } } - s _ { i } - \Delta t _ { i } } { c _ { 0 i } } - 1 } \end{array}$ , where $T _ { \mathrm { m a x } }$ is the maximum allowed duration for a route. 
4. Start time: Set the start time $e _ { i } = ( 1 + ( u _ { i } - 1 ) \cdot r _ { i } ) \cdot c _ { 0 i }$ , where $r _ { i } \sim U ( 0 , 1 )$ 
5. End time: Compute the end time $l _ { i } = e _ { i } + \Delta t _ { i }$ 
For the depot node, the time window is fixed to $[ 0 , T _ { \mathrm { m a x } } ]$ and the service time is set to 0. In addition, the vehicle speed is 1.0. 
Open Route (O). The O constraint alters the route structure, allowing vehicles to finish at any customer node instead of returning to the depot. It is implemented by setting a binary indicator $o =$ 1, without additional data. When combined with other constraints, feasibility checks are adjusted dynamically: 
? With L, the route length is computed without the return-to-depot distance. 
? With TW, arrival time calculations omit the depot return segment. 
? With C, B, or MB, the constraint logic remains unchanged, while the depot return requirement is removed. 
Mixed Backhauls (MB) The demand configuration follows the same setup as the backhaul constraint. For instances involving MB, a binary flag $\mu$ is set to 1 and 0 otherwise. This flag is used to distinguish between instances with and without the MB constraint. 
1 
Published as a conference paper at ICLR 2026 

Table 4: 48 VRP tasks with 7 constraints. 32 VRP tasks with Mixed Backhauls or Multi-Depots or both are used to evaluate model’s scalability in adapting to unseen constraints.

<table><tr><td>VRP task</td><td>Capacity (C)</td><td>Open Route (O)</td><td>Backhaul (B)</td><td>Mixed Backhauls (MB)</td><td>Duration Limit (L)</td><td>Time Window (TW)</td><td>Multi-Depot (MD)</td></tr><tr><td>CVRP</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OVRP</td><td>?</td><td>?</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VRPB</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td></td></tr><tr><td>VRPL</td><td>?</td><td></td><td></td><td></td><td>?</td><td></td><td></td></tr><tr><td>VRPTW</td><td>?</td><td></td><td></td><td></td><td></td><td>?</td><td></td></tr><tr><td>OVRPTW</td><td>?</td><td>?</td><td></td><td></td><td></td><td>?</td><td></td></tr><tr><td>OVRPL</td><td>?</td><td>?</td><td></td><td></td><td>?</td><td></td><td></td></tr><tr><td>OVRPB</td><td>?</td><td>?</td><td>?</td><td></td><td></td><td></td><td></td></tr><tr><td>VRPBL</td><td>?</td><td></td><td>?</td><td></td><td>?</td><td></td><td></td></tr><tr><td>VRPBTW</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td></td></tr><tr><td>VRPLTW</td><td>?</td><td></td><td></td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>OVRPBL</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td></td><td></td></tr><tr><td>OVRPTBW</td><td>?</td><td>?</td><td>?</td><td></td><td></td><td>?</td><td></td></tr><tr><td>OVRPLTW</td><td>?</td><td>?</td><td></td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>VRPBLTW</td><td>?</td><td></td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>OVRPBLTW</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td></td></tr><tr><td>VRPMB</td><td>?</td><td></td><td>?</td><td>?</td><td></td><td></td><td></td></tr><tr><td>OVRPMB</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td></td><td></td></tr><tr><td>VRPMBL</td><td>?</td><td></td><td>?</td><td>?</td><td>?</td><td></td><td></td></tr><tr><td>VRPMBTW</td><td>?</td><td></td><td>?</td><td>?</td><td></td><td>?</td><td></td></tr><tr><td>OVRPMBL</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td></td></tr><tr><td>OVRPTBW</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td></td></tr><tr><td>OVRMLTW</td><td>?</td><td></td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>OVRMLTW</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td></tr><tr><td>MDCVRP</td><td>?</td><td></td><td></td><td></td><td></td><td></td><td>?</td></tr><tr><td>MDOVRP</td><td>?</td><td>?</td><td></td><td></td><td></td><td></td><td>?</td></tr><tr><td>MDVRP</td><td>?</td><td></td><td>?</td><td></td><td></td><td></td><td>?</td></tr><tr><td>MDVRPL</td><td>?</td><td></td><td></td><td></td><td>?</td><td></td><td>?</td></tr><tr><td>MDVRPTW</td><td>?</td><td></td><td></td><td></td><td></td><td>?</td><td>?</td></tr><tr><td>MDOVRPTW</td><td>?</td><td>?</td><td></td><td></td><td></td><td>?</td><td>?</td></tr><tr><td>MDOVRPL</td><td>?</td><td>?</td><td></td><td></td><td>?</td><td></td><td>?</td></tr><tr><td>MDOVRPB</td><td>?</td><td>?</td><td>?</td><td></td><td></td><td></td><td>?</td></tr><tr><td>MDVRPBL</td><td>?</td><td></td><td>?</td><td></td><td>?</td><td></td><td>?</td></tr><tr><td>MDVRPBTW</td><td>?</td><td></td><td>?</td><td></td><td></td><td>?</td><td>?</td></tr><tr><td>MDVRPLTW</td><td>?</td><td></td><td></td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDOVRPBL</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td></td><td>?</td></tr><tr><td>MDOVRPTBW</td><td>?</td><td>?</td><td>?</td><td></td><td></td><td>?</td><td>?</td></tr><tr><td>MDOVRPLTW</td><td>?</td><td>?</td><td></td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDOVRPTW</td><td>?</td><td>?</td><td>?</td><td></td><td></td><td></td><td>?</td></tr><tr><td>MDVRPBLTW</td><td>?</td><td></td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDVRPBLTW</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDVRPMB</td><td>?</td><td></td><td>?</td><td>?</td><td></td><td></td><td>?</td></tr><tr><td>MDOVRPMB</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td></td><td>?</td></tr><tr><td>MDVRPMBL</td><td>?</td><td></td><td>?</td><td>?</td><td>?</td><td></td><td>?</td></tr><tr><td>MDVRPMBTW</td><td>?</td><td></td><td>?</td><td>?</td><td></td><td>?</td><td>?</td></tr><tr><td>MDOVRPMBL</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td></tr><tr><td>MDOVRPMTW</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td></tr><tr><td>MDOVRPMTW</td><td>?</td><td>?</td><td>?</td><td>?</td><td></td><td>?</td><td>?</td></tr><tr><td>MDVRPMTW</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr><tr><td>MDVRPMTW</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr></table>
Multi-Depots (MD) The single-depot setting is extended to a multi-depot configuration. Vehicles may start from any depot but must return to the depot they depart from. Appendix E.3 reports results on 24 variants with the MD constraint. Following RouteFinder, the number of depots is fixed to three. 
# B DETAILED ARCHITECTURE OF COEKS
# B.1 ENCODER
The encoder transforms static node features into embeddings for various VRP tasks. For the $i$ -th $( i \in \{ 1 , \ldots , n \} )$ customer node, the static feature is defined as $F _ { i } = \{ x _ { i } , y _ { i } , d _ { i } , p _ { i } , e _ { i } , l _ { i } , s _ { i } \}$ , where $x _ { i } , y _ { i }$ represent coordinates, $d _ { i } , p _ { i }$ denote linehaul and backhaul requirements, $e _ { i } , l _ { i }$ specify the time window, and $s _ { i }$ indicates service time. The depot node $F _ { 0 } = \{ x _ { 0 } , y _ { 0 } , \mu \}$ , where $\mu$ is a binary flag indicating the presence of mixed backhaul. These features are projected into an initial embedding $h ^ { 0 } \in \mathbb { R } ^ { ( \bar { n } + 1 ) \times \bar { d } }$ through linear layers: 
$$
h ^ {0} = \operatorname {C o n c a t} \left(W _ {s _ {1}} F _ {0}, W _ {s _ {2}} F _ {1}, W _ {s _ {2}} F _ {2}, \dots , W _ {s _ {2}} F _ {n}\right). \tag {9}
$$
$W _ { s _ { 1 } } \in \mathbb { R } ^ { 3 \times d }$ and $W _ { s _ { 2 } } \in \mathbb { R } ^ { 7 \times d }$ are learnable parameter matrices, where $d = 1 2 8$ . $N$ $( N = 6 )$ encoder layers process $h ^ { 0 }$ to produce the final embeddings $h ^ { N }$ . Each encoder layer comprises 
2 
Published as a conference paper at ICLR 2026 
two components: a multi-head attention (MHA) layer followed by a feedforward network (FFN) layer. Both are integrated with residual connections and instance normalization (IN) to stabilize training (Vaswani et al., 2017). Formally, for the $\ell$ -th layer $( \ell \in [ 0 , N - 1 ] )$ ): 
$$
\tilde {h} _ {i} ^ {\ell} = \mathrm {I N} \left(h _ {i} ^ {\ell} + \mathrm {M H A} \left(h _ {i} ^ {\ell}, h _ {i} ^ {\ell}, h _ {i} ^ {\ell}\right)\right), \tag {10}
$$
$$
h _ {i} ^ {\ell + 1} = \operatorname {I N} \left(\tilde {h} _ {i} ^ {\ell} + \operatorname {F F N} \left(\tilde {h} _ {i} ^ {\ell}\right)\right). \tag {11}
$$
Multi-head attention (MHA). The MHA mechanism employs $A$ ( $A = 8$ ) attention heads to compute diverse node interactions in parallel. Their outputs are then aggregated into a unified representation. For an input embedding $\bar { h } _ { i } ^ { \ell } \in \mathbb { R } ^ { d }$ , each head $a \in \{ 1 , 2 , \ldots , A \}$ computes query $( Q )$ , key $( K )$ , and value $( V )$ vectors: 
$$
Q _ {i} ^ {\ell , a} = W _ {Q} ^ {a} h _ {i} ^ {\ell}, \quad K _ {i} ^ {\ell , a} = W _ {K} ^ {a} h _ {i} ^ {\ell}, \quad V _ {i} ^ {\ell , a} = W _ {V} ^ {a} h _ {i} ^ {\ell}, \tag {12}
$$
where $W _ { Q } ^ { a } , W _ { K } ^ { a } , W _ { V } ^ { a } \in \mathbb { R } ^ { d _ { k } \times d }$ are learnable parameter matrices, and $d _ { k } = d / A = 1 6$ . These projections are computed for all nodes $i \in \{ 0 , 1 , \ldots , n \}$ , with node 0 as the depot. The compatibility between nodes $i$ and $j \in \{ 0 , 1 , \ldots , n \}$ is measured via scaled dot-product attention, followed by a softmax: 
$$
u _ {i j} ^ {\ell , a} = \operatorname {S o f t m a x} \left(\frac {\left(Q _ {i} ^ {\ell , a}\right) ^ {T} K _ {j} ^ {\ell , a}}{\sqrt {d}}\right). \tag {13}
$$
A weighted sum over the value vectors is then computed for each head: 
$$
z _ {i} ^ {\ell , a} = \sum_ {j = 0} ^ {n} u _ {i j} ^ {\ell , a} V _ {j} ^ {\ell , a}. \tag {14}
$$
The outputs from heads are concatenated. Finally, a linear transformation is applied to obtain the output of the $i$ -th node in the MHA layer: 
$$
\operatorname {M H A} \left(h _ {i} ^ {\ell}, h _ {i} ^ {\ell}, h _ {i} ^ {\ell}\right) = \operatorname {C o n c a t} \left(z _ {i} ^ {\ell , 1}, z _ {i} ^ {\ell , 2}, \dots , z _ {i} ^ {\ell , A}\right) W _ {O}, \tag {15}
$$
where $W _ { O } \in \mathbb { R } ^ { d \times d }$ is a learnable parameter matrix. 
Feedforward network (FFN). The FFN layer contains two linear layers with a ReLU activation: 
$$
\operatorname {F F N} \left(\tilde {h} _ {i} ^ {\ell}\right) = W _ {F _ {1}} ^ {\ell} \cdot \operatorname {R e L U} \left(W _ {F _ {2}} ^ {\ell} \tilde {h} _ {i} ^ {\ell}\right), \tag {16}
$$
where $W _ { F _ { 1 } } ^ { \ell } \in \mathbb { R } ^ { d _ { f } \times d }$ and $W _ { F _ { 2 } } ^ { \ell } \in \mathbb { R } ^ { d \times d _ { f } }$ are learnable parameter matrices, and $d _ { f } = 5 1 2$ denotes the hidden dimension. In our framework, each FFN layer in each encoder layer is replaced with a CoEKS layer, comprising $k$ FFNs $\{ \mathrm { F F N _ { 1 } } , . . . , \mathrm { F F N } _ { k } \}$ . $k = 5$ aligns with the number of VRP constraints considered in our experiments. It can be flexibly extended to adapt new constraints. According to Eq. (8), Eq. (11) can be rewritten as: 
$$
h _ {i} ^ {\ell + 1} = \operatorname {I N} \left(\tilde {h} _ {i} ^ {\ell} + \sum_ {j \in C S} \operatorname {F F N} _ {j} \left(\tilde {h} _ {i} ^ {\ell}\right) \cdot S _ {j} \left(f _ {s} \left(\tilde {h} _ {i} ^ {\ell}\right)\right)\right). \tag {17}
$$
where $C S$ is the set of constraints activated for the current instance, $S _ { j } ( \cdot )$ represents the $j$ -th activated combiner function, and $f _ { s } ( \cdot )$ is the shared transformation layer. 
# B.2 DECODER
The decoder constructs solutions by sequentially selecting nodes based on static embeddings $h ^ { N }$ (produced by the encoder) and dynamic features $\mathcal { D } _ { t } = \{ c _ { t } , \bar { t } _ { t } , d _ { t } , o _ { t } , b _ { t } \}$ , where $c _ { t } , t _ { t } , d _ { t } , o _ { t } , b _ { t }$ represent the remaining linehaul capacity, current time, current route length, binary open route indicator and remaining backhaul capacity, respectively. At decoding step $t$ , the context embedding is computed as $h _ { t } ^ { c } = W _ { c } \cdot { \mathrm { C o n c a t } } ( h _ { t - 1 } ^ { N } , { \mathcal { D } } _ { t } )$ , where $W _ { c } \in \mathbb { R } ^ { d \times ( \bar { d } + 5 ) }$ is a learnable parameter matrix and $h _ { t - 1 } ^ { N }$ denotes the node embedding visited at step $t - 1$ . The context embedding is then updated via the MHA layer and the identity mapping function (IDT) (Huang et al., 2025): 
$$
h _ {t} ^ {c ^ {\prime}} = \operatorname {M H A} \left(h _ {t} ^ {c}, h ^ {N}, h ^ {N}\right) + \operatorname {I D T} \left(h _ {t} ^ {c}\right). \tag {18}
$$
3 
Published as a conference paper at ICLR 2026 
In MHA, $h _ { t } ^ { c }$ is used to compute queries, and $h ^ { N }$ is used to compute keys and values: 
$$
Q ^ {c, a} = W _ {Q} ^ {c, a} h _ {c} ^ {t}, \quad K ^ {c, a} = W _ {K} ^ {c, a} h ^ {N}, \quad V ^ {c, a} = W _ {V} ^ {c, a} h ^ {N}, \tag {19}
$$
where $W _ { Q } ^ { c , a } \ \in \ \mathbb { R } ^ { d _ { k } \times ( d + 5 ) }$ , $W _ { K } ^ { c , a }$ , W c,aV ∈ Rdk×d are learnable parameter matrices of the $a$ -th attention head. Then, the output of the MHA layer is obtained by Eqs. (13)-(15). The IDT function explicitly injects context information into $h _ { t } ^ { c \prime }$ , complementing the attention-based update, 
$$
\operatorname {I D T} \left(h _ {t} ^ {c}\right) = h _ {t - 1} ^ {N} + W ^ {\mathrm {I D T}} \mathcal {D} _ {t}, \tag {20}
$$
where $W ^ { \mathrm { I D T } } \in \mathbb { R } ^ { d \times 5 }$ is a learnable parameter matrix. $h _ { t } ^ { c \prime }$ is passed through an FFN layer (see Eq. 16) with a residual connection to generate the query $q _ { t } ^ { c }$ . The logits for all nodes are then computed as: 
$$
s _ {t} ^ {i} = \left\{ \begin{array}{l l} C \cdot \tanh  \left(\frac {\left(q _ {t} ^ {c}\right) ^ {T} h _ {i} ^ {N}}{\sqrt {d}}\right), & \text {i f} i \in \mathcal {F} _ {t} \\ - \infty , & \text {o t h e r w i s e} \end{array} \right. \tag {21}
$$
where $C = 1 0$ is a clipping hyperparameter that bounds the logits to promote exploration. $\mathcal { F } _ { t }$ represents the set of feasible nodes at step $t$ , defined by task-specific constraints. Finally, the node selection probability is computed using softmax over the logits. 
# C TRAINING AND FINE-TUNING DATASETS
# C.1 TRAINING DATASET
The training task set (see Section 5) follows the configuration in MVMoE (Zhou et al., 2024a), covering CVRP, OVRP, VRPB, VRPL, VRPTW, and OVRPTW tasks. In addition, we include the OVRPL task, which is motivated by two key reasons: (1) More complex instance generation. Instead of applying a fixed duration limit $L$ as in prior work (Liu et al., 2024b; Zhou et al., 2024a; Huang et al., 2025), we adopt a sampling-based strategy (Berto et al., 2024) to generate $L$ , resulting in more diverse and challenging instances. (2) Complex interaction between constraints O and L. During training, the VRP tasks involving O and L constraints consistently degrade generalization performance (see validation curves in Figure 5). This instability arises from complex interaction between the two constraints: O enlarges the solution space by removing the depot-return requirement, while L restricts it with strict route-length bounds, resulting in convergence difficulties. To address this, the OVRPL task is included in the training set and all models are retrained to ensure a fair and consistent comparison. 
# C.2 FINE-TUNING DATASET
This section explains the rationale for including the VRPMB and VRPMBTW tasks in the finetuning task set. Initially, all methods are fine-tuned solely on the VRPMB task, treating the remaining VRP tasks with MB as out-of-distribution (OOD) generalization targets. As shown in Figure 6, the generalization performance of VRP tasks involving both TW and MB degrades significantly. The results may arise from the spatio-temporal conflict between the flexible routing requirements of MB and the strict deadlines imposed by TW. To mitigate this, VRPMBTW is added to the finetuning task set. As a result, our method consistently improves performance across all tasks, whereas RF-TE and ReLD-MoEL exhibit convergence failures on several tasks. These findings underscore the challenges of adapting to unseen constraints and demonstrate the superior scalability of CoEKS. 
# D EFFECTS OF DIFFERENT TRAINING SETS
To further investigate the impact of CoEKS under different training datasets, we consider incorporating all possible combinations of basic constraints into the training set for VRP tasks, reflecting RouteFinder’s philosophy of establishing a foundational model for VRPs (Berto et al., 2024). To ensure a fair comparison, all methods are trained under the same settings. 
4 
Published as a conference paper at ICLR 2026 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/26e1b3ec8015fe5349bfb133098c8a001032ee247f560fa783ba8af241bca62c.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/2e37b50c74ef4e395928753e4b610e0f669e69f54edbb9c684ae349d4caa3c99.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/a17fb0d642e60ba498af6a67ded404dc4ba409a689fe74e093afa30b11f95d55.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/c17ce3f53a0681943ab9f731342716f804505f992ba83b8b3c9dc721c198cd6c.jpg)


Figure 5: The validation curves of ReLD-MoEL trained without (w/o) OVRPL and with (w.) OVRPL on $n = 5 0$ .

# D.1 TRAINING ON 16 VRP TASKS
The ID generalization results are reported in Table 5, where task types of the test set are identical to the training set. Among the neural methods, CoEKS delivers the best overall performance. CaDA shows strength on TW-constrained tasks but struggles to maintain competitive performance on the remaining variants. In contrast, CoEKS consistently achieves top ranks across all tasks, highlighting its superior ability to balance diverse VRP variants within a unified model. These results represent an in-distribution (ID) scenario, as all constraint combinations are included in the training set. 
Beyond this ID setting, CoEKS significantly outperforms CaDA in the OOD scenario (see Table 2), which underscores its potential as a foundation model capable of generalizing to unseen tasks. 
# D.2 FINE-TUNING ON ALL VRP TASKS WITH MB
To evaluate scalability to new constraints, we previously fine-tuned CoEKS on a small set of tasks with new constraints and evaluated generalization to all tasks. Building on this, we further consider fine-tuning on all VRP tasks with new constraints, aligning with RouteFinder. Following this setup, all methods are fine-tuned on all VRP tasks with MB. The results are presented in Table 6, where CoEKS consistently demonstrates superior performance across all tasks. Furthermore, the relative gap with comparative methods widens, suggesting that the new expert can further refine itself through interaction with diverse experts. This highlights CoEKS’s exceptional scalability to adapt a new constraint. 
# E ADDITIONAL EMPIRICAL RESULTS
# E.1 COMPARISON WITH MOE
Difference from MoE. 1) Semantically grounded routing: MoE architectures typically use learned gating to select top-k experts per node embedding, lacking semantic alignment or cross-task reuse. In contrast, CoEKS better leverages prior knowledge to combine experts, which is both interpretable and efficient. 2) Broader expert vision: Gating mechanism of MOE-based methods (Zhou et al., 2024a; Huang et al., 2025) restricts expert vision to a narrow node subset, which weakens experts’ cognition of task-level knowledge. In contrast, CoEKS effectively learns constraint-level knowledge through dedicated experts and understand task-level knowledge via combination of experts. 3) Stable utilization and load balancing: Existing MOE-based methods (Zhou et al., 2024a; Huang et al., 
5 
Published as a conference paper at ICLR 2026 

VRPMB

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/a1f06998bd6f219d943b26e4b66e3a5b4412f92909c560fb7b8c3c0c442c8406.jpg)


VRPMBL

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/af765546ee555e5440b3a9a59f8341334e1bb034f585f0b7714e8ee28cf1cdc8.jpg)


VRPMBTW

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/03ebac161db7ef6c4884231b376614fe9505e843d237cce062daae52f52e0ee8.jpg)


OVRPMB

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/0f1ea8fbcd8df2668b9e86c1194b6bd87eb8356d5680b82b8a9d72f6712880ca.jpg)


VRPMBLTW

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/eaccbd7a2d137f356094366606b12a42307229389baaa6d7c44783173bfe476d.jpg)


OVRPMBL

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/76e57a634f9da6d7bb61004017d661689c674ee07566a6ba416237cbaf01c1ac.jpg)


OVRPMBTW

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/a913decf1eaaa1873d8bdda7eec2b14add1e8643d840ff3c66b0e557a9d003a2.jpg)


OVRPMBLTW

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/e871b3442670c509ffa1ac320b184cdc89fb225b757d40dde5824ad8d78afd97.jpg)


Figure 6: The validation curves of CoEKS, RF-TE, and ReLD-MoEL trained without (w/o) VRPMBTW and with (w.) VRPMBTW on $n = 5 0$ .

2025) typically rely on gating mechanisms to route data to experts, which may lead to load balancing issues. In contrast, CoEKS explicitly activates experts based on task constraints, ensuring more stable and balanced expert utilization. 4) Scalability via plugging in new experts: New constraints can be handled by adding and fine-tuning a dedicated expert without modifying the rest of the trained model, as validated in Section 5 (Q2). This structural modularity offers practical advantages over MOE-based methods. 
Parameter efficiency analysis compared with MoE methods: When compared with MoE-based methods (e.g., ReLD-MoEL (Huang et al., 2025)), CoEKS shares the same decoder architecture, but CoEKS’s encoder has one additional expert. To validate CoEKS’s parameter efficiency, we use five experts in ReLD-MoEL-E5, resulting in a total of 4.6 million parameters, which matches that of CoEKS. During training, CoEKS activates an average of two experts, aligning with ReLD-MoEL-E5 that activates the top-2 experts. During inference, the number of activated experts in ReLD-MoEL-E5 is dynamically adjusted based on task constraints to maintain consistent parameter usage with CoEKS. The results are presented in Tables 7 and 8. Given the same total and activated parameters, 
6 
Published as a conference paper at ICLR 2026 

Table 5: Performance on 1K test instances of 16 VRPs (the training set includes all 16 VRPs).

<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="3">n = 50</td><td colspan="3">n = 100</td><td rowspan="2">Method</td><td colspan="3">n = 50</td><td colspan="3">n = 100</td></tr><tr><td>Obj.</td><td>Gap</td><td>Time</td><td>Obj.</td><td>Gap</td><td>Time</td><td>Obj.</td><td>Gap</td><td>Time</td><td>Obj.</td><td>Gap</td><td>Time</td></tr><tr><td rowspan="8">CVRP</td><td>HGS-PyVRP#</td><td>10.372</td><td>*</td><td>10.4m</td><td>15.628</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>16.031</td><td>*</td><td>10.4m</td><td>25.423</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>10.572</td><td>1.907%</td><td>10.4m</td><td>16.280</td><td>4.178%</td><td>20.8m</td><td>OR-Tools#</td><td>16.089</td><td>0.347%</td><td>10.4m</td><td>25.814</td><td>1.506%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>10.520</td><td>1.429%</td><td>1s</td><td>15.910</td><td>1.844%</td><td>7s</td><td>POMO-MTL</td><td>16.421</td><td>2.432%</td><td>1s</td><td>26.417</td><td>3.896%</td><td>7s</td></tr><tr><td>MVMOE</td><td>10.501</td><td>1.240%</td><td>2s</td><td>15.880</td><td>1.641%</td><td>9s</td><td>MVMOE</td><td>16.397</td><td>2.287%</td><td>2s</td><td>26.389</td><td>3.780%</td><td>9s</td></tr><tr><td>RF-TE</td><td>10.509</td><td>1.330%</td><td>1s</td><td>15.861</td><td>1.533%</td><td>7s</td><td>RF-TE</td><td>16.362</td><td>2.060%</td><td>1s</td><td>26.267</td><td>3.304%</td><td>7s</td></tr><tr><td>CaDA</td><td>10.505</td><td>1.281%</td><td>3s</td><td>15.856</td><td>1.489%</td><td>11s</td><td>CaDA</td><td>16.291</td><td>1.611%</td><td>2s</td><td>26.078</td><td>2.560%</td><td>12s</td></tr><tr><td>ReLD-MoEL</td><td>10.482</td><td>1.062%</td><td>2s</td><td>15.832</td><td>1.340%</td><td>9s</td><td>ReLD-MoEL</td><td>16.381</td><td>2.171%</td><td>2s</td><td>26.320</td><td>3.515%</td><td>9s</td></tr><tr><td>CoEKS</td><td>10.477</td><td>1.017%</td><td>2s</td><td>15.816</td><td>1.242%</td><td>9s</td><td>CoEKS</td><td>16.332</td><td>1.873%</td><td>2s</td><td>26.209</td><td>3.070%</td><td>9s</td></tr><tr><td rowspan="8">OVPR</td><td>HGS-PyVRP#</td><td>6.507</td><td>*</td><td>10.4m</td><td>9.725</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>10.587</td><td>*</td><td>10.4m</td><td>15.766</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>6.553</td><td>0.686%</td><td>10.4m</td><td>9.995</td><td>2.732%</td><td>20.8m</td><td>OR-Tools#</td><td>10.570</td><td>2.343%</td><td>10.4m</td><td>16.466</td><td>5.302%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>6.716</td><td>3.185%</td><td>1s</td><td>10.193</td><td>4.786%</td><td>6s</td><td>POMO-MTL</td><td>10.774</td><td>1.722%</td><td>1s</td><td>16.132</td><td>2.324%</td><td>6s</td></tr><tr><td>MVMOE</td><td>6.702</td><td>2.967%</td><td>2s</td><td>10.164</td><td>4.490%</td><td>9s</td><td>MVMOE</td><td>10.749</td><td>1.491%</td><td>2s</td><td>16.088</td><td>2.047%</td><td>9s</td></tr><tr><td>RF-TE</td><td>6.687</td><td>2.731%</td><td>1s</td><td>10.119</td><td>4.031%</td><td>6s</td><td>RF-TE</td><td>10.750</td><td>1.514%</td><td>1s</td><td>16.057</td><td>1.865%</td><td>6s</td></tr><tr><td>CaDA</td><td>6.684</td><td>2.679%</td><td>2s</td><td>10.116</td><td>3.987%</td><td>12s</td><td>CaDA</td><td>10.745</td><td>1.465%</td><td>2s</td><td>16.043</td><td>1.768%</td><td>11s</td></tr><tr><td>ReLD-MoEL</td><td>6.679</td><td>2.616%</td><td>2s</td><td>10.101</td><td>3.851%</td><td>9s</td><td>ReLD-MoEL</td><td>10.728</td><td>1.303%</td><td>2s</td><td>16.032</td><td>1.695%</td><td>9s</td></tr><tr><td>CoEKS</td><td>6.667</td><td>2.424%</td><td>2s</td><td>10.073</td><td>3.562%</td><td>8s</td><td>CoEKS</td><td>10.724</td><td>1.266%</td><td>2s</td><td>16.023</td><td>1.644%</td><td>9s</td></tr><tr><td rowspan="8">VRBP</td><td>HGS-PyVRP#</td><td>9.687</td><td>*</td><td>10.4m</td><td>14.377</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>10.510</td><td>*</td><td>10.4m</td><td>16.926</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>9.802</td><td>1.159%</td><td>10.4m</td><td>14.933</td><td>3.853%</td><td>20.8m</td><td>OR-Tools#</td><td>10.519</td><td>0.078%</td><td>10.4m</td><td>17.027</td><td>0.583%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>10.032</td><td>3.556%</td><td>1s</td><td>15.054</td><td>4.725%</td><td>6s</td><td>POMO-MTL</td><td>10.673</td><td>1.526%</td><td>1s</td><td>17.418</td><td>2.880%</td><td>7s</td></tr><tr><td>MVMOE</td><td>10.008</td><td>3.298%</td><td>2s</td><td>15.012</td><td>4.432%</td><td>8s</td><td>MVMOE</td><td>10.671</td><td>1.511%</td><td>2s</td><td>17.429</td><td>2.946%</td><td>10s</td></tr><tr><td>RF-TE</td><td>9.986</td><td>3.083%</td><td>1s</td><td>14.934</td><td>3.891%</td><td>6s</td><td>RF-TE</td><td>10.654</td><td>1.350%</td><td>1s</td><td>17.333</td><td>2.377%</td><td>7s</td></tr><tr><td>CaDA</td><td>9.978</td><td>2.987%</td><td>2s</td><td>14.932</td><td>3.873%</td><td>11s</td><td>CaDA</td><td>10.622</td><td>1.041%</td><td>2s</td><td>17.230</td><td>1.772%</td><td>12s</td></tr><tr><td>ReLD-MoEL</td><td>9.967</td><td>2.875%</td><td>2s</td><td>14.921</td><td>3.799%</td><td>9s</td><td>ReLD-MoEL</td><td>10.658</td><td>1.391%</td><td>2s</td><td>17.368</td><td>2.593%</td><td>10s</td></tr><tr><td>CoEKS</td><td>9.948</td><td>2.674%</td><td>2s</td><td>14.884</td><td>3.546%</td><td>9s</td><td>CoEKS</td><td>10.638</td><td>1.192%</td><td>2s</td><td>17.313</td><td>2.263%</td><td>10s</td></tr><tr><td rowspan="8">VRBPBL</td><td>HGS-PyVRP#</td><td>10.186</td><td>*</td><td>10.4m</td><td>14.779</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>18.361</td><td>*</td><td>10.4m</td><td>29.026</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>10.331</td><td>1.390%</td><td>10.4m</td><td>15.426</td><td>4.338%</td><td>20.8m</td><td>OR-Tools#</td><td>18.422</td><td>0.332%</td><td>10.4m</td><td>29.830</td><td>2.770%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>10.675</td><td>4.733%</td><td>1s</td><td>15.688</td><td>6.103%</td><td>7s</td><td>POMO-MTL</td><td>19.001</td><td>2.186%</td><td>1s</td><td>30.934</td><td>3.740%</td><td>7s</td></tr><tr><td>MVMOE</td><td>10.632</td><td>4.309%</td><td>2s</td><td>15.621</td><td>5.643%</td><td>9s</td><td>MVMOE</td><td>18.981</td><td>2.083%</td><td>2s</td><td>30.905</td><td>3.648%</td><td>10s</td></tr><tr><td>RF-TE</td><td>10.584</td><td>3.856%</td><td>1s</td><td>15.515</td><td>4.950%</td><td>7s</td><td>RF-TE</td><td>18.942</td><td>1.885%</td><td>1s</td><td>30.719</td><td>3.026%</td><td>7s</td></tr><tr><td>CaDA</td><td>10.569</td><td>3.708%</td><td>2s</td><td>15.506</td><td>4.872%</td><td>11s</td><td>CaDA</td><td>18.858</td><td>1.432%</td><td>2s</td><td>30.531</td><td>2.393%</td><td>13s</td></tr><tr><td>ReLD-MoEL</td><td>10.567</td><td>3.675%</td><td>2s</td><td>15.498</td><td>4.828%</td><td>9s</td><td>ReLD-MoEL</td><td>18.959</td><td>1.966%</td><td>2s</td><td>30.800</td><td>3.299%</td><td>10s</td></tr><tr><td>CoEKS</td><td>10.546</td><td>3.480%</td><td>2s</td><td>15.466</td><td>4.621%</td><td>9s</td><td>CoEKS</td><td>18.913</td><td>1.728%</td><td>2s</td><td>30.680</td><td>2.896%</td><td>10s</td></tr><tr><td rowspan="8">VRBPBLW</td><td>HGS-PyVRP#</td><td>18.292</td><td>*</td><td>10.4m</td><td>29.467</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>16.356</td><td>*</td><td>10.4m</td><td>25.757</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>18.366</td><td>0.383%</td><td>10.4m</td><td>29.945</td><td>1.597%</td><td>20.8m</td><td>OR-Tools#</td><td>16.441</td><td>0.499%</td><td>10.4m</td><td>26.259</td><td>1.899%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>18.647</td><td>1.915%</td><td>1s</td><td>30.447</td><td>3.324%</td><td>7s</td><td>POMO-MTL</td><td>16.833</td><td>2.886%</td><td>1s</td><td>26.895</td><td>4.379%</td><td>7s</td></tr><tr><td>MVMOE</td><td>18.637</td><td>1.863%</td><td>2s</td><td>30.439</td><td>3.292%</td><td>10s</td><td>MVMOE</td><td>16.804</td><td>2.712%</td><td>2s</td><td>26.858</td><td>4.234%</td><td>9s</td></tr><tr><td>RF-TE</td><td>18.604</td><td>1.685%</td><td>1s</td><td>30.265</td><td>2.702%</td><td>7s</td><td>RF-TE</td><td>16.751</td><td>2.389%</td><td>1s</td><td>26.717</td><td>3.690%</td><td>7s</td></tr><tr><td>CaDA</td><td>18.519</td><td>1.227%</td><td>2s</td><td>30.080</td><td>2.064%</td><td>12s</td><td>CaDA</td><td>16.682</td><td>1.964%</td><td>2s</td><td>26.525</td><td>2.945%</td><td>13s</td></tr><tr><td>ReLD-MoEL</td><td>18.611</td><td>1.725%</td><td>2s</td><td>30.349</td><td>2.986%</td><td>10s</td><td>ReLD-MoEL</td><td>16.767</td><td>2.496%</td><td>2s</td><td>26.768</td><td>3.894%</td><td>10s</td></tr><tr><td>CoEKS</td><td>18.567</td><td>1.489%</td><td>2s</td><td>30.213</td><td>2.523%</td><td>10s</td><td>CoEKS</td><td>16.737</td><td>2.298%</td><td>2s</td><td>26.673</td><td>3.517%</td><td>10s</td></tr><tr><td rowspan="8">OVRBPBLW</td><td>HGS-PyVRP#</td><td>6.898</td><td>*</td><td>10.4m</td><td>10.335</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>6.899</td><td>*</td><td>10.4m</td><td>10.335</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>6.928</td><td>0.412%</td><td>10.4m</td><td>10.577</td><td>2.315%</td><td>20.8m</td><td>OR-Tools#</td><td>6.927</td><td>0.386%</td><td>10.4m</td><td>10.582</td><td>2.363%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>7.106</td><td>2.989%</td><td>1s</td><td>10.852</td><td>4.973%</td><td>7s</td><td>POMO-MTL</td><td>7.111</td><td>3.046%</td><td>1s</td><td>10.863</td><td>5.081%</td><td>7s</td></tr><tr><td>MVMOE</td><td>7.086</td><td>2.696%</td><td>2s</td><td>10.825</td><td>4.707%</td><td>9s</td><td>MVMOE</td><td>7.096</td><td>2.818%</td><td>2s</td><td>10.832</td><td>4.778%</td><td>9s</td></tr><tr><td>RF-TE</td><td>7.075</td><td>2.538%</td><td>1s</td><td>10.769</td><td>4.179%</td><td>6s</td><td>RF-TE</td><td>7.079</td><td>2.580%</td><td>1s</td><td>10.772</td><td>4.207%</td><td>7s</td></tr><tr><td>CaDA</td><td>7.062</td><td>2.341%</td><td>2s</td><td>10.745</td><td>3.941%</td><td>12s</td><td>CaDA</td><td>7.064</td><td>2.362%</td><td>2s</td><td>10.748</td><td>3.962%</td><td>11s</td></tr><tr><td>ReLD-MoEL</td><td>7.064</td><td>2.375%</td><td>2s</td><td>10.754</td><td>4.034%</td><td>9s</td><td>ReLD-MoEL</td><td>7.063</td><td>2.354%</td><td>2s</td><td>10.750</td><td>3.990%</td><td>9s</td></tr><tr><td>CoEKS</td><td>7.048</td><td>2.139%</td><td>2s</td><td>10.710</td><td>3.611%</td><td>9s</td><td>CoEKS</td><td>7.050</td><td>2.155%</td><td>2s</td><td>10.711</td><td>3.616%</td><td>9s</td></tr><tr><td rowspan="8">OVRBPBLW</td><td>HGS-PyVRP#</td><td>11.668</td><td>*</td><td>10.4m</td><td>19.156</td><td>*</td><td>20.8m</td><td>HGS-PyVRP#</td><td>11.669</td><td>*</td><td>10.4m</td><td>19.156</td><td>*</td><td>20.8m</td></tr><tr><td>OR-Tools#</td><td>11.681</td><td>0.106%</td><td>10.4m</td><td>19.305</td><td>0.767%</td><td>20.8m</td><td>OR-Tools#</td><td>11.682</td><td>0.109%</td><td>10.4m</td><td>19.303</td><td>0.728%</td><td>20.8m</td></tr><tr><td>POMO-MTL</td><td>11.823</td><td>1.304%</td><td>1s</td><td>19.635</td><td>2.482%</td><td>7s</td><td>POMO-MTL</td><td>11.821</td><td>1.293%</td><td>1s</td><td>19.631</td><td>2.464%</td><td>7s</td></tr><tr><td>MVMOE</td><td>11.815</td><td>1.244%</td><td>2s</td><td>19.657</td><td>2.603%</td><td>10s</td><td>MVMOE</td><td>11.814</td><td>1.231%</td><td>2s</td><td>19.654</td><td>2.587%</td><td>10s</td></tr><tr><td>RF-TE</td><td>11.804</td><td>1.145%</td><td>1s</td><td>19.552</td><td>2.049%</td><td>7s</td><td>RF-TE</td><td>11.804</td><td>1.140%</td><td>1s</td><td>19.551</td><td>2.045%</td><td>8s</td></tr><tr><td>CaDA</td><td>11.767</td><td>0.832%</td><td>3s</td><td>19.434</td><td>1.430%</td><td>13s</td><td>CaDA</td><td>11.766</td><td>0.828%</td><td>3s</td><td>19.435</td><td>1.432%</td><td>13s</td></tr><tr><td>ReLD-MoEL</td><td>11.807</td><td>1.170%</td><td>2s</td><td>19.586</td><td>2.228%</td><td>11s</td><td>ReLD-MoEL</td><td>11.808</td><td>1.177%</td><td>2s</td><td>19.586</td><td>2.230%</td><td>11s</td></tr><tr><td>CoEKS</td><td>11.789</td><td>1.024%</td><td>2s</td><td>19.541</td><td>1.990%</td><td>10s</td><td>CoEKS</td><td>11.791</td><td>1.033%</td><td>2s</td><td>19.539</td><td>1.980%</td><td>10s</td></tr></table>

bold: Best results among learning-based methods. 


underline: Second-best results among learning-based methods. 


#: Results are adopted from Berto et al. (2024) for the convenience of comparison. 

CoEKS consistently outperforms ReLD-MoEL-E5 across all tasks, in both ID and OOD scenarios. 
This underscores that CoEKS achieves superior parameter efficiency through its model architecture. 
# E.2 T-SNE VISUALIZATION ANALYSIS
To gain insights into how the experts in CoEKS learn and specialize, we visualize their embedding tokens using t-distributed Stochastic Neighbor Embedding (t-SNE). We performed t-SNE analysis 
Published as a conference paper at ICLR 2026 

Table 6: Fine-tuning performance on all VRPs with MB.

<table><tr><td rowspan="2">Method</td><td colspan="2">VRPMB</td><td colspan="2">OVRPMB</td><td colspan="2">VRPMBL</td><td colspan="2">VRPMBTW</td><td colspan="2">OVRPMBL</td><td colspan="2">OVRPMBTW</td><td colspan="2">VRPMBLTW</td><td colspan="2">OVRPMBLTW</td></tr><tr><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td><td>Cost</td><td>Gap</td></tr><tr><td>HGS-PyVRP</td><td>9.09</td><td>*</td><td>6.11</td><td>*</td><td>16.31</td><td>*</td><td>9.49</td><td>*</td><td>6.11</td><td>*</td><td>10.47</td><td>*</td><td>16.01</td><td>*</td><td>10.47</td><td>*</td></tr><tr><td>RF-TE-AL</td><td>11.74</td><td>29.66%</td><td>9.44</td><td>54.62%</td><td>11.27</td><td>18.94%</td><td>18.46</td><td>15.59%</td><td>8.58</td><td>40.46%</td><td>13.27</td><td>27.14%</td><td>18.88</td><td>16.06%</td><td>13.29</td><td>27.33%</td></tr><tr><td>RF-TE-EAL</td><td>9.36</td><td>2.98%</td><td>6.26</td><td>2.47%</td><td>9.74</td><td>2.67%</td><td>16.40</td><td>2.38%</td><td>6.26</td><td>2.42%</td><td>10.66</td><td>1.73%</td><td>16.79</td><td>2.94%</td><td>10.66</td><td>1.76%</td></tr><tr><td>ReLD-MoEL-AL</td><td>10.63</td><td>17.07%</td><td>8.13</td><td>33.15%</td><td>11.09</td><td>16.99%</td><td>18.22</td><td>13.88%</td><td>8.21</td><td>34.49%</td><td>12.61</td><td>20.60%</td><td>18.64</td><td>14.39%</td><td>12.64</td><td>20.93%</td></tr><tr><td>ReLD-MoEL-EAL</td><td>9.34</td><td>2.73%</td><td>6.28</td><td>2.66%</td><td>9.71</td><td>2.28%</td><td>16.39</td><td>2.34%</td><td>6.28</td><td>2.73%</td><td>10.66</td><td>1.75%</td><td>16.78</td><td>2.86%</td><td>10.66</td><td>1.80%</td></tr><tr><td>CoEKS+</td><td>9.25</td><td>1.73%</td><td>6.21</td><td>1.68%</td><td>9.68</td><td>1.97%</td><td>16.35</td><td>2.07%</td><td>6.22</td><td>1.75%</td><td>10.63</td><td>1.44%</td><td>16.74</td><td>2.65%</td><td>10.64</td><td>1.54%</td></tr><tr><td>CoEKSc+</td><td>9.24</td><td>1.66%</td><td>6.20</td><td>1.50%</td><td>9.66</td><td>1.81%</td><td>16.35</td><td>2.07%</td><td>6.21</td><td>1.57%</td><td>10.63</td><td>1.43%</td><td>16.73</td><td>2.56%</td><td>10.63</td><td>1.50%</td></tr></table>

Table 7: Parameter efficiency comparison. (In-distribution tasks $( n = 5 0 )$ ))

<table><tr><td>Method\Gap↓</td><td>CVRP</td><td>OVRP</td><td>VRPB</td><td>VRPL</td><td>VRPTW</td><td>OVRPTW</td><td>OVRPL</td><td>(ID Avg.)</td></tr><tr><td>ReLD-MoEL-E5</td><td>1.086%</td><td>2.324%</td><td>2.509%</td><td>1.180%</td><td>2.308%</td><td>1.628%</td><td>2.465%</td><td>2.069%</td></tr><tr><td>CoEKS</td><td>0.891%</td><td>2.138%</td><td>2.497%</td><td>1.152%</td><td>2.050%</td><td>1.393%</td><td>2.135%</td><td>1.751%</td></tr></table>

Table 8: Parameter efficiency comparison. (Out-of-distribution tasks $\mathit { \Delta } n = 5 0 $ ))

<table><tr><td>Method\Gap↓</td><td>OVRPB</td><td>VRPBL</td><td>VRPBTW</td><td>VRPLTW</td><td>OVRPBL</td><td>OVRPBTW</td><td>OVRPLTW</td><td>VRPBLTW</td><td>OVRPBLTW</td><td>(OOD Avg.)</td></tr><tr><td>ReLD-MoEL-E5</td><td>6.753%</td><td>4.323%</td><td>3.609%</td><td>4.491%</td><td>6.738%</td><td>2.673%</td><td>1.780%</td><td>6.262%</td><td>3.026%</td><td>4.406%</td></tr><tr><td>CoEKS</td><td>4.913%</td><td>4.387%</td><td>3.210%</td><td>2.949%</td><td>4.747%</td><td>2.566%</td><td>1.603%</td><td>3.714%</td><td>2.797%</td><td>3.432%</td></tr></table>
on the expert embeddings for the full-constraint task OVRPBLTW. For each expert at each encoder layer, we sampled 5,000 embedding samples and projected them into a 2D space for visualization. 
As illustrated in Figure 7, the t-SNE plots reveal a significant overlap in the embeddings of all experts, except for the capacity expert $( E _ { C } )$ , within the lower layers (layer 1). This indicates that the lower layers capture shared, transferable representations, aligning with our design where knowledge sharing is applied at the lower layers. The capacity expert $( E _ { C } )$ , which is active across all VRP tasks, learns more universal representations, resulting in its distinct and stable representation early in the model. As we move to deeper layers, the clusters become clearly separated, showing that experts gradually specialize and align with their assigned constraints. 
We also visualize the expert embeddings for the variant without the knowledge sharing strategy in Figure 8. In this case, experts’ representations begin to separate even in the first layer, indicating early specialization and a lack of transferable knowledge. This confirms that our knowledge sharing strategy is necessary for building meaningful cross-task representations. 
# E.3 SCALABILITY TO CONSTRAINT MULTI-DEPOTS
To further verify the scalability of CoEKS to new constraints, we introduced the Multi-Depot (MD) tasks, which is an extension of the single-depot tasks. Following the configuration of RouteFinder, we set the number of depots to 3. In total, we added 24 new VRP tasks, which include: 
? 16 tasks that incorporate only the new MD constraint. 
? 8 tasks that combine the new MD with the new MB constraint. 
The experimental results are presented in Tables 9, 10, Table 11and 12, where CoEKS continues to achieve the best performance under both few-shot fine-tuning and zero-shot generalization. These findings strongly support our claim that CoEKS is robustly scalable to diverse and previously unseen constraints. 
# E.4 CVRPLIB BENCHMARK
On the CVRPLIB benchmark dataset, we conduct additional experiments to evaluate the generalization ability of the model on real-world instances. Several representative cross-task methods are compared as a supplementary validation of OOD generalization performance. Each model is trained on uniformly distributed instances with $n = 1 0 0$ . Additionally, the original POMO model (Kwon et al., 2020) trained on a single task is also reported. The evaluation primarily focuses on large-scale datasets $\mathrm { ( } n > 5 0 0 \mathrm { ) }$ ) in the classic Set-X (Uchoa et al., 2017), following the setup in MVMoE (Zhou 
8 
Published as a conference paper at ICLR 2026 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/486091a146f37d2ac43bc09b1df0ee29f92e053f44f82fd78e356f5946c3d3ab.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/0fb0554d85552d760550d6fc9d3ca551cd479c09a56b40be6a05a0f8b5a876a4.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/8aa80e813fb3bdc8b847ed2dc3a82d404de6717d81f281e3a28df5e56d9832b7.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/cba8f8a7dcb137eac132b00a37b3ababd6112097e0099d71b79a6ae0d38f94b3.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/dc3113fe8f67f4b9873dd7880d6e6e4d915b86bfca149a6c0f9c33ed38283265.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/6b2167fcecbfe512a6f805a1fc600165d8755acf08aabee5fae1ebedaea36c5a.jpg)


Figure 7: t-SNE visualization of 5 experts’ latent representations across encoder layers.


Table 9: Zero-shot generalization performance (the gap to the best traditional solver) on 24 VRPs with MD. These methods do not add new experts and are only allowed to activate experts corresponding to previously known constraints.

<table><tr><td>Method\Gap↓</td><td>MDCVRP</td><td>MDOVRP</td><td>MDOVRPB</td><td>MDOVRPBL</td><td>MDOVRPBLTW</td><td>MDOVRPBTW</td><td>MDOVRPL</td><td>MDOVRPLTW</td></tr><tr><td>RF-EAL</td><td>37.514%</td><td>29.887%</td><td>37.903%</td><td>40.128%</td><td>38.314%</td><td>45.111%</td><td>29.934%</td><td>34.305%</td></tr><tr><td>ReLD-MoEL</td><td>39.359%</td><td>20.266%</td><td>33.504%</td><td>32.543%</td><td>25.134%</td><td>28.718%</td><td>20.108%</td><td>25.210%</td></tr><tr><td>CoEKS</td><td>38.174%</td><td>15.678%</td><td>22.690%</td><td>21.883%</td><td>21.920%</td><td>22.901%</td><td>15.616%</td><td>20.949%</td></tr><tr><td>Method\Gap↓</td><td>MDOVRPTW</td><td>MDVRPB</td><td>MDVRPBL</td><td>MDVRPBLTW</td><td>MDVRPBTW</td><td>MDVRPL</td><td>MDVRPLTW</td><td>MDVRPTW</td></tr><tr><td>RF-EAL</td><td>40.836%</td><td>45.689%</td><td>59.955%</td><td>42.723%</td><td>46.764%</td><td>43.181%</td><td>39.398%</td><td>42.167%</td></tr><tr><td>ReLD-MoEL</td><td>28.634%</td><td>48.687%</td><td>47.975%</td><td>30.311%</td><td>39.094%</td><td>39.211%</td><td>30.267%</td><td>36.833%</td></tr><tr><td>CoEKS</td><td>21.730%</td><td>38.612%</td><td>38.188%</td><td>27.884%</td><td>31.094%</td><td>36.288%</td><td>26.986%</td><td>28.731%</td></tr></table>

Table 10: Zero-shot generalization performance (the gap to the best traditional solver) on 8 VRPs with MB with MD. These methods do not add new experts and are only allowed to activate experts corresponding to previously known constraints.

<table><tr><td>Method\Gap↓</td><td>MDOVRPMB</td><td>MDOVRPMBL</td><td>MDOVRPMBLTW</td><td>MDOVRPMBTW</td><td>MDVRPMB</td><td>MDVRPMBL</td><td>MDVRPMBLTW</td><td>MDVRPMBTW</td></tr><tr><td>RF-EAL</td><td>44.410%</td><td>44.523%</td><td>38.577%</td><td>38.608%</td><td>58.138%</td><td>57.110%</td><td>41.310%</td><td>41.821%</td></tr><tr><td>ReLD-MoEL</td><td>47.741%</td><td>45.581%</td><td>26.322%</td><td>29.812%</td><td>64.400%</td><td>60.381%</td><td>31.776%</td><td>38.646%</td></tr><tr><td>CoEKS</td><td>35.186%</td><td>33.701%</td><td>24.938%</td><td>25.365%</td><td>53.428%</td><td>50.735%</td><td>30.900%</td><td>32.874%</td></tr></table>
9 
Published as a conference paper at ICLR 2026 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/3c33c6e4d7111e38101acb4d4fe7ee113f698c02a1e7d8f22faef740f37312e5.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/65726df7a2125965bab4bd8252f85242a0a752a440bf225844ad6381e169e79c.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/738ef4706e116d01deadf432ab550a50080d70bc2fc2741803f51f134ab04de0.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/e9834a1f72a65fb7298a3f270065830c9915663d77576886f77b23035924b4ee.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/353fedba1bf05095179187cad047bf64c6c80a99b465d8021287519fd12042fd.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-25/1feda2c7-d3a7-49e8-b0b0-1577686023a4/e4378cabd721bcc88ba01d2020d22d11760432130b786ac5cfdb120b5424dca0.jpg)


Figure 8: t-SNE visualization of 5 experts’ latent representations across encoder layers (without knowledge sharing strategy)


Table 11: Fine-tuning performance (the gap to the best traditional solver) on 24 VRPs with MD.

<table><tr><td>Method\Gap↓</td><td>MDCVRP</td><td>MDOVRP</td><td>MDOVRPB</td><td>MDOVRPBL</td><td>MDOVRPBLTW</td><td>MDOVRPBTW</td><td>MDOVRPL</td><td>MDOVRPLTW</td></tr><tr><td>RF-EAL</td><td>15.251%</td><td>14.250%</td><td>18.996%</td><td>19.479%</td><td>19.841%</td><td>19.315%</td><td>15.421%</td><td>17.977%</td></tr><tr><td>ReLD-MoEL</td><td>15.235%</td><td>13.525%</td><td>17.125%</td><td>16.933%</td><td>11.405%</td><td>11.278%</td><td>13.333%</td><td>10.758%</td></tr><tr><td>CoEKS</td><td>8.928%</td><td>6.491%</td><td>7.599%</td><td>7.588%</td><td>5.324%</td><td>5.318%</td><td>6.543%</td><td>5.038%</td></tr><tr><td>Method\Gap↓</td><td>MDOVRPTW</td><td>MDVRPB</td><td>MDVRBL</td><td>MDVRPBLTW</td><td>MDVRPBTW</td><td>MDVRPL</td><td>MDVRPLTW</td><td>MDVRPTW</td></tr><tr><td>RF-EAL</td><td>17.300%</td><td>21.943%</td><td>23.149%</td><td>25.549%</td><td>24.098%</td><td>15.537%</td><td>21.952%</td><td>20.244%</td></tr><tr><td>ReLD-MoEL</td><td>10.622%</td><td>20.408%</td><td>20.572%</td><td>15.326%</td><td>15.340%</td><td>15.315%</td><td>14.502%</td><td>13.966%</td></tr><tr><td>CoEKS</td><td>4.978%</td><td>12.093%</td><td>12.487%</td><td>9.532%</td><td>9.324%</td><td>9.109%</td><td>8.970%</td><td>8.642%</td></tr></table>

Table 12: Fine-tuning performance (the gap to the best traditional solver) on 8 VRPs with MB with MD.

<table><tr><td>Method\Gap↓</td><td>MDOVRPMB</td><td>MDOVRPMBL</td><td>MDOVRPMBLTW</td><td>MDOVRPMBTW</td><td>MDVRPMB</td><td>MDVRPMBL</td><td>MDVRPMBLTW</td><td>MDVRPMBTW</td></tr><tr><td>RF-EAL</td><td>16.536%</td><td>18.378%</td><td>18.301%</td><td>17.592%</td><td>33.316%</td><td>21.222%</td><td>21.698%</td><td>19.965%</td></tr><tr><td>ReLD-MoEL</td><td>17.790%</td><td>17.543%</td><td>11.430%</td><td>11.206%</td><td>23.673%</td><td>22.101%</td><td>15.081%</td><td>14.417%</td></tr><tr><td>CoEKS</td><td>6.959%</td><td>7.061%</td><td>5.250%</td><td>5.130%</td><td>12.128%</td><td>12.032%</td><td>9.122%</td><td>8.804%</td></tr></table>
et al., 2024a) and RouteFinder (Berto et al., 2024). CoEKS achieves the best OOD generalization, surpassing the state-of-the-art ReLD-MoEL (Huang et al., 2025). The performance gains be-
10 
Published as a conference paper at ICLR 2026 
come more pronounced as the problem size increases. Notably, the single-task training method (i.e., POMO) demonstrates limited generalization ability on diverse real-world benchmarks, potentially due to overfitting the uniform training distribution. Conversely, cross-task training substantially enhances model generalization. 

Table 13: Results on large-scale CVRPLIB instances. # Results are adopted from MVMoE(Zhou et al., 2024a), with the model trained on a single task.

<table><tr><td rowspan="2">Set-X Instance</td><td colspan="2">POMO#</td><td colspan="2">RF-TE</td><td colspan="2">POMO-MTL</td><td colspan="2">MVMoE</td><td colspan="2">ReLD-MoEL</td><td colspan="2">CoEKS</td></tr><tr><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td><td>Obj.</td><td>Gap</td></tr><tr><td>X-n502-k39</td><td>75617</td><td>9.232%</td><td>72098</td><td>4.149%</td><td>84021</td><td>21.372%</td><td>81611</td><td>17.891%</td><td>73073</td><td>5.557%</td><td>73947</td><td>6.820%</td></tr><tr><td>X-n513-k21</td><td>30518</td><td>26.102%</td><td>30330</td><td>25.325%</td><td>29022</td><td>19.921%</td><td>27368</td><td>13.086%</td><td>27063</td><td>11.826%</td><td>27124</td><td>12.078%</td></tr><tr><td>X-n524-k153</td><td>201877</td><td>30.586%</td><td>168473</td><td>8.978%</td><td>173838</td><td>12.449%</td><td>174427</td><td>12.830%</td><td>174430</td><td>12.832%</td><td>174325</td><td>12.764%</td></tr><tr><td>X-n536-k96</td><td>106073</td><td>11.837%</td><td>102320</td><td>7.880%</td><td>106851</td><td>12.657%</td><td>105167</td><td>10.882%</td><td>102548</td><td>8.121%</td><td>103497</td><td>9.121%</td></tr><tr><td>X-n548-k50</td><td>103093</td><td>18.908%</td><td>102078</td><td>17.737%</td><td>102217</td><td>17.897%</td><td>107767</td><td>24.299%</td><td>99121</td><td>14.326%</td><td>103115</td><td>18.933%</td></tr><tr><td>X-n561-k42</td><td>49370</td><td>15.575%</td><td>49632</td><td>16.188%</td><td>48553</td><td>13.662%</td><td>47759</td><td>11.803%</td><td>47022</td><td>10.078%</td><td>46838</td><td>9.647%</td></tr><tr><td>X-n573-k30</td><td>83545</td><td>64.871%</td><td>55296</td><td>9.123%</td><td>60870</td><td>20.123%</td><td>66531</td><td>31.295%</td><td>57249</td><td>12.977%</td><td>54699</td><td>7.945%</td></tr><tr><td>X-n586-k159</td><td>229887</td><td>20.792%</td><td>208397</td><td>9.501%</td><td>211421</td><td>11.089%</td><td>214247</td><td>12.574%</td><td>206793</td><td>8.658%</td><td>205612</td><td>8.037%</td></tr><tr><td>X-n599-k92</td><td>150572</td><td>38.839%</td><td>117226</td><td>8.091%</td><td>122028</td><td>12.519%</td><td>126915</td><td>17.025%</td><td>116463</td><td>7.388%</td><td>116547</td><td>7.465%</td></tr><tr><td>X-n613-k62</td><td>68451</td><td>14.976%</td><td>68066</td><td>14.329%</td><td>82141</td><td>37.971%</td><td>67944</td><td>14.124%</td><td>67272</td><td>12.996%</td><td>66050</td><td>10.943%</td></tr><tr><td>X-n627-k43</td><td>84434</td><td>35.825%</td><td>69046</td><td>11.071%</td><td>70923</td><td>14.090%</td><td>70572</td><td>13.526%</td><td>68141</td><td>9.615%</td><td>67571</td><td>8.698%</td></tr><tr><td>X-n641-k35</td><td>75573</td><td>18.672%</td><td>73071</td><td>14.740%</td><td>72378</td><td>13.652%</td><td>70445</td><td>10.616%</td><td>69360</td><td>8.913%</td><td>68650</td><td>7.798%</td></tr><tr><td>X-n655-k131</td><td>127211</td><td>19.134%</td><td>112355</td><td>5.221%</td><td>123144</td><td>15.325%</td><td>126352</td><td>18.329%</td><td>120650</td><td>12.989%</td><td>113905</td><td>6.673%</td></tr><tr><td>X-n670-k130</td><td>208079</td><td>42.197%</td><td>167786</td><td>14.661%</td><td>167131</td><td>14.214%</td><td>168834</td><td>15.377%</td><td>169163</td><td>15.602%</td><td>167007</td><td>14.129%</td></tr><tr><td>X-n685-k75</td><td>79482</td><td>16.534%</td><td>77681</td><td>13.893%</td><td>99452</td><td>45.813%</td><td>78080</td><td>14.478%</td><td>78090</td><td>14.493%</td><td>76402</td><td>12.018%</td></tr><tr><td>X-n701-k44</td><td>97843</td><td>19.433%</td><td>92541</td><td>12.961%</td><td>90283</td><td>10.205%</td><td>89840</td><td>9.664%</td><td>87883</td><td>7.275%</td><td>87862</td><td>7.249%</td></tr><tr><td>X-n716-k35</td><td>51381</td><td>18.463%</td><td>50333</td><td>16.047%</td><td>49420</td><td>13.942%</td><td>50218</td><td>15.782%</td><td>47981</td><td>10.624%</td><td>47793</td><td>10.191%</td></tr><tr><td>X-n733-k159</td><td>159098</td><td>16.823%</td><td>162059</td><td>18.997%</td><td>184714</td><td>35.633%</td><td>153087</td><td>12.409%</td><td>153884</td><td>12.995%</td><td>150508</td><td>10.516%</td></tr><tr><td>X-n749-k98</td><td>87786</td><td>13.611%</td><td>85623</td><td>10.812%</td><td>88493</td><td>14.526%</td><td>86961</td><td>12.543%</td><td>86380</td><td>11.791%</td><td>84974</td><td>9.972%</td></tr><tr><td>X-n766-k71</td><td>135464</td><td>18.395%</td><td>132819</td><td>16.083%</td><td>127674</td><td>11.587%</td><td>129107</td><td>12.839%</td><td>126139</td><td>10.245%</td><td>125801</td><td>9.950%</td></tr><tr><td>X-n783-k48</td><td>90289</td><td>24.733%</td><td>86445</td><td>19.422%</td><td>84220</td><td>16.348%</td><td>82163</td><td>13.507%</td><td>80269</td><td>10.890%</td><td>79444</td><td>9.751%</td></tr><tr><td>X-n801-k40</td><td>124278</td><td>69.536%</td><td>92149</td><td>25.696%</td><td>96438</td><td>31.546%</td><td>88091</td><td>20.161%</td><td>85315</td><td>16.374%</td><td>86477</td><td>17.959%</td></tr><tr><td>X-n819-k171</td><td>193451</td><td>22.344%</td><td>187863</td><td>18.810%</td><td>188537</td><td>19.236%</td><td>187714</td><td>18.715%</td><td>175282</td><td>10.853%</td><td>173464</td><td>9.703%</td></tr><tr><td>X-n837-k142</td><td>237884</td><td>22.787%</td><td>209629</td><td>8.203%</td><td>218437</td><td>12.749%</td><td>223912</td><td>15.575%</td><td>210889</td><td>8.853%</td><td>208673</td><td>7.709%</td></tr><tr><td>X-n856-k95</td><td>152528</td><td>71.447%</td><td>99082</td><td>11.372%</td><td>157894</td><td>77.479%</td><td>175074</td><td>96.790%</td><td>100320</td><td>12.763%</td><td>98740</td><td>10.987%</td></tr><tr><td>X-n876-k59</td><td>119764</td><td>20.609%</td><td>109566</td><td>10.339%</td><td>110488</td><td>11.268%</td><td>115516</td><td>16.331%</td><td>106631</td><td>7.384%</td><td>106684</td><td>7.437%</td></tr><tr><td>X-n895-k37</td><td>70245</td><td>30.421%</td><td>67995</td><td>26.244%</td><td>67527</td><td>25.375%</td><td>64649</td><td>20.032%</td><td>62172</td><td>15.433%</td><td>61740</td><td>14.631%</td></tr><tr><td>X-n916-k207</td><td>399372</td><td>21.324%</td><td>354011</td><td>7.544%</td><td>382125</td><td>16.084%</td><td>372237</td><td>13.080%</td><td>355853</td><td>8.103%</td><td>352206</td><td>6.995%</td></tr><tr><td>X-n936-k151</td><td>237625</td><td>79.049%</td><td>164931</td><td>24.275%</td><td>193030</td><td>45.447%</td><td>160648</td><td>21.047%</td><td>160460</td><td>20.906%</td><td>158551</td><td>19.467%</td></tr><tr><td>X-n957-k87</td><td>130850</td><td>53.104%</td><td>110516</td><td>29.311%</td><td>108401</td><td>26.837%</td><td>127388</td><td>49.053%</td><td>101629</td><td>18.913%</td><td>103700</td><td>21.336%</td></tr><tr><td>X-n979-k58</td><td>147687</td><td>24.132%</td><td>133825</td><td>12.481%</td><td>134759</td><td>13.266%</td><td>132546</td><td>11.406%</td><td>129738</td><td>9.046%</td><td>129074</td><td>8.487%</td></tr><tr><td>X-n1001-k43</td><td>100399</td><td>38.759%</td><td>92837</td><td>28.308%</td><td>89098</td><td>23.140%</td><td>86107</td><td>19.006%</td><td>81081</td><td>12.060%</td><td>80458</td><td>11.199%</td></tr><tr><td>Avg. Gap</td><td colspan="2">29.66%</td><td colspan="2">14.931%</td><td colspan="2">21.482%</td><td colspan="2">19.252%</td><td colspan="2">11.590%</td><td colspan="2">10.832%</td></tr></table>
# F THE USE OF LARGE LANGUAGE MODELS (LLMS)
In this research, we employed Large Language Models (LLMs) as a general-purpose tool to assist with writing polish. These LLMs were utilized to enhance textual clarity without contributing to research conception or methodological development. 