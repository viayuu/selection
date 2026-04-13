arXiv:2410.09693v2 [math.OC] 24 May 2025 
Neural Solver Selection for Combinatorial Optimization 
Chengrui Gao * 1 2 Haopu Shang * 1 2 Ke Xue 1 2 Chao Qian 1 2 
# Abstract
Machine learning has increasingly been employed to solve NP-hard combinatorial optimization problems, resulting in the emergence of neural solvers that demonstrate remarkable performance, even with minimal domain-specific knowledge. To date, the community has created numerous opensource neural solvers with distinct motivations and inductive biases. While considerable efforts are devoted to designing powerful single solvers, our findings reveal that existing solvers typically demonstrate complementary performance across different problem instances. This suggests that significant improvements could be achieved through effective coordination of neural solvers at the instance level. In this work, we propose the first general framework to coordinate the neural solvers, which involves feature extraction, selection model, and selection strategy, aiming to allocate each instance to the most suitable solvers. To instantiate, we collect several typical neural solvers with state-of-the-art performance as alternatives, and explore various methods for each component of the framework. We evaluated our framework on two typical problems, Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP). Experimental results show that our framework can effectively distribute instances and the resulting composite solver can achieve significantly better performance (e.g., reduce the optimality gap by $0 . 8 8 \%$ on TSPLIB and $0 . 7 1 \%$ on CVRPLIB) than the best individual neural solver with little extra time cost. Our code is available at https://github.com/lamda-bbo/neuralsolver-selection. 
# 1. Introduction
Combinatorial Optimization Problems (COPs) involve finding an optimal solution over a set of combinatorial alternatives, which has broad and important applications such as logistics (Konstantakopoulos et al., 2022) and manufacturing (Zhang et al., 2019). To solve COPs, traditional approaches usually depend on heuristics designed by experts, requiring extensive domain knowledge and considerable effort. Recently, machine learning techniques have been introduced to automatically discover effective heuristics for COPs (Bengio et al., 2021; Cappart et al., 2023), leading to the burgeoning development of end-to-end neural solvers that employ deep neural networks to generate solutions for problem instances (Bello et al., 2017; Kool et al., 2019; Joshi et al., 2019). Compared to traditional approaches, these end-to-end neural solvers can not only get rid of the heavy reliance on expertise, but also realize better inference efficiency (Bello et al., 2017). 
To enhance the capabilities of neural solvers, a variety of methods have been proposed, with intensive effort on the design of frameworks, network architectures, and training procedures. For example, to improve the performance across different distributions, (Jiang et al., 2022) proposed adaptively joint training over varied distributions, and (Bi et al., 2022) leveraged knowledge distillation to integrate the models trained on different distributions. For generalization on large-scale instances, (Fu et al., 2021) implemented a divide-and-conquer strategy, (Luo et al., 2023) proposed a heavy-decoder structure to better capture the relationship among nodes, while (Gao et al., 2024) utilized the local transferability and introduced an additional local policy model. Diffusion models (Sun & Yang, 2023) have also been adapted to generate the distribution of optimal solutions, demonstrating impressive results. More works include bisimulation quotienting (Drakulic et al., 2023), latent space search (Chalumeau et al., 2023), local reconstruction (Cheng et al., 2023; Ye et al., 2024; Zheng et al., 2024) and so on. 
As various neural solvers are emerging in the community, the state-of-the-art records for the overall performance on benchmark problems are frequently refreshed. However, the detailed comparison of these neural solvers on each instance has been rarely discussed. Here, we empirically examined the performance of several prevailing neural solvers 
*Equal contribution 1National Key Laboratory for Novel Software Technology, Nanjing University, China 2School of Artificial Intelligence, Nanjing University, China. Correspondence to: Chao Qian <qianc@nju.edu.cn>. 
Proceedings of the $4 2 ^ { n d }$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). 
1 
Neural Solver Selection for Combinatorial Optimization 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/ba73f90e853b5539a1384a1e7b8be61ccce38b601bde9a897aae721406be120f.jpg)


(a) Percentages of Winning

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/00620f5b4d6de70b9d6234736ecf819163cf63e45f01ff4d305819456daab5e1.jpg)


(b) Optimality Gap (Average)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/b5a99bf9d4bb9c3894a14aae5885654401be1775249e0f1f52ce504c7fc1de86.jpg)


(c) Our Framework


Figure 1: (a), (b): Observation from the comparison of prevailing neural solvers at instance level. Details of the settings are provided in Section 4.1. (c): Our proposed selection framework.

on each instance, as illustrated in Figure 1(a) and 1(b). As expected from the no-free-lunch theorem (Wolpert & Macready, 1997), we find that: 
? As shown in Figure 1(a), there exists no single neural solver that can dominate all the others on every instance, and different neural solvers win on different instances, demonstrating their complementary performance at instance level. 
? As shown in Figure 1(b), the modification of instance distribution can almost reverse the domination relationship of neural solvers, which further verifies that different neural solvers are good at instances with specific characteristics due to their intrinsic inductive biases. 
These observations suggest that it may potentially bring impressive improvements to the overall performance, if multiple neural solvers are coordinated to solve instances together. In fact, recent works have already made preliminary attempts from the perspective of ensemble learning (Jiang et al., 2023) and population-based training (Grinsztajn et al., 2023). However, their individual solvers share the same architecture, resulting in limited diversity. On the other hand, as all of the individual solvers should run during inference, these methods can hardly achieve ideal efficiency. 
Motivated by the observations above, we, for the first time, propose a general framework to coordinate end-to-end neural solvers for COPs at the instance level by selecting suitable individual solvers for each instance, as illustrated in Figure 1(c). Specifically, our proposed framework consists of three key components, which are summarized as follows: 
? Feature extraction: For each problem instance, extract features for effectively identifying their characteristics. 
? Selection model: Based on the features of instances, train a selection model that can be utilized to identify suitable solvers for each instance. 
? Selection strategy: Due to the intricate structures of 
COPs, using only the most suitable individual solver predicted by the selection model may fail. Therefore, it is important to design robust selection strategies based on the confidence of the selection model. 
To verify the effectiveness of our proposed framework, we collect several prevailing open-source neural solvers and their released models with competitive performance in the community to construct the pool of individual solvers, and provide several implementations for each component of the framework. For feature extraction, we utilize the graph attention network (Velickovi ˇ c et al. ′ , 2018; Kool et al., 2019) to encode COP instances, and further propose a refined encoder with pooling to leverage the hierarchical structures of COPs. For selection model, we train it from the perspective of classification and ranking, respectively. We also implement several selection strategies, including top- $k$ selection, rejection-based selection, and so on. Detailed descriptions are provided in Section 3. We conduct experiments on two widely studied COPs: Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP). Experimental results exhibit that our framework can generally select suitable individual solvers for each instance to achieve significantly better performance with limited extra time consumption. Compared to the best individual solver, our framework reduces the optimality gap by $0 . 8 2 \%$ on synthetic TSP, $2 . 0 0 \%$ on synthetic CVRP, $0 . 8 8 \%$ on TSPLIB (Reinelt, 1991), and $0 . 7 1 \%$ on CVRPLIB Set-X (Uchoa et al., 2017). As the first preliminary attempt on neural solver selection for COPs, we also analyze the influence of various implementations of components, and discuss on future directions. 
# 2. Related works
# 2.1. End-to-end neural solvers for COPs
Traditional approaches for COPs have achieved impressive results, but they often rely on problem-specific heuristics 
2 
Neural Solver Selection for Combinatorial Optimization 
and domain knowledge by experts (Helsgaun, 2000; 2017). Instead, recent efforts focus on utilizing end-to-end learning methods. A prominent fashion is autoregression, which employs graph neural networks in an encoder-decoder framework and progressively extends a partial solution until a complete solution is constructed (Vinyals et al., 2015; Bello et al., 2017; Kool et al., 2019). However, these methods tend to exhibit poor generalization performance across distributions and scales (Joshi et al., 2022). To address the generalization issue, considerable efforts have been dedicated from various perspectives, including meta-learning (Zhou et al., 2023), knowledge distillation (Bi et al., 2022), prompt learning (Liu et al., 2024), instance-conditioned adaptation (Zhou et al., 2024a), adversarial training (Wang et al., 2024) and nested local views (Fang et al., 2024). 
Another popular kind of end-to-end learning methods is non-regressive, which predicts or generates the distributions of potential solutions. Typically, (Joshi et al., 2019; Ye et al., 2023) employed graph neural networks to predict the probability of components appearing in an optimal solution, represented with the form of heatmap. Diffusion models (Sun & Yang, 2023; Sanokowski et al., 2024) have also been adapted to generate the distribution of optimal solutions, demonstrating better expressiveness than classical push-forward generative models (Salmona et al., 2022). 
# 2.2. Solving COPs with multiple neural solvers
Recent studies have made preliminary attempts to integrate multiple neural solvers to enhance overall performance on COPs. For example, (Jiang et al., 2023) adopted ensemble learning, where multiple neural solvers with identical architecture are trained on different instance distributions through Bootstrap sampling. During inference, the outputs of all the solvers are averaged at each action step. (Grinsztajn et al., 2023) proposed a population-based training method Poppy, where multiple decoders with a shared encoder are trained simultaneously as a population of solvers, with a reward targeting at maximizing the overall performance of the population. When solving a problem instance, each solver generates solutions independently, and the best solution is selected as the final result. However, these works suffer from heavy cost as multiple solvers have to be run for each instance. Even they propose to share a common encoder for each solver, experiments still demonstrate undesired inference time (Grinsztajn et al., 2023). On the other hand, different solvers share the same neural architecture, which may limit the diversity and thus the final performance. 
Consider that the burgeoning community has proposed many methods from various perspectives, resulting in diverse neural solvers with different inductive biases. Properly coordinating them can potentially bring significant improvement on overall performance. Motivated by the observation in 
Figure 1(a) and 1(b), we propose to select suitable ones from a pool of diverse individual solvers for each instance. Note that similar idea has been utilized in the area of algorithm selection (Kerschke et al., 2019; He et al., 2025) and model selection (Zhang et al., 2023), but has never been explored in the area of neural combinatorial optimization. By solver selection at instance level, any existing or newly constructed neural solver can be utilized, and only the selected individual solvers need to be run in inference, thereby maintaining high efficiency. 
# 3. The proposed framework
This section introduces our proposed framework of coordinating neural solvers for COPs. Our target is learning to select the suitable solvers for each problem instance. To address it, the framework comprises three key components: 
Feature extraction: To select the most suitable neural solvers for each instance, it is essential to extract the instance features, which is challenging as the COPs are usually intricate. In this work, we first utilize the graph attention encoder (Kool et al., 2019) to encode COP instances, and further propose a refined graph encoder with pooling, which can leverage the hierarchical structures of COPs. 
Selection model: We train a neural selection model with the graph encoder to identify the most suitable solvers. Specifically, we implement two loss functions from the perspectives of classification and ranking. 
Selection strategies: Due to the complexity of COPs, it may be risky to rely solely on the selection model to identify the best solver. To address this, we propose compromise strategies to allocate multiple solvers (if necessary) to a single instance based on the confidence levels of the selection model, pursuing better performance with limited extra cost. 
# 3.1. Feature extraction
For feature extraction, it depends on the COP to be solved. Here, we use the two most prevailing problems, TSP and CVPR, in the neural solver community for COPs (Kwon et al., 2020; Luo et al., 2023; Drakulic et al., 2023) as examples, which will also be employed in our experiments. TSP and CVRP involve finding optimal routes over a set of nodes. For TSP, the objective is to find the shortest possible route that visits each node exactly once and returns to the starting node. Each TSP instance consists of nodes distributed in Euclidean space. For CVRP, the goal is to plan routes for multiple vehicles to serve customer nodes with varying demands, starting and ending at a depot node, while minimizing the total travel distance and satisfying vehicle capacity constraints (Dantzig & Ramser, 1959). Both TSP and CVRP instances can be represented as fully connected graphs, where nodes correspond to locations (cities or cus-
3 
Neural Solver Selection for Combinatorial Optimization 
tomers). The graph representation makes them suitable for encoding using Graph Neural Networks (GNNs), which can effectively capture the structural information inherent in these problems (Khalil et al., 2017; Kool et al., 2019). In this paper, we design two types of GNN-based encoders tailored for TSP and CVRP instances as follows. 
Graph attention encoder. We take the CVRP as an example to describe the computation of the graph encoder. The raw features $\pmb { x } \in \mathbb { R } ^ { N \times 3 }$ of a CVRP instance are a set of nodes $\{ ( x _ { i } , y _ { i } , m _ { i } ) | i \in [ N ] \}$ , where $( x _ { i } , y _ { i } )$ are the node coordinates, $m _ { i }$ is the node demand, $N$ is the number of nodes, and $[ N ]$ denotes the set $\{ 1 , 2 , \ldots , N \}$ . First, a linear layer is employed on every node for initial node embeddings, i.e., $H ^ { 0 } = \pmb { x } W$ , where $W \in \mathbb { R } ^ { 3 \times d }$ are the weights and $d$ denotes the embedding dimension. Given initial embeddings, multiple graph attention layers (Velickovi ˇ c et al. ′ , 2018; Kool et al., 2019) are applied to iteratively update the node embeddings as $H ^ { l } =$ $H ^ { l } = \bar { \mathrm { A t t e n t i o n L a y e r } } ^ { l } ( \bar { H ^ { l - 1 } } )$ $^ l ( H ^ { \bar { l } - 1 } )$ , where $l \in [ L ]$ and $L$ is the number of layers. Since the graphs of TSP and CVRP are both fully connected, the graph attention layer covers every pair of nodes and self-connections, which becomes similar to the self-attention mechanism (Vaswani et al., 2017). Details of the attention layer are in Appendix A.1. Finally, the node embeddings output by the last layer are averaged to form the instance representation, like most COP encoders (Khalil et al., 2017; Kool et al., 2019). 
Hierarchical graph encoder. Averaging the final node embeddings may result in sub-optimal instance representations that are too flat to effectively capture the hierarchical structures inherent in COPs (Goh et al., 2024). 
Inspired by (Lee et al., 
2019), we design a hierarchical graph attention encoder to address this limitation, which successively downsamples the graph of an instance using graph pooling, and aggregates features from each downsampling level to construct a comprehensive graph representation, as illustrated in Figure 2. The hierarchical graph encoder contains $L$ blocks. 
In each block, several graph attention layers are applied, and then the graph pooling layer selects representative nodes to form a coarsened graph that preserves important features. Consider the l-th block, where the number of selected nodes is denoted as $N _ { l }$ . To quantify the representativeness of each node for graph pooling, an additional graph attention layer is introduced to compute representative scores. Specifically, 
the graph attention layer computes score embeddings $H _ { \mathrm { s c o r e } } ^ { l }$ based on the current node embeddings $H ^ { l }$ , which encode rich information about the graph structure and node features. These score embeddings $H _ { \mathrm { s c o r e } } ^ { l }$ are then mapped to scalar representative scores via linear layer. The complete process is shown as follows: 
$$
H _ {\text {s c o r e}} ^ {l} = \text {A t t e n t i o n L a y e r} _ {\text {s c o r e}} ^ {l} (H ^ {l}), Z ^ {l} = \sigma \left(H _ {\text {s c o r e}} ^ {l} W _ {\text {s c o r e}} ^ {l}\right),
$$
where $Z ^ { l } ~ \in ~ \mathbb { R } ^ { N ^ { l - 1 } \times 1 }$ are the representative scores of the $N ^ { l - 1 }$ nodes preserved in the $( l - 1 )$ -th block, $\sigma$ is a non-linear function (here we use the tanh function), and $W _ { \mathrm { s c o r e } } ^ { l } \in \mathbb { R } ^ { d \times 1 }$ are the parameters of the linear layer. Subsequently, we sort the nodes according to their representative scores and select top- $N ^ { l }$ nodes $( { \bar { N } } ^ { l } < N ^ { l - 1 } < N )$ to preserve. To make this pooling layer trainable via backpropagation (LeCun et al., 2002), we further combine the representative score together with the embeddings of their corresponding nodes as follows: $\tilde { H } ^ { l } = H ^ { l } + Z ^ { l } \mathbf { 1 }$ , where $\mathbf { 1 } \in \mathbb { R } ^ { 1 \times d }$ is a vector with all the elements being 1. Intuitively, this operation can separate the embeddings of high scored nodes from the embeddings of low scored nodes. 
Figure 2 shows the complete encoder, where $L$ blocks are stacked, and each block is formed by several graph attention layers followed by a pooling layer. By successively applying $L$ encoder blocks, the number of preserved nodes gradually decreases as $N ^ { l } = \alpha \cdot N ^ { l - 1 }$ $\scriptstyle { \alpha }$ is set to 0.8 in our experiments). This process constructs a hierarchy of the original graph and its coarsened versions, enabling the encoder to capture multi-level structural information effectively. Within each block, we apply a readout layer that aggregates the embeddings after the graph attention layers by mean pooling and max pooling (Lee et al., 2019), i.e., 
$$
\boldsymbol {o} ^ {l} = \sigma (\operatorname {M e a n} (H ^ {l}) \| \operatorname {M a x} (H ^ {l})),
$$
where Mean() computes the average embedding over the nodes, Max() computes the maximum along the column dimension, $\parallel$ denotes concatenation, and $\sigma$ is a non-linear function. The result $o ^ { l }$ provides the representation of the $l$ -th coarsened graph. At the last layer, we also readout $\mathbf { o } ^ { L + 1 }$ from the final embeddings. To form the hierarchical instance representation, we sum the representations of all levels as o = PL+1l=1 $o = \sum _ { l = 1 } ^ { L + 1 } o ^ { l }$ . 
# 3.2. Selection model
We employ a Multiple-Layer Perception (MLP) to predict the compatibility scores of neural solvers, where a higher score indicates that it is more suitable to allocate the instance to the corresponding neural solver. This MLP model takes the instance representation and the instance scale $N$ as input and outputs a score vector, where the value of each index is the score of the corresponding neural solver. In summary, 
4 
Neural Solver Selection for Combinatorial Optimization 
the graph encoder and the MLP are cascaded to compose a selection model, which can produce the compatibility scores of solvers for the COP instances in an end-to-end manner. Advanced neural solver features can be incorporated for richer information, as discussed in Section 5. However, we find that even using fixed indices of neural solvers can be effective, which will be clearly shown in our experiments. We train the selection model using a supervised dataset comprising thousands of synthetic COP instances. The objective values obtained by the neural solvers are recorded as supervision information. Intuitively, a neural solver with a lower objective value (for minimization) has a higher compatibility score. To learn it, we employ two losses from the perspectives of classification and ranking. 
Classification. The selection problem is formulated as classification, where the most suitable neural solver for a given instance serves as the ground truth label. By employing classification loss functions such as cross-entropy loss in our experiments, we can train a selection model. However, this approach focuses on identifying the optimal neural solver and ignores sub-optimal solvers, which may lead to unsatisfactory performance when the selection is inaccurate. 
Ranking. The neural solvers can be sorted according to the objective values they obtain, thereby forming a ranking of the given solvers, denoted by $\phi : [ M ] \to [ M ]$ , where $\phi ( i )$ is the index of the rank- $\cdot i$ solver and $M$ is the number of solvers. We then train the selection model by maximizing the likelihood of producing correct rankings based on the computed scores (Xia et al., 2008), 
$$
\max  _ {\theta} \mathbb {E} _ {I} \left[ \sum_ {i = 1} ^ {M} \log \frac {\exp \left(g _ {\theta} (I) _ {\phi_ {I} (i)}\right)}{\sum_ {j = i} ^ {M} \exp \left(g _ {\theta} (I) _ {\phi_ {I} (j)}\right)} \right],
$$
where $g _ { \theta }$ denotes the selection model with parameters $\theta$ , $I$ denotes a problem instance, and $\phi _ { I }$ is the ground-truth ranking on instance $I$ . This ranking loss can leverage the dominance relationship of all neural solvers, including suboptimal ones, which can make the selection more robust. 
# 3.3. Selection strategies
Considering that the intricate structures of COPs may pose great challenges to the selection model, besides greedy selection (choosing the highest-scoring solver), we propose several compromise strategies that allow multiple solvers for a single instance based on the confidence level of the selection model, aiming to improve the overall performance with little extra cost. 
? Top- $k$ selection. The top- $k$ selection method can be adopted for better optimality, where we select and execute the neural solvers with top- $k$ scores for each instance, thus constructing a portfolio of multiple solvers. 
This approach increases the likelihood of including the optimal solver but incurs additional computational overhead due to the execution of multiple solvers. 
? Rejection-based selection. To balance efficiency and effectiveness, we propose the rejection-based selection strategy, which adaptively selects greedy or top- $k$ selection. Recognizing that the confidence of greedy selection varies between instances, an advanced strategy is to employ the top- $k$ selection for low-confidence instances to enhance performance and utilize only greedy selection for high-confidence ones to minimize computational cost. To implement this strategy, we can use a confidence measure to determine whether to accept or reject greedy selection. If the confidence in greedy selection is below a threshold, we reject it and apply the top- $k$ selection to the instance. In this paper, we adopt the simple yet effective softmax response (Hendrycks & Gimpel, 2017) as the confidence measure and define the threshold by rejecting a certain fraction of test instances with the lowest confidence levels. 
? Top- $p$ selection. We further propose a top- $p$ selection strategy that selects the smallest subset of solvers with normalized scores sum up to at least $p$ . It can adaptively determine the number of selected neural solvers by covering a certain amount of confidence, rather than relying on a fixed number $k$ . The value of $p$ is predefined or adjusted according to time budget. 
# 4. Experiments
To examine the effectiveness of our proposed selection framework, we conduct experiments on TSP and CVRP, investigating the following Research Questions (RQ): RQ1: How does the proposed selection framework perform compared to individual neural solvers? RQ2: How does the proposed selection framework perform when the problem distribution shifts and the problem scale increases? RQ3: How do different implementations of components affect the performance of the framework? We introduce the experimental settings in Section 4.1 and investigate the above RQs in Section 4.2. 
# 4.1. Experimental settings
We generate synthetic TSP and CVRP instances by sampling node coordinates from Gaussian mixture distributions (more details of the data generation are provided in Appendix A.2.1). For training, we generate 10, 000 TSP and CVRP instances and apply 8-fold instance augmentation (Kwon et al., 2020). For test, a smaller synthetic datasets with 1, 000 instances is used. Figures 1(a) and 1(b) in Section 1 depict results on the CVRP test dataset. To evaluate the out-of-distribution performance, we utilize 
5 
Neural Solver Selection for Combinatorial Optimization 
two well-known benchmarks with more complex problem distributions and larger problem scales (up to $N = 1 0 0 2$ ): TSPLIB (Reinelt, 1991) and CVRPLIB Set-X (Uchoa et al., 2017). For TSPLIB, we select a subset of instances with $N \leq 1 0 0 2$ , and CVRPLIB Set-X includes instances ranging from $N = 1 0 0$ to 1000. These problem scales are larger than $N \in [ 5 0 , 5 0 0 ]$ in our training datasets. Notably, in Appendix A.10, we also include experiments on larger scales up to $N = 2 0 0 0$ for further evaluation. 
Open-source neural solvers. We choose recent opensource neural solvers with state-of-the-art performance as the candidates, including Omni (Zhou et al., 2023), BQ (Drakulic et al., 2023), LEHD (Luo et al., 2023), DIFUSCO (Sun & Yang, 2023), T2T (Li et al., 2023), ELG (Gao et al., 2024), INViT (Fang et al., 2024) and MV-MoE (Zhou et al., 2024b). More details about the realization can be found in Appendix A.2.2. Considering that some neural solvers contribute little to the overall performance, we iteratively eliminate the least contributive solver from the candidates, resulting in a more compact neural solver zoo. This process reduces the zoo size to 7 solvers for TSP and 5 for CVRP. Further details of the elimination procedure are provided in Appendix A.3. 
Hyperparameters. Limited by space, we provide the details of hyperparameters in our experiments in Appendix A.2.3, including the hyperparameters of graph encoders, hyperparameters of training, and hyperparameter of selection strategies. 
Performance metrics. Following previous studies, we use the gap to the best-known solution $\frac { c _ { I } ( \hat { \sigma } ) - c _ { I } ( \sigma ^ { * } ) } { c _ { I } ( \sigma ^ { * } ) }$ as the performance metric, called optimality gap, where $\hat { \sigma }$ is the solution obtained by each method, $\sigma ^ { * }$ is the best-known solution computed by extensive search of expert solvers (Helsgaun, 2017; Vidal, 2022), and $c _ { I } ( )$ is the cost function of problem instance I. We also report the average time (including both the running of solvers and selection) to evaluate efficiency, . 
# 4.2. Experimental results
RQ1: How does the proposed selection framework perform compared to individual neural solvers? In Table 1, we present the performance of several implementations of our selection framework on synthetic TSP and CVRP, alongside the results of the top-3 individual neural solvers1. We can observe that all implementations of our framework outperform the best neural solver on both TSP and CVRP, demonstrating the effectiveness of our framework. For example, using ranking loss and the top- $k$ selection strategy with $k = 2$ , our framework achieves average optimality gaps of $1 . 5 1 \%$ on TSP and $4 . 8 2 \%$ on CVRP, surpassing the 
best individual solver’s gaps of $2 . 3 3 \%$ on TSP and $6 . 8 2 \%$ on CVRP, achieved by DIFUSCO and Omni, respectively. Moreover, except utilizing the top- $k$ strategy, our selection framework is nearly as efficient as running a single solver. In some cases, our framework can obtain better optimality gaps while consuming even less time. For instance, using ranking loss and greedy selection on TSP leads to the average optimality gap $1 . 8 6 \%$ with 1.33s, while the best individual solver DIFUSCO achieves $2 . 3 3 \%$ gap with 1.45s. In Table 1, Oracle (the fourth row) denotes the optimal performance for selection, which is obtained by running all individual solvers for each instance and selecting the best one. The best optimality gaps achieved by our selection framework (using ranking loss and top- $k$ selection with $k = 2$ ) are close to Oracle, with gaps of $1 . 5 1 \%$ on TSP and $4 . 8 1 \%$ on CVRP, compared to Oracle’s gaps of $1 . 2 4 \%$ on TSP and $4 . 6 4 \%$ on CVRP. Furthermore, our framework can offer significant speed advantages over Oracle, e.g., consuming an average time of 2.56s on TSP, whereas Oracle requires an average time of 8.93s. Note that complete results for all individual solvers are provided in Appendix A.12. 
Extension of RQ1: Is the performance of the top- $k$ selection better than the solver portfolio with the same size? The top- $k$ strategy enhances the performance by running a selected subset of the solver zoo for each instance, which certainly costs more time than individual solvers. For a fair comparison, we benchmark our top- $k$ selection method against a solver portfolio of the same size $k$ . We construct this solver portfolio by exhaustively enumerating all possible subsets of size $k$ and selecting the one with the best overall performance. As shown in Appendix A.6, our top- $k$ selection consistently outperforms the size- $k$ solver portfolio across $k = \{ 1 , 2 , 3 , 4 \}$ on all datasets, i.e., TSP, CVRP, TSPLIB and CVRPLIB Set-X, demonstrating the effectiveness of our selection model. 
RQ2: How does the proposed selection framework perform when the problem distribution shifts and the problem scale increases? We evaluate the generalization performance on two benchmarks, TSPLIB and CVRPLIB Set-X, which contain out-of-distribution and larger-scale instances. As shown in Table 2, all implementations of our selection framework generalize well, where the ranking model using top- $k$ selection improves the optimality gap by $0 . 8 8 \%$ (i.e., $1 . 9 5 \% - 1 . 0 7 \%$ ) on TSPLIB and by $0 . 7 1 \%$ (i.e., $6 . 1 0 \%$ - $5 . 3 9 \%$ ) on CVRPLIB Set-X, compared to the best individual solvers T2T and ELG on these two benchmarks. These results show that our selection framework is robust against the distribution shifts and increases in problem scale. 
RQ3: How do different implementations affect performance? We evaluate and compare different implementations of the three components in our framework: 
1DIFUSCO and T2T have multiple trained models. We only report the best results of these models. 
(1) Feature extraction methods. We compare the man-
6 
Neural Solver Selection for Combinatorial Optimization 

Table 1: Empirical results on synthetic TSP and CVRP datasets, reporting the mean (standard deviation) over five independent runs. The top three individual solvers are included for comparison, and Oracle denotes the optimal performance for selection, which is computed by running all individual solvers in the zoo for each instance and selecting the best one. The best individual solver and its results are underlined, and the best optimality gaps, excluding Oracle, are highlighted in boldface.

<table><tr><td rowspan="2">Methods</td><td colspan="2">TSP</td></tr><tr><td>Gap</td><td>Time</td></tr><tr><td>BQ (3rd)</td><td>3.00%</td><td>1.40s</td></tr><tr><td>T2T (2nd)</td><td>2.40%</td><td>1.58s</td></tr><tr><td>DIFUSCO (1st)</td><td>2.33%</td><td>1.45s</td></tr><tr><td>Oracle</td><td>1.24%</td><td>8.93s</td></tr><tr><td colspan="3">Selection by classification</td></tr><tr><td>Greedy</td><td>1.94% (0.02%)</td><td>1.36s (0.01s)</td></tr><tr><td>Top-k (k=2)</td><td>1.53% (0.01%)</td><td>2.52s (0.04s)</td></tr><tr><td>Rejection (20%)</td><td>1.81% (0.01%)</td><td>1.63s (0.01s)</td></tr><tr><td>Top-p (p=0.5)</td><td>1.84% (0.03%)</td><td>1.55s (0.06s)</td></tr><tr><td colspan="3">Selection by ranking</td></tr><tr><td>Greedy</td><td>1.86% (0.01%)</td><td>1.33s (0.01s)</td></tr><tr><td>Top-k (k=2)</td><td>1.51% (0.02%)</td><td>2.56s (0.03s)</td></tr><tr><td>Rejection (20%)</td><td>1.75% (0.02%)</td><td>1.63s (0.01s)</td></tr><tr><td>Top-p (p=0.5)</td><td>1.68% (0.02%)</td><td>1.86s (0.07s)</td></tr></table>
ual features (Smith-Miles et al., 2010) (see Appendix A.4), graph attention encoder (Kool et al., 2019), and hierarchical graph encoder in Table 3. All methods are trained using ranking loss, and we report the optimality gap with greedy selection. As shown in Table 3, even the simplest manual features perform well, achieving better results than the best individual solver across three datasets — TSP, CVRP, and TSPLIB. This further validates the effectiveness of our selection framework. Comparing the third and fourth columns, we observe that the graph attention encoder consistently outperforms manual features on all datasets, verifying the superiority of learned features. More manual features are compared in Appendix A.5. Furthermore, by comparing the fourth and fifth columns, we find that while the graph attention encoder has already been effective on synthetic datasets, introducing the hierarchical encoder can further improve generalization performance on out-of-distribution datasets, TSPLIB and CVRPLIB Set-X, which is quite important in practice. This enhanced generalization capability may be attributed to the hierarchical encoder’s ability to leverage the inherent hierarchical structures in COPs. More ablation studies of the hierarchical encoder are in Appendix A.7. 
(2) Loss functions to train the selection model. We can clearly observe from Tables 1 and 2 that the model trained with ranking loss generally outperforms that trained with classification loss, particularly when employing top- $p$ selection or under out-of-distribution settings. We also compare their accuracy of selecting the best single individual, which is similar as shown Appendix A.9. Thus, the benefit of ranking loss over classification loss shows the importance of incorporating the dominance relationships among sub-
<table><tr><td rowspan="2">Methods</td><td colspan="2">CVRP</td></tr><tr><td>Gap</td><td>Time</td></tr><tr><td>LEHD (3rd)</td><td>7.37%</td><td>1.01s</td></tr><tr><td>BQ (2nd)</td><td>7.20%</td><td>1.59s</td></tr><tr><td>Omni (1st)</td><td>6.82%</td><td>0.24s</td></tr><tr><td>Oracle</td><td>4.64%</td><td>4.38s</td></tr><tr><td colspan="3">Selection by classification</td></tr><tr><td>Greedy</td><td>5.35% (0.02%)</td><td>0.64s (0.01s)</td></tr><tr><td>Top-k (k = 2)</td><td>4.81% (0.01%)</td><td>1.87s (0.03s)</td></tr><tr><td>Rejection (20%)</td><td>5.19% (0.03%)</td><td>0.77s (0.01s)</td></tr><tr><td>Top-p (p = 0.8)</td><td>5.16% (0.03%)</td><td>0.87s (0.08s)</td></tr><tr><td colspan="3">Selection by ranking</td></tr><tr><td>Greedy</td><td>5.31% (0.01%)</td><td>0.62s (0.01s)</td></tr><tr><td>Top-k (k = 2)</td><td>4.82% (0.01%)</td><td>1.90s (0.04s)</td></tr><tr><td>Rejection (20%)</td><td>5.15% (0.02%)</td><td>0.74s (0.01s)</td></tr><tr><td>Top-p (p = 0.8)</td><td>4.99% (0.02%)</td><td>1.03s (0.03s)</td></tr></table>
optimal solvers. 
(3) Selection strategies. Greedy selection is efficient by selecting only the predicted best solver. Instead, top- $k$ selection selects the best $k$ solvers for better optimality gaps, but resulting in longer time. Rejection-based and top- $p$ selection provide a trade-off between optimality gap and time. Here, we focus on the evaluation of rejection-based and top- $p$ selection. We tune their hyperparameters (e.g., rejection ratio, $k$ , and $p$ ) to obtain a range of results, provided in Appendix A.8. The results show that the rejection-based selection with smaller $k$ $k = 2$ or 3) tends to achieve better trade-off. Comparing top- $p$ and rejection-based selection, their performance has no obvious difference. This is expected as their principles are both running more individual neural solvers when the confidence of the selection model is insufficient. However, the top- $p$ selection may be preferable in practice since only one hyperparameter $p$ is associated. 
# 4.3. Exploration on neural solver features
To enable generalization to unseen neural solvers, we propose a preliminary feature extraction method that leverages representative instances to characterize each neural solver. Specifically, for a given neural solver, we sort those instances where the neural solver performs the best in ascending order according to the ratio of the objective value that the solver obtains to the runner-up objective value and select the top $1 \%$ as its representative instances. Then, we use an instance encoder to obtain embeddings for representative instances as their token vectors. A two-layer transformer model further processes these token vectors to learn a sum-
7 
Neural Solver Selection for Combinatorial Optimization 

Table 2: Generalization results to TSPLIB and CVRPLIB Set-X datasets, which contain real-world out-of-distribution instances with larger scales.

<table><tr><td rowspan="2">Methods</td><td colspan="2">TSPLIB</td></tr><tr><td>Gap</td><td>Time</td></tr><tr><td>BQ (3rd)</td><td>3.04%</td><td>1.44s</td></tr><tr><td>DIFUSCO (2nd)</td><td>2.13%</td><td>1.44s</td></tr><tr><td>T2T (1st)</td><td>1.95%</td><td>1.74s</td></tr><tr><td>Oracle</td><td>0.89%</td><td>9.14s</td></tr><tr><td colspan="3">Selection by classification</td></tr><tr><td>Greedy</td><td>1.54% (0.05%)</td><td>1.33s (0.02s)</td></tr><tr><td>Top-k (k = 2)</td><td>1.22% (0.10%)</td><td>2.47s (0.02s)</td></tr><tr><td>Rejection (20%)</td><td>1.42% (0.11%)</td><td>1.54s (0.03s)</td></tr><tr><td>Top-p (p = 0.5)</td><td>1.49% (0.11%)</td><td>1.37s (0.02s)</td></tr><tr><td colspan="3">Selection by ranking</td></tr><tr><td>Greedy</td><td>1.33% (0.06%)</td><td>1.28s (0.03s)</td></tr><tr><td>Top-k (k = 2)</td><td>1.07% (0.03%)</td><td>2.48s (0.02s)</td></tr><tr><td>Rejection (20%)</td><td>1.26% (0.03%)</td><td>1.51s (0.04s)</td></tr><tr><td>Top-p (p = 0.5)</td><td>1.28% (0.04%)</td><td>1.46s (0.06s)</td></tr></table>
<table><tr><td rowspan="2">Methods</td><td colspan="2">CVRPLIB Set-X</td></tr><tr><td>Gap</td><td>Time</td></tr><tr><td>BQ (3rd)</td><td>10.31%</td><td>2.60s</td></tr><tr><td>Omni (2nd)</td><td>6.21%</td><td>0.38s</td></tr><tr><td>ELG (1st)</td><td>6.10%</td><td>1.31s</td></tr><tr><td>Oracle</td><td>5.10%</td><td>6.81s</td></tr><tr><td colspan="3">Selection by classification</td></tr><tr><td>Greedy</td><td>5.96% (0.12%)</td><td>1.06s (0.08s)</td></tr><tr><td>Top-k (k = 2)</td><td>5.44% (0.08%)</td><td>2.40s (0.25s)</td></tr><tr><td>Rejection (20%)</td><td>5.83% (0.12%)</td><td>1.31s (0.09s)</td></tr><tr><td>Top-p (p = 0.8)</td><td>5.79% (0.09%)</td><td>1.42s (0.17s)</td></tr><tr><td colspan="3">Selection by ranking</td></tr><tr><td>Greedy</td><td>5.76% (0.04%)</td><td>1.31s (0.10s)</td></tr><tr><td>Top-k (k = 2)</td><td>5.39% (0.06%)</td><td>2.56s (0.13s)</td></tr><tr><td>Rejection (20%)</td><td>5.63% (0.05%)</td><td>1.60s (0.08s)</td></tr><tr><td>Top-p (p = 0.8)</td><td>5.61% (0.03%)</td><td>1.72s (0.08s)</td></tr></table>

Table 3: Mean (standard deviation) of optimality gaps of different feature extraction methods. All the models are trained using ranking loss, and employ greedy selection.

<table><tr><td>Datasets</td><td>Best solver</td><td>Manual</td><td>Attention encoder</td><td>Hierarchical encdoer</td></tr><tr><td>TSP</td><td>2.33%</td><td>1.97% (0.01%)</td><td>1.87% (0.02%)</td><td>1.86% (0.01%)</td></tr><tr><td>CVRP</td><td>6.82%</td><td>5.49% (0.08%)</td><td>5.30% (0.01%)</td><td>5.31% (0.01%)</td></tr><tr><td>TSPLIB</td><td>1.95%</td><td>1.83% (0.03%)</td><td>1.45% (0.11%)</td><td>1.33% (0.06%)</td></tr><tr><td>CVRPLIB</td><td>6.10%</td><td>6.35% (0.06%)</td><td>5.87% (0.06%)</td><td>5.76% (0.04%)</td></tr></table>
marized feature, which serves as the feature representation of the neural solver. 
To integrate a newly added neural solver, we first identify its representative instances from a synthetic dataset and employ the aforementioned networks to compute its feature representation. The selection model can then leverage the newly added solver by considering its feature during selection, without the need for any fine-tuning. The implementation details of each component are described as follows. 
Instance tokenization. We use a hierarchical graph encoder as the tokenization encoder to generate embeddings for each representative instance. To stabilize the instance tokens during training, we update $\theta ^ { \prime }$ using a momentum-based moving average of the parameters $\theta$ of the instance feature encoder: $\theta ^ { \prime }  m \cdot \theta ^ { \prime } + ( 1 - m ) \cdot \theta$ , where $m \in [ 0 , 1 )$ is a momentum coefficient (We set $m = 0 . 9 9$ in experiments). Only the parameters $\theta$ are updated via back-propagation. This momentum update ensures that $\theta ^ { \prime }$ evolves more smoothly than $\theta$ , resulting in stable instance tokenization. 
Transformer architecture. For each neural solver, we utilize the tokens of its representative instances along with a learnable summary token to compute a summary repre-
sentation. We apply two attention layers for this purpose. The first layer is a self-attention mechanism applied over all tokens (including the summary token), enabling interactions among them. The second attention layer uses only the summary token as the query and all tokens as keys and values, effectively aggregating information from all tokens into the summary token. The final embedding of the summary token is then output as the neural solver’s representation. 
Selection model with neural solver features. The selection model integrates both the instance features and the neural solver features to output a score for each instancesolver pair. We employ an MLP to compute these scores. For each instance, the scores across all neural solvers are normalized to derive the probability distribution. 
To evaluate the effectiveness of this method, we remove the second-best neural solver from the current solver zoo, train the selection model using the supervision information over the pruned solver zoo, and reintroduce the removed solver during testing. Figure 3 presents the top- $k$ selection performance with and without the newly added (extra) solver. The results show that the performance with the newly added solver is generally better than the performance without it, demonstrating that the selection model can leverage the 
8 
Neural Solver Selection for Combinatorial Optimization 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/569c7ad508d76dc2eb53310c7144538656328a4b514dd8bb55b179e12c862900.jpg)


Figure 3: Performance of introducing an extra neural solver based on the neural solver feature. The results of top- $k$ selection with and without extra neural solver are presented. The horizontal axis represents the number of selected solvers.

information of unseen solvers without any finetuning. In other words, the selection model can generalize to unseen solvers. However, we observe a slight decrease in top-1 performance, indicating that the current feature design is not accurate enough for the most precise top-1 selection. This highlights the need for further refinement and optimization to our preliminary feature extraction method. 
# 5. Conclusion and discussions
In this paper, we propose a general framework for neural solver selection for the first time, which can effectively select suitable solvers for each instance, leading to significantly better performance with little additional computational time, as validated by the extensive experiments on two well-studied COPs, TSP and CVRP. Besides TSP and CVRP, our proposed selection framework is adaptable to other problems. For new problems, one only needs to customize the feature extraction component. For instance, when adapting our framework to scheduling problems, one can adjust the graph attention encoder according to MatNet (Kwon et al., 2021) (i.e., add edge embeddings). We hope this preliminary work can open a new line for neural combinatorial optimization. Within the proposed selection framework, we preliminarily investigate several implementations of the three key components: Feature extraction, training loss functions, and selection strategies. Techniques such as hierarchical graph encoder, ranking loss, rejection-based selection, and top- $p$ selection notably enhance overall performance. Beyond the techniques presented, we discuss several promising avenues for further research under this framework. 
Feature extraction for neural solvers. Our method, which uses fixed indices for neural solvers, assumes a static neural solver zoo and cannot directly utilize any newly added neural solvers during deployment. To enable zero-shot generalization to unseen neural solvers, it is essential to construct a smooth feature space for solvers, where those with similar preferences and biases are positioned closely together. In 
Section 4.3, we design a preliminary method for extracting features of neural solvers to facilitate generalization to unseen solvers. The results show that it enables generalization to unseen neural solvers, where adding an extra solver can improve the selection performance. For future improvements, some approaches may be worth exploring, such as utilizing large language models to encode the neural solvers from their codes and descriptions (Wu et al., 2024), or learning representations from their trained parameters (Kofinas et al., 2024), which can involve internal solver information and potentially bring improvements. 
Runtime-aware selection for learn-to-seach solvers. In this paper, since the average runtime of most individual neural solvers is short (approximately 1–2 seconds), we ignored their time difference during the training of the selection model, and only used the objective values obtained by the neural solvers as supervision information (by classification or ranking). However, if there are some time-consuming learn-to-search solvers, such as NeuOpt (Ma et al., 2021; 2023) and local reconstruction methods (Kim et al., 2021; Ye et al., 2024), in the solver pool, the runtime should be considered in the performance ranking. In such cases, developing a runtime-aware selection method to balance computational time and solution optimality would be necessary. 
Enhance the neural solver zoo by training. As shown in Figure 6, current neural solvers can exhibit complementary performance over instances without any modification, which has motivated our framework of neural solver selection. Inspired by the population-based training (Grinsztajn et al., 2023), we can further enhance their complementary ability through finetuning, i.e., each neural solver is finetuned on those instances where it performs the best. We can also train new solvers from scratch by maximizing their performance contribution to the current solver zoo and iteratively add such new solvers for enhancement. Moreover, to facilitate the training and deployment of a neural solver zoo, it is essential to develop a unified platform that provides interfaces for executing and training diverse neural solvers, such as an extension to the existing RL4CO (Berto et al., 2024). 
# Impact Statement
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. 
# Acknowledgement
This work was supported by the Science and Technology Project of the State Grid Corporation of China (5700- 202440332A-2-1-ZX). 
9 
Neural Solver Selection for Combinatorial Optimization 
# References


Bachlechner, T., Majumder, B. P., Mao, H. H., Cottrell, G., and McAuley, J. J. Rezero is all you need: fast convergence at large depth. In Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence (UAI), pp. 1352–1361, Virtual, 2021. 




Bai, Y., Zhao, W., and Gomes, C. P. Zero training overhead portfolios for learning to solve combinatorial problems. arXiv:2102.03002, 2021. 




Bello, I., Pham, H., Le, Q. V., Norouzi, M., and Bengio, S. Neural combinatorial optimization with reinforcement learning. In Proceedings of the 5th International Conference on Learning Representations (ICLR), Toulon, France, 2017. 




Bengio, Y., Lodi, A., and Prouvost, A. Machine learning for combinatorial optimization: A methodological tour d’horizon. European Journal of Operational Research, 290(2):405–421, 2021. 




Berto, F., Hua, C., Park, J., Luttmann, L., Ma, Y., Bu, F., Wang, J., Ye, H., Kim, M., Choi, S., Zepeda, N. G., Hottung, A., Zhou, J., Bi, J., Hu, Y., Liu, F., Kim, H., Son, J., Kim, H., Angioni, D., Kool, W., Cao, Z., Zhang, J., Shin, K., Wu, C., Ahn, S., Song, G., Kwon, C., Xie, L., and Park, J. RL4CO: An extensive reinforcement learning for combinatorial optimization benchmark. arXiv:2306.17100, 2024. https://github.com/ ai4co/rl4co. 




Bi, J., Ma, Y., Wang, J., Cao, Z., Chen, J., Sun, Y., and Chee, Y. M. Learning generalizable models for vehicle routing problems via knowledge distillation. In Advances in Neural Information Processing Systems 35 (NeurIPS), pp. 31226–31238, New Orleans, LA, 2022. 




Campello, R. J. G. B., Moulavi, D., and Sander, J. Densitybased clustering based on hierarchical density estimates. In Proceedings of the 17th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD), pp. 160–172, Gold Coast, Australia, 2013. 




Cappart, Q., Chetelat, D., Khalil, E. B., Lodi, A., Morris, C., ′ and Velickovi ˇ c, P. Combinatorial optimization and rea-′ soning with graph neural networks. Journal of Machine Learning Research, 24(130):1–61, 2023. 




Chalumeau, F., Surana, S., Bonnet, C., Grinsztajn, N., Pretorius, A., Laterre, A., and Barrett, T. Combinatorial optimization with policy adaptation using latent space search. In Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 7947–7959, New Orleans, LA, 2023. 




Cheng, H., Zheng, H., Cong, Y., Jiang, W., and Pu, S. Select and optimize: Learning to aolve large-scale TSP instances. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS), pp. 1219–1231, Valencia, Spain, 2023. 




Dantzig, G. B. and Ramser, J. H. The truck dispatching problem. Management Science, 6(1):80–91, 1959. 




Drakulic, D., Michel, S., Mai, F., Sors, A., and Andreoli, J.- M. BQ-NCO: Bisimulation quotienting for generalizable neural combinatorial optimization. In Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 77416– 77429, New Orleans, LA, 2023. 




Fang, H., Song, Z., Weng, P., and Ban, Y. INViT: A generalizable routing problem solver with invariant nested view transformer. In Proceedings of the 41st International Conference on Machine Learning (ICML), pp. 12973–12992, Vienna, Austria, 2024. 




Fu, Z.-H., Qiu, K.-B., and Zha, H. Generalize a small pre-trained model to arbitrarily large TSP instances. In Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI), pp. 7474–7482, Virtual, 2021. 




Gao, C., Shang, H., Xue, K., Li, D., and Qian, C. Towards generalizable neural solvers for vehicle routing problems via ensemble with transferrable local policy. In Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI), pp. 6914–6922, Jeju, Korea, 2024. 




Goh, Y. L., Cao, Z., Ma, Y., Dong, Y., Dupty, M. H., and Lee, W. S. Hierarchical neural constructive solver for realworld TSP scenarios. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), pp. 884–895, Barcelona, Spain, 2024. 




Grinsztajn, N., Furelos-Blanco, D., Surana, S., Bonnet, C., and Barrett, T. Winner takes it all: Training performant RL populations for combinatorial optimization. pp. 48485–48509, New Orleans, LA, 2023. 




He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, Las Vegas, NV, 2016. 




He, X., Shang, H., and Qian, C. How to train algorithm selection models: Insights from black-box continuous optimization. In 15th Workshop on Evolutionary Computation for the Automated Design of Algorithms at GECCO’25, Malaga, Spain, 2025. ′ 




Heins, J., Bossek, J., Pohl, J., Seiler, M., Trautmann, H., and Kerschke, P. On the potential of normalized TSP features for automated algorithm selection. In Proceedings of the 


10 
Neural Solver Selection for Combinatorial Optimization 


16th ACM/SIGEVO Workshop on Foundations of Genetic Algorithms (FOGA), pp. 1–15, Virtual Event, Austria, 2021. 




Helsgaun, K. An effective implementation of the lin– kernighan traveling salesman heuristic. European Journal of Operational Research, 126(1):106–130, 2000. 




Helsgaun, K. An extension of the lin-kernighan-helsgaun TSP solver for constrained traveling salesman and vehicle routing problems. Technical report, 2017. 




Hendrycks, D. and Gimpel, K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In Proceedings of the 5th International Conference on Learning Representations (ICLR), Toulon, France, 2017. 




Jiang, Y., Wu, Y., Cao, Z., and Zhang, J. Learning to solve routing problems via distributionally robust optimization. In Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI), pp. 9786–9794, Virtual, 2022. 




Jiang, Y., Cao, Z., Wu, Y., Song, W., and Zhang, J. Ensemble-based deep reinforcement learning for vehicle routing problems under distribution shift. In Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 53112–53125, New Orleans, LA, 2023. 




Joshi, C. K., Laurent, T., and Bresson, X. An efficient graph convolutional network technique for the travelling salesman problem. arXiv:1906.01227, 2019. 




Joshi, C. K., Cappart, Q., Rousseau, L., and Laurent, T. Learning the travelling salesperson problem requires rethinking generalization. Constraints, 27(1-2):70–98, 2022. 




Kerschke, P., Hoos, H. H., Neumann, F., and Trautmann, H. Automated algorithm selection: Survey and perspectives. IEEE Transactions on Evolutionary Computation, 27(1): 3–45, 2019. 




Khalil, E. B., Dai, H., Zhang, Y., Dilkina, B., and Song, L. Learning combinatorial optimization algorithms over graphs. In Advances in Neural Information Processing Systems 30 (NeurIPS), pp. 6348–6358, Long Beach, CA, 2017. 




Kim, M., Park, J., and kim, j. Learning collaborative policies to solve NP-hard routing problems. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems 34 (NeurIPS), pp. 10418–10430, 2021. 




Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR), San Diego, CA, 2015. 




Kofinas, M., Knyazev, B., Zhang, Y., Chen, Y., Burghouts, G. J., Gavves, E., Snoek, C. G., and Zhang, D. W. Graph neural networks for learning equivariant representations of neural networks. In Proceedings of the 12th International Conference on Learning Representations (ICLR), Vienna, Austria, 2024. 




Konstantakopoulos, G. D., Gayialis, S. P., and Kechagias, E. P. Vehicle routing problem and related algorithms for logistics distribution: A literature review and classification. Operational Research, 22(3):2033–2062, 2022. 




Kool, W., van Hoof, H., and Welling, M. Attention, learn to solve routing problems! In Proceedings of the 7th International Conference on Learning Representations (ICLR), New Orleans, LA, 2019. 




Kwon, Y.-D., Choo, J., Kim, B., Yoon, I., Gwon, Y., and Min, S. POMO: Policy optimization with multiple optima for reinforcement learning. In Advances in Neural Information Processing Systems 33 (NeurIPS), pp. 21188– 21198, Virtual, 2020. 




Kwon, Y.-D., Choo, J., Yoon, I., Park, M., Park, D., and Gwon, Y. Matrix encoding networks for neural combinatorial optimization. volume 34, pp. 5138–5149, Sydney, Australia, 2021. 




LeCun, Y., Bottou, L., Orr, G. B., and Muller, K.-R. Effi-¨ cient backprop. In Neural networks: Tricks of the trade, volume 7700, pp. 9–48. 2002. 




Lee, J., Lee, I., and Kang, J. Self-attention graph pooling. In Proceedings of the 36th International Conference on Machine Learning (ICML), pp. 3734–3743, Long Beach, California, 2019. 




Li, Y., Guo, J., Wang, R., and Yan, J. T2T: From distribution learning in training to gradient search in testing for combinatorial optimization. In Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 50020–50040, New Orleans, LA, 2023. 




Liu, F., Lin, X., Liao, W., Wang, Z., Zhang, Q., Tong, X., and Yuan, M. Prompt learning for generalized vehicle routing. In Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI), pp. 6976– 6984, Jeju, Korea, 2024. 




Luo, F., Lin, X., Liu, F., Zhang, Q., and Wang, Z. Neural combinatorial optimization with heavy decoder: Toward large scale generalization. In Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 8845–8864, New Orleans, LA, 2023. 




Ma, Y., Li, J., Cao, Z., Song, W., Zhang, L., Chen, Z., and Tang, J. Learning to iteratively solve routing problems with dual-aspect collaborative transformer. In Advances 


11 
Neural Solver Selection for Combinatorial Optimization 


in Neural Information Processing Systems 34 (NeurIPS), pp. 11096–11107, Virtual, 2021. 




Ma, Y., Cao, Z., and Chee, Y. M. Learning to search feasible and infeasible regions of routing problems with flexible neural $k$ -Opt. In Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 49555–49578, New Orleans, LA, 2023. 




Manchanda, S., Michel, S., Drakulic, D., and Andreoli, J.-M. On the generalization of neural combinatorial optimization heuristics. In Proceedings of the 33rd/26th Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD), pp. 426–442, Grenoble, France, 2022. 




Nazari, M., Oroojlooy, A., Taka′c, M., and Snyder, L. V. ˇ Reinforcement learning for solving the vehicle routing problem. In Advances in Neural Information Processing Systems 31 (NeurIPS), pp. 9861–9871, Montreal, Canada, ′ 2018. 




Reinelt, G. TSPLIB - A traveling salesman problem library. ORSA Journal on Computing, 3(4):376–384, 1991. 




Salmona, A., De Bortoli, V., Delon, J., and Desolneux, A. Can push-forward generative models fit multimodal distributions? In Advances in Neural Information Processing Systems 35 (NeurIPS), pp. 10766–10779, New Orleans, LA, 2022. 




Sanokowski, S., Hochreiter, S., and Lehner, S. A diffusion model framework for unsupervised neural combinatorial optimization. In Proceedings of the 41st International Conference on Machine Learning (ICML), Vienna, Austria, 2024. 




Seiler, M., Pohl, J., Bossek, J., Kerschke, P., and Trautmann, H. Deep learning as a competitive feature-free approach for automated algorithm selection on the traveling salesperson problem. In Proceedings of 16th International Conference on Parallel Problem Solving from Nature (PPSN), pp. 48–64, Leiden, The Netherlands, 2020. 




Smith-Miles, K., Van Hemert, J., and Lim, X. Y. Understanding TSP difficulty by learning from evolved instances. In Proceedings of the 4th International Conference on Learning and Intelligent Optimization (LION), pp. 266– 280, Venice, Italy, 2010. 




Sun, Z. and Yang, Y. DIFUSCO: Graph-based diffusion solvers for combinatorial optimization. In Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 3706–3731, New Orleans, LA, 2023. 




Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., and Subramanian, A. New benchmark instances for the capacitated vehicle routing problem. European Journal of Operational Research, 257(3):845–858, 2017. 




Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, ?., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems 30 (NeurIPS), pp. 5998–6008, Long Beach, CA, 2017. 




Velickovi ˇ c, P., Cucurull, G., Casanova, A., Romero, A., Li ′ o,` P., and Bengio, Y. Graph attention networks. In Proceedings of the 6th International Conference on Learning Representations (ICLR), Vancouver, Canada, 2018. 




Vidal, T. Hybrid genetic search for the CVRP: Open-source implementation and swap* neighborhood. Computers & Operations Research, 140:105643, 2022. 




Vinyals, O., Fortunato, M., and Jaitly, N. Pointer networks. In Advances in Neural Information Processing Systems 28 (NeurIPS), pp. 2692–2700, Montreal, Canada, 2015. 




Wang, C., Yu, Z., McAleer, S., Yu, T., and Yang, Y. ASP: Learn a universal neural solver! IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(6):4102– 4114, 2024. 




Wolpert, D. H. and Macready, W. G. No free lunch theorems for optimization. IEEE Transactions on Evolutionary Computation, 1(1):67–82, 1997. 




Wu, X., Zhong, Y., Wu, J., Jiang, B., and Tan, K. C. Large language model-enhanced algorithm selection: Towards comprehensive algorithm representation. In Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI), pp. 5235–5244, Jeju, Korea, 2024. 




Xia, F., Liu, T.-Y., Wang, J., Zhang, W., and Li, H. Listwise approach to learning to rank: Theory and algorithm. In Proceedings of the 25th International Conference on Machine Learning (ICML), pp. 1192–1199, Helsinki, Finland, 2008. 




Ye, H., Wang, J., Cao, Z., Liang, H., and Li, Y. DeepACO: neural-enhanced ant systems for combinatorial optimization. Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 43706–43728, 2023. 




Ye, H., Wang, J., Liang, H., Cao, Z., Li, Y., and Li, F. GLOP: Learning global partition and local construction for solving large-scale routing problems in real-time. In Proceedings of the 38th AAAI Conference on Artificial Intelligence, pp. 20284–20292, Vancouver, Canada, 2024. 




Zhang, J., Ding, G., Zou, Y., Qin, S., and Fu, J. Review of job shop scheduling research and its new perspectives under industry 4.0. Journal of Intelligent Manufacturing, 30:1809–1830, 2019. 




Zhang, Y.-K., Huang, T.-J., Ding, Y.-X., Zhan, D.-C., and Ye, H.-J. Model spider: Learning to rank pre-trained 


12 
Neural Solver Selection for Combinatorial Optimization 


models efficiently. Advances in Neural Information Processing Systems 36 (NeurIPS), pp. 13692–13719, 2023. 




Zheng, Z., Zhou, C., Xialiang, T., Yuan, M., and Wang, Z. UDC: A unified neural divide-and-conquer framework for large-scale combinatorial optimization problems. arXiv:2407.00312, 2024. 




Zhou, C., Lin, X., Wang, Z., Tong, X., Yuan, M., and Zhang, Q. Instance-conditioned adaptation for largescale generalization of neural combinatorial optimization. arXiv:2405.01906, 2024a. 




Zhou, J., Wu, Y., Song, W., Cao, Z., and Zhang, J. Towards omni-generalizable neural methods for vehicle routing problems. In Proceedings of the 40th International Conference on Machine Learning (ICML), pp. 42769–42789, Honolulu, HI, 2023. 




Zhou, J., Cao, Z., Wu, Y., Song, W., Ma, Y., Zhang, J., and Xu, C. MVMoE: Multi-task vehicle routing solver with mixture-of-experts. In Proceedings of the 39th International Conference on Machine Learning (ICML), pp. 61804–61824, Vienna, Austria, 2024b. 


13 
Neural Solver Selection for Combinatorial Optimization 
# A. Appendix
# A.1. Graph attention layer
The graph attention layer is composed of two sub-layers: a multi-head attention sub-layer (Vaswani et al., 2017) and a feed-forward sub-layer. Each sub-layer is equipped with residual connection (He et al., 2016) and ReZero normalization (Bachlechner et al., 2021) for stable convergence of training. Denote the embedding of the $i$ -th node as $\boldsymbol { h } _ { i }$ (i.e., the $i$ -th row of $H$ ). Since the graphs of TSP and CVRP are typically considered to be fully connected, the graph attention layer is calculated as 
$$
\begin{array}{l} \hat {\boldsymbol {h}} _ {i} = \boldsymbol {h} _ {i} ^ {l - 1} + \alpha^ {l} \mathrm {M H A} _ {i} ^ {l} (\boldsymbol {h} _ {1} ^ {l - 1}, \boldsymbol {h} _ {2} ^ {l - 1},..., \boldsymbol {h} _ {N} ^ {l - 1}), \\ \boldsymbol {h} _ {i} ^ {l} = \hat {\boldsymbol {h}} _ {i} + \alpha^ {l} \mathrm {F F} (\hat {\boldsymbol {h}} _ {i}), \\ \end{array}
$$
where $l$ is the layer index, $i$ is the node index, $\alpha ^ { l }$ is a learnable parameter used in the ReZero normalization, MHA and FF are short for the multi-head attention and the feed-forward network, respectively. For the implementations of the basic components MHA and FF, we refer to (Vaswani et al., 2017) for details. Specifically, following the common settings of previous works, we set the dimension of $^ { h }$ to 128, the number of heads in MHA to 8, and the hidden dimension of FF to 512. 
# A.2. Supplement of experimental settings
# A.2.1. DATA GENERATION
Synthetic TSP and CVRP instances are generated for training, where the node coordinates, demands, and vehicle capacities are all sampled from manually defined distributions. The scale of each instance is sampled from [50, 500] randomly. Details of node coordinates, vehicle capacity, and node demands are introduced as follows. 
Node coordinates To generate diverse training instances, we utilize Gaussian mixture distributions to sample the node coordinates for both TSP and CVRP, which is common in previous works (Manchanda et al., 2022; Zhou et al., 2023) and demonstrates effectiveness on approximating various node distributions with different hardness levels (Smith-Miles et al., 2010). First, we randomly select the number of Gaussian components $c \sim U ( 0 , 1 5 )$ (when $c = 0$ , we use the uniform distribution) and partition the nodes randomly into $c$ groups, one for each component. For each Gaussian component, we sample the mean coordinates $\pmb { \mu } = ( x _ { \mu } , y _ { \mu } )$ by $x _ { \mu } \sim U ( 0 , 1 )$ and $y _ { \mu } \sim U ( 0 , 1 )$ , and sample the variances $\operatorname { v a r } _ { x }$ and $\operatorname { v a r } _ { y }$ uniformly from [1, 100]. The covariance cov is sampled uniformly from $[ - \sqrt { \mathrm { v a r } _ { x } \cdot \mathrm { v a r } _ { y } }$ , $\sqrt { \mathrm { v a r } _ { x } \cdot \mathrm { v a r } _ { y } } )$ , forming the covariance matrix $\Sigma = { \left[ \begin{array} { l l } { \operatorname { v a r } _ { x } } & { \operatorname { c o v } } \\ { \operatorname { c o v } } & { \operatorname { v a r } _ { y } } \end{array} \right] } .$ . Node coordinates are sampled from the $N ( \pmb { \mu } , \Sigma )$ and then scaled to the square of vary $x , y \in [ 0 , 1 ]$ . Unlike conventional Gaussian mixture distributions (Manchanda et al., 2022; Zhou et al., 2023), which often use an identity covariance matrix, our approach employs randomized covariance matrices $\Sigma$ . This modification can produce more diverse instances by introducing more variability in the node distributions. 
Vehicle capacity and node demands We employ two vehicle capacity distributions to generate CVRP instances: (1) Scale-related distribution (Zhou et al., 2023): The vehicle capacity is proportional to the scale $N$ , defined as $\begin{array} { r } { Q = 3 0 + \lceil \frac { N } { 5 } \rceil } \end{array}$ (2) Triangular distributions (Uchoa et al., 2017): The parameters of the triangular distribution include the upper limit $u b$ mode $m$ , and lower limit $l b$ , which are randomly sampled in succession as follows: $u b \sim U ( 2 0 , \frac { N } { 2 } )$ , $m \sim U ( 5 , u b )$ , and $l b \sim U ( 3 , m )$ . The triangular distribution $T ( l b , m , u b )$ is then used to generate vehicle capacities, resulting in more diverse CVRP instances compared to the fixed capacity setting (Nazari et al., 2018). Each capacity distribution is selected with equal probability. Node demands $m _ { i }$ are sampled uniformly from $U ( 1 , 1 0 )$ and normalized by dividing by $Q$ . 
# A.2.2. DETAILED SETTINGS OF OPEN SOURCE SOLVERS
We choose recent open-source neural solvers with state-of-the-art performance as the candidates, including Omni (Zhou et al., 2023), BQ (Drakulic et al., 2023), LEHD (Luo et al., 2023), DIFUSCO (Sun & Yang, 2023), T2T (Li et al., 2023), ELG (Gao et al., 2024), INViT (Fang et al., 2024) and MVMoE (Zhou et al., 2024b). Greedy decoding is used for all the methods to avoid stochasticity. We set the pomo size to 100 and the augmentation number to 8 for the methods based on POMO (Kwon et al., 2020). The number of denoising steps is set to 50 and the number of 2-opt iterations is set to 100 for diffusion-based methods. These individual solvers constitute a neural solver zoo. Ideally, if we can always select the best solver from the zoo for each instance, the optimal performance is achieved, which is also the performance upper bound 
14 
Neural Solver Selection for Combinatorial Optimization 
of our selection model. Considering that some neural solvers contribute little to the overall performance, we iteratively eliminate the least contributive solver from the candidates, resulting in a more compact neural solver zoo, which reduces the zoo size to 7 solvers for TSP and 5 for CVRP. Further details of the elimination procedure are provided in Appendix A.3. 
# A.2.3. DETAILED SETTINGS OF HYPERPARAMETERS
(1) Hyperparameters of graph encoders. For the graph attention encoder, we set the number of layers to 4. For the hierarchical graph encoder, we use 2 blocks where each block has 2 attention layers. The embedding dimension is set to 128. Other details of encoders can be found in Appendix A.1. (2) Hyperparameters of training. The Adam optimizer (Kingma & Ba, 2015) is employed for training, where we set the learning rate to $1 \times 1 0 ^ { - 4 }$ and the weight decay to $1 \times 1 0 ^ { - 6 }$ . The number of epochs is set to 50. The final model is chosen according to the performance on a validation dataset with 1, 000 synthetic instances. We train 5 selection models using different random seeds and report the mean and standard deviation of their performance. (3) Hyperparameters of selection strategies. For the top- $k$ strategy, we set $k = 2$ . For the rejection-based strategy, we reject the $20 \%$ of instances with the lowest confidence levels (i.e., the highest selection probability of all individual solvers), and apply top-2 selection to these rejected instances. For the top- $p$ strategy, we set $p = 0 . 5$ for TSP and $p = 0 . 8$ for CVRP. 
# A.3. Eliminate useless neural solvers
The preserved neural solvers should have distinct strengths in certain problem instances, ensuring that they can bring significant improvements in overall performance. Motivated by this, we propose a simple yet effective heuristic strategy to build the neural solver zoo based on the assessment of their contribution to the overall performance. 
Given the alternative neural solvers $\pmb { S } = \{ s _ { 1 } , s _ { 2 } , s _ { 3 } , \ldots \}$ , we assess the contribution of a specific solver $s _ { i } \in S$ by the degradation of performance after removing it. That is, the assessment of $s _ { i }$ can be formalized as 
$$
\mathcal {A} \left(s _ {i}\right) = \mathbb {E} _ {I} \left[ \mathcal {P} _ {I} (\boldsymbol {S}) - \mathcal {P} _ {I} \left(\boldsymbol {S} / s _ {i}\right) \right],
$$
where $\mathcal { P } _ { I } ( \cdot )$ denotes the performance of a neural solver zoo on instance $I$ . Here we use the percentage of the optimality gap to define $\mathcal { P } _ { I } ( \cdot )$ and employ a validation set for the estimation of expectations. According to this criteria, we can estimate the alternative solvers and remove the one with the lowest assessed contribution from $_ { s }$ . This process repeats iteratively until for all $s _ { i } \in S$ , $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } _ { i } )$ surpasses the predefined threshold $\delta$ , indicating the significance of each alternative neural solver. In practice, we collect the prevailing competitive neural solvers in the community to compose the original $s$ and set $\delta$ as $0 . 0 1 \%$ . 
The neural solver zoos before and after elimination are listed in Table 4. Note that for DIFUSCO and T2T, multiple models are released. We collect both the models trained on the $\Nu = 1 0 0$ dataset and the $\Nu = 5 0 0$ dataset as alternatives simultaneously. 

Table 4: The neural solver zoo before and after elimination.

<table><tr><td>Stage</td><td>Neural solver zoo for TSP</td><td>Neural solver zoo for CVRP</td></tr><tr><td>Before elimination</td><td>BQ, LEHD, Omni, ELG, INViT, DIFUSCO (N=100), DIFUSCO (N=500), T2T (N=100), T2T (N=500)</td><td>BQ, LEHD, Omni, ELG, INViT, MVMoE</td></tr><tr><td>After elimination</td><td>BQ, LEHD, ELG, DIFUSCO (N=100), DIFUSCO (N=500), T2T (N=100), T2T (N=500)</td><td>BQ, LEHD, Omni, ELG, MVMoE</td></tr></table>
# A.4. Manual features
We reproduce the manual features proposed by (Smith-Miles et al., 2010), which use statistical information and cluster analysis results to describe the characteristics of TSP. In this paper, we adopt these features: the standard deviation of the distances, the coordinates of the instance centroid, the radius of the TSP instance, the fraction of distinct distances, the variance of the normalized nearest neighbour distances (nNNd’s), the coefficient of variation of the nNNd’s, the ratio of 
15 
Neural Solver Selection for Combinatorial Optimization 
the number of clusters to the number of nodes (Here we use HDBSCAN algorithm (Campello et al., 2013) to generate clusters), the ratio of number of outliers to nodes, and the mean radius of the clusters. For CVRP, we further add the mean and standard deviation of node demands to the features. 
# A.5. Comparisons with traditional algorithm selection methods
To further demonstrate the effectiveness of our proposed techniques, we provide additional comparison results between our proposed method and existing algorithm selection methods for non-neural TSP solvers (Smith-Miles et al., 2010; Seiler et al., 2020), as shown in Table 5. In fact, the method of using features from (Smith-Miles et al., 2010) and our ranking model was also compared in Table 3. The R package salesperson2 provides the up-to-now most comprehensive collection of features for TSP and is widely used in algorithm selection methods (Seiler et al., 2020; Heins et al., 2021). Based on the feature set of salesperson, we reproduce an advanced algorithm selection method (Seiler et al., 2020) following the pipeline that computes hand-crafted features, conducts feature selection, and applies random forest for classification, where we employ the univariate statistical test to select important features. Besides, we also combine the salesperson features with our ranking model for ablation, denoted by ”(Seiler et al., 2020) $^ +$ Ranking” in Table 5. 

Table 5: Comparison experiments with algorithm selection methods for TSP. We report the mean (standard deviation) over five independent runs.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Synthetic TSP</td><td colspan="2">TSPLIB</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Single best solver</td><td>2.33%</td><td>1.45s</td><td>1.95%</td><td>1.74s</td></tr><tr><td>Oracle</td><td>1.24%</td><td>8.93s</td><td>0.89%</td><td>9.14s</td></tr><tr><td colspan="5">Algorithm selection methods</td></tr><tr><td>(Smith-Miles et al., 2010) + Ranking</td><td>1.97% (0.01%)</td><td>1.37s (0.01s)</td><td>1.83% (0.03%)</td><td>1.32s (0.05s)</td></tr><tr><td>(Seiler et al., 2020)</td><td>2.12% (0.04%)</td><td>1.35s (0.00s)</td><td>1.56% (0.01%)</td><td>1.34s (0.05s)</td></tr><tr><td>(Seiler et al., 2020) + Ranking</td><td>1.95% (0.01%)</td><td>1.33s (0.03s)</td><td>1.55% (0.03%)</td><td>1.27s (0.06s)</td></tr><tr><td colspan="5">Our selection method using ranking</td></tr><tr><td>Greedy</td><td>1.86% (0.01%)</td><td>1.33s (0.01s)</td><td>1.33% (0.06%)</td><td>1.28s (0.03s)</td></tr><tr><td>Top-k (k=2)</td><td>1.51% (0.02%)</td><td>2.56s (0.03s)</td><td>1.07% (0.03%)</td><td>2.48s (0.02s)</td></tr><tr><td>Rejection (20%)</td><td>1.75% (0.02%)</td><td>1.63s (0.01s)</td><td>1.26% (0.03%)</td><td>1.51s (0.04s)</td></tr><tr><td>Top-p (p=0.5)</td><td>1.68% (0.02%)</td><td>1.86s (0.07s)</td><td>1.28% (0.04%)</td><td>1.46s (0.06s)</td></tr></table>
The experimental results in Table 5 indicate that our proposed method can achieve superior performance than advanced algorithm selection methods on both synthetic TSP and TSPLIB. Comparing the fifth and sixth rows, our proposed hierarchical encoder demonstrates superior performance over the salesperson features, especially on the out-of-distribution benchmark TSPLIB. Additionally, the comparison of the fourth and fifth rows shows that our deep learning-based ranking model achieves better results than traditional classification methods. Furthermore, the results of the last three rows illustrate that our proposed adaptive selection strategies effectively enhance optimality with minimal increases in time consumption. 
# A.6. Results of top- $k$ selection.
We compare the performance of our top- $k$ selection and the solver portfolio with the same size $k$ on four datasets, including TSP, CVRP, TSPLIB and CVRPLIB Set-X. As shown in Figure 4, our top- $k$ selection consistently outperforms the size- $k$ solver portfolio across $k \in \{ 1 , 2 , 3 , 4 \}$ . We also observe that the performance of our top- $k$ selection is close to the Oracle when $k = 4$ . Moreover, it is expected that the performance improvement of our top- $k$ selection gradually diminishes as $k$ increases, since the performance of solver portfolio is also approaching the Oracle (the gray line). 
Related works, such as ZTop (Bai et al., 2021), employ a fixed set of neural solvers to construct a portfolio for all instances, resembling the static portfolio approach compared in this study. In contrast, our top- $k$ selection strategy dynamically constructs instance-specific portfolios, offering greater flexibility and a higher potential for performance improvement. As demonstrated in Figure 4, our method consistently outperforms the static portfolio approach across all portfolio sizes. 
2https//github.com/jakobbossek/salesperson 
16 
Neural Solver Selection for Combinatorial Optimization 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/6cf4b4feb31840ab9dc42a51b3794e7a43a599005fd7a919df0599794ab7194a.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/84db35d80def5339bde9e53950b022842f4c1620238180cfb55d7f1b9746ca2d.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/ae72f165c57e80a020515793832b49c219561efdac135374416d84f94f3d22b1.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/b5e42cfec93558e87cfaef05f7baaed606a123123bcb2d1e7236dd732284b7c1.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/d87557e8434eb2d4a32310f18ac6c3054b4ca757e9a60f37290a49abb64332a4.jpg)


Figure 4: Comparisons of the proposed top- $k$ selection and the solver portfolio with size $k$ .

# A.7. Ablation of the hierarchical graph encoder.
The proposed hierarchical graph encoder utilizes graph pooling to downsample the instance graph and aggregates features obtained from multiple levels of the downsampled graphs. A graphical illustration of the downsampling process is provided in Appendix A.11, where we can find some consistent patterns which are intuitively reasonable. To evaluate the effectiveness of the graph pooling, we employ a graph encoder that aggregates the features from multiple layers for comparison, which is a clear ablation study since the main difference is that it does not have the graph pooling operation. 
The results in Table 6 show that using our hierarchical graph encoder outperforms the encoder that simply accumulates multi-layer features, especially in terms of the generalization performance on CVRPLIB Set-X. This demonstrates the effectiveness of the graph pooling operation. 

Table 6: Ablation study of hierarchical graph encoder. We report the mean and standard deviation of five independent runs. All the models are trained using ranking loss, and employ greedy selection.

<table><tr><td>Datasets</td><td>Attention encoder</td><td>+ Multi-layer features</td><td>Hierarchical encoder</td></tr><tr><td>TSP</td><td>1.87% (0.02%)</td><td>1.87% (0.01%)</td><td>1.86% (0.01%)</td></tr><tr><td>CVRP</td><td>5.30% (0.01%)</td><td>5.30% (0.02%)</td><td>5.31% (0.01%)</td></tr><tr><td>TSPLIB</td><td>1.45% (0.11%)</td><td>1.35% (0.05%)</td><td>1.33% (0.06%)</td></tr><tr><td>CVRPLIB</td><td>5.87% (0.06%)</td><td>5.86% (0.08%)</td><td>5.76% (0.04%)</td></tr></table>
To evaluate the computational efficiency of the hierarchical encoder, we provide detailed comparisons of the computation cost and optimality between our hierarchical encoder and a typical graph encoder. The results are shown in Table 7, which includes the inference time per instance on TSPLIB, training time per epoch, and the average optimality gap on TSPLIB. 

Table 7: Comparisons of the computation cost and optimality between our hierarchical encoder and a typical attention encoder.

<table><tr><td>Methods</td><td>Inference time of selection model</td><td>Inference time of neural solvers</td><td>Training time each epoch</td><td>Optimality gap</td></tr><tr><td>Naive attention encoder</td><td>0.0054s</td><td>1.2600s</td><td>1m40s</td><td>1.54%</td></tr><tr><td>Hierarchical encoder</td><td>0.0070s</td><td>1.2961s</td><td>2m30s</td><td>1.37%</td></tr></table>
We can observe from the second column that the introduction of our hierarchical encoder will increase the inference time of the selection model a little bit, e.g., from 0.0054s to 0.0070s. However, as shown in the second and third columns, the 
17 
Neural Solver Selection for Combinatorial Optimization 
inference time of the selection model is orders of magnitude shorter than that of the neural solvers, so the inference efficiency of the selection model is less of a concern. The fourth column shows that the training time per epoch of the na¨?ve encoder and the hierarchical encoder are $1 \mathrm { m } 4 0 \mathrm { s }$ and $2 \mathrm { m } 3 0 \mathrm { s }$ , respectively. Although the hierarchical encoder slows the training, the total runtime for 50 epochs is still only 2 hours, which is acceptable in most scenarios. Therefore, the performance metric (i.e., optimality gap) of different encoders is more crucial, especially the generalization performance. If the encoder learns robust representations, we can directly transfer the selection model to different datasets in a zero-shot manner, saving the time for fine-tuning and adaptation. Considering the better generalization (e.g., the optimality gap decreases from $1 . 5 4 \%$ t o $1 . 3 7 \%$ ), we believe that the proposed hierarchical encoder is a better choice. 
# A.8. Detailed comparisons of selection strategies
According to the mechanisms of the four selection strategies, they have different preferences in the trade-off of efficiency and optimality. Generally, for efficiency, Greedy $>$ Rejection $\approx$ Top-p > Top-k, for optimality, Top- $k >$ Rejection ≈ Top- $p$ $>$ Greedy. Meanwhile, the hyper-parameters of them can be used for balancing efficiency and optimality as well. As a result, the choice of different selection strategies can be decided by the users according to their preference, and we suggest using Top-p or Rejection as the default choices since they can adaptively select solvers based on the confidence. 
The rejection-based selection and top- $p$ selection are both designed to achieve better performance with little additional time consumption. To evaluate them in detail, we tune their parameters (e.g., rejection ratio, $k$ , and $p$ ) to obtain a range of results. For the rejection-based selection, we use $k \in \{ 2 , 3 , 4 \}$ and vary the rejection ratio from 0.05 to 0.85 in increments of 0.05. For the top- $p$ selection, we adjust the value of $p$ from 0.40 to 0.95 in increments of 0.01. The results of the optimality gap and time consumption are provided in Figure 5. As shown in the figures, the rejection strategy with smaller $k$ tends to achieve better optimality gaps using the same time consumption. Therefore, we recommend $k = 2$ or 3 when using rejection-based selection. Comparing top- $p$ selection and rejection-based selection, we can not definitively conclude which strategy is superior, which is expected since they share a similar idea of utilizing confidence levels to decide whether to employ multiple solvers. However, the top- $p$ selection may be preferable in practice due to its simplicity, where only a single hyperparameter $p$ requires tuning. 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/0852700b7cc138dd4f8c48c43828d772bcf9763ba1c77379e3397bdf16e8b9ba.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/fa537da8f0a967c6354636c92a9ac638c066d4a1291136ad554ff00ef5d9e0fa.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/c7bd2fb35ef0eeb06d19d69c2e5ba679f6d3cb0796617563ac7699c876d8e3a2.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/c4921c9f401e22b14be9557629ba3cc0e4ebcf78f6583e561829ebede25c09c4.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/312d9391b5a2a2624c37bdbd1fa37635be7a12b6d109196fde4cab65267c9f2d.jpg)


(a) Selection by classification

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/2b6080fa66eedad2d6d46d811788425f792261a093c7b2f68ba24435c2968880.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/85e050278b1a33e6ec1c91afef94dc350661d7bf12132906e4b190c5d870790f.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/36a683e295c28b7c9e4ecbced1a23e41dfe1dd5686609b85cf9f022150ca3b2d.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/a7153e1a8a5361950144e806e257fdafbababf3811444492bdb7e29644879ef5.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/25210f4e39c1bf624088c3da2914d257de7e7276033eb5aa8e98c2b052d4f048.jpg)


(b) Selection by ranking


Figure 5: Performance of the rejection-based and top- $p$ selection.

18 
Neural Solver Selection for Combinatorial Optimization 
# A.9. Selection accuracy
We present the accuracy of selecting the optimal neural solver in Table 8. The results show that the ranking model and classification model generally have similar selection accuracy, except that the ranking model achieves better accuracy than the classification model on CVRPLIB. 

Table 8: Accuracy of models trained by different losses using greedy selection. We report the mean and standard deviation of five independent runs.

<table><tr><td>Metrics</td><td>Classification</td><td>Ranking</td></tr><tr><td>Accuracy on TSP</td><td>36% (1%)</td><td>35% (1%)</td></tr><tr><td>Accuracy on CVRP</td><td>61% (1%)</td><td>62% (0%)</td></tr><tr><td>Accuracy on TSPLIB</td><td>40% (3%)</td><td>40% (7%)</td></tr><tr><td>Accuracy on CVRPLIB</td><td>52% (2%)</td><td>56% (3%)</td></tr></table>
# A.10. Additional results for more neural solvers and larger-scale datasets
We add two divide-and-conquer solvers, GLOP (Ye et al., 2024) and UDC (Zheng et al., 2024), to our solver pool, increase the problem scale from $N \in [ 5 0 , 5 0 0 ]$ to $N \in [ 5 0 0 , 2 0 0 0 ]$ to conduct new experiments. The results shown in Table 9 demonstrate that our framework can be compatible with more neural solvers and can also improve performance over the single best solver on larger-scale instances. 

Table 9: Experimental results on the larger-scale instances with $N \in [ 5 0 0 , 2 0 0 0 ]$ . We report the mean (standard deviation) over five independent runs.

<table><tr><td rowspan="2">Methods</td><td colspan="2">Synthetic TSP with N ∈ [500, 2000]</td></tr><tr><td>Gap</td><td>Time</td></tr><tr><td>Single best solver</td><td>6.104%</td><td>8.369s</td></tr><tr><td>Ours (Greedy)</td><td>5.540% (0.038%)</td><td>8.322s (0.036s)</td></tr><tr><td>Ours (Top-k, k = 2)</td><td>5.369% (0.003%)</td><td>15.566s (0.085s)</td></tr><tr><td>Single best of new solver pool</td><td>3.562%</td><td>5.274s</td></tr><tr><td>Ours with new solvers (Greedy)</td><td>3.126% (0.002%)</td><td>6.892s (0.006s)</td></tr><tr><td>Ours with new solvers (Top-k, k = 2)</td><td>2.955% (0.005%)</td><td>13.713s (0.036s)</td></tr></table>
19 
Neural Solver Selection for Combinatorial Optimization 
# A.11. Illustration of the nodes sampled by hierarchical graph encoder
We illustrate the retained nodes after downsampling. Surprisingly, we can find some consistent patterns which are intuitively reasonable. We summarize them as three main points: 
? Cluster nodes. As illustrated in Figures 6(a) and 6(b), when instances contain certain clusters, the hierarchical encoder tends to select a subset of “representative” nodes from each cluster, efficiently describing the entire spatial distribution. 
? Specific blocks. As illustrated in Figures 6(c) and 6(d), when instances contain specific complex geometric patterns like squares (Figure 6(c)) and arrays (Figure 6(d)), the hierarchical encoder can capture the nodes of these important areas to identify their characteristics. 
? Boundary nodes. For instances without clear sub-components, the hierarchical encoder tends to focus on boundary nodes that describe the global shape, as illustrated in Figures 6(e) and 6(f). 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/6c4072df7bee1ee94f453673a16369c1cb26433cb172c7e14ce8424d53628b61.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/2eaca54f23559de1fc3091cb1d2c2c2ab4516c406c70ff3b6977540749fa545b.jpg)


(a)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/db2d480c4adc9e92c9d04138b182a05c4ecf4387a3b2990c83b698e7a03890e0.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/3ecd33b9b229178e81ecd80183b70c13e49d90e22d1748e9c4b5f0b76eea1d4d.jpg)


(b)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/ff264ee3b5ab2c2dc6530b814ba018a853d159e8de896bbf616ab3cb95468f3d.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/35ccdb0e2288384c96bf3d470d284aeb8772bfb3b9bf5cdbf745a5a5bcc3ca3e.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/3bd1cfa4ae322a03db1c441a230cd3dabb514929103491ef4b1b9a646ef2457e.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/a1dd6d9c53da51f0233dfb2ef89c15584077b322d577cf4d910dedcd7e74dd89.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/c453c89933733833be781828e7205b6e973b90f58590a75deaf22324d5875627.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/1f53a55f94aa612540140756aa92bf76c1ccab1cb94b11d078be273c400a571e.jpg)


(e)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/3840a6f5a9a2e7b9b099779af8d7d71148fa605cfd7c4c8cc3432b4c6a7e7b22.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/eeeb3a43-97a9-4b30-a0a9-18ec9ba9b769/ab69560c21a3c5109fcb984389f998505f54480bd1439f0fc28e5f5a25a44409.jpg)


(f)


Figure 6: Illustrations of nodes selected by the hierarchical encoder. Each sub-figure represents an instance of TSP. The blue nodes represent the original instance, and the red nodes represent the retained nodes after down-sampling by the hierarchical encoder.

# A.12. Complete results with all neural solvers
The complete experimental results with all individual solvers are presented in Table 10 and Table 11, where the average objective values are provided for further comparisons. 
20 
Neural Solver Selection for Combinatorial Optimization 

Table 10: Empirical results on synthetic TSP and TSPLIB datasets, reporting the mean (standard deviation) over five independent runs. All individual solvers are included for comparison. The suffixes ’-N100’ and ’-N500’ indicate models trained on datasets with $\Nu { = } 1 0 0$ and $\mathrm { N } { = } 5 0 0$ , respectively. Obj denotes the average objective value on the dataset.

<table><tr><td rowspan="2">Methods</td><td colspan="3">TSP</td></tr><tr><td>Obj</td><td>Gap</td><td>Time</td></tr><tr><td>BQ</td><td>8.13</td><td>3.00%</td><td>1.61s</td></tr><tr><td>ELG</td><td>8.18</td><td>3.70%</td><td>0.45s</td></tr><tr><td>LEHD</td><td>8.17</td><td>3.57%</td><td>0.86s</td></tr><tr><td>T2T-N100</td><td>8.08</td><td>2.48%</td><td>1.71s</td></tr><tr><td>T2T-N500</td><td>8.08</td><td>2.40%</td><td>1.98s</td></tr><tr><td>DIFUSCO-N100</td><td>8.11</td><td>2.84%</td><td>1.46s</td></tr><tr><td>DIFUSCO-N500</td><td>8.07</td><td>2.33%</td><td>1.45s</td></tr><tr><td>Oracle</td><td>7.99</td><td>1.24%</td><td>8.93s</td></tr><tr><td colspan="4">Selection by classification</td></tr><tr><td>Greedy</td><td>8.04 (0.00)</td><td>1.94% (0.02%)</td><td>1.36s (0.01s)</td></tr><tr><td>Top-k (k=2)</td><td>8.01 (0.00)</td><td>1.53% (0.01%)</td><td>2.52s (0.04s)</td></tr><tr><td>Rejection (20%)</td><td>8.03 (0.00)</td><td>1.81% (0.01%)</td><td>1.63s (0.01s)</td></tr><tr><td>Top-p (p=0.5)</td><td>8.03 (0.00)</td><td>1.84% (0.03%)</td><td>1.55s (0.06s)</td></tr><tr><td colspan="4">Selection by ranking</td></tr><tr><td>Greedy</td><td>8.04 (0.00)</td><td>1.86% (0.01%)</td><td>1.33s (0.01s)</td></tr><tr><td>Top-k (k=2)</td><td>8.01 (0.00)</td><td>1.51% (0.02%)</td><td>2.56s (0.03s)</td></tr><tr><td>Rejection (20%)</td><td>8.03 (0.00)</td><td>1.75% (0.02%)</td><td>1.63s (0.01s)</td></tr><tr><td>Top-p (p=0.5)</td><td>8.02 (0.00)</td><td>1.68% (0.02%)</td><td>1.86s (0.07s)</td></tr></table>
<table><tr><td rowspan="2">Methods</td><td colspan="3">TSPLIB</td></tr><tr><td>Obj</td><td>Gap</td><td>Time</td></tr><tr><td>BQ</td><td>8.29</td><td>3.04%</td><td>1.44s</td></tr><tr><td>ELG</td><td>8.29</td><td>3.05%</td><td>0.40s</td></tr><tr><td>LEHD</td><td>8.26</td><td>2.57%</td><td>0.88s</td></tr><tr><td>T2T-N100</td><td>8.22</td><td>2.09%</td><td>1.76s</td></tr><tr><td>T2T-N500</td><td>8.21</td><td>1.95%</td><td>1.74s</td></tr><tr><td>DIFUSCO-N100</td><td>8.23</td><td>2.25%</td><td>1.44s</td></tr><tr><td>DIFUSCO-N500</td><td>8.22</td><td>2.13%</td><td>1.44s</td></tr><tr><td>Oracle</td><td>8.12</td><td>0.89%</td><td>9.14s</td></tr></table>
<table><tr><td colspan="4">Selection by classification</td></tr><tr><td>Greedy</td><td>8.17 (0.00)</td><td>1.54% (0.05%)</td><td>1.33s (0.02s)</td></tr><tr><td>Top-k (k = 2)</td><td>8.15 (0.01)</td><td>1.22% (0.10%)</td><td>2.47s (0.02s)</td></tr><tr><td>Rejection (20%)</td><td>8.16 (0.01)</td><td>1.42% (0.11%)</td><td>1.54s (0.03s)</td></tr><tr><td>Top-p (p = 0.5)</td><td>8.17 (0.01)</td><td>1.49% (0.11%)</td><td>1.37s (0.02s)</td></tr></table>
<table><tr><td colspan="4">Selection by ranking</td></tr><tr><td>Greedy</td><td>8.16 (0.00)</td><td>1.33% (0.06%)</td><td>1.28s (0.03s)</td></tr><tr><td>Top-k (k=2)</td><td>8.14 (0.00)</td><td>1.07% (0.03%)</td><td>2.48s (0.02s)</td></tr><tr><td>Rejection (20%)</td><td>8.15 (0.00)</td><td>1.26% (0.03%)</td><td>1.51s (0.04s)</td></tr><tr><td>Top-p (p=0.5)</td><td>8.15 (0.00)</td><td>1.28% (0.04%)</td><td>1.46s (0.06s)</td></tr></table>
Neural Solver Selection for Combinatorial Optimization 

Table 11: Empirical results on synthetic CVRP and CVRPLIB Set-X datasets, reporting the mean (standard deviation) over five independent runs. All individual solvers are included for comparison. Obj denotes the average objective value on the dataset.

<table><tr><td rowspan="2">Methods</td><td colspan="3">CVRP</td></tr><tr><td>Obj</td><td>Gap</td><td>Time</td></tr><tr><td>BQ</td><td>18.39</td><td>7.20%</td><td>1.59s</td></tr><tr><td>ELG</td><td>18.49</td><td>7.81%</td><td>0.82s</td></tr><tr><td>LEHD</td><td>18.42</td><td>7.37%</td><td>1.01s</td></tr><tr><td>MVMoE</td><td>19.48</td><td>13.56%</td><td>0.70s</td></tr><tr><td>Omni</td><td>18.32</td><td>6.82%</td><td>0.24s</td></tr><tr><td>Oracle</td><td>17.95</td><td>4.64%</td><td>4.38s</td></tr><tr><td colspan="4">Selection by classification</td></tr><tr><td>Greedy</td><td>18.07 (0.00)</td><td>5.35% (0.02%)</td><td>0.64s (0.01s)</td></tr><tr><td>Top-k (k = 2)</td><td>17.98 (0.00)</td><td>4.81% (0.01%)</td><td>1.87s (0.03s)</td></tr><tr><td>Rejection (20%)</td><td>18.04 (0.01)</td><td>5.19% (0.03%)</td><td>0.77s (0.01s)</td></tr><tr><td>Top-p (p = 0.8)</td><td>18.04 (0.01)</td><td>5.16% (0.03%)</td><td>0.87s (0.08s)</td></tr><tr><td colspan="4">Selection by ranking</td></tr><tr><td>Greedy</td><td>18.06 (0.00)</td><td>5.31% (0.01%)</td><td>0.62s (0.01s)</td></tr><tr><td>Top-k (k = 2)</td><td>17.98 (0.00)</td><td>4.82% (0.01%)</td><td>1.90s (0.04s)</td></tr><tr><td>Rejection (20%)</td><td>18.04 (0.00)</td><td>5.15% (0.02%)</td><td>0.74s (0.01s)</td></tr><tr><td>Top-p (p = 0.8)</td><td>18.01 (0.00)</td><td>4.99% (0.02%)</td><td>1.03s (0.03s)</td></tr><tr><td colspan="4"></td></tr><tr><td rowspan="2">Methods</td><td colspan="3">CVRPLIB Set-X</td></tr><tr><td>Obj</td><td>Gap</td><td>Time</td></tr><tr><td>BQ</td><td>71.21</td><td>10.31%</td><td>2.60s</td></tr><tr><td>ELG</td><td>68.50</td><td>6.10%</td><td>1.31s</td></tr><tr><td>LEHD</td><td>73.40</td><td>13.70%</td><td>1.60s</td></tr><tr><td>MVMoE</td><td>74.59</td><td>15.54%</td><td>0.90s</td></tr><tr><td>Omni</td><td>68.57</td><td>6.21%</td><td>0.38s</td></tr><tr><td>Oracle</td><td>67.85</td><td>5.10%</td><td>6.81s</td></tr><tr><td colspan="4">Selection by classification</td></tr><tr><td>Greedy</td><td>68.41 (0.08)</td><td>5.96% (0.12%)</td><td>1.06s (0.08s)</td></tr><tr><td>Top-k (k = 2)</td><td>68.07 (0.05)</td><td>5.44% (0.08%)</td><td>2.40s (0.25s)</td></tr><tr><td>Rejection (20%)</td><td>68.32 (0.08)</td><td>5.83% (0.12%)</td><td>1.31s (0.09s)</td></tr><tr><td>Top-p (p = 0.8)</td><td>68.30 (0.06)</td><td>5.79% (0.09%)</td><td>1.42s (0.17s)</td></tr><tr><td colspan="4">Selection by ranking</td></tr><tr><td>Greedy</td><td>68.28 (0.03)</td><td>5.76% (0.04%)</td><td>1.31s (0.10s)</td></tr><tr><td>Top-k (k = 2)</td><td>68.04 (0.04)</td><td>5.39% (0.06%)</td><td>2.56s (0.13s)</td></tr><tr><td>Rejection (20%)</td><td>68.19 (0.03)</td><td>5.63% (0.05%)</td><td>1.60s (0.08s)</td></tr><tr><td>Top-p (p = 0.8)</td><td>68.18 (0.02)</td><td>5.61% (0.03%)</td><td>1.72s (0.08s)</td></tr></table>