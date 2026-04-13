arXiv:1911.04462v3 [cs.LG] 2 Jul 2020 
Neural Contextual Bandits with UCB-based Exploration 
# Dongruo Zhou 1 Lihong Li 2 Quanquan Gu 1
# Abstract
We study the stochastic contextual bandit problem, where the reward is generated from an unknown function with additive noise. No assumption is made about the reward function other than boundedness. We propose a new algorithm, NeuralUCB, which leverages the representation power of deep neural networks and uses a neural network-based random feature mapping to construct an upper confidence bound (UCB) of reward for efficient exploration. We prove that, under standard assumptions, NeuralUCB achieves $\widetilde { O } ( \sqrt { T } )$ regret, where $T$ is the number of rounds. To the best of our knowledge, it is the first neural network-based contextual bandit algorithm with a near-optimal regret guarantee. We also show the algorithm is empirically competitive against representative baselines in a number of benchmarks. 
# 1. Introduction
The stochastic contextual bandit problem has been extensively studied in machine learning (Langford & Zhang, 2008; Bubeck & Cesa-Bianchi, 2012; Lattimore & Szepesvari ´ , 2019): at round $t \in \{ 1 , 2 , \ldots , T \}$ , an agent is presented with a set of $K$ actions, each of which is associated with a $d$ -dimensional feature vector. After choosing an action, the agent will receive a stochastic reward generated from some unknown distribution conditioned on the action’s feature vector. The goal of the agent is to maximize the expected cumulative rewards over $T$ rounds. Contextual bandit algorithms have been applied to many real-world applications, such as personalized recommendation, advertising and Web search. 
The most studied model in the literature is linear contextual bandits (Auer, 2002; Abe et al., 2003; Dani et al., 2008; Rusmevichientong & Tsitsiklis, 2010), which assumes that 
1Department of Computer Science, University of California, Los Angeles, CA 90095, USA 2Google Research, USA. Correspondence to: Quanquan Gu <qgu@cs.ucla.edu>. 
Proceedings of the ${ { \it 3 7 } ^ { t h } }$ International Conference on Machine Learning, Vienna, Austria, PMLR 119, 2020. Copyright 2020 by the author(s). 
the expected reward at each round is linear in the feature vector. While successful in both theory and practice (Li et al., 2010; Chu et al., 2011; Abbasi-Yadkori et al., 2011), the linear-reward assumption it makes often fails to hold in practice, which motivates the study of nonlinear or nonparametric contextual bandits (Filippi et al., 2010; Srinivas et al., 2010; Bubeck et al., 2011; Valko et al., 2013). However, they still require fairly restrictive assumptions on the reward function. For instance, Filippi et al. (2010) make a generalized linear model assumption on the reward, Bubeck et al. (2011) require it to have a Lipschitz continuous property in a proper metric space, and Valko et al. (2013) assume the reward function belongs to some Reproducing Kernel Hilbert Space (RKHS). 
In order to overcome the above shortcomings, deep neural networks (DNNs) (Goodfellow et al., 2016) have been introduced to learn the underlying reward function in contextual bandit problem, thanks to their strong representation power. We call these approaches collectively as neural contextual bandit algorithms. Given the fact that DNNs enable the agent to make use of nonlinear models with less domain knowledge, existing work (Riquelme et al., 2018; Zahavy & Mannor, 2019) study neural-linear bandits. That is, they use all but the last layers of a DNN as a feature map, which transforms contexts from the raw input space to a low-dimensional space, usually with better representation and less frequent updates. Then they learn a linear exploration policy on top of the last hidden layer of the DNN with more frequent updates. These attempts have achieved great empirical success, but no regret guarantees are provided. 
In this paper, we consider provably efficient neural contextual bandit algorithms. The new algorithm, NeuralUCB, uses a neural network to learn the unknown reward function, and follows a UCB strategy for exploration. At the core of the algorithm is the novel use of DNN-based random feature mappings to construct the UCB. Its regret analysis is built on recent advances on optimization and generalization of deep neural networks (Jacot et al., 2018; Arora et al., 2019; Cao & Gu, 2019). Crucially, the analysis makes no modeling assumptions about the reward function, other than that it be bounded. While the main focus of our paper is theoretical, we also show in a few benchmark problems the effectiveness of NeuralUCB, and demonstrate its benefits against several representative baselines. 
Neural Contextual Bandits with UCB-based Exploration 
Our main contributions are as follows: 
• We propose a neural contextual bandit algorithm that can be regarded as an extension of existing (generalized) linear bandit algorithms (Abbasi-Yadkori et al., 2011; Filippi et al., 2010; Li et al., 2010; 2017) to the case of arbitrary bounded reward functions. 
• We prove that, under standard assumptions, our algorithm is able to achieve $\widetilde { O } ( \widetilde { d } \sqrt { T } )$ regret, where $\widetilde { d }$ is the effective dimension of a neural tangent kernel matrix and $T$ is the number of rounds. The bound recovers the ex-√ isting $\widetilde { O } ( d \sqrt { T } )$ regret for linear contextual bandit as a special case (Abbasi-Yadkori et al., 2011), where $d$ is the dimension of context. 
• We demonstrate empirically the effectiveness of the algorithm in both synthetic and benchmark problems. 
Notation: Scalars are denoted by lower case letters, vectors by lower case bold face letters, and matrices by upper case bold face letters. For a positive integer $k$ , $[ k ]$ denotes $\{ 1 , \ldots , k \}$ . For a vector $\pmb \theta \in \mathbb R ^ { d }$ , we denote its $\ell _ { 2 }$ norm by $\begin{array} { r } { \| \pmb { \theta } \| _ { 2 } = \sqrt { \sum _ { i = 1 } ^ { d } \theta _ { i } ^ { 2 } } } \end{array}$ Pi=1 and its $j$ -th coordinate by $[ \pmb { \theta } ] _ { j }$ . For a matrix $\mathbf { A } \in \mathbb { R } ^ { d \times d }$ , we denote its spectral norm, Frobenius norm, and $( i , j )$ -th entry by $\| \mathbf { A } \| _ { 2 } , \| \mathbf { A } \| _ { F }$ , and $[ \mathbf { A } ] _ { i , j }$ , respectively. We denote a sequence of vectors by $\{ \pmb { \theta } _ { j } \} _ { j = 1 } ^ { t }$ , and similarly for matrices. For two sequences $\left\{ a _ { n } \right\}$ and $\left\{ b _ { n } \right\}$ , we use $a _ { n } = O ( b _ { n } )$ to denote that there exists some constant $C > 0$ such that $a _ { n } \leq C b _ { n }$ ; similarly, $a _ { n } = \Omega ( b _ { n } )$ means there exists some constant $C ^ { \prime } > 0$ such that $a _ { n } \geq C ^ { \prime } b _ { n }$ . In addition, we use ${ \widetilde { O } } ( \cdot )$ to hide logarithmic factors. We say a random variable $X$ is $\nu$ -sub-Gaussian if $\mathbb { E } \exp ( \lambda ( X - \mathbb { E } X ) ) \le \exp ( \lambda ^ { 2 } \nu ^ { 2 } / 2 )$ for any $\lambda > 0$ . 
# 2. Problem Setting
We consider the stochastic $K$ -armed contextual bandit problem, where the total number of rounds $T$ is known. At round $t \in [ T ]$ , the agent observes the context consisting of $K$ feature vectors: $\{ \mathbf { x } _ { t , a } \in \mathbb { R } ^ { d } \mid a \in [ K ] \}$ . The agent selects an action $a _ { t }$ and receives a reward $r _ { t , a _ { t } }$ . For brevity, we denote by $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { T K }$ the collection of $\left\{ \mathbf { x } _ { 1 , 1 } , \mathbf { x } _ { 1 , 2 } , \ldots , \mathbf { x } _ { T , K } \right\}$ . Our goal is to maximize the following pseudo regret (or regret for short): 
$$
R _ {T} = \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \left(r _ {t, a _ {t} ^ {*}} - r _ {t, a _ {t}}\right) \right], \tag {2.1}
$$
where $a _ { t } ^ { * } = \mathrm { a r g m a x } _ { a \in [ K ] } \mathbb { E } [ r _ { t , a } ]$ is the optimal action at round $t$ that maximizes the expected reward. 
This work makes the following assumption about reward generation: for any round $t$ , 
$$
r _ {t, a _ {t}} = h \left(\mathbf {x} _ {t, a _ {t}}\right) + \xi_ {t}, \tag {2.2}
$$
where $h$ is an unknown function satisfying $0 \leq h ( \mathbf { x } ) \leq 1$ for any $\mathbf { x }$ , and $\xi _ { t }$ is $\nu$ -sub-Gaussian noise conditioned on $\mathbf { x } _ { 1 , a _ { 1 } } , \ldots , \mathbf { x } _ { t - 1 , a _ { t - 1 } }$ satisfying $\mathbb { E } \xi _ { t } ~ = ~ 0$ . The $\nu$ -sub-Gaussian assumption for $\xi _ { t }$ is standard in the stochastic bandit literature (e.g., Abbasi-Yadkori et al., 2011; Li et al., 2017), and is satisfied by, for example, any bounded noise. The bounded $h$ assumption holds true when $h$ belongs to linear functions, generalized linear functions, Gaussian processes, and kernel functions with bounded RKHS norm over a bounded domain, among others. 
In order to learn the reward function $h$ in (2.2), we propose to use a fully connected neural networks with depth $L \geq 2$ 
$$
f (\mathbf {x}; \boldsymbol {\theta}) = \sqrt {m} \mathbf {W} _ {L} \sigma \left(\mathbf {W} _ {L - 1} \sigma \left(\dots \sigma \left(\mathbf {W} _ {1} \mathbf {x}\right)\right)\right), \tag {2.3}
$$
where $\sigma ( x ) ~ = ~ \operatorname* { m a x } \{ x , 0 \}$ is the rectified linear unit (ReLU) activation function, $\begin{array} { r l r } { { \bf W } _ { 1 } } & { { } \in } & { \mathbb { R } ^ { m \times d } , { \bf W } _ { i } \quad \in } \end{array}$ $\mathbb { R } ^ { m \times m } , 2 \ \leq \ i \ \leq \ L - 1 , \mathbf { W } _ { L } \ \in \ \mathbb { R } ^ { m \times 1 }$ , and $\theta \ =$ $\big [ \mathrm { v e c } ( \mathbf { W } _ { 1 } ) ^ { \top } , \ldots , \mathrm { v e c } ( \mathbf { W } _ { L } ) ^ { \top } \big ] ^ { \top } \in \mathbb { R } ^ { p }$ with $p = m + m d +$ $m ^ { 2 } ( L - 1 )$ . Without loss of generality, we assume that the width of each hidden layer is the same (i.e., $m$ ) for convenience in analysis. We denote the gradient of the neural network function by $\mathbf { g } ( \mathbf { x } ; \pmb { \theta } ) = \nabla _ { \pmb { \theta } } f ( \mathbf { x } ; \pmb { \theta } ) \in \mathbb { R } ^ { p }$ . 
# 3. The NeuralUCB Algorithm
The key idea of NeuralUCB (Algorithm 1) is to use a neural network $f ( \mathbf { x } ; \pmb { \theta } )$ to predict the reward of context $\mathbf { x }$ , and upper confidence bounds computed from the network to guide exploration (Auer, 2002). 
Initialization It initializes the network by randomly generating each entry of $\pmb \theta$ from an appropriate Gaussian distribution: for $1 \leq l \leq L - 1 ,$ $\mathbf { W } _ { l }$ is set to be $\left( \begin{array} { c c } { { { \bf W } } } & { { { \bf 0 } } } \\ { { { \bf 0 } } } & { { { \bf W } } } \end{array} \right)$ where each entry of W is generated independently from $N ( 0 , 4 / m ) ; \mathbf { W } _ { L }$ is set to $( \mathbf { w } ^ { \top } , - \mathbf { w } ^ { \top } )$ , where each entry of w is generated independently from $N ( 0 , 2 / m )$ . 
Learning At round $t$ , Algorithm 1 observes the contexts for all actions, $\{ \mathbf { x } _ { t , a } \} _ { a = 1 } ^ { K }$ . First, it computes an upper confidence bound $U _ { t , a }$ for each action $a$ , based on $\mathbf { x } _ { t , a }$ , $\theta _ { t - 1 }$ (the current neural network parameter), and a positive scaling factor $\gamma _ { t - 1 }$ . It then chooses action $a _ { t }$ with the largest $U _ { t , a }$ and receives the corresponding reward $r _ { t , a _ { t } }$ . At the end of round $t$ , NeuralUCB updates $\theta _ { t }$ by applying Algorithm 2 to (approximately) minimize $L ( \pmb \theta )$ using gradient descent, and updates $\gamma _ { t }$ . We choose gradient descent in Algorithm 2 for the simplicity of analysis, although the training method can be replaced by stochastic gradient descent with a more involved analysis (Allen-Zhu et al., 2019; Zou et al., 2019). 
Neural Contextual Bandits with UCB-based Exploration 

Algorithm 1 NeuralUCB

1: Input: Number of rounds $T$ , regularization parameter $\lambda$ , exploration parameter $\nu$ , confidence parameter $\delta$ , norm parameter $S$ , step size $\eta$ , number of gradient descent steps $J$ , network width $m$ , network depth $L$ .  
2: Initialization: Randomly initialize $\theta_0$ as described in the text  
3: Initialize $\mathbf{Z}_0 = \lambda \mathbf{I}$ 4: for $t = 1, \dots, T$ do  
5: Observe $\{\mathbf{x}_{t,a}\}_{a=1}^K$ 6: for $a = 1, \dots, K$ do  
7: Compute $U_{t,a} = f(\mathbf{x}_{t,a}; \theta_{t-1}) + \gamma_{t-1} \sqrt{\mathbf{g}(\mathbf{x}_{t,a}; \theta_{t-1})^\top \mathbf{Z}_{t-1}^{-1} \mathbf{g}(\mathbf{x}_{t,a}; \theta_{t-1}) / m}$ 8: Let $a_t = \operatorname{argmax}_{a \in [K]} U_{t,a}$ 9: end for  
10: Play $a_t$ and observe reward $r_{t,a}$ 11: Compute $\mathbf{Z}_t = \mathbf{Z}_{t-1} + \mathbf{g}(\mathbf{x}_{t,a}; \theta_{t-1}) \mathbf{g}(\mathbf{x}_{t,a}; \theta_{t-1})^\top / m$ 12: Let $\theta_t = \mathrm{TrainNN}(\lambda, \eta, J, m, \{\mathbf{x}_{i,a_i}\}_{i=1}^t, \{r_{i,a_i}\}_{i=1}^t, \theta_0)$ 13: Compute  
\[
\gamma_t = \sqrt{1 + C_1 m^{-1/6} \sqrt{\log m} L^4 t^{7/6} \lambda^{-7/6}} \cdot \left( \nu \sqrt{\log \frac{\det \mathbf{Z}_t}{\det \lambda I}} + C_2 m^{-1/6} \sqrt{\log m} L^4 t^{5/3} \lambda^{-1/6} - 2 \log \delta + \sqrt{\lambda} S \right)
+ (\lambda + C_3 t L) \left[ (1 - \eta m \lambda)^{J/2} \sqrt{t / \lambda} + m^{-1/6} \sqrt{\log m} L^{7/2} t^{5/3} \lambda^{-5/3} (1 + \sqrt{t / \lambda}) \right]. 

14: end for

Algorithm 2 TrainNN $(\lambda ,\eta ,U,m,\{\mathbf{x}_{i,a_i}\}_{i = 1}^t,\{r_{i,a_i}\}_{i = 1}^t,\pmb{\theta}^{(0)})$ 1: Input: Regularization parameter $\lambda$ , step size $\eta$ , number of gradient descent steps $U$ , network width $m$ , contexts $\{\mathbf{x}_{i,a_i}\}_{i = 1}^t$ rewards $\{r_{i,a_i}\}_{i = 1}^t$ , initial parameter $\pmb{\theta}^{(0)}$ 2: Define $\mathcal{L}(\pmb {\theta}) = \sum_{i = 1}^{t}(f(\mathbf{x}_{i,a_i};\pmb {\theta}) - r_{i,a_i})^2 /2+$ $m\lambda \| \pmb {\theta} - \pmb{\theta}^{(0)}\| _2^2 /2.$ 3: for $j = 0,\dots ,J - 1$ do   
4: $\pmb{\theta}^{(j + 1)} = \pmb{\theta}^{(j)} - \eta \nabla \mathcal{L}(\pmb{\theta}^{(j)})$ 5: end for   
6: Return $\pmb{\theta}^{(J)}$ 
Comparison with Existing Algorithms We compare NeuralUCB with other neural contextual bandit algorithms. Allesiardo et al. (2014) proposed NeuralBandit which consists of $K$ neural networks. It uses a committee of networks to compute the score of each action and chooses an action with the $\epsilon$ -greedy strategy. In contrast, our NeuralUCB uses upper confidence bound-based exploration, which is more effective than $\epsilon$ -greedy. In addition, our algorithm only uses one neural network instead of $K$ networks, thus can be computationally more efficient. 
Lipton et al. (2018) used Thompson sampling on deep neural networks (through variational inference) in reinforcement learning; a variant is proposed by Azizzadenesheli et al. (2018) that works well on a set of Atari benchmarks. Riquelme et al. (2018) proposed NeuralLinear, which uses the first $L - 1$ layers of a $L$ -layer DNN to learn a representation, then applies Thompson sampling on the last layer to 
choose action. Zahavy & Mannor (2019) proposed a NeuralLinear with limited memory (NeuralLinearLM), which also uses the first $L - 1$ layers of a $L$ -layer DNN to learn a representation and applies Thompson sampling on the last layer. Instead of computing the exact mean and variance in Thompson sampling, NeuralLinearLM only computes their approximation. Unlike NeuralLinear and NeuralLinearLM, NeuralUCB uses the entire DNN to learn the representation and constructs the upper confidence bound based on the random feature mapping defined by the neural network gradient. Finally, Kveton et al. (2020) studied the use of reward perturbation for exploration in neural network-based bandit algorithms. 
A Variant of NeuralUCB called NeuralUCB0 is described in Appendix E. It can be viewed as a simplified version of NeuralUCB where only the first-order Taylor approximation of the neural network around the initialized parameter is updated through online ridge regression. In this sense, NeuralUCB0 can be seen as KernelUCB (Valko et al., 2013) specialized to the Neural Tangent Kernel (Jacot et al., 2018), or LinUCB (Li et al., 2010) with Neural Tangent Random Features (Cao & Gu, 2019). 
While this variant has a comparable regret bound as NeuralUCB, we expect the latter to be stronger in practice. Indeed, as shown by Allen-Zhu & Li (2019), the Neural Tangent Kernel does not seem to completely realize the representation power of neural networks in supervised learning. A similar phenomenon will be demonstrated for contextual bandit learning in Section 7. 
Neural Contextual Bandits with UCB-based Exploration 
# 4. Regret Analysis
This section analyzes the regret of NeuralUCB. Recall that $\{ \mathbf { x } ^ { i } \} _ { i = 1 } ^ { T K }$ is the collection of all $\{ \mathbf { x } _ { t , a } \}$ . Our regret analysis is built upon the recently proposed neural tangent kernel matrix (Jacot et al., 2018): 
Definition 4.1 (Jacot et al. (2018); Cao & Gu (2019)). Let $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { T K }$ be a set of contexts. Define 
$$
\widetilde {\mathbf {H}} _ {i, j} ^ {(1)} = \boldsymbol {\Sigma} _ {i, j} ^ {(1)} = \langle \mathbf {x} ^ {i}, \mathbf {x} ^ {j} \rangle , \qquad \mathbf {A} _ {i, j} ^ {(l)} = \left( \begin{array}{c c} \boldsymbol {\Sigma} _ {i, i} ^ {(l)} & \boldsymbol {\Sigma} _ {i, j} ^ {(l)} \\ \boldsymbol {\Sigma} _ {i, j} ^ {(l)} & \boldsymbol {\Sigma} _ {j, j} ^ {(l)} \end{array} \right),
$$
$$
\boldsymbol {\Sigma} _ {i, j} ^ {(l + 1)} = 2 \mathbb {E} _ {(u, v) \sim N (\mathbf {0}, \mathbf {A} _ {i, j} ^ {(l)})} \left[ \sigma (u) \sigma (v) \right],
$$
$$
\widetilde {\mathbf {H}} _ {i, j} ^ {(l + 1)} = 2 \widetilde {\mathbf {H}} _ {i, j} ^ {(l)} \mathbb {E} _ {(u, v) \sim N (\mathbf {0}, \mathbf {A} _ {i, j} ^ {(l)})} [ \sigma^ {\prime} (u) \sigma^ {\prime} (v) ] + \pmb {\Sigma} _ {i, j} ^ {(l + 1)}.
$$
Then, $\mathbf { H } = ( \widetilde { \mathbf { H } } ^ { ( L ) } + \boldsymbol { \Sigma } ^ { ( L ) } ) / 2$ is called the neural tangent kernel (NTK) matrix on the context set. 
In the above definition, the Gram matrix H of the NTK on the contexts {xi}T Ki=1 for L-layer neural networks is defined $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { T K }$ $L$ recursively from the input layer all the way to the output layer of the network. Interested readers are referred to Jacot et al. (2018) for more details about neural tangent kernels. 
With Definition 4.1, we may state the following assumption on the contexts: $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { T K }$ . 
Assumption 4.2. $\mathbf { H } \succeq \lambda _ { 0 } \mathbf { I }$ . Moreover, for any $1 \leq i \leq$ $T K$ , $\| \mathbf { x } ^ { i } \| _ { 2 } = 1$ and $[ \mathbf { x } ^ { i } ] _ { j } = [ \mathbf { x } ^ { i } ] _ { j + d / 2 }$ . 
The first part of the assumption says that the neural tangent kernel matrix is non-singular, a mild assumption commonly made in the related literature (Du et al., 2019a; Arora et al., 2019; Cao & Gu, 2019). It can be satisfied as long as no two contexts in {xi}T Ki=1 $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { T K }$ are parallel. The second part is also mild and is just for convenience in analysis: for any context x, $\| \mathbf { x } \| _ { 2 } = 1$ , we can always construct a new context $\mathbf { x } ^ { \prime } = [ \mathbf { x } ^ { \top } , \mathbf { x } ^ { \top } ] ^ { \top } / \sqrt { 2 }$ to satisfy Assumption 4.2. It can be verified that if $\pmb { \theta } _ { 0 }$ is initialized as in NeuralUCB, then $f ( \mathbf { x } ^ { i } ; \pmb { \theta } _ { 0 } ) = 0$ for any $i \in [ T K ]$ . 
Next we define the effective dimension of the neural tangent kernel matrix. 
Definition 4.3. The effective dimension $\widetilde { d }$ of the neural tangent kernel matrix on contexts $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { T K }$ e is defined as 
$$
\widetilde {d} = \frac {\log \det  (\mathbf {I} + \mathbf {H} / \lambda)}{\log (1 + T K / \lambda)}. \tag {4.1}
$$
Remark 4.4. The notion of effective dimension was first introduced by Valko et al. (2013) for analyzing kernel contextual bandits, which was defined by the eigenvalues of any kernel matrix restricted to the given contexts. We adapt a similar but different definition of Yang & Wang (2019), 
which was used for the analysis of kernel-based Q-learning. Suppose the dimension of the reproducing kernel Hilbert space induced by the given kernel is $\widehat { d }$ and the feature mapping $\psi : \mathbb { R } ^ { d } \to \mathbb { R } ^ { \hat { d } }$ induced by the given kernel satisfies $\| \psi ( \mathbf { x } ) \| _ { 2 } \leq 1$ for any $\mathbf { x } \in \mathbb { R } ^ { d }$ . Then, it can be verified that $\widetilde d \leq \widehat d$ , as shown in Appendix A.1. Intuitively, $\widetilde { d }$ measures how quickly the eigenvalues of H diminish, and only depends on $T$ logarithmically in several special cases (Valko et al., 2013). 
Now we are ready to present the main result, which provides the regret bound $R _ { T }$ of Algorithm 1. 
Theorem 4.5. Let $\tilde { d }$ be the effective dimension, and $\mathbf { h } =$ $[ h ( \mathbf { x } ^ { i } ) ] _ { i = 1 } ^ { T K } \in \mathbb { R } ^ { T K }$ e. There exist constant $C _ { 1 } , C _ { 2 } > 0$ , such that for any $\delta \in ( 0 , 1 )$ , if 
$$
m \geq \operatorname {p o l y} (T, L, K, \lambda^ {- 1}, \lambda_ {0} ^ {- 1}, S ^ {- 1}, \log (1 / \delta)), \tag {4.2}
$$
$$
\eta = C _ {1} (m T L + m \lambda) ^ {- 1},
$$
$\lambda \geq \operatorname* { m a x } \{ 1 , S ^ { - 2 } \}$ , and $S \geq \sqrt { 2 \mathbf { h } ^ { \top } \mathbf { H } ^ { - 1 } \mathbf { h } }$ , then with probability at least $1 - \delta$ , the regret of Algorithm 1 satisfies 
$$
\begin{array}{l} R _ {T} \leq 3 \sqrt {T} \sqrt {\tilde {d} \log (1 + T K / \lambda) + 2} \\ \cdot \left[ \nu \sqrt {\tilde {d} \log (1 + T K / \lambda) + 2 - 2 \log \delta} \right. \\ + (\lambda + C _ {2} T L) (1 - \lambda / (T L)) ^ {J / 2} \sqrt {T / \lambda} \\ \left. + + 2 \sqrt {\lambda} S \right] + 1. \tag {4.3} \\ \end{array}
$$
Remark 4.6. It is worth noting that, simply applying results for linear bandits to our algorithm would lead to a linear dependence of $p$ or $\sqrt { p }$ in the regret. Such a bound is vacuous since in our setting $p$ would be very large compared with the number of rounds $T$ and the input context dimension $d$ . In contrast, our regret bound only depends on $\hat { d }$ , which can be much smaller than $p$ . 
Remark 4.7. Our regret bound (4.3) has a term $( \lambda +$ $C _ { 2 } T L ) ( 1 - \lambda / ( T L ) ) ^ { \overline { { J } } / 2 } \sqrt { T / \lambda }$ , which characterizes the optimization error of Algorithm 2 after $J$ iterations. Setting 
$$
J = 2 \log \frac {\lambda S}{\sqrt {T} (\lambda + C _ {2} T L)} \frac {T L}{\lambda} = \widetilde {O} (T L / \lambda), \tag {4.4}
$$
which is independent of $m$ , we have $( \lambda + C _ { 2 } T L ) ( 1 -$ $\lambda / ( T L ) ) ^ { J / 2 } { \sqrt { T / \lambda } } \leq { \sqrt { \lambda } } S$ , so the optimization error is dominated by $\sqrt { \lambda } S$ . Hence, the order of the regret bound is not affected by the error of optimization. 
Remark 4.8. With $\nu$ and $\lambda$ treated as constants, $S \ =$ $\sqrt { 2 \mathbf { h } ^ { \top } \mathbf { H } ^ { - 1 } \mathbf { h } }$ , and $J$ given in (4.4), the regret bound (4.3) becomes $R _ { T } = \widetilde { O } \Big ( \sqrt { \widetilde { d T } } \sqrt { \operatorname* { m a x } \{ \widetilde { d } , \mathbf { h } ^ { \top } \mathbf { H } ^ { - 1 } \mathbf { h } \} } \Big )$ . Specifically, if $h$ belongs to the RKHS $\mathcal { H }$ induced by the neural 
Neural Contextual Bandits with UCB-based Exploration 
tangent kernel with bounded RKHS norm $\| h \| _ { \mathcal { H } }$ , we have $\| h \| _ { \mathcal { H } } \ge \sqrt { \mathbf { h } ^ { \top } \mathbf { H } ^ { - 1 } \mathbf { h } }$ ; see Appendix A.2 for more details. Thus our regret bound can be further written as 
$$
R _ {T} = \widetilde {O} \left(\sqrt {\tilde {d} T} \sqrt {\max  \{\tilde {d} , \| h \| _ {\mathcal {H}} \}}\right). \tag {4.5}
$$
The high-probability result in Theorem 4.5 can be used to obtain a bound on the expected regret. 
Corollary 4.9. Under the same conditions in Theorem 4.5, there exists a positive constant $C$ such that 
$$
\begin{array}{l} \mathbb {E} [ R _ {T} ] \\ \leq 2 + 3 \sqrt {T} \sqrt {\tilde {d} \log (1 + T K / \lambda) + 2} \\ \cdot \left[ \nu \sqrt {\tilde {d} \log (1 + T K / \lambda) + 2 + 2 \log T} \right. \\ \left. + 2 \sqrt {\lambda} S + (\lambda + C T L) (1 - \lambda / (T L)) ^ {J / 2} \sqrt {T / \lambda} \right]. \\ \end{array}
$$
# 5. Proof of Main Result
This section outlines the proof of Theorem 4.5, which has to deal with the following technical challenges: 
• We do not make parametric assumptions on the reward function as some previous work (Filippi et al., 2010; Chu et al., 2011; Abbasi-Yadkori et al., 2011). 
• To avoid strong parametric assumptions, we use overparameterized neural networks, which implies $m$ (and thus $p \mathrm { , }$ ) is very large. Therefore, we need to make sure the regret bound is independent of $m$ . 
• Unlike the static feature mapping used in kernel bandit algorithms (Valko et al., 2013), NeuralUCB uses a neural network $f ( \mathbf { x } ; \pmb { \theta } _ { t } )$ and its gradient $\mathbf { g } ( \mathbf { x } ; \pmb { \theta } _ { t } )$ as a dynamic feature mapping depending on $\theta _ { t }$ . This difference makes the analysis of NeuralUCB more difficult. 
These challenges are addressed by the following technical lemmas, whose proofs are gathered in the appendix. 
Lemma 5.1. There exists a positive constant $\bar { C }$ such that for any $\delta \in ( 0 , 1 )$ , if $m \ge \bar { C } T ^ { 4 } K ^ { 4 } L ^ { 6 } \log ( T ^ { 2 } K ^ { 2 } L / \delta ) / \lambda _ { 0 } ^ { 4 }$ , then with probability at least $1 - \delta$ , there exists a $\pmb { \theta } ^ { * } \in \mathbb { R } ^ { p }$ such that 
$$
\begin{array}{l} h (\mathbf {x} ^ {i}) = \langle \mathbf {g} (\mathbf {x} ^ {i}; \boldsymbol {\theta} _ {0}), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \rangle , \\ \sqrt {m} \| \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \| _ {2} \leq \sqrt {2 \mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h}}, \tag {5.1} \\ \end{array}
$$
for all $i \in [ T K ]$ . 
Lemma 5.1 suggests that with high probability, the reward function restricted to $\{ { \bf x } ^ { i } \} _ { i = 1 } ^ { T K }$ can be regarded as a linear 
function of $\mathbf { g } ( \mathbf { x } ^ { i } ; \mathbf { \boldsymbol { \theta } } _ { 0 } )$ parameterized by $\pmb { \theta } ^ { * } - \pmb { \theta } _ { 0 }$ , where $\pmb { \theta } ^ { * }$ lies in a ball centered at $\pmb { \theta } _ { 0 }$ . Note that here $\pmb { \theta } ^ { * }$ is not a ground truth parameter for the reward function. Instead, it is introduced only for the sake of analysis. Equipped with Lemma 5.1, we can utilize existing results on linear bandits (Abbasi-Yadkori et al., 2011) to show that with high probability, $\pmb { \theta } ^ { * }$ lies in the sequence of confidence sets. 
Lemma 5.2. There exist positive constants $\bar { C } _ { 1 }$ and $\bar { C } _ { 2 }$ such that for any $\delta \in ( 0 , 1 )$ , if $\eta \leq \bar { C } _ { 1 } ( T m L + m \lambda ) ^ { - 1 }$ and 
$$
\begin{array}{l} m \geq \bar {C} _ {2} \max  \left\{T ^ {7} \lambda^ {- 7} L ^ {2 1} (\log m) ^ {3}, \right. \\ \lambda^ {- 1 / 2} L ^ {- 3 / 2} \left(\log \left(T K L ^ {2} / \delta\right)\right) ^ {3 / 2} \}, \\ \end{array}
$$
then with probability at least $1 - \delta$ , we have $\lVert { \pmb \theta } _ { t } - { \pmb \theta } _ { 0 } \rVert _ { 2 } \leq$ $2 \sqrt { t / ( m \lambda ) }$ and $\| \pmb { \theta } ^ { * } - \pmb { \theta } _ { t } \| _ { \mathbf { Z } _ { t } } \leq \gamma _ { t } / \sqrt { m }$ for all $t \in [ T ]$ where $\gamma _ { t }$ is defined in Algorithm 1. 
Lemma 5.3. Let $a _ { t } ^ { * } = \operatorname { a r g m a x } _ { a \in [ K ] } h ( \mathbf { x } _ { t , a } )$ . There exists a positive constant $\bar { C }$ such that for any $\delta \in ( 0 , 1 )$ , if $\eta$ and $m$ satisfy the same conditions as in Lemma 5.2, then with probability at least $1 - \delta$ , we have 
$$
\begin{array}{l} h \left(\mathbf {x} _ {t, a _ {t} ^ {*}}\right) - h \left(\mathbf {x} _ {t, a _ {t}}\right) \\ \leq 2 \gamma_ {t - 1} \min  \left\{\| \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}}, 1 \right\} \\ + \bar {C} \left(S m ^ {- 1 / 6} \sqrt {\log m} T ^ {7 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2} \right. \\ \left. + m ^ {- 1 / 6} \sqrt {\log m} T ^ {5 / 3} \lambda^ {- 2 / 3} L ^ {3}\right). \\ \end{array}
$$
Lemma 5.3 gives an upper bound for $h \big ( \mathbf { x } _ { t , a _ { t } ^ { * } } \big ) - h \big ( \mathbf { x } _ { t , a _ { t } } \big )$ which can be used to bound the regret $R _ { T }$ . It is worth noting that $\gamma _ { t }$ has a term $\log \operatorname* { d e t } \mathbf { Z } _ { t }$ . A trivial upper bound of $\log \operatorname* { d e t } \mathbf { Z } _ { t }$ would result in a quadratic dependence on the network width $m$ , since the dimension of $\mathbf { Z } _ { t }$ is $p =$ $m d + m ^ { 2 } ( L - 2 ) + m$ . Instead, we use the next lemma to establish an $m$ -independent upper bound. The dependence on $\hat { d }$ is similar to Valko et al. (2013, Lemma 4), but the proof is different as our notion of effective dimension is different. 
Lemma $\{ \bar { C } _ { i } \} _ { i = 1 } ^ { 3 }$ such that for any 5.4. There exist positive constants $\delta \in ( 0 , 1 )$ , if $\begin{array} { r l } { m } & { { } \geq } \end{array}$ $\bar { C } _ { 1 } \operatorname * { m a x } \left\{ T ^ { 7 } \lambda ^ { - 7 } L ^ { 2 1 } ( \log m ) ^ { 3 } , T ^ { 6 } K ^ { 6 } L ^ { 6 } ( \log ( T K L ^ { 2 } / \delta ) ) ^ { 3 / 2 } \right\}$ and $\eta \leq \bar { C } _ { 2 } ( T m L + m \lambda ) ^ { - 1 }$ , then with probability at least $1 - \delta$ , we have 
$$
\begin{array}{l} \sqrt {\sum_ {t = 1} ^ {T} \gamma_ {t - 1} ^ {2} \min  \left\{\| \mathbf {g} \left(\mathbf {x} _ {t , a _ {t}} ; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} ^ {2} , 1 \right\}} \\ \leq \sqrt {\tilde {d} \log (1 + T K / \lambda) + \Gamma_ {1}} \\ \left[ \Gamma_ {2} \left(\nu \sqrt {\tilde {d} \log (1 + T K / \lambda) + \Gamma_ {1} - 2 \log \delta} + \sqrt {\lambda} S\right) \right. \\ + (\lambda + \bar {C} _ {3} t L) \left[ (1 - \eta m \lambda) ^ {J / 2} \sqrt {T / \lambda} + \Gamma_ {3} (1 + \sqrt {T / \lambda}) \right], \\ \end{array}
$$
Neural Contextual Bandits with UCB-based Exploration 
where 
$$
\Gamma_ {1} = 1 + \bar {C} _ {3} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {5 / 3} \lambda^ {- 1 / 6},
$$
$$
\Gamma_ {2} = \sqrt {1 + \bar {C} _ {3} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {7 / 6} \lambda^ {- 7 / 6}},
$$
$$
\Gamma_ {3} = m ^ {- 1 / 6} \sqrt {\log m} L ^ {7 / 2} T ^ {5 / 3} \lambda^ {- 5 / 3}.
$$
We are now ready to prove the main result. 
Proof of Theorem 4.5. Lemma 5.3 implies that the total regret $R _ { T }$ can be bounded as follows with a constant $C _ { 1 } > 0$ : 
$$
\begin{array}{l} R _ {T} = \sum_ {t = 1} ^ {T} \left[ h \left(\mathbf {x} _ {t, a _ {t} ^ {*}}\right) - h \left(\mathbf {x} _ {t, a _ {t}}\right) \right] \\ \leq 2 \sum_ {t = 1} ^ {T} \gamma_ {t - 1} \min  \left\{\| \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}}, 1 \right\} \\ + C _ {1} \left(S m ^ {- 1 / 6} \sqrt {\log m} T ^ {1 3 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2} \right. \\ \left. + m ^ {- 1 / 6} \sqrt {\log m} T ^ {8 / 3} \lambda^ {- 2 / 3} L ^ {3}\right). \\ \end{array}
$$
It can be further bounded as follows: 
$$
\begin{array}{l} R _ {T} \leq 2 \sqrt {T \sum_ {t = 1} ^ {T} \gamma_ {t - 1} ^ {2} \min  \left\{\| \mathbf {g} \left(\mathbf {x} _ {t , a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} ^ {2} , 1 \right\}} \\ + C _ {1} \left(S m ^ {- 1 / 6} \sqrt {\log m} T ^ {1 3 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2} \right. \\ + m ^ {- 1 / 6} \sqrt {\log m} T ^ {8 / 3} \lambda^ {- 2 / 3} L ^ {3}) \\ \leq 2 \sqrt {T} \cdot \sqrt {\tilde {d} \log (1 + T K / \lambda) + \Gamma_ {1}} \\ \left[ \Gamma_ {2} \left(\nu \sqrt {\tilde {d} \log (1 + T K / \lambda) + \Gamma_ {1} - 2 \log \delta} + \sqrt {\lambda} S\right) \right. \\ + (\lambda + C _ {2} T L) \left[ (1 - \eta m \lambda) ^ {J / 2} \sqrt {T / \lambda} \right. \\ \left. + \Gamma_ {3} (1 + \sqrt {T / \lambda}) \right] \Bigg ] \\ + C _ {1} \left(S m ^ {- 1 / 6} \sqrt {\log m} T ^ {1 3 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2} \right. \\ + m ^ {- 1 / 6} \sqrt {\log m} T ^ {8 / 3} \lambda^ {- 2 / 3} L ^ {3}) \\ \leq 3 \sqrt {T} \sqrt {\tilde {d} \log (1 + T K / \lambda) + 2} \\ \cdot \left[ \nu \sqrt {\tilde {d} \log (1 + T K / \lambda) + 2 - 2 \log \delta} \right. \\ + (\lambda + C _ {3} T L) (1 - \eta m \lambda) ^ {J / 2} \sqrt {T / \lambda} \\ \left. + 2 \sqrt {\lambda} S \right] + 1, \\ \end{array}
$$
where $C _ { 1 } , C _ { 2 } , C _ { 3 }$ are positive constants, the first inequality is due to Cauchy-Schwarz inequality, the second inequality due to Lemma 5.4, and the third inequality holds for sufficiently large $m$ . This completes our proof. □ 
# 6. Related Work
Contextual Bandits There is a line of extensive work on linear bandits (e.g., Abe et al., 2003; Auer, 2002; Abe et al., 2003; Dani et al., 2008; Rusmevichientong & Tsitsiklis, 2010; Li et al., 2010; Chu et al., 2011; Abbasi-Yadkori et al., 2011). Many of these algorithms are based on the idea of upper confidence bounds, and are shown to achieve near-optimal regret bounds. Our algorithm is also based on UCB exploration, and the regret bound reduces to that of Abbasi-Yadkori et al. (2011) in the linear case. 
To deal with nonlinearity, a few authors have considered generalized linear bandits (Filippi et al., 2010; Li et al., 2017; Jun et al., 2017), where the reward function is a composition of a linear function and a (nonlinear) link function. Such models are special cases of what we study in this work. 
More general nonlinear bandits without making strong modeling assumptions have also be considered. One line of work is the family of expert learning algorithms (Auer et al., 2002; Beygelzimer et al., 2011) that typically have a time complexity linear in the number of experts (which in many cases can be exponential in the number of parameters). 
A second approach is to reduce a bandit problem to supervised learning, such as the epoch-greedy algorithm (Langford & Zhang, 2008) that has a non-optimal $O ( T ^ { 2 / 3 } )$ regret. Later, Agarwal et al. (2014) develop an algorithm that enjoys a near-optimal regret, but relies on an oracle, whose implementation still requires proper modeling assumptions. 
A third approach uses nonparametric modeling, such as perceptrons (Kakade et al., 2008), random forests (Feraud et al.´ , 2016), Gaussian processes and kernels (Kleinberg et al., 2008; Srinivas et al., 2010; Krause & Ong, 2011; Bubeck et al., 2011). The most relevant is by Valko et al. (2013), who assumed that the reward function lies in an RKHS with bounded RKHS norm and developed a UCB-based algorithm. They also proved an $\widetilde { O } ( \sqrt { d T } )$ regret, where $\widetilde { d }$ is a form of effective dimension similar to ours. Compared with these interesting works, our neural network-based algorithm avoids the need to carefully choose a good kernel or metric, and can be computationally more efficient in large-scale problems. Recently, Foster & Rakhlin (2020) proposed contextual bandit algorithms with regression oracles which achieve a dimension-independent $O ( T ^ { 3 / 4 } )$ regret. Compared with Foster & Rakhlin (2020), NeuralUCB achieves a dimension-dependent $\widetilde { O } ( \widetilde { d } \sqrt { T } )$ regret with a better dependence on the time horizon. 
Neural Networks Substantial progress has been made to understand the expressive power of DNNs, in connection to the network depth (Telgarsky, 2015; 2016; Liang & Srikant, 2016; Yarotsky, 2017; 2018; Hanin, 2017), as well as network width (Lu et al., 2017; Hanin & Sellke, 2017). 
Neural Contextual Bandits with UCB-based Exploration 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/4b82e7c2-1d69-4b88-860d-f5a4d7f97956/4fe4581a021e92e87e27e84e9cf355b931bbc69249bd0882d0f7741514f7f2d4.jpg)


(a) $h _ { 1 } ( \mathbf { x } ) = 1 0 ( \mathbf { x } ^ { \top } \mathbf { a } ) ^ { 2 }$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/4b82e7c2-1d69-4b88-860d-f5a4d7f97956/17fba2179ba3dbd97626966116bf4df3c342ec3d527da7622f3996d6d508a832.jpg)


$( \mathbf { b } ) \ h _ { 2 } ( \mathbf { x } ) = \mathbf { x } ^ { \top } \mathbf { A } ^ { \top } \mathbf { A } \mathbf { x }$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/4b82e7c2-1d69-4b88-860d-f5a4d7f97956/5dd5646ab25fae2cfbc3f3b0a551de4d1eba8b1df47b3ef45bfcd17fad99c2f8.jpg)


$( \mathbf { c } ) \ h _ { 3 } ( \mathbf { x } ) = \cos ( 3 \mathbf { x } ^ { \top } \mathbf { a } )$


Figure 1. Comparison of NeuralUCB and baseline algorithms on synthetic datasets.

The present paper on neural contextual bandit algorithms is inspired by these theoretical justifications and empirical evidence in the literature. 
Our regret analysis for NeuralUCB makes use of recent advances in optimizing a DNN. A series of works show that (stochastic) gradient descent can find global minima of the training loss (Li & Liang, 2018; Du et al., 2019b; Allen-Zhu et al., 2019; Du et al., 2019a; Zou et al., 2019; Zou & Gu, 2019). For the generalization of DNNs, a number of authors (Daniely, 2017; Cao & Gu, 2019; 2020; Arora et al., 2019; Chen et al., 2019) show that by using (stochastic) gradient descent, the parameters of a DNN are located in a particular regime and the generalization bound of DNNs can be characterized by the best function in the corresponding neural tangent kernel space (Jacot et al., 2018). 
# 7. Experiments
In this section, we evaluate NeuralUCB empirically and compare it with seven representative baselines: (1) LinUCB, which is also based on UCB but adopts a linear representation; (2) GLMUCB (Filippi et al., 2010), which applies a nonlinear link function over a linear function; (3) KernelUCB (Valko et al., 2013), a kernelised UCB algorithm which makes use of a predefined kernel function; (4) BootstrappedNN (Efron, 1982; Riquelme et al., 2018), which simultaneously trains a set of neural networks using bootstrapped samples and at every round chooses an action based on the prediction of a randomly picked model; (5) Neural $\epsilon$ -Greedy, which replaces the UCB-based exploration in Algorithm 1 by $\epsilon$ -greedy; (6) Neura $\mathrm { U C B _ { 0 } }$ , as described in Section 3; and (7) Neural $\epsilon$ -Greedy0, same as Neura $\mathsf { I U C B } _ { 0 }$ but with $\epsilon$ -greedy exploration. We use the cumulative regret as the performance metric. 
# 7.1. Synthetic Datasets
In the first set of experiments, we use contextual bandits with context dimension $d = 2 0$ and $K = 4$ actions. The number of rounds $T = 1 0 \ 0 0 0$ . The contextual vectors 
$\left\{ \mathbf { x } _ { 1 , 1 } , \dotsc , \mathbf { x } _ { T , K } \right\}$ are chosen uniformly at random from the unit ball. The reward function $h$ is one of the following: 
$$
h _ {1} (\mathbf {x}) = 1 0 \left(\mathbf {x} ^ {\top} \mathbf {a}\right) ^ {2},
$$
$$
h _ {2} (\mathbf {x}) = \mathbf {x} ^ {\top} \mathbf {A} ^ {\top} \mathbf {A} \mathbf {x},
$$
$$
h _ {3} (\mathbf {x}) = \cos (3 \mathbf {x} ^ {\top} \mathbf {a}),
$$
where each entry of $\mathbf { A } \in \mathbb { R } ^ { d \times d }$ is randomly generated from $N ( 0 , 1 )$ , a is randomly generated from uniform distribution over unit ball. For each $h _ { i } ( \cdot )$ , the reward is generated by $r _ { t , a } = h _ { i } ( \mathbf { x } _ { t , a } ) + \xi _ { t }$ , where $\xi _ { t } \sim N ( 0 , 1 )$ . 
Following Li et al. (2010), we implement LinUCB using a constant $\alpha$ (for the variance term in the UCB). We do a grid search for $\alpha$ over $\{ 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ . For GLMUCB, we use the sigmoid function as the link function and adapt the online Newton step method to accelerate the computation (Zhang et al., 2016; Jun et al., 2017). We do grid searches over $\{ 0 . 1 , 1 , 1 0 \}$ for regularization parameter, $\{ 1 , 1 0 , 1 0 0 \}$ for step size, $\{ 0 . 0 1 , 0 . 1 , 1 \}$ for exploration parameter. For KernelUCB, we use the radial basis function (RBF) kernel with parameter $\sigma$ , and set the regularization parameter to 1. Grid searches over $\{ 0 . 1 , 1 , 1 0 \}$ for $\sigma$ and $\{ 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ for the exploration parameter are done. To accelerate the calculation, we stop adding contexts to KernelUCB after 1000 rounds, following the same setting for Gaussian Process in Riquelme et al. (2018). For all five neural algorithms, we choose a two-layer neural network $f ( \mathbf { x } ; \pmb { \theta } ) =$ $\sqrt { m } \mathbf { W } _ { 2 } \sigma ( \mathbf { W } _ { 1 } \mathbf { x } )$ with network width $m = 2 0$ , where $\pmb \theta =$ $[ \mathrm { v e c } ( \mathbf { W } _ { 1 } ) ^ { \top }$ , $\mathrm { v e c } ( \mathbf { W } _ { 2 } ) ^ { \top } ] \in \mathbb { R } ^ { p }$ and $p = m d + m = 4 2 0$ . 1 Moreover, we set $\gamma _ { t } ~ = ~ \gamma$ in NeuralUCB, and do a grid search over $\{ 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ . For NeuralUCB0, we do grid searches for $\nu$ over $\{ 0 . 1 , 1 , 1 0 \}$ , for $\lambda$ over $\{ 0 . 1 , 1 , 1 0 \}$ , for $\delta$ over $\{ 0 . 0 1 , 0 . 1 , 1 \}$ , for $S$ over $\{ 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ . For Neural $\epsilon$ -Greedy and Neural $\epsilon$ -Greedy0, we do a grid search for $\epsilon$ over $\left. 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 0 . 2 \right.$ . For BootstrappedNN, we follow Riquelme et al. (2018) to set the number of models to be 10 and the transition probability to be 0.8. To accelerate 
1Note that the bound on the required network width $m$ is likely not tight. Therefore, in experiments we choose $m$ to be relatively large, but not as large as theory suggests. 
Neural Contextual Bandits with UCB-based Exploration 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/4b82e7c2-1d69-4b88-860d-f5a4d7f97956/28cc3abcb37b985951858982bcc09bd10162fcc5f0fa01782dfa315dea12d692.jpg)


(a) covertype

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/4b82e7c2-1d69-4b88-860d-f5a4d7f97956/d78d6cf81d0d458bed5de3823efcfef99f303459dc5612c39c0c35cf3a9c5c32.jpg)


(b) magic

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/4b82e7c2-1d69-4b88-860d-f5a4d7f97956/16dacd49dd39cf541d64d86418fa12ba9fd73ecd2c98982d22e51a17b24d6ab7.jpg)


(c) statlog

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/4b82e7c2-1d69-4b88-860d-f5a4d7f97956/e183880a509d4f0d68f3d42cec94f7d711461204b6ce2f93c1663f4087b3dae0.jpg)


(d) mnist


Figure 2. Comparison of NeuralUCB and baseline algorithms on real-world datasets.


Table 1. Dataset statistics

<table><tr><td>DATASET</td><td>COVER-TYPE</td><td>MAGIC</td><td>STATLOG</td><td>MNIST</td></tr><tr><td>FEATURE DIMENSION</td><td>54</td><td>10</td><td>8</td><td>784</td></tr><tr><td>NUMBER OF CLASSES</td><td>7</td><td>2</td><td>7</td><td>10</td></tr><tr><td>NUMBER OF INSTANCES</td><td>581012</td><td>19020</td><td>58000</td><td>60000</td></tr></table>
the training process, for BootstrappedNN, NeuralUCB and Neural $\epsilon$ -Greedy, we update the parameter $\theta _ { t }$ by TrainNN every 50 rounds. We use stochastic gradient descent with batch size 50, $J = t$ at round $t$ , and do a grid search for step size $\eta$ over $\{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 \}$ . For all grid-searched parameters, we choose the best of them for the comparison. All experiments are repeated 10 times, and the averaged results reported for comparison. 
# 7.2. Real-world Datasets
We evaluate our algorithms on real-world datasets from the UCI Machine Learning Repository (Dua & Graff, 2017): covertype, magic, and statlog. We also evaluate our algorithms on mnist dataset (LeCun et al., 1998). These are all $K$ -class classification datasets (Table 1), and are converted into $K$ -armed contextual bandits (Beygelzimer & Langford, 2009). The number of rounds is set as $T = 1 5 0 0 0$ . Following Riquelme et al. (2018), we create 
contextual bandit problems based on the prediction accuracy. In detail, to transform a classification problem with $k$ -classes into a bandit problem, we adapts the disjoint model (Li et al., 2010) which transforms each contextual vector $\mathbf { x } \in \mathbb { R } ^ { d }$ into $k$ vectors $\mathbf { x } ^ { ( 1 ) } = ( \mathbf { x } , \mathbf { 0 } , \ldots , \mathbf { 0 } ) , \ldots , \mathbf { x } ^ { ( k ) } =$ $( \mathbf { 0 } , \dots , \mathbf { 0 } ,  { \mathbf { x } } ) \in \mathbb { R } ^ { d k }$ . The agent received regret 0 if he classifies the context correctly, and 1 otherwise. For all the algorithms, We reshuffle the order of contexts and repeat the experiment for 10 runs. Averaged results are reported for comparison. 
For LinUCB, GLMUCB and KernelUCB, we tune their parameters as Section 7.1 suggests. For BootstrappedNN, NeuralUCB, Neura $\mathrm { U C B _ { 0 } }$ , Neural $\epsilon$ -Greedy and Neural - Greedy0, we choose a two-layer neural network with width $m = 1 0 0$ . For NeuralUCB and NeuralUCB0, since it is computationally expensive to store and compute a whole matrix $\mathbf { Z } _ { t }$ , we use a diagonal matrix which consists of the diagonal elements of $\mathbf { Z } _ { t }$ to approximate $\mathbf { Z } _ { t }$ . To accelerate the training process, for BootstrappedNN, NeuralUCB and Neural $\epsilon$ -Greedy, we update the parameter $\theta _ { t }$ by TrainNN every 100 rounds starting from round 2000. We do grid searches for $\lambda$ over $\{ 1 0 ^ { - i } \} , i = 1 , 2 , 3 , 4$ , for $\eta$ over $\{ 2 \times$ $1 0 ^ { - i } , 5 \times 1 0 ^ { - i } \} , i = 1 , 2 , 3 , 4$ . We set $J = 1 0 0 0$ and use stochastic gradient descent with batch size 500 to train the networks. For the rest of parameters, we tune them as those in Section 7.1 and choose the best of them for comparison. 
Neural Contextual Bandits with UCB-based Exploration 
# 7.3. Results
Figures 1 and 2 show the cumulative regret of all algorithms. First, due to the nonlinearity of reward functions $h$ , Lin-UCB fails to learn them for nearly all tasks. GLMUCB is only able to learn the true reward functions for certain tasks due to its simple link function. In contrast, thanks to the neural network representation and efficient exploration, NeuralUCB achieves a substantially lower regret. The performance of Neural $\epsilon$ -Greedy is between the two. This suggests that while Neural $\epsilon$ -Greedy can capture the nonlinearity of the underlying reward function, $\epsilon$ -Greedy based exploration is not as effective as UCB based exploration. This confirms the effectiveness of NeuralUCB for contextual bandit problems with nonlinear reward functions. Second, it is worth noting that NeuralUCB and Neural $\epsilon$ -Greedy outperform Neura $\mathrm { U C B _ { 0 } }$ and Neural $\epsilon$ -Greedy0. This suggests that using deep neural networks to predict the reward function is better than using a fixed feature mapping associated with the Neural Tangent Kernel, which mirrors similar findings in supervised learning (Allen-Zhu & Li, 2019). Furthermore, we can see that KernelUCB is not as good as NeuralUCB, which suggests the limitation of simple kernels like RBF compared to flexible neural networks. What’s more, BootstrappedNN can be competitive, approaching the performance of NeuralUCB in some datasets. However, it requires to maintain and train multiple neural networks, so is computationally more expensive than our approach, especially in large-scale problems. 
# 8. Conclusion
In this paper, we proposed NeuralUCB, a new algorithm for stochastic contextual bandits based on neural networks and upper confidence bounds. Building on recent advances in optimization and generalization of deep neural networks, we showed that for an arbitrary bounded reward function, our algorithm achieves an $\widetilde { O } ( \widetilde { d } \sqrt { T } )$ regret bound. Promising empirical results on both synthetic and real-world data corroborated our theoretical findings, and suggested the potential of the algorithm in practice. 
We conclude the paper with a suggested direction for future research. Given the focus on UCB exploration in this work, a natural open question is provably efficient exploration based on randomized strategies, when DNNs are used. These methods are effective in practice, but existing regret analyses are mostly for shallow (i.e., linear or generalized linear) models (Chapelle & Li, 2011; Agrawal & Goyal, 2013; Russo et al., 2018; Kveton et al., 2020). Extending them to DNNs will be interesting. Meanwhile, our current analysis of NeuralUCB is based on the NTK theory. While NTK facilitates the analysis, it has its own limitations, and we will leave the analysis of NeuralUCB beyond NTK as future work. 
# Acknowledgement
We would like to thank the anonymous reviewers for their helpful comments. This research was sponsored in part by the National Science Foundation IIS-1904183 and IIS-1906169. The views and conclusions contained in this paper are those of the authors and should not be interpreted as representing any funding agencies. 
# References


Abbasi-Yadkori, Y., Pal, D., and Szepesv ´ ari, C. Improved ´ algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems, pp. 2312–2320, 2011. 




Abe, N., Biermann, A. W., and Long, P. M. Reinforcement learning with immediate rewards and linear hypotheses. Algorithmica, 37(4):263–293, 2003. 




Agarwal, A., Hsu, D., Kale, S., Langford, J., Li, L., and Schapire, R. E. Taming the monster: A fast and simple algorithm for contextual bandits. In Proceedings of the 31st International Conference on Machine Learning (ICML), pp. 1638–1646, 2014. 




Agrawal, S. and Goyal, N. Thompson sampling for contextual bandits with linear payoffs. In International Conference on Machine Learning, pp. 127–135, 2013. 




Allen-Zhu, Z. and Li, Y. What can ResNet learn efficiently, going beyond kernels? In Advances in Neural Information Processing Systems, 2019. 




Allen-Zhu, Z., Li, Y., and Song, Z. A convergence theory for deep learning via over-parameterization. In International Conference on Machine Learning, pp. 242–252, 2019. 




Allesiardo, R., Feraud, R., and Bouneffouf, D. A neural ´ networks committee for the contextual bandit problem. In International Conference on Neural Information Processing, pp. 374–381. Springer, 2014. 




Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R., and Wang, R. On exact computation with an infinitely wide neural net. In Advances in Neural Information Processing Systems, 2019. 




Auer, P. Using confidence bounds for exploitationexploration trade-offs. Journal of Machine Learning Research, 3(Nov):397–422, 2002. 




Auer, P., Cesa-Bianchi, N., Freund, Y., and Schapire, R. E. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002. 




Azizzadenesheli, K., Brunskill, E., and Anandkumar, A. Efficient exploration through Bayesian deep Q-networks. 


Neural Contextual Bandits with UCB-based Exploration 


In 2018 Information Theory and Applications Workshop (ITA), pp. 1–9. IEEE, 2018. 




Beygelzimer, A. and Langford, J. The offset tree for learning with partial labels. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 129–138, 2009. 




Beygelzimer, A., Langford, J., Li, L., Reyzin, L., and Schapire, R. E. Contextual bandit algorithms with supervised learning guarantees. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 19–26, 2011. 




Bubeck, S. and Cesa-Bianchi, N. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends in Machine Learning, 5(1):1–122, 2012. 




Bubeck, S., Munos, R., Stoltz, G., and Szepesvari, C. X-´ armed bandits. Journal of Machine Learning Research, 12(May):1655–1695, 2011. 




Cao, Y. and Gu, Q. Generalization bounds of stochastic gradient descent for wide and deep neural networks. In Advances in Neural Information Processing Systems, 2019. 




Cao, Y. and Gu, Q. Generalization error bounds of gradient descent for learning over-parameterized deep relu networks. In the Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020. 




Chapelle, O. and Li, L. An empirical evaluation of thompson sampling. In Advances in neural information processing systems, pp. 2249–2257, 2011. 




Chen, Z., Cao, Y., Zou, D., and Gu, Q. How much overparameterization is sufficient to learn deep relu networks? arXiv preprint arXiv:1911.12360, 2019. 




Chu, W., Li, L., Reyzin, L., and Schapire, R. Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 208–214, 2011. 




Dani, V., Hayes, T. P., and Kakade, S. M. Stochastic linear optimization under bandit feedback. 2008. 




Daniely, A. SGD learns the conjugate kernel class of the network. In Advances in Neural Information Processing Systems, pp. 2422–2430, 2017. 




Du, S., Lee, J., Li, H., Wang, L., and Zhai, X. Gradient descent finds global minima of deep neural networks. In International Conference on Machine Learning, pp. 1675–1685, 2019a. 




Du, S. S., Zhai, X., Poczos, B., and Singh, A. Gradient descent provably optimizes over-parameterized neural networks. In International Conference on Learning Representations, 2019b. URL https://openreview. net/forum?id=S1eK3i09YQ. 




Dua, D. and Graff, C. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml. 




Efron, B. The jackknife, the bootstrap, and other resampling plans, volume 38. Siam, 1982. 




Feraud, R., Allesiardo, R., Urvoy, T., and Cl ´ erot, F. Random ´ forest for the contextual bandit problem. In Artificial Intelligence and Statistics, pp. 93–101, 2016. 




Filippi, S., Cappe, O., Garivier, A., and Szepesvari, C. Para- ´ metric bandits: The generalized linear case. In Advances in Neural Information Processing Systems, pp. 586–594, 2010. 




Foster, D. J. and Rakhlin, A. Beyond ucb: Optimal and efficient contextual bandits with regression oracles. arXiv preprint arXiv:2002.04926, 2020. 




Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning. MIT Press, 2016. http://www. deeplearningbook.org. 




Hanin, B. Universal function approximation by deep neural nets with bounded width and ReLU activations. arXiv preprint arXiv:1708.02691, 2017. 




Hanin, B. and Sellke, M. Approximating continuous functions by ReLU nets of minimal width. arXiv preprint arXiv:1710.11278, 2017. 




Jacot, A., Gabriel, F., and Hongler, C. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in neural information processing systems, pp. 8571–8580, 2018. 




Jun, K.-S., Bhargava, A., Nowak, R. D., and Willett, R. Scalable generalized linear bandits: Online computation and hashing. In Advances in Neural Information Processing Systems 30 (NIPS), pp. 99–109, 2017. 




Kakade, S. M., Shalev-Shwartz, S., and Tewari, A. Efficient bandit algorithms for online multiclass prediction. In Proceedings of the 25th international conference on Machine learning, pp. 440–447, 2008. 




Kleinberg, R., Slivkins, A., and Upfal, E. Multi-armed bandits in metric spaces. In Proceedings of the fortieth annual ACM symposium on Theory of computing, pp. 681–690. ACM, 2008. 


Neural Contextual Bandits with UCB-based Exploration 


Krause, A. and Ong, C. S. Contextual Gaussian process bandit optimization. In Advances in neural information processing systems, pp. 2447–2455, 2011. 




Kveton, B., Zaheer, M., Szepesvri, C., Li, L., Ghavamzadeh, M., and Boutilier, C. Randomized exploration in generalized linear bandits. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, 2020. 




Langford, J. and Zhang, T. The epoch-greedy algorithm for contextual multi-armed bandits. In Advances in Neural Information Processing Systems 20 (NIPS), pp. 1096– 1103, 2008. 




Lattimore, T. and Szepesvari, C. ´ Bandit Algorithms. Cambridge University Press, 2019. In press. 




LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. 




Li, L., Chu, W., Langford, J., and Schapire, R. E. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pp. 661–670. ACM, 2010. 




Li, L., Lu, Y., and Zhou, D. Provably optimal algorithms for generalized linear contextual bandits. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 2071–2080. JMLR. org, 2017. 




Li, Y. and Liang, Y. Learning overparameterized neural networks via stochastic gradient descent on structured data. In Advances in Neural Information Processing Systems, pp. 8157–8166, 2018. 




Liang, S. and Srikant, R. Why deep neural networks for function approximation? arXiv preprint arXiv:1610.04161, 2016. 




Lipton, Z., Li, X., Gao, J., Li, L., Ahmed, F., and Deng, L. BBQ-networks: Efficient exploration in deep reinforcement learning for task-oriented dialogue systems. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018. 




Lu, Z., Pu, H., Wang, F., Hu, Z., and Wang, L. The expressive power of neural networks: A view from the width. In Advances in neural information processing systems, pp. 6231–6239, 2017. 




Riquelme, C., Tucker, G., and Snoek, J. Deep Bayesian bandits showdown. In International Conference on Learning Representations, 2018. 




Rusmevichientong, P. and Tsitsiklis, J. N. Linearly parameterized bandits. Mathematics of Operations Research, 35 (2):395–411, 2010. 




Russo, D., Roy, B. V., Kazerouni, A., Osband, I., and Wen, Z. A tutorial on Thompson sampling. Foundations and Trends in Machine Learning, 11(1):1–96, 2018. 




Srinivas, N., Krause, A., Kakade, S., and Seeger, M. Gaussian process optimization in the bandit setting: no regret and experimental design. In Proceedings of the 27th International Conference on International Conference on Machine Learning, pp. 1015–1022. Omnipress, 2010. 




Telgarsky, M. Representation benefits of deep feedforward networks. arXiv preprint arXiv:1509.08101, 2015. 




Telgarsky, M. Benefits of depth in neural networks. arXiv preprint arXiv:1602.04485, 2016. 




Valko, M., Korda, N., Munos, R., Flaounas, I., and Cristianini, N. Finite-time analysis of kernelised contextual bandits. arXiv preprint arXiv:1309.6869, 2013. 




Yang, L. F. and Wang, M. Reinforcement leaning in feature space: Matrix bandit, kernels, and regret bound. arXiv preprint arXiv:1905.10389, 2019. 




Yarotsky, D. Error bounds for approximations with deep ReLU networks. Neural Networks, 94:103–114, 2017. 




Yarotsky, D. Optimal approximation of continuous functions by very deep ReLU networks. arXiv preprint arXiv:1802.03620, 2018. 




Zahavy, T. and Mannor, S. Deep neural linear bandits: Overcoming catastrophic forgetting through likelihood matching. arXiv preprint arXiv:1901.08612, 2019. 




Zhang, L., Yang, T., Jin, R., Xiao, Y., and Zhou, Z.-H. Online stochastic linear optimization under one-bit feedback. In International Conference on Machine Learning, pp. 392–401, 2016. 




Zou, D. and Gu, Q. An improved analysis of training overparameterized deep neural networks. In Advances in Neural Information Processing Systems, 2019. 




Zou, D., Cao, Y., Zhou, D., and Gu, Q. Stochastic gradient descent optimizes over-parameterized deep ReLU networks. Machine Learning, 2019. 


Neural Contextual Bandits with UCB-based Exploration 
# A. Proof of Additional Results in Section 4
# A.1. Verification of Remark 4.4
Suppose there exists a mapping $\psi : \mathbb { R } ^ { d } \to \mathbb { R } ^ { \widehat { d } }$ satisfying $\| \psi ( \mathbf { x } ) \| _ { 2 } ~ \leq ~ 1$ which maps any context $\textbf { x } \in \ \mathbb { R } ^ { d }$ to the Hilbert space $\mathcal { H }$ associated with the Gram matrix $\mathbf { H } \in \mathbb { R } ^ { T K \times T K }$ over contexts $\{ \mathbf { x } ^ { i } \} _ { i = 1 } ^ { T K }$ . Then $\mathbf { H } = \Psi ^ { \top } \Psi$ , where $\Psi = [ \psi ( \mathbf { x } ^ { 1 } ) , \dots , \psi ( \mathbf { x } ^ { T K } ) ] \in \mathbb { R } ^ { { \widehat { d } } \times T K }$ . Thus, we can bound the effective dimension $\hat { d }$ as follows 
$$
\widetilde {d} = \frac {\log \det  [ \mathbf {I} + \mathbf {H} / \lambda ]}{\log (1 + T K / \lambda)} = \frac {\log \det  [ \mathbf {I} + \boldsymbol {\Psi} \boldsymbol {\Psi} ^ {\top} / \lambda ]}{\log (1 + T K / \lambda)} \leq \widehat {d} \cdot \frac {\log \| \mathbf {I} + \boldsymbol {\Psi} \boldsymbol {\Psi} ^ {\top} / \lambda \| _ {2}}{\log (1 + T K / \lambda)}.
$$
where the second equality holds due to the fact that $\operatorname* { d e t } ( \mathbf { I } + \mathbf { A } ^ { \top } \mathbf { A } / \lambda ) = \operatorname* { d e t } ( \mathbf { I } + \mathbf { A } \mathbf { A } ^ { \top } / \lambda )$ holds for any matrix A, and the inequality holds since de $\mathbf { t A } \leq \| \mathbf { A } \| _ { 2 } ^ { \widehat { d } }$ for any $\mathbf { A } \in \mathbb { R } ^ { \widehat { d } \times \widehat { d } } .$ . Clearly, $\widetilde d \le \widehat d$ as long as $\left. \mathbf { I } + \boldsymbol { \Psi } \boldsymbol { \Psi } ^ { \top } / \lambda \right. _ { 2 } \leq 1 + T K / \lambda$ . Indeed, 
$$
\left\| \mathbf {I} + \boldsymbol {\Psi} \boldsymbol {\Psi} ^ {\top} / \lambda \right\| _ {2} \leq 1 + \left\| \boldsymbol {\Psi} \boldsymbol {\Psi} ^ {\top} \right\| _ {2} / \lambda \leq 1 + \sum_ {i = 1} ^ {T K} \left\| \boldsymbol {\psi} (\mathbf {x} ^ {i}) \boldsymbol {\psi} (\mathbf {x} ^ {i}) ^ {\top} \right\| _ {2} / \lambda \leq 1 + T K / \lambda ,
$$
where the first inequality is due to triangle inequality and the fact $\lambda \geq 1$ , the second inequality holds due to the definition of $\Psi$ and triangle inequality, and the last inequality is by $\| \psi ( \mathbf { x } ^ { i } ) \| _ { 2 } \leq 1$ for any $1 \leq i \leq T K$ . 
# A.2. Verification of Remark 4.8
Let $K ( \cdot , \cdot )$ be the NTK kernel, then for $i , j \in [ T K ]$ , we have $\mathbf { H } _ { i , j } = K ( \mathbf { x } ^ { i } , \mathbf { x } ^ { j } )$ . Suppose that $h \in \mathcal H$ , then $h$ can be decomposed as $h = h _ { \mathbf { H } } + h _ { \perp }$ , where $\begin{array} { r } { h _ { \mathbf { H } } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { T K } \alpha _ { i } K ( \mathbf { x } , \mathbf { x } ^ { i } ) } \end{array}$ is the projection of $h$ to the function space spanned by $\{ K ( \mathbf { x } , \mathbf { x } ^ { i } ) \} _ { i = 1 } ^ { T K }$ and $h _ { \perp }$ is the orthogonal part. By definition we have $h ( \mathbf { x } ^ { i } ) = h _ { \mathbf { H } } ( \mathbf { x } ^ { i } )$ for $i \in [ T K ]$ , thus 
$$
\begin{array}{l} \mathbf {h} = \left[ h \left(\mathbf {x} ^ {1}\right), \dots , h \left(\mathbf {x} ^ {T K}\right) \right] ^ {\top} \\ = \left[ h _ {\mathbf {H}} \left(\mathbf {x} ^ {1}\right), \dots , h _ {\mathbf {H}} \left(\mathbf {x} ^ {T K}\right) \right] ^ {\top} \\ = \left[ \sum_ {i = 1} ^ {T K} \alpha_ {i} K (\mathbf {x} ^ {1}, \mathbf {x} ^ {i}), \dots , \sum_ {i = 1} ^ {T K} \alpha_ {i} K (\mathbf {x} ^ {T K}, \mathbf {x} ^ {i}) \right] ^ {\top} \\ = \mathbf {H} \alpha , \\ \end{array}
$$
which implies that ${ \pmb { \alpha } } = \mathbf { H } ^ { - 1 } \mathbf { h }$ . Thus, we have 
$$
\| h \| _ {\mathcal {H}} \geq \| h _ {\mathbf {H}} \| _ {\mathcal {H}} = \sqrt {\boldsymbol {\alpha} ^ {\top} \mathbf {H} \boldsymbol {\alpha}} = \sqrt {\mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {H} \mathbf {H} ^ {- 1} \mathbf {h}} = \sqrt {\mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h}}.
$$
# A.3. Proof of Corollary 4.9
Proof of Corollary 4.9. Notice that $R _ { T } \leq T$ since $0 \leq h ( \mathbf { x } ) \leq 1$ . Thus, with the fact that with probability at least $1 - \delta$ (4.3) holds, we can bound $\mathbb { E } [ R _ { T } ]$ as 
$$
\begin{array}{l} \mathbb {E} \left[ R _ {T} \right] \leq (1 - \delta) \left(3 \sqrt {T} \sqrt {\tilde {d} \log (1 + T K / \lambda) + 2} \left[ \nu \sqrt {\tilde {d} \log (1 + T K / \lambda) + 2 - 2 \log \delta} \right. \right. \\ \left. + 2 \sqrt {\lambda} S + \left(\lambda + C _ {2} T L\right) \left(1 - \eta m \lambda\right) ^ {J / 2} \sqrt {T / \lambda} \right] + 1\bigg) + \delta T. \tag {A.1} \\ \end{array}
$$
Taking $\delta = 1 / T$ completes the proof. 
# B. Proof of Lemmas in Section 5
# B.1. Proof of Lemma 5.1
We start with the following lemma: 
Neural Contextual Bandits with UCB-based Exploration 
Lemma B.1. Let $\mathbf { G } = [ \mathbf { g } ( \mathbf { x } ^ { 1 } ; \pmb { \theta } _ { 0 } ) , \dots , \mathbf { g } ( \mathbf { x } ^ { T K } ; \pmb { \theta } _ { 0 } ) ] / \sqrt { m } \in \mathbb { R } ^ { p \times ( T K ) }$ . Let H be the NTK matrix as defined in Definition 4.1. For any $\delta \in ( 0 , 1 )$ , if 
$$
m = \Omega \bigg (\frac {L ^ {6} \log (T K L / \delta)}{\epsilon^ {4}} \bigg),
$$
then with probability at least $1 - \delta$ , we have 
$$
\| \mathbf {G} ^ {\top} \mathbf {G} - \mathbf {H} \| _ {F} \leq T K \epsilon .
$$
We begin to prove Lemma 5.1. 
Proof of Lemma 5.1. By Assumption 4.2, we know that $\lambda _ { 0 } > 0$ . By the choice of $m$ , we have $m \ge \Omega ( L ^ { 6 } \log ( T K L / \delta ) / \epsilon ^ { 4 } )$ , where $\epsilon = \lambda _ { 0 } / ( 2 T K )$ . Thus, due to Lemma B.1, with probability at least $1 - \delta$ , we have $\| \mathbf G ^ { \top } \mathbf G - \mathbf H \| _ { F } \leq T K \epsilon = \lambda _ { 0 } / 2$ . That leads to 
$$
\mathbf {G} ^ {\top} \mathbf {G} \succeq \mathbf {H} - \| \mathbf {G} ^ {\top} \mathbf {G} - \mathbf {H} \| _ {F} \mathbf {I} \succeq \mathbf {H} - \lambda_ {0} \mathbf {I} / 2 \succeq \mathbf {H} / 2 \succ 0, \tag {B.1}
$$
where the first inequality holds due to the triangle inequality, the third and fourth inequality holds due to $\mathbf { H } \succeq \lambda _ { 0 } \mathbf { I } \succ 0$ . Thus, suppose the singular value decomposition of $\mathbf { G }$ is $\mathbf { G } = \mathbf { P A Q } ^ { \top }$ , √ $\mathbf { P } \in \mathbb { R } ^ { p \times T K }$ , A ∈ RTK×TK, $\mathbf { Q } \in \mathbb { R } ^ { T K \times T K }$ , we have $\mathbf A \succ 0$ . Now we are going to show that $\pmb { \theta } ^ { * } = \pmb { \theta } _ { 0 } + \mathbf { P } \mathbf { A } ^ { - 1 } \mathbf { Q } ^ { \top } \mathbf { h } / \sqrt { m }$ satisfies (5.1). First, we have 
$$
\mathbf {G} ^ {\top} \sqrt {m} \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0}\right) = \mathbf {Q} \mathbf {A} \mathbf {P} ^ {\top} \mathbf {P} \mathbf {A} ^ {- 1} \mathbf {Q} ^ {\top} \mathbf {h} = \mathbf {h},
$$
which suggests that for any $i$ , $\langle \mathbf { g } ( \mathbf { x } ^ { i } ; \pmb { \theta } _ { 0 } ) , \pmb { \theta } ^ { * } - \pmb { \theta } _ { 0 } \rangle = h ( \mathbf { x } ^ { i } )$ . We also have 
$$
m \| \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \| _ {2} ^ {2} = \mathbf {h} ^ {\top} \mathbf {Q} \mathbf {A} ^ {- 2} \mathbf {Q} ^ {\top} \mathbf {h} = \mathbf {h} ^ {\top} (\mathbf {G} ^ {\top} \mathbf {G}) ^ {- 1} \mathbf {h} \leq 2 \mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h},
$$
where the last inequality holds due to (B.1). This completes the proof. 
# B.2. Proof of Lemma 5.2
In this section we prove Lemma 5.2. For simplicity, we define $\bar { \mathbf { Z } } _ { t } , \bar { \mathbf { b } } _ { t } , \bar { \boldsymbol { \gamma } } _ { t }$ as follows: 
$$
\bar {\mathbf {Z}} _ {t} = \lambda \mathbf {I} + \sum_ {i = 1} ^ {t} \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) ^ {\top} / m,
$$
$$
\bar {\mathbf {b}} _ {t} = \sum_ {i = 1} ^ {t} r _ {i, a _ {i}} \mathbf {g} (\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}) / \sqrt {m},
$$
$$
\bar {\gamma} _ {t} = \nu \sqrt {\log \frac {\det \bar {\mathbf {Z}} _ {t}}{\det \lambda \mathbf {I}} - 2 \log \delta} + \sqrt {\lambda} S.
$$
We need the following lemmas. The first lemma shows that the network parameter $\theta _ { t }$ at round $t$ can be well approximated by ${ \pmb { \theta } } _ { 0 } + \bar { \bf Z } _ { t } ^ { - 1 } \bar { \bf b } _ { t } / \sqrt { m }$ . 
Lemma B.2. There exist constants $\{ \bar { C } _ { i } \} _ { i = 1 } ^ { 5 } > 0$ such that for any $\delta > 0$ , if for all $t \in [ T ] , \eta , m$ satisfy 
$$
2 \sqrt {t / (m \lambda)} \geq \bar {C} _ {1} m ^ {- 3 / 2} L ^ {- 3 / 2} [ \log (T K L ^ {2} / \delta) ] ^ {3 / 2},
$$
$$
2 \sqrt {t / (m \lambda)} \leq \bar {C} _ {2} \min  \left\{L ^ {- 6} [ \log m ] ^ {- 3 / 2}, \left(m (\lambda \eta) ^ {2} L ^ {- 6} t ^ {- 1} (\log m) ^ {- 1}\right) ^ {3 / 8} \right\},
$$
$$
\eta \leq \bar {C} _ {3} (m \lambda + t m L) ^ {- 1},
$$
$$
m ^ {1 / 6} \geq \bar {C} _ {4} \sqrt {\log m} L ^ {7 / 2} t ^ {7 / 6} \lambda^ {- 7 / 6} (1 + \sqrt {t / \lambda}),
$$
then with probability at least $1 - \delta$ , we have that $\| \pmb \theta _ { t } - \pmb \theta _ { 0 } \| _ { 2 } \le 2 \sqrt { t / ( m \lambda ) }$ and 
$$
\| \pmb {\theta} _ {t} - \pmb {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m} \| _ {2} \leq (1 - \eta m \lambda) ^ {J / 2} \sqrt {t / (m \lambda)} + \bar {C} _ {5} m ^ {- 2 / 3} \sqrt {\log m} L ^ {7 / 2} t ^ {5 / 3} \lambda^ {- 5 / 3} (1 + \sqrt {t / \lambda}).
$$
Neural Contextual Bandits with UCB-based Exploration 
Next lemma shows the error bounds for $\bar { \mathbf Z } _ { t }$ and $\mathbf { Z } _ { t }$ . 
Lemma B.3. There exist constants $\{ \bar { C } _ { i } \} _ { i = 1 } ^ { 5 } > 0$ such that for any $\delta > 0$ , if $m$ satisfies that 
$$
\bar {C} _ {1} m ^ {- 3 / 2} L ^ {- 3 / 2} [ \log (T K L ^ {2} / \delta) ] ^ {3 / 2} \leq 2 \sqrt {t / (m \lambda)} \leq \bar {C} _ {2} L ^ {- 6} [ \log m ] ^ {- 3 / 2}, \forall t \in [ T ],
$$
then with probability at least $1 - \delta$ , for any $t \in [ T ]$ , we have 
$$
\left\| \mathbf {Z} _ {t} \right\| _ {2} \leq \lambda + \bar {C} _ {3} t L,
$$
$$
\left\| \bar {\mathbf {Z}} _ {t} - \mathbf {Z} _ {t} \right\| _ {F} \leq \bar {C} _ {4} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} t ^ {7 / 6} \lambda^ {- 1 / 6},
$$
$$
\left| \log \frac {\det  (\bar {\mathbf {Z}} _ {t})}{\det  (\lambda \mathbf {I})} - \log \frac {\det  (\mathbf {Z} _ {t})}{\det  (\lambda \mathbf {I})} \right| \leq \bar {C} _ {5} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} t ^ {5 / 3} \lambda^ {- 1 / 6}.
$$
With above lemmas, we prove Lemma 5.2 as follows. 
Proof of Lemma 5.2. By Lemma B.2 we know that $\| \pmb \theta _ { t } - \pmb \theta _ { 0 } \| _ { 2 } \le 2 \sqrt { t / ( m \lambda ) }$ . By Lemma 5.1, with probability at least $1 - \delta$ , there exists $\pmb { \theta } ^ { * }$ such that for any $1 \leq t \leq T$ , 
$$
h \left(\mathbf {x} _ {t, a _ {t}}\right) = \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {0}\right) / \sqrt {m}, \sqrt {m} \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0}\right) \right\rangle , \tag {B.2}
$$
$$
\sqrt {m} \| \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \| _ {2} \leq \sqrt {2 \mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h}} \leq S, \tag {B.3}
$$
where the second inequality holds since $S \geq \sqrt { 2 \mathbf { h } ^ { \top } \mathbf { H } ^ { - 1 } \mathbf { h } }$ in the statement of Lemma 5.2. Thus, conditioned on (B.2) and (B.3), by Theorem 2 in Abbasi-Yadkori et al. (2011), with probability at least $1 - \delta$ , for any $1 \leq t \leq T$ , $\pmb { \theta } ^ { * }$ satisfies that 
$$
\left\| \sqrt {m} \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0}\right) - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} \right\| _ {\bar {\mathbf {Z}} _ {t}} \leq \bar {\gamma} _ {t}. \tag {B.4}
$$
We now prove that $\| \pmb { \theta } ^ { * } - \pmb { \theta } _ { t } \| _ { \mathbf { Z } _ { t } } \leq \gamma _ { t } / \sqrt { m }$ . From the triangle inequality, 
$$
\left\| \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {t} \right\| _ {\mathbf {Z} _ {t}} \leq \underbrace {\left\| \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m} \right\| _ {\mathbf {Z} _ {t}}} _ {I _ {1}} + \underbrace {\left\| \boldsymbol {\theta} _ {t} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m} \right\| _ {\mathbf {Z} _ {t}}} _ {I _ {2}}. \tag {B.5}
$$
We bound $I _ { 1 }$ and $I _ { 2 }$ separately. For $I _ { 1 }$ , we have 
$$
\begin{array}{l} I _ {1} ^ {2} = \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) ^ {\top} \mathbf {Z} _ {t} \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) \\ = \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) ^ {\top} \bar {\mathbf {Z}} _ {t} \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) \\ + \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) ^ {\top} \left(\mathbf {Z} _ {t} - \bar {\mathbf {Z}} _ {t}\right) \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) \\ \leq \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) ^ {\top} \bar {\mathbf {Z}} _ {t} \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) \\ + \frac {\left\| \mathbf {Z} _ {t} - \bar {\mathbf {Z}} _ {t} \right\| _ {2}}{\lambda} \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) ^ {\top} \bar {\mathbf {Z}} _ {t} \left(\boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m}\right) \\ \leq \left(1 + \| \mathbf {Z} _ {t} - \bar {\mathbf {Z}} _ {t} \| _ {2} / \lambda\right) \bar {\gamma} _ {t} ^ {2} / m, \tag {B.6} \\ \end{array}
$$
where the first inequality holds due to the fact that $\mathbf { x } ^ { \top } \mathbf { A } \mathbf { x } \leq \mathbf { x } ^ { \top } \mathbf { B } \mathbf { x } \cdot \| \mathbf { A } \| _ { 2 } / \lambda _ { \operatorname* { m i n } } ( \mathbf { B } )$ for some $\mathbf { B } \succ 0$ and the fact that $\lambda _ { \operatorname* { m i n } } ( \bar { \mathbf Z } _ { t } ) \geq \lambda$ , the second inequality holds due to (B.4). We have 
$$
\left\| \bar {\mathbf {Z}} _ {t} - \mathbf {Z} _ {t} \right\| _ {2} \leq \left\| \bar {\mathbf {Z}} _ {t} - \mathbf {Z} _ {t} \right\| _ {F} \leq C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} t ^ {7 / 6} \lambda^ {- 1 / 6}, \tag {B.7}
$$
where the first inequality holds due to the fact that $\| \mathbf { A } \| _ { 2 } \leq \| \mathbf { A } \| _ { F }$ , the second inequality holds due to Lemma B.3. We also have 
$$
\begin{array}{l} \bar {\gamma} _ {t} = \nu \sqrt {\log \frac {\det \bar {\mathbf {Z}} _ {t}}{\det \lambda \mathbf {I}} - 2 \log \delta} + \sqrt {\lambda} S \\ = \nu \sqrt {\log \frac {\det \mathbf {Z} _ {t}}{\det \lambda \mathbf {I}} + \log \frac {\det \bar {\mathbf {Z}} _ {t}}{\det \lambda \mathbf {I}} - \log \frac {\det \mathbf {Z} _ {t}}{\det \lambda \mathbf {I}} - 2 \log \delta} + \sqrt {\lambda} S \\ \leq \nu \sqrt {\log \frac {\det  \mathbf {Z} _ {t}}{\det  \lambda \mathbf {I}} + C _ {2} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} t ^ {5 / 3} \lambda^ {- 1 / 6} - 2 \log \delta} + \sqrt {\lambda} S, \tag {B.8} \\ \end{array}
$$
Neural Contextual Bandits with UCB-based Exploration 
where $C _ { 1 } , C _ { 2 } > 0$ are two constants, the inequality holds due to Lemma B.3. Substituting (B.7) and (B.8) into (B.6), we have 
$$
\begin{array}{l} I _ {1} \leq \sqrt {1 + \| \mathbf {Z} _ {t} - \bar {\mathbf {Z}} _ {t} \| _ {2} / \lambda} \bar {\gamma} _ {t} / \sqrt {m} \\ \leq \sqrt {1 + C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} t ^ {7 / 6} \lambda^ {- 7 / 6}} / \sqrt {m} \\ \cdot \left(\nu \sqrt {\log \frac {\det  \mathbf {Z} _ {t}}{\det  \lambda \mathbf {I}} + C _ {2} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} t ^ {5 / 3} \lambda^ {- 1 / 6} - 2 \log \delta} + \sqrt {\lambda} S\right). \tag {B.9} \\ \end{array}
$$
For $I _ { 2 }$ , we have 
$$
\begin{array}{l} I _ {2} = \left\| \boldsymbol {\theta} _ {t} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m} \right\| _ {\mathbf {Z} _ {t}} \\ \leq \left\| \mathbf {Z} _ {t} \right\| _ {2} \cdot \left\| \boldsymbol {\theta} _ {t} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m} \right\| _ {2} \\ \leq \left(\lambda + C _ {3} t L\right) \| \boldsymbol {\theta} _ {t} - \boldsymbol {\theta} _ {0} - \bar {\mathbf {Z}} _ {t} ^ {- 1} \bar {\mathbf {b}} _ {t} / \sqrt {m} \| _ {2} \\ \leq \left(\lambda + C _ {3} t L\right) \left[ \left(1 - \eta m \lambda\right) ^ {J / 2} \sqrt {t / (m \lambda)} + m ^ {- 2 / 3} \sqrt {\log m} L ^ {7 / 2} t ^ {5 / 3} \lambda^ {- 5 / 3} \left(1 + \sqrt {t / \lambda}\right) \right], \tag {B.10} \\ \end{array}
$$
where $C _ { 3 } > 0$ is a constant, the first inequality holds since for any vector a, the second inequality holds due to $\| \mathbf { Z } _ { t } \| _ { 2 } \leq$ $\lambda + C _ { 3 } t L$ by Lemma B.3, the third inequality holds due to Lemma B.2. Substituting (B.9) and (B.10) into (B.5), we obtain $\left\| \pmb { \theta } ^ { * } - \pmb { \theta } _ { t } \right\| _ { \mathbf { Z } _ { t } } \leq \gamma _ { t } / \sqrt { m }$ . This completes the proof. □ 
# B.3. Proof of Lemma 5.3
The proof starts with three lemmas that bound the error terms of the function value and gradient of neural networks. 
Lemma B.4 (Lemma 4.1, Cao & Gu (2019)). There exist constants $\{ \bar { C } _ { i } \} _ { i = 1 } ^ { 3 } > 0$ such that for any $\delta > 0$ , if $\tau$ satisfies that 
$$
\bar {C} _ {1} m ^ {- 3 / 2} L ^ {- 3 / 2} [ \log (T K L ^ {2} / \delta) ] ^ {3 / 2} \leq \tau \leq \bar {C} _ {2} L ^ {- 6} [ \log m ] ^ {- 3 / 2},
$$
then with probability at least $1 - \delta$ , for all $\widetilde { \theta } , \widehat { \theta }$ satisfying $\| \widetilde { \pmb { \theta } } - \pmb { \theta } _ { 0 } \| _ { 2 } \leq \tau , \| \widehat { \pmb { \theta } } - \pmb { \theta } _ { 0 } \| _ { 2 } \leq \tau$ and $j \in [ T K ]$ we have 
$$
\left| f (\mathbf {x} ^ {j}; \widetilde {\boldsymbol {\theta}}) - f (\mathbf {x} ^ {j}; \widehat {\boldsymbol {\theta}}) - \langle \mathbf {g} (\mathbf {x} ^ {j}; \widehat {\boldsymbol {\theta}}), \widetilde {\boldsymbol {\theta}} - \widehat {\boldsymbol {\theta}} \rangle \right| \leq \bar {C} _ {3} \tau^ {4 / 3} L ^ {3} \sqrt {m \log m}.
$$
Lemma B.5 (Theorem 5, Allen-Zhu et al. (2019)). There exist constants $\{ \bar { C } _ { i } \} _ { i = 1 } ^ { 3 } > 0$ such that for any $\delta \in ( 0 , 1 )$ , if $\tau$ satisfies that 
$$
\bar {C} _ {1} m ^ {- 3 / 2} L ^ {- 3 / 2} \max  \left\{\log^ {- 3 / 2} m, \log^ {3 / 2} (T K / \delta) \right\} \leq \tau \leq \bar {C} _ {2} L ^ {- 9 / 2} \log^ {- 3} m,
$$
then with probability at least $1 - \delta$ , for all $\lVert { \pmb \theta } - { \pmb \theta } _ { 0 } \rVert _ { 2 } \leq \tau$ and $j \in [ T K ]$ we have 
$$
\left\| \mathbf {g} \left(\mathbf {x} ^ {j}; \boldsymbol {\theta}\right) - \mathbf {g} \left(\mathbf {x} ^ {j}; \boldsymbol {\theta} _ {0}\right) \right\| _ {2} \leq \bar {C} _ {3} \sqrt {\log m} \tau^ {1 / 3} L ^ {3} \left\| \mathbf {g} \left(\mathbf {x} ^ {j}; \boldsymbol {\theta} _ {0}\right) \right\| _ {2}.
$$
Lemma B.6 (Lemma B.3, Cao & Gu (2019)). There exist constants $\{ \bar { C } _ { i } \} _ { i = 1 } ^ { 3 } > 0$ such that for any $\delta > 0$ , if $\tau$ satisfies that 
$$
\bar {C} _ {1} m ^ {- 3 / 2} L ^ {- 3 / 2} \left[ \log \left(T K L ^ {2} / \delta\right) \right] ^ {3 / 2} \leq \tau \leq \bar {C} _ {2} L ^ {- 6} \left[ \log m \right] ^ {- 3 / 2},
$$
then with probability at least $1 - \delta$ , for any $\lVert { \pmb \theta } - { \pmb \theta } _ { 0 } \rVert _ { 2 } \leq \tau$ and $j \in [ T K ]$ we have $\| \mathbf { g } ( \mathbf { x } ^ { j } ; \pmb { \theta } ) \| _ { F } \leq \bar { C } _ { 3 } \sqrt { m L }$ . 
Proof of Lemma 5.3. We follow the regret bound analysis in Abbasi-Yadkori et al. (2011); Valko et al. (2013). Denote $a _ { t } ^ { * } = \operatorname { a r g m a x } _ { a \in [ K ] } h ( \mathbf { x } _ { t , a } )$ and $\mathcal { C } _ { t } = \{ \pmb \theta : \| \pmb \theta - \pmb \theta _ { t } \| _ { \pmb { Z } _ { t } } \le \gamma _ { t } / \sqrt { m } \}$ . By Lemma 5.2, for all $1 \leq t \leq T$ , we have $\| \pmb \theta _ { t } - \pmb \theta _ { 0 } \| _ { 2 } \le 2 \sqrt { t / ( m \lambda ) }$ and $\pmb { \theta } ^ { * } \in \mathcal { C } _ { t }$ . By the choice of $m$ , Lemmas B.4, B.5 and B.6 hold. Thus, $h ( \mathbf { x } _ { t , a _ { t } ^ { * } } ) - h ( \mathbf { x } _ { t , a _ { t } } )$ can 
Neural Contextual Bandits with UCB-based Exploration 
be bounded as follows: 
$$
\begin{array}{l} h (\mathbf {x} _ {t, a _ {t} ^ {*}}) - h (\mathbf {x} _ {t, a _ {t}}) \\ = \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t} ^ {*}}; \boldsymbol {\theta} _ {0}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle - \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {0}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle \\ \leq \langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t} ^ {*}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \rangle - \langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \rangle \\ + \| \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \| _ {2} (\| \mathbf {g} \left(\mathbf {x} _ {t, a _ {t} ^ {*}}; \boldsymbol {\theta} _ {t - 1}\right) - \mathbf {g} \left(\mathbf {x} _ {t, a _ {t} ^ {*}}; \boldsymbol {\theta} _ {0}\right) \| _ {2} + \| \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right) - \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {0}\right) \| _ {2}) \\ \leq \langle \mathbf {g} (\mathbf {x} _ {t, a _ {t} ^ {*}}; \boldsymbol {\theta} _ {t - 1}), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \rangle - \langle \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \rangle + C _ {1} \sqrt {\mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h}} m ^ {- 1 / 6} \sqrt {\log m} t ^ {1 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2} \\ \leq \underbrace {\max  _ {\boldsymbol {\theta} \in \mathcal {C} _ {t - 1}} \left\langle \mathbf {g} \left(\mathbf {x} _ {t , a _ {t} ^ {*}} ; \boldsymbol {\theta} _ {t - 1}\right) , \boldsymbol {\theta} - \boldsymbol {\theta} _ {0} \right\rangle - \left\langle \mathbf {g} \left(\mathbf {x} _ {t , a _ {t}} ; \boldsymbol {\theta} _ {t - 1}\right) , \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle} _ {I _ {1}} + C _ {1} \sqrt {\mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h}} m ^ {- 1 / 6} \sqrt {\log m} t ^ {1 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2}, \tag {B.11} \\ \end{array}
$$
where the equality holds due to Lemma 5.1, the first inequality holds due to triangle inequality, the second inequality holds due to Lemmas 5.1, B.5, B.6, the third inequality holds due to $\theta ^ { * } \in \mathcal { C } _ { t - 1 }$ . Denote 
$$
\widetilde {U} _ {t, a} = \langle \mathbf {g} (\mathbf {x} _ {t, a}; \pmb {\theta} _ {t - 1}), \pmb {\theta} _ {t - 1} - \pmb {\theta} _ {0} \rangle + \gamma_ {t - 1} \sqrt {\mathbf {g} (\mathbf {x} _ {t , a} ; \pmb {\theta} _ {t - 1}) ^ {\top} \mathbf {Z} _ {t - 1} ^ {- 1} \mathbf {g} (\mathbf {x} _ {t , a} ; \pmb {\theta} _ {t - 1}) / m},
$$
then we have $\begin{array} { r } { \widetilde { U } _ { t , a } = \operatorname* { m a x } _ { \pmb { \theta } \in \mathcal { C } _ { t - 1 } } \langle \mathbf { g } ( \mathbf { x } _ { t , a } ; \pmb { \theta } _ { t - 1 } ) , \pmb { \theta } - \pmb { \theta } _ { 0 } \rangle } \end{array}$ due to the fact that 
$$
\max_{\mathbf{x}:\| \mathbf{x} - \mathbf{b}\|_{\mathbf{A}}\leq c}\langle \mathbf{a},\mathbf{x}\rangle = \langle \mathbf{a},\mathbf{b}\rangle +c\sqrt{\mathbf{a}^{\top}\mathbf{A}^{-1}\mathbf{a}}.
$$
Recall the definition of $U _ { t , a }$ from Algorithm 1, we also have 
$$
\begin{array}{l} \left| U _ {t, a} - \tilde {U} _ {t, a} \right| = \left| f \left(\mathbf {x} _ {t, a}; \boldsymbol {\theta} _ {t - 1}\right) - \langle \mathbf {g} \left(\mathbf {x} _ {t, a}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} _ {t - 1} - \boldsymbol {\theta} _ {0} \rangle \right| \\ = \left| f \left(\mathbf {x} _ {t, a}; \boldsymbol {\theta} _ {t - 1}\right) - f \left(\mathbf {x} _ {t, a}; \boldsymbol {\theta} _ {0}\right) - \langle \mathbf {g} \left(\mathbf {x} _ {t, a}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} _ {t - 1} - \boldsymbol {\theta} _ {0} \rangle \right| \\ \leq C _ {2} m ^ {- 1 / 6} \sqrt {\log m} t ^ {2 / 3} \lambda^ {- 2 / 3} L ^ {3}, \tag {B.12} \\ \end{array}
$$
where $C _ { 2 } > 0$ is a constant, the second equality holds due to $f ( \mathbf { x } ^ { j } ; \pmb { \theta } _ { 0 } ) = 0$ by the random initialization of $\pmb { \theta } _ { 0 }$ , the inequality holds due to Lemma B.4 with the fact $\| \dot { \pmb { \theta _ { t - 1 } } } - \pmb { \theta _ { 0 } } \| _ { 2 } \leq 2 \sqrt { t / ( m \lambda ) } )$ . Since $\theta ^ { * } \in \mathcal { C } _ { t - 1 }$ , then $I _ { 1 }$ in (B.11) can be bounded as 
$$
\begin{array}{l} \max _ {\boldsymbol {\theta} \in \mathcal {C} _ {t - 1}} \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t} ^ {*}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} - \boldsymbol {\theta} _ {0} \right\rangle - \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle \\ = \tilde {U} _ {t, a _ {t} ^ {*}} - \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle \\ \leq U _ {t, a _ {t} ^ {*}} - \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle + C _ {2} m ^ {- 1 / 6} \sqrt {\log m} t ^ {2 / 3} \lambda^ {- 2 / 3} L ^ {3} \\ \leq U _ {t, a _ {t}} - \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle + C _ {2} m ^ {- 1 / 6} \sqrt {\log m} t ^ {2 / 3} \lambda^ {- 2 / 3} L ^ {3} \\ \leq \widetilde {U} _ {t, a _ {t}} - \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle + 2 C _ {2} m ^ {- 1 / 6} \sqrt {\log m} t ^ {2 / 3} \lambda^ {- 2 / 3} L ^ {3}, \tag {B.13} \\ \end{array}
$$
where the first inequality holds due to (B.12), the second inequality holds since $a _ { t } = \operatorname { a r g m a x } _ { a } U _ { t , a }$ , the third inequality holds due to (B.12). Furthermore, 
$$
\begin{array}{l} \tilde {U} _ {t, a _ {t}} - \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle \\ = \max  _ {\boldsymbol {\theta} \in \mathcal {C} _ {t - 1}} \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} - \boldsymbol {\theta} _ {0} \right\rangle - \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {0} \right\rangle \\ = \max  _ {\boldsymbol {\theta} \in \mathcal {C} _ {t - 1}} \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} - \boldsymbol {\theta} _ {t - 1} \right\rangle - \left\langle \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right), \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {t - 1} \right\rangle \\ \leq \max  _ {\boldsymbol {\theta} \in \mathcal {C} _ {t - 1}} \left\| \boldsymbol {\theta} - \boldsymbol {\theta} _ {t - 1} \right\| _ {\mathbf {Z} _ {t - 1}} \| \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}) \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} + \left\| \boldsymbol {\theta} ^ {*} - \boldsymbol {\theta} _ {t - 1} \right\| _ {\mathbf {Z} _ {t - 1}} \| \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}) \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} \\ \leq 2 \gamma_ {t - 1} \| \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}}, \tag {B.14} \\ \end{array}
$$
Neural Contextual Bandits with UCB-based Exploration 
where the first inequality holds due to Holder inequality, the second inequality holds due to Lemma ¨ 5.2. Combining (B.11), (B.13) and (B.14), we have 
$$
\begin{array}{l} h \left(\mathbf {x} _ {t, a _ {t} ^ {*}}\right) - h \left(\mathbf {x} _ {t, a _ {t}}\right) \\ \leq 2 \gamma_ {t - 1} \| \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} + C _ {1} \sqrt {\mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h}} m ^ {- 1 / 6} \sqrt {\log m} t ^ {1 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2} \\ + 2 C _ {2} m ^ {- 1 / 6} \sqrt {\log m} t ^ {2 / 3} \lambda^ {- 2 / 3} L ^ {3} \\ \leq \min  \left\{2 \gamma_ {t - 1} \| \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} + C _ {1} \sqrt {\mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h}} m ^ {- 1 / 6} \sqrt {\log m} t ^ {1 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2} \right. \\ \left. + 2 C _ {2} m ^ {- 1 / 6} \sqrt {\log m} t ^ {2 / 3} \lambda^ {- 2 / 3} L ^ {3}, 1 \right\} \\ \leq \min  \left\{2 \gamma_ {t - 1} \| \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}}, 1 \right\} + C _ {1} \sqrt {\mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h}} m ^ {- 1 / 6} \sqrt {\log m} t ^ {1 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2} \\ + 2 C _ {2} m ^ {- 1 / 6} \sqrt {\log m} t ^ {2 / 3} \lambda^ {- 2 / 3} L ^ {3} \\ \leq 2 \gamma_ {t - 1} \min  \left\{\| \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}}, 1 \right\} + C _ {1} \sqrt {\mathbf {h} ^ {\top} \mathbf {H} ^ {- 1} \mathbf {h}} m ^ {- 1 / 6} \sqrt {\log m} t ^ {1 / 6} \lambda^ {- 1 / 6} L ^ {7 / 2} \\ + 2 C _ {2} m ^ {- 1 / 6} \sqrt {\log m} t ^ {2 / 3} \lambda^ {- 2 / 3} L ^ {3}, \tag {B.15} \\ \end{array}
$$
where the second inequality holds due to the fact that $0 \leq h ( \mathbf { x } _ { t , a _ { t } ^ { * } } ) - h ( \mathbf { x } _ { t , a _ { t } } ) \leq 1$ , the third inequality holds due to the fact√ that√ $\operatorname* { m i n } \{ a + b , 1 \} \leq \operatorname* { m i n } \{ a , 1 \} + b$ , the fourth inequality holds due to the fact $\gamma _ { t - 1 } \geq \sqrt { \lambda } S \geq 1$ . Finally, by the fact that $\sqrt { 2 \mathbf { h } \mathbf { H } ^ { - 1 } \mathbf { h } } \leq S$ , the proof completes. □ 
# B.4. Proof of Lemma 5.4
In this section we prove Lemma 5.4, we need the following lemma from Abbasi-Yadkori et al. (2011). 
Lemma B.7 (Lemma 11, Abbasi-Yadkori et al. (2011)). We have the following inequality: 
$$
\sum_ {t = 1} ^ {T} \min  \left\{\| \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} ^ {2}, 1 \right\} \leq 2 \log \frac {\det  \mathbf {Z} _ {T}}{\det  \lambda \mathbf {I}}.
$$
Proof of Lemma 5.4. First by the definition of $\gamma _ { t }$ , we know that $\gamma _ { t }$ is a monotonic function w.r.t. det $\mathbf { Z } _ { t }$ . By the definition of $\mathbf { Z } _ { t }$ , we know that $\mathbf { Z } _ { T } \succeq \mathbf { Z } _ { t }$ , which implies that d $\operatorname { e t } \mathbf { Z } _ { t } \leq \operatorname* { d e t } \mathbf { Z } _ { T }$ . Thus, $\gamma _ { t } \leq \gamma _ { T }$ . Second, by Lemma B.7 we know that 
$$
\begin{array}{l} \sum_ {t = 1} ^ {T} \min \left\{\| \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \pmb {\theta} _ {t - 1}) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} ^ {2}, 1 \right\} \\ \leq 2 \log \frac {\det \mathbf {Z} _ {T}}{\det \lambda \mathbf {I}} \\ \leq 2 \log \frac {\det \bar {\mathbf {Z}} _ {T}}{\det \lambda \mathbf {I}} + C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {5 / 3} \lambda^ {- 1 / 6}, \tag {B.16} \\ \end{array}
$$
where the second inequality holds due to Lemma B.3. Next we are going to bound $\log \operatorname* { d e t } \bar { \mathbf { Z } } _ { T }$ . Denote ${ \bf G } =$ $[ \mathbf { g } ( \mathbf { x } ^ { 1 } ; \pmb { \theta } _ { 0 } ) / \sqrt { m } , \dots , \mathbf { g } ( \mathbf { x } ^ { T K } ; \pmb { \theta } _ { 0 } ) / \sqrt { m } ] \in \mathbb { R } ^ { p \times ( T K ) }$ , then we have 
$$
\begin{array}{l} \log \frac {\det  \bar {\mathbf {Z}} _ {T}}{\det  \lambda \mathbf {I}} = \log \det  \left(\mathbf {I} + \sum_ {t = 1} ^ {T} \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {0}) \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {0}) ^ {\top} / (m \lambda)\right) \\ \leq \log \det  \left(\mathbf {I} + \sum_ {i = 1} ^ {T K} \mathbf {g} \left(\mathbf {x} ^ {i}; \boldsymbol {\theta} _ {0}\right) \mathbf {g} \left(\mathbf {x} ^ {i}; \boldsymbol {\theta} _ {0}\right) ^ {\top} / (m \lambda)\right) \\ = \log \det  \left(\mathbf {I} + \mathbf {G G} ^ {\top} / \lambda\right) \\ = \log \det  \left(\mathbf {I} + \mathbf {G} ^ {\top} \mathbf {G} / \lambda\right), \tag {B.17} \\ \end{array}
$$
Neural Contextual Bandits with UCB-based Exploration 
where the inequality holds naively, the third equality holds since for any matrix $\mathbf { A } \in \mathbb { R } ^ { p \times T K }$ , we have $\begin{array} { r l } { \operatorname* { d e t } ( \mathbf { I } + \mathbf { A } \mathbf { A } ^ { \top } ) = } & { { } } \end{array}$ $\operatorname* { d e t } ( \mathbf { I } + \mathbf { A } ^ { \top } \mathbf { A } )$ . We can further bound (B.17) as follows: 
$$
\begin{array}{l} \log \det \left(\mathbf {I} + \mathbf {G} ^ {\top} \mathbf {G} / \lambda\right) = \log \det \left(\mathbf {I} + \mathbf {H} / \lambda + (\mathbf {G} ^ {\top} \mathbf {G} - \mathbf {H}) / \lambda\right) \\ \leq \log \det  \left(\mathbf {I} + \mathbf {H} / \lambda\right) + \langle (\mathbf {I} + \mathbf {H} / \lambda) ^ {- 1}, (\mathbf {G} ^ {\top} \mathbf {G} - \mathbf {H}) / \lambda \rangle \\ \leq \log \det  \left(\mathbf {I} + \mathbf {H} / \lambda\right) + \| (\mathbf {I} + \mathbf {H} / \lambda) ^ {- 1} \| _ {F} \| \mathbf {G} ^ {\top} \mathbf {G} - \mathbf {H} \| _ {F} / \lambda \\ \leq \log \det  \left(\mathbf {I} + \mathbf {H} / \lambda\right) + \sqrt {T K} \| \mathbf {G} ^ {\top} \mathbf {G} - \mathbf {H} \| _ {F} \\ \leq \log \det  \left(\mathbf {I} + \mathbf {H} / \lambda\right) + 1 \\ = \tilde {d} \log (1 + T K / \lambda) + 1, \tag {B.18} \\ \end{array}
$$
where the first inequality holds due to the concavity of $\log \operatorname* { d e t } ( \cdot )$ , the second inequality holds due to the fact that $\langle \mathbf { A } , \mathbf { B } \rangle \leq \| \mathbf { A } \| _ { F } \| \mathbf { B } \| _ { F }$ , the third inequality holds due to the facts that $\mathbf { I } + \mathbf { H } / \lambda \succeq \mathbf { I } .$ , $\lambda \geq 1$ and $\| \mathbf { A } \| _ { F } \leq \sqrt { T K } \| \mathbf { A } \| _ { 2 }$ for any $\mathbf { A } \in \mathbb { R } ^ { T K \times T K }$ , the fourth inequality holds by Lemma B.1 with the choice of $m$ , the fifth inequality holds by the definition of effective dimension in Definition 4.3, and the last inequality holds due to the choice of $\lambda$ . Substituting (B.18) into (B.17), we obtain that 
$$
\log \frac {\det  \bar {\mathbf {Z}} _ {T}}{\det  \lambda \mathbf {I}} \leq \widetilde {d} \log (1 + T K / \lambda) + 1. \tag {B.19}
$$
Substituting (B.19) into (B.16), we have 
$$
\sum_ {t = 1} ^ {T} \min  \left\{\| \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} ^ {2}, 1 \right\} \leq 2 \widetilde {d} \log (1 + T K / \lambda) + 2 + C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {5 / 3} \lambda^ {- 1 / 6}. \tag {B.20}
$$
We now bound $\gamma _ { T }$ , which is 
$$
\begin{array}{l} \gamma_ {T} = \sqrt {1 + C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {7 / 6} \lambda^ {- 7 / 6}} \\ \cdot \left(\nu \sqrt {\log \frac {\det  \mathbf {Z} _ {T}}{\det  \lambda \mathbf {I}} + C _ {2} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {5 / 3} \lambda^ {- 1 / 6} - 2 \log \delta} + \sqrt {\lambda} S\right) \\ + (\lambda + C _ {3} T L) \Big [ (1 - \eta m \lambda) ^ {J / 2} \sqrt {T / (m \lambda)} + m ^ {- 2 / 3} \sqrt {\log m} L ^ {7 / 2} T ^ {5 / 3} \lambda^ {- 5 / 3} (1 + \sqrt {T / \lambda}) \Big ] \\ \leq \sqrt {1 + C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {7 / 6} \lambda^ {- 7 / 6}} \\ \cdot \left(\nu \sqrt {\log \frac {\det  \bar {\mathbf {Z}} _ {T}}{\det  \lambda \mathbf {I}} + 2 C _ {2} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {5 / 3} \lambda^ {- 1 / 6} - 2 \log \delta} + \sqrt {\lambda} S\right) \\ + (\lambda + C _ {3} T L) \left[ (1 - \eta m \lambda) ^ {J / 2} \sqrt {T / (m \lambda)} + m ^ {- 2 / 3} \sqrt {\log m} L ^ {7 / 2} T ^ {5 / 3} \lambda^ {- 5 / 3} (1 + \sqrt {T / \lambda}) \right], \tag {B.21} \\ \end{array}
$$
Neural Contextual Bandits with UCB-based Exploration 
where the inequality holds due to Lemma B.3. Finally, we have 
$$
\begin{array}{l} \sqrt {\sum_ {t = 1} ^ {T} \gamma_ {t - 1} ^ {2} \min  \left\{\| \mathbf {g} \left(\mathbf {x} _ {t , a _ {t}} ; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} ^ {2} , 1 \right\}} \\ \leq \gamma_ {T} \sqrt {\sum_ {t = 1} ^ {T} \min  \left\{\| \mathbf {g} \left(\mathbf {x} _ {t , a _ {t}} ; \boldsymbol {\theta} _ {t - 1}\right) / \sqrt {m} \| _ {\mathbf {Z} _ {t - 1} ^ {- 1}} ^ {2} , 1 \right\}} \\ \leq \sqrt {\log \frac {\det  \bar {\mathbf {Z}} _ {T}}{\det  \lambda \mathbf {I}} + C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {5 / 3} \lambda^ {- 1 / 6}} \left[ \sqrt {1 + C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {7 / 6} \lambda^ {- 7 / 6}} \right] \\ \cdot \left(\nu \sqrt {\log \frac {\det  \bar {\mathbf {Z}} _ {T}}{\det  \lambda \mathbf {I}}} + 2 C _ {2} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {5 / 3} \lambda^ {- 1 / 6} - 2 \log \delta + \sqrt {\lambda} S\right) \\ + (\lambda + C _ {3} T L) \left[ (1 - \eta m \lambda) ^ {J / 2} \sqrt {T / (m \lambda)} + m ^ {- 3 / 2} \sqrt {\log m} L ^ {7 / 2} T ^ {5 / 3} \lambda^ {- 5 / 3} (1 + \sqrt {T / \lambda}) \right] \\ \leq \sqrt {\tilde {d} \log (1 + T K / \lambda) + 1 + C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {5 / 3} \lambda^ {- 1 / 6}} \left[ \sqrt {1 + C _ {1} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {7 / 6} \lambda^ {- 7 / 6}} \right] \\ \cdot \left(\nu \sqrt {\tilde {d} \log (1 + T K / \lambda) + 1 + 2 C _ {2} m ^ {- 1 / 6} \sqrt {\log m} L ^ {4} T ^ {5 / 3} \lambda^ {- 1 / 6} - 2 \log \delta} + \sqrt {\lambda} S\right) \\ + (\lambda + C _ {3} T L) \Big [ (1 - \eta m \lambda) ^ {J / 2} \sqrt {T / (m \lambda)} + m ^ {- 3 / 2} \sqrt {\log m} L ^ {7 / 2} T ^ {5 / 3} \lambda^ {- 5 / 3} (1 + \sqrt {T / \lambda}) \Big ] \Big ], \\ \end{array}
$$
where the first inequality holds due to the fact that $\gamma _ { t - 1 } \leq \gamma _ { T }$ , the second inequality holds due to (B.20) and (B.21), the third inequality holds due to (B.19). This completes our proof. □ 
# C. Proofs of Technical Lemmas in Appendix B
# C.1. Proof of Lemma B.1
In this section we prove Lemma B.1, we need the following lemma from Arora et al. (2019): 
Lemma C.1 (Theorem 3.1, Arora et al. (2019)). Fix $\epsilon > 0$ and $\delta \in ( 0 , 1 )$ . Suppose that 
$$
m = \Omega \bigg (\frac {L ^ {6} \log (L / \delta)}{\epsilon^ {4}} \bigg),
$$
then for any $i , j \in [ T K ]$ , with probability at least $1 - \delta$ over random initialization of $\pmb { \theta } _ { 0 }$ , we have 
$$
\left| \left\langle \mathbf {g} \left(\mathbf {x} ^ {i}; \boldsymbol {\theta} _ {0}\right), \mathbf {g} \left(\mathbf {x} ^ {j}; \boldsymbol {\theta} _ {0}\right) \right\rangle / m - \mathbf {H} _ {i, j} \right| \leq \epsilon . \tag {C.1}
$$
Proof of Lemma B.1. Taking union bound over $i , j \in [ T K ]$ , we have that if 
$$
m = \Omega \bigg (\frac {L ^ {6} \log (T ^ {2} K ^ {2} L / \delta)}{\epsilon^ {4}} \bigg),
$$
then with probability at least $1 - \delta$ , (C.1) holds for all $( i , j ) \in [ T K ] \times [ T K ]$ . Therefore, we have 
$$
\| \mathbf {G} ^ {\top} \mathbf {G} - \mathbf {H} \| _ {F} = \sqrt {\sum_ {i = 1} ^ {T K} \sum_ {j = 1} ^ {T K} | \langle \mathbf {g} (\mathbf {x} ^ {i} ; \boldsymbol {\theta} _ {0}) , \mathbf {g} (\mathbf {x} ^ {j} ; \boldsymbol {\theta} _ {0}) \rangle / m - \mathbf {H} _ {i , j} | ^ {2}} \leq T K \epsilon .
$$
□ 
Neural Contextual Bandits with UCB-based Exploration 
# C.2. Proof of Lemma B.2
In this section we prove Lemma B.2. During the proof, for simplicity, we omit the subscript $t$ by default. We define the following quantities: 
$$
\mathbf {J} ^ {(j)} = \left(\mathbf {g} \left(\mathbf {x} _ {1, a _ {1}}; \boldsymbol {\theta} ^ {(j)}\right), \dots , \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} ^ {(j)}\right)\right) \in \mathbb {R} ^ {(m d + m ^ {2} (L - 2) + m) \times t},
$$
$$
\mathbf {H} ^ {(j)} = [ \mathbf {J} ^ {(j)} ] ^ {\top} \mathbf {J} ^ {(j)} \in \mathbb {R} ^ {t \times t},
$$
$$
\mathbf {f} ^ {(j)} = \left(f \left(\mathbf {x} _ {1, a _ {1}}; \boldsymbol {\theta} ^ {(j)}\right), \dots , f \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} ^ {(j)}\right)\right) ^ {\top} \in \mathbb {R} ^ {t \times 1},
$$
$$
\mathbf {y} = \left(r _ {1, a _ {1}}, \dots , r _ {t, a _ {t}}\right) \in \mathbb {R} ^ {t \times 1}.
$$
Then the update rule of $\pmb \theta ^ { ( j ) }$ can be written as follows: 
$$
\boldsymbol {\theta} ^ {(j + 1)} = \boldsymbol {\theta} ^ {(j)} - \eta \left[ \mathbf {J} ^ {(j)} \left(\mathbf {f} ^ {(j)} - \mathbf {y}\right) + m \lambda \left(\boldsymbol {\theta} ^ {(j)} - \boldsymbol {\theta} ^ {(0)}\right) \right]. \tag {C.2}
$$
We also define the following auxiliary sequence $\{ \widetilde { \pmb { \theta } } ^ { ( k ) } \}$ during the proof: 
$$
\widetilde {\boldsymbol {\theta}} ^ {(0)} = \boldsymbol {\theta} ^ {(0)}, \widetilde {\boldsymbol {\theta}} ^ {(j + 1)} = \widetilde {\boldsymbol {\theta}} ^ {(j)} - \eta \left[ \mathbf {J} ^ {(0)} \left(\left[ \mathbf {J} ^ {(0)} \right] ^ {\top} \left(\widetilde {\boldsymbol {\theta}} ^ {(j)} - \widetilde {\boldsymbol {\theta}} ^ {(0)}\right) - \mathbf {y}\right) + m \lambda \left(\widetilde {\boldsymbol {\theta}} ^ {(j)} - \widetilde {\boldsymbol {\theta}} ^ {(0)}\right) \right].
$$
Next lemma provides perturbation bounds for $\mathbf { J } ^ { ( j ) } , \mathbf { H } ^ { ( j ) }$ and $\| \mathbf { f } ^ { ( j + 1 ) } - \mathbf { f } ^ { ( j ) } - [ \mathbf { J } ^ { ( j ) } ] ^ { \top } ( \pmb { \theta } ^ { ( j + 1 ) } - \pmb { \theta } ^ { ( j ) } ) \| _ { 2 } .$ 
Lemma C.2. There exist constants $\{ \bar { C } _ { i } \} _ { i = 1 } ^ { 6 } > 0$ such that for any $\delta > 0$ , if $\tau$ satisfies that 
$$
\bar {C} _ {1} m ^ {- 3 / 2} L ^ {- 3 / 2} \left[ \log \left(T K L ^ {2} / \delta\right) \right] ^ {3 / 2} \leq \tau \leq \bar {C} _ {2} L ^ {- 6} \left[ \log m \right] ^ {- 3 / 2},
$$
then with probability at least $1 - \delta$ , if for any $j \in [ J ] , \| \pmb { \theta } ^ { ( j ) } - \pmb { \theta } ^ { ( 0 ) } \| _ { 2 } \leq \tau$ , we have the following inequalities for any $j , s \in [ J ]$ , 
$$
\left\| \mathbf {J} ^ {(j)} \right\| _ {F} \leq \bar {C} _ {4} \sqrt {t m L}, \tag {C.3}
$$
$$
\left\| \mathbf {J} ^ {(j)} - \mathbf {J} ^ {(0)} \right\| _ {F} \leq \bar {C} _ {5} \sqrt {t m \log m} \tau^ {1 / 3} L ^ {7 / 2}, \tag {C.4}
$$
$$
\left\| \mathbf {f} ^ {(s)} - \mathbf {f} ^ {(j)} - \left[ \mathbf {J} ^ {(j)} \right] ^ {\top} \left(\boldsymbol {\theta} ^ {(s)} - \boldsymbol {\theta} ^ {(j)}\right) \right\| _ {2} \leq \bar {C} _ {6} \tau^ {4 / 3} L ^ {3} \sqrt {t m \log m}, \tag {C.5}
$$
$$
\| \mathbf {y} \| _ {2} \leq \sqrt {t}. \tag {C.6}
$$
Next lemma gives an upper bound for $\| \mathbf { f } ^ { ( j ) } - \mathbf { y } \| _ { 2 }$ . 
Lemma C.3. There exist constants $\{ \bar { C } _ { i } \} _ { i = 1 } ^ { 4 } > 0$ such that for any $\delta > 0$ , if $\tau , \eta$ satisfy that 
$$
\bar {C} _ {1} m ^ {- 3 / 2} L ^ {- 3 / 2} \left[ \log \left(T K L ^ {2} / \delta\right) \right] ^ {3 / 2} \leq \tau \leq \bar {C} _ {2} L ^ {- 6} \left[ \log m \right] ^ {- 3 / 2},,
$$
$$
\eta \leq \bar {C} _ {3} (m \lambda + t m L) ^ {- 1},
$$
$$
\tau^ {8 / 3} \leq \bar {C} _ {4} m (\lambda \eta) ^ {2} L ^ {- 6} t ^ {- 1} (\log m) ^ {- 1},
$$
then with probability at least $1 - \delta$ , if for any $j \in [ J ] , \| \pmb { \theta } ^ { ( j ) } - \pmb { \theta } ^ { ( 0 ) } \| _ { 2 } \leq \tau$ , we have that for any $j \in [ J ] , \| \mathbf { f } ^ { ( j ) } - \mathbf { y } \| _ { 2 } \leq 2 { \sqrt { t } }$ 
Next lemma gives an upper bound of the distance between auxiliary sequence $\lVert \widetilde { \pmb { \theta } } ^ { ( j ) } - \pmb { \theta } ^ { ( 0 ) } \rVert _ { 2 }$ . 
Lemma C.4. There exist constants $\{ \bar { C } _ { i } \} _ { i = 1 } ^ { 3 } > 0$ such that for any $\delta \in ( 0 , 1 )$ , if $\tau , \eta$ satisfy that 
$$
\bar {C} _ {1} m ^ {- 3 / 2} L ^ {- 3 / 2} \left[ \log \left(T K L ^ {2} / \delta\right) \right] ^ {3 / 2} \leq \tau \leq \bar {C} _ {2} L ^ {- 6} \left[ \log m \right] ^ {- 3 / 2},,
$$
$$
\eta \leq \bar {C} _ {3} (t m L + m \lambda) ^ {- 1},
$$
then with probability at least $1 - \delta$ , we have that for any $j \in [ J ]$ , 
$$
\begin{array}{l} \left\| \widetilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(0)} \right\| _ {2} \leq \sqrt {t / (m \lambda)}, \\ \left\| \tilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(0)} - \bar {\mathbf {Z}} ^ {- 1} \bar {\mathbf {b}} / \sqrt {m} \right\| _ {2} \leq (1 - \eta m \lambda) ^ {j / 2} \sqrt {t / (m \lambda)} \\ \end{array}
$$
Neural Contextual Bandits with UCB-based Exploration 
With above lemmas, we prove Lemma B.2 as follows. 
Proof of Lemma B.2. Set $\tau = 2 \sqrt { t / ( m \lambda ) }$ . First we assume that $\lVert { \pmb \theta } ^ { ( j ) } - { \pmb \theta } ^ { ( 0 ) } \rVert _ { 2 } \leq \tau$ for all $0 \le j \le J$ . Then with this assumption and the choice of $m , \tau$ , we have that Lemma C.2, C.3 and C.4 hold. Then we have 
$$
\begin{array}{l} \left\| \boldsymbol {\theta} ^ {(j + 1)} - \widetilde {\boldsymbol {\theta}} ^ {(j + 1)} \right\| _ {2} = \left\| \boldsymbol {\theta} ^ {(j)} - \widetilde {\boldsymbol {\theta}} ^ {(j)} - \eta (\mathbf {J} ^ {(j)} - \mathbf {J} ^ {(0)}) (\mathbf {f} ^ {(j)} - \mathbf {y}) - \eta m \lambda (\boldsymbol {\theta} ^ {(j)} - \widetilde {\boldsymbol {\theta}} ^ {(j)}) \right. \\ \left. - \eta \mathbf {J} ^ {(0)} \left(\mathbf {f} ^ {(j)} - \left[ \mathbf {J} ^ {(0)} \right] ^ {\top} \left(\tilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(0)}\right)\right) \right\| _ {2} \\ = \left\| (1 - \eta m \lambda) (\boldsymbol {\theta} ^ {(j)} - \widetilde {\boldsymbol {\theta}} ^ {(j)}) - \eta (\mathbf {J} ^ {(j)} - \mathbf {J} ^ {(0)}) (\mathbf {f} ^ {(j)} - \mathbf {y}) \right. \\ \left. - \eta \mathbf {J} ^ {(0)} \left[ \mathbf {f} ^ {(j)} - \left[ \mathbf {J} ^ {(0)} \right] ^ {\top} \left(\boldsymbol {\theta} ^ {(j)} - \boldsymbol {\theta} ^ {(0)}\right) + \left[ \mathbf {J} ^ {(0)} \right] ^ {\top} \left(\boldsymbol {\theta} ^ {(j)} - \widetilde {\boldsymbol {\theta}} ^ {(j)}\right) \right] \right\| _ {2} \\ \leq \underbrace {\eta \left\| (\mathbf {J} ^ {(j)} - \mathbf {J} ^ {(0)}) (\mathbf {f} ^ {(j)} - \mathbf {y}) \right\| _ {2}} _ {I _ {1}} + \underbrace {\eta \| \mathbf {J} ^ {(0)} \| _ {2} \| \mathbf {f} ^ {(j)} - [ \mathbf {J} ^ {(0)} ] (\boldsymbol {\theta} ^ {(j)} - \boldsymbol {\theta} ^ {(0)}) \| _ {2}} _ {I _ {2}} \\ + \underbrace {\left\| \left[ \mathbf {I} - \eta (m \lambda \mathbf {I} + \mathbf {H} ^ {(0)}) \right] \left(\widetilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(j)}\right) \right\| _ {2}} _ {I _ {3}}, \tag {C.7} \\ \end{array}
$$
where the inequality holds due to triangle inequality. We now bound $I _ { 1 } , I _ { 2 }$ and $I _ { 3 }$ separately. For $I _ { 1 }$ , we have 
$$
I _ {1} \leq \eta \left\| \mathbf {J} ^ {(j)} - \mathbf {J} ^ {(0)} \right\| _ {2} \| \mathbf {f} ^ {(j)} - \mathbf {y} \| _ {2} \leq \eta C _ {2} t \sqrt {m \log m} \tau^ {1 / 3} L ^ {7 / 2}, \tag {C.8}
$$
where $C _ { 2 } > 0$ is a constant, the first inequality holds due to the definition of matrix spectral norm and the second inequality holds due to (C.4) in Lemma C.2 and Lemma C.3. For $I _ { 2 }$ , we have 
$$
I _ {2} \leq \eta \left\| \mathbf {J} ^ {(0)} \right\| _ {2} \left\| \mathbf {f} ^ {(j)} - \mathbf {J} ^ {(0)} \left(\boldsymbol {\theta} ^ {(j)} - \boldsymbol {\theta} ^ {(0)}\right) \right\| _ {2} \leq \eta C _ {3} t m L ^ {7 / 2} \tau^ {4 / 3} \sqrt {\log m}, \tag {C.9}
$$
where $C _ { 3 } > 0$ , the first inequality holds due to matrix spectral norm, the second inequality holds due to (C.3) and (C.5) in Lemma C.2 and the fact that $\mathbf { f } ^ { ( 0 ) } = \mathbf { 0 }$ by random initialization over ${ \pmb \theta } ^ { ( 0 ) }$ . For $I _ { 3 }$ , we have 
$$
I _ {3} \leq \left\| \mathbf {I} - \eta (m \lambda \mathbf {I} + \mathbf {H} ^ {(0)}) \right\| _ {2} \left\| \widetilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(j)} \right\| _ {2} \leq (1 - \eta m \lambda) \left\| \widetilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(j)} \right\| _ {2}, \tag {C.10}
$$
where the first inequality holds due to spectral norm inequality, the second inequality holds since 
$$
\eta (m \lambda \mathbf {I} + \mathbf {H} ^ {(0)}) = \eta (m \lambda \mathbf {I} + [ \mathbf {J} ^ {(0)} ] ^ {\top} \mathbf {J} ^ {(0)}) \preceq \eta (m \lambda \mathbf {I} + C _ {1} t m L \mathbf {I}) \preceq \mathbf {I},
$$
for some $C _ { 1 } > 0$ , the first inequality holds due to (C.3) in Lemma C.2, the second inequality holds due to the choice of $\eta$ Substituting (C.8), (C.9) and (C.10) into (C.7), we obtain 
$$
\left\| \boldsymbol {\theta} ^ {(j + 1)} - \widetilde {\boldsymbol {\theta}} ^ {(j + 1)} \right\| _ {2} \leq (1 - \eta m \lambda) \left\| \boldsymbol {\theta} ^ {(j)} - \widetilde {\boldsymbol {\theta}} ^ {(j)} \right\| _ {2} + C _ {4} \left(\eta t \sqrt {m \log m} \tau^ {1 / 3} L ^ {7 / 2} + \eta t m L ^ {7 / 2} \tau^ {4 / 3} \sqrt {\log m}\right), \tag {C.11}
$$
where $C _ { 4 } > 0$ is a constant. By recursively applying (C.11) from 0 to $j$ , we have 
$$
\begin{array}{l} \left\| \boldsymbol {\theta} ^ {(j + 1)} - \widetilde {\boldsymbol {\theta}} ^ {(j + 1)} \right\| _ {2} \leq C _ {4} \frac {\eta t \sqrt {m \log m} \tau^ {1 / 3} L ^ {7 / 2} + \eta t m L ^ {7 / 2} \tau^ {4 / 3} \sqrt {\log m}}{\eta m \lambda} \\ = C _ {5} m ^ {- 2 / 3} \sqrt {\log m} L ^ {7 / 2} t ^ {5 / 3} \lambda^ {- 5 / 3} (1 + \sqrt {t / \lambda}) \\ \leq \frac {\tau}{2}, \tag {C.12} \\ \end{array}
$$
where $C _ { 5 } > 0$ is a constant, the equality holds by the definition of $\tau$ , the last inequality holds due to the choice of $m$ , where 
$$
m ^ {1 / 6} \geq C _ {6} \sqrt {\log m} L ^ {7 / 2} t ^ {7 / 6} \lambda^ {- 7 / 6} (1 + \sqrt {t / \lambda}),
$$
and $C _ { 6 } > 0$ is a constant. Thus, for any $j \in [ J ]$ , we have 
$$
\left\| \boldsymbol {\theta} ^ {(j)} - \boldsymbol {\theta} ^ {(0)} \right\| _ {2} \leq \left\| \widetilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(0)} \right\| _ {2} + \left\| \boldsymbol {\theta} ^ {(j)} - \widetilde {\boldsymbol {\theta}} ^ {(j)} \right\| _ {2} \leq \sqrt {t / (m \lambda)} + \tau / 2 = \tau , \tag {C.13}
$$
Neural Contextual Bandits with UCB-based Exploration 
where the first inequality holds due to triangle inequality, the second inequality holds due to Lemma C.4. (C.13) suggests that our assumption $\| \pmb \theta ^ { ( j ) } - \pmb \theta ^ { ( 0 ) } \| _ { 2 } \le \tau$ holds for any $j$ . Note that we have the following inequality by Lemma C.4: 
$$
\left\| \widetilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(0)} - (\bar {\mathbf {Z}}) ^ {- 1} \bar {\mathbf {b}} / \sqrt {m} \right\| _ {2} \leq (1 - \eta m \lambda) ^ {j} \sqrt {t / (m \lambda)}. \tag {C.14}
$$
Using (C.12) and (C.14), we have 
$$
\left\| \boldsymbol {\theta} ^ {(j)} - \boldsymbol {\theta} ^ {(0)} - \bar {\mathbf {Z}} ^ {- 1} \bar {\mathbf {b}} / \sqrt {m} \right\| _ {2} \leq (1 - \eta m \lambda) ^ {j / 2} \sqrt {t / (m \lambda)} + C _ {5} m ^ {- 2 / 3} \sqrt {\log m} L ^ {7 / 2} t ^ {5 / 3} \lambda^ {- 5 / 3} (1 + \sqrt {t / \lambda}).
$$
This completes the proof. 
# C.3. Proof of Lemma B.3
In this section we prove Lemma B.3. 
Proof of Lemma B.3. Set $\tau = 2 \sqrt { t / ( m \lambda ) }$ . By Lemma B.2 we have that $\lVert { \pmb \theta } _ { i } - { \pmb \theta } _ { 0 } \rVert _ { 2 } \leq \tau$ for $i \in [ t ]$ . $\| \mathbf Z _ { t } \| _ { 2 }$ can be bounded as follows. 
$$
\begin{array}{l} \| \mathbf {Z} _ {t} \| _ {2} = \left\| \lambda \mathbf {I} + \sum_ {i = 1} ^ {t} \mathbf {g} (\mathbf {x} _ {i, a _ {i}}; \pmb {\theta} _ {i - 1}) \mathbf {g} (\mathbf {x} _ {i, a _ {i}}; \pmb {\theta} _ {i - 1}) ^ {\top} / m \right\| _ {2} \\ \leq \lambda + \left\| \lambda \mathbf {I} + \sum_ {i = 1} ^ {t} \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i - 1}\right) \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i - 1}\right) ^ {\top} / m \right\| _ {2} \\ \leq \lambda + \sum_ {i = 1} ^ {t} \left\| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i - 1}\right) \right\| _ {2} ^ {2} / m \\ \leq \lambda + C _ {0} t L, \\ \end{array}
$$
where $C _ { 0 } > 0$ is a constant, the first inequality holds due to the fact that $\lVert \mathbf { a } \mathbf { a } ^ { \top } \rVert _ { F } = \lVert \mathbf { a } \rVert _ { 2 } ^ { 2 }$ , the second inequality holds due to Lemma B.6 with the fact that $\lVert { \pmb \theta } _ { i } - { \pmb \theta } _ { 0 } \rVert _ { 2 } \leq \tau$ . We bound $\| \mathbf Z _ { t } - \bar { \mathbf Z } _ { t } \| _ { 2 }$ as follows. We have 
$$
\begin{array}{l} \left\| \mathbf {Z} _ {t} - \bar {\mathbf {Z}} _ {t} \right\| _ {F} = \left\| \sum_ {i = 1} ^ {t} \left(\mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) ^ {\top} - \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}\right) \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}\right) ^ {\top}\right) / m \right\| _ {F} \\ \leq \sum_ {i = 1} ^ {t} \left\| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) ^ {\top} - \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}\right) \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}\right) ^ {\top} \right\| _ {F} / m \\ \leq \sum_ {i = 1} ^ {t} \left(\left\| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) \right\| _ {2} + \left\| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}\right) \right\| _ {2}\right) \left\| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) - \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}\right) \right\| _ {2} / m, \tag {C.15} \\ \end{array}
$$
where the first inequality holds due to triangle inequality, the second inequality holds the fact that $\| \mathbf { a } \mathbf { a } ^ { \top } - \mathbf { b } \mathbf { b } ^ { \top } \| _ { F } \leq$ $( \| \mathbf { a } \| _ { 2 } + \| \mathbf { b } \| _ { 2 } ) \| \mathbf { a } - \mathbf { b } \| _ { 2 }$ for any vectors a, b. To bound (C.15), we have 
$$
\left\| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) \right\| _ {2}, \left\| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}\right) \right\| _ {2} \leq C _ {1} \sqrt {m L}, \tag {C.16}
$$
where $C _ { 1 } > 0$ is a constant, the inequality holds due to Lemma B.6 with the fact that $\lVert { \pmb \theta } _ { i } - { \pmb \theta } _ { 0 } \rVert _ { 2 } \leq \tau$ . We also have 
$$
\left\| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) - \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}\right) \right\| _ {2} \leq C _ {2} \sqrt {\log m} \tau^ {1 / 3} L ^ {3} \| \mathbf {g} \left(\mathbf {x} _ {j}; \boldsymbol {\theta} _ {0}\right) \| _ {2} \leq C _ {3} \sqrt {m \log m} \tau^ {1 / 3} L ^ {7 / 2}, \tag {C.17}
$$
where $C _ { 2 } , C _ { 3 } > 0$ are constants, the first inequality holds due to Lemma B.5 with the fact that $\lVert { \pmb \theta } _ { i } - { \pmb \theta } _ { 0 } \rVert _ { 2 } \leq \tau$ , the second inequality holds due to Lemma B.6. Substituting (C.16) and (C.17) into (C.15), we have 
$$
\left\| \mathbf {Z} _ {t} - \bar {\mathbf {Z}} _ {t} \right\| _ {F} \leq C _ {4} t \sqrt {\log m} \tau^ {1 / 3} L ^ {4},
$$
where $C _ { 4 } > 0$ is a constant. We now bound $\log \operatorname* { d e t } \bar { \mathbf Z } _ { t } - \log \operatorname* { d e t } \mathbf Z _ { t }$ . It is easy to verify that $\bar { \mathbf Z } _ { t } = \lambda \mathbf I + \bar { \mathbf J } \bar { \mathbf J } ^ { \top }$ , $\mathbf Z _ { t } = \lambda \mathbf I + \mathbf J \mathbf J ^ { \top }$ , where 
$$
\begin{array}{l} \bar {\mathbf {J}} = \left(\mathbf {g} \left(\mathbf {x} _ {1, a _ {1}}; \boldsymbol {\theta} _ {0}\right), \dots , \mathbf {g} \left(\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {0}\right)\right) / \sqrt {m}, \\ \mathbf {J} = \left(\mathbf {g} (\mathbf {x} _ {1, a _ {1}}; \boldsymbol {\theta} _ {0}), \dots , \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \boldsymbol {\theta} _ {t - 1})\right) / \sqrt {m}. \\ \end{array}
$$
Neural Contextual Bandits with UCB-based Exploration 
We have the following inequalities: 
$$
\begin{array}{l} \log \frac {\det (\bar {\mathbf {Z}} _ {t})}{\det (\lambda \mathbf {I})} - \log \frac {\det (\mathbf {Z} _ {t})}{\det (\lambda \mathbf {I})} = \log \det (\mathbf {I} + \bar {\mathbf {J}} \bar {\mathbf {J}} ^ {\top} / \lambda) - \log \det (\mathbf {I} + \mathbf {J} \mathbf {J} ^ {\top} / \lambda) \\ = \log \det (\mathbf {I} + \bar {\mathbf {J}} ^ {\top} \bar {\mathbf {J}} / \lambda) - \log \det (\mathbf {I} + \mathbf {J} ^ {\top} \mathbf {J} / \lambda) \\ \leq \left\langle \left(\mathbf {I} + \mathbf {J} ^ {\top} \mathbf {J} / \lambda\right) ^ {- 1}, \bar {\mathbf {J}} ^ {\top} \bar {\mathbf {J}} - \mathbf {J} ^ {\top} \mathbf {J} \right\rangle \\ \leq \left\| \left(\mathbf {I} + \mathbf {J} ^ {\top} \mathbf {J} / \lambda\right) ^ {- 1} \right\| _ {F} \left\| \bar {\mathbf {J}} ^ {\top} \bar {\mathbf {J}} - \mathbf {J} ^ {\top} \mathbf {J} \right\| _ {F} \\ \leq \sqrt {t} \| (\mathbf {I} + \mathbf {J} ^ {\top} \mathbf {J} / \lambda) ^ {- 1} \| _ {2} \| \bar {\mathbf {J}} ^ {\top} \bar {\mathbf {J}} - \mathbf {J} ^ {\top} \mathbf {J} \| _ {F} \\ \leq \sqrt {t} \| \bar {\mathbf {J}} ^ {\top} \bar {\mathbf {J}} - \mathbf {J} ^ {\top} \mathbf {J} \| _ {F}, \tag {C.18} \\ \end{array}
$$
where the second equality holds due to the fact that $\operatorname* { d e t } ( \mathbf { I } + \mathbf { A } \mathbf { A } ^ { \top } ) = \operatorname* { d e t } ( \mathbf { I } + \mathbf { A } ^ { \top } \mathbf { A } )$ , the first inequality holds due to the fact that log det function is convex, the second inequality hold due to the fact that $\langle \mathbf { A } , \mathbf { B } \rangle \leq \| \mathbf { A } \| _ { F } \| \mathbf { B } \| _ { F }$ , the third inequality holds since $\mathbf { I } + \mathbf { J } ^ { \top } \mathbf { J } / \lambda$ is a $t$ -dimension matrix, the fourth inequality holds since $\mathbf { I } + \mathbf { J } ^ { \top } \mathbf { J } / \lambda \succeq \mathbf { I } .$ We have 
$$
\begin{array}{l} \left\| \bar {\mathbf {J}} ^ {\top} \bar {\mathbf {J}} - \mathbf {J} ^ {\top} \mathbf {J} \right\| _ {F} \\ \leq t \max  _ {1 \leq i, j \leq t} \left| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}\right) ^ {\top} \mathbf {g} \left(\mathbf {x} _ {j, a _ {j}}; \boldsymbol {\theta} _ {0}\right) - \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}\right) ^ {\top} \mathbf {g} \left(\mathbf {x} _ {j, a _ {j}}; \boldsymbol {\theta} _ {j}\right) \right| / m \\ \leq t \max _ {1 \leq i, j \leq t} \left\| \mathbf {g} (\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {0}) - \mathbf {g} (\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} _ {i}) \right\| _ {2} \left\| \mathbf {g} (\mathbf {x} _ {j, a _ {j}}; \boldsymbol {\theta} _ {j}) \right\| _ {2} / m \\ + \left\| \mathbf {g} (\mathbf {x} _ {j, a _ {j}}; \pmb {\theta} _ {0}) - \mathbf {g} (\mathbf {x} _ {j, a _ {j}}; \pmb {\theta} _ {j}) \right\| _ {2} \left\| \mathbf {g} (\mathbf {x} _ {i, a _ {i}}; \pmb {\theta} _ {0}) \right\| _ {2} / m \\ \leq C _ {5} t \sqrt {\log m} \tau^ {1 / 3} L ^ {4}, \tag {C.19} \\ \end{array}
$$
where $C _ { 5 } > 0$ is a constant, the first inequality holds due to the fact that $\| \mathbf { A } \| _ { F } \leq t \operatorname* { m a x } | \mathbf { A } _ { i , j } |$ for any $\mathbf { A } \in \mathbb { R } ^ { t \times t }$ , the second inequality holds due to the fact $| \mathbf { a } ^ { \top } \mathbf { a } ^ { \prime } - \mathbf { b } ^ { \top } \mathbf { b } ^ { \prime } | \leq \| \mathbf { a } - \mathbf { b } \| _ { 2 } \| \mathbf { b } ^ { \prime } \| _ { 2 } + \| \mathbf { a } ^ { \prime } - \mathbf { b } ^ { \prime } \| _ { 2 } \| \mathbf { a } \| _ { 2 }$ , the third inequality holds due to (C.16) and (C.17). Substituting (C.19) into (C.18), we obtain 
$$
\log \frac {\operatorname * {d e t} (\bar {\mathbf {Z}} _ {t})}{\operatorname * {d e t} (\lambda \mathbf {I})} - \log \frac {\operatorname * {d e t} (\mathbf {Z} _ {t})}{\operatorname * {d e t} (\lambda \mathbf {I})} \leq C _ {5} t ^ {3 / 2} \sqrt {\log m} \tau^ {1 / 3} L ^ {4}.
$$
Using the same method, we also have 
$$
\log \frac {\mathrm {d e t} (\mathbf {Z} _ {t})}{\mathrm {d e t} (\lambda \mathbf {I})} - \log \frac {\mathrm {d e t} (\bar {\mathbf {Z}} _ {t})}{\mathrm {d e t} (\lambda \mathbf {I})} \leq C _ {5} t ^ {3 / 2} \sqrt {\log m} \tau^ {1 / 3} L ^ {4}.
$$
This completes our proof. 
# D. Proofs of Lemmas in Appendix C
# D.1. Proof of Lemma C.2
In this section we give the proof of Lemma C.2. 
Proof of Lemma C.2. It can be verified that $\tau$ satisfies the conditions of Lemmas B.4, B.5 and B.6. Thus, Lemmas B.4, B.5 and B.6 hold. We will show that for any $j \in [ J ]$ , the following inequalities hold. First, we have 
$$
\left\| \mathbf {J} ^ {(j)} \right\| _ {F} \leq \sqrt {t} \max  _ {i \in [ t ]} \left\| \mathbf {g} \left(\mathbf {x} _ {i, a _ {i}}; \boldsymbol {\theta} ^ {(j)}\right) \right\| _ {2} \leq C _ {1} \sqrt {t m L}, \tag {D.1}
$$
where $C _ { 1 } > 0$ is a constant, the first inequality holds due to the fact that $\| \mathbf { J } ^ { ( j ) } \| _ { F } \leq \sqrt { t } \| \mathbf { J } ^ { ( j ) } \| _ { 2 , \infty }$ , the second inequality holds due to Lemma B.6. 
We also have 
$$
\left\| \mathbf {J} ^ {(j)} - \mathbf {J} ^ {(0)} \right\| _ {F} \leq C _ {2} \sqrt {\log m} \tau^ {1 / 3} L ^ {3} \left\| \mathbf {J} ^ {(0)} \right\| _ {F} \leq C _ {3} \sqrt {t m \log m} \tau^ {1 / 3} L ^ {7 / 2}, \tag {D.2}
$$
Neural Contextual Bandits with UCB-based Exploration 
where $C _ { 2 } , C _ { 3 } > 0$ are constants, the first inequality holds due to Lemma B.5 with the assumption that $\lVert { \pmb \theta } ^ { ( j ) } - { \pmb \theta } ^ { ( 0 ) } \rVert _ { 2 } \leq \tau$ the second inequality holds due to (D.1). 
We also have 
$$
\begin{array}{l} \left\| \mathbf {f} ^ {(s)} - \mathbf {f} ^ {(j)} - [ \mathbf {J} ^ {(j)} ] ^ {\top} (\boldsymbol {\theta} ^ {(s)} - \boldsymbol {\theta} ^ {(j)}) \right\| _ {2} \\ \leq \max _ {i \in [ t ]} \sqrt {t} \big | f (\mathbf {x} _ {i, a _ {i}}; \pmb {\theta} ^ {(s)}) - f (\mathbf {x} _ {i, a _ {i}}; \pmb {\theta} ^ {(j)}) - \langle \mathbf {g} (\mathbf {x} _ {i, a _ {i}}; \pmb {\theta} ^ {(j)}), \pmb {\theta} ^ {(s)} - \pmb {\theta} ^ {(j)} \rangle \big | \\ \leq C _ {4} \tau^ {4 / 3} L ^ {3} \sqrt {t m \log m}, \\ \end{array}
$$
where $C _ { 4 } > 0$ is a constant, the first inequality holds due to the the fact that $\left\| \mathbf { x } \right\| _ { 2 } \leq \sqrt { t } \operatorname* { m a x } \left| x _ { i } \right|$ for any $\mathbf { x } \in \mathbb { R } ^ { t }$ , the second inequality holds due to Lemma B.4 with the assumption that $\| \pmb \theta ^ { ( j ) } - \pmb \theta ^ { ( 0 ) } \| _ { 2 } \le \tau , \| \pmb \theta ^ { ( s ) } - \pmb \theta ^ { ( 0 ) } \| _ { 2 } \le \tau$ . 
For $\| \mathbf { y } \| _ { 2 }$ , we have $\begin{array} { r } { \| \mathbf { y } \| _ { 2 } \leq \sqrt { t } \operatorname* { m a x } _ { 1 \leq i \leq t } | r ( \mathbf { x } _ { i , a _ { i } } ) | \leq \sqrt { t } . } \end{array}$ . This completes our proof. 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/4b82e7c2-1d69-4b88-860d-f5a4d7f97956/0bfbff6760d60dcf3fbb09e85ddc9bfa4c5eb9db7a2d712c32c3aa9a2e142cf4.jpg)

# D.2. Proof of Lemma C.3
Proof of Lemma C.3. It can be verified that $\tau$ satisfies the conditions of Lemma C.2, thus Lemma C.2 holds. Recall that the loss function $L$ is defined as 
$$
L (\pmb {\theta}) = \frac {1}{2} \| \mathbf {f} (\pmb {\theta}) - \mathbf {y} \| _ {2} ^ {2} + \frac {m \lambda}{2} \| \pmb {\theta} - \pmb {\theta} ^ {(0)} \| _ {2} ^ {2}.
$$
We define $\mathbf { J } ( \pmb { \theta } )$ and $\mathbf f ( \pmb \theta )$ as follows: 
$$
\begin{array}{l} \mathbf {J} (\pmb {\theta}) = \left(\mathbf {g} (\mathbf {x} _ {1, a _ {1}}; \pmb {\theta}), \dots , \mathbf {g} (\mathbf {x} _ {t, a _ {t}}; \pmb {\theta})\right) \in \mathbb {R} ^ {(m d + m ^ {2} (L - 2) + m) \times t}, \\ \mathbf {f} (\pmb {\theta}) = \left(f (\mathbf {x} _ {1, a _ {1}}; \pmb {\theta}), \dots , f (\mathbf {x} _ {t, a _ {t}}; \pmb {\theta})\right) ^ {\top} \in \mathbb {R} ^ {t \times 1}. \\ \end{array}
$$
Suppose $\lVert { \pmb \theta } - { \pmb \theta } ^ { ( 0 ) } \rVert _ { 2 } \leq \tau$ . Then by the fact that $\| \cdot \| _ { 2 } ^ { 2 } / 2$ is 1-strongly convex and 1-smooth, we have the following inequalities: 
$$
\begin{array}{l} L (\boldsymbol {\theta} ^ {\prime}) - L (\boldsymbol {\theta}) \\ \leq \langle \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y}, \mathbf {f} (\boldsymbol {\theta} ^ {\prime}) - \mathbf {f} (\boldsymbol {\theta}) \rangle + \frac {1}{2} \left\| \mathbf {f} (\boldsymbol {\theta} ^ {\prime}) - \mathbf {f} (\boldsymbol {\theta}) \right\| _ {2} ^ {2} + m \lambda \langle \boldsymbol {\theta} - \boldsymbol {\theta} ^ {(0)}, \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \rangle + \frac {m \lambda}{2} \left\| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\| _ {2} ^ {2} \\ = \langle \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y}, [ \mathbf {J} (\boldsymbol {\theta}) ] ^ {\top} (\boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta}) + \mathbf {e} \rangle + \frac {1}{2} \left\| [ \mathbf {J} (\boldsymbol {\theta}) ] ^ {\top} (\boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta}) + \mathbf {e} \right\| _ {2} ^ {2} \\ + m \lambda \left\langle \boldsymbol {\theta} - \boldsymbol {\theta} ^ {(0)}, \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\rangle + \frac {m \lambda}{2} \left\| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\| _ {2} ^ {2} \\ = \langle \mathbf {J} (\boldsymbol {\theta}) (\mathbf {f} (\boldsymbol {\theta}) - \mathbf {y}) + m \lambda (\boldsymbol {\theta} - \boldsymbol {\theta} ^ {(0)}), \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \rangle + \langle \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y}, \mathbf {e} \rangle \\ + \frac {1}{2} \left\| [ \mathbf {J} (\boldsymbol {\theta}) ] ^ {\top} \left(\boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta}\right) + \mathbf {e} \right\| _ {2} ^ {2} + \frac {m \lambda}{2} \left\| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\| _ {2} ^ {2} \\ = \left\langle \nabla L (\boldsymbol {\theta}), \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\rangle + \underbrace {\left\langle \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} , \mathbf {e} \right\rangle + \frac {1}{2} \left\| [ \mathbf {J} (\boldsymbol {\theta}) ] ^ {\top} \left(\boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta}\right) + \mathbf {e} \right\| _ {2} ^ {2} + \frac {m \lambda}{2} \left\| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\| _ {2} ^ {2}} _ {I _ {1}}, \tag {D.3} \\ \end{array}
$$
where ${ \bf e } = { \bf f } ( \pmb { \theta } ^ { \prime } ) - { \bf f } ( \pmb { \theta } ) - { \bf J } ( \pmb { \theta } ) ^ { \top } ( \pmb { \theta } ^ { \prime } - \pmb { \theta } )$ . $I _ { 1 }$ can be bounded as follows: 
$$
\begin{array}{l} I _ {1} \leq \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2} + \| \mathbf {J} (\boldsymbol {\theta}) \| _ {2} ^ {2} \| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \| _ {2} ^ {2} + \| \mathbf {e} \| _ {2} ^ {2} + \frac {m \lambda}{2} \left\| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\| _ {2} ^ {2} \\ \leq \frac {C _ {1}}{2} \left((m \lambda + t m L) \| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \| _ {2} ^ {2}\right) + \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2} + \| \mathbf {e} \| _ {2} ^ {2}, \tag {D.4} \\ \end{array}
$$
where the first inequality holds due to Cauchy-Schwarz inequality, the second inequality holds due to the fact that√ $\| \mathbf { J } ( \pmb { \theta } ) \| _ { 2 } \leq C _ { 2 } \sqrt { t m L }$ with $\lVert { \pmb \theta } - { \pmb \theta } ^ { ( 0 ) } \rVert _ { 2 } \leq \tau$ by (C.3) in Lemma C.2. Substituting (D.4) into (D.3), we obtain 
$$
L \left(\boldsymbol {\theta} ^ {\prime}\right) - L (\boldsymbol {\theta}) \leq \langle \nabla L (\boldsymbol {\theta}), \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \rangle + \frac {C _ {1}}{2} \left((m \lambda + t m L) \left\| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\| _ {2} ^ {2}\right) + \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2} + \| \mathbf {e} \| _ {2} ^ {2}. \tag {D.5}
$$
Neural Contextual Bandits with UCB-based Exploration 
Taking $\pmb { \theta } ^ { \prime } = \pmb { \theta } - \eta \nabla L ( \pmb { \theta } )$ , then by (D.5), we have 
$$
L (\boldsymbol {\theta} - \eta \nabla L (\boldsymbol {\theta})) - L (\boldsymbol {\theta}) \leq - \eta \| \nabla L (\boldsymbol {\theta}) \| _ {2} ^ {2} \left[ 1 - C _ {1} (m \lambda + t m L) \eta \right] + \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2} + \| \mathbf {e} \| _ {2} ^ {2}. \tag {D.6}
$$
By the 1-strongly convexity of $\| \cdot \| _ { 2 } ^ { 2 }$ , we further have 
$$
\begin{array}{l} L \left(\boldsymbol {\theta} ^ {\prime}\right) - L (\boldsymbol {\theta}) \\ \geq \langle \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y}, \mathbf {f} \left(\boldsymbol {\theta} ^ {\prime}\right) - \mathbf {f} (\boldsymbol {\theta}) \rangle + m \lambda \langle \boldsymbol {\theta} - \boldsymbol {\theta} ^ {(0)}, \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \rangle + \frac {m \lambda}{2} \| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \| _ {2} ^ {2} \\ = \left\langle \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y}, \left[ \mathbf {J} (\boldsymbol {\theta}) \right] ^ {\top} \left(\boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta}\right) + \mathbf {e} \right\rangle + m \lambda \left\langle \boldsymbol {\theta} - \boldsymbol {\theta} ^ {(0)}, \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\rangle + \frac {m \lambda}{2} \left\| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\| _ {2} ^ {2} \\ = \left\langle \nabla L (\boldsymbol {\theta}), \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\rangle + \frac {m \lambda}{2} \left\| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \right\| _ {2} ^ {2} + \left\langle \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y}, \mathbf {e} \right\rangle \\ \geq \langle \nabla L (\boldsymbol {\theta}), \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \rangle + \frac {m \lambda}{2} \| \boldsymbol {\theta} ^ {\prime} - \boldsymbol {\theta} \| _ {2} ^ {2} - \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2} \\ \geq - \frac {\left\| \nabla L (\boldsymbol {\theta}) \right\| _ {2} ^ {2}}{2 m \lambda} - \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2}, \tag {D.7} \\ \end{array}
$$
where the second inequality holds due to Cauchy-Schwarz inequality, the last inequality holds due to the fact that $\langle \mathbf { a } , \mathbf { x } \rangle +$ $c \| \mathbf { x } \| _ { 2 } ^ { 2 } \geq - \| \mathbf { a } \| _ { 2 } ^ { 2 } / ( 4 c )$ for any vectors a, x and $c > 0$ . Substituting (D.7) into (D.6), we obtain 
$$
\begin{array}{l} L (\boldsymbol {\theta} - \eta \nabla L (\boldsymbol {\theta})) - L (\boldsymbol {\theta}) \\ \leq 2 m \lambda \eta (1 - C _ {1} (m \lambda + t m L) \eta) [ L (\boldsymbol {\theta} ^ {\prime}) - L (\boldsymbol {\theta}) + \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2} ] + \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2} + \| \mathbf {e} \| _ {2} ^ {2} \\ \leq m \lambda \eta \left[ L (\boldsymbol {\theta} ^ {\prime}) - L (\boldsymbol {\theta}) + \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2} \right] + \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} \| \mathbf {e} \| _ {2} + \| \mathbf {e} \| _ {2} ^ {2} \\ \leq m \lambda \eta \left[ L (\boldsymbol {\theta} ^ {\prime}) - L (\boldsymbol {\theta}) + \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} ^ {2} / 8 + 2 \| \mathbf {e} \| _ {2} ^ {2} \right] + m \lambda \eta \| \mathbf {f} (\boldsymbol {\theta}) - \mathbf {y} \| _ {2} ^ {2} / 8 + 2 \| \mathbf {e} \| _ {2} ^ {2} / (m \lambda \eta) + \| \mathbf {e} \| _ {2} ^ {2} \\ \leq m \lambda \eta \left(L \left(\boldsymbol {\theta} ^ {\prime}\right) - L (\boldsymbol {\theta}) / 2\right) + \| \mathbf {e} \| _ {2} ^ {2} \left(1 + 2 m \lambda \eta + 2 / (m \lambda \eta)\right), \tag {D.8} \\ \end{array}
$$
where the second inequality holds due to the choice of $\eta$ , third inequality holds due to Young’s inequality, fourth inequality holds due to the fact that $\| \mathbf { f } ( \pmb \theta ) - \mathbf { y } \| _ { 2 } ^ { 2 } \leq 2 L ( \pmb \theta )$ . Now taking $\pmb \theta = \mathbf \bar { \pmb \theta } ^ { ( j ) }$ and $\pmb { \theta } ^ { \prime } = \pmb { \theta } ^ { ( 0 ) }$ , rearranging (D.8), with the fact that $\pmb { \theta } ^ { ( j + 1 ) } = \pmb { \theta } ^ { ( j ) } - \eta \nabla L ( \pmb { \theta } ^ { ( j ) } )$ , we have 
$$
\begin{array}{l} L \left(\boldsymbol {\theta} ^ {(j + 1)}\right) - L \left(\boldsymbol {\theta} ^ {(0)}\right) \\ \leq (1 - m \lambda \eta / 2) \left[ L \left(\boldsymbol {\theta} ^ {(j)}\right) - L \left(\boldsymbol {\theta} ^ {(0)}\right) \right] + m \lambda \eta / 2 L \left(\boldsymbol {\theta} ^ {(0)}\right) + \| \mathbf {e} \| _ {2} ^ {2} \left(1 + 2 m \lambda \eta + 2 / (m \lambda \eta)\right) \\ \leq (1 - m \lambda \eta / 2) \left[ L \left(\boldsymbol {\theta} ^ {(j)}\right) - L \left(\boldsymbol {\theta} ^ {(0)}\right) \right] + m \lambda \eta / 2 \cdot t + m \lambda \eta / 2 \cdot t \\ \leq (1 - m \lambda \eta / 2) \left[ L \left(\boldsymbol {\theta} ^ {(j)}\right) - L \left(\boldsymbol {\theta} ^ {(0)}\right) \right] + m \lambda \eta t, \tag {D.9} \\ \end{array}
$$
where the second inequality holds due to the fact that $L ( \pmb \theta ^ { ( 0 ) } ) = \| \mathbf f ( \pmb \theta ^ { ( 0 ) } ) - \mathbf y \| _ { 2 } ^ { 2 } / 2 = \| \mathbf y \| _ { 2 } ^ { 2 } / 2 \leq t$ , and 
$$
\left(1 + 2 m \lambda \eta + 2 / (m \lambda \eta)\right) \| \mathbf {e} \| _ {2} ^ {2} \leq 3 / (m \lambda \eta) \cdot C _ {2} \tau^ {8 / 3} L ^ {6} t m \log m \leq t m \lambda \eta / 2, \tag {D.10}
$$
where the first inequality holds due to (C.5) in Lemma C.2, the second inequality holds due to the choice of $\tau$ . Recursively applying (D.9) for $u$ times, we have 
$$
L (\pmb {\theta} ^ {(j + 1)}) - L (\pmb {\theta} ^ {(0)}) \leq \frac {m \lambda \eta t}{m \lambda \eta / 2} = 2 t,
$$
which implies that $\| \mathbf { f } ^ { ( j + 1 ) } - \mathbf { y } \| _ { 2 } \leq 2 { \sqrt { t } }$ . This completes our proof. 
# D.3. Proof of Lemma C.4
In this section we prove Lemma C.4. 
Proof of Lemma C.4. It can be verified that $\tau$ satisfies the conditions of Lemma C.2, thus Lemma C.2 holds. It is worth noting that $\widetilde { \pmb { \theta } } ^ { ( j ) }$ is the sequence generated by applying gradient descent on the following problem: 
$$
\min _ {\pmb {\theta}} \widetilde {\mathcal {L}} (\pmb {\theta}) = \frac {1}{2} \| [ \mathbf {J} ^ {(0)} ] ^ {\top} (\pmb {\theta} - \pmb {\theta} ^ {(0)}) - \mathbf {y} \| _ {2} ^ {2} + \frac {m \lambda}{2} \| \pmb {\theta} - \pmb {\theta} ^ {(0)} \| _ {2} ^ {2}.
$$
Neural Contextual Bandits with UCB-based Exploration 
Then $\lVert \pmb { \theta } ^ { ( 0 ) } - \widetilde { \pmb { \theta } } ^ { ( j ) } \rVert _ { 2 }$ can be bounded as 
$$
\begin{array}{l} \frac {m \lambda}{2} \| \boldsymbol {\theta} ^ {(0)} - \widetilde {\boldsymbol {\theta}} ^ {(j)} \| _ {2} ^ {2} \leq \frac {1}{2} \| [ \mathbf {J} ^ {(0)} ] ^ {\top} (\widetilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(0)}) - \mathbf {y} \| _ {2} ^ {2} + \frac {m \lambda}{2} \| \widetilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(0)} \| _ {2} ^ {2} \\ \leq \frac {1}{2} \| [ \mathbf {J} ^ {(0)} ] ^ {\top} (\widetilde {\boldsymbol {\theta}} ^ {(0)} - \boldsymbol {\theta} ^ {(0)}) - \mathbf {y} \| _ {2} ^ {2} + \frac {m \lambda}{2} \| \widetilde {\boldsymbol {\theta}} ^ {(0)} - \boldsymbol {\theta} ^ {(0)} \| _ {2} ^ {2} \\ \leq t / 2, \\ \end{array}
$$
where the first inequality holds trivially, the second inequality holds due to the monotonic decreasing property brought by gradient descent, the third inequality holds due to (C.6) in Lemma C.2. It is easy to verify that $\widetilde { \mathcal { L } }$ is a $m \lambda$ -strongly convex and function and $C _ { 1 } ( t m L + m \lambda )$ -smooth function, since 
$$
\nabla^ {2} \widetilde {\mathcal {L}} \preceq \big (\left\| \mathbf {J} ^ {(0)} \right\| _ {2} ^ {2} + m \lambda \big) \mathbf {I} \preceq C _ {1} (t m L + m \lambda),
$$
where the first inequality holds due to the definition of $\widetilde { \mathcal { L } }$ , the second inequality holds due to (C.3) in Lemma C.2. Since we choose $\eta \leq C _ { 2 } ( t m L + m \lambda ) ^ { - 1 }$ for some small enough $C _ { 2 } > 0$ , then by standard results of gradient descent on ridge linear regression, $\widetilde { \pmb { \theta } } ^ { ( j ) }$ converges to ${ \pmb \theta } ^ { ( 0 ) } + ( \bar { \bf Z } ) ^ { - 1 } \bar { \bf b } / \sqrt { m }$ with the convergence rate 
$$
\begin{array}{l} \left\| \tilde {\boldsymbol {\theta}} ^ {(j)} - \boldsymbol {\theta} ^ {(0)} - \bar {\mathbf {Z}} ^ {- 1} \mathbf {b} / \sqrt {m} \right\| _ {2} ^ {2} \leq (1 - \eta m \lambda) ^ {j} \cdot \frac {2}{m \lambda} \left(\mathcal {L} \left(\boldsymbol {\theta} ^ {(0)}\right) - \mathcal {L} \left(\boldsymbol {\theta} ^ {(0)} + \bar {\mathbf {Z}} ^ {- 1} \mathbf {b} / \sqrt {m}\right)\right) \\ \leq \frac {2 (1 - \eta m \lambda) ^ {j}}{m \lambda} \mathcal {L} (\boldsymbol {\theta} ^ {(0)}) \\ = \frac {2 (1 - \eta m \lambda) ^ {j}}{m \lambda} \cdot \frac {\| \mathbf {y} \| _ {2} ^ {2}}{2} \\ \leq (1 - \eta m \lambda) ^ {j} t, \\ \end{array}
$$
where the first inequality holds due to the convergence result for gradient descent and the fact that ${ \pmb { \theta } } ^ { ( 0 ) } + ( \bar { \bf Z } ) ^ { - 1 } \bar { \bf b } / \sqrt { m }$ is the minimal solution to $\mathcal { L }$ , the second inequality holds since $\mathcal { L } \geq 0$ , the last inequality holds due to Lemma C.2. 
# E. A Variant of NeuralUCB
In this section, we present a variant of NeuralUCB called NeuralUCB0. Compared with Algorithm 1, The main differences between NeuralUCB and NeuralUCB0 are as follows: NeuralUCB uses gradient descent to train a deep neural network to learn the reward function $h ( \mathbf { x } )$ based on observed contexts and rewards. In contrast, Neura $\mathrm { U C B _ { 0 } }$ uses matrix inversions to obtain parameters in closed forms. At each round, NeuralUCB uses the current DNN parameters $( \pmb \theta _ { t } )$ to compute an upper confidence bound. In contrast, NeuralUCB0 computes the UCB using the initial parameters $\mathbf { \Omega } ( \pmb { \theta } _ { 0 } )$ . 
Neural Contextual Bandits with UCB-based Exploration 
# Algorithm 3 NeuralUCB0
1: Input: number of rounds $T$ , regularization parameter $\lambda$ , exploration parameter $\nu$ , confidence parameter $\delta$ , norm parameter $S$ , network width $m$ , network depth $L$ 
2: Initialization: Generate each entry of $\mathbf { W } _ { l }$ independently from $N ( 0 , 2 / m )$ for $1 \leq l \leq L - 1$ , and each entry of $\mathbf { W } _ { L }$ independently from $N ( 0 , 1 / m )$ . Define $\phi ( \mathbf { x } ) = \mathbf { g } ( \mathbf { x } ; \pmb { \theta } _ { 0 } ) / \sqrt { m }$ , where $\pmb { \theta } _ { 0 } = [ \mathrm { v e c } ( \mathbf { W } _ { 1 } ) ^ { \top } , \dots , \mathrm { v e c } ( \mathbf { W } _ { L } ) ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { p }$ 
3: $\mathbf { Z } _ { 0 } = \lambda \mathbf { I }$ , $\mathbf { b } _ { 0 } = \mathbf { 0 }$ 
4: for $t = 1 , \dots , T$ do 
5: Observe $\{ \mathbf { x } _ { t , a } \} _ { a = 1 } ^ { K }$ and compute 
$$
\left(a _ {t}, \widetilde {\boldsymbol {\theta}} _ {t, a _ {t}}\right) = \operatorname * {a r g m a x} _ {a \in [ K ], \boldsymbol {\theta} \in \mathcal {C} _ {t - 1}} \left\langle \phi \left(\mathbf {x} _ {t, a}\right), \boldsymbol {\theta} - \boldsymbol {\theta} _ {0} \right\rangle \tag {E.1}
$$
6: Play $a _ { t }$ and receive reward $r _ { t , a _ { t } }$ 
7: Compute 
$$
\mathbf {Z} _ {t} = \mathbf {Z} _ {t - 1} + \phi (\mathbf {x} _ {t, a _ {t}}) \phi (\mathbf {x} _ {t, a _ {t}}) ^ {\top} \in \mathbb {R} ^ {p \times p}, \mathbf {b} _ {t} = \mathbf {b} _ {t - 1} + r _ {t, a _ {t}} \phi (\mathbf {x} _ {t, a _ {t}}) \in \mathbb {R} ^ {p}
$$
8: Compute $\pmb \theta _ { t } = \mathbf Z _ { t } ^ { - 1 } \mathbf b _ { t } + \pmb \theta _ { 0 } \in \mathbb { R } ^ { p }$ 
9: Construct $\mathcal { C } _ { t }$ as 
$$
\mathcal {C} _ {t} = \left\{\boldsymbol {\theta}: \| \boldsymbol {\theta} _ {t} - \boldsymbol {\theta} \| _ {\mathbf {Z} _ {t}} \leq \gamma_ {t} \right\}, \quad \text {w h e r e} \quad \gamma_ {t} = \nu \sqrt {\log \frac {\det  \mathbf {Z} _ {t}}{\det  \lambda \mathbf {I}} - 2 \log \delta} + \sqrt {\lambda} S \tag {E.2}
$$
10: end for 