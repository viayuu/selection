# Surrogate-Assisted Evolutionary Algorithm With Model and Infill Criterion Auto-Configuration
Lindong Xie , Genghui Li, Zhenkun Wang , Member, IEEE, Laizhong Cui , Senior Member, IEEE, and Maoguo Gong , Senior Member, IEEE

1114
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024

## Abstract
Surrogate-assisted evolutionary algorithms (SAEAs) have proven to be effective in solving computationally expensive optimization problems (EOPs). However, the performance of SAEAs heavily relies on the surrogate model and infill criterion used. To improve the generalization of SAEAs and enable them to solve a wide range of EOPs, this article proposes an SAEA called AutoSAEA, which features model and infill criterion auto-configuration. Specifically, AutoSAEA formulates model and infill criterion selection as a two-level multiarmed bandit problem (TL-MAB). The first and second levels cooperate in selecting the surrogate model and infill criterion, respectively. A two-level reward (TL-R) measures the value of the surrogate model and infill criterion, while a two-level upper confidence bound (TLUCB) selects the model and infill criterion in an online manner. Numerous experiments validate the superiority of AutoSAEA over some state-of-the-art SAEAs on complex benchmark problems and a real-world oil reservoir production optimization problem.

## Index Terms
Auto algorithm design, expensive optimization, surrogate-assisted evolutionary algorithm (SAEA), two-level multiarmed bandit (TL-MAB).

Manuscript received 28 January 2023; revised 7 May 2023; accepted 27 June 2023. Date of publication 3 July 2023; date of current version 1 August 2024. This work was supported in part by the National Natural Science Foundation of China under Grant 62206120, Grant 62106096, and Grant 62036006; and in part by the Shenzhen Technology Plan under Grant JCYJ20220530113013031. (Lindong Xie and Genghui Li contributed equally to this work.) (Corresponding author: Zhenkun Wang.)

Lindong Xie and Genghui Li are with the School of System Design and Intelligent Manufacturing, Southern University of Science and Technology, Shenzhen 518055, China (e-mail: 12132679@mail.sustech.edu.cn; genghuili2-c@my.cityu.edu.hk).

Zhenkun Wang is with the School of System Design and Intelligent Manufacturing and the Department of Computer Science and Engineering, Southern University of Science and Technology, Shenzhen 518055, China (e-mail: wangzhenkun90@gmail.com).

Laizhong Cui is with the College of Computer Science and Software Engineering and the Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), Shenzhen University, Shenzhen 518060, China (e-mail: cuilz@szu.edu.cn).

Maoguo Gong is with the Key Laboratory of Collaborative Intelligence Systems, Ministry of Education, Xidian University, Xi’an 710071, China (e-mail: gong@ieee.org).

This article has supplementary material provided by the authors and color versions of one or more figures available at https://doi.org/10.1109/ TEVC.2023.3291614.

Digital Object Identifier 10.1109/TEVC.2023.3291614
1089-778X c⃝2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.

Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply.

XIE et al.: SAEA
1115

## I. INTRODUCTION
EXPENSIVE optimization problems (EOPs) are prevalent in many engineering designs and applications (e.g., aerodynamic structures design [1], automobile crash analysis [2], and space tethered-net system design [3]). Unfortunately, solving such problems requires conducting computationally expensive computer simulations or costly physics experiments since the analytical expressions of the problem are unavailable. Generally, the EOP can be expressed as follows:
\[min \ f(x)\]
\[s.t. \ x_{l} \leq x \leq x_{u} \tag{1}\]
where \(x=(x_{1}, ..., x_{D})^{\top}\) is the decision vector of D variables, \(x_{l}=(x_{l, 1}, ..., x_{l, D})^{\top}\) and \(x_{u}=(x_{u, 1}, ..., x_{u, D})^{\top}\) are the lower and upper bounds of the search space, respectively, and \(f(x)\) denotes a scalar objective function. We assume that the analytical expression of \(f(x)\) is unavailable and the calculation of \(f(x)\) is costly.

Evolutionary algorithms have been shown to be popular and effective for solving black-box optimization problems [4], [5]. However, they often perform poorly on EOPs due to the large number of function evaluations (FEs) required in their optimization process. This limitation becomes more significant in the case of EOPs [6]. To address this issue, surrogate-assisted evolutionary algorithms (SAEAs) have been proposed. In SAEAs, the optimization is mainly driven by computationally cheap surrogate models. The popular surrogate models used in SAEAs include regression models, such as radial basis function (RBF) [7], Gaussian process model (GP) [8], and polynomial response surface (PRS) [9], and classification models, such as k -nearest neighbor (KNN) [10] and support vector machine model (SVM) [11]. Based on the adopted surrogate model, existing SAEAs can be roughly grouped into three categories: 1) single-model SAEAs [12], [13], [14], [15]; 2) multiplemodel SAEAs [16], [17], [18], [19], [20], [21]; and 3) adaptive-model SAEAs [22], [23], [24], [25].

Single-model SAEAs use a fixed surrogate model throughout the optimization process, typically combining a fixed infill criterion, such as GP with expected improvement (EI) [26] and lower confidence bound (LCB) [27], RBF with local search [28], or KNN with prescreening [29], to select candidate solutions for FEs. However, due to the limited training samples available and the inability of a single model to effectively fit various problem landscapes, single-model SAEAs often struggle to solve different EOPs effectively [18], [30].

In order to enhance the robustness and scalability of singlemodel SAEAs, multiple-model SAEAs have been developed, including hierarchical-model SAEAs and ensemble-model SAEAs [31]. Hierarchical-model SAEAs commonly combine a global model and a local model [16], [18], [32], [33], [34]. The global model approximates the fitness landscape to search for promising regions, and the local model is used to conduct a refined search in the found local promising regions. Ensemble-model SAEAs typically train several different base surrogate models to cooperatively implement a more precise prediction [9], [19], [20], [21]. Generally, multiple-model SAEAs outperform single-model ones in most cases [15]. However, the computational cost of multiplemodel SAEAs is always higher compared to the single-model SAEAs [35], [36]. Additionally, some base models may not fit the problem landscape well, so their efficiency may still be low.

To leverage well-established models effectively and reduce computational complexity, adaptive-model SAEAs have been proposed that automatically select the suitable model (or infill criterion) in an online manner [22], [24], [25]. However, the performance of such algorithms dramatically depends on the design of the adaptive selection strategy. It is well known that the surrogate model is tightly coupled with the infill criterion in SAEAs, and they cooperatively affect the performance of SAEAs. A model with different infill criteria can realize different balances of exploiting better solutions and exploring unexplored search regions. Naturally, a better balance between exploration and exploitation may be achieved by cooperatively considering the model and the infill criterion in a hierarchical coupled way. Unfortunately, existing adaptive-model SAEAs are either only for model or infill criterion selection but do not study adaptive selection for both cooperatively. Therefore, to improve the performance of SAEAs by filling this gap, this article proposes an SAEA (called AutoSAEA) with the model and infill criterion autoconfiguration by using the hierarchical multiarmed bandit (MAB) method.

The contributions of this article are summarized as follows.
1) Formulating the surrogate model and infill criterion cooperative selection as a two-level MAB problem (TL-MAB) [37], [38], [39].
2) Designing a two-level reward (TL-R) to measure the utility of the surrogate model and infill criterion in a cooperative manner.
3) Adopting a two-level upper confidence bound (TLUCB) to select the surrogate model and infill criterion cooperatively in an online manner.
4) Proposing an SAEA with a surrogate model and infill criterion auto-configuration, and verifying its advantages by comparing it with some state-of-the-art SAEAs on two sets of complex benchmark problems and a realworld problem.

The remainder of this article is organized as follows. Section II reviews the related work. Section III introduces the background knowledge involved in the proposed algorithm. Section IV describes the proposed algorithm in detail. Section V presents the numerical experiments conducted to validate the effectiveness of the proposed algorithm. Finally, Section VI summarizes this article and discusses future work directions.

Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply.

1116
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024

## II. RELATED WORK
1) Single-Model SAEAs: Single-model SAEAs can either train a global model using all evaluated solutions or a local model with some good ones to predict the quality of new solutions. For example, the Bayesian optimization framework is a classic and popular single-model method [40], [41], [42]. It uses the GP model to build a surrogate for the objective and quantify the uncertainty in the surrogate. Then, it optimizes an acquisition function (e.g., EI [40] and LCB [43]) defined from the surrogate to decide the new solution for evaluation. GPEME [13] first constructs a local GP model in the lowerdimensional space, followed by the application of prescreening with LCB to determine which offspring can be selected for FE. CPS-MOEA [44] employs a local KNN classifier to filter out potentially nondominated solutions for real evaluation. SACOSO [45] evaluates the new position of the particle that has the minimum fitness value predicted by a global RBF model. In addition, SHPSO [46] and SAMSO [15] construct a local RBF model and a global RBF model, respectively, to select particles with predicted fitness values that are better than their personal best ones for FEs. MGP-SLPSO [14] formulates the approximated fitness and its corresponding uncertainty provided by a global GP model as a bi-objective problem and applies a nondominated sorting method to select the offspring for FEs. Furthermore, CA-LLSO [47] trains a local gradient boosting classifier to predict the level of the offspring produced by the level-based learning swarm optimizer.

2) Multiple-Model SAEAs: Multiple-model SAEAs mainly combine a global and a local model or train multiple models simultaneously to balance exploration and exploitation. For example, CAL-SAPSO [19] identifies the best and most uncertain solutions for FEs using a global ensemble surrogate model. Additionally, the optimum of the local ensemble model found by the local search is also used for expensive evaluation. HeE-MOEA [35] utilizes an SVM and two RBF models to build an ensemble model, which is combined with LCB and EI criteria to screen the offspring for expensive evaluations. ESAO [32] employs a global RBF to screen the offspring with the minimum predicted fitness value for FE. Besides, the optimum of the local RBF model found by the differential evolution (DE) is also evaluated. GSGA [33] adopts a local RBF-assisted trust-region method and a global GP-assisted genetic algorithm for exploitation and exploration, respectively. GL-SADE [18] trains a global RBF model and a local GP model to select offspring for FEs. Moreover, when the local GP model finds the current best solution, DE is applied further to search for its optimal solution. ESCO [21] constructs multiple RBF models on various low-dimensional sample sets, and then selects the models with superior performance to compose an ensemble surrogate.

3) Adaptive-Model SAEAs: Adaptive-model SAEAs always design an adaptive selection strategy to automatically choose a model or an infill criterion in an online manner. For example, GP-Hedge [22] adaptively selects an appropriate acquisition function from a portfolio of well-established ones, such as the probability of improvement, EI, and UCB. This selection is based on an online MAB strategy. Specifically, the optimal solutions of all acquisition functions form a candidate solution pool, from which solutions are selected for FEs based on the cumulative rewards of their respective acquisition functions. While GP-Hedge only adaptively selects the acquisition functions, it fixes the surrogate model. Moreover, its computational efficiency is low since it needs to optimize multiple acquisition functions in each generation. ASMEA [30] first constructs a base model pool using GP, RBF, PRS, and linear Shepard interpolation. It then uses the prediction residual error sum of squares (PRESS) and root mean square error (RMSE) to select several meta-models from the base model pool for constructing five ensemble models, which are then included in the base model pool. Finally, a promising surrogate model is adaptively selected from the base model pool based on its minimum RMSE. While ASMEA only adaptively selects the models, it fixes the prescreening as the infill criterion. Moreover, its computational efficiency is low since it needs to conduct cross-validation for each base model. ESA [25] first constructs a pool of four different infill criteria, including DE evolutionary screening, surrogate-assisted local search, full-crossover strategy, and surrogate-assisted trust-region local search. Among them, the DE evolutionary screening strategy prefers exploration, the surrogate-assisted local search and trust-region local search strategies use different local search methods to favor exploitation, and the full-crossover strategy integrates good genes from historical solutions. Moreover, Q -learning is used to adjust the selection probability of each infill criterion through feedback information received during the optimization process. However, ESA fixes the RBF as the surrogate model.

Additionally, some adaptive-model SAEAs have been proposed to deal with expensive multiobjective optimization problems. For instance, KTA2 [23] divides the optimization process into three states (i.e., convergence-demand, diversitydemand, and uncertainty-demand states) and uses the distances from solutions to the estimated ideal point and the pure diversity indicator to estimate the state. Moreover, the infill criterion is adaptively chosen based on the estimated optimization state, which guides the solution selection for FEs by taking into account the requirements on convergence, diversity, and model uncertainty separately. RVMM [48] uses the GP as the surrogate model and develops an adaptive model management strategy to adaptively choose the convergence-related criterion and diversity-related criterion. The adaptive model management strategy is assisted by two sets of reference vectors. One set of adaptive reference vectors focuses on convergence, while the other set of fixed reference vectors concentrates on diversity. The GP and RBF models are the most popular surrogate models due to their high fidelity and simplicity. Generally, the RBF model is computationally more efficient than the GP model. However, the GP model can provide uncertain information about its predictions. Therefore, to take advantage of the GP and RBF models in dealing with different problems, IBEAMS [24] designs an acceptable reliability tolerance criterion to adaptively determine whether to use the GP model or the RBF model in environmental selection. Specifically, when the uncertainty of the GP model exceeds the acceptable error, the RBF model is used. Otherwise, the GP models are used.

Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply.

XIE et al.: SAEA
1117

## III. BACKGROUND
### A. Surrogate Models
1) GP Model: Given a training data set \(\{x_{i}, f(x_{i})\}_{i=1}^{N}\) , GP predicts the output \(f_{GP}(x)\) of a solution x in the way of: \(f_{GP}(x)=\mu(x)+\epsilon(x)\) , where \(\mu(x)\) represents a global trend of the training data, and \(\epsilon(x) \sim N(0, \sigma^{2}(x))\) is a normal distribution. \(\mu(x)\) and \(\sigma^{2}(x)\) are estimated as follows [40]:
\[\mu(x)=k(x)^{\top} K^{-1} F \tag{2}\]
\[\sigma^{2}(x)=\kappa(x)-k(x)^{\top} K^{-1} k(x) \tag{3}\]
where \(k(x)=[C(x, x_{1}), ..., C(x, x_{N})]^{\top}\) ; K is an \(N ×N\) matrix and \(K_{i, j}=C(x_{i}, x_{j})\) ; \(\kappa(x)=C(x, x)\) ,and \(F=[f(x_{1}), ..., f(x_{N})]^{\top}\) . The covariance function \(C(\cdot, \cdot)\) can be commonly calculated as follows:
\[C\left(x_{i}, x_{j}\right)=exp \left(-\sum_{d=1}^{D} \theta_{d}\left|x_{i, d}-x_{j, d}\right|^{2}\right) \tag{4}\]
where the hyper-parameters θ can be obtained by maximizing the likelihood function [33].

Finally, for an unknown solution x , its predicted fitness value \(\hat{f}_{GP}(x)\) and uncertainty \(\hat{s}(x)\) are given in the following:
\[\hat{f}_{GP}(x)=\mu(x)+k(x)^{\top} K^{-1}(F-I \mu(x)) \tag{5}\]
\[\hat{s}^{2}(x)=\sigma^{2}(x)\left[1-k(x)^{\top} K^{-1} k(x)+\frac{\left(1-I^{\top} K^{-1} k(x)\right)^{2}}{I^{\top} K^{-1} I}\right] \tag{6}\]
where I is an \(N ×1\) unit vector, and \(\hat{s}(x)=\sqrt{\hat{s}^{2}(x)}\) is the RMSE.

2) RBF Model: The RBF model uses a linear combination of basis functions to approximate the fitness landscape. For the given training data set \(\{x_{i}, f(x_{i})\}_{i=1}^{N}\) , the form of the RBF model can be formulated as follows [49]:
\[\hat{f}_{RBF}(x)=\sum_{i=1}^{N} w_{i} \varphi\left(\left\| x-x_{i}\right\| \right)+b_{0}+\sum_{j=1}^{D} b_{j} x_{j} \tag{7}\]
where \(w=(w_{1}, w_{2}, ..., w_{N})^{\top}\) is the weight vector of the basis function, and this article adopts the cubic function \(\varphi(\left\|x-x_{i}\right\|)=(\left\|x-x_{i}\right\|)^{3}\) , where \(\|\cdot\|\) denotes the Euclidean distance. \(b=(b_{0}, b_{1}, ..., b_{D})^{\top}\) is the coefficient of the first-order polynomial.

The parameters w and b in (7) can be obtained as follows:
\[\left[\begin{array}{l}w \\ b\end{array}\right]=\left[\begin{array}{cc}\Phi & P \\ P^{\top} & 0_{(D+1) \times(D+1)}\end{array}\right]^{\dagger}\left[\begin{array}{c}F \\ 0_{(D+1) × 1}\end{array}\right] \tag{8}\]
where \(\Phi_{i, j}=\varphi(\left\|x_{i}-x_{j}\right\|_{2})\) , \(i, j=1,2, ..., N\) : \(P=[P_{1}, ..., P_{N}]^{\top}\) and \(P_{i}=[1, x_{i, 1}, ..., x_{i, D}]^{\top} ; 0_{(D+1) \times(D+1)}\) is a zero matrix of \((D+1) \times(D+1)\) ,and \(\dagger\) is the generalized inverse, and \(F=[f(x_{1}), ..., f(x_{N})]^{\top}\) .

3) PRS Model: The commonly used second-order polynomial is defined as follows [9]:
\[\hat{f}_{PRS}(x)=\beta_{0}+\sum_{i=1}^{D} \beta_{i} x_{i}+\sum_{i=1}^{D} \sum_{j \geq i}^{D} \beta_{i j} x_{i} x_{j} \tag{9}\]
where coefficients \(\beta_{0}\) , \(\beta_{i}\) ,and \(\beta_{i j}\) are regression parameters for the intercept, the linear term, and the quadratic term, respectively.

For the given training data set \(\{x_{i}, f(x_{i})\}_{i=1}^{N}\) , the unknown coefficients of the above polynomial model \(\beta=(\beta_{0}, ...,\beta_{D}, \beta_{1,1}, ..., \beta_{1, D}, \beta_{2,3}, ..., \beta_{2, D}, ..., \beta_{D-1, D}, \beta_{D, D})^{\top}\) can be obtained by the least square method as follows [50]:
\[\beta=\left(P^{\top} P\right)^{-1} P^{\top} F \tag{10}\]
where \(P=[P_{1}, ..., P_{N}]^{\top}\) , \(P_{i}=[1, x_{i, 1}, ..., x_{i, D}]^{\top}\) ,and \(F=[f(x_{1}), ..., f(x_{N})]^{\top}\) .

4) KNN Model: KNN is a popular classifier [29], [44]. Based on a training data set \(\{x_{i}, f(x_{i}), y(x_{i})\}_{i=1}^{N}\) , where \(f(x_{i})\) and \(y(x_{i})\) denote the objective function value and class/level of the solution \(x_{i}\) , respectively. For a new solution x , its class/level is predicted as follows:
\[y_{KNN}(x)=mode\left(\left\{y\left(x_{1}\right), ..., y\left(x_{K}\right)\right\}\right) \tag{11}\]
where \(x_{1}, ..., x_{K}\) are K nearest neighbors of x , and mode() denotes the mode of a set. In this article, K is set to 1 [29], [44], which means that the new solution x is assigned the same class as its nearest neighbor.

In this article, all surrogate models are trained based on the current population P in each iteration.

Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply.

1118
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024

### B. Infill Criteria
Since the surrogate models are tightly coupled with the infill criteria in SAEAs, the corresponding infill criteria for the surrogate models mentioned in the previous section are introduced as follows.

1) Infill Criteria for the GP Model: The LCB and EI criteria are commonly combined with GP for determining new solutions for FEs, which are expressed sequentially as follows [13], [33]:
\[x^{*}=\underset{x \in \mathcal{X}}{arg min }\left(\hat{f}_{GP}(x)-w \hat{s}(x)\right) \tag{12}\]
where w is set to 2 in this article [13], [27]. x denotes a set of unknown solutions.

\[x^{*}=\underset{x \in \mathcal{X}}{arg max }\left(\left(f_{min }-\hat{f}_{GP}(x)\right) \Phi(Z)+\hat{s}(x) \varphi(Z)\right) \tag{13}\]
where \(Z=([f_{min }-\hat{f}_{GP}(x)] /[\hat{s}(x)])\) , \(f_{min }\) is the current minimum objective function value, and \(\Phi(\cdot)\) and \(\varphi(\cdot)\) are the density and cumulative distribution functions of standard normal distribution, respectively.

This article adopts the DE mutation and crossover operators to generate the new solution set \(X=\{o_{1}, ..., o_{N}\}\) , where N is the population size. To be specific, assume that the current population is P , \(o_{i}=(o_{i, 1}, ..., o_{i, D})\) is generated as follows:
\[v_{i}=x_{i}+F× (x_{b}-x_{i})+F× (x_{r1}-x_{r2}) \tag{14}\]
\[o_{i, j}=\left\{\begin{array}{l}v_{i, j}, if \ rand_{i, j} \leq C R \quad or \ j=j_{rand } \\ x_{i, j}, otherwise \end{array}\right. \tag{15}\]
where \(x_{i}\) is the ith solution in P , \(x_{r 1}\) and \(x_{r 2}\) are randomly selected from the current population P , \(x_{b}\) is the best solution of P : F is the scale factor, and it is set to 0.5 in this article. \(rand_{i,j}\) and \(j_{rand }\) are randomly selected from [0, 1] and \(\{1, ..., D\}\) , respectively, and \(C R \in[0,1]\) denotes the crossover rate, which is set to 0.9 in this article.

Note that if \(o_{i, j}\) violates the boundary constraints, it will be repaired as follows:
\[o_{i, j}=x_{l, j}+rand \times\left(x_{u, j}-x_{l, j}\right) . \quad(16)\]

2) Infill Criteria for the RBF and PRS Models: For all regression models (e.g., RBF and PRS), prescreening and local search are widely used in SAEAs, which are formulated as follows [28], [51]:
\[x^{*}=\underset{x \in \mathcal{X}}{arg min } \hat{f}_{RBF / PRS}(x) \tag{17}\]
where x and \(\hat{f}_{RBF / PRS}(x)\) denote the unknown solution and its predicted fitness value by RBF/PRS, respectively. It should be noted that if x is a finite set of solutions, (17) belongs to prescreening, and it denotes a local search if x is a subspace of the search space.

In this article, the above DE operator is used to generate x for the prescreening. For the local search, x is defined as follows:
\[\mathcal{X}=[l b, u b] \quad(18)\]
where \(l b=(l_{1}, ..., l_{D})^{\top}\) and \(u b=(u_{1}, ..., u_{D})^{\top}\) , \(l_{j}=min \{x_{i, j}, x_{i} \in P\}\) , and \(u_{j}=max \{x_{i, j}, x_{i} \in P\}\) , \(j=1, ..., D\) . Moreover, DE is used to solve (17) in the local search in this article. Note that in this article, the DE local search is conducted with a population size of N and a maximum number of 100D + 1000 generations [25];

3) Infill Criteria for the KNN Model: Classification models are often used to predict the level of the unknown solutions [52]. To this end, the current population is evenly divided into L levels \(P^{1}, ..., P^{L}\) based on the fitness, and the best \((N / L)\) solutions belong to the first level \(P^{1}\) . Two infill criteria (e.g., L1-exploitation and L1-exploration) have been designed for the classifier model. They are defined as follows [47].

1) \(L1\) -Exploitation: The L1-exploitation criterion is defined as
\[x^{*}=arg min _{x \in \mathcal{X}^{1}} max \left\{\| x-p\| , p \in \mathcal{P}^{1}\right\} . \tag{19}\]

2) \(L1\) -Exploration: The LI-exploration is defined as
\[x^{*}=arg max _{x \in \mathcal{X}^{1}} min \left\{\| x-p\| , p \in \mathcal{P}^{1}\right\} \tag{20}\]
where \(x^{1}\) includes the solution in new generated solution set x whose predicted level by the KNN model is L1. In this article, x is also generated by the above-described DE operators, and the number of levels is set to 5.

For the models and infill criteria discussed above, we have the following remarks. Over the past two decades, a variety of regression and classification models have been employed to aid evolutionary algorithms in solving complex optimization problems. As far as our knowledge goes, GP, RBF, PRS, and KNN are the most effective and commonly used models in state-of-the-art SAEAs [9], [19], [44], each with unique advantages in handling diverse problems. In this article, our goal is to develop an SAEA with auto-configuration of models and infill criteria. Therefore, we have chosen these well-known models for their established strengths. However, it should be noted that other models and infill criteria can also be easily incorporated into our proposed algorithm. Moreover, regarding the parameters involved in the infill criteria, we have tested several representative values for each parameter in our preliminary experiments. The experimental results show that they do not have a significant influence. Therefore, we use commonly accepted settings for these parameters in this article.

Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply.

XIE et al.: SAEA
1119

### C. Hierarchical Multiarmed Bandit
MAB is a powerful tool for online learning [37]. In the standard MAB problem, the player must pick an arm \(a_{t}\) from a set of arms A at each time slot t and then obtain a reward \(r_{a_{i}}\) , which is produced from a distribution that is unknown to the player. The performance of an algorithm in MAB is typically measured by the regret, and the player’s goal is to minimize the expected cumulative regret over a sequence of T time slots. The expected cumulative regret can be expressed as follows [37]:
\[\mathbb{E}\left[R_{T}\right]=\sum_{t=1}^{T} r_{a_{t}^{*}}-r_{a_{t}} \tag{21}\]
where \(a_{t}^{*}\) denotes the unknown best arm at the t -th time slot.

Hierarchical MAB (HMAB) is an extension of the standard MAB. Two commonly used hierarchical structures have been developed [39].

1) Disjoint Clustering Structure: The K arms are classified into a set of clusters, and each arm \(a \in A\) belongs to only one cluster. The player first picks a cluster and then chooses an arm in the selected cluster. Finally, a reward is produced by the chosen arm. An illustrative example of disjoint clustering is shown in Fig. 1(a).

2) Hierarchical Clustering Structure: The arms can be divided into multiple levels. The arms on the same level are different from each other, and the different highlevel arms can be associated with the same low-level arms. The player first selects an arm from the high-level and then chooses an arm from its associated low-level. Finally, a reward is returned by all chosen arms cooperatively. An illustrative example of hierarchical clustering is shown in Fig. 1(b).

Fig. 1. Illustrative examples of HMAB clustering structures. (a) Disjoint clustering structure. (b) Hierarchical clustering structure.

Many theories and experiments have demonstrated that HMAB has significant advantages over standard MAB [37], [38], [39], [53], [54], [55]. In general, its benefits can be summarized into three aspects: 1) the regret bound of HMAB is lower than that of standard MAB; 2) HMAB can reduce the size of the arm space, especially when the arm space is large; and 3) HMAB can greatly mitigate the chances of selecting suboptimal arms, leading to improved performance in solving EOPs. Therefore, we propose integrating HMAB into SAEAs to enable the adaptive and cooperative selection of surrogate models and infill criteria. In this article, we consider the hierarchical clustering structure of HMAB, and its details will be explained in the following section.

## IV. AUTOSAEA
### A. Algorithm Structure
The framework of the proposed AutoSAEA is shown in Algorithm 1. Its input includes the population size N (line 2), the maximum number of FEs MaxFEs (line 3), a set of highlevel arms \(\mathcal{A}^H\) (line 4), a set of low-level arms \(\mathcal{A}^L\) (line 5), and the parameter α (line 6). The high-level and low-level arms denote the surrogate model and the infill criterion, respectively. In this article, we set \(\mathcal{A}^H=\{a_{1}^{H}=GP, a_{2}^{H}=RBF, a_{3}^{H}=PRS, a_{4}^{H}=KNN\}\) and \(\mathcal{A}^L=\{a_{1}^{L}=LCB, a_{2}^{L}=EI, a_{3}^{L}=prescreening, a_{4}^{L}=local \ search, a_{5}^{L}=prescreening, a_{6}^{L}=local \ search, a_{7}^{L}=L1-exploitation, a_{8}^{L}= L1-exploration\}\)
Note that \(a_{3}^{L}\) and \(a_{4}^{L}\) are associated with RBF, and \(a_{5}^{L}\) and \(a_{6}^{L}\) are associated with PRS.

```
Algorithm 1: AutoSAEA(N, MaxFEs, 𝒜ᴴ, 𝒜ᴸ, α).
1: Input:
2:   Population size: N
3:   Maximum number of FEs: MaxFEs
4:   High-level arm set: 𝒜ᴴ = {aᴴ₁, . . . , aᴴ₄}
5:   Low-level arm set: 𝒜ᴸ = {aᴸ₁, . . . , aᴸ₈}
6:   Parameters: α
7: Initialization:
8:   Use LHS to sample N solutions in the search space
9:   Evaluate and save them into the database 𝒟
10:  Set Qᴴₐ(1), a ∈ 𝒜ᴴ and Qᴸₐ, a ∈ 𝒜ᴸ to 0
11:  Set Tᴴₐ(1), a ∈ 𝒜ᴴ and Tᴸₐ, a ∈ 𝒜ᴸ to 0
12:  Set CA based on 𝒜ᴴ and 𝒜ᴸ
13:  Set FEs = N and t = 1
14: while FEs < MaxFEs do
15:   Select N best solutions from 𝒟 as the population 𝒫
16:   if t ≤ |CA| then
17:     Select the t-th combinatorial arm in CA
18:     Set aᴴₜ and aᴸₜ as the first and second arm in the selected combinatorial arm, respectively
19:   else
20:     Choose aᴴₜ and aᴸₜ by TL-UCB
21:   end if
22:   Obtain xₜ by aᴴₜ and aᴸₜ cooperatively
23:   Evaluate xₜ
24:   𝒟 = 𝒟 ∪ xₜ
25:   FEs = FEs + 1
26:   Tᴴ_{aᴴₜ}(t + 1) = Tᴴ_{aᴴₜ}(t) + 1, Tᴸ_{aᴸₜ}(t + 1) = Tᴸ_{aᴸₜ}(t) + 1
27:   Update the value of aᴴₜ and aᴸₜ by TL-R
28:   t = t + 1
29: end while
30: Output: The best solution in 𝒟
```

In the initialization (lines 7–13), an initial population with N solutions is generated by the Latin hypercube sampling (LHS) method (line 8). These solutions are evaluated directly and saved into a database D (line 9). The value for each arm and the number of selections for each arm is set to 0 (lines 10 and 11). Moreover, to make each arm be selected at least once, a legal combinatorial arm set is set as \(CA = \{(GP, LCB), (GP, EI), (RBF, prescreening), (RBF, local \ search), (PRS, prescreening), (PRS, local \ search), (KNN, L1-exploitation), (KNN, L1-exploration)\}\) (line 12). Finally, the current number of FEs and the iteration number are set to N and 1, respectively (line 13).

In the optimization process (lines 14–29), it first selects the N best solutions from D as the population P (line 15). If the number of iterations t is less than the cardinality of the combinatorial arm set CA (line 16), the t -th combinatorial arm in CA is chosen (line 17). Then, the high-level arm \(a_{t}^{H}\) and low-level arm \(a_{t}^{L}\) are set to the first and second arms in the selected combinatorial arm (line 18), respectively. Otherwise, they are determined by the TL-UCB method (line 20). When the high-level arm \(a_{t}^{H}\) and low-level arm \(a_{t}^{L}\) are selected, a new solution \(x_{t}\) is determined by them cooperatively (line 22). To be specific:
1) when \(a_{t}^{H}\) represents the GP model, an offspring population with N solutions is first generated by DE operators. Then, the offspring solution with the best LCB value (if \(a_{t}^{L}=LCB\) ) or best EI value (if \(a_{t}^{L}=EI\) ) is chosen as \(x_{t}\) ;
2) when \(a_{t}^{H}\) represents the RBF model or PRS model, if \(a_{t}^{L}\) = prescreening, an offspring population with N solutions is first generated by DE operators. Then, the offspring solution with the best \(\hat{f}_{RBF}(x)\) value (if \(a_{t}^{H}=RBF\) ) or \(\hat{f}_{PRS}(x)\) value (if \(a_{t}^{H}=PRS\) ) is chosen as \(x_{t}\) .If \(a_{t}^{L}\) = local search, DE is used to optimize \(\hat{f}_{RBF}(x)\) (if \(a_{t}^{H}=RBF\) ) or \(\hat{f}_{PRS}(x)\) (if \(a_{t}^{H}=PRS\) ) to get \(x_{t}\) . Note that in this article, the DE local search is conducted with a population size of N and a maximum number of 100D + 1000 generations [25];
3) when \(a_{t}^{H}\) represents the KNN model, an offspring population with N solutions is first generated using DE operators. Then, \(x_{t}\) is selected based on L1-exploitation (if \(a_{t}^{L}=L 1\) -exploitation) or L1-exploration (if \(a_{t}^{L}=\) L1-exploration) from these offspring solutions.

Then, \(x_{t}\) is evaluated and added into the database D (lines 23 and 24), and the FEs is increased by 1 (line 25). Finally, the number of selections for \(a_{t}^{H}\) and \(a_{t}^{L}\) is increased by 1 (line 26), and their values are updated by the TL-R method (line 27).

When the maximum computational budget is consumed, the best solution in the database D is output as the final solution (line 30). In the following, we will give a detailed introduction to the two core components, TL-UCB and TL-R, sequentially.

Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply.

1120
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024

### B. TL-UCB
TL-UCB is the policy to select the high-level arm (surrogate model) and low-level arm (infill criterion). To choose a suitable high-level arm, the high-level UCB is executed, which is formulated as follows:
\[a_{t}^{H}=\underset{a \in \mathcal{A}^{H}}{arg max }\left[Q_{a}^{H}(t)+\sqrt{\frac{\alpha ln (t)}{T_{a}^{H}(t)}}\right] \tag{22}\]
where \(Q_{a}^{H}(t)\) denotes the value of the high-level arm a at the t -th iteration and \(T_{a}^{H}(t)\) is the number of selections of the high-level arm a during the past \(t-1\) iterations, t is the current iteration number, and α is a control parameter used to balance the tradeoff between exploiting well-performing arms and exploring rarely selected arms.

```
Algorithm 2: (aᴴₜ, aᴸₜ) = TL-UCB(𝒜ᴴ, 𝒜ᴸ, t, Tᴴ, Tᴸ, Qᴴ(t), Qᴸ(t), α).
1: Input:
2:   High-level arms set: 𝒜ᴴ = {aᴴ₁, . . . , aᴴ₄}
3:   Low-level arm set: 𝒜ᴸ = {aᴸ₁, . . . , aᴸ₈}
4:   Current number of iterations: t
5:   Number of selection for each arm: Tᴴ(t) and Tᴸ(t)
6:   The value for each arm: Qᴴ(t) and Qᴸ(t)
7:   Parameter: α
8: Set 𝒜ᴸ_{aᴴ₁} = {aᴸ₁, aᴸ₂}, 𝒜ᴸ_{aᴴ₂} = {aᴸ₃, aᴸ₄}, 𝒜ᴸ_{aᴴ₃} = {aᴸ₅, aᴸ₆}, 𝒜ᴸ_{aᴴ₄} = {aᴸ₇, aᴸ₈}
9: aᴴₜ = arg max_{a∈𝒜ᴴ} [ Qᴴₐ(t) + √( (α ln(t)) / Tᴴₐ(t) ) ]
10: aᴸₜ = arg max_{a∈𝒜ᴸ_{aᴴₜ}} [ Qᴸₐ(t) + √( (α ln(t)) / Tᴸₐ(t) ) ]
11: Output: High-level arm aᴴₜ and low-level arm aᴸₜ
```

When the high-level arm is determined, the low-level arm is picked up by the low-level UCB, which is defined as follows:
\[a_{t}^{L}=\underset{a \in \mathcal{A}_{a_{t}^{H}}^{L}}{arg max }\left[Q_{a}^{L}(t)+\sqrt{\frac{\alpha ln (t)}{T_{a}^{L}(t)}}\right] \tag{23}\]
where \(Q_{a}^{L}(t)\) denotes the value of the low-level arm a in the t -th iteration, \(T_{a}^{L}(t)\) is the number of selections of the low-level arm a during the past \(t-1\) iterations. Here, it should be noted that not each high-level arm can be associated with all low-level ones. Therefore, \(\mathcal{A}_{a_{t}^{H}}^{L}\) denotes the low-level arms that can be associated with the high level arm \(a_{t}^{H}\) . In this article, the association relationship between the high-level arms and low-level arms is shown in Fig. 2.

Fig. 2. Association relationship between the high-level arms and low-level ones.

Moreover, the pseudo-code of the TL-UCB is provided in Algorithm 2.

### C. TL-R
After determining the high-level arm (surrogate model) and low-level arm (infill criterion), the corresponding surrogate model is trained based on the current population P . Then, a new solution \(x_{t}\) is obtained by the selected high-level and lowlevel arms cooperatively. To measure the optimization utility of the chosen \(a_{t}^{L}\) and \(a_{t}^{H}\) , a TL-R is designed as follows.

1) Low-Level Reward:
\[r_{a_{t}^{L}}=-\frac{1}{N} I\left(x_{t}\right)+\frac{N+1}{N} \tag{24}\]
where N is the population size and \(I(x_{t})\) denotes the ranking of \(x_{t}\) in the current population P . Specifically, if \(x_{t}\) is the current best solution, \(I(x_{t})=1\) such that \(r_{a_{t}^{L}}=1\) . On the contrary, if \(I(x_{t})=N+1\) such that \(r_{a_{t}^{L}}=0\) . The low-level reward \(r_{a_{t}^{L}}\) has four characteristics: 1) \(r_{a_{t}^{L}}\) is bounded in [0, 1]; 2) \(r_{a_{t}^{L}}\) is linearly proportional to the ranking \(I(x_{t})\) of the newly generated solution \(x_{t}\) , the better the ranking, the larger the reward; 3) the low-level rewards are nonsparse; and 4) it remains stationary to some extent throughout the entire optimization process.

Based on the low-level reward \(r_{a_{t}^{L}}\) , the value of \(a_{t}^{L}\) is updated as follows:
\[Q_{a_{t}^{L}}^{L}(t+1)=\frac{T_{a_{t}^{L}}^{L}(t) Q_{a_{t}^{L}}^{L}(t)+r_{a_{t}^{L}}}{T_{a_{t}^{L}}^{L}(t)+1} \tag{25}\]

It should be noted that the value of the high-level arm \(a_{t}^{H}\) is increased if and only if the low-level reward is larger than its current value [e.g., \(r_{a_{t}^{L}} ≥Q_{a_{t}^{L}}^{L}(t)\)]

2) High-Level Reward: After the value \(Q_{a_{t}^{L}}^{L}\) of the low-level arm \(a_{t}^{L}\) is updated, the reward for the high-level arm \(a_{t}^{H}\) is designed as follows:
\[r_{a_{t}^{H}}=\frac{Q_{a_{t}^{L}}^{L}(t+1)-Q_{a_{t}^{L}}^{L}(t)}{\left|\mathcal{A}_{a_{t}^{H}}^{L}\right|} \tag{26}\]
where \(\mathcal{A}_{a_{t}^{H}}^{L}\) denotes the low-level arm set that is associated with the high-level arm \(a_{t}^{H}\) . Clearly, the high-level reward \(r_{a_{t}^{H}}\) can be positive or negative. To be specific, if the value of the low-level arm \(a_{t}^{L}\) increases (e.g., \(Q_{a_{t}^{L}}^{L}(t+1)-Q_{a_{t}^{L}}^{L}(t) ≥0\) ), the reward of the high-level arm is positive. Otherwise, it is negative.

Based on the high-level reward \(r_{a_{t}^{H}}\) , the value of \(a_{t}^{H}\) is updated as follows:
\[Q_{a_{t}^{H}}^{H}(t+1)=Q_{a_{t}^{H}}^{H}(t)+r_{a_{t}^{H}} . \tag{27}\]

Clearly, if the high-level reward of \(a_{t}^{H}\) is positive, its value will be increased. Otherwise, its value will be decreased. It should be pointed out that the value \(Q_{a_{t}^{H}}^{H}(t+1)\) of \(a_{t}^{H}\) is the mean of the value of its associated low-level arms. To be specific, putting (26) into (27), we have
\[Q_{a_{t}^{H}}^{H}(t+1)=\frac{\left|\mathcal{A}_{a_{t}^{H}}^{L}\right| \cdot Q_{a_{t}^{H}}^{H}(t)+Q_{a_{t}^{L}}^{L}(t+1)-Q_{a_{t}^{L}}^{L}(t)}{\left|\mathcal{A}_{a_{t}^{H}}^{L}\right|} . \tag{28}\]

Assume that \(|\mathcal{A}_{a_{t}^{H}}^{L}| \cdot Q_{a_{t}^{H}}^{H}(t)=\sum_{a^{L} \in \mathcal{A}_{a_{t}^{H}}^{L}} Q_{a^{L}}^{L}(t)\) holds for the t -th generation. Therefore
\[Q_{a_{t}^{H}}^{H}(t+1)=\frac{\sum_{a^{L} \in \mathcal{A}_{a_{t}^{H}}^{L}} Q_{a^{L}}^{L}(t)+Q_{a_{t}^{L}}^{L}(t+1)-Q_{a_{t}^{L}}^{L}(t)}{\left|\mathcal{A}_{a_{t}^{H}}^{L}\right|} . \tag{29}\]

Since the value of the un-selected low-level arm is not updated, \(Q_{a^{L}}^{L}(t+1)=Q_{a^{L}}^{L}(t),(a^{L} \neq a_{t}^{L})\) . Finally, we have
\[Q_{a_{t}^{H}}^{H}(t+1)=\frac{\sum_{a^{L} \in \mathcal{A}_{a_{t}^{H}}^{L}} Q_{a^{L}}^{L}(t+1)}{\left|\mathcal{A}_{a_{t}^{H}}^{L}\right|} . \tag{30}\]

Since the values of both the low-level arm and high-level arm are initialized to 0, the equation \(|\mathcal{A}_{a_{t}^{H}}^{L}| \cdot Q_{a_{t}^{H}}^{H}(t)=\sum_{a^{L} \in \mathcal{A}_{a_{t}^{H}}^{L}} Q_{a^{L}}^{L}(t)\) holds in the initialization (i.e., \(t=1\) ). Therefore, in each generation, the value of \(a_{t+1}^{H}\) is exactly the mean of the values of its associated low-level arms.

The pseudocode of TL-R is provided in Algorithm 3. In addition, Fig. 3 illustrates a numerical example of reward propagation in TL-R. In this figure, we assume that at the t -th generation, TL-UCB selects the high-level arm \(a_{t}^{H}\) and low-level arm \(a_{t}^{L}\) , and generates a solution \(x_{t}\) by them cooperatively. Suppose the low-level reward is \(r_{a_{t}^{L}}=0.8\) ,as computed using (24), and the value of the low-level arm is \(Q_{a_{t}^{L}}^{L}(t+1)=0.6\) , as obtained from (25). Then, using (26), we can compute the reward of the high-level arm \(r_{a_{t}^{H}}=0.05\) , and finally, the value of the high-level arm \(Q_{a_{t}^{H}}^{H}(t+1)=0.7\) is determined using (27). The path of the reward is indicated by solid red lines.

Fig. 3. Numerical example of reward propagation in TL-R.

```
Algorithm 3: (Qᴴ_{aᴴₜ}(t+1), Qᴸ_{aᴸₜ}(t+1)) = TL-R(N, P, xₜ, Tᴸ_{aᴸₜ}(t), Qᴸ_{aᴸₜ}(t), Qᴴ_{aᴴₜ}(t)).
1: Input:
2:   Population size: N
3:   Population: P
4:   New generated solution: xₜ
5:   Number of selections of aᴸₜ: Tᴸ_{aᴸₜ}(t)
6:   The value of aᴸₜ and aᴴₜ: Qᴸ_{aᴸₜ}(t) and Qᴴ_{aᴴₜ}(t)
7: Calculate the low-level reward r_{aᴸₜ} of aᴸₜ as (24)
8: Calculate Qᴸ_{aᴸₜ}(t + 1) of aᴸₜ as (25)
9: Calculate the high-level reward r_{aᴴₜ} of aᴴₜ as (26)
10: Calculate Qᴴ_{aᴴₜ}(t + 1) of aᴴₜ as (27)
11: Output: The updated values Qᴴ_{aᴴₜ}(t +1) and Qᴸ_{aᴸₜ}(t +1) of aᴴₜ and aᴸₜ
```

Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply.

XIE et al.: SAEA
1121

Compared with other existing adaptive-model SAEAs [22], [23], [24], [25], the proposed AutoSAEA has the following differences.
1) Although the model and infill criterion configuration are common, they differ from those used in the existing adaptive-model SAEAs.
2) AutoSAEA can adaptively choose the surrogate model and infill criterion in an online manner. However, existing adaptive-model SAEAs are either only for model or infill criterion selection but do not study adaptive selection for both cooperatively.
3) AutoSAEA formulates the surrogate model and infill criterion selection as a TL-MAB. Moreover, a TL-R and a TL-UCB designed to effectively capture good coupling behavior between the surrogate model and infill criterion. The adopted online learning strategy of AutoSAEA is obviously different from those of the existing adaptive SAEAs.