IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024 
# Surrogate-Assisted Evolutionary Algorithm With Model and Infill Criterion Auto-Configuration
Lindong Xie , Genghui Li, Zhenkun Wang , Member, IEEE, Laizhong Cui , Senior Member, IEEE, and Maoguo Gong , Senior Member, IEEE 
Abstract—Surrogate-assisted evolutionary algorithms (SAEAs) have proven to be effective in solving computationally expensive optimization problems (EOPs). However, the performance of SAEAs heavily relies on the surrogate model and infill criterion used. To improve the generalization of SAEAs and enable them to solve a wide range of EOPs, this article proposes an SAEA called AutoSAEA, which features model and infill criterion auto-configuration. Specifically, AutoSAEA formulates model and infill criterion selection as a two-level multiarmed bandit problem (TL-MAB). The first and second levels cooperate in selecting the surrogate model and infill criterion, respectively. A two-level reward (TL-R) measures the value of the surrogate model and infill criterion, while a two-level upper confidence bound (TL-UCB) selects the model and infill criterion in an online manner. Numerous experiments validate the superiority of AutoSAEA over some state-of-the-art SAEAs on complex benchmark problems and a real-world oil reservoir production optimization problem. 
Index Terms—Auto algorithm design, expensive optimization, surrogate-assisted evolutionary algorithm (SAEA), two-level multiarmed bandit (TL-MAB). 
# I. INTRODUCTION
E XPENSIVE optimization problems (EOPs) are prevalentin many engineering designs and applications (e.g., aeroand space tethered-net system design [3]). Unfortunately, solving such problems requires conducting computationally 
Manuscript received 28 January 2023; revised 7 May 2023; accepted 27 June 2023. Date of publication 3 July 2023; date of current version 1 August 2024. This work was supported in part by the National Natural Science Foundation of China under Grant 62206120, Grant 62106096, and Grant 62036006; and in part by the Shenzhen Technology Plan under Grant JCYJ20220530113013031. (Lindong Xie and Genghui Li contributed equally to this work.) (Corresponding author: Zhenkun Wang.) 
Lindong Xie and Genghui Li are with the School of System Design and Intelligent Manufacturing, Southern University of Science and Technology, Shenzhen 518055, China (e-mail: 12132679@mail.sustech.edu.cn; genghuili2- $\operatorname { c } @$ my.cityu.edu.hk). 
Zhenkun Wang is with the School of System Design and Intelligent Manufacturing and the Department of Computer Science and Engineering, Southern University of Science and Technology, Shenzhen 518055, China (e-mail: wangzhenku $9 0 @$ gmail.com). 
Laizhong Cui is with the College of Computer Science and Software Engineering and the Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), Shenzhen University, Shenzhen 518060, China (e-mail: cuilz@szu.edu.cn). 
Maoguo Gong is with the Key Laboratory of Collaborative Intelligence Systems, Ministry of Education, Xidian University, Xi’an 710071, China (e-mail: gong@ieee.org). 
This article has supplementary material provided by the authors and color versions of one or more figures available at https://doi.org/10.1109/ TEVC.2023.3291614. 
Digital Object Identifier 10.1109/TEVC.2023.3291614 
expensive computer simulations or costly physics experiments since the analytical expressions of the problem are unavailable. Generally, the EOP can be expressed as follows: 
$$
\min  f (\mathbf {x})
$$
$$
\mathrm {s . t .} \quad \mathbf {x} _ {l} \leq \mathbf {x} \leq \mathbf {x} _ {u} \tag {1}
$$
where $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { D } ) ^ { \top }$ is the decision vector of $D$ variables, $\mathbf { x } _ { l } = ( x _ { l , 1 } , \dots , x _ { l , D } ) ^ { \top }$ and $\mathbf { x } _ { u } = ( x _ { u , 1 } , \ldots , x _ { u , D } ) ^ { \top }$ are the lower and upper bounds of the search space, respectively, and $f ( \mathbf { x } )$ denotes a scalar objective function. We assume that the analytical expression of $f ( \mathbf { x } )$ is unavailable and the calculation of $f ( \mathbf { x } )$ is costly. 
Evolutionary algorithms have been shown to be popular and effective for solving black-box optimization problems [4], [5]. However, they often perform poorly on EOPs due to the large number of function evaluations (FEs) required in their optimization process. This limitation becomes more significant in the case of EOPs [6]. To address this issue, surrogate-assisted evolutionary algorithms (SAEAs) have been proposed. In SAEAs, the optimization is mainly driven by computationally cheap surrogate models. The popular surrogate models used in SAEAs include regression models, such as radial basis function (RBF) [7], Gaussian process model (GP) [8], and polynomial response surface (PRS) [9], and classification models, such as $k$ -nearest neighbor (KNN) [10] and support vector machine model (SVM) [11]. Based on the adopted surrogate model, existing SAEAs can be roughly grouped into three categories: 1) single-model SAEAs [12], [13], [14], [15]; 2) multiplemodel SAEAs [16], [17], [18], [19], [20], [21]; and 3) adaptive-model SAEAs [22], [23], [24], [25]. 
Single-model SAEAs use a fixed surrogate model throughout the optimization process, typically combining a fixed infill criterion, such as GP with expected improvement (EI) [26] and lower confidence bound (LCB) [27], RBF with local search [28], or KNN with prescreening [29], to select candidate solutions for FEs. However, due to the limited training samples available and the inability of a single model to effectively fit various problem landscapes, single-model SAEAs often struggle to solve different EOPs effectively [18], [30]. 
In order to enhance the robustness and scalability of singlemodel SAEAs, multiple-model SAEAs have been developed, including hierarchical-model SAEAs and ensemble-model 
1114 
1089-778X $\circledcirc$ 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. 
See https://www.ieee.org/publications/rights/index.html for more information. 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
XIE et al.: SAEA 
SAEAs [31]. Hierarchical-model SAEAs commonly combine a global model and a local model [16], [18], [32], [33], [34]. The global model approximates the fitness landscape to search for promising regions, and the local model is used to conduct a refined search in the found local promising regions. Ensemble-model SAEAs typically train several different base surrogate models to cooperatively implement a more precise prediction [9], [19], [20], [21]. Generally, multiple-model SAEAs outperform single-model ones in most cases [15]. However, the computational cost of multiplemodel SAEAs is always higher compared to the single-model SAEAs [35], [36]. Additionally, some base models may not fit the problem landscape well, so their efficiency may still be low. 
To leverage well-established models effectively and reduce computational complexity, adaptive-model SAEAs have been proposed that automatically select the suitable model (or infill criterion) in an online manner [22], [24], [25]. However, the performance of such algorithms dramatically depends on the design of the adaptive selection strategy. It is well known that the surrogate model is tightly coupled with the infill criterion in SAEAs, and they cooperatively affect the performance of SAEAs. A model with different infill criteria can realize different balances of exploiting better solutions and exploring unexplored search regions. Naturally, a better balance between exploration and exploitation may be achieved by cooperatively considering the model and the infill criterion in a hierarchical coupled way. Unfortunately, existing adaptive-model SAEAs are either only for model or infill criterion selection but do not study adaptive selection for both cooperatively. Therefore, to improve the performance of SAEAs by filling this gap, this article proposes an SAEA (called AutoSAEA) with the model and infill criterion autoconfiguration by using the hierarchical multiarmed bandit (MAB) method. 
The contributions of this article are summarized as follows. 
1) Formulating the surrogate model and infill criterion cooperative selection as a two-level MAB problem (TL-MAB) [37], [38], [39]. 
2) Designing a two-level reward (TL-R) to measure the utility of the surrogate model and infill criterion in a cooperative manner. 
3) Adopting a two-level upper confidence bound (TL-UCB) to select the surrogate model and infill criterion cooperatively in an online manner. 
4) Proposing an SAEA with a surrogate model and infill criterion auto-configuration, and verifying its advantages by comparing it with some state-of-the-art SAEAs on two sets of complex benchmark problems and a realworld problem. 
The remainder of this article is organized as follows. Section II reviews the related work. Section III introduces the background knowledge involved in the proposed algorithm. Section IV describes the proposed algorithm in detail. Section V presents the numerical experiments conducted to validate the effectiveness of the proposed algorithm. Finally, Section VI summarizes this article and discusses future work directions. 
# II. RELATED WORK
1) Single-Model SAEAs: Single-model SAEAs can either train a global model using all evaluated solutions or a local model with some good ones to predict the quality of new solutions. For example, the Bayesian optimization framework is a classic and popular single-model method [40], [41], [42]. It uses the GP model to build a surrogate for the objective and quantify the uncertainty in the surrogate. Then, it optimizes an acquisition function (e.g., EI [40] and LCB [43]) defined from the surrogate to decide the new solution for evaluation. GPEME [13] first constructs a local GP model in the lowerdimensional space, followed by the application of prescreening with LCB to determine which offspring can be selected for FE. CPS-MOEA [44] employs a local KNN classifier to filter out potentially nondominated solutions for real evaluation. SA-COSO [45] evaluates the new position of the particle that has the minimum fitness value predicted by a global RBF model. In addition, SHPSO [46] and SAMSO [15] construct a local RBF model and a global RBF model, respectively, to select particles with predicted fitness values that are better than their personal best ones for FEs. MGP-SLPSO [14] formulates the approximated fitness and its corresponding uncertainty provided by a global GP model as a bi-objective problem and applies a nondominated sorting method to select the offspring for FEs. Furthermore, CA-LLSO [47] trains a local gradient boosting classifier to predict the level of the offspring produced by the level-based learning swarm optimizer. 
2) Multiple-Model SAEAs: Multiple-model SAEAs mainly combine a global and a local model or train multiple models simultaneously to balance exploration and exploitation. For example, CAL-SAPSO [19] identifies the best and most uncertain solutions for FEs using a global ensemble surrogate model. Additionally, the optimum of the local ensemble model found by the local search is also used for expensive evaluation. HeE-MOEA [35] utilizes an SVM and two RBF models to build an ensemble model, which is combined with LCB and EI criteria to screen the offspring for expensive evaluations. ESAO [32] employs a global RBF to screen the offspring with the minimum predicted fitness value for FE. Besides, the optimum of the local RBF model found by the differential evolution (DE) is also evaluated. GSGA [33] adopts a local RBF-assisted trust-region method and a global GP-assisted genetic algorithm for exploitation and exploration, respectively. GL-SADE [18] trains a global RBF model and a local GP model to select offspring for FEs. Moreover, when the local GP model finds the current best solution, DE is applied further to search for its optimal solution. ESCO [21] constructs multiple RBF models on various low-dimensional sample sets, and then selects the models with superior performance to compose an ensemble surrogate. 
3) Adaptive-Model SAEAs: Adaptive-model SAEAs always design an adaptive selection strategy to automatically choose a model or an infill criterion in an online manner. For example, GP-Hedge [22] adaptively selects an appropriate acquisition function from a portfolio of well-established ones, such as the probability of improvement, EI, and UCB. This selection is based on an online MAB strategy. Specifically, 
1115 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024 
the optimal solutions of all acquisition functions form a candidate solution pool, from which solutions are selected for FEs based on the cumulative rewards of their respective acquisition functions. While GP-Hedge only adaptively selects the acquisition functions, it fixes the surrogate model. Moreover, its computational efficiency is low since it needs to optimize multiple acquisition functions in each generation. ASMEA [30] first constructs a base model pool using GP, RBF, PRS, and linear Shepard interpolation. It then uses the prediction residual error sum of squares (PRESS) and root mean square error (RMSE) to select several meta-models from the base model pool for constructing five ensemble models, which are then included in the base model pool. Finally, a promising surrogate model is adaptively selected from the base model pool based on its minimum RMSE. While ASMEA only adaptively selects the models, it fixes the prescreening as the infill criterion. Moreover, its computational efficiency is low since it needs to conduct cross-validation for each base model. ESA [25] first constructs a pool of four different infill criteria, including DE evolutionary screening, surrogate-assisted local search, full-crossover strategy, and surrogate-assisted trust-region local search. Among them, the DE evolutionary screening strategy prefers exploration, the surrogate-assisted local search and trust-region local search strategies use different local search methods to favor exploitation, and the full-crossover strategy integrates good genes from historical solutions. Moreover, $Q$ -learning is used to adjust the selection probability of each infill criterion through feedback information received during the optimization process. However, ESA fixes the RBF as the surrogate model. 
Additionally, some adaptive-model SAEAs have been proposed to deal with expensive multiobjective optimization problems. For instance, KTA2 [23] divides the optimization process into three states (i.e., convergence-demand, diversitydemand, and uncertainty-demand states) and uses the distances from solutions to the estimated ideal point and the pure diversity indicator to estimate the state. Moreover, the infill criterion is adaptively chosen based on the estimated optimization state, which guides the solution selection for FEs by taking into account the requirements on convergence, diversity, and model uncertainty separately. RVMM [48] uses the GP as the surrogate model and develops an adaptive model management strategy to adaptively choose the convergence-related criterion and diversity-related criterion. The adaptive model management strategy is assisted by two sets of reference vectors. One set of adaptive reference vectors focuses on convergence, while the other set of fixed reference vectors concentrates on diversity. The GP and RBF models are the most popular surrogate models due to their high fidelity and simplicity. Generally, the RBF model is computationally more efficient than the GP model. However, the GP model can provide uncertain information about its predictions. Therefore, to take advantage of the GP and RBF models in dealing with different problems, IBEA-MS [24] designs an acceptable reliability tolerance criterion to adaptively determine whether to use the GP model or the RBF model in environmental selection. Specifically, when the uncertainty of the GP model exceeds the acceptable error, the RBF model is used. Otherwise, the GP models are used. 
# III. BACKGROUND
# A. Surrogate Models
1) GP Model: Given a training data set $\{ \mathbf { x } _ { i } , f ( \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ , GP predicts the output $f _ { \mathrm { G P } } ( \mathbf { x } )$ of a solution $\mathbf { X }$ in the way of: $f _ { \mathrm { G P } } ( \mathbf { x } ) = \mu ( \mathbf { x } ) + \epsilon ( \mathbf { x } )$ , where $\mu ( \mathbf { x } )$ represents a global trend of the training data, and $\epsilon ( \mathbf { x } ) \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ( \mathbf { x } ) )$ is a normal distribution. $\mu ( \mathbf { x } )$ and $\sigma ^ { 2 } \mathbf { \left( x \right) }$ are estimated as follows [40]: 
$$
\mu (\mathbf {x}) = \mathbf {k} (\mathbf {x}) ^ {\top} \mathbf {K} ^ {- 1} \mathbf {F} \tag {2}
$$
$$
\sigma^ {2} (\mathbf {x}) = \kappa (\mathbf {x}) - \mathbf {k} (\mathbf {x}) ^ {\top} \mathbf {K} ^ {- 1} \mathbf {k} (\mathbf {x}) \tag {3}
$$
where $\mathbf k ( \mathbf x ) \ = \ [ C ( \mathbf x , \mathbf x _ { 1 } ) , \dots , C ( \mathbf x , \mathbf x _ { N } ) ] ^ { \top }$ ; K is an $N \times N$ matrix and $K _ { i , j } ~ = ~ C ( \mathbf { x } _ { i } , \mathbf { x } _ { j } )$ ; $\begin{array} { r } { { \boldsymbol { \kappa } } ( \mathbf { x } ) = { \boldsymbol { C } } ( \mathbf { x } , \mathbf { x } ) } \end{array}$ , and ${ \textbf { F } } =$ $[ f ( \mathbf { x } _ { 1 } ) , \dotsc , f ( \mathbf { x } _ { N } ) ] ^ { \intercal }$ . The covariance function $C ( \cdot , \cdot )$ can be commonly calculated as follows: 
$$
C \left(\mathbf {x} _ {i}, \mathbf {x} _ {j}\right) = \exp \left(- \sum_ {d = 1} ^ {D} \theta_ {d} \left| x _ {i, d} - x _ {j, d} \right| ^ {2}\right) \tag {4}
$$
where the hyper-parameters $\theta$ can be obtained by maximizing the likelihood function [33]. 
Finally, for an unknown solution $\mathbf { X } ,$ , its predicted fitness value $\hat { f } _ { \mathrm { G P } } ( \mathbf { x } )$ and uncertainty $\hat { s } ( \mathbf { x } )$ are given in the following: 
$$
\hat {f} _ {\mathrm {G P}} (\mathbf {x}) = \mu (\mathbf {x}) + \mathbf {k} (\mathbf {x}) ^ {\top} \mathbf {K} ^ {- 1} (\mathbf {F} - \mathbf {I} \mu (\mathbf {x})) \tag {5}
$$
$$
\hat {s} ^ {2} (\mathbf {x}) = \sigma^ {2} (\mathbf {x}) \left[ 1 - \mathbf {k} (\mathbf {x}) ^ {\top} \mathbf {K} ^ {- 1} \mathbf {k} (\mathbf {x}) + \frac {\left(1 - \mathbf {I} ^ {\top} \mathbf {K} ^ {- 1} \mathbf {k} (\mathbf {x})\right) ^ {2}}{\mathbf {I} ^ {\top} \mathbf {K} ^ {- 1} \mathbf {I}} \right] \tag {6}
$$
where I is an $N \times 1$ unit vector, and $\hat { s } ( \mathbf { x } ) = \sqrt { \hat { s } ^ { 2 } ( \mathbf { x } ) }$ is the RMSE. 
2) RBF Model: The RBF model uses a linear combination of basis functions to approximate the fitness landscape. For the given training data set $\{ \mathbf { x } _ { i } , f ( \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ = , the form of the RBF model can be formulated as follows [49]: 
$$
\hat {f} _ {\mathrm {R B F}} (\mathbf {x}) = \sum_ {i = 1} ^ {N} w _ {i} \varphi \left(\| \mathbf {x} - \mathbf {x} _ {i} \|\right) + b _ {0} + \sum_ {j = 1} ^ {D} b _ {j} x _ {j} \tag {7}
$$
where $\mathbf { w } = ( w _ { 1 } , w _ { 2 } , \ldots , w _ { N } ) ^ { \top }$ is the weight vector of the basis function, and this article adopts the cubic function $\varphi ( \| \mathbf { x } - \mathbf { x } _ { i } \| ) = ( \| \mathbf { x } - \mathbf { x } _ { i } \| ) ^ { 3 }$ , where $\left\| \cdot \right\|$ denotes the Euclidean distance. $\mathbf { b } = ( b _ { 0 } , b _ { 1 } , \ldots , b _ { D } ) ^ { \top }$ is the coefficient of the first-order polynomial. 
The parameters w and $\mathbf { b }$ in (7) can be obtained as follows: 
$$
\left[ \begin{array}{l} \mathbf {w} \\ \mathbf {b} \end{array} \right] = \left[ \begin{array}{c c} \boldsymbol {\Phi} & \mathbf {P} \\ \mathbf {P} ^ {\top} & \mathbf {0} _ {(D + 1) \times (D + 1)} \end{array} \right] ^ {\dagger} \left[ \begin{array}{l} \mathbf {F} \\ \mathbf {0} _ {(D + 1) \times 1} \end{array} \right] \tag {8}
$$
where $\begin{array} { r c l } { \Phi _ { i , j } } & { = } & { \varphi ( \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 } ) } \end{array}$ , $i , j ~ = ~ 1 , 2 , \ldots , N$ ; $\begin{array} { r l } { \mathbf { p } } & { { } = } \end{array}$ $[ \mathbf { P } _ { 1 } , \dots , \mathbf { P } _ { N } ] ^ { \top }$ and $\mathbf { P } _ { i } = [ 1 , x _ { i , 1 } , \ldots , x _ { i , D } ] ^ { \top } ; \mathbf { 0 } _ { ( D + 1 ) \times ( D + 1 ) }$ is a zero matrix of $( D + 1 ) \times ( D + 1 )$ , and $^ \dagger$ is the generalized inverse, and $\mathbf { F } = [ f ( \mathbf { x } _ { 1 } ) , \dots , f ( \mathbf { x } _ { N } ) ] ^ { \intercal }$ . 
3) PRS Model: The commonly used second-order polynomial is defined as follows [9]: 
$$
\hat {f} _ {\mathrm {P R S}} (\mathbf {x}) = \beta_ {0} + \sum_ {i = 1} ^ {D} \beta_ {i} x _ {i} + \sum_ {i = 1} ^ {D} \sum_ {j \geq i} ^ {D} \beta_ {i j} x _ {i} x _ {j} \tag {9}
$$
1116 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
XIE et al.: SAEA 
where coefficients $\beta _ { 0 } , \ \beta _ { i }$ , and $\beta _ { i j }$ are regression parameters for the intercept, the linear term, and the quadratic term, respectively. 
For the given training data set $\{ \mathbf { x } _ { i } , f ( \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ , the unknown coefficients of the above polynomial model $\beta ~ = ~ ( \beta _ { 0 } , \ldots ,$ , βD, β1,1, . . . , β1,D, β2,3, . . . , $\beta _ { 2 , D } , . . . , \ \beta _ { D - 1 , D } , \beta _ { D , D } ) ^ { \top }$ can be obtained by the least square method as follows [50]: 
$$
\boldsymbol {\beta} = \left(\mathbf {P} ^ {\top} \mathbf {P}\right) ^ {- 1} \mathbf {P} ^ {\top} \mathbf {F} \tag {10}
$$
where $\mathbf { P } = [ \mathbf { P } _ { 1 } , \ldots , \mathbf { P } _ { N } ] ^ { \top }$ , $\mathbf { P } _ { i } = [ 1 , x _ { i , 1 } , \ldots , x _ { i , D } ] ^ { \top }$ , and ${ \bf F } =$ $[ f ( \mathbf { x } _ { 1 } ) , \dots , f ( \mathbf { x } _ { N } ) ] ^ { \top }$ . 
4) KNN Model: KNN is a popular classifier [29], [44]. Based on a training data set $\{ \mathbf { x } _ { i } , f ( \mathbf { x } _ { i } ) , y ( \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ , where $f ( \mathbf { x } _ { i } )$ and $y ( \mathbf { x } _ { i } )$ denote the objective function value and class/level of the solution $\mathbf { x } _ { i }$ , respectively. For a new solution $\mathbf { X }$ , its class/level is predicted as follows: 
$$
y _ {\mathrm {K N N}} (\mathbf {x}) = \operatorname {m o d e} \left(\left\{y \left(\mathbf {x} _ {1}\right), \dots , y \left(\mathbf {x} _ {K}\right) \right\}\right) \tag {11}
$$
where $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { K }$ are $K$ nearest neighbors of $\mathbf { X }$ , and mode() denotes the mode of a set. In this article, $K$ is set to 1 [29], [44], which means that the new solution $\mathbf { X }$ is assigned the same class as its nearest neighbor. 
In this article, all surrogate models are trained based on the current population $\mathcal { P }$ in each iteration. 
# B. Infill Criteria
Since the surrogate models are tightly coupled with the infill criteria in SAEAs, the corresponding infill criteria for the surrogate models mentioned in the previous section are introduced as follows. 
1) Infill Criteria for the GP Model: The LCB and EI criteria are commonly combined with GP for determining new solutions for FEs, which are expressed sequentially as follows [13], [33]: 
$$
\mathbf {x} ^ {*} = \underset {\mathbf {x} \in \mathcal {X}} {\arg \min } \left(\hat {f} _ {\mathrm {G P}} (\mathbf {x}) - w \hat {s} (\mathbf {x})\right) \tag {12}
$$
where $w$ is set to 2 in this article [13], [27]. $\mathcal { X }$ denotes a set of unknown solutions. 
$$
\mathbf {x} ^ {*} = \underset {\mathbf {x} \in \mathcal {X}} {\arg \max } \left(\left(f _ {\min } - \hat {f} _ {\mathrm {G P}} (\mathbf {x})\right) \Phi (Z) + \hat {s} (\mathbf {x}) \varphi (Z)\right) \tag {13}
$$
where $Z = ( [ f _ { \mathrm { m i n } } - \hat { f } _ { \mathrm { G P } } ( { \bf x } ) ] / [ \hat { s } ( { \bf x } ) ] ) , { } _ { , }$ $f _ { \mathrm { m i n } }$ is the current minimum objective function value, and $\Phi ( \cdot )$ and $\varphi ( \cdot )$ are the density and cumulative distribution functions of standard normal distribution, respectively. 
This article adopts the DE mutation and crossover operators to generate the new solution set $\mathcal { X } = \{ \mathbf { o } _ { 1 } , \dotsc \dotsc , \mathbf { o } _ { N } \}$ , where $N$ is the population size. To be specific, assume that the current population is $\mathcal { P }$ , $\mathbf { 0 } _ { i } = ( o _ { i , 1 } , \ldots , o _ { i , D } )$ is generated as follows: 
$$
\mathbf {v} _ {i} = \mathbf {x} _ {i} + F \times \left(\mathbf {x} _ {b} - \mathbf {x} _ {i}\right) + F \times \left(\mathbf {x} _ {r 1} - \mathbf {x} _ {r 2}\right) \tag {14}
$$
$$
o _ {i, j} = \left\{ \begin{array}{l} v _ {i, j}, \text {i f} \quad \operatorname {r a n d} _ {i, j} \leq C R \quad \text {o r} \quad j = j _ {\text {r a n d}} \\ x _ {i, j}, \text {o t h e r w i s e} \end{array} \right. \tag {15}
$$
where $\mathbf { x } _ { i }$ is the ith solution in $\mathcal { P } , \ \mathbf { x } _ { r 1 }$ and $\mathbf { X } _ { \boldsymbol { r } 2 }$ are randomly selected from the current population $\mathcal { P }$ , $\mathbf { X } _ { b }$ is the best solution of ; $F$ is the scale factor, and it is set to 0.5 in this 
article. $\mathrm { r a n d } _ { i , j }$ and $j _ { \mathrm { r a n d } }$ are randomly selected from [0, 1] and $\{ 1 , \ldots , D \}$ , respectively, and $C R \in [ 0 , 1 ]$ denotes the crossover rate, which is set to 0.9 in this article. 
Note that if $o _ { i , j }$ violates the boundary constraints, it will be repaired as follows: 
$$
o _ {i, j} = x _ {l, j} + \operatorname {r a n d} \times \left(x _ {u, j} - x _ {l, j}\right). \tag {16}
$$
2) Infill Criteria for the RBF and PRS Models: For all regression models (e.g., RBF and PRS), prescreening and local search are widely used in SAEAs, which are formulated as follows [28], [51]: 
$$
\mathbf {x} ^ {*} = \underset {\mathbf {x} \in \mathcal {X}} {\arg \min } \hat {f} _ {\mathrm {R B F / P R S}} (\mathbf {x}) \tag {17}
$$
where $\mathbf { X }$ and $\hat { f } _ { \mathrm { R B F / P R S } } ( \mathbf { x } )$ denote the unknown solution and its predicted fitness value by RBF/PRS, respectively. It should be noted that if $\mathcal { X }$ is a finite set of solutions, (17) belongs to prescreening, and it denotes a local search if $\mathcal { X }$ is a subspace of the search space. 
In this article, the above DE operator is used to generate $\mathcal { X }$ for the prescreening. For the local search, $\mathcal { X }$ is defined as follows: 
$$
\mathcal {X} = [ \mathbf {l b}, \mathbf {u b} ] \tag {18}
$$
where $\mathbf { l b } \ = \ ( l _ { 1 } , \ldots , l _ { D } ) ^ { \top }$ and $\mathbf { u } \mathbf { b } \ = \ ( u _ { 1 } , \ldots , u _ { D } ) ^ { \top }$ , $l _ { j } ~ =$ $\operatorname* { m i n } \{ x _ { i , j } , \mathbf { x } _ { i } \in \mathcal { P } \}$ , and $u _ { j } = \operatorname* { m a x } \{ x _ { i , j } , \mathbf { x } _ { i } \in \mathcal { P } \}$ , $j = 1 , \dots , D$ . Moreover, DE is used to solve (17) in the local search in this article. 
3) Infill Criteria for the KNN Model: Classification models are often used to predict the level of the unknown solutions [52]. To this end, the current population is evenly divided into $L$ levels $\mathcal { P } ^ { 1 } , \ldots , \mathcal { P } ^ { L }$ based on the fitness, and the best $( N / L )$ solutions belong to the first level $\mathcal { P } ^ { 1 }$ . Two infill criteria (e.g., L1-exploitation and L1-exploration) have been designed for the classifier model. They are defined as follows [47]. 
1) L1-Exploitation: The L1-exploitation criterion is defined as 
$$
\mathbf {x} ^ {*} = \underset {\mathbf {x} \in \mathcal {X} ^ {1}} {\arg \min } \max  \left\{\| \mathbf {x} - \mathbf {p} \|, \mathbf {p} \in \mathcal {P} ^ {1} \right\}. \tag {19}
$$
2) L1-Exploration: The L1-exploration is defined as 
$$
\mathbf {x} ^ {*} = \underset {\mathbf {x} \in \mathcal {X} ^ {1}} {\arg \max } \min  \left\{\| \mathbf {x} - \mathbf {p} \|, \mathbf {p} \in \mathcal {P} ^ {1} \right\} \tag {20}
$$
where $\mathcal { X } ^ { 1 }$ includes the solution in new generated solution set $\mathcal { X }$ whose predicted level by the KNN model is L1. In this article, $\mathcal { X }$ is also generated by the above-described DE operators, and the number of levels is set to 5. 
For the models and infill criteria discussed above, we have the following remarks. Over the past two decades, a variety of regression and classification models have been employed to aid evolutionary algorithms in solving complex optimization problems. As far as our knowledge goes, GP, RBF, PRS, and KNN are the most effective and commonly used models in state-of-the-art SAEAs [9], [19], [44], each with unique advantages in handling diverse problems. In this article, our goal is to develop an SAEA with auto-configuration of models and infill criteria. Therefore, we have chosen these well-known models for their established strengths. However, it should be 
1117 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/58147bb2e5a4f2548d5972a5efd5130e95c65da461d51bc81ee6e1dbc74c345f.jpg)


Fig. 1. Illustrative examples of HMAB clustering structures. (a) Disjoint clustering structure. (b) Hierarchical clustering structure.

noted that other models and infill criteria can also be easily incorporated into our proposed algorithm. Moreover, regarding the parameters involved in the infill criteria, we have tested several representative values for each parameter in our preliminary experiments. The experimental results show that they do not have a significant influence. Therefore, we use commonly accepted settings for these parameters in this article. 
# C. Hierarchical Multiarmed Bandit
MAB is a powerful tool for online learning [37]. In the standard MAB problem, the player must pick an arm $a _ { t }$ from a set of arms $\mathcal { A }$ at each time slot t and then obtain a reward $r _ { a _ { t } }$ , which is produced from a distribution that is unknown to the player. The performance of an algorithm in MAB is typically measured by the regret, and the player’s goal is to minimize the expected cumulative regret over a sequence of $T$ time slots. The expected cumulative regret can be expressed as follows [37]: 
$$
\mathbb {E} \left[ R _ {T} \right] = \sum_ {t = 1} ^ {T} r _ {a _ {t} ^ {*}} - r _ {a _ {t}} \tag {21}
$$
where $a _ { t } ^ { * }$ denotes the unknown best arm at the $t$ -th time slot. 
Hierarchical MAB (HMAB) is an extension of the standard MAB. Two commonly used hierarchical structures have been developed [39]. 
1) Disjoint Clustering Structure: The $K$ arms are classified into a set of clusters, and each arm $a \in { \mathcal { A } }$ belongs to only one cluster. The player first picks a cluster and then chooses an arm in the selected cluster. Finally, a reward is produced by the chosen arm. An illustrative example of disjoint clustering is shown in Fig. 1(a). 
2) Hierarchical Clustering Structure: The arms can be divided into multiple levels. The arms on the same level are different from each other, and the different highlevel arms can be associated with the same low-level arms. The player first selects an arm from the high-level and then chooses an arm from its associated low-level. Finally, a reward is returned by all chosen arms cooperatively. An illustrative example of hierarchical clustering is shown in Fig. 1(b). 
Many theories and experiments have demonstrated that HMAB has significant advantages over standard MAB [37], [38], [39], [53], [54], [55]. In general, its benefits can be summarized into three aspects: 1) the regret bound of HMAB is lower than that of standard MAB; 2) HMAB can reduce the size of the arm space, especially when the arm space is large; 

Algorithm 1: AutoSAEA(N, MaxFEs, AH, AL, α).

1: Input:  
2: Population size: $N$ 3: Maximum number of FEs: MaxFEs  
4: High-level arm set: $\mathcal{A}^H = \{a_1^H, \ldots, a_4^H\}$ 5: Low-level arm set: $\mathcal{A}^L = \{a_1^L, \ldots, a_8^L\}$ 6: Parameters: $\alpha$ 7: Initialization:  
8: Use LHS to sample $N$ solutions in the search space  
9: Evaluate and save them into the database $\mathcal{D}$ 10: Set $Q_a^H(1)$ , $a \in \mathcal{A}^H$ and $Q_a^L$ , $a \in \mathcal{A}^L$ to 0  
11: Set $T_a^H(1)$ , $a \in \mathcal{A}^H$ and $T_a^L$ , $a \in \mathcal{A}^L$ to 0  
12: Set $\mathcal{CA}$ based on $\mathcal{A}^H$ and $\mathcal{A}^L$ 13: Set $FEs = N$ and $t = 1$ 14: while $FEs < MaxFEs$ do  
15: Select $N$ best solutions from $\mathcal{D}$ as the population $\mathcal{P}$ 16: if $t \leq |\mathcal{CA}|$ then  
17: Select the $t$ -th combinatorial arm in $\mathcal{CA}$ 18: Set $a_t^H$ and $a_t^L$ as the first and second arm in the selected combinatorial arm, respectively  
19: else  
20: Choose $a_t^H$ and $a_t^L$ by TL-UCB  
21: end if  
22: Obtain $\mathbf{x}_t$ by $a_t^H$ and $a_t^L$ cooperatively  
23: Evaluate $\mathbf{x}_t$ 24: $\mathcal{D} = \mathcal{D} \cup \mathbf{x}_t$ 25: FEs = FEs + 1  
26: $T_{a_t^H}(t + 1) = T_{a_t^H}(t) + 1$ , $T_{a_t^L}(t + 1) = T_{a_t^L}(t) + 1$ 27: Update the value of $a_t^H$ and $a_t^L$ by TL-R  
28: $t = t + 1$ 29: end while  
30: Output: The best solution in $\mathcal{D}$ 
and 3) HMAB can greatly mitigate the chances of selecting suboptimal arms, leading to improved performance in solving EOPs. Therefore, we propose integrating HMAB into SAEAs to enable the adaptive and cooperative selection of surrogate models and infill criteria. In this article, we consider the hierarchical clustering structure of HMAB, and its details will be explained in the following section. 
# IV. AUTOSAEA
# A. Algorithm Structure
The framework of the proposed AutoSAEA is shown in Algorithm 1. Its input includes the population size $N$ (line 2), the maximum number of FEs MaxFEs (line 3), a set of highlevel arms $\mathcal { A } ^ { H }$ (line 4), a set of low-level arms $\mathcal { A } ^ { L }$ (line 5), and the parameter $\alpha$ (line 6). The high-level and low-level arms denote the surrogate model and the infill criterion, respectively. In this article, we set $\mathcal { A } ^ { H } = \{ a _ { 1 } ^ { H } = \mathrm { G P } ,$ $a _ { 2 } ^ { H } = \mathrm { R B F }$ , $a _ { 3 } ^ { H } =$ PRS, $a _ { 4 } ^ { H } = \mathsf { K N N } \}$ and $\mathcal { A } ^ { L } = \bar { \{ a _ { 1 } ^ { L } = \mathrm { L C B } }$ , $a _ { 2 } ^ { L } = \mathrm { E I }$ , $a _ { 3 } ^ { L } =$ prescreening, $a _ { 4 } ^ { L } =$ local search, $\bar { a } _ { 5 } ^ { L } =$ prescreening, $a _ { 6 . } ^ { L ^ { - } } =$ local search, $a _ { 7 } ^ { L } = \mathrm { L } 1$ -exploitation, $a _ { 8 } ^ { L } = \mathrm { L } 1$ -exploration}.1 
1Note that $a _ { 3 } ^ { L }$ and $a _ { 4 } ^ { L }$ are associated with RBF, and $a _ { 5 } ^ { L }$ and $a _ { 6 } ^ { L }$ are associated with PRS. 
1118 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
XIE et al.: SAEA 
In the initialization (lines 7–13), an initial population with $N$ solutions is generated by the Latin hypercube sampling (LHS) method (line 8). These solutions are evaluated directly and saved into a database $\mathcal { D }$ (line 9). The value for each arm and the number of selections for each arm is set to 0 (lines 10 and 11). Moreover, to make each arm be selected at least once, a legal combinatorial arm set is set as $\mathcal { C A }$ $= \{ ( \mathrm { G P } , \mathrm { L C B } )$ , (GP, EI), (RBF, prescreening), (RBF, local search), (PRS, prescreening), (PRS, local search), (KNN, L1- exploitation), (KNN, L1-exploration)} (line 12). Finally, the current number of FEs and the iteration number are set to $N$ and 1, respectively (line 13). 
In the optimization process (lines 14–29), it first selects the $N$ best solutions from $\mathcal { D }$ as the population $\mathcal { P }$ (line 15). If the number of iterations $t$ is less than the cardinality of the combinatorial arm set $\mathcal { C A }$ (line 16), the $t \cdot$ -th combinatorial arm in $\mathcal { C A }$ is chosen (line 17). Then, the high-level arm $a _ { t } ^ { H }$ and low-level arm $a _ { t } ^ { L }$ are set to the first and second arms in the selected combinatorial arm (line 18), respectively. Otherwise, they are determined by the TL-UCB method (line 20). When the high-level arm $a _ { t } ^ { H }$ and low-level arm $a _ { t } ^ { L }$ are selected, a new solution $\mathbf { X } _ { t }$ is determined by them cooperatively (line 22). To be specific: 
1) when $a _ { t } ^ { H }$ represents the GP model, an offspring population with $N$ solutions is first generated by DE operators. Then, the offspring solution with the best LCB value (if $a _ { t } ^ { L } = \operatorname { L C B } )$ or best EI value (if $a _ { t } ^ { L } = \operatorname { E I } )$ is chosen as $\mathbf { X } _ { t }$ ; 2) when $a _ { t } ^ { H }$ represents the RBF model or PRS model, if $\begin{array} { r l } { a _ { t } ^ { L } } & { { } = } \end{array}$ prescreening, an offspring population with $N$ solutions is first generated by DE operators. Then, the offspring solution with the best $\hat { f } _ { \mathrm { R B F } } ^ { \mathrm { ^ { c } } } ( \mathbf { x } )$ value (if $a _ { t } ^ { H } \ = \ \mathrm { R B F } )$ or $\hat { f } _ { \mathrm { P R S } } ( \mathbf { x } )$ value (if $a _ { t } ^ { H } = \operatorname { P R S }$ ) is chosen as $\mathbf { X } _ { t }$ . If $a _ { t } ^ { L } =$ local search, DE is used to optimize $\hat { f } _ { \mathrm { R B F } } ( \mathbf { x } )$ (if $a _ { t } ^ { H } = \operatorname { R B F } )$ or $\hat { f } _ { \mathrm { P R S } } ( \mathbf { x } )$ (if $a _ { t } ^ { H } \ = \ \bar { \mathrm { P R S } } )$ to get $\mathbf { X } _ { t }$ . Note that in this article, the DE local search is conducted with a population size of $N$ and a maximum number of $1 0 0 D + 1 0 0 0$ generations [25]; 
3) when $a _ { t } ^ { H }$ represents the KNN model, an offspring population with $N$ solutions is first generated using DE operators. Then, $\mathbf { X } _ { t }$ is selected based on L1-exploitation (if $a _ { t } ^ { L } = \operatorname { L } 1$ -exploitation) or L1-exploration (if $a _ { t } ^ { L } =$ L1-exploration) from these offspring solutions. 
Then, $\mathbf { X } _ { t }$ is evaluated and added into the database $\mathcal { D }$ (lines 23 and 24), and the FEs is increased by 1 (line 25). Finally, the number of selections for $a _ { t } ^ { H }$ and $a _ { t } ^ { L }$ is increased by 1 (line 26), and their values are updated by the TL-R method (line 27). 
When the maximum computational budget is consumed, the best solution in the database $\mathcal { D }$ is output as the final solution (line 30). In the following, we will give a detailed introduction to the two core components, TL-UCB and TL-R, sequentially. 
# B. TL-UCB
TL-UCB is the policy to select the high-level arm (surrogate model) and low-level arm (infill criterion). To choose a suitable high-level arm, the high-level UCB is executed, which 
Algorithm 2: $: ( a _ { t } ^ { H } , a _ { t } ^ { L } ) = \mathrm { T L } \mathrm { - } \mathrm { U C B } ( \mathcal { A } ^ { H } , \mathcal { A } ^ { L } , t , T ^ { H } , T ^ { L } , \mathcal { Q } ^ { H } ( t ) ,$ $Q ^ { L } ( t ) , \alpha )$ . 
# 1: Input:
2: High-level arms set: $\mathcal { A } ^ { H } = \{ a _ { 1 } ^ { H } , \ldots , a _ { 4 } ^ { H } \}$ 
3: Low-level arm set: $\mathcal { A } ^ { L } = \{ a _ { 1 } ^ { L } , \cdot \cdot \cdot , a _ { 8 } ^ { L } \}$ 
4: Current number of iterations: t 
5: Number of selection for each arm: $T ^ { H } ( t )$ and $T ^ { L } ( t )$ 
6: The value for each arm: $Q ^ { H } ( t )$ and $Q ^ { L } ( t )$ 
7: Parameter: $\alpha$ 
8: Set $A _ { a _ { 1 } ^ { H } } ^ { L } = \{ a _ { 1 } ^ { L } , a _ { 2 } ^ { L } \}$ , $A _ { a _ { \circ } ^ { H } } ^ { L } = \{ a _ { 3 } ^ { L } , a _ { 4 } ^ { L } \}$ 
$$
A _ {a _ {3} ^ {H}} ^ {L ^ {1}} = \{a _ {5} ^ {L}, a _ {6} ^ {L} \}, A _ {a _ {4} ^ {H}} ^ {L ^ {2}} = \{a _ {7} ^ {L}, a _ {8} ^ {L} \}
$$
9: 
10: $\begin{array} { r } { a _ { t } ^ { H } = \underset { a \in \mathcal { A } ^ { H } } { \arg \operatorname* { m a x } } \Big [ \mathcal { Q } _ { a } ^ { H } ( t ) + \sqrt { \frac { \alpha \ln ( t ) } { T _ { a } ^ { H } ( t ) } } \Big ] } \\ { a _ { t } ^ { L } = \underset { a \in \mathcal { A } _ { a _ { t } ^ { H } } ^ { L } } { \arg \operatorname* { m a x } } \Big [ \mathcal { Q } _ { a } ^ { L } ( t ) + \sqrt { \frac { \alpha \ln ( t ) } { T _ { a } ^ { L } ( t ) } } \Big ] } \\ { - } \end{array}$ a∈ALaH 
11: Output: High-level arm $a _ { t } ^ { H }$ and low-level arm $a _ { t } ^ { L }$ 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/44af3ef3673d409a5ffbfeacc910f274fcde0aa772c62b2d1bede96f5ee01044.jpg)


Fig. 2. Association relationship between the high-level arms and low-level ones.

is formulated as follows: 
$$
a _ {t} ^ {H} = \underset {a \in \mathcal {A} ^ {H}} {\arg \max } \left[ Q _ {a} ^ {H} (t) + \sqrt {\frac {\alpha \ln (t)}{T _ {a} ^ {H} (t)}} \right] \tag {22}
$$
where $Q _ { a } ^ { H } ( t )$ denotes the value of the high-level arm a at the $t \cdot$ -th iteration and $T _ { a } ^ { H } ( t )$ is the number of selections of the high-level arm a during the past $t - 1$ iterations, $t$ is the current iteration number, and $\alpha$ is a control parameter used to balance the tradeoff between exploiting well-performing arms and exploring rarely selected arms. 
When the high-level arm is determined, the low-level arm is picked up by the low-level UCB, which is defined as follows: 
$$
a _ {t} ^ {L} = \underset {a \in \mathcal {A} _ {a _ {t} ^ {H}} ^ {L}} {\arg \max } \left[ Q _ {a} ^ {L} (t) + \sqrt {\frac {\alpha \ln (t)}{T _ {a} ^ {L} (t)}} \right] \tag {23}
$$
where $Q _ { a } ^ { L } ( t )$ denotes the value of the low-level arm a in the $t$ -th iteration, $T _ { a } ^ { L } ( t )$ is the number of selections of the low-level arm a during the past $t \mathrm { ~ - ~ } 1$ iterations. Here, it should be noted that not each high-level arm can be associated with all low-level ones. Therefore, $\underset { - } { \mathcal { A } } _ { a _ { t } ^ { H } . } ^ { L }$ denotes the low-level arms that can be associated with the highlevel arm $a _ { t } ^ { H }$ . In this article, the association relationship between the high-level arms and low-level arms is shown in Fig. 2. 
Moreover, the pseudo-code of the TL-UCB is provided in Algorithm 2. 
1119 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024 
# C. TL-R
After determining the high-level arm (surrogate model) and low-level arm (infill criterion), the corresponding surrogate model is trained based on the current population $\mathcal { P }$ . Then, a new solution $\mathbf { X } _ { t }$ is obtained by the selected high-level and lowlevel arms cooperatively. To measure the optimization utility of the chosen $a _ { t } ^ { L }$ and $a _ { t } ^ { \bar { H } }$ , a TL-R is designed as follows. 
1) Low-Level Reward: 
$$
r _ {a _ {t} ^ {L}} = - \frac {1}{N} I (\mathbf {x} _ {t}) + \frac {N + 1}{N} \tag {24}
$$
where $N$ is the population size and $I ( { \bf x } _ { t } )$ denotes the ranking of $\mathbf { X } _ { t }$ in the current population $\mathcal { P }$ . Specifically, if $\mathbf { X } _ { t }$ is the current best solution, $I ( \mathbf { x } _ { t } ) = 1$ such that $r _ { a _ { t } ^ { L } } = 1$ . On the contrary, if $I ( { \bf x } _ { t } ) = N + 1$ such that $r _ { a _ { t } ^ { L } } = 0$ t . The low-level reward $r _ { a _ { t } ^ { L } }$ has four characteristics: 1) $r _ { a _ { t } ^ { L } }$ is bounded in [0, 1]; 2) $r _ { a _ { t } ^ { L } }$ ti s linearly proportional to the ranking $I ( { \bf x } _ { t } )$ of the newly generated solution $\mathbf { X } _ { t }$ , the better the ranking, the larger the reward; 3) the low-level rewards are nonsparse; and 4) it remains stationary to some extent throughout the entire optimization process. 
Based on the low-level reward $r _ { a _ { t } ^ { L } }$ , the value of $a _ { t } ^ { L }$ is updated as follows: 
$$
Q _ {a _ {t} ^ {L}} ^ {L} (t + 1) = \frac {T _ {a _ {t} ^ {L}} ^ {L} (t) Q _ {a _ {t} ^ {L}} ^ {L} (t) + r _ {a _ {t} ^ {L}}}{T _ {a _ {t} ^ {L}} ^ {L} (t) + 1}. \tag {25}
$$
It should be noted that the value of the high-level arm $a _ { t } ^ { H }$ is increased if and only if the low-level reward is larger than its current value [e.g., $\dot { r } _ { a _ { t } ^ { L } } \geq Q _ { a _ { t } ^ { L } } ^ { L } ( t ) ]$ . 
2) High-Level Reward: After the value $Q _ { a _ { t } ^ { L } } ^ { L }$ of the low-level arm $a _ { t } ^ { L }$ is updated, the reward for the high-level arm $a _ { t } ^ { H }$ is designed as follows: 
$$
r _ {a _ {t} ^ {H}} = \frac {Q _ {a _ {t} ^ {L}} ^ {L} (t + 1) - Q _ {a _ {t} ^ {L}} ^ {L} (t)}{\left| \mathcal {A} _ {a _ {t} ^ {H}} ^ {L} \right|} \tag {26}
$$
where $\mathcal { A } _ { a _ { t } ^ { H } } ^ { L }$ denotes the low-level arm set that is associated with the high-level arm $a _ { t } ^ { H }$ . Clearly, the high-level reward $r _ { a _ { t } ^ { H } }$ can be positive or negative. To be specific, if the value of the low-level arm $a _ { t } ^ { L }$ increases (e.g., $\bar { Q _ { a _ { t } ^ { L } } ^ { L } } ( t + 1 ) - Q _ { a _ { t } ^ { L } } ^ { L } ( t ) \geq 0 )$ , the reward of the high-level arm is positive. Otherwise, it is negative. 
Based on the high-level reward $r _ { a _ { t } ^ { H } }$ , the value of $a _ { t } ^ { H }$ is updated as follows: 
$$
Q _ {a _ {t} ^ {H}} ^ {H} (t + 1) = Q _ {a _ {t} ^ {H}} ^ {H} (t) + r _ {a _ {t} ^ {H}}. \tag {27}
$$
Clearly, if the high-level reward of $a _ { t } ^ { H }$ is positive, its value will be increased. Otherwise, its value will be decreased. It should be pointed out that the value $\underset { \ast } { \boldsymbol { Q } } _ { \boldsymbol { a } _ { t } ^ { H } . } ^ { H } ( t + 1 )$ of $a _ { t } ^ { H }$ is the mean tof the value of its associated low-level arms. To be specific, putting (26) into (27), we have 
$$
Q _ {a _ {t} ^ {H}} ^ {H} (t + 1) = \frac {\left| \mathcal {A} _ {a _ {t} ^ {H}} ^ {L} \right| \cdot Q _ {a _ {t} ^ {H}} ^ {H} (t) + Q _ {a _ {t} ^ {L}} ^ {L} (t + 1) - Q _ {a _ {t} ^ {L}} ^ {L} (t)}{\left| \mathcal {A} _ {a _ {t} ^ {H}} ^ {L} \right|}. \tag {28}
$$
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/c70897e711907717b0c6c3b2ba8774f2a3069c823f95adaad6c0b910cb043053.jpg)


Fig. 3. Numerical example of reward propagation in TL-R.

Al $\mathop { \bf g o r i t h m } _ { \tau } 3 : ( Q _ { a _ { t } ^ { H } } ^ { H } ( t + 1 ) , Q _ { a _ { t } ^ { L } } ^ { L } ( t + 1 ) ) = \mathrm { T L } { \cdot } \mathrm { R } ( N , \mathcal { P } , { \bf x } _ { t } , T _ { a _ { t } ^ { L } } ^ { L } ( t ) ,$ QLaLt (t), QHaHt (t)). 
# 1: Input:
2: Population size: $N$ 
3: Population: $\mathcal { P }$ 
4: New generated solution: $\mathbf { X } _ { t }$ 
5: Number of selections of $a _ { t } ^ { L }$ : $T _ { a _ { t } ^ { L } } ^ { L } ( t )$ 
6: The value of $a _ { t } ^ { L }$ and $a _ { t } ^ { H } \colon Q _ { a _ { t } ^ { L } } ^ { L } ( t ) \qquad $ and $Q _ { a _ { t } ^ { H } } ^ { H } ( t )$ 
7: Calculate the low-level reward $r _ { a _ { t } ^ { L } }$ of $a _ { t } ^ { L }$ as (24) 
8: Calculate $Q _ { a _ { t } ^ { L } } ^ { L } ( t + 1 )$ of $a _ { t } ^ { L }$ as (25) 
9: Calculate the high-level reward $r _ { a _ { t } ^ { H } }$ of $a _ { t } ^ { H }$ as (26) 
10: Calculate $Q _ { a _ { t } ^ { H } } ^ { H } ( t + 1 )$ of $a _ { t } ^ { H }$ as (27) 
11: Output: The updated values $Q _ { a _ { t } ^ { H } } ^ { H } ( t + 1 )$ and $Q _ { a _ { t } ^ { L } } ^ { L } ( t + 1 )$ of $a _ { t } ^ { H }$ and $a _ { t } ^ { L }$ 
Assume that $\begin{array} { r } { | \mathcal { A } _ { a _ { t } ^ { H } } ^ { L } | \cdot \mathcal { Q } _ { a _ { t } ^ { H } } ^ { H } ( t ) = \sum _ { a ^ { L } \in \mathcal { A } _ { a _ { t } ^ { H } } ^ { L } } \mathcal { Q } _ { a ^ { L } } ^ { L } ( t ) } \end{array}$ H QLaL (t) holds for the $t$ -th generation. Therefore 
$$
Q _ {a _ {t} ^ {H}} ^ {H} (t + 1) = \frac {\sum_ {a ^ {L} \in \mathcal {A} _ {a _ {t} ^ {H}} ^ {L}} Q _ {a ^ {L}} ^ {L} (t) + Q _ {a _ {t} ^ {L}} ^ {L} (t + 1) - Q _ {a _ {t} ^ {L}} ^ {L} (t)}{| \mathcal {A} _ {a _ {t} ^ {H}} ^ {L} |}. \tag {29}
$$
Since the value of the un-selected low-level arm is not updated, $\mathcal { Q } _ { a ^ { L } } ^ { L } ( t + 1 ) = \mathcal { Q } _ { a ^ { L } } ^ { L } ( t )$ , $( a ^ { L } \neq a _ { t } ^ { L } )$ . Finally, we have 
$$
Q _ {a _ {t} ^ {H}} ^ {H} (t + 1) = \frac {\sum_ {a ^ {L} \in \mathcal {A} _ {a _ {t} ^ {H}} ^ {L}} Q _ {a ^ {L}} ^ {L} (t + 1)}{| \mathcal {A} _ {a _ {t} ^ {H}} ^ {L} |}. \tag {30}
$$
Since the values of both the low-level arm and high-level arm are initialized to 0, the equation $\vert \mathcal { A } _ { a _ { t } ^ { H } } ^ { L } \vert \ \cdot \ Q _ { a _ { t } ^ { H } } ^ { \bar { H } } ( t ) =$ $\sum _ { a ^ { L } \in \mathcal { A } _ { a _ { t } ^ { H } } ^ { L } } \mathcal { Q } _ { a ^ { L } } ^ { L } ( t )$ holds in the initialization (i.e., $\ t \ = \ 1 ,$ ). tTherefore, in each generation, the value of $a _ { t + 1 } ^ { H }$ is exactly the mean of the values of its associated low-level arms. 
The pseudocode of TL-R is provided in Algorithm 3. In addition, Fig. 3 illustrates a numerical example of reward propagation in TL-R. In this figure, we assume that at the t-th generation, TL-UCB selects the high-level arm $a _ { t } ^ { H }$ and low-level arm $a _ { t } ^ { L }$ , and generates a solution $\mathbf { X } _ { t }$ by them cooperatively. Suppose the low-level reward is $r _ { a _ { t } ^ { L } } ~ = ~ 0 . 8$ , as computed using (24), and the value of the low-level arm is $\mathcal { Q } _ { a _ { t } ^ { L } } ^ { L } ( t + 1 ) = \bar { 0 . 6 } \bar { }$ a , as obtained from (25). Then, using (26), we can compute the reward of the high-level arm $r _ { a _ { t } ^ { H } } = 0 . 0 5$ , and finally, the value of the high-level arm $\mathcal { Q } _ { a _ { t } ^ { H } } ^ { H } ( t + 1 ) = 0 . 7$ is determined using (27). The path of the reward is indicated by solid red lines. 
1120 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
XIE et al.: SAEA 
Compared with other existing adaptive-model SAEAs [22], [23], [24], [25], the proposed AutoSAEA has the following differences. 
1) Although the model and infill criterion configuration are common, they differ from those used in the existing adaptive-model SAEAs. 
2) AutoSAEA can adaptively choose the surrogate model and infill criterion in an online manner. However, existing adaptive-model SAEAs are either only for model selection or infill criterion selection but do not study adaptive selection for both cooperatively. 
3) AutoSAEA formulates the surrogate model and infill criterion selection as a TL-MAB. Moreover, a TL-R and a TL-UCB designed to effectively capture good coupling behavior between the surrogate model and infill criterion. The adopted online learning strategy of AutoSAEA is obviously different from those of the existing adaptive SAEAs. 
# V. NUMERICAL EXPERIMENTS
# A. Benchmark Problems
Commonly used benchmark problems to evaluate the performance of the SAEAs are the Ellipsoid function, Rosenbrock function, Ackley function, Griewank function, Rastrigin function, and a few problems from CEC2005 competition [14], [15], [16], [18], [19], [32], [47], [56]. However, most of these problems are simple and hardly represent the complex characteristics of the real-world EOPs. Therefore, we first adopt CEC2005 [56] and CEC2015 problems [57] with $D = 1 0$ and $D = 3 0$ to evaluate the performance of our proposed AutoSAEA. Moreover, performance is measured using the function error metric $f ( \mathbf { x } _ { * } ) { - } f ( \mathbf { x } ^ { o } )$ , where $\mathbf { X } _ { * }$ denotes the best solution found by the algorithm, and $\mathbf { x } ^ { o }$ is the real optimum. To analyze the significant difference between the results obtained by all test algorithms, the Wilcoxon rank-sum test and Friedman test with the Hommel post-hoc procedure are conducted at a significant level of 0.05 [58], [59]. The symbols “+,” “=,” and “−” denote that AutoSAEA is statistically better than, competitive with, and worse than the compared algorithm, respectively. 
# B. Comparison With Existing SAEAs
In this section, comparison experiments are conducted between AutoSAEA and two traditional Bayesian optimization methods (i.e., GP-LCB [43] and GP-EI [40]) and eight state-of-the-art SAEAs (i.e., IKAEA (2021) [42], ESAO (2019) [32], CA-LLSO (2020) [47], TS-DDEO (2021) [31], SA-MPSO (2021) [16], SAMFEO (2022) [58], GL-SADE (2022) [18], and ESA (2022) [25]). The parameters of all compared algorithms remain the same as in their original papers. For the proposed AutoSAEA, parameters $N$ and $\alpha$ are set to 100 and 2.5, respectively. The statistical comparison results are obtained under 1000 FEs and 20 independent runs [15], [16], [18], [32], [47] on each problem. The mean and standard deviation of the obtained function error values for each algorithm on the CEC2005 10D problems, CEC2005 30D problems, CEC2015 10D problems, and CEC2015 30D problems are 
provided in Tables I–IV of the supplementary file, respectively, and the statistical results are shown in Table I. 
For the CEC2005 10D problems, AutoSAEA exhibits significant advantages over all compared algorithms. Specifically, it outperforms GP-LCB, GP-EI, IKAEA, ESAO, CA-LLSO, TS-DDEO, SA-MPSO, SAMFEO, GL-SADE, and ESA on 8, 9, 12, 13, 10, 14, 11, 11, 14, and 9 out of 15 problems, respectively. Moreover, it is only worse than CA-LLSO on 1 out of 15 problems. Based on the Friedman test, AutoSAEA ranks first and significantly outperforms all compared algorithms except GP-LCB and ESA. 
For the CEC2005 30D problems, AutoSAEA is significantly better than all compared algorithms in most cases. It outperforms GP-LCB, GP-EI, IKAEA, ESAO, CA-LLSO, TS-DDEO, SA-MPSO, SAMFEO, GL-SADE, and ESA on 7, 8, 12, 11, 10, 9, 12, 8, 12, and 8 out of 15 problems, respectively. Moreover, it is only worse than GP-LCB, GP-EI, ESAO, CA-LLSO, TS-DDEO, SA-MPSO, SAMFEO, GL-SADE, and ESA on 2, 2, 1, 1, 2, 1, 4, 2, and 3 out of 15 problems. AutoSAEA takes first place and is significantly better than all compared algorithms except GP-LCB, GP-EI, SAMFEO, and ESA according to the Friedman test results. 
For the CEC2015 10D problems, AutoSAEA achieves the best results in most cases. Specifically, it is significantly better than or at least as good as TS-DDEO and ESAO in all cases. Additionally, it is only worse than GP-LCB, GP-EI, IKAEA, CA-LLSO, SA-MPSO, SAMFEO, GL-SADE, and ESA on 2, 2, 1, 1, 1, 2, 2, and 2 out of 15 problems, respectively. Based on the statistical results of the Friedman test, AutoSAEA is the champion and significantly outperforms all compared SAEAs except GP-LCB, GP-EI, SAMFEO, and ESA. 
For the CEC2015 30D problems, AutoSAEA is still the winner in the overall comparison of algorithms. It is significantly better than GP-LCB, GP-EI, IKAEA, ESAO, CA-LLSO, TS-DDEO, SA-MPSO, SAMFEO, GL-SADE, and ESA on 5, 6, 12, 12, 9, 9, 12, 6, 12, and 6 out of 15 cases, respectively. Moreover, it is only worse than GP-LCB, GP-EI, IKAEA, ESAO, CA-LLSO, TS-DDEO, SA-MPSO, SAMFEO, GL-SADE, and ESA on 2, 3, 1, 0, 3, 2, 1, 1, 1, and 1 out of 15 problems, respectively. According to the statistical results of the Friedman test, AutoSAEA ranks first and is significantly better than IKAEA, ESAO, CA-LLSO, TS-DDEO, SA-MPSO, and GL-SADE. 
To evaluate the performance of SAEAs, it is necessary to test them on various problems with different characteristics. It is well known that different models and infill criteria are suitable for solving different problems [23], [24], [25], [30]. The proposed AutoSAEA does not focus on developing new models or infill criteria but aims to leverage the existing well-established ones to improve the performance of SAEA. Therefore, although the surrogate model and infill criterion adopted in AutoSAEA are common configurations, the reason why it performs better than other state-of-the-art SAEAs is explained as follows. 
1) Compared With Single-Model SAEAs: Single-model SAEAs use a fixed surrogate model and infill criterion for all problems. While these algorithms can solve specific problems well, they often perform poorly on 
1121 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024 

TABLE I COMPARISON RESULTS BETWEEN AUTOSAEA AND ITS COMPETITORS AND THE RANKING OF EACH ALGORITHM ON CEC2005 PROBLEMS AND CEC2015 PROBLEMS WITH $D = 1 0$ AND $D = 3 0$

<table><tr><td colspan="12">CEC2005 10D problems</td></tr><tr><td></td><td>GP-LCB</td><td>GP-EI</td><td>IKAEA</td><td>ESAO</td><td>CA-LLSO</td><td>TS-DDEO</td><td>SA-MPSO</td><td>SAMFEO</td><td>GL-SADE</td><td>ESA</td><td>AutoSAEA</td></tr><tr><td>+/-/ -</td><td>8/7/0</td><td>9/6/0</td><td>12/3/0</td><td>13/2/0</td><td>10/4/1</td><td>14/1/0</td><td>11/4/0</td><td>11/4/0</td><td>14/1/0</td><td>9/6/0</td><td>NA</td></tr><tr><td>Ranking</td><td>4.2</td><td>5.0</td><td>7.3</td><td>8.7</td><td>6.3</td><td>7.2</td><td>6.7</td><td>5.9</td><td>7.8</td><td>4.1</td><td>1.9</td></tr><tr><td>p-value</td><td>0.12</td><td>0.03</td><td>0.0001</td><td>0</td><td>0.001</td><td>0.0001</td><td>0.0004</td><td>0.004</td><td>0</td><td>0.12</td><td>NA</td></tr><tr><td colspan="12">CEC2005 30D problems</td></tr><tr><td></td><td>GP-LCB</td><td>GP-EI</td><td>IKAEA</td><td>ESAO</td><td>CA-LLSO</td><td>TS-DDEO</td><td>SA-MPSO</td><td>SAMFEO</td><td>GL-SADE</td><td>ESA</td><td>AutoSAEA</td></tr><tr><td>+/-/ -</td><td>7/6/2</td><td>8/5/2</td><td>12/3/0</td><td>11/3/1</td><td>10/4/1</td><td>9/4/2</td><td>12/2/1</td><td>8/3/4</td><td>12/1/2</td><td>8/4/3</td><td>NA</td></tr><tr><td>Ranking</td><td>4.7</td><td>4.6</td><td>8.8</td><td>8.0</td><td>6.7</td><td>6.1</td><td>7.5</td><td>4.1</td><td>7.8</td><td>5.0</td><td>2.7</td></tr><tr><td>p-value</td><td>0.30</td><td>0.30</td><td>0</td><td>0.0001</td><td>0.005</td><td>0.03</td><td>0.0004</td><td>0.30</td><td>0.0002</td><td>0.22</td><td>NA</td></tr><tr><td colspan="12">CEC2015 10D problems</td></tr><tr><td></td><td>GP-LCB</td><td>GP-EI</td><td>IKAEA</td><td>ESAO</td><td>CA-LLSO</td><td>TS-DDEO</td><td>SA-MPSO</td><td>SAMFEO</td><td>GL-SADE</td><td>ESA</td><td>AutoSAEA</td></tr><tr><td>+/-/ -</td><td>7/6/2</td><td>7/6/2</td><td>11/3/1</td><td>14/1/0</td><td>12/2/1</td><td>11/4/0</td><td>12/2/1</td><td>11/2/2</td><td>12/1/2</td><td>10/3/2</td><td>NA</td></tr><tr><td>Ranking</td><td>4.5</td><td>4.1</td><td>6.3</td><td>9.1</td><td>7.7</td><td>7.7</td><td>5.9</td><td>5.0</td><td>7.7</td><td>5.1</td><td>2.6</td></tr><tr><td>p-value</td><td>0.23</td><td>0.23</td><td>0.01</td><td>0</td><td>0.0002</td><td>0.002</td><td>0.03</td><td>0.18</td><td>0.0002</td><td>0.18</td><td>NA</td></tr><tr><td colspan="12">CEC2015 30D problems</td></tr><tr><td></td><td>GP-LCB</td><td>GP-EI</td><td>IKAEA</td><td>ESAO</td><td>CA-LLSO</td><td>TS-DDEO</td><td>SA-MPSO</td><td>SAMFEO</td><td>GL-SADE</td><td>ESA</td><td>AutoSAEA</td></tr><tr><td>+/-/ -</td><td>5/8/2</td><td>6/6/3</td><td>12/2/1</td><td>12/3/0</td><td>9/3/3</td><td>9/4/2</td><td>12/2/1</td><td>6/8/1</td><td>12/2/1</td><td>6/8/1</td><td>NA</td></tr><tr><td>Ranking</td><td>3.4</td><td>4.1</td><td>8.4</td><td>8.3</td><td>7.0</td><td>5.9</td><td>7.4</td><td>5.2</td><td>9.2</td><td>4.3</td><td>2.9</td></tr><tr><td>p-value</td><td>0.74</td><td>0.74</td><td>0.0001</td><td>0.0001</td><td>0.004</td><td>0.05</td><td>0.001</td><td>0.26</td><td>0</td><td>0.74</td><td>NA</td></tr></table>
others. Therefore, when evaluating their performance on a sufficiently diverse set of problems, single-model SAEAs tend to perform worse than multiple-model SAEAs and adaptive-model SAEAs [18], [19], [35]. 
2) Compared With Multiple-Model SAEAs: While multiplemodel SAEAs can leverage the advantages of different models and infill criteria, they cannot identify which ones are most suitable for a given problem. This often leads to high consumption of computational resources on ineffective models and infill criteria [23]. 
3) Compared With Adaptive-Model SAEAs: Adaptivemodel SAEAs aim to not only exploit the advantages of different models and infill criteria but also identify the most suitable ones for a given problem. However, the performance of adaptive SAEAs depends on the specific model and infill criterion configuration used, as well as the adaptive selection strategy. Compared to existing adaptive-model SAEAs, AutoSAEA not only enriches the model/infill criterion configuration but also adopts a more effective adaptive selection strategy (i.e., TL-MAB) to select them in a hierarchical, coupled way. Therefore, AutoSAEA can outperform existing state-of-the-art adaptive-model SAEAs. 
Due to page limitations, we have included the convergence profiles of all tested algorithms, as well as the computational complexity of AutoSAEA and running times of all compared algorithms, in the supplementary file. 
# C. Adaptive Behavior Analysis
AutoSAEA formulates the surrogate model and infill criterion selection as a TL-MAB problem and designs a TL-R and a TL-UCB to address it. To demonstrate the effectiveness of the adaptive surrogate model and infill criterion selection used in AutoSAEA, we first compare it with eleven variants. 
1) V-CA1 to V-CA8: AutoSAEA uses the fixed combinatorial arms in $\mathcal { C A }$ during the entire optimization process. V-CAi denotes that it uses the ith $( i \ = \ 1 , \ldots , 8 )$ combinatorial arm. 
2) V-Random: The surrogate model and infill criterion are randomly selected. 
3) V-SLUCB: AutoSAEA uses a single-level UCB and a single-level reward to adaptively choose the combinatorial arms from $\mathcal { C A }$ . 
4) V-Q: AutoSAEA uses the $Q$ -learning method [25] to adaptively select the combinatorial arms from $\mathcal { C A }$ in each iteration. 
The experiments are conducted on the CEC2005 10D and 30D problems, with all experimental setups kept the same as in the previous section. The mean and standard deviation of the function error value for each variant are provided in Tables V and VI of the supplementary file, and the statistical results are shown in Table II. 
For the CEC2005 10D problems, the Wilcoxon rank-sum test indicates that AutoSAEA significantly outperforms its variants. Specifically, it outperforms or competes with all variants on all problems, except that it is worse than VCA-5 and V-Q on one problem each out of 15. Overall, AutoSAEA achieves the best ranking and significantly outperforms VCA-2, VCA-4, VCA-5, VCA-6, VCA-7, and VCA-8 according to the Friedman test. 
For the CEC2005 30D problems, based on the Wilcoxon rank-sum test, AutoSAEA is significantly better than all variants in most cases, and it is only outperformed by V-CA1, V-CA2, VCA-5, V-CA7, V-Random, and V-Q on 2, 2, 2, 2, 2, and 2 out of 15 problems, respectively. 
While AutoSAEA exhibits remarkable advantages over some of its variants, such as V-CA4, V-CA5, V-CA6, V-CA7, and V-CA8, it is worth noting that there is no significant difference between AutoSAEA and some of its variants, namely, V-CA1, V-CA2, V-CA3, V-Random, V-SLUCB, and V-Q. This indicates that AutoSAEA does not have advantages over its variants on some problems. The reasons behind this are provided as follows, based on the detailed analysis of the results. 
1) For some problems, such as CEC2005 F8 $( D = 1 0 )$ ), CEC2005 F15 $\textit { D } = \ 1 0 )$ ), CEC2005 F8 $\textit { D } = \ 3 0 )$ , CEC2005 F11 $( D = 3 0 )$ ), and CEC2005 F14 $( D = 3 0 )$ ), no models and infill criteria work effectively, so AutoSAEA cannot address them well. 
1122 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
XIE et al.: SAEA 

TABLE II COMPARISON RESULTS BETWEEN AUTOSAEA AND ITS VARIANTS AND THEIR RANKING ON CEC2005 10D AND 30D PROBLEMS

<table><tr><td colspan="12">CEC2005 10D problems</td></tr><tr><td></td><td>V-CA1</td><td>V-CA2</td><td>V-CA3</td><td>V-CA4</td><td>V-CA5</td><td>V-CA6</td><td>V-CA7</td><td>V-CA8</td><td>V-Random</td><td>V-SLUCB</td><td>V-Q</td></tr><tr><td>+/-/</td><td>8/7/0</td><td>9/6/0</td><td>8/7/0</td><td>14/1/0</td><td>13/1/1</td><td>14/1/0</td><td>13/2/0</td><td>14/1/0</td><td>8/7/0</td><td>2/13/0</td><td>11/3/1</td></tr><tr><td>Ranking</td><td>5.4</td><td>6.6</td><td>5.3</td><td>9.1</td><td>7.3</td><td>8.9</td><td>9.5</td><td>10.9</td><td>4.0</td><td>3.1</td><td>5.1</td></tr><tr><td>p-value</td><td>0.13</td><td>0.01</td><td>0.13</td><td>0</td><td>0.002</td><td>0</td><td>0</td><td>0</td><td>0.47</td><td>0.61</td><td>0.13</td></tr><tr><td colspan="12">CEC2005 30D problems</td></tr><tr><td></td><td>V-CA1</td><td>V-CA2</td><td>V-CA3</td><td>V-CA4</td><td>V-CA5</td><td>V-CA6</td><td>V-CA7</td><td>V-CA8</td><td>V-Random</td><td>V-SLUCB</td><td>V-Q</td></tr><tr><td>+/-/</td><td>7/6/2</td><td>8/5/2</td><td>8/7/0</td><td>11/4/0</td><td>11/2/2</td><td>13/2/0</td><td>11/2/2</td><td>13/2/0</td><td>9/4/2</td><td>4/11/0</td><td>9/4/2</td></tr><tr><td>Ranking</td><td>4.2</td><td>4.9</td><td>5.5</td><td>8.5</td><td>8.5</td><td>11.3</td><td>8.2</td><td>10.7</td><td>4.8</td><td>3.5</td><td>6.1</td></tr><tr><td>p-value</td><td>0.41</td><td>0.38</td><td>0.18</td><td>0</td><td>0</td><td>0</td><td>0.0001</td><td>0</td><td>0.38</td><td>0.51</td><td>0.003</td></tr></table>
2) For some problems, such as CEC2005 F9 $( D = 1 0 )$ ), CEC2005 F10 $( D = 1 0 )$ ), CEC2005 F12 $\mathit { D } = \mathit { 1 0 } )$ , CEC2005 F2 $ { \boldsymbol { D } } = 3 0 $ ), and CEC2005 F3 $( D = 3 0 )$ ), they can be addressed well by various models and infill criteria, so AutoSAEA does not have significant advantages over its some variants on these problems. 
To visually demonstrate the adaptive behavior of our proposed AutoSAEA, we show the selected high-level arm (surrogate model) and low-level arm (infill criterion) during the optimization process of AutoSAEA in solving four representative problems, namely, CEC2005 F4 $( D = 1 0 )$ ), CEC2005 F10 $\mathit { D } = 1 0$ ), CEC2005 F2 $( D = 3 0 )$ ), and CEC2005 F12 $( D = 3 0 )$ ), in Fig. 4. Additionally, we count the number of times each surrogate model and infill criterion is chosen in Fig. 5. From Figs. 4 and 5, we observe the following. 
1) For CEC2005 F4 $\mathit { D } = 1 0 $ ), the PRS with prescreening is selected most frequently. This is because this problem is unimodal, its landscape is smooth, and its dimensionality is small, making PRS a good choice for approximating this problem. 
2) For CEC2005 F10 $( D ~ = ~ 1 0 )$ , the algorithm prefers different models at different optimization stages. Specifically, the whole optimization procedure can be roughly divided into five stages, and GP, KNN, RBF, PRS, and KNN are, respectively, selected most frequently. 
3) For CEC2005 F2 $\left( D \ = \ 3 0 \right)$ , the GP model with EI and LCB is selected most frequently, especially in later stages. Interestingly, GP always combines LCB when $6 5 0 < F E s < 7 5 0$ . 
4) For CEC2005 F12 $ { \boldsymbol { D } } = 3 0 $ ), GP with LCB or EI is mainly chosen when $1 0 0 < F E s < 3 5 0$ . However, when $3 5 0 < F E s < 5 5 0$ , the RBF model with prescreening or local search is preferred, and the GP model is no longer selected. When $5 5 0 < F E s < 1 0 0 0$ , RBF and GP are the most and second most commonly chosen models, respectively. 
In summary, for different problems and different optimization stages, the selected surrogate model and infill criterion are different. Therefore, combined with the outstanding performance, the effectiveness of the surrogate model and infill criterion auto-configuration in AutoSAEA has been verified. 
# D. Oil Reservoir Production Optimization Problem
AutoSAEA is applied to an oil reservoir production problem in the section to demonstrate its advantage further. This 

TABLE III RESULT OF GP-LCB, GP-EI, IKAEA, ESAO, TS-DDEO, SA-MPSO, GL-SADE, ESA, AND AUTOSAEA ON THE OIL RESERVOIR PRODUCTION OPTIMIZATION PROBLEM. THE UNIT OF TIME IS SECONDS

<table><tr><td>Algorithm</td><td>Mean(std)</td><td>Median</td><td>Worst</td><td>Best</td><td>Time</td></tr><tr><td>GP-LCB</td><td>1.34e+04(2.2e+02)=</td><td>1.35e+04</td><td>1.31e+00</td><td>1.36e+04</td><td>2.58e+05</td></tr><tr><td>GP-EI</td><td>1.34e+04(1.7e+02)=</td><td>1.35e+04</td><td>1.32e+04</td><td>1.36e+04</td><td>2.18e+05</td></tr><tr><td>IKAEA</td><td>1.28e+04(1.9e+02)+</td><td>1.27e+04</td><td>1.26e+04</td><td>1.31e+04</td><td>2.45e+05</td></tr><tr><td>ESAO</td><td>1.27e+04(6.4e+02)+</td><td>1.30e+04</td><td>1.16e+04</td><td>1.32e+04</td><td>2.64e+05</td></tr><tr><td>TS-DDEO</td><td>1.34e+04(1.7e+02)=</td><td>1.34e+04</td><td>1.31e+04</td><td>1.36e+04</td><td>3.04e+05</td></tr><tr><td>SA-MPSO</td><td>1.13e+04(1.3e+02)+</td><td>1.13e+04</td><td>1.12e+04</td><td>1.15e+04</td><td>3.52e+05</td></tr><tr><td>GL-SADE</td><td>1.31e+04(3.8e+02)=</td><td>1.33e+04</td><td>1.28e+04</td><td>1.35e+04</td><td>3.48e+05</td></tr><tr><td>ESA</td><td>1.34e+04(1.7e+02)=</td><td>1.33e+04</td><td>1.33e+04</td><td>1.37e+04</td><td>2.98e+05</td></tr><tr><td>AutoSAEA</td><td>1.36e+04(3.0e+02)</td><td>1.35e+04</td><td>1.32e+04</td><td>1.41e+04</td><td>2.13e+05</td></tr></table>
production optimization problem aims to find the optimal control parameters for the water-injection-rate of injection wells and fluid-production-rate of production wells in the expected life to maximize the net present value (NPV). In general, the oil reservoir production optimization problem can be expressed in the following form [60]: 
$$
\begin{array}{l} \max  \sum_ {t = 1} ^ {T} \Delta_ {t} r _ {o} Q _ {o, t} (\mathbf {x}) - r _ {w} Q _ {w, t} (\mathbf {x}) - r _ {i} Q _ {i, t} (\mathbf {x}) \\ \text {s . t .} 0 \leq x _ {i, t} \leq 5 0 0, i = 1, \dots , 8, t = 1, \dots , 5 \tag {31} \\ \end{array}
$$
where $\mathbf { x } ~ = ~ ( x _ { 1 , 1 } , \ldots , x _ { 1 , 5 } , \ldots , x _ { 8 , 1 } , \ldots , x _ { 8 , 5 } ) ^ { \top }$ is the decision vector, and $x _ { i , t }$ denotes the flow rate of the ith well at the $t$ -th time step. Therefore, 40 variables need to be optimized. The lower and upper bounds of each decision variable are set to 0 STB/day and 500 STB/day, respectively. $T = 5$ denotes the number of time steps. $\Delta _ { t } = 7 2 0$ is the time of the t-th time step. ro, $r _ { w }$ , and $r _ { i }$ , respectively, denote the oilproduction revenue, water-production cost, and water-injection cost, which are set to 20 USD/STB, 1 USD/STB, and 3 USD/STB, respectively. $Q _ { o , t }$ , $Q _ { w , t } .$ , and $Q _ { i , t }$ denote the oilproduction rate, water-production rate, and injection-flow-rate at the $t \cdot$ -th time step, respectively, which are calculated by a numerical simulator. In this case, the Egg model [61], including eight water-injection wells and four production wells, is selected as the reservoir simulator. More information about the Egg model can be found in [61]. In this article, the MRST toolbox [62] is adopted to conduct the Egg model. Note that it takes about 40 s to evaluate one solution using the simulator. 
To demonstrate the performance of AutoSAEA in solving the oil reservoir production optimization problem, eight existing SAEAs are used to test. To make a fair comparison, the initial number of solutions and the total number of FEs for all algorithms are set to 100 and 
1123 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/779a804971078aef57bb63776acb2241e0f143b61aff973225d2064926f5861d.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/e89f0ee2d35b15a1574a4aec94f87c4472117d2aeaaf49feacde077eaf9d1dbe.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/d0d6d91b6eca8bc46fb22d4e687dcc913a7abe9e730dcd1c204ce1be8c02b83b.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/4d5c68ecb25f030b3950494f216a4719eb536b9475b6a86b2414879b450375e6.jpg)


Fig. 4. Selected surrogate model and infill criterion during the optimization process of AutoSAEA on CEC2005 F4 $D = 1 0$ ), CEC2005 F10 $D = 1 0$ ), CEC2005 F2 $( D = 3 0$ ), and CEC2005 F12 $ { \boldsymbol { D } } = 3 0 $ ).

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/15178e18a089a63c4069ef9cb774d288d575bb334970a83625401957b6608284.jpg)


Fig. 5. Number of times each surrogate model and infill criterion is chosen during the optimization process of AutoSAEA on CEC2005 F4 $( D = 1 0 )$ ), CEC2005 F10 $( D = 1 0 )$ ), CEC2005 F2 $( D = 3 0 )$ ), and CEC2005 F12 $ { D } = 3 0 $ ).

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/b9ffb5a03f8b0b3fd638067aa53ba8ae53f7d2c78b8f5494fb270af7dc8e50c9.jpg)


Fig. 6. Average convergence profile of the median NPV value and its interquartile ranges of GP-LCB, GP-EI, IKAEA, ESAO, TS-DDEO, SA-MPSO, GL-SADE, ESA, and AutoSAEA on the oil reservoir production optimization problem.

1000, respectively. Moreover, each algorithm conducts five independent runs. 
The statistical results are provided in Table III. It can be seen that AutoSAEA can get the best average performance. Based on the Wilcoxon rank-sum test, AutoSAEA is significantly better than IKAEA, ESAO, and SA-MPSO and performs competitively with GP-LCB, GP-EI, ESA, TS-DDEO, and GL-SADE. Moreover, the running time of AutoSAEA is also the lowest, which shows the better computational efficiency of AutoSAEA compared with its competitors. Besides, the average convergence performance of each algorithm on this problem is plotted in Fig. 6, from which we can find 
that AutoSAEA also has some advantages in solving this oil reservoir production optimization problem. 
# VI. CONCLUSION
The surrogate model and infill criterion are vital to the performance of SAEAs. To enhance the ability of SAEA to solve various EOPs, this article proposes a TL-MAB to cooperatively choose the surrogate model and infill criterion in an online manner. To achieve this, a TL-R mechanism is defined to measure the optimization utility of the surrogate model and infill criterion in a hierarchical and coupled manner. Additionally, a TL-UCB strategy is designed to choose them cooperatively and adaptively. With these adaptive selection components, an auto SAEA has been proposed, called AutoSAEA. The performance of AutoSAEA has been demonstrated by comparing it with several state-of-the-art SAEAs on two sets of benchmark problems (CEC2005 and CEC2015) and a real-world oil reservoir production optimization problem. Furthermore, the adaptive behavior of AutoSAEA has been numerically verified, and the sensitivity of its control parameters has been analyzed. 
In the future, we plan to extend the adaptive surrogate model and infill criterion selection method to expensive multiobjective optimization. Moreover, we will use the proposed AutoSAEA to tackle other real-world EOPs. 
1124 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
XIE et al.: SAEA 
# REFERENCES


[1] Y. Lian, A. Oyama, and M.-S. Liou, “Progress in design optimization using evolutionary algorithms for aerodynamic problems,” Progr. Aerosp. Sci., vol. 46, nos. 5–6, pp. 199–223, 2010. 




[2] T. W. Simpson, A. J. Booker, D. Ghosh, A. A. Giunta, P. N. Koch, and R.-J. Yang, “Approximation methods in multidisciplinary analysis and optimization: A panel discussion,” Struct. Multidiscipl. Optim., vol. 27, no. 5, pp. 302–313, 2004. 




[3] Q. Chen, G. Li, Q. Zhang, Q. Tang, and G. Zhang, “Optimal design of passive control of space tethered-net capture system,” IEEE Access, vol. 7, pp. 131383–131394, 2019. 




[4] Z. Wang, Y.-S. Ong, J. Sun, A. Gupta, and Q. Zhang, “A generator for multiobjective test problems with difficult-to-approximate Pareto front boundaries,” IEEE Trans. Evol. Comput., vol. 23, no. 4, pp. 556–571, Aug. 2019. 




[5] W. Gao, Z. Wei, M. Gong, and G. G. Yen, “Solving expensive multimodal optimization problem by a decomposition differential evolution algorithm,” IEEE Trans. Cybern., vol. 53, no. 4, pp. 2236–2246, Apr. 2023. 




[6] G. Li and Q. Zhang, “Multiple penalties and multiple local surrogates for expensive constrained optimization,” IEEE Trans. Evol. Comput., vol. 25, no. 4, pp. 769–778, Aug. 2021. 




[7] G. Li, L. Xie, Z. Wang, H. Wang, and M. Gong, “Evolutionary algorithm with individual-distribution search strategy and regression-classification surrogates for expensive optimization,” Inf. Sci., vol. 634, pp. 423–442, Jul. 2023. 




[8] H. Wang, H. Xu, and Z. Zhang, “High-dimensional multi-objective Bayesian optimization with block coordinate updates: Case studies in intelligent transportation system,” IEEE Trans. Intell. Transp. Syst., early access, Feb. 7, 2023, doi: 10.1109/TITS.2023.3241069. 




[9] T. Goel, R. T. Hafkta, and W. Shyy, “Comparing error estimation measures for polynomial and kriging approximation of noise-free functions,” Struct. Multidiscip. Optim., vol. 38, no. 5, pp. 429–442, 2009. 




[10] L. M. Zouhal and T. Denoeux, “An evidence-theoretic k-NN rule with parameter optimization,” IEEE Trans. Syst., Man, Cybern. C, Appl. Rev., vol. 28, no. 2, pp. 263–271, Mar. 1998. 




[11] S. M. Clarke, J. H. Griebsch, and T. W. Simpson, “Analysis of support vector regression for approximation of complex engineering analyses,” J. Mech. Design, vol. 127, no. 6, pp. 1077–1087, 2005. 




[12] Y. Jin, M. Olhofer, and B. Sendhoff, “A framework for evolutionary optimization with approximate fitness functions,” IEEE Trans. Evol. Comput., vol. 6, no. 5, pp. 481–494, Oct. 2002. 




[13] B. Liu, Q. Zhang, and G. G. E. Gielen, “A Gaussian process surrogate model assisted evolutionary algorithm for medium scale expensive optimization problems,” IEEE Trans. Evol. Comput., vol. 18, no. 2, pp. 180–192, Apr. 2014. 




[14] J. Tian, Y. Tan, J. Zeng, C. Sun, and Y. Jin, “Multiobjective infill criterion driven Gaussian process-assisted particle swarm optimization of highdimensional expensive problems,” IEEE Trans. Evol. Comput., vol. 23, no. 3, pp. 459–472, Jul. 2019. 




[15] F. Li, X. Cai, L. Gao, and W. Shen, “A surrogate-assisted multiswarm optimization algorithm for high-dimensional computationally expensive problems,” IEEE Trans. Cybern., vol. 51, no. 3, pp. 1390–1402, Mar. 2021. 




[16] Y. Liu, J. Liu, and Y. Jin, “Surrogate-assisted multipopulation particle swarm optimizer for high-dimensional expensive optimization,” IEEE Trans. Syst., Man, Cybern., Syst., vol. 52, no. 7, pp. 4671–4684, Jul. 2022. 




[17] J. Liu, Y. Wang, G. Sun, and T. Pang, “Multisurrogate-assisted ant colony optimization for expensive optimization problems with continuous and categorical variables,” IEEE Trans. Cybern., vol. 52, no. 11, pp. 11348–11361, Nov. 2022. 




[18] W. Wang, H.-L. Liu, and K. C. Tan, “A surrogate-assisted differential evolution algorithm for high-dimensional expensive optimization problems,” IEEE Trans. Cybern., vol. 53, no. 4, pp. 2685–2697, Apr. 2023. 




[19] H. Wang, Y. Jin, and J. Doherty, “Committee-based active learning for surrogate-assisted particle swarm optimization of expensive problems,” IEEE Trans. Cybern., vol. 47, no. 9, pp. 2664–2677, Sep. 2017. 




[20] T. Sonoda and M. Nakata, “Multiple classifiers-assisted evolutionary algorithm based on decomposition for high-dimensional multi-objective problems,” IEEE Trans. Evol. Comput., vol. 26, no. 6, pp. 1581–1595, Dec. 2022. 




[21] X. Wu, Q. Lin, J. Li, K. C. Tan, and V. C. Leung, “An ensemble surrogate-based coevolutionary algorithm for solving large-scale expensive optimization problems,” IEEE Trans. Cybern., early access, Sep. 16, 2022, doi: 10.1109/TCYB.2022.3200517. 




[22] M. W. Hoffman, E. Brochu, and N. De Freitas, “Portfolio allocation for Bayesian optimization,” in Proc. UAI, 2011, pp. 327–336. 




[23] Z. Song, H. Wang, C. He, and Y. Jin, “A Kriging-assisted twoarchive evolutionary algorithm for expensive many-objective optimization,” IEEE Trans. Evol. Comput., vol. 25, no. 6, pp. 1013–1027, Dec. 2021. 




[24] Z. Liu, H. Wang, and Y. Jin, “Performance indicator-based adaptive model selection for offline data-driven multiobjective evolutionary optimization,” IEEE Trans. Cybern., early access, May 13, 2022, doi: 10.1109/TCYB.2022.3170344. 




[25] H. Zhen, W. Gong, and L. Wang, “Evolutionary sampling agent for expensive problems,” IEEE Trans. Evol. Comput., vol. 27, no. 3, pp. 716–727, Jun. 2023. 




[26] Q. Zhang, W. Liu, E. Tsang, and B. Virginas, “Expensive multiobjective optimization by MOEA/D with Gaussian process model,” IEEE Trans. Evol. Comput., vol. 14, no. 3, pp. 456–474, Jun. 2010. 




[27] J. E. Dennis and V. Torczon, “Managing approximation models in optimization,” Multidiscip. Design Optim. State-Art, vol. 5, pp. 330–347, Nov. 1998. 




[28] G. Li, Q. Zhang, Q. Lin, and W. Gao, “A three-level radial basis function method for expensive optimization,” IEEE Trans. Cybern., vol. 52, no. 7, pp. 5720–5731, Jul. 2022. 




[29] J. Zhang, A. Zhou, and G. Zhang, “A multiobjective evolutionary algorithm based on decomposition and preselection,” in Bio-Inspired Computing—Theories and Applications. Heidelberg, Germany: Springer, 2015, pp. 631–642. 




[30] M. Yu, X. Li, and J. Liang, “A dynamic surrogate-assisted evolutionary algorithm framework for expensive structural optimization,” Struct. Multidiscip. Optim., vol. 61, no. 2, pp. 711–729, 2020. 




[31] H. Zhen, W. Gong, L. Wang, F. Ming, and Z. Liao, “Two-stage datadriven evolutionary optimization for high-dimensional expensive problems,” IEEE Trans. Cybern., vol. 53, no. 4, pp. 2368–2379, Apr. 2023. 




[32] X. Wang, G. G. Wang, B. Song, P. Wang, and Y. Wang, “A novel evolutionary sampling assisted optimization method for high-dimensional expensive problems,” IEEE Trans. Evol. Comput., vol. 23, no. 5, pp. 815–827, Oct. 2019. 




[33] X. Cai, L. Gao, and X. Li, “Efficient generalized surrogate-assisted evolutionary algorithm for high-dimensional expensive problems,” IEEE Trans. Evol. Comput., vol. 24, no. 2, pp. 365–379, Apr. 2020. 




[34] Z. Wang et al., “Multiobjective optimization-aided decision-making system for large-scale manufacturing planning,” IEEE Trans. Cybern., vol. 52, no. 8, pp. 8326–8339, Aug. 2022. 




[35] D. Guo, Y. Jin, J. Ding, and T. Chai, “Heterogeneous ensemble-based infill criterion for evolutionary multiobjective optimization of expensive problems,” IEEE Trans. Cybern., vol. 49, no. 3, pp. 1012–1025, Mar. 2019. 




[36] J.-Y. Li, Z.-H. Zhan, H. Wang, and J. Zhang, “Data-driven evolutionary algorithm with perturbation-based ensemble surrogates,” IEEE Trans. Cybern., vol. 51, no. 8, pp. 3925–3937, Aug. 2021. 




[37] L. Kocsis and C. Szepesvári, “Bandit based Monte-Carlo planning,” in Proc. Eur. Conf. Mach. Learn., 2006, pp. 282–293. 




[38] S. Pandey, D. Chakrabarti, and D. Agarwal, “Multi-armed bandit problems with dependent arms,” in Proc. 24th Int. Conf. Mach. Learn., 2007, pp. 721–728. 




[39] E. Carlsson, D. Dubhashi, and F. D. Johansson, “Thompson sampling for bandits with clustered arms,” 2021, arXiv:2109.01656. 




[40] D. R. Jones, M. Schonlau, and W. J. Welch, “Efficient global optimization of expensive black-box functions,” J. Global Optim., vol. 13, no. 4, p. 455, 1998. 




[41] P. I. Frazier, “A tutorial on Bayesian optimization,” 2018, arXiv:1807.02811. 




[42] D. Zhan and H. Xing, “A fast Kriging-assisted evolutionary algorithm based on incremental learning,” IEEE Trans. Evol. Comput., vol. 25, no. 5, pp. 941–955, Oct. 2021. 




[43] D. D. Cox and S. John, “A statistical method for global optimization,” in Proc. IEEE Int. Conf. Syst., Man, Cybern., 1992, pp. 1241–1246. 




[44] J. Zhang, A. Zhou, and G. Zhang, “A classification and Pareto domination based multiobjective evolutionary algorithm,” in Proc. IEEE Congr. Evol. Comput. (CEC), 2015, pp. 2883–2890. 




[45] C. Sun, Y. Jin, R. Cheng, J. Ding, and J. Zeng, “Surrogate-assisted cooperative swarm optimization of high-dimensional expensive problems,” IEEE Trans. Evol. Comput., vol. 21, no. 4, pp. 644–660, Aug. 2017. 




[46] Y. Haibo, T. Ying, Z. Jianchao, S. Chaoli, and J. Yaochu, “Surrogate-assisted hierarchical particle swarm optimization,” Inf. Sci., vols. 454–455, pp. 59–72, Jul. 2018. 


1125 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 
IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 28, NO. 4, AUGUST 2024 


[47] F.-F. Wei et al., “A classifier-assisted level-based learning swarm optimizer for expensive optimization,” IEEE Trans. Evol. Comput., vol. 25, no. 2, pp. 219–233, Apr. 2020. 




[48] Q. Liu, R. Cheng, Y. Jin, M. Heiderich, and T. Rodemann, “Reference vector-assisted adaptive model management for surrogate-assisted manyobjective optimization,” IEEE Trans. Syst., Man, Cybern., Syst., vol. 52, no. 12, pp. 7760–7773, Dec. 2022. 




[49] G. Li, Q. Zhang, J. Sun, and Z. Han, “Radial basis function assisted optimization method with batch infill sampling criterion for expensive optimization,” in Proc. IEEE Congr. Evol. Comput. (CEC), 2019, pp. 1664–1671. 




[50] A. I. Khuri and S. Mukhopadhyay, “Response surface methodology,” Wiley Interdiscipl. Rev. Comput. Stat., vol. 2, no. 2, pp. 128–149, 2010. 




[51] W. Gao, G. Li, Q. Zhang, Y. Luo, and Z. Wang, “Solving nonlinear equation systems by a two-phase evolutionary algorithm,” IEEE Trans. Syst., Man, Cybern., Syst., vol. 51, no. 9, pp. 5652–5663, Sep. 2021. 




[52] L. Pan, C. He, Y. Tian, H. Wang, X. Zhang, and Y. Jin, “A classificationbased surrogate-assisted evolutionary algorithm for expensive manyobjective optimization,” IEEE Trans. Evol. Comput., vol. 23, no. 1, pp. 74–88, Feb. 2019. 




[53] T. Zhao, M. Li, and M. Poloczek, “Fast reconfigurable antenna state selection with hierarchical Thompson sampling,” in Proc. IEEE Int. Conf. Commun. (ICC), 2019, pp. 1–6. 




[54] R. Singh, F. Liu, Y. Sun, and N. Shroff, “Multi-armed bandits with dependent arms,” 2020, arXiv:2010.09478. 




[55] J. Hong, B. Kveton, M. Zaheer, and M. Ghavamzadeh, “Hierarchical Bayesian bandits,” in Proc. Int. Conf. Artif. Intell. Stat., 2022, pp. 7724–7741. 




[56] P. N. Suganthan et al., “Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization,” Nanyang Technol. Univ., Singapore, IIT, Kanpur, India, KanGAL Rep. #2005005, 2005. 




[57] J. Liang, B. Qu, P. Suganthan, and Q. Chen, “Problem definitions and evaluation criteria for the CEC 2015 competition on learning-based realparameter single objective optimization,” Comput. Intell. Lab., Nanyang Technol. Univ., Singapore, Rep. 201411A, 2014. 




[58] G. Li, Z. Wang, and M. Gong, “Expensive optimization via surrogateassisted and model-free evolutionary optimization,” IEEE Trans. Syst., Man, Cybern., Syst., vol. 53, no. 5, pp. 2758–2769, May 2023. 




[59] Z. Wang, Y.-S. Ong, and H. Ishibuchi, “On scalable multiobjective test problems with hardly dominated boundaries,” IEEE Trans. Evol. Comput., vol. 23, no. 2, pp. 217–231, Apr. 2019. 




[60] G. Chen, X. Luo, J. J. Jiao, and X. Xue, “Data-driven evolutionary algorithm for oil reservoir well-placement and control optimization,” Fuel, vol. 326, Oct. 2022, Art. no. 125125. 




[61] J.-D. Jansen, R.-M. Fonseca, S. Kahrobaei, M. Siraj, G. Van Essen, and P. Van den Hof, “The egg model—A geological ensemble for reservoir simulation,” Geosci. Data J., vol. 1, no. 2, pp. 192–195, 2014. 




[62] K.-A. Lie, An Introduction to Reservoir Simulation Using MATLAB/GNU Octave: User Guide for the MATLAB Reservoir Simulation Toolbox (MRST). Cambridge, U.K.: Cambridge Univ. Press, 2019. 


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/dc7390dbe9dc1a32efade67d50278f3408f3a0b74300f5d8094edd67517d39fc.jpg)

Zhenkun Wang (Member, IEEE) received the Ph.D. degree in circuits and systems from Xidian University, Xi’an, China, in 2016. 
From 2017 to 2020, he was a Postdoctoral Research Fellow with the School of Computer Science and Engineering, Nanyang Technological University, Singapore, and with the Department of Computer Science, City University of Hong Kong, Hong Kong. He is currently an Assistant Professor with the School of System Design and Intelligent Manufacturing and the Department of Computer 
Science and Engineering, Southern University of Science and Technology, Shenzhen, China. His research interests include evolutionary computation, optimization, machine learning, and their applications. 
Dr. Wang is an Associate Editor of the Swarm and Evolutionary Computation. 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/54c267a7d20d6bd63a49ddf2abccbf54ee64267c0d2e39675f49cd7b1903cd0d.jpg)

Laizhong Cui (Senior Member, IEEE) received the B.S. degree from Jilin University, Changchun, China, in 2007, and the Ph.D. degree in computer science and technology from Tsinghua University, Beijing, China, in 2012. 
He is currently a Professor with the College of Computer Science and Software Engineering, Shenzhen University, Shenzhen, China. He led more than ten scientific research projects, including National Key Research and Development Plan of China, National Natural Science Foundation of 
China, Guangdong Natural Science Foundation of China, and Shenzhen Basic Research Plan. He has published more than 100 papers, including IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS, IEEE TRANSACTIONS ON COMPUTERS, IEEE TRANSACTIONS ON PARALLEL AND DISTRIBUTED SYSTEMS, IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, IEEE TRANSACTIONS ON MULTIMEDIA, IEEE INTERNET OF THINGS JOURNAL, IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS, IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY, IEEE TRANSACTIONS ON NETWORK AND SERVICE MANAGEMENT, ACM Transactions on Internet Technology, IEEE NETWORK, IEEE INFOCOM, ACM MM, IEEE ICNP, and IEEE ICDCS. His research interests include future Internet architecture and protocols, edge computing, multimedia systems and applications, Blockchain, Internet of Things, cloud computing, and federated learning. 
Prof. Cui serves as an Associate Editor or a member of Editorial Board for several international journals, including IEEE INTERNET OF THINGS JOURNAL, IEEE TRANSACTIONS ON CLOUD COMPUTING, IEEE TRANSACTIONS ON NETWORK AND SERVICE MANAGEMENT, and International Journal of Machine Learning and Cybernetics. He is a Distinguished Member of the CCF. 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/c848d0d1710dd3e35ec6554974841505a9fdec34729a06b53683e034ba10daf8.jpg)

Lindong Xie received the B.S. degree in mechanical and electronic engineering from Henan University of Science and Technology, Luoyang, China, in 2021. He is currently pursuing the M.S. degree with the School of System Design and Intelligent Manufacturing, Southern University of Science and Technology, Shenzhen, China. 
His current research interests include data-driven evolutionary algorithms, and machine learning and their applications. 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/fc9ae32b7c88ec2cbd10662981491ff9c7071c5ddf4c9f3a28edf71ff94a6479.jpg)

Genghui Li received the M.Sc. degree in computer science and technology from Shenzhen University, Shenzhen, China, in 2016, and the Ph.D. degree in computer science from the City University of Hong Kong, Hong Kong, China, in 2021. 
He is currently a Postdoctoral Fellow with the Southern University of Science and Technology, Shenzhen. His research interests include evolutionary computation, computational intelligence, and machine learning. 
![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-06/b16a9b1e-dfb1-48d1-b289-c5de6f1b501e/194691076d9e30408013148618a02c8337595ca48f58611c9d7f5b0a65f9dbd9.jpg)

Maoguo Gong (Senior Member, IEEE) received the B.Eng. degree (Hons.) in electronic engineering and Ph.D. degree in electronic science and technology from Xidian University, Xi’an, China, in 2003 and 2009, respectively. 
Since 2006, he has been a Teacher with Xidian University. He was promoted to an Associate Professor and a Full Professor in 2008 and 2010, respectively, both with exceptive admission. He is leading or has completed over 20 projects as the Principle Investigator, funded by the National 
Natural Science Foundation of China and the National Key Research and Development Program of China. He has published over 100 papers in journals and conferences, and holds over 20 granted patents as the first inventor. His research interests are broadly in the area of computational intelligence, with applications to optimization, learning, data mining, and image understanding. 
Dr. Gong is the Director of the Chinese Association for Artificial Intelligence-Youth Branch, the Senior Member of Chinese Computer Federation, and an Associate Editor or an Editorial Board Member for over five journals, including the IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS and the IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTATIONAL INTELLIGENCE. 
1126 
Authorized licensed use limited to: Southern University of Science and Technology. Downloaded on March 12,2026 at 12:40:23 UTC from IEEE Xplore. Restrictions apply. 