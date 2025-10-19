# 因果推論の論文

## [Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning](https://arxiv.org/abs/1706.03461)

### Abstract

There is growing interest in estimating and analyzing heterogeneous treatment effects in experimental and observational studies. We describe a number of meta-algorithms that can take advantage of any supervised learning or regression method in machine learning and statistics to estimate the Conditional Average Treatment Effect (CATE) function. Meta-algorithms build on base algorithms---such as Random Forests (RF), Bayesian Additive Regression Trees (BART) or neural networks---to estimate the CATE, a function that the base algorithms are not designed to estimate directly. We introduce a new meta-algorithm, the X-learner, that is provably efficient when the number of units in one treatment group is much larger than in the other, and can exploit structural properties of the CATE function. For example, if the CATE function is linear and the response functions in treatment and control are Lipschitz continuous, the X-learner can still achieve the parametric rate under regularity conditions.

### 概要

実験・観察研究において異質処置効果の推定と分析への関心が高まっている。本研究では、機械学習・統計の教師あり学習や回帰手法を活用して条件付き平均処置効果（CATE）関数を推定する多数のメタアルゴリズムを説明する。メタアルゴリズムは、ランダムフォレスト（RF）、ベイジアン加法回帰木（BART）、ニューラルネットワークなどのベースアルゴリズムを基盤として、これらが直接推定するよう設計されていないCATE関数を推定する。本研究では新しいメタアルゴリズムであるX-learnerを導入し、これは一方の処置群のユニット数が他方よりも大幅に多い場合に証明可能な効率性を持ち、CATE関数の構造的特性を活用できる。例えば、CATE関数が線形で、処置・対照群の応答関数がリプシッツ連続である場合、X-learnerは正則条件下でパラメトリック率を達成できる。

### 提案手法

## [B-Learner: Quasi-Oracle Bounds on Heterogeneous Causal Effects Under Hidden Confounding](https://arxiv.org/abs/2304.10577)

### Abstract

Estimating heterogeneous treatment effects from observational data is a crucial task across many fields, helping policy and decision-makers take better actions. There has been recent progress on robust and efficient methods for estimating the conditional average treatment effect (CATE) function, but these methods often do not take into account the risk of hidden confounding, which could arbitrarily and unknowingly bias any causal estimate based on observational data. We propose a meta-learner called the B-Learner, which can efficiently learn sharp bounds on the CATE function under limits on the level of hidden confounding.

### 概要

観察データからの異質処置効果の推定は多くの分野で重要な課題であり、政策立案者や意思決定者がより良い行動を取るのに役立つ。条件付き平均処置効果（CATE）関数を推定するための頑健で効率的な手法について最近の進歩があるが、これらの手法は隠れた交絡のリスクを考慮しないことが多く、観察データに基づく因果推定を任意にかつ知らないうちにバイアスさせる可能性がある。本研究では、隠れた交絡のレベルに制限を設けた下で、CATE関数に対する鋭い境界を効率的に学習できるB-Learnerと呼ばれるメタ学習器を提案する。

### 課題

### 提案手法

## [Meta-learning for heterogeneous treatment effect estimation with closed-form solvers](https://arxiv.org/abs/2305.11353)

### Abstract

This article proposes a meta-learning method for estimating the conditional average treatment effect (CATE) from a few observational data. The proposed method learns how to estimate CATEs from multiple tasks and uses the knowledge for unseen tasks. In the proposed method, based on the meta-learner framework, we decompose the CATE estimation problem into sub-problems. For each sub-problem, we formulate our estimation models using neural networks with task-shared and task-specific parameters. With our formulation, we can obtain optimal task-specific parameters in a closed form that are differentiable with respect to task-shared parameters, making it possible to perform effective meta-learning.

### 概要

本論文では、少数の観察データから条件付き平均処置効果（CATE）を推定するメタ学習手法を提案する。概要は複数のタスクからCATEの推定方法を学習し、その知識を未知のタスクに活用する。概要では、メタ学習器フレームワークに基づいてCATE推定問題をサブ問題に分解する。各サブ問題に対して、タスク共有パラメータとタスク固有パラメータを持つニューラルネットワークを用いて推定モデルを定式化する。この定式化により、タスク共有パラメータに関して微分可能な閉形式で最適なタスク固有パラメータを得ることができ、効果的なメタ学習を実行することが可能になる。

### 提案手法

## [A Meta-Learning Approach for Estimating Heterogeneous Treatment Effects Under Hölder Continuity](https://www.mdpi.com/2227-7390/13/11/1739)

### Abstract

Estimating heterogeneous treatment effects plays a vital role in many statistical applications, such as precision medicine and precision marketing. In this paper, we propose a novel meta-learner, termed RXlearner for estimating the conditional average treatment effect (CATE) within the general framework of meta-algorithms. RXlearner enhances the weighting mechanism of the traditional Xlearner to improve estimation accuracy. We establish non-asymptotic error bounds for RXlearner under a continuity classification criterion, specifically assuming that the response function satisfies Hölder continuity.

### 概要

異質処置効果の推定は、精密医療や精密マーケティングなど多くの統計的応用において重要な役割を果たしている。本論文では、メタアルゴリズムの一般的フレームワーク内で条件付き平均処置効果（CATE）を推定するRXlearnerと呼ばれる新規メタ学習器を提案する。RXlearnerは従来のXlearnerの重み付けメカニズムを強化して推定精度を向上させる。応答関数がヘルダー連続性を満たすと仮定した連続性分類基準下で、RXlearnerの非漸近誤差境界を確立する。

### 提案手法

## [M-learner:A Flexible And Powerful Framework To Study Heterogeneous Treatment Effect In Mediation Model](https://arxiv.org/abs/2505.17917)

### Abstract

We propose a novel method, termed the M-learner, for estimating heterogeneous indirect and total treatment effects and identifying relevant subgroups within a mediation framework. The procedure comprises four key steps. First, we compute individual-level conditional average indirect/total treatment effect Second, we construct a distance matrix based on pairwise differences. Third, we apply tSNE to project this matrix into a low-dimensional Euclidean space, followed by K-means clustering to identify subgroup structures. Finally, we calibrate and refine the clusters using a threshold-based procedure to determine the optimal configuration.

### 概要

媒介フレームワーク内で異質間接・総処置効果を推定し、関連サブグループを識別するM-learnerと呼ばれる新規手法を提案する。手順は4つの主要ステップから構成される。まず、個人レベルの条件付き平均間接・総処置効果を計算する。次に、ペアワイズ差に基づく距離行列を構築する。第三に、tSNEを適用してこの行列を低次元ユークリッド空間に投影し、続いてK-meansクラスタリングを用いてサブグループ構造を識別する。最後に、最適な構成を決定するため閾値ベース手順を用いてクラスターを較正・精緻化する。

### 提案手法

## [Differentially Private Learners for Heterogeneous Treatment Effects](https://arxiv.org/abs/2503.03486)

### Abstract

Patient data is widely used to estimate heterogeneous treatment effects and thus understand the effectiveness and safety of drugs. Yet, patient data includes highly sensitive information that must be kept private. In this work, we aim to estimate the conditional average treatment effect (CATE) from observational data under differential privacy. Specifically, we present DP-CATE, a novel framework for CATE estimation that is Neyman-orthogonal and further ensures differential privacy of the estimates.

### 概要

患者データは異質処置効果を推定し、薬物の有効性と安全性を理解するために広く使用されている。しかし、患者データには秘匿すべき高度に機密性の高い情報が含まれている。本研究では、差分プライバシー下で観察データから条件付き平均処置効果（CATE）を推定することを目指す。具体的には、ネイマン直交性を持ちつつ推定値の差分プライバシーを保証する、CATE推定のための新規フレームワークDP-CATEを提示する。

### 提案手法

## [A Meta-learner for Heterogeneous Effects in Difference-in-Differences](https://arxiv.org/abs/2502.04699)

### Abstract

We address the problem of estimating heterogeneous treatment effects in panel data, adopting the popular Difference-in-Differences (DiD) framework under the conditional parallel trends assumption. We propose a novel doubly robust meta-learner for the Conditional Average Treatment Effect on the Treated (CATT), reducing the estimation to a convex risk minimization problem involving a set of auxiliary models. Our framework allows for the flexible estimation of the CATT, when conditioning on any subset of variables of interest using generic machine learning.

### 概要

条件付き平行トレンド仮定下で人気の差分の差分（DiD）フレームワークを採用し、パネルデータにおける異質処置効果の推定問題に取り組む。処置群の条件付き平均処置効果（CATT）のための新規二重頑健メタ学習器を提案し、推定を補助モデルの集合を含む凸リスク最小化問題に帰着させる。本フレームワークは、汎用機械学習を用いて任意の関心変数のサブセットで条件付けた場合のCATTの柔軟な推定を可能にする。

### 提案手法

## [Robust CATE Estimation Using Novel Ensemble Methods](https://arxiv.org/abs/2407.03690)

### Abstract

The estimation of Conditional Average Treatment Effects (CATE) is crucial for understanding the heterogeneity of treatment effects in clinical trials. We evaluate the performance of common methods, including causal forests and various meta-learners, across a diverse set of scenarios, revealing that each of the methods struggles in one or more of the tested scenarios. To address this limitation of existing methods, we propose two new ensemble methods that integrate multiple estimators to enhance prediction stability and performance - Stacked X-Learner which uses the X-Learner with model stacking for estimating the nuisance functions, and Consensus Based Averaging (CBA), which averages only the models with highest internal agreement.

### 概要

条件付き平均処置効果（CATE）の推定は、臨床試験における処置効果の異質性を理解するために重要である。因果フォレストや様々なメタ学習器を含む一般的手法の性能を多様なシナリオで評価し、各手法がテストされたシナリオの1つ以上で苦戦することを明らかにした。既存手法のこの限界に対処するため、予測の安定性と性能を向上させるために複数の推定器を統合する2つの新しいアンサンブル手法を提案する：ニューサンス関数の推定にモデルスタッキングを用いたX-LearnerであるStacked X-Learnerと、最も高い内部一致を持つモデルのみを平均化するConsensus Based Averaging（CBA）である。

### 提案手法

## [Hybrid Meta-learners for Estimating Heterogeneous Treatment Effects](https://arxiv.org/abs/2506.13680)

### Abstract

Estimating conditional average treatment effects (CATE) from observational data involves modeling decisions that differ from supervised learning, particularly concerning how to regularize model complexity. Previous approaches can be grouped into two primary "meta-learner" paradigms that impose distinct inductive biases. Indirect meta-learners first fit and regularize separate potential outcome (PO) models and then estimate CATE by taking their difference, whereas direct meta-learners construct and directly regularize estimators for the CATE function itself. In this paper, we introduce the Hybrid Learner (H-learner), a novel regularization strategy that interpolates between the direct and indirect regularizations depending on the dataset at hand.

### 概要

観察データから条件付き平均処置効果（CATE）を推定することは、特にモデルの複雑性を正則化する方法に関して、教師あり学習とは異なるモデリング決定を含む。従来のアプローチは、異なる帰納的バイアスを課す2つの主要な「メタ学習器」パラダイムにグループ化できる。間接メタ学習器は最初に別々のポテンシャル結果（PO）モデルを適合・正則化し、次にそれらの差を取ってCATEを推定するが、直接メタ学習器はCATE関数自体の推定器を構築・直接正則化する。本論文では、手持ちのデータセットに応じて直接・間接正則化を補間する新規正則化戦略であるハイブリッド学習器（H-learner）を導入する。

### 提案手法

## [Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms](https://arxiv.org/abs/2101.10943)

### Abstract

The need to evaluate treatment effectiveness is ubiquitous in most of empirical science, and interest in flexibly investigating effect heterogeneity is growing rapidly. To do so, a multitude of model-agnostic, nonparametric meta-learners have been proposed in recent years. Such learners decompose the treatment effect estimation problem into separate sub-problems, each solvable using standard supervised learning methods. Choosing between different meta-learners in a data-driven manner is difficult, as it requires access to counterfactual information. Therefore, with the ultimate goal of building better understanding of the conditions under which some learners can be expected to perform better than others a priori, we theoretically analyze four broad meta-learning strategies which rely on plug-in estimation and pseudo-outcome regression.

### 概要

処置効果を評価する必要性はほとんどの実証科学に遍在し、効果の異質性を柔軟に調査することへの関心が急速に高まっている。これを行うため、近年多くのモデル非依存・ノンパラメトリックメタ学習器が提案されてきた。このような学習器は処置効果推定問題を別々のサブ問題に分解し、それぞれを標準的教師あり学習手法で解決可能にする。異なるメタ学習器間をデータ駆動的に選択することは、反実仮想情報へのアクセスを要求するため困難である。従って、どの学習器が他より良い性能を示すと事前に期待できる条件についてより良い理解を構築することを最終目標とし、プラグイン推定と擬似結果回帰に依存する4つの広範なメタ学習戦略を理論的に分析する。

### 提案手法
