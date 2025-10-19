# Metalearner

## 目次

## T-Learner

### アルゴリズム

```math
\begin{array}{llr}
\rm{procedure} & \rm{T-LEARNER(X, Y, W)} \\
& \\
&\hat{\mu}_0 = M_0(Y^0 \sim X^0) & CGのデータでモデルを学習\\
&\hat{\mu}_1 = M_1(Y^1 \sim X^1) & TGのデータでモデルを学習\\
&\hat{\tau} = \hat{\mu}_1(x) - \hat{\mu}_0(x)  & CATEを計算
\end{array}
```

### 概要

CATEの構造が非常に複雑で、TGとCGのそれぞれのデータ生成過程に共通の傾向がない場合には、特に優れた性能を発揮する傾向にある。

## S-Learner

### アルゴリズム

```math
\begin{array}{llr}
\rm{procedure} & \rm{S-LEARNER(X, Y, W)} \\
& \\
&\hat{\mu} = M(Y \sim (X, W)) & 全データでモデルを学習\\
&\hat{\tau} = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)  & CATEを計算
\end{array}
```

### 概要

TGCGデータの生成過程が等しい場合とCATEが多くの場所で0である場合において、最も良い推定を行うことが確認できる。

## X-Learner

### アルゴリズム

```math
\begin{array}{llr}
\rm{procedure} & \rm{X-LEARNER(X, Y, W)} \\
& \\
&\hat{\mu}_0 = M_0(Y^0 \sim X^0) & TG/CGのデータでモデルを学習\\
&\hat{\mu}_1 = M_1(Y^1 \sim X^1) & \\
& \\
&\tilde{D}^1 = Y_i^1 - \hat{\mu}_0(X_i^1) & TG/CGグループのCATEを補完\\
&\tilde{D}^0 = \hat{\mu}_1(X_i^0) - Y_i^0 & \\
& \\
& \hat{\tau}_1 = M_3(\tilde{D}^1 \sim X^1) & CATEを二つの方法で推定\\
& \hat{\tau}_0 = M_4(\tilde{D}^0 \sim X^0) & \\
& \\
&\hat{\tau}(x) = g(x) \hat{\tau}_0(x) + (1 - g(x)) \hat{\tau}_1(x) & 重み付け平均でCATEを計算
\end{array}
```

### 概要

CATEに構造的な仮定がある場合や，TGもしくはCGのデータ量が他方のよりもはるかに大きい場合に特に優れた性能を発揮する。
真のCATEに0の部分がある場合、通常はS-learnerほどではないが、T-learnerよりは良い推定ができる。CATEが非常に複雑な構造である場合には、S-learnerやT-learnerよりも良い推定ができる。

## R-learner

### アルゴリズム

```math
\begin{array}{llr}
\rm{procedure} & \rm{R-LEARNER(X, Y, W)} \\
& \\
&\hat{\mu} = M_{\mu}(Y \sim X) & CVを予測するモデルを学習\\
&\hat{e} = M_e(W \sim X) & 傾向スコアを予測するモデルを学習\\
& \\
& \hat{L}_n(\tau) = \frac{1}{n} \sum_{i=1}^N \{Y_i - \hat{m}(X_i) - (W_i - \hat{e}(X_i))\tau(X_i) \}^2 & \\
& \hat{\tau} = \arg \min_{\tau} \hat{L}_n(\tau) & \rm{Robinson Loss}を最小化する\tauを学習\\
\end{array}
```

### 概要

ロビンソン分解(Robinson decomposition)を利用したモデル.

$m(X_i) = E[Y_i | X_i], \hspace{0.1cm} e(X_i) = E[W_i | X_i] $
$\tau(X_i) = \mathbb{E}[Y_i(W=1) - Y_i(W=0)|X_i]$とすると

```math
\begin{align*}
\epsilon_i &= Y_i - \mathbb{E}[Y_i | X_i] - (W_i - \mathbb{E}[W_i | X_i])\tau(X_i) \\
&= Y_i - m(X_i) - (W_i - e(X_i))\tau(X_i)
\end{align*}
```

であり

```math
\mathbb{E}[\epsilon_i | X_i, W] = 0
```

である.

## DR-learner

### アルゴリズム

```math
\begin{array}{llr}
\rm{procedure} & \rm{DR-LEARNER(X, Y, W)} \\
& \\
&\hat{\mu}_0 = M_0(Y^0 \sim X^0) & TG/CGのデータでモデルを学習\\
&\hat{\mu}_1 = M_1(Y^1 \sim X^1) &\\
& \hat{e} = M_e(W \sim X) & 傾向スコアを予測するモデルを学習\\
&\\
& \phi = \frac{W - \hat{e}(X)}{\hat{e}(X)(1 - \hat{e}(X))} (Y - \hat{m}_W(X)) \hat{m}_1(X) - \hat{m}_0(X)& CATEのDR推定量を計算 \\
& \hat{\tau} = M_{\tau}(\phi \sim X) & DR推定量を予測するモデルを学習\\

\end{array}
```

### 概要

CATEのDR推定量を計算するモデル.
![dr_learner_cross_fitting](image/dr_learner_cross_fitting.png)

## 参考

- [Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning](https://arxiv.org/pdf/1706.03461)
- [Quasi-oracle estimation of heterogeneous treatment effects](https://par.nsf.gov/servlets/purl/10311702)
- [METALEARNERS FOR RANKING TREATMENT EFFECTS](https://arxiv.org/pdf/2405.02183)
