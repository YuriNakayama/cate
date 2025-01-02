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

### R-learner

```math
```

## 参考

- [Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning](https://arxiv.org/pdf/1706.03461)
