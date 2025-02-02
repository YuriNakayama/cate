# 定義

## 推定量

データ$\mathcal{D} = \{x_i\}_{i=1}^n$は同時分布$p(\mathcal{D}) = p(x_1, \dots, x_n) = \prod_{i=1}^n p(x_i)$から独立に生成されたものとする.
このデータに基づき, 推定目標$V$を統計的に近似する**推定量(estimator)**$\hat{V}(\mathcal{D})$を構築する.
このとき推定量を評価するために, 推定量の統計的性質を分析することが重要.

## 推定量の性質

今回は推定量の統計的性質として, バイアスとバリアンスを考える.
推定量$\hat{V}$のバイアスとは, 推定目標と推定量の期待値の差であり, 次のように定義される.
$$\rm{Bias}[\hat{V}(\mathcal{D})] = |V - \mathbb{E}_{p(\mathcal{D})}[\hat{V}(\mathcal{D})]|$$
推定量$\hat{V}$のバリアンスとは, 推定量のばらつきの大きさであり, 次のように定義される.
$$\rm{Var}[\hat{V}(\mathcal{D})] = \mathbb{E}_{p(\mathcal{D})}[(\hat{V}(\mathcal{D}) - \mathbb{E}_{p(\mathcal{D})}[\hat{V}(\mathcal{D})])^2]$$

:::note info
平均二乗誤差(Mean Squared Error: MSE)はバイアスとバリアンスの和で表される.

```math
\rm{MSE}[\hat{V}(\mathcal{D})] = \rm{Bias}[\hat{V}(\mathcal{D})]^2 + \rm{Var}[\hat{V}(\mathcal{D})]
```
