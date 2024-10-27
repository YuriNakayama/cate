import numpy as np
import pandas as pd


class UpliftByPercentile:
    """
    https://note.com/dd_techblog/n/nb1ae45e79148
    スコア上位k%までのユーザーに対するUplift(tgのcv率-cgのcv率)を計算する
    """

    def __init__(self, k: float) -> None:
        self.k = k

    def __call__(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> float:
        data = pd.concat(
            [score, group, conversion], keys=["score", "group", "conversion"], axis=1
        ).sort_values(by="score", ascending=False)
        top_k_data = data.iloc[: int(len(data) * self.k / 100)]

        # Calculate conversion rates for treatment and control groups
        tg_flg = top_k_data["group"] == 1
        tg_conversion_rate = top_k_data.loc[tg_flg, "conversion"].mean()
        cg_conversion_rate = top_k_data.loc[~tg_flg, "conversion"].mean()
        return float(tg_conversion_rate - cg_conversion_rate)


class QiniByPercentile:
    """
    https://note.com/dd_techblog/n/nb1ae45e79148
    スコア上位k%までのユーザーに対するQini値を計算する
    """

    def __init__(self, k: float) -> None:
        self.k = k

    def __call__(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> float:
        data = pd.concat(
            [score, group, conversion], keys=["score", "group", "conversion"], axis=1
        ).sort_values(by="score", ascending=False)
        top_k_data = data.iloc[: int(len(data) * self.k / 100)]

        # Calculate cumulative gains for treatment and control groups
        tg_flg = top_k_data["group"] == 1
        tg_conversion = top_k_data.loc[tg_flg, "conversion"].sum()
        cg_conversion = top_k_data.loc[~tg_flg, "conversion"].sum()
        tg_num = tg_flg.sum()
        cg_num = (~tg_flg).sum()
        return float(tg_conversion - cg_conversion * (tg_num / cg_num))


class Auuc:
    r"""
    UpliftCurveとbaselineに囲まれた部分の面積を計算する.
    AUUC = \sum_{k=1}^n AUUC_{\pi}(k)
    AUUC_{\pi}(k) =  AUL_{\pi}^T(k) - AUL_{\pi}^C(k) = \sum_{i=1}^k (R_{\pi}^T(i) - R_{\pi}^C(i)) - \frac{k}{2}(\bar{R}^T(k) - \bar{R}^C(k))
    """

    def __init__(self, bin_num: int = 10_000) -> None:
        self.bin_num = bin_num

    def __call__(
        self, score: pd.Series, group: pd.Series, conversion: pd.Series
    ) -> float:
        data = pd.concat(
            [score, group, conversion], keys=["score", "group", "conversion"], axis=1
        ).sort_values(by="score", ascending=False)
        data["rank"] = np.ceil(
            np.arange(1, len(data) + 1) / len(data) * self.bin_num
        ).astype(int)

        tg_flg = data["group"] == 1
        average_uplift = (
            data.loc[tg_flg, "conversion"].mean()
            - data.loc[~tg_flg, "conversion"].mean()
        )
        auuc = 0.0
        for _, rank_flg in data.groupby("rank").groups.items():
            top_k_data = data.loc[rank_flg]
            if tg_flg.loc[rank_flg].sum() != 0 and (~tg_flg[rank_flg]).sum() != 0:
                tg_conversion = top_k_data.loc[tg_flg, "conversion"].mean()
                cg_conversion = top_k_data.loc[~tg_flg, "conversion"].mean()
                uplift = tg_conversion - cg_conversion
                auuc += uplift - average_uplift / 2
        return auuc


class Qini:
    """
    https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_1H4OS12b02/_pdf
    """


class QiniCurve:
    """
    https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_1H4OS12b02/_pdf
    """

    pass


class UpliftCurve:
    pass
