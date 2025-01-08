import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

from cate.base.metrics.evaluate import (
    AbstractImageArtifact,
    AbstractMetric,
    AbstractTableArtifact,
)


class Rocauc(AbstractMetric):
    """
    Calculate the area under the receiver operating characteristic curve (ROC AUC).

    Attributes:
        None

    Methods:
        _calculate(pred, y, w): Calculates the ROC AUC and returns it as a float.
    """  # noqa: E501

    @property
    def name(self) -> str:
        return "rocauc"

    def _calculate(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> float:
        from sklearn.metrics import roc_auc_score

        return roc_auc_score(y, pred)  # type: ignore


class UpliftByPercentile(AbstractMetric):
    """
    https://note.com/dd_techblog/n/nb1ae45e79148
    スコア上位k%までのユーザーに対するUplift(tgのcv率-cgのcv率)を計算する
    """

    def __init__(self, k: float) -> None:
        self.k = k

    @property
    def name(self) -> str:
        return f"uplift_at_{int(self.k * 100)}"

    def _calculate(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> float:
        data = self.shape_data(pred, y, w)
        top_k_data = data.iloc[: int(len(data) * self.k)]

        # Calculate conversion rates for treatment and control groups
        tg_flg = top_k_data["group"] == 1
        tg_conversion_rate = top_k_data.loc[tg_flg, "conversion"].mean()
        cg_conversion_rate = top_k_data.loc[~tg_flg, "conversion"].mean()
        if tg_conversion_rate is np.nan or cg_conversion_rate is np.nan:
            return 0.0
        return float(tg_conversion_rate - cg_conversion_rate)


class QiniByPercentile(AbstractMetric):
    """
    https://note.com/dd_techblog/n/nb1ae45e79148
    スコア上位k%までのユーザーに対するQini値を計算する
    """

    def __init__(self, k: float) -> None:
        self.k = k

    @property
    def name(self) -> str:
        return f"qini_at_{int(self.k * 100)}"

    def _calculate(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> float:
        data = self.shape_data(pred, y, w)
        top_k_data = data.iloc[: int(len(data) * self.k)]

        # Calculate cumulative gains for treatment and control groups
        tg_flg = top_k_data["group"] == 1
        tg_conversion = top_k_data.loc[tg_flg, "conversion"].sum()
        cg_conversion = top_k_data.loc[~tg_flg, "conversion"].sum()
        tg_num = tg_flg.sum()
        cg_num = (~tg_flg).sum()
        if cg_num == 0:
            return 0.0
        return float(tg_conversion - cg_conversion * (tg_num / cg_num))


class Auuc(AbstractMetric):
    r"""
    UpliftCurveとbaselineに囲まれた部分の面積を計算する.
    AUUC = \sum_{k=1}^n AUUC_{\pi}(k)
    AUUC_{\pi}(k) =  AUL_{\pi}^T(k) - AUL_{\pi}^C(k) = \sum_{i=1}^k (R_{\pi}^T(i) - R_{\pi}^C(i)) - \frac{k}{2}(\bar{R}^T(k) - \bar{R}^C(k))
    """  # noqa: E501

    def __init__(self, bin_num: int = 100) -> None:
        self.bin_num = bin_num

    @property
    def name(self) -> str:
        return "auuc"

    def _calculate(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> float:
        # TODO: wがnp.float_や0,1のbinary以外の場合エラーが起きる
        data = self.shape_data(pred, y, w)
        data["rank"] = np.ceil(
            np.arange(1, len(data) + 1) / len(data) * self.bin_num
        ).astype(int)

        tg_flg = data["group"] == 1
        if tg_flg.sum() == 0 or (~tg_flg).sum() == 0:
            return 0.0
        average_uplift = (
            data.loc[tg_flg, "conversion"].mean()
            - data.loc[~tg_flg, "conversion"].mean()
        )
        auuc = 0.0
        ranks = set(data["rank"].unique())
        for rank in ranks:
            rank_flg = data["rank"] <= rank
            top_k_data = data.loc[rank_flg]
            if tg_flg.loc[rank_flg].sum() != 0 and (~tg_flg[rank_flg]).sum() != 0:
                tg_conversion = top_k_data.loc[tg_flg, "conversion"].mean()
                cg_conversion = top_k_data.loc[~tg_flg, "conversion"].mean()
                uplift = tg_conversion - cg_conversion
                auuc += uplift - average_uplift / 2
            else:
                auuc += 0.0 - average_uplift / 2
        return auuc


class Qini:
    """
    https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_1H4OS12b02/_pdf
    """


class QiniCurve:
    """
    https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_1H4OS12b02/_pdf
    """


class UpliftCurve(AbstractImageArtifact):
    """
    This class generates an uplift curve, which is a graphical representation of the uplift
    (difference in conversion rates) between the treatment and control groups across different
    percentiles of the predicted scores.

    Attributes:
        bin_num (int): The number of bins to divide the data into for calculating the uplift curve.

    Methods:
        _calculate(pred, y, w): Calculates the uplift curve and returns it as a matplotlib Figure.
    """  # noqa: E501

    def __init__(self, bin_num: int = 100) -> None:
        self.bin_num = bin_num

    @property
    def name(self) -> str:
        return "uplift_curve"

    def _plot(self, data: pd.DataFrame) -> Figure:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.tick_params(labelsize=14)
        ax.set_xlabel("percentile", fontsize=18)
        ax.set_ylabel("uplift", fontsize=18)
        ax.tick_params(length=10, width=1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        ax.plot(data["uplift_x"], data["uplift_y"], label="uplift")
        ax.plot(
            data["baseline_x"],
            data["baseline_y"],
            label="random",
        )
        ax.legend()
        return fig

    def _calculate(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> pd.DataFrame:
        data = self.shape_data(pred, y, w)

        data["rank"] = np.ceil(
            np.arange(1, len(data) + 1) / len(data) * self.bin_num
        ).astype(int)

        tg_flg = data["group"] == 1
        average_uplift = (
            data.loc[tg_flg, "conversion"].mean()
            - data.loc[~tg_flg, "conversion"].mean()
        )
        uplifts = []
        ranks = set(data["rank"].unique())
        for rank in ranks:
            rank_flg = data["rank"] <= rank
            top_k_data = data.loc[rank_flg]
            if (tg_flg & rank_flg).sum() != 0 and (~tg_flg & rank_flg).sum() != 0:
                tg_conversion = top_k_data.loc[tg_flg, "conversion"].mean()
                cg_conversion = top_k_data.loc[~tg_flg, "conversion"].mean()
                uplift = tg_conversion - cg_conversion
                uplifts.append(uplift)
            else:
                uplifts.append(0.0)
        result = pd.DataFrame(
            {
                "baseline_x": np.arange(0, 1, 1 / len(ranks)),
                "baseline_y": np.arange(0, average_uplift, average_uplift / len(ranks)),
                "uplift_x": np.arange(0, 1, 1 / len(ranks)),
                "uplift_y": uplifts,
            }
        )
        return result


class Outputs(AbstractTableArtifact):
    """
    This class generates a table of the predicted uplifts, actual conversions, and group assignments
    for each observation in the dataset.

    Attributes:
        None

    Methods:
        _calculate(pred, y, w): Creates a pandas DataFrame containing the predicted uplifts, actual
        conversions, and group assignments for each observation.
    """  # noqa: E501

    def __init__(self, n: int | None = None, shuffle: bool = False) -> None:
        self.n = n
        self.shuffle = shuffle

    @property
    def name(self) -> str:
        return "outputs"

    def _calculate(
        self,
        pred: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_ | np.int_],
        w: npt.NDArray[np.float_ | np.int_],
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "pred": pred,
                "conversion": y,
                "group": w,
            }
        )
        if self.shuffle:
            df = df.sample(frac=1, random_state=42)

        if self.n is not None:
            return df.head(self.n)
        return df
