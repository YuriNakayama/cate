import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class Ticks:
    def __init__(
        self,
        ticks: list[str | int | float] | None = None,
        labels: list[str] | None = None,
    ) -> None:
        self.ticks = ticks
        self.labels = labels

    def __call__(self, ax: Axes) -> Axes:
        if self.ticks is not None:
            ax.set_xticks(self.ticks)
        if self.labels is not None:
            ax.set_xticklabels(self.labels)
        return ax


class LinePlot:
    def __init__(self, *, x_ticks: Ticks = Ticks()) -> None:
        self.x_ticks = x_ticks

    def __call__(
        self, data: pd.DataFrame, title: str, x_label: str, y_label: str
    ) -> Figure:
        plt.ioff()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.tick_params(labelsize=14, length=10, width=1)
        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel(y_label, fontsize=18)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        ax = self.x_ticks(ax)

        ax.set_title(title, fontsize=24)
        ax.plot(data)
        ax.legend(data.columns, fontsize=18, framealpha=0)
        return fig
