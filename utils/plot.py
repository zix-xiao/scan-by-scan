import logging
import os
from typing import List, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib_venn import venn2
from scipy import stats


def plot_scatter(
    x: pd.Series,
    y: pd.Series,
    log_x: bool = False,
    log_y: bool = False,
    data: Union[None, pd.DataFrame] = None,
    filter_thres: float = 0,
    interactive: bool = False,
    hover_data: Union[None, List] = None,  # only used if interactive is true
    color: Union[None, pd.Series] = None,
    show_diag: bool = True,
    show_conf: Union[None, tuple] = None,
    save_dir: Union[None, str] = None,
    x_label: Union[None, str] = None,
    y_label: Union[None, str] = None,
    title: Union[None, str] = None,
):
    """
    Generate scatter plot with correlation coefficient and number of data points,
    with other annotations possible

    :color: a pandas series for color, default is color with density info

    return
    -> valid_idx: index kept after filter
    """
    valid_idx = np.where((x > filter_thres) & (y > filter_thres))
    x_name = str(x.name)
    y_name = str(y.name)

    if data is not None:
        data = data.iloc[valid_idx[0], :].copy()
    else:
        data = pd.DataFrame({x_name: x.values[valid_idx], y_name: y.values[valid_idx]})

    x_log = x.values[valid_idx]
    y_log = y.values[valid_idx]
    if log_x:
        x_name += "_log"
        x_log = np.log10(x.values[valid_idx])
        data[x_name] = x_log
    if log_y:
        y_name += "_log"
        y_log = np.log10(y.values[valid_idx])
        data[y_name] = y_log

    if x_label is None:
        x_label = x_name
    if y_label is None:
        y_label = y_name
    if title is None:
        title = "Corr. of" + x_name + " and " + y_name

    PearsonR = stats.pearsonr(x=x_log, y=y_log)  # w/ log and w/o outliers
    SpearmanR = stats.spearmanr(a=x_log, b=y_log)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=x_log, y=y_log)
    print(
        "Data: ",
        x_name,
        y_name,
        ", slope = ",
        np.round(slope, 3).item(),
        ", intercept = ",
        np.round(intercept, 3).item(),
        ", PearsonR = ",
        np.round(PearsonR[0], 3).item(),
        ", SpearmanR = ",
        np.round(SpearmanR[0], 3).item(),
    )

    # calculate the point density
    if color is None:
        xy = np.vstack([x_log, y_log])
        color = stats.gaussian_kde(xy)(xy)
    else:
        color = color.values[valid_idx]

    RegressionY = x_log * slope + intercept
    AbsResidue = abs(y_log - RegressionY)

    if interactive:
        fig = px.scatter(
            data,
            x=x_name,
            y=y_name,
            color=color,
            hover_data=hover_data,
            title=title,
            labels={x_name: x_label, y_name: y_label},
            trendline="ols",
        )
        if show_diag:  # Add a diagonal line y = x
            fig.update_layout(
                shapes=[
                    dict(
                        type="line",
                        x0=min(x_log),
                        y0=min(x_log),
                        x1=max(x_log),
                        y1=max(x_log),
                        line=dict(color="red", width=2),
                    )
                ]
            )
        # TODO: change to smart/relative positioning
        fig.add_annotation(
            x=6.5, y=10, text="N = " + str(x_log.shape[0]), showarrow=False
        )
        fig.add_annotation(
            x=6.5,
            y=11,
            text="Prs.r = "
            + "{:.3f}".format(PearsonR[0])
            + ", Sprm.r = "
            + "{:.3f}".format(SpearmanR[0]),
            showarrow=False,
        )
        fig.show()

    else:
        # Plot with correlation
        ax = sns.regplot(x=x_log, y=y_log, scatter=False, fit_reg=True)
        sns.scatterplot(x=x_log, y=y_log, hue=color, linewidth=0, legend=False)  # type: ignore
        ax.annotate(
            "N = " + str(x_log.shape[0]),
            xy=(0.2, 0.85),
            xycoords="figure fraction",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=7,
        )
        ax.annotate(
            "Prs.r = "
            + "{:.3f}".format(PearsonR[0])
            + ", Sprm.r = "
            + "{:.3f}".format(SpearmanR[0]),
            xy=(0.2, 0.8),
            xycoords="figure fraction",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=7,
        )
        ax.annotate(
            "slp. = "
            + "{:.3f}".format(slope)
            + ", intrcpt. = "
            + "{:.3f}".format(intercept),
            xy=(0.2, 0.75),
            xycoords="figure fraction",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=7,
        )

        min_val = min(x_log)
        max_val = max(x_log)
        if show_diag:
            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                linestyle="--",
                color="k",
                lw=1,
                label="y=x",
            )
        if show_conf is not None:
            plt.plot(
                [min_val, max_val],
                [min_val + show_conf[0], max_val + show_conf[0]],
                linestyle="--",
                color="green",
                lw=2,
                label="lower bound",
            )
            plt.plot(
                [min_val, max_val],
                [min_val + show_conf[1], max_val + show_conf[1]],
                linestyle="--",
                color="green",
                lw=2,
                label="upper bound",
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.suptitle(title + y_name)
        save_plot(
            save_dir=save_dir, fig_type_name="CorrQuantification", fig_spec_name=y_name
        )

    return RegressionY.T, AbsResidue.T, valid_idx


def plot_venn2(
    set1: Set,
    set2: Set,
    label1: str,
    label2: str,
    save_dir: str | None = None,
    title: str | None = None,
):
    venn2([set1, set2], set_labels=(label1, label2))
    if title is not None:
        plt.title(title)
    save_plot(save_dir=save_dir, fig_type_name="VennDiag", fig_spec_name=title)


def plot_comparison(
    y_true: pd.Series,
    y_pred: pd.Series,
    x_true: Union[None, pd.Series] = None,
    x_pred: Union[None, pd.Series] = None,
    log_y: bool = False,
):
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    if log_y:
        y_pred = np.log10(y_pred + 1)
        y_true = np.log10(y_true + 1)
    if x_true is None:
        x_true = np.arange(len(y_true))
    if x_pred is None:
        x_pred = np.arange(len(y_pred))
    axs[0].vlines(x=x_true, ymin=0, ymax=y_true)
    axs[1].vlines(x=x_pred, ymin=-y_pred, ymax=0)

    # enforce same y axis limits
    axs[0].set_ylim([0, max(axs[0].get_ylim()[1], abs(axs[1].get_ylim()[0]))])
    axs[1].set_ylim([-axs[0].get_ylim()[1], 0])


def save_plot(save_dir, fig_type_name, fig_spec_name):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(
            fname=os.path.join(save_dir, fig_type_name + "_" + fig_spec_name + ".png"),
            dpi=300,
        )
        plt.close()
    else:
        plt.show()
