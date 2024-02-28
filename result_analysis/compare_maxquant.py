""" Module for comparing with maxquant results """
import logging
from typing import Literal, List
import pandas as pd
import matplotlib.pyplot as plt
from utils.plot import save_plot
from utils.tools import _perc_fmt

Logger = logging.getLogger(__name__)


def merge_with_maxquant_exp(
    maxquant_exp_df: pd.DataFrame,
    maxquant_ref_df: pd.DataFrame,
):
    """compare the inferred intensities from maxquant and SBS,
        when a different dictionary then experiment MQ result is used

    :MQ_exp: the maxquant result of the raw data whose MS1 scans were used for inference
    :MQ_dict: the maxquant result used for SBS inference
    """

    maxquant_ref_and_exp = pd.merge(
        left=maxquant_ref_df[
            [
                "Modified sequence",
                "Charge",
                "predicted_RT",
                "m/z",
                "Mass",
                "Length",
                "id",
                "RT_search_left",
                "RT_search_right",
            ]
        ],
        right=maxquant_exp_df[
            [
                "Modified sequence",
                "Charge",
                "Calibrated retention time start",
                "Calibrated retention time finish",
                "Calibrated retention time",
                "Retention time",
                "Intensity",
            ]
        ],
        on=["Modified sequence", "Charge"],
        how="right",
        indicator=True,
    )
    Logger.debug("Maxquant experiment file has %s entries.", maxquant_exp_df.shape[0])
    Logger.debug(
        "columns after merge MQ dict and MQ exp %s", maxquant_ref_and_exp.columns
    )
    return maxquant_ref_and_exp


def evaluate_rt_overlap(
    maxquant_ref_and_exp: pd.DataFrame, save_dir: str | None = None
):
    """evaluate the RT overlap between the experiment and reference file"""

    def _categorize_ranges(row):
        if (
            row["RT_search_left"] <= row["Calibrated retention time start"]
            and row["RT_search_right"] >= row["Calibrated retention time finish"]
        ):
            return "full_overlap"
        elif (
            row["RT_search_right"] < row["Calibrated retention time start"]
            or row["RT_search_left"] > row["Calibrated retention time finish"]
        ):
            return "partial_overlap"
        else:
            return "no_overlap"

    maxquant_ref_and_exp["RT_overlap"] = maxquant_ref_and_exp.apply(
        _categorize_ranges, axis=1
    )
    Logger.info(
        "RT overlap counts: %s", maxquant_ref_and_exp["RT_overlap"].value_counts()
    )
    plt.pie(
        maxquant_ref_and_exp["RT_overlap"].value_counts().values,
        labels=maxquant_ref_and_exp["RT_overlap"].value_counts().index,
        autopct=lambda x: _perc_fmt(x, maxquant_ref_and_exp.shape[0]),
    )
    save_plot(save_dir=save_dir, fig_type_name="PieChart_", fig_spec_name="RT_overlap")
    return maxquant_ref_and_exp


def filter_merged_by_rt_overlap(
    maxquant_ref_and_exp: pd.DataFrame,
    condition: List[Literal["full_overlap", "partial_overlap", "no_overlap"]]
    | None = None,
):
    """filter the merged maxquant result by RT_overlap condition"""
    if condition is None:
        condition = ["full_overlap", "partial_overlap"]
        Logger.info("No RT_overlap condition given, using %s", condition)
    n_pre_filter = maxquant_ref_and_exp.shape[0]
    filtered = maxquant_ref_and_exp.loc[
        maxquant_ref_and_exp["RT_overlap"].isin(condition), :
    ]
    n_post_filter = filtered.shape[0]
    Logger.info(
        "Removing %s entries with RT_overlap %s, %s entries left.",
        n_pre_filter - n_post_filter,
        condition,
        n_post_filter,
    )
    Logger.debug("columns after filter by RT %s", n_post_filter)
    return filtered


def sum_pcm_intensity_from_exp(maxquant_ref_and_exp: pd.DataFrame):
    """sum the intensity of the precursors from the experiment file

    In case of multiple PCM start and finish are the RT range of the precursor
    """
    n_pre_agg = maxquant_ref_and_exp.shape[0]
    maxquant_ref_and_exp_sum_intensity = (
        maxquant_ref_and_exp.groupby(["Modified sequence", "Charge"])
        .agg(
            {
                "Calibrated retention time start": "min",
                "Calibrated retention time finish": "max",
                "Calibrated retention time": "median",
                "Retention time": "median",
                "Intensity": "sum",
                "id": "first",
                "Mass": "first",
                "m/z": "first",
                "Length": "first",
            }
        )
        .reset_index()
    )
    n_post_agg = maxquant_ref_and_exp_sum_intensity.shape[0]
    Logger.info(
        "Removing %s entries with aggregation over PCM, %s entries left.",
        n_pre_agg - n_post_agg,
        n_post_agg,
    )
    Logger.debug("columns after agg %s", maxquant_ref_and_exp_sum_intensity.columns)
    return maxquant_ref_and_exp_sum_intensity


def add_sum_act_cols(
    maxquant_ref_and_exp_sum_intensity: pd.DataFrame,
    maxquant_ref_sum_act_col: list,
    maxquant_ref_df: pd.DataFrame,
):
    """add the sum activation columns from the activation columns"""
    maxquant_ref_and_exp_sum_intensity_act = pd.merge(
        left=maxquant_ref_df[
            ["Modified sequence", "Charge", "predicted_RT"] + maxquant_ref_sum_act_col
        ],
        right=maxquant_ref_and_exp_sum_intensity,
        on=["Modified sequence", "Charge"],
        how="right",
    )
    # empty field in 'id' because some entries are in ss but not in exp.

    return maxquant_ref_and_exp_sum_intensity_act
