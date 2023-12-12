from cProfile import label
import logging
import os
from re import T
from typing import List, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optimization.dictionary import ExtractPeak
from utils.plot import plot_scatter, plot_venn2, save_plot

Logger = logging.getLogger(__name__)


def PlotTrueAndPredict(x, prediction, true, log: bool = False):
    if log:
        prediction = np.log10(prediction + 1)
        true = np.log10(true + 1)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0)

    axs[0].vlines(x=x, ymin=0, ymax=true)
    axs[1].vlines(x=x, ymin=0, ymax=prediction)


def PlotIsoPatternsAndScan(
    Maxquant_result,
    MS1Scans: pd.DataFrame | None = None,
    infer_intensity: Union[pd.Series, np.ndarray, None] = None,
    lower_plot: Literal["infer", "obs"] = "obs",
    scan_idx: Union[int, None] = None,
    precursor_idx: Union[List[int], None] = None,
    precursor_id: Union[List[int], None] = None,
    mzrange: Union[None, list] = None,
    log_intensity: bool = False,
    save_dir=None,
):
    # preprocess data
    # TODO: return within scan Pearson and Jaccard distance --> redundant
    # TODO: return atom composition --> IsoSpecPy incompatibility
    # TODO: clean code and if possible factorize part of it! --> scan by scan notebook

    # Find precursor index if only precursor id is provided:
    if precursor_id is not None:
        precursor_idx = Maxquant_result.loc[
            Maxquant_result["id"].isin(precursor_id)
        ].index

    match lower_plot:
        case "obs":
            # Find an appropriate scan if not provided
            if scan_idx is None:
                if precursor_idx is None:
                    raise ValueError("Please provide a precursor index.")
                RT = np.max(
                    Maxquant_result.loc[precursor_idx, "Retention time"].values
                )  # take the later RT of the precursors
                scan_idx = np.abs(MS1Scans["starttime"] - RT).argmin()
                scan_time = MS1Scans.loc[scan_idx, "starttime"]
                Logger.info(
                    "Precursors %s retention time %s, \n show later RT %s with"
                    " corresponding scan index %s         with scan time %s",
                    precursor_idx,
                    Maxquant_result.loc[precursor_idx, "Retention time"].values,
                    RT,
                    scan_idx,
                    scan_time,
                )
            OneScan = MS1Scans.iloc[scan_idx, :]
            OneScanMZ = np.array(OneScan["mzarray"])
            IsoMZ = None

            # Find the range of mz in MS1 scan to plot
            if precursor_idx is not None:
                IsoMZ = Maxquant_result.loc[precursor_idx, "IsoMZ"]
                IsoMZ_flatten = np.concatenate(IsoMZ.values).ravel()
                IsoMZ_range = [np.min(IsoMZ_flatten) - 1, np.max(IsoMZ_flatten) + 1]
                OneScanMZinRange = OneScanMZ[
                    (OneScanMZ > IsoMZ_range[0]) & (OneScanMZ < IsoMZ_range[1])
                ]
                OneScanMZinRangeIdx = np.where(
                    (OneScanMZ > IsoMZ_range[0]) & (OneScanMZ < IsoMZ_range[1])
                )[0]
            else:
                if not isinstance(mzrange, list):
                    raise TypeError(
                        "mzrange should be a list, or provide an int for precursor"
                        " index."
                    )
                OneScanMZinRange = OneScanMZ[
                    (OneScanMZ > mzrange[0]) & (OneScanMZ < mzrange[1])
                ]
                OneScanMZinRangeIdx = np.where(
                    (OneScanMZ > mzrange[0]) & (OneScanMZ < mzrange[1])
                )[0]

            # Calculating values for visualization
            Intensity = np.array(OneScan["intarray"])[OneScanMZinRangeIdx]
            if log_intensity:  # +1 to avoid divide by zero error
                Intensity = np.log10(Intensity + 1)
            peak_results = ExtractPeak(x=OneScanMZinRange, y=Intensity)
            peaks_idx = peak_results["apex_mzidx"]
            print("Peak results:")
            print(peak_results)
        case "infer":
            if infer_intensity is None:
                raise ValueError("please provide infer_intensity.")
            Intensity = infer_intensity.values
            if log_intensity:
                Intensity = np.log10(infer_intensity.values + 1)
            if precursor_idx is not None:
                IsoMZ = Maxquant_result.loc[precursor_idx, "IsoMZ"]
                IsoMZ_flatten = np.concatenate(IsoMZ.values).ravel()
                IsoMZ_range = [np.min(IsoMZ_flatten) - 1, np.max(IsoMZ_flatten) + 1]
                InferinRange = Intensity[
                    (infer_intensity.index > IsoMZ_range[0])
                    & (infer_intensity.index < IsoMZ_range[1])
                ]
                InferinRangeIdx = infer_intensity.index[
                    (infer_intensity.index > IsoMZ_range[0])
                    & (infer_intensity.index < IsoMZ_range[1])
                ]

    # Plot
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    if precursor_idx is not None:
        colormap = plt.cm.bwr  # nipy_spectral, Set1,Paired
        colors = [colormap(i) for i in np.linspace(0, 1, len(precursor_idx))]
        for i, precursor in enumerate(precursor_idx):
            axs[0].vlines(
                x=Maxquant_result.loc[precursor, "IsoMZ"],
                ymin=0,
                ymax=Maxquant_result.loc[precursor, "IsoAbundance"],
                label=precursor,
                color=colors[i],
            )
            print(
                "Isotope Pattern:",
                precursor,
                Maxquant_result.loc[precursor, "IsoMZ"],
                Maxquant_result.loc[precursor, "IsoAbundance"],
            )
        axs[0].set_title("Up: Isotope Pattern, Down: MS1 Scan " + str(scan_idx))
    match lower_plot:
        case "obs":
            axs[1].vlines(x=OneScanMZinRange, ymin=-Intensity, ymax=0, label="MS peaks")
            axs[1].hlines(
                y=-peak_results["peak_height"],
                xmin=peak_results["start_mz"],
                xmax=peak_results["end_mz"],
                linewidth=2,
                color="black",
            )
            axs[1].vlines(
                x=OneScanMZinRange[peaks_idx],
                ymin=-Intensity[peaks_idx],
                ymax=0,
                color="orange",
                label="inferred apex",
            )
            axs[1].plot(
                OneScanMZinRange[peaks_idx],
                -Intensity[peaks_idx],
                "x",
                color="orange",
                label="inferred apex",
            )
        case "infer":
            Logger.debug(
                "infer m/z and intensities: %s, %s", InferinRangeIdx, InferinRange
            )
            axs[1].vlines(  # x = infer_intensity.index,
                x=InferinRangeIdx,
                ymin=-InferinRange,
                ymax=0,
                label="inferred intensity",
            )
            axs[1].set_xlim(IsoMZ_range)
    fig.legend(loc="center right", bbox_to_anchor=(1.15, 0.5))

    # save result
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        figname = (
            "SpecAndIsoPatterns_scan"
            + str(scan_idx)
            + "_precursor"
            + str(precursor_idx)
            + ".png"
        )
        plt.savefig(fname=os.path.join(save_dir, figname), dpi=300)
        plt.close()
    else:
        plt.show()

    return None


def ReportInensity(intensity: pd.Series, save_dir: Union[str, None]):
    print("--------------", intensity.name, "-----------------")
    n_MQ_nonzero = intensity[intensity > 1].count()  # always filter at 1
    print("Non zero intensity in", intensity.name, "=", n_MQ_nonzero)
    np.log10(intensity + 1).hist()  # type: ignore
    plt.title("Distribution of " + str(intensity.name) + " Intensity (log)")
    plt.xlabel("Intensity (log)")
    plt.ylabel("Count")
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, "distr_" + str(intensity.name) + ".png"), dpi=300
        )
        plt.close()


def ReportDistr(data: pd.Series, suffix: Union[str, None], save_dir: Union[str, None]):
    name = str(data.name)
    if suffix is not None:  # add suffix to name if desired
        name += "_" + suffix
    print("--------------", name, "-----------------")
    print(data.describe())
    data.hist()
    plt.title("Distribution of " + name)
    plt.xlabel(name)
    plt.ylabel("Count")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "distr_" + name + ".png"), dpi=300)
        plt.close()


def GenerateResultReport(
    scan_record: pd.DataFrame,
    # emptyScans:pd.DataFrame,
    # NonEmptyScans:pd.DataFrame,
    intensity_cols: list,
    save_dir: Union[str, None] = None,
):
    for col in intensity_cols:
        ReportInensity(intensity=col, save_dir=save_dir)

    scan_record["n_CandidateByRT"] = scan_record["CandidatePrecursorByRT"].apply(
        lambda x: len(x) if x is not None else 0
    )
    scan_record["n_filteredCandidate"] = scan_record["FilteredPrecursor"].apply(
        lambda x: len(x) if x is not None else 0
    )
    scan_record["preservedRatio_IE_filter"] = (
        scan_record["n_filteredCandidate"] / scan_record["n_CandidateByRT"]
    )

    emptyScans = scan_record.loc[scan_record["n_filteredCandidate"] <= 1]
    NonEmptyScans = scan_record.loc[scan_record["n_filteredCandidate"] > 1]
    print("--------------Empty Scans-----------------")
    for i in [
        "Time",
        "n_CandidateByRT",
        "n_filteredCandidate",
        "preservedRatio_IE_filter",
    ]:
        try:
            ReportDistr(data=emptyScans[i], suffix="EmptyScans", save_dir=save_dir)
        except:
            print("Column ", i, "does not exist!")

    print("--------------Non Empty Scans-----------------")
    for i in [
        "Loss",
        "Cosine Dist",
        "n_CandidateByRT",
        "n_filteredCandidate",
        "preservedRatio_IE_filter",
        "IntensityExplained",
        "PeaksExplained",
        "NumberHighlyCorrDictCandidate",
    ]:
        try:
            ReportDistr(
                data=NonEmptyScans[i], suffix="nonEmptyScans", save_dir=save_dir
            )
        except:
            print("Column ", i, "does not exist!")

    sns.lineplot(x=NonEmptyScans["Time"], y=NonEmptyScans["Cosine Dist"])
    plt.title("Cosine Distance By Scan")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "cos_dist_byscan.png"), dpi=300)
        plt.close()

    return scan_record


def PlotCorr(
    ReferenceIntensity,
    SumActivation,
    data=None,
    filter_thres: float = 1,
    interactive: bool = False,
    show_diag: bool = True,
    color: Union[None, pd.Series] = None,
    hover_data: Union[None, List] = [
        "Modified sequence",
        "Leading proteins",
        "Charge",
        "id",
    ],
    save_dir: Union[None, str] = None,
):
    RegressionIntensity, AbsResidue, valid_idx = plot_scatter(
        x=ReferenceIntensity,
        y=SumActivation,
        log_x=True,
        log_y=True,
        data=data,
        filter_thres=filter_thres,
        interactive=interactive,
        hover_data=hover_data,
        show_diag=show_diag,
        color=color,
        title="Corr. of Quant. Results ",
        save_dir=save_dir,
        x_label="Reference (log)",
        y_label="Infered (log)",
    )
    return RegressionIntensity, AbsResidue, valid_idx


def PlotActivation(
    MaxquantEntry: pd.Series,
    PrecursorTimeProfiles: list,
    PrecursorTimeProfileLabels: list,
    MS1ScansNoArray: pd.DataFrame,
    RT_tol: float = 0.0,
    log_intensity: bool = False,
    x_ticks: Literal["time", "scan index"] = "time",
    RT_ref: str = "predicted_RT",
    save_dir=None,
):
    EluteRange = [
        MaxquantEntry["Calibrated retention time start"].values[0],
        MaxquantEntry["Calibrated retention time finish"].values[0],
    ]
    RTrange = [
        MaxquantEntry[RT_ref].values[0] - RT_tol,
        MaxquantEntry[RT_ref].values[0] + RT_tol,
    ]
    Logger.debug("RTrange: %s", RTrange)
    ScanIdx = MS1ScansNoArray[
        (MS1ScansNoArray["starttime"] >= RTrange[0])
        & (MS1ScansNoArray["starttime"] <= RTrange[1])
    ].index
    Logger.debug("ScanIdx: %s", ScanIdx)
    TimeProfiles = pd.DataFrame(
        dict(zip(PrecursorTimeProfileLabels, PrecursorTimeProfiles))
    )
    TimeProfiles = TimeProfiles.set_index(MS1ScansNoArray["starttime"])
    ActivationInRange = TimeProfiles.iloc[ScanIdx, :]

    if log_intensity:
        ActivationInRange = np.log10(ActivationInRange + 1)
    sns.scatterplot(data=ActivationInRange, legend=False)  # type: ignore
    sns.lineplot(data=ActivationInRange, legend="brief")

    # experimental RT range
    for x in EluteRange:
        plt.axvline(x=x, linewidth=2, color="black", label="dict RT range")
    plt.axvline(
        x=MaxquantEntry["Retention time"].values[0],
        linewidth=2,
        color="grey",
        label="dict RT",
    )
    Logger.info(
        "dict RT can be either from reference file or from experiment file, depending"
        " on the MaxquantEntry specified."
    )

    # reference RT for reconstruction
    for x in RTrange:
        plt.axvline(x=x, linewidth=2, color="red", label="SBS_RT_tol")
    plt.axvline(
        x=MaxquantEntry[RT_ref].values[0],
        linewidth=2,
        color="orange",
        label="SBS_RT_ref",
    )

    plt.legend(title="Smoothing", bbox_to_anchor=(1.5, 1), loc="upper right")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Activation")
    title = (
        MaxquantEntry["Modified sequence"].values[0]
        + ", "
        + str(MaxquantEntry["Charge"].values[0])
    )
    plt.title(title)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(
            fname=os.path.join(save_dir, title.replace("|", "_") + ".png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    ActivationInRange_df = ActivationInRange
    ActivationInRange_df["Scan index"] = ScanIdx
    ActivationInRange_df = ActivationInRange_df.reset_index()
    if x_ticks == "scan index":
        x, loc = plt.xticks()
        print(loc)
        plt.xticks(
            x,
            ActivationInRange_df.loc[
                ActivationInRange_df["starttime"].isin(x), "Scan index"
            ].values,
        )
        plt.xlabel("Scan index")
    return ActivationInRange_df


def FindStartAndEndScan(activation: np.ndarray, thres: float = 1.0):
    """
    Given the activation matrix, finds the first and last value for each row (precursor)
    that is larger than given threshold

    :actiavtion:
    :thres:
    """
    # Find the indices where the values are greater than 1
    row_indices, col_indices = np.where(activation > thres)

    # Group col_indices by row_indices
    grouped_indices = pd.Series(col_indices).groupby(row_indices)

    # Calculate the minimum and maximum values for each group
    min_indices = grouped_indices.min()
    max_indices = grouped_indices.max()

    # Create a DataFrame to store the results
    df = pd.DataFrame(
        {
            "id": min_indices.index,
            "Scan Index_start_SBS": min_indices.values,
            "Scan Index_end_SBS": max_indices.values,
        }
    )
    df["CountScan_SBS"] = df["Scan Index_start_SBS"] - df["Scan Index_end_SBS"] + 1

    return df


def plot_alphas_across_scan(
    scan_record: pd.DataFrame,
    x: Literal["Scan", "Time"] = "Scan",
):
    NonEmptyScans = scan_record.loc[scan_record["BestAlpha"] != None]
    EmptyScans = scan_record.loc[scan_record["BestAlpha"] == None]
    minimal_alpha = min(filter(lambda x: x > 0, NonEmptyScans["BestAlpha"]))
    zero_replace = minimal_alpha / 10
    Logger.info(
        "Alpha range: %s, replacing all zeros with %s",
        NonEmptyScans["BestAlpha"].unique(),
        zero_replace,
    )
    NonEmptyScans["alpha_to_plot"] = np.log10(
        NonEmptyScans["BestAlpha"].replace(0, zero_replace, inplace=False)
    )

    plt.plot(NonEmptyScans[x], NonEmptyScans["alpha_to_plot"])  # alphas
    plt.scatter(
        x=EmptyScans[x],
        y=np.repeat(np.log10(zero_replace), EmptyScans.shape[0]),
        s=10,
        marker="x",
        c="black",
    )  # empty scans
    plt.xlabel(x)
    plt.ylabel("Alpha (log)")
    plt.title("Best Alpha over " + x)


def compare_act_sum_with_MQ(
    MQ_exp: pd.DataFrame,
    MQ_dict: pd.DataFrame,
    RT_tol: float,
    MQ_dict_sum_act_col: list,
    return_ori_merge: bool = False,
):
    """compare the inferred intensities from maxquant and SBS,
        when a different dictionary then experiment MQ result is used

    :MQ_exp: the maxquant result of the raw data whose MS1 scans were used for inference
    :MQ_dict: the maxquant result used for SBS inference
    :RT_tol: the RT tolerence used for SBS inference
    """

    # MQ_dict = MQ_dict[['Modified sequence', 'Charge', 'predicted_RT', 'SumActivation']]
    MQ_merged_dict = pd.merge(
        left=MQ_dict[["Modified sequence", "Charge", "predicted_RT", "id"]],
        right=MQ_exp[
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
    Logger.debug("Maxquant experiment file has %s entries.", MQ_exp.shape[0])
    Logger.debug("columns after merge MQ dict and MQ exp %s", MQ_merged_dict.columns)
    MQ_merged_dict["RT_diff"] = (
        MQ_merged_dict["predicted_RT"] - MQ_merged_dict["Retention time"]
    )

    MQ_merged_filtered_RT = MQ_merged_dict.loc[abs(MQ_merged_dict["RT_diff"]) <= RT_tol]
    Logger.debug(
        "Keeping %s entries with RT difference within %s",
        MQ_merged_filtered_RT.shape[0],
        RT_tol,
    )
    Logger.debug("columns after filter by RT %s", MQ_merged_filtered_RT.columns)

    MQ_merged_filtered_RT_sum_intensity = (
        MQ_merged_filtered_RT.groupby(["Modified sequence", "Charge"])
        .agg(
            {
                "Calibrated retention time start": "min",
                "Calibrated retention time finish": "max",
                "Calibrated retention time": "median",
                "Retention time": "median",
                "Intensity": "sum",
                "id": "first",
            }
        )
        .reset_index()
    )
    Logger.debug(
        "Aggregate by Modified sequence and Charge, keeping %s entries",
        MQ_merged_filtered_RT_sum_intensity.shape[0],
    )
    Logger.debug("columns after agg %s", MQ_merged_filtered_RT_sum_intensity.columns)

    MQ_merged_final = pd.merge(
        left=MQ_dict[
            ["Modified sequence", "Charge", "predicted_RT"] + MQ_dict_sum_act_col
        ],
        right=MQ_merged_filtered_RT_sum_intensity,
        on=["Modified sequence", "Charge"],
        how="right",
    )
    MQ_merged_final["id"] = MQ_merged_final["id"].astype(int)
    if return_ori_merge:
        return MQ_merged_final, MQ_merged_dict
    else:
        return MQ_merged_final


class SBSResult:
    def __init__(
        self,
        ref_df: pd.DataFrame,
        exp_df: pd.DataFrame,
        RT_tol: float,
        sum_raw: pd.DataFrame | None = None,
        sum_gaussian: pd.DataFrame | None = None,
        sum_minima: pd.DataFrame | None = None,
        sum_peak: pd.DataFrame | None = None,
        SumActCol: List[str] | None = None,
    ) -> None:
        assert any(
            item is not None for item in [sum_raw, sum_peak, sum_gaussian, sum_minima]
        )
        sum_cols = []
        for sum in [sum_raw, sum_peak, sum_gaussian, sum_minima]:
            if sum is not None:
                sum.reset_index(drop=True, inplace=True)
                sum_cols += list(sum.columns)
        if SumActCol is None:
            self.SumActCol = sum_cols
        else:
            self.SumActCol = SumActCol
        pp_sumactivation = pd.concat(
            [
                sum_raw,
                sum_gaussian,
                sum_minima,
                sum_peak,
            ],
            axis=1,
        )

        pp_sumactivation = pp_sumactivation.set_index(ref_df.index)
        self.ref_df = pd.concat([ref_df, pp_sumactivation], axis=1)
        self.ref_df["Reverse"] = np.where(ref_df["Reverse"].isnull(), 0, 1)
        self.exp_df = exp_df.copy()
        self.exp_df["Reverse"] = np.where(exp_df["Reverse"].isnull(), 0, 1)
        self.ref_exp_df_inner, self.ref_exp_df_outer = compare_act_sum_with_MQ(
            MQ_dict=self.ref_df,
            MQ_exp=self.exp_df,
            RT_tol=RT_tol,
            MQ_dict_sum_act_col=self.SumActCol,
            return_ori_merge=True,
        )
        try:
            self.ref_df_non_zero = self.ref_df.loc[
                self.ref_df["SumActivationRaw"] > 0, :
            ]
        except:
            Logger.warn(
                "SumActivationRaw column does not exist, use %s for generating"
                " ref_df_non_zero."
            )
            self.ref_df_non_zero = self.ref_df.loc[
                self.ref_df[self.SumActCol[0]] > 0, :
            ]

    def plot_intensity_corr(
        self,
        ref_col: str = "Intensity",
        inf_col: str = "AUCActivationRaw",
        interactive: bool = False,
        save_dir: str | None = None,
        **kwargs,
    ):
        RegIntensity, AbsResidue, valid_idx = PlotCorr(
            self.ref_exp_df_inner[ref_col],
            self.ref_exp_df_inner[inf_col],
            data=self.ref_exp_df_inner,
            interactive=interactive,
            save_dir=save_dir,
            **kwargs,
        )

    def plot_overlap_with_MQ(self, save_dir: str | None = None):
        self.exp_df_unique_PCM = (
            self.exp_df.groupby(["Modified sequence", "Charge", "Reverse"])
            .agg(
                {
                    "Retention time": "median",
                    "Intensity": "sum",
                    "Calibrated retention time start": "min",
                    "Calibrated retention time finish": "max",
                }
            )
            .reset_index()
        )
        self.exp_df_unique_PCM["precursor"] = self.exp_df_unique_PCM[
            "Modified sequence"
        ] + self.exp_df_unique_PCM["Charge"].astype(str)

        self.ref_df_non_zero["precursor"] = self.ref_df_non_zero[
            "Modified sequence"
        ] + self.ref_df_non_zero["Charge"].astype(str)

        plot_venn2(
            set1=set(
                self.exp_df_unique_PCM.loc[
                    self.exp_df_unique_PCM["Reverse"] == 0, "precursor"
                ].to_list()
            ),
            set2=set(
                self.ref_df_non_zero.loc[
                    self.ref_df_non_zero["Reverse"] == 0, "precursor"
                ].to_list()
            ),
            label1="Maxquant",
            label2="SBS",
            save_dir=save_dir,
            title="IdentificationOfTarget",
        )

        plot_venn2(
            set1=set(
                self.exp_df_unique_PCM.loc[
                    (self.exp_df_unique_PCM["Reverse"] == 0)
                    & (self.exp_df_unique_PCM["Intensity"] > 0),
                    "precursor",
                ].to_list()
            ),
            set2=set(
                self.ref_df_non_zero.loc[
                    self.ref_df_non_zero["Reverse"] == 0, "precursor"
                ].to_list()
            ),
            label1="Maxquant",
            label2="SBS",
            save_dir=save_dir,
            title="QuantificationOfTarget",
        )

    def eval_target_decoy(
        self, ref_col: str = "AUCActivationRaw", save_dir: str | None = None
    ):
        self.TDC_table = self.ref_df_non_zero.groupby("Reverse").agg(
            {"id": "count", ref_col: "mean"}
        )
        fig, axs = plt.subplots(ncols=3, width_ratios=[1, 1, 2], figsize=(12, 5))
        sns.countplot(data=self.ref_df_non_zero, x="Reverse", ax=axs[0])
        axs[0].bar_label(axs[0].containers[0])
        sns.barplot(
            data=self.ref_df_non_zero, x="Reverse", y="AUCActivationRaw", ax=axs[1]
        )
        axs[1].bar_label(axs[1].containers[0])
        axs[1].set_ylabel("Mean Activation")
        sns.kdeplot(
            data=self.ref_df_non_zero,
            x=np.log10(1 + self.ref_df_non_zero["AUCActivationRaw"]),
            hue="Reverse",
            ax=axs[2],
        )
        axs[2].set_xlabel("Activation (log)")
        save_plot(save_dir=save_dir, fig_type_name="TDC", fig_spec_name="")
        return self.TDC_table
