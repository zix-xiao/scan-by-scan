import numpy as np
import pandas as pd

import IsoSpecPy as iso
import matplotlib.pyplot as plt

from typing import Union, Literal
from utils.plot import plot_comparison
from utils.config import _AlignMethods
from utils.tools import ExtractPeak
import logging
from sklearn.metrics import jaccard_score

Logger = logging.getLogger(__name__)


def CalcModpeptIsopattern(
    modpept: str, charge: int, ab_thres: float = 0.005, mod_CAM: bool = True
):
    """takes a peptide sequence with modification and charge,
    calculate and return the two LISTs of isotope pattern with all isotopes m/z value
    with abundance larger than ab_thres, both sorted by isotope mass

    :modpept: str
    :charge: charge state of the percursor, int
    :mzrange: limitation of detection of mz range
    :mm: bin size of mz value, int
    :ab_thres: the threshold for filtering isotopes, float

    return: two list
    """

    # account for extra atoms from modification and water
    # count extra atoms
    n_H = 2 + charge  # 2 from water and others from charge (proton)
    n_Mox = modpept.count("M(ox)")
    modpept = modpept.replace("(ox)", "")
    n_acetylN = modpept.count("(ac)")
    modpept = modpept.replace("(ac)", "")
    if mod_CAM:
        n_C = modpept.count("C")
    else:
        n_C = 0
    # addition of extra atoms
    atom_composition = iso.ParseFASTA(modpept)
    atom_composition["H"] += 3 * n_C + n_H + 2 * n_acetylN
    atom_composition["C"] += 2 * n_C + 2 * n_acetylN
    atom_composition["N"] += 1 * n_C
    atom_composition["O"] += 1 * n_C + 1 + n_acetylN + 1 * n_Mox

    # Isotope calculation
    formula = "".join(
        ["%s%s" % (key, value) for key, value in atom_composition.items()]
    )
    iso_distr = iso.IsoThreshold(formula=formula, threshold=ab_thres, absolute=True)
    iso_distr.sort_by_mass()
    mz_sortByMass = iso_distr.np_masses() / charge
    probs_sortByMass = iso_distr.np_probs()

    return mz_sortByMass, probs_sortByMass


# TODO: main method is moved to Dict class, consider archieve it
def AlignMZ(
    anchor: pd.DataFrame,
    precursorRow: pd.Series,
    col_to_align=["mzarray_obs", "mzarray_calc"],
    mz_tol=1e-4,
    primaryAbundanceThres: float = 0.05,
    AbundanceMissingThres: float = 0.4,
    method: _AlignMethods = "2stepNN",  # case peakRange is moved to Dict class
    verbose=False,
):
    sample = pd.DataFrame(
        {
            "mzarray_calc": precursorRow["IsoMZ"],
            "abundance": precursorRow["IsoAbundance"],
        }
    )
    alignment = None
    mzDelta_mean = np.nan
    mzDelta_std = np.nan
    match method:
        case "2stepNN":
            primaryIsotope = sample.loc[sample["abundance"] >= primaryAbundanceThres]
            primaryAlignment = pd.merge_asof(
                left=anchor.sort_values(col_to_align[0]),
                right=primaryIsotope.sort_values(col_to_align[1]),
                left_on=col_to_align[0],
                right_on=col_to_align[1],
                tolerance=mz_tol,
                direction="nearest",
            ).dropna(
                axis=0
            )  # type: ignore
            if primaryAlignment.shape[0] > 0:
                primaryAlignment["alignmentRun"] = "primary"
                anchor = anchor[
                    ~anchor["mzarray_obs"].isin(primaryAlignment["mzarray_obs"])
                ]
                secondaryIsotope = sample.loc[
                    sample["abundance"] < primaryAbundanceThres
                ]
                secondaryAlignment = pd.merge_asof(
                    left=anchor.sort_values(col_to_align[0]),
                    right=secondaryIsotope.sort_values(col_to_align[1]),
                    left_on=col_to_align[0],
                    right_on=col_to_align[1],
                    tolerance=mz_tol,
                    direction="nearest",
                ).dropna(
                    axis=0
                )  # type: ignore
                secondaryAlignment["alignmentRun"] = "secondary"
                alignment = pd.concat([primaryAlignment, secondaryAlignment], axis=0)
                alignment["mzDelta"] = (
                    alignment["mzarray_obs"] - alignment["mzarray_calc"]
                )
                mzDelta_mean = alignment["mzDelta"].mean()
                mzDelta_std = alignment["mzDelta"].std()

    if alignment is not None:
        IsotopeNotObs = sample[~sample["mzarray_calc"].isin(alignment["mzarray_calc"])]
        AbundanceNotObs = IsotopeNotObs["abundance"].sum()
        n_matchedIso = alignment.shape[0]

    else:
        IsotopeNotObs = sample
        AbundanceNotObs = 1
        n_matchedIso = 0
    IsKept = AbundanceNotObs <= AbundanceMissingThres
    if verbose:
        return (
            n_matchedIso,
            AbundanceNotObs,
            IsKept,
            mzDelta_mean,
            mzDelta_std,
            alignment,
            IsotopeNotObs,
        )
    else:
        return (
            n_matchedIso,
            AbundanceNotObs,
            IsKept,
            mzDelta_mean,
            mzDelta_std,
            None,
            None,
        )


def _get_RT_edge(
    precursorRow: pd.Series,
    MS1Scans: pd.DataFrame,
    ScanIdx: int,
    direction: Literal[1, -1],
    ScanIdx_left: int,
    ScanIdx_right: int,
    IsInLastScan: Union[None, bool] = True,
    AbundanceMissingThres: float = 0.4,
):
    """
    Given a seeding scan index 'ScanIdx' that contains the target precursor,
    find the closest edge.

    :precursorRow:
    :MS1Scans:
    :ScanIdx:
    :direction: the direction for which search space is extended,
                1 stand for right range and -1 stand for left range
    :ScanIdx_left: the left edge of search limit
    :ScanIdx_right: the right edge of search limit
    :IsInLastScan: whether the precursor is in the previous scan (by time)
    :AbundanceMissingThres:

    """

    # Calculate IsInThisScan and IsInLastScan
    MS1Intensity = pd.DataFrame(
        {
            "mzarray_obs": MS1Scans.iloc[ScanIdx, :]["mzarray"],
            "intensity": MS1Scans.iloc[ScanIdx, :]["intarray"],
        }
    )
    if IsInLastScan is None:
        _, _, IsInLastScan, _, _, _, _ = AlignMZ(
            anchor=MS1Intensity,
            precursorRow=precursorRow,
            verbose=False,
            method="peakRange",
            AbundanceMissingThres=AbundanceMissingThres,
        )
    if IsInLastScan:
        Logger.debug(
            "Is in scan %s and search for %s scan %s",
            ScanIdx,
            direction,
            ScanIdx + direction,
        )
        ScanIdx += direction
        MS1Intensity = pd.DataFrame(
            {
                "mzarray_obs": MS1Scans.iloc[ScanIdx, :]["mzarray"],
                "intensity": MS1Scans.iloc[ScanIdx, :]["intarray"],
            }
        )
        _, _, IsInThisScan, _, _, _, _ = AlignMZ(
            anchor=MS1Intensity,
            precursorRow=precursorRow,
            verbose=False,
            method="peakRange",
            AbundanceMissingThres=AbundanceMissingThres,
        )
    else:
        if ScanIdx <= ScanIdx_right and ScanIdx >= ScanIdx_left:
            Logger.debug(
                "Is not in scan %s and search for %s scan %s",
                ScanIdx,
                -direction,
                ScanIdx - direction,
            )
            ScanIdx -= direction
            MS1Intensity = pd.DataFrame(
                {
                    "mzarray_obs": MS1Scans.iloc[ScanIdx, :]["mzarray"],
                    "intensity": MS1Scans.iloc[ScanIdx, :]["intarray"],
                }
            )
            _, _, IsInThisScan, _, _, _, _ = AlignMZ(
                anchor=MS1Intensity,
                precursorRow=precursorRow,
                verbose=False,
                method="peakRange",
                AbundanceMissingThres=AbundanceMissingThres,
            )
        else:
            IsInThisScan = 3

    # Recursive behavior
    match (int(IsInLastScan) + int(IsInThisScan)):
        case 1:
            Logger.info("Found scan index with direction %s: %s", direction, ScanIdx)
            return ScanIdx
        case 3 | 4:
            Logger.info("Scan index out of predefined range, stop searching")
            return None
        case 0 | 2:  # consecutive N or Y
            return _get_RT_edge(
                precursorRow=precursorRow,
                MS1Scans=MS1Scans,
                ScanIdx_left=ScanIdx_left,
                ScanIdx_right=ScanIdx_right,
                direction=direction,
                ScanIdx=ScanIdx,
                IsInLastScan=IsInThisScan,
            )


def _search_BiDir_scans(
    precursorRow: pd.Series,
    MS1Scans: pd.DataFrame,
    ScanIdx: int,
    ScanIdx_left: int,
    ScanIdx_right: int,
    step: int,
    AbundanceMissingThres: float = 0.4,
):
    """
    Given a seeding scan (that does not contain target precursor),
    Search for left and right scan until target is found or reach search limit.

    :precursorRow:
    :MS1Scans:
    :ScanIdx:
    :ScanIdx_left: the left edge of search limit
    :ScanIdx_right: the right edge of search limit
    :step: the distance (+/-) between candidate scans and start scan
    :AbundanceMissingThres:
    """
    if (
        ScanIdx - step >= ScanIdx_left
    ):  # ensure search limit, only use left because of symmetricality
        MS1Intensity_next_left = pd.DataFrame(
            {
                "mzarray_obs": MS1Scans.iloc[ScanIdx - step, :]["mzarray"],
                "intensity": MS1Scans.iloc[ScanIdx - step, :]["intarray"],
            }
        )
        _, _, IsInNextLeft, _, _, _, _ = AlignMZ(
            anchor=MS1Intensity_next_left,
            precursorRow=precursorRow,
            verbose=False,
            method="peakRange",
            AbundanceMissingThres=AbundanceMissingThres,
        )
        MS1Intensity_next_right = pd.DataFrame(
            {
                "mzarray_obs": MS1Scans.iloc[ScanIdx + step, :]["mzarray"],
                "intensity": MS1Scans.iloc[ScanIdx + step, :]["intarray"],
            }
        )
        _, _, IsInNextRight, _, _, _, _ = AlignMZ(
            anchor=MS1Intensity_next_right,
            precursorRow=precursorRow,
            verbose=False,
            method="peakRange",
            AbundanceMissingThres=AbundanceMissingThres,
        )

        match (IsInNextLeft, IsInNextRight):
            case (0, 0):
                Logger.debug(
                    "Precursor %s not observed in scan %s and %s, search with increased"
                    " step",
                    precursorRow["id"],
                    ScanIdx - step,
                    ScanIdx + step,
                )
                step += 1
                return _search_BiDir_scans(
                    precursorRow,
                    MS1Scans,
                    ScanIdx,
                    ScanIdx_left=ScanIdx_left,
                    ScanIdx_right=ScanIdx_right,
                    step=step,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
            case (0, 1):
                Logger.debug(
                    "Precursor %s not observed in scan %s but in %s, search for right"
                    " edge",
                    precursorRow["id"],
                    ScanIdx - step,
                    ScanIdx + step,
                )
                Left = ScanIdx + step
                Right = _get_RT_edge(
                    precursorRow=precursorRow,
                    MS1Scans=MS1Scans,
                    ScanIdx=ScanIdx + step,
                    direction=1,
                    ScanIdx_left=ScanIdx + step,
                    ScanIdx_right=ScanIdx_right,
                    IsInLastScan=True,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
                return Left, Right
            case (1, 0):
                Logger.debug(
                    "Precursor %s observed in scan %s but not in %s, search for left"
                    " edge",
                    precursorRow["id"],
                    ScanIdx - step,
                    ScanIdx + step,
                )
                Right = ScanIdx - step
                Left = _get_RT_edge(
                    precursorRow=precursorRow,
                    MS1Scans=MS1Scans,
                    ScanIdx=ScanIdx - step,
                    direction=-1,
                    ScanIdx_left=ScanIdx_left,
                    ScanIdx_right=ScanIdx - step,
                    IsInLastScan=True,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
                return Left, Right
            case (1, 1):
                Logger.warning(
                    "Precursor %s observed in equal distance scan %s and %s,           "
                    "                      incorporate empty scans in the middle",
                    precursorRow["id"],
                    ScanIdx - step,
                    ScanIdx + step,
                )
                Left = _get_RT_edge(
                    precursorRow=precursorRow,
                    MS1Scans=MS1Scans,
                    ScanIdx=ScanIdx - step,
                    direction=-1,
                    ScanIdx_left=ScanIdx_left,
                    ScanIdx_right=ScanIdx - step,
                    IsInLastScan=True,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
                Right = _get_RT_edge(
                    precursorRow=precursorRow,
                    MS1Scans=MS1Scans,
                    ScanIdx=ScanIdx + step,
                    direction=1,
                    ScanIdx_left=ScanIdx - step,
                    ScanIdx_right=ScanIdx_right,
                    IsInLastScan=True,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
                return Left, Right
    else:
        Logger.info("Scan index out of predefined range, stop searching")
        return None, None


def locate_RT_range(
    precursorRow: pd.Series,
    MS1Scans: pd.DataFrame,
    ScanIdx: int,
    search_range: int = 100,
    step: int = 1,
    AbundanceMissingThres: float = 0.4,
):
    """
    Given a seeding scan index 'ScanIdx', find the start (side = 1) or end (side = -1) scan.

    ScanIdx has an impact on the final result, it only finds the nearest
    fulfilling condition.
    [IMPORTANT] Assumption is that ScanIdx (starting scan)
    needs to be closed enough to truth, else it will stop at the closet
    occurrence.
    Use two case scenario: whether the starting seed scan contains
    the target or not

    :precursorRow:
    :MS1Scans:
    :ScanIdx:
    :search_range: the number of scans to be searched till stop
    :AbundanceMissingThres:
    """
    ScanIdx_left = ScanIdx - search_range
    ScanIdx_right = ScanIdx + search_range
    Logger.debug(
        "Start scan = %s, Scan edge = (%s, %s)", ScanIdx, ScanIdx_left, ScanIdx_right
    )
    MS1Intensity = pd.DataFrame(
        {
            "mzarray_obs": MS1Scans.iloc[ScanIdx, :]["mzarray"],
            "intensity": MS1Scans.iloc[ScanIdx, :]["intarray"],
        }
    )

    _, _, IsInThisScan, _, _, _, _ = AlignMZ(
        anchor=MS1Intensity,
        precursorRow=precursorRow,
        verbose=False,
        method="peakRange",
        AbundanceMissingThres=AbundanceMissingThres,
    )
    if IsInThisScan:
        Logger.debug(
            "Precursor %s observed in scan %s, search for left and right edge",
            precursorRow["id"],
            ScanIdx,
        )
        Left = _get_RT_edge(
            precursorRow,
            MS1Scans,
            ScanIdx=ScanIdx - 1,
            direction=-1,
            ScanIdx_right=ScanIdx_right,
            ScanIdx_left=ScanIdx_left,
            IsInLastScan=True,
            AbundanceMissingThres=AbundanceMissingThres,
        )
        Right = _get_RT_edge(
            precursorRow,
            MS1Scans,
            ScanIdx=ScanIdx + 1,
            direction=1,
            ScanIdx_right=ScanIdx_right,
            ScanIdx_left=ScanIdx_left,
            IsInLastScan=True,
            AbundanceMissingThres=AbundanceMissingThres,
        )
        return Left, Right

    else:
        Logger.debug(
            "Precursor %s is not observed in seeding Scan %s, start searching scan %s"
            " and %s.",
            precursorRow["id"],
            ScanIdx,
            ScanIdx - step,
            ScanIdx + step,
        )
        return _search_BiDir_scans(
            precursorRow=precursorRow,
            MS1Scans=MS1Scans,
            ScanIdx=ScanIdx,
            ScanIdx_left=ScanIdx_left,
            ScanIdx_right=ScanIdx_right,
            step=step,
            AbundanceMissingThres=AbundanceMissingThres,
        )


class Dict:
    def __init__(
        self,
        CandidateByRT: pd.DataFrame,
        OneScan: Union[pd.DataFrame, pd.Series],
        AbundanceMissingThres: float = 0.4,
        rel_height: float = 0.75,
    ) -> None:
        self.n_cand_by_RT = CandidateByRT.shape[0]
        Logger.info("Number of candidates by RT %s", self.n_cand_by_RT)

        self.AbundanceMissingThres = AbundanceMissingThres
        self.MS1Intensity = pd.DataFrame(
            {"mzarray_obs": OneScan["mzarray"], "intensity": OneScan["intarray"]}
        )
        self.peak_results = ExtractPeak(
            np.array(self.MS1Intensity["mzarray_obs"]),
            np.array(self.MS1Intensity["intensity"]),
            rel_height=rel_height,
        )
        CandidateByRT_iso = (
            CandidateByRT[["id", "IsoMZ", "IsoAbundance"]]
            .explode(["IsoMZ", "IsoAbundance"])
            .astype({"IsoMZ": "float64", "IsoAbundance": "float64"})
        )

        CandidateByRT_iso = pd.merge_asof(
            CandidateByRT_iso.sort_values("IsoMZ"),
            self.peak_results,
            left_on="IsoMZ",
            right_on="apex_mz",
            allow_exact_matches=True,
            direction="nearest",
        )
        CandidateByRT_iso["matched"] = (
            CandidateByRT_iso["IsoMZ"] >= CandidateByRT_iso["start_mz"]
        ) & (CandidateByRT_iso["IsoMZ"] <= CandidateByRT_iso["end_mz"])
        self.IsoAbundanceByID = (
            CandidateByRT_iso.groupby(["id", "matched"])["IsoAbundance"]
            .sum()
            .reset_index()
        )
        CandidateByRT_isoMatched = CandidateByRT_iso[(CandidateByRT_iso["matched"])]
        Logger.info(
            "Number of candidates after isotope match %s",
            len(CandidateByRT_isoMatched["id"].unique()),
        )
        if len(CandidateByRT_isoMatched["id"].unique()) > 0:
            CandidateByRT_isoFiltered = self.IsoAbundanceByID[
                self.IsoAbundanceByID["matched"]
                & (
                    self.IsoAbundanceByID["IsoAbundance"]
                    >= (1 - self.AbundanceMissingThres)
                )
            ]
            self.filteredCandidateIdx = (
                CandidateByRT_isoFiltered["id"].unique().tolist()
            )
            Logger.info(
                "Number of candidates after isotope abundance filter %s",
                len(self.filteredCandidateIdx),
            )
            if len(self.filteredCandidateIdx) > 0:
                CandidateByRT_isoFiltered_sumAbundance = (
                    CandidateByRT_isoMatched[
                        CandidateByRT_isoMatched["id"].isin(self.filteredCandidateIdx)
                    ]
                    .groupby(["id", "apex_mz"])["IsoAbundance"]
                    .sum()
                    .reset_index()
                )
                candid_dict = CandidateByRT_isoFiltered_sumAbundance.pivot(
                    index="apex_mz", columns="id", values="IsoAbundance"
                )
                self.dict = pd.merge(
                    left=self.peak_results[["apex_mz"]],
                    right=candid_dict,
                    left_on="apex_mz",
                    right_index=True,
                    how="left",
                ).set_index("apex_mz")
                self.dict.loc[0] = 1 - self.dict.sum(
                    axis=0
                )  # append mz for accumulated not observed iso abundance
                self.dict = self.dict.fillna(0)
                Logger.debug(
                    "[Double check] Number of candidates by RT and abundance filter %s",
                    len(self.dict.columns.values),
                )

                self.obs_int = pd.DataFrame(
                    {
                        "mzarray_obs": np.append(
                            self.peak_results["apex_mz"].values,
                            [0],  # append 0 mz for not observed iso abundance
                            axis=0,
                        ),
                        "intensity": np.append(
                            self.peak_results["peak_intensity_sum"].values, [0], axis=0
                        ),
                    }
                )
            else:
                self.dict = None
                self.filteredCandidateIdx = None
        else:
            self.dict = None
            self.filteredCandidateIdx = None

    def plot_observe_iso_abundance(self):
        self.IsoAbundanceByID[self.IsoAbundanceByID["matched"]]["IsoAbundance"].hist(
            bins=100
        )
        plt.title("Distribution of Observed Isotope Abundance")
        ymin, ymax = plt.ylim()
        plt.vlines(
            x=1 - self.AbundanceMissingThres,
            ymin=ymin,
            ymax=ymax,
            label="Abundance Thres",
            color="r",
        )

    def get_feature_corr(
        self,
        corr_thres: float = 0.9,
        calc_jaccard_sim: bool = True,
        plot_hmap: bool = True,
        plot_hist: bool = True,
    ):
        corr_matrix = self.dict.corr()
        if plot_hmap:
            corr_matrix.style.background_gradient(cmap="coolwarm")
            plt.matshow(corr_matrix)
            plt.show()
        self.sol = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        ).stack()
        self.high_corr_sol = self.sol[self.sol >= corr_thres]  # type: ignore
        if calc_jaccard_sim:
            jaccard_sim = []
            for pairs, _ in self.high_corr_sol.items():
                jaccard_sim.append(
                    jaccard_score(
                        self.dict.loc[:, pairs[0]] > 0,
                        self.dict.loc[:, pairs[1]] > 0,
                        average="binary",
                    )
                )
            self.high_corr_sol = pd.DataFrame(
                {"Pearson R": self.high_corr_sol, "Jaccard Similarity": jaccard_sim}
            )

        if plot_hist:
            self.sol.hist(log=True, bins=100)
            ymin, ymax = plt.ylim()
            plt.vlines(x=0.95, ymin=ymin, ymax=ymax, linestyles="dashed")
            plt.vlines(x=0.9, ymin=ymin, ymax=ymax, linestyles="dashed")
            plt.vlines(
                x=corr_thres, ymin=ymin, ymax=ymax, linestyles="dashed", color="r"
            )

        Logger.info(
            "Number of candidate pairs with correlation larger than %s: %s",
            corr_thres,
            self.high_corr_sol.shape[0],
        )
        tmp = self.high_corr_sol.reset_index()
        self.HighCorrCand = np.union1d(tmp["level_0"].unique(), tmp["level_1"].unique())
        Logger.info(
            "Number of candidated involved in high correlation: %s",
            len(self.HighCorrCand),
        )


# TODO: peakRange method is moved to Dict class, consider archieve
def ConstructDict(
    CandidatePrecursorsByRT: pd.DataFrame,
    OneScan: Union[pd.DataFrame, pd.Series],
    method: _AlignMethods = "2stepNN",
    AbundanceMissingThres: float = 0.4,
    mz_tol: float = 0.01,
    rel_height: float = 0.75,
):
    """
    Use Candidate precursors that are preselected using RT information
    to construct dictionary using isotope envelops

    TODO: add arg explanation
    """
    MS1Intensity = pd.DataFrame(
        {"mzarray_obs": OneScan["mzarray"], "intensity": OneScan["intarray"]}
    )
    peak_results = None
    logging.debug("Prepare data.")

    if method == "peakRange":
        peak_results = ExtractPeak(
            np.array(MS1Intensity["mzarray_obs"]),
            np.array(MS1Intensity["intensity"]),
            rel_height=rel_height,
        )
        merge_key = "apex_mz"
        CandidateDict = peak_results[[merge_key]]
        y_true = pd.DataFrame(
            {
                "mzarray_obs": peak_results["apex_mz"],
                "intensity": peak_results["peak_intensity_sum"],
            }
        )
        logging.debug("peak extraction")

    elif method == "2stepNN":
        merge_key = "mzarray_obs"
        CandidateDict = MS1Intensity[[merge_key]]
        y_true = MS1Intensity
        peak_results = None

    # MZ alignment with row operation
    AlignmentResult = CandidatePrecursorsByRT.copy()
    logging.info("number of row alignment %s", CandidatePrecursorsByRT.shape[0])
    (
        AlignmentResult.loc[:, "n_matchedIso"],
        AlignmentResult.loc[:, "AbundanceNotObs"],
        AlignmentResult.loc[:, "IsKept"],
        AlignmentResult.loc[:, "mzDelta_mean"],
        AlignmentResult.loc[:, "mzDelta_std"],
        alignment,
        IsotopeNotObs,
    ) = zip(
        *CandidatePrecursorsByRT.apply(
            lambda row: AlignMZ(
                MS1Intensity,
                row,
                method=method,
                # peak_results=peak_results,
                mz_tol=mz_tol,
                verbose=True,
                AbundanceMissingThres=AbundanceMissingThres,
            ),
            axis=1,
        )
    )
    logging.debug("align mz row by row.")

    # merge each filtered precursor into dictionary
    filteredIdx = np.where(AlignmentResult["IsKept"])[0]
    filteredPrecursorIdx = AlignmentResult[AlignmentResult["IsKept"]].index
    for idx, precursor_idx in zip(filteredIdx, filteredPrecursorIdx):
        right = alignment[idx].groupby([merge_key])["abundance"].sum()
        CandidateDict = pd.merge(
            CandidateDict, right, on=merge_key, how="outer"
        ).rename(columns={"abundance": precursor_idx}, inplace=False)
    logging.debug("merge dictionaries")
    CandidateDict = CandidateDict.groupby([merge_key]).sum()
    return (
        CandidateDict.fillna(0),
        AlignmentResult,
        alignment,
        IsotopeNotObs,
        y_true,
        peak_results,
    )


def compare_isotope_pattern(MQ_withIso: pd.DataFrame, seq_A_idx: int, seq_B_idx: int):
    plot_comparison(
        y_true=MQ_withIso.loc[MQ_withIso["id"] == seq_A_idx, "IsoAbundance"].values[0],
        y_pred=MQ_withIso.loc[MQ_withIso["id"] == seq_B_idx, "IsoAbundance"].values[0],
        x_true=MQ_withIso.loc[MQ_withIso["id"] == seq_A_idx, "IsoMZ"].values[0],
        x_pred=MQ_withIso.loc[MQ_withIso["id"] == seq_B_idx, "IsoMZ"].values[0],
    )
    print()
