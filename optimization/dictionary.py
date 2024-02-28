import logging
from typing import Union
import numpy as np
import pandas as pd

import IsoSpecPy as iso
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import networkx as nx

from utils.plot import plot_comparison
from utils.tools import ExtractPeak


Logger = logging.getLogger(__name__)


def calculate_modpept_isopattern(
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
    formula = "".join([f"{key}{value}" for key, value in atom_composition.items()])
    iso_distr = iso.IsoThreshold(formula=formula, threshold=ab_thres, absolute=True)
    iso_distr.sort_by_mass()
    mz_sort_by_mass = iso_distr.np_masses() / charge
    probs_sort_by_mass = iso_distr.np_probs()

    return mz_sort_by_mass, probs_sort_by_mass


def _connect_collinear_candidate_pairs(collinear_pairs: pd.Series):
    """
    Given a series of collinear pairs, connect them into a graph and return the graph object

    :collinear_pairs: pd.Series, index being collinear pairs and values being correlation coefficient
    """
    G = nx.from_edgelist(collinear_pairs.index)
    # Find connected components
    connected_components_df = nx.to_pandas_adjacency(G)
    # Create an empty pandas series to store the connected precursors
    connected_precursors = pd.Series(index=connected_components_df.index, dtype=object)

    # Iterate over each precursor
    for precursor in connected_components_df.index:
        # Get the connected precursors for the current precursor
        connected = connected_components_df.columns[
            connected_components_df.loc[precursor] == 1
        ].tolist()
        # Store the connected precursors in the series
        connected_precursors[precursor] = connected
    connected_precursors.sort_index(inplace=True)
    return connected_precursors


class Dict:
    def __init__(
        self,
        candidate_by_rt: pd.DataFrame,
        one_scan: Union[pd.DataFrame, pd.Series],
        abundance_missing_thres: float = 0.4,
        rel_height: float = 0.75,
    ) -> None:
        self.n_candidate_by_rt = candidate_by_rt.shape[0]
        self.candidate_by_rt = candidate_by_rt
        Logger.info("Number of candidates by RT %s", self.n_candidate_by_rt)

        self.abundance_missing_thres = abundance_missing_thres
        ms1_int = pd.DataFrame(
            {"mzarray_obs": one_scan["mzarray"], "intensity": one_scan["intarray"]}
        )
        self.peak_results = ExtractPeak(
            np.array(ms1_int["mzarray_obs"]),
            np.array(ms1_int["intensity"]),
            rel_height=rel_height,
        )

        self.dict = None
        self.filtered_candidate_idx = None
        self.iso_abundance_by_id = None
        self.obs_peak_int = None
        self.match_candidate_with_peak()

        self.corr_matrix = None
        self.high_corr_sol = None
        self.collinear_precursors = None
        self.collinear_sets = None

    def match_candidate_with_peak(self):
        candidate_by_rt_iso = (
            self.candidate_by_rt[["id", "IsoMZ", "IsoAbundance"]]
            .explode(["IsoMZ", "IsoAbundance"])
            .astype({"IsoMZ": "float64", "IsoAbundance": "float64"})
        )

        candidate_by_rt_iso = pd.merge_asof(
            candidate_by_rt_iso.sort_values("IsoMZ"),
            self.peak_results,
            left_on="IsoMZ",
            right_on="apex_mz",
            allow_exact_matches=True,
            direction="nearest",
        )
        candidate_by_rt_iso["matched"] = (
            candidate_by_rt_iso["IsoMZ"] >= candidate_by_rt_iso["start_mz"]
        ) & (candidate_by_rt_iso["IsoMZ"] <= candidate_by_rt_iso["end_mz"])
        self.iso_abundance_by_id = (
            candidate_by_rt_iso.groupby(["id", "matched"])["IsoAbundance"]
            .sum()
            .reset_index()
        )
        candidate_by_rt_iso_matched = candidate_by_rt_iso[
            (candidate_by_rt_iso["matched"])
        ]
        Logger.info(
            "Number of candidates after isotope match %s",
            len(candidate_by_rt_iso_matched["id"].unique()),
        )

        # filter candidates by isotope abundance
        if len(candidate_by_rt_iso_matched["id"].unique()) > 0:
            candidate_by_rt_iso_filtered = self.iso_abundance_by_id[
                self.iso_abundance_by_id["matched"]
                & (
                    self.iso_abundance_by_id["IsoAbundance"]
                    >= (1 - self.abundance_missing_thres)
                )
            ]
            self.filtered_candidate_idx = (
                candidate_by_rt_iso_filtered["id"].unique().tolist()
            )
            Logger.info(
                "Number of candidates after isotope abundance filter %s",
                len(self.filtered_candidate_idx),
            )
            if len(self.filtered_candidate_idx) > 0:
                candidate_by_rt_iso_filtered_sum_abundance = (
                    candidate_by_rt_iso_matched[
                        candidate_by_rt_iso_matched["id"].isin(
                            self.filtered_candidate_idx
                        )
                    ]
                    .groupby(["id", "apex_mz"])["IsoAbundance"]
                    .sum()
                    .reset_index()
                )
                candid_dict = candidate_by_rt_iso_filtered_sum_abundance.pivot(
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

                self.obs_peak_int = pd.DataFrame(
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

    def get_feature_corr(
        self,
        corr_thres: float = 0.9,
        calc_jaccard_sim: bool = True,
        plot_hmap: bool = False,
        plot_collinear_hist: bool = False,
    ):
        self.corr_matrix = self.dict.corr()
        if plot_hmap:
            self.corr_matrix.style.background_gradient(cmap="coolwarm")
            plt.matshow(self.corr_matrix)
            plt.show()
        sol = self.corr_matrix.where(
            np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool)
        ).stack()
        self.high_corr_sol = sol[sol >= corr_thres]  # type: ignore
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

        # self.collinear_sets_list = _connect_collinear_candidate_pairs(
        #     self.high_corr_sol
        # )
        # Create a graph from the adjacency matrix
        graph = nx.from_edgelist(self.high_corr_sol.index)

        # Get the connected nodes
        self.collinear_sets = list(nx.connected_components(graph))
        self.collinear_precursors = _connect_collinear_candidate_pairs(
            self.high_corr_sol
        )

        if plot_collinear_hist:
            self.high_corr_sol.plot.hist()
            ymin, ymax = plt.ylim()
            plt.vlines(x=0.95, ymin=ymin, ymax=ymax, linestyles="dashed")
            plt.vlines(x=0.9, ymin=ymin, ymax=ymax, linestyles="dashed")
            plt.vlines(
                x=corr_thres, ymin=ymin, ymax=ymax, linestyles="dashed", color="r"
            )
            plt.title("Distribution of Highly Correlation Coefficients")

        Logger.info(
            "Number of collinear sets: %s with correlation threshold %s",
            len(self.collinear_sets),
            corr_thres,
        )
        Logger.info(
            "Number of candidates involved in high correlation: %s",
            sum(len(s) for s in self.collinear_sets),
        )

    def plot_observe_iso_abundance(self):
        self.iso_abundance_by_id[self.iso_abundance_by_id["matched"]][
            "IsoAbundance"
        ].hist(bins=100)
        plt.title("Distribution of Observed Isotope Abundance")
        ymin, ymax = plt.ylim()
        plt.vlines(
            x=1 - self.abundance_missing_thres,
            ymin=ymin,
            ymax=ymax,
            label="Abundance Thres",
            color="r",
        )


def compare_isotope_pattern(MQ_withIso: pd.DataFrame, seq_A_idx: int, seq_B_idx: int):
    plot_comparison(
        y_true=MQ_withIso.loc[MQ_withIso["id"] == seq_A_idx, "IsoAbundance"].values[0],
        y_pred=MQ_withIso.loc[MQ_withIso["id"] == seq_B_idx, "IsoAbundance"].values[0],
        x_true=MQ_withIso.loc[MQ_withIso["id"] == seq_A_idx, "IsoMZ"].values[0],
        x_pred=MQ_withIso.loc[MQ_withIso["id"] == seq_B_idx, "IsoMZ"].values[0],
    )
    print()
