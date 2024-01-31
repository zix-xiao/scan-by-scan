import logging
import os
from multiprocessing import cpu_count
from typing import Callable, List, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import spatial

from sklearn.decomposition import sparse_encode
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
)

from optimization.dictionary import Dict
from utils.plot import plot_comparison, plot_isopattern_and_obs
from utils.config import _algo, _alpha_criteria, _alpha_opt_metric, _loss, _pp_method
from optimization.custom_models import CustomLinearModel, mean_square_root_error

Logger = logging.getLogger(__name__)


class Quant:
    """
    Joint identification and quantification of candidates for given MS1 scan

    :preprocessing: if CandidateDict and obs_data are preprocessed, if 'sqrt' \
        then act should be squared, and cos_dist calculation should also be \
            squared
    """

    def __init__(
        self,
        CandidateDict: pd.DataFrame,
        obs_data: pd.DataFrame,
        filteredPrecursorIdx: Union[np.ndarray, list],
        preprocessing_method: _pp_method,
    ) -> None:
        self.filteredPrecursorIdx = filteredPrecursorIdx
        self.obs_data_raw = self.obs_data = obs_data[["intensity"]].values
        self.obs_mz = obs_data[["mzarray_obs"]].values[:, 0]
        Logger.debug("obs mz (index) dimension: %s", self.obs_mz.shape)
        if preprocessing_method == "raw":
            self.dictionary = CandidateDict[filteredPrecursorIdx].values
            self.obs_data = obs_data[["intensity"]].values  # shape (n_mzvalues, 1)
        elif preprocessing_method == "sqrt":
            self.dictionary = np.sqrt(CandidateDict[filteredPrecursorIdx].values)
            self.obs_data = np.sqrt(
                obs_data[["intensity"]].values
            )  # shape (n_mzvalues, 1)
        self.preprocessing = preprocessing_method
        self.alphas = []
        self.acts = []
        self.inferences = []
        self.nonzeros = []
        self.metric = []

    def optimize(
        self,
        alpha: float,
        loss: _loss = "lasso",
        algorithm: _algo = "lasso_lars",
        max_iter: int = 1000,
        metric: _alpha_opt_metric = "RMSE",
    ):
        self.alphas.append(alpha)
        match loss:
            case "lasso":
                act = sparse_encode(
                    X=self.obs_data.T,
                    dictionary=self.dictionary.T,
                    algorithm=algorithm,
                    positive=True,
                    alpha=alpha,
                    max_iter=max_iter,
                    verbose=50,
                    # n_jobs=-1
                )
                Logger.debug("dimension of act: %s", act.shape)
            case "sqrt_lasso":
                sol = CustomLinearModel(
                    residue_loss=mean_square_root_error,
                    X=self.dictionary,
                    Y=self.obs_data,
                    reg_norm="l1",
                    reg_param=alpha,
                )
                sol.fit(method=algorithm)  # maxiter=max_iter,
                Logger.debug("dimension of sol.beta: %s", sol.beta.shape)
                act = sol.beta.reshape(1, -1)
                Logger.debug("dimension of act: %s", act.shape)
        if self.preprocessing == "raw":
            self.acts.append(act[0])
            self.infer = np.matmul(act, self.dictionary.T)
        elif self.preprocessing == "sqrt":
            self.acts.append(np.square(act[0]))
            self.infer = np.square(np.matmul(act[0], self.dictionary.T))

        self.inferences.append(self.infer)
        self.nonzeros.append(np.count_nonzero(act[0] > 1))

        self.metric_used = metric
        if metric == "cos_dist":
            self.metric.append(self.CalcCosDist())
        elif metric == "RMSE":
            self.metric.append(self.CalcRMSE())

    def optimizeAlphas(
        self,
        alphas: Union[List, np.ndarray],
        loss: _loss = "lasso",
        algorithm: _algo = "lasso_lars",
        criteria: _alpha_criteria = "min",
        metric: _alpha_opt_metric = "cos_dist",
        eps: Union[None, float] = None,
        max_iter: int = 1000,
        PlotTrace: bool = False,
        save_dir: Union[str, None] = None,
    ):
        """
        iterate through different alphas and algorithm,
        for ols, set algo = 'threshold' and alpha = 0.

        :alphas: if
        :algorithms:
        """

        if criteria == "convergence" and eps is None:
            eps = 0.0001
        if loss == "lasso":
            self.optimize(alpha=0, algorithm="threshold", metric=metric)
        else:
            self.optimize(
                alpha=0,
                loss=loss,
                algorithm=algorithm,
                metric=metric,
                max_iter=max_iter,
            )
        if len(alphas) > 0:
            match criteria:
                case "min":
                    for a in alphas:
                        self.optimize(
                            alpha=a,
                            loss=loss,
                            algorithm=algorithm,
                            metric=metric,
                            max_iter=max_iter,
                        )
                    BestAlphaIdx = np.array(self.metric).argmin()
                    self.best_alpha = self.alphas[BestAlphaIdx]
                    Logger.info(
                        "Minimal distance reached at alpha = %s", self.best_alpha
                    )
                case "convergence":
                    BestAlphaIdx = None
                    for a in alphas:
                        Logger.debug("Current alpha = %s", a)
                        self.optimize(
                            alpha=a,
                            loss=loss,
                            metric=metric,
                            algorithm=algorithm,
                            max_iter=max_iter,
                        )
                        Logger.debug("Alpha list %s", self.alphas)
                        diff = self.metric[-2] - self.metric[-1]
                        Logger.debug("Alpha = %s, tol = %s", a, diff)
                        if diff <= eps and diff > 0:
                            BestAlphaIdx = self.alphas.index(a)
                            self.best_alpha = self.alphas[BestAlphaIdx]
                            Logger.info(
                                "Reached convergence criteria at alpha = %s",
                                self.best_alpha,
                            )
                            break
                        if diff < 0:
                            BestAlphaIdx = alphas.index(a) - 1
                            self.best_alpha = self.alphas[BestAlphaIdx]
                            Logger.warning(
                                "Increasing %s! Using previous alpha = %s as best"
                                " candidate!",
                                self.metric_used,
                                self.best_alpha,
                            )
                            break
                    if BestAlphaIdx is None:
                        BestAlphaIdx = np.array(self.metric).argmin()
                        self.best_alpha = self.alphas[BestAlphaIdx]
                        Logger.warning(
                            "Convergence not reached! Using alpha = %s with minimal"
                            " distance as candidate!",
                            self.best_alpha,
                        )
        else:
            BestAlphaIdx = 0
            self.best_alpha = self.alphas[BestAlphaIdx]
            Logger.info("Alpha not specified, using alpha = %s", self.best_alpha)
        self.infer = self.inferences[BestAlphaIdx]
        self.act = self.acts[BestAlphaIdx]

        if PlotTrace:  # plot trace with 2 y-axis: metric and count_nonzeors
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("log10(alpha+1)")

            color = "tab:red"
            ax1.set_ylabel(self.metric_used, color=color)
            ax1.plot(np.log10(np.array(self.alphas) + 1), self.metric, color=color)
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.plot(
                np.log10(self.best_alpha + 1), self.metric[BestAlphaIdx], "x", color="r"
            )
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = "tab:blue"
            ax2.set_ylabel(
                "Num of Non-zero Act.", color=color
            )  # we already handled the x-label with ax1
            ax2.plot(np.log10(np.array(self.alphas) + 1), self.nonzeros, color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            ax1.annotate(
                "N0 = " + str(self.dictionary.shape[1]),
                xy=(0.1, 0.9),
                xycoords="axes fraction",
            )

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(fname=os.path.join(save_dir, "alphaTrace.png"), dpi=300)
                plt.close()
            else:
                plt.show()

    def analyzeID(self, trueIDidx: List, alpha: Union[float, None] = None):
        if alpha is None:
            Logger.warning(
                "Alpha not specified, using best alpha value = %s", self.best_alpha
            )
            alpha = self.best_alpha

        result_idx = self.alphas.index(alpha)
        inferIDidx = self.filteredPrecursorIdx[self.acts[result_idx] > 1]
        Logger.info("Number of non-zero actiavation = %s", len(inferIDidx))
        y_pred = [element in set(inferIDidx) for element in self.filteredPrecursorIdx]
        y_true = [element in set(trueIDidx) for element in self.filteredPrecursorIdx]
        self.confusion_matrix = confusion_matrix(
            y_true=y_true, y_pred=y_pred, normalize=None, labels=[True, False]
        )
        self.IDresult = pd.DataFrame(
            {
                "Candidate": self.filteredPrecursorIdx,
                "Activation": self.acts[result_idx] > 1,
                "y_pred": y_pred,
                "y_true": y_true,
            }
        )
        self.cls_report = classification_report(
            y_true=y_true, y_pred=y_pred, output_dict=True
        )

    def PlotObsAndInfer(self, log_intensity: bool = False):
        plot_comparison(
            y_true=self.obs_data_raw.flatten(),
            y_pred=self.infer.flatten(),
            log_y=log_intensity,
        )

    def calc_precursor_reconstruct_cos_dist(self):
        self.reconstruc_cos_dist = []
        for idx, val in enumerate(self.act):
            if val > 0:
                mask_act = self.act.copy()
                mask_act[idx] = 0
                y_minus_i = np.matmul(self.dictionary, mask_act)
                y_i = self.obs_data_raw.reshape(-1) - y_minus_i
                iso_mz_mask = (
                    self.dictionary[:, idx] != 0
                )  # only consider nonzero input in dict
                self.reconstruc_cos_dist.append(
                    spatial.distance.cosine(
                        self.dictionary[:, idx][iso_mz_mask], y_i.flatten()[iso_mz_mask]
                    )
                )
            else:
                self.reconstruc_cos_dist.append(0)
        return self.reconstruc_cos_dist

    def CalcRMSE(self):
        return mean_squared_error(
            y_true=self.obs_data_raw.flatten(),
            y_pred=self.infer.flatten(),
            squared=False,
        )

    def CalcCosDist(self):
        return spatial.distance.cosine(
            self.obs_data_raw.flatten(), self.infer.flatten()
        )

    # def CalcExplainedValues(self): TODO: specify which alpha?
    #     return len(self.infer_nonzero)/self.obs_data.shape[0]
    def plot_iso_pattern_and_infer(
        self,
        Maxquant_result: pd.DataFrame,
        precursor_id: List[int] | None = None,
        precursor_idx: List[int] | None = None,
        log_intensity: bool = False,
    ):
        plot_isopattern_and_obs(
            Maxquant_result=Maxquant_result,
            infer_intensity=pd.Series(data=self.infer[0], index=self.obs_mz),
            lower_plot="infer",
            precursor_id=precursor_id,
            precursor_idx=precursor_idx,
            log_intensity=log_intensity,
        )

    def CalcExplainedInt(
        self,
    ):  # TODO: does not consider the correctness of explained peaks
        return self.infer.sum() / self.obs_data_raw.sum()


def process_one_scan(
    scan_idx: int,
    OneScan: pd.core.series.Series,
    Maxquant_result: pd.DataFrame,
    AbundanceMissingThres: float = 0.4,
    metric: _alpha_opt_metric = "cos_dist",
    alpha_criteria: _alpha_criteria = "convergence",
    alphas: Union[List, np.ndarray] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    loss: _loss = "lasso",
    opt_algo: _algo = "lasso_cd",
    preprocessing_method: _pp_method = "raw",
    corr_thres: float = 0.9,
    max_iter: int = 1000,
    return_interim_results: bool = False,
    return_precursor_scan_cos_dist: bool = False,
    plot_alpha_trace: bool = False,
    plot_obs_and_infer: bool = False,
):
    """
    process one scan using lasso regression, return  alignment, \
        activation and scan summary

    :scan_idx:
    :OneScan: a row in dataframe MS1Scans
    :
    """
    Logger.debug("Start.")

    CandidatePrecursorsByRT = Maxquant_result.loc[
        (Maxquant_result["RT_search_left"] <= OneScan["starttime"])
        & (Maxquant_result["RT_search_right"] >= OneScan["starttime"])
    ]
    Logger.debug("Filter by RT.")

    if CandidatePrecursorsByRT.shape[0] > 0:
        ScanDict = Dict(
            CandidateByRT=CandidatePrecursorsByRT,
            OneScan=OneScan,
            AbundanceMissingThres=AbundanceMissingThres,
        )
        CandidateDict = ScanDict.dict
        filteredPrecursorIdx = ScanDict.filteredCandidateIdx
        if CandidateDict is not None:
            y_true = ScanDict.obs_int

            ScanDict.get_feature_corr(
                corr_thres=corr_thres,
                calc_jaccard_sim=False,
                plot_hist=False,
                plot_hmap=False,
            )
            num_corr_dict_candidate = ScanDict.high_corr_sol.shape[0]
            Logger.debug("Construct dictionary")

            PrecursorQuant = Quant(
                CandidateDict=CandidateDict,
                obs_data=y_true,
                filteredPrecursorIdx=filteredPrecursorIdx,
                preprocessing_method=preprocessing_method,
            )

            PrecursorQuant.optimizeAlphas(
                alphas=alphas,
                loss=loss,
                metric=metric,
                algorithm=opt_algo,
                criteria=alpha_criteria,
                PlotTrace=plot_alpha_trace,
                max_iter=max_iter,
            )
            Logger.debug("Quant - optimize alphas")
            if plot_obs_and_infer:
                PrecursorQuant.PlotObsAndInfer(log_intensity=False)
                Logger.debug("Quant - Plot")

            activation = {
                "precursor": filteredPrecursorIdx,
                "activation": PrecursorQuant.act,
            }
            if return_precursor_scan_cos_dist:
                cos_dist = PrecursorQuant.calc_precursor_reconstruct_cos_dist()
                precursor_cos_dist = {
                    "precursor": filteredPrecursorIdx,
                    "cos_dist": cos_dist,
                }
            cos_dist = PrecursorQuant.CalcCosDist()
            rmse = PrecursorQuant.CalcRMSE()
            scan_sum = (
                scan_idx,
                OneScan["starttime"],
                CandidatePrecursorsByRT.index,
                filteredPrecursorIdx,
                num_corr_dict_candidate,
                PrecursorQuant.best_alpha,
                cos_dist,
                PrecursorQuant.CalcExplainedInt(),
            )
            Logger.info(
                "scan index %s: activation successfully calculated with cosine distance"
                " %s, RMSE %s.",
                scan_idx,
                cos_dist,
                rmse,
            )
        else:
            activation = None
            precursor_cos_dist = None
            scan_sum = (
                scan_idx,
                OneScan["starttime"],
                CandidatePrecursorsByRT.index,
                filteredPrecursorIdx,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            Logger.info(
                "scan index %s: less than 2 valid candidates after isotope pattern"
                " matching.",
                scan_idx,
            )

    else:
        CandidateDict = None
        activation = None
        precursor_cos_dist = None
        scan_sum = (
            scan_idx,
            OneScan["starttime"],
            [],
            [],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
        Logger.info("scan index %s: no valid candidate by RT.", scan_idx)

    result_dict_onescan = {}
    result_dict_onescan[scan_idx] = {
        "activation": activation,
        "CandidateDict": CandidateDict,
        "scans_record": scan_sum,
    }
    match (return_interim_results, return_precursor_scan_cos_dist):
        case (True, True):
            result_dict_onescan[scan_idx]["precursor_cos_dist"] = precursor_cos_dist
            return result_dict_onescan, ScanDict, PrecursorQuant
        case (True, False):
            result_dict_onescan[scan_idx]["precursor_cos_dist"] = None
            return result_dict_onescan, ScanDict, PrecursorQuant
        case (False, True):
            result_dict_onescan[scan_idx]["precursor_cos_dist"] = precursor_cos_dist
            return result_dict_onescan
        case (False, False):
            result_dict_onescan[scan_idx]["precursor_cos_dist"] = None
            return result_dict_onescan


def parallel(func=None, args=(), merge_func=lambda x: x, parallelism=cpu_count()):
    def decorator(func: Callable):
        def inner(*args, **kwargs):
            results = Parallel(n_jobs=parallelism)(
                delayed(func)(*args, **kwargs) for i in range()
            )
            return merge_func(results)

        return inner

    if func is None:
        # decorator was used like @parallel(...)
        return decorator
    else:
        # decorator was used like @parallel, without parens
        return decorator(func)


def start_logger_if_necessary():
    Logger = logging.getLogger(__name__)
    if len(Logger.handlers) == 0:
        Logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        fh = logging.FileHandler("out.log", mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        Logger.addHandler(sh)
        Logger.addHandler(fh)
    return Logger


def process_scans_parallel(
    n_jobs: int,
    MS1Scans: pd.DataFrame,
    Maxquant_result: pd.DataFrame,
    AbundanceMissingThres: float = 0.4,
    alpha_criteria: _alpha_criteria = "convergence",
    alphas: Union[List, np.ndarray] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    loss: _loss = "lasso",
    opt_algo: _algo = "lasso_cd",
    metric: _alpha_opt_metric = "cos_dist",
    preprocessing_method: _pp_method = "raw",
    corr_thres: float = 0.95,
    max_iter: int = 1000,
    return_precursor_scan_cos_dist: bool = False,
):
    start_logger_if_necessary()
    result_list = Parallel(n_jobs=n_jobs)(
        delayed(process_one_scan)(
            scan_idx=scan_idx,
            OneScan=OneScan,
            Maxquant_result=Maxquant_result,
            AbundanceMissingThres=AbundanceMissingThres,
            alpha_criteria=alpha_criteria,
            alphas=alphas,
            metric=metric,
            loss=loss,
            opt_algo=opt_algo,
            preprocessing_method=preprocessing_method,
            corr_thres=corr_thres,
            max_iter=max_iter,
            return_interim_results=False,
            return_precursor_scan_cos_dist=return_precursor_scan_cos_dist,
        )
        for scan_idx, OneScan in MS1Scans.iterrows()
    )
    result_dict = dict(pair for d in result_list for pair in d.items())
    return result_dict
